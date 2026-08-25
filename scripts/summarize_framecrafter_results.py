#!/usr/bin/env python3
"""Summarise immutable FrameCrafter preprocessing snapshots and render evidence.

The script joins a ``preprocess_report_*.json`` with its manifest, rather than
guessing acceptance from an image directory.  Accepted candidates are ranked by
reported sharpness gain.  If a scene has no accepted candidate, exactly one
best rejected candidate is visualised and is prominently labelled REJECTED.

Examples
--------
python scripts/summarize_framecrafter_results.py \
  --scene fr1_desk=/absolute/path/to/fr1_snapshot \
  --scene fr2_xyz=/absolute/path/to/preprocess_report_SIGNATURE_ID.json \
  --output-dir /absolute/path/to/framecrafter_summary --top-k 3

Directories are expected to contain one immutable report/manifest pair.  When
several snapshots share a directory, pass the exact report or manifest path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPORT_SCHEMA = "unblur_slam.framecrafter_preprocess_report.v1"
MANIFEST_SCHEMA = "unblur_slam.framecrafter_manifest.v1"
SUMMARY_SCHEMA = "unblur_slam.framecrafter_result_summary.v1"
_METRIC_NAMES = (
    "sharpness_gain",
    "depth_consistency",
    "photometric_error",
    "reprojection_error_px",
)


@dataclass(frozen=True)
class Snapshot:
    scene: str
    input_root: Path
    report_path: Path
    manifest_path: Path
    report: Mapping[str, Any]
    manifest: Mapping[str, Any]


class ArtifactResolver:
    """Resolve artifacts locally first so copied immutable snapshots still work."""

    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot
        self._cache: dict[str, Path] = {}
        self._local_files: list[Path] | None = None

    def resolve(self, value: Any, label: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} is missing a non-empty path")
        if value in self._cache:
            return self._cache[value]

        declared = Path(value).expanduser()
        direct_candidates: list[Path] = []
        if not declared.is_absolute():
            direct_candidates.extend(
                (
                    self.snapshot.report_path.parent / declared,
                    self.snapshot.manifest_path.parent / declared,
                    self.snapshot.input_root / declared,
                )
            )

        # Prefer a snapshot-local copy, even when the recorded absolute path is
        # still reachable.  That preserves the semantics of an immutable copy.
        local = self._best_local_match(declared)
        if local is not None:
            result = local.resolve()
        else:
            result = next(
                (path.resolve() for path in direct_candidates if path.is_file()),
                declared.resolve() if declared.is_file() else None,
            )
            if result is None:
                raise FileNotFoundError(f"{label} does not exist: {declared}")
        self._cache[value] = result
        return result

    def _best_local_match(self, declared: Path) -> Path | None:
        root = self.snapshot.input_root
        if not root.is_dir():
            return None
        if self._local_files is None:
            self._local_files = [path for path in root.rglob("*") if path.is_file()]
        candidates = [path for path in self._local_files if path.name == declared.name]
        if not candidates:
            return None
        declared_parts = declared.parts

        def suffix_score(candidate: Path) -> int:
            score = 0
            for left, right in zip(reversed(candidate.parts), reversed(declared_parts)):
                if left != right:
                    break
                score += 1
            return score

        scores = [(suffix_score(path), path) for path in candidates]
        best_score = max(score for score, _ in scores)
        best = sorted(path for score, path in scores if score == best_score)
        if len(best) > 1:
            raise ValueError(
                f"ambiguous snapshot-local artifact {declared.name!r}: "
                + ", ".join(str(path) for path in best)
            )
        return best[0]


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _identity(payload: Mapping[str, Any]) -> tuple[str, str]:
    return str(payload.get("preprocess_signature", "")), str(
        payload.get("generation_id", "")
    )


def _matching_json_files(root: Path, schema: str) -> list[Path]:
    prefix = "preprocess_report" if schema == REPORT_SCHEMA else "manifest"
    matches: list[Path] = []
    for path in sorted(root.rglob(f"{prefix}*.json")):
        try:
            if _read_json(path).get("schema") == schema:
                matches.append(path.resolve())
        except ValueError:
            continue
    return matches


def _local_pair_file(
    owner_path: Path,
    owner_payload: Mapping[str, Any],
    *,
    schema: str,
    input_root: Path,
) -> Path | None:
    signature, generation_id = _identity(owner_payload)
    prefix = "manifest" if schema == MANIFEST_SCHEMA else "preprocess_report"
    exact_name = (
        f"{prefix}_{signature}_{generation_id}.json"
        if signature and generation_id
        else None
    )
    search_roots = [owner_path.parent]
    if input_root != owner_path.parent:
        search_roots.append(input_root)
    candidates: list[Path] = []
    for root in search_roots:
        paths = root.rglob(exact_name) if exact_name else root.rglob(f"{prefix}*.json")
        for path in paths:
            path = path.resolve()
            if path in candidates:
                continue
            try:
                payload = _read_json(path)
            except ValueError:
                continue
            if payload.get("schema") == schema and (
                not signature or _identity(payload) == (signature, generation_id)
            ):
                candidates.append(path)
    if len(candidates) > 1:
        same_dir = [path for path in candidates if path.parent == owner_path.parent]
        if len(same_dir) == 1:
            return same_dir[0]
        raise ValueError(
            f"multiple matching immutable {prefix} files for {owner_path}: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0] if candidates else None


def _declared_pair_file(
    owner_path: Path, owner_payload: Mapping[str, Any], key: str, schema: str
) -> Path | None:
    value = owner_payload.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner_path.parent / path
    if not path.is_file():
        return None
    payload = _read_json(path)
    if payload.get("schema") != schema:
        raise ValueError(f"declared pair has the wrong schema: {path}")
    return path.resolve()


def load_snapshot(scene: str | None, input_path: Path | str) -> Snapshot:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"snapshot input does not exist: {path}")
    input_root = path if path.is_dir() else path.parent

    if path.is_dir():
        reports = _matching_json_files(path, REPORT_SCHEMA)
        if len(reports) != 1:
            detail = "none" if not reports else ", ".join(str(item) for item in reports)
            raise ValueError(
                f"snapshot directory must contain exactly one report; found {detail}. "
                "Pass an exact report path when a directory contains multiple snapshots."
            )
        report_path = reports[0]
        report = _read_json(report_path)
        manifest_path = _local_pair_file(
            report_path,
            report,
            schema=MANIFEST_SCHEMA,
            input_root=input_root,
        ) or _declared_pair_file(report_path, report, "manifest", MANIFEST_SCHEMA)
    else:
        payload = _read_json(path)
        if payload.get("schema") == REPORT_SCHEMA:
            report_path, report = path, payload
            manifest_path = _local_pair_file(
                report_path,
                report,
                schema=MANIFEST_SCHEMA,
                input_root=input_root,
            ) or _declared_pair_file(report_path, report, "manifest", MANIFEST_SCHEMA)
        elif payload.get("schema") == MANIFEST_SCHEMA:
            manifest_path, manifest = path, payload
            report_path = _local_pair_file(
                manifest_path,
                manifest,
                schema=REPORT_SCHEMA,
                input_root=input_root,
            ) or _declared_pair_file(
                manifest_path, manifest, "preprocess_report_path", REPORT_SCHEMA
            )
            if report_path is None:
                raise FileNotFoundError(f"no matching preprocess report for {path}")
            report = _read_json(report_path)
        else:
            raise ValueError(f"unsupported FrameCrafter JSON schema in {path}")
    if manifest_path is None:
        raise FileNotFoundError(f"no matching manifest for {report_path}")
    manifest = _read_json(manifest_path)
    if report.get("schema") != REPORT_SCHEMA or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("FrameCrafter report/manifest schema mismatch")
    report_identity = _identity(report)
    manifest_identity = _identity(manifest)
    if all(report_identity) and report_identity != manifest_identity:
        raise ValueError(
            f"report/manifest snapshot identity mismatch: {report_identity} vs "
            f"{manifest_identity}"
        )

    inferred_scene = str(report.get("scene", "")).strip()
    if not inferred_scene:
        inferred_scene = report_path.parent.name
        if inferred_scene.lower() in {"generated", "snapshot", "snapshots"}:
            inferred_scene = report_path.parent.parent.name
    scene_name = (scene or inferred_scene).strip()
    if not scene_name:
        raise ValueError(f"cannot infer scene name for {report_path}")
    return Snapshot(
        scene=scene_name,
        input_root=input_root,
        report_path=report_path.resolve(),
        manifest_path=manifest_path.resolve(),
        report=report,
        manifest=manifest,
    )


def _integer(value: Any, label: str, *, fallback: int | None = None) -> int:
    if value is None and fallback is not None:
        return fallback
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a non-negative integer") from error
    if result < 0 or isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _metric(record: Mapping[str, Any], name: str) -> float | None:
    metrics = record.get("metrics", record.get("gate_metrics", {}))
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(name)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rank_key(record: Mapping[str, Any]) -> tuple[float, str]:
    gain = _metric(record, "sharpness_gain")
    return (float("-inf") if gain is None else gain, str(record.get("target_id", "")))


def _reason_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        reasons = record.get("reasons", [])
        if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes)):
            continue
        for reason in reasons:
            key = str(reason)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def summarise_counts(snapshot: Snapshot) -> dict[str, Any]:
    report, manifest = snapshot.report, snapshot.manifest
    planned_records = report.get("planned", [])
    accepted_records = report.get("accepted", [])
    rejected_records = report.get("rejected", [])
    for label, records in (
        ("planned", planned_records),
        ("accepted", accepted_records),
        ("rejected", rejected_records),
    ):
        if not isinstance(records, list) or not all(
            isinstance(record, Mapping) for record in records
        ):
            raise ValueError(f"{snapshot.scene}: report {label} must be a list of objects")

    source = _integer(
        report.get("source_frame_count"), f"{snapshot.scene}.source_frame_count"
    )
    if _integer(
        manifest.get("source_frame_count"),
        f"{snapshot.scene}.manifest.source_frame_count",
    ) != source:
        raise ValueError(f"{snapshot.scene}: report/manifest source counts disagree")
    selected = _integer(
        report.get("selected_target_count"),
        f"{snapshot.scene}.selected_target_count",
        fallback=_integer(
            report.get("planned_target_count"),
            f"{snapshot.scene}.planned_target_count",
            fallback=len(planned_records),
        ),
    )
    planned = _integer(
        report.get("planned_total_before_cap"),
        f"{snapshot.scene}.planned_total_before_cap",
        fallback=selected,
    )
    accepted = _integer(
        report.get("accepted_target_count"),
        f"{snapshot.scene}.accepted_target_count",
        fallback=len(accepted_records),
    )
    rejected = _integer(
        report.get("rejected_target_count"),
        f"{snapshot.scene}.rejected_target_count",
        fallback=len(rejected_records),
    )
    if selected != len(planned_records):
        raise ValueError(f"{snapshot.scene}: selected count disagrees with planned records")
    if accepted != len(accepted_records) or rejected != len(rejected_records):
        raise ValueError(f"{snapshot.scene}: gate counts disagree with outcome records")
    if accepted + rejected > selected or planned < selected:
        raise ValueError(f"{snapshot.scene}: impossible planned/selected/outcome counts")
    generated = _integer(
        manifest.get("generated_frame_count"),
        f"{snapshot.scene}.manifest.generated_frame_count",
    )
    if generated != accepted:
        raise ValueError(f"{snapshot.scene}: manifest generated count is not accepted count")

    excluded_before_generation = planned - selected
    selected_without_gate_outcome = selected - accepted - rejected
    if excluded_before_generation < 0 or selected_without_gate_outcome < 0:
        raise ValueError(f"{snapshot.scene}: negative FrameCrafter pipeline count")

    planned_by_id = {str(record.get("target_id", "")): record for record in planned_records}
    accepted_with_reasons = [
        {**planned_by_id.get(str(record.get("target_id", "")), {}), **record}
        for record in accepted_records
    ]
    rejected_with_reasons = [
        {**planned_by_id.get(str(record.get("target_id", "")), {}), **record}
        for record in rejected_records
    ]
    return {
        "source": source,
        "planned": planned,
        "selected": selected,
        "accepted": accepted,
        "rejected": rejected,
        # ``planned`` is the complete pre-cap proposal count, while records in
        # report["planned"] are only the scene-wide selected subset.  Keep the
        # pre-generation cap exclusion distinct from an interrupted/plan-only
        # run whose selected targets have no gate outcome.
        "not_evaluated": excluded_before_generation,
        "selected_without_gate_outcome": selected_without_gate_outcome,
        "accepted_over_source": _ratio(accepted, source),
        "accepted_over_source_plus_accepted": _ratio(accepted, source + accepted),
        "accepted_over_planned": _ratio(accepted, planned),
        "accepted_over_selected": _ratio(accepted, selected),
        "selected_reason_counts": _reason_counts(planned_records),
        "accepted_reason_counts": _reason_counts(accepted_with_reasons),
        "rejected_reason_counts": _reason_counts(rejected_with_reasons),
    }


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _format_metric(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _metadata_lines(
    target_id: str,
    reasons: Sequence[str],
    record: Mapping[str, Any],
    status: str,
    *,
    crop: tuple[int, int, int, int] | None = None,
) -> list[str]:
    metrics = {name: _metric(record, name) for name in _METRIC_NAMES}
    lines = [f"{status}  target={target_id}", f"reasons={','.join(reasons) or 'none'}"]
    lines.append(
        "sharpness_gain={}  depth_consistency={}  photometric_error={}  "
        "reprojection_error_px={}".format(
            _format_metric(metrics["sharpness_gain"]),
            _format_metric(metrics["depth_consistency"]),
            _format_metric(metrics["photometric_error"]),
            _format_metric(metrics["reprojection_error_px"]),
        )
    )
    if crop is not None:
        x, y, width, height = crop
        lines.append(
            f"100% ZOOM  shared native crop x={x}, y={y}, w={width}, h={height}; "
            "1 output pixel = 1 source pixel"
        )
    return lines


def _wrap_lines(lines: Sequence[str], width_pixels: int, font: ImageFont.ImageFont) -> list[str]:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    wrapped: list[str] = []
    for line in lines:
        if draw.textbbox((0, 0), line, font=font)[2] <= width_pixels:
            wrapped.append(line)
            continue
        text_width = max(1, draw.textbbox((0, 0), line, font=font)[2])
        approximate_chars = max(16, int(len(line) * width_pixels / text_width))
        wrapped.extend(
            textwrap.wrap(
                line,
                width=approximate_chars,
                break_long_words=True,
                break_on_hyphens=False,
            )
            or [""]
        )
    return wrapped


def _auto_detail_crop(
    images: Sequence[Image.Image], requested_size: int
) -> tuple[int, int, int, int]:
    width, height = images[0].size
    size = min(requested_size, width, height)
    if size <= 0:
        raise ValueError("zoom size must be positive")
    if size == width and size == height:
        return 0, 0, size, size

    grayscale = np.mean(
        [np.asarray(image.convert("L"), dtype=np.float32) / 255.0 for image in images],
        axis=0,
    )
    dx = np.zeros_like(grayscale)
    dy = np.zeros_like(grayscale)
    dx[:, 1:] = np.abs(np.diff(grayscale, axis=1))
    dy[1:, :] = np.abs(np.diff(grayscale, axis=0))
    detail = dx + dy
    integral = np.pad(detail, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    margin_x = min(max(0, width // 20), max(0, (width - size) // 2))
    margin_y = min(max(0, height // 20), max(0, (height - size) // 2))
    x_values = list(
        range(
            margin_x,
            max(margin_x + 1, width - size - margin_x + 1),
            max(1, size // 24),
        )
    )
    y_values = list(
        range(
            margin_y,
            max(margin_y + 1, height - size - margin_y + 1),
            max(1, size // 24),
        )
    )
    if width - size - margin_x not in x_values:
        x_values.append(max(0, width - size - margin_x))
    if height - size - margin_y not in y_values:
        y_values.append(max(0, height - size - margin_y))

    best_score, best_x, best_y = float("-inf"), 0, 0
    for y in y_values:
        y = min(max(0, y), height - size)
        for x in x_values:
            x = min(max(0, x), width - size)
            score = float(
                integral[y + size, x + size]
                - integral[y, x + size]
                - integral[y + size, x]
                + integral[y, x]
            )
            # Stable tie-breaking picks the crop nearest the image centre.
            centre_penalty = 1e-9 * (
                (x + size / 2 - width / 2) ** 2
                + (y + size / 2 - height / 2) ** 2
            )
            score -= centre_penalty
            if score > best_score:
                best_score, best_x, best_y = score, x, y
    return int(best_x), int(best_y), int(size), int(size)


def _compose_triptych(
    images: Sequence[Image.Image],
    lines: Sequence[str],
    *,
    status: str,
    crop: tuple[int, int, int, int] | None,
) -> Image.Image:
    if len(images) != 3 or len({image.size for image in images}) != 1:
        raise ValueError("left/generated/right must have the same resolution")
    panels = list(images)
    if crop is not None:
        x, y, width, height = crop
        panels = [image.crop((x, y, x + width, y + height)) for image in images]
    panel_width, panel_height = panels[0].size
    canvas_width = panel_width * 3
    body_font, label_font = _font(18), _font(18, bold=True)
    wrapped = _wrap_lines(lines, max(1, canvas_width - 24), body_font)
    line_height = max(22, body_font.getbbox("Ag")[3] - body_font.getbbox("Ag")[1] + 6)
    header_height = 16 + line_height * len(wrapped)
    label_height = 32
    canvas = Image.new(
        "RGB", (canvas_width, header_height + label_height + panel_height), (15, 15, 15)
    )
    draw = ImageDraw.Draw(canvas)
    accent = (29, 150, 73) if status == "ACCEPTED" else (210, 48, 48)
    draw.rectangle((0, 0, 9, header_height - 1), fill=accent)
    y = 8
    for index, line in enumerate(wrapped):
        draw.text(
            (18, y),
            line,
            font=label_font if index == 0 else body_font,
            fill=(255, 255, 255),
        )
        y += line_height
    labels = ("LEFT SOURCE", "FRAMECRAFTER", "RIGHT SOURCE")
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * panel_width
        draw.rectangle(
            (x, header_height, x + panel_width, header_height + label_height),
            fill=(30, 30, 30),
        )
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text(
            (x + (panel_width - (bbox[2] - bbox[0])) / 2, header_height + 5),
            label,
            font=label_font,
            fill=(245, 245, 245),
        )
        canvas.paste(panel, (x, header_height + label_height))
        if index:
            draw.line((x, header_height, x, canvas.height), fill=(255, 255, 255), width=2)
    return canvas


def _safe_stem(target_id: str, rank: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", target_id).strip("._")[:72] or "target"
    digest = hashlib.sha256(target_id.encode("utf-8")).hexdigest()[:8]
    return f"rank_{rank:03d}_{clean}_{digest}"


def _open_rgb(path: Path, label: str) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load {label} image {path}: {error}") from error


def render_scene(
    snapshot: Snapshot,
    output_dir: Path,
    *,
    top_k: int,
    zoom_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report, manifest = snapshot.report, snapshot.manifest
    planned = {
        str(record.get("target_id", "")): record
        for record in report.get("planned", [])
        if isinstance(record, Mapping)
    }
    accepted = sorted(
        (record for record in report.get("accepted", []) if isinstance(record, Mapping)),
        key=_rank_key,
        reverse=True,
    )
    rejected = sorted(
        (record for record in report.get("rejected", []) if isinstance(record, Mapping)),
        key=_rank_key,
        reverse=True,
    )
    ranked_accepted = [
        {
            "rank": rank,
            "target_id": str(record.get("target_id", "")),
            "sharpness_gain": _metric(record, "sharpness_gain"),
        }
        for rank, record in enumerate(accepted, start=1)
    ]
    if accepted:
        selected = accepted if top_k == 0 else accepted[:top_k]
        visual_records = [("ACCEPTED", record) for record in selected]
    elif rejected:
        visual_records = [("REJECTED", rejected[0])]
    else:
        visual_records = []

    frames = manifest.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError(f"{snapshot.scene}: manifest frames must be a list")
    originals = {
        _integer(frame.get("source_index"), "original.source_index"): frame
        for frame in frames
        if isinstance(frame, Mapping) and frame.get("kind") == "original"
    }
    synthetics = {
        str(frame.get("target_id", "")): frame
        for frame in frames
        if isinstance(frame, Mapping) and frame.get("kind") == "synthetic"
    }
    resolver = ArtifactResolver(snapshot)
    scene_dir = (output_dir / re.sub(r"[^A-Za-z0-9_.-]+", "_", snapshot.scene)).resolve()
    scene_dir.mkdir(parents=True, exist_ok=True)
    visuals: list[dict[str, Any]] = []
    for rank, (status, outcome) in enumerate(visual_records, start=1):
        target_id = str(outcome.get("target_id", ""))
        plan = planned.get(target_id)
        if plan is None:
            raise ValueError(f"{snapshot.scene}: no planned record for target {target_id}")
        left_index = _integer(plan.get("left_index"), f"{target_id}.left_index")
        right_index = _integer(plan.get("right_index"), f"{target_id}.right_index")
        if left_index not in originals or right_index not in originals:
            raise ValueError(
                f"{snapshot.scene}: target {target_id} endpoints are absent from manifest"
            )
        if status == "ACCEPTED":
            synthetic = synthetics.get(target_id, {})
            generated_value = outcome.get("rgb_path", synthetic.get("rgb_path"))
        else:
            generated_value = outcome.get("candidate_rgb_path")
        left_path = resolver.resolve(originals[left_index].get("rgb_path"), "left source")
        generated_path = resolver.resolve(generated_value, "generated candidate")
        right_path = resolver.resolve(originals[right_index].get("rgb_path"), "right source")
        images = (
            _open_rgb(left_path, "left source"),
            _open_rgb(generated_path, "generated candidate"),
            _open_rgb(right_path, "right source"),
        )
        if len({image.size for image in images}) != 1:
            raise ValueError(
                f"{snapshot.scene}/{target_id}: source and generated resolutions differ: "
                f"{[image.size for image in images]}; 100% zoom would be misleading"
            )
        crop = _auto_detail_crop(images, zoom_size)
        reasons_value = plan.get("reasons", [])
        reasons = (
            [str(value) for value in reasons_value]
            if isinstance(reasons_value, Sequence) and not isinstance(reasons_value, (str, bytes))
            else []
        )
        stem = _safe_stem(target_id, rank)
        full_path = (scene_dir / f"{stem}_{status.lower()}_full.png").resolve()
        zoom_path = (scene_dir / f"{stem}_{status.lower()}_zoom100.png").resolve()
        _compose_triptych(
            images,
            _metadata_lines(target_id, reasons, outcome, status),
            status=status,
            crop=None,
        ).save(full_path)
        _compose_triptych(
            images,
            _metadata_lines(target_id, reasons, outcome, status, crop=crop),
            status=status,
            crop=crop,
        ).save(zoom_path)
        visuals.append(
            {
                "rank": rank,
                "status": status,
                "counted_as_added": status == "ACCEPTED",
                "target_id": target_id,
                "reasons": reasons,
                "failures": [str(value) for value in outcome.get("failures", [])],
                "metrics": {name: _metric(outcome, name) for name in _METRIC_NAMES},
                "left_source_path": str(left_path),
                "generated_path": str(generated_path),
                "right_source_path": str(right_path),
                "full_triptych_path": str(full_path),
                "zoom100_triptych_path": str(zoom_path),
                "zoom_crop_xywh": list(crop),
            }
        )
    return ranked_accepted, visuals


def summarise_snapshots(
    snapshots: Sequence[Snapshot],
    output_dir: Path | str,
    *,
    top_k: int = 3,
    zoom_size: int = 192,
) -> Mapping[str, Any]:
    if top_k < 0:
        raise ValueError("top_k must be non-negative (zero means all accepted)")
    if zoom_size < 1:
        raise ValueError("zoom_size must be positive")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if len({snapshot.scene for snapshot in snapshots}) != len(snapshots):
        raise ValueError("scene names must be unique")

    scenes: list[dict[str, Any]] = []
    for snapshot in snapshots:
        counts = summarise_counts(snapshot)
        ranked, visuals = render_scene(
            snapshot, output, top_k=top_k, zoom_size=zoom_size
        )
        scenes.append(
            {
                "scene": snapshot.scene,
                "report_path": str(snapshot.report_path),
                "manifest_path": str(snapshot.manifest_path),
                **counts,
                "accepted_ranked_by_sharpness_gain": ranked,
                "visuals": visuals,
            }
        )
    json_path = (output / "summary.json").resolve()
    csv_path = (output / "summary.csv").resolve()
    payload: Mapping[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "output_dir": str(output),
        "summary_json_path": str(json_path),
        "summary_csv_path": str(csv_path),
        "top_k": top_k,
        "zoom_size_pixels": zoom_size,
        "scenes": scenes,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    fieldnames = (
        "scene",
        "source",
        "planned",
        "selected",
        "accepted",
        "rejected",
        "not_evaluated",
        "selected_without_gate_outcome",
        "accepted_over_source",
        "accepted_over_source_plus_accepted",
        "accepted_over_planned",
        "accepted_over_selected",
        "best_visual_status",
        "best_visual_target_id",
        "best_visual_sharpness_gain",
        "best_visual_full_triptych_path",
        "best_visual_zoom100_triptych_path",
        "report_path",
        "manifest_path",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scene in scenes:
            visual = scene["visuals"][0] if scene["visuals"] else {}
            writer.writerow(
                {
                    **{key: scene.get(key) for key in fieldnames},
                    "best_visual_status": visual.get("status"),
                    "best_visual_target_id": visual.get("target_id"),
                    "best_visual_sharpness_gain": visual.get("metrics", {}).get(
                        "sharpness_gain"
                    ),
                    "best_visual_full_triptych_path": visual.get("full_triptych_path"),
                    "best_visual_zoom100_triptych_path": visual.get(
                        "zoom100_triptych_path"
                    ),
                }
            )
    print(
        json.dumps(
            {
                "summary_json": str(json_path),
                "summary_csv": str(csv_path),
                "scene_count": len(scenes),
            },
            ensure_ascii=False,
        )
    )
    return payload


def _parse_scene_spec(specification: str) -> tuple[str | None, Path]:
    if "=" in specification:
        scene, value = specification.split("=", 1)
        if not scene.strip() or not value.strip():
            raise ValueError(f"invalid --scene specification {specification!r}")
        return scene.strip(), Path(value)
    return None, Path(specification)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Exact report/manifest or a directory containing one immutable pair.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Name a scene explicitly; may be repeated.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Accepted visualisations per scene; 0 renders all (default: 3).",
    )
    parser.add_argument(
        "--zoom-size", type=int, default=192, help="Native-pixel square crop size."
    )
    args = parser.parse_args(argv)
    if not args.inputs and not args.scene:
        parser.error("provide at least one input or --scene NAME=PATH")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        specifications = [(None, Path(value)) for value in args.inputs]
        specifications.extend(_parse_scene_spec(value) for value in args.scene)
        snapshots = [load_snapshot(scene, path) for scene, path in specifications]
        summarise_snapshots(
            snapshots,
            args.output_dir,
            top_k=args.top_k,
            zoom_size=args.zoom_size,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
