#!/usr/bin/env python3
"""Render auditable evidence for the advanced FrameCrafter preprocessor.

The preprocessor report is the source of truth for quality gates and context
provenance.  The paired manifest is the source of truth for what was actually
injected into SLAM.  For every gated target this script writes:

* a contact sheet containing every conditioning image and its role/provenance;
* a full-resolution-support triptych (left support, generated, right support);
* a shared native-pixel detail crop (one output pixel per source pixel).

Advanced reports are required by default.  ``--legacy-policy compat`` can
render older v1 reports by inferring partitions and falling back to target
endpoints when local gate supports were not recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPORT_SCHEMA = "unblur_slam.framecrafter_preprocess_report.v1"
MANIFEST_SCHEMA = "unblur_slam.framecrafter_manifest.v1"
SUMMARY_SCHEMA = "unblur_slam.framecrafter_advanced_visualization.v1"

PARTITION_ORDER = ("sharp_accepted", "geometry_only", "rejected")
PARTITION_LABELS = {
    "sharp_accepted": "SHARP ACCEPTED",
    "geometry_only": "GEOMETRY ONLY",
    "rejected": "REJECTED",
}
PARTITION_COLORS = {
    "sharp_accepted": (29, 150, 73),
    "geometry_only": (220, 155, 0),
    "rejected": (210, 48, 48),
}
METRIC_NAMES = (
    "sharpness_gain",
    "depth_coverage",
    "depth_consistency",
    "photometric_error",
    "reprojection_error_px",
    "reprojection_valid_ratio",
)


@dataclass(frozen=True)
class Snapshot:
    input_root: Path
    report_path: Path
    manifest_path: Path
    report: Mapping[str, Any]
    manifest: Mapping[str, Any]


class ArtifactResolver:
    """Resolve recorded paths, preferring an immutable snapshot-local copy."""

    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot
        self._files: list[Path] | None = None
        self._cache: dict[str, Path] = {}

    def resolve(self, value: Any, label: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} is missing a non-empty path")
        if value in self._cache:
            return self._cache[value]
        declared = Path(value).expanduser()
        direct = []
        if not declared.is_absolute():
            direct = [
                self.snapshot.report_path.parent / declared,
                self.snapshot.manifest_path.parent / declared,
                self.snapshot.input_root / declared,
            ]
        local = self._local_match(declared)
        if local is not None:
            result = local.resolve()
        else:
            candidates = [path.resolve() for path in direct if path.is_file()]
            if declared.is_file():
                candidates.append(declared.resolve())
            if not candidates:
                raise FileNotFoundError(f"{label} does not exist: {declared}")
            result = candidates[0]
        self._cache[value] = result
        return result

    def _local_match(self, declared: Path) -> Path | None:
        if not self.snapshot.input_root.is_dir():
            return None
        if self._files is None:
            self._files = [
                path for path in self.snapshot.input_root.rglob("*") if path.is_file()
            ]
        matches = [path for path in self._files if path.name == declared.name]
        if not matches:
            return None

        def suffix_score(path: Path) -> int:
            score = 0
            for left, right in zip(reversed(path.parts), reversed(declared.parts)):
                if left != right:
                    break
                score += 1
            return score

        best_score = max(suffix_score(path) for path in matches)
        best = sorted(path for path in matches if suffix_score(path) == best_score)
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


def _find_pair(
    owner_path: Path,
    owner: Mapping[str, Any],
    *,
    wanted_schema: str,
    declared_key: str,
) -> Path:
    value = owner.get(declared_key)
    if isinstance(value, str) and value.strip():
        declared = Path(value).expanduser()
        if not declared.is_absolute():
            declared = owner_path.parent / declared
        if declared.is_file() and _read_json(declared).get("schema") == wanted_schema:
            return declared.resolve()
    signature, generation_id = _identity(owner)
    prefix = "manifest" if wanted_schema == MANIFEST_SCHEMA else "preprocess_report"
    exact = (
        f"{prefix}_{signature}_{generation_id}.json"
        if signature and generation_id
        else None
    )
    paths = (
        list(owner_path.parent.rglob(exact))
        if exact
        else list(owner_path.parent.rglob(f"{prefix}*.json"))
    )
    matches: list[Path] = []
    for path in paths:
        try:
            payload = _read_json(path)
        except ValueError:
            continue
        if payload.get("schema") != wanted_schema:
            continue
        if signature and _identity(payload) != (signature, generation_id):
            continue
        matches.append(path.resolve())
    if len(matches) != 1:
        detail = "none" if not matches else ", ".join(str(path) for path in matches)
        raise ValueError(f"cannot identify one paired {prefix} JSON; found {detail}")
    return matches[0]


def load_snapshot(input_path: Path | str) -> Snapshot:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"FrameCrafter snapshot does not exist: {path}")
    input_root = path if path.is_dir() else path.parent
    if path.is_dir():
        candidates = []
        for candidate in path.rglob("preprocess_report*.json"):
            try:
                if _read_json(candidate).get("schema") == REPORT_SCHEMA:
                    candidates.append(candidate.resolve())
            except ValueError:
                pass
        if len(candidates) != 1:
            detail = "none" if not candidates else ", ".join(map(str, candidates))
            raise ValueError(
                "snapshot directory must contain exactly one preprocess report; "
                f"found {detail}"
            )
        report_path = candidates[0]
        report = _read_json(report_path)
        manifest_path = _find_pair(
            report_path,
            report,
            wanted_schema=MANIFEST_SCHEMA,
            declared_key="manifest",
        )
        manifest = _read_json(manifest_path)
    else:
        payload = _read_json(path)
        if payload.get("schema") == REPORT_SCHEMA:
            report_path, report = path, payload
            manifest_path = _find_pair(
                path,
                payload,
                wanted_schema=MANIFEST_SCHEMA,
                declared_key="manifest",
            )
            manifest = _read_json(manifest_path)
        elif payload.get("schema") == MANIFEST_SCHEMA:
            manifest_path, manifest = path, payload
            report_path = _find_pair(
                path,
                payload,
                wanted_schema=REPORT_SCHEMA,
                declared_key="preprocess_report_path",
            )
            report = _read_json(report_path)
        else:
            raise ValueError(f"unsupported FrameCrafter JSON schema in {path}")
    if report.get("schema") != REPORT_SCHEMA or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("FrameCrafter report/manifest schema mismatch")
    left_identity, right_identity = _identity(report), _identity(manifest)
    if all(left_identity) and left_identity != right_identity:
        raise ValueError(
            f"report/manifest identity mismatch: {left_identity} vs {right_identity}"
        )
    return Snapshot(
        input_root=input_root,
        report_path=report_path.resolve(),
        manifest_path=manifest_path.resolve(),
        report=report,
        manifest=manifest,
    )


def _objects(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{label} must be a list of objects")
    return list(value)


def _legacy_partitions(report: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {
        name: [] for name in PARTITION_ORDER
    }
    for record in _objects(report.get("accepted", []), "report.accepted"):
        name = str(record.get("acceptance_class", "sharp_accepted"))
        result[name if name in result else "sharp_accepted"].append(record)
    for record in _objects(report.get("rejected", []), "report.rejected"):
        failures = {str(value) for value in record.get("failures", [])}
        name = str(record.get("acceptance_class", ""))
        if name not in result:
            name = "geometry_only" if failures and failures <= {"sharpness_gain"} else "rejected"
        result[name].append(record)
    return result


def quality_partitions(
    report: Mapping[str, Any], *, legacy_policy: str
) -> dict[str, list[Mapping[str, Any]]]:
    value = report.get("quality_partition")
    if not isinstance(value, Mapping):
        if legacy_policy == "fail":
            raise ValueError(
                "report has no advanced quality_partition; rerun the advanced "
                "preprocessor or pass --legacy-policy compat"
            )
        return _legacy_partitions(report)
    result = {
        name: _objects(value.get(name, []), f"quality_partition.{name}")
        for name in PARTITION_ORDER
    }
    seen: dict[str, str] = {}
    for name, records in result.items():
        for record in records:
            target_id = str(record.get("target_id", ""))
            if not target_id:
                raise ValueError(f"quality_partition.{name} contains an empty target_id")
            if target_id in seen:
                raise ValueError(
                    f"target {target_id} appears in both {seen[target_id]} and {name}"
                )
            seen[target_id] = name
    return result


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
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
            pass
    return ImageFont.load_default()


def _open_rgb(path: Path, label: str) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load {label} image {path}: {error}") from error


def _wrap(lines: Sequence[str], width: int, font: ImageFont.ImageFont) -> list[str]:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    output: list[str] = []
    for line in lines:
        remaining = str(line)
        if not remaining:
            output.append("")
            continue
        while draw.textbbox((0, 0), remaining, font=font)[2] > width:
            low, high = 1, len(remaining)
            while low < high:
                middle = (low + high + 1) // 2
                if draw.textbbox((0, 0), remaining[:middle], font=font)[2] <= width:
                    low = middle
                else:
                    high = middle - 1
            split = max(1, low)
            whitespace = remaining.rfind(" ", 0, split + 1)
            if whitespace >= max(1, split // 2):
                split = whitespace
            output.append(remaining[:split].rstrip())
            remaining = remaining[split:].lstrip()
        output.append(remaining)
    return output


def _header(
    width: int,
    lines: Sequence[str],
    partition: str,
) -> Image.Image:
    body, bold = _font(17), _font(18, bold=True)
    wrapped: list[str] = []
    for index, line in enumerate(lines):
        wrapped.extend(
            _wrap([line], max(1, width - 30), bold if index == 0 else body)
        )
    line_height = 24
    height = 16 + line_height * len(wrapped)
    image = Image.new("RGB", (width, height), (16, 16, 16))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 9, height - 1), fill=PARTITION_COLORS[partition])
    for index, line in enumerate(wrapped):
        draw.text(
            (18, 7 + index * line_height),
            line,
            font=bold if index == 0 else body,
            fill=(255, 255, 255),
        )
    return image


def _safe_stem(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")[:80] or "target"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{clean}_{digest}"


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric(record: Mapping[str, Any], name: str) -> float | None:
    metrics = record.get("metrics", record.get("gate_metrics", {}))
    return _number(metrics.get(name)) if isinstance(metrics, Mapping) else None


def _format_number(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _score_label(item: Mapping[str, Any]) -> tuple[str, str]:
    score = item.get("score", {})
    if not isinstance(score, Mapping):
        return "n/a", _format_number(score)
    return _format_number(score.get("overlap")), _format_number(score.get("total"))


def _conditioning_attempted_evssm(item: Mapping[str, Any]) -> bool:
    return str(item.get("resolved_mode", "")).lower() == "evssm" or bool(
        item.get("fallback_reason")
    )


def render_conditioning_sheet(
    target_id: str,
    partition: str,
    conditioning: Sequence[Mapping[str, Any]],
    resolver: ArtifactResolver,
    output_path: Path,
    *,
    panel_width: int,
) -> dict[str, int]:
    if not conditioning:
        raise ValueError(f"target {target_id} has no conditioning provenance")
    images: list[Image.Image] = []
    labels: list[list[str]] = []
    for ordinal, item in enumerate(conditioning):
        path = resolver.resolve(item.get("resolved_path"), f"conditioning[{ordinal}]")
        image = _open_rgb(path, f"conditioning[{ordinal}]")
        width = panel_width
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        images.append(image)
        overlap, total = _score_label(item)
        mode = str(item.get("resolved_mode", "unknown")).upper()
        requested = str(item.get("requested_mode", "unknown")).upper()
        fallback = str(item.get("fallback_reason") or "none")
        labels.append(
            [
                f"#{ordinal + 1} {item.get('role', 'unknown')} | {mode}",
                f"src={item.get('source_index', 'n/a')} requested={requested}",
                f"overlap={overlap} total={total}",
                f"fallback={fallback}",
            ]
        )
    rows = 1 if len(images) <= 3 else 2
    columns = math.ceil(len(images) / rows)
    image_height = max(image.height for image in images)
    label_height = 100
    header = _header(
        columns * panel_width,
        [
            f"{PARTITION_LABELS[partition]}  target={target_id}",
            f"ALL {len(images)} CONDITIONING VIEWS | role, RAW/EVSSM, overlap, score, fallback",
        ],
        partition,
    )
    canvas = Image.new(
        "RGB",
        (columns * panel_width, header.height + rows * (image_height + label_height)),
        (24, 24, 24),
    )
    canvas.paste(header, (0, 0))
    draw = ImageDraw.Draw(canvas)
    body, bold = _font(14), _font(15, bold=True)
    for index, (image, lines) in enumerate(zip(images, labels)):
        row, column = divmod(index, columns)
        x = column * panel_width
        y = header.height + row * (image_height + label_height)
        image_y = y + (image_height - image.height) // 2
        canvas.paste(image, (x, image_y))
        label_y = y + image_height
        draw.rectangle(
            (x, label_y, x + panel_width - 1, label_y + label_height - 1),
            fill=(35, 35, 35),
        )
        for line_index, line in enumerate(lines):
            draw.text(
                (x + 7, label_y + 5 + line_index * 22),
                str(line),
                font=bold if line_index == 0 else body,
                fill=(248, 248, 248),
            )
        draw.rectangle(
            (x, y, x + panel_width - 1, label_y + label_height - 1),
            outline=(100, 100, 100),
            width=1,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    attempted = sum(_conditioning_attempted_evssm(item) for item in conditioning)
    fallback = sum(bool(item.get("fallback_reason")) for item in conditioning)
    return {
        "conditioning_count": len(conditioning),
        "evssm_attempted_count": attempted,
        "evssm_fallback_count": fallback,
    }


def _detail_crop(images: Sequence[Image.Image], requested_size: int) -> tuple[int, int, int, int]:
    width, height = images[0].size
    size = min(requested_size, width, height)
    if size < 1:
        raise ValueError("detail crop size must be positive")
    gray = np.mean(
        [np.asarray(image.convert("L"), dtype=np.float32) / 255.0 for image in images],
        axis=0,
    )
    dx, dy = np.zeros_like(gray), np.zeros_like(gray)
    dx[:, 1:] = np.abs(np.diff(gray, axis=1))
    dy[1:, :] = np.abs(np.diff(gray, axis=0))
    detail = dx + dy
    integral = np.pad(detail, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    step = max(1, size // 20)
    best = (float("-inf"), 0, 0)
    for y in range(0, height - size + 1, step):
        for x in range(0, width - size + 1, step):
            score = float(
                integral[y + size, x + size]
                - integral[y, x + size]
                - integral[y + size, x]
                + integral[y, x]
            )
            score -= 1.0e-9 * (
                (x + size / 2 - width / 2) ** 2
                + (y + size / 2 - height / 2) ** 2
            )
            if score > best[0]:
                best = score, x, y
    # Include the far image borders when the step does not divide the extent.
    candidates = ((width - size, best[2]), (best[1], height - size), (width - size, height - size))
    for x, y in candidates:
        score = float(
            integral[y + size, x + size]
            - integral[y, x + size]
            - integral[y + size, x]
            + integral[y, x]
        )
        if score > best[0]:
            best = score, x, y
    return int(best[1]), int(best[2]), int(size), int(size)


def _triptych(
    images: Sequence[Image.Image],
    partition: str,
    lines: Sequence[str],
    *,
    crop: tuple[int, int, int, int] | None,
) -> Image.Image:
    if len(images) != 3 or len({image.size for image in images}) != 1:
        raise ValueError("gate support and generated images must have one resolution")
    panels = list(images)
    if crop is not None:
        x, y, width, height = crop
        panels = [image.crop((x, y, x + width, y + height)) for image in images]
    panel_width, panel_height = panels[0].size
    header = _header(panel_width * 3, lines, partition)
    label_height = 34
    canvas = Image.new(
        "RGB",
        (panel_width * 3, header.height + label_height + panel_height),
        (20, 20, 20),
    )
    canvas.paste(header, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _font(17, bold=True)
    for index, (panel, label) in enumerate(
        zip(panels, ("LEFT GATE SUPPORT", "FRAMECRAFTER", "RIGHT GATE SUPPORT"))
    ):
        x = index * panel_width
        draw.rectangle(
            (x, header.height, x + panel_width, header.height + label_height),
            fill=(34, 34, 34),
        )
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (x + (panel_width - (bbox[2] - bbox[0])) / 2, header.height + 5),
            label,
            font=font,
            fill=(250, 250, 250),
        )
        canvas.paste(panel, (x, header.height + label_height))
        if index:
            draw.line((x, header.height, x, canvas.height), fill=(255, 255, 255), width=2)
    return canvas


def _conditioning_for_target(
    plan: Mapping[str, Any],
    batch_by_id: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    value = plan.get("conditioning")
    if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
        return list(value)
    batch = batch_by_id.get(str(plan.get("batch_id", "")), {})
    value = batch.get("conditioning")
    if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
        return list(value)
    return []


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _scene_name(report: Mapping[str, Any], report_path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    value = str(report.get("scene", "")).strip()
    if value:
        return value
    parent = report_path.parent.name
    return report_path.parent.parent.name if parent in {"generated", "snapshot"} else parent


def visualize(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    scene: str | None = None,
    crop_size: int = 192,
    conditioning_panel_width: int = 320,
    legacy_policy: str = "fail",
    target_ids: Sequence[str] = (),
    max_targets: int = 0,
) -> dict[str, Any]:
    if crop_size < 1 or conditioning_panel_width < 64 or max_targets < 0:
        raise ValueError("crop/panel sizes must be positive and max_targets non-negative")
    if legacy_policy not in {"fail", "compat"}:
        raise ValueError("legacy_policy must be fail or compat")
    snapshot = load_snapshot(input_path)
    report, manifest = snapshot.report, snapshot.manifest
    partitions = quality_partitions(report, legacy_policy=legacy_policy)
    planned_records = _objects(report.get("planned", []), "report.planned")
    planned = {str(record.get("target_id", "")): record for record in planned_records}
    batches = _objects(report.get("generation_batches", []), "report.generation_batches")
    batch_by_id = {str(batch.get("batch_id", "")): batch for batch in batches}
    frames = _objects(manifest.get("frames", []), "manifest.frames")
    originals = {
        int(frame["source_index"]): frame
        for frame in frames
        if frame.get("kind") == "original" and frame.get("source_index") is not None
    }
    synthetics = {
        str(frame.get("target_id", "")): frame
        for frame in frames
        if frame.get("kind") == "synthetic"
    }
    selected_ids = set(target_ids)
    outcomes: list[tuple[str, Mapping[str, Any]]] = []
    for partition in PARTITION_ORDER:
        for record in sorted(partitions[partition], key=lambda item: str(item.get("target_id", ""))):
            target_id = str(record.get("target_id", ""))
            if not selected_ids or target_id in selected_ids:
                outcomes.append((partition, record))
    if selected_ids:
        missing = sorted(selected_ids - {str(record.get("target_id", "")) for _, record in outcomes})
        if missing:
            raise ValueError(f"requested targets are absent from quality_partition: {missing}")
    if max_targets:
        outcomes = outcomes[:max_targets]
    if not outcomes:
        raise ValueError("report contains no gated targets to visualize")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolver = ArtifactResolver(snapshot)
    target_summaries: list[dict[str, Any]] = []
    for partition, outcome in outcomes:
        target_id = str(outcome.get("target_id", ""))
        plan = planned.get(target_id)
        if plan is None:
            raise ValueError(f"no planned record for gated target {target_id}")
        conditioning = _conditioning_for_target(plan, batch_by_id)
        if not conditioning and legacy_policy == "fail":
            raise ValueError(
                f"target {target_id} has no advanced conditioning provenance; "
                "rerun preprocessing or pass --legacy-policy compat"
            )
        supports = outcome.get("gate_support_source_indices")
        if not (
            isinstance(supports, Sequence)
            and not isinstance(supports, (str, bytes))
            and len(supports) == 2
        ):
            if legacy_policy == "fail":
                raise ValueError(
                    f"target {target_id} has no gate_support_source_indices; "
                    "rerun preprocessing or pass --legacy-policy compat"
                )
            supports = [plan.get("left_index"), plan.get("right_index")]
        try:
            support_indices = [int(supports[0]), int(supports[1])]
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid gate supports for target {target_id}: {supports}") from error
        if any(index not in originals for index in support_indices):
            raise ValueError(
                f"gate supports {support_indices} for target {target_id} are absent from manifest originals"
            )
        synthetic = synthetics.get(target_id, {})
        generated_value = outcome.get("rgb_path", outcome.get("candidate_rgb_path"))
        if not generated_value:
            generated_value = synthetic.get("rgb_path")
        left_path = resolver.resolve(originals[support_indices[0]].get("rgb_path"), "left gate support")
        generated_path = resolver.resolve(generated_value, "generated candidate")
        right_path = resolver.resolve(originals[support_indices[1]].get("rgb_path"), "right gate support")
        images = [
            _open_rgb(left_path, "left gate support"),
            _open_rgb(generated_path, "generated candidate"),
            _open_rgb(right_path, "right gate support"),
        ]
        if len({image.size for image in images}) != 1:
            raise ValueError(
                f"{target_id}: support/generated resolutions differ: {[image.size for image in images]}"
            )
        stem = _safe_stem(target_id)
        contact_path = output / f"{stem}_conditioning.png"
        if conditioning:
            context_counts = render_conditioning_sheet(
                target_id,
                partition,
                conditioning,
                resolver,
                contact_path,
                panel_width=conditioning_panel_width,
            )
        else:
            placeholder = _header(
                max(640, conditioning_panel_width),
                [
                    f"{PARTITION_LABELS[partition]}  target={target_id}",
                    "LEGACY REPORT: conditioning provenance was not recorded",
                ],
                partition,
            )
            placeholder.save(contact_path)
            context_counts = {
                "conditioning_count": 0,
                "evssm_attempted_count": 0,
                "evssm_fallback_count": 0,
            }
        metrics = {name: _metric(outcome, name) for name in METRIC_NAMES}
        lines = [
            f"{PARTITION_LABELS[partition]}  target={target_id}",
            "support={} | sharp_gain={} depth_consistency={} photo_error={} reproj_px={}".format(
                support_indices,
                _format_number(metrics["sharpness_gain"]),
                _format_number(metrics["depth_consistency"]),
                _format_number(metrics["photometric_error"]),
                _format_number(metrics["reprojection_error_px"]),
            ),
        ]
        full_path = output / f"{stem}_gate_full.png"
        _triptych(images, partition, lines, crop=None).save(full_path)
        crop = _detail_crop(images, crop_size)
        detail_path = output / f"{stem}_gate_detail100.png"
        detail_lines = [
            *lines,
            f"100% NATIVE DETAIL x={crop[0]} y={crop[1]} w={crop[2]} h={crop[3]} | 1 px = 1 source px",
        ]
        _triptych(images, partition, detail_lines, crop=crop).save(detail_path)
        failures = outcome.get("failures", [])
        target_summaries.append(
            {
                "target_id": target_id,
                "quality_partition": partition,
                "actually_injected": target_id in synthetics,
                "gate_support_source_indices": support_indices,
                "metrics": metrics,
                "failures": [str(value) for value in failures] if isinstance(failures, list) else [],
                **context_counts,
                "conditioning_sheet_path": str(contact_path.resolve()),
                "gate_full_path": str(full_path.resolve()),
                "gate_detail100_path": str(detail_path.resolve()),
                "detail_crop_xywh": list(crop),
                "left_support_path": str(left_path),
                "generated_path": str(generated_path),
                "right_support_path": str(right_path),
            }
        )

    manifest_injected = set(synthetics)
    partition_ids = {
        name: {str(record.get("target_id", "")) for record in records}
        for name, records in partitions.items()
    }
    unknown_injected = sorted(manifest_injected - set().union(*partition_ids.values()))
    if unknown_injected:
        raise ValueError(
            "manifest contains injected targets absent from quality partitions: "
            f"{unknown_injected}"
        )
    # Count conditioning once per generation batch, not once per target sharing
    # the same diffusion call.
    unique_conditioning: list[Mapping[str, Any]] = []
    seen_batches: set[str] = set()
    for plan in planned_records:
        batch_id = str(plan.get("batch_id", ""))
        identity = batch_id or f"target:{plan.get('target_id', '')}"
        if identity in seen_batches:
            continue
        seen_batches.add(identity)
        unique_conditioning.extend(_conditioning_for_target(plan, batch_by_id))
    attempted = sum(_conditioning_attempted_evssm(item) for item in unique_conditioning)
    fallback = sum(bool(item.get("fallback_reason")) for item in unique_conditioning)
    source_count = int(manifest.get("source_frame_count", len(originals)))
    selected_count = int(report.get("selected_target_count", len(planned_records)))
    injected_count = len(manifest_injected)
    result: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "scene": _scene_name(report, snapshot.report_path, scene),
        "report_path": str(snapshot.report_path),
        "manifest_path": str(snapshot.manifest_path),
        "preprocess_signature": report.get("preprocess_signature"),
        "generation_id": report.get("generation_id"),
        "acceptance_mode": report.get("acceptance_mode", "legacy_unknown"),
        "source_frame_count": source_count,
        "selected_target_count": selected_count,
        "sharp_accepted_count": len(partitions["sharp_accepted"]),
        "geometry_only_count": len(partitions["geometry_only"]),
        "rejected_count": len(partitions["rejected"]),
        "actual_injected_count": injected_count,
        "actual_injected_sharp_count": len(manifest_injected & partition_ids["sharp_accepted"]),
        "actual_injected_geometry_only_count": len(manifest_injected & partition_ids["geometry_only"]),
        "actual_injected_over_source": _ratio(injected_count, source_count),
        "actual_injected_over_source_plus_injected": _ratio(
            injected_count, source_count + injected_count
        ),
        "actual_injected_over_selected": _ratio(injected_count, selected_count),
        "conditioning_view_count_unique_batches": len(unique_conditioning),
        "evssm_attempted_count_unique_batches": attempted,
        "evssm_fallback_count_unique_batches": fallback,
        "evssm_fallback_ratio": _ratio(fallback, attempted),
        "visualized_target_count": len(target_summaries),
        "legacy_policy": legacy_policy,
        "targets": target_summaries,
    }
    summary_json = output / "summary.json"
    summary_csv = output / "summary.csv"
    summary_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    csv_fields = [key for key in result if key != "targets"]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerow({key: result[key] for key in csv_fields})
    result["summary_json"] = str(summary_json.resolve())
    result["summary_csv"] = str(summary_csv.resolve())
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Advanced report, paired manifest, or directory containing one pair.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", help="Optional display name; defaults to report/path.")
    parser.add_argument(
        "--crop-size",
        type=int,
        default=192,
        help="Native-pixel square used by the 100%% detail triptych.",
    )
    parser.add_argument("--conditioning-panel-width", type=int, default=320)
    parser.add_argument(
        "--legacy-policy",
        choices=("fail", "compat"),
        default="fail",
        help="Fail clearly on old reports, or render best-effort legacy evidence.",
    )
    parser.add_argument("--target", action="append", default=[], help="Target id to render; repeatable.")
    parser.add_argument("--max-targets", type=int, default=0, help="0 renders every gated target.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = visualize(
            args.input,
            args.output_dir,
            scene=args.scene,
            crop_size=args.crop_size,
            conditioning_panel_width=args.conditioning_panel_width,
            legacy_policy=args.legacy_policy,
            target_ids=args.target,
            max_targets=args.max_targets,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(
        json.dumps(
            {
                "scene": result["scene"],
                "visualized_target_count": result["visualized_target_count"],
                "summary_json": result["summary_json"],
                "summary_csv": result["summary_csv"],
            }
        )
    )


if __name__ == "__main__":
    main()
