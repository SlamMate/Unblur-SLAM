#!/usr/bin/env python3
"""Audit and visualize the bounded fr2_xyz causal-video smoke.

The comparison protocol is intentionally fixed before looking at the causal
result:

* metric scope: the 11 clear-GT source frames in the 221-frame prefix;
* checkpoint: iteration 100;
* visualization sources: 0, 72, and 220;
* panel order: GT | baseline | replay | causal.

By default the script only validates inputs and prints a JSON summary.  Passing
``--output`` writes the audit JSON/CSV and montages, but only after all arm,
source-set, GT-half, RGB-domain, and provenance checks have passed.  This keeps
an invalid/partial causal run from silently entering a report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml


EXPECTED_SOURCES = [0, 9, 15, 49, 58, 72, 89, 109, 125, 166, 220]
VISUAL_SOURCES = [0, 72, 220]
EXPECTED_ITERATION = 100
EXPECTED_CHECKPOINTS = [25, 50, 100]
EXPECTED_METRIC_SCOPE = "clear_gt_prefix_smoke"
EXPECTED_EVSSM_SHA256 = (
    "4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41"
)
ARM_ORDER = ("baseline", "replay", "causal")
FINAL_METRIC_RE = re.compile(
    r"^mean psnr: (?P<psnr>[-+0-9.eE]+), "
    r"ssim: (?P<ssim>[-+0-9.eE]+), "
    r"lpips: (?P<lpips>[-+0-9.eE]+),",
    re.MULTILINE,
)
ATE_RE = re.compile(r"'rmse':\s*(?P<rmse>[-+0-9.eE]+)")
FALLBACK_RE = re.compile(
    r"causal candidate failed EVSSM gate; frame=(?P<frame>\d+) "
    r"vs_evssm=(?P<vs>[-+0-9.eE]+) "
    r"fallback_gain=(?P<gain>[-+0-9.eE]+) "
    r"(?:apply=(?P<apply>True|False)|"
    r"tracking=(?P<tracking>[a-z_]+) cache=(?P<cache>[a-z_]+))"
)
SAFE_RE = re.compile(
    r"TRACKING: streaming deblur frame=(?P<frame>\d+).*?"
    r"replace=(?P<replace>True|False)"
)


class ContractError(RuntimeError):
    """Raised when a report input violates the predeclared protocol."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ContractError(f"missing {label}: {path}")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain a JSON object: {path}")
    return value


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain a YAML mapping: {path}")
    return value


def _read_checkpoint_metrics(scene_root: Path) -> dict[str, Any]:
    report = _load_json(
        scene_root / "refinement_checkpoint_metrics.json",
        "refinement checkpoint metrics",
    )
    if report.get("metric_scope") != EXPECTED_METRIC_SCOPE:
        raise ContractError(
            f"{scene_root}: metric_scope={report.get('metric_scope')!r}, "
            f"expected {EXPECTED_METRIC_SCOPE!r}"
        )
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ContractError(f"{scene_root}: checkpoints must be a list")
    iterations = [int(entry.get("iteration", -1)) for entry in checkpoints]
    if iterations != EXPECTED_CHECKPOINTS:
        raise ContractError(
            f"{scene_root}: checkpoint iterations {iterations}, "
            f"expected {EXPECTED_CHECKPOINTS}"
        )
    for entry in checkpoints:
        sources = [
            int(value) for value in entry.get("evaluated_source_indices", [])
        ]
        if sources != EXPECTED_SOURCES:
            raise ContractError(
                f"{scene_root}: iteration {entry.get('iteration')} source set/order "
                f"{sources}, expected {EXPECTED_SOURCES}"
            )
        if int(entry.get("num_frames", -1)) != len(EXPECTED_SOURCES):
            raise ContractError(
                f"{scene_root}: iteration {entry.get('iteration')} num_frames is not 11"
            )
    matching = [
        entry
        for entry in checkpoints
        if int(entry.get("iteration", -1)) == EXPECTED_ITERATION
    ]
    if len(matching) != 1:
        raise ContractError(
            f"{scene_root}: expected one iteration-{EXPECTED_ITERATION} entry"
        )
    checkpoint = matching[0]
    selected = report.get("selected_checkpoint") or {}
    if int(selected.get("iteration", -1)) != EXPECTED_ITERATION:
        raise ContractError(f"{scene_root}: selected checkpoint is not iteration 100")
    if bool(report.get("test_metric_used_for_selection", True)):
        raise ContractError(f"{scene_root}: test metric was used for selection")
    return {
        "report": report,
        "checkpoint": checkpoint,
        "psnr_by_iteration": {
            str(int(entry["iteration"])): float(entry["mean_psnr_db"])
            for entry in checkpoints
        },
    }


def _read_tracking_timestamps(scene_root: Path) -> list[int]:
    video_path = _require_file(scene_root / "video.npz", "final depth video")
    with np.load(video_path) as archive:
        if "timestamps" not in archive:
            raise ContractError(f"{video_path}: missing timestamps")
        timestamps = [int(round(float(value))) for value in archive["timestamps"]]
    if timestamps != EXPECTED_SOURCES:
        raise ContractError(
            f"{scene_root}: actual tracking timestamps {timestamps}, "
            f"expected {EXPECTED_SOURCES}"
        )
    return timestamps


def _read_final_metrics(arm_root: Path) -> dict[str, float]:
    log_path = _require_file(arm_root / "launch.log", "persistent launch log")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = list(FINAL_METRIC_RE.finditer(log_text))
    if len(matches) != 1:
        raise ContractError(
            f"{log_path}: expected one final PSNR/SSIM/LPIPS line, got {len(matches)}"
        )
    values = {key: float(value) for key, value in matches[0].groupdict().items()}
    return values


def _read_ate(scene_root: Path) -> float:
    path = _require_file(
        scene_root / "traj" / "metrics_full_traj.txt", "full-trajectory metrics"
    )
    match = ATE_RE.search(path.read_text(encoding="utf-8", errors="replace"))
    if match is None:
        raise ContractError(f"{path}: could not parse ATE RMSE")
    return float(match.group("rmse"))


def _read_runtime(scene_root: Path) -> dict[str, float]:
    runtime = _load_json(scene_root / "runtime_stats.json", "runtime stats")
    fields = (
        "online_inference_time",
        "total_inference_time",
        "peak_gpu_memory",
        "peak_cpu_memory",
    )
    try:
        return {field: float(runtime[field]) for field in fields}
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"{scene_root}: invalid runtime stats") from error


def _read_replay_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "present": False,
            "observe_events": 0,
            "sample_events": 0,
            "sample_steps": [],
            "samples_per_step": {},
        }
    counts: dict[int, int] = {}
    observe_events = 0
    sample_events = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("event") == "observe":
                observe_events += 1
            elif row.get("event") == "sample":
                sample_events += 1
                step = int(row["step"])
                counts[step] = counts.get(step, 0) + 1
    return {
        "present": True,
        "observe_events": observe_events,
        "sample_events": sample_events,
        "sample_steps": sorted(counts),
        "samples_per_step": {str(step): counts[step] for step in sorted(counts)},
    }


def _tensor_from_payload(payload: Any, path: Path) -> torch.Tensor:
    if isinstance(payload, dict):
        payload = payload.get("tensor")
    if not torch.is_tensor(payload):
        raise ContractError(f"{path}: expected a tensor or {{'tensor': tensor}}")
    return payload.detach().float().cpu()


def _validate_causal_rgb_domain(scene_root: Path) -> dict[str, float]:
    sharp_root = scene_root / "sharp"
    paths = sorted(sharp_root.glob("*.pt"), key=lambda path: int(path.stem))
    if not paths:
        raise ContractError(f"{scene_root}: causal run has no saved sharp tensors")
    minimum = float("inf")
    maximum = float("-inf")
    for path in paths:
        tensor = _tensor_from_payload(torch.load(path, map_location="cpu"), path)
        if not bool(torch.isfinite(tensor).all()):
            raise ContractError(f"{path}: saved RGB contains NaN/Inf")
        minimum = min(minimum, float(tensor.min()))
        maximum = max(maximum, float(tensor.max()))
    tolerance = 1e-5
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ContractError(
            f"{scene_root}: causal saved RGB domain [{minimum}, {maximum}] is not [0,1]; "
            "possible in-place ImageNet-normalization alias"
        )
    return {
        "saved_tensor_count": len(paths),
        "saved_rgb_min": minimum,
        "saved_rgb_max": maximum,
    }


def _causal_log_stats(arm_root: Path, expected_frames: int) -> dict[str, Any]:
    log_path = _require_file(arm_root / "launch.log", "causal launch log")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    fallbacks = list(FALLBACK_RE.finditer(text))
    safe = list(SAFE_RE.finditer(text))
    if len(fallbacks) + len(safe) != expected_frames:
        raise ContractError(
            f"{log_path}: causal decisions={len(fallbacks) + len(safe)}, "
            f"expected {expected_frames}"
        )
    decided_frames = sorted(
        [int(match.group("frame")) for match in fallbacks]
        + [int(match.group("frame")) for match in safe]
    )
    if decided_frames != list(range(expected_frames)):
        raise ContractError(f"{log_path}: causal decisions do not cover 0..220 exactly")
    fallback_apply = sum(match.group("apply") == "True" for match in fallbacks)
    fallback_tracking_raw = sum(
        match.group("tracking") == "raw" for match in fallbacks
    )
    fallback_cache_evssm = sum(match.group("cache") == "evssm" for match in fallbacks)
    safe_replace = sum(match.group("replace") == "True" for match in safe)
    return {
        "fallback_count": len(fallbacks),
        "fallback_apply_count": fallback_apply,
        "fallback_tracking_raw_count": fallback_tracking_raw,
        "fallback_cache_evssm_count": fallback_cache_evssm,
        "safe_count": len(safe),
        "safe_replace_count": safe_replace,
        "vs_evssm_values": [float(match.group("vs")) for match in fallbacks],
        "fallback_gain_values": [float(match.group("gain")) for match in fallbacks],
    }


def _validate_frontend_provenance(arm: str, scene_root: Path) -> dict[str, Any]:
    cfg_path = scene_root / "cfg.yaml"
    cfg = _load_yaml(cfg_path, "resolved config")
    deblur = cfg.get("deblur") or {}
    frontend = str(deblur.get("frontend", ""))
    expected_frontend = "causal_evssm" if arm == "causal" else "evssm"
    if frontend != expected_frontend:
        raise ContractError(
            f"{scene_root}: frontend={frontend!r}, expected {expected_frontend!r}"
        )
    if str(deblur.get("turtle_checkpoint", "")) or str(
        deblur.get("turtle_config", "")
    ):
        raise ContractError(f"{scene_root}: Turtle/GoPro path must be empty")
    evssm_sha = str(
        deblur.get("evssm_checkpoint_sha256", cfg.get("evssm_checkpoint_sha256", ""))
    )
    if arm == "causal" and evssm_sha != EXPECTED_EVSSM_SHA256:
        raise ContractError(
            f"{scene_root}: causal teacher EVSSM SHA={evssm_sha}, "
            f"expected {EXPECTED_EVSSM_SHA256}"
        )
    result: dict[str, Any] = {
        "frontend": frontend,
        "resolved_config_sha256": _sha256_file(cfg_path),
        "evssm_checkpoint": str(cfg.get("evssm_checkpoint", "")),
        "evssm_checkpoint_sha256": evssm_sha,
    }
    if arm == "causal":
        checkpoint = _require_file(
            Path(str(deblur.get("causal_checkpoint", ""))), "causal checkpoint"
        )
        configured_sha = str(deblur.get("causal_checkpoint_sha256", ""))
        actual_sha = _sha256_file(checkpoint)
        if configured_sha != actual_sha:
            raise ContractError(
                f"{scene_root}: causal checkpoint SHA mismatch: "
                f"configured={configured_sha}, actual={actual_sha}"
            )
        teacher = deblur.get("causal_teacher_provenance") or {}
        if teacher.get("evssm_checkpoint_sha256") != EXPECTED_EVSSM_SHA256:
            raise ContractError(f"{scene_root}: causal teacher is not official EVSSM")
        if not bool(teacher.get("teacher_artifacts_verified", False)):
            raise ContractError(f"{scene_root}: causal teacher artifacts not verified")
        result.update(
            {
                "causal_checkpoint": str(checkpoint),
                "causal_checkpoint_sha256": actual_sha,
                "causal_teacher_provenance": teacher,
            }
        )
    return result


def _split_gt_render(path: Path) -> tuple[Image.Image, Image.Image]:
    _require_file(path, "GT/render checkpoint image")
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    if width % 2 != 0 or width // 2 != 512 or height != 384:
        raise ContractError(
            f"{path}: expected a 1024x384 GT|render image, got {width}x{height}"
        )
    midpoint = width // 2
    return image.crop((0, 0, midpoint, height)), image.crop(
        (midpoint, 0, width, height)
    )


def _image_pixel_sha(image: Image.Image) -> str:
    return _sha256_bytes(np.asarray(image, dtype=np.uint8).tobytes(order="C"))


def _load_render_panels(scene_roots: dict[str, Path]) -> dict[str, Any]:
    panels: dict[str, Any] = {arm: {} for arm in scene_roots}
    gt_hashes: dict[int, str] = {}
    for arm, scene_root in scene_roots.items():
        render_root = (
            scene_root
            / "refinement_checkpoints"
            / f"iter_{EXPECTED_ITERATION:06d}"
            / "clear_gt_renders"
        )
        expected_names = {
            f"source_{source:06d}_gt_render.png" for source in EXPECTED_SOURCES
        }
        actual_names = {path.name for path in render_root.glob("*_gt_render.png")}
        if actual_names != expected_names:
            raise ContractError(
                f"{render_root}: render file set differs from the 11-source contract"
            )
        for source in EXPECTED_SOURCES:
            path = render_root / f"source_{source:06d}_gt_render.png"
            gt, render = _split_gt_render(path)
            gt_hash = _image_pixel_sha(gt)
            if source in gt_hashes and gt_hashes[source] != gt_hash:
                raise ContractError(
                    f"source {source}: GT half differs between comparison arms"
                )
            gt_hashes[source] = gt_hash
            panels[arm][source] = {"path": path, "gt": gt, "render": render}
    return {"panels": panels, "gt_pixel_sha256": gt_hashes}


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _labeled_panel(image: Image.Image, label: str, source: int) -> Image.Image:
    title_height = 42
    panel = Image.new("RGB", (image.width, image.height + title_height), (20, 20, 20))
    panel.paste(image, (0, title_height))
    draw = ImageDraw.Draw(panel)
    draw.text(
        (12, 9),
        f"{label} | source {source}",
        fill=(245, 245, 245),
        font=_font(20),
    )
    return panel


def _make_source_montage(
    source: int, panels: dict[str, dict[int, dict[str, Any]]]
) -> Image.Image:
    ordered = [
        _labeled_panel(panels["baseline"][source]["gt"], "GT", source),
        _labeled_panel(
            panels["baseline"][source]["render"], "baseline", source
        ),
        _labeled_panel(panels["replay"][source]["render"], "replay", source),
        _labeled_panel(panels["causal"][source]["render"], "causal", source),
    ]
    output = Image.new(
        "RGB", (sum(image.width for image in ordered), ordered[0].height), (0, 0, 0)
    )
    offset = 0
    for image in ordered:
        output.paste(image, (offset, 0))
        offset += image.width
    return output


def _write_reports(
    output: Path,
    summary: dict[str, Any],
    panels: dict[str, dict[int, dict[str, Any]]],
) -> None:
    if output.exists() and any(output.iterdir()):
        raise ContractError(f"refusing to overwrite non-empty report dir: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        with (temporary / "three_arm_audit.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        metric_fields = (
            "arm",
            "psnr_iter_25_db",
            "psnr_iter_50_db",
            "psnr_iter_100_db",
            "checkpoint_psnr_db",
            "final_psnr_db",
            "final_ssim",
            "final_lpips",
            "ate_rmse_m",
            "online_inference_time_s",
            "total_inference_time_s",
            "tracking_keyframes",
            "offline_replay_samples",
            "online_replay_samples",
            "causal_fallback_count",
            "causal_safe_count",
            "causal_replace_count",
        )
        with (temporary / "three_arm_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=metric_fields)
            writer.writeheader()
            for arm in ARM_ORDER:
                values = summary["arms"][arm]
                causal = values.get("causal_decisions", {})
                writer.writerow(
                    {
                        "arm": arm,
                        "psnr_iter_25_db": values[
                            "checkpoint_psnr_db_by_iteration"
                        ]["25"],
                        "psnr_iter_50_db": values[
                            "checkpoint_psnr_db_by_iteration"
                        ]["50"],
                        "psnr_iter_100_db": values[
                            "checkpoint_psnr_db_by_iteration"
                        ]["100"],
                        "checkpoint_psnr_db": values["checkpoint_psnr_db"],
                        "final_psnr_db": values["final_metrics"]["psnr"],
                        "final_ssim": values["final_metrics"]["ssim"],
                        "final_lpips": values["final_metrics"]["lpips"],
                        "ate_rmse_m": values["ate_rmse_m"],
                        "online_inference_time_s": values["runtime"][
                            "online_inference_time"
                        ],
                        "total_inference_time_s": values["runtime"][
                            "total_inference_time"
                        ],
                        "tracking_keyframes": len(values["tracking_timestamps"]),
                        "offline_replay_samples": values["offline_replay"][
                            "sample_events"
                        ],
                        "online_replay_samples": values["online_replay"][
                            "sample_events"
                        ],
                        "causal_fallback_count": causal.get("fallback_count", ""),
                        "causal_safe_count": causal.get("safe_count", ""),
                        "causal_replace_count": causal.get(
                            "safe_replace_count", ""
                        ),
                    }
                )
        source_montages = []
        for source in VISUAL_SOURCES:
            montage = _make_source_montage(source, panels)
            montage.save(
                temporary / f"source_{source:06d}_gt_baseline_replay_causal.png"
            )
            source_montages.append(montage)
        gap = 8
        combined = Image.new(
            "RGB",
            (
                source_montages[0].width,
                sum(image.height for image in source_montages)
                + gap * (len(source_montages) - 1),
            ),
            (0, 0, 0),
        )
        y = 0
        for montage in source_montages:
            combined.paste(montage, (0, y))
            y += montage.height + gap
        combined.save(temporary / "selected_sources_gt_baseline_replay_causal.png")
        if output.exists():
            output.rmdir()
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def audit(root: Path, arms: Iterable[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_arms = tuple(arms)
    if not selected_arms:
        raise ContractError("at least one arm is required")
    scene_roots = {arm: root / arm / "freiburg2_xyz" for arm in selected_arms}
    summary: dict[str, Any] = {
        "schema": "unblur_slam.fr2_xyz_causal_smoke_report.v1",
        "metric_scope": EXPECTED_METRIC_SCOPE,
        "iteration": EXPECTED_ITERATION,
        "checkpoint_iterations": EXPECTED_CHECKPOINTS,
        "expected_sources": EXPECTED_SOURCES,
        "visual_sources_predeclared": VISUAL_SOURCES,
        "arms": {},
    }
    for arm, scene_root in scene_roots.items():
        if not scene_root.is_dir():
            raise ContractError(f"missing completed arm: {scene_root}")
        checkpoint = _read_checkpoint_metrics(scene_root)
        final_metrics = _read_final_metrics(root / arm)
        arm_summary: dict[str, Any] = {
            "checkpoint_psnr_db": float(
                checkpoint["checkpoint"]["mean_psnr_db"]
            ),
            "checkpoint_psnr_db_by_iteration": checkpoint[
                "psnr_by_iteration"
            ],
            "final_metrics": final_metrics,
            "tracking_timestamps": _read_tracking_timestamps(scene_root),
            "ate_rmse_m": _read_ate(scene_root),
            "runtime": _read_runtime(scene_root),
            "offline_replay": _read_replay_csv(
                scene_root / "resplat_replay_final.csv"
            ),
            "online_replay": _read_replay_csv(
                scene_root / "resplat_replay_online.csv"
            ),
            "provenance": _validate_frontend_provenance(arm, scene_root),
        }
        if arm == "causal":
            arm_summary["causal_rgb_domain"] = _validate_causal_rgb_domain(
                scene_root
            )
            arm_summary["causal_decisions"] = _causal_log_stats(
                root / arm, expected_frames=221
            )
        summary["arms"][arm] = arm_summary
    render_info = _load_render_panels(scene_roots)
    summary["gt_pixel_sha256"] = {
        str(source): digest
        for source, digest in render_info["gt_pixel_sha256"].items()
    }
    return summary, render_info["panels"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/srv/szha0669/unblur-slam/slam_smoke/fr2_xyz_causal_smoke"
        ),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARM_ORDER,
        default=list(ARM_ORDER),
        help="completed arms to validate; montages require all three",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the final report only after every validation succeeds",
    )
    args = parser.parse_args()
    try:
        summary, panels = audit(args.root.resolve(), args.arms)
        if args.output is not None:
            if tuple(args.arms) != ARM_ORDER:
                raise ContractError("writing montages requires baseline replay causal")
            _write_reports(args.output.resolve(), summary, panels)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except ContractError as error:
        print(f"report contract failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
