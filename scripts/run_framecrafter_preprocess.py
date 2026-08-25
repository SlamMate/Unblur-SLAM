#!/usr/bin/env python3
"""Create a pose-aware FrameCrafter-augmented sequence and manifest.

The default production backend dynamically imports an external checkout that
implements the FrameCrafter Python API.  No model is downloaded by this script. A
deterministic endpoint blend exists only to exercise CPU plumbing and requires
the explicit ``--allow-test-only-backend`` flag.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.framecrafter_pipeline import (  # noqa: E402
    GateConfig,
    PythonAPIFrameCrafterBackend,
    SyntheticFrameResult,
    TestOnlyBlendBackend,
    build_manifest,
    evaluate_candidate,
    frame_by_source_index,
    load_frames_csv,
    plan_framecrafter_generation_batches,
    plan_interpolated_targets,
    read_depth,
    read_rgb,
    save_depth_png,
    save_framecrafter_npz,
    save_rgb,
    select_scene_wide_targets,
    source_input_digest,
    synthetic_output_digest,
    targets_from_planner_json,
    validate_pose_source,
    write_manifest,
)
from src.framecrafter_advanced import (  # noqa: E402
    AdvancedPlannerConfig,
    build_role_aware_batches,
    load_evssm_resolver,
    plan_anchor_overlap_targets,
    select_advanced_scene_targets,
)
from src.framecrafter_context import ContextSelectionConfig  # noqa: E402
from src.framecrafter_sharding import (  # noqa: E402
    assigned_batch_ids,
    build_shard_runtime_identity,
    canonical_sha256,
    validate_global_plan_against_contract,
    validate_runtime_identity_against_contract,
    validate_shard_contract,
    write_shard_envelope,
)


PREPROCESS_SIGNATURE_SCHEMA = "unblur_slam.framecrafter_preprocess_signature.v2"

_SHARD_OPERATIONAL_PARAMETERS = frozenset(
    {
        "device",
        "vram_limit",
        "allow_test_only_backend",
        "plan_only",
        "shard_index",
        # These paths already have content identities under ``inputs``.
        "anchor_indices",
        "evssm_metadata",
    }
)

# Keep this list explicit: it is the cache-invalidation contract for every CLI
# option that can change planning, generation, depth fusion, or acceptance.  The
# input/model paths themselves are represented separately below with stronger
# content/stat identities.
_SIGNATURE_PARAMETER_NAMES = (
    "depth_scale",
    "output_depth_scale",
    "pose_convention",
    "pose_source",
    "fx",
    "fy",
    "cx",
    "cy",
    "laplacian_threshold",
    "blur_quantile",
    "translation_step",
    "rotation_step_deg",
    "blur_region_inserts",
    "max_inserts",
    "max_targets",
    "context_count",
    "min_contexts",
    "planner_mode",
    "anchor_indices",
    "only_gap_left",
    "only_gap_right",
    "target_pair_overlap",
    "hard_submap_overlap",
    "overlap_sample_stride",
    "include_blurry_regions",
    "feature_refinement",
    "feature_detector",
    "feature_model",
    "feature_ambiguity_low",
    "feature_ambiguity_high",
    "feature_overlap_weight",
    "feature_refine_rotation",
    "feature_min_inlier_ratio",
    "feature_max_rotation_correction_deg",
    "pnp_refinement",
    "pnp_detector",
    "pnp_max_features",
    "pnp_ratio_test",
    "pnp_mutual_check",
    "pnp_min_keypoints",
    "pnp_min_matches",
    "pnp_min_depth",
    "pnp_max_depth",
    "pnp_min_laplacian_variance",
    "pnp_ambiguity_low",
    "pnp_ambiguity_high",
    "pnp_ransac_reprojection_error_px",
    "pnp_ransac_confidence",
    "pnp_ransac_iterations",
    "pnp_min_inliers",
    "pnp_min_inlier_ratio",
    "pnp_max_reprojection_rmse_px",
    "pnp_max_rotation_correction_deg",
    "pnp_max_translation_correction",
    "local_blurry_contexts",
    "sharp_contexts",
    "context_local_radius",
    "context_search_radius",
    "min_sharp_context_overlap",
    "context_sharp_quantile",
    "context_image_mode",
    "hybrid_evssm_roles",
    "evssm_metadata",
    "evssm_min_confidence",
    "evssm_min_sharpness_gain",
    "evssm_min_consistency",
    "evssm_local_gate_enabled",
    "evssm_local_tile_size",
    "evssm_local_tile_stride",
    "evssm_local_max_brightness_drop",
    "evssm_local_min_edge_retention",
    "evssm_local_min_laplacian_retention",
    "evssm_local_max_tile_mae",
    "evssm_local_max_dark_expansion",
    "evssm_local_dark_luma_threshold",
    "evssm_local_min_raw_luma",
    "evssm_local_min_raw_edge",
    "evssm_local_min_raw_laplacian",
    "evssm_fallback",
    "acceptance_mode",
    "backend",
    "device",
    "vram_limit",
    "height",
    "width",
    "resize_mode",
    "num_inference_steps",
    "seed",
    "cfg_scale",
    "allow_test_only_backend",
    "plan_only",
    "min_sharpness_gain",
    "min_depth_coverage",
    "min_depth_consistency",
    "max_photometric_error",
    "max_reprojection_error_px",
    "min_reprojection_valid_ratio",
    "depth_abs_tolerance",
    "depth_rel_tolerance",
    "allow_missing_depth_gates",
    "shard_index",
)


def _resolved_path(value: Path | str | None) -> Path | None:
    if value is None or str(value) == "":
        return None
    return Path(value).expanduser().resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_file_identity(value: Path | str | None, *, required: bool) -> Any:
    """Return path plus content identity, deliberately independent of mtime."""

    path = _resolved_path(value)
    if path is None:
        if required:
            raise ValueError(
                "frames_csv is required to compute preprocessing signature"
            )
        return None
    if not path.is_file():
        raise FileNotFoundError(f"signature input file does not exist: {path}")
    return {
        "path": str(path),
        "size": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _path_identity(value: Path | str | None) -> str | None:
    path = _resolved_path(value)
    return None if path is None else str(path)


def _artifact_identity(value: Path | str | None) -> Any:
    """Identify a model/repository artifact by content, including large shards."""

    path = _resolved_path(value)
    if path is None:
        return None
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    identity = {
        "path": str(path),
        "exists": True,
        "kind": "directory" if path.is_dir() else "file",
        "size": int(stat.st_size),
    }
    if path.is_file():
        # Model identity is a scientific provenance boundary.  Size/mtime is
        # not sufficient: cp -p/rsync can replace bytes while preserving both.
        identity["sha256"] = _sha256_file(path)
        return identity

    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for child in sorted(path.rglob("*")):
        if not child.is_file() or any(
            part in {".git", "__pycache__"} for part in child.relative_to(path).parts
        ):
            continue
        child_stat = child.stat()
        relative = child.relative_to(path).as_posix()
        record = f"{relative}\0{child_stat.st_size}\0".encode("utf-8")
        digest.update(record)
        digest.update(_sha256_file(child).encode("ascii"))
        file_count += 1
        total_size += int(child_stat.st_size)
    identity.update(
        tree_sha256=digest.hexdigest(),
        file_count=file_count,
        total_file_bytes=total_size,
    )
    return identity


def _referenced_input_identities(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Hash every RGB/depth file that can affect planning or RGB-D gates."""

    frames = load_frames_csv(
        getattr(args, "frames_csv"),
        image_root=getattr(args, "image_root", None),
        depth_root=getattr(args, "depth_root", None),
        default_intrinsics=default_intrinsics(args),
        pose_convention=str(getattr(args, "pose_convention", "c2w")),
        compute_missing_sharpness=False,
        expected_pose_source=validate_pose_source(getattr(args, "pose_source", "")),
        require_pose_provenance=True,
    )
    identities = []
    for frame in frames:
        identities.append(
            {
                "source_index": int(frame.source_index),
                "rgb": _content_file_identity(frame.rgb_path, required=True),
                "depth": _content_file_identity(frame.depth_path, required=False),
            }
        )
    return identities


def preprocess_signature_payload(args: argparse.Namespace) -> dict[str, object]:
    """Build the stable, JSON-serializable preprocessing cache identity.

    CSV, referenced RGB-D inputs, external model assets, and the local pipeline
    implementation are content-addressed.  This intentionally reads large
    model shards once per preflight: stale generated observations are more
    expensive and scientifically unsafe than that I/O cost.
    """

    return {
        "schema": PREPROCESS_SIGNATURE_SCHEMA,
        "inputs": {
            "frames_csv": _content_file_identity(
                getattr(args, "frames_csv", None), required=True
            ),
            "planner_json": _content_file_identity(
                getattr(args, "planner_json", None), required=False
            ),
            "anchor_indices": _content_file_identity(
                getattr(args, "anchor_indices", None), required=False
            ),
            "evssm_metadata": _content_file_identity(
                getattr(args, "evssm_metadata", None), required=False
            ),
            "shard_contract": _content_file_identity(
                getattr(args, "shard_contract", None), required=False
            ),
            "image_root": _path_identity(getattr(args, "image_root", None)),
            "depth_root": _path_identity(getattr(args, "depth_root", None)),
            "referenced_frames": _referenced_input_identities(args),
        },
        "artifacts": {
            "framecrafter_repo": _artifact_identity(
                getattr(args, "framecrafter_repo", None)
            ),
            "checkpoint": _artifact_identity(getattr(args, "checkpoint", None)),
            "base_model_dir": _artifact_identity(getattr(args, "base_model_dir", None)),
        },
        "implementation": {
            "pipeline": _content_file_identity(
                ROOT / "src" / "framecrafter_pipeline.py", required=True
            ),
            "advanced_pipeline": _content_file_identity(
                ROOT / "src" / "framecrafter_advanced.py", required=True
            ),
            "overlap": _content_file_identity(
                ROOT / "src" / "framecrafter_overlap.py", required=True
            ),
            "pnp": _content_file_identity(
                ROOT / "src" / "framecrafter_pnp.py", required=True
            ),
            "context": _content_file_identity(
                ROOT / "src" / "framecrafter_context.py", required=True
            ),
            "sharding": _content_file_identity(
                ROOT / "src" / "framecrafter_sharding.py", required=True
            ),
            "preprocess": _content_file_identity(Path(__file__), required=True),
            "trajectory_export": _content_file_identity(
                ROOT / "scripts" / "export_framecrafter_trajectory.py",
                required=True,
            ),
            "dataset_adapter": _content_file_identity(
                ROOT / "src" / "utils" / "augmented_dataset.py", required=True
            ),
        },
        "parameters": {
            name: (
                str(getattr(args, name).expanduser().resolve())
                if isinstance(getattr(args, name, None), Path)
                else getattr(args, name, None)
            )
            for name in _SIGNATURE_PARAMETER_NAMES
        },
        # Generated manifest paths are absolute, so moving the preprocessing
        # directory is also a material change even when pixel inputs match.
        "output_dir": _path_identity(getattr(args, "output_dir", None)),
    }


def _preprocess_signature_from_payload(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def compute_preprocess_signature(args: argparse.Namespace) -> str:
    """Return the worker-local cache key used by preprocessing and run.py."""

    return _preprocess_signature_from_payload(preprocess_signature_payload(args))


def _content_only_identity(value: Any) -> Any:
    """Remove machine-local artifact paths while retaining their content hashes."""

    if isinstance(value, dict):
        return {
            str(key): _content_only_identity(item)
            for key, item in value.items()
            if str(key) != "path" and not str(key).endswith("_path")
        }
    if isinstance(value, list):
        return [_content_only_identity(item) for item in value]
    if isinstance(value, tuple):
        return [_content_only_identity(item) for item in value]
    return value


def shard_runtime_identity_from_signature_payload(
    payload: dict[str, object],
) -> dict[str, str]:
    """Project a worker-local cache payload onto one cross-worker identity.

    Device placement, output paths, the plan-only switch, and the deterministic
    worker index are operational.  Every source/model byte, semantic generation
    or gate parameter, and implementation byte remains bound.
    """

    inputs = dict(payload.get("inputs", {}))
    for key in ("shard_contract", "image_root", "depth_root"):
        inputs.pop(key, None)
    parameters = {
        str(key): value
        for key, value in dict(payload.get("parameters", {})).items()
        if str(key) not in _SHARD_OPERATIONAL_PARAMETERS
    }
    return build_shard_runtime_identity(
        source_identity=_content_only_identity(inputs),
        model_artifact_identity=_content_only_identity(
            payload.get("artifacts", {})
        ),
        semantic_config=parameters,
        implementation_identity=_content_only_identity(
            payload.get("implementation", {})
        ),
    )


def compute_shard_runtime_identity(args: argparse.Namespace) -> dict[str, str]:
    """Recompute the canonical scientific identity from actual worker args."""

    return shard_runtime_identity_from_signature_payload(
        preprocess_signature_payload(args)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pose-aware FrameCrafter preprocessing for Unblur-SLAM"
    )
    inputs = parser.add_argument_group("input trajectory and images")
    inputs.add_argument("--frames-csv", type=Path, required=True)
    inputs.add_argument("--planner-json", type=Path)
    inputs.add_argument("--image-root", type=Path)
    inputs.add_argument("--depth-root", type=Path)
    inputs.add_argument(
        "--depth-scale",
        type=float,
        required=True,
        help="source depth units per metre (for example 5000 for TUM PNG)",
    )
    inputs.add_argument(
        "--output-depth-scale",
        type=float,
        default=5000.0,
        help="uint16 PNG units per metre; 5000 matches the TUM config.",
    )
    inputs.add_argument("--pose-convention", choices=("c2w", "w2c"), default="c2w")
    inputs.add_argument(
        "--pose-source",
        default="droid_traj_est_not_align",
        help="Non-GT pose provenance written to the manifest.",
    )
    inputs.add_argument("--fx", type=float)
    inputs.add_argument("--fy", type=float)
    inputs.add_argument("--cx", type=float)
    inputs.add_argument("--cy", type=float)

    planning = parser.add_argument_group("direct planner (ignored with --planner-json)")
    planning.add_argument(
        "--planner-mode",
        choices=("legacy_pose_blur", "overlap_blur", "overlap_blur_feature"),
        default="legacy_pose_blur",
    )
    planning.add_argument(
        "--anchor-indices",
        type=Path,
        help="DROID video.npz timestamps or an increasing tracking-anchor index list.",
    )
    planning.add_argument("--only-gap-left", type=int)
    planning.add_argument("--only-gap-right", type=int)
    planning.add_argument("--target-pair-overlap", type=float, default=0.65)
    planning.add_argument("--hard-submap-overlap", type=float, default=0.05)
    planning.add_argument("--overlap-sample-stride", type=int, default=4)
    planning.add_argument(
        "--include-blurry-regions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    planning.add_argument("--feature-refinement", action="store_true")
    planning.add_argument("--feature-detector", choices=("orb", "sift"), default="orb")
    planning.add_argument(
        "--feature-model",
        choices=("homography", "fundamental", "essential"),
        default="essential",
    )
    planning.add_argument("--feature-ambiguity-low", type=float, default=0.15)
    planning.add_argument("--feature-ambiguity-high", type=float, default=0.75)
    planning.add_argument("--feature-overlap-weight", type=float, default=0.20)
    planning.add_argument("--feature-refine-rotation", action="store_true")
    planning.add_argument("--feature-min-inlier-ratio", type=float, default=0.35)
    planning.add_argument(
        "--feature-max-rotation-correction-deg", type=float, default=12.0
    )
    planning.add_argument(
        "--pnp-refinement",
        action="store_true",
        help=(
            "Use RAW RGB-D correspondences plus solvePnPRansac to refine an "
            "ambiguous anchor pair before recomputing overlap."
        ),
    )
    planning.add_argument("--pnp-detector", choices=("orb", "sift"), default="orb")
    planning.add_argument("--pnp-max-features", type=int, default=3000)
    planning.add_argument("--pnp-ratio-test", type=float, default=0.75)
    planning.add_argument(
        "--pnp-mutual-check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    planning.add_argument("--pnp-min-keypoints", type=int, default=12)
    planning.add_argument("--pnp-min-matches", type=int, default=8)
    planning.add_argument("--pnp-min-depth", type=float, default=1.0e-4)
    planning.add_argument("--pnp-max-depth", type=float, default=20.0)
    planning.add_argument("--pnp-min-laplacian-variance", type=float, default=0.0)
    planning.add_argument("--pnp-ambiguity-low", type=float, default=0.15)
    planning.add_argument("--pnp-ambiguity-high", type=float, default=0.75)
    planning.add_argument(
        "--pnp-ransac-reprojection-error-px", type=float, default=3.0
    )
    planning.add_argument("--pnp-ransac-confidence", type=float, default=0.999)
    planning.add_argument("--pnp-ransac-iterations", type=int, default=200)
    planning.add_argument("--pnp-min-inliers", type=int, default=8)
    planning.add_argument("--pnp-min-inlier-ratio", type=float, default=0.35)
    planning.add_argument(
        "--pnp-max-reprojection-rmse-px", type=float, default=2.0
    )
    planning.add_argument(
        "--pnp-max-rotation-correction-deg", type=float, default=12.0
    )
    planning.add_argument(
        "--pnp-max-translation-correction", type=float, default=0.25
    )
    planning.add_argument("--laplacian-threshold", type=float)
    planning.add_argument("--blur-quantile", type=float, default=0.30)
    planning.add_argument("--translation-step", type=float, default=0.08)
    planning.add_argument("--rotation-step-deg", type=float, default=6.0)
    planning.add_argument("--blur-region-inserts", type=int, default=1)
    planning.add_argument("--max-inserts", type=int, default=4)
    planning.add_argument(
        "--max-targets",
        type=int,
        default=256,
        help=(
            "scene-wide cap: retain large-pose-gap targets first, then uniformly "
            "sample remaining chronological candidates (never a prefix truncation)"
        ),
    )
    planning.add_argument("--context-count", type=int, default=6)
    planning.add_argument("--min-contexts", type=int, default=3)
    planning.add_argument("--local-blurry-contexts", type=int, default=2)
    planning.add_argument("--sharp-contexts", type=int, default=2)
    planning.add_argument("--context-local-radius", type=int, default=8)
    planning.add_argument("--context-search-radius", type=int, default=32)
    planning.add_argument("--min-sharp-context-overlap", type=float, default=0.25)
    planning.add_argument("--context-sharp-quantile", type=float, default=0.65)
    planning.add_argument(
        "--context-image-mode", choices=("raw", "evssm", "hybrid"), default="raw"
    )
    planning.add_argument(
        "--hybrid-evssm-roles",
        nargs="+",
        default=("local_blurry_inside",),
        help=(
            "Context roles allowed to use EVSSM in hybrid mode. The production "
            "default keeps endpoints and sharp guides RAW."
        ),
    )
    planning.add_argument("--evssm-metadata", type=Path)
    planning.add_argument("--evssm-min-confidence", type=float, default=0.50)
    planning.add_argument("--evssm-min-sharpness-gain", type=float, default=1.0)
    planning.add_argument("--evssm-min-consistency", type=float, default=0.70)
    planning.add_argument(
        "--evssm-local-gate-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    planning.add_argument("--evssm-local-tile-size", type=int, default=32)
    planning.add_argument("--evssm-local-tile-stride", type=int, default=16)
    planning.add_argument(
        "--evssm-local-max-brightness-drop", type=float, default=0.30
    )
    planning.add_argument(
        "--evssm-local-min-edge-retention", type=float, default=0.50
    )
    planning.add_argument(
        "--evssm-local-min-laplacian-retention", type=float, default=0.50
    )
    planning.add_argument("--evssm-local-max-tile-mae", type=float, default=0.20)
    planning.add_argument(
        "--evssm-local-max-dark-expansion", type=float, default=0.30
    )
    planning.add_argument(
        "--evssm-local-dark-luma-threshold", type=float, default=96.0 / 255.0
    )
    planning.add_argument("--evssm-local-min-raw-luma", type=float, default=0.10)
    planning.add_argument("--evssm-local-min-raw-edge", type=float, default=0.01)
    planning.add_argument(
        "--evssm-local-min-raw-laplacian", type=float, default=0.01
    )
    planning.add_argument("--evssm-fallback", choices=("error", "raw"), default="error")

    backend = parser.add_argument_group("generation backend")
    backend.add_argument(
        "--backend", choices=("python_api", "test_only_blend"), default="python_api"
    )
    backend.add_argument("--framecrafter-repo", type=Path)
    backend.add_argument("--checkpoint", type=Path)
    backend.add_argument("--base-model-dir", type=Path)
    backend.add_argument("--device", default="cuda:0")
    backend.add_argument("--vram-limit", type=float, default=20.0)
    backend.add_argument("--height", type=int, default=480)
    backend.add_argument("--width", type=int, default=832)
    backend.add_argument(
        "--resize-mode", choices=("stretch", "crop"), default="stretch"
    )
    backend.add_argument("--num-inference-steps", type=int, default=20)
    backend.add_argument("--seed", type=int, default=43)
    backend.add_argument("--cfg-scale", type=float, default=1.0)
    backend.add_argument("--allow-test-only-backend", action="store_true")
    backend.add_argument(
        "--plan-only",
        action="store_true",
        help="Write per-target official NPZ inputs without loading a generator.",
    )

    gates = parser.add_argument_group("candidate acceptance gates")
    gates.add_argument(
        "--acceptance-mode",
        choices=("sharp", "geometry"),
        default="sharp",
        help=(
            "sharp injects only sharp+geometry passing views; geometry also injects "
            "views that pass RGB-D/photo/reprojection gates but miss sharpness."
        ),
    )
    gates.add_argument("--min-sharpness-gain", type=float, default=1.05)
    gates.add_argument("--min-depth-coverage", type=float, default=0.30)
    gates.add_argument("--min-depth-consistency", type=float, default=0.50)
    gates.add_argument("--max-photometric-error", type=float, default=0.20)
    gates.add_argument("--max-reprojection-error-px", type=float, default=2.0)
    gates.add_argument("--min-reprojection-valid-ratio", type=float, default=0.05)
    gates.add_argument("--depth-abs-tolerance", type=float, default=0.03)
    gates.add_argument("--depth-rel-tolerance", type=float, default=0.03)
    gates.add_argument(
        "--allow-missing-depth-gates",
        action="store_true",
        help="RGB-only ablation: accept candidates without depth/reprojection gates.",
    )

    sharding = parser.add_argument_group("deterministic multi-machine generation")
    sharding.add_argument(
        "--shard-contract",
        type=Path,
        help="Immutable global plan contract shared verbatim by every worker.",
    )
    sharding.add_argument(
        "--shard-index",
        type=int,
        help="Zero-based deterministic worker index from --shard-contract.",
    )
    sharding.add_argument(
        "--shard-envelope",
        type=Path,
        help="Immutable output sidecar consumed by merge_framecrafter_shards.py.",
    )

    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def default_intrinsics(args: argparse.Namespace) -> np.ndarray | None:
    values = tuple(getattr(args, name, None) for name in ("fx", "fy", "cx", "cy"))
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("provide all of --fx/--fy/--cx/--cy, or none")
    return np.array(
        [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def make_backend(args: argparse.Namespace):
    if args.backend == "test_only_blend":
        return TestOnlyBlendBackend(allow_test_only=args.allow_test_only_backend)
    if args.framecrafter_repo is None or args.checkpoint is None:
        raise ValueError(
            "python_api backend requires --framecrafter-repo and --checkpoint; "
            "this script never downloads the 14B model"
        )
    return PythonAPIFrameCrafterBackend(
        repo_path=args.framecrafter_repo,
        checkpoint_path=args.checkpoint,
        device=args.device,
        vram_limit=args.vram_limit,
        base_model_dir=args.base_model_dir,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
    )


def validate_backend_artifacts(args: argparse.Namespace) -> None:
    """Reject missing production dependencies before creating output files."""

    if bool(getattr(args, "plan_only", False)):
        return
    if args.backend == "test_only_blend":
        if not bool(args.allow_test_only_backend):
            raise ValueError("test_only_blend requires --allow-test-only-backend")
        return
    if args.backend != "python_api":
        raise ValueError(f"unsupported FrameCrafter backend {args.backend!r}")
    if args.framecrafter_repo is None or args.checkpoint is None:
        raise ValueError(
            "python_api backend requires --framecrafter-repo and --checkpoint"
        )
    repo = Path(args.framecrafter_repo).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    base_model = (
        None
        if args.base_model_dir is None
        else Path(args.base_model_dir).expanduser().resolve()
    )
    if not (repo / "model.py").is_file():
        raise FileNotFoundError(
            f"FrameCrafter-compatible model.py not found: {repo / 'model.py'}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FrameCrafter checkpoint not found: {checkpoint}")
    if base_model is None or not base_model.is_dir():
        raise FileNotFoundError(
            "FrameCrafter base_model_dir must be an existing Wan2.1-I2V directory"
        )
    if not any(path.is_file() for path in base_model.rglob("*")):
        raise FileNotFoundError(
            "FrameCrafter base_model_dir is empty; provide the actual Wan2.1-I2V assets"
        )


def _validate_shard_cli(args: argparse.Namespace) -> bool:
    values = (
        getattr(args, "shard_contract", None),
        getattr(args, "shard_index", None),
        getattr(args, "shard_envelope", None),
    )
    enabled = any(value is not None for value in values)
    if enabled and not all(value is not None for value in values):
        raise ValueError(
            "--shard-contract, --shard-index and --shard-envelope must be used together"
        )
    if not enabled:
        return False
    if bool(getattr(args, "plan_only", False)):
        raise ValueError(
            "--plan-only cannot be combined with a worker shard; build the global "
            "plan first, create its contract, then launch production workers"
        )
    if str(getattr(args, "backend", "python_api")) != "python_api":
        raise ValueError("shard envelopes require the real python_api backend")
    return True


def _load_shard_contract(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    contract_path = Path(args.shard_contract).expanduser().resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(f"shard contract does not exist: {contract_path}")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"shard contract is not valid JSON: {contract_path}"
        ) from error
    if not isinstance(contract, dict):
        raise ValueError("shard contract root must be an object")
    validate_shard_contract(contract)
    # Also validates the requested index against shard_count before planning or
    # model construction starts.
    assigned_batch_ids(contract, int(args.shard_index))
    return contract_path, contract


def _batch_plan_record(
    batch: Any, context_report: dict[str, object]
) -> dict[str, object]:
    target_ids = [target.target_id for target in batch.targets]
    endpoint_positions = [
        position
        for target in batch.targets
        for position in (target.left_position, target.right_position)
    ]
    return {
        "batch_id": batch.batch_id,
        "target_ids": target_ids,
        "target_count": len(batch.targets),
        "context_source_indices": [frame.source_index for frame in batch.contexts],
        "context_ids": [frame.frame_id for frame in batch.contexts],
        "context_count": len(batch.contexts),
        "total_view_count": len(batch.contexts) + len(batch.targets),
        "endpoint_position_min": min(endpoint_positions),
        "endpoint_position_max": max(endpoint_positions),
        "endpoint_position_span": batch.endpoint_position_span,
        "max_endpoint_position_span": batch.max_endpoint_position_span,
        **context_report,
    }


def _planned_target_records(
    batch: Any, context_report: dict[str, object]
) -> list[dict[str, object]]:
    target_ids = [target.target_id for target in batch.targets]
    context_source_indices = [frame.source_index for frame in batch.contexts]
    context_ids = [frame.frame_id for frame in batch.contexts]
    return [
        {
            "target_id": target.target_id,
            "left_index": target.left_index,
            "right_index": target.right_index,
            "left_position": target.left_position,
            "right_position": target.right_position,
            "alpha": target.alpha,
            "reasons": list(target.reasons),
            "batch_id": batch.batch_id,
            "batch_target_ids": target_ids,
            "batch_target_position": batch_target_position,
            "batch_target_count": len(batch.targets),
            "context_source_indices": context_source_indices,
            "context_ids": context_ids,
            "batch_context_count": len(batch.contexts),
            "batch_policy": context_report.get(
                "batch_policy", "legacy_local_multi_gap_v1"
            ),
            "context_selection_policy": context_report.get(
                "context_selection_policy", "legacy_temporal_pose_v1"
            ),
            "conditioning": context_report.get("conditioning", []),
        }
        for batch_target_position, target in enumerate(batch.targets)
    ]


def _global_plan_records(
    batches: list[Any], context_reports: dict[str, dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    generation_batches: list[dict[str, object]] = []
    planned: list[dict[str, object]] = []
    for batch in batches:
        context_report = context_reports.get(batch.batch_id, {})
        generation_batches.append(_batch_plan_record(batch, context_report))
        planned.extend(_planned_target_records(batch, context_report))
    return generation_batches, planned


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def finite_metrics(metrics: dict[str, object]) -> dict[str, object]:
    """Convert accidental infinities to null before JSON serialization."""

    output: dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            output[key] = None
        elif isinstance(value, np.generic):
            output[key] = value.item()
        else:
            output[key] = value
    return output


def geometry_confidence(metrics: dict[str, object], args: argparse.Namespace) -> float:
    """Confidence from RGB-D/photo/reprojection terms, excluding sharpness."""

    def value(name: str, default: float = 0.0) -> float:
        raw = metrics.get(name)
        return default if raw is None else float(raw)

    terms = [
        np.clip(value("depth_coverage") / max(1.0e-12, args.min_depth_coverage), 0, 1),
        np.clip(
            value("depth_consistency") / max(1.0e-12, args.min_depth_consistency),
            0,
            1,
        ),
        np.clip(
            1.0
            - value("photometric_error", float("inf"))
            / max(1.0e-12, args.max_photometric_error),
            0,
            1,
        ),
        np.clip(
            1.0
            - value("reprojection_error_px", float("inf"))
            / max(1.0e-12, args.max_reprojection_error_px),
            0,
            1,
        ),
        np.clip(
            value("reprojection_valid_ratio")
            / max(1.0e-12, args.min_reprojection_valid_ratio),
            0,
            1,
        ),
    ]
    return float(np.clip(np.mean(terms), 0.0, 1.0))


def run_preprocess(
    args: argparse.Namespace, *, precomputed_signature: str | None = None
) -> dict[str, object]:
    """Run one exclusive preprocessing writer for an output directory."""

    shard_enabled = _validate_shard_cli(args)
    if shard_enabled:
        _load_shard_contract(args)
    validate_backend_artifacts(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".framecrafter_preprocess.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another FrameCrafter preprocessor is writing this output directory: "
                f"{output_dir}"
            ) from error
        try:
            return _run_preprocess_locked(
                args, precomputed_signature=precomputed_signature
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _run_preprocess_locked(
    args: argparse.Namespace, *, precomputed_signature: str | None = None
) -> dict[str, object]:
    """Implementation entered only while the per-output advisory lock is held."""

    signature_payload = preprocess_signature_payload(args)
    current_signature = _preprocess_signature_from_payload(signature_payload)
    shard_runtime_identity = shard_runtime_identity_from_signature_payload(
        signature_payload
    )
    if _validate_shard_cli(args):
        _, identity_contract = _load_shard_contract(args)
        # This is deliberately before CSV planning, NPZ publication, backend
        # construction, or any generation call.  A worker with different model
        # bytes, seed/steps/gates, source inputs, or implementation cannot start.
        validate_runtime_identity_against_contract(
            identity_contract, shard_runtime_identity
        )
    preprocess_signature = (
        current_signature
        if precomputed_signature is None
        else str(precomputed_signature)
    )
    if len(preprocess_signature) != 64 or any(
        character not in "0123456789abcdef" for character in preprocess_signature
    ):
        raise ValueError("precomputed FrameCrafter signature is not a SHA-256 digest")
    if preprocess_signature != current_signature:
        raise RuntimeError(
            "FrameCrafter source/model/code changed after cache preflight; "
            "refusing to generate under a stale signature"
        )
    pose_source = validate_pose_source(args.pose_source)
    output_dir = args.output_dir.expanduser().resolve()
    generation_id = uuid.uuid4().hex
    artifact_dir = output_dir / "artifacts" / preprocess_signature / generation_id
    frames = load_frames_csv(
        args.frames_csv,
        image_root=args.image_root,
        depth_root=args.depth_root,
        default_intrinsics=default_intrinsics(args),
        pose_convention=args.pose_convention,
        expected_pose_source=pose_source,
        require_pose_provenance=True,
    )
    planner_mode = str(getattr(args, "planner_mode", "legacy_pose_blur"))
    advanced_planning = None
    if args.planner_json is not None:
        targets = targets_from_planner_json(args.planner_json, frames)
        target_selection_policy = "scene_wide_large_pose_gap_priority_then_uniform_v1"
    elif planner_mode in {"overlap_blur", "overlap_blur_feature"}:
        anchor_path = getattr(args, "anchor_indices", None)
        if anchor_path is None:
            raise ValueError(
                f"planner_mode={planner_mode} requires --anchor-indices from the "
                "first-pass DROID tracking stream"
            )
        only_left = getattr(args, "only_gap_left", None)
        only_right = getattr(args, "only_gap_right", None)
        if (only_left is None) != (only_right is None):
            raise ValueError(
                "--only-gap-left and --only-gap-right must be used together"
            )
        only_gap = None if only_left is None else (int(only_left), int(only_right))
        advanced_planning = plan_anchor_overlap_targets(
            frames,
            anchor_path,
            depth_scale=args.depth_scale,
            config=AdvancedPlannerConfig(
                target_pair_overlap=float(getattr(args, "target_pair_overlap", 0.65)),
                hard_submap_overlap=float(getattr(args, "hard_submap_overlap", 0.05)),
                max_inserts=int(args.max_inserts),
                sample_stride=int(getattr(args, "overlap_sample_stride", 4)),
                depth_abs_tolerance=float(args.depth_abs_tolerance),
                depth_rel_tolerance=float(args.depth_rel_tolerance),
                include_blurry_regions=bool(
                    getattr(args, "include_blurry_regions", True)
                ),
                blur_quantile=float(args.blur_quantile),
                laplacian_threshold=args.laplacian_threshold,
                blur_region_inserts=int(args.blur_region_inserts),
                feature_refinement=bool(
                    planner_mode == "overlap_blur_feature"
                    or getattr(args, "feature_refinement", False)
                ),
                feature_detector=str(getattr(args, "feature_detector", "orb")),
                feature_model=str(getattr(args, "feature_model", "essential")),
                feature_ambiguity_low=float(
                    getattr(args, "feature_ambiguity_low", 0.15)
                ),
                feature_ambiguity_high=float(
                    getattr(args, "feature_ambiguity_high", 0.75)
                ),
                feature_overlap_weight=float(
                    getattr(args, "feature_overlap_weight", 0.20)
                ),
                feature_refine_rotation=bool(
                    getattr(args, "feature_refine_rotation", False)
                ),
                feature_min_inlier_ratio=float(
                    getattr(args, "feature_min_inlier_ratio", 0.35)
                ),
                feature_max_rotation_correction_deg=float(
                    getattr(args, "feature_max_rotation_correction_deg", 12.0)
                ),
                pnp_refinement=bool(getattr(args, "pnp_refinement", False)),
                pnp_detector=str(getattr(args, "pnp_detector", "orb")),
                pnp_max_features=int(getattr(args, "pnp_max_features", 3000)),
                pnp_ratio_test=float(getattr(args, "pnp_ratio_test", 0.75)),
                pnp_mutual_check=bool(getattr(args, "pnp_mutual_check", True)),
                pnp_min_keypoints=int(getattr(args, "pnp_min_keypoints", 12)),
                pnp_min_matches=int(getattr(args, "pnp_min_matches", 8)),
                pnp_min_depth=float(getattr(args, "pnp_min_depth", 1.0e-4)),
                pnp_max_depth=float(getattr(args, "pnp_max_depth", 20.0)),
                pnp_min_laplacian_variance=float(
                    getattr(args, "pnp_min_laplacian_variance", 0.0)
                ),
                pnp_ambiguity_low=float(
                    getattr(args, "pnp_ambiguity_low", 0.15)
                ),
                pnp_ambiguity_high=float(
                    getattr(args, "pnp_ambiguity_high", 0.75)
                ),
                pnp_ransac_reprojection_error_px=float(
                    getattr(args, "pnp_ransac_reprojection_error_px", 3.0)
                ),
                pnp_ransac_confidence=float(
                    getattr(args, "pnp_ransac_confidence", 0.999)
                ),
                pnp_ransac_iterations=int(
                    getattr(args, "pnp_ransac_iterations", 200)
                ),
                pnp_min_inliers=int(getattr(args, "pnp_min_inliers", 8)),
                pnp_min_inlier_ratio=float(
                    getattr(args, "pnp_min_inlier_ratio", 0.35)
                ),
                pnp_max_reprojection_rmse_px=float(
                    getattr(args, "pnp_max_reprojection_rmse_px", 2.0)
                ),
                pnp_max_rotation_correction_deg=float(
                    getattr(args, "pnp_max_rotation_correction_deg", 12.0)
                ),
                pnp_max_translation_correction=float(
                    getattr(args, "pnp_max_translation_correction", 0.25)
                ),
            ),
            only_gap=only_gap,
        )
        targets = list(advanced_planning.targets)
        target_selection_policy = "scene_wide_overlap_priority_then_uniform_v1"
    else:
        targets = plan_interpolated_targets(
            frames,
            laplacian_threshold=args.laplacian_threshold,
            blur_quantile=args.blur_quantile,
            translation_step=args.translation_step,
            rotation_step_deg=args.rotation_step_deg,
            blur_region_inserts=args.blur_region_inserts,
            max_inserts=args.max_inserts,
        )
        target_selection_policy = "scene_wide_large_pose_gap_priority_then_uniform_v1"
    planned_total_before_cap = len(targets)
    targets = (
        select_advanced_scene_targets(targets, args.max_targets)
        if advanced_planning is not None
        else select_scene_wide_targets(targets, args.max_targets)
    )
    global_selected_target_count = len(targets)

    if args.allow_missing_depth_gates and not args.plan_only:
        raise ValueError(
            "RGB-only FrameCrafter candidates cannot be injected into the current "
            "RGB-D mapper; provide source depth and keep depth gates enabled"
        )
    if not args.plan_only:
        required_depth_indices = {
            index
            for target in targets
            for index in (target.left_index, target.right_index)
        }
        missing_depth = sorted(
            index
            for index in required_depth_indices
            if frame_by_source_index(frames, index).depth_path is None
        )
        if missing_depth:
            raise ValueError(
                "source depth is required before expensive FrameCrafter generation; "
                f"missing source indices: {missing_depth}"
            )

    context_reports: dict[str, dict[str, object]] = {}
    if advanced_planning is not None:
        image_mode = str(getattr(args, "context_image_mode", "raw"))
        evssm_metadata = getattr(args, "evssm_metadata", None)
        if image_mode in {"evssm", "hybrid"} and evssm_metadata is None:
            raise ValueError(
                f"context_image_mode={image_mode} requires --evssm-metadata; "
                "run precompute_framecrafter_evssm.py first"
            )
        evssm_resolver, evssm_records = load_evssm_resolver(
            evssm_metadata,
            require_production=not bool(
                getattr(args, "allow_test_only_backend", False)
            ),
        )
        batches, context_reports = build_role_aware_batches(
            frames,
            targets,
            context_config=ContextSelectionConfig(
                context_budget=int(args.context_count),
                min_contexts=int(args.min_contexts),
                local_blurry_count=int(getattr(args, "local_blurry_contexts", 2)),
                sharp_context_count=int(getattr(args, "sharp_contexts", 2)),
                local_radius=int(getattr(args, "context_local_radius", 8)),
                min_sharp_overlap=float(
                    getattr(args, "min_sharp_context_overlap", 0.25)
                ),
                blur_quantile=float(args.blur_quantile),
                sharp_quantile=float(getattr(args, "context_sharp_quantile", 0.65)),
                seed=int(args.seed),
                image_mode=image_mode,
                evssm_min_confidence=float(getattr(args, "evssm_min_confidence", 0.50)),
                evssm_min_sharpness_gain=float(
                    getattr(args, "evssm_min_sharpness_gain", 1.0)
                ),
                evssm_min_consistency=float(
                    getattr(args, "evssm_min_consistency", 0.70)
                ),
                hybrid_evssm_roles=tuple(
                    getattr(args, "hybrid_evssm_roles", ("local_blurry_inside",))
                ),
                evssm_local_gate_enabled=bool(
                    getattr(args, "evssm_local_gate_enabled", True)
                ),
                evssm_local_tile_size=int(
                    getattr(args, "evssm_local_tile_size", 32)
                ),
                evssm_local_tile_stride=int(
                    getattr(args, "evssm_local_tile_stride", 16)
                ),
                evssm_local_max_brightness_drop=float(
                    getattr(args, "evssm_local_max_brightness_drop", 0.30)
                ),
                evssm_local_min_edge_retention=float(
                    getattr(args, "evssm_local_min_edge_retention", 0.50)
                ),
                evssm_local_min_laplacian_retention=float(
                    getattr(args, "evssm_local_min_laplacian_retention", 0.50)
                ),
                evssm_local_max_tile_mae=float(
                    getattr(args, "evssm_local_max_tile_mae", 0.20)
                ),
                evssm_local_max_dark_expansion=float(
                    getattr(args, "evssm_local_max_dark_expansion", 0.30)
                ),
                evssm_local_dark_luma_threshold=float(
                    getattr(args, "evssm_local_dark_luma_threshold", 96.0 / 255.0)
                ),
                evssm_local_min_raw_luma=float(
                    getattr(args, "evssm_local_min_raw_luma", 0.10)
                ),
                evssm_local_min_raw_edge=float(
                    getattr(args, "evssm_local_min_raw_edge", 0.01)
                ),
                evssm_local_min_raw_laplacian=float(
                    getattr(args, "evssm_local_min_raw_laplacian", 0.01)
                ),
            ),
            depth_scale=args.depth_scale,
            context_search_radius=int(getattr(args, "context_search_radius", 32)),
            evssm_resolver=evssm_resolver,
            evssm_records=evssm_records,
        )
        fallbacks = [
            item
            for report in context_reports.values()
            for item in report["conditioning"]
            if item.get("fallback_reason")
        ]
        if fallbacks and str(getattr(args, "evssm_fallback", "error")) == "error":
            first = fallbacks[0]
            raise ValueError(
                "EVSSM conditioning fell back to RAW while evssm_fallback=error: "
                f"source={first.get('source_index')} reason={first.get('fallback_reason')}"
            )
    else:
        batches = plan_framecrafter_generation_batches(
            frames,
            targets,
            context_count=args.context_count,
            min_contexts=args.min_contexts,
        )

    global_batches = list(batches)
    global_generation_batches, global_planned = _global_plan_records(
        global_batches, context_reports
    )
    global_generation_batch_count = len(global_batches)
    shard_provenance: dict[str, object] | None = None
    if _validate_shard_cli(args):
        contract_path, shard_contract = _load_shard_contract(args)
        validate_global_plan_against_contract(
            shard_contract,
            global_generation_batches,
            global_planned,
        )
        shard_index = int(args.shard_index)
        assigned_ids = assigned_batch_ids(shard_contract, shard_index)
        assigned_set = set(assigned_ids)
        batches = [batch for batch in global_batches if batch.batch_id in assigned_set]
        if {batch.batch_id for batch in batches} != assigned_set:
            raise ValueError(
                "shard contract assigns generation batches absent from worker global plan"
            )
        local_target_ids = {
            target.target_id for batch in batches for target in batch.targets
        }
        targets = [target for target in targets if target.target_id in local_target_ids]
        context_reports = {
            batch_id: report
            for batch_id, report in context_reports.items()
            if batch_id in assigned_set
        }
        shard_provenance = {
            "schema": "unblur_slam.framecrafter_worker_shard.v1",
            "contract_path": str(contract_path),
            "contract_sha256": _sha256_file(contract_path),
            "experiment_signature": shard_contract["experiment_signature"],
            "canonical_preprocess_signature": shard_contract[
                "canonical_preprocess_signature"
            ],
            "runtime_identity_sha256": canonical_sha256(
                shard_runtime_identity
            ),
            "assignment_table_sha256": shard_contract["assignment_table_sha256"],
            "global_plan_sha256": shard_contract["global_plan_sha256"],
            "shard_index": shard_index,
            "shard_count": int(shard_contract["shard_count"]),
            "assigned_batch_ids": list(assigned_ids),
            "global_counts": {
                "planned_total_before_cap": planned_total_before_cap,
                "selected_target_count": global_selected_target_count,
                "generation_batch_count": global_generation_batch_count,
            },
            "local_counts": {
                "selected_target_count": len(targets),
                "generation_batch_count": len(batches),
            },
        }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    planned: list[dict[str, object]] = []
    generation_batches: list[dict[str, object]] = []
    for batch in batches:
        batch_context_report = context_reports.get(batch.batch_id, {})
        batch_record = _batch_plan_record(batch, batch_context_report)
        batch_planned = _planned_target_records(batch, batch_context_report)
        npz_path = save_framecrafter_npz(
            artifact_dir / "framecrafter_npz" / f"{batch.batch_id}.npz",
            batch.contexts,
            batch.targets,
        )
        poses_npz_sha256 = _sha256_file(npz_path)
        batch_record.update(poses_npz=str(npz_path), poses_npz_sha256=poses_npz_sha256)
        generation_batches.append(batch_record)
        for record in batch_planned:
            record.update(poses_npz=str(npz_path), poses_npz_sha256=poses_npz_sha256)
            planned.append(record)

    accepted: list[SyntheticFrameResult] = []
    accepted_report: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    sharp_accepted_report: list[dict[str, object]] = []
    geometry_only_report: list[dict[str, object]] = []
    geometry_rejected_report: list[dict[str, object]] = []
    acceptance_mode = str(getattr(args, "acceptance_mode", "sharp"))
    empty_production_shard = shard_provenance is not None and not batches
    backend = None if args.plan_only or empty_production_shard else make_backend(args)
    gate_config = GateConfig(
        min_sharpness_gain=args.min_sharpness_gain,
        min_depth_coverage=args.min_depth_coverage,
        min_depth_consistency=args.min_depth_consistency,
        max_photometric_error=args.max_photometric_error,
        max_reprojection_error_px=args.max_reprojection_error_px,
        min_reprojection_valid_ratio=args.min_reprojection_valid_ratio,
        depth_abs_tolerance=args.depth_abs_tolerance,
        depth_rel_tolerance=args.depth_rel_tolerance,
        require_depth=not args.allow_missing_depth_gates,
    )
    frame_position_by_source = {
        frame.source_index: position for position, frame in enumerate(frames)
    }

    for batch in batches:
        if args.plan_only:
            continue
        restore_size_hws: list[tuple[int, int]] = []
        endpoint_frames = []
        for target in batch.targets:
            left = frame_by_source_index(frames, target.left_index)
            right = frame_by_source_index(frames, target.right_index)
            left_shape = read_rgb(left.rgb_path).shape[:2]
            right_shape = read_rgb(right.rgb_path).shape[:2]
            if left_shape != right_shape:
                raise ValueError(
                    "bracketing frames must share a resolution: "
                    f"{left_shape} vs {right_shape}"
                )
            restore_size_hws.append(left_shape)
            endpoint_frames.append((left, right))
        generated_batch = backend.generate_many(
            batch.contexts,
            batch.targets,
            height=args.height,
            width=args.width,
            resize_mode=args.resize_mode,
            restore_size_hws=restore_size_hws,
        )
        if len(generated_batch) != len(batch.targets):
            raise RuntimeError(
                f"backend returned {len(generated_batch)} outputs for "
                f"{len(batch.targets)} targets in {batch.batch_id}"
            )
        batch_target_ids = tuple(target.target_id for target in batch.targets)
        for batch_target_position, (target, generated, endpoints) in enumerate(
            zip(batch.targets, generated_batch, endpoint_frames)
        ):
            left, right = endpoints
            gate_left, gate_right = left, right
            if advanced_planning is not None:
                left_position = frame_position_by_source[target.left_index]
                right_position = frame_position_by_source[target.right_index]
                local_timestamps = np.asarray(
                    [
                        frame.timestamp
                        for frame in frames[left_position : right_position + 1]
                    ],
                    dtype=np.float64,
                )
                relative_right = int(
                    np.searchsorted(local_timestamps, target.timestamp, side="right")
                )
                support_right_position = min(
                    right_position, left_position + max(1, relative_right)
                )
                support_left_position = max(left_position, support_right_position - 1)
                gate_left = frames[support_left_position]
                gate_right = frames[support_right_position]
            left_depth = (
                read_depth(gate_left.depth_path, args.depth_scale)
                if gate_left.depth_path
                else None
            )
            right_depth = (
                read_depth(gate_right.depth_path, args.depth_scale)
                if gate_right.depth_path
                else None
            )
            gate = evaluate_candidate(
                generated,
                gate_left,
                gate_right,
                target,
                left_depth=left_depth,
                right_depth=right_depth,
                config=gate_config,
            )
            metrics = finite_metrics(gate.metrics)
            batch_provenance: dict[str, object] = {
                "batch_id": batch.batch_id,
                "batch_target_ids": list(batch_target_ids),
                "batch_target_position": batch_target_position,
                "context_source_indices": [
                    frame.source_index for frame in batch.contexts
                ],
                "context_ids": [frame.frame_id for frame in batch.contexts],
                "gate_support_source_indices": [
                    gate_left.source_index,
                    gate_right.source_index,
                ],
            }
            geometry_failures = [
                failure for failure in gate.failures if failure != "sharpness_gain"
            ]
            sharp_failures = [
                failure for failure in gate.failures if failure == "sharpness_gain"
            ]
            acceptance_class = (
                "sharp_accepted"
                if not geometry_failures and not sharp_failures
                else "geometry_only"
                if not geometry_failures
                else "rejected"
            )
            inject_candidate = acceptance_class == "sharp_accepted" or (
                acceptance_class == "geometry_only" and acceptance_mode == "geometry"
            )
            if not inject_candidate:
                directory = (
                    "geometry_only_rgb"
                    if acceptance_class == "geometry_only"
                    else "rejected_rgb"
                )
                rejected_rgb = save_rgb(
                    artifact_dir / directory / f"{target.target_id}.png",
                    generated,
                )
                rejected_record = {
                    "target_id": target.target_id,
                    **batch_provenance,
                    "acceptance_class": acceptance_class,
                    "failures": list(gate.failures),
                    "geometry_failures": geometry_failures,
                    "sharp_failures": sharp_failures,
                    "metrics": metrics,
                    "candidate_rgb_path": str(rejected_rgb),
                    "candidate_rgb_sha256": _sha256_file(rejected_rgb),
                }
                rejected.append(rejected_record)
                if acceptance_class == "geometry_only":
                    geometry_only_report.append(dict(rejected_record))
                else:
                    geometry_rejected_report.append(dict(rejected_record))
                continue

            rgb_path = save_rgb(
                artifact_dir
                / (
                    "rgb"
                    if acceptance_class == "sharp_accepted"
                    else "geometry_only_rgb"
                )
                / f"{target.target_id}.png",
                generated,
            )
            depth_path = None
            if gate.fused_depth is not None:
                # Invalid pixels remain zero; the downstream loader can reconstruct
                # the validity mask as depth > 0.
                fused_depth = np.where(
                    gate.fused_depth_valid, gate.fused_depth, 0.0
                ).astype(np.float32)
                depth_path = save_depth_png(
                    artifact_dir / "depth" / f"{target.target_id}.png",
                    fused_depth,
                    depth_scale=args.output_depth_scale,
                )
            selected_confidence = (
                gate.confidence
                if acceptance_class == "sharp_accepted"
                else geometry_confidence(metrics, args)
            )
            accepted.append(
                SyntheticFrameResult(
                    target=target,
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    confidence=selected_confidence,
                    source_ids=tuple(frame.frame_id for frame in batch.contexts),
                    gate_metrics=metrics,
                    batch_id=batch.batch_id,
                    batch_target_ids=batch_target_ids,
                    batch_target_position=batch_target_position,
                    acceptance_class=acceptance_class,
                )
            )
            accepted_record = {
                "target_id": target.target_id,
                **batch_provenance,
                "acceptance_class": acceptance_class,
                "confidence": selected_confidence,
                "raw_gate_confidence": gate.confidence,
                "geometry_failures": geometry_failures,
                "sharp_failures": sharp_failures,
                "metrics": metrics,
                "rgb_path": str(rgb_path),
                "depth_path": None if depth_path is None else str(depth_path),
            }
            accepted_report.append(accepted_record)
            if acceptance_class == "sharp_accepted":
                sharp_accepted_report.append(dict(accepted_record))
            else:
                geometry_only_report.append(dict(accepted_record))

    manifest_path = (
        output_dir / f"manifest_{preprocess_signature}_{generation_id}.json"
    ).resolve()
    backend_name = (
        "python_api"
        if empty_production_shard
        else "plan_only"
        if backend is None
        else backend.backend_name
    )
    backend_test_only = False if backend is None else bool(backend.test_only)
    manifest = build_manifest(frames, accepted, pose_source=pose_source)
    for entry in manifest["frames"]:
        entry["rgb_sha256"] = _sha256_file(Path(entry["rgb_path"]))
        entry["depth_sha256"] = (
            None
            if entry.get("depth_path") in (None, "")
            else _sha256_file(Path(entry["depth_path"]))
        )
    accepted_output_sha256 = synthetic_output_digest(manifest["frames"])
    source_input_sha256 = source_input_digest(manifest["frames"])
    report = {
        "schema": "unblur_slam.framecrafter_preprocess_report.v1",
        "backend": backend_name,
        "backend_test_only": backend_test_only,
        "uses_ground_truth_pose": False,
        "pose_source": pose_source,
        "preprocess_signature": preprocess_signature,
        "shard_runtime_identity": shard_runtime_identity,
        "generation_id": generation_id,
        "source_frame_count": len(frames),
        "planned_total_before_cap": planned_total_before_cap,
        "planned_target_count": len(targets),
        "selected_target_count": len(targets),
        "target_selection_policy": target_selection_policy,
        "planner_mode": planner_mode,
        "acceptance_mode": acceptance_mode,
        "max_targets": args.max_targets,
        "generation_batch_count": len(batches),
        "backend_generate_call_count": (
            0 if backend is None else int(backend.generate_call_count)
        ),
        "accepted_target_count": len(accepted),
        "rejected_target_count": len(rejected),
        "sharp_accepted_target_count": len(sharp_accepted_report),
        "geometry_only_target_count": len(geometry_only_report),
        "geometry_rejected_target_count": len(geometry_rejected_report),
        "accepted_output_sha256": accepted_output_sha256,
        "source_input_sha256": source_input_sha256,
        "manifest": str(manifest_path),
        "generation_batches": generation_batches,
        "planned": planned,
        "accepted": accepted_report,
        "rejected": rejected,
        "quality_partition": {
            "sharp_accepted": sharp_accepted_report,
            "geometry_only": geometry_only_report,
            "rejected": geometry_rejected_report,
        },
        "overlap_planning": (
            None
            if advanced_planning is None
            else {
                "anchor_source_indices": list(advanced_planning.anchor_source_indices),
                "anchor_pair_count": len(advanced_planning.pairs),
                "blur_threshold": advanced_planning.blur_threshold,
                "sparse_target_count": advanced_planning.sparse_target_count,
                "blurry_target_count": advanced_planning.blurry_target_count,
                "submap_boundaries": [
                    list(value) for value in advanced_planning.submap_boundaries
                ],
                "pairs": [value.as_dict() for value in advanced_planning.pairs],
            }
        ),
    }
    if shard_provenance is not None:
        report["shard"] = shard_provenance
    report_path = output_dir / (
        f"preprocess_report_{preprocess_signature}_{generation_id}.json"
    )
    _write_json_atomic(report_path, report)
    manifest.update(
        preprocess_signature=preprocess_signature,
        generation_id=generation_id,
        backend=backend_name,
        backend_test_only=backend_test_only,
        accepted_output_sha256=accepted_output_sha256,
        source_input_sha256=source_input_sha256,
        preprocess_report_path=str(report_path.resolve()),
        preprocess_report_sha256=_sha256_file(report_path),
    )
    write_manifest(manifest_path, manifest)
    shard_envelope = None
    if shard_provenance is not None:
        _, shard_contract = _load_shard_contract(args)
        shard_envelope = write_shard_envelope(
            shard_contract,
            shard_index=int(args.shard_index),
            report_path=report_path,
            manifest_path=manifest_path,
            output_path=Path(args.shard_envelope),
        )
    summary = {
        key: report[key]
        for key in (
            "backend",
            "source_frame_count",
            "planned_target_count",
            "accepted_target_count",
            "rejected_target_count",
            "manifest",
        )
    }
    if shard_envelope is not None:
        summary["shard_envelope"] = str(shard_envelope)
    print(json.dumps(summary))
    return report


def main() -> None:
    run_preprocess(parse_args())


if __name__ == "__main__":
    main()
