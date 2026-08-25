#!/usr/bin/env python3
"""Preflight or run revision-v2 fixed-11KF EVSSM/ReSplat-state3 fusion pair.

The treatment performs at most one synchronous official cvg/ReSplat call,
after source 166 has become the eighth actually mapped keyframe.  Sources 0
and 9 in the frozen schedule are skipped for invalid depth, while source 220
remains as one downstream online mapping update after fusion.  The subprocess,
artifact checks, trial merge, post-merge gate, and any rollback all execute
inside Unblur-SLAM's online timer.  Importing this module and its default
preflight path do not create a CUDA context.

Revision v2 captures the just-mapped camera before the pose-window optimizer
rebinds its local ``viewpoint`` variable.  It writes only to a new ``_v2``
root; the fail-closed v1 outputs are preserved and never reused as run input.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/unblur_slam.yaml"
FIXED_SCRIPT = REPO_ROOT / "scripts/run_fr2_fixed_kf_frontend_ablation_221.py"
_SPEC = importlib.util.spec_from_file_location("fixed_kf_frontend_base", FIXED_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
FIXED = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = FIXED
_SPEC.loader.exec_module(FIXED)
BASE = FIXED.BASE

EXPECTED_FIXED_SOURCE_KEYFRAMES = FIXED.EXPECTED_FIXED_SOURCE_KEYFRAMES
EXPECTED_FIRST_EIGHT_MAPPED_SOURCES = (15, 49, 58, 72, 89, 109, 125, 166)
EXPECTED_DOWNSTREAM_MAPPED_SOURCES = (220,)
FROZEN_BASELINE_VIDEO = FIXED.FROZEN_BASELINE_VIDEO
FROZEN_BASELINE_VIDEO_SHA256 = FIXED.FROZEN_BASELINE_VIDEO_SHA256
PHYSICAL_GPU = FIXED.PHYSICAL_GPU
EXPECTED_RESPLAT_CHECKPOINT_SHA256 = (
    "548993fede0d9536d2d914cbe51e0ebea0ad6f88c898c909e02127d59bb2be9a"
)
EXPECTED_RESPLAT_COMMIT = "cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1"
CONFIGS = {
    "baseline": REPO_ROOT
    / "configs/local/fr2_xyz_fixed_kf_resplat3_fusion_221_v2/evssm_no_fusion.yaml",
    "fused": REPO_ROOT
    / "configs/local/fr2_xyz_fixed_kf_resplat3_fusion_221_v2/evssm_resplat3_active_fusion.yaml",
}
OUTPUT_ROOT = Path(
    "/srv/szha0669/unblur-slam/slam_paired/fr2_xyz_fixed_kf_resplat3_fusion_221_v2"
).resolve()
SUPERSEDED_V1_OUTPUT_ROOT = Path(
    "/srv/szha0669/unblur-slam/slam_paired/fr2_xyz_fixed_kf_resplat3_fusion_221"
).resolve()
OUTPUTS = {
    "baseline": (OUTPUT_ROOT / "evssm_no_fusion").resolve(),
    "fused": (OUTPUT_ROOT / "evssm_resplat3_active_fusion").resolve(),
}
ALLOWED_PAIR_DIFFS = {
    "data.output",
    "mapping.official_resplat_active_fusion.enabled",
}
GPU_MONITOR_INTERVAL_MS = 250
REAL_BRIDGE_PROBE_ROOT = Path(
    "/srv/szha0669/unblur-slam/official_resplat_sidecar_smoke/"
    "fr2_xyz_motion_only_first_closed8_turtle_gopro_small8v_v4"
)
REAL_BRIDGE_PROBE_SNAPSHOT = REAL_BRIDGE_PROBE_ROOT / (
    "snapshots/submap-0000-d753b6a3be957e25/snapshot_manifest.json"
)
REAL_BRIDGE_PROBE_NATIVE = REAL_BRIDGE_PROBE_ROOT / (
    "published/submap-0000-d753b6a3be957e25/native_gaussians_local.npz"
)
REAL_BRIDGE_PROBE_NATIVE_SHA256 = (
    "9d2d328b45a80159cd64f719f26c6cd15b1625c6b8ac4602acb20e3f3b2e7480"
)
CODE_PROVENANCE_FILES = {
    "experiment_runner": Path(__file__).resolve(),
    "mapper": REPO_ROOT / "src/mapper.py",
    "active_fusion_helper": REPO_ROOT
    / "src/refinement/official_resplat_active_fusion.py",
    "active_map_merge": REPO_ROOT / "src/refinement/active_map_merge.py",
    "world_bridge": REPO_ROOT / "src/refinement/resplat_unblur_bridge.py",
    "sidecar_runner": REPO_ROOT / "scripts/run_official_resplat_sidecar.py",
    "sidecar_verifier": REPO_ROOT / "src/refinement/official_resplat_sidecar.py",
    "gaussian_model": REPO_ROOT
    / "thirdparty/gaussian_splatting/scene/gaussian_model.py",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_provenance() -> dict[str, Any]:
    files = {}
    for label, path in CODE_PROVENANCE_FILES.items():
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"implementation file is missing: {resolved}")
        files[label] = {"path": str(resolved), "sha256": _sha256_file(resolved)}
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status_lines = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return {
        "schema": "unblur_slam.implementation_provenance.v1",
        "git_head": head,
        "git_worktree_clean": not status_lines,
        "git_dirty_entry_count": len(status_lines),
        "git_dirty_disclosure": "dirty_or_untracked_files_present"
        if status_lines
        else "clean",
        "files": files,
    }


def validate_mapper_hook_binding(mapper_source: Optional[str] = None) -> dict[str, Any]:
    """AST-audit the v2 current-camera capture around the pose-window loop."""

    mapper_path = REPO_ROOT / "src/mapper.py"
    source = (
        mapper_path.read_text(encoding="utf-8")
        if mapper_source is None
        else str(mapper_source)
    )
    tree = ast.parse(source, filename=str(mapper_path))
    mapper_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Mapper"
        ),
        None,
    )
    if mapper_class is None:
        raise ValueError("Mapper class is missing from mapper.py")
    run_method = next(
        (
            node
            for node in mapper_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
        ),
        None,
    )
    if run_method is None:
        raise ValueError("Mapper.run is missing")

    captures = []
    viewpoint_rebinds = []
    fusion_hooks = []
    prune_calls = []
    for node in ast.walk(run_method):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            if names == ["mapped_viewpoint"] and isinstance(value, ast.Name):
                if value.id == "viewpoint":
                    captures.append(node)
            if "viewpoint" in names and isinstance(value, ast.Subscript):
                if (
                    isinstance(value.value, ast.Attribute)
                    and isinstance(value.value.value, ast.Name)
                    and value.value.value.id == "self"
                    and value.value.attr == "cameras"
                ):
                    viewpoint_rebinds.append(node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "_active_fusion_after_mapped_keyframe":
                fusion_hooks.append(node)
            if node.func.attr == "map" and any(
                keyword.arg == "prune"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            ):
                prune_calls.append(node)
    if len(captures) != 1:
        raise ValueError("Mapper.run must capture mapped_viewpoint exactly once")
    regular_hooks = [
        node
        for node in fusion_hooks
        if len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "mapped_viewpoint"
    ]
    if len(regular_hooks) != 1:
        raise ValueError("regular active-fusion hook must consume mapped_viewpoint exactly once")
    capture = captures[0]
    hook = regular_hooks[0]
    rebinds_between = [
        node for node in viewpoint_rebinds if capture.lineno < node.lineno < hook.lineno
    ]
    if not rebinds_between:
        raise ValueError("mapped_viewpoint capture is not before the pose-window rebind")
    prune_before = [node for node in prune_calls if capture.lineno < node.lineno < hook.lineno]
    if len(prune_before) != 1:
        raise ValueError("regular active-fusion hook must follow exactly one prune mapping call")
    required_runtime_guards = (
        "mapped_viewpoint is not self.cameras[video_idx]",
        "int(mapped_viewpoint.uid) != int(video_idx)",
        "int(mapped_viewpoint.timestamp) != int(idx)",
    )
    missing = [fragment for fragment in required_runtime_guards if fragment not in source]
    if missing:
        raise ValueError("Mapper mapped-viewpoint runtime guard drifted: " + ", ".join(missing))
    return {
        "schema": "unblur_slam.mapper_mapped_viewpoint_hook_binding.v2",
        "accepted": True,
        "mapper_path": str(mapper_path.resolve()) if mapper_source is None else None,
        "mapper_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "capture_line": int(capture.lineno),
        "pose_window_rebind_lines": [int(node.lineno) for node in rebinds_between],
        "prune_mapping_line": int(prune_before[0].lineno),
        "regular_fusion_hook_line": int(hook.lineno),
        "hook_argument": "mapped_viewpoint",
        "runtime_identity_and_source_guards": True,
    }


class _WholeDeviceGpuMonitor:
    """Continuously sample physical GPU memory/utilization with nvidia-smi."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.handle: Any = None
        self.process: Optional[subprocess.Popen[str]] = None

    def start(self) -> None:
        self.handle = self.output.open("x", encoding="utf-8", buffering=1)
        command = [
            "nvidia-smi",
            f"--id={PHYSICAL_GPU}",
            "--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
            f"--loop-ms={GPU_MONITOR_INTERVAL_MS}",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=self.handle,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        time.sleep(0.05)
        if self.process.poll() is not None:
            stderr = "" if self.process.stderr is None else self.process.stderr.read()
            self.handle.close()
            raise RuntimeError(f"continuous nvidia-smi monitor failed to start: {stderr}")

    def stop(self) -> dict[str, Any]:
        if self.process is None or self.handle is None:
            raise RuntimeError("GPU monitor was not started")
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.handle.close()
        rows = []
        for line in self.output.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 5:
                continue
            timestamp, index, uuid, memory_mib, utilization = fields
            try:
                row = {
                    "timestamp": timestamp,
                    "physical_gpu": int(index),
                    "uuid": uuid,
                    "memory_used_mib": int(memory_mib),
                    "utilization_percent": int(utilization),
                }
            except ValueError:
                continue
            if row["physical_gpu"] == int(PHYSICAL_GPU):
                rows.append(row)
        if not rows:
            raise RuntimeError("continuous nvidia-smi monitor produced no valid samples")
        return {
            "schema": "unblur_slam.whole_device_gpu_monitor.v1",
            "physical_gpu": int(PHYSICAL_GPU),
            "uuid": rows[0]["uuid"],
            "requested_interval_ms": GPU_MONITOR_INTERVAL_MS,
            "continuous_subprocess_sampling": True,
            "sample_count": len(rows),
            "first_timestamp": rows[0]["timestamp"],
            "last_timestamp": rows[-1]["timestamp"],
            "maximum_memory_used_mib": max(row["memory_used_mib"] for row in rows),
            "maximum_utilization_percent": max(row["utilization_percent"] for row in rows),
            "raw_csv": str(self.output),
            "is_exact_instantaneous_peak": False,
        }


def _load_configs() -> dict[str, dict[str, Any]]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from thirdparty.glorie_slam import config as config_io

    return {
        arm: config_io.load_config(str(path), str(DEFAULT_CONFIG))
        for arm, path in CONFIGS.items()
    }


def _fixed_contract(cfg: Mapping[str, Any], arm: str) -> dict[str, Any]:
    from src.utils.fixed_keyframes import parse_fixed_source_keyframe_contract

    parsed = parse_fixed_source_keyframe_contract(cfg)
    if parsed is None:
        raise ValueError(f"{arm}: fixed-source-keyframe contract is disabled")
    if tuple(parsed["source_indices"]) != EXPECTED_FIXED_SOURCE_KEYFRAMES:
        raise ValueError(f"{arm}: fixed source-keyframe schedule drifted")
    disclosure = cfg.get("fixed_kf_resplat3_active_fusion_221", {}) or {}
    expected = {
        "schema": "unblur_slam.fr2_xyz_fixed_kf_resplat3_active_fusion_221.v2",
        "experiment_revision": "v2",
        "supersedes_v1_mapper_hook_binding_bug": True,
        "mapped_viewpoint_binding_contract": "captured_before_pose_window_loop",
        "v1_output_preserved_and_not_reused": True,
        "conditional_on_frozen_evssm_baseline_schedule": True,
        "shares_pose_estimates_between_arms": False,
        "uses_ground_truth_poses_for_slam_or_fusion": False,
        "selection_membership_clear_gt_conditioned": True,
        "uses_ground_truth_poses_or_depths_for_fusion": False,
        "uses_independent_clear_pixels_for_fusion": False,
        "uses_clear_gt_metrics_for_fusion_gate": False,
        "runtime_baseline_artifact_dependency": False,
        "trigger_after_mapped_keyframe_count": 8,
        "expected_first_eight_mapped_source_indices": list(
            EXPECTED_FIRST_EIGHT_MAPPED_SOURCES
        ),
        "trigger_source_index": 166,
        "downstream_online_mapped_source_indices_after_fusion": list(
            EXPECTED_DOWNSTREAM_MAPPED_SOURCES
        ),
        "maximum_fusion_attempts": 1,
        "synchronous_blocking_timing": True,
        "subprocess_gate_and_merge_inside_online_timer": True,
        "official_recurrent_updates": 3,
        "official_selected_state_index_zero_based": 2,
        "fourth_recurrent_state_computed": False,
        "merge_policy": "append_voxel_deduplicated_capped",
        "postmerge_reconstruction_gate_can_rollback": True,
        "frozen_baseline_video_npz_sha256": FROZEN_BASELINE_VIDEO_SHA256,
    }
    for key, wanted in expected.items():
        if disclosure.get(key) != wanted:
            raise ValueError(f"{arm}: fusion disclosure {key} drifted")
    frozen = Path(str(disclosure.get("frozen_baseline_video_npz", ""))).expanduser().resolve()
    if frozen != FROZEN_BASELINE_VIDEO:
        raise ValueError(f"{arm}: frozen fixed-schedule provenance path drifted")
    return parsed


def _validate_arm(cfg: Mapping[str, Any], arm: str) -> dict[str, Any]:
    mapping = cfg.get("mapping", {}) or {}
    training = mapping.get("Training", {}) or {}
    observed = (
        str(cfg.get("dataset", "")).lower(),
        str(cfg.get("scene", "")),
        int(cfg.get("max_frames", -1)),
        int(cfg.get("stride", -1)),
        int(cfg.get("setup_seed", -1)),
        str(cfg.get("device", "")),
        bool(cfg.get("warmup_mapper", False)),
        bool(cfg.get("clear_init", True)),
        int((cfg.get("cam", {}) or {}).get("W_out", -1)),
        int((cfg.get("cam", {}) or {}).get("H_out", -1)),
        int(training.get("init_itr_num", -1)),
        int(training.get("mapping_itr_num", -1)),
        int(training.get("tracking_itr_num", -1)),
        int(mapping.get("final_refine_iters", -1)),
    )
    expected = (
        "tumrgbd", "freiburg2_xyz", 221, 1, 43, "cuda:0", True, False,
        512, 384, 1050, 100, 100, 100,
    )
    if observed != expected:
        raise ValueError(f"{arm}: compute contract drifted: {observed}")
    if mapping.get("online_plotting") is not False:
        raise ValueError(f"{arm}: online plotting must be disabled")
    if mapping.get("eval_before_final_ba") is not False:
        raise ValueError(f"{arm}: pre-final-BA evaluation must be disabled")
    if mapping.get("hydrate_missing_droid_keyframes") is not True:
        raise ValueError(f"{arm}: complete-prefix keyframe hydration is required")
    replay = mapping.get("resplat", {}) or {}
    if (
        replay.get("enabled") is not False
        or replay.get("online_enabled") is not False
        or int(replay.get("extra_iters", -1)) != 0
    ):
        raise ValueError(f"{arm}: residual replay must remain disabled")
    submaps = cfg.get("submaps", {}) or {}
    if bool(submaps.get("enabled", False)) or bool(
        (submaps.get("official_resplat_sidecar", {}) or {}).get("enabled", False)
    ):
        raise ValueError(f"{arm}: asynchronous submap sidecar must remain disabled")
    if bool((cfg.get("framecrafter", {}) or {}).get("enabled", False)):
        raise ValueError(f"{arm}: FrameCrafter must remain disabled")
    deblur = cfg.get("deblur", {}) or {}
    if str(deblur.get("frontend", "")) != "evssm":
        raise ValueError(f"{arm}: both arms must use official Unblur-SLAM EVSSM")
    if str(deblur.get("causal_checkpoint", "")):
        raise ValueError(f"{arm}: causal custom frontend must remain disabled")
    if str(cfg.get("evssm_checkpoint_sha256", "")) != BASE.EXPECTED_SHA256["evssm"]:
        raise ValueError(f"{arm}: EVSSM checkpoint digest drifted")
    evaluation = cfg.get("evaluation", {}) or {}
    if (
        evaluation.get("clear_gt_scope") != "prefix_smoke"
        or tuple(evaluation.get("expected_clear_gt_source_indices", ()))
        != EXPECTED_FIXED_SOURCE_KEYFRAMES
    ):
        raise ValueError(f"{arm}: clear-GT prefix scope drifted")
    output = Path(str((cfg.get("data", {}) or {}).get("output", ""))).expanduser().resolve()
    if output != OUTPUTS[arm]:
        raise ValueError(f"{arm}: output path drifted: {output}")
    fixed = _fixed_contract(cfg, arm)

    from src.refinement.official_resplat_active_fusion import ActiveFusionConfig

    fusion = ActiveFusionConfig.from_dict(
        mapping.get("official_resplat_active_fusion", {}) or {},
        default_output_root=output / "freiburg2_xyz" / "official_resplat_active_fusion",
    )
    if fusion.enabled is not (arm == "fused"):
        raise ValueError(f"{arm}: active-fusion enable switch drifted")
    if fusion.resplat_repo_commit != EXPECTED_RESPLAT_COMMIT:
        raise ValueError(f"{arm}: official ReSplat commit pin drifted")
    if fusion.expected_checkpoint_sha256 != EXPECTED_RESPLAT_CHECKPOINT_SHA256:
        raise ValueError(f"{arm}: official ReSplat checkpoint pin drifted")
    return {"fixed": fixed, "fusion": fusion}


def _load_and_validate_configs() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    configs = _load_configs()
    contracts = {arm: _validate_arm(configs[arm], arm) for arm in ("baseline", "fused")}
    raw_differences = BASE._pair_differences(configs["baseline"], configs["fused"])
    differences = {
        path: {
            "baseline": record.get("baseline"),
            "fused": record.get("turtle"),
        }
        for path, record in raw_differences.items()
    }
    unexpected = sorted(set(differences) - ALLOWED_PAIR_DIFFS)
    if unexpected:
        raise ValueError(
            "paired configs differ outside output/fusion switch: " + ", ".join(unexpected)
        )
    if set(differences) != ALLOWED_PAIR_DIFFS:
        raise ValueError(f"paired config difference set drifted: {sorted(differences)}")
    return configs, {"contracts": contracts, "differences": differences}


def _inspect_resplat(fusion: Any) -> dict[str, Any]:
    paired_script = REPO_ROOT / "scripts/run_paired_official_resplat_smoke.py"
    spec = importlib.util.spec_from_file_location("paired_resplat_inspection", paired_script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    repo = module.inspect_official_repo(Path(fusion.resplat_repo))
    if repo["commit"] != EXPECTED_RESPLAT_COMMIT:
        raise ValueError(f"official ReSplat checkout commit drifted: {repo['commit']}")
    checkpoint = BASE._require_artifact(
        fusion.checkpoint,
        fusion.expected_checkpoint_sha256,
        EXPECTED_RESPLAT_CHECKPOINT_SHA256,
        "official ReSplat checkpoint",
    )
    executable = Path(fusion.python_executable).expanduser().absolute()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"official ReSplat Python is not executable: {executable}")
    executable_realpath = executable.resolve(strict=True)
    runner = Path(fusion.runner_script).expanduser().resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"official ReSplat sidecar runner is missing: {runner}")
    runner_text = runner.read_text(encoding="utf-8")
    if (
        "--num-refine" not in runner_text
        or "requested_recurrent_updates" not in runner_text
        or '"covariances":' not in runner_text
    ):
        raise ValueError(
            "official sidecar runner lacks explicit state3/covariance contract"
        )
    return {
        "repository": repo,
        "checkpoint": {"path": str(checkpoint), "sha256": EXPECTED_RESPLAT_CHECKPOINT_SHA256},
        "python_executable": str(executable),
        "python_executable_realpath": str(executable_realpath),
        "python_executable_symlink_preserved_lexically": str(executable)
        != str(executable_realpath),
        "runner_script": str(runner),
        "refinement_updates": 3,
        "selected_state_index_zero_based": 2,
        "fourth_state_computed": False,
    }


def _real_artifact_bridge_probe(fusion: Any) -> dict[str, Any]:
    """Numerically exercise the bridge and static merge gates on real output.

    This historical artifact predates native covariance export.  Covariances
    are reconstructed from its official per-view scale/xyzw rotation tensors
    solely for this CPU preflight.  Production state3 artifacts must contain
    the new sixth, authoritative ``covariances`` array and are verified by the
    sidecar/world-artifact verifier before any map merge.
    """

    import numpy as np
    from scipy.spatial.transform import Rotation

    from src.refinement.resplat_unblur_bridge import build_unblur_world_arrays

    if not REAL_BRIDGE_PROBE_SNAPSHOT.is_file() or not REAL_BRIDGE_PROBE_NATIVE.is_file():
        raise FileNotFoundError("real 71680-Gaussian bridge preflight artifact is missing")
    native_sha = _sha256_file(REAL_BRIDGE_PROBE_NATIVE)
    if native_sha != REAL_BRIDGE_PROBE_NATIVE_SHA256:
        raise ValueError("real bridge-probe native artifact digest drifted")
    snapshot = json.loads(REAL_BRIDGE_PROBE_SNAPSHOT.read_text(encoding="utf-8"))
    frames = snapshot.get("frames") or []
    if len(frames) != 8:
        raise ValueError("real bridge probe requires exactly eight context frames")
    with np.load(REAL_BRIDGE_PROBE_NATIVE, allow_pickle=False) as archive:
        if set(archive.files) != {"means", "scales", "rotations", "harmonics", "opacities"}:
            raise ValueError("historical real bridge-probe artifact layout drifted")
        native = {name: np.asarray(archive[name]) for name in archive.files}
    count = int(native["means"].shape[0])
    if count != 71_680 or count % len(frames):
        raise ValueError("real bridge-probe topology is not official fixed 8-view topology")
    per_view = count // len(frames)
    pivot = np.asarray(frames[len(frames) // 2]["c2w_opencv"], dtype=np.float64)
    covariance_blocks = []
    for slot, frame in enumerate(frames):
        source = np.asarray(frame["c2w_opencv"], dtype=np.float64)
        aligned_rotation = pivot[:3, :3].T @ source[:3, :3]
        begin, end = slot * per_view, (slot + 1) * per_view
        native_rotations = Rotation.from_quat(
            native["rotations"][begin:end]
        ).as_matrix()
        local_rotations = np.einsum(
            "ij,njk->nik", aligned_rotation, native_rotations, optimize=True
        )
        scales = native["scales"][begin:end].astype(np.float64)
        covariance_blocks.append(
            np.einsum(
                "nij,nj,nkj->nik",
                local_rotations,
                scales * scales,
                local_rotations,
                optimize=True,
            )
        )
    covariances = np.concatenate(covariance_blocks, axis=0)
    arrays, metadata = build_unblur_world_arrays(
        means_local=native["means"],
        covariances_local=covariances,
        harmonics_local=native["harmonics"],
        opacities=native["opacities"],
        pivot_c2w=pivot,
        owner_frame_ids=[int(frame["frame_id"]) for frame in frames],
        owner_sequence_ordinals=[int(frame["sequence_ordinal"]) for frame in frames],
    )
    if arrays["harmonics_world"].shape != (71_680, 3, 1):
        raise ValueError("real bridge probe did not produce strict DC-only harmonics")
    if (
        metadata.get("official_no_rotate_sh") is not True
        or int(metadata.get("source_harmonic_dimension", -1)) != 16
        or int(metadata.get("imported_harmonic_dimension", -1)) != 1
        or int(metadata.get("dropped_higher_order_harmonics", -1)) != 15
    ):
        raise ValueError("real bridge probe harmonic contract drifted")
    required_arrays = (
        "means_world",
        "scales_world",
        "rotations_world_wxyz",
        "harmonics_world",
        "opacities",
        "owner_frame_ids",
    )
    if not all(np.isfinite(arrays[name]).all() for name in required_arrays):
        raise ValueError("real bridge probe produced non-finite active-merge arrays")
    merge = dict(fusion.merge)
    static_mask = (
        (arrays["opacities"] >= float(merge["min_opacity"]))
        & np.all(arrays["scales_world"] >= float(merge["min_scale"]), axis=1)
        & np.all(arrays["scales_world"] <= float(merge["max_scale"]), axis=1)
        & np.all(
            np.abs(arrays["means_world"]) <= float(merge["max_abs_position"]),
            axis=1,
        )
    )
    static_survivors = int(np.count_nonzero(static_mask))
    incoming_voxels = np.floor(
        arrays["means_world"][static_mask] / float(merge["voxel_size"])
    ).astype(np.int64)
    unique_incoming = int(np.unique(incoming_voxels, axis=0).shape[0])
    capped_candidates = min(unique_incoming, int(merge["max_new_gaussians"]))
    if capped_candidates < int(merge["min_new_gaussians"]):
        raise ValueError("real bridge probe fails preregistered static merge minimum")
    factorization = dict(metadata.get("factorization") or {})
    if int(factorization.get("significant_negative_gaussian_count", -1)) != 0:
        raise ValueError("real bridge probe has covariance spectrum beyond tolerance")
    return {
        "schema": "unblur_slam.real_resplat_bridge_cpu_preflight.v1",
        "accepted": True,
        "gpu_used": False,
        "production_artifact_accepted_as_current_schema": False,
        "historical_probe_only": True,
        "generated_from_historical_turtle_smoke_observations": True,
        "experiment_uses_this_artifact_as_runtime_input": False,
        "experiment_frontend_both_arms": "official_unblur_slam_evssm",
        "covariance_source": (
            "legacy_scale_xyzw_per_view_reconstruction_for_cpu_preflight_only"
        ),
        "production_requires_authoritative_native_covariances_array": True,
        "native_path": str(REAL_BRIDGE_PROBE_NATIVE),
        "native_sha256": native_sha,
        "snapshot_path": str(REAL_BRIDGE_PROBE_SNAPSHOT),
        "gaussian_count": count,
        "native_array_count": 5,
        "production_native_array_count": 6,
        "world_harmonics_shape": list(arrays["harmonics_world"].shape),
        "source_harmonic_dimension": 16,
        "imported_harmonic_dimension": 1,
        "dropped_higher_order_harmonics": 15,
        "static_gate_survivor_count": static_survivors,
        "incoming_voxel_unique_count": unique_incoming,
        "capped_candidate_count_before_active_map_collision": capped_candidates,
        "active_map_collision_gate_exercised": False,
        "active_map_collision_gate_deferred_to_runtime": True,
        "factorization": factorization,
    }


def preflight(
    *, arms: Iterable[str] = ("baseline", "fused"), check_output_available: bool = True
) -> dict[str, Any]:
    selected = tuple(arms)
    if not selected or any(arm not in CONFIGS for arm in selected):
        raise ValueError(f"invalid arm selection: {selected}")
    configs, validation = _load_and_validate_configs()
    if check_output_available:
        for arm in selected:
            if OUTPUTS[arm].exists() or OUTPUTS[arm].is_symlink():
                raise FileExistsError(f"refusing to overwrite {arm} output: {OUTPUTS[arm]}")
    baseline = configs["baseline"]
    droid = BASE._require_artifact(
        baseline["tracking"].get("pretrained"),
        baseline["tracking"].get("pretrained_sha256"),
        BASE.EXPECTED_SHA256["droid"],
        "DROID checkpoint",
    )
    omnidata = BASE._require_artifact(
        baseline["mono_prior"].get("depth_pretrained"),
        baseline["mono_prior"].get("depth_pretrained_sha256"),
        BASE.EXPECTED_SHA256["omnidata"],
        "Omnidata checkpoint",
    )
    evssm = BASE._require_artifact(
        baseline.get("evssm_checkpoint"),
        baseline.get("evssm_checkpoint_sha256"),
        BASE.EXPECTED_SHA256["evssm"],
        "EVSSM checkpoint",
    )
    resplat = _inspect_resplat(validation["contracts"]["fused"]["fusion"])
    real_bridge_probe = _real_artifact_bridge_probe(
        validation["contracts"]["fused"]["fusion"]
    )
    mapper_hook_binding = validate_mapper_hook_binding()
    implementation = implementation_provenance()
    frozen = FIXED._validate_frozen_baseline_provenance()

    from src.utils.datasets import get_dataset
    from src.utils.eval_frames import (
        PREFIX_SMOKE_METRIC_SCOPE,
        clear_gt_metric_scope,
        clear_gt_source_indices,
        validate_clear_gt_protocol_scope,
    )

    previous_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        dataset = get_dataset(baseline, device="cpu")
    finally:
        os.chdir(previous_cwd)
    if len(dataset) != 221:
        raise ValueError(f"bounded dataset has {len(dataset)} frames, expected 221")
    protocol = clear_gt_source_indices(baseline, dataset)
    if protocol is None or tuple(sorted(protocol)) != BASE.EXPECTED_FULL_PROTOCOL:
        raise ValueError("published fr2_xyz clear-GT protocol drifted")
    available = validate_clear_gt_protocol_scope(baseline, dataset)
    if available is None or tuple(sorted(available)) != EXPECTED_FIXED_SOURCE_KEYFRAMES:
        raise ValueError("bounded dataset does not expose the exact 11-frame prefix")
    if clear_gt_metric_scope(baseline) != PREFIX_SMOKE_METRIC_SCOPE:
        raise ValueError("bounded metric label drifted")

    return {
        "schema": "unblur_slam.fr2_xyz_fixed_kf_resplat3_active_fusion_221_preflight.v2",
        "scope": {
            "experiment_revision": "v2",
            "superseded_v1_output_root": str(SUPERSEDED_V1_OUTPUT_ROOT),
            "v1_outputs_preserved_and_not_reused": True,
            "v1_fused_run_failed_closed_due_to_mapper_hook_binding": True,
            "source_first": 0,
            "source_last": 220,
            "source_count": 221,
            "fixed_source_keyframes": list(EXPECTED_FIXED_SOURCE_KEYFRAMES),
            "clear_gt_metric_scope": PREFIX_SMOKE_METRIC_SCOPE,
            "official_online_optimization_budget": True,
            "final_refine_iterations": 100,
            "complete_three_sequence_paper_benchmark": False,
            "paper_26k_offline_refinement": False,
        },
        "paired_contract": {
            "seed": 43,
            "resolution_wh": [512, 384],
            "init_iterations": 1050,
            "mapping_iterations_per_keyframe": 100,
            "tracking_iterations": 100,
            "final_refine_iterations": 100,
            "same_official_evssm_both_arms": True,
            "poses_and_depths_estimated_independently_per_arm": True,
            "allowed_resolved_config_differences": validation["differences"],
            "baseline_resolved_sha256": BASE._canonical_sha256(configs["baseline"]),
            "fused_resolved_sha256": BASE._canonical_sha256(configs["fused"]),
        },
        "fusion_contract": {
            "mapped_viewpoint_binding": mapper_hook_binding,
            "trigger_after_mapped_keyframe_count": 8,
            "expected_first_eight_actually_mapped_source_indices": list(
                EXPECTED_FIRST_EIGHT_MAPPED_SOURCES
            ),
            "trigger_source_index": 166,
            "downstream_online_mapped_source_indices_after_fusion": list(
                EXPECTED_DOWNSTREAM_MAPPED_SOURCES
            ),
            "synchronous_inside_online_timer": True,
            "maximum_attempts": 1,
            "official_recurrent_updates": 3,
            "selected_state_index_zero_based": 2,
            "fourth_state_computed": False,
            "selection_membership_clear_gt_conditioned": True,
            "ground_truth_poses_or_depths_consumed": False,
            "independent_clear_pixels_consumed": False,
            "clear_gt_metrics_consumed_by_gate": False,
            "merge": dict(validation["contracts"]["fused"]["fusion"].merge),
            "sidecar_quality_gate": dict(
                validation["contracts"]["fused"]["fusion"].sidecar_quality_gate
            ),
            "postmerge_quality_gate": dict(
                validation["contracts"]["fused"]["fusion"].postmerge_quality_gate
            ),
        },
        "artifacts": {
            "droid": {"path": str(droid), "sha256": BASE.EXPECTED_SHA256["droid"]},
            "omnidata": {"path": str(omnidata), "sha256": BASE.EXPECTED_SHA256["omnidata"]},
            "evssm": {"path": str(evssm), "sha256": BASE.EXPECTED_SHA256["evssm"]},
            "official_resplat": resplat,
            "real_71680_gaussian_cpu_bridge_probe": real_bridge_probe,
            "frozen_fixed_schedule": frozen,
        },
        "implementation_provenance": implementation,
        "execution": {
            "selected_arms": list(selected),
            "sequential_same_physical_gpu": int(PHYSICAL_GPU),
            "process_device": "cuda:0",
            "outputs": {arm: str(OUTPUTS[arm]) for arm in selected},
        },
    }


def _interrupt(process: subprocess.Popen[str], log: Any) -> int:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
    try:
        return int(process.wait(timeout=30))
    except subprocess.TimeoutExpired:
        log.write("[launcher] SIGINT timeout; forwarding SIGTERM\n")
        log.flush()
        os.killpg(process.pid, signal.SIGTERM)
        return int(process.wait())


def _run_arm(arm: str, audit: Mapping[str, Any]) -> int:
    output = OUTPUTS[arm]
    output.mkdir(parents=True, exist_ok=False)
    (output / "preflight.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    command = [sys.executable, str(REPO_ROOT / "run.py"), str(CONFIGS[arm])]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": PHYSICAL_GPU,
            "PYTHONUNBUFFERED": "1",
            "UNBLUR_SKIP_NR_IQA": "1",
        }
    )
    started = time.monotonic()
    code = -1
    monitor = _WholeDeviceGpuMonitor(output / "gpu_monitor.csv")
    with (output / "launch.log").open("x", encoding="utf-8", buffering=1) as log:
        log.write(f"[launcher] fixed-11KF EVSSM ReSplat-state3 pair revision=v2 arm={arm}\n")
        log.write("[launcher] source_keyframes=" + json.dumps(list(EXPECTED_FIXED_SOURCE_KEYFRAMES)) + "\n")
        log.write("[launcher] init=1050 mapping=100 tracking=100 final=100 seed=43\n")
        log.write(
            "[launcher] fusion=" + ("state3_sync_after_mapped_source166" if arm == "fused" else "disabled") + "\n"
        )
        log.write(
            "[launcher] schedule_membership_clear_gt_conditioned=true "
            "gt_pose_depth_clear_pixels_metrics_consumed=false "
            "subprocess_merge_inside_online_timer=true\n"
        )
        log.write(f"[launcher] whole_device_gpu_monitor_interval_ms={GPU_MONITOR_INTERVAL_MS}\n")
        monitor.start()
        process: Optional[subprocess.Popen[str]] = None
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            code = int(process.wait())
        except KeyboardInterrupt:
            log.write("[launcher] KeyboardInterrupt; forwarding SIGINT\n")
            code = 130 if process is None else _interrupt(process, log)
        finally:
            gpu_summary = monitor.stop()
            (output / "gpu_monitor_summary.json").write_text(
                json.dumps(gpu_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            log.write(
                "[launcher] whole_device_gpu_max_mib="
                + str(gpu_summary["maximum_memory_used_mib"])
                + " samples="
                + str(gpu_summary["sample_count"])
                + "\n"
            )
        log.write(f"[launcher] exit_code={code}\n")
    (output / "launcher_runtime.json").write_text(
        json.dumps(
            {
                "schema": "unblur_slam.external_wall_runtime.v1",
                "experiment_revision": "v2",
                "arm": arm,
                "wall_runtime_seconds": time.monotonic() - started,
                "exit_code": code,
                "physical_gpu": int(PHYSICAL_GPU),
                "process_device": "cuda:0",
                "fixed_source_keyframes": list(EXPECTED_FIXED_SOURCE_KEYFRAMES),
                "fusion_enabled": arm == "fused",
                "whole_device_gpu_monitor_summary": str(
                    output / "gpu_monitor_summary.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return code


def _selected_arms(value: str) -> tuple[str, ...]:
    return ("baseline", "fused") if value == "all" else (value,)


def run_pair(selection: str) -> int:
    arms = _selected_arms(selection)
    audit = preflight(arms=arms, check_output_available=True)
    for arm in arms:
        guard = FIXED._assert_physical_gpu_free()
        arm_audit = json.loads(json.dumps(audit))
        arm_audit["execution"]["gpu_free_guard_before_launch"] = guard
        print(f"[launch] {arm}: physical GPU {PHYSICAL_GPU} -> process cuda:0")
        code = _run_arm(arm, arm_audit)
        if code != 0:
            return code
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true", help="CPU-only validation (default)")
    action.add_argument("--run", action="store_true", help="launch selected arm(s)")
    parser.add_argument("--arm", choices=("baseline", "fused", "all"), default="all")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.run:
            return run_pair(args.arm)
        print(json.dumps(preflight(arms=_selected_arms(args.arm)), indent=2, sort_keys=True))
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
