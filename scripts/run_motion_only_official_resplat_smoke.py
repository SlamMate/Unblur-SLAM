#!/usr/bin/env python3
"""Run the frozen motion-only TURTLE -> official ReSplat sidecar smoke.

The keyframe pool and expected official-FPS contexts come exclusively from a
pre-evaluation frozen DROID run.  This runner never compares against the 26K
map and never imports the historical residual replay sampler.  Its stages are
separate so CPU preflight, TURTLE materialization, COLMAP export, and official
ReSplat inference can be audited or resumed without overwriting any artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_motion_only_resplat_protocol import (  # noqa: E402
    CONFIG as DEFAULT_CONFIG,
    OUTPUT_SCHEMA as PROTOCOL_SCHEMA,
    _load_json,
    _sha,
    _validate_config,
    inspect_official_resplat,
    load_frozen_inputs,
    sha256_file,
)
from scripts.export_tum_official_resplat_scene import export_scene  # noqa: E402
from scripts.materialize_tum_turtle_stream import (  # noqa: E402
    FR2_XYZ_DISTORTION,
    FR2_XYZ_HEIGHT_EDGE,
    FR2_XYZ_WIDTH_EDGE,
    materialize_tum_turtle_stream,
    official_artifact_record,
)
from src.turtle_backend import (  # noqa: E402
    TurtleStreamingBackend,
    build_turtle_model,
    validate_turtle_artifacts,
)


AUDIT_SCHEMA = "unblur_slam.motion_only_official_resplat_smoke.v1"
TURTLE_SCHEMA = "unblur_slam.turtle_stream_materialization.v1"
SCENE_SCHEMA = "unblur_slam.official_resplat_colmap_scene.v1"
RESPLAT_SCHEMA = "unblur_slam.paired_official_resplat_smoke.v1"
PAIRED_RUNNER = ROOT / "scripts/run_paired_official_resplat_smoke.py"
SOURCE_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _verified_path_record(value: object, label: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain path and sha256")
    path = Path(str(value.get("path", ""))).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    expected = _sha(value.get("sha256"), f"{label}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return path, actual


def _load_protocol(
    config: Mapping[str, Any], config_path: Path | str
) -> tuple[Path, dict[str, Any]]:
    path = Path(str(config["outputs"]["protocol_dir"])).expanduser().resolve()
    manifest_path, manifest = _load_json(path / "protocol_manifest.json", "protocol manifest")
    if manifest.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("protocol manifest has the wrong schema")
    config_record = manifest.get("config")
    configured_path, configured_sha = _verified_path_record(
        config_record, "protocol config record"
    )
    expected_config = Path(config_path).expanduser().resolve()
    if configured_path != expected_config or configured_sha != sha256_file(expected_config):
        raise ValueError("protocol was built from different config bytes")
    safety = manifest.get("safety") or {}
    required_true = (
        "selection_frozen_before_evaluation",
        "same_frozen_run_pose_source_for_keyframes_and_contexts",
        "official_fps_must_match_postflight",
    )
    required_false = (
        "clear_gt_membership_file_opened",
        "ground_truth_pose_file_opened",
        "ground_truth_image_opened",
        "depth_file_opened",
        "metric_computed",
        "old_clear_conditioned_artifact_read",
        "old_clear_conditioned_artifact_overwritten",
    )
    if any(safety.get(key) is not True for key in required_true) or any(
        safety.get(key) is not False for key in required_false
    ):
        raise ValueError("protocol safety contract drifted")
    return manifest_path, manifest


def _source_indices(protocol: Mapping[str, Any], key: str) -> list[int]:
    values = [int(value) for value in protocol[key]["source_indices"]]
    if not values or values != sorted(set(values)):
        raise ValueError(f"{key}.source_indices must be non-empty, sorted, and unique")
    return values


def preflight(config_path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    source, config = _load_json(config_path, "pipeline config")
    _validate_config(config)
    frozen = load_frozen_inputs(config)
    official = inspect_official_resplat(config)
    protocol_path, protocol = _load_protocol(config, source)
    keyframes = _source_indices(protocol, "keyframe_selection")
    if keyframes != frozen["keyframes"]:
        raise ValueError("protocol keyframes disagree with the frozen DROID video")
    contexts = [
        int(value)
        for value in protocol["resplat_selection"]["context_source_indices"]
    ]
    targets = [
        int(value)
        for value in protocol["resplat_selection"]["target_source_indices"]
    ]
    if (
        len(contexts) != int(config["selection"]["num_context"])
        or sorted(contexts + targets) != keyframes
        or set(contexts) & set(targets)
    ):
        raise ValueError("protocol context/target partition is invalid")

    turtle = config["official_turtle"]
    turtle_artifacts = validate_turtle_artifacts(
        {
            "turtle_repo": turtle["repository"],
            "turtle_config": turtle["config"],
            "turtle_checkpoint": turtle["checkpoint"],
            "turtle_repo_commit": turtle["commit"],
            "turtle_config_sha256": turtle["config_sha256"],
            "turtle_checkpoint_sha256": turtle["checkpoint_sha256"],
        },
        load_weights=False,
    )
    execution = config["execution"]
    if (
        execution.get("physical_gpu") != 1
        or execution.get("cuda_visible_devices") != "1"
        or execution.get("process_device") != "cuda:0"
        or turtle.get("inference_precision") != "fp16"
    ):
        raise ValueError("GPU/FP16 execution contract drifted")
    resplat_python = Path(os.path.expanduser(str(execution["resplat_python"])))
    if not resplat_python.is_absolute():
        raise ValueError("official ReSplat Python must be a lexical absolute path")
    if not resplat_python.is_file() or not os.access(resplat_python, os.X_OK):
        raise FileNotFoundError(f"official ReSplat Python is unavailable: {resplat_python}")
    resplat_python_realpath = Path(os.path.realpath(resplat_python))
    if not resplat_python_realpath.is_file() or not os.access(
        resplat_python_realpath, os.X_OK
    ):
        raise FileNotFoundError(
            f"official ReSplat Python realpath is unavailable: {resplat_python_realpath}"
        )
    outputs = {
        key: Path(str(value)).expanduser().resolve()
        for key, value in config["outputs"].items()
    }
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("pipeline output paths must be distinct")
    for key in ("turtle_dir", "scene_dir", "resplat_dir", "audit_dir"):
        if outputs[key] == outputs["protocol_dir"]:
            raise ValueError("new output aliases immutable protocol directory")
    return {
        "schema": AUDIT_SCHEMA,
        "preflight_only": True,
        "config": {"path": str(source), "sha256": sha256_file(source)},
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "frozen_tracking": {
            "freeze_marker": str(frozen["freeze_path"]),
            "freeze_sha256": frozen["freeze_sha256"],
            "trajectory": str(frozen["trajectory_path"]),
            "trajectory_sha256": frozen["trajectory_sha256"],
            "frames_csv": str(frozen["frames_csv"]),
            "frames_csv_sha256": frozen["frames_csv_sha256"],
        },
        "selection": {
            "motion_keyframes": keyframes,
            "motion_keyframe_count": len(keyframes),
            "expected_context_source_indices": contexts,
            "expected_target_source_indices": targets,
            "target_count": len(targets),
        },
        "official_turtle": {
            "repository": str(turtle_artifacts.repo),
            "commit": turtle_artifacts.commit,
            "checkpoint_sha256": turtle_artifacts.checkpoint_sha256,
            "inference_precision": "fp16",
        },
        "official_resplat": official,
        "execution": {
            **dict(execution),
            "resplat_python_lexical": str(resplat_python),
            "resplat_python_realpath": str(resplat_python_realpath),
            "lexical_environment_path_preserved": True,
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
        "excluded": dict(config["excluded"]),
        "safety": {
            "selection_frozen_before_this_pipeline": True,
            "clear_gt_membership_used": False,
            "26k_comparison": False,
            "old_artifact_overwrite": False,
        },
    }


def materialize_turtle(audit: Mapping[str, Any], config: Mapping[str, Any]) -> Path:
    import torch

    output = Path(audit["outputs"]["turtle_dir"])
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite TURTLE output: {output}")
    turtle = config["official_turtle"]
    deblur_cfg = {
        "turtle_repo": turtle["repository"],
        "turtle_config": turtle["config"],
        "turtle_checkpoint": turtle["checkpoint"],
        "turtle_repo_commit": turtle["commit"],
        "turtle_config_sha256": turtle["config_sha256"],
        "turtle_checkpoint_sha256": turtle["checkpoint_sha256"],
    }
    artifacts = validate_turtle_artifacts(deblur_cfg, load_weights=False)
    model = build_turtle_model(artifacts, device="cuda:0")
    backend = TurtleStreamingBackend(
        model, device="cuda:0", inference_precision=turtle["inference_precision"]
    )
    turtle_record = official_artifact_record(artifacts, model)
    manifest = materialize_tum_turtle_stream(
        frames_csv=Path(audit["frozen_tracking"]["frames_csv"]),
        output_dir=output,
        turtle_repo=turtle["repository"],
        turtle_config=turtle["config"],
        turtle_checkpoint=turtle["checkpoint"],
        start_index=0,
        end_index=220,
        emitted_source_indices=audit["selection"]["motion_keyframes"],
        width=512,
        height=384,
        width_edge=FR2_XYZ_WIDTH_EDGE,
        height_edge=FR2_XYZ_HEIGHT_EDGE,
        distortion=FR2_XYZ_DISTORTION,
        device="cuda:0",
        progress_every=50,
        _backend=backend,
        _turtle_record=turtle_record,
    )
    _, payload = _load_json(manifest, "TURTLE stream manifest")
    if payload.get("schema") != TURTLE_SCHEMA:
        raise ValueError("TURTLE materializer returned the wrong schema")
    if payload.get("selection", {}).get("emitted_source_indices") != audit[
        "selection"
    ]["motion_keyframes"]:
        raise ValueError("TURTLE emitted selection differs from frozen DROID keyframes")
    del backend, model
    torch.cuda.empty_cache()
    return manifest


def _export_pose_bundle_path(audit: Mapping[str, Any]) -> Path:
    scene = Path(audit["outputs"]["scene_dir"])
    return scene.with_name(f"{scene.name}_estimated_pose_bundle.npz")


def materialize_export_pose_bundle(
    audit: Mapping[str, Any], config: Mapping[str, Any]
) -> Path:
    """Create a minimal, immutable export view of the frozen DROID poses.

    The frozen selection archive intentionally includes negative provenance
    assertions such as ``reference_pose_arrays_present=false``.  The formal
    scene exporter rejects *any* archive key whose name advertises a reference
    sidecar, even when that key is a scalar false assertion.  Keep the frozen
    bytes untouched and derive a content-verifiable archive containing only
    the estimated pose array and the two provenance fields required by the
    exporter.
    """

    source = Path(audit["frozen_tracking"]["trajectory"]).expanduser().resolve()
    expected_source_sha = str(audit["frozen_tracking"]["trajectory_sha256"])
    if not source.is_file() or sha256_file(source) != expected_source_sha:
        raise ValueError("frozen trajectory changed before pose-bundle export")
    pose_key = str(config["tracking"]["pose_key"])
    expected_pose_source = str(config["tracking"]["pose_source"])
    with np.load(source, allow_pickle=False) as frozen:
        required = {
            pose_key,
            "pose_source",
            "uses_ground_truth_pose",
            "reference_pose_arrays_present",
        }
        missing = required - set(frozen.files)
        if missing:
            raise ValueError(
                f"frozen trajectory lacks provenance fields: {sorted(missing)}"
            )
        poses = np.asarray(frozen[pose_key]).copy()
        pose_source = str(np.asarray(frozen["pose_source"]).reshape(()).item())
        uses_gt = bool(
            np.asarray(frozen["uses_ground_truth_pose"]).reshape(()).item()
        )
        reference_present = bool(
            np.asarray(frozen["reference_pose_arrays_present"]).reshape(()).item()
        )
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or not np.isfinite(poses).all()
        or pose_source != expected_pose_source
        or uses_gt
        or reference_present
    ):
        raise ValueError("frozen trajectory failed minimal pose-bundle provenance gate")

    destination = _export_pose_bundle_path(audit)

    def verify_existing(path: Path) -> None:
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(f"invalid existing export pose bundle: {path}")
        with np.load(path, allow_pickle=False) as bundle:
            if set(bundle.files) != {
                pose_key,
                "pose_source",
                "uses_ground_truth_pose",
            }:
                raise ValueError("existing export pose bundle has unexpected fields")
            existing_poses = np.asarray(bundle[pose_key])
            existing_source = str(
                np.asarray(bundle["pose_source"]).reshape(()).item()
            )
            existing_uses_gt = bool(
                np.asarray(bundle["uses_ground_truth_pose"]).reshape(()).item()
            )
        if (
            not np.array_equal(existing_poses, poses)
            or existing_source != pose_source
            or existing_uses_gt
        ):
            raise ValueError("existing export pose bundle differs from frozen estimate")

    if destination.exists() or destination.is_symlink():
        verify_existing(destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(
                handle,
                **{
                    pose_key: poses,
                    "pose_source": np.asarray(pose_source),
                    "uses_ground_truth_pose": np.asarray(False),
                },
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            verify_existing(destination)
        verify_existing(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def export_resplat_scene(audit: Mapping[str, Any], config: Mapping[str, Any]) -> Path:
    turtle_manifest = Path(audit["outputs"]["turtle_dir"]) / "manifest.json"
    if not turtle_manifest.is_file():
        raise FileNotFoundError("TURTLE stream must be materialized before scene export")
    destination = Path(audit["outputs"]["scene_dir"])
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite ReSplat scene: {destination}")
    resplat = config["official_resplat"]
    pose_bundle = materialize_export_pose_bundle(audit, config)
    result = export_scene(
        frames_csv=Path(audit["frozen_tracking"]["frames_csv"]),
        output_dir=destination,
        selected_indices=audit["selection"]["motion_keyframes"],
        selection_provenance={
            "kind": "frozen_motion_only_protocol",
            "protocol_manifest": audit["protocol"]["path"],
            "protocol_manifest_sha256": audit["protocol"]["sha256"],
            "frozen_trajectory": audit["frozen_tracking"]["trajectory"],
            "frozen_trajectory_sha256": audit["frozen_tracking"][
                "trajectory_sha256"
            ],
            "derived_estimated_pose_bundle": str(pose_bundle),
            "derived_estimated_pose_bundle_sha256": sha256_file(pose_bundle),
            "clear_gt_membership_used": False,
        },
        image_mode="turtle",
        images_json=turtle_manifest,
        image_root=None,
        resplat_repo=Path(resplat["repository"]),
        model_preset=resplat["model_preset"],
        checkpoint=Path(resplat["checkpoint"]),
        expected_checkpoint_sha256=resplat["checkpoint_sha256"],
        trajectory_npz=pose_bundle,
        trajectory_key=config["tracking"]["pose_key"],
        formal_smoke=True,
    )
    _, payload = _load_json(result / "manifest.json", "ReSplat scene manifest")
    if payload.get("schema") != SCENE_SCHEMA or payload.get("selection", {}).get(
        "source_indices"
    ) != audit["selection"]["motion_keyframes"]:
        raise ValueError("exported scene selection/schema drifted")
    return result / "manifest.json"


def _run_logged(command: Sequence[str], log_path: Path, environment: Mapping[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        log.write("command=" + json.dumps(list(command)) + "\n")
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        code = int(process.wait())
        log.write(f"exit_code={code}\n")
    if code != 0:
        raise RuntimeError(f"official ReSplat runner failed with exit code {code}")


def run_official_resplat(audit: Mapping[str, Any], config: Mapping[str, Any]) -> Path:
    scene = Path(audit["outputs"]["scene_dir"])
    if not (scene / "manifest.json").is_file():
        raise FileNotFoundError("ReSplat scene must be exported before inference")
    destination = Path(audit["outputs"]["resplat_dir"])
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite official ReSplat output: {destination}")
    resplat = config["official_resplat"]
    command = [
        str(Path(os.path.expanduser(str(config["execution"]["resplat_python"])))),
        str(PAIRED_RUNNER),
        "--scene-path",
        str(scene),
        "--scene-manifest",
        str(scene / "manifest.json"),
        "--resplat-repo",
        str(Path(resplat["repository"]).expanduser().resolve()),
        "--checkpoint",
        str(Path(resplat["checkpoint"]).expanduser().resolve()),
        "--expected-checkpoint-sha256",
        resplat["checkpoint_sha256"],
        "--output-dir",
        str(destination),
        "--model-preset",
        resplat["model_preset"],
        "--device",
        "cuda:0",
        "--context-selection",
        "fps",
        "--expected-target-count",
        str(audit["selection"]["target_count"]),
        "--near",
        "0.01",
        "--far",
        "200.0",
        "--render-chunk-size",
        "4",
        "--max-save-images",
        str(audit["selection"]["target_count"]),
        "--save-ply",
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    log = destination.parent / f".{destination.name}.launch.log"
    _run_logged(command, log, environment)
    return destination / "run_manifest.json"


def _source_from_name(value: object) -> int:
    stem = Path(str(value)).stem
    if not stem.isdigit():
        raise ValueError(f"ReSplat image name is not a source index: {value!r}")
    return int(stem)


def finalize_audit(audit: Mapping[str, Any], config: Mapping[str, Any]) -> Path:
    turtle_path, turtle = _load_json(
        Path(audit["outputs"]["turtle_dir"]) / "manifest.json", "TURTLE manifest"
    )
    scene_path, scene = _load_json(
        Path(audit["outputs"]["scene_dir"]) / "manifest.json", "scene manifest"
    )
    run_path, run = _load_json(
        Path(audit["outputs"]["resplat_dir"]) / "run_manifest.json",
        "official ReSplat run manifest",
    )
    if turtle.get("schema") != TURTLE_SCHEMA or scene.get("schema") != SCENE_SCHEMA:
        raise ValueError("upstream output schema drifted")
    if run.get("schema") != RESPLAT_SCHEMA:
        raise ValueError("official ReSplat output schema drifted")
    context_sources = [
        _source_from_name(value) for value in run["selection"]["context_names"]
    ]
    target_sources = [
        _source_from_name(value) for value in run["selection"]["target_names"]
    ]
    if context_sources != audit["selection"]["expected_context_source_indices"]:
        raise ValueError(
            "official ReSplat FPS contexts do not match the frozen preflight: "
            f"{context_sources} != {audit['selection']['expected_context_source_indices']}"
        )
    if target_sources != audit["selection"]["expected_target_source_indices"]:
        raise ValueError("official ReSplat target set does not match the frozen preflight")
    destination = Path(audit["outputs"]["audit_dir"])
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite pipeline audit: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    installed = False
    try:
        payload = {
            **dict(audit),
            "preflight_only": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": {
                "estimated_pose_bundle": {
                    "path": str(_export_pose_bundle_path(audit)),
                    "sha256": sha256_file(_export_pose_bundle_path(audit)),
                    "derived_from_frozen_trajectory_sha256": audit[
                        "frozen_tracking"
                    ]["trajectory_sha256"],
                },
                "turtle_manifest": {
                    "path": str(turtle_path),
                    "sha256": sha256_file(turtle_path),
                },
                "scene_manifest": {"path": str(scene_path), "sha256": sha256_file(scene_path)},
                "resplat_run_manifest": {
                    "path": str(run_path),
                    "sha256": sha256_file(run_path),
                },
            },
            "postflight": {
                "official_fps_context_source_indices": context_sources,
                "target_source_indices": target_sources,
                "matches_frozen_protocol": True,
                "selection_independent": True,
                "26k_comparison_performed": False,
            },
            "metrics": run.get("metrics"),
        }
        manifest = staging / "pipeline_audit.json"
        manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        digest = sha256_file(manifest)
        (staging / "pipeline_audit.sha256").write_text(
            f"{digest}  pipeline_audit.json\n", encoding="utf-8"
        )
        os.rename(staging, destination)
        installed = True
    finally:
        if not installed and staging.exists():
            shutil.rmtree(staging)
    return destination / "pipeline_audit.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=("preflight", "turtle", "scene", "resplat", "audit", "all"),
        default="preflight",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _, config = _load_json(args.config, "pipeline config")
    audit = preflight(args.config)
    if args.stage == "preflight":
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0
    if args.stage in {"turtle", "all"}:
        print(f"TURTLE manifest: {materialize_turtle(audit, config)}")
    if args.stage in {"scene", "all"}:
        print(f"scene manifest: {export_resplat_scene(audit, config)}")
    if args.stage in {"resplat", "all"}:
        print(f"ReSplat run manifest: {run_official_resplat(audit, config)}")
    if args.stage in {"audit", "all"}:
        print(f"pipeline audit: {finalize_audit(audit, config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
