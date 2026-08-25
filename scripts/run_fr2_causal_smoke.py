#!/usr/bin/env python3
"""Preflight or launch the bounded fr2_xyz causal-EVSSM smoke matrix.

The default action is a CPU-only preflight of *all three* arms.  The bounded
221-frame run evaluates eleven published clear-GT frames and is explicitly
labelled ``clear_gt_prefix_smoke``; it is not a complete TUM paper metric.

Examples
--------
CPU-only validation::

    /srv/szha0669/unblur-slam/env/bin/python \
        scripts/run_fr2_causal_smoke.py --preflight

Launch one arm on physical GPU 1 after all-arm preflight::

    /srv/szha0669/unblur-slam/env/bin/python \
        scripts/run_fr2_causal_smoke.py --run causal --gpu 1

An arm root containing any file or directory is never overwritten.  Archive
failed output explicitly before relaunching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thirdparty.glorie_slam import config as config_io  # noqa: E402
from src.refinement.resplat_replay import validate_resplat_config  # noqa: E402
from src.utils.datasets import get_dataset  # noqa: E402
from src.utils.eval_frames import (  # noqa: E402
    PREFIX_SMOKE_METRIC_SCOPE,
    clear_gt_metric_scope,
    clear_gt_source_indices,
    validate_clear_gt_protocol_scope,
)


CONFIG_ROOT = REPO_ROOT / "configs" / "local" / "fr2_xyz_causal_smoke"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "unblur_slam.yaml"
OUTPUT_ROOT = Path(
    "/srv/szha0669/unblur-slam/slam_smoke/fr2_xyz_causal_smoke"
).resolve()
ARM_CONFIGS = {
    "baseline": CONFIG_ROOT / "baseline.yaml",
    "replay": CONFIG_ROOT / "replay.yaml",
    "causal": CONFIG_ROOT / "causal.yaml",
}

EXPECTED_FULL_PROTOCOL = (
    0, 9, 15, 49, 58, 72, 89, 109, 125, 166, 220, 319, 374, 407,
    435, 470, 483, 523, 568, 704, 750, 789, 827, 926, 1004, 1160,
    1251, 1342, 1409, 1460, 1553, 1692, 1795, 1889, 1978, 2055,
    2206, 2282, 2358, 2425, 2590, 2764,
)
EXPECTED_PREFIX = EXPECTED_FULL_PROTOCOL[:11]
EXPECTED_HASHES = {
    "droid": "46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526",
    "omnidata": "a0fab23fee64aa9e4bbe0b520b18b196ea7594a7f719c1d8c10cf11dcb6e4a1e",
    "evssm": "4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41",
    "causal": "fd593194319d99accddd5e6d6deb7694f9c98063830c6bd5997d8c1fcc85c78d",
}
_TEACHER_SCHEMA = "unblur_slam.video_deblur_teacher_provenance.v1"
_CAUSAL_FORMAT = "unblur_slam.causal_video_deblur.torchscript.v1"


def _load(arm: str) -> dict[str, Any]:
    return config_io.load_config(ARM_CONFIGS[arm], DEFAULT_CONFIG)


def _arm_root(cfg: Mapping[str, Any]) -> Path:
    return Path(cfg["data"]["output"]).expanduser().resolve()


def _scene_output(cfg: Mapping[str, Any]) -> Path:
    return (_arm_root(cfg) / str(cfg["scene"])).resolve()


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    digest = str(value).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _validate_artifact(path_value: object, sha_value: object, label: str) -> Path:
    if not path_value:
        raise ValueError(f"{label} path must be configured")
    path = Path(str(path_value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    expected = _require_sha256(sha_value, f"{label} configured SHA-256")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: configured {expected}, actual {actual}"
        )
    return path


def _flatten(value: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if isinstance(value, dict):
        flattened: dict[tuple[str, ...], object] = {}
        for key, child in value.items():
            if not prefix and str(key) == "inherit_from":
                continue
            flattened.update(_flatten(child, prefix + (str(key),)))
        return flattened
    return {prefix: value}


def _diff_paths(left: Mapping[str, Any], right: Mapping[str, Any]) -> set[str]:
    left_flat = _flatten(dict(left))
    right_flat = _flatten(dict(right))
    sentinel = object()
    return {
        ".".join(path)
        for path in set(left_flat) | set(right_flat)
        if left_flat.get(path, sentinel) != right_flat.get(path, sentinel)
    }


def _assert_static_matrix_contract(configs: Mapping[str, dict[str, Any]]) -> None:
    if set(configs) != set(ARM_CONFIGS):
        raise ValueError("the causal smoke must validate all three arms")

    expected_differences = {
        "replay": {
            "data.output",
            "mapping.resplat.enabled",
            "mapping.resplat.online_enabled",
            "mapping.resplat.extra_iters",
        },
        "causal": {
            "data.output",
            "mapping.resplat.enabled",
            "mapping.resplat.online_enabled",
            "mapping.resplat.extra_iters",
            "deblur.frontend",
            "deblur.causal_checkpoint",
            "deblur.causal_checkpoint_sha256",
        },
    }
    baseline = configs["baseline"]
    for arm, expected in expected_differences.items():
        actual = _diff_paths(baseline, configs[arm])
        if actual != expected:
            raise ValueError(
                f"{arm}: resolved config drift outside the treatment contract; "
                f"actual differences={sorted(actual)}, expected={sorted(expected)}"
            )

    expected_switches = {
        "baseline": ("evssm", False, False, 0, 100),
        "replay": ("evssm", True, True, 25, 75),
        "causal": ("causal_evssm", True, True, 25, 75),
    }
    outputs: list[Path] = []
    for arm, cfg in configs.items():
        if str(cfg.get("dataset", "")).lower() not in {"tumrgbd", "tumrgb"}:
            raise ValueError(f"{arm}: expected the TUM RGB-D dataset")
        if str(cfg.get("scene")) != "freiburg2_xyz":
            raise ValueError(f"{arm}: expected scene=freiburg2_xyz")
        common = (
            int(cfg["max_frames"]),
            int(cfg["stride"]),
            int(cfg["setup_seed"]),
            int(cfg["cam"]["W_out"]),
            int(cfg["cam"]["H_out"]),
            int(cfg["mapping"]["final_refine_iters"]),
            int(cfg["mapping"]["Training"]["init_itr_num"]),
            int(cfg["mapping"]["Training"]["mapping_itr_num"]),
            int(cfg["mapping"]["Training"]["tracking_itr_num"]),
        )
        if common != (221, 1, 43, 512, 384, 100, 100, 10, 10):
            raise ValueError(f"{arm}: shared data/compute contract drifted: {common}")
        if not bool(cfg.get("warmup_mapper", False)):
            raise ValueError(f"{arm}: warmup_mapper must be true")
        if bool(cfg.get("clear_init", True)):
            raise ValueError(f"{arm}: clear_init must remain false")
        if bool((cfg.get("framecrafter", {}) or {}).get("enabled", False)):
            raise ValueError(f"{arm}: FrameCrafter must be disabled")
        if bool((cfg.get("submaps", {}) or {}).get("enabled", False)):
            raise ValueError(f"{arm}: submaps must be disabled")
        if not bool(cfg["tracking"]["backend"].get("final_ba", False)):
            raise ValueError(f"{arm}: tracking.backend.final_ba must be true")
        if not bool(cfg["mapping"].get("hydrate_missing_droid_keyframes", False)):
            raise ValueError(f"{arm}: final-BA camera hydration must be enabled")

        smoke = cfg.get("causal_smoke", {}) or {}
        if (
            smoke.get("schema") != "unblur_slam.fr2_xyz_causal_smoke.v1"
            or smoke.get("full_paper_metric") is not False
            or int(smoke.get("source_first", -1)) != 0
            or int(smoke.get("source_last", -1)) != 220
            or tuple(smoke.get("expected_clear_gt_source_indices", ()))
            != EXPECTED_PREFIX
        ):
            raise ValueError(f"{arm}: bounded-smoke audit metadata is invalid")
        evaluation = cfg.get("evaluation", {}) or {}
        if (
            str(evaluation.get("clear_gt_scope")) != "prefix_smoke"
            or tuple(evaluation.get("expected_clear_gt_source_indices", ()))
            != EXPECTED_PREFIX
            or clear_gt_metric_scope(cfg) != PREFIX_SMOKE_METRIC_SCOPE
        ):
            raise ValueError(f"{arm}: prefix-smoke evaluation contract is invalid")

        replay = cfg["mapping"]["resplat"]
        validate_resplat_config(replay)
        frontend, enabled, online, replay_iters, uniform_iters = expected_switches[arm]
        actual_switches = (
            str(cfg["deblur"]["frontend"]).lower(),
            bool(replay["enabled"]),
            bool(replay["online_enabled"]),
            int(replay["extra_iters"]),
            int(cfg["mapping"]["final_refine_iters"]) - int(replay["extra_iters"]),
        )
        if actual_switches != (
            frontend, enabled, online, replay_iters, uniform_iters
        ):
            raise ValueError(
                f"{arm}: treatment switches {actual_switches} do not match "
                f"{expected_switches[arm]}"
            )
        if str(replay.get("budget_mode")) != "replace_tail":
            raise ValueError(f"{arm}: replay budget must use replace_tail")
        if int(replay.get("online_replay_views", -1)) != 2:
            raise ValueError(f"{arm}: online replay must replace exactly two views")
        if list(replay.get("checkpoint_steps", [])) != [25, 50, 100]:
            raise ValueError(f"{arm}: checkpoint steps must be 25/50/100")
        if int(replay.get("checkpoint_interval", -1)) != 25:
            raise ValueError(f"{arm}: checkpoint interval must be 25")

        deblur = cfg.get("deblur", {}) or {}
        if str(deblur.get("frontend", "")).lower() == "turtle_streaming":
            raise ValueError(f"{arm}: GoPro/TURTLE is forbidden in this matrix")
        for key in ("turtle_repo", "turtle_config", "turtle_checkpoint"):
            if str(deblur.get(key, "") or ""):
                raise ValueError(f"{arm}: active {key} is forbidden")

        expected_output = (OUTPUT_ROOT / arm).resolve()
        output = _arm_root(cfg)
        if output != expected_output:
            raise ValueError(f"{arm}: output {output} != {expected_output}")
        outputs.append(output)
    if len(set(outputs)) != len(ARM_CONFIGS):
        raise ValueError(f"arm outputs are not isolated: {outputs}")


def _validate_weight_contract(configs: Mapping[str, dict[str, Any]]) -> None:
    baseline = configs["baseline"]
    expected_config_hashes = {
        "droid": baseline["tracking"].get("pretrained_sha256"),
        "omnidata": baseline["mono_prior"].get("depth_pretrained_sha256"),
        "evssm": baseline.get("evssm_checkpoint_sha256"),
        "causal": configs["causal"]["deblur"].get("causal_checkpoint_sha256"),
    }
    for label, expected in EXPECTED_HASHES.items():
        if _require_sha256(expected_config_hashes[label], label) != expected:
            raise ValueError(f"{label}: config does not contain the pinned SHA-256")
    _validate_artifact(
        baseline["tracking"]["pretrained"], EXPECTED_HASHES["droid"], "DROID"
    )
    _validate_artifact(
        baseline["mono_prior"]["depth_pretrained"],
        EXPECTED_HASHES["omnidata"],
        "Omnidata",
    )
    _validate_artifact(
        baseline["evssm_checkpoint"], EXPECTED_HASHES["evssm"], "Unblur EVSSM"
    )


def _update_array_hash(digest: Any, label: str, value: object) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())


def _dataset_identity(dataset: object) -> str:
    """Content-address the resolved reader contract without decoding pixels."""

    digest = hashlib.sha256()
    for label in ("color_paths", "depth_paths", "gt_paths"):
        values = list(getattr(dataset, label))
        if len(values) != 221:
            raise ValueError(f"dataset {label} has {len(values)} entries, expected 221")
        records = []
        for value in values:
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"dataset {label} entry is missing: {path}")
            records.append((str(path), int(path.stat().st_size)))
        digest.update(label.encode("ascii") + b"\0")
        digest.update(json.dumps(records, separators=(",", ":")).encode("utf-8"))
    _update_array_hash(digest, "poses", getattr(dataset, "poses"))
    _update_array_hash(
        digest, "image_timestamps", getattr(dataset, "image_timestamps")
    )
    _update_array_hash(digest, "intrinsic", getattr(dataset, "intrinsic"))
    _update_array_hash(digest, "distortion", getattr(dataset, "distortion"))
    for source_index in range(221):
        metadata = dataset.frame_info(source_index)
        if (
            int(metadata.get("source_index", -1)) != source_index
            or bool(metadata.get("synthetic", True))
            or not bool(metadata.get("eval", False))
        ):
            raise ValueError(
                f"dataset frame {source_index} is not the matching original eval frame"
            )
    return digest.hexdigest()


def _validate_causal_export(cfg: dict[str, Any]) -> dict[str, Any]:
    deblur = cfg.get("deblur", {}) or {}
    if str(deblur.get("frontend", "")).lower() != "causal_evssm":
        raise ValueError("causal arm must use frontend=causal_evssm")
    runtime_evssm = _validate_artifact(
        cfg.get("evssm_checkpoint"),
        cfg.get("evssm_checkpoint_sha256"),
        "causal runtime Unblur EVSSM",
    )
    if _sha256_file(runtime_evssm) != EXPECTED_HASHES["evssm"]:
        raise ValueError("causal runtime EVSSM is not the pinned Unblur artifact")
    checkpoint = _validate_artifact(
        deblur.get("causal_checkpoint"),
        deblur.get("causal_checkpoint_sha256"),
        "causal EVSSM export",
    )
    if _sha256_file(checkpoint) != EXPECTED_HASHES["causal"]:
        raise ValueError("causal export is not the pinned smoke artifact")
    history = int(deblur.get("causal_history", 0))
    if history != 5 or not bool(deblur.get("stream_every_frame", False)):
        raise ValueError("causal arm requires history=5 and stream_every_frame=true")
    if not bool(deblur.get("stream_apply_to_tracking", False)):
        raise ValueError("causal arm must apply the stream to tracking")
    if bool(deblur.get("stream_replace_sharp", True)):
        raise ValueError("causal arm must not replace annotated sharp frames")
    for key, expected in (
        ("stream_min_laplacian_gain", 0.02),
        ("stream_min_vs_evssm_gain", 0.0),
    ):
        value = float(deblur.get(key, float("nan")))
        if not math.isfinite(value) or value != expected:
            raise ValueError(f"deblur.{key} must equal {expected}")

    extra_files = {"metadata.json": ""}
    model = torch.jit.load(str(checkpoint), map_location="cpu", _extra_files=extra_files)
    raw_metadata = extra_files["metadata.json"]
    if isinstance(raw_metadata, bytes):
        raw_metadata = raw_metadata.decode("utf-8")
    if not raw_metadata:
        raise ValueError("causal export is missing metadata.json")
    try:
        metadata = json.loads(raw_metadata)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("causal export has invalid metadata.json") from error
    if metadata.get("format") != _CAUSAL_FORMAT:
        raise ValueError("causal export format is not the supported v1 contract")
    model_config = metadata.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("causal export is missing model_config")
    if (
        int(model_config.get("max_history", 0)) != history
        or str(model_config.get("input_domain", "")).lower() != "evssm"
        or bool(model_config.get("use_teacher_input", True))
    ):
        raise ValueError("causal export model_config disagrees with runtime")

    training = metadata.get("training_contract")
    if not isinstance(training, dict) or (
        training.get("stream_prefix_padding") != "repeat_first_frame_on_left"
        or training.get("supervised_output")
        != "newest_frame_at_every_sequence_position"
    ):
        raise ValueError("causal export training/runtime prefix contract is invalid")

    provenance = metadata.get("teacher_provenance")
    if not isinstance(provenance, dict) or provenance.get("schema") != _TEACHER_SCHEMA:
        raise ValueError("causal export is missing teacher provenance")
    if (
        provenance.get("storage") != "precomputed_png_rgb8"
        or provenance.get("teacher_domain") != "evssm_restored_rgb_0_1"
        or provenance.get("teacher_artifacts_verified") is not True
        or _require_sha256(
            provenance.get("evssm_checkpoint_sha256"), "teacher EVSSM SHA-256"
        )
        != EXPECTED_HASHES["evssm"]
    ):
        raise ValueError("causal export teacher provenance is incompatible")
    configured_evssm = runtime_evssm
    metadata_evssm = Path(provenance.get("evssm_checkpoint", "")).expanduser().resolve()
    if metadata_evssm != configured_evssm:
        raise ValueError("causal export teacher path is not the runtime Unblur EVSSM")
    _validate_artifact(
        provenance.get("precompute_report"),
        provenance.get("precompute_report_sha256"),
        "causal teacher precompute report",
    )
    _validate_artifact(
        provenance.get("teacher_manifest"),
        provenance.get("teacher_manifest_sha256"),
        "causal teacher manifest",
    )

    with torch.no_grad():
        probe = torch.zeros(1, history, 3, 16, 16)
        output = model(probe)
    if (
        not torch.is_tensor(output)
        or tuple(output.shape) != (1, 3, 16, 16)
        or not bool(torch.isfinite(output).all())
    ):
        raise ValueError("causal export failed the finite CPU BTCHW contract")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "evssm_sha256": EXPECTED_HASHES["evssm"],
        "history": history,
        "teacher_storage": provenance["storage"],
    }


def preflight() -> dict[str, dict[str, Any]]:
    """Validate all arms on CPU without starting a SLAM/GPU worker."""

    configs = {arm: _load(arm) for arm in ARM_CONFIGS}
    _assert_static_matrix_contract(configs)
    _validate_weight_contract(configs)

    input_roots = set()
    identities: dict[str, str] = {}
    previous_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        for arm, cfg in configs.items():
            input_root = (
                Path(cfg["data"]["dataset_root"]).expanduser()
                / str(cfg["data"]["input_folder"])
            ).resolve()
            for filename in ("rgb.txt", "depth.txt", "groundtruth.txt"):
                if not (input_root / filename).is_file():
                    raise FileNotFoundError(
                        f"{arm}: TUM input is missing {input_root / filename}"
                    )
            input_roots.add(input_root)

            dataset = get_dataset(cfg, device="cpu")
            if len(dataset) != 221:
                raise ValueError(f"{arm}: dataset has {len(dataset)} frames, expected 221")
            protocol = clear_gt_source_indices(cfg, dataset)
            if protocol is None or tuple(sorted(protocol)) != EXPECTED_FULL_PROTOCOL:
                raise ValueError(f"{arm}: published fr2 clear-GT protocol drifted")
            available = validate_clear_gt_protocol_scope(cfg, dataset)
            if available is None or tuple(sorted(available)) != EXPECTED_PREFIX:
                raise ValueError(f"{arm}: expected the exact eleven-frame prefix")
            identity = _dataset_identity(dataset)
            identities[arm] = identity
            replay = cfg["mapping"]["resplat"]
            uniform = int(cfg["mapping"]["final_refine_iters"]) - (
                int(replay["extra_iters"]) if bool(replay["enabled"]) else 0
            )
            priority = int(replay["extra_iters"]) if bool(replay["enabled"]) else 0
            print(
                f"[preflight] {arm}: source=221 (0..220), "
                f"clear_gt_prefix=11, metric_scope={PREFIX_SMOKE_METRIC_SCOPE}, "
                f"resolution=512x384, refine={uniform}+{priority}=100, "
                f"data_sha256={identity[:12]}, output={_arm_root(cfg)}"
            )
    finally:
        os.chdir(previous_cwd)

    if len(input_roots) != 1:
        raise ValueError(f"arms resolved different input roots: {input_roots}")
    if len(set(identities.values())) != 1:
        raise ValueError(f"arms resolved different reader data: {identities}")
    causal_audit = _validate_causal_export(configs["causal"])
    print(
        "[preflight] causal provenance: official Unblur EVSSM "
        f"{causal_audit['evssm_sha256'][:12]}, export "
        f"{causal_audit['checkpoint_sha256'][:12]}, history=5, "
        f"teacher={causal_audit['teacher_storage']}"
    )
    return configs


def _assert_output_available(cfg: Mapping[str, Any]) -> None:
    arm_root = _arm_root(cfg)
    if arm_root.exists() and any(arm_root.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty smoke arm root: {arm_root}"
        )
    scene_output = _scene_output(cfg)
    if scene_output.exists():
        raise FileExistsError(
            f"refusing to reuse an existing scene output: {scene_output}"
        )


def _interrupt_process_group(process: subprocess.Popen[str], log: Any) -> int:
    """Forward interruption to the complete spawned SLAM process group."""

    log.write("\n[launcher] KeyboardInterrupt; forwarding SIGINT\n")
    log.flush()
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
    try:
        return int(process.wait(timeout=30))
    except subprocess.TimeoutExpired:
        log.write("[launcher] SIGINT timeout; forwarding SIGTERM\n")
        log.flush()
        os.killpg(process.pid, signal.SIGTERM)
    try:
        return int(process.wait(timeout=15))
    except subprocess.TimeoutExpired:
        log.write("[launcher] SIGTERM timeout; forwarding SIGKILL\n")
        log.flush()
        os.killpg(process.pid, signal.SIGKILL)
        return int(process.wait())


def run_arm(arm: str, gpu: str) -> int:
    if not re.fullmatch(r"\d+", str(gpu)):
        raise ValueError("--gpu must be one non-negative physical CUDA index")
    configs = preflight()  # Always validate all three arms before one launch.
    cfg = configs[arm]
    _assert_output_available(cfg)

    arm_root = _arm_root(cfg)
    arm_root.mkdir(parents=True, exist_ok=True)
    log_path = arm_root / "launch.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["UNBLUR_SKIP_NR_IQA"] = "1"
    command = [sys.executable, str(REPO_ROOT / "run.py"), str(ARM_CONFIGS[arm])]
    print(f"[launch] CUDA_VISIBLE_DEVICES={gpu} {' '.join(command)}")
    print(f"[launch] persistent log={log_path}")

    # Exclusive creation is the atomic overwrite guard for the persistent log.
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        log.write(
            "[launcher] fr2_xyz bounded causal smoke\n"
            f"[launcher] arm={arm}\n"
            f"[launcher] physical_gpu={gpu}; process_device=cuda:0\n"
            f"[launcher] metric_scope={PREFIX_SMOKE_METRIC_SCOPE}; "
            "not_a_complete_paper_metric=true\n"
            f"[launcher] command={' '.join(command)}\n"
        )
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            return_code = int(process.wait())
        except KeyboardInterrupt:
            return_code = _interrupt_process_group(process, log)
        log.write(f"\n[launcher] exit_code={return_code}\n")
        log.flush()
        return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--preflight",
        action="store_true",
        help="CPU-only all-arm validation (default; starts no SLAM worker)",
    )
    action.add_argument("--run", choices=sorted(ARM_CONFIGS), help="launch one arm")
    parser.add_argument(
        "--gpu",
        default="0",
        help="physical CUDA device exposed to run.py as cuda:0",
    )
    args = parser.parse_args()
    if args.run:
        return run_arm(args.run, args.gpu)
    preflight()
    print("[preflight] PASS: all three arms validated; no GPU worker was started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
