#!/usr/bin/env python3
"""Fail-closed launch preflight for the preregistered TURTLE DPDD v4 ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping


SCHEMA = "unblur_slam.turtle_replica424_dpdd_png16_mixed_validation_only.v4"
DATASET_MANIFEST_SHA256 = (
    "68c787fdc61222701b3d63359a7be767a03aec8d22588e566e1140b58e9be4c9"
)
MATERIALIZATION_AUDIT_SHA256 = (
    "f3ad4488a5c7cedb5742126e363e2179af0cde3497c357b82ecbfd3712fecbc2"
)
TRAIN_MANIFEST_SHA256 = (
    "a2a97790de739b9d59efea0fc811255618e33b32ba0e2874e25f35c1f66c933c"
)
VALIDATION_MANIFEST_SHA256 = (
    "926b45f3717cf5d99c59ce04ce7e78c320419c72530fb6c690b7c6fe14660712"
)
REPOSITORY = "JacobLinCool/DPDD"
REVISION = "52e4035a045ea1763313b9ce2b27cf2e620cfc30"
TURTLE_COMMIT = "7094f4221b64ad0962b4f27ff1b76d788836e804"
TURTLE_CONFIG_SHA256 = "123b07de8d3f329769562e2f943e08fdf86c576c405634bad199ced95b25aa23"
TURTLE_ARCH_SHA256 = "4d19c676f92574dbad493eb591312fdeaf2b3b519f57410af2ed95fdbef5f058"
TURTLE_CHECKPOINT_SHA256 = "10334b3e81d0416bcde5ccaca960dc81dbfb5b6d23e53fadaf7896d72b580c82"
EVSSM_CHECKPOINT_SHA256 = "4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41"
REPLICA_TRAIN_SHA256 = "bd7caa189374683c8ffd7e8fce83cb62e5f69b73f6048808c4808dc2b4ecd2ba"
REPLICA_VALIDATION_SHA256 = "1aa8cc7a01b82c7d759c3db70e6c7e796a26d09398f3a1fd1592d787db9f886b"
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
LPIPS_LINEAR_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
TEST_METADATA_EXPOSURE = (
    "filenames_lfs_oids_sizes_split_aggregate_row0_url_and_manifest_text_seen_before_freeze"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_sha(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"missing {label}: {path}")
    actual = sha256_file(path)
    require(actual == expected, f"{label} SHA256 mismatch: {actual}")


def check_gpu1_idle() -> Mapping[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 4, "unexpected nvidia-smi row")
        rows.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_used_mib": int(fields[2]),
                "utilization_percent": int(fields[3]),
            }
        )
    matches = [row for row in rows if row["index"] == 1]
    require(len(matches) == 1, "physical GPU1 is missing or ambiguous")
    gpu = matches[0]
    require(gpu["name"] == "NVIDIA RTX A6000", "physical GPU1 model changed")
    require(gpu["memory_used_mib"] <= 64, "physical GPU1 is not idle")
    require(gpu["utilization_percent"] <= 5, "physical GPU1 utilization is not idle")
    return gpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_path = args.contract.expanduser().resolve()
    check_sha(contract_path, args.expected_contract_sha256.lower(), "preregistered contract")
    contract = load_json(contract_path)
    require(contract.get("schema") == SCHEMA, "contract schema mismatch")
    require(
        contract.get("status")
        == "preregistered_before_any_mixed_gpu_training_or_dpdd_model_evaluation",
        "contract status mismatch",
    )
    require(contract.get("seeds") == [17, 42, 73], "seed contract changed")
    require(set(contract.get("arms", {})) == {"G", "V", "S", "M"}, "arm matrix changed")
    training = contract.get("training", {})
    require(training.get("fixed_terminal_only") is True, "checkpoint rule changed")
    require(training.get("validation_during_training") is False, "training may read validation")
    require(training.get("optimizer_steps_per_trained_arm") == 78, "step budget changed")
    require(training.get("attempted_optimizer_steps") == 78, "attempted step budget changed")
    require(training.get("executed_optimizer_steps") == 78, "executed step budget changed")
    require(training.get("amp_skipped_optimizer_steps") == 0, "AMP skip contract changed")
    require(training.get("crop_size") == 128 and training.get("dpdd_batch_size") == 5, "batch/crop changed")
    require(training.get("amp") is True, "AMP contract changed")
    require(training.get("mixed_step") == "two_backward_one_joint_step", "mixed-step contract changed")

    evaluation = contract.get("evaluation", {})
    dpdd_eval = evaluation.get("dpdd_validation", {})
    require(dpdd_eval.get("arms_per_seed") == ["G", "V", "S", "M"], "DPDD eval arms changed")
    require(dpdd_eval.get("precision") == "CUDA_FP16", "DPDD eval precision changed")
    require(dpdd_eval.get("warmup_unmeasured_calls_per_arm") == 1, "warmup changed")
    replica_eval = evaluation.get("replica_temporal_validation", {})
    require(len(replica_eval.get("controls", [])) == 5, "temporal controls changed")
    require(replica_eval.get("ordered_replay_max_abs") == 0.000001, "replay gate changed")
    gates = contract.get("preregistered_validation_gates", {})
    require(gates.get("aggregation_contract", {}).get("all_gates_conjunctive") is True, "aggregation gate changed")
    require(
        gates.get("useful_history", {}).get("interaction_formula")
        == "(M.normal_psnr-M.reset_psnr)-(S.normal_psnr-S.reset_psnr), evaluated on the exact same steady frames",
        "interaction formula changed",
    )
    require(
        gates.get("replay_contract", {}).get(
            "ordered_replay_max_abs_over_every_pixel_frame_arm_and_seed_max"
        )
        == 0.000001,
        "terminal replay contract changed",
    )

    data = contract.get("data", {}).get("dpdd", {})
    require(data.get("repository") == REPOSITORY, "contract DPDD repository changed")
    require(data.get("revision") == REVISION, "contract DPDD revision changed")
    require(data.get("materializer_test_requests_made") == 0, "test requests must be zero")
    require(
        data.get("materializer_test_metadata_exposure") == TEST_METADATA_EXPOSURE,
        "contract test metadata disclosure changed",
    )
    require(data.get("train_manifest_sha256") == TRAIN_MANIFEST_SHA256, "train hash changed")
    require(
        data.get("validation_manifest_sha256") == VALIDATION_MANIFEST_SHA256,
        "validation hash changed",
    )

    dataset_path = Path(data["dataset_manifest"]).resolve()
    audit_path = Path(data["materialization_audit"]).resolve()
    check_sha(dataset_path, DATASET_MANIFEST_SHA256, "DPDD dataset manifest")
    check_sha(audit_path, MATERIALIZATION_AUDIT_SHA256, "DPDD materialization audit")
    check_sha(Path(data["train_manifest"]), TRAIN_MANIFEST_SHA256, "DPDD train manifest")
    check_sha(
        Path(data["validation_manifest"]),
        VALIDATION_MANIFEST_SHA256,
        "DPDD validation manifest",
    )

    dataset = load_json(dataset_path)
    require(dataset.get("repository") == REPOSITORY, "dataset repository mismatch")
    require(dataset.get("revision") == REVISION, "dataset revision mismatch")
    require(dataset.get("splits") == {"train": 350, "validation": 74}, "split counts changed")
    require(dataset.get("asset_count") == 848, "asset count changed")
    disclosure = dataset.get("test_disclosure", {})
    require(disclosure.get("requests_made_by_this_materializer") == 0, "materializer touched test")
    require(disclosure.get("metadata_pristine") is False, "test metadata pristine flag changed")
    require(disclosure.get("metadata_exposure") == TEST_METADATA_EXPOSURE, "metadata exposure changed")
    require(disclosure.get("pixels_opened") is False, "test pixels were opened")
    require(disclosure.get("images_decoded") is False, "test images were decoded")
    require(disclosure.get("metrics_opened") is False, "test metrics were opened")
    require(disclosure.get("split_supported_by_this_materializer") is False, "test split became supported")
    distribution = dataset.get("distribution", {})
    require(distribution.get("dataset_card_declared_license") == "mit", "license claim changed")
    require(bool(distribution.get("license_scope_warning")), "license warning is missing")

    audit = load_json(audit_path)
    require(audit.get("status") == "pass", "materialization audit did not pass")
    require(audit.get("pair_count") == 424 and audit.get("asset_count") == 848, "audit counts changed")
    require(audit.get("asset_bytes") == 7243127232, "audit byte count changed")
    require(
        audit.get("image_contract", {}).get("unique_sizes") == [[1680, 1120]],
        "DPDD dimensions changed",
    )
    require(audit.get("image_contract", {}).get("all_png_ihdr_16bit_rgb") is True, "PNG16 audit failed")
    require(audit.get("image_contract", {}).get("all_opencv_uint16_hwc3") is True, "decode audit failed")
    disjoint = audit.get("disjoint_audit", {})
    require(all(disjoint.values()) and len(disjoint) == 4, "path/content disjoint audit failed")
    test_audit = audit.get("test_audit", {})
    require(test_audit.get("local_test_paths") == 0, "local test paths appeared")
    require(test_audit.get("network_requests_by_auditor") == 0, "offline auditor used network")

    repo_root = Path(__file__).resolve().parents[1]
    pin_paths = {
        "launch_preflight_sha256": Path(__file__).resolve(),
        "mixed_trainer_sha256": repo_root / "scripts/train_turtle_mixed_defocus.py",
        "turtle_single_image_evaluator_sha256": repo_root / "scripts/evaluate_turtle_single_image_defocus.py",
        "evssm_validation_evaluator_sha256": repo_root / "scripts/evaluate_evssm_dpdd_validation.py",
        "turtle_streaming_evaluator_sha256": repo_root / "scripts/evaluate_turtle_streaming.py",
        "turtle_backend_sha256": repo_root / "src/turtle_backend.py",
        "evssm_backend_sha256": repo_root / "src/deblur_backends.py",
        "materializer_sha256": repo_root / "scripts/materialize_dpdd_hf_png16.py",
        "materialization_auditor_sha256": repo_root / "scripts/audit_dpdd_hf_png16_materialization.py",
        "mixed_trainer_test_sha256": repo_root / "tests/test_turtle_mixed_defocus_training.py",
        "turtle_single_image_evaluator_test_sha256": repo_root / "tests/test_turtle_single_image_defocus_eval.py",
        "evssm_evaluator_test_sha256": repo_root / "tests/test_evssm_dpdd_validation.py",
        "materializer_test_sha256": repo_root / "tests/test_materialize_dpdd_hf_png16.py",
        "materialization_auditor_test_sha256": repo_root / "tests/test_audit_dpdd_hf_png16_materialization.py",
    }
    pins = contract.get("implementation_pins", {})
    for name, path in pin_paths.items():
        check_sha(path, pins.get(name, ""), name)

    model_contract = contract.get("model", {})
    turtle_repo = Path(model_contract["repo"]).resolve()
    require(turtle_repo.is_dir(), "TURTLE repository is missing")
    head = subprocess.run(
        ["git", "-C", str(turtle_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(head == TURTLE_COMMIT and model_contract.get("repo_commit") == head, "TURTLE commit changed")
    tracked_status = subprocess.run(
        ["git", "-C", str(turtle_repo), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(not tracked_status, "TURTLE tracked tree is dirty")
    check_sha(Path(model_contract["config"]), TURTLE_CONFIG_SHA256, "TURTLE config")
    check_sha(
        turtle_repo / "basicsr/models/archs/turtle_t1_arch.py",
        TURTLE_ARCH_SHA256,
        "TURTLE architecture",
    )
    check_sha(
        Path(model_contract["official_gopro_checkpoint"]),
        TURTLE_CHECKPOINT_SHA256,
        "TURTLE GoPro checkpoint",
    )
    evssm = evaluation.get("official_evssm_reference", {})
    check_sha(Path(evssm["checkpoint"]), EVSSM_CHECKPOINT_SHA256, "official EVSSM checkpoint")
    replica_train = contract.get("data", {}).get("replica_train", {})
    replica_validation = contract.get("data", {}).get("replica_validation", {})
    check_sha(Path(replica_train["manifest"]), REPLICA_TRAIN_SHA256, "Replica train manifest")
    check_sha(
        Path(replica_validation["manifest"]),
        REPLICA_VALIDATION_SHA256,
        "Replica validation manifest",
    )
    check_sha(
        Path("/home/szha0669/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"),
        ALEXNET_SHA256,
        "LPIPS AlexNet backbone",
    )
    check_sha(
        Path(
            "/srv/szha0669/unblur-slam/env/lib/python3.10/site-packages/torchmetrics/functional/image/lpips_models/alex.pth"
        ),
        LPIPS_LINEAR_SHA256,
        "LPIPS AlexNet linear weights",
    )
    import cv2
    import torch
    import torchmetrics

    require(torch.__version__ == pins.get("torch_version"), "torch version changed")
    require(torch.version.cuda == pins.get("cuda_version"), "CUDA version changed")
    require(torchmetrics.__version__ == pins.get("torchmetrics_version"), "torchmetrics version changed")
    require(cv2.__version__ == pins.get("opencv_version"), "OpenCV version changed")

    runtime = contract.get("runtime", {})
    require(runtime.get("physical_gpu") == 1, "physical GPU contract changed")
    require(runtime.get("visible_device") == "CUDA_VISIBLE_DEVICES=1", "visibility contract changed")
    require(runtime.get("script_device") == "cuda:0", "logical CUDA device changed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "preflight requires CUDA_VISIBLE_DEVICES=1")
    output_root = Path(runtime["output_root"]).resolve()
    require(not output_root.exists(), f"fresh output root already exists: {output_root}")
    gpu = check_gpu1_idle()
    print(
        json.dumps(
            {
                "schema": "unblur_slam.turtle_dpdd_mixed_v4_preflight.v1",
                "status": "pass",
                "contract": str(contract_path),
                "contract_sha256": sha256_file(contract_path),
                "output_root_absent": True,
                "dpdd_test_requests": 0,
                "dpdd_test_pixels_opened": False,
                "gpu": gpu,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
