#!/usr/bin/env python3
"""Standard-library contracts for the BSD+DPDD TURTLE validation-only study.

This module intentionally never imports torch, decodes an image, invokes a GPU,
or accepts a test manifest.  It validates protocol invariants and, once the BSD
materialization exists, reads only JSON/JSONL metadata plus filesystem stats for
the explicitly bound train and validation assets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "unblur_slam.turtle_bsd_dpdd_causal_validation_only.v1"
BOUND_STATUS = "preregistered_before_any_bsd_or_dpdd_training_or_bsd_model_evaluation"
TEMPLATE_STATUS = "protocol_preregistered_data_binding_pending_no_launch"
REFERENCE_BOUND_STATUS = "bsd_validation_references_preregistered_training_blocked"
BSD_SEQUENCE_SCHEMA = "unblur_slam.bsd_paired_video_sequence.v1"
BSD_DATASET_SCHEMA = "unblur_slam.bsd_materialization.v1"
BSD_AUDIT_SCHEMA = "unblur_slam.bsd_materialization_audit.v1"
EXPOSURES = ("3ms24ms",)
SEEDS = (17, 42, 73)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UNBOUND_PREFIX = "__BIND_"
FORBIDDEN_SPLIT_COMPONENTS = frozenset({"test", "testing", "benchmark_test"})

# Every Python source that can affect materialization, model construction,
# training, evaluation, reporting, planning, or execution is content-addressed
# in the frozen contract.  This single map is shared by CPU preflight and the
# GPU executor so a dependency cannot be checked under one name and executed
# through another path.
IMPLEMENTATION_PIN_PATHS = {
    "official_bsd_backend_sha256": "src/turtle_official_bsd_backend.py",
    "contract_library_sha256": "scripts/bsd_dpdd_contract.py",
    "cpu_preflight_sha256": "scripts/preflight_turtle_bsd_dpdd_v1.py",
    "plan_runner_sha256": "scripts/run_turtle_bsd_dpdd_v1.py",
    "B_BD_trainer_sha256": "scripts/train_turtle_bsd_dpdd.py",
    "streaming_trainer_dependency_sha256": "scripts/train_turtle_streaming.py",
    "mixed_defocus_trainer_dependency_sha256": "scripts/train_turtle_mixed_defocus.py",
    "bsd_streaming_evaluator_sha256": "scripts/evaluate_turtle_bsd_streaming.py",
    "bsd_dpdd_evaluator_sha256": "scripts/evaluate_turtle_bsd_dpdd.py",
    "cpu_contract_test_sha256": "tests/test_turtle_bsd_dpdd_v1_contract.py",
    "runtime_executor_contract_test_sha256": "tests/test_bsd_dpdd_runtime_executor_v4.py",
    "evaluator_reporter_contract_test_sha256": "tests/test_bsd_v4_evaluator_reporter.py",
    "evssm_dpdd_contract_test_sha256": "tests/test_evssm_dpdd_validation.py",
    "existing_streaming_evaluator_sha256": "scripts/evaluate_turtle_streaming.py",
    "existing_single_image_evaluator_sha256": "scripts/evaluate_turtle_single_image_defocus.py",
    "existing_evssm_dpdd_evaluator_sha256": "scripts/evaluate_evssm_dpdd_validation.py",
    "existing_evssm_video_precompute_sha256": "scripts/precompute_video_deblur_evssm.py",
    "evssm_builder_dependency_sha256": "scripts/precompute_framecrafter_evssm.py",
    "framecrafter_pipeline_import_dependency_sha256": "src/framecrafter_pipeline.py",
    "evssm_architecture_dependency_sha256": "thirdparty/EVSSM/models/EVSSM.py",
    "evssm_backend_dependency_sha256": "src/deblur_backends.py",
    "video_dataset_dependency_sha256": "src/video_deblur/dataset.py",
    "bsd_acquisition_dependency_sha256": "scripts/acquire_bsd_3ms24ms.py",
    "full_bsd_materializer_sha256": "scripts/materialize_bsd_3ms24ms.py",
    "validation_prefix_materializer_sha256": "scripts/materialize_bsd_3ms24ms_validation_prefix.py",
    "reference_contract_binder_sha256": "scripts/bind_bsd_validation_reference_contract.py",
    "runtime_guard_sha256": "scripts/bsd_dpdd_runtime.py",
    "true_executor_sha256": "scripts/execute_turtle_bsd_dpdd_v1.py",
    "final_reporter_sha256": "scripts/report_turtle_bsd_dpdd_v1.py",
    "direct_float_evssm_bsd_evaluator_sha256": "scripts/evaluate_evssm_bsd_validation.py",
    "turtle_backend_shared_autocast_sha256": "src/turtle_backend.py",
}


class ContractError(RuntimeError):
    """Raised before launch when a frozen study invariant is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path | str) -> str:
    candidate = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_sha256(value: Any, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise ContractError(f"{label} must be one lowercase SHA256")
    return normalized


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_code_bundle_declaration(contract: Mapping[str, Any]) -> Mapping[str, str]:
    """Validate the frozen source inventory without opening any source file."""

    pins = contract.get("implementation_pins")
    require(isinstance(pins, Mapping), "implementation pins are missing")
    require(
        set(pins) == set(IMPLEMENTATION_PIN_PATHS),
        "implementation pin inventory changed",
    )
    normalized = {
        key: normalized_sha256(pins[key], label=f"implementation pin {key}")
        for key in IMPLEMENTATION_PIN_PATHS
    }
    bundle = contract.get("code_bundle")
    require(isinstance(bundle, Mapping), "frozen code bundle is missing")
    _require_exact(bundle, "schema", "unblur_slam.bsd_dpdd_code_bundle.v1")
    _require_exact(bundle, "root", str(Path(__file__).resolve().parents[1]))
    _require_exact(bundle, "hash_algorithm", "sha256")
    _require_exact(
        bundle,
        "canonical_digest_algorithm",
        "sha256_canonical_json_sorted_keys",
    )
    files = bundle.get("files")
    expected_files = {
        key: str((Path(__file__).resolve().parents[1] / relative).resolve())
        for key, relative in IMPLEMENTATION_PIN_PATHS.items()
    }
    require(
        isinstance(files, Mapping) and files == expected_files,
        "code-bundle path inventory differs from implementation pins",
    )
    rows = [
        {"pin": key, "path": expected_files[key], "sha256": normalized[key]}
        for key in sorted(expected_files)
    ]
    require(
        bundle.get("bundle_sha256") == canonical_json_sha256(rows),
        "code-bundle aggregate SHA256 changed",
    )
    _require_exact(bundle, "file_count", len(expected_files))
    _require_exact(bundle, "executor_rehash_inside_global_lock_before_cuda_or_pixels", True)
    _require_exact(bundle, "preflight_rehash", True)
    return normalized


def load_json_object(path: Path | str) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ContractError(f"missing JSON artifact: {candidate}") from error
    except json.JSONDecodeError as error:
        raise ContractError(f"invalid JSON artifact: {candidate}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON artifact must contain an object: {candidate}")
    return value


def load_contract(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).expanduser().resolve()
    require(candidate.is_file(), f"contract does not exist: {candidate}")
    digest = sha256_file(candidate)
    if expected_sha256 is not None:
        require(
            digest == normalized_sha256(expected_sha256, label="contract SHA256"),
            f"contract SHA256 mismatch: {digest}",
        )
    contract = load_json_object(candidate)
    return candidate, contract, digest


def find_unbound_values(value: Any, *, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.extend(find_unbound_values(child, prefix=f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_unbound_values(child, prefix=f"{prefix}[{index}]"))
    elif isinstance(value, str) and value.startswith(UNBOUND_PREFIX):
        found.append(prefix)
    return found


def _require_exact(mapping: Mapping[str, Any], key: str, expected: Any) -> None:
    actual = mapping.get(key)
    require(actual == expected, f"contract field {key!r} changed: {actual!r}")


def validate_protocol(
    contract: Mapping[str, Any],
    *,
    allow_template: bool,
    reference_only: bool = False,
) -> list[str]:
    """Validate every result-sensitive rule without opening any dataset pixels."""

    _require_exact(contract, "schema", SCHEMA)
    status = contract.get("status")
    require(
        not (allow_template and reference_only),
        "template and reference-only modes are mutually exclusive",
    )
    allowed_status = (
        TEMPLATE_STATUS
        if allow_template
        else REFERENCE_BOUND_STATUS
        if reference_only
        else BOUND_STATUS
    )
    require(status == allowed_status, f"contract status must be {allowed_status!r}")
    if reference_only:
        require(contract.get("launch_authorized") is False, "reference contract must block training")
        require(
            contract.get("reference_launch_authorized") is True,
            "reference contract did not authorize validation-only E/G/O",
        )
    else:
        require(
            contract.get("launch_authorized") is (not allow_template),
            "launch_authorized disagrees with template/bound mode",
        )
        require(
            contract.get("reference_launch_authorized") is (not allow_template),
            "reference launch authorization disagrees with mode",
        )
    _require_exact(contract, "seeds", list(SEEDS))
    arms = contract.get("arms")
    require(isinstance(arms, Mapping) and set(arms) == {"E", "G", "O", "B", "BD"}, "arm matrix changed")
    _require_exact(arms["E"], "architecture", "EVSSM")
    _require_exact(arms["G"], "architecture", "TURTLE_t1")
    _require_exact(arms["O"], "architecture", "TURTLE_t0")
    for arm in ("B", "BD"):
        _require_exact(arms[arm], "architecture", "TURTLE_t1")

    models = contract.get("models")
    require(isinstance(models, Mapping), "model identities are missing")
    evssm = models.get("evssm_E")
    require(isinstance(evssm, Mapping), "E model identity is missing")
    for key, expected in {
        "checkpoint_sha256": "4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41",
        "architecture_sha256": "d28b59d22be7b5a9d2e0247062b25142d62fc9fff16de66963f2e001620accd4",
        "backend_source_sha256": "7638d98c8af8baca5d7f414ee2a13e9612d7c1c7b4dda8d6b497bdeb5a8f3b68",
        "builder_source_sha256": "69b918137859b1131bdf4bc5490879cba4370a01e0254c46321e26b9aec4f5a9",
        "strict_state_load": True,
        "inference_precision": "CUDA_FP32",
    }.items():
        _require_exact(evssm, key, expected)
    _require_exact(
        evssm,
        "import_graph",
        {
            "entry_class": "thirdparty.EVSSM.models.EVSSM.EVSSM",
            "backend_class": "src.deblur_backends.EVSSMBackend",
            "checkpoint_builder": "scripts.precompute_framecrafter_evssm.build_evssm_inference",
            "source_files_content_pinned": True,
            "required_runtime_modules": [
                "torch",
                "einops",
                "mamba_ssm",
                "torchvision",
                "numpy",
            ],
        },
    )

    comparison = contract.get("comparison_contract")
    require(isinstance(comparison, Mapping), "comparison contract is missing")
    _require_exact(
        comparison,
        "same_architecture_primary",
        ["B_minus_G", "BD_minus_B", "BD_minus_G"],
    )
    _require_exact(
        comparison,
        "raw_common_baseline",
        {
            "identity": "the exact paired BSD blurry RGB input scored against its paired sharp RGB target",
            "model": "none",
            "state": "none",
            "required_in_arms": ["E", "G", "O", "B", "BD"],
            "per_frame_identity_and_metric_consistency_required": True,
            "metric_max_abs_tolerance": 1e-12,
            "reported_deltas": [
                "E_minus_raw",
                "G_minus_raw",
                "O_minus_raw",
                "B_minus_raw",
                "BD_minus_raw",
            ],
        },
    )
    require("not FLOP matched" in str(comparison.get("compute_disclosure")), "compute disclosure changed")

    data = contract.get("data")
    require(isinstance(data, Mapping), "data contract is missing")
    bsd = data.get("bsd")
    require(isinstance(bsd, Mapping), "BSD contract is missing")
    _require_exact(bsd, "blur_origin", "real_camera_long_exposure")
    _require_exact(bsd, "synthetic_high_fps_average", False)
    _require_exact(bsd, "frequency_fps", 15)
    _require_exact(bsd, "resolution_width_height", [640, 480])
    _require_exact(bsd, "exposure_setting", "3ms24ms")
    _require_exact(
        bsd,
        "official_split_facts",
        {
            "train_sequences_declared": 60,
            "train_pairs_declared": 6000,
            "validation_sequences": 20,
            "validation_pairs": 2000,
            "sealed_test_sequences_declared_only": 20,
            "sealed_test_pairs_declared_only": 3000,
        },
    )
    _require_exact(bsd, "canonical_sequence_schema", BSD_SEQUENCE_SCHEMA)
    sealed = bsd.get("sealed_test")
    require(isinstance(sealed, Mapping), "BSD sealed-test disclosure is missing")
    for key in (
        "test_pixel_paths_in_contract",
        "test_pixels_opened",
        "test_images_decoded",
        "test_model_outputs_computed",
        "test_metrics_computed",
    ):
        require(sealed.get(key) is False, f"BSD sealed-test invariant failed: {key}")
    require(sealed.get("test_manifest_path_in_contract") is None, "BSD test manifest must be absent")

    dpdd = data.get("dpdd")
    require(isinstance(dpdd, Mapping), "DPDD contract is missing")
    _require_exact(dpdd, "repository", "JacobLinCool/DPDD")
    _require_exact(dpdd, "revision", "52e4035a045ea1763313b9ce2b27cf2e620cfc30")
    _require_exact(dpdd, "train_pairs", 350)
    _require_exact(dpdd, "validation_pairs", 74)
    require(dpdd.get("sealed_test_pixels_opened") is False, "DPDD test pixels are not sealed")
    require(dpdd.get("sealed_test_metrics_computed") is False, "DPDD test metrics are not sealed")

    training = contract.get("training")
    require(isinstance(training, Mapping), "training contract is missing")
    fixed_training = {
        "trained_arms": ["B", "BD"],
        "terminal_checkpoint_only": True,
        "validation_during_training": False,
        "early_stopping": False,
        "bsd_passes": 5,
        "one_clip_per_sequence_per_pass": True,
        "bsd_sequences_per_pass": 60,
        "bsd_clip_length": 5,
        "optimizer_steps": 300,
        "attempted_optimizer_steps": 300,
        "executed_optimizer_steps": 300,
        "amp_skipped_optimizer_steps": 0,
        "loss_start_frame_index": 1,
        "crop_size": 192,
        "future_frames_used": False,
        "optimizer": "AdamW",
        "betas": [0.9, 0.99],
        "scheduler": "CosineAnnealingLR_Tmax300_eta_min_1e-7",
        "amp": True,
        "amp_overflow_policy": "fail_closed_no_checkpoint",
        "groupwise_gradient_clip_norm": 1.0,
    }
    for key, expected in fixed_training.items():
        _require_exact(training, key, expected)
    train_dpdd = training.get("dpdd")
    require(isinstance(train_dpdd, Mapping), "training DPDD contract is missing")
    for key, expected in {
        "arm": "BD_only",
        "passes": 1,
        "pairs": 350,
        "batch_size": 5,
        "backward_steps": 70,
        "cache": "K/V=None before every independent image and returned K/V discarded",
    }.items():
        _require_exact(train_dpdd, key, expected)
    scope = training.get("trainable_scope")
    require(
        scope
        == {
            "history_attention_tensors": 56,
            "history_attention_parameters": 3475994,
            "spatial_head_tensors": 30,
            "spatial_head_parameters": 105283,
            "all_other_tensors_frozen": True,
            "dpdd_history_gradient": "forbidden_and_asserted_none",
        },
        "trainable parameter scope changed",
    )
    require("K/V=None" in str(training.get("cache")), "training K/V must start empty")
    require("never carried across sequences" in str(training.get("cache")), "training K/V boundary changed")
    _require_exact(
        training,
        "learning_rates",
        {"history_attention": 0.00001, "spatial_head": 0.00001},
    )
    _require_exact(
        training,
        "weight_decay",
        {"history_attention": 0.001, "spatial_head": 0.0001},
    )
    _require_exact(
        training,
        "bsd_objective",
        {
            "l1": 1.0,
            "fft_l1": 0.1,
            "adjacent_temporal_delta_l1": 0.1,
            "ordered_vs_cyclic_shuffled_strict_past_rank": 1.0,
            "rank_margin": 0.0001,
        },
    )
    _require_exact(
        training,
        "dpdd_objective",
        {"l1": 1.0, "fft_l1": 0.1, "temporal": 0.0},
    )
    require("70 deterministic" in str(train_dpdd.get("placement")), "DPDD placement changed")
    require("restored before" in str(train_dpdd.get("rng_contract")), "B/BD RNG replay contract changed")

    evaluation = contract.get("evaluation")
    require(isinstance(evaluation, Mapping), "evaluation contract is missing")
    _require_exact(evaluation, "precision", {"E": "CUDA_FP32", "G_O_B_BD": "CUDA_FP16"})
    _require_exact(evaluation, "padding", "right_bottom_to_multiple_8_then_crop_back")
    _require_exact(evaluation, "warmup_unmeasured_calls_per_model", 1)
    _require_exact(evaluation, "native_resolution_bsd", [640, 480])
    _require_exact(
        evaluation,
        "two_pass_execution",
        {
            "pass_1": "timing_only_normal_inference_no_target_decode_no_metrics_no_history_controls",
            "pass_2": "independent_quality_and_history_control_replay_not_timed",
            "state_reset_between_passes_and_sequences": True,
            "timing_samples_from_pass_1_only": True,
            "quality_or_controls_in_timing": False,
        },
    )
    bsd_eval = evaluation.get("bsd_validation")
    require(isinstance(bsd_eval, Mapping), "BSD validation contract is missing")
    _require_exact(bsd_eval, "arms", ["E", "G", "O", "B", "BD"])
    _require_exact(bsd_eval, "ordered_pairs", 2000)
    _require_exact(bsd_eval, "steady_frame_index_min", 3)
    _require_exact(bsd_eval, "steady_frames", 1940)
    _require_exact(
        bsd_eval,
        "metrics",
        ["RGB_PSNR", "RGB_SSIM", "RGB_L1", "synchronized_model_step_latency"],
    )
    _require_exact(bsd_eval, "reporting", ["all_frames", "steady_pooled", "per_sequence"])
    _require_exact(
        bsd_eval,
        "raw_common_baseline",
        {
            "reported": ["all_frames", "steady_pooled", "per_sequence", "per_frame"],
            "same_frame_order_required_across_all_arms": True,
            "same_per_frame_metrics_required_across_all_arms": True,
            "max_abs_metric_difference": 1e-12,
        },
    )
    _require_exact(
        bsd_eval,
        "turtle_history_controls",
        [
            "normal_stream",
            "reset_cache_on_frozen_control_subset",
            "repeat_current_for_complete_past_prefix_on_frozen_control_subset",
            "independent_incremental_ordered_replay_all_frames",
            "cyclic_shuffled_complete_strict_past_on_frozen_control_subset",
        ],
    )
    _require_exact(
        bsd_eval,
        "expensive_history_control_frame_indices_per_sequence",
        [3, 19, 39, 59, 79, 99],
    )
    _require_exact(bsd_eval, "expensive_history_control_frames", 120)
    _require_exact(bsd_eval, "ordered_replay_frames", 2000)
    _require_exact(
        bsd_eval,
        "history_forward_accounting_per_100_frame_sequence_excluding_warmup",
        814,
    )
    _require_exact(
        bsd_eval,
        "history_forward_accounting_per_full_20_sequence_model_excluding_warmup",
        16280,
    )
    _require_exact(bsd_eval, "normal_and_all_controls_share_backend_autocast_path", True)
    _require_exact(bsd_eval, "history_controls_in_latency", False)
    _require_exact(bsd_eval, "timing_only_normal_forwards_per_full_model", 2000)
    _require_exact(bsd_eval, "quality_history_forwards_per_full_turtle_model", 16280)
    _require_exact(bsd_eval, "combined_forwards_per_full_turtle_model_excluding_warmup", 18280)
    _require_exact(bsd_eval, "timing_and_quality_passes_independent", True)
    dpdd_eval = evaluation.get("dpdd_validation")
    require(isinstance(dpdd_eval, Mapping), "DPDD validation contract is missing")
    _require_exact(dpdd_eval, "arms", ["E", "G", "O", "B", "BD"])
    _require_exact(dpdd_eval, "ordered_pairs", 74)
    _require_exact(
        dpdd_eval,
        "two_pass_execution",
        {
            "timing_only_model_steps_per_arm": 74,
            "quality_model_steps_per_arm": 74,
            "combined_model_steps_per_arm_excluding_warmup": 148,
            "targets_metrics_and_lpips_in_timing_pass": False,
            "timing_and_quality_passes_independent": True,
        },
    )
    _require_exact(
        dpdd_eval,
        "latency_disclosure",
        "E_FP32_and_TURTLE_FP16_dedicated_model_step_latency_reported_separately_not_precision_FLOP_architecture_or_SLAM_matched",
    )
    _require_exact(
        dpdd_eval,
        "metrics",
        [
            "RGB_PSNR",
            "RGB_SSIM",
            "AlexNet_LPIPS",
            "RGB_L1",
            "synchronized_model_step_latency",
        ],
    )

    gates = contract.get("preregistered_validation_gates")
    require(isinstance(gates, Mapping), "validation gates are missing")
    require(
        gates.get("aggregation", {}).get("all_three_B_BD_seeds_and_all_primary_gates_conjunctive") is True,
        "three-seed conjunctive rule changed",
    )
    expected_gate_fragments = {
        ("bsd_adaptation_B_minus_G", "per_seed_steady_pooled_psnr_db_min"): 0.1,
        ("dpdd_value_BD_minus_B", "per_seed_psnr_db_min"): 0.1,
        ("bsd_preservation_BD_minus_B", "per_seed_steady_pooled_psnr_db_min"): -0.1,
        ("history_value", "B_and_BD_per_seed_steady_pooled_normal_minus_reset_psnr_db_min"): 0.02,
        ("temporal_preservation", "per_seed_BD_minus_B_normal_steady_pooled_psnr_db_min"): -0.05,
        ("ordered_replay", "max_abs_over_every_pixel_frame_and_G_O_B_BD_checkpoint_max"): 0.000001,
    }
    for (section, key), expected in expected_gate_fragments.items():
        require(gates.get(section, {}).get(key) == expected, f"gate {section}.{key} changed")
    for section, expected in {
        "bsd_adaptation_B_minus_G": {
            "per_seed_steady_pooled_psnr_db_min": 0.1,
            "per_seed_steady_pooled_ssim_min": 0.0,
            "per_seed_steady_pooled_l1_max": 0.0,
        },
        "dpdd_value_BD_minus_B": {
            "per_seed_psnr_db_min": 0.1,
            "per_seed_ssim_min": 0.0,
            "per_seed_lpips_max": 0.0,
            "per_seed_l1_max": 0.0,
        },
        "bsd_preservation_BD_minus_B": {
            "per_seed_steady_pooled_psnr_db_min": -0.1,
            "per_seed_steady_pooled_ssim_min": -0.002,
            "per_seed_steady_pooled_l1_max": 0.002,
        },
    }.items():
        observed = gates.get(section, {})
        for key, value in expected.items():
            require(observed.get(key) == value, f"gate {section}.{key} changed")
    history_gate = gates.get("history_value", {})
    require(
        "official_bsd_O_minus_G" not in gates,
        "O quality must be descriptive and absent from primary gates",
    )
    require(history_gate.get("arms") == ["B", "BD"], "O history leaked into primary gates")
    for key in (
        "B_and_BD_per_seed_steady_pooled_normal_minus_reset_psnr_db_min",
        "B_and_BD_per_seed_steady_pooled_normal_minus_repeat_psnr_db_min",
        "B_and_BD_per_seed_steady_pooled_normal_minus_shuffled_psnr_db_min",
    ):
        require(history_gate.get(key) == 0.02, f"gate history_value.{key} changed")
    require(
        not any(
            key in history_gate
            for key in (
                "O_steady_pooled_normal_minus_reset_psnr_db_min",
                "O_steady_pooled_normal_minus_repeat_psnr_db_min",
                "O_steady_pooled_normal_minus_shuffled_psnr_db_min",
            )
        ),
        "O history thresholds must be absent from primary gates",
    )
    require(
        history_gate.get("aggregation_frames_per_sequence") == [3, 19, 39, 59, 79, 99],
        "history gate subset changed",
    )
    require(
        history_gate.get("O_history_quality_authorizes_training") is False,
        "O history quality must never gate training launch",
    )

    outputs = contract.get("sealed_outputs")
    require(isinstance(outputs, Mapping), "sealed-output contract is missing")
    require(outputs.get("overwrite") is False, "formal outputs must not overwrite")
    require(outputs.get("test_report") is None, "test report path must remain absent")
    _require_exact(outputs, "checkpoint_naming", "checkpoints/{B|BD}/seed_{17|42|73}.pth")
    _require_exact(outputs, "validation_only_report", "validation_only_report.json")
    runtime = contract.get("runtime")
    require(isinstance(runtime, Mapping), "runtime contract is missing")
    from scripts.bsd_dpdd_runtime import (  # local import preserves stdlib-only import graph
        CHILD_RUNTIME_ENVIRONMENT,
        validate_static_runtime_contract,
    )

    validate_static_runtime_contract(runtime)
    _require_exact(
        contract,
        "torch_runtime_policy",
        {
            "torch_version": "2.3.1",
            "torch_cuda_build": "12.1",
            "cudnn_version": 8902,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "deterministic_algorithms": True,
            "deterministic_algorithms_warn_only": False,
            "deterministic_debug_mode": 2,
            "float32_matmul_precision": "highest",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVIDIA_TF32_OVERRIDE": "0",
        },
    )
    _require_exact(
        contract,
        "executor_child_environment",
        dict(CHILD_RUNTIME_ENVIRONMENT),
    )
    environment = contract.get("environment_fingerprint")
    require(isinstance(environment, Mapping), "environment fingerprint is missing")
    require(
        environment.get("schema")
        == "unblur_slam.bsd_dpdd_environment_fingerprint.v1",
        "environment fingerprint schema changed",
    )
    environment_without_digest = dict(environment)
    environment_digest = environment_without_digest.pop("fingerprint_sha256", None)
    require(
        environment_digest == canonical_json_sha256(environment_without_digest),
        "environment fingerprint digest changed",
    )
    # The exact v4 GPU identity, environment fingerprint, deterministic backend
    # policy, and unified lock contract are validated below after the source
    # bundle declaration.  Keeping this call here ensures no protocol can omit
    # its content-addressed implementation inventory.
    validate_code_bundle_declaration(contract)

    _require_exact(
        contract,
        "required_final_report_disclosures",
        {
            "O_vs_G": "O is official TURTLE t0 and G is official TURTLE t1; they differ in architecture, training data, and training budget, so O-minus-G is descriptive and not a causal estimate.",
            "E": "E is a stateless external single-frame EVSSM reference and is not a same-method or streaming-history arm.",
            "precision": "E runs CUDA FP32 with TF32 disabled; TURTLE runs CUDA FP16 autocast with TF32 disabled, so latency is model-step scoped but not precision-, FLOP-, or architecture-matched.",
            "latency": "Latency is a dedicated timing-only normal model-step pass after one unmeasured warm-up and excludes decode, targets, metrics, quality pass, history controls, reporting, and SLAM.",
            "dpdd_latency": "DPDD E and TURTLE latency is reported separately from dedicated timing-only passes; E is FP32 and TURTLE is FP16, so it is not precision-, FLOP-, architecture-, or SLAM-matched.",
            "scope": "Results support restoration-module quality and model-step latency only, not SLAM quality, SLAM latency, FPS, or end-to-end online speed.",
        },
    )
    forbidden = contract.get("claims_forbidden")
    require(isinstance(forbidden, list), "forbidden-claims list is missing")
    for statement in (
        "G and O have the same architecture or training budget",
        "O-minus-G is a causal estimate of architecture, data, or training-budget effects",
        "E is a same-method arm",
        "E is a stateful causal streaming-video model",
        "E FP32 and TURTLE FP16 latency are precision-matched or FLOP-matched",
        "restoration-module validation quality or model-step latency is SLAM quality, SLAM latency, FPS, or end-to-end online speed",
    ):
        require(statement in forbidden, f"required forbidden claim is missing: {statement}")

    unbound = find_unbound_values(contract)
    if allow_template:
        require(unbound, "template unexpectedly contains no unbound values")
    elif reference_only:
        allowed_unbound = {
            "$.data.bsd.train_manifest",
            "$.data.bsd.train_manifest_sha256",
        }
        require(
            set(unbound) == allowed_unbound,
            "reference-only contract must leave exactly BSD train manifest/path "
            f"unbound, observed {unbound}",
        )
        binding = contract.get("reference_binding")
        require(isinstance(binding, Mapping), "reference binding receipt is missing")
        require(
            binding.get("code_bundle_sha256")
            == contract["code_bundle"]["bundle_sha256"],
            "reference binding code-bundle fingerprint changed",
        )
        require(
            binding.get("environment_fingerprint_sha256")
            == contract["environment_fingerprint"]["fingerprint_sha256"],
            "reference binding environment fingerprint changed",
        )
        _require_exact(
            binding,
            "logical_cuda0_mapping",
            {
                "physical_gpu": 1,
                "physical_gpu_uuid": runtime["expected_gpu_uuid"],
                "physical_gpu_serial": runtime["expected_gpu_serial"],
                "logical_device": "cuda:0",
                "hardware_not_queried_by_binder": True,
            },
        )
    else:
        require(not unbound, f"bound contract still contains placeholders: {unbound}")
    return unbound


def _forbid_test_path(path: Path, *, label: str) -> None:
    lowered = {part.lower() for part in path.parts}
    overlap = lowered & FORBIDDEN_SPLIT_COMPONENTS
    require(not overlap, f"{label} contains a forbidden test path component: {sorted(overlap)}")


def _relative_asset(value: Any, root: Path, *, label: str, require_file: bool) -> Path:
    raw = Path(str(value))
    require(not raw.is_absolute(), f"{label} must be relative to the BSD dataset root")
    _forbid_test_path(raw, label=label)
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ContractError(f"{label} escapes the BSD dataset root") from error
    if require_file:
        require(resolved.is_file(), f"missing {label}: {resolved}")
    return resolved


@dataclass(frozen=True)
class VideoManifestInventory:
    path: Path
    sha256: str
    split: str
    sequence_count: int
    frame_count: int
    per_exposure_sequences: Mapping[str, int]
    capture_ids: frozenset[str]
    blurry_paths: frozenset[Path]
    sharp_paths: frozenset[Path]
    blurry_hashes: frozenset[str]
    sharp_hashes: frozenset[str]


def inspect_bsd_sequence_manifest(
    path: Path | str,
    *,
    dataset_root: Path | str,
    expected_sha256: str,
    expected_split: str,
    expected_sequences: int,
    expected_frames: int,
    expected_per_exposure_sequences: int,
    require_assets: bool = True,
    verify_content: bool = True,
) -> VideoManifestInventory:
    """Inspect canonical train/validation JSONL without decoding any image.

    Asset bytes are SHA-256 verified by default.  Hashing is deliberately
    independent of image decoding, so this function remains safe for CPU
    preflight while detecting a path/content substitution after materializing.
    """

    manifest = Path(path).expanduser().resolve()
    _forbid_test_path(manifest, label=f"BSD {expected_split} manifest")
    require(manifest.is_file(), f"missing BSD {expected_split} manifest: {manifest}")
    digest = sha256_file(manifest)
    require(
        digest == normalized_sha256(expected_sha256, label=f"BSD {expected_split} manifest SHA256"),
        f"BSD {expected_split} manifest SHA256 mismatch: {digest}",
    )
    split = str(expected_split).strip().lower()
    require(split in {"train", "validation"}, "only BSD train/validation metadata may be inspected")
    root = Path(dataset_root).expanduser().resolve()
    _forbid_test_path(root, label="BSD dataset root")
    require(root.is_dir(), f"BSD dataset root does not exist: {root}")

    capture_ids: set[str] = set()
    sequence_ids: set[str] = set()
    blurry_paths: set[Path] = set()
    sharp_paths: set[Path] = set()
    blurry_hashes: set[str] = set()
    sharp_hashes: set[str] = set()
    per_exposure = {name: 0 for name in EXPOSURES}
    frame_count = 0
    require(
        expected_frames > 0
        and expected_sequences > 0
        and expected_frames % expected_sequences == 0,
        "BSD expected frame/sequence counts are inconsistent",
    )
    frames_per_sequence = expected_frames // expected_sequences
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(f"invalid BSD JSONL at {manifest}:{line_number}") from error
        require(isinstance(row, Mapping), f"BSD row {line_number} must be an object")
        # Reject the split before resolving or stat'ing any referenced asset.
        row_split = str(row.get("split", "")).strip().lower()
        require(row_split != "test", "BSD test rows are sealed")
        require(row_split == split, f"BSD row {line_number} split mismatch: {row_split!r}")
        require(row.get("schema") == BSD_SEQUENCE_SCHEMA, f"BSD row {line_number} schema mismatch")
        exposure = str(row.get("exposure", ""))
        require(exposure in EXPOSURES, f"BSD row {line_number} has unknown exposure")
        require(row.get("temporal_order") == "gap_free_capture_order", "BSD temporal order is not canonical")
        require(
            row.get("paired_target_alignment") == "center_aligned_synchronized",
            "BSD blurry/sharp pairing is not center-aligned synchronized",
        )
        capture_id = str(row.get("capture_id", "")).strip()
        sequence_id = str(row.get("sequence", "")).strip()
        require(capture_id and sequence_id, f"BSD row {line_number} lacks capture/sequence identity")
        require(sequence_id not in sequence_ids, f"duplicate BSD sequence: {sequence_id}")
        sequence_ids.add(sequence_id)
        capture_ids.add(capture_id)
        per_exposure[exposure] += 1

        blurry_values = row.get("blurry")
        sharp_values = row.get("sharp")
        blurry_sha_values = row.get("blurry_sha256")
        sharp_sha_values = row.get("sharp_sha256")
        frame_indices = row.get("frame_indices")
        require(
            all(isinstance(value, list) for value in (blurry_values, sharp_values, blurry_sha_values, sharp_sha_values)),
            f"BSD row {line_number} frame arrays are missing",
        )
        require(
            isinstance(frame_indices, list),
            f"BSD row {line_number} frame_indices are missing",
        )
        require(
            frame_indices == list(range(frames_per_sequence)),
            f"BSD row {line_number} indices must be exactly 0..{frames_per_sequence - 1}",
        )
        require(
            row.get("frame_count") == frames_per_sequence,
            f"BSD row {line_number} frame_count changed",
        )
        lengths = {
            len(value)
            for value in (
                blurry_values,
                sharp_values,
                blurry_sha_values,
                sharp_sha_values,
            )
        }
        require(
            lengths == {frames_per_sequence},
            f"BSD row {line_number} must contain exactly "
            f"{frames_per_sequence} aligned pairs",
        )
        for frame_index, (blurry_value, sharp_value, blurry_sha, sharp_sha) in enumerate(
            zip(blurry_values, sharp_values, blurry_sha_values, sharp_sha_values)
        ):
            label = f"BSD {split} row {line_number} frame {frame_index}"
            blurry = _relative_asset(blurry_value, root, label=f"{label} blurry", require_file=require_assets)
            sharp = _relative_asset(sharp_value, root, label=f"{label} sharp", require_file=require_assets)
            require(blurry != sharp, f"{label} reuses one path")
            blurry_digest = normalized_sha256(blurry_sha, label=f"{label} blurry SHA256")
            sharp_digest = normalized_sha256(sharp_sha, label=f"{label} sharp SHA256")
            expected_basename = f"{frame_index:08d}.png"
            require(
                blurry.name == expected_basename and sharp.name == expected_basename,
                f"{label} must pair the exact {expected_basename} capture index",
            )
            require(blurry_digest != sharp_digest, f"{label} has identical blurry/sharp bytes")
            if require_assets and verify_content:
                require(
                    sha256_file(blurry) == blurry_digest,
                    f"{label} blurry asset SHA256 mismatch",
                )
                require(
                    sha256_file(sharp) == sharp_digest,
                    f"{label} sharp asset SHA256 mismatch",
                )
            require(blurry not in blurry_paths and sharp not in sharp_paths, f"duplicate BSD path in {split}")
            require(blurry_digest not in blurry_hashes and sharp_digest not in sharp_hashes, f"duplicate BSD content in {split}")
            blurry_paths.add(blurry)
            sharp_paths.add(sharp)
            blurry_hashes.add(blurry_digest)
            sharp_hashes.add(sharp_digest)
            frame_count += 1

    require(len(sequence_ids) == expected_sequences, f"BSD {split} sequence count mismatch")
    require(frame_count == expected_frames, f"BSD {split} frame count mismatch")
    require(
        per_exposure == {name: expected_per_exposure_sequences for name in EXPOSURES},
        f"BSD {split} exposure balance mismatch: {per_exposure}",
    )
    require(not (blurry_paths & sharp_paths), f"BSD {split} blurry/sharp paths overlap")
    require(not (blurry_hashes & sharp_hashes), f"BSD {split} blurry/sharp content overlaps")
    return VideoManifestInventory(
        path=manifest,
        sha256=digest,
        split=split,
        sequence_count=len(sequence_ids),
        frame_count=frame_count,
        per_exposure_sequences=per_exposure,
        capture_ids=frozenset(capture_ids),
        blurry_paths=frozenset(blurry_paths),
        sharp_paths=frozenset(sharp_paths),
        blurry_hashes=frozenset(blurry_hashes),
        sharp_hashes=frozenset(sharp_hashes),
    )


def assert_train_validation_disjoint(
    train: VideoManifestInventory,
    validation: VideoManifestInventory,
) -> None:
    require(train.split == "train" and validation.split == "validation", "BSD split roles changed")
    require(not (train.capture_ids & validation.capture_ids), "BSD train/validation capture IDs overlap")
    train_paths = train.blurry_paths | train.sharp_paths
    validation_paths = validation.blurry_paths | validation.sharp_paths
    require(not (train_paths & validation_paths), "BSD train/validation paths overlap")
    train_hashes = train.blurry_hashes | train.sharp_hashes
    validation_hashes = validation.blurry_hashes | validation.sharp_hashes
    require(not (train_hashes & validation_hashes), "BSD train/validation content overlaps")


def approximately_even_positions(*, total_steps: int, selected_steps: int) -> tuple[int, ...]:
    """Return deterministic zero-based placements, including neither bias nor duplicates."""

    require(total_steps > 0 and 0 < selected_steps <= total_steps, "invalid scheduled-step counts")
    positions = tuple(((index + 1) * total_steps - 1) // selected_steps for index in range(selected_steps))
    require(len(set(positions)) == selected_steps, "scheduled-step placement contains duplicates")
    require(positions[0] >= 0 and positions[-1] < total_steps, "scheduled-step placement is out of range")
    return positions


def deterministic_bsd_schedule(
    sequence_names: Sequence[str],
    sequence_lengths: Sequence[int],
    *,
    seed: int,
    passes: int = 5,
    clip_length: int = 5,
) -> list[tuple[int, int, int]]:
    """Return (pass, sequence-index, contiguous-start) without reading pixels."""

    import random

    require(seed in SEEDS, "formal BSD seed must be 17, 42, or 73")
    require(len(sequence_names) == len(sequence_lengths) and sequence_names, "invalid BSD inventory")
    require(len(set(sequence_names)) == len(sequence_names), "BSD sequence names must be unique")
    require(all(length >= clip_length for length in sequence_lengths), "BSD sequence shorter than clip")
    schedule: list[tuple[int, int, int]] = []
    for pass_index in range(passes):
        order = list(range(len(sequence_names)))
        random.Random(seed + 10_000 + pass_index).shuffle(order)
        for sequence_index in order:
            maximum_start = sequence_lengths[sequence_index] - clip_length
            start_rng = random.Random(
                seed + 20_000 + pass_index * 1_000_003 + sequence_index * 9_176
            )
            start = start_rng.randrange(maximum_start + 1)
            schedule.append((pass_index, sequence_index, start))
    return schedule


def summarize_inventory(inventory: VideoManifestInventory) -> dict[str, Any]:
    return {
        "path": str(inventory.path),
        "sha256": inventory.sha256,
        "split": inventory.split,
        "sequence_count": inventory.sequence_count,
        "frame_count": inventory.frame_count,
        "per_exposure_sequences": dict(inventory.per_exposure_sequences),
        "capture_id_count": len(inventory.capture_ids),
        "asset_bytes_opened_for_sha256": True,
        "images_decoded": False,
        "pixel_values_used": False,
    }


def command_digest(commands: Iterable[Sequence[str]]) -> str:
    canonical = json.dumps(list(commands), sort_keys=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
