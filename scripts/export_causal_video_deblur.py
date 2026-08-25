#!/usr/bin/env python3
"""Export a trained causal video deblurrer to the frontend TorchScript contract."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video_deblur import build_causal_video_deblur
from src.video_deblur.dataset import TEACHER_PROVENANCE_SCHEMA, sha256_file


CHECKPOINT_FORMAT_V1 = "unblur_slam.causal_video_deblur.v1"
CHECKPOINT_FORMAT_V3 = "unblur_slam.causal_video_deblur.v3"
CHECKPOINT_FORMAT_V4 = "unblur_slam.causal_video_deblur.v4"
TORCHSCRIPT_FORMAT_V1 = "unblur_slam.causal_video_deblur.torchscript.v1"
TORCHSCRIPT_FORMAT_V3 = "unblur_slam.causal_video_deblur.torchscript.v3"
TORCHSCRIPT_FORMAT_V4 = "unblur_slam.causal_video_deblur.torchscript.v4"
OBJECTIVE_SCHEMA_V3 = "unblur_slam.causal_video_deblur.objective.v3"
REFINEMENT_SCHEMA_V3 = "unblur_slam.causal_video_deblur.refinement.v3"
TRAINING_CONTRACT_SCHEMA_V3 = "unblur_slam.causal_video_deblur.training.v3"
OBJECTIVE_SCHEMA_V4 = "unblur_slam.causal_video_deblur.objective.v4"
REFINEMENT_SCHEMA_V4 = "unblur_slam.causal_video_deblur.refinement.v4"
TRAINING_CONTRACT_SCHEMA_V4 = "unblur_slam.causal_video_deblur.training.v4"
OPTIMIZATION_CONTRACT_SCHEMA_V4 = (
    "unblur_slam.causal_video_deblur.optimization.v4"
)
WARM_START_SCHEMA_V4 = "unblur_slam.causal_video_deblur.warm_start.v4"
RNG_STATE_SCHEMA_V4 = "unblur_slam.causal_video_deblur.rng_state.v4"
NUMPY_RNG_ENCODING_V4 = (
    "numpy.random.RandomState.MT19937.keys_torch_int64.v1"
)
CHECKPOINT_MIGRATION_SCHEMA_V1 = (
    "unblur_slam.causal_video_deblur.checkpoint_serialization_migration.v1"
)
CHECKPOINT_MIGRATION_KIND_V1 = "numpy_rng_ndarray_uint32_to_torch_int64.v1"
CHECKPOINT_SEMANTIC_DIGEST_SCHEMA_V1 = (
    "unblur_slam.causal_video_deblur.semantic_checkpoint_digest.v1"
)
CHECKPOINT_SEMANTIC_DIGEST_ALGORITHM_V1 = "sha256_canonical_tree_v1"
CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1 = [
    "/checkpoint_migration",
    "/rng_state/numpy_random_state/1",
    "/rng_state/numpy_random_state_encoding",
]
DEPLOYMENT_SELECTION_SCHEMA_V3 = (
    "unblur_slam.causal_video_deblur.layered_deployment_selection.v3"
)
DEPLOYMENT_LAYER_REPORT_SCHEMA_V1 = (
    "unblur_slam.causal_video_deblur.layer_selection_report.v1"
)
DEPLOYMENT_SELECTION_POLICY_V1 = (
    "replica424_temporal_val_then_room2_one_shot.v1"
)
EVALUATOR_SCHEMA_V3 = "unblur_slam.causal_video_deblur_smoke_eval.v3"
EVALUATOR_SCHEMA_V4 = "unblur_slam.causal_video_deblur_smoke_eval.v4"
TRAINING_REQUIRED_SELECTOR_V1 = "evssm_relative_multimetric_gate.v1"
TRAINING_REQUIRED_SELECTOR_V4 = "motion_aligned_evssm_multimetric_gate.v1"
REGISTERED_CONTRACT_SCHEMA = "unblur_slam.causal_evssm_replica424_experiment.v1"
REGISTERED_CONTRACT_SHA256 = (
    "e2f9b411725b52bf8d214871304c12d5a78a29e536d6baeb7a4fbff8a2b8bb3c"
)
REGISTERED_EVSSM_SHA256 = (
    "4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41"
)
REGISTERED_V4_CONTRACT_SCHEMA = (
    "unblur_slam.causal_evssm_alignment_replica424_experiment.v4"
)
# Updated only when the pre-training v4 contract file is deliberately changed.
# This is a content pin, not a value computed dynamically at export time.
REGISTERED_V4_CONTRACT_SHA256 = (
    "511dbcce9bad94ef10b3b5af9615d1bfed1300cf273cac5a9b57779c0413563d"
)
REGISTERED_V4_WARM_START_SHA256 = (
    "8338d007762d9e626bd7f85722140a70ff0084e58f1bbe8dfd338daca346b0e4"
)
# This registered run completed before the NumPy RNG state was made
# weights-only-safe.  Its sole terminal checkpoint and the one audited
# serialization-only migration are therefore immutable parts of the v4
# lineage.  A future native-safe training run must use a new registered
# contract rather than omitting this lineage under the existing 511dbc pin.
REGISTERED_V4_PRE_MIGRATION_CHECKPOINT_SHA256 = (
    "ad80e84f67f6c979de96ce2a65ceeb7201b2cf7f7159af64d8fe2c2face030e0"
)
REGISTERED_V4_SAFE_CHECKPOINT_SHA256 = (
    "92a1ab5301355e923fbd8c2059bbb0c5bdbe041cc00880b21591efdfd7de5bfd"
)
REGISTERED_V4_CHECKPOINT_SEMANTIC_SHA256 = (
    "a533ca551efc7543034ee73b64539c2056c913dcc4f9183df8ebbec4426c2c9d"
)
REGISTERED_V4_DATA_IDENTITY = {
    "train_manifest_sha256": (
        "bd7caa189374683c8ffd7e8fce83cb62e5f69b73f6048808c4808dc2b4ecd2ba"
    ),
    "train_precompute_report_sha256": (
        "9fad0d8c90e64fc5ef471bef85c374b5a09393f33ca16fb3dabb5a1bb206a3e0"
    ),
    "train_teacher_manifest_sha256": (
        "1e1f9ab0d28ec3d7f391c9d4bcb6184ea275829af3fc824c7e42195bbba1f24e"
    ),
    "val_manifest_sha256": (
        "1aa8cc7a01b82c7d759c3db70e6c7e796a26d09398f3a1fd1592d787db9f886b"
    ),
    "val_precompute_report_sha256": (
        "2a394089ead9b6ef069fab1885b20d11805b90b83d32f3cbd180fd4490cd8d4a"
    ),
    "val_teacher_manifest_sha256": (
        "b9f2b86a18705bb427799fc1823491cf7dd6f9a7e3c54af9889a09fe1073e6fc"
    ),
    "evssm_checkpoint_sha256": REGISTERED_EVSSM_SHA256,
}
REGISTERED_V4_BASE_MODEL_CONFIG = {
    "channels": 32,
    "num_heads": 4,
    "num_blocks": 2,
    "max_history": 3,
    "use_teacher_input": False,
    "input_domain": "evssm",
    "max_residual": 8.0 / 255.0,
}
REGISTERED_V4_MODEL_CONFIG = {
    **REGISTERED_V4_BASE_MODEL_CONFIG,
    "motion_alignment": {
        "mode": "coarse_local_correlation_v1",
        "match_channels": 16,
        "radius": 8,
        "temperature": 0.05,
    },
}
DEPLOYMENT_SELECTION_SCHEMA_V4 = (
    "unblur_slam.causal_video_deblur.layered_deployment_selection.v4"
)
DEPLOYMENT_LAYER_REPORT_SCHEMA_V4 = (
    "unblur_slam.causal_video_deblur.layer_selection_report.v4"
)
DEPLOYMENT_SELECTION_POLICY_V4 = (
    "replica424_alignment_temporal_val_then_room2_one_shot.v4"
)
ALIGNMENT_DIAGNOSTICS_SCHEMA_V4 = (
    "unblur_slam.causal_video_deblur.alignment_diagnostics.v4"
)
TEMPORAL_VALIDATION_MANIFEST_SHA256 = (
    "1aa8cc7a01b82c7d759c3db70e6c7e796a26d09398f3a1fd1592d787db9f886b"
)
ROOM2_ONE_SHOT_MANIFEST_SHA256 = (
    "49464ac272a747675923ad28f5a659b2413a0975818af7bb35de39a90eaa1ba8"
)
ROOM2_ONE_SHOT_FRAME_COUNT = 174
ROOM2_ONE_SHOT_FRAME_IDENTITY_SHA256 = (
    "b9438030e9c7bc06179db662d60738e15a30d48e3f45c1b1aa1ae3451c1a3861"
)
ROOM2_ONE_SHOT_SOURCE_ROOT = Path(
    "/srv/szha0669/unblur-slam/causal_video_data"
)
DEPLOYMENT_THRESHOLDS = {
    "temporal_validation": {
        "steady_psnr_delta_db_min": 0.1,
        "steady_ssim_delta_min": 0.0,
        "steady_relative_l1_delta_max": -0.005,
        "steady_gt_temporal_difference_relative_delta_max": -0.01,
        "normal_vs_repeat_current_psnr_delta_db_min": 0.05,
        "normal_vs_repeat_current_temporal_relative_delta_max": -0.01,
        "normal_vs_history1_psnr_delta_db_min": 0.05,
        "laplacian_gate_pass_ratio_min": 0.25,
        "accepted_oracle_precision_min": 0.8,
        "worst_run_psnr_delta_db_min": -0.1,
        "prefix_psnr_delta_db_min": -0.02,
    },
    "room2_one_shot": {
        "psnr_delta_db_min": 0.05,
        "ssim_delta_min": 0.0,
        "relative_l1_delta_max": 0.0,
        "lpips_delta_max": 0.0,
        "gt_temporal_difference_delta_max": 0.0,
        "steady_normal_vs_repeat_current_psnr_delta_db_min": 0.03,
        "laplacian_gate_pass_ratio_min": 0.2,
        "accepted_oracle_precision_min": 0.8,
        "nondegraded_long_runs_min": 10,
        "long_runs_total": 16,
        "worst_run_psnr_delta_db_min": -0.1,
    },
}
ORACLE_GOOD_DEFINITION = {
    "delta_psnr_db_min": -0.02,
    "delta_ssim_min": -0.0002,
    "relative_l1_max": 0.005,
}
EXPECTED_LPIPS_PROTOCOL = {
    "implementation": "torchmetrics.image.lpip",
    "network": "alex",
    "normalize_input_0_1": True,
    "per_frame_state_reset": True,
}


def _canonical_semantic_node(value: object, path: tuple[object, ...]) -> object:
    """Build a deterministic typed tree for serialization-neutral hashing."""

    rng_keys_path = ("rng_state", "numpy_random_state", 1)
    if path == rng_keys_path:
        if isinstance(value, torch.Tensor):
            if value.dtype != torch.int64 or value.numel() != 624:
                raise ValueError("semantic digest received malformed tensor RNG keys")
            array = value.detach().cpu().numpy()
            if np.any(array < 0) or np.any(array > np.iinfo(np.uint32).max):
                raise ValueError("semantic digest RNG keys exceed uint32")
            array = array.astype("<u4", copy=True)
        elif isinstance(value, np.ndarray):
            if value.dtype != np.uint32 or value.size != 624:
                raise ValueError("semantic digest received malformed ndarray RNG keys")
            array = value.reshape(-1).astype("<u4", copy=True)
        else:
            raise ValueError("semantic digest received unsupported RNG keys")
        return {
            "type": "mt19937_uint32_keys",
            "shape": [624],
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "type": "torch_tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise ValueError("semantic digest forbids object NumPy arrays")
        return {
            "type": "numpy_array",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, dict):
        items = []
        for key, child in value.items():
            if path == () and key == "checkpoint_migration":
                continue
            if path == ("rng_state",) and key == "numpy_random_state_encoding":
                continue
            key_node = _canonical_semantic_node(key, path + ("<key>",))
            child_node = _canonical_semantic_node(child, path + (key,))
            items.append((key_node, child_node))
        items.sort(
            key=lambda item: json.dumps(
                item[0], sort_keys=True, separators=(",", ":")
            )
        )
        return {"type": "dict", "items": items}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [
                _canonical_semantic_node(child, path + (index,))
                for index, child in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [
                _canonical_semantic_node(child, path + (index,))
                for index, child in enumerate(value)
            ],
        }
    if value is None:
        return {"type": "none"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": str(value)}
    if type(value) is float:
        return {"type": "float", "value": value.hex()}
    if type(value) is str:
        return {"type": "str", "value": value}
    if type(value) is bytes:
        return {
            "type": "bytes",
            "sha256": hashlib.sha256(value).hexdigest(),
            "length": len(value),
        }
    raise ValueError(
        f"semantic digest does not support payload type {type(value).__name__}"
    )


def checkpoint_semantic_digest(checkpoint: dict[str, object]) -> str:
    """Hash scientific checkpoint state independent of the RNG representation."""

    canonical = _canonical_semantic_node(checkpoint, ())
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_checkpoint_migration(
    checkpoint: dict[str, object],
) -> Optional[dict[str, object]]:
    migration = checkpoint.get("checkpoint_migration")
    if migration is None:
        return None
    if not isinstance(migration, dict) or set(migration) != {
        "schema",
        "kind",
        "source_checkpoint_sha256",
        "allowed_changes",
        "semantic_digest",
    }:
        raise ValueError("checkpoint_migration has an invalid field set")
    if migration.get("schema") != CHECKPOINT_MIGRATION_SCHEMA_V1:
        raise ValueError("checkpoint_migration schema mismatch")
    if migration.get("kind") != CHECKPOINT_MIGRATION_KIND_V1:
        raise ValueError("checkpoint_migration kind mismatch")
    _sha256_digest(
        migration.get("source_checkpoint_sha256"),
        "checkpoint_migration.source_checkpoint_sha256",
    )
    if migration.get("allowed_changes") != CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1:
        raise ValueError("checkpoint_migration allowed_changes mismatch")
    semantic = migration.get("semantic_digest")
    if not isinstance(semantic, dict) or semantic != {
        "schema": CHECKPOINT_SEMANTIC_DIGEST_SCHEMA_V1,
        "algorithm": CHECKPOINT_SEMANTIC_DIGEST_ALGORITHM_V1,
        "sha256": semantic.get("sha256") if isinstance(semantic, dict) else None,
        "source_and_target_equal": True,
    }:
        raise ValueError("checkpoint_migration semantic_digest contract mismatch")
    digest = _sha256_digest(
        semantic.get("sha256"), "checkpoint_migration.semantic_digest.sha256"
    )
    if checkpoint_semantic_digest(checkpoint) != digest:
        raise ValueError("checkpoint_migration semantic digest does not match target")
    return dict(migration)


def validate_registered_v4_checkpoint_migration(
    checkpoint: dict[str, object], checkpoint_sha256: object
) -> dict[str, object]:
    """Bind the completed 511dbc run to its sole audited safe checkpoint."""

    target_sha256 = _sha256_digest(
        checkpoint_sha256, "registered v4 tensor-safe checkpoint SHA-256"
    )
    if target_sha256 != REGISTERED_V4_SAFE_CHECKPOINT_SHA256:
        raise ValueError(
            "registered v4 export requires the pinned tensor-safe checkpoint SHA-256"
        )
    migration = validate_checkpoint_migration(checkpoint)
    if migration is None:
        raise ValueError(
            "registered v4 export requires the audited checkpoint_migration lineage"
        )
    if migration.get("source_checkpoint_sha256") != (
        REGISTERED_V4_PRE_MIGRATION_CHECKPOINT_SHA256
    ):
        raise ValueError(
            "registered v4 migration source is not the pinned formal terminal"
        )
    semantic = migration.get("semantic_digest")
    if not isinstance(semantic, dict) or semantic.get("sha256") != (
        REGISTERED_V4_CHECKPOINT_SEMANTIC_SHA256
    ):
        raise ValueError(
            "registered v4 migration semantic digest is not the audited value"
        )
    return {
        "schema": CHECKPOINT_MIGRATION_SCHEMA_V1,
        "kind": CHECKPOINT_MIGRATION_KIND_V1,
        "source_checkpoint_sha256": (
            REGISTERED_V4_PRE_MIGRATION_CHECKPOINT_SHA256
        ),
        "target_checkpoint_sha256": target_sha256,
        "allowed_changes": list(CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1),
        "semantic_digest": {
            "schema": CHECKPOINT_SEMANTIC_DIGEST_SCHEMA_V1,
            "algorithm": CHECKPOINT_SEMANTIC_DIGEST_ALGORITHM_V1,
            "sha256": REGISTERED_V4_CHECKPOINT_SEMANTIC_SHA256,
            "source_and_target_equal": True,
        },
    }


def _sha256_digest(value: object, label: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def validate_teacher_provenance(
    value: object, *, input_domain: str
) -> dict[str, object]:
    if not isinstance(value, dict) or value.get("schema") != TEACHER_PROVENANCE_SCHEMA:
        raise ValueError("checkpoint is missing validated teacher_provenance")
    provenance = dict(value)
    storage = str(provenance.get("storage", ""))
    domain = str(provenance.get("teacher_domain", ""))
    if storage == "none":
        if str(input_domain) == "evssm":
            raise ValueError("EVSSM-domain checkpoint cannot have teacher storage=none")
        if domain != "none" or provenance.get("evssm_checkpoint_sha256") is not None:
            raise ValueError("teacher storage=none has inconsistent provenance")
        return provenance
    if storage not in {"runtime_evssm_float_tensor", "precomputed_png_rgb8"}:
        raise ValueError(f"unsupported teacher storage {storage!r}")
    if domain != "evssm_restored_rgb_0_1":
        raise ValueError(f"unsupported teacher domain {domain!r}")
    provenance["evssm_checkpoint_sha256"] = _sha256_digest(
        provenance.get("evssm_checkpoint_sha256"), "evssm_checkpoint_sha256"
    )
    if storage == "precomputed_png_rgb8":
        for key in ("precompute_report_sha256", "teacher_manifest_sha256"):
            provenance[key] = _sha256_digest(provenance.get(key), key)
        if not bool(provenance.get("teacher_artifacts_verified", False)):
            raise ValueError("cached teacher artifacts were not verified before training")
    return provenance


def validate_v3_contracts(checkpoint: dict[str, object], config: dict[str, object]) -> None:
    """Fail closed if a purported v3 artifact omits its safety/loss contract."""
    try:
        max_residual = float(config["max_residual"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("v3 model_config must contain numeric max_residual") from error
    if not math.isfinite(max_residual) or max_residual <= 0.0 or max_residual > 1.0:
        raise ValueError("v3 max_residual must be finite and in (0, 1]")

    refinement = checkpoint.get("refinement_contract")
    if not isinstance(refinement, dict) or refinement.get("schema") != REFINEMENT_SCHEMA_V3:
        raise ValueError("v3 checkpoint is missing refinement_contract")
    if refinement.get("formula") != (
        "output = input + max_residual * tanh(residual_logits)"
    ):
        raise ValueError("v3 refinement formula is unsupported")
    try:
        refinement_max_residual = float(refinement["max_residual"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("v3 refinement max_residual must be numeric") from error
    if refinement_max_residual != max_residual:
        raise ValueError("v3 refinement max_residual disagrees with model_config")
    if refinement.get("identity_safe_initialization") != (
        "zero_weight_and_bias_output_head"
    ):
        raise ValueError("v3 checkpoint does not prove identity-safe initialization")
    expected_base = (
        "frozen_evssm_input"
        if str(config.get("input_domain", "raw")) == "evssm"
        else "raw_input"
    )
    if refinement.get("base") != expected_base:
        raise ValueError("v3 refinement base disagrees with input_domain")

    objective = checkpoint.get("objective_contract")
    if not isinstance(objective, dict) or objective.get("schema") != OBJECTIVE_SCHEMA_V3:
        raise ValueError("v3 checkpoint is missing objective_contract")
    required_weight_paths = (
        ("primary_reconstruction", "l1_weight"),
        ("primary_reconstruction", "fft_l1_weight"),
        ("evssm_fidelity", "weight"),
        ("temporal_delta", "weight"),
        ("edge", "weight"),
        ("laplacian_gate", "weight"),
        ("legacy_latest_evssm_distillation", "weight"),
    )
    for section, key in required_weight_paths:
        value = objective.get(section)
        if not isinstance(value, dict):
            raise ValueError(f"v3 objective is missing {section}")
        try:
            weight = float(value[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"v3 objective is missing {section}.{key}") from error
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"v3 objective weight {section}.{key} is invalid")
    primary = objective["primary_reconstruction"]
    if primary.get("fft_normalization") != "ortho":
        raise ValueError("v3 FFT objective must declare norm='ortho'")
    if objective["evssm_fidelity"].get("frames") != (
        "all_causal_prefix_positions"
    ):
        raise ValueError("v3 EVSSM fidelity must cover the causal prefix")
    if objective["temporal_delta"].get("frames") != (
        "two_shifted_full_history_windows"
    ):
        raise ValueError("v3 temporal delta must use two full rolling windows")
    if objective["edge"].get("operator") != (
        "first_order_xy_plus_runtime_grayscale_zero_pad_laplacian"
    ):
        raise ValueError("v3 edge objective must include gradient and Laplacian terms")
    if objective["laplacian_gate"].get("runtime_gate_alignment") != (
        "rgb_mean_then_four_neighbour_zero_pad_laplacian_unbiased_variance"
    ):
        raise ValueError("v3 Laplacian objective does not match the runtime gate")
    if objective["laplacian_gate"].get("minimum_relative_gain") != 0.0 or (
        objective["laplacian_gate"].get("variance_floor") != 1.0e-6
    ):
        raise ValueError("v3 Laplacian objective uses an unsupported gate threshold")

    training = checkpoint.get("training_contract")
    if not isinstance(training, dict) or training.get("schema") != (
        TRAINING_CONTRACT_SCHEMA_V3
    ):
        raise ValueError("v3 checkpoint is missing training_contract")
    if training.get("temporal_output") != "rolling_two_window_forward":
        raise ValueError("v3 training did not use two full rolling windows")
    history = int(config.get("max_history", 0))
    if training.get("rolling_window_length") != history or training.get(
        "training_clip_length"
    ) != history + 1:
        raise ValueError("v3 rolling-window lengths disagree with model_config")
    if training.get("causality") != (
        "strict_upper_triangular_temporal_attention_mask"
    ):
        raise ValueError("v3 checkpoint does not prove strict causal attention")
    if training.get("fft_normalization") != "ortho":
        raise ValueError("v3 training contract must declare FFT norm='ortho'")

    selection = checkpoint.get("checkpoint_selection")
    if not isinstance(selection, dict) or selection.get("metric") != "val_psnr":
        raise ValueError("v3 checkpoint is missing checkpoint-selection metadata")
    if selection.get("deployment_status") != "not_deployment_selected" or selection.get(
        "required_deployment_selector"
    ) != TRAINING_REQUIRED_SELECTOR_V1:
        raise ValueError("v3 training checkpoint must require external deployment selection")
    optimization = checkpoint.get("optimization_contract")
    if not isinstance(optimization, dict) or optimization.get("schema") != (
        "unblur_slam.causal_video_deblur.optimization.v3"
    ):
        raise ValueError("v3 checkpoint is missing optimization_contract")
    if int(optimization.get("gradient_accumulation_micro_batches", 0)) < 1:
        raise ValueError("v3 gradient-accumulation contract is invalid")
    if optimization.get("lr_schedule") != "linear_warmup_then_cosine" or (
        optimization.get("schedule_unit") != "optimizer_step"
    ):
        raise ValueError("v3 learning-rate schedule contract is invalid")


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_v4_contracts(
    checkpoint: dict[str, object],
    config: dict[str, object],
    *,
    checkpoint_sha256: Optional[str] = None,
) -> None:
    """Validate the alignment-v4 training artifact before any export.

    V4 is deliberately a new contract rather than a permissive extension of
    v3.  In particular, it must prove the exact warm start, fresh optimizer,
    two registered phases, alignment API, and unchanged bounded-EVSSM
    refinement.  A malformed v4 checkpoint is never interpreted as v3.
    """

    expected_alignment = REGISTERED_V4_MODEL_CONFIG["motion_alignment"]
    if config != REGISTERED_V4_MODEL_CONFIG:
        raise ValueError("v4 model_config architecture is not exactly preregistered")
    registered_contract = checkpoint.get("registered_contract")
    if not isinstance(registered_contract, dict) or {
        "schema": registered_contract.get("schema"),
        "sha256": registered_contract.get("sha256"),
    } != {
        "schema": REGISTERED_V4_CONTRACT_SCHEMA,
        "sha256": REGISTERED_V4_CONTRACT_SHA256,
    }:
        raise ValueError("v4 checkpoint is not bound to the preregistered contract")
    if type(checkpoint.get("step")) is not int or checkpoint.get("step") != 600:
        raise ValueError("v4 export requires the terminal optimizer step 600")
    if type(checkpoint.get("epoch")) is not int or checkpoint.get("epoch") != 25:
        raise ValueError("v4 export requires the registered terminal epoch 25")
    if checkpoint.get("training_phase") != "joint":
        raise ValueError("v4 terminal checkpoint must be in the joint phase")
    if checkpoint.get("data_identity") != REGISTERED_V4_DATA_IDENTITY:
        raise ValueError("v4 checkpoint data_identity is not preregistered")
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, dict) or rng_state.get("schema") != (
        RNG_STATE_SCHEMA_V4
    ) or rng_state.get("checkpoint_boundary") != (
        "epoch_end_no_pending_accumulation"
    ):
        raise ValueError("v4 checkpoint has no registered terminal RNG boundary")
    for key in (
        "torch_cpu_rng_state",
        "train_loader_generator_state",
        "alignment_loader_generator_state",
    ):
        value = rng_state.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.device.type != "cpu"
            or value.ndim != 1
            or value.numel() < 1
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"v4 RNG state {key} must be a nonempty contiguous CPU "
                "uint8 vector"
            )
    if not isinstance(rng_state.get("python_random_state"), tuple) or not isinstance(
        rng_state.get("numpy_random_state"), tuple
    ):
        raise ValueError("v4 Python/NumPy RNG states are malformed")
    numpy_random_state = rng_state["numpy_random_state"]
    if len(numpy_random_state) != 5 or numpy_random_state[0] != "MT19937":
        raise ValueError("v4 NumPy RNG state must be a five-item MT19937 tuple")
    numpy_keys = numpy_random_state[1]
    if (
        not isinstance(numpy_keys, torch.Tensor)
        or numpy_keys.dtype != torch.int64
        or numpy_keys.device.type != "cpu"
        or numpy_keys.ndim != 1
        or numpy_keys.numel() != 624
        or not numpy_keys.is_contiguous()
    ):
        raise ValueError(
            "exportable v4 NumPy RNG keys must use the weights-only-safe "
            "contiguous CPU 624-element int64 tensor encoding"
        )
    if rng_state.get("numpy_random_state_encoding") != NUMPY_RNG_ENCODING_V4:
        raise ValueError("exportable v4 NumPy RNG state has no exact encoding tag")
    if bool((numpy_keys < 0).any().item()) or bool(
        (numpy_keys > (2**32 - 1)).any().item()
    ):
        raise ValueError("v4 NumPy RNG keys exceed the uint32 value range")
    numpy_position, numpy_has_gauss, numpy_cached_gaussian = (
        numpy_random_state[2:]
    )
    if type(numpy_position) is not int or not 0 <= numpy_position <= 624:
        raise ValueError("v4 NumPy RNG position must be an integer in [0,624]")
    if type(numpy_has_gauss) is not int or numpy_has_gauss not in {0, 1}:
        raise ValueError("v4 NumPy RNG has_gauss must be 0 or 1")
    if type(numpy_cached_gaussian) is not float or not math.isfinite(
        float(numpy_cached_gaussian)
    ):
        raise ValueError("v4 NumPy RNG cached Gaussian must be finite")
    if int(config.get("max_history", 0)) != 3:
        raise ValueError("v4 model history must be exactly 3")
    if config.get("input_domain") != "evssm":
        raise ValueError("v4 model must refine the EVSSM input domain")
    max_residual = _finite_float(config.get("max_residual"), "v4 max_residual")
    if not math.isclose(max_residual, 8.0 / 255.0, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("v4 max_residual must be exactly 8/255")

    refinement = checkpoint.get("refinement_contract")
    if not isinstance(refinement, dict) or refinement.get("schema") != (
        REFINEMENT_SCHEMA_V4
    ):
        raise ValueError("v4 checkpoint is missing refinement_contract")
    if refinement.get("base") != "frozen_evssm_input" or refinement.get(
        "formula"
    ) != "output = input + max_residual * tanh(residual_logits)":
        raise ValueError("v4 refinement formula/base is unsupported")
    if not math.isclose(
        _finite_float(refinement.get("max_residual"), "v4 refinement max_residual"),
        max_residual,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("v4 refinement max_residual disagrees with model_config")

    warm_start = checkpoint.get("warm_start_provenance")
    if not isinstance(warm_start, dict) or warm_start.get("schema") != (
        WARM_START_SCHEMA_V4
    ):
        raise ValueError("v4 checkpoint is missing warm_start_provenance")
    if _sha256_digest(
        warm_start.get("source_sha256"), "v4 warm start source_sha256"
    ) != REGISTERED_V4_WARM_START_SHA256:
        raise ValueError("v4 warm start is not the preregistered H3 epoch20")
    if warm_start.get("source_format") != CHECKPOINT_FORMAT_V3:
        raise ValueError("v4 warm start source must be a v3 checkpoint")
    if warm_start.get("source_model_config") != REGISTERED_V4_BASE_MODEL_CONFIG:
        raise ValueError("v4 warm start source_model_config is not preregistered")
    if warm_start.get("optimizer_state_loaded") is not False:
        raise ValueError("v4 must start with a fresh optimizer")
    _sha256_digest(
        warm_start.get("source_state_key_digest_sha256"),
        "v4 warm start source_state_key_digest_sha256",
    )
    copied_key_count = warm_start.get("copied_key_count")
    if type(copied_key_count) is not int or copied_key_count < 1:
        raise ValueError("v4 warm start copied_key_count must be positive")
    allowed_missing = warm_start.get("allowed_missing_alignment_keys")
    expected_missing = {
        "motion_alignment_gate",
        "motion_aligner.match_projection.weight",
        "motion_aligner.offsets",
    }
    if not isinstance(allowed_missing, list) or set(allowed_missing) != expected_missing:
        raise ValueError("v4 warm start has an unregistered missing-key set")
    identity_probe = warm_start.get("identity_probe")
    if not isinstance(identity_probe, dict) or identity_probe.get("passed") is not True:
        raise ValueError("v4 warm start identity probe did not pass")
    tolerance = _finite_float(identity_probe.get("atol"), "v4 identity probe atol")
    difference = _finite_float(
        identity_probe.get("max_abs_difference"),
        "v4 identity probe max_abs_difference",
    )
    if tolerance < 0.0 or difference < 0.0 or difference > tolerance:
        raise ValueError("v4 warm start identity probe exceeds its tolerance")

    training = checkpoint.get("training_contract")
    if not isinstance(training, dict) or training.get("schema") != (
        TRAINING_CONTRACT_SCHEMA_V4
    ):
        raise ValueError("v4 checkpoint is missing training_contract")
    if training.get("stream_prefix_padding") != "repeat_first_frame_on_left":
        raise ValueError("v4 training has the wrong prefix-padding contract")
    if int(training.get("training_clip_length", 0)) != 4 or int(
        training.get("rolling_window_length", 0)
    ) != 3:
        raise ValueError("v4 training must use H+1=4 clips and H=3 windows")
    if training.get("diagnostic_method") != (
        "forward_sequence_with_motion_diagnostics"
    ) or training.get("alignment_disabled_method") != (
        "forward_sequence_alignment_disabled"
    ):
        raise ValueError("v4 training does not bind the registered diagnostic API")
    for count_key in ("real_transition_slots",):
        count = training.get(count_key)
        if type(count) is not int or count < 1:
            raise ValueError(f"v4 training {count_key} must be a positive integer")
    expected_inventory = {
        "train_clips": 234,
        "train_sequences": 127,
        "unique_real_transitions": 107,
        "alignment_sampler_clips": 107,
    }
    for key, expected in expected_inventory.items():
        if type(training.get(key)) is not int or training.get(key) != expected:
            raise ValueError(f"v4 training inventory {key} must be {expected}")
    if training.get("alignment_sampler_policy") != (
        "clips_with_at_least_one_real_transition"
    ):
        raise ValueError("v4 phase1 sampler is not restricted to real-edge clips")
    if training.get("dropped_tail_policy") != (
        "shuffle_then_drop_incomplete_microbatch_each_epoch"
    ):
        raise ValueError("v4 training loader tail policy is not preregistered")
    if training.get("terminal_checkpoint_policy") != (
        "unconditional_atomic_save_at_exact_optimizer_step_600_before_exit"
    ):
        raise ValueError("v4 terminal checkpoint policy is not preregistered")
    if training.get("resume_rng_policy") != (
        "epoch_boundary_python_numpy_torch_cpu_and_loader_generators"
    ):
        raise ValueError("v4 resume RNG policy is not preregistered")

    objective = checkpoint.get("objective_contract")
    if not isinstance(objective, dict) or objective.get("schema") != (
        OBJECTIVE_SCHEMA_V4
    ):
        raise ValueError("v4 checkpoint is missing objective_contract")
    primary = objective.get("primary_reconstruction")
    fidelity = objective.get("evssm_fidelity")
    temporal_objective = objective.get("temporal_delta")
    edge = objective.get("edge")
    laplacian = objective.get("laplacian_gate")
    distillation = objective.get("legacy_latest_evssm_distillation")
    if not all(
        isinstance(value, dict)
        for value in (
            primary,
            fidelity,
            temporal_objective,
            edge,
            laplacian,
            distillation,
        )
    ):
        raise ValueError("v4 objective is missing registered base loss sections")
    expected_base_weights = (
        (primary, "l1_weight", 1.0),
        (primary, "fft_l1_weight", 0.1),
        (fidelity, "weight", 0.1),
        (temporal_objective, "weight", 0.05),
        (edge, "weight", 0.05),
        (laplacian, "weight", 0.02),
        (distillation, "weight", 0.0),
    )
    for section, key, expected in expected_base_weights:
        if not math.isclose(
            _finite_float(section.get(key), f"v4 objective {key}"),
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"v4 objective {key} is not preregistered")
    if primary.get("fft_normalization") != "ortho":
        raise ValueError("v4 FFT objective must use norm='ortho'")
    if temporal_objective.get("reference") != (
        "motion_compensated_sharp_gt_difference"
    ) or temporal_objective.get("detached_flow") is not True or (
        temporal_objective.get("real_transitions_only") is not True
    ):
        raise ValueError("v4 temporal objective is not motion-aligned and detached")
    if edge.get("operator") != (
        "first_order_xy_plus_runtime_grayscale_zero_pad_laplacian"
    ) or laplacian.get("runtime_gate_alignment") != (
        "rgb_mean_then_four_neighbour_zero_pad_laplacian_unbiased_variance"
    ):
        raise ValueError("v4 edge/Laplacian objective is not runtime-aligned")

    alignment_objective = objective.get("motion_alignment")
    if not isinstance(alignment_objective, dict):
        raise ValueError("v4 objective is missing motion_alignment")
    expected_alignment_weights = {
        "photometric_weight": 1.0,
        "gradient_weight": 0.2,
        "smooth_weight": 0.01,
        "joint_phase_scale": 0.05,
    }
    for key, expected in expected_alignment_weights.items():
        if not math.isclose(
            _finite_float(alignment_objective.get(key), f"v4 objective {key}"),
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"v4 objective {key} is not preregistered")
    if alignment_objective.get("real_transitions_only") is not True:
        raise ValueError("v4 alignment objective must exclude repeated prefix slots")

    optimization = checkpoint.get("optimization_contract")
    if not isinstance(optimization, dict) or optimization.get("schema") != (
        OPTIMIZATION_CONTRACT_SCHEMA_V4
    ):
        raise ValueError("v4 checkpoint is missing optimization_contract")
    if optimization.get("optimizer") != "AdamW" or optimization.get(
        "optimizer_state_from_v3_loaded"
    ) is not False:
        raise ValueError("v4 optimization must use a fresh AdamW optimizer")
    if optimization.get("execution_device") != "cpu" or optimization.get(
        "amp_requested"
    ) is not False or optimization.get("amp_effective") is not False:
        raise ValueError(
            "v4 optimization must use the preregistered CPU execution without AMP"
        )
    expected_loader_contract = {
        "batch_size": 4,
        "num_workers": 0,
        "gradient_accumulation_micro_batches": 2,
        "effective_batch_size": 8,
        "drop_last": True,
        "alignment_loader_clips_per_epoch": 104,
        "alignment_loader_dropped_clips_per_epoch": 3,
        "alignment_micro_batches_per_epoch": 26,
        "joint_loader_clips_per_epoch": 232,
        "joint_loader_dropped_clips_per_epoch": 2,
        "joint_micro_batches_per_epoch": 58,
        "drop_incomplete_accumulation_group": False,
        "loader_generator_seeds": {
            "joint": 1042,
            "alignment_only": 2042,
        },
        "resume_boundary": "epoch_end_no_pending_accumulation",
        "resume_rng_state": (
            "python_numpy_torch_cpu_and_both_loader_generators"
        ),
    }
    if any(
        optimization.get(key) != expected
        for key, expected in expected_loader_contract.items()
    ):
        raise ValueError("v4 loader tail/effective-batch contract is not preregistered")
    if optimization.get("lr_schedule") != "fixed_by_phase" or optimization.get(
        "optimizer_reset_at_phase_boundary"
    ) is not False:
        raise ValueError("v4 optimizer schedule/phase transition is not preregistered")
    if int(optimization.get("total_optimizer_steps", 0)) != 600:
        raise ValueError("v4 optimization phase lengths are not preregistered")
    if not math.isclose(
        _finite_float(optimization.get("weight_decay"), "v4 weight_decay"),
        1.0e-3,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("v4 weight_decay is not preregistered")
    phases = optimization.get("phases")
    if not isinstance(phases, list) or len(phases) != 2:
        raise ValueError("v4 optimization must contain two phases")
    expected_phases = (
        {
            "name": "alignment_only",
            "start_step_inclusive": 0,
            "end_step_exclusive": 100,
            "optimizer_steps": 100,
            "base_trainable": False,
            "trainable_parameters": [
                "motion_aligner.match_projection.weight",
            ],
            "base_lr": 0.0,
            "alignment_lr": 2.0e-4,
        },
        {
            "name": "joint",
            "start_step_inclusive": 100,
            "end_step_exclusive": 600,
            "optimizer_steps": 500,
            "base_trainable": True,
            "trainable_parameters": [
                "base_parameters",
                "motion_aligner.match_projection.weight",
                "motion_alignment_gate",
            ],
            "base_lr": 2.0e-5,
            "alignment_lr": 2.0e-4,
        },
    )
    if tuple(phases) != expected_phases:
        raise ValueError("v4 optimizer phases/LRs are not preregistered")

    selection = checkpoint.get("checkpoint_selection")
    if not isinstance(selection, dict) or selection.get("deployment_status") != (
        "not_deployment_selected"
    ) or selection.get("required_deployment_selector") != (
        TRAINING_REQUIRED_SELECTOR_V4
    ):
        raise ValueError("v4 checkpoint does not require the v4 external selector")
    if checkpoint_sha256 is None:
        raise ValueError(
            "registered v4 export requires the checkpoint file SHA-256"
        )
    validate_registered_v4_checkpoint_migration(checkpoint, checkpoint_sha256)


def json_safe_validation_metrics(value: object) -> object:
    if not isinstance(value, dict):
        return None
    normalized: dict[str, object] = {}
    for key in ("psnr", "ssim"):
        try:
            metric = float(value[key])
        except (KeyError, TypeError, ValueError):
            normalized[key] = None
        else:
            normalized[key] = metric if math.isfinite(metric) else None
    return normalized


def _selection_metric(metrics: object, key: str, layer: str) -> float:
    if not isinstance(metrics, dict):
        raise ValueError(f"{layer} layer report is missing metrics")
    try:
        raw_value = metrics[key]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{layer} selection metric {key!r} is missing") from error
    if isinstance(raw_value, bool):
        raise ValueError(f"{layer} selection metric {key!r} must be numeric")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{layer} selection metric {key!r} is missing") from error
    if not math.isfinite(value):
        raise ValueError(f"{layer} selection metric {key!r} must be finite")
    return value


def _resolve_evidence_path(value: object, anchor: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = anchor.parent / path
    return path.resolve()


def _load_sha_bound_json(
    path: Path, expected_sha256: object, label: str
) -> dict[str, object]:
    digest = _sha256_digest(expected_sha256, f"{label}.sha256")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if sha256_file(path) != digest:
        raise ValueError(f"{label} SHA-256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _validate_evaluator_identity(
    payload: dict[str, object],
    report_path: Path,
    *,
    label: str,
    history: int,
    source_checkpoint_sha256: str,
    evssm_checkpoint_sha256: str,
    evaluator_schema: str = EVALUATOR_SCHEMA_V3,
) -> tuple[Path, str, Path]:
    if payload.get("schema") != evaluator_schema:
        raise ValueError(f"{label} has an unsupported evaluator schema")
    if payload.get("input_domain") != "evssm":
        raise ValueError(f"{label} must evaluate the EVSSM input domain")
    reported_history = payload.get("history")
    if type(reported_history) is not int or reported_history != history:
        raise ValueError(f"{label} history must be {history}")
    teacher = payload.get("teacher_provenance")
    if not isinstance(teacher, dict) or teacher.get(
        "evssm_checkpoint_sha256"
    ) != evssm_checkpoint_sha256:
        raise ValueError(f"{label} does not use the registered Unblur-SLAM EVSSM")
    if _sha256_digest(
        payload.get("source_checkpoint_sha256"),
        f"{label}.source_checkpoint_sha256",
    ) != source_checkpoint_sha256:
        raise ValueError(f"{label} uses a different source checkpoint")

    artifact = _resolve_evidence_path(
        payload.get("checkpoint"), report_path, f"{label}.checkpoint"
    )
    artifact_sha256 = _sha256_digest(
        payload.get("evaluated_artifact_sha256"),
        f"{label}.evaluated_artifact_sha256",
    )
    if not artifact.is_file():
        raise FileNotFoundError(f"{label} evaluated artifact does not exist: {artifact}")
    if sha256_file(artifact) != artifact_sha256:
        raise ValueError(f"{label} evaluated artifact SHA-256 mismatch")
    manifest = _resolve_evidence_path(
        payload.get("manifest"), report_path, f"{label}.manifest"
    )
    return artifact, artifact_sha256, manifest


def _index_evaluator_frames(
    payload: dict[str, object], label: str
) -> dict[tuple[str, int], dict[str, object]]:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{label}.frames must be a non-empty list")
    frame_count = payload.get("frame_count")
    if type(frame_count) is not int or frame_count != len(frames):
        raise ValueError(f"{label}.frame_count must match frames")
    indexed: dict[tuple[str, int], dict[str, object]] = {}
    for position, row in enumerate(frames):
        row_label = f"{label}.frames[{position}]"
        if not isinstance(row, dict):
            raise ValueError(f"{row_label} must be an object")
        sequence = row.get("sequence")
        frame_index = row.get("frame_index")
        if (
            not isinstance(sequence, str)
            or not sequence
            or type(frame_index) is not int
        ):
            raise ValueError(f"{row_label} has an invalid frame key")
        if frame_index < 0:
            raise ValueError(f"{row_label} has a negative frame index")
        key = (sequence, frame_index)
        if key in indexed:
            raise ValueError(f"{label} repeats frame {key!r}")
        if row.get("history_stage") not in {"prefix", "steady_state"}:
            raise ValueError(f"{row_label} has an invalid history_stage")
        for path_key in ("blurry_path", "sharp_path"):
            if not isinstance(row.get(path_key), str) or not row[path_key]:
                raise ValueError(f"{row_label}.{path_key} must be a path")
        causal = row.get("causal")
        if not isinstance(causal, dict):
            raise ValueError(f"{row_label}.causal must be an object")
        _selection_metric(causal, "psnr", row_label)
        indexed[key] = row
    return indexed


def _canonical_frame_keys_sha256(keys: list[tuple[str, int]]) -> str:
    canonical = json.dumps(
        [[sequence, frame_index] for sequence, frame_index in sorted(keys)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _same_number(actual: float, declared: object, label: str) -> None:
    try:
        expected = float(declared)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(expected) or not math.isclose(
        actual, expected, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError(f"{label} does not match evaluator rows")


def _validate_history1_control(
    *,
    layer_report: dict[str, object],
    layer_report_path: Path,
    metrics: dict[str, float],
    checkpoint_sha256: str,
    evssm_checkpoint_sha256: str,
    expected_manifest_sha256: str,
    h3_evaluator_schema: str = EVALUATOR_SCHEMA_V3,
) -> dict[str, object]:
    h3_entry = layer_report.get("evaluator_report")
    if not isinstance(h3_entry, dict):
        raise ValueError("temporal_validation is missing its H3 evaluator report")
    h3_report_path = _resolve_evidence_path(
        h3_entry.get("path"), layer_report_path, "H3 evaluator report"
    )
    h3_report_sha256 = _sha256_digest(
        h3_entry.get("sha256"), "H3 evaluator report SHA-256"
    )
    h3_payload = _load_sha_bound_json(
        h3_report_path, h3_report_sha256, "H3 evaluator report"
    )
    _, h3_artifact_sha256, h3_manifest = _validate_evaluator_identity(
        h3_payload,
        h3_report_path,
        label="H3 evaluator report",
        history=3,
        source_checkpoint_sha256=checkpoint_sha256,
        evssm_checkpoint_sha256=evssm_checkpoint_sha256,
        evaluator_schema=h3_evaluator_schema,
    )

    control = layer_report.get("history1_control")
    if not isinstance(control, dict):
        raise ValueError("temporal_validation is missing the H1 spatial control")
    h1_checkpoint_sha256 = _sha256_digest(
        control.get("checkpoint_sha256"), "history1_control.checkpoint_sha256"
    )
    if h1_checkpoint_sha256 == checkpoint_sha256:
        raise ValueError("H1 control must use a distinct training checkpoint")
    if control.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("H1 control does not use the temporal-validation manifest")
    if control.get("evssm_checkpoint_sha256") != evssm_checkpoint_sha256:
        raise ValueError("H1 control does not use the registered Unblur-SLAM EVSSM")
    if type(control.get("history")) is not int or control.get("history") != 1:
        raise ValueError("H1 control history must be the integer 1")
    h1_report_path = _resolve_evidence_path(
        control.get("evaluator_report"),
        layer_report_path,
        "H1 evaluator report",
    )
    h1_report_sha256 = _sha256_digest(
        control.get("evaluator_report_sha256"),
        "history1_control.evaluator_report_sha256",
    )
    h1_payload = _load_sha_bound_json(
        h1_report_path, h1_report_sha256, "H1 evaluator report"
    )
    h1_artifact, h1_artifact_sha256, h1_manifest = _validate_evaluator_identity(
        h1_payload,
        h1_report_path,
        label="H1 evaluator report",
        history=1,
        source_checkpoint_sha256=h1_checkpoint_sha256,
        evssm_checkpoint_sha256=evssm_checkpoint_sha256,
    )
    declared_h1_artifact = _resolve_evidence_path(
        control.get("evaluated_artifact"),
        layer_report_path,
        "history1_control.evaluated_artifact",
    )
    if declared_h1_artifact != h1_artifact:
        raise ValueError("H1 control is bound to a different evaluated artifact")
    if control.get("evaluated_artifact_sha256") != h1_artifact_sha256:
        raise ValueError("H1 control evaluated artifact SHA-256 mismatch")
    if h1_manifest != h3_manifest:
        raise ValueError("H1 and H3 evaluator reports use different manifests")

    h3_frames = _index_evaluator_frames(h3_payload, "H3 evaluator report")
    h1_frames = _index_evaluator_frames(h1_payload, "H1 evaluator report")
    if set(h3_frames) != set(h1_frames):
        raise ValueError("H1 and H3 evaluator reports cover different frame keys")
    steady_keys = sorted(
        key
        for key, row in h3_frames.items()
        if row.get("history_stage") == "steady_state"
    )
    if not steady_keys:
        raise ValueError("H3 evaluator report has no steady-state frames")
    for key in steady_keys:
        h3_row = h3_frames[key]
        h1_row = h1_frames[key]
        if h1_row.get("history_stage") != "steady_state":
            raise ValueError(f"H1 frame {key!r} is not steady-state")
        for path_key in ("blurry_path", "sharp_path"):
            if h1_row.get(path_key) != h3_row.get(path_key):
                raise ValueError(f"H1 and H3 frame {key!r} use different source paths")

    keys_sha256 = _canonical_frame_keys_sha256(steady_keys)
    if control.get("steady_frame_keys_sha256") != keys_sha256:
        raise ValueError("H1 control steady-frame key SHA-256 mismatch")
    h3_psnr = sum(
        _selection_metric(h3_frames[key]["causal"], "psnr", "H3 steady frame")
        for key in steady_keys
    ) / len(steady_keys)
    h1_psnr = sum(
        _selection_metric(h1_frames[key]["causal"], "psnr", "H1 steady frame")
        for key in steady_keys
    ) / len(steady_keys)
    delta = h3_psnr - h1_psnr
    _same_number(h3_psnr, control.get("h3_steady_psnr_db"), "H3 steady PSNR")
    _same_number(h1_psnr, control.get("h1_steady_psnr_db"), "H1 steady PSNR")
    _same_number(
        delta,
        metrics["normal_vs_history1_psnr_delta_db"],
        "H3-minus-H1 steady PSNR delta",
    )
    threshold = DEPLOYMENT_THRESHOLDS["temporal_validation"][
        "normal_vs_history1_psnr_delta_db_min"
    ]
    if delta < threshold:
        raise ValueError("checkpoint failed the H3-versus-H1 Layer1 selection")
    return {
        "checkpoint_sha256": h1_checkpoint_sha256,
        "evaluator_report_sha256": h1_report_sha256,
        "evaluated_artifact_sha256": h1_artifact_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "evssm_checkpoint_sha256": evssm_checkpoint_sha256,
        "history": 1,
        "steady_frame_keys_sha256": keys_sha256,
        "h3_steady_psnr_db": h3_psnr,
        "h1_steady_psnr_db": h1_psnr,
        "normal_vs_history1_psnr_delta_db": delta,
        "h3_evaluator_report_sha256": h3_report_sha256,
        "h3_evaluated_artifact_sha256": h3_artifact_sha256,
    }


def _mean(values: list[float], label: str) -> float:
    if not values:
        raise ValueError(f"cannot compute {label} from zero samples")
    result = sum(values) / len(values)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _relative(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    result = numerator / denominator - 1.0
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _room2_metric(
    row: dict[str, object], source: str, metric: str, label: str
) -> float:
    source_metrics = row.get(source)
    if not isinstance(source_metrics, dict):
        raise ValueError(f"{label}.{source} must be an object")
    return _selection_metric(source_metrics, metric, f"{label}.{source}")


def _room2_temporal_metric(
    row: dict[str, object], source: str, label: str
) -> float:
    temporal = row.get("temporal")
    if not isinstance(temporal, dict):
        raise ValueError(f"{label}.temporal must be an object")
    source_metrics = temporal.get(source)
    if not isinstance(source_metrics, dict):
        raise ValueError(f"{label}.temporal.{source} must be an object")
    return _selection_metric(
        source_metrics,
        "gt_difference_error_l1_not_warp",
        f"{label}.temporal.{source}",
    )


def _canonical_frame_identity_sha256(
    frames: dict[tuple[str, int], dict[str, object]], report_path: Path
) -> str:
    identities = []
    for (sequence, frame_index), row in sorted(frames.items()):
        blurry = _resolve_evidence_path(
            row.get("blurry_path"), report_path, "room2 blurry_path"
        )
        sharp = _resolve_evidence_path(
            row.get("sharp_path"), report_path, "room2 sharp_path"
        )
        identities.append([sequence, frame_index, str(blurry), str(sharp)])
    canonical = json.dumps(
        identities, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_room2_rows(
    payload: dict[str, object], report_path: Path
) -> list[dict[str, object]]:
    indexed = _index_evaluator_frames(payload, "room2 evaluator report")
    if len(indexed) != ROOM2_ONE_SHOT_FRAME_COUNT:
        raise ValueError(
            f"room2 evaluator must contain {ROOM2_ONE_SHOT_FRAME_COUNT} frames"
        )
    identity_sha256 = _canonical_frame_identity_sha256(indexed, report_path)
    if identity_sha256 != ROOM2_ONE_SHOT_FRAME_IDENTITY_SHA256:
        raise ValueError("room2 evaluator frame keys/paths are not canonical")

    temporal_pair_count = 0
    rows = []
    for key, row in sorted(indexed.items()):
        label = f"room2 frame {key!r}"
        expected_stage = "prefix" if key[1] < 2 else "steady_state"
        if row.get("history_stage") != expected_stage:
            raise ValueError(f"{label} has the wrong H3 history_stage")
        for source in ("evssm", "causal", "causal_repeat_current"):
            for metric in ("psnr", "ssim", "l1", "lpips"):
                value = _room2_metric(row, source, metric, label)
                if metric in {"l1", "lpips"} and value < 0.0:
                    raise ValueError(f"{label}.{source}.{metric} must be non-negative")

        gate = row.get("runtime_gate_proxy")
        if not isinstance(gate, dict):
            raise ValueError(f"{label}.runtime_gate_proxy must be an object")
        laplacian = {
            source: _selection_metric(
                gate, f"{source}_laplacian_variance", f"{label}.runtime_gate_proxy"
            )
            for source in ("blurry", "evssm", "causal")
        }
        if any(value < 0.0 for value in laplacian.values()):
            raise ValueError(f"{label} Laplacian variances must be non-negative")
        gain_evssm = (laplacian["causal"] - laplacian["evssm"]) / max(
            laplacian["evssm"], 1.0e-12
        )
        gain_blurry = (laplacian["causal"] - laplacian["blurry"]) / max(
            laplacian["blurry"], 1.0e-12
        )
        _same_number(
            gain_evssm,
            gate.get("causal_vs_evssm_gain"),
            f"{label} causal-vs-EVSSM Laplacian gain",
        )
        _same_number(
            gain_blurry,
            gate.get("causal_vs_blurry_gain"),
            f"{label} causal-vs-blurry Laplacian gain",
        )
        expected_pass = gain_evssm >= 0.0 and gain_blurry >= 0.02
        if gate.get("passes_default_gate") is not expected_pass:
            raise ValueError(f"{label} has an inconsistent runtime gate decision")

        if key[1] == 0:
            if row.get("temporal") is not None:
                raise ValueError(f"{label}.temporal must reset at the run boundary")
        else:
            temporal_pair_count += 1
            for source in ("evssm", "causal", "causal_repeat_current"):
                value = _room2_temporal_metric(row, source, label)
                if value < 0.0:
                    raise ValueError(f"{label} temporal error must be non-negative")
        rows.append(row)
    if type(payload.get("temporal_pair_count")) is not int or payload.get(
        "temporal_pair_count"
    ) != temporal_pair_count:
        raise ValueError("room2 evaluator temporal_pair_count is inconsistent")
    return rows


def _room2_quality(
    rows: list[dict[str, object]], *, require_lpips: bool
) -> dict[str, object]:
    means: dict[str, dict[str, float]] = {}
    missing = []
    for source in ("evssm", "causal", "causal_repeat_current"):
        means[source] = {}
        for metric in ("psnr", "ssim", "l1", "lpips"):
            values = []
            for row in rows:
                try:
                    values.append(_room2_metric(row, source, metric, "room2 frame"))
                except ValueError:
                    if metric != "lpips":
                        raise
                    missing.append(f"frames[].{source}.lpips")
                    values = []
                    break
            if values:
                means[source][metric] = _mean(values, f"room2 {source}.{metric}")
    missing = sorted(set(missing)) if require_lpips else []
    delta_lpips = (
        means["causal"]["lpips"] - means["evssm"]["lpips"]
        if "lpips" in means["causal"] and "lpips" in means["evssm"]
        else None
    )
    return {
        "frame_count": len(rows),
        "mean": means,
        "delta_psnr_db": means["causal"]["psnr"] - means["evssm"]["psnr"],
        "delta_ssim": means["causal"]["ssim"] - means["evssm"]["ssim"],
        "relative_l1": _relative(
            means["causal"]["l1"], means["evssm"]["l1"], "room2 relative L1"
        ),
        "history_delta_psnr_db": (
            means["causal"]["psnr"]
            - means["causal_repeat_current"]["psnr"]
        ),
        "missing_metrics": missing,
        "delta_lpips": delta_lpips,
    }


def _room2_temporal(rows: list[dict[str, object]]) -> dict[str, object]:
    temporal_rows = [row for row in rows if row.get("temporal") is not None]
    means = {
        source: _mean(
            [
                _room2_temporal_metric(row, source, "room2 temporal frame")
                for row in temporal_rows
            ],
            f"room2 {source} temporal error",
        )
        for source in ("evssm", "causal", "causal_repeat_current")
    }
    return {
        "pair_count": len(temporal_rows),
        "evssm_gt_difference_error_l1": means["evssm"],
        "causal_gt_difference_error_l1": means["causal"],
        "repeat_current_gt_difference_error_l1": means[
            "causal_repeat_current"
        ],
        "causal_minus_evssm_gt_difference_error_l1": (
            means["causal"] - means["evssm"]
        ),
        "causal_relative_to_evssm": _relative(
            means["causal"], means["evssm"], "room2 temporal vs EVSSM"
        ),
        "causal_relative_to_repeat_current": _relative(
            means["causal"],
            means["causal_repeat_current"],
            "room2 temporal vs repeat-current",
        ),
    }


def _room2_gate_and_oracle(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    accepted = [
        row
        for row in rows
        if row["runtime_gate_proxy"]["passes_default_gate"] is True
    ]
    decisions = []
    for row in accepted:
        evssm_l1 = _room2_metric(row, "evssm", "l1", "oracle frame")
        relative_l1 = _relative(
            _room2_metric(row, "causal", "l1", "oracle frame"),
            evssm_l1,
            "oracle relative L1",
        )
        delta_psnr = _room2_metric(
            row, "causal", "psnr", "oracle frame"
        ) - _room2_metric(row, "evssm", "psnr", "oracle frame")
        delta_ssim = _room2_metric(
            row, "causal", "ssim", "oracle frame"
        ) - _room2_metric(row, "evssm", "ssim", "oracle frame")
        good = (
            delta_psnr >= ORACLE_GOOD_DEFINITION["delta_psnr_db_min"]
            and delta_ssim >= ORACLE_GOOD_DEFINITION["delta_ssim_min"]
            and relative_l1 <= ORACLE_GOOD_DEFINITION["relative_l1_max"]
        )
        decisions.append(
            {
                "sequence": row["sequence"],
                "frame_index": row["frame_index"],
                "delta_psnr_db": delta_psnr,
                "delta_ssim": delta_ssim,
                "relative_l1": relative_l1,
                "oracle_good": bool(good),
            }
        )
    good_count = sum(item["oracle_good"] is True for item in decisions)
    return {
        "frame_count": len(rows),
        "pass_count": len(accepted),
        "pass_ratio": len(accepted) / len(rows),
        "oracle_good_count": good_count,
        "oracle_precision": good_count / len(accepted) if accepted else 0.0,
        "oracle_definition": dict(ORACLE_GOOD_DEFINITION),
        "accepted_frames": decisions,
    }


def _room2_per_run(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sequence"]), []).append(row)
    runs = []
    for sequence, run_rows in sorted(grouped.items()):
        steady = [row for row in run_rows if row["history_stage"] == "steady_state"]
        runs.append(
            {
                "sequence": sequence,
                "frame_count": len(run_rows),
                "steady_count": len(steady),
                "delta_psnr_db": _mean(
                    [
                        _room2_metric(row, "causal", "psnr", sequence)
                        - _room2_metric(row, "evssm", "psnr", sequence)
                        for row in run_rows
                    ],
                    f"{sequence} delta PSNR",
                ),
                "steady_delta_psnr_db": (
                    _mean(
                        [
                            _room2_metric(row, "causal", "psnr", sequence)
                            - _room2_metric(row, "evssm", "psnr", sequence)
                            for row in steady
                        ],
                        f"{sequence} steady delta PSNR",
                    )
                    if steady
                    else None
                ),
            }
        )
    return runs


def _validate_room2_evidence(
    *,
    layer_report: dict[str, object],
    layer_report_path: Path,
    metrics: dict[str, float],
    checkpoint_sha256: str,
    evssm_checkpoint_sha256: str,
    expected_manifest_sha256: str,
    expected_candidate_artifact_sha256: str,
    evaluator_schema: str = EVALUATOR_SCHEMA_V3,
) -> dict[str, object]:
    if layer_report.get("oracle_good_definition") != ORACLE_GOOD_DEFINITION:
        raise ValueError("room2 oracle-good definition is not preregistered")
    if layer_report.get("lpips_required") is not True or layer_report.get(
        "lpips_computed"
    ) is not True:
        raise ValueError("room2 selection requires computed LPIPS")
    if layer_report.get("missing_metrics") != []:
        raise ValueError("room2 selection report has missing metrics")
    if type(layer_report.get("history")) is not int or layer_report.get(
        "history"
    ) != 3:
        raise ValueError("room2 layer history must be 3")
    if layer_report.get("input_domain") != "evssm":
        raise ValueError("room2 layer must use the EVSSM input domain")

    source_manifest = layer_report.get("source_manifest")
    if not isinstance(source_manifest, dict):
        raise ValueError("room2 layer is missing source_manifest evidence")
    if source_manifest.get("sha256") != expected_manifest_sha256:
        raise ValueError("room2 source manifest SHA-256 is not preregistered")
    source_root = _resolve_evidence_path(
        source_manifest.get("source_root"),
        layer_report_path,
        "room2 source_root",
    )
    if source_root != ROOM2_ONE_SHOT_SOURCE_ROOT.resolve():
        raise ValueError("room2 source_root is not preregistered")
    if source_manifest.get("frame_count") != ROOM2_ONE_SHOT_FRAME_COUNT:
        raise ValueError("room2 source manifest must contain 174 frames")
    if source_manifest.get("frame_identity_schema") != (
        "sorted_compact_json_sequence_index_blurry_sharp.v1"
    ) or source_manifest.get("frame_identity_sha256") != (
        ROOM2_ONE_SHOT_FRAME_IDENTITY_SHA256
    ):
        raise ValueError("room2 source-manifest frame identity is not preregistered")

    entry = layer_report.get("evaluator_report")
    if not isinstance(entry, dict):
        raise ValueError("room2 layer is missing evaluator_report evidence")
    report_path = _resolve_evidence_path(
        entry.get("path"), layer_report_path, "room2 evaluator report"
    )
    report_sha256 = _sha256_digest(
        entry.get("sha256"), "room2 evaluator report SHA-256"
    )
    payload = _load_sha_bound_json(
        report_path, report_sha256, "room2 evaluator report"
    )
    _, artifact_sha256, _ = _validate_evaluator_identity(
        payload,
        report_path,
        label="room2 evaluator report",
        history=3,
        source_checkpoint_sha256=checkpoint_sha256,
        evssm_checkpoint_sha256=evssm_checkpoint_sha256,
        evaluator_schema=evaluator_schema,
    )
    if artifact_sha256 != expected_candidate_artifact_sha256:
        raise ValueError("room2 evaluator uses a different candidate artifact")
    if payload.get("lpips_computed") is not True:
        raise ValueError("room2 evaluator did not compute LPIPS")
    if payload.get("lpips_protocol") != EXPECTED_LPIPS_PROTOCOL:
        raise ValueError("room2 evaluator uses an unsupported LPIPS protocol")

    rows = _validate_room2_rows(payload, report_path)
    quality_all = _room2_quality(rows, require_lpips=True)
    steady = [row for row in rows if row["history_stage"] == "steady_state"]
    quality_steady = _room2_quality(steady, require_lpips=False)
    temporal = _room2_temporal(rows)
    gate = _room2_gate_and_oracle(rows)
    runs = _room2_per_run(rows)
    long_runs = [run for run in runs if int(run["frame_count"]) >= 3]
    nondegraded = sum(
        float(run["steady_delta_psnr_db"]) >= 0.0 for run in long_runs
    )
    recomputed_metrics = {
        "psnr_delta_db": quality_all["delta_psnr_db"],
        "ssim_delta": quality_all["delta_ssim"],
        "relative_l1_delta": quality_all["relative_l1"],
        "lpips_delta": quality_all["delta_lpips"],
        "gt_temporal_difference_delta": temporal[
            "causal_minus_evssm_gt_difference_error_l1"
        ],
        "steady_normal_vs_repeat_current_psnr_delta_db": quality_steady[
            "history_delta_psnr_db"
        ],
        "laplacian_gate_pass_ratio": gate["pass_ratio"],
        "accepted_oracle_precision": gate["oracle_precision"],
        "nondegraded_long_runs": nondegraded,
        "long_runs_total": len(long_runs),
        "worst_run_psnr_delta_db": min(
            float(run["delta_psnr_db"]) for run in runs
        ),
    }
    for key, value in recomputed_metrics.items():
        _same_number(float(value), metrics[key], f"room2 metric {key}")

    details = layer_report.get("details")
    if not isinstance(details, dict):
        raise ValueError("room2 layer is missing recomputed details")
    declared_gate = details.get("gate_and_oracle_all_frames")
    if not isinstance(declared_gate, dict) or declared_gate.get(
        "oracle_definition"
    ) != ORACLE_GOOD_DEFINITION:
        raise ValueError("room2 details use an unregistered oracle definition")
    return {
        "evaluator_report_sha256": report_sha256,
        "evaluated_artifact_sha256": artifact_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "frame_count": len(rows),
        "frame_identity_sha256": ROOM2_ONE_SHOT_FRAME_IDENTITY_SHA256,
        "lpips_protocol": dict(EXPECTED_LPIPS_PROTOCOL),
        "oracle_good_definition": dict(ORACLE_GOOD_DEFINITION),
        "metrics": recomputed_metrics,
        "gate_pass_count": gate["pass_count"],
        "oracle_good_count": gate["oracle_good_count"],
        "per_run_count": len(runs),
    }


def _validate_temporal_layer(metrics: object) -> dict[str, float]:
    keys = (
        "steady_psnr_delta_db",
        "steady_ssim_delta",
        "steady_relative_l1_delta",
        "steady_gt_temporal_difference_relative_delta",
        "normal_vs_repeat_current_psnr_delta_db",
        "normal_vs_repeat_current_temporal_relative_delta",
        "normal_vs_history1_psnr_delta_db",
        "laplacian_gate_pass_ratio",
        "accepted_oracle_precision",
        "worst_run_psnr_delta_db",
        "prefix_psnr_delta_db",
    )
    values = {
        key: _selection_metric(metrics, key, "temporal_validation") for key in keys
    }
    thresholds = DEPLOYMENT_THRESHOLDS["temporal_validation"]
    eligible = (
        values["steady_psnr_delta_db"] >= thresholds["steady_psnr_delta_db_min"]
        and values["steady_ssim_delta"] >= thresholds["steady_ssim_delta_min"]
        and values["steady_relative_l1_delta"]
        <= thresholds["steady_relative_l1_delta_max"]
        and values["steady_gt_temporal_difference_relative_delta"]
        <= thresholds["steady_gt_temporal_difference_relative_delta_max"]
        and values["normal_vs_repeat_current_psnr_delta_db"]
        >= thresholds["normal_vs_repeat_current_psnr_delta_db_min"]
        and values["normal_vs_repeat_current_temporal_relative_delta"]
        <= thresholds["normal_vs_repeat_current_temporal_relative_delta_max"]
        and values["normal_vs_history1_psnr_delta_db"]
        >= thresholds["normal_vs_history1_psnr_delta_db_min"]
        and values["laplacian_gate_pass_ratio"]
        >= thresholds["laplacian_gate_pass_ratio_min"]
        and values["accepted_oracle_precision"]
        >= thresholds["accepted_oracle_precision_min"]
        and values["worst_run_psnr_delta_db"]
        >= thresholds["worst_run_psnr_delta_db_min"]
        and values["prefix_psnr_delta_db"]
        >= thresholds["prefix_psnr_delta_db_min"]
    )
    if not 0.0 <= values["laplacian_gate_pass_ratio"] <= 1.0:
        raise ValueError("temporal_validation Laplacian pass ratio must be in [0,1]")
    if not 0.0 <= values["accepted_oracle_precision"] <= 1.0:
        raise ValueError("temporal_validation oracle precision must be in [0,1]")
    if not eligible:
        raise ValueError("checkpoint failed temporal-validation Layer1 selection")
    return values


def _validate_room2_layer(metrics: object) -> dict[str, float]:
    keys = (
        "psnr_delta_db",
        "ssim_delta",
        "relative_l1_delta",
        "lpips_delta",
        "gt_temporal_difference_delta",
        "steady_normal_vs_repeat_current_psnr_delta_db",
        "laplacian_gate_pass_ratio",
        "accepted_oracle_precision",
        "nondegraded_long_runs",
        "long_runs_total",
        "worst_run_psnr_delta_db",
    )
    values = {key: _selection_metric(metrics, key, "room2_one_shot") for key in keys}
    thresholds = DEPLOYMENT_THRESHOLDS["room2_one_shot"]
    eligible = (
        values["psnr_delta_db"] >= thresholds["psnr_delta_db_min"]
        and values["ssim_delta"] >= thresholds["ssim_delta_min"]
        and values["relative_l1_delta"] <= thresholds["relative_l1_delta_max"]
        and values["lpips_delta"] <= thresholds["lpips_delta_max"]
        and values["gt_temporal_difference_delta"]
        <= thresholds["gt_temporal_difference_delta_max"]
        and values["steady_normal_vs_repeat_current_psnr_delta_db"]
        >= thresholds["steady_normal_vs_repeat_current_psnr_delta_db_min"]
        and values["laplacian_gate_pass_ratio"]
        >= thresholds["laplacian_gate_pass_ratio_min"]
        and values["accepted_oracle_precision"]
        >= thresholds["accepted_oracle_precision_min"]
        and values["nondegraded_long_runs"]
        >= thresholds["nondegraded_long_runs_min"]
        and values["long_runs_total"] == thresholds["long_runs_total"]
        and values["worst_run_psnr_delta_db"]
        >= thresholds["worst_run_psnr_delta_db_min"]
    )
    if not 0.0 <= values["laplacian_gate_pass_ratio"] <= 1.0:
        raise ValueError("room2_one_shot Laplacian pass ratio must be in [0,1]")
    if not 0.0 <= values["accepted_oracle_precision"] <= 1.0:
        raise ValueError("room2_one_shot oracle precision must be in [0,1]")
    for key in ("nondegraded_long_runs", "long_runs_total"):
        if not values[key].is_integer():
            raise ValueError(f"room2_one_shot {key} must be an integer count")
    if values["nondegraded_long_runs"] > values["long_runs_total"]:
        raise ValueError("room2 nondegraded run count exceeds total runs")
    if not eligible:
        raise ValueError("checkpoint failed room2 one-shot Layer2 selection")
    return values


def _validate_v4_alignment_evidence(
    payload: dict[str, object],
    *,
    label: str,
    expected_transition_count: Optional[int] = None,
    require_lpips: bool = False,
) -> dict[str, object]:
    """Validate v4 alignment diagnostics and both registered controls.

    Alignment statistics are diagnostic rather than post-hoc performance
    thresholds.  The export gate nevertheless verifies their identity,
    finiteness, range, configured flow bound, and one-to-one coverage of real
    adjacent transitions.
    """

    if payload.get("schema") != EVALUATOR_SCHEMA_V4:
        raise ValueError(f"{label} must use the v4 evaluator schema")
    indexed = _index_evaluator_frames(payload, label)
    transition_rows: list[dict[str, object]] = []
    for key, row in sorted(indexed.items()):
        disabled = row.get("causal_alignment_disabled")
        if not isinstance(disabled, dict):
            raise ValueError(f"{label} frame {key!r} lacks alignment-disabled control")
        for metric in ("psnr", "ssim", "l1"):
            _selection_metric(disabled, metric, f"{label} alignment-disabled")
        if require_lpips:
            _selection_metric(disabled, "lpips", f"{label} alignment-disabled")
        alignment = row.get("motion_alignment")
        if key[1] == 0:
            if alignment is not None:
                raise ValueError(f"{label} first frame {key!r} has a transition")
        else:
            if not isinstance(alignment, dict) or alignment.get("schema") != (
                ALIGNMENT_DIAGNOSTICS_SCHEMA_V4
            ):
                raise ValueError(f"{label} frame {key!r} lacks v4 motion evidence")
            if alignment.get("from_frame_index") != key[1] - 1 or alignment.get(
                "to_frame_index"
            ) != key[1]:
                raise ValueError(f"{label} frame {key!r} has wrong transition identity")
            if alignment.get("finite_fraction") != 1.0:
                raise ValueError(f"{label} frame {key!r} has non-finite motion")
            quarter = alignment.get("flow_quarter_pixels")
            input_flow = alignment.get("flow_input_pixels")
            confidence = alignment.get("confidence")
            warp_valid = alignment.get("warp_valid")
            if not all(
                isinstance(value, dict)
                for value in (quarter, input_flow, confidence, warp_valid)
            ):
                raise ValueError(f"{label} frame {key!r} has incomplete motion stats")
            quarter_p95 = _selection_metric(
                quarter, "magnitude_p95", f"{label} quarter flow"
            )
            quarter_max = _selection_metric(
                quarter, "magnitude_max", f"{label} quarter flow"
            )
            quarter_component = _selection_metric(
                quarter, "component_abs_max", f"{label} quarter flow"
            )
            input_p95 = _selection_metric(
                input_flow, "magnitude_p95", f"{label} input flow"
            )
            input_max = _selection_metric(
                input_flow, "magnitude_max", f"{label} input flow"
            )
            input_dx_max = _selection_metric(
                input_flow, "dx_abs_max", f"{label} input flow"
            )
            input_dy_max = _selection_metric(
                input_flow, "dy_abs_max", f"{label} input flow"
            )
            flow_shape = alignment.get("flow_shape")
            input_shape = alignment.get("input_shape")
            scales = alignment.get("quarter_to_input_scale")
            if (
                not isinstance(flow_shape, list)
                or len(flow_shape) != 3
                or flow_shape[0] != 2
                or not all(type(value) is int and value > 0 for value in flow_shape)
                or not isinstance(input_shape, list)
                or len(input_shape) != 3
                or input_shape[0] != 3
                or not all(type(value) is int and value > 0 for value in input_shape)
                or not isinstance(scales, dict)
            ):
                raise ValueError(f"{label} frame {key!r} has invalid flow/input shape")
            expected_scale_x = float(input_shape[2]) / float(flow_shape[2])
            expected_scale_y = float(input_shape[1]) / float(flow_shape[1])
            scale_x = _selection_metric(scales, "x", f"{label} input flow scale")
            scale_y = _selection_metric(scales, "y", f"{label} input flow scale")
            if not math.isclose(
                scale_x, expected_scale_x, rel_tol=0.0, abs_tol=1.0e-5
            ) or not math.isclose(
                scale_y, expected_scale_y, rel_tol=0.0, abs_tol=1.0e-5
            ):
                raise ValueError(f"{label} frame {key!r} has inconsistent input scale")
            if (
                min(
                    quarter_p95,
                    quarter_max,
                    quarter_component,
                    input_p95,
                    input_max,
                    input_dx_max,
                    input_dy_max,
                )
                < 0.0
                or quarter_p95 > quarter_max
                or input_p95 > input_max
                or quarter_component > 16.0 + 1.0e-5
                or input_dx_max > 16.0 * scale_x + 1.0e-5
                or input_dy_max > 16.0 * scale_y + 1.0e-5
                or input_max
                > math.sqrt((16.0 * scale_x) ** 2 + (16.0 * scale_y) ** 2)
                + 1.0e-5
            ):
                raise ValueError(f"{label} frame {key!r} has invalid flow statistics")
            confidence_values = {
                name: _selection_metric(
                    confidence, name, f"{label} alignment confidence"
                )
                for name in ("mean", "p05", "p50", "p95", "min", "max")
            }
            valid_values = {
                name: _selection_metric(
                    warp_valid, name, f"{label} warp-valid"
                )
                for name in ("mean", "min", "max")
            }
            if any(
                value < 0.0 or value > 1.0
                for value in (*confidence_values.values(), *valid_values.values())
            ):
                raise ValueError(f"{label} frame {key!r} has out-of-range masks")
            transition_rows.append(alignment)

    transition_count = len(transition_rows)
    if type(payload.get("transition_count")) is not int or payload.get(
        "transition_count"
    ) != transition_count:
        raise ValueError(f"{label} transition_count is inconsistent")
    if type(payload.get("temporal_pair_count")) is not int or payload.get(
        "temporal_pair_count"
    ) != transition_count:
        raise ValueError(f"{label} transitions must equal real temporal pairs")
    if expected_transition_count is not None and transition_count != (
        expected_transition_count
    ):
        raise ValueError(
            f"{label} must contain {expected_transition_count} real transitions"
        )

    summary = payload.get("alignment_diagnostics")
    if not isinstance(summary, dict) or summary.get("schema") != (
        ALIGNMENT_DIAGNOSTICS_SCHEMA_V4
    ):
        raise ValueError(f"{label} is missing aggregate alignment diagnostics")
    if summary.get("transition_count") != transition_count or summary.get(
        "expected_transition_count"
    ) != transition_count:
        raise ValueError(f"{label} aggregate transition count is inconsistent")
    integrity = summary.get("integrity")
    expected_integrity = {
        "transition_count_matches": True,
        "all_finite": True,
        "flow_within_configured_bound": True,
        "confidence_in_0_1": True,
        "warp_valid_in_0_1": True,
        "passed": True,
    }
    if integrity != expected_integrity:
        raise ValueError(f"{label} alignment integrity did not pass")
    aggregate_quarter = summary.get("flow_quarter_pixels")
    aggregate_input = summary.get("flow_input_pixels")
    aggregate_confidence = summary.get("confidence")
    aggregate_valid = summary.get("warp_valid")
    if not all(
        isinstance(value, dict)
        for value in (
            aggregate_quarter,
            aggregate_input,
            aggregate_confidence,
            aggregate_valid,
        )
    ):
        raise ValueError(f"{label} aggregate alignment statistics are incomplete")
    for stats, names, stats_label in (
        (
            aggregate_quarter,
            ("magnitude_p95", "magnitude_max", "component_abs_max"),
            "quarter flow",
        ),
        (aggregate_input, ("magnitude_p95", "magnitude_max"), "input flow"),
        (
            aggregate_confidence,
            ("mean", "p05", "p50", "p95", "min", "max"),
            "confidence",
        ),
        (aggregate_valid, ("mean", "min", "max"), "warp-valid"),
    ):
        for name in names:
            _selection_metric(stats, name, f"{label} aggregate {stats_label}")
        if stats.get("finite_fraction") != 1.0:
            raise ValueError(f"{label} aggregate {stats_label} is not finite")
    if _selection_metric(
        aggregate_quarter, "component_abs_max", f"{label} aggregate quarter flow"
    ) > 16.0 + 1.0e-5 or aggregate_quarter.get(
        "configured_component_abs_max"
    ) != 16.0:
        raise ValueError(f"{label} aggregate flow exceeds the v4 contract")

    controls = summary.get("controls")
    disabled_control = payload.get("alignment_disabled_control")
    if not isinstance(controls, dict) or not isinstance(
        controls.get("repeat_current"), dict
    ) or not isinstance(controls.get("alignment_disabled"), dict):
        raise ValueError(f"{label} is missing repeat/alignment-disabled controls")
    if controls.get("alignment_disabled") != disabled_control:
        raise ValueError(f"{label} alignment-disabled control is inconsistent")
    return {
        "schema": ALIGNMENT_DIAGNOSTICS_SCHEMA_V4,
        "transition_count": transition_count,
        "flow_quarter_magnitude_p95": _selection_metric(
            aggregate_quarter, "magnitude_p95", f"{label} aggregate quarter flow"
        ),
        "flow_quarter_magnitude_max": _selection_metric(
            aggregate_quarter, "magnitude_max", f"{label} aggregate quarter flow"
        ),
        "flow_input_magnitude_p95": _selection_metric(
            aggregate_input, "magnitude_p95", f"{label} aggregate input flow"
        ),
        "flow_input_magnitude_max": _selection_metric(
            aggregate_input, "magnitude_max", f"{label} aggregate input flow"
        ),
        "confidence_mean": _selection_metric(
            aggregate_confidence, "mean", f"{label} aggregate confidence"
        ),
        "warp_valid_mean": _selection_metric(
            aggregate_valid, "mean", f"{label} aggregate warp-valid"
        ),
        "integrity_passed": True,
    }


def _validate_v4_layer_checkpoint_migration(
    layer_report: dict[str, object],
    expected_migration: dict[str, object],
) -> dict[str, object]:
    """Require selector evidence to repeat the exact checkpoint lineage."""

    provenance = layer_report.get("v4_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("v4 Layer1 is missing checkpoint provenance")
    target_sha256 = expected_migration.get("target_checkpoint_sha256")
    if provenance.get("source_checkpoint_sha256") != target_sha256:
        raise ValueError(
            "v4 Layer1 provenance is bound to a different tensor-safe checkpoint"
        )
    migration = provenance.get("checkpoint_migration")
    if not isinstance(migration, dict) or migration != expected_migration:
        raise ValueError(
            "v4 Layer1 checkpoint_migration does not match the checkpoint payload"
        )
    return dict(migration)


def _resolve_layer_report(
    *,
    selection_report_path: Path,
    layer_name: str,
    layer_entry: object,
    expected_manifest_sha256: str,
    checkpoint_sha256: str,
    evssm_checkpoint_sha256: str,
    preceding_report_sha256: Optional[str] = None,
    expected_candidate_artifact_sha256: Optional[str] = None,
    layer_report_schema: str = DEPLOYMENT_LAYER_REPORT_SCHEMA_V1,
    registered_contract_sha256: str = REGISTERED_CONTRACT_SHA256,
    h3_evaluator_schema: str = EVALUATOR_SCHEMA_V3,
    require_v4_alignment_diagnostics: bool = False,
    expected_v4_checkpoint_migration: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    if not isinstance(layer_entry, dict):
        raise ValueError(f"deployment selection is missing {layer_name} layer")
    if layer_entry.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError(f"{layer_name} manifest SHA-256 is not preregistered")
    expected_report_sha256 = _sha256_digest(
        layer_entry.get("report_sha256"), f"{layer_name}.report_sha256"
    )
    report_value = layer_entry.get("report")
    if not report_value:
        raise ValueError(f"{layer_name} layer is missing report path")
    report_path = Path(str(report_value)).expanduser()
    if not report_path.is_absolute():
        report_path = selection_report_path.parent / report_path
    report_path = report_path.resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"{layer_name} layer report does not exist: {report_path}")
    if sha256_file(report_path) != expected_report_sha256:
        raise ValueError(f"{layer_name} layer report SHA-256 mismatch")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{layer_name} layer report is not valid JSON") from error
    if not isinstance(report, dict) or report.get("schema") != layer_report_schema:
        raise ValueError(f"{layer_name} layer report has an unsupported schema")
    if layer_report_schema == DEPLOYMENT_LAYER_REPORT_SCHEMA_V4 and report.get(
        "policy"
    ) != DEPLOYMENT_SELECTION_POLICY_V4:
        raise ValueError(f"{layer_name} v4 layer policy is not preregistered")
    if report.get("layer") != layer_name:
        raise ValueError(f"{layer_name} layer report has the wrong role")
    if report.get("registered_contract_sha256") != registered_contract_sha256:
        raise ValueError(f"{layer_name} layer report is not bound to the contract")
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"{layer_name} layer report uses a different checkpoint")
    if report.get("evssm_checkpoint_sha256") != evssm_checkpoint_sha256:
        raise ValueError(f"{layer_name} layer report uses a different EVSSM baseline")
    if report.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError(f"{layer_name} layer report uses a different manifest")
    if report.get("thresholds") != DEPLOYMENT_THRESHOLDS[layer_name]:
        raise ValueError(f"{layer_name} layer thresholds are not preregistered")
    if report.get("eligible") is not True:
        raise ValueError(f"{layer_name} layer report is not eligible")

    checkpoint_migration = None
    if layer_name == "temporal_validation":
        if report.get("role") != "checkpoint_and_history_selection":
            raise ValueError("temporal_validation report has the wrong selection role")
        metrics = _validate_temporal_layer(report.get("metrics"))
        history1_control = _validate_history1_control(
            layer_report=report,
            layer_report_path=report_path,
            metrics=metrics,
            checkpoint_sha256=checkpoint_sha256,
            evssm_checkpoint_sha256=evssm_checkpoint_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
            h3_evaluator_schema=h3_evaluator_schema,
        )
        if require_v4_alignment_diagnostics:
            if expected_v4_checkpoint_migration is not None:
                checkpoint_migration = _validate_v4_layer_checkpoint_migration(
                    report, expected_v4_checkpoint_migration
                )
            else:
                checkpoint_migration = None
            h3_entry = report.get("evaluator_report")
            if not isinstance(h3_entry, dict):
                raise ValueError("v4 temporal layer is missing evaluator evidence")
            h3_report_path = _resolve_evidence_path(
                h3_entry.get("path"), report_path, "v4 H3 evaluator report"
            )
            h3_payload = _load_sha_bound_json(
                h3_report_path,
                h3_entry.get("sha256"),
                "v4 H3 evaluator report",
            )
            alignment_evidence = _validate_v4_alignment_evidence(
                h3_payload,
                label="v4 temporal evaluator",
                expected_transition_count=14,
                require_lpips=False,
            )
            if report.get("alignment_integrity_passed") is not True:
                raise ValueError("v4 Layer1 did not require alignment integrity")
        else:
            alignment_evidence = None
            checkpoint_migration = None
    else:
        if report.get("role") != "scene_disjoint_one_shot_test":
            raise ValueError("room2 report has the wrong one-shot role")
        if report.get("opened_after_temporal_validation_report_sha256") != (
            preceding_report_sha256
        ):
            raise ValueError("room2 report is not chained after the accepted Layer1 report")
        if report.get("tuning_after_open") is not False:
            raise ValueError("room2 one-shot report permits post-test tuning")
        metrics = _validate_room2_layer(report.get("metrics"))
        if expected_candidate_artifact_sha256 is None:
            raise ValueError("room2 selection is missing the accepted H3 artifact")
        room2_evidence = _validate_room2_evidence(
            layer_report=report,
            layer_report_path=report_path,
            metrics=metrics,
            checkpoint_sha256=checkpoint_sha256,
            evssm_checkpoint_sha256=evssm_checkpoint_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_candidate_artifact_sha256=(
                expected_candidate_artifact_sha256
            ),
            evaluator_schema=h3_evaluator_schema,
        )
        if require_v4_alignment_diagnostics:
            evaluator_entry = report.get("evaluator_report")
            if not isinstance(evaluator_entry, dict):
                raise ValueError("v4 room2 layer is missing evaluator evidence")
            evaluator_path = _resolve_evidence_path(
                evaluator_entry.get("path"), report_path, "v4 room2 evaluator report"
            )
            evaluator_payload = _load_sha_bound_json(
                evaluator_path,
                evaluator_entry.get("sha256"),
                "v4 room2 evaluator report",
            )
            alignment_evidence = _validate_v4_alignment_evidence(
                evaluator_payload,
                label="v4 room2 evaluator",
                expected_transition_count=None,
                require_lpips=True,
            )
            if report.get("alignment_integrity_passed") is not True:
                raise ValueError("v4 Layer2 did not require alignment integrity")
        else:
            alignment_evidence = None
        history1_control = None
    normalized = {
        "layer": layer_name,
        "report_sha256": expected_report_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "metrics": metrics,
        "eligible": True,
    }
    if history1_control is not None:
        normalized["history1_control"] = history1_control
    if alignment_evidence is not None:
        normalized["alignment_evidence"] = alignment_evidence
    if checkpoint_migration is not None:
        normalized["checkpoint_migration"] = checkpoint_migration
    if layer_name == "room2_one_shot":
        normalized["evaluator_evidence"] = room2_evidence
    return normalized


def validate_deployment_selection(
    report_path: Path,
    checkpoint_path: Path,
    teacher_provenance: dict[str, object],
) -> dict[str, object]:
    """Validate preregistered Layer1 temporal-val then Layer2 room2 selection."""
    if not report_path.is_file():
        raise FileNotFoundError(f"deployment selection report does not exist: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("deployment selection report is not valid JSON") from error
    if not isinstance(report, dict) or report.get("schema") != (
        DEPLOYMENT_SELECTION_SCHEMA_V3
    ):
        raise ValueError("deployment selection report has an unsupported schema")
    if report.get("policy") != DEPLOYMENT_SELECTION_POLICY_V1:
        raise ValueError("deployment selection report policy is not preregistered")
    if report.get("thresholds") != DEPLOYMENT_THRESHOLDS:
        raise ValueError("deployment selection thresholds differ from the contract")
    if report.get("oracle_good_definition") != ORACLE_GOOD_DEFINITION:
        raise ValueError("deployment selection oracle-good definition is not preregistered")
    registered_contract = report.get("registered_contract")
    if not isinstance(registered_contract, dict) or registered_contract != {
        "schema": REGISTERED_CONTRACT_SCHEMA,
        "sha256": REGISTERED_CONTRACT_SHA256,
    }:
        raise ValueError("deployment selection is not bound to the registered contract")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("deployment selection report is bound to a different checkpoint")
    expected_teacher_sha = teacher_provenance.get("evssm_checkpoint_sha256")
    if expected_teacher_sha != REGISTERED_EVSSM_SHA256:
        raise ValueError("deployment selection requires the registered Unblur-SLAM EVSSM")
    if report.get("evssm_checkpoint_sha256") != expected_teacher_sha:
        raise ValueError("deployment selection report uses a different EVSSM baseline")
    if report.get("tum_used_for_selection") is not False:
        raise ValueError("TUM may only be used after layered deployment selection")
    layers = report.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("deployment selection report is missing layered reports")
    temporal = _resolve_layer_report(
        selection_report_path=report_path,
        layer_name="temporal_validation",
        layer_entry=layers.get("temporal_validation"),
        expected_manifest_sha256=TEMPORAL_VALIDATION_MANIFEST_SHA256,
        checkpoint_sha256=checkpoint_sha256,
        evssm_checkpoint_sha256=expected_teacher_sha,
    )
    room2 = _resolve_layer_report(
        selection_report_path=report_path,
        layer_name="room2_one_shot",
        layer_entry=layers.get("room2_one_shot"),
        expected_manifest_sha256=ROOM2_ONE_SHOT_MANIFEST_SHA256,
        checkpoint_sha256=checkpoint_sha256,
        evssm_checkpoint_sha256=expected_teacher_sha,
        preceding_report_sha256=str(temporal["report_sha256"]),
        expected_candidate_artifact_sha256=str(
            temporal["history1_control"]["h3_evaluated_artifact_sha256"]
        ),
    )
    if report.get("eligible") is not True:
        raise ValueError("layered deployment selection is not eligible")
    return {
        "schema": DEPLOYMENT_SELECTION_SCHEMA_V3,
        "policy": DEPLOYMENT_SELECTION_POLICY_V1,
        "selection_report_sha256": sha256_file(report_path),
        "oracle_good_definition": dict(ORACLE_GOOD_DEFINITION),
        "registered_contract": dict(registered_contract),
        "checkpoint_sha256": checkpoint_sha256,
        "evssm_checkpoint_sha256": expected_teacher_sha,
        "tum_used_for_selection": False,
        "layers": {
            "temporal_validation": temporal,
            "room2_one_shot": room2,
        },
        "eligible": True,
    }


def validate_v4_deployment_selection(
    report_path: Path,
    checkpoint_path: Path,
    teacher_provenance: dict[str, object],
) -> dict[str, object]:
    """Validate future v4 Layer1/Layer2 evidence, rejecting every v3 report.

    The v4 report and both layer reports have distinct schemas and are bound
    to the immutable alignment preregistration.  Layer1 additionally proves
    complete alignment diagnostics on the pinned 16-frame split before the
    chained, locked room2 report is opened.
    """

    if not report_path.is_file():
        raise FileNotFoundError(
            f"v4 deployment selection report does not exist: {report_path}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("v4 deployment selection report is not valid JSON") from error
    if not isinstance(report, dict) or report.get("schema") != (
        DEPLOYMENT_SELECTION_SCHEMA_V4
    ):
        raise ValueError(
            "v4 deployment requires a v4 layered report; v3 reports are forbidden"
        )
    if report.get("policy") != DEPLOYMENT_SELECTION_POLICY_V4:
        raise ValueError("v4 deployment selection policy is not preregistered")
    if report.get("thresholds") != DEPLOYMENT_THRESHOLDS:
        raise ValueError("v4 deployment thresholds differ from the v3 scientific gates")
    if report.get("oracle_good_definition") != ORACLE_GOOD_DEFINITION:
        raise ValueError("v4 oracle-good definition is not preregistered")
    registered_contract = report.get("registered_contract")
    if not isinstance(registered_contract, dict) or registered_contract != {
        "schema": REGISTERED_V4_CONTRACT_SCHEMA,
        "sha256": REGISTERED_V4_CONTRACT_SHA256,
    }:
        raise ValueError("v4 selection is not bound to the alignment contract")
    if report.get("warm_start_checkpoint_sha256") != (
        REGISTERED_V4_WARM_START_SHA256
    ):
        raise ValueError("v4 selection is not bound to the registered warm start")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError as error:
        raise RuntimeError(
            "v4 deployment validation requires torch.load(weights_only=True)"
        ) from error
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != (
        CHECKPOINT_FORMAT_V4
    ):
        raise ValueError("v4 selection checkpoint is not a v4 training payload")
    checkpoint_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("v4 selection checkpoint is missing model_config")
    validate_v4_contracts(
        checkpoint,
        dict(checkpoint_config),
        checkpoint_sha256=checkpoint_sha256,
    )
    checkpoint_migration = validate_registered_v4_checkpoint_migration(
        checkpoint, checkpoint_sha256
    )
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("v4 selection is bound to a different checkpoint")
    expected_teacher_sha = teacher_provenance.get("evssm_checkpoint_sha256")
    if expected_teacher_sha != REGISTERED_EVSSM_SHA256 or report.get(
        "evssm_checkpoint_sha256"
    ) != expected_teacher_sha:
        raise ValueError("v4 selection must use the official Unblur-SLAM EVSSM")
    if report.get("tum_used_for_selection") is not False:
        raise ValueError("TUM may only be used after v4 layered selection")
    if report.get("room2_opened_only_after_temporal_pass") is not True:
        raise ValueError("v4 report does not prove Layer1-before-room2 ordering")
    if report.get("tuning_after_room2_open") is not False:
        raise ValueError("v4 report permits tuning after opening room2")
    if report.get("alignment_diagnostics_schema") != (
        ALIGNMENT_DIAGNOSTICS_SCHEMA_V4
    ):
        raise ValueError("v4 report does not require alignment diagnostics")
    layers = report.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("v4 deployment selection is missing layered reports")
    temporal = _resolve_layer_report(
        selection_report_path=report_path,
        layer_name="temporal_validation",
        layer_entry=layers.get("temporal_validation"),
        expected_manifest_sha256=TEMPORAL_VALIDATION_MANIFEST_SHA256,
        checkpoint_sha256=checkpoint_sha256,
        evssm_checkpoint_sha256=str(expected_teacher_sha),
        layer_report_schema=DEPLOYMENT_LAYER_REPORT_SCHEMA_V4,
        registered_contract_sha256=REGISTERED_V4_CONTRACT_SHA256,
        h3_evaluator_schema=EVALUATOR_SCHEMA_V4,
        require_v4_alignment_diagnostics=True,
        expected_v4_checkpoint_migration=checkpoint_migration,
    )
    room2 = _resolve_layer_report(
        selection_report_path=report_path,
        layer_name="room2_one_shot",
        layer_entry=layers.get("room2_one_shot"),
        expected_manifest_sha256=ROOM2_ONE_SHOT_MANIFEST_SHA256,
        checkpoint_sha256=checkpoint_sha256,
        evssm_checkpoint_sha256=str(expected_teacher_sha),
        preceding_report_sha256=str(temporal["report_sha256"]),
        expected_candidate_artifact_sha256=str(
            temporal["history1_control"]["h3_evaluated_artifact_sha256"]
        ),
        layer_report_schema=DEPLOYMENT_LAYER_REPORT_SCHEMA_V4,
        registered_contract_sha256=REGISTERED_V4_CONTRACT_SHA256,
        h3_evaluator_schema=EVALUATOR_SCHEMA_V4,
        require_v4_alignment_diagnostics=True,
    )
    if report.get("eligible") is not True:
        raise ValueError("v4 layered deployment selection is not eligible")
    return {
        "schema": DEPLOYMENT_SELECTION_SCHEMA_V4,
        "policy": DEPLOYMENT_SELECTION_POLICY_V4,
        "selection_report_sha256": sha256_file(report_path),
        "oracle_good_definition": dict(ORACLE_GOOD_DEFINITION),
        "registered_contract": dict(registered_contract),
        "warm_start_checkpoint_sha256": REGISTERED_V4_WARM_START_SHA256,
        "checkpoint_sha256": checkpoint_sha256,
        "evssm_checkpoint_sha256": expected_teacher_sha,
        "tum_used_for_selection": False,
        "room2_opened_only_after_temporal_pass": True,
        "tuning_after_room2_open": False,
        "alignment_diagnostics_schema": ALIGNMENT_DIAGNOSTICS_SCHEMA_V4,
        "checkpoint_migration": checkpoint_migration,
        "layers": {
            "temporal_validation": temporal,
            "room2_one_shot": room2,
        },
        "eligible": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection-report",
        type=Path,
        help=(
            "required for v3/v4 deployment: preregistered temporal-validation "
            "Layer1 plus room2 one-shot Layer2 reports bound to the exact "
            "checkpoint, generation-specific contract, and EVSSM SHA-256"
        ),
    )
    parser.add_argument(
        "--diagnostic-output",
        action="store_true",
        help=(
            "export a v3/v4 TorchScript solely for evaluator input without a "
            "selection report; metadata marks it deployment_eligible=false"
        ),
    )
    parser.add_argument("--verify-height", type=int, default=32)
    parser.add_argument("--verify-width", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError as error:
        raise RuntimeError(
            "checkpoint export requires a PyTorch version with "
            "torch.load(weights_only=True) support"
        ) from error
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            "expected a causal-video checkpoint with model and model_config keys; "
            "a single-frame EVSSM params checkpoint cannot be exported directly"
        )
    checkpoint_format = str(checkpoint.get("format", CHECKPOINT_FORMAT_V1))
    if checkpoint_format not in {
        CHECKPOINT_FORMAT_V1,
        CHECKPOINT_FORMAT_V3,
        CHECKPOINT_FORMAT_V4,
    }:
        raise ValueError(f"unsupported causal-video checkpoint format {checkpoint_format!r}")
    config = dict(checkpoint.get("model_config", {}))
    if bool(config.get("use_teacher_input", False)):
        raise ValueError(
            "teacher-input checkpoints cannot be exported to the one-input "
            "streaming runtime; train with EVSSM distillation and "
            "use_teacher_input=false"
        )
    teacher_provenance = validate_teacher_provenance(
        checkpoint.get("teacher_provenance"),
        input_domain=str(config.get("input_domain", "raw")),
    )
    training_contract = checkpoint.get("training_contract")
    if not isinstance(training_contract, dict) or training_contract.get(
        "stream_prefix_padding"
    ) != "repeat_first_frame_on_left":
        raise ValueError(
            "checkpoint does not prove the streaming prefix-padding training contract"
        )
    if checkpoint_format in {CHECKPOINT_FORMAT_V3, CHECKPOINT_FORMAT_V4}:
        if checkpoint_format == CHECKPOINT_FORMAT_V3:
            validate_v3_contracts(checkpoint, config)
        else:
            validate_v4_contracts(
                checkpoint,
                config,
                checkpoint_sha256=checkpoint_sha256,
            )
        if args.selection_report is not None and args.diagnostic_output:
            raise ValueError(
                "--diagnostic-output and --selection-report are mutually exclusive"
            )
        if args.selection_report is None and not args.diagnostic_output:
            raise ValueError(
                f"{checkpoint_format.rsplit('.', 1)[-1]} checkpoints are "
                "diagnostic training artifacts; "
                "use --diagnostic-output for evaluator input or provide "
                "--selection-report for deployment export"
            )
        if args.selection_report is not None:
            if checkpoint_format == CHECKPOINT_FORMAT_V4:
                if type(checkpoint.get("step")) is not int or checkpoint.get(
                    "step"
                ) != 600:
                    raise ValueError(
                        "formal v4 export requires the completed 600-step checkpoint"
                    )
                deployment_selection = validate_v4_deployment_selection(
                    args.selection_report.expanduser().resolve(),
                    checkpoint_path,
                    teacher_provenance,
                )
            else:
                deployment_selection = validate_deployment_selection(
                    args.selection_report.expanduser().resolve(),
                    checkpoint_path,
                    teacher_provenance,
                )
            artifact_role = "deployment_selected"
            deployment_eligible = True
        else:
            deployment_selection = None
            artifact_role = "diagnostic_evaluation_only"
            deployment_eligible = False
    else:
        if args.diagnostic_output:
            raise ValueError(
                "--diagnostic-output is only valid for v3/v4 checkpoints"
            )
        deployment_selection = None
        artifact_role = "legacy_v1"
        deployment_eligible = None
    model = build_causal_video_deblur(config).eval()
    model.load_state_dict(checkpoint["model"], strict=True)

    scripted = torch.jit.script(model)
    preserved_attrs = ["forward_sequence"]
    if checkpoint_format == CHECKPOINT_FORMAT_V4:
        preserved_attrs.extend(
            [
                "forward_sequence_with_motion_diagnostics",
                "forward_sequence_alignment_disabled",
            ]
        )
    scripted = torch.jit.freeze(scripted, preserved_attrs=preserved_attrs)
    history = int(config.get("max_history", 5))
    torch.manual_seed(0)
    example = torch.rand(1, history, 3, args.verify_height, args.verify_width)
    with torch.no_grad():
        eager_output = model(example)
        scripted_output = scripted(example)
        eager_sequence = model.forward_sequence(example)
        scripted_sequence = scripted.forward_sequence(example)
    max_error = float((eager_output - scripted_output).abs().max().item())
    sequence_max_error = float(
        (eager_sequence - scripted_sequence).abs().max().item()
    )
    if max(max_error, sequence_max_error) > 1.0e-5:
        raise RuntimeError(
            "TorchScript verification failed: "
            f"forward={max_error}, forward_sequence={sequence_max_error}"
        )
    if not torch.equal(scripted_output, scripted_sequence[:, -1]):
        raise RuntimeError("frozen forward is not forward_sequence's newest frame")
    motion_diagnostics_max_error = None
    alignment_disabled_max_error = None
    if checkpoint_format == CHECKPOINT_FORMAT_V4:
        with torch.no_grad():
            eager_motion = model.forward_sequence_with_motion_diagnostics(example)
            scripted_motion = scripted.forward_sequence_with_motion_diagnostics(
                example
            )
            eager_disabled = model.forward_sequence_alignment_disabled(example)
            scripted_disabled = scripted.forward_sequence_alignment_disabled(
                example
            )
        if len(eager_motion) != 4 or len(scripted_motion) != 4:
            raise RuntimeError("v4 motion diagnostics must return four tensors")
        motion_errors = [
            float((eager - compiled).abs().max().item())
            for eager, compiled in zip(eager_motion, scripted_motion)
        ]
        motion_diagnostics_max_error = max(motion_errors)
        alignment_disabled_max_error = float(
            (eager_disabled - scripted_disabled).abs().max().item()
        )
        if max(motion_diagnostics_max_error, alignment_disabled_max_error) > 1.0e-5:
            raise RuntimeError(
                "v4 diagnostic TorchScript verification failed: "
                f"motion={motion_diagnostics_max_error}, "
                f"alignment_disabled={alignment_disabled_max_error}"
            )
        if not torch.equal(scripted_motion[0], scripted_sequence):
            raise RuntimeError(
                "v4 diagnostics prediction disagrees with forward_sequence"
            )
        if scripted_motion[1].shape[:3] != (1, history - 1, 2) or (
            scripted_motion[2].shape[:3] != (1, history - 1, 1)
        ) or scripted_motion[3].shape != scripted_motion[2].shape:
            raise RuntimeError("v4 diagnostic tensors violate the adjacent-flow API")
    max_observed_residual = float(
        (scripted_sequence - example).abs().max().item()
    )
    if checkpoint_format in {CHECKPOINT_FORMAT_V3, CHECKPOINT_FORMAT_V4}:
        max_residual = float(config["max_residual"])
        if max_observed_residual > max_residual + 1.0e-6:
            raise RuntimeError(
                "bounded-residual verification failed: "
                f"observed={max_observed_residual}, configured={max_residual}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": (
            TORCHSCRIPT_FORMAT_V4
            if checkpoint_format == CHECKPOINT_FORMAT_V4
            else (
                TORCHSCRIPT_FORMAT_V3
                if checkpoint_format == CHECKPOINT_FORMAT_V3
                else TORCHSCRIPT_FORMAT_V1
            )
        ),
        "checkpoint_format": checkpoint_format,
        "source_checkpoint_sha256": checkpoint_sha256,
        "artifact_role": artifact_role,
        "deployment_eligible": deployment_eligible,
        "input": "BTCHW float RGB [0,1]",
        "output": "BCHW newest frame",
        "model_config": config,
        "teacher_provenance": teacher_provenance,
        "training_contract": dict(training_contract),
        "objective_contract": checkpoint.get("objective_contract"),
        "optimization_contract": checkpoint.get("optimization_contract"),
        "refinement_contract": checkpoint.get("refinement_contract"),
        "checkpoint_selection": checkpoint.get("checkpoint_selection"),
        "deployment_selection": deployment_selection,
        "validation_metrics": json_safe_validation_metrics(
            checkpoint.get("validation_metrics")
        ),
        "export_verification": {
            "height": int(args.verify_height),
            "width": int(args.verify_width),
            "eager_script_max_abs_error": max_error,
            "forward_sequence_max_abs_error": sequence_max_error,
            "max_observed_abs_residual": max_observed_residual,
        },
        "exported_methods": ["forward", *preserved_attrs],
    }
    if checkpoint_format == CHECKPOINT_FORMAT_V4:
        metadata["warm_start_provenance"] = checkpoint.get(
            "warm_start_provenance"
        )
        metadata["registered_contract"] = checkpoint.get("registered_contract")
        metadata["data_identity"] = checkpoint.get("data_identity")
        metadata["source_checkpoint_epoch"] = checkpoint.get("epoch")
        metadata["source_checkpoint_step"] = checkpoint.get("step")
        metadata["training_phase"] = checkpoint.get("training_phase")
        rng_state = checkpoint.get("rng_state")
        metadata["rng_state_provenance"] = {
            "schema": (
                rng_state.get("schema") if isinstance(rng_state, dict) else None
            ),
            "checkpoint_boundary": (
                rng_state.get("checkpoint_boundary")
                if isinstance(rng_state, dict)
                else None
            ),
            "captured": isinstance(rng_state, dict),
        }
        if checkpoint.get("checkpoint_migration") is not None:
            metadata["checkpoint_migration"] = checkpoint.get(
                "checkpoint_migration"
            )
        metadata["export_verification"]["motion_diagnostics_max_abs_error"] = (
            motion_diagnostics_max_error
        )
        metadata["export_verification"]["alignment_disabled_max_abs_error"] = (
            alignment_disabled_max_error
        )
    extra_files = {
        "metadata.json": json.dumps(metadata, sort_keys=True, allow_nan=False)
    }
    torch.jit.save(scripted, str(args.output), _extra_files=extra_files)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "artifact_role": artifact_role,
                "deployment_eligible": deployment_eligible,
                "bytes": args.output.stat().st_size,
                "max_abs_error": max_error,
                "forward_sequence_max_abs_error": sequence_max_error,
                "max_observed_abs_residual": max_observed_residual,
                "model_config": config,
                "teacher_provenance": teacher_provenance,
                "objective_contract": checkpoint.get("objective_contract"),
                "refinement_contract": checkpoint.get("refinement_contract"),
            }
        )
    )


if __name__ == "__main__":
    main()
