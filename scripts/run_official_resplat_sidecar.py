#!/usr/bin/env python3
"""Run official cvg/ReSplat init+N on one immutable closed-submap snapshot.

This entry point is intentionally launched with the official ReSplat Python
environment as a fresh process.  It writes a native, local-gauge Gaussian NPZ
and a bound manifest atomically.  It never imports or mutates Unblur-SLAM's
active ``GaussianModel``.  It additionally exports a coordinate-audited
snapshot-world payload in Unblur parameter layout; export is not map mutation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence


SNAPSHOT_SCHEMA = "unblur_slam.official_resplat_closed_submap_snapshot.v1"
RESULT_SCHEMA = "unblur_slam.official_resplat_closed_submap_result.v1"
OFFICIAL_PRESET = "dl3dv_8v_256x448_small"
NUM_CONTEXT = 8
NUM_REFINE = 4  # backward-compatible default; state3 passes --num-refine 3
MIN_REFINE = 1
MAX_REFINE = 4
LATENT_DOWNSAMPLE = 4


def _load_unblur_bridge() -> Any:
    """Load the bridge without importing Unblur's regular ``src`` package."""

    import importlib.util

    path = Path(__file__).resolve().parents[1] / "src/refinement/resplat_unblur_bridge.py"
    specification = importlib.util.spec_from_file_location(
        "_unblur_resplat_world_bridge", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load Unblur/ReSplat bridge: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_snapshot_data_provenance(payload: Mapping[str, Any]) -> None:
    legacy_pose = bool(payload.get("uses_ground_truth_pose", True))
    if legacy_pose or bool(payload.get("uses_ground_truth_pose_or_depth", legacy_pose)):
        raise ValueError("ground-truth pose/depth is forbidden in online sidecars")
    if bool(payload.get("uses_independent_clear_pixels", False)):
        raise ValueError("independent clear pixels are forbidden in online sidecars")
    if bool(payload.get("uses_clear_gt_metrics", False)):
        raise ValueError("clear-GT metrics are forbidden in online sidecars")
    membership = bool(payload.get("uses_clear_gt_membership", True))
    if payload.get(
        "selection_membership_clear_gt_conditioned", membership
    ) is not membership:
        raise ValueError("clear-GT-conditioned membership disclosure mismatch")
    if not membership:
        return
    if payload.get("integration_mode") != "online_mapper":
        raise ValueError("conditioned membership is allowed only for online_mapper")
    provenance = payload.get("source_provenance") or {}
    expected = {
        "selection_membership_clear_gt_conditioned": True,
        "uses_ground_truth_pose_or_depth": False,
        "uses_independent_clear_pixels": False,
        "uses_clear_gt_metrics": False,
    }
    for key, value in expected.items():
        if provenance.get(key) is not value:
            raise ValueError(f"conditioned-membership provenance {key} drifted")


def _load_snapshot(root: Path) -> dict[str, Any]:
    path = root / "snapshot_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid snapshot manifest: {path}") from error
    if payload.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("wrong closed-submap snapshot schema")
    if not bool(payload.get("closed")):
        raise ValueError("official sidecar requires a closed submap")
    allowed_modes = {
        "online_mapper": "online_mapper_closed_submap_membership",
        "independent_queue_smoke": "droid_motion_filter_first_closed_8kf_prefix",
    }
    mode = payload.get("integration_mode")
    if mode not in allowed_modes or payload.get("selection_source") != allowed_modes[mode]:
        raise ValueError("snapshot integration mode/selection source mismatch")
    if mode == "independent_queue_smoke" and not payload.get("source_provenance"):
        raise ValueError("independent queue snapshot lacks bound source provenance")
    _validate_snapshot_data_provenance(payload)
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != NUM_CONTEXT:
        raise ValueError("official small8v sidecar requires exactly 8 frames")
    unsigned = dict(payload)
    expected = str(unsigned.pop("snapshot_sha256", ""))
    unsigned.pop("snapshot_id", None)
    actual = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if expected != actual:
        raise ValueError("snapshot manifest hash mismatch")
    closure = int(payload["closure_sequence_ordinal"])
    ordinals = [int(frame["sequence_ordinal"]) for frame in frames]
    if ordinals != sorted(ordinals) or len(set(ordinals)) != NUM_CONTEXT:
        raise ValueError("snapshot order is not a unique causal prefix")
    if any(value > closure for value in ordinals):
        raise ValueError("future frame entered the closed-submap snapshot")
    for frame in frames:
        relative = Path(str(frame["image_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("snapshot image path escapes its root")
        image = root / relative
        if _sha256_file(image) != frame.get("image_sha256"):
            raise ValueError("snapshot image hash mismatch")
    return payload


def _tensor_versions(gaussians: object) -> dict[str, int]:
    result: dict[str, int] = {}
    for name in (
        "means",
        "covariances",
        "harmonics",
        "opacities",
        "scales",
        "rotations",
        "rotations_unnorm",
    ):
        value = getattr(gaussians, name, None)
        if value is not None:
            result[name] = int(value._version)
    return result


def run_official_refinement_states_core(
    *,
    encoder: object,
    decoder: object,
    batch: Mapping[str, Any],
    stage_runner: Any,
    num_refine: int,
) -> tuple[object, object, dict[str, Any]]:
    """Execute one initializer and exactly ``num_refine`` recurrent states."""

    if isinstance(num_refine, bool) or not MIN_REFINE <= int(num_refine) <= MAX_REFINE:
        raise ValueError(f"num_refine must be in [{MIN_REFINE},{MAX_REFINE}]")
    num_refine = int(num_refine)

    initial_output = stage_runner(
        "encoder_init0",
        lambda: encoder(
            batch["context"],
            global_step=0,
            deterministic=False,
            visualization_dump=None,
        ),
    )
    if not isinstance(initial_output, dict):
        raise RuntimeError("official initializer returned no condition features")
    initial = initial_output.get("gaussians")
    condition_features = initial_output.get("condition_features")
    if initial is None or condition_features is None:
        raise RuntimeError("official initializer result is incomplete")
    versions = _tensor_versions(initial)
    update_output = stage_runner(
        f"forward_update_refine{num_refine}",
        lambda: encoder.forward_update(
            batch["context"],
            batch["target"],
            condition_features,
            initial,
            decoder,
            None,
        ),
    )
    if _tensor_versions(initial) != versions:
        raise RuntimeError("official forward_update mutated init state in place")
    sequence = update_output.get("gaussian")
    if not isinstance(sequence, (list, tuple)) or len(sequence) != num_refine:
        raise RuntimeError(
            "official forward_update state count mismatch: expected "
            f"{num_refine}, got {len(sequence) if isinstance(sequence, (list, tuple)) else 'non-sequence'}"
        )
    return initial, sequence[-1], {
        "encoder_forward_calls": 1,
        "forward_update_calls": 1,
        "init_object_passed_directly": True,
        "initial_state_mutated_in_place": False,
        "target_api_slot": "same_closed_context_views",
        "independent_process": True,
        "requested_recurrent_updates": num_refine,
        "returned_recurrent_states": len(sequence),
        "selected_state_index_zero_based": num_refine - 1,
        "fourth_state_computed": num_refine >= 4,
    }


def run_official_refinement_core(
    *,
    encoder: object,
    decoder: object,
    batch: Mapping[str, Any],
    stage_runner: Any,
    num_refine: int = NUM_REFINE,
) -> tuple[object, dict[str, Any]]:
    """Backward-compatible wrapper returning only the selected final state."""

    _, refined, contract = run_official_refinement_states_core(
        encoder=encoder,
        decoder=decoder,
        batch=batch,
        stage_runner=stage_runner,
        num_refine=num_refine,
    )
    return refined, contract


def _render_context_views(
    torch_module: Any,
    decoder: object,
    gaussians: object,
    batch: Mapping[str, Any],
    *,
    chunk_size: int = 4,
) -> tuple[Any, Any]:
    context = batch["context"]
    _, count, _, height, width = context["image"].shape
    colors = []
    alphas = []
    for start in range(0, count, chunk_size):
        end = min(count, start + chunk_size)
        rendered = decoder.forward(
            gaussians,
            context["extrinsics"][:, start:end],
            context["intrinsics"][:, start:end],
            context["near"][:, start:end],
            context["far"][:, start:end],
            (height, width),
            depth_mode=None,
        )
        colors.append(rendered.color[0])
        alphas.append(rendered.accumulated_alpha[0])
    return torch_module.cat(colors, dim=0), torch_module.cat(alphas, dim=0)


def _context_reconstruction_metrics(
    *,
    torch_module: Any,
    rendered: Any,
    alphas: Any,
    observations: Any,
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure only the eight causal inputs; no clear/evaluation image exists."""

    from src.evaluation.metrics import compute_psnr, compute_ssim

    prediction = rendered.clamp(0.0, 1.0)
    reference = observations.to(prediction.device).clamp(0.0, 1.0)
    psnr = compute_psnr(reference, prediction)
    ssim = compute_ssim(reference, prediction)
    per_pixel_l1 = (reference - prediction).abs().mean(dim=1)
    l1 = per_pixel_l1.mean(dim=(1, 2))
    mask = alphas > 0.01
    masked_l1 = []
    mask_fraction = []
    for index in range(int(prediction.shape[0])):
        selected = mask[index]
        mask_fraction.append(float(selected.float().mean().item()))
        if bool(selected.any().item()):
            masked_l1.append(float(per_pixel_l1[index][selected].mean().item()))
        else:
            masked_l1.append(float(l1[index].item()))
    per_view = []
    for index, frame in enumerate(frames):
        per_view.append(
            {
                "frame_id": frame["frame_id"],
                "sequence_ordinal": int(frame["sequence_ordinal"]),
                "psnr_db": float(psnr[index].item()),
                "ssim": float(ssim[index].item()),
                "l1": float(l1[index].item()),
                "masked_l1": masked_l1[index],
                "alpha_mask_fraction": mask_fraction[index],
            }
        )
    return {
        "mean_psnr_db": float(psnr.mean().item()),
        "mean_ssim": float(ssim.mean().item()),
        "mean_l1": float(l1.mean().item()),
        "mean_masked_l1": float(sum(masked_l1) / len(masked_l1)),
        "per_view": per_view,
    }


def _to_numpy(torch_module: Any, tensor: Any) -> "Any":
    del torch_module
    return tensor.detach().float().cpu().contiguous().numpy()


def _geometry_summary(np: Any, arrays: Mapping[str, Any]) -> dict[str, Any]:
    means = arrays["means"]
    scales = arrays["scales"]
    rotations = arrays["rotations"]
    all_arrays = tuple(arrays.values())
    total = sum(int(value.size) for value in all_arrays)
    finite = sum(int(np.isfinite(value).sum()) for value in all_arrays)
    distances = np.linalg.norm(means, axis=-1)
    quaternion_norms = np.linalg.norm(rotations, axis=-1)
    return {
        "coordinate_frame": "middle_context_camera_local_opencv",
        "gaussian_count": int(means.shape[0]),
        "finite_fraction": float(finite / total) if total else 0.0,
        "median_distance_from_local_origin": float(np.median(distances)),
        "p95_distance_from_local_origin": float(np.quantile(distances, 0.95)),
        "max_distance_from_local_origin": float(np.max(distances)),
        "p95_scale": float(np.quantile(scales, 0.95)),
        "max_scale": float(np.max(scales)),
        "max_quaternion_norm_deviation": float(
            np.max(np.abs(quaternion_norms - 1.0))
        ),
    }


def run(args: argparse.Namespace) -> Path:
    start_wall = time.monotonic()
    snapshot_root = args.snapshot_dir.expanduser().resolve()
    snapshot = _load_snapshot(snapshot_root)
    destination = args.output_dir.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite sidecar output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=str(destination.parent))
    )
    installed = False
    try:
        # Local helper imports must happen before import_official_infer rewrites
        # sys.path to expose cvg/ReSplat's PEP-420 ``src`` namespace.
        bridge = _load_unblur_bridge()
        from run_paired_official_resplat_smoke import (
            CudaStageRecorder,
            _configure_reproducibility,
            _resolve_checkpoint,
            import_official_infer,
            inspect_official_repo,
        )

        import numpy as np
        import torch

        if args.model_preset != OFFICIAL_PRESET:
            raise ValueError(f"sidecar is pinned to {OFFICIAL_PRESET}")
        if not torch.cuda.is_available():
            raise RuntimeError("official ReSplat sidecar requires CUDA")
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("--device must select CUDA")

        repo_record = inspect_official_repo(args.resplat_repo)
        official = import_official_infer(args.resplat_repo)
        checkpoint, checkpoint_sha, preset = _resolve_checkpoint(
            repo=Path(repo_record["path"]),
            official=official,
            preset_name=args.model_preset,
            checkpoint=args.checkpoint,
            expected_sha256=args.expected_checkpoint_sha256,
        )
        if int(preset["num_context"]) != NUM_CONTEXT:
            raise ValueError("official preset context count is not 8")
        preset_default_refine = int(preset["num_refine"])
        if preset_default_refine != NUM_REFINE:
            raise ValueError("official small8v preset default refinement count drifted")
        if not MIN_REFINE <= int(args.num_refine) <= MAX_REFINE:
            raise ValueError(
                f"--num-refine must be in [{MIN_REFINE},{MAX_REFINE}]"
            )

        reproducibility = _configure_reproducibility(torch, np, args.seed)
        frames = snapshot["frames"]
        image_paths = [str(snapshot_root / frame["image_path"]) for frame in frames]
        first_width, first_height = [int(value) for value in frames[0]["image_size_wh"]]
        target_height, target_width = official.compute_target_shape(
            first_height,
            first_width,
            int(preset["max_resolution"]),
            None,
        )
        context_images = official.load_and_preprocess_images(
            image_paths, target_height, target_width
        )
        context_c2w = torch.tensor(
            [frame["c2w_opencv"] for frame in frames], dtype=torch.float32
        )
        context_intrinsics = torch.tensor(
            [frame["intrinsics_normalized"] for frame in frames], dtype=torch.float32
        )

        # forward_update currently derives recurrent residuals only from its
        # context argument.  Duplicate the same closed, past-only eight views
        # into the target-shaped API slot; no future/evaluation image exists.
        batch = official.build_batch(
            context_images,
            context_images,
            context_c2w,
            context_c2w,
            context_intrinsics,
            context_intrinsics,
            args.near,
            args.far,
            str(snapshot["snapshot_id"]),
            str(device),
        )
        encoder, decoder, data_shim = official.build_model(
            experiment=args.experiment,
            checkpoint=str(checkpoint),
            num_refine=int(args.num_refine),
            image_shape=(target_height, target_width),
            overrides=list(preset["overrides"]) + list(args.overrides),
            device=str(device),
            no_strict_load=False,
        )
        batch = data_shim(batch)
        recorder = CudaStageRecorder(torch, device)
        with torch.no_grad():
            initial, refined, execution_contract = run_official_refinement_states_core(
                encoder=encoder,
                decoder=decoder,
                batch=batch,
                stage_runner=recorder,
                num_refine=int(args.num_refine),
            )
            initial_context_render, initial_context_alpha = recorder(
                "render_context_init0",
                lambda: _render_context_views(torch, decoder, initial, batch),
            )
            refined_context_render, refined_context_alpha = recorder(
                f"render_context_state{int(args.num_refine)}",
                lambda: _render_context_views(torch, decoder, refined, batch),
            )
            initial_reconstruction = _context_reconstruction_metrics(
                torch_module=torch,
                rendered=initial_context_render,
                alphas=initial_context_alpha,
                observations=batch["context"]["image"][0],
                frames=frames,
            )
            refined_reconstruction = _context_reconstruction_metrics(
                torch_module=torch,
                rendered=refined_context_render,
                alphas=refined_context_alpha,
                observations=batch["context"]["image"][0],
                frames=frames,
            )

        arrays = {
            "means": _to_numpy(torch, refined.means[0]),
            # Preserve the exact official selected-state covariance as the
            # authoritative conversion source.  It may be numerically PSD
            # rather than strictly SPD in float32; the world bridge records
            # and repairs only roundoff-sized negative eigenvalues.
            "covariances": _to_numpy(torch, refined.covariances[0]),
            "scales": _to_numpy(torch, refined.scales[0]),
            "rotations": _to_numpy(torch, refined.rotations[0]),
            "harmonics": _to_numpy(torch, refined.harmonics[0]),
            "opacities": _to_numpy(torch, refined.opacities[0]),
        }
        covariance_local = arrays["covariances"]
        expected_gaussian_count = (
            NUM_CONTEXT
            * (target_height // LATENT_DOWNSAMPLE)
            * (target_width // LATENT_DOWNSAMPLE)
        )
        if int(arrays["means"].shape[0]) != expected_gaussian_count:
            raise RuntimeError(
                "official small8v fixed topology mismatch: expected "
                f"{expected_gaussian_count}, got {arrays['means'].shape[0]}"
            )
        geometry = _geometry_summary(np, arrays)
        gaussian_path = staging / "native_gaussians_local.npz"
        np.savez_compressed(gaussian_path, **arrays)
        middle = frames[NUM_CONTEXT // 2]
        world_arrays, world_contract = bridge.build_unblur_world_arrays(
            means_local=arrays["means"],
            covariances_local=covariance_local,
            harmonics_local=arrays["harmonics"],
            opacities=arrays["opacities"],
            pivot_c2w=middle["c2w_opencv"],
            owner_frame_ids=[int(frame["frame_id"]) for frame in frames],
            owner_sequence_ordinals=[int(frame["sequence_ordinal"]) for frame in frames],
        )
        world_contract.update(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "source_pose_revision": int(snapshot["pose_revision"]),
                "source_pose_hashes": [frame["pose_hash"] for frame in frames],
                "pivot_context_index_zero_based": NUM_CONTEXT // 2,
                "pivot_frame_id": middle["frame_id"],
                "pivot_sequence_ordinal": int(middle["sequence_ordinal"]),
                "pivot_pose_hash": middle["pose_hash"],
                "owner_frame_ids": [int(frame["frame_id"]) for frame in frames],
                "owner_sequence_ordinals": [
                    int(frame["sequence_ordinal"]) for frame in frames
                ],
                "refinement_state": int(args.num_refine),
            }
        )
        world_path = staging / "unblur_gaussians_snapshot_world.npz"
        np.savez_compressed(world_path, **world_arrays)
        stage_records = recorder.records
        metric_delta = {
            "mean_psnr_db": refined_reconstruction["mean_psnr_db"]
            - initial_reconstruction["mean_psnr_db"],
            "mean_ssim": refined_reconstruction["mean_ssim"]
            - initial_reconstruction["mean_ssim"],
            "mean_l1": refined_reconstruction["mean_l1"]
            - initial_reconstruction["mean_l1"],
            "mean_masked_l1": refined_reconstruction["mean_masked_l1"]
            - initial_reconstruction["mean_masked_l1"],
        }
        manifest = {
            "schema": RESULT_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_class": "native_official_resplat_closed_submap_sidecar",
            "integration_mode": snapshot["integration_mode"],
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "submap_id": snapshot["submap_id"],
            "source_pose_revision": snapshot["pose_revision"],
            "source_pose_hashes": [frame["pose_hash"] for frame in frames],
            "past_only": True,
            "future_views_used": False,
            "ground_truth_used": False,
            "selection_membership_clear_gt_conditioned": bool(
                snapshot.get("uses_clear_gt_membership", False)
            ),
            "ground_truth_pose_or_depth_used": False,
            "independent_clear_pixels_used": False,
            "clear_gt_metrics_used": False,
            "official_resplat": {
                "repository": repo_record,
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha,
                },
                "model_preset": args.model_preset,
                "num_context": NUM_CONTEXT,
                "num_refine": int(args.num_refine),
                "preset_default_num_refine": preset_default_refine,
                "runtime_refinement_override": int(args.num_refine)
                != preset_default_refine,
            },
            "execution_contract": execution_contract,
            "image_shape_hw": [target_height, target_width],
            "fixed_topology": {
                "latent_downsample": LATENT_DOWNSAMPLE,
                "expected_gaussian_count": expected_gaussian_count,
                "arbitrary_n_input_supported": False,
            },
            "near": args.near,
            "far": args.far,
            "local_coordinate_contract": {
                "frame": "middle_context_camera_local_opencv",
                "middle_context_frame_id": middle["frame_id"],
                "local_to_global_c2w_opencv": middle["c2w_opencv"],
                "safe_for_active_unblur_map_merge": False,
            },
            "geometry": geometry,
            "context_reconstruction": {
                "uses_clear_gt": False,
                "inputs": "eight_context_observations",
                "same_observations_for_init0_and_state3": int(args.num_refine) == 3,
                "alpha_mask": "official_decoder_accumulated_alpha>0.01",
                "init0": initial_reconstruction,
                f"state{int(args.num_refine)}": refined_reconstruction,
                f"state{int(args.num_refine)}_minus_init0": metric_delta,
            },
            "cuda_stages": stage_records,
            "wall_runtime_seconds": float(time.monotonic() - start_wall),
            "reproducibility": reproducibility,
            "outputs": {
                "native_gaussians_npz": gaussian_path.name,
                "native_gaussians_npz_sha256": _sha256_file(gaussian_path),
                "npz_arrays": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in arrays.items()
                },
                "unblur_world_gaussians_npz": world_path.name,
                "unblur_world_gaussians_npz_sha256": _sha256_file(world_path),
                "unblur_world_npz_arrays": bridge.array_manifest(world_arrays),
                "root_created_atomically": True,
            },
            "unblur_world_artifact": world_contract,
            "active_map_merge_performed": False,
            "native_to_unblur_conversion_performed": True,
            "merge_compatibility": "coordinate_converted_candidate_requires_mapper_gates",
        }
        (staging / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(staging, destination)
        installed = True
        return destination
    finally:
        if not installed and staging.exists():
            shutil.rmtree(staging)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resplat-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--model-preset", default=OFFICIAL_PRESET)
    parser.add_argument("--experiment", default="dl3dv")
    parser.add_argument(
        "--num-refine",
        type=int,
        default=NUM_REFINE,
        choices=range(MIN_REFINE, MAX_REFINE + 1),
        help=(
            "Number of actual recurrent updates. State3 uses 3; the official "
            "small8v preset default is 4. No unselected fourth state is computed."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        output = run(parse_args(argv))
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2
    print(f"official ReSplat sidecar saved atomically to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
