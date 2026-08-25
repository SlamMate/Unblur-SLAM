# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from thirdparty.glorie_slam import config
from src.slam import SLAM
from src.utils.datasets import get_dataset
from src.utils.eval_frames import (
    clear_gt_scope_mode,
    validate_clear_gt_protocol_scope,
)
from src.refinement.resplat_replay import validate_resplat_config
from time import gmtime, strftime
from colorama import Fore,Style

import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _existing_file(value, label):
    if not value:
        raise ValueError(f"{label} must be configured")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_TEACHER_PROVENANCE_SCHEMA = "unblur_slam.video_deblur_teacher_provenance.v1"


def _sha256_digest(value, label):
    value = str(value).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _validate_causal_teacher_provenance(metadata, frontend, cfg):
    """Bind an exported causal adapter to its actual training input domain."""

    provenance = metadata.get("teacher_provenance")
    if not isinstance(provenance, dict) or provenance.get("schema") != _TEACHER_PROVENANCE_SCHEMA:
        raise ValueError(
            "causal deblur export is missing validated teacher_provenance"
        )
    provenance = dict(provenance)
    storage = str(provenance.get("storage", ""))
    teacher_domain = str(provenance.get("teacher_domain", ""))
    input_domain = str(metadata.get("model_config", {}).get("input_domain", "raw")).lower()

    if storage == "none":
        if teacher_domain != "none" or provenance.get("evssm_checkpoint_sha256") is not None:
            raise ValueError("causal teacher storage=none has inconsistent provenance")
        if input_domain == "evssm" or frontend == "causal_evssm":
            raise ValueError("causal_evssm cannot use teacher storage=none")
    elif storage in {"runtime_evssm_float_tensor", "precomputed_png_rgb8"}:
        if teacher_domain != "evssm_restored_rgb_0_1":
            raise ValueError(
                f"unsupported causal teacher domain {teacher_domain!r}"
            )
        teacher_sha = _sha256_digest(
            provenance.get("evssm_checkpoint_sha256"),
            "teacher_provenance.evssm_checkpoint_sha256",
        )
        provenance["evssm_checkpoint_sha256"] = teacher_sha
        if storage == "precomputed_png_rgb8":
            for key in ("precompute_report_sha256", "teacher_manifest_sha256"):
                provenance[key] = _sha256_digest(
                    provenance.get(key), f"teacher_provenance.{key}"
                )
            if provenance.get("teacher_artifacts_verified") is not True:
                raise ValueError(
                    "cached causal teacher artifacts were not verified before training"
                )
    else:
        raise ValueError(f"unsupported causal teacher storage {storage!r}")

    if frontend == "causal_evssm":
        evssm_checkpoint = _existing_file(
            cfg.get("evssm_checkpoint", ""), "evssm_checkpoint"
        )
        actual_evssm_sha = _sha256_file(evssm_checkpoint)
        if provenance.get("evssm_checkpoint_sha256") != actual_evssm_sha:
            raise ValueError(
                "causal_evssm teacher SHA-256 does not match the configured "
                "runtime evssm_checkpoint"
            )
        cfg["evssm_checkpoint"] = str(evssm_checkpoint)
        cfg["evssm_checkpoint_sha256"] = actual_evssm_sha

    provenance["storage"] = storage
    provenance["teacher_domain"] = teacher_domain
    return provenance


def _framecrafter_export_id(trajectory_path, input_root, fc, cfg):
    """Content-address the deterministic trajectory-to-CSV export contract."""

    rgb_list = _existing_file(input_root / "rgb.txt", "TUM rgb.txt")
    depth_list = _existing_file(input_root / "depth.txt", "TUM depth.txt")
    exporter = Path(__file__).resolve().parent / "scripts" / "export_framecrafter_trajectory.py"
    payload = {
        "schema": "unblur_slam.framecrafter_trajectory_export.v1",
        "tum_root": str(Path(input_root).expanduser().resolve()),
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": _sha256_file(trajectory_path),
        "trajectory_key": fc.get("trajectory_key", "traj_est_not_align") or None,
        "rgb_list_sha256": _sha256_file(rgb_list),
        "depth_list_sha256": _sha256_file(depth_list),
        "exporter_sha256": _sha256_file(exporter),
        "fx": float(cfg["cam"]["fx"]),
        "fy": float(cfg["cam"]["fy"]),
        "cx": float(cfg["cam"]["cx"]),
        "cy": float(cfg["cam"]["cy"]),
        "max_rgb_depth_dt": float(fc.get("max_rgb_depth_dt", 0.08)),
        "frame_rate": float(fc.get("frame_rate", 32.0)),
        "stride": int(cfg.get("stride", 1)),
        "max_frames": int(cfg.get("max_frames", -1)),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_framecrafter_manifest(path, expected_signature=None):
    path = _existing_file(path, "framecrafter.manifest")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    from src.framecrafter_pipeline import validate_manifest_payload

    validate_manifest_payload(
        payload, manifest_path=path, require_provenance=True
    )
    if expected_signature is not None:
        actual_signature = payload.get("preprocess_signature")
        if actual_signature != expected_signature:
            raise ValueError(
                "FrameCrafter manifest does not match the current source "
                "trajectory/images/model/settings: "
                f"expected signature {expected_signature}, got {actual_signature!r}"
            )
    return path


def _framecrafter_namespace(cfg, output_dir):
    fc = cfg["framecrafter"]
    dataset_root = Path(cfg["data"]["dataset_root"]).expanduser()
    input_root = dataset_root / cfg["data"]["input_folder"]

    def optional_path(key, default=None):
        value = fc.get(key, "")
        if value:
            return Path(value).expanduser().resolve()
        return default

    preprocess_dir = optional_path(
        "output_dir", Path(output_dir).resolve() / "framecrafter"
    )
    frames_csv = fc.get("frames_csv", "") or fc.get("pose_path", "")
    trajectory_npz = fc.get("trajectory_npz", "")
    trajectory_path = None
    generated_csv = None
    if trajectory_npz:
        trajectory_path = _existing_file(
            trajectory_npz, "framecrafter.trajectory_npz"
        )
        export_id = _framecrafter_export_id(
            trajectory_path, input_root.resolve(), fc, cfg
        )
        generated_csv = (
            preprocess_dir
            / "trajectory_exports"
            / f"estimated_frames_{export_id}.csv"
        ).resolve()
    if frames_csv and trajectory_npz:
        configured_csv = Path(frames_csv).expanduser().resolve()
        if configured_csv != generated_csv:
            raise ValueError(
                "configure exactly one of framecrafter.frames_csv and "
                "framecrafter.trajectory_npz; a saved auto-generated "
                "estimated_frames.csv is the only allowed overlap"
            )
    if trajectory_npz:
        if str(cfg.get("dataset", "")).lower() not in {"tumrgbd", "tumrgb"}:
            raise ValueError(
                "automatic trajectory-to-CSV export currently supports TUM RGB-D only"
            )
        from scripts.export_framecrafter_trajectory import run as export_trajectory

        trajectory_key = fc.get("trajectory_key", "traj_est_not_align") or None
        frames_csv = str(
            export_trajectory(
                SimpleNamespace(
                    trajectory_npz=_existing_file(
                        trajectory_path, "framecrafter.trajectory_npz"
                    ),
                    trajectory_key=trajectory_key,
                    tum_root=input_root.resolve(),
                    config=None,
                    output=generated_csv,
                    fx=float(cfg["cam"]["fx"]),
                    fy=float(cfg["cam"]["fy"]),
                    cx=float(cfg["cam"]["cx"]),
                    cy=float(cfg["cam"]["cy"]),
                    max_rgb_depth_dt=float(fc.get("max_rgb_depth_dt", 0.08)),
                    frame_rate=float(fc.get("frame_rate", 32.0)),
                    stride=int(cfg.get("stride", 1)),
                    max_frames=int(cfg.get("max_frames", -1)),
                )
            )
        )
    if not frames_csv:
        raise ValueError(
            "framecrafter.auto_prepare=true requires framecrafter.frames_csv or "
            "framecrafter.trajectory_npz from an unaligned first pass"
        )
    hybrid_roles_value = fc.get("hybrid_evssm_roles", ["local_blurry_inside"])
    if isinstance(hybrid_roles_value, str):
        hybrid_evssm_roles = tuple(
            role.strip() for role in hybrid_roles_value.split(",") if role.strip()
        )
    else:
        hybrid_evssm_roles = tuple(str(role) for role in hybrid_roles_value)
    return SimpleNamespace(
        frames_csv=_existing_file(frames_csv, "framecrafter.frames_csv"),
        planner_json=optional_path("planner_json"),
        planner_mode=str(fc.get("planner_mode", "legacy_pose_blur")),
        anchor_indices=optional_path("anchor_indices"),
        only_gap_left=fc.get("only_gap_left"),
        only_gap_right=fc.get("only_gap_right"),
        image_root=optional_path("image_root", input_root.resolve()),
        depth_root=optional_path("depth_root", input_root.resolve()),
        depth_scale=float(fc.get("depth_scale", cfg["cam"].get("png_depth_scale", 1.0))),
        output_depth_scale=float(
            fc.get("output_depth_scale", cfg["cam"].get("png_depth_scale", 5000.0))
        ),
        pose_convention=str(fc.get("pose_convention", "c2w")),
        pose_source=str(fc.get("pose_source", "droid_traj_est_not_align")),
        fx=float(cfg["cam"]["fx"]),
        fy=float(cfg["cam"]["fy"]),
        cx=float(cfg["cam"]["cx"]),
        cy=float(cfg["cam"]["cy"]),
        laplacian_threshold=fc.get("laplacian_threshold"),
        blur_quantile=float(fc.get("blur_quantile", 0.30)),
        translation_step=float(fc.get("translation_step", 0.08)),
        rotation_step_deg=float(fc.get("rotation_step_deg", 6.0)),
        blur_region_inserts=int(fc.get("blur_region_inserts", 1)),
        max_inserts=int(fc.get("max_inserts_per_gap", 4)),
        max_targets=int(fc.get("max_generated_frames", 256)),
        target_pair_overlap=float(fc.get("target_pair_overlap", 0.65)),
        hard_submap_overlap=float(fc.get("hard_submap_overlap", 0.05)),
        overlap_sample_stride=int(fc.get("overlap_sample_stride", 4)),
        include_blurry_regions=bool(fc.get("include_blurry_regions", True)),
        feature_refinement=bool(fc.get("feature_refinement", False)),
        feature_detector=str(fc.get("feature_detector", "orb")),
        feature_model=str(fc.get("feature_model", "essential")),
        feature_ambiguity_low=float(fc.get("feature_ambiguity_low", 0.15)),
        feature_ambiguity_high=float(fc.get("feature_ambiguity_high", 0.75)),
        feature_overlap_weight=float(fc.get("feature_overlap_weight", 0.20)),
        feature_refine_rotation=bool(fc.get("feature_refine_rotation", False)),
        feature_min_inlier_ratio=float(fc.get("feature_min_inlier_ratio", 0.35)),
        feature_max_rotation_correction_deg=float(
            fc.get("feature_max_rotation_correction_deg", 12.0)
        ),
        pnp_refinement=bool(fc.get("pnp_refinement", False)),
        pnp_detector=str(fc.get("pnp_detector", "orb")),
        pnp_max_features=int(fc.get("pnp_max_features", 3000)),
        pnp_ratio_test=float(fc.get("pnp_ratio_test", 0.75)),
        pnp_mutual_check=bool(fc.get("pnp_mutual_check", True)),
        pnp_min_keypoints=int(fc.get("pnp_min_keypoints", 12)),
        pnp_min_matches=int(fc.get("pnp_min_matches", 8)),
        pnp_min_depth=float(fc.get("pnp_min_depth", 1.0e-4)),
        pnp_max_depth=float(fc.get("pnp_max_depth", 20.0)),
        pnp_min_laplacian_variance=float(
            fc.get("pnp_min_laplacian_variance", 0.0)
        ),
        pnp_ambiguity_low=float(fc.get("pnp_ambiguity_low", 0.15)),
        pnp_ambiguity_high=float(fc.get("pnp_ambiguity_high", 0.75)),
        pnp_ransac_reprojection_error_px=float(
            fc.get("pnp_ransac_reprojection_error_px", 3.0)
        ),
        pnp_ransac_confidence=float(fc.get("pnp_ransac_confidence", 0.999)),
        pnp_ransac_iterations=int(fc.get("pnp_ransac_iterations", 200)),
        pnp_min_inliers=int(fc.get("pnp_min_inliers", 8)),
        pnp_min_inlier_ratio=float(fc.get("pnp_min_inlier_ratio", 0.35)),
        pnp_max_reprojection_rmse_px=float(
            fc.get("pnp_max_reprojection_rmse_px", 2.0)
        ),
        pnp_max_rotation_correction_deg=float(
            fc.get("pnp_max_rotation_correction_deg", 12.0)
        ),
        pnp_max_translation_correction=float(
            fc.get("pnp_max_translation_correction", 0.25)
        ),
        context_count=int(fc.get("context_views", 6)),
        min_contexts=int(fc.get("min_context_views", 3)),
        local_blurry_contexts=int(fc.get("local_blurry_contexts", 2)),
        sharp_contexts=int(fc.get("sharp_contexts", 2)),
        context_local_radius=int(fc.get("context_local_radius", 8)),
        context_search_radius=int(fc.get("context_search_radius", 32)),
        min_sharp_context_overlap=float(
            fc.get("min_sharp_context_overlap", 0.25)
        ),
        context_sharp_quantile=float(fc.get("context_sharp_quantile", 0.65)),
        context_image_mode=str(fc.get("context_image_mode", "raw")),
        hybrid_evssm_roles=hybrid_evssm_roles,
        evssm_metadata=optional_path("evssm_metadata"),
        evssm_min_confidence=float(fc.get("evssm_min_confidence", 0.50)),
        evssm_min_sharpness_gain=float(
            fc.get("evssm_min_sharpness_gain", 1.0)
        ),
        evssm_min_consistency=float(fc.get("evssm_min_consistency", 0.70)),
        evssm_local_gate_enabled=bool(fc.get("evssm_local_gate_enabled", True)),
        evssm_local_tile_size=int(fc.get("evssm_local_tile_size", 32)),
        evssm_local_tile_stride=int(fc.get("evssm_local_tile_stride", 16)),
        evssm_local_max_brightness_drop=float(
            fc.get("evssm_local_max_brightness_drop", 0.30)
        ),
        evssm_local_min_edge_retention=float(
            fc.get("evssm_local_min_edge_retention", 0.50)
        ),
        evssm_local_min_laplacian_retention=float(
            fc.get("evssm_local_min_laplacian_retention", 0.50)
        ),
        evssm_local_max_tile_mae=float(
            fc.get("evssm_local_max_tile_mae", 0.20)
        ),
        evssm_local_max_dark_expansion=float(
            fc.get("evssm_local_max_dark_expansion", 0.30)
        ),
        evssm_local_dark_luma_threshold=float(
            fc.get("evssm_local_dark_luma_threshold", 96.0 / 255.0)
        ),
        evssm_local_min_raw_luma=float(
            fc.get("evssm_local_min_raw_luma", 0.10)
        ),
        evssm_local_min_raw_edge=float(
            fc.get("evssm_local_min_raw_edge", 0.01)
        ),
        evssm_local_min_raw_laplacian=float(
            fc.get("evssm_local_min_raw_laplacian", 0.01)
        ),
        evssm_fallback=str(fc.get("evssm_fallback", "error")),
        backend=str(fc.get("backend", "python_api")),
        framecrafter_repo=optional_path("repo_path"),
        checkpoint=optional_path("checkpoint"),
        base_model_dir=optional_path("base_model_dir"),
        device=str(fc.get("device", cfg.get("device", "cuda:0"))),
        vram_limit=float(fc.get("vram_limit_gb", 20.0)),
        height=int(fc.get("height", 480)),
        width=int(fc.get("width", 832)),
        resize_mode=str(fc.get("resize_mode", "stretch")),
        num_inference_steps=int(fc.get("inference_steps", 20)),
        seed=int(fc.get("seed", cfg.get("setup_seed", 43))),
        cfg_scale=float(fc.get("cfg_scale", 1.0)),
        allow_test_only_backend=bool(fc.get("allow_test_only_backend", False)),
        plan_only=False,
        acceptance_mode=str(fc.get("acceptance_mode", "sharp")),
        min_sharpness_gain=float(fc.get("min_sharpness_gain", 1.05)),
        min_depth_coverage=float(fc.get("min_depth_coverage", 0.30)),
        min_depth_consistency=float(fc.get("min_depth_consistency", 0.50)),
        max_photometric_error=float(fc.get("max_photometric_error", 0.20)),
        max_reprojection_error_px=float(
            fc.get("max_reprojection_error_px", 2.0)
        ),
        min_reprojection_valid_ratio=float(
            fc.get("min_reprojection_valid_ratio", 0.05)
        ),
        depth_abs_tolerance=float(fc.get("depth_abs_tolerance", 0.03)),
        depth_rel_tolerance=float(fc.get("depth_rel_tolerance", 0.03)),
        allow_missing_depth_gates=bool(
            fc.get("allow_missing_depth_gates", False)
        ),
        output_dir=preprocess_dir,
    )


def prepare_or_validate_inputs(cfg, output_dir):
    """Resolve optional preprocessors before workers receive the config."""
    # The published TUM clear-frame protocol includes observations inside the
    # DROID warmup window (for example source 0/9/15 on fr2_xyz).  Mapping only
    # the last warmup keyframe would make baseline and enhanced runs evaluate
    # different subsets, depending on whether an enhanced frame happened to
    # request warmup replay.  Apply one recorded policy to both variants.
    if str(cfg.get("dataset", "")).lower() in {"tumrgbd", "tumrgb"}:
        if not bool(cfg.get("warmup_mapper", False)):
            cfg["warmup_mapper"] = True
            print(
                "Paper TUM protocol: enabling warmup_mapper so every "
                "clear-GT tracking anchor can reach mapping"
            )
    framecrafter = cfg.get("framecrafter", {}) or {}
    if framecrafter.get("enabled", False):
        manifest = str(framecrafter.get("manifest", "") or "")
        auto_prepare = bool(framecrafter.get("auto_prepare", False))

        if auto_prepare:
            from scripts.run_framecrafter_preprocess import (
                compute_preprocess_signature,
                run_preprocess,
                validate_backend_artifacts,
            )

            preprocess_args = _framecrafter_namespace(cfg, output_dir)
            # The real SLAM entrypoint never permits the endpoint-blend smoke
            # backend, even when the standalone preprocessor explicitly allows it.
            if str(preprocess_args.backend) != "python_api":
                raise ValueError(
                    "run.py accepts only framecrafter.backend=python_api; "
                    "test_only_blend is restricted to preprocessing tests"
                )
            validate_backend_artifacts(preprocess_args)
            expected_signature = compute_preprocess_signature(preprocess_args)
            preprocess_dir = Path(preprocess_args.output_dir).resolve()
            reuse_requested = bool(framecrafter.get("reuse_existing", True))
            if manifest:
                cache_candidates = [Path(manifest).expanduser().resolve()]
            else:
                cache_candidates = sorted(
                    preprocess_dir.glob(
                        f"manifest_{expected_signature}_*.json"
                    ),
                    key=lambda path: path.stat().st_mtime_ns,
                    reverse=True,
                )
            manifest = ""
            if reuse_requested:
                for cache_candidate in cache_candidates:
                    if not cache_candidate.is_file():
                        continue
                    try:
                        _validate_framecrafter_manifest(
                            cache_candidate,
                            expected_signature=expected_signature,
                        )
                        manifest = str(cache_candidate)
                        print(
                            "Reusing immutable FrameCrafter snapshot "
                            f"{cache_candidate.name}"
                        )
                        break
                    except (
                        ValueError,
                        TypeError,
                        KeyError,
                        OSError,
                        json.JSONDecodeError,
                    ) as error:
                        print(
                            "Ignoring stale/incompatible FrameCrafter cache "
                            f"{cache_candidate}: {error}"
                        )
            if not manifest:
                report = run_preprocess(
                    preprocess_args,
                    precomputed_signature=expected_signature,
                )
                if int(report.get("accepted_target_count", 0)) <= 0:
                    raise RuntimeError(
                        "FrameCrafter produced no candidate that passed the RGB-D gates"
                    )
                if report.get("preprocess_signature") != expected_signature:
                    raise RuntimeError(
                        "FrameCrafter preprocessing returned an unexpected content signature"
                    )
                manifest = str(report["manifest"])
            cfg["framecrafter"]["manifest"] = str(
                _validate_framecrafter_manifest(
                    manifest, expected_signature=expected_signature
                )
            )
            cfg["framecrafter"]["preprocess_signature"] = expected_signature
        else:
            if not manifest:
                raise ValueError(
                    "framecrafter.enabled=true requires an existing signed "
                    "production manifest or framecrafter.auto_prepare=true"
                )
            validated_manifest = _validate_framecrafter_manifest(manifest)
            with validated_manifest.open("r", encoding="utf-8") as handle:
                manifest_payload = json.load(handle)
            cfg["framecrafter"]["manifest"] = str(validated_manifest)
            cfg["framecrafter"]["preprocess_signature"] = str(
                manifest_payload["preprocess_signature"]
            )

    deblur = cfg.get("deblur", {}) or {}
    frontend = str(deblur.get("frontend", "evssm")).lower()
    if frontend in {"causal_torchscript", "causal_evssm"}:
        checkpoint = _existing_file(
            deblur.get("causal_checkpoint", ""), "deblur.causal_checkpoint"
        )
        history = int(deblur.get("causal_history", 0))
        if history < 1:
            raise ValueError("deblur.causal_history must be positive")
        if not bool(deblur.get("stream_every_frame", False)):
            raise ValueError(
                f"{frontend} requires deblur.stream_every_frame=true; "
                "keyframe-only calls do not form the advertised video stream"
            )
        # Loading on CPU catches a state-dict file accidentally supplied in
        # place of the exported streaming TorchScript module.  The export
        # metadata is part of the runtime contract: without it we cannot
        # prove that the configured temporal window fits the model.
        extra_files = {"metadata.json": ""}
        causal_model = torch.jit.load(
            str(checkpoint), map_location="cpu", _extra_files=extra_files
        )
        raw_metadata = extra_files.get("metadata.json", "")
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        if not raw_metadata:
            raise ValueError(
                "deblur.causal_checkpoint is missing export metadata; use "
                "scripts/export_causal_video_deblur.py"
            )
        try:
            metadata = json.loads(raw_metadata)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "deblur.causal_checkpoint has invalid metadata.json"
            ) from error
        expected_format = "unblur_slam.causal_video_deblur.torchscript.v1"
        if metadata.get("format") != expected_format:
            raise ValueError(
                "unsupported causal deblur checkpoint format: "
                f"{metadata.get('format')!r}"
            )
        try:
            max_history = int(metadata["model_config"]["max_history"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "causal deblur metadata must contain model_config.max_history"
            ) from error
        if history != max_history:
            raise ValueError(
                f"deblur.causal_history={history} must equal the checkpoint's "
                f"trained max_history={max_history}"
            )
        if bool(metadata["model_config"].get("use_teacher_input", False)):
            raise ValueError(
                "causal deblur checkpoint requires teacher frames at runtime; "
                "deploy only a one-input model trained with EVSSM distillation"
            )
        input_domain = str(
            metadata["model_config"].get("input_domain", "raw")
        ).lower()
        expected_domain = "evssm" if frontend == "causal_evssm" else "raw"
        if input_domain != expected_domain:
            raise ValueError(
                f"deblur.frontend={frontend} requires a causal checkpoint "
                f"trained with input_domain={expected_domain!r}, got "
                f"{input_domain!r}"
            )
        teacher_provenance = _validate_causal_teacher_provenance(
            metadata, frontend, cfg
        )
        if frontend == "causal_evssm":
            try:
                min_vs_evssm = float(
                    deblur.get("stream_min_vs_evssm_gain", 0.0)
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "deblur.stream_min_vs_evssm_gain must be a finite non-negative number"
                ) from error
            if not np.isfinite(min_vs_evssm) or min_vs_evssm < 0.0:
                raise ValueError(
                    "deblur.stream_min_vs_evssm_gain must be a finite non-negative number"
                )
        try:
            with torch.no_grad():
                probe = torch.zeros(1, history, 3, 16, 16)
                probe_output = causal_model(probe)
        except Exception as error:
            raise ValueError(
                "causal deblur checkpoint failed the single-input CPU runtime contract"
            ) from error
        if (
            not torch.is_tensor(probe_output)
            or tuple(probe_output.shape) != (1, 3, 16, 16)
            or not bool(torch.isfinite(probe_output).all())
        ):
            raise ValueError(
                "causal deblur checkpoint must return finite BCHW RGB with "
                "the input spatial resolution"
            )
        cfg["deblur"]["causal_checkpoint"] = str(checkpoint)
        cfg["deblur"]["causal_checkpoint_sha256"] = _sha256_file(checkpoint)
        cfg["deblur"]["causal_teacher_provenance"] = teacher_provenance
        cfg["deblur"]["causal_teacher_storage"] = teacher_provenance["storage"]
    elif frontend in {"turtle_streaming", "turtle_bsd_streaming"}:
        if not bool(deblur.get("stream_every_frame", False)):
            raise ValueError(
                f"{frontend} requires deblur.stream_every_frame=true; "
                "its official K/V state must advance on every video frame"
            )
        try:
            min_gain = float(deblur.get("stream_min_laplacian_gain", 0.0))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "deblur.stream_min_laplacian_gain must be a finite non-negative number"
            ) from error
        forced_bsd_diagnostic = bool(
            frontend == "turtle_bsd_streaming"
            and min_gain == -1.0
            and bool(
                (cfg.get("paired_official_online_budget_221", {}) or {}).get(
                    "unsafe_forced_bsd_tracking_diagnostic", False
                )
            )
            and bool(
                (cfg.get("paired_official_online_budget_221", {}) or {}).get(
                    "safe_primary_report_member", True
                )
                is False
            )
        )
        if not np.isfinite(min_gain) or (min_gain < 0.0 and not forced_bsd_diagnostic):
            raise ValueError(
                "deblur.stream_min_laplacian_gain must be finite and non-negative "
                "outside the explicit unsafe BSD forced-tracking diagnostic"
            )
        # Strict CPU construction catches a mis-keyed or shape-incompatible
        # file before SLAM allocates any GPU state.  BSD uses the official t0
        # model; GoPro uses t1, so their architecture validators stay separate.
        if frontend == "turtle_bsd_streaming":
            from src.turtle_official_bsd_backend import (
                validate_official_bsd_artifacts,
            )

            artifacts = validate_official_bsd_artifacts(
                repo=deblur.get("turtle_repo"),
                config=deblur.get("turtle_config"),
                checkpoint=deblur.get("turtle_checkpoint"),
                checkpoint_sha256=deblur.get("turtle_checkpoint_sha256"),
                load_weights=True,
            )
            commit = str(artifacts.checkpoint_metadata["turtle_repo_commit"])
        else:
            from src.turtle_backend import validate_turtle_artifacts

            artifacts = validate_turtle_artifacts(deblur, load_weights=True)
            commit = artifacts.commit
        cfg["deblur"].update(
            {
                "turtle_repo": str(artifacts.repo),
                "turtle_config": str(artifacts.config),
                "turtle_checkpoint": str(artifacts.checkpoint),
                "turtle_repo_commit": commit,
                "turtle_config_sha256": (
                    artifacts.config_sha256
                    if hasattr(artifacts, "config_sha256")
                    else _sha256_file(artifacts.config)
                ),
                "turtle_checkpoint_sha256": artifacts.checkpoint_sha256,
                "turtle_checkpoint_metadata": dict(
                    artifacts.checkpoint_metadata
                ),
            }
        )
    elif frontend == "precomputed":
        root = Path(deblur.get("precomputed_root", "")).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"deblur.precomputed_root does not exist: {root}")
        cfg["deblur"]["precomputed_root"] = str(root)
    elif frontend != "evssm":
        raise ValueError(f"Unsupported deblur.frontend={frontend!r}")

    resplat = validate_resplat_config(
        cfg.get("mapping", {}).get("resplat", {}) or {}
    )
    if resplat.get("enabled", False):
        if not cfg.get("tracking", {}).get("backend", {}).get("final_ba", False):
            raise ValueError(
                "mapping.resplat requires tracking.backend.final_ba=true so final_refine runs"
            )
        mode = str(resplat.get("budget_mode", "replace_tail"))
        if mode not in {"replace_tail", "extend"}:
            raise ValueError("mapping.resplat.budget_mode must be replace_tail or extend")
        replay_iters = int(resplat.get("extra_iters", 0))
        base_budget = int(cfg["mapping"]["final_refine_iters"])
        if replay_iters < 0 or (mode == "replace_tail" and replay_iters > base_budget):
            raise ValueError("Invalid residual replay iteration budget")

    return cfg


def validate_clear_gt_scope(cfg, dataset):
    """Fail before workers start if the configured metric scope is invalid."""
    available = validate_clear_gt_protocol_scope(cfg, dataset)
    if available is None:
        return None
    mode = clear_gt_scope_mode(cfg)
    if mode == "prefix_smoke":
        print(
            "Evaluation scope: clear-GT protocol prefix smoke "
            f"({len(available)} frames; NOT a complete paper metric)"
        )
    else:
        print(
            "Evaluation scope: clear-GT only "
            f"({len(available)} complete protocol frames)"
        )
    return available

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument("--only_tracking", action="store_true", help="Only tracking is triggered")
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn', force=True)

    repository_root = Path(__file__).resolve().parent
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (
            config_path.resolve()
            if config_path.is_file()
            else (repository_root / config_path).resolve()
        )
    else:
        config_path = config_path.resolve()
    # Repository configs intentionally use repository-relative dataset,
    # checkpoint and output paths. Make that contract independent of the
    # shell's launch directory.
    os.chdir(repository_root)
    cfg = config.load_config(
        config_path, repository_root / 'configs' / 'unblur_slam.yaml'
    )
    setup_seed(cfg['setup_seed'])

    if args.only_tracking:
        cfg['only_tracking'] = True
        cfg['mono_prior']['predict_online'] = True

    output_dir = cfg['data']['output']
    output_dir = output_dir+f"/{cfg['scene']}"

    start_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    start_info = "-"*30+Fore.LIGHTRED_EX+\
                 f"\nStart Unblur-SLAM at {start_time},\n"+Style.RESET_ALL+ \
                 f"   scene: {cfg['dataset']}-{cfg['scene']},\n" \
                 f"   only_tracking: {cfg['only_tracking']},\n" \
                 f"   output: {output_dir}\n"+ \
                 "-"*30
    print(start_info)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prepare_or_validate_inputs(cfg, output_dir)
    dataset = get_dataset(cfg)
    validate_clear_gt_scope(cfg, dataset)
    config.save_config(cfg, f'{output_dir}/cfg.yaml')

    slam = SLAM(cfg,dataset)
    slam.run()

    end_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print("-"*30+Fore.LIGHTRED_EX+f"\nUnblur-SLAM finishes!\n"+Style.RESET_ALL+f"{end_time}\n"+"-"*30)
