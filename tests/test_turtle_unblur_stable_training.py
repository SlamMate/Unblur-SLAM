#!/usr/bin/env python3
"""CPU contracts for the resumable Unblur-style TURTLE trainer."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_turtle_unblur_stable import (
    BASE_LR,
    BATCH_SIZE_PER_GPU,
    GLOBAL_BATCH_SIZE,
    DDP_WORLD_SIZE,
    DDP_BACKEND,
    AMP_MAX_SAME_BATCH_RETRIES,
    AMP_INITIAL_SCALE,
    AMP_GROWTH_INTERVAL,
    CLIP_LENGTH,
    CROP_SIZE,
    FFT_WEIGHT,
    PHOTOMETRIC_TRANSFORM,
    SCHEMA,
    TOTAL_STEPS,
    _read_contract,
    _next_amp_retry_scale,
    best_integer_alignment,
    causal_clip_loss,
    configure_full_scopes,
    load_paired_image_train_manifest,
    srgb_to_linear,
    verify_content_addressed_video_manifest,
)
from scripts.build_unblur_hf_pair_manifest import rows as build_pair_rows
from scripts.build_paired_video_train_manifest import from_manifest, _write_exclusive
from scripts.materialize_replica_blurry_office3 import materialize as materialize_office3
from scripts.train_turtle_streaming import (
    DEFAULT_TURTLE_CHECKPOINT,
    DEFAULT_TURTLE_CONFIG,
    DEFAULT_TURTLE_REPO,
)
from src.turtle_backend import (
    PINNED_TURTLE_ARCH_SHA256,
    PINNED_TURTLE_CHECKPOINT_SHA256,
    PINNED_TURTLE_COMMIT,
    PINNED_TURTLE_CONFIG_SHA256,
    TurtleStreamingBackend,
    STABLE_DEPLOY_CHECKPOINT_FORMAT,
    TURTLE_CACHE_CONTRACT,
    linear_to_srgb,
    build_turtle_model_from_scratch,
    load_turtle_model,
    sha256_file,
    srgb_to_linear as runtime_srgb_to_linear,
    validate_turtle_checkpoint_payload,
)


class _CausalIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, pair, k_cache, v_cache):
        current = pair[:, 1] * self.scale
        caches = [None, None, None] + [current.mean(dim=(-2, -1))] * 5
        return current, caches, [value for value in caches]


class _LinearDomainIdentity(_CausalIdentity):
    def __init__(self) -> None:
        super().__init__()
        self.turtle_checkpoint_metadata = {"input_domain": "linear_srgb"}


class StableTrainerContracts(unittest.TestCase):
    def test_ddp_local3_gradient_mean_equals_global6(self) -> None:
        torch.manual_seed(7)
        inputs = torch.randn(GLOBAL_BATCH_SIZE, 4)
        targets = torch.randn(GLOBAL_BATCH_SIZE, 2)
        global_model = nn.Linear(4, 2, bias=False)
        initial = global_model.weight.detach().clone()
        torch.nn.functional.mse_loss(global_model(inputs), targets).backward()
        expected = global_model.weight.grad.detach().clone()
        local_gradients = []
        for rank in range(DDP_WORLD_SIZE):
            local_model = nn.Linear(4, 2, bias=False)
            local_model.weight.data.copy_(initial)
            start = rank * BATCH_SIZE_PER_GPU
            end = start + BATCH_SIZE_PER_GPU
            torch.nn.functional.mse_loss(
                local_model(inputs[start:end]), targets[start:end]
            ).backward()
            local_gradients.append(local_model.weight.grad.detach().clone())
        torch.testing.assert_close(torch.stack(local_gradients).mean(0), expected)

    def test_amp_retry_scale_halves_without_underflow(self) -> None:
        self.assertEqual(_next_amp_retry_scale(1024.0), 512.0)
        self.assertEqual(_next_amp_retry_scale(1.0), 1.0)
        with self.assertRaises(ValueError):
            _next_amp_retry_scale(float("nan"))

    def test_office3_linear_exposure_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            for index in range(181):
                value = 0 if index < 18 else 255
                Image.new("RGB", (4, 3), (value, value, value)).save(
                    source / f"rgb_{index * 2}.png"
                )
            output = root / "output"
            audit = materialize_office3(source, output)
            self.assertEqual(audit["formation"]["complete_exposures"], 5)
            self.assertEqual(audit["formation"]["trailing_frames_ignored"], 1)
            self.assertTrue((output / "sharp/rgb_36.png").is_file())
            self.assertTrue((output / "blur/rgb_0.png").is_file())
            self.assertEqual(audit["source"]["numeric_gap_count"], 180)
            verify_content_addressed_video_manifest(
                output / "manifests/train.jsonl", output
            )

    def test_exact_srgb_inverse_transfer(self) -> None:
        values = torch.tensor([0.0, 0.04045, 1.0])
        actual = srgb_to_linear(values)
        expected = torch.tensor([0.0, 0.04045 / 12.92, 1.0])
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(actual, runtime_srgb_to_linear(values))
        torch.testing.assert_close(
            linear_to_srgb(actual), values, rtol=1e-5, atol=1e-7
        )

    def test_linear_domain_runtime_roundtrip_and_internal_state(self) -> None:
        backend = TurtleStreamingBackend(_LinearDomainIdentity(), device="cpu")
        image = torch.tensor([[[[0.02, 0.5], [0.8, 1.0]]] * 3])
        output = backend.step(image, timestamp=0)
        torch.testing.assert_close(output, image, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(
            backend.previous_frame, runtime_srgb_to_linear(image)
        )
        self.assertEqual(backend.state_info()["photometric_domain"], "linear_srgb")

    def test_stable_deploy_metadata_is_fail_closed(self) -> None:
        metadata = {
            "format": STABLE_DEPLOY_CHECKPOINT_FORMAT,
            "stage": "defocus_rehearsal",
            "step": TOTAL_STEPS,
            "initialization_root": "random_scratch_pinned_turtle_architecture",
            "official_gopro_checkpoint_used_for_initialization": False,
            "turtle_repo_commit": PINNED_TURTLE_COMMIT,
            "turtle_arch_sha256": PINNED_TURTLE_ARCH_SHA256,
            "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
            "input_domain": "linear_srgb",
            "output_domain": "linear_srgb",
            "photometric_transform": PHOTOMETRIC_TRANSFORM,
            "cache_contract": TURTLE_CACHE_CONTRACT,
        }
        payload = {"params": {"weight": torch.ones(1)}, "metadata": metadata}
        _, accepted = validate_turtle_checkpoint_payload(
            payload,
            checkpoint_sha256="a" * 64,
            expected_checkpoint_sha256="a" * 64,
        )
        self.assertEqual(accepted["kind"], "finetuned_unblur_stable")
        rejected = dict(metadata)
        rejected["stage"] = "motion"
        with self.assertRaisesRegex(ValueError, "unsafe stable"):
            validate_turtle_checkpoint_payload(
                {"params": payload["params"], "metadata": rejected},
                checkpoint_sha256="b" * 64,
                expected_checkpoint_sha256="b" * 64,
            )

    def test_integer_alignment_recovers_known_shift(self) -> None:
        target = torch.zeros(1, 1, 16, 16)
        target[..., 5:9, 6:10] = 1.0
        prediction = torch.zeros_like(target)
        prediction[..., 6:10, 4:8] = 1.0
        prediction_view, target_view, shift = best_integer_alignment(
            prediction, target, radius=3
        )
        self.assertEqual(shift, (1, -2))
        self.assertEqual(float((prediction_view - target_view).abs().max()), 0.0)

    def test_causal_clip_keeps_five_frame_graph(self) -> None:
        model = _CausalIdentity()
        blurry = torch.rand(CLIP_LENGTH, 3, 16, 16)
        sharp = blurry.clone()
        loss, audit = causal_clip_loss(
            model,
            blurry,
            sharp,
            device=torch.device("cpu"),
            alignment_radius=0,
            amp=False,
        )
        self.assertEqual(len(audit["frames"]), CLIP_LENGTH)
        loss.backward()
        self.assertIsNotNone(model.scale.grad)
        self.assertTrue(torch.isfinite(model.scale.grad))

        batched = torch.rand(BATCH_SIZE_PER_GPU, CLIP_LENGTH, 3, 16, 16)
        batch_model = _CausalIdentity()
        batch_loss, batch_audit = causal_clip_loss(
            batch_model,
            batched,
            batched.clone(),
            device=torch.device("cpu"),
            alignment_radius=0,
            amp=False,
        )
        self.assertEqual(len(batch_audit["frames"]), CLIP_LENGTH)
        self.assertTrue(all(
            len(frame["samples"]) == BATCH_SIZE_PER_GPU
            for frame in batch_audit["frames"]
        ))
        batch_loss.backward()
        self.assertIsNotNone(batch_model.scale.grad)

    def test_contract_rejects_test_and_bad_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "train.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            payload = {
                "schema": SCHEMA,
                "stage": "motion_base",
                "initialization": {"kind": "random_scratch_pinned_turtle_architecture"},
                "implementation": [{
                    "path": str(manifest), "sha256": sha256_file(manifest),
                }],
                "test_pixels_permitted": False,
                "total_steps": TOTAL_STEPS,
                "crop_size": CROP_SIZE,
                "clip_length": CLIP_LENGTH,
                "optimizer": "AdamW",
                "learning_rate": BASE_LR,
                "weight_decay": 1e-3,
                "betas": [0.9, 0.9],
                "scheduler": "CosineAnnealingLR",
                "scheduler_t_max": TOTAL_STEPS,
                "scheduler_eta_min": 1e-7,
                "gradient_clip_norm": 1.0,
                "fft_weight": FFT_WEIGHT,
                "photometric_transform": PHOTOMETRIC_TRANSFORM,
                "initialization_root": "random_scratch_pinned_turtle_architecture",
                "official_gopro_checkpoint_used_for_initialization": False,
                "turtle_repo_commit": PINNED_TURTLE_COMMIT,
                "turtle_arch_sha256": PINNED_TURTLE_ARCH_SHA256,
                "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
                "seed": 42,
                "checkpoint_every": 6_000,
                "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "ddp_world_size": DDP_WORLD_SIZE,
                "ddp_backend": DDP_BACKEND,
                "gradient_reduction": "ddp_mean_equivalent_to_global_batch6_mean",
                "amp_overflow_policy": "lower_scale_and_retry_exact_same_batch_without_scheduler_or_sample_advance",
                "amp_max_same_batch_retries": AMP_MAX_SAME_BATCH_RETRIES,
                "amp_initial_scale": AMP_INITIAL_SCALE,
                "amp_growth_interval": AMP_GROWTH_INTERVAL,
                "ddp_source_choice": "identical_seed_and_allreduce_identity_check",
                "ddp_rank_data_rng": "seed_plus_rank_times_1000000_plus_source_offset",
                "distributed_topology": "single_node_exact_visible_gpu_count",
                "launch_workflow": "same_dual_gpu_lease_step1_probe_then_exact_checkpoint_resume_to_300000",
                "video_batch_implementation": "ddp2_local_batch3_global_batch6_five_frame_bptt_one_optimizer_step",
                "amp": True,
                "validation_during_training": False,
                "data_mix_disclosure": {
                    "paper_discloses_exact_weights": False,
                    "weights_are_preregistered_extension": True,
                    "sampling": "strict_stage_order_then_one_source_per_step_ddp_global_batch6",
                    "model_selection_or_validation_during_training": False,
                    "bsd_dpdd_validation_and_tum_are_held_out": True,
                    "bsd_training_pixels_used": False,
                    "tum_training_pixels_used": False,
                },
                "sources": [
                    {
                        "name": name, "kind": "video", "split": "train",
                        "weight": weight, "alignment_radius": 0,
                        "manifest": str(manifest),
                        "manifest_sha256": sha256_file(manifest),
                        "root": str(root),
                        "provenance": {
                            "repository": repository,
                            "revision": revision,
                            "artifacts": [{
                                "path": str(manifest),
                                "sha256": sha256_file(manifest),
                            }],
                        },
                    }
                    for name, weight, repository, revision in (
                        ("reds", 0.55, "snah/REDS", "62dc25d16e6f43d2214f1b365023abda86f7a0ae"),
                        ("gopro_blur_gamma", 0.45, "snah/GOPRO_Large", "592978466ae510d2734b199cad2fc79a346bda1c"),
                    )
                ],
            }
            contract = root / "contract.json"
            contract.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_read_contract(contract)["schema"], SCHEMA)
            for world_size, local_batch in ((3, 2), (6, 1)):
                payload["ddp_world_size"] = world_size
                payload["batch_size_per_gpu"] = local_batch
                payload["video_batch_implementation"] = (
                    f"ddp{world_size}_local_batch{local_batch}_global_batch6_"
                    "five_frame_bptt_one_optimizer_step"
                )
                contract.write_text(json.dumps(payload), encoding="utf-8")
                accepted = _read_contract(contract)
                self.assertEqual(accepted["global_batch_size"], 6)
            payload["ddp_world_size"] = 4
            payload["batch_size_per_gpu"] = 1
            payload["video_batch_implementation"] = (
                "ddp4_local_batch1_global_batch6_five_frame_bptt_one_optimizer_step"
            )
            contract.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ddp_world_size"):
                _read_contract(contract)
            payload["ddp_world_size"] = 6
            payload["batch_size_per_gpu"] = 1
            payload["video_batch_implementation"] = (
                "ddp6_local_batch1_global_batch6_five_frame_bptt_one_optimizer_step"
            )
            payload["sources"][0]["split"] = "test"
            contract.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only train"):
                _read_contract(contract)

    def test_actual_turtle_partition_is_exhaustive(self) -> None:
        torch.manual_seed(42)
        model, metadata = build_turtle_model_from_scratch(
            DEFAULT_TURTLE_REPO,
            config=DEFAULT_TURTLE_CONFIG,
            device=torch.device("cpu"),
        )
        self.assertEqual(metadata["kind"], "random_initialization")
        scopes = configure_full_scopes(model)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in scopes.history + scopes.spatial),
        )
        self.assertGreater(sum(p.numel() for p in scopes.spatial), 50_000_000)

    def test_generic_pair_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            low_dir, high_dir = root / "train/input", root / "train/target"
            low_dir.mkdir(parents=True); high_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), (10, 20, 30)).save(low_dir / "0.png")
            Image.new("RGB", (16, 16), (11, 21, 31)).save(high_dir / "0.png")
            payload = build_pair_rows(root, [("toy", low_dir, high_dir)])
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(payload[0]) + "\n", encoding="utf-8")
            records = load_paired_image_train_manifest(manifest, root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].name, "toy/0.png")

    def test_video_canonicalization_binds_assets_and_rejects_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train/blur").mkdir(parents=True)
            (root / "train/sharp").mkdir(parents=True)
            blurry, sharp = [], []
            for index in range(5):
                low = root / f"train/blur/{index:03d}.png"
                high = root / f"train/sharp/{index:03d}.png"
                Image.new("RGB", (16, 16), (index, 1, 2)).save(low)
                Image.new("RGB", (16, 16), (index, 3, 4)).save(high)
                blurry.append(str(low.relative_to(root)))
                sharp.append(str(high.relative_to(root)))
            source = root / "source.jsonl"
            source.write_text(json.dumps({
                "sequence": "s", "blurry": blurry, "sharp": sharp,
            }) + "\n", encoding="utf-8")
            canonical = root / "canonical.jsonl"
            _write_exclusive(from_manifest(root, source, "fake"), canonical)
            verify_content_addressed_video_manifest(canonical, root)
            tampered = json.loads(canonical.read_text(encoding="utf-8"))
            tampered["blurry"][0] = "test/missing.png"
            bad = root / "bad.jsonl"
            bad.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sealed split"):
                verify_content_addressed_video_manifest(bad, root)


if __name__ == "__main__":
    unittest.main()
