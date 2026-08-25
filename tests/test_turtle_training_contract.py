#!/usr/bin/env python3
"""CPU-only contracts for TURTLE streaming fine-tuning and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_turtle_streaming import (  # noqa: E402
    evaluate_sequences,
    history_control_outputs,
    image_metrics,
    prepare_output_directory,
    temporal_metrics,
)
from scripts.train_turtle_streaming import (  # noqa: E402
    PairedSequenceDataset,
    build_checkpoint_metadata,
    configure_trainable_scope,
    load_sequence_manifest,
    save_checkpoint,
    train_sequence,
    train_sequence_full_bptt,
)
from src.turtle_backend import (  # noqa: E402
    PINNED_TURTLE_CHECKPOINT_SHA256,
    validate_turtle_checkpoint_payload,
)


def _write_image(path: Path, value: int, *, pattern: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pattern:
        yy, xx = np.mgrid[:12, :16]
        array = np.stack(
            (
                (xx * 13 + value) % 256,
                (yy * 19 + value) % 256,
                ((xx + yy) * 7 + value) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
    else:
        array = np.full((8, 8, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _paired_manifest(
    root: Path,
    lengths: tuple[int, ...] = (3, 2),
    *,
    identical_targets: bool = False,
    pattern: bool = False,
) -> Path:
    lines = []
    for sequence_index, length in enumerate(lengths):
        blurry = []
        sharp = []
        for frame_index in range(length):
            blur_path = root / f"s{sequence_index}/blur_{frame_index}.png"
            sharp_path = root / f"s{sequence_index}/sharp_{frame_index}.png"
            value = 20 + sequence_index * 40 + frame_index * 10
            _write_image(blur_path, value, pattern=pattern)
            _write_image(
                sharp_path,
                value if identical_targets else value + 5,
                pattern=pattern,
            )
            blurry.append(str(blur_path.relative_to(root)))
            sharp.append(str(sharp_path.relative_to(root)))
        lines.append(
            json.dumps(
                {
                    "sequence": f"sequence_{sequence_index}",
                    "blurry": blurry,
                    "sharp": sharp,
                }
            )
        )
    manifest = root / "paired.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


class FakeTrainTurtle(torch.nn.Module):
    """Tiny differentiable implementation of TURTLE's public forward API."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.9))
        self.calls = []

    def forward(self, pair, k_cache=None, v_cache=None):
        self.calls.append(
            {
                "pair": pair.detach().clone(),
                "had_k": k_cache is not None,
                "had_v": v_cache is not None,
            }
        )
        current = pair[:, 1]
        # Keep a harmless dependency on the cache to exercise graph detachment.
        prior = current.new_zeros(()) if k_cache is None else k_cache[3].mean()
        restored = current * self.scale + prior * 0.0
        marker = self.scale.reshape(1) * 0.0 + float(len(self.calls))
        k_new = [None, None, None] + [marker.clone() for _ in range(5)]
        v_new = [None, None, None] + [marker.clone() for _ in range(5)]
        return restored, k_new, v_new


class CountingAdamW(torch.optim.AdamW):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure=closure)


class FakeHistoryTurtle(torch.nn.Module):
    """Cache-sensitive model with TURTLE's exact sparse eight-slot mask."""

    use_both_input = False

    def forward(self, pair, k_cache=None, v_cache=None):
        current = pair[:, 1]
        history = current.new_zeros(())
        if k_cache is not None:
            history = k_cache[3].mean()
        restored = (current + 0.1 * history).clamp(0.0, 1.0)
        marker = current.mean().reshape(1)
        k_new = [None, None, None] + [marker.clone() for _ in range(5)]
        v_new = [None, None, None] + [marker.clone() for _ in range(5)]
        return restored, k_new, v_new


class FakeEvaluationBackend:
    def __init__(self):
        self.reset_count = 0
        self.calls = []

    def reset(self):
        self.reset_count += 1

    def step(self, image, timestamp=None):
        self.calls.append((self.reset_count, timestamp, image.detach().clone()))
        return (image + 0.02).clamp(0.0, 1.0)


class TurtleTrainingSequenceTest(unittest.TestCase):
    def test_full_sequence_bptt_steps_once_after_all_frames(self):
        model = FakeTrainTurtle()
        optimizer = CountingAdamW(model.parameters(), lr=0.0)
        blurry = torch.stack(
            [torch.full((3, 8, 8), value) for value in (0.1, 0.2, 0.3)]
        )
        sharp = blurry * 0.8
        row = train_sequence_full_bptt(
            model,
            blurry,
            sharp,
            optimizer,
            device=torch.device("cpu"),
            fft_weight=0.1,
            loss_start_frame=1,
        )
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(optimizer.step_count, 1)
        self.assertEqual(row["frames"], 3)
        self.assertEqual(row["supervised_frames"], 2)
        self.assertTrue(model.calls[1]["had_k"])

    def test_invalid_trainable_scope_fails_closed(self):
        with self.assertRaises(ValueError):
            configure_trainable_scope(FakeTrainTurtle(), "spatial_adapter")

    def test_training_left_repeats_start_and_never_reuses_another_sequence_cache(self):
        model = FakeTrainTurtle()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
        blurry = torch.stack(
            [torch.full((3, 8, 8), value) for value in (0.1, 0.2, 0.3)]
        )
        sharp = blurry * 0.8

        rows = train_sequence(
            model,
            blurry,
            sharp,
            optimizer,
            device=torch.device("cpu"),
            step_budget=3,
            fft_weight=0.1,
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(model.calls), 3)
        first, second, third = model.calls
        self.assertFalse(first["had_k"])
        self.assertFalse(first["had_v"])
        self.assertTrue(torch.equal(first["pair"][:, 0], blurry[0:1]))
        self.assertTrue(torch.equal(first["pair"][:, 1], blurry[0:1]))
        self.assertTrue(second["had_k"])
        self.assertTrue(third["had_v"])
        self.assertTrue(torch.equal(second["pair"][:, 0], blurry[0:1]))
        self.assertTrue(torch.equal(second["pair"][:, 1], blurry[1:2]))

        next_blurry = torch.full((1, 3, 8, 8), 0.7)
        train_sequence(
            model,
            next_blurry,
            next_blurry,
            optimizer,
            device=torch.device("cpu"),
            step_budget=1,
        )
        next_call = model.calls[-1]
        self.assertFalse(next_call["had_k"])
        self.assertFalse(next_call["had_v"])
        self.assertTrue(torch.equal(next_call["pair"][:, 0], next_blurry))
        self.assertTrue(torch.equal(next_call["pair"][:, 1], next_blurry))

    def test_sequence_dataset_applies_one_transform_to_all_frames_and_modalities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _paired_manifest(
                root,
                lengths=(3, 2),
                identical_targets=True,
                pattern=True,
            )
            dataset = PairedSequenceDataset(
                manifest, root=root, crop_size=8, augment=True, seed=19
            )
            dataset.set_epoch(4)
            first = dataset[0]
            repeated = dataset[0]
            self.assertEqual(len(dataset), 2)
            self.assertEqual(tuple(first["blurry"].shape), (3, 3, 8, 8))
            self.assertTrue(torch.equal(first["blurry"], first["sharp"]))
            self.assertTrue(torch.equal(first["blurry"], repeated["blurry"]))

            dataset.set_epoch(5)
            changed_epoch = dataset[0]
            # It is possible but extremely unlikely for all random transform
            # choices to coincide; the deterministic epoch seed is the contract.
            self.assertFalse(torch.equal(first["blurry"], changed_epoch["blurry"]))

    def test_manifest_keeps_jsonl_records_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _paired_manifest(root, lengths=(3, 2))
            records = load_sequence_manifest(manifest, root=root)
            self.assertEqual([record.name for record in records], ["sequence_0", "sequence_1"])
            self.assertEqual([len(record.blurry) for record in records], [3, 2])


class TurtleCheckpointContractTest(unittest.TestCase):
    def test_metadata_is_strict_loader_compatible_and_binds_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest = root / "train.jsonl"
            val_manifest = root / "val.jsonl"
            train_manifest.write_text("train contract\n", encoding="utf-8")
            val_manifest.write_text("validation contract\n", encoding="utf-8")
            metadata = build_checkpoint_metadata(
                base_metadata={
                    "kind": "official_gopro",
                    "base_checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
                },
                train_manifest=train_manifest,
                val_manifest=val_manifest,
                seed=42,
                steps=7,
                crop_size=192,
                augment=True,
                fft_weight=0.1,
                learning_rate=1.0e-5,
                weight_decay=1.0e-4,
                betas=(0.9, 0.99),
                amp=True,
                best_val_psnr=31.25,
            )
            digest = "a" * 64
            state, normalized = validate_turtle_checkpoint_payload(
                {"params": {"weight": torch.ones(1)}, "metadata": metadata},
                checkpoint_sha256=digest,
                expected_checkpoint_sha256=digest,
            )
            self.assertIn("weight", state)
            self.assertEqual(normalized["kind"], "finetuned")
            self.assertFalse(normalized["uses_gt"])
            self.assertFalse(normalized["uses_gt_pose"])
            self.assertTrue(normalized["uses_sharp_rgb_supervision"])
            self.assertEqual(
                normalized["history"]["sequence_boundary"], "hard_reset"
            )
            self.assertEqual(len(normalized["manifests"]["train_sha256"]), 64)

    def test_checkpoint_save_refuses_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "finetuned.pth"
            model = torch.nn.Linear(2, 2)
            digest = save_checkpoint(
                output, model=model, metadata={"format": "test"}
            )
            self.assertEqual(len(digest), 64)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_name(output.name + ".sha256").is_file())
            with self.assertRaises(FileExistsError):
                save_checkpoint(output, model=model, metadata={"format": "test"})


class TurtleEvaluationContractTest(unittest.TestCase):
    def test_history_controls_change_only_explicit_cache_content(self):
        model = FakeHistoryTurtle().eval()
        current = torch.full((1, 3, 8, 8), 0.3)
        past = [
            torch.full((1, 3, 8, 8), value) for value in (0.1, 0.2)
        ]
        outputs = history_control_outputs(model, current, past)
        self.assertEqual(
            set(outputs),
            {
                "turtle_reset_cache",
                "turtle_repeat_current",
                "turtle_replayed_ordered",
                "turtle_shuffled_history",
            },
        )
        self.assertFalse(
            torch.equal(outputs["turtle_reset_cache"], outputs["turtle_repeat_current"])
        )
        self.assertFalse(
            torch.equal(
                outputs["turtle_replayed_ordered"],
                outputs["turtle_shuffled_history"],
            )
        )

    def test_history_controls_replay_complete_prefix_not_only_direct_capacity(self):
        model = FakeHistoryTurtle().eval()
        current = torch.full((1, 3, 8, 8), 0.6)
        past = [
            torch.full((1, 3, 8, 8), value)
            for value in (0.1, 0.2, 0.3, 0.4, 0.5)
        ]
        calls = {"count": 0}
        original = model.forward

        def counted(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        model.forward = counted
        history_control_outputs(model, current, past)
        # reset=1 plus three controls, each with five prefix calls + target.
        self.assertEqual(calls["count"], 1 + 3 * (len(past) + 1))

    def test_metrics_and_temporal_difference_have_explicit_meaning(self):
        zero = torch.zeros(3, 8, 8)
        half = torch.full((3, 8, 8), 0.5)
        one = torch.ones(3, 8, 8)
        spatial = image_metrics(half, one)
        self.assertAlmostEqual(spatial["l1"], 0.5)
        self.assertAlmostEqual(spatial["psnr"], 6.0205999, places=5)
        temporal = temporal_metrics(half, zero, one, zero)
        self.assertAlmostEqual(temporal["adjacent_change_l1"], 0.5)
        self.assertAlmostEqual(
            temporal["gt_temporal_difference_error_l1"], 0.5
        )

    def test_evaluation_resets_at_sequence_boundaries_and_writes_triptychs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _paired_manifest(root, lengths=(2, 2))
            records = load_sequence_manifest(manifest, root=root)
            output = root / "evaluation"
            output.mkdir()
            backend = FakeEvaluationBackend()
            summary = evaluate_sequences(
                records,
                backend,
                device=torch.device("cpu"),
                output_dir=output,
                max_visuals=2,
                visual_panel_width=32,
                checkpoint_metadata={"kind": "fake"},
            )
            self.assertEqual(summary["frame_count"], 4)
            self.assertEqual(summary["sequence_count"], 2)
            self.assertEqual(summary["temporal_pair_count"], 2)
            self.assertEqual(backend.reset_count, 2)
            self.assertEqual([call[1] for call in backend.calls], [0, 1, 0, 1])
            self.assertEqual(summary["sources"], ["raw", "turtle", "gt"])
            self.assertIsNone(
                summary["performance"]["peak_cuda_memory_allocated_bytes"]
            )
            self.assertEqual(len(list((output / "visuals").glob("*.png"))), 2)

    def test_nonempty_evaluation_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            output.mkdir()
            (output / "metrics.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_output_directory(output)


if __name__ == "__main__":
    unittest.main()
