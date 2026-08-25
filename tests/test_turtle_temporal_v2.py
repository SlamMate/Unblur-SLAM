#!/usr/bin/env python3
"""CPU contracts for TURTLE's strictly causal temporal-order v2 objective."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_turtle_streaming import (  # noqa: E402
    _replay_current_from_past,
    cyclically_shuffled_past_indices,
    train_sequence_full_bptt,
)


class CountingAdamW(torch.optim.AdamW):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure=closure)


class OrderSensitiveTurtle(torch.nn.Module):
    """Tiny recurrent model with the official sparse eight-slot cache API."""

    use_both_input = False

    def __init__(self):
        super().__init__()
        self.history_scale = torch.nn.Parameter(torch.tensor(0.1))
        self.observed_current_means: list[float] = []

    def forward(self, pair, k_cache=None, v_cache=None):
        current = pair[:, 1]
        self.observed_current_means.append(float(current.detach().mean()))
        previous_state = (
            current.new_zeros(()) if k_cache is None else k_cache[3].mean()
        )
        # The asymmetric recurrence makes [a,b] and [b,a] distinguishable.
        state = 0.25 * previous_state + current.mean()
        restored = current + self.history_scale * previous_state
        marker = state.reshape(1)
        k_new = [None, None, None] + [marker.clone() for _ in range(5)]
        v_new = [None, None, None] + [marker.clone() for _ in range(5)]
        return restored, k_new, v_new


def _sequence(values: tuple[float, ...]) -> torch.Tensor:
    return torch.stack([torch.full((3, 8, 8), value) for value in values])


class TurtleTemporalV2Test(unittest.TestCase):
    def test_cyclic_shuffle_is_complete_past_only_and_nonidentity(self) -> None:
        self.assertEqual(cyclically_shuffled_past_indices(2), (1, 0))
        self.assertEqual(
            cyclically_shuffled_past_indices(5), (1, 2, 3, 4, 0)
        )
        with self.assertRaises(ValueError):
            cyclically_shuffled_past_indices(1)

    def test_replay_rejects_current_future_duplicates_and_missing_past(self) -> None:
        model = OrderSensitiveTurtle()
        blurry = _sequence((0.1, 0.2, 0.3, 0.4))
        for invalid in ((0, 2), (0, 0), (0,), (0, 1, 3)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _replay_current_from_past(
                    model,
                    blurry,
                    anchor_index=2,
                    past_indices=invalid,
                    device=torch.device("cpu"),
                )

    def test_v2_full_record_bptt_adds_order_signal_without_future_frames(
        self,
    ) -> None:
        model = OrderSensitiveTurtle()
        optimizer = CountingAdamW(model.parameters(), lr=0.0)
        blurry = _sequence((0.1, 0.2, 0.3))
        # Ordered history state at frame 2 is 0.225, making this the ordered target.
        sharp = blurry.clone()
        sharp[2] = sharp[2] + 0.1 * 0.225

        row = train_sequence_full_bptt(
            model,
            blurry,
            sharp,
            optimizer,
            device=torch.device("cpu"),
            fft_weight=0.0,
            temporal_delta_weight=0.1,
            order_contrast_weight=0.5,
            order_contrast_margin=5.0e-5,
            loss_start_frame=1,
        )

        self.assertEqual(optimizer.step_count, 1)
        self.assertEqual(row["frames"], 3)
        self.assertEqual(row["supervised_frames"], 2)
        self.assertEqual(row["order_anchor_index"], 2)
        self.assertGreater(row["temporal_delta_l1"], 0.0)
        self.assertGreater(row["shuffled_minus_ordered_anchor_l1"], 0.0)
        # First the complete ordered stream, then only [past1,past0,current2].
        observed = model.observed_current_means
        expected = [0.1, 0.2, 0.3, 0.2, 0.1, 0.3]
        self.assertEqual(len(observed), len(expected))
        for actual, wanted in zip(observed, expected):
            self.assertAlmostEqual(actual, wanted, places=6)

    def test_v2_order_objective_rejects_too_short_records(self) -> None:
        model = OrderSensitiveTurtle()
        optimizer = CountingAdamW(model.parameters(), lr=0.0)
        blurry = _sequence((0.1, 0.2))
        with self.assertRaisesRegex(ValueError, "at least three"):
            train_sequence_full_bptt(
                model,
                blurry,
                blurry,
                optimizer,
                device=torch.device("cpu"),
                order_contrast_weight=0.5,
                loss_start_frame=1,
            )


if __name__ == "__main__":
    unittest.main()
