#!/usr/bin/env python3
"""CPU contracts for the pinned, incremental TURTLE frontend."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.turtle_backend import (  # noqa: E402
    FINETUNED_CHECKPOINT_FORMAT,
    PINNED_TURTLE_CHECKPOINT_SHA256,
    PINNED_TURTLE_COMMIT,
    PINNED_TURTLE_ARCH_SHA256,
    PINNED_TURTLE_CONFIG_SHA256,
    TURTLE_CACHE_CONTRACT,
    TurtleStreamingBackend,
    build_turtle_model,
    load_turtle_model,
    normalize_turtle_inference_precision,
    validate_turtle_checkpoint_payload,
    validate_turtle_artifacts,
)


TURTLE_REPO = Path("/srv/szha0669/unblur-slam/external/TURTLE")
TURTLE_CONFIG = TURTLE_REPO / "options/Turtle_Deblur_Gopro.yml"
TURTLE_CHECKPOINT = Path(
    "/srv/szha0669/unblur-slam/pretrained/turtle/GoPro_Deblur.pth"
)


class FakeOfficialTurtle(torch.nn.Module):
    """Small model exposing the same two-frame + eight-cache API."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, pair, k_cache=None, v_cache=None):
        prior = 0.0 if k_cache is None else float(k_cache[0].flatten()[0])
        self.calls.append(
            {
                "pair": pair.detach().clone(),
                "had_k": k_cache is not None,
                "had_v": v_cache is not None,
                "prior": prior,
            }
        )
        current = pair[:, 1]
        marker = torch.tensor(
            [float(len(self.calls))], dtype=current.dtype, device=current.device
        )
        cache = [marker.clone() for _ in range(8)]
        # Depend on the incoming cache so the test proves state was forwarded.
        output = current + prior * 0.01
        return output, cache, [value.clone() for value in cache]


class TurtleStreamingContractTest(unittest.TestCase):
    def test_one_incremental_call_per_frame_and_cache_forwarding(self):
        model = FakeOfficialTurtle()
        backend = TurtleStreamingBackend(model, device="cpu")
        first = torch.full((1, 3, 8, 12), 0.1)
        second = torch.full((1, 3, 8, 12), 0.2)
        third = torch.full((1, 3, 8, 12), 0.3)

        out1 = backend(first, timestamp=0)
        out2 = backend(second, timestamp=1)
        out3 = backend(third, timestamp=2)

        self.assertEqual(len(model.calls), 3)
        self.assertEqual(backend.cache_updates, 3)
        self.assertFalse(model.calls[0]["had_k"])
        self.assertTrue(model.calls[1]["had_k"])
        self.assertTrue(model.calls[2]["had_v"])
        self.assertEqual(model.calls[1]["prior"], 1.0)
        self.assertEqual(model.calls[2]["prior"], 2.0)
        self.assertEqual(tuple(model.calls[0]["pair"].shape), (1, 2, 3, 8, 12))
        self.assertTrue(torch.equal(model.calls[0]["pair"][:, 0], first))
        self.assertTrue(torch.equal(model.calls[0]["pair"][:, 1], first))
        self.assertTrue(torch.equal(model.calls[1]["pair"][:, 0], first))
        self.assertTrue(torch.equal(model.calls[1]["pair"][:, 1], second))
        self.assertTrue(torch.allclose(out1, first))
        self.assertTrue(torch.allclose(out2, second + 0.01))
        self.assertTrue(torch.allclose(out3, third + 0.02))

    def test_resolution_or_time_reversal_resets_causal_state(self):
        model = FakeOfficialTurtle()
        backend = TurtleStreamingBackend(model, device="cpu")
        backend(torch.zeros(1, 3, 8, 8), timestamp=10)
        backend(torch.zeros(1, 3, 12, 8), timestamp=11)
        self.assertFalse(model.calls[1]["had_k"])
        self.assertEqual(backend.reset_count, 1)

        backend(torch.zeros(1, 3, 12, 8), timestamp=9)
        self.assertFalse(model.calls[2]["had_k"])
        self.assertEqual(backend.reset_count, 2)

    def test_invalid_stream_inputs_fail_closed(self):
        backend = TurtleStreamingBackend(FakeOfficialTurtle(), device="cpu")
        with self.assertRaises(ValueError):
            backend(torch.zeros(2, 3, 8, 8))
        with self.assertRaises(ValueError):
            backend(torch.zeros(1, 1, 8, 8))
        bad = torch.zeros(1, 3, 8, 8)
        bad[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            backend(bad)

    def test_inference_precision_is_explicit_and_cuda_only_for_fp16(self):
        self.assertEqual(normalize_turtle_inference_precision(None), "fp32")
        self.assertEqual(normalize_turtle_inference_precision("FP16"), "fp16")
        with self.assertRaises(ValueError):
            normalize_turtle_inference_precision("auto")
        with self.assertRaises(ValueError):
            TurtleStreamingBackend(
                FakeOfficialTurtle(),
                device="cpu",
                inference_precision="fp16",
            )

        backend = TurtleStreamingBackend(
            FakeOfficialTurtle(),
            device="cpu",
            inference_precision="fp32",
        )
        self.assertEqual(backend.state_info()["inference_precision"], "fp32")


class TurtlePipelineWiringTest(unittest.TestCase):
    def test_run_preflight_strict_loads_and_records_provenance(self):
        from run import prepare_or_validate_inputs
        from thirdparty.glorie_slam.config import load_config

        config_path = (
            ROOT
            / "configs/local/fr2_xyz_resplat_smoke/turtle_offline_online_replay.yaml"
        )
        cfg = load_config(config_path, ROOT / "configs/unblur_slam.yaml")
        prepare_or_validate_inputs(cfg, "/tmp/turtle-preflight-does-not-write")
        self.assertEqual(cfg["deblur"]["frontend"], "turtle_streaming")
        self.assertEqual(
            cfg["deblur"]["turtle_checkpoint_metadata"]["kind"],
            "official_gopro",
        )
        self.assertEqual(
            cfg["deblur"]["turtle_checkpoint_sha256"],
            PINNED_TURTLE_CHECKPOINT_SHA256,
        )

        cfg["deblur"]["stream_every_frame"] = False
        with self.assertRaises(ValueError):
            prepare_or_validate_inputs(cfg, "/tmp/turtle-preflight-does-not-write")

    def test_motion_filter_advances_every_frame_before_laplacian_gate(self):
        from thirdparty.glorie_slam.motion_filter import MotionFilter

        motion_filter = MotionFilter.__new__(MotionFilter)
        motion_filter.cfg = {
            "deblur": {
                "stream_every_frame": True,
                "stream_apply_to_tracking": True,
                "stream_min_laplacian_gain": 0.10,
                "stream_replace_sharp": True,
            },
            "verbose": False,
        }
        motion_filter.deblur_backend = object()
        motion_filter.deblur_backend_name = "turtle_streaming"
        calls = []

        def apply(image, stream=None, timestamp=None):
            del stream
            calls.append(timestamp)
            return image + 0.05

        motion_filter.apply_evssm_deblur = apply
        # First frame falls below the gate; second clears it. The backend must
        # still be invoked exactly once for both, otherwise K/V time is broken.
        laplacians = iter((1.0, 1.05, 1.0, 1.20))
        motion_filter._laplacian_value = lambda image: next(laplacians)
        image = torch.zeros(1, 3, 8, 8)

        rejected, candidate1, replaced1, gain1 = motion_filter._streaming_deblur(
            image, 0, None
        )
        accepted, candidate2, replaced2, gain2 = motion_filter._streaming_deblur(
            image, 1, None
        )
        self.assertEqual(calls, [0, 1])
        self.assertFalse(replaced1)
        self.assertTrue(torch.equal(rejected, image))
        self.assertTrue(torch.equal(candidate1, image + 0.05))
        self.assertAlmostEqual(gain1, 0.05)
        self.assertTrue(replaced2)
        self.assertTrue(torch.equal(accepted, candidate2))
        self.assertAlmostEqual(gain2, 0.20)

    def test_tracker_marks_turtle_as_an_active_sharp_frontend(self):
        import src.tracker as tracker_module

        class NoopMotionFilter:
            def __init__(self, *args, **kwargs):
                del args, kwargs

        cfg = {
            "device": "cpu",
            "tracking": {
                "frontend": {"window": 8, "enable_online_ba": False},
                "motion_filter": {"thresh": 2.5},
                "backend": {"ba_freq": 20},
            },
            "mapping": {"every_keyframe": 1},
            "deblur": {"frontend": "turtle_streaming"},
            "fake_sharp": False,
            "sharp_judge": False,
        }
        slam = SimpleNamespace(
            cfg=cfg,
            droid_net=object(),
            video=object(),
            verbose=False,
            only_tracking=False,
            save_dir="/tmp/not-written",
            frontend=object(),
            online_ba=object(),
            printer=object(),
        )
        with mock.patch.object(tracker_module, "MotionFilter", NoopMotionFilter):
            tracker = tracker_module.Tracker(slam, pipe=None)
        self.assertTrue(tracker.fake_sharp)


class TurtlePinnedArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [
            path
            for path in (TURTLE_REPO, TURTLE_CONFIG, TURTLE_CHECKPOINT)
            if not path.exists()
        ]
        if missing:
            raise unittest.SkipTest(f"pinned TURTLE artifacts unavailable: {missing}")

    @staticmethod
    def config():
        return {
            "turtle_repo": str(TURTLE_REPO),
            "turtle_config": str(TURTLE_CONFIG),
            "turtle_checkpoint": str(TURTLE_CHECKPOINT),
            "turtle_repo_commit": PINNED_TURTLE_COMMIT,
            "turtle_checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
        }

    def test_real_gopro_checkpoint_strict_loads_on_cpu(self):
        model, metadata = load_turtle_model(
            TURTLE_REPO,
            TURTLE_CHECKPOINT,
            config=TURTLE_CONFIG,
            device="cpu",
        )
        self.assertFalse(model.training)
        self.assertEqual(len(model.state_dict()), 633)
        self.assertEqual(metadata["turtle_repo_commit"], PINNED_TURTLE_COMMIT)
        self.assertEqual(
            metadata["checkpoint_sha256"], PINNED_TURTLE_CHECKPOINT_SHA256
        )
        self.assertEqual(model.turtle_checkpoint_metadata, metadata)

    def test_configured_hash_cannot_repin_an_unreviewed_checkpoint(self):
        config = self.config()
        config["turtle_checkpoint_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_turtle_artifacts(config)

    def test_finetuned_payload_requires_complete_causal_provenance(self):
        state = {"weight": torch.ones(1)}
        metadata = {
            "format": FINETUNED_CHECKPOINT_FORMAT,
            "base_checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
            "turtle_repo_commit": PINNED_TURTLE_COMMIT,
            "turtle_arch_sha256": PINNED_TURTLE_ARCH_SHA256,
            "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
            "input_domain": "raw",
            "cache_contract": TURTLE_CACHE_CONTRACT,
        }
        normalized, returned = validate_turtle_checkpoint_payload(
            {"state_dict": state, "metadata": metadata},
            checkpoint_sha256="1" * 64,
            expected_checkpoint_sha256="1" * 64,
        )
        self.assertIs(normalized, state)
        self.assertEqual(returned["kind"], "finetuned")
        self.assertEqual(returned["checkpoint_sha256"], "1" * 64)

        bad_metadata = dict(metadata)
        bad_metadata["cache_contract"] = "rolling_window"
        with self.assertRaises(ValueError):
            validate_turtle_checkpoint_payload(
                {"params": state, "metadata": bad_metadata},
                checkpoint_sha256="2" * 64,
                expected_checkpoint_sha256="2" * 64,
            )


if __name__ == "__main__":
    unittest.main()
