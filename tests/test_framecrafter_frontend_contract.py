"""CPU contracts for retaining augmented/recovered Frontend keyframes."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tracker import _preserve_latest_keyframe, _should_map_full_warmup
from thirdparty.glorie_slam.frontend import (
    Frontend,
    should_cull_latest_keyframe,
)
from thirdparty.glorie_slam.motion_filter import MotionFilter


class _Graph:
    def __init__(self):
        self.corr = None
        self.ii = torch.tensor([0])
        self.updates = []
        self.removed = []

    def add_proximity_factors(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        self.updates.append(kwargs.get("opt_type"))

    def rm_keyframe(self, index):
        self.removed.append(index)


class _Video:
    def __init__(self):
        self.counter = SimpleNamespace(value=4)
        self.poses = torch.arange(35, dtype=torch.float32).reshape(5, 7)
        self.disps = torch.ones(5, 2, 2)
        self.dirty = None

    def distance(self, *args, **kwargs):
        return torch.tensor(0.1)

    def get_lock(self):
        return nullcontext()

    def set_dirty(self, start, end):
        self.dirty = (int(start), int(end))


class _SyntheticDepthStream:
    def frame_info(self, index):
        return {"synthetic": True, "source_index": -1}

    def __getitem__(self, index):
        return index, None, torch.full((2, 3), 7.0), None, None


def _frontend_for_update():
    frontend = Frontend.__new__(Frontend)
    frontend.t1 = 2
    frontend.graph = _Graph()
    frontend.video = _Video()
    frontend.max_age = 10
    frontend.frontend_window = 8
    frontend.frontend_radius = 2
    frontend.frontend_nms = 1
    frontend.frontend_thresh = 16.0
    frontend.beta = 0.3
    frontend.iters1 = 1
    frontend.iters2 = 2
    frontend.keyframe_thresh = 1.0
    frontend.enable_loop = False
    return frontend


class FrameCrafterFrontendContractTest(unittest.TestCase):
    def test_preserved_latest_skips_cull_but_runs_second_stage_ba(self):
        frontend = _frontend_for_update()

        with mock.patch("torch.cuda.empty_cache"):
            frontend._Frontend__update(preserve_latest=True)

        self.assertEqual(frontend.graph.removed, [])
        self.assertEqual(frontend.video.counter.value, 4)
        self.assertEqual(frontend.t1, 3)
        self.assertEqual(len(frontend.graph.updates), 3)
        self.assertEqual(frontend.video.dirty, (0, 3))

    def test_default_nearby_keyframe_still_culls(self):
        self.assertTrue(should_cull_latest_keyframe(0.1, 1.0))
        self.assertFalse(
            should_cull_latest_keyframe(0.1, 1.0, preserve_latest=True)
        )

    def test_preserve_contract_is_explicit_and_requires_append(self):
        synthetic = {
            "synthetic": True,
            "streaming_replaced": False,
            "motion_keyframe": False,
            "appended": True,
        }
        recovered = {
            "synthetic": False,
            "streaming_replaced": True,
            "motion_keyframe": True,
            "appended": True,
        }
        rejected = {**recovered, "appended": False}
        anchor_only = {**recovered, "motion_keyframe": False}
        evssm_fallback = {
            **recovered,
            "streaming_replaced": False,
            "streaming_evssm_fallback": True,
        }
        tum_tracking_anchor = {
            "synthetic": False,
            "streaming_replaced": False,
            "motion_keyframe": False,
            "tracking_anchor": True,
            "appended": True,
        }

        self.assertTrue(_preserve_latest_keyframe(synthetic))
        self.assertTrue(_preserve_latest_keyframe(recovered))
        self.assertFalse(_preserve_latest_keyframe(rejected))
        self.assertFalse(_preserve_latest_keyframe(anchor_only))
        # A causal-gate fallback follows the legacy single-frame EVSSM
        # selection policy; it is not an extra causal motion recovery.
        self.assertFalse(_preserve_latest_keyframe(evssm_fallback))
        self.assertTrue(_preserve_latest_keyframe(tum_tracking_anchor))

    def test_pending_preserved_observation_forces_warmup_replay(self):
        synthetic_pending = _preserve_latest_keyframe(
            {
                "synthetic": True,
                "streaming_replaced": False,
                "motion_keyframe": False,
                "appended": True,
            }
        )
        recovered_pending = _preserve_latest_keyframe(
            {
                "synthetic": False,
                "streaming_replaced": True,
                "motion_keyframe": True,
                "appended": True,
            }
        )
        self.assertTrue(_should_map_full_warmup(False, False, synthetic_pending))
        self.assertTrue(_should_map_full_warmup(False, False, recovered_pending))
        self.assertTrue(_should_map_full_warmup(False, True, False))
        self.assertFalse(_should_map_full_warmup(False, False, False))
        self.assertFalse(_should_map_full_warmup(True, True, True))

    def test_synthetic_minus_one_source_index_uses_generated_depth(self):
        motion_filter = MotionFilter.__new__(MotionFilter)
        depth = motion_filter._mono_depth_for_frame(
            4,
            torch.zeros(1, 3, 2, 3),
            _SyntheticDepthStream(),
        )
        self.assertTrue(torch.equal(depth, torch.full((2, 3), 7.0)))


if __name__ == "__main__":
    unittest.main()
