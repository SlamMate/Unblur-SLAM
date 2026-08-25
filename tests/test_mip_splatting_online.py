#!/usr/bin/env python3
"""CPU contracts for the online Mip-Splatting scale/opacity filter."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _Camera:
    deblur_fail = False
    R = torch.eye(3)
    T = torch.zeros(3)
    fx = 50.0
    fy = 40.0
    cx = 50.0
    cy = 40.0
    image_width = 100
    image_height = 80


def _cpu_model():
    from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel

    model = object.__new__(GaussianModel)
    model._xyz = torch.tensor(
        [[0.0, 0.0, 1.0], [1.2, 0.0, 2.0], [4.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    model._scaling = torch.zeros((3, 3), dtype=torch.float32)
    model._opacity = torch.zeros((3, 1), dtype=torch.float32)
    model.scaling_activation = torch.exp
    model.opacity_activation = torch.sigmoid
    model.filter_3D = torch.empty((0, 1), dtype=torch.float32)
    model._mip_filter_cameras = ()
    model._mip_filter_kernel_variance = 0.2
    model._mip_splatting_enabled = False
    return model


class OnlineMipSplattingTests(unittest.TestCase):
    def test_official_3d_filter_and_energy_compensation(self):
        model = _cpu_model()
        model.configure_mip_splatting([_Camera()], enabled=True, refresh=True)
        expected = torch.tensor(
            [[math.sqrt(0.2) / 50.0], [2.0 * math.sqrt(0.2) / 50.0],
             [2.0 * math.sqrt(0.2) / 50.0]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(model.filter_3D, expected)
        filtered_scale = model.get_scaling_with_3D_filter
        torch.testing.assert_close(
            filtered_scale,
            torch.sqrt(torch.ones((3, 3)) + expected.square()),
        )
        coefficient = torch.sqrt(
            torch.ones(3) / (torch.ones((3, 3)) + expected.square()).prod(dim=1)
        )
        torch.testing.assert_close(
            model.get_opacity_with_3D_filter,
            0.5 * coefficient.unsqueeze(1),
        )

    def test_topology_invalidation_recomputes_and_invalid_inputs_fail(self):
        model = _cpu_model()
        model.configure_mip_splatting([_Camera()], enabled=True, refresh=True)
        model.invalidate_mip_filter()
        self.assertEqual(tuple(model.filter_3D.shape), (0, 1))
        self.assertEqual(tuple(model.get_scaling_with_3D_filter.shape), (3, 3))
        with self.assertRaisesRegex(ValueError, "positive"):
            model.configure_mip_splatting(
                [_Camera()], enabled=True, kernel_variance=0.0
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            model.configure_mip_splatting([], enabled=True)

    def test_renderer_couples_3d_filter_to_2d_antialiasing(self):
        source = (
            ROOT / "thirdparty/gaussian_splatting/gaussian_renderer/__init__.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(source.count("antialiasing=_mip_splatting_enabled(pc)"), 2)
        self.assertEqual(source.count("get_scaling_with_3D_filter"), 4)
        self.assertEqual(source.count("get_opacity_with_3D_filter"), 2)


if __name__ == "__main__":
    unittest.main()
