import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "report_turtle_resplat_smoke",
    ROOT / "scripts" / "report_turtle_resplat_smoke.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _fake_turtle_manifest(tmp_path: Path, *, camera_k=None):
    image_path = tmp_path / "turtle.png"
    image = Image.fromarray(np.zeros((384, 512, 3), dtype=np.uint8), mode="RGB")
    image.save(image_path)
    png_sha = MODULE.sha256_file(image_path)
    pixel_sha = MODULE.pixels_sha256(image)
    raw_sha = "a" * 64
    frames = []
    for index in MODULE.CLEAR_42:
        frames.append(
            {
                "provider": MODULE.EXPECTED_TURTLE_PROVIDER,
                "source_index": index,
                "input": {
                    "sha256": raw_sha,
                    "preprocessed_pixel_sha256": pixel_sha,
                },
                "output": {
                    "path": str(image_path),
                    "sha256": png_sha,
                    "pixel_sha256": pixel_sha,
                },
                "output_sha256": png_sha,
            }
        )
    steps = [
        {
            "step_index": index,
            "source_index": index,
            "cache_present_before": index > 0,
            "cache_present_after": True,
            "cache_update_ordinal": index + 1,
            "reset_count": 1,
            "latency_ms": 10.0,
        }
        for index in range(2765)
    ]
    manifest = {
        "schema": MODULE.TURTLE_SCHEMA,
        "camera": {
            "width": 512,
            "height": 384,
            "resize_before_crop_width": 528,
            "resize_before_crop_height": 400,
            "crop_edges": {"left": 8, "right": 8, "top": 8, "bottom": 8},
            "K": camera_k if camera_k is not None else [list(row) for row in MODULE.TRACKER_K],
        },
        "turtle": {
            "checkpoint": {"sha256": MODULE.EXPECTED_TURTLE_CHECKPOINT_SHA256}
        },
        "safety": {
            "ground_truth_images_used": False,
            "ground_truth_poses_used": False,
            "depth_used": False,
        },
        "stream": {
            "processed_source_indices": list(range(2765)),
            "processed_count": 2765,
            "step_count": 2765,
            "cache_updates": 2765,
            "reset_count": 1,
            "first_pair": "self",
            "persistent_kv": True,
            "steps": steps,
        },
        "selection": {
            "emitted_source_indices": list(MODULE.CLEAR_42),
            "emitted_count": 42,
        },
        "performance": {
            "latency_ms": {"mean": 10, "median": 10, "p95": 10, "max": 10},
            "stream_wall_seconds": 27.65,
            "peak_cuda_memory_allocated_bytes": 1,
        },
        "frames": frames,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    old_manifest = {
        "frames": [
            {"source_index": index, "raw_sha256": raw_sha} for index in MODULE.CLEAR_42
        ]
    }
    references = {index: image.copy() for index in MODULE.CLEAR_42}
    return manifest_path, old_manifest, references


class TurtleResplatReportContractTests(unittest.TestCase):
    def test_pre_registered_sets_and_tracker_k_are_frozen(self):
        self.assertEqual(len(MODULE.CLEAR_42), 42)
        self.assertEqual(len(MODULE.EVSSM_26), 26)
        self.assertLess(set(MODULE.EVSSM_26), set(MODULE.CLEAR_42))
        self.assertEqual(MODULE.FIXED_FIVE, (49, 483, 1342, 2055, 2764))
        self.assertLessEqual(set(MODULE.FIXED_FIVE), set(MODULE.EVSSM_26))
        expected = np.asarray(
            [
                [520.9 * 528 / 640, 0, 325.1 * 528 / 640 - 8],
                [0, 521.0 * 400 / 480, 249.7 * 400 / 480 - 8],
                [0, 0, 1],
            ]
        )
        self.assertTrue(
            np.allclose(np.asarray(MODULE.TRACKER_K), expected, rtol=0, atol=1e-12)
        )

    def test_psnr_ssim_exact_reference_control_and_perturbation(self):
        reference = Image.fromarray(
            np.full((32, 32, 3), 100, dtype=np.uint8), mode="RGB"
        )
        exact = MODULE.psnr_ssim(reference, reference.copy())
        self.assertEqual(
            exact,
            {"mse": 0.0, "psnr_db": "Infinity", "ssim": 1.0, "exact_match": True},
        )
        candidate = Image.fromarray(
            np.full((32, 32, 3), 101, dtype=np.uint8), mode="RGB"
        )
        changed = MODULE.psnr_ssim(reference, candidate)
        self.assertFalse(changed["exact_match"])
        self.assertGreater(changed["mse"], 0)
        self.assertIsInstance(changed["psnr_db"], float)
        self.assertLess(changed["ssim"], 1)

    def test_unwarped_temporal_reference_control_is_zero_error(self):
        previous = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), mode="RGB")
        current = Image.fromarray(
            np.full((16, 16, 3), 25, dtype=np.uint8), mode="RGB"
        )
        values = MODULE._temporal_unwarped(previous, current, previous, current)
        self.assertAlmostEqual(values["adjacent_change_l1"], 25 / 255)
        self.assertEqual(values["reference_temporal_difference_error_l1"], 0.0)

    def test_native_geometry_update_statistics_are_same_index_diagnostics(self):
        initial = np.asarray([[0, 0, 0], [1, 2, 3], [4, 5, 6]], dtype=np.float64)
        refined = initial.copy()
        refined[1, 0] += 1.0
        refined[2, 2] += 6.0
        values = MODULE._geometry_stats_from_arrays(initial, refined)
        self.assertTrue(values["same_index_diagnostic_only"])
        self.assertTrue(values["all_positions_finite"])
        self.assertEqual(values["vertex_count_init0"], 3)
        self.assertEqual(values["nonzero_displacement_count"], 2)
        displacement = values["position_displacement_m_assuming_scene_scale"]
        self.assertEqual(displacement["max"], 6.0)
        self.assertEqual(displacement["count_gt_1m"], 1)
        self.assertEqual(displacement["count_gt_5m"], 1)

    def test_turtle_manifest_contract_accepts_tracker_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, old, references = _fake_turtle_manifest(Path(temporary))
            images, latency, record = MODULE.load_turtle_artifact(path, old, references)
        self.assertEqual(set(images), set(MODULE.CLEAR_42))
        self.assertEqual(latency["steady_state_p95_ms"], 10.0)
        self.assertTrue(latency["thirty_fps_p95_feasible"])
        self.assertTrue(
            record["provenance"]["preprocessed_inputs_pixel_identical_to_clear_reference"]
        )

    def test_turtle_manifest_contract_rejects_direct_resize_k(self):
        direct_k = [[416.72, 0, 260.08], [0, 416.8, 199.76], [0, 0, 1]]
        with tempfile.TemporaryDirectory() as temporary:
            path, old, references = _fake_turtle_manifest(
                Path(temporary), camera_k=direct_k
            )
            with self.assertRaisesRegex(MODULE.AuditError, "exact fr2 tracker K"):
                MODULE.load_turtle_artifact(path, old, references)

    def test_acceptance_doc_forbids_400_step_as_26k(self):
        text = (
            ROOT / "docs" / "TURTLE_RESPLAT_SMOKE_ACCEPTANCE_ZH.md"
        ).read_text(encoding="utf-8")
        self.assertIn("400-step smoke 等价于正式 26K baseline", text)
        self.assertIn("formal_26k_result_present=false", text)
        self.assertNotIn("HISTORICAL", text)  # prose, not a hidden machine-only caveat


if __name__ == "__main__":
    unittest.main()
