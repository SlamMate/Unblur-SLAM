#!/usr/bin/env python3
"""CPU and source contracts for the paired official ReSplat smoke runner."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_paired_official_resplat_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_paired_official_resplat_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class FakeGaussians:
    def __init__(self, marker: float) -> None:
        self.means = torch.tensor([marker])
        self.covariances = torch.tensor([marker + 1.0])
        self.harmonics = torch.tensor([marker + 2.0])
        self.opacities = torch.tensor([marker + 3.0])
        self.scales = torch.tensor([marker + 4.0])
        self.rotations = torch.tensor([marker + 5.0])
        self.rotations_unnorm = torch.tensor([marker + 6.0])


class FakeEncoder:
    def __init__(self) -> None:
        self.initial = FakeGaussians(10.0)
        self.refined = [FakeGaussians(float(index)) for index in range(4)]
        self.forward_calls = 0
        self.update_calls = 0
        self.received_init = None
        self.received_target = None

    def __call__(self, context, **kwargs):
        self.forward_calls += 1
        self.forward_kwargs = kwargs
        return {"gaussians": self.initial, "condition_features": object()}

    def forward_update(
        self, context, target, condition_features, init_gaussians, decoder, context_remain
    ):
        self.update_calls += 1
        self.received_init = init_gaussians
        self.received_target = target
        return {"gaussian": self.refined}


class FakeDecoder:
    def __init__(self) -> None:
        self.gaussians_seen = []

    def forward(
        self, gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode=None
    ):
        self.gaussians_seen.append(gaussians)
        count = extrinsics.shape[1]
        height, width = image_shape
        marker = float(gaussians.means.reshape(-1)[0])
        return SimpleNamespace(
            color=torch.full((1, count, 3, height, width), marker),
            depth=torch.full((1, count, height, width), marker),
        )


def immediate_stage(name, operation):
    return operation()


class PairedOfficialReSplatSmokeTests(unittest.TestCase):
    def _batch(self):
        views = 5
        height, width = 2, 3
        return {
            "context": {"image": torch.zeros(1, 2, 3, height, width)},
            "target": {
                "image": torch.zeros(1, views, 3, height, width),
                "extrinsics": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, views, 1, 1),
                "intrinsics": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, views, 1, 1),
                "near": torch.ones(1, views),
                "far": torch.ones(1, views) * 10,
            },
        }

    def test_explicit_source_selection_preserves_declared_order(self) -> None:
        names = ["00000010.png", "00000020.png", "00000030.png"]
        scene_data = {"image_names": names}
        manifest = {
            "frames": [
                {"source_index": 10, "image_name": names[0]},
                {"source_index": 20, "image_name": names[1]},
                {"source_index": 30, "image_name": names[2]},
            ]
        }
        context, target = RUNNER.select_explicit_source_views(
            scene_data=scene_data,
            scene_manifest=manifest,
            context_source_indices=[30, 10],
            target_source_indices=[20, 30],
            expected_context_count=2,
        )
        self.assertEqual(context, [2, 0])
        self.assertEqual(target, [1, 2])

    def test_explicit_source_selection_rejects_missing_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "absent from scene manifest"):
            RUNNER.select_explicit_source_views(
                scene_data={"image_names": ["00000010.png"]},
                scene_manifest={
                    "frames": [{"source_index": 10, "image_name": "00000010.png"}]
                },
                context_source_indices=[10],
                target_source_indices=[99],
                expected_context_count=1,
            )

    def test_one_encoder_same_init_object_same_targets_and_four_updates(self) -> None:
        encoder = FakeEncoder()
        decoder = FakeDecoder()
        result = RUNNER.run_paired_inference_core(
            torch_module=torch,
            encoder=encoder,
            decoder=decoder,
            batch=self._batch(),
            num_refine=4,
            render_chunk_size=2,
            stage_runner=immediate_stage,
        )

        self.assertEqual(encoder.forward_calls, 1)
        self.assertEqual(encoder.update_calls, 1)
        self.assertIs(encoder.received_init, encoder.initial)
        self.assertNotIn("image", encoder.received_target)
        self.assertIs(result["init_gaussians"], encoder.initial)
        self.assertIs(result["refined_gaussians"], encoder.refined[-1])
        self.assertEqual(tuple(result["init_rendered"].shape), (5, 3, 2, 3))
        self.assertEqual(tuple(result["refined_rendered"].shape), (5, 3, 2, 3))
        self.assertTrue(torch.all(result["init_rendered"] == 10.0))
        self.assertTrue(torch.all(result["refined_rendered"] == 3.0))
        self.assertEqual(result["contract"]["encoder_forward_calls"], 1)
        self.assertTrue(
            result["contract"]["init_object_passed_directly_to_forward_update"]
        )
        self.assertFalse(
            result["contract"]["initial_state_in_place_mutation_detected"]
        )
        self.assertFalse(result["contract"]["target_rgb_passed_to_forward_update"])

    def test_metric_reference_manifest_is_hash_audited_and_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(b"first-reference")
            second.write_bytes(b"second-reference")
            manifest = root / "references.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": RUNNER.METRIC_REFERENCE_SCHEMA,
                        "selection_fixed_before_reference_loading": True,
                        "frames": [
                            {
                                "source_index": 10,
                                "path": str(first),
                                "sha256": RUNNER.sha256_file(first),
                                "included_in_formal_aggregate": True,
                            },
                            {
                                "source_index": 20,
                                "path": str(second),
                                "sha256": RUNNER.sha256_file(second),
                                "included_in_formal_aggregate": False,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            paths, record = RUNNER.load_metric_reference_paths(manifest, [20, 10])
            self.assertEqual(paths, [str(second), str(first)])
            self.assertFalse(record["passed_to_encoder_or_forward_update"])

    def test_rejects_non_four_update_pair(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_refine=4"):
            RUNNER.run_paired_inference_core(
                torch_module=torch,
                encoder=FakeEncoder(),
                decoder=FakeDecoder(),
                batch=self._batch(),
                num_refine=0,
                render_chunk_size=2,
                stage_runner=immediate_stage,
            )

    def test_output_root_is_non_overwriting_and_staged_as_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "paired"
            staging = RUNNER._prepare_staging(destination)
            self.assertEqual(staging.parent, destination.parent)
            self.assertNotEqual(staging, destination)
            staging.rmdir()

            destination.mkdir()
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                RUNNER._prepare_staging(destination)

    def test_source_contract_has_one_encoder_and_one_forward_update_call(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_paired_inference_core"
        )
        encoder_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "encoder"
        ]
        update_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "forward_update"
        ]
        self.assertEqual(len(encoder_calls), 1)
        self.assertEqual(len(update_calls), 1)
        self.assertTrue(
            any(
                isinstance(argument, ast.Name) and argument.id == "init_gaussians"
                for argument in update_calls[0].args
            )
        )

        build_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "build_model"
        ]
        self.assertEqual(len(build_calls), 1)
        strict_keyword = next(
            keyword
            for keyword in build_calls[0].keywords
            if keyword.arg == "no_strict_load"
        )
        self.assertIsInstance(strict_keyword.value, ast.Constant)
        self.assertIs(strict_keyword.value.value, False)

    def test_official_import_filters_competing_regular_src_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            official = root / "resplat"
            competing = root / "unblur"
            (official / "scripts").mkdir(parents=True)
            (official / "src" / "misc").mkdir(parents=True)
            (official / "src" / "misc" / "value.py").write_text(
                "IDENTITY = 'official'\n", encoding="utf-8"
            )
            (official / "scripts" / "infer_colmap.py").write_text(
                "from src.misc.value import IDENTITY\n", encoding="utf-8"
            )
            (competing / "src").mkdir(parents=True)
            (competing / "src" / "__init__.py").write_text("", encoding="utf-8")

            original_path = list(sys.path)
            original_src = {
                name: module
                for name, module in sys.modules.items()
                if name == "src" or name.startswith("src.")
            }
            try:
                for name in list(original_src):
                    sys.modules.pop(name, None)
                sys.path[:] = [str(competing), *original_path]
                module = RUNNER.import_official_infer(official)
                self.assertEqual(module.IDENTITY, "official")
                self.assertNotIn(str(competing), sys.path)
            finally:
                for name in list(sys.modules):
                    if name == "src" or name.startswith("src."):
                        sys.modules.pop(name, None)
                sys.modules.update(original_src)
                sys.path[:] = original_path

if __name__ == "__main__":
    unittest.main()
