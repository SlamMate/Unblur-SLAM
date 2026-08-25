#!/usr/bin/env python3
"""CPU contracts for the paired TURTLE precision benchmark."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_turtle_runtime import _load_frames, _percentile  # noqa: E402


class TurtleRuntimeBenchmarkContractTest(unittest.TestCase):
    def test_percentile_is_linear_and_bounded(self):
        self.assertEqual(_percentile([1.0, 2.0, 3.0], 0.0), 1.0)
        self.assertEqual(_percentile([1.0, 2.0, 3.0], 1.0), 3.0)
        self.assertEqual(_percentile([1.0, 2.0, 3.0], 0.5), 2.0)

    def test_manifest_loader_binds_rgb_bytes_and_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "frame.png"
            Image.fromarray(np.zeros((384, 512, 3), dtype=np.uint8)).save(image_path)
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "frames": [
                            {
                                "source_index": 7,
                                "output": {
                                    "path": str(image_path),
                                    "sha256": digest,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            frames, provenance = _load_frames(manifest_path, 1)
            self.assertEqual(tuple(frames[0].shape), (1, 3, 384, 512))
            self.assertEqual(provenance[0]["source_index"], 7)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["frames"][0]["output"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_frames(manifest_path, 1)


if __name__ == "__main__":
    unittest.main()
