#!/usr/bin/env python3
"""Build role-aware M-blurry-to-N-sharp FrameCrafter LoRA samples.

Input JSONL example (one object per sequence)::

    {"sequence":"turtle",
     "blurry":["blur/000.png", "blur/001.png", "..."],
     "sharp":["sharp/000.png", "sharp/001.png", "..."],
     "evssm":["evssm/000.png", "evssm/001.png", "..."],
     "trajectory_path":"first_pass/traj.npz",
     "trajectory_key":"traj_est_not_align",
     "pose_provenance":{"kind":"estimated_unaligned", "source":"first-pass DROID"},
     "K":[[600,0,320],[0,600,240],[0,0,1]],
     "camera_convention":"opencv"}

Ground-truth, reference, or aligned trajectories are rejected.  Every copied
asset and camera source is content-addressed in the resulting samples file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.framecrafter_lora_dataset import build_paired_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Paired sequence JSONL.")
    parser.add_argument("--output-root", type=Path, required=True, help="New role-aware dataset directory.")
    parser.add_argument("--num-input-frames", type=int, default=6, help="M blurry/EVSSM context views.")
    parser.add_argument("--num-output-frames", type=int, default=1, help="N sharp target views.")
    parser.add_argument("--sample-stride", type=int, default=1, help="Stride between target groups.")
    parser.add_argument("--max-samples-per-sequence", type=int, default=None)
    parser.add_argument(
        "--context-mode",
        choices=("raw", "evssm", "hybrid"),
        default="raw",
        help="Choose raw blurry contexts, EVSSM contexts, or a deterministic mixture.",
    )
    parser.add_argument(
        "--hybrid-evssm-fraction",
        type=float,
        default=0.5,
        help="Fraction of nearest context cameras using EVSSM in hybrid mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_paired_dataset(
        args.manifest,
        args.output_root,
        num_input_frames=args.num_input_frames,
        num_output_frames=args.num_output_frames,
        sample_stride=args.sample_stride,
        max_samples_per_sequence=args.max_samples_per_sequence,
        context_mode=args.context_mode,
        hybrid_evssm_fraction=args.hybrid_evssm_fraction,
    )
    print(
        json.dumps(
            {
                "status": "built",
                "output_root": str(summary.output_root),
                "sequence_count": summary.sequence_count,
                "sample_count": summary.sample_count,
                "M": summary.num_input_frames,
                "N": summary.num_output_frames,
                "samples_sha256": summary.samples_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
