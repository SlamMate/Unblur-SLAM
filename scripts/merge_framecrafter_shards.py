#!/usr/bin/env python3
"""Fail-closed merger for independently generated FrameCrafter shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.framecrafter_sharding import merge_shard_envelopes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a complete set of FrameCrafter shard envelopes. Raw preprocess "
            "reports are rejected because they cannot prove completeness or a "
            "shared model/configuration."
        )
    )
    parser.add_argument(
        "envelopes",
        type=Path,
        nargs="+",
        help="one validated shard-envelope JSON per shard index",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, manifest_path = merge_shard_envelopes(
        args.envelopes, args.output_dir
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
