#!/usr/bin/env python3
"""Build one immutable two-machine contract from a global FrameCrafter plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.framecrafter_sharding import (  # noqa: E402
    build_shard_contract,
    canonical_sha256,
    validate_shard_runtime_identity,
    write_shard_contract,
)


def _json_value(value: str, label: str) -> Any:
    path = Path(value).expanduser()
    raw = path.read_text(encoding="utf-8") if path.is_file() else value
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} must be JSON text or a JSON file") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument(
        "--model-identity",
        default=None,
        help=(
            "Optional JSON/file assertion whose canonical hash must equal the "
            "actual model-artifact identity recorded by --plan-report."
        ),
    )
    parser.add_argument(
        "--config-identity",
        default=None,
        help=(
            "Optional JSON/file assertion whose canonical hash must equal the "
            "actual semantic config identity recorded by --plan-report."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.plan_report.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "unblur_slam.framecrafter_preprocess_report.v1":
        raise ValueError("--plan-report is not a FrameCrafter preprocess report")
    batches, planned = report.get("generation_batches"), report.get("planned")
    if not isinstance(batches, list) or not isinstance(planned, list):
        raise ValueError("global plan report lacks generation_batches/planned")
    runtime_identity = report.get("shard_runtime_identity")
    validate_shard_runtime_identity(runtime_identity)
    if args.model_identity is not None:
        model_identity = _json_value(args.model_identity, "--model-identity")
        if canonical_sha256(model_identity) != runtime_identity[
            "model_artifact_identity_sha256"
        ]:
            raise ValueError(
                "--model-identity differs from actual plan-report model artifacts"
            )
    if args.config_identity is not None:
        config_identity = _json_value(args.config_identity, "--config-identity")
        if canonical_sha256(config_identity) != runtime_identity[
            "semantic_config_sha256"
        ]:
            raise ValueError(
                "--config-identity differs from actual plan-report semantic config"
            )
    contract = build_shard_contract(
        batches,
        planned,
        shard_count=args.shard_count,
        runtime_identity=runtime_identity,
    )
    output = write_shard_contract(contract, args.output)
    print(
        json.dumps(
            {
                "contract": str(output),
                "experiment_signature": contract["experiment_signature"],
                "shard_count": contract["shard_count"],
                "assignment_table_sha256": contract["assignment_table_sha256"],
                "canonical_preprocess_signature": runtime_identity[
                    "canonical_preprocess_signature"
                ],
                "model_artifact_identity_sha256": runtime_identity[
                    "model_artifact_identity_sha256"
                ],
                "semantic_config_sha256": runtime_identity[
                    "semantic_config_sha256"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
