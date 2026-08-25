#!/usr/bin/env python3
"""Run the one-step DDP probe and exact resume under one dual-GPU lease."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_turtle_unblur_stable import CHECKPOINT_FORMAT, sha256_file


PYTHON = Path("/srv/szha0669/unblur-slam/env/bin/python")
TRAINER = ROOT / "scripts/train_turtle_unblur_stable.py"


def _torchrun(contract: Path, contract_sha256: str, world_size: int, *extra: str) -> list[str]:
    return [
        str(PYTHON), "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={world_size}", str(TRAINER),
        "--contract", str(contract), "--contract-sha256", contract_sha256,
        "--device", "cuda", *extra,
    ]


def run(contract: Path, contract_sha256: str) -> None:
    contract = contract.expanduser().resolve()
    if sha256_file(contract) != contract_sha256.lower():
        raise ValueError("contract SHA mismatch")
    payload = json.loads(contract.read_text(encoding="utf-8"))
    world_size = int(payload["ddp_world_size"])
    output = Path(str(payload["output_dir"])).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"fresh DDP output already exists: {output}")
    subprocess.run(
        _torchrun(contract, contract_sha256, world_size, "--stop-after-step", "1"),
        cwd=ROOT, check=True,
    )
    checkpoint = output / "checkpoints/step_000001.pth"
    digest_path = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    if not checkpoint.is_file() or not digest_path.is_file():
        raise RuntimeError("one-step probe did not publish its checkpoint and digest")
    digest_token = digest_path.read_text(encoding="ascii").split()[0]
    if sha256_file(checkpoint) != digest_token:
        raise RuntimeError("one-step probe checkpoint digest mismatch")
    observed = torch.load(checkpoint, map_location="cpu")
    distributed = observed.get("distributed", {})
    if (
        observed.get("format") != CHECKPOINT_FORMAT
        or int(observed.get("step", -1)) != 1
        or int(distributed.get("world_size", -1)) != world_size
        or len(distributed.get("rank_states", [])) != world_size
        or not isinstance(observed.get("optimizer"), dict)
    ):
        raise RuntimeError("one-step DDP checkpoint contract failed")
    subprocess.run(
        _torchrun(contract, contract_sha256, world_size, "--resume", str(checkpoint)),
        cwd=ROOT, check=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.contract, args.contract_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
