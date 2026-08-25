#!/usr/bin/env python3
"""Build one immutable train-only manifest from paired Unblur-SLAM folders."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Tuple

from PIL import Image


SCHEMA = "unblur_slam.paired_image_train.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_pair(value: str) -> Tuple[str, Path, Path]:
    parts = value.split("::")
    if len(parts) != 3 or not parts[0]:
        raise argparse.ArgumentTypeError("--pair must be NAME::INPUT_DIR::TARGET_DIR")
    return parts[0], Path(parts[1]).expanduser().resolve(), Path(parts[2]).expanduser().resolve()


def _files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    if {part.lower() for part in directory.parts} & {
        "test", "testing", "validation", "valid", "val"
    }:
        raise ValueError(f"non-training directory is forbidden: {directory}")
    result = {
        path.name: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    if not result:
        raise ValueError(f"no images in {directory}")
    return result


def rows(root: Path, pairs: Iterable[Tuple[str, Path, Path]]) -> list[dict]:
    root = root.resolve()
    output = []
    names = set()
    used_paths = set()
    for label, input_dir, target_dir in pairs:
        inputs, targets = _files(input_dir), _files(target_dir)
        if set(inputs) != set(targets):
            missing_target = sorted(set(inputs) - set(targets))[:5]
            missing_input = sorted(set(targets) - set(inputs))[:5]
            raise ValueError(
                f"{label}: filename mismatch, missing target={missing_target}, "
                f"missing input={missing_input}"
            )
        for filename in sorted(inputs):
            blurry, sharp = inputs[filename].resolve(), targets[filename].resolve()
            for path in (blurry, sharp):
                try:
                    path.relative_to(root)
                except ValueError as error:
                    raise ValueError(f"asset is outside --root: {path}") from error
                if path in used_paths:
                    raise ValueError(f"asset reused across pairs: {path}")
                used_paths.add(path)
            name = f"{label}/{filename}"
            if name in names:
                raise ValueError(f"duplicate logical name: {name}")
            names.add(name)
            with Image.open(blurry) as low, Image.open(sharp) as high:
                low.verify(); high.verify()
            with Image.open(blurry) as low, Image.open(sharp) as high:
                if low.mode != "RGB" or high.mode != "RGB" or low.size != high.size:
                    raise ValueError(f"{name}: paired assets must be same-size RGB")
                width, height = low.size
            output.append({
                "schema": SCHEMA,
                "name": name,
                "split": "train",
                "blurry": str(blurry.relative_to(root)),
                "sharp": str(sharp.relative_to(root)),
                "source_sha256": sha256_file(blurry),
                "target_sha256": sha256_file(sharp),
                "width": width,
                "height": height,
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pair", action="append", type=_parse_pair, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.exists() or destination.with_suffix(destination.suffix + ".sha256").exists():
        raise FileExistsError(f"refusing overwrite: {destination}")
    payload = rows(args.root.expanduser().resolve(), args.pair)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in payload:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256_file(destination)
    digest_path = destination.with_suffix(destination.suffix + ".sha256")
    descriptor = os.open(digest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(f"{digest}  {destination.name}\n")
        handle.flush(); os.fsync(handle.fileno())
    print(json.dumps({"manifest": str(destination), "sha256": digest, "rows": len(payload)}))


if __name__ == "__main__":
    main()
