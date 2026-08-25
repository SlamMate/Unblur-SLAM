#!/usr/bin/env python3
"""Build a content-addressed train-only paired-video JSONL manifest.

Two input modes are supported: canonicalize an existing sequence JSONL, or
pair two aligned directory trees whose immediate child directories are
sequences.  The output paths are relative to ``--root`` and every referenced
asset is bound by SHA256.  Test/validation paths are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


SCHEMA = "unblur_slam.paired_video_train.v1"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.relative_to(root)
    lowered = {part.lower() for part in resolved.relative_to(root).parts}
    if lowered & {"test", "testing", "validation", "valid", "val"}:
        raise ValueError(f"non-training path is forbidden: {resolved}")
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return resolved


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _write_exclusive(rows: Iterable[Mapping[str, Any]], output: Path) -> int:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    if count == 0:
        output.unlink(missing_ok=True)
        raise ValueError("no eligible sequences were written")
    return count


def _canonical_row(
    *, name: str, blurry: list[Path], sharp: list[Path], root: Path, dataset: str
) -> Mapping[str, Any]:
    if not name or len(blurry) < 5 or len(blurry) != len(sharp):
        raise ValueError(f"bad sequence {name!r}: paired length must be equal and >=5")
    blurry = [_safe(path, root) for path in blurry]
    sharp = [_safe(path, root) for path in sharp]
    if len(set(blurry + sharp)) != len(blurry) + len(sharp):
        raise ValueError(f"sequence {name!r} reuses an asset")
    return {
        "schema": SCHEMA,
        "dataset": dataset,
        "split": "train",
        "sequence": name,
        "frame_count": len(blurry),
        "frame_indices": list(range(len(blurry))),
        "blurry": [_relative(path, root) for path in blurry],
        "sharp": [_relative(path, root) for path in sharp],
        "blurry_sha256": [sha256_file(path) for path in blurry],
        "sharp_sha256": [sha256_file(path) for path in sharp],
    }


def from_trees(root: Path, blurry_root: Path, sharp_root: Path, dataset: str):
    blurry_sequences = {path.name: path for path in blurry_root.iterdir() if path.is_dir()}
    sharp_sequences = {path.name: path for path in sharp_root.iterdir() if path.is_dir()}
    if not blurry_sequences or set(blurry_sequences) != set(sharp_sequences):
        raise ValueError("blurry/sharp sequence directory sets differ or are empty")
    for name in sorted(blurry_sequences):
        blurry = sorted(
            path for path in blurry_sequences[name].iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        sharp = sorted(
            path for path in sharp_sequences[name].iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if [path.name for path in blurry] != [path.name for path in sharp]:
            raise ValueError(f"frame names differ for sequence {name}")
        yield _canonical_row(
            name=f"{dataset}_{name}", blurry=blurry, sharp=sharp,
            root=root, dataset=dataset,
        )


def from_nested(
    root: Path, sequence_root: Path, blurry_subdir: str,
    sharp_subdir: str, dataset: str,
):
    sequences = sorted(path for path in sequence_root.iterdir() if path.is_dir())
    if not sequences:
        raise ValueError("nested sequence root is empty")
    for sequence in sequences:
        low_root, high_root = sequence / blurry_subdir, sequence / sharp_subdir
        if not low_root.is_dir() or not high_root.is_dir():
            raise ValueError(f"missing paired subdirectories in {sequence}")
        blurry = sorted(
            path for path in low_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        sharp = sorted(
            path for path in high_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if [path.name for path in blurry] != [path.name for path in sharp]:
            raise ValueError(f"frame names differ for sequence {sequence.name}")
        yield _canonical_row(
            name=f"{dataset}_{sequence.name}", blurry=blurry, sharp=sharp,
            root=root, dataset=dataset,
        )


def from_manifest(
    root: Path, source: Path, dataset: str, *, drop_short: bool = False
):
    names = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} is not an object")
            if str(payload.get("split", "train")).lower() not in {"train", "training"}:
                raise ValueError("source manifest contains a non-train record")
            name = str(payload.get("sequence", payload.get("name", "")))
            if not name or name in names:
                raise ValueError("sequence names must be non-empty and unique")
            names.add(name)
            blurry_values = payload.get("blurry", payload.get("input"))
            sharp_values = payload.get("sharp", payload.get("target"))
            if not isinstance(blurry_values, list) or not isinstance(sharp_values, list):
                raise ValueError(f"sequence {name!r} paths must be arrays")
            if len(blurry_values) < 5 or len(sharp_values) < 5:
                if drop_short:
                    continue
                raise ValueError(f"sequence {name!r} has fewer than five frames")
            yield _canonical_row(
                name=name,
                blurry=[root / str(value) for value in blurry_values],
                sharp=[root / str(value) for value in sharp_values],
                root=root,
                dataset=dataset,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-manifest", type=Path)
    group.add_argument("--blurry-root", type=Path)
    group.add_argument("--sequence-root", type=Path)
    parser.add_argument("--sharp-root", type=Path)
    parser.add_argument("--blurry-subdir")
    parser.add_argument("--sharp-subdir")
    parser.add_argument(
        "--drop-short", action="store_true",
        help="drop and disclose source records shorter than the fixed 5-frame clip",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if args.source_manifest is not None:
        rows = from_manifest(
            root, args.source_manifest.expanduser().resolve(), args.dataset,
            drop_short=args.drop_short,
        )
    elif args.blurry_root is not None:
        if args.sharp_root is None:
            parser.error("--sharp-root is required with --blurry-root")
        rows = from_trees(
            root, args.blurry_root.expanduser().resolve(),
            args.sharp_root.expanduser().resolve(), args.dataset,
        )
    else:
        if not args.blurry_subdir or not args.sharp_subdir:
            parser.error("--blurry-subdir and --sharp-subdir are required with --sequence-root")
        rows = from_nested(
            root, args.sequence_root.expanduser().resolve(),
            args.blurry_subdir, args.sharp_subdir, args.dataset,
        )
    count = _write_exclusive(rows, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": sha256_file(args.output.resolve()),
        "sequence_records": count,
        "short_records_dropped": bool(args.drop_short),
    }))


if __name__ == "__main__":
    main()
