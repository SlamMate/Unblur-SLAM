#!/usr/bin/env python3
"""Materialize the revision-pinned DPDD combined train/val PNG16 mirror.

Only two exact metadata files are requested.  The repository is never listed
and the test split is never requested.  Assets are staged, decoded losslessly,
audited for path/content overlap, and atomically published without overwrite.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import ctypes
import errno
import hashlib
import json
import os
import posixpath
import shutil
import struct
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Sequence


REPOSITORY = "JacobLinCool/DPDD"
REVISION = "52e4035a045ea1763313b9ce2b27cf2e620cfc30"
CONFIG = "combined"
WORKERS = 8
SCHEMA = "unblur_slam.dpdd_hf_png16_materialization.v1"
PAIR_SCHEMA = "unblur_slam.dpdd_hf_png16_pair.v1"
CANONICAL_PAIR_SCHEMA = "unblur_slam.dpdd_hf_canonical_pair.v1"
PREFLIGHT_SCHEMA = "unblur_slam.dpdd_hf_png16_preflight.v1"
README_PATH = "README.md"
README_BYTES = 2888
README_SHA256 = "2e643ed51910427a42e3402471c72ecd84682704d09af3485cf26bfc8634de0d"


@dataclass(frozen=True)
class SplitSpec:
    canonical: str
    hf_split: str
    metadata_path: str
    asset_root: str
    rows: int
    metadata_bytes: int
    metadata_sha256: str


SPLIT_SPECS: Mapping[str, SplitSpec] = {
    "train": SplitSpec(
        "train",
        "train",
        "config/combined/train/metadata.jsonl",
        "dd_dp_dataset_png/train_c",
        350,
        54950,
        "f63e8acb070a027f1ab56754f6ca5029c0eea80eaa2d7688bb86a65c38434023",
    ),
    "validation": SplitSpec(
        "validation",
        "val",
        "config/combined/val/metadata.jsonl",
        "dd_dp_dataset_png/val_c",
        74,
        11322,
        "55e7fe96a12686053c6726afab15dd8a84ba380c3aa4f03ac5839b7e9d16605d",
    ),
}


class MaterializationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Pair:
    split: str
    hf_split: str
    row_idx: int
    source_repo_path: str
    target_repo_path: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise MaterializationError(f"refusing to overwrite {path}") from error
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        raise


def _repo_url(repo_path: str) -> str:
    quoted = urllib.parse.quote(repo_path, safe="/")
    return f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{quoted}"


def _default_fetch(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Unblur-SLAM-DPDD-PNG16/1"}, method="GET"
    )
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    raise MaterializationError(f"metadata request failed: {url}") from last_error


def _default_open(url: str, timeout: float) -> BinaryIO:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Unblur-SLAM-DPDD-PNG16/1"}, method="GET"
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except (OSError, urllib.error.URLError) as error:
        raise MaterializationError(f"asset request failed: {url}") from error


def _normalize_splits(splits: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in splits:
        split = str(value).strip().lower()
        if split == "val":
            split = "validation"
        if split not in SPLIT_SPECS:
            raise MaterializationError(
                "only combined train and validation are authorized; test is sealed"
            )
        if split not in result:
            result.append(split)
    if not result:
        raise MaterializationError("at least one split is required")
    return tuple(result)


def _resolve_asset_path(raw: Any, spec: SplitSpec, role: str, row_idx: int) -> str:
    if not isinstance(raw, str) or not raw:
        raise MaterializationError(
            f"{spec.canonical} row {row_idx} invalid {role} path"
        )
    if "?" in raw or "#" in raw or "\\" in raw:
        raise MaterializationError(f"{spec.canonical} row {row_idx} unsafe {role} path")
    joined = posixpath.normpath(
        posixpath.join(posixpath.dirname(spec.metadata_path), raw)
    )
    prefix = f"{spec.asset_root}/{role}/"
    if not joined.startswith(prefix) or joined == prefix.rstrip("/"):
        raise MaterializationError(
            f"{spec.canonical} row {row_idx} {role} escapes pinned asset root"
        )
    if not joined.lower().endswith(".png"):
        raise MaterializationError(f"{spec.canonical} row {row_idx} is not PNG")
    return joined


def _parse_metadata(payload: bytes, spec: SplitSpec) -> list[Pair]:
    if len(payload) != spec.metadata_bytes or _sha256(payload) != spec.metadata_sha256:
        raise MaterializationError(
            f"{spec.canonical} pinned metadata bytes/SHA mismatch"
        )
    lines = payload.splitlines()
    if len(lines) != spec.rows:
        raise MaterializationError(
            f"{spec.canonical} expected {spec.rows} metadata rows, found {len(lines)}"
        )
    pairs: list[Pair] = []
    for row_idx, raw_line in enumerate(lines):
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MaterializationError(
                f"{spec.canonical} row {row_idx} is invalid JSON"
            ) from error
        if not isinstance(row, Mapping):
            raise MaterializationError(
                f"{spec.canonical} row {row_idx} is not an object"
            )
        source = _resolve_asset_path(
            row.get("source_file_name"), spec, "source", row_idx
        )
        target = _resolve_asset_path(
            row.get("target_file_name"), spec, "target", row_idx
        )
        if source == target:
            raise MaterializationError(
                f"{spec.canonical} row {row_idx} aliases source/target"
            )
        pairs.append(Pair(spec.canonical, spec.hf_split, row_idx, source, target))
    return pairs


def prepare_preflight(
    splits: Sequence[str] = ("train", "validation"),
    *,
    timeout: float = 60.0,
    fetch: Callable[[str, float], bytes] = _default_fetch,
    split_specs: Mapping[str, SplitSpec] = SPLIT_SPECS,
) -> tuple[Mapping[str, Any], list[Pair], Mapping[str, bytes]]:
    normalized = _normalize_splits(splits)
    # The card and only the two explicitly selected metadata paths are fetched.
    readme = fetch(_repo_url(README_PATH), timeout)
    if len(readme) != README_BYTES or _sha256(readme) != README_SHA256:
        raise MaterializationError("pinned README bytes/SHA mismatch")
    if b"license: mit" not in readme.lower():
        raise MaterializationError("pinned dataset card no longer declares MIT")

    metadata: dict[str, bytes] = {}
    pairs: list[Pair] = []
    for split in normalized:
        spec = split_specs[split]
        payload = fetch(_repo_url(spec.metadata_path), timeout)
        metadata[split] = payload
        pairs.extend(_parse_metadata(payload, spec))

    source_paths = {pair.source_repo_path for pair in pairs}
    target_paths = {pair.target_repo_path for pair in pairs}
    all_paths = [
        path
        for pair in pairs
        for path in (pair.source_repo_path, pair.target_repo_path)
    ]
    train_paths = {
        path
        for pair in pairs
        if pair.split == "train"
        for path in (pair.source_repo_path, pair.target_repo_path)
    }
    validation_paths = {
        path
        for pair in pairs
        if pair.split == "validation"
        for path in (pair.source_repo_path, pair.target_repo_path)
    }
    if len(set(all_paths)) != len(all_paths):
        raise MaterializationError("duplicate asset locator found in selected metadata")
    if source_paths & target_paths:
        raise MaterializationError("source/target locator sets overlap")
    if train_paths & validation_paths:
        raise MaterializationError("train/validation locator sets overlap")

    report: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "repository": REPOSITORY,
        "revision": REVISION,
        "config": CONFIG,
        "splits": {split: split_specs[split].rows for split in normalized},
        "metadata": {
            split: {
                "repo_path": split_specs[split].metadata_path,
                "bytes": len(metadata[split]),
                "sha256": _sha256(metadata[split]),
            }
            for split in normalized
        },
        "asset_count": len(all_paths),
        "asset_requests_made": 0,
        "locator_audit": {
            "globally_unique": True,
            "source_target_disjoint": True,
            "train_validation_disjoint": True,
        },
        "distribution": {
            "kind": "third_party_huggingface_mirror_of_dpdd_png16",
            "official_dpdd_download": False,
            "dataset_card_declared_license": "mit",
            "license_scope_warning": (
                "The pinned third-party HF card declares MIT; that claim does not "
                "establish redistribution rights for the original DPDD images."
            ),
            "readme": {
                "repo_path": README_PATH,
                "bytes": len(readme),
                "sha256": _sha256(readme),
            },
        },
        "test_disclosure": {
            "metadata_pristine": False,
            "metadata_exposure": (
                "filenames_lfs_oids_sizes_split_aggregate_row0_url_and_manifest_text_"
                "seen_before_freeze"
            ),
            "images_decoded": False,
            "pixels_opened": False,
            "metrics_opened": False,
            "requests_made_by_this_materializer": 0,
            "split_supported_by_this_materializer": False,
        },
    }
    return report, pairs, metadata


def _verify_png16(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as stream:
        header = stream.read(33)
    if len(header) < 33 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise MaterializationError(f"not a PNG: {path}")
    length = struct.unpack(">I", header[8:12])[0]
    if length != 13 or header[12:16] != b"IHDR":
        raise MaterializationError(f"missing canonical PNG IHDR: {path}")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", header[16:26])
    if width < 1 or height < 1 or bit_depth != 16 or color_type != 2:
        raise MaterializationError(
            f"expected 16-bit RGB PNG, got {width}x{height} depth={bit_depth} color={color_type}"
        )
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
    except Exception as error:
        raise MaterializationError(f"Pillow could not verify PNG: {path}") from error
    try:
        import cv2
        import numpy as np

        decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if (
            decoded is None
            or decoded.dtype != np.uint16
            or decoded.ndim != 3
            or decoded.shape != (height, width, 3)
        ):
            actual = None if decoded is None else (str(decoded.dtype), decoded.shape)
            raise MaterializationError(f"lossless decode mismatch for {path}: {actual}")
    except ImportError as error:
        raise MaterializationError("Pillow, OpenCV and NumPy are required") from error
    return {
        "width": width,
        "height": height,
        "channels": 3,
        "png_ihdr_bit_depth": bit_depth,
        "png_ihdr_color_type": color_type,
        "decoded_dtype": "uint16",
        "decoder": "opencv_imread_unchanged",
        "pillow_verify": True,
    }


def _download_one(
    repo_path: str,
    destination: Path,
    timeout: float,
    open_asset: Callable[[str, float], BinaryIO],
) -> Mapping[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = _repo_url(repo_path)
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            digest = hashlib.sha256()
            size = 0
            response = open_asset(url, timeout)
            with contextlib.closing(response), destination.open("xb") as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    digest.update(block)
                    size += len(block)
                stream.flush()
                os.fsync(stream.fileno())
            if size == 0:
                raise MaterializationError(f"empty asset: {repo_path}")
            image = _verify_png16(destination)
            return {
                "repo_path": repo_path,
                "revision_url": url,
                "relative_path": "",
                "bytes": size,
                "sha256": digest.hexdigest(),
                **image,
            }
        except BaseException as error:
            last_error = error
            with contextlib.suppress(FileNotFoundError):
                destination.unlink()
            if attempt < 2:
                time.sleep(0.25 * (attempt + 1))
    raise MaterializationError(
        f"failed asset after 3 attempts: {repo_path}"
    ) from last_error


def _publish_noreplace(stage: Path, output: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise MaterializationError("atomic no-overwrite requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(stage), -100, os.fsencode(output), 1) == 0:
        return
    number = ctypes.get_errno()
    if number == errno.EEXIST:
        raise MaterializationError(f"refusing to overwrite output: {output}")
    raise MaterializationError(f"atomic publication failed: {os.strerror(number)}")


def _audit_hashes(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    all_hashes = [
        record[role]["sha256"] for record in records for role in ("source", "target")
    ]
    sources = {record["source"]["sha256"] for record in records}
    targets = {record["target"]["sha256"] for record in records}
    train = {
        record[role]["sha256"]
        for record in records
        if record["split"] == "train"
        for role in ("source", "target")
    }
    validation = {
        record[role]["sha256"]
        for record in records
        if record["split"] == "validation"
        for role in ("source", "target")
    }
    if any(
        record["source"]["sha256"] == record["target"]["sha256"] for record in records
    ):
        raise MaterializationError("a pair has identical source and target content")
    if len(set(all_hashes)) != len(all_hashes):
        raise MaterializationError("duplicate asset content detected globally")
    if sources & targets:
        raise MaterializationError("source/target content sets overlap")
    if train & validation:
        raise MaterializationError("train/validation content sets overlap")
    return {
        "globally_unique": True,
        "source_target_disjoint": True,
        "train_validation_disjoint": True,
        "per_pair_source_target_distinct": True,
    }


def materialize(
    output: Path,
    splits: Sequence[str] = ("train", "validation"),
    *,
    timeout: float = 120.0,
    workers: int = WORKERS,
    fetch: Callable[[str, float], bytes] = _default_fetch,
    open_asset: Callable[[str, float], BinaryIO] = _default_open,
    split_specs: Mapping[str, SplitSpec] = SPLIT_SPECS,
) -> Path:
    output = output.expanduser().resolve()
    if output.exists():
        raise MaterializationError(f"refusing to overwrite output: {output}")
    if workers != WORKERS:
        raise MaterializationError(f"worker count is fixed at {WORKERS}")
    output.parent.mkdir(parents=True, exist_ok=True)
    preflight, pairs, metadata = prepare_preflight(
        splits, timeout=timeout, fetch=fetch, split_specs=split_specs
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    completed = False
    try:
        for split, payload in metadata.items():
            _write_new(stage / "provenance" / f"{split}_metadata.jsonl", payload)

        results: dict[tuple[int, str], Mapping[str, Any]] = {}
        futures: dict[
            concurrent.futures.Future[Mapping[str, Any]], tuple[int, str, Path]
        ] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for pair_idx, pair in enumerate(pairs):
                for role, repo_path in (
                    ("source", pair.source_repo_path),
                    ("target", pair.target_repo_path),
                ):
                    destination = stage / pair.split / role / f"{pair.row_idx:06d}.png"
                    future = pool.submit(
                        _download_one, repo_path, destination, timeout, open_asset
                    )
                    futures[future] = (pair_idx, role, destination)
            for future in concurrent.futures.as_completed(futures):
                pair_idx, role, destination = futures[future]
                result = dict(future.result())
                result["relative_path"] = destination.relative_to(stage).as_posix()
                results[(pair_idx, role)] = result

        records: list[Mapping[str, Any]] = []
        for pair_idx, pair in enumerate(pairs):
            source = results[(pair_idx, "source")]
            target = results[(pair_idx, "target")]
            if (source["width"], source["height"]) != (
                target["width"],
                target["height"],
            ):
                raise MaterializationError(
                    f"{pair.split} row {pair.row_idx} source/target dimensions differ"
                )
            records.append(
                {
                    "schema": PAIR_SCHEMA,
                    "repository": REPOSITORY,
                    "revision": REVISION,
                    "config": CONFIG,
                    "split": pair.split,
                    "hf_split": pair.hf_split,
                    "row_idx": pair.row_idx,
                    "source": source,
                    "target": target,
                    "pairing": "same_view_defocused_to_all_in_focus",
                }
            )
        hash_audit = _audit_hashes(records)
        pairs_path = stage / "pairs.jsonl"
        _write_new(pairs_path, b"".join(_canonical_bytes(record) for record in records))
        canonical_manifests: dict[str, Mapping[str, Any]] = {}
        for split in preflight["splits"]:
            canonical_records = [
                {
                    "schema": CANONICAL_PAIR_SCHEMA,
                    "name": f"dpdd_{record['split']}_{record['row_idx']:06d}",
                    "split": record["split"],
                    "defocus": record["source"]["relative_path"],
                    "sharp": record["target"]["relative_path"],
                    "source_sha256": record["source"]["sha256"],
                    "target_sha256": record["target"]["sha256"],
                }
                for record in records
                if record["split"] == split
            ]
            canonical_path = stage / "manifests" / f"{split}.jsonl"
            _write_new(
                canonical_path,
                b"".join(_canonical_bytes(record) for record in canonical_records),
            )
            canonical_manifests[split] = {
                "path": canonical_path.relative_to(stage).as_posix(),
                "sha256": sha256_file(canonical_path),
                "rows": len(canonical_records),
                "schema": CANONICAL_PAIR_SCHEMA,
                "paths_relative_to": "dataset_root",
            }
        asset_bytes = sum(
            int(record[role]["bytes"])
            for record in records
            for role in ("source", "target")
        )
        manifest = {
            "schema": SCHEMA,
            "repository": REPOSITORY,
            "revision": REVISION,
            "config": CONFIG,
            "splits": dict(preflight["splits"]),
            "pair_count": len(records),
            "asset_count": len(records) * 2,
            "asset_bytes": asset_bytes,
            "workers": WORKERS,
            "pairs_jsonl": {"path": "pairs.jsonl", "sha256": sha256_file(pairs_path)},
            "canonical_manifests": canonical_manifests,
            "metadata": preflight["metadata"],
            "distribution": preflight["distribution"],
            "test_disclosure": preflight["test_disclosure"],
            "locator_audit": preflight["locator_audit"],
            "content_hash_audit": hash_audit,
            "image_contract": {
                "encoding": "PNG",
                "ihdr_bit_depth": 16,
                "ihdr_color_type": 2,
                "decoded_dtype": "uint16",
                "decoded_shape": "HWC3",
                "decode_mode": "opencv_imread_unchanged",
                "files_preserved_byte_exact": True,
            },
            "publication": {"atomic": True, "overwrite": False},
        }
        _write_new(stage / "dataset_manifest.json", _canonical_bytes(manifest))
        files = sorted(path for path in stage.rglob("*") if path.is_file())
        _write_new(
            stage / "SHA256SUMS",
            "".join(
                f"{sha256_file(path)}  {path.relative_to(stage).as_posix()}\n"
                for path in files
            ).encode("utf-8"),
        )
        _publish_noreplace(stage, output)
        completed = True
        return output / "dataset_manifest.json"
    finally:
        if not completed:
            shutil.rmtree(stage, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        report, _, _ = prepare_preflight(args.splits, timeout=args.timeout)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        raise MaterializationError(
            "--output is required unless --preflight-only is used"
        )
    print(materialize(args.output, args.splits, timeout=args.timeout))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MaterializationError as error:
        raise SystemExit(f"error: {error}") from error
