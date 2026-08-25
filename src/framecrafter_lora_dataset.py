"""Role-aware paired-data adapter for FrameCrafter LoRA fine-tuning.

The upstream ``WanNVSDataset`` samples every image from one folder and then
randomly assigns the sampled images to context or target roles.  That contract
cannot express the training example needed here: M blurry (or EVSSM) context
views and N *sharp* target views at explicitly recorded cameras.  It also
stores ``repeat`` but does not apply it in ``__len__`` in the currently pinned
FrameCrafter checkout.

This module deliberately keeps the official model/loss/training loop intact.
``scripts/launch_framecrafter_lora.py`` injects ``PairedFrameCrafterDataset``
as ``WanNVSDataset`` before executing the official
``model_training/train.py``.  The adapter consumes ``samples.jsonl`` produced
by ``scripts/build_framecrafter_lora_dataset.py`` and returns the exact keys
expected by the official training module.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


SCHEMA_VERSION = "framecrafter-paired-lora-v1"
SAFE_POSE_KINDS = frozenset(
    {
        "estimated_unaligned",
        "slam_estimate",
        "droid_estimate",
        "tracking_estimate",
        "non_gt_estimate",
    }
)
_UNSAFE_POSE_TOKEN = re.compile(r"(^|[^a-z])(gt|ground[ _-]*truth|reference|aligned)([^a-z]|$)", re.I)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _validate_pose_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("pose_provenance must be an object with an explicit non-GT kind")
    result = dict(value)
    kind = str(result.get("kind", "")).strip().lower()
    if kind not in SAFE_POSE_KINDS:
        raise ValueError(
            "pose_provenance.kind must explicitly denote an unaligned estimate; "
            f"got {kind!r}, expected one of {sorted(SAFE_POSE_KINDS)}"
        )
    searchable = " ".join(str(result.get(key, "")) for key in ("source", "key", "description"))
    if _UNSAFE_POSE_TOKEN.search(searchable):
        raise ValueError(f"ground-truth/aligned pose provenance is forbidden: {searchable!r}")
    result["kind"] = kind
    return result


def _load_poses(record: Mapping[str, Any], manifest_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    provenance = _validate_pose_provenance(record.get("pose_provenance", {}))
    if "c2w" in record:
        poses = np.asarray(record["c2w"], dtype=np.float32)
        provenance["inline_sha256"] = canonical_sha256(record["c2w"])
    else:
        raw_path = record.get("trajectory_path")
        if not raw_path:
            raise ValueError("each sequence needs either inline c2w or trajectory_path")
        path = _resolve_path(str(raw_path), manifest_dir)
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                requested = record.get("trajectory_key")
                if requested is None:
                    safe_keys = (
                        "traj_est_not_align",
                        "traj_est_unaligned",
                        "c2w_est",
                        "poses_est",
                    )
                    requested = next((key for key in safe_keys if key in archive.files), None)
                if requested is None or requested not in archive.files:
                    raise ValueError(
                        f"{path} has no explicitly safe estimated trajectory key; "
                        "set trajectory_key to an unaligned estimate"
                    )
                if _UNSAFE_POSE_TOKEN.search(str(requested)):
                    raise ValueError(f"unsafe trajectory key: {requested!r}")
                poses = np.asarray(archive[str(requested)], dtype=np.float32)
                provenance["key"] = str(requested)
        elif suffix == ".npy":
            poses = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        else:
            flat = np.loadtxt(path, dtype=np.float32)
            if flat.ndim == 1:
                flat = flat[None, :]
            if flat.shape[-1] != 16:
                raise ValueError(f"trajectory text must contain 16 floats per row: {path}")
            poses = flat.reshape(-1, 4, 4)
        provenance.update({"source": str(path), "source_sha256": sha256_file(path)})

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"c2w must have shape [F,4,4], got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("c2w contains NaN or infinity")
    bottom = poses[:, 3, :]
    if not np.allclose(bottom, np.array([0, 0, 0, 1], np.float32), atol=1e-4):
        raise ValueError("c2w matrices must be homogeneous transforms with bottom row [0,0,0,1]")
    return poses, provenance


def _load_intrinsics(record: Mapping[str, Any], manifest_dir: Path, count: int) -> np.ndarray:
    value: Any
    if "K" in record:
        value = record["K"]
    elif "intrinsics" in record:
        value = record["intrinsics"]
    elif "intrinsics_path" in record:
        path = _resolve_path(str(record["intrinsics_path"]), manifest_dir)
        if path.suffix.lower() == ".npz":
            key = str(record.get("intrinsics_key", "K"))
            with np.load(path, allow_pickle=False) as archive:
                if key not in archive.files:
                    raise ValueError(f"intrinsics key {key!r} missing from {path}")
                value = archive[key]
        else:
            value = np.load(path, allow_pickle=False)
    else:
        fx = record.get("fl_x")
        fy = record.get("fl_y", fx)
        cx, cy = record.get("cx"), record.get("cy")
        if None in (fx, fy, cx, cy):
            raise ValueError("each sequence needs K/intrinsics or fl_x,fl_y,cx,cy")
        value = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    intrinsics = np.asarray(value, dtype=np.float32)
    if intrinsics.shape == (3, 3):
        intrinsics = np.repeat(intrinsics[None], count, axis=0)
    if intrinsics.shape != (count, 3, 3):
        raise ValueError(f"intrinsics must have shape [3,3] or [F,3,3], got {intrinsics.shape}")
    if not np.isfinite(intrinsics).all() or np.any(intrinsics[:, (0, 1), (0, 1)] <= 0):
        raise ValueError("intrinsics must be finite with positive fx/fy")
    return intrinsics


def _copy_asset(source: Path, output_root: Path, sequence: str, role: str, index: int) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with Image.open(source) as image:
            width, height = image.size
            image.verify()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid image {source}: {exc}") from exc
    suffix = source.suffix.lower() or ".png"
    relative = Path("assets") / sequence / role / f"{index:06d}{suffix}"
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    if destination.exists():
        if sha256_file(destination) != source_hash:
            raise FileExistsError(f"refusing to overwrite different asset: {destination}")
    else:
        shutil.copy2(source, destination)
    return {
        "path": relative.as_posix(),
        "sha256": source_hash,
        "source_path": str(source),
        "width": int(width),
        "height": int(height),
    }


def _safe_sequence_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._")
    if not name:
        raise ValueError("sequence name is empty after sanitization")
    return name


def _nearest_context_indices(frame_count: int, targets: Sequence[int], count: int) -> list[int]:
    target_set = set(int(index) for index in targets)
    center = float(sum(targets)) / len(targets)
    candidates = [index for index in range(frame_count) if index not in target_set]
    candidates.sort(key=lambda index: (min(abs(index - target) for target in targets), abs(index - center), index))
    if len(candidates) < count:
        raise ValueError(f"need {count} context cameras plus {len(targets)} targets, have only {frame_count} frames")
    return sorted(candidates[:count])


def _choose_context_mode(
    requested: str,
    rank: int,
    evssm_available: bool,
    hybrid_evssm_fraction: float,
) -> str:
    if requested == "raw":
        return "raw_blurry"
    if requested == "evssm":
        if not evssm_available:
            raise ValueError("context_mode=evssm requires an EVSSM image for every selected context")
        return "evssm"
    # Deterministic: closest context views (rank 0 first) receive EVSSM.
    use_evssm = evssm_available and rank < int(round(hybrid_evssm_fraction * 1_000_000))
    return "evssm" if use_evssm else "raw_blurry"


@dataclass(frozen=True)
class BuildSummary:
    output_root: Path
    sample_count: int
    sequence_count: int
    num_input_frames: int
    num_output_frames: int
    samples_sha256: str


def build_paired_dataset(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    num_input_frames: int = 6,
    num_output_frames: int = 1,
    sample_stride: int = 1,
    max_samples_per_sequence: int | None = None,
    context_mode: str = "raw",
    hybrid_evssm_fraction: float = 0.5,
) -> BuildSummary:
    """Build a deterministic role-aware training corpus from paired sequences.

    Input is JSONL with ``blurry``, ``sharp``, camera intrinsics, c2w poses,
    and an explicit ``pose_provenance.kind`` from :data:`SAFE_POSE_KINDS`.
    Paths are resolved relative to the input manifest.
    """

    manifest = Path(manifest_path).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    if num_input_frames < 1 or num_output_frames < 1:
        raise ValueError("num_input_frames and num_output_frames must be positive")
    if sample_stride < 1:
        raise ValueError("sample_stride must be positive")
    if context_mode not in {"raw", "evssm", "hybrid"}:
        raise ValueError("context_mode must be raw, evssm, or hybrid")
    if not 0.0 <= hybrid_evssm_fraction <= 1.0:
        raise ValueError("hybrid_evssm_fraction must be in [0,1]")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {manifest}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"manifest row {line_number} must be an object")
        records.append(value)
    if not records:
        raise ValueError("paired manifest is empty")

    root.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for record in records:
        sequence = _safe_sequence_name(record.get("sequence", ""))
        if sequence in used_names:
            raise ValueError(f"duplicate sequence name: {sequence}")
        used_names.add(sequence)
        blurry = list(record.get("blurry", []))
        sharp = list(record.get("sharp", []))
        evssm = list(record.get("evssm", [])) if record.get("evssm") is not None else []
        if not blurry or len(blurry) != len(sharp):
            raise ValueError(f"sequence {sequence!r} needs equal non-empty blurry/sharp lists")
        if evssm and len(evssm) != len(blurry):
            raise ValueError(f"sequence {sequence!r} EVSSM list length differs from blurry list")
        frame_count = len(blurry)
        poses, pose_provenance = _load_poses(record, manifest.parent)
        if len(poses) != frame_count:
            raise ValueError(f"sequence {sequence!r}: {len(poses)} poses for {frame_count} paired frames")
        intrinsics = _load_intrinsics(record, manifest.parent, frame_count)
        convention = str(record.get("camera_convention", "opencv")).lower()
        if convention not in {"opencv", "opengl"}:
            raise ValueError("camera_convention must be opencv or opengl")

        raw_assets = [
            _copy_asset(_resolve_path(path, manifest.parent), root, sequence, "raw_blurry", index)
            for index, path in enumerate(blurry)
        ]
        sharp_assets = [
            _copy_asset(_resolve_path(path, manifest.parent), root, sequence, "sharp_target", index)
            for index, path in enumerate(sharp)
        ]
        evssm_assets: list[dict[str, Any] | None] = [None] * frame_count
        for index, path in enumerate(evssm):
            if path is not None and str(path).strip():
                evssm_assets[index] = _copy_asset(
                    _resolve_path(str(path), manifest.parent), root, sequence, "evssm", index
                )

        sequence_samples = 0
        final_start = frame_count - num_output_frames
        for target_start in range(0, final_start + 1, sample_stride):
            if max_samples_per_sequence is not None and sequence_samples >= max_samples_per_sequence:
                break
            target_indices = list(range(target_start, target_start + num_output_frames))
            contexts = _nearest_context_indices(frame_count, target_indices, num_input_frames)
            # Rank by proximity so hybrid EVSSM selection is temporal, not filename based.
            by_proximity = sorted(contexts, key=lambda index: (min(abs(index - t) for t in target_indices), index))
            evssm_budget = int(round(num_input_frames * hybrid_evssm_fraction))
            evssm_indices = set(by_proximity[:evssm_budget])

            context_rows = []
            for index in contexts:
                if context_mode == "evssm":
                    mode = _choose_context_mode("evssm", 0, evssm_assets[index] is not None, 1.0)
                elif context_mode == "hybrid":
                    mode = "evssm" if index in evssm_indices and evssm_assets[index] is not None else "raw_blurry"
                else:
                    mode = "raw_blurry"
                asset = evssm_assets[index] if mode == "evssm" else raw_assets[index]
                assert asset is not None
                context_rows.append(
                    {
                        **asset,
                        "frame_index": index,
                        "role": "context",
                        "image_mode": mode,
                        "c2w": poses[index].tolist(),
                        "c2w_sha256": canonical_sha256(poses[index].tolist()),
                        "K": intrinsics[index].tolist(),
                        "K_sha256": canonical_sha256(intrinsics[index].tolist()),
                    }
                )
            target_rows = [
                {
                    **sharp_assets[index],
                    "frame_index": index,
                    "role": "sharp_target",
                    "image_mode": "sharp",
                    "c2w": poses[index].tolist(),
                    "c2w_sha256": canonical_sha256(poses[index].tolist()),
                    "K": intrinsics[index].tolist(),
                    "K_sha256": canonical_sha256(intrinsics[index].tolist()),
                }
                for index in target_indices
            ]
            sample_id = f"{sequence}__{target_start:06d}__m{num_input_frames}n{num_output_frames}"
            samples.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "camera_convention": convention,
                    "pose_provenance": pose_provenance,
                    "contexts": context_rows,
                    "targets": target_rows,
                    "camera_sha256": canonical_sha256(
                        [
                            {"c2w": frame["c2w"], "K": frame["K"]}
                            for frame in context_rows + target_rows
                        ]
                    ),
                }
            )
            sequence_samples += 1

    if not samples:
        raise ValueError("no samples were produced")
    samples_path = root / "samples.jsonl"
    payload = "".join(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n" for sample in samples)
    samples_path.write_text(payload, encoding="utf-8")
    samples_hash = sha256_file(samples_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "samples_path": samples_path.name,
        "samples_sha256": samples_hash,
        "sequence_count": len(records),
        "sample_count": len(samples),
        "num_input_frames": num_input_frames,
        "num_output_frames": num_output_frames,
        "context_mode": context_mode,
        "hybrid_evssm_fraction": hybrid_evssm_fraction,
        "pose_contract": "non-GT, unaligned estimate only",
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BuildSummary(
        output_root=root,
        sample_count=len(samples),
        sequence_count=len(records),
        num_input_frames=num_input_frames,
        num_output_frames=num_output_frames,
        samples_sha256=samples_hash,
    )


def _resize_cover(image: Image.Image, target_height: int, target_width: int) -> tuple[Image.Image, float, float]:
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    crop_x = (resized_width - target_width) / 2.0
    crop_y = (resized_height - target_height) / 2.0
    left, top = int(round(crop_x)), int(round(crop_y))
    return resized.crop((left, top, left + target_width, top + target_height)), crop_x, crop_y


def _scale_K(K: np.ndarray, source_width: int, source_height: int, height: int, width: int) -> np.ndarray:
    scale = max(width / source_width, height / source_height)
    resized_width, resized_height = round(source_width * scale), round(source_height * scale)
    crop_x = (resized_width - width) / 2.0
    crop_y = (resized_height - height) / 2.0
    result = K.astype(np.float32, copy=True)
    result[0, 0] *= scale
    result[1, 1] *= scale
    result[0, 2] = result[0, 2] * scale - crop_x
    result[1, 2] = result[1, 2] * scale - crop_y
    return result


@torch.no_grad()
def _normalize_cameras(c2w: torch.Tensor) -> torch.Tensor:
    rotation = c2w[:, :3, :3]
    translation = c2w[:, :3, 3]
    anchor_rotation = rotation[-1]
    anchor_translation = translation[-1]
    aligned_rotation = anchor_rotation.transpose(0, 1) @ rotation
    aligned_translation = (
        anchor_rotation.transpose(0, 1) @ (translation - anchor_translation).unsqueeze(-1)
    ).squeeze(-1)
    scale = aligned_translation.norm(dim=-1).mean().clamp_min(1e-12)
    result = torch.zeros_like(c2w)
    result[:, :3, :3] = aligned_rotation
    result[:, :3, 3] = aligned_translation / scale
    result[:, 3, 3] = 1
    return result


def _plucker(K: torch.Tensor, c2w: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x, y = torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy")
    coords = torch.stack((x + 0.5, y + 0.5, torch.ones_like(x)), dim=-1).float()
    directions = torch.einsum("tij,hwj->thwi", torch.linalg.inv(K), coords)
    directions = torch.einsum("tij,thwj->thwi", c2w[:, :3, :3], directions)
    directions = F.normalize(directions, p=2, dim=-1)
    origins = torch.broadcast_to(c2w[:, None, None, :3, 3], directions.shape)
    moment = torch.cross(origins, directions, dim=-1)
    rays = torch.cat((directions, moment), dim=-1).permute(0, 3, 1, 2)
    return torch.nn.PixelUnshuffle(8)(rays)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty samples file: {path}")
    for row in rows:
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported sample schema: {row.get('schema_version')!r}")
        _validate_pose_provenance(row.get("pose_provenance", {}))
    return rows


class PairedFrameCrafterDataset(Dataset):
    """Drop-in, role-aware replacement for the official ``WanNVSDataset``."""

    def __init__(
        self,
        base_path: str,
        metadata_path: str | None,
        repeat: int,
        num_frames: int,
        height: int,
        width: int,
        height_division_factor: int = 8,
        width_division_factor: int = 8,
        time_division_factor: int = 4,
        time_division_remainder: int = 1,
        sampling_strategy: str = "prob_random",
        num_dataset_samples: int = 1000,
        no_pixel_unshuffle: bool = False,
        num_input_frames: int | None = None,
        num_output_frames: int | None = None,
        min_input_frames: int = 3,
        min_output_frames: int = 1,
        **_: Any,
    ) -> None:
        del height_division_factor, width_division_factor, time_division_factor
        del time_division_remainder, sampling_strategy, min_input_frames, min_output_frames
        self.base_path = Path(base_path).expanduser().resolve()
        if metadata_path and Path(metadata_path).is_file():
            samples_path = Path(metadata_path).expanduser().resolve()
        else:
            samples_path = self.base_path / "samples.jsonl"
        self.samples = _load_jsonl(samples_path)[: int(num_dataset_samples)]
        self.repeat = int(repeat)
        self.height, self.width = int(height), int(width)
        if self.repeat < 1:
            raise ValueError("dataset repeat must be positive")
        if self.height % 8 or self.width % 8:
            raise ValueError("height and width must be divisible by 8")
        if no_pixel_unshuffle:
            raise ValueError("FrameCrafter in_dim=420 requires PixelUnshuffle raymaps")
        first_m = len(self.samples[0]["contexts"])
        first_n = len(self.samples[0]["targets"])
        self.num_input_frames = first_m if num_input_frames is None else int(num_input_frames)
        self.num_output_frames = first_n if num_output_frames is None else int(num_output_frames)
        if int(num_frames) != self.num_input_frames + self.num_output_frames:
            raise ValueError(
                f"num_frames={num_frames} but paired contract is M+N="
                f"{self.num_input_frames}+{self.num_output_frames}"
            )
        for sample in self.samples:
            if len(sample["contexts"]) != self.num_input_frames or len(sample["targets"]) != self.num_output_frames:
                raise ValueError("all paired samples must use the same M-to-N split")
        # The official runner mutates these fields for curriculum datasets.
        self.current_epoch = 0
        self.num_epochs = 1

    def __len__(self) -> int:
        # This is intentionally unlike the pinned official dataset: repeat is
        # part of the logical index space and therefore really affects steps.
        return len(self.samples) * self.repeat

    def _asset_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.base_path / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[int(index) % len(self.samples)]
        frames = list(sample["contexts"]) + list(sample["targets"])
        processed: list[Image.Image] = []
        Ks: list[np.ndarray] = []
        poses: list[np.ndarray] = []
        for frame in frames:
            path = self._asset_path(frame["path"])
            if sha256_file(path) != frame["sha256"]:
                raise RuntimeError(f"training asset hash mismatch: {path}")
            if canonical_sha256(frame["c2w"]) != frame["c2w_sha256"]:
                raise RuntimeError(f"training c2w hash mismatch in {sample['sample_id']}")
            if canonical_sha256(frame["K"]) != frame["K_sha256"]:
                raise RuntimeError(f"training K hash mismatch in {sample['sample_id']}")
            with Image.open(path) as source:
                image = source.convert("RGB")
                source_width, source_height = image.size
                resized, _, _ = _resize_cover(image, self.height, self.width)
            processed.append(resized)
            Ks.append(_scale_K(np.asarray(frame["K"], np.float32), source_width, source_height, self.height, self.width))
            poses.append(np.asarray(frame["c2w"], np.float32))

        c2w = torch.from_numpy(np.stack(poses)).float()
        if sample.get("camera_convention", "opencv") == "opengl":
            w2c = torch.linalg.inv(c2w)
            w2c[:, [1, 2], :] *= -1
            c2w = torch.linalg.inv(w2c)
        c2w = _normalize_cameras(c2w)
        raymap = _plucker(torch.from_numpy(np.stack(Ks)).float(), c2w, self.height, self.width)
        context_images = processed[: self.num_input_frames]
        return {
            "input_images": context_images,
            "target_images": processed,
            "raymap": raymap,
            "prompt": "",
            "sample_id": sample["sample_id"],
            "provenance": {
                "pose": sample["pose_provenance"],
                "contexts": [frame["sha256"] for frame in sample["contexts"]],
                "targets": [frame["sha256"] for frame in sample["targets"]],
            },
        }


def official_dataset_repeat_is_broken(framecrafter_root: str | Path) -> bool:
    """Return whether the pinned upstream dataset ignores ``self.repeat``.

    AST inspection avoids importing the 14B training stack merely for a
    preflight.  An unknown/missing implementation fails closed as broken.
    """

    source = Path(framecrafter_root) / "diffsynth" / "core" / "data" / "dataset.py"
    if not source.is_file():
        return True
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return True
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__len__":
            text = ast.unparse(node)
            if "self.repeat" in text:
                return False
    return True


def validate_cpu_contract(dataset: PairedFrameCrafterDataset) -> dict[str, Any]:
    sample = dataset[0]
    expected_frames = dataset.num_input_frames + dataset.num_output_frames
    expected_shape = (expected_frames, 6 * 64, dataset.height // 8, dataset.width // 8)
    if tuple(sample["raymap"].shape) != expected_shape:
        raise RuntimeError(f"raymap shape {tuple(sample['raymap'].shape)} != {expected_shape}")
    if len(sample["input_images"]) != dataset.num_input_frames:
        raise RuntimeError("input image count violates M contract")
    if len(sample["target_images"]) != expected_frames:
        raise RuntimeError("target image count violates M+N contract")
    return {
        "status": "passed",
        "logical_length": len(dataset),
        "sample_id": sample["sample_id"],
        "M": dataset.num_input_frames,
        "N": dataset.num_output_frames,
        "height": dataset.height,
        "width": dataset.width,
        "raymap_shape": list(sample["raymap"].shape),
        "pose_kind": sample["provenance"]["pose"]["kind"],
    }


__all__ = [
    "BuildSummary",
    "PairedFrameCrafterDataset",
    "SAFE_POSE_KINDS",
    "SCHEMA_VERSION",
    "build_paired_dataset",
    "canonical_sha256",
    "official_dataset_repeat_is_broken",
    "sha256_file",
    "validate_cpu_contract",
]
