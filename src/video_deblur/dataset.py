"""JSONL sequence dataset for causal video deblurring.

Each non-empty JSONL line describes one ordered video sequence using either
parallel arrays::

    {"sequence":"turtle", "blurry":[...], "sharp":[...], "teacher":[...]}

or explicit frame objects::

    {"sequence":"turtle", "frames":[
      {"blurry":"blur/000.png", "sharp":"sharp/000.png"}, ...]}

``input``/``blur`` and ``target``/``gt`` are accepted aliases.  Relative paths
are resolved against ``root`` when supplied, otherwise against the manifest's
directory.  All random spatial transforms are shared across every frame and
modality in a clip, which is essential for temporal supervision.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


EVSSM_PRECOMPUTE_SCHEMA = "unblur_slam.video_deblur_evssm_precompute.v1"
TEACHER_PROVENANCE_SCHEMA = "unblur_slam.video_deblur_teacher_provenance.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_digest(value: object, label: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _report_path(value: object, report_path: Path, label: str) -> Path:
    if not value:
        raise ValueError(f"precompute report is missing {label}")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = report_path.parent / path
    return path.resolve()


def _manifest_teacher_paths(manifest: Path) -> Tuple[int, List[Path]]:
    """Return sequence count and flattened cached-teacher paths."""

    paths: List[Path] = []
    sequence_count = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"manifest line {line_number} must be an object")
            sequence_count += 1
            if "frames" in payload:
                frames = payload["frames"]
                if not isinstance(frames, list) or not frames:
                    raise ValueError(f"manifest line {line_number} has no frames")
                teacher_values = [
                    _optional_first(frame, ("teacher", "single_frame_teacher"))
                    if isinstance(frame, dict)
                    else None
                    for frame in frames
                ]
            else:
                value = _optional_first(payload, ("teacher", "single_frame_teacher"))
                teacher_values = list(value) if isinstance(value, list) else []
            if not teacher_values or any(value is None for value in teacher_values):
                raise ValueError(
                    "an EVSSM precompute manifest must contain a teacher for every frame"
                )
            for value in teacher_values:
                path = Path(str(value)).expanduser()
                if not path.is_absolute():
                    path = manifest.parent / path
                paths.append(path.resolve())
    return sequence_count, paths


def load_evssm_precompute_report(
    report: str,
    *,
    expected_manifest: Optional[str] = None,
    verify_teacher_artifacts: bool = True,
) -> Dict[str, object]:
    """Validate a frozen-EVSSM cache and return compact training provenance.

    The output manifest digest, its ordered teacher paths, every cached PNG
    digest, and the EVSSM checkpoint digest are bound together by the report.
    This prevents training an ``evssm``-domain adapter on an untracked mixture
    of single-frame teachers.
    """

    report_path = Path(report).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"EVSSM precompute report does not exist: {report_path}")
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema") != EVSSM_PRECOMPUTE_SCHEMA:
        raise ValueError("unsupported EVSSM precompute report schema")

    manifest = _report_path(payload.get("output_manifest"), report_path, "output_manifest")
    if not manifest.is_file():
        raise FileNotFoundError(f"precomputed EVSSM manifest does not exist: {manifest}")
    expected_manifest_digest = _sha256_digest(
        payload.get("output_manifest_sha256"), "output_manifest_sha256"
    )
    actual_manifest_digest = sha256_file(manifest)
    if actual_manifest_digest != expected_manifest_digest:
        raise ValueError("precomputed EVSSM output manifest SHA-256 mismatch")
    if expected_manifest is not None:
        requested_manifest = Path(expected_manifest).expanduser().resolve()
        if requested_manifest != manifest:
            raise ValueError(
                "training manifest does not match precompute_report.output_manifest"
            )

    checkpoint_digest = _sha256_digest(
        payload.get("checkpoint_sha256"), "checkpoint_sha256"
    )
    checkpoint_path = _report_path(
        payload.get("checkpoint"), report_path, "checkpoint"
    )
    if checkpoint_path.is_file() and sha256_file(checkpoint_path) != checkpoint_digest:
        raise ValueError("precompute report EVSSM checkpoint SHA-256 mismatch")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("precompute report frames must be an array")
    if int(payload.get("frame_count", -1)) != len(frames):
        raise ValueError("precompute report frame_count mismatch")
    sequence_count, manifest_teachers = _manifest_teacher_paths(manifest)
    if int(payload.get("sequence_count", -1)) != sequence_count:
        raise ValueError("precompute report sequence_count mismatch")
    if len(manifest_teachers) != len(frames):
        raise ValueError("precompute report and output manifest frame counts differ")

    for flat_index, (record, manifest_teacher) in enumerate(
        zip(frames, manifest_teachers)
    ):
        if not isinstance(record, dict):
            raise ValueError(f"precompute report frame {flat_index} must be an object")
        report_teacher = _report_path(
            record.get("teacher"), report_path, f"frames[{flat_index}].teacher"
        )
        if report_teacher != manifest_teacher:
            raise ValueError(
                f"precompute report teacher path mismatch at frame {flat_index}"
            )
        teacher_digest = _sha256_digest(
            record.get("teacher_sha256"),
            f"frames[{flat_index}].teacher_sha256",
        )
        if verify_teacher_artifacts:
            if not report_teacher.is_file():
                raise FileNotFoundError(
                    f"precomputed EVSSM teacher does not exist: {report_teacher}"
                )
            if sha256_file(report_teacher) != teacher_digest:
                raise ValueError(
                    f"precomputed EVSSM teacher SHA-256 mismatch at frame {flat_index}"
                )

    return {
        "schema": TEACHER_PROVENANCE_SCHEMA,
        "storage": "precomputed_png_rgb8",
        "teacher_domain": "evssm_restored_rgb_0_1",
        "evssm_checkpoint_sha256": checkpoint_digest,
        "evssm_checkpoint": str(checkpoint_path),
        "precompute_report": str(report_path),
        "precompute_report_sha256": sha256_file(report_path),
        "teacher_manifest": str(manifest),
        "teacher_manifest_sha256": actual_manifest_digest,
        "teacher_artifacts_verified": bool(verify_teacher_artifacts),
        "sequence_count": sequence_count,
        "frame_count": len(frames),
    }


@dataclass(frozen=True)
class _SequenceRecord:
    name: str
    blurry: Tuple[Path, ...]
    sharp: Tuple[Path, ...]
    teacher: Optional[Tuple[Path, ...]]


def _first(mapping: Dict[str, object], keys: Sequence[str]) -> object:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError("missing one of: " + ", ".join(keys))


def _optional_first(mapping: Dict[str, object], keys: Sequence[str]) -> Optional[object]:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


class VideoDeblurJsonlDataset(Dataset):
    """Create fixed-length causal clips from sequence-oriented JSONL records."""

    def __init__(
        self,
        manifest: Optional[str],
        clip_length: int = 5,
        stride: int = 1,
        crop_size: int = 0,
        augment: bool = False,
        root: Optional[str] = None,
        pad_short_sequences: bool = True,
        precompute_report: Optional[str] = None,
        verify_precompute_artifacts: bool = True,
    ):
        super().__init__()
        if clip_length < 1:
            raise ValueError("clip_length must be positive")
        if stride < 1:
            raise ValueError("stride must be positive")
        if crop_size < 0:
            raise ValueError("crop_size cannot be negative")
        self.teacher_provenance: Optional[Dict[str, object]] = None
        if precompute_report is not None:
            self.teacher_provenance = load_evssm_precompute_report(
                precompute_report,
                expected_manifest=manifest,
                verify_teacher_artifacts=verify_precompute_artifacts,
            )
            self.manifest = Path(
                str(self.teacher_provenance["teacher_manifest"])
            ).resolve()
        elif manifest is not None:
            self.manifest = Path(manifest).expanduser().resolve()
        else:
            raise ValueError("manifest or precompute_report must be supplied")
        if not self.manifest.is_file():
            raise FileNotFoundError(f"JSONL manifest does not exist: {self.manifest}")
        self.root = Path(root).expanduser().resolve() if root else self.manifest.parent
        self.clip_length = int(clip_length)
        self.stride = int(stride)
        self.crop_size = int(crop_size)
        self.augment = bool(augment)
        self.pad_short_sequences = bool(pad_short_sequences)

        self.sequences = self._load_manifest()
        self.clips: List[Tuple[int, Tuple[int, ...]]] = []
        for sequence_index, record in enumerate(self.sequences):
            length = len(record.blurry)
            if self.pad_short_sequences:
                # Streaming inference repeats frame zero on the left until a
                # full history exists.  Train the exact same prefixes, so the
                # first H-1 outputs are supervised instead of never becoming a
                # clip target.  Prefix targets are always included regardless
                # of steady-state stride.
                prefix_end = min(length, self.clip_length - 1)
                targets = list(range(prefix_end))
                targets.extend(range(prefix_end, length, self.stride))
                if targets and targets[-1] != length - 1:
                    targets.append(length - 1)
                for target in targets:
                    start = max(0, target - self.clip_length + 1)
                    indices = tuple(range(start, target + 1))
                    padding = (indices[0],) * (self.clip_length - len(indices))
                    self.clips.append((sequence_index, padding + indices))
            elif length >= self.clip_length:
                starts = list(range(0, length - self.clip_length + 1, self.stride))
                last_start = length - self.clip_length
                if starts[-1] != last_start:
                    starts.append(last_start)
                for start in starts:
                    self.clips.append(
                        (sequence_index, tuple(range(start, start + self.clip_length)))
                    )
        if not self.clips:
            raise ValueError("manifest produced no usable clips")

    def _resolve(self, value: object) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def _load_manifest(self) -> List[_SequenceRecord]:
        records: List[_SequenceRecord] = []
        with self.manifest.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON on {self.manifest}:{line_number}: {error}"
                    ) from error
                if not isinstance(payload, dict):
                    raise ValueError(f"manifest line {line_number} must be a JSON object")
                name = str(payload.get("sequence", payload.get("name", f"line_{line_number}")))

                if "frames" in payload:
                    frames = payload["frames"]
                    if not isinstance(frames, list) or not frames:
                        raise ValueError(f"sequence {name!r} has no frames")
                    blurry_values: List[object] = []
                    sharp_values: List[object] = []
                    teacher_values: List[object] = []
                    teacher_complete = True
                    for frame in frames:
                        if not isinstance(frame, dict):
                            raise ValueError(f"sequence {name!r} contains a non-object frame")
                        blurry_values.append(_first(frame, ("blurry", "blur", "input", "lq")))
                        sharp_values.append(_first(frame, ("sharp", "target", "gt")))
                        teacher_value = _optional_first(frame, ("teacher", "single_frame_teacher"))
                        if teacher_value is None:
                            teacher_complete = False
                        else:
                            teacher_values.append(teacher_value)
                else:
                    blurry_object = _first(payload, ("blurry", "blur", "input", "lq"))
                    sharp_object = _first(payload, ("sharp", "target", "gt"))
                    if not isinstance(blurry_object, list) or not isinstance(sharp_object, list):
                        raise ValueError(f"sequence {name!r} paths must be arrays")
                    blurry_values = list(blurry_object)
                    sharp_values = list(sharp_object)
                    teacher_object = _optional_first(payload, ("teacher", "single_frame_teacher"))
                    teacher_complete = teacher_object is not None
                    if teacher_object is not None:
                        if not isinstance(teacher_object, list):
                            raise ValueError(f"sequence {name!r} teacher paths must be an array")
                        teacher_values = list(teacher_object)
                    else:
                        teacher_values = []

                if not blurry_values:
                    raise ValueError(f"sequence {name!r} has no frames")
                if len(blurry_values) != len(sharp_values):
                    raise ValueError(f"sequence {name!r} has mismatched blurry/sharp lengths")
                if teacher_complete and len(teacher_values) != len(blurry_values):
                    raise ValueError(f"sequence {name!r} has incomplete teacher frames")

                record = _SequenceRecord(
                    name=name,
                    blurry=tuple(self._resolve(value) for value in blurry_values),
                    sharp=tuple(self._resolve(value) for value in sharp_values),
                    teacher=(
                        tuple(self._resolve(value) for value in teacher_values)
                        if teacher_complete
                        else None
                    ),
                )
                for path in record.blurry + record.sharp + (record.teacher or ()):
                    if not path.is_file():
                        raise FileNotFoundError(f"missing frame in sequence {name!r}: {path}")
                records.append(record)
        if not records:
            raise ValueError(f"manifest contains no sequences: {self.manifest}")
        return records

    @staticmethod
    def _read_rgb(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _shared_transform(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        height, width = tensors[0].shape[-2:]
        for tensor in tensors:
            if tensor.shape != tensors[0].shape:
                raise ValueError("every image in a clip must have the same shape")

        if self.crop_size:
            if height < self.crop_size or width < self.crop_size:
                raise ValueError(
                    f"crop_size={self.crop_size} exceeds image size {height}x{width}"
                )
            if self.augment:
                top = int(torch.randint(0, height - self.crop_size + 1, (1,)).item())
                left = int(torch.randint(0, width - self.crop_size + 1, (1,)).item())
            else:
                top = (height - self.crop_size) // 2
                left = (width - self.crop_size) // 2
            tensors = [
                tensor[:, top : top + self.crop_size, left : left + self.crop_size]
                for tensor in tensors
            ]

        if self.augment and bool(torch.randint(0, 2, (1,)).item()):
            tensors = [tensor.flip(-1) for tensor in tensors]
        if self.augment and bool(torch.randint(0, 2, (1,)).item()):
            tensors = [tensor.flip(-2) for tensor in tensors]
        if self.augment:
            # Match EVSSM/BasicSR's paired-image use_rot augmentation while
            # applying exactly the same transform to every temporal frame and
            # modality in the clip.
            quarter_turns = int(torch.randint(0, 4, (1,)).item())
            if quarter_turns:
                tensors = [torch.rot90(tensor, quarter_turns, (-2, -1)) for tensor in tensors]
        return tensors

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sequence_index, frame_indices = self.clips[index]
        record = self.sequences[sequence_index]
        blurry = [self._read_rgb(record.blurry[i]) for i in frame_indices]
        sharp = [self._read_rgb(record.sharp[i]) for i in frame_indices]
        has_teacher = record.teacher is not None
        teacher = (
            [self._read_rgb(record.teacher[i]) for i in frame_indices]
            if record.teacher is not None
            else [torch.zeros_like(image) for image in blurry]
        )
        transformed = self._shared_transform(blurry + sharp + teacher)
        length = len(frame_indices)
        blurry_tensor = torch.stack(transformed[:length])
        sharp_tensor = torch.stack(transformed[length : length * 2])
        teacher_tensor = torch.stack(transformed[length * 2 :])
        frame_indices_tensor = torch.tensor(frame_indices, dtype=torch.long)
        # A repeated left-prefix frame is context padding, not a physical
        # transition.  Likewise, any future non-unit manifest jump must fail
        # closed instead of silently becoming motion supervision.  For H1 the
        # slice is deliberately an empty bool tensor with shape [0].
        transition_valid = (
            frame_indices_tensor[1:] == frame_indices_tensor[:-1] + 1
        )
        return {
            "blurry": blurry_tensor,
            "sharp": sharp_tensor,
            "teacher": teacher_tensor,
            "has_teacher": torch.tensor(has_teacher, dtype=torch.bool),
            "sequence": record.name,
            "frame_indices": frame_indices_tensor,
            "transition_valid": transition_valid,
            "start": torch.tensor(frame_indices[0], dtype=torch.long),
            "target_index": torch.tensor(frame_indices[-1], dtype=torch.long),
            "prefix_padding": torch.tensor(
                self.clip_length - len(set(frame_indices[: self.clip_length])),
                dtype=torch.long,
            ),
        }
