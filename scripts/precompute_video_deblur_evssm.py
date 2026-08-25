#!/usr/bin/env python3
"""Precompute frozen EVSSM outputs for ordered video-deblur JSONL data.

The output JSONL preserves blurry/sharp paths and adds a complete ``teacher``
array.  Every artifact is content-bound in ``precompute_report.json`` so an
EVSSM-domain causal adapter can be trained without rerunning the expensive
single-frame teacher every epoch.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_arrays(payload: dict) -> tuple[list[str], list[str]]:
    if "frames" in payload:
        frames = payload["frames"]
        if not isinstance(frames, list) or not frames:
            raise ValueError("frames must be a non-empty array")
        blurry = [str(frame.get("blurry", frame.get("blur", frame.get("input", "")))) for frame in frames]
        sharp = [str(frame.get("sharp", frame.get("target", frame.get("gt", "")))) for frame in frames]
    else:
        blurry = [str(value) for value in payload.get("blurry", payload.get("blur", []))]
        sharp = [str(value) for value in payload.get("sharp", payload.get("target", []))]
    if not blurry or len(blurry) != len(sharp) or any(not value for value in blurry + sharp):
        raise ValueError("each sequence requires equal non-empty blurry/sharp paths")
    return blurry, sharp


def _resolve(path: str, base: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _save_rgb(path: Path, tensor: torch.Tensor) -> None:
    array = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(array * 255.0).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    manifest = args.input_manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not manifest.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("input manifest and EVSSM checkpoint must exist")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = args.data_root.expanduser().resolve() if args.data_root else manifest.parent
    device = torch.device(args.device)

    from thirdparty.EVSSM.models.EVSSM import EVSSM

    model = EVSSM().to(device).eval()
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("params", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("EVSSM checkpoint must contain a params state dict")
    model.load_state_dict(state, strict=True)

    records = []
    report_frames = []
    cache: dict[Path, Path] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    for sequence_index, line in enumerate(lines):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("manifest entries must be JSON objects")
        blurry_values, sharp_values = _sequence_arrays(payload)
        teacher_values = []
        for frame_index, blurry_value in enumerate(blurry_values):
            blurry_path = _resolve(blurry_value, base)
            sharp_path = _resolve(sharp_values[frame_index], base)
            teacher_path = cache.get(blurry_path)
            if teacher_path is None:
                teacher_path = output_dir / "teacher" / f"seq_{sequence_index:04d}" / f"{frame_index:06d}.png"
                image = _load_rgb(blurry_path, device)
                height, width = image.shape[-2:]
                h_pad = (4 - height % 4) % 4
                w_pad = (4 - width % 4) % 4
                padded = F.pad(image, (0, w_pad, 0, h_pad), mode="reflect") if (h_pad or w_pad) else image
                with torch.no_grad():
                    restored = model(padded)[:, :, :height, :width]
                _save_rgb(teacher_path, restored)
                cache[blurry_path] = teacher_path
            teacher_values.append(str(teacher_path))
            report_frames.append(
                {
                    "sequence_index": sequence_index,
                    "frame_index": frame_index,
                    "blurry": str(blurry_path),
                    "blurry_sha256": _sha256(blurry_path),
                    "sharp": str(sharp_path),
                    "sharp_sha256": _sha256(sharp_path),
                    "teacher": str(teacher_path),
                    "teacher_sha256": _sha256(teacher_path),
                }
            )
        records.append(
            {
                "sequence": str(payload.get("sequence", payload.get("name", f"sequence_{sequence_index}"))),
                "blurry": [str(_resolve(value, base)) for value in blurry_values],
                "sharp": [str(_resolve(value, base)) for value in sharp_values],
                "teacher": teacher_values,
                "teacher_kind": "frozen_unblur_slam_evssm",
            }
        )

    output_manifest = output_dir / "sequences_with_evssm.jsonl"
    with output_manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    report = {
        "schema": "unblur_slam.video_deblur_evssm_precompute.v1",
        "input_manifest": str(manifest),
        "input_manifest_sha256": _sha256(manifest),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": _sha256(output_manifest),
        "sequence_count": len(records),
        "frame_count": len(report_frames),
        "frames": report_frames,
    }
    with (output_dir / "precompute_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in ("output_manifest", "sequence_count", "frame_count", "checkpoint_sha256")}))


if __name__ == "__main__":
    main()
