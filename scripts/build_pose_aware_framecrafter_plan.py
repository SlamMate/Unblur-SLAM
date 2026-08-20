#!/usr/bin/env python3
"""Build a pose/blur-aware interpolation plan for an external FrameCrafter run.

Input CSV columns:
  frame,timestamp,tx,ty,tz,qx,qy,qz,qw[,laplacian]

This script only plans synthetic views. It never treats generated frames as
ground truth and never invokes a generation model.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def percentile(values, q):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("Cannot compute percentile of an empty sequence")
    pos = min(1.0, max(0.0, float(q))) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    alpha = pos - lo
    return ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha


def normalize_quaternion(row):
    q = [float(row[key]) for key in ("qx", "qy", "qz", "qw")]
    norm = math.sqrt(sum(v * v for v in q))
    if norm <= 1e-12:
        raise ValueError(f"Invalid zero quaternion for frame {row['frame']}")
    return [v / norm for v in q]


def rotation_delta_deg(q0, q1):
    dot = abs(sum(a * b for a, b in zip(q0, q1)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def laplacian_energy(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read frame for Laplacian score: {path}")
    return float(np.abs(cv2.Laplacian(image, cv2.CV_64F)).mean())


def load_rows(csv_path, image_root):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"frame", "timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"}
    if not rows:
        raise ValueError(f"No frames found in {csv_path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    for row in rows:
        row["translation"] = [float(row[key]) for key in ("tx", "ty", "tz")]
        row["quaternion"] = normalize_quaternion(row)
        if row.get("laplacian", "").strip():
            row["laplacian_value"] = float(row["laplacian"])
        else:
            if image_root is None:
                raise ValueError("CSV lacks laplacian values; provide --image-root")
            row["laplacian_value"] = laplacian_energy(image_root / row["frame"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--laplacian-threshold", type=float)
    parser.add_argument("--blur-quantile", type=float, default=0.30)
    parser.add_argument("--translation-step", type=float, default=0.08)
    parser.add_argument("--rotation-step-deg", type=float, default=6.0)
    parser.add_argument("--blur-region-inserts", type=int, default=1)
    parser.add_argument("--max-inserts", type=int, default=4)
    args = parser.parse_args()

    rows = load_rows(args.frames_csv, args.image_root)
    lap_values = [row["laplacian_value"] for row in rows]
    threshold = (
        float(args.laplacian_threshold)
        if args.laplacian_threshold is not None
        else percentile(lap_values, args.blur_quantile)
    )
    for row in rows:
        row["is_blurry"] = bool(row["laplacian_value"] < threshold)

    segments = []
    for left, right in zip(rows[:-1], rows[1:]):
        translation = math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(left["translation"], right["translation"])
        ))
        rotation = rotation_delta_deg(left["quaternion"], right["quaternion"])
        pose_ratio = max(
            translation / max(args.translation_step, 1e-12),
            rotation / max(args.rotation_step_deg, 1e-12),
        )
        pose_inserts = max(0, int(math.ceil(pose_ratio)) - 1)
        blur_inserts = max(0, args.blur_region_inserts) if left["is_blurry"] and right["is_blurry"] else 0
        inserts = min(max(0, args.max_inserts), max(pose_inserts, blur_inserts))
        if inserts <= 0:
            continue
        reasons = []
        if blur_inserts:
            reasons.append("consecutive_blurry_region")
        if pose_inserts:
            reasons.append("large_pose_gap")
        segments.append({
            "left_frame": left["frame"],
            "right_frame": right["frame"],
            "left_timestamp": left["timestamp"],
            "right_timestamp": right["timestamp"],
            "insert_count": inserts,
            "alphas": [(index + 1) / float(inserts + 1) for index in range(inserts)],
            "reasons": reasons,
            "left_laplacian": left["laplacian_value"],
            "right_laplacian": right["laplacian_value"],
            "translation_delta": translation,
            "rotation_delta_deg": rotation,
        })

    payload = {
        "schema": "unblur_slam.pose_aware_framecrafter_plan.v1",
        "source_csv": str(args.frames_csv.resolve()),
        "frame_count": len(rows),
        "blur_threshold": threshold,
        "blur_frame_count": sum(int(row["is_blurry"]) for row in rows),
        "synthetic_frame_count": sum(item["insert_count"] for item in segments),
        "policy": {
            "translation_step": args.translation_step,
            "rotation_step_deg": args.rotation_step_deg,
            "blur_region_inserts": args.blur_region_inserts,
            "max_inserts": args.max_inserts,
            "generated_frames_are_ground_truth": False,
            "required_post_gate": [
                "temporal_consistency",
                "pose_reprojection_consistency",
                "sharpness_gain",
            ],
        },
        "segments": segments,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps({
        "frames": payload["frame_count"],
        "blur_frames": payload["blur_frame_count"],
        "segments": len(segments),
        "synthetic_frames": payload["synthetic_frame_count"],
        "output": str(args.output_json),
    }))


if __name__ == "__main__":
    main()
