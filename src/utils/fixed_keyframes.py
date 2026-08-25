"""Fail-closed helpers for preregistered fixed-keyframe ablations.

The schedule is expressed in the dataset's original ``source_index`` domain.
It is deliberately configuration-only: no run output, evaluation image, or
ground-truth pose is consulted while either arm is executing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


FIXED_SOURCE_KEYFRAME_SCHEMA = "unblur_slam.fixed_source_keyframes.v1"


def parse_fixed_source_keyframe_contract(
    cfg: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate and normalize ``tracking.fixed_source_keyframes``.

    Missing or explicitly disabled contracts preserve the original dynamic
    keyframe policy.  Enabled contracts are intentionally strict so a typo
    cannot silently turn a controlled ablation back into a dynamic run.
    """

    tracking = cfg.get("tracking", {}) or {}
    raw = tracking.get("fixed_source_keyframes")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("tracking.fixed_source_keyframes must be a mapping")
    if raw.get("enabled") is not True:
        if raw.get("enabled") is False:
            return None
        raise ValueError("tracking.fixed_source_keyframes.enabled must be boolean")
    if raw.get("schema") != FIXED_SOURCE_KEYFRAME_SCHEMA:
        raise ValueError("fixed-source-keyframe schema is missing or unsupported")
    if raw.get("coordinate_domain") != "dataset_source_index":
        raise ValueError(
            "fixed source keyframes must use coordinate_domain=dataset_source_index"
        )
    if raw.get("strict_exact") is not True:
        raise ValueError("fixed source keyframes require strict_exact=true")
    if raw.get("runtime_baseline_artifact_dependency") is not False:
        raise ValueError(
            "fixed source keyframes must not read a baseline artifact at runtime"
        )
    if raw.get("uses_ground_truth_poses") is not False:
        raise ValueError("fixed source keyframes must not use ground-truth poses")

    selection_source = str(raw.get("selection_source", "")).strip()
    if not selection_source:
        raise ValueError("fixed source keyframes require a declared selection_source")
    values = raw.get("source_indices")
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray))
        or not values
    ):
        raise ValueError("fixed source keyframes require a non-empty source_indices list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("fixed source keyframe indices must be integers")
    indices = tuple(int(value) for value in values)
    if indices[0] != 0:
        raise ValueError("fixed source keyframes must include source frame 0 first")
    if any(value < 0 for value in indices):
        raise ValueError("fixed source keyframe indices must be non-negative")
    if tuple(sorted(set(indices))) != indices:
        raise ValueError("fixed source keyframe indices must be strictly increasing")

    return {
        "schema": FIXED_SOURCE_KEYFRAME_SCHEMA,
        "coordinate_domain": "dataset_source_index",
        "strict_exact": True,
        "runtime_baseline_artifact_dependency": False,
        "uses_ground_truth_poses": False,
        "selection_source": selection_source,
        "source_indices": indices,
        "source_index_set": frozenset(indices),
    }


def assert_exact_fixed_source_keyframes(
    expected: Sequence[int], actual: Sequence[int]
) -> None:
    """Raise when a completed run did not preserve the preregistered schedule."""

    expected_tuple = tuple(int(value) for value in expected)
    actual_tuple = tuple(int(value) for value in actual)
    if actual_tuple != expected_tuple:
        raise RuntimeError(
            "fixed-source-keyframe contract violated: "
            f"expected {list(expected_tuple)}, got {list(actual_tuple)}"
        )
