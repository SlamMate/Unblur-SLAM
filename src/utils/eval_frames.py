"""Single source of truth for paper-aligned clear-GT evaluation frames."""

from pathlib import Path
from typing import Optional, Set


FULL_PAPER_SCOPE = "full_paper"
PREFIX_SMOKE_SCOPE = "prefix_smoke"
FULL_PAPER_METRIC_SCOPE = "clear_gt_only"
PREFIX_SMOKE_METRIC_SCOPE = "clear_gt_prefix_smoke"
_VALID_CLEAR_GT_SCOPES = {FULL_PAPER_SCOPE, PREFIX_SMOKE_SCOPE}


_TUM_INDEX_FILES = {
    "freiburg1_desk": "fr1_desk_indices_sharp.txt",
    "fr1_desk": "fr1_desk_indices_sharp.txt",
    "freiburg2_xyz": "fr2_xyz_indices_sharp.txt",
    "fr2_xyz": "fr2_xyz_indices_sharp.txt",
    "freiburg3_office": "fr3_office_indices_sharp.txt",
    "fr3_office": "fr3_office_indices_sharp.txt",
}

_HOLD_DATASETS = {
    "deblur_nerf_motion",
    "deblur_nerf_defocus",
    "exblurf_motion",
    "real_camera_motion_blur",
    "deblur_nerf_motion_whole",
    "deblur_nerf_motion_no_deblur_no_refine",
}


def _tum_index_file(scene: str) -> Optional[Path]:
    scene = scene.lower()
    filename = next(
        (value for token, value in _TUM_INDEX_FILES.items() if token in scene),
        None,
    )
    if filename is None:
        return None
    return Path(__file__).resolve().parents[2] / "scripts" / filename


def clear_gt_source_indices(config, frame_reader=None) -> Optional[Set[int]]:
    """Resolve original-frame indices used by the paper's image metrics.

    Generated views are deliberately not represented here.  For dataset
    families that advertise a clear-GT protocol, a missing/malformed protocol
    file is an error rather than permission to silently evaluate all frames.
    """

    dataset = str(config.get("dataset", "")).lower()
    scene = str(config.get("scene", "")).lower()

    if dataset in {"tumrgbd", "tumrgb"}:
        index_file = _tum_index_file(scene)
        if index_file is None:
            return None
        if not index_file.is_file():
            raise FileNotFoundError(
                f"paper clear-GT index file does not exist: {index_file}"
            )
        indices = {
            int(line.strip())
            for line in index_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not indices:
            raise ValueError(f"paper clear-GT index file is empty: {index_file}")
        return indices

    if dataset not in _HOLD_DATASETS:
        return None

    data = config.get("data", {}) or {}
    scene_dir = (
        Path(data.get("dataset_root", "")).expanduser()
        / str(data.get("input_folder", ""))
        / str(config.get("scene", ""))
    ).resolve()
    hold_files = sorted(scene_dir.glob("hold=*"))
    if len(hold_files) != 1:
        raise FileNotFoundError(
            "expected exactly one hold=X clear-GT protocol file in "
            f"{scene_dir}, found {len(hold_files)}"
        )
    try:
        hold = int(hold_files[0].name.split("=", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid hold protocol filename: {hold_files[0]}") from error
    if hold < 1:
        raise ValueError(f"hold interval must be positive: {hold_files[0]}")
    if frame_reader is None:
        raise ValueError("frame_reader is required to resolve hold=X indices")
    if hasattr(frame_reader, "original_frame_count"):
        frame_count = int(frame_reader.original_frame_count)
    else:
        frame_count = len(frame_reader)
    return set(range(0, frame_count, hold))


def available_clear_gt_source_indices(config, frame_reader) -> Optional[Set[int]]:
    """Return the complete paper protocol subset available in this run.

    ``clear_gt_source_indices`` defines the protocol in original-source index
    space.  An augmented FrameCrafter stream is longer than the source video,
    so filtering with ``len(frame_reader)`` would incorrectly make generated
    views look like additional source frames.  This helper is the sole place
    where the protocol is clipped to the configured source sequence.
    """

    indices = clear_gt_source_indices(config, frame_reader)
    if indices is None:
        return None
    if frame_reader is None:
        raise ValueError("frame_reader is required to resolve available clear-GT frames")
    frame_count = (
        int(frame_reader.original_frame_count)
        if hasattr(frame_reader, "original_frame_count")
        else len(frame_reader)
    )
    available = {int(index) for index in indices if 0 <= int(index) < frame_count}
    if not available:
        raise ValueError(
            "paper clear-GT protocol has no frame inside the configured source range"
        )
    return available


def clear_gt_scope_mode(config) -> str:
    """Return the configured clear-GT policy, rejecting unknown modes.

    ``full_paper`` is the default and retains the historical fail-closed
    behavior.  ``prefix_smoke`` is reserved for explicitly bounded TUM
    functionality runs and never carries the paper metric label.
    """

    evaluation = config.get("evaluation", {}) or {}
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a mapping")
    mode = str(evaluation.get("clear_gt_scope", FULL_PAPER_SCOPE)).lower()
    if mode not in _VALID_CLEAR_GT_SCOPES:
        raise ValueError(
            "evaluation.clear_gt_scope must be full_paper or prefix_smoke"
        )
    return mode


def clear_gt_metric_scope(config) -> str:
    """Return the JSON metric label corresponding to the validated policy."""

    mode = clear_gt_scope_mode(config)
    if mode == PREFIX_SMOKE_SCOPE:
        return PREFIX_SMOKE_METRIC_SCOPE
    return FULL_PAPER_METRIC_SCOPE


def validate_clear_gt_protocol_scope(config, frame_reader) -> Optional[Set[int]]:
    """Validate full-paper or explicitly bounded prefix-smoke coverage.

    The authoritative membership always comes from
    :func:`available_clear_gt_source_indices`.  A prefix-smoke config must
    additionally state the expected indices as an audit assertion; it cannot
    use that list to introduce or remove evaluation frames.
    """

    mode = clear_gt_scope_mode(config)
    available = available_clear_gt_source_indices(config, frame_reader)
    if available is None:
        if mode == PREFIX_SMOKE_SCOPE:
            raise ValueError(
                "evaluation.clear_gt_scope=prefix_smoke requires a TUM "
                "published clear-GT protocol"
            )
        return None
    if int(config.get("stride", 1)) != 1:
        raise ValueError(
            "clear-GT evaluation requires stride=1 so source indices remain "
            "aligned with the published protocol"
        )

    protocol = clear_gt_source_indices(config, frame_reader)
    if protocol is None:
        raise AssertionError("available clear-GT frames require a protocol")
    protocol_sorted = sorted(int(index) for index in protocol)
    available_sorted = sorted(int(index) for index in available)

    if mode == FULL_PAPER_SCOPE:
        if set(available_sorted) != set(protocol_sorted):
            missing = sorted(set(protocol_sorted) - set(available_sorted))
            raise ValueError(
                "configured max_frames/source data truncates the published "
                f"clear-GT protocol; missing source indices {missing}"
            )
        return set(available_sorted)

    dataset = str(config.get("dataset", "")).lower()
    if dataset not in {"tumrgbd", "tumrgb"}:
        raise ValueError("prefix_smoke is restricted to published TUM protocols")
    if len(available_sorted) < 2:
        raise ValueError("prefix_smoke requires at least two clear-GT frames")
    if len(available_sorted) >= len(protocol_sorted):
        raise ValueError(
            "prefix_smoke must be a proper prefix; use full_paper for the "
            "complete protocol"
        )
    if available_sorted != protocol_sorted[: len(available_sorted)]:
        raise ValueError(
            "prefix_smoke available frames must form the exact leading prefix "
            "of the published clear-GT protocol"
        )

    evaluation = config.get("evaluation", {}) or {}
    asserted = evaluation.get("expected_clear_gt_source_indices")
    if not isinstance(asserted, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in asserted
    ):
        raise ValueError(
            "prefix_smoke requires evaluation.expected_clear_gt_source_indices "
            "as an ordered integer list"
        )
    if asserted != available_sorted:
        raise ValueError(
            "evaluation.expected_clear_gt_source_indices does not match the "
            f"authoritative available protocol prefix: expected {available_sorted}, "
            f"got {asserted}"
        )
    return set(available_sorted)
