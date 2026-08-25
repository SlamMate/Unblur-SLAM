#!/usr/bin/env python3
"""Run a paired init-vs-refined smoke test with the official cvg/ReSplat model.

The comparison is paired by construction: the official encoder is called once,
the resulting initial Gaussian object is rendered, and that exact same Python
object is then passed to ``encoder.forward_update`` for four official recurrent
updates.  Both states are rendered at the same target views and evaluated with
the official ``scripts/infer_colmap.py`` helpers.

This is a standalone evaluator.  It neither changes the official ReSplat
checkout nor injects ReSplat-like logic into the Unblur-SLAM mapper.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from offline_fair_gpu_contract import validate_gpu_contract


SCHEMA = "unblur_slam.paired_official_resplat_smoke.v1"
METRIC_REFERENCE_SCHEMA = "unblur_slam.offline_fair_metric_references.v1"
OFFICIAL_RESPLAT_URL = "https://github.com/cvg/resplat"
DEFAULT_PRESET = "dl3dv_8v_256x448_small"
PAIRED_NUM_REFINE = 4
PAIRED_TARGET_COUNT = 34
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path | str) -> str:
    """Return a streaming SHA-256 digest without loading large checkpoints."""

    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"cannot inspect official ReSplat repository: {repo}") from error
    return result.stdout.strip()


def _normalize_git_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized == "git@github.com:cvg/resplat":
        normalized = OFFICIAL_RESPLAT_URL
    return normalized.lower()


def inspect_official_repo(repo: Path | str) -> dict[str, Any]:
    """Fail closed unless *repo* is a clean tracked checkout of cvg/resplat."""

    root = Path(repo).expanduser().resolve()
    infer_script = root / "scripts" / "infer_colmap.py"
    model_zoo = root / "MODEL_ZOO.md"
    if not infer_script.is_file() or not model_zoo.is_file():
        raise ValueError(
            "ReSplat repository lacks scripts/infer_colmap.py or MODEL_ZOO.md: "
            f"{root}"
        )
    origin = _git(root, "remote", "get-url", "origin")
    if _normalize_git_url(origin) != _normalize_git_url(OFFICIAL_RESPLAT_URL):
        raise ValueError(f"ReSplat origin is not official cvg/resplat: {origin}")
    commit = _git(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"invalid ReSplat commit id: {commit!r}")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError("official ReSplat checkout has tracked modifications")
    return {
        "path": str(root),
        "origin": origin,
        "expected_origin": OFFICIAL_RESPLAT_URL,
        "commit": commit,
        "tracked_worktree_clean": True,
        "infer_colmap_path": str(infer_script),
        "infer_colmap_sha256": sha256_file(infer_script),
        "model_zoo_path": str(model_zoo),
        "model_zoo_sha256": sha256_file(model_zoo),
    }


def import_official_infer(repo: Path | str) -> ModuleType:
    """Import the pinned official inference script without copying its code."""

    root = Path(repo).expanduser().resolve()
    script = root / "scripts" / "infer_colmap.py"
    # cvg/resplat deliberately uses a PEP-420 ``src`` namespace (there is no
    # src/__init__.py).  If this runner is launched from the Unblur-SLAM repo,
    # its regular ``src`` package would otherwise win Python's import search
    # even though the official root is first on sys.path.  Remove only search
    # roots that expose a competing regular package; keep stdlib/site-packages.
    existing_src = sys.modules.get("src")
    if existing_src is not None:
        namespace_paths = [
            Path(value).resolve() for value in getattr(existing_src, "__path__", ())
        ]
        if (root / "src").resolve() not in namespace_paths:
            raise RuntimeError(
                "a non-ReSplat 'src' package is already imported; launch the paired "
                "runner as a fresh standalone Python process"
            )
    filtered_path: list[str] = []
    for entry in sys.path:
        candidate = Path(entry or os.getcwd()).resolve()
        if candidate == root:
            continue
        if (candidate / "src" / "__init__.py").is_file():
            continue
        filtered_path.append(entry)
    sys.path[:] = [str(root), *filtered_path]
    module_name = "_official_cvg_resplat_infer_colmap"
    specification = importlib.util.spec_from_file_location(module_name, script)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import official inference script: {script}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _load_scene_manifest(scene_path: Path, manifest_path: Optional[Path]) -> dict[str, Any]:
    source = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else (scene_path / "manifest.json").resolve()
    )
    if not source.is_file():
        raise FileNotFoundError(f"scene manifest not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid scene manifest JSON: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError("scene manifest root must be a JSON object")
    schema = str(payload.get("schema", ""))
    if not schema.startswith("unblur_slam.official_resplat_colmap_scene."):
        raise ValueError(f"not an audited official-ReSplat scene manifest: {schema!r}")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("scene manifest has no frames")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "schema": schema,
        "payload": payload,
    }


def _resolve_checkpoint(
    *,
    repo: Path,
    official: ModuleType,
    preset_name: str,
    checkpoint: Optional[Path],
    expected_sha256: Optional[str],
) -> tuple[Path, str, Mapping[str, Any]]:
    try:
        preset = dict(official.MODEL_PRESETS[preset_name])
    except KeyError as error:
        raise ValueError(f"unknown official ReSplat preset: {preset_name}") from error
    if int(preset["num_refine"]) != PAIRED_NUM_REFINE:
        raise ValueError(
            f"paired_refine4 requires an official four-update preset; {preset_name} "
            f"declares {preset['num_refine']}"
        )
    source = (
        checkpoint.expanduser().resolve()
        if checkpoint is not None
        else (repo / str(preset["checkpoint"])).resolve()
    )
    if not source.is_file():
        raise FileNotFoundError(f"official ReSplat checkpoint not found: {source}")
    digest = sha256_file(source)
    filename_match = re.search(r"-([0-9a-f]{8})\.pth$", source.name)
    if filename_match is None or not digest.startswith(filename_match.group(1)):
        raise ValueError(
            "checkpoint bytes do not match the SHA-256 prefix in the official filename: "
            f"{source.name} -> {digest}"
        )
    if expected_sha256 is not None:
        expected = expected_sha256.strip().lower()
        if not SHA256_RE.fullmatch(expected):
            raise ValueError("--expected-checkpoint-sha256 must be 64 lowercase hex digits")
        if digest != expected:
            raise ValueError(
                f"checkpoint SHA-256 mismatch: expected {expected}, got {digest}"
            )
    return source, digest, preset


def _tensor_versions(gaussians: object) -> dict[str, int]:
    """Record tensor version counters to detect in-place mutation of init state."""

    result: dict[str, int] = {}
    for name in (
        "means",
        "covariances",
        "harmonics",
        "opacities",
        "scales",
        "rotations",
        "rotations_unnorm",
    ):
        value = getattr(gaussians, name, None)
        version = getattr(value, "_version", None)
        if version is not None:
            result[name] = int(version)
    if not result:
        raise RuntimeError("official encoder returned no auditable Gaussian tensors")
    return result


def render_target_views(
    torch_module: Any,
    decoder: object,
    gaussians: object,
    batch: Mapping[str, Any],
    render_chunk_size: int,
) -> tuple[Any, Any]:
    """Render the batch target views through the official decoder, in chunks."""

    _, _, _, height, width = batch["target"]["image"].shape
    count = int(batch["target"]["extrinsics"].shape[1])
    colors = []
    depths = []
    for start in range(0, count, render_chunk_size):
        end = min(start + render_chunk_size, count)
        output = decoder.forward(
            gaussians,
            batch["target"]["extrinsics"][:, start:end],
            batch["target"]["intrinsics"][:, start:end],
            batch["target"]["near"][:, start:end],
            batch["target"]["far"][:, start:end],
            (height, width),
            depth_mode=None,
        )
        colors.append(output.color[0])
        depths.append(output.depth[0])
    return torch_module.cat(colors, dim=0), torch_module.cat(depths, dim=0)


StageRunner = Callable[[str, Callable[[], Any]], Any]


def run_paired_inference_core(
    *,
    torch_module: Any,
    encoder: object,
    decoder: object,
    batch: Mapping[str, Any],
    num_refine: int,
    render_chunk_size: int,
    stage_runner: StageRunner,
    visualization_dump: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Execute exactly one official initialization and one recurrent update call."""

    if num_refine != PAIRED_NUM_REFINE:
        raise ValueError(f"paired experiment requires num_refine={PAIRED_NUM_REFINE}")

    initial_output = stage_runner(
        "encoder_init0",
        lambda: encoder(
            batch["context"],
            global_step=0,
            deterministic=False,
            visualization_dump=visualization_dump,
        ),
    )
    if not isinstance(initial_output, dict):
        raise RuntimeError(
            "four-update official encoder must return gaussians and condition_features"
        )
    init_gaussians = initial_output.get("gaussians")
    condition_features = initial_output.get("condition_features")
    if init_gaussians is None or condition_features is None:
        raise RuntimeError("official encoder omitted gaussians or condition_features")
    versions_after_encoder = _tensor_versions(init_gaussians)

    init_rendered, init_depth = stage_runner(
        "render_init0",
        lambda: render_target_views(
            torch_module, decoder, init_gaussians, batch, render_chunk_size
        ),
    )
    versions_after_init_render = _tensor_versions(init_gaussians)
    if versions_after_init_render != versions_after_encoder:
        raise RuntimeError("official decoder mutated the initial Gaussian state in place")

    # Contract-critical call: pass the exact object returned by the sole encoder
    # forward pass.  Do not clone, reconstruct, reload, or re-run initialization.
    # The pinned official update needs only target geometry for optional renders;
    # removing RGB here makes metric references provably unavailable to it.
    target_geometry = {
        key: value for key, value in batch["target"].items() if key != "image"
    }
    refine_output = stage_runner(
        "forward_update_refine4",
        lambda: encoder.forward_update(
            batch["context"],
            target_geometry,
            condition_features,
            init_gaussians,
            decoder,
            None,
        ),
    )
    versions_after_update = _tensor_versions(init_gaussians)
    if versions_after_update != versions_after_encoder:
        raise RuntimeError("official forward_update mutated the initial Gaussian state in place")
    try:
        refined_sequence = refine_output["gaussian"]
        refined_gaussians = refined_sequence[-1]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("official forward_update returned no refined Gaussian state") from error
    if len(refined_sequence) != num_refine:
        raise RuntimeError(
            "official forward_update iteration count mismatch: "
            f"expected {num_refine}, got {len(refined_sequence)}"
        )

    refined_rendered, refined_depth = stage_runner(
        "render_refine4",
        lambda: render_target_views(
            torch_module, decoder, refined_gaussians, batch, render_chunk_size
        ),
    )
    return {
        "init_gaussians": init_gaussians,
        "refined_gaussians": refined_gaussians,
        "init_rendered": init_rendered,
        "refined_rendered": refined_rendered,
        "init_depth": init_depth,
        "refined_depth": refined_depth,
        "visualization_dump": visualization_dump,
        "contract": {
            "encoder_forward_calls": 1,
            "forward_update_calls": 1,
            "init_object_passed_directly_to_forward_update": True,
            "initial_state_tensor_versions": versions_after_encoder,
            "versions_after_init_render": versions_after_init_render,
            "versions_after_forward_update": versions_after_update,
            "initial_state_in_place_mutation_detected": False,
            "target_rgb_passed_to_forward_update": False,
        },
    }


class CudaStageRecorder:
    """Measure synchronized core GPU stages with CUDA events and allocator peaks."""

    def __init__(self, torch_module: Any, device: object) -> None:
        self.torch = torch_module
        self.device = device
        self.records: dict[str, dict[str, Any]] = {}
        self.process_peak_allocated_bytes = int(
            torch_module.cuda.max_memory_allocated(device)
        )

    def __call__(self, name: str, operation: Callable[[], Any]) -> Any:
        torch = self.torch
        torch.cuda.synchronize(self.device)
        # Preserve the process-scope maximum before resetting the allocator's
        # stage counter.  This captures temporary model/checkpoint-load peaks
        # that occurred before the first recurrent stage.
        self.process_peak_allocated_bytes = max(
            self.process_peak_allocated_bytes,
            int(torch.cuda.max_memory_allocated(self.device)),
        )
        torch.cuda.reset_peak_memory_stats(self.device)
        baseline = int(torch.cuda.memory_allocated(self.device))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        torch.cuda.synchronize(self.device)
        peak = int(torch.cuda.max_memory_allocated(self.device))
        final = int(torch.cuda.memory_allocated(self.device))
        self.process_peak_allocated_bytes = max(
            self.process_peak_allocated_bytes, peak
        )
        self.records[name] = {
            "elapsed_ms": round(float(start.elapsed_time(end)), 3),
            "allocated_before_bytes": baseline,
            "allocated_after_bytes": final,
            "peak_allocated_bytes": peak,
            "peak_increment_over_baseline_bytes": max(0, peak - baseline),
        }
        return result


def _configure_reproducibility(torch: Any, numpy: Any, seed: int) -> dict[str, Any]:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # gsplat/CUDA may not provide deterministic implementations for every op.
    # Warn-only requests deterministic kernels wherever available without
    # disguising an unsupported kernel as a successful strict guarantee.
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "python_random_seeded": True,
        "numpy_random_seeded": True,
        "torch_cpu_seeded": True,
        "torch_all_cuda_devices_seeded": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": True,
    }


def _mean_delta(init_metrics: Mapping[str, Any], refine_metrics: Mapping[str, Any]) -> dict[str, float]:
    initial = init_metrics["mean"]
    refined = refine_metrics["mean"]
    return {
        key: round(float(refined[key]) - float(initial[key]), 4)
        for key in ("psnr", "ssim", "lpips")
    }


def _prepare_staging(destination: Path) -> Path:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output root: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=str(destination.parent))
    )


def _scene_manifest_image_names(payload: Mapping[str, Any]) -> list[str]:
    names = []
    for offset, frame in enumerate(payload.get("frames", [])):
        if not isinstance(frame, dict) or not str(frame.get("image_name", "")):
            raise ValueError(f"scene manifest frame {offset} has no image_name")
        names.append(str(frame["image_name"]))
    if len(set(names)) != len(names):
        raise ValueError("scene manifest contains duplicate image_name values")
    return sorted(names)


def select_explicit_source_views(
    *,
    scene_data: Mapping[str, Any],
    scene_manifest: Mapping[str, Any],
    context_source_indices: Sequence[int],
    target_source_indices: Sequence[int],
    expected_context_count: int,
) -> tuple[list[int], list[int]]:
    """Resolve preregistered source IDs without consulting pixels or poses."""

    context_sources = [int(value) for value in context_source_indices]
    target_sources = [int(value) for value in target_source_indices]
    if len(context_sources) != expected_context_count:
        raise ValueError(
            "explicit context count mismatch: "
            f"expected {expected_context_count}, got {len(context_sources)}"
        )
    if len(set(context_sources)) != len(context_sources):
        raise ValueError("explicit context source indices contain duplicates")
    if not target_sources or len(set(target_sources)) != len(target_sources):
        raise ValueError("explicit target source indices must be non-empty and unique")

    frames = scene_manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("scene manifest has no frame records for explicit selection")
    name_by_source: dict[int, str] = {}
    for offset, record in enumerate(frames):
        if not isinstance(record, Mapping):
            raise ValueError(f"scene manifest frame {offset} is not an object")
        try:
            source = int(record["source_index"])
            name = str(record["image_name"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"scene manifest frame {offset} lacks source_index/image_name"
            ) from error
        if source in name_by_source:
            raise ValueError(f"duplicate scene-manifest source_index={source}")
        name_by_source[source] = name
    index_by_name = {
        str(name): index for index, name in enumerate(scene_data["image_names"])
    }
    if len(index_by_name) != len(scene_data["image_names"]):
        raise ValueError("official scene data contains duplicate image names")

    def resolve(sources: Sequence[int], label: str) -> list[int]:
        missing_sources = [source for source in sources if source not in name_by_source]
        if missing_sources:
            raise ValueError(f"{label} sources absent from scene manifest: {missing_sources}")
        missing_names = [
            name_by_source[source]
            for source in sources
            if name_by_source[source] not in index_by_name
        ]
        if missing_names:
            raise ValueError(f"{label} images absent from official scene: {missing_names}")
        return [index_by_name[name_by_source[source]] for source in sources]

    return resolve(context_sources, "context"), resolve(target_sources, "target")


def load_metric_reference_paths(
    reference_json: Path | str, target_source_indices: Sequence[int]
) -> tuple[list[str], dict[str, Any]]:
    """Resolve hash-audited metric-only images in declared target order."""

    source = Path(reference_json).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"metric-reference JSON not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid metric-reference JSON: {source}") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != METRIC_REFERENCE_SCHEMA:
        raise ValueError("wrong metric-reference schema")
    if payload.get("selection_fixed_before_reference_loading") is not True:
        raise ValueError("metric references were not frozen after index selection")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("metric-reference JSON has no frames")
    by_source: dict[int, dict[str, Any]] = {}
    for position, record in enumerate(frames):
        if not isinstance(record, Mapping):
            raise ValueError(f"metric-reference frame {position} is not an object")
        try:
            source_index = int(record["source_index"])
            declared = str(record["sha256"]).strip().lower()
            reference = Path(str(record["path"])).expanduser()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid metric-reference frame {position}") from error
        if source_index in by_source or not SHA256_RE.fullmatch(declared):
            raise ValueError(f"invalid/duplicate metric reference source={source_index}")
        if not reference.is_absolute():
            reference = source.parent / reference
        reference = reference.resolve()
        if not reference.is_file():
            raise FileNotFoundError(f"metric reference does not exist: {reference}")
        actual = sha256_file(reference)
        if actual != declared:
            raise ValueError(f"metric-reference SHA-256 mismatch: {reference}")
        by_source[source_index] = {
            "source_index": source_index,
            "path": str(reference),
            "sha256": actual,
            "included_in_formal_aggregate": bool(
                record.get("included_in_formal_aggregate", False)
            ),
        }
    requested = [int(value) for value in target_source_indices]
    missing = [value for value in requested if value not in by_source]
    if missing:
        raise ValueError(f"metric references missing target sources: {missing}")
    selected = [by_source[value] for value in requested]
    return [record["path"] for record in selected], {
        "schema": METRIC_REFERENCE_SCHEMA,
        "manifest_path": str(source),
        "manifest_sha256": sha256_file(source),
        "selected": selected,
        "passed_to_encoder_or_forward_update": False,
    }


def run(args: argparse.Namespace) -> Path:
    """Run the formal paired experiment and atomically install its outputs."""

    # Import heavy dependencies only for a real run; CPU source-contract tests
    # can import this module without importing the official CUDA stack.
    import numpy as np
    import torch
    from PIL import Image

    del Image  # PIL availability is required by the official helper import.

    scene_path = args.scene_path.expanduser().resolve()
    if not scene_path.is_dir():
        raise FileNotFoundError(f"COLMAP scene directory not found: {scene_path}")
    destination = args.output_dir.expanduser().resolve()
    staging = _prepare_staging(destination)
    installed = False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("paired official ReSplat smoke requires CUDA")
        device = torch.device(args.device)
        if str(device) != "cuda:0":
            raise ValueError("formal paired ReSplat requires logical --device cuda:0")
        expected_gpu_values = (
            args.expected_physical_index,
            args.expected_cuda_visible_devices,
            args.expected_gpu_name,
            args.expected_gpu_uuid,
            args.expected_gpu_serial,
        )
        if any(value is not None for value in expected_gpu_values) and not all(
            value is not None for value in expected_gpu_values
        ):
            raise ValueError("formal GPU identity arguments must be provided together")
        if all(value is not None for value in expected_gpu_values):
            gpu_binding = validate_gpu_contract(
                {
                    "physical_index": args.expected_physical_index,
                    "visible_devices": args.expected_cuda_visible_devices,
                    "logical_device": "cuda:0",
                    "name": args.expected_gpu_name,
                    "uuid": args.expected_gpu_uuid,
                    "serial": args.expected_gpu_serial,
                },
                require_visible_mask=True,
                require_idle=False,
            )
        else:
            gpu_binding = {
                "identity_verified": False,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "logical_device": "cuda:0",
            }
        reproducibility = _configure_reproducibility(torch, np, args.seed)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        terminal_started = time.perf_counter()

        repo_record = inspect_official_repo(args.resplat_repo)
        if (
            args.expected_resplat_commit is not None
            and repo_record["commit"] != args.expected_resplat_commit
        ):
            raise ValueError(
                "official ReSplat commit mismatch: "
                f"expected {args.expected_resplat_commit}, got {repo_record['commit']}"
            )
        repo = Path(repo_record["path"])
        official = import_official_infer(repo)
        checkpoint, checkpoint_sha, preset = _resolve_checkpoint(
            repo=repo,
            official=official,
            preset_name=args.model_preset,
            checkpoint=args.checkpoint,
            expected_sha256=args.expected_checkpoint_sha256,
        )
        scene_manifest = _load_scene_manifest(scene_path, args.scene_manifest)
        scene_manifest_names = _scene_manifest_image_names(scene_manifest["payload"])

        scene_data = official.load_colmap_scene(
            str(scene_path), args.sparse_dir, args.images_dir
        )
        if sorted(scene_data["image_names"]) != scene_manifest_names:
            raise ValueError(
                "loaded COLMAP image names do not exactly match the audited scene manifest"
            )
        if not scene_data["image_paths"]:
            raise ValueError("official loader found no scene images")

        with official.Image.open(scene_data["image_paths"][0]) as first_image:
            original_width, original_height = first_image.size
        target_height, target_width = official.compute_target_shape(
            original_height,
            original_width,
            int(preset["max_resolution"]),
            tuple(args.image_shape) if args.image_shape is not None else None,
        )

        num_context = int(preset["num_context"])
        explicit_selection = args.context_source_indices is not None
        if explicit_selection != (args.target_source_indices is not None):
            raise ValueError(
                "--context-source-indices and --target-source-indices must be used together"
            )
        if explicit_selection:
            context_indices, target_indices = select_explicit_source_views(
                scene_data=scene_data,
                scene_manifest=scene_manifest["payload"],
                context_source_indices=args.context_source_indices,
                target_source_indices=args.target_source_indices,
                expected_context_count=num_context,
            )
        else:
            context_indices = official.select_context_views(
                scene_data["c2w"], num_context, args.context_selection
            )
            target_indices = official.select_target_views(
                len(scene_data["image_paths"]), context_indices, "remaining", None
            )
        if len(context_indices) != num_context:
            raise ValueError(
                f"official preset requires {num_context} context views, got {len(context_indices)}"
            )
        if len(target_indices) != args.expected_target_count:
            raise ValueError(
                "paired smoke target count mismatch: "
                f"expected {args.expected_target_count}, got {len(target_indices)}"
            )

        context_paths = [scene_data["image_paths"][i] for i in context_indices]
        target_paths = [scene_data["image_paths"][i] for i in target_indices]
        context_names = [scene_data["image_names"][i] for i in context_indices]
        target_names = [scene_data["image_names"][i] for i in target_indices]
        context_images = official.load_and_preprocess_images(
            context_paths, target_height, target_width
        )
        target_images = official.load_and_preprocess_images(
            target_paths, target_height, target_width
        )
        encoder, decoder, data_shim = official.build_model(
            experiment=args.experiment,
            checkpoint=str(checkpoint),
            num_refine=PAIRED_NUM_REFINE,
            image_shape=(target_height, target_width),
            overrides=list(preset["overrides"]) + list(args.overrides),
            device=str(device),
            # This is an official-model contract, not a permissive checkpoint
            # compatibility probe.  Every released parameter must match.
            no_strict_load=False,
        )
        context_c2w = torch.tensor(
            scene_data["c2w"][context_indices], dtype=torch.float32
        )
        target_c2w = torch.tensor(
            scene_data["c2w"][target_indices], dtype=torch.float32
        )
        context_intrinsics = torch.tensor(
            scene_data["intrinsics"][context_indices], dtype=torch.float32
        )
        target_intrinsics = torch.tensor(
            scene_data["intrinsics"][target_indices], dtype=torch.float32
        )
        batch = official.build_batch(
            context_images,
            target_images,
            context_c2w,
            target_c2w,
            context_intrinsics,
            target_intrinsics,
            args.near,
            args.far,
            scene_path.name,
            str(device),
        )
        batch = data_shim(batch)

        recorder = CudaStageRecorder(torch, device)
        visualization_dump: Optional[dict[str, Any]] = {} if args.save_depth else None
        with torch.no_grad():
            paired = run_paired_inference_core(
                torch_module=torch,
                encoder=encoder,
                decoder=decoder,
                batch=batch,
                num_refine=PAIRED_NUM_REFINE,
                render_chunk_size=args.render_chunk_size,
                stage_runner=recorder,
                visualization_dump=visualization_dump,
            )

        torch.cuda.synchronize(device)
        terminal_reconstruction_wall = time.perf_counter() - terminal_started
        terminal_reconstruction_peak = max(
            recorder.process_peak_allocated_bytes,
            int(torch.cuda.max_memory_allocated(device)),
        )

        # Metric-only references are deliberately resolved and loaded after the
        # terminal renderer has finished and its primary clock has stopped.
        if args.target_reference_json is None:
            metric_reference_images = target_images
            metric_reference_record = {
                "mode": "scene_input_fallback_not_formal_offline_protocol",
                "passed_to_encoder_or_forward_update": False,
            }
        else:
            if args.target_source_indices is None:
                raise ValueError(
                    "--target-reference-json requires explicit target source indices"
                )
            metric_reference_paths, metric_reference_record = load_metric_reference_paths(
                args.target_reference_json, args.target_source_indices
            )
            metric_reference_record["mode"] = "frozen_metric_only_reference"
            metric_reference_images = official.load_and_preprocess_images(
                metric_reference_paths, target_height, target_width
            )

        init_dir = staging / "paired_init0"
        refine_dir = staging / "paired_refine4"
        official.save_outputs(
            paired["init_rendered"],
            paired["init_gaussians"],
            batch,
            str(init_dir),
            target_names,
            save_images=True,
            save_ply=args.save_ply,
            save_depth=args.save_depth,
            context_image_names=context_names,
            context_images=context_images,
            visualization_dump=paired["visualization_dump"],
            rendered_depth=paired["init_depth"],
            max_save_images=args.max_save_images,
        )
        official.save_outputs(
            paired["refined_rendered"],
            paired["refined_gaussians"],
            batch,
            str(refine_dir),
            target_names,
            save_images=True,
            save_ply=args.save_ply,
            save_depth=args.save_depth,
            context_image_names=context_names,
            context_images=context_images,
            visualization_dump=None,
            rendered_depth=paired["refined_depth"],
            max_save_images=args.max_save_images,
        )
        init_metrics = official.compute_metrics(
            paired["init_rendered"], metric_reference_images, target_names,
            str(init_dir), str(device)
        )
        refine_metrics = official.compute_metrics(
            paired["refined_rendered"], metric_reference_images, target_names,
            str(refine_dir), str(device)
        )

        script_path = Path(__file__).resolve()
        stage_records = recorder.records
        total_core_ms = round(
            sum(float(record["elapsed_ms"]) for record in stage_records.values()), 3
        )
        peak_allocated = max(
            int(record["peak_allocated_bytes"]) for record in stage_records.values()
        )
        source_by_image_name = {
            str(frame["image_name"]): int(frame["source_index"])
            for frame in scene_manifest["payload"]["frames"]
        }
        artifact_sources = (
            [int(value) for value in args.target_source_indices]
            if args.target_source_indices is not None
            else [source_by_image_name[name] for name in target_names]
        )
        refined_artifacts = []
        for source_index, target_name in zip(artifact_sources, target_names):
            rendered_path = refine_dir / "rendered" / f"{Path(target_name).stem}.png"
            if not rendered_path.is_file():
                raise FileNotFoundError(f"missing saved refined render: {rendered_path}")
            refined_artifacts.append(
                {
                    "source_index": int(source_index),
                    "relative_path": str(rendered_path.relative_to(staging)),
                    "sha256": sha256_file(rendered_path),
                    "quantization": "official_resplat_mul255_astype_uint8_floor",
                }
            )
        manifest = {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_class": "paired_official_cvg_resplat_init0_vs_refine4",
            "runner": {
                "path": str(script_path),
                "sha256": sha256_file(script_path),
            },
            "official_resplat": {
                "repository": repo_record,
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha,
                },
                "model_preset": args.model_preset,
                "experiment": args.experiment,
                "num_context": num_context,
                "num_refine": PAIRED_NUM_REFINE,
                "official_helpers_used": [
                    "load_colmap_scene",
                    "select_context_views",
                    "select_target_views",
                    "load_and_preprocess_images",
                    "build_model",
                    "build_batch",
                    "encoder.forward_update",
                    "decoder.forward",
                    "save_outputs",
                    "compute_metrics",
                ],
            },
            "scene": {
                "path": str(scene_path),
                "manifest_path": scene_manifest["path"],
                "manifest_sha256": scene_manifest["sha256"],
                "manifest_schema": scene_manifest["schema"],
                "image_count": len(scene_data["image_names"]),
            },
            "selection": {
                "context_strategy": (
                    "explicit_preregistered_source_indices"
                    if explicit_selection
                    else args.context_selection
                ),
                "explicit_source_selection": explicit_selection,
                "context_source_indices": (
                    None
                    if args.context_source_indices is None
                    else [int(value) for value in args.context_source_indices]
                ),
                "target_source_indices": (
                    None
                    if args.target_source_indices is None
                    else [int(value) for value in args.target_source_indices]
                ),
                "context_target_overlap_count": len(
                    set(context_indices).intersection(target_indices)
                ),
                "context_indices_zero_based": [int(i) for i in context_indices],
                "context_names": context_names,
                "target_indices_zero_based": [int(i) for i in target_indices],
                "target_names": target_names,
                "target_count": len(target_names),
                "same_targets_for_init0_and_refine4": True,
            },
            "image_shape": [target_height, target_width],
            "gpu_binding": gpu_binding,
            "near": args.near,
            "far": args.far,
            "reproducibility": {
                **reproducibility,
                "torch_version": torch.__version__,
                "cuda_runtime_version": torch.version.cuda,
                "device_argument": str(device),
                "device_name": torch.cuda.get_device_name(device),
                "encoder_deterministic_argument": False,
                "note": (
                    "The official inference path passes deterministic=False; fixed RNG seeds "
                    "and warn-only deterministic algorithms are recorded explicitly."
                ),
            },
            "paired_contract": paired["contract"],
            "cuda_core_runtime": {
                "events_synchronized": True,
                "stages": stage_records,
                "total_core_ms_sum": total_core_ms,
                "peak_allocated_bytes_across_stages": peak_allocated,
            },
            "terminal_reconstruction": {
                "primary": True,
                "wall_seconds": terminal_reconstruction_wall,
                "peak_allocated_bytes": terminal_reconstruction_peak,
                "clock": "synchronized_time_perf_counter",
                "wall_scope": "terminal_backend_setup_plus_core_no_metrics_or_artifact_io",
                "peak_scope": "process_setup_plus_core_before_metrics_and_artifact_io",
                "includes": [
                    "official repository and checkpoint verification",
                    "scene and input preprocessing",
                    "checkpoint/model load",
                    "initializer",
                    "four recurrent updates",
                    "query rendering",
                ],
                "excludes": [
                    "metric-reference loading",
                    "metric computation",
                    "output PNG/PLY artifact I/O",
                ],
            },
            "metrics": {
                "reference": metric_reference_record,
                "resolution_hw": [target_height, target_width],
                "paired_init0": init_metrics["mean"],
                "paired_refine4": refine_metrics["mean"],
                "refine4_minus_init0": _mean_delta(init_metrics, refine_metrics),
                "lpips_direction": "lower_is_better",
            },
            "outputs": {
                "paired_init0": "paired_init0",
                "paired_refine4": "paired_refine4",
                "paired_refine4_rendered": refined_artifacts,
                "root_installed_atomically": True,
            },
        }
        manifest_path = staging / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing concurrent overwrite of output: {destination}")
        os.rename(staging, destination)
        installed = True
        return destination
    finally:
        if not installed and staging.exists():
            shutil.rmtree(staging)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-path", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path)
    parser.add_argument("--resplat-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-resplat-commit")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-preset", default=DEFAULT_PRESET)
    parser.add_argument("--experiment", default="dl3dv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-physical-index", type=int)
    parser.add_argument("--expected-cuda-visible-devices")
    parser.add_argument("--expected-gpu-name")
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--expected-gpu-serial")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--context-selection", choices=("fps", "uniform"), default="fps")
    parser.add_argument("--context-source-indices", type=int, nargs="+")
    parser.add_argument("--target-source-indices", type=int, nargs="+")
    parser.add_argument("--target-reference-json", type=Path)
    parser.add_argument("--expected-target-count", type=int, default=PAIRED_TARGET_COUNT)
    parser.add_argument("--image-shape", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=200.0)
    parser.add_argument("--render-chunk-size", type=int, default=4)
    parser.add_argument("--max-save-images", type=int, default=PAIRED_TARGET_COUNT)
    parser.add_argument("--save-ply", action="store_true")
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        output = run(parse_args(argv))
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"paired official ReSplat smoke saved atomically to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
