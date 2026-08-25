"""Strict CPU-loadable runtime for TURTLE's official BSD 3ms-24ms model.

The upstream BSD checkpoint is *not* the GoPro ``t1`` architecture used by
``src.turtle_backend``.  Upstream's own inference recipe selects ``model_type
= 't0'`` (``turtle_arch.py``) while parsing ``Turtle_Derain_VRDS.yml``.  The
apparently deraining-named YAML is therefore a pinned provenance quirk, not a
license to substitute the GoPro config or architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Optional, Tuple

import torch
import yaml

from src.turtle_backend import TurtleStreamingBackend


PINNED_TURTLE_COMMIT = "7094f4221b64ad0962b4f27ff1b76d788836e804"
PINNED_BSD_ARCH_SHA256 = (
    "0e2b18ee87797094a10d41da097bfe49d46b147c78a35c45ca88fcd72c9ee247"
)
PINNED_BSD_CONFIG_SHA256 = (
    "f99ce8e32f1ec87b7ab3933158d3297e97de083fa034efa231c224ab787ad0b1"
)
PINNED_UPSTREAM_INFERENCE_SHA256 = (
    "73f8b16a014f888f49bdf4bf3a1da40f4d930056a01dfe6ab07e9678b7384b78"
)
PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256 = (
    "183d5e344488382a39c32aad86559e6fd568954134ddfee8793f5851a2cdf809"
)
PINNED_OFFICIAL_BSD_CHECKPOINT_BYTES = 234_739_171
PINNED_OFFICIAL_BSD_STATE_TENSORS = 621
PINNED_OFFICIAL_BSD_PARAMETERS = 58_620_800
PINNED_OFFICIAL_BSD_CHECKPOINT = Path(
    "/srv/szha0669/real_video_data/bsd_3ms24ms_official_quarantine/BSD_Deblur.pth"
)
PINNED_TURTLE_REPO = Path("/srv/szha0669/unblur-slam/external/TURTLE")
PINNED_BSD_CONFIG = PINNED_TURTLE_REPO / "options/Turtle_Derain_VRDS.yml"
BSD_FINETUNED_CHECKPOINT_FORMAT = (
    "unblur_slam.turtle_official_bsd_dpdd_spatial.checkpoint.v1"
)
BSD_CACHE_CONTRACT = "official_t0_kv_8_incremental"
BSD_SPATIAL_PREFIXES = ("refinement.", "ending.")
BSD_SPATIAL_PARAMETER_TENSORS = 30
BSD_SPATIAL_PARAMETER_COUNT = 105_283

_ARCH_RELATIVE_PATH = Path("basicsr/models/archs/turtle_arch.py")
_INFERENCE_RELATIVE_PATH = Path("basicsr/inference.py")


@dataclass(frozen=True)
class OfficialBsdArtifacts:
    repo: Path
    config: Path
    architecture: Path
    inference_recipe: Path
    checkpoint: Path
    options: Mapping[str, Any]
    checkpoint_sha256: str
    checkpoint_metadata: Mapping[str, Any]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_file(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"missing {label}: {candidate}")
    return candidate


def _required_directory(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"missing {label}: {candidate}")
    return candidate


def _assert_hash(path: Path, expected: str, *, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


def _load_options(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        options = yaml.safe_load(stream)
    if not isinstance(options, dict):
        raise ValueError("official BSD TURTLE options must be a mapping")
    # Upstream inference explicitly pairs this exact VRDS-named YAML with t0
    # for BSD.  Hashing the whole YAML is the primary contract; these fields
    # make the architectural consequences reviewable in reports.
    expected = {
        "n_sequence": 5,
        "n_colors": 3,
        "dim": 64,
        "Enc_blocks": [2, 6, 10],
        "Middle_blocks": 11,
        "Dec_blocks": [10, 6, 2],
        "num_refinement_blocks": 2,
        "use_both_input": False,
        "num_heads": [1, 2, 4, 8],
        "num_frames_tocache": 3,
        "patch_size": 192,
    }
    mismatches = {
        key: (options.get(key), wanted)
        for key, wanted in expected.items()
        if options.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"official BSD TURTLE option mismatch: {mismatches}")
    return options


def _torch_load(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("official BSD TURTLE checkpoint must contain a mapping")
    return payload


def _state_dict(value: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, dict) or not value:
        raise ValueError("official BSD TURTLE state dict is missing")
    if not all(isinstance(key, str) and torch.is_tensor(tensor) for key, tensor in value.items()):
        raise ValueError("official BSD TURTLE state dict must map names to tensors")
    if all(key.startswith("module.") for key in value):
        value = {key[len("module.") :]: tensor for key, tensor in value.items()}
    return value


def _metadata(value: Any) -> Mapping[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("fine-tuned official-BSD checkpoint metadata is missing")
    return value


def validate_bsd_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    expected_checkpoint_sha256: Optional[str] = None,
) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
    actual = str(checkpoint_sha256).strip().lower()
    expected = (
        None
        if expected_checkpoint_sha256 is None
        else str(expected_checkpoint_sha256).strip().lower()
    )
    if expected is not None and actual != expected:
        raise ValueError(f"official-BSD checkpoint hash mismatch: {actual}")
    if actual == PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256:
        if set(payload) != {"params"}:
            raise ValueError("official BSD checkpoint payload keys changed")
        state = _state_dict(payload["params"])
        if len(state) != PINNED_OFFICIAL_BSD_STATE_TENSORS:
            raise ValueError("official BSD checkpoint tensor count changed")
        if sum(tensor.numel() for tensor in state.values()) != PINNED_OFFICIAL_BSD_PARAMETERS:
            raise ValueError("official BSD checkpoint parameter count changed")
        if {tensor.dtype for tensor in state.values()} != {torch.float32}:
            raise ValueError("official BSD checkpoint must contain only float32 tensors")
        return state, {
            "format": "official_turtle.params",
            "kind": "official_bsd_3ms24ms_t0",
            "checkpoint_sha256": actual,
            "base_checkpoint_sha256": actual,
            "turtle_repo_commit": PINNED_TURTLE_COMMIT,
            "turtle_arch_variant": "t0",
            "turtle_arch_sha256": PINNED_BSD_ARCH_SHA256,
            "turtle_config_sha256": PINNED_BSD_CONFIG_SHA256,
            "upstream_inference_sha256": PINNED_UPSTREAM_INFERENCE_SHA256,
            "input_domain": "raw",
            "cache_contract": BSD_CACHE_CONTRACT,
        }
    if expected is None:
        raise ValueError("a fine-tuned official-BSD checkpoint requires its explicit SHA256")
    if set(payload) != {"params", "metadata"}:
        raise ValueError("fine-tuned official-BSD checkpoint payload keys changed")
    state = _state_dict(payload["params"])
    metadata = dict(_metadata(payload["metadata"]))
    required = {
        "format": BSD_FINETUNED_CHECKPOINT_FORMAT,
        "base_checkpoint_sha256": PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256,
        "turtle_repo_commit": PINNED_TURTLE_COMMIT,
        "turtle_arch_variant": "t0",
        "turtle_arch_sha256": PINNED_BSD_ARCH_SHA256,
        "turtle_config_sha256": PINNED_BSD_CONFIG_SHA256,
        "upstream_inference_sha256": PINNED_UPSTREAM_INFERENCE_SHA256,
        "input_domain": "raw",
        "cache_contract": BSD_CACHE_CONTRACT,
        "mode": "OD",
    }
    mismatches = {
        key: (metadata.get(key), wanted)
        for key, wanted in required.items()
        if metadata.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"unsafe official-BSD fine-tuned metadata: {mismatches}")
    metadata["kind"] = "official_bsd_dpdd_spatial_finetuned"
    metadata["checkpoint_sha256"] = actual
    return state, metadata


def validate_official_bsd_artifacts(
    *,
    repo: Path | str = PINNED_TURTLE_REPO,
    config: Path | str = PINNED_BSD_CONFIG,
    checkpoint: Path | str = PINNED_OFFICIAL_BSD_CHECKPOINT,
    checkpoint_sha256: Optional[str] = None,
    load_weights: bool = False,
) -> OfficialBsdArtifacts:
    repo_path = _required_directory(repo, label="TURTLE repository")
    config_path = _required_file(config, label="official BSD TURTLE config")
    checkpoint_path = _required_file(checkpoint, label="official BSD TURTLE checkpoint")
    architecture = _required_file(repo_path / _ARCH_RELATIVE_PATH, label="TURTLE t0 architecture")
    inference = _required_file(repo_path / _INFERENCE_RELATIVE_PATH, label="upstream inference recipe")
    if _git_head(repo_path) != PINNED_TURTLE_COMMIT:
        raise ValueError("TURTLE repository commit changed")
    _assert_hash(architecture, PINNED_BSD_ARCH_SHA256, label="TURTLE t0 architecture")
    _assert_hash(config_path, PINNED_BSD_CONFIG_SHA256, label="official BSD config")
    _assert_hash(inference, PINNED_UPSTREAM_INFERENCE_SHA256, label="upstream inference recipe")
    digest = sha256_file(checkpoint_path)
    configured = checkpoint_sha256
    if digest == PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256:
        if checkpoint_path.stat().st_size != PINNED_OFFICIAL_BSD_CHECKPOINT_BYTES:
            raise ValueError("official BSD checkpoint byte size changed")
        if configured is not None and str(configured).lower() != digest:
            raise ValueError("configured official BSD checkpoint SHA256 disagrees")
        configured = digest
    elif configured is None:
        raise ValueError("fine-tuned official-BSD checkpoint requires explicit SHA256")
    state, metadata = validate_bsd_checkpoint_payload(
        _torch_load(checkpoint_path),
        checkpoint_sha256=digest,
        expected_checkpoint_sha256=configured,
    )
    artifacts = OfficialBsdArtifacts(
        repo=repo_path,
        config=config_path,
        architecture=architecture,
        inference_recipe=inference,
        checkpoint=checkpoint_path,
        options=_load_options(config_path),
        checkpoint_sha256=digest,
        checkpoint_metadata=metadata,
    )
    if load_weights:
        model = build_official_bsd_turtle_model(artifacts, state_dict=state, device="cpu")
        del model
    return artifacts


def _import_t0(architecture: Path, digest: str):
    module_name = "_unblur_slam_pinned_turtle_t0_" + digest[:12]
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, architecture)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import TURTLE t0 architecture: {architecture}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def build_official_bsd_turtle_model(
    artifacts: OfficialBsdArtifacts,
    *,
    state_dict: Optional[Mapping[str, torch.Tensor]] = None,
    device: Any = "cpu",
) -> torch.nn.Module:
    module = _import_t0(artifacts.architecture, PINNED_BSD_ARCH_SHA256)
    model = module.make_model(dict(artifacts.options))
    if state_dict is None:
        state_dict, _ = validate_bsd_checkpoint_payload(
            _torch_load(artifacts.checkpoint),
            checkpoint_sha256=artifacts.checkpoint_sha256,
            expected_checkpoint_sha256=artifacts.checkpoint_sha256,
        )
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("official BSD checkpoint failed strict t0 architecture load")
    if sum(parameter.numel() for parameter in model.parameters()) != PINNED_OFFICIAL_BSD_PARAMETERS:
        raise ValueError("constructed official BSD model parameter count changed")
    if bool(getattr(model, "use_both_input", True)):
        raise ValueError("official BSD model unexpectedly uses the left image")
    model.requires_grad_(False)
    model.turtle_checkpoint_metadata = dict(artifacts.checkpoint_metadata)
    return model.to(torch.device(device)).eval()


def load_official_bsd_turtle_model(
    checkpoint: Path | str = PINNED_OFFICIAL_BSD_CHECKPOINT,
    *,
    repo: Path | str = PINNED_TURTLE_REPO,
    config: Path | str = PINNED_BSD_CONFIG,
    device: Any = "cpu",
    checkpoint_sha256: Optional[str] = None,
) -> Tuple[torch.nn.Module, Mapping[str, Any]]:
    artifacts = validate_official_bsd_artifacts(
        repo=repo,
        config=config,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        load_weights=False,
    )
    model = build_official_bsd_turtle_model(artifacts, device=device)
    return model, dict(artifacts.checkpoint_metadata)


class OfficialBsdTurtleStreamingBackend(TurtleStreamingBackend):
    """Existing one-call-per-frame K/V runtime with a strictly loaded t0 model."""

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path | str = PINNED_OFFICIAL_BSD_CHECKPOINT,
        *,
        device: Any = "cpu",
        inference_precision: str = "fp32",
        checkpoint_sha256: Optional[str] = None,
    ) -> "OfficialBsdTurtleStreamingBackend":
        model, _ = load_official_bsd_turtle_model(
            checkpoint,
            device=device,
            checkpoint_sha256=checkpoint_sha256,
        )
        return cls(model, device=device, inference_precision=inference_precision)

    @classmethod
    def from_config(
        cls,
        deblur_cfg: Mapping[str, Any],
        *,
        device: Any = "cuda:0",
    ) -> "OfficialBsdTurtleStreamingBackend":
        """Construct the pinned BSD-t0 stream from an auditable SLAM config."""

        artifacts = validate_official_bsd_artifacts(
            repo=deblur_cfg.get("turtle_repo", PINNED_TURTLE_REPO),
            config=deblur_cfg.get("turtle_config", PINNED_BSD_CONFIG),
            checkpoint=deblur_cfg.get(
                "turtle_checkpoint", PINNED_OFFICIAL_BSD_CHECKPOINT
            ),
            checkpoint_sha256=deblur_cfg.get("turtle_checkpoint_sha256"),
            load_weights=False,
        )
        model = build_official_bsd_turtle_model(artifacts, device=device)
        backend = cls(
            model,
            device=device,
            inference_precision=deblur_cfg.get(
                "turtle_inference_precision", "fp32"
            ),
        )
        backend.artifacts = artifacts
        return backend
