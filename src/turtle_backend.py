"""Pinned TURTLE streaming-deblur runtime for Unblur-SLAM.

This adapter intentionally calls the upstream TURTLE network once per input
frame and feeds the eight key/value tensors returned by that call into the
next call.  It is therefore an incremental causal runtime, not a sliding
window wrapper that recomputes past frames.

The third-party checkout and GoPro checkpoint are treated as immutable
artifacts.  Their revision/content hashes are verified before a model is
constructed, and the architecture is imported directly from its pinned file
without adding TURTLE's ``basicsr`` namespace to the process-wide import path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import yaml


PINNED_TURTLE_COMMIT = "7094f4221b64ad0962b4f27ff1b76d788836e804"
PINNED_TURTLE_ARCH_SHA256 = (
    "4d19c676f92574dbad493eb591312fdeaf2b3b519f57410af2ed95fdbef5f058"
)
PINNED_TURTLE_CONFIG_SHA256 = (
    "123b07de8d3f329769562e2f943e08fdf86c576c405634bad199ced95b25aa23"
)
PINNED_TURTLE_CHECKPOINT_SHA256 = (
    "10334b3e81d0416bcde5ccaca960dc81dbfb5b6d23e53fadaf7896d72b580c82"
)
FINETUNED_CHECKPOINT_FORMAT = "unblur_slam.turtle_streaming.checkpoint.v1"
STABLE_DEPLOY_CHECKPOINT_FORMAT = "unblur_slam.turtle_unblur_stable_deploy.v2"
TURTLE_CACHE_CONTRACT = "official_kv_8_incremental"
TURTLE_INFERENCE_PRECISIONS = frozenset({"fp32", "fp16"})

_ARCH_RELATIVE_PATH = Path("basicsr/models/archs/turtle_t1_arch.py")


@dataclass(frozen=True)
class TurtleArtifacts:
    """Validated immutable inputs required to construct official TURTLE."""

    repo: Path
    config: Path
    checkpoint: Path
    architecture: Path
    options: Mapping[str, Any]
    commit: str
    architecture_sha256: str
    config_sha256: str
    checkpoint_sha256: str
    checkpoint_metadata: Mapping[str, Any]


def _required_path(value: Any, label: str, *, directory: bool = False) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{label} must be configured for turtle_streaming")
    path = Path(str(value)).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{label} {kind} does not exist: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_turtle_inference_precision(value: Any) -> str:
    """Validate the opt-in runtime precision without changing model weights."""

    precision = str("fp32" if value is None else value).strip().lower()
    if precision not in TURTLE_INFERENCE_PRECISIONS:
        raise ValueError(
            "deblur.turtle_inference_precision must be one of "
            f"{sorted(TURTLE_INFERENCE_PRECISIONS)}, got {value!r}"
        )
    return precision


def _git_head(repo: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"turtle_repo is not a readable Git checkout: {repo}") from error
    return completed.stdout.strip().lower()


def _assert_expected_hash(label: str, actual: str, expected: str) -> None:
    if actual.lower() != expected.lower():
        raise ValueError(
            f"{label} hash mismatch: expected {expected.lower()}, got {actual.lower()}"
        )


def _load_options(config_path: Path) -> Mapping[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        options = yaml.safe_load(handle)
    if not isinstance(options, dict):
        raise ValueError(f"TURTLE options must contain a YAML mapping: {config_path}")

    expected = {
        "model": "Turtle_t1_arch",
        "type": "deblurring",
        "scale": 1,
        "n_colors": 3,
        "dim": 64,
        "Enc_blocks": [2, 6, 10],
        "Middle_blocks": 11,
        "Dec_blocks": [10, 6, 2],
        "num_frames_tocache": 3,
        "use_both_input": False,
    }
    mismatches = {
        key: (options.get(key), value)
        for key, value in expected.items()
        if options.get(key) != value
    }
    if mismatches:
        formatted = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in sorted(mismatches.items())
        )
        raise ValueError(f"TURTLE GoPro options are incompatible: {formatted}")
    return options


def _torch_load_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility.
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("TURTLE checkpoint must contain a mapping")
    return payload


def _normalize_state_dict(state_dict: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("TURTLE checkpoint state dict is missing or empty")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state_dict.items()):
        raise ValueError("TURTLE state dict must map string keys to tensors")
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def _metadata_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("TURTLE fine-tuned checkpoint metadata is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("TURTLE fine-tuned checkpoint requires mapping metadata")
    return value


def validate_turtle_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    expected_checkpoint_sha256: Optional[str] = None,
) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
    """Validate official GoPro or content-addressed fine-tuned state.

    Fine-tuned checkpoints must retain the exact official architecture and
    causal K/V contract.  Their metadata is deliberately strict so an
    unrelated state dict cannot be mistaken for the streaming model.
    """

    actual_hash = str(checkpoint_sha256).lower()
    expected_hash = (
        None if expected_checkpoint_sha256 is None else str(expected_checkpoint_sha256).lower()
    )
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(
            "TURTLE checkpoint hash mismatch: "
            f"expected {expected_hash}, got {actual_hash}"
        )

    if actual_hash == PINNED_TURTLE_CHECKPOINT_SHA256:
        state_dict = _normalize_state_dict(payload.get("params"))
        metadata = {
            "format": "official_turtle.params",
            "kind": "official_gopro",
            "checkpoint_sha256": actual_hash,
            "base_checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
            "turtle_repo_commit": PINNED_TURTLE_COMMIT,
            "turtle_arch_sha256": PINNED_TURTLE_ARCH_SHA256,
            "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
            "input_domain": "raw",
            "cache_contract": TURTLE_CACHE_CONTRACT,
        }
        return state_dict, metadata

    if expected_hash is None:
        raise ValueError(
            "a fine-tuned TURTLE checkpoint requires an explicit, content-addressing "
            "deblur.turtle_checkpoint_sha256"
        )
    state_value = payload.get("params", payload.get("state_dict"))
    state_dict = _normalize_state_dict(state_value)
    metadata = dict(_metadata_mapping(payload.get("metadata")))
    if metadata.get("format") == STABLE_DEPLOY_CHECKPOINT_FORMAT:
        stable_expected = {
            "format": STABLE_DEPLOY_CHECKPOINT_FORMAT,
            "stage": "defocus_rehearsal",
            "step": 300_000,
            "initialization_root": "random_scratch_pinned_turtle_architecture",
            "official_gopro_checkpoint_used_for_initialization": False,
            "turtle_repo_commit": PINNED_TURTLE_COMMIT,
            "turtle_arch_sha256": PINNED_TURTLE_ARCH_SHA256,
            "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
            "input_domain": "linear_srgb",
            "output_domain": "linear_srgb",
            "photometric_transform": "exact_srgb_to_linear_before_model_and_loss",
            "cache_contract": TURTLE_CACHE_CONTRACT,
        }
        mismatches = {
            key: (metadata.get(key), wanted)
            for key, wanted in stable_expected.items()
            if metadata.get(key) != wanted
        }
        if mismatches:
            formatted = ", ".join(
                f"{key}={actual!r} (expected {wanted!r})"
                for key, (actual, wanted) in sorted(mismatches.items())
            )
            raise ValueError(f"unsafe stable TURTLE deploy metadata: {formatted}")
        metadata["kind"] = "finetuned_unblur_stable"
        metadata["checkpoint_sha256"] = actual_hash
        return state_dict, metadata
    expected_metadata = {
        "format": FINETUNED_CHECKPOINT_FORMAT,
        "base_checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
        "turtle_repo_commit": PINNED_TURTLE_COMMIT,
        "turtle_arch_sha256": PINNED_TURTLE_ARCH_SHA256,
        "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
        "input_domain": "raw",
        "cache_contract": TURTLE_CACHE_CONTRACT,
    }
    mismatches = {
        key: (metadata.get(key), wanted)
        for key, wanted in expected_metadata.items()
        if metadata.get(key) != wanted
    }
    if mismatches:
        formatted = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in sorted(mismatches.items())
        )
        raise ValueError(f"unsafe TURTLE fine-tuned checkpoint metadata: {formatted}")
    metadata["kind"] = "finetuned"
    metadata["checkpoint_sha256"] = actual_hash
    return state_dict, metadata


def _import_pinned_architecture(artifacts: TurtleArtifacts):
    module_name = (
        "_unblur_slam_pinned_turtle_t1_"
        + artifacts.architecture_sha256[:12]
    )
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, artifacts.architecture)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import TURTLE architecture: {artifacts.architecture}")
    module = importlib.util.module_from_spec(spec)
    # Keeping the uniquely named module registered makes PyTorch diagnostics
    # and model introspection deterministic without exposing ``basicsr``.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def build_turtle_model(
    artifacts: TurtleArtifacts,
    *,
    device: Any = "cpu",
) -> torch.nn.Module:
    """Construct official TURTLE and strictly load the pinned GoPro weights."""

    module = _import_pinned_architecture(artifacts)
    model = module.make_model(dict(artifacts.options))
    payload = _torch_load_payload(artifacts.checkpoint)
    state_dict, metadata = validate_turtle_checkpoint_payload(
        payload,
        checkpoint_sha256=artifacts.checkpoint_sha256,
        expected_checkpoint_sha256=artifacts.checkpoint_sha256,
    )
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        # ``strict=True`` already raises, but retain an explicit invariant for
        # PyTorch variants that return rather than raise.
        raise ValueError(
            "TURTLE checkpoint is incompatible with its pinned architecture: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.requires_grad_(False)
    model.turtle_checkpoint_metadata = dict(metadata)
    return model.to(torch.device(device)).eval()


def load_turtle_model(
    repo: Any,
    checkpoint: Any,
    *,
    config: Any,
    device: Any = "cpu",
    checkpoint_sha256: Optional[str] = None,
    repo_commit: str = PINNED_TURTLE_COMMIT,
) -> Tuple[torch.nn.Module, Mapping[str, Any]]:
    """Public safe loader reusable by SLAM, training and evaluation tools.

    Returns ``(model, normalized_checkpoint_metadata)``.  The official GoPro
    file needs no supplied hash because its digest is compiled into this
    runtime. A fine-tuned file must provide its complete SHA256 and embedded
    provenance metadata conforming to :data:`FINETUNED_CHECKPOINT_FORMAT`.
    """

    deblur_cfg = {
        "turtle_repo": str(repo),
        "turtle_config": str(config),
        "turtle_checkpoint": str(checkpoint),
        "turtle_repo_commit": str(repo_commit),
    }
    if checkpoint_sha256 is not None:
        deblur_cfg["turtle_checkpoint_sha256"] = str(checkpoint_sha256)
    artifacts = validate_turtle_artifacts(deblur_cfg, load_weights=False)
    model = build_turtle_model(artifacts, device=device)
    return model, dict(artifacts.checkpoint_metadata)


def build_turtle_model_from_scratch(
    repo: Any,
    *,
    config: Any,
    device: Any = "cpu",
    repo_commit: str = PINNED_TURTLE_COMMIT,
) -> Tuple[torch.nn.Module, Mapping[str, Any]]:
    """Construct the pinned TURTLE architecture without loading any weights."""

    repo_path = _required_path(repo, "turtle_repo", directory=True)
    config_path = _required_path(config, "turtle_config")
    architecture = _required_path(
        repo_path / _ARCH_RELATIVE_PATH, "TURTLE pinned architecture"
    )
    observed_commit = _git_head(repo_path)
    if str(repo_commit).lower() != PINNED_TURTLE_COMMIT or observed_commit != PINNED_TURTLE_COMMIT:
        raise ValueError("scratch TURTLE repository revision mismatch")
    arch_hash = sha256_file(architecture)
    config_hash = sha256_file(config_path)
    _assert_expected_hash("TURTLE architecture", arch_hash, PINNED_TURTLE_ARCH_SHA256)
    _assert_expected_hash("TURTLE GoPro config", config_hash, PINNED_TURTLE_CONFIG_SHA256)
    artifacts = TurtleArtifacts(
        repo=repo_path,
        config=config_path,
        checkpoint=Path(),
        architecture=architecture,
        options=_load_options(config_path),
        commit=observed_commit,
        architecture_sha256=arch_hash,
        config_sha256=config_hash,
        checkpoint_sha256="",
        checkpoint_metadata={},
    )
    module = _import_pinned_architecture(artifacts)
    model = module.make_model(dict(artifacts.options))
    metadata = {
        "format": "unblur_slam.turtle_scratch_initialization.v1",
        "kind": "random_initialization",
        "turtle_repo_commit": observed_commit,
        "turtle_arch_sha256": arch_hash,
        "turtle_config_sha256": config_hash,
        "input_domain": "linear_srgb",
        "output_domain": "linear_srgb",
        "cache_contract": TURTLE_CACHE_CONTRACT,
    }
    model.turtle_checkpoint_metadata = dict(metadata)
    return model.to(torch.device(device)), metadata


def validate_turtle_artifacts(
    deblur_cfg: Mapping[str, Any],
    *,
    load_weights: bool = False,
) -> TurtleArtifacts:
    """Fail closed unless repo, config, architecture and checkpoint are pinned.

    ``load_weights=True`` additionally constructs the complete official model
    on CPU and performs a strict state-dict load.  It does not run inference or
    touch a CUDA device.
    """

    normalize_turtle_inference_precision(
        deblur_cfg.get("turtle_inference_precision", "fp32")
    )
    repo = _required_path(deblur_cfg.get("turtle_repo"), "deblur.turtle_repo", directory=True)
    config_path = _required_path(
        deblur_cfg.get("turtle_config"), "deblur.turtle_config"
    )
    checkpoint = _required_path(
        deblur_cfg.get("turtle_checkpoint"), "deblur.turtle_checkpoint"
    )
    architecture = _required_path(
        repo / _ARCH_RELATIVE_PATH, "TURTLE pinned architecture"
    )

    commit = _git_head(repo)
    expected_commit = str(
        deblur_cfg.get("turtle_repo_commit", PINNED_TURTLE_COMMIT)
    ).lower()
    # A caller may repeat the pin in YAML for auditability, but may not loosen
    # the code-level reproducibility contract by changing it.
    if expected_commit != PINNED_TURTLE_COMMIT:
        raise ValueError(
            "deblur.turtle_repo_commit does not match this runtime's pinned "
            f"revision {PINNED_TURTLE_COMMIT}"
        )
    if commit != PINNED_TURTLE_COMMIT:
        raise ValueError(
            f"TURTLE checkout revision mismatch: expected {PINNED_TURTLE_COMMIT}, got {commit}"
        )

    arch_hash = sha256_file(architecture)
    config_hash = sha256_file(config_path)
    checkpoint_hash = sha256_file(checkpoint)
    _assert_expected_hash("TURTLE architecture", arch_hash, PINNED_TURTLE_ARCH_SHA256)
    _assert_expected_hash("TURTLE GoPro config", config_hash, PINNED_TURTLE_CONFIG_SHA256)
    configured_config_hash = str(
        deblur_cfg.get("turtle_config_sha256", PINNED_TURTLE_CONFIG_SHA256)
    ).lower()
    if configured_config_hash != PINNED_TURTLE_CONFIG_SHA256:
        raise ValueError("deblur.turtle_config_sha256 may not override the pinned config")
    configured_checkpoint_hash = deblur_cfg.get("turtle_checkpoint_sha256")
    if checkpoint_hash == PINNED_TURTLE_CHECKPOINT_SHA256:
        if configured_checkpoint_hash is not None and str(configured_checkpoint_hash).lower() != checkpoint_hash:
            raise ValueError(
                "deblur.turtle_checkpoint_sha256 disagrees with the pinned GoPro checkpoint"
            )
        expected_checkpoint_hash = checkpoint_hash
    else:
        if configured_checkpoint_hash is None:
            raise ValueError(
                "fine-tuned TURTLE weights require deblur.turtle_checkpoint_sha256"
            )
        expected_checkpoint_hash = str(configured_checkpoint_hash).lower()

    _, checkpoint_metadata = validate_turtle_checkpoint_payload(
        _torch_load_payload(checkpoint),
        checkpoint_sha256=checkpoint_hash,
        expected_checkpoint_sha256=expected_checkpoint_hash,
    )

    artifacts = TurtleArtifacts(
        repo=repo,
        config=config_path,
        checkpoint=checkpoint,
        architecture=architecture,
        options=_load_options(config_path),
        commit=commit,
        architecture_sha256=arch_hash,
        config_sha256=config_hash,
        checkpoint_sha256=checkpoint_hash,
        checkpoint_metadata=checkpoint_metadata,
    )
    if load_weights:
        model = build_turtle_model(artifacts, device="cpu")
        del model
    return artifacts


def _detach_cache(cache: Optional[Sequence[Optional[torch.Tensor]]]):
    if cache is None:
        return None
    if not isinstance(cache, (tuple, list)) or len(cache) != 8:
        raise RuntimeError(
            "official TURTLE must return exactly eight key/value cache tensors"
        )
    return [None if value is None else value.detach() for value in cache]


def srgb_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    """Exact IEC 61966-2-1 inverse transfer on normalized RGB tensors."""

    tensor = tensor.clamp(0.0, 1.0)
    return torch.where(
        tensor <= 0.04045,
        tensor / 12.92,
        ((tensor + 0.055) / 1.055).pow(2.4),
    )


def linear_to_srgb(tensor: torch.Tensor) -> torch.Tensor:
    """Exact IEC 61966-2-1 forward transfer on normalized linear RGB."""

    tensor = tensor.clamp(0.0, 1.0)
    return torch.where(
        tensor <= 0.0031308,
        tensor * 12.92,
        1.055 * tensor.pow(1.0 / 2.4) - 0.055,
    ).clamp(0.0, 1.0)


class TurtleStreamingBackend:
    """One-call-per-frame causal adapter over official TURTLE K/V caches."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: Any = "cpu",
        inference_precision: str = "fp32",
    ):
        self.model = model
        self.device = torch.device(device)
        self.inference_precision = normalize_turtle_inference_precision(
            inference_precision
        )
        if self.inference_precision == "fp16" and self.device.type != "cuda":
            raise ValueError(
                "deblur.turtle_inference_precision=fp16 requires a CUDA device"
            )
        self._autocast_dtype = (
            torch.float16 if self.inference_precision == "fp16" else None
        )
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        metadata = getattr(self.model, "turtle_checkpoint_metadata", {})
        self.photometric_domain = str(metadata.get("input_domain", "raw"))
        output_domain = str(metadata.get("output_domain", self.photometric_domain))
        if self.photometric_domain not in {"raw", "linear_srgb"}:
            raise ValueError(f"unsupported TURTLE input domain: {self.photometric_domain}")
        if output_domain != self.photometric_domain:
            raise ValueError("TURTLE input/output photometric domains must match")
        self.k_cache: Optional[Sequence[Optional[torch.Tensor]]] = None
        self.v_cache: Optional[Sequence[Optional[torch.Tensor]]] = None
        self.previous_frame: Optional[torch.Tensor] = None
        self.resolution: Optional[Tuple[int, int]] = None
        self.last_timestamp: Optional[float] = None
        self.frames_seen = 0
        self.cache_updates = 0
        self.reset_count = 0

    @classmethod
    def from_config(
        cls,
        deblur_cfg: Mapping[str, Any],
        *,
        device: Any = "cuda:0",
    ) -> "TurtleStreamingBackend":
        artifacts = validate_turtle_artifacts(deblur_cfg, load_weights=False)
        model = build_turtle_model(artifacts, device=device)
        backend = cls(
            model,
            device=device,
            inference_precision=deblur_cfg.get(
                "turtle_inference_precision", "fp32"
            ),
        )
        backend.artifacts = artifacts
        return backend

    def reset(self) -> None:
        """Start a new causal stream and release all official cache tensors."""

        self.k_cache = None
        self.v_cache = None
        self.previous_frame = None
        self.resolution = None
        self.last_timestamp = None
        self.reset_count += 1

    def state_info(self) -> Mapping[str, Any]:
        return {
            "frames_seen": self.frames_seen,
            "cache_updates": self.cache_updates,
            "reset_count": self.reset_count,
            "resolution": self.resolution,
            "last_timestamp": self.last_timestamp,
            "has_cache": self.k_cache is not None and self.v_cache is not None,
            "inference_precision": self.inference_precision,
            "photometric_domain": self.photometric_domain,
        }

    def _to_model_domain(self, tensor: torch.Tensor) -> torch.Tensor:
        return srgb_to_linear(tensor) if self.photometric_domain == "linear_srgb" else tensor

    def _from_model_domain(self, tensor: torch.Tensor) -> torch.Tensor:
        return linear_to_srgb(tensor) if self.photometric_domain == "linear_srgb" else tensor

    def _forward_with_explicit_state(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        k_cache: Optional[Sequence[Optional[torch.Tensor]]],
        v_cache: Optional[Sequence[Optional[torch.Tensor]]],
    ):
        """One official step shared by the stream and every replay control.

        Keeping pair construction, autocast, validation, float conversion and
        cache detachment in this single path prevents the former FP16-normal /
        FP32-control mismatch from manufacturing an apparent history effect.
        """

        pair = torch.stack((previous[0], current[0]), dim=0).unsqueeze(0)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self._autocast_dtype,
            enabled=self._autocast_dtype is not None,
        ):
            result = self.model(pair, k_cache, v_cache)
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise RuntimeError(
                "official TURTLE forward must return (restored, k_cache, v_cache)"
            )
        restored, new_k, new_v = result
        if not torch.is_tensor(restored) or restored.ndim != 4:
            raise RuntimeError("official TURTLE restored output must be BCHW")
        if tuple(restored.shape) != tuple(current.shape):
            raise RuntimeError(
                "official TURTLE changed the frame shape: "
                f"input={tuple(current.shape)}, output={tuple(restored.shape)}"
            )
        if not bool(torch.isfinite(restored).all()):
            raise RuntimeError("official TURTLE returned NaN or Inf")
        return (
            restored.to(dtype=torch.float32).clamp(0.0, 1.0),
            _detach_cache(new_k),
            _detach_cache(new_v),
        )

    @torch.no_grad()
    def replay_step(
        self,
        image: torch.Tensor,
        *,
        k_cache: Optional[Sequence[Optional[torch.Tensor]]] = None,
        v_cache: Optional[Sequence[Optional[torch.Tensor]]] = None,
        previous_frame: Optional[torch.Tensor] = None,
    ):
        """Evaluate explicit causal state without mutating the live stream.

        This is the only supported path for reset/repeat/ordered/shuffled
        controls.  It intentionally uses the exact autocast path of ``step``.
        """

        if not torch.is_tensor(image) or image.ndim != 4:
            raise ValueError("TURTLE replay input must be BCHW")
        if image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("TURTLE replay requires one RGB frame")
        current_external = image.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        current = self._to_model_domain(current_external)
        previous = (
            current
            if previous_frame is None
            else self._to_model_domain(
                previous_frame.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
            )
        )
        if tuple(previous.shape) != tuple(current.shape):
            raise ValueError("TURTLE replay previous/current shapes differ")
        restored, new_k, new_v = self._forward_with_explicit_state(
            current, previous, k_cache, v_cache
        )
        return self._from_model_domain(restored), new_k, new_v

    @torch.no_grad()
    def step(self, image: torch.Tensor, timestamp: Any = None) -> torch.Tensor:
        """Consume exactly one frame and advance the official K/V state once."""

        if not torch.is_tensor(image) or image.ndim != 4:
            raise ValueError("turtle_streaming input must be a BCHW tensor")
        if image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError(
                "turtle_streaming requires one RGB frame per incremental call, "
                f"got {tuple(image.shape)}"
            )
        if not bool(torch.isfinite(image).all()):
            raise ValueError("turtle_streaming input contains NaN or Inf")

        current_external = image.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        current = self._to_model_domain(current_external)
        current_resolution = tuple(int(value) for value in current.shape[-2:])
        numeric_timestamp = None if timestamp is None else float(timestamp)

        stream_discontinuity = (
            self.resolution is not None and self.resolution != current_resolution
        ) or (
            numeric_timestamp is not None
            and self.last_timestamp is not None
            and numeric_timestamp <= self.last_timestamp
        )
        if stream_discontinuity:
            self.reset()

        previous = current if self.previous_frame is None else self.previous_frame
        restored, self.k_cache, self.v_cache = self._forward_with_explicit_state(
            current, previous, self.k_cache, self.v_cache
        )
        self.previous_frame = current.detach()
        self.resolution = current_resolution
        self.last_timestamp = numeric_timestamp
        self.frames_seen += 1
        self.cache_updates += 1
        return self._from_model_domain(restored)

    def __call__(self, image: torch.Tensor, timestamp: Any = None) -> torch.Tensor:
        return self.step(image, timestamp=timestamp)
