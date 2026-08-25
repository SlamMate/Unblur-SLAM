"""Pluggable deblurring frontends for Unblur-SLAM.

All backends accept and return a float BCHW tensor in [0, 1].  The default
EVSSM path remains behaviorally equivalent to the original tracker code.
"""

from collections import deque
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


def _unwrap_output(output):
    if isinstance(output, dict):
        for key in ("sharp", "output", "pred", "image"):
            if key in output:
                output = output[key]
                break
        else:
            raise RuntimeError(f"Causal deblur output dict has unsupported keys: {sorted(output)}")
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("Causal deblur model returned an empty sequence")
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Deblur backend must return a tensor, got {type(output)!r}")
    if output.dim() == 5:
        output = output[:, -1]
    if output.dim() == 3:
        output = output.unsqueeze(0)
    if output.dim() != 4:
        raise RuntimeError(f"Deblur backend output must be BCHW or BTCHW, got {tuple(output.shape)}")
    return output


class EVSSMBackend:
    def __init__(self, model, device):
        self.model = model
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, image, timestamp=None):
        del timestamp
        image = image.to(device=self.device, dtype=torch.float32)
        _, _, height, width = image.shape
        h_pad = (4 - height % 4) % 4
        w_pad = (4 - width % 4) % 4
        padded = F.pad(image, (0, w_pad, 0, h_pad), mode="reflect") if (h_pad or w_pad) else image
        output = _unwrap_output(self.model(padded))
        return output[:, :, :height, :width].clamp(0.0, 1.0)


class PrecomputedBackend:
    def __init__(self, root, pattern="{timestamp}.png", device="cuda:0"):
        self.root = Path(root)
        self.pattern = str(pattern)
        self.device = torch.device(device)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Precomputed deblur root does not exist: {self.root}")

    @staticmethod
    def _timestamp_key(timestamp):
        value = float(timestamp)
        return str(int(value)) if value.is_integer() else str(timestamp)

    @torch.no_grad()
    def __call__(self, image, timestamp=None):
        if timestamp is None:
            raise ValueError("precomputed deblur backend requires a frame timestamp/index")
        key = self._timestamp_key(timestamp)
        path = self.root / self.pattern.format(timestamp=key, index=key)
        if not path.is_file():
            raise FileNotFoundError(f"Missing precomputed sharp frame: {path}")
        output = TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
        if output.shape[-2:] != image.shape[-2:]:
            output = F.interpolate(output, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return output.clamp(0.0, 1.0)


class CausalTorchScriptBackend:
    """Streaming video-deblur adapter with an explicit TorchScript contract.

    The model receives ``frames`` shaped [1, T, 3, H, W]. It may return a
    BCHW tensor, a BTCHW tensor, the first item of a tuple/list, or a dict with
    one of: sharp/output/pred/image. The backend keeps only past/current frames.
    """

    def __init__(
        self,
        checkpoint,
        history=5,
        device="cuda:0",
        expected_input_domain=None,
    ):
        self.checkpoint = Path(checkpoint)
        self.history = max(1, int(history))
        self.device = torch.device(device)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Causal TorchScript checkpoint does not exist: {self.checkpoint}")
        extra_files = {"metadata.json": ""}
        self.model = torch.jit.load(
            str(self.checkpoint),
            map_location=self.device,
            _extra_files=extra_files,
        ).eval()
        raw_metadata = extra_files.get("metadata.json", "")
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        try:
            self.metadata = json.loads(raw_metadata) if raw_metadata else {}
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("causal TorchScript metadata.json is invalid") from error
        model_config = self.metadata.get("model_config", {})
        if model_config:
            try:
                trained_history = int(model_config["max_history"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "causal TorchScript metadata must contain model_config.max_history"
                ) from error
            if self.history != trained_history:
                raise ValueError(
                    f"runtime history={self.history} must equal trained "
                    f"max_history={trained_history}"
                )
            if expected_input_domain is not None:
                actual_domain = str(model_config.get("input_domain", "raw")).lower()
                if actual_domain != str(expected_input_domain).lower():
                    raise ValueError(
                        f"causal TorchScript input_domain={actual_domain!r} does not "
                        f"match runtime domain={expected_input_domain!r}"
                    )
        self.frames = deque(maxlen=self.history)

    @torch.no_grad()
    def __call__(self, image, timestamp=None):
        del timestamp
        image = image.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        if image.shape[0] != 1:
            raise ValueError(f"Streaming backend currently requires batch size 1, got {image.shape[0]}")
        frame = image[0]
        if self.frames and self.frames[-1].shape != frame.shape:
            # A new stream/resolution must not be stacked with stale history.
            self.frames.clear()
        self.frames.append(frame)
        history = list(self.frames)
        while len(history) < self.history:
            history.insert(0, history[0])
        sequence = torch.stack(history, dim=0).unsqueeze(0)
        output = _unwrap_output(self.model(sequence))
        if output.shape[-2:] != image.shape[-2:]:
            output = F.interpolate(output, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return output.clamp(0.0, 1.0)


class CausalEVSSMBackend:
    """Frozen EVSSM followed by a causal temporal residual adapter.

    Each raw frame is processed by the original Unblur-SLAM EVSSM exactly
    once.  The adapter then consumes the rolling sequence of EVSSM outputs,
    so its identity initialization reproduces the existing single-frame
    frontend and fine-tuning can only learn a temporal correction.
    """

    def __init__(
        self,
        evssm_model,
        checkpoint,
        history=5,
        device="cuda:0",
        teacher_storage=None,
    ):
        if evssm_model is None:
            raise ValueError("causal_evssm requires an initialized EVSSM model")
        self.evssm = EVSSMBackend(evssm_model, device)
        self.temporal = CausalTorchScriptBackend(
            checkpoint=checkpoint,
            history=history,
            device=device,
            expected_input_domain="evssm",
        )
        embedded_provenance = self.temporal.metadata.get("teacher_provenance", {})
        embedded_storage = str(embedded_provenance.get("storage", ""))
        if teacher_storage is not None and embedded_storage and str(teacher_storage) != embedded_storage:
            raise ValueError(
                "configured causal teacher storage disagrees with TorchScript metadata"
            )
        self.teacher_storage = str(teacher_storage or embedded_storage)
        if self.teacher_storage not in {
            "runtime_evssm_float_tensor",
            "precomputed_png_rgb8",
        }:
            raise ValueError(
                "causal_evssm requires validated runtime or cached EVSSM teacher storage"
            )
        self.last_evssm_output = None
        self.last_temporal_input = None

    @property
    def frames(self):
        return self.temporal.frames

    @torch.no_grad()
    def __call__(self, image, timestamp=None):
        restored = self.evssm(image, timestamp=timestamp)
        self.last_evssm_output = restored.detach()
        if self.teacher_storage == "precomputed_png_rgb8":
            # Training loaded cached PNGs through uint8 RGB. Reproduce the
            # exact round-to-nearest quantization before the temporal adapter.
            temporal_input = torch.round(restored * 255.0) / 255.0
        else:
            temporal_input = restored
        self.last_temporal_input = temporal_input.detach()
        return self.temporal(temporal_input, timestamp=timestamp)


def build_deblur_backend(cfg, evssm_model=None, device="cuda:0"):
    deblur_cfg = cfg.get("deblur", {}) or {}
    name = str(deblur_cfg.get("frontend", "evssm")).lower()
    if name == "evssm":
        if evssm_model is None:
            raise ValueError("EVSSM backend requires an initialized model")
        return name, EVSSMBackend(evssm_model, device)
    if name == "precomputed":
        return name, PrecomputedBackend(
            root=deblur_cfg.get("precomputed_root", ""),
            pattern=deblur_cfg.get("precomputed_pattern", "{timestamp}.png"),
            device=device,
        )
    if name == "causal_torchscript":
        return name, CausalTorchScriptBackend(
            checkpoint=deblur_cfg.get("causal_checkpoint", ""),
            history=deblur_cfg.get("causal_history", 5),
            device=device,
            expected_input_domain="raw",
        )
    if name == "causal_evssm":
        return name, CausalEVSSMBackend(
            evssm_model=evssm_model,
            checkpoint=deblur_cfg.get("causal_checkpoint", ""),
            history=deblur_cfg.get("causal_history", 5),
            device=device,
            teacher_storage=deblur_cfg.get("causal_teacher_storage"),
        )
    if name == "turtle_streaming":
        # Lazy import keeps the default EVSSM path independent of TURTLE's
        # optional YAML/einops runtime and avoids polluting the basicsr namespace.
        from src.turtle_backend import TurtleStreamingBackend

        return name, TurtleStreamingBackend.from_config(
            deblur_cfg, device=device
        )
    if name == "turtle_bsd_streaming":
        # The official BSD checkpoint uses TURTLE-t0 rather than the GoPro
        # t1 architecture.  It nevertheless shares the same incremental,
        # strictly causal K/V runtime contract.
        from src.turtle_official_bsd_backend import (
            OfficialBsdTurtleStreamingBackend,
        )

        return name, OfficialBsdTurtleStreamingBackend.from_config(
            deblur_cfg, device=device
        )
    raise ValueError(f"Unsupported deblur.frontend={name!r}")
