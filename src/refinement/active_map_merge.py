"""Optimizer-safe insertion of converted ReSplat Gaussians into the live map.

The official ReSplat sidecar does *not* emit tensors in the active
``GaussianModel`` convention.  In particular, its saved means live in a
middle-camera gauge and its quaternions are source-camera-local ``xyzw``.
This module deliberately starts after that conversion: callers must provide
world-space means, world-space ``wxyz`` quaternions, and world-basis spherical
harmonics in ReSplat's ``[N, 3, SH]`` layout.

Keeping the conversion boundary explicit prevents a native sidecar from being
silently appended with the wrong coordinate or quaternion convention.  The
merge itself is conservative and fail-closed: all validation and filtering is
completed before the first active-map mutation, and too few accepted points
leaves the map untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Mapping, Optional

import torch


ACTIVE_MAP_MERGE_SCHEMA = "unblur_slam.active_gaussian_merge.v1"
_PARAMETER_NAMES = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")


@dataclass(frozen=True)
class ActiveMapMergeConfig:
    """Validated gates for one active-map merge.

    ``append`` is the safe first experiment because it never deletes an
    existing Gaussian.  ``replace_owned`` removes only active Gaussians whose
    anchor keyframe is explicitly listed in ``replace_kf_ids`` and only after
    the incoming batch has passed every gate.
    """

    mode: str = "append"
    min_opacity: float = 0.10
    min_scale: float = 1.0e-6
    max_scale: float = 0.20
    voxel_size: float = 0.01
    max_new_gaussians: int = 20_000
    min_new_gaussians: int = 1
    zero_sh_rest: bool = True
    max_abs_position: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mode not in {"append", "replace_owned"}:
            raise ValueError("active-map merge mode must be append or replace_owned")
        for name in ("min_opacity", "min_scale", "max_scale", "voxel_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        if not 0.0 < float(self.min_opacity) < 1.0:
            raise ValueError("min_opacity must be strictly between zero and one")
        if not 0.0 < float(self.min_scale) <= float(self.max_scale):
            raise ValueError("scale bounds must satisfy 0 < min_scale <= max_scale")
        if float(self.voxel_size) <= 0.0:
            raise ValueError("voxel_size must be positive")
        for name in ("max_new_gaussians", "min_new_gaussians"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if int(self.min_new_gaussians) > int(self.max_new_gaussians):
            raise ValueError("min_new_gaussians cannot exceed max_new_gaussians")
        if not isinstance(self.zero_sh_rest, bool):
            raise ValueError("zero_sh_rest must be boolean")
        if self.max_abs_position is not None:
            if isinstance(self.max_abs_position, bool) or not math.isfinite(
                float(self.max_abs_position)
            ):
                raise ValueError("max_abs_position must be finite when set")
            if float(self.max_abs_position) <= 0.0:
                raise ValueError("max_abs_position must be positive when set")

    @classmethod
    def from_value(
        cls, value: "ActiveMapMergeConfig | Mapping[str, Any] | None"
    ) -> "ActiveMapMergeConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("merge_config must be a mapping or ActiveMapMergeConfig")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                "unknown active-map merge config fields: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(value))


def _named_model_tensors(model: Any) -> dict[str, torch.Tensor]:
    return {
        "xyz": model._xyz,
        "f_dc": model._features_dc,
        "f_rest": model._features_rest,
        "opacity": model._opacity,
        "scaling": model._scaling,
        "rotation": model._rotation,
    }


def _validate_active_model(model: Any) -> None:
    tensors = _named_model_tensors(model)
    count = int(tensors["xyz"].shape[0])
    if tensors["xyz"].ndim != 2 or tensors["xyz"].shape[1:] != (3,):
        raise ValueError("active Gaussian xyz tensor must have shape [N,3]")
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or int(tensor.shape[0]) != count:
            raise ValueError(f"active Gaussian tensor {name} is not row-aligned")
    if tensors["f_dc"].ndim != 3 or tensors["f_dc"].shape[1:] != (1, 3):
        raise ValueError("active DC features must have shape [N,1,3]")
    if tensors["f_rest"].ndim != 3 or tensors["f_rest"].shape[2] != 3:
        raise ValueError("active residual SH features must have shape [N,K,3]")
    if tensors["opacity"].shape[1:] != (1,):
        raise ValueError("active opacity tensor must have shape [N,1]")
    if tensors["rotation"].shape[1:] != (4,):
        raise ValueError("active rotation tensor must have shape [N,4]")
    if tensors["scaling"].ndim != 2:
        raise ValueError("active scaling tensor must have shape [N,S]")
    if getattr(model, "optimizer", None) is None:
        raise RuntimeError("active-map merge requires an initialized optimizer")
    if not isinstance(model.unique_kfIDs, torch.Tensor) or model.unique_kfIDs.device.type != "cpu":
        raise ValueError("active unique_kfIDs must remain a CPU tensor")
    if not isinstance(model.n_obs, torch.Tensor) or model.n_obs.device.type != "cpu":
        raise ValueError("active n_obs must remain a CPU tensor")
    if int(model.unique_kfIDs.numel()) != count or int(model.n_obs.numel()) != count:
        raise ValueError("active ownership metadata is not row-aligned")


def optimizer_state_shapes_valid(model: Any) -> bool:
    """Return whether all named Adam groups match the live model tensors."""

    expected = _named_model_tensors(model)
    seen: set[str] = set()
    for group in model.optimizer.param_groups:
        name = group.get("name")
        if name not in expected:
            continue
        if name in seen or len(group.get("params", ())) != 1:
            return False
        seen.add(name)
        parameter = group["params"][0]
        if parameter is not expected[name]:
            return False
        state = model.optimizer.state.get(parameter)
        if state is not None:
            for moment_name in ("exp_avg", "exp_avg_sq"):
                moment = state.get(moment_name)
                if moment is None or tuple(moment.shape) != tuple(parameter.shape):
                    return False
    return seen == set(expected)


def _as_float_tensor(value: Any, *, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=like.device, dtype=like.dtype).contiguous()
    return torch.as_tensor(value, device=like.device, dtype=like.dtype).contiguous()


def _as_owner_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu")
    else:
        value = torch.as_tensor(value, device="cpu")
    if value.dtype == torch.bool or value.ndim != 1:
        raise ValueError("owner_kf_ids must be a rank-one integer tensor")
    if value.is_floating_point():
        if not bool(torch.isfinite(value).all()) or not bool((value == value.round()).all()):
            raise ValueError("owner_kf_ids must contain finite integers")
    return value.to(dtype=torch.int64).contiguous()


def _authorized_owner_ids(value: Any) -> torch.Tensor:
    owners = _as_owner_tensor(value)
    if owners.numel() == 0:
        raise ValueError("replace_kf_ids must list at least one authorized owner")
    owners = torch.unique(owners, sorted=True)
    if bool((owners < 0).any()):
        raise ValueError("replace_kf_ids cannot contain negative keyframe ids")
    return owners


def _keep_best_per_voxel(
    means: torch.Tensor, opacities: torch.Tensor
) -> torch.Tensor:
    """Return a deterministic mask selecting max opacity, then first row."""

    if means.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=means.device)
    _voxels, inverse = torch.unique(means, dim=0, sorted=True, return_inverse=True)
    group_count = int(inverse.max().item()) + 1
    scores = opacities.reshape(-1)
    maxima = torch.full(
        (group_count,), -torch.inf, dtype=scores.dtype, device=scores.device
    )
    maxima.scatter_reduce_(0, inverse, scores, reduce="amax", include_self=True)
    tied = scores == maxima[inverse]
    row = torch.arange(scores.numel(), device=scores.device, dtype=torch.int64)
    sentinel = torch.full_like(row, scores.numel())
    first = torch.full(
        (group_count,), scores.numel(), dtype=torch.int64, device=scores.device
    )
    first.scatter_reduce_(
        0, inverse, torch.where(tied, row, sentinel), reduce="amin", include_self=True
    )
    return row == first[inverse]


def _active_voxel_collisions(
    incoming_voxels: torch.Tensor, active_xyz: torch.Tensor, voxel_size: float
) -> torch.Tensor:
    if incoming_voxels.shape[0] == 0 or active_xyz.shape[0] == 0:
        return torch.zeros(
            incoming_voxels.shape[0], dtype=torch.bool, device=incoming_voxels.device
        )
    finite_active = torch.isfinite(active_xyz).all(dim=1)
    if not bool(finite_active.any()):
        return torch.zeros(
            incoming_voxels.shape[0], dtype=torch.bool, device=incoming_voxels.device
        )
    active_voxels = torch.floor(active_xyz[finite_active] / voxel_size).to(torch.int64)
    combined = torch.cat((active_voxels, incoming_voxels), dim=0)
    _unique, inverse = torch.unique(combined, dim=0, sorted=True, return_inverse=True)
    active_groups = torch.unique(inverse[: active_voxels.shape[0]], sorted=True)
    return torch.isin(inverse[active_voxels.shape[0] :], active_groups)


def merge_resplat_submap_into_active_model(
    model: Any,
    *,
    means_world: Any,
    scales_linear: Any,
    rotations_wxyz_world: Any,
    harmonics_world: Any,
    opacities_probability: Any,
    owner_kf_ids: Any,
    replace_kf_ids: Any,
    merge_config: ActiveMapMergeConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate and merge a converted official-ReSplat batch into ``model``.

    This function mutates the active map only when at least
    ``min_new_gaussians`` survive.  Optimizer parameters and Adam moments are
    resized through the model's existing prune/densification APIs, so mapping
    can safely continue after a successful merge.
    """

    started = time.perf_counter()
    config = ActiveMapMergeConfig.from_value(merge_config)
    _validate_active_model(model)
    if not optimizer_state_shapes_valid(model):
        raise RuntimeError("optimizer groups/states are not aligned before active merge")

    tensors = _named_model_tensors(model)
    xyz = _as_float_tensor(means_world, like=tensors["xyz"])
    scales = _as_float_tensor(scales_linear, like=tensors["scaling"])
    rotations = _as_float_tensor(rotations_wxyz_world, like=tensors["rotation"])
    harmonics = _as_float_tensor(harmonics_world, like=tensors["f_dc"])
    opacities = _as_float_tensor(opacities_probability, like=tensors["opacity"])
    owners = _as_owner_tensor(owner_kf_ids)
    authorized = _authorized_owner_ids(replace_kf_ids)

    count = int(xyz.shape[0]) if xyz.ndim >= 1 else -1
    expected_sh = 1 + int(tensors["f_rest"].shape[1])
    expected_scale = int(tensors["scaling"].shape[1])
    if xyz.ndim != 2 or tuple(xyz.shape[1:]) != (3,):
        raise ValueError("means_world must have shape [N,3]")
    if scales.ndim != 2 or tuple(scales.shape) != (count, expected_scale):
        raise ValueError(f"scales_linear must have shape [N,{expected_scale}]")
    if rotations.ndim != 2 or tuple(rotations.shape) != (count, 4):
        raise ValueError("rotations_wxyz_world must have shape [N,4]")
    if harmonics.ndim != 3 or tuple(harmonics.shape) != (count, 3, expected_sh):
        raise ValueError(f"harmonics_world must have shape [N,3,{expected_sh}]")
    if opacities.ndim == 1:
        opacities = opacities[:, None]
    if opacities.ndim != 2 or tuple(opacities.shape) != (count, 1):
        raise ValueError("opacities_probability must have shape [N] or [N,1]")
    if int(owners.numel()) != count:
        raise ValueError("owner_kf_ids must contain one id per incoming Gaussian")

    report: dict[str, Any] = {
        "schema": ACTIVE_MAP_MERGE_SCHEMA,
        "mode": config.mode,
        "config": asdict(config),
        "active_map_changed": False,
        "before_count": int(tensors["xyz"].shape[0]),
        "input_count": count,
        "removed_owned_count": 0,
        "rejected_nonfinite": 0,
        "rejected_position": 0,
        "rejected_opacity": 0,
        "rejected_scale": 0,
        "rejected_rotation": 0,
        "rejected_owner": 0,
        "rejected_incoming_voxel": 0,
        "rejected_active_voxel": 0,
        "rejected_capacity": 0,
        "accepted_count": 0,
        "after_count": int(tensors["xyz"].shape[0]),
        "optimizer_state_shapes_valid": True,
    }

    alive = torch.ones(count, dtype=torch.bool, device=xyz.device)

    def reject(key: str, valid: torch.Tensor) -> None:
        nonlocal alive
        rejected = alive & ~valid
        report[key] = int(rejected.sum().item())
        alive = alive & valid

    finite = (
        torch.isfinite(xyz).all(dim=1)
        & torch.isfinite(scales).all(dim=1)
        & torch.isfinite(rotations).all(dim=1)
        & torch.isfinite(harmonics).all(dim=(1, 2))
        & torch.isfinite(opacities).all(dim=1)
    )
    reject("rejected_nonfinite", finite)
    if config.max_abs_position is None:
        position_valid = torch.ones_like(alive)
    else:
        position_valid = (xyz.abs() <= float(config.max_abs_position)).all(dim=1)
    reject("rejected_position", position_valid)
    opacity_valid = (
        (opacities[:, 0] > 0.0)
        & (opacities[:, 0] < 1.0)
        & (opacities[:, 0] >= float(config.min_opacity))
    )
    reject("rejected_opacity", opacity_valid)
    scale_valid = (
        (scales >= float(config.min_scale)).all(dim=1)
        & (scales <= float(config.max_scale)).all(dim=1)
    )
    reject("rejected_scale", scale_valid)
    rotation_norm = torch.linalg.vector_norm(rotations, dim=1)
    reject("rejected_rotation", rotation_norm > 1.0e-8)
    owner_valid_cpu = torch.isin(owners, authorized)
    reject("rejected_owner", owner_valid_cpu.to(device=alive.device))

    candidate = torch.nonzero(alive, as_tuple=False).flatten()
    if candidate.numel():
        voxels = torch.floor(xyz[candidate] / float(config.voxel_size)).to(torch.int64)
        keep = _keep_best_per_voxel(voxels, opacities[candidate])
        report["rejected_incoming_voxel"] = int((~keep).sum().item())
        candidate = candidate[keep]
        voxels = voxels[keep]
    else:
        voxels = torch.empty((0, 3), dtype=torch.int64, device=xyz.device)

    current_owners = model.unique_kfIDs.to(dtype=torch.int64)
    owned_mask_cpu = torch.isin(current_owners, authorized)
    if config.mode == "replace_owned":
        retained_mask_device = (~owned_mask_cpu).to(device=xyz.device)
        removable = int(owned_mask_cpu.sum().item())
    else:
        retained_mask_device = torch.ones(
            report["before_count"], dtype=torch.bool, device=xyz.device
        )
        removable = 0
    if candidate.numel():
        collisions = _active_voxel_collisions(
            voxels, tensors["xyz"][retained_mask_device], float(config.voxel_size)
        )
        report["rejected_active_voxel"] = int(collisions.sum().item())
        candidate = candidate[~collisions]

    capacity = int(config.max_new_gaussians)
    if int(candidate.numel()) > capacity:
        score = opacities[candidate, 0]
        order = torch.argsort(score, descending=True, stable=True)
        report["rejected_capacity"] = int(candidate.numel()) - capacity
        candidate = candidate[order[:capacity]]

    accepted = int(candidate.numel())
    report["accepted_count"] = accepted
    if accepted < int(config.min_new_gaussians):
        report["status"] = "rejected_too_few_accepted"
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        return report

    selected_xyz = xyz[candidate]
    selected_scales = scales[candidate]
    selected_rotations = rotations[candidate]
    selected_rotations = selected_rotations / torch.linalg.vector_norm(
        selected_rotations, dim=1, keepdim=True
    )
    selected_harmonics = harmonics[candidate].transpose(1, 2).contiguous()
    if config.zero_sh_rest and selected_harmonics.shape[1] > 1:
        selected_harmonics[:, 1:, :] = 0.0
    selected_opacities = opacities[candidate]
    selected_owners = owners[candidate.detach().cpu()].to(dtype=torch.int32)

    opacity_logits = torch.log(selected_opacities) - torch.log1p(-selected_opacities)
    log_scales = torch.log(selected_scales)
    with torch.no_grad():
        if config.mode == "replace_owned" and removable:
            model.prune_points(owned_mask_cpu.to(device=xyz.device))
            report["removed_owned_count"] = removable
        model.densification_postfix(
            selected_xyz,
            selected_harmonics[:, :1, :],
            selected_harmonics[:, 1:, :],
            opacity_logits,
            log_scales,
            selected_rotations,
            new_kf_ids=selected_owners,
            new_n_obs=torch.zeros(accepted, dtype=torch.int32, device="cpu"),
        )

    report["after_count"] = int(model._xyz.shape[0])
    report["active_map_changed"] = True
    report["optimizer_state_shapes_valid"] = optimizer_state_shapes_valid(model)
    if not report["optimizer_state_shapes_valid"]:
        raise RuntimeError("optimizer groups/states are not aligned after active merge")
    expected_after = report["before_count"] - removable + accepted
    if report["after_count"] != expected_after:
        raise RuntimeError("active-map size changed by an unexpected amount")
    report["status"] = "merged"
    report["elapsed_seconds"] = float(time.perf_counter() - started)
    return report


__all__ = [
    "ACTIVE_MAP_MERGE_SCHEMA",
    "ActiveMapMergeConfig",
    "merge_resplat_submap_into_active_model",
    "optimizer_state_shapes_valid",
]
