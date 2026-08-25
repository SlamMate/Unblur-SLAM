#!/usr/bin/env python3
"""CPU contracts for optimizer-safe official-ReSplat active-map merging."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_optional_cuda_imports() -> None:
    # GaussianModel's map editing code is pure torch, but its module imports
    # I/O/CUDA extensions that are intentionally absent from the CPU test env.
    def missing(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is None
        except (ImportError, ValueError):
            return name not in sys.modules

    if missing("open3d"):
        sys.modules.setdefault("open3d", ModuleType("open3d"))
    if missing("plyfile"):
        plyfile = ModuleType("plyfile")
        plyfile.PlyData = object
        plyfile.PlyElement = object
        sys.modules.setdefault("plyfile", plyfile)
    if missing("simple_knn"):
        simple_knn = ModuleType("simple_knn")
        extension = ModuleType("simple_knn._C")
        extension.distCUDA2 = lambda value: value
        sys.modules.setdefault("simple_knn", simple_knn)
        sys.modules.setdefault("simple_knn._C", extension)


_stub_optional_cuda_imports()

from src.refinement.active_map_merge import (  # noqa: E402
    ACTIVE_MAP_MERGE_SCHEMA,
    ActiveMapMergeConfig,
    optimizer_state_shapes_valid,
)
from thirdparty.gaussian_splatting.scene.gaussian_model import (  # noqa: E402
    GaussianModel,
)


PARAM_ATTRS = {
    "xyz": "_xyz",
    "f_dc": "_features_dc",
    "f_rest": "_features_rest",
    "opacity": "_opacity",
    "scaling": "_scaling",
    "rotation": "_rotation",
}


def _model() -> GaussianModel:
    model = GaussianModel.__new__(GaussianModel)
    model._xyz = torch.nn.Parameter(
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
    )
    model._features_dc = torch.nn.Parameter(torch.full((4, 1, 3), 0.1))
    model._features_rest = torch.nn.Parameter(torch.full((4, 3, 3), 0.2))
    model._opacity = torch.nn.Parameter(torch.zeros((4, 1)))
    model._scaling = torch.nn.Parameter(torch.full((4, 3), -3.0))
    model._rotation = torch.nn.Parameter(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
    )
    model.unique_kfIDs = torch.tensor([10, 10, 20, 30], dtype=torch.int32)
    model.n_obs = torch.tensor([2, 3, 4, 5], dtype=torch.int32)
    model.xyz_gradient_accum = torch.ones((4, 1))
    model.denom = torch.ones((4, 1))
    model.max_radii2D = torch.ones(4)
    groups = [
        {"params": [getattr(model, attribute)], "lr": 1.0e-3, "name": name}
        for name, attribute in PARAM_ATTRS.items()
    ]
    model.optimizer = torch.optim.Adam(groups, lr=0.0)
    # Materialize Adam moments so the test checks prefix preservation, not
    # merely lazy/no-state parameter replacement.
    loss = sum(
        getattr(model, attribute).square().sum()
        for attribute in PARAM_ATTRS.values()
    )
    loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    assert optimizer_state_shapes_valid(model)
    return model


def _batch(means: torch.Tensor, owners: torch.Tensor | None = None) -> dict:
    count = int(means.shape[0])
    harmonics = torch.arange(count * 3 * 4, dtype=torch.float32).reshape(count, 3, 4)
    return {
        "means_world": means,
        "scales_linear": torch.full((count, 3), 0.02),
        "rotations_wxyz_world": torch.tensor([[2.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        "harmonics_world": harmonics,
        "opacities_probability": torch.full((count,), 0.8),
        "owner_kf_ids": (
            owners if owners is not None else torch.full((count,), 10, dtype=torch.int64)
        ),
        "replace_kf_ids": [10],
    }


def _optimizer_moments(model: GaussianModel) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    result = {}
    for group in model.optimizer.param_groups:
        name = group.get("name")
        if name in PARAM_ATTRS:
            state = model.optimizer.state[group["params"][0]]
            result[name] = (state["exp_avg"].clone(), state["exp_avg_sq"].clone())
    return result


def _optimizer_can_continue(model: GaussianModel) -> None:
    loss = sum(getattr(model, attribute).square().sum() for attribute in PARAM_ATTRS.values())
    loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    assert optimizer_state_shapes_valid(model)


def test_append_filters_and_preserves_adam_prefix() -> None:
    model = _model()
    moments = _optimizer_moments(model)
    batch = _batch(
        torch.tensor(
            [
                [0.001, 0.0, 0.0],  # collides with active map
                [0.200, 0.0, 0.0],  # wins incoming voxel
                [0.201, 0.0, 0.0],  # lower-opacity duplicate
                [0.400, 0.0, 0.0],  # accepted
                [0.600, 0.0, 0.0],  # rejected by opacity
            ]
        )
    )
    batch["opacities_probability"] = torch.tensor([0.9, 0.8, 0.6, 0.7, 0.05])
    report = model.merge_resplat_submap(
        **batch,
        merge_config={
            "mode": "append",
            "voxel_size": 0.05,
            "min_opacity": 0.1,
            "min_new_gaussians": 2,
            "max_new_gaussians": 10,
            "zero_sh_rest": True,
        },
    )

    assert report["schema"] == ACTIVE_MAP_MERGE_SCHEMA
    assert report["status"] == "merged"
    assert report["active_map_changed"] is True
    assert report["before_count"] == 4
    assert report["rejected_opacity"] == 1
    assert report["rejected_incoming_voxel"] == 1
    assert report["rejected_active_voxel"] == 1
    assert report["accepted_count"] == 2
    assert report["after_count"] == 6
    assert model.unique_kfIDs.tolist() == [10, 10, 20, 30, 10, 10]
    assert bool((model._features_rest[-2:] == 0).all())
    assert bool(torch.allclose(torch.linalg.vector_norm(model._rotation[-2:], dim=1), torch.ones(2)))
    for group in model.optimizer.param_groups:
        name = group.get("name")
        if name not in moments:
            continue
        state = model.optimizer.state[group["params"][0]]
        before_avg, before_sq = moments[name]
        assert bool(torch.equal(state["exp_avg"][:4], before_avg))
        assert bool(torch.equal(state["exp_avg_sq"][:4], before_sq))
        assert bool((state["exp_avg"][4:] == 0).all())
        assert bool((state["exp_avg_sq"][4:] == 0).all())
    _optimizer_can_continue(model)


def test_replace_owned_keeps_unrelated_rows_and_their_adam_state() -> None:
    model = _model()
    original_xyz = model._xyz.detach().clone()
    moments = _optimizer_moments(model)
    batch = _batch(torch.tensor([[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    report = model.merge_resplat_submap(
        **batch,
        merge_config={
            "mode": "replace_owned",
            "voxel_size": 0.01,
            "max_new_gaussians": 2,
            "zero_sh_rest": False,
        },
    )

    assert report["removed_owned_count"] == 2
    assert report["accepted_count"] == 2
    assert report["after_count"] == 4
    assert model.unique_kfIDs.tolist() == [20, 30, 10, 10]
    assert bool(torch.allclose(model._xyz[:2], original_xyz[2:]))
    assert bool((model._features_rest[-2:] != 0).any())
    for group in model.optimizer.param_groups:
        name = group.get("name")
        if name not in moments:
            continue
        state = model.optimizer.state[group["params"][0]]
        before_avg, before_sq = moments[name]
        assert bool(torch.equal(state["exp_avg"][:2], before_avg[2:]))
        assert bool(torch.equal(state["exp_avg_sq"][:2], before_sq[2:]))
        assert bool((state["exp_avg"][2:] == 0).all())
        assert bool((state["exp_avg_sq"][2:] == 0).all())
    _optimizer_can_continue(model)


def test_too_few_is_transactional_and_gate_accounting_is_explicit() -> None:
    model = _model()
    parameter_ids = {name: id(getattr(model, attribute)) for name, attribute in PARAM_ATTRS.items()}
    tensors = {name: getattr(model, attribute).detach().clone() for name, attribute in PARAM_ATTRS.items()}
    owners_before = model.unique_kfIDs.clone()
    batch = _batch(
        torch.tensor(
            [
                [4.0, 0.0, 0.0],
                [float("nan"), 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [7.0, 0.0, 0.0],
            ]
        ),
        owners=torch.tensor([10, 10, 10, 10, 99]),
    )
    batch["scales_linear"][2, 0] = 2.0
    batch["rotations_wxyz_world"][3].zero_()
    report = model.merge_resplat_submap(
        **batch,
        merge_config={
            "mode": "replace_owned",
            "min_new_gaussians": 2,
            "max_new_gaussians": 10,
        },
    )

    assert report["status"] == "rejected_too_few_accepted"
    assert report["active_map_changed"] is False
    assert report["rejected_nonfinite"] == 1
    assert report["rejected_scale"] == 1
    assert report["rejected_rotation"] == 1
    assert report["rejected_owner"] == 1
    assert report["accepted_count"] == 1
    assert report["removed_owned_count"] == 0
    for name, attribute in PARAM_ATTRS.items():
        assert id(getattr(model, attribute)) == parameter_ids[name]
        assert bool(torch.equal(getattr(model, attribute), tensors[name]))
    assert bool(torch.equal(model.unique_kfIDs, owners_before))
    assert optimizer_state_shapes_valid(model)


def test_config_and_shape_validation_fail_closed() -> None:
    model = _model()
    before = model._xyz.detach().clone()
    batch = _batch(torch.tensor([[4.0, 0.0, 0.0]]))
    try:
        model.merge_resplat_submap(
            **batch,
            merge_config={"mode": "replace_everything"},
        )
    except ValueError as error:
        assert "append or replace_owned" in str(error)
    else:
        raise AssertionError("unsafe merge mode was accepted")
    assert bool(torch.equal(model._xyz, before))

    malformed = dict(batch)
    malformed["harmonics_world"] = torch.zeros((1, 4, 3))
    try:
        model.merge_resplat_submap(**malformed)
    except ValueError as error:
        assert "harmonics_world" in str(error)
    else:
        raise AssertionError("wrong SH layout was accepted")
    assert bool(torch.equal(model._xyz, before))

    for invalid in (
        {"min_opacity": 0.0},
        {"voxel_size": 0.0},
        {"min_new_gaussians": 3, "max_new_gaussians": 2},
        {"unknown_gate": 1},
    ):
        try:
            ActiveMapMergeConfig.from_value(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config was accepted: {invalid}")


def main() -> None:
    test_append_filters_and_preserves_adam_prefix()
    test_replace_owned_keeps_unrelated_rows_and_their_adam_state()
    test_too_few_is_transactional_and_gate_accounting_is_explicit()
    test_config_and_shape_validation_fail_closed()
    print("active_map_merge_cpu_contracts=PASS")


if __name__ == "__main__":
    main()
