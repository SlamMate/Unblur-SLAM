#!/usr/bin/env python3
"""Core CPU contracts for source-scoped mixed TURTLE training."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_turtle_mixed_defocus import (  # noqa: E402
    ParameterScopes,
    DPDD_CONFIG,
    DPDD_DATASET_MANIFEST_SCHEMA,
    DPDD_REPOSITORY,
    DPDD_REVISION,
    configure_parameter_scopes,
    deterministic_single_schedule,
    deterministic_video_schedule,
    execute_checked_optimizer_step,
    load_dpdd_dataset_contract,
    source_scoped_optimizer_step,
)
from scripts.evaluate_turtle_single_image_defocus import CANONICAL_PAIR_SCHEMA  # noqa: E402
from scripts.train_turtle_streaming import PairedSequenceDataset  # noqa: E402
from src.turtle_backend import load_turtle_model, sha256_file  # noqa: E402


class _TinyScopedTurtle(torch.nn.Module):
    use_both_input = False

    def __init__(self):
        super().__init__()
        self.history = torch.nn.Parameter(torch.tensor(0.1))
        self.spatial = torch.nn.Parameter(torch.tensor(0.9))
        self.random_draws = []

    def forward(self, pair, k_cache=None, v_cache=None):
        self.random_draws.append((int(pair.shape[0]), float(torch.rand(()))))
        current = pair[:, 1]
        history_value = current.new_zeros((current.shape[0], 1, 1, 1))
        if k_cache is not None:
            history_value = k_cache[3].mean(dim=tuple(range(1, k_cache[3].ndim)), keepdim=True)
        restored = current * self.spatial + history_value
        marker = current.mean(dim=(1, 2, 3), keepdim=True) * self.history
        k_new = [None, None, None] + [marker.clone() for _ in range(5)]
        v_new = [None, None, None] + [marker.clone() for _ in range(5)]
        return restored, k_new, v_new


class _CountingAdamW(torch.optim.AdamW):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure=closure)


def _scopes(model):
    return ParameterScopes(
        history=[model.history],
        spatial=[model.spatial],
        history_names=["history"],
        spatial_names=["spatial"],
    )


def test_formal_schedules_match_video_exposure_and_cover_all_dpdd() -> None:
    names = [f"pair_{index:03d}" for index in range(350)]
    single = deterministic_single_schedule(names, seed=17)
    flattened = [index for batch in single for index in batch]
    assert len(single) == 78
    assert all(len(batch) == 5 for batch in single)
    assert len(flattened) == 390
    assert len(set(flattened[:350])) == 350
    assert flattened[350:] == flattened[:40]

    video = deterministic_video_schedule(list(range(26)), seed=17)
    assert len(video) == 78
    assert [pass_index for pass_index, _ in video].count(0) == 26
    assert [pass_index for pass_index, _ in video].count(1) == 26
    assert [pass_index for pass_index, _ in video].count(2) == 26
    assert video == deterministic_video_schedule(list(range(26)), seed=17)


def test_mixed_uses_two_backwards_but_exactly_one_joint_optimizer_step() -> None:
    model = _TinyScopedTurtle()
    scopes = _scopes(model)
    optimizer = _CountingAdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=78)
    single_blur = torch.full((2, 3, 8, 8), 0.2)
    single_sharp = torch.full((2, 3, 8, 8), 0.25)
    video_blur = torch.stack(
        [torch.full((3, 8, 8), value) for value in (0.1, 0.2, 0.3)]
    )
    video_sharp = video_blur + 0.03
    history_before = model.history.detach().clone()
    row = source_scoped_optimizer_step(
        model,
        scopes,
        optimizer,
        mode="M",
        device=torch.device("cpu"),
        single_batch=(single_blur, single_sharp),
        video_batch=(video_blur, video_sharp),
        scheduler=scheduler,
    )
    assert optimizer.step_count == 1
    assert scheduler.last_epoch == 1
    assert "single_loss" in row and "video_loss" in row
    assert not torch.equal(history_before, model.history.detach())


def test_single_only_step_leaves_history_parameter_byte_identical() -> None:
    model = _TinyScopedTurtle()
    scopes = _scopes(model)
    optimizer = _CountingAdamW([model.spatial], lr=1e-2, weight_decay=0.1)
    single_blur = torch.full((2, 3, 8, 8), 0.2)
    single_sharp = torch.full((2, 3, 8, 8), 0.25)
    history_before = model.history.detach().clone()
    source_scoped_optimizer_step(
        model,
        scopes,
        optimizer,
        mode="S",
        device=torch.device("cpu"),
        single_batch=(single_blur, single_sharp),
        video_batch=None,
    )
    assert optimizer.step_count == 1
    assert model.history.grad is None
    assert torch.equal(history_before, model.history.detach())


def test_dpdd_rng_is_restored_before_matched_video_forward() -> None:
    single = (
        torch.full((2, 3, 8, 8), 0.2),
        torch.full((2, 3, 8, 8), 0.25),
    )
    video_blur = torch.stack(
        [torch.full((3, 8, 8), value) for value in (0.1, 0.2, 0.3)]
    )
    video = (video_blur, video_blur + 0.03)

    torch.manual_seed(123)
    video_model = _TinyScopedTurtle()
    video_optimizer = _CountingAdamW(video_model.parameters(), lr=0.0)
    source_scoped_optimizer_step(
        video_model,
        _scopes(video_model),
        video_optimizer,
        mode="V",
        device=torch.device("cpu"),
        single_batch=None,
        video_batch=video,
    )
    video_draws = [value for batch, value in video_model.random_draws if batch == 1]

    torch.manual_seed(123)
    mixed_model = _TinyScopedTurtle()
    mixed_optimizer = _CountingAdamW(mixed_model.parameters(), lr=0.0)
    source_scoped_optimizer_step(
        mixed_model,
        _scopes(mixed_model),
        mixed_optimizer,
        mode="M",
        device=torch.device("cpu"),
        single_batch=single,
        video_batch=video,
    )
    mixed_video_draws = [value for batch, value in mixed_model.random_draws if batch == 1]
    assert mixed_video_draws == video_draws


def test_pinned_model_parameter_partition_is_exact() -> None:
    model, _ = load_turtle_model(
        "/srv/szha0669/unblur-slam/external/TURTLE",
        "/srv/szha0669/unblur-slam/pretrained/turtle/GoPro_Deblur.pth",
        config="/srv/szha0669/unblur-slam/external/TURTLE/options/Turtle_Deblur_Gopro.yml",
        device="cpu",
    )
    scopes = configure_parameter_scopes(model, "M")
    assert len(scopes.history) == 56
    assert sum(parameter.numel() for parameter in scopes.history) == 3_475_994
    assert len(scopes.spatial) == 30
    assert sum(parameter.numel() for parameter in scopes.spatial) == 105_283


def test_real_replica_manifest_resolves_only_against_explicit_video_root() -> None:
    manifest = Path(
        "/srv/szha0669/unblur-slam/causal_video_data/"
        "replica424_v1/manifests/train.jsonl"
    )
    video_root = Path("/srv/szha0669/unblur-slam/causal_video_data")
    dataset = PairedSequenceDataset(
        manifest,
        root=video_root,
        crop_size=128,
        augment=False,
        seed=17,
    )
    assert len(dataset.records) == 127
    assert sum(len(record.blurry) >= 3 for record in dataset.records) == 26
    assert dataset.records[0].blurry[0] == (
        video_root / "replica_blur/room_1/blur/rgb_72.png"
    ).resolve()
    sample = dataset[0]
    assert tuple(sample["blurry"].shape) == (5, 3, 128, 128)
    assert tuple(sample["sharp"].shape) == (5, 3, 128, 128)


def test_dpdd_dataset_manifest_binds_revision_license_disclosure_and_train_hash() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifests = root / "manifests"
        manifests.mkdir()
        train = manifests / "train.jsonl"
        validation = manifests / "validation.jsonl"
        train.write_text('{"fixture":"train"}\n', encoding="utf-8")
        validation.write_text('{"fixture":"validation"}\n', encoding="utf-8")
        dataset_manifest = root / "dataset_manifest.json"
        dataset_manifest.write_text(
            json.dumps(
                {
                    "schema": DPDD_DATASET_MANIFEST_SCHEMA,
                    "repository": DPDD_REPOSITORY,
                    "revision": DPDD_REVISION,
                    "config": DPDD_CONFIG,
                    "splits": {"train": 350, "validation": 74},
                    "distribution": {
                        "dataset_card_declared_license": "mit",
                        "license_scope_warning": "mirror claim is not original rights",
                    },
                    "test_disclosure": {
                        "metadata_pristine": False,
                        "images_decoded": False,
                        "pixels_opened": False,
                        "metrics_opened": False,
                        "split_supported_by_this_materializer": False,
                    },
                    "canonical_manifests": {
                        "train": {
                            "path": "manifests/train.jsonl",
                            "sha256": sha256_file(train),
                            "rows": 350,
                            "schema": CANONICAL_PAIR_SCHEMA,
                            "paths_relative_to": "dataset_root",
                        },
                        "validation": {
                            "path": "manifests/validation.jsonl",
                            "sha256": sha256_file(validation),
                            "rows": 74,
                            "schema": CANONICAL_PAIR_SCHEMA,
                            "paths_relative_to": "dataset_root",
                        },
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        provenance = load_dpdd_dataset_contract(
            dataset_manifest,
            expected_dataset_manifest_sha256=sha256_file(dataset_manifest),
            train_manifest=train,
            expected_train_manifest_sha256=sha256_file(train),
        )
        assert provenance["repository"] == DPDD_REPOSITORY
        assert provenance["revision"] == DPDD_REVISION
        assert provenance["test_metadata_pristine"] is False
        assert provenance["canonical_train_manifest_sha256"] == sha256_file(train)


def test_amp_skip_fails_before_scheduler_advance() -> None:
    class _SkippingScaler:
        def step(self, optimizer):
            return None

        def update(self):
            return None

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = _CountingAdamW([parameter], lr=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=78)
    try:
        execute_checked_optimizer_step(
            optimizer,
            amp_enabled=True,
            scaler=_SkippingScaler(),
            scheduler=scheduler,
        )
    except FloatingPointError as error:
        assert "skipped" in str(error)
    else:
        raise AssertionError("a skipped AMP update was counted as an executed step")
    assert optimizer.step_count == 0
    assert scheduler.last_epoch == 0


if __name__ == "__main__":
    test_formal_schedules_match_video_exposure_and_cover_all_dpdd()
    test_mixed_uses_two_backwards_but_exactly_one_joint_optimizer_step()
    test_single_only_step_leaves_history_parameter_byte_identical()
    test_dpdd_rng_is_restored_before_matched_video_forward()
    test_pinned_model_parameter_partition_is_exact()
    test_real_replica_manifest_resolves_only_against_explicit_video_root()
    test_dpdd_dataset_manifest_binds_revision_license_disclosure_and_train_hash()
    test_amp_skip_fails_before_scheduler_advance()
    print("8 mixed-training CPU contracts passed")
