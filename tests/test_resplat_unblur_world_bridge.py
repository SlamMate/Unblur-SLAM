#!/usr/bin/env python3
"""CPU contracts for exact state3 and ReSplat-to-Unblur coordinates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_official_resplat_sidecar import (  # noqa: E402
    run_official_refinement_states_core,
)
from src.refinement.official_resplat_sidecar import (  # noqa: E402
    OfficialReSplatSidecarQueue,
    SidecarConfig,
    SidecarFrameInput,
    load_snapshot,
    materialize_closed_submap_snapshot,
    sha256_file,
    verify_result_artifacts,
    verify_unblur_world_artifact,
)
from src.refinement.resplat_unblur_bridge import (  # noqa: E402
    array_manifest,
    build_unblur_world_arrays,
)


REAL_STATE4_ROOT = Path(
    "/srv/szha0669/unblur-slam/official_resplat_sidecar_smoke/"
    "fr2_xyz_motion_only_first_closed8_turtle_gopro_small8v_v4"
)


def _pivot() -> np.ndarray:
    angle = 0.43
    c, s = np.cos(angle), np.sin(angle)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    result[:3, 3] = (1.2, -0.4, 0.7)
    return result


def _local_state(count: int = 16) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(91)
    means = rng.normal(size=(count, 3)).astype(np.float32)
    rotations = np.repeat(np.eye(3)[None], count, axis=0)
    scales = rng.uniform(0.02, 0.3, size=(count, 3))
    covariances = np.einsum(
        "nij,nj,nkj->nik", rotations, scales * scales, rotations
    ).astype(np.float32)
    return {
        "means": means,
        "covariances": covariances,
        "harmonics": rng.normal(size=(count, 3, 16)).astype(np.float32),
        "opacities": rng.uniform(0.05, 0.95, size=count).astype(np.float32),
        "scales": scales.astype(np.float32),
        "rotations": np.tile(np.asarray((0.0, 0.0, 0.0, 1.0), np.float32), (count, 1)),
    }


def test_exact_state3_does_not_compute_state4() -> None:
    class VersionValue:
        _version = 0

    initial = SimpleNamespace(means=VersionValue())
    states = [SimpleNamespace(means=VersionValue()) for _ in range(3)]

    class Encoder:
        def __call__(self, context, **kwargs):
            assert context == "past8"
            assert kwargs["global_step"] == 0
            return {"gaussians": initial, "condition_features": "condition"}

        def forward_update(self, context, target, condition, state, decoder, remain):
            assert (context, target) == ("past8", "same8")
            assert condition == "condition" and state is initial
            assert decoder == "decoder" and remain is None
            return {"gaussian": states}

    init_result, state3, contract = run_official_refinement_states_core(
        encoder=Encoder(),
        decoder="decoder",
        batch={"context": "past8", "target": "same8"},
        stage_runner=lambda _name, operation: operation(),
        num_refine=3,
    )
    assert init_result is initial
    assert state3 is states[2]
    assert contract["requested_recurrent_updates"] == 3
    assert contract["returned_recurrent_states"] == 3
    assert contract["selected_state_index_zero_based"] == 2
    assert contract["fourth_state_computed"] is False


def test_queue_passes_state3_to_fresh_official_process() -> None:
    config = SidecarConfig(
        output_root="/tmp/non_running_state3_contract",
        python_executable="/official/bin/python",
        runner_script="/repo/scripts/run_official_resplat_sidecar.py",
        resplat_repo="/official/resplat",
        checkpoint="/weights/small8.pth",
        expected_checkpoint_sha256="0" * 64,
        cuda_visible_devices="1",
        refinement_updates=3,
    )
    queue = OfficialReSplatSidecarQueue(config)
    command = queue._command(Path("/snapshot"), Path("/output"))
    position = command.index("--num-refine")
    assert command[position + 1] == "3"
    try:
        SidecarConfig(refinement_updates=0)
    except ValueError as error:
        assert "between 1 and 4" in str(error)
    else:
        raise AssertionError("zero recurrent updates passed the state contract")


def test_covariance_authoritative_world_conversion() -> None:
    native = _local_state()
    pivot = _pivot()
    arrays, metadata = build_unblur_world_arrays(
        means_local=native["means"],
        covariances_local=native["covariances"],
        harmonics_local=native["harmonics"],
        opacities=native["opacities"],
        pivot_c2w=pivot,
        owner_frame_ids=range(8),
        owner_sequence_ordinals=range(10, 18),
    )
    rotation, translation = pivot[:3, :3], pivot[:3, 3]
    assert np.allclose(
        arrays["means_world"], native["means"] @ rotation.T + translation, atol=1e-6
    )
    expected_covariance = np.einsum(
        "ij,njk,lk->nil", rotation, native["covariances"], rotation
    )
    assert np.allclose(arrays["covariances_world"], expected_covariance, atol=1e-6)
    assert metadata["native_scale_rotation_copied"] is False
    assert metadata["covariance_source"] == "official_refined_gaussians.covariances"
    assert arrays["owner_frame_ids"].tolist() == [i for i in range(8) for _ in range(2)]
    assert arrays["unblur_features_dc"].shape == (16, 1, 3)
    assert arrays["harmonics_world"].shape == (16, 3, 1)
    assert np.array_equal(arrays["harmonics_world"], native["harmonics"][:, :, :1])
    assert arrays["unblur_features_rest"].shape == (16, 0, 3)
    assert metadata["official_no_rotate_sh"] is True
    assert metadata["source_harmonic_dimension"] == 16
    assert metadata["imported_harmonic_dimension"] == 1
    assert metadata["dropped_higher_order_harmonics"] == 15


def test_native_six_array_conversion_contract_and_legacy_five_array_compatibility() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        native = _local_state()
        six = root / "native6.npz"
        np.savez_compressed(six, **native)
        values = tuple(native.values())
        total = sum(int(value.size) for value in values)
        finite = sum(int(np.isfinite(value).sum()) for value in values)
        distances = np.linalg.norm(native["means"].astype(np.float64), axis=-1)
        scales = native["scales"].astype(np.float64)
        quaternion_norms = np.linalg.norm(
            native["rotations"].astype(np.float64), axis=-1
        )
        geometry = {
            "gaussian_count": int(native["means"].shape[0]),
            "finite_fraction": float(finite / total),
            "p95_distance_from_local_origin": float(np.quantile(distances, 0.95)),
            "max_distance_from_local_origin": float(np.max(distances)),
            "p95_scale": float(np.quantile(scales, 0.95)),
            "max_scale": float(np.max(scales)),
            "max_quaternion_norm_deviation": float(
                np.max(np.abs(quaternion_norms - 1.0))
            ),
        }
        result = {
            "native_to_unblur_conversion_performed": True,
            "geometry": geometry,
            "outputs": {
                "native_gaussians_npz": six.name,
                "native_gaussians_npz_sha256": sha256_file(six),
                "npz_arrays": array_manifest(native),
            },
        }
        accepted = verify_result_artifacts(result, root)
        assert accepted.accepted, accepted.reasons

        five_arrays = {name: value for name, value in native.items() if name != "covariances"}
        five = root / "native5.npz"
        np.savez_compressed(five, **five_arrays)
        legacy = json.loads(json.dumps(result))
        legacy["native_to_unblur_conversion_performed"] = False
        legacy["outputs"]["native_gaussians_npz"] = five.name
        legacy["outputs"]["native_gaussians_npz_sha256"] = sha256_file(five)
        legacy["outputs"]["npz_arrays"] = array_manifest(five_arrays)
        legacy_gate = verify_result_artifacts(legacy, root)
        assert legacy_gate.accepted, legacy_gate.reasons

        missing_covariance = json.loads(json.dumps(legacy))
        missing_covariance["native_to_unblur_conversion_performed"] = True
        rejected = verify_result_artifacts(missing_covariance, root)
        assert not rejected.accepted
        assert "native_gaussian_array_contract_incomplete" in rejected.reasons
        assert "native_gaussian_npz_arrays_incomplete" in rejected.reasons


def _snapshot(root: Path) -> tuple[Path, dict]:
    pivot = _pivot()
    frames = []
    for index in range(8):
        pose = pivot.copy()
        pose[0, 3] += index * 0.01
        frames.append(
            SidecarFrameInput(
                frame_id=index,
                sequence_ordinal=10 + index,
                c2w=pose,
                intrinsics_px=((10.0, 0.0, 8.0), (0.0, 11.0, 6.0), (0.0, 0.0, 1.0)),
                image=np.full((12, 16, 3), index * 10, dtype=np.uint8),
            )
        )
    path = materialize_closed_submap_snapshot(
        snapshots_root=root,
        submap_id=0,
        record_keyframe_ids=range(8),
        frames=frames,
        closure_sequence_ordinal=17,
        pose_revision=23,
    )
    return path, load_snapshot(path)


def test_world_artifact_gate_binds_pivot_owners_and_raw_layout() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _, snapshot = _snapshot(root / "snapshots")
        native = _local_state()
        middle = snapshot["frames"][4]
        arrays, metadata = build_unblur_world_arrays(
            means_local=native["means"],
            covariances_local=native["covariances"],
            harmonics_local=native["harmonics"],
            opacities=native["opacities"],
            pivot_c2w=middle["c2w_opencv"],
            owner_frame_ids=range(8),
            owner_sequence_ordinals=range(10, 18),
        )
        metadata.update(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "source_pose_revision": 23,
                "source_pose_hashes": [f["pose_hash"] for f in snapshot["frames"]],
                "pivot_context_index_zero_based": 4,
                "pivot_frame_id": 4,
                "pivot_sequence_ordinal": 14,
                "pivot_pose_hash": middle["pose_hash"],
                "owner_frame_ids": list(range(8)),
                "owner_sequence_ordinals": list(range(10, 18)),
                "refinement_state": 3,
            }
        )
        native_path = root / "native_gaussians_local.npz"
        np.savez_compressed(
            native_path,
            means=native["means"],
            covariances=native["covariances"],
            scales=native["scales"],
            rotations=native["rotations"],
            harmonics=native["harmonics"],
            opacities=native["opacities"],
        )
        world_path = root / "unblur_gaussians_snapshot_world.npz"
        np.savez_compressed(world_path, **arrays)
        result = {
            "official_resplat": {"num_refine": 3},
            "native_to_unblur_conversion_performed": True,
            "unblur_world_artifact": metadata,
            "outputs": {
                "native_gaussians_npz": native_path.name,
                "native_gaussians_npz_sha256": sha256_file(native_path),
                "unblur_world_gaussians_npz": world_path.name,
                "unblur_world_gaussians_npz_sha256": sha256_file(world_path),
                "unblur_world_npz_arrays": array_manifest(arrays),
            },
        }
        gate = verify_unblur_world_artifact(result, root, snapshot)
        assert gate.accepted, gate.reasons
        assert gate.measurements["max_mean_transform_error"] < 1e-5
        assert gate.measurements["max_covariance_factorization_error"] < 1e-5

        # The duplicated local covariance in the converted artifact must be a
        # bit-exact copy of the official selected-state native array.
        tampered_arrays = dict(arrays)
        tampered_arrays["source_covariances_local"] = arrays[
            "source_covariances_local"
        ].copy()
        tampered_arrays["source_covariances_local"][0, 0, 0] += np.float32(1e-4)
        tampered_world = root / "tampered_world.npz"
        np.savez_compressed(tampered_world, **tampered_arrays)
        covariance_tampered = json.loads(json.dumps(result))
        covariance_tampered["outputs"]["unblur_world_gaussians_npz"] = (
            tampered_world.name
        )
        covariance_tampered["outputs"]["unblur_world_gaussians_npz_sha256"] = (
            sha256_file(tampered_world)
        )
        covariance_tampered["outputs"]["unblur_world_npz_arrays"] = array_manifest(
            tampered_arrays
        )
        rejected_covariance = verify_unblur_world_artifact(
            covariance_tampered, root, snapshot
        )
        assert not rejected_covariance.accepted
        assert (
            "world_source_covariances_not_exact_native_copy"
            in rejected_covariance.reasons
        )

        legacy_native = root / "legacy_native_five_arrays.npz"
        np.savez_compressed(
            legacy_native,
            means=native["means"],
            scales=native["scales"],
            rotations=native["rotations"],
            harmonics=native["harmonics"],
            opacities=native["opacities"],
        )
        missing_native_covariance = json.loads(json.dumps(result))
        missing_native_covariance["outputs"]["native_gaussians_npz"] = (
            legacy_native.name
        )
        missing_native_covariance["outputs"]["native_gaussians_npz_sha256"] = (
            sha256_file(legacy_native)
        )
        rejected_missing = verify_unblur_world_artifact(
            missing_native_covariance, root, snapshot
        )
        assert not rejected_missing.accepted
        assert (
            "conversion_native_gaussian_six_array_contract_failed"
            in rejected_missing.reasons
        )
        assert "native_covariances_missing_for_world_audit" in rejected_missing.reasons

        tampered = json.loads(json.dumps(result))
        tampered["unblur_world_artifact"]["pivot_pose_hash"] = "0" * 64
        rejected = verify_unblur_world_artifact(tampered, root, snapshot)
        assert not rejected.accepted
        assert "world_artifact_middle_pivot_binding_mismatch" in rejected.reasons


def test_real_71680_state4_float32_psd_regression() -> None:
    """Exercise the real official payload that exposed negative roundoff eigs."""

    if not REAL_STATE4_ROOT.is_dir():
        return
    from scipy.spatial.transform import Rotation

    published = next((REAL_STATE4_ROOT / "published").iterdir())
    snapshot_root = REAL_STATE4_ROOT / "snapshots" / published.name
    snapshot = json.loads((snapshot_root / "snapshot_manifest.json").read_text())
    frames = snapshot["frames"]
    with np.load(published / "native_gaussians_local.npz") as archive:
        native = {name: np.asarray(archive[name]) for name in archive.files}
    count = int(native["means"].shape[0])
    per_view = count // 8
    local_rotation = Rotation.from_quat(
        native["rotations"].astype(np.float64)
    ).as_matrix().astype(np.float32)
    scales = native["scales"].astype(np.float32)
    base_covariance = np.einsum(
        "nij,nj,nkj->nik",
        local_rotation,
        scales * scales,
        local_rotation,
        optimize=True,
    ).astype(np.float32)
    pivot = np.asarray(frames[4]["c2w_opencv"], dtype=np.float32)
    normalized_rotations = []
    for frame in frames:
        c2w = np.asarray(frame["c2w_opencv"], dtype=np.float32)
        normalized_rotations.append(
            (np.linalg.inv(pivot).astype(np.float32) @ c2w)[:3, :3]
        )
    owner_rotation = np.concatenate(
        [
            np.broadcast_to(normalized_rotations[index], (per_view, 3, 3))
            for index in range(8)
        ]
    ).astype(np.float32)
    covariance_local = (
        owner_rotation @ base_covariance @ owner_rotation.transpose(0, 2, 1)
    ).astype(np.float32)
    raw_eigenvalues = np.linalg.eigvalsh(covariance_local.astype(np.float64))
    assert int(np.count_nonzero(raw_eigenvalues[:, 0] <= 0.0)) > 20_000

    arrays, metadata = build_unblur_world_arrays(
        means_local=native["means"],
        covariances_local=covariance_local,
        harmonics_local=native["harmonics"],
        opacities=native["opacities"],
        pivot_c2w=frames[4]["c2w_opencv"],
        owner_frame_ids=[int(frame["frame_id"]) for frame in frames],
        owner_sequence_ordinals=[int(frame["sequence_ordinal"]) for frame in frames],
    )
    factorization = metadata["factorization"]
    assert factorization["clamped_gaussian_count"] > 20_000
    assert factorization["significant_negative_gaussian_count"] == 0
    assert factorization["max_psd_correction"] < 1e-6
    assert arrays["harmonics_world"].shape == (71_680, 3, 1)
    assert arrays["unblur_features_rest"].shape == (71_680, 0, 3)
    assert all(np.isfinite(value).all() for value in arrays.values())


def main() -> None:
    test_exact_state3_does_not_compute_state4()
    test_queue_passes_state3_to_fresh_official_process()
    test_covariance_authoritative_world_conversion()
    test_native_six_array_conversion_contract_and_legacy_five_array_compatibility()
    test_world_artifact_gate_binds_pivot_owners_and_raw_layout()
    test_real_71680_state4_float32_psd_regression()
    print("ReSplat state3/world bridge CPU contracts passed")


if __name__ == "__main__":
    main()
