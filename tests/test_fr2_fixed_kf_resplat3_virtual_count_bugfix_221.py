#!/usr/bin/env python3
"""CPU-only contracts for the isolated v6 virtual-count bugfix rerun."""

from __future__ import annotations

import copy
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load(
    "test_virtual_count_bugfix_runner",
    ROOT / "scripts/run_fr2_fixed_kf_resplat3_virtual_count_bugfix_221.py",
)
REPORT = _load(
    "test_virtual_count_bugfix_report",
    ROOT / "scripts/report_fr2_fixed_kf_resplat3_virtual_count_bugfix_221.py",
)


def _raises(error_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_v6_direct_v5_config_and_exact_bugfix() -> None:
    v6, v5 = RUNNER._load_configs()
    differences = RUNNER._validate_config(v6, v5)
    assert set(differences) == RUNNER.ALLOWED_V6_V5_CONFIG_DIFFERENCES
    assert (v6.get("mapping") or {}) == (v5.get("mapping") or {})
    direct = RUNNER._validate_direct_v5_inheritance()
    assert direct["directly_inherits_v5_common"] is True
    bugfix = RUNNER._validate_bugfix()
    assert bugfix["assignment_added"] is True
    assert bugfix["positive_count_guard_added"] is True
    assert bugfix["approved_block_occurrences"] == 1
    assert bugfix["v5_mapper_sha256_after_reversion"] == RUNNER.V5_OLD_MAPPER_SHA256
    assert bugfix["v6_mapper_size_bytes"] - bugfix["v5_mapper_size_bytes"] == len(
        RUNNER.APPROVED_BUGFIX_BLOCK.encode("utf-8")
    )


class _View:
    def __init__(self, uid: int, *, deblur: bool, count: int = 5, array_count: int | None = None):
        self.uid = uid
        self.timestamp = 100 + uid
        self.deblur_fail = deblur
        self.n_virtual_cams = count
        self.exposure_a = torch.tensor(0.0)
        self.exposure_b = torch.tensor(0.0)
        self.original_image = torch.full((3, 4, 4), 0.25)
        actual = count if array_count is None else array_count
        self._virtual = tuple([list(range(actual)) for _ in range(4)])

    def get_virtual_extrinsics(self):
        return self._virtual


def _fake_mapper(views):
    from src.mapper import Mapper

    mapper = object.__new__(Mapper)
    mapper.official_resplat_active_fusion_cfg = SimpleNamespace(
        postmerge_quality_gate={"l1_weight": 0.8, "one_minus_ssim_weight": 0.2}
    )
    mapper.cameras = {view.uid: view for view in views}
    mapper.gaussians = object()
    mapper.pipeline_params = object()
    mapper.background = object()
    return mapper


def test_context_metrics_executes_normal_and_deblur_middle_view() -> None:
    import src.mapper as mapper_module

    views = [_View(0, deblur=False)] + [_View(i, deblur=True) for i in range(1, 8)]
    mapper = _fake_mapper(views)
    calls = {"regular": 0, "virtual": 0}
    old_render = mapper_module.render
    old_virtual = mapper_module.render_virtual
    old_ssim = mapper_module.ssim

    def render(*args, **kwargs):
        calls["regular"] += 1
        return {"render": torch.full((3, 4, 4), 0.25)}

    def render_virtual(*args, R, t, theta, rho, **kwargs):
        calls["virtual"] += 1
        assert (R, t, theta, rho) == (2, 2, 2, 2)
        return {"render": torch.full((3, 4, 4), 0.25)}

    try:
        mapper_module.render = render
        mapper_module.render_virtual = render_virtual
        mapper_module.ssim = lambda prediction, observation: torch.tensor(1.0)
        result = mapper._active_fusion_context_metrics(tuple(range(8)))
    finally:
        mapper_module.render = old_render
        mapper_module.render_virtual = old_virtual
        mapper_module.ssim = old_ssim
    assert len(result["per_view"]) == 8
    assert calls == {"regular": 1, "virtual": 7}
    assert math.isfinite(result["mean_l1"])
    assert math.isfinite(result["mean_ssim"])
    assert math.isfinite(result["mean_composite"])
    assert result["mean_l1"] == 0.0


def test_context_metrics_rejects_nonpositive_or_mismatched_virtual_count() -> None:
    zero = _fake_mapper([_View(0, deblur=True, count=0)])
    _raises(RuntimeError, zero._active_fusion_context_metrics, (0,))
    mismatch = _fake_mapper([_View(0, deblur=True, count=5, array_count=4)])
    _raises(RuntimeError, mismatch._active_fusion_context_metrics, (0,))


def test_frozen_v5_failure_and_full_preflight() -> None:
    failure = RUNNER._v5_failure_binding()
    assert failure["frozen_tree"]["tree_sha256"] == RUNNER.V5_FAILED_TREE_SHA256
    assert failure["frozen_tree"]["file_count"] == 38
    assert failure["frozen_tree"]["total_bytes"] == 45445346
    assert failure["reason"] == RUNNER.V5_FAILURE_SIGNATURE
    assert failure["premerge_gates_all_accepted"] is True
    assert failure["merge_started"] is False
    assert failure["active_map_changed"] is False
    assert not RUNNER.OUTPUT_ROOT.exists()
    record = RUNNER.preflight(check_output_available=True)
    assert record["scope"]["v5_runtime_artifacts_reused"] is False
    assert record["scientific_contract"]["official_resplat_fresh_rerun"] is True
    assert not RUNNER.OUTPUT_ROOT.exists()


def test_report_missing_output_fails_closed() -> None:
    _raises(REPORT.ContractError, REPORT.build_report, RUNNER.OUTPUT_ROOT)


def test_report_preflight_tampering_and_overwrite_fail_closed() -> None:
    v6, _ = RUNNER._load_configs()
    preflight = RUNNER.preflight(check_output_available=True)
    REPORT._validate_v6_preflight(v6, preflight, RUNNER.OUTPUT)
    for section, field in (
        ("implementation_provenance", "schema"),
        ("execution", "process_visible_device"),
        ("v5_cpu_preflight_reused_as_read_only_validation", "schema"),
    ):
        tampered = copy.deepcopy(preflight)
        tampered[section][field] = "tampered"
        _raises(
            REPORT.ContractError,
            REPORT._validate_v6_preflight,
            v6,
            tampered,
            RUNNER.OUTPUT,
        )
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        report = {"schema": "test"}
        REPORT.write_report(report, output)
        _raises(FileExistsError, REPORT.write_report, report, output)


def main() -> None:
    test_v6_direct_v5_config_and_exact_bugfix()
    test_context_metrics_executes_normal_and_deblur_middle_view()
    test_context_metrics_rejects_nonpositive_or_mismatched_virtual_count()
    test_frozen_v5_failure_and_full_preflight()
    test_report_missing_output_fails_closed()
    test_report_preflight_tampering_and_overwrite_fail_closed()
    print("virtual-count bugfix v6 CPU contracts: PASS")


if __name__ == "__main__":
    main()
