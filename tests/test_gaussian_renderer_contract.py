#!/usr/bin/env python3
"""CPU/static contract for the installed Gaussian rasterizer settings API."""

import ast
import inspect
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_installed_settings_require_explicit_disabled_antialiasing() -> None:
    from diff_gaussian_rasterization import GaussianRasterizationSettings

    signature = inspect.signature(GaussianRasterizationSettings)
    assert "antialiasing" in signature.parameters
    settings = GaussianRasterizationSettings(
        image_height=8,
        image_width=10,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4),
        projmatrix=torch.eye(4),
        projmatrix_raw=torch.eye(4),
        sh_degree=0,
        campos=torch.zeros(3),
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    assert settings.antialiasing is False


def test_every_shared_renderer_constructor_disables_antialiasing() -> None:
    renderer = (
        ROOT
        / "thirdparty"
        / "gaussian_splatting"
        / "gaussian_renderer"
        / "__init__.py"
    )
    tree = ast.parse(renderer.read_text(encoding="utf-8"), filename=str(renderer))
    constructors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "GaussianRasterizationSettings"
    ]
    assert len(constructors) == 2, "render and render_virtual must both be audited"
    for constructor in constructors:
        keywords = {keyword.arg: keyword.value for keyword in constructor.keywords}
        assert "antialiasing" in keywords
        value = keywords["antialiasing"]
        assert isinstance(value, ast.Constant) and value.value is False


def main() -> None:
    test_installed_settings_require_explicit_disabled_antialiasing()
    test_every_shared_renderer_constructor_disables_antialiasing()
    print("gaussian_renderer_contract=PASS")


if __name__ == "__main__":
    main()
