from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


_ROOT = Path(__file__).parents[1]
_EXAMPLES = _ROOT / "examples"
_NOTEBOOKS = (
    "curvature.py",
    "poisson.py",
    "harmonic_extension.py",
    "mean_curvature_flow.py",
    "homology_and_periods.py",
    "hodge_decomposition.py",
    "connection_and_holonomy.py",
)
_REQUIRED_SECTIONS = (
    "Mathematical question",
    "From mathematics to PolyGeo",
    "Computation",
    "Visualization",
    "Evaluation",
    "Interpretation",
)
_REQUIRED_ACCEPTANCE = {
    "curvature.py": ("min < 0 < max", "rigid/scale invariance"),
    "poisson.py": ("K=M\\Delta", "physical residual"),
    "harmonic_extension.py": ("annulus boundary", "all-boundary backend calls"),
    "mean_curvature_flow.py": (
        "independently recomputed frozen energy",
        "source/target exact identity laws",
    ),
    "homology_and_periods.py": (
        "wrapped coordinate one-forms",
        "real primal cycles, not integral dual",
    ),
    "hodge_decomposition.py": (
        "all three components are nonzero",
        "pairwise weighted orthogonality",
        "no-backend endpoint law",
    ),
    "connection_and_holonomy.py": (
        "local contractible holonomy",
        "global generator obstruction",
    ),
}


def test_expected_notebooks_have_the_learning_and_layout_contract() -> None:
    assert not (_EXAMPLES / "conformal_parameterization.py").exists()
    for name in _NOTEBOOKS:
        source = (_EXAMPLES / name).read_text(encoding="utf-8")
        assert "mo.hstack" not in source
        assert "make_subplots" not in source
        assert "width=" not in source
        for section in _REQUIRED_SECTIONS:
            assert section in source, f"{name} is missing {section}"
        for acceptance in _REQUIRED_ACCEPTANCE[name]:
            assert acceptance in source, f"{name} is missing {acceptance}"


def test_connection_notebook_is_registered_in_docs_and_ci() -> None:
    notebook = "examples/connection_and_holonomy.py"
    assert notebook in (_ROOT / ".github/workflows/verify.yml").read_text(
        encoding="utf-8"
    )
    assert notebook in (_EXAMPLES / "README.md").read_text(encoding="utf-8")
    assert (
        "connection and holonomy"
        in (_ROOT / "README.md").read_text(encoding="utf-8").lower()
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", _NOTEBOOKS)
def test_notebook_exports_cleanly(name: str, tmp_path: Path) -> None:
    marimo = shutil.which("marimo")
    assert marimo is not None
    output = tmp_path / f"{Path(name).stem}.html"
    result = subprocess.run(
        [
            marimo,
            "export",
            "html",
            str(_EXAMPLES / name),
            "--no-include-code",
            "-o",
            str(output),
        ],
        cwd=_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.stat().st_size > 1_000
    assert not list(_EXAMPLES.glob("*.html"))
