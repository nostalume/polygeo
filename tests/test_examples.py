from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


_ROOT = Path(__file__).parents[1]
_NOTEBOOKS = tuple(sorted((_ROOT / "examples").glob("*.py")))


@pytest.mark.slow
@pytest.mark.parametrize("notebook", _NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_exports_cleanly(notebook: Path, tmp_path: Path) -> None:
    marimo = shutil.which("marimo")
    assert marimo is not None
    output = tmp_path / f"{notebook.stem}.html"
    result = subprocess.run(
        [
            marimo,
            "export",
            "html",
            str(notebook),
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
    assert not list(notebook.parent.glob("*.html"))
