from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("fixture_name", "diagnostic_count"),
    [
        ("typecheck_simplicial_invalid.txt", 6),
        ("typecheck_geometry_invalid.txt", 3),
        ("typecheck_operators_invalid.txt", 8),
        ("typecheck_subspaces_invalid.txt", 8),
    ],
)
def test_negative_type_contracts_are_enforced(
    fixture_name: str,
    diagnostic_count: int,
    tmp_path: Path,
) -> None:
    ty = shutil.which("ty")
    if ty is None:
        raise AssertionError("Ty is required by the project test environment")

    source = Path(__file__).with_name(fixture_name)
    fixture = tmp_path / f"{source.stem}.py"
    fixture.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    result = subprocess.run(
        [ty, "check", "--error-on-warning", str(fixture)],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 1, output
    assert output.count("error[") == diagnostic_count, output
