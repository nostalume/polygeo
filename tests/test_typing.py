from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import runpy
import shutil
import subprocess


_FIXTURES = Path(__file__).with_name("typing")
_EXPECTED = re.compile(r"^# ty-expect: (?P<rules>[-a-z0-9, ]+)$", re.MULTILINE)
_DIAGNOSTIC = re.compile(
    r"(?P<path>[^\r\n:]+\.py):\d+:\d+: error\[(?P<rule>[-a-z0-9]+)\]"
)


def _expectations(path: Path) -> Counter[str]:
    match = _EXPECTED.search(path.read_text(encoding="utf-8"))
    assert match is not None, f"{path} has no ty-expect header"
    return Counter(rule.strip() for rule in match.group("rules").split(","))


def test_diagnostic_parser_accepts_absolute_windows_paths() -> None:
    diagnostic = (
        r"C:\work\polygeo\tests\typing\case_invalid.py:12:3: "
        "error[invalid-argument-type] rejected"
    )
    match = _DIAGNOSTIC.search(diagnostic)
    assert match is not None
    assert Path(match.group("path")).name == "case_invalid.py"
    assert match.group("rule") == "invalid-argument-type"


def test_positive_type_fixtures_execute() -> None:
    fixtures = sorted(
        path for path in _FIXTURES.glob("*.py") if not path.stem.endswith("_invalid")
    )
    assert fixtures
    for fixture in fixtures:
        runpy.run_path(str(fixture), run_name=f"__typing_{fixture.stem}__")


def test_negative_type_contracts_are_enforced() -> None:
    fixtures = sorted(_FIXTURES.glob("*_invalid.py"))
    assert fixtures
    ty = shutil.which("ty")
    assert ty is not None, "Ty is required by the project test environment"

    result = subprocess.run(
        [
            ty,
            "check",
            "--error-on-warning",
            "--no-force-exclude",
            "--output-format",
            "concise",
            *(str(path) for path in fixtures),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    output = result.stdout + result.stderr
    actual: dict[str, Counter[str]] = {path.name: Counter() for path in fixtures}
    for match in _DIAGNOSTIC.finditer(output):
        actual[Path(match.group("path")).name][match.group("rule")] += 1

    assert result.returncode == 1, output
    assert all("unresolved-import" not in rules for rules in actual.values()), output
    for fixture in fixtures:
        assert actual[fixture.name] == _expectations(fixture), output


def test_typing_fixtures_are_flat_python_sources() -> None:
    assert _FIXTURES.is_dir()
    assert not [path for path in _FIXTURES.rglob("*") if path.is_dir()]
    assert not [path for path in _FIXTURES.iterdir() if path.suffix != ".py"]
