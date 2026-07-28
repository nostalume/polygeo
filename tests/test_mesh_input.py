from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from polygeo import Geometry, MeshError, load_surface


class _Mesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.vertices = vertices
        self.faces = faces


_FACES = np.array([[0, 1, 2]], dtype=np.int64)


def _install_trimesh(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
) -> list[tuple[object, bool]]:
    calls: list[tuple[object, bool]] = []

    def load(source: object, *, process: bool) -> object:
        calls.append((source, process))
        return payload

    monkeypatch.setitem(
        sys.modules,
        "trimesh",
        SimpleNamespace(Trimesh=_Mesh, load=load),
    )
    return calls


def test_root_import_does_not_import_trimesh() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; from pathlib import Path; "
                f"sys.path.insert(0, {str(Path(__file__).parents[1] / 'src')!r}); "
                "import polygeo; "
                "assert not any(name == 'trimesh' or name.startswith('trimesh.') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_load_surface_requires_mesh_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def missing_trimesh(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ):
        if name == "trimesh":
            error = ModuleNotFoundError("backend detail")
            error.name = "trimesh"
            raise error
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "trimesh", raising=False)
    monkeypatch.setattr(builtins, "__import__", missing_trimesh)

    with pytest.raises(
        MeshError,
        match=r"^mesh input requires the optional polygeo\[mesh\] dependency$",
    ):
        load_surface("surface.obj")


def test_load_surface_owns_complete_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    calls = _install_trimesh(monkeypatch, _Mesh(vertices, faces))
    source = tmp_path / "surface.obj"

    geometry = load_surface(source)

    assert isinstance(geometry, Geometry)
    assert calls == [(source, False)]
    np.testing.assert_array_equal(geometry.positions, vertices)
    np.testing.assert_array_equal(geometry.complex.simplices(2), faces)

    vertices[:] = 9.0
    faces[:] = 0
    np.testing.assert_array_equal(
        geometry.positions,
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(
        geometry.complex.simplices(2), np.array([[0, 1, 2]], dtype=np.int64)
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (object(), "exactly one triangular mesh"),
        (
            _Mesh(
                np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
                np.array([[0, 1]], dtype=np.int64),
            ),
            "triangular faces",
        ),
        (
            _Mesh(
                np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, 1.0]], dtype=np.float64),
                np.array([[0, 1, 2]], dtype=np.int64),
            ),
            "finite vertex positions",
        ),
    ],
)
def test_load_surface_rejects_invalid_payloads(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    message: str,
) -> None:
    _install_trimesh(monkeypatch, payload)
    with pytest.raises(MeshError, match=message):
        load_surface("bad.mesh")


def test_load_surface_hides_backend_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        Trimesh = _Mesh

        @staticmethod
        def load(source: object, *, process: bool) -> object:
            raise RuntimeError("secret backend wording")

    monkeypatch.setitem(sys.modules, "trimesh", Broken)
    with pytest.raises(MeshError, match=r"^failed to load surface mesh$") as captured:
        load_surface("bad.mesh")
    assert "secret backend wording" not in str(captured.value)


def test_load_surface_preserves_trailing_unreferenced_vertices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [9.0, 9.0, 9.0],
        ],
        dtype=np.float64,
    )
    _install_trimesh(monkeypatch, _Mesh(vertices, _FACES.copy()))
    geometry = load_surface(tmp_path / "unused.obj")
    assert geometry.complex.vertex_count == 4
    np.testing.assert_array_equal(geometry.positions, vertices)


def test_load_surface_closes_malformed_vertex_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenMesh(_Mesh):
        @property
        def vertices(self):
            raise TypeError("backend detail")

    broken = object.__new__(BrokenMesh)
    broken.faces = _FACES.copy()
    _install_trimesh(monkeypatch, broken)
    with pytest.raises(
        MeshError, match=r"^surface mesh is not admissible$"
    ) as captured:
        load_surface("broken.obj")
    assert "backend detail" not in str(captured.value)
