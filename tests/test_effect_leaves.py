from __future__ import annotations

import importlib
import gc
import sys
from types import SimpleNamespace

import numpy as np

import polygeo


def test_mesh_effect_has_one_public_leaf_and_root_reexports() -> None:
    mesh = importlib.import_module("polygeo.mesh")
    assert mesh.load_surface is polygeo.load_surface
    assert mesh.MeshError is polygeo.MeshError
    assert mesh.__all__ == ["MeshError", "load_surface"]
    assert "trimesh" not in sys.modules


def test_mesh_loader_is_lazy_and_returns_admitted_realization(monkeypatch) -> None:
    assert "trimesh" not in sys.modules

    class Mesh:
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]], dtype=np.int64)

    fake = SimpleNamespace(Trimesh=Mesh, load=lambda *_args, **_kwargs: Mesh())
    monkeypatch.setitem(sys.modules, "trimesh", fake)
    expected_positions = Mesh.vertices.copy()
    geometry = polygeo.load_surface("fixture.obj")
    Mesh.vertices[:] = 9.0
    Mesh.faces[:] = 0
    np.testing.assert_array_equal(geometry.positions_numpy_copy(), expected_positions)
    np.testing.assert_array_equal(geometry.complex.simplices(2), [[0, 1, 2]])


def test_mesh_leaf_rejects_scene_without_exposing_backend_details(monkeypatch) -> None:
    class Mesh:
        pass

    fake = SimpleNamespace(Trimesh=Mesh, load=lambda *_args, **_kwargs: object())
    monkeypatch.setitem(sys.modules, "trimesh", fake)
    with np.testing.assert_raises_regex(
        polygeo.MeshError, "surface mesh is not admissible"
    ):
        polygeo.load_surface("scene.glb")


def test_plotting_consumes_explicit_snapshots() -> None:
    complex_ = polygeo.Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64)
    )
    geometry = polygeo.EuclideanRealization.from_positions(
        complex_, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )
    figure = polygeo.plot_geometry(geometry)
    assert len(figure.data) >= 2


def test_plotting_rejects_projection_axes_outside_the_snapshot() -> None:
    complex_ = polygeo.Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64)
    )
    geometry = polygeo.EuclideanRealization.from_positions(
        complex_,
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
    )
    surface = polygeo.TriangleSurface.admit(geometry)
    vectors = surface.vertex_vectors(np.ones((3, 3), dtype=np.float64))

    with np.testing.assert_raises(polygeo.PlotError):
        polygeo.plot_geometry(geometry, axes=(0, 3))
    with np.testing.assert_raises(polygeo.PlotError):
        polygeo.plot_surface_vectors(vectors, axes=(0, 3))


def test_plot_form_accepts_only_full_cochains_from_the_geometry() -> None:
    complex_ = polygeo.Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64)
    )
    geometry = polygeo.Geometry.from_positions(
        complex_, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )
    form = complex_.binary64_cochain_space(1).admit_numpy(np.array([1.0, -2.0, 3.0]))
    figure = polygeo.plot_form(geometry, form)
    assert tuple(figure.data[-1].customdata[:, 0]) == (0, 1, 2)

    selected = complex_.binary64_cochain_space(
        1, indices=np.array([0, 2], dtype=np.int64)
    ).admit_numpy(np.array([1.0, 3.0]))
    foreign = (
        polygeo.Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
        .binary64_cochain_space(1)
        .admit_numpy(np.ones(3))
    )
    with np.testing.assert_raises(polygeo.PlotError):
        polygeo.plot_form(geometry, selected)
    with np.testing.assert_raises(polygeo.PlotError):
        polygeo.plot_form(geometry, foreign)


def test_plot_homology_cycle_realizes_one_direct_group_selection() -> None:
    complex_ = polygeo.Complex.from_maximal_simplices(
        np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    )
    geometry = polygeo.Geometry.from_positions(
        complex_, np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    )
    group = polygeo.prepare_integral_homology(complex_.chain_complex(), [1])[1]
    figure = polygeo.plot_homology_cycle(geometry, group, 0)
    assert len(figure.data[-1].customdata) == 4

    foreign = polygeo.Complex.from_maximal_simplices(
        np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    )
    foreign_group = polygeo.prepare_integral_homology(foreign.chain_complex(), [1])[1]
    with np.testing.assert_raises(polygeo.PlotError):
        polygeo.plot_homology_cycle(geometry, foreign_group, 0)


def test_plotting_rejects_intrinsic_dimension_above_two() -> None:
    complex_ = polygeo.Complex.from_maximal_simplices(
        np.array([[0, 1, 2, 3]], dtype=np.int64)
    )
    geometry = polygeo.Geometry.from_positions(complex_, np.eye(4))
    with np.testing.assert_raises(polygeo.PlotError):
        polygeo.plot_geometry(geometry, axes=(0, 1, 2))


def test_plot_snapshot_survives_math_owners_and_scales_to_a_grid() -> None:
    side = 32
    grid = np.arange((side + 1) ** 2, dtype=np.int64).reshape(side + 1, side + 1)
    lower = np.column_stack(
        (grid[:-1, :-1].ravel(), grid[:-1, 1:].ravel(), grid[1:, 1:].ravel())
    )
    upper = np.column_stack(
        (grid[:-1, :-1].ravel(), grid[1:, 1:].ravel(), grid[1:, :-1].ravel())
    )
    faces = np.vstack((lower, upper))
    horizontal, vertical = np.meshgrid(
        np.arange(side + 1, dtype=np.float64),
        np.arange(side + 1, dtype=np.float64),
    )
    coordinates = np.column_stack((horizontal.ravel(), vertical.ravel()))
    complex_ = polygeo.Complex.from_maximal_simplices(faces)
    geometry = polygeo.Geometry.from_positions(complex_, coordinates)
    figure = polygeo.plot_geometry(geometry)
    del geometry, complex_, coordinates, faces
    gc.collect()

    assert len(figure.data[-1].x) == (side + 1) ** 2
    assert figure.to_json()
