from __future__ import annotations

import importlib
import gc
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial import Delaunay

import polygeo


def test_mesh_effect_has_one_public_leaf_and_root_reexports() -> None:
    mesh = importlib.import_module("polygeo.mesh")
    assert mesh.load_surface is polygeo.mesh.load_surface
    assert mesh.MeshError is polygeo.mesh.MeshError
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
    geometry = polygeo.mesh.load_surface("fixture.obj")
    Mesh.vertices[:] = 9.0
    Mesh.faces[:] = 0
    np.testing.assert_array_equal(geometry.positions_numpy_copy(), expected_positions)
    np.testing.assert_array_equal(
        geometry.topology.simplices_numpy_copy(2), [[0, 1, 2]]
    )


def test_mesh_leaf_rejects_scene_without_exposing_backend_details(monkeypatch) -> None:
    class Mesh:
        pass

    fake = SimpleNamespace(Trimesh=Mesh, load=lambda *_args, **_kwargs: object())
    monkeypatch.setitem(sys.modules, "trimesh", fake)
    with np.testing.assert_raises_regex(
        polygeo.mesh.MeshError, "surface mesh is not admissible"
    ):
        polygeo.mesh.load_surface("scene.glb")


def test_plotting_consumes_explicit_snapshots() -> None:
    complex_ = polygeo.topology.Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64)
    )
    geometry = polygeo.geometry.Geometry.from_positions(
        complex_, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )
    figure = polygeo.plot.geometry(geometry)
    assert len(figure.data) >= 2
    edges, vertices = figure.data[-2:]
    assert (edges.line.color, edges.line.width) == ("#4b5563", 1.5)
    assert (vertices.marker.color, vertices.marker.size) == ("#6b7280", 3)
    assert figure.layout.legend.orientation is None
    assert figure.layout.margin.b is None


def test_plotting_rejects_projection_axes_outside_the_snapshot() -> None:
    complex_ = polygeo.topology.Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64)
    )
    geometry = polygeo.geometry.Geometry.from_positions(
        complex_,
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
    )
    surface = polygeo.geometry.TriangleSurface.admit(geometry)
    vectors = surface.vertex_field(np.ones((3, 3), dtype=np.float64))
    vector_figure = polygeo.plot.vectors(vectors, axes=(0, 1))
    vector_trace, head_trace = vector_figure.data[-2:]
    assert vector_trace.name == "vectors"
    assert head_trace.name == "vector heads"
    assert head_trace.marker.symbol == "triangle-up"
    assert vector_trace.legendgroup == head_trace.legendgroup == "vectors"
    assert vector_trace.showlegend is True
    assert head_trace.showlegend is False
    assert vector_figure.layout.legend.orientation is None
    assert vector_figure.layout.margin.b is None
    vector_3d_figure = polygeo.plot.vectors(vectors)
    vector_3d_trace, head_3d_trace = vector_3d_figure.data[-2:]
    assert head_3d_trace.type == "cone"
    assert head_3d_trace.name == "vector heads"
    assert vector_3d_trace.legendgroup == head_3d_trace.legendgroup == "vectors"
    assert vector_3d_trace.showlegend is True
    assert head_3d_trace.showlegend is False

    with np.testing.assert_raises(polygeo.plot.PlotError):
        polygeo.plot.geometry(geometry, axes=(0, 3))
    with np.testing.assert_raises(polygeo.plot.PlotError):
        polygeo.plot.vectors(vectors, axes=(0, 3))


def test_plot_form_accepts_only_full_cochains_from_the_geometry() -> None:
    complex_ = polygeo.topology.Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64)
    )
    geometry = polygeo.geometry.Geometry.from_positions(
        complex_, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )
    form = complex_.binary64_cochain_space(1).admit_numpy(np.array([1.0, -2.0, 3.0]))
    figure = polygeo.plot.form(geometry, form)
    assert tuple(figure.data[-1].customdata[:, 0]) == (0, 1, 2)
    assert figure.data[-1].marker.size == 5
    geometry_3d = polygeo.geometry.Geometry.from_positions(
        complex_, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )
    figure_3d = polygeo.plot.form(geometry_3d, form)
    for coefficient_figure in (figure, figure_3d):
        coefficient_trace = coefficient_figure.data[-1]
        legend = coefficient_figure.layout.legend
        assert legend.orientation == "h"
        assert legend.x == 0.5
        assert legend.xanchor == "center"
        assert legend.y < 0.0
        assert legend.yanchor == "top"
        assert coefficient_figure.layout.margin.b >= 80
        assert coefficient_trace.marker.cmin == -coefficient_trace.marker.cmax
        assert coefficient_trace.marker.colorbar.orientation == "v"
        assert coefficient_trace.marker.colorbar.x > 1.0
        assert coefficient_trace.marker.colorbar.xanchor == "left"
        assert coefficient_trace.marker.colorbar.title.text == "form"

    selected = complex_.binary64_cochain_space(
        1, indices=np.array([0, 2], dtype=np.int64)
    ).admit_numpy(np.array([1.0, 3.0]))
    foreign = (
        polygeo.topology.Complex.from_maximal_simplices(
            np.array([[0, 1, 2]], dtype=np.int64)
        )
        .binary64_cochain_space(1)
        .admit_numpy(np.ones(3))
    )
    with np.testing.assert_raises(polygeo.plot.PlotError):
        polygeo.plot.form(geometry, selected)
    with np.testing.assert_raises(polygeo.plot.PlotError):
        polygeo.plot.form(geometry, foreign)


def test_plot_direction_field_preserves_order_four_symmetry() -> None:
    rings = np.repeat(np.arange(1, 4), 16)
    sections = np.tile(np.arange(16), 3)
    radii = rings / 3
    angles = 2.0 * np.pi * (sections + 0.173 * rings) / 16
    positions = np.vstack(
        (
            [[0.0, 0.0]],
            np.column_stack(
                (1.4 * radii * np.cos(angles), 0.8 * radii * np.sin(angles))
            ),
        )
    )
    domain = polygeo.topology.Complex.from_maximal_simplices(
        np.asarray(Delaunay(positions).simplices, dtype=np.int64)
    )
    geometry = polygeo.geometry.Geometry.from_positions(
        domain,
        np.column_stack((positions, np.zeros(len(positions), dtype=np.float64))),
    )
    surface = polygeo.geometry.TriangleSurface.admit(geometry)
    field = surface.boundary_direction(4, geometry.metric(), 0.0)

    figure = polygeo.plot.direction(field, scale=0.15)
    branches = tuple(trace for trace in figure.data if trace.name.startswith("branch "))
    assert tuple(trace.name for trace in branches) == (
        "branch 1/4",
        "branch 2/4",
        "branch 3/4",
        "branch 4/4",
    )
    assert all(trace.line.color == "#2563eb" for trace in branches)


def test_plot_homology_cycle_realizes_one_direct_group_selection() -> None:
    complex_ = polygeo.topology.Complex.from_maximal_simplices(
        np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    )
    geometry = polygeo.geometry.Geometry.from_positions(
        complex_, np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    )
    group = polygeo.chain.analyze_integral_homology(complex_.chain_complex(), [1])[1]
    figure = polygeo.plot.homology_cycle(geometry, group, 0)
    assert len(figure.data[-1].customdata) == 4
    assert figure.layout.legend.orientation == "h"
    assert figure.layout.legend.y < 0.0
    assert figure.layout.margin.b >= 80
    assert figure.data[-1].marker.colorbar.orientation == "v"
    assert figure.data[-1].marker.colorbar.x > 1.0
    assert figure.data[-1].marker.colorbar.title.text == "homology cycle"

    foreign = polygeo.topology.Complex.from_maximal_simplices(
        np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    )
    foreign_group = polygeo.chain.analyze_integral_homology(
        foreign.chain_complex(), [1]
    )[1]
    with np.testing.assert_raises(polygeo.plot.PlotError):
        polygeo.plot.homology_cycle(geometry, foreign_group, 0)


def test_plotting_rejects_intrinsic_dimension_above_two() -> None:
    complex_ = polygeo.topology.Complex.from_maximal_simplices(
        np.array([[0, 1, 2, 3]], dtype=np.int64)
    )
    geometry = polygeo.geometry.Geometry.from_positions(complex_, np.eye(4))
    with np.testing.assert_raises(polygeo.plot.PlotError):
        polygeo.plot.geometry(geometry, axes=(0, 1, 2))


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
    complex_ = polygeo.topology.Complex.from_maximal_simplices(faces)
    geometry = polygeo.geometry.Geometry.from_positions(complex_, coordinates)
    figure = polygeo.plot.geometry(geometry)
    del geometry, complex_, coordinates, faces
    gc.collect()

    assert len(figure.data[-1].x) == (side + 1) ** 2
    assert figure.to_json()
