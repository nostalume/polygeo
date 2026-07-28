from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

import numpy as np
import pytest

from polygeo import (
    ORDINARY_FORM,
    CochainSpace,
    CochainSubspace,
    Complex,
    FaceVectors,
    Geometry,
    VertexVectors,
    real_homology_basis,
)
from polygeo.plotting import (
    PlotError,
    plot_cochain,
    plot_geometry,
    plot_homology_cycle,
    plot_surface_vectors,
)


def _triangle() -> Geometry:
    complex_ = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
    return Geometry.from_positions(
        complex_,
        np.array([[0.0, 0.0], [1.0, 0.0], [0.2, 0.8]], dtype=np.float64),
    )


def _cycle() -> Geometry:
    complex_ = Complex.from_maximal_simplices(
        np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    )
    return Geometry.from_positions(
        complex_,
        np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        ),
    )


def _tetrahedron_surface() -> Geometry:
    complex_ = Complex.from_maximal_simplices(
        np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    )
    return Geometry.from_positions(
        complex_,
        np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )


def test_root_import_does_not_import_plotly() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; from pathlib import Path; "
                f"sys.path.insert(0, {str(Path(__file__).parents[1] / 'src')!r}); "
                "import polygeo; "
                "assert not any(name == 'plotly' or name.startswith('plotly.') "
                "for name in sys.modules)"
            ),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_geometry_plot_is_responsive_serializable_and_owned() -> None:
    geometry = _triangle()
    before = geometry.positions
    figure = plot_geometry(geometry, title="Triangle")

    assert figure.layout.autosize
    assert figure.layout.width is None
    assert "Triangle" in figure.layout.title.text
    assert json.loads(figure.to_json())["data"]
    np.testing.assert_array_equal(geometry.positions, before)

    figure.data[0].visible = False
    np.testing.assert_array_equal(geometry.positions, before)


def test_geometry_plot_supports_projection_and_rejects_invalid_domains() -> None:
    triangle = _triangle()
    positions = np.column_stack(
        [triangle.positions, np.ones((triangle.complex.vertex_count, 2))]
    ).astype(np.float64)
    high_ambient = Geometry.from_positions(triangle.complex, positions)
    figure = plot_geometry(high_ambient, axes=(0, 1, 3))
    assert len(figure.data) >= 1

    with pytest.raises(PlotError, match="projection"):
        plot_geometry(high_ambient)
    with pytest.raises(PlotError, match="projection"):
        plot_geometry(high_ambient, axes=(0, 0))
    with pytest.raises(PlotError, match="projection"):
        plot_geometry(high_ambient, axes=(0, 4))

    volume = Complex.from_maximal_simplices(np.array([[0, 1, 2, 3]], dtype=np.int64))
    geometry = Geometry.from_positions(
        volume,
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )
    with pytest.raises(PlotError, match="dimension"):
        plot_geometry(geometry)


@pytest.mark.parametrize("support", ["vertex", "face"])
def test_surface_vector_plot_uses_exact_support_locations(support: str) -> None:
    geometry = _tetrahedron_surface()
    if support == "vertex":
        anchors = geometry.positions
        field = VertexVectors(geometry, np.ones_like(anchors))
    else:
        anchors = geometry.positions[geometry.complex.simplices(2)].mean(axis=1)
        field = FaceVectors(geometry, np.ones_like(anchors))

    figure = plot_surface_vectors(field, scale=0.25, title=f"{support} vectors")
    trace = next(item for item in figure.data if item.name == "surface vectors")
    x = np.asarray(trace.x, dtype=np.float64).reshape(-1, 3)
    np.testing.assert_allclose(x[:, 0], anchors[:, 0])
    np.testing.assert_allclose(x[:, 1], anchors[:, 0] + 0.25)
    heads = next(
        item for item in figure.data if item.name == "surface vector arrowheads"
    )
    assert heads.type == "cone"
    assert heads.anchor == "tip"
    assert heads.sizemode == "absolute"
    assert heads.sizeref == 0.09
    np.testing.assert_allclose(np.asarray(heads.x), anchors[:, 0] + 0.25)
    np.testing.assert_allclose(np.asarray(heads.y), anchors[:, 1] + 0.25)
    np.testing.assert_allclose(np.asarray(heads.z), anchors[:, 2] + 0.25)
    np.testing.assert_allclose(np.asarray(heads.u), 0.25)
    np.testing.assert_allclose(np.asarray(heads.v), 0.25)
    np.testing.assert_allclose(np.asarray(heads.w), 0.25)
    assert figure.layout.width is None


def test_surface_vector_plot_rejects_invalid_scale() -> None:
    geometry = _tetrahedron_surface()
    field = VertexVectors(geometry, np.ones_like(geometry.positions))
    with pytest.raises(PlotError, match="scale"):
        plot_surface_vectors(field, scale=0.0)


def test_surface_vector_plot_encodes_magnitude_as_shaft_length() -> None:
    geometry = _tetrahedron_surface()
    vectors = np.zeros_like(geometry.positions)
    vectors[0] = [2.0, 0.0, 0.0]
    vectors[1] = [0.0, 1.0, 0.0]
    field = VertexVectors(geometry, vectors)
    before = field.vectors
    figure = plot_surface_vectors(field, scale=0.25)

    shafts = next(item for item in figure.data if item.name == "surface vectors")
    points = np.column_stack([shafts.x, shafts.y, shafts.z]).reshape(-1, 3, 3)
    np.testing.assert_allclose(
        np.linalg.norm(points[:, 1] - points[:, 0], axis=1),
        [0.5, 0.25, 0.0, 0.0],
    )
    heads = next(
        item for item in figure.data if item.name == "surface vector arrowheads"
    )
    np.testing.assert_allclose(
        np.column_stack([heads.u, heads.v, heads.w]),
        [[0.5, 0.0, 0.0], [0.0, 0.25, 0.0]],
    )
    assert json.loads(figure.to_json())["data"][-1]["type"] == "cone"

    heads.u = [9.0, 9.0]
    np.testing.assert_array_equal(field.vectors, before)


def test_surface_vector_arrowheads_use_representable_scaled_directions() -> None:
    geometry = _tetrahedron_surface()
    tiny = np.nextafter(0.0, 1.0)
    vectors = np.zeros_like(geometry.positions)
    vectors[0, 0] = tiny
    field = VertexVectors(geometry, vectors)
    before = field.vectors

    figure = plot_surface_vectors(field, scale=1.0)
    heads = next(
        item for item in figure.data if item.name == "surface vector arrowheads"
    )
    np.testing.assert_array_equal(np.asarray(heads.u), [tiny])
    np.testing.assert_array_equal(field.vectors, before)

    underflow = np.zeros_like(vectors)
    underflow[0, 0] = 0.5
    figure = plot_surface_vectors(VertexVectors(geometry, underflow), scale=float(tiny))
    assert all(item.name != "surface vector arrowheads" for item in figure.data)


def test_surface_vector_plot_leaves_two_dimensional_and_zero_fields_headless() -> None:
    triangle = _triangle()
    planar = plot_surface_vectors(
        VertexVectors(triangle, np.ones_like(triangle.positions))
    )
    zero = plot_surface_vectors(
        VertexVectors(_tetrahedron_surface(), np.zeros((4, 3), dtype=np.float64))
    )
    assert all(item.name != "surface vector arrowheads" for item in planar.data)
    assert all(item.name != "surface vector arrowheads" for item in zero.data)


def test_cochain_plot_preserves_exact_simplex_correspondence() -> None:
    geometry = _triangle()
    space = CochainSpace(geometry.complex, 0)
    values = np.array([-2.0, 0.0, 3.0], dtype=np.float64)
    form = space.form(values, ORDINARY_FORM)
    figure = plot_cochain(geometry, form)

    colored = next(trace for trace in figure.data if trace.name == "coefficients")
    assert colored.showlegend is False
    np.testing.assert_array_equal(np.asarray(colored.marker.color), values)
    np.testing.assert_array_equal(form.coefficients(), values)
    assert figure.layout.width is None

    subspace = CochainSubspace(space, np.array([0, 2], dtype=np.int64))
    selected = subspace.form(np.array([4.0, -1.0]), ORDINARY_FORM)
    selected_figure = plot_cochain(geometry, selected)
    selected_trace = next(
        trace for trace in selected_figure.data if trace.name == "coefficients"
    )
    np.testing.assert_array_equal(np.asarray(selected_trace.customdata), [0, 2])


def test_face_cochain_mesh_defers_hover_to_face_centroids() -> None:
    geometry = _tetrahedron_surface()
    space = CochainSpace(geometry.complex, 2)
    values = np.arange(space.size, dtype=np.float64)
    figure = plot_cochain(geometry, space.form(values, ORDINARY_FORM))
    mesh = next(trace for trace in figure.data if trace.name == "coefficient faces")
    assert mesh.hoverinfo == "skip"


def test_cochain_plot_rejects_foreign_equal_shaped_geometry() -> None:
    geometry = _triangle()
    foreign = _triangle()
    form = CochainSpace(geometry.complex, 0).form(
        np.ones(geometry.complex.vertex_count), ORDINARY_FORM
    )
    with pytest.raises(PlotError, match="different complex"):
        plot_cochain(foreign, form)


def test_edge_cochain_uses_canonical_midpoints_not_vectors() -> None:
    geometry = _triangle()
    space = CochainSpace(geometry.complex, 1)
    values = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    figure = plot_cochain(geometry, space.form(values, ORDINARY_FORM))
    trace = next(item for item in figure.data if item.name == "coefficients")
    edges = geometry.complex.simplices(1)
    midpoints = geometry.positions[edges].mean(axis=1)
    np.testing.assert_allclose(np.asarray(trace.x), midpoints[:, 0])
    np.testing.assert_allclose(np.asarray(trace.y), midpoints[:, 1])
    np.testing.assert_array_equal(np.asarray(trace.marker.color), values)


def test_homology_cycle_plot_uses_chain_coefficients() -> None:
    geometry = _cycle()
    basis = real_homology_basis(geometry.complex, 1)
    before = basis.cycle_coefficients().copy()
    figure = plot_homology_cycle(geometry, basis, 0)
    trace = next(item for item in figure.data if item.name == "cycle")
    assert trace.showlegend is False
    coefficients = basis.cycle_coefficients()[:, 0].toarray().reshape(-1)
    selected = np.flatnonzero(coefficients)

    np.testing.assert_array_equal(np.asarray(trace.customdata)[:, 0], selected)
    np.testing.assert_array_equal(
        np.asarray(trace.customdata)[:, 1], coefficients[selected]
    )
    assert (basis.cycle_coefficients() != before).nnz == 0

    with pytest.raises(PlotError, match="cycle"):
        plot_homology_cycle(geometry, basis, True)
    with pytest.raises(PlotError, match="cycle"):
        plot_homology_cycle(geometry, basis, 1)
    with pytest.raises(PlotError, match="different complex"):
        plot_homology_cycle(_cycle(), basis, 0)


def test_geometry_plot_supports_ambient_zero_and_canonical_hover_indices() -> None:
    point = Complex.from_maximal_simplices(np.array([[0]], dtype=np.int64))
    point_geometry = Geometry.from_positions(point, np.empty((1, 0), dtype=np.float64))
    point_figure = plot_geometry(point_geometry)
    point_trace = next(trace for trace in point_figure.data if trace.name == "vertices")
    np.testing.assert_array_equal(np.asarray(point_trace.x), [0.0])
    np.testing.assert_array_equal(np.asarray(point_trace.y), [0.0])

    triangle = _triangle()
    figure = plot_geometry(triangle)
    edge_trace = next(trace for trace in figure.data if trace.name == "edge indices")
    face_trace = next(trace for trace in figure.data if trace.name == "face indices")
    np.testing.assert_array_equal(
        np.asarray(edge_trace.customdata),
        np.arange(triangle.complex.simplex_count(1)),
    )
    np.testing.assert_array_equal(
        np.asarray(face_trace.customdata),
        np.arange(triangle.complex.simplex_count(2)),
    )


def test_face_cochain_colors_canonical_triangle_cells() -> None:
    geometry = _tetrahedron_surface()
    space = CochainSpace(geometry.complex, 2)
    values = np.array([-2.0, -1.0, 1.0, 3.0], dtype=np.float64)
    figure = plot_cochain(geometry, space.form(values, ORDINARY_FORM))
    cells = next(item for item in figure.data if item.name == "coefficient faces")
    np.testing.assert_array_equal(np.asarray(cells.intensity), values)
    assert cells.intensitymode == "cell"
    assert json.loads(figure.to_json())["data"]


def test_degree_two_homology_highlights_chain_owned_faces() -> None:
    geometry = _tetrahedron_surface()
    basis = real_homology_basis(geometry.complex, 2)
    figure = plot_homology_cycle(geometry, basis, 0)
    faces = next(item for item in figure.data if item.name == "cycle faces")
    coefficients = basis.cycle_coefficients()[:, 0].toarray().reshape(-1)
    selected = np.flatnonzero(coefficients)
    np.testing.assert_array_equal(np.asarray(faces.customdata)[:, 0], selected)
    np.testing.assert_array_equal(
        np.asarray(faces.customdata)[:, 1], coefficients[selected]
    )


@pytest.mark.parametrize(
    ("adapter", "arguments"),
    [
        (plot_geometry, (object(),)),
        (plot_cochain, (object(), object())),
        (plot_homology_cycle, (object(), object(), 0)),
    ],
)
def test_plotting_boundary_rejects_malformed_values(
    adapter: Any, arguments: tuple[object, ...]
) -> None:
    with pytest.raises(PlotError, match="requires"):
        adapter(*arguments)

    with pytest.raises(PlotError, match="title"):
        cast(Any, plot_geometry)(_triangle(), title=object())

    for malformed_axes in (object(), 0):
        with pytest.raises(PlotError, match="projection"):
            cast(Any, plot_geometry)(_triangle(), axes=malformed_axes)


def test_plotting_dimension_and_empty_subspace_matrix() -> None:
    segment = Complex.from_maximal_simplices(np.array([[0, 1]], dtype=np.int64))
    line = Geometry.from_positions(segment, np.array([[0.0], [2.0]], dtype=np.float64))
    line_figure = plot_geometry(line)
    line_vertices = next(
        trace for trace in line_figure.data if trace.name == "vertices"
    )
    np.testing.assert_array_equal(np.asarray(line_vertices.y), [0.0, 0.0])

    tetrahedron = _tetrahedron_surface()
    figure_3d = plot_geometry(tetrahedron)
    assert figure_3d.layout.scene.aspectmode == "data"
    assert figure_3d.layout.width is None

    high_positions = np.column_stack(
        [tetrahedron.positions, np.ones(tetrahedron.complex.vertex_count)]
    )
    high = Geometry.from_positions(tetrahedron.complex, high_positions)
    projected = plot_geometry(high, axes=(0, 2))
    assert projected.layout.yaxis.scaleanchor == "x"

    face_space = CochainSpace(tetrahedron.complex, 2)
    selected_space = CochainSubspace(face_space, np.array([1, 3], dtype=np.int64))
    selected_form = selected_space.form(
        np.array([-4.0, 2.0], dtype=np.float64), ORDINARY_FORM
    )
    selected_figure = plot_cochain(tetrahedron, selected_form)
    selected_cells = next(
        trace for trace in selected_figure.data if trace.name == "coefficient faces"
    )
    np.testing.assert_array_equal(np.asarray(selected_cells.intensity), [-4.0, 2.0])

    empty_space = CochainSubspace(
        CochainSpace(tetrahedron.complex, 0), np.array([], dtype=np.int64)
    )
    empty = plot_cochain(
        tetrahedron,
        empty_space.form(np.array([], dtype=np.float64), ORDINARY_FORM),
    )
    empty_trace = next(trace for trace in empty.data if trace.name == "coefficients")
    assert len(empty_trace.x) == 0


def test_degree_zero_homology_and_cycle_serialization() -> None:
    points = Complex.from_maximal_simplices(np.array([[0], [1]], dtype=np.int64))
    geometry = Geometry.from_positions(
        points, np.array([[0.0], [1.0]], dtype=np.float64)
    )
    basis = real_homology_basis(points, 0)
    figure = plot_homology_cycle(geometry, basis, 0)
    assert json.loads(figure.to_json())["data"]
    trace = next(item for item in figure.data if item.name == "cycle")
    assert np.asarray(trace.customdata).shape[1] == 2
