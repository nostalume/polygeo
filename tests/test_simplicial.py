from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.sparse import csgraph

from polygeo import (
    ORDINARY_FORM,
    BoundaryUnknown,
    CochainSpace,
    Complex,
    Connected,
    ConnectivityUnknown,
    OrdinaryForm,
    OrientationUnknown,
    Oriented,
    Simplicial,
    SimplicialError,
    TriangleManifold,
    WithBoundary,
    WithoutBoundary,
)


def _tetrahedron_boundary() -> np.ndarray:
    return np.array(
        [
            [1, 2, 3],
            [0, 3, 2],
            [0, 1, 3],
            [0, 2, 1],
        ],
        dtype=np.int64,
    )


def _disk() -> np.ndarray:
    return np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def test_connectivity_uses_scipy_only_at_the_size_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = csgraph.connected_components

    def tracked(graph: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(graph.shape[0])
        return original(graph, *args, **kwargs)

    monkeypatch.setattr(csgraph, "connected_components", tracked)
    for vertex_count in (127, 128):
        vertices = np.arange(vertex_count, dtype=np.int64)
        edges = np.column_stack((vertices[:-1], vertices[1:]))
        refined = Complex.from_maximal_simplices(
            edges, vertex_count=vertex_count
        ).connected()
        assert isinstance(refined.connectivity_state, Connected)

    assert calls == [128]

    with pytest.raises(SimplicialError, match="the complex is disconnected"):
        Complex.from_maximal_simplices(
            np.arange(128, dtype=np.int64)[:, None]
        ).connected()


def test_construction_owns_source_and_exposed_arrays() -> None:
    maximal = _disk()
    complex_ = Complex.from_maximal_simplices(maximal)
    maximal[0, 0] = 99

    first = complex_.simplices(2)
    first[0, 0] = 88

    assert complex_.dimension == 2
    assert complex_.vertex_count == 4
    np.testing.assert_array_equal(
        complex_.simplices(2),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )


def test_construction_builds_canonical_arbitrary_dimensional_closure() -> None:
    simplex_4 = np.array([[4, 2, 0, 3, 1]], dtype=np.int64)
    complex_ = Complex.from_maximal_simplices(simplex_4)

    assert complex_.dimension == 4
    assert [complex_.simplex_count(k) for k in range(5)] == [5, 10, 10, 5, 1]
    np.testing.assert_array_equal(complex_.simplices(4), [[0, 1, 2, 3, 4]])
    assert complex_.orientations(4).tolist() == [-1]


def test_construction_rejects_invalid_maximal_simplices() -> None:
    invalid = [
        np.array([], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
        np.array([[0.0, 1.0, 2.0]]),
        np.array([[0, -1, 2]], dtype=np.int64),
        np.array([[0, 1, 1]], dtype=np.int64),
        np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64),
    ]
    for maximal in invalid:
        with pytest.raises(SimplicialError):
            Complex.from_maximal_simplices(maximal)

    with pytest.raises(SimplicialError):
        Complex.from_maximal_simplices(_disk(), vertex_count=3)


def test_boundary_matrix_is_intrinsic_fresh_and_squares_to_zero() -> None:
    complex_ = Complex.from_maximal_simplices(np.array([[0, 1, 2, 3]], dtype=np.int64))

    boundary_1 = complex_.boundary_matrix(1)
    boundary_2 = complex_.boundary_matrix(2)
    boundary_3 = complex_.boundary_matrix(3)

    assert boundary_1.shape == (4, 6)
    assert boundary_2.shape == (6, 4)
    assert boundary_3.shape == (4, 1)
    assert (boundary_1 @ boundary_2).nnz == 0
    assert (boundary_2 @ boundary_3).nnz == 0

    boundary_2.data[:] = 0
    assert complex_.boundary_matrix(2).nnz == 12
    assert complex_.boundary_matrix(0).shape == (0, 4)

    with pytest.raises(SimplicialError):
        complex_.boundary_matrix(-1)
    with pytest.raises(SimplicialError):
        complex_.boundary_matrix(4)


def test_boundary_matrix_preserves_input_top_orientation() -> None:
    forward = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
    reverse = Complex.from_maximal_simplices(np.array([[0, 2, 1]], dtype=np.int64))

    np.testing.assert_array_equal(
        reverse.boundary_matrix(2).toarray(),
        -forward.boundary_matrix(2).toarray(),
    )


def test_subset_closure_star_link_and_purity() -> None:
    complex_ = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
    vertex_zero = complex_.subset(
        (
            np.array([True, False, False]),
            np.zeros(3, dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),
        )
    )

    star = vertex_zero.star()
    assert star.mask(0).tolist() == [True, False, False]
    assert star.mask(1).tolist() == [True, True, False]
    assert star.mask(2).tolist() == [True]

    link = vertex_zero.link()
    assert link.mask(0).tolist() == [False, True, True]
    assert link.mask(1).tolist() == [False, False, True]
    assert link.mask(2).tolist() == [False]
    assert link.closure().is_pure(1)

    face = complex_.subset(
        (
            np.zeros(3, dtype=np.bool_),
            np.zeros(3, dtype=np.bool_),
            np.ones(1, dtype=np.bool_),
        )
    )
    closure = face.closure()
    assert closure.mask(0).all()
    assert closure.mask(1).all()
    assert closure.mask(2).all()
    assert closure.closure().same_members(closure)
    assert closure.is_pure(2)


def test_subset_rejects_wrong_shapes_and_foreign_comparison() -> None:
    left = Complex.from_maximal_simplices(_disk())
    right = Complex.from_maximal_simplices(_disk())

    with pytest.raises(SimplicialError):
        left.subset((np.ones(4, dtype=np.bool_),))

    left_empty = left.subset(
        tuple(
            np.zeros(left.simplex_count(k), dtype=np.bool_)
            for k in range(left.dimension + 1)
        )
    )
    right_empty = right.subset(
        tuple(
            np.zeros(right.simplex_count(k), dtype=np.bool_)
            for k in range(right.dimension + 1)
        )
    )
    with pytest.raises(SimplicialError):
        left_empty.same_members(right_empty)

    for invalid_degree in (-1, left.dimension + 1):
        with pytest.raises(SimplicialError):
            left_empty.mask(invalid_degree)
        with pytest.raises(SimplicialError):
            left_empty.is_pure(invalid_degree)


def test_refinement_methods_preserve_data_and_strengthen_one_state() -> None:
    raw = Complex.from_maximal_simplices(_tetrahedron_boundary())
    triangle = raw.triangle_manifold()
    oriented = triangle.oriented()
    closed = oriented.without_boundary()
    domain = closed.connected()

    assert isinstance(raw.boundary_state, BoundaryUnknown)
    assert isinstance(raw.orientation_state, OrientationUnknown)
    assert isinstance(raw.connectivity_state, ConnectivityUnknown)
    assert isinstance(raw.topology_state, Simplicial)

    assert isinstance(domain.boundary_state, WithoutBoundary)
    assert isinstance(domain.orientation_state, Oriented)
    assert isinstance(domain.connectivity_state, Connected)
    assert isinstance(domain.topology_state, TriangleManifold)
    assert domain.shares_data_with(raw)


def test_triangle_manifold_rejects_wrong_dimension_nonmanifold_edge_and_fan() -> None:
    graph = Complex.from_maximal_simplices(np.array([[0, 1], [1, 2]], dtype=np.int64))
    with pytest.raises(SimplicialError):
        graph.triangle_manifold()

    nonmanifold_edge = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
    )
    with pytest.raises(SimplicialError):
        nonmanifold_edge.triangle_manifold()

    disconnected_fan = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
    )
    with pytest.raises(SimplicialError):
        disconnected_fan.triangle_manifold()


def test_oriented_boundary_and_connected_refinements_fail_independently() -> None:
    flipped = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 3, 2]], dtype=np.int64)
    ).triangle_manifold()
    with pytest.raises(SimplicialError):
        flipped.oriented()

    disk = Complex.from_maximal_simplices(_disk()).triangle_manifold().oriented()
    assert isinstance(disk.with_boundary().boundary_state, WithBoundary)
    with pytest.raises(SimplicialError):
        disk.without_boundary()

    sphere = Complex.from_maximal_simplices(_tetrahedron_boundary()).triangle_manifold()
    assert isinstance(sphere.without_boundary().boundary_state, WithoutBoundary)
    with pytest.raises(SimplicialError):
        sphere.with_boundary()

    nonregular = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
    )
    assert not hasattr(nonregular, "closed")
    with pytest.raises(SimplicialError, match="codimension-one regular"):
        getattr(nonregular, "without_boundary")()

    disconnected = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    ).triangle_manifold()
    with pytest.raises(SimplicialError):
        disconnected.connected()


def test_cochain_spaces_and_forms_own_coefficients() -> None:
    left = Complex.from_maximal_simplices(_disk())
    right = Complex.from_maximal_simplices(_disk())
    left_space = left.cochain_space(1)
    right_space = right.cochain_space(1)
    coefficients = np.arange(left_space.size, dtype=np.float64)

    form = left_space.form(coefficients, ORDINARY_FORM)
    coefficients[:] = -1
    exposed = form.coefficients()
    exposed[:] = -2

    np.testing.assert_array_equal(
        form.coefficients(),
        np.arange(left_space.size, dtype=np.float64),
    )
    assert isinstance(form.semantics, OrdinaryForm)
    assert form.space.belongs_to(left)
    assert not form.space.belongs_to(right)
    assert not left_space.same_space(right_space)

    with pytest.raises(SimplicialError):
        left_space.form(np.zeros(left_space.size + 1), ORDINARY_FORM)
    with pytest.raises(SimplicialError):
        left.cochain_space(3)
    with pytest.raises(SimplicialError):
        CochainSpace(left, 3)


def test_form_requires_real_finite_coefficients() -> None:
    space = Complex.from_maximal_simplices(_disk()).cochain_space(0)
    values = np.zeros(space.size)
    values[0] = np.nan
    with pytest.raises(SimplicialError):
        space.form(values, ORDINARY_FORM)

    complex_values = np.ones(space.size, dtype=np.complex128) * (1.0 + 2.0j)
    with pytest.raises(SimplicialError, match="real"):
        space.form(complex_values, ORDINARY_FORM)

    for unsupported in (
        np.full(space.size, "coefficient", dtype=object),
        np.full(space.size, object(), dtype=object),
        np.full(space.size, 1.0 + 2.0j, dtype=object),
    ):
        with pytest.raises(SimplicialError, match="numeric"):
            space.form(unsupported, ORDINARY_FORM)
