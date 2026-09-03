from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.sparse import csgraph

from polygeo.form import ElementError
from polygeo.topology import Complex, SimplicialError


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


def test_connectivity_no_longer_uses_python_scipy_size_gate(
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
        assert refined.require_connected() is None

    assert calls == []

    with pytest.raises(SimplicialError, match="the complex is disconnected"):
        Complex.from_maximal_simplices(
            np.arange(128, dtype=np.int64)[:, None]
        ).connected()


def test_construction_owns_source_and_exposed_arrays() -> None:
    maximal = _disk()
    complex_ = Complex.from_maximal_simplices(maximal)
    maximal[0, 0] = 99

    first = complex_.simplices_numpy_copy(2)
    first[0, 0] = 88

    assert complex_.dimension == 2
    assert complex_.vertex_count == 4
    np.testing.assert_array_equal(
        complex_.simplices_numpy_copy(2),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )


def test_construction_builds_canonical_arbitrary_dimensional_closure() -> None:
    simplex_4 = np.array([[4, 2, 0, 3, 1]], dtype=np.int64)
    complex_ = Complex.from_maximal_simplices(simplex_4)

    assert complex_.dimension == 4
    assert [complex_.simplex_count(k) for k in range(5)] == [5, 10, 10, 5, 1]
    np.testing.assert_array_equal(complex_.simplices_numpy_copy(4), [[0, 1, 2, 3, 4]])
    assert complex_.orientations_numpy_copy(4).tolist() == [-1]


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

    boundary_1 = complex_.boundary_scipy_copy(1)
    boundary_2 = complex_.boundary_scipy_copy(2)
    boundary_3 = complex_.boundary_scipy_copy(3)

    assert boundary_1.shape == (4, 6)
    assert boundary_2.shape == (6, 4)
    assert boundary_3.shape == (4, 1)
    assert (boundary_1 @ boundary_2).nnz == 0
    assert (boundary_2 @ boundary_3).nnz == 0

    boundary_2.data[:] = 0
    assert complex_.boundary_scipy_copy(2).nnz == 12
    assert complex_.boundary_scipy_copy(0).shape == (0, 4)

    with pytest.raises(SimplicialError):
        complex_.boundary_scipy_copy(-1)
    with pytest.raises(SimplicialError):
        complex_.boundary_scipy_copy(4)


def test_boundary_matrix_preserves_input_top_orientation() -> None:
    forward = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
    reverse = Complex.from_maximal_simplices(np.array([[0, 2, 1]], dtype=np.int64))

    np.testing.assert_array_equal(
        reverse.boundary_scipy_copy(2).toarray(),
        -forward.boundary_scipy_copy(2).toarray(),
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
    assert star.mask_numpy_copy(0).tolist() == [True, False, False]
    assert star.mask_numpy_copy(1).tolist() == [True, True, False]
    assert star.mask_numpy_copy(2).tolist() == [True]

    link = vertex_zero.link()
    assert link.mask_numpy_copy(0).tolist() == [False, True, True]
    assert link.mask_numpy_copy(1).tolist() == [False, False, True]
    assert link.mask_numpy_copy(2).tolist() == [False]
    assert link.closure().is_pure(1)

    face = complex_.subset(
        (
            np.zeros(3, dtype=np.bool_),
            np.zeros(3, dtype=np.bool_),
            np.ones(1, dtype=np.bool_),
        )
    )
    closure = face.closure()
    assert closure.mask_numpy_copy(0).all()
    assert closure.mask_numpy_copy(1).all()
    assert closure.mask_numpy_copy(2).all()
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
            left_empty.mask_numpy_copy(invalid_degree)
        with pytest.raises(SimplicialError):
            left_empty.is_pure(invalid_degree)


def test_refinement_methods_preserve_literal_identity() -> None:
    raw = Complex.from_maximal_simplices(_tetrahedron_boundary())
    triangle = raw.triangle_manifold()
    oriented = triangle.oriented()
    closed = oriented.without_boundary()
    domain = closed.connected()

    assert triangle is raw
    assert oriented is raw
    assert closed is raw
    assert domain is raw
    domain.require_triangle()
    domain.require_oriented()
    domain.require_without_boundary()
    domain.require_connected()
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
    assert disk.with_boundary() is disk
    with pytest.raises(SimplicialError):
        disk.without_boundary()

    sphere = Complex.from_maximal_simplices(_tetrahedron_boundary()).triangle_manifold()
    assert sphere.without_boundary() is sphere
    with pytest.raises(SimplicialError):
        sphere.with_boundary()

    nonregular = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
    )
    assert not hasattr(nonregular, "closed")
    with pytest.raises(SimplicialError) as unknown:
        getattr(nonregular, "without_boundary")()
    assert unknown.value.reason == "capability_not_admitted"

    disconnected = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    ).triangle_manifold()
    with pytest.raises(SimplicialError):
        disconnected.connected()


def test_cochain_spaces_and_forms_own_coefficients() -> None:
    left = Complex.from_maximal_simplices(_disk())
    right = Complex.from_maximal_simplices(_disk())
    left_space = left.binary64_cochain_space(1)
    right_space = right.binary64_cochain_space(1)
    coefficients = np.arange(left_space.size, dtype=np.float64)

    form = left_space.admit_numpy(coefficients)
    coefficients[:] = -1
    exposed = form.coefficients_numpy_copy()
    exposed[:] = -2

    np.testing.assert_array_equal(
        form.coefficients_numpy_copy(),
        np.arange(left_space.size, dtype=np.float64),
    )
    assert form.space.same_space(left.binary64_cochain_space(1))
    assert not form.space.same_space(right.binary64_cochain_space(1))
    assert not left_space.same_space(right_space)

    with pytest.raises(ElementError):
        left_space.admit_numpy(np.zeros(left_space.size + 1))
    with pytest.raises(SimplicialError):
        left.binary64_cochain_space(3)


def test_form_requires_real_finite_coefficients() -> None:
    space = Complex.from_maximal_simplices(_disk()).binary64_cochain_space(0)
    values = np.zeros(space.size)
    values[0] = np.nan
    with pytest.raises(ElementError):
        space.admit_numpy(values)

    complex_values = np.ones(space.size, dtype=np.complex128) * (1.0 + 2.0j)
    with pytest.raises((TypeError, ValueError)):
        space.admit_numpy(complex_values)

    for unsupported in (
        np.full(space.size, "coefficient", dtype=object),
        np.full(space.size, object(), dtype=object),
        np.full(space.size, 1.0 + 2.0j, dtype=object),
    ):
        with pytest.raises((TypeError, ValueError)):
            space.admit_numpy(unsupported)
