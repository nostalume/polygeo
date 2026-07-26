from __future__ import annotations

import numpy as np
import pytest
import polygeo.simplicial as _simplicial

from polygeo import (
    CodimensionOneRegular,
    Complex,
    SimplicialError,
    TriangleManifold,
    WithBoundary,
    WithoutBoundary,
    topological_boundary,
)


def test_codimension_one_regular_extracts_dimension_zero_empty_boundary() -> None:
    regular = Complex.from_maximal_simplices(
        np.array([[0], [1]], dtype=np.int64)
    ).codimension_one_regular()

    boundary = topological_boundary(regular)

    assert all(
        not boundary.mask(degree).any() for degree in range(regular.dimension + 1)
    )
    assert boundary.complex is regular


def test_codimension_one_regular_extracts_path_endpoints_and_cycle_empty_boundary() -> (
    None
):
    path = Complex.from_maximal_simplices(
        np.array([[0, 1], [1, 2]], dtype=np.int64)
    ).codimension_one_regular()
    boundary = topological_boundary(path)

    assert boundary.mask(0).tolist() == [True, False, True]
    assert not boundary.mask(1).any()

    cycle = Complex.from_maximal_simplices(
        np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    ).codimension_one_regular()
    cycle_boundary = topological_boundary(cycle)
    assert all(
        not cycle_boundary.mask(degree).any() for degree in range(cycle.dimension + 1)
    )


def test_codimension_one_regular_extracts_disk_rim_with_complete_unsigned_closure() -> (
    None
):
    disk = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    ).codimension_one_regular()

    boundary = topological_boundary(disk)

    assert boundary.mask(0).tolist() == [True, True, True, True]
    assert boundary.mask(1).tolist() == [True, False, True, True, True]
    assert boundary.mask(2).tolist() == [False, False]
    assert boundary.closure().same_members(boundary)
    assert boundary.is_pure(1)


def test_codimension_one_regular_extracts_single_simplex_boundary_in_dimensions_three_and_four() -> (
    None
):
    for dimension in (3, 4):
        domain = Complex.from_maximal_simplices(
            np.array([list(range(dimension + 1))], dtype=np.int64)
        ).codimension_one_regular()

        boundary = topological_boundary(domain)

        for degree in range(dimension):
            assert boundary.mask(degree).all()
        assert not boundary.mask(dimension).any()
        assert boundary.is_pure(dimension - 1)


def test_closed_triangle_manifold_is_codimension_one_regular_with_empty_boundary() -> (
    None
):
    sphere = Complex.from_maximal_simplices(
        np.array(
            [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
            dtype=np.int64,
        )
    ).triangle_manifold()

    boundary = topological_boundary(sphere)

    assert isinstance(sphere.topology_state, TriangleManifold)
    assert isinstance(sphere.topology_state, CodimensionOneRegular)
    assert all(
        not boundary.mask(degree).any() for degree in range(sphere.dimension + 1)
    )


def test_codimension_one_regular_rejects_branching_and_nonpure_input() -> None:
    branching_graph = Complex.from_maximal_simplices(
        np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int64)
    )
    with pytest.raises(SimplicialError, match="one or two"):
        branching_graph.codimension_one_regular()

    branching_surface = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
    )
    with pytest.raises(SimplicialError, match="one or two"):
        branching_surface.codimension_one_regular()

    isolated_vertex = Complex.from_maximal_simplices(
        np.array([[0, 1]], dtype=np.int64), vertex_count=3
    )
    with pytest.raises(SimplicialError, match="pure"):
        isolated_vertex.codimension_one_regular()


def test_codimension_one_evidence_cannot_be_forged_or_replaced() -> None:
    branching = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
    )
    empty_masks = tuple(
        np.zeros(branching.simplex_count(degree), dtype=np.bool_)
        for degree in range(branching.dimension + 1)
    )
    forged_state = object.__new__(CodimensionOneRegular)
    with pytest.raises(SimplicialError, match="verified topology evidence"):
        Complex(
            getattr(branching, "_data"),
            branching.boundary_state,
            branching.orientation_state,
            branching.connectivity_state,
            forged_state,
        )

    regular = Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64)
    ).codimension_one_regular()
    with pytest.raises(AttributeError):
        setattr(regular.topology_state, "_boundary_masks", empty_masks)
    with pytest.raises(AttributeError):
        setattr(regular, "_topology_state", branching.topology_state)


def test_complex_constructor_rejects_forged_evidence_and_boundary_state() -> None:
    raw = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
    )
    empty_masks = tuple(
        np.zeros(raw.simplex_count(degree), dtype=np.bool_)
        for degree in range(raw.dimension + 1)
    )

    class ForgedEvidence(_simplicial._BoundaryEvidence):
        pass

    forged_evidence = object.__new__(ForgedEvidence)
    object.__setattr__(forged_evidence, "_data", getattr(raw, "_data"))
    object.__setattr__(forged_evidence, "_masks", empty_masks)
    object.__setattr__(forged_evidence, "_sealed", True)
    with pytest.raises(SimplicialError, match="verified topology evidence"):
        Complex(
            getattr(raw, "_data"),
            raw.boundary_state,
            raw.orientation_state,
            raw.connectivity_state,
            CodimensionOneRegular(),
            forged_evidence,
        )

    regular = Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64)
    ).codimension_one_regular()
    with pytest.raises(SimplicialError, match="conflicts with topology evidence"):
        Complex(
            getattr(regular, "_data"),
            WithoutBoundary(),
            regular.orientation_state,
            regular.connectivity_state,
            regular.topology_state,
            getattr(regular, "_boundary_evidence"),
        )

    with pytest.raises(SimplicialError, match="requires regular topology evidence"):
        Complex(
            getattr(raw, "_data"),
            WithBoundary(),
            raw.orientation_state,
            raw.connectivity_state,
            raw.topology_state,
        )


def test_topological_boundary_closure_does_not_overflow_high_incidence() -> None:
    triangles = np.array(
        [[0, 2 * index + 1, 2 * index + 2] for index in range(128)],
        dtype=np.int64,
    )
    domain = Complex.from_maximal_simplices(triangles).codimension_one_regular()

    boundary = topological_boundary(domain)

    assert boundary.mask(0).all()
    assert boundary.mask(1).all()
    assert not boundary.mask(2).any()
    assert boundary.closure().same_members(boundary)
