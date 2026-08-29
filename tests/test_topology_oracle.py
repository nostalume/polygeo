"""Independent-oracle equivalence and mutation-sensitivity contracts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from polygeo import Complex, SimplicialError, topological_boundary

from topology_cases import simplex_case, triangle_grid
from topology_oracle import OracleComplex, OracleError, admit_oracle


def _assert_topology_equal(public: Complex, oracle: OracleComplex) -> None:
    assert public.vertex_count == oracle.vertex_count
    assert public.dimension == oracle.dimension
    for degree in range(oracle.dimension + 1):
        np.testing.assert_array_equal(
            public.simplices(degree), oracle.simplices[degree]
        )
        np.testing.assert_array_equal(
            public.orientations(degree), oracle.orientations[degree]
        )
        observed = public.boundary_matrix(degree)
        expected = oracle.boundaries[degree]
        np.testing.assert_array_equal(observed.data, expected.data)
        np.testing.assert_array_equal(observed.indices, expected.indices)
        np.testing.assert_array_equal(observed.indptr, expected.indptr)


@pytest.mark.parametrize("dimension", range(5))
@pytest.mark.parametrize("reversed_top", [False, True])
def test_independent_oracle_matches_public_exact_topology(
    dimension: int, reversed_top: bool
) -> None:
    maximal = simplex_case(dimension, reversed_top=reversed_top)
    _assert_topology_equal(
        Complex.from_maximal_simplices(maximal), admit_oracle(maximal)
    )


def test_independent_oracle_does_not_call_public_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise AssertionError("the frozen oracle called the public implementation")

    monkeypatch.setattr(Complex, "from_maximal_simplices", broken)
    oracle = admit_oracle(np.array([[0, 1, 2]], dtype=np.int64))

    assert oracle.dimension == 2
    np.testing.assert_array_equal(oracle.boundaries[2].toarray(), [[1], [-1], [1]])


def test_independent_oracle_detects_a_wrong_boundary_sign() -> None:
    maximal = np.array([[0, 1, 2]], dtype=np.int64)
    observed = Complex.from_maximal_simplices(maximal).boundary_matrix(2).toarray()
    expected = admit_oracle(maximal).boundaries[2].toarray()
    observed[0, 0] *= -1

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize(
    ("maximal", "public_operation", "oracle_operation"),
    [
        (
            np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64),
            lambda value: value.triangle_manifold(),
            lambda value: value.triangle_boundary(),
        ),
        (
            np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64),
            lambda value: value.oriented(),
            lambda value: value.require_oriented(),
        ),
        (
            np.array([[0, 1], [2, 3]], dtype=np.int64),
            lambda value: value.connected(),
            lambda value: value.require_connected(),
        ),
    ],
)
def test_independent_oracle_matches_refinement_rejection(
    maximal: np.ndarray,
    public_operation: Callable[[Complex], object],
    oracle_operation: Callable[[OracleComplex], object],
) -> None:
    with pytest.raises(SimplicialError):
        public_operation(Complex.from_maximal_simplices(maximal))
    with pytest.raises(OracleError):
        oracle_operation(admit_oracle(maximal))


def test_independent_oracle_matches_refinements_and_boundary_masks() -> None:
    maximal = triangle_grid(2)
    public = Complex.from_maximal_simplices(maximal)
    oracle = admit_oracle(maximal)

    public.codimension_one_regular()
    public.with_boundary()
    public.triangle_manifold()
    public.oriented()
    public.connected()
    oracle.triangle_boundary()
    oracle.require_oriented()
    oracle.require_connected()

    expected = oracle.regular_boundary()
    observed = topological_boundary(public)
    for degree in range(public.dimension + 1):
        np.testing.assert_array_equal(observed.mask(degree), expected[degree])


def test_independent_oracle_matches_subset_relations_and_purity() -> None:
    maximal = triangle_grid(2)
    public = Complex.from_maximal_simplices(maximal)
    oracle = admit_oracle(maximal)
    masks = tuple(
        np.zeros(public.simplex_count(degree), dtype=np.bool_)
        for degree in range(public.dimension + 1)
    )
    masks[0][4] = True
    masks[1][0] = True
    public_subset = public.subset(masks)
    oracle_subset = oracle.subset(masks)

    for public_result, oracle_result in (
        (public_subset.closure(), oracle_subset.closure()),
        (public_subset.star(), oracle_subset.star()),
        (public_subset.link(), oracle_subset.link()),
    ):
        for degree in range(public.dimension + 1):
            np.testing.assert_array_equal(
                public_result.mask(degree), oracle_result.masks[degree]
            )
    for degree in range(public.dimension + 1):
        assert public_subset.is_pure(degree) == oracle_subset.is_pure(degree)
