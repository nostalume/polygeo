"""Topology projections are exact, final, and caller-owned."""

from __future__ import annotations

import gc

import numpy as np
import pytest
from scipy.sparse import csr_array

from polygeo.topology import Complex as NativeComplex
from polygeo.topology import SimplicialError as NativeSimplicialError

from topology_cases import simplex_case
from topology_oracle import admit_oracle


def test_fixed_shape_exports_are_final_owned_numpy_arrays() -> None:
    native = NativeComplex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    )
    subset = native.subset(
        (
            np.array([True, False, False, False]),
            np.zeros(5, dtype=np.bool_),
            np.zeros(2, dtype=np.bool_),
        )
    )
    selection = native.selection(1, np.array([0, 2, 4], dtype=np.int64))
    data, indices, indptr, shape = native.boundary_parts_numpy_copy(2)

    outputs = (
        (native.simplices_numpy_copy(2), np.dtype(np.int64), (2, 3)),
        (native.orientations_numpy_copy(2), np.dtype(np.int8), (2,)),
        (subset.mask_numpy_copy(0), np.dtype(np.bool_), (4,)),
        (selection.indices_numpy_copy(), np.dtype(np.int64), (3,)),
        (data, np.dtype(np.int8), (6,)),
        (indices, np.dtype(np.int32), (6,)),
        (indptr, np.dtype(np.int32), (6,)),
    )
    assert shape == (5, 2)
    for output, dtype, shape in outputs:
        assert isinstance(output, np.ndarray)
        assert output.dtype == dtype
        assert output.shape == shape
        assert output.flags.c_contiguous
        assert output.flags.owndata


def test_dense_exports_are_fresh_and_mutation_isolated() -> None:
    native = NativeComplex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
    first_simplices = native.simplices_numpy_copy(2)
    first_orientation = native.orientations_numpy_copy(2)
    first_simplices[:] = 9
    first_orientation[:] = 0

    np.testing.assert_array_equal(native.simplices_numpy_copy(2), [[0, 1, 2]])
    np.testing.assert_array_equal(native.orientations_numpy_copy(2), [1])


def test_owned_dense_exports_outlive_topology_handles() -> None:
    native = NativeComplex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
    subset = native.subset(
        (
            np.array([True, False, False]),
            np.zeros(3, dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),
        )
    )
    selection = native.selection(1, np.array([0, 2], dtype=np.int64))
    outputs = (
        native.simplices_numpy_copy(2),
        native.orientations_numpy_copy(2),
        subset.mask_numpy_copy(0),
        selection.indices_numpy_copy(),
    )
    del native, subset, selection
    gc.collect()

    expected = ([[0, 1, 2]], [1], [True, False, False], [0, 2])
    for output, values in zip(outputs, expected, strict=True):
        np.testing.assert_array_equal(output, values)


@pytest.mark.parametrize("dimension", range(5))
@pytest.mark.parametrize("reversed_top", [False, True])
def test_topology_and_csr_projection_match_exact_oracle(
    dimension: int, reversed_top: bool
) -> None:
    maximal = simplex_case(dimension, reversed_top=reversed_top)
    oracle = admit_oracle(maximal)
    native = NativeComplex.from_maximal_simplices(maximal)

    assert native.vertex_count == oracle.vertex_count
    assert native.dimension == oracle.dimension
    for degree in range(dimension + 1):
        np.testing.assert_array_equal(
            native.simplices_numpy_copy(degree), oracle.simplices[degree]
        )
        np.testing.assert_array_equal(
            native.orientations_numpy_copy(degree), oracle.orientations[degree]
        )

        projected = native.boundary_scipy_copy(degree)
        expected = oracle.boundaries[degree]
        assert isinstance(projected, csr_array)
        assert projected.shape == expected.shape
        np.testing.assert_array_equal(projected.data, expected.data)
        np.testing.assert_array_equal(projected.indices, expected.indices)
        np.testing.assert_array_equal(projected.indptr, expected.indptr)


def test_projection_is_caller_owned_and_outlives_topology_handles() -> None:
    native = NativeComplex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
    projected = native.boundary_scipy_copy(2)
    expected = projected.copy()
    del native
    gc.collect()

    projected.data[:] = 0
    assert projected.nnz == expected.nnz
    np.testing.assert_array_equal(expected.toarray(), [[1], [-1], [1]], strict=False)
    assert np.count_nonzero(projected.data) == 0


def test_projection_mutation_never_changes_retained_boundary() -> None:
    native = NativeComplex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
    first = native.boundary_scipy_copy(2)
    first.data[:] = 0
    second = native.boundary_scipy_copy(2)

    np.testing.assert_array_equal(second.toarray(), [[1], [-1], [1]])


def test_projection_rejects_unrepresented_degree_with_reason() -> None:
    native = NativeComplex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))

    with pytest.raises(NativeSimplicialError) as caught:
        native.boundary_scipy_copy(3)
    assert caught.value.reason == "degree_outside"
