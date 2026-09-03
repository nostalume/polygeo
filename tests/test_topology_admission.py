"""Topology admission, error, ownership, and lifetime contracts."""

from __future__ import annotations

import gc

import numpy as np
import pytest

from polygeo.topology import Complex as NativeComplex
from polygeo.topology import SimplicialError, SimplicialError as NativeSimplicialError


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ],
)
def test_admission_accepts_every_fixed_width_integer_dtype(
    dtype: np.dtype,
) -> None:
    source = np.array([[0, 1, 2]], dtype=dtype)
    native = NativeComplex.from_maximal_simplices(source)

    np.testing.assert_array_equal(
        native.boundary_scipy_copy(2).toarray(), [[1], [-1], [1]]
    )


def test_admission_accepts_strided_and_non_native_endian_input() -> None:
    storage = np.array([[0, 99, 1, 99, 2]], dtype=np.int64)
    strided = storage[:, ::2]
    non_native = np.array([[0, 1, 2]], dtype=">i8")

    expected = np.array([[1], [-1], [1]], dtype=np.int8)
    np.testing.assert_array_equal(
        NativeComplex.from_maximal_simplices(strided).boundary_scipy_copy(2).toarray(),
        expected,
    )
    np.testing.assert_array_equal(
        NativeComplex.from_maximal_simplices(non_native)
        .boundary_scipy_copy(2)
        .toarray(),
        expected,
    )


@pytest.mark.parametrize(
    ("invalid", "reason"),
    [
        (np.array([[False, True]], dtype=np.bool_), "unsupported_dtype"),
        (np.array([[0, 1, 2]], dtype=object), "unsupported_dtype"),
        (np.array([0, 1, 2], dtype=np.int64), "candidate_shape"),
        (np.array([[0, -1, 2]], dtype=np.int64), "negative_index"),
        (np.array([[0, 1, 1]], dtype=np.int64), "repeated_vertex"),
    ],
)
def test_admission_maps_stable_family_and_reason(
    invalid: np.ndarray, reason: str
) -> None:
    with pytest.raises(NativeSimplicialError) as caught:
        NativeComplex.from_maximal_simplices(invalid)

    assert isinstance(caught.value, SimplicialError)
    assert caught.value.reason == reason


def test_owner_detaches_from_input_and_distinguishes_equal_admissions() -> None:
    source = np.array([[0, 1, 2]], dtype=np.int64)
    first = NativeComplex.from_maximal_simplices(source)
    second = NativeComplex.from_maximal_simplices(source)
    source[:] = 99

    np.testing.assert_array_equal(first.orientations_numpy_copy(2), [1])
    assert not first.shares_data_with(second)
    assert first.shares_data_with(first)


def test_owner_bound_subset_keeps_refined_owner_alive() -> None:
    raw = NativeComplex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    )
    raw.triangle_manifold()
    boundary = raw.boundary_subset()
    del raw
    gc.collect()

    np.testing.assert_array_equal(
        boundary.mask_numpy_copy(1), [True, False, True, True, True]
    )


def test_topology_owner_has_no_implicit_array_coercion() -> None:
    native = NativeComplex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))

    assert not hasattr(native, "__array__")
    coerced = np.asarray(native)
    assert coerced.dtype == np.dtype(object)
    assert coerced.item() is native
