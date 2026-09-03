"""Halfedge admission, navigation, topology, projection, and conversion laws."""

from __future__ import annotations

import numpy as np
import pytest

from polygeo.chain import DEFAULT_LAW_LIMIT, ChainLawLimit
from polygeo.topology import (
    Complex,
    HalfedgeError,
    HalfedgeSurface,
)


NEXT = np.array([1, 2, 0, 5, 3, 4], dtype=np.int64)
TWIN = np.array([3, 4, 5, 0, 1, 2], dtype=np.int64)
EXTERIOR = np.array([3], dtype=np.int64)


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64],
)
def test_admission_accepts_fixed_width_integer_arrays(dtype: np.dtype) -> None:
    surface = HalfedgeSurface.from_permutations(
        NEXT.astype(dtype), TWIN.astype(dtype), exterior_faces=EXTERIOR.astype(dtype)
    )
    assert (surface.halfedge_count, surface.vertex_count, surface.edge_count) == (
        6,
        3,
        3,
    )


def test_admission_accepts_striding_and_non_native_endian() -> None:
    storage = np.column_stack((NEXT, np.full(6, 99, dtype=np.int64))).ravel()
    surface = HalfedgeSurface.from_permutations(
        storage[::2], TWIN.astype(">i8"), exterior_faces=EXTERIOR.astype(">u8")
    )
    np.testing.assert_array_equal(surface.next_numpy_copy(), NEXT)


@pytest.mark.parametrize(
    ("next_", "twin", "reason"),
    [
        (np.array([1, -1]), np.array([1, 0]), "negative_index"),
        (np.array([1, 0], dtype=object), np.array([1, 0]), "unsupported_dtype"),
        (np.array([[1, 0]]), np.array([1, 0]), "halfedge_shape"),
        (np.array([1, 0]), np.array([0, 1]), "twin_law"),
    ],
)
def test_admission_has_stable_structured_failures(
    next_: np.ndarray, twin: np.ndarray, reason: str
) -> None:
    with pytest.raises(HalfedgeError) as caught:
        HalfedgeSurface.from_permutations(next_, twin)
    assert caught.value.reason == reason


def test_navigation_order_copy_isolation_and_immutability() -> None:
    surface = HalfedgeSurface.from_permutations(NEXT, TWIN, exterior_faces=EXTERIOR)
    assert (
        surface.face_orbit_count,
        surface.material_face_count,
        surface.exterior_face_count,
    ) == (2, 1, 1)
    np.testing.assert_array_equal(surface.twin_numpy_copy(), TWIN)
    np.testing.assert_array_equal(surface.vertex_of_numpy_copy(), [0, 1, 2, 1, 2, 0])
    np.testing.assert_array_equal(surface.edge_of_numpy_copy(), [0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(surface.face_of_numpy_copy(), [0, 0, 0, 1, 1, 1])
    offsets, exterior, material = surface.boundary_cycles_numpy_copy()
    np.testing.assert_array_equal(offsets, [0, 3])
    np.testing.assert_array_equal(exterior, [3, 5, 4])
    np.testing.assert_array_equal(material, [0, 2, 1])
    exterior[:] = 0
    np.testing.assert_array_equal(surface.boundary_cycles_numpy_copy()[1], [3, 5, 4])
    with pytest.raises(AttributeError):
        setattr(surface, "vertex_count", 0)
    assert not any(
        hasattr(surface, name) for name in ("cells", "cw_complex", "cell_complex")
    )


def test_boundary_projection_is_a_fresh_owned_int64_copy() -> None:
    surface = HalfedgeSurface.from_permutations(NEXT, TWIN, exterior_faces=EXTERIOR)
    first = surface.boundary_scipy_copy(2)
    second = surface.boundary_scipy_copy(2)
    assert first is not second
    assert first.data.dtype == np.dtype(np.int64)
    expected = second.toarray().copy()
    first.data[:] = 0
    np.testing.assert_array_equal(surface.boundary_scipy_copy(2).toarray(), expected)


def test_halfedge_owner_has_no_implicit_array_protocol() -> None:
    surface = HalfedgeSurface.from_permutations(NEXT, TWIN, exterior_faces=EXTERIOR)
    assert not hasattr(surface, "__array__")


def test_topology_facts_are_owner_local_and_genus_requires_connectedness() -> None:
    disk = HalfedgeSurface.from_permutations(NEXT, TWIN, exterior_faces=EXTERIOR)
    assert disk.boundary_component_count == 1
    assert disk.connected_component_count == 1
    assert disk.euler_characteristic == 1
    assert disk.genus == 0

    disconnected = HalfedgeSurface.from_permutations(
        np.array([1, 2, 0, 4, 5, 3, 8, 6, 7, 11, 9, 10]),
        np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]),
        exterior_faces=np.array([6, 9]),
    )
    assert disconnected.connected_component_count == 2
    assert disconnected.euler_characteristic == 2
    assert disconnected.genus is None


def test_explicit_complex_surface_conversion_retains_owners_and_owned_maps() -> None:
    complex_ = (
        Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
        .triangle_manifold()
        .oriented()
    )
    surface, forward = HalfedgeSurface.from_complex(complex_)
    assert forward.source.same_owner(complex_.chain_complex())
    assert forward.target.same_owner(surface.chain_complex())
    assert (surface.vertex_count, surface.edge_count, surface.material_face_count) == (
        3,
        3,
        1,
    )
    permutation, signs = forward.signed_permutation_numpy_copy(1)
    permutation[:] = 0
    signs[:] = 0
    assert np.unique(forward.signed_permutation_numpy_copy(1)[0]).size == 3
    assert set(forward.signed_permutation_numpy_copy(1)[1]) <= {-1, 1}

    rebuilt, reverse = surface.to_complex()
    assert reverse.source.same_owner(surface.chain_complex())
    assert reverse.target.same_owner(rebuilt.chain_complex())
    assert not rebuilt.shares_data_with(complex_)
    np.testing.assert_array_equal(rebuilt.simplices_numpy_copy(2), [[0, 1, 2]])


def test_conversion_limit_is_flat_structured_and_retryable() -> None:
    assert DEFAULT_LAW_LIMIT.retained_logical_bytes == 128 * 1024 * 1024
    assert DEFAULT_LAW_LIMIT.peak_live_logical_bytes == 512 * 1024 * 1024
    assert DEFAULT_LAW_LIMIT.terms == 100_000_000
    complex_ = (
        Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
        .triangle_manifold()
        .oriented()
    )
    limit = ChainLawLimit(retained_logical_bytes=0, peak_live_logical_bytes=0, terms=0)
    with pytest.raises(HalfedgeError) as caught:
        HalfedgeSurface.from_complex(complex_, limit=limit)
    assert caught.value.reason == "resource_limit"
    assert caught.value.details["axis"] == "retained_logical_bytes"
    required = caught.value.details["required"]
    rejected_limit = caught.value.details["limit"]
    assert isinstance(required, int) and isinstance(rejected_limit, int)
    assert required > rejected_limit

    surface, _ = HalfedgeSurface.from_complex(complex_)
    assert surface.material_face_count == 1


def test_conversion_requires_admission_and_rejects_non_simplicial_reverse() -> None:
    unrefined = Complex.from_maximal_simplices(np.array([[0, 1, 2]]))
    with pytest.raises(HalfedgeError) as caught:
        HalfedgeSurface.from_complex(unrefined)
    assert caught.value.reason == "capability_not_admitted"

    quotient = HalfedgeSurface.from_permutations(
        np.array([1, 2, 0, 4, 5, 3]),
        np.array([4, 5, 3, 2, 0, 1]),
    )
    with pytest.raises(HalfedgeError) as caught:
        quotient.to_complex()
    assert caught.value.reason == "conversion_not_simplicial"
