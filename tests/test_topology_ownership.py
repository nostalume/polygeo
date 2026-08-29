"""Topology errors, subsets, selections, and projections retain their owners."""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest

from polygeo import Complex as NativeComplex
from polygeo import SimplexSelection as NativeSelection
from polygeo import SimplexSubset as NativeSubset
from polygeo import SimplicialError as NativeSimplicialError


def _disk() -> NativeComplex:
    return NativeComplex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    )


class _NoScalarGetitem(np.ndarray):
    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, (int, np.integer)):
            raise AssertionError(
                "native ndarray admission must not use scalar __getitem__"
            )
        return super().__getitem__(key)


def _without_scalar_getitem(values: np.ndarray) -> np.ndarray:
    return values.view(_NoScalarGetitem)


def _failure(operation: Callable[[], object]) -> tuple[str, Mapping[str, int | str]]:
    with pytest.raises(NativeSimplicialError) as caught:
        operation()
    assert caught.value.reason is not None
    return caught.value.reason, caught.value.details


def test_all_capabilities_resolve_on_one_owner() -> None:
    owner = _disk()
    owner.codimension_one_regular()
    owner.codimension_one_regular()
    owner.triangle_manifold()
    owner.triangle_manifold()
    owner.oriented()
    owner.oriented()
    owner.connected()
    owner.connected()
    owner.with_boundary()

    owner.require_regular()
    owner.require_triangle()
    owner.require_oriented()
    owner.require_connected()
    owner.require_with_boundary()
    np.testing.assert_array_equal(
        owner.boundary_mask(1), [True, False, True, True, True]
    )


def test_topology_handles_never_expose_pointer_valued_owner_identity() -> None:
    owner = _disk()
    subset = owner.subset(
        (
            np.zeros(4, dtype=np.bool_),
            np.zeros(5, dtype=np.bool_),
            np.zeros(2, dtype=np.bool_),
        )
    )
    handles = (
        owner,
        subset,
        owner.selection(1, np.array([0], dtype=np.int64)),
    )

    assert all(not hasattr(handle, "_debug_owner_id") for handle in handles)


def test_cached_rejection_details_are_stable_and_caller_isolated() -> None:
    owner = NativeComplex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 3, 2]], dtype=np.int64)
    )
    failures: list[NativeSimplicialError] = []
    for _ in range(2):
        with pytest.raises(NativeSimplicialError) as caught:
            owner.oriented()
        failures.append(caught.value)

    assert failures[0].reason == failures[1].reason == "orientation"
    assert failures[0].details == failures[1].details
    assert set(failures[0].details) == {"codimension_one_simplex"}
    assert failures[0].details is not failures[1].details


def test_structured_rejections_preserve_exact_counterexamples() -> None:
    assert _failure(
        lambda: NativeComplex.from_maximal_simplices(
            np.array([[0, -1, 2]], dtype=np.int64)
        )
    ) == ("negative_index", {"value": -1})
    assert _failure(
        lambda: NativeComplex.from_maximal_simplices(
            np.array([[0, 1, 1]], dtype=np.int64)
        )
    ) == ("repeated_vertex", {"vertex": 1})
    assert _failure(
        lambda: NativeComplex.from_maximal_simplices(
            np.array([[0, 1, 2]], dtype=np.int64), vertex_count=2
        )
    ) == ("vertex_extent", {"declared": 2, "required": 3})

    disk = _disk()
    assert _failure(lambda: disk.boundary_matrix(3)) == (
        "degree_outside",
        {"degree": 3},
    )
    assert _failure(
        lambda: NativeComplex.from_maximal_simplices(
            np.array([[0, 1, 2]], dtype=np.int64), vertex_count=4
        ).codimension_one_regular()
    ) == ("not_pure", {"vertex": 3})
    assert _failure(
        lambda: NativeComplex.from_maximal_simplices(
            np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
        ).codimension_one_regular()
    ) == (
        "codimension_one_incidence",
        {"degree": 1, "simplex": 0, "cofaces": 3},
    )
    assert _failure(
        lambda: NativeComplex.from_maximal_simplices(
            np.array([[0, 1], [1, 2]], dtype=np.int64)
        ).triangle_manifold()
    ) == ("triangle_dimension", {"actual_dimension": 1})
    assert _failure(
        lambda: NativeComplex.from_maximal_simplices(
            np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
        ).triangle_manifold()
    ) == ("vertex_link", {"vertex": 0})
    assert _failure(
        lambda: NativeComplex.from_maximal_simplices(
            np.array([[0, 1], [2, 3]], dtype=np.int64)
        ).connected()
    ) == ("disconnected", {"unreachable_vertex": 2})
    disk.codimension_one_regular()
    assert _failure(lambda: disk.without_boundary()) == (
        "boundary_present",
        {"codimension_one_simplex": 0},
    )


def test_independent_proof_failures_are_classified() -> None:
    incoherent = NativeComplex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 3, 2]], dtype=np.int64)
    )
    with pytest.raises(NativeSimplicialError) as orientation:
        incoherent.oriented()
    assert orientation.value.reason == "orientation"

    disconnected = NativeComplex.from_maximal_simplices(
        np.array([[0, 1], [2, 3]], dtype=np.int64)
    )
    with pytest.raises(NativeSimplicialError) as connectivity:
        disconnected.connected()
    assert connectivity.value.reason == "disconnected"

    point = NativeComplex.from_maximal_simplices(np.array([[0]], dtype=np.int64))
    point.codimension_one_regular()
    point.without_boundary()
    with pytest.raises(NativeSimplicialError) as boundary:
        point.with_boundary()
    assert boundary.value.reason == "boundary_absent"


def test_subset_copies_admission_and_projection_and_survives_owner() -> None:
    owner = _disk()
    masks = (
        np.array([True, False, False, False]),
        np.zeros(5, dtype=np.bool_),
        np.zeros(2, dtype=np.bool_),
    )
    subset = owner.subset(masks)
    masks[0][:] = False
    exposed = subset.mask(0)
    exposed[:] = False
    link = subset.link()
    del owner, subset
    gc.collect()

    assert isinstance(link, NativeSubset)
    assert link.same_members(link)
    np.testing.assert_array_equal(link.mask(0), [False, True, True, True])
    np.testing.assert_array_equal(link.mask(1), [False, False, False, True, True])


def test_subset_admission_reads_strided_buffers_without_scalar_python_access() -> None:
    owner = _disk()
    masks = (
        _without_scalar_getitem(np.array([True, False] * 4, dtype=np.bool_)[::2]),
        _without_scalar_getitem(np.zeros(10, dtype=np.bool_)[::2]),
        _without_scalar_getitem(np.zeros(4, dtype=np.bool_)[::2]),
    )

    subset = owner.subset(masks)

    np.testing.assert_array_equal(subset.mask(0), [True, True, True, True])


def test_subset_relations_and_foreign_owner_are_exact() -> None:
    owner = _disk()
    subset = owner.subset(
        (
            np.array([True, False, False, False]),
            np.zeros(5, dtype=np.bool_),
            np.zeros(2, dtype=np.bool_),
        )
    )
    closure = subset.closure()
    star = subset.star()
    assert closure.same_members(closure.closure())
    assert subset.is_pure(0)
    np.testing.assert_array_equal(star.mask(2), [True, True])
    foreign = _disk().subset(
        (
            np.zeros(4, dtype=np.bool_),
            np.zeros(5, dtype=np.bool_),
            np.zeros(2, dtype=np.bool_),
        )
    )
    with pytest.raises(NativeSimplicialError) as caught:
        subset.same_members(foreign)
    assert caught.value.reason == "owner_mismatch"


def test_canonical_boundary_subset_has_explicit_owned_copy() -> None:
    owner = _disk()
    owner.codimension_one_regular()
    boundary = owner.boundary_subset()
    owned = boundary.owned_copy()
    exposed = owned.mask(1)
    exposed[:] = False

    assert boundary.same_members(owned)
    np.testing.assert_array_equal(owned.mask(1), [True, False, True, True, True])
    np.testing.assert_array_equal(boundary.mask(1), owned.mask(1))


def test_selection_is_canonical_copied_and_owner_bound() -> None:
    owner = _disk()
    indices = np.array([0, 2, 4], dtype=np.int64)
    selected = owner.selection(1, indices)
    indices[:] = 1
    exposed = selected.indices()
    exposed[:] = 1
    complement = selected.complement()

    assert isinstance(selected, NativeSelection)
    np.testing.assert_array_equal(selected.indices(), [0, 2, 4])
    np.testing.assert_array_equal(complement.indices(), [1, 3])
    assert selected.same_selection(owner.selection(1, np.array([0, 2, 4])))
    with pytest.raises(NativeSimplicialError) as caught:
        selected.same_selection(_disk().selection(1, np.array([0, 2, 4])))
    assert caught.value.reason == "owner_mismatch"


def test_selection_admission_reads_strided_buffer_without_scalar_python_access() -> (
    None
):
    owner = _disk()
    indices = _without_scalar_getitem(np.array([0, 99, 2, 99, 4], dtype=np.int64)[::2])

    selected = owner.selection(1, indices)

    np.testing.assert_array_equal(selected.indices(), [0, 2, 4])


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
def test_selection_admission_accepts_every_fixed_width_integer_dtype(
    dtype: np.dtype,
) -> None:
    selected = _disk().selection(1, np.array([0, 2, 4], dtype=dtype))

    np.testing.assert_array_equal(selected.indices(), [0, 2, 4])


def test_selection_admission_preserves_shape_domain_and_endian_errors() -> None:
    owner = _disk()
    non_native = owner.selection(1, np.array([0, 2, 4], dtype=">i8"))
    np.testing.assert_array_equal(non_native.indices(), [0, 2, 4])

    for invalid, reason in (
        (np.array([[0, 2]], dtype=np.int64), "selection_shape"),
        (np.array([-1], dtype=np.int64), "selection_index_outside"),
        (np.array([0, 0], dtype=np.int64), "selection_not_strict"),
        (np.array([5], dtype=np.uint64), "selection_index_outside"),
    ):
        with pytest.raises(NativeSimplicialError) as caught:
            owner.selection(1, invalid)
        assert caught.value.reason == reason


def test_subset_admission_preserves_mask_shape_classification() -> None:
    owner = _disk()
    valid = (
        np.zeros(4, dtype=np.bool_),
        np.zeros(5, dtype=np.bool_),
        np.zeros(2, dtype=np.bool_),
    )
    for invalid in (valid[:-1], (*valid, np.zeros(1, dtype=np.bool_))):
        with pytest.raises(NativeSimplicialError) as caught:
            owner.subset(invalid)
        assert caught.value.reason == "mask_shape"
