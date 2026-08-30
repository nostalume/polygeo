"""One Complex owner accumulates independently admitted capabilities."""

from __future__ import annotations

import numpy as np
import pytest

from polygeo import Complex, SimplicialError, topological_boundary
from polygeo import _polygeo_native as _core


DISK = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def test_refinement_returns_the_one_complex_and_has_no_state_axes() -> None:
    complex_ = Complex.from_maximal_simplices(DISK)

    assert complex_.triangle_manifold() is complex_
    assert complex_.oriented() is complex_
    assert complex_.with_boundary() is complex_
    assert complex_.connected() is complex_
    assert complex_.shares_data_with(complex_)
    assert not hasattr(complex_, "boundary_state")
    assert not hasattr(complex_, "orientation_state")
    assert not hasattr(complex_, "connectivity_state")
    assert not hasattr(complex_, "topology_state")
    np.testing.assert_array_equal(
        complex_.disk_boundary_vertices_numpy_copy(),
        [0, 1, 2, 3],
    )


def test_require_is_query_only_and_domain_operations_do_not_refine() -> None:
    complex_ = Complex.from_maximal_simplices(DISK)

    with pytest.raises(SimplicialError) as unknown:
        complex_.require_triangle()
    assert unknown.value.reason == "capability_not_admitted"
    assert unknown.value.details == {"capability": "triangle"}

    with pytest.raises(SimplicialError) as domain_unknown:
        topological_boundary(complex_)
    assert domain_unknown.value.reason == "capability_not_admitted"

    complex_.triangle_manifold()
    complex_.require_triangle()
    np.testing.assert_array_equal(
        topological_boundary(complex_).mask(1),
        [True, False, True, True, True],
    )


def test_require_replays_cached_rejection_without_recomputation() -> None:
    complex_ = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 3, 2]], dtype=np.int64)
    )

    with pytest.raises(SimplicialError) as refined:
        complex_.oriented()
    with pytest.raises(SimplicialError) as required:
        complex_.require_oriented()

    assert refined.value.reason == required.value.reason == "orientation"
    assert refined.value.details == required.value.details
    assert refined.value.details is not required.value.details


def test_capabilities_refine_and_query_one_owner() -> None:
    native = Complex.from_maximal_simplices(DISK)

    with pytest.raises(SimplicialError) as unknown:
        native.require_connected()
    assert unknown.value.reason == "capability_not_admitted"
    assert unknown.value.details == {"capability": "connected"}

    assert native.connected() is native
    assert native.require_connected() is None
    assert all(
        not hasattr(_core, name)
        for name in (
            "NativeComplex",
            "NativeSubset",
            "NativeSelection",
            "NativeTopologyError",
        )
    )


def test_refinement_order_is_identity_preserving_and_independent() -> None:
    first = Complex.from_maximal_simplices(DISK)
    second = Complex.from_maximal_simplices(DISK)

    assert first.connected().oriented().triangle_manifold().with_boundary() is first
    assert second.triangle_manifold().with_boundary().oriented().connected() is second
    first.require_connected()
    first.require_oriented()
    first.require_triangle()
    first.require_with_boundary()
    second.require_connected()
    second.require_oriented()
    second.require_triangle()
    second.require_with_boundary()
