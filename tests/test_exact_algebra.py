"""Exact algebra preserves coefficients, variance, maps, and representations."""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from polygeo import (
    BigIntEncoding,
    ChainError,
    CsrRepresentation,
    IntegralChainComplex,
    IntegerCsrParts,
    QQ,
    RationalCsrParts,
    ReducedFractionEncoding,
)
from polygeo import _polygeo_native as native


FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def integral_complex() -> IntegralChainComplex:
    return native.Complex.from_maximal_simplices(FACES).chain_complex()


def test_exact_integer_and_rational_values_preserve_variance_and_large_scalars() -> (
    None
):
    integers = integral_complex()
    huge = 1 << 200
    chains = integers[1]
    cochains = integers.dual()[1]
    chain = chains.element([(3, -2), (0, huge), (3, 7), (2, 0)])
    cochain = cochains.element({0: 3, 3: -1})

    assert (chains.degree, chains.dimension) == (1, 5)
    assert chain.to_python_copy() == ((0, 3), (huge, 5))
    assert cochain.evaluate(chain) == 3 * huge - 5
    assert not hasattr(chain, "dual")
    assert not hasattr(chain, "__array__")

    rationals = integers.over(QQ)
    rational = rationals[1].element({0: Fraction(6, 10), 3: Fraction(-8, 12)})
    assert rational.to_python_copy() == (
        (0, 3),
        (Fraction(3, 5), Fraction(-2, 3)),
    )
    assert rationals.coefficient_system == "Q"


def test_exact_cup_product_preserves_coefficients_owner_and_domain_errors() -> None:
    integers = integral_complex()
    cochains = integers.dual()
    alpha = cochains[1].element({0: 2})
    beta = cochains[1].element({3: 3})

    assert alpha.cup(beta).to_python_copy() == ((0,), (6,))

    foreign = integral_complex().dual()[1].element({0: 1})
    with pytest.raises(ChainError) as wrong_owner:
        alpha.cup(foreign)
    assert wrong_owner.value.reason == "space_mismatch"

    rational = integers.over(QQ).dual()[1].element({0: Fraction(1)})
    with pytest.raises(ChainError) as wrong_system:
        alpha.cup(cast(Any, rational))
    assert wrong_system.value.reason == "space_mismatch"

    chain = integers[1].element({0: 1})
    with pytest.raises(ChainError) as wrong_variance:
        alpha.cup(cast(Any, chain))
    assert wrong_variance.value.reason == "space_mismatch"

    triangle = (
        native.Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
        .triangle_manifold()
        .oriented()
    )
    halfedge, _ = native.HalfedgeSurface.from_complex(triangle)
    halfedge_cochain = halfedge.chain_complex().dual()[1].element({0: 1})
    with pytest.raises(ChainError) as wrong_domain:
        halfedge_cochain.cup(halfedge_cochain)
    assert wrong_domain.value.reason == "not_simplicial"
    assert wrong_domain.value.details == {}


def test_exact_transport_preserves_signed_multilimb_integers_and_fractions() -> None:
    integers = integral_complex()
    positive = (1 << 4097) + 12345
    negative = -((1 << 4096) + 257)
    chain = integers[1].element({0: positive, 3: negative})

    assert chain.to_python_copy() == ((0, 3), (positive, negative))

    rational = integers.over(QQ)[1].element({0: Fraction(negative, positive)})
    assert rational.to_python_copy() == ((0,), (Fraction(negative, positive),))


def test_maps_reject_wrong_variance_owner_and_coefficient_system() -> None:
    integers = integral_complex()
    chain = integers[1].element({0: 2})
    cochain = integers.dual()[1].element({0: 2})
    boundary = integers.boundary(1)

    assert (boundary.source.degree, boundary.target.degree) == (1, 0)
    assert boundary.apply(chain).to_python_copy() == ((0, 1), (-2, 2))
    with pytest.raises(ChainError) as wrong_variance:
        boundary.apply(cast(Any, cochain))
    assert wrong_variance.value.args[0] == "space_mismatch"

    foreign = integral_complex()[1].element({0: 2})
    with pytest.raises(ChainError) as wrong_owner:
        boundary.apply(foreign)
    assert wrong_owner.value.args[0] == "space_mismatch"

    rational = integers.over(QQ)[1].element({0: Fraction(2)})
    with pytest.raises(ChainError) as wrong_system:
        boundary.apply(cast(Any, rational))
    assert wrong_system.value.args[0] == "space_mismatch"

    square = integers.boundary(0).compose(boundary)
    assert square.apply(chain).to_python_copy() == ((), ())
    rational_boundary = boundary.over(QQ)
    assert rational_boundary.apply(rational).to_python_copy() == (
        (0, 1),
        (Fraction(-2), Fraction(2)),
    )


def test_csr_has_distinct_exact_projections_and_checked_int64_copy() -> None:
    integers = integral_complex()
    boundary = integers.boundary(2)
    estimate = CsrRepresentation.estimate(boundary, BigIntEncoding)
    representation = CsrRepresentation.build(
        boundary, BigIntEncoding, estimate.as_limit()
    )

    parts = representation.to_python_copy()
    assert parts.shape == (5, 2)
    assert parts.coefficients == (1, -1, 1, -1, 1, 1)
    with pytest.raises(ValueError):
        parts.column_indices[0] = 99
    fresh = representation.to_python_copy()
    assert int(fresh.column_indices[0]) == int(parts.column_indices[0])
    assert not np.shares_memory(fresh.column_indices, parts.column_indices)

    scipy = representation.to_scipy_int64_copy()
    assert (
        scipy.data.dtype
        == scipy.indices.dtype
        == scipy.indptr.dtype
        == np.dtype(np.int64)
    )
    assert scipy.shape == (5, 2)

    rational_map = integers.over(QQ).boundary(2)
    rational_estimate = CsrRepresentation.estimate(
        rational_map, ReducedFractionEncoding
    )
    rational_representation = CsrRepresentation.build(
        rational_map, ReducedFractionEncoding, rational_estimate.as_limit()
    )
    rational_parts = rational_representation.to_python_copy()
    assert isinstance(rational_parts, RationalCsrParts)
    assert rational_parts.numerators == (1, -1, 1, -1, 1, 1)
    assert rational_parts.denominators == (1,) * 6
    with pytest.raises(ChainError) as rejected:
        rational_representation.to_scipy_int64_copy()
    assert rejected.value.args[0] == "coefficient_system"


def test_csr_admission_reports_axis_required_limit_and_phase() -> None:
    boundary = integral_complex().boundary(2)
    estimate = CsrRepresentation.estimate(boundary, BigIntEncoding)
    assert estimate.nnz_bound > 0
    assert estimate.scratch_entries_bound == 0
    assert estimate.scalar_steps_bound >= estimate.nnz_bound
    assert (
        estimate.peak_live_logical_bytes_bound > estimate.retained_logical_bytes_bound
    )
    assert not estimate.canonicalization_required
    limit = estimate.as_limit().replace(
        peak_live_logical_bytes=estimate.peak_live_logical_bytes_bound - 1
    )

    with pytest.raises(ChainError) as invalid:
        estimate.as_limit().replace(
            retained_logical_bytes=estimate.peak_live_logical_bytes_bound + 1
        )
    assert invalid.value.reason == "limit"

    with pytest.raises(ChainError) as caught:
        CsrRepresentation.build(boundary, BigIntEncoding, limit)

    reason, _, details = caught.value.args
    assert reason == "resource_limit"
    assert details == {
        "axis": "peak_live_logical_bytes",
        "required": estimate.peak_live_logical_bytes_bound,
        "limit": limit.peak_live_logical_bytes,
        "phase": "estimate",
    }


@pytest.mark.parametrize(
    ("entries", "reason"),
    [
        ({99: 1}, "basis_index_outside"),
        ([(0, 1, 2)], "coordinate_shape"),
        ({0: object()}, "coefficient_value"),
    ],
)
def test_coordinate_admission_is_classified(entries: object, reason: str) -> None:
    space = integral_complex()[1]
    with pytest.raises(ChainError) as caught:
        space.element(cast(Any, entries))
    assert caught.value.args[0] == reason


def test_zero_denominator_is_rejected_before_fraction_construction() -> None:
    space = integral_complex().over(QQ)[1]
    hostile = SimpleNamespace(numerator=1, denominator=0)
    with pytest.raises(ChainError) as caught:
        space.element(cast(Any, {0: hostile}))
    assert caught.value.args[0] == "coefficient_value"


def test_exact_types_and_explicit_scipy_copy_remain_distinct() -> None:
    integers = native.Complex.from_maximal_simplices(FACES).chain_complex()
    chain = integers[1].element({0: 1 << 130})
    assert chain.to_python_copy() == ((0,), (1 << 130,))
    with pytest.raises(ChainError) as rejected:
        integers[1].element({99: 1})
    assert rejected.value.reason == "basis_index_outside"
    assert rejected.value.details == {"index": 99, "bound": 5}

    boundary = integers.boundary(1)
    estimate = CsrRepresentation.estimate(boundary, BigIntEncoding)
    representation = CsrRepresentation.build(
        boundary, BigIntEncoding, estimate.as_limit()
    )
    scipy_copy = representation.to_scipy_int64_copy()
    scipy_copy.data[:] = 0
    integer_parts = representation.to_python_copy()
    assert isinstance(integer_parts, IntegerCsrParts)
    assert integer_parts.coefficients != (0,) * scipy_copy.nnz
    with pytest.raises(ValueError):
        integer_parts.column_indices[0] = 99
    assert not hasattr(representation, "__array__")

    rationals = integers.over(QQ)
    rational_map = rationals.boundary(1)
    rational_estimate = CsrRepresentation.estimate(
        rational_map, ReducedFractionEncoding
    )
    rational_representation = CsrRepresentation.build(
        rational_map, ReducedFractionEncoding, rational_estimate.as_limit()
    )
    rational_parts = rational_representation.to_python_copy()
    assert isinstance(rational_parts, RationalCsrParts)
    assert rational_parts.denominators == (1,) * len(rational_parts.numerators)
