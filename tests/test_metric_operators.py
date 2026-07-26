from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from scipy.sparse import csr_array

from polygeo import (
    ORDINARY_FORM,
    Complex,
    FieldSemantics,
    Geometry,
    LinearMap,
    OperatorError,
    codifferential,
    exterior_derivative,
    hodge_laplacian,
    weighted_pairing,
)


class AlternateSemantics(FieldSemantics):
    pass


def _simplex(dimension: int) -> tuple[Complex, Geometry]:
    complex_ = Complex.from_maximal_simplices(
        np.array([list(range(dimension + 1))], dtype=np.int64)
    )
    positions = np.vstack(
        [
            np.zeros((1, dimension), dtype=np.float64),
            np.eye(dimension, dtype=np.float64),
        ]
    )
    return complex_, Geometry.from_positions(complex_, positions)


def _segment(length: float = 6.0) -> tuple[Complex, Geometry]:
    complex_ = Complex.from_maximal_simplices(np.array([[0, 1]], dtype=np.int64))
    geometry = Geometry.from_positions(
        complex_,
        np.array([[0.0], [length]], dtype=np.float64),
    )
    return complex_, geometry


def test_weighted_pairing_uses_signed_hodge_weights_directly() -> None:
    complex_, geometry = _segment()
    C0 = complex_.cochain_space(0)
    left = C0.form(np.array([1.0, 2.0]), ORDINARY_FORM)
    right = C0.form(np.array([3.0, 4.0]), ORDINARY_FORM)

    actual = weighted_pairing(geometry, left, right)

    assert actual == 33.0
    assert weighted_pairing(geometry, left, right) == weighted_pairing(
        geometry, right, left
    )


def test_weighted_pairing_rejects_foreign_spaces_and_nonfinite_output() -> None:
    complex_, geometry = _segment()
    foreign, foreign_geometry = _segment()
    left = complex_.cochain_space(0).form(np.ones(2), ORDINARY_FORM)
    foreign_form = foreign.cochain_space(0).form(np.ones(2), ORDINARY_FORM)

    with pytest.raises(OperatorError, match="same cochain space"):
        weighted_pairing(geometry, left, foreign_form)
    with pytest.raises(OperatorError, match="different complex"):
        weighted_pairing(foreign_geometry, left, left)

    huge = complex_.cochain_space(1).form(
        np.array([np.finfo(np.float64).max]), ORDINARY_FORM
    )
    with pytest.raises(OperatorError, match="not representable"):
        weighted_pairing(geometry, huge, huge)


def test_weighted_pairing_avoids_representable_product_underflow() -> None:
    complex_, geometry = _segment(1e200)
    C1 = complex_.cochain_space(1)
    left = C1.form(np.array([1e-200]), ORDINARY_FORM)
    right = C1.form(np.array([1e300]), AlternateSemantics())

    actual = weighted_pairing(geometry, left, right)

    assert actual == pytest.approx(1e-100, rel=2e-15, abs=0.0)


def test_weighted_pairing_avoids_cancelling_product_overflow() -> None:
    complex_ = Complex.from_maximal_simplices(np.array([[0], [1]], dtype=np.int64))
    geometry = Geometry.from_positions(
        complex_, np.array([[0.0], [1.0]], dtype=np.float64)
    )
    C0 = complex_.cochain_space(0)
    maximum = np.finfo(np.float64).max
    left = C0.form(np.array([maximum, maximum]), ORDINARY_FORM)
    right = C0.form(np.array([2.0, -2.0]), ORDINARY_FORM)

    assert weighted_pairing(geometry, left, right) == 0.0


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_weighted_pairing_recomputes_subnormal_intermediate(sign: float) -> None:
    complex_, geometry = _segment(2.0)
    C1 = complex_.cochain_space(1)
    smallest = float.fromhex("0x0.0000000000001p-1022")
    left_value = sign * 3.0 * smallest
    right_value = 1e308
    left = C1.form(np.array([left_value]), ORDINARY_FORM)
    right = C1.form(np.array([right_value]), ORDINARY_FORM)
    expected = float(Fraction(left_value) * Fraction(0.5) * Fraction(right_value))

    assert weighted_pairing(geometry, left, right) == expected


def test_codifferential_matches_hand_derived_interval_oracle() -> None:
    length = 6.0
    complex_, geometry = _segment(length)
    C0 = complex_.cochain_space(0)
    C1 = complex_.cochain_space(1)
    derivative = exterior_derivative(C0, C1)

    delta = codifferential(geometry, derivative)

    expected = (2.0 / length**2) * np.array([[-1.0], [1.0]])
    np.testing.assert_array_equal(delta.matrix().toarray(), expected)
    assert delta.source is C1
    assert delta.target is C0


def test_codifferential_satisfies_weighted_adjoint_identity() -> None:
    complex_, geometry = _simplex(4)
    C2 = complex_.cochain_space(2)
    C3 = complex_.cochain_space(3)
    derivative = exterior_derivative(C2, C3)
    delta = codifferential(geometry, derivative)
    alpha = C2.form(np.linspace(-0.5, 1.5, C2.size), ORDINARY_FORM)
    beta = C3.form(np.linspace(2.0, -1.0, C3.size), ORDINARY_FORM)

    left = weighted_pairing(geometry, derivative.apply(alpha), beta)
    right = weighted_pairing(geometry, alpha, delta.apply(beta))

    assert left == pytest.approx(right, rel=2e-14, abs=2e-14)


def test_codifferential_rejects_nonadjacent_map_and_foreign_geometry() -> None:
    complex_, geometry = _simplex(2)
    C0 = complex_.cochain_space(0)
    C2 = complex_.cochain_space(2)
    nonadjacent = LinearMap(C0, C2, csr_array((C2.size, C0.size)))

    with pytest.raises(OperatorError, match="adjacent degrees"):
        codifferential(geometry, nonadjacent)

    foreign, foreign_geometry = _simplex(2)
    derivative = exterior_derivative(C0, complex_.cochain_space(1))
    with pytest.raises(OperatorError, match="different complex"):
        codifferential(foreign_geometry, derivative)
    assert foreign is foreign_geometry.complex


def test_codifferential_avoids_representable_intermediate_overflow() -> None:
    complex_ = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
    geometry = Geometry.from_positions(
        complex_,
        10.0 * np.array([[0.0, 0.0], [2.0, 0.0], [0.2, 0.1]]),
    )
    C0 = complex_.cochain_space(0)
    C1 = complex_.cochain_space(1)
    derivative = LinearMap(
        C0,
        C1,
        csr_array(
            (
                np.array([5e307]),
                (np.array([1]), np.array([2])),
            ),
            shape=(C1.size, C0.size),
        ),
    )

    actual = codifferential(geometry, derivative).matrix().toarray()[2, 1]
    expected = (5e307 / 92.5) * 9.0

    assert actual == pytest.approx(expected, rel=2e-15)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_codifferential_recomputes_subnormal_quotient(sign: float) -> None:
    complex_ = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
    geometry = Geometry.from_positions(
        complex_,
        1e-153 * np.array([[0.0, 0.0], [2.0, 0.0], [0.2, 0.1]]),
    )
    C1 = complex_.cochain_space(1)
    C2 = complex_.cochain_space(2)
    smallest = float.fromhex("0x0.0000000000001p-1022")
    coefficient = sign * 15.0 * smallest
    derivative = LinearMap(
        C1,
        C2,
        csr_array(
            (
                np.array([coefficient]),
                (np.array([0]), np.array([1])),
            ),
            shape=(C2.size, C1.size),
        ),
    )
    previous = (geometry.dual_measures(1) / geometry.primal_measures(1))[1]
    current = (geometry.dual_measures(2) / geometry.primal_measures(2))[0]
    expected = float(
        Fraction(coefficient) * Fraction(float(current)) / Fraction(float(previous))
    )

    actual = codifferential(geometry, derivative).matrix().toarray()[1, 0]

    assert actual == expected


def test_hodge_laplacian_matches_interval_oracle_and_constant_law() -> None:
    length = 6.0
    complex_, geometry = _segment(length)
    C0 = complex_.cochain_space(0)

    laplacian = hodge_laplacian(geometry, C0)

    expected = (2.0 / length**2) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_array_equal(laplacian.matrix().toarray(), expected)
    constant = C0.form(np.ones(C0.size), ORDINARY_FORM)
    np.testing.assert_array_equal(
        laplacian.apply(constant).coefficients(), np.zeros(C0.size)
    )


def test_hodge_laplacian_handles_terminal_and_zero_dimensions() -> None:
    segment, segment_geometry = _segment()
    C1 = segment.cochain_space(1)
    top = hodge_laplacian(segment_geometry, C1)
    np.testing.assert_array_equal(top.matrix().toarray(), [[1.0 / 9.0]])

    point, point_geometry = _simplex(0)
    C0 = point.cochain_space(0)
    zero = hodge_laplacian(point_geometry, C0)
    assert zero.source is C0
    assert zero.target is C0
    assert zero.matrix().shape == (1, 1)
    assert zero.matrix().nnz == 0


def test_hodge_laplacian_is_weighted_self_adjoint_in_dimension_four() -> None:
    dimension = 4
    complex_, geometry = _simplex(dimension)

    for degree in range(dimension + 1):
        space = complex_.cochain_space(degree)
        matrix = hodge_laplacian(geometry, space).matrix().toarray()
        weights = geometry.dual_measures(degree) / geometry.primal_measures(degree)
        weighted = weights[:, np.newaxis] * matrix
        np.testing.assert_allclose(weighted, weighted.T, rtol=2e-13, atol=2e-13)


def test_zero_hodge_is_allowed_until_reciprocal_is_required() -> None:
    complex_ = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
    geometry = Geometry.from_positions(
        complex_,
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
    )
    C0 = complex_.cochain_space(0)
    C1 = complex_.cochain_space(1)
    C2 = complex_.cochain_space(2)

    codifferential(geometry, exterior_derivative(C0, C1))
    hodge_laplacian(geometry, C0)

    with pytest.raises(OperatorError, match="zero reciprocal Hodge weight"):
        codifferential(geometry, exterior_derivative(C1, C2))
    with pytest.raises(OperatorError, match="zero reciprocal Hodge weight"):
        hodge_laplacian(geometry, C1)
    with pytest.raises(OperatorError, match="zero reciprocal Hodge weight"):
        hodge_laplacian(geometry, C2)


def test_metric_operators_restore_strict_numpy_policy() -> None:
    complex_, geometry = _segment()
    C0 = complex_.cochain_space(0)
    C1 = complex_.cochain_space(1)
    value = C0.form(np.ones(C0.size), ORDINARY_FORM)
    previous = np.seterr(all="raise")
    try:
        weighted_pairing(geometry, value, value)
        codifferential(geometry, exterior_derivative(C0, C1))
        hodge_laplacian(geometry, C0)
        assert all(policy == "raise" for policy in np.geterr().values())
    finally:
        np.seterr(**previous)
