from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_array

from polygeo import (
    ORDINARY_FORM,
    Complex,
    DualCochainSpace,
    FieldSemantics,
    Geometry,
    GeometryError,
    LinearMap,
    OperatorError,
    exterior_derivative,
    hodge_star,
)


def _simplex(dimension: int) -> Complex:
    return Complex.from_maximal_simplices(
        np.array([list(range(dimension + 1))], dtype=np.int64)
    )


class AlternateSemantics(FieldSemantics):
    pass


def _geometry(dimension: int) -> tuple[Complex, Geometry]:
    complex_ = _simplex(dimension)
    positions = np.vstack(
        [
            np.zeros((1, dimension), dtype=np.float64),
            np.eye(dimension, dtype=np.float64),
        ]
    )
    return complex_, Geometry.from_positions(complex_, positions)


def test_hodge_star_derives_subordinate_dual_space() -> None:
    complex_, geometry = _geometry(4)
    source = complex_.cochain_space(3)

    star = hodge_star(geometry, source)

    assert star.source is source
    assert isinstance(star.target, DualCochainSpace)
    assert star.target.geometry is geometry
    assert star.target.primal is source
    assert star.target.complex is complex_
    assert star.target.primal_degree == 3
    assert star.target.degree == 1
    assert star.target.size == source.size
    assert star.target.same_space(hodge_star(geometry, source).target)


@pytest.mark.parametrize(
    ("degree", "expected"),
    [
        (0, np.array([3.0, 3.0], dtype=np.float64)),
        (1, np.array([1.0 / 6.0], dtype=np.float64)),
    ],
)
def test_hodge_star_uses_signed_dual_to_primal_ratio(
    degree: int,
    expected: np.ndarray,
) -> None:
    complex_ = _simplex(1)
    geometry = Geometry.from_positions(
        complex_,
        np.array([[0.0], [6.0]], dtype=np.float64),
    )
    source = complex_.cochain_space(degree)

    star = hodge_star(geometry, source)

    np.testing.assert_array_equal(star.matrix().diagonal(), expected)
    semantics = AlternateSemantics()
    value = source.form(np.ones(source.size, dtype=np.float64), semantics)
    result = star.apply(value)
    assert result.space is star.target
    assert result.semantics is semantics
    np.testing.assert_array_equal(result.coefficients(), expected)


@pytest.mark.parametrize(
    "positions",
    [
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.array([[0.0, 0.0], [2.0, 0.0], [0.2, 0.1]], dtype=np.float64),
    ],
)
def test_hodge_star_preserves_finite_signed_and_zero_coefficients(
    positions: np.ndarray,
) -> None:
    complex_ = _simplex(2)
    geometry = Geometry.from_positions(complex_, positions)
    source = complex_.cochain_space(1)
    expected = geometry.dual_measures(1) / geometry.primal_measures(1)

    diagonal = hodge_star(geometry, source).matrix().diagonal()

    np.testing.assert_array_equal(diagonal, expected)
    assert np.any(diagonal <= 0.0)


def test_hodge_star_rejects_foreign_geometry() -> None:
    complex_, _ = _geometry(2)
    foreign_complex, foreign_geometry = _geometry(2)
    source = complex_.cochain_space(1)

    with pytest.raises(OperatorError, match="different complex"):
        hodge_star(foreign_geometry, source)

    assert foreign_complex is foreign_geometry.complex


def test_hodge_star_is_independent_of_global_numpy_error_policy() -> None:
    complex_ = _simplex(1)
    step = float.fromhex("0x0.0000000000002p-1022")
    geometry = Geometry.from_positions(
        complex_,
        np.array([[0.0], [step]], dtype=np.float64),
    )
    source = complex_.cochain_space(0)
    previous = np.seterr(all="raise")
    try:
        diagonal = hodge_star(geometry, source).matrix().diagonal()
    finally:
        np.seterr(**previous)

    np.testing.assert_array_equal(diagonal, [step / 2.0, step / 2.0])


def test_hodge_star_obeys_arbitrary_dimensional_scale_law() -> None:
    dimension = 4
    complex_, geometry = _geometry(dimension)
    scale = 3.0
    scaled = Geometry.from_positions(complex_, scale * geometry.positions)

    for degree in range(dimension + 1):
        source = complex_.cochain_space(degree)
        original = hodge_star(geometry, source).matrix().diagonal()
        actual = hodge_star(scaled, source).matrix().diagonal()
        expected = scale ** (dimension - 2 * degree) * original
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.0)


def test_hodge_star_rejects_unrepresentable_nonzero_ratio() -> None:
    complex_ = _simplex(1)
    step = float.fromhex("0x0.0000000000001p-1022")
    geometry = Geometry.from_positions(
        complex_,
        np.array([[0.0], [step]], dtype=np.float64),
    )

    with pytest.raises(OperatorError, match="not representable"):
        hodge_star(geometry, complex_.cochain_space(1))


def test_hodge_star_closes_unrepresentable_dual_measure_error() -> None:
    complex_ = _simplex(1)
    step = float.fromhex("0x0.0000000000001p-1022")
    geometry = Geometry.from_positions(
        complex_,
        np.array([[0.0], [step]], dtype=np.float64),
    )

    previous = np.seterr(all="raise")
    try:
        with pytest.raises(
            OperatorError,
            match="measures are not representable",
        ) as exc_info:
            hodge_star(geometry, complex_.cochain_space(0))
        assert isinstance(exc_info.value.__cause__, GeometryError)
        assert all(policy == "raise" for policy in np.geterr().values())
    finally:
        np.seterr(**previous)


def test_exterior_derivative_uses_exact_source_and_target_spaces() -> None:
    complex_ = _simplex(4)
    source = complex_.cochain_space(3)
    target = complex_.cochain_space(4)

    derivative = exterior_derivative(source, target)

    assert derivative.source is source
    assert derivative.target is target
    assert derivative.source.complex is complex_
    assert derivative.target.complex is complex_
    assert derivative.matrix().shape == (target.size, source.size)


def test_exterior_derivative_is_oriented_endpoint_difference() -> None:
    complex_ = _simplex(2)
    values = np.array([2.0, 5.0, 11.0], dtype=np.float64)
    source = complex_.cochain_space(0)
    target = complex_.cochain_space(1)
    value = source.form(values, ORDINARY_FORM)
    expected: list[float] = []
    for edge, orientation in zip(
        complex_.simplices(1),
        complex_.orientations(1),
        strict=True,
    ):
        start, stop = int(edge[0]), int(edge[1])
        expected.append(int(orientation) * (values[stop] - values[start]))

    result = exterior_derivative(source, target).apply(value)

    np.testing.assert_array_equal(result.coefficients(), expected)


def test_exterior_derivative_is_nilpotent_above_surface_dimension() -> None:
    complex_ = _simplex(4)
    space_2 = complex_.cochain_space(2)
    space_3 = complex_.cochain_space(3)
    space_4 = complex_.cochain_space(4)
    semantics = AlternateSemantics()
    value = space_2.form(np.arange(space_2.size, dtype=np.float64), semantics)
    derivative_2 = exterior_derivative(space_2, space_3)
    derivative_3 = exterior_derivative(space_3, space_4)

    composed = derivative_3.compose(derivative_2)
    result = composed.apply(value)

    assert composed.source is space_2
    assert composed.target is space_4
    assert result.semantics is semantics
    np.testing.assert_array_equal(result.coefficients(), np.zeros(space_4.size))
    assert composed.matrix().nnz == 0


def test_exterior_derivative_is_nilpotent_through_dimension_eight() -> None:
    for dimension in range(1, 9):
        complex_ = _simplex(dimension)
        for degree in range(dimension - 1):
            source = complex_.cochain_space(degree)
            middle = complex_.cochain_space(degree + 1)
            target = complex_.cochain_space(degree + 2)
            first = exterior_derivative(source, middle)
            second = exterior_derivative(middle, target)

            assert second.compose(first).matrix().nnz == 0


def test_linear_map_composition_matches_sequential_application() -> None:
    complex_ = _simplex(2)
    source = complex_.cochain_space(0)
    middle = complex_.cochain_space(1)
    target = complex_.cochain_space(2)
    before = LinearMap(
        source,
        middle,
        csr_array(
            np.arange(1, middle.size * source.size + 1, dtype=np.float64).reshape(
                middle.size, source.size
            )
        ),
    )
    after = LinearMap(
        middle,
        target,
        csr_array(
            np.arange(1, target.size * middle.size + 1, dtype=np.float64).reshape(
                target.size, middle.size
            )
        ),
    )
    semantics = AlternateSemantics()
    value = source.form(np.arange(source.size, dtype=np.float64), semantics)

    composed = after.compose(before)

    assert composed.source is source
    assert composed.target is target
    assert composed.apply(value).semantics is semantics
    np.testing.assert_array_equal(
        composed.apply(value).coefficients(),
        after.apply(before.apply(value)).coefficients(),
    )
    np.testing.assert_array_equal(
        composed.matrix().toarray(),
        (after.matrix() @ before.matrix()).toarray(),
    )


def test_linear_map_composition_requires_exact_intermediate_space() -> None:
    left = _simplex(2)
    right = _simplex(2)
    before = LinearMap(
        left.cochain_space(0),
        left.cochain_space(1),
        left.boundary_matrix(1).transpose().tocsr(),
    )
    after = LinearMap(
        right.cochain_space(1),
        right.cochain_space(2),
        right.boundary_matrix(2).transpose().tocsr(),
    )

    with pytest.raises(OperatorError, match="intermediate"):
        after.compose(before)


def test_map_rejects_form_from_equal_but_distinct_complex() -> None:
    left = _simplex(3)
    right = _simplex(3)
    source = left.cochain_space(1)
    target = left.cochain_space(2)
    derivative = exterior_derivative(source, target)
    foreign = right.cochain_space(1).form(
        np.zeros(right.simplex_count(1)), ORDINARY_FORM
    )

    with pytest.raises(OperatorError, match="map source"):
        derivative.apply(foreign)


def test_linear_map_rejects_misaligned_or_nonfinite_representation() -> None:
    complex_ = _simplex(2)
    source = complex_.cochain_space(0)
    target = complex_.cochain_space(1)

    with pytest.raises(OperatorError, match="shape"):
        LinearMap(source, target, csr_array((target.size, source.size + 1)))

    nonfinite = csr_array(
        (
            np.array([np.inf]),
            (np.array([0]), np.array([0])),
        ),
        shape=(target.size, source.size),
    )
    with pytest.raises(OperatorError, match="finite"):
        LinearMap(source, target, nonfinite)

    complex_data = csr_array(
        np.ones((target.size, source.size), dtype=np.complex128) * (1.0 + 1.0j)
    )
    with pytest.raises(OperatorError, match="real"):
        LinearMap(source, target, complex_data)

    unsupported_data = csr_array(np.eye(source.size, dtype=np.float64))
    unsupported_data.data = np.full(unsupported_data.nnz, object(), dtype=object)
    with pytest.raises(OperatorError, match="float64"):
        LinearMap(source, source, unsupported_data)

    indptr = np.full(target.size + 1, 2, dtype=np.int64)
    indptr[0] = 0
    duplicate_overflow = csr_array(
        (
            np.array([1e308, 1e308]),
            np.array([0, 0], dtype=np.int64),
            indptr,
        ),
        shape=(target.size, source.size),
    )
    with pytest.raises(OperatorError, match="finite"):
        LinearMap(source, target, duplicate_overflow)

    for invalid_index in (-1, source.size):
        malformed = csr_array(
            (
                np.array([1.0]),
                np.array([invalid_index], dtype=np.int64),
                np.array([0, 1, *([1] * (target.size - 1))], dtype=np.int64),
            ),
            shape=(target.size, source.size),
        )
        with pytest.raises(OperatorError, match="CSR"):
            LinearMap(source, target, malformed)


def test_linear_map_rejects_spaces_from_different_complexes() -> None:
    left = _simplex(2)
    right = _simplex(2)
    source = left.cochain_space(0)
    target = right.cochain_space(1)
    matrix = csr_array((target.size, source.size), dtype=np.float64)

    with pytest.raises(OperatorError, match="different complexes"):
        LinearMap(source, target, matrix)


def test_exterior_derivative_rejects_foreign_target_before_assembly() -> None:
    source_complex = _simplex(1)
    target_complex = _simplex(2)
    source = source_complex.cochain_space(1)
    target = target_complex.cochain_space(2)

    with pytest.raises(OperatorError, match="different complexes"):
        exterior_derivative(source, target)


def test_linear_map_owns_its_sparse_representation() -> None:
    complex_ = _simplex(2)
    space = complex_.cochain_space(0)
    matrix = csr_array(np.eye(space.size, dtype=np.float64))
    map_ = LinearMap(space, space, matrix)
    value = space.form(np.arange(space.size, dtype=np.float64), ORDINARY_FORM)

    matrix.data[:] = 0.0
    exposed = map_.matrix()
    exposed.data[:] = 0.0

    np.testing.assert_array_equal(
        map_.apply(value).coefficients(), value.coefficients()
    )


def test_linear_map_exposes_canonical_sparse_representation() -> None:
    complex_ = _simplex(1)
    space = complex_.cochain_space(0)
    representation = csr_array(
        (
            np.array([1.0, 0.0, 2.0, 3.0]),
            np.array([1, 0, 1, 0], dtype=np.int64),
            np.array([0, 3, 4], dtype=np.int64),
        ),
        shape=(space.size, space.size),
    )

    canonical = LinearMap(space, space, representation).matrix()

    assert canonical.has_canonical_format
    assert canonical.nnz == 2
    np.testing.assert_array_equal(canonical.toarray(), [[0.0, 3.0], [3.0, 0.0]])


def test_linear_map_rejects_nonfinite_application_result() -> None:
    complex_ = _simplex(1)
    space = complex_.cochain_space(0)
    matrix = csr_array(np.eye(space.size, dtype=np.float64) * 1e308)
    map_ = LinearMap(space, space, matrix)
    value = space.form(np.full(space.size, 2.0), ORDINARY_FORM)

    with pytest.raises(OperatorError, match="non-finite"):
        map_.apply(value)


@pytest.mark.parametrize(("source_degree", "target_degree"), [(1, 1), (1, 3), (2, 1)])
def test_exterior_derivative_rejects_nonadjacent_spaces(
    source_degree: int,
    target_degree: int,
) -> None:
    complex_ = _simplex(3)
    source = complex_.cochain_space(source_degree)
    target = complex_.cochain_space(target_degree)

    with pytest.raises(OperatorError, match="adjacent"):
        exterior_derivative(source, target)
