from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from fractions import Fraction
from typing import Any, cast

import numpy as np
import polygeo.algorithms as algorithms_module
import pytest
from scipy.sparse import csr_array

from polygeo import (
    ORDINARY_FORM,
    AlgorithmError,
    BasisCoordinates,
    Certified,
    CochainSubspace,
    Complex,
    ConditionEvidence,
    FieldSemantics,
    Geometry,
    HodgeComponents,
    HodgeEvidence,
    MeanZeroProblem,
    OperatorError,
    PositiveHodgeMetric,
    ResidualEvidence,
    assemble_poisson,
    eliminate_dirichlet,
    hodge_decomposition,
    hodge_laplacian,
    impose_mean_zero,
    prepare_direct,
    prepare_least_squares,
    real_homology_basis,
)
from polygeo.operators import _hodge_laplacian_from_weights


def test_weighted_gauge_shift_uses_one_final_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def tracked_fraction(*args: int | float) -> Fraction:
        nonlocal calls
        calls += 1
        return Fraction(*args)

    monkeypatch.setattr(algorithms_module, "Fraction", tracked_fraction)
    weights = np.array([2.0**-1000, 1.0, 2.0**900], dtype=np.float64)
    values = np.array([2.0**900, -3.0, 2.0**-1000], dtype=np.float64)
    result = algorithms_module._subtract_weighted_mean(weights, values)
    exact = sum(
        (
            Fraction(float(w)) * Fraction(float(v))
            for w, v in zip(weights, values, strict=True)
        ),
        start=Fraction(),
    ) / sum((Fraction(float(w)) for w in weights), start=Fraction())

    np.testing.assert_array_equal(result, values - float(exact))
    assert calls == 1

    with pytest.raises(ZeroDivisionError, match=r"Fraction\(1, 0\)"):
        algorithms_module._subtract_weighted_mean(
            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        )


class AlternateSemantics(FieldSemantics):
    pass


def _cycle(size: int) -> Complex:
    edges = np.array(
        [(index, (index + 1) % size) for index in range(size)], dtype=np.int64
    )
    return Complex.from_maximal_simplices(edges)


@pytest.mark.parametrize("dimension", range(5))
def test_real_homology_basis_obeys_sphere_laws_through_dimension_four(
    dimension: int,
) -> None:
    complex_ = _closed_simplex_boundary(dimension).complex
    for degree in range(dimension + 1):
        basis = real_homology_basis(complex_, degree)
        expected = 1 if degree == 0 or (dimension > 0 and degree == dimension) else 0
        cycles = basis.cycle_coefficients()

        assert basis.complex is complex_
        assert basis.degree == degree
        assert basis.dimension == expected
        assert isinstance(cycles, csr_array)
        assert cycles.shape == (complex_.simplex_count(degree), expected)
        if degree:
            np.testing.assert_array_equal(
                complex_.boundary_matrix(degree) @ cycles.toarray(),
                np.zeros((complex_.simplex_count(degree - 1), expected)),
            )


def test_real_homology_periods_bind_exact_basis_and_preserve_ownership() -> None:
    complex_ = _cycle(7)
    basis = real_homology_basis(complex_, 1)
    coefficients = np.arange(1.0, 8.0)
    form = complex_.cochain_space(1).form(coefficients, ORDINARY_FORM)

    periods = basis.periods(form)
    expected = basis.cycle_coefficients().T @ coefficients

    assert isinstance(periods, BasisCoordinates)
    assert periods.basis is basis
    assert periods.values == tuple(float(value) for value in expected)
    first = basis.cycle_coefficients()
    first.data[:] = 99.0
    assert not np.array_equal(first.toarray(), basis.cycle_coefficients().toarray())
    assert np.array_equal(form.coefficients(), coefficients)
    with pytest.raises(AttributeError, match="immutable"):
        setattr(basis, "_degree", 0)


def test_real_homology_is_deterministic_and_torsion_is_not_real_homology() -> None:
    first = real_homology_basis(_cycle(8), 1).cycle_coefficients()
    second = real_homology_basis(_cycle(8), 1).cycle_coefficients()
    np.testing.assert_array_equal(first.toarray(), second.toarray())
    assert first.toarray()[np.flatnonzero(first.toarray())[0]] > 0.0

    projective_plane = Complex.from_maximal_simplices(
        np.array(
            [
                (1, 2, 3),
                (1, 2, 6),
                (1, 3, 5),
                (1, 4, 5),
                (1, 4, 6),
                (2, 3, 4),
                (2, 4, 5),
                (2, 5, 6),
                (3, 4, 6),
                (3, 5, 6),
            ],
            dtype=np.int64,
        )
        - 1
    )
    assert real_homology_basis(projective_plane, 1).dimension == 0


def test_real_homology_rejects_invalid_domain_periods_and_resource_exhaustion() -> None:
    complex_ = _cycle(5)
    basis = real_homology_basis(complex_, 1)
    foreign = _cycle(5).cochain_space(1).form(np.ones(5), ORDINARY_FORM)
    alternate = complex_.cochain_space(1).form(np.ones(5), AlternateSemantics())

    with pytest.raises(AlgorithmError, match="degree"):
        real_homology_basis(complex_, -1)
    with pytest.raises(AlgorithmError, match="different complex"):
        basis.periods(foreign)
    with pytest.raises(AlgorithmError, match="ordinary"):
        basis.periods(cast(Any, alternate))
    with pytest.raises(AlgorithmError, match="resource limit"):
        real_homology_basis(_cycle(500), 1)
    disconnected = Complex.from_maximal_simplices(
        np.arange(500, dtype=np.int64)[:, None]
    )
    with pytest.raises(AlgorithmError, match="resource limit"):
        real_homology_basis(disconnected, 0)


def test_exact_workspace_couples_live_cells_and_coefficient_growth() -> None:
    algorithms_module._require_exact_workspace(60_000, 60_000, 1)
    with pytest.raises(AlgorithmError, match="resource limit"):
        algorithms_module._require_exact_workspace(60_000, 60_000, 4_096)
    admitted, pivots = algorithms_module._exact_rref(
        np.array([[1 << 4_095]], dtype=object), other_cells=0
    )
    assert admitted == [[Fraction(1)]]
    assert pivots == [0]
    with pytest.raises(AlgorithmError, match="resource limit"):
        algorithms_module._exact_rref(
            np.array([[1 << 4_096]], dtype=object), other_cells=0
        )


def test_exact_cycle_binary64_conversion_accepts_only_round_trippable_integers() -> (
    None
):
    assert algorithms_module._retain_cycle([1 << 60]) == ((0,), (1 << 60,))
    with pytest.raises(AlgorithmError, match="not representable"):
        algorithms_module._retain_cycle([(1 << 53) + 1])


def test_basis_coordinates_validate_dimension_and_finiteness() -> None:
    basis = real_homology_basis(_cycle(5), 1)
    with pytest.raises(AlgorithmError, match="dimension"):
        BasisCoordinates(basis, ())
    with pytest.raises(AlgorithmError, match="finite"):
        BasisCoordinates(basis, (math.inf,))


def _regular_simplex(dimension: int, *, scale: float = 1.0) -> Geometry:
    complex_ = Complex.from_maximal_simplices(
        np.arange(dimension + 1, dtype=np.int64)[None, :]
    )
    if dimension == 0:
        positions = np.empty((1, 0), dtype=np.float64)
    else:
        eye = np.eye(dimension + 1, dtype=np.float64)
        positions = scale * (eye - np.mean(eye, axis=0, keepdims=True))
    return Geometry.from_positions(complex_, positions)


def _shared_edge_metric(first_height: float, second_height: float) -> Geometry:
    complex_ = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
    )
    return Geometry.from_positions(
        complex_,
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, first_height],
                [0.5, -second_height],
            ],
            dtype=np.float64,
        ),
    )


def _segment(length: float = 6.0) -> Geometry:
    complex_ = Complex.from_maximal_simplices(np.array([[0, 1]], dtype=np.int64))
    return Geometry.from_positions(
        complex_, np.array([[0.0], [length]], dtype=np.float64)
    )


def _closed_simplex_boundary(dimension: int) -> Geometry:
    if dimension == 0:
        raw = Complex.from_maximal_simplices(np.array([[0]], dtype=np.int64))
        positions = np.empty((1, 0), dtype=np.float64)
    else:
        vertex_count = dimension + 2
        facets = np.array(
            [
                [vertex for vertex in range(vertex_count) if vertex != omitted]
                for omitted in range(vertex_count)
            ],
            dtype=np.int64,
        )
        raw = Complex.from_maximal_simplices(facets)
        eye = np.eye(vertex_count, dtype=np.float64)
        positions = eye - np.mean(eye, axis=0, keepdims=True)
    closed = raw.codimension_one_regular().without_boundary().connected()
    return Geometry.from_positions(closed, positions)


@pytest.mark.parametrize(
    ("dimension", "degree"),
    [(dimension, degree) for dimension in range(5) for degree in range(dimension + 1)],
)
def test_hodge_decomposition_certifies_all_laws_through_dimension_four(
    dimension: int,
    degree: int,
) -> None:
    geometry = _regular_simplex(dimension)
    metric = PositiveHodgeMetric(geometry)
    complex_ = geometry.complex
    space = complex_.cochain_space(degree)
    form = space.form(np.linspace(-1.0, 2.0, space.size), ORDINARY_FORM)
    calls = 0

    def prepare(operator):
        nonlocal calls
        calls += 1
        assert operator.target is space
        return prepare_least_squares(operator)

    certified = hodge_decomposition(metric, form, prepare)
    components = certified.output
    evidence = certified.evidence
    component_forms = (components.exact, components.coexact, components.harmonic)
    exact, coexact, harmonic = (
        component.coefficients() for component in component_forms
    )
    weights = metric.weights(degree)
    expected_calls = int(degree > 0) + int(degree < dimension)

    assert isinstance(certified, Certified)
    assert isinstance(components, HodgeComponents)
    assert isinstance(evidence, HodgeEvidence)
    assert calls == expected_calls
    assert all(component.space is space for component in component_forms)
    assert all(component.semantics is ORDINARY_FORM for component in component_forms)
    np.testing.assert_allclose(exact + coexact + harmonic, form.coefficients())
    orthogonal = float(np.dot(weights * exact, coexact))
    orthogonal_scale = float(np.sum(np.abs(weights * exact * coexact)))
    assert abs(orthogonal) <= np.sqrt(np.finfo(np.float64).eps) * max(
        orthogonal_scale, 1.0
    )
    if degree < dimension:
        derivative = complex_.boundary_matrix(degree + 1).transpose()
        np.testing.assert_allclose(derivative @ harmonic, 0.0, atol=1.0e-12)
    if degree > 0:
        boundary = complex_.boundary_matrix(degree)
        np.testing.assert_allclose(boundary @ (weights * harmonic), 0.0, atol=1.0e-12)
    assert evidence.reconstruction.residual_norm <= evidence.reconstruction.limit
    assert evidence.orthogonality.residual_norm <= evidence.orthogonality.limit
    assert isinstance(evidence.exact_condition, ConditionEvidence)
    assert isinstance(evidence.coexact_condition, ConditionEvidence)


def test_hodge_decomposition_endpoint_uses_no_backend_and_owns_components() -> None:
    geometry = _regular_simplex(0)
    metric = PositiveHodgeMetric(geometry)
    space = geometry.complex.cochain_space(0)
    form = space.form(np.array([3.0]), ORDINARY_FORM)

    def forbidden(operator):
        raise AssertionError(operator)

    result = hodge_decomposition(metric, form, forbidden)
    np.testing.assert_array_equal(result.output.exact.coefficients(), [0.0])
    np.testing.assert_array_equal(result.output.coexact.coefficients(), [0.0])
    np.testing.assert_array_equal(result.output.harmonic.coefficients(), [3.0])
    returned = result.output.harmonic.coefficients()
    returned[:] = 99.0
    np.testing.assert_array_equal(result.output.harmonic.coefficients(), [3.0])


@pytest.mark.parametrize("dimension", range(1, 5))
def test_hodge_decomposition_retains_nontrivial_top_harmonic_period(
    dimension: int,
) -> None:
    geometry = _closed_simplex_boundary(dimension)
    metric = PositiveHodgeMetric(geometry)
    complex_ = geometry.complex
    space = complex_.cochain_space(dimension)
    basis = real_homology_basis(complex_, dimension)
    cycle = basis.cycle_coefficients().toarray()[:, 0]
    form = space.form(cycle, ORDINARY_FORM)

    result = hodge_decomposition(metric, form, prepare_least_squares)
    harmonic = result.output.harmonic
    periods = basis.periods(harmonic)

    assert len(periods.values) == 1
    assert periods.values[0] != 0.0
    assert np.linalg.norm(harmonic.coefficients()) > 0.0


def test_hodge_decomposition_rejects_identity_semantics_and_closes_solver_errors() -> (
    None
):
    geometry = _regular_simplex(1)
    metric = PositiveHodgeMetric(geometry)
    space = geometry.complex.cochain_space(0)
    alternate = space.form(np.ones(space.size), AlternateSemantics())
    foreign = (
        _regular_simplex(1)
        .complex.cochain_space(0)
        .form(np.ones(space.size), ORDINARY_FORM)
    )

    with pytest.raises(AlgorithmError, match="ordinary"):
        hodge_decomposition(metric, cast(Any, alternate), prepare_least_squares)
    with pytest.raises(AlgorithmError, match="different complex"):
        hodge_decomposition(metric, foreign, prepare_least_squares)

    def fail(operator):
        raise ValueError("private injected text")

    with pytest.raises(AlgorithmError, match="projection") as caught:
        hodge_decomposition(
            metric, space.form(np.ones(space.size), ORDINARY_FORM), fail
        )
    assert "private injected text" not in str(caught.value)


def test_hodge_decomposition_preserves_numpy_error_policy() -> None:
    geometry = _regular_simplex(2)
    metric = PositiveHodgeMetric(geometry)
    space = geometry.complex.cochain_space(1)
    previous = np.seterr(all="raise")
    try:
        expected = np.geterr().copy()
        hodge_decomposition(
            metric,
            space.form(np.arange(space.size, dtype=np.float64), ORDINARY_FORM),
            prepare_least_squares,
        )
        assert np.geterr() == expected
    finally:
        np.seterr(**previous)


@pytest.mark.parametrize("dimension", range(5))
def test_positive_hodge_metric_admits_every_degree_through_dimension_four(
    dimension: int,
) -> None:
    geometry = _regular_simplex(dimension)

    metric = PositiveHodgeMetric(geometry)

    assert metric.geometry is geometry
    for degree in range(dimension + 1):
        expected = geometry.dual_measures(degree) / geometry.primal_measures(degree)
        np.testing.assert_array_equal(metric.weights(degree), expected)
        assert np.all(metric.weights(degree) > 0.0)


def test_positive_hodge_metric_retains_owned_read_only_state() -> None:
    geometry = _regular_simplex(3)
    metric = PositiveHodgeMetric(geometry)
    expected = metric.weights(1)

    returned = metric.weights(1)
    returned[:] = -1.0

    np.testing.assert_array_equal(metric.weights(1), expected)
    assert metric.weights(1).flags.owndata
    with pytest.raises(AlgorithmError, match="degree"):
        metric.weights(-1)
    with pytest.raises(AlgorithmError, match="degree"):
        metric.weights(4)


def test_positive_hodge_metric_preserves_sign_under_representable_scales() -> None:
    for scale in (1.0e-60, 1.0, 1.0e60):
        metric = PositiveHodgeMetric(_regular_simplex(4, scale=scale))
        assert all(
            np.all(metric.weights(degree) > 0.0)
            for degree in range(metric.geometry.complex.dimension + 1)
        )


def test_positive_hodge_metric_rejects_negative_and_exact_zero_weights() -> None:
    with pytest.raises(AlgorithmError, match="strictly positive"):
        PositiveHodgeMetric(_shared_edge_metric(math.nextafter(1.0, 0.0), 0.25))
    with pytest.raises(AlgorithmError, match="strictly positive"):
        PositiveHodgeMetric(_shared_edge_metric(1.0, 0.25))


def test_positive_hodge_metric_exact_fallback_admits_reachable_cancellation() -> None:
    geometry = _shared_edge_metric(math.nextafter(1.0, math.inf), 0.25)

    metric = PositiveHodgeMetric(geometry)

    edges = geometry.complex.simplices(1)
    shared = next(index for index, edge in enumerate(edges) if tuple(edge) == (0, 1))
    assert metric.weights(1)[shared] > 0.0
    assert metric.weights(1)[shared] < np.finfo(np.float64).eps


def test_positive_hodge_metric_preserves_numpy_error_policy() -> None:
    before = np.seterr(all="raise")
    try:
        expected = np.geterr().copy()
        PositiveHodgeMetric(_regular_simplex(2))
        assert np.geterr() == expected
        with pytest.raises(AlgorithmError):
            PositiveHodgeMetric(_shared_edge_metric(1.0, 0.25))
        assert np.geterr() == expected
    finally:
        np.seterr(**before)


def test_positive_hodge_metric_requires_geometry_at_runtime() -> None:
    complex_ = Complex.from_maximal_simplices(np.array([[0, 1]], dtype=np.int64))
    with pytest.raises(AlgorithmError, match="Geometry"):
        PositiveHodgeMetric(cast(Any, complex_))


def test_positive_hodge_metric_does_not_expose_mutable_internal_arrays() -> None:
    metric = PositiveHodgeMetric(_regular_simplex(2))
    with pytest.raises(AttributeError, match="immutable"):
        setattr(metric, "_weights", ())
    assert not hasattr(metric, "is_positive")
    assert not hasattr(metric, "condition_number")


def _assert_geometry_type(value: Geometry[Any]) -> None:
    assert isinstance(value, Geometry)


def test_metric_geometry_is_a_geometry() -> None:
    metric = PositiveHodgeMetric(_regular_simplex(1))
    _assert_geometry_type(metric.geometry)


def test_assemble_poisson_preserves_pointwise_density_and_positive_sign() -> None:
    geometry = _segment()
    metric = PositiveHodgeMetric(geometry)
    C0 = geometry.complex.cochain_space(0)
    density = C0.form(np.array([2.0, -3.0]), AlternateSemantics())

    system = assemble_poisson(metric, density)

    expected = hodge_laplacian(geometry, C0)
    np.testing.assert_array_equal(
        system.operator.matrix().toarray(), expected.matrix().toarray()
    )
    assert system.operator.source is C0
    assert system.operator.target is C0
    assert system.rhs is density
    assert system.rhs.semantics is density.semantics
    np.testing.assert_array_equal(system.rhs.coefficients(), [2.0, -3.0])
    assert system.operator.matrix().toarray()[0, 0] > 0.0
    assert system.operator.matrix().toarray()[0, 1] < 0.0


def test_retained_metric_laplacian_kernel_matches_public_path_all_degrees() -> None:
    for dimension in range(5):
        geometry = _regular_simplex(dimension)
        metric = PositiveHodgeMetric(geometry)
        for degree in range(dimension + 1):
            space = geometry.complex.cochain_space(degree)
            admitted = _hodge_laplacian_from_weights(
                space,
                previous=metric.weights(degree - 1) if degree > 0 else None,
                current=metric.weights(degree),
                following=(metric.weights(degree + 1) if degree < dimension else None),
            )
            np.testing.assert_array_equal(
                admitted.matrix().toarray(),
                hodge_laplacian(geometry, space).matrix().toarray(),
            )


def test_assemble_poisson_matches_stiffness_equation_without_double_load() -> None:
    geometry = _segment(6.0)
    metric = PositiveHodgeMetric(geometry)
    C0 = geometry.complex.cochain_space(0)
    density = C0.form(np.array([4.0, -2.0]), ORDINARY_FORM)

    system = assemble_poisson(metric, density)

    mass = np.diag(metric.weights(0))
    stiffness = mass @ system.operator.matrix().toarray()
    derivative = geometry.complex.boundary_matrix(1).toarray().T
    expected_stiffness = derivative.T @ np.diag(metric.weights(1)) @ derivative
    np.testing.assert_array_equal(stiffness, expected_stiffness)
    np.testing.assert_array_equal(
        mass @ system.rhs.coefficients(), metric.weights(0) * density.coefficients()
    )


def test_assemble_poisson_rejects_foreign_identity_and_nonzero_degree() -> None:
    geometry = _segment()
    metric = PositiveHodgeMetric(geometry)
    foreign = _segment()
    foreign_density = foreign.complex.cochain_space(0).form(np.ones(2), ORDINARY_FORM)
    edge_density = geometry.complex.cochain_space(1).form(np.ones(1), ORDINARY_FORM)

    with pytest.raises(AlgorithmError, match="different complex"):
        assemble_poisson(metric, foreign_density)
    with pytest.raises(AlgorithmError, match="degree-zero"):
        assemble_poisson(metric, cast(Any, edge_density))


@pytest.mark.parametrize("scale", [1.0e-200, 1.0e200])
def test_assemble_poisson_closes_unrepresentable_operator_errors(scale: float) -> None:
    base = _closed_simplex_boundary(1)
    geometry = Geometry.from_positions(base.complex, base.positions * scale)
    metric = PositiveHodgeMetric(geometry)
    C0 = geometry.complex.cochain_space(0)
    with pytest.raises(AlgorithmError, match="operator is not representable") as caught:
        assemble_poisson(metric, C0.form(np.zeros(C0.size), ORDINARY_FORM))
    assert isinstance(caught.value.__cause__, OperatorError)


def test_assemble_poisson_composes_with_ordinary_dirichlet_elimination() -> None:
    raw_geometry = _regular_simplex(2)
    regular = raw_geometry.complex.codimension_one_regular()
    geometry = Geometry.from_positions(regular, raw_geometry.positions)
    metric = PositiveHodgeMetric(geometry)
    C0 = geometry.complex.cochain_space(0)
    density = C0.form(np.arange(C0.size, dtype=np.float64), ORDINARY_FORM)
    boundary = CochainSubspace(C0, np.arange(C0.size, dtype=np.int64))
    values = boundary.form(np.array([1.0, 2.0, 3.0]), ORDINARY_FORM)

    problem = eliminate_dirichlet(assemble_poisson(metric, density), boundary, values)

    assert problem.operator.matrix().shape == (0, 0)
    np.testing.assert_array_equal(
        problem.reconstruct(
            problem.interior.form(np.array([], dtype=np.float64), ORDINARY_FORM)
        ).coefficients(),
        values.coefficients(),
    )


def test_residual_evidence_is_frozen_and_scale_consistent() -> None:
    zero = ResidualEvidence(0.0, 0.0, np.sqrt(np.finfo(np.float64).eps))
    assert zero.residual_norm == 0.0
    with pytest.raises(FrozenInstanceError):
        setattr(zero, "scale", 1.0)
    with pytest.raises(AlgorithmError, match="residual evidence"):
        ResidualEvidence(-1.0, 1.0, 1.0)
    with pytest.raises(AlgorithmError, match="zero residual scale"):
        ResidualEvidence(1.0, 0.0, 1.0)
    with pytest.raises(AlgorithmError, match="limit"):
        ResidualEvidence(1.0, 1.0, 0.5)


def test_mean_zero_problem_is_factory_only() -> None:
    with pytest.raises(AlgorithmError, match="impose_mean_zero"):
        MeanZeroProblem()


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_mean_zero_poisson_solves_closed_complexes_and_certifies_original_system(
    dimension: int,
) -> None:
    geometry = _closed_simplex_boundary(dimension)
    metric = PositiveHodgeMetric(geometry)
    C0 = geometry.complex.cochain_space(0)
    candidate = np.linspace(-1.25, 2.0, C0.size)
    weights = metric.weights(0)
    expected = candidate - np.dot(weights, candidate) / np.sum(weights)
    operator = assemble_poisson(
        metric, C0.form(np.zeros(C0.size), ORDINARY_FORM)
    ).operator
    density_coefficients = np.asarray(operator.matrix() @ expected).ravel()
    density = C0.form(density_coefficients, ORDINARY_FORM)
    retained_density = density.coefficients()
    calls = 0

    def prepare(operator):
        nonlocal calls
        calls += 1
        assert isinstance(operator.source, CochainSubspace)
        assert operator.source is operator.target
        np.testing.assert_array_equal(operator.source.indices(), np.arange(1, C0.size))
        return prepare_direct(operator)

    problem = impose_mean_zero(metric, density)
    solution = problem.solve(prepare)

    assert calls == 1
    assert problem.operator.source is C0
    assert problem.operator.target is C0
    assert problem.rhs is density
    assert solution.form.space is C0
    assert solution.equation_space is C0
    np.testing.assert_array_equal(density.coefficients(), retained_density)
    np.testing.assert_allclose(
        problem.operator.matrix() @ solution.form.coefficients(),
        density.coefficients(),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert abs(np.dot(weights, solution.form.coefficients())) <= (
        np.sqrt(np.finfo(np.float64).eps)
        * np.sum(np.abs(weights * solution.form.coefficients()))
    )
    assert solution.relative_residual <= np.sqrt(np.finfo(np.float64).eps)
    assert problem.compatibility_evidence.residual_norm <= (
        problem.compatibility_evidence.limit * problem.compatibility_evidence.scale
    )


def test_mean_zero_zero_density_has_exact_zero_compatibility_evidence() -> None:
    geometry = _closed_simplex_boundary(1)
    metric = PositiveHodgeMetric(geometry)
    C0 = geometry.complex.cochain_space(0)
    problem = impose_mean_zero(metric, C0.form(np.zeros(C0.size), ORDINARY_FORM))

    assert problem.compatibility_evidence.residual_norm == 0.0
    assert problem.compatibility_evidence.scale == 0.0
    with pytest.raises(AttributeError, match="immutable"):
        setattr(problem, "_rhs", problem.rhs)
    np.testing.assert_array_equal(
        problem.solve(prepare_direct).form.coefficients(), np.zeros(C0.size)
    )


def test_mean_zero_point_returns_without_backend_and_requires_zero_density() -> None:
    geometry = _closed_simplex_boundary(0)
    metric = PositiveHodgeMetric(geometry)
    C0 = geometry.complex.cochain_space(0)
    problem = impose_mean_zero(metric, C0.form(np.zeros(1), ORDINARY_FORM))
    calls = 0

    def forbidden_prepare(operator):
        nonlocal calls
        calls += 1
        raise AssertionError(operator)

    solution = problem.solve(forbidden_prepare)

    assert calls == 0
    np.testing.assert_array_equal(solution.form.coefficients(), [0.0])
    assert solution.residual_norm == 0.0
    assert solution.residual_scale == 0.0
    with pytest.raises(AlgorithmError, match="incompatible"):
        impose_mean_zero(metric, C0.form(np.ones(1), ORDINARY_FORM))


def test_impose_mean_zero_rejects_incompatibility_without_projection() -> None:
    geometry = _closed_simplex_boundary(2)
    metric = PositiveHodgeMetric(geometry)
    C0 = geometry.complex.cochain_space(0)
    density = C0.form(np.ones(C0.size), ORDINARY_FORM)
    retained = density.coefficients()

    with pytest.raises(AlgorithmError, match="incompatible"):
        impose_mean_zero(metric, density)

    np.testing.assert_array_equal(density.coefficients(), retained)


def test_mean_zero_compatibility_is_scale_relative_and_preserves_error_policy() -> None:
    geometry = _closed_simplex_boundary(2)
    metric = PositiveHodgeMetric(geometry)
    C0 = geometry.complex.cochain_space(0)
    base = assemble_poisson(
        metric, C0.form(np.zeros(C0.size), ORDINARY_FORM)
    ).operator.matrix() @ np.linspace(-1.0, 1.0, C0.size)
    previous = np.seterr(all="raise")
    try:
        expected_policy = np.geterr().copy()
        for amplitude in (1.0e-100, 1.0e100):
            impose_mean_zero(
                metric,
                C0.form(np.asarray(base * amplitude).ravel(), ORDINARY_FORM),
            )
            assert np.geterr() == expected_policy
        with pytest.raises(AlgorithmError, match="incompatible"):
            impose_mean_zero(metric, C0.form(np.ones(C0.size), ORDINARY_FORM))
        assert np.geterr() == expected_policy
    finally:
        np.seterr(**previous)


def test_mean_zero_exact_threshold_is_authoritative() -> None:
    geometry = _closed_simplex_boundary(1)
    metric = PositiveHodgeMetric(geometry)
    C0 = geometry.complex.cochain_space(0)
    positive = float((1 << 26) + 1)
    negative = -float((1 << 26) - 1)
    problem = impose_mean_zero(
        metric,
        C0.form(np.array([positive, negative, 0.0]), ORDINARY_FORM),
    )
    assert (
        problem.compatibility_evidence.residual_norm
        == problem.compatibility_evidence.limit
    )
    assert problem.compatibility_evidence.scale == 1.0


@pytest.mark.parametrize("scale", [1.0e100, 8.0e153])
def test_mean_zero_solve_normalizes_extreme_surface_scales(scale: float) -> None:
    base = _closed_simplex_boundary(2)
    geometry = Geometry.from_positions(base.complex, base.positions * scale)
    metric = PositiveHodgeMetric(geometry)
    C0 = geometry.complex.cochain_space(0)
    expected = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
    operator = assemble_poisson(
        metric, C0.form(np.zeros(C0.size), ORDINARY_FORM)
    ).operator
    density = C0.form(np.asarray(operator.matrix() @ expected).ravel(), ORDINARY_FORM)
    problem = impose_mean_zero(metric, density)
    solution = problem.solve(prepare_direct)
    assert np.all(np.isfinite(solution.form.coefficients()))
    assert solution.relative_residual <= np.sqrt(np.finfo(np.float64).eps)
    assert problem.compatibility_evidence.scale in (0.0, 1.0)


def test_impose_mean_zero_rejects_wrong_domain_identity_and_semantics() -> None:
    closed_geometry = _closed_simplex_boundary(1)
    metric = PositiveHodgeMetric(closed_geometry)
    C0 = closed_geometry.complex.cochain_space(0)
    alternate = C0.form(np.zeros(C0.size), AlternateSemantics())
    foreign_geometry = _closed_simplex_boundary(1)
    foreign = foreign_geometry.complex.cochain_space(0).form(
        np.zeros(C0.size), ORDINARY_FORM
    )
    path = (
        Complex.from_maximal_simplices(np.array([[0, 1]], dtype=np.int64))
        .codimension_one_regular()
        .with_boundary()
        .connected()
    )
    path_geometry = Geometry.from_positions(
        path, np.array([[0.0], [1.0]], dtype=np.float64)
    )
    path_metric = PositiveHodgeMetric(path_geometry)
    path_density = path.cochain_space(0).form(np.zeros(2), ORDINARY_FORM)

    with pytest.raises(AlgorithmError, match="ordinary"):
        impose_mean_zero(metric, cast(Any, alternate))
    with pytest.raises(AlgorithmError, match="different complex"):
        impose_mean_zero(metric, foreign)
    with pytest.raises(AlgorithmError, match="boundaryless"):
        impose_mean_zero(cast(Any, path_metric), cast(Any, path_density))
