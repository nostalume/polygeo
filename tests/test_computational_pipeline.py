"""Binary64 spaces feed realization, problems, flow, and surface computation."""

from __future__ import annotations

import gc

import numpy as np
import pytest

import polygeo


def _equilateral_torus() -> tuple[polygeo.Complex, polygeo.EuclideanRealization]:
    major_sections = 3
    minor_sections = 3
    faces: list[tuple[int, int, int]] = []
    for major in range(major_sections):
        for minor in range(minor_sections):
            lower = major * minor_sections + minor
            major_next = ((major + 1) % major_sections) * minor_sections + minor
            diagonal = ((major + 1) % major_sections) * minor_sections + (
                minor + 1
            ) % minor_sections
            minor_next = major * minor_sections + (minor + 1) % minor_sections
            faces.extend(((lower, major_next, diagonal), (lower, diagonal, minor_next)))
    vertex_count = major_sections * minor_sections
    domain = (
        polygeo.Complex.from_maximal_simplices(np.asarray(faces, dtype=np.int64))
        .triangle_manifold()
        .oriented()
        .without_boundary()
        .connected()
    )
    return domain, polygeo.EuclideanRealization.from_positions(
        domain, np.eye(vertex_count, dtype=np.float64)
    )


def test_domain_issues_binary64_space_element_and_operator() -> None:
    complex_ = polygeo.Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64)
    )
    zero = complex_.binary64_cochain_space(0)
    one = complex_.binary64_cochain_space(1)

    assert isinstance(zero, polygeo.Binary64Space)
    assert zero.variance == "cochain"
    assert zero.degree == 0
    assert zero.size == 3
    assert zero.same_space(complex_.binary64_cochain_space(0))

    value = zero.admit_numpy(np.array([1.0, 2.0, 4.0], dtype=np.float64))
    derivative = zero.exterior_derivative()
    result = derivative.apply(value)

    assert isinstance(value, polygeo.Binary64Element)
    assert isinstance(derivative, polygeo.LinearOperator)
    assert derivative.source.same_space(zero)
    assert derivative.target.same_space(one)
    assert result.space.same_space(one)
    np.testing.assert_array_equal(
        result.coefficients_numpy_copy(), np.array([1.0, 3.0, 2.0])
    )


def test_selected_spaces_share_one_carrier_and_complete_operator_algebra() -> None:
    complex_ = polygeo.Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64)
    )
    full = complex_.binary64_cochain_space(0)
    selected = complex_.binary64_cochain_space(
        0, indices=np.array([0, 2], dtype=np.int64)
    )
    assert type(selected) is type(full)

    restriction = selected.restriction()
    extension = selected.extension_by_zero()
    value = full.admit_numpy(np.array([2.0, 3.0, 5.0]))
    selected_value = value.apply(restriction)
    np.testing.assert_array_equal(selected_value.coefficients_numpy_copy(), [2.0, 5.0])

    composed = extension.compose(restriction)
    np.testing.assert_array_equal(
        composed.apply(value).coefficients_numpy_copy(), [2.0, 0.0, 5.0]
    )
    np.testing.assert_array_equal(
        composed.to_scipy_copy().toarray(), np.diag([1.0, 0.0, 1.0])
    )
    zero = selected.zero_to(full)
    np.testing.assert_array_equal(
        zero.apply(selected_value).coefficients_numpy_copy(), np.zeros(full.size)
    )


def test_realization_allocating_projections_are_explicit_copies() -> None:
    complex_ = polygeo.Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64)
    )
    source = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    realization = polygeo.EuclideanRealization.from_positions(complex_, source)

    first = realization.positions_numpy_copy()
    second = realization.positions_numpy_copy()
    assert not np.shares_memory(first, second)
    first[0, 0] = 99.0
    np.testing.assert_array_equal(realization.positions_numpy_copy(), source)
    assert not hasattr(realization, "positions")
    assert not hasattr(realization, "primal_measures")
    assert not hasattr(realization, "dual_measures")


def test_problem_preparation_workspace_and_solve_share_one_owner_graph() -> None:
    complex_ = polygeo.Complex.from_maximal_simplices(
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    )
    positions = np.array(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
    )
    realization = polygeo.EuclideanRealization.from_positions(complex_, positions)
    metric = realization.positive_metric()
    weights = metric.hodge_coefficients_numpy_copy(0)
    density = complex_.binary64_cochain_space(0).admit_numpy(
        np.array([weights[1], -weights[0], 0.0, 0.0])
    )
    density_problem = metric.mean_zero_poisson_density(density)
    source = complex_.binary64_cochain_space(0).admit_numpy(positions[:, 0])
    surface = polygeo.TriangleSurface.admit(realization)
    load = -surface.divergence(surface.gradient(source))
    problem = metric.mean_zero_poisson_load(load)
    prepared = density_problem.prepare()
    del complex_, realization, metric, density, density_problem, source, surface, load
    gc.collect()
    workspace = prepared.workspace_for(problem)
    solution = prepared.solve(problem, workspace)

    assert isinstance(problem, polygeo.Problem)
    assert isinstance(prepared, polygeo.PreparedProblem)
    assert isinstance(workspace, polygeo.SolveWorkspace)
    assert isinstance(solution, polygeo.PoissonSolution)
    assert solution.residual_bound <= 1.0e-12
    assert solution.gauge_bound <= 1.0e-12


def test_harmonic_one_form_basis_projects_existing_cochain_handles() -> None:
    domain, realization = _equilateral_torus()
    group = polygeo.analyze_integral_homology(domain.chain_complex(), [1])[1]
    basis = realization.positive_metric().harmonic_one_form_basis(group)

    assert isinstance(basis, polygeo.HarmonicOneFormBasis)
    assert basis.rank == group.free_rank == 2
    assert len(basis.forms) == basis.rank
    chain_space = domain.binary64_chain_space(1)
    periods = np.array(
        [
            [
                np.dot(
                    form.coefficients_numpy_copy(),
                    chain_space.realize_integral(
                        group.free_cycle(row)
                    ).coefficients_numpy_copy(),
                )
                for form in basis.forms
            ]
            for row in range(group.free_rank)
        ]
    )
    np.testing.assert_allclose(
        periods, np.eye(group.free_rank), atol=basis.residual_limit
    )
    assert basis.maximum_closedness_residual <= basis.residual_limit
    assert basis.maximum_coclosedness_residual <= basis.residual_limit
    assert basis.maximum_identity_period_residual <= basis.residual_limit


def test_problem_limits_and_cancellation_are_classified_failures() -> None:
    complex_ = polygeo.Complex.from_maximal_simplices(
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    )
    realization = polygeo.Geometry.from_positions(
        complex_,
        np.array(
            [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
        ),
    )
    metric = realization.positive_metric()
    weights = metric.hodge_coefficients_numpy_copy(0)
    density = complex_.binary64_cochain_space(0).admit_numpy(
        np.array([weights[1], -weights[0], 0.0, 0.0])
    )
    problem = metric.mean_zero_poisson_density(density)

    token = polygeo.CancellationToken()
    token.cancel()
    with pytest.raises(polygeo.SolveError) as cancelled:
        problem.prepare(cancellation=token)
    assert cancelled.value.reason == "cancelled"

    with pytest.raises(polygeo.SolveError) as limited:
        problem.prepare(storage=polygeo.StorageLimit(0, 0))
    assert limited.value.reason == "resource_limit"


def test_flow_publishes_new_realization_without_source_mutation() -> None:
    complex_ = polygeo.Complex.from_maximal_simplices(
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    )
    positions = np.array(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
    )
    source = polygeo.EuclideanRealization.from_positions(complex_, positions)
    metric = source.positive_metric()
    scalar = complex_.binary64_cochain_space(0).admit_numpy(positions[:, 0])
    heat_problem = metric.heat_evolution(scalar, 0.1)
    heat_prepared = heat_problem.prepare()
    heat_solution = heat_prepared.solve(
        heat_problem, heat_prepared.workspace_for(heat_problem)
    )
    step = metric.frozen_mean_curvature_flow(0.1)

    assert isinstance(step, polygeo.FlowStep)
    assert isinstance(heat_solution, polygeo.HeatSolution)
    assert heat_solution.value.space.same_space(scalar.space)
    assert heat_solution.residual_bound <= 1.0e-10
    assert heat_solution.mass_residual_bound <= 1.0e-12
    assert heat_solution.energy_after <= heat_solution.energy_before
    np.testing.assert_array_equal(source.positions_numpy_copy(), positions)
    assert step.energy_after <= step.energy_before
    assert step.residual_bound <= 1.0e-10

    token = polygeo.CancellationToken()
    token.cancel()
    with pytest.raises(polygeo.SolveError) as cancelled:
        metric.frozen_mean_curvature_flow(0.1, cancellation=token)
    assert cancelled.value.reason == "cancelled"

    with pytest.raises(polygeo.SolveError) as bounded:
        metric.frozen_mean_curvature_flow(0.1, storage=polygeo.StorageLimit(0, 0))
    assert bounded.value.reason == "resource_limit"

    with pytest.raises(polygeo.SurfaceError) as invalid_time:
        metric.frozen_mean_curvature_flow(0.0)
    assert invalid_time.value.reason == "time_step"


def test_surface_lscm_returns_one_certified_planar_realization() -> None:
    complex_ = polygeo.Complex.from_maximal_simplices(
        np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int64)
    )
    positions = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.2],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, -0.1],
            [0.0, 0.0, 0.5],
        ]
    )
    source = polygeo.EuclideanRealization.from_positions(complex_, positions)
    surface = polygeo.TriangleSurface.admit(source)
    solution = surface.least_squares_conformal_map((0, 2))

    assert isinstance(solution, polygeo.LeastSquaresConformalMapSolution)
    assert solution.realization.complex.shares_data_with(source.complex)
    mapped = solution.realization.positions_numpy_copy()
    np.testing.assert_array_equal(mapped[[0, 2]], [[0.0, 0.0], [1.0, 0.0]])
    assert solution.required_rank == solution.observed_rank == 6
    assert np.isfinite(solution.condition_indicator)
    assert solution.residual_bound < 1.0
    assert solution.minimum_normalized_signed_twice_area > 0.0

    token = polygeo.CancellationToken()
    token.cancel()
    with pytest.raises(polygeo.SolveError) as cancelled:
        surface.least_squares_conformal_map((0, 2), cancellation=token)
    assert cancelled.value.reason == "cancelled"
    with pytest.raises(polygeo.SolveError) as bounded:
        surface.least_squares_conformal_map((0, 2), storage=polygeo.StorageLimit(0, 0))
    assert bounded.value.reason == "resource_limit"
    with pytest.raises(polygeo.SurfaceError) as interior:
        surface.least_squares_conformal_map((0, 4))
    assert interior.value.reason == "anchor_not_boundary"


def test_triangle_surface_uses_one_field_carrier_and_explicit_copies() -> None:
    complex_ = polygeo.Complex.from_maximal_simplices(
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    )
    positions = np.array(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
    )
    realization = polygeo.EuclideanRealization.from_positions(complex_, positions)
    surface = polygeo.TriangleSurface.admit(realization)
    normals = surface.face_unit_normals()
    curvature = surface.gaussian_curvature_measure()
    scalar = complex_.binary64_cochain_space(0).admit_numpy(positions[:, 0])
    gradient = surface.gradient(scalar)
    divergence = surface.divergence(gradient)

    assert type(normals) is polygeo.EntityVectors
    assert polygeo.VertexVectors is polygeo.EntityVectors
    assert polygeo.FaceVectors is polygeo.EntityVectors
    assert normals.is_face_supported
    assert normals.vectors_numpy_copy().shape == (4, 3)
    assert curvature.coefficients_numpy_copy().shape == (4,)
    assert gradient.is_face_supported
    assert divergence.space.variance == "chain"
    areas = realization.primal_measures_numpy_copy(2)
    gradient_values = gradient.vectors_numpy_copy()
    np.testing.assert_allclose(
        np.dot(scalar.coefficients_numpy_copy(), divergence.coefficients_numpy_copy()),
        -np.sum(areas * np.einsum("ij,ij->i", gradient_values, gradient_values)),
        rtol=2.0e-14,
        atol=2.0e-14,
    )

    cycles = realization.complex.integral_dual_cycle_basis()
    connection = surface.levi_civita_connection()
    evidence = connection.holonomy(cycles)
    assert cycles.rank == 0
    assert connection.symmetry_order == 1
    assert np.isfinite(evidence.local_error)
    assert evidence.limit > 0.0
    transports = connection.transports_numpy_copy()
    powered = surface.connection(2, np.zeros(transports.shape[0]))
    assert powered.symmetry_order == 2
    np.testing.assert_allclose(
        powered.transports_numpy_copy(),
        np.column_stack(
            (
                transports[:, 0] ** 2 - transports[:, 1] ** 2,
                2.0 * transports[:, 0] * transports[:, 1],
            )
        ),
        atol=2.0e-14,
    )
    with pytest.raises(ValueError, match="symmetry_order must be positive"):
        surface.connection(0, np.zeros(transports.shape[0]))
    flat_power = surface.connection(
        2, -2.0 * np.arctan2(transports[:, 1], transports[:, 0])
    )
    power_field = flat_power.require_integrable(cycles).direction_field(0.0)
    power_singularities = power_field.singularities()
    assert power_singularities.symmetry_order == 2
    assert sum(power_singularities.charges.to_python_copy()[1]) == 4
    assert (
        power_singularities.maximum_quantization_residual
        <= power_singularities.residual_limit
    )

    homology = polygeo.analyze_integral_homology(complex_.chain_complex(), [1])
    harmonic = realization.positive_metric().harmonic_one_form_basis(homology[1])
    requested = complex_.chain_complex().dual()[0].element({0: 1, 1: 1, 2: 1, 3: 1})
    direction = surface.minimum_energy_direction_field(
        2, realization.positive_metric(), harmonic, cycles, requested, [], 0.25
    )
    assert direction.symmetry_order == 2
    assert direction.power_directions_numpy_copy().shape == (4, 2)
    assert direction.ambient_vector_branch_numpy_copy(0).vectors_numpy_copy().shape == (
        4,
        3,
    )
    singularities = direction.singularities()
    assert isinstance(singularities, polygeo.DirectionFieldSingularities)
    assert singularities.symmetry_order == 2
    assert singularities.charges.to_python_copy() == requested.to_python_copy()
    assert singularities.maximum_quantization_residual <= singularities.residual_limit
