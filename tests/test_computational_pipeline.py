"""Forms feed geometry, problems, flow, and surface computation."""

from __future__ import annotations

from typing import cast

import gc

import numpy as np
import pytest

import polygeo


def _equilateral_torus() -> tuple[polygeo.topology.Complex, polygeo.geometry.Geometry]:
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
        polygeo.topology.Complex.from_maximal_simplices(
            np.asarray(faces, dtype=np.int64)
        )
        .triangle_manifold()
        .oriented()
        .without_boundary()
        .connected()
    )
    return domain, polygeo.geometry.Geometry.from_positions(
        domain, np.eye(vertex_count, dtype=np.float64)
    )


def test_domain_issues_binary64_space_element_and_operator() -> None:
    complex_ = polygeo.topology.Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64)
    )
    zero = complex_.binary64_cochain_space(0)
    one = complex_.binary64_cochain_space(1)

    assert isinstance(zero, polygeo.form.Space)
    assert zero.variance == "cochain"
    assert zero.degree == 0
    assert zero.size == 3
    assert zero.same_space(complex_.binary64_cochain_space(0))

    value = zero.admit_numpy(np.array([1.0, 2.0, 4.0], dtype=np.float64))
    derivative = zero.exterior_derivative()
    result = derivative.apply(value)

    assert isinstance(value, polygeo.form.Element)
    assert isinstance(derivative, polygeo.form.Operator)
    assert derivative.source.same_space(zero)
    assert derivative.target.same_space(one)
    assert result.space.same_space(one)
    np.testing.assert_array_equal(
        result.coefficients_numpy_copy(), np.array([1.0, 3.0, 2.0])
    )


def test_selected_spaces_share_one_carrier_and_complete_operator_algebra() -> None:
    complex_ = polygeo.topology.Complex.from_maximal_simplices(
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
    zero = selected.zero_to(full)
    np.testing.assert_array_equal(
        zero.apply(selected_value).coefficients_numpy_copy(), np.zeros(full.size)
    )


def test_realization_allocating_projections_are_explicit_copies() -> None:
    complex_ = polygeo.topology.Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64)
    )
    source = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    realization = polygeo.geometry.Geometry.from_positions(complex_, source)

    first = realization.positions_numpy_copy()
    second = realization.positions_numpy_copy()
    assert not np.shares_memory(first, second)
    first[0, 0] = 99.0
    np.testing.assert_array_equal(realization.positions_numpy_copy(), source)
    assert not hasattr(realization, "positions")
    assert not hasattr(realization, "primal_measures")
    assert not hasattr(realization, "dual_measures")


def test_problem_preparation_workspace_and_solve_share_one_owner_graph() -> None:
    complex_ = polygeo.topology.Complex.from_maximal_simplices(
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    )
    positions = np.array(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
    )
    realization = polygeo.geometry.Geometry.from_positions(complex_, positions)
    metric = realization.metric()
    weights = metric.hodge_coefficients_numpy_copy(0)
    density = complex_.binary64_cochain_space(0).admit_numpy(
        np.array([weights[1], -weights[0], 0.0, 0.0])
    )
    density_problem = metric.mean_zero_poisson_density(density)
    source = complex_.binary64_cochain_space(0).admit_numpy(positions[:, 0])
    surface = polygeo.geometry.TriangleSurface.admit(realization)
    load = -surface.divergence(surface.gradient(source))
    problem = metric.mean_zero_poisson_load(load)
    prepared = density_problem.prepare()
    del complex_, realization, metric, density, density_problem, source, surface, load
    gc.collect()
    workspace = prepared.workspace_for(problem)
    solution = prepared.solve(problem, workspace)

    assert isinstance(problem, polygeo.solve.Problem)
    assert isinstance(prepared, polygeo.solve.Prepared)
    assert isinstance(workspace, polygeo.solve.Workspace)
    assert isinstance(solution, polygeo.solve.PoissonResult)
    assert solution.residual_bound <= 1.0e-12
    assert solution.gauge_bound <= 1.0e-12


def test_harmonic_one_form_basis_projects_existing_cochain_handles() -> None:
    domain, realization = _equilateral_torus()
    group = polygeo.chain.analyze_integral_homology(domain.chain_complex(), [1])[1]
    basis = realization.metric().harmonic_basis(group)

    assert isinstance(basis, polygeo.field.HarmonicBasis)
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
    complex_ = polygeo.topology.Complex.from_maximal_simplices(
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    )
    realization = polygeo.geometry.Geometry.from_positions(
        complex_,
        np.array(
            [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
        ),
    )
    metric = realization.metric()
    weights = metric.hodge_coefficients_numpy_copy(0)
    density = complex_.binary64_cochain_space(0).admit_numpy(
        np.array([weights[1], -weights[0], 0.0, 0.0])
    )
    problem = metric.mean_zero_poisson_density(density)

    token = polygeo.solve.CancellationToken()
    token.cancel()
    with pytest.raises(polygeo.solve.SolveError) as cancelled:
        problem.prepare(cancellation=token)
    assert cancelled.value.reason == "cancelled"

    with pytest.raises(polygeo.solve.SolveError) as limited:
        problem.prepare(
            policy=polygeo.solve.Policy(storage=polygeo.solve.StorageLimit(0, 0))
        )
    assert limited.value.reason == "resource_limit"


def test_flow_publishes_new_realization_without_source_mutation() -> None:
    complex_ = polygeo.topology.Complex.from_maximal_simplices(
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    )
    positions = np.array(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
    )
    source = polygeo.geometry.Geometry.from_positions(complex_, positions)
    metric = source.metric()
    scalar = complex_.binary64_cochain_space(0).admit_numpy(positions[:, 0])
    heat_problem = metric.heat_evolution(scalar, 0.1)
    heat_prepared = heat_problem.prepare()
    heat_solution = heat_prepared.solve(
        heat_problem, heat_prepared.workspace_for(heat_problem)
    )
    step = metric.frozen_mean_curvature_flow(0.1)

    assert isinstance(step, polygeo.geometry.FlowStep)
    assert isinstance(heat_solution, polygeo.solve.HeatResult)
    assert heat_solution.value.space.same_space(scalar.space)
    assert heat_solution.residual_bound <= 1.0e-10
    assert heat_solution.mass_residual_bound <= 1.0e-12
    assert heat_solution.energy_after <= heat_solution.energy_before
    np.testing.assert_array_equal(source.positions_numpy_copy(), positions)
    assert step.energy_after <= step.energy_before
    assert step.residual_bound <= 1.0e-10

    token = polygeo.solve.CancellationToken()
    token.cancel()
    with pytest.raises(polygeo.solve.SolveError) as cancelled:
        metric.frozen_mean_curvature_flow(0.1, cancellation=token)
    assert cancelled.value.reason == "cancelled"

    with pytest.raises(polygeo.solve.SolveError) as bounded:
        metric.frozen_mean_curvature_flow(
            0.1, policy=polygeo.solve.Policy(storage=polygeo.solve.StorageLimit(0, 0))
        )
    assert bounded.value.reason == "resource_limit"

    with pytest.raises(polygeo.geometry.SurfaceError) as invalid_time:
        metric.frozen_mean_curvature_flow(0.0)
    assert invalid_time.value.reason == "time_step"


def test_surface_lscm_returns_one_certified_planar_realization() -> None:
    complex_ = polygeo.topology.Complex.from_maximal_simplices(
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
    source = polygeo.geometry.Geometry.from_positions(complex_, positions)
    surface = polygeo.geometry.TriangleSurface.admit(source)
    solution = surface.conformal_map((0, 2))

    assert isinstance(solution, polygeo.geometry.ConformalMap)
    assert solution.geometry.topology.shares_data_with(source.topology)
    mapped = solution.geometry.positions_numpy_copy()
    np.testing.assert_array_equal(mapped[[0, 2]], [[0.0, 0.0], [1.0, 0.0]])
    assert solution.required_rank == solution.observed_rank == 6
    assert np.isfinite(solution.condition_indicator)
    assert solution.residual_bound < 1.0
    assert solution.minimum_normalized_signed_twice_area > 0.0

    token = polygeo.solve.CancellationToken()
    token.cancel()
    with pytest.raises(polygeo.solve.SolveError) as cancelled:
        surface.conformal_map((0, 2), cancellation=token)
    assert cancelled.value.reason == "cancelled"
    with pytest.raises(polygeo.solve.SolveError) as bounded:
        surface.conformal_map(
            (0, 2),
            policy=polygeo.solve.Policy(storage=polygeo.solve.StorageLimit(0, 0)),
        )
    assert bounded.value.reason == "resource_limit"
    with pytest.raises(polygeo.geometry.SurfaceError) as interior:
        surface.conformal_map((0, 4))
    assert interior.value.reason == "anchor_not_boundary"


def test_triangle_surface_uses_one_field_carrier_and_explicit_copies() -> None:
    complex_ = polygeo.topology.Complex.from_maximal_simplices(
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    )
    positions = np.array(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
    )
    realization = polygeo.geometry.Geometry.from_positions(complex_, positions)
    surface = polygeo.geometry.TriangleSurface.admit(realization)
    normals = surface.face_unit_normals()
    curvature = surface.gaussian_curvature_measure()
    scalar = complex_.binary64_cochain_space(0).admit_numpy(positions[:, 0])
    gradient = surface.gradient(scalar)
    divergence = surface.divergence(gradient)

    assert normals.support_degree == 2
    assert normals.values_numpy_copy().shape == (4, 3)
    with pytest.raises(polygeo.geometry.SurfaceError) as wrong_support:
        surface.divergence(
            cast(polygeo.geometry.FaceField, surface.vertex_field(np.zeros((4, 3))))
        )
    assert wrong_support.value.reason == "field_shape"
    assert curvature.coefficients_numpy_copy().shape == (4,)
    assert gradient.support_degree == 2
    assert divergence.space.variance == "chain"
    areas = realization.primal_measures_numpy_copy(2)
    gradient_values = gradient.values_numpy_copy()
    np.testing.assert_allclose(
        np.dot(scalar.coefficients_numpy_copy(), divergence.coefficients_numpy_copy()),
        -np.sum(areas * np.einsum("ij,ij->i", gradient_values, gradient_values)),
        rtol=2.0e-14,
        atol=2.0e-14,
    )

    cycles = realization.topology.dual_cycles()
    connection = surface.levi_civita()
    evidence = connection.holonomy(cycles)
    assert cycles.rank == 0
    assert connection.symmetry_order == 1
    assert np.isfinite(evidence.local_error)
    assert evidence.limit > 0.0
    transports = connection.transports_numpy_copy()
    np.testing.assert_array_equal(
        connection.interior_edge_indices_numpy_copy(),
        np.arange(realization.topology.simplex_count(1)),
    )
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
    integrable = flat_power.require_integrable()
    power_field = integrable.direction(0.0)
    assert power_field.connection.crossing_error == integrable.crossing_error
    power_singularities = power_field.singularities()
    assert power_singularities.symmetry_order == 2
    assert sum(power_singularities.charges.to_python_copy()[1]) == 4
    assert power_singularities.boundary_turns_python_copy() == ()
    assert (
        power_singularities.maximum_quantization_residual
        <= power_singularities.residual_limit
    )

    homology = polygeo.chain.analyze_integral_homology(complex_.chain_complex(), [1])
    harmonic = realization.metric().harmonic_basis(homology[1])
    requested = complex_.chain_complex().dual()[0].element({0: 1, 1: 1, 2: 1, 3: 1})
    direction = surface.direction_field(
        2, realization.metric(), harmonic, cycles, requested, [], 0.25
    )
    assert direction.symmetry_order == 2
    assert direction.power_directions_numpy_copy().shape == (4, 2)
    assert direction.ambient_branch_numpy_copy(0).values_numpy_copy().shape == (
        4,
        3,
    )
    singularities = direction.singularities()
    assert isinstance(singularities, polygeo.field.Singularities)
    assert singularities.symmetry_order == 2
    assert singularities.charges.to_python_copy() == requested.to_python_copy()
    assert singularities.maximum_quantization_residual <= singularities.residual_limit

    sections = 16
    angles = 2.0 * np.pi * np.arange(sections) / sections
    disk_positions = np.vstack(
        (
            np.column_stack((np.cos(angles), np.sin(angles), np.zeros(sections))),
            np.zeros((1, 3)),
        )
    )
    disk_faces = np.column_stack(
        (
            np.arange(sections),
            np.roll(np.arange(sections), -1),
            np.full(sections, sections),
        )
    )
    disk = polygeo.topology.Complex.from_maximal_simplices(disk_faces.astype(np.int64))
    disk_realization = polygeo.geometry.Geometry.from_positions(disk, disk_positions)
    disk_surface = polygeo.geometry.TriangleSurface.admit(disk_realization)
    boundary_field = disk_surface.boundary_direction(2, disk_realization.metric(), 0.0)
    assert boundary_field.power_directions_numpy_copy().shape == (sections, 2)
    assert boundary_field.singularities().boundary_turns_python_copy() == (0,)
