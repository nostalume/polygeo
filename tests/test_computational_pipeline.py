"""Binary64 spaces feed realization, problems, flow, and surface computation."""

from __future__ import annotations

import gc

import numpy as np
import pytest

import polygeo


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
    problem = metric.mean_zero_poisson(density)
    del complex_, realization, metric, density
    gc.collect()
    prepared = problem.prepare()
    workspace = prepared.workspace_for(problem)
    solution = prepared.solve(problem, workspace)

    assert isinstance(problem, polygeo.Problem)
    assert isinstance(prepared, polygeo.PreparedProblem)
    assert isinstance(workspace, polygeo.SolveWorkspace)
    assert isinstance(solution, polygeo.PoissonSolution)
    assert solution.residual_bound <= 1.0e-12
    assert solution.gauge_bound <= 1.0e-12


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
    problem = metric.mean_zero_poisson(density)

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
    problem = source.positive_metric().frozen_mean_curvature_flow(0.1)
    prepared = problem.prepare()
    workspace = prepared.workspace_for(problem)
    step = prepared.solve(problem, workspace)

    assert isinstance(step, polygeo.FlowStep)
    np.testing.assert_array_equal(source.positions_numpy_copy(), positions)
    assert step.energy_after <= step.energy_before
    assert step.residual_bound <= 1.0e-10


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

    assert type(normals) is polygeo.EntityVectors
    assert polygeo.VertexVectors is polygeo.EntityVectors
    assert polygeo.FaceVectors is polygeo.EntityVectors
    assert normals.is_face_supported
    assert normals.vectors_numpy_copy().shape == (4, 3)
    assert curvature.coefficients_numpy_copy().shape == (4,)

    cycles = realization.complex.integral_dual_cycle_basis()
    connection = surface.levi_civita_connection()
    evidence = connection.holonomy(cycles)
    assert cycles.rank == 0
    assert np.isfinite(evidence.local_error)
    assert evidence.limit > 0.0
