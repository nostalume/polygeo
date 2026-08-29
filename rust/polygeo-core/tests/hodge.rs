use std::sync::Arc;

use polygeo_core::{
    Binary64Cochain, Binary64Element, Binary64Space, CancellationToken, CandidateInput, Cochain,
    ComplexCore, EuclideanRealization, NativeExecutor, NondegenerateCapability, PairingCapability,
    PositiveMetric, RealizationLimit, SolveExt, StorageLimit, WorkLimit,
};

const STORAGE: StorageLimit = StorageLimit::new(u64::MAX, u64::MAX).unwrap();
const WORK: WorkLimit = WorkLimit::new(u64::MAX);

fn tetrahedron_metric() -> polygeo_core::PositiveMetric {
    let topology = ComplexCore::admit(
        CandidateInput::signed([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], 4, 3, None).unwrap(),
    )
    .unwrap();
    EuclideanRealization::admit(
        topology,
        3,
        vec![
            1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
        ],
        RealizationLimit::DEFAULT,
    )
    .unwrap()
    .circumcentric_pairing()
    .unwrap()
    .require_positive()
    .unwrap()
}

fn cycle_metric(vertices: usize) -> PositiveMetric {
    let edges = (0..vertices).flat_map(|vertex| {
        [vertex, (vertex + 1) % vertices].map(|index| i64::try_from(index).unwrap())
    });
    let topology =
        ComplexCore::admit(CandidateInput::signed(edges, vertices, 2, Some(vertices)).unwrap())
            .unwrap();
    let positions = (0..vertices)
        .flat_map(|vertex| {
            let angle = std::f64::consts::TAU * f64::from(u32::try_from(vertex).unwrap())
                / f64::from(u32::try_from(vertices).unwrap());
            [angle.cos(), angle.sin()]
        })
        .collect();
    EuclideanRealization::admit(topology, 2, positions, RealizationLimit::DEFAULT)
        .unwrap()
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap()
}

fn source(metric: &PositiveMetric, degree: usize, values: Vec<f64>) -> Binary64Cochain {
    let space = Binary64Space::<Cochain>::full(Arc::clone(metric.realization().topology()), degree)
        .unwrap();
    Binary64Element::admit(space, values).unwrap()
}

#[test]
fn hodge_decomposition_certifies_reconstruction_and_subspace_laws() {
    let metric = tetrahedron_metric();
    let space =
        Binary64Space::<Cochain>::full(Arc::clone(metric.realization().topology()), 1).unwrap();
    let source = Binary64Element::admit(space, vec![1.0, -2.0, 3.0, 0.5, -1.5, 2.5]).unwrap();
    let problem = metric.hodge_decomposition(source.clone()).unwrap();
    let prepared = problem
        .prepare_with(&NativeExecutor::sequential(), STORAGE, WORK)
        .unwrap();
    let mut workspace = prepared.workspace_for(&problem, STORAGE).unwrap();
    let result = prepared.solve(&problem, &mut workspace, WORK).unwrap();

    for index in 0..source.coefficients().len() {
        let reconstructed = result.exact().coefficients()[index]
            + result.coexact().coefficients()[index]
            + result.harmonic().coefficients()[index];
        assert!((reconstructed - source.coefficients()[index]).abs() <= 1.0e-11);
    }

    let derivative = result.exact().exterior_derivative().unwrap();
    assert!(
        derivative
            .coefficients()
            .iter()
            .all(|value| value.abs() <= 1.0e-11)
    );
    let codifferential = metric.codifferential(1).unwrap();
    let coexact_coclosed = codifferential.apply(result.coexact()).unwrap();
    assert!(
        coexact_coclosed
            .coefficients()
            .iter()
            .all(|value| value.abs() <= 1.0e-11)
    );

    let harmonic_closed = result.harmonic().exterior_derivative().unwrap();
    let harmonic_coclosed = codifferential.apply(result.harmonic()).unwrap();
    assert!(
        harmonic_closed
            .coefficients()
            .iter()
            .chain(harmonic_coclosed.coefficients())
            .all(|value| value.abs() <= 1.0e-11)
    );

    let weights = metric.hodge_coefficients_slice(1).unwrap();
    let orthogonality: f64 = weights
        .iter()
        .zip(result.exact().coefficients())
        .zip(result.coexact().coefficients())
        .map(|((&weight, &exact), &coexact)| weight * exact * coexact)
        .sum();
    assert!(orthogonality.abs() <= 1.0e-11);
    assert!(result.evidence().reconstruction_bound() <= 1.0e-11);
    assert_eq!(result.evidence().exact_rank(), 3);
    assert_eq!(result.evidence().coexact_rank(), 3);
    assert!(result.evidence().exact_condition_indicator().is_finite());
    assert!(result.evidence().coexact_condition_indicator().is_finite());
}

#[test]
fn endpoint_degrees_and_nontrivial_harmonic_space_use_empty_images() {
    let metric = cycle_metric(5);
    let executor = NativeExecutor::sequential();

    let vertex_problem = metric
        .hodge_decomposition(source(&metric, 0, vec![1.0, 2.0, 4.0, 8.0, 16.0]))
        .unwrap();
    let vertex_prepared = vertex_problem
        .prepare_with(&executor, STORAGE, WORK)
        .unwrap();
    let mut vertex_workspace = vertex_prepared
        .workspace_for(&vertex_problem, STORAGE)
        .unwrap();
    let vertex = vertex_prepared
        .solve(&vertex_problem, &mut vertex_workspace, WORK)
        .unwrap();
    assert!(
        vertex
            .exact()
            .coefficients()
            .iter()
            .all(|&value| value == 0.0)
    );

    let edge_source = source(&metric, 1, vec![1.0; 5]);
    let edge_problem = metric.hodge_decomposition(edge_source.clone()).unwrap();
    let edge_prepared = edge_problem.prepare_with(&executor, STORAGE, WORK).unwrap();
    let mut edge_workspace = edge_prepared.workspace_for(&edge_problem, STORAGE).unwrap();
    let edge = edge_prepared
        .solve(&edge_problem, &mut edge_workspace, WORK)
        .unwrap();
    assert!(
        edge.coexact()
            .coefficients()
            .iter()
            .all(|&value| value == 0.0)
    );
    assert!(
        edge.harmonic()
            .coefficients()
            .iter()
            .any(|value| value.abs() > 0.5)
    );
}

#[test]
fn preparation_reuses_two_factors_across_sources_and_rejects_foreign_owner() {
    let metric = tetrahedron_metric();
    let first = metric
        .hodge_decomposition(source(&metric, 1, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        .unwrap();
    let second = metric
        .hodge_decomposition(source(&metric, 1, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
        .unwrap();
    let prepared = first
        .prepare_with(&NativeExecutor::sequential(), STORAGE, WORK)
        .unwrap();
    let mut workspace = prepared.workspace_for(&second, STORAGE).unwrap();
    let reused = prepared.solve(&second, &mut workspace, WORK).unwrap();
    let repeated = second
        .prepare_with(&NativeExecutor::sequential(), STORAGE, WORK)
        .unwrap();
    let mut repeated_workspace = repeated.workspace_for(&second, STORAGE).unwrap();
    let repeated = repeated
        .solve(&second, &mut repeated_workspace, WORK)
        .unwrap();
    for (left, right) in [reused.exact(), reused.coexact(), reused.harmonic()]
        .into_iter()
        .zip([repeated.exact(), repeated.coexact(), repeated.harmonic()])
    {
        assert!(
            left.coefficients()
                .iter()
                .zip(right.coefficients())
                .all(|(left, right)| left.to_bits() == right.to_bits())
        );
    }
    assert_eq!(
        reused.evidence().exact_rank(),
        repeated.evidence().exact_rank()
    );
    assert_eq!(
        reused.evidence().coexact_rank(),
        repeated.evidence().coexact_rank()
    );

    let foreign_metric = tetrahedron_metric();
    let foreign = foreign_metric
        .hodge_decomposition(source(
            &foreign_metric,
            1,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ))
        .unwrap();
    assert_eq!(
        prepared
            .workspace_for(&foreign, STORAGE)
            .unwrap_err()
            .reason(),
        "problem_mismatch"
    );
}

#[test]
fn hodge_resources_and_cancellation_fail_without_publication() {
    let metric = tetrahedron_metric();
    let problem = metric
        .hodge_decomposition(source(&metric, 1, vec![1.0; 6]))
        .unwrap();
    let executor = NativeExecutor::sequential();
    let zero = StorageLimit::new(0, 0).unwrap();
    assert_eq!(
        problem
            .prepare_with(&executor, zero, WORK)
            .unwrap_err()
            .reason(),
        "resource_limit"
    );
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert_eq!(
        problem
            .prepare_with_cancellation(&executor, STORAGE, WORK, &cancelled)
            .unwrap_err()
            .reason(),
        "cancelled"
    );
    let prepared = problem.prepare_with(&executor, STORAGE, WORK).unwrap();
    let mut workspace = prepared.workspace_for(&problem, STORAGE).unwrap();
    assert_eq!(
        prepared
            .solve(&problem, &mut workspace, WorkLimit::new(0))
            .unwrap_err()
            .reason(),
        "resource_limit"
    );
    assert_eq!(
        prepared
            .solve_cancellable(&problem, &mut workspace, WORK, &cancelled)
            .unwrap_err()
            .reason(),
        "cancelled"
    );
}
