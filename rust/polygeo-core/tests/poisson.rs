use std::{num::NonZeroUsize, sync::Arc};

use polygeo_core::{
    chain::Chain, chain::Cochain, form::Chain as Binary64Chain, form::Cochain as Binary64Cochain,
    form::Element as Binary64Element, form::Space as Binary64Space, geometry::Geometry,
    geometry::Limit, geometry::Metric, geometry::NondegenerateCapability,
    geometry::PairingCapability, geometry::TriangleSurface, solve::CancellationToken,
    solve::Executor, solve::Policy, solve::SolveExt, solve::StorageLimit, solve::WorkLimit,
    topology::CandidateInput, topology::Complex as ComplexCore,
};

fn sphere_metric() -> Metric {
    let topology = ComplexCore::admit(
        CandidateInput::signed([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], 4, 3, None).unwrap(),
    )
    .unwrap();
    let positions = vec![
        1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
    ];
    Geometry::admit(topology, 3, positions, Limit::DEFAULT)
        .unwrap()
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap()
}

fn density(metric: &Metric, coefficients: Vec<f64>) -> Binary64Cochain {
    let space =
        Binary64Space::<Cochain>::full(Arc::clone(metric.realization().topology()), 0).unwrap();
    Binary64Element::admit(space, coefficients).unwrap()
}

fn load(metric: &Metric, coefficients: Vec<f64>) -> Binary64Chain {
    let space =
        Binary64Space::<Chain>::full(Arc::clone(metric.realization().topology()), 0).unwrap();
    Binary64Element::admit(space, coefficients).unwrap()
}

fn compatible_density(metric: &Metric, left: usize, right: usize) -> Binary64Cochain {
    let weights = metric.hodge_coefficients_slice(0).unwrap();
    let mut coefficients = vec![0.0; weights.len()];
    coefficients[left] = weights[right];
    coefficients[right] = -weights[left];
    density(metric, coefficients)
}

fn cycle_metric(vertex_count: usize) -> Metric {
    let edges = (0..vertex_count).flat_map(|vertex| {
        [vertex, (vertex + 1) % vertex_count].map(|index| u64::try_from(index).unwrap())
    });
    let topology = ComplexCore::admit(
        CandidateInput::unsigned(edges, vertex_count, 2, Some(vertex_count)).unwrap(),
    )
    .unwrap();
    let positions = (0..vertex_count)
        .flat_map(|vertex| {
            let vertex = f64::from(u32::try_from(vertex).unwrap());
            let count = f64::from(u32::try_from(vertex_count).unwrap());
            let angle = std::f64::consts::TAU * vertex / count;
            [angle.cos(), angle.sin()]
        })
        .collect();
    Geometry::admit(topology, 2, positions, Limit::DEFAULT)
        .unwrap()
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap()
}

fn point_metric() -> Metric {
    let topology = ComplexCore::admit(CandidateInput::signed([0], 1, 1, None).unwrap()).unwrap();
    Geometry::admit(topology, 1, vec![0.0], Limit::DEFAULT)
        .unwrap()
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap()
}

#[test]
fn mean_zero_poisson_solves_weak_equation_and_gauge() {
    let metric = sphere_metric();
    let rho = compatible_density(&metric, 0, 1);
    let problem = metric.mean_zero_poisson_density(rho.clone()).unwrap();
    let executor = Executor::parallel(NonZeroUsize::new(2).unwrap());
    let storage = StorageLimit::new(u64::MAX, u64::MAX).unwrap();
    let work = WorkLimit::new(u64::MAX);
    let prepared = problem
        .prepare(Policy::new(executor, storage, work))
        .unwrap();
    let mut workspace = prepared.workspace_for(&problem).unwrap();
    let solution = prepared.solve(&problem, &mut workspace).unwrap();

    let u = solution.potential();
    let laplacian = metric.laplacian(0).unwrap().apply(u).unwrap();
    for (&actual, &expected) in laplacian.coefficients().iter().zip(rho.coefficients()) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    let weights = metric.hodge_coefficients_slice(0).unwrap();
    let gauge: f64 = weights
        .iter()
        .zip(u.coefficients())
        .map(|(&weight, &value)| weight * value)
        .sum();
    assert!(gauge.abs() <= 1.0e-12);
    assert!(solution.evidence().residual_bound() <= 1.0e-12);
    assert_eq!(solution.evidence().exact_fallback_rows(), 0);
}

#[test]
fn preparation_reuses_factors_across_compatible_rhs_and_rejects_foreign_owner() {
    let metric = sphere_metric();
    let first = metric
        .mean_zero_poisson_density(compatible_density(&metric, 0, 1))
        .unwrap();
    let second = metric
        .mean_zero_poisson_density(compatible_density(&metric, 1, 2))
        .unwrap();
    let executor = Executor::sequential();
    let storage = StorageLimit::new(u64::MAX, u64::MAX).unwrap();
    let work = WorkLimit::new(u64::MAX);
    let prepared = first.prepare(Policy::new(executor, storage, work)).unwrap();
    let mut workspace = prepared.workspace_for(&second).unwrap();
    prepared.solve(&second, &mut workspace).unwrap();

    let foreign_metric = sphere_metric();
    let foreign = foreign_metric
        .mean_zero_poisson_density(compatible_density(&foreign_metric, 0, 1))
        .unwrap();
    assert_eq!(
        prepared.workspace_for(&foreign).unwrap_err().reason(),
        "problem_mismatch"
    );
}

#[test]
fn admission_rejects_incompatible_density() {
    let metric = sphere_metric();
    let error = metric
        .mean_zero_poisson_density(density(&metric, vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap_err();
    assert_eq!(error.reason(), "incompatible_rhs");
}

#[test]
fn point_has_the_canonical_analytic_zero_solution() {
    let metric = point_metric();
    let problem = metric
        .mean_zero_poisson_density(density(&metric, vec![0.0]))
        .unwrap();
    let executor = Executor::sequential();
    let storage = StorageLimit::new(0, 0).unwrap();
    let work = WorkLimit::new(0);
    let prepared = problem
        .prepare(Policy::new(executor, storage, work))
        .unwrap();
    let mut workspace = prepared.workspace_for(&problem).unwrap();
    let solution = prepared.solve(&problem, &mut workspace).unwrap();
    assert_eq!(solution.potential().coefficients(), &[0.0]);
}

#[test]
fn resource_and_cancellation_fail_before_publication() {
    let metric = sphere_metric();
    let problem = metric
        .mean_zero_poisson_density(compatible_density(&metric, 0, 1))
        .unwrap();
    let executor = Executor::sequential();
    let zero = StorageLimit::new(0, 0).unwrap();
    assert_eq!(
        problem
            .prepare(Policy::new(executor, zero, WorkLimit::new(u64::MAX)))
            .unwrap_err()
            .reason(),
        "resource_limit"
    );
    let unlimited = StorageLimit::new(u64::MAX, u64::MAX).unwrap();
    let cancelled_prepare = CancellationToken::new();
    cancelled_prepare.cancel();
    assert_eq!(
        problem
            .prepare_cancellable(
                Policy::new(executor, unlimited, WorkLimit::new(u64::MAX)),
                &cancelled_prepare,
            )
            .unwrap_err()
            .reason(),
        "cancelled"
    );
    let prepared = problem
        .prepare(Policy::new(executor, unlimited, WorkLimit::new(u64::MAX)))
        .unwrap();
    let mut workspace = prepared.workspace_for(&problem).unwrap();
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert_eq!(
        prepared
            .solve_cancellable(&problem, &mut workspace, &cancellation)
            .unwrap_err()
            .reason(),
        "cancelled"
    );
}

#[test]
fn large_cycle_uses_the_bounded_sparse_path() {
    let metric = cycle_metric(65);
    let rho = compatible_density(&metric, 7, 31);
    let problem = metric.mean_zero_poisson_density(rho).unwrap();
    let executor = Executor::sequential();
    let storage = StorageLimit::new(u64::MAX, u64::MAX).unwrap();
    let work = WorkLimit::new(u64::MAX);
    let prepared = problem
        .prepare(Policy::new(executor, storage, work))
        .unwrap();
    let mut workspace = prepared.workspace_for(&problem).unwrap();
    let solution = prepared.solve(&problem, &mut workspace).unwrap();
    assert!(solution.evidence().residual_bound() <= 1.0e-10);
    assert_eq!(solution.evidence().exact_fallback_rows(), 0);
}

#[test]
fn integrated_load_uses_the_same_gauged_solver_without_density_conversion() {
    let metric = sphere_metric();
    let surface = TriangleSurface::admit(Arc::clone(metric.realization())).unwrap();
    let source = density(&metric, vec![1.0, 7.0, -1.0, 2.0]);
    let integrated = surface
        .divergence(&surface.gradient(&source).unwrap())
        .unwrap()
        .negated();
    let problem = metric.mean_zero_poisson_load(integrated).unwrap();
    let density_problem = metric
        .mean_zero_poisson_density(compatible_density(&metric, 0, 1))
        .unwrap();
    let executor = Executor::sequential();
    let storage = StorageLimit::new(u64::MAX, u64::MAX).unwrap();
    let work = WorkLimit::new(u64::MAX);
    let prepared = density_problem
        .prepare(Policy::new(executor, storage, work))
        .unwrap();
    let mut workspace = prepared.workspace_for(&problem).unwrap();
    let solution = prepared.solve(&problem, &mut workspace).unwrap();

    let weights = metric.hodge_coefficients_slice(0).unwrap();
    let mean = weights
        .iter()
        .zip(source.coefficients())
        .map(|(&mass, &value)| mass * value)
        .sum::<f64>()
        / weights.iter().sum::<f64>();
    for (&actual, &original) in solution
        .potential()
        .coefficients()
        .iter()
        .zip(source.coefficients())
    {
        assert!((actual - (original - mean)).abs() <= 1.0e-12);
    }
    assert!(solution.evidence().residual_bound() <= 1.0e-12);

    let incompatible = load(&metric, vec![1.0, 0.0, 0.0, 0.0]);
    assert_eq!(
        metric
            .mean_zero_poisson_load(incompatible)
            .unwrap_err()
            .reason(),
        "incompatible_rhs"
    );
}
