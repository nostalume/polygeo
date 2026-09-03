use std::sync::Arc;

use polygeo_core::{
    chain::Cochain, form::Cochain as Binary64Cochain, form::Element as Binary64Element,
    form::Space as Binary64Space, geometry::Geometry, geometry::Limit, geometry::Metric,
    geometry::PairingCapability, geometry::SurfaceError, solve::CancellationToken, solve::Executor,
    solve::HeatProblem, solve::HeatResult, solve::Policy, solve::ProblemError, solve::SolveError,
    solve::SolveExt, solve::StorageLimit, solve::WorkLimit, topology::CandidateInput,
    topology::Complex as ComplexCore,
};

fn realization(scale: f64, translation: [f64; 3]) -> Arc<Geometry> {
    let topology = ComplexCore::admit(
        CandidateInput::signed([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], 4, 3, None).unwrap(),
    )
    .unwrap();
    let base = [
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ];
    let positions = base
        .into_iter()
        .flat_map(|point| {
            point
                .into_iter()
                .zip(translation)
                .map(move |(x, t)| scale * x + t)
        })
        .collect();
    Geometry::admit(topology, 3, positions, Limit::DEFAULT).unwrap()
}

fn unlimited() -> (StorageLimit, WorkLimit) {
    (
        StorageLimit::new(u64::MAX, u64::MAX).unwrap(),
        WorkLimit::new(u64::MAX),
    )
}

fn positive_metric(scale: f64) -> (Arc<Geometry>, Metric) {
    let realization = realization(scale, [0.0; 3]);
    let metric = realization
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap();
    (realization, metric)
}

fn scalar(metric: &Metric, coefficients: Vec<f64>) -> Binary64Cochain {
    let space =
        Binary64Space::<Cochain>::full(Arc::clone(metric.realization().topology()), 0).unwrap();
    Binary64Element::admit(space, coefficients).unwrap()
}

fn solve_heat(problem: &HeatProblem) -> HeatResult {
    let (storage, work) = unlimited();
    let prepared = problem
        .prepare(Policy::new(Executor::sequential(), storage, work))
        .unwrap();
    let mut workspace = prepared.workspace_for(problem).unwrap();
    prepared.solve(problem, &mut workspace).unwrap()
}

fn solve_flow(metric: &Metric, time_step: f64) -> polygeo_core::geometry::FlowStep {
    let (storage, work) = unlimited();
    metric
        .frozen_mean_curvature_flow(
            time_step,
            Limit::DEFAULT,
            Policy::new(Executor::sequential(), storage, work),
            &CancellationToken::new(),
        )
        .unwrap()
}

#[test]
fn scalar_heat_evolution_preserves_space_mass_and_energy() {
    let (_, metric) = positive_metric(1.0);
    let initial = scalar(&metric, vec![3.0, -2.0, 5.0, 1.0]);
    let problem = metric.heat_evolution(initial.clone(), 0.1).unwrap();
    let solution = solve_heat(&problem);

    assert!(solution.value().space().same_space(initial.space()));
    assert!(solution.residual_bound() <= 1.0e-11);
    assert!(solution.mass_residual_bound() <= 1.0e-12);
    assert!(solution.energy_after() <= solution.energy_before());
}

#[test]
fn scalar_heat_is_shift_and_scale_covariant_and_reuses_its_factor() {
    fn evolve(scale: f64, time_step: f64, shift: f64) -> Vec<f64> {
        let (_, metric) = positive_metric(scale);
        let initial = scalar(
            &metric,
            [3.0, -2.0, 5.0, 1.0]
                .into_iter()
                .map(|value| value + shift)
                .collect(),
        );
        let problem = metric.heat_evolution(initial, time_step).unwrap();
        solve_heat(&problem).value().coefficients().to_vec()
    }

    let base = evolve(1.0, 0.1, 0.0);
    let transformed = evolve(3.0, 0.9, 17.0);
    for (&left, &right) in base.iter().zip(&transformed) {
        assert!((right - left - 17.0).abs() <= 2.0e-12);
    }

    let (_, metric) = positive_metric(1.0);
    let first = metric
        .heat_evolution(scalar(&metric, vec![1.0, 0.0, 0.0, 0.0]), 0.1)
        .unwrap();
    let second_value = scalar(&metric, vec![0.0, 1.0, 0.0, 0.0]);
    let second = metric.heat_evolution(second_value.clone(), 0.1).unwrap();
    let (storage, work) = unlimited();
    let prepared = first
        .prepare(Policy::new(Executor::sequential(), storage, work))
        .unwrap();
    let mut workspace = prepared.workspace_for(&second).unwrap();
    prepared.solve(&second, &mut workspace).unwrap();
    let changed_time = metric.heat_evolution(second_value, 0.2).unwrap();
    assert_eq!(
        prepared.workspace_for(&changed_time).unwrap_err(),
        SolveError::ProblemMismatch
    );
}

#[test]
fn scalar_heat_admission_and_zero_dimensional_identity_are_explicit() {
    let executor = Executor::sequential();
    let (_, metric) = positive_metric(1.0);
    let initial = scalar(&metric, vec![1.0; 4]);
    assert_eq!(
        metric.heat_evolution(initial.clone(), 0.0).unwrap_err(),
        ProblemError::TimeStep
    );
    let (_, foreign_metric) = positive_metric(1.0);
    let foreign_value = scalar(&foreign_metric, vec![1.0; 4]);
    assert_eq!(
        metric.heat_evolution(foreign_value, 0.1).unwrap_err(),
        ProblemError::SpaceMismatch
    );
    let bounded = metric.heat_evolution(initial, 0.1).unwrap();
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let (storage, work) = unlimited();
    assert_eq!(
        bounded
            .prepare_cancellable(Policy::new(executor, storage, work), &cancelled)
            .unwrap_err(),
        SolveError::Cancelled
    );
    assert_eq!(
        bounded
            .prepare(Policy::new(
                executor,
                StorageLimit::new(0, 0).unwrap(),
                work,
            ))
            .unwrap_err(),
        SolveError::ResourceLimit
    );
    let prepared = bounded
        .prepare(Policy::new(executor, storage, work))
        .unwrap();
    let mut workspace = prepared.workspace_for(&bounded).unwrap();
    prepared.solve(&bounded, &mut workspace).unwrap();

    let triangle =
        ComplexCore::admit(CandidateInput::signed([0_i64, 1, 2], 1, 3, Some(3)).unwrap()).unwrap();
    let triangle = Geometry::admit(
        triangle,
        2,
        vec![0.0, 0.0, 1.0, 0.0, 0.5, 3.0_f64.sqrt() / 2.0],
        Limit::DEFAULT,
    )
    .unwrap();
    let triangle_metric = triangle
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap();
    let triangle_problem = triangle_metric
        .heat_evolution(scalar(&triangle_metric, vec![1.0, 0.0, 0.0]), 0.1)
        .unwrap();
    solve_heat(&triangle_problem);

    let points =
        ComplexCore::admit(CandidateInput::signed([0_i64, 1], 2, 1, Some(2)).unwrap()).unwrap();
    let points = Geometry::admit(points, 1, vec![0.0, 2.0], Limit::DEFAULT).unwrap();
    let point_metric = points
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap();
    let point_values = scalar(&point_metric, vec![2.0, 7.0]);
    let problem = point_metric
        .heat_evolution(point_values.clone(), 0.5)
        .unwrap();
    let storage = StorageLimit::new(0, 0).unwrap();
    let work = WorkLimit::new(0);
    let prepared = problem
        .prepare(Policy::new(executor, storage, work))
        .unwrap();
    let mut workspace = prepared.workspace_for(&problem).unwrap();
    let solution = prepared.solve(&problem, &mut workspace).unwrap();
    assert_eq!(solution.value().coefficients(), point_values.coefficients());
}

#[test]
fn deformation_is_immutable_and_composes_through_admission() {
    let source = realization(1.0, [0.0; 3]);
    let identity = source
        .deform(source.positions().to_vec(), Limit::DEFAULT)
        .unwrap();
    assert!(!Arc::ptr_eq(&source, &identity));
    assert!(Arc::ptr_eq(source.topology(), identity.topology()));
    assert_eq!(source.positions(), identity.positions());

    let translated: Vec<_> = identity
        .positions()
        .chunks_exact(3)
        .flat_map(|row| [row[0] + 2.0, row[1] - 3.0, row[2] + 5.0])
        .collect();
    let target = identity.deform(translated.clone(), Limit::DEFAULT).unwrap();
    assert_eq!(target.positions(), translated);
    assert!((source.positions()[0] - 1.0).abs() <= f64::EPSILON);

    let mut invalid = translated;
    let duplicate = invalid[0..3].to_vec();
    invalid[3..6].copy_from_slice(&duplicate);
    assert!(target.deform(invalid, Limit::DEFAULT).is_err());
    assert!((target.positions()[0] - 3.0).abs() <= f64::EPSILON);
}

#[test]
fn frozen_flow_solves_all_coordinates_atomically_and_decreases_energy() {
    let source = realization(1.0, [4.0, -3.0, 2.0]);
    let metric = source
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap();
    let step = solve_flow(&metric, 0.1);

    assert!(Arc::ptr_eq(source.topology(), step.target().topology()));
    assert!(!Arc::ptr_eq(&source, step.target()));
    assert!((source.positions()[0] - 5.0).abs() <= f64::EPSILON);
    assert!(step.evidence().energy_after() <= step.evidence().energy_before());
    assert!(step.evidence().residual_bound() <= 1.0e-11);
    assert!(step.evidence().centroid_residual_bound() <= 1.0e-12);
}

#[test]
fn changed_metric_and_cancellation_cannot_publish_a_partial_step() {
    let source = realization(1.0, [0.0; 3]);
    let metric = source
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap();
    let (storage, work) = unlimited();

    let changed_positions = source.positions().iter().map(|value| 2.0 * value).collect();
    let changed = source.deform(changed_positions, Limit::DEFAULT).unwrap();
    assert!(Arc::ptr_eq(source.topology(), changed.topology()));
    let changed_metric = changed
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap();
    let changed_step = solve_flow(&changed_metric, 0.1);
    assert!(!Arc::ptr_eq(&changed, changed_step.target()));

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert_eq!(
        metric
            .frozen_mean_curvature_flow(
                0.1,
                Limit::DEFAULT,
                Policy::new(Executor::sequential(), storage, work),
                &cancellation,
            )
            .unwrap_err()
            .solve(),
        Some(SolveError::Cancelled)
    );
    assert!((source.positions()[0] - 1.0).abs() <= f64::EPSILON);

    for time_step in [0.0, f64::NAN] {
        assert_eq!(
            metric
                .frozen_mean_curvature_flow(
                    time_step,
                    Limit::DEFAULT,
                    Policy::new(Executor::sequential(), storage, work),
                    &CancellationToken::new(),
                )
                .unwrap_err()
                .surface(),
            Some(SurfaceError::TimeStep)
        );
    }

    assert_eq!(
        metric
            .frozen_mean_curvature_flow(
                0.1,
                Limit::DEFAULT,
                Policy::new(
                    Executor::sequential(),
                    StorageLimit::new(0, 0).unwrap(),
                    work,
                ),
                &CancellationToken::new(),
            )
            .unwrap_err()
            .solve(),
        Some(SolveError::ResourceLimit)
    );
    assert_eq!(
        metric
            .frozen_mean_curvature_flow(
                0.1,
                Limit::DEFAULT,
                Policy::new(Executor::sequential(), storage, WorkLimit::new(0)),
                &CancellationToken::new(),
            )
            .unwrap_err()
            .solve(),
        Some(SolveError::ResourceLimit)
    );

    let target_limit = Limit::new(
        StorageLimit::new(0, 0).unwrap(),
        u64::MAX,
        WorkLimit::new(u64::MAX),
    );
    assert_eq!(
        metric
            .frozen_mean_curvature_flow(
                0.1,
                target_limit,
                Policy::new(Executor::sequential(), storage, work),
                &CancellationToken::new(),
            )
            .unwrap_err()
            .solve(),
        Some(SolveError::Numerical)
    );
    assert!((source.positions()[0] - 1.0).abs() <= f64::EPSILON);
}

#[test]
fn batched_coordinates_equal_the_same_flow_under_axis_permutation() {
    fn solve(source: &Arc<Geometry>) -> Vec<f64> {
        let metric = source
            .circumcentric_pairing()
            .unwrap()
            .require_positive()
            .unwrap();
        solve_flow(&metric, 0.13).target().positions().to_vec()
    }

    let source = realization(1.0, [2.0, -7.0, 3.0]);
    let ordinary = solve(&source);
    let permuted_positions = source
        .positions()
        .chunks_exact(3)
        .flat_map(|row| [row[2], row[0], row[1]])
        .collect();
    let permuted = source.deform(permuted_positions, Limit::DEFAULT).unwrap();
    let permuted_step = solve(&permuted);
    for (left, right) in ordinary.chunks_exact(3).zip(permuted_step.chunks_exact(3)) {
        for axis in 0..3 {
            assert!((right[axis] - [left[2], left[0], left[1]][axis]).abs() <= 2.0e-12);
        }
    }
}

#[test]
fn flow_is_translation_and_scale_covariant() {
    fn step(scale: f64, translation: [f64; 3], time: f64) -> Vec<f64> {
        let source = realization(scale, translation);
        let metric = source
            .circumcentric_pairing()
            .unwrap()
            .require_positive()
            .unwrap();
        solve_flow(&metric, time).target().positions().to_vec()
    }

    let base = step(1.0, [0.0; 3], 0.1);
    let transformed = step(3.0, [7.0, -5.0, 11.0], 0.9);
    for (left, right) in base.chunks_exact(3).zip(transformed.chunks_exact(3)) {
        for axis in 0..3 {
            let expected = 3.0 * left[axis] + [7.0, -5.0, 11.0][axis];
            assert!((right[axis] - expected).abs() <= 2.0e-12);
        }
    }
}
