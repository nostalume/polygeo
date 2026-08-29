use std::sync::Arc;

use polygeo_core::{
    CancellationToken, CandidateInput, ComplexCore, EuclideanRealization, NativeExecutor,
    RealizationLimit, SolveError, SolveExt, StorageLimit, WorkLimit,
};

fn realization(scale: f64, translation: [f64; 3]) -> Arc<EuclideanRealization> {
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
    EuclideanRealization::admit(topology, 3, positions, RealizationLimit::DEFAULT).unwrap()
}

fn unlimited() -> (StorageLimit, WorkLimit) {
    (
        StorageLimit::new(u64::MAX, u64::MAX).unwrap(),
        WorkLimit::new(u64::MAX),
    )
}

#[test]
fn deformation_is_immutable_and_composes_through_admission() {
    let source = realization(1.0, [0.0; 3]);
    let identity = source
        .deform(source.positions().to_vec(), RealizationLimit::DEFAULT)
        .unwrap();
    assert!(!Arc::ptr_eq(&source, &identity));
    assert!(Arc::ptr_eq(source.topology(), identity.topology()));
    assert_eq!(source.positions(), identity.positions());

    let translated: Vec<_> = identity
        .positions()
        .chunks_exact(3)
        .flat_map(|row| [row[0] + 2.0, row[1] - 3.0, row[2] + 5.0])
        .collect();
    let target = identity
        .deform(translated.clone(), RealizationLimit::DEFAULT)
        .unwrap();
    assert_eq!(target.positions(), translated);
    assert!((source.positions()[0] - 1.0).abs() <= f64::EPSILON);

    let mut invalid = translated;
    let duplicate = invalid[0..3].to_vec();
    invalid[3..6].copy_from_slice(&duplicate);
    assert!(target.deform(invalid, RealizationLimit::DEFAULT).is_err());
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
    let problem = metric.frozen_mean_curvature_flow(0.1).unwrap();
    let (storage, work) = unlimited();
    let prepared = problem
        .prepare_with(&NativeExecutor::sequential(), storage, work)
        .unwrap();
    let mut workspace = prepared.workspace_for(&problem, storage).unwrap();
    let step = prepared.solve(&problem, &mut workspace, work).unwrap();

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
    let problem = metric.frozen_mean_curvature_flow(0.1).unwrap();
    let (storage, work) = unlimited();
    let prepared = problem
        .prepare_with(&NativeExecutor::sequential(), storage, work)
        .unwrap();

    let changed_positions = source.positions().iter().map(|value| 2.0 * value).collect();
    let changed = source
        .deform(changed_positions, RealizationLimit::DEFAULT)
        .unwrap();
    assert!(Arc::ptr_eq(source.topology(), changed.topology()));
    let changed_metric = changed
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap();
    let changed_problem = changed_metric.frozen_mean_curvature_flow(0.1).unwrap();
    assert_eq!(
        prepared
            .workspace_for(&changed_problem, storage)
            .unwrap_err(),
        SolveError::ProblemMismatch
    );

    let mut workspace = prepared.workspace_for(&problem, storage).unwrap();
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert_eq!(
        prepared
            .solve_cancellable(&problem, &mut workspace, work, &cancellation)
            .unwrap_err(),
        SolveError::Cancelled
    );
    assert!((source.positions()[0] - 1.0).abs() <= f64::EPSILON);

    assert!(metric.frozen_mean_curvature_flow(0.0).is_err());
    assert!(metric.frozen_mean_curvature_flow(f64::NAN).is_err());

    let target_limit = RealizationLimit::new(
        StorageLimit::new(0, 0).unwrap(),
        u64::MAX,
        WorkLimit::new(u64::MAX),
    );
    let bounded = metric
        .frozen_mean_curvature_flow_with_limit(0.1, target_limit)
        .unwrap();
    let mut bounded_workspace = prepared.workspace_for(&bounded, storage).unwrap();
    assert_eq!(
        prepared
            .solve(&bounded, &mut bounded_workspace, work)
            .unwrap_err(),
        SolveError::Numerical
    );
    assert!((source.positions()[0] - 1.0).abs() <= f64::EPSILON);
}

#[test]
fn batched_coordinates_equal_the_same_flow_under_axis_permutation() {
    fn solve(source: &Arc<EuclideanRealization>) -> Vec<f64> {
        let metric = source
            .circumcentric_pairing()
            .unwrap()
            .require_positive()
            .unwrap();
        let problem = metric.frozen_mean_curvature_flow(0.13).unwrap();
        let (storage, work) = unlimited();
        let prepared = problem
            .prepare_with(&NativeExecutor::sequential(), storage, work)
            .unwrap();
        let mut workspace = prepared.workspace_for(&problem, storage).unwrap();
        prepared
            .solve(&problem, &mut workspace, work)
            .unwrap()
            .target()
            .positions()
            .to_vec()
    }

    let source = realization(1.0, [2.0, -7.0, 3.0]);
    let ordinary = solve(&source);
    let permuted_positions = source
        .positions()
        .chunks_exact(3)
        .flat_map(|row| [row[2], row[0], row[1]])
        .collect();
    let permuted = source
        .deform(permuted_positions, RealizationLimit::DEFAULT)
        .unwrap();
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
        let problem = metric.frozen_mean_curvature_flow(time).unwrap();
        let (storage, work) = unlimited();
        let prepared = problem
            .prepare_with(&NativeExecutor::sequential(), storage, work)
            .unwrap();
        let mut workspace = prepared.workspace_for(&problem, storage).unwrap();
        prepared
            .solve(&problem, &mut workspace, work)
            .unwrap()
            .target()
            .positions()
            .to_vec()
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
