use std::{mem::size_of, sync::Arc};

use polygeo_core::{
    chain::Cochain, chain::HomologyLimit, chain::IntegralHomology,
    form::Cochain as Binary64Cochain, form::Element as Binary64Element,
    form::Space as Binary64Space, geometry::Geometry, geometry::Limit, geometry::Metric,
    geometry::NondegenerateCapability, geometry::PairingCapability, solve::CancellationToken,
    solve::Executor, solve::Policy, solve::SolveError, solve::SolveExt, solve::StorageLimit,
    solve::WorkLimit, topology::CandidateInput, topology::Complex as ComplexCore,
};

const STORAGE: StorageLimit = StorageLimit::new(u64::MAX, u64::MAX).unwrap();
const WORK: WorkLimit = WorkLimit::new(u64::MAX);

fn tetrahedron_metric() -> polygeo_core::geometry::Metric {
    let topology = ComplexCore::admit(
        CandidateInput::signed([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], 4, 3, None).unwrap(),
    )
    .unwrap();
    Geometry::admit(
        topology,
        3,
        vec![
            1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
        ],
        Limit::DEFAULT,
    )
    .unwrap()
    .circumcentric_pairing()
    .unwrap()
    .require_positive()
    .unwrap()
}

fn cycle_metric(vertices: usize) -> Metric {
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
    Geometry::admit(topology, 2, positions, Limit::DEFAULT)
        .unwrap()
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap()
}

fn equilateral_torus() -> (Metric, IntegralHomology) {
    equilateral_torus_transformed(1.0, 0.0)
}

fn equilateral_torus_transformed(scale: f64, translation: f64) -> (Metric, IntegralHomology) {
    let major_sections = 3;
    let minor_sections = 3;
    let mut faces = Vec::new();
    for major in 0..major_sections {
        for minor in 0..minor_sections {
            let lower = major * minor_sections + minor;
            let major_next = ((major + 1) % major_sections) * minor_sections + minor;
            let diagonal =
                ((major + 1) % major_sections) * minor_sections + (minor + 1) % minor_sections;
            let minor_next = major * minor_sections + (minor + 1) % minor_sections;
            faces.extend([[lower, major_next, diagonal], [lower, diagonal, minor_next]]);
        }
    }
    let vertex_count = major_sections * minor_sections;
    let topology = ComplexCore::admit(
        CandidateInput::unsigned(
            faces
                .iter()
                .flatten()
                .map(|value| u64::try_from(*value).unwrap()),
            faces.len(),
            3,
            Some(vertex_count),
        )
        .unwrap(),
    )
    .unwrap();
    let mut positions = vec![translation; vertex_count * vertex_count];
    for vertex in 0..vertex_count {
        positions[vertex * vertex_count + vertex] += scale;
    }
    let homology =
        IntegralHomology::analyze(&topology.chain_complex(), [1], HomologyLimit::DEFAULT).unwrap();
    let metric = Geometry::admit(topology, vertex_count, positions, Limit::DEFAULT)
        .unwrap()
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap();
    (metric, homology)
}

fn source(metric: &Metric, degree: usize, values: Vec<f64>) -> Binary64Cochain {
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
        .prepare(Policy::new(Executor::sequential(), STORAGE, WORK))
        .unwrap();
    let mut workspace = prepared.workspace_for(&problem).unwrap();
    let result = prepared.solve(&problem, &mut workspace).unwrap();

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
fn harmonic_one_form_basis_is_closed_coclosed_and_period_normalized() {
    let (metric, homology) = equilateral_torus();
    let group = homology.group(1).unwrap();
    let basis = metric
        .harmonic_one_form_basis(
            group,
            Policy::new(Executor::sequential(), STORAGE, WORK),
            &CancellationToken::new(),
        )
        .unwrap();

    assert_eq!(group.free_rank(), 2);
    assert_eq!(basis.rank(), group.free_rank());
    assert_eq!(basis.forms().len(), group.free_rank());
    for (column, form) in basis.forms().iter().enumerate() {
        let periods = group.periods_binary64(form).unwrap();
        for (row, &period) in periods.iter().enumerate() {
            let expected = f64::from(row == column);
            assert!((period - expected).abs() <= basis.residual_limit());
        }
    }
    assert!(basis.maximum_closedness_residual() <= basis.residual_limit());
    assert!(basis.maximum_coclosedness_residual() <= basis.residual_limit());
    assert!(basis.maximum_identity_period_residual() <= basis.residual_limit());
}

#[test]
fn harmonic_basis_empty_case_and_failures_publish_no_partial_basis() {
    let sphere = tetrahedron_metric();
    let sphere_homology = IntegralHomology::analyze(
        &sphere.realization().topology().chain_complex(),
        [0, 1],
        HomologyLimit::DEFAULT,
    )
    .unwrap();
    let empty = sphere
        .harmonic_one_form_basis(
            sphere_homology.group(1).unwrap(),
            Policy::new(Executor::sequential(), STORAGE, WORK),
            &CancellationToken::new(),
        )
        .unwrap();
    assert_eq!(empty.rank(), 0);
    assert!(empty.forms().is_empty());
    assert_eq!(empty.maximum_closedness_residual().to_bits(), 0);
    assert_eq!(empty.maximum_coclosedness_residual().to_bits(), 0);
    assert_eq!(empty.maximum_identity_period_residual().to_bits(), 0);
    assert_eq!(empty.residual_limit().to_bits(), 0);

    assert_harmonic_basis_failures(&sphere, &sphere_homology);
}

fn assert_harmonic_basis_failures(sphere: &Metric, sphere_homology: &IntegralHomology) {
    assert_eq!(
        sphere
            .harmonic_one_form_basis(
                sphere_homology.group(0).unwrap(),
                Policy::new(Executor::sequential(), STORAGE, WORK),
                &CancellationToken::new(),
            )
            .unwrap_err()
            .solve(),
        Some(SolveError::ProblemMismatch)
    );

    let (metric, homology) = equilateral_torus();
    let final_basis_bytes = u64::try_from(
        homology.group(1).unwrap().free_rank()
            * metric
                .realization()
                .topology()
                .basis(1)
                .unwrap()
                .row_count()
            * size_of::<f64>(),
    )
    .unwrap();
    assert_eq!(
        metric
            .harmonic_one_form_basis(
                homology.group(1).unwrap(),
                Policy::new(
                    Executor::sequential(),
                    StorageLimit::new(final_basis_bytes, u64::MAX).unwrap(),
                    WORK,
                ),
                &CancellationToken::new(),
            )
            .unwrap()
            .rank(),
        2
    );
    assert_eq!(
        sphere
            .harmonic_one_form_basis(
                homology.group(1).unwrap(),
                Policy::new(Executor::sequential(), STORAGE, WORK),
                &CancellationToken::new(),
            )
            .unwrap_err()
            .solve(),
        Some(SolveError::ProblemMismatch)
    );
    assert_eq!(
        metric
            .harmonic_one_form_basis(
                homology.group(1).unwrap(),
                Policy::new(
                    Executor::sequential(),
                    StorageLimit::new(0, 0).unwrap(),
                    WORK,
                ),
                &CancellationToken::new(),
            )
            .unwrap_err()
            .solve(),
        Some(SolveError::ResourceLimit)
    );
    assert_eq!(
        metric
            .harmonic_one_form_basis(
                homology.group(1).unwrap(),
                Policy::new(Executor::sequential(), STORAGE, WorkLimit::new(0)),
                &CancellationToken::new(),
            )
            .unwrap_err()
            .solve(),
        Some(SolveError::ResourceLimit)
    );
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert_eq!(
        metric
            .harmonic_one_form_basis(
                homology.group(1).unwrap(),
                Policy::new(Executor::sequential(), STORAGE, WORK),
                &cancelled,
            )
            .unwrap_err()
            .solve(),
        Some(SolveError::Cancelled)
    );
}

#[test]
fn harmonic_basis_is_rigid_motion_and_uniform_scale_invariant() {
    let (first_metric, first_homology) = equilateral_torus_transformed(1.0, 0.0);
    let (second_metric, second_homology) = equilateral_torus_transformed(7.0, -13.0);
    let first = first_metric
        .harmonic_one_form_basis(
            first_homology.group(1).unwrap(),
            Policy::new(Executor::sequential(), STORAGE, WORK),
            &CancellationToken::new(),
        )
        .unwrap();
    let second = second_metric
        .harmonic_one_form_basis(
            second_homology.group(1).unwrap(),
            Policy::new(Executor::sequential(), STORAGE, WORK),
            &CancellationToken::new(),
        )
        .unwrap();

    for (left, right) in first.forms().iter().zip(second.forms()) {
        for (&left, &right) in left.coefficients().iter().zip(right.coefficients()) {
            assert!((left - right).abs() <= 1.0e-13);
        }
    }
}

#[test]
fn endpoint_degrees_and_nontrivial_harmonic_space_use_empty_images() {
    let metric = cycle_metric(5);
    let executor = Executor::sequential();

    let vertex_problem = metric
        .hodge_decomposition(source(&metric, 0, vec![1.0, 2.0, 4.0, 8.0, 16.0]))
        .unwrap();
    let vertex_prepared = vertex_problem
        .prepare(Policy::new(executor, STORAGE, WORK))
        .unwrap();
    let mut vertex_workspace = vertex_prepared.workspace_for(&vertex_problem).unwrap();
    let vertex = vertex_prepared
        .solve(&vertex_problem, &mut vertex_workspace)
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
    let edge_prepared = edge_problem
        .prepare(Policy::new(executor, STORAGE, WORK))
        .unwrap();
    let mut edge_workspace = edge_prepared.workspace_for(&edge_problem).unwrap();
    let edge = edge_prepared
        .solve(&edge_problem, &mut edge_workspace)
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
        .prepare(Policy::new(Executor::sequential(), STORAGE, WORK))
        .unwrap();
    let mut workspace = prepared.workspace_for(&second).unwrap();
    let reused = prepared.solve(&second, &mut workspace).unwrap();
    let repeated = second
        .prepare(Policy::new(Executor::sequential(), STORAGE, WORK))
        .unwrap();
    let mut repeated_workspace = repeated.workspace_for(&second).unwrap();
    let repeated = repeated.solve(&second, &mut repeated_workspace).unwrap();
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
        prepared.workspace_for(&foreign).unwrap_err().reason(),
        "problem_mismatch"
    );
}

#[test]
fn hodge_resources_and_cancellation_fail_without_publication() {
    let metric = tetrahedron_metric();
    let problem = metric
        .hodge_decomposition(source(&metric, 1, vec![1.0; 6]))
        .unwrap();
    let executor = Executor::sequential();
    let zero = StorageLimit::new(0, 0).unwrap();
    assert_eq!(
        problem
            .prepare(Policy::new(executor, zero, WORK))
            .unwrap_err()
            .reason(),
        "resource_limit"
    );
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert_eq!(
        problem
            .prepare_cancellable(Policy::new(executor, STORAGE, WORK), &cancelled)
            .unwrap_err()
            .reason(),
        "cancelled"
    );
    let prepared = problem
        .prepare(Policy::new(executor, STORAGE, WORK))
        .unwrap();
    let mut workspace = prepared.workspace_for(&problem).unwrap();
    assert_eq!(
        prepared
            .solve_cancellable(&problem, &mut workspace, &cancelled)
            .unwrap_err()
            .reason(),
        "cancelled"
    );
}
