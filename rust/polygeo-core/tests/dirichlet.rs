use std::sync::Arc;

use polygeo_core::{
    chain::Cochain, form::Cochain as Binary64Cochain, form::Element as Binary64Element,
    form::Space as Binary64Space, geometry::Geometry, geometry::Limit, geometry::Metric,
    geometry::NondegenerateCapability, geometry::PairingCapability, solve::CancellationToken,
    solve::Executor, solve::Policy, solve::ProblemError, solve::SolveExt, solve::StorageLimit,
    solve::WorkLimit, topology::CandidateInput, topology::Complex as ComplexCore,
};

const STORAGE: StorageLimit = StorageLimit::new(u64::MAX, u64::MAX).unwrap();
const WORK: WorkLimit = WorkLimit::new(u64::MAX);

fn cochain(space: Binary64Space<Cochain>, values: Vec<f64>) -> Binary64Cochain {
    Binary64Element::admit(space, values).unwrap()
}

fn triangle() -> Arc<ComplexCore> {
    ComplexCore::admit(CandidateInput::signed([0, 1, 2], 1, 3, None).unwrap()).unwrap()
}

fn hexagonal_disk() -> Metric {
    let triangles = (0..6).flat_map(|index| {
        let next = 1 + (index + 1) % 6;
        [0, 1 + index, next].map(i64::from)
    });
    let topology =
        ComplexCore::admit(CandidateInput::signed(triangles, 6, 3, Some(7)).unwrap()).unwrap();
    let positions = std::iter::once([0.0, 0.0])
        .chain((0..6).map(|index| {
            let angle = std::f64::consts::TAU * f64::from(index) / 6.0;
            [angle.cos(), angle.sin()]
        }))
        .flatten()
        .collect();
    Geometry::admit(topology, 2, positions, Limit::DEFAULT)
        .unwrap()
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap()
}

#[test]
fn generic_dirichlet_uses_the_operator_equation_and_exact_prescription() {
    let owner = triangle();
    let full = Binary64Space::<Cochain>::full(Arc::clone(&owner), 0).unwrap();
    let boundary = Arc::new(owner.selection(0, vec![0, 2]).unwrap());
    let prescribed = cochain(
        Binary64Space::selected(Arc::clone(&boundary)).unwrap(),
        vec![7.0, -3.0],
    );
    let problem = full
        .identity()
        .dirichlet(cochain(full, vec![2.0, 5.0, 11.0]), prescribed)
        .unwrap();
    let prepared = problem
        .prepare(Policy::new(Executor::sequential(), STORAGE, WORK))
        .unwrap();
    let mut workspace = prepared.workspace_for(&problem).unwrap();
    let solution = prepared.solve(&problem, &mut workspace).unwrap();

    assert_eq!(solution.value().coefficients(), &[7.0, 5.0, -3.0]);
    assert_eq!(solution.evidence().exact_fallback_rows(), 0);
}

#[test]
fn harmonic_extension_uses_the_true_boundary_and_reuses_its_factor() {
    let metric = hexagonal_disk();
    let owner = metric.realization().topology();
    owner.refine_regular().unwrap();
    let boundary = Arc::new(
        owner
            .boundary_subset()
            .unwrap()
            .canonical_selection(0)
            .unwrap(),
    );
    let boundary_space = Binary64Space::selected(Arc::clone(&boundary)).unwrap();
    let first = metric
        .harmonic_extension(cochain(
            boundary_space.clone(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ))
        .unwrap();
    let prepared = first
        .prepare(Policy::new(Executor::sequential(), STORAGE, WORK))
        .unwrap();
    let second = metric
        .harmonic_extension(cochain(boundary_space, vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
        .unwrap();
    let mut workspace = prepared.workspace_for(&second).unwrap();
    let solution = prepared.solve(&second, &mut workspace).unwrap();

    assert!((solution.value().coefficients()[0] - 3.5).abs() <= 1.0e-12);
    for (&index, &expected) in boundary
        .indices()
        .iter()
        .zip([6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0].iter())
    {
        assert_eq!(
            solution.value().coefficients()[index].to_bits(),
            expected.to_bits()
        );
    }
    let laplacian = metric
        .laplacian(0)
        .unwrap()
        .apply(solution.value())
        .unwrap();
    assert!(laplacian.coefficients()[0].abs() <= 1.0e-12);
}

#[test]
fn admission_rejects_nonboundary_and_foreign_selected_values() {
    let metric = hexagonal_disk();
    let owner = metric.realization().topology();
    let interior = Arc::new(owner.selection(0, vec![0]).unwrap());
    let error = metric
        .harmonic_extension(cochain(
            Binary64Space::selected(interior).unwrap(),
            vec![1.0],
        ))
        .unwrap_err();
    assert_eq!(error, ProblemError::BoundarySelection);

    let foreign = triangle();
    let selection = Arc::new(foreign.selection(0, vec![0]).unwrap());
    let error = metric
        .harmonic_extension(cochain(
            Binary64Space::selected(selection).unwrap(),
            vec![1.0],
        ))
        .unwrap_err();
    assert_eq!(error, ProblemError::SpaceMismatch);
}

#[test]
fn full_boundary_has_an_analytic_extension_and_singular_general_system_fails() {
    let owner = triangle();
    owner.refine_regular().unwrap();
    let boundary = Arc::new(
        owner
            .boundary_subset()
            .unwrap()
            .canonical_selection(0)
            .unwrap(),
    );
    let realization = Geometry::admit(
        Arc::clone(&owner),
        2,
        vec![0.0, 0.0, 1.0, 0.0, 0.5, 3.0_f64.sqrt() / 2.0],
        Limit::DEFAULT,
    )
    .unwrap();
    let metric = realization
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap();
    let values = cochain(
        Binary64Space::selected(Arc::clone(&boundary)).unwrap(),
        vec![1.0, -2.0, 4.0],
    );
    let problem = metric.harmonic_extension(values).unwrap();
    let prepared = problem
        .prepare(Policy::new(Executor::sequential(), STORAGE, WORK))
        .unwrap();
    let mut workspace = prepared.workspace_for(&problem).unwrap();
    assert_eq!(
        prepared
            .solve(&problem, &mut workspace)
            .unwrap()
            .value()
            .coefficients(),
        &[1.0, -2.0, 4.0]
    );

    let full = Binary64Space::<Cochain>::full(owner, 0).unwrap();
    let prescribed = cochain(
        Binary64Space::selected(Arc::new(boundary.owner().selection(0, vec![0]).unwrap())).unwrap(),
        vec![0.0],
    );
    let singular = full
        .zero_to(&full)
        .dirichlet(cochain(full, vec![0.0; 3]), prescribed)
        .unwrap();
    assert_eq!(
        singular
            .prepare(Policy::new(Executor::sequential(), STORAGE, WORK))
            .unwrap_err()
            .reason(),
        "factorization"
    );

    let empty = Arc::new(boundary.owner().selection(0, Vec::new()).unwrap());
    let full = Binary64Space::<Cochain>::full(Arc::clone(boundary.owner()), 0).unwrap();
    let unconstrained = full
        .identity()
        .dirichlet(
            cochain(full, vec![3.0, 2.0, 1.0]),
            cochain(Binary64Space::selected(empty).unwrap(), Vec::new()),
        )
        .unwrap();
    let prepared = unconstrained
        .prepare(Policy::new(Executor::sequential(), STORAGE, WORK))
        .unwrap();
    let mut workspace = prepared.workspace_for(&unconstrained).unwrap();
    assert_eq!(
        prepared
            .solve(&unconstrained, &mut workspace)
            .unwrap()
            .value()
            .coefficients(),
        &[3.0, 2.0, 1.0]
    );
}

#[test]
fn reuse_keys_resources_and_cancellation_are_checked_before_publication() {
    let owner = triangle();
    let full = Binary64Space::<Cochain>::full(Arc::clone(&owner), 0).unwrap();
    let selection = Arc::new(owner.selection(0, vec![0]).unwrap());
    let selected = Binary64Space::selected(Arc::clone(&selection)).unwrap();
    let operator = full.identity();
    let first = operator
        .dirichlet(
            cochain(full.clone(), vec![1.0, 2.0, 3.0]),
            cochain(selected.clone(), vec![4.0]),
        )
        .unwrap();
    assert_eq!(
        first
            .prepare(Policy::new(
                Executor::sequential(),
                StorageLimit::new(0, 0).unwrap(),
                WORK,
            ))
            .unwrap_err()
            .reason(),
        "resource_limit"
    );
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert_eq!(
        first
            .prepare_cancellable(
                Policy::new(Executor::sequential(), STORAGE, WORK),
                &cancelled,
            )
            .unwrap_err()
            .reason(),
        "cancelled"
    );
    let prepared = first
        .prepare(Policy::new(Executor::sequential(), STORAGE, WORK))
        .unwrap();
    let changed_values = operator
        .dirichlet(
            cochain(full.clone(), vec![9.0, 8.0, 7.0]),
            cochain(selected, vec![-1.0]),
        )
        .unwrap();
    let mut workspace = prepared.workspace_for(&changed_values).unwrap();
    prepared.solve(&changed_values, &mut workspace).unwrap();
    let replacement = Arc::new(owner.selection(0, vec![0]).unwrap());
    let distinct_selection = operator
        .dirichlet(
            cochain(full, vec![9.0, 8.0, 7.0]),
            cochain(Binary64Space::selected(replacement).unwrap(), vec![-1.0]),
        )
        .unwrap();
    assert_eq!(
        prepared
            .workspace_for(&distinct_selection)
            .unwrap_err()
            .reason(),
        "problem_mismatch"
    );
}
