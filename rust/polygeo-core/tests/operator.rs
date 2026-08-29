use std::sync::Arc;

use polygeo_core::{
    Binary64Chain, Binary64Cochain, Binary64CochainSpace, Binary64Element, Binary64Space,
    CandidateInput, Cochain, ComplexCore, EuclideanRealization, NondegenerateCapability,
    PairingCapability, PositiveMetric, RealizationLimit,
    operator::{LinearOperator, OperatorError, compose},
};

fn triangle() -> Arc<ComplexCore> {
    ComplexCore::admit(CandidateInput::signed([0, 1, 2], 1, 3, None).unwrap()).unwrap()
}

fn metric() -> PositiveMetric {
    metric_scaled(1.0)
}

fn metric_scaled(scale: f64) -> PositiveMetric {
    let height = 3.0_f64.sqrt() / 2.0;
    EuclideanRealization::admit(
        triangle(),
        2,
        vec![0.0, 0.0, scale, 0.0, 0.5 * scale, height * scale],
        RealizationLimit::DEFAULT,
    )
    .unwrap()
    .circumcentric_pairing()
    .unwrap()
    .require_positive()
    .unwrap()
}

fn cochain(space: Binary64CochainSpace, coefficients: Vec<f64>) -> Binary64Cochain {
    Binary64Element::admit(space, coefficients).unwrap()
}

fn simplex(dimension: usize) -> Arc<ComplexCore> {
    ComplexCore::admit(
        CandidateInput::signed(
            (0..=dimension).map(|vertex| i64::try_from(vertex).unwrap()),
            1,
            dimension + 1,
            Some(dimension + 1),
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn differential_is_one_operator_authority_and_value_tail_call() {
    let space = Binary64Space::full(triangle(), 0).unwrap();
    let value = cochain(space.clone(), vec![2.0, 5.0, 11.0]);
    let differential: LinearOperator<Cochain, Cochain> = space.exterior_derivative().unwrap();

    assert_eq!(
        differential.apply(&value).unwrap().coefficients(),
        &[3.0, 9.0, 6.0]
    );
    assert_eq!(
        value.exterior_derivative().unwrap().coefficients(),
        differential.apply(&value).unwrap().coefficients()
    );
    assert!(differential.same_identity(&differential.clone()));
    assert!(differential.same_identity(&space.exterior_derivative().unwrap()));

    let second = differential.target().exterior_derivative().unwrap();
    assert_eq!(
        compose(&second, &differential)
            .unwrap()
            .apply(&value)
            .unwrap()
            .coefficients(),
        &[0.0]
    );
}

#[test]
fn differential_squares_to_zero_through_dimension_five() {
    for dimension in 0..=5 {
        let owner = simplex(dimension);
        for degree in 0..=dimension {
            let space = Binary64Space::full(Arc::clone(&owner), degree).unwrap();
            let value = cochain(space.clone(), vec![1.0; space.size()]);
            let first = space.exterior_derivative().unwrap();
            let second = first.target().exterior_derivative().unwrap();
            assert!(
                compose(&second, &first)
                    .unwrap()
                    .apply(&value)
                    .unwrap()
                    .coefficients()
                    .iter()
                    .all(|coefficient| *coefficient == 0.0)
            );
        }
    }
}

#[test]
fn riesz_inverse_and_codifferential_obey_pairing_laws() {
    let metric = metric();
    let owner = Arc::clone(metric.realization().topology());
    let c0 = Binary64Space::full(Arc::clone(&owner), 0).unwrap();
    let c1 = Binary64Space::full(owner, 1).unwrap();
    let x = cochain(c0.clone(), vec![1.0, -2.0, 4.0]);
    let y = cochain(c1.clone(), vec![3.0, 5.0, -1.0]);

    let riesz = metric.riesz(1).unwrap();
    let represented: Binary64Chain = riesz.apply(&y).unwrap();
    let recovered = metric
        .inverse_riesz(1)
        .unwrap()
        .apply(&represented)
        .unwrap();
    assert_eq!(recovered.coefficients(), y.coefficients());

    let dx = c0.exterior_derivative().unwrap().apply(&x).unwrap();
    let delta_y = metric.codifferential(1).unwrap().apply(&y).unwrap();
    let w0 = metric.hodge_coefficients_slice(0).unwrap();
    let w1 = metric.hodge_coefficients_slice(1).unwrap();
    let left = dot_weighted(dx.coefficients(), y.coefficients(), w1);
    let right = dot_weighted(x.coefficients(), delta_y.coefficients(), w0);
    assert!((left - right).abs() <= 1.0e-12);
}

#[test]
fn laplacian_matches_direct_codifferential_compositions() {
    let metric = metric();
    let owner = Arc::clone(metric.realization().topology());
    let c0 = Binary64Space::full(owner, 0).unwrap();
    let x = cochain(c0.clone(), vec![1.0, -2.0, 4.0]);
    let d = c0.exterior_derivative().unwrap();
    let delta = metric.codifferential(1).unwrap();
    let composed = compose(&delta, &d).unwrap().apply(&x).unwrap();
    let direct = metric.laplacian(0).unwrap().apply(&x).unwrap();

    assert_close(direct.coefficients(), composed.coefficients());
}

#[test]
fn metric_operators_preserve_expected_scale_exponents() {
    let unit = metric_scaled(1.0);
    let scaled = metric_scaled(4.0);
    for degree in 0..=2 {
        let rank = unit
            .realization()
            .topology()
            .chain_view()
            .basis_size(degree)
            .unwrap();
        let unit_space =
            Binary64Space::full(Arc::clone(unit.realization().topology()), degree).unwrap();
        let scaled_space =
            Binary64Space::full(Arc::clone(scaled.realization().topology()), degree).unwrap();
        let unit_value = cochain(unit_space, vec![1.0; rank]);
        let scaled_value = cochain(scaled_space, vec![1.0; rank]);
        let unit_result = unit.riesz(degree).unwrap().apply(&unit_value).unwrap();
        let scaled_result = scaled.riesz(degree).unwrap().apply(&scaled_value).unwrap();
        let expected = 4.0_f64.powi(2 - 2 * i32::try_from(degree).unwrap());
        for (&left, &right) in scaled_result
            .coefficients()
            .iter()
            .zip(unit_result.coefficients())
        {
            assert!((left - expected * right).abs() <= 1.0e-12);
        }
    }
}

#[test]
fn selection_restricts_and_extends_without_implicit_representation() {
    let owner = triangle();
    let selection = Arc::new(owner.selection(0, vec![0, 2]).unwrap());
    let full = Binary64Space::full(Arc::clone(&owner), 0).unwrap();
    let value = cochain(full, vec![2.0, 5.0, 11.0]);
    let restriction = selection.restriction::<Cochain>().unwrap();
    let extension = selection.extension_by_zero::<Cochain>().unwrap();

    let selected = restriction.apply(&value).unwrap();
    assert_eq!(selected.coefficients(), &[2.0, 11.0]);
    assert_eq!(
        extension.apply(&selected).unwrap().coefficients(),
        &[2.0, 0.0, 11.0]
    );
}

#[test]
fn top_degree_and_empty_selection_publish_canonical_zero_values() {
    let owner = triangle();
    let top: Binary64CochainSpace = Binary64Space::full(Arc::clone(&owner), 2).unwrap();
    let top_value = cochain(top.clone(), vec![7.0]);
    let beyond = top
        .exterior_derivative()
        .unwrap()
        .apply(&top_value)
        .unwrap();
    assert_eq!(beyond.space().degree(), 3);
    assert!(beyond.coefficients().is_empty());

    let empty = Arc::new(owner.selection(1, Vec::new()).unwrap());
    let full: Binary64CochainSpace = Binary64Space::full(owner, 1).unwrap();
    let value = cochain(full, vec![1.0, 2.0, 3.0]);
    let restricted = empty
        .restriction::<Cochain>()
        .unwrap()
        .apply(&value)
        .unwrap();
    assert!(restricted.coefficients().is_empty());
    assert_eq!(
        empty
            .extension_by_zero::<Cochain>()
            .unwrap()
            .apply(&restricted)
            .unwrap()
            .coefficients(),
        &[0.0; 3]
    );
    assert_eq!(
        metric().codifferential(0).unwrap_err(),
        OperatorError::DegreeOutside
    );
}

#[test]
fn identity_shares_payload_zero_is_dense_and_foreign_values_fail() {
    let space: Binary64CochainSpace = Binary64Space::full(triangle(), 0).unwrap();
    let target: Binary64CochainSpace = Binary64Space::full(triangle(), 0).unwrap();
    let value = cochain(space.clone(), vec![2.0, 5.0, 11.0]);
    let identity = space.identity();
    let same = identity.apply(&value).unwrap();
    assert_eq!(same.coefficients().as_ptr(), value.coefficients().as_ptr());

    let zero = space.zero_to(&space);
    assert_eq!(zero.apply(&value).unwrap().coefficients(), &[0.0; 3]);
    assert_eq!(
        identity.apply(&cochain(target, vec![1.0; 3])).unwrap_err(),
        OperatorError::SpaceMismatch
    );
}

#[test]
fn composition_rejects_foreign_equal_rank_intermediates() {
    let left: Binary64CochainSpace = Binary64Space::full(triangle(), 0).unwrap();
    let right: Binary64CochainSpace = Binary64Space::full(triangle(), 0).unwrap();

    assert_eq!(
        compose(&right.identity(), &left.identity()).unwrap_err(),
        OperatorError::SpaceMismatch
    );
}

#[test]
fn flat_composition_accepts_64_steps_and_rejects_65() {
    let owner = triangle();
    let selection = Arc::new(owner.selection(0, vec![0, 2]).unwrap());
    let restriction = selection.restriction::<Cochain>().unwrap();
    let extension = selection.extension_by_zero::<Cochain>().unwrap();
    let cycle = compose(&extension, &restriction).unwrap();
    let mut plan = cycle.clone();
    for _ in 1..32 {
        plan = compose(&cycle, &plan).unwrap();
    }
    assert_eq!(plan.execution_steps(), 64);
    let value = cochain(plan.source().clone(), vec![2.0, 5.0, 11.0]);
    assert_eq!(
        plan.apply(&value).unwrap().coefficients(),
        &[2.0, 0.0, 11.0]
    );
    assert_eq!(
        compose(&cycle, &plan).unwrap_err(),
        OperatorError::PlanLimit
    );
}

fn dot_weighted(left: &[f64], right: &[f64], weights: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .zip(weights)
        .map(|((&left, &right), &weight)| left * right * weight)
        .sum()
}

fn assert_close(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (&left, &right) in left.iter().zip(right) {
        assert!((left - right).abs() <= 1.0e-12, "{left} != {right}");
    }
}
