use std::sync::Arc;

use num_bigint::BigInt;
use polygeo_core::{
    Binary64Chain, Binary64ChainSpace, Binary64Cochain, Binary64CochainSpace, Binary64Element,
    Binary64ElementError, Binary64Space, CandidateInput, ComplexCore, HalfedgeInput,
    HalfedgeSurfaceCore, OperatorError,
};

fn triangle() -> Arc<ComplexCore> {
    ComplexCore::admit(CandidateInput::signed([0, 1, 2], 1, 3, None).unwrap()).unwrap()
}

#[test]
fn one_variance_indexed_carrier_realizes_chains_and_cochains() {
    let owner = triangle();
    let chain_space: Binary64ChainSpace = Binary64Space::full(Arc::clone(&owner), 0).unwrap();
    let cochain_space: Binary64CochainSpace = Binary64Space::full(Arc::clone(&owner), 0).unwrap();
    let exact_chain = owner
        .chain_complex()
        .space(0)
        .unwrap()
        .element([(1, BigInt::from(3))])
        .unwrap();
    let exact_cochain = owner
        .chain_complex()
        .dual()
        .space(0)
        .unwrap()
        .element([(1, BigInt::from(5))])
        .unwrap();

    let chain: Binary64Chain =
        Binary64Element::realize_integral(chain_space, &exact_chain).unwrap();
    let cochain: Binary64Cochain =
        Binary64Element::realize_integral(cochain_space, &exact_cochain).unwrap();

    assert_eq!(chain.space().variance(), "chain");
    assert_eq!(cochain.space().variance(), "cochain");
    assert_eq!(chain.coefficients(), &[0.0, 3.0, 0.0]);
    assert_eq!(cochain.coefficients(), &[0.0, 5.0, 0.0]);
}

#[test]
fn integral_realization_rejects_foreign_degree_owner_and_selection() {
    let owner = triangle();
    let foreign = triangle();
    let exact = owner
        .chain_complex()
        .space(0)
        .unwrap()
        .element([(0, BigInt::from(1))])
        .unwrap();
    let wrong_degree: Binary64ChainSpace = Binary64Space::full(Arc::clone(&owner), 1).unwrap();
    let wrong_owner: Binary64ChainSpace = Binary64Space::full(foreign, 0).unwrap();
    let selection = Arc::new(owner.selection(0, vec![0]).unwrap());
    let selected: Binary64ChainSpace = Binary64Space::selected(selection).unwrap();

    for space in [wrong_degree, wrong_owner, selected] {
        assert_eq!(
            Binary64Element::realize_integral(space, &exact)
                .unwrap_err()
                .reason(),
            "space_mismatch"
        );
    }
}

#[test]
fn derivative_squares_to_zero_in_the_retained_basis() {
    let value = Binary64Cochain::admit(
        Binary64CochainSpace::full(triangle(), 0).unwrap(),
        vec![2.0, 5.0, 11.0],
    )
    .unwrap();
    let derivative = value.exterior_derivative().unwrap();
    assert_eq!(derivative.coefficients(), &[3.0, 9.0, 6.0]);
    assert_eq!(
        derivative.exterior_derivative().unwrap().coefficients(),
        &[0.0]
    );
}

#[test]
fn equal_sized_selections_remain_distinct_basis_handles() {
    let owner = triangle();
    let left = Arc::new(owner.selection(0, vec![0, 1]).unwrap());
    let right = Arc::new(owner.selection(0, vec![1, 2]).unwrap());
    let left = Binary64CochainSpace::selected(left).unwrap();
    let right = Binary64CochainSpace::selected(right).unwrap();
    assert!(!left.same_space(&right));
}

#[test]
fn halfedge_binary64_cochains_share_the_chain_domain_differential() {
    let input = HalfedgeInput::unsigned(vec![1, 2, 0, 5, 3, 4], vec![3, 4, 5, 0, 1, 2], vec![3], 6)
        .unwrap();
    let owner = HalfedgeSurfaceCore::admit(input).unwrap();
    let exact = owner.chain_complex().dual().space(0).unwrap();
    let space = Binary64CochainSpace::from_basis(&exact);
    let value = Binary64Cochain::admit(space, vec![2.0, 5.0, 11.0]).unwrap();

    let derivative: Result<Binary64Cochain, OperatorError> = value.exterior_derivative();
    let second = derivative.unwrap().exterior_derivative().unwrap();
    assert!(second.coefficients().iter().all(|&value| value == 0.0));

    let error = Binary64Cochain::admit(Binary64CochainSpace::from_basis(&exact), vec![f64::NAN; 3])
        .unwrap_err();
    assert_eq!(error, Binary64ElementError::NonFinite);
}

#[test]
fn element_admission_and_operator_failures_remain_distinct() {
    let owner = triangle();
    let exact_space = owner.chain_complex().dual().space(0).unwrap();
    let exact = exact_space
        .element([(0, BigInt::from(1u8) << 2048usize)])
        .unwrap();
    let space = Binary64CochainSpace::from_basis(&exact_space);
    assert_eq!(
        Binary64Cochain::realize_integral(space, &exact).unwrap_err(),
        Binary64ElementError::ScalarConversion
    );

    let selection = Arc::new(owner.selection(0, vec![0]).unwrap());
    let selected = Binary64Cochain::admit(
        Binary64CochainSpace::selected(selection).unwrap(),
        vec![1.0],
    )
    .unwrap();
    assert_eq!(
        selected.exterior_derivative().unwrap_err(),
        OperatorError::FullSpaceRequired
    );
}

#[test]
fn zero_space_cannot_alias_a_represented_full_degree() {
    let owner = triangle();
    assert_eq!(
        Binary64CochainSpace::zero(owner, 0).unwrap_err().reason(),
        "degree_outside"
    );
}
