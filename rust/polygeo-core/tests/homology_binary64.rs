use std::sync::Arc;

use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;
use polygeo_core::{
    Binary64Element, Binary64Space, CandidateInput, Chain, Cochain, ComplexCore, HomologyLimit,
    IntegralHomology,
};

fn circle() -> Arc<ComplexCore> {
    ComplexCore::admit(CandidateInput::signed([0, 1, 1, 2, 0, 2], 3, 2, Some(3)).unwrap()).unwrap()
}

fn projective_plane() -> Arc<ComplexCore> {
    let faces = [
        0, 1, 2, 0, 1, 3, 0, 2, 4, 0, 3, 5, 0, 4, 5, 1, 2, 5, 1, 3, 4, 1, 4, 5, 2, 3, 4, 2, 3, 5,
    ];
    ComplexCore::admit(CandidateInput::signed(faces, 10, 3, Some(6)).unwrap()).unwrap()
}

#[test]
fn free_cycles_project_explicitly_and_periods_use_the_canonical_pairing() {
    let owner = circle();
    let chain = owner.chain_complex();
    let homology = IntegralHomology::prepare(&chain, [1], HomologyLimit::DEFAULT).unwrap();
    let group = homology.group(1).unwrap();

    let cycles = group.realize_free_cycles_binary64().unwrap();
    assert_eq!(cycles.len(), group.free_rank());
    assert!(
        cycles[0]
            .space()
            .same_space(&Binary64Space::<Chain>::from_basis(
                &chain.space(1).unwrap()
            ))
    );
    let exact_cycle = group.free_cycle(0).unwrap();
    for (&index, coefficient) in exact_cycle.indices().iter().zip(exact_cycle.coefficients()) {
        assert_eq!(
            cycles[0].coefficients()[index].to_bits(),
            coefficient.to_f64().unwrap().to_bits()
        );
    }
    let repeated = group.realize_free_cycles_binary64().unwrap();
    assert!(!std::ptr::eq(
        cycles[0].coefficients(),
        repeated[0].coefficients()
    ));

    let cochain = Binary64Element::admit(
        Binary64Space::<Cochain>::full(Arc::clone(&owner), 1).unwrap(),
        vec![1.0, 2.0, 4.0],
    )
    .unwrap();
    let periods = group.periods_binary64(&cochain).unwrap();
    let exact = group.free_cycle(0).unwrap();
    let expected = exact
        .indices()
        .iter()
        .zip(exact.coefficients())
        .map(|(&index, coefficient)| coefficient.to_f64().unwrap() * cochain.coefficients()[index])
        .sum::<f64>();
    assert_eq!(periods.as_ref(), &[expected]);
}

#[test]
fn integral_realization_rejects_rounding_an_exact_coefficient() {
    let owner = circle();
    let chain = owner.chain_complex();
    let exact_space = chain.space(1).unwrap();
    let exact = exact_space
        .element([(0, (BigInt::from(1_u64) << 53) + 1_u8)])
        .unwrap();
    let binary64 = Binary64Space::<Chain>::from_basis(&exact_space);
    assert_eq!(
        Binary64Element::realize_integral(binary64, &exact)
            .unwrap_err()
            .reason(),
        "scalar_conversion"
    );

    let exact_power = exact_space
        .element([(0, BigInt::from(1_u8) << 60)])
        .unwrap();
    let realized = Binary64Element::realize_integral(
        Binary64Space::<Chain>::from_basis(&exact_space),
        &exact_power,
    )
    .unwrap();
    assert_eq!(
        realized.coefficients()[0].to_bits(),
        2.0_f64.powi(60).to_bits()
    );
}

#[test]
fn periods_reject_foreign_spaces_and_exact_overflow() {
    let owner = circle();
    let chain = owner.chain_complex();
    let homology = IntegralHomology::prepare(&chain, [1], HomologyLimit::DEFAULT).unwrap();
    let group = homology.group(1).unwrap();

    let foreign = circle();
    let foreign_cochain = Binary64Element::admit(
        Binary64Space::<Cochain>::full(foreign, 1).unwrap(),
        vec![0.0; 3],
    )
    .unwrap();
    assert_eq!(
        group
            .periods_binary64(&foreign_cochain)
            .unwrap_err()
            .reason(),
        "space_mismatch"
    );

    let wrong_degree = Binary64Element::admit(
        Binary64Space::<Cochain>::full(Arc::clone(&owner), 0).unwrap(),
        vec![0.0; 3],
    )
    .unwrap();
    assert_eq!(
        group.periods_binary64(&wrong_degree).unwrap_err().reason(),
        "space_mismatch"
    );

    let mut values = vec![0.0; 3];
    let cycle = group.free_cycle(0).unwrap();
    for (&index, coefficient) in cycle.indices().iter().zip(cycle.coefficients()) {
        values[index] = if coefficient.sign() == Sign::Minus {
            -f64::MAX
        } else {
            f64::MAX
        };
    }
    let overflowing =
        Binary64Element::admit(Binary64Space::<Cochain>::full(owner, 1).unwrap(), values).unwrap();
    assert_eq!(
        group.periods_binary64(&overflowing).unwrap_err().reason(),
        "scalar_conversion"
    );
}

#[test]
fn torsion_remains_exact_and_is_not_projected_as_a_zero_generator() {
    let owner = projective_plane();
    let chain = owner.chain_complex();
    let homology = IntegralHomology::prepare(&chain, [1], HomologyLimit::DEFAULT).unwrap();
    let group = homology.group(1).unwrap();
    assert_eq!(group.torsion_orders(), &[BigInt::from(2)]);
    assert!(group.realize_free_cycles_binary64().unwrap().is_empty());

    let cochain = Binary64Element::admit(
        Binary64Space::<Cochain>::full(owner, 1).unwrap(),
        vec![0.0; chain.space(1).unwrap().basis_size()],
    )
    .unwrap();
    assert!(group.periods_binary64(&cochain).unwrap().is_empty());
    assert_eq!(group.torsion_orders(), &[BigInt::from(2)]);

    let foreign = projective_plane();
    let foreign_cochain = Binary64Element::admit(
        Binary64Space::<Cochain>::full(foreign, 1).unwrap(),
        vec![0.0; chain.space(1).unwrap().basis_size()],
    )
    .unwrap();
    assert_eq!(
        group
            .periods_binary64(&foreign_cochain)
            .unwrap_err()
            .reason(),
        "space_mismatch"
    );
}
