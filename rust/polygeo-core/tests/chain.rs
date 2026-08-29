#[allow(dead_code)]
mod common;

use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::Zero;
use polygeo_core::ExactRational;
use polygeo_core::{
    BigIntEncoding, CandidateInput, CoefficientSlice, CoefficientSystem, ComplexCore,
    CsrBuildLimit, CsrRepresentation, EuclideanDomain, Field, FractionField, FractionFieldOf,
    HalfedgeSurfaceCore, IntegerRing, PresentationError, RationalField, ReducedFractionEncoding,
    Ring, RingMorphism, WorkLimit, compose,
};

fn triangle() -> Arc<ComplexCore> {
    let candidate = CandidateInput::signed([0_i64, 1, 2], 1, 3, Some(3)).unwrap();
    ComplexCore::admit(candidate).unwrap()
}

fn simplex(dimension: usize) -> Arc<ComplexCore> {
    let width = dimension + 1;
    let candidate = CandidateInput::signed(
        (0..width).map(|index| i64::try_from(index).unwrap()),
        1,
        width,
        Some(width),
    )
    .unwrap();
    ComplexCore::admit(candidate).unwrap()
}

#[test]
fn rational_field_is_normalized_and_canonically_contains_integers() {
    fn accepts_euclidean_domain<A: EuclideanDomain>(_algebra: &A) {}
    fn accepts_fraction_field<Base, Extension>(_base: &Base, _extension: &Extension)
    where
        Base: EuclideanDomain,
        Extension: FractionFieldOf<Base>,
    {
    }

    let integers = IntegerRing;
    let rationals = FractionField::new(integers);
    let same_rationals = FractionField::new(IntegerRing);

    accepts_euclidean_domain(&integers);
    accepts_fraction_field(&integers, &rationals);
    assert!(rationals.same_system(&same_rationals));
    assert!(rationals.base_ring().same_system(&integers));
    assert_eq!(
        integers.quotient_remainder(&BigInt::from(-17), &BigInt::from(5)),
        Some((BigInt::from(-4), BigInt::from(3)))
    );
    assert_eq!(
        integers.quotient_remainder(&BigInt::from(-17), &BigInt::from(-5)),
        Some((BigInt::from(4), BigInt::from(3)))
    );
    assert_eq!(
        integers.quotient_remainder(&BigInt::from(1), &BigInt::zero()),
        None
    );
    assert_eq!(
        integers.gcd(&BigInt::from(-18), &BigInt::from(24)),
        BigInt::from(6)
    );

    let six = rationals.inject(&BigInt::from(6));
    let eight = rationals.inject(&BigInt::from(8));
    assert_eq!(
        rationals.divide(&rationals.negate(&six), &eight),
        Some(ExactRational::new(BigInt::from(-3), BigInt::from(4)))
    );
    assert_eq!(rationals.inverse(&rationals.zero()), None);
}

#[test]
fn lazy_rational_base_change_commutes_with_chain_actions_and_csr() {
    let owner = triangle();
    let integers = owner.chain_complex();
    let rationals = RationalField::new(IntegerRing);
    let rational_complex = integers.over(rationals);
    let integral_boundary = integers.boundary(2).unwrap();
    let rational_boundary = integral_boundary.over(rationals);

    let changed_space = integers.space(2).unwrap().over(rationals);
    assert!(
        changed_space
            .identity()
            .same_identity(&rational_complex.space(2).unwrap().identity())
    );

    let huge = BigInt::from(1_u8) << 220_usize;
    let integral = integers
        .space(2)
        .unwrap()
        .element([(0, huge.clone())])
        .unwrap();
    let rational = integral.over(rationals);
    assert_eq!(
        rational.coefficients(),
        &[ExactRational::from_integer(huge)]
    );
    let changed_after = integral_boundary.apply(&integral).unwrap().over(rationals);
    let applied_after = rational_boundary.apply(&rational).unwrap();
    assert_eq!(changed_after.indices(), applied_after.indices());
    assert_eq!(changed_after.coefficients(), applied_after.coefficients());

    let large_fraction = ExactRational::new(
        BigInt::from(1_u8) << 300_usize,
        (BigInt::from(1_u8) << 257_usize) + BigInt::from(1),
    );
    let fractional = rational_complex
        .space(2)
        .unwrap()
        .element([(0, large_fraction.clone())])
        .unwrap();
    assert_eq!(
        rational_boundary.apply(&fractional).unwrap().coefficients(),
        &[
            large_fraction.clone(),
            -large_fraction.clone(),
            large_fraction,
        ]
    );

    assert!(
        integral_boundary
            .dual()
            .over(rationals)
            .same_identity(&rational_boundary.dual())
    );
    let integral_composite = compose(&integers.boundary(1).unwrap(), &integral_boundary).unwrap();
    let rational_composite = integral_composite.over(rationals);
    assert_eq!(
        rational_composite.execution_steps(),
        integral_composite.execution_steps()
    );
    assert!(
        rational_composite
            .apply(&fractional)
            .unwrap()
            .indices()
            .is_empty()
    );
}

#[test]
fn rational_csr_shares_patterns_not_products() {
    let integers = triangle().chain_complex();
    let rationals = RationalField::new(IntegerRing);
    let integral_boundary = integers.boundary(2).unwrap();
    let rational_boundary = integral_boundary.over(rationals);
    let fractional = integers
        .space(2)
        .unwrap()
        .over(rationals)
        .element([(0, ExactRational::new(BigInt::from(5), BigInt::from(7)))])
        .unwrap();
    let integer_estimate = CsrRepresentation::estimate(&integral_boundary, BigIntEncoding).unwrap();
    let rational_estimate =
        CsrRepresentation::estimate(&rational_boundary, ReducedFractionEncoding).unwrap();
    assert_eq!(rational_estimate.coefficient_bits_bound(), 1);
    let integer_csr = CsrRepresentation::build(
        &integral_boundary,
        BigIntEncoding,
        CsrBuildLimit::for_estimate(integer_estimate),
    )
    .unwrap();
    let rational_csr = CsrRepresentation::build(
        &rational_boundary,
        ReducedFractionEncoding,
        CsrBuildLimit::for_estimate(rational_estimate),
    )
    .unwrap();
    assert_eq!(
        integer_csr.matrix().pattern(),
        rational_csr.matrix().pattern()
    );
    assert!(
        rational_csr
            .coefficients()
            .iter()
            .all(|coefficient| coefficient.denom() == &BigInt::from(1))
    );
    assert_eq!(
        rational_csr.apply(&fractional).unwrap().coefficients(),
        rational_boundary.apply(&fractional).unwrap().coefficients()
    );
    let second = CsrRepresentation::build(
        &rational_boundary,
        ReducedFractionEncoding,
        CsrBuildLimit::for_estimate(rational_estimate),
    )
    .unwrap();
    assert_ne!(
        rational_csr.row_offsets().as_ptr(),
        second.row_offsets().as_ptr()
    );
}

#[test]
fn rational_base_change_preserves_halfedge_signed_and_empty_recipes() {
    let rationals = RationalField::new(IntegerRing);
    let surface = HalfedgeSurfaceCore::admit(common::polygon_disk(3)).unwrap();
    let halfedge = surface.chain_complex();
    let halfedge_boundary = halfedge.boundary(2).unwrap();
    let face = halfedge
        .space(2)
        .unwrap()
        .element([(0, BigInt::from(-11))])
        .unwrap();
    let changed = halfedge_boundary.apply(&face).unwrap().over(rationals);
    let applied = halfedge_boundary
        .over(rationals)
        .apply(&face.over(rationals))
        .unwrap();
    assert_eq!(changed.indices(), applied.indices());
    assert_eq!(changed.coefficients(), applied.coefficients());

    let (_simplicial, correspondence) = surface.to_complex().unwrap();
    let signed = correspondence.isomorphism().forward(2).unwrap();
    let signed_value = signed.source().element([(0, BigInt::from(13))]).unwrap();
    let changed = signed.apply(&signed_value).unwrap().over(rationals);
    let applied = signed
        .over(rationals)
        .apply(&signed_value.over(rationals))
        .unwrap();
    assert_eq!(changed.indices(), applied.indices());
    assert_eq!(changed.coefficients(), applied.coefficients());

    let empty = HalfedgeSurfaceCore::admit(common::empty_surface())
        .unwrap()
        .chain_complex()
        .over(rationals);
    let zero = empty.space(0).unwrap().element(std::iter::empty()).unwrap();
    assert!(
        empty
            .boundary(0)
            .unwrap()
            .apply(&zero)
            .unwrap()
            .indices()
            .is_empty()
    );
}

#[test]
fn spaces_admit_one_canonical_exact_sparse_carrier() {
    let owner = triangle();
    let complex = owner.chain_complex();
    let space = complex.space(1).unwrap();
    assert_eq!(space.variance(), "chain");
    assert_eq!(complex.dual().space(1).unwrap().variance(), "cochain");
    assert_eq!(complex.space(3).unwrap_err().reason(), "degree_outside");
    let huge = BigInt::from(1_u8) << 200_usize;

    let chain = space
        .element([
            (2, BigInt::from(5)),
            (0, huge.clone()),
            (2, BigInt::from(-2)),
            (1, BigInt::zero()),
            (0, -huge),
        ])
        .unwrap();

    assert_eq!(chain.degree(), 1);
    assert_eq!(chain.indices(), &[2]);
    assert_eq!(chain.coefficients(), &[BigInt::from(3)]);

    let empty = space.element(std::iter::empty()).unwrap();
    assert!(empty.indices().is_empty());
    assert!(empty.coefficients().is_empty());
}

#[test]
fn element_admission_rejects_indices_outside_the_induced_basis() {
    let owner = triangle();
    let complex = owner.chain_complex();
    let space = complex.space(2).unwrap();

    let error = space.element([(1, BigInt::from(1))]).unwrap_err();

    assert_eq!(error.reason(), "basis_index_outside");
    assert_eq!(error.index(), Some(1));
    assert_eq!(error.bound(), Some(1));
}

#[test]
fn exact_dual_evaluation_is_bilinear_and_owner_bound() {
    let owner = triangle();
    let complex = owner.chain_complex();
    let chain_space = complex.space(1).unwrap();
    let cochain_space = complex.dual().space(1).unwrap();
    let scale = BigInt::from(1_u8) << 180_usize;

    let chain = chain_space
        .element([(0, scale.clone()), (2, BigInt::from(-3))])
        .unwrap();
    let cochain = cochain_space
        .element([
            (0, BigInt::from(7)),
            (1, BigInt::from(11)),
            (2, scale.clone()),
        ])
        .unwrap();

    let expected = BigInt::from(7) * &scale - BigInt::from(3) * &scale;
    assert_eq!(cochain.evaluate(&chain).unwrap(), expected);

    let doubled_chain = chain_space
        .element([(0, &scale + &scale), (2, BigInt::from(-6))])
        .unwrap();
    assert_eq!(
        cochain.evaluate(&doubled_chain).unwrap(),
        BigInt::from(2) * cochain.evaluate(&chain).unwrap()
    );

    let foreign = triangle().chain_complex();
    let foreign_chain = foreign
        .space(1)
        .unwrap()
        .element([(0, BigInt::from(1))])
        .unwrap();
    assert_eq!(
        cochain.evaluate(&foreign_chain).unwrap_err().reason(),
        "space_mismatch"
    );

    let wrong_degree = complex
        .space(0)
        .unwrap()
        .element([(0, BigInt::from(1))])
        .unwrap();
    assert_eq!(
        complex
            .dual()
            .space(0)
            .unwrap()
            .element([(0, BigInt::from(1))])
            .unwrap()
            .evaluate(&wrong_degree)
            .unwrap(),
        BigInt::from(1)
    );
    assert_eq!(
        cochain.evaluate(&wrong_degree).unwrap_err().reason(),
        "space_mismatch"
    );
}

#[test]
fn dual_projection_borrows_and_elements_retain_one_owner_handle() {
    let owner = HalfedgeSurfaceCore::admit(common::polygon_disk(4)).unwrap();
    let complex = owner.chain_complex();
    let handles_before_dual = Arc::strong_count(&owner);
    let dual = complex.dual();

    assert_eq!(Arc::strong_count(&owner), handles_before_dual);
    assert_eq!(dual.space(2).unwrap().basis_size(), 1);

    let element = complex
        .space(2)
        .unwrap()
        .element([(0, BigInt::from(9))])
        .unwrap();
    drop(complex);
    drop(owner);

    assert_eq!(element.basis_size(), 1);
    assert_eq!(element.coefficients(), &[BigInt::from(9)]);
}

#[test]
fn exact_boundary_maps_cover_zero_endpoint_and_square_to_zero() {
    let owner = triangle();
    let complex = owner.chain_complex();
    let huge = BigInt::from(1_u8) << 170_usize;
    let triangle_chain = complex
        .space(2)
        .unwrap()
        .element([(0, huge.clone())])
        .unwrap();

    let boundary_two = complex.boundary(2).unwrap();
    let boundary_one = complex.boundary(1).unwrap();
    let boundary_zero = complex.boundary(0).unwrap();
    let edge_cycle = boundary_two.apply(&triangle_chain).unwrap();
    let vertex_cycle = boundary_one.apply(&edge_cycle).unwrap();
    let below_zero = boundary_zero.apply(&vertex_cycle).unwrap();

    assert_eq!(boundary_two.source().degree(), 2);
    assert_eq!(boundary_two.target().degree(), 1);
    assert!(vertex_cycle.indices().is_empty());
    assert_eq!(below_zero.degree(), -1);
    assert_eq!(below_zero.basis_size(), 0);
    assert!(below_zero.indices().is_empty());

    let wrong_degree = complex
        .space(1)
        .unwrap()
        .element([(0, BigInt::from(1))])
        .unwrap();
    assert_eq!(
        boundary_two.apply(&wrong_degree).unwrap_err().reason(),
        "space_mismatch"
    );
    let foreign = triangle().chain_complex();
    let foreign_triangle = foreign
        .space(2)
        .unwrap()
        .element([(0, BigInt::from(1))])
        .unwrap();
    assert_eq!(
        boundary_two.apply(&foreign_triangle).unwrap_err().reason(),
        "space_mismatch"
    );
}

#[test]
fn algebraic_dual_reverses_arrows_and_satisfies_the_pairing_law() {
    let owner = triangle();
    let complex = owner.chain_complex();
    let boundary_two = complex.boundary(2).unwrap();
    let coboundary_one = boundary_two.dual();
    let scale = BigInt::from(1_u8) << 160_usize;
    let alpha = complex
        .dual()
        .space(1)
        .unwrap()
        .element([(0, BigInt::from(2)), (2, -scale.clone())])
        .unwrap();
    let triangle = complex.space(2).unwrap().element([(0, scale)]).unwrap();

    let left = coboundary_one
        .apply(&alpha)
        .unwrap()
        .evaluate(&triangle)
        .unwrap();
    let right = alpha
        .evaluate(&boundary_two.apply(&triangle).unwrap())
        .unwrap();

    assert_eq!(coboundary_one.source().degree(), 1);
    assert_eq!(coboundary_one.target().degree(), 2);
    assert_eq!(left, right);
    assert!(boundary_two.same_identity(&coboundary_one.dual()));

    let delta_zero = complex.boundary(1).unwrap().dual();
    let delta_one = complex.boundary(2).unwrap().dual();
    let vertex_cochain = complex
        .dual()
        .space(0)
        .unwrap()
        .element([(0, BigInt::from(3)), (1, BigInt::from(-4))])
        .unwrap();
    assert!(
        delta_one
            .apply(&delta_zero.apply(&vertex_cochain).unwrap())
            .unwrap()
            .indices()
            .is_empty()
    );

    let top_delta = complex.dual().coboundary(2).unwrap();
    let top_cochain = complex
        .dual()
        .space(2)
        .unwrap()
        .element([(0, BigInt::from(5))])
        .unwrap();
    let above_top = top_delta.apply(&top_cochain).unwrap();
    assert_eq!(above_top.degree(), 3);
    assert_eq!(above_top.basis_size(), 0);
    assert!(above_top.indices().is_empty());
}

#[test]
fn halfedge_i64_boundaries_preserve_orientation_and_reuse_incidence() {
    let positive_owner = HalfedgeSurfaceCore::admit(common::unigon()).unwrap();
    let positive = positive_owner.chain_complex();
    let map = positive.boundary(2).unwrap();
    let face = positive
        .space(2)
        .unwrap()
        .element([(0, BigInt::from(13))])
        .unwrap();
    let before = positive_owner
        .chain_view()
        .boundary(2)
        .unwrap()
        .coefficients();
    let before_pointer = match before {
        CoefficientSlice::I64(coefficients) => coefficients.as_ptr(),
        CoefficientSlice::I8(_) => unreachable!(),
    };

    assert_eq!(
        map.apply(&face).unwrap().coefficients(),
        &[BigInt::from(13)]
    );
    assert_eq!(
        map.apply(&face).unwrap().coefficients(),
        &[BigInt::from(13)]
    );
    let after = positive_owner
        .chain_view()
        .boundary(2)
        .unwrap()
        .coefficients();
    let after_pointer = match after {
        CoefficientSlice::I64(coefficients) => coefficients.as_ptr(),
        CoefficientSlice::I8(_) => unreachable!(),
    };
    assert_eq!(before_pointer, after_pointer);

    let negative_owner =
        HalfedgeSurfaceCore::admit(common::input(vec![0, 1], vec![1, 0], vec![0])).unwrap();
    let negative = negative_owner.chain_complex();
    let face = negative
        .space(2)
        .unwrap()
        .element([(0, BigInt::from(13))])
        .unwrap();
    assert_eq!(
        negative
            .boundary(2)
            .unwrap()
            .apply(&face)
            .unwrap()
            .coefficients(),
        &[BigInt::from(-13)]
    );
}

#[test]
fn identity_map_reuses_the_same_mathematical_endpoints() {
    let owner = triangle();
    let complex = owner.chain_complex();
    let space = complex.space(1).unwrap();
    let identity = space.identity();
    let chain = space
        .element([(2, BigInt::from(8)), (0, BigInt::from(-2))])
        .unwrap();

    let result = identity.apply(&chain).unwrap();

    assert_eq!(result.indices(), chain.indices());
    assert_eq!(result.coefficients(), chain.coefficients());
    assert_eq!(result.indices().as_ptr(), chain.indices().as_ptr());
    assert_eq!(
        result.coefficients().as_ptr(),
        chain.coefficients().as_ptr()
    );
    assert!(identity.same_identity(&space.identity()));
}

#[test]
fn degree_maps_square_to_zero_in_dimensions_zero_through_five() {
    for dimension in 0..=5 {
        let owner = simplex(dimension);
        let complex = owner.chain_complex();
        for degree in 0..=dimension {
            let space = complex.space(degree).unwrap();
            let value = space
                .element((0..space.basis_size()).map(|index| {
                    let coefficient = BigInt::from(index + 1) << (degree + 70);
                    (index, coefficient)
                }))
                .unwrap();
            let first = complex.boundary(degree).unwrap().apply(&value).unwrap();
            if degree == 0 {
                assert_eq!(first.degree(), -1);
                assert!(first.indices().is_empty());
            } else {
                let second = complex.boundary(degree - 1).unwrap().apply(&first).unwrap();
                assert!(second.indices().is_empty());
            }
        }

        let dual = complex.dual();
        for degree in 0..=dimension {
            let space = dual.space(degree).unwrap();
            let value = space
                .element((0..space.basis_size()).map(|index| {
                    let coefficient = BigInt::from(index + 3) << (degree + 75);
                    (index, coefficient)
                }))
                .unwrap();
            let first = dual.coboundary(degree).unwrap().apply(&value).unwrap();
            if degree == dimension {
                assert_eq!(first.degree(), isize::try_from(dimension + 1).unwrap());
                assert!(first.indices().is_empty());
            } else {
                let second = dual.coboundary(degree + 1).unwrap().apply(&first).unwrap();
                assert!(second.indices().is_empty());
            }
        }
    }
}

#[test]
fn presentation_equality_is_budgeted_and_enables_explicit_transport() {
    let left_owner = triangle();
    let right_owner = triangle();
    let left = left_owner.chain_complex();
    let right = right_owner.chain_complex();

    assert!(!left.same_owner(&right));
    let witness = left
        .identify_presentation(&right, WorkLimit::new(10_000))
        .unwrap();
    let value = left
        .space(1)
        .unwrap()
        .element([(0, BigInt::from(7)), (2, BigInt::from(-9))])
        .unwrap();
    let transported = witness.forward(&value).unwrap();

    assert_eq!(transported.indices(), value.indices());
    assert_eq!(transported.coefficients(), value.coefficients());
    assert_eq!(transported.indices().as_ptr(), value.indices().as_ptr());
    assert_eq!(
        transported.coefficients().as_ptr(),
        value.coefficients().as_ptr()
    );
    assert_eq!(
        witness.inverse(&transported).unwrap().indices(),
        value.indices()
    );
    assert_eq!(
        left.identify_presentation(&right, WorkLimit::new(0))
            .unwrap_err(),
        PresentationError::ComparisonSteps {
            required: 1,
            limit: 0,
        }
    );

    let different = simplex(1).chain_complex();
    assert_eq!(
        left.identify_presentation(&different, WorkLimit::new(10_000))
            .unwrap_err(),
        PresentationError::Mismatch
    );
    let surface = HalfedgeSurfaceCore::admit(common::polygon_disk(3))
        .unwrap()
        .chain_complex();
    assert_eq!(
        left.identify_presentation(&surface, WorkLimit::new(10_000))
            .unwrap_err(),
        PresentationError::NotComparable
    );

    let same_owner = left
        .identify_presentation(&left, WorkLimit::new(0))
        .unwrap();
    assert!(same_owner.forward(&value).is_ok());
}

#[test]
fn halfedge_presentations_compare_exactly_without_owner_cache() {
    let left = HalfedgeSurfaceCore::admit(common::polygon_disk(4))
        .unwrap()
        .chain_complex();
    let right = HalfedgeSurfaceCore::admit(common::polygon_disk(4))
        .unwrap()
        .chain_complex();
    let different = HalfedgeSurfaceCore::admit(common::annulus())
        .unwrap()
        .chain_complex();

    assert!(
        left.identify_presentation(&right, WorkLimit::new(50_000))
            .is_ok()
    );
    assert_eq!(
        left.identify_presentation(&different, WorkLimit::new(50_000))
            .unwrap_err(),
        PresentationError::Mismatch
    );
    assert!(
        left.identify_presentation(&right, WorkLimit::new(1))
            .is_err()
    );
}

#[test]
fn basis_identification_is_explicit_and_invertible() {
    let owner = triangle();
    let complex = owner.chain_complex();
    let space = complex.space(1).unwrap();
    let identification = space.basis_identification();
    let chain = space
        .element([(0, BigInt::from(4)), (2, BigInt::from(-5))])
        .unwrap();
    let cochain = identification.forward(&chain).unwrap();

    assert_eq!(cochain.degree(), chain.degree());
    assert_eq!(cochain.indices(), chain.indices());
    assert_eq!(cochain.coefficients(), chain.coefficients());
    assert_eq!(cochain.indices().as_ptr(), chain.indices().as_ptr());
    assert_eq!(
        cochain.coefficients().as_ptr(),
        chain.coefficients().as_ptr()
    );
    assert_eq!(
        identification.inverse(&cochain).unwrap().coefficients(),
        chain.coefficients()
    );
    assert_eq!(cochain.evaluate(&chain).unwrap(), BigInt::from(41));

    let foreign = triangle().chain_complex();
    let foreign_chain = foreign
        .space(1)
        .unwrap()
        .element([(0, BigInt::from(1))])
        .unwrap();
    assert_eq!(
        identification.forward(&foreign_chain).unwrap_err().reason(),
        "space_mismatch"
    );
}

#[test]
fn composition_is_flat_owner_checked_and_matches_sequential_action() {
    let owner = triangle();
    let complex = owner.chain_complex();
    let boundary_two = complex.boundary(2).unwrap();
    let boundary_one = complex.boundary(1).unwrap();
    let value = complex
        .space(2)
        .unwrap()
        .element([(0, BigInt::from(1_u8) << 240_usize)])
        .unwrap();

    let composite = compose(&boundary_one, &boundary_two).unwrap();
    assert_eq!(composite.execution_steps(), 2);
    let sequential = boundary_one
        .apply(&boundary_two.apply(&value).unwrap())
        .unwrap();
    let direct = composite.apply(&value).unwrap();
    assert_eq!(direct.indices(), sequential.indices());
    assert_eq!(direct.coefficients(), sequential.coefficients());

    let normalized = compose(&complex.space(0).unwrap().identity(), &composite).unwrap();
    assert_eq!(normalized.execution_steps(), 2);
    assert_eq!(
        normalized.apply(&value).unwrap().coefficients(),
        direct.coefficients()
    );

    let foreign = triangle().chain_complex().boundary(1).unwrap();
    assert_eq!(
        compose(&foreign, &boundary_two).unwrap_err().reason(),
        "space_mismatch"
    );
}

#[test]
fn explicit_integer_csr_is_deterministic_budgeted_and_map_bound() {
    let owner = triangle();
    let complex = owner.chain_complex();
    let map = complex.boundary(2).unwrap();
    let estimate = CsrRepresentation::estimate(&map, BigIntEncoding).unwrap();

    assert_eq!(estimate.shape(), (3, 1));
    assert_eq!(estimate.nnz_bound(), 3);
    assert!(estimate.retained_logical_bytes_bound() > 0);
    assert!(estimate.scalar_steps_bound() >= estimate.nnz_bound() as u64);
    assert_eq!(estimate.coefficient_bits_bound(), 1);
    assert!(!estimate.canonicalization_required());
    assert_eq!(
        estimate,
        CsrRepresentation::estimate(&map, BigIntEncoding).unwrap()
    );

    let representation =
        CsrRepresentation::build(&map, BigIntEncoding, CsrBuildLimit::for_estimate(estimate))
            .unwrap();
    assert!(representation.represented_map().same_identity(&map));
    assert_eq!(representation.shape(), (3, 1));
    assert_eq!(representation.row_offsets(), &[0, 1, 2, 3]);
    assert_eq!(representation.column_indices(), &[0, 0, 0]);
    assert_eq!(
        representation.coefficients(),
        &[BigInt::from(1), BigInt::from(-1), BigInt::from(1)]
    );

    let huge = BigInt::from(1_u8) << 300_usize;
    let value = complex.space(2).unwrap().element([(0, huge)]).unwrap();
    let direct = map.apply(&value).unwrap();
    for _ in 0..2 {
        let represented = representation.apply(&value).unwrap();
        assert_eq!(represented.indices(), direct.indices());
        assert_eq!(represented.coefficients(), direct.coefficients());
    }

    let second =
        CsrRepresentation::build(&map, BigIntEncoding, CsrBuildLimit::for_estimate(estimate))
            .unwrap();
    assert_ne!(
        representation.row_offsets().as_ptr(),
        second.row_offsets().as_ptr()
    );
    let shared = representation.clone();
    assert_eq!(
        representation.row_offsets().as_ptr(),
        shared.row_offsets().as_ptr()
    );
    assert_eq!(
        representation.coefficients().as_ptr(),
        shared.coefficients().as_ptr()
    );
}

#[test]
fn csr_of_dual_is_the_canonical_transpose() {
    let owner = triangle();
    let complex = owner.chain_complex();
    let primal = complex.boundary(2).unwrap();
    let dual = primal.dual();
    let estimate = CsrRepresentation::estimate(&dual, BigIntEncoding).unwrap();
    assert!(estimate.scratch_entries_bound() > 0);
    let representation =
        CsrRepresentation::build(&dual, BigIntEncoding, CsrBuildLimit::for_estimate(estimate))
            .unwrap();

    assert_eq!(representation.shape(), (1, 3));
    assert_eq!(representation.row_offsets(), &[0, 3]);
    assert_eq!(representation.column_indices(), &[0, 1, 2]);
    assert_eq!(
        representation.coefficients(),
        &[BigInt::from(1), BigInt::from(-1), BigInt::from(1)]
    );
}
