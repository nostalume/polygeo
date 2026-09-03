use num_bigint::BigInt;
use num_traits::Zero;
use polygeo_core::{
    chain::HomologyLimit, chain::IntegralChain, chain::IntegralChainComplex,
    chain::IntegralHomology, solve::StorageLimit, solve::WorkLimit, topology::CandidateInput,
    topology::Complex as ComplexCore, topology::HalfedgeSurface as HalfedgeSurfaceCore,
};
use proptest::prelude::*;
use std::sync::Arc;

fn complex(rows: &[&[i64]]) -> Arc<ComplexCore> {
    let width = rows.first().map_or(0, |row| row.len());
    let values = rows.iter().flat_map(|row| row.iter().copied());
    let vertex_count = rows
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .max()
        .map_or(0, |value| usize::try_from(value + 1).unwrap());
    ComplexCore::admit(
        CandidateInput::signed(values, rows.len(), width, Some(vertex_count)).unwrap(),
    )
    .unwrap()
}

fn analyze(
    owner: &Arc<ComplexCore>,
    degrees: impl IntoIterator<Item = usize>,
) -> (IntegralChainComplex, IntegralHomology) {
    let chain = owner.chain_complex();
    let homology = IntegralHomology::analyze(&chain, degrees, HomologyLimit::DEFAULT).unwrap();
    (chain, homology)
}

fn simplex_boundary(dimension: usize, shift: usize) -> Arc<ComplexCore> {
    let vertex_count = dimension + 2;
    let mut values = Vec::with_capacity(vertex_count * (dimension + 1));
    for omitted in 0..vertex_count {
        let mut facet = (0..vertex_count)
            .filter(|&vertex| vertex != omitted)
            .map(|vertex| i64::try_from((vertex + shift) % vertex_count).unwrap())
            .collect::<Vec<_>>();
        facet.sort_unstable();
        values.extend(facet);
    }
    ComplexCore::admit(
        CandidateInput::signed(values, vertex_count, dimension + 1, Some(vertex_count)).unwrap(),
    )
    .unwrap()
}

fn assert_cycle(chain: &IntegralChainComplex, degree: usize, cycle: &IntegralChain) {
    assert!(
        chain
            .boundary(degree)
            .unwrap()
            .apply(cycle)
            .unwrap()
            .coefficients()
            .is_empty()
    );
}

fn tetrahedron() -> Arc<ComplexCore> {
    let owner = complex(&[&[1, 2, 3], &[0, 3, 2], &[0, 1, 3], &[0, 2, 1]]);
    owner.refine_triangle().unwrap();
    owner.refine_oriented().unwrap();
    owner
}

#[test]
fn ordinary_integral_groups_cover_endpoints_free_cycles_and_duplicates() {
    let point = complex(&[&[0]]);
    let (_, empty) = analyze(&point, []);
    assert!(empty.degrees().is_empty());

    let (point_chain, point_homology) = analyze(&point, [0, 0]);
    assert_eq!(point_homology.degrees(), &[0]);
    let h0 = point_homology.group(0).unwrap();
    assert_eq!(h0.degree(), 0);
    assert_eq!(h0.free_rank(), 1);
    assert!(h0.torsion_orders().is_empty());
    assert_cycle(&point_chain, 0, h0.free_cycle(0).unwrap());

    let interval = complex(&[&[0, 1]]);
    let (_, interval_homology) = analyze(&interval, [0, 1]);
    assert_eq!(interval_homology.group(0).unwrap().free_rank(), 1);
    assert_eq!(interval_homology.group(1).unwrap().free_rank(), 0);

    let circle = complex(&[&[0, 1], &[1, 2], &[0, 2]]);
    let (circle_chain, circle_homology) = analyze(&circle, [1, 0, 1]);
    assert_eq!(circle_homology.degrees(), &[0, 1]);
    let h1 = circle_homology.group(1).unwrap();
    assert_eq!(h1.free_rank(), 1);
    assert_cycle(&circle_chain, 1, h1.free_cycle(0).unwrap());

    let disconnected = complex(&[&[0], &[1]]);
    let (_, disconnected_homology) = analyze(&disconnected, [0]);
    assert_eq!(disconnected_homology.group(0).unwrap().free_rank(), 2);
}

#[test]
fn sphere_and_projective_plane_preserve_exact_representative_relations() {
    let sphere = complex(&[&[0, 1, 2], &[0, 1, 3], &[0, 2, 3], &[1, 2, 3]]);
    let (sphere_chain, sphere_homology) = analyze(&sphere, 0..=2);
    assert_eq!(
        (0..=2)
            .map(|degree| sphere_homology.group(degree).unwrap().free_rank())
            .collect::<Vec<_>>(),
        [1, 0, 1]
    );
    assert_cycle(
        &sphere_chain,
        2,
        sphere_homology.group(2).unwrap().free_cycle(0).unwrap(),
    );

    let projective_plane = complex(&[
        &[0, 1, 2],
        &[0, 1, 3],
        &[0, 2, 4],
        &[0, 3, 5],
        &[0, 4, 5],
        &[1, 2, 5],
        &[1, 3, 4],
        &[1, 4, 5],
        &[2, 3, 4],
        &[2, 3, 5],
    ]);
    let (rp2_chain, rp2_homology) = analyze(&projective_plane, [0, 1, 2]);
    let h1 = rp2_homology.group(1).unwrap();
    assert_eq!(h1.free_rank(), 0);
    assert_eq!(h1.torsion_orders(), &[BigInt::from(2)]);
    let cycle = h1.torsion_cycle(0).unwrap();
    let bound = h1.torsion_bound(0).unwrap();
    assert_cycle(&rp2_chain, 1, cycle);
    let boundary = rp2_chain.boundary(2).unwrap().apply(bound).unwrap();
    let doubled = rp2_chain
        .space(1)
        .unwrap()
        .element(
            cycle
                .indices()
                .iter()
                .copied()
                .zip(cycle.coefficients().iter().map(|value| value * 2)),
        )
        .unwrap();
    assert_eq!(boundary.indices(), doubled.indices());
    assert_eq!(boundary.coefficients(), doubled.coefficients());
}

#[test]
fn invalid_degree_and_each_semantic_ceiling_fail_before_publication() {
    let chain = complex(&[&[0, 1], &[1, 2], &[0, 2]]).chain_complex();
    let invalid = IntegralHomology::analyze(&chain, [2], HomologyLimit::DEFAULT).unwrap_err();
    assert_eq!(invalid.reason(), "degree_outside");

    let defaults = HomologyLimit::DEFAULT;
    assert!(
        defaults.storage().retained_logical_bytes() <= defaults.storage().peak_live_logical_bytes()
    );
    assert!(defaults.smith_steps().steps() > 0);
    for (limit, axis) in [
        (
            defaults.with_storage(
                StorageLimit::new(0, defaults.storage().peak_live_logical_bytes()).unwrap(),
            ),
            "retained_logical_bytes",
        ),
        (
            defaults.with_storage(StorageLimit::new(1, 1).unwrap()),
            "peak_live_logical_bytes",
        ),
        (defaults.with_coefficient_bits(0), "coefficient_bits"),
        (defaults.with_smith_steps(WorkLimit::new(0)), "smith_steps"),
    ] {
        let error = IntegralHomology::analyze(&chain, [0, 1], limit).unwrap_err();
        assert_eq!(error.reason(), "resource_limit");
        let (actual_axis, required, admitted) = error.resource_limit().unwrap();
        assert_eq!(actual_axis, axis);
        assert!(required > admitted);
    }
}

#[test]
fn representatives_remain_bound_to_the_analyzed_owner() {
    let owner = complex(&[&[0, 1], &[1, 2], &[0, 2]]);
    let (chain, homology) = analyze(&owner, [1]);
    let cycle = homology.group(1).unwrap().free_cycle(0).unwrap();
    let foreign = complex(&[&[0, 1], &[1, 2], &[0, 2]])
        .chain_complex()
        .boundary(1)
        .unwrap();

    assert_cycle(&chain, 1, cycle);
    assert_eq!(foreign.apply(cycle).unwrap_err().reason(), "space_mismatch");
    assert!(cycle.coefficients().iter().any(|value| !value.is_zero()));
}

#[test]
fn analysis_rows_are_stable_and_transport_across_checked_correspondences() {
    let source = tetrahedron();
    let source_chain = source.chain_complex();
    let analysis = IntegralHomology::analyze(&source_chain, 0..=2, HomologyLimit::DEFAULT).unwrap();
    assert!(std::ptr::eq(
        analysis.group(2).unwrap().free_cycle(0).unwrap(),
        analysis.group(2).unwrap().free_cycle(0).unwrap(),
    ));

    let (surface, forward) = HalfedgeSurfaceCore::from_complex(&source).unwrap();
    let transported = analysis
        .transport(&forward, HomologyLimit::DEFAULT)
        .unwrap();
    assert!(transported.chain_complex().same_owner(forward.target()));
    assert!(!transported.chain_complex().same_owner(&source_chain));
    for degree in 0..=2 {
        let source_group = analysis.group(degree).unwrap();
        let target_group = transported.group(degree).unwrap();
        assert_eq!(source_group.free_rank(), target_group.free_rank());
        assert_eq!(source_group.torsion_orders(), target_group.torsion_orders());
        for index in 0..target_group.free_rank() {
            assert_cycle(
                transported.chain_complex(),
                degree,
                target_group.free_cycle(index).unwrap(),
            );
        }
    }

    let (_, backward) = surface.to_complex().unwrap();
    let roundtrip = transported
        .transport(&backward, HomologyLimit::DEFAULT)
        .unwrap();
    for degree in 0..=2 {
        assert_eq!(
            roundtrip.group(degree).unwrap().free_rank(),
            analysis.group(degree).unwrap().free_rank()
        );
    }
}

#[test]
fn transport_rejects_foreign_sources_and_exhausted_smith_steps() {
    let source = tetrahedron();
    let (surface, relation) = HalfedgeSurfaceCore::from_complex(&source).unwrap();
    let source_chain = source.chain_complex();
    let analysis =
        IntegralHomology::analyze(&source_chain, [0, 1, 2], HomologyLimit::DEFAULT).unwrap();

    let foreign = tetrahedron();
    let (_, foreign_relation) = HalfedgeSurfaceCore::from_complex(&foreign).unwrap();
    assert_eq!(
        analysis
            .transport(&foreign_relation, HomologyLimit::DEFAULT)
            .unwrap_err()
            .reason(),
        "owner_mismatch"
    );
    let exhausted = analysis
        .transport(
            &relation,
            HomologyLimit::DEFAULT.with_smith_steps(WorkLimit::new(0)),
        )
        .unwrap_err();
    let (axis, required, limit) = exhausted.resource_limit().unwrap();
    assert_eq!(axis, "smith_steps");
    assert!(required > limit);
    assert!(!surface.chain_complex().same_owner(&source_chain));
}

proptest! {
    #[test]
    fn simplex_boundary_spheres_through_degree_four_have_expected_groups(
        dimension in 1_usize..=4,
        shift in 0_usize..6,
    ) {
        let owner = simplex_boundary(dimension, shift);
        let (chain, homology) = analyze(&owner, 0..=dimension);
        for degree in 0..=dimension {
            let expected = usize::from(degree == 0 || degree == dimension);
            let group = homology.group(degree).unwrap();
            prop_assert_eq!(group.free_rank(), expected);
            prop_assert!(group.torsion_orders().is_empty());
            if expected == 1 {
                assert_cycle(&chain, degree, group.free_cycle(0).unwrap());
            }
        }
    }
}
