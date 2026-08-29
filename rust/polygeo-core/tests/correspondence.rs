#[allow(dead_code)]
mod common;

use std::collections::BTreeSet;
use std::sync::Arc;

use num_bigint::BigInt;
use polygeo_core::{
    BigIntEncoding, CandidateInput, ChainLawLimit, ComplexCore, CorrespondenceDirection,
    CsrBuildLimit, CsrRepresentation, HalfedgeSurfaceCore, StorageLimit, WorkLimit, compose,
};

fn oriented_complex(rows: &[[i64; 3]]) -> Arc<ComplexCore> {
    let candidate = CandidateInput::signed(
        rows.iter().flat_map(|row| row.iter().copied()),
        rows.len(),
        3,
        None,
    )
    .unwrap();
    let owner = ComplexCore::admit(candidate).unwrap();
    owner.refine_triangle().unwrap();
    owner.refine_oriented().unwrap();
    owner
}

#[test]
fn conversion_chain_law_is_preflighted_and_retryable() {
    let complex = tetrahedron();
    let no_storage = ChainLawLimit::new(StorageLimit::new(0, 0).unwrap(), WorkLimit::new(0));
    let error = HalfedgeSurfaceCore::from_complex_with_limit(&complex, no_storage).unwrap_err();
    assert_eq!(error.reason(), "resource_limit");
    assert!(
        matches!(error.resource_limit(), Some(("retained_logical_bytes", required, 0)) if required > 0)
    );

    let no_terms = ChainLawLimit::new(
        StorageLimit::new(u64::MAX, u64::MAX).unwrap(),
        WorkLimit::new(0),
    );
    assert!(matches!(
        HalfedgeSurfaceCore::from_complex_with_limit(&complex, no_terms)
            .unwrap_err()
            .resource_limit(),
        Some(("terms", required, 0)) if required > 0
    ));
    let (_, correspondence) = HalfedgeSurfaceCore::from_complex(&complex).unwrap();
    correspondence.verify_chain_law().unwrap();
}

fn tetrahedron() -> Arc<ComplexCore> {
    oriented_complex(&[[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
}

fn assert_signed_bijections(correspondence: &polygeo_core::SurfaceCorrespondence) {
    for degree in 0..=2 {
        let permutation = correspondence.permutation(degree).unwrap();
        assert_eq!(
            permutation
                .target_of_source()
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len(),
            permutation.len()
        );
        assert!(
            permutation
                .signs()
                .iter()
                .all(|sign| matches!(sign, -1 | 1))
        );
        for source in 0..permutation.len() {
            let (target, sign) = permutation.map_basis(source).unwrap();
            assert_eq!(permutation.inverse_basis(target).unwrap(), (source, sign));
        }
    }
}

#[test]
fn oriented_triangle_complex_constructs_distinct_surface_and_checked_correspondence() {
    let complex = tetrahedron();
    let (surface, correspondence) = HalfedgeSurfaceCore::from_complex(&complex).unwrap();

    assert_eq!(
        correspondence.direction(),
        CorrespondenceDirection::ComplexToSurface
    );
    assert!(Arc::ptr_eq(correspondence.complex_owner(), &complex));
    assert!(Arc::ptr_eq(correspondence.surface_owner(), &surface));
    assert_eq!(
        (
            surface.vertex_count(),
            surface.edge_count(),
            surface.material_face_count()
        ),
        (4, 6, 4)
    );
    assert_eq!(surface.euler_characteristic(), 2);
    assert_eq!(surface.genus(), Some(0));
    assert_signed_bijections(&correspondence);
    correspondence.verify_chain_law().unwrap();

    let relation = correspondence.isomorphism();
    let source_value = relation
        .source()
        .space(2)
        .unwrap()
        .element([(0, BigInt::from(17))])
        .unwrap();
    let target_value = relation.forward(2).unwrap().apply(&source_value).unwrap();
    let recovered = relation.inverse(2).unwrap().apply(&target_value).unwrap();
    assert_eq!(recovered.indices(), source_value.indices());
    assert_eq!(recovered.coefficients(), source_value.coefficients());

    let target_cochain = relation
        .target()
        .dual()
        .space(2)
        .unwrap()
        .element([(0, BigInt::from(3))])
        .unwrap();
    let source_cochain = relation
        .dual()
        .forward(2)
        .unwrap()
        .apply(&target_cochain)
        .unwrap();
    assert_eq!(
        source_cochain.basis_size(),
        relation.source().space(2).unwrap().basis_size()
    );

    let forward = relation.forward(2).unwrap();
    let inverse = relation.inverse(2).unwrap();
    let cancelled = compose(&inverse, &forward).unwrap();
    assert_eq!(cancelled.execution_steps(), 0);
    assert_eq!(
        cancelled.apply(&source_value).unwrap().coefficients(),
        source_value.coefficients()
    );

    let target_boundary = relation.target().boundary(2).unwrap();
    let composite = compose(&target_boundary, &forward).unwrap();
    assert_eq!(composite.execution_steps(), 2);
    assert!(composite.same_identity(&composite.dual().dual()));
    let estimate = CsrRepresentation::estimate(&composite, BigIntEncoding).unwrap();
    let representation = CsrRepresentation::build(
        &composite,
        BigIntEncoding,
        CsrBuildLimit::for_estimate(estimate),
    )
    .unwrap();
    let direct = composite.apply(&source_value).unwrap();
    let represented = representation.apply(&source_value).unwrap();
    assert_eq!(represented.indices(), direct.indices());
    assert_eq!(represented.coefficients(), direct.coefficients());
}

#[test]
fn triangular_surface_constructs_distinct_complex_and_reverse_correspondence() {
    let surface = HalfedgeSurfaceCore::admit(common::polygon_disk(3)).unwrap();
    let (complex, correspondence) = surface.to_complex().unwrap();

    assert_eq!(
        correspondence.direction(),
        CorrespondenceDirection::SurfaceToComplex
    );
    assert!(Arc::ptr_eq(correspondence.complex_owner(), &complex));
    assert!(Arc::ptr_eq(correspondence.surface_owner(), &surface));
    assert_eq!(complex.basis(2).unwrap().row_count(), 1);
    assert!(!Arc::ptr_eq(
        &HalfedgeSurfaceCore::from_complex(&complex).unwrap().0,
        &surface,
    ));
    assert_signed_bijections(&correspondence);
    correspondence.verify_chain_law().unwrap();
}

#[test]
fn quotient_and_polygonal_surfaces_have_no_simplicial_reverse() {
    for input in [
        common::unigon(),
        common::one_vertex_torus(),
        common::annulus(),
    ] {
        let surface = HalfedgeSurfaceCore::admit(input).unwrap();
        let error = surface.to_complex().unwrap_err();
        assert_eq!(error.reason(), "conversion_not_simplicial");
    }
}

#[test]
fn forward_conversion_requires_previously_admitted_capabilities() {
    let candidate = CandidateInput::signed([0_i64, 1, 2], 1, 3, None).unwrap();
    let complex = ComplexCore::admit(candidate).unwrap();
    assert_eq!(
        complex.require_triangle().unwrap_err().reason(),
        "capability_not_admitted"
    );
    assert_eq!(
        complex.require_oriented().unwrap_err().reason(),
        "capability_not_admitted"
    );
}
