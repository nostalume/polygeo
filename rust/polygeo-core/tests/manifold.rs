use std::sync::Arc;

use polygeo_core::{CandidateInput, ComplexCore, TopologyError};

fn admit(rows: &[&[i128]], vertex_count: Option<usize>) -> Arc<ComplexCore> {
    let width = rows[0].len();
    let candidate = CandidateInput::signed(
        rows.iter().flat_map(|row| row.iter().copied()),
        rows.len(),
        width,
        vertex_count,
    )
    .unwrap();
    ComplexCore::admit(candidate).unwrap()
}

#[test]
fn triangle_manifold_refinement_shares_owner_and_exact_boundary_masks() {
    let raw = admit(&[&[0, 1, 2], &[0, 2, 3]], None);
    let manifold = raw.refine_triangle().unwrap();

    assert!(std::ptr::eq(manifold.owner(), raw.as_ref()));
    assert_eq!(manifold.regular().boundary_mask(0).unwrap(), &[true; 4]);
    assert_eq!(
        manifold.regular().boundary_mask(1).unwrap(),
        &[true, false, true, true, true]
    );
    assert_eq!(
        manifold.regular().boundary_mask(2).unwrap(),
        &[false, false]
    );
}

#[test]
fn triangle_manifold_rejects_false_claims_with_stable_reasons() {
    let graph = admit(&[&[0, 1], &[1, 2]], None);
    assert_eq!(
        graph.refine_triangle().unwrap_err(),
        TopologyError::triangle_dimension(1)
    );

    let nonmanifold_edge = admit(&[&[0, 1, 2], &[1, 0, 3], &[0, 1, 4]], None);
    assert_eq!(
        nonmanifold_edge.refine_triangle().unwrap_err(),
        TopologyError::codimension_one_incidence(1, 0, 3)
    );

    let disconnected_fan = admit(&[&[0, 1, 2], &[0, 3, 4]], None);
    assert_eq!(
        disconnected_fan.refine_triangle().unwrap_err(),
        TopologyError::vertex_link(0)
    );

    let isolated_vertex = admit(&[&[0, 1, 2]], Some(4));
    assert_eq!(
        isolated_vertex.refine_triangle().unwrap_err(),
        TopologyError::not_pure(3)
    );
}
