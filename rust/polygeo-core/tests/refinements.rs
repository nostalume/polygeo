use std::sync::Arc;

use polygeo_core::{CandidateInput, CodimensionOneRegularCapability, ComplexCore, TopologyError};

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
fn regular_boundary_is_packed_closed_and_shared_by_owner_views() {
    let owner = admit(&[&[0, 1, 2], &[0, 2, 3]], None);
    let regular = owner.refine_regular().unwrap();
    let triangle = owner.refine_triangle().unwrap();
    let bounded = triangle.regular().with_boundary().unwrap();

    assert_eq!(regular.boundary_mask(0).unwrap(), vec![true; 4]);
    assert_eq!(
        regular.boundary_mask(1).unwrap(),
        vec![true, false, true, true, true]
    );
    assert_eq!(regular.boundary_mask(2).unwrap(), vec![false, false]);
    assert!(std::ptr::eq(regular.owner(), triangle.owner()));
    assert!(std::ptr::eq(regular.owner(), bounded.regular().owner()));
}

#[test]
fn all_six_refinements_reuse_success_once_and_remain_owner_bound() {
    let owner = admit(&[&[0, 1, 2], &[0, 2, 3]], None);
    let foreign = admit(&[&[0, 1, 2], &[0, 2, 3]], None);

    let regular_a = owner.refine_regular().unwrap();
    let regular_b = owner.refine_regular().unwrap();
    let triangle_a = owner.refine_triangle().unwrap();
    let triangle_b = owner.refine_triangle().unwrap();
    let oriented_a = owner.refine_oriented().unwrap();
    let oriented_b = owner.refine_oriented().unwrap();
    let connected_a = owner.refine_connected().unwrap();
    let connected_b = owner.refine_connected().unwrap();
    regular_a.with_boundary().unwrap();
    triangle_a.regular().with_boundary().unwrap();

    for view_owner in [
        regular_a.owner(),
        regular_b.owner(),
        triangle_b.owner(),
        oriented_a.owner(),
        oriented_b.owner(),
        connected_a.owner(),
        connected_b.owner(),
    ] {
        assert!(std::ptr::eq(view_owner, owner.as_ref()));
    }
    assert!(!std::ptr::eq(
        regular_a.owner(),
        foreign.refine_regular().unwrap().owner()
    ));
}

#[test]
fn concurrent_success_returns_borrowed_views() {
    let owner = admit(&[&[0, 1, 2], &[0, 2, 3]], None);
    let workers = (0..8)
        .map(|_| {
            let shared = Arc::clone(&owner);
            std::thread::spawn(move || {
                shared.refine_regular().unwrap();
                shared.refine_triangle().unwrap();
                shared.refine_oriented().unwrap();
                shared.refine_connected().unwrap();
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap();
    }

    assert!(owner.refine_regular().is_ok());
    assert!(owner.refine_triangle().is_ok());
    assert!(owner.refine_oriented().is_ok());
    assert!(owner.refine_connected().is_ok());
}

#[test]
fn boundary_classification_is_exclusive_and_dimension_zero_is_closed() {
    let disk = admit(&[&[0, 1, 2]], None);
    let disk_regular = disk.refine_regular().unwrap();
    assert!(disk_regular.with_boundary().is_ok());
    assert_eq!(
        disk_regular.without_boundary().unwrap_err(),
        TopologyError::boundary_present(0)
    );

    let point = admit(&[&[0]], None);
    let point_regular = point.refine_regular().unwrap();
    assert!(point_regular.without_boundary().is_ok());
    assert_eq!(
        point_regular.with_boundary().unwrap_err(),
        TopologyError::BoundaryAbsent
    );

    let sphere = admit(&[&[0, 2, 1], &[0, 1, 3], &[0, 3, 2], &[1, 2, 3]], None);
    let closed = sphere.refine_regular().unwrap().without_boundary().unwrap();
    assert_eq!(closed.regular().boundary_mask(1).unwrap(), vec![false; 6]);
}

#[test]
fn borrowed_boundary_views_carry_constructive_endpoint_evidence() {
    fn entailed_regular(value: &impl CodimensionOneRegularCapability) -> usize {
        value.as_regular().owner().vertex_count()
    }

    let disk = admit(&[&[0, 1, 2]], None);
    let bounded = disk.refine_regular().unwrap().with_boundary().unwrap();
    let witness = bounded.codimension_one_simplex();
    assert!(witness < disk.basis(1).unwrap().row_count());
    assert!(
        bounded
            .regular()
            .boundary_mask(1)
            .unwrap()
            .get(witness)
            .copied()
            .unwrap()
    );
    assert_eq!(entailed_regular(&bounded), 3);

    let point = admit(&[&[0]], None);
    let closed = point.refine_regular().unwrap().without_boundary().unwrap();
    assert_eq!(closed.regular().boundary_mask(0).unwrap(), vec![false]);
    assert_eq!(entailed_regular(&closed), 1);
}

#[test]
fn regularity_is_arbitrary_dimensional_and_checks_purity_and_top_incidence() {
    let tetrahedron = admit(&[&[0, 1, 2, 3]], None);
    assert!(tetrahedron.refine_regular().is_ok());

    let impure = admit(&[&[0, 1, 2, 3]], Some(5));
    assert_eq!(
        impure.refine_regular().unwrap_err(),
        TopologyError::not_pure(4)
    );

    let over_incident = admit(&[&[0, 1, 2], &[1, 0, 3], &[0, 1, 4]], None);
    assert_eq!(
        over_incident.refine_regular().unwrap_err(),
        TopologyError::codimension_one_incidence(1, 0, 3)
    );
}

#[test]
fn orientation_and_connectivity_are_independent_predicates() {
    let coherent = admit(&[&[0, 1, 2], &[0, 2, 3]], None);
    assert!(coherent.refine_oriented().is_ok());
    assert!(coherent.refine_connected().is_ok());

    let incoherent = admit(&[&[0, 1, 2], &[0, 3, 2]], None);
    assert_eq!(
        incoherent.refine_oriented().unwrap_err(),
        TopologyError::orientation(1)
    );

    let disconnected = admit(&[&[0, 1], &[2, 3]], None);
    assert_eq!(
        disconnected.refine_connected().unwrap_err(),
        TopologyError::disconnected(2)
    );
}

#[test]
fn require_is_query_only_and_replays_cached_rejection() {
    let owner = admit(&[&[0, 1, 2], &[0, 3, 2]], None);
    assert_eq!(
        owner.require_oriented().unwrap_err(),
        TopologyError::capability_not_admitted("oriented")
    );
    assert_eq!(
        owner.refine_oriented().unwrap_err(),
        TopologyError::orientation(1)
    );
    assert_eq!(
        owner.require_oriented().unwrap_err(),
        TopologyError::orientation(1)
    );
}
