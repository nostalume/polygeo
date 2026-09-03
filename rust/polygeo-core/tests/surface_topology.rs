use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};
use polygeo_core::{
    chain::IntegralCochain, topology::CandidateInput, topology::Complex as ComplexCore,
};
use std::sync::Arc;

fn admit(faces: &[[usize; 3]], vertex_count: usize) -> Arc<ComplexCore> {
    ComplexCore::admit(
        CandidateInput::unsigned(
            faces.iter().flatten().map(|value| *value as u64),
            faces.len(),
            3,
            Some(vertex_count),
        )
        .unwrap(),
    )
    .unwrap()
}

fn torus(major_sections: usize, minor_sections: usize, remove_first: bool) -> Arc<ComplexCore> {
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
    if remove_first {
        faces.remove(0);
    }
    admit(&faces, major_sections * minor_sections)
}

fn assert_exactly_closed(owner: &Arc<ComplexCore>, cycle: &IntegralCochain) {
    let chain = owner.chain_complex();
    let edge_cochains = chain.dual().space(1).unwrap();
    assert!(cycle.space().same_based_space(&edge_cochains));
    let residual = chain.dual().coboundary(1).unwrap().apply(cycle).unwrap();
    assert!(residual.indices().is_empty());
}

fn python_dual_presentation(
    owner: &Arc<ComplexCore>,
    cycle: &IntegralCochain,
) -> Vec<(usize, i64)> {
    let mut source_faces = vec![(usize::MAX, 0_i64); owner.basis(1).unwrap().row_count()];
    for (edge, face, sign) in owner.chain_view().boundary(2).unwrap().exact_entries() {
        if face < source_faces[edge].0 {
            source_faces[edge] = (face, sign);
        }
    }
    cycle
        .indices()
        .iter()
        .copied()
        .zip(cycle.coefficients())
        .map(|(edge, coefficient)| (edge, -source_faces[edge].1 * coefficient.to_i64().unwrap()))
        .collect()
}

#[test]
fn disk_fact_is_cached_on_the_topology_owner_and_preserves_identity() {
    let owner = admit(&[[0, 1, 2], [0, 2, 3]], 4);

    let first = owner.refine_disk().unwrap();
    let second = owner.require_disk().unwrap();

    assert!(std::ptr::eq(first.owner(), owner.as_ref()));
    assert!(std::ptr::eq(first.owner(), second.owner()));
    assert!(std::ptr::eq(first.triangle().owner(), owner.as_ref()));
    assert_eq!(first.boundary_vertices().unwrap().as_ref(), [0, 1, 2, 3]);

    let reversed =
        ComplexCore::admit(CandidateInput::signed([0, 2, 1, 0, 3, 2], 2, 3, Some(4)).unwrap())
            .unwrap();
    assert_eq!(
        reversed
            .refine_disk()
            .unwrap()
            .boundary_vertices()
            .unwrap()
            .as_ref(),
        [0, 3, 2, 1]
    );
}

#[test]
fn disk_fact_rejects_multiple_boundary_components_and_wrong_euler_characteristic() {
    let annulus = admit(
        &[
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [2, 0, 3],
            [2, 3, 5],
        ],
        6,
    );
    assert_eq!(
        annulus.refine_disk().unwrap_err().reason(),
        "disk_boundary_components"
    );

    let punctured_torus = torus(4, 5, true);
    assert_eq!(
        punctured_torus.refine_disk().unwrap_err().reason(),
        "disk_euler_characteristic"
    );
}

#[test]
fn exact_dual_cycles_have_one_topology_owner_and_exact_closure() {
    let owner = torus(4, 5, false);
    let first = owner.integral_dual_cycle_basis().unwrap();
    let second = owner.integral_dual_cycle_basis().unwrap();

    assert!(first.chain_complex().same_owner(&owner.chain_complex()));
    assert_eq!(first.rank(), 2);
    assert_eq!(
        first.generator_edge_indices(),
        second.generator_edge_indices()
    );
    assert_eq!(first.generator_edge_indices(), [54, 59]);
    assert_eq!(first.generator_edge_indices().len(), first.rank());
    for index in 0..first.rank() {
        let cycle = first.cocycle(index).unwrap();
        assert_exactly_closed(&owner, cycle);
        assert!(cycle.coefficients().iter().any(|value| !value.is_zero()));
        assert_eq!(
            cycle.indices(),
            second.cocycle(index).unwrap().indices(),
            "the tree-cotree presentation must be deterministic"
        );
        assert_eq!(
            cycle.coefficients(),
            second.cocycle(index).unwrap().coefficients()
        );
        let generator = first.generator_edge_indices()[index];
        let position = cycle
            .indices()
            .iter()
            .position(|edge| *edge == generator)
            .unwrap();
        assert_eq!(cycle.coefficients()[position].abs(), BigInt::from(1_u8));
    }
    assert_eq!(
        python_dual_presentation(&owner, first.cocycle(0).unwrap()),
        [
            (42, 1),
            (43, -1),
            (45, 1),
            (46, -1),
            (48, 1),
            (49, -1),
            (51, 1),
            (52, -1),
            (53, -1),
            (54, 1),
        ]
    );
    assert_eq!(
        python_dual_presentation(&owner, first.cocycle(1).unwrap()),
        [
            (16, 1),
            (22, -1),
            (23, 1),
            (26, -1),
            (38, -1),
            (41, -1),
            (42, -1),
            (43, 1),
            (45, -1),
            (46, 1),
            (48, -1),
            (49, 1),
            (51, -1),
            (59, 1),
        ]
    );
}

#[test]
fn sphere_has_no_integral_dual_generators_but_keeps_exact_owner_identity() {
    let owner = admit(&[[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], 4);
    let cycles = owner.integral_dual_cycle_basis().unwrap();

    assert_eq!(cycles.rank(), 0);
    assert!(cycles.generator_edge_indices().is_empty());
    assert!(cycles.chain_complex().same_owner(&owner.chain_complex()));
}

#[test]
fn dual_cycles_reject_boundary_and_orientation_before_publishing_data() {
    let disk = admit(&[[0, 1, 2]], 3);
    assert_eq!(
        disk.integral_dual_cycle_basis().unwrap_err().reason(),
        "boundary_present"
    );

    let incoherent = admit(&[[0, 1, 2], [0, 1, 3]], 4);
    assert_eq!(
        incoherent.integral_dual_cycle_basis().unwrap_err().reason(),
        "orientation"
    );
}
