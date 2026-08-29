mod common;

use polygeo_core::{FaceKind, HalfedgeInput, HalfedgeSurfaceCore, TopologyDetails, TopologyError};
use proptest::prelude::*;

use common::{
    annulus, disconnected_triangles, empty_surface, input, one_vertex_torus, polygon_disk, unigon,
};

#[test]
fn owner_issued_halfedge_navigation_preserves_owner_and_category() {
    let surface = HalfedgeSurfaceCore::admit(polygon_disk(3)).unwrap();
    let halfedge = surface.halfedge(0).unwrap();

    assert_eq!(halfedge.index(), 0);
    assert_eq!(halfedge.next().index(), 1);
    assert_eq!(halfedge.twin().index(), 3);
    assert_eq!(halfedge.vertex().index(), 0);
    assert_eq!(halfedge.edge().index(), 0);
    assert_eq!(halfedge.face_orbit().index(), 0);
    assert!(std::ptr::eq(halfedge.owner(), surface.as_ref()));
}

#[test]
fn entity_equality_is_nominal_to_owner_and_index() {
    let first = HalfedgeSurfaceCore::admit(polygon_disk(3)).unwrap();
    let second = HalfedgeSurfaceCore::admit(polygon_disk(3)).unwrap();

    assert_eq!(first.halfedge(0).unwrap(), first.halfedge(0).unwrap());
    assert_ne!(first.halfedge(0).unwrap(), first.halfedge(1).unwrap());
    assert_ne!(first.halfedge(0).unwrap(), second.halfedge(0).unwrap());
}

#[test]
fn disk_exposes_separate_face_domains_and_explicit_owned_materialization() {
    let surface = HalfedgeSurfaceCore::admit(polygon_disk(3)).unwrap();

    assert_eq!(surface.halfedge_count(), 6);
    assert_eq!(surface.vertex_count(), 3);
    assert_eq!(surface.edge_count(), 3);
    assert_eq!(surface.face_orbit_count(), 2);
    assert_eq!(surface.material_face_count(), 1);
    assert_eq!(surface.exterior_face_count(), 1);
    assert_eq!(
        surface
            .face_orbits()
            .map(polygeo_core::FaceOrbit::kind)
            .collect::<Vec<_>>(),
        [FaceKind::Material, FaceKind::Exterior]
    );

    let material = surface.material_faces().next().unwrap();
    assert_eq!(material.index(), 0);
    assert_eq!(material.face_orbit().index(), 0);
    assert_eq!(
        material
            .halfedges()
            .map(polygeo_core::Halfedge::index)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );

    let packed = surface.materialize_face_orbits().unwrap();
    assert_eq!(packed.representatives(), [0, 3]);
    assert_eq!(packed.offsets(), [0, 3, 6]);
    assert_eq!(packed.halfedges(), [0, 1, 2, 3, 5, 4]);

    let (mut representatives, _, _) = packed.into_parts();
    representatives[0] = usize::MAX;
    assert_eq!(
        surface.materialize_face_orbits().unwrap().representatives(),
        [0, 3]
    );
}

#[test]
fn admits_annulus_unigon_torus_and_disconnected_presentations() {
    let annulus = HalfedgeSurfaceCore::admit(annulus()).unwrap();
    assert_eq!(
        (
            annulus.halfedge_count(),
            annulus.vertex_count(),
            annulus.edge_count(),
            annulus.material_face_count(),
            annulus.exterior_face_count(),
        ),
        (24, 8, 12, 4, 2)
    );

    let unigon = HalfedgeSurfaceCore::admit(unigon()).unwrap();
    assert_eq!(
        (
            unigon.halfedge_count(),
            unigon.vertex_count(),
            unigon.edge_count(),
            unigon.material_face_count(),
            unigon.exterior_face_count(),
        ),
        (2, 1, 1, 1, 1)
    );
    assert_eq!(
        unigon
            .halfedge(1)
            .unwrap()
            .as_exterior()
            .unwrap()
            .next()
            .index(),
        1
    );

    let torus = HalfedgeSurfaceCore::admit(one_vertex_torus()).unwrap();
    assert_eq!(
        (
            torus.vertex_count(),
            torus.edge_count(),
            torus.material_face_count(),
            torus.exterior_face_count(),
        ),
        (1, 3, 2, 0)
    );
    assert_eq!(
        torus
            .vertices()
            .next()
            .unwrap()
            .halfedges()
            .map(polygeo_core::Halfedge::index)
            .collect::<Vec<_>>(),
        [0, 5, 2, 4, 1, 3]
    );
    assert_eq!(
        torus.materialize_edge_orbits().unwrap().halfedges(),
        [0, 4, 1, 5, 2, 3]
    );
    assert_eq!(
        torus.materialize_face_orbits().unwrap().halfedges(),
        [0, 1, 2, 3, 4, 5]
    );
    let digest = torus
        .halfedges()
        .fold(14_695_981_039_346_656_037_u64, |digest, halfedge| {
            [
                halfedge.index(),
                halfedge.next().index(),
                halfedge.twin().index(),
                halfedge.vertex().index(),
                halfedge.edge().index(),
                halfedge.face_orbit().index(),
            ]
            .into_iter()
            .fold(digest, |digest, value| {
                (digest ^ u64::try_from(value).unwrap()).wrapping_mul(1_099_511_628_211)
            })
        });
    assert_eq!(digest, 12_237_241_496_720_959_109);

    let disconnected = HalfedgeSurfaceCore::admit(disconnected_triangles()).unwrap();
    assert_eq!(
        (
            disconnected.vertex_count(),
            disconnected.edge_count(),
            disconnected.material_face_count(),
            disconnected.exterior_face_count(),
        ),
        (6, 6, 2, 2)
    );
}

#[test]
fn compact_material_face_indices_map_bijectively_to_material_orbits() {
    let surface = HalfedgeSurfaceCore::admit(input(vec![0, 1], vec![1, 0], vec![0])).unwrap();
    let orbits = surface.face_orbits().collect::<Vec<_>>();

    assert_eq!(orbits[0].kind(), FaceKind::Exterior);
    assert_eq!(orbits[1].kind(), FaceKind::Material);
    assert!(orbits[0].as_material().is_none());
    assert_eq!(orbits[1].as_material().unwrap().index(), 0);
    assert_eq!(
        surface.material_faces().next().unwrap().face_orbit(),
        orbits[1]
    );
}

#[test]
fn refinements_expose_the_two_boundary_navigation_laws() {
    for input in [polygon_disk(3), annulus(), unigon()] {
        let surface = HalfedgeSurfaceCore::admit(input).unwrap();
        for halfedge in surface.halfedges() {
            if let Some(exterior) = halfedge.as_exterior() {
                assert_eq!(exterior.next().halfedge(), halfedge.next());
            }
            if let Some(boundary) = halfedge.as_material_boundary() {
                assert_eq!(boundary.next().halfedge(), halfedge.twin().next().twin());
            }
        }
    }

    let disk = HalfedgeSurfaceCore::admit(polygon_disk(3)).unwrap();
    let mut material = disk.halfedge(0).unwrap().as_material_boundary().unwrap();
    let mut material_cycle = Vec::new();
    for _ in 0..3 {
        material_cycle.push(material.index());
        material = material.next();
    }
    assert_eq!(material_cycle, [0, 2, 1]);

    let mut exterior = disk.halfedge(3).unwrap().as_exterior().unwrap();
    let mut exterior_cycle = Vec::new();
    for _ in 0..3 {
        exterior_cycle.push(exterior.index());
        exterior = exterior.next();
    }
    assert_eq!(exterior_cycle, [3, 5, 4]);
}

#[test]
fn an_exterior_seed_classifies_its_complete_face_orbit() {
    let input =
        HalfedgeInput::unsigned([1_u8, 2, 0, 5, 3, 4], [3_u8, 4, 5, 0, 1, 2], [4_u8], 6).unwrap();
    let surface = HalfedgeSurfaceCore::admit(input).unwrap();
    let exterior = surface.halfedge(4).unwrap().face_orbit();

    assert_eq!(exterior.kind(), FaceKind::Exterior);
    assert_eq!(
        exterior
            .halfedges()
            .map(|halfedge| halfedge.as_exterior().unwrap().next().index())
            .collect::<Vec<_>>(),
        [5, 4, 3]
    );
}

#[test]
fn empty_surface_has_empty_domains_and_valid_empty_materializations() {
    let surface = HalfedgeSurfaceCore::admit(empty_surface()).unwrap();

    assert_eq!(surface.halfedge_count(), 0);
    assert_eq!(surface.vertex_count(), 0);
    assert_eq!(surface.edge_count(), 0);
    assert_eq!(surface.face_orbit_count(), 0);
    assert_eq!(surface.material_face_count(), 0);
    assert_eq!(surface.exterior_face_count(), 0);
    assert_eq!(surface.halfedges().len(), 0);
    assert_eq!(surface.vertices().len(), 0);
    assert_eq!(surface.edges().len(), 0);
    assert_eq!(surface.face_orbits().len(), 0);
    assert_eq!(surface.material_faces().len(), 0);
    assert_eq!(surface.materialize_vertex_orbits().unwrap().offsets(), [0]);
    assert_eq!(surface.materialize_edge_orbits().unwrap().offsets(), [0]);
    assert_eq!(surface.materialize_face_orbits().unwrap().offsets(), [0]);
    assert_eq!(
        surface.halfedge(0).unwrap_err(),
        TopologyError::halfedge_range("halfedge", 0, 0, 0)
    );
}

#[test]
fn exterior_seed_ingestion_ignores_hostile_size_hints_and_caps_count() {
    struct HostileHint {
        yielded: bool,
    }

    impl Iterator for HostileHint {
        type Item = u8;

        fn next(&mut self) -> Option<Self::Item> {
            if self.yielded {
                None
            } else {
                self.yielded = true;
                Some(3)
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (usize::MAX, Some(usize::MAX))
        }
    }

    let input = HalfedgeInput::unsigned(
        [1_u8, 2, 0, 5, 3, 4],
        [3_u8, 4, 5, 0, 1, 2],
        HostileHint { yielded: false },
        6,
    )
    .unwrap();
    assert_eq!(
        HalfedgeSurfaceCore::admit(input)
            .unwrap()
            .exterior_face_count(),
        1
    );

    assert_eq!(
        HalfedgeInput::unsigned([1_u8, 0], [1_u8, 0], [0_u8, 1, 0], 2).unwrap_err(),
        TopologyError::HalfedgeShape
    );
}

#[test]
fn rejects_shape_range_permutation_and_twin_law_failures_with_details() {
    assert_eq!(
        HalfedgeInput::unsigned([0_u8], [1_u8], [], 2).unwrap_err(),
        TopologyError::HalfedgeShape
    );

    let range = HalfedgeSurfaceCore::admit(input(
        vec![1, 2, 6, 5, 3, 4],
        vec![3, 4, 5, 0, 1, 2],
        vec![3],
    ))
    .unwrap_err();
    assert_eq!(range.reason(), "halfedge_range");
    assert!(matches!(
        range.details(),
        TopologyDetails::HalfedgeEntry {
            relation: "next",
            halfedge: 2,
            value: 6,
            bound: 6,
        }
    ));

    let exterior_range =
        HalfedgeInput::unsigned([1_u8, 2, 0, 5, 3, 4], [3_u8, 4, 5, 0, 1, 2], [6_u8], 6).unwrap();
    assert_eq!(
        HalfedgeSurfaceCore::admit(exterior_range).unwrap_err(),
        TopologyError::halfedge_range("exterior_seed", 6, 6, 6)
    );

    let duplicate = HalfedgeSurfaceCore::admit(input(
        vec![1, 2, 0, 5, 3, 4],
        vec![3, 3, 5, 0, 1, 2],
        vec![3],
    ))
    .unwrap_err();
    assert_eq!(duplicate.reason(), "halfedge_permutation");
    assert!(matches!(
        duplicate.details(),
        TopologyDetails::HalfedgeEntry {
            relation: "twin",
            halfedge: 1,
            value: 3,
            bound: 6,
        }
    ));

    let fixed = HalfedgeSurfaceCore::admit(input(vec![0, 1], vec![0, 1], vec![])).unwrap_err();
    assert_eq!(fixed, TopologyError::twin_law(0, 0, 0));

    let noninvolutive =
        HalfedgeSurfaceCore::admit(input(vec![0, 1, 2, 3], vec![1, 2, 3, 0], vec![])).unwrap_err();
    assert_eq!(noninvolutive, TopologyError::twin_law(0, 1, 2));
}

#[test]
fn rejects_inconsistent_exterior_classification_and_boundary_cycles() {
    let duplicate_seed =
        HalfedgeInput::unsigned([1_u8, 2, 0, 5, 3, 4], [3_u8, 4, 5, 0, 1, 2], [3_u8, 4], 6)
            .unwrap();
    let error = HalfedgeSurfaceCore::admit(duplicate_seed).unwrap_err();
    assert_eq!(error, TopologyError::exterior_inconsistency(4, 1));

    let both_exterior =
        HalfedgeInput::unsigned([1_u8, 2, 0, 5, 3, 4], [3_u8, 4, 5, 0, 1, 2], [0_u8, 3], 6)
            .unwrap();
    let error = HalfedgeSurfaceCore::admit(both_exterior).unwrap_err();
    assert_eq!(error, TopologyError::exterior_inconsistency(0, 3));

    let pinched_boundary = HalfedgeInput::unsigned(
        [1_u8, 2, 0, 3, 4, 5],
        [3_u8, 4, 5, 0, 1, 2],
        [3_u8, 4, 5],
        6,
    )
    .unwrap();
    let error = HalfedgeSurfaceCore::admit(pinched_boundary).unwrap_err();
    assert_eq!(error.reason(), "boundary_cycle");
    assert!(matches!(error.details(), TopologyDetails::Exterior { .. }));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn rejects_halfedge_counts_beyond_the_proved_i64_domain_before_allocation() {
    let unsupported = usize::try_from(i64::MAX).unwrap() + 1;
    let error = HalfedgeInput::unsigned(
        Vec::<u8>::new(),
        Vec::<u8>::new(),
        Vec::<u8>::new(),
        unsupported,
    )
    .unwrap_err();

    assert_eq!(error, TopologyError::CountOverflow);
}

proptest! {
    #[test]
    fn polygon_admission_retains_permutation_and_orbit_laws(vertex_count in 3_usize..64) {
        let surface = HalfedgeSurfaceCore::admit(polygon_disk(vertex_count)).unwrap();

        prop_assert_eq!(surface.vertex_count(), vertex_count);
        prop_assert_eq!(surface.edge_count(), vertex_count);
        prop_assert_eq!(surface.material_face_count(), 1);
        prop_assert_eq!(surface.exterior_face_count(), 1);
        for halfedge in surface.halfedges() {
            prop_assert_ne!(halfedge.twin(), halfedge);
            prop_assert_eq!(halfedge.twin().twin(), halfedge);
        }
        for face in surface.face_orbits().filter(|face| face.kind() == FaceKind::Exterior) {
            for halfedge in face.halfedges() {
                prop_assert_eq!(halfedge.as_exterior().unwrap().next().halfedge(), halfedge.next());
            }
        }
    }
}

#[test]
fn admitted_surfaces_publish_exact_topology_facts() {
    let disk = HalfedgeSurfaceCore::admit(polygon_disk(3)).unwrap();
    assert_eq!(disk.boundary_component_count(), 1);
    assert_eq!(disk.connected_component_count(), 1);
    assert_eq!(disk.euler_characteristic(), 1);
    assert_eq!(disk.genus(), Some(0));

    let ring = HalfedgeSurfaceCore::admit(annulus()).unwrap();
    assert_eq!(ring.boundary_component_count(), 2);
    assert_eq!(ring.euler_characteristic(), 0);
    assert_eq!(ring.genus(), Some(0));

    let torus = HalfedgeSurfaceCore::admit(one_vertex_torus()).unwrap();
    assert_eq!(torus.boundary_component_count(), 0);
    assert_eq!(torus.euler_characteristic(), 0);
    assert_eq!(torus.genus(), Some(1));

    let disconnected = HalfedgeSurfaceCore::admit(disconnected_triangles()).unwrap();
    assert_eq!(disconnected.connected_component_count(), 2);
    assert_eq!(disconnected.euler_characteristic(), 2);
    assert_eq!(disconnected.genus(), None);
}
