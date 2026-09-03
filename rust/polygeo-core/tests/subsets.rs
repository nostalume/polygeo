use std::collections::BTreeSet;
use std::sync::Arc;

use polygeo_core::{
    topology::CandidateInput, topology::Complex as ComplexCore, topology::TopologyError,
};

fn admit(rows: &[&[i128]]) -> Arc<ComplexCore> {
    let width = rows[0].len();
    ComplexCore::admit(
        CandidateInput::signed(
            rows.iter().flat_map(|row| row.iter().copied()),
            rows.len(),
            width,
            None,
        )
        .unwrap(),
    )
    .unwrap()
}

fn disk() -> Arc<ComplexCore> {
    admit(&[&[0, 1, 2], &[0, 2, 3]])
}

#[test]
fn packed_subset_relations_match_exact_nonclosed_fixture() {
    let owner = disk();
    let subset = owner
        .subset(&[
            vec![true, false, false, false],
            vec![false; 5],
            vec![false; 2],
        ])
        .unwrap();

    let closure = subset.closure().unwrap();
    let star = subset.star().unwrap();
    let link = subset.link().unwrap();
    assert_eq!(closure.mask(0).unwrap(), vec![true, false, false, false]);
    assert_eq!(closure.mask(1).unwrap(), vec![false; 5]);
    assert_eq!(star.mask(0).unwrap(), vec![true, false, false, false]);
    assert_eq!(star.mask(1).unwrap(), vec![true, true, true, false, false]);
    assert_eq!(star.mask(2).unwrap(), vec![true, true]);
    assert_eq!(link.mask(0).unwrap(), vec![false, true, true, true]);
    assert_eq!(link.mask(1).unwrap(), vec![false, false, false, true, true]);
    assert_eq!(link.mask(2).unwrap(), vec![false, false]);
    assert!(closure.same_members(&closure.closure().unwrap()).unwrap());
    assert!(subset.is_pure(0).unwrap());
    assert!(!subset.is_pure(1).unwrap());
}

#[test]
fn subset_builder_packs_single_pass_degree_iterators_directly() {
    let owner = disk();
    let mut builder = owner.subset_builder().unwrap();
    builder.push_degree([true, false, false, false]).unwrap();
    builder.push_degree(std::iter::repeat_n(false, 5)).unwrap();
    builder.push_degree(std::iter::repeat_n(false, 2)).unwrap();

    let subset = builder.finish().unwrap();

    assert_eq!(subset.mask(0).unwrap(), [true, false, false, false]);
}

#[test]
fn regular_boundary_is_a_native_closed_pure_subset() {
    let owner = disk();
    owner.refine_regular().unwrap();
    let boundary = owner.boundary_subset().unwrap();

    assert!(boundary.same_members(&boundary.closure().unwrap()).unwrap());
    assert!(boundary.is_pure(1).unwrap());
    let copied = boundary.to_owned_subset().unwrap();
    assert!(boundary.same_members(&copied).unwrap());
    assert!(copied.same_members(&copied.closure().unwrap()).unwrap());
    assert!(
        boundary
            .closure()
            .unwrap()
            .same_members(&copied.closure().unwrap())
            .unwrap()
    );
    assert!(
        boundary
            .star()
            .unwrap()
            .same_members(&copied.star().unwrap())
            .unwrap()
    );
    assert!(
        boundary
            .link()
            .unwrap()
            .same_members(&copied.link().unwrap())
            .unwrap()
    );
    assert_eq!(boundary.is_pure(1).unwrap(), copied.is_pure(1).unwrap());
    let mut exposed = copied.mask(1).unwrap();
    exposed.fill(false);
    assert_eq!(copied.mask(1).unwrap(), vec![true, false, true, true, true]);
    assert_eq!(boundary.mask(1).unwrap(), copied.mask(1).unwrap());

    let foreign = disk();
    foreign.refine_regular().unwrap();
    let foreign_boundary = foreign.boundary_subset().unwrap();
    assert_eq!(
        boundary.same_members(&foreign_boundary),
        Err(TopologyError::OwnerMismatch)
    );
}

#[test]
fn subsets_reject_shape_degree_and_foreign_owner() {
    let owner = disk();
    assert_eq!(
        owner.subset(&[vec![false; 4]]).unwrap_err(),
        TopologyError::MaskShape
    );
    let first = owner
        .subset(&[vec![false; 4], vec![false; 5], vec![false; 2]])
        .unwrap();
    let foreign_owner = disk();
    let foreign = foreign_owner
        .subset(&[vec![false; 4], vec![false; 5], vec![false; 2]])
        .unwrap();
    assert_eq!(
        first.same_members(&foreign),
        Err(TopologyError::OwnerMismatch)
    );
    assert_eq!(first.mask(3), Err(TopologyError::degree_outside(3)));
    assert_eq!(first.is_pure(3), Err(TopologyError::degree_outside(3)));
}

#[test]
fn canonical_selection_and_complement_are_owner_bound() {
    let owner = disk();
    let selected = owner.selection(1, vec![0, 2, 4]).unwrap();
    let complement = selected.complement().unwrap();

    assert_eq!(selected.indices(), &[0, 2, 4]);
    assert_eq!(complement.indices(), &[1, 3]);
    assert!(Arc::ptr_eq(selected.owner(), &owner));
    assert!(
        selected
            .same_selection(&owner.selection(1, vec![0, 2, 4]).unwrap())
            .unwrap()
    );
    assert!(!selected.same_selection(&complement).unwrap());
    assert_eq!(
        owner.selection(1, vec![0, 0]).unwrap_err(),
        TopologyError::SelectionNotStrict
    );
    assert_eq!(
        owner.selection(1, vec![5]).unwrap_err(),
        TopologyError::SelectionIndexOutside
    );
    assert_eq!(
        selected.same_selection(&disk().selection(1, vec![0, 2, 4]).unwrap()),
        Err(TopologyError::OwnerMismatch)
    );
}

#[test]
fn exhaustive_disk_masks_match_set_truth() {
    let owner = disk();
    let counts = (0..=owner.dimension())
        .map(|degree| owner.basis(degree).unwrap().row_count())
        .collect::<Vec<_>>();
    let total = counts.iter().sum::<usize>();
    for bits in 0_u64..(1_u64 << total) {
        let mut cursor = 0;
        let masks = counts
            .iter()
            .map(|count| {
                let values = (0..*count)
                    .map(|index| bits & (1 << (cursor + index)) != 0)
                    .collect::<Vec<_>>();
                cursor += count;
                values
            })
            .collect::<Vec<_>>();
        let subset = owner.subset(&masks).unwrap();
        let selected = selected_sets(&owner, &masks);
        assert_relation(&owner, &subset.closure().unwrap(), &selected, true, false);
        assert_relation(&owner, &subset.star().unwrap(), &selected, false, true);
        assert_link(&owner, &subset.link().unwrap(), &selected);
        for degree in 0..=owner.dimension() {
            assert_eq!(
                subset.is_pure(degree).unwrap(),
                pure_truth(&selected, degree)
            );
        }
    }
}

#[test]
fn arbitrary_dimensional_link_matches_set_truth_in_deterministic_basis_order() {
    let owner = admit(&[&[0, 1, 2, 3, 4]]);
    let counts = (0..=owner.dimension())
        .map(|degree| owner.basis(degree).unwrap().row_count())
        .collect::<Vec<_>>();
    let mut masks = counts
        .iter()
        .map(|count| vec![false; *count])
        .collect::<Vec<_>>();
    masks[1][0] = true;
    masks[2][5] = true;
    let subset = owner.subset(&masks).unwrap();
    let selected = selected_sets(&owner, &masks);

    assert_link(&owner, &subset.link().unwrap(), &selected);
    assert_link(&owner, &subset.link().unwrap(), &selected);
}

fn selected_sets(owner: &ComplexCore, masks: &[Vec<bool>]) -> Vec<(BTreeSet<usize>, usize)> {
    (0..=owner.dimension())
        .flat_map(|degree| {
            masks[degree]
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, selected)| *selected)
                .map(move |(index, _)| {
                    (
                        owner
                            .basis(degree)
                            .unwrap()
                            .row(index)
                            .unwrap()
                            .iter()
                            .copied()
                            .collect(),
                        degree,
                    )
                })
        })
        .collect()
}

fn assert_relation(
    owner: &ComplexCore,
    observed: &polygeo_core::topology::Subset,
    selected: &[(BTreeSet<usize>, usize)],
    faces: bool,
    cofaces: bool,
) {
    for degree in 0..=owner.dimension() {
        let basis = owner.basis(degree).unwrap();
        let expected = (0..basis.row_count())
            .map(|index| {
                let candidate = basis
                    .row(index)
                    .unwrap()
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>();
                selected.iter().any(|(simplex, _)| {
                    (faces && candidate.is_subset(simplex))
                        || (cofaces && simplex.is_subset(&candidate))
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(observed.mask(degree).unwrap(), expected);
    }
}

fn assert_link(
    owner: &ComplexCore,
    observed: &polygeo_core::topology::Subset,
    selected: &[(BTreeSet<usize>, usize)],
) {
    let all = (0..=owner.dimension())
        .flat_map(|degree| {
            let basis = owner.basis(degree).unwrap();
            (0..basis.row_count()).map(move |index| {
                basis
                    .row(index)
                    .unwrap()
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
            })
        })
        .collect::<Vec<_>>();
    for degree in 0..=owner.dimension() {
        let basis = owner.basis(degree).unwrap();
        let expected = (0..basis.row_count())
            .map(|index| {
                let candidate = basis
                    .row(index)
                    .unwrap()
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>();
                selected.iter().any(|(simplex, _)| {
                    candidate.is_disjoint(simplex)
                        && all.contains(&candidate.union(simplex).copied().collect())
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(observed.mask(degree).unwrap(), expected);
    }
}

fn pure_truth(selected: &[(BTreeSet<usize>, usize)], degree: usize) -> bool {
    !selected.is_empty()
        && selected
            .iter()
            .enumerate()
            .filter(|(index, (simplex, _))| {
                !selected
                    .iter()
                    .enumerate()
                    .any(|(other_index, (other, _))| {
                        *index != other_index
                            && simplex.len() < other.len()
                            && simplex.is_subset(other)
                    })
            })
            .all(|(_, (_, item_degree))| *item_degree == degree)
}
