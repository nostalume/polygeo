use std::collections::BTreeMap;
use std::sync::Arc;

use polygeo_core::{
    topology::CandidateInput, topology::Complex as ComplexCore, topology::TopologyError,
};
use proptest::prelude::*;

fn candidate(rows: &[&[i128]], vertex_count: Option<usize>) -> CandidateInput {
    let row_count = rows.len();
    let row_width = rows.first().map_or(0, |row| row.len());
    assert!(rows.iter().all(|row| row.len() == row_width));
    CandidateInput::signed(
        rows.iter().flat_map(|row| row.iter().copied()),
        row_count,
        row_width,
        vertex_count,
    )
    .unwrap()
}

fn assert_boundary_squared_zero(complex: &ComplexCore) {
    for degree in 2..=complex.dimension() {
        let lower = complex.boundary(degree - 1).unwrap();
        let upper = complex.boundary(degree).unwrap();
        let lower_indptr = lower.indptr();
        let upper_indptr = upper.indptr();
        for row in 0..lower.shape().0 {
            let mut product = BTreeMap::<usize, i16>::new();
            for lower_position in lower_indptr[row]..lower_indptr[row + 1] {
                let inner = lower.indices()[lower_position];
                let lower_value = i16::from(lower.data()[lower_position]);
                for upper_position in upper_indptr[inner]..upper_indptr[inner + 1] {
                    let column = upper.indices()[upper_position];
                    *product.entry(column).or_default() +=
                        lower_value * i16::from(upper.data()[upper_position]);
                }
            }
            assert!(product.values().all(|value| *value == 0));
        }
    }
}

#[test]
fn admits_canonical_simplex_and_retains_each_boundary() {
    let complex = ComplexCore::admit(candidate(&[&[4, 2, 0, 3, 1]], None)).unwrap();

    assert_eq!(
        (0..=4)
            .map(|degree| complex.basis(degree).unwrap().row_count())
            .collect::<Vec<_>>(),
        [5, 10, 10, 5, 1]
    );
    assert_eq!(complex.orientation(4).unwrap(), &[-1]);
    assert!(std::ptr::eq(
        complex.boundary(2).unwrap(),
        complex.boundary(2).unwrap()
    ));
    assert_eq!(complex.immediate_face_rows(2).unwrap().len(), 30);
}

#[test]
fn exact_boundaries_square_to_zero_in_dimensions_zero_through_five() {
    for dimension in 0..=5 {
        let row = (0..=dimension).map(i128::from).collect::<Vec<_>>();
        let complex = ComplexCore::admit(candidate(&[&row], None)).unwrap();

        assert_boundary_squared_zero(&complex);
    }
}

#[test]
fn endpoint_boundary_has_zero_rows_and_one_column_per_vertex() {
    let complex = ComplexCore::admit(candidate(&[&[0], &[1]], None)).unwrap();
    let boundary = complex.boundary(0).unwrap();

    assert_eq!(boundary.shape(), (0, 2));
    assert!(boundary.data().is_empty());
}

#[test]
fn rejects_false_domain_claims_with_stable_reasons() {
    assert_eq!(
        CandidateInput::signed([0_i128, -1, 2], 1, 3, None).unwrap_err(),
        TopologyError::negative_index(-1)
    );
    assert_eq!(
        CandidateInput::signed([0_i128, 1, 1], 1, 3, None).unwrap_err(),
        TopologyError::repeated_vertex(1)
    );
    let cases = [
        (
            candidate(&[&[0, 1, 2], &[2, 1, 0]], None),
            TopologyError::DuplicateMaximalSimplex,
        ),
        (
            candidate(&[&[0, 1, 2]], Some(2)),
            TopologyError::vertex_extent(2, 3),
        ),
    ];

    for (candidate, expected) in cases {
        assert_eq!(ComplexCore::admit(candidate).unwrap_err(), expected);
    }
}

#[test]
fn separately_admitted_equal_inputs_have_distinct_owners() {
    let first = ComplexCore::admit(candidate(&[&[0, 1, 2]], None)).unwrap();
    let second = ComplexCore::admit(candidate(&[&[0, 1, 2]], None)).unwrap();

    assert!(!Arc::ptr_eq(&first, &second));
}

#[test]
fn rejects_empty_shape_mismatch_and_count_overflow_before_admission() {
    let empty = CandidateInput::signed(Vec::<i64>::new(), 0, 0, None).unwrap();
    assert_eq!(
        ComplexCore::admit(empty).unwrap_err(),
        TopologyError::EmptyMaximalSimplices
    );
    assert_eq!(
        CandidateInput::signed([0_i64], 1, 2, None).unwrap_err(),
        TopologyError::CandidateShape
    );
    assert_eq!(
        CandidateInput::signed(Vec::<i64>::new(), usize::MAX, 2, None).unwrap_err(),
        TopologyError::CountOverflow
    );
    assert_eq!(
        CandidateInput::unsigned([u128::MAX], 1, 1, None).unwrap_err(),
        TopologyError::index_overflow(u128::MAX)
    );
}

#[test]
fn direct_boundary_assembly_has_exact_canonical_payload() {
    let complex = ComplexCore::admit(candidate(&[&[0, 1, 2]], None)).unwrap();
    let boundary = complex.boundary(2).unwrap();

    assert_eq!(boundary.indptr(), vec![0, 1, 2, 3]);
    assert_eq!(boundary.indices(), &[0, 0, 0]);
    assert_eq!(boundary.data(), &[1, -1, 1]);
}

#[test]
fn canonical_boundary_reuses_every_borrowed_storage_slice() {
    let complex = ComplexCore::admit(candidate(&[&[0, 1, 2]], None)).unwrap();
    let first = complex.boundary(2).unwrap();
    let repeated = complex.boundary(2).unwrap();

    assert_eq!(first.indptr().as_ptr(), repeated.indptr().as_ptr());
    assert_eq!(first.indices().as_ptr(), repeated.indices().as_ptr());
    assert_eq!(first.data().as_ptr(), repeated.data().as_ptr());
}

proptest! {
    #[test]
    fn simplex_chain_law_survives_arbitrary_single_swap(
        dimension in 0_usize..7,
        left_seed in any::<usize>(),
        right_seed in any::<usize>(),
    ) {
        let mut row = (0..=dimension)
            .map(|value| i128::try_from(value).unwrap())
            .collect::<Vec<_>>();
        if row.len() > 1 {
            let left = left_seed % row.len();
            let right = right_seed % row.len();
            row.swap(left, right);
        }
        let complex = ComplexCore::admit(candidate(&[&row], None)).unwrap();
        assert_boundary_squared_zero(&complex);
    }
}
