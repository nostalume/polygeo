#[allow(dead_code)]
mod common;

use polygeo_core::{
    topology::CandidateInput, topology::CoefficientSlice, topology::Complex as ComplexCore,
    topology::HalfedgeSurface as HalfedgeSurfaceCore, topology::TopologyError,
};
use proptest::prelude::*;

use common::{empty_surface, input, one_vertex_torus, polygon_disk, unigon};

fn compose_exact(
    lower_shape: (usize, usize),
    lower: &[(usize, usize, i64)],
    upper_shape: (usize, usize),
    upper: &[(usize, usize, i64)],
) -> Vec<i64> {
    assert_eq!(lower_shape.1, upper_shape.0);
    let mut product = vec![0_i64; lower_shape.0 * upper_shape.1];
    for &(row, middle, left) in lower {
        for &(upper_middle, column, right) in upper {
            if middle == upper_middle {
                product[row * upper_shape.1 + column] += left * right;
            }
        }
    }
    product
}

#[test]
fn halfedge_chain_view_retains_exact_native_boundaries_once() {
    let surface = HalfedgeSurfaceCore::admit(polygon_disk(3)).unwrap();
    let chain = surface.chain_view();

    assert_eq!(chain.dimension(), 2);
    assert_eq!(chain.basis_size(0).unwrap(), 3);
    assert_eq!(chain.basis_size(1).unwrap(), 3);
    assert_eq!(chain.basis_size(2).unwrap(), 1);
    assert_eq!(
        chain.basis_size(3).unwrap_err(),
        TopologyError::degree_outside(3)
    );

    let boundary_zero = chain.boundary(0).unwrap();
    assert_eq!(boundary_zero.shape(), (0, 3));
    assert!(boundary_zero.exact_entries().next().is_none());

    let boundary_one = chain.boundary(1).unwrap();
    assert_eq!(boundary_one.shape(), (3, 3));
    assert_eq!(
        boundary_one.indptr().as_ptr(),
        surface.chain_view().boundary(1).unwrap().indptr().as_ptr()
    );
    assert!(matches!(
        boundary_one.coefficients(),
        CoefficientSlice::I64(_)
    ));
    assert_eq!(
        boundary_one.exact_entries().collect::<Vec<_>>(),
        [
            (0, 0, -1),
            (0, 2, 1),
            (1, 0, 1),
            (1, 1, -1),
            (2, 1, 1),
            (2, 2, -1),
        ]
    );

    let boundary_two = chain.boundary(2).unwrap();
    assert_eq!(boundary_two.shape(), (3, 1));
    assert!(matches!(
        boundary_two.coefficients(),
        CoefficientSlice::I64(_)
    ));
    assert_eq!(
        boundary_two.exact_entries().collect::<Vec<_>>(),
        [(0, 0, 1), (1, 0, 1), (2, 0, 1)]
    );
    let data_pointer = match boundary_two.coefficients() {
        CoefficientSlice::I64(values) => values.as_ptr(),
        CoefficientSlice::I8(_) => unreachable!(),
    };
    let repeated_pointer = match surface.chain_view().boundary(2).unwrap().coefficients() {
        CoefficientSlice::I64(values) => values.as_ptr(),
        CoefficientSlice::I8(_) => unreachable!(),
    };
    assert_eq!(data_pointer, repeated_pointer);

    assert_eq!(
        compose_exact(
            boundary_one.shape(),
            &boundary_one.exact_entries().collect::<Vec<_>>(),
            boundary_two.shape(),
            &boundary_two.exact_entries().collect::<Vec<_>>(),
        ),
        [0, 0, 0]
    );
}

#[test]
fn surface_projection_aggregates_cancellation_loops_and_orientation() {
    let cancellation = HalfedgeSurfaceCore::admit(input(vec![1, 0], vec![1, 0], vec![])).unwrap();
    assert_eq!(
        cancellation
            .chain_view()
            .boundary(2)
            .unwrap()
            .exact_entries()
            .collect::<Vec<_>>(),
        []
    );

    let loops = HalfedgeSurfaceCore::admit(one_vertex_torus()).unwrap();
    assert_eq!(loops.chain_view().boundary(1).unwrap().shape(), (1, 3));
    assert_eq!(
        loops
            .chain_view()
            .boundary(1)
            .unwrap()
            .exact_entries()
            .collect::<Vec<_>>(),
        []
    );

    let positive = HalfedgeSurfaceCore::admit(unigon()).unwrap();
    assert_eq!(
        positive
            .chain_view()
            .boundary(2)
            .unwrap()
            .exact_entries()
            .collect::<Vec<_>>(),
        [(0, 0, 1)]
    );
    let negative = HalfedgeSurfaceCore::admit(input(vec![0, 1], vec![1, 0], vec![0])).unwrap();
    assert_eq!(
        negative
            .chain_view()
            .boundary(2)
            .unwrap()
            .exact_entries()
            .collect::<Vec<_>>(),
        [(0, 0, -1)]
    );
}

#[test]
fn empty_surface_has_three_empty_based_degrees() {
    let surface = HalfedgeSurfaceCore::admit(empty_surface()).unwrap();
    let chain = surface.chain_view();

    assert_eq!(
        (0..=2)
            .map(|degree| chain.basis_size(degree).unwrap())
            .collect::<Vec<_>>(),
        [0, 0, 0]
    );
    assert_eq!(chain.boundary(0).unwrap().shape(), (0, 0));
    assert_eq!(chain.boundary(1).unwrap().shape(), (0, 0));
    assert_eq!(chain.boundary(2).unwrap().shape(), (0, 0));
}

#[test]
fn simplicial_chain_view_preserves_compact_i8_incidence() {
    let candidate = CandidateInput::signed([0_i64, 1, 2], 1, 3, Some(3)).unwrap();
    let complex = ComplexCore::admit(candidate).unwrap();
    let chain = complex.chain_view();

    assert_eq!(chain.dimension(), 2);
    assert_eq!(chain.basis_size(2).unwrap(), 1);
    assert!(matches!(
        chain.boundary(2).unwrap().coefficients(),
        CoefficientSlice::I8(_)
    ));
    assert_eq!(
        chain.boundary(2).unwrap().indptr().as_ptr(),
        chain.boundary(2).unwrap().indptr().as_ptr()
    );
    let boundary_one = chain.boundary(1).unwrap();
    let boundary_two = chain.boundary(2).unwrap();
    assert_eq!(
        compose_exact(
            boundary_one.shape(),
            &boundary_one.exact_entries().collect::<Vec<_>>(),
            boundary_two.shape(),
            &boundary_two.exact_entries().collect::<Vec<_>>(),
        ),
        [0, 0, 0]
    );
}

proptest! {
    #[test]
    fn polygon_chain_projection_is_bounded_and_squares_to_zero(vertex_count in 3_usize..64) {
        let surface = HalfedgeSurfaceCore::admit(polygon_disk(vertex_count)).unwrap();
        let chain = surface.chain_view();
        let boundary_one = chain.boundary(1).unwrap();
        let boundary_two = chain.boundary(2).unwrap();
        let lower = boundary_one.exact_entries().collect::<Vec<_>>();
        let upper = boundary_two.exact_entries().collect::<Vec<_>>();

        let coefficient_bound = u64::try_from(surface.halfedge_count()).unwrap();
        let coefficients_are_bounded = lower
            .iter()
            .chain(&upper)
            .all(|entry| entry.2.unsigned_abs() <= coefficient_bound);
        prop_assert!(coefficients_are_bounded);
        prop_assert!(compose_exact(boundary_one.shape(), &lower, boundary_two.shape(), &upper)
            .into_iter()
            .all(|coefficient| coefficient == 0));
    }
}
