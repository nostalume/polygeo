use faer::{
    Side,
    linalg::solvers::{Solve, SolveLstsq},
    mat,
    sparse::{
        SparseColMat, Triplet,
        linalg::solvers::{Llt, Lu, Qr, SymbolicLlt, SymbolicLu, SymbolicQr},
    },
};

fn matrix(values: [f64; 5]) -> SparseColMat<usize, f64> {
    SparseColMat::try_new_from_triplets(
        3,
        3,
        &[
            Triplet::new(0, 0, values[0]),
            Triplet::new(1, 0, values[1]),
            Triplet::new(0, 1, values[2]),
            Triplet::new(1, 1, values[3]),
            Triplet::new(2, 2, values[4]),
        ],
    )
    .unwrap()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-11,
        "{actual} != {expected}"
    );
}

fn assert_expected_solution(solution: &faer::Mat<f64>) {
    let expected = [[1.0, 2.0], [2.0, -1.0], [3.0, 4.0]];
    for row in 0..3 {
        for col in 0..2 {
            assert_close(solution[(row, col)], expected[row][col]);
        }
    }
}

#[test]
fn csc_materialization_preserves_shape_entries_and_contiguous_values() {
    let matrix = matrix([4.0, 1.0, 1.0, 3.0, 2.0]);

    assert_eq!((matrix.nrows(), matrix.ncols()), (3, 3));
    assert_eq!(matrix.compute_nnz(), 5);
    assert_eq!(matrix.val().len(), 5);
    assert_close(matrix[(0, 0)], 4.0);
    assert_close(matrix[(2, 2)], 2.0);
}

#[test]
fn sparse_qr_reuses_symbolic_pattern_for_changed_values_and_batched_rhs() {
    let first = matrix([4.0, 1.0, 1.0, 3.0, 2.0]);
    let second = matrix([5.0, 1.0, 1.0, 4.0, 3.0]);
    let symbolic = SymbolicQr::try_new(first.symbolic()).unwrap();
    let first_factor = Qr::try_new_with_symbolic(symbolic.clone(), first.as_ref()).unwrap();
    let second_factor = Qr::try_new_with_symbolic(symbolic, second.as_ref()).unwrap();

    let first_rhs = mat![[6.0, 7.0], [7.0, -1.0], [6.0, 8.0]];
    let second_rhs = mat![[7.0, 9.0], [9.0, -2.0], [9.0, 12.0]];
    let first_solution = first_factor.solve(first_rhs.as_ref());
    let second_solution = second_factor.solve(second_rhs.as_ref());
    assert_expected_solution(&first_solution);
    assert_expected_solution(&second_solution);
}

#[test]
fn sparse_qr_solves_rectangular_batched_least_squares() {
    let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(
        3,
        2,
        &[
            Triplet::new(0, 0, 1.0),
            Triplet::new(2, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 1, 1.0),
        ],
    )
    .unwrap();
    let factor = Qr::try_new_with_symbolic(
        SymbolicQr::try_new(matrix.symbolic()).unwrap(),
        matrix.as_ref(),
    )
    .unwrap();
    let rhs = mat![[1.0, 2.0], [2.0, -1.0], [3.0, 1.0]];
    let solution = factor.solve_lstsq(rhs.as_ref());

    assert_eq!((solution.nrows(), solution.ncols()), (2, 2));
    for (actual, expected) in [
        (solution[(0, 0)], 1.0),
        (solution[(1, 0)], 2.0),
        (solution[(0, 1)], 2.0),
        (solution[(1, 1)], -1.0),
    ] {
        assert_close(actual, expected);
    }
}

#[test]
fn retained_factors_are_send_and_sync_for_binary64() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Qr<usize, f64>>();
    assert_send_sync::<SymbolicQr<usize>>();
}

#[test]
fn dense_profile_covers_direct_least_squares_and_rank_revealing_fallbacks() {
    let square = mat![[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]];
    let rhs = mat![[6.0, 7.0], [7.0, -1.0], [6.0, 8.0]];
    assert_expected_solution(
        &square
            .as_ref()
            .llt(Side::Lower)
            .unwrap()
            .solve(rhs.as_ref()),
    );
    assert_expected_solution(&square.as_ref().partial_piv_lu().solve(rhs.as_ref()));
    assert_expected_solution(&square.as_ref().qr().solve(rhs.as_ref()));
    assert_expected_solution(&square.as_ref().col_piv_qr().solve(rhs.as_ref()));
    assert_expected_solution(&square.as_ref().thin_svd().unwrap().solve(rhs.as_ref()));

    let rectangular = mat![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let rhs = mat![[1.0, 2.0], [2.0, -1.0], [3.0, 1.0]];
    let qr_solution = rectangular.as_ref().col_piv_qr().solve_lstsq(rhs.as_ref());
    let svd_solution = rectangular
        .as_ref()
        .thin_svd()
        .unwrap()
        .solve_lstsq(rhs.as_ref());
    for (actual, expected) in [
        (qr_solution[(0, 0)], 1.0),
        (qr_solution[(1, 0)], 2.0),
        (svd_solution[(0, 1)], 2.0),
        (svd_solution[(1, 1)], -1.0),
    ] {
        assert_close(actual, expected);
    }
}

#[test]
fn sparse_direct_profile_covers_llt_and_lu_with_batched_rhs() {
    let matrix = matrix([4.0, 1.0, 1.0, 3.0, 2.0]);
    let rhs = mat![[6.0, 7.0], [7.0, -1.0], [6.0, 8.0]];
    let llt = Llt::try_new_with_symbolic(
        SymbolicLlt::try_new(matrix.symbolic(), Side::Lower).unwrap(),
        matrix.as_ref(),
        Side::Lower,
    )
    .unwrap();
    let lu = Lu::try_new_with_symbolic(
        SymbolicLu::try_new(matrix.symbolic()).unwrap(),
        matrix.as_ref(),
    )
    .unwrap();

    assert_expected_solution(&llt.solve(rhs.as_ref()));
    assert_expected_solution(&lu.solve(rhs.as_ref()));
}

#[test]
fn sparse_lu_singularity_requires_problem_level_candidate_validation() {
    let singular = SparseColMat::<usize, f64>::try_new_from_triplets(
        2,
        2,
        &[
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 0, 2.0),
            Triplet::new(0, 1, 2.0),
            Triplet::new(1, 1, 4.0),
        ],
    )
    .unwrap();
    let factor = Lu::try_new_with_symbolic(
        SymbolicLu::try_new(singular.symbolic()).unwrap(),
        singular.as_ref(),
    )
    .unwrap();
    let solution = factor.solve(mat![[1.0], [0.0]].as_ref());
    let residual = (solution[(0, 0)] + 2.0 * solution[(1, 0)] - 1.0)
        .hypot(2.0 * solution[(0, 0)] + 4.0 * solution[(1, 0)]);

    assert!(
        !solution[(0, 0)].is_finite() || !solution[(1, 0)].is_finite() || residual > 1.0e-10,
        "a backend-successful singular solve must fail candidate validation"
    );
}
