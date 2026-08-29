use super::{
    BigInt, BigRational, DenseSquare, RealizationError, binary64_from_exact_rounded,
    determinant_f64, exact_from_binary64,
};

#[test]
fn exact_binary64_round_trip_preserves_all_boundary_classes() {
    for value in [
        0.0,
        -0.0,
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        -1.5,
        1.0,
        f64::MAX,
    ] {
        let rounded = binary64_from_exact_rounded(&exact_from_binary64(value)).unwrap();
        let expected = if value == 0.0 { 0.0 } else { value };
        assert_eq!(rounded.to_bits(), expected.to_bits());
    }
}

#[test]
fn exact_to_binary64_uses_ties_to_even_and_rejects_nonzero_underflow() {
    let denominator = BigInt::from(1_u8) << 53_usize;
    let tie_to_one = BigRational::new(&denominator + 1_u8, denominator.clone());
    let tie_to_next_even = BigRational::new(&denominator + 3_u8, denominator);
    let below_tie =
        &tie_to_one - BigRational::new(BigInt::from(1_u8), BigInt::from(1_u8) << 200_usize);
    let above_tie =
        &tie_to_one + BigRational::new(BigInt::from(1_u8), BigInt::from(1_u8) << 200_usize);
    assert_eq!(
        binary64_from_exact_rounded(&tie_to_one).unwrap().to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        binary64_from_exact_rounded(&tie_to_next_even)
            .unwrap()
            .to_bits(),
        1.0_f64.to_bits() + 2
    );
    assert_eq!(
        binary64_from_exact_rounded(&below_tie).unwrap().to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        binary64_from_exact_rounded(&above_tie).unwrap().to_bits(),
        1.0_f64.to_bits() + 1
    );

    let half_min_subnormal = BigRational::new(BigInt::from(1_u8), BigInt::from(1_u8) << 1075_usize);
    assert_eq!(
        binary64_from_exact_rounded(&half_min_subnormal),
        Err(RealizationError::Unrepresentable)
    );
    let overflow = exact_from_binary64(f64::MAX) * BigInt::from(2_u8);
    assert_eq!(
        binary64_from_exact_rounded(&overflow),
        Err(RealizationError::Unrepresentable)
    );
}

fn nested_determinant(mut matrix: Vec<Vec<f64>>) -> f64 {
    let mut determinant = 1.0;
    for column in 0..matrix.len() {
        let pivot = (column..matrix.len())
            .max_by(|left, right| {
                matrix[*left][column]
                    .abs()
                    .total_cmp(&matrix[*right][column].abs())
            })
            .unwrap();
        matrix.swap(column, pivot);
        let pivot_value = matrix[column][column];
        determinant *= pivot_value;
        let (pivot_rows, trailing_rows) = matrix.split_at_mut(column + 1);
        let pivot_row = &pivot_rows[column];
        for target in trailing_rows {
            let factor = target[column] / pivot_value;
            for (target, pivot) in target[column + 1..]
                .iter_mut()
                .zip(&pivot_row[column + 1..])
            {
                *target -= factor * pivot;
            }
        }
    }
    determinant
}

#[test]
fn dense_layout_matches_an_independent_nested_determinant() {
    const ORDER: usize = 5;
    let flat = DenseSquare::try_from_fn(ORDER, |row, column| {
        if row == column {
            6.0
        } else {
            1.0 / super::float(row + column + 1)
        }
    })
    .unwrap();
    let nested = flat
        .values
        .chunks_exact(ORDER)
        .map(<[f64]>::to_vec)
        .collect::<Vec<_>>();

    assert_eq!(
        determinant_f64(&mut flat.clone()).to_bits(),
        nested_determinant(nested).to_bits()
    );
}
