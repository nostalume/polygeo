use std::cmp::Ordering;

use num_bigint::BigInt;
use num_traits::{Signed, Zero};

use crate::form_impl::{dyadic, next_up, rounded_dyadic};

pub(crate) fn exact_dot_is_zero(left: &[f64], right: &[f64]) -> bool {
    let filter = adaptive_product_sum(left.iter().copied().zip(right.iter().copied()), 0.0);
    filter.accepted
}

#[derive(Clone, Copy)]
pub(crate) struct AdaptiveVerdict {
    pub(crate) accepted: bool,
    pub(crate) bound: f64,
    pub(crate) exact_fallback: bool,
}

pub(crate) fn adaptive_product_sum(
    terms: impl Clone + Iterator<Item = (f64, f64)>,
    tolerance: f64,
) -> AdaptiveVerdict {
    adaptive_scalar_product_sum(terms.map(|(left, right)| [left, right]), tolerance)
}

pub(crate) fn adaptive_triple_product_sum(
    terms: impl Clone + Iterator<Item = (f64, f64, f64)>,
    tolerance: f64,
) -> AdaptiveVerdict {
    adaptive_scalar_product_sum(
        terms.map(|(first, second, third)| [first, second, third]),
        tolerance,
    )
}

fn adaptive_scalar_product_sum<const N: usize>(
    terms: impl Clone + Iterator<Item = [f64; N]>,
    tolerance: f64,
) -> AdaptiveVerdict {
    if !tolerance.is_finite() {
        return AdaptiveVerdict {
            accepted: false,
            bound: f64::INFINITY,
            exact_fallback: false,
        };
    }
    let mut sum = 0.0_f64;
    let mut magnitude = 0.0_f64;
    for factors in terms.clone() {
        let product = factors.into_iter().product::<f64>();
        sum += product;
        magnitude = next_up(magnitude + product.abs());
    }
    let operations = terms.clone().count().saturating_mul(N).saturating_add(1);
    let operation_count = f64::from(u32::try_from(operations).unwrap_or(u32::MAX));
    // One product and one addition per term are covered using epsilon rather
    // than unit roundoff; every bound operation is then rounded outward.
    let gamma = operation_count * f64::EPSILON;
    let bound = if gamma >= 1.0 {
        f64::INFINITY
    } else {
        next_up(next_up(gamma / (1.0 - gamma)) * magnitude + operation_count * f64::from_bits(1))
    };
    if next_up(sum.abs() + bound) <= tolerance {
        return AdaptiveVerdict {
            accepted: true,
            bound: next_up(sum.abs() + bound),
            exact_fallback: false,
        };
    }
    if sum.abs() > next_up(tolerance + bound) {
        return AdaptiveVerdict {
            accepted: false,
            bound: next_up(sum.abs() + bound),
            exact_fallback: false,
        };
    }
    let (value, exponent) = exact_scalar_product_sum(terms);
    AdaptiveVerdict {
        accepted: exact_abs_le(&value, exponent, tolerance),
        bound: next_up(sum.abs() + bound),
        exact_fallback: true,
    }
}

pub(crate) fn adaptive_product_value(
    terms: impl Clone + Iterator<Item = (f64, f64)>,
) -> Option<(f64, bool)> {
    let term_count = terms
        .clone()
        .filter(|&(left, right)| left != 0.0 && right != 0.0)
        .count();
    if term_count == 0 {
        return Some((0.0, false));
    }
    let mut sum = 0.0_f64;
    let mut magnitude = 0.0_f64;
    for (left, right) in terms.clone() {
        let product = left * right;
        sum += product;
        magnitude = next_up(magnitude + product.abs());
    }
    let count = term_count.saturating_mul(2).saturating_add(1);
    let count = f64::from(u32::try_from(count).unwrap_or(u32::MAX));
    let bound = next_up(count * f64::EPSILON * magnitude);
    if sum.is_finite() && sum.abs() > 8.0 * bound {
        return Some((sum, false));
    }
    let (value, exponent) = exact_product_sum(terms);
    let rounded = rounded_dyadic(&value, exponent)?;
    rounded.is_finite().then_some((rounded, true))
}

pub(crate) fn adaptive_product_sign(
    terms: impl Clone + Iterator<Item = (f64, f64)>,
) -> Option<(Ordering, bool)> {
    if terms
        .clone()
        .any(|(left, right)| !left.is_finite() || !right.is_finite())
    {
        return None;
    }
    let mut sum = 0.0_f64;
    let mut magnitude = 0.0_f64;
    let mut count = 0_usize;
    for (left, right) in terms.clone() {
        let product = left * right;
        sum += product;
        magnitude = next_up(magnitude + product.abs());
        count = count.saturating_add(1);
    }
    let operations = count.saturating_mul(2).saturating_add(1);
    let operations = f64::from(u32::try_from(operations).unwrap_or(u32::MAX));
    let bound = next_up(operations * f64::EPSILON * magnitude);
    if sum.is_finite() && sum.abs() > 8.0 * bound {
        return Some((sum.total_cmp(&0.0), false));
    }
    let (value, _) = exact_product_sum(terms);
    Some((value.cmp(&BigInt::zero()), true))
}

fn exact_product_sum(terms: impl Clone + Iterator<Item = (f64, f64)>) -> (BigInt, i32) {
    exact_scalar_product_sum(terms.map(|(left, right)| [left, right]))
}

fn exact_scalar_product_sum<const N: usize>(
    terms: impl Clone + Iterator<Item = [f64; N]>,
) -> (BigInt, i32) {
    let exponent = terms
        .clone()
        .filter(|factors| factors.iter().all(|&factor| factor != 0.0))
        .map(|factors| factors.into_iter().map(|factor| dyadic(factor).1).sum())
        .min();
    let Some(exponent) = exponent else {
        return (BigInt::zero(), 0);
    };
    let value = terms
        .filter(|factors| factors.iter().all(|&factor| factor != 0.0))
        .map(|factors| {
            factors.into_iter().map(dyadic).fold(
                (BigInt::from(1), 0),
                |(value, exponent), (factor, factor_exponent)| {
                    (value * factor, exponent + factor_exponent)
                },
            )
        })
        .fold(BigInt::zero(), |sum, (value, shift)| {
            sum + (value << usize::try_from(shift - exponent).expect("nonnegative dyadic shift"))
        });
    (value, exponent)
}

fn exact_abs_le(value: &BigInt, exponent: i32, tolerance: f64) -> bool {
    if tolerance.is_infinite() {
        return true;
    }
    let (limit, limit_exponent) = dyadic(tolerance);
    if exponent >= limit_exponent {
        (value.abs() << usize::try_from(exponent - limit_exponent).expect("nonnegative shift"))
            <= limit
    } else {
        value.abs()
            <= (limit << usize::try_from(limit_exponent - exponent).expect("nonnegative shift"))
    }
}

#[cfg(test)]
mod tests {
    use super::{adaptive_product_sum, adaptive_product_value};

    #[test]
    fn adaptive_filter_and_exact_fallback_make_the_same_threshold_decision() {
        let hidden = [(1.0, 1.0), (2.0_f64.powi(-53), 1.0), (-1.0, 1.0)];
        let rejected = adaptive_product_sum(hidden.into_iter(), 0.0);
        assert!(!rejected.accepted);
        assert!(rejected.exact_fallback);

        let cancelled = [(1.0, 1.0), (-1.0, 1.0)];
        let exact = adaptive_product_sum(cancelled.into_iter(), 0.0);
        assert!(exact.accepted);
        assert!(exact.exact_fallback);

        let filtered = adaptive_product_sum(cancelled.into_iter(), 1.0e-12);
        assert!(filtered.accepted);
        assert!(!filtered.exact_fallback);
    }

    #[test]
    fn exact_value_fallback_rounds_wide_and_subnormal_dyadics_once() {
        let cancelled = [(f64::MAX, 1.0), (-f64::MAX, 1.0), (1.0, 1.0)];
        assert_eq!(
            adaptive_product_value(cancelled.into_iter()),
            Some((1.0, true))
        );

        let half_subnormal = [(f64::MIN_POSITIVE, 2.0_f64.powi(-53))];
        assert_eq!(
            adaptive_product_value(half_subnormal.into_iter()),
            Some((0.0, true))
        );
        let tie_to_even = [(f64::MIN_POSITIVE, 3.0 * 2.0_f64.powi(-53))];
        assert_eq!(
            adaptive_product_value(tie_to_even.into_iter()),
            Some((2.0 * f64::from_bits(1), true))
        );
    }
}
