use std::{marker::PhantomData, sync::Arc};

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{FromPrimitive, One, Signed, ToPrimitive, Zero};

use crate::chain::{BasedDegree, ChainDomain, visit_wedge_face_pairs, wedge_normalization};
use crate::{
    BigIntEncoding, CanonicalSelection, Chain, Cochain, CoefficientSystem, ComplexCore, Element,
    IntegerRing, Space, TopologyError, Variance,
};

/// Failure to admit or explicitly realize a binary64 element.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Binary64ElementError {
    CoefficientCount,
    NonFinite,
    SpaceMismatch,
    ScalarConversion,
    Allocation,
    Topology(TopologyError),
}

impl Binary64ElementError {
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::CoefficientCount => "coefficient_count",
            Self::NonFinite => "non_finite",
            Self::SpaceMismatch => "space_mismatch",
            Self::ScalarConversion => "scalar_conversion",
            Self::Allocation => "allocation",
            Self::Topology(error) => error.reason(),
        }
    }
}

impl std::fmt::Display for Binary64ElementError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CoefficientCount => {
                formatter.write_str("coefficients do not align with the binary64 space")
            }
            Self::NonFinite => formatter.write_str("coefficients must be finite binary64 values"),
            Self::SpaceMismatch => {
                formatter.write_str("element belongs to a different based space")
            }
            Self::ScalarConversion => {
                formatter.write_str("exact coefficients cannot be realized as binary64")
            }
            Self::Allocation => formatter.write_str("binary64 output allocation failed"),
            Self::Topology(error) => std::fmt::Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for Binary64ElementError {}

impl From<TopologyError> for Binary64ElementError {
    fn from(error: TopologyError) -> Self {
        Self::Topology(error)
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Binary64Basis {
    Full(BasedDegree),
    Selected(Arc<CanonicalSelection>),
}

impl Binary64Basis {
    pub(crate) fn degree(&self) -> isize {
        match self {
            Self::Full(basis) => basis.degree,
            Self::Selected(selection) => selected_degree(selection),
        }
    }

    pub(crate) fn size(&self) -> usize {
        match self {
            Self::Full(basis) => basis.basis_size,
            Self::Selected(selection) => selection.len(),
        }
    }

    pub(crate) fn same_basis(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Full(left), Self::Full(right)) => left.same_based_module(right),
            (Self::Selected(left), Self::Selected(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }

    pub(crate) const fn full(&self) -> Option<&BasedDegree> {
        match self {
            Self::Full(basis) => Some(basis),
            Self::Selected(_) => None,
        }
    }

    pub(crate) const fn selection(&self) -> Option<&Arc<CanonicalSelection>> {
        match self {
            Self::Selected(selection) => Some(selection),
            Self::Full(_) => None,
        }
    }
}

/// One coefficient-independent basis interpreted over binary64 coordinates.
#[derive(Clone, Debug)]
pub struct Binary64Space<K: Variance> {
    basis: Binary64Basis,
    variance: PhantomData<fn() -> K>,
}

impl<K: Variance> Binary64Space<K> {
    /// Derive a full simplicial basis without copying topology data.
    ///
    /// # Errors
    ///
    /// Returns a topology error when `degree` is not represented.
    pub fn full(owner: Arc<ComplexCore>, degree: usize) -> Result<Self, TopologyError> {
        let basis_size = owner.chain_view().basis_size(degree)?;
        Ok(Self::from_full(BasedDegree {
            domain: ChainDomain::Simplicial(owner),
            degree: isize::try_from(degree).map_err(|_| TopologyError::CountOverflow)?,
            basis_size,
        }))
    }

    /// Retain the coefficient-independent basis of an existing exact space.
    #[must_use]
    pub fn from_basis<A: CoefficientSystem>(basis: &Space<A, K>) -> Self {
        Self::from_full(BasedDegree::new(basis))
    }

    /// Retain one canonical simplicial sub-basis.
    ///
    /// # Errors
    ///
    /// Returns an overflow error when its degree exceeds the exact degree domain.
    pub fn selected(selection: Arc<CanonicalSelection>) -> Result<Self, TopologyError> {
        isize::try_from(selection.degree()).map_err(|_| TopologyError::CountOverflow)?;
        Ok(Self {
            basis: Binary64Basis::Selected(selection),
            variance: PhantomData,
        })
    }

    /// Derive a zero simplicial module after the represented complex.
    ///
    /// # Errors
    ///
    /// Returns an error unless `degree` lies beyond the represented complex.
    pub fn zero(owner: Arc<ComplexCore>, degree: usize) -> Result<Self, TopologyError> {
        if degree <= owner.dimension() {
            return Err(TopologyError::degree_outside(degree));
        }
        Ok(Self::from_full(BasedDegree {
            domain: ChainDomain::Simplicial(owner),
            degree: isize::try_from(degree).map_err(|_| TopologyError::CountOverflow)?,
            basis_size: 0,
        }))
    }

    pub(crate) fn from_full(basis: BasedDegree) -> Self {
        Self {
            basis: Binary64Basis::Full(basis),
            variance: PhantomData,
        }
    }

    #[must_use]
    pub fn degree(&self) -> isize {
        self.basis.degree()
    }

    #[must_use]
    pub fn size(&self) -> usize {
        self.basis.size()
    }

    #[must_use]
    pub const fn variance(&self) -> &'static str {
        K::NAME
    }

    #[must_use]
    pub const fn is_full(&self) -> bool {
        matches!(self.basis, Binary64Basis::Full(_))
    }

    #[must_use]
    pub fn same_space(&self, other: &Self) -> bool {
        self.same_basis(other)
    }

    pub(crate) fn same_basis<L: Variance>(&self, other: &Binary64Space<L>) -> bool {
        self.basis.same_basis(&other.basis)
    }

    pub(crate) fn full_basis(&self) -> Option<&BasedDegree> {
        self.basis.full()
    }

    /// Borrow the canonical selection carried by a selected space.
    #[must_use]
    pub fn canonical_selection(&self) -> Option<&Arc<CanonicalSelection>> {
        self.basis.selection()
    }

    pub(crate) const fn basis(&self) -> &Binary64Basis {
        &self.basis
    }
}

fn selected_degree(selection: &CanonicalSelection) -> isize {
    isize::try_from(selection.degree()).expect("selected degree was admitted")
}

/// Immutable contiguous binary64 coordinates in one variance-indexed basis.
///
/// Chain and cochain indices cannot be mixed:
///
/// ```compile_fail
/// use polygeo_core::{Binary64ChainSpace, Binary64Cochain, Binary64Element, Binary64Space, Variance};
/// fn same<K: Variance>(_: &Binary64Space<K>, _: &Binary64Element<K>) {}
/// fn mismatch(space: &Binary64ChainSpace, value: &Binary64Cochain) {
///     same(space, value);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Binary64Element<K: Variance> {
    space: Binary64Space<K>,
    coefficients: Arc<[f64]>,
}

pub type Binary64ChainSpace = Binary64Space<Chain>;
pub type Binary64CochainSpace = Binary64Space<Cochain>;
pub type Binary64Chain = Binary64Element<Chain>;
pub type Binary64Cochain = Binary64Element<Cochain>;

impl<K: Variance> Binary64Element<K> {
    /// Admit one owned contiguous coefficient buffer.
    ///
    /// # Errors
    ///
    /// Returns a count or finite-value admission failure.
    pub fn admit(
        space: Binary64Space<K>,
        coefficients: Vec<f64>,
    ) -> Result<Self, Binary64ElementError> {
        if coefficients.len() != space.size() {
            return Err(Binary64ElementError::CoefficientCount);
        }
        if coefficients.iter().any(|value| !value.is_finite()) {
            return Err(Binary64ElementError::NonFinite);
        }
        Ok(Self {
            space,
            coefficients: coefficients.into(),
        })
    }

    /// Explicitly realize exact integral coordinates as binary64.
    ///
    /// # Errors
    ///
    /// Returns a space mismatch or unrepresentable scalar failure.
    pub fn realize_integral(
        space: Binary64Space<K>,
        exact: &Element<IntegerRing, K, BigIntEncoding>,
    ) -> Result<Self, Binary64ElementError> {
        let Some(basis) = space.full_basis() else {
            return Err(Binary64ElementError::SpaceMismatch);
        };
        if !basis.same_space(exact.space()) {
            return Err(Binary64ElementError::SpaceMismatch);
        }
        let mut coefficients = vec![0.0; space.size()];
        for (&index, value) in exact.indices().iter().zip(exact.coefficients()) {
            coefficients[index] =
                exact_integer_to_binary64(value).ok_or(Binary64ElementError::ScalarConversion)?;
        }
        Self::admit(space, coefficients)
    }

    #[must_use]
    pub const fn space(&self) -> &Binary64Space<K> {
        &self.space
    }

    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Return the additive inverse in the same binary64 space.
    #[must_use]
    pub fn negated(&self) -> Self {
        Self {
            space: self.space.clone(),
            coefficients: self
                .coefficients
                .iter()
                .map(|value| -*value)
                .collect::<Vec<_>>()
                .into(),
        }
    }

    pub(crate) fn from_shared(space: Binary64Space<K>, coefficients: Arc<[f64]>) -> Self {
        Self {
            space,
            coefficients,
        }
    }

    pub(crate) const fn shared_coefficients(&self) -> &Arc<[f64]> {
        &self.coefficients
    }
}

impl Binary64Element<Cochain> {
    /// Form the antisymmetrized simplicial wedge product in binary64 arithmetic.
    ///
    /// # Errors
    ///
    /// Rejects selected bases, foreign owners, invalid topology, allocation
    /// failure, degree overflow, or a non-finite result.
    pub fn wedge(&self, other: &Self) -> Result<Self, Binary64ElementError> {
        let left = self
            .space
            .full_basis()
            .ok_or(Binary64ElementError::SpaceMismatch)?;
        let right = other
            .space
            .full_basis()
            .ok_or(Binary64ElementError::SpaceMismatch)?;
        if !left.domain.same_owner(&right.domain) {
            return Err(Binary64ElementError::SpaceMismatch);
        }
        let owner = left
            .domain
            .simplicial_owner()
            .ok_or(Binary64ElementError::SpaceMismatch)?;
        let left_degree = usize::try_from(left.degree).map_err(|_| TopologyError::CountOverflow)?;
        let right_degree =
            usize::try_from(right.degree).map_err(|_| TopologyError::CountOverflow)?;
        let target_degree = left_degree
            .checked_add(right_degree)
            .ok_or(TopologyError::CountOverflow)?;
        let target = if target_degree <= owner.dimension() {
            Binary64Space::full(Arc::clone(owner), target_degree)?
        } else {
            Binary64Space::zero(Arc::clone(owner), target_degree)?
        };
        if self.coefficients.is_empty() || other.coefficients.is_empty() || target.size() == 0 {
            return Self::admit(target, Vec::new());
        }

        let normalization = wedge_normalization(left_degree, right_degree)?
            .to_f64()
            .ok_or(Binary64ElementError::ScalarConversion)?;
        let mut sums = Vec::new();
        let mut corrections = Vec::new();
        sums.try_reserve_exact(target.size())
            .map_err(|_| Binary64ElementError::Allocation)?;
        corrections
            .try_reserve_exact(target.size())
            .map_err(|_| Binary64ElementError::Allocation)?;
        sums.resize(target.size(), 0.0_f64);
        corrections.resize(target.size(), 0.0_f64);
        let mut finite = true;
        visit_wedge_face_pairs(owner, left_degree, right_degree, |target_index, pair| {
            let left = self.coefficients[pair.left];
            let right = other.coefficients[pair.right];
            let mut term = if left.abs() >= right.abs() {
                (left / normalization) * right
            } else {
                left * (right / normalization)
            };
            if pair.sign < 0 {
                term = -term;
            }
            let sum = sums[target_index];
            let next = sum + term;
            let correction = if sum.abs() >= term.abs() {
                (sum - next) + term
            } else {
                (term - next) + sum
            };
            sums[target_index] = next;
            corrections[target_index] += correction;
            finite &= term.is_finite() && next.is_finite() && corrections[target_index].is_finite();
        })?;
        if !finite {
            return Err(Binary64ElementError::NonFinite);
        }
        for (sum, correction) in sums.iter_mut().zip(corrections) {
            *sum += correction;
            if !sum.is_finite() {
                return Err(Binary64ElementError::NonFinite);
            }
        }
        Self::admit(target, sums)
    }
}

fn exact_integer_to_binary64(value: &BigInt) -> Option<f64> {
    let binary64 = value.to_f64().filter(|value| value.is_finite())?;
    (BigInt::from_f64(binary64).as_ref() == Some(value)).then_some(binary64)
}

pub(crate) fn exact_integer_binary64_dot(
    indices: &[usize],
    exact: &[BigInt],
    dense: &[f64],
) -> Option<f64> {
    adaptive_integer_binary64_dot(indices, exact, dense).map(|(value, _)| value)
}

fn adaptive_integer_binary64_dot(
    indices: &[usize],
    exact: &[BigInt],
    dense: &[f64],
) -> Option<(f64, bool)> {
    let terms = indices
        .iter()
        .copied()
        .zip(exact)
        .filter(|&(index, coefficient)| !coefficient.is_zero() && dense[index] != 0.0);
    let mut sum = 0.0_f64;
    let mut magnitude = 0.0_f64;
    let mut count = 0_usize;
    let mut fast = true;
    let mut single_unit = false;
    for (index, coefficient) in terms.clone() {
        let unit = coefficient.abs().is_one();
        let Some(coefficient) = exact_integer_to_binary64(coefficient) else {
            fast = false;
            break;
        };
        let product = coefficient * dense[index];
        sum += product;
        magnitude = next_up(magnitude + product.abs());
        count += 1;
        single_unit = count == 1 && unit;
    }
    if fast && count == 1 && single_unit && sum.is_finite() {
        return Some((sum, false));
    }
    if fast && sum.is_finite() && sum != 0.0 {
        let operations = count.saturating_mul(2).saturating_add(1);
        let operations = f64::from(u32::try_from(operations).unwrap_or(u32::MAX));
        let gamma = operations * f64::EPSILON;
        let bound = if gamma < 1.0 {
            next_up(gamma / (1.0 - gamma) * magnitude + operations * f64::from_bits(1))
        } else {
            f64::INFINITY
        };
        let rounding_margin = ((sum - next_down(sum)) * 0.5).min((next_up(sum) - sum) * 0.5);
        if bound < rounding_margin {
            return Some((sum, false));
        }
    }
    exact_integer_binary64_dot_fallback(terms, dense).map(|value| (value, true))
}

fn exact_integer_binary64_dot_fallback<'a>(
    terms: impl Clone + Iterator<Item = (usize, &'a BigInt)>,
    dense: &[f64],
) -> Option<f64> {
    let exponent = terms.clone().map(|(index, _)| dyadic(dense[index]).1).min();
    let Some(exponent) = exponent else {
        return Some(0.0);
    };
    let value = terms.fold(BigInt::ZERO, |sum, (index, coefficient)| {
        let (significand, shift) = dyadic(dense[index]);
        sum + ((coefficient * significand)
            << usize::try_from(shift - exponent).expect("nonnegative"))
    });
    let rounded = rounded_dyadic(&value, exponent)?;
    (rounded.is_finite() && (rounded != 0.0 || value.is_zero())).then_some(rounded)
}

pub(crate) fn rounded_dyadic(value: &BigInt, exponent: i32) -> Option<f64> {
    if value.is_zero() {
        return Some(0.0);
    }
    let magnitude = value.magnitude();
    let bits = magnitude.bits();
    let highest = i64::from(exponent) + i64::try_from(bits).ok()? - 1;
    let rounded = if highest >= -1022 {
        let shift = bits.saturating_sub(53);
        let significand = rounded_shift_right(magnitude, shift)?;
        let scale = i64::from(exponent) + i64::try_from(shift).ok()?;
        significand.to_f64()? * 2.0_f64.powi(i32::try_from(scale).ok()?)
    } else {
        let unit_shift = i64::from(exponent) + 1074;
        let units = if unit_shift >= 0 {
            magnitude << usize::try_from(unit_shift).ok()?
        } else {
            rounded_shift_right(magnitude, u64::try_from(-unit_shift).ok()?)?
        };
        units.to_f64()? * f64::from_bits(1)
    };
    Some(if value.sign() == Sign::Minus {
        -rounded
    } else {
        rounded
    })
}

fn rounded_shift_right(value: &BigUint, shift: u64) -> Option<BigUint> {
    if shift == 0 {
        return Some(value.clone());
    }
    let shift = usize::try_from(shift).ok()?;
    let quotient = value >> shift;
    let remainder = value - (&quotient << shift);
    let half = BigUint::from(1_u8) << (shift - 1);
    Some(
        if remainder > half || remainder == half && quotient.bit(0) {
            quotient + 1_u8
        } else {
            quotient
        },
    )
}

pub(crate) fn dyadic(value: f64) -> (BigInt, i32) {
    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let exponent = i32::try_from((bits >> 52) & 0x7ff).expect("binary64 exponent fits i32");
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, exponent) = if exponent == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, exponent - 1023 - 52)
    };
    let value = BigInt::from(significand);
    (if negative { -value } else { value }, exponent)
}

pub(crate) fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == -0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value >= 0.0 { bits + 1 } else { bits - 1 })
}

fn next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

impl Binary64Element<Cochain> {
    /// Apply retained incidence transpose directly, without CSR materialization.
    ///
    /// # Errors
    ///
    /// Returns an operator construction or evaluation failure.
    pub fn exterior_derivative(&self) -> Result<Self, crate::OperatorError> {
        self.space.exterior_derivative()?.apply(self)
    }
}

#[cfg(test)]
mod tests {
    use super::adaptive_integer_binary64_dot;
    use num_bigint::BigInt;

    #[test]
    fn integer_binary64_dot_agrees_across_fast_and_exact_paths() {
        let unit = [BigInt::from(1)];
        let fast = adaptive_integer_binary64_dot(&[0], &unit, &[3.0]).unwrap();
        assert_eq!(fast.0.to_bits(), 3.0_f64.to_bits());
        assert!(!fast.1);

        let coefficients = [BigInt::from(1), BigInt::from(1), BigInt::from(1)];
        let exact = adaptive_integer_binary64_dot(
            &[0, 1, 2],
            &coefficients,
            &[1.0, 2.0_f64.powi(-53), -1.0],
        )
        .unwrap();
        assert_eq!(exact.0.to_bits(), 2.0_f64.powi(-53).to_bits());
        assert!(exact.1);

        let subnormal = adaptive_integer_binary64_dot(&[0], &unit, &[f64::from_bits(1)]).unwrap();
        assert_eq!(subnormal.0.to_bits(), 1);
        assert!(!subnormal.1);
        assert!(
            adaptive_integer_binary64_dot(
                &[0, 1],
                &[BigInt::from(1), BigInt::from(1)],
                &[f64::MAX, f64::MAX]
            )
            .is_none()
        );
    }
}
