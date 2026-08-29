use std::collections::BTreeMap;
use std::mem::size_of;
use std::sync::Arc;

use crate::chain::{
    AtomicRecipe, BasedDegree, ChainDomain, ChainError, CompositionPlan, Element, LinearMap,
    MapRecipe, Space, Variance,
};
use crate::coefficient::{
    BigIntEncoding, IntegerRing, RationalField, ReducedFractionEncoding, Ring, ValueEncoding,
};
use crate::correspondence::SignedPermutation;
pub use crate::sparse::{CsrMatrix, CsrPattern};
use crate::{BoundaryRef, CoefficientSlice, StorageLimit, TopologyError, WorkLimit};
use num_bigint::BigInt;

type EncodedCsr<A, E> = CsrMatrix<Box<[<E as ValueEncoding<A>>::Stored]>>;

trait CsrEncoding<A: Ring>: ValueEncoding<A> {
    fn logical_value_bytes(coefficient_bits: u64) -> Result<u64, RepresentationError>;
}

impl CsrEncoding<IntegerRing> for BigIntEncoding {
    fn logical_value_bytes(coefficient_bits: u64) -> Result<u64, RepresentationError> {
        bigint_logical_bytes(coefficient_bits)
    }
}

impl CsrEncoding<RationalField> for ReducedFractionEncoding {
    fn logical_value_bytes(coefficient_bits: u64) -> Result<u64, RepresentationError> {
        bigint_logical_bytes(coefficient_bits)?
            .checked_add(bigint_logical_bytes(1)?)
            .ok_or(RepresentationError::Overflow)
    }
}

fn bigint_logical_bytes(coefficient_bits: u64) -> Result<u64, RepresentationError> {
    coefficient_bits
        .div_ceil(u64::from(usize::BITS))
        .checked_mul(size_of::<usize>() as u64)
        .and_then(|digits| (size_of::<BigInt>() as u64).checked_add(digits))
        .ok_or(RepresentationError::Overflow)
}

/// Deterministic pre-allocation bound for one CSR construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CsrEstimate {
    shape: (usize, usize),
    nnz_bound: usize,
    coefficient_bits_bound: u64,
    retained_logical_bytes_bound: u64,
    peak_live_logical_bytes_bound: u64,
    scratch_entries_bound: usize,
    scalar_steps_bound: u64,
    canonicalization_required: bool,
}

impl CsrEstimate {
    /// Output matrix shape.
    #[must_use]
    pub const fn shape(self) -> (usize, usize) {
        self.shape
    }

    /// Exact nonzero count or safe output upper bound.
    #[must_use]
    pub const fn nnz_bound(self) -> usize {
        self.nnz_bound
    }

    /// Safe bound for retained and temporary logical storage bytes.
    #[must_use]
    pub const fn coefficient_bits_bound(self) -> u64 {
        self.coefficient_bits_bound
    }

    /// Logical bytes retained by the published CSR product.
    #[must_use]
    pub const fn retained_logical_bytes_bound(self) -> u64 {
        self.retained_logical_bytes_bound
    }

    /// Maximum simultaneous input, candidate, and scratch logical bytes.
    #[must_use]
    pub const fn peak_live_logical_bytes_bound(self) -> u64 {
        self.peak_live_logical_bytes_bound
    }

    /// Safe upper bound for temporary coordinate entries owned by the builder.
    #[must_use]
    pub const fn scratch_entries_bound(self) -> usize {
        self.scratch_entries_bound
    }

    /// Safe charged-step bound for construction.
    #[must_use]
    pub const fn scalar_steps_bound(self) -> u64 {
        self.scalar_steps_bound
    }

    /// Whether construction must merge or sort generated coordinates.
    #[must_use]
    pub const fn canonicalization_required(self) -> bool {
        self.canonicalization_required
    }
}

/// Explicit resource ceiling for one CSR build.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CsrBuildLimit {
    storage: StorageLimit,
    coefficient_bits: u64,
    scalar_steps: WorkLimit,
}

impl CsrBuildLimit {
    /// Construct the exact conservative limit reported by an estimate.
    #[must_use]
    pub const fn for_estimate(estimate: CsrEstimate) -> Self {
        Self {
            storage: StorageLimit {
                retained_logical_bytes: estimate.retained_logical_bytes_bound,
                peak_live_logical_bytes: estimate.peak_live_logical_bytes_bound,
            },
            coefficient_bits: estimate.coefficient_bits_bound,
            scalar_steps: WorkLimit::new(estimate.scalar_steps_bound),
        }
    }

    #[must_use]
    pub const fn storage(self) -> StorageLimit {
        self.storage
    }

    #[must_use]
    pub const fn coefficient_bits(self) -> u64 {
        self.coefficient_bits
    }

    #[must_use]
    pub const fn scalar_steps(self) -> WorkLimit {
        self.scalar_steps
    }

    #[must_use]
    pub const fn with_storage(mut self, limit: StorageLimit) -> Self {
        self.storage = limit;
        self
    }

    #[must_use]
    pub const fn with_coefficient_bits(mut self, limit: u64) -> Self {
        self.coefficient_bits = limit;
        self
    }

    #[must_use]
    pub const fn with_scalar_steps(mut self, limit: WorkLimit) -> Self {
        self.scalar_steps = limit;
        self
    }

    fn rejection(self, estimate: CsrEstimate) -> Option<RepresentationError> {
        let storage = self.storage;
        if estimate.retained_logical_bytes_bound > storage.retained_logical_bytes() {
            Some(RepresentationError::RetainedLogicalBytes {
                required: estimate.retained_logical_bytes_bound,
                limit: storage.retained_logical_bytes(),
            })
        } else if estimate.peak_live_logical_bytes_bound > storage.peak_live_logical_bytes() {
            Some(RepresentationError::PeakLiveLogicalBytes {
                required: estimate.peak_live_logical_bytes_bound,
                limit: storage.peak_live_logical_bytes(),
            })
        } else if estimate.coefficient_bits_bound > self.coefficient_bits {
            Some(RepresentationError::CoefficientBits {
                required: estimate.coefficient_bits_bound,
                limit: self.coefficient_bits,
            })
        } else if estimate.scalar_steps_bound > self.scalar_steps.steps() {
            Some(RepresentationError::ScalarSteps {
                required: estimate.scalar_steps_bound,
                limit: self.scalar_steps.steps(),
            })
        } else {
            None
        }
    }
}

/// Failure to estimate or build one explicit representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepresentationError {
    /// The sealed materializer does not admit this recipe/encoding pair.
    Unavailable,
    /// Checked index, work, or byte arithmetic overflowed.
    Overflow,
    RetainedLogicalBytes {
        required: u64,
        limit: u64,
    },
    PeakLiveLogicalBytes {
        required: u64,
        limit: u64,
    },
    CoefficientBits {
        required: u64,
        limit: u64,
    },
    ScalarSteps {
        required: u64,
        limit: u64,
    },
    BuildScalarSteps {
        required: u64,
        limit: u64,
    },
    /// Fallible outer-buffer reservation failed.
    Allocation,
    /// Retained topology became unavailable.
    Topology(TopologyError),
}

impl RepresentationError {
    /// Stable machine-readable reason.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::Unavailable => "representation_unavailable",
            Self::Overflow => "representation_overflow",
            Self::RetainedLogicalBytes { .. }
            | Self::PeakLiveLogicalBytes { .. }
            | Self::CoefficientBits { .. }
            | Self::ScalarSteps { .. }
            | Self::BuildScalarSteps { .. } => "resource_limit",
            Self::Allocation => "allocation",
            Self::Topology(error) => error.reason(),
        }
    }

    /// Rejected domain ceiling, when applicable.
    #[must_use]
    pub const fn resource_limit(self) -> Option<(&'static str, u64, u64)> {
        match self {
            Self::RetainedLogicalBytes { required, limit } => {
                Some(("retained_logical_bytes", required, limit))
            }
            Self::PeakLiveLogicalBytes { required, limit } => {
                Some(("peak_live_logical_bytes", required, limit))
            }
            Self::CoefficientBits { required, limit } => {
                Some(("coefficient_bits", required, limit))
            }
            Self::ScalarSteps { required, limit } | Self::BuildScalarSteps { required, limit } => {
                Some(("scalar_steps", required, limit))
            }
            _ => None,
        }
    }

    #[must_use]
    pub const fn resource_phase(self) -> Option<&'static str> {
        match self {
            Self::RetainedLogicalBytes { .. }
            | Self::PeakLiveLogicalBytes { .. }
            | Self::CoefficientBits { .. }
            | Self::ScalarSteps { .. } => Some("estimate"),
            Self::BuildScalarSteps { .. } => Some("build"),
            _ => None,
        }
    }
}

impl std::fmt::Display for RepresentationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.reason())
    }
}

impl std::error::Error for RepresentationError {}

/// Caller-owned immutable CSR representation of one named map.
#[derive(Debug)]
pub struct CsrRepresentation<A, S, T, E>
where
    A: Ring,
    S: Variance,
    T: Variance,
    E: ValueEncoding<A>,
{
    represented_map: LinearMap<A, S, T>,
    physical: Arc<EncodedCsr<A, E>>,
}

impl<A, S, T, E> Clone for CsrRepresentation<A, S, T, E>
where
    A: Ring,
    S: Variance,
    T: Variance,
    E: ValueEncoding<A>,
{
    fn clone(&self) -> Self {
        Self {
            represented_map: self.represented_map.clone(),
            physical: Arc::clone(&self.physical),
        }
    }
}

#[expect(
    private_bounds,
    reason = "public factories are restricted to the crate's admitted encodings"
)]
impl<A, S, T, E> CsrRepresentation<A, S, T, E>
where
    A: Ring,
    S: Variance,
    T: Variance,
    E: CsrEncoding<A>,
{
    /// Estimate construction without publishing or retaining a product.
    ///
    /// # Errors
    ///
    /// Returns a checked-overflow, unsupported-recipe, or topology failure.
    pub fn estimate(
        map: &LinearMap<A, S, T>,
        _encoding: E,
    ) -> Result<CsrEstimate, RepresentationError> {
        estimate_csr::<A, S, T, E>(map)
    }

    /// Build one immutable caller-owned CSR product under an explicit limit.
    ///
    /// # Errors
    ///
    /// Rejects an excessive estimate before allocating product buffers and
    /// publishes no partial representation on any later failure.
    pub fn build(
        map: &LinearMap<A, S, T>,
        _encoding: E,
        limit: CsrBuildLimit,
    ) -> Result<Self, RepresentationError> {
        let estimate = estimate_csr::<A, S, T, E>(map)?;
        if let Some(error) = limit.rejection(estimate) {
            return Err(error);
        }
        let mut meter = BuildMeter::new(limit.scalar_steps);
        let physical = build_map_csr::<A, S, T, E>(map, &mut meter)?;
        if physical.pattern.nnz() > estimate.nnz_bound {
            return Err(RepresentationError::Overflow);
        }
        Ok(Self {
            represented_map: map.clone(),
            physical: Arc::new(physical),
        })
    }
}

impl<A, S, T, E> CsrRepresentation<A, S, T, E>
where
    A: Ring,
    S: Variance,
    T: Variance,
    E: ValueEncoding<A>,
{
    /// Exact mathematical map represented by this product.
    #[must_use]
    pub const fn represented_map(&self) -> &LinearMap<A, S, T> {
        &self.represented_map
    }

    /// Borrow raw physical CSR storage, which carries no independent map identity.
    #[must_use]
    pub fn matrix(&self) -> &CsrMatrix<Box<[E::Stored]>> {
        self.physical.as_ref()
    }

    /// Matrix shape `(target rank, source rank)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.physical.pattern.shape()
    }

    /// Canonical row offsets.
    #[must_use]
    pub fn row_offsets(&self) -> &[usize] {
        self.physical.pattern.row_offsets()
    }

    /// Canonical column indices.
    #[must_use]
    pub fn column_indices(&self) -> &[usize] {
        self.physical.pattern.column_indices()
    }

    /// Arbitrary-precision coefficients aligned with the canonical pattern.
    #[must_use]
    pub fn coefficients(&self) -> &[E::Stored] {
        self.physical.values()
    }

    /// Apply the retained CSR without rebuilding it or consulting topology.
    ///
    /// # Errors
    ///
    /// Returns [`ChainError::SpaceMismatch`] for a value outside the represented
    /// source space.
    pub fn apply(&self, value: &Element<A, S, E>) -> Result<Element<A, T, E>, ChainError> {
        if !self.represented_map.source.same_based_module(&value.space) {
            return Err(ChainError::SpaceMismatch);
        }
        Ok(apply_csr(
            self.physical.as_ref(),
            value,
            &self.represented_map.target,
        ))
    }
}

#[derive(Debug)]
struct BuildMeter {
    limit: u64,
    used: u64,
}

impl BuildMeter {
    const fn new(limit: WorkLimit) -> Self {
        Self {
            limit: limit.steps(),
            used: 0,
        }
    }

    fn charge(&mut self, count: u64) -> Result<(), RepresentationError> {
        let required = self
            .used
            .checked_add(count)
            .ok_or(RepresentationError::Overflow)?;
        if required > self.limit {
            return Err(RepresentationError::BuildScalarSteps {
                required,
                limit: self.limit,
            });
        }
        self.used = required;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct MatrixBound {
    rows: usize,
    columns: usize,
    nnz: usize,
    coefficient_bits: u64,
    work: u64,
    retained_bytes: u64,
    build_peak_bytes: u64,
    scratch_entries: usize,
    structure: MatrixStructure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MatrixStructure {
    General,
    SignedPermutation,
    Zero,
}

fn estimate_csr<A, S, T, E>(map: &LinearMap<A, S, T>) -> Result<CsrEstimate, RepresentationError>
where
    A: Ring,
    S: Variance,
    T: Variance,
    E: CsrEncoding<A>,
{
    let bound = match &map.recipe {
        MapRecipe::Atomic(recipe) => estimate_atomic::<A, E>(
            &BasedDegree::new(&map.source),
            &BasedDegree::new(&map.target),
            recipe,
        )?,
        MapRecipe::Composite { plan, dual } => estimate_composite::<A, E>(plan, *dual)?,
    };
    Ok(CsrEstimate {
        shape: (bound.rows, bound.columns),
        nnz_bound: bound.nnz,
        coefficient_bits_bound: bound.coefficient_bits,
        retained_logical_bytes_bound: bound.retained_bytes,
        peak_live_logical_bytes_bound: input_logical_bytes(map)?
            .checked_add(bound.build_peak_bytes)
            .ok_or(RepresentationError::Overflow)?,
        scratch_entries_bound: bound.scratch_entries,
        scalar_steps_bound: bound.work,
        canonicalization_required: match &map.recipe {
            MapRecipe::Atomic(AtomicRecipe::Coboundary { .. }) | MapRecipe::Composite { .. } => {
                true
            }
            MapRecipe::Atomic(_) => false,
        },
    })
}

fn estimate_composite<A: Ring, E: CsrEncoding<A>>(
    plan: &CompositionPlan,
    dual: bool,
) -> Result<MatrixBound, RepresentationError> {
    if dual {
        estimate_composite_order::<A, E, _>(
            (0..plan.len())
                .rev()
                .map(|index| estimate_step::<A, E>(plan.step(index, true))),
        )
    } else {
        estimate_composite_order::<A, E, _>(
            (0..plan.len()).map(|index| estimate_step::<A, E>(plan.step(index, false))),
        )
    }
}

fn estimate_composite_order<A, E, I>(mut bounds: I) -> Result<MatrixBound, RepresentationError>
where
    A: Ring,
    E: CsrEncoding<A>,
    I: Iterator<Item = Result<MatrixBound, RepresentationError>>,
{
    let mut current = bounds.next().ok_or(RepresentationError::Unavailable)??;
    for next in bounds {
        let next = next?;
        if current.rows != next.columns {
            return Err(RepresentationError::Unavailable);
        }
        let (nnz, multiply_pairs, coefficient_bits, structure): (
            usize,
            usize,
            u64,
            MatrixStructure,
        ) = match (current.structure, next.structure) {
            (MatrixStructure::Zero, _) | (_, MatrixStructure::Zero) => {
                (0, 0, 0, MatrixStructure::Zero)
            }
            (MatrixStructure::SignedPermutation, MatrixStructure::SignedPermutation) => (
                current.nnz.min(next.nnz),
                current.nnz.min(next.nnz),
                1,
                MatrixStructure::SignedPermutation,
            ),
            (MatrixStructure::SignedPermutation, MatrixStructure::General) => (
                next.nnz,
                next.nnz,
                next.coefficient_bits,
                MatrixStructure::General,
            ),
            (MatrixStructure::General, MatrixStructure::SignedPermutation) => (
                current.nnz,
                current.nnz,
                current.coefficient_bits,
                MatrixStructure::General,
            ),
            (MatrixStructure::General, MatrixStructure::General) => {
                let dense_nnz = checked_product(next.rows, current.columns)?;
                let pair_bound = checked_product(next.nnz, current.nnz)?;
                let nnz = dense_nnz
                    .min(pair_bound)
                    .min(checked_product(next.rows, current.nnz)?)
                    .min(checked_product(current.columns, next.nnz)?);
                let coefficient_bits = current
                    .coefficient_bits
                    .checked_add(next.coefficient_bits)
                    .and_then(|bits| {
                        u64::try_from(ceil_log2(current.rows.max(1)))
                            .ok()
                            .and_then(|growth| bits.checked_add(growth))
                    })
                    .ok_or(RepresentationError::Overflow)?;
                (nnz, pair_bound, coefficient_bits, MatrixStructure::General)
            }
        };
        let multiply_work = u64::try_from(multiply_pairs)
            .ok()
            .and_then(|pairs| pairs.checked_mul(4))
            .ok_or(RepresentationError::Overflow)?;
        let retained_bytes = matrix_bytes::<A, E>(next.rows, nnz, coefficient_bits)?;
        let scratch_entries = current
            .scratch_entries
            .max(next.scratch_entries)
            .max(multiply_pairs);
        let multiply_scratch = scratch_bytes::<A, E>(scratch_entries, coefficient_bits)?;
        let build_peak_bytes = current
            .build_peak_bytes
            .max(checked_sum(&[
                current.retained_bytes,
                next.build_peak_bytes,
            ])?)
            .max(checked_sum(&[
                current.retained_bytes,
                next.retained_bytes,
                retained_bytes,
                multiply_scratch,
            ])?);
        current = MatrixBound {
            rows: next.rows,
            columns: current.columns,
            nnz,
            coefficient_bits,
            work: current
                .work
                .checked_add(next.work)
                .and_then(|value| value.checked_add(multiply_work))
                .ok_or(RepresentationError::Overflow)?,
            retained_bytes,
            build_peak_bytes,
            scratch_entries,
            structure,
        };
    }
    Ok(current)
}

fn estimate_step<A: Ring, E: CsrEncoding<A>>(
    (source, target, recipe): (&BasedDegree, &BasedDegree, AtomicRecipe),
) -> Result<MatrixBound, RepresentationError> {
    estimate_atomic::<A, E>(source, target, &recipe)
}

fn estimate_atomic<A, E>(
    source: &BasedDegree,
    target: &BasedDegree,
    recipe: &AtomicRecipe,
) -> Result<MatrixBound, RepresentationError>
where
    A: Ring,
    E: CsrEncoding<A>,
{
    let rows = target.basis_size;
    let columns = source.basis_size;
    let (nnz, coefficient_bits, structure) = match recipe {
        AtomicRecipe::Boundary { degree }
        | AtomicRecipe::Coboundary {
            boundary_degree: degree,
        } => {
            let boundary = source
                .domain
                .view()
                .boundary(*degree)
                .map_err(RepresentationError::Topology)?;
            let bits = boundary
                .exact_entries()
                .map(|(_, _, coefficient)| signed_bits(coefficient))
                .max()
                .unwrap_or(0);
            (
                boundary.indices().len(),
                u64::try_from(bits).map_err(|_| RepresentationError::Overflow)?,
                MatrixStructure::General,
            )
        }
        AtomicRecipe::SignedPermutation { permutation, .. } => {
            (permutation.len(), 1, MatrixStructure::SignedPermutation)
        }
        AtomicRecipe::Identity => (
            rows,
            u64::from(rows != 0),
            MatrixStructure::SignedPermutation,
        ),
        AtomicRecipe::Zero => (0, 0, MatrixStructure::Zero),
    };
    let scratch_entries = match recipe {
        AtomicRecipe::Coboundary { .. } => nnz,
        _ => 0,
    };
    let retained_bytes = matrix_bytes::<A, E>(rows, nnz, coefficient_bits)?;
    let work = u64::try_from(nnz).map_err(|_| RepresentationError::Overflow)?;
    let work = if matches!(recipe, AtomicRecipe::Coboundary { .. }) {
        work.checked_add(canonicalization_steps(nnz)?)
            .ok_or(RepresentationError::Overflow)?
    } else {
        work
    };
    Ok(MatrixBound {
        rows,
        columns,
        nnz,
        coefficient_bits,
        work,
        retained_bytes,
        build_peak_bytes: retained_bytes
            .checked_add(scratch_bytes::<A, E>(scratch_entries, coefficient_bits)?)
            .ok_or(RepresentationError::Overflow)?,
        scratch_entries,
        structure,
    })
}

fn signed_bits(value: i64) -> usize {
    usize::try_from(i64::BITS - value.unsigned_abs().leading_zeros()).expect("bit width fits usize")
}

fn ceil_log2(value: usize) -> usize {
    usize::try_from(usize::BITS - value.saturating_sub(1).leading_zeros())
        .expect("bit width fits usize")
}

fn checked_product(left: usize, right: usize) -> Result<usize, RepresentationError> {
    left.checked_mul(right).ok_or(RepresentationError::Overflow)
}

fn canonicalization_steps(entries: usize) -> Result<u64, RepresentationError> {
    let logarithm =
        u64::try_from(ceil_log2(entries.max(2))).map_err(|_| RepresentationError::Overflow)?;
    let entries = u64::try_from(entries).map_err(|_| RepresentationError::Overflow)?;
    entries
        .checked_mul(logarithm)
        .and_then(|sort| sort.checked_mul(2))
        .and_then(|sort| sort.checked_add(entries))
        .ok_or(RepresentationError::Overflow)
}

fn matrix_bytes<A: Ring, E: CsrEncoding<A>>(
    rows: usize,
    nnz: usize,
    coefficient_bits: u64,
) -> Result<u64, RepresentationError> {
    let offsets = u64::try_from(rows)
        .ok()
        .and_then(|rows| rows.checked_add(1))
        .and_then(|count| count.checked_mul(size_of::<usize>() as u64))
        .ok_or(RepresentationError::Overflow)?;
    let entry_bytes = (size_of::<usize>() as u64)
        .checked_add(E::logical_value_bytes(coefficient_bits)?)
        .ok_or(RepresentationError::Overflow)?;
    offsets
        .checked_add(
            u64::try_from(nnz)
                .ok()
                .and_then(|nnz| nnz.checked_mul(entry_bytes))
                .ok_or(RepresentationError::Overflow)?,
        )
        .ok_or(RepresentationError::Overflow)
}

fn scratch_bytes<A: Ring, E: CsrEncoding<A>>(
    entries: usize,
    coefficient_bits: u64,
) -> Result<u64, RepresentationError> {
    let entry_bytes = (2 * size_of::<usize>() as u64)
        .checked_add(E::logical_value_bytes(coefficient_bits)?)
        .ok_or(RepresentationError::Overflow)?;
    u64::try_from(entries)
        .ok()
        .and_then(|entries| entries.checked_mul(entry_bytes))
        .ok_or(RepresentationError::Overflow)
}

fn boundary_logical_bytes(boundary: BoundaryRef<'_>) -> Result<u64, RepresentationError> {
    let index_bytes = u64::try_from(boundary.indptr().len() + boundary.indices().len())
        .ok()
        .and_then(|count| count.checked_mul(size_of::<usize>() as u64))
        .ok_or(RepresentationError::Overflow)?;
    let coefficient_bytes = match boundary.coefficients() {
        CoefficientSlice::I8(values) => {
            u64::try_from(values.len()).map_err(|_| RepresentationError::Overflow)?
        }
        CoefficientSlice::I64(values) => u64::try_from(values.len())
            .ok()
            .and_then(|count| count.checked_mul(size_of::<i64>() as u64))
            .ok_or(RepresentationError::Overflow)?,
    };
    index_bytes
        .checked_add(coefficient_bytes)
        .ok_or(RepresentationError::Overflow)
}

fn signed_permutation_logical_bytes(
    permutation: &SignedPermutation,
) -> Result<u64, RepresentationError> {
    u64::try_from(permutation.len())
        .ok()
        .and_then(|count| count.checked_mul((2 * size_of::<usize>() + size_of::<i8>()) as u64))
        .ok_or(RepresentationError::Overflow)
}

fn checked_sum(values: &[u64]) -> Result<u64, RepresentationError> {
    values.iter().try_fold(0_u64, |sum, value| {
        sum.checked_add(*value).ok_or(RepresentationError::Overflow)
    })
}

fn input_logical_bytes<A, S, T>(map: &LinearMap<A, S, T>) -> Result<u64, RepresentationError>
where
    A: Ring,
    S: Variance,
    T: Variance,
{
    let mut boundaries = Vec::<(ChainDomain, usize)>::new();
    let mut permutations = Vec::<Arc<SignedPermutation>>::new();
    match &map.recipe {
        MapRecipe::Atomic(recipe) => collect_input(
            &BasedDegree::new(&map.source),
            recipe,
            &mut boundaries,
            &mut permutations,
        )?,
        MapRecipe::Composite { plan, dual } => {
            boundaries
                .try_reserve_exact(plan.len())
                .map_err(|_| RepresentationError::Allocation)?;
            permutations
                .try_reserve_exact(plan.len())
                .map_err(|_| RepresentationError::Allocation)?;
            for index in 0..plan.len() {
                let (source, _, recipe) = plan.step(index, *dual);
                collect_input(source, &recipe, &mut boundaries, &mut permutations)?;
            }
        }
    }
    let boundary_bytes = boundaries.iter().try_fold(0_u64, |sum, (domain, degree)| {
        let boundary = domain
            .view()
            .boundary(*degree)
            .map_err(RepresentationError::Topology)?;
        sum.checked_add(boundary_logical_bytes(boundary)?)
            .ok_or(RepresentationError::Overflow)
    })?;
    permutations
        .iter()
        .try_fold(boundary_bytes, |sum, permutation| {
            sum.checked_add(signed_permutation_logical_bytes(permutation)?)
                .ok_or(RepresentationError::Overflow)
        })
}

fn collect_input(
    source: &BasedDegree,
    recipe: &AtomicRecipe,
    boundaries: &mut Vec<(ChainDomain, usize)>,
    permutations: &mut Vec<Arc<SignedPermutation>>,
) -> Result<(), RepresentationError> {
    match recipe {
        AtomicRecipe::Boundary { degree }
        | AtomicRecipe::Coboundary {
            boundary_degree: degree,
        } => {
            if !boundaries
                .iter()
                .any(|(domain, seen)| *seen == *degree && domain.same_owner(&source.domain))
            {
                boundaries
                    .try_reserve(1)
                    .map_err(|_| RepresentationError::Allocation)?;
                boundaries.push((source.domain.clone(), *degree));
            }
        }
        AtomicRecipe::SignedPermutation { permutation, .. } => {
            if !permutations
                .iter()
                .any(|seen| Arc::ptr_eq(seen, permutation))
            {
                permutations
                    .try_reserve(1)
                    .map_err(|_| RepresentationError::Allocation)?;
                permutations.push(Arc::clone(permutation));
            }
        }
        AtomicRecipe::Identity | AtomicRecipe::Zero => {}
    }
    Ok(())
}

fn build_map_csr<A, S, T, E>(
    map: &LinearMap<A, S, T>,
    meter: &mut BuildMeter,
) -> Result<EncodedCsr<A, E>, RepresentationError>
where
    A: Ring,
    S: Variance,
    T: Variance,
    E: CsrEncoding<A>,
{
    match &map.recipe {
        MapRecipe::Atomic(recipe) => build_atomic::<A, E>(
            &BasedDegree::new(&map.source),
            &BasedDegree::new(&map.target),
            recipe,
            map.source.complex.coefficient_system(),
            meter,
        ),
        MapRecipe::Composite { plan, dual } => {
            build_composite::<A, E>(plan, *dual, map.source.complex.coefficient_system(), meter)
        }
    }
}

fn build_composite<A: Ring, E: CsrEncoding<A>>(
    plan: &CompositionPlan,
    dual: bool,
    algebra: &A,
    meter: &mut BuildMeter,
) -> Result<EncodedCsr<A, E>, RepresentationError> {
    if dual {
        build_composite_order::<A, E, _>(plan, true, (0..plan.len()).rev(), algebra, meter)
    } else {
        build_composite_order::<A, E, _>(plan, false, 0..plan.len(), algebra, meter)
    }
}

fn build_composite_order<A, E, I>(
    plan: &CompositionPlan,
    dual: bool,
    mut indices: I,
    algebra: &A,
    meter: &mut BuildMeter,
) -> Result<EncodedCsr<A, E>, RepresentationError>
where
    A: Ring,
    E: CsrEncoding<A>,
    I: Iterator<Item = usize>,
{
    let first = indices.next().ok_or(RepresentationError::Unavailable)?;
    let mut current = build_step::<A, E>(plan.step(first, dual), algebra, meter)?;
    for index in indices {
        let next = build_step::<A, E>(plan.step(index, dual), algebra, meter)?;
        current = multiply_csr::<A, E>(&next, &current, algebra, meter)?;
    }
    Ok(current)
}

fn build_step<A: Ring, E: CsrEncoding<A>>(
    (source, target, recipe): (&BasedDegree, &BasedDegree, AtomicRecipe),
    algebra: &A,
    meter: &mut BuildMeter,
) -> Result<EncodedCsr<A, E>, RepresentationError> {
    build_atomic::<A, E>(source, target, &recipe, algebra, meter)
}

fn build_atomic<A, E>(
    source: &BasedDegree,
    target: &BasedDegree,
    recipe: &AtomicRecipe,
    algebra: &A,
    meter: &mut BuildMeter,
) -> Result<EncodedCsr<A, E>, RepresentationError>
where
    A: Ring,
    E: CsrEncoding<A>,
{
    match recipe {
        AtomicRecipe::Boundary { degree } => {
            let boundary = source
                .domain
                .view()
                .boundary(*degree)
                .map_err(RepresentationError::Topology)?;
            build_boundary_csr::<A, E>(boundary, algebra, meter)
        }
        AtomicRecipe::Coboundary { boundary_degree } => {
            let boundary = source
                .domain
                .view()
                .boundary(*boundary_degree)
                .map_err(RepresentationError::Topology)?;
            let mut entries = Vec::new();
            entries
                .try_reserve_exact(boundary.indices().len())
                .map_err(|_| RepresentationError::Allocation)?;
            for (row, column, coefficient) in boundary.exact_entries() {
                meter.charge(1)?;
                entries.push((column, row, algebra.lift_i64(coefficient)));
            }
            canonical_csr::<A, E>(
                target.basis_size,
                source.basis_size,
                entries,
                algebra,
                meter,
                false,
            )
        }
        AtomicRecipe::SignedPermutation {
            permutation,
            inverse,
        } => build_signed_csr::<A, E>(
            permutation,
            *inverse,
            target.basis_size,
            source.basis_size,
            algebra,
            meter,
        ),
        AtomicRecipe::Identity => {
            let rank = source.basis_size;
            let mut columns = Vec::new();
            let mut values = Vec::new();
            columns
                .try_reserve_exact(rank)
                .map_err(|_| RepresentationError::Allocation)?;
            values
                .try_reserve_exact(rank)
                .map_err(|_| RepresentationError::Allocation)?;
            for index in 0..rank {
                meter.charge(1)?;
                columns.push(index);
                values.push(E::encode(algebra.lift_i64(1)));
            }
            direct_csr(rank, rank, columns, values)
        }
        AtomicRecipe::Zero => {
            direct_csr(target.basis_size, source.basis_size, Vec::new(), Vec::new())
        }
    }
}

fn build_boundary_csr<A: Ring, E: CsrEncoding<A>>(
    boundary: BoundaryRef<'_>,
    algebra: &A,
    meter: &mut BuildMeter,
) -> Result<EncodedCsr<A, E>, RepresentationError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(boundary.indices().len())
        .map_err(|_| RepresentationError::Allocation)?;
    match boundary.coefficients() {
        CoefficientSlice::I8(coefficients) => {
            for &coefficient in coefficients {
                meter.charge(1)?;
                values.push(E::encode(algebra.lift_i64(i64::from(coefficient))));
            }
        }
        CoefficientSlice::I64(coefficients) => {
            for &coefficient in coefficients {
                meter.charge(1)?;
                values.push(E::encode(algebra.lift_i64(coefficient)));
            }
        }
    }
    physical_csr(
        boundary.shape(),
        boundary.indptr().to_vec(),
        boundary.indices().to_vec(),
        values,
    )
}

fn build_signed_csr<A: Ring, E: CsrEncoding<A>>(
    permutation: &SignedPermutation,
    inverse: bool,
    rows: usize,
    columns: usize,
    algebra: &A,
    meter: &mut BuildMeter,
) -> Result<EncodedCsr<A, E>, RepresentationError> {
    let rank = permutation.len();
    let mut output_columns = Vec::new();
    let mut values = Vec::new();
    output_columns
        .try_reserve_exact(rank)
        .map_err(|_| RepresentationError::Allocation)?;
    values
        .try_reserve_exact(rank)
        .map_err(|_| RepresentationError::Allocation)?;
    for row in 0..rank {
        meter.charge(1)?;
        let (column, sign) = if inverse {
            permutation.map_basis(row)
        } else {
            permutation.inverse_basis(row)
        }
        .map_err(RepresentationError::Topology)?;
        output_columns.push(column);
        values.push(E::encode(algebra.lift_i64(i64::from(sign))));
    }
    direct_csr(rows, columns, output_columns, values)
}

fn direct_csr<V>(
    rows: usize,
    columns: usize,
    column_indices: Vec<usize>,
    values: Vec<V>,
) -> Result<CsrMatrix<Box<[V]>>, RepresentationError> {
    if column_indices.len() != values.len()
        || (!column_indices.is_empty() && column_indices.len() != rows)
    {
        return Err(RepresentationError::Unavailable);
    }
    let mut row_offsets = Vec::new();
    row_offsets
        .try_reserve_exact(rows.checked_add(1).ok_or(RepresentationError::Overflow)?)
        .map_err(|_| RepresentationError::Allocation)?;
    if column_indices.is_empty() {
        row_offsets.resize(rows + 1, 0);
    } else {
        row_offsets.extend(0..=rows);
    }
    physical_csr((rows, columns), row_offsets, column_indices, values)
}

fn canonical_csr<A: Ring, E: CsrEncoding<A>>(
    rows: usize,
    columns: usize,
    mut entries: Vec<(usize, usize, A::Element)>,
    algebra: &A,
    meter: &mut BuildMeter,
    already_sorted: bool,
) -> Result<EncodedCsr<A, E>, RepresentationError> {
    if !already_sorted {
        meter.charge(
            canonicalization_steps(entries.len())?
                .checked_sub(
                    u64::try_from(entries.len()).map_err(|_| RepresentationError::Overflow)?,
                )
                .ok_or(RepresentationError::Overflow)?,
        )?;
        entries.sort_unstable_by_key(|(row, column, _)| (*row, *column));
    }
    let mut row_offsets = Vec::new();
    let offset_count = rows.checked_add(1).ok_or(RepresentationError::Overflow)?;
    row_offsets
        .try_reserve_exact(offset_count)
        .map_err(|_| RepresentationError::Allocation)?;
    let mut column_indices = Vec::new();
    let mut values = Vec::new();
    column_indices
        .try_reserve_exact(entries.len())
        .map_err(|_| RepresentationError::Allocation)?;
    values
        .try_reserve_exact(entries.len())
        .map_err(|_| RepresentationError::Allocation)?;
    row_offsets.push(0);
    let mut entries = entries.into_iter().peekable();
    for row in 0..rows {
        while let Some((entry_row, column, _)) = entries.peek() {
            if *entry_row != row {
                break;
            }
            let column = *column;
            if column >= columns {
                return Err(RepresentationError::Unavailable);
            }
            let mut sum = algebra.zero();
            while let Some((entry_row, entry_column, _)) = entries.peek() {
                if *entry_row != row || *entry_column != column {
                    break;
                }
                let (_, _, value) = entries.next().expect("peeked entry exists");
                meter.charge(1)?;
                algebra.add_assign(&mut sum, &value);
            }
            if !algebra.is_zero(&sum) {
                column_indices.push(column);
                values.push(E::encode(sum));
            }
        }
        row_offsets.push(column_indices.len());
    }
    if entries.next().is_some() {
        return Err(RepresentationError::Unavailable);
    }
    physical_csr((rows, columns), row_offsets, column_indices, values)
}

fn physical_csr<V>(
    shape: (usize, usize),
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<V>,
) -> Result<CsrMatrix<Box<[V]>>, RepresentationError> {
    CsrMatrix::try_from_parts(shape, row_offsets, column_indices, values)
        .map_err(|_| RepresentationError::Unavailable)
}

fn multiply_csr<A: Ring, E: CsrEncoding<A>>(
    after: &EncodedCsr<A, E>,
    before: &EncodedCsr<A, E>,
    algebra: &A,
    meter: &mut BuildMeter,
) -> Result<EncodedCsr<A, E>, RepresentationError> {
    if before.pattern.shape.0 != after.pattern.shape.1 {
        return Err(RepresentationError::Unavailable);
    }
    let rows = after.pattern.shape.0;
    let columns = before.pattern.shape.1;
    let mut entries = Vec::new();
    for row in 0..rows {
        let mut accumulated = BTreeMap::<usize, A::Element>::new();
        for after_position in after.pattern.row_offsets[row]..after.pattern.row_offsets[row + 1] {
            let middle = after.pattern.column_indices[after_position];
            let after_value = E::element(&after.values[after_position]);
            for before_position in
                before.pattern.row_offsets[middle]..before.pattern.row_offsets[middle + 1]
            {
                meter.charge(3)?;
                let column = before.pattern.column_indices[before_position];
                let term =
                    algebra.multiply(after_value, E::element(&before.values[before_position]));
                algebra.add_assign(
                    accumulated.entry(column).or_insert_with(|| algebra.zero()),
                    &term,
                );
            }
        }
        entries
            .try_reserve(accumulated.len())
            .map_err(|_| RepresentationError::Allocation)?;
        entries.extend(
            accumulated
                .into_iter()
                .filter(|(_, coefficient)| !algebra.is_zero(coefficient))
                .map(|(column, coefficient)| (row, column, coefficient)),
        );
    }
    canonical_csr::<A, E>(rows, columns, entries, algebra, meter, true)
}

fn apply_csr<A, S, T, E>(
    matrix: &EncodedCsr<A, E>,
    value: &Element<A, S, E>,
    target: &Space<A, T>,
) -> Element<A, T, E>
where
    A: Ring,
    S: Variance,
    T: Variance,
    E: ValueEncoding<A>,
{
    let algebra = target.complex.coefficient_system();
    let mut output_indices = Vec::new();
    let mut output_coefficients = Vec::new();
    for row in 0..matrix.pattern.shape.0 {
        let mut sum = algebra.zero();
        for position in matrix.pattern.row_offsets[row]..matrix.pattern.row_offsets[row + 1] {
            let column = matrix.pattern.column_indices[position];
            if let Ok(input_position) = value.indices().binary_search(&column) {
                let product = algebra.multiply(
                    E::element(&matrix.values[position]),
                    E::element(&value.coefficients()[input_position]),
                );
                algebra.add_assign(&mut sum, &product);
            }
        }
        if !algebra.is_zero(&sum) {
            output_indices.push(row);
            output_coefficients.push(E::encode(sum));
        }
    }
    Element::<A, S, E>::from_canonical(target, output_indices, output_coefficients)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimation_rejects_checked_arithmetic_overflow() {
        assert_eq!(
            checked_product(usize::MAX, 2),
            Err(RepresentationError::Overflow)
        );
        assert_eq!(
            matrix_bytes::<IntegerRing, BigIntEncoding>(usize::MAX, 1, 1),
            Err(RepresentationError::Overflow)
        );
    }

    #[test]
    fn build_meter_classifies_dynamic_exhaustion() {
        let mut meter = BuildMeter::new(WorkLimit::new(0));
        let error = meter.charge(1).unwrap_err();
        assert_eq!(error.resource_limit(), Some(("scalar_steps", 1, 0)));
        assert_eq!(error.resource_phase(), Some("build"));
    }

    #[test]
    fn compressed_rows_reject_every_structural_law_before_publication() {
        let admit = |shape, offsets: &[usize], columns: &[usize], values: &[i8]| {
            CsrMatrix::try_from_parts(shape, offsets, columns, values)
        };

        assert!(admit((2, 3), &[0, 1], &[0], &[1]).is_err());
        assert!(admit((2, 3), &[1, 1, 1], &[], &[]).is_err());
        assert!(admit((2, 3), &[0, 2, 1], &[0], &[1]).is_err());
        assert!(admit((2, 3), &[0, 0, 2], &[0], &[1]).is_err());
        assert!(admit((1, 3), &[0, 2], &[0, 1], &[1]).is_err());
        assert!(admit((1, 3), &[0, 2], &[1, 1], &[1, 1]).is_err());
        assert!(admit((1, 3), &[0, 2], &[2, 1], &[1, 1]).is_err());
        assert!(admit((1, 3), &[0, 1], &[3], &[1]).is_err());

        assert!(admit((0, 0), &[0], &[], &[]).is_ok());
        let valid = admit((2, 3), &[0, 2, 3], &[0, 2, 1], &[1, -1, 1])
            .unwrap_or_else(|_| panic!("valid compressed rows rejected"));
        assert_eq!(valid.pattern().row_offsets(), &[0, 2, 3]);
    }
}
