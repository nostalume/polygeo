use std::fmt::{Display, Formatter};
use std::mem::size_of;

use num_bigint::{BigInt, Sign};
use num_traits::{One, Zero};

use crate::form::exact_integer_binary64_dot;
use crate::{
    Binary64Chain, Binary64Cochain, Binary64Element, Binary64ElementError, Binary64Space, Chain,
    ChainError, ChainIsomorphism, CoefficientSlice, IntegerRing, IntegralChain,
    IntegralChainComplex, LinearMap, StorageLimit, WorkLimit,
};

/// Explicit resource ceiling for one integral-homology analysis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HomologyLimit {
    storage: StorageLimit,
    coefficient_bits: u64,
    smith_steps: WorkLimit,
}

impl HomologyLimit {
    /// Named default ceiling; callers may lower any individual axis.
    pub const DEFAULT: Self = Self {
        storage: StorageLimit {
            retained_logical_bytes: 128 * 1024 * 1024,
            peak_live_logical_bytes: 512 * 1024 * 1024,
        },
        coefficient_bits: 65_536,
        smith_steps: WorkLimit::new(100_000_000),
    };

    #[must_use]
    pub const fn storage(self) -> StorageLimit {
        self.storage
    }

    #[must_use]
    pub const fn coefficient_bits(self) -> u64 {
        self.coefficient_bits
    }

    #[must_use]
    pub const fn smith_steps(self) -> WorkLimit {
        self.smith_steps
    }

    #[must_use]
    pub const fn with_storage(mut self, value: StorageLimit) -> Self {
        self.storage = value;
        self
    }

    #[must_use]
    pub const fn with_coefficient_bits(mut self, value: u64) -> Self {
        self.coefficient_bits = value;
        self
    }

    #[must_use]
    pub const fn with_smith_steps(mut self, value: WorkLimit) -> Self {
        self.smith_steps = value;
        self
    }
}

/// Classified failure of an unpublished integral-homology analysis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HomologyError {
    DegreeOutside,
    RetainedLogicalBytes { required: u64, limit: u64 },
    PeakLiveLogicalBytes { required: u64, limit: u64 },
    CoefficientBits { required: u64, limit: u64 },
    SmithSteps { required: u64, limit: u64 },
    Overflow,
    Allocation,
    InvalidChain,
    OwnerMismatch,
    InternalInvariant,
}

impl HomologyError {
    /// Stable machine-readable reason.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::DegreeOutside => "degree_outside",
            Self::RetainedLogicalBytes { .. }
            | Self::PeakLiveLogicalBytes { .. }
            | Self::CoefficientBits { .. }
            | Self::SmithSteps { .. } => "resource_limit",
            Self::Overflow => "overflow",
            Self::Allocation => "allocation",
            Self::InvalidChain => "invalid_chain",
            Self::OwnerMismatch => "owner_mismatch",
            Self::InternalInvariant => "internal_invariant",
        }
    }

    /// Rejected semantic ceiling, when applicable.
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
            Self::SmithSteps { required, limit } => Some(("smith_steps", required, limit)),
            _ => None,
        }
    }
}

impl Display for HomologyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.reason())
    }
}

impl std::error::Error for HomologyError {}

#[derive(Debug)]
struct HomologyUse {
    limit: HomologyLimit,
    smith_steps: u64,
    input_logical_bytes: u64,
    retained_logical_bytes: u64,
}

impl HomologyUse {
    const fn new(limit: HomologyLimit) -> Self {
        Self {
            limit,
            smith_steps: 0,
            input_logical_bytes: 0,
            retained_logical_bytes: 0,
        }
    }

    fn input(&mut self, bytes: u64, bits: u64) -> Result<(), HomologyError> {
        self.coefficients_bound(bits)?;
        self.input_logical_bytes = bytes;
        self.peak(0)
    }

    fn workspace(&self, bits: u64, bytes: u64) -> Result<(), HomologyError> {
        self.coefficients_bound(bits)?;
        self.peak(bytes)
    }

    fn charge(&mut self, count: usize) -> Result<(), HomologyError> {
        let count = u64::try_from(count).map_err(|_| HomologyError::Overflow)?;
        let next = self
            .smith_steps
            .checked_add(count)
            .ok_or(HomologyError::Overflow)?;
        let limit = self.limit.smith_steps.steps();
        if next > limit {
            return Err(HomologyError::SmithSteps {
                required: next,
                limit,
            });
        }
        self.smith_steps = next;
        Ok(())
    }

    fn coefficients<'a>(
        &self,
        values: impl IntoIterator<Item = &'a BigInt>,
    ) -> Result<(), HomologyError> {
        self.coefficients_bound(values.into_iter().map(BigInt::bits).max().unwrap_or(0))
    }

    fn coefficients_bound(&self, required: u64) -> Result<(), HomologyError> {
        let limit = self.limit.coefficient_bits;
        if required > limit {
            return Err(HomologyError::CoefficientBits { required, limit });
        }
        Ok(())
    }

    fn retain(&mut self, bytes: u64) -> Result<(), HomologyError> {
        let required = self
            .retained_logical_bytes
            .checked_add(bytes)
            .ok_or(HomologyError::Overflow)?;
        let limit = self.limit.storage.retained_logical_bytes();
        if required > limit {
            return Err(HomologyError::RetainedLogicalBytes { required, limit });
        }
        self.retained_logical_bytes = required;
        self.peak(0)
    }

    fn peak(&self, workspace: u64) -> Result<(), HomologyError> {
        let required = self
            .input_logical_bytes
            .checked_add(self.retained_logical_bytes)
            .and_then(|bytes| bytes.checked_add(workspace))
            .ok_or(HomologyError::Overflow)?;
        let limit = self.limit.storage.peak_live_logical_bytes();
        if required > limit {
            return Err(HomologyError::PeakLiveLogicalBytes { required, limit });
        }
        Ok(())
    }
}

fn zero_integers(count: usize) -> Result<Vec<BigInt>, HomologyError> {
    let mut output = reserved_vec(count)?;
    output.resize_with(count, BigInt::zero);
    Ok(output)
}

fn reserved_vec<T>(count: usize) -> Result<Vec<T>, HomologyError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| HomologyError::Allocation)?;
    Ok(values)
}

fn scalar_bytes(bits: u64) -> Result<usize, HomologyError> {
    usize::try_from(bits.div_ceil(u64::from(usize::BITS)))
        .ok()
        .and_then(|digits| digits.checked_mul(size_of::<usize>()))
        .and_then(|magnitude| size_of::<BigInt>().checked_add(magnitude))
        .ok_or(HomologyError::Overflow)
}

fn logical_bytes(bytes: usize) -> Result<u64, HomologyError> {
    u64::try_from(bytes).map_err(|_| HomologyError::Overflow)
}

fn product_bits(values: impl IntoIterator<Item = u64>) -> Result<u64, HomologyError> {
    values.into_iter().try_fold(0_u64, |bits, value| {
        bits.checked_add(value).ok_or(HomologyError::Overflow)
    })
}

fn add_product_bits(target: u64, source: u64, factor: u64) -> Result<u64, HomologyError> {
    target
        .max(product_bits([source, factor])?)
        .checked_add(1)
        .ok_or(HomologyError::Overflow)
}

fn ceil_log2(value: usize) -> u64 {
    u64::from(usize::BITS - value.max(1).saturating_sub(1).leading_zeros())
}

fn boundary_input_bytes(boundary: crate::BoundaryRef<'_>) -> Result<u64, HomologyError> {
    let indices = boundary
        .indptr()
        .len()
        .checked_add(boundary.indices().len())
        .and_then(|count| count.checked_mul(size_of::<usize>()))
        .ok_or(HomologyError::Overflow)?;
    let coefficients = match boundary.coefficients() {
        CoefficientSlice::I8(values) => values.len(),
        CoefficientSlice::I64(values) => values
            .len()
            .checked_mul(size_of::<i64>())
            .ok_or(HomologyError::Overflow)?,
    };
    logical_bytes(
        indices
            .checked_add(coefficients)
            .ok_or(HomologyError::Overflow)?,
    )
}

fn initial_sparse_bytes(
    shapes: &[(usize, usize)],
    nonzeros: usize,
    coefficient_bits: u64,
) -> Result<u64, HomologyError> {
    let matrix_outer = shapes
        .len()
        .checked_mul(size_of::<SparseMatrix>() + size_of::<Vec<bool>>())
        .ok_or(HomologyError::Overflow)?;
    let basis_slots = shapes.iter().try_fold(0_usize, |total, (rows, columns)| {
        total
            .checked_add(*rows)
            .and_then(|value| value.checked_add(*columns))
            .ok_or(HomologyError::Overflow)
    })?;
    let active = shapes
        .iter()
        .try_fold(0_usize, |total, (_, columns)| total.checked_add(*columns))
        .ok_or(HomologyError::Overflow)?;
    let entries = nonzeros
        .checked_mul(
            size_of::<Option<Entry>>() + 2 * size_of::<usize>() + scalar_bytes(coefficient_bits)?,
        )
        .ok_or(HomologyError::Overflow)?;
    let active = active
        .checked_mul(size_of::<bool>())
        .ok_or(HomologyError::Overflow)?;
    logical_bytes(
        matrix_outer
            .checked_add(
                basis_slots
                    .checked_mul(size_of::<Vec<usize>>())
                    .ok_or(HomologyError::Overflow)?,
            )
            .and_then(|bytes| bytes.checked_add(active))
            .and_then(|bytes| bytes.checked_add(entries))
            .ok_or(HomologyError::Overflow)?,
    )
}

fn try_empty_vectors<T>(count: usize) -> Result<Vec<Vec<T>>, HomologyError> {
    let mut output = reserved_vec(count)?;
    output.resize_with(count, Vec::new);
    Ok(output)
}

#[derive(Clone, Debug)]
struct Entry {
    row: usize,
    column: usize,
    value: BigInt,
}

#[derive(Clone, Debug)]
struct SparseMatrix {
    rows: usize,
    columns: usize,
    entries: Vec<Option<Entry>>,
    row_ids: Vec<Vec<usize>>,
    column_ids: Vec<Vec<usize>>,
    live: usize,
}

impl SparseMatrix {
    fn new(rows: usize, columns: usize, capacity: usize) -> Result<Self, HomologyError> {
        let entries = reserved_vec(capacity)?;
        Ok(Self {
            rows,
            columns,
            entries,
            row_ids: try_empty_vectors(rows)?,
            column_ids: try_empty_vectors(columns)?,
            live: 0,
        })
    }

    fn entry_id(&self, row: usize, column: usize) -> Option<usize> {
        self.row_ids.get(row)?.iter().copied().find(|&id| {
            self.entries[id]
                .as_ref()
                .is_some_and(|entry| entry.column == column)
        })
    }

    fn set(&mut self, row: usize, column: usize, value: BigInt) -> Result<(), HomologyError> {
        if row >= self.rows || column >= self.columns {
            return Err(HomologyError::InternalInvariant);
        }
        if let Some(id) = self.entry_id(row, column) {
            if value.is_zero() {
                self.entries[id] = None;
                self.live -= 1;
            } else {
                self.entries[id]
                    .as_mut()
                    .ok_or(HomologyError::InternalInvariant)?
                    .value = value;
            }
            return Ok(());
        }
        if value.is_zero() {
            return Ok(());
        }
        self.entries
            .try_reserve(1)
            .map_err(|_| HomologyError::Allocation)?;
        self.row_ids[row]
            .try_reserve(1)
            .map_err(|_| HomologyError::Allocation)?;
        self.column_ids[column]
            .try_reserve(1)
            .map_err(|_| HomologyError::Allocation)?;
        let id = self.entries.len();
        self.entries.push(Some(Entry { row, column, value }));
        self.row_ids[row].push(id);
        self.column_ids[column].push(id);
        self.live += 1;
        Ok(())
    }

    fn get(&self, row: usize, column: usize) -> BigInt {
        self.entry_id(row, column)
            .and_then(|id| self.entries[id].as_ref())
            .map_or_else(BigInt::zero, |entry| entry.value.clone())
    }

    fn row_values(&self, row: usize) -> Result<Vec<(usize, BigInt)>, HomologyError> {
        let mut output = reserved_vec(self.row_ids[row].len())?;
        output.extend(self.row_ids[row].iter().filter_map(|&id| {
            self.entries[id]
                .as_ref()
                .map(|entry| (entry.column, entry.value.clone()))
        }));
        Ok(output)
    }

    fn column_values(&self, column: usize) -> Result<Vec<(usize, BigInt)>, HomologyError> {
        let mut output = reserved_vec(self.column_ids[column].len())?;
        output.extend(self.column_ids[column].iter().filter_map(|&id| {
            self.entries[id]
                .as_ref()
                .map(|entry| (entry.row, entry.value.clone()))
        }));
        Ok(output)
    }

    fn remove_row(&mut self, row: usize) {
        for &id in &self.row_ids[row] {
            if self.entries[id].take().is_some() {
                self.live -= 1;
            }
        }
    }

    fn remove_column(&mut self, column: usize) {
        for &id in &self.column_ids[column] {
            if self.entries[id].take().is_some() {
                self.live -= 1;
            }
        }
    }

    fn live_entries(&self) -> impl Iterator<Item = &Entry> {
        self.entries.iter().filter_map(Option::as_ref)
    }

    fn row_nnz(&self, row: usize) -> usize {
        self.row_ids[row]
            .iter()
            .filter(|&&id| self.entries[id].is_some())
            .count()
    }

    fn column_nnz(&self, column: usize) -> usize {
        self.column_ids[column]
            .iter()
            .filter(|&&id| self.entries[id].is_some())
            .count()
    }
}

#[derive(Clone, Debug)]
struct Cancellation {
    degree: usize,
    upper_basis: usize,
    pivot: BigInt,
    upper_correction: Box<[(usize, BigInt)]>,
}

#[derive(Debug)]
struct ReducedChain {
    boundaries: Vec<SparseMatrix>,
    active: Vec<Vec<bool>>,
    cancellations: Vec<Cancellation>,
}

impl ReducedChain {
    fn ingest(
        chain: &IntegralChainComplex,
        meter: &mut HomologyUse,
    ) -> Result<Self, HomologyError> {
        let view = chain.chain_view();
        let count = view
            .dimension()
            .checked_add(1)
            .ok_or(HomologyError::Overflow)?;
        let mut shapes = reserved_vec(count)?;
        let mut nonzeros = 0_usize;
        let mut input_bytes = 0_u64;
        let mut input_bits = 0_u64;
        for degree in 0..count {
            let boundary = view
                .boundary(degree)
                .map_err(|_| HomologyError::InternalInvariant)?;
            nonzeros = nonzeros
                .checked_add(boundary.exact_entries().len())
                .ok_or(HomologyError::Overflow)?;
            input_bytes = input_bytes
                .checked_add(boundary_input_bytes(boundary)?)
                .ok_or(HomologyError::Overflow)?;
            input_bits = input_bits.max(
                boundary
                    .exact_entries()
                    .map(|(_, _, value)| {
                        u64::from(i64::BITS - value.unsigned_abs().leading_zeros())
                    })
                    .max()
                    .unwrap_or(0),
            );
            shapes.push(boundary.shape());
        }
        meter.input(input_bytes, input_bits)?;
        meter.workspace(
            input_bits,
            initial_sparse_bytes(&shapes, nonzeros, input_bits)?,
        )?;
        meter.charge(nonzeros)?;

        let mut boundaries = reserved_vec(count)?;
        for (degree, &(rows, columns)) in shapes.iter().enumerate() {
            let boundary = view
                .boundary(degree)
                .map_err(|_| HomologyError::InternalInvariant)?;
            let mut matrix = SparseMatrix::new(rows, columns, boundary.exact_entries().len())?;
            for (row, column, value) in boundary.exact_entries() {
                matrix.set(row, column, BigInt::from(value))?;
            }
            boundaries.push(matrix);
        }
        let mut active = reserved_vec(count)?;
        for &(_, columns) in &shapes {
            let mut degree = reserved_vec(columns)?;
            degree.resize(columns, true);
            active.push(degree);
        }
        let output = Self {
            boundaries,
            active,
            cancellations: Vec::new(),
        };
        output.observe(meter)?;
        Ok(output)
    }

    fn observe(&self, meter: &HomologyUse) -> Result<(), HomologyError> {
        let (bits, bytes) = self.workspace_profile()?;
        meter.workspace(bits, bytes)
    }

    fn workspace_profile(&self) -> Result<(u64, u64), HomologyError> {
        let entry_slots = self
            .boundaries
            .iter()
            .map(|matrix| matrix.entries.len())
            .sum::<usize>();
        let index_slots = self
            .boundaries
            .iter()
            .flat_map(|matrix| matrix.row_ids.iter().chain(&matrix.column_ids))
            .map(Vec::len)
            .sum::<usize>();
        let active = self.active.iter().map(Vec::len).sum::<usize>();
        let bytes = entry_slots
            .checked_mul(size_of::<Option<Entry>>())
            .and_then(|value| value.checked_add(index_slots * size_of::<usize>()))
            .and_then(|value| value.checked_add(active * size_of::<bool>()))
            .and_then(|value| {
                value.checked_add(
                    self.cancellations
                        .iter()
                        .map(|entry| {
                            size_of::<Cancellation>()
                                + entry.upper_correction.len() * size_of::<(usize, BigInt)>()
                        })
                        .sum::<usize>(),
                )
            })
            .ok_or(HomologyError::Overflow)?;
        let bits = self
            .boundaries
            .iter()
            .flat_map(SparseMatrix::live_entries)
            .map(|entry| entry.value.bits())
            .chain(self.cancellations.iter().flat_map(|entry| {
                std::iter::once(entry.pivot.bits())
                    .chain(entry.upper_correction.iter().map(|(_, value)| value.bits()))
            }))
            .max()
            .unwrap_or(0);
        let coefficient_bytes = self
            .boundaries
            .iter()
            .flat_map(SparseMatrix::live_entries)
            .map(|entry| usize::try_from(entry.value.bits().div_ceil(8)))
            .chain(self.cancellations.iter().flat_map(|entry| {
                std::iter::once(usize::try_from(entry.pivot.bits().div_ceil(8))).chain(
                    entry
                        .upper_correction
                        .iter()
                        .map(|(_, value)| usize::try_from(value.bits().div_ceil(8))),
                )
            }))
            .try_fold(0_usize, |total, bytes| {
                total
                    .checked_add(bytes.map_err(|_| HomologyError::Overflow)?)
                    .ok_or(HomologyError::Overflow)
            })?;
        Ok((
            bits,
            logical_bytes(
                bytes
                    .checked_add(coefficient_bytes)
                    .ok_or(HomologyError::Overflow)?,
            )?,
        ))
    }

    fn active_indices(&self, degree: usize) -> Result<Vec<usize>, HomologyError> {
        let active = &self.active[degree];
        let mut output = reserved_vec(active.iter().filter(|&&value| value).count())?;
        output.extend(
            active
                .iter()
                .enumerate()
                .filter_map(|(index, &active)| active.then_some(index)),
        );
        Ok(output)
    }

    fn unit_pivot(&self) -> Option<(usize, usize, usize)> {
        self.boundaries
            .iter()
            .enumerate()
            .skip(1)
            .flat_map(|(degree, matrix)| {
                matrix
                    .live_entries()
                    .filter(|entry| entry.value == BigInt::one() || entry.value == -BigInt::one())
                    .map(move |entry| {
                        let fill = matrix
                            .row_nnz(entry.row)
                            .saturating_sub(1)
                            .saturating_mul(matrix.column_nnz(entry.column).saturating_sub(1));
                        (fill, degree, entry.row, entry.column)
                    })
            })
            .min_by(|left, right| {
                (left.0, left.1, left.2, left.3).cmp(&(right.0, right.1, right.2, right.3))
            })
            .map(|(_, degree, row, column)| (degree, row, column))
    }

    fn contract(&mut self, meter: &mut HomologyUse) -> Result<(), HomologyError> {
        loop {
            meter.charge(self.boundaries.iter().map(|matrix| matrix.live).sum())?;
            let Some(pivot) = self.unit_pivot() else {
                return Ok(());
            };
            self.cancel(pivot, meter)?;
        }
    }

    fn cancel(
        &mut self,
        (degree, row, column): (usize, usize, usize),
        meter: &mut HomologyUse,
    ) -> Result<(), HomologyError> {
        let pivot = self.boundaries[degree].get(row, column).clone();
        let row_values = self.boundaries[degree].row_values(row)?;
        let column_values = self.boundaries[degree].column_values(column)?;
        let fill = row_values
            .len()
            .saturating_sub(1)
            .saturating_mul(column_values.len().saturating_sub(1));
        meter.charge(
            fill.saturating_add(row_values.len())
                .saturating_add(column_values.len()),
        )?;
        let predicted_bits = self
            .boundaries
            .iter()
            .flat_map(SparseMatrix::live_entries)
            .map(|entry| entry.value.bits())
            .max()
            .unwrap_or(0)
            .saturating_mul(2)
            .saturating_add(2);
        meter.workspace(
            predicted_bits,
            logical_bytes(
                self.logical_bytes_with_fill(
                    fill.saturating_add(row_values.len()),
                    predicted_bits,
                )?,
            )?,
        )?;

        let mut upper_correction = Vec::new();
        upper_correction
            .try_reserve_exact(row_values.len().saturating_sub(1))
            .map_err(|_| HomologyError::Allocation)?;
        upper_correction.extend(
            row_values
                .iter()
                .filter(|(other, _)| *other != column)
                .cloned(),
        );
        for (other_row, left) in column_values.iter().filter(|(other, _)| *other != row) {
            for (other_column, right) in row_values.iter().filter(|(other, _)| *other != column) {
                let correction = left * &pivot * right;
                let value = self.boundaries[degree].get(*other_row, *other_column) - correction;
                self.boundaries[degree].set(*other_row, *other_column, value)?;
            }
        }
        self.boundaries[degree].remove_row(row);
        self.boundaries[degree].remove_column(column);
        self.boundaries[degree - 1].remove_column(row);
        if degree + 1 < self.boundaries.len() {
            self.boundaries[degree + 1].remove_row(column);
        }
        self.active[degree - 1][row] = false;
        self.active[degree][column] = false;
        self.cancellations
            .try_reserve(1)
            .map_err(|_| HomologyError::Allocation)?;
        self.cancellations.push(Cancellation {
            degree,
            upper_basis: column,
            pivot,
            upper_correction: upper_correction.into_boxed_slice(),
        });
        self.observe(meter)?;
        Ok(())
    }

    fn logical_bytes_with_fill(
        &self,
        fill: usize,
        coefficient_bits: u64,
    ) -> Result<usize, HomologyError> {
        self.boundaries
            .iter()
            .map(|matrix| matrix.entries.len())
            .sum::<usize>()
            .checked_add(fill)
            .and_then(|slots| {
                slots.checked_mul(
                    size_of::<Option<Entry>>()
                        + 2 * size_of::<usize>()
                        + scalar_bytes(coefficient_bits).ok()?,
                )
            })
            .ok_or(HomologyError::Overflow)
    }

    fn verify_chain(&self, meter: &mut HomologyUse) -> Result<(), HomologyError> {
        for degree in 1..self.boundaries.len() {
            verify_zero_product(
                &self.boundaries[degree - 1],
                &self.boundaries[degree],
                meter,
            )?;
        }
        Ok(())
    }

    fn lift(
        &self,
        degree: usize,
        vector: &mut [BigInt],
        meter: &mut HomologyUse,
    ) -> Result<(), HomologyError> {
        for cancellation in self
            .cancellations
            .iter()
            .rev()
            .filter(|entry| entry.degree == degree)
        {
            meter.charge(cancellation.upper_correction.len())?;
            let mut coefficient = BigInt::zero();
            for (basis, factor) in &cancellation.upper_correction {
                meter.charge(1)?;
                meter.coefficients_bound(add_product_bits(
                    coefficient.bits(),
                    factor.bits(),
                    vector[*basis].bits(),
                )?)?;
                coefficient += factor * &vector[*basis];
            }
            meter.coefficients_bound(product_bits([
                cancellation.pivot.bits(),
                coefficient.bits(),
            ])?)?;
            vector[cancellation.upper_basis] = -&cancellation.pivot * coefficient;
        }
        Ok(())
    }
}

fn verify_zero_product(
    left: &SparseMatrix,
    right: &SparseMatrix,
    meter: &mut HomologyUse,
) -> Result<(), HomologyError> {
    if left.columns != right.rows {
        return Err(HomologyError::InvalidChain);
    }
    let contributions = right.live_entries().try_fold(0_usize, |total, entry| {
        total
            .checked_add(left.column_nnz(entry.row))
            .ok_or(HomologyError::Overflow)
    })?;
    meter.charge(contributions)?;
    meter.workspace(
        product_bits([
            left.live_entries()
                .map(|entry| entry.value.bits())
                .max()
                .unwrap_or(0),
            right
                .live_entries()
                .map(|entry| entry.value.bits())
                .max()
                .unwrap_or(0),
            ceil_log2(contributions),
        ])?,
        logical_bytes(
            contributions
                .checked_mul(size_of::<(usize, usize, BigInt)>())
                .ok_or(HomologyError::Overflow)?,
        )?,
    )?;
    let mut products = reserved_vec(contributions)?;
    for right_entry in right.live_entries() {
        for &id in &left.column_ids[right_entry.row] {
            if let Some(left_entry) = &left.entries[id] {
                products.push((
                    left_entry.row,
                    right_entry.column,
                    &left_entry.value * &right_entry.value,
                ));
            }
        }
    }
    products.sort_unstable_by_key(|entry| (entry.0, entry.1));
    let mut cursor = 0;
    while cursor < products.len() {
        let key = (products[cursor].0, products[cursor].1);
        let mut sum = BigInt::zero();
        while cursor < products.len() && (products[cursor].0, products[cursor].1) == key {
            sum += &products[cursor].2;
            cursor += 1;
        }
        if !sum.is_zero() {
            return Err(HomologyError::InvalidChain);
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct DenseMatrix {
    rows: usize,
    columns: usize,
    values: Vec<BigInt>,
}

impl DenseMatrix {
    fn zeros(rows: usize, columns: usize) -> Result<Self, HomologyError> {
        let cells = rows.checked_mul(columns).ok_or(HomologyError::Overflow)?;
        let mut values = reserved_vec(cells)?;
        values.resize_with(cells, BigInt::zero);
        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    fn from_sparse(
        matrix: &SparseMatrix,
        rows: &[usize],
        columns: &[usize],
    ) -> Result<Self, HomologyError> {
        let mut output = Self::zeros(rows.len(), columns.len())?;
        for entry in matrix.live_entries() {
            let Ok(row) = rows.binary_search(&entry.row) else {
                continue;
            };
            let Ok(column) = columns.binary_search(&entry.column) else {
                continue;
            };
            output.at_mut(row, column).clone_from(&entry.value);
        }
        Ok(output)
    }

    fn trailing_rows(&self, start: usize) -> Result<Self, HomologyError> {
        let mut output = Self::zeros(self.rows - start, self.columns)?;
        for row in start..self.rows {
            for column in 0..self.columns {
                output
                    .at_mut(row - start, column)
                    .clone_from(self.at(row, column));
            }
        }
        Ok(output)
    }

    const fn cells(&self) -> usize {
        self.values.len()
    }

    fn at(&self, row: usize, column: usize) -> &BigInt {
        &self.values[row * self.columns + column]
    }

    fn at_mut(&mut self, row: usize, column: usize) -> &mut BigInt {
        &mut self.values[row * self.columns + column]
    }

    fn swap_rows(&mut self, left: usize, right: usize) {
        if left != right {
            for column in 0..self.columns {
                self.values
                    .swap(left * self.columns + column, right * self.columns + column);
            }
        }
    }

    fn swap_columns(&mut self, left: usize, right: usize) {
        if left != right {
            for row in 0..self.rows {
                self.values
                    .swap(row * self.columns + left, row * self.columns + right);
            }
        }
    }

    fn add_row(&mut self, target: usize, source: usize, factor: &BigInt) {
        for column in 0..self.columns {
            let value = self.at(target, column) + self.at(source, column) * factor;
            *self.at_mut(target, column) = value;
        }
    }

    fn add_column(&mut self, target: usize, source: usize, factor: &BigInt) {
        for row in 0..self.rows {
            let value = self.at(row, target) + self.at(row, source) * factor;
            *self.at_mut(row, target) = value;
        }
    }

    fn negate_row(&mut self, row: usize) {
        for column in 0..self.columns {
            *self.at_mut(row, column) = -self.at(row, column);
        }
    }

    fn max_bits(&self) -> u64 {
        self.values.iter().map(BigInt::bits).max().unwrap_or(0)
    }
}

fn dense_logical_bytes(matrix: &DenseMatrix, bits: u64) -> Result<u64, HomologyError> {
    logical_bytes(
        matrix
            .cells()
            .checked_mul(scalar_bytes(bits)?)
            .ok_or(HomologyError::Overflow)?,
    )
}

#[derive(Clone, Debug)]
enum BasisOp {
    Swap(usize, usize),
    Add {
        target: usize,
        source: usize,
        factor: BigInt,
    },
    Negate(usize),
}

#[derive(Debug)]
struct SmithForm {
    diagonal: Vec<BigInt>,
    row_ops: Vec<BasisOp>,
    column_ops: Vec<BasisOp>,
}

impl SmithForm {
    fn rank(&self) -> usize {
        self.diagonal
            .iter()
            .take_while(|value| !value.is_zero())
            .count()
    }
}

fn smith_reduce(
    matrix: &mut DenseMatrix,
    adjacent: Option<&mut DenseMatrix>,
    meter: &mut HomologyUse,
    base_logical_bytes: u64,
) -> Result<SmithForm, HomologyError> {
    SmithReducer::new(matrix, adjacent, meter, base_logical_bytes).reduce()
}

struct SmithReducer<'a> {
    matrix: &'a mut DenseMatrix,
    adjacent: Option<&'a mut DenseMatrix>,
    meter: &'a mut HomologyUse,
    base_logical_bytes: u64,
    row_ops: Vec<BasisOp>,
    column_ops: Vec<BasisOp>,
}

impl<'a> SmithReducer<'a> {
    const fn new(
        matrix: &'a mut DenseMatrix,
        adjacent: Option<&'a mut DenseMatrix>,
        meter: &'a mut HomologyUse,
        base_logical_bytes: u64,
    ) -> Self {
        Self {
            matrix,
            adjacent,
            meter,
            base_logical_bytes,
            row_ops: Vec::new(),
            column_ops: Vec::new(),
        }
    }

    fn reduce(mut self) -> Result<SmithForm, HomologyError> {
        let diagonal_size = self.matrix.rows.min(self.matrix.columns);
        for diagonal in 0..diagonal_size {
            self.meter.charge(
                self.matrix
                    .rows
                    .saturating_sub(diagonal)
                    .saturating_mul(self.matrix.columns.saturating_sub(diagonal)),
            )?;
            let Some((row, column)) = smallest_nonzero(self.matrix, diagonal) else {
                break;
            };
            self.row_swap(diagonal, row)?;
            self.column_swap(diagonal, column)?;
            self.reduce_pivot(diagonal)?;
            if self.matrix.at(diagonal, diagonal).sign() == Sign::Minus {
                self.row_negate(diagonal)?;
            }
        }
        let mut diagonal = reserved_vec(diagonal_size)?;
        diagonal.extend(
            (0..diagonal_size).map(|index| std::mem::take(self.matrix.at_mut(index, index))),
        );
        if diagonal
            .windows(2)
            .any(|pair| !pair[0].is_zero() && !(&pair[1] % &pair[0]).is_zero())
        {
            return Err(HomologyError::InternalInvariant);
        }
        Ok(SmithForm {
            diagonal,
            row_ops: self.row_ops,
            column_ops: self.column_ops,
        })
    }

    fn reduce_pivot(&mut self, diagonal: usize) -> Result<(), HomologyError> {
        loop {
            if self.reduce_column(diagonal)? || self.reduce_row(diagonal)? {
                continue;
            }
            self.meter.charge(
                self.matrix
                    .rows
                    .saturating_sub(diagonal + 1)
                    .saturating_mul(self.matrix.columns.saturating_sub(diagonal + 1)),
            )?;
            let Some(row) = first_not_divisible(self.matrix, diagonal) else {
                return Ok(());
            };
            self.row_add(diagonal, row, BigInt::one())?;
        }
    }

    fn reduce_column(&mut self, diagonal: usize) -> Result<bool, HomologyError> {
        for row in diagonal + 1..self.matrix.rows {
            if self.matrix.at(row, diagonal).is_zero() {
                continue;
            }
            self.meter.charge(1)?;
            let quotient = self.matrix.at(row, diagonal) / self.matrix.at(diagonal, diagonal);
            let remainder = self.matrix.at(row, diagonal) % self.matrix.at(diagonal, diagonal);
            if !quotient.is_zero() {
                self.row_add(row, diagonal, -quotient)?;
            }
            if !remainder.is_zero() {
                self.row_swap(diagonal, row)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn reduce_row(&mut self, diagonal: usize) -> Result<bool, HomologyError> {
        for column in diagonal + 1..self.matrix.columns {
            if self.matrix.at(diagonal, column).is_zero() {
                continue;
            }
            self.meter.charge(1)?;
            let quotient = self.matrix.at(diagonal, column) / self.matrix.at(diagonal, diagonal);
            let remainder = self.matrix.at(diagonal, column) % self.matrix.at(diagonal, diagonal);
            if !quotient.is_zero() {
                self.column_add(column, diagonal, -quotient)?;
            }
            if !remainder.is_zero() {
                self.column_swap(diagonal, column)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn preflight(&mut self, factor: Option<&BigInt>, rows: bool) -> Result<(), HomologyError> {
        let log = if rows {
            &self.row_ops
        } else {
            &self.column_ops
        };
        let work = if rows {
            self.matrix.columns
        } else {
            self.matrix.rows
        };
        let adjacent_bits = self.adjacent.as_deref().map_or(0, DenseMatrix::max_bits);
        let adjacent_cells = self.adjacent.as_deref().map_or(0, DenseMatrix::cells);
        let current_bits = self.matrix.max_bits().max(adjacent_bits);
        let bits = factor.map_or(Ok(current_bits), |factor| {
            add_product_bits(current_bits, current_bits, factor.bits())
        })?;
        let matrix_bytes = logical_bytes(
            self.matrix
                .cells()
                .checked_add(adjacent_cells)
                .ok_or(HomologyError::Overflow)?
                .checked_mul(scalar_bytes(bits)?)
                .ok_or(HomologyError::Overflow)?,
        )?;
        let log_bytes = logical_bytes(
            log.len()
                .checked_add(1)
                .and_then(|count| {
                    count.checked_mul(size_of::<BasisOp>() + scalar_bytes(bits).ok()?)
                })
                .ok_or(HomologyError::Overflow)?,
        )?;
        self.meter.workspace(
            bits,
            self.base_logical_bytes
                .checked_add(matrix_bytes)
                .and_then(|bytes| bytes.checked_add(log_bytes))
                .ok_or(HomologyError::Overflow)?,
        )?;
        self.meter.charge(work)
    }

    fn row_swap(&mut self, left: usize, right: usize) -> Result<(), HomologyError> {
        if left == right {
            return Ok(());
        }
        self.preflight(None, true)?;
        reserve_op(&mut self.row_ops)?;
        self.matrix.swap_rows(left, right);
        self.row_ops.push(BasisOp::Swap(left, right));
        Ok(())
    }

    fn row_add(
        &mut self,
        target: usize,
        source: usize,
        factor: BigInt,
    ) -> Result<(), HomologyError> {
        self.preflight(Some(&factor), true)?;
        reserve_op(&mut self.row_ops)?;
        self.matrix.add_row(target, source, &factor);
        self.row_ops.push(BasisOp::Add {
            target,
            source,
            factor,
        });
        Ok(())
    }

    fn row_negate(&mut self, row: usize) -> Result<(), HomologyError> {
        self.preflight(None, true)?;
        reserve_op(&mut self.row_ops)?;
        self.matrix.negate_row(row);
        self.row_ops.push(BasisOp::Negate(row));
        Ok(())
    }

    fn column_swap(&mut self, left: usize, right: usize) -> Result<(), HomologyError> {
        if left == right {
            return Ok(());
        }
        self.preflight(None, false)?;
        reserve_op(&mut self.column_ops)?;
        self.matrix.swap_columns(left, right);
        if let Some(adjacent) = self.adjacent.as_deref_mut() {
            adjacent.swap_rows(left, right);
        }
        self.column_ops.push(BasisOp::Swap(left, right));
        Ok(())
    }

    fn column_add(
        &mut self,
        target: usize,
        source: usize,
        factor: BigInt,
    ) -> Result<(), HomologyError> {
        self.preflight(Some(&factor), false)?;
        reserve_op(&mut self.column_ops)?;
        self.matrix.add_column(target, source, &factor);
        if let Some(adjacent) = self.adjacent.as_deref_mut() {
            adjacent.add_row(source, target, &(-&factor));
        }
        self.column_ops.push(BasisOp::Add {
            target,
            source,
            factor,
        });
        Ok(())
    }
}

fn reserve_op(log: &mut Vec<BasisOp>) -> Result<(), HomologyError> {
    log.try_reserve(1).map_err(|_| HomologyError::Allocation)
}

fn smallest_nonzero(matrix: &DenseMatrix, start: usize) -> Option<(usize, usize)> {
    (start..matrix.rows)
        .flat_map(|row| (start..matrix.columns).map(move |column| (row, column)))
        .filter(|&(row, column)| !matrix.at(row, column).is_zero())
        .min_by_key(|&(row, column)| (matrix.at(row, column).bits(), row, column))
}

fn first_not_divisible(matrix: &DenseMatrix, diagonal: usize) -> Option<usize> {
    let pivot = matrix.at(diagonal, diagonal);
    (diagonal + 1..matrix.rows).find(|&row| {
        (diagonal + 1..matrix.columns).any(|column| !(matrix.at(row, column) % pivot).is_zero())
    })
}

fn inverse_row_basis(
    size: usize,
    index: usize,
    operations: &[BasisOp],
    meter: &mut HomologyUse,
) -> Result<Vec<BigInt>, HomologyError> {
    meter.workspace(
        1,
        logical_bytes(
            size.checked_mul(scalar_bytes(1)?)
                .ok_or(HomologyError::Overflow)?,
        )?,
    )?;
    let mut vector = zero_integers(size)?;
    vector[index] = BigInt::one();
    for operation in operations.iter().rev() {
        match operation {
            BasisOp::Swap(left, right) => vector.swap(*left, *right),
            BasisOp::Add {
                target,
                source,
                factor,
            } => {
                meter.charge(1)?;
                meter.coefficients_bound(add_product_bits(
                    vector[*target].bits(),
                    factor.bits(),
                    vector[*source].bits(),
                )?)?;
                let correction = factor * &vector[*source];
                vector[*target] -= correction;
            }
            BasisOp::Negate(row) => vector[*row] = -&vector[*row],
        }
    }
    Ok(vector)
}

fn column_basis(
    size: usize,
    index: usize,
    operations: &[BasisOp],
    meter: &mut HomologyUse,
) -> Result<Vec<BigInt>, HomologyError> {
    meter.workspace(
        1,
        logical_bytes(
            size.checked_mul(scalar_bytes(1)?)
                .ok_or(HomologyError::Overflow)?,
        )?,
    )?;
    let mut vector = zero_integers(size)?;
    vector[index] = BigInt::one();
    apply_column_ops(&mut vector, operations, meter)?;
    Ok(vector)
}

fn apply_column_ops(
    vector: &mut [BigInt],
    operations: &[BasisOp],
    meter: &mut HomologyUse,
) -> Result<(), HomologyError> {
    for operation in operations.iter().rev() {
        match operation {
            BasisOp::Swap(left, right) => vector.swap(*left, *right),
            BasisOp::Add {
                target,
                source,
                factor,
            } => {
                meter.charge(1)?;
                meter.coefficients_bound(add_product_bits(
                    vector[*source].bits(),
                    factor.bits(),
                    vector[*target].bits(),
                )?)?;
                let correction = factor * &vector[*target];
                vector[*source] += correction;
            }
            BasisOp::Negate(row) => vector[*row] = -&vector[*row],
        }
    }
    Ok(())
}

fn scatter(values: &[BigInt], basis: &[usize], size: usize) -> Result<Vec<BigInt>, HomologyError> {
    let mut output = zero_integers(size)?;
    for (value, &index) in values.iter().zip(basis) {
        output[index].clone_from(value);
    }
    Ok(output)
}

fn chain_from_vector(
    chain: &IntegralChainComplex,
    degree: usize,
    values: &[BigInt],
) -> Result<IntegralChain, HomologyError> {
    chain
        .space(degree)
        .map_err(|_| HomologyError::InternalInvariant)?
        .element(
            values
                .iter()
                .enumerate()
                .filter(|(_, value)| !value.is_zero())
                .map(|(index, value)| (index, value.clone())),
        )
        .map_err(|_| HomologyError::InternalInvariant)
}

#[derive(Debug)]
struct GroupData {
    free_cycles: Box<[IntegralChain]>,
    torsion_orders: Box<[BigInt]>,
    torsion_cycles: Box<[IntegralChain]>,
    torsion_bounds: Box<[IntegralChain]>,
}

impl GroupData {
    fn chains(&self) -> impl Iterator<Item = &IntegralChain> {
        self.free_cycles
            .iter()
            .chain(&self.torsion_cycles)
            .chain(&self.torsion_bounds)
    }

    fn transport(
        &self,
        degree: usize,
        isomorphism: &ChainIsomorphism<IntegerRing>,
        meter: &mut HomologyUse,
    ) -> Result<Self, HomologyError> {
        let cycles = isomorphism
            .forward(degree)
            .map_err(|_| HomologyError::InvalidChain)?;
        let free_cycles = transport_chains(&cycles, &self.free_cycles, meter)?;
        let torsion_cycles = transport_chains(&cycles, &self.torsion_cycles, meter)?;
        let torsion_bounds = if self.torsion_bounds.is_empty() {
            Box::new([])
        } else {
            let bounds = isomorphism
                .forward(degree + 1)
                .map_err(|_| HomologyError::InvalidChain)?;
            transport_chains(&bounds, &self.torsion_bounds, meter)?
        };
        Ok(Self {
            free_cycles,
            torsion_orders: self.torsion_orders.clone(),
            torsion_cycles,
            torsion_bounds,
        })
    }
}

fn bigint_values_bytes<'a>(
    values: impl IntoIterator<Item = &'a BigInt>,
) -> Result<u64, HomologyError> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(logical_bytes(scalar_bytes(value.bits())?)?)
            .ok_or(HomologyError::Overflow)
    })
}

fn chain_payload_bytes(chain: &IntegralChain) -> Result<u64, HomologyError> {
    let indices = chain
        .indices()
        .len()
        .checked_mul(size_of::<usize>())
        .ok_or(HomologyError::Overflow)?;
    logical_bytes(indices)?
        .checked_add(bigint_values_bytes(chain.coefficients())?)
        .ok_or(HomologyError::Overflow)
}

fn boundary_result_bits(
    chain: &IntegralChainComplex,
    degree: usize,
    value: &IntegralChain,
) -> Result<u64, HomologyError> {
    let input = value
        .coefficients()
        .iter()
        .map(BigInt::bits)
        .max()
        .unwrap_or(0);
    let boundary = chain
        .chain_view()
        .boundary(degree)
        .map_err(|_| HomologyError::InternalInvariant)?;
    let coefficients = boundary
        .exact_entries()
        .map(|(_, _, coefficient)| {
            u64::from(i64::BITS - coefficient.unsigned_abs().leading_zeros())
        })
        .max()
        .unwrap_or(0);
    product_bits([input, coefficients, ceil_log2(value.indices().len())])
}

fn group_payload_bytes(group: &GroupData) -> Result<u64, HomologyError> {
    let chains = group.chains().try_fold(0_u64, |total, chain| {
        total
            .checked_add(logical_bytes(size_of::<IntegralChain>())?)
            .and_then(|bytes| bytes.checked_add(chain_payload_bytes(chain).ok()?))
            .ok_or(HomologyError::Overflow)
    })?;
    chains
        .checked_add(bigint_values_bytes(&group.torsion_orders)?)
        .ok_or(HomologyError::Overflow)
}

fn analysis_payload_bytes(analysis: &IntegralHomology) -> Result<u64, HomologyError> {
    let outer = analysis
        .degrees
        .len()
        .checked_mul(size_of::<usize>() + size_of::<GroupData>())
        .ok_or(HomologyError::Overflow)?;
    analysis
        .groups
        .iter()
        .try_fold(logical_bytes(outer)?, |total, group| {
            total
                .checked_add(group_payload_bytes(group)?)
                .ok_or(HomologyError::Overflow)
        })
}

fn group_max_bits(group: &GroupData) -> u64 {
    group
        .chains()
        .flat_map(crate::Element::coefficients)
        .chain(group.torsion_orders.iter())
        .map(BigInt::bits)
        .max()
        .unwrap_or(0)
}

fn transport_chains(
    map: &LinearMap<IntegerRing, Chain, Chain>,
    values: &[IntegralChain],
    meter: &mut HomologyUse,
) -> Result<Box<[IntegralChain]>, HomologyError> {
    let mut output = reserved_vec(values.len())?;
    for value in values {
        meter.charge(value.indices().len())?;
        meter.coefficients(value.coefficients())?;
        let mapped = map.apply(value).map_err(map_chain_error)?;
        meter.coefficients(mapped.coefficients())?;
        output.push(mapped);
    }
    Ok(output.into_boxed_slice())
}

/// Immutable exact integral-homology analysis over one chain authority.
#[derive(Debug)]
pub struct IntegralHomology {
    chain: IntegralChainComplex,
    degrees: Box<[usize]>,
    groups: Box<[GroupData]>,
}

impl IntegralHomology {
    /// Prepare ordinary absolute integral homology for canonicalized degrees.
    ///
    /// # Errors
    ///
    /// Returns a classified degree, resource, allocation, or exact-law failure
    /// without publishing a partial analysis.
    pub fn prepare(
        chain: &IntegralChainComplex,
        requested_degrees: impl IntoIterator<Item = usize>,
        limit: HomologyLimit,
    ) -> Result<Self, HomologyError> {
        let mut meter = HomologyUse::new(limit);
        let mut degrees = Vec::new();
        for degree in requested_degrees {
            let count = degrees
                .len()
                .checked_add(1)
                .ok_or(HomologyError::Overflow)?;
            meter.workspace(
                0,
                logical_bytes(
                    count
                        .checked_mul(size_of::<usize>())
                        .ok_or(HomologyError::Overflow)?,
                )?,
            )?;
            degrees
                .try_reserve(1)
                .map_err(|_| HomologyError::Allocation)?;
            degrees.push(degree);
        }
        degrees.sort_unstable();
        degrees.dedup();
        if degrees.iter().any(|&degree| degree > chain.dimension()) {
            return Err(HomologyError::DegreeOutside);
        }
        if degrees.is_empty() {
            return Ok(Self {
                chain: chain.clone(),
                degrees: Box::new([]),
                groups: Box::new([]),
            });
        }

        meter.retain(logical_bytes(
            degrees
                .len()
                .checked_mul(size_of::<usize>() + size_of::<GroupData>())
                .ok_or(HomologyError::Overflow)?,
        )?)?;
        let mut reduced = ReducedChain::ingest(chain, &mut meter)?;
        reduced.verify_chain(&mut meter)?;
        reduced.contract(&mut meter)?;
        reduced.verify_chain(&mut meter)?;

        let mut groups = reserved_vec(degrees.len())?;
        for &degree in &degrees {
            let group = analyze_degree(chain, &reduced, degree, &mut meter)?;
            meter.retain(group_payload_bytes(&group)?)?;
            groups.push(group);
        }
        Ok(Self {
            chain: chain.clone(),
            degrees: degrees.into_boxed_slice(),
            groups: groups.into_boxed_slice(),
        })
    }

    /// Canonical sorted unique requested degrees.
    #[must_use]
    pub fn degrees(&self) -> &[usize] {
        &self.degrees
    }

    /// Borrow one requested homology group.
    #[must_use]
    pub fn group(&self, degree: usize) -> Option<HomologyGroup<'_>> {
        self.degrees
            .binary_search(&degree)
            .ok()
            .map(|index| HomologyGroup {
                degree: self.degrees[index],
                data: &self.groups[index],
                chain: &self.chain,
            })
    }

    /// Retained exact chain authority.
    #[must_use]
    pub const fn chain_complex(&self) -> &IntegralChainComplex {
        &self.chain
    }

    /// Transport retained representatives through a checked chain isomorphism.
    ///
    /// This constructs a distinct target-owned analysis; it does not identify
    /// nominal owners or recompute normal forms.
    ///
    /// # Errors
    ///
    /// Returns `owner_mismatch` unless the isomorphism starts at this exact
    /// chain authority. Resource and exact-law failures publish no result.
    pub fn transport(
        &self,
        isomorphism: &ChainIsomorphism<IntegerRing>,
        limit: HomologyLimit,
    ) -> Result<Self, HomologyError> {
        if !self.chain.same_owner(isomorphism.source()) {
            return Err(HomologyError::OwnerMismatch);
        }
        let mut meter = HomologyUse::new(limit);
        meter.charge(self.groups.len())?;
        let input_bits = self.groups.iter().map(group_max_bits).max().unwrap_or(0);
        meter.input(analysis_payload_bytes(self)?, input_bits)?;
        meter.retain(logical_bytes(
            self.groups
                .len()
                .checked_mul(size_of::<usize>() + size_of::<GroupData>())
                .ok_or(HomologyError::Overflow)?,
        )?)?;

        let mut groups = reserved_vec(self.groups.len())?;
        for (&degree, group) in self.degrees.iter().zip(&self.groups) {
            let group = group.transport(degree, isomorphism, &mut meter)?;
            meter.retain(group_payload_bytes(&group)?)?;
            groups.push(group);
        }
        let target = isomorphism.target().clone();
        for (&degree, group) in self.degrees.iter().zip(&groups) {
            verify_group(&target, degree, group, &mut meter)?;
        }
        Ok(Self {
            chain: target,
            degrees: self.degrees.clone(),
            groups: groups.into_boxed_slice(),
        })
    }
}

fn analyze_degree(
    chain: &IntegralChainComplex,
    reduced: &ReducedChain,
    degree: usize,
    meter: &mut HomologyUse,
) -> Result<GroupData, HomologyError> {
    let lower_basis = if degree == 0 {
        Vec::new()
    } else {
        reduced.active_indices(degree - 1)?
    };
    let basis = reduced.active_indices(degree)?;
    let upper_basis = if degree == chain.dimension() {
        Vec::new()
    } else {
        reduced.active_indices(degree + 1)?
    };
    let (reduced_bits, reduced_bytes) = reduced.workspace_profile()?;
    let dense_cells = lower_basis
        .len()
        .checked_mul(basis.len())
        .and_then(|cells| {
            basis
                .len()
                .checked_mul(upper_basis.len())
                .and_then(|successor| cells.checked_add(successor))
        })
        .ok_or(HomologyError::Overflow)?;
    meter.workspace(
        reduced_bits,
        reduced_bytes
            .checked_add(logical_bytes(
                dense_cells
                    .checked_mul(scalar_bytes(reduced_bits)?)
                    .ok_or(HomologyError::Overflow)?,
            )?)
            .ok_or(HomologyError::Overflow)?,
    )?;
    let mut boundary = DenseMatrix::from_sparse(&reduced.boundaries[degree], &lower_basis, &basis)?;
    let mut successor = if degree == chain.dimension() {
        DenseMatrix::zeros(basis.len(), 0)?
    } else {
        DenseMatrix::from_sparse(&reduced.boundaries[degree + 1], &basis, &upper_basis)?
    };
    let dense_bits = boundary.max_bits().max(successor.max_bits());
    meter.workspace(
        reduced_bits.max(dense_bits),
        reduced_bytes
            .checked_add(dense_logical_bytes(&boundary, dense_bits)?)
            .and_then(|bytes| bytes.checked_add(dense_logical_bytes(&successor, dense_bits).ok()?))
            .ok_or(HomologyError::Overflow)?,
    )?;
    let kernel_form = smith_reduce(&mut boundary, Some(&mut successor), meter, reduced_bytes)?;
    let kernel_rank = kernel_form.rank();
    for row in 0..kernel_rank {
        if (0..successor.columns).any(|column| !successor.at(row, column).is_zero()) {
            return Err(HomologyError::InvalidChain);
        }
    }
    let mut image = successor.trailing_rows(kernel_rank)?;
    let retained_dense_bits = boundary.max_bits().max(successor.max_bits());
    let retained_dense_bytes = reduced_bytes
        .checked_add(dense_logical_bytes(&boundary, retained_dense_bits)?)
        .and_then(|bytes| {
            bytes.checked_add(dense_logical_bytes(&successor, retained_dense_bits).ok()?)
        })
        .ok_or(HomologyError::Overflow)?;
    let quotient_form = smith_reduce(&mut image, None, meter, retained_dense_bytes)?;
    DegreeReduction {
        chain,
        reduced,
        meter,
        degree,
        basis,
        upper_basis,
        kernel_rank,
        kernel_form,
        quotient_form,
    }
    .publish()
}

struct DegreeReduction<'a> {
    chain: &'a IntegralChainComplex,
    reduced: &'a ReducedChain,
    meter: &'a mut HomologyUse,
    degree: usize,
    basis: Vec<usize>,
    upper_basis: Vec<usize>,
    kernel_rank: usize,
    kernel_form: SmithForm,
    quotient_form: SmithForm,
}

impl DegreeReduction<'_> {
    fn publish(mut self) -> Result<GroupData, HomologyError> {
        let capacity = self.basis.len() - self.kernel_rank;
        let mut free = Vec::new();
        let mut orders = Vec::new();
        let mut torsion = Vec::new();
        let mut bounds = Vec::new();
        for length in [&mut free, &mut torsion, &mut bounds] {
            length
                .try_reserve_exact(capacity)
                .map_err(|_| HomologyError::Allocation)?;
        }
        orders
            .try_reserve_exact(capacity)
            .map_err(|_| HomologyError::Allocation)?;
        let quotient_rank = self.quotient_form.rank();
        for generator in 0..capacity {
            let diagonal = self.quotient_form.diagonal.get(generator);
            if diagonal.is_some_and(|value| value == &BigInt::one()) {
                continue;
            }
            if generator < quotient_rank {
                orders.push(diagonal.cloned().ok_or(HomologyError::InternalInvariant)?);
                torsion.push(self.cycle(generator)?);
                bounds.push(self.bound(generator)?);
            } else {
                free.push(self.cycle(generator)?);
            }
        }
        let data = GroupData {
            free_cycles: free.into_boxed_slice(),
            torsion_orders: orders.into_boxed_slice(),
            torsion_cycles: torsion.into_boxed_slice(),
            torsion_bounds: bounds.into_boxed_slice(),
        };
        verify_group(self.chain, self.degree, &data, self.meter)?;
        Ok(data)
    }

    fn cycle(&mut self, generator: usize) -> Result<IntegralChain, HomologyError> {
        let mut coordinates = zero_integers(self.basis.len())?;
        let quotient = inverse_row_basis(
            coordinates.len() - self.kernel_rank,
            generator,
            &self.quotient_form.row_ops,
            self.meter,
        )?;
        coordinates[self.kernel_rank..].clone_from_slice(&quotient);
        apply_column_ops(&mut coordinates, &self.kernel_form.column_ops, self.meter)?;
        publish_original(
            self.chain,
            self.reduced,
            self.meter,
            self.degree,
            &coordinates,
            &self.basis,
        )
    }

    fn bound(&mut self, generator: usize) -> Result<IntegralChain, HomologyError> {
        let coordinates = column_basis(
            self.upper_basis.len(),
            generator,
            &self.quotient_form.column_ops,
            self.meter,
        )?;
        publish_original(
            self.chain,
            self.reduced,
            self.meter,
            self.degree + 1,
            &coordinates,
            &self.upper_basis,
        )
    }
}

fn publish_original(
    chain: &IntegralChainComplex,
    reduced: &ReducedChain,
    meter: &mut HomologyUse,
    degree: usize,
    coordinates: &[BigInt],
    basis: &[usize],
) -> Result<IntegralChain, HomologyError> {
    let size = chain
        .space(degree)
        .map_err(|_| HomologyError::InternalInvariant)?
        .basis_size();
    let mut original = scatter(coordinates, basis, size)?;
    reduced.lift(degree, &mut original, meter)?;
    meter.coefficients(&original)?;
    chain_from_vector(chain, degree, &original)
}

fn verify_group(
    chain: &IntegralChainComplex,
    degree: usize,
    group: &GroupData,
    meter: &mut HomologyUse,
) -> Result<(), HomologyError> {
    for cycle in group.free_cycles.iter().chain(&group.torsion_cycles) {
        meter.charge(cycle.indices().len())?;
        meter.coefficients_bound(boundary_result_bits(chain, degree, cycle)?)?;
        let boundary = chain
            .boundary(degree)
            .map_err(|_| HomologyError::InternalInvariant)?
            .apply(cycle)
            .map_err(map_chain_error)?;
        if !boundary.coefficients().is_empty() {
            return Err(HomologyError::InternalInvariant);
        }
    }
    for ((order, cycle), bound) in group
        .torsion_orders
        .iter()
        .zip(&group.torsion_cycles)
        .zip(&group.torsion_bounds)
    {
        meter.charge(cycle.indices().len() + bound.indices().len())?;
        meter.coefficients_bound(boundary_result_bits(chain, degree + 1, bound)?)?;
        let boundary = chain
            .boundary(degree + 1)
            .map_err(|_| HomologyError::InternalInvariant)?
            .apply(bound)
            .map_err(map_chain_error)?;
        let cycle_bits = cycle
            .coefficients()
            .iter()
            .map(BigInt::bits)
            .max()
            .unwrap_or(0);
        meter.coefficients_bound(product_bits([cycle_bits, order.bits()])?)?;
        let multiple = chain
            .space(degree)
            .map_err(|_| HomologyError::InternalInvariant)?
            .element(
                cycle
                    .indices()
                    .iter()
                    .copied()
                    .zip(cycle.coefficients().iter().map(|value| value * order)),
            )
            .map_err(map_chain_error)?;
        if boundary.indices() != multiple.indices()
            || boundary.coefficients() != multiple.coefficients()
        {
            return Err(HomologyError::InternalInvariant);
        }
    }
    Ok(())
}

const fn map_chain_error(error: ChainError) -> HomologyError {
    match error {
        ChainError::SpaceMismatch | ChainError::BasisIndexOutside { .. } => {
            HomologyError::InternalInvariant
        }
        ChainError::Topology(_) => HomologyError::InvalidChain,
    }
}

/// Borrowed view of one requested exact integral-homology group.
#[derive(Clone, Copy, Debug)]
pub struct HomologyGroup<'a> {
    degree: usize,
    data: &'a GroupData,
    chain: &'a IntegralChainComplex,
}

impl<'a> HomologyGroup<'a> {
    #[must_use]
    pub const fn degree(self) -> usize {
        self.degree
    }

    #[must_use]
    pub fn free_rank(self) -> usize {
        self.data.free_cycles.len()
    }

    #[must_use]
    pub fn torsion_orders(self) -> &'a [BigInt] {
        &self.data.torsion_orders
    }

    #[must_use]
    pub fn free_cycle(self, index: usize) -> Option<&'a IntegralChain> {
        self.data.free_cycles.get(index)
    }

    /// Explicitly realize the retained free generators in their binary64 chain basis.
    ///
    /// # Errors
    ///
    /// Rejects allocation failure or any exact coefficient without an exact finite
    /// binary64 representation.
    pub fn realize_free_cycles_binary64(
        self,
    ) -> Result<Box<[Binary64Chain]>, Binary64ElementError> {
        let exact_space = self
            .chain
            .space(self.degree)
            .map_err(|_| Binary64ElementError::SpaceMismatch)?;
        let space = Binary64Space::<Chain>::from_basis(&exact_space);
        let mut cycles = Vec::new();
        cycles
            .try_reserve_exact(self.data.free_cycles.len())
            .map_err(|_| Binary64ElementError::Allocation)?;
        for cycle in &self.data.free_cycles {
            cycles.push(Binary64Element::realize_integral(space.clone(), cycle)?);
        }
        Ok(cycles.into_boxed_slice())
    }

    /// Evaluate a binary64 cochain on every retained exact free generator.
    ///
    /// # Errors
    ///
    /// Rejects a foreign owner or degree, allocation failure, and a nonzero exact
    /// period that overflows or underflows finite binary64.
    pub fn periods_binary64(
        self,
        cochain: &Binary64Cochain,
    ) -> Result<Box<[f64]>, Binary64ElementError> {
        let exact_space = self
            .chain
            .space(self.degree)
            .map_err(|_| Binary64ElementError::SpaceMismatch)?;
        let expected = Binary64Space::<Chain>::from_basis(&exact_space);
        if !expected.same_basis(cochain.space()) {
            return Err(Binary64ElementError::SpaceMismatch);
        }
        let mut periods = Vec::new();
        periods
            .try_reserve_exact(self.data.free_cycles.len())
            .map_err(|_| Binary64ElementError::Allocation)?;
        for cycle in &self.data.free_cycles {
            periods.push(
                exact_integer_binary64_dot(
                    cycle.indices(),
                    cycle.coefficients(),
                    cochain.coefficients(),
                )
                .ok_or(Binary64ElementError::ScalarConversion)?,
            );
        }
        Ok(periods.into_boxed_slice())
    }

    #[must_use]
    pub fn torsion_cycle(self, index: usize) -> Option<&'a IntegralChain> {
        self.data.torsion_cycles.get(index)
    }

    #[must_use]
    pub fn torsion_bound(self, index: usize) -> Option<&'a IntegralChain> {
        self.data.torsion_bounds.get(index)
    }
}

#[cfg(test)]
mod determinantal_oracle {
    use num_integer::Integer;
    use num_traits::Signed;

    use super::*;

    fn combinations(bound: usize, size: usize) -> Vec<Vec<usize>> {
        (0..1_usize << bound)
            .filter(|mask| mask.count_ones() as usize == size)
            .map(|mask| {
                (0..bound)
                    .filter(|index| mask & (1 << index) != 0)
                    .collect()
            })
            .collect()
    }

    fn determinant(matrix: &DenseMatrix, rows: &[usize], columns: &[usize]) -> BigInt {
        match rows.len() {
            0 => BigInt::one(),
            1 => matrix.at(rows[0], columns[0]).clone(),
            _ => columns
                .iter()
                .enumerate()
                .map(|(offset, &column)| {
                    let remaining = columns
                        .iter()
                        .copied()
                        .filter(|candidate| *candidate != column)
                        .collect::<Vec<_>>();
                    let minor = determinant(matrix, &rows[1..], &remaining);
                    if offset % 2 == 0 {
                        matrix.at(rows[0], column) * minor
                    } else {
                        -(matrix.at(rows[0], column) * minor)
                    }
                })
                .sum(),
        }
    }

    fn invariant_factors(matrix: &DenseMatrix) -> Vec<BigInt> {
        let mut previous = BigInt::one();
        (1..=matrix.rows.min(matrix.columns))
            .map(|size| {
                let divisor = combinations(matrix.rows, size)
                    .iter()
                    .flat_map(|rows| {
                        combinations(matrix.columns, size)
                            .into_iter()
                            .map(|columns| determinant(matrix, rows, &columns).abs())
                    })
                    .fold(BigInt::zero(), |gcd, minor| gcd.gcd(&minor));
                if divisor.is_zero() {
                    BigInt::zero()
                } else {
                    let factor = &divisor / &previous;
                    previous = divisor;
                    factor
                }
            })
            .collect()
    }

    fn matrix(rows: usize, columns: usize, values: &[i64]) -> DenseMatrix {
        assert_eq!(values.len(), rows * columns);
        DenseMatrix {
            rows,
            columns,
            values: values.iter().copied().map(BigInt::from).collect(),
        }
    }

    #[test]
    fn smith_invariants_match_independent_determinantal_divisors() {
        for (mut candidate, expected) in [
            (matrix(2, 2, &[2, 4, 6, 8]), vec![2, 4]),
            (matrix(2, 3, &[4, 6, 8, 2, 4, 6]), vec![2, 2]),
            (matrix(3, 3, &[2, 0, 0, 0, 6, 0, 0, 0, 0]), vec![2, 6, 0]),
            (matrix(2, 2, &[0, 0, 0, 0]), vec![0, 0]),
        ] {
            let expected = expected.into_iter().map(BigInt::from).collect::<Vec<_>>();
            assert_eq!(invariant_factors(&candidate), expected);
            let mut meter = HomologyUse::new(HomologyLimit::DEFAULT);
            let form = smith_reduce(&mut candidate, None, &mut meter, 0).unwrap();
            assert_eq!(form.diagonal, expected);
        }
    }

    #[test]
    fn cumulative_use_rejects_before_growth_and_distinguishes_storage_lifecycle() {
        let storage = StorageLimit::new(10, 10).unwrap();
        let limit = HomologyLimit::DEFAULT
            .with_storage(storage)
            .with_smith_steps(WorkLimit::new(3));
        let mut use_ = HomologyUse::new(limit);
        use_.input(6, 1).unwrap();
        assert_eq!(
            use_.workspace(1, 5).unwrap_err().resource_limit(),
            Some(("peak_live_logical_bytes", 11, 10))
        );
        use_.charge(2).unwrap();
        assert_eq!(
            use_.charge(2).unwrap_err().resource_limit(),
            Some(("smith_steps", 4, 3))
        );

        let retained = HomologyLimit::DEFAULT.with_storage(StorageLimit::new(5, 10).unwrap());
        assert_eq!(
            HomologyUse::new(retained)
                .retain(6)
                .unwrap_err()
                .resource_limit(),
            Some(("retained_logical_bytes", 6, 5))
        );

        let mut candidate = matrix(2, 1, &[1, 1]);
        let original = candidate.clone();
        let mut use_ = HomologyUse::new(HomologyLimit::DEFAULT.with_coefficient_bits(1));
        let error = SmithReducer::new(&mut candidate, None, &mut use_, 0)
            .row_add(0, 1, BigInt::from(2))
            .unwrap_err();
        assert_eq!(error.resource_limit(), Some(("coefficient_bits", 4, 1)));
        assert_eq!(candidate.values, original.values);
    }
}
