use std::{collections::BTreeMap, mem::size_of};

use num_traits::ToPrimitive;

use crate::form::Binary64Basis;
use crate::operator::{AtomicRecipe, LinearOperator, OperatorError, OperatorRecipe, degree};
use crate::sparse::{CsrMatrix, InvalidCompressedRows};
use crate::{
    BoundaryRef, CoefficientSlice, RealizationError, StorageLimit, TopologyError, Variance,
    WorkLimit,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Estimate {
    shape: (usize, usize),
    nnz_bound: usize,
    topology_entries_bound: u64,
    retained_logical_bytes_bound: u64,
    peak_live_logical_bytes_bound: u64,
    scalar_steps_bound: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BuildLimit {
    storage: StorageLimit,
    work: WorkLimit,
}

impl BuildLimit {
    fn for_estimate(estimate: Estimate) -> Self {
        Self {
            storage: StorageLimit::new(
                estimate.retained_logical_bytes_bound,
                estimate.peak_live_logical_bytes_bound,
            )
            .expect("an estimate preserves retained <= peak"),
            work: WorkLimit::new(estimate.scalar_steps_bound),
        }
    }

    fn rejection(self, estimate: Estimate) -> Option<BuildError> {
        let checks = [
            (
                "retained_logical_bytes",
                estimate.retained_logical_bytes_bound,
                self.storage.retained_logical_bytes(),
            ),
            (
                "peak_live_logical_bytes",
                estimate.peak_live_logical_bytes_bound,
                self.storage.peak_live_logical_bytes(),
            ),
            (
                "scalar_steps",
                estimate.scalar_steps_bound,
                self.work.steps(),
            ),
        ];
        checks
            .into_iter()
            .find(|(_, required, limit)| required > limit)
            .map(|(axis, required, limit)| BuildError::Resource {
                axis,
                required,
                limit,
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BuildError {
    Operator(OperatorError),
    Overflow,
    Allocation,
    InvalidStorage,
    Resource {
        axis: &'static str,
        required: u64,
        limit: u64,
    },
}

impl From<OperatorError> for BuildError {
    fn from(error: OperatorError) -> Self {
        Self::Operator(error)
    }
}

impl From<TopologyError> for BuildError {
    fn from(error: TopologyError) -> Self {
        Self::Operator(error.into())
    }
}

impl From<RealizationError> for BuildError {
    fn from(error: RealizationError) -> Self {
        Self::Operator(error.into())
    }
}

impl From<InvalidCompressedRows> for BuildError {
    fn from(_: InvalidCompressedRows) -> Self {
        Self::InvalidStorage
    }
}

#[derive(Clone, Copy)]
struct Bound {
    rows: usize,
    columns: usize,
    nnz: usize,
    topology_entries: u64,
    retained: u64,
    peak: u64,
    work: u64,
}

fn estimate<S: Variance, T: Variance>(
    operator: &LinearOperator<S, T>,
) -> Result<Estimate, BuildError> {
    let bound = match &operator.recipe {
        OperatorRecipe::Atomic(recipe) => {
            atomic_bound(operator.source.basis(), operator.target.basis(), recipe)?
        }
        OperatorRecipe::Composite(steps) => {
            let mut source = operator.source.basis();
            let mut bounds = steps.iter().map(|step| {
                let bound = atomic_bound(source, &step.target, &step.recipe);
                source = &step.target;
                bound
            });
            let mut current = bounds.next().ok_or(BuildError::Overflow)??;
            for next in bounds {
                current = product_bound(next?, current)?;
            }
            current
        }
    };
    Ok(Estimate {
        shape: (bound.rows, bound.columns),
        nnz_bound: bound.nnz,
        topology_entries_bound: bound.topology_entries,
        retained_logical_bytes_bound: bound.retained,
        peak_live_logical_bytes_bound: bound.peak,
        scalar_steps_bound: bound.work,
    })
}

fn atomic_bound(
    source: &Binary64Basis,
    target: &Binary64Basis,
    recipe: &AtomicRecipe,
) -> Result<Bound, BuildError> {
    let (rows, columns) = (target.size(), source.size());
    let (nnz, topology_entries, candidates, scratch) = match recipe {
        AtomicRecipe::Differential => {
            let target = target.full().ok_or(OperatorError::FullSpaceRequired)?;
            let target_degree =
                usize::try_from(target.degree).map_err(|_| OperatorError::DegreeOutside)?;
            if target_degree == 0 || target_degree > target.domain.view().dimension() {
                (0, 0, 0, 0)
            } else {
                let boundary = boundary(target, target_degree)?;
                let nnz = boundary.indices().len();
                (nnz, as_u64(nnz)?, nnz, rows)
            }
        }
        AtomicRecipe::Restriction | AtomicRecipe::ExtensionByZero => {
            let selection = selected(match recipe {
                AtomicRecipe::Restriction => target,
                _ => source,
            })?;
            (selection.len(), 0, selection.len(), 0)
        }
        AtomicRecipe::Riesz(realization) | AtomicRecipe::InverseRiesz(realization) => {
            realization.hodge_coefficients(degree(source)?)?;
            (rows, 0, rows, 0)
        }
        AtomicRecipe::Codifferential(realization) => {
            let boundary = realization
                .topology()
                .chain_view()
                .boundary(degree(source)?)?;
            let nnz = boundary.indices().len();
            (nnz, as_u64(nnz)?, nnz, 0)
        }
        AtomicRecipe::Laplacian(realization) => {
            let (topology, candidates, scratch) = laplacian_counts(realization, degree(source)?)?;
            (
                rows.checked_mul(columns)
                    .ok_or(BuildError::Overflow)?
                    .min(candidates),
                topology,
                candidates,
                scratch,
            )
        }
        AtomicRecipe::Identity => (rows, 0, rows, 0),
        AtomicRecipe::Zero => (0, 0, 0, 0),
    };
    let retained = matrix_bytes(rows, nnz)?;
    let scratch = scratch_bytes(rows, scratch)?;
    Ok(Bound {
        rows,
        columns,
        nnz,
        topology_entries,
        retained,
        peak: retained.checked_add(scratch).ok_or(BuildError::Overflow)?,
        work: as_u64(candidates)?
            .checked_add(as_u64(nnz)?)
            .ok_or(BuildError::Overflow)?,
    })
}

fn product_bound(after: Bound, before: Bound) -> Result<Bound, BuildError> {
    if before.rows != after.columns {
        return Err(BuildError::Operator(OperatorError::SpaceMismatch));
    }
    let candidates = before
        .nnz
        .checked_mul(after.nnz)
        .ok_or(BuildError::Overflow)?;
    let nnz = after
        .rows
        .checked_mul(before.columns)
        .ok_or(BuildError::Overflow)?
        .min(candidates);
    let retained = matrix_bytes(after.rows, nnz)?;
    let scratch = scratch_bytes(after.rows, candidates)?;
    let peak = before
        .peak
        .max(
            before
                .retained
                .checked_add(after.peak)
                .ok_or(BuildError::Overflow)?,
        )
        .max(
            before
                .retained
                .checked_add(after.retained)
                .and_then(|bytes| bytes.checked_add(retained))
                .and_then(|bytes| bytes.checked_add(scratch))
                .ok_or(BuildError::Overflow)?,
        );
    Ok(Bound {
        rows: after.rows,
        columns: before.columns,
        nnz,
        topology_entries: before
            .topology_entries
            .checked_add(after.topology_entries)
            .ok_or(BuildError::Overflow)?,
        retained,
        peak,
        work: before
            .work
            .checked_add(after.work)
            .and_then(|work| work.checked_add(as_u64(candidates).ok()?))
            .and_then(|work| work.checked_add(as_u64(nnz).ok()?))
            .ok_or(BuildError::Overflow)?,
    })
}

fn build<S: Variance, T: Variance>(
    operator: &LinearOperator<S, T>,
    limit: BuildLimit,
) -> Result<CsrMatrix<Box<[f64]>>, BuildError> {
    let estimate = estimate(operator)?;
    if let Some(error) = limit.rejection(estimate) {
        return Err(error);
    }
    let matrix = match &operator.recipe {
        OperatorRecipe::Atomic(recipe) => {
            build_atomic(operator.source.basis(), operator.target.basis(), recipe)?
        }
        OperatorRecipe::Composite(steps) => {
            let mut source = operator.source.basis();
            let mut steps = steps.iter();
            let first = steps.next().ok_or(BuildError::Overflow)?;
            let mut current = build_atomic(source, &first.target, &first.recipe)?;
            source = &first.target;
            for step in steps {
                let next = build_atomic(source, &step.target, &step.recipe)?;
                current = multiply(&next, &current)?;
                source = &step.target;
            }
            current
        }
    };
    if matrix.pattern().shape() != estimate.shape || matrix.pattern().nnz() > estimate.nnz_bound {
        return Err(BuildError::Overflow);
    }
    Ok(matrix)
}

fn build_atomic(
    source: &Binary64Basis,
    target: &Binary64Basis,
    recipe: &AtomicRecipe,
) -> Result<CsrMatrix<Box<[f64]>>, BuildError> {
    match recipe {
        AtomicRecipe::Differential => differential(source.size(), target),
        AtomicRecipe::Restriction => restriction(source.size(), selected(target)?),
        AtomicRecipe::ExtensionByZero => extension(target.size(), selected(source)?),
        AtomicRecipe::Riesz(realization) => {
            diagonal(realization.hodge_coefficients(degree(source)?)?, false)
        }
        AtomicRecipe::InverseRiesz(realization) => {
            diagonal(realization.hodge_coefficients(degree(source)?)?, true)
        }
        AtomicRecipe::Codifferential(realization) => codifferential(realization, degree(source)?),
        AtomicRecipe::Laplacian(realization) => laplacian(realization, degree(source)?),
        AtomicRecipe::Identity => identity(source.size()),
        AtomicRecipe::Zero => empty(target.size(), source.size()),
    }
}

fn differential(
    source_size: usize,
    target: &Binary64Basis,
) -> Result<CsrMatrix<Box<[f64]>>, BuildError> {
    let target = target.full().ok_or(OperatorError::FullSpaceRequired)?;
    let degree = usize::try_from(target.degree).map_err(|_| OperatorError::DegreeOutside)?;
    if degree == 0 || degree > target.domain.view().dimension() {
        return empty(target.basis_size, source_size);
    }
    let boundary = target.domain.view().boundary(degree)?;
    transpose_boundary(boundary)
}

fn transpose_boundary(boundary: BoundaryRef<'_>) -> Result<CsrMatrix<Box<[f64]>>, BuildError> {
    let (columns, rows) = boundary.shape();
    let mut counts = filled(rows, 0_usize)?;
    for &row in boundary.indices() {
        counts[row] = counts[row].checked_add(1).ok_or(BuildError::Overflow)?;
    }
    let offsets = prefix(&counts)?;
    let mut next = offsets[..rows].to_vec();
    let mut indices = filled(boundary.indices().len(), 0_usize)?;
    let mut values = filled(boundary.indices().len(), 0.0)?;
    for (row, range) in boundary.indptr().windows(2).enumerate() {
        for position in range[0]..range[1] {
            let output = next[boundary.indices()[position]];
            next[boundary.indices()[position]] += 1;
            indices[output] = row;
            values[output] = coefficient(boundary.coefficients(), position);
        }
    }
    matrix((rows, columns), offsets, indices, values)
}

fn restriction(
    source_size: usize,
    selection: &crate::CanonicalSelection,
) -> Result<CsrMatrix<Box<[f64]>>, BuildError> {
    let rows = selection.len();
    let mut offsets = sequence(rows + 1)?;
    let columns = copy_slice(selection.indices())?;
    let values = filled(rows, 1.0)?;
    matrix(
        (rows, source_size),
        std::mem::take(&mut offsets),
        columns,
        values,
    )
}

fn extension(
    target_size: usize,
    selection: &crate::CanonicalSelection,
) -> Result<CsrMatrix<Box<[f64]>>, BuildError> {
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(target_size + 1)
        .map_err(|_| BuildError::Allocation)?;
    let mut columns = Vec::new();
    columns
        .try_reserve_exact(selection.len())
        .map_err(|_| BuildError::Allocation)?;
    offsets.push(0);
    let mut selected = selection.indices().iter().copied().enumerate().peekable();
    for row in 0..target_size {
        if selected.peek().is_some_and(|(_, index)| *index == row) {
            columns.push(selected.next().ok_or(BuildError::Overflow)?.0);
        }
        offsets.push(columns.len());
    }
    matrix(
        (target_size, selection.len()),
        offsets,
        columns,
        filled(selection.len(), 1.0)?,
    )
}

fn diagonal(values: &[f64], inverse: bool) -> Result<CsrMatrix<Box<[f64]>>, BuildError> {
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(values.len())
        .map_err(|_| BuildError::Allocation)?;
    coefficients.extend(
        values
            .iter()
            .map(|&value| if inverse { 1.0 / value } else { value }),
    );
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err(BuildError::Operator(OperatorError::NonFinite));
    }
    matrix(
        (values.len(), values.len()),
        sequence(values.len() + 1)?,
        sequence(values.len())?,
        coefficients,
    )
}

fn identity(rank: usize) -> Result<CsrMatrix<Box<[f64]>>, BuildError> {
    matrix(
        (rank, rank),
        sequence(rank + 1)?,
        sequence(rank)?,
        filled(rank, 1.0)?,
    )
}

fn codifferential(
    realization: &crate::EuclideanRealization,
    degree: usize,
) -> Result<CsrMatrix<Box<[f64]>>, BuildError> {
    let boundary = realization.topology().chain_view().boundary(degree)?;
    let source_weights = realization.hodge_coefficients(degree)?;
    let target_weights = realization.hodge_coefficients(degree - 1)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(boundary.indices().len())
        .map_err(|_| BuildError::Allocation)?;
    values.extend(boundary.exact_entries().map(|(row, column, value)| {
        value.to_f64().expect("i64 has a finite binary64 image") * source_weights[column]
            / target_weights[row]
    }));
    if values.iter().any(|value| !value.is_finite()) {
        return Err(BuildError::Operator(OperatorError::NonFinite));
    }
    matrix(
        boundary.shape(),
        copy_slice(boundary.indptr())?,
        copy_slice(boundary.indices())?,
        values,
    )
}

fn laplacian(
    realization: &crate::EuclideanRealization,
    degree: usize,
) -> Result<CsrMatrix<Box<[f64]>>, BuildError> {
    let owner = realization.topology();
    let rank = owner.chain_view().basis_size(degree)?;
    let weights = realization.hodge_coefficients(degree)?;
    let mut rows = row_maps(rank)?;
    if degree > 0 {
        let boundary = owner.chain_view().boundary(degree)?;
        let lower_weights = realization.hodge_coefficients(degree - 1)?;
        for (lower, range) in boundary.indptr().windows(2).enumerate() {
            for left in range[0]..range[1] {
                let row = boundary.indices()[left];
                let left = coefficient(boundary.coefficients(), left) / lower_weights[lower];
                for right in range[0]..range[1] {
                    let column = boundary.indices()[right];
                    add(
                        &mut rows[row],
                        column,
                        left * coefficient(boundary.coefficients(), right) * weights[column],
                    )?;
                }
            }
        }
    }
    if degree < owner.dimension() {
        let boundary = owner.chain_view().boundary(degree + 1)?;
        let upper_weights = realization.hodge_coefficients(degree + 1)?;
        let mut columns = row_maps(boundary.shape().1)?;
        for (row, column, value) in boundary.exact_entries() {
            columns[column].insert(
                row,
                value.to_f64().expect("i64 has a finite binary64 image"),
            );
        }
        for (upper, faces) in columns.iter().enumerate() {
            for (&row, &left) in faces {
                for (&column, &right) in faces {
                    add(
                        &mut rows[row],
                        column,
                        left * upper_weights[upper] * right / weights[row],
                    )?;
                }
            }
        }
    }
    rows_to_csr(rank, rank, rows)
}

fn multiply(
    after: &CsrMatrix<Box<[f64]>>,
    before: &CsrMatrix<Box<[f64]>>,
) -> Result<CsrMatrix<Box<[f64]>>, BuildError> {
    if before.pattern().shape().0 != after.pattern().shape().1 {
        return Err(BuildError::Operator(OperatorError::SpaceMismatch));
    }
    let rows = after.pattern().shape().0;
    let mut output = row_maps(rows)?;
    for (row, (middle, left)) in after.rows().enumerate() {
        for (&middle, &left) in middle.iter().zip(left) {
            let (columns, right) = before.row(middle).ok_or(BuildError::InvalidStorage)?;
            for (&column, &right) in columns.iter().zip(right) {
                add(&mut output[row], column, left * right)?;
            }
        }
    }
    rows_to_csr(rows, before.pattern().shape().1, output)
}

fn rows_to_csr(
    row_count: usize,
    column_count: usize,
    rows: Vec<BTreeMap<usize, f64>>,
) -> Result<CsrMatrix<Box<[f64]>>, BuildError> {
    let nnz = rows
        .iter()
        .map(|row| row.values().filter(|&&value| value != 0.0).count())
        .sum();
    let mut offsets = Vec::new();
    let mut columns = Vec::new();
    let mut values = Vec::new();
    offsets
        .try_reserve_exact(row_count + 1)
        .map_err(|_| BuildError::Allocation)?;
    columns
        .try_reserve_exact(nnz)
        .map_err(|_| BuildError::Allocation)?;
    values
        .try_reserve_exact(nnz)
        .map_err(|_| BuildError::Allocation)?;
    offsets.push(0);
    for row in rows {
        for (column, value) in row {
            if value != 0.0 {
                columns.push(column);
                values.push(value);
            }
        }
        offsets.push(columns.len());
    }
    matrix((row_count, column_count), offsets, columns, values)
}

fn add(row: &mut BTreeMap<usize, f64>, column: usize, value: f64) -> Result<(), BuildError> {
    let value = row.get(&column).copied().unwrap_or(0.0) + value;
    if !value.is_finite() {
        return Err(BuildError::Operator(OperatorError::NonFinite));
    }
    row.insert(column, value);
    Ok(())
}

fn empty(rows: usize, columns: usize) -> Result<CsrMatrix<Box<[f64]>>, BuildError> {
    matrix(
        (rows, columns),
        filled(rows.checked_add(1).ok_or(BuildError::Overflow)?, 0)?,
        Vec::new(),
        Vec::new(),
    )
}

fn matrix(
    shape: (usize, usize),
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
) -> Result<CsrMatrix<Box<[f64]>>, BuildError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(BuildError::Operator(OperatorError::NonFinite));
    }
    Ok(CsrMatrix::try_from_parts(shape, offsets, columns, values)?)
}

fn selected(endpoint: &Binary64Basis) -> Result<&crate::CanonicalSelection, BuildError> {
    let Binary64Basis::Selected(selection) = endpoint else {
        return Err(BuildError::Operator(OperatorError::SpaceMismatch));
    };
    Ok(selection)
}

fn boundary(
    target: &crate::chain::BasedDegree,
    degree: usize,
) -> Result<BoundaryRef<'_>, BuildError> {
    if degree == 0 || degree > target.domain.view().dimension() {
        return Err(BuildError::Operator(OperatorError::DegreeOutside));
    }
    Ok(target.domain.view().boundary(degree)?)
}

fn laplacian_counts(
    realization: &crate::EuclideanRealization,
    degree: usize,
) -> Result<(u64, usize, usize), BuildError> {
    let owner = realization.topology();
    let mut topology = 0_u64;
    let mut candidates = 0_usize;
    let mut scratch = 0_usize;
    if degree > 0 {
        let boundary = owner.chain_view().boundary(degree)?;
        topology = topology
            .checked_add(as_u64(boundary.indices().len())?)
            .ok_or(BuildError::Overflow)?;
        candidates = boundary.indptr().windows(2).try_fold(0_usize, |sum, row| {
            let width = row[1] - row[0];
            sum.checked_add(width.checked_mul(width).ok_or(BuildError::Overflow)?)
                .ok_or(BuildError::Overflow)
        })?;
    }
    if degree < owner.dimension() {
        let boundary = owner.chain_view().boundary(degree + 1)?;
        let entries = boundary.indices().len();
        topology = topology
            .checked_add(as_u64(entries)?)
            .ok_or(BuildError::Overflow)?;
        let width = degree.checked_add(2).ok_or(BuildError::Overflow)?;
        let upper = boundary.shape().1;
        candidates = candidates
            .checked_add(
                upper
                    .checked_mul(width.checked_mul(width).ok_or(BuildError::Overflow)?)
                    .ok_or(BuildError::Overflow)?,
            )
            .ok_or(BuildError::Overflow)?;
        scratch = entries;
    }
    Ok((topology, candidates, scratch))
}

fn matrix_bytes(rows: usize, nnz: usize) -> Result<u64, BuildError> {
    let offsets = rows.checked_add(1).ok_or(BuildError::Overflow)?;
    let bytes = offsets
        .checked_mul(size_of::<usize>())
        .and_then(|bytes| {
            nnz.checked_mul(size_of::<usize>() + size_of::<f64>())
                .and_then(|entries| bytes.checked_add(entries))
        })
        .ok_or(BuildError::Overflow)?;
    as_u64(bytes)
}

fn scratch_bytes(rows: usize, entries: usize) -> Result<u64, BuildError> {
    let bytes = rows
        .checked_mul(size_of::<BTreeMap<usize, f64>>())
        .and_then(|bytes| {
            entries
                .checked_mul(2 * size_of::<usize>() + size_of::<f64>())
                .and_then(|entries| bytes.checked_add(entries))
        })
        .ok_or(BuildError::Overflow)?;
    as_u64(bytes)
}

fn prefix(counts: &[usize]) -> Result<Vec<usize>, BuildError> {
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(counts.len() + 1)
        .map_err(|_| BuildError::Allocation)?;
    offsets.push(0_usize);
    for &count in counts {
        offsets.push(
            offsets
                .last()
                .copied()
                .ok_or(BuildError::Overflow)?
                .checked_add(count)
                .ok_or(BuildError::Overflow)?,
        );
    }
    Ok(offsets)
}

fn row_maps(count: usize) -> Result<Vec<BTreeMap<usize, f64>>, BuildError> {
    filled(count, BTreeMap::new())
}

fn filled<T: Clone>(count: usize, value: T) -> Result<Vec<T>, BuildError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|_| BuildError::Allocation)?;
    output.resize(count, value);
    Ok(output)
}

fn sequence(count: usize) -> Result<Vec<usize>, BuildError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|_| BuildError::Allocation)?;
    output.extend(0..count);
    Ok(output)
}

fn copy_slice<T: Copy>(values: &[T]) -> Result<Vec<T>, BuildError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| BuildError::Allocation)?;
    output.extend_from_slice(values);
    Ok(output)
}

fn coefficient(coefficients: CoefficientSlice<'_>, position: usize) -> f64 {
    match coefficients {
        CoefficientSlice::I8(values) => f64::from(values[position]),
        CoefficientSlice::I64(values) => values[position]
            .to_f64()
            .expect("i64 has a finite binary64 image"),
    }
}

fn as_u64(value: usize) -> Result<u64, BuildError> {
    u64::try_from(value).map_err(|_| BuildError::Overflow)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use num_traits::ToPrimitive;

    use crate::operator::compose;
    use crate::{
        Binary64Element, Binary64Space, CandidateInput, Cochain, ComplexCore, EuclideanRealization,
        LinearOperator, NondegenerateCapability, PairingCapability, PositiveMetric,
        RealizationLimit, Variance, WorkLimit,
    };

    use super::{BuildError, BuildLimit, build, estimate};

    fn triangle() -> Arc<ComplexCore> {
        ComplexCore::admit(CandidateInput::signed([0, 1, 2], 1, 3, None).unwrap()).unwrap()
    }

    fn metric() -> PositiveMetric {
        let height = 3.0_f64.sqrt() / 2.0;
        EuclideanRealization::admit(
            triangle(),
            2,
            vec![0.0, 0.0, 1.0, 0.0, 0.5, height],
            RealizationLimit::DEFAULT,
        )
        .unwrap()
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap()
    }

    fn assert_equivalent<S: Variance, T: Variance>(
        operator: &LinearOperator<S, T>,
        coefficients: Vec<f64>,
    ) {
        let value = Binary64Element::admit(operator.source().clone(), coefficients).unwrap();
        let estimate = estimate(operator).unwrap();
        let matrix = build(operator, BuildLimit::for_estimate(estimate)).unwrap();
        let actual = apply(&matrix, value.coefficients());
        let expected = operator.apply(&value).unwrap();
        let scale = value
            .coefficients()
            .iter()
            .chain(expected.coefficients())
            .chain(&actual)
            .map(|value| value.abs())
            .fold(1.0, f64::max);
        let tolerance = f64::EPSILON
            * scale
            * estimate
                .scalar_steps_bound
                .max(1)
                .to_f64()
                .expect("u64 has a finite binary64 image")
            * 8.0;
        assert_eq!(actual.len(), expected.coefficients().len());
        for (&left, &right) in actual.iter().zip(expected.coefficients()) {
            assert!((left - right).abs() <= tolerance, "{left} != {right}");
        }
    }

    fn apply(matrix: &crate::sparse::CsrMatrix<Box<[f64]>>, input: &[f64]) -> Vec<f64> {
        matrix
            .rows()
            .map(|(columns, coefficients)| {
                columns
                    .iter()
                    .zip(coefficients)
                    .map(|(&column, &coefficient)| coefficient * input[column])
                    .sum()
            })
            .collect()
    }

    #[test]
    fn atomic_csr_formulas_match_every_direct_operation() {
        let owner = triangle();
        let space = Binary64Space::full(Arc::clone(&owner), 0).unwrap();
        assert_equivalent(&space.exterior_derivative().unwrap(), vec![2.0, 5.0, 11.0]);
        assert_equivalent(&space.identity(), vec![2.0, 5.0, 11.0]);
        assert_equivalent(&space.zero_to(&space), vec![2.0, 5.0, 11.0]);

        let selection = Arc::new(owner.selection(0, vec![0, 2]).unwrap());
        assert_equivalent(
            &selection.restriction::<Cochain>().unwrap(),
            vec![2.0, 5.0, 11.0],
        );
        assert_equivalent(
            &selection.extension_by_zero::<Cochain>().unwrap(),
            vec![2.0, 11.0],
        );

        let metric = metric();
        assert_equivalent(&metric.riesz(1).unwrap(), vec![3.0, 5.0, -1.0]);
        assert_equivalent(&metric.inverse_riesz(1).unwrap(), vec![3.0, 5.0, -1.0]);
        assert_equivalent(&metric.codifferential(1).unwrap(), vec![3.0, 5.0, -1.0]);
        assert_equivalent(&metric.laplacian(0).unwrap(), vec![2.0, 5.0, 11.0]);
        assert_equivalent(&metric.laplacian(1).unwrap(), vec![3.0, 5.0, -1.0]);
    }

    #[test]
    fn sparse_composition_is_bounded_and_matches_flat_direct_plans() {
        let owner = triangle();
        let selection = Arc::new(owner.selection(0, vec![0, 2]).unwrap());
        let restriction = selection.restriction::<Cochain>().unwrap();
        let extension = selection.extension_by_zero::<Cochain>().unwrap();
        let cycle = compose(&extension, &restriction).unwrap();
        assert_eq!(restriction.execution_steps(), 1);
        assert_equivalent(&restriction, vec![2.0, 5.0, 11.0]);
        let mut plan = cycle.clone();
        for length in [4, 16, 64] {
            while plan.execution_steps() < length {
                plan = compose(&cycle, &plan).unwrap();
            }
            assert_eq!(plan.execution_steps(), length);
            assert_equivalent(&plan, vec![2.0, 5.0, 11.0]);
        }
    }

    #[test]
    fn build_rejects_each_resource_axis_before_publication() {
        let operator = Binary64Space::full(triangle(), 0)
            .unwrap()
            .exterior_derivative()
            .unwrap();
        let estimate = estimate(&operator).unwrap();
        let exact = BuildLimit::for_estimate(estimate);
        let storage = exact.storage;
        let retained = BuildLimit {
            storage: crate::StorageLimit::new(
                estimate.retained_logical_bytes_bound - 1,
                storage.peak_live_logical_bytes(),
            )
            .unwrap(),
            work: exact.work,
        };
        let peak = BuildLimit {
            storage: crate::StorageLimit::new(
                storage.retained_logical_bytes(),
                estimate.peak_live_logical_bytes_bound - 1,
            )
            .unwrap(),
            work: exact.work,
        };
        let work = BuildLimit {
            storage,
            work: WorkLimit::new(estimate.scalar_steps_bound - 1),
        };
        assert!(matches!(
            build(&operator, retained),
            Err(BuildError::Resource {
                axis: "retained_logical_bytes",
                ..
            })
        ));
        assert!(matches!(
            build(&operator, peak),
            Err(BuildError::Resource {
                axis: "peak_live_logical_bytes",
                ..
            })
        ));
        assert!(matches!(
            build(&operator, work),
            Err(BuildError::Resource {
                axis: "scalar_steps",
                ..
            })
        ));
        assert!(build(&operator, exact).is_ok());
    }

    #[test]
    fn empty_and_exact_dyadic_edges_preserve_mathematical_zero() {
        let owner = triangle();
        let top = Binary64Space::full(Arc::clone(&owner), 2).unwrap();
        assert_equivalent(&top.exterior_derivative().unwrap(), vec![8.0]);

        let empty = Arc::new(owner.selection(1, Vec::new()).unwrap());
        assert_equivalent(
            &empty.restriction::<Cochain>().unwrap(),
            vec![1.0, 2.0, 4.0],
        );
        assert_equivalent(&empty.extension_by_zero::<Cochain>().unwrap(), Vec::new());

        let zero_space = Binary64Space::full(owner, 0).unwrap();
        let first = zero_space.exterior_derivative().unwrap();
        let square = compose(&first.target().exterior_derivative().unwrap(), &first).unwrap();
        let dyadic = vec![2.0_f64.powi(40), -2.0_f64.powi(-40), 3.0];
        let value = Binary64Element::admit(square.source().clone(), dyadic).unwrap();
        assert_equivalent(&square, value.coefficients().to_vec());
        assert_eq!(
            apply(
                &build(
                    &square,
                    BuildLimit::for_estimate(estimate(&square).unwrap())
                )
                .unwrap(),
                value.coefficients(),
            ),
            vec![0.0]
        );
    }
}
