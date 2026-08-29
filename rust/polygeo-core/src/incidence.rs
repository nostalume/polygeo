use crate::ExactRational;
use crate::TopologyError;
use crate::complex::ComplexCore;
use crate::halfedge::HalfedgeSurfaceCore;
use crate::sparse::CsrMatrix;
use num_traits::{ToPrimitive, Zero};

#[derive(Debug)]
pub struct Basis {
    values: Box<[usize]>,
    row_count: usize,
    row_width: usize,
}

impl Basis {
    pub(crate) fn from_flat(
        values: Vec<usize>,
        row_count: usize,
        row_width: usize,
    ) -> Result<Self, TopologyError> {
        let expected = row_count
            .checked_mul(row_width)
            .ok_or(TopologyError::CountOverflow)?;
        if values.len() != expected {
            return Err(TopologyError::InternalInvariant);
        }
        Ok(Self {
            values: values.into_boxed_slice(),
            row_count,
            row_width,
        })
    }

    /// Number of canonical basis rows.
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.row_count
    }

    /// Width of each canonical simplex row.
    #[must_use]
    pub const fn row_width(&self) -> usize {
        self.row_width
    }

    /// One canonical basis row.
    #[must_use]
    pub fn row(&self, index: usize) -> Option<&[usize]> {
        let start = index.checked_mul(self.row_width)?;
        let end = start.checked_add(self.row_width)?;
        self.values.get(start..end)
    }

    /// Flat row-major canonical values.
    #[must_use]
    pub fn values(&self) -> &[usize] {
        &self.values
    }

    pub(crate) fn binary_search(&self, target: &[usize]) -> Result<usize, TopologyError> {
        if target.len() != self.row_width {
            return Err(TopologyError::InternalInvariant);
        }
        let mut lower = 0_usize;
        let mut upper = self.row_count;
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            let row = self.row(middle).ok_or(TopologyError::InternalInvariant)?;
            match row.cmp(target) {
                std::cmp::Ordering::Less => lower = middle + 1,
                std::cmp::Ordering::Greater => upper = middle,
                std::cmp::Ordering::Equal => return Ok(middle),
            }
        }
        Err(TopologyError::InternalInvariant)
    }
}

#[derive(Debug)]
pub(crate) struct CanonicalMaximal {
    pub(crate) vertex_count: usize,
    pub(crate) rows: Basis,
    pub(crate) signs: Box<[i8]>,
}

#[derive(Debug)]
struct FixedRowBuilder {
    width: usize,
    values: Vec<usize>,
    row_count: usize,
}

impl FixedRowBuilder {
    fn try_with_capacity(width: usize, rows: usize) -> Result<Self, TopologyError> {
        let value_count = width
            .checked_mul(rows)
            .ok_or(TopologyError::CountOverflow)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(value_count)
            .map_err(|_| TopologyError::Allocation)?;
        Ok(Self {
            width,
            values,
            row_count: 0,
        })
    }

    fn push(&mut self, row: &[usize]) -> Result<(), TopologyError> {
        if row.len() != self.width {
            return Err(TopologyError::InternalInvariant);
        }
        self.row_count = self
            .row_count
            .checked_add(1)
            .ok_or(TopologyError::CountOverflow)?;
        self.values.extend_from_slice(row);
        Ok(())
    }

    fn finish_sorted_unique(self) -> Result<Basis, TopologyError> {
        let mut order = (0..self.row_count).collect::<Vec<_>>();
        order.sort_unstable_by(|left, right| {
            let left_start = left * self.width;
            let right_start = right * self.width;
            self.values[left_start..left_start + self.width]
                .cmp(&self.values[right_start..right_start + self.width])
        });

        let mut unique_values = Vec::new();
        unique_values
            .try_reserve_exact(self.values.len())
            .map_err(|_| TopologyError::Allocation)?;
        let mut unique_rows = 0_usize;
        let mut previous: Option<usize> = None;
        for row_index in order {
            let start = row_index * self.width;
            let row = &self.values[start..start + self.width];
            let is_duplicate = previous.is_some_and(|previous_index| {
                let previous_start = previous_index * self.width;
                self.values[previous_start..previous_start + self.width] == *row
            });
            if !is_duplicate {
                unique_values.extend_from_slice(row);
                unique_rows = unique_rows
                    .checked_add(1)
                    .ok_or(TopologyError::CountOverflow)?;
                previous = Some(row_index);
            }
        }
        Basis::from_flat(unique_values, unique_rows, self.width)
    }
}

/// Retained canonical signed boundary matrix.
#[derive(Debug)]
pub struct CanonicalBoundary {
    degree: usize,
    pub(crate) storage: CsrMatrix<Box<[i8]>>,
}

impl CanonicalBoundary {
    /// Represented boundary degree.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Matrix shape `(lower_basis_size, source_basis_size)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.storage.pattern().shape()
    }

    /// Canonical nonzero coefficients.
    #[must_use]
    pub fn data(&self) -> &[i8] {
        self.storage.values()
    }

    /// Canonical compressed-column indexes within each CSR row.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        self.storage.pattern().column_indices()
    }

    /// Canonical CSR row-pointer storage.
    #[must_use]
    pub fn indptr(&self) -> &[usize] {
        self.storage.pattern().row_offsets()
    }
}

/// Borrowed coefficients in their retained native integer width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoefficientSlice<'a> {
    I8(&'a [i8]),
    I64(&'a [i64]),
}

#[derive(Debug)]
pub(crate) struct NativeBoundary64 {
    degree: usize,
    storage: CsrMatrix<Box<[i64]>>,
}

impl NativeBoundary64 {
    pub(crate) fn try_from_csr(
        degree: usize,
        shape: (usize, usize),
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<i64>,
    ) -> Result<Self, TopologyError> {
        let storage = boundary_storage(shape, indptr, indices, data)?;
        Ok(Self { degree, storage })
    }

    pub(crate) fn exact_entries(&self) -> ExactEntries<'_> {
        BoundaryRef::i64(self).exact_entries()
    }
}

#[derive(Debug)]
pub(crate) struct SurfaceChain {
    boundaries: [NativeBoundary64; 3],
}

impl SurfaceChain {
    pub(crate) const fn new(boundaries: [NativeBoundary64; 3]) -> Self {
        Self { boundaries }
    }

    fn boundary(&self, degree: usize) -> Result<&NativeBoundary64, TopologyError> {
        self.boundaries
            .get(degree)
            .ok_or(TopologyError::degree_outside(degree))
    }
}

#[derive(Debug, Clone, Copy)]
enum ChainOwner<'a> {
    Simplicial(&'a ComplexCore),
    Halfedge(&'a HalfedgeSurfaceCore),
}

/// Sealed borrowed view of one topology owner's exact based chain complex.
#[derive(Debug, Clone, Copy)]
pub struct ChainView<'a> {
    owner: ChainOwner<'a>,
}

impl<'a> ChainView<'a> {
    pub(crate) const fn simplicial(owner: &'a ComplexCore) -> Self {
        Self {
            owner: ChainOwner::Simplicial(owner),
        }
    }

    pub(crate) const fn halfedge(owner: &'a HalfedgeSurfaceCore) -> Self {
        Self {
            owner: ChainOwner::Halfedge(owner),
        }
    }

    /// Maximum represented chain degree.
    #[must_use]
    pub const fn dimension(self) -> usize {
        match self.owner {
            ChainOwner::Simplicial(owner) => owner.dimension,
            ChainOwner::Halfedge(_) => 2,
        }
    }

    /// Rank of the ordered basis in one degree.
    ///
    /// # Errors
    ///
    /// Returns `degree_outside` when the degree is not represented.
    pub fn basis_size(self, degree: usize) -> Result<usize, TopologyError> {
        match self.owner {
            ChainOwner::Simplicial(owner) => owner.layout.count(degree),
            ChainOwner::Halfedge(owner) => match degree {
                0 => Ok(owner.vertex_count()),
                1 => Ok(owner.edge_count()),
                2 => Ok(owner.material_face_count()),
                _ => Err(TopologyError::degree_outside(degree)),
            },
        }
    }

    /// Borrow one retained exact boundary map.
    ///
    /// # Errors
    ///
    /// Returns `degree_outside` when the degree is not represented.
    pub fn boundary(self, degree: usize) -> Result<BoundaryRef<'a>, TopologyError> {
        match self.owner {
            ChainOwner::Simplicial(owner) => owner
                .boundaries
                .get(degree)
                .map(BoundaryRef::i8)
                .ok_or(TopologyError::degree_outside(degree)),
            ChainOwner::Halfedge(owner) => owner.chain.boundary(degree).map(BoundaryRef::i64),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum BoundaryStorageRef<'a> {
    I8(&'a CanonicalBoundary),
    I64(&'a NativeBoundary64),
}

/// Sealed borrowed sparse boundary with native-width storage.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryRef<'a> {
    storage: BoundaryStorageRef<'a>,
}

impl<'a> BoundaryRef<'a> {
    const fn i8(boundary: &'a CanonicalBoundary) -> Self {
        Self {
            storage: BoundaryStorageRef::I8(boundary),
        }
    }

    const fn i64(boundary: &'a NativeBoundary64) -> Self {
        Self {
            storage: BoundaryStorageRef::I64(boundary),
        }
    }

    /// Represented boundary degree.
    #[must_use]
    pub const fn degree(self) -> usize {
        match self.storage {
            BoundaryStorageRef::I8(boundary) => boundary.degree,
            BoundaryStorageRef::I64(boundary) => boundary.degree,
        }
    }

    /// Matrix shape `(lower_basis_size, source_basis_size)`.
    #[must_use]
    pub fn shape(self) -> (usize, usize) {
        match self.storage {
            BoundaryStorageRef::I8(boundary) => boundary.storage.pattern().shape(),
            BoundaryStorageRef::I64(boundary) => boundary.storage.pattern().shape(),
        }
    }

    /// Retained coefficients without widening or copying.
    #[must_use]
    pub fn coefficients(self) -> CoefficientSlice<'a> {
        match self.storage {
            BoundaryStorageRef::I8(boundary) => CoefficientSlice::I8(boundary.storage.values()),
            BoundaryStorageRef::I64(boundary) => CoefficientSlice::I64(boundary.storage.values()),
        }
    }

    /// Retained CSR column indices.
    #[must_use]
    pub fn indices(self) -> &'a [usize] {
        match self.storage {
            BoundaryStorageRef::I8(boundary) => boundary.storage.pattern().column_indices(),
            BoundaryStorageRef::I64(boundary) => boundary.storage.pattern().column_indices(),
        }
    }

    /// CSR row pointers, borrowed for every retained proper matrix.
    #[must_use]
    pub fn indptr(self) -> &'a [usize] {
        match self.storage {
            BoundaryStorageRef::I8(boundary) => boundary.storage.pattern().row_offsets(),
            BoundaryStorageRef::I64(boundary) => boundary.storage.pattern().row_offsets(),
        }
    }

    /// Iterate `(row, column, widened_i64)` without allocating.
    #[must_use]
    pub fn exact_entries(self) -> ExactEntries<'a> {
        ExactEntries {
            shape: self.shape(),
            indptr: self.indptr(),
            indices: self.indices(),
            coefficients: self.coefficients(),
            row: 0,
            position: 0,
        }
    }

    pub(crate) fn apply_transpose_binary64(
        self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), TopologyError> {
        if (input.len(), output.len()) != self.shape() {
            return Err(TopologyError::InternalInvariant);
        }
        match self.storage {
            BoundaryStorageRef::I8(boundary) => apply_transpose_rows(
                boundary.storage.pattern().row_offsets(),
                boundary.storage.pattern().column_indices(),
                boundary.storage.values(),
                input,
                output,
                f64::from,
            ),
            BoundaryStorageRef::I64(boundary) => apply_transpose_rows(
                boundary.storage.pattern().row_offsets(),
                boundary.storage.pattern().column_indices(),
                boundary.storage.values(),
                input,
                output,
                binary64_from_i64_rounded,
            ),
        }
        Ok(())
    }
}

fn binary64_from_i64_rounded(value: i64) -> f64 {
    value
        .to_f64()
        .expect("every i64 has a finite rounded binary64 representation")
}

fn apply_transpose_rows<T: Copy>(
    row_offsets: &[usize],
    column_indices: &[usize],
    values: &[T],
    input: &[f64],
    output: &mut [f64],
    binary64: impl Fn(T) -> f64,
) {
    for (row, offsets) in row_offsets.windows(2).enumerate() {
        for position in offsets[0]..offsets[1] {
            output[column_indices[position]] += binary64(values[position]) * input[row];
        }
    }
}

fn boundary_storage<T>(
    shape: (usize, usize),
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<T>,
) -> Result<CsrMatrix<Box<[T]>>, TopologyError> {
    CsrMatrix::try_from_parts(shape, row_offsets, column_indices, values)
        .map_err(|_| TopologyError::InternalInvariant)
}

/// Allocation-free exact iterator over one retained sparse boundary.
#[derive(Clone)]
pub struct ExactEntries<'a> {
    shape: (usize, usize),
    indptr: &'a [usize],
    indices: &'a [usize],
    coefficients: CoefficientSlice<'a>,
    row: usize,
    position: usize,
}

impl Iterator for ExactEntries<'_> {
    type Item = (usize, usize, i64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.position == self.indices.len() {
            return None;
        }
        while self.row < self.shape.0 && self.position >= self.indptr[self.row + 1] {
            self.row += 1;
        }
        let coefficient = match self.coefficients {
            CoefficientSlice::I8(values) => i64::from(values[self.position]),
            CoefficientSlice::I64(values) => values[self.position],
        };
        let entry = (self.row, self.indices[self.position], coefficient);
        self.position += 1;
        Some(entry)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.indices.len() - self.position;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ExactEntries<'_> {}

#[derive(Clone, Copy)]
pub(crate) enum IncidenceAxis {
    Rows,
    Columns,
}

/// Select a deterministic exact independent subset of incidence rows or columns.
pub(crate) fn independent_incidence(
    boundary: BoundaryRef<'_>,
    axis: IncidenceAxis,
) -> Result<Box<[usize]>, TopologyError> {
    let (source_rows, source_columns) = boundary.shape();
    let (rows, columns) = match axis {
        IncidenceAxis::Rows => (source_columns, source_rows),
        IncidenceAxis::Columns => (source_rows, source_columns),
    };
    let cells = rows
        .checked_mul(columns)
        .ok_or(TopologyError::CountOverflow)?;
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(cells)
        .map_err(|_| TopologyError::Allocation)?;
    matrix.resize(cells, ExactRational::zero());
    for (row, column, value) in boundary.exact_entries() {
        let (row, column) = match axis {
            IncidenceAxis::Rows => (column, row),
            IncidenceAxis::Columns => (row, column),
        };
        matrix[row * columns + column] = ExactRational::from_integer(value.into());
    }
    let mut pivots = Vec::new();
    pivots
        .try_reserve_exact(rows.min(columns))
        .map_err(|_| TopologyError::Allocation)?;
    let mut pivot_row = 0;
    for column in 0..columns {
        let Some(selected) =
            (pivot_row..rows).find(|&row| !matrix[row * columns + column].is_zero())
        else {
            continue;
        };
        if selected != pivot_row {
            for index in 0..columns {
                matrix.swap(selected * columns + index, pivot_row * columns + index);
            }
        }
        let pivot = matrix[pivot_row * columns + column].clone();
        for index in column..columns {
            matrix[pivot_row * columns + index] /= &pivot;
        }
        for row in pivot_row + 1..rows {
            let factor = matrix[row * columns + column].clone();
            if factor.is_zero() {
                continue;
            }
            for index in column..columns {
                let pivot_value = matrix[pivot_row * columns + index].clone();
                matrix[row * columns + index] -= factor.clone() * pivot_value;
            }
        }
        pivots.push(column);
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }
    Ok(pivots.into_boxed_slice())
}

/// Retained immutable indexing for all represented simplex degrees.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DegreeLayout {
    counts: Box<[usize]>,
    word_offsets: Box<[usize]>,
}

impl DegreeLayout {
    pub(crate) fn from_bases(bases: &[Basis]) -> Result<Self, TopologyError> {
        let mut counts = Vec::new();
        counts
            .try_reserve_exact(bases.len())
            .map_err(|_| TopologyError::Allocation)?;
        counts.extend(bases.iter().map(Basis::row_count));
        Self::from_owned_counts(counts)
    }

    #[cfg(test)]
    pub(crate) fn from_counts(counts: &[usize]) -> Result<Self, TopologyError> {
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(counts.len())
            .map_err(|_| TopologyError::Allocation)?;
        owned.extend_from_slice(counts);
        Self::from_owned_counts(owned)
    }

    fn from_owned_counts(counts: Vec<usize>) -> Result<Self, TopologyError> {
        #[cfg(test)]
        LAYOUT_BUILDS.with(|builds| builds.set(builds.get() + 1));
        let offset_count = counts
            .len()
            .checked_add(1)
            .ok_or(TopologyError::CountOverflow)?;
        let mut word_offsets = Vec::new();
        word_offsets
            .try_reserve_exact(offset_count)
            .map_err(|_| TopologyError::Allocation)?;
        word_offsets.push(0_usize);
        for count in counts.iter().copied() {
            word_offsets.push(
                word_offsets
                    .last()
                    .copied()
                    .ok_or(TopologyError::InternalInvariant)?
                    .checked_add(count.div_ceil(u64::BITS as usize))
                    .ok_or(TopologyError::CountOverflow)?,
            );
        }
        Ok(Self {
            counts: counts.into_boxed_slice(),
            word_offsets: word_offsets.into_boxed_slice(),
        })
    }

    pub(crate) fn counts(&self) -> &[usize] {
        &self.counts
    }

    pub(crate) fn count(&self, degree: usize) -> Result<usize, TopologyError> {
        self.counts
            .get(degree)
            .copied()
            .ok_or(TopologyError::degree_outside(degree))
    }

    pub(crate) fn word_offsets(&self) -> &[usize] {
        &self.word_offsets
    }
}

#[cfg(test)]
thread_local! {
    pub(crate) static LAYOUT_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(crate) fn try_filled<T: Clone>(length: usize, value: T) -> Result<Vec<T>, TopologyError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| TopologyError::Allocation)?;
    output.resize(length, value);
    Ok(output)
}

/// Private fallible union-find scratch shared by topology admissions.
pub(crate) struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    pub(crate) fn try_new(length: usize) -> Result<Self, TopologyError> {
        let mut parent = Vec::new();
        parent
            .try_reserve_exact(length)
            .map_err(|_| TopologyError::Allocation)?;
        parent.extend(0..length);
        Ok(Self { parent })
    }

    pub(crate) fn join(&mut self, left: usize, right: usize) -> bool {
        let left = self.root(left);
        let right = self.root(right);
        if left == right {
            return false;
        }
        self.parent[right] = left;
        true
    }

    pub(crate) fn is_root(&mut self, value: usize) -> bool {
        self.root(value) == value
    }

    fn root(&mut self, mut value: usize) -> usize {
        while self.parent[value] != value {
            self.parent[value] = self.parent[self.parent[value]];
            value = self.parent[value];
        }
        value
    }
}

pub(crate) fn validated_permutation_sign(row: &[usize]) -> Result<i8, TopologyError> {
    let mut inversions = 0_usize;
    for left in 0..row.len() {
        for right in (left + 1)..row.len() {
            if row[left] == row[right] {
                return Err(TopologyError::repeated_vertex(row[left]));
            }
            inversions += usize::from(row[left] > row[right]);
        }
    }
    Ok(if inversions.is_multiple_of(2) { 1 } else { -1 })
}

fn checked_binomial(total: usize, selected: usize) -> Option<usize> {
    if selected > total {
        return Some(0);
    }
    let selected = selected.min(total - selected);
    let mut result = 1_usize;
    for step in 0..selected {
        result = result.checked_mul(total - step)?;
        result /= step + 1;
    }
    Some(result)
}

pub(crate) fn canonical_faces(maximal: &Basis, selected: usize) -> Result<Basis, TopologyError> {
    let per_row =
        checked_binomial(maximal.row_width(), selected).ok_or(TopologyError::CountOverflow)?;
    let upper_bound = per_row
        .checked_mul(maximal.row_count())
        .ok_or(TopologyError::CountOverflow)?;
    let mut rows = FixedRowBuilder::try_with_capacity(selected, upper_bound)?;
    let mut selected_values = Vec::new();
    selected_values
        .try_reserve_exact(selected)
        .map_err(|_| TopologyError::Allocation)?;
    for row_index in 0..maximal.row_count() {
        let row = maximal
            .row(row_index)
            .ok_or(TopologyError::InternalInvariant)?;
        selected_values.clear();
        collect_combinations(row, selected, 0, &mut selected_values, &mut rows)?;
    }
    rows.finish_sorted_unique()
}

fn collect_combinations(
    row: &[usize],
    selected: usize,
    start: usize,
    current: &mut Vec<usize>,
    output: &mut FixedRowBuilder,
) -> Result<(), TopologyError> {
    if current.len() == selected {
        return output.push(current);
    }
    let remaining = selected - current.len();
    let final_start = row
        .len()
        .checked_sub(remaining)
        .ok_or(TopologyError::InternalInvariant)?;
    for index in start..=final_start {
        current.push(row[index]);
        collect_combinations(row, selected, index + 1, current, output)?;
        current.pop();
    }
    Ok(())
}

type BoundaryBuild = (Vec<CanonicalBoundary>, Vec<Box<[usize]>>);

pub(crate) fn build_boundaries(
    bases: &[Basis],
    orientations: &[Box<[i8]>],
) -> Result<BoundaryBuild, TopologyError> {
    let mut boundaries = Vec::new();
    let mut immediate_faces = Vec::new();
    boundaries
        .try_reserve_exact(bases.len())
        .map_err(|_| TopologyError::Allocation)?;
    immediate_faces
        .try_reserve_exact(bases.len())
        .map_err(|_| TopologyError::Allocation)?;

    for degree in 0..bases.len() {
        let columns = bases[degree].row_count();
        if degree == 0 {
            boundaries.push(CanonicalBoundary {
                degree,
                storage: boundary_storage((0, columns), vec![0], Vec::new(), Vec::new())?,
            });
            immediate_faces.push(Box::default());
            continue;
        }

        let lower = &bases[degree - 1];
        let nonzeros = columns
            .checked_mul(degree + 1)
            .ok_or(TopologyError::CountOverflow)?;
        let mut faces = Vec::new();
        faces
            .try_reserve_exact(nonzeros)
            .map_err(|_| TopologyError::Allocation)?;
        let mut face = Vec::new();
        face.try_reserve_exact(degree)
            .map_err(|_| TopologyError::Allocation)?;

        for column in 0..columns {
            let simplex = bases[degree]
                .row(column)
                .ok_or(TopologyError::InternalInvariant)?;
            for removed in 0..=degree {
                face.clear();
                face.extend_from_slice(&simplex[..removed]);
                face.extend_from_slice(&simplex[(removed + 1)..]);
                let row = lower.binary_search(&face)?;
                faces.push(row);
            }
        }

        let mut row_counts = Vec::new();
        row_counts
            .try_reserve_exact(lower.row_count())
            .map_err(|_| TopologyError::Allocation)?;
        row_counts.resize(lower.row_count(), 0_usize);
        for row in faces.iter().copied() {
            row_counts[row] = row_counts[row]
                .checked_add(1)
                .ok_or(TopologyError::CountOverflow)?;
        }

        let mut indptr = Vec::new();
        let indptr_length = lower
            .row_count()
            .checked_add(1)
            .ok_or(TopologyError::CountOverflow)?;
        indptr
            .try_reserve_exact(indptr_length)
            .map_err(|_| TopologyError::Allocation)?;
        indptr.push(0_usize);
        for count in row_counts {
            let next = indptr
                .last()
                .copied()
                .ok_or(TopologyError::InternalInvariant)?
                .checked_add(count)
                .ok_or(TopologyError::CountOverflow)?;
            indptr.push(next);
        }

        let mut cursors = Vec::new();
        cursors
            .try_reserve_exact(lower.row_count())
            .map_err(|_| TopologyError::Allocation)?;
        cursors.extend_from_slice(&indptr[..lower.row_count()]);
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(nonzeros)
            .map_err(|_| TopologyError::Allocation)?;
        indices.resize(nonzeros, 0_usize);
        let mut data = Vec::new();
        data.try_reserve_exact(nonzeros)
            .map_err(|_| TopologyError::Allocation)?;
        data.resize(nonzeros, 0_i8);

        for (incidence, row) in faces.iter().copied().enumerate() {
            let column = incidence / (degree + 1);
            let removed = incidence % (degree + 1);
            let position = cursors[row];
            indices[position] = column;
            let alternating = if removed.is_multiple_of(2) { 1 } else { -1 };
            data[position] = orientations[degree][column] * alternating;
            cursors[row] = position
                .checked_add(1)
                .ok_or(TopologyError::CountOverflow)?;
        }
        let storage = boundary_storage((lower.row_count(), columns), indptr, indices, data)?;
        boundaries.push(CanonicalBoundary { degree, storage });
        immediate_faces.push(faces.into_boxed_slice());
    }
    Ok((boundaries, immediate_faces))
}
