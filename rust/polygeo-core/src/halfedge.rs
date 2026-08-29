use std::sync::Arc;

use crate::TopologyError;
use crate::incidence::{ChainView, DisjointSet, NativeBoundary64, SurfaceChain, try_filled};
use crate::{ChainComplex, IntegerRing, IntegralChainComplex};

const UNASSIGNED: usize = usize::MAX;

/// Checked transport payload for one halfedge combinatorial surface.
#[derive(Debug)]
pub struct HalfedgeInput {
    next: Box<[usize]>,
    twin: Box<[usize]>,
    exterior_seeds: Box<[usize]>,
}

impl HalfedgeInput {
    /// Consume native-index buffers and derive the halfedge count without copying.
    ///
    /// # Errors
    ///
    /// Returns a stable count or shape error before mathematical admission.
    pub fn native(
        next: Box<[usize]>,
        twin: Box<[usize]>,
        exterior_seeds: Box<[usize]>,
    ) -> Result<Self, TopologyError> {
        checked_count(next.len())?;
        if twin.len() != next.len() || exterior_seeds.len() > next.len() {
            return Err(TopologyError::HalfedgeShape);
        }
        Ok(Self {
            next,
            twin,
            exterior_seeds,
        })
    }

    /// Admit signed integer relations into the native index domain.
    ///
    /// # Errors
    ///
    /// Returns a stable count, allocation, shape, sign, or index error before
    /// the payload can reach mathematical admission.
    pub fn signed<T>(
        next: impl IntoIterator<Item = T>,
        twin: impl IntoIterator<Item = T>,
        exterior_seeds: impl IntoIterator<Item = T>,
        halfedge_count: usize,
    ) -> Result<Self, TopologyError>
    where
        T: TryInto<i128>,
    {
        checked_count(halfedge_count)?;
        let convert = |value: T| {
            let value = value.try_into().map_err(|_| TopologyError::IndexOverflow)?;
            if value < 0 {
                Err(TopologyError::negative_index(value))
            } else {
                usize::try_from(value)
                    .map_err(|_| TopologyError::index_overflow(value.cast_unsigned()))
            }
        };
        Self::checked(
            next.into_iter().map(convert),
            twin.into_iter().map(convert),
            exterior_seeds.into_iter().map(convert),
            halfedge_count,
        )
    }

    /// Admit unsigned integer relations into the native index domain.
    ///
    /// # Errors
    ///
    /// Returns a stable count, allocation, shape, or index error before the
    /// payload can reach mathematical admission.
    pub fn unsigned<T>(
        next: impl IntoIterator<Item = T>,
        twin: impl IntoIterator<Item = T>,
        exterior_seeds: impl IntoIterator<Item = T>,
        halfedge_count: usize,
    ) -> Result<Self, TopologyError>
    where
        T: TryInto<u128>,
    {
        checked_count(halfedge_count)?;
        let convert = |value: T| {
            let value = value.try_into().map_err(|_| TopologyError::IndexOverflow)?;
            usize::try_from(value).map_err(|_| TopologyError::index_overflow(value))
        };
        Self::checked(
            next.into_iter().map(convert),
            twin.into_iter().map(convert),
            exterior_seeds.into_iter().map(convert),
            halfedge_count,
        )
    }

    fn checked(
        next: impl IntoIterator<Item = Result<usize, TopologyError>>,
        twin: impl IntoIterator<Item = Result<usize, TopologyError>>,
        exterior_seeds: impl IntoIterator<Item = Result<usize, TopologyError>>,
        halfedge_count: usize,
    ) -> Result<Self, TopologyError> {
        let next = collect_exact(next, halfedge_count)?;
        let twin = collect_exact(twin, halfedge_count)?;
        let exterior_seeds = collect_bounded(exterior_seeds, halfedge_count)?;
        Ok(Self {
            next,
            twin,
            exterior_seeds,
        })
    }
}

fn checked_count(halfedge_count: usize) -> Result<(), TopologyError> {
    let maximum = usize::try_from(i64::MAX).unwrap_or(usize::MAX);
    if halfedge_count > maximum {
        return Err(TopologyError::CountOverflow);
    }
    Ok(())
}

fn collect_exact(
    values: impl IntoIterator<Item = Result<usize, TopologyError>>,
    expected: usize,
) -> Result<Box<[usize]>, TopologyError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(expected)
        .map_err(|_| TopologyError::Allocation)?;
    for value in values {
        if output.len() == expected {
            return Err(TopologyError::HalfedgeShape);
        }
        output.push(value?);
    }
    if output.len() != expected {
        return Err(TopologyError::HalfedgeShape);
    }
    Ok(output.into_boxed_slice())
}

fn collect_bounded(
    values: impl IntoIterator<Item = Result<usize, TopologyError>>,
    maximum: usize,
) -> Result<Box<[usize]>, TopologyError> {
    let values = values.into_iter();
    let mut output = Vec::new();
    output
        .try_reserve(values.size_hint().0.min(maximum))
        .map_err(|_| TopologyError::Allocation)?;
    for value in values {
        if output.len() == maximum {
            return Err(TopologyError::HalfedgeShape);
        }
        output
            .try_reserve(1)
            .map_err(|_| TopologyError::Allocation)?;
        output.push(value?);
    }
    Ok(output.into_boxed_slice())
}

/// Classification of a `next` orbit in halfedge terminology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceKind {
    Material,
    Exterior,
}

#[derive(Debug, Clone, Copy)]
struct Entity<'a> {
    owner: &'a HalfedgeSurfaceCore,
    index: usize,
}

impl PartialEq for Entity<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.owner, other.owner) && self.index == other.index
    }
}

impl Eq for Entity<'_> {}

/// Owner-issued halfedge with admitted range and identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Halfedge<'a>(Entity<'a>);

impl<'a> Halfedge<'a> {
    #[must_use]
    pub const fn owner(self) -> &'a HalfedgeSurfaceCore {
        self.0.owner
    }

    /// Explicitly forget owner and entity proof for transport.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0.index
    }

    #[must_use]
    pub fn next(self) -> Self {
        Self(Entity {
            owner: self.0.owner,
            index: self.0.owner.next[self.0.index],
        })
    }

    #[must_use]
    pub fn twin(self) -> Self {
        Self(Entity {
            owner: self.0.owner,
            index: self.0.owner.twin[self.0.index],
        })
    }

    #[must_use]
    pub fn vertex(self) -> Vertex<'a> {
        Vertex(Entity {
            owner: self.0.owner,
            index: self.0.owner.vertices.of_halfedge[self.0.index],
        })
    }

    #[must_use]
    pub fn edge(self) -> Edge<'a> {
        Edge(Entity {
            owner: self.0.owner,
            index: self.0.owner.edges.of_halfedge[self.0.index],
        })
    }

    #[must_use]
    pub fn face_orbit(self) -> FaceOrbit<'a> {
        FaceOrbit(Entity {
            owner: self.0.owner,
            index: self.0.owner.faces.of_halfedge[self.0.index],
        })
    }

    #[must_use]
    pub fn as_exterior(self) -> Option<ExteriorHalfedge<'a>> {
        (self.face_orbit().kind() == FaceKind::Exterior).then_some(ExteriorHalfedge(self))
    }

    #[must_use]
    pub fn as_material_boundary(self) -> Option<MaterialBoundaryHalfedge<'a>> {
        (self.face_orbit().kind() == FaceKind::Material
            && self.twin().face_orbit().kind() == FaceKind::Exterior)
            .then_some(MaterialBoundaryHalfedge(self))
    }
}

/// Owner-issued vertex orbit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vertex<'a>(Entity<'a>);

impl<'a> Vertex<'a> {
    #[must_use]
    pub const fn owner(self) -> &'a HalfedgeSurfaceCore {
        self.0.owner
    }

    #[must_use]
    pub const fn index(self) -> usize {
        self.0.index
    }

    #[must_use]
    pub fn representative(self) -> Halfedge<'a> {
        self.0
            .owner
            .issued_halfedge(self.0.owner.vertices.representatives[self.0.index])
    }

    #[must_use]
    pub fn halfedges(self) -> impl ExactSizeIterator<Item = Halfedge<'a>> + 'a {
        let owner = self.0.owner;
        owner
            .vertices
            .members(self.0.index)
            .iter()
            .copied()
            .map(move |index| owner.issued_halfedge(index))
    }
}

/// Owner-issued edge orbit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge<'a>(Entity<'a>);

impl<'a> Edge<'a> {
    #[must_use]
    pub const fn owner(self) -> &'a HalfedgeSurfaceCore {
        self.0.owner
    }

    #[must_use]
    pub const fn index(self) -> usize {
        self.0.index
    }

    #[must_use]
    pub fn representative(self) -> Halfedge<'a> {
        self.0
            .owner
            .issued_halfedge(self.0.owner.edges.representatives[self.0.index])
    }

    #[must_use]
    pub fn halfedges(self) -> impl ExactSizeIterator<Item = Halfedge<'a>> + 'a {
        let owner = self.0.owner;
        owner
            .edges
            .members(self.0.index)
            .iter()
            .copied()
            .map(move |index| owner.issued_halfedge(index))
    }
}

/// Owner-issued `next` orbit, including material and exterior faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceOrbit<'a>(Entity<'a>);

impl<'a> FaceOrbit<'a> {
    #[must_use]
    pub const fn owner(self) -> &'a HalfedgeSurfaceCore {
        self.0.owner
    }

    #[must_use]
    pub const fn index(self) -> usize {
        self.0.index
    }

    #[must_use]
    pub fn kind(self) -> FaceKind {
        self.0.owner.face_kinds[self.0.index]
    }

    #[must_use]
    pub fn representative(self) -> Halfedge<'a> {
        self.0
            .owner
            .issued_halfedge(self.0.owner.faces.representatives[self.0.index])
    }

    #[must_use]
    pub fn halfedges(self) -> impl ExactSizeIterator<Item = Halfedge<'a>> + 'a {
        let owner = self.0.owner;
        owner
            .faces
            .members(self.0.index)
            .iter()
            .copied()
            .map(move |index| owner.issued_halfedge(index))
    }

    #[must_use]
    pub fn as_material(self) -> Option<MaterialFace<'a>> {
        let index = self.0.owner.material_face_of_orbit[self.0.index];
        (index != UNASSIGNED).then_some(MaterialFace(Entity {
            owner: self.0.owner,
            index,
        }))
    }
}

/// Owner-issued compact material-face basis element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterialFace<'a>(Entity<'a>);

impl<'a> MaterialFace<'a> {
    #[must_use]
    pub const fn owner(self) -> &'a HalfedgeSurfaceCore {
        self.0.owner
    }

    #[must_use]
    pub const fn index(self) -> usize {
        self.0.index
    }

    #[must_use]
    pub fn face_orbit(self) -> FaceOrbit<'a> {
        FaceOrbit(Entity {
            owner: self.0.owner,
            index: self.0.owner.material_faces[self.0.index],
        })
    }

    #[must_use]
    pub fn halfedges(self) -> impl ExactSizeIterator<Item = Halfedge<'a>> + 'a {
        self.face_orbit().halfedges()
    }
}

/// Halfedge refined to an exterior face orbit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExteriorHalfedge<'a>(Halfedge<'a>);

impl<'a> ExteriorHalfedge<'a> {
    #[must_use]
    pub const fn owner(self) -> &'a HalfedgeSurfaceCore {
        self.0.owner()
    }

    #[must_use]
    pub const fn index(self) -> usize {
        self.0.index()
    }

    #[must_use]
    pub const fn halfedge(self) -> Halfedge<'a> {
        self.0
    }

    #[must_use]
    pub fn next(self) -> Self {
        let owner = self.0.owner();
        Self(owner.issued_halfedge(owner.boundary_next[self.0.index()]))
    }
}

/// Material halfedge whose twin belongs to an exterior face orbit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterialBoundaryHalfedge<'a>(Halfedge<'a>);

impl<'a> MaterialBoundaryHalfedge<'a> {
    #[must_use]
    pub const fn owner(self) -> &'a HalfedgeSurfaceCore {
        self.0.owner()
    }

    #[must_use]
    pub const fn index(self) -> usize {
        self.0.index()
    }

    #[must_use]
    pub const fn halfedge(self) -> Halfedge<'a> {
        self.0
    }

    #[must_use]
    pub fn next(self) -> Self {
        let owner = self.0.owner();
        let exterior = owner.twin[self.0.index()];
        Self(owner.issued_halfedge(owner.twin[owner.boundary_next[exterior]]))
    }
}

#[derive(Debug)]
struct OrbitIndex {
    of_halfedge: Box<[usize]>,
    representatives: Box<[usize]>,
    offsets: Box<[usize]>,
    halfedges: Box<[usize]>,
}

impl OrbitIndex {
    fn validated(relation: &'static str, permutation: &[usize]) -> Result<Self, TopologyError> {
        let mut of_halfedge = try_filled(permutation.len(), UNASSIGNED)?;
        validate_permutation_into(relation, permutation, &mut of_halfedge)?;
        of_halfedge.fill(UNASSIGNED);
        Self::enumerate_with_storage(of_halfedge, |halfedge| permutation[halfedge])
    }

    fn enumerate(
        count: usize,
        successor: impl FnMut(usize) -> usize,
    ) -> Result<Self, TopologyError> {
        Self::enumerate_with_storage(try_filled(count, UNASSIGNED)?, successor)
    }

    fn enumerate_with_storage(
        mut of_halfedge: Vec<usize>,
        mut successor: impl FnMut(usize) -> usize,
    ) -> Result<Self, TopologyError> {
        #[cfg(test)]
        record_orbit_enumeration();
        let count = of_halfedge.len();
        let mut representatives = Vec::new();
        let mut offsets = Vec::new();
        let mut halfedges = Vec::new();
        representatives
            .try_reserve_exact(count)
            .map_err(|_| TopologyError::Allocation)?;
        offsets
            .try_reserve_exact(count.checked_add(1).ok_or(TopologyError::CountOverflow)?)
            .map_err(|_| TopologyError::Allocation)?;
        halfedges
            .try_reserve_exact(count)
            .map_err(|_| TopologyError::Allocation)?;
        offsets.push(0);

        for representative in 0..count {
            if of_halfedge[representative] != UNASSIGNED {
                continue;
            }
            let orbit = representatives.len();
            representatives.push(representative);
            let mut halfedge = representative;
            loop {
                if of_halfedge[halfedge] != UNASSIGNED {
                    if halfedge != representative {
                        return Err(TopologyError::InternalInvariant);
                    }
                    break;
                }
                of_halfedge[halfedge] = orbit;
                halfedges.push(halfedge);
                halfedge = successor(halfedge);
            }
            offsets.push(halfedges.len());
        }

        Ok(Self {
            of_halfedge: of_halfedge.into_boxed_slice(),
            representatives: representatives.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
            halfedges: halfedges.into_boxed_slice(),
        })
    }

    const fn count(&self) -> usize {
        self.offsets.len() - 1
    }

    fn members(&self, orbit: usize) -> &[usize] {
        &self.halfedges[self.offsets[orbit]..self.offsets[orbit + 1]]
    }

    fn materialize(&self) -> Result<OrbitMaterialization, TopologyError> {
        Ok(OrbitMaterialization {
            representatives: try_copy(&self.representatives)?,
            offsets: try_copy(&self.offsets)?,
            halfedges: try_copy(&self.halfedges)?,
        })
    }
}

#[derive(Debug)]
struct EdgeIndex {
    of_halfedge: Box<[usize]>,
    representatives: Box<[usize]>,
    offsets: Box<[usize]>,
    halfedges: Box<[usize]>,
}

impl EdgeIndex {
    fn validated(twin: &[usize]) -> Result<Self, TopologyError> {
        #[cfg(test)]
        record_orbit_enumeration();
        let mut of_halfedge = try_filled(twin.len(), UNASSIGNED)?;
        validate_permutation_into("twin", twin, &mut of_halfedge)?;
        validate_twin(twin)?;
        of_halfedge.fill(UNASSIGNED);
        let mut representatives = Vec::new();
        let mut offsets = Vec::new();
        let mut halfedges = Vec::new();
        representatives
            .try_reserve_exact(twin.len() / 2)
            .map_err(|_| TopologyError::Allocation)?;
        halfedges
            .try_reserve_exact(twin.len())
            .map_err(|_| TopologyError::Allocation)?;
        offsets
            .try_reserve_exact(
                twin.len()
                    .checked_div(2)
                    .and_then(|count| count.checked_add(1))
                    .ok_or(TopologyError::CountOverflow)?,
            )
            .map_err(|_| TopologyError::Allocation)?;
        offsets.push(0);
        for representative in 0..twin.len() {
            if of_halfedge[representative] != UNASSIGNED {
                continue;
            }
            let orbit = representatives.len();
            representatives.push(representative);
            of_halfedge[representative] = orbit;
            of_halfedge[twin[representative]] = orbit;
            halfedges.push(representative);
            halfedges.push(twin[representative]);
            offsets.push(halfedges.len());
        }
        Ok(Self {
            of_halfedge: of_halfedge.into_boxed_slice(),
            representatives: representatives.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
            halfedges: halfedges.into_boxed_slice(),
        })
    }

    const fn count(&self) -> usize {
        self.representatives.len()
    }

    fn members(&self, orbit: usize) -> &[usize] {
        &self.halfedges[self.offsets[orbit]..self.offsets[orbit + 1]]
    }

    fn materialize(&self) -> Result<OrbitMaterialization, TopologyError> {
        Ok(OrbitMaterialization {
            representatives: try_copy(&self.representatives)?,
            offsets: try_copy(&self.offsets)?,
            halfedges: try_copy(&self.halfedges)?,
        })
    }
}

fn try_copy(values: &[usize]) -> Result<Box<[usize]>, TopologyError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| TopologyError::Allocation)?;
    output.extend_from_slice(values);
    Ok(output.into_boxed_slice())
}

/// Explicit caller-owned packed orbit materialization.
#[derive(Debug, PartialEq, Eq)]
pub struct OrbitMaterialization {
    representatives: Box<[usize]>,
    offsets: Box<[usize]>,
    halfedges: Box<[usize]>,
}

/// Structural parts returned when explicitly dismantling an orbit materialization.
pub type OrbitMaterializationParts = (Box<[usize]>, Box<[usize]>, Box<[usize]>);

impl OrbitMaterialization {
    #[must_use]
    pub const fn representatives(&self) -> &[usize] {
        &self.representatives
    }

    #[must_use]
    pub const fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    #[must_use]
    pub const fn halfedges(&self) -> &[usize] {
        &self.halfedges
    }

    #[must_use]
    pub fn into_parts(self) -> OrbitMaterializationParts {
        (self.representatives, self.offsets, self.halfedges)
    }
}

#[derive(Debug, Clone, Copy)]
struct BoundaryEntry {
    row: usize,
    column: usize,
    coefficient: i64,
}

fn checked_coefficient_sum(
    current: i64,
    contribution: i64,
    halfedge_count: usize,
) -> Result<i64, TopologyError> {
    let sum = current
        .checked_add(contribution)
        .ok_or(TopologyError::CountOverflow)?;
    let bound = u64::try_from(halfedge_count).map_err(|_| TopologyError::CountOverflow)?;
    if sum.unsigned_abs() > bound {
        return Err(TopologyError::CountOverflow);
    }
    Ok(sum)
}

fn build_boundary(
    degree: usize,
    shape: (usize, usize),
    entries: Vec<BoundaryEntry>,
    halfedge_count: usize,
) -> Result<NativeBoundary64, TopologyError> {
    if entries
        .windows(2)
        .any(|pair| pair[0].column > pair[1].column)
    {
        return Err(TopologyError::InternalInvariant);
    }
    let mut row_counts = try_filled(shape.0, 0_usize)?;
    for entry in &entries {
        let row = entry.row;
        let column = entry.column;
        if row >= shape.0 || column >= shape.1 {
            return Err(TopologyError::InternalInvariant);
        }
        row_counts[row] = row_counts[row]
            .checked_add(1)
            .ok_or(TopologyError::CountOverflow)?;
    }

    let mut indptr = Vec::new();
    indptr
        .try_reserve_exact(shape.0.checked_add(1).ok_or(TopologyError::CountOverflow)?)
        .map_err(|_| TopologyError::Allocation)?;
    indptr.push(0_usize);
    for count in row_counts {
        indptr.push(
            indptr
                .last()
                .copied()
                .ok_or(TopologyError::InternalInvariant)?
                .checked_add(count)
                .ok_or(TopologyError::CountOverflow)?,
        );
    }
    let mut cursors = try_copy(&indptr[..shape.0])?.into_vec();
    let mut indices = try_filled(entries.len(), 0_usize)?;
    let mut data = try_filled(entries.len(), 0_i64)?;
    for entry in entries {
        let position = cursors[entry.row];
        indices[position] = entry.column;
        data[position] = entry.coefficient;
        cursors[entry.row] = position
            .checked_add(1)
            .ok_or(TopologyError::CountOverflow)?;
    }

    let mut write = 0;
    let mut compact_indptr = Vec::new();
    compact_indptr
        .try_reserve_exact(indptr.len())
        .map_err(|_| TopologyError::Allocation)?;
    compact_indptr.push(0_usize);
    for row in 0..shape.0 {
        let mut read = indptr[row];
        let end = indptr[row + 1];
        while read < end {
            let column = indices[read];
            let mut coefficient = 0_i64;
            while read < end && indices[read] == column {
                coefficient = checked_coefficient_sum(coefficient, data[read], halfedge_count)?;
                read += 1;
            }
            if coefficient != 0 {
                indices[write] = column;
                data[write] = coefficient;
                write += 1;
            }
        }
        compact_indptr.push(write);
    }
    indices.truncate(write);
    data.truncate(write);
    NativeBoundary64::try_from_csr(degree, shape, compact_indptr, indices, data)
}

fn verify_boundary_square(
    boundary_one: &NativeBoundary64,
    boundary_two: &NativeBoundary64,
    edge_count: usize,
    halfedge_count: usize,
) -> Result<(), TopologyError> {
    let mut lower = boundary_one.exact_entries().collect::<Vec<_>>();
    lower.sort_unstable_by_key(|(row, column, _)| (*column, *row));
    let mut offsets = try_filled(
        edge_count
            .checked_add(1)
            .ok_or(TopologyError::CountOverflow)?,
        0_usize,
    )?;
    for &(_, edge, _) in &lower {
        offsets[edge + 1] = offsets[edge + 1]
            .checked_add(1)
            .ok_or(TopologyError::CountOverflow)?;
    }
    for edge in 0..edge_count {
        offsets[edge + 1] = offsets[edge + 1]
            .checked_add(offsets[edge])
            .ok_or(TopologyError::CountOverflow)?;
    }

    let upper = boundary_two.exact_entries().collect::<Vec<_>>();
    let capacity = upper
        .len()
        .checked_mul(2)
        .ok_or(TopologyError::CountOverflow)?;
    let mut product = Vec::new();
    product
        .try_reserve_exact(capacity)
        .map_err(|_| TopologyError::Allocation)?;
    for &(edge, face, upper_coefficient) in &upper {
        for &(vertex, _, lower_coefficient) in &lower[offsets[edge]..offsets[edge + 1]] {
            product.push(BoundaryEntry {
                row: vertex,
                column: face,
                coefficient: lower_coefficient
                    .checked_mul(upper_coefficient)
                    .ok_or(TopologyError::CountOverflow)?,
            });
        }
    }
    product.sort_unstable_by_key(|entry| (entry.row, entry.column));
    let mut position = 0;
    while position < product.len() {
        let row = product[position].row;
        let column = product[position].column;
        let mut coefficient = 0_i64;
        while position < product.len()
            && product[position].row == row
            && product[position].column == column
        {
            coefficient = checked_coefficient_sum(
                coefficient,
                product[position].coefficient,
                halfedge_count,
            )?;
            position += 1;
        }
        if coefficient != 0 {
            return Err(TopologyError::InternalInvariant);
        }
    }
    Ok(())
}

fn build_surface_chain(
    halfedge_count: usize,
    vertices: &OrbitIndex,
    edges: &EdgeIndex,
    faces: &OrbitIndex,
    twin: &[usize],
    material_faces: &[usize],
) -> Result<SurfaceChain, TopologyError> {
    #[cfg(test)]
    CHAIN_PROJECTIONS.set(CHAIN_PROJECTIONS.get() + 1);
    let boundary_zero = build_boundary(0, (0, vertices.count()), Vec::new(), halfedge_count)?;

    let mut boundary_one_entries = Vec::new();
    boundary_one_entries
        .try_reserve_exact(halfedge_count)
        .map_err(|_| TopologyError::Allocation)?;
    for (edge, representative) in edges.representatives.iter().copied().enumerate() {
        let origin = vertices.of_halfedge[representative];
        let terminal = vertices.of_halfedge[twin[representative]];
        if origin != terminal {
            boundary_one_entries.push(BoundaryEntry {
                row: origin,
                column: edge,
                coefficient: -1,
            });
            boundary_one_entries.push(BoundaryEntry {
                row: terminal,
                column: edge,
                coefficient: 1,
            });
        }
    }
    let boundary_one = build_boundary(
        1,
        (vertices.count(), edges.count()),
        boundary_one_entries,
        halfedge_count,
    )?;

    let mut boundary_two_entries = Vec::new();
    boundary_two_entries
        .try_reserve_exact(halfedge_count)
        .map_err(|_| TopologyError::Allocation)?;
    for (material_face, face_orbit) in material_faces.iter().copied().enumerate() {
        for halfedge in faces.members(face_orbit).iter().copied() {
            let edge = edges.of_halfedge[halfedge];
            boundary_two_entries.push(BoundaryEntry {
                row: edge,
                column: material_face,
                coefficient: if edges.representatives[edge] == halfedge {
                    1
                } else {
                    -1
                },
            });
        }
    }
    let boundary_two = build_boundary(
        2,
        (edges.count(), material_faces.len()),
        boundary_two_entries,
        halfedge_count,
    )?;
    verify_boundary_square(&boundary_one, &boundary_two, edges.count(), halfedge_count)?;
    Ok(SurfaceChain::new([
        boundary_zero,
        boundary_one,
        boundary_two,
    ]))
}

/// Immutable authority for one admitted orientable halfedge surface.
#[derive(Debug)]
pub struct HalfedgeSurfaceCore {
    next: Box<[usize]>,
    twin: Box<[usize]>,
    boundary_next: Box<[usize]>,
    vertices: OrbitIndex,
    edges: EdgeIndex,
    faces: OrbitIndex,
    face_kinds: Box<[FaceKind]>,
    material_faces: Box<[usize]>,
    material_face_of_orbit: Box<[usize]>,
    pub(crate) chain: SurfaceChain,
    facts: SurfaceFacts,
}

#[derive(Debug)]
struct SurfaceFacts {
    boundary_components: usize,
    connected_components: usize,
    euler_characteristic: i64,
    genus: Option<usize>,
}

impl HalfedgeSurfaceCore {
    pub(crate) const fn presentation_next(&self) -> &[usize] {
        &self.next
    }

    pub(crate) const fn presentation_twin(&self) -> &[usize] {
        &self.twin
    }

    pub(crate) const fn presentation_face_kinds(&self) -> &[FaceKind] {
        &self.face_kinds
    }

    /// Admit one checked permutation presentation into an immutable owner.
    ///
    /// # Errors
    ///
    /// Returns a classified topology error before any owner is published.
    pub fn admit(input: HalfedgeInput) -> Result<Arc<Self>, TopologyError> {
        let HalfedgeInput {
            next,
            twin,
            exterior_seeds,
        } = input;
        let faces = OrbitIndex::validated("next", &next)?;
        let edges = EdgeIndex::validated(&twin)?;
        let vertices = OrbitIndex::enumerate(next.len(), |halfedge| next[twin[halfedge]])?;

        let mut face_kinds = try_filled(faces.count(), FaceKind::Material)?;
        for seed in exterior_seeds.iter().copied() {
            let face = faces.of_halfedge.get(seed).copied().ok_or_else(|| {
                TopologyError::halfedge_range("exterior_seed", seed, seed, next.len())
            })?;
            if face_kinds[face] == FaceKind::Exterior {
                return Err(TopologyError::exterior_inconsistency(seed, face));
            }
            face_kinds[face] = FaceKind::Exterior;
        }

        for edge in 0..edges.count() {
            let halfedge = edges.representatives[edge];
            let paired = twin[halfedge];
            if face_kinds[faces.of_halfedge[halfedge]] == FaceKind::Exterior
                && face_kinds[faces.of_halfedge[paired]] == FaceKind::Exterior
            {
                return Err(TopologyError::exterior_inconsistency(halfedge, paired));
            }
        }

        let mut exterior_at_vertex = try_filled(vertices.count(), UNASSIGNED)?;
        let mut boundary_next = try_filled(next.len(), UNASSIGNED)?;
        for halfedge in 0..next.len() {
            if face_kinds[faces.of_halfedge[halfedge]] != FaceKind::Exterior {
                continue;
            }
            let vertex = vertices.of_halfedge[halfedge];
            if exterior_at_vertex[vertex] != UNASSIGNED {
                return Err(TopologyError::boundary_cycle(
                    halfedge,
                    exterior_at_vertex[vertex],
                ));
            }
            exterior_at_vertex[vertex] = halfedge;
            boundary_next[halfedge] = next[halfedge];
        }

        let mut material_faces = Vec::new();
        material_faces
            .try_reserve_exact(face_kinds.len())
            .map_err(|_| TopologyError::Allocation)?;
        material_faces.extend(
            face_kinds
                .iter()
                .enumerate()
                .filter_map(|(face, kind)| (*kind == FaceKind::Material).then_some(face)),
        );
        let mut material_face_of_orbit = try_filled(face_kinds.len(), UNASSIGNED)?;
        for (material_face, face_orbit) in material_faces.iter().copied().enumerate() {
            material_face_of_orbit[face_orbit] = material_face;
        }
        let chain = build_surface_chain(
            next.len(),
            &vertices,
            &edges,
            &faces,
            &twin,
            &material_faces,
        )?;
        let topology_facts = surface_facts(
            &vertices,
            &edges,
            &twin,
            faces.count() - material_faces.len(),
            material_faces.len(),
        )?;

        Ok(Arc::new(Self {
            next,
            twin,
            boundary_next: boundary_next.into_boxed_slice(),
            vertices,
            edges,
            faces,
            face_kinds: face_kinds.into_boxed_slice(),
            material_faces: material_faces.into_boxed_slice(),
            material_face_of_orbit: material_face_of_orbit.into_boxed_slice(),
            chain,
            facts: topology_facts,
        }))
    }

    /// Retain this topology owner as an exact integral chain complex.
    #[must_use]
    pub fn chain_complex(self: &Arc<Self>) -> IntegralChainComplex {
        ChainComplex::halfedge(Arc::clone(self), IntegerRing)
    }

    #[must_use]
    pub const fn halfedge_count(&self) -> usize {
        self.next.len()
    }

    /// Borrow this topology through the sealed exact chain interface.
    #[must_use]
    pub const fn chain_view(&self) -> ChainView<'_> {
        ChainView::halfedge(self)
    }

    /// Issue one owner-bound halfedge after a single range check.
    ///
    /// # Errors
    ///
    /// Returns `halfedge_range` when `index` is outside this owner.
    pub fn halfedge(&self, index: usize) -> Result<Halfedge<'_>, TopologyError> {
        if index >= self.halfedge_count() {
            return Err(TopologyError::halfedge_range(
                "halfedge",
                index,
                index,
                self.halfedge_count(),
            ));
        }
        Ok(Halfedge(Entity { owner: self, index }))
    }

    fn issued_halfedge(&self, index: usize) -> Halfedge<'_> {
        Halfedge(Entity { owner: self, index })
    }

    #[must_use]
    pub fn halfedges(&self) -> impl ExactSizeIterator<Item = Halfedge<'_>> + '_ {
        (0..self.halfedge_count()).map(|index| self.issued_halfedge(index))
    }

    #[must_use]
    pub const fn vertex_count(&self) -> usize {
        self.vertices.count()
    }

    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.edges.count()
    }

    #[must_use]
    pub const fn face_orbit_count(&self) -> usize {
        self.faces.count()
    }

    #[must_use]
    pub const fn material_face_count(&self) -> usize {
        self.material_faces.len()
    }

    #[must_use]
    pub const fn exterior_face_count(&self) -> usize {
        self.faces.count() - self.material_faces.len()
    }

    #[must_use]
    pub const fn boundary_component_count(&self) -> usize {
        self.facts.boundary_components
    }

    #[must_use]
    pub const fn connected_component_count(&self) -> usize {
        self.facts.connected_components
    }

    #[must_use]
    pub const fn euler_characteristic(&self) -> i64 {
        self.facts.euler_characteristic
    }

    #[must_use]
    pub const fn genus(&self) -> Option<usize> {
        self.facts.genus
    }

    #[must_use]
    pub fn vertices(&self) -> impl ExactSizeIterator<Item = Vertex<'_>> + '_ {
        (0..self.vertex_count()).map(|index| Vertex(Entity { owner: self, index }))
    }

    #[must_use]
    pub fn edges(&self) -> impl ExactSizeIterator<Item = Edge<'_>> + '_ {
        (0..self.edge_count()).map(|index| Edge(Entity { owner: self, index }))
    }

    #[must_use]
    pub fn face_orbits(&self) -> impl ExactSizeIterator<Item = FaceOrbit<'_>> + '_ {
        (0..self.face_orbit_count()).map(|index| FaceOrbit(Entity { owner: self, index }))
    }

    #[must_use]
    pub fn material_faces(&self) -> impl ExactSizeIterator<Item = MaterialFace<'_>> + '_ {
        (0..self.material_face_count()).map(|index| MaterialFace(Entity { owner: self, index }))
    }

    /// Materialize caller-owned packed vertex-orbit arrays.
    ///
    /// # Errors
    ///
    /// Returns an allocation error without changing the owner.
    pub fn materialize_vertex_orbits(&self) -> Result<OrbitMaterialization, TopologyError> {
        self.vertices.materialize()
    }

    /// Materialize caller-owned packed edge-orbit arrays.
    ///
    /// # Errors
    ///
    /// Returns an allocation error without changing the owner.
    pub fn materialize_edge_orbits(&self) -> Result<OrbitMaterialization, TopologyError> {
        self.edges.materialize()
    }

    /// Materialize caller-owned packed face-orbit arrays.
    ///
    /// # Errors
    ///
    /// Returns an allocation error without changing the owner.
    pub fn materialize_face_orbits(&self) -> Result<OrbitMaterialization, TopologyError> {
        self.faces.materialize()
    }
}

fn surface_facts(
    vertices: &OrbitIndex,
    edges: &EdgeIndex,
    twin: &[usize],
    boundary_components: usize,
    material_faces: usize,
) -> Result<SurfaceFacts, TopologyError> {
    let mut components = DisjointSet::try_new(vertices.count())?;
    for representative in edges.representatives.iter().copied() {
        let left = vertices.of_halfedge[representative];
        let right = vertices.of_halfedge[twin[representative]];
        components.join(left, right);
    }
    let connected_components = (0..vertices.count())
        .filter(|&vertex| components.is_root(vertex))
        .count();
    let vertex_count = i64::try_from(vertices.count()).map_err(|_| TopologyError::CountOverflow)?;
    let edge_count = i64::try_from(edges.count()).map_err(|_| TopologyError::CountOverflow)?;
    let face_count = i64::try_from(material_faces).map_err(|_| TopologyError::CountOverflow)?;
    let euler_characteristic = vertex_count
        .checked_sub(edge_count)
        .and_then(|value| value.checked_add(face_count))
        .ok_or(TopologyError::CountOverflow)?;
    let genus = if connected_components == 1 {
        let boundary =
            i64::try_from(boundary_components).map_err(|_| TopologyError::CountOverflow)?;
        let numerator = 2_i64
            .checked_sub(boundary)
            .and_then(|value| value.checked_sub(euler_characteristic))
            .ok_or(TopologyError::CountOverflow)?;
        if numerator < 0 || numerator % 2 != 0 {
            return Err(TopologyError::InternalInvariant);
        }
        Some(usize::try_from(numerator / 2).map_err(|_| TopologyError::CountOverflow)?)
    } else {
        None
    };
    Ok(SurfaceFacts {
        boundary_components,
        connected_components,
        euler_characteristic,
        genus,
    })
}

fn validate_permutation_into(
    relation: &'static str,
    values: &[usize],
    seen: &mut [usize],
) -> Result<(), TopologyError> {
    for (halfedge, value) in values.iter().copied().enumerate() {
        if value >= values.len() {
            return Err(TopologyError::halfedge_range(
                relation,
                halfedge,
                value,
                values.len(),
            ));
        }
        if seen[value] != UNASSIGNED {
            return Err(TopologyError::halfedge_permutation(
                relation,
                halfedge,
                value,
                values.len(),
            ));
        }
        seen[value] = halfedge;
    }
    Ok(())
}

fn validate_twin(twin: &[usize]) -> Result<(), TopologyError> {
    for (halfedge, paired) in twin.iter().copied().enumerate() {
        let twin_back = twin[paired];
        if paired == halfedge || twin_back != halfedge {
            return Err(TopologyError::twin_law(halfedge, paired, twin_back));
        }
    }
    Ok(())
}

#[cfg(test)]
std::thread_local! {
    static ORBIT_ENUMERATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static CHAIN_PROJECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn record_orbit_enumeration() {
    ORBIT_ENUMERATIONS.set(ORBIT_ENUMERATIONS.get() + 1);
}

#[cfg(test)]
fn orbit_enumeration_count() -> usize {
    ORBIT_ENUMERATIONS.get()
}

#[cfg(test)]
mod tests {
    use super::{
        CHAIN_PROJECTIONS, HalfedgeInput, HalfedgeSurfaceCore, checked_coefficient_sum,
        orbit_enumeration_count,
    };
    use crate::TopologyError;

    #[test]
    fn warmed_navigation_does_not_reenter_orbit_enumeration() {
        let input =
            HalfedgeInput::unsigned([1_u8, 2, 0, 5, 3, 4], [3_u8, 4, 5, 0, 1, 2], [3_u8], 6)
                .unwrap();
        let before = orbit_enumeration_count();
        let surface = HalfedgeSurfaceCore::admit(input).unwrap();
        let admitted = orbit_enumeration_count();
        assert_eq!(admitted - before, 3);

        let mut digest = 0_usize;
        for _ in 0..1_024 {
            for halfedge in surface.halfedges() {
                digest ^= halfedge.next().index();
                digest ^= halfedge.twin().index();
                let _ = halfedge.vertex();
                let _ = halfedge.edge();
                let _ = halfedge.face_orbit();
            }
        }
        std::hint::black_box(digest);
        assert_eq!(orbit_enumeration_count(), admitted);
    }

    #[test]
    fn native_input_transfers_owned_buffers_without_copying() {
        let next = vec![1, 2, 0, 5, 3, 4];
        let next_pointer = next.as_ptr();
        let input = HalfedgeInput::native(
            next.into_boxed_slice(),
            vec![3, 4, 5, 0, 1, 2].into_boxed_slice(),
            vec![3].into_boxed_slice(),
        )
        .unwrap();
        let surface = HalfedgeSurfaceCore::admit(input).unwrap();

        assert_eq!(surface.next.as_ptr(), next_pointer);
    }

    #[test]
    fn warmed_chain_views_do_not_rebuild_projection() {
        let input =
            HalfedgeInput::unsigned([1_u8, 2, 0, 5, 3, 4], [3_u8, 4, 5, 0, 1, 2], [3_u8], 6)
                .unwrap();
        let before = CHAIN_PROJECTIONS.get();
        let surface = HalfedgeSurfaceCore::admit(input).unwrap();
        let admitted = CHAIN_PROJECTIONS.get();
        assert_eq!(admitted - before, 1);

        for _ in 0..1_024 {
            let entries = surface
                .chain_view()
                .boundary(2)
                .unwrap()
                .exact_entries()
                .count();
            std::hint::black_box(entries);
        }
        assert_eq!(CHAIN_PROJECTIONS.get(), admitted);
    }

    #[test]
    fn coefficient_overflow_fails_without_large_allocation() {
        let bound = usize::try_from(i64::MAX).unwrap_or(usize::MAX);
        assert_eq!(
            checked_coefficient_sum(i64::MAX, 1, bound).unwrap_err(),
            TopologyError::CountOverflow
        );
        assert_eq!(
            checked_coefficient_sum(1, 1, 1).unwrap_err(),
            TopologyError::CountOverflow
        );
    }
}
