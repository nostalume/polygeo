use once_cell::sync::OnceCell;

use crate::TopologyError;
use crate::complex::ComplexCore;
use crate::incidence::{Basis, DisjointSet, try_filled};
use crate::mask::{PackedDegreeMasks, PackedDegreeMasksBuilder};

#[derive(Debug)]
struct CenteredLinkEdges {
    offsets: Box<[usize]>,
    endpoints: Box<[[usize; 2]]>,
}

impl CenteredLinkEdges {
    fn partition(triangles: &Basis, vertex_count: usize) -> Result<Self, TopologyError> {
        let mut counts = try_filled(vertex_count, 0_usize)?;
        for triangle_index in 0..triangles.row_count() {
            let triangle = triangles
                .row(triangle_index)
                .ok_or(TopologyError::InternalInvariant)?;
            for center in triangle {
                counts[*center] = counts[*center]
                    .checked_add(1)
                    .ok_or(TopologyError::CountOverflow)?;
            }
        }
        if counts.contains(&0) {
            return Err(TopologyError::NotPure);
        }

        let mut offsets = Vec::new();
        offsets
            .try_reserve_exact(
                vertex_count
                    .checked_add(1)
                    .ok_or(TopologyError::CountOverflow)?,
            )
            .map_err(|_| TopologyError::Allocation)?;
        offsets.push(0_usize);
        for count in counts {
            let next = offsets
                .last()
                .copied()
                .ok_or(TopologyError::InternalInvariant)?
                .checked_add(count)
                .ok_or(TopologyError::CountOverflow)?;
            offsets.push(next);
        }
        let record_count = offsets
            .last()
            .copied()
            .ok_or(TopologyError::InternalInvariant)?;
        let mut cursors = Vec::new();
        cursors
            .try_reserve_exact(vertex_count)
            .map_err(|_| TopologyError::Allocation)?;
        cursors.extend_from_slice(&offsets[..vertex_count]);
        let mut endpoints = try_filled(record_count, [0_usize; 2])?;
        for triangle_index in 0..triangles.row_count() {
            let triangle = triangles
                .row(triangle_index)
                .ok_or(TopologyError::InternalInvariant)?;
            let records = [
                (triangle[0], [triangle[1], triangle[2]]),
                (triangle[1], [triangle[0], triangle[2]]),
                (triangle[2], [triangle[0], triangle[1]]),
            ];
            for (center, edge) in records {
                let position = cursors[center];
                endpoints[position] = edge;
                cursors[center] = position
                    .checked_add(1)
                    .ok_or(TopologyError::CountOverflow)?;
            }
        }
        Ok(Self {
            offsets: offsets.into_boxed_slice(),
            endpoints: endpoints.into_boxed_slice(),
        })
    }

    fn for_center(&self, center: usize) -> Result<&[[usize; 2]], TopologyError> {
        let end_index = center.checked_add(1).ok_or(TopologyError::CountOverflow)?;
        let start = self
            .offsets
            .get(center)
            .copied()
            .ok_or(TopologyError::InternalInvariant)?;
        let end = self
            .offsets
            .get(end_index)
            .copied()
            .ok_or(TopologyError::InternalInvariant)?;
        self.endpoints
            .get(start..end)
            .ok_or(TopologyError::InternalInvariant)
    }
}

#[derive(Debug)]
struct LinkScratch {
    degrees: Vec<u8>,
    first: Vec<usize>,
    second: Vec<usize>,
    touched: Vec<usize>,
}

impl LinkScratch {
    fn try_new(vertex_count: usize) -> Result<Self, TopologyError> {
        Ok(Self {
            degrees: try_filled(vertex_count, 0_u8)?,
            first: try_filled(vertex_count, 0_usize)?,
            second: try_filled(vertex_count, 0_usize)?,
            touched: Vec::new(),
        })
    }

    fn validate(&mut self, edges: &[[usize; 2]]) -> Result<(), TopologyError> {
        self.touched.clear();
        for [left, right] in edges {
            self.attach(*left, *right)?;
            self.attach(*right, *left)?;
        }
        if self.touched.is_empty() {
            return Err(TopologyError::VertexLink);
        }

        let mut endpoints = 0_usize;
        let mut path_start = None;
        for vertex in self.touched.iter().copied() {
            match self.degrees[vertex] {
                1 => {
                    endpoints += 1;
                    path_start = Some(vertex);
                }
                2 => {}
                _ => return Err(TopologyError::VertexLink),
            }
        }
        let path = endpoints == 2;
        let cycle = endpoints == 0;
        if !(path || cycle) {
            return Err(TopologyError::VertexLink);
        }
        let start = if path {
            path_start.ok_or(TopologyError::InternalInvariant)?
        } else {
            self.touched[0]
        };
        self.walk_component(start, path)?;
        for vertex in self.touched.iter().copied() {
            self.degrees[vertex] = 0;
        }
        Ok(())
    }

    fn attach(&mut self, vertex: usize, neighbor: usize) -> Result<(), TopologyError> {
        let degree = self
            .degrees
            .get_mut(vertex)
            .ok_or(TopologyError::InternalInvariant)?;
        match *degree {
            0 => {
                self.touched
                    .try_reserve(1)
                    .map_err(|_| TopologyError::Allocation)?;
                self.touched.push(vertex);
                self.first[vertex] = neighbor;
                *degree = 1;
            }
            1 if self.first[vertex] != neighbor => {
                self.second[vertex] = neighbor;
                *degree = 2;
            }
            _ => return Err(TopologyError::VertexLink),
        }
        Ok(())
    }

    fn walk_component(&self, start: usize, path: bool) -> Result<(), TopologyError> {
        let mut previous = None;
        let mut current = start;
        for step in 0..self.touched.len() {
            let next = match self.degrees[current] {
                1 => {
                    let neighbor = self.first[current];
                    if previous == Some(neighbor) {
                        if path && step + 1 == self.touched.len() {
                            return Ok(());
                        }
                        return Err(TopologyError::VertexLink);
                    }
                    neighbor
                }
                2 => {
                    if previous == Some(self.first[current]) {
                        self.second[current]
                    } else {
                        self.first[current]
                    }
                }
                _ => return Err(TopologyError::InternalInvariant),
            };
            if next == start {
                if !path && step + 1 == self.touched.len() {
                    return Ok(());
                }
                return Err(TopologyError::VertexLink);
            }
            if step + 1 == self.touched.len() {
                return Err(TopologyError::VertexLink);
            }
            previous = Some(current);
            current = next;
        }
        Err(TopologyError::VertexLink)
    }
}

#[derive(Debug)]
enum DomainVerdict<E, R> {
    Admitted(E),
    Rejected(R),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegularRejection {
    NotPure {
        vertex: usize,
    },
    CodimensionOneIncidence {
        degree: usize,
        simplex: usize,
        cofaces: usize,
    },
}

impl RegularRejection {
    const fn error(self) -> TopologyError {
        match self {
            Self::NotPure { vertex } => {
                let _ = vertex;
                TopologyError::not_pure(vertex)
            }
            Self::CodimensionOneIncidence {
                degree,
                simplex,
                cofaces,
            } => {
                let _ = (degree, simplex, cofaces);
                TopologyError::codimension_one_incidence(degree, simplex, cofaces)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TriangleRejection {
    Regular(RegularRejection),
    WrongDimension { actual: usize },
    VertexLink { vertex: usize },
}

impl TriangleRejection {
    const fn error(self) -> TopologyError {
        match self {
            Self::Regular(rejection) => rejection.error(),
            Self::WrongDimension { actual } => {
                let _ = actual;
                TopologyError::triangle_dimension(actual)
            }
            Self::VertexLink { vertex } => {
                let _ = vertex;
                TopologyError::vertex_link(vertex)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OrientationRejection {
    codimension_one_simplex: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConnectivityRejection {
    unreachable_vertex: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OperationalFailure(TopologyError);

impl OperationalFailure {
    const fn error(self) -> TopologyError {
        self.0
    }

    const fn internal(error: TopologyError) -> Self {
        match error {
            TopologyError::Allocation
            | TopologyError::CountOverflow
            | TopologyError::InternalInvariant => Self(error),
            _ => Self(TopologyError::InternalInvariant),
        }
    }
}

type CheckResult<E, R> = Result<DomainVerdict<E, R>, OperationalFailure>;

fn initialize<E, R>(
    cell: &OnceCell<DomainVerdict<E, R>>,
    checker: impl FnOnce() -> CheckResult<E, R>,
) -> Result<&DomainVerdict<E, R>, TopologyError> {
    cell.get_or_try_init(checker)
        .map_err(OperationalFailure::error)
}

#[derive(Debug)]
pub(crate) struct RegularEvidence {
    boundary: PackedDegreeMasks,
    kind: BoundaryKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundaryKind {
    Empty,
    NonEmpty {
        first_codimension_one_simplex: usize,
    },
}

#[derive(Debug, Default)]
pub(crate) struct EvidenceStore {
    regular: OnceCell<DomainVerdict<RegularEvidence, RegularRejection>>,
    triangle: OnceCell<DomainVerdict<(), TriangleRejection>>,
    oriented: OnceCell<DomainVerdict<(), OrientationRejection>>,
    connected: OnceCell<DomainVerdict<(), ConnectivityRejection>>,
    disk: OnceCell<DomainVerdict<(), DiskRejection>>,
    #[cfg(test)]
    computations: TestComputationCounts,
}

#[cfg(test)]
#[derive(Debug, Default)]
struct TestComputationCounts {
    regular: std::sync::atomic::AtomicUsize,
    triangle: std::sync::atomic::AtomicUsize,
    oriented: std::sync::atomic::AtomicUsize,
    connected: std::sync::atomic::AtomicUsize,
    disk: std::sync::atomic::AtomicUsize,
}

#[derive(Debug, Clone, Copy)]
enum DiskRejection {
    BoundaryComponents(usize),
    EulerCharacteristic(i128),
}

impl DiskRejection {
    const fn error(self) -> TopologyError {
        match self {
            Self::BoundaryComponents(count) => TopologyError::disk_boundary_components(count),
            Self::EulerCharacteristic(value) => TopologyError::disk_euler_characteristic(value),
        }
    }
}

/// Borrowed proof of arbitrary-dimensional codimension-one regularity.
#[derive(Debug, Clone, Copy)]
pub struct RegularView<'a> {
    owner: &'a ComplexCore,
    row: &'a RegularEvidence,
}

impl<'a> RegularView<'a> {
    #[must_use]
    pub const fn owner(&self) -> &ComplexCore {
        self.owner
    }

    /// Export one caller-owned Boolean degree mask.
    ///
    /// # Errors
    ///
    /// Returns a degree or allocation error.
    pub fn boundary_mask(&self, degree: usize) -> Result<Vec<bool>, TopologyError> {
        self.row.boundary.export_degree(degree, &self.owner.layout)
    }

    /// Fill one caller-owned Boolean degree mask directly.
    ///
    /// # Errors
    ///
    /// Returns a degree or mask-shape error.
    pub fn write_boundary_mask(
        &self,
        degree: usize,
        output: &mut [bool],
    ) -> Result<(), TopologyError> {
        self.row
            .boundary
            .write_degree(degree, &self.owner.layout, output)
    }

    /// Project the constructive nonempty-boundary classification.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::BoundaryAbsent`] for the canonical empty row.
    pub fn with_boundary(self) -> Result<WithBoundaryView<'a>, TopologyError> {
        match self.row.kind {
            BoundaryKind::Empty => Err(TopologyError::BoundaryAbsent),
            BoundaryKind::NonEmpty {
                first_codimension_one_simplex,
            } => Ok(WithBoundaryView {
                regular: self,
                witness: first_codimension_one_simplex,
            }),
        }
    }

    /// Project the empty-boundary classification.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::BoundaryPresent`] for a canonical nonempty row.
    pub fn without_boundary(self) -> Result<WithoutBoundaryView<'a>, TopologyError> {
        match self.row.kind {
            BoundaryKind::Empty => Ok(WithoutBoundaryView { regular: self }),
            BoundaryKind::NonEmpty {
                first_codimension_one_simplex,
            } => Err(TopologyError::boundary_present(
                first_codimension_one_simplex,
            )),
        }
    }

    pub(crate) const fn packed_boundary(self) -> &'a PackedDegreeMasks {
        &self.row.boundary
    }
}

/// Borrowed constructive proof of a nonempty topological boundary.
#[derive(Debug, Clone, Copy)]
pub struct WithBoundaryView<'a> {
    regular: RegularView<'a>,
    witness: usize,
}

impl WithBoundaryView<'_> {
    #[must_use]
    pub const fn regular(&self) -> RegularView<'_> {
        self.regular
    }

    #[must_use]
    pub const fn codimension_one_simplex(&self) -> usize {
        self.witness
    }
}

/// Borrowed proof of an empty topological boundary.
#[derive(Debug, Clone, Copy)]
pub struct WithoutBoundaryView<'a> {
    regular: RegularView<'a>,
}

impl WithoutBoundaryView<'_> {
    #[must_use]
    pub const fn regular(&self) -> RegularView<'_> {
        self.regular
    }
}

/// Borrowed proof of two-dimensional triangle-manifold topology.
#[derive(Debug, Clone, Copy)]
pub struct TriangleView<'a> {
    owner: &'a ComplexCore,
    regular: &'a RegularEvidence,
}

impl TriangleView<'_> {
    #[must_use]
    pub const fn owner(&self) -> &ComplexCore {
        self.owner
    }

    #[must_use]
    pub const fn regular(&self) -> RegularView<'_> {
        RegularView {
            owner: self.owner,
            row: self.regular,
        }
    }
}

/// Borrowed proof of coherent top-dimensional orientation.
#[derive(Debug, Clone, Copy)]
pub struct OrientedView<'a> {
    owner: &'a ComplexCore,
}

impl OrientedView<'_> {
    #[must_use]
    pub const fn owner(&self) -> &ComplexCore {
        self.owner
    }
}

/// Borrowed proof of one-skeleton connectivity.
#[derive(Debug, Clone, Copy)]
pub struct ConnectedView<'a> {
    owner: &'a ComplexCore,
}

impl ConnectedView<'_> {
    #[must_use]
    pub const fn owner(&self) -> &ComplexCore {
        self.owner
    }
}

/// Borrowed proof that one connected oriented triangle manifold is a disk.
#[derive(Debug, Clone, Copy)]
pub struct DiskView<'a> {
    owner: &'a ComplexCore,
    regular: &'a RegularEvidence,
}

impl DiskView<'_> {
    #[must_use]
    pub const fn owner(&self) -> &ComplexCore {
        self.owner
    }

    #[must_use]
    pub const fn triangle(&self) -> TriangleView<'_> {
        TriangleView {
            owner: self.owner,
            regular: self.regular,
        }
    }

    /// Boundary vertices in the orientation induced by the coherent face chain.
    ///
    /// The smallest boundary vertex starts the returned cycle. Reversing every
    /// top-dimensional orientation therefore preserves the start and reverses
    /// the remaining order.
    ///
    /// # Errors
    ///
    /// Returns an allocation error or an internal-invariant error if retained
    /// disk evidence and canonical incidence disagree.
    pub fn boundary_vertices(&self) -> Result<Box<[usize]>, TopologyError> {
        let vertex_count = self.owner.vertex_count();
        let edge_basis = self.owner.basis(1)?;
        let mut successor = try_filled(vertex_count, usize::MAX)?;
        let mut boundary_edge_count = 0_usize;

        for (edge, _, coefficient) in self.owner.chain_view().boundary(2)?.exact_entries() {
            if !self
                .regular
                .boundary
                .contains(1, edge, &self.owner.layout)?
            {
                continue;
            }
            let endpoints = edge_basis
                .row(edge)
                .ok_or(TopologyError::InternalInvariant)?;
            let [low, high] = endpoints else {
                return Err(TopologyError::InternalInvariant);
            };
            let (source, target) = match coefficient {
                1 => (*low, *high),
                -1 => (*high, *low),
                _ => return Err(TopologyError::InternalInvariant),
            };
            if successor[source] != usize::MAX {
                return Err(TopologyError::InternalInvariant);
            }
            successor[source] = target;
            boundary_edge_count = boundary_edge_count
                .checked_add(1)
                .ok_or(TopologyError::CountOverflow)?;
        }

        let start = successor
            .iter()
            .position(|&vertex| vertex != usize::MAX)
            .ok_or(TopologyError::InternalInvariant)?;
        let mut cycle = Vec::new();
        cycle
            .try_reserve_exact(boundary_edge_count)
            .map_err(|_| TopologyError::Allocation)?;
        let mut visited = try_filled(vertex_count, false)?;
        let mut current = start;
        for _ in 0..boundary_edge_count {
            if current >= vertex_count || successor[current] == usize::MAX || visited[current] {
                return Err(TopologyError::InternalInvariant);
            }
            visited[current] = true;
            cycle.push(current);
            current = successor[current];
        }
        if current != start || cycle.len() != boundary_edge_count {
            return Err(TopologyError::InternalInvariant);
        }
        Ok(cycle.into_boxed_slice())
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A capability view belonging to one admitted simplicial complex.
pub trait SimplicialCapability: sealed::Sealed {
    #[must_use]
    fn complex(&self) -> &ComplexCore;
}

/// Entailment surface for codimension-one regular capability views.
pub trait CodimensionOneRegularCapability: SimplicialCapability {
    #[must_use]
    fn as_regular(&self) -> RegularView<'_>;
}

/// Triangle-manifold capability entails codimension-one regularity.
pub trait TriangleManifoldCapability: CodimensionOneRegularCapability {}

/// Nonempty-boundary capability entails codimension-one regularity.
pub trait WithBoundaryCapability: CodimensionOneRegularCapability {}

/// Empty-boundary capability entails codimension-one regularity.
pub trait WithoutBoundaryCapability: CodimensionOneRegularCapability {}

/// Orientation is independent of regularity, connectivity, and boundary state.
pub trait OrientedCapability: SimplicialCapability {}

/// Connectivity is independent of regularity, orientation, and boundary state.
pub trait ConnectedCapability: SimplicialCapability {}

impl sealed::Sealed for RegularView<'_> {}
impl sealed::Sealed for TriangleView<'_> {}
impl sealed::Sealed for OrientedView<'_> {}
impl sealed::Sealed for ConnectedView<'_> {}
impl sealed::Sealed for WithBoundaryView<'_> {}
impl sealed::Sealed for WithoutBoundaryView<'_> {}
impl sealed::Sealed for DiskView<'_> {}

impl SimplicialCapability for RegularView<'_> {
    fn complex(&self) -> &ComplexCore {
        self.owner
    }
}

impl CodimensionOneRegularCapability for RegularView<'_> {
    fn as_regular(&self) -> RegularView<'_> {
        *self
    }
}

impl SimplicialCapability for TriangleView<'_> {
    fn complex(&self) -> &ComplexCore {
        self.owner
    }
}

impl CodimensionOneRegularCapability for TriangleView<'_> {
    fn as_regular(&self) -> RegularView<'_> {
        self.regular()
    }
}

impl TriangleManifoldCapability for TriangleView<'_> {}

impl SimplicialCapability for WithBoundaryView<'_> {
    fn complex(&self) -> &ComplexCore {
        self.regular.owner
    }
}

impl CodimensionOneRegularCapability for WithBoundaryView<'_> {
    fn as_regular(&self) -> RegularView<'_> {
        self.regular
    }
}

impl WithBoundaryCapability for WithBoundaryView<'_> {}

impl SimplicialCapability for WithoutBoundaryView<'_> {
    fn complex(&self) -> &ComplexCore {
        self.regular.owner
    }
}

impl CodimensionOneRegularCapability for WithoutBoundaryView<'_> {
    fn as_regular(&self) -> RegularView<'_> {
        self.regular
    }
}

impl WithoutBoundaryCapability for WithoutBoundaryView<'_> {}

impl SimplicialCapability for OrientedView<'_> {
    fn complex(&self) -> &ComplexCore {
        self.owner
    }
}

impl OrientedCapability for OrientedView<'_> {}

impl SimplicialCapability for ConnectedView<'_> {
    fn complex(&self) -> &ComplexCore {
        self.owner
    }
}

impl ConnectedCapability for ConnectedView<'_> {}

impl SimplicialCapability for DiskView<'_> {
    fn complex(&self) -> &ComplexCore {
        self.owner
    }
}

impl CodimensionOneRegularCapability for DiskView<'_> {
    fn as_regular(&self) -> RegularView<'_> {
        RegularView {
            owner: self.owner,
            row: self.regular,
        }
    }
}

impl TriangleManifoldCapability for DiskView<'_> {}
impl WithBoundaryCapability for DiskView<'_> {}
impl OrientedCapability for DiskView<'_> {}
impl ConnectedCapability for DiskView<'_> {}

impl ComplexCore {
    /// Query an already-resolved regularity verdict without initializing it.
    ///
    /// # Errors
    ///
    /// Returns `capability_not_admitted` while unqueried, or the cached
    /// deterministic rejection.
    pub fn require_regular(&self) -> Result<RegularView<'_>, TopologyError> {
        match self
            .evidence
            .regular
            .get()
            .ok_or(TopologyError::capability_not_admitted("regular"))?
        {
            DomainVerdict::Admitted(row) => Ok(RegularView { owner: self, row }),
            DomainVerdict::Rejected(rejection) => Err(rejection.error()),
        }
    }

    /// Query an already-resolved triangle-manifold verdict without initializing it.
    ///
    /// # Errors
    ///
    /// Returns `capability_not_admitted` while unqueried, or the cached
    /// deterministic rejection.
    pub fn require_triangle(&self) -> Result<TriangleView<'_>, TopologyError> {
        let verdict = self
            .evidence
            .triangle
            .get()
            .ok_or(TopologyError::capability_not_admitted("triangle"))?;
        match verdict {
            DomainVerdict::Admitted(()) => {
                let Some(DomainVerdict::Admitted(regular)) = self.evidence.regular.get() else {
                    return Err(TopologyError::InternalInvariant);
                };
                Ok(TriangleView {
                    owner: self,
                    regular,
                })
            }
            DomainVerdict::Rejected(rejection) => Err(rejection.error()),
        }
    }

    /// Query an already-resolved orientation verdict without initializing it.
    ///
    /// # Errors
    ///
    /// Returns `capability_not_admitted` while unqueried, or the cached
    /// deterministic rejection.
    pub fn require_oriented(&self) -> Result<OrientedView<'_>, TopologyError> {
        match self
            .evidence
            .oriented
            .get()
            .ok_or(TopologyError::capability_not_admitted("oriented"))?
        {
            DomainVerdict::Admitted(()) => Ok(OrientedView { owner: self }),
            DomainVerdict::Rejected(rejection) => Err(TopologyError::orientation(
                rejection.codimension_one_simplex,
            )),
        }
    }

    /// Query an already-resolved connectivity verdict without initializing it.
    ///
    /// # Errors
    ///
    /// Returns `capability_not_admitted` while unqueried, or the cached
    /// deterministic rejection.
    pub fn require_connected(&self) -> Result<ConnectedView<'_>, TopologyError> {
        match self
            .evidence
            .connected
            .get()
            .ok_or(TopologyError::capability_not_admitted("connected"))?
        {
            DomainVerdict::Admitted(()) => Ok(ConnectedView { owner: self }),
            DomainVerdict::Rejected(rejection) => {
                Err(TopologyError::disconnected(rejection.unreachable_vertex))
            }
        }
    }

    /// Query the already-resolved global disk verdict without initializing it.
    ///
    /// # Errors
    ///
    /// Returns `capability_not_admitted` while unqueried, or the cached
    /// deterministic rejection.
    pub fn require_disk(&self) -> Result<DiskView<'_>, TopologyError> {
        match self
            .evidence
            .disk
            .get()
            .ok_or(TopologyError::capability_not_admitted("disk"))?
        {
            DomainVerdict::Admitted(()) => {
                let regular = self.require_regular()?;
                Ok(DiskView {
                    owner: self,
                    regular: regular.row,
                })
            }
            DomainVerdict::Rejected(rejection) => Err(rejection.error()),
        }
    }

    /// Query the nonempty-boundary classification from admitted regular evidence.
    ///
    /// # Errors
    ///
    /// Returns an unqueried regularity error, its cached rejection, or
    /// `boundary_absent`.
    pub fn require_with_boundary(&self) -> Result<WithBoundaryView<'_>, TopologyError> {
        self.require_regular()?.with_boundary()
    }

    /// Query the empty-boundary classification from admitted regular evidence.
    ///
    /// # Errors
    ///
    /// Returns an unqueried regularity error, its cached rejection, or
    /// `boundary_present`.
    pub fn require_without_boundary(&self) -> Result<WithoutBoundaryView<'_>, TopologyError> {
        self.require_regular()?.without_boundary()
    }

    /// Initialize or query the canonical regularity verdict.
    ///
    /// # Errors
    ///
    /// Returns the cached deterministic rejection or an uncached operational
    /// failure.
    pub fn refine_regular(&self) -> Result<RegularView<'_>, TopologyError> {
        match self.regular_verdict()? {
            DomainVerdict::Admitted(row) => Ok(RegularView { owner: self, row }),
            DomainVerdict::Rejected(rejection) => Err(rejection.error()),
        }
    }

    /// Initialize or query the canonical triangle-manifold verdict.
    ///
    /// Regularity is always published first.
    ///
    /// # Errors
    ///
    /// Returns the cached deterministic rejection or an uncached operational
    /// failure.
    pub fn refine_triangle(&self) -> Result<TriangleView<'_>, TopologyError> {
        let regular = self.regular_verdict()?;
        let verdict = initialize(&self.evidence.triangle, || {
            #[cfg(test)]
            self.evidence
                .computations
                .triangle
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            match regular {
                DomainVerdict::Admitted(_) => compute_triangle(self),
                DomainVerdict::Rejected(rejection) => Ok(DomainVerdict::Rejected(
                    TriangleRejection::Regular(*rejection),
                )),
            }
        })?;
        match verdict {
            DomainVerdict::Admitted(()) => {
                let regular = match regular {
                    DomainVerdict::Admitted(regular) => regular,
                    DomainVerdict::Rejected(_) => {
                        return Err(TopologyError::InternalInvariant);
                    }
                };
                Ok(TriangleView {
                    owner: self,
                    regular,
                })
            }
            DomainVerdict::Rejected(rejection) => Err(rejection.error()),
        }
    }

    /// Initialize or query the canonical orientation verdict.
    ///
    /// # Errors
    ///
    /// Returns the cached deterministic rejection or an uncached operational
    /// failure.
    pub fn refine_oriented(&self) -> Result<OrientedView<'_>, TopologyError> {
        let verdict = initialize(&self.evidence.oriented, || {
            #[cfg(test)]
            self.evidence
                .computations
                .oriented
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(compute_oriented(self))
        })?;
        match verdict {
            DomainVerdict::Admitted(()) => Ok(OrientedView { owner: self }),
            DomainVerdict::Rejected(rejection) => {
                let _ = rejection.codimension_one_simplex;
                Err(TopologyError::orientation(
                    rejection.codimension_one_simplex,
                ))
            }
        }
    }

    /// Initialize or query the canonical connectivity verdict.
    ///
    /// # Errors
    ///
    /// Returns the cached deterministic rejection or an uncached operational
    /// failure.
    pub fn refine_connected(&self) -> Result<ConnectedView<'_>, TopologyError> {
        let verdict = initialize(&self.evidence.connected, || {
            #[cfg(test)]
            self.evidence
                .computations
                .connected
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            compute_connected(self)
        })?;
        match verdict {
            DomainVerdict::Admitted(()) => Ok(ConnectedView { owner: self }),
            DomainVerdict::Rejected(rejection) => {
                let _ = rejection.unreachable_vertex;
                Err(TopologyError::disconnected(rejection.unreachable_vertex))
            }
        }
    }

    /// Initialize or query the canonical global disk verdict.
    ///
    /// Triangle-manifold, orientation, connectivity, and nonempty-boundary
    /// evidence are admitted first and remain owned by this complex.
    ///
    /// # Errors
    ///
    /// Returns a prerequisite rejection, a cached disk rejection, or an
    /// uncached operational failure.
    pub fn refine_disk(&self) -> Result<DiskView<'_>, TopologyError> {
        let triangle = self.refine_triangle()?;
        self.refine_oriented()?;
        self.refine_connected()?;
        triangle.regular().with_boundary()?;
        let verdict = initialize(&self.evidence.disk, || {
            #[cfg(test)]
            self.evidence
                .computations
                .disk
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            compute_disk(self, triangle.regular())
        })?;
        match verdict {
            DomainVerdict::Admitted(()) => Ok(DiskView {
                owner: self,
                regular: triangle.regular,
            }),
            DomainVerdict::Rejected(rejection) => Err(rejection.error()),
        }
    }

    fn regular_verdict(
        &self,
    ) -> Result<&DomainVerdict<RegularEvidence, RegularRejection>, TopologyError> {
        initialize(&self.evidence.regular, || {
            #[cfg(test)]
            self.evidence
                .computations
                .regular
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            compute_regular(self)
        })
    }

    pub(crate) fn admitted_regular_view(&self) -> Option<RegularView<'_>> {
        match self.evidence.regular.get()? {
            DomainVerdict::Admitted(row) => Some(RegularView { owner: self, row }),
            DomainVerdict::Rejected(_) => None,
        }
    }
}

fn compute_regular(owner: &ComplexCore) -> CheckResult<RegularEvidence, RegularRejection> {
    let layout = &owner.layout;
    let counts = layout.counts();
    let mut boundary =
        PackedDegreeMasksBuilder::empty(layout).map_err(OperationalFailure::internal)?;
    if owner.dimension == 0 {
        return Ok(DomainVerdict::Admitted(RegularEvidence {
            boundary: boundary.finish(),
            kind: BoundaryKind::Empty,
        }));
    }

    let mut used = try_filled(owner.vertex_count, false).map_err(OperationalFailure::internal)?;
    let top = &owner.bases[owner.dimension];
    for simplex in 0..top.row_count() {
        let row = top
            .row(simplex)
            .ok_or(OperationalFailure(TopologyError::InternalInvariant))?;
        for vertex in row {
            used[*vertex] = true;
        }
    }
    if let Some(vertex) = used.iter().position(|used| !used) {
        return Ok(DomainVerdict::Rejected(RegularRejection::NotPure {
            vertex,
        }));
    }

    let codimension = owner.dimension - 1;
    let mut first_boundary = None;
    for (simplex, (cofaces, _)) in owner.boundaries[owner.dimension].storage.rows().enumerate() {
        if !(1..=2).contains(&cofaces.len()) {
            return Ok(DomainVerdict::Rejected(
                RegularRejection::CodimensionOneIncidence {
                    degree: codimension,
                    simplex,
                    cofaces: cofaces.len(),
                },
            ));
        }
        if cofaces.len() == 1 {
            first_boundary.get_or_insert(simplex);
            boundary
                .set(codimension, simplex, true)
                .map_err(OperationalFailure::internal)?;
        }
    }
    for degree in (1..=codimension).rev() {
        for simplex in 0..counts[degree] {
            if !boundary
                .contains(degree, simplex)
                .map_err(OperationalFailure::internal)?
            {
                continue;
            }
            let width = degree
                .checked_add(1)
                .ok_or(OperationalFailure(TopologyError::CountOverflow))?;
            let start = simplex
                .checked_mul(width)
                .ok_or(OperationalFailure(TopologyError::CountOverflow))?;
            for face in owner.immediate_faces[degree]
                .get(start..start + width)
                .ok_or(OperationalFailure(TopologyError::InternalInvariant))?
            {
                boundary
                    .set(degree - 1, *face, true)
                    .map_err(OperationalFailure::internal)?;
            }
        }
    }
    let kind = first_boundary.map_or(BoundaryKind::Empty, |simplex| BoundaryKind::NonEmpty {
        first_codimension_one_simplex: simplex,
    });
    Ok(DomainVerdict::Admitted(RegularEvidence {
        boundary: boundary.finish(),
        kind,
    }))
}

fn compute_triangle(owner: &ComplexCore) -> CheckResult<(), TriangleRejection> {
    if owner.dimension != 2 {
        return Ok(DomainVerdict::Rejected(TriangleRejection::WrongDimension {
            actual: owner.dimension,
        }));
    }
    let triangles = &owner.bases[2];
    let links =
        CenteredLinkEdges::partition(triangles, owner.vertex_count).map_err(
            |error| match error {
                TopologyError::Allocation | TopologyError::CountOverflow => {
                    OperationalFailure(error)
                }
                _ => OperationalFailure(TopologyError::InternalInvariant),
            },
        )?;
    let mut scratch =
        LinkScratch::try_new(owner.vertex_count).map_err(OperationalFailure::internal)?;
    for vertex in 0..owner.vertex_count {
        match scratch.validate(
            links
                .for_center(vertex)
                .map_err(OperationalFailure::internal)?,
        ) {
            Ok(()) => {}
            Err(TopologyError::VertexLink) => {
                return Ok(DomainVerdict::Rejected(TriangleRejection::VertexLink {
                    vertex,
                }));
            }
            Err(error) => return Err(OperationalFailure::internal(error)),
        }
    }
    Ok(DomainVerdict::Admitted(()))
}

fn compute_oriented(owner: &ComplexCore) -> DomainVerdict<(), OrientationRejection> {
    if owner.dimension == 0 {
        return DomainVerdict::Admitted(());
    }
    for (simplex, (_, values)) in owner.boundaries[owner.dimension].storage.rows().enumerate() {
        if values.len() > 2
            || (values.len() == 2 && values.iter().map(|v| i16::from(*v)).sum::<i16>() != 0)
        {
            return DomainVerdict::Rejected(OrientationRejection {
                codimension_one_simplex: simplex,
            });
        }
    }
    DomainVerdict::Admitted(())
}

fn compute_connected(owner: &ComplexCore) -> CheckResult<(), ConnectivityRejection> {
    if owner.dimension == 0 {
        return if owner.vertex_count == 1 {
            Ok(DomainVerdict::Admitted(()))
        } else {
            Ok(DomainVerdict::Rejected(ConnectivityRejection {
                unreachable_vertex: 1,
            }))
        };
    }
    let mut seen = try_filled(owner.vertex_count, false).map_err(OperationalFailure::internal)?;
    let mut pending = Vec::new();
    pending
        .try_reserve_exact(owner.vertex_count)
        .map_err(|_| OperationalFailure(TopologyError::Allocation))?;
    pending.push(0_usize);
    while let Some(vertex) = pending.pop() {
        if seen[vertex] {
            continue;
        }
        seen[vertex] = true;
        let (edges, _) = owner.boundaries[1]
            .storage
            .row(vertex)
            .ok_or(OperationalFailure(TopologyError::InternalInvariant))?;
        for edge in edges.iter().copied() {
            let start = edge
                .checked_mul(2)
                .ok_or(OperationalFailure(TopologyError::CountOverflow))?;
            let endpoints = owner.immediate_faces[1]
                .get(start..start + 2)
                .ok_or(OperationalFailure(TopologyError::InternalInvariant))?;
            for endpoint in endpoints {
                if !seen[*endpoint] {
                    pending.push(*endpoint);
                }
            }
        }
    }
    if let Some(unreachable_vertex) = seen.iter().position(|seen| !seen) {
        Ok(DomainVerdict::Rejected(ConnectivityRejection {
            unreachable_vertex,
        }))
    } else {
        Ok(DomainVerdict::Admitted(()))
    }
}

fn compute_disk(owner: &ComplexCore, regular: RegularView<'_>) -> CheckResult<(), DiskRejection> {
    let mut components =
        DisjointSet::try_new(owner.vertex_count).map_err(OperationalFailure::internal)?;
    let mut present =
        try_filled(owner.vertex_count, false).map_err(OperationalFailure::internal)?;
    let edge_count = owner.bases[1].row_count();
    for edge in 0..edge_count {
        if !regular
            .packed_boundary()
            .contains(1, edge, &owner.layout)
            .map_err(OperationalFailure::internal)?
        {
            continue;
        }
        let endpoints = owner.bases[1]
            .row(edge)
            .ok_or(OperationalFailure(TopologyError::InternalInvariant))?;
        let [left, right] = *endpoints else {
            return Err(OperationalFailure(TopologyError::InternalInvariant));
        };
        present[left] = true;
        present[right] = true;
        components.join(left, right);
    }
    let boundary_components = (0..owner.vertex_count)
        .filter(|vertex| present[*vertex] && components.is_root(*vertex))
        .count();
    if boundary_components != 1 {
        return Ok(DomainVerdict::Rejected(DiskRejection::BoundaryComponents(
            boundary_components,
        )));
    }

    let euler_characteristic =
        owner
            .layout
            .counts()
            .iter()
            .enumerate()
            .fold(0_i128, |value, (degree, count)| {
                if degree.is_multiple_of(2) {
                    value + *count as i128
                } else {
                    value - *count as i128
                }
            });
    if euler_characteristic != 1 {
        return Ok(DomainVerdict::Rejected(DiskRejection::EulerCharacteristic(
            euler_characteristic,
        )));
    }
    Ok(DomainVerdict::Admitted(()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use once_cell::sync::OnceCell;

    use super::{
        CodimensionOneRegularCapability, DomainVerdict, RegularRejection,
        TriangleManifoldCapability, initialize,
    };
    use crate::{CandidateInput, ComplexCore, TopologyError};

    fn admit(rows: &[[i128; 3]]) -> Arc<ComplexCore> {
        let candidate =
            CandidateInput::signed(rows.iter().flatten().copied(), rows.len(), 3, None).unwrap();
        ComplexCore::admit(candidate).unwrap()
    }

    #[test]
    fn concurrent_deterministic_rejection_publishes_one_counterexample() {
        let owner = admit(&[[0, 1, 2], [1, 0, 3], [0, 1, 4]]);
        let workers = (0..8)
            .map(|_| {
                let owner = Arc::clone(&owner);
                std::thread::spawn(move || owner.refine_regular().unwrap_err())
            })
            .collect::<Vec<_>>();

        for worker in workers {
            assert_eq!(
                worker.join().unwrap(),
                TopologyError::codimension_one_incidence(1, 0, 3)
            );
        }
        assert_eq!(
            owner.evidence.computations.regular.load(Ordering::Relaxed),
            1
        );
        assert!(matches!(
            owner.evidence.regular.get(),
            Some(DomainVerdict::Rejected(
                RegularRejection::CodimensionOneIncidence {
                    degree: 1,
                    simplex: 0,
                    cofaces: 3
                }
            ))
        ));
    }

    #[test]
    fn concurrent_success_publishes_each_queried_row_once() {
        let owner = admit(&[[0, 1, 2], [0, 2, 3]]);
        let workers = (0..8)
            .map(|_| {
                let owner = Arc::clone(&owner);
                std::thread::spawn(move || {
                    owner.refine_disk().unwrap();
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }

        let counts = &owner.evidence.computations;
        assert_eq!(counts.regular.load(Ordering::Relaxed), 1);
        assert_eq!(counts.triangle.load(Ordering::Relaxed), 1);
        assert_eq!(counts.oriented.load(Ordering::Relaxed), 1);
        assert_eq!(counts.connected.load(Ordering::Relaxed), 1);
        assert_eq!(counts.disk.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn operational_failure_is_not_published_and_can_retry() {
        let cell = OnceCell::<DomainVerdict<(), ()>>::new();
        let attempts = AtomicUsize::new(0);

        let first = initialize(&cell, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Err(super::OperationalFailure(TopologyError::Allocation))
        });
        assert_eq!(first.unwrap_err(), TopologyError::Allocation);
        assert!(cell.get().is_none());

        let admitted = initialize(&cell, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Ok(DomainVerdict::Admitted(()))
        })
        .unwrap();
        assert!(matches!(admitted, DomainVerdict::Admitted(())));
        initialize(&cell, || unreachable!()).unwrap();
        assert_eq!(attempts.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn triangle_view_entails_same_owner_regular_view_without_recomputation() {
        fn require_regular(value: &impl CodimensionOneRegularCapability) -> usize {
            value.as_regular().owner().vertex_count()
        }
        fn require_triangle(value: &impl TriangleManifoldCapability) -> usize {
            value.complex().dimension()
        }

        let owner = admit(&[[0, 1, 2], [0, 2, 3]]);
        let triangle = owner.refine_triangle().unwrap();
        assert_eq!(require_regular(&triangle), 4);
        assert_eq!(require_triangle(&triangle), 2);
        assert!(std::ptr::eq(triangle.regular().owner(), owner.as_ref()));
        assert_eq!(
            owner.evidence.computations.regular.load(Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn orthogonal_query_order_does_not_change_identity_or_repeat_work() {
        let owner = admit(&[[0, 1, 2], [0, 2, 3]]);
        let connected = owner.refine_connected().unwrap();
        let oriented = owner.refine_oriented().unwrap();
        owner.refine_oriented().unwrap();
        owner.refine_connected().unwrap();

        assert!(std::ptr::eq(connected.owner(), oriented.owner()));
        assert_eq!(
            owner
                .evidence
                .computations
                .connected
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            owner.evidence.computations.oriented.load(Ordering::Relaxed),
            1
        );
    }
}
