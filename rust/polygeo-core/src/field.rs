use std::{fmt, mem::size_of, num::NonZeroU32, sync::Arc};

use faer::{
    Mat, MatMut,
    dyn_stack::{MemBuffer, MemStack},
    sparse::Triplet,
};

use num_bigint::BigInt;
use num_traits::ToPrimitive;
use once_cell::sync::OnceCell;

use crate::numeric::adaptive_product_value;
use crate::solve_impl::{
    SolveExt, check_cancelled, checked_cells, checked_sum, checked_work_product, cubic_work,
    factor_dense_square, factor_scale, factor_solve_requirement, factor_sparse_triplets,
    logical_bytes, logical_f64, matrix_bytes, require_stable_dense_lu, require_storage,
    require_work, solve_factor, sparse_phase_policy_bytes,
};
use crate::surface::{
    SurfaceError, TriangleSurface, compensated_add, cross, dot, norm, normalize, product_sum, row3,
    subtract,
};
use crate::{
    Binary64Chain, Binary64Cochain, Binary64Element, Binary64Space, CancellationToken,
    CanonicalBoundary, Chain, Cochain, ComplexCore, Executor, FaceVectors, HomologyGroup,
    IntegralCochain, IntegralDualCycleBasis, Metric, NondegenerateCapability, PairingCapability,
    Policy, SolveError, StorageLimit, SurfaceComputationError, WorkLimit,
};

impl TriangleSurface {
    /// Borrow the first canonical tangent axis for every face.
    ///
    /// # Errors
    ///
    /// Returns an unrepresentable frame failure.
    pub fn first_frame_axes(&self) -> Result<&[f64], SurfaceError> {
        Ok(&self.surface_rows()?.first)
    }

    /// Borrow the second canonical tangent axis for every face.
    ///
    /// # Errors
    ///
    /// Returns an unrepresentable frame failure.
    pub fn second_frame_axes(&self) -> Result<&[f64], SurfaceError> {
        Ok(&self.surface_rows()?.second)
    }

    pub(crate) fn boundary_power_directions(
        &self,
        symmetry_order: NonZeroU32,
        boundary_angle_offset: f64,
    ) -> Result<(Vec<usize>, Vec<f64>), SurfaceError> {
        if !boundary_angle_offset.is_finite() {
            return Err(SurfaceError::NonFinite);
        }
        self.realization()
            .topology()
            .refine_regular()?
            .with_boundary()?;
        let (_, component_edges) = oriented_boundary_components(self.realization().topology())?;
        boundary_face_power_directions(
            self,
            symmetry_order,
            boundary_angle_offset,
            &component_edges,
        )
    }

    /// Construct canonical Levi-Civita face transport.
    ///
    /// # Errors
    ///
    /// Requires a connected regular surface and representable frames.
    pub fn levi_civita_connection(
        self: &Arc<Self>,
    ) -> Result<Arc<SurfaceConnection>, SurfaceError> {
        self.realization().topology().refine_regular()?;
        self.realization().topology().refine_connected()?;
        let dual = dual_edges(self.realization().topology())?;
        let interior_edges: Arc<[usize]> = (0..dual.edge_count)
            .filter(|&edge| dual.incidence_count(edge) == 2)
            .collect::<Vec<_>>()
            .into();
        let edges = self.realization().topology().basis(1)?;
        let first = self.first_frame_axes()?;
        let second = self.second_frame_axes()?;
        let mut transports = Vec::with_capacity(2 * interior_edges.len());
        for &edge in interior_edges.iter() {
            let endpoints = edges.row(edge).ok_or(SurfaceError::IndexOutside)?;
            let axis = normalize(subtract(
                self.point(endpoints[1])?,
                self.point(endpoints[0])?,
            ))
            .ok_or(SurfaceError::Unrepresentable)?;
            let (source, target, _) = dual.edge(edge)?;
            let source_normal = cross(row3(first, source)?, row3(second, source)?);
            let target_normal = cross(row3(first, target)?, row3(second, target)?);
            let angle = dot(axis, cross(source_normal, target_normal))
                .atan2(dot(source_normal, target_normal));
            let rotated = rodrigues(row3(first, source)?, axis, angle);
            transports.extend(normalize_complex([
                dot(rotated, row3(first, target)?),
                dot(rotated, row3(second, target)?),
            ])?);
        }
        Ok(Arc::new(SurfaceConnection {
            surface: Arc::clone(self),
            symmetry_order: NonZeroU32::MIN,
            interior_edges,
            transports: transports.into(),
            evidence: OnceCell::new(),
        }))
    }

    /// Construct order-`N` Levi-Civita power transport with one deviation per interior dual edge.
    ///
    /// # Errors
    ///
    /// Rejects irregularity, disconnection, shape, nonfinite, or representation failures.
    pub fn connection(
        self: &Arc<Self>,
        symmetry_order: NonZeroU32,
        deviations: &[f64],
    ) -> Result<Arc<SurfaceConnection>, SurfaceError> {
        self.levi_civita_connection()?
            .with_powered_deviations(symmetry_order, deviations)
    }
}

fn oriented_boundary_edge(
    edge_basis: &crate::Basis,
    boundary: &CanonicalBoundary,
    edge: usize,
) -> Result<Option<(usize, usize, usize)>, SurfaceError> {
    let start = *boundary
        .indptr()
        .get(edge)
        .ok_or(SurfaceError::IndexOutside)?;
    let stop = *boundary
        .indptr()
        .get(edge + 1)
        .ok_or(SurfaceError::IndexOutside)?;
    if stop - start != 1 {
        return Ok(None);
    }
    let endpoints = edge_basis.row(edge).ok_or(SurfaceError::IndexOutside)?;
    let [low, high] = endpoints else {
        return Err(SurfaceError::Unrepresentable);
    };
    let (source, target) = match boundary.data()[start] {
        1 => (*low, *high),
        -1 => (*high, *low),
        _ => return Err(SurfaceError::Unrepresentable),
    };
    Ok(Some((source, target, boundary.indices()[start])))
}

fn oriented_boundary_components(
    topology: &ComplexCore,
) -> Result<(Vec<usize>, Vec<usize>), SurfaceError> {
    let edge_basis = topology.basis(1)?;
    let boundary = topology.boundary(2)?;
    let mut edge_at_source = vec![usize::MAX; topology.vertex_count()];
    let mut incoming = vec![false; topology.vertex_count()];
    let mut boundary_edge_count = 0_usize;
    for edge in 0..edge_basis.row_count() {
        let Some((source, target, _)) = oriented_boundary_edge(edge_basis, boundary, edge)? else {
            continue;
        };
        if edge_at_source[source] != usize::MAX || incoming[target] {
            return Err(SurfaceError::Unrepresentable);
        }
        edge_at_source[source] = edge;
        incoming[target] = true;
        boundary_edge_count = boundary_edge_count
            .checked_add(1)
            .ok_or(SurfaceError::Overflow)?;
    }

    let mut offsets = vec![0_usize];
    let mut edges = Vec::with_capacity(boundary_edge_count);
    let mut visited = vec![false; edge_basis.row_count()];
    for start in 0..topology.vertex_count() {
        let first_edge = edge_at_source[start];
        if first_edge == usize::MAX || visited[first_edge] {
            continue;
        }
        let mut vertex = start;
        loop {
            let edge = edge_at_source[vertex];
            if edge == usize::MAX || visited[edge] {
                return Err(SurfaceError::Unrepresentable);
            }
            visited[edge] = true;
            edges.push(edge);
            let (_, target, _) = oriented_boundary_edge(edge_basis, boundary, edge)?
                .ok_or(SurfaceError::Unrepresentable)?;
            vertex = target;
            if vertex == start {
                break;
            }
        }
        offsets.push(edges.len());
    }
    if edges.len() != boundary_edge_count {
        return Err(SurfaceError::Unrepresentable);
    }
    Ok((offsets, edges))
}

fn boundary_edge_geometry(
    surface: &TriangleSurface,
    edge: usize,
) -> Result<(usize, [f64; 3], f64), SurfaceError> {
    let topology = surface.realization().topology();
    let (source, target, face) =
        oriented_boundary_edge(topology.basis(1)?, topology.boundary(2)?, edge)?
            .ok_or(SurfaceError::Unrepresentable)?;
    let displacement = subtract(surface.point(target)?, surface.point(source)?);
    let length = norm(displacement);
    let tangent = normalize(displacement).ok_or(SurfaceError::Unrepresentable)?;
    if !length.is_finite() {
        return Err(SurfaceError::Unrepresentable);
    }
    Ok((face, tangent, length))
}

fn boundary_edge_power_direction(
    surface: &TriangleSurface,
    symmetry_order: NonZeroU32,
    edge: usize,
) -> Result<(usize, [f64; 2], f64), SurfaceError> {
    let (face, tangent, length) = boundary_edge_geometry(surface, edge)?;
    let direction = complex_power(
        normalize_complex([
            dot(tangent, row3(surface.first_frame_axes()?, face)?),
            dot(tangent, row3(surface.second_frame_axes()?, face)?),
        ])?,
        i64::from(symmetry_order.get()),
    )?;
    Ok((face, direction, length))
}

#[derive(Clone, Copy, Debug, Default)]
struct BoundaryTargetAccumulator {
    length_scale: f64,
    sum: [f64; 2],
    correction: [f64; 2],
    weight_sum: f64,
    term_count: u32,
}

fn boundary_face_power_directions(
    surface: &TriangleSurface,
    symmetry_order: NonZeroU32,
    boundary_angle_offset: f64,
    edges: &[usize],
) -> Result<(Vec<usize>, Vec<f64>), SurfaceError> {
    let mut accumulators = vec![BoundaryTargetAccumulator::default(); surface.face_count()];
    for &edge in edges {
        let (face, _, length) = boundary_edge_geometry(surface, edge)?;
        accumulators[face].length_scale = accumulators[face].length_scale.max(length);
    }

    for &edge in edges {
        let (face, direction, length) =
            boundary_edge_power_direction(surface, symmetry_order, edge)?;
        let accumulator = &mut accumulators[face];
        let weight = length / accumulator.length_scale;
        for (axis, direction) in direction.into_iter().enumerate() {
            compensated_add(
                &mut accumulator.sum[axis],
                &mut accumulator.correction[axis],
                weight * direction,
            )?;
        }
        accumulator.weight_sum += weight;
        accumulator.term_count = accumulator
            .term_count
            .checked_add(1)
            .ok_or(SurfaceError::Overflow)?;
    }

    let order = f64::from(symmetry_order.get());
    let offset_angle = boundary_angle_offset.rem_euclid(std::f64::consts::TAU / order) * order;
    let offset = [offset_angle.cos(), offset_angle.sin()];
    let mut faces = Vec::new();
    let mut power_directions = Vec::new();
    for (face, accumulator) in accumulators.into_iter().enumerate() {
        if accumulator.term_count == 0 {
            continue;
        }
        let sum = [
            product_sum([(accumulator.sum[0], 1.0), (accumulator.correction[0], 1.0)])?,
            product_sum([(accumulator.sum[1], 1.0), (accumulator.correction[1], 1.0)])?,
        ];
        let magnitude = sum[0].hypot(sum[1]);
        let operation_count = f64::from(accumulator.term_count.saturating_add(2));
        let limit = 64.0 * f64::EPSILON * operation_count * accumulator.weight_sum;
        if !magnitude.is_finite() || magnitude <= limit {
            return Err(SurfaceError::Unrepresentable);
        }
        faces.push(face);
        power_directions.extend(complex_multiply(normalize_complex(sum)?, offset));
    }
    Ok((faces, power_directions))
}

fn rodrigues(value: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let cosine = angle.cos();
    let sine = angle.sin();
    let parallel = dot(axis, value) * (1.0 - cosine);
    let crossed = cross(axis, value);
    [
        value[0] * cosine + crossed[0] * sine + axis[0] * parallel,
        value[1] * cosine + crossed[1] * sine + axis[1] * parallel,
        value[2] * cosine + crossed[2] * sine + axis[2] * parallel,
    ]
}

pub(crate) fn complex_multiply(left: [f64; 2], right: [f64; 2]) -> [f64; 2] {
    [
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    ]
}

pub(crate) fn normalize_complex(value: [f64; 2]) -> Result<[f64; 2], SurfaceError> {
    let magnitude = value[0].hypot(value[1]);
    if magnitude == 0.0 || !magnitude.is_finite() {
        return Err(SurfaceError::Unrepresentable);
    }
    let value = [value[0] / magnitude, value[1] / magnitude];
    value
        .into_iter()
        .all(f64::is_finite)
        .then_some(value)
        .ok_or(SurfaceError::Unrepresentable)
}

pub(crate) fn complex_conjugate(value: [f64; 2]) -> [f64; 2] {
    [value[0], -value[1]]
}

pub(crate) fn complex_power(mut value: [f64; 2], exponent: i64) -> Result<[f64; 2], SurfaceError> {
    if exponent < 0 {
        value = complex_conjugate(value);
    }
    let mut result = [1.0, 0.0];
    let mut remaining = exponent.unsigned_abs();
    while remaining != 0 {
        if remaining & 1 != 0 {
            result = normalize_complex(complex_multiply(result, value))?;
        }
        remaining >>= 1;
        if remaining != 0 {
            value = normalize_complex(complex_multiply(value, value))?;
        }
    }
    Ok(result)
}

pub(crate) fn powered_transport(
    base: [f64; 2],
    order: NonZeroU32,
    deviation: f64,
) -> Result<[f64; 2], SurfaceError> {
    normalize_complex(complex_multiply(
        complex_power(base, i64::from(order.get()))?,
        [deviation.cos(), deviation.sin()],
    ))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DualEdges<'a> {
    boundary: &'a CanonicalBoundary,
    edge_count: usize,
}

impl DualEdges<'_> {
    fn incidence_count(self, edge: usize) -> usize {
        self.boundary.indptr()[edge + 1] - self.boundary.indptr()[edge]
    }

    pub(crate) fn edge(self, edge: usize) -> Result<(usize, usize, i8), SurfaceError> {
        if edge >= self.edge_count {
            return Err(SurfaceError::IndexOutside);
        }
        let start = self.boundary.indptr()[edge];
        let stop = self.boundary.indptr()[edge + 1];
        if stop - start != 2 {
            return Err(SurfaceError::BoundaryPresent);
        }
        let entries = [
            (self.boundary.indices()[start], self.boundary.data()[start]),
            (
                self.boundary.indices()[start + 1],
                self.boundary.data()[start + 1],
            ),
        ];
        let (source, target) = if entries[0].0 < entries[1].0 {
            (entries[0], entries[1])
        } else {
            (entries[1], entries[0])
        };
        Ok((source.0, target.0, source.1))
    }
}

pub(crate) fn dual_edges(topology: &ComplexCore) -> Result<DualEdges<'_>, SurfaceError> {
    let boundary = topology.boundary(2)?;
    let edge_count = topology.basis(1)?.row_count();
    let dual = DualEdges {
        boundary,
        edge_count,
    };
    for edge in 0..edge_count {
        if !matches!(dual.incidence_count(edge), 1 | 2) {
            return Err(SurfaceError::Unrepresentable);
        }
    }
    Ok(dual)
}

/// Compact circular holonomy error bounds for one connection.
#[derive(Clone, Copy, Debug, PartialEq)]

pub struct HolonomyEvidence {
    local_error: f64,
    generator_error: f64,
    limit: f64,
}

/// Exact symmetric-field singularities admitted from binary64 angle evidence.
#[derive(Clone, Debug)]
pub struct DirectionFieldSingularities {
    symmetry_order: NonZeroU32,
    charges: IntegralCochain,
    boundary_turns: Box<[BigInt]>,
    maximum_quantization_residual: f64,
    residual_limit: f64,
}

impl DirectionFieldSingularities {
    #[must_use]
    pub const fn symmetry_order(&self) -> NonZeroU32 {
        self.symmetry_order
    }

    #[must_use]
    pub const fn charges(&self) -> &IntegralCochain {
        &self.charges
    }

    #[must_use]
    pub fn boundary_turns(&self) -> &[BigInt] {
        &self.boundary_turns
    }

    #[must_use]
    pub const fn maximum_quantization_residual(&self) -> f64 {
        self.maximum_quantization_residual
    }

    #[must_use]
    pub const fn residual_limit(&self) -> f64 {
        self.residual_limit
    }
}

impl HolonomyEvidence {
    #[must_use]
    pub const fn local_error(self) -> f64 {
        self.local_error
    }

    #[must_use]
    pub const fn generator_error(self) -> f64 {
        self.generator_error
    }

    #[must_use]
    pub const fn limit(self) -> f64 {
        self.limit
    }
}

#[derive(Debug)]
struct IntegrabilityEvidence {
    phases: Arc<[f64]>,
    crossing_error: f64,
}

/// One normalized unit-complex transport per selected canonical interior dual edge.
pub struct SurfaceConnection {
    surface: Arc<TriangleSurface>,
    symmetry_order: NonZeroU32,
    pub(crate) interior_edges: Arc<[usize]>,
    transports: Arc<[f64]>,
    evidence: OnceCell<Result<IntegrabilityEvidence, SurfaceError>>,
}

impl fmt::Debug for SurfaceConnection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SurfaceConnection")
            .field("symmetry_order", &self.symmetry_order)
            .field("edges", &(self.transports.len() / 2))
            .finish_non_exhaustive()
    }
}

impl SurfaceConnection {
    #[must_use]
    pub const fn surface(&self) -> &Arc<TriangleSurface> {
        &self.surface
    }

    #[must_use]
    pub const fn symmetry_order(&self) -> NonZeroU32 {
        self.symmetry_order
    }

    #[must_use]
    pub fn transports(&self) -> &[f64] {
        &self.transports
    }

    /// Borrow the canonical primal-edge index of every compact transport row.
    #[must_use]
    pub fn interior_edge_indices(&self) -> &[usize] {
        &self.interior_edges
    }

    pub(crate) fn with_powered_deviations(
        self: &Arc<Self>,
        symmetry_order: NonZeroU32,
        deviations: &[f64],
    ) -> Result<Arc<Self>, SurfaceError> {
        if deviations.len() != self.interior_edges.len() {
            return Err(SurfaceError::FieldShape);
        }
        let mut transports = Vec::with_capacity(self.transports.len());
        for (row, &deviation) in deviations.iter().enumerate() {
            transports.extend(powered_transport(
                complex_at(&self.transports, row)?,
                symmetry_order,
                deviation,
            )?);
        }
        Ok(Arc::new(Self {
            surface: Arc::clone(&self.surface),
            symmetry_order,
            interior_edges: Arc::clone(&self.interior_edges),
            transports: transports.into(),
            evidence: OnceCell::new(),
        }))
    }

    fn transport(&self, edge: usize) -> Result<[f64; 2], SurfaceError> {
        let row = self
            .interior_edges
            .binary_search(&edge)
            .map_err(|_| SurfaceError::IndexOutside)?;
        complex_at(&self.transports, row)
    }

    fn admitted_evidence(&self) -> Result<&IntegrabilityEvidence, SurfaceError> {
        self.evidence
            .get()
            .and_then(|evidence| evidence.as_ref().ok())
            .ok_or(SurfaceError::NotIntegrable)
    }

    /// Compute local and primitive-generator circular holonomy errors.
    ///
    /// # Errors
    ///
    /// Rejects a cycle basis from another topology owner.
    pub fn holonomy(
        &self,
        cycles: &IntegralDualCycleBasis,
    ) -> Result<HolonomyEvidence, SurfaceError> {
        self.surface.require_closed()?;
        if !cycles
            .chain_complex()
            .same_owner(&self.surface.realization().topology().chain_complex())
        {
            return Err(SurfaceError::OwnerMismatch);
        }
        let dual = dual_edges(self.surface.realization().topology())?;
        let local_error = local_holonomy_error(self, dual)?;
        let mut generator_error = 0.0_f64;
        for index in 0..cycles.rank() {
            let cycle = cycles.cocycle(index).ok_or(SurfaceError::IndexOutside)?;
            let mut product = [1.0, 0.0];
            for (&edge, coefficient) in cycle.indices().iter().zip(cycle.coefficients()) {
                let coefficient = coefficient.to_i64().ok_or(SurfaceError::Unrepresentable)?;
                let (_, _, source_sign) = dual.edge(edge)?;
                let exponent = -i64::from(source_sign) * coefficient;
                product = normalize_complex(complex_multiply(
                    product,
                    complex_power(self.transport(edge)?, exponent)?,
                ))?;
            }
            generator_error = generator_error.max(product[1].atan2(product[0]).abs());
        }
        let edge_count =
            u32::try_from(self.surface.edge_count().max(1)).map_err(|_| SurfaceError::Overflow)?;
        Ok(HolonomyEvidence {
            local_error,
            generator_error,
            limit: 128.0 * f64::EPSILON * f64::from(edge_count),
        })
    }

    /// Refine an integrable connection by exhaustive dual-graph propagation.
    ///
    /// # Errors
    ///
    /// Rejects a disconnected dual graph or crossing residual above the fixed limit.
    pub fn require_integrable(self: &Arc<Self>) -> Result<IntegrableConnection, SurfaceError> {
        let evidence = self.evidence.get_or_init(|| {
            let (phases, crossing_error) = propagate_phases(self)?;
            let edge_count = u32::try_from(self.interior_edges.len().max(1))
                .map_err(|_| SurfaceError::Overflow)?;
            let limit = 128.0 * f64::EPSILON * f64::from(edge_count);
            if crossing_error > limit {
                return Err(SurfaceError::NotIntegrable);
            }
            Ok(IntegrabilityEvidence {
                phases: phases.into(),
                crossing_error,
            })
        });
        evidence.as_ref().map_err(|error| *error)?;
        Ok(IntegrableConnection {
            connection: Arc::clone(self),
        })
    }
}

pub(crate) fn complex_at(values: &[f64], index: usize) -> Result<[f64; 2], SurfaceError> {
    let start = index.checked_mul(2).ok_or(SurfaceError::Overflow)?;
    values
        .get(start..start + 2)
        .and_then(|value| value.try_into().ok())
        .ok_or(SurfaceError::IndexOutside)
}

fn local_holonomy_error(
    connection: &SurfaceConnection,
    dual: DualEdges<'_>,
) -> Result<f64, SurfaceError> {
    let topology = connection.surface.realization().topology();
    let incidence = topology.boundary(1)?;
    let mut maximum = 0.0_f64;
    for vertex in 0..topology.vertex_count() {
        let start = incidence.indptr()[vertex];
        let stop = incidence.indptr()[vertex + 1];
        let mut product = [1.0, 0.0];
        for (&edge, &incidence_sign) in incidence.indices()[start..stop]
            .iter()
            .zip(&incidence.data()[start..stop])
        {
            let (_, _, source_sign) = dual.edge(edge)?;
            let exponent = -i64::from(source_sign) * i64::from(incidence_sign);
            product = normalize_complex(complex_multiply(
                product,
                complex_power(connection.transport(edge)?, exponent)?,
            ))?;
        }
        maximum = maximum.max(product[1].atan2(product[0]).abs());
    }
    Ok(maximum)
}

fn propagate_phases(connection: &SurfaceConnection) -> Result<(Vec<f64>, f64), SurfaceError> {
    let dual = dual_edges(connection.surface.realization().topology())?;
    let face_count = connection.surface.face_count();
    let mut adjacency = vec![Vec::new(); face_count];
    for (row, &edge) in connection.interior_edges.iter().enumerate() {
        let (source, target, _) = dual.edge(edge)?;
        adjacency[source].push((target, row, true));
        adjacency[target].push((source, row, false));
    }
    let mut phases = vec![0.0; 2 * face_count];
    phases[0] = 1.0;
    let mut visited = vec![false; face_count];
    visited[0] = true;
    let mut pending = vec![0_usize];
    let mut cursor = 0;
    while cursor < pending.len() {
        let face = pending[cursor];
        cursor += 1;
        for &(neighbor, row, forward) in &adjacency[face] {
            if visited[neighbor] {
                continue;
            }
            let transport = complex_at(&connection.transports, row)?;
            let phase = complex_at(&phases, face)?;
            let next = normalize_complex(complex_multiply(
                if forward {
                    transport
                } else {
                    complex_conjugate(transport)
                },
                phase,
            ))?;
            phases[2 * neighbor..2 * neighbor + 2].copy_from_slice(&next);
            visited[neighbor] = true;
            pending.push(neighbor);
        }
    }
    if visited.iter().any(|visited| !visited) {
        return Err(SurfaceError::Unrepresentable);
    }
    let mut maximum = 0.0_f64;
    for (row, &edge) in connection.interior_edges.iter().enumerate() {
        let (source, target, _) = dual.edge(edge)?;
        let expected = complex_multiply(
            complex_at(&connection.transports, row)?,
            complex_at(&phases, source)?,
        );
        let residual = complex_multiply(complex_at(&phases, target)?, complex_conjugate(expected));
        maximum = maximum.max(residual[1].atan2(residual[0]).abs());
    }
    Ok((phases, maximum))
}

/// Arc-only evidence view for one admitted integrable connection.
#[derive(Clone, Debug)]

pub struct IntegrableConnection {
    connection: Arc<SurfaceConnection>,
}

impl IntegrableConnection {
    #[must_use]
    pub const fn connection(&self) -> &Arc<SurfaceConnection> {
        &self.connection
    }

    /// Integrate one unit-complex face direction from face zero.
    ///
    /// # Errors
    ///
    /// Rejects a nonfinite anchor or unavailable admitted evidence.
    pub fn direction_field(&self, anchor_angle: f64) -> Result<FaceDirectionField, SurfaceError> {
        if !anchor_angle.is_finite() {
            return Err(SurfaceError::NonFinite);
        }
        let evidence = self.connection.admitted_evidence()?;
        let power_anchor = f64::from(self.connection.symmetry_order.get()) * anchor_angle;
        let anchor = [power_anchor.cos(), power_anchor.sin()];
        let mut power_directions = Vec::with_capacity(evidence.phases.len());
        for phase in evidence.phases.chunks_exact(2) {
            power_directions.extend(normalize_complex(complex_multiply(
                phase
                    .try_into()
                    .map_err(|_| SurfaceError::Unrepresentable)?,
                anchor,
            ))?);
        }
        Ok(FaceDirectionField {
            integrable: self.clone(),
            power_directions: power_directions.into(),
        })
    }

    /// Borrow the cached crossing residual that admitted this refinement.
    ///
    /// # Errors
    ///
    /// Returns an internal admission failure if the evidence is unavailable.
    pub fn crossing_error(&self) -> Result<f64, SurfaceError> {
        self.connection
            .admitted_evidence()
            .map(|evidence| evidence.crossing_error)
    }
}

/// Unit-complex directions in the canonical face frames of one connection.
#[derive(Clone, Debug)]
pub struct FaceDirectionField {
    integrable: IntegrableConnection,
    power_directions: Arc<[f64]>,
}

fn interior_direction_charges(
    field: &FaceDirectionField,
    levi_civita: &SurfaceConnection,
    boundary_vertices: &[bool],
    entries: &mut Vec<(usize, BigInt)>,
) -> Result<(BigInt, f64, usize), SurfaceError> {
    let surface = &field.integrable.connection.surface;
    let topology = surface.realization().topology();
    let curvature = surface.gaussian_curvature_measure()?;
    let incidence = topology.boundary(1)?;
    let dual = dual_edges(topology)?;
    let order = f64::from(field.symmetry_order().get());
    let mut total = BigInt::from(0);
    let mut maximum_residual = 0.0_f64;
    let (mut maximum_valence, mut terms) = (0_usize, Vec::new());
    for vertex in 0..topology.vertex_count() {
        if *boundary_vertices
            .get(vertex)
            .ok_or(SurfaceError::IndexOutside)?
        {
            continue;
        }
        let start = incidence.indptr()[vertex];
        let stop = incidence.indptr()[vertex + 1];
        maximum_valence = maximum_valence.max(stop - start);
        terms.clear();
        terms
            .try_reserve(stop - start + 1)
            .map_err(|_| SurfaceError::Overflow)?;
        terms.push((order, curvature.coefficients()[vertex]));
        for (&edge, &incidence_sign) in incidence.indices()[start..stop]
            .iter()
            .zip(&incidence.data()[start..stop])
        {
            let (source, target, source_sign) = dual.edge(edge)?;
            let expected = complex_multiply(
                powered_transport(levi_civita.transport(edge)?, field.symmetry_order(), 0.0)?,
                complex_at(&field.power_directions, source)?,
            );
            let mismatch = complex_multiply(
                complex_at(&field.power_directions, target)?,
                complex_conjugate(expected),
            );
            let traversal = -i64::from(source_sign) * i64::from(incidence_sign);
            let traversal = traversal.to_f64().ok_or(SurfaceError::Unrepresentable)?;
            terms.push((-traversal, mismatch[1].atan2(mismatch[0])));
        }
        let (numerator, _) =
            adaptive_product_value(terms.iter().copied()).ok_or(SurfaceError::Unrepresentable)?;
        let raw = numerator / std::f64::consts::TAU;
        let rounded = raw.round();
        let charge = rounded.to_i64().ok_or(SurfaceError::Unrepresentable)?;
        maximum_residual = maximum_residual.max((raw - rounded).abs());
        if charge != 0 {
            entries.push((vertex, BigInt::from(charge)));
        }
        total += charge;
    }
    Ok((total, maximum_residual, maximum_valence))
}

fn relative_boundary_turns(
    field: &FaceDirectionField,
) -> Result<(Vec<BigInt>, BigInt, f64, usize), SurfaceError> {
    let surface = &field.integrable.connection.surface;
    let (offsets, edges) = oriented_boundary_components(surface.realization().topology())?;
    let mut residuals = Vec::with_capacity(2 * edges.len());
    for &edge in &edges {
        let (face, tangent_power, _) =
            boundary_edge_power_direction(surface, field.symmetry_order(), edge)?;
        residuals.extend(normalize_complex(complex_multiply(
            complex_at(&field.power_directions, face)?,
            complex_conjugate(tangent_power),
        ))?);
    }

    let order = f64::from(field.symmetry_order().get());
    let antipodal_limit = 256.0 * f64::EPSILON * order;
    let mut turns = Vec::with_capacity(offsets.len().saturating_sub(1));
    let mut total = BigInt::from(0);
    let mut maximum_residual = 0.0_f64;
    let (mut maximum_edges, mut terms) = (0_usize, Vec::new());
    for component in offsets.windows(2) {
        let start = component[0];
        let stop = component[1];
        if start == stop {
            return Err(SurfaceError::Unrepresentable);
        }
        maximum_edges = maximum_edges.max(stop - start);
        terms.clear();
        terms
            .try_reserve(stop - start)
            .map_err(|_| SurfaceError::Overflow)?;
        for row in start..stop {
            let next = if row + 1 == stop { start } else { row + 1 };
            let increment = normalize_complex(complex_multiply(
                complex_at(&residuals, next)?,
                complex_conjugate(complex_at(&residuals, row)?),
            ))?;
            let angle = increment[1].atan2(increment[0]);
            if std::f64::consts::PI - angle.abs() <= antipodal_limit {
                return Err(SurfaceError::Unrepresentable);
            }
            terms.push((1.0, angle));
        }
        let (numerator, _) =
            adaptive_product_value(terms.iter().copied()).ok_or(SurfaceError::Unrepresentable)?;
        let raw = numerator / std::f64::consts::TAU;
        let rounded = raw.round();
        let turn = rounded.to_i64().ok_or(SurfaceError::Unrepresentable)?;
        maximum_residual = maximum_residual.max((raw - rounded).abs());
        turns.push(BigInt::from(turn));
        total += turn;
    }
    Ok((turns, total, maximum_residual, maximum_edges))
}

impl FaceDirectionField {
    #[must_use]
    pub const fn connection(&self) -> &IntegrableConnection {
        &self.integrable
    }

    #[must_use]
    pub fn symmetry_order(&self) -> NonZeroU32 {
        self.integrable.connection.symmetry_order()
    }

    #[must_use]
    pub fn power_directions(&self) -> &[f64] {
        &self.power_directions
    }

    /// Calculate exact charges with O(E log E) compact lookup and quantization evidence.
    ///
    /// # Errors
    ///
    /// Rejects unavailable surface geometry, indeterminate binary64 rounding, or
    /// a result that violates the exact relative charge law.
    pub fn singularities(&self) -> Result<DirectionFieldSingularities, SurfaceError> {
        let surface = &self.integrable.connection.surface;
        let topology = surface.realization().topology();
        let symmetry_order = self.symmetry_order();
        let order = f64::from(symmetry_order.get());
        let regular = topology.refine_regular()?;
        let boundary_vertices = regular.boundary_mask(0)?;
        let levi_civita = surface.levi_civita_connection()?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(topology.vertex_count())
            .map_err(|_| SurfaceError::Overflow)?;
        let (charge_total, charge_residual, maximum_valence) =
            interior_direction_charges(self, &levi_civita, &boundary_vertices, &mut entries)?;
        let (boundary_turns, boundary_total, boundary_residual, maximum_boundary_edges) =
            relative_boundary_turns(self)?;
        let maximum_residual = charge_residual.max(boundary_residual);
        let operation_count = u32::try_from(
            maximum_valence
                .saturating_add(3)
                .max(maximum_boundary_edges.saturating_add(3)),
        )
        .unwrap_or(u32::MAX);
        let residual_limit = 4096.0 * f64::EPSILON * f64::from(operation_count) * order;
        let euler = i128::try_from(topology.vertex_count())
            .ok()
            .and_then(|value| value.checked_sub(i128::try_from(surface.edge_count()).ok()?))
            .and_then(|value| value.checked_add(i128::try_from(surface.face_count()).ok()?))
            .ok_or(SurfaceError::Overflow)?;
        let expected_total = BigInt::from(symmetry_order.get()) * BigInt::from(euler);
        if residual_limit >= 0.5
            || maximum_residual > residual_limit
            || charge_total - boundary_total != expected_total
        {
            return Err(SurfaceError::Unrepresentable);
        }
        let space = topology.chain_complex().dual().space(0)?;
        let charges = space
            .element(entries)
            .map_err(|_| SurfaceError::Unrepresentable)?;
        Ok(DirectionFieldSingularities {
            symmetry_order,
            charges,
            boundary_turns: boundary_turns.into_boxed_slice(),
            maximum_quantization_residual: maximum_residual,
            residual_limit,
        })
    }

    /// Allocate one explicit ambient branch from the retained power coordinates.
    ///
    /// # Errors
    ///
    /// Returns an unavailable-frame or representability failure.
    pub fn ambient_vector_branch_copy(&self, branch: usize) -> Result<FaceVectors, SurfaceError> {
        let symmetry_order =
            usize::try_from(self.symmetry_order().get()).map_err(|_| SurfaceError::Overflow)?;
        if branch >= symmetry_order {
            return Err(SurfaceError::IndexOutside);
        }
        let order = f64::from(self.symmetry_order().get());
        let branch = branch.to_f64().ok_or(SurfaceError::Unrepresentable)?;
        let first = self.integrable.connection.surface.first_frame_axes()?;
        let second = self.integrable.connection.surface.second_frame_axes()?;
        let mut values = Vec::with_capacity(3 * self.integrable.connection.surface.face_count());
        for face in 0..self.integrable.connection.surface.face_count() {
            let power_direction = complex_at(&self.power_directions, face)?;
            let angle = power_direction[1].atan2(power_direction[0]) / order
                + std::f64::consts::TAU * branch / order;
            let direction = [angle.cos(), angle.sin()];
            let first = row3(first, face)?;
            let second = row3(second, face)?;
            values.extend([
                direction[0] * first[0] + direction[1] * second[0],
                direction[0] * first[1] + direction[1] * second[1],
                direction[0] * first[2] + direction[1] * second[2],
            ]);
        }
        self.integrable.connection.surface.face_vectors(values)
    }
}

/// Period-normalized harmonic degree-one cochains over one positive metric.
#[derive(Clone, Debug)]
pub struct HarmonicOneFormBasis {
    forms: Box<[Binary64Cochain]>,
    maximum_closedness_residual: f64,
    maximum_coclosedness_residual: f64,
    maximum_identity_period_residual: f64,
    residual_limit: f64,
}

impl HarmonicOneFormBasis {
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.forms.len()
    }

    #[must_use]
    pub const fn forms(&self) -> &[Binary64Cochain] {
        &self.forms
    }

    #[must_use]
    pub const fn maximum_closedness_residual(&self) -> f64 {
        self.maximum_closedness_residual
    }

    #[must_use]
    pub const fn maximum_coclosedness_residual(&self) -> f64 {
        self.maximum_coclosedness_residual
    }

    #[must_use]
    pub const fn maximum_identity_period_residual(&self) -> f64 {
        self.maximum_identity_period_residual
    }

    #[must_use]
    pub const fn residual_limit(&self) -> f64 {
        self.residual_limit
    }
}

impl crate::Metric {
    /// Construct harmonic degree-one cochains dual to exact free homology cycles.
    ///
    /// # Errors
    /// Rejects a foreign or non-degree-one homology group, an unsuitable surface
    /// topology, exhausted resources, cancellation, or failed numerical
    /// certification.
    pub fn harmonic_one_form_basis(
        &self,
        group: HomologyGroup<'_>,
        policy: Policy,
        cancellation: &CancellationToken,
    ) -> Result<HarmonicOneFormBasis, SurfaceComputationError> {
        let topology = self.realization().topology();
        if group.degree() != 1 || !group.chain_complex().same_owner(&topology.chain_complex()) {
            return Err(SolveError::ProblemMismatch.into());
        }
        if !group.torsion_orders().is_empty() {
            return Err(SolveError::ProblemMismatch.into());
        }
        let edge_count = topology.basis(1).map_err(SurfaceError::from)?.row_count();
        let face_count = topology.basis(2).map_err(SurfaceError::from)?.row_count();
        require_harmonic_basis_resources(
            topology.vertex_count(),
            edge_count,
            face_count,
            group.free_rank(),
            policy.storage(),
            policy.work(),
        )
        .map_err(SurfaceComputationError::Solve)?;
        check_cancelled(cancellation).map_err(SurfaceComputationError::Solve)?;
        let seeds = topology
            .integral_dual_cycle_basis()
            .map_err(SurfaceError::from)?;
        if seeds.rank() != group.free_rank() {
            return Err(SolveError::ProblemMismatch.into());
        }
        harmonic_one_form_basis(
            self,
            group,
            &seeds,
            policy.executor(),
            policy.storage(),
            policy.work(),
            cancellation,
        )
        .map_err(SurfaceComputationError::Solve)
    }
}

fn harmonic_one_form_basis(
    metric: &Metric,
    group: HomologyGroup<'_>,
    seeds: &IntegralDualCycleBasis,
    executor: Executor,
    storage: StorageLimit,
    work: WorkLimit,
    cancellation: &CancellationToken,
) -> Result<HarmonicOneFormBasis, SolveError> {
    check_cancelled(cancellation)?;
    let rank = seeds.rank();
    let topology = metric.realization().topology();
    if rank == 0 {
        return Ok(empty_harmonic_one_form_basis());
    }
    let edge_space = Binary64Space::<Cochain>::full(Arc::clone(topology), 1)
        .map_err(|_| SolveError::Numerical)?;
    let harmonic_seeds = project_harmonic_seeds(
        metric,
        seeds,
        &edge_space,
        executor,
        storage,
        work,
        cancellation,
    )?;
    let forms =
        normalize_harmonic_periods(group, &edge_space, &harmonic_seeds, executor, cancellation)?;
    certify_harmonic_one_form_basis(metric, group, forms)
}

fn require_harmonic_basis_resources(
    vertex_count: usize,
    edge_count: usize,
    face_count: usize,
    rank: usize,
    storage: StorageLimit,
    work: WorkLimit,
) -> Result<(), SolveError> {
    let retained_cells = checked_cells(rank, edge_count)?;
    let retained = logical_f64(retained_cells)?;
    let reduced_vertices = vertex_count.saturating_sub(1);
    let stiffness_entries = reduced_vertices
        .checked_add(checked_cells(2, edge_count)?)
        .ok_or(SolveError::ResourceLimit)?;
    let poisson_factor = sparse_phase_policy_bytes(reduced_vertices, stiffness_entries, 0)?;
    // One edge seed plus vertex rhs, load, potential, and solution values.
    let streamed_seed = logical_f64(checked_sum([edge_count, checked_cells(4, vertex_count)?])?)?;
    let projection_peak = checked_sum([retained, streamed_seed, poisson_factor])?;
    let period_factor = matrix_bytes(rank)?;
    let period_phase = checked_sum([retained, retained, period_factor, period_factor])?;
    let certification = logical_f64(checked_sum([edge_count, vertex_count, face_count])?)?;
    let peak = projection_peak
        .max(period_phase)
        .max(checked_sum([retained, certification])?);
    require_storage(storage, retained, peak)?;
    let repeated_products = checked_work_product(
        checked_sum([
            checked_cells(reduced_vertices, reduced_vertices)?,
            retained_cells,
        ])?,
        rank,
    )?;
    let required_work = checked_sum([
        cubic_work(reduced_vertices)?,
        cubic_work(rank)?,
        repeated_products,
    ])?;
    require_work(work, required_work)
}

fn empty_harmonic_one_form_basis() -> HarmonicOneFormBasis {
    HarmonicOneFormBasis {
        forms: Box::new([]),
        maximum_closedness_residual: 0.0,
        maximum_coclosedness_residual: 0.0,
        maximum_identity_period_residual: 0.0,
        residual_limit: 0.0,
    }
}

fn project_harmonic_seeds(
    metric: &Metric,
    seeds: &IntegralDualCycleBasis,
    edge_space: &Binary64Space<Cochain>,
    executor: Executor,
    storage: StorageLimit,
    work: WorkLimit,
    cancellation: &CancellationToken,
) -> Result<Vec<Binary64Cochain>, SolveError> {
    let rank = seeds.rank();
    let internal_storage = StorageLimit::new(
        storage.peak_live_logical_bytes(),
        storage.peak_live_logical_bytes(),
    )
    .ok_or(SolveError::ResourceLimit)?;
    let codifferential = metric
        .codifferential(1)
        .map_err(|_| SolveError::Numerical)?;
    let vertex_riesz = metric.riesz(0).map_err(|_| SolveError::Numerical)?;
    let make_seed_problem = |index| -> Result<_, SolveError> {
        let exact = seeds.cocycle(index).ok_or(SolveError::Numerical)?;
        let seed = Binary64Element::realize_integral(edge_space.clone(), exact)
            .map_err(|_| SolveError::Numerical)?;
        let rhs = codifferential
            .apply(&seed)
            .map_err(|_| SolveError::Numerical)?;
        let load = vertex_riesz
            .apply(&rhs)
            .map_err(|_| SolveError::Numerical)?;
        let problem = metric
            .mean_zero_poisson_load(load)
            .map_err(|_| SolveError::Numerical)?;
        Ok((seed, problem))
    };
    let first = make_seed_problem(0)?;
    let policy = Policy::new(executor, internal_storage, work);
    let prepared = first.1.prepare_cancellable(policy, cancellation)?;
    let mut harmonic_seeds = Vec::new();
    harmonic_seeds
        .try_reserve_exact(rank)
        .map_err(|_| SolveError::Allocation)?;
    for pair in std::iter::once(Ok(first)).chain((1..rank).map(make_seed_problem)) {
        let (seed, problem) = pair?;
        check_cancelled(cancellation)?;
        let mut workspace = prepared.workspace_for(&problem)?;
        let solution = prepared.solve_cancellable(&problem, &mut workspace, cancellation)?;
        let exact = solution
            .potential()
            .exterior_derivative()
            .map_err(|_| SolveError::Numerical)?;
        let coefficients = seed
            .coefficients()
            .iter()
            .zip(exact.coefficients())
            .map(|(&seed, &exact)| seed - exact)
            .collect::<Vec<_>>();
        harmonic_seeds.push(
            Binary64Element::admit(edge_space.clone(), coefficients)
                .map_err(|_| SolveError::Numerical)?,
        );
    }
    Ok(harmonic_seeds)
}

fn normalize_harmonic_periods(
    group: HomologyGroup<'_>,
    edge_space: &Binary64Space<Cochain>,
    harmonic_seeds: &[Binary64Cochain],
    executor: Executor,
    cancellation: &CancellationToken,
) -> Result<Vec<Binary64Cochain>, SolveError> {
    let rank = harmonic_seeds.len();
    let edge_count = edge_space.size();
    let mut periods = Mat::<f64>::zeros(rank, rank);
    for (column, harmonic) in harmonic_seeds.iter().enumerate() {
        for (row, &period) in group
            .periods_binary64(harmonic)
            .map_err(|_| SolveError::ProblemMismatch)?
            .iter()
            .enumerate()
        {
            periods[(row, column)] = period;
        }
    }
    let factor = factor_dense_square(periods, executor, cancellation)?;
    require_stable_dense_lu(&factor)?;
    let requirement = factor_solve_requirement(&factor, executor, rank);
    let mut buffer = MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?;
    let scale = factor_scale(&factor);
    let mut coordinates = Mat::<f64>::zeros(rank, rank);
    for index in 0..rank {
        coordinates[(index, index)] = 1.0 / scale;
    }
    solve_factor(
        &factor,
        coordinates.as_mut(),
        executor,
        MemStack::new(&mut buffer),
    );
    check_cancelled(cancellation)?;

    let mut forms = Vec::new();
    forms
        .try_reserve_exact(rank)
        .map_err(|_| SolveError::Allocation)?;
    for column in 0..rank {
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(edge_count)
            .map_err(|_| SolveError::Allocation)?;
        for edge in 0..edge_count {
            let (value, _) = adaptive_product_value(
                harmonic_seeds
                    .iter()
                    .enumerate()
                    .map(|(row, form)| (form.coefficients()[edge], coordinates[(row, column)])),
            )
            .ok_or(SolveError::Numerical)?;
            coefficients.push(value);
        }
        forms.push(
            Binary64Element::admit(edge_space.clone(), coefficients)
                .map_err(|_| SolveError::Numerical)?,
        );
    }
    Ok(forms)
}

fn certify_harmonic_one_form_basis(
    metric: &Metric,
    group: HomologyGroup<'_>,
    forms: Vec<Binary64Cochain>,
) -> Result<HarmonicOneFormBasis, SolveError> {
    let codifferential = metric
        .codifferential(1)
        .map_err(|_| SolveError::Numerical)?;
    let mut closedness = 0.0_f64;
    let mut coclosedness = 0.0_f64;
    let mut identity_period = 0.0_f64;
    for (column, form) in forms.iter().enumerate() {
        let derivative = form
            .exterior_derivative()
            .map_err(|_| SolveError::Numerical)?;
        closedness = closedness.max(maximum_absolute(derivative.coefficients())?);
        let codifferential = codifferential
            .apply(form)
            .map_err(|_| SolveError::Numerical)?;
        coclosedness = coclosedness.max(maximum_absolute(codifferential.coefficients())?);
        for (row, &period) in group
            .periods_binary64(form)
            .map_err(|_| SolveError::ProblemMismatch)?
            .iter()
            .enumerate()
        {
            identity_period = identity_period.max((period - f64::from(row == column)).abs());
        }
    }
    let operation_count = forms
        .len()
        .saturating_add(metric.realization().topology().vertex_count())
        .saturating_add(
            metric
                .realization()
                .topology()
                .basis(1)
                .map_err(|_| SolveError::Numerical)?
                .row_count(),
        );
    let residual_limit = 4096.0
        * f64::EPSILON
        * f64::from(u32::try_from(operation_count.max(1)).unwrap_or(u32::MAX));
    if !residual_limit.is_finite()
        || closedness > residual_limit
        || coclosedness > residual_limit
        || identity_period > residual_limit
    {
        return Err(SolveError::Numerical);
    }
    Ok(HarmonicOneFormBasis {
        forms: forms.into_boxed_slice(),
        maximum_closedness_residual: closedness,
        maximum_coclosedness_residual: coclosedness,
        maximum_identity_period_residual: identity_period,
        residual_limit,
    })
}

fn maximum_absolute(values: &[f64]) -> Result<f64, SolveError> {
    values.iter().try_fold(0.0_f64, |maximum, &value| {
        value
            .is_finite()
            .then_some(maximum.max(value.abs()))
            .ok_or(SolveError::Numerical)
    })
}

fn dual_edge_values(
    surface: &TriangleSurface,
    dual: DualEdges<'_>,
    chain: &Binary64Chain,
) -> Result<Vec<f64>, SurfaceComputationError> {
    let topology = surface.realization().topology();
    let expected =
        Binary64Space::<Chain>::full(Arc::clone(topology), 1).map_err(|_| SolveError::Numerical)?;
    if !expected.same_space(chain.space()) {
        return Err(SolveError::ProblemMismatch.into());
    }
    chain
        .coefficients()
        .iter()
        .enumerate()
        .map(|(edge, &value)| {
            let (_, _, source_sign) = dual.edge(edge)?;
            Ok(f64::from(source_sign) * value)
        })
        .collect()
}

fn dual_period(
    dual: DualEdges<'_>,
    cycles: &IntegralDualCycleBasis,
    cycle_index: usize,
    values: &[f64],
) -> Result<f64, SurfaceComputationError> {
    let cycle = cycles
        .cocycle(cycle_index)
        .ok_or(SurfaceError::IndexOutside)?;
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(cycle.indices().len())
        .map_err(|_| SolveError::Allocation)?;
    for (&edge, coefficient) in cycle.indices().iter().zip(cycle.coefficients()) {
        let coefficient = coefficient
            .to_f64()
            .filter(|value| value.is_finite())
            .ok_or(SurfaceError::Unrepresentable)?;
        let (_, _, source_sign) = dual.edge(edge)?;
        terms.push((-f64::from(source_sign) * coefficient, values[edge]));
    }
    adaptive_product_value(terms.into_iter())
        .map(|(value, _)| value)
        .ok_or_else(|| SolveError::Numerical.into())
}

fn same_integral_coordinates(left: &IntegralCochain, right: &IntegralCochain) -> bool {
    left.space().same_based_space(right.space())
        && left.indices() == right.indices()
        && left.coefficients() == right.coefficients()
}

fn require_direction_field_resources(
    vertex_count: usize,
    edge_count: usize,
    face_count: usize,
    rank: usize,
    storage: StorageLimit,
    work: WorkLimit,
) -> Result<(), SolveError> {
    let deviations = logical_f64(edge_count)?;
    let transports = logical_f64(checked_cells(2, edge_count)?)?;
    let power_directions = logical_f64(checked_cells(2, face_count)?)?;
    let retained = checked_sum([transports, power_directions])?;
    let reduced_vertices = vertex_count.saturating_sub(1);
    let stiffness_entries = reduced_vertices
        .checked_add(checked_cells(2, edge_count)?)
        .ok_or(SolveError::ResourceLimit)?;
    let poisson_factor = sparse_phase_policy_bytes(reduced_vertices, stiffness_entries, 0)?;
    // Curvature/load/potential/solution values and gradient/coexact/deviation values.
    let coexact_values = logical_f64(checked_sum([
        checked_cells(4, vertex_count)?,
        checked_cells(3, edge_count)?,
    ])?)?;
    let harmonic_values = logical_f64(checked_cells(rank, edge_count)?)?;
    let period_system = checked_sum([matrix_bytes(rank)?, matrix_bytes(rank)?])?;
    let coexact_phase = checked_sum([deviations, coexact_values, poisson_factor])?;
    let harmonic_phase = checked_sum([transports, deviations, harmonic_values, period_system])?;
    let reconstruction = checked_sum([transports, deviations, retained])?;
    let peak = coexact_phase.max(harmonic_phase).max(reconstruction);
    require_storage(storage, retained, peak)?;
    let harmonic_products = checked_work_product(edge_count, rank)?;
    let edge_pass = u64::try_from(edge_count).map_err(|_| SolveError::ResourceLimit)?;
    // The repeated edge passes are coexact, reconstruction, period, and certification.
    let required_work = checked_sum([
        cubic_work(reduced_vertices)?,
        cubic_work(rank)?,
        harmonic_products,
        edge_pass,
        edge_pass,
        edge_pass,
        edge_pass,
    ])?;
    require_work(work, required_work)
}

#[derive(Clone, Copy)]
struct DirectionSystem<'a> {
    surface: &'a TriangleSurface,
    dual: DualEdges<'a>,
    symmetry_order: NonZeroU32,
    metric: &'a Metric,
}

fn solve_coexact_direction_adjustment(
    system: DirectionSystem<'_>,
    charges: &IntegralCochain,
    policy: Policy,
    cancellation: &CancellationToken,
) -> Result<Vec<f64>, SurfaceComputationError> {
    let DirectionSystem {
        surface,
        dual,
        symmetry_order,
        metric,
    } = system;
    let topology = surface.realization().topology();
    let curvature = surface.gaussian_curvature_measure()?;
    let order = f64::from(symmetry_order.get());
    let mut load_values = curvature
        .coefficients()
        .iter()
        .map(|value| -order * *value)
        .collect::<Vec<_>>();
    for (&vertex, coefficient) in charges.indices().iter().zip(charges.coefficients()) {
        let charge = coefficient
            .to_f64()
            .filter(|value| value.is_finite())
            .ok_or(SurfaceError::Unrepresentable)?;
        load_values[vertex] += std::f64::consts::TAU * charge;
    }
    let load_space =
        Binary64Space::<Chain>::full(Arc::clone(topology), 0).map_err(|_| SolveError::Numerical)?;
    let load =
        Binary64Element::admit(load_space, load_values).map_err(|_| SolveError::Numerical)?;
    let problem = metric
        .mean_zero_poisson_load(load)
        .map_err(|_| SolveError::Numerical)?;
    let prepared = problem.prepare_cancellable(policy, cancellation)?;
    let mut workspace = prepared.workspace_for(&problem)?;
    let solution = prepared.solve_cancellable(&problem, &mut workspace, cancellation)?;
    let gradient = solution
        .potential()
        .exterior_derivative()
        .map_err(|_| SolveError::Numerical)?;
    let coexact = metric
        .riesz(1)
        .map_err(|_| SolveError::Numerical)?
        .apply(&gradient)
        .map_err(|_| SolveError::Numerical)?;
    dual_edge_values(surface, dual, &coexact)
}

fn add_harmonic_direction_adjustment(
    system: DirectionSystem<'_>,
    levi_civita: &SurfaceConnection,
    harmonic_basis: &HarmonicOneFormBasis,
    dual_cycles: &IntegralDualCycleBasis,
    generator_turns: &[i64],
    deviations: &mut [f64],
    execution: (Executor, &CancellationToken),
) -> Result<(), SurfaceComputationError> {
    let DirectionSystem {
        surface,
        dual,
        symmetry_order,
        metric,
    } = system;
    let (executor, cancellation) = execution;
    let rank = dual_cycles.rank();
    if rank == 0 {
        return Ok(());
    }
    let levi_civita_angles = levi_civita
        .transports()
        .chunks_exact(2)
        .map(|value| value[1].atan2(value[0]))
        .collect::<Vec<_>>();
    let order = f64::from(symmetry_order.get());
    let mut harmonic_dual = Vec::new();
    harmonic_dual
        .try_reserve_exact(rank)
        .map_err(|_| SolveError::Allocation)?;
    let riesz = metric.riesz(1).map_err(|_| SolveError::Numerical)?;
    for form in harmonic_basis.forms() {
        let chain = riesz.apply(form).map_err(|_| SolveError::ProblemMismatch)?;
        harmonic_dual.push(dual_edge_values(surface, dual, &chain)?);
    }
    let mut periods = Mat::<f64>::zeros(rank, rank);
    let mut target = Mat::<f64>::zeros(rank, 1);
    for row in 0..rank {
        let base_angle = dual_period(dual, dual_cycles, row, &levi_civita_angles)?;
        let coexact_period = dual_period(dual, dual_cycles, row, deviations)?;
        let turns = generator_turns[row]
            .to_f64()
            .ok_or(SurfaceError::Unrepresentable)?;
        target[(row, 0)] = std::f64::consts::TAU * turns - order * base_angle - coexact_period;
        for column in 0..rank {
            periods[(row, column)] = dual_period(dual, dual_cycles, row, &harmonic_dual[column])?;
        }
    }
    let factor = factor_dense_square(periods, executor, cancellation)?;
    require_stable_dense_lu(&factor)?;
    let requirement = factor_solve_requirement(&factor, executor, 1);
    let mut buffer = MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?;
    let scale = factor_scale(&factor);
    for row in 0..rank {
        target[(row, 0)] /= scale;
    }
    solve_factor(
        &factor,
        target.as_mut(),
        executor,
        MemStack::new(&mut buffer),
    );
    for (edge, deviation) in deviations.iter_mut().enumerate() {
        let (correction, _) = adaptive_product_value(
            harmonic_dual
                .iter()
                .enumerate()
                .map(|(column, form)| (form[edge], target[(column, 0)])),
        )
        .ok_or(SolveError::Numerical)?;
        *deviation += correction;
    }
    let operation_count =
        u32::try_from(deviations.len().saturating_add(rank).saturating_add(1)).unwrap_or(u32::MAX);
    let period_limit = 8192.0 * f64::EPSILON * f64::from(operation_count) * order;
    if period_limit >= std::f64::consts::PI {
        return Err(SolveError::Numerical.into());
    }
    for (row, &turns) in generator_turns.iter().enumerate() {
        let turns = turns.to_f64().ok_or(SurfaceError::Unrepresentable)?;
        let observed = order * dual_period(dual, dual_cycles, row, &levi_civita_angles)?
            + dual_period(dual, dual_cycles, row, deviations)?;
        if (observed - std::f64::consts::TAU * turns).abs() > period_limit {
            return Err(SolveError::Numerical.into());
        }
    }
    Ok(())
}

impl TriangleSurface {
    /// Construct a minimum-energy symmetric direction field with exact charges and turns.
    ///
    /// The supplied degree-zero integral cochain fixes power charges. Generator turns are
    /// lifted power holonomies in the order of `dual_cycles`; `anchor_angle` fixes the
    /// remaining global rotation modulo the supplied symmetry order.
    ///
    /// # Errors
    /// Rejects foreign inputs, a charge sum different from order times Euler characteristic, mismatched
    /// generator dimensions, exhausted resources, cancellation, or failed certification.
    #[expect(
        clippy::too_many_arguments,
        reason = "mathematical inputs and execution policies remain explicit"
    )]
    pub fn minimum_energy_direction_field(
        self: &Arc<Self>,
        symmetry_order: NonZeroU32,
        metric: &Metric,
        harmonic_basis: &HarmonicOneFormBasis,
        dual_cycles: &IntegralDualCycleBasis,
        charges: &IntegralCochain,
        generator_turns: &[i64],
        anchor_angle: f64,
        policy: Policy,
        cancellation: &CancellationToken,
    ) -> Result<FaceDirectionField, SurfaceComputationError> {
        let topology = self.realization().topology();
        if !Arc::ptr_eq(metric.realization(), self.realization())
            || !dual_cycles
                .chain_complex()
                .same_owner(&topology.chain_complex())
        {
            return Err(SolveError::ProblemMismatch.into());
        }
        let charge_space = topology
            .chain_complex()
            .dual()
            .space(0)
            .map_err(SurfaceError::from)?;
        if !charges.space().same_based_space(&charge_space)
            || harmonic_basis.rank() != dual_cycles.rank()
            || generator_turns.len() != dual_cycles.rank()
            || !anchor_angle.is_finite()
        {
            return Err(SolveError::ProblemMismatch.into());
        }
        let edge_count = topology.basis(1).map_err(SurfaceError::from)?.row_count();
        require_direction_field_resources(
            topology.vertex_count(),
            edge_count,
            self.face_count(),
            harmonic_basis.rank(),
            policy.storage(),
            policy.work(),
        )?;
        check_cancelled(cancellation)?;
        let euler = i128::try_from(topology.vertex_count())
            .ok()
            .and_then(|value| value.checked_sub(i128::try_from(edge_count).ok()?))
            .and_then(|value| value.checked_add(i128::try_from(self.face_count()).ok()?))
            .ok_or(SurfaceError::Overflow)?;
        let expected_charge =
            num_bigint::BigInt::from(symmetry_order.get()) * num_bigint::BigInt::from(euler);
        if charges.coefficients().iter().sum::<num_bigint::BigInt>() != expected_charge {
            return Err(SolveError::ProblemMismatch.into());
        }

        let dual = dual_edges(topology)?;
        let system = DirectionSystem {
            surface: self,
            dual,
            symmetry_order,
            metric,
        };
        let mut deviations =
            solve_coexact_direction_adjustment(system, charges, policy, cancellation)?;
        let levi_civita = self.levi_civita_connection()?;
        add_harmonic_direction_adjustment(
            system,
            &levi_civita,
            harmonic_basis,
            dual_cycles,
            generator_turns,
            &mut deviations,
            (policy.executor(), cancellation),
        )?;
        check_cancelled(cancellation)?;
        let field = levi_civita
            .with_powered_deviations(symmetry_order, &deviations)?
            .require_integrable()?
            .direction_field(anchor_angle)?;
        let observed = field.singularities()?;
        if observed.symmetry_order() != symmetry_order
            || !same_integral_coordinates(observed.charges(), charges)
        {
            return Err(SolveError::Numerical.into());
        }
        Ok(field)
    }

    /// Construct a boundary-aligned symmetric field minimizing connection deviation
    /// within the lift sector selected by its relaxed Dirichlet extension.
    ///
    /// # Errors
    ///
    /// Rejects a closed or incompatible surface, ambiguous targets or phase lifts,
    /// exhausted resources, cancellation, factorization, or failed certification.
    pub fn boundary_aligned_direction_field(
        self: &Arc<Self>,
        symmetry_order: NonZeroU32,
        metric: &Metric,
        boundary_angle_offset: f64,
        policy: Policy,
        cancellation: &CancellationToken,
    ) -> Result<FaceDirectionField, SurfaceComputationError> {
        if !Arc::ptr_eq(metric.realization(), self.realization()) {
            return Err(SolveError::ProblemMismatch.into());
        }
        let topology = self.realization().topology();
        topology.refine_connected().map_err(SurfaceError::from)?;
        topology
            .refine_regular()
            .map_err(SurfaceError::from)?
            .with_boundary()
            .map_err(SurfaceError::from)?;
        require_boundary_direction_field_resources(
            self.face_count(),
            self.edge_count(),
            policy.storage(),
            policy.work(),
        )?;
        check_cancelled(cancellation)?;
        boundary_aligned_direction_field(
            self,
            symmetry_order,
            metric,
            boundary_angle_offset,
            policy.executor(),
            cancellation,
        )
    }
}
fn require_boundary_direction_field_resources(
    face_count: usize,
    edge_count: usize,
    storage: StorageLimit,
    work: WorkLimit,
) -> Result<(), SolveError> {
    let relaxed_rank = face_count.checked_mul(2).ok_or(SolveError::ResourceLimit)?;
    // Target directions + final directions, target faces, and the temporary mask.
    let target_peak = checked_sum([
        logical_f64(checked_cells(4, face_count)?)?,
        logical_bytes(face_count, size_of::<usize>() + size_of::<bool>())?,
    ])?;
    // Directions + diagonal; base/powered transports + weights; reduced positions.
    let graph = checked_sum([
        logical_f64(checked_sum([
            checked_cells(3, face_count)?,
            checked_cells(5, edge_count)?,
        ])?)?,
        logical_bytes(face_count, size_of::<Option<usize>>())?,
    ])?;
    let relaxed_entries = relaxed_rank
        .checked_add(checked_cells(8, edge_count)?)
        .ok_or(SolveError::ResourceLimit)?;
    let scalar_entries = face_count
        .checked_add(checked_cells(2, edge_count)?)
        .ok_or(SolveError::ResourceLimit)?;
    let relaxed = sparse_phase_policy_bytes(relaxed_rank, relaxed_entries, relaxed_rank)?;
    let scalar_cells = checked_sum([edge_count, face_count])?;
    let scalar = sparse_phase_policy_bytes(face_count, scalar_entries, scalar_cells)?;
    let retained = logical_f64(checked_sum([
        checked_cells(2, edge_count)?,
        checked_cells(2, face_count)?,
    ])?)?;
    let reconstruction = logical_f64(checked_cells(3, edge_count)?)?;
    require_storage(
        storage,
        retained,
        target_peak.max(checked_sum([
            graph,
            relaxed.max(scalar).max(reconstruction),
        ])?),
    )?;
    let linear_passes = checked_sum([relaxed_entries, scalar_entries, edge_count, face_count])?;
    let required_work = checked_sum([
        cubic_work(relaxed_rank)?,
        cubic_work(face_count)?,
        checked_work_product(linear_passes, 1)?,
    ])?;
    require_work(work, required_work)
}

fn solve_weighted_positive_system(
    rank: usize,
    triplets: &[Triplet<usize, usize, f64>],
    scale: f64,
    values: &mut [f64],
    execution: (Executor, &CancellationToken),
) -> Result<(), SolveError> {
    let (executor, cancellation) = execution;
    if values.len() != rank {
        return Err(SolveError::Numerical);
    }
    if rank == 0 {
        return Ok(());
    }
    let factor = factor_sparse_triplets(rank, triplets, scale, executor, cancellation)?;
    let requirement = factor_solve_requirement(&factor, executor, 1);
    let mut buffer = MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?;
    for value in values.iter_mut() {
        *value /= scale;
    }
    solve_factor(
        &factor,
        MatMut::from_column_major_slice_mut(values, rank, 1),
        executor,
        MemStack::new(&mut buffer),
    );
    check_cancelled(cancellation)?;
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::Numerical)
}

fn relative_max_residual(gradient: &[f64], scale: f64) -> Result<f64, SolveError> {
    let error = gradient.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let residual = if scale == 0.0 { error } else { error / scale };
    residual
        .is_finite()
        .then_some(residual)
        .ok_or(SolveError::Numerical)
}

fn certified_connection_angle(
    transport: [f64; 2],
    source: [f64; 2],
    target: [f64; 2],
    antipodal_limit: f64,
) -> Result<f64, SurfaceError> {
    let expected = complex_multiply(transport, source);
    let mismatch = normalize_complex(complex_multiply(target, complex_conjugate(expected)))?;
    let angle = mismatch[1].atan2(mismatch[0]);
    if std::f64::consts::PI - angle.abs() <= antipodal_limit {
        return Err(SurfaceError::Unrepresentable);
    }
    Ok(angle)
}

#[expect(
    clippy::too_many_arguments,
    reason = "one edge contribution keeps its mathematical rows and output buffers explicit"
)]
fn add_relaxed_edge(
    positions: &[Option<usize>],
    directions: &[f64],
    source: usize,
    target: usize,
    transport: [f64; 2],
    weight: f64,
    scale: f64,
    triplets: &mut Vec<Triplet<usize, usize, f64>>,
    values: &mut [f64],
) -> Result<(), SurfaceError> {
    let rotation = [[transport[0], -transport[1]], [transport[1], transport[0]]];
    match (positions[source], positions[target]) {
        (Some(source), Some(target)) => {
            for (target_axis, row) in rotation.iter().enumerate() {
                for (source_axis, &coefficient) in row.iter().enumerate() {
                    let source = 2 * source + source_axis;
                    let target = 2 * target + target_axis;
                    let value = -weight * coefficient / scale;
                    triplets.extend(
                        [(source, target), (target, source)]
                            .map(|(row, column)| Triplet::new(row, column, value)),
                    );
                }
            }
        }
        (Some(source), None) => {
            let target = complex_at(directions, target)?;
            let value = complex_multiply(complex_conjugate(transport), target);
            values[2 * source] += weight * value[0];
            values[2 * source + 1] += weight * value[1];
        }
        (None, Some(target)) => {
            let source = complex_at(directions, source)?;
            let value = complex_multiply(transport, source);
            values[2 * target] += weight * value[0];
            values[2 * target + 1] += weight * value[1];
        }
        (None, None) => {}
    }
    Ok(())
}

fn relaxed_boundary_extension(
    dual: DualEdges<'_>,
    interior_edges: &[usize],
    levi_civita_power: &[f64],
    weights: &[f64],
    free_rows: (&[Option<usize>], &[f64]),
    directions: &mut [f64],
    execution: (Executor, &CancellationToken),
) -> Result<(), SurfaceComputationError> {
    let (positions, diagonal) = free_rows;
    let scale = diagonal.iter().copied().fold(0.0_f64, f64::max);
    if !diagonal.is_empty() && (!scale.is_finite() || scale <= 0.0) {
        return Err(SolveError::Factorization.into());
    }
    let rank = 2 * diagonal.len();
    let mut relaxed = {
        let mut triplets =
            Vec::with_capacity(rank.saturating_add(interior_edges.len().saturating_mul(8)));
        for (position, &value) in diagonal.iter().enumerate() {
            for axis in 0..2 {
                let row = 2 * position + axis;
                triplets.push(Triplet::new(row, row, value / scale));
            }
        }
        let mut values = vec![0.0_f64; rank];
        for (row, (&edge, &weight)) in interior_edges.iter().zip(weights).enumerate() {
            let (source, target, _) = dual.edge(edge)?;
            add_relaxed_edge(
                positions,
                directions,
                source,
                target,
                complex_at(levi_civita_power, row)?,
                weight,
                scale,
                &mut triplets,
                &mut values,
            )?;
        }
        solve_weighted_positive_system(rank, &triplets, scale, &mut values, execution)?;
        values
    };
    for (face, position) in positions.iter().enumerate() {
        if let Some(position) = position {
            directions[2 * face..2 * face + 2]
                .copy_from_slice(&relaxed[2 * position..2 * position + 2]);
        }
    }

    relaxed.fill(0.0);
    let mut gradient_scale = 0.0_f64;
    for (row, (&edge, &weight)) in interior_edges.iter().zip(weights).enumerate() {
        let (source, target, _) = dual.edge(edge)?;
        let transport = complex_at(levi_civita_power, row)?;
        let expected = complex_multiply(transport, complex_at(directions, source)?);
        let target_direction = complex_at(directions, target)?;
        let difference = [
            target_direction[0] - expected[0],
            target_direction[1] - expected[1],
        ];
        gradient_scale = gradient_scale.max(weight * difference[0].abs());
        gradient_scale = gradient_scale.max(weight * difference[1].abs());
        if let Some(position) = positions[source] {
            let difference = complex_multiply(complex_conjugate(transport), difference);
            relaxed[2 * position] -= weight * difference[0];
            relaxed[2 * position + 1] -= weight * difference[1];
        }
        if let Some(position) = positions[target] {
            relaxed[2 * position] += weight * difference[0];
            relaxed[2 * position + 1] += weight * difference[1];
        }
    }
    if relative_max_residual(&relaxed, gradient_scale)? > 1.0e-10 {
        return Err(SolveError::Numerical.into());
    }
    let magnitude_scale = directions
        .chunks_exact(2)
        .map(|value| value[0].hypot(value[1]))
        .fold(0.0_f64, f64::max);
    let nonzero_limit = 4096.0
        * f64::EPSILON
        * f64::from(u32::try_from(positions.len().max(1)).unwrap_or(u32::MAX))
        * magnitude_scale.max(1.0);
    for direction in directions.chunks_exact_mut(2) {
        let magnitude = direction[0].hypot(direction[1]);
        if magnitude <= nonzero_limit {
            return Err(SolveError::Numerical.into());
        }
        direction.copy_from_slice(&normalize_complex([direction[0], direction[1]])?);
    }
    Ok(())
}

fn fixed_sector_deviations(
    dual: DualEdges<'_>,
    interior_edges: &[usize],
    powered_base: (NonZeroU32, &[f64]),
    weights: &[f64],
    free_rows: (&[Option<usize>], &mut [f64]),
    directions: &mut [f64],
    execution: (Executor, &CancellationToken),
) -> Result<Vec<f64>, SurfaceComputationError> {
    let (symmetry_order, levi_civita_power) = powered_base;
    let (positions, diagonal) = free_rows;
    let order = f64::from(symmetry_order.get());
    let antipodal_limit = 256.0 * f64::EPSILON * order;
    let scale = diagonal.iter().copied().fold(0.0_f64, f64::max);
    let mut lifts = Vec::with_capacity(interior_edges.len());
    for (row, &edge) in interior_edges.iter().enumerate() {
        let (source, target, _) = dual.edge(edge)?;
        lifts.push(certified_connection_angle(
            complex_at(levi_civita_power, row)?,
            complex_at(directions, source)?,
            complex_at(directions, target)?,
            antipodal_limit,
        )?);
    }

    let angles = {
        let capacity = diagonal
            .len()
            .saturating_add(interior_edges.len().saturating_mul(2));
        let mut triplets = Vec::with_capacity(capacity);
        for (position, &value) in diagonal.iter().enumerate() {
            triplets.push(Triplet::new(position, position, value / scale));
        }
        let mut values = vec![0.0_f64; diagonal.len()];
        for ((&edge, &weight), &lift) in interior_edges.iter().zip(weights).zip(&lifts) {
            let (source, target, _) = dual.edge(edge)?;
            if let Some(position) = positions[source] {
                values[position] += weight * lift;
            }
            if let Some(position) = positions[target] {
                values[position] -= weight * lift;
            }
            if let (Some(source), Some(target)) = (positions[source], positions[target]) {
                triplets.push(Triplet::new(source, target, -weight / scale));
                triplets.push(Triplet::new(target, source, -weight / scale));
            }
        }
        solve_weighted_positive_system(diagonal.len(), &triplets, scale, &mut values, execution)?;
        values
    };

    {
        diagonal.fill(0.0);
        let mut gradient_scale = 0.0_f64;
        for ((&edge, &weight), lift) in interior_edges.iter().zip(weights).zip(&mut lifts) {
            let (source, target, _) = dual.edge(edge)?;
            let source_angle = positions[source].map_or(0.0, |position| angles[position]);
            let target_angle = positions[target].map_or(0.0, |position| angles[position]);
            *lift += target_angle - source_angle;
            gradient_scale = gradient_scale.max(weight * lift.abs());
            if let Some(position) = positions[source] {
                diagonal[position] -= weight * *lift;
            }
            if let Some(position) = positions[target] {
                diagonal[position] += weight * *lift;
            }
        }
        if relative_max_residual(diagonal, gradient_scale)? > 1.0e-10 {
            return Err(SolveError::Numerical.into());
        }
        for (face, position) in positions.iter().enumerate() {
            if let Some(position) = position {
                let angle = angles[*position];
                let correction = [angle.cos(), angle.sin()];
                let direction = complex_multiply(complex_at(directions, face)?, correction);
                directions[2 * face..2 * face + 2].copy_from_slice(&normalize_complex(direction)?);
            }
        }
    }

    let branch_limit = 8192.0 * f64::EPSILON * order;
    for (row, (&edge, selected)) in interior_edges.iter().zip(&mut lifts).enumerate() {
        let selected_value = *selected;
        if std::f64::consts::PI - selected_value.abs() <= antipodal_limit {
            return Err(SurfaceError::Unrepresentable.into());
        }
        let (source, target, _) = dual.edge(edge)?;
        let principal = certified_connection_angle(
            complex_at(levi_civita_power, row)?,
            complex_at(directions, source)?,
            complex_at(directions, target)?,
            antipodal_limit,
        )?;
        if (principal - selected_value).abs() > branch_limit {
            return Err(SolveError::Numerical.into());
        }
        *selected = principal;
    }
    Ok(lifts)
}

fn boundary_aligned_direction_field(
    surface: &Arc<TriangleSurface>,
    symmetry_order: NonZeroU32,
    metric: &Metric,
    boundary_angle_offset: f64,
    executor: Executor,
    cancellation: &CancellationToken,
) -> Result<FaceDirectionField, SurfaceComputationError> {
    let topology = surface.realization().topology();
    let face_count = surface.face_count();
    let (mut directions, positions) = {
        let (target_faces, target_directions) =
            surface.boundary_power_directions(symmetry_order, boundary_angle_offset)?;
        if target_directions.len() != target_faces.len().saturating_mul(2) {
            return Err(SolveError::Numerical.into());
        }
        let mut boundary_faces = vec![false; face_count];
        let mut directions = vec![0.0_f64; 2 * face_count];
        for (row, &face) in target_faces.iter().enumerate() {
            let boundary = boundary_faces
                .get_mut(face)
                .ok_or(SurfaceError::IndexOutside)?;
            if std::mem::replace(boundary, true) {
                return Err(SolveError::Numerical.into());
            }
            directions[2 * face..2 * face + 2]
                .copy_from_slice(&complex_at(&target_directions, row)?);
        }
        let positions = boundary_faces
            .into_iter()
            .scan(0, |position, boundary| {
                let reduced = (!boundary).then_some(*position);
                *position += usize::from(!boundary);
                Some(reduced)
            })
            .collect::<Vec<_>>();
        (directions, positions)
    };

    let levi_civita = surface.levi_civita_connection()?;
    let interior_edges = levi_civita.interior_edges.as_ref();
    let mut levi_civita_power = Vec::with_capacity(levi_civita.transports().len());
    for transport in levi_civita.transports().chunks_exact(2) {
        let transport = transport.try_into().map_err(|_| SolveError::Numerical)?;
        levi_civita_power.extend(powered_transport(transport, symmetry_order, 0.0)?);
    }
    let dual = dual_edges(topology)?;
    let hodge = metric
        .hodge_coefficients_slice(1)
        .map_err(|_| SolveError::Numerical)?;
    let mut weights = Vec::with_capacity(interior_edges.len());
    let mut diagonal = vec![0.0_f64; positions.iter().flatten().count()];
    for &edge in interior_edges {
        let weight = 1.0 / *hodge.get(edge).ok_or(SolveError::Numerical)?;
        if !weight.is_finite() || weight <= 0.0 {
            return Err(SolveError::Numerical.into());
        }
        weights.push(weight);
        let (source, target, _) = dual.edge(edge)?;
        if let Some(position) = positions[source] {
            diagonal[position] += weight;
        }
        if let Some(position) = positions[target] {
            diagonal[position] += weight;
        }
    }
    relaxed_boundary_extension(
        dual,
        interior_edges,
        levi_civita_power.as_slice(),
        &weights,
        (&positions, &diagonal),
        &mut directions,
        (executor, cancellation),
    )?;
    let deviations = fixed_sector_deviations(
        dual,
        interior_edges,
        (symmetry_order, levi_civita_power.as_slice()),
        &weights,
        (&positions, &mut diagonal),
        &mut directions,
        (executor, cancellation),
    )?;

    check_cancelled(cancellation)?;
    let order = f64::from(symmetry_order.get());
    let anchor = complex_at(&directions, 0)?;
    let anchor_angle = anchor[1].atan2(anchor[0]) / order;
    let field = levi_civita
        .with_powered_deviations(symmetry_order, &deviations)?
        .require_integrable()?
        .direction_field(anchor_angle)?;
    let field_limit = 8192.0
        * f64::EPSILON
        * order
        * f64::from(u32::try_from(interior_edges.len().max(1)).unwrap_or(u32::MAX));
    for face in 0..face_count {
        let observed = complex_at(field.power_directions(), face)?;
        let expected = complex_at(&directions, face)?;
        let error = (observed[0] - expected[0])
            .abs()
            .max((observed[1] - expected[1]).abs());
        if error > field_limit {
            return Err(SolveError::Numerical.into());
        }
    }
    field.singularities()?;
    check_cancelled(cancellation)?;
    Ok(field)
}
