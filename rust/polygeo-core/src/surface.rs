use std::fmt;
use std::num::NonZeroU32;
use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::ToPrimitive;
use once_cell::sync::OnceCell;

use crate::problem::adaptive_product_value;
use crate::{
    Binary64Chain, Binary64ChainSpace, Binary64Cochain, Binary64CochainSpace, Binary64Element,
    CanonicalBoundary, ComplexCore, EuclideanRealization, IntegralCochain, IntegralDualCycleBasis,
    NondegenerateCapability, PairingCapability, PositiveMetric, TopologyError,
};

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn norm(value: [f64; 3]) -> f64 {
    let scale = value.into_iter().map(f64::abs).fold(0.0_f64, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return scale;
    }
    let scaled = value.map(|entry| entry / scale);
    scale * dot(scaled, scaled).sqrt()
}

fn normalize(value: [f64; 3]) -> Option<[f64; 3]> {
    let scale = value.into_iter().map(f64::abs).fold(0.0_f64, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return None;
    }
    let scaled = value.map(|entry| entry / scale);
    let length = dot(scaled, scaled).sqrt();
    let normalized = scaled.map(|entry| entry / length);
    normalized
        .into_iter()
        .all(f64::is_finite)
        .then_some(normalized)
}

fn triangle_angle(left: [f64; 3], right: [f64; 3]) -> Option<f64> {
    let left = normalize(left)?;
    let right = normalize(right)?;
    let sine = norm(cross(left, right));
    let cosine = dot(left, right);
    let angle = sine.atan2(cosine);
    angle.is_finite().then_some(angle)
}

type LocalDifferentialRows = ([usize; 3], [[f64; 3]; 3], f64, f64);
pub(crate) type LocalConformalCoefficients = ([usize; 3], [[f64; 2]; 3]);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Support {
    Vertex,
    Face,
}

/// One contiguous ambient-vector field over a realized entity basis.
///
/// `VertexVectors` and `FaceVectors` are semantic aliases over this carrier;
/// support is admitted by surface-owned constructors rather than caller tags.
#[derive(Clone, Debug)]
pub struct EntityVectors {
    realization: Arc<EuclideanRealization>,
    support: Support,
    values: Arc<[f64]>,
}

pub type VertexVectors = EntityVectors;
pub type FaceVectors = EntityVectors;

impl EntityVectors {
    fn admit(
        realization: Arc<EuclideanRealization>,
        support: Support,
        values: Vec<f64>,
    ) -> Result<Self, SurfaceError> {
        let entities = match support {
            Support::Vertex => realization.topology().vertex_count(),
            Support::Face => realization
                .topology()
                .basis(2)
                .map_err(SurfaceError::Topology)?
                .row_count(),
        };
        let expected = entities
            .checked_mul(realization.ambient_dimension())
            .ok_or(SurfaceError::Overflow)?;
        if values.len() != expected {
            return Err(SurfaceError::FieldShape);
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(SurfaceError::NonFinite);
        }
        Ok(Self {
            realization,
            support,
            values: values.into(),
        })
    }

    #[must_use]
    pub const fn realization(&self) -> &Arc<EuclideanRealization> {
        &self.realization
    }

    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.values.len() / self.fiber_dimension()
    }

    #[must_use]
    pub fn fiber_dimension(&self) -> usize {
        self.realization.ambient_dimension()
    }

    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    #[must_use]
    pub const fn is_vertex_supported(&self) -> bool {
        matches!(self.support, Support::Vertex)
    }

    #[must_use]
    pub const fn is_face_supported(&self) -> bool {
        matches!(self.support, Support::Face)
    }

    /// Return an owned field whose entity vectors have unit length.
    ///
    /// # Errors
    ///
    /// Rejects a zero or unrepresentable vector.
    pub fn normalized(&self) -> Result<Self, SurfaceError> {
        let dimension = self.fiber_dimension();
        let mut values = Vec::with_capacity(self.values.len());
        for row in self.values.chunks_exact(dimension) {
            let scale = row.iter().copied().map(f64::abs).fold(0.0, f64::max);
            if scale == 0.0 || !scale.is_finite() {
                return Err(SurfaceError::ZeroVector);
            }
            let length = row
                .iter()
                .map(|value| (value / scale) * (value / scale))
                .sum::<f64>()
                .sqrt();
            values.extend(row.iter().map(|value| value / scale / length));
        }
        Self::admit(Arc::clone(&self.realization), self.support, values)
    }
}

/// Failure to admit or evaluate a triangle-surface value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SurfaceError {
    Topology(TopologyError),
    AmbientDimension,
    FieldShape,
    NonFinite,
    ZeroVector,
    BoundaryPresent,
    OwnerMismatch,
    NotIntegrable,
    IndexOutside,
    Overflow,
    Unrepresentable,
    TimeStep,
    CoincidentAnchor,
    AnchorNotBoundary,
}

impl fmt::Display for SurfaceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Topology(_) => "triangle-surface topology requirement failed",
            Self::AmbientDimension => "triangle-surface operation requires ambient dimension three",
            Self::FieldShape => "entity-field shape does not match its support",
            Self::NonFinite => "surface value is not finite",
            Self::ZeroVector => "cannot normalize a zero vector",
            Self::BoundaryPresent => "surface operation requires an empty boundary",
            Self::OwnerMismatch => "surface values require the same topology owner",
            Self::NotIntegrable => "surface connection is not integrable",
            Self::IndexOutside => "surface index lies outside its admitted basis",
            Self::Overflow => "surface size arithmetic overflowed",
            Self::Unrepresentable => "surface result is not representable",
            Self::TimeStep => "flow time step must be positive and finite",
            Self::CoincidentAnchor => "surface anchors must be distinct vertices",
            Self::AnchorNotBoundary => "surface anchor must lie on the disk boundary",
        })
    }
}

impl std::error::Error for SurfaceError {}

impl From<TopologyError> for SurfaceError {
    fn from(error: TopologyError) -> Self {
        Self::Topology(error)
    }
}

/// Certified diagnostics for one atomically published flow step.
#[derive(Clone, Copy, Debug)]
pub struct FlowEvidence {
    energy_before: f64,
    energy_after: f64,
    residual_bound: f64,
    centroid_residual_bound: f64,
}

impl FlowEvidence {
    #[must_use]
    pub const fn energy_before(self) -> f64 {
        self.energy_before
    }
    #[must_use]
    pub const fn energy_after(self) -> f64 {
        self.energy_after
    }
    #[must_use]
    pub const fn residual_bound(self) -> f64 {
        self.residual_bound
    }
    #[must_use]
    pub const fn centroid_residual_bound(self) -> f64 {
        self.centroid_residual_bound
    }

    pub(crate) const fn new(
        energy_before: f64,
        energy_after: f64,
        residual_bound: f64,
        centroid_residual_bound: f64,
    ) -> Self {
        Self {
            energy_before,
            energy_after,
            residual_bound,
            centroid_residual_bound,
        }
    }
}

/// One admitted deformation produced by a frozen-metric flow solve.
#[derive(Clone, Debug)]
pub struct FlowStep {
    target: Arc<EuclideanRealization>,
    evidence: FlowEvidence,
}

/// Certified diagnostics for one least-squares conformal parameterization.
#[derive(Clone, Copy, Debug)]
pub struct LeastSquaresConformalMapEvidence {
    required_rank: usize,
    observed_rank: usize,
    condition_indicator: f64,
    residual_bound: f64,
    minimum_normalized_signed_twice_area: f64,
    exact_fallback_faces: usize,
}

impl LeastSquaresConformalMapEvidence {
    #[must_use]
    pub const fn required_rank(self) -> usize {
        self.required_rank
    }
    #[must_use]
    pub const fn observed_rank(self) -> usize {
        self.observed_rank
    }
    #[must_use]
    pub const fn condition_indicator(self) -> f64 {
        self.condition_indicator
    }
    #[must_use]
    pub const fn residual_bound(self) -> f64 {
        self.residual_bound
    }
    #[must_use]
    pub const fn minimum_normalized_signed_twice_area(self) -> f64 {
        self.minimum_normalized_signed_twice_area
    }
    #[must_use]
    pub const fn exact_fallback_faces(self) -> usize {
        self.exact_fallback_faces
    }

    pub(crate) const fn new(
        required_rank: usize,
        observed_rank: usize,
        condition_indicator: f64,
        residual_bound: f64,
        minimum_normalized_signed_twice_area: f64,
        exact_fallback_faces: usize,
    ) -> Self {
        Self {
            required_rank,
            observed_rank,
            condition_indicator,
            residual_bound,
            minimum_normalized_signed_twice_area,
            exact_fallback_faces,
        }
    }
}

/// One admitted planar realization and its LSCM computation evidence.
#[derive(Clone, Debug)]
pub struct LeastSquaresConformalMapSolution {
    realization: Arc<EuclideanRealization>,
    evidence: LeastSquaresConformalMapEvidence,
}

impl LeastSquaresConformalMapSolution {
    #[must_use]
    pub const fn realization(&self) -> &Arc<EuclideanRealization> {
        &self.realization
    }
    #[must_use]
    pub const fn evidence(&self) -> LeastSquaresConformalMapEvidence {
        self.evidence
    }

    pub(crate) const fn new(
        realization: Arc<EuclideanRealization>,
        evidence: LeastSquaresConformalMapEvidence,
    ) -> Self {
        Self {
            realization,
            evidence,
        }
    }
}

impl FlowStep {
    #[must_use]
    pub const fn target(&self) -> &Arc<EuclideanRealization> {
        &self.target
    }
    #[must_use]
    pub const fn evidence(&self) -> FlowEvidence {
        self.evidence
    }

    pub(crate) const fn new(target: Arc<EuclideanRealization>, evidence: FlowEvidence) -> Self {
        Self { target, evidence }
    }
}

#[derive(Debug)]
struct SurfaceRows {
    first: Box<[f64]>,
    second: Box<[f64]>,
}

#[derive(Debug)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "consumed by bounded field construction")
)]
pub(crate) struct BoundaryDirectionTargets {
    component_offsets: Box<[usize]>,
    component_edges: Box<[usize]>,
    faces: Box<[usize]>,
    power_directions: Box<[f64]>,
}

/// One admitted oriented triangle-manifold realization in ambient dimension three.
pub struct TriangleSurface {
    realization: Arc<EuclideanRealization>,
    rows: OnceCell<SurfaceRows>,
}

impl fmt::Debug for TriangleSurface {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TriangleSurface")
            .field("faces", &self.face_count())
            .field("edges", &self.edge_count())
            .finish_non_exhaustive()
    }
}

impl TriangleSurface {
    /// Admit dimension-specific surface laws over one realization owner.
    ///
    /// # Errors
    ///
    /// Requires ambient dimension three and admitted triangle/orientation laws.
    pub fn admit(realization: Arc<EuclideanRealization>) -> Result<Arc<Self>, SurfaceError> {
        if realization.ambient_dimension() != 3 {
            return Err(SurfaceError::AmbientDimension);
        }
        realization.topology().refine_triangle()?;
        realization.topology().refine_oriented()?;
        Ok(Arc::new(Self {
            realization,
            rows: OnceCell::new(),
        }))
    }

    #[must_use]
    pub const fn realization(&self) -> &Arc<EuclideanRealization> {
        &self.realization
    }

    #[must_use]
    pub fn face_count(&self) -> usize {
        self.realization
            .topology()
            .basis(2)
            .map_or(0, crate::Basis::row_count)
    }

    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.realization
            .topology()
            .basis(1)
            .map_or(0, crate::Basis::row_count)
    }

    /// Admit one owned contiguous vertex-vector field on this realization.
    ///
    /// # Errors
    ///
    /// Rejects a shape mismatch or nonfinite coefficient.
    pub fn vertex_vectors(&self, values: Vec<f64>) -> Result<VertexVectors, SurfaceError> {
        EntityVectors::admit(Arc::clone(&self.realization), Support::Vertex, values)
    }

    /// Admit one owned contiguous face-vector field on this realization.
    ///
    /// # Errors
    ///
    /// Rejects a shape mismatch or nonfinite coefficient.
    pub fn face_vectors(&self, values: Vec<f64>) -> Result<FaceVectors, SurfaceError> {
        EntityVectors::admit(Arc::clone(&self.realization), Support::Face, values)
    }

    /// Compute the constant piecewise-affine scalar gradient on every face.
    ///
    /// # Errors
    ///
    /// Rejects a foreign, selected, or non-vertex cochain and unrepresentable arithmetic.
    pub fn gradient(&self, source: &Binary64Cochain) -> Result<FaceVectors, SurfaceError> {
        let expected = Binary64CochainSpace::full(Arc::clone(self.realization.topology()), 0)?;
        if !expected.same_space(source.space()) {
            return Err(SurfaceError::OwnerMismatch);
        }
        let mut values = Vec::with_capacity(3 * self.face_count());
        for face in 0..self.face_count() {
            let (vertices, rows, twice_area, scale) = self.local_differential_rows(face)?;
            let base = source.coefficients()[vertices[0]];
            let first = difference(source.coefficients()[vertices[1]], base)?;
            let second = difference(source.coefficients()[vertices[2]], base)?;
            for (&first_row, &second_row) in rows[1].iter().zip(&rows[2]) {
                let first_gradient = first_row / twice_area / scale;
                let second_gradient = second_row / twice_area / scale;
                if !first_gradient.is_finite() || !second_gradient.is_finite() {
                    return Err(SurfaceError::Unrepresentable);
                }
                values.push(product_sum([
                    (first, first_gradient),
                    (second, second_gradient),
                ])?);
            }
        }
        EntityVectors::admit(Arc::clone(&self.realization), Support::Face, values)
    }

    /// Compute the weak divergence load of one face-supported ambient vector field.
    ///
    /// The returned degree-zero chain is the negative adjoint of [`Self::gradient`].
    /// On a surface with boundary this selects the natural no-flux boundary law.
    ///
    /// # Errors
    ///
    /// Rejects a foreign or vertex-supported field and unrepresentable arithmetic.
    pub fn divergence(&self, field: &FaceVectors) -> Result<Binary64Chain, SurfaceError> {
        if !Arc::ptr_eq(field.realization(), &self.realization) {
            return Err(SurfaceError::OwnerMismatch);
        }
        if !field.is_face_supported() {
            return Err(SurfaceError::FieldShape);
        }
        let vertex_count = self.realization.topology().vertex_count();
        let mut sums = vec![0.0; vertex_count];
        let mut corrections = vec![0.0; vertex_count];
        for face in 0..self.face_count() {
            let (vertices, rows, _, scale) = self.local_differential_rows(face)?;
            let vector = row3(field.values(), face)?;
            for (vertex, row) in vertices.into_iter().zip(rows) {
                let weighted = row.map(|value| 0.5 * scale * value);
                if weighted.into_iter().any(|value| !value.is_finite()) {
                    return Err(SurfaceError::Unrepresentable);
                }
                let contribution = -product_sum([
                    (weighted[0], vector[0]),
                    (weighted[1], vector[1]),
                    (weighted[2], vector[2]),
                ])?;
                compensated_add(&mut sums[vertex], &mut corrections[vertex], contribution)?;
            }
        }
        for (sum, correction) in sums.iter_mut().zip(corrections) {
            *sum = product_sum([(*sum, 1.0), (correction, 1.0)])?;
        }
        let space = Binary64ChainSpace::full(Arc::clone(self.realization.topology()), 0)?;
        Binary64Element::admit(space, sums).map_err(|_| SurfaceError::Unrepresentable)
    }

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

    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "consumed by bounded field construction")
    )]
    pub(crate) fn boundary_direction_targets(
        &self,
        symmetry_order: NonZeroU32,
        boundary_angle_offset: f64,
    ) -> Result<BoundaryDirectionTargets, SurfaceError> {
        if !boundary_angle_offset.is_finite() {
            return Err(SurfaceError::NonFinite);
        }
        self.realization
            .topology()
            .refine_regular()?
            .with_boundary()?;
        let (component_offsets, component_edges) =
            oriented_boundary_components(self.realization.topology())?;
        let (faces, power_directions) = boundary_face_power_directions(
            self,
            symmetry_order,
            boundary_angle_offset,
            &component_edges,
        )?;
        Ok(BoundaryDirectionTargets {
            component_offsets: component_offsets.into_boxed_slice(),
            component_edges: component_edges.into_boxed_slice(),
            faces: faces.into_boxed_slice(),
            power_directions: power_directions.into_boxed_slice(),
        })
    }

    fn surface_rows(&self) -> Result<&SurfaceRows, SurfaceError> {
        self.rows.get_or_try_init(|| {
            let mut first = Vec::with_capacity(3 * self.face_count());
            let mut second = Vec::with_capacity(3 * self.face_count());
            for face in 0..self.face_count() {
                let points = self.face_points(face)?;
                let normal = self.face_normal(face)?;
                let first_axis = normalize(subtract(points[1], points[0]))
                    .ok_or(SurfaceError::Unrepresentable)?;
                let second_axis =
                    normalize(cross(normal, first_axis)).ok_or(SurfaceError::Unrepresentable)?;
                first.extend(first_axis);
                second.extend(second_axis);
            }
            Ok(SurfaceRows {
                first: first.into_boxed_slice(),
                second: second.into_boxed_slice(),
            })
        })
    }

    /// Compute oriented unit normals without retaining a third frame row.
    ///
    /// # Errors
    ///
    /// Returns an allocation, shape, or representability failure.
    pub fn face_unit_normals(&self) -> Result<FaceVectors, SurfaceError> {
        let mut values = Vec::with_capacity(3 * self.face_count());
        for face in 0..self.face_count() {
            values.extend(self.face_normal(face)?);
        }
        EntityVectors::admit(Arc::clone(&self.realization), Support::Face, values)
    }

    /// Compute normalized sums of incident oriented face normals.
    ///
    /// # Errors
    ///
    /// Requires a closed surface and nonzero representable vertex sums.
    pub fn uniform_vertex_normals(&self) -> Result<VertexVectors, SurfaceError> {
        self.require_closed()?;
        let mut values = vec![0.0; 3 * self.realization.topology().vertex_count()];
        let faces = self.realization.topology().basis(2)?;
        for face in 0..self.face_count() {
            let normal = self.face_normal(face)?;
            for &vertex in faces.row(face).ok_or(SurfaceError::IndexOutside)? {
                add_row(&mut values, vertex, normal)?;
            }
        }
        EntityVectors::admit(Arc::clone(&self.realization), Support::Vertex, values)?.normalized()
    }

    /// Compute normalized tip-angle-weighted incident face normals.
    ///
    /// # Errors
    ///
    /// Requires a closed surface and representable triangle angles.
    pub fn tip_angle_vertex_normals(&self) -> Result<VertexVectors, SurfaceError> {
        self.require_closed()?;
        let mut values = vec![0.0; 3 * self.realization.topology().vertex_count()];
        let faces = self.realization.topology().basis(2)?;
        for face in 0..self.face_count() {
            let vertices = faces.row(face).ok_or(SurfaceError::IndexOutside)?;
            let points = self.face_points(face)?;
            let normal = self.face_normal(face)?;
            for corner in 0..3 {
                let angle = triangle_angle(
                    subtract(points[(corner + 1) % 3], points[corner]),
                    subtract(points[(corner + 2) % 3], points[corner]),
                )
                .ok_or(SurfaceError::Unrepresentable)?;
                add_row(
                    &mut values,
                    vertices[corner],
                    normal.map(|value| angle * value),
                )?;
            }
        }
        EntityVectors::admit(Arc::clone(&self.realization), Support::Vertex, values)?.normalized()
    }

    /// Compute normalized sphere-inscribed cyclic-edge normal directions.
    ///
    /// # Errors
    ///
    /// Requires a closed surface and representable edge directions and weights.
    pub fn sphere_inscribed_vertex_normals(&self) -> Result<VertexVectors, SurfaceError> {
        self.require_closed()?;
        let faces = self.realization.topology().basis(2)?;
        let orientations = self.realization.topology().orientation(2)?;
        let mut entries = Vec::with_capacity(3 * self.face_count());
        let mut offset = f64::NEG_INFINITY;
        for (face, &orientation) in orientations.iter().enumerate() {
            let canonical = faces.row(face).ok_or(SurfaceError::IndexOutside)?;
            let vertices = if orientation < 0 {
                [canonical[0], canonical[2], canonical[1]]
            } else {
                [canonical[0], canonical[1], canonical[2]]
            };
            let points = [
                self.point(vertices[0])?,
                self.point(vertices[1])?,
                self.point(vertices[2])?,
            ];
            for corner in 0..3 {
                let left = subtract(points[(corner + 1) % 3], points[corner]);
                let right = subtract(points[(corner + 2) % 3], points[corner]);
                let left_length = norm(left);
                let right_length = norm(right);
                let left = normalize(left).ok_or(SurfaceError::Unrepresentable)?;
                let right = normalize(right).ok_or(SurfaceError::Unrepresentable)?;
                let log_weight = -(left_length.ln() + right_length.ln());
                if !log_weight.is_finite() {
                    return Err(SurfaceError::Unrepresentable);
                }
                offset = offset.max(log_weight);
                entries.push((vertices[corner], cross(left, right), log_weight));
            }
        }
        let mut values = vec![0.0; 3 * self.realization.topology().vertex_count()];
        for (vertex, direction, log_weight) in entries {
            let weight = (log_weight - offset).exp();
            add_row(&mut values, vertex, direction.map(|value| weight * value))?;
        }
        EntityVectors::admit(Arc::clone(&self.realization), Support::Vertex, values)?.normalized()
    }

    /// Compute the gradient of total triangle area at canonical vertices.
    ///
    /// # Errors
    ///
    /// Returns an unrepresentable arithmetic failure.
    pub fn surface_area_gradient(&self) -> Result<VertexVectors, SurfaceError> {
        let mut values = vec![0.0; 3 * self.realization.topology().vertex_count()];
        let faces = self.realization.topology().basis(2)?;
        for face in 0..self.face_count() {
            let vertices = faces.row(face).ok_or(SurfaceError::IndexOutside)?;
            let points = self.face_points(face)?;
            let geometric_normal = normalize(cross(
                subtract(points[1], points[0]),
                subtract(points[2], points[0]),
            ))
            .ok_or(SurfaceError::Unrepresentable)?;
            for corner in 0..3 {
                let edge = subtract(points[(corner + 1) % 3], points[(corner + 2) % 3]);
                let contribution = cross(edge, geometric_normal).map(|value| 0.5 * value);
                add_row(&mut values, vertices[corner], contribution)?;
            }
        }
        EntityVectors::admit(Arc::clone(&self.realization), Support::Vertex, values)
    }

    /// Compute the signed enclosed-volume gradient at canonical vertices.
    ///
    /// # Errors
    ///
    /// Requires a closed surface and representable arithmetic.
    pub fn volume_gradient(&self) -> Result<VertexVectors, SurfaceError> {
        self.require_closed()?;
        let mut values = vec![0.0; 3 * self.realization.topology().vertex_count()];
        let faces = self.realization.topology().basis(2)?;
        let areas = self
            .realization
            .primal_measures(2)
            .map_err(|_| SurfaceError::Unrepresentable)?;
        for (face, &area) in areas.iter().enumerate() {
            let vertices = faces.row(face).ok_or(SurfaceError::IndexOutside)?;
            let contribution = self.face_normal(face)?.map(|value| area * value / 3.0);
            for &vertex in vertices {
                add_row(&mut values, vertex, contribution)?;
            }
        }
        EntityVectors::admit(Arc::clone(&self.realization), Support::Vertex, values)
    }

    /// Compute integrated Gaussian curvature as canonical vertex angle defects.
    ///
    /// # Errors
    ///
    /// Returns a topology, arithmetic, or binary64 admission failure.
    pub fn gaussian_curvature_measure(&self) -> Result<Binary64Cochain, SurfaceError> {
        let vertex_count = self.realization.topology().vertex_count();
        let mut angle_sums = vec![0.0; vertex_count];
        let faces = self.realization.topology().basis(2)?;
        for face in 0..self.face_count() {
            let vertices = faces.row(face).ok_or(SurfaceError::IndexOutside)?;
            let points = self.face_points(face)?;
            for corner in 0..3 {
                let left = subtract(points[(corner + 1) % 3], points[corner]);
                let right = subtract(points[(corner + 2) % 3], points[corner]);
                angle_sums[vertices[corner]] +=
                    triangle_angle(left, right).ok_or(SurfaceError::Unrepresentable)?;
            }
        }
        let boundary = self
            .realization
            .topology()
            .refine_regular()?
            .boundary_mask(0)?;
        let coefficients = angle_sums
            .into_iter()
            .zip(boundary)
            .map(|(sum, boundary)| if boundary { std::f64::consts::PI } else { 2.0 * std::f64::consts::PI } - sum)
            .collect();
        let space = Binary64CochainSpace::full(Arc::clone(self.realization.topology()), 0)?;
        Binary64Element::admit(space, coefficients).map_err(|_| SurfaceError::Unrepresentable)
    }

    /// Apply the positive metric Hodge Laplacian to all position coordinates.
    ///
    /// # Errors
    ///
    /// Requires the same realization, a closed surface, and representable operator action.
    pub fn mean_curvature_vectors(
        &self,
        metric: &PositiveMetric,
    ) -> Result<VertexVectors, SurfaceError> {
        self.require_closed()?;
        if !Arc::ptr_eq(metric.realization(), &self.realization) {
            return Err(SurfaceError::OwnerMismatch);
        }
        let operator = metric
            .laplacian(0)
            .map_err(|_| SurfaceError::Unrepresentable)?;
        let vertex_count = self.realization.topology().vertex_count();
        let mut values = vec![0.0; 3 * vertex_count];
        for axis in 0..3 {
            let coordinates = self
                .realization
                .positions()
                .chunks_exact(3)
                .map(|point| point[axis])
                .collect();
            let input = Binary64Element::admit(operator.source().clone(), coordinates)
                .map_err(|_| SurfaceError::Unrepresentable)?;
            let output = operator
                .apply(&input)
                .map_err(|_| SurfaceError::Unrepresentable)?;
            for (vertex, coefficient) in output.coefficients().iter().copied().enumerate() {
                values[3 * vertex + axis] = coefficient;
            }
        }
        EntityVectors::admit(Arc::clone(&self.realization), Support::Vertex, values)
    }

    /// Construct canonical Levi-Civita face transport.
    ///
    /// # Errors
    ///
    /// Requires a connected regular surface and representable frames.
    pub fn levi_civita_connection(
        self: &Arc<Self>,
    ) -> Result<Arc<SurfaceConnection>, SurfaceError> {
        let edge_count = interior_edge_indices(self.realization.topology())?.len();
        self.connection(NonZeroU32::MIN, &vec![0.0; edge_count])
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
        self.realization.topology().refine_regular()?;
        self.realization.topology().refine_connected()?;
        let interior_edges = interior_edge_indices(self.realization.topology())?;
        if deviations.len() != interior_edges.len() {
            return Err(SurfaceError::FieldShape);
        }
        if deviations.iter().any(|value| !value.is_finite()) {
            return Err(SurfaceError::NonFinite);
        }
        let dual = dual_edges(self.realization.topology())?;
        let edges = self.realization.topology().basis(1)?;
        let first = self.first_frame_axes()?;
        let second = self.second_frame_axes()?;
        let mut transports = Vec::with_capacity(2 * interior_edges.len());
        for (&edge, &deviation_angle) in interior_edges.iter().zip(deviations) {
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
            let base = normalize_complex([
                dot(rotated, row3(first, target)?),
                dot(rotated, row3(second, target)?),
            ])?;
            let deviation = [deviation_angle.cos(), deviation_angle.sin()];
            let powered_base = complex_power(base, i64::from(symmetry_order.get()))?;
            transports.extend(normalize_complex(complex_multiply(
                powered_base,
                deviation,
            ))?);
        }
        Ok(Arc::new(SurfaceConnection {
            surface: Arc::clone(self),
            symmetry_order,
            interior_edges: interior_edges.into(),
            transports: transports.into(),
            evidence: OnceCell::new(),
        }))
    }

    fn require_closed(&self) -> Result<(), SurfaceError> {
        self.realization
            .topology()
            .refine_regular()?
            .without_boundary()
            .map(|_| ())
            .map_err(|_| SurfaceError::BoundaryPresent)
    }

    fn face_normal(&self, face: usize) -> Result<[f64; 3], SurfaceError> {
        let points = self.face_points(face)?;
        let mut normal = normalize(cross(
            subtract(points[1], points[0]),
            subtract(points[2], points[0]),
        ))
        .ok_or(SurfaceError::Unrepresentable)?;
        let orientation = *self
            .realization
            .topology()
            .orientation(2)?
            .get(face)
            .ok_or(SurfaceError::IndexOutside)?;
        if orientation < 0 {
            normal = normal.map(|value| -value);
        }
        Ok(normal)
    }

    fn face_points(&self, face: usize) -> Result<[[f64; 3]; 3], SurfaceError> {
        let vertices = self
            .realization
            .topology()
            .basis(2)?
            .row(face)
            .ok_or(SurfaceError::IndexOutside)?;
        Ok([
            self.point(vertices[0])?,
            self.point(vertices[1])?,
            self.point(vertices[2])?,
        ])
    }

    fn point(&self, vertex: usize) -> Result<[f64; 3], SurfaceError> {
        row3(self.realization.positions(), vertex)
    }

    fn local_differential_rows(&self, face: usize) -> Result<LocalDifferentialRows, SurfaceError> {
        let vertices: [usize; 3] = self
            .realization
            .topology()
            .basis(2)?
            .row(face)
            .and_then(|row| row.try_into().ok())
            .ok_or(SurfaceError::IndexOutside)?;
        let points = self.face_points(face)?;
        let first = subtract(points[1], points[0]);
        let second = subtract(points[2], points[0]);
        let scale = first
            .into_iter()
            .chain(second)
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        if scale == 0.0 || !scale.is_finite() {
            return Err(SurfaceError::Unrepresentable);
        }
        let first = first.map(|value| value / scale);
        let second = second.map(|value| value / scale);
        let area_vector = cross(first, second);
        let twice_area = norm(area_vector);
        if twice_area == 0.0 || !twice_area.is_finite() {
            return Err(SurfaceError::Unrepresentable);
        }
        let normal = area_vector.map(|value| value / twice_area);
        let rows = [
            cross(normal, subtract(second, first)),
            cross(normal, second.map(|value| -value)),
            cross(normal, first),
        ];
        Ok((vertices, rows, twice_area, scale))
    }

    /// Construct the dimensionless complex LSCM row in the admitted face orientation.
    pub(crate) fn oriented_local_conformal_coefficients(
        &self,
        face: usize,
    ) -> Result<LocalConformalCoefficients, SurfaceError> {
        let canonical: [usize; 3] = self
            .realization
            .topology()
            .basis(2)?
            .row(face)
            .and_then(|row| row.try_into().ok())
            .ok_or(SurfaceError::IndexOutside)?;
        let orientation = *self
            .realization
            .topology()
            .orientation(2)?
            .get(face)
            .ok_or(SurfaceError::IndexOutside)?;
        let vertices = if orientation < 0 {
            [canonical[0], canonical[2], canonical[1]]
        } else {
            canonical
        };
        let points = [
            self.point(vertices[0])?,
            self.point(vertices[1])?,
            self.point(vertices[2])?,
        ];
        let first = subtract(points[1], points[0]);
        let second = subtract(points[2], points[0]);
        let scale = first
            .into_iter()
            .chain(second)
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        if scale == 0.0 || !scale.is_finite() {
            return Err(SurfaceError::Unrepresentable);
        }
        let first = first.map(|value| value / scale);
        let second = second.map(|value| value / scale);
        let first_length = norm(first);
        let twice_area = norm(cross(first, second));
        if first_length == 0.0
            || twice_area == 0.0
            || !first_length.is_finite()
            || !twice_area.is_finite()
        {
            return Err(SurfaceError::Unrepresentable);
        }
        let x2 = dot(first, second) / first_length;
        let y2 = twice_area / first_length;
        let divisor = twice_area.sqrt();
        let coefficients = [
            [(x2 - first_length) / divisor, y2 / divisor],
            [-x2 / divisor, -y2 / divisor],
            [first_length / divisor, 0.0],
        ];
        coefficients
            .into_iter()
            .flatten()
            .all(f64::is_finite)
            .then_some((vertices, coefficients))
            .ok_or(SurfaceError::Unrepresentable)
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

fn difference(left: f64, right: f64) -> Result<f64, SurfaceError> {
    product_sum([(left, 1.0), (right, -1.0)])
}

fn product_sum<const N: usize>(terms: [(f64, f64); N]) -> Result<f64, SurfaceError> {
    adaptive_product_value(terms.into_iter())
        .map(|(value, _)| value)
        .ok_or(SurfaceError::Unrepresentable)
}

fn compensated_add(sum: &mut f64, correction: &mut f64, value: f64) -> Result<(), SurfaceError> {
    let combined = *sum + value;
    if !combined.is_finite() {
        return Err(SurfaceError::Unrepresentable);
    }
    let error = if sum.abs() >= value.abs() {
        (*sum - combined) + value
    } else {
        (value - combined) + *sum
    };
    *correction += error;
    if !correction.is_finite() {
        return Err(SurfaceError::Unrepresentable);
    }
    *sum = combined;
    Ok(())
}

fn add_row(values: &mut [f64], row: usize, contribution: [f64; 3]) -> Result<(), SurfaceError> {
    let start = row.checked_mul(3).ok_or(SurfaceError::Overflow)?;
    let target = values
        .get_mut(start..start + 3)
        .ok_or(SurfaceError::IndexOutside)?;
    for (target, contribution) in target.iter_mut().zip(contribution) {
        *target += contribution;
    }
    Ok(())
}

fn row3(values: &[f64], row: usize) -> Result<[f64; 3], SurfaceError> {
    let start = row.checked_mul(3).ok_or(SurfaceError::Overflow)?;
    values
        .get(start..start + 3)
        .and_then(|values| values.try_into().ok())
        .ok_or(SurfaceError::IndexOutside)
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

fn complex_multiply(left: [f64; 2], right: [f64; 2]) -> [f64; 2] {
    [
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    ]
}

fn normalize_complex(value: [f64; 2]) -> Result<[f64; 2], SurfaceError> {
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

fn complex_conjugate(value: [f64; 2]) -> [f64; 2] {
    [value[0], -value[1]]
}

fn complex_power(mut value: [f64; 2], exponent: i64) -> Result<[f64; 2], SurfaceError> {
    if exponent < 0 {
        value = complex_conjugate(value);
    }
    let mut result = [1.0, 0.0];
    for _ in 0..exponent.unsigned_abs() {
        result = normalize_complex(complex_multiply(result, value))?;
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DualEdges<'a> {
    boundary: &'a CanonicalBoundary,
    edge_count: usize,
}

impl DualEdges<'_> {
    fn is_interior(self, edge: usize) -> Result<bool, SurfaceError> {
        if edge >= self.edge_count {
            return Err(SurfaceError::IndexOutside);
        }
        let count = self.boundary.indptr()[edge + 1] - self.boundary.indptr()[edge];
        match count {
            1 => Ok(false),
            2 => Ok(true),
            _ => Err(SurfaceError::Unrepresentable),
        }
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
        dual.is_interior(edge)?;
    }
    Ok(dual)
}

fn interior_edge_indices(topology: &ComplexCore) -> Result<Vec<usize>, SurfaceError> {
    let dual = dual_edges(topology)?;
    (0..dual.edge_count)
        .filter_map(|edge| match dual.is_interior(edge) {
            Ok(true) => Some(Ok(edge)),
            Ok(false) => None,
            Err(error) => Some(Err(error)),
        })
        .collect()
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
    interior_edges: Arc<[usize]>,
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

    /// Copy the canonical primal-edge index of every compact transport row.
    #[must_use]
    pub fn interior_edge_indices_copy(&self) -> Box<[usize]> {
        self.interior_edges.as_ref().into()
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
            .same_owner(&self.surface.realization.topology().chain_complex())
        {
            return Err(SurfaceError::OwnerMismatch);
        }
        let dual = dual_edges(self.surface.realization.topology())?;
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

fn complex_at(values: &[f64], index: usize) -> Result<[f64; 2], SurfaceError> {
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
    let topology = connection.surface.realization.topology();
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
    let dual = dual_edges(connection.surface.realization.topology())?;
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
            connection: Arc::clone(&self.connection),
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
    connection: Arc<SurfaceConnection>,
    power_directions: Arc<[f64]>,
}

fn interior_direction_charges(
    field: &FaceDirectionField,
    levi_civita_power: &SurfaceConnection,
    boundary_vertices: &[bool],
    entries: &mut Vec<(usize, BigInt)>,
) -> Result<(BigInt, f64, usize), SurfaceError> {
    let surface = &field.connection.surface;
    let topology = surface.realization.topology();
    let curvature = surface.gaussian_curvature_measure()?;
    let incidence = topology.boundary(1)?;
    let dual = dual_edges(topology)?;
    let order = f64::from(field.symmetry_order().get());
    let mut total = BigInt::from(0);
    let mut maximum_residual = 0.0_f64;
    let mut maximum_valence = 0_usize;
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
        let mut terms = Vec::new();
        terms
            .try_reserve_exact(stop - start + 1)
            .map_err(|_| SurfaceError::Overflow)?;
        terms.push((order, curvature.coefficients()[vertex]));
        for (&edge, &incidence_sign) in incidence.indices()[start..stop]
            .iter()
            .zip(&incidence.data()[start..stop])
        {
            let (source, target, source_sign) = dual.edge(edge)?;
            let expected = complex_multiply(
                levi_civita_power.transport(edge)?,
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
            adaptive_product_value(terms.into_iter()).ok_or(SurfaceError::Unrepresentable)?;
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
    let surface = &field.connection.surface;
    let (offsets, edges) = oriented_boundary_components(surface.realization.topology())?;
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
    let mut maximum_edges = 0_usize;
    for component in offsets.windows(2) {
        let start = component[0];
        let stop = component[1];
        if start == stop {
            return Err(SurfaceError::Unrepresentable);
        }
        maximum_edges = maximum_edges.max(stop - start);
        let mut terms = Vec::new();
        terms
            .try_reserve_exact(stop - start)
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
            adaptive_product_value(terms.into_iter()).ok_or(SurfaceError::Unrepresentable)?;
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
    pub const fn connection(&self) -> &Arc<SurfaceConnection> {
        &self.connection
    }

    #[must_use]
    pub fn symmetry_order(&self) -> NonZeroU32 {
        self.connection.symmetry_order()
    }

    #[must_use]
    pub fn power_directions(&self) -> &[f64] {
        &self.power_directions
    }

    /// Calculate exact symmetric-field singularity charges with quantization evidence.
    ///
    /// # Errors
    ///
    /// Rejects unavailable surface geometry, indeterminate binary64 rounding, or
    /// a result that violates the exact relative charge law.
    pub fn singularities(&self) -> Result<DirectionFieldSingularities, SurfaceError> {
        let surface = &self.connection.surface;
        let topology = surface.realization.topology();
        let symmetry_order = self.symmetry_order();
        let order = f64::from(symmetry_order.get());
        let regular = topology.refine_regular()?;
        let boundary_vertices = regular.boundary_mask(0)?;
        let levi_civita_power = surface.connection(
            symmetry_order,
            &vec![0.0; self.connection.interior_edges.len()],
        )?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(topology.vertex_count())
            .map_err(|_| SurfaceError::Overflow)?;
        let (charge_total, charge_residual, maximum_valence) =
            interior_direction_charges(self, &levi_civita_power, &boundary_vertices, &mut entries)?;
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

    /// Borrow the connection-owned crossing residual without duplicating it.
    ///
    /// # Errors
    ///
    /// Returns an internal admission failure if the evidence is unavailable.
    pub fn crossing_error(&self) -> Result<f64, SurfaceError> {
        self.connection
            .admitted_evidence()
            .map(|evidence| evidence.crossing_error)
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
        let first = self.connection.surface.first_frame_axes()?;
        let second = self.connection.surface.second_frame_axes()?;
        let mut values = Vec::with_capacity(3 * self.connection.surface.face_count());
        for face in 0..self.connection.surface.face_count() {
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
        EntityVectors::admit(
            Arc::clone(self.connection.surface.realization()),
            Support::Face,
            values,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CandidateInput, RealizationLimit};

    fn annulus(scale: f64, translation: [f64; 3]) -> Arc<EuclideanRealization> {
        let faces = [
            0_u64, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7,
        ];
        let topology = ComplexCore::admit(
            CandidateInput::unsigned(faces, 8, 3, Some(8)).expect("valid annulus input"),
        )
        .expect("valid annulus topology");
        let points = [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-0.4, -0.4, 0.0],
            [0.4, -0.4, 0.0],
            [0.4, 0.4, 0.0],
            [-0.4, 0.4, 0.0],
        ];
        let positions = points
            .into_iter()
            .flat_map(|point| {
                point
                    .into_iter()
                    .zip(translation)
                    .map(move |(coordinate, offset)| scale * coordinate + offset)
            })
            .collect();
        EuclideanRealization::admit(topology, 3, positions, RealizationLimit::DEFAULT)
            .expect("valid annulus realization")
    }

    #[test]
    fn boundary_direction_targets_follow_induced_cycles_and_similarity() {
        let order = NonZeroU32::new(4).expect("nonzero order");
        let angle = 0.17;
        let surface = TriangleSurface::admit(annulus(1.0, [0.0; 3])).expect("valid surface");
        let targets = surface
            .boundary_direction_targets(order, angle)
            .expect("determinate targets");

        assert_eq!(&*targets.component_offsets, &[0, 4, 8]);
        assert_eq!(targets.component_edges.len(), 8);
        assert_eq!(targets.faces.len(), 8);
        let topology = surface.realization().topology();
        let edges = topology.basis(1).expect("edge basis");
        let boundary = topology.boundary(2).expect("face boundary");
        let first = surface.first_frame_axes().expect("first frame");
        let second = surface.second_frame_axes().expect("second frame");
        let offset = [
            (f64::from(order.get()) * angle).cos(),
            (f64::from(order.get()) * angle).sin(),
        ];
        for component in targets.component_offsets.windows(2) {
            let component_edges = &targets.component_edges[component[0]..component[1]];
            let mut previous_target = None;
            let mut first_source = None;
            for &edge in component_edges {
                let start = boundary.indptr()[edge];
                let face = boundary.indices()[start];
                let coefficient = boundary.data()[start];
                let endpoints = edges.row(edge).expect("edge endpoints");
                let (source, target) = if coefficient > 0 {
                    (endpoints[0], endpoints[1])
                } else {
                    (endpoints[1], endpoints[0])
                };
                assert_eq!(previous_target.unwrap_or(source), source);
                first_source.get_or_insert(source);
                previous_target = Some(target);

                let tangent = normalize(subtract(
                    surface.point(target).expect("target point"),
                    surface.point(source).expect("source point"),
                ))
                .expect("nonzero edge");
                let coordinates = normalize_complex([
                    dot(tangent, row3(first, face).expect("first axis")),
                    dot(tangent, row3(second, face).expect("second axis")),
                ])
                .expect("tangent coordinates");
                let expected = complex_multiply(
                    complex_power(coordinates, i64::from(order.get())).expect("power"),
                    offset,
                );
                let target_row = targets
                    .faces
                    .iter()
                    .position(|&candidate| candidate == face)
                    .expect("boundary face");
                let actual = complex_at(&targets.power_directions, target_row).expect("target");
                assert!((actual[0] - expected[0]).abs() <= 64.0 * f64::EPSILON);
                assert!((actual[1] - expected[1]).abs() <= 64.0 * f64::EPSILON);
            }
            assert_eq!(previous_target, first_source);
        }

        let transformed = TriangleSurface::admit(annulus(7.5, [3.0, -5.0, 2.0]))
            .expect("valid transformed surface")
            .boundary_direction_targets(order, angle)
            .expect("transformed targets");
        assert_eq!(targets.component_offsets, transformed.component_offsets);
        assert_eq!(targets.component_edges, transformed.component_edges);
        assert_eq!(targets.faces, transformed.faces);
        for (&left, &right) in targets
            .power_directions
            .iter()
            .zip(transformed.power_directions.iter())
        {
            assert!((left - right).abs() <= 128.0 * f64::EPSILON);
        }
    }

    #[test]
    fn boundary_direction_targets_aggregate_edges_and_reject_ambiguity() {
        let quad = ComplexCore::admit(
            CandidateInput::unsigned([0_u64, 1, 2, 0, 2, 3], 2, 3, Some(4))
                .expect("valid quad input"),
        )
        .expect("valid quad topology");
        let quad = TriangleSurface::admit(
            EuclideanRealization::admit(
                quad,
                3,
                vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                RealizationLimit::DEFAULT,
            )
            .expect("valid quad realization"),
        )
        .expect("valid quad surface");
        for (order, expected) in [
            (1, [2.0_f64 / 5.0_f64.sqrt(), 1.0 / 5.0_f64.sqrt()]),
            (2, [1.0, 0.0]),
            (4, [1.0, 0.0]),
        ] {
            let targets = quad
                .boundary_direction_targets(NonZeroU32::new(order).expect("nonzero order"), 0.0)
                .expect("determinate multi-edge targets");
            let face = targets
                .faces
                .iter()
                .position(|&face| face == 0)
                .expect("first boundary face");
            let actual = complex_at(&targets.power_directions, face).expect("target");
            assert!((actual[0] - expected[0]).abs() <= 64.0 * f64::EPSILON);
            assert!((actual[1] - expected[1]).abs() <= 64.0 * f64::EPSILON);
        }

        let triangle = TriangleSurface::admit(
            EuclideanRealization::admit(
                ComplexCore::admit(
                    CandidateInput::unsigned([0_u64, 1, 2], 1, 3, Some(3))
                        .expect("valid triangle input"),
                )
                .expect("valid triangle topology"),
                3,
                vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                RealizationLimit::DEFAULT,
            )
            .expect("valid triangle realization"),
        )
        .expect("valid triangle surface");
        assert_eq!(
            triangle
                .boundary_direction_targets(NonZeroU32::MIN, 0.0)
                .expect_err("closed tangent sum is ambiguous"),
            SurfaceError::Unrepresentable
        );
    }
}
