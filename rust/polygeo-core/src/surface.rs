use std::{cmp::Ordering as CmpOrdering, fmt, mem::size_of, sync::Arc};

use faer::{
    Mat, MatMut, Spec,
    dyn_stack::{MemBuffer, MemStack, StackReq},
    linalg::qr::col_pivoting::{factor as qr_factor, solve as qr_solve},
    perm::PermRef,
};
use once_cell::sync::OnceCell;

use crate::numeric::{adaptive_product_sign, adaptive_product_value};
use crate::solve_impl::{
    SystemRef, check_cancelled, checked_work_product, cubic_work, dirichlet_energy, factor_scale,
    factor_solve_requirement, factor_stiffness, fill_centered_mass_rhs, flow_residual,
    logical_bytes, logical_f64, matrix_bytes, require_storage, require_work, solve_factor,
    stiffness_endpoints, weighted_centroid,
};
use crate::{
    Binary64Chain, Binary64ChainSpace, Binary64Cochain, Binary64CochainSpace, Binary64Element,
    CancellationToken, Executor, Geometry, GeometryError, Limit, Metric, NondegenerateCapability,
    PairingCapability, Policy, SolveError, StorageLimit, SurfaceComputationError, TopologyError,
    WorkLimit,
};

pub(crate) fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

pub(crate) fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

pub(crate) fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

pub(crate) fn norm(value: [f64; 3]) -> f64 {
    let scale = value.into_iter().map(f64::abs).fold(0.0_f64, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return scale;
    }
    let scaled = value.map(|entry| entry / scale);
    scale * dot(scaled, scaled).sqrt()
}

pub(crate) fn normalize(value: [f64; 3]) -> Option<[f64; 3]> {
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

/// One contiguous ambient-vector field over a realized simplex degree.
#[derive(Clone, Debug)]
pub struct EntityVectors<const DEGREE: usize> {
    realization: Arc<Geometry>,
    values: Arc<[f64]>,
}

pub type VertexVectors = EntityVectors<0>;
pub type FaceVectors = EntityVectors<2>;

impl<const DEGREE: usize> EntityVectors<DEGREE> {
    fn admit(realization: Arc<Geometry>, values: Vec<f64>) -> Result<Self, SurfaceError> {
        let entities = realization
            .topology()
            .basis(DEGREE)
            .map_err(SurfaceError::Topology)?
            .row_count();
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
            values: values.into(),
        })
    }

    #[must_use]
    pub const fn realization(&self) -> &Arc<Geometry> {
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
    pub const fn support_degree(&self) -> usize {
        DEGREE
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
        Self::admit(Arc::clone(&self.realization), values)
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
    target: Arc<Geometry>,
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
    realization: Arc<Geometry>,
    evidence: LeastSquaresConformalMapEvidence,
}

impl LeastSquaresConformalMapSolution {
    #[must_use]
    pub const fn realization(&self) -> &Arc<Geometry> {
        &self.realization
    }
    #[must_use]
    pub const fn evidence(&self) -> LeastSquaresConformalMapEvidence {
        self.evidence
    }

    pub(crate) const fn new(
        realization: Arc<Geometry>,
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
    pub const fn target(&self) -> &Arc<Geometry> {
        &self.target
    }
    #[must_use]
    pub const fn evidence(&self) -> FlowEvidence {
        self.evidence
    }

    pub(crate) const fn new(target: Arc<Geometry>, evidence: FlowEvidence) -> Self {
        Self { target, evidence }
    }
}

#[derive(Debug)]
pub(crate) struct SurfaceRows {
    pub(crate) first: Box<[f64]>,
    pub(crate) second: Box<[f64]>,
}

/// One admitted oriented triangle-manifold realization in ambient dimension three.
pub struct TriangleSurface {
    realization: Arc<Geometry>,
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
    pub fn admit(realization: Arc<Geometry>) -> Result<Arc<Self>, SurfaceError> {
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
    pub const fn realization(&self) -> &Arc<Geometry> {
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
        EntityVectors::admit(Arc::clone(&self.realization), values)
    }

    /// Admit one owned contiguous face-vector field on this realization.
    ///
    /// # Errors
    ///
    /// Rejects a shape mismatch or nonfinite coefficient.
    pub fn face_vectors(&self, values: Vec<f64>) -> Result<FaceVectors, SurfaceError> {
        EntityVectors::admit(Arc::clone(&self.realization), values)
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
        EntityVectors::admit(Arc::clone(&self.realization), values)
    }

    /// Compute the weak divergence load of one face-supported ambient vector field.
    ///
    /// The returned degree-zero chain is the negative adjoint of [`Self::gradient`].
    /// On a surface with boundary this selects the natural no-flux boundary law.
    ///
    /// # Errors
    ///
    /// Rejects a foreign field or unrepresentable arithmetic.
    pub fn divergence(&self, field: &FaceVectors) -> Result<Binary64Chain, SurfaceError> {
        if !Arc::ptr_eq(field.realization(), &self.realization) {
            return Err(SurfaceError::OwnerMismatch);
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

    pub(crate) fn surface_rows(&self) -> Result<&SurfaceRows, SurfaceError> {
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
        EntityVectors::admit(Arc::clone(&self.realization), values)
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
        EntityVectors::admit(Arc::clone(&self.realization), values)?.normalized()
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
        EntityVectors::admit(Arc::clone(&self.realization), values)?.normalized()
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
        EntityVectors::admit(Arc::clone(&self.realization), values)?.normalized()
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
        EntityVectors::admit(Arc::clone(&self.realization), values)
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
        EntityVectors::admit(Arc::clone(&self.realization), values)
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
    pub fn mean_curvature_vectors(&self, metric: &Metric) -> Result<VertexVectors, SurfaceError> {
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
        EntityVectors::admit(Arc::clone(&self.realization), values)
    }

    pub(crate) fn require_closed(&self) -> Result<(), SurfaceError> {
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

    pub(crate) fn point(&self, vertex: usize) -> Result<[f64; 3], SurfaceError> {
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

fn difference(left: f64, right: f64) -> Result<f64, SurfaceError> {
    product_sum([(left, 1.0), (right, -1.0)])
}

pub(crate) fn product_sum<const N: usize>(terms: [(f64, f64); N]) -> Result<f64, SurfaceError> {
    adaptive_product_value(terms.into_iter())
        .map(|(value, _)| value)
        .ok_or(SurfaceError::Unrepresentable)
}

pub(crate) fn compensated_add(
    sum: &mut f64,
    correction: &mut f64,
    value: f64,
) -> Result<(), SurfaceError> {
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

pub(crate) fn row3(values: &[f64], row: usize) -> Result<[f64; 3], SurfaceError> {
    let start = row.checked_mul(3).ok_or(SurfaceError::Overflow)?;
    values
        .get(start..start + 3)
        .and_then(|values| values.try_into().ok())
        .ok_or(SurfaceError::IndexOutside)
}

impl TriangleSurface {
    /// Compute one bounded least-squares conformal parameterization of an oriented disk.
    ///
    /// The two distinct boundary anchors map to `(0, 0)` and `(1, 0)` in caller order.
    ///
    /// # Errors
    /// Rejects a non-disk domain, invalid anchors, exhausted resources, cancellation,
    /// numerical rank loss, or a target face without positive admitted orientation.
    pub fn least_squares_conformal_map(
        &self,
        anchors: [usize; 2],
        realization_limit: Limit,
        policy: Policy,
        cancellation: &CancellationToken,
    ) -> Result<LeastSquaresConformalMapSolution, SurfaceComputationError> {
        let disk = self
            .realization()
            .topology()
            .refine_disk()
            .map_err(SurfaceError::from)?;
        let boundary = disk.boundary_vertices().map_err(SurfaceError::from)?;
        let vertex_count = self.realization().topology().vertex_count();
        if anchors[0] == anchors[1] {
            return Err(SurfaceError::CoincidentAnchor.into());
        }
        for &anchor in &anchors {
            if anchor >= vertex_count {
                return Err(SurfaceError::IndexOutside.into());
            }
            if !boundary.contains(&anchor) {
                return Err(SurfaceError::AnchorNotBoundary.into());
            }
        }
        least_squares_conformal_map(
            self,
            anchors,
            realization_limit,
            policy.executor(),
            policy.storage(),
            policy.work(),
            cancellation,
        )
        .map_err(SurfaceComputationError::Solve)
    }
}
impl crate::Metric {
    /// Compute and atomically publish one frozen-metric mean-curvature-flow step.
    ///
    /// # Errors
    /// Rejects an unsuitable surface, invalid time, exhausted resources,
    /// cancellation, failed factorization, or failed numerical certification.
    pub fn frozen_mean_curvature_flow(
        &self,
        time_step: f64,
        realization_limit: Limit,
        policy: Policy,
        cancellation: &CancellationToken,
    ) -> Result<FlowStep, SurfaceComputationError> {
        require_frozen_flow_domain(self, time_step)?;
        frozen_flow_step(
            self,
            time_step,
            realization_limit,
            policy.executor(),
            policy.storage(),
            policy.work(),
            cancellation,
        )
        .map_err(SurfaceComputationError::Solve)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ScaledNorm {
    scale: f64,
    sum_squares: f64,
}

impl ScaledNorm {
    fn add(&mut self, value: f64) -> Result<(), SolveError> {
        let value = value.abs();
        if !value.is_finite() {
            return Err(SolveError::Numerical);
        }
        if value == 0.0 {
            return Ok(());
        }
        if self.scale < value {
            let ratio = self.scale / value;
            self.sum_squares = 1.0 + self.sum_squares * ratio * ratio;
            self.scale = value;
        } else {
            let ratio = value / self.scale;
            self.sum_squares += ratio * ratio;
        }
        self.sum_squares
            .is_finite()
            .then_some(())
            .ok_or(SolveError::Numerical)
    }

    fn value(self) -> f64 {
        self.scale * self.sum_squares.sqrt()
    }
}

fn least_squares_conformal_map(
    surface: &TriangleSurface,
    anchors: [usize; 2],
    realization_limit: Limit,
    executor: Executor,
    storage: StorageLimit,
    work: WorkLimit,
    cancellation: &CancellationToken,
) -> Result<LeastSquaresConformalMapSolution, SolveError> {
    check_cancelled(cancellation)?;
    let vertex_count = surface.realization().topology().vertex_count();
    let face_count = surface.face_count();
    let (rows, columns, block, scratch) = require_least_squares_conformal_resources(
        vertex_count,
        face_count,
        executor,
        storage,
        work,
    )?;
    let mut system =
        assemble_least_squares_conformal_system(surface, anchors, rows, columns, cancellation)?;
    let (observed_rank, condition_indicator) =
        solve_least_squares_conformal_system(&mut system, block, scratch, executor, cancellation)?;
    let positions = least_squares_conformal_positions(vertex_count, anchors, &system)?;
    let residual_bound = least_squares_conformal_residual(surface, &positions, &system)?;
    let (minimum_area, exact_fallback_faces) =
        least_squares_conformal_orientation(surface, &positions)?;
    check_cancelled(cancellation)?;
    let target = Geometry::admit(
        Arc::clone(surface.realization().topology()),
        2,
        positions,
        realization_limit,
    )
    .map_err(realization_solve_error)?;
    check_cancelled(cancellation)?;
    Ok(LeastSquaresConformalMapSolution::new(
        target,
        LeastSquaresConformalMapEvidence::new(
            columns,
            observed_rank,
            condition_indicator,
            residual_bound,
            minimum_area,
            exact_fallback_faces,
        ),
    ))
}

struct LeastSquaresConformalSystem {
    matrix: Mat<f64>,
    right_hand_side: Vec<f64>,
    free_position: Vec<usize>,
    matrix_norm: f64,
    right_hand_side_norm: f64,
}

fn require_least_squares_conformal_resources(
    vertex_count: usize,
    face_count: usize,
    executor: Executor,
    storage: StorageLimit,
    work: WorkLimit,
) -> Result<(usize, usize, usize, StackReq), SolveError> {
    let rows = face_count.checked_mul(2).ok_or(SolveError::ResourceLimit)?;
    let columns = vertex_count
        .checked_sub(2)
        .and_then(|value| value.checked_mul(2))
        .ok_or(SolveError::ResourceLimit)?;
    if rows < columns || columns == 0 {
        return Err(SolveError::Factorization);
    }
    let block = qr_factor::recommended_block_size::<f64>(rows, columns).max(1);
    let factor = qr_factor::qr_in_place_scratch::<usize, f64>(
        rows,
        columns,
        block,
        executor.par(),
        Spec::default(),
    );
    let solve = qr_solve::solve_lstsq_in_place_scratch::<usize, f64>(
        rows,
        columns,
        block,
        1,
        executor.par(),
    );
    let scratch = StackReq::any_of(&[factor, solve]);
    let matrix_cells = rows.checked_mul(columns).ok_or(SolveError::ResourceLimit)?;
    let f64_cells = matrix_cells
        .checked_add(
            block
                .checked_mul(columns)
                .ok_or(SolveError::ResourceLimit)?,
        )
        .and_then(|value| value.checked_add(rows))
        .and_then(|value| value.checked_add(vertex_count.checked_mul(2)?))
        .ok_or(SolveError::ResourceLimit)?;
    let usize_cells = columns
        .checked_mul(2)
        .and_then(|value| value.checked_add(vertex_count))
        .ok_or(SolveError::ResourceLimit)?;
    let peak = logical_f64(f64_cells)?
        .checked_add(logical_bytes(usize_cells, size_of::<usize>())?)
        .and_then(|value| value.checked_add(u64::try_from(scratch.size_bytes()).ok()?))
        .ok_or(SolveError::ResourceLimit)?;
    require_storage(storage, 0, peak)?;
    let factor_work = checked_work_product(matrix_cells, columns)?;
    let remaining = checked_work_product(matrix_cells, 8)?
        .checked_add(checked_work_product(face_count, 64)?)
        .ok_or(SolveError::ResourceLimit)?;
    require_work(
        work,
        factor_work
            .checked_add(remaining)
            .ok_or(SolveError::ResourceLimit)?,
    )?;
    Ok((rows, columns, block, scratch))
}

fn assemble_least_squares_conformal_system(
    surface: &TriangleSurface,
    anchors: [usize; 2],
    rows: usize,
    columns: usize,
    cancellation: &CancellationToken,
) -> Result<LeastSquaresConformalSystem, SolveError> {
    let vertex_count = surface.realization().topology().vertex_count();
    let mut free_position = vec![usize::MAX; vertex_count];
    for (position, vertex) in (0..vertex_count)
        .filter(|vertex| !anchors.contains(vertex))
        .enumerate()
    {
        free_position[vertex] = position;
    }
    let mut matrix = Mat::zeros(rows, columns);
    let mut right_hand_side = vec![0.0; rows];
    let mut matrix_norm = ScaledNorm::default();
    for face in 0..surface.face_count() {
        check_cancelled(cancellation)?;
        let (vertices, coefficients) = surface
            .oriented_local_conformal_coefficients(face)
            .map_err(|_| SolveError::Numerical)?;
        for (vertex, coefficient) in vertices.into_iter().zip(coefficients) {
            add_conformal_corner(
                &mut matrix,
                &mut right_hand_side,
                &mut matrix_norm,
                &free_position,
                anchors[1],
                face,
                vertex,
                coefficient,
            )?;
        }
    }
    let right_hand_side_norm = scaled_norm(&right_hand_side)?;
    let matrix_norm = matrix_norm.value();
    if matrix_norm == 0.0 || right_hand_side_norm == 0.0 {
        return Err(SolveError::Factorization);
    }
    Ok(LeastSquaresConformalSystem {
        matrix,
        right_hand_side,
        free_position,
        matrix_norm,
        right_hand_side_norm,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "one face-corner row contribution"
)]
fn add_conformal_corner(
    matrix: &mut Mat<f64>,
    right_hand_side: &mut [f64],
    matrix_norm: &mut ScaledNorm,
    free_position: &[usize],
    nonzero_anchor: usize,
    face: usize,
    vertex: usize,
    [real, imaginary]: [f64; 2],
) -> Result<(), SolveError> {
    let real_row = 2 * face;
    let imaginary_row = real_row + 1;
    let position = free_position[vertex];
    if position == usize::MAX {
        if vertex == nonzero_anchor {
            right_hand_side[real_row] -= real;
            right_hand_side[imaginary_row] -= imaginary;
        }
        return Ok(());
    }
    let real_column = 2 * position;
    let imaginary_column = real_column + 1;
    let values = [real, -imaginary, imaginary, real];
    matrix[(real_row, real_column)] = values[0];
    matrix[(real_row, imaginary_column)] = values[1];
    matrix[(imaginary_row, real_column)] = values[2];
    matrix[(imaginary_row, imaginary_column)] = values[3];
    values
        .into_iter()
        .try_for_each(|value| matrix_norm.add(value))
}

fn solve_least_squares_conformal_system(
    system: &mut LeastSquaresConformalSystem,
    block: usize,
    scratch: StackReq,
    executor: Executor,
    cancellation: &CancellationToken,
) -> Result<(usize, f64), SolveError> {
    let rows = system.matrix.nrows();
    let columns = system.matrix.ncols();
    let mut householder = Mat::zeros(block, columns);
    let mut permutation = vec![0_usize; columns];
    let mut inverse_permutation = vec![0_usize; columns];
    let mut memory = MemBuffer::try_new(scratch).map_err(|_| SolveError::Allocation)?;
    check_cancelled(cancellation)?;
    qr_factor::qr_in_place(
        system.matrix.as_mut(),
        householder.as_mut(),
        &mut permutation,
        &mut inverse_permutation,
        executor.par(),
        MemStack::new(&mut memory),
        Spec::default(),
    );
    check_cancelled(cancellation)?;
    let diagonal = (0..columns).map(|index| system.matrix[(index, index)].abs());
    let maximum = diagonal.clone().fold(0.0_f64, f64::max);
    let threshold = f64::EPSILON.sqrt() * maximum;
    let rank = diagonal.clone().filter(|value| *value > threshold).count();
    let minimum = diagonal.fold(f64::INFINITY, f64::min);
    let condition = maximum / minimum;
    if rank != columns || !condition.is_finite() {
        return Err(SolveError::Factorization);
    }
    let mut rhs = MatMut::from_column_major_slice_mut(&mut system.right_hand_side, rows, 1);
    let stack = MemStack::new(&mut memory);
    qr_solve::solve_lstsq_in_place(
        system.matrix.as_ref(),
        householder.as_ref(),
        system.matrix.as_ref(),
        PermRef::new_checked(&permutation, &inverse_permutation, columns),
        rhs.as_mut(),
        executor.par(),
        stack,
    );
    check_cancelled(cancellation)?;
    system.right_hand_side[..columns]
        .iter()
        .all(|value| value.is_finite())
        .then_some((rank, condition))
        .ok_or(SolveError::Numerical)
}

fn least_squares_conformal_positions(
    vertex_count: usize,
    anchors: [usize; 2],
    system: &LeastSquaresConformalSystem,
) -> Result<Vec<f64>, SolveError> {
    let mut positions = vec![0.0; 2 * vertex_count];
    positions[2 * anchors[1]] = 1.0;
    for (vertex, &position) in system.free_position.iter().enumerate() {
        if position != usize::MAX {
            positions[2 * vertex] = system.right_hand_side[2 * position];
            positions[2 * vertex + 1] = system.right_hand_side[2 * position + 1];
        }
    }
    positions
        .iter()
        .all(|value| value.is_finite())
        .then_some(positions)
        .ok_or(SolveError::Numerical)
}

fn least_squares_conformal_residual(
    surface: &TriangleSurface,
    positions: &[f64],
    system: &LeastSquaresConformalSystem,
) -> Result<f64, SolveError> {
    let columns = system.matrix.ncols();
    let solution_norm = scaled_norm(&system.right_hand_side[..columns])?;
    let mut residual_norm = ScaledNorm::default();
    for face in 0..surface.face_count() {
        let (vertices, coefficients) = surface
            .oriented_local_conformal_coefficients(face)
            .map_err(|_| SolveError::Numerical)?;
        for imaginary_part in [false, true] {
            let value = conformal_row_value(vertices, coefficients, positions, imaginary_part)?;
            residual_norm.add(value)?;
        }
    }
    let denominator = system.matrix_norm * solution_norm + system.right_hand_side_norm;
    let residual = residual_norm.value() / denominator;
    residual
        .is_finite()
        .then_some(residual)
        .ok_or(SolveError::Numerical)
}

fn conformal_row_value(
    vertices: [usize; 3],
    coefficients: [[f64; 2]; 3],
    positions: &[f64],
    imaginary_part: bool,
) -> Result<f64, SolveError> {
    adaptive_product_value(vertices.into_iter().zip(coefficients).flat_map(
        |(vertex, [real, imaginary])| {
            if imaginary_part {
                [
                    (imaginary, positions[2 * vertex]),
                    (real, positions[2 * vertex + 1]),
                ]
            } else {
                [
                    (real, positions[2 * vertex]),
                    (-imaginary, positions[2 * vertex + 1]),
                ]
            }
        },
    ))
    .map(|(value, _)| value)
    .ok_or(SolveError::Numerical)
}

fn least_squares_conformal_orientation(
    surface: &TriangleSurface,
    positions: &[f64],
) -> Result<(f64, usize), SolveError> {
    let mut minimum = f64::INFINITY;
    let mut exact_fallback_faces = 0_usize;
    for face in 0..surface.face_count() {
        let (vertices, _) = surface
            .oriented_local_conformal_coefficients(face)
            .map_err(|_| SolveError::Numerical)?;
        let terms = normalized_signed_area_terms(vertices, positions)?;
        let (sign, exact_fallback) =
            adaptive_product_sign(terms.into_iter()).ok_or(SolveError::Numerical)?;
        if sign != CmpOrdering::Greater {
            return Err(SolveError::Numerical);
        }
        exact_fallback_faces += usize::from(exact_fallback);
        let area = adaptive_product_value(terms.into_iter())
            .ok_or(SolveError::Numerical)?
            .0;
        if area <= 0.0 || !area.is_finite() {
            return Err(SolveError::Numerical);
        }
        minimum = minimum.min(area);
    }
    Ok((minimum, exact_fallback_faces))
}

fn normalized_signed_area_terms(
    vertices: [usize; 3],
    positions: &[f64],
) -> Result<[(f64, f64); 2], SolveError> {
    let point = |vertex| [positions[2 * vertex], positions[2 * vertex + 1]];
    let [p0, p1, p2] = vertices.map(point);
    let first = [p1[0] - p0[0], p1[1] - p0[1]];
    let second = [p2[0] - p0[0], p2[1] - p0[1]];
    let scale = first
        .into_iter()
        .chain(second)
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return Err(SolveError::Numerical);
    }
    Ok([
        (first[0] / scale, second[1] / scale),
        (-first[1] / scale, second[0] / scale),
    ])
}

fn scaled_norm(values: &[f64]) -> Result<f64, SolveError> {
    let mut norm = ScaledNorm::default();
    values.iter().try_for_each(|&value| norm.add(value))?;
    Ok(norm.value())
}

fn realization_solve_error(error: GeometryError) -> SolveError {
    if error.resource_limit().is_some() {
        SolveError::ResourceLimit
    } else if error == GeometryError::Allocation {
        SolveError::Allocation
    } else {
        SolveError::Numerical
    }
}

fn require_frozen_flow_domain(metric: &crate::Metric, time_step: f64) -> Result<(), SurfaceError> {
    if !time_step.is_finite() || time_step <= 0.0 {
        return Err(SurfaceError::TimeStep);
    }
    let realization = metric.realization();
    if realization.ambient_dimension() != 3 {
        return Err(SurfaceError::AmbientDimension);
    }
    realization.topology().refine_triangle()?;
    realization.topology().refine_oriented()?;
    realization
        .topology()
        .refine_regular()?
        .without_boundary()
        .map_err(|_| SurfaceError::BoundaryPresent)?;
    realization.topology().refine_connected()?;
    Ok(())
}

fn frozen_flow_step(
    metric: &crate::Metric,
    time_step: f64,
    realization_limit: Limit,
    executor: Executor,
    storage: StorageLimit,
    work: WorkLimit,
    cancellation: &CancellationToken,
) -> Result<FlowStep, SolveError> {
    check_cancelled(cancellation)?;
    let source = metric.realization();
    let n = source.topology().vertex_count();
    let dimension = source.ambient_dimension();
    let cells = n.checked_mul(dimension).ok_or(SolveError::ResourceLimit)?;
    let quadratic = n
        .checked_mul(n)
        .and_then(|value| value.checked_mul(dimension))
        .ok_or(SolveError::ResourceLimit)?;
    let solve_work = u64::try_from(quadratic.max(cells.saturating_mul(8)))
        .map_err(|_| SolveError::ResourceLimit)?;
    let total_work = cubic_work(n)?
        .checked_add(solve_work)
        .ok_or(SolveError::ResourceLimit)?;
    require_work(work, total_work)?;

    let factor_bytes = matrix_bytes(n)?;
    require_storage(storage, 0, factor_bytes.saturating_mul(2))?;
    let free = (0..n).collect::<Vec<_>>();
    let factor = factor_stiffness(
        SystemRef::Parabolic { metric, time_step },
        &free,
        executor,
        cancellation,
    )?;
    check_cancelled(cancellation)?;

    let requirement =
        StackReq::new::<f64>(cells).and(factor_solve_requirement(&factor, executor, dimension));
    let workspace_bytes =
        u64::try_from(requirement.size_bytes()).map_err(|_| SolveError::ResourceLimit)?;
    let peak = factor_bytes
        .saturating_mul(2)
        .checked_add(workspace_bytes)
        .ok_or(SolveError::ResourceLimit)?;
    require_storage(storage, 0, peak)?;
    let mut buffer = MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?;
    let stack = MemStack::new(&mut buffer);
    if !stack.can_hold(requirement) {
        return Err(SolveError::ResourceLimit);
    }

    let masses = metric
        .hodge_coefficients_slice(0)
        .map_err(|_| SolveError::Numerical)?;
    let centroid = weighted_centroid(masses, source.positions(), dimension)?;
    let scale = factor_scale(&factor);
    let (mut rhs_storage, stack) = stack.make_with(cells, |_| 0.0_f64);
    {
        let mut rhs = MatMut::from_column_major_slice_mut(&mut rhs_storage, n, dimension);
        fill_centered_mass_rhs(masses, source.positions(), &centroid, scale, rhs.as_mut())?;
        solve_factor(&factor, rhs.as_mut(), executor, stack);
    }
    check_cancelled(cancellation)?;

    let (_, endpoints) = stiffness_endpoints(metric)?;
    let residual_bound = flow_residual(metric, time_step, &rhs_storage, &centroid, &endpoints)?;
    let mut positions = vec![0.0; cells];
    for vertex in 0..n {
        for axis in 0..dimension {
            positions[vertex * dimension + axis] = rhs_storage[axis * n + vertex] + centroid[axis];
        }
    }
    drop(rhs_storage);
    let target_centroid = weighted_centroid(masses, &positions, dimension)?;
    let centroid_scale = positions
        .iter()
        .chain(&centroid)
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    let centroid_residual_bound = target_centroid
        .iter()
        .zip(&centroid)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
        / centroid_scale;
    let energy_before = dirichlet_energy(metric, source.positions(), dimension, &endpoints)?;
    let energy_after = dirichlet_energy(metric, &positions, dimension, &endpoints)?;
    if !residual_bound.is_finite()
        || !centroid_residual_bound.is_finite()
        || !energy_before.is_finite()
        || !energy_after.is_finite()
        || residual_bound > 1.0e-10
        || centroid_residual_bound > 1.0e-12
        || energy_after > energy_before + 128.0 * f64::EPSILON * energy_before.max(1.0)
    {
        return Err(SolveError::Numerical);
    }
    check_cancelled(cancellation)?;
    let target = source
        .deform(positions, realization_limit)
        .map_err(|_| SolveError::Numerical)?;
    check_cancelled(cancellation)?;
    Ok(FlowStep::new(
        target,
        FlowEvidence::new(
            energy_before,
            energy_after,
            residual_bound,
            centroid_residual_bound,
        ),
    ))
}
