use std::{num::NonZeroU32, sync::Arc};

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use polygeo_core::field::{
    Connection as SurfaceConnection, Direction as FaceDirectionField,
    DualCycles as IntegralDualCycleBasis, Holonomy as HolonomyEvidence, IntegrableConnection,
    Singularities as DirectionFieldSingularities,
};
use polygeo_core::geometry::{
    ConformalMap as LeastSquaresConformalMapSolution, FaceField as FaceVectors,
    Geometry as EuclideanRealization, SurfaceError, TriangleSurface, VertexField as VertexVectors,
};
use polygeo_core::topology::TopologyError;
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule, PyTuple};

use crate::array::{filled_array_1d, filled_array_2d};
use crate::chain::{ExactElement, NativeChainElement, bigint_tuple};
use crate::classified_exception;
use crate::form::{Element, PyBinary64Element};
use crate::realization::{NativeEuclideanRealization, PyPositiveMetric, PyRealizationLimit};

create_exception!(_polygeo_native, SurfaceErrorPy, PyValueError);

pub(crate) fn surface_error(error: SurfaceError) -> PyErr {
    Python::attach(|py| {
        classified_exception(
            py,
            SurfaceErrorPy::new_err(error.to_string()),
            match error {
                SurfaceError::Topology(error) => error.reason(),
                SurfaceError::AmbientDimension => "ambient_dimension",
                SurfaceError::FieldShape => "field_shape",
                SurfaceError::NonFinite => "non_finite",
                SurfaceError::ZeroVector => "zero_vector",
                SurfaceError::BoundaryPresent => "boundary_present",
                SurfaceError::OwnerMismatch => "owner_mismatch",
                SurfaceError::NotIntegrable => "not_integrable",
                SurfaceError::IndexOutside => "index_outside",
                SurfaceError::Overflow => "count_overflow",
                SurfaceError::Unrepresentable => "unrepresentable",
                SurfaceError::TimeStep => "time_step",
                SurfaceError::CoincidentAnchor => "coincident_anchor",
                SurfaceError::AnchorNotBoundary => "anchor_not_boundary",
                _ => "surface",
            },
            PyDict::new(py).unbind(),
        )
    })
}

#[pyclass(
    name = "TriangleSurface",
    frozen,
    module = "polygeo.geometry",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyTriangleSurface {
    inner: Arc<TriangleSurface>,
}

#[pymethods]
impl PyTriangleSurface {
    #[staticmethod]
    fn admit(realization: &NativeEuclideanRealization) -> PyResult<Self> {
        Ok(Self {
            inner: TriangleSurface::admit(Arc::clone(realization.owner()))
                .map_err(surface_error)?,
        })
    }
    #[getter]
    fn geometry(&self) -> NativeEuclideanRealization {
        NativeEuclideanRealization::from_owner(Arc::clone(self.inner.realization()))
    }
    #[getter]
    fn face_count(&self) -> usize {
        self.inner.face_count()
    }
    #[getter]
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    fn vertex_field(&self, values: &Bound<'_, PyAny>) -> PyResult<PyEntityVectors> {
        let values = values.extract::<PyReadonlyArray2<'_, f64>>()?;
        vertex_field(
            self.inner
                .vertex_vectors(values.as_array().iter().copied().collect()),
        )
    }
    fn face_field(&self, values: &Bound<'_, PyAny>) -> PyResult<PyEntityVectors> {
        let values = values.extract::<PyReadonlyArray2<'_, f64>>()?;
        face_field(
            self.inner
                .face_vectors(values.as_array().iter().copied().collect()),
        )
    }
    fn gradient(&self, source: &PyBinary64Element) -> PyResult<PyEntityVectors> {
        let Element::Cochain(source) = &source.inner else {
            return Err(surface_error(SurfaceError::FieldShape));
        };
        face_field(self.inner.gradient(source))
    }
    fn divergence(&self, field: &PyEntityVectors) -> PyResult<PyBinary64Element> {
        Ok(PyBinary64Element {
            inner: Element::Chain(
                self.inner
                    .divergence(field.face().map_err(surface_error)?)
                    .map_err(surface_error)?,
            ),
        })
    }
    fn first_frame_axes_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        project_rows(
            py,
            self.inner.first_frame_axes().map_err(surface_error)?,
            self.inner.face_count(),
            3,
        )
    }
    fn second_frame_axes_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        project_rows(
            py,
            self.inner.second_frame_axes().map_err(surface_error)?,
            self.inner.face_count(),
            3,
        )
    }
    fn face_unit_normals(&self) -> PyResult<PyEntityVectors> {
        face_field(self.inner.face_unit_normals())
    }
    fn uniform_vertex_normals(&self) -> PyResult<PyEntityVectors> {
        vertex_field(self.inner.uniform_vertex_normals())
    }
    fn tip_angle_vertex_normals(&self) -> PyResult<PyEntityVectors> {
        vertex_field(self.inner.tip_angle_vertex_normals())
    }
    fn sphere_inscribed_vertex_normals(&self) -> PyResult<PyEntityVectors> {
        vertex_field(self.inner.sphere_inscribed_vertex_normals())
    }
    fn surface_area_gradient(&self) -> PyResult<PyEntityVectors> {
        vertex_field(self.inner.surface_area_gradient())
    }
    fn volume_gradient(&self) -> PyResult<PyEntityVectors> {
        vertex_field(self.inner.volume_gradient())
    }
    fn gaussian_curvature_measure(&self) -> PyResult<PyBinary64Element> {
        Ok(PyBinary64Element {
            inner: Element::Cochain(
                self.inner
                    .gaussian_curvature_measure()
                    .map_err(surface_error)?,
            ),
        })
    }
    fn mean_curvature_vectors(&self, metric: &PyPositiveMetric) -> PyResult<PyEntityVectors> {
        vertex_field(self.inner.mean_curvature_vectors(&metric.inner))
    }
    fn levi_civita(&self) -> PyResult<PySurfaceConnection> {
        connection(self.inner.levi_civita_connection())
    }
    fn connection(
        &self,
        symmetry_order: u32,
        deviations: &Bound<'_, PyAny>,
    ) -> PyResult<PySurfaceConnection> {
        let symmetry_order = NonZeroU32::new(symmetry_order)
            .ok_or_else(|| PyValueError::new_err("symmetry_order must be positive"))?;
        let deviations = deviations.extract::<PyReadonlyArray1<'_, f64>>()?;
        let deviations = deviations.as_array().iter().copied().collect::<Vec<_>>();
        connection(self.inner.connection(symmetry_order, &deviations))
    }

    #[pyo3(signature = (symmetry_order, metric, harmonic_basis, dual_cycles, charges, generator_turns, anchor_angle, *, policy=None, cancellation=None))]
    #[expect(
        clippy::too_many_arguments,
        reason = "the binding preserves the explicit native call boundary"
    )]
    fn direction_field(
        &self,
        py: Python<'_>,
        symmetry_order: u32,
        metric: &PyPositiveMetric,
        harmonic_basis: &crate::solve::PyHarmonicOneFormBasis,
        dual_cycles: &PyIntegralDualCycleBasis,
        charges: &NativeChainElement,
        generator_turns: Vec<i64>,
        anchor_angle: f64,
        policy: Option<&crate::solve::PyPolicy>,
        cancellation: Option<&crate::solve::PyCancellationToken>,
    ) -> PyResult<PyFaceDirectionField> {
        let symmetry_order = NonZeroU32::new(symmetry_order)
            .ok_or_else(|| PyValueError::new_err("symmetry_order must be positive"))?;
        let ExactElement::IntegerCochain(charges) = &charges.inner else {
            return Err(PyValueError::new_err("charges must be an integral cochain"));
        };
        let policy = crate::solve::policy(policy);
        let cancellation = crate::solve::cancellation_token(cancellation);
        let surface = Arc::clone(&self.inner);
        let metric = metric.inner.clone();
        let harmonic_basis = harmonic_basis.inner.clone();
        let dual_cycles = dual_cycles.inner.clone();
        let charges = charges.clone();
        py.detach(move || {
            surface.minimum_energy_direction_field(
                symmetry_order,
                &metric,
                &harmonic_basis,
                &dual_cycles,
                &charges,
                &generator_turns,
                anchor_angle,
                policy,
                &cancellation,
            )
        })
        .map(|inner| PyFaceDirectionField { inner })
        .map_err(crate::solve::surface_computation_error)
    }

    #[pyo3(signature = (symmetry_order, metric, boundary_angle_offset, *, policy=None, cancellation=None))]
    fn boundary_direction(
        &self,
        py: Python<'_>,
        symmetry_order: u32,
        metric: &PyPositiveMetric,
        boundary_angle_offset: f64,
        policy: Option<&crate::solve::PyPolicy>,
        cancellation: Option<&crate::solve::PyCancellationToken>,
    ) -> PyResult<PyFaceDirectionField> {
        let symmetry_order = NonZeroU32::new(symmetry_order)
            .ok_or_else(|| PyValueError::new_err("symmetry_order must be positive"))?;
        let policy = crate::solve::policy(policy);
        let cancellation = crate::solve::cancellation_token(cancellation);
        let surface = Arc::clone(&self.inner);
        let metric = metric.inner.clone();
        py.detach(move || {
            surface.boundary_aligned_direction_field(
                symmetry_order,
                &metric,
                boundary_angle_offset,
                policy,
                &cancellation,
            )
        })
        .map(|inner| PyFaceDirectionField { inner })
        .map_err(crate::solve::surface_computation_error)
    }

    #[pyo3(signature = (anchors, *, limit=None, policy=None, cancellation=None))]
    fn conformal_map(
        &self,
        py: Python<'_>,
        anchors: [usize; 2],
        limit: Option<&PyRealizationLimit>,
        policy: Option<&crate::solve::PyPolicy>,
        cancellation: Option<&crate::solve::PyCancellationToken>,
    ) -> PyResult<PyLeastSquaresConformalMapSolution> {
        let policy = crate::solve::policy(policy);
        let limit = limit.copied().unwrap_or(PyRealizationLimit::DEFAULT).core();
        let cancellation = crate::solve::cancellation_token(cancellation);
        let surface = Arc::clone(&self.inner);
        py.detach(move || {
            surface.least_squares_conformal_map(anchors, limit, policy, &cancellation)
        })
        .map(|inner| PyLeastSquaresConformalMapSolution { inner })
        .map_err(crate::solve::surface_computation_error)
    }
}

#[pyclass(
    name = "ConformalMap",
    frozen,
    module = "polygeo.geometry",
    skip_from_py_object
)]
pub(crate) struct PyLeastSquaresConformalMapSolution {
    inner: LeastSquaresConformalMapSolution,
}

#[pymethods]
impl PyLeastSquaresConformalMapSolution {
    #[getter]
    fn geometry(&self) -> NativeEuclideanRealization {
        NativeEuclideanRealization::from_owner(Arc::clone(self.inner.realization()))
    }
    #[getter]
    fn required_rank(&self) -> usize {
        self.inner.evidence().required_rank()
    }
    #[getter]
    fn observed_rank(&self) -> usize {
        self.inner.evidence().observed_rank()
    }
    #[getter]
    fn condition_indicator(&self) -> f64 {
        self.inner.evidence().condition_indicator()
    }
    #[getter]
    fn residual_bound(&self) -> f64 {
        self.inner.evidence().residual_bound()
    }
    #[getter]
    fn minimum_normalized_signed_twice_area(&self) -> f64 {
        self.inner.evidence().minimum_normalized_signed_twice_area()
    }
    #[getter]
    fn exact_fallback_faces(&self) -> usize {
        self.inner.evidence().exact_fallback_faces()
    }
}

#[pyclass(
    name = "VectorField",
    frozen,
    module = "polygeo.geometry",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyEntityVectors {
    inner: PyEntityVectorValue,
}

#[derive(Clone, Debug)]
enum PyEntityVectorValue {
    Vertex(VertexVectors),
    Face(FaceVectors),
}

impl PyEntityVectors {
    fn parts(&self) -> (&Arc<EuclideanRealization>, &[f64]) {
        match &self.inner {
            PyEntityVectorValue::Vertex(field) => (field.realization(), field.values()),
            PyEntityVectorValue::Face(field) => (field.realization(), field.values()),
        }
    }

    fn face(&self) -> Result<&FaceVectors, SurfaceError> {
        match &self.inner {
            PyEntityVectorValue::Face(field) => Ok(field),
            PyEntityVectorValue::Vertex(_) => Err(SurfaceError::FieldShape),
        }
    }
}

#[pymethods]
impl PyEntityVectors {
    #[getter]
    fn geometry(&self) -> NativeEuclideanRealization {
        NativeEuclideanRealization::from_owner(Arc::clone(self.parts().0))
    }
    #[getter]
    fn entity_count(&self) -> usize {
        self.parts().1.len() / self.fiber_dimension()
    }
    #[getter]
    fn fiber_dimension(&self) -> usize {
        self.parts().0.ambient_dimension()
    }
    #[getter]
    fn support_degree(&self) -> usize {
        match &self.inner {
            PyEntityVectorValue::Vertex(field) => field.support_degree(),
            PyEntityVectorValue::Face(field) => field.support_degree(),
        }
    }
    fn values_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let (realization, values) = self.parts();
        let dimension = realization.ambient_dimension();
        project_rows(py, values, values.len() / dimension, dimension)
    }
    fn normalized(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            PyEntityVectorValue::Vertex(field) => {
                PyEntityVectorValue::Vertex(field.normalized().map_err(surface_error)?)
            }
            PyEntityVectorValue::Face(field) => {
                PyEntityVectorValue::Face(field.normalized().map_err(surface_error)?)
            }
        };
        Ok(Self { inner })
    }
}

#[pyclass(
    name = "Connection",
    frozen,
    module = "polygeo.field",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PySurfaceConnection {
    inner: Arc<SurfaceConnection>,
}

#[pymethods]
impl PySurfaceConnection {
    #[getter]
    fn surface(&self) -> PyTriangleSurface {
        PyTriangleSurface {
            inner: Arc::clone(self.inner.surface()),
        }
    }
    #[getter]
    fn symmetry_order(&self) -> u32 {
        self.inner.symmetry_order().get()
    }
    fn transports_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        project_rows(
            py,
            self.inner.transports(),
            self.inner.transports().len() / 2,
            2,
        )
    }
    fn interior_edge_indices_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let values = self.inner.interior_edge_indices();
        filled_array_1d(py, values.len(), |output: &mut [i64]| {
            for (target, &value) in output.iter_mut().zip(values) {
                *target = i64::try_from(value).map_err(|_| TopologyError::IndexOverflow)?;
            }
            Ok(())
        })
    }
    fn holonomy(&self, cycles: &PyIntegralDualCycleBasis) -> PyResult<PyHolonomyEvidence> {
        Ok(PyHolonomyEvidence {
            inner: self.inner.holonomy(&cycles.inner).map_err(surface_error)?,
        })
    }
    fn require_integrable(&self) -> PyResult<PyIntegrableConnection> {
        Ok(PyIntegrableConnection {
            inner: self.inner.require_integrable().map_err(surface_error)?,
        })
    }
}

#[pyclass(
    name = "DualCycles",
    frozen,
    module = "polygeo.field",
    skip_from_py_object
)]
pub(crate) struct PyIntegralDualCycleBasis {
    pub(crate) inner: IntegralDualCycleBasis,
}

#[pymethods]
impl PyIntegralDualCycleBasis {
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn generator_edge_indices_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let values = self.inner.generator_edge_indices();
        filled_array_1d(py, values.len(), |output: &mut [i64]| {
            for (target, &value) in output.iter_mut().zip(values) {
                *target = i64::try_from(value).map_err(|_| TopologyError::IndexOverflow)?;
            }
            Ok(())
        })
    }
    fn cocycle(&self, index: usize) -> PyResult<NativeChainElement> {
        self.inner
            .cocycle(index)
            .cloned()
            .map(|value| NativeChainElement {
                inner: ExactElement::IntegerCochain(value),
            })
            .ok_or_else(|| PyValueError::new_err("cycle index outside basis"))
    }
}

#[pyclass(
    name = "Holonomy",
    frozen,
    module = "polygeo.field",
    skip_from_py_object
)]
struct PyHolonomyEvidence {
    inner: HolonomyEvidence,
}

#[pymethods]
impl PyHolonomyEvidence {
    #[getter]
    fn local_error(&self) -> f64 {
        self.inner.local_error()
    }
    #[getter]
    fn generator_error(&self) -> f64 {
        self.inner.generator_error()
    }
    #[getter]
    fn limit(&self) -> f64 {
        self.inner.limit()
    }
}

#[pyclass(
    name = "IntegrableConnection",
    frozen,
    module = "polygeo.field",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyIntegrableConnection {
    inner: IntegrableConnection,
}

#[pymethods]
impl PyIntegrableConnection {
    #[getter]
    fn connection(&self) -> PySurfaceConnection {
        PySurfaceConnection {
            inner: Arc::clone(self.inner.connection()),
        }
    }
    fn direction(&self, anchor_angle: f64) -> PyResult<PyFaceDirectionField> {
        Ok(PyFaceDirectionField {
            inner: self
                .inner
                .direction_field(anchor_angle)
                .map_err(surface_error)?,
        })
    }
    #[getter]
    fn crossing_error(&self) -> PyResult<f64> {
        self.inner.crossing_error().map_err(surface_error)
    }
}

#[pyclass(
    name = "Direction",
    frozen,
    module = "polygeo.field",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyFaceDirectionField {
    inner: FaceDirectionField,
}

#[pymethods]
impl PyFaceDirectionField {
    #[getter]
    fn connection(&self) -> PyIntegrableConnection {
        PyIntegrableConnection {
            inner: self.inner.connection().clone(),
        }
    }
    #[getter]
    fn symmetry_order(&self) -> u32 {
        self.inner.symmetry_order().get()
    }
    fn power_directions_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        project_rows(
            py,
            self.inner.power_directions(),
            self.inner.connection().connection().surface().face_count(),
            2,
        )
    }
    fn ambient_branch_numpy_copy(&self, branch: usize) -> PyResult<PyEntityVectors> {
        face_field(self.inner.ambient_vector_branch_copy(branch))
    }
    fn singularities(&self) -> PyResult<PyDirectionFieldSingularities> {
        Ok(PyDirectionFieldSingularities {
            inner: self.inner.singularities().map_err(surface_error)?,
        })
    }
}

#[pyclass(
    name = "Singularities",
    frozen,
    module = "polygeo.field",
    skip_from_py_object
)]
struct PyDirectionFieldSingularities {
    inner: DirectionFieldSingularities,
}

#[pymethods]
impl PyDirectionFieldSingularities {
    #[getter]
    fn symmetry_order(&self) -> u32 {
        self.inner.symmetry_order().get()
    }
    #[getter]
    fn charges(&self) -> NativeChainElement {
        NativeChainElement {
            inner: ExactElement::IntegerCochain(self.inner.charges().clone()),
        }
    }
    fn boundary_turns_python_copy(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        bigint_tuple(py, self.inner.boundary_turns())
    }
    #[getter]
    fn maximum_quantization_residual(&self) -> f64 {
        self.inner.maximum_quantization_residual()
    }
    #[getter]
    fn residual_limit(&self) -> f64 {
        self.inner.residual_limit()
    }
}

fn vertex_field(value: Result<VertexVectors, SurfaceError>) -> PyResult<PyEntityVectors> {
    Ok(PyEntityVectors {
        inner: PyEntityVectorValue::Vertex(value.map_err(surface_error)?),
    })
}
fn face_field(value: Result<FaceVectors, SurfaceError>) -> PyResult<PyEntityVectors> {
    Ok(PyEntityVectors {
        inner: PyEntityVectorValue::Face(value.map_err(surface_error)?),
    })
}
fn connection(
    value: Result<Arc<SurfaceConnection>, SurfaceError>,
) -> PyResult<PySurfaceConnection> {
    Ok(PySurfaceConnection {
        inner: value.map_err(surface_error)?,
    })
}
fn project_rows(
    py: Python<'_>,
    values: &[f64],
    rows: usize,
    columns: usize,
) -> PyResult<Py<PyAny>> {
    filled_array_2d(py, rows, columns, |output| {
        output.copy_from_slice(values);
        Ok(())
    })
}

pub(crate) fn register_geometry(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let error = module.py().get_type::<SurfaceErrorPy>();
    error.setattr("__module__", "polygeo.geometry")?;
    module.add("SurfaceError", error)?;
    module.add_class::<PyTriangleSurface>()?;
    module.add_class::<PyLeastSquaresConformalMapSolution>()?;
    module.add_class::<PyEntityVectors>()?;
    let field = module.getattr("VectorField")?;
    module.add("VertexField", field.clone())?;
    module.add("FaceField", field)
}

pub(crate) fn register_field(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySurfaceConnection>()?;
    module.add_class::<PyIntegralDualCycleBasis>()?;
    module.add_class::<PyHolonomyEvidence>()?;
    module.add_class::<PyIntegrableConnection>()?;
    module.add_class::<PyFaceDirectionField>()?;
    module.add_class::<PyDirectionFieldSingularities>()
}
