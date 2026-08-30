use std::sync::Arc;

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use polygeo_core::{
    EntityVectors, FaceDirectionField, HolonomyEvidence, IntegrableConnection,
    IntegralDualCycleBasis, LeastSquaresConformalMapSolution, SurfaceConnection, SurfaceError,
    TriangleSurface,
};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};

use crate::form::{Element, PyBinary64Element};
use crate::realization::{NativeEuclideanRealization, PyPositiveMetric, PyRealizationLimit};
use crate::{
    ExactElement, NativeChainElement, classified_exception, filled_array_1d, filled_array_2d,
};

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
    module = "polygeo",
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
            inner: TriangleSurface::admit(Arc::clone(realization.topology()))
                .map_err(surface_error)?,
        })
    }
    #[getter]
    fn realization(&self) -> NativeEuclideanRealization {
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

    fn vertex_vectors(&self, values: &Bound<'_, PyAny>) -> PyResult<PyEntityVectors> {
        let values = values.extract::<PyReadonlyArray2<'_, f64>>()?;
        Ok(PyEntityVectors {
            inner: self
                .inner
                .vertex_vectors(values.as_array().iter().copied().collect())
                .map_err(surface_error)?,
        })
    }
    fn face_vectors(&self, values: &Bound<'_, PyAny>) -> PyResult<PyEntityVectors> {
        let values = values.extract::<PyReadonlyArray2<'_, f64>>()?;
        Ok(PyEntityVectors {
            inner: self
                .inner
                .face_vectors(values.as_array().iter().copied().collect())
                .map_err(surface_error)?,
        })
    }
    fn gradient(&self, source: &PyBinary64Element) -> PyResult<PyEntityVectors> {
        let Element::Cochain(source) = &source.inner else {
            return Err(surface_error(SurfaceError::FieldShape));
        };
        field(self.inner.gradient(source))
    }
    fn divergence(&self, field: &PyEntityVectors) -> PyResult<PyBinary64Element> {
        Ok(PyBinary64Element {
            inner: Element::Chain(self.inner.divergence(&field.inner).map_err(surface_error)?),
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
        field(self.inner.face_unit_normals())
    }
    fn uniform_vertex_normals(&self) -> PyResult<PyEntityVectors> {
        field(self.inner.uniform_vertex_normals())
    }
    fn tip_angle_vertex_normals(&self) -> PyResult<PyEntityVectors> {
        field(self.inner.tip_angle_vertex_normals())
    }
    fn sphere_inscribed_vertex_normals(&self) -> PyResult<PyEntityVectors> {
        field(self.inner.sphere_inscribed_vertex_normals())
    }
    fn surface_area_gradient(&self) -> PyResult<PyEntityVectors> {
        field(self.inner.surface_area_gradient())
    }
    fn volume_gradient(&self) -> PyResult<PyEntityVectors> {
        field(self.inner.volume_gradient())
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
        field(self.inner.mean_curvature_vectors(&metric.inner))
    }
    fn levi_civita_connection(&self) -> PyResult<PySurfaceConnection> {
        connection(self.inner.levi_civita_connection())
    }
    fn connection(&self, deviations: &Bound<'_, PyAny>) -> PyResult<PySurfaceConnection> {
        let deviations = deviations.extract::<PyReadonlyArray1<'_, f64>>()?;
        let deviations = deviations.as_array().iter().copied().collect::<Vec<_>>();
        connection(self.inner.connection(&deviations))
    }

    #[pyo3(signature = (anchors, *, realization_limit=None, executor=None, storage=None, work=None, cancellation=None))]
    fn least_squares_conformal_map(
        &self,
        anchors: [usize; 2],
        realization_limit: Option<&PyRealizationLimit>,
        executor: Option<&crate::solve::PyNativeExecutor>,
        storage: Option<&crate::solve::PyStorageLimit>,
        work: Option<&crate::solve::PyWorkLimit>,
        cancellation: Option<&crate::solve::PyCancellationToken>,
    ) -> PyResult<PyLeastSquaresConformalMapSolution> {
        let (executor, storage, work) = crate::solve::policies(executor, storage, work);
        let realization_limit = realization_limit
            .copied()
            .unwrap_or(PyRealizationLimit::DEFAULT)
            .core();
        let cancellation = cancellation
            .map_or_else(polygeo_core::CancellationToken::new, |value| {
                value.inner.clone()
            });
        let surface = Arc::clone(&self.inner);
        Python::attach(|py| {
            py.detach(move || {
                surface.least_squares_conformal_map(
                    anchors,
                    realization_limit,
                    &executor,
                    storage,
                    work,
                    &cancellation,
                )
            })
        })
        .map(|inner| PyLeastSquaresConformalMapSolution { inner })
        .map_err(crate::solve::surface_computation_error)
    }
}

#[pyclass(
    name = "LeastSquaresConformalMapSolution",
    frozen,
    module = "polygeo",
    skip_from_py_object
)]
pub(crate) struct PyLeastSquaresConformalMapSolution {
    inner: LeastSquaresConformalMapSolution,
}

#[pymethods]
impl PyLeastSquaresConformalMapSolution {
    #[getter]
    fn realization(&self) -> NativeEuclideanRealization {
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
    name = "EntityVectors",
    frozen,
    module = "polygeo",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyEntityVectors {
    inner: EntityVectors,
}

#[pymethods]
impl PyEntityVectors {
    #[getter]
    fn realization(&self) -> NativeEuclideanRealization {
        NativeEuclideanRealization::from_owner(Arc::clone(self.inner.realization()))
    }
    #[getter]
    fn entity_count(&self) -> usize {
        self.inner.entity_count()
    }
    #[getter]
    fn fiber_dimension(&self) -> usize {
        self.inner.fiber_dimension()
    }
    #[getter]
    fn is_vertex_supported(&self) -> bool {
        self.inner.is_vertex_supported()
    }
    #[getter]
    fn is_face_supported(&self) -> bool {
        self.inner.is_face_supported()
    }
    fn vectors_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        project_rows(
            py,
            self.inner.values(),
            self.inner.entity_count(),
            self.inner.fiber_dimension(),
        )
    }
    fn normalized(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.normalized().map_err(surface_error)?,
        })
    }
}

#[pyclass(
    name = "SurfaceConnection",
    frozen,
    module = "polygeo",
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
    fn transports_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        project_rows(
            py,
            self.inner.transports(),
            self.inner.surface().edge_count(),
            2,
        )
    }
    fn compose(&self, after: &Self) -> PyResult<Self> {
        connection(self.inner.compose(&after.inner))
    }
    fn holonomy(&self, cycles: &PyIntegralDualCycleBasis) -> PyResult<PyHolonomyEvidence> {
        Ok(PyHolonomyEvidence {
            inner: self.inner.holonomy(&cycles.inner).map_err(surface_error)?,
        })
    }
    fn require_integrable(
        &self,
        cycles: &PyIntegralDualCycleBasis,
    ) -> PyResult<PyIntegrableConnection> {
        Ok(PyIntegrableConnection {
            inner: self
                .inner
                .require_integrable(&cycles.inner)
                .map_err(surface_error)?,
        })
    }
}

#[pyclass(
    name = "IntegralDualCycleBasis",
    frozen,
    module = "polygeo",
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
                *target =
                    i64::try_from(value).map_err(|_| polygeo_core::TopologyError::IndexOverflow)?;
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
    name = "HolonomyEvidence",
    frozen,
    module = "polygeo",
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
    module = "polygeo",
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
    fn direction_field(&self, anchor_phase: f64) -> PyResult<PyFaceDirectionField> {
        Ok(PyFaceDirectionField {
            inner: self
                .inner
                .direction_field(anchor_phase)
                .map_err(surface_error)?,
        })
    }
    #[getter]
    fn crossing_error(&self) -> PyResult<f64> {
        self.inner.crossing_error().map_err(surface_error)
    }
}

#[pyclass(
    name = "FaceDirectionField",
    frozen,
    module = "polygeo",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyFaceDirectionField {
    inner: FaceDirectionField,
}

#[pymethods]
impl PyFaceDirectionField {
    #[getter]
    fn connection(&self) -> PySurfaceConnection {
        PySurfaceConnection {
            inner: Arc::clone(self.inner.connection()),
        }
    }
    fn directions_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        project_rows(
            py,
            self.inner.directions(),
            self.inner.connection().surface().face_count(),
            2,
        )
    }
    fn ambient_vectors_numpy_copy(&self) -> PyResult<PyEntityVectors> {
        field(self.inner.ambient_vectors_copy())
    }
    #[getter]
    fn crossing_error(&self) -> PyResult<f64> {
        self.inner.crossing_error().map_err(surface_error)
    }
}

fn field(value: Result<EntityVectors, SurfaceError>) -> PyResult<PyEntityVectors> {
    Ok(PyEntityVectors {
        inner: value.map_err(surface_error)?,
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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("SurfaceError", module.py().get_type::<SurfaceErrorPy>())?;
    module.add_class::<PyTriangleSurface>()?;
    module.add_class::<PyLeastSquaresConformalMapSolution>()?;
    module.add_class::<PyEntityVectors>()?;
    let field = module.getattr("EntityVectors")?;
    module.add("VertexVectors", field.clone())?;
    module.add("FaceVectors", field)?;
    module.add_class::<PySurfaceConnection>()?;
    module.add_class::<PyIntegralDualCycleBasis>()?;
    module.add_class::<PyHolonomyEvidence>()?;
    module.add_class::<PyIntegrableConnection>()?;
    module.add_class::<PyFaceDirectionField>()
}
