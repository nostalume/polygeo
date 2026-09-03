use std::sync::Arc;

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use polygeo_core::geometry::{
    Geometry as EuclideanRealization, GeometryError as RealizationError, Limit as RealizationLimit,
    Metric as PositiveMetric, MetricError as CoreMetricError, NondegenerateCapability,
    PairingCapability,
};
use polygeo_core::solve::{ProblemError, StorageLimit, WorkLimit};
use polygeo_core::topology::Complex as ComplexCore;
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule, PyType};

use crate::array::{filled_array_1d, filled_array_2d};
use crate::classified_exception;
use crate::form::{Operator, PyLinearOperator};
use crate::topology::NativeComplex;

create_exception!(
    _polygeo_native,
    GeometryError,
    PyValueError,
    "Classified Euclidean realization failure."
);

create_exception!(_polygeo_native, MetricError, PyValueError);

fn metric_error(error: CoreMetricError) -> PyErr {
    Python::attach(|py| {
        classified_exception(
            py,
            MetricError::new_err(error.to_string()),
            match error {
                CoreMetricError::Degenerate { .. } => "degenerate",
                CoreMetricError::Indefinite => "indefinite",
                _ => "metric",
            },
            PyDict::new(py).unbind(),
        )
    })
}

fn geometry_error(error: RealizationError) -> PyErr {
    Python::attach(|py| {
        let details = PyDict::new(py);
        if let Some((axis, required, limit)) = error.resource_limit()
            && details
                .set_item("axis", axis)
                .and_then(|()| details.set_item("required", required))
                .and_then(|()| details.set_item("limit", limit))
                .is_err()
        {
            return geometry_input_error("translation", "failed to translate realization failure");
        }
        classified_exception(
            py,
            GeometryError::new_err(error.to_string()),
            error.reason(),
            details.unbind(),
        )
    })
}

fn geometry_input_error(reason: &'static str, message: &'static str) -> PyErr {
    Python::attach(|py| {
        classified_exception(
            py,
            GeometryError::new_err(message),
            reason,
            PyDict::new(py).unbind(),
        )
    })
}

#[pyclass(
    name = "Limit",
    frozen,
    module = "polygeo.geometry",
    skip_from_py_object
)]
#[derive(Clone, Copy)]
pub(crate) struct PyRealizationLimit {
    retained_logical_bytes: u64,
    peak_live_logical_bytes: u64,
    coefficient_bits: u64,
    exact_steps: u64,
}

impl PyRealizationLimit {
    pub(crate) const DEFAULT: Self = Self {
        retained_logical_bytes: 128 * 1024 * 1024,
        peak_live_logical_bytes: 512 * 1024 * 1024,
        coefficient_bits: 65_536,
        exact_steps: 100_000_000,
    };

    pub(crate) fn core(self) -> RealizationLimit {
        let storage = StorageLimit::new(self.retained_logical_bytes, self.peak_live_logical_bytes)
            .expect("Python construction preserves the storage lifecycle");
        RealizationLimit::new(
            storage,
            self.coefficient_bits,
            WorkLimit::new(self.exact_steps),
        )
    }
}

#[pymethods]
impl PyRealizationLimit {
    #[new]
    #[pyo3(signature = (*, retained_logical_bytes=134_217_728, peak_live_logical_bytes=536_870_912, coefficient_bits=65_536, exact_steps=100_000_000))]
    fn new(
        retained_logical_bytes: u64,
        peak_live_logical_bytes: u64,
        coefficient_bits: u64,
        exact_steps: u64,
    ) -> PyResult<Self> {
        StorageLimit::new(retained_logical_bytes, peak_live_logical_bytes).ok_or_else(|| {
            geometry_input_error(
                "limit",
                "peak_live_logical_bytes must contain retained_logical_bytes",
            )
        })?;
        Ok(Self {
            retained_logical_bytes,
            peak_live_logical_bytes,
            coefficient_bits,
            exact_steps,
        })
    }

    #[getter]
    const fn retained_logical_bytes(&self) -> u64 {
        self.retained_logical_bytes
    }
    #[getter]
    const fn peak_live_logical_bytes(&self) -> u64 {
        self.peak_live_logical_bytes
    }
    #[getter]
    const fn coefficient_bits(&self) -> u64 {
        self.coefficient_bits
    }
    #[getter]
    const fn exact_steps(&self) -> u64 {
        self.exact_steps
    }
}

#[pyclass(name = "Geometry", frozen, module = "polygeo.geometry")]
pub(crate) struct NativeEuclideanRealization {
    owner: Arc<EuclideanRealization>,
}

impl NativeEuclideanRealization {
    pub(crate) fn owner(&self) -> &Arc<EuclideanRealization> {
        &self.owner
    }

    pub(crate) fn from_owner(owner: Arc<EuclideanRealization>) -> Self {
        Self { owner }
    }
}

#[pymethods]
impl NativeEuclideanRealization {
    #[classmethod]
    fn __class_getitem__<'py>(
        class: &Bound<'py, PyType>,
        _parameter: &Bound<'_, PyAny>,
    ) -> Bound<'py, PyType> {
        class.clone()
    }

    #[new]
    #[pyo3(signature = (complex, positions, *, limit=None))]
    fn new(
        py: Python<'_>,
        complex: &Bound<'_, PyAny>,
        positions: &Bound<'_, PyAny>,
        limit: Option<&PyRealizationLimit>,
    ) -> PyResult<Self> {
        Self::admit(
            py,
            complex,
            positions,
            limit.copied().unwrap_or(PyRealizationLimit::DEFAULT),
        )
    }

    #[staticmethod]
    #[pyo3(signature = (complex, positions, *, limit=None))]
    fn from_positions(
        py: Python<'_>,
        complex: &Bound<'_, PyAny>,
        positions: &Bound<'_, PyAny>,
        limit: Option<&PyRealizationLimit>,
    ) -> PyResult<Self> {
        Self::admit(
            py,
            complex,
            positions,
            limit.copied().unwrap_or(PyRealizationLimit::DEFAULT),
        )
    }

    #[getter]
    fn topology(&self) -> NativeComplex {
        NativeComplex {
            owner: Arc::clone(self.owner.topology()),
        }
    }

    #[getter]
    fn ambient_dimension(&self) -> usize {
        self.owner.ambient_dimension()
    }

    fn positions_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        filled_array_2d(
            py,
            self.owner.topology().vertex_count(),
            self.owner.ambient_dimension(),
            |output| {
                output.copy_from_slice(self.owner.positions());
                Ok(())
            },
        )
    }

    fn primal_measures_numpy_copy(&self, py: Python<'_>, degree: isize) -> PyResult<Py<PyAny>> {
        let values = self
            .owner
            .primal_measures(admit_degree(degree)?)
            .map_err(geometry_error)?;
        filled_array_1d(py, values.len(), |output| {
            output.copy_from_slice(values);
            Ok(())
        })
    }

    fn dual_measures_numpy_copy(&self, py: Python<'_>, degree: isize) -> PyResult<Py<PyAny>> {
        let degree = admit_degree(degree)?;
        let values = py
            .detach(|| self.owner.dual_measures(degree))
            .map_err(geometry_error)?;
        filled_array_1d(py, values.len(), |output| {
            output.copy_from_slice(values);
            Ok(())
        })
    }

    fn metric(&self, py: Python<'_>) -> PyResult<PyPositiveMetric> {
        let owner = Arc::clone(&self.owner);
        let pairing = py
            .detach(move || owner.circumcentric_pairing())
            .map_err(geometry_error)?;
        let inner = pairing.require_positive().map_err(metric_error)?;
        Ok(PyPositiveMetric { inner })
    }
}

impl NativeEuclideanRealization {
    fn admit(
        py: Python<'_>,
        complex: &Bound<'_, PyAny>,
        positions: &Bound<'_, PyAny>,
        limit: PyRealizationLimit,
    ) -> PyResult<Self> {
        let topology = topology_owner(complex)?;
        let array = positions
            .extract::<PyReadonlyArray2<'_, f64>>()
            .map_err(|_| {
                geometry_input_error("position_dtype", "positions must be a float64 ndarray")
            })?;
        let shape = array.shape();
        if shape.len() != 2 {
            return Err(geometry_input_error(
                "position_shape",
                "positions must be a two-dimensional float64 ndarray",
            ));
        }
        let ambient = shape[1];
        let copied = array.as_array().iter().copied().collect::<Vec<_>>();
        let owner = py
            .detach(|| EuclideanRealization::admit(topology, ambient, copied, limit.core()))
            .map_err(geometry_error)?;
        Ok(Self { owner })
    }
}

#[pyclass(
    name = "Metric",
    frozen,
    module = "polygeo.geometry",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyPositiveMetric {
    pub(crate) inner: PositiveMetric,
}

#[pymethods]
impl PyPositiveMetric {
    #[getter]
    fn geometry(&self) -> NativeEuclideanRealization {
        NativeEuclideanRealization {
            owner: Arc::clone(self.inner.realization()),
        }
    }

    fn hodge_coefficients_numpy_copy(&self, py: Python<'_>, degree: usize) -> PyResult<Py<PyAny>> {
        let values = self
            .inner
            .hodge_coefficients_slice(degree)
            .map_err(geometry_error)?;
        filled_array_1d(py, values.len(), |output| {
            output.copy_from_slice(values);
            Ok(())
        })
    }

    fn riesz(&self, degree: usize) -> PyResult<PyLinearOperator> {
        Ok(PyLinearOperator {
            inner: Operator::CochainChain(
                self.inner
                    .riesz(degree)
                    .map_err(crate::form::operator_error)?,
            ),
        })
    }

    fn inverse_riesz(&self, degree: usize) -> PyResult<PyLinearOperator> {
        Ok(PyLinearOperator {
            inner: Operator::ChainCochain(
                self.inner
                    .inverse_riesz(degree)
                    .map_err(crate::form::operator_error)?,
            ),
        })
    }

    fn codifferential(&self, degree: usize) -> PyResult<PyLinearOperator> {
        Ok(PyLinearOperator {
            inner: Operator::CochainCochain(
                self.inner
                    .codifferential(degree)
                    .map_err(crate::form::operator_error)?,
            ),
        })
    }

    fn laplacian(&self, degree: usize) -> PyResult<PyLinearOperator> {
        Ok(PyLinearOperator {
            inner: Operator::CochainCochain(
                self.inner
                    .laplacian(degree)
                    .map_err(crate::form::operator_error)?,
            ),
        })
    }

    #[pyo3(signature = (group, *, policy=None, cancellation=None))]
    fn harmonic_basis(
        &self,
        py: Python<'_>,
        group: &crate::homology::PyHomologyGroup,
        policy: Option<&crate::solve::PyPolicy>,
        cancellation: Option<&crate::solve::PyCancellationToken>,
    ) -> PyResult<crate::solve::PyHarmonicOneFormBasis> {
        let policy = crate::solve::policy(policy);
        let cancellation = crate::solve::cancellation_token(cancellation);
        let metric = self.inner.clone();
        let analysis = Arc::clone(&group.analysis);
        let degree = group.degree;
        py.detach(move || {
            let group = analysis
                .group(degree)
                .expect("Python homology group retains one admitted analysis row");
            metric.harmonic_one_form_basis(group, policy, &cancellation)
        })
        .map(|inner| crate::solve::PyHarmonicOneFormBasis { inner })
        .map_err(crate::solve::surface_computation_error)
    }

    fn mean_zero_poisson_density(
        &self,
        density: &crate::form::PyBinary64Element,
    ) -> PyResult<crate::solve::PyProblem> {
        let crate::form::Element::Cochain(density) = &density.inner else {
            return Err(crate::solve::problem_error(ProblemError::SpaceMismatch));
        };
        Ok(crate::solve::PyProblem {
            inner: crate::solve::Problem::MeanZero(
                self.inner
                    .mean_zero_poisson_density(density.clone())
                    .map_err(crate::solve::problem_error)?,
            ),
        })
    }

    fn mean_zero_poisson_load(
        &self,
        load: &crate::form::PyBinary64Element,
    ) -> PyResult<crate::solve::PyProblem> {
        let crate::form::Element::Chain(load) = &load.inner else {
            return Err(crate::solve::problem_error(ProblemError::SpaceMismatch));
        };
        Ok(crate::solve::PyProblem {
            inner: crate::solve::Problem::MeanZero(
                self.inner
                    .mean_zero_poisson_load(load.clone())
                    .map_err(crate::solve::problem_error)?,
            ),
        })
    }

    fn harmonic_extension(
        &self,
        values: &crate::form::PyBinary64Element,
    ) -> PyResult<crate::solve::PyProblem> {
        let crate::form::Element::Cochain(values) = &values.inner else {
            return Err(crate::solve::problem_error(ProblemError::SpaceMismatch));
        };
        Ok(crate::solve::PyProblem {
            inner: crate::solve::Problem::Harmonic(
                self.inner
                    .harmonic_extension(values.clone())
                    .map_err(crate::solve::problem_error)?,
            ),
        })
    }

    fn hodge_decomposition(
        &self,
        source: &crate::form::PyBinary64Element,
    ) -> PyResult<crate::solve::PyProblem> {
        let crate::form::Element::Cochain(source) = &source.inner else {
            return Err(crate::solve::problem_error(ProblemError::SpaceMismatch));
        };
        Ok(crate::solve::PyProblem {
            inner: crate::solve::Problem::Hodge(
                self.inner
                    .hodge_decomposition(source.clone())
                    .map_err(crate::solve::problem_error)?,
            ),
        })
    }

    fn heat_evolution(
        &self,
        source: &crate::form::PyBinary64Element,
        time_step: f64,
    ) -> PyResult<crate::solve::PyProblem> {
        let crate::form::Element::Cochain(source) = &source.inner else {
            return Err(crate::solve::problem_error(ProblemError::SpaceMismatch));
        };
        Ok(crate::solve::PyProblem {
            inner: crate::solve::Problem::Heat(
                self.inner
                    .heat_evolution(source.clone(), time_step)
                    .map_err(crate::solve::problem_error)?,
            ),
        })
    }

    #[pyo3(signature = (time_step, *, limit=None, policy=None, cancellation=None))]
    fn frozen_mean_curvature_flow(
        &self,
        py: Python<'_>,
        time_step: f64,
        limit: Option<&PyRealizationLimit>,
        policy: Option<&crate::solve::PyPolicy>,
        cancellation: Option<&crate::solve::PyCancellationToken>,
    ) -> PyResult<crate::solve::PyFlowStep> {
        let policy = crate::solve::policy(policy);
        let limit = limit.copied().unwrap_or(PyRealizationLimit::DEFAULT).core();
        let cancellation = crate::solve::cancellation_token(cancellation);
        let metric = self.inner.clone();
        py.detach(move || {
            metric.frozen_mean_curvature_flow(time_step, limit, policy, &cancellation)
        })
        .map(|inner| crate::solve::PyFlowStep { inner })
        .map_err(crate::solve::surface_computation_error)
    }
}

pub(crate) fn topology_owner(complex: &Bound<'_, PyAny>) -> PyResult<Arc<ComplexCore>> {
    complex
        .extract::<PyRef<'_, NativeComplex>>()
        .map(|value| Arc::clone(&value.owner))
        .map_err(|_| {
            geometry_input_error(
                "topology_owner",
                "realization requires one admitted simplicial Complex",
            )
        })
}

fn admit_degree(degree: isize) -> PyResult<usize> {
    usize::try_from(degree).map_err(|_| geometry_error(RealizationError::DegreeOutside))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let geometry_error = module.py().get_type::<GeometryError>();
    geometry_error.setattr("__module__", "polygeo.geometry")?;
    module.add("GeometryError", geometry_error)?;
    module.add_class::<PyRealizationLimit>()?;
    module.add(
        "DEFAULT_LIMIT",
        Py::new(module.py(), PyRealizationLimit::DEFAULT)?,
    )?;
    module
        .add_class::<NativeEuclideanRealization>()
        .and_then(|()| module.add_class::<PyPositiveMetric>())?;
    let metric_error = module.py().get_type::<MetricError>();
    metric_error.setattr("__module__", "polygeo.geometry")?;
    module.add("MetricError", metric_error)?;
    Ok(())
}
