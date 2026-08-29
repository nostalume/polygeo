use std::sync::Arc;

use polygeo_core::{
    HomologyError as CoreHomologyError, HomologyLimit as CoreHomologyLimit,
    IntegralHomology as CoreIntegralHomology, StorageLimit, WorkLimit,
};
use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule, PyTuple};

use super::{
    ChainError, ExactComplex, ExactElement, NativeChainComplex, NativeChainElement, bigint_tuple,
    classified_exception,
};

create_exception!(
    _polygeo_native,
    HomologyError,
    ChainError,
    "Classified exact integral-homology failure."
);

fn error(reason: &'static str, message: impl Into<String>) -> PyErr {
    Python::attach(|py| {
        let details = PyDict::new(py);
        let error = HomologyError::new_err((reason, message.into(), details.clone().unbind()));
        classified_exception(py, error, reason, details.unbind())
    })
}

fn homology_error(value: CoreHomologyError) -> PyErr {
    Python::attach(|py| {
        let details = PyDict::new(py);
        if let Some((axis, required, limit)) = value.resource_limit()
            && details
                .set_item("axis", axis)
                .and_then(|()| details.set_item("required", required))
                .and_then(|()| details.set_item("limit", limit))
                .is_err()
        {
            return error("translation", "failed to translate homology failure");
        }
        let exception =
            HomologyError::new_err((value.reason(), value.to_string(), details.clone().unbind()));
        classified_exception(py, exception, value.reason(), details.unbind())
    })
}

fn limit(value: Option<&Bound<'_, PyAny>>, current: u64) -> PyResult<u64> {
    value.map_or(Ok(current), |value| {
        value
            .extract()
            .map_err(|_| error("limit", "homology limit axis must fit u64"))
    })
}

#[pyclass(
    name = "HomologyLimit",
    frozen,
    module = "polygeo",
    skip_from_py_object
)]
#[derive(Clone, Copy)]
struct PyHomologyLimit {
    inner: CoreHomologyLimit,
}

impl PyHomologyLimit {
    const DEFAULT: Self = Self {
        inner: CoreHomologyLimit::DEFAULT,
    };

    const fn core(self) -> CoreHomologyLimit {
        self.inner
    }
}

#[pymethods]
impl PyHomologyLimit {
    #[getter]
    const fn retained_logical_bytes(&self) -> u64 {
        self.inner.storage().retained_logical_bytes()
    }

    #[getter]
    const fn peak_live_logical_bytes(&self) -> u64 {
        self.inner.storage().peak_live_logical_bytes()
    }

    #[getter]
    const fn coefficient_bits(&self) -> u64 {
        self.inner.coefficient_bits()
    }

    #[getter]
    const fn smith_steps(&self) -> u64 {
        self.inner.smith_steps().steps()
    }

    #[pyo3(signature = (*, retained_logical_bytes=None, peak_live_logical_bytes=None, coefficient_bits=None, smith_steps=None))]
    fn replace(
        &self,
        retained_logical_bytes: Option<&Bound<'_, PyAny>>,
        peak_live_logical_bytes: Option<&Bound<'_, PyAny>>,
        coefficient_bits: Option<&Bound<'_, PyAny>>,
        smith_steps: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let storage = StorageLimit::new(
            limit(retained_logical_bytes, self.retained_logical_bytes())?,
            limit(peak_live_logical_bytes, self.peak_live_logical_bytes())?,
        )
        .ok_or_else(|| {
            error(
                "limit",
                "peak_live_logical_bytes must contain retained_logical_bytes",
            )
        })?;
        Ok(Self {
            inner: self
                .inner
                .with_storage(storage)
                .with_coefficient_bits(limit(coefficient_bits, self.coefficient_bits())?)
                .with_smith_steps(WorkLimit::new(limit(smith_steps, self.smith_steps())?)),
        })
    }
}

#[pyclass(name = "IntegralHomology", frozen, module = "polygeo")]
struct PyIntegralHomology {
    analysis: Arc<CoreIntegralHomology>,
}

impl PyIntegralHomology {
    fn prepare(
        py: Python<'_>,
        chain: &NativeChainComplex,
        degrees: Vec<usize>,
        limit: CoreHomologyLimit,
    ) -> PyResult<Self> {
        let ExactComplex::Integer(chain) = &chain.inner else {
            return Err(error(
                "coefficient_system",
                "integral homology requires a chain complex over Z",
            ));
        };
        let chain = chain.clone();
        py.detach(move || CoreIntegralHomology::prepare(&chain, degrees, limit))
            .map(|analysis| Self {
                analysis: Arc::new(analysis),
            })
            .map_err(homology_error)
    }
}

#[pymethods]
impl PyIntegralHomology {
    #[getter]
    fn degrees(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.analysis.degrees())?.unbind())
    }

    fn __getitem__(&self, degree: usize) -> PyResult<PyHomologyGroup> {
        self.analysis
            .group(degree)
            .ok_or_else(|| error("degree_not_requested", "homology degree was not requested"))?;
        Ok(PyHomologyGroup {
            analysis: Arc::clone(&self.analysis),
            degree,
        })
    }
}

#[pyclass(name = "HomologyGroup", frozen, module = "polygeo")]
struct PyHomologyGroup {
    analysis: Arc<CoreIntegralHomology>,
    degree: usize,
}

impl PyHomologyGroup {
    fn row(&self) -> polygeo_core::HomologyGroup<'_> {
        self.analysis
            .group(self.degree)
            .expect("native group retains one admitted analysis row")
    }
}

fn chain(value: Option<&polygeo_core::IntegralChain>) -> PyResult<NativeChainElement> {
    value
        .cloned()
        .map(|value| NativeChainElement {
            inner: ExactElement::IntegerChain(value),
        })
        .ok_or_else(|| {
            error(
                "generator_outside",
                "homology generator is outside the group",
            )
        })
}

#[pymethods]
impl PyHomologyGroup {
    #[getter]
    const fn degree(&self) -> usize {
        self.degree
    }

    #[getter]
    fn free_rank(&self) -> usize {
        self.row().free_rank()
    }

    #[getter]
    fn torsion_orders(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        bigint_tuple(py, self.row().torsion_orders())
    }

    fn free_cycle(&self, index: usize) -> PyResult<NativeChainElement> {
        chain(self.row().free_cycle(index))
    }

    fn torsion_cycle(&self, index: usize) -> PyResult<NativeChainElement> {
        chain(self.row().torsion_cycle(index))
    }

    fn torsion_bound(&self, index: usize) -> PyResult<NativeChainElement> {
        chain(self.row().torsion_bound(index))
    }
}

#[pyfunction]
#[pyo3(signature = (chain, degrees, *, limit=None))]
fn prepare_integral_homology(
    py: Python<'_>,
    chain: &NativeChainComplex,
    degrees: Vec<usize>,
    limit: Option<&PyHomologyLimit>,
) -> PyResult<PyIntegralHomology> {
    PyIntegralHomology::prepare(
        py,
        chain,
        degrees,
        limit.copied().unwrap_or(PyHomologyLimit::DEFAULT).core(),
    )
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyHomologyLimit>()?;
    module.add_class::<PyIntegralHomology>()?;
    module.add_class::<PyHomologyGroup>()?;
    module.add(
        "DEFAULT_HOMOLOGY_LIMIT",
        Py::new(module.py(), PyHomologyLimit::DEFAULT)?,
    )?;
    module.add_function(wrap_pyfunction!(prepare_integral_homology, module)?)?;
    module
        .getattr("prepare_integral_homology")?
        .setattr("__module__", "polygeo")?;
    let error = module.py().get_type::<HomologyError>();
    error.setattr("__module__", "polygeo")?;
    module.add("HomologyError", error)
}
