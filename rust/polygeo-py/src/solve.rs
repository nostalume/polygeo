use polygeo_core::{
    CancellationToken, DirichletProblem, FlowProblem, HarmonicExtension, HodgeProblem,
    MeanZeroPoisson, NativeExecutor, Prepared, ProblemError, SolveError, SolveExt, SolveWorkspace,
    StorageLimit, WorkLimit,
};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};

use crate::classified_exception;
use crate::form::{Element, PyBinary64Element};
use crate::realization::NativeEuclideanRealization;

create_exception!(_polygeo_native, ProblemErrorPy, PyValueError);
create_exception!(_polygeo_native, SolveErrorPy, PyValueError);

pub(crate) fn problem_error(error: ProblemError) -> PyErr {
    Python::attach(|py| {
        classified_exception(
            py,
            ProblemErrorPy::new_err(error.to_string()),
            error.reason(),
            PyDict::new(py).unbind(),
        )
    })
}

fn solve_error(error: SolveError) -> PyErr {
    Python::attach(|py| {
        classified_exception(
            py,
            SolveErrorPy::new_err(error.to_string()),
            error.reason(),
            PyDict::new(py).unbind(),
        )
    })
}

#[pyclass(name = "StorageLimit", frozen, module = "polygeo", skip_from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct PyStorageLimit {
    retained: u64,
    peak: u64,
}

impl PyStorageLimit {
    const DEFAULT: Self = Self {
        retained: 128 * 1024 * 1024,
        peak: 512 * 1024 * 1024,
    };
    fn core(self) -> StorageLimit {
        StorageLimit::new(self.retained, self.peak).expect("admitted limit")
    }
}

#[pymethods]
impl PyStorageLimit {
    #[new]
    fn new(retained_logical_bytes: u64, peak_live_logical_bytes: u64) -> PyResult<Self> {
        StorageLimit::new(retained_logical_bytes, peak_live_logical_bytes)
            .ok_or_else(|| PyValueError::new_err("peak storage must contain retained storage"))?;
        Ok(Self {
            retained: retained_logical_bytes,
            peak: peak_live_logical_bytes,
        })
    }
    #[getter]
    fn retained_logical_bytes(&self) -> u64 {
        self.retained
    }
    #[getter]
    fn peak_live_logical_bytes(&self) -> u64 {
        self.peak
    }
}

#[pyclass(name = "WorkLimit", frozen, module = "polygeo", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyWorkLimit {
    steps: u64,
}

impl PyWorkLimit {
    const DEFAULT: Self = Self { steps: 100_000_000 };
    fn core(self) -> WorkLimit {
        WorkLimit::new(self.steps)
    }
}

#[pymethods]
impl PyWorkLimit {
    #[new]
    fn new(steps: u64) -> Self {
        Self { steps }
    }
    #[getter]
    const fn steps(&self) -> u64 {
        self.steps
    }
}

#[pyclass(
    name = "NativeExecutor",
    frozen,
    module = "polygeo",
    skip_from_py_object
)]
#[derive(Clone, Copy)]
pub(crate) struct PyNativeExecutor {
    inner: NativeExecutor,
}

#[pymethods]
impl PyNativeExecutor {
    #[staticmethod]
    fn sequential() -> Self {
        Self {
            inner: NativeExecutor::sequential(),
        }
    }
}

#[pyclass(
    name = "CancellationToken",
    frozen,
    module = "polygeo",
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyCancellationToken {
    inner: CancellationToken,
}

#[pymethods]
impl PyCancellationToken {
    #[new]
    fn new() -> Self {
        Self {
            inner: CancellationToken::new(),
        }
    }
    fn cancel(&self) {
        self.inner.cancel();
    }
    #[getter]
    fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Problem {
    MeanZero(MeanZeroPoisson),
    Dirichlet(DirichletProblem),
    Harmonic(HarmonicExtension),
    Hodge(HodgeProblem),
    Flow(FlowProblem),
}

#[pyclass(name = "Problem", frozen, module = "polygeo", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyProblem {
    pub(crate) inner: Problem,
}

enum PreparedProblem {
    MeanZero(Prepared<MeanZeroPoisson>),
    Dirichlet(Prepared<DirichletProblem>),
    Harmonic(Prepared<HarmonicExtension>),
    Hodge(Prepared<HodgeProblem>),
    Flow(Prepared<FlowProblem>),
}

#[pyclass(
    name = "PreparedProblem",
    frozen,
    module = "polygeo",
    skip_from_py_object
)]
pub(crate) struct PyPreparedProblem {
    inner: PreparedProblem,
}

#[pyclass(name = "SolveWorkspace", module = "polygeo", skip_from_py_object)]
pub(crate) struct PySolveWorkspace {
    inner: SolveWorkspace,
}

fn policies(
    executor: Option<&PyNativeExecutor>,
    storage: Option<&PyStorageLimit>,
    work: Option<&PyWorkLimit>,
) -> (NativeExecutor, StorageLimit, WorkLimit) {
    (
        executor.map_or(NativeExecutor::sequential(), |x| x.inner),
        storage.copied().unwrap_or(PyStorageLimit::DEFAULT).core(),
        work.cloned().unwrap_or(PyWorkLimit::DEFAULT).core(),
    )
}

#[pymethods]
impl PyProblem {
    #[pyo3(signature = (*, executor=None, storage=None, work=None, cancellation=None))]
    fn prepare(
        &self,
        py: Python<'_>,
        executor: Option<&PyNativeExecutor>,
        storage: Option<&PyStorageLimit>,
        work: Option<&PyWorkLimit>,
        cancellation: Option<&PyCancellationToken>,
    ) -> PyResult<PyPreparedProblem> {
        let (executor, storage, work) = policies(executor, storage, work);
        let token = cancellation.map_or_else(CancellationToken::new, |x| x.inner.clone());
        let problem = self.inner.clone();
        let inner = py
            .detach(move || match problem {
                Problem::MeanZero(x) => x
                    .prepare_with_cancellation(&executor, storage, work, &token)
                    .map(PreparedProblem::MeanZero),
                Problem::Dirichlet(x) => x
                    .prepare_with_cancellation(&executor, storage, work, &token)
                    .map(PreparedProblem::Dirichlet),
                Problem::Harmonic(x) => x
                    .prepare_with_cancellation(&executor, storage, work, &token)
                    .map(PreparedProblem::Harmonic),
                Problem::Hodge(x) => x
                    .prepare_with_cancellation(&executor, storage, work, &token)
                    .map(PreparedProblem::Hodge),
                Problem::Flow(x) => x
                    .prepare_with_cancellation(&executor, storage, work, &token)
                    .map(PreparedProblem::Flow),
            })
            .map_err(solve_error)?;
        Ok(PyPreparedProblem { inner })
    }
}

#[pymethods]
impl PyPreparedProblem {
    #[pyo3(signature = (problem, *, storage=None))]
    fn workspace_for(
        &self,
        problem: &PyProblem,
        storage: Option<&PyStorageLimit>,
    ) -> PyResult<PySolveWorkspace> {
        let storage = storage.copied().unwrap_or(PyStorageLimit::DEFAULT).core();
        let inner = match (&self.inner, &problem.inner) {
            (PreparedProblem::MeanZero(a), Problem::MeanZero(b)) => a.workspace_for(b, storage),
            (PreparedProblem::Dirichlet(a), Problem::Dirichlet(b)) => a.workspace_for(b, storage),
            (PreparedProblem::Harmonic(a), Problem::Harmonic(b)) => a.workspace_for(b, storage),
            (PreparedProblem::Hodge(a), Problem::Hodge(b)) => a.workspace_for(b, storage),
            (PreparedProblem::Flow(a), Problem::Flow(b)) => a.workspace_for(b, storage),
            _ => return Err(solve_error(SolveError::ProblemMismatch)),
        }
        .map_err(solve_error)?;
        Ok(PySolveWorkspace { inner })
    }

    #[pyo3(signature = (problem, workspace, *, work=None, cancellation=None))]
    fn solve(
        &self,
        py: Python<'_>,
        problem: &PyProblem,
        workspace: &mut PySolveWorkspace,
        work: Option<&PyWorkLimit>,
        cancellation: Option<&PyCancellationToken>,
    ) -> PyResult<Py<PyAny>> {
        let work = work.cloned().unwrap_or(PyWorkLimit::DEFAULT).core();
        let token = cancellation.map_or_else(CancellationToken::new, |x| x.inner.clone());
        match (&self.inner, &problem.inner) {
            (PreparedProblem::MeanZero(a), Problem::MeanZero(b)) => {
                let value = py
                    .detach(|| a.solve_cancellable(b, &mut workspace.inner, work, &token))
                    .map_err(solve_error)?;
                Ok(Py::new(py, PyPoissonSolution { inner: value })?.into_any())
            }
            (PreparedProblem::Dirichlet(a), Problem::Dirichlet(b)) => {
                let value = py
                    .detach(|| a.solve_cancellable(b, &mut workspace.inner, work, &token))
                    .map_err(solve_error)?;
                Ok(Py::new(py, PyDirichletSolution { inner: value })?.into_any())
            }
            (PreparedProblem::Harmonic(a), Problem::Harmonic(b)) => {
                let value = py
                    .detach(|| a.solve_cancellable(b, &mut workspace.inner, work, &token))
                    .map_err(solve_error)?;
                Ok(Py::new(py, PyDirichletSolution { inner: value })?.into_any())
            }
            (PreparedProblem::Hodge(a), Problem::Hodge(b)) => {
                let value = py
                    .detach(|| a.solve_cancellable(b, &mut workspace.inner, work, &token))
                    .map_err(solve_error)?;
                Ok(Py::new(py, PyHodgeDecomposition { inner: value })?.into_any())
            }
            (PreparedProblem::Flow(a), Problem::Flow(b)) => {
                let value = py
                    .detach(|| a.solve_cancellable(b, &mut workspace.inner, work, &token))
                    .map_err(solve_error)?;
                Ok(Py::new(py, PyFlowStep { inner: value })?.into_any())
            }
            _ => Err(solve_error(SolveError::ProblemMismatch)),
        }
    }
}

#[pyclass(
    name = "PoissonSolution",
    frozen,
    module = "polygeo",
    skip_from_py_object
)]
struct PyPoissonSolution {
    inner: polygeo_core::PoissonSolution,
}
#[pymethods]
impl PyPoissonSolution {
    #[getter]
    fn potential(&self) -> PyBinary64Element {
        PyBinary64Element {
            inner: Element::Cochain(self.inner.potential().clone()),
        }
    }
    #[getter]
    fn residual_bound(&self) -> f64 {
        self.inner.evidence().residual_bound()
    }
    #[getter]
    fn gauge_bound(&self) -> f64 {
        self.inner.evidence().gauge_bound()
    }
    #[getter]
    fn exact_fallback_rows(&self) -> usize {
        self.inner.evidence().exact_fallback_rows()
    }
}

#[pyclass(
    name = "DirichletSolution",
    frozen,
    module = "polygeo",
    skip_from_py_object
)]
struct PyDirichletSolution {
    inner: polygeo_core::DirichletSolution,
}
#[pymethods]
impl PyDirichletSolution {
    #[getter]
    fn value(&self) -> PyBinary64Element {
        PyBinary64Element {
            inner: Element::Cochain(self.inner.value().clone()),
        }
    }
    #[getter]
    fn residual_bound(&self) -> f64 {
        self.inner.evidence().residual_bound()
    }
    #[getter]
    fn exact_fallback_rows(&self) -> usize {
        self.inner.evidence().exact_fallback_rows()
    }
}

#[pyclass(
    name = "HodgeDecomposition",
    frozen,
    module = "polygeo",
    skip_from_py_object
)]
struct PyHodgeDecomposition {
    inner: polygeo_core::HodgeDecomposition,
}
#[pymethods]
impl PyHodgeDecomposition {
    #[getter]
    fn exact(&self) -> PyBinary64Element {
        PyBinary64Element {
            inner: Element::Cochain(self.inner.exact().clone()),
        }
    }
    #[getter]
    fn coexact(&self) -> PyBinary64Element {
        PyBinary64Element {
            inner: Element::Cochain(self.inner.coexact().clone()),
        }
    }
    #[getter]
    fn harmonic(&self) -> PyBinary64Element {
        PyBinary64Element {
            inner: Element::Cochain(self.inner.harmonic().clone()),
        }
    }
    #[getter]
    fn reconstruction_bound(&self) -> f64 {
        self.inner.evidence().reconstruction_bound()
    }
    #[getter]
    fn orthogonality_bound(&self) -> f64 {
        self.inner.evidence().orthogonality_bound()
    }
}

#[pyclass(name = "FlowStep", frozen, module = "polygeo", skip_from_py_object)]
struct PyFlowStep {
    inner: polygeo_core::FlowStep,
}
#[pymethods]
impl PyFlowStep {
    #[getter]
    fn target(&self) -> NativeEuclideanRealization {
        NativeEuclideanRealization::from_owner(self.inner.target().clone())
    }
    #[getter]
    fn energy_before(&self) -> f64 {
        self.inner.evidence().energy_before()
    }
    #[getter]
    fn energy_after(&self) -> f64 {
        self.inner.evidence().energy_after()
    }
    #[getter]
    fn residual_bound(&self) -> f64 {
        self.inner.evidence().residual_bound()
    }
    #[getter]
    fn centroid_residual_bound(&self) -> f64 {
        self.inner.evidence().centroid_residual_bound()
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ProblemError", module.py().get_type::<ProblemErrorPy>())?;
    module.add("SolveError", module.py().get_type::<SolveErrorPy>())?;
    module.add_class::<PyStorageLimit>()?;
    module.add_class::<PyWorkLimit>()?;
    module.add_class::<PyNativeExecutor>()?;
    module.add_class::<PyCancellationToken>()?;
    module.add_class::<PyProblem>()?;
    module.add_class::<PyPreparedProblem>()?;
    module.add_class::<PySolveWorkspace>()?;
    module.add_class::<PyPoissonSolution>()?;
    module.add_class::<PyDirichletSolution>()?;
    module.add_class::<PyHodgeDecomposition>()?;
    module.add_class::<PyFlowStep>()
}
