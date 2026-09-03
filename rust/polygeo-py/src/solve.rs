use polygeo_core::field::{HarmonicBasis, HodgeDecomposition, HodgeProblem};
use polygeo_core::geometry::FlowStep;
use polygeo_core::solve::{
    CancellationToken, DirichletProblem, DirichletResult as DirichletSolution,
    Executor as NativeExecutor, HarmonicExtension, HeatProblem, HeatResult as HeatSolution,
    PoissonProblem as MeanZeroPoisson, PoissonResult as PoissonSolution, Policy as CorePolicy,
    Prepared, ProblemError, SolveError, SolveExt, StorageLimit, SurfaceComputationError, WorkLimit,
    Workspace as SolveWorkspace,
};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule, PyTuple};

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

pub(crate) fn solve_error(error: SolveError) -> PyErr {
    Python::attach(|py| {
        classified_exception(
            py,
            SolveErrorPy::new_err(error.to_string()),
            error.reason(),
            PyDict::new(py).unbind(),
        )
    })
}

pub(crate) fn surface_computation_error(error: SurfaceComputationError) -> PyErr {
    match error {
        SurfaceComputationError::Surface(error) => crate::surface::surface_error(error),
        SurfaceComputationError::Solve(error) => solve_error(error),
        _ => solve_error(SolveError::Numerical),
    }
}

#[pyclass(
    name = "StorageLimit",
    frozen,
    module = "polygeo.solve",
    skip_from_py_object
)]
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

#[pyclass(
    name = "WorkLimit",
    frozen,
    module = "polygeo.solve",
    skip_from_py_object
)]
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
    name = "Executor",
    frozen,
    module = "polygeo.solve",
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

#[pyclass(name = "Policy", frozen, module = "polygeo.solve", skip_from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct PyPolicy {
    pub(crate) inner: CorePolicy,
}

#[pymethods]
impl PyPolicy {
    #[new]
    #[pyo3(signature = (*, executor=None, storage=None, work=None))]
    fn new(
        executor: Option<&PyNativeExecutor>,
        storage: Option<&PyStorageLimit>,
        work: Option<&PyWorkLimit>,
    ) -> Self {
        Self {
            inner: core_policy(executor, storage, work),
        }
    }

    #[getter]
    fn executor(&self) -> PyNativeExecutor {
        PyNativeExecutor {
            inner: self.inner.executor(),
        }
    }

    #[getter]
    fn storage(&self) -> PyStorageLimit {
        let storage = self.inner.storage();
        PyStorageLimit {
            retained: storage.retained_logical_bytes(),
            peak: storage.peak_live_logical_bytes(),
        }
    }

    #[getter]
    fn work(&self) -> PyWorkLimit {
        PyWorkLimit {
            steps: self.inner.work().steps(),
        }
    }
}

#[pyclass(
    name = "CancellationToken",
    frozen,
    module = "polygeo.solve",
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyCancellationToken {
    pub(crate) inner: CancellationToken,
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
    Heat(HeatProblem),
}

#[pyclass(
    name = "Problem",
    frozen,
    module = "polygeo.solve",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyProblem {
    pub(crate) inner: Problem,
}

enum PreparedProblem {
    MeanZero(Prepared<MeanZeroPoisson>),
    Dirichlet(Prepared<DirichletProblem>),
    Harmonic(Prepared<HarmonicExtension>),
    Hodge(Prepared<HodgeProblem>),
    Heat(Prepared<HeatProblem>),
}

#[pyclass(
    name = "Prepared",
    frozen,
    module = "polygeo.solve",
    skip_from_py_object
)]
pub(crate) struct PyPreparedProblem {
    inner: PreparedProblem,
}

#[pyclass(name = "Workspace", module = "polygeo.solve", skip_from_py_object)]
pub(crate) struct PySolveWorkspace {
    inner: SolveWorkspace,
}

fn core_policy(
    executor: Option<&PyNativeExecutor>,
    storage: Option<&PyStorageLimit>,
    work: Option<&PyWorkLimit>,
) -> CorePolicy {
    CorePolicy::new(
        executor.map_or(NativeExecutor::sequential(), |x| x.inner),
        storage.copied().unwrap_or(PyStorageLimit::DEFAULT).core(),
        work.cloned().unwrap_or(PyWorkLimit::DEFAULT).core(),
    )
}

pub(crate) fn policy(value: Option<&PyPolicy>) -> CorePolicy {
    value.map_or_else(|| core_policy(None, None, None), |value| value.inner)
}

pub(crate) fn cancellation_token(value: Option<&PyCancellationToken>) -> CancellationToken {
    value.map_or_else(CancellationToken::new, |value| value.inner.clone())
}

#[pymethods]
impl PyProblem {
    #[pyo3(signature = (*, policy=None, cancellation=None))]
    fn prepare(
        &self,
        py: Python<'_>,
        policy: Option<&PyPolicy>,
        cancellation: Option<&PyCancellationToken>,
    ) -> PyResult<PyPreparedProblem> {
        let policy = crate::solve::policy(policy);
        let token = cancellation_token(cancellation);
        let problem = self.inner.clone();
        let inner = py
            .detach(move || match problem {
                Problem::MeanZero(x) => x
                    .prepare_cancellable(policy, &token)
                    .map(PreparedProblem::MeanZero),
                Problem::Dirichlet(x) => x
                    .prepare_cancellable(policy, &token)
                    .map(PreparedProblem::Dirichlet),
                Problem::Harmonic(x) => x
                    .prepare_cancellable(policy, &token)
                    .map(PreparedProblem::Harmonic),
                Problem::Hodge(x) => x
                    .prepare_cancellable(policy, &token)
                    .map(PreparedProblem::Hodge),
                Problem::Heat(x) => x
                    .prepare_cancellable(policy, &token)
                    .map(PreparedProblem::Heat),
            })
            .map_err(solve_error)?;
        Ok(PyPreparedProblem { inner })
    }
}

#[pymethods]
impl PyPreparedProblem {
    fn workspace_for(&self, problem: &PyProblem) -> PyResult<PySolveWorkspace> {
        let inner = match (&self.inner, &problem.inner) {
            (PreparedProblem::MeanZero(a), Problem::MeanZero(b)) => a.workspace_for(b),
            (PreparedProblem::Dirichlet(a), Problem::Dirichlet(b)) => a.workspace_for(b),
            (PreparedProblem::Harmonic(a), Problem::Harmonic(b)) => a.workspace_for(b),
            (PreparedProblem::Hodge(a), Problem::Hodge(b)) => a.workspace_for(b),
            (PreparedProblem::Heat(a), Problem::Heat(b)) => a.workspace_for(b),
            _ => return Err(solve_error(SolveError::ProblemMismatch)),
        }
        .map_err(solve_error)?;
        Ok(PySolveWorkspace { inner })
    }

    #[pyo3(signature = (problem, workspace, *, cancellation=None))]
    fn solve(
        &self,
        py: Python<'_>,
        problem: &PyProblem,
        workspace: &mut PySolveWorkspace,
        cancellation: Option<&PyCancellationToken>,
    ) -> PyResult<Py<PyAny>> {
        let token = cancellation_token(cancellation);
        match (&self.inner, &problem.inner) {
            (PreparedProblem::MeanZero(a), Problem::MeanZero(b)) => {
                let value = py
                    .detach(|| a.solve_cancellable(b, &mut workspace.inner, &token))
                    .map_err(solve_error)?;
                Ok(Py::new(py, PyPoissonSolution { inner: value })?.into_any())
            }
            (PreparedProblem::Dirichlet(a), Problem::Dirichlet(b)) => {
                let value = py
                    .detach(|| a.solve_cancellable(b, &mut workspace.inner, &token))
                    .map_err(solve_error)?;
                Ok(Py::new(py, PyDirichletSolution { inner: value })?.into_any())
            }
            (PreparedProblem::Harmonic(a), Problem::Harmonic(b)) => {
                let value = py
                    .detach(|| a.solve_cancellable(b, &mut workspace.inner, &token))
                    .map_err(solve_error)?;
                Ok(Py::new(py, PyDirichletSolution { inner: value })?.into_any())
            }
            (PreparedProblem::Hodge(a), Problem::Hodge(b)) => {
                let value = py
                    .detach(|| a.solve_cancellable(b, &mut workspace.inner, &token))
                    .map_err(solve_error)?;
                Ok(Py::new(py, PyHodgeDecomposition { inner: value })?.into_any())
            }
            (PreparedProblem::Heat(a), Problem::Heat(b)) => {
                let value = py
                    .detach(|| a.solve_cancellable(b, &mut workspace.inner, &token))
                    .map_err(solve_error)?;
                Ok(Py::new(py, PyHeatSolution { inner: value })?.into_any())
            }
            _ => Err(solve_error(SolveError::ProblemMismatch)),
        }
    }
}

#[pyclass(
    name = "PoissonResult",
    frozen,
    module = "polygeo.solve",
    skip_from_py_object
)]
struct PyPoissonSolution {
    inner: PoissonSolution,
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
    name = "DirichletResult",
    frozen,
    module = "polygeo.solve",
    skip_from_py_object
)]
struct PyDirichletSolution {
    inner: DirichletSolution,
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
    module = "polygeo.field",
    skip_from_py_object
)]
struct PyHodgeDecomposition {
    inner: HodgeDecomposition,
}

#[pyclass(
    name = "HarmonicBasis",
    frozen,
    module = "polygeo.field",
    skip_from_py_object
)]
pub(crate) struct PyHarmonicOneFormBasis {
    pub(crate) inner: HarmonicBasis,
}

#[pymethods]
impl PyHarmonicOneFormBasis {
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    #[getter]
    fn forms(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(
            py,
            self.inner
                .forms()
                .iter()
                .cloned()
                .map(|form| PyBinary64Element {
                    inner: Element::Cochain(form),
                }),
        )?
        .unbind())
    }

    #[getter]
    fn maximum_closedness_residual(&self) -> f64 {
        self.inner.maximum_closedness_residual()
    }

    #[getter]
    fn maximum_coclosedness_residual(&self) -> f64 {
        self.inner.maximum_coclosedness_residual()
    }

    #[getter]
    fn maximum_identity_period_residual(&self) -> f64 {
        self.inner.maximum_identity_period_residual()
    }

    #[getter]
    fn residual_limit(&self) -> f64 {
        self.inner.residual_limit()
    }
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

#[pyclass(
    name = "HeatResult",
    frozen,
    module = "polygeo.solve",
    skip_from_py_object
)]
struct PyHeatSolution {
    inner: HeatSolution,
}
#[pymethods]
impl PyHeatSolution {
    #[getter]
    fn value(&self) -> PyBinary64Element {
        PyBinary64Element {
            inner: Element::Cochain(self.inner.value().clone()),
        }
    }
    #[getter]
    fn residual_bound(&self) -> f64 {
        self.inner.residual_bound()
    }
    #[getter]
    fn mass_residual_bound(&self) -> f64 {
        self.inner.mass_residual_bound()
    }
    #[getter]
    fn energy_before(&self) -> f64 {
        self.inner.energy_before()
    }
    #[getter]
    fn energy_after(&self) -> f64 {
        self.inner.energy_after()
    }
    #[getter]
    fn exact_fallback_rows(&self) -> usize {
        self.inner.exact_fallback_rows()
    }
}

#[pyclass(
    name = "FlowStep",
    frozen,
    module = "polygeo.geometry",
    skip_from_py_object
)]
pub(crate) struct PyFlowStep {
    pub(crate) inner: FlowStep,
}
#[pymethods]
impl PyFlowStep {
    #[getter]
    fn geometry(&self) -> NativeEuclideanRealization {
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

pub(crate) fn register_solve(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let problem_error = module.py().get_type::<ProblemErrorPy>();
    problem_error.setattr("__module__", "polygeo.solve")?;
    module.add("ProblemError", problem_error)?;
    let solve_error = module.py().get_type::<SolveErrorPy>();
    solve_error.setattr("__module__", "polygeo.solve")?;
    module.add("SolveError", solve_error)?;
    module.add_class::<PyStorageLimit>()?;
    module.add_class::<PyWorkLimit>()?;
    module.add_class::<PyNativeExecutor>()?;
    module.add_class::<PyPolicy>()?;
    module.add_class::<PyCancellationToken>()?;
    module.add_class::<PyProblem>()?;
    module.add_class::<PyPreparedProblem>()?;
    module.add_class::<PySolveWorkspace>()?;
    module.add_class::<PyPoissonSolution>()?;
    module.add_class::<PyDirichletSolution>()?;
    module.add_class::<PyHeatSolution>()
}

pub(crate) fn register_field(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyHodgeDecomposition>()?;
    module.add_class::<PyHarmonicOneFormBasis>()
}

pub(crate) fn register_geometry(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyFlowStep>()
}
