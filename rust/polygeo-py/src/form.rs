use numpy::PyReadonlyArray1;
use polygeo_core::{
    Binary64Chain, Binary64ChainSpace, Binary64Cochain, Binary64CochainSpace,
    Binary64Element as CoreBinary64Element, Binary64ElementError, Chain, Cochain,
    LinearOperator as CoreLinearOperator, OperatorError, Variance,
};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule, PyType};

use crate::{
    ExactElement, NativeChainElement, classified_exception, filled_array_1d, filled_array_2d,
};

create_exception!(_polygeo_native, Binary64ElementErrorPy, PyValueError);
create_exception!(_polygeo_native, OperatorErrorPy, PyValueError);

fn classified(reason: &'static str, message: String, operator: bool) -> PyErr {
    Python::attach(|py| {
        let error = if operator {
            OperatorErrorPy::new_err(message)
        } else {
            Binary64ElementErrorPy::new_err(message)
        };
        classified_exception(py, error, reason, PyDict::new(py).unbind())
    })
}

pub(crate) fn element_error(error: Binary64ElementError) -> PyErr {
    classified(error.reason(), error.to_string(), false)
}

pub(crate) fn operator_error(error: OperatorError) -> PyErr {
    classified(error.reason(), error.to_string(), true)
}

#[derive(Clone, Debug)]
pub(crate) enum Space {
    Chain(Binary64ChainSpace),
    Cochain(Binary64CochainSpace),
}

impl Space {
    fn degree(&self) -> isize {
        match self {
            Self::Chain(x) => x.degree(),
            Self::Cochain(x) => x.degree(),
        }
    }
    fn size(&self) -> usize {
        match self {
            Self::Chain(x) => x.size(),
            Self::Cochain(x) => x.size(),
        }
    }
    const fn variance(&self) -> &'static str {
        match self {
            Self::Chain(_) => "chain",
            Self::Cochain(_) => "cochain",
        }
    }
    const fn is_full(&self) -> bool {
        match self {
            Self::Chain(x) => x.is_full(),
            Self::Cochain(x) => x.is_full(),
        }
    }
    fn same_space(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Chain(a), Self::Chain(b)) => a.same_space(b),
            (Self::Cochain(a), Self::Cochain(b)) => a.same_space(b),
            _ => false,
        }
    }
}

#[pyclass(
    name = "Binary64Space",
    frozen,
    module = "polygeo",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyBinary64Space {
    pub(crate) inner: Space,
}

#[pymethods]
impl PyBinary64Space {
    #[classmethod]
    fn __class_getitem__<'py>(
        class: &Bound<'py, PyType>,
        _parameter: &Bound<'_, PyAny>,
    ) -> Bound<'py, PyType> {
        class.clone()
    }

    #[getter]
    fn degree(&self) -> isize {
        self.inner.degree()
    }
    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }
    #[getter]
    fn variance(&self) -> &'static str {
        self.inner.variance()
    }
    #[getter]
    fn is_full(&self) -> bool {
        self.inner.is_full()
    }
    fn same_space(&self, other: &Self) -> bool {
        self.inner.same_space(&other.inner)
    }

    fn admit_numpy(&self, coefficients: &Bound<'_, PyAny>) -> PyResult<PyBinary64Element> {
        let coefficients = coefficients.extract::<PyReadonlyArray1<'_, f64>>()?;
        let values = coefficients.as_array().iter().copied().collect();
        let inner = match &self.inner {
            Space::Chain(space) => {
                Element::Chain(Binary64Chain::admit(space.clone(), values).map_err(element_error)?)
            }
            Space::Cochain(space) => Element::Cochain(
                Binary64Cochain::admit(space.clone(), values).map_err(element_error)?,
            ),
        };
        Ok(PyBinary64Element { inner })
    }

    fn realize_integral(&self, exact: &NativeChainElement) -> PyResult<PyBinary64Element> {
        let inner = match (&self.inner, &exact.inner) {
            (Space::Chain(space), ExactElement::IntegerChain(value)) => Element::Chain(
                Binary64Chain::realize_integral(space.clone(), value).map_err(element_error)?,
            ),
            (Space::Cochain(space), ExactElement::IntegerCochain(value)) => Element::Cochain(
                Binary64Cochain::realize_integral(space.clone(), value).map_err(element_error)?,
            ),
            _ => {
                return Err(classified(
                    "space_mismatch",
                    "exact element has a different variance or basis".into(),
                    false,
                ));
            }
        };
        Ok(PyBinary64Element { inner })
    }

    fn identity(&self) -> PyLinearOperator {
        let inner = match &self.inner {
            Space::Chain(x) => Operator::ChainChain(x.identity()),
            Space::Cochain(x) => Operator::CochainCochain(x.identity()),
        };
        PyLinearOperator { inner }
    }

    fn zero_to(&self, target: &Self) -> PyLinearOperator {
        let inner = match (&self.inner, &target.inner) {
            (Space::Chain(a), Space::Chain(b)) => Operator::ChainChain(a.zero_to(b)),
            (Space::Chain(a), Space::Cochain(b)) => Operator::ChainCochain(a.zero_to(b)),
            (Space::Cochain(a), Space::Chain(b)) => Operator::CochainChain(a.zero_to(b)),
            (Space::Cochain(a), Space::Cochain(b)) => Operator::CochainCochain(a.zero_to(b)),
        };
        PyLinearOperator { inner }
    }

    fn restriction(&self) -> PyResult<PyLinearOperator> {
        let inner = match &self.inner {
            Space::Chain(space) => Operator::ChainChain(
                space
                    .canonical_selection()
                    .ok_or_else(|| operator_error(OperatorError::SpaceMismatch))?
                    .restriction()
                    .map_err(operator_error)?,
            ),
            Space::Cochain(space) => Operator::CochainCochain(
                space
                    .canonical_selection()
                    .ok_or_else(|| operator_error(OperatorError::SpaceMismatch))?
                    .restriction()
                    .map_err(operator_error)?,
            ),
        };
        Ok(PyLinearOperator { inner })
    }

    fn extension_by_zero(&self) -> PyResult<PyLinearOperator> {
        let inner = match &self.inner {
            Space::Chain(space) => Operator::ChainChain(
                space
                    .canonical_selection()
                    .ok_or_else(|| operator_error(OperatorError::SpaceMismatch))?
                    .extension_by_zero()
                    .map_err(operator_error)?,
            ),
            Space::Cochain(space) => Operator::CochainCochain(
                space
                    .canonical_selection()
                    .ok_or_else(|| operator_error(OperatorError::SpaceMismatch))?
                    .extension_by_zero()
                    .map_err(operator_error)?,
            ),
        };
        Ok(PyLinearOperator { inner })
    }

    fn exterior_derivative(&self) -> PyResult<PyLinearOperator> {
        let Space::Cochain(space) = &self.inner else {
            return Err(operator_error(OperatorError::SpaceMismatch));
        };
        Ok(PyLinearOperator {
            inner: Operator::CochainCochain(space.exterior_derivative().map_err(operator_error)?),
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Element {
    Chain(Binary64Chain),
    Cochain(Binary64Cochain),
}

#[pyclass(
    name = "Binary64Element",
    frozen,
    module = "polygeo",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyBinary64Element {
    pub(crate) inner: Element,
}

#[pymethods]
impl PyBinary64Element {
    #[classmethod]
    fn __class_getitem__<'py>(
        class: &Bound<'py, PyType>,
        _parameter: &Bound<'_, PyAny>,
    ) -> Bound<'py, PyType> {
        class.clone()
    }
    #[getter]
    fn space(&self) -> PyBinary64Space {
        PyBinary64Space {
            inner: match &self.inner {
                Element::Chain(x) => Space::Chain(x.space().clone()),
                Element::Cochain(x) => Space::Cochain(x.space().clone()),
            },
        }
    }
    fn coefficients_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let values = match &self.inner {
            Element::Chain(x) => x.coefficients(),
            Element::Cochain(x) => x.coefficients(),
        };
        filled_array_1d(py, values.len(), |output| {
            output.copy_from_slice(values);
            Ok(())
        })
    }

    fn __neg__(&self) -> Self {
        Self {
            inner: match &self.inner {
                Element::Chain(value) => Element::Chain(value.negated()),
                Element::Cochain(value) => Element::Cochain(value.negated()),
            },
        }
    }

    fn apply(&self, operator: &PyLinearOperator) -> PyResult<Self> {
        operator.apply(self)
    }

    fn wedge(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let (Element::Cochain(left), Element::Cochain(right)) = (&self.inner, &other.inner) else {
            return Err(element_error(Binary64ElementError::SpaceMismatch));
        };
        let left = left.clone();
        let right = right.clone();
        let inner = py
            .detach(move || left.wedge(&right))
            .map_err(element_error)?;
        Ok(Self {
            inner: Element::Cochain(inner),
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Operator {
    ChainChain(CoreLinearOperator<Chain, Chain>),
    ChainCochain(CoreLinearOperator<Chain, Cochain>),
    CochainChain(CoreLinearOperator<Cochain, Chain>),
    CochainCochain(CoreLinearOperator<Cochain, Cochain>),
}

impl Operator {
    fn source(&self) -> Space {
        match self {
            Self::ChainChain(x) => Space::Chain(x.source().clone()),
            Self::ChainCochain(x) => Space::Chain(x.source().clone()),
            Self::CochainChain(x) => Space::Cochain(x.source().clone()),
            Self::CochainCochain(x) => Space::Cochain(x.source().clone()),
        }
    }
    fn target(&self) -> Space {
        match self {
            Self::ChainChain(x) => Space::Chain(x.target().clone()),
            Self::ChainCochain(x) => Space::Cochain(x.target().clone()),
            Self::CochainChain(x) => Space::Chain(x.target().clone()),
            Self::CochainCochain(x) => Space::Cochain(x.target().clone()),
        }
    }
}

#[pyclass(
    name = "LinearOperator",
    frozen,
    module = "polygeo",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyLinearOperator {
    pub(crate) inner: Operator,
}

#[pymethods]
impl PyLinearOperator {
    #[classmethod]
    fn __class_getitem__<'py>(
        class: &Bound<'py, PyType>,
        _parameter: &Bound<'_, PyAny>,
    ) -> Bound<'py, PyType> {
        class.clone()
    }
    #[getter]
    fn source(&self) -> PyBinary64Space {
        PyBinary64Space {
            inner: self.inner.source(),
        }
    }
    #[getter]
    fn target(&self) -> PyBinary64Space {
        PyBinary64Space {
            inner: self.inner.target(),
        }
    }
    fn same_identity(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (Operator::ChainChain(a), Operator::ChainChain(b)) => a.same_identity(b),
            (Operator::ChainCochain(a), Operator::ChainCochain(b)) => a.same_identity(b),
            (Operator::CochainChain(a), Operator::CochainChain(b)) => a.same_identity(b),
            (Operator::CochainCochain(a), Operator::CochainCochain(b)) => a.same_identity(b),
            _ => false,
        }
    }
    #[getter]
    fn execution_steps(&self) -> usize {
        match &self.inner {
            Operator::ChainChain(x) => x.execution_steps(),
            Operator::ChainCochain(x) => x.execution_steps(),
            Operator::CochainChain(x) => x.execution_steps(),
            Operator::CochainCochain(x) => x.execution_steps(),
        }
    }
    fn compose(&self, before: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &before.inner) {
            (Operator::ChainChain(a), Operator::ChainChain(b)) => {
                Operator::ChainChain(polygeo_core::operator::compose(a, b).map_err(operator_error)?)
            }
            (Operator::ChainChain(a), Operator::CochainChain(b)) => Operator::CochainChain(
                polygeo_core::operator::compose(a, b).map_err(operator_error)?,
            ),
            (Operator::ChainCochain(a), Operator::ChainChain(b)) => Operator::ChainCochain(
                polygeo_core::operator::compose(a, b).map_err(operator_error)?,
            ),
            (Operator::ChainCochain(a), Operator::CochainChain(b)) => Operator::CochainCochain(
                polygeo_core::operator::compose(a, b).map_err(operator_error)?,
            ),
            (Operator::CochainChain(a), Operator::ChainCochain(b)) => {
                Operator::ChainChain(polygeo_core::operator::compose(a, b).map_err(operator_error)?)
            }
            (Operator::CochainChain(a), Operator::CochainCochain(b)) => Operator::CochainChain(
                polygeo_core::operator::compose(a, b).map_err(operator_error)?,
            ),
            (Operator::CochainCochain(a), Operator::ChainCochain(b)) => Operator::ChainCochain(
                polygeo_core::operator::compose(a, b).map_err(operator_error)?,
            ),
            (Operator::CochainCochain(a), Operator::CochainCochain(b)) => Operator::CochainCochain(
                polygeo_core::operator::compose(a, b).map_err(operator_error)?,
            ),
            _ => return Err(operator_error(OperatorError::SpaceMismatch)),
        };
        Ok(Self { inner })
    }
    fn apply(&self, value: &PyBinary64Element) -> PyResult<PyBinary64Element> {
        let inner = match (&self.inner, &value.inner) {
            (Operator::ChainChain(op), Element::Chain(x)) => {
                Element::Chain(op.apply(x).map_err(operator_error)?)
            }
            (Operator::ChainCochain(op), Element::Chain(x)) => {
                Element::Cochain(op.apply(x).map_err(operator_error)?)
            }
            (Operator::CochainChain(op), Element::Cochain(x)) => {
                Element::Chain(op.apply(x).map_err(operator_error)?)
            }
            (Operator::CochainCochain(op), Element::Cochain(x)) => {
                Element::Cochain(op.apply(x).map_err(operator_error)?)
            }
            _ => return Err(operator_error(OperatorError::SpaceMismatch)),
        };
        Ok(PyBinary64Element { inner })
    }

    fn to_scipy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let values = match &self.inner {
            Operator::ChainChain(x) => dense_columns(x),
            Operator::ChainCochain(x) => dense_columns(x),
            Operator::CochainChain(x) => dense_columns(x),
            Operator::CochainCochain(x) => dense_columns(x),
        }
        .map_err(operator_error)?;
        let rows = self.inner.target().size();
        let columns = self.inner.source().size();
        let dense = filled_array_2d(py, rows, columns, |output| {
            output.copy_from_slice(&values);
            Ok(())
        })?;
        let sparse = PyModule::import(py, "scipy.sparse")?;
        Ok(sparse.getattr("csr_array")?.call1((dense,))?.unbind())
    }

    fn dirichlet(
        &self,
        rhs: &PyBinary64Element,
        prescribed: &PyBinary64Element,
    ) -> PyResult<crate::solve::PyProblem> {
        let (
            Operator::CochainCochain(operator),
            Element::Cochain(rhs),
            Element::Cochain(prescribed),
        ) = (&self.inner, &rhs.inner, &prescribed.inner)
        else {
            return Err(crate::solve::problem_error(
                polygeo_core::ProblemError::SpaceMismatch,
            ));
        };
        Ok(crate::solve::PyProblem {
            inner: crate::solve::Problem::Dirichlet(
                operator
                    .dirichlet(rhs.clone(), prescribed.clone())
                    .map_err(crate::solve::problem_error)?,
            ),
        })
    }
}

fn dense_columns<S: Variance, T: Variance>(
    operator: &CoreLinearOperator<S, T>,
) -> Result<Vec<f64>, OperatorError> {
    let rows = operator.target().size();
    let columns = operator.source().size();
    let mut dense = vec![0.0; rows.saturating_mul(columns)];
    for column in 0..columns {
        let mut unit = vec![0.0; columns];
        unit[column] = 1.0;
        let value = CoreBinary64Element::admit(operator.source().clone(), unit)
            .map_err(|_| OperatorError::SpaceMismatch)?;
        let output = operator.apply(&value)?;
        for (row, &coefficient) in output.coefficients().iter().enumerate() {
            dense[row * columns + column] = coefficient;
        }
    }
    Ok(dense)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "Binary64ElementError",
        module.py().get_type::<Binary64ElementErrorPy>(),
    )?;
    module.add("OperatorError", module.py().get_type::<OperatorErrorPy>())?;
    module.add_class::<PyBinary64Space>()?;
    module.add_class::<PyBinary64Element>()?;
    module.add_class::<PyLinearOperator>()?;
    let space = module.getattr("Binary64Space")?;
    module.add("Binary64ChainSpace", space.clone())?;
    module.add("Binary64CochainSpace", space)?;
    let element = module.getattr("Binary64Element")?;
    module.add("Binary64Chain", element.clone())?;
    module.add("Binary64Cochain", element)
}
