use num_bigint::BigInt;
use numpy::{PyArray1, PyArrayMethods};
use polygeo_core::chain::{
    BigIntEncoding, Chain, ChainComplex, ChainError as CoreChainError,
    ChainIsomorphism as CoreChainIsomorphism, ChainLawLimit, Cochain, CompositionError,
    Csr as CsrRepresentation, CsrBuildLimit, CsrError as RepresentationError, CsrEstimate,
    Element as ExactCoreElement, ExactRational, IntegerRing, LinearMap, RationalField,
    ReducedFractionEncoding, Space, Variance, compose,
};
use polygeo_core::solve::{StorageLimit, WorkLimit};
use polygeo_core::topology::{TopologyDetailValue, TopologyError};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyInt, PyModule, PyTuple, PyType};

use crate::array::{fill_indices, filled_array_1d};
use crate::classified_exception;
use crate::halfedge::halfedge_degree;
use crate::topology::{halfedge_topology_error, transport_error};

type PyBoundaryParts = (Py<PyAny>, Py<PyAny>, Py<PyAny>, (usize, usize));

create_exception!(
    _polygeo_native,
    ChainError,
    PyValueError,
    "Classified exact chain-algebra failure."
);

#[pyclass(name = "IntegerRing", frozen, module = "polygeo.chain")]
struct PyIntegerRing;

#[pyclass(name = "RationalField", frozen, module = "polygeo.chain")]
struct PyRationalField;

#[pyclass(name = "Chain", frozen, module = "polygeo.chain")]
struct PyChainVariance;

#[pyclass(name = "Cochain", frozen, module = "polygeo.chain")]
struct PyCochainVariance;

#[pyclass(name = "BigIntEncoding", frozen, module = "polygeo.chain")]
struct PyBigIntEncoding;

#[pyclass(name = "ReducedFractionEncoding", frozen, module = "polygeo.chain")]
struct PyReducedFractionEncoding;

fn exact_error(reason: &'static str, message: &'static str) -> PyErr {
    Python::attach(|py| chain_exception(py, reason, message, PyDict::new(py).unbind()))
}

fn chain_exception(
    py: Python<'_>,
    reason: &'static str,
    message: impl Into<String>,
    details: Py<PyDict>,
) -> PyErr {
    let message = message.into();
    let error = ChainError::new_err((reason, message, details.clone_ref(py)));
    classified_exception(py, error, reason, details)
}

fn chain_error(error: CoreChainError) -> PyErr {
    Python::attach(|py| {
        let details = PyDict::new(py);
        let translation_failed = match error {
            CoreChainError::BasisIndexOutside { index, bound } => {
                details.set_item("index", index).is_err()
                    || details.set_item("bound", bound).is_err()
            }
            CoreChainError::SpaceMismatch
            | CoreChainError::NotSimplicial
            | CoreChainError::CoefficientFieldRequired
            | CoreChainError::NormalizationNotInvertible
            | CoreChainError::Topology(_) => false,
        };
        if translation_failed {
            return exact_error("translation", "failed to translate chain failure");
        }
        chain_exception(py, error.reason(), error.to_string(), details.unbind())
    })
}

fn composition_error(error: CompositionError) -> PyErr {
    exact_error(error.reason(), "exact map composition was rejected")
}

fn chain_topology_error(error: TopologyError) -> PyErr {
    Python::attach(|py| {
        let details = PyDict::new(py);
        for field in error.details().fields() {
            let result = match field.value() {
                TopologyDetailValue::Signed(value) => details.set_item(field.name(), value),
                TopologyDetailValue::Unsigned(value) => details.set_item(field.name(), value),
                TopologyDetailValue::Index(value) => details.set_item(field.name(), value),
                TopologyDetailValue::Text(value) => details.set_item(field.name(), value),
                _ => continue,
            };
            if result.is_err() {
                return exact_error("translation", "failed to translate topology failure");
            }
        }
        chain_exception(py, error.reason(), error.to_string(), details.unbind())
    })
}

fn representation_error(error: RepresentationError) -> PyErr {
    Python::attach(|py| {
        let details = PyDict::new(py);
        if let Some((axis, required, limit)) = error.resource_limit() {
            let translated = details
                .set_item("axis", axis)
                .and_then(|()| details.set_item("required", required))
                .and_then(|()| details.set_item("limit", limit))
                .and_then(|()| {
                    details.set_item(
                        "phase",
                        error
                            .resource_phase()
                            .expect("resource details always carry a phase"),
                    )
                });
            if translated.is_err() {
                return exact_error("translation", "failed to translate representation failure");
            }
        }
        chain_exception(py, error.reason(), error.to_string(), details.unbind())
    })
}

#[pyclass(
    name = "ChainLawLimit",
    frozen,
    module = "polygeo.chain",
    skip_from_py_object
)]
#[derive(Clone, Copy)]
pub(crate) struct PyChainLawLimit {
    retained_logical_bytes: u64,
    peak_live_logical_bytes: u64,
    terms: u64,
}

impl PyChainLawLimit {
    pub(crate) const DEFAULT: Self = Self {
        retained_logical_bytes: 128 * 1024 * 1024,
        peak_live_logical_bytes: 512 * 1024 * 1024,
        terms: 100_000_000,
    };

    pub(crate) fn core(self) -> ChainLawLimit {
        let storage = StorageLimit::new(self.retained_logical_bytes, self.peak_live_logical_bytes)
            .expect("Python construction preserves storage lifecycle");
        ChainLawLimit::new(storage, WorkLimit::new(self.terms))
    }
}

#[pymethods]
impl PyChainLawLimit {
    #[new]
    #[pyo3(signature = (*, retained_logical_bytes=134_217_728, peak_live_logical_bytes=536_870_912, terms=100_000_000))]
    fn new(
        retained_logical_bytes: u64,
        peak_live_logical_bytes: u64,
        terms: u64,
    ) -> PyResult<Self> {
        StorageLimit::new(retained_logical_bytes, peak_live_logical_bytes).ok_or_else(|| {
            transport_error(
                "limit",
                "peak_live_logical_bytes must contain retained_logical_bytes",
            )
        })?;
        Ok(Self {
            retained_logical_bytes,
            peak_live_logical_bytes,
            terms,
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
    const fn terms(&self) -> u64 {
        self.terms
    }
}

#[pyclass(
    name = "CsrEstimate",
    frozen,
    module = "polygeo.chain",
    skip_from_py_object
)]
#[derive(Clone, Copy)]
struct PyCsrEstimate {
    inner: CsrEstimate,
}

#[pymethods]
impl PyCsrEstimate {
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn nnz_bound(&self) -> usize {
        self.inner.nnz_bound()
    }

    #[getter]
    fn coefficient_bits_bound(&self) -> u64 {
        self.inner.coefficient_bits_bound()
    }

    #[getter]
    fn retained_logical_bytes_bound(&self) -> u64 {
        self.inner.retained_logical_bytes_bound()
    }

    #[getter]
    fn peak_live_logical_bytes_bound(&self) -> u64 {
        self.inner.peak_live_logical_bytes_bound()
    }

    #[getter]
    fn scratch_entries_bound(&self) -> usize {
        self.inner.scratch_entries_bound()
    }

    #[getter]
    fn scalar_steps_bound(&self) -> u64 {
        self.inner.scalar_steps_bound()
    }

    #[getter]
    fn canonicalization_required(&self) -> bool {
        self.inner.canonicalization_required()
    }

    fn as_limit(&self) -> PyCsrBuildLimit {
        PyCsrBuildLimit::for_estimate(self.inner)
    }
}

#[pyclass(
    name = "CsrBuildLimit",
    frozen,
    module = "polygeo.chain",
    skip_from_py_object
)]
#[derive(Clone, Copy)]
struct PyCsrBuildLimit {
    retained_logical_bytes: u64,
    peak_live_logical_bytes: u64,
    coefficient_bits: u64,
    scalar_steps: u64,
}

impl PyCsrBuildLimit {
    const fn for_estimate(estimate: CsrEstimate) -> Self {
        Self {
            retained_logical_bytes: estimate.retained_logical_bytes_bound(),
            peak_live_logical_bytes: estimate.peak_live_logical_bytes_bound(),
            coefficient_bits: estimate.coefficient_bits_bound(),
            scalar_steps: estimate.scalar_steps_bound(),
        }
    }
}

#[pymethods]
impl PyCsrBuildLimit {
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
    const fn scalar_steps(&self) -> u64 {
        self.scalar_steps
    }

    #[pyo3(signature = (*, retained_logical_bytes=None, peak_live_logical_bytes=None, coefficient_bits=None, scalar_steps=None))]
    fn replace(
        &self,
        retained_logical_bytes: Option<u64>,
        peak_live_logical_bytes: Option<u64>,
        coefficient_bits: Option<u64>,
        scalar_steps: Option<u64>,
    ) -> PyResult<Self> {
        let changed = Self {
            retained_logical_bytes: retained_logical_bytes.unwrap_or(self.retained_logical_bytes),
            peak_live_logical_bytes: peak_live_logical_bytes
                .unwrap_or(self.peak_live_logical_bytes),
            coefficient_bits: coefficient_bits.unwrap_or(self.coefficient_bits),
            scalar_steps: scalar_steps.unwrap_or(self.scalar_steps),
        };
        if StorageLimit::new(
            changed.retained_logical_bytes,
            changed.peak_live_logical_bytes,
        )
        .is_none()
        {
            return Err(exact_error(
                "limit",
                "peak_live_logical_bytes must contain retained_logical_bytes",
            ));
        }
        Ok(changed)
    }
}

fn admitted_build_limit(estimate: CsrEstimate, limit: PyCsrBuildLimit) -> CsrBuildLimit {
    let storage = StorageLimit::new(limit.retained_logical_bytes, limit.peak_live_logical_bytes)
        .expect("Python limit construction preserves the storage lifecycle");
    CsrBuildLimit::for_estimate(estimate)
        .with_storage(storage)
        .with_coefficient_bits(limit.coefficient_bits)
        .with_scalar_steps(WorkLimit::new(limit.scalar_steps))
}

fn filled_exact_i64(
    py: Python<'_>,
    length: usize,
    fill: impl FnOnce(&mut [i64]) -> PyResult<()>,
) -> PyResult<Py<PyAny>> {
    let array = PyArray1::<i64>::zeros(py, length, false);
    let mut writable = array
        .try_readwrite()
        .map_err(|_| exact_error("projection", "failed to acquire owned projection storage"))?;
    let output = writable
        .as_slice_mut()
        .map_err(|_| exact_error("projection", "owned projection storage is not contiguous"))?;
    fill(output)?;
    drop(writable);
    Ok(array.unbind().into_any())
}

fn checked_bigint_i64(value: &BigInt) -> Option<i64> {
    i64::try_from(value).ok()
}

fn fill_exact_indices(values: &[usize], output: &mut [i64]) -> PyResult<()> {
    for (target, value) in output.iter_mut().zip(values) {
        *target = i64::try_from(*value).map_err(|_| {
            exact_error(
                "index_overflow",
                "an exact CSR index is outside the requested int64 projection",
            )
        })?;
    }
    Ok(())
}

#[pyclass(name = "ChainIsomorphism", frozen, module = "polygeo.chain")]
pub(crate) struct PyChainIsomorphism {
    pub(crate) relation: CoreChainIsomorphism<IntegerRing>,
}

#[pymethods]
impl PyChainIsomorphism {
    #[getter]
    fn source(&self) -> NativeChainComplex {
        NativeChainComplex {
            inner: ExactComplex::Integer(self.relation.source().clone()),
        }
    }

    #[getter]
    fn target(&self) -> NativeChainComplex {
        NativeChainComplex {
            inner: ExactComplex::Integer(self.relation.target().clone()),
        }
    }

    fn forward(&self, degree: usize) -> PyResult<NativeLinearMap> {
        self.relation
            .forward(degree)
            .map(|map| NativeLinearMap {
                inner: ExactMap::IntegerChain(map),
            })
            .map_err(chain_topology_error)
    }

    fn inverse(&self, degree: usize) -> PyResult<NativeLinearMap> {
        self.relation
            .inverse(degree)
            .map(|map| NativeLinearMap {
                inner: ExactMap::IntegerChain(map),
            })
            .map_err(chain_topology_error)
    }

    fn signed_permutation_numpy_copy(
        &self,
        py: Python<'_>,
        degree: isize,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (targets, signs) = self
            .relation
            .signed_permutation(halfedge_degree(degree)?)
            .map_err(halfedge_topology_error)?;
        let target_copy = filled_array_1d(py, targets.len(), |output| {
            fill_indices::<i64>(targets.iter().copied(), output)
        })?;
        let sign_copy = filled_array_1d(py, signs.len(), |output| {
            output.copy_from_slice(signs);
            Ok(())
        })?;
        Ok((target_copy, sign_copy))
    }
}

#[derive(Clone)]
pub(crate) enum ExactComplex {
    Integer(ChainComplex<IntegerRing>),
    Rational(ChainComplex<RationalField>),
}

#[pyclass(name = "ChainComplex", frozen, module = "polygeo.chain")]
pub(crate) struct NativeChainComplex {
    pub(crate) inner: ExactComplex,
}

impl NativeChainComplex {
    fn chain_space(&self, degree: usize) -> PyResult<NativeChainSpace> {
        let inner = match &self.inner {
            ExactComplex::Integer(value) => {
                ExactSpace::IntegerChain(value.space(degree).map_err(chain_topology_error)?)
            }
            ExactComplex::Rational(value) => {
                ExactSpace::RationalChain(value.space(degree).map_err(chain_topology_error)?)
            }
        };
        Ok(NativeChainSpace { inner })
    }

    fn cochain_space(&self, degree: usize) -> PyResult<NativeChainSpace> {
        let inner = match &self.inner {
            ExactComplex::Integer(value) => ExactSpace::IntegerCochain(
                value.dual().space(degree).map_err(chain_topology_error)?,
            ),
            ExactComplex::Rational(value) => ExactSpace::RationalCochain(
                value.dual().space(degree).map_err(chain_topology_error)?,
            ),
        };
        Ok(NativeChainSpace { inner })
    }

    fn coboundary_map(&self, degree: usize) -> PyResult<NativeLinearMap> {
        let inner = match &self.inner {
            ExactComplex::Integer(value) => ExactMap::IntegerCochain(
                value
                    .dual()
                    .coboundary(degree)
                    .map_err(chain_topology_error)?,
            ),
            ExactComplex::Rational(value) => ExactMap::RationalCochain(
                value
                    .dual()
                    .coboundary(degree)
                    .map_err(chain_topology_error)?,
            ),
        };
        Ok(NativeLinearMap { inner })
    }

    fn over_q(&self) -> PyResult<Self> {
        match &self.inner {
            ExactComplex::Integer(value) => Ok(Self {
                inner: ExactComplex::Rational(value.over(RationalField::new(IntegerRing))),
            }),
            ExactComplex::Rational(_) => Err(exact_error(
                "coefficient_relation",
                "the complex is already over Q",
            )),
        }
    }
}

#[pymethods]
impl NativeChainComplex {
    #[getter]
    fn coefficient_system(&self) -> &'static str {
        match self.inner {
            ExactComplex::Integer(_) => "Z",
            ExactComplex::Rational(_) => "Q",
        }
    }

    #[getter]
    fn dimension(&self) -> usize {
        match &self.inner {
            ExactComplex::Integer(value) => value.dimension(),
            ExactComplex::Rational(value) => value.dimension(),
        }
    }

    fn __getitem__(&self, degree: usize) -> PyResult<NativeChainSpace> {
        self.chain_space(degree)
    }

    fn dual(&self) -> NativeCochainComplex {
        NativeCochainComplex {
            inner: self.inner.clone(),
        }
    }

    fn boundary(&self, degree: usize) -> PyResult<NativeLinearMap> {
        let inner = match &self.inner {
            ExactComplex::Integer(value) => {
                ExactMap::IntegerChain(value.boundary(degree).map_err(chain_topology_error)?)
            }
            ExactComplex::Rational(value) => {
                ExactMap::RationalChain(value.boundary(degree).map_err(chain_topology_error)?)
            }
        };
        Ok(NativeLinearMap { inner })
    }

    fn over(&self, _coefficients: &PyRationalField) -> PyResult<Self> {
        self.over_q()
    }

    fn same_owner(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (ExactComplex::Integer(left), ExactComplex::Integer(right)) => left.same_owner(right),
            (ExactComplex::Rational(left), ExactComplex::Rational(right)) => left.same_owner(right),
            _ => false,
        }
    }
}

#[pyclass(name = "CochainComplex", frozen, module = "polygeo.chain")]
struct NativeCochainComplex {
    inner: ExactComplex,
}

#[pymethods]
impl NativeCochainComplex {
    fn __getitem__(&self, degree: usize) -> PyResult<NativeChainSpace> {
        NativeChainComplex {
            inner: self.inner.clone(),
        }
        .cochain_space(degree)
    }

    fn coboundary(&self, degree: usize) -> PyResult<NativeLinearMap> {
        NativeChainComplex {
            inner: self.inner.clone(),
        }
        .coboundary_map(degree)
    }
}

#[derive(Clone)]
enum ExactSpace {
    IntegerChain(Space<IntegerRing, Chain>),
    IntegerCochain(Space<IntegerRing, Cochain>),
    RationalChain(Space<RationalField, Chain>),
    RationalCochain(Space<RationalField, Cochain>),
}

#[pyclass(name = "Space", frozen, module = "polygeo.chain")]
struct NativeChainSpace {
    inner: ExactSpace,
}

fn coordinate_items<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    value.call_method0("items").or_else(|_| Ok(value.clone()))
}

fn coordinate_pair<'py>(item: &Bound<'py, PyAny>) -> PyResult<(usize, Bound<'py, PyAny>)> {
    let pair = item.cast::<PyTuple>().map_err(|_| {
        exact_error(
            "coordinate_shape",
            "coordinates must contain index-value pairs",
        )
    })?;
    if pair.len() != 2 {
        return Err(exact_error(
            "coordinate_shape",
            "coordinates must contain index-value pairs",
        ));
    }
    let index = pair.get_item(0)?.extract::<usize>().map_err(|_| {
        exact_error(
            "coordinate_shape",
            "coordinate indices must be nonnegative integers",
        )
    })?;
    Ok((index, pair.get_item(1)?))
}

fn bigint_from_python(value: &Bound<'_, PyAny>) -> PyResult<BigInt> {
    let integer = value.cast::<PyInt>()?;
    let byte_count = integer
        .call_method0("bit_length")?
        .extract::<usize>()?
        .checked_add(8)
        .ok_or_else(|| PyValueError::new_err("integer is too large to transfer"))?
        / 8;
    let kwargs = PyDict::new(value.py());
    kwargs.set_item("signed", true)?;
    let bytes = integer.call_method("to_bytes", (byte_count, "little"), Some(&kwargs))?;
    Ok(BigInt::from_signed_bytes_le(
        bytes.cast::<PyBytes>()?.as_bytes(),
    ))
}

fn bigint_to_python<'py>(py: Python<'py>, value: &BigInt) -> PyResult<Bound<'py, PyAny>> {
    let bytes = value.to_signed_bytes_le();
    let kwargs = PyDict::new(py);
    kwargs.set_item("signed", true)?;
    py.get_type::<PyInt>().call_method(
        "from_bytes",
        (PyBytes::new(py, &bytes), "little"),
        Some(&kwargs),
    )
}

pub(crate) fn bigint_tuple<'py, 'a>(
    py: Python<'py>,
    values: impl IntoIterator<Item = &'a BigInt>,
) -> PyResult<Py<PyTuple>> {
    let values = values
        .into_iter()
        .map(|value| bigint_to_python(py, value))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, values)?.unbind())
}

fn rational_to_python<'py>(
    py: Python<'py>,
    constructor: &Bound<'py, PyAny>,
    value: &ExactRational,
) -> PyResult<Bound<'py, PyAny>> {
    constructor.call1((
        bigint_to_python(py, value.numer())?,
        bigint_to_python(py, value.denom())?,
    ))
}

fn rational_tuple(py: Python<'_>, values: &[ExactRational]) -> PyResult<Py<PyTuple>> {
    let constructor = PyModule::import(py, "fractions")?.getattr("Fraction")?;
    let values = values
        .iter()
        .map(|value| rational_to_python(py, &constructor, value))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, values)?.unbind())
}

fn integer_coordinates(value: &Bound<'_, PyAny>) -> PyResult<Vec<(usize, BigInt)>> {
    let iterable = coordinate_items(value)?;
    let iterator = iterable.try_iter().map_err(|_| {
        exact_error(
            "coordinate_shape",
            "coordinates must be a mapping or iterable",
        )
    })?;
    let mut output = Vec::new();
    for item in iterator {
        let (index, coefficient) = coordinate_pair(&item?)?;
        let coefficient = bigint_from_python(&coefficient).map_err(|_| {
            exact_error(
                "coefficient_value",
                "Z coefficients must be Python integers",
            )
        })?;
        output.push((index, coefficient));
    }
    Ok(output)
}

fn rational_coordinates(value: &Bound<'_, PyAny>) -> PyResult<Vec<(usize, ExactRational)>> {
    let iterable = coordinate_items(value)?;
    let iterator = iterable.try_iter().map_err(|_| {
        exact_error(
            "coordinate_shape",
            "coordinates must be a mapping or iterable",
        )
    })?;
    let mut output = Vec::new();
    for item in iterator {
        let (index, coefficient) = coordinate_pair(&item?)?;
        let numerator = coefficient
            .getattr("numerator")
            .and_then(|value| bigint_from_python(&value))
            .map_err(|_| {
                exact_error(
                    "coefficient_value",
                    "Q coefficients must provide exact numerator and denominator integers",
                )
            })?;
        let denominator = coefficient
            .getattr("denominator")
            .and_then(|value| bigint_from_python(&value))
            .map_err(|_| {
                exact_error(
                    "coefficient_value",
                    "Q coefficients must provide exact numerator and denominator integers",
                )
            })?;
        if denominator == BigInt::from(0_u8) {
            return Err(exact_error(
                "coefficient_value",
                "Q coefficients must have a nonzero denominator",
            ));
        }
        let coefficient = ExactRational::new(numerator, denominator);
        output.push((index, coefficient));
    }
    Ok(output)
}

#[pymethods]
impl NativeChainSpace {
    #[getter]
    fn degree(&self) -> isize {
        match &self.inner {
            ExactSpace::IntegerChain(value) => value.degree(),
            ExactSpace::IntegerCochain(value) => value.degree(),
            ExactSpace::RationalChain(value) => value.degree(),
            ExactSpace::RationalCochain(value) => value.degree(),
        }
    }

    #[getter]
    fn dimension(&self) -> usize {
        match &self.inner {
            ExactSpace::IntegerChain(value) => value.basis_size(),
            ExactSpace::IntegerCochain(value) => value.basis_size(),
            ExactSpace::RationalChain(value) => value.basis_size(),
            ExactSpace::RationalCochain(value) => value.basis_size(),
        }
    }

    fn element(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<NativeChainElement> {
        let coordinates = match &self.inner {
            ExactSpace::IntegerChain(_) | ExactSpace::IntegerCochain(_) => {
                ExactCoordinates::Integer(integer_coordinates(value)?)
            }
            ExactSpace::RationalChain(_) | ExactSpace::RationalCochain(_) => {
                ExactCoordinates::Rational(rational_coordinates(value)?)
            }
        };
        let space = self.inner.clone();
        let inner = py
            .detach(move || admit_exact_element(&space, coordinates))
            .map_err(chain_error)?;
        Ok(NativeChainElement { inner })
    }
}

enum ExactCoordinates {
    Integer(Vec<(usize, BigInt)>),
    Rational(Vec<(usize, ExactRational)>),
}

fn admit_exact_element(
    space: &ExactSpace,
    coordinates: ExactCoordinates,
) -> Result<ExactElement, CoreChainError> {
    match (space, coordinates) {
        (ExactSpace::IntegerChain(space), ExactCoordinates::Integer(values)) => {
            space.element(values).map(ExactElement::IntegerChain)
        }
        (ExactSpace::IntegerCochain(space), ExactCoordinates::Integer(values)) => {
            space.element(values).map(ExactElement::IntegerCochain)
        }
        (ExactSpace::RationalChain(space), ExactCoordinates::Rational(values)) => {
            space.element(values).map(ExactElement::RationalChain)
        }
        (ExactSpace::RationalCochain(space), ExactCoordinates::Rational(values)) => {
            space.element(values).map(ExactElement::RationalCochain)
        }
        _ => Err(CoreChainError::SpaceMismatch),
    }
}

#[derive(Clone)]
pub(crate) enum ExactElement {
    IntegerChain(ExactCoreElement<IntegerRing, Chain, BigIntEncoding>),
    IntegerCochain(ExactCoreElement<IntegerRing, Cochain, BigIntEncoding>),
    RationalChain(ExactCoreElement<RationalField, Chain, ReducedFractionEncoding>),
    RationalCochain(ExactCoreElement<RationalField, Cochain, ReducedFractionEncoding>),
}

#[pyclass(name = "Element", frozen, module = "polygeo.chain")]
pub(crate) struct NativeChainElement {
    pub(crate) inner: ExactElement,
}

#[pymethods]
impl NativeChainElement {
    #[getter]
    fn degree(&self) -> isize {
        match &self.inner {
            ExactElement::IntegerChain(value) => value.degree(),
            ExactElement::IntegerCochain(value) => value.degree(),
            ExactElement::RationalChain(value) => value.degree(),
            ExactElement::RationalCochain(value) => value.degree(),
        }
    }

    #[getter]
    fn dimension(&self) -> usize {
        match &self.inner {
            ExactElement::IntegerChain(value) => value.basis_size(),
            ExactElement::IntegerCochain(value) => value.basis_size(),
            ExactElement::RationalChain(value) => value.basis_size(),
            ExactElement::RationalCochain(value) => value.basis_size(),
        }
    }

    fn to_python_copy(&self, py: Python<'_>) -> PyResult<(Py<PyTuple>, Py<PyTuple>)> {
        match &self.inner {
            ExactElement::IntegerChain(value) => Ok((
                PyTuple::new(py, value.indices())?.unbind(),
                bigint_tuple(py, value.coefficients())?,
            )),
            ExactElement::IntegerCochain(value) => Ok((
                PyTuple::new(py, value.indices())?.unbind(),
                bigint_tuple(py, value.coefficients())?,
            )),
            ExactElement::RationalChain(value) => Ok((
                PyTuple::new(py, value.indices())?.unbind(),
                rational_tuple(py, value.coefficients())?,
            )),
            ExactElement::RationalCochain(value) => Ok((
                PyTuple::new(py, value.indices())?.unbind(),
                rational_tuple(py, value.coefficients())?,
            )),
        }
    }

    fn evaluate(&self, py: Python<'_>, chain: &Self) -> PyResult<Py<PyAny>> {
        let cochain = self.inner.clone();
        let chain = chain.inner.clone();
        match py
            .detach(move || evaluate_exact(&cochain, &chain))
            .map_err(chain_error)?
        {
            ExactEvaluation::Integer(value) => bigint_to_python(py, &value).map(Bound::unbind),
            ExactEvaluation::Rational(value) => {
                let constructor = PyModule::import(py, "fractions")?.getattr("Fraction")?;
                rational_to_python(py, &constructor, &value).map(Bound::unbind)
            }
        }
    }

    fn cup(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let left = self.inner.clone();
        let right = other.inner.clone();
        let inner = py
            .detach(move || cup_exact(&left, &right))
            .map_err(chain_error)?;
        Ok(Self { inner })
    }

    fn wedge(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let left = self.inner.clone();
        let right = other.inner.clone();
        let inner = py
            .detach(move || wedge_exact(&left, &right))
            .map_err(chain_error)?;
        Ok(Self { inner })
    }
}

enum ExactEvaluation {
    Integer(BigInt),
    Rational(ExactRational),
}

fn evaluate_exact(
    cochain: &ExactElement,
    chain: &ExactElement,
) -> Result<ExactEvaluation, CoreChainError> {
    match (cochain, chain) {
        (ExactElement::IntegerCochain(cochain), ExactElement::IntegerChain(chain)) => {
            cochain.evaluate(chain).map(ExactEvaluation::Integer)
        }
        (ExactElement::RationalCochain(cochain), ExactElement::RationalChain(chain)) => {
            cochain.evaluate(chain).map(ExactEvaluation::Rational)
        }
        _ => Err(CoreChainError::SpaceMismatch),
    }
}

fn cup_exact(left: &ExactElement, right: &ExactElement) -> Result<ExactElement, CoreChainError> {
    match (left, right) {
        (ExactElement::IntegerCochain(left), ExactElement::IntegerCochain(right)) => {
            left.cup(right).map(ExactElement::IntegerCochain)
        }
        (ExactElement::RationalCochain(left), ExactElement::RationalCochain(right)) => {
            left.cup(right).map(ExactElement::RationalCochain)
        }
        _ => Err(CoreChainError::SpaceMismatch),
    }
}

fn wedge_exact(left: &ExactElement, right: &ExactElement) -> Result<ExactElement, CoreChainError> {
    match (left, right) {
        (ExactElement::RationalCochain(left), ExactElement::RationalCochain(right)) => {
            left.wedge(right).map(ExactElement::RationalCochain)
        }
        (ExactElement::IntegerCochain(_), ExactElement::IntegerCochain(_)) => {
            Err(CoreChainError::CoefficientFieldRequired)
        }
        _ => Err(CoreChainError::SpaceMismatch),
    }
}

#[derive(Clone)]
enum ExactMap {
    IntegerChain(LinearMap<IntegerRing, Chain, Chain>),
    IntegerCochain(LinearMap<IntegerRing, Cochain, Cochain>),
    RationalChain(LinearMap<RationalField, Chain, Chain>),
    RationalCochain(LinearMap<RationalField, Cochain, Cochain>),
}

#[pyclass(name = "LinearMap", frozen, module = "polygeo.chain")]
struct NativeLinearMap {
    inner: ExactMap,
}

#[pymethods]
impl NativeLinearMap {
    #[getter]
    fn source(&self) -> NativeChainSpace {
        let inner = match &self.inner {
            ExactMap::IntegerChain(value) => ExactSpace::IntegerChain(value.source().clone()),
            ExactMap::IntegerCochain(value) => ExactSpace::IntegerCochain(value.source().clone()),
            ExactMap::RationalChain(value) => ExactSpace::RationalChain(value.source().clone()),
            ExactMap::RationalCochain(value) => ExactSpace::RationalCochain(value.source().clone()),
        };
        NativeChainSpace { inner }
    }

    #[getter]
    fn target(&self) -> NativeChainSpace {
        let inner = match &self.inner {
            ExactMap::IntegerChain(value) => ExactSpace::IntegerChain(value.target().clone()),
            ExactMap::IntegerCochain(value) => ExactSpace::IntegerCochain(value.target().clone()),
            ExactMap::RationalChain(value) => ExactSpace::RationalChain(value.target().clone()),
            ExactMap::RationalCochain(value) => ExactSpace::RationalCochain(value.target().clone()),
        };
        NativeChainSpace { inner }
    }

    fn apply(&self, py: Python<'_>, value: &NativeChainElement) -> PyResult<NativeChainElement> {
        let map = self.inner.clone();
        let value = value.inner.clone();
        let inner = py
            .detach(move || apply_exact_map(&map, &value))
            .map_err(chain_error)?;
        Ok(NativeChainElement { inner })
    }

    fn dual(&self) -> Self {
        let inner = match &self.inner {
            ExactMap::IntegerChain(value) => ExactMap::IntegerCochain(value.dual()),
            ExactMap::IntegerCochain(value) => ExactMap::IntegerChain(value.dual()),
            ExactMap::RationalChain(value) => ExactMap::RationalCochain(value.dual()),
            ExactMap::RationalCochain(value) => ExactMap::RationalChain(value.dual()),
        };
        Self { inner }
    }

    fn compose(&self, before: &Self) -> PyResult<Self> {
        let inner = match (&self.inner, &before.inner) {
            (ExactMap::IntegerChain(after), ExactMap::IntegerChain(before)) => {
                ExactMap::IntegerChain(compose(after, before).map_err(composition_error)?)
            }
            (ExactMap::IntegerCochain(after), ExactMap::IntegerCochain(before)) => {
                ExactMap::IntegerCochain(compose(after, before).map_err(composition_error)?)
            }
            (ExactMap::RationalChain(after), ExactMap::RationalChain(before)) => {
                ExactMap::RationalChain(compose(after, before).map_err(composition_error)?)
            }
            (ExactMap::RationalCochain(after), ExactMap::RationalCochain(before)) => {
                ExactMap::RationalCochain(compose(after, before).map_err(composition_error)?)
            }
            _ => return Err(chain_error(CoreChainError::SpaceMismatch)),
        };
        Ok(Self { inner })
    }

    fn over(&self, _coefficients: &PyRationalField) -> PyResult<Self> {
        self.over_q()
    }
}

fn apply_exact_map(map: &ExactMap, value: &ExactElement) -> Result<ExactElement, CoreChainError> {
    match (map, value) {
        (ExactMap::IntegerChain(map), ExactElement::IntegerChain(value)) => {
            map.apply(value).map(ExactElement::IntegerChain)
        }
        (ExactMap::IntegerCochain(map), ExactElement::IntegerCochain(value)) => {
            map.apply(value).map(ExactElement::IntegerCochain)
        }
        (ExactMap::RationalChain(map), ExactElement::RationalChain(value)) => {
            map.apply(value).map(ExactElement::RationalChain)
        }
        (ExactMap::RationalCochain(map), ExactElement::RationalCochain(value)) => {
            map.apply(value).map(ExactElement::RationalCochain)
        }
        _ => Err(CoreChainError::SpaceMismatch),
    }
}

impl NativeLinearMap {
    fn over_q(&self) -> PyResult<Self> {
        let coefficients = RationalField::new(IntegerRing);
        let inner = match &self.inner {
            ExactMap::IntegerChain(value) => ExactMap::RationalChain(value.over(coefficients)),
            ExactMap::IntegerCochain(value) => ExactMap::RationalCochain(value.over(coefficients)),
            ExactMap::RationalChain(_) | ExactMap::RationalCochain(_) => {
                return Err(exact_error(
                    "coefficient_relation",
                    "the map is already over Q",
                ));
            }
        };
        Ok(Self { inner })
    }

    fn admits_encoding(&self, encoding: &str) -> bool {
        matches!(
            (&self.inner, encoding),
            (
                ExactMap::IntegerChain(_) | ExactMap::IntegerCochain(_),
                "bigint"
            ) | (
                ExactMap::RationalChain(_) | ExactMap::RationalCochain(_),
                "fraction"
            )
        )
    }
}

fn estimate_exact_csr(map: &ExactMap, encoding: &str) -> Result<CsrEstimate, RepresentationError> {
    match (map, encoding) {
        (ExactMap::IntegerChain(map), "bigint") => CsrRepresentation::estimate(map, BigIntEncoding),
        (ExactMap::IntegerCochain(map), "bigint") => {
            CsrRepresentation::estimate(map, BigIntEncoding)
        }
        (ExactMap::RationalChain(map), "fraction") => {
            CsrRepresentation::estimate(map, ReducedFractionEncoding)
        }
        (ExactMap::RationalCochain(map), "fraction") => {
            CsrRepresentation::estimate(map, ReducedFractionEncoding)
        }
        _ => Err(RepresentationError::Unavailable),
    }
}

fn build_exact_csr(
    map: &ExactMap,
    encoding: &str,
    limit: CsrBuildLimit,
) -> Result<ExactCsr, RepresentationError> {
    match (map, encoding) {
        (ExactMap::IntegerChain(map), "bigint") => {
            CsrRepresentation::build(map, BigIntEncoding, limit).map(ExactCsr::IntegerChain)
        }
        (ExactMap::IntegerCochain(map), "bigint") => {
            CsrRepresentation::build(map, BigIntEncoding, limit).map(ExactCsr::IntegerCochain)
        }
        (ExactMap::RationalChain(map), "fraction") => {
            CsrRepresentation::build(map, ReducedFractionEncoding, limit)
                .map(ExactCsr::RationalChain)
        }
        (ExactMap::RationalCochain(map), "fraction") => {
            CsrRepresentation::build(map, ReducedFractionEncoding, limit)
                .map(ExactCsr::RationalCochain)
        }
        _ => Err(RepresentationError::Unavailable),
    }
}

fn coefficient_encoding_error() -> PyErr {
    exact_error(
        "coefficient_system",
        "the encoding does not represent this coefficient system",
    )
}

fn public_encoding(encoding: &Bound<'_, PyType>) -> PyResult<&'static str> {
    let py = encoding.py();
    if encoding.is(py.get_type::<PyBigIntEncoding>()) {
        Ok("bigint")
    } else if encoding.is(py.get_type::<PyReducedFractionEncoding>()) {
        Ok("fraction")
    } else {
        Err(coefficient_encoding_error())
    }
}

#[derive(Clone)]
enum ExactCsr {
    IntegerChain(CsrRepresentation<IntegerRing, Chain, Chain, BigIntEncoding>),
    IntegerCochain(CsrRepresentation<IntegerRing, Cochain, Cochain, BigIntEncoding>),
    RationalChain(CsrRepresentation<RationalField, Chain, Chain, ReducedFractionEncoding>),
    RationalCochain(CsrRepresentation<RationalField, Cochain, Cochain, ReducedFractionEncoding>),
}

#[pyclass(name = "Csr", frozen, module = "polygeo.chain")]
struct NativeCsrRepresentation {
    inner: ExactCsr,
}

impl NativeCsrRepresentation {
    fn scipy_int64_parts(&self, py: Python<'_>) -> PyResult<PyBoundaryParts> {
        match &self.inner {
            ExactCsr::IntegerChain(value) => integer_scipy_parts(py, value),
            ExactCsr::IntegerCochain(value) => integer_scipy_parts(py, value),
            ExactCsr::RationalChain(_) | ExactCsr::RationalCochain(_) => {
                Err(coefficient_encoding_error())
            }
        }
    }
}

fn csr_pattern_arrays(
    py: Python<'_>,
    row_offsets: &[usize],
    column_indices: &[usize],
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let offsets = filled_array_1d(py, row_offsets.len(), |target| {
        fill_indices::<i64>(row_offsets.iter().copied(), target)
    })?;
    let columns = filled_array_1d(py, column_indices.len(), |target| {
        fill_indices::<i64>(column_indices.iter().copied(), target)
    })?;
    offsets.bind(py).call_method1("setflags", (false,))?;
    columns.bind(py).call_method1("setflags", (false,))?;
    Ok((offsets, columns))
}

#[pyclass(name = "IntegerCsrParts", frozen, module = "polygeo.chain")]
struct PyIntegerCsrParts {
    row_offsets: Py<PyAny>,
    column_indices: Py<PyAny>,
    coefficients: Py<PyTuple>,
    shape: (usize, usize),
}

#[pymethods]
impl PyIntegerCsrParts {
    #[getter]
    fn row_offsets(&self, py: Python<'_>) -> Py<PyAny> {
        self.row_offsets.clone_ref(py)
    }

    #[getter]
    fn column_indices(&self, py: Python<'_>) -> Py<PyAny> {
        self.column_indices.clone_ref(py)
    }

    #[getter]
    fn coefficients(&self, py: Python<'_>) -> Py<PyTuple> {
        self.coefficients.clone_ref(py)
    }

    #[getter]
    const fn shape(&self) -> (usize, usize) {
        self.shape
    }
}

#[pyclass(name = "RationalCsrParts", frozen, module = "polygeo.chain")]
struct PyRationalCsrParts {
    row_offsets: Py<PyAny>,
    column_indices: Py<PyAny>,
    numerators: Py<PyTuple>,
    denominators: Py<PyTuple>,
    shape: (usize, usize),
}

#[pymethods]
impl PyRationalCsrParts {
    #[getter]
    fn row_offsets(&self, py: Python<'_>) -> Py<PyAny> {
        self.row_offsets.clone_ref(py)
    }

    #[getter]
    fn column_indices(&self, py: Python<'_>) -> Py<PyAny> {
        self.column_indices.clone_ref(py)
    }

    #[getter]
    fn numerators(&self, py: Python<'_>) -> Py<PyTuple> {
        self.numerators.clone_ref(py)
    }

    #[getter]
    fn denominators(&self, py: Python<'_>) -> Py<PyTuple> {
        self.denominators.clone_ref(py)
    }

    #[getter]
    const fn shape(&self) -> (usize, usize) {
        self.shape
    }
}

fn integer_parts_record(
    py: Python<'_>,
    representation: &CsrRepresentation<IntegerRing, impl Variance, impl Variance, BigIntEncoding>,
) -> PyResult<PyIntegerCsrParts> {
    let (row_offsets, column_indices) = csr_pattern_arrays(
        py,
        representation.row_offsets(),
        representation.column_indices(),
    )?;
    Ok(PyIntegerCsrParts {
        row_offsets,
        column_indices,
        coefficients: bigint_tuple(py, representation.coefficients())?,
        shape: representation.shape(),
    })
}

fn rational_parts_record(
    py: Python<'_>,
    representation: &CsrRepresentation<
        RationalField,
        impl Variance,
        impl Variance,
        ReducedFractionEncoding,
    >,
) -> PyResult<PyRationalCsrParts> {
    let (row_offsets, column_indices) = csr_pattern_arrays(
        py,
        representation.row_offsets(),
        representation.column_indices(),
    )?;
    Ok(PyRationalCsrParts {
        row_offsets,
        column_indices,
        numerators: bigint_tuple(
            py,
            representation
                .coefficients()
                .iter()
                .map(ExactRational::numer),
        )?,
        denominators: bigint_tuple(
            py,
            representation
                .coefficients()
                .iter()
                .map(ExactRational::denom),
        )?,
        shape: representation.shape(),
    })
}

#[pymethods]
impl NativeCsrRepresentation {
    #[staticmethod]
    fn estimate(
        py: Python<'_>,
        map: &NativeLinearMap,
        encoding: &Bound<'_, PyType>,
    ) -> PyResult<PyCsrEstimate> {
        let encoding = public_encoding(encoding)?.to_owned();
        if !map.admits_encoding(&encoding) {
            return Err(coefficient_encoding_error());
        }
        let map = map.inner.clone();
        let inner = py
            .detach(move || estimate_exact_csr(&map, &encoding))
            .map_err(representation_error)?;
        Ok(PyCsrEstimate { inner })
    }

    #[staticmethod]
    fn build(
        py: Python<'_>,
        map: &NativeLinearMap,
        encoding: &Bound<'_, PyType>,
        limit: &PyCsrBuildLimit,
    ) -> PyResult<Self> {
        let encoding = public_encoding(encoding)?.to_owned();
        if !map.admits_encoding(&encoding) {
            return Err(coefficient_encoding_error());
        }
        let estimate = estimate_exact_csr(&map.inner, &encoding).map_err(representation_error)?;
        let admitted = admitted_build_limit(estimate, *limit);
        let map = map.inner.clone();
        let inner = py
            .detach(move || build_exact_csr(&map, &encoding, admitted))
            .map_err(representation_error)?;
        Ok(Self { inner })
    }

    #[getter]
    fn represented_map(&self) -> NativeLinearMap {
        let inner = match &self.inner {
            ExactCsr::IntegerChain(value) => {
                ExactMap::IntegerChain(value.represented_map().clone())
            }
            ExactCsr::IntegerCochain(value) => {
                ExactMap::IntegerCochain(value.represented_map().clone())
            }
            ExactCsr::RationalChain(value) => {
                ExactMap::RationalChain(value.represented_map().clone())
            }
            ExactCsr::RationalCochain(value) => {
                ExactMap::RationalCochain(value.represented_map().clone())
            }
        };
        NativeLinearMap { inner }
    }

    #[getter]
    fn encoding(&self, py: Python<'_>) -> Py<PyType> {
        match self.inner {
            ExactCsr::IntegerChain(_) | ExactCsr::IntegerCochain(_) => {
                py.get_type::<PyBigIntEncoding>().unbind()
            }
            ExactCsr::RationalChain(_) | ExactCsr::RationalCochain(_) => {
                py.get_type::<PyReducedFractionEncoding>().unbind()
            }
        }
    }

    fn to_python_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            ExactCsr::IntegerChain(value) => {
                Py::new(py, integer_parts_record(py, value)?).map(Py::into_any)
            }
            ExactCsr::IntegerCochain(value) => {
                Py::new(py, integer_parts_record(py, value)?).map(Py::into_any)
            }
            ExactCsr::RationalChain(value) => {
                Py::new(py, rational_parts_record(py, value)?).map(Py::into_any)
            }
            ExactCsr::RationalCochain(value) => {
                Py::new(py, rational_parts_record(py, value)?).map(Py::into_any)
            }
        }
    }

    fn to_scipy_int64_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let (data, indices, offsets, shape) = self.scipy_int64_parts(py)?;
        let storage = PyTuple::new(py, [data.bind(py), indices.bind(py), offsets.bind(py)])?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("shape", shape)?;
        kwargs.set_item("copy", false)?;
        PyModule::import(py, "scipy.sparse")?
            .getattr("csr_array")?
            .call((storage,), Some(&kwargs))
            .map(Bound::unbind)
    }

    fn apply(&self, py: Python<'_>, value: &NativeChainElement) -> PyResult<NativeChainElement> {
        let representation = self.inner.clone();
        let value = value.inner.clone();
        let inner = py
            .detach(move || apply_exact_csr(&representation, &value))
            .map_err(chain_error)?;
        Ok(NativeChainElement { inner })
    }
}

fn apply_exact_csr(
    representation: &ExactCsr,
    value: &ExactElement,
) -> Result<ExactElement, CoreChainError> {
    match (representation, value) {
        (ExactCsr::IntegerChain(map), ExactElement::IntegerChain(value)) => {
            map.apply(value).map(ExactElement::IntegerChain)
        }
        (ExactCsr::IntegerCochain(map), ExactElement::IntegerCochain(value)) => {
            map.apply(value).map(ExactElement::IntegerCochain)
        }
        (ExactCsr::RationalChain(map), ExactElement::RationalChain(value)) => {
            map.apply(value).map(ExactElement::RationalChain)
        }
        (ExactCsr::RationalCochain(map), ExactElement::RationalCochain(value)) => {
            map.apply(value).map(ExactElement::RationalCochain)
        }
        _ => Err(CoreChainError::SpaceMismatch),
    }
}

fn integer_scipy_parts<S, T>(
    py: Python<'_>,
    representation: &CsrRepresentation<IntegerRing, S, T, BigIntEncoding>,
) -> PyResult<PyBoundaryParts>
where
    S: Variance,
    T: Variance,
{
    let coefficients = filled_exact_i64(py, representation.coefficients().len(), |output| {
        for (target, value) in output.iter_mut().zip(representation.coefficients()) {
            *target = checked_bigint_i64(value).ok_or_else(|| {
                exact_error(
                    "coefficient_overflow",
                    "an exact coefficient is outside the requested int64 projection",
                )
            })?;
        }
        Ok(())
    })?;
    let indices = filled_exact_i64(py, representation.column_indices().len(), |output| {
        fill_exact_indices(representation.column_indices(), output)
    })?;
    let offsets = filled_exact_i64(py, representation.row_offsets().len(), |output| {
        fill_exact_indices(representation.row_offsets(), output)
    })?;
    Ok((coefficients, indices, offsets, representation.shape()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyIntegerRing>()?;
    module.add_class::<PyRationalField>()?;
    module.add_class::<PyChainVariance>()?;
    module.add_class::<PyCochainVariance>()?;
    module.add_class::<PyBigIntEncoding>()?;
    module.add_class::<PyReducedFractionEncoding>()?;
    module.add_class::<PyChainIsomorphism>()?;
    module.add_class::<NativeChainComplex>()?;
    module.add_class::<NativeCochainComplex>()?;
    module.add_class::<NativeChainSpace>()?;
    module.add_class::<NativeChainElement>()?;
    module.add_class::<NativeLinearMap>()?;
    module.add_class::<PyCsrEstimate>()?;
    module.add_class::<PyCsrBuildLimit>()?;
    module.add_class::<PyChainLawLimit>()?;
    module.add_class::<NativeCsrRepresentation>()?;
    module.add_class::<PyIntegerCsrParts>()?;
    module.add_class::<PyRationalCsrParts>()?;
    module.add("ZZ", Py::new(module.py(), PyIntegerRing)?)?;
    module.add("QQ", Py::new(module.py(), PyRationalField)?)?;
    module.add(
        "DEFAULT_LAW_LIMIT",
        Py::new(module.py(), PyChainLawLimit::DEFAULT)?,
    )?;
    crate::homology::register(module)?;
    let error = module.py().get_type::<ChainError>();
    error.setattr("__module__", "polygeo.chain")?;
    module.add("ChainError", error)
}

#[cfg(test)]
mod exact_projection_tests {
    use super::*;

    #[test]
    fn int64_projection_rejects_arbitrary_precision_overflow() {
        let value = BigInt::from(1_u8) << 200;
        assert_eq!(checked_bigint_i64(&value), None);
        assert_eq!(checked_bigint_i64(&BigInt::from(i64::MIN)), Some(i64::MIN));
        assert_eq!(checked_bigint_i64(&BigInt::from(i64::MAX)), Some(i64::MAX));
    }
}
