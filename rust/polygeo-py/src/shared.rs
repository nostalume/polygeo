use std::sync::Arc;

use num_bigint::BigInt;
use numpy::ndarray::Dimension;
use numpy::{Element, PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArrayDyn};
use polygeo_core::{
    BigIntEncoding, BoundaryRef, CandidateInput, CanonicalSelection, Chain, ChainComplex,
    ChainError as CoreChainError, ChainIsomorphism as CoreChainIsomorphism, ChainLawLimit, Cochain,
    CoefficientSlice, ComplexCore, CompositionError, CorrespondenceDirection, CsrBuildLimit,
    CsrEstimate, CsrRepresentation, ExactRational, FaceKind, HalfedgeInput, HalfedgeSurfaceCore,
    IntegerRing, IsomorphismError, LinearMap, RationalField, ReducedFractionEncoding,
    RepresentationError, SimplexSubset, Space, StorageLimit, SurfaceCorrespondence,
    TopologyDetailValue, TopologyError, WorkLimit, compose,
};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyCFunction, PyDict, PyInt, PyModule, PyTuple, PyType};

create_exception!(
    _polygeo_native,
    SimplicialError,
    PyValueError,
    "Classified simplicial topology failure."
);

create_exception!(
    _polygeo_native,
    HalfedgeError,
    SimplicialError,
    "Classified halfedge topology failure."
);

create_exception!(
    _polygeo_native,
    ChainError,
    PyValueError,
    "Classified exact chain-algebra failure."
);

type PyBoundaryParts = (Py<PyAny>, Py<PyAny>, Py<PyAny>, (usize, usize));

#[pyclass(name = "IntegerRing", frozen, module = "polygeo")]
struct PyIntegerRing;

#[pyclass(name = "RationalField", frozen, module = "polygeo")]
struct PyRationalField;

#[pyclass(name = "Chain", frozen, module = "polygeo")]
struct PyChainVariance;

#[pyclass(name = "Cochain", frozen, module = "polygeo")]
struct PyCochainVariance;

#[pyclass(name = "BigIntEncoding", frozen, module = "polygeo")]
struct PyBigIntEncoding;

#[pyclass(name = "ReducedFractionEncoding", frozen, module = "polygeo")]
struct PyReducedFractionEncoding;

fn topology_error(error: TopologyError) -> PyErr {
    Python::attach(|py| {
        let translated = (|| -> PyResult<PyErr> {
            let details = PyDict::new(py);
            for field in error.details().fields() {
                match field.value() {
                    TopologyDetailValue::Signed(value) => details.set_item(field.name(), value)?,
                    TopologyDetailValue::Unsigned(value) => {
                        details.set_item(field.name(), value)?;
                    }
                    TopologyDetailValue::Index(value) => details.set_item(field.name(), value)?,
                    TopologyDetailValue::Text(value) => details.set_item(field.name(), value)?,
                    _ => {}
                }
            }
            Ok(topology_exception(
                py,
                error.reason(),
                error.to_string(),
                details.unbind(),
            ))
        })();
        translated.unwrap_or_else(|translation_error| translation_error)
    })
}

fn transport_error(reason: &'static str, message: &'static str) -> PyErr {
    Python::attach(|py| {
        topology_exception(py, reason, message, PyDict::new(py).unbind())
    })
}

fn topology_exception(
    py: Python<'_>,
    reason: &'static str,
    message: impl Into<String>,
    details: Py<PyDict>,
) -> PyErr {
    let error = SimplicialError::new_err(message.into());
    let value = error.value(py);
    let _ = value.setattr("_reason", reason);
    let proxy = PyModule::import(py, "types")
        .and_then(|module| module.getattr("MappingProxyType"))
        .and_then(|constructor| constructor.call1((details,)));
    if let Ok(proxy) = proxy {
        let _ = value.setattr("_details", proxy);
    }
    error
}

fn halfedge_exception(
    py: Python<'_>,
    reason: &'static str,
    message: impl Into<String>,
    details: Py<PyDict>,
) -> PyErr {
    let error = HalfedgeError::new_err(message.into());
    let value = error.value(py);
    let _ = value.setattr("_reason", reason);
    let proxy = PyModule::import(py, "types")
        .and_then(|module| module.getattr("MappingProxyType"))
        .and_then(|constructor| constructor.call1((details,)));
    if let Ok(proxy) = proxy {
        let _ = value.setattr("_details", proxy);
    }
    error
}

fn halfedge_transport_error(error: PyErr) -> PyErr {
    Python::attach(|py| {
        if !error.is_instance_of::<SimplicialError>(py) {
            return error;
        }
        let original = error.value(py);
        let translated = HalfedgeError::new_err(
            original
                .str()
                .map_or_else(|_| "halfedge admission failed".into(), |text| text.to_string()),
        );
        let value = translated.value(py);
        if let Ok(reason) = original.getattr("_reason") {
            let _ = value.setattr("_reason", reason);
        }
        if let Ok(details) = original.getattr("_details") {
            let _ = value.setattr("_details", details);
        }
        translated
    })
}

fn halfedge_topology_error(error: TopologyError) -> PyErr {
    Python::attach(|py| {
        let details = PyDict::new(py);
        for field in error.details().fields() {
            let translated = match field.value() {
                TopologyDetailValue::Signed(value) => details.set_item(field.name(), value),
                TopologyDetailValue::Unsigned(value) => details.set_item(field.name(), value),
                TopologyDetailValue::Index(value) => details.set_item(field.name(), value),
                TopologyDetailValue::Text(value) => details.set_item(field.name(), value),
                _ => continue,
            };
            if translated.is_err() {
                return halfedge_exception(
                    py,
                    "translation",
                    "failed to translate halfedge failure",
                    PyDict::new(py).unbind(),
                );
            }
        }
        halfedge_exception(py, error.reason(), error.to_string(), details.unbind())
    })
}

fn halfedge_isomorphism_error(error: IsomorphismError) -> PyErr {
    if let IsomorphismError::Topology(error) = error {
        return halfedge_topology_error(error);
    }
    Python::attach(|py| {
        let details = PyDict::new(py);
        if let Some((axis, required, limit)) = error.resource_limit()
            && details
                .set_item("axis", axis)
                .and_then(|()| details.set_item("required", required))
                .and_then(|()| details.set_item("limit", limit))
                .is_err()
        {
            return halfedge_exception(
                py,
                "translation",
                "failed to translate halfedge chain-law failure",
                PyDict::new(py).unbind(),
            );
        }
        halfedge_exception(py, error.reason(), error.to_string(), details.unbind())
    })
}

fn install_topology_error_properties(py: Python<'_>, error: &Bound<'_, PyType>) -> PyResult<()> {
    let reason_getter = PyCFunction::new_closure(
        py,
        Some(c"reason"),
        None,
        |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
            args.get_item(0)?.getattr("_reason").map(Bound::unbind)
        },
    )?;
    let details_getter = PyCFunction::new_closure(
        py,
        Some(c"details"),
        None,
        |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
            args.get_item(0)?.getattr("_details").map(Bound::unbind)
        },
    )?;
    let property = PyModule::import(py, "builtins")?.getattr("property")?;
    error.setattr("reason", property.call1((reason_getter,))?)?;
    error.setattr("details", property.call1((details_getter,))?)?;
    Ok(())
}

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

fn classified_exception(
    py: Python<'_>,
    error: PyErr,
    reason: &'static str,
    details: Py<PyDict>,
) -> PyErr {
    let value = error.value(py);
    let _ = value.setattr("reason", reason);
    let proxy = PyModule::import(py, "types")
        .and_then(|module| module.getattr("MappingProxyType"))
        .and_then(|constructor| constructor.call1((details,)));
    if let Ok(proxy) = proxy {
        let _ = value.setattr("details", proxy);
    }
    error
}

fn chain_error(error: CoreChainError) -> PyErr {
    Python::attach(|py| {
        let details = PyDict::new(py);
        let translation_failed = match error {
            CoreChainError::BasisIndexOutside { index, bound } => {
                details.set_item("index", index).is_err()
                    || details.set_item("bound", bound).is_err()
            }
            CoreChainError::SpaceMismatch | CoreChainError::Topology(_) => false,
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
    module = "polygeo",
    skip_from_py_object
)]
#[derive(Clone, Copy)]
struct PyChainLawLimit {
    retained_logical_bytes: u64,
    peak_live_logical_bytes: u64,
    terms: u64,
}

impl PyChainLawLimit {
    const DEFAULT: Self = Self {
        retained_logical_bytes: 128 * 1024 * 1024,
        peak_live_logical_bytes: 512 * 1024 * 1024,
        terms: 100_000_000,
    };

    fn core(self) -> ChainLawLimit {
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
    module = "polygeo",
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
    module = "polygeo",
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

fn filled_array<T, D>(
    array: Bound<'_, PyArray<T, D>>,
    fill: impl FnOnce(&mut [T]) -> Result<(), TopologyError>,
) -> PyResult<Py<PyAny>>
where
    T: Element,
    D: Dimension,
{
    let mut writable = array
        .try_readwrite()
        .map_err(|_| topology_error(TopologyError::InternalInvariant))?;
    let output = writable
        .as_slice_mut()
        .map_err(|_| topology_error(TopologyError::InternalInvariant))?;
    fill(output).map_err(topology_error)?;
    drop(writable);
    Ok(array.unbind().into_any())
}

fn filled_array_1d<T>(
    py: Python<'_>,
    length: usize,
    fill: impl FnOnce(&mut [T]) -> Result<(), TopologyError>,
) -> PyResult<Py<PyAny>>
where
    T: Element,
{
    filled_array(PyArray1::<T>::zeros(py, length, false), fill)
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

fn filled_array_2d<T>(
    py: Python<'_>,
    rows: usize,
    columns: usize,
    fill: impl FnOnce(&mut [T]) -> Result<(), TopologyError>,
) -> PyResult<Py<PyAny>>
where
    T: Element,
{
    filled_array(PyArray2::<T>::zeros(py, [rows, columns], false), fill)
}

fn fill_indices<T>(
    values: impl IntoIterator<Item = usize>,
    output: &mut [T],
) -> Result<(), TopologyError>
where
    T: TryFrom<usize>,
{
    for (target, value) in output.iter_mut().zip(values) {
        *target = T::try_from(value).map_err(|_| TopologyError::IndexOverflow)?;
    }
    Ok(())
}

fn project_boundary(py: Python<'_>, boundary: BoundaryRef<'_>) -> PyResult<PyBoundaryParts> {
    let shape = boundary.shape();
    let indptr = boundary.indptr();
    let compact = [shape.0, shape.1, boundary.indices().len()]
        .into_iter()
        .chain(boundary.indices().iter().copied())
        .chain(indptr.iter().copied())
        .all(|value| i32::try_from(value).is_ok());
    let data = match boundary.coefficients() {
        CoefficientSlice::I8(values) => filled_array_1d(py, values.len(), |output| {
            output.copy_from_slice(values);
            Ok(())
        })?,
        CoefficientSlice::I64(values) => filled_array_1d(py, values.len(), |output| {
            output.copy_from_slice(values);
            Ok(())
        })?,
    };
    let (indices, indptr) = if compact {
        (
            filled_array_1d(py, boundary.indices().len(), |output| {
                fill_indices::<i32>(boundary.indices().iter().copied(), output)
            })?,
            filled_array_1d(py, indptr.len(), |output| {
                fill_indices::<i32>(indptr.iter().copied(), output)
            })?,
        )
    } else {
        (
            filled_array_1d(py, boundary.indices().len(), |output| {
                fill_indices::<i64>(boundary.indices().iter().copied(), output)
            })?,
            filled_array_1d(py, indptr.len(), |output| {
                fill_indices::<i64>(indptr.iter().copied(), output)
            })?,
        )
    };
    Ok((data, indices, indptr, shape))
}

fn normalized_integer(
    value: &Bound<'_, PyAny>,
    dimension: usize,
    shape_reason: &'static str,
    subject: &'static str,
) -> PyResult<(Py<PyAny>, bool)> {
    if value.getattr("ndim")?.extract::<usize>()? != dimension {
        return Err(transport_error(
            shape_reason,
            "integer array has the wrong rank",
        ));
    }
    let dtype = value.getattr("dtype")?;
    let kind = dtype.getattr("kind")?.extract::<char>()?;
    let item_size = dtype.getattr("itemsize")?.extract::<usize>()?;
    if !matches!(kind, 'i' | 'u') || !matches!(item_size, 1 | 2 | 4 | 8) {
        return Err(transport_error("unsupported_dtype", subject));
    }
    let signed = kind == 'i';
    let native = value.call_method1("astype", (if signed { "int64" } else { "uint64" },))?;
    Ok((native.unbind(), signed))
}

fn copy_indices<T>(
    value: &Bound<'_, PyAny>,
    mut convert: impl FnMut(T) -> Result<usize, TopologyError>,
) -> PyResult<Vec<usize>>
where
    T: Element + Copy,
{
    let array = value.extract::<PyReadonlyArrayDyn<'_, T>>()?;
    let view = array.as_array();
    let mut output = Vec::new();
    output
        .try_reserve_exact(view.len())
        .map_err(|_| topology_error(TopologyError::Allocation))?;
    for value in view.iter().copied() {
        output.push(convert(value).map_err(topology_error)?);
    }
    Ok(output)
}

fn admit_integer_matrix(
    value: &Bound<'_, PyAny>,
    vertex_count: Option<usize>,
) -> PyResult<CandidateInput> {
    let (native, signed) = normalized_integer(
        value,
        2,
        "candidate_shape",
        "maximal simplices require a fixed-width integer dtype",
    )?;
    let shape = value.getattr("shape")?.extract::<(usize, usize)>()?;
    if signed {
        let array = native
            .bind(value.py())
            .extract::<PyReadonlyArrayDyn<'_, i64>>()?;
        CandidateInput::signed(
            array.as_array().iter().copied(),
            shape.0,
            shape.1,
            vertex_count,
        )
        .map_err(topology_error)
    } else {
        let array = native
            .bind(value.py())
            .extract::<PyReadonlyArrayDyn<'_, u64>>()?;
        CandidateInput::unsigned(
            array.as_array().iter().copied(),
            shape.0,
            shape.1,
            vertex_count,
        )
        .map_err(topology_error)
    }
}

fn copy_halfedge_indices(value: &Bound<'_, PyAny>) -> PyResult<Box<[usize]>> {
    let (native, signed) = normalized_integer(
        value,
        1,
        "halfedge_shape",
        "halfedge relations require a fixed-width integer dtype",
    )
    .map_err(halfedge_transport_error)?;
    let converted = if signed {
        copy_indices::<i64>(native.bind(value.py()), |value| {
            usize::try_from(value).map_err(|_| {
                if value < 0 {
                    TopologyError::negative_index(i128::from(value))
                } else {
                    TopologyError::index_overflow(value.cast_unsigned().into())
                }
            })
        })
    } else {
        copy_indices::<u64>(native.bind(value.py()), |value| {
            usize::try_from(value).map_err(|_| TopologyError::index_overflow(value.into()))
        })
    };
    converted
        .map(Vec::into_boxed_slice)
        .map_err(halfedge_transport_error)
}

fn pack_boolean_degrees(
    owner: &Arc<ComplexCore>,
    value: &Bound<'_, PyAny>,
) -> PyResult<SimplexSubset> {
    let iterator = value.try_iter().map_err(|_| {
        transport_error(
            "mask_shape",
            "subset masks must be an iterable covering every degree",
        )
    })?;
    let mut builder = owner.subset_builder().map_err(topology_error)?;
    for item in iterator {
        let item = item?;
        let array = item
            .extract::<PyReadonlyArrayDyn<'_, bool>>()
            .map_err(|_| transport_error("mask_shape", "subset masks must be Boolean arrays"))?;
        let view = array.as_array();
        if view.ndim() != 1 {
            return Err(transport_error(
                "mask_shape",
                "each subset mask must be one-dimensional",
            ));
        }
        builder
            .push_degree(view.iter().copied())
            .map_err(topology_error)?;
    }
    builder.finish().map_err(topology_error)
}

fn copy_selection_indices(value: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let (native, signed) = normalized_integer(
        value,
        1,
        "selection_shape",
        "selection indices require a fixed-width integer dtype",
    )?;
    if signed {
        copy_indices::<i64>(native.bind(value.py()), |value| {
            usize::try_from(value).map_err(|_| TopologyError::SelectionIndexOutside)
        })
    } else {
        copy_indices::<u64>(native.bind(value.py()), |value| {
            usize::try_from(value).map_err(|_| TopologyError::SelectionIndexOutside)
        })
    }
}
