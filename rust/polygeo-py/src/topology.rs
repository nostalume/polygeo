use std::sync::Arc;

use numpy::PyReadonlyArrayDyn;
use polygeo_core::chain::IsomorphismError;
use polygeo_core::form::{ChainSpace as Binary64ChainSpace, CochainSpace as Binary64CochainSpace};
use polygeo_core::topology::{
    Basis, BoundaryRef, CandidateInput, CoefficientSlice, Complex as ComplexCore,
    Selection as CanonicalSelection, Subset as SimplexSubset, TopologyDetailValue, TopologyError,
};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyCFunction, PyDict, PyModule, PyTuple, PyType};

use crate::array::{copy_indices, fill_indices, filled_array_1d, filled_array_2d};
use crate::chain::{ExactComplex, NativeChainComplex};
use crate::form;
use crate::surface;

pub(crate) type PyBoundaryParts = (Py<PyAny>, Py<PyAny>, Py<PyAny>, (usize, usize));

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

pub(crate) fn topology_error(error: TopologyError) -> PyErr {
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

pub(crate) fn transport_error(reason: &'static str, message: &'static str) -> PyErr {
    Python::attach(|py| topology_exception(py, reason, message, PyDict::new(py).unbind()))
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

pub(crate) fn halfedge_exception(
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
        let translated = HalfedgeError::new_err(original.str().map_or_else(
            |_| "halfedge admission failed".into(),
            |text| text.to_string(),
        ));
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

pub(crate) fn halfedge_topology_error(error: TopologyError) -> PyErr {
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

pub(crate) fn halfedge_isomorphism_error(error: IsomorphismError) -> PyErr {
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

pub(crate) fn project_boundary(
    py: Python<'_>,
    boundary: BoundaryRef<'_>,
) -> PyResult<PyBoundaryParts> {
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

pub(crate) fn copy_halfedge_indices(value: &Bound<'_, PyAny>) -> PyResult<Box<[usize]>> {
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

#[pyclass(
    name = "Complex",
    frozen,
    module = "polygeo.topology",
    skip_from_py_object
)]
pub(crate) struct NativeComplex {
    pub(crate) owner: Arc<ComplexCore>,
}

fn topology_degree(degree: isize) -> PyResult<usize> {
    usize::try_from(degree).map_err(|_| {
        Python::attach(|py| {
            topology_exception(
                py,
                "degree_outside",
                "degree is outside the complex",
                PyDict::new(py).unbind(),
            )
        })
    })
}

impl NativeComplex {
    fn selected_space(
        &self,
        degree: usize,
        indices: &Bound<'_, PyAny>,
        variance: &str,
    ) -> PyResult<form::PyBinary64Space> {
        let copied = copy_selection_indices(indices)?;
        let selection = Arc::new(
            self.owner
                .selection(degree, copied)
                .map_err(topology_error)?,
        );
        let inner = match variance {
            "chain" => {
                form::Space::Chain(Binary64ChainSpace::selected(selection).map_err(topology_error)?)
            }
            "cochain" => form::Space::Cochain(
                Binary64CochainSpace::selected(selection).map_err(topology_error)?,
            ),
            _ => unreachable!("the public methods fix variance"),
        };
        Ok(form::PyBinary64Space { inner })
    }
}

#[pymethods]
impl NativeComplex {
    #[classmethod]
    fn __class_getitem__<'py>(
        class: &Bound<'py, PyType>,
        _parameter: &Bound<'_, PyAny>,
    ) -> Bound<'py, PyType> {
        class.clone()
    }

    #[staticmethod]
    #[pyo3(signature = (maximal_simplices, *, vertex_count=None))]
    fn from_maximal_simplices(
        maximal_simplices: &Bound<'_, PyAny>,
        vertex_count: Option<usize>,
    ) -> PyResult<Self> {
        let candidate = admit_integer_matrix(maximal_simplices, vertex_count)?;
        let owner = ComplexCore::admit(candidate).map_err(topology_error)?;
        Ok(Self { owner })
    }

    #[getter]
    fn vertex_count(&self) -> usize {
        self.owner.vertex_count()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.owner.dimension()
    }

    fn simplex_count(&self, degree: isize) -> PyResult<usize> {
        self.owner
            .basis(topology_degree(degree)?)
            .map(Basis::row_count)
            .map_err(topology_error)
    }

    fn orientations_numpy_copy(&self, py: Python<'_>, degree: isize) -> PyResult<Py<PyAny>> {
        let orientation = self
            .owner
            .orientation(topology_degree(degree)?)
            .map_err(topology_error)?;
        filled_array_1d(py, orientation.len(), |output| {
            output.copy_from_slice(orientation);
            Ok(())
        })
    }

    fn simplices_numpy_copy(&self, py: Python<'_>, degree: isize) -> PyResult<Py<PyAny>> {
        let basis = self
            .owner
            .basis(topology_degree(degree)?)
            .map_err(topology_error)?;
        filled_array_2d(py, basis.row_count(), basis.row_width(), |output| {
            fill_indices::<i64>(basis.values().iter().copied(), output)
        })
    }

    fn boundary_parts_numpy_copy(
        &self,
        py: Python<'_>,
        degree: isize,
    ) -> PyResult<PyBoundaryParts> {
        project_boundary(
            py,
            self.owner
                .chain_view()
                .boundary(topology_degree(degree)?)
                .map_err(topology_error)?,
        )
    }

    fn boundary_scipy_copy(&self, py: Python<'_>, degree: isize) -> PyResult<Py<PyAny>> {
        let (data, indices, indptr, shape) = self.boundary_parts_numpy_copy(py, degree)?;
        let keywords = PyDict::new(py);
        keywords.set_item("shape", shape)?;
        keywords.set_item("copy", false)?;
        PyModule::import(py, "scipy.sparse")?
            .getattr("csr_array")?
            .call(((data, indices, indptr),), Some(&keywords))
            .map(Bound::unbind)
    }

    fn chain_complex(&self) -> NativeChainComplex {
        NativeChainComplex {
            inner: ExactComplex::Integer(self.owner.chain_complex()),
        }
    }

    fn triangle_manifold(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        slf.owner.refine_triangle().map_err(topology_error)?;
        Ok(slf)
    }

    fn codimension_one_regular(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        slf.owner.refine_regular().map_err(topology_error)?;
        Ok(slf)
    }

    fn oriented(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        slf.owner.refine_oriented().map_err(topology_error)?;
        Ok(slf)
    }

    fn connected(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        slf.owner.refine_connected().map_err(topology_error)?;
        Ok(slf)
    }

    fn with_boundary(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        slf.owner.require_with_boundary().map_err(topology_error)?;
        Ok(slf)
    }

    fn without_boundary(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        slf.owner
            .require_without_boundary()
            .map_err(topology_error)?;
        Ok(slf)
    }

    fn require_triangle(&self) -> PyResult<()> {
        self.owner
            .require_triangle()
            .map(|_| ())
            .map_err(topology_error)
    }

    fn require_regular(&self) -> PyResult<()> {
        self.owner
            .require_regular()
            .map(|_| ())
            .map_err(topology_error)
    }

    fn require_oriented(&self) -> PyResult<()> {
        self.owner
            .require_oriented()
            .map(|_| ())
            .map_err(topology_error)
    }

    fn require_connected(&self) -> PyResult<()> {
        self.owner
            .require_connected()
            .map(|_| ())
            .map_err(topology_error)
    }

    fn require_with_boundary(&self) -> PyResult<()> {
        self.owner
            .require_with_boundary()
            .map(|_| ())
            .map_err(topology_error)
    }

    fn require_without_boundary(&self) -> PyResult<()> {
        self.owner
            .require_without_boundary()
            .map(|_| ())
            .map_err(topology_error)
    }

    fn boundary_mask_numpy_copy(&self, py: Python<'_>, degree: isize) -> PyResult<Py<PyAny>> {
        let degree = topology_degree(degree)?;
        let count = self
            .owner
            .basis(degree)
            .map(Basis::row_count)
            .map_err(topology_error)?;
        filled_array_1d(py, count, |output| {
            self.owner
                .require_regular()?
                .write_boundary_mask(degree, output)
        })
    }

    fn disk_boundary_vertices_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let vertices = self
            .owner
            .refine_disk()
            .and_then(|disk| disk.boundary_vertices())
            .map_err(topology_error)?;
        filled_array_1d(py, vertices.len(), |output| {
            fill_indices::<i64>(vertices.iter().copied(), output)
        })
    }

    fn boundary_subset(slf: &Bound<'_, Self>) -> PyResult<NativeSubset> {
        let borrowed = slf.borrow();
        borrowed
            .owner
            .boundary_subset()
            .map(|subset| NativeSubset {
                complex: slf.clone().unbind(),
                subset,
            })
            .map_err(topology_error)
    }

    fn subset(slf: &Bound<'_, Self>, masks: &Bound<'_, PyAny>) -> PyResult<NativeSubset> {
        let borrowed = slf.borrow();
        pack_boolean_degrees(&borrowed.owner, masks).map(|subset| NativeSubset {
            complex: slf.clone().unbind(),
            subset,
        })
    }

    fn selection(
        slf: &Bound<'_, Self>,
        degree: isize,
        indices: &Bound<'_, PyAny>,
    ) -> PyResult<NativeSelection> {
        let degree = topology_degree(degree)?;
        let copied = copy_selection_indices(indices)?;
        let borrowed = slf.borrow();
        borrowed
            .owner
            .selection(degree, copied)
            .map(|selection| NativeSelection {
                complex: slf.clone().unbind(),
                selection: Arc::new(selection),
            })
            .map_err(topology_error)
    }

    fn shares_data_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
    }

    #[pyo3(signature = (degree, *, indices=None))]
    fn binary64_chain_space(
        &self,
        degree: isize,
        indices: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<form::PyBinary64Space> {
        if let Some(indices) = indices {
            return self.selected_space(topology_degree(degree)?, indices, "chain");
        }
        Ok(form::PyBinary64Space {
            inner: form::Space::Chain(
                Binary64ChainSpace::full(self.owner.clone(), topology_degree(degree)?)
                    .map_err(topology_error)?,
            ),
        })
    }

    #[pyo3(signature = (degree, *, indices=None))]
    fn binary64_cochain_space(
        &self,
        degree: isize,
        indices: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<form::PyBinary64Space> {
        if let Some(indices) = indices {
            return self.selected_space(topology_degree(degree)?, indices, "cochain");
        }
        Ok(form::PyBinary64Space {
            inner: form::Space::Cochain(
                Binary64CochainSpace::full(self.owner.clone(), topology_degree(degree)?)
                    .map_err(topology_error)?,
            ),
        })
    }

    fn dual_cycles(&self) -> PyResult<surface::PyIntegralDualCycleBasis> {
        Ok(surface::PyIntegralDualCycleBasis {
            inner: self
                .owner
                .integral_dual_cycle_basis()
                .map_err(topology_error)?,
        })
    }
}

#[pyclass(
    name = "Subset",
    frozen,
    module = "polygeo.topology",
    skip_from_py_object
)]
struct NativeSubset {
    complex: Py<NativeComplex>,
    subset: SimplexSubset,
}

#[pymethods]
impl NativeSubset {
    #[classmethod]
    fn __class_getitem__<'py>(
        class: &Bound<'py, PyType>,
        _parameter: &Bound<'_, PyAny>,
    ) -> Bound<'py, PyType> {
        class.clone()
    }

    #[getter]
    fn topology(&self, py: Python<'_>) -> Py<NativeComplex> {
        self.complex.clone_ref(py)
    }

    fn mask_numpy_copy(&self, py: Python<'_>, degree: isize) -> PyResult<Py<PyAny>> {
        let degree = topology_degree(degree)?;
        let count = self
            .subset
            .owner()
            .basis(degree)
            .map(Basis::row_count)
            .map_err(topology_error)?;
        filled_array_1d(py, count, |output| self.subset.write_mask(degree, output))
    }

    fn owned_copy(&self, py: Python<'_>) -> PyResult<Self> {
        self.subset
            .to_owned_subset()
            .map(|subset| Self {
                complex: self.complex.clone_ref(py),
                subset,
            })
            .map_err(topology_error)
    }

    fn closure(&self, py: Python<'_>) -> PyResult<Self> {
        self.subset
            .closure()
            .map(|subset| Self {
                complex: self.complex.clone_ref(py),
                subset,
            })
            .map_err(topology_error)
    }

    fn star(&self, py: Python<'_>) -> PyResult<Self> {
        self.subset
            .star()
            .map(|subset| Self {
                complex: self.complex.clone_ref(py),
                subset,
            })
            .map_err(topology_error)
    }

    fn link(&self, py: Python<'_>) -> PyResult<Self> {
        self.subset
            .link()
            .map(|subset| Self {
                complex: self.complex.clone_ref(py),
                subset,
            })
            .map_err(topology_error)
    }

    fn is_pure(&self, degree: isize) -> PyResult<bool> {
        self.subset
            .is_pure(topology_degree(degree)?)
            .map_err(topology_error)
    }

    fn same_members(&self, other: &Self) -> PyResult<bool> {
        self.subset
            .same_members(&other.subset)
            .map_err(topology_error)
    }
}

#[pyclass(
    name = "Selection",
    frozen,
    module = "polygeo.topology",
    skip_from_py_object
)]
pub(crate) struct NativeSelection {
    complex: Py<NativeComplex>,
    pub(crate) selection: Arc<CanonicalSelection>,
}

#[pymethods]
impl NativeSelection {
    #[getter]
    fn topology(&self, py: Python<'_>) -> Py<NativeComplex> {
        self.complex.clone_ref(py)
    }

    #[getter]
    fn degree(&self) -> usize {
        self.selection.degree()
    }

    #[getter]
    fn size(&self) -> usize {
        self.selection.len()
    }

    fn indices_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        filled_array_1d(py, self.selection.len(), |output| {
            fill_indices::<i64>(self.selection.indices().iter().copied(), output)
        })
    }

    fn complement(&self, py: Python<'_>) -> PyResult<Self> {
        self.selection
            .complement()
            .map(|selection| Self {
                complex: self.complex.clone_ref(py),
                selection: Arc::new(selection),
            })
            .map_err(topology_error)
    }

    fn same_selection(&self, other: &Self) -> PyResult<bool> {
        self.selection
            .same_selection(&other.selection)
            .map_err(topology_error)
    }
}

#[pyfunction]
fn topological_boundary(complex: &Bound<'_, NativeComplex>) -> PyResult<NativeSubset> {
    NativeComplex::boundary_subset(complex)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeComplex>()?;
    module.add_class::<NativeSubset>()?;
    module.add_class::<NativeSelection>()?;
    module.add_function(wrap_pyfunction!(topological_boundary, module)?)?;
    module
        .getattr("topological_boundary")?
        .setattr("__module__", "polygeo.topology")?;
    crate::halfedge::register(module)?;
    let topology_error = module.py().get_type::<SimplicialError>();
    topology_error.setattr("__module__", "polygeo.topology")?;
    install_topology_error_properties(module.py(), &topology_error)?;
    module.add("SimplicialError", topology_error)?;
    let halfedge_error = module.py().get_type::<HalfedgeError>();
    halfedge_error.setattr("__module__", "polygeo.topology")?;
    module.add("HalfedgeError", halfedge_error)
}
