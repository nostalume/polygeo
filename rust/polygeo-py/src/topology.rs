#[pyclass(name = "Complex", frozen, module = "polygeo", skip_from_py_object)]
struct NativeComplex {
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
            "chain" => form::Space::Chain(
                polygeo_core::Binary64ChainSpace::selected(selection)
                    .map_err(topology_error)?,
            ),
            "cochain" => form::Space::Cochain(
                polygeo_core::Binary64CochainSpace::selected(selection)
                    .map_err(topology_error)?,
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
            .map(polygeo_core::Basis::row_count)
            .map_err(topology_error)
    }

    fn orientations(&self, py: Python<'_>, degree: isize) -> PyResult<Py<PyAny>> {
        let orientation = self
            .owner
            .orientation(topology_degree(degree)?)
            .map_err(topology_error)?;
        filled_array_1d(py, orientation.len(), |output| {
            output.copy_from_slice(orientation);
            Ok(())
        })
    }

    fn simplices(&self, py: Python<'_>, degree: isize) -> PyResult<Py<PyAny>> {
        let basis = self
            .owner
            .basis(topology_degree(degree)?)
            .map_err(topology_error)?;
        filled_array_2d(py, basis.row_count(), basis.row_width(), |output| {
            fill_indices::<i64>(basis.values().iter().copied(), output)
        })
    }

    fn boundary_parts(&self, py: Python<'_>, degree: isize) -> PyResult<PyBoundaryParts> {
        project_boundary(
            py,
            self.owner
                .chain_view()
                .boundary(topology_degree(degree)?)
                .map_err(topology_error)?,
        )
    }

    fn boundary_matrix(&self, py: Python<'_>, degree: isize) -> PyResult<Py<PyAny>> {
        let (data, indices, indptr, shape) = self.boundary_parts(py, degree)?;
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
        slf.owner.require_without_boundary().map_err(topology_error)?;
        Ok(slf)
    }

    fn require_triangle(&self) -> PyResult<()> {
        self.owner.require_triangle().map(|_| ()).map_err(topology_error)
    }

    fn require_regular(&self) -> PyResult<()> {
        self.owner.require_regular().map(|_| ()).map_err(topology_error)
    }

    fn require_oriented(&self) -> PyResult<()> {
        self.owner.require_oriented().map(|_| ()).map_err(topology_error)
    }

    fn require_connected(&self) -> PyResult<()> {
        self.owner.require_connected().map(|_| ()).map_err(topology_error)
    }

    fn require_with_boundary(&self) -> PyResult<()> {
        self.owner.require_with_boundary().map(|_| ()).map_err(topology_error)
    }

    fn require_without_boundary(&self) -> PyResult<()> {
        self.owner.require_without_boundary().map(|_| ()).map_err(topology_error)
    }

    fn boundary_mask(&self, py: Python<'_>, degree: isize) -> PyResult<Py<PyAny>> {
        let degree = topology_degree(degree)?;
        let count = self
            .owner
            .basis(degree)
            .map(polygeo_core::Basis::row_count)
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
                polygeo_core::Binary64ChainSpace::full(
                    self.owner.clone(),
                    topology_degree(degree)?,
                )
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
                polygeo_core::Binary64CochainSpace::full(
                    self.owner.clone(),
                    topology_degree(degree)?,
                )
                    .map_err(topology_error)?,
            ),
        })
    }

    fn integral_dual_cycle_basis(&self) -> PyResult<surface::PyIntegralDualCycleBasis> {
        Ok(surface::PyIntegralDualCycleBasis {
            inner: self
                .owner
                .integral_dual_cycle_basis()
                .map_err(topology_error)?,
        })
    }
}

#[pyclass(name = "SimplexSubset", frozen, module = "polygeo", skip_from_py_object)]
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
    fn complex(&self, py: Python<'_>) -> Py<NativeComplex> {
        self.complex.clone_ref(py)
    }

    fn mask(&self, py: Python<'_>, degree: isize) -> PyResult<Py<PyAny>> {
        let degree = topology_degree(degree)?;
        let count = self
            .subset
            .owner()
            .basis(degree)
            .map(polygeo_core::Basis::row_count)
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

#[pyclass(name = "SimplexSelection", frozen, module = "polygeo", skip_from_py_object)]
pub(crate) struct NativeSelection {
    complex: Py<NativeComplex>,
    pub(crate) selection: Arc<CanonicalSelection>,
}

#[pymethods]
impl NativeSelection {
    #[getter]
    fn complex(&self, py: Python<'_>) -> Py<NativeComplex> {
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

    fn indices(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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
