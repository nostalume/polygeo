#[pyclass(name = "HalfedgeSurface", frozen, module = "polygeo", skip_from_py_object)]
struct NativeHalfedgeSurface {
    owner: Arc<HalfedgeSurfaceCore>,
}

fn halfedge_degree(degree: isize) -> PyResult<usize> {
    usize::try_from(degree).map_err(|_| {
        Python::attach(|py| {
            halfedge_exception(
                py,
                "degree_outside",
                "degree is outside the surface",
                PyDict::new(py).unbind(),
            )
        })
    })
}

impl NativeHalfedgeSurface {
    fn relation(
        &self,
        py: Python<'_>,
        project: impl Fn(polygeo_core::Halfedge<'_>) -> usize,
    ) -> PyResult<Py<PyAny>> {
        filled_array_1d(py, self.owner.halfedge_count(), |output| {
            fill_indices::<i64>(self.owner.halfedges().map(project), output)
        })
    }

    fn boundary_parts(&self, py: Python<'_>, degree: isize) -> PyResult<PyBoundaryParts> {
        project_boundary(
            py,
            self.owner
                .chain_view()
                .boundary(halfedge_degree(degree)?)
                .map_err(halfedge_topology_error)?,
        )
    }
}

#[pymethods]
impl NativeHalfedgeSurface {
    #[staticmethod]
    #[pyo3(signature = (complex, *, limit=None))]
    fn from_complex(
        py: Python<'_>,
        complex: &Bound<'_, PyAny>,
        limit: Option<PyRef<'_, PyChainLawLimit>>,
    ) -> PyResult<(Py<Self>, Py<NativeSurfaceCorrespondence>)> {
        let complex = complex.extract::<Py<NativeComplex>>().map_err(|_| {
            halfedge_exception(
                py,
                "owner",
                "conversion requires a simplicial complex",
                PyDict::new(py).unbind(),
            )
        })?;
        let owner = Arc::clone(&complex.borrow(py).owner);
        let limit = limit.map_or(PyChainLawLimit::DEFAULT, |value| *value).core();
        let (owner, correspondence) = py
            .detach(move || HalfedgeSurfaceCore::from_complex_with_limit(&owner, limit))
            .map_err(halfedge_isomorphism_error)?;
        let surface = Py::new(py, Self { owner })?;
        let witness = Py::new(
            py,
            NativeSurfaceCorrespondence {
                source: complex.into_any(),
                target: surface.clone_ref(py).into_any(),
                correspondence,
            },
        )?;
        Ok((surface, witness))
    }

    #[staticmethod]
    #[pyo3(signature = (next, twin, *, exterior_faces=None))]
    fn from_permutations(
        py: Python<'_>,
        next: &Bound<'_, PyAny>,
        twin: &Bound<'_, PyAny>,
        exterior_faces: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let exterior_faces = match exterior_faces {
            Some(values) => copy_halfedge_indices(values)?,
            None => Vec::new().into_boxed_slice(),
        };
        let input = HalfedgeInput::native(
            copy_halfedge_indices(next)?,
            copy_halfedge_indices(twin)?,
            exterior_faces,
        )
        .map_err(halfedge_topology_error)?;
        py.detach(move || HalfedgeSurfaceCore::admit(input))
            .map(|owner| Self { owner })
            .map_err(halfedge_topology_error)
    }

    #[getter]
    fn halfedge_count(&self) -> usize { self.owner.halfedge_count() }
    #[getter]
    fn vertex_count(&self) -> usize { self.owner.vertex_count() }
    #[getter]
    fn edge_count(&self) -> usize { self.owner.edge_count() }
    #[getter]
    fn face_orbit_count(&self) -> usize { self.owner.face_orbit_count() }
    #[getter]
    fn material_face_count(&self) -> usize { self.owner.material_face_count() }
    #[getter]
    fn exterior_face_count(&self) -> usize { self.owner.exterior_face_count() }
    #[getter]
    fn boundary_component_count(&self) -> usize { self.owner.boundary_component_count() }
    #[getter]
    fn connected_component_count(&self) -> usize { self.owner.connected_component_count() }
    #[getter]
    fn euler_characteristic(&self) -> i64 { self.owner.euler_characteristic() }
    #[getter]
    fn genus(&self) -> Option<usize> { self.owner.genus() }

    fn next(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.relation(py, |halfedge| halfedge.next().index())
    }
    fn twin(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.relation(py, |halfedge| halfedge.twin().index())
    }
    fn vertex_of(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.relation(py, |halfedge| halfedge.vertex().index())
    }
    fn edge_of(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.relation(py, |halfedge| halfedge.edge().index())
    }
    fn face_of(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.relation(py, |halfedge| halfedge.face_orbit().index())
    }

    fn boundary_cycles(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let exterior_len = self.owner.face_orbits()
            .filter(|face| face.kind() == FaceKind::Exterior)
            .map(|face| face.halfedges().len()).sum();
        let offsets = filled_array_1d(py, self.owner.exterior_face_count() + 1, |output| {
            let mut running = 0_i64;
            output[0] = running;
            for (target, face) in output[1..].iter_mut().zip(
                self.owner.face_orbits().filter(|face| face.kind() == FaceKind::Exterior),
            ) {
                running += i64::try_from(face.halfedges().len())
                    .map_err(|_| TopologyError::IndexOverflow)?;
                *target = running;
            }
            Ok(())
        })?;
        let exterior = filled_array_1d(py, exterior_len, |output| {
            fill_indices::<i64>(
                self.owner.face_orbits()
                    .filter(|face| face.kind() == FaceKind::Exterior)
                    .flat_map(|face| face.halfedges().map(polygeo_core::Halfedge::index)),
                output,
            )
        })?;
        let material = filled_array_1d(py, exterior_len, |output| {
            fill_indices::<i64>(
                self.owner.face_orbits()
                    .filter(|face| face.kind() == FaceKind::Exterior)
                    .flat_map(|face| face.halfedges().map(|halfedge| halfedge.twin().index())),
                output,
            )
        })?;
        Ok((offsets, exterior, material))
    }

    fn boundary_matrix(&self, py: Python<'_>, degree: isize) -> PyResult<Py<PyAny>> {
        let (data, indices, indptr, shape) = self.boundary_parts(py, degree)?;
        let keywords = PyDict::new(py);
        keywords.set_item("shape", shape)?;
        keywords.set_item("copy", false)?;
        PyModule::import(py, "scipy.sparse")?.getattr("csr_array")?
            .call(((data, indices, indptr),), Some(&keywords)).map(Bound::unbind)
    }

    fn chain_complex(&self) -> NativeChainComplex {
        NativeChainComplex { inner: ExactComplex::Integer(self.owner.chain_complex()) }
    }

    #[pyo3(signature = (*, limit=None))]
    fn to_complex(
        slf: &Bound<'_, Self>,
        limit: Option<PyRef<'_, PyChainLawLimit>>,
    ) -> PyResult<(Py<NativeComplex>, Py<NativeSurfaceCorrespondence>)> {
        let py = slf.py();
        let owner = Arc::clone(&slf.borrow().owner);
        let limit = limit.map_or(PyChainLawLimit::DEFAULT, |value| *value).core();
        let (owner, correspondence) = py.detach(move || owner.to_complex_with_limit(limit))
            .map_err(halfedge_isomorphism_error)?;
        let complex = Py::new(py, NativeComplex { owner })?;
        let witness = Py::new(py, NativeSurfaceCorrespondence {
            source: slf.clone().into_any().unbind(),
            target: complex.clone_ref(py).into_any(),
            correspondence,
        })?;
        Ok((complex, witness))
    }
}

#[pyclass(name = "SurfaceCorrespondence", frozen, module = "polygeo", skip_from_py_object)]
struct NativeSurfaceCorrespondence {
    source: Py<PyAny>,
    target: Py<PyAny>,
    correspondence: SurfaceCorrespondence,
}

#[pymethods]
impl NativeSurfaceCorrespondence {
    #[getter]
    fn source(&self, py: Python<'_>) -> Py<PyAny> { self.source.clone_ref(py) }
    #[getter]
    fn target(&self, py: Python<'_>) -> Py<PyAny> { self.target.clone_ref(py) }
    #[getter]
    fn direction(&self) -> &'static str {
        match self.correspondence.direction() {
            CorrespondenceDirection::ComplexToSurface => "complex_to_surface",
            CorrespondenceDirection::SurfaceToComplex => "surface_to_complex",
        }
    }

    fn signed_permutation(&self, py: Python<'_>, degree: isize) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let permutation = self.correspondence.permutation(halfedge_degree(degree)?)
            .map_err(halfedge_topology_error)?;
        let targets = filled_array_1d(py, permutation.len(), |output| {
            fill_indices::<i64>(permutation.target_of_source().iter().copied(), output)
        })?;
        let signs = filled_array_1d(py, permutation.len(), |output| {
            output.copy_from_slice(permutation.signs());
            Ok(())
        })?;
        Ok((targets, signs))
    }

    fn chain_isomorphism(&self) -> PyChainIsomorphism {
        PyChainIsomorphism { relation: self.correspondence.isomorphism().clone() }
    }
}
