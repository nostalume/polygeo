use std::sync::Arc;

use polygeo_core::topology::{
    FaceKind, Halfedge, HalfedgeInput, HalfedgeSurface as HalfedgeSurfaceCore, TopologyError,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};

use crate::array::{fill_indices, filled_array_1d};
use crate::chain::{ExactComplex, NativeChainComplex, PyChainIsomorphism, PyChainLawLimit};
use crate::topology::{
    NativeComplex, PyBoundaryParts, copy_halfedge_indices, halfedge_exception,
    halfedge_isomorphism_error, halfedge_topology_error, project_boundary,
};

#[pyclass(
    name = "HalfedgeSurface",
    frozen,
    module = "polygeo.topology",
    skip_from_py_object
)]
pub(crate) struct NativeHalfedgeSurface {
    owner: Arc<HalfedgeSurfaceCore>,
}

pub(crate) fn halfedge_degree(degree: isize) -> PyResult<usize> {
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
        project: impl Fn(Halfedge<'_>) -> usize,
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
    ) -> PyResult<(Py<Self>, PyChainIsomorphism)> {
        let complex = complex.extract::<Py<NativeComplex>>().map_err(|_| {
            halfedge_exception(
                py,
                "owner",
                "conversion requires a simplicial complex",
                PyDict::new(py).unbind(),
            )
        })?;
        let owner = Arc::clone(&complex.borrow(py).owner);
        let limit = limit
            .map_or(PyChainLawLimit::DEFAULT, |value| *value)
            .core();
        let (owner, correspondence) = py
            .detach(move || HalfedgeSurfaceCore::from_complex_with_limit(&owner, limit))
            .map_err(halfedge_isomorphism_error)?;
        let surface = Py::new(py, Self { owner })?;
        Ok((
            surface,
            PyChainIsomorphism {
                relation: correspondence,
            },
        ))
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
    fn halfedge_count(&self) -> usize {
        self.owner.halfedge_count()
    }
    #[getter]
    fn vertex_count(&self) -> usize {
        self.owner.vertex_count()
    }
    #[getter]
    fn edge_count(&self) -> usize {
        self.owner.edge_count()
    }
    #[getter]
    fn face_orbit_count(&self) -> usize {
        self.owner.face_orbit_count()
    }
    #[getter]
    fn material_face_count(&self) -> usize {
        self.owner.material_face_count()
    }
    #[getter]
    fn exterior_face_count(&self) -> usize {
        self.owner.exterior_face_count()
    }
    #[getter]
    fn boundary_component_count(&self) -> usize {
        self.owner.boundary_component_count()
    }
    #[getter]
    fn connected_component_count(&self) -> usize {
        self.owner.connected_component_count()
    }
    #[getter]
    fn euler_characteristic(&self) -> i64 {
        self.owner.euler_characteristic()
    }
    #[getter]
    fn genus(&self) -> Option<usize> {
        self.owner.genus()
    }

    fn next_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.relation(py, |halfedge| halfedge.next().index())
    }
    fn twin_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.relation(py, |halfedge| halfedge.twin().index())
    }
    fn vertex_of_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.relation(py, |halfedge| halfedge.vertex().index())
    }
    fn edge_of_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.relation(py, |halfedge| halfedge.edge().index())
    }
    fn face_of_numpy_copy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.relation(py, |halfedge| halfedge.face_orbit().index())
    }

    fn boundary_cycles_numpy_copy(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let exterior_len = self
            .owner
            .face_orbits()
            .filter(|face| face.kind() == FaceKind::Exterior)
            .map(|face| face.halfedges().len())
            .sum();
        let offsets = filled_array_1d(py, self.owner.exterior_face_count() + 1, |output| {
            let mut running = 0_i64;
            output[0] = running;
            for (target, face) in output[1..].iter_mut().zip(
                self.owner
                    .face_orbits()
                    .filter(|face| face.kind() == FaceKind::Exterior),
            ) {
                running += i64::try_from(face.halfedges().len())
                    .map_err(|_| TopologyError::IndexOverflow)?;
                *target = running;
            }
            Ok(())
        })?;
        let exterior = filled_array_1d(py, exterior_len, |output| {
            fill_indices::<i64>(
                self.owner
                    .face_orbits()
                    .filter(|face| face.kind() == FaceKind::Exterior)
                    .flat_map(|face| face.halfedges().map(Halfedge::index)),
                output,
            )
        })?;
        let material = filled_array_1d(py, exterior_len, |output| {
            fill_indices::<i64>(
                self.owner
                    .face_orbits()
                    .filter(|face| face.kind() == FaceKind::Exterior)
                    .flat_map(|face| face.halfedges().map(|halfedge| halfedge.twin().index())),
                output,
            )
        })?;
        Ok((offsets, exterior, material))
    }

    fn boundary_scipy_copy(&self, py: Python<'_>, degree: isize) -> PyResult<Py<PyAny>> {
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

    #[pyo3(signature = (*, limit=None))]
    fn to_complex(
        slf: &Bound<'_, Self>,
        limit: Option<PyRef<'_, PyChainLawLimit>>,
    ) -> PyResult<(Py<NativeComplex>, PyChainIsomorphism)> {
        let py = slf.py();
        let owner = Arc::clone(&slf.borrow().owner);
        let limit = limit
            .map_or(PyChainLawLimit::DEFAULT, |value| *value)
            .core();
        let (owner, correspondence) = py
            .detach(move || owner.to_complex_with_limit(limit))
            .map_err(halfedge_isomorphism_error)?;
        let complex = Py::new(py, NativeComplex { owner })?;
        Ok((
            complex,
            PyChainIsomorphism {
                relation: correspondence,
            },
        ))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeHalfedgeSurface>()
}
