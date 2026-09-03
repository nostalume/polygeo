//! Private native extension backing the contextual `polygeo` modules.

mod array;
mod chain;
mod form;
mod halfedge;
mod homology;
mod realization;
mod solve;
mod surface;
mod topology;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

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

fn domain<'py>(
    py: Python<'py>,
    root: &Bound<'py, PyModule>,
    name: &str,
) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(py, name)?;
    root.add_submodule(&module)?;
    Ok(module)
}

#[pymodule]
fn _polygeo_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let topology = domain(py, module, "topology")?;
    crate::topology::register(&topology)?;
    let chain = domain(py, module, "chain")?;
    crate::chain::register(&chain)?;
    let form = domain(py, module, "form")?;
    crate::form::register(&form)?;
    let geometry = domain(py, module, "geometry")?;
    crate::realization::register(&geometry)?;
    crate::surface::register_geometry(&geometry)?;
    crate::solve::register_geometry(&geometry)?;
    let solve = domain(py, module, "solve")?;
    crate::solve::register_solve(&solve)?;
    let field = domain(py, module, "field")?;
    crate::solve::register_field(&field)?;
    crate::surface::register_field(&field)
}
