//! Registration-only entry point for the private `PyO3` extension.

mod form;
mod homology;
mod realization;
mod solve;
mod surface;

include!("shared.rs");
include!("topology.rs");
include!("halfedge.rs");
include!("chain.rs");
include!("registration.rs");

#[pymodule]
fn _polygeo_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register(module)
}
