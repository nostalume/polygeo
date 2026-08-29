pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyIntegerRing>()?;
    module.add_class::<PyRationalField>()?;
    module.add_class::<PyChainVariance>()?;
    module.add_class::<PyCochainVariance>()?;
    module.add_class::<PyBigIntEncoding>()?;
    module.add_class::<PyReducedFractionEncoding>()?;
    module.add_class::<NativeComplex>()?;
    module.add_class::<NativeHalfedgeSurface>()?;
    module.add_class::<NativeSurfaceCorrespondence>()?;
    module.add_class::<PyChainIsomorphism>()?;
    module.add_class::<NativeSubset>()?;
    module.add_class::<NativeSelection>()?;
    module.add_function(wrap_pyfunction!(topological_boundary, module)?)?;
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
        "DEFAULT_CHAIN_LAW_LIMIT",
        Py::new(module.py(), PyChainLawLimit::DEFAULT)?,
    )?;
    homology::register(module)?;
    form::register(module)?;
    realization::register(module)?;
    solve::register(module)?;
    surface::register(module)?;
    let topology_error = module.py().get_type::<SimplicialError>();
    topology_error.setattr("__module__", "polygeo")?;
    install_topology_error_properties(module.py(), &topology_error)?;
    module.add("SimplicialError", topology_error)?;
    let halfedge_error = module.py().get_type::<HalfedgeError>();
    halfedge_error.setattr("__module__", "polygeo")?;
    module.add("HalfedgeError", halfedge_error)?;
    let chain_error = module.py().get_type::<ChainError>();
    chain_error.setattr("__module__", "polygeo")?;
    module.add("ChainError", chain_error)?;
    Ok(())
}
