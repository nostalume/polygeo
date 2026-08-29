#[pyclass(name = "ChainIsomorphism", frozen, module = "polygeo")]
struct PyChainIsomorphism {
    relation: CoreChainIsomorphism<IntegerRing>,
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
}

#[derive(Clone)]
enum ExactComplex {
    Integer(ChainComplex<IntegerRing>),
    Rational(ChainComplex<RationalField>),
}

#[pyclass(name = "ChainComplex", frozen, module = "polygeo")]
struct NativeChainComplex {
    inner: ExactComplex,
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

#[pyclass(name = "CochainComplex", frozen, module = "polygeo")]
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

#[pyclass(name = "Space", frozen, module = "polygeo")]
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

fn bigint_tuple<'py, 'a>(
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
enum ExactElement {
    IntegerChain(polygeo_core::Element<IntegerRing, Chain, BigIntEncoding>),
    IntegerCochain(polygeo_core::Element<IntegerRing, Cochain, BigIntEncoding>),
    RationalChain(polygeo_core::Element<RationalField, Chain, ReducedFractionEncoding>),
    RationalCochain(polygeo_core::Element<RationalField, Cochain, ReducedFractionEncoding>),
}

#[pyclass(name = "Element", frozen, module = "polygeo")]
struct NativeChainElement {
    inner: ExactElement,
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

#[derive(Clone)]
enum ExactMap {
    IntegerChain(LinearMap<IntegerRing, Chain, Chain>),
    IntegerCochain(LinearMap<IntegerRing, Cochain, Cochain>),
    RationalChain(LinearMap<RationalField, Chain, Chain>),
    RationalCochain(LinearMap<RationalField, Cochain, Cochain>),
}

#[pyclass(name = "LinearMap", frozen, module = "polygeo")]
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

#[pyclass(name = "CsrRepresentation", frozen, module = "polygeo")]
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

#[pyclass(name = "IntegerCsrParts", frozen, module = "polygeo")]
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

#[pyclass(name = "RationalCsrParts", frozen, module = "polygeo")]
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
    representation: &CsrRepresentation<
        IntegerRing,
        impl polygeo_core::Variance,
        impl polygeo_core::Variance,
        BigIntEncoding,
    >,
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
        impl polygeo_core::Variance,
        impl polygeo_core::Variance,
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
    S: polygeo_core::Variance,
    T: polygeo_core::Variance,
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
