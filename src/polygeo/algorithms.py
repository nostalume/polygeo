from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Protocol, final

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from .geometry import (
    Geometry,
    GeometryError,
    _all_dual_measures_with_signs,
    _GeometryDomain,
)
from .numerics import (
    binary64_from_lattice,
    binary64_lattice,
    binary64_ratio,
    binary64_sum_product_lattice,
)
from .operators import (
    LinearMap,
    OperatorError,
    _codifferential_matrix,
    _hodge_laplacian_from_weights,
    _hodge_weights_from_measures,
    _MetricOperatorDomain,
    _OperatorDomain,
)
from .simplicial import (
    ORDINARY_FORM,
    CochainSpace,
    CochainSubspace,
    CodimensionOneRegular,
    Complex,
    Connected,
    FieldSemantics,
    Form,
    OrdinaryForm,
    OrientationState,
    WithBoundary,
    WithoutBoundary,
    topological_boundary,
)
from .solvers import (
    LinearSolution,
    NumericalError,
    PrepareLeastSquares,
    PrepareLinearSolve,
    _normalized_residual_evidence,
)
from .systems import AssembledSystem, SystemError, eliminate_dirichlet


class AlgorithmError(ValueError):
    """Raised when an algorithm capability or operation cannot be admitted."""


class _CoordinateBasis(Protocol):
    @property
    def dimension(self) -> int: ...


@dataclass(frozen=True, slots=True)
@final
class BasisCoordinates[Basis: _CoordinateBasis]:
    """Finite coordinates bound to one exact ordered basis product."""

    basis: Basis
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != self.basis.dimension:
            raise AlgorithmError("coordinate count does not match basis dimension")
        if not all(np.isfinite(value) for value in self.values):
            raise AlgorithmError("basis coordinates must be finite")


_REAL_HOMOLOGY_ADMISSION = object()


@final
class RealHomologyBasis[K: _OperatorDomain, Degree: int]:
    """Deterministic rational homology cycles bound to one exact complex."""

    __slots__ = ("_complex", "_cycles", "_degree", "_sealed")
    _complex: K
    _cycles: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    _degree: Degree
    _sealed: bool

    def __init__(
        self,
        complex_: K | None = None,
        degree: Degree | None = None,
        cycles: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] | None = None,
        *,
        _admission: object | None = None,
    ) -> None:
        if _admission is not _REAL_HOMOLOGY_ADMISSION:
            raise AlgorithmError(
                "RealHomologyBasis must be created by real_homology_basis()"
            )
        if complex_ is None or degree is None or cycles is None:
            raise AlgorithmError("real homology admission is incomplete")
        self._complex = complex_
        self._degree = degree
        self._cycles = cycles
        self._sealed = True

    @classmethod
    def _from_admitted(
        cls,
        complex_: K,
        degree: Degree,
        cycles: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    ) -> RealHomologyBasis[K, Degree]:
        return cls(complex_, degree, cycles, _admission=_REAL_HOMOLOGY_ADMISSION)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("real homology bases are immutable")
        super().__setattr__(name, value)

    @property
    def complex(self) -> K:
        return self._complex

    @property
    def degree(self) -> Degree:
        return self._degree

    @property
    def dimension(self) -> int:
        return len(self._cycles)

    def cycle_coefficients(self) -> csr_array:
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for column, (indices, coefficients) in enumerate(self._cycles):
            rows.extend(indices)
            columns.extend([column] * len(indices))
            values.extend(float(value) for value in coefficients)
        return csr_array(
            (values, (rows, columns)),
            shape=(self._complex.simplex_count(self._degree), self.dimension),
            dtype=np.float64,
        )

    def periods(
        self,
        form: Form[CochainSpace[K, Degree], OrdinaryForm],
    ) -> BasisCoordinates[RealHomologyBasis[K, Degree]]:
        if (
            not isinstance(form.space, CochainSpace)
            or form.space.degree != self._degree
        ):
            raise AlgorithmError("period form degree does not match homology degree")
        if form.space.complex is not self._complex:
            raise AlgorithmError("period form belongs to a different complex")
        if type(form.semantics) is not OrdinaryForm:
            raise AlgorithmError("periods require ordinary field semantics")
        coefficients = form.coefficients()
        periods: list[float] = []
        for indices, cycle in self._cycles:
            exact = binary64_sum_product_lattice(
                (float(value) for value in cycle),
                (float(coefficients[index]) for index in indices),
            )
            try:
                value = binary64_from_lattice(exact)
            except OverflowError as error:
                raise AlgorithmError("period is not representable") from error
            if not np.isfinite(value) or (value == 0.0 and exact != 0):
                raise AlgorithmError("period is not representable")
            periods.append(value)
        return BasisCoordinates(self, tuple(periods))


_EXACT_CELL_LIMIT = 200_000
_EXACT_BIT_LIMIT = 4_096
_EXACT_MEMORY_LIMIT = 64 * 1_024 * 1_024
_EXACT_OTHER_BYTES_PER_CELL = 16
_EXACT_FRACTION_BASE_BYTES = 256


def _require_exact_workspace(
    fraction_cells: int, other_cells: int, coefficient_bits: int
) -> None:
    if (
        coefficient_bits > _EXACT_BIT_LIMIT
        or fraction_cells + other_cells > _EXACT_CELL_LIMIT
    ):
        raise AlgorithmError("exact elimination resource limit exceeded")
    integer_bytes = 4 * ((coefficient_bits + 29) // 30)
    fraction_bytes = _EXACT_FRACTION_BASE_BYTES + 2 * integer_bytes
    other_bytes = _EXACT_OTHER_BYTES_PER_CELL + integer_bytes
    estimated = fraction_cells * fraction_bytes + other_cells * other_bytes
    if estimated > _EXACT_MEMORY_LIMIT:
        raise AlgorithmError("exact elimination resource limit exceeded")


def real_homology_basis[K: _OperatorDomain, Degree: int](
    complex_: K,
    degree: Degree,
) -> RealHomologyBasis[K, Degree]:
    """Construct bounded exact rational homology in canonical chain order."""
    if not isinstance(complex_, Complex):
        raise AlgorithmError("real homology requires a Complex")
    dimension = complex_.dimension
    if degree < 0 or degree > dimension:
        raise AlgorithmError("homology degree is outside the complex")
    columns = complex_.simplex_count(degree)
    rows = complex_.simplex_count(degree - 1) if degree else 0
    upper_columns = complex_.simplex_count(degree + 1) if degree < dimension else 0
    _require_exact_workspace(0, rows * columns + columns * upper_columns, 1)
    boundary = (
        np.zeros((rows, columns), dtype=np.int64)
        if degree == 0
        else complex_.boundary_matrix(degree).toarray().astype(np.int64)
    )
    upper = (
        np.zeros((columns, 0), dtype=np.int64)
        if degree == dimension
        else complex_.boundary_matrix(degree + 1).toarray().astype(np.int64)
    )

    boundary_rref, boundary_pivots = _exact_rref(boundary, other_cells=upper.size)
    free = [column for column in range(columns) if column not in boundary_pivots]
    kernel: list[list[int]] = []
    retained_bits = max(
        (
            max(abs(value.numerator).bit_length(), value.denominator.bit_length())
            for row in boundary_rref
            for value in row
        ),
        default=1,
    )
    for column in free:
        live_bits = max(retained_bits, _integer_vectors_bits(kernel))
        fraction_cells = boundary.size + columns
        other_cells = upper.size + boundary.size + (len(kernel) + 1) * columns
        _require_exact_workspace(fraction_cells, other_cells, live_bits)
        vector = [Fraction() for _ in range(columns)]
        vector[column] = Fraction(1)
        for row, pivot in enumerate(boundary_pivots):
            vector[pivot] = -boundary_rref[row][column]
        kernel.append(
            _primitive_cycle(
                vector,
                fraction_cells=fraction_cells,
                other_cells=other_cells,
                other_bits=live_bits,
            )
        )

    del boundary_rref
    kernel_bits = _integer_vectors_bits(kernel)
    upper_rref, image_pivots = _exact_rref(
        upper,
        other_cells=boundary.size + len(kernel) * columns,
        other_bits=kernel_bits,
    )
    del upper_rref
    image = [[int(value) for value in upper[:, column]] for column in image_pivots]
    combined_columns = image + kernel
    combined_cells = columns * len(combined_columns)
    combined_bits = _integer_vectors_bits(combined_columns)
    _require_exact_workspace(
        0,
        boundary.size + upper.size + 2 * combined_cells,
        combined_bits,
    )
    combined = np.asarray(combined_columns, dtype=object).T.reshape(
        columns, len(combined_columns)
    )
    del upper
    combined_rref, quotient_pivots = _exact_rref(
        combined,
        other_cells=boundary.size + combined_cells,
        other_bits=combined_bits,
    )
    del combined_rref
    quotient = [
        kernel[column - len(image)]
        for column in quotient_pivots
        if column >= len(image)
    ]
    expected = columns - len(boundary_pivots) - len(image_pivots)
    if len(quotient) != expected:
        raise AlgorithmError("exact homology quotient certification failed")
    _require_exact_cycles(boundary, quotient)
    retained = tuple(_retain_cycle(cycle) for cycle in quotient)
    return RealHomologyBasis._from_admitted(complex_, degree, retained)


def _exact_rref(
    values: NDArray[np.int64] | NDArray[np.object_],
    *,
    other_cells: int = 0,
    other_bits: int = 1,
) -> tuple[list[list[Fraction]], list[int]]:
    rows, columns = values.shape
    cells = rows * columns
    initial_bits = max(
        max((abs(int(value)).bit_length() for value in values.flat), default=1),
        other_bits,
    )
    fraction_cells = cells + columns
    retained_cells = other_cells + cells
    _require_exact_workspace(fraction_cells, retained_cells, initial_bits)

    def admit(value: Fraction) -> Fraction:
        bits = max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        _require_exact_workspace(fraction_cells, retained_cells, max(bits, other_bits))
        return value

    matrix = [[Fraction(int(value)) for value in row] for row in values.tolist()]
    pivots: list[int] = []
    pivot_row = 0
    for column in range(columns):
        selected = next(
            (row for row in range(pivot_row, rows) if matrix[row][column]), None
        )
        if selected is None:
            continue
        matrix[pivot_row], matrix[selected] = matrix[selected], matrix[pivot_row]
        pivot = matrix[pivot_row][column]
        matrix[pivot_row] = [admit(value / pivot) for value in matrix[pivot_row]]
        active_rows = (
            row for row in range(rows) if row != pivot_row and matrix[row][column]
        )
        for row in active_rows:
            factor = matrix[row][column]
            matrix[row] = [
                admit(value - factor * source)
                for value, source in zip(matrix[row], matrix[pivot_row], strict=True)
            ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    return matrix, pivots


def _integer_vectors_bits(vectors: list[list[int]]) -> int:
    return max(
        (abs(value).bit_length() for vector in vectors for value in vector), default=1
    )


def _primitive_cycle(
    vector: list[Fraction],
    *,
    fraction_cells: int = 0,
    other_cells: int = 0,
    other_bits: int = 1,
) -> list[int]:
    denominator = 1
    for value in vector:
        denominator = math.lcm(denominator, value.denominator)
        _require_exact_workspace(
            fraction_cells,
            other_cells,
            max(denominator.bit_length(), other_bits),
        )
    integers: list[int] = []
    for value in vector:
        integer = value.numerator * (denominator // value.denominator)
        _require_exact_workspace(
            fraction_cells,
            other_cells,
            max(abs(integer).bit_length(), other_bits),
        )
        integers.append(integer)
    divisor = math.gcd(*integers)
    integers = [value // divisor for value in integers]
    first = next(value for value in integers if value)
    if first < 0:
        integers = [-value for value in integers]
    return integers


def _require_exact_cycles(boundary: NDArray[np.int64], cycles: list[list[int]]) -> None:
    if any(
        sum(
            int(boundary[row, column]) * cycle[column]
            for column in range(boundary.shape[1])
        )
        != 0
        for cycle in cycles
        for row in range(boundary.shape[0])
    ):
        raise AlgorithmError("exact homology closure certification failed")


def _retain_cycle(cycle: list[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    indices = tuple(index for index, value in enumerate(cycle) if value)
    values = tuple(cycle[index] for index in indices)
    if any(
        not np.isfinite(float(value)) or int(float(value)) != value for value in values
    ):
        raise AlgorithmError("homology coefficients are not representable")
    return indices, values


class PositiveHodgeMetric[K: _MetricOperatorDomain]:
    """All-degree positivity evidence for one represented binary64 Hodge metric."""

    __slots__ = ("_geometry", "_sealed", "_weights")
    _geometry: Geometry[K]
    _sealed: bool
    _weights: tuple[NDArray[np.float64], ...]

    def __init__(self, geometry: Geometry[K]) -> None:
        if not isinstance(geometry, Geometry):
            raise AlgorithmError("positive Hodge metric requires a Geometry")
        try:
            dual_measures, signs = _all_dual_measures_with_signs(geometry)
            weights = tuple(
                _hodge_weights_from_measures(dual, geometry._measures[degree])
                for degree, dual in enumerate(dual_measures)
            )
        except (GeometryError, OperatorError, ValueError) as error:
            raise AlgorithmError("Hodge weights are not representable") from error
        if any(np.any(sign <= 0) for sign in signs) or any(
            not np.all(np.isfinite(weight)) or np.any(weight <= 0.0)
            for weight in weights
        ):
            raise AlgorithmError(
                "represented Hodge weights must be finite and strictly positive"
            )
        retained: list[NDArray[np.float64]] = []
        for weight in weights:
            owned = np.array(weight, dtype=np.float64, order="C", copy=True)
            owned.flags.writeable = False
            retained.append(owned)
        self._geometry = geometry
        self._weights = tuple(retained)
        self._sealed = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("positive Hodge metrics are immutable")
        super().__setattr__(name, value)

    @property
    def geometry(self) -> Geometry[K]:
        return self._geometry

    def weights(self, degree: int) -> NDArray[np.float64]:
        if degree < 0 or degree >= len(self._weights):
            raise AlgorithmError("Hodge weight degree is outside the complex")
        return self._weights[degree].copy()


def assemble_poisson[
    K: _MetricOperatorDomain,
    S: FieldSemantics,
](
    metric: PositiveHodgeMetric[K],
    density: Form[CochainSpace[K, Literal[0]], S],
) -> AssembledSystem[
    CochainSpace[K, Literal[0]],
    CochainSpace[K, Literal[0]],
    S,
]:
    """Assemble the positive degree-zero Hodge Laplacian with pointwise density."""
    if not isinstance(metric, PositiveHodgeMetric):
        raise AlgorithmError("Poisson assembly requires a positive Hodge metric")
    if not isinstance(density.space, CochainSpace) or density.space.degree != 0:
        raise AlgorithmError(
            "Poisson density must belong to a degree-zero cochain space"
        )
    if density.space.complex is not metric.geometry.complex:
        raise AlgorithmError("Poisson metric and density belong to a different complex")
    following = metric._weights[1] if metric.geometry.complex.dimension > 0 else None
    try:
        operator = _hodge_laplacian_from_weights(
            density.space,
            previous=None,
            current=metric._weights[0],
            following=following,
        )
    except OperatorError as error:
        raise AlgorithmError("Poisson operator is not representable") from error
    return AssembledSystem(operator, density)


@dataclass(frozen=True, slots=True)
@final
class ResidualEvidence:
    residual_norm: float
    scale: float
    limit: float

    def __post_init__(self) -> None:
        if not all(
            np.isfinite(value) and value >= 0.0
            for value in (self.residual_norm, self.scale)
        ):
            raise AlgorithmError("residual evidence must be finite and nonnegative")
        if not np.isfinite(self.limit) or self.limit <= 0.0:
            raise AlgorithmError("residual evidence limit must be finite and positive")
        if self.scale == 0.0 and self.residual_norm:
            raise AlgorithmError("zero residual scale requires zero residual")
        if self.scale and self.residual_norm / self.scale > self.limit:
            raise AlgorithmError("residual evidence exceeds its admitted limit")


@dataclass(frozen=True, slots=True)
@final
class Certified[Output, Evidence]:
    """One output paired with descriptive verification evidence."""

    output: Output
    evidence: Evidence


@final
class VertexMap[K: _GeometryDomain, TargetDimension: int]:
    """Canonical vertex correspondence between two complete geometries."""

    __slots__ = ("_sealed", "_source", "_target", "_target_dimension")
    _sealed: bool
    _source: Geometry[K]
    _target: Geometry[K]
    _target_dimension: TargetDimension

    def __init__(self) -> None:
        raise AlgorithmError("VertexMap must be created by vertex_map()")

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("vertex maps are immutable")
        object.__setattr__(self, name, value)

    @property
    def source(self) -> Geometry[K]:
        return self._source

    @property
    def target(self) -> Geometry[K]:
        return self._target

    @property
    def target_dimension(self) -> TargetDimension:
        return self._target_dimension


def vertex_map[K: _GeometryDomain, TargetDimension: int](
    source: Geometry[K],
    target: Geometry[K],
    target_dimension: TargetDimension,
) -> VertexMap[K, TargetDimension]:
    """Bind complete source and target geometries by canonical vertex identity."""
    if not isinstance(source, Geometry) or not isinstance(target, Geometry):
        raise AlgorithmError("vertex_map requires complete geometries")
    if source.complex is not target.complex:
        raise AlgorithmError("vertex map geometries must share one exact complex")
    if (
        type(target_dimension) is not int
        or target.ambient_dimension != target_dimension
    ):
        raise AlgorithmError("vertex map target dimension does not match geometry")
    admitted: VertexMap[K, TargetDimension] = object.__new__(VertexMap)
    object.__setattr__(admitted, "_source", source)
    object.__setattr__(admitted, "_target", target)
    object.__setattr__(admitted, "_target_dimension", target_dimension)
    object.__setattr__(admitted, "_sealed", True)
    return admitted


@dataclass(frozen=True, slots=True)
@final
class ConditionEvidence:
    """A descriptive condition-admission heuristic, not an error bound."""

    indicator: float
    limit: float

    def __post_init__(self) -> None:
        valid = (
            np.isfinite(self.indicator)
            and np.isfinite(self.limit)
            and 0.0 <= self.indicator <= self.limit
            and self.limit > 0.0
        )
        if not valid:
            raise AlgorithmError("condition evidence exceeds its admitted limit")


@dataclass(frozen=True, slots=True)
@final
class HodgeComponents[K: _MetricOperatorDomain, Degree: int]:
    exact: Form[CochainSpace[K, Degree], OrdinaryForm]
    coexact: Form[CochainSpace[K, Degree], OrdinaryForm]
    harmonic: Form[CochainSpace[K, Degree], OrdinaryForm]

    def __post_init__(self) -> None:
        components = (self.exact, self.coexact, self.harmonic)
        if not all(
            self.exact.space.same_space(component.space) for component in components[1:]
        ):
            raise AlgorithmError("Hodge components must share one exact cochain space")
        if any(
            type(component.semantics) is not OrdinaryForm for component in components
        ):
            raise AlgorithmError("Hodge components require ordinary field semantics")


@dataclass(frozen=True, slots=True)
@final
class HodgeEvidence:
    reconstruction: ResidualEvidence
    orthogonality: ResidualEvidence
    harmonic_closure: ResidualEvidence
    harmonic_coclosure: ResidualEvidence
    exact_condition: ConditionEvidence
    coexact_condition: ConditionEvidence


def hodge_decomposition[K: _MetricOperatorDomain, Degree: int](
    metric: PositiveHodgeMetric[K],
    form: Form[CochainSpace[K, Degree], OrdinaryForm],
    prepare: PrepareLeastSquares[
        CochainSubspace[CochainSpace[K, Degree]], CochainSpace[K, Degree]
    ],
) -> Certified[HodgeComponents[K, Degree], HodgeEvidence]:
    """Decompose one ordinary cochain into exact, coexact, and harmonic parts."""
    if not isinstance(metric, PositiveHodgeMetric):
        raise AlgorithmError("Hodge decomposition requires a positive Hodge metric")
    if not isinstance(form.space, CochainSpace):
        raise AlgorithmError("Hodge decomposition requires a primal cochain")
    if form.space.complex is not metric.geometry.complex:
        raise AlgorithmError("Hodge metric and form belong to a different complex")
    if type(form.semantics) is not OrdinaryForm:
        raise AlgorithmError("Hodge decomposition requires ordinary field semantics")
    space = form.space
    degree = space.degree
    values = form.coefficients()
    weights = metric._weights[degree]
    try:
        exact_image, coexact_image = _hodge_image_bases(metric, space)
        exact, exact_condition = _project_hodge_image(
            space, weights, values, exact_image, prepare
        )
        coexact, coexact_condition = _project_hodge_image(
            space, weights, values, coexact_image, prepare
        )
    except AlgorithmError:
        raise
    except (AttributeError, NumericalError, OperatorError, ValueError) as error:
        raise AlgorithmError("Hodge projection failed") from error
    with np.errstate(all="ignore"):
        harmonic = values - exact - coexact
    if not np.all(np.isfinite(harmonic)):
        raise AlgorithmError("Hodge harmonic remainder is not representable")

    components = HodgeComponents(
        space.form(exact, ORDINARY_FORM),
        space.form(coexact, ORDINARY_FORM),
        space.form(harmonic, ORDINARY_FORM),
    )
    complex_ = space.complex
    closure_matrix = (
        complex_.boundary_matrix(degree + 1).transpose().tocsr()
        if degree < complex_.dimension
        else csr_array((0, space.size), dtype=np.float64)
    )
    coclosure_matrix = (
        complex_.boundary_matrix(degree).tocsr()
        if degree > 0
        else csr_array((0, space.size), dtype=np.float64)
    )
    evidence = HodgeEvidence(
        _reconstruction_evidence(values, exact, coexact, harmonic),
        _weighted_orthogonality_evidence(weights, exact, coexact),
        _matrix_residual_evidence(closure_matrix, harmonic, reference=values),
        _matrix_residual_evidence(
            coclosure_matrix, harmonic, weights, reference=values
        ),
        exact_condition,
        coexact_condition,
    )
    return Certified(components, evidence)


def _hodge_image_bases[K: _MetricOperatorDomain, Degree: int](
    metric: PositiveHodgeMetric[K],
    space: CochainSpace[K, Degree],
) -> tuple[csr_array, csr_array]:
    complex_ = space.complex
    degree = space.degree
    exact = csr_array((space.size, 0), dtype=np.float64)
    coexact = csr_array((space.size, 0), dtype=np.float64)
    dense_cells = 0
    if degree > 0:
        derivative = complex_.boundary_matrix(degree).transpose().tocsr()
        dense_cells += derivative.shape[0] * derivative.shape[1]
        _require_exact_workspace(0, dense_cells, 1)
        _, pivots = _exact_rref(derivative.toarray().astype(np.int64))
        exact = csr_array(derivative[:, pivots], dtype=np.float64)
    if degree < complex_.dimension:
        upper = complex_.boundary_matrix(degree + 1).tocsr()
        dense_cells += upper.shape[0] * upper.shape[1]
        _require_exact_workspace(0, dense_cells, 1)
        _, pivots = _exact_rref(upper.toarray().astype(np.int64))
        full = _codifferential_matrix(
            upper.transpose().tocsr(),
            metric._weights[degree],
            metric._weights[degree + 1],
        )
        coexact = csr_array(full[:, pivots], dtype=np.float64)
    return exact, coexact


def _project_hodge_image[K: _MetricOperatorDomain, Degree: int](
    space: CochainSpace[K, Degree],
    weights: NDArray[np.float64],
    values: NDArray[np.float64],
    image: csr_array,
    prepare: PrepareLeastSquares[
        CochainSubspace[CochainSpace[K, Degree]], CochainSpace[K, Degree]
    ],
) -> tuple[NDArray[np.float64], ConditionEvidence]:
    limit = float(np.sqrt(np.finfo(np.float64).eps))
    if image.shape[1] == 0:
        return np.zeros(space.size, dtype=np.float64), ConditionEvidence(0.0, limit)
    scale = float(np.max(weights))
    with np.errstate(all="ignore"):
        normalized_weights = weights / scale
        roots = np.sqrt(normalized_weights)
        weighted_image = image.multiply(roots[:, None]).tocsr()
        weighted_values = roots * values
    if (
        not np.all(np.isfinite(roots))
        or np.any((weights != 0.0) & (normalized_weights == 0.0))
        or np.any((image.data != 0.0) & (weighted_image.data == 0.0))
        or np.any((values != 0.0) & (weighted_values == 0.0))
    ):
        raise AlgorithmError("Hodge weight normalization is not representable")
    coordinates = CochainSubspace(space, np.arange(image.shape[1], dtype=np.int64))
    operator = LinearMap(coordinates, space, weighted_image)
    solution = prepare(operator)(space.form(weighted_values, ORDINARY_FORM))
    if not solution.form.space.same_space(
        coordinates
    ) or not solution.equation_space.same_space(space):
        raise AlgorithmError("Hodge projection returned foreign solver evidence")
    with np.errstate(all="ignore"):
        component = np.asarray(image @ solution.form.coefficients()).reshape(-1)
    if not np.all(np.isfinite(component)):
        raise AlgorithmError("Hodge component is not representable")
    return component, ConditionEvidence(
        solution.condition_indicator, solution.condition_limit
    )


def _fraction_evidence(
    residuals: list[Fraction], scales: list[Fraction]
) -> ResidualEvidence:
    pairs = tuple(zip(residuals, scales, strict=True))
    if any(scale == 0 and residual for residual, scale in pairs):
        raise AlgorithmError("zero evidence scale has nonzero residual")
    ratios = [abs(residual) / scale for residual, scale in pairs if scale]
    limit = float(np.sqrt(np.finfo(np.float64).eps))
    if not ratios:
        return ResidualEvidence(0.0, 0.0, limit)
    exact = max(ratios)
    relative = float(exact)
    if relative == 0.0 and exact:
        relative = math.nextafter(0.0, math.inf)
    return ResidualEvidence(relative, 1.0, limit)


def _reconstruction_evidence(
    original: NDArray[np.float64],
    exact: NDArray[np.float64],
    coexact: NDArray[np.float64],
    harmonic: NDArray[np.float64],
) -> ResidualEvidence:
    residuals: list[Fraction] = []
    scales: list[Fraction] = []
    for values in zip(original, exact, coexact, harmonic, strict=True):
        terms = [Fraction(float(value)) for value in values]
        residuals.append(terms[0] - terms[1] - terms[2] - terms[3])
        scales.append(sum((abs(value) for value in terms), start=Fraction()))
    return _fraction_evidence(residuals, scales)


def _weighted_orthogonality_evidence(
    weights: NDArray[np.float64],
    exact: NDArray[np.float64],
    coexact: NDArray[np.float64],
) -> ResidualEvidence:
    terms = [
        Fraction(float(weight)) * Fraction(float(left)) * Fraction(float(right))
        for weight, left, right in zip(weights, exact, coexact, strict=True)
    ]
    return _fraction_evidence(
        [sum(terms, start=Fraction())],
        [sum((abs(term) for term in terms), start=Fraction())],
    )


def _matrix_residual_evidence(
    matrix: csr_array,
    values: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
    *,
    reference: NDArray[np.float64] | None = None,
) -> ResidualEvidence:
    residuals: list[Fraction] = []
    scales: list[Fraction] = []
    samples = (values,) if reference is None else (values, reference)
    for row in range(matrix.shape[0]):
        row_terms: list[list[Fraction]] = [[] for _ in samples]
        for offset in range(int(matrix.indptr[row]), int(matrix.indptr[row + 1])):
            column = int(matrix.indices[offset])
            coefficient = Fraction(float(matrix.data[offset]))
            if weights is not None:
                coefficient *= Fraction(float(weights[column]))
            for sample, terms in zip(samples, row_terms, strict=True):
                terms.append(coefficient * Fraction(float(sample[column])))
        residuals.append(sum(row_terms[0], start=Fraction()))
        scales.append(
            max(
                sum((abs(term) for term in terms), start=Fraction())
                for terms in row_terms
            )
        )
    return _fraction_evidence(residuals, scales)


_MEAN_ZERO_ADMISSION = object()


@final
class MeanZeroProblem[K: _MetricOperatorDomain]:
    """A compatible boundaryless Poisson problem with a weighted mean-zero gauge."""

    __slots__ = (
        "_compatibility_evidence",
        "_metric",
        "_operator",
        "_rhs",
        "_sealed",
    )
    _compatibility_evidence: ResidualEvidence
    _metric: PositiveHodgeMetric[K]
    _operator: LinearMap[CochainSpace[K, Literal[0]], CochainSpace[K, Literal[0]]]
    _rhs: Form[CochainSpace[K, Literal[0]], OrdinaryForm]
    _sealed: bool

    def __init__(
        self,
        metric: PositiveHodgeMetric[K] | None = None,
        system: AssembledSystem[
            CochainSpace[K, Literal[0]],
            CochainSpace[K, Literal[0]],
            OrdinaryForm,
        ]
        | None = None,
        evidence: ResidualEvidence | None = None,
        *,
        _admission: object | None = None,
    ) -> None:
        if _admission is not _MEAN_ZERO_ADMISSION:
            raise AlgorithmError(
                "MeanZeroProblem must be created by impose_mean_zero()"
            )
        if metric is None or system is None or evidence is None:
            raise AlgorithmError("mean-zero admission is incomplete")
        self._metric = metric
        self._operator = system.operator
        self._rhs = system.rhs
        self._compatibility_evidence = evidence
        self._sealed = True

    @classmethod
    def _from_admitted(
        cls,
        metric: PositiveHodgeMetric[K],
        system: AssembledSystem[
            CochainSpace[K, Literal[0]],
            CochainSpace[K, Literal[0]],
            OrdinaryForm,
        ],
        evidence: ResidualEvidence,
    ) -> MeanZeroProblem[K]:
        return cls(metric, system, evidence, _admission=_MEAN_ZERO_ADMISSION)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("mean-zero problems are immutable")
        super().__setattr__(name, value)

    @property
    def operator(
        self,
    ) -> LinearMap[CochainSpace[K, Literal[0]], CochainSpace[K, Literal[0]]]:
        return self._operator

    @property
    def rhs(self) -> Form[CochainSpace[K, Literal[0]], OrdinaryForm]:
        return self._rhs

    @property
    def compatibility_evidence(self) -> ResidualEvidence:
        return self._compatibility_evidence

    def solve(
        self,
        prepare: PrepareLinearSolve[
            CochainSubspace[CochainSpace[K, Literal[0]]],
            CochainSubspace[CochainSpace[K, Literal[0]]],
        ],
    ) -> LinearSolution[
        CochainSpace[K, Literal[0]],
        CochainSpace[K, Literal[0]],
        OrdinaryForm,
    ]:
        parent = self._rhs.space
        if parent.size == 1:
            zero = parent.form(np.zeros(1, dtype=np.float64), ORDINARY_FORM)
            return LinearSolution(zero, parent, 0.0, 0.0, 0.0)

        interior = CochainSubspace(parent, np.arange(1, parent.size, dtype=np.int64))
        reduced_operator, reduced_rhs = _normalized_anchor_system(
            self._metric,
            self._rhs.coefficients(),
            interior,
        )
        reduced_solution = prepare(reduced_operator)(reduced_rhs)
        if not reduced_solution.form.space.same_space(interior):
            raise AlgorithmError("anchored solver returned a foreign solution space")
        anchored = np.zeros(parent.size, dtype=np.float64)
        anchored[1:] = reduced_solution.form.coefficients()
        shifted = _subtract_weighted_mean(self._metric._weights[0], anchored)
        _require_weighted_condition(
            self._metric._weights[0], shifted, label="weighted gauge"
        )

        matrix = self._operator.matrix()
        residual_norm, residual_scale = _normalized_residual_evidence(
            matrix,
            tuple(binary64_ratio(float(value)) for value in matrix.data),
            shifted,
            self._rhs.coefficients(),
        )
        relative = (
            0.0
            if residual_norm == 0.0 and residual_scale == 0.0
            else residual_norm / residual_scale
        )
        return LinearSolution(
            parent.form(shifted, ORDINARY_FORM),
            parent,
            residual_norm,
            residual_scale,
            relative,
        )


def impose_mean_zero[
    O: OrientationState,
    T: CodimensionOneRegular,
](
    metric: PositiveHodgeMetric[Complex[WithoutBoundary, O, Connected, T]],
    density: Form[
        CochainSpace[Complex[WithoutBoundary, O, Connected, T], Literal[0]],
        OrdinaryForm,
    ],
) -> MeanZeroProblem[Complex[WithoutBoundary, O, Connected, T]]:
    """Admit a compatible closed connected scalar Poisson problem."""
    if not isinstance(metric, PositiveHodgeMetric):
        raise AlgorithmError("mean-zero Poisson requires a positive Hodge metric")
    complex_ = metric.geometry.complex
    if not isinstance(complex_.boundary_state, WithoutBoundary):
        raise AlgorithmError("mean-zero Poisson requires a boundaryless complex")
    if not isinstance(complex_.connectivity_state, Connected):
        raise AlgorithmError("mean-zero Poisson requires a connected complex")
    if not isinstance(complex_.topology_state, CodimensionOneRegular):
        raise AlgorithmError("mean-zero Poisson requires codimension-one regularity")
    if not isinstance(density.space, CochainSpace) or density.space.degree != 0:
        raise AlgorithmError("mean-zero density must belong to a degree-zero space")
    if density.space.complex is not complex_:
        raise AlgorithmError(
            "mean-zero metric and density belong to a different complex"
        )
    if type(density.semantics) is not OrdinaryForm:
        raise AlgorithmError("mean-zero Poisson requires ordinary field semantics")

    evidence = _compatibility_evidence(
        metric._weights[0], density.coefficients(), label="compatibility"
    )
    system = assemble_poisson(metric, density)
    return MeanZeroProblem._from_admitted(metric, system, evidence)


def harmonic_extension[
    O: OrientationState,
    T: CodimensionOneRegular,
    S: FieldSemantics,
](
    metric: PositiveHodgeMetric[Complex[WithBoundary, O, Connected, T]],
    boundary_values: Form[
        CochainSubspace[
            CochainSpace[Complex[WithBoundary, O, Connected, T], Literal[0]]
        ],
        S,
    ],
    prepare: PrepareLinearSolve[
        CochainSubspace[
            CochainSpace[Complex[WithBoundary, O, Connected, T], Literal[0]]
        ],
        CochainSubspace[
            CochainSpace[Complex[WithBoundary, O, Connected, T], Literal[0]]
        ],
    ],
) -> LinearSolution[
    CochainSpace[Complex[WithBoundary, O, Connected, T], Literal[0]],
    CochainSubspace[CochainSpace[Complex[WithBoundary, O, Connected, T], Literal[0]]],
    S,
]:
    """Extend canonical degree-zero boundary values by a discrete harmonic field."""
    if not isinstance(metric, PositiveHodgeMetric):
        raise AlgorithmError("harmonic extension requires a positive Hodge metric")
    complex_ = metric.geometry.complex
    if (
        not isinstance(complex_.boundary_state, WithBoundary)
        or not isinstance(complex_.connectivity_state, Connected)
        or not isinstance(complex_.topology_state, CodimensionOneRegular)
    ):
        raise AlgorithmError(
            "harmonic extension requires a connected regular complex with boundary"
        )
    if not isinstance(boundary_values, Form) or not isinstance(
        boundary_values.space, CochainSubspace
    ):
        raise AlgorithmError("harmonic extension requires canonical boundary values")
    parent = boundary_values.space.parent
    if not isinstance(parent, CochainSpace) or parent.degree != 0:
        raise AlgorithmError("harmonic extension requires degree-zero boundary values")
    if parent.complex is not complex_:
        raise AlgorithmError("harmonic extension values belong to a different complex")
    canonical = np.flatnonzero(topological_boundary(complex_).mask(0))
    if not np.array_equal(boundary_values.space.indices(), canonical):
        raise AlgorithmError("harmonic extension requires the canonical boundary")

    zero = parent.form(
        np.zeros(parent.size, dtype=np.float64), boundary_values.semantics
    )
    try:
        problem = eliminate_dirichlet(
            assemble_poisson(metric, zero),
            boundary_values.space,
            boundary_values,
        )
    except SystemError as error:
        raise AlgorithmError("harmonic extension assembly failed") from error
    if problem.interior.size == 0:
        empty = problem.interior.form(
            np.empty(0, dtype=np.float64), boundary_values.semantics
        )
        return LinearSolution(
            problem.reconstruct(empty), problem.interior, 0.0, 0.0, 0.0
        )
    try:
        solution = problem.solve(prepare)
    except Exception as error:
        raise AlgorithmError("harmonic extension solve failed") from error
    if not isinstance(solution, LinearSolution):
        raise AlgorithmError("harmonic extension solver returned malformed evidence")
    if not solution.equation_space.same_space(problem.interior):
        raise AlgorithmError(
            "harmonic extension solver returned a foreign equation space"
        )
    return solution


def _weighted_lattice(
    weights: NDArray[np.float64],
    values: NDArray[np.float64],
) -> tuple[int, int]:
    terms = tuple(
        binary64_sum_product_lattice((float(weight),), (float(value),))
        for weight, value in zip(weights, values, strict=True)
    )
    return sum(terms), sum(abs(term) for term in terms)


def _require_weighted_condition(
    weights: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    label: str,
) -> tuple[int, int, float]:
    exact, scale_exact = _weighted_lattice(weights, values)
    limit = float(np.sqrt(np.finfo(np.float64).eps))
    limit_numerator, limit_denominator = limit.as_integer_ratio()
    if scale_exact == 0:
        if exact != 0:
            raise AlgorithmError(f"{label} has zero scale but nonzero residual")
    elif abs(exact) * limit_denominator > scale_exact * limit_numerator:
        raise AlgorithmError(f"Poisson density is incompatible with {label}")
    return exact, scale_exact, limit


def _compatibility_evidence(
    weights: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    label: str,
) -> ResidualEvidence:
    exact, scale_exact, limit = _require_weighted_condition(
        weights, values, label=label
    )
    if scale_exact == 0:
        return ResidualEvidence(0.0, 0.0, limit)
    exact_ratio = Fraction(abs(exact), scale_exact)
    relative = float(exact_ratio)
    if relative == 0.0 and exact != 0:
        relative = math.nextafter(0.0, math.inf)
    elif Fraction(relative) > exact_ratio:
        relative = math.nextafter(relative, 0.0)
    return ResidualEvidence(relative, 1.0, limit)


def _normalized_anchor_system[K: _MetricOperatorDomain](
    metric: PositiveHodgeMetric[K],
    density: NDArray[np.float64],
    interior: CochainSubspace[CochainSpace[K, Literal[0]]],
) -> tuple[
    LinearMap[
        CochainSubspace[CochainSpace[K, Literal[0]]],
        CochainSubspace[CochainSpace[K, Literal[0]]],
    ],
    Form[CochainSubspace[CochainSpace[K, Literal[0]]], OrdinaryForm],
]:
    stiffness_lattice = tuple(
        binary64_lattice(float(value)) for value in metric._weights[1]
    )
    load_lattice = tuple(
        binary64_sum_product_lattice((float(weight),), (float(value),))
        for weight, value in zip(metric._weights[0], density, strict=True)
    )
    magnitude = max(
        (abs(value) for value in (*stiffness_lattice, *load_lattice)),
        default=0,
    )
    if magnitude == 0:
        raise AlgorithmError("anchored Poisson system has zero scale")

    normalized_weights = np.array(
        [_normalized_lattice_value(value, magnitude) for value in stiffness_lattice],
        dtype=np.float64,
    )
    derivative = (
        metric.geometry.complex.boundary_matrix(1)
        .transpose()
        .astype(np.float64)
        .tocsr()
    )
    with np.errstate(all="ignore"):
        weighted = derivative.multiply(normalized_weights[:, None])
        normalized = (derivative.transpose() @ weighted).tocsr()
    if not np.all(np.isfinite(normalized.data)):
        raise AlgorithmError("normalized Poisson stiffness is not representable")
    indices = interior.indices()
    reduced = csr_array(normalized[indices][:, indices])
    reduced_load = np.array(
        [
            _normalized_lattice_value(load_lattice[index], magnitude)
            for index in indices
        ],
        dtype=np.float64,
    )
    operator = LinearMap(interior, interior, reduced)
    return operator, interior.form(reduced_load, ORDINARY_FORM)


def _normalized_lattice_value(exact: int, magnitude: int) -> float:
    value = float(Fraction(exact, magnitude))
    if not np.isfinite(value) or (value == 0.0 and exact != 0):
        raise AlgorithmError("normalized Poisson coefficient is not representable")
    return value


def _subtract_weighted_mean(
    weights: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    numerator = binary64_sum_product_lattice(
        (float(weight) for weight in weights),
        (float(value) for value in values),
    )
    denominator = sum(binary64_lattice(float(weight)) for weight in weights)
    if denominator == 0:
        Fraction() / Fraction()
    try:
        mean = float(Fraction(numerator, denominator))
    except OverflowError as error:
        raise AlgorithmError("weighted gauge mean is not representable") from error
    with np.errstate(all="ignore"):
        shifted = values - mean
    if not np.all(np.isfinite(shifted)):
        raise AlgorithmError("weighted gauge shift is not representable")
    return shifted


__all__ = [
    "AlgorithmError",
    "BasisCoordinates",
    "Certified",
    "ConditionEvidence",
    "HodgeComponents",
    "HodgeEvidence",
    "MeanZeroProblem",
    "PositiveHodgeMetric",
    "RealHomologyBasis",
    "ResidualEvidence",
    "VertexMap",
    "assemble_poisson",
    "hodge_decomposition",
    "harmonic_extension",
    "impose_mean_zero",
    "real_homology_basis",
    "vertex_map",
]
