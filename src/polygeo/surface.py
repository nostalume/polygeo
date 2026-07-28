"""Capabilities and algorithms specific to Euclidean triangle-manifold realizations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Iterable, Literal, final

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array, diags

from .algorithms import (
    AlgorithmError,
    Certified,
    PositiveHodgeMetric,
    ResidualEvidence,
    VertexMap,
    vertex_map,
)
from .geometry import Geometry, GeometryError
from .operators import LinearMap, OperatorError, hodge_laplacian
from .solvers import (
    LinearSolution,
    PrepareLinearSolve,
    _linear_residual_evidence,
)
from .simplicial import (
    ORDINARY_FORM,
    BoundaryState,
    CochainSpace,
    Complex,
    Connected,
    ConnectivityState,
    Form,
    OrientationState,
    Oriented,
    OrdinaryForm,
    TriangleManifold,
    WithBoundary,
    WithoutBoundary,
    topological_boundary,
)


class SurfaceError(ValueError):
    """Stable failure boundary for surface-specific admission and algorithms."""


@final
class Disk[K: Complex[WithBoundary, Oriented, Connected, TriangleManifold]]:
    """Exact-complex-bound evidence that one admitted triangle surface is a disk."""

    __slots__ = ("_complex", "_sealed")
    _complex: K
    _sealed: bool

    def __init__(self) -> None:
        raise SurfaceError("Disk must be created by disk()")

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("Disk evidence is immutable")
        object.__setattr__(self, name, value)

    @property
    def complex(self) -> K:
        return self._complex


def disk[K: Complex[WithBoundary, Oriented, Connected, TriangleManifold]](
    complex_: K,
) -> Disk[K]:
    """Admit exact global disk topology for one refined triangle manifold."""
    if (
        not isinstance(complex_, Complex)
        or not isinstance(complex_.boundary_state, WithBoundary)
        or not isinstance(complex_.orientation_state, Oriented)
        or not isinstance(complex_.connectivity_state, Connected)
        or not isinstance(complex_.topology_state, TriangleManifold)
    ):
        raise SurfaceError("disk requires a connected oriented triangle manifold")

    boundary = topological_boundary(complex_)
    vertices = np.flatnonzero(boundary.mask(0))
    edges = complex_.simplices(1)[boundary.mask(1)]
    if not _is_one_cycle(vertices, edges):
        raise SurfaceError("disk boundary must have exactly one closed component")
    euler_characteristic = sum(
        (-1) ** degree * complex_.simplex_count(degree)
        for degree in range(complex_.dimension + 1)
    )
    if euler_characteristic != 1:
        raise SurfaceError("disk Euler characteristic must equal one")
    evidence: Disk[K] = object.__new__(Disk)
    object.__setattr__(evidence, "_complex", complex_)
    object.__setattr__(evidence, "_sealed", True)
    return evidence


def _is_one_cycle(vertices: NDArray[np.int64], edges: NDArray[np.int64]) -> bool:
    if len(vertices) < 3 or len(edges) != len(vertices):
        return False
    index = {int(vertex): offset for offset, vertex in enumerate(vertices)}
    adjacency: list[list[int]] = [[] for _ in vertices]
    for left, right in edges:
        if int(left) not in index or int(right) not in index:
            return False
        adjacency[index[int(left)]].append(index[int(right)])
        adjacency[index[int(right)]].append(index[int(left)])
    if any(len(neighbors) != 2 for neighbors in adjacency):
        return False
    visited = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return len(visited) == len(vertices)


@final
class VertexVectors[K: Complex]:
    """Owned ambient vectors at canonical vertices of one exact geometry."""

    __slots__ = ("_geometry", "_vectors")

    def __init__(self, geometry: Geometry[K], vectors: NDArray[np.float64]) -> None:
        if not isinstance(geometry, Geometry):
            raise SurfaceError("surface vectors require geometry")
        self._geometry = geometry
        self._vectors = _admit_vectors(
            geometry, vectors, geometry.complex.vertex_count, "vertex"
        )

    @property
    def geometry(self) -> Geometry[K]:
        return self._geometry

    @property
    def vectors(self) -> NDArray[np.float64]:
        return self._vectors.copy()

    def normalized(self) -> VertexVectors[K]:
        return VertexVectors(self._geometry, _normalize_vectors(self._vectors))


@final
class FaceVectors[K: Complex]:
    """Owned ambient vectors at canonical triangles of one exact geometry."""

    __slots__ = ("_geometry", "_vectors")

    def __init__(self, geometry: Geometry[K], vectors: NDArray[np.float64]) -> None:
        if not isinstance(geometry, Geometry):
            raise SurfaceError("surface vectors require geometry")
        self._geometry = geometry
        self._vectors = _admit_vectors(
            geometry, vectors, geometry.complex.simplex_count(2), "face"
        )

    @property
    def geometry(self) -> Geometry[K]:
        return self._geometry

    @property
    def vectors(self) -> NDArray[np.float64]:
        return self._vectors.copy()

    def normalized(self) -> FaceVectors[K]:
        return FaceVectors(self._geometry, _normalize_vectors(self._vectors))


def _admit_vectors[K: Complex](
    geometry: Geometry[K],
    vectors: NDArray[np.float64],
    count: int,
    support: str,
) -> NDArray[np.float64]:
    if not isinstance(geometry, Geometry):
        raise SurfaceError("surface vectors require geometry")
    if (
        not isinstance(vectors, np.ndarray)
        or vectors.dtype != np.dtype(np.float64)
        or vectors.shape != (count, geometry.ambient_dimension)
    ):
        raise SurfaceError(f"surface vectors require one vector per {support}")
    if not np.all(np.isfinite(vectors)):
        raise SurfaceError("surface vectors must be finite")
    owned = np.array(vectors, dtype=np.float64, order="C", copy=True)
    owned.flags.writeable = False
    return owned


def _normalize_vectors(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    scales = np.max(np.abs(vectors), axis=1)
    if np.any(scales == 0.0) or not np.all(np.isfinite(scales)):
        raise SurfaceError("cannot normalize zero vectors")
    scaled = vectors / scales[:, None]
    lengths = np.linalg.norm(scaled, axis=1)
    normalized = scaled / lengths[:, None]
    if not np.all(np.isfinite(normalized)):
        raise SurfaceError("cannot normalize zero vectors")
    return normalized


def _face_directions(points: NDArray[np.float64]) -> NDArray[np.float64]:
    left = _normalize_vectors(points[:, 1] - points[:, 0])
    right = _normalize_vectors(points[:, 2] - points[:, 0])
    return _normalize_vectors(np.cross(left, right))


def _unit_vectors_with_log_lengths(
    vectors: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    scales = np.max(np.abs(vectors), axis=1)
    if np.any(scales == 0.0) or not np.all(np.isfinite(scales)):
        raise SurfaceError("sphere-inscribed normals are not representable")
    scaled = vectors / scales[:, None]
    lengths = np.linalg.norm(scaled, axis=1)
    units = scaled / lengths[:, None]
    logs = np.log(scales) + np.log(lengths)
    if not np.all(np.isfinite(units)) or not np.all(np.isfinite(logs)):
        raise SurfaceError("sphere-inscribed normals are not representable")
    return units, logs


def _surface_points[K: Complex](
    geometry: Geometry[K], *, oriented: bool = False, closed: bool = False
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    if not isinstance(geometry, Geometry) or not isinstance(
        geometry.complex.topology_state, TriangleManifold
    ):
        raise SurfaceError("surface vectors require triangle-manifold geometry")
    if geometry.ambient_dimension != 3:
        raise SurfaceError("surface vectors require ambient dimension three")
    if oriented and not isinstance(geometry.complex.orientation_state, Oriented):
        raise SurfaceError("oriented surface vectors require oriented geometry")
    if closed and not isinstance(geometry.complex.boundary_state, WithoutBoundary):
        raise SurfaceError("vertex normals require a closed surface")
    return geometry.complex.simplices(2), geometry.positions


def _oriented_face_data[K: Complex](
    geometry: Geometry[K], *, closed: bool = False
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    faces, positions = _surface_points(geometry, oriented=True, closed=closed)
    points = positions[faces]
    normals = _face_directions(points)
    normals *= geometry.complex.orientations(2)[:, None]
    return faces, points, normals


def face_unit_normals[
    B: BoundaryState,
    C: ConnectivityState,
](
    geometry: Geometry[Complex[B, Oriented, C, TriangleManifold]],
) -> FaceVectors[Complex[B, Oriented, C, TriangleManifold]]:
    """Return oriented unit normals at canonical triangles in ambient three-space."""
    _, _, normals = _oriented_face_data(geometry)
    return FaceVectors(geometry, normals)


def surface_area_gradient[
    B: BoundaryState,
    O: OrientationState,
    C: ConnectivityState,
](
    geometry: Geometry[Complex[B, O, C, TriangleManifold]],
) -> VertexVectors[Complex[B, O, C, TriangleManifold]]:
    """Return the unnormalized gradient of total triangle area at each vertex."""
    faces, positions = _surface_points(geometry)
    points = positions[faces]
    normals = _face_directions(points)
    gradient = np.zeros_like(positions)
    for corner in range(3):
        contribution = 0.5 * np.cross(
            points[:, (corner + 1) % 3] - points[:, (corner + 2) % 3], normals
        )
        np.add.at(gradient, faces[:, corner], contribution)
    if not np.all(np.isfinite(gradient)):
        raise SurfaceError("surface area gradient is not representable")
    return VertexVectors(geometry, gradient)


def volume_gradient[C: ConnectivityState](
    geometry: Geometry[Complex[WithoutBoundary, Oriented, C, TriangleManifold]],
) -> VertexVectors[Complex[WithoutBoundary, Oriented, C, TriangleManifold]]:
    """Return the signed enclosed-volume gradient at canonical vertices."""
    faces, _, normals = _oriented_face_data(geometry, closed=True)
    areas = geometry.primal_measures(2)
    gradient = np.zeros_like(geometry.positions)
    contributions = areas[:, None] * normals / 3.0
    for corner in range(3):
        np.add.at(gradient, faces[:, corner], contributions)
    return VertexVectors(geometry, gradient)


def uniform_vertex_normals[C: ConnectivityState](
    geometry: Geometry[Complex[WithoutBoundary, Oriented, C, TriangleManifold]],
) -> VertexVectors[Complex[WithoutBoundary, Oriented, C, TriangleManifold]]:
    """Return normalized sums of incident oriented unit face normals."""
    faces, _, normals = _oriented_face_data(geometry, closed=True)
    values = np.zeros_like(geometry.positions)
    for corner in range(3):
        np.add.at(values, faces[:, corner], normals)
    return VertexVectors(geometry, values).normalized()


def tip_angle_vertex_normals[C: ConnectivityState](
    geometry: Geometry[Complex[WithoutBoundary, Oriented, C, TriangleManifold]],
) -> VertexVectors[Complex[WithoutBoundary, Oriented, C, TriangleManifold]]:
    """Return normalized tip-angle-weighted incident face normals."""
    faces, points, normals = _oriented_face_data(geometry, closed=True)
    angles = _triangle_angles(points)
    values = np.zeros_like(geometry.positions)
    for corner in range(3):
        np.add.at(values, faces[:, corner], angles[:, corner, None] * normals)
    return VertexVectors(geometry, values).normalized()


def sphere_inscribed_vertex_normals[C: ConnectivityState](
    geometry: Geometry[Complex[WithoutBoundary, Oriented, C, TriangleManifold]],
) -> VertexVectors[Complex[WithoutBoundary, Oriented, C, TriangleManifold]]:
    """Return Crane's sphere-inscribed cyclic edge normal directions."""
    faces, _, _ = _oriented_face_data(geometry, closed=True)
    oriented_faces = faces.copy()
    reversed_faces = geometry.complex.orientations(2) < 0
    oriented_faces[reversed_faces, 1], oriented_faces[reversed_faces, 2] = (
        faces[reversed_faces, 2],
        faces[reversed_faces, 1],
    )
    positions = geometry.positions
    points = positions[oriented_faces]
    entries: list[
        tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]
    ] = []
    for corner in range(3):
        origin = points[:, corner]
        left, left_logs = _unit_vectors_with_log_lengths(
            points[:, (corner + 1) % 3] - origin
        )
        right, right_logs = _unit_vectors_with_log_lengths(
            points[:, (corner + 2) % 3] - origin
        )
        entries.append(
            (
                oriented_faces[:, corner],
                np.cross(left, right),
                -(left_logs + right_logs),
            )
        )

    offset = max(float(np.max(log_weights)) for _, _, log_weights in entries)
    values = np.zeros_like(positions)
    for vertices, directions, log_weights in entries:
        contributions = directions * np.exp(log_weights - offset)[:, None]
        np.add.at(values, vertices, contributions)
    if not np.all(np.isfinite(values)):
        raise SurfaceError("sphere-inscribed normals are not representable")
    return VertexVectors(geometry, values).normalized()


def mean_curvature_vectors[O: OrientationState](
    metric: PositiveHodgeMetric[
        Complex[WithoutBoundary, O, Connected, TriangleManifold]
    ],
) -> VertexVectors[Complex[WithoutBoundary, O, Connected, TriangleManifold]]:
    """Return the mass-normalized positive cotan Laplacian of the embedding."""
    if not isinstance(metric, PositiveHodgeMetric):
        raise SurfaceError("mean curvature vectors require a positive Hodge metric")
    geometry = metric.geometry
    _surface_points(geometry, closed=True)
    space = geometry.complex.cochain_space(0)
    operator = hodge_laplacian(geometry, space)
    values = np.empty_like(geometry.positions)
    try:
        for coordinate in range(geometry.ambient_dimension):
            form = space.form(geometry.positions[:, coordinate], ORDINARY_FORM)
            values[:, coordinate] = operator.apply(form).coefficients()
    except (OperatorError, ValueError) as error:
        raise SurfaceError("mean curvature vectors are not representable") from error
    return VertexVectors(geometry, values)


def gaussian_curvature_measure[
    B: BoundaryState,
    O: OrientationState,
    C: ConnectivityState,
](
    geometry: Geometry[Complex[B, O, C, TriangleManifold]],
) -> Form[CochainSpace[Complex[B, O, C, TriangleManifold], Literal[0]], OrdinaryForm]:
    """Return integrated vertex angle defects in the canonical degree-zero basis."""
    if not isinstance(geometry, Geometry) or not isinstance(
        geometry.complex.topology_state, TriangleManifold
    ):
        raise SurfaceError("curvature requires triangle-manifold geometry")

    complex_ = geometry.complex
    faces = complex_.simplices(2)
    positions = geometry.positions
    angle_sums = np.zeros(complex_.vertex_count, dtype=np.float64)
    for start in range(0, len(faces), 4096):
        batch = faces[start : start + 4096]
        try:
            angles = _triangle_angles(positions[batch])
        except np.linalg.LinAlgError as error:
            raise SurfaceError("curvature angle evaluation failed") from error
        np.add.at(angle_sums, batch.ravel(), angles.ravel())

    boundary = topological_boundary(complex_).mask(0)
    with np.errstate(invalid="ignore"):
        curvature = np.where(boundary, math.pi, 2.0 * math.pi) - angle_sums
    if not np.all(np.isfinite(curvature)):
        raise SurfaceError("curvature is not representable")
    return complex_.cochain_space(0).form(curvature, ORDINARY_FORM)


def _triangle_angles(points: NDArray[np.float64]) -> NDArray[np.float64]:
    angles = np.empty((len(points), 3), dtype=np.float64)
    for corner in range(3):
        left = points[:, (corner + 1) % 3] - points[:, corner]
        right = points[:, (corner + 2) % 3] - points[:, corner]
        left_scale = np.max(np.abs(left), axis=1)
        right_scale = np.max(np.abs(right), axis=1)
        with np.errstate(all="ignore"):
            left /= left_scale[:, None]
            right /= right_scale[:, None]
            left /= np.linalg.norm(left, axis=1)[:, None]
            right /= np.linalg.norm(right, axis=1)[:, None]
            dot = np.sum(left * right, axis=1)
            _, triangular = np.linalg.qr(np.stack((left, right), axis=2))
            area = np.abs(triangular[:, 0, 0] * triangular[:, 1, 1])
            angles[:, corner] = np.arctan2(area, dot)
        suspicious = (~np.isfinite(angles[:, corner])) | (area == 0.0)
        for index in np.flatnonzero(suspicious):
            angles[index, corner] = _exact_triangle_angle(
                points[index, corner],
                points[index, (corner + 1) % 3],
                points[index, (corner + 2) % 3],
            )
    if not np.all(np.isfinite(angles)):
        raise SurfaceError("triangle angles are not representable")
    return angles


def _exact_triangle_angle(
    origin: NDArray[np.float64],
    left_point: NDArray[np.float64],
    right_point: NDArray[np.float64],
) -> float:
    left = tuple(
        Fraction(float(value)) - Fraction(float(base))
        for value, base in zip(left_point, origin, strict=True)
    )
    right = tuple(
        Fraction(float(value)) - Fraction(float(base))
        for value, base in zip(right_point, origin, strict=True)
    )
    left_norm = sum((value * value for value in left), start=Fraction())
    right_norm = sum((value * value for value in right), start=Fraction())
    dot = sum((a * b for a, b in zip(left, right, strict=True)), start=Fraction())
    determinant = left_norm * right_norm - dot * dot
    if determinant <= 0:
        raise SurfaceError("triangle angle is exactly degenerate")
    with localcontext() as context:
        context.prec = 80
        area = (
            Decimal(determinant.numerator) / Decimal(determinant.denominator)
        ).sqrt()
        dot_decimal = Decimal(dot.numerator) / Decimal(dot.denominator)
        scale = max(area, abs(dot_decimal))
        return math.atan2(float(area / scale), float(dot_decimal / scale))


@dataclass(frozen=True, slots=True)
@final
class FrozenFlowEvidence:
    time_step: float
    energy_before: float
    energy_after: float
    centroid: ResidualEvidence
    solves: tuple[ResidualEvidence, ...]

    def __post_init__(self) -> None:
        if (
            type(self.time_step) is not float
            or not math.isfinite(self.time_step)
            or self.time_step <= 0.0
        ):
            raise SurfaceError("flow time step must be finite and positive")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (self.energy_before, self.energy_after)
        ):
            raise SurfaceError("flow energies must be finite and nonnegative")
        if self.energy_after > self.energy_before:
            raise SurfaceError("flow frozen energy increased")
        if (
            not isinstance(self.centroid, ResidualEvidence)
            or type(self.solves) is not tuple
        ):
            raise SurfaceError("flow evidence requires residual evidence")
        if any(not isinstance(solve, ResidualEvidence) for solve in self.solves):
            raise SurfaceError("flow evidence requires residual evidence")


def mean_curvature_flow_step[O: OrientationState](
    metric: PositiveHodgeMetric[
        Complex[WithoutBoundary, O, Connected, TriangleManifold]
    ],
    time_step: float,
    prepare: PrepareLinearSolve[
        CochainSpace[
            Complex[WithoutBoundary, O, Connected, TriangleManifold], Literal[0]
        ],
        CochainSpace[
            Complex[WithoutBoundary, O, Connected, TriangleManifold], Literal[0]
        ],
    ],
) -> Certified[
    VertexMap[Complex[WithoutBoundary, O, Connected, TriangleManifold], int],
    FrozenFlowEvidence,
]:
    """Advance one closed surface by a frozen-metric implicit Euler step."""
    if not isinstance(metric, PositiveHodgeMetric):
        raise SurfaceError("flow requires a positive Hodge metric")
    geometry = metric.geometry
    complex_ = geometry.complex
    if (
        not isinstance(complex_.boundary_state, WithoutBoundary)
        or not isinstance(complex_.connectivity_state, Connected)
        or not isinstance(complex_.topology_state, TriangleManifold)
    ):
        raise SurfaceError("flow requires a closed connected triangle manifold")
    if type(time_step) is not float or not math.isfinite(time_step) or time_step <= 0.0:
        raise SurfaceError("flow time step must be finite and positive")

    try:
        operator, rhs_weights, derivative, edge_weights = _flow_operator(
            metric, time_step
        )
    except (AlgorithmError, OperatorError, OverflowError, ValueError) as error:
        raise SurfaceError("flow operator is not representable") from error
    positions = geometry.positions
    centered, centroid = _center_positions(metric.weights(0), positions)
    space = complex_.cochain_space(0)
    updated_centered = np.empty_like(centered)
    solves: list[ResidualEvidence] = []
    try:
        prepared = prepare(operator)
        for coordinate in range(geometry.ambient_dimension):
            rhs = space.form(rhs_weights * centered[:, coordinate], ORDINARY_FORM)
            solution = prepared(rhs)
            if not isinstance(solution, LinearSolution):
                raise SurfaceError("flow solver returned malformed solver evidence")
            if not solution.form.space.same_space(
                space
            ) or not solution.equation_space.same_space(space):
                raise SurfaceError("flow solver returned foreign solver evidence")
            updated_centered[:, coordinate] = solution.form.coefficients()
            norm, scale, _ = _linear_residual_evidence(operator, solution.form, rhs)
            solves.append(
                ResidualEvidence(norm, scale, float(np.sqrt(np.finfo(np.float64).eps)))
            )
    except Exception as error:
        raise SurfaceError("flow solve failed") from error

    with np.errstate(all="ignore"):
        updated_positions = updated_centered + centroid
    if not np.all(np.isfinite(updated_positions)):
        raise SurfaceError("flow output positions are not representable")
    try:
        target = Geometry.from_positions(complex_, updated_positions)
    except GeometryError as error:
        raise SurfaceError("flow output geometry is not admissible") from error

    evidence = FrozenFlowEvidence(
        time_step,
        _frozen_energy(derivative, edge_weights, centered),
        _frozen_energy(derivative, edge_weights, updated_centered),
        _centroid_evidence(metric.weights(0), positions, target.positions),
        tuple(solves),
    )
    try:
        output = vertex_map(geometry, target, geometry.ambient_dimension)
    except AlgorithmError as error:
        raise SurfaceError("flow output map is not admissible") from error
    return Certified(output, evidence)


def _flow_operator[O: OrientationState](
    metric: PositiveHodgeMetric[
        Complex[WithoutBoundary, O, Connected, TriangleManifold]
    ],
    time_step: float,
) -> tuple[
    LinearMap[
        CochainSpace[
            Complex[WithoutBoundary, O, Connected, TriangleManifold], Literal[0]
        ],
        CochainSpace[
            Complex[WithoutBoundary, O, Connected, TriangleManifold], Literal[0]
        ],
    ],
    NDArray[np.float64],
    csr_array,
    NDArray[np.float64],
]:
    complex_ = metric.geometry.complex
    space = complex_.cochain_space(0)
    mass = metric.weights(0)
    edge_weights = metric.weights(1)
    derivative = complex_.boundary_matrix(1).transpose().tocsr()
    stiffness = (
        derivative.transpose() @ derivative.multiply(edge_weights[:, None])
    ).tocsr()
    mass_scale = float(np.max(mass))
    stiffness_scale = float(np.max(np.abs(stiffness.data), initial=0.0))
    if mass_scale <= 0.0 or stiffness_scale <= 0.0:
        raise SurfaceError("flow operator has zero scale")
    mass_exact = Fraction(mass_scale)
    stiffness_exact = Fraction(time_step) * Fraction(stiffness_scale)
    common = max(mass_exact, stiffness_exact)
    mass_factor = float(mass_exact / common)
    stiffness_factor = float(stiffness_exact / common)
    if mass_factor == 0.0 or stiffness_factor == 0.0:
        raise SurfaceError("flow operator scaling is not representable")
    with np.errstate(all="ignore"):
        normalized_mass = mass / mass_scale
        rhs_weights = mass_factor * normalized_mass
        normalized_stiffness = (stiffness / stiffness_scale) * stiffness_factor
        matrix = diags(rhs_weights, format="csr") + normalized_stiffness
    if not np.all(np.isfinite(matrix.data)) or not np.all(np.isfinite(rhs_weights)):
        raise SurfaceError("flow operator is not representable")
    return (
        LinearMap(space, space, csr_array(matrix)),
        rhs_weights,
        derivative,
        edge_weights,
    )


def _center_positions(
    mass: NDArray[np.float64], positions: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    normalized = mass / float(np.max(mass))
    anchor = positions[0]
    with np.errstate(all="ignore"):
        offsets = positions - anchor
        centroid = anchor + normalized @ offsets / math.fsum(normalized)
        centered = positions - centroid
    if not np.all(np.isfinite(centered)) or not np.all(np.isfinite(centroid)):
        raise SurfaceError("flow centroid is not representable")
    return centered, centroid


def _frozen_energy(
    derivative: csr_array,
    edge_weights: NDArray[np.float64],
    positions: NDArray[np.float64],
) -> float:
    with np.errstate(all="ignore"):
        differences = np.asarray(derivative @ positions)
        energy = 0.5 * float(np.sum(edge_weights[:, None] * differences * differences))
    if not math.isfinite(energy) or energy < 0.0:
        raise SurfaceError("flow frozen energy is not representable")
    return energy


def _centroid_evidence(
    mass: NDArray[np.float64],
    before: NDArray[np.float64],
    after: NDArray[np.float64],
) -> ResidualEvidence:
    ratios: list[Fraction] = []
    for coordinate in range(before.shape[1]):
        terms = [
            Fraction(float(weight)) * (Fraction(float(right)) - Fraction(float(left)))
            for weight, left, right in zip(
                mass, before[:, coordinate], after[:, coordinate], strict=True
            )
        ]
        scale = sum((abs(term) for term in terms), start=Fraction())
        if scale:
            ratios.append(abs(sum(terms, start=Fraction())) / scale)
    limit = float(np.sqrt(np.finfo(np.float64).eps))
    if not ratios:
        return ResidualEvidence(0.0, 0.0, limit)
    return ResidualEvidence(float(max(ratios)), 1.0, limit)


class _ImmutableSurfaceProduct:
    __slots__ = ()

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("surface product is immutable")
        object.__setattr__(self, name, value)


@final
class TriangleFrames[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](_ImmutableSurfaceProduct):
    """Deterministic right-handed tangent frames on canonical oriented faces."""

    __slots__ = ("_first", "_geometry", "_normals", "_sealed", "_second")
    _geometry: Geometry[K]
    _first: NDArray[np.float64]
    _second: NDArray[np.float64]
    _normals: NDArray[np.float64]
    _sealed: bool

    def __init__(self) -> None:
        raise SurfaceError("TriangleFrames must be created by triangle_frames()")

    @property
    def geometry(self) -> Geometry[K]:
        return self._geometry

    def first_axes(self) -> NDArray[np.float64]:
        return self._first.copy()

    def second_axes(self) -> NDArray[np.float64]:
        return self._second.copy()

    def normals(self) -> NDArray[np.float64]:
        return self._normals.copy()


@final
class SurfaceConnection[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](_ImmutableSurfaceProduct):
    """Geometry-bound SO(2) transport on canonically oriented dual edges."""

    __slots__ = (
        "_deviations",
        "_dual_edges",
        "_frames",
        "_geometry",
        "_levi_civita",
        "_products",
        "_sealed",
    )
    _geometry: Geometry[K]
    _frames: TriangleFrames[K]
    _dual_edges: NDArray[np.int64]
    _levi_civita: NDArray[np.complex128]
    _deviations: NDArray[np.float64]
    _products: NDArray[np.complex128]
    _sealed: bool

    def __init__(self) -> None:
        raise SurfaceError("SurfaceConnection must be created by a connection factory")

    @property
    def geometry(self) -> Geometry[K]:
        return self._geometry

    @property
    def frames(self) -> TriangleFrames[K]:
        return self._frames

    def dual_edges(self) -> NDArray[np.int64]:
        return self._dual_edges.copy()

    def levi_civita_products(self) -> NDArray[np.complex128]:
        return self._levi_civita.copy()

    def deviation_angles(self) -> NDArray[np.float64]:
        return self._deviations.copy()

    def transport_products(self) -> NDArray[np.complex128]:
        return self._products.copy()

    def transport(self, source_face: int, target_face: int) -> complex:
        matches = np.flatnonzero(
            np.logical_or(
                np.all(self._dual_edges == (source_face, target_face), axis=1),
                np.all(self._dual_edges == (target_face, source_face), axis=1),
            )
        )
        if len(matches) != 1 or source_face == target_face:
            raise SurfaceError("transport requires two adjacent faces")
        edge = int(matches[0])
        product = complex(self._products[edge])
        if tuple(self._dual_edges[edge]) == (source_face, target_face):
            return product
        return product.conjugate()


@final
class IntegralDualCycles[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](_ImmutableSurfaceProduct):
    """Deterministic primitive integral tree-cotree dual generators."""

    __slots__ = ("_coefficients", "_generator_edges", "_geometry", "_sealed")
    _geometry: Geometry[K]
    _coefficients: NDArray[np.int64]
    _generator_edges: NDArray[np.int64]
    _sealed: bool

    def __init__(self) -> None:
        raise SurfaceError(
            "IntegralDualCycles must be created by integral_dual_cycles()"
        )

    @property
    def geometry(self) -> Geometry[K]:
        return self._geometry

    @property
    def dimension(self) -> int:
        return self._coefficients.shape[1]

    def cycle_coefficients(self) -> csr_array:
        return csr_array(self._coefficients.copy(), dtype=np.int64)

    def generator_edge_indices(self) -> NDArray[np.int64]:
        return self._generator_edges.copy()


@dataclass(frozen=True, slots=True)
@final
class HolonomyEvidence:
    """Descriptive circular errors for local cells and integral generators."""

    local_products: tuple[complex, ...]
    generator_products: tuple[complex, ...]
    local_error: float
    generator_error: float
    limit: float

    def __post_init__(self) -> None:
        scalars = (self.local_error, self.generator_error, self.limit)
        if not all(math.isfinite(value) and value >= 0.0 for value in scalars):
            raise SurfaceError("holonomy evidence must be finite and nonnegative")
        if not all(
            math.isfinite(value.real) and math.isfinite(value.imag)
            for value in self.local_products + self.generator_products
        ):
            raise SurfaceError("holonomy products must be finite")


@final
class IntegrableConnection[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](_ImmutableSurfaceProduct):
    """Factory-only authority for represented circular consistency within its limit."""

    __slots__ = ("_connection", "_cycles", "_evidence", "_phases", "_sealed")
    _connection: SurfaceConnection[K]
    _cycles: IntegralDualCycles[K]
    _evidence: HolonomyEvidence
    _phases: NDArray[np.complex128]
    _sealed: bool

    def __init__(self) -> None:
        raise SurfaceError(
            "IntegrableConnection must be created by admit_integrable_connection()"
        )

    @property
    def connection(self) -> SurfaceConnection[K]:
        return self._connection

    @property
    def cycles(self) -> IntegralDualCycles[K]:
        return self._cycles

    @property
    def evidence(self) -> HolonomyEvidence:
        return self._evidence


@final
class FaceDirectionField[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](_ImmutableSurfaceProduct):
    """Wrapped face phases and ambient unit tangent vectors on one geometry."""

    __slots__ = (
        "_anchor_face",
        "_anchor_phase",
        "_connection",
        "_geometry",
        "_phases",
        "_sealed",
        "_vectors",
    )
    _geometry: Geometry[K]
    _connection: SurfaceConnection[K]
    _anchor_face: int
    _anchor_phase: float
    _phases: NDArray[np.float64]
    _vectors: NDArray[np.float64]
    _sealed: bool

    def __init__(self) -> None:
        raise SurfaceError(
            "FaceDirectionField must be created by integrate_direction_field()"
        )

    @property
    def geometry(self) -> Geometry[K]:
        return self._geometry

    @property
    def connection(self) -> SurfaceConnection[K]:
        return self._connection

    @property
    def anchor_face(self) -> int:
        return self._anchor_face

    @property
    def anchor_phase(self) -> float:
        return self._anchor_phase

    def phases(self) -> NDArray[np.float64]:
        return self._phases.copy()

    def vectors(self) -> NDArray[np.float64]:
        return self._vectors.copy()


@dataclass(frozen=True, slots=True)
@final
class DirectionFieldEvidence:
    """Descriptive absolute circular crossing-consistency evidence."""

    crossing_error: float
    limit: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.crossing_error)
            or not math.isfinite(self.limit)
            or self.crossing_error < 0.0
            or self.limit < 0.0
            or self.crossing_error > self.limit
        ):
            raise SurfaceError("direction-field crossing evidence exceeds its limit")


def _connection_geometry[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](geometry: Geometry[K]) -> None:
    if (
        not isinstance(geometry, Geometry)
        or geometry.ambient_dimension != 3
        or not isinstance(geometry.complex.boundary_state, WithoutBoundary)
        or not isinstance(geometry.complex.orientation_state, Oriented)
        or not isinstance(geometry.complex.connectivity_state, Connected)
        or not isinstance(geometry.complex.topology_state, TriangleManifold)
    ):
        raise SurfaceError(
            "surface connections require closed connected oriented triangle geometry in R3"
        )


def _freeze_array[Scalar: np.generic](values: NDArray[Scalar]) -> NDArray[Scalar]:
    owned = np.array(values, order="C", copy=True)
    owned.flags.writeable = False
    return owned


def triangle_frames[K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]](
    geometry: Geometry[K],
) -> TriangleFrames[K]:
    """Construct deterministic right-handed frames from canonical face edges."""
    _connection_geometry(geometry)
    faces, points, normals = _oriented_face_data(geometry, closed=True)
    del faces
    first = _normalize_vectors(points[:, 1] - points[:, 0])
    second = _normalize_vectors(np.cross(normals, first))
    frames: TriangleFrames[K] = object.__new__(TriangleFrames)
    object.__setattr__(frames, "_geometry", geometry)
    object.__setattr__(frames, "_first", _freeze_array(first))
    object.__setattr__(frames, "_second", _freeze_array(second))
    object.__setattr__(frames, "_normals", _freeze_array(normals))
    object.__setattr__(frames, "_sealed", True)
    return frames


def _dual_edges[K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]](
    geometry: Geometry[K],
) -> NDArray[np.int64]:
    boundary = geometry.complex.boundary_matrix(2)
    if np.any(np.diff(boundary.indptr) != 2):
        raise SurfaceError("closed surface dual edges require two adjacent faces")
    return np.sort(boundary.indices.reshape(-1, 2), axis=1)


def _levi_civita_products[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](
    geometry: Geometry[K],
    frames: TriangleFrames[K],
    dual_edges: NDArray[np.int64],
) -> NDArray[np.complex128]:
    edges = geometry.complex.simplices(1)
    positions = geometry.positions
    axes = _normalize_vectors(positions[edges[:, 1]] - positions[edges[:, 0]])
    normals = frames._normals
    first = frames._first
    source = dual_edges[:, 0]
    target = dual_edges[:, 1]
    sine = np.einsum("ij,ij->i", axes, np.cross(normals[source], normals[target]))
    cosine = np.einsum("ij,ij->i", normals[source], normals[target])
    angles = np.arctan2(sine, cosine)
    cosines = np.cos(angles)[:, None]
    sines = np.sin(angles)[:, None]
    source_first = first[source]
    rotated = (
        source_first * cosines
        + np.cross(axes, source_first) * sines
        + axes * np.einsum("ij,ij->i", axes, source_first)[:, None] * (1.0 - cosines)
    )
    second = frames._second
    products = (
        np.einsum("ij,ij->i", rotated, first[target])
        + 1j * np.einsum("ij,ij->i", rotated, second[target])
    ).astype(np.complex128)
    magnitudes = np.abs(products)
    if np.any(magnitudes == 0.0) or not np.all(np.isfinite(magnitudes)):
        raise SurfaceError("Levi-Civita transport is not representable")
    return products / magnitudes


def _surface_connection[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](geometry: Geometry[K], deviations: NDArray[np.float64]) -> SurfaceConnection[K]:
    _connection_geometry(geometry)
    dual_edges = _dual_edges(geometry)
    if (
        not isinstance(deviations, np.ndarray)
        or deviations.dtype != np.dtype(np.float64)
        or deviations.shape != (len(dual_edges),)
        or not np.all(np.isfinite(deviations))
    ):
        raise SurfaceError(
            "connection deviation angles require one finite value per dual edge"
        )
    frames = triangle_frames(geometry)
    levi_civita = _levi_civita_products(geometry, frames, dual_edges)
    products = levi_civita * np.exp(1j * deviations)
    products /= np.abs(products)
    connection: SurfaceConnection[K] = object.__new__(SurfaceConnection)
    object.__setattr__(connection, "_geometry", geometry)
    object.__setattr__(connection, "_frames", frames)
    object.__setattr__(connection, "_dual_edges", _freeze_array(dual_edges))
    object.__setattr__(connection, "_levi_civita", _freeze_array(levi_civita))
    object.__setattr__(connection, "_deviations", _freeze_array(deviations))
    object.__setattr__(connection, "_products", _freeze_array(products))
    object.__setattr__(connection, "_sealed", True)
    return connection


def levi_civita_connection[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](geometry: Geometry[K]) -> SurfaceConnection[K]:
    """Return zero-deviation discrete Levi-Civita face transport."""
    _connection_geometry(geometry)
    return _surface_connection(
        geometry, np.zeros(geometry.complex.simplex_count(1), dtype=np.float64)
    )


def surface_connection[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](geometry: Geometry[K], deviation_angles: NDArray[np.float64]) -> SurfaceConnection[K]:
    """Compose Levi-Civita transport with retained lifted in-plane deviations."""
    return _surface_connection(geometry, deviation_angles)


def _join(parent: list[int], left: int, right: int) -> bool:
    while parent[left] != left:
        parent[left] = parent[parent[left]]
        left = parent[left]
    while parent[right] != right:
        parent[right] = parent[parent[right]]
        right = parent[right]
    if left == right:
        return False
    parent[right] = left
    return True


def _tree_path(
    adjacency: list[list[tuple[int, int]]], source: int, target: int
) -> list[tuple[int, int, int]]:
    parent = {source: (-1, -1)}
    pending = [source]
    for current in pending:
        if current == target:
            break
        for neighbor, edge in adjacency[current]:
            if neighbor not in parent:
                parent[neighbor] = (current, edge)
                pending.append(neighbor)
    if target not in parent:
        raise SurfaceError("dual spanning tree is disconnected")
    path: list[tuple[int, int, int]] = []
    current = target
    while current != source:
        previous, edge = parent[current]
        path.append((previous, current, edge))
        current = previous
    path.reverse()
    return path


def integral_dual_cycles[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](geometry: Geometry[K]) -> IntegralDualCycles[K]:
    """Admit a deterministic saturated integral tree-cotree generator basis."""
    _connection_geometry(geometry)
    complex_ = geometry.complex
    edges = complex_.simplices(1)
    dual_edges = _dual_edges(geometry)
    vertex_parent = list(range(complex_.vertex_count))
    primal_tree: set[int] = set()
    for edge, (left, right) in enumerate(edges):
        if _join(vertex_parent, int(left), int(right)):
            primal_tree.add(edge)
    if len(primal_tree) != complex_.vertex_count - 1:
        raise SurfaceError("primal spanning tree is disconnected")

    face_count = complex_.simplex_count(2)
    face_parent = list(range(face_count))
    dual_tree: set[int] = set()
    for edge, (source, target) in enumerate(dual_edges):
        if edge not in primal_tree and _join(face_parent, int(source), int(target)):
            dual_tree.add(edge)
    if len(dual_tree) != face_count - 1:
        raise SurfaceError("dual spanning tree is disconnected")

    generators = np.array(
        [
            edge
            for edge in range(len(edges))
            if edge not in primal_tree and edge not in dual_tree
        ],
        dtype=np.int64,
    )
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(face_count)]
    for edge in dual_tree:
        source, target = map(int, dual_edges[edge])
        adjacency[source].append((target, edge))
        adjacency[target].append((source, edge))
    for neighbors in adjacency:
        neighbors.sort(key=lambda item: (item[1], item[0]))

    coefficients = np.zeros((len(edges), len(generators)), dtype=np.int64)
    for column, edge in enumerate(generators):
        source, target = map(int, dual_edges[edge])
        coefficients[edge, column] = 1
        for left, right, tree_edge in _tree_path(adjacency, target, source):
            canonical = tuple(map(int, dual_edges[tree_edge]))
            coefficients[tree_edge, column] = 1 if (left, right) == canonical else -1

    residual = np.zeros((face_count, len(generators)), dtype=np.int64)
    np.add.at(residual, dual_edges[:, 0], -coefficients)
    np.add.at(residual, dual_edges[:, 1], coefficients)
    if np.any(residual):
        raise SurfaceError("integral dual generators are not exactly closed")

    cycles: IntegralDualCycles[K] = object.__new__(IntegralDualCycles)
    object.__setattr__(cycles, "_geometry", geometry)
    object.__setattr__(cycles, "_coefficients", _freeze_array(coefficients))
    object.__setattr__(cycles, "_generator_edges", _freeze_array(generators))
    object.__setattr__(cycles, "_sealed", True)
    return cycles


def _local_cycle_products[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](
    geometry: Geometry[K],
    dual_edges: NDArray[np.int64],
    transports: NDArray[np.complex128],
) -> tuple[complex, ...]:
    incidence = geometry.complex.boundary_matrix(1)
    products: list[complex] = []
    for vertex in range(geometry.complex.vertex_count):
        start, stop = incidence.indptr[vertex : vertex + 2]
        incident = incidence.indices[start:stop]
        adjacency: dict[int, list[tuple[int, int]]] = {}
        for edge in incident:
            source, target = map(int, dual_edges[edge])
            adjacency.setdefault(source, []).append((target, int(edge)))
            adjacency.setdefault(target, []).append((source, int(edge)))
        if not adjacency or any(
            len(neighbors) != 2 for neighbors in adjacency.values()
        ):
            raise SurfaceError("vertex dual cell is not one closed cycle")
        start = min(adjacency)
        previous = -1
        current = start
        following, edge = min(adjacency[current])
        cycle: list[tuple[int, int]] = []
        while True:
            canonical = tuple(map(int, dual_edges[edge]))
            cycle.append((edge, 1 if (current, following) == canonical else -1))
            previous, current = current, following
            if current == start:
                break
            candidates = [item for item in adjacency[current] if item[0] != previous]
            if len(candidates) != 1:
                raise SurfaceError("vertex dual cell traversal is ambiguous")
            following, edge = candidates[0]
        products.append(_cycle_product(transports, sorted(cycle)))
    return tuple(products)


def _cycle_product(
    transports: NDArray[np.complex128],
    coefficients: Iterable[tuple[int | np.integer, int | np.integer]],
) -> complex:
    product = 1.0 + 0.0j
    for edge, exponent in coefficients:
        product *= complex(transports[int(edge)]) ** int(exponent)
        product /= abs(product)
    return product


def _circular_error(products: tuple[complex, ...]) -> float:
    return max(
        (abs(math.atan2(value.imag, value.real)) for value in products), default=0.0
    )


def _circular_limit(edge_count: int) -> float:
    return 128.0 * float(np.finfo(np.float64).eps) * max(1, edge_count)


def connection_holonomy[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](connection: SurfaceConnection[K], cycles: IntegralDualCycles[K]) -> HolonomyEvidence:
    """Evaluate local-cell and noncontractible-generator SO(2) products."""
    if not isinstance(connection, SurfaceConnection) or not isinstance(
        cycles, IntegralDualCycles
    ):
        raise SurfaceError("holonomy requires admitted connection and dual cycles")
    if cycles.geometry is not connection.geometry:
        raise SurfaceError("connection and dual cycles require the same geometry")
    transports = connection._products
    local = _local_cycle_products(
        connection.geometry, connection._dual_edges, transports
    )
    generators: list[complex] = []
    for coefficients in cycles._coefficients.T:
        edges = np.flatnonzero(coefficients)
        generators.append(
            _cycle_product(transports, zip(edges, coefficients[edges], strict=True))
        )
    generator_products = tuple(generators)
    limit = _circular_limit(len(transports))
    return HolonomyEvidence(
        local,
        generator_products,
        _circular_error(local),
        _circular_error(generator_products),
        limit,
    )


def _propagate_phases[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](connection: SurfaceConnection[K]) -> tuple[NDArray[np.complex128], float]:
    dual_edges = connection._dual_edges
    products = connection._products
    face_count = connection.geometry.complex.simplex_count(2)
    adjacency: list[list[tuple[int, int, bool]]] = [[] for _ in range(face_count)]
    for edge, (source, target) in enumerate(dual_edges):
        adjacency[int(source)].append((int(target), edge, True))
        adjacency[int(target)].append((int(source), edge, False))
    phases = np.zeros(face_count, dtype=np.complex128)
    phases[0] = 1.0 + 0.0j
    pending = [0]
    visited = {0}
    for face in pending:
        for neighbor, edge, forward in adjacency[face]:
            if neighbor in visited:
                continue
            transport = products[edge] if forward else np.conjugate(products[edge])
            phases[neighbor] = transport * phases[face]
            phases[neighbor] /= abs(phases[neighbor])
            visited.add(neighbor)
            pending.append(neighbor)
    if len(visited) != face_count:
        raise SurfaceError("connection dual graph is disconnected")
    errors = []
    for edge, (source, target) in enumerate(dual_edges):
        expected = products[edge] * phases[source]
        residual = phases[target] * np.conjugate(expected)
        errors.append(abs(math.atan2(float(residual.imag), float(residual.real))))
    return phases, max(errors, default=0.0)


def admit_integrable_connection[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](
    connection: SurfaceConnection[K], cycles: IntegralDualCycles[K]
) -> IntegrableConnection[K]:
    """Admit exact-domain-bound represented consistency under a fixed circular limit."""
    evidence = connection_holonomy(connection, cycles)
    phases, crossing_error = _propagate_phases(connection)
    if (
        evidence.local_error > evidence.limit
        or evidence.generator_error > evidence.limit
        or crossing_error > evidence.limit
    ):
        raise SurfaceError("surface connection is not integrable")
    capability: IntegrableConnection[K] = object.__new__(IntegrableConnection)
    object.__setattr__(capability, "_connection", connection)
    object.__setattr__(capability, "_cycles", cycles)
    object.__setattr__(capability, "_evidence", evidence)
    object.__setattr__(capability, "_phases", _freeze_array(phases))
    object.__setattr__(capability, "_sealed", True)
    return capability


def integrate_direction_field[
    K: Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
](
    capability: IntegrableConnection[K], *, anchor_phase: float = 0.0
) -> Certified[FaceDirectionField[K], DirectionFieldEvidence]:
    """Integrate one authorized face field from a deterministic face-zero anchor."""
    if not isinstance(capability, IntegrableConnection):
        raise SurfaceError("direction fields require an IntegrableConnection")
    if not math.isfinite(anchor_phase):
        raise SurfaceError("direction-field anchor phase must be finite")
    connection = capability.connection
    phase_products = capability._phases * np.exp(1j * anchor_phase)
    phases = np.angle(phase_products).astype(np.float64)
    frames = connection.frames
    vectors = (
        np.cos(phases)[:, None] * frames.first_axes()
        + np.sin(phases)[:, None] * frames.second_axes()
    )
    _, crossing_error = _propagate_phases(connection)
    evidence = DirectionFieldEvidence(crossing_error, capability.evidence.limit)
    field: FaceDirectionField[K] = object.__new__(FaceDirectionField)
    object.__setattr__(field, "_geometry", connection.geometry)
    object.__setattr__(field, "_connection", connection)
    object.__setattr__(field, "_anchor_face", 0)
    object.__setattr__(field, "_anchor_phase", float(anchor_phase))
    object.__setattr__(field, "_phases", _freeze_array(phases))
    object.__setattr__(field, "_vectors", _freeze_array(vectors))
    object.__setattr__(field, "_sealed", True)
    return Certified(field, evidence)


__all__ = [
    "DirectionFieldEvidence",
    "Disk",
    "FaceDirectionField",
    "FaceVectors",
    "FrozenFlowEvidence",
    "HolonomyEvidence",
    "IntegralDualCycles",
    "IntegrableConnection",
    "SurfaceConnection",
    "SurfaceError",
    "TriangleFrames",
    "VertexVectors",
    "admit_integrable_connection",
    "connection_holonomy",
    "disk",
    "face_unit_normals",
    "gaussian_curvature_measure",
    "integral_dual_cycles",
    "integrate_direction_field",
    "levi_civita_connection",
    "mean_curvature_flow_step",
    "mean_curvature_vectors",
    "sphere_inscribed_vertex_normals",
    "surface_area_gradient",
    "surface_connection",
    "tip_angle_vertex_normals",
    "triangle_frames",
    "uniform_vertex_normals",
    "volume_gradient",
]
