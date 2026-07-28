"""Public API for the PolyGeo simplicial core."""

from os import PathLike
from pathlib import Path

import numpy as np

from .algorithms import (
    AlgorithmError,
    BasisCoordinates,
    Certified,
    ConditionEvidence,
    HodgeComponents,
    HodgeEvidence,
    MeanZeroProblem,
    PositiveHodgeMetric,
    RealHomologyBasis,
    ResidualEvidence,
    VertexMap,
    assemble_poisson,
    harmonic_extension,
    hodge_decomposition,
    impose_mean_zero,
    real_homology_basis,
    vertex_map,
)
from .geometry import Geometry, GeometryError
from .operators import (
    DualCochainSpace,
    LinearMap,
    OperatorError,
    codifferential,
    extend_zero,
    exterior_derivative,
    hodge_laplacian,
    hodge_star,
    restrict,
    weighted_pairing,
)
from .plotting import (
    PlotError,
    plot_cochain,
    plot_geometry,
    plot_homology_cycle,
    plot_surface_vectors,
)
from .simplicial import (
    ORDINARY_FORM,
    BoundaryState,
    BoundaryUnknown,
    CochainSpace,
    CochainSubspace,
    CodimensionOneRegular,
    Complex,
    Connected,
    ConnectivityState,
    ConnectivityUnknown,
    FieldSemantics,
    Form,
    OneForm,
    OrdinaryForm,
    OrientationState,
    OrientationUnknown,
    Oriented,
    Simplicial,
    SimplicialError,
    SimplexSubset,
    TopologyState,
    TriangleManifold,
    TwoForm,
    WithBoundary,
    WithoutBoundary,
    ZeroForm,
    topological_boundary,
)
from .solvers import (
    LeastSquaresSolution,
    LinearSolution,
    NumericalError,
    PrepareLeastSquares,
    PrepareLinearSolve,
    PreparedLeastSquares,
    PreparedLinearSolve,
    prepare_direct,
    prepare_least_squares,
)
from .systems import (
    AssembledSystem,
    DirichletProblem,
    SystemError,
    eliminate_dirichlet,
)
from .surface import (
    DirectionFieldEvidence,
    Disk,
    FaceDirectionField,
    FaceVectors,
    FrozenFlowEvidence,
    HolonomyEvidence,
    IntegralDualCycles,
    IntegrableConnection,
    SurfaceConnection,
    SurfaceError,
    TriangleFrames,
    VertexVectors,
    admit_integrable_connection,
    connection_holonomy,
    disk,
    face_unit_normals,
    gaussian_curvature_measure,
    integral_dual_cycles,
    integrate_direction_field,
    levi_civita_connection,
    mean_curvature_flow_step,
    mean_curvature_vectors,
    sphere_inscribed_vertex_normals,
    surface_area_gradient,
    surface_connection,
    tip_angle_vertex_normals,
    triangle_frames,
    uniform_vertex_normals,
    volume_gradient,
)


class MeshError(ValueError):
    """Invalid mesh input or unavailable optional mesh-loading behavior."""


def load_surface(source: str | PathLike[str]) -> Geometry[Complex]:
    """Load one triangular surface as an owned, unrefined geometry value."""
    try:
        import trimesh
    except ModuleNotFoundError as error:
        if error.name == "trimesh":
            raise MeshError(
                "mesh input requires the optional polygeo[mesh] dependency"
            ) from error
        raise MeshError("mesh input backend failed to import") from error
    except ImportError as error:
        raise MeshError("mesh input backend failed to import") from error

    try:
        payload = trimesh.load(Path(source), process=False)
    except Exception as error:
        raise MeshError("failed to load surface mesh") from error

    if not isinstance(payload, trimesh.Trimesh):
        raise MeshError("surface input must contain exactly one triangular mesh")

    try:
        positions = np.asarray(payload.vertices)
        faces = np.asarray(payload.faces)
        finite_positions = np.all(np.isfinite(positions))
    except Exception as error:
        raise MeshError("surface mesh is not admissible") from error
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise MeshError("surface input must contain triangular faces")
    if positions.ndim != 2 or not finite_positions:
        raise MeshError("surface input must contain finite vertex positions")

    try:
        complex_ = Complex.from_maximal_simplices(faces, vertex_count=len(positions))
        return Geometry.from_positions(complex_, positions)
    except (GeometryError, SimplicialError, TypeError, ValueError) as error:
        raise MeshError("surface mesh is not admissible") from error


__all__ = [
    "ORDINARY_FORM",
    "AlgorithmError",
    "AssembledSystem",
    "BasisCoordinates",
    "BoundaryState",
    "BoundaryUnknown",
    "Certified",
    "CochainSpace",
    "CochainSubspace",
    "CodimensionOneRegular",
    "Complex",
    "ConditionEvidence",
    "Connected",
    "ConnectivityState",
    "ConnectivityUnknown",
    "DirichletProblem",
    "DirectionFieldEvidence",
    "Disk",
    "DualCochainSpace",
    "FaceDirectionField",
    "FaceVectors",
    "FieldSemantics",
    "Form",
    "FrozenFlowEvidence",
    "Geometry",
    "GeometryError",
    "HodgeComponents",
    "HodgeEvidence",
    "HolonomyEvidence",
    "IntegralDualCycles",
    "IntegrableConnection",
    "LeastSquaresSolution",
    "LinearMap",
    "LinearSolution",
    "MeanZeroProblem",
    "MeshError",
    "NumericalError",
    "OneForm",
    "OrdinaryForm",
    "PlotError",
    "PositiveHodgeMetric",
    "RealHomologyBasis",
    "ResidualEvidence",
    "OrientationState",
    "OrientationUnknown",
    "OperatorError",
    "Oriented",
    "PrepareLeastSquares",
    "PrepareLinearSolve",
    "PreparedLeastSquares",
    "PreparedLinearSolve",
    "SimplexSubset",
    "Simplicial",
    "SimplicialError",
    "SystemError",
    "SurfaceConnection",
    "SurfaceError",
    "TopologyState",
    "TriangleFrames",
    "TriangleManifold",
    "TwoForm",
    "VertexMap",
    "VertexVectors",
    "WithBoundary",
    "WithoutBoundary",
    "ZeroForm",
    "admit_integrable_connection",
    "assemble_poisson",
    "codifferential",
    "connection_holonomy",
    "eliminate_dirichlet",
    "extend_zero",
    "disk",
    "face_unit_normals",
    "gaussian_curvature_measure",
    "integral_dual_cycles",
    "integrate_direction_field",
    "levi_civita_connection",
    "harmonic_extension",
    "hodge_decomposition",
    "exterior_derivative",
    "hodge_laplacian",
    "hodge_star",
    "impose_mean_zero",
    "load_surface",
    "mean_curvature_flow_step",
    "mean_curvature_vectors",
    "prepare_direct",
    "prepare_least_squares",
    "plot_cochain",
    "plot_geometry",
    "plot_homology_cycle",
    "plot_surface_vectors",
    "real_homology_basis",
    "restrict",
    "sphere_inscribed_vertex_normals",
    "surface_area_gradient",
    "surface_connection",
    "tip_angle_vertex_normals",
    "topological_boundary",
    "triangle_frames",
    "uniform_vertex_normals",
    "volume_gradient",
    "weighted_pairing",
    "vertex_map",
]
