"""Public API for the PolyGeo native mathematical core."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Any

from ._polygeo_native import (
    BigIntEncoding,
    Binary64Chain,
    Binary64ChainSpace,
    Binary64Cochain,
    Binary64CochainSpace,
    Binary64Element,
    Binary64ElementError,
    Binary64Space,
    CancellationToken,
    Chain,
    ChainComplex,
    ChainError,
    ChainIsomorphism,
    ChainLawLimit,
    Cochain,
    CochainComplex,
    Complex,
    CsrBuildLimit,
    CsrEstimate,
    CsrRepresentation,
    DEFAULT_CHAIN_LAW_LIMIT,
    DEFAULT_HOMOLOGY_LIMIT,
    DEFAULT_REALIZATION_LIMIT,
    DirichletSolution,
    Element,
    EntityVectors,
    EuclideanRealization,
    FaceDirectionField,
    FaceVectors,
    FlowStep,
    Form,
    GeometryError,
    HalfedgeError,
    HalfedgeSurface,
    HodgeDecomposition,
    HolonomyEvidence,
    HomologyError,
    HomologyGroup,
    HomologyLimit,
    IntegerCsrParts,
    IntegerRing,
    IntegralDualCycleBasis,
    IntegralHomology,
    IntegrableConnection,
    LinearMap,
    LinearOperator,
    NativeExecutor,
    OperatorError,
    PoissonSolution,
    PositiveMetric,
    PreparedProblem,
    Problem,
    ProblemError,
    QQ,
    RationalCsrParts,
    RationalField,
    RealizationLimit,
    ReducedFractionEncoding,
    SimplexSelection,
    SimplexSubset,
    SimplicialError,
    SolveError,
    SolveWorkspace,
    Space,
    StorageLimit,
    SurfaceConnection,
    SurfaceCorrespondence,
    SurfaceError,
    TriangleSurface,
    VertexVectors,
    WorkLimit,
    ZZ,
    prepare_integral_homology,
    topological_boundary,
)
from .mesh import MeshError, load_surface
from .plotting import (
    PlotError,
    plot_form,
    plot_geometry,
    plot_homology_cycle,
    plot_surface_vectors,
)

Geometry = EuclideanRealization

if TYPE_CHECKING:
    type IntegralChainComplex = ChainComplex[int]
    type RationalChainComplex = ChainComplex[Fraction]
    type IntegralCochainComplex = CochainComplex[int]
    type RationalCochainComplex = CochainComplex[Fraction]
    type IntegralChainSpace[Degree: int] = Space[int, Chain, Degree]
    type RationalChainSpace[Degree: int] = Space[Fraction, Chain, Degree]
    type IntegralCochainSpace[Degree: int] = Space[int, Cochain, Degree]
    type RationalCochainSpace[Degree: int] = Space[Fraction, Cochain, Degree]
    type IntegralChain[Degree: int] = Element[IntegralChainSpace[Degree]]
    type RationalChain[Degree: int] = Element[RationalChainSpace[Degree]]
    type IntegralCochain[Degree: int] = Element[IntegralCochainSpace[Degree]]
    type RationalCochain[Degree: int] = Element[RationalCochainSpace[Degree]]
    type IntegralLinearMap[
        SourceSpace: Space[int, Any, int],
        TargetSpace: Space[int, Any, int],
    ] = LinearMap[SourceSpace, TargetSpace]
    type RationalLinearMap[
        SourceSpace: Space[Fraction, Any, int],
        TargetSpace: Space[Fraction, Any, int],
    ] = LinearMap[SourceSpace, TargetSpace]
else:
    IntegralChainComplex = ChainComplex
    RationalChainComplex = ChainComplex
    IntegralCochainComplex = CochainComplex
    RationalCochainComplex = CochainComplex
    IntegralChainSpace = Space
    RationalChainSpace = Space
    IntegralCochainSpace = Space
    RationalCochainSpace = Space
    IntegralChain = Element
    RationalChain = Element
    IntegralCochain = Element
    RationalCochain = Element
    IntegralLinearMap = LinearMap
    RationalLinearMap = LinearMap


__all__ = [
    "BigIntEncoding",
    "Binary64Chain",
    "Binary64ChainSpace",
    "Binary64Cochain",
    "Binary64CochainSpace",
    "Binary64Element",
    "Binary64ElementError",
    "Binary64Space",
    "CancellationToken",
    "Chain",
    "ChainComplex",
    "ChainError",
    "ChainIsomorphism",
    "ChainLawLimit",
    "Cochain",
    "CochainComplex",
    "Complex",
    "CsrBuildLimit",
    "CsrEstimate",
    "CsrRepresentation",
    "DEFAULT_CHAIN_LAW_LIMIT",
    "DEFAULT_HOMOLOGY_LIMIT",
    "DEFAULT_REALIZATION_LIMIT",
    "DirichletSolution",
    "Element",
    "EntityVectors",
    "EuclideanRealization",
    "FaceDirectionField",
    "FaceVectors",
    "FlowStep",
    "Form",
    "Geometry",
    "GeometryError",
    "HodgeDecomposition",
    "HolonomyEvidence",
    "HomologyError",
    "HomologyGroup",
    "HomologyLimit",
    "HalfedgeError",
    "HalfedgeSurface",
    "IntegralChain",
    "IntegralChainComplex",
    "IntegralChainSpace",
    "IntegralCochain",
    "IntegralCochainComplex",
    "IntegralCochainSpace",
    "IntegralDualCycleBasis",
    "IntegralHomology",
    "IntegralLinearMap",
    "IntegrableConnection",
    "IntegerCsrParts",
    "IntegerRing",
    "LinearOperator",
    "LinearMap",
    "MeshError",
    "NativeExecutor",
    "OperatorError",
    "PlotError",
    "PoissonSolution",
    "PositiveMetric",
    "PreparedProblem",
    "Problem",
    "ProblemError",
    "QQ",
    "RationalChain",
    "RationalChainComplex",
    "RationalChainSpace",
    "RationalCochain",
    "RationalCochainComplex",
    "RationalCochainSpace",
    "RationalCsrParts",
    "RationalField",
    "RationalLinearMap",
    "RealizationLimit",
    "ReducedFractionEncoding",
    "SimplexSubset",
    "SimplexSelection",
    "SimplicialError",
    "SolveError",
    "SolveWorkspace",
    "Space",
    "StorageLimit",
    "SurfaceConnection",
    "SurfaceCorrespondence",
    "SurfaceError",
    "TriangleSurface",
    "VertexVectors",
    "WorkLimit",
    "ZZ",
    "load_surface",
    "plot_form",
    "plot_geometry",
    "plot_homology_cycle",
    "plot_surface_vectors",
    "prepare_integral_homology",
    "topological_boundary",
]
