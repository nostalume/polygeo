//! Proof-bearing native mathematical core for `PolyGeo`.

#[path = "chain.rs"]
mod chain_impl;
mod coefficient;
mod complex;
mod correspondence;
mod csr;
mod evidence;
#[path = "field.rs"]
mod field_impl;
#[path = "form.rs"]
mod form_impl;
#[path = "geometry.rs"]
mod geometry_impl;
mod halfedge;
mod homology;
mod incidence;
mod mask;
mod numeric;
mod operator;
#[path = "solve.rs"]
mod solve_impl;
mod subset;
mod surface;
pub mod topology;

/// Exact coefficients, chains, maps, homology, and bounded CSR materialization.
pub mod chain {
    pub use crate::chain_impl::{
        BasisIdentification, Chain, ChainComplex, ChainError, ChainIsomorphism, ChainLawLimit,
        ChainMap, Cochain, CompositionError, DualChainIsomorphism, DualComplex, Element,
        IntegralChain, IntegralChainComplex, IntegralCochain, IntegralLinearMap, IsomorphismError,
        LinearMap, PresentationEquality, PresentationError, Space, Variance, compose,
    };
    pub use crate::coefficient::{
        BigIntEncoding, CoefficientSystem, CommutativeRing, EuclideanDomain, ExactRational, Field,
        FractionField, FractionFieldOf, IntegerRing, IntegralDomain, RationalField,
        ReducedFractionEncoding, Ring, RingMorphism, ValueEncoding,
    };
    pub use crate::csr::{Csr, CsrBuildLimit, CsrError, CsrEstimate};
    pub use crate::homology::{HomologyError, HomologyGroup, HomologyLimit, IntegralHomology};
}

/// Binary64 chains, cochains, and matrix-free operators.
pub mod form {
    pub use crate::form_impl::{
        Binary64Chain as Chain, Binary64ChainSpace as ChainSpace, Binary64Cochain as Cochain,
        Binary64CochainSpace as CochainSpace, Binary64Element as Element,
        Binary64ElementError as ElementError, Binary64Space as Space,
    };
    pub use crate::operator::{LinearOperator as Operator, OperatorError, compose};
}

/// Admitted Euclidean geometry, metric evidence, and surface algorithms.
pub mod geometry {
    pub use crate::geometry_impl::{
        CircumcentricPairing, Geometry, GeometryError, Limit, Metric, MetricError,
        NondegenerateCapability, NondegeneratePairing, PairingCapability,
    };
    pub use crate::surface::{
        EntityVectors as VectorField, FaceVectors as FaceField, FlowEvidence, FlowStep,
        LeastSquaresConformalMapEvidence as ConformalMapEvidence,
        LeastSquaresConformalMapSolution as ConformalMap, SurfaceError, TriangleSurface,
        VertexVectors as VertexField,
    };
}

/// Numerical problem admission, preparation, workspace, policy, and results.
pub mod solve {
    pub use crate::solve_impl::{
        CancellationToken, DirichletEvidence, DirichletProblem, DirichletResult, Executor,
        HarmonicExtension, HeatProblem, HeatResult, PoissonProblem, PoissonResult, Policy,
        Prepared, Problem, ProblemError, ResidualEvidence, SolveError, SolveExt, StorageLimit,
        SurfaceComputationError, WorkLimit, Workspace,
    };
}

/// Discrete connections, Hodge decomposition, harmonic bases, and direction fields.
pub mod field {
    pub use crate::chain_impl::IntegralDualCycleBasis as DualCycles;
    pub use crate::field_impl::{
        DirectionFieldSingularities as Singularities, FaceDirectionField as Direction,
        HarmonicOneFormBasis as HarmonicBasis, HolonomyEvidence as Holonomy, IntegrableConnection,
        SurfaceConnection as Connection,
    };
    pub use crate::solve_impl::{HodgeDecomposition, HodgeEvidence, HodgeProblem};
}

pub(crate) use chain_impl::{
    Chain, ChainComplex, ChainError, ChainIsomorphism, ChainLawLimit, Cochain, Element,
    IntegralChain, IntegralChainComplex, IntegralCochain, IntegralDualCycleBasis, IsomorphismError,
    LinearMap, Space, Variance,
};
pub(crate) use coefficient::{BigIntEncoding, CoefficientSystem, ExactRational, IntegerRing};
pub(crate) use complex::{CandidateInput, ComplexCore};
pub(crate) use form_impl::{
    Binary64Chain, Binary64ChainSpace, Binary64Cochain, Binary64CochainSpace, Binary64Element,
    Binary64ElementError, Binary64Space,
};
pub(crate) use geometry_impl::{
    CircumcentricPairing, Geometry, GeometryError, Limit, Metric, NondegenerateCapability,
    NondegeneratePairing, PairingCapability,
};
pub(crate) use halfedge::{HalfedgeInput, HalfedgeSurfaceCore, MaterialFace};
pub(crate) use homology::HomologyGroup;
pub(crate) use incidence::{Basis, BoundaryRef, CanonicalBoundary, ChainView, CoefficientSlice};
pub(crate) use operator::{LinearOperator, OperatorError};
pub(crate) use solve_impl::{
    CancellationToken, Executor, Policy, SolveError, StorageLimit, SurfaceComputationError,
    WorkLimit,
};
pub(crate) use subset::CanonicalSelection;
pub(crate) use surface::{FaceVectors, SurfaceError};
pub(crate) use topology::TopologyError;
