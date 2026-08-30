//! Proof-bearing native mathematical core for `PolyGeo`.

mod chain;
mod coefficient;
mod complex;
mod correspondence;
mod evidence;
mod form;
mod halfedge;
mod homology;
mod incidence;
mod mask;
pub mod operator;
mod problem;
mod realization;
mod representation;
mod solve;
mod sparse;
mod subset;
mod surface;

pub use chain::{
    BasisIdentification, Chain, ChainComplex, ChainError, ChainIsomorphism, ChainLawLimit,
    ChainMap, Cochain, CompositionError, DualChainIsomorphism, DualComplex, Element, IntegralChain,
    IntegralChainComplex, IntegralCochain, IntegralDualCycleBasis, IntegralLinearMap,
    IsomorphismError, LinearMap, PresentationEquality, PresentationError, Space, Variance, compose,
};
pub use coefficient::{
    BigIntEncoding, CoefficientSystem, CommutativeRing, EuclideanDomain, ExactRational, Field,
    FractionField, FractionFieldOf, IntegerRing, IntegralDomain, RationalField,
    ReducedFractionEncoding, Ring, RingMorphism, ValueEncoding,
};
pub use complex::{CandidateInput, ComplexCore};
pub use correspondence::{CorrespondenceDirection, SignedPermutation, SurfaceCorrespondence};
pub use evidence::{
    CodimensionOneRegularCapability, ConnectedCapability, ConnectedView, DiskView,
    OrientedCapability, OrientedView, RegularView, SimplicialCapability,
    TriangleManifoldCapability, TriangleView, WithBoundaryCapability, WithBoundaryView,
    WithoutBoundaryCapability, WithoutBoundaryView,
};
pub use form::{
    Binary64Chain, Binary64ChainSpace, Binary64Cochain, Binary64CochainSpace, Binary64Element,
    Binary64ElementError, Binary64Space,
};
pub use halfedge::{
    Edge, ExteriorHalfedge, FaceKind, FaceOrbit, Halfedge, HalfedgeInput, HalfedgeSurfaceCore,
    MaterialBoundaryHalfedge, MaterialFace, OrbitMaterialization, OrbitMaterializationParts,
    Vertex,
};
pub use homology::{HomologyError, HomologyGroup, HomologyLimit, IntegralHomology};
pub use incidence::{
    Basis, BoundaryRef, CanonicalBoundary, ChainView, CoefficientSlice, ExactEntries,
};
pub use operator::{LinearOperator, OperatorError};
pub use problem::{
    DirichletEvidence, DirichletProblem, DirichletSolution, HarmonicExtension, HeatProblem,
    HeatSolution, HodgeDecomposition, HodgeEvidence, HodgeProblem, MeanZeroPoisson,
    PoissonSolution, Problem, ProblemError, ResidualEvidence,
};
pub use realization::{
    CircumcentricPairing, EuclideanRealization, MetricError, NondegenerateCapability,
    NondegeneratePairing, PairingCapability, PositiveMetric, RealizationError, RealizationLimit,
};
pub use representation::{
    CsrBuildLimit, CsrEstimate, CsrMatrix, CsrPattern, CsrRepresentation, RepresentationError,
};
pub use solve::{
    CancellationToken, NativeExecutor, Prepared, SolveError, SolveExt, SolveWorkspace,
    SurfaceComputationError,
};
pub use subset::{CanonicalSelection, SimplexSubset, SubsetBuilder};
pub use surface::{
    EntityVectors, FaceDirectionField, FaceVectors, FlowEvidence, FlowStep, HolonomyEvidence,
    IntegrableConnection, LeastSquaresConformalMapEvidence, LeastSquaresConformalMapSolution,
    SurfaceConnection, SurfaceError, TriangleSurface, VertexVectors,
};

/// Portable logical-storage ceiling for one unpublished operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StorageLimit {
    retained_logical_bytes: u64,
    peak_live_logical_bytes: u64,
}

impl StorageLimit {
    /// Construct a valid lifecycle ceiling.
    #[must_use]
    pub const fn new(retained_logical_bytes: u64, peak_live_logical_bytes: u64) -> Option<Self> {
        if retained_logical_bytes <= peak_live_logical_bytes {
            Some(Self {
                retained_logical_bytes,
                peak_live_logical_bytes,
            })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn retained_logical_bytes(self) -> u64 {
        self.retained_logical_bytes
    }

    #[must_use]
    pub const fn peak_live_logical_bytes(self) -> u64 {
        self.peak_live_logical_bytes
    }
}

/// Platform-stable ceiling whose charged step is defined by the consuming operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkLimit(u64);

impl WorkLimit {
    #[must_use]
    pub const fn new(steps: u64) -> Self {
        Self(steps)
    }

    #[must_use]
    pub const fn steps(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TopologyErrorKind {
    CandidateShape,
    EmptyMaximalSimplices,
    NegativeIndex,
    IndexOverflow,
    RepeatedVertex,
    DuplicateMaximalSimplex,
    VertexExtent,
    CountOverflow,
    Allocation,
    MaskShape,
    MaskIndexOutside,
    SelectionNotStrict,
    SelectionIndexOutside,
    DegreeOutside,
    TriangleDimension,
    NotPure,
    CodimensionOneIncidence,
    VertexLink,
    Orientation,
    Disconnected,
    BoundaryPresent,
    BoundaryAbsent,
    DiskBoundaryComponents,
    DiskEulerCharacteristic,
    CapabilityNotAdmitted,
    OwnerMismatch,
    HalfedgeShape,
    HalfedgeRange,
    HalfedgePermutation,
    TwinLaw,
    ExteriorInconsistency,
    BoundaryCycle,
    ConversionNotSimplicial,
    CorrespondenceLaw,
    InternalInvariant,
}

/// Bounded structured counterexample carried across the language boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TopologyDetails {
    None,
    NegativeIndex {
        value: i128,
    },
    IndexOverflow {
        value: u128,
    },
    RepeatedVertex {
        vertex: usize,
    },
    VertexExtent {
        declared: usize,
        required: usize,
    },
    Degree {
        degree: usize,
    },
    ActualDimension {
        actual_dimension: usize,
    },
    Vertex {
        vertex: usize,
    },
    Incidence {
        degree: usize,
        simplex: usize,
        cofaces: usize,
    },
    CodimensionOneSimplex {
        codimension_one_simplex: usize,
    },
    UnreachableVertex {
        unreachable_vertex: usize,
    },
    BoundaryComponents {
        boundary_components: usize,
    },
    EulerCharacteristic {
        euler_characteristic: i128,
    },
    Capability {
        capability: &'static str,
    },
    HalfedgeEntry {
        relation: &'static str,
        halfedge: usize,
        value: usize,
        bound: usize,
    },
    Twin {
        halfedge: usize,
        twin: usize,
        twin_back: usize,
    },
    Exterior {
        halfedge: usize,
        related: usize,
    },
}

/// Scalar value in the stable named-field projection of topology details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TopologyDetailValue {
    Signed(i128),
    Unsigned(u128),
    Index(usize),
    Text(&'static str),
}

/// One named scalar in the stable projection of a topology counterexample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopologyDetailField {
    name: &'static str,
    value: TopologyDetailValue,
}

impl TopologyDetailField {
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.name
    }

    #[must_use]
    pub const fn value(self) -> TopologyDetailValue {
        self.value
    }
}

impl TopologyDetails {
    /// Project variant-specific details into one adapter-facing field schema.
    // Keeping this one exhaustive match intact makes field names reviewable as
    // a single schema and prevents adapters from rebuilding variant knowledge.
    #[expect(
        clippy::too_many_lines,
        reason = "one exhaustive adapter schema keeps variant fields reviewable"
    )]
    pub fn fields(self) -> impl Iterator<Item = TopologyDetailField> {
        let field = |name, value| Some(TopologyDetailField { name, value });
        let none = None;
        match self {
            Self::None => [none, none, none, none],
            Self::NegativeIndex { value } => [
                field("value", TopologyDetailValue::Signed(value)),
                none,
                none,
                none,
            ],
            Self::IndexOverflow { value } => [
                field("value", TopologyDetailValue::Unsigned(value)),
                none,
                none,
                none,
            ],
            Self::RepeatedVertex { vertex } | Self::Vertex { vertex } => [
                field("vertex", TopologyDetailValue::Index(vertex)),
                none,
                none,
                none,
            ],
            Self::VertexExtent { declared, required } => [
                field("declared", TopologyDetailValue::Index(declared)),
                field("required", TopologyDetailValue::Index(required)),
                none,
                none,
            ],
            Self::Degree { degree } => [
                field("degree", TopologyDetailValue::Index(degree)),
                none,
                none,
                none,
            ],
            Self::ActualDimension { actual_dimension } => [
                field(
                    "actual_dimension",
                    TopologyDetailValue::Index(actual_dimension),
                ),
                none,
                none,
                none,
            ],
            Self::Incidence {
                degree,
                simplex,
                cofaces,
            } => [
                field("degree", TopologyDetailValue::Index(degree)),
                field("simplex", TopologyDetailValue::Index(simplex)),
                field("cofaces", TopologyDetailValue::Index(cofaces)),
                none,
            ],
            Self::CodimensionOneSimplex {
                codimension_one_simplex,
            } => [
                field(
                    "codimension_one_simplex",
                    TopologyDetailValue::Index(codimension_one_simplex),
                ),
                none,
                none,
                none,
            ],
            Self::UnreachableVertex { unreachable_vertex } => [
                field(
                    "unreachable_vertex",
                    TopologyDetailValue::Index(unreachable_vertex),
                ),
                none,
                none,
                none,
            ],
            Self::BoundaryComponents {
                boundary_components,
            } => [
                field(
                    "boundary_components",
                    TopologyDetailValue::Index(boundary_components),
                ),
                none,
                none,
                none,
            ],
            Self::EulerCharacteristic {
                euler_characteristic,
            } => [
                field(
                    "euler_characteristic",
                    TopologyDetailValue::Signed(euler_characteristic),
                ),
                none,
                none,
                none,
            ],
            Self::Capability { capability } => [
                field("capability", TopologyDetailValue::Text(capability)),
                none,
                none,
                none,
            ],
            Self::HalfedgeEntry {
                relation,
                halfedge,
                value,
                bound,
            } => [
                field("relation", TopologyDetailValue::Text(relation)),
                field("halfedge", TopologyDetailValue::Index(halfedge)),
                field("value", TopologyDetailValue::Index(value)),
                field("bound", TopologyDetailValue::Index(bound)),
            ],
            Self::Twin {
                halfedge,
                twin,
                twin_back,
            } => [
                field("halfedge", TopologyDetailValue::Index(halfedge)),
                field("twin", TopologyDetailValue::Index(twin)),
                field("twin_back", TopologyDetailValue::Index(twin_back)),
                none,
            ],
            Self::Exterior { halfedge, related } => [
                field("halfedge", TopologyDetailValue::Index(halfedge)),
                field("related", TopologyDetailValue::Index(related)),
                none,
                none,
            ],
        }
        .into_iter()
        .flatten()
    }
}

/// Stable reason, message, and optional mathematical counterexample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopologyError {
    kind: TopologyErrorKind,
    details: TopologyDetails,
}

#[expect(
    non_upper_case_globals,
    reason = "legacy error constants retain their established public spelling"
)]
impl TopologyError {
    pub const CandidateShape: Self = Self::plain(TopologyErrorKind::CandidateShape);
    pub const EmptyMaximalSimplices: Self = Self::plain(TopologyErrorKind::EmptyMaximalSimplices);
    pub const NegativeIndex: Self = Self::plain(TopologyErrorKind::NegativeIndex);
    pub const IndexOverflow: Self = Self::plain(TopologyErrorKind::IndexOverflow);
    pub const RepeatedVertex: Self = Self::plain(TopologyErrorKind::RepeatedVertex);
    pub const DuplicateMaximalSimplex: Self =
        Self::plain(TopologyErrorKind::DuplicateMaximalSimplex);
    pub const VertexExtent: Self = Self::plain(TopologyErrorKind::VertexExtent);
    pub const CountOverflow: Self = Self::plain(TopologyErrorKind::CountOverflow);
    /// A fallible topology-buffer reservation failed.
    ///
    /// This does not promise recovery from allocator or process aborts outside
    /// Rust's fallible reservation APIs.
    pub const Allocation: Self = Self::plain(TopologyErrorKind::Allocation);
    pub const MaskShape: Self = Self::plain(TopologyErrorKind::MaskShape);
    pub const MaskIndexOutside: Self = Self::plain(TopologyErrorKind::MaskIndexOutside);
    pub const SelectionNotStrict: Self = Self::plain(TopologyErrorKind::SelectionNotStrict);
    pub const SelectionIndexOutside: Self = Self::plain(TopologyErrorKind::SelectionIndexOutside);
    pub const DegreeOutside: Self = Self::plain(TopologyErrorKind::DegreeOutside);
    pub const TriangleDimension: Self = Self::plain(TopologyErrorKind::TriangleDimension);
    pub const NotPure: Self = Self::plain(TopologyErrorKind::NotPure);
    pub const CodimensionOneIncidence: Self =
        Self::plain(TopologyErrorKind::CodimensionOneIncidence);
    pub const VertexLink: Self = Self::plain(TopologyErrorKind::VertexLink);
    pub const Orientation: Self = Self::plain(TopologyErrorKind::Orientation);
    pub const Disconnected: Self = Self::plain(TopologyErrorKind::Disconnected);
    pub const BoundaryPresent: Self = Self::plain(TopologyErrorKind::BoundaryPresent);
    pub const BoundaryAbsent: Self = Self::plain(TopologyErrorKind::BoundaryAbsent);
    pub const DiskBoundaryComponents: Self = Self::plain(TopologyErrorKind::DiskBoundaryComponents);
    pub const DiskEulerCharacteristic: Self =
        Self::plain(TopologyErrorKind::DiskEulerCharacteristic);
    pub const CapabilityNotAdmitted: Self = Self::plain(TopologyErrorKind::CapabilityNotAdmitted);
    pub const OwnerMismatch: Self = Self::plain(TopologyErrorKind::OwnerMismatch);
    pub const HalfedgeShape: Self = Self::plain(TopologyErrorKind::HalfedgeShape);
    pub const HalfedgePermutation: Self = Self::plain(TopologyErrorKind::HalfedgePermutation);
    pub const TwinLaw: Self = Self::plain(TopologyErrorKind::TwinLaw);
    pub const ExteriorInconsistency: Self = Self::plain(TopologyErrorKind::ExteriorInconsistency);
    pub const BoundaryCycle: Self = Self::plain(TopologyErrorKind::BoundaryCycle);
    pub const ConversionNotSimplicial: Self =
        Self::plain(TopologyErrorKind::ConversionNotSimplicial);
    pub const CorrespondenceLaw: Self = Self::plain(TopologyErrorKind::CorrespondenceLaw);
    pub const InternalInvariant: Self = Self::plain(TopologyErrorKind::InternalInvariant);

    const fn plain(kind: TopologyErrorKind) -> Self {
        Self {
            kind,
            details: TopologyDetails::None,
        }
    }

    #[must_use]
    pub const fn negative_index(value: i128) -> Self {
        Self {
            kind: TopologyErrorKind::NegativeIndex,
            details: TopologyDetails::NegativeIndex { value },
        }
    }

    #[must_use]
    pub const fn index_overflow(value: u128) -> Self {
        Self {
            kind: TopologyErrorKind::IndexOverflow,
            details: TopologyDetails::IndexOverflow { value },
        }
    }

    #[must_use]
    pub const fn repeated_vertex(vertex: usize) -> Self {
        Self {
            kind: TopologyErrorKind::RepeatedVertex,
            details: TopologyDetails::RepeatedVertex { vertex },
        }
    }

    #[must_use]
    pub const fn vertex_extent(declared: usize, required: usize) -> Self {
        Self {
            kind: TopologyErrorKind::VertexExtent,
            details: TopologyDetails::VertexExtent { declared, required },
        }
    }

    #[must_use]
    pub const fn degree_outside(degree: usize) -> Self {
        Self {
            kind: TopologyErrorKind::DegreeOutside,
            details: TopologyDetails::Degree { degree },
        }
    }

    #[must_use]
    pub const fn triangle_dimension(actual_dimension: usize) -> Self {
        Self {
            kind: TopologyErrorKind::TriangleDimension,
            details: TopologyDetails::ActualDimension { actual_dimension },
        }
    }

    #[must_use]
    pub const fn not_pure(vertex: usize) -> Self {
        Self {
            kind: TopologyErrorKind::NotPure,
            details: TopologyDetails::Vertex { vertex },
        }
    }

    #[must_use]
    pub const fn codimension_one_incidence(degree: usize, simplex: usize, cofaces: usize) -> Self {
        Self {
            kind: TopologyErrorKind::CodimensionOneIncidence,
            details: TopologyDetails::Incidence {
                degree,
                simplex,
                cofaces,
            },
        }
    }

    #[must_use]
    pub const fn vertex_link(vertex: usize) -> Self {
        Self {
            kind: TopologyErrorKind::VertexLink,
            details: TopologyDetails::Vertex { vertex },
        }
    }

    #[must_use]
    pub const fn orientation(codimension_one_simplex: usize) -> Self {
        Self {
            kind: TopologyErrorKind::Orientation,
            details: TopologyDetails::CodimensionOneSimplex {
                codimension_one_simplex,
            },
        }
    }

    #[must_use]
    pub const fn disconnected(unreachable_vertex: usize) -> Self {
        Self {
            kind: TopologyErrorKind::Disconnected,
            details: TopologyDetails::UnreachableVertex { unreachable_vertex },
        }
    }

    #[must_use]
    pub const fn disk_boundary_components(boundary_components: usize) -> Self {
        Self {
            kind: TopologyErrorKind::DiskBoundaryComponents,
            details: TopologyDetails::BoundaryComponents {
                boundary_components,
            },
        }
    }

    #[must_use]
    pub const fn disk_euler_characteristic(euler_characteristic: i128) -> Self {
        Self {
            kind: TopologyErrorKind::DiskEulerCharacteristic,
            details: TopologyDetails::EulerCharacteristic {
                euler_characteristic,
            },
        }
    }

    #[must_use]
    pub const fn boundary_present(codimension_one_simplex: usize) -> Self {
        Self {
            kind: TopologyErrorKind::BoundaryPresent,
            details: TopologyDetails::CodimensionOneSimplex {
                codimension_one_simplex,
            },
        }
    }

    #[must_use]
    pub const fn capability_not_admitted(capability: &'static str) -> Self {
        Self {
            kind: TopologyErrorKind::CapabilityNotAdmitted,
            details: TopologyDetails::Capability { capability },
        }
    }

    #[must_use]
    pub const fn halfedge_range(
        relation: &'static str,
        halfedge: usize,
        value: usize,
        bound: usize,
    ) -> Self {
        Self {
            kind: TopologyErrorKind::HalfedgeRange,
            details: TopologyDetails::HalfedgeEntry {
                relation,
                halfedge,
                value,
                bound,
            },
        }
    }

    #[must_use]
    pub const fn halfedge_permutation(
        relation: &'static str,
        halfedge: usize,
        value: usize,
        bound: usize,
    ) -> Self {
        Self {
            kind: TopologyErrorKind::HalfedgePermutation,
            details: TopologyDetails::HalfedgeEntry {
                relation,
                halfedge,
                value,
                bound,
            },
        }
    }

    #[must_use]
    pub const fn twin_law(halfedge: usize, twin: usize, twin_back: usize) -> Self {
        Self {
            kind: TopologyErrorKind::TwinLaw,
            details: TopologyDetails::Twin {
                halfedge,
                twin,
                twin_back,
            },
        }
    }

    #[must_use]
    pub const fn exterior_inconsistency(halfedge: usize, related: usize) -> Self {
        Self {
            kind: TopologyErrorKind::ExteriorInconsistency,
            details: TopologyDetails::Exterior { halfedge, related },
        }
    }

    #[must_use]
    pub const fn boundary_cycle(halfedge: usize, related: usize) -> Self {
        Self {
            kind: TopologyErrorKind::BoundaryCycle,
            details: TopologyDetails::Exterior { halfedge, related },
        }
    }

    #[must_use]
    pub const fn details(self) -> TopologyDetails {
        self.details
    }
}

impl TopologyError {
    /// Stable machine-readable reason class for adapter translation.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self.kind {
            TopologyErrorKind::CandidateShape => "candidate_shape",
            TopologyErrorKind::EmptyMaximalSimplices => "empty_maximal_simplices",
            TopologyErrorKind::NegativeIndex => "negative_index",
            TopologyErrorKind::IndexOverflow => "index_overflow",
            TopologyErrorKind::RepeatedVertex => "repeated_vertex",
            TopologyErrorKind::DuplicateMaximalSimplex => "duplicate_maximal_simplex",
            TopologyErrorKind::VertexExtent => "vertex_extent",
            TopologyErrorKind::CountOverflow => "count_overflow",
            TopologyErrorKind::Allocation => "allocation",
            TopologyErrorKind::MaskShape => "mask_shape",
            TopologyErrorKind::MaskIndexOutside => "mask_index_outside",
            TopologyErrorKind::SelectionNotStrict => "selection_not_strict",
            TopologyErrorKind::SelectionIndexOutside => "selection_index_outside",
            TopologyErrorKind::DegreeOutside => "degree_outside",
            TopologyErrorKind::TriangleDimension => "triangle_dimension",
            TopologyErrorKind::NotPure => "not_pure",
            TopologyErrorKind::CodimensionOneIncidence => "codimension_one_incidence",
            TopologyErrorKind::VertexLink => "vertex_link",
            TopologyErrorKind::Orientation => "orientation",
            TopologyErrorKind::Disconnected => "disconnected",
            TopologyErrorKind::BoundaryPresent => "boundary_present",
            TopologyErrorKind::BoundaryAbsent => "boundary_absent",
            TopologyErrorKind::DiskBoundaryComponents => "disk_boundary_components",
            TopologyErrorKind::DiskEulerCharacteristic => "disk_euler_characteristic",
            TopologyErrorKind::CapabilityNotAdmitted => "capability_not_admitted",
            TopologyErrorKind::OwnerMismatch => "owner_mismatch",
            TopologyErrorKind::HalfedgeShape => "halfedge_shape",
            TopologyErrorKind::HalfedgeRange => "halfedge_range",
            TopologyErrorKind::HalfedgePermutation => "halfedge_permutation",
            TopologyErrorKind::TwinLaw => "twin_law",
            TopologyErrorKind::ExteriorInconsistency => "exterior_inconsistency",
            TopologyErrorKind::BoundaryCycle => "boundary_cycle",
            TopologyErrorKind::ConversionNotSimplicial => "conversion_not_simplicial",
            TopologyErrorKind::CorrespondenceLaw => "correspondence_law",
            TopologyErrorKind::InternalInvariant => "internal_invariant",
        }
    }

    const fn message(self) -> &'static str {
        match self.kind {
            TopologyErrorKind::CandidateShape => {
                "candidate storage does not match its declared shape"
            }
            TopologyErrorKind::EmptyMaximalSimplices => {
                "maximal simplices must be a nonempty matrix"
            }
            TopologyErrorKind::NegativeIndex => "vertex indices must be nonnegative",
            TopologyErrorKind::IndexOverflow => "vertex index is outside the native index domain",
            TopologyErrorKind::RepeatedVertex => "a simplex cannot repeat a vertex",
            TopologyErrorKind::DuplicateMaximalSimplex => "duplicate maximal simplex identity",
            TopologyErrorKind::VertexExtent => "vertex_count does not contain every index",
            TopologyErrorKind::CountOverflow => "topology size exceeds the supported count domain",
            TopologyErrorKind::Allocation => "a fallible topology buffer reservation failed",
            TopologyErrorKind::MaskShape => {
                "packed mask does not align with canonical degree counts"
            }
            TopologyErrorKind::MaskIndexOutside => "mask index is outside its canonical degree",
            TopologyErrorKind::SelectionNotStrict => {
                "selection indices must be strictly increasing"
            }
            TopologyErrorKind::SelectionIndexOutside => {
                "selection index is outside its degree basis"
            }
            TopologyErrorKind::DegreeOutside => "degree is outside the complex",
            TopologyErrorKind::TriangleDimension => {
                "triangle-manifold refinement requires dimension two"
            }
            TopologyErrorKind::NotPure => "codimension-one regular input must be pure",
            TopologyErrorKind::CodimensionOneIncidence => {
                "every codimension-one simplex needs one or two top cofaces"
            }
            TopologyErrorKind::VertexLink => "a vertex link must be one path or one cycle",
            TopologyErrorKind::Orientation => "top-simplex orientations are not coherent",
            TopologyErrorKind::Disconnected => "the complex is disconnected",
            TopologyErrorKind::BoundaryPresent => "the complex has a nonempty topological boundary",
            TopologyErrorKind::BoundaryAbsent => "the complex has an empty topological boundary",
            TopologyErrorKind::DiskBoundaryComponents => {
                "a disk boundary must have exactly one connected component"
            }
            TopologyErrorKind::DiskEulerCharacteristic => {
                "a disk must have Euler characteristic one"
            }
            TopologyErrorKind::CapabilityNotAdmitted => {
                "the required capability has not been admitted"
            }
            TopologyErrorKind::OwnerMismatch => "values belong to different admitted complexes",
            TopologyErrorKind::HalfedgeShape => {
                "next and twin storage must match the declared halfedge count"
            }
            TopologyErrorKind::HalfedgeRange => {
                "a halfedge relation points outside its admitted domain"
            }
            TopologyErrorKind::HalfedgePermutation => "a halfedge relation is not a permutation",
            TopologyErrorKind::TwinLaw => "twin must be a fixed-point-free involution",
            TopologyErrorKind::ExteriorInconsistency => {
                "exterior face classification is inconsistent"
            }
            TopologyErrorKind::BoundaryCycle => {
                "exterior face orbits do not define disjoint boundary cycles"
            }
            TopologyErrorKind::ConversionNotSimplicial => {
                "halfedge presentation has no admitted simplicial reverse conversion"
            }
            TopologyErrorKind::CorrespondenceLaw => {
                "signed basis correspondence does not commute with the boundary"
            }
            TopologyErrorKind::InternalInvariant => "internal canonical topology invariant failed",
        }
    }
}

impl std::fmt::Display for TopologyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.message())
    }
}

impl std::error::Error for TopologyError {}
