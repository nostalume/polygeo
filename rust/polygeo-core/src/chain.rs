use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use num_bigint::BigInt;

use crate::coefficient::{
    BigIntEncoding, CoefficientSystem, CommutativeRing, ExactRational, Field, IntegerRing,
    RationalField, ReducedFractionEncoding, Ring, RingMorphism, ValueEncoding,
};
use crate::correspondence::SignedPermutation;
use crate::incidence::{DisjointSet, try_filled};
use crate::{
    BoundaryRef, ChainView, CoefficientSlice, ComplexCore, HalfedgeSurfaceCore, StorageLimit,
    TopologyError, WorkLimit,
};

mod sealed {
    pub trait Variance {}
}

/// Formal chain/cochain index used by spaces and map endpoints.
pub trait Variance: sealed::Variance + Clone + Copy + Debug + 'static {
    /// Stable diagnostic name.
    const NAME: &'static str;
}

/// Covariant chain-space index.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Chain;

impl sealed::Variance for Chain {}

impl Variance for Chain {
    const NAME: &'static str = "chain";
}

/// Contravariant algebraic-cochain-space index.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Cochain;

impl sealed::Variance for Cochain {}

impl Variance for Cochain {
    const NAME: &'static str = "cochain";
}

#[derive(Clone, Debug)]
pub(crate) enum ChainDomain {
    Simplicial(Arc<ComplexCore>),
    Halfedge(Arc<HalfedgeSurfaceCore>),
}

impl ChainDomain {
    pub(crate) fn view(&self) -> ChainView<'_> {
        match self {
            Self::Simplicial(owner) => owner.chain_view(),
            Self::Halfedge(owner) => owner.chain_view(),
        }
    }

    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Simplicial(left), Self::Simplicial(right)) => Arc::ptr_eq(left, right),
            (Self::Halfedge(left), Self::Halfedge(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }

    const fn same_schema(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (Self::Simplicial(_), Self::Simplicial(_)) | (Self::Halfedge(_), Self::Halfedge(_))
        )
    }

    fn compare_descriptors(
        &self,
        other: &Self,
        budget: &mut ComparisonBudget,
    ) -> Result<bool, PresentationError> {
        match (self, other) {
            (Self::Simplicial(left), Self::Simplicial(right)) => {
                if !budget.compare(&left.vertex_count(), &right.vertex_count())?
                    || !budget.compare(&left.dimension(), &right.dimension())?
                {
                    return Ok(false);
                }
                for degree in 0..=left.dimension() {
                    let left_basis = left.basis(degree).map_err(PresentationError::Topology)?;
                    let right_basis = right.basis(degree).map_err(PresentationError::Topology)?;
                    if !budget.compare(&left_basis.row_width(), &right_basis.row_width())?
                        || !budget.compare_slice(left_basis.values(), right_basis.values())?
                        || !budget.compare_slice(
                            left.orientation(degree)
                                .map_err(PresentationError::Topology)?,
                            right
                                .orientation(degree)
                                .map_err(PresentationError::Topology)?,
                        )?
                    {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            (Self::Halfedge(left), Self::Halfedge(right)) => Ok(budget
                .compare_slice(left.presentation_next(), right.presentation_next())?
                && budget.compare_slice(left.presentation_twin(), right.presentation_twin())?
                && budget.compare_slice(
                    left.presentation_face_kinds(),
                    right.presentation_face_kinds(),
                )?),
            _ => Err(PresentationError::NotComparable),
        }
    }

    pub(crate) const fn simplicial_owner(&self) -> Option<&Arc<ComplexCore>> {
        match self {
            Self::Simplicial(owner) => Some(owner),
            Self::Halfedge(_) => None,
        }
    }
}

/// One exact based chain-complex authority over an admitted coefficient system.
#[derive(Clone, Debug)]
pub struct ChainComplex<A: CoefficientSystem> {
    domain: ChainDomain,
    coefficients: A,
}

impl<A: CoefficientSystem> ChainComplex<A> {
    pub(crate) fn simplicial(owner: Arc<ComplexCore>, coefficients: A) -> Self {
        Self {
            domain: ChainDomain::Simplicial(owner),
            coefficients,
        }
    }

    pub(crate) fn halfedge(owner: Arc<HalfedgeSurfaceCore>, coefficients: A) -> Self {
        Self {
            domain: ChainDomain::Halfedge(owner),
            coefficients,
        }
    }

    pub(crate) fn chain_view(&self) -> ChainView<'_> {
        self.domain.view()
    }

    /// Maximum represented degree.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.domain.view().dimension()
    }

    /// Borrow the admitted coefficient system.
    #[must_use]
    pub const fn coefficient_system(&self) -> &A {
        &self.coefficients
    }

    /// Derive one chain degree space from this authority.
    ///
    /// # Errors
    ///
    /// Returns `degree_outside` when the degree is not represented.
    pub fn space(&self, degree: usize) -> Result<Space<A, Chain>, TopologyError> {
        Space::derive(self, degree)
    }

    /// Borrow the algebraic-dual complex without copying its basis or boundary.
    #[must_use]
    pub const fn dual(&self) -> DualComplex<'_, A> {
        DualComplex { primal: self }
    }

    /// Whether two handles retain the same admitted owner and coefficient system.
    #[must_use]
    pub fn same_owner(&self, other: &Self) -> bool {
        self.domain.same_owner(&other.domain) && self.coefficients.same_system(&other.coefficients)
    }

    /// Compare exact canonical presentations under an explicit scalar-work budget.
    ///
    /// A successful witness enables coordinate transport but does not merge
    /// nominal owners. Same-owner comparison is constant time and consumes no
    /// budget.
    ///
    /// # Errors
    ///
    /// Returns a classified mismatch, incompatible-schema, budget, or retained
    /// topology failure without publishing a witness.
    pub fn identify_presentation(
        &self,
        other: &Self,
        work_limit: WorkLimit,
    ) -> Result<PresentationEquality<A>, PresentationError> {
        if self.same_owner(other) {
            return Ok(PresentationEquality::new(self, other));
        }
        if !self.coefficients.same_system(&other.coefficients) {
            return Err(PresentationError::Mismatch);
        }
        if !self.domain.same_schema(&other.domain) {
            return Err(PresentationError::NotComparable);
        }

        let mut budget = ComparisonBudget::new(work_limit);
        if !self
            .domain
            .compare_descriptors(&other.domain, &mut budget)?
            || !budget.compare(&self.dimension(), &other.dimension())?
        {
            return Err(PresentationError::Mismatch);
        }
        for degree in 0..=self.dimension() {
            if !budget.compare(
                &self
                    .domain
                    .view()
                    .basis_size(degree)
                    .map_err(PresentationError::Topology)?,
                &other
                    .domain
                    .view()
                    .basis_size(degree)
                    .map_err(PresentationError::Topology)?,
            )? || !compare_boundaries(
                self.domain
                    .view()
                    .boundary(degree)
                    .map_err(PresentationError::Topology)?,
                other
                    .domain
                    .view()
                    .boundary(degree)
                    .map_err(PresentationError::Topology)?,
                &mut budget,
            )? {
                return Err(PresentationError::Mismatch);
            }
        }
        Ok(PresentationEquality::new(self, other))
    }
}

#[derive(Debug)]
struct ComparisonBudget {
    limit: u64,
    used: u64,
}

impl ComparisonBudget {
    const fn new(limit: WorkLimit) -> Self {
        Self {
            limit: limit.steps(),
            used: 0,
        }
    }

    fn charge(&mut self) -> Result<(), PresentationError> {
        let required = self
            .used
            .checked_add(1)
            .ok_or(PresentationError::Overflow)?;
        if required > self.limit {
            return Err(PresentationError::ComparisonSteps {
                required,
                limit: self.limit,
            });
        }
        self.used = required;
        Ok(())
    }

    fn compare<T: Eq + ?Sized>(&mut self, left: &T, right: &T) -> Result<bool, PresentationError> {
        self.charge()?;
        Ok(left == right)
    }

    fn compare_slice<T: Eq>(&mut self, left: &[T], right: &[T]) -> Result<bool, PresentationError> {
        if !self.compare(&left.len(), &right.len())? {
            return Ok(false);
        }
        for (left, right) in left.iter().zip(right) {
            if !self.compare(left, right)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn compare_boundaries(
    left: BoundaryRef<'_>,
    right: BoundaryRef<'_>,
    budget: &mut ComparisonBudget,
) -> Result<bool, PresentationError> {
    if !budget.compare(&left.shape(), &right.shape())? {
        return Ok(false);
    }
    let mut left = left.exact_entries();
    let mut right = right.exact_entries();
    loop {
        match (left.next(), right.next()) {
            (Some(left), Some(right)) => {
                if !budget.compare(&left, &right)? {
                    return Ok(false);
                }
            }
            (None, None) => return Ok(true),
            _ => {
                budget.charge()?;
                return Ok(false);
            }
        }
    }
}

/// Failed exact presentation comparison.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresentationError {
    /// Canonical descriptors or exact boundaries differ.
    Mismatch,
    /// The owners do not share one canonical presentation schema.
    NotComparable,
    /// The explicit scalar-comparison budget was exhausted.
    ComparisonSteps { required: u64, limit: u64 },
    /// Checked comparison-work arithmetic overflowed.
    Overflow,
    /// A retained topology invariant became unavailable.
    Topology(TopologyError),
}

impl PresentationError {
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::Mismatch => "presentation_mismatch",
            Self::NotComparable => "presentation_not_comparable",
            Self::ComparisonSteps { .. } => "resource_limit",
            Self::Overflow => "comparison_overflow",
            Self::Topology(error) => error.reason(),
        }
    }

    #[must_use]
    pub const fn resource_limit(self) -> Option<(&'static str, u64, u64)> {
        match self {
            Self::ComparisonSteps { required, limit } => {
                Some(("comparison_steps", required, limit))
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for PresentationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Mismatch => "canonical presentations differ",
            Self::NotComparable => "canonical presentation schemas are incompatible",
            Self::ComparisonSteps { .. } => "presentation comparison step limit exceeded",
            Self::Overflow => "presentation comparison work overflowed",
            Self::Topology(error) => return std::fmt::Display::fmt(error, formatter),
        })
    }
}

impl std::error::Error for PresentationError {}

/// Exact equality of two independently owned canonical presentations.
#[derive(Clone, Debug)]
pub struct PresentationEquality<A: CoefficientSystem> {
    left: ChainComplex<A>,
    right: ChainComplex<A>,
}

impl<A: CoefficientSystem> PresentationEquality<A> {
    fn new(left: &ChainComplex<A>, right: &ChainComplex<A>) -> Self {
        Self {
            left: left.clone(),
            right: right.clone(),
        }
    }

    /// Transport coordinates from the left owner to the right owner.
    ///
    /// # Errors
    ///
    /// Returns [`ChainError::SpaceMismatch`] unless the value belongs to the
    /// witnessed left owner.
    pub fn forward<K, E>(&self, value: &Element<A, K, E>) -> Result<Element<A, K, E>, ChainError>
    where
        K: Variance,
        E: ValueEncoding<A>,
    {
        transport_presented(value, &self.left, &self.right)
    }

    /// Transport coordinates from the right owner to the left owner.
    ///
    /// # Errors
    ///
    /// Returns [`ChainError::SpaceMismatch`] unless the value belongs to the
    /// witnessed right owner.
    pub fn inverse<K, E>(&self, value: &Element<A, K, E>) -> Result<Element<A, K, E>, ChainError>
    where
        K: Variance,
        E: ValueEncoding<A>,
    {
        transport_presented(value, &self.right, &self.left)
    }
}

fn transport_presented<A, K, E>(
    value: &Element<A, K, E>,
    source: &ChainComplex<A>,
    target: &ChainComplex<A>,
) -> Result<Element<A, K, E>, ChainError>
where
    A: CoefficientSystem,
    K: Variance,
    E: ValueEncoding<A>,
{
    if !value.space.complex.same_owner(source) {
        return Err(ChainError::SpaceMismatch);
    }
    let target = Space {
        complex: target.clone(),
        degree: value.space.degree,
        basis_size: value.space.basis_size,
        variance: PhantomData,
    };
    Ok(value.copy_to(&target))
}

impl<A: Ring> ChainComplex<A> {
    /// Change coefficients through an admitted canonical ring morphism.
    ///
    /// This shares the immutable topology owner and does not materialize maps.
    #[must_use]
    pub fn over<B: RingMorphism<A>>(&self, coefficients: B) -> ChainComplex<B> {
        ChainComplex {
            domain: self.domain.clone(),
            coefficients,
        }
    }

    /// Exact boundary differential from degree `k` to degree `k - 1`.
    ///
    /// Degree zero targets the derived zero module in degree `-1`.
    ///
    /// # Errors
    ///
    /// Returns `degree_outside` when the source degree is not represented.
    pub fn boundary(&self, degree: usize) -> Result<LinearMap<A, Chain, Chain>, TopologyError> {
        let source = self.space(degree)?;
        let target = if degree == 0 {
            Space::zero(self, -1)
        } else {
            Space::derive(self, degree - 1)?
        };
        Ok(LinearMap::new(
            source,
            target,
            AtomicRecipe::Boundary { degree },
        ))
    }
}

/// Borrowed algebraic-dual projection of one chain complex.
#[derive(Clone, Copy, Debug)]
pub struct DualComplex<'a, A: CoefficientSystem> {
    primal: &'a ChainComplex<A>,
}

impl<A: CoefficientSystem> DualComplex<'_, A> {
    /// Derive one cochain degree space in the induced dual basis.
    ///
    /// # Errors
    ///
    /// Returns `degree_outside` when the degree is not represented.
    pub fn space(&self, degree: usize) -> Result<Space<A, Cochain>, TopologyError> {
        Space::derive(self.primal, degree)
    }
}

impl<A: Ring> DualComplex<'_, A> {
    /// Exact coboundary differential from degree `k` to degree `k + 1`.
    ///
    /// The top represented degree targets its derived zero module.
    ///
    /// # Errors
    ///
    /// Returns `degree_outside` when the source degree is not represented, or
    /// `count_overflow` when its successor cannot be represented.
    pub fn coboundary(
        &self,
        degree: usize,
    ) -> Result<LinearMap<A, Cochain, Cochain>, TopologyError> {
        let source = Space::derive(self.primal, degree)?;
        let successor = degree.checked_add(1).ok_or(TopologyError::CountOverflow)?;
        if degree == self.primal.dimension() {
            let successor = isize::try_from(successor).map_err(|_| TopologyError::CountOverflow)?;
            return Ok(LinearMap::new(
                source,
                Space::zero(self.primal, successor),
                AtomicRecipe::Zero,
            ));
        }
        self.primal.boundary(successor).map(|map| map.dual())
    }
}

/// One owner-, degree-, variance-, and coefficient-indexed finite free space.
#[derive(Clone, Debug)]
pub struct Space<A: CoefficientSystem, K: Variance> {
    pub(crate) complex: ChainComplex<A>,
    degree: isize,
    basis_size: usize,
    variance: PhantomData<fn() -> K>,
}

impl<A: CoefficientSystem, K: Variance> Space<A, K> {
    pub(crate) fn derive(complex: &ChainComplex<A>, degree: usize) -> Result<Self, TopologyError> {
        let basis_size = complex.domain.view().basis_size(degree)?;
        let degree = isize::try_from(degree).map_err(|_| TopologyError::CountOverflow)?;
        Ok(Self {
            complex: complex.clone(),
            degree,
            basis_size,
            variance: PhantomData,
        })
    }

    fn zero(complex: &ChainComplex<A>, degree: isize) -> Self {
        Self {
            complex: complex.clone(),
            degree,
            basis_size: 0,
            variance: PhantomData,
        }
    }

    fn reindex<L: Variance>(&self) -> Space<A, L> {
        Space {
            complex: self.complex.clone(),
            degree: self.degree,
            basis_size: self.basis_size,
            variance: PhantomData,
        }
    }

    /// Degree represented by this space.
    #[must_use]
    pub const fn degree(&self) -> isize {
        self.degree
    }

    /// Rank of this finite free space.
    #[must_use]
    pub const fn basis_size(&self) -> usize {
        self.basis_size
    }

    /// Stable diagnostic variance name.
    #[must_use]
    pub const fn variance(&self) -> &'static str {
        K::NAME
    }

    pub(crate) fn same_based_module<L: Variance>(&self, other: &Space<A, L>) -> bool {
        self.degree == other.degree && self.complex.same_owner(&other.complex)
    }

    /// Whether another variance-indexed handle names the same owner and degree basis.
    #[must_use]
    pub fn same_based_space<L: Variance>(&self, other: &Space<A, L>) -> bool {
        self.same_based_module(other)
    }
}

impl<A: Ring, K: Variance> Space<A, K> {
    /// Change coefficients while retaining the same based topology owner.
    #[must_use]
    pub fn over<B: RingMorphism<A>>(&self, coefficients: B) -> Space<B, K> {
        Space {
            complex: self.complex.over(coefficients),
            degree: self.degree,
            basis_size: self.basis_size,
            variance: PhantomData,
        }
    }

    /// Canonical identity endomorphism of this exact space.
    #[must_use]
    pub fn identity(&self) -> LinearMap<A, K, K> {
        LinearMap::new(self.clone(), self.clone(), AtomicRecipe::Identity)
    }
}

impl<A: CoefficientSystem> Space<A, Chain> {
    /// Explicit coordinate identification with the induced dual basis.
    #[must_use]
    pub fn basis_identification(&self) -> BasisIdentification<A> {
        BasisIdentification {
            chain_space: self.clone(),
        }
    }
}

/// Same-owner coordinate identification between a chosen basis and its dual.
#[derive(Clone, Debug)]
pub struct BasisIdentification<A: CoefficientSystem> {
    chain_space: Space<A, Chain>,
}

impl<A: CoefficientSystem> BasisIdentification<A> {
    /// Reinterpret chain coordinates in the explicitly chosen induced dual basis.
    ///
    /// # Errors
    ///
    /// Returns [`ChainError::SpaceMismatch`] for a foreign owner or degree.
    pub fn forward<E>(
        &self,
        value: &Element<A, Chain, E>,
    ) -> Result<Element<A, Cochain, E>, ChainError>
    where
        E: ValueEncoding<A>,
    {
        if !self.chain_space.same_based_module(&value.space) {
            return Err(ChainError::SpaceMismatch);
        }
        Ok(value.copy_to(&self.chain_space.reindex()))
    }

    /// Recover chain coordinates from the explicitly identified dual basis.
    ///
    /// # Errors
    ///
    /// Returns [`ChainError::SpaceMismatch`] for a foreign owner or degree.
    pub fn inverse<E>(
        &self,
        value: &Element<A, Cochain, E>,
    ) -> Result<Element<A, Chain, E>, ChainError>
    where
        E: ValueEncoding<A>,
    {
        let cochain_space: Space<A, Cochain> = self.chain_space.reindex();
        if !cochain_space.same_based_module(&value.space) {
            return Err(ChainError::SpaceMismatch);
        }
        Ok(value.copy_to(&self.chain_space))
    }
}

impl<K: Variance> Space<IntegerRing, K> {
    /// Admit exact integer coordinates and publish one canonical sparse element.
    ///
    /// # Errors
    ///
    /// Returns [`ChainError`] when a basis index is outside this space.
    pub fn element<I>(
        &self,
        entries: I,
    ) -> Result<Element<IntegerRing, K, BigIntEncoding>, ChainError>
    where
        I: IntoIterator<Item = (usize, BigInt)>,
    {
        Element::admit(self, entries)
    }
}

impl<K: Variance> Space<RationalField, K> {
    /// Admit exact normalized rational coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`ChainError`] when a basis index is outside this space.
    pub fn element<I>(
        &self,
        entries: I,
    ) -> Result<Element<RationalField, K, ReducedFractionEncoding>, ChainError>
    where
        I: IntoIterator<Item = (usize, ExactRational)>,
    {
        Element::admit(self, entries)
    }
}

/// Failure to admit or combine exact chain-algebra values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChainError {
    /// A sparse coordinate names no basis element in its space.
    BasisIndexOutside { index: usize, bound: usize },
    /// Values do not inhabit the same owner-, degree-, and coefficient-bound module.
    SpaceMismatch,
    /// An operation requiring canonical simplices received another chain domain.
    NotSimplicial,
    /// An operation requiring division received coefficients that form only a ring.
    CoefficientFieldRequired,
    /// The wedge averaging factor is not invertible in the coefficient field.
    NormalizationNotInvertible,
    /// A retained topology recipe became unavailable after admission.
    Topology(TopologyError),
}

impl ChainError {
    /// Stable machine-readable reason.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::BasisIndexOutside { .. } => "basis_index_outside",
            Self::SpaceMismatch => "space_mismatch",
            Self::NotSimplicial => "not_simplicial",
            Self::CoefficientFieldRequired => "coefficient_field_required",
            Self::NormalizationNotInvertible => "normalization_not_invertible",
            Self::Topology(error) => error.reason(),
        }
    }

    /// Offending basis index, when applicable.
    #[must_use]
    pub const fn index(self) -> Option<usize> {
        match self {
            Self::BasisIndexOutside { index, .. } => Some(index),
            Self::SpaceMismatch
            | Self::NotSimplicial
            | Self::CoefficientFieldRequired
            | Self::NormalizationNotInvertible
            | Self::Topology(_) => None,
        }
    }

    /// Basis rank against which an index was checked, when applicable.
    #[must_use]
    pub const fn bound(self) -> Option<usize> {
        match self {
            Self::BasisIndexOutside { bound, .. } => Some(bound),
            Self::SpaceMismatch
            | Self::NotSimplicial
            | Self::CoefficientFieldRequired
            | Self::NormalizationNotInvertible
            | Self::Topology(_) => None,
        }
    }
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BasisIndexOutside { index, bound } => {
                write!(formatter, "basis index {index} is outside rank {bound}")
            }
            Self::SpaceMismatch => formatter.write_str("values belong to different exact spaces"),
            Self::NotSimplicial => {
                formatter.write_str("operation requires a canonical simplicial complex")
            }
            Self::CoefficientFieldRequired => {
                formatter.write_str("operation requires field coefficients")
            }
            Self::NormalizationNotInvertible => {
                formatter.write_str("wedge normalization is not invertible in this field")
            }
            Self::Topology(error) => std::fmt::Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for ChainError {}

impl From<TopologyError> for ChainError {
    fn from(error: TopologyError) -> Self {
        Self::Topology(error)
    }
}

/// Immutable canonical sparse coordinates in one exact space.
#[derive(Clone, Debug)]
pub struct Element<A, K, E>
where
    A: CoefficientSystem,
    K: Variance,
    E: ValueEncoding<A>,
{
    pub(crate) space: Space<A, K>,
    coordinates: Arc<Coordinates<E::Stored>>,
}

#[derive(Debug)]
struct Coordinates<V> {
    indices: Box<[usize]>,
    coefficients: Box<[V]>,
}

impl<A, K, E> Element<A, K, E>
where
    A: CoefficientSystem,
    K: Variance,
    E: ValueEncoding<A>,
{
    fn admit<I>(space: &Space<A, K>, entries: I) -> Result<Self, ChainError>
    where
        I: IntoIterator<Item = (usize, A::Element)>,
    {
        let bound = space.basis_size();
        let mut coordinates = Vec::new();
        for (index, coefficient) in entries {
            if index >= bound {
                return Err(ChainError::BasisIndexOutside { index, bound });
            }
            coordinates.push((index, coefficient));
        }
        coordinates.sort_unstable_by_key(|(index, _)| *index);

        let algebra = space.complex.coefficient_system();
        let mut indices = Vec::with_capacity(coordinates.len());
        let mut coefficients = Vec::with_capacity(coordinates.len());
        let mut coordinates = coordinates.into_iter();
        if let Some((mut current_index, mut current_coefficient)) = coordinates.next() {
            for (index, coefficient) in coordinates {
                if index == current_index {
                    algebra.add_assign(&mut current_coefficient, &coefficient);
                    continue;
                }
                if !algebra.is_zero(&current_coefficient) {
                    indices.push(current_index);
                    coefficients.push(E::encode(current_coefficient));
                }
                current_index = index;
                current_coefficient = coefficient;
            }
            if !algebra.is_zero(&current_coefficient) {
                indices.push(current_index);
                coefficients.push(E::encode(current_coefficient));
            }
        }
        Ok(Self {
            space: space.clone(),
            coordinates: Arc::new(Coordinates {
                indices: indices.into_boxed_slice(),
                coefficients: coefficients.into_boxed_slice(),
            }),
        })
    }

    pub(crate) fn from_canonical<L: Variance>(
        space: &Space<A, L>,
        indices: Vec<usize>,
        coefficients: Vec<E::Stored>,
    ) -> Element<A, L, E> {
        debug_assert_eq!(indices.len(), coefficients.len());
        debug_assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
        Element {
            space: space.clone(),
            coordinates: Arc::new(Coordinates {
                indices: indices.into_boxed_slice(),
                coefficients: coefficients.into_boxed_slice(),
            }),
        }
    }

    fn empty<L: Variance>(space: &Space<A, L>) -> Element<A, L, E> {
        Element {
            space: space.clone(),
            coordinates: Arc::new(Coordinates {
                indices: Box::default(),
                coefficients: Box::default(),
            }),
        }
    }

    fn copy_to<L: Variance>(&self, space: &Space<A, L>) -> Element<A, L, E> {
        Element {
            space: space.clone(),
            coordinates: Arc::clone(&self.coordinates),
        }
    }

    fn move_to<L: Variance>(self, space: &Space<A, L>) -> Element<A, L, E> {
        Element {
            space: space.clone(),
            coordinates: self.coordinates,
        }
    }

    /// Degree of the containing space.
    #[must_use]
    pub const fn degree(&self) -> isize {
        self.space.degree()
    }

    /// Rank of the containing space.
    #[must_use]
    pub fn basis_size(&self) -> usize {
        self.space.basis_size()
    }

    /// Strictly increasing nonzero-coordinate basis indices.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.coordinates.indices
    }

    /// Encoded nonzero coefficients aligned with [`Self::indices`].
    #[must_use]
    pub fn coefficients(&self) -> &[E::Stored] {
        &self.coordinates.coefficients
    }

    /// Borrow the exact containing space.
    #[must_use]
    pub const fn space(&self) -> &Space<A, K> {
        &self.space
    }
}

impl<K: Variance> Element<IntegerRing, K, BigIntEncoding> {
    /// Explicitly inject integral coordinates into the exact rational space.
    #[must_use]
    pub fn over(
        &self,
        coefficients: RationalField,
    ) -> Element<RationalField, K, ReducedFractionEncoding> {
        let space = self.space.over(coefficients);
        let values = self
            .coefficients()
            .iter()
            .map(|value| coefficients.inject(value))
            .collect();
        Element::<RationalField, K, ReducedFractionEncoding>::from_canonical(
            &space,
            self.indices().to_vec(),
            values,
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) enum AtomicRecipe {
    Boundary {
        degree: usize,
    },
    Coboundary {
        boundary_degree: usize,
    },
    SignedPermutation {
        permutation: Arc<SignedPermutation>,
        inverse: bool,
    },
    Identity,
    Zero,
}

impl AtomicRecipe {
    pub(crate) fn dual(&self) -> Self {
        match self {
            Self::Boundary { degree } => Self::Coboundary {
                boundary_degree: *degree,
            },
            Self::Coboundary { boundary_degree } => Self::Boundary {
                degree: *boundary_degree,
            },
            Self::SignedPermutation {
                permutation,
                inverse,
            } => Self::SignedPermutation {
                permutation: Arc::clone(permutation),
                inverse: !inverse,
            },
            Self::Identity => Self::Identity,
            Self::Zero => Self::Zero,
        }
    }

    fn same_identity(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Boundary { degree: left }, Self::Boundary { degree: right })
            | (
                Self::Coboundary {
                    boundary_degree: left,
                },
                Self::Coboundary {
                    boundary_degree: right,
                },
            ) => left == right,
            (
                Self::SignedPermutation {
                    permutation: left,
                    inverse: left_inverse,
                },
                Self::SignedPermutation {
                    permutation: right,
                    inverse: right_inverse,
                },
            ) => Arc::ptr_eq(left, right) && left_inverse == right_inverse,
            (Self::Identity, Self::Identity) | (Self::Zero, Self::Zero) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BasedDegree {
    pub(crate) domain: ChainDomain,
    pub(crate) degree: isize,
    pub(crate) basis_size: usize,
}

impl BasedDegree {
    pub(crate) fn new<A: CoefficientSystem, K: Variance>(space: &Space<A, K>) -> Self {
        Self {
            domain: space.complex.domain.clone(),
            degree: space.degree,
            basis_size: space.basis_size,
        }
    }

    pub(crate) fn successor(&self) -> Result<Self, TopologyError> {
        let degree = self
            .degree
            .checked_add(1)
            .ok_or(TopologyError::CountOverflow)?;
        let basis_size = usize::try_from(degree)
            .ok()
            .filter(|&degree| degree <= self.domain.view().dimension())
            .map_or(Ok(0), |degree| self.domain.view().basis_size(degree))?;
        Ok(Self {
            domain: self.domain.clone(),
            degree,
            basis_size,
        })
    }

    pub(crate) fn same_space<A: CoefficientSystem, K: Variance>(
        &self,
        space: &Space<A, K>,
    ) -> bool {
        self.degree == space.degree && self.domain.same_owner(&space.complex.domain)
    }

    fn reindex<A: CoefficientSystem, K: Variance>(&self, coefficients: &A) -> Space<A, K> {
        Space {
            complex: ChainComplex {
                domain: self.domain.clone(),
                coefficients: coefficients.clone(),
            },
            degree: self.degree,
            basis_size: self.basis_size,
            variance: PhantomData,
        }
    }

    pub(crate) fn same_based_module(&self, other: &Self) -> bool {
        self.degree == other.degree && self.domain.same_owner(&other.domain)
    }
}

#[derive(Clone, Debug)]
struct PlanStep {
    target: BasedDegree,
    recipe: AtomicRecipe,
}

#[derive(Clone, Debug)]
pub(crate) struct CompositionPlan {
    source: BasedDegree,
    steps: Vec<PlanStep>,
}

impl CompositionPlan {
    fn new<A: CoefficientSystem, K: Variance>(source: &Space<A, K>) -> Self {
        Self {
            source: BasedDegree::new(source),
            steps: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.steps.len()
    }

    pub(crate) fn step(
        &self,
        index: usize,
        dual: bool,
    ) -> (&BasedDegree, &BasedDegree, AtomicRecipe) {
        let source = if index == 0 {
            &self.source
        } else {
            &self.steps[index - 1].target
        };
        let step = &self.steps[index];
        if dual {
            (&step.target, source, step.recipe.dual())
        } else {
            (source, &step.target, step.recipe.clone())
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum MapRecipe {
    Atomic(AtomicRecipe),
    Composite {
        plan: Arc<CompositionPlan>,
        dual: bool,
    },
}

impl MapRecipe {
    fn dual(&self) -> Self {
        match self {
            Self::Atomic(recipe) => Self::Atomic(recipe.dual()),
            Self::Composite { plan, dual } => Self::Composite {
                plan: Arc::clone(plan),
                dual: !dual,
            },
        }
    }

    fn same_identity(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Atomic(left), Self::Atomic(right)) => left.same_identity(right),
            (
                Self::Composite {
                    plan: left,
                    dual: left_dual,
                },
                Self::Composite {
                    plan: right,
                    dual: right_dual,
                },
            ) => Arc::ptr_eq(left, right) && left_dual == right_dual,
            _ => false,
        }
    }

    fn execution_steps(&self) -> usize {
        match self {
            Self::Atomic(AtomicRecipe::Identity | AtomicRecipe::Zero) => 0,
            Self::Atomic(_) => 1,
            Self::Composite { plan, .. } => plan.steps.len(),
        }
    }
}

/// One exact degree-level linear map with proof-carrying endpoints.
#[derive(Clone, Debug)]
pub struct LinearMap<A, S, T>
where
    A: Ring,
    S: Variance,
    T: Variance,
{
    pub(crate) source: Space<A, S>,
    pub(crate) target: Space<A, T>,
    pub(crate) recipe: MapRecipe,
}

impl<A, S, T> LinearMap<A, S, T>
where
    A: Ring,
    S: Variance,
    T: Variance,
{
    fn new(source: Space<A, S>, target: Space<A, T>, recipe: AtomicRecipe) -> Self {
        Self {
            source,
            target,
            recipe: MapRecipe::Atomic(recipe),
        }
    }

    fn from_recipe(source: Space<A, S>, target: Space<A, T>, recipe: MapRecipe) -> Self {
        Self {
            source,
            target,
            recipe,
        }
    }

    /// Exact source space.
    #[must_use]
    pub const fn source(&self) -> &Space<A, S> {
        &self.source
    }

    /// Exact target space.
    #[must_use]
    pub const fn target(&self) -> &Space<A, T> {
        &self.target
    }

    /// Whether two handles denote the same canonical owner-derived map.
    #[must_use]
    pub fn same_identity(&self, other: &Self) -> bool {
        self.recipe.same_identity(&other.recipe)
            && self.source.same_based_module(&other.source)
            && self.target.same_based_module(&other.target)
    }

    /// Number of normalized atomic actions executed by [`Self::apply`].
    #[must_use]
    pub fn execution_steps(&self) -> usize {
        self.recipe.execution_steps()
    }

    /// Change coefficients through an admitted canonical ring morphism.
    ///
    /// The result shares the atomic recipe or flat composite plan and does not
    /// construct a matrix representation.
    #[must_use]
    pub fn over<B: RingMorphism<A>>(&self, coefficients: B) -> LinearMap<B, S, T> {
        LinearMap::from_recipe(
            self.source.over(coefficients.clone()),
            self.target.over(coefficients),
            self.recipe.clone(),
        )
    }

    /// Apply this exact map without materializing another representation.
    ///
    /// A boundary recipe scans the retained CSR nonzeros and binary-searches
    /// the sparse input. A coboundary recipe visits only retained CSR rows
    /// named by the sparse input, then canonicalizes their contributions.
    /// Identity copies the immutable sparse output buffers; zero allocates no
    /// coordinate entries.
    ///
    /// # Errors
    ///
    /// Returns [`ChainError::SpaceMismatch`] when the value is not in the
    /// exact source space.
    pub fn apply<E>(&self, value: &Element<A, S, E>) -> Result<Element<A, T, E>, ChainError>
    where
        E: ValueEncoding<A>,
    {
        if !self.source.same_based_module(&value.space) {
            return Err(ChainError::SpaceMismatch);
        }
        match &self.recipe {
            MapRecipe::Atomic(AtomicRecipe::Boundary { degree }) => {
                let boundary = self.source.complex.domain.view().boundary(*degree)?;
                Ok(apply_boundary(boundary, value, &self.target))
            }
            MapRecipe::Atomic(AtomicRecipe::Coboundary { boundary_degree }) => {
                let boundary = self
                    .source
                    .complex
                    .domain
                    .view()
                    .boundary(*boundary_degree)?;
                apply_coboundary(boundary, value, &self.target)
            }
            MapRecipe::Atomic(AtomicRecipe::SignedPermutation {
                permutation,
                inverse,
            }) => apply_signed_permutation(permutation, *inverse, value, &self.target),
            MapRecipe::Atomic(AtomicRecipe::Identity) => Ok(value.copy_to(&self.target)),
            MapRecipe::Atomic(AtomicRecipe::Zero) => Ok(Element::<A, S, E>::empty(&self.target)),
            MapRecipe::Composite { plan, dual } => {
                apply_composite(plan, *dual, value, &self.target)
            }
        }
    }
}

impl<A: Ring> LinearMap<A, Chain, Chain> {
    /// Contravariant algebraic dual in the induced dual bases.
    #[must_use]
    pub fn dual(&self) -> LinearMap<A, Cochain, Cochain> {
        LinearMap::from_recipe(
            self.target.reindex(),
            self.source.reindex(),
            self.recipe.dual(),
        )
    }
}

impl<A: Ring> LinearMap<A, Cochain, Cochain> {
    /// Contravariant algebraic dual in the induced primal bases.
    #[must_use]
    pub fn dual(&self) -> LinearMap<A, Chain, Chain> {
        LinearMap::from_recipe(
            self.target.reindex(),
            self.source.reindex(),
            self.recipe.dual(),
        )
    }
}

const MAX_COMPOSITION_STEPS: usize = 4_096;

/// Failure to admit one exact flat map composition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositionError {
    /// The target of the first map is not the exact source of the second.
    SpaceMismatch,
    /// The normalized execution plan exceeds the admitted finite bound.
    PlanLimit,
    /// A retained topology relation became unavailable.
    Topology(TopologyError),
}

impl CompositionError {
    /// Stable machine-readable reason.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::SpaceMismatch => "space_mismatch",
            Self::PlanLimit => "composition_plan_limit",
            Self::Topology(error) => error.reason(),
        }
    }
}

impl std::fmt::Display for CompositionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.reason())
    }
}

impl std::error::Error for CompositionError {}

/// Compose two exact maps after validating their nominal intermediate space.
///
/// The returned map owns one normalized flat execution plan. Composition does
/// not construct or cache a matrix representation.
///
/// # Errors
///
/// Returns [`CompositionError::SpaceMismatch`] for a foreign intermediate
/// module, or [`CompositionError::PlanLimit`] before publishing an oversized
/// plan.
pub fn compose<A, S, M, T>(
    after: &LinearMap<A, M, T>,
    before: &LinearMap<A, S, M>,
) -> Result<LinearMap<A, S, T>, CompositionError>
where
    A: Ring,
    S: Variance,
    M: Variance,
    T: Variance,
{
    if !before.target.same_based_module(&after.source) {
        return Err(CompositionError::SpaceMismatch);
    }
    if recipe_is_zero(&before.recipe) || recipe_is_zero(&after.recipe) {
        return Ok(LinearMap::new(
            before.source.clone(),
            after.target.clone(),
            AtomicRecipe::Zero,
        ));
    }

    let mut plan = CompositionPlan::new(&before.source);
    append_effective_steps(before, &mut plan)?;
    append_effective_steps(after, &mut plan)?;
    if plan.steps.len() > MAX_COMPOSITION_STEPS {
        return Err(CompositionError::PlanLimit);
    }
    let recipe = match plan.steps.len() {
        0 => MapRecipe::Atomic(AtomicRecipe::Identity),
        1 => MapRecipe::Atomic(plan.steps[0].recipe.clone()),
        _ => MapRecipe::Composite {
            plan: Arc::new(plan),
            dual: false,
        },
    };
    Ok(LinearMap::from_recipe(
        before.source.clone(),
        after.target.clone(),
        recipe,
    ))
}

fn recipe_is_zero(recipe: &MapRecipe) -> bool {
    matches!(recipe, MapRecipe::Atomic(AtomicRecipe::Zero))
}

fn append_effective_steps<A, S, T>(
    map: &LinearMap<A, S, T>,
    output: &mut CompositionPlan,
) -> Result<(), CompositionError>
where
    A: Ring,
    S: Variance,
    T: Variance,
{
    match &map.recipe {
        MapRecipe::Atomic(AtomicRecipe::Identity) => Ok(()),
        MapRecipe::Atomic(recipe) => push_normalized_step(
            output,
            &BasedDegree::new(&map.source),
            &BasedDegree::new(&map.target),
            recipe.clone(),
        ),
        MapRecipe::Composite { plan, dual: false } => {
            for index in 0..plan.steps.len() {
                let (source, target, recipe) = plan.step(index, false);
                push_normalized_step(output, source, target, recipe)?;
            }
            Ok(())
        }
        MapRecipe::Composite { plan, dual: true } => {
            for index in (0..plan.steps.len()).rev() {
                let (source, target, recipe) = plan.step(index, true);
                push_normalized_step(output, source, target, recipe)?;
            }
            Ok(())
        }
    }
}

fn push_normalized_step(
    output: &mut CompositionPlan,
    source: &BasedDegree,
    target: &BasedDegree,
    recipe: AtomicRecipe,
) -> Result<(), CompositionError> {
    if output.steps.len() >= MAX_COMPOSITION_STEPS {
        return Err(CompositionError::PlanLimit);
    }
    let Some(previous) = output.steps.last() else {
        if !output.source.same_based_module(source) {
            return Err(CompositionError::SpaceMismatch);
        }
        output.steps.push(PlanStep {
            target: target.clone(),
            recipe,
        });
        return Ok(());
    };
    if !previous.target.same_based_module(source) {
        return Err(CompositionError::SpaceMismatch);
    }
    let Some(permutation) = compose_signed_recipes(&recipe, &previous.recipe)? else {
        output.steps.push(PlanStep {
            target: target.clone(),
            recipe,
        });
        return Ok(());
    };
    let previous_source = if output.steps.len() == 1 {
        &output.source
    } else {
        &output.steps[output.steps.len() - 2].target
    };
    if permutation_is_identity(&permutation) && previous_source.same_based_module(target) {
        output.steps.pop();
        return Ok(());
    }
    let previous = output
        .steps
        .last_mut()
        .expect("the previous step was observed");
    *previous = PlanStep {
        target: target.clone(),
        recipe: AtomicRecipe::SignedPermutation {
            permutation: Arc::new(permutation),
            inverse: false,
        },
    };
    Ok(())
}

fn compose_signed_recipes(
    after: &AtomicRecipe,
    before: &AtomicRecipe,
) -> Result<Option<SignedPermutation>, CompositionError> {
    let (
        AtomicRecipe::SignedPermutation {
            permutation: after,
            inverse: after_inverse,
        },
        AtomicRecipe::SignedPermutation {
            permutation: before,
            inverse: before_inverse,
        },
    ) = (after, before)
    else {
        return Ok(None);
    };
    if after.len() != before.len() {
        return Err(CompositionError::SpaceMismatch);
    }
    let mut targets = Vec::with_capacity(before.len());
    let mut signs = Vec::with_capacity(before.len());
    for source in 0..before.len() {
        let (middle, before_sign) = signed_basis(before, *before_inverse, source)?;
        let (target, after_sign) = signed_basis(after, *after_inverse, middle)?;
        targets.push(target);
        signs.push(before_sign * after_sign);
    }
    SignedPermutation::admit(targets, signs)
        .map(Some)
        .map_err(CompositionError::Topology)
}

fn signed_basis(
    permutation: &SignedPermutation,
    inverse: bool,
    index: usize,
) -> Result<(usize, i8), CompositionError> {
    if inverse {
        permutation.inverse_basis(index)
    } else {
        permutation.map_basis(index)
    }
    .map_err(CompositionError::Topology)
}

fn permutation_is_identity(permutation: &SignedPermutation) -> bool {
    permutation
        .target_of_source()
        .iter()
        .copied()
        .zip(permutation.signs().iter().copied())
        .enumerate()
        .all(|(source, (target, sign))| source == target && sign == 1)
}

fn apply_composite<A, S, T, E>(
    plan: &CompositionPlan,
    dual: bool,
    value: &Element<A, S, E>,
    target: &Space<A, T>,
) -> Result<Element<A, T, E>, ChainError>
where
    A: Ring,
    S: Variance,
    T: Variance,
    E: ValueEncoding<A>,
{
    let current = if dual {
        apply_composite_order(plan, true, (0..plan.steps.len()).rev(), value)?
    } else {
        apply_composite_order(plan, false, 0..plan.steps.len(), value)?
    };
    match current {
        Some(current) if current.space.same_based_module(target) => Ok(current.move_to(target)),
        Some(_) => Err(ChainError::SpaceMismatch),
        None => Ok(value.copy_to(target)),
    }
}

fn apply_composite_order<A, S, E, I>(
    plan: &CompositionPlan,
    dual: bool,
    indices: I,
    value: &Element<A, S, E>,
) -> Result<Option<Element<A, S, E>>, ChainError>
where
    A: Ring,
    S: Variance,
    E: ValueEncoding<A>,
    I: Iterator<Item = usize>,
{
    let mut current = None;
    for index in indices {
        let (source, target, recipe) = plan.step(index, dual);
        let map = LinearMap::new(
            source.reindex::<A, S>(value.space.complex.coefficient_system()),
            target.reindex::<A, S>(value.space.complex.coefficient_system()),
            recipe,
        );
        current = Some(match &current {
            Some(current) => map.apply(current)?,
            None => map.apply(value)?,
        });
    }
    Ok(current)
}

fn apply_boundary<A, S, T, E>(
    boundary: BoundaryRef<'_>,
    value: &Element<A, S, E>,
    target: &Space<A, T>,
) -> Element<A, T, E>
where
    A: Ring,
    S: Variance,
    T: Variance,
    E: ValueEncoding<A>,
{
    match boundary.coefficients() {
        CoefficientSlice::I8(coefficients) => {
            apply_boundary_rows(boundary, value, target, |position| {
                i64::from(coefficients[position])
            })
        }
        CoefficientSlice::I64(coefficients) => {
            apply_boundary_rows(boundary, value, target, |position| coefficients[position])
        }
    }
}

fn apply_boundary_rows<A, S, T, E, F>(
    boundary: BoundaryRef<'_>,
    value: &Element<A, S, E>,
    target: &Space<A, T>,
    coefficient_at: F,
) -> Element<A, T, E>
where
    A: Ring,
    S: Variance,
    T: Variance,
    E: ValueEncoding<A>,
    F: Fn(usize) -> i64,
{
    let algebra = target.complex.coefficient_system();
    let row_offsets = boundary.indptr();
    let columns = boundary.indices();
    let mut output_indices = Vec::new();
    let mut output_coefficients = Vec::new();
    for row in 0..boundary.shape().0 {
        let mut sum = algebra.zero();
        let start = row_offsets[row];
        let end = row_offsets[row + 1];
        for (offset, &column) in columns[start..end].iter().enumerate() {
            let position = start + offset;
            if let Ok(input_position) = value.indices().binary_search(&column) {
                let incidence = algebra.lift_i64(coefficient_at(position));
                let term = algebra.multiply(
                    &incidence,
                    E::element(&value.coefficients()[input_position]),
                );
                algebra.add_assign(&mut sum, &term);
            }
        }
        if !algebra.is_zero(&sum) {
            output_indices.push(row);
            output_coefficients.push(E::encode(sum));
        }
    }
    Element::<A, S, E>::from_canonical(target, output_indices, output_coefficients)
}

fn apply_coboundary<A, S, T, E>(
    boundary: BoundaryRef<'_>,
    value: &Element<A, S, E>,
    target: &Space<A, T>,
) -> Result<Element<A, T, E>, ChainError>
where
    A: Ring,
    S: Variance,
    T: Variance,
    E: ValueEncoding<A>,
{
    match boundary.coefficients() {
        CoefficientSlice::I8(coefficients) => {
            apply_coboundary_rows(boundary, value, target, |position| {
                i64::from(coefficients[position])
            })
        }
        CoefficientSlice::I64(coefficients) => {
            apply_coboundary_rows(boundary, value, target, |position| coefficients[position])
        }
    }
}

fn apply_coboundary_rows<A, S, T, E, F>(
    boundary: BoundaryRef<'_>,
    value: &Element<A, S, E>,
    target: &Space<A, T>,
    coefficient_at: F,
) -> Result<Element<A, T, E>, ChainError>
where
    A: Ring,
    S: Variance,
    T: Variance,
    E: ValueEncoding<A>,
    F: Fn(usize) -> i64,
{
    let algebra = target.complex.coefficient_system();
    let row_offsets = boundary.indptr();
    let columns = boundary.indices();
    let output_capacity = value
        .indices()
        .iter()
        .map(|&row| row_offsets[row + 1] - row_offsets[row])
        .sum();
    let mut output = Vec::with_capacity(output_capacity);
    for (input_position, row) in value.indices().iter().copied().enumerate() {
        let start = row_offsets[row];
        let end = row_offsets[row + 1];
        for (offset, &column) in columns[start..end].iter().enumerate() {
            let position = start + offset;
            let incidence = algebra.lift_i64(coefficient_at(position));
            output.push((
                column,
                algebra.multiply(
                    &incidence,
                    E::element(&value.coefficients()[input_position]),
                ),
            ));
        }
    }
    Element::admit(target, output)
}

fn apply_signed_permutation<A, S, T, E>(
    permutation: &SignedPermutation,
    inverse: bool,
    value: &Element<A, S, E>,
    target: &Space<A, T>,
) -> Result<Element<A, T, E>, ChainError>
where
    A: Ring,
    S: Variance,
    T: Variance,
    E: ValueEncoding<A>,
{
    let algebra = target.complex.coefficient_system();
    let mut output = Vec::with_capacity(value.indices().len());
    for (position, index) in value.indices().iter().copied().enumerate() {
        let (mapped, sign) = if inverse {
            permutation.inverse_basis(index)?
        } else {
            permutation.map_basis(index)?
        };
        let coefficient = E::element(&value.coefficients()[position]);
        output.push((
            mapped,
            if sign == 1 {
                coefficient.clone()
            } else {
                algebra.negate(coefficient)
            },
        ));
    }
    Element::admit(target, output)
}

/// Degree-wise exact chain map between two based chain complexes.
#[derive(Clone, Debug)]
pub struct ChainMap<A: Ring> {
    source: ChainComplex<A>,
    target: ChainComplex<A>,
    degrees: Box<[AtomicRecipe]>,
}

impl<A: Ring> ChainMap<A> {
    /// Source chain complex.
    #[must_use]
    pub const fn source(&self) -> &ChainComplex<A> {
        &self.source
    }

    /// Target chain complex.
    #[must_use]
    pub const fn target(&self) -> &ChainComplex<A> {
        &self.target
    }

    /// Exact degree component.
    ///
    /// # Errors
    ///
    /// Returns `degree_outside` when the degree is not represented.
    pub fn degree(&self, degree: usize) -> Result<LinearMap<A, Chain, Chain>, TopologyError> {
        let recipe = self
            .degrees
            .get(degree)
            .ok_or(TopologyError::degree_outside(degree))?
            .clone();
        Ok(LinearMap::new(
            self.source.space(degree)?,
            self.target.space(degree)?,
            recipe,
        ))
    }
}

/// Lifecycle and incidence-work ceiling for one chain-law admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChainLawLimit {
    storage: StorageLimit,
    terms: WorkLimit,
}

impl ChainLawLimit {
    pub const DEFAULT: Self = Self {
        storage: StorageLimit::new(128 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("default storage limit is valid"),
        terms: WorkLimit::new(100_000_000),
    };

    #[must_use]
    pub const fn new(storage: StorageLimit, terms: WorkLimit) -> Self {
        Self { storage, terms }
    }

    #[must_use]
    pub const fn storage(self) -> StorageLimit {
        self.storage
    }

    #[must_use]
    pub const fn terms(self) -> WorkLimit {
        self.terms
    }
}

impl Default for ChainLawLimit {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Failed exact admission of a proposed chain isomorphism.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IsomorphismError {
    /// A degree map, inverse, endpoint, or commuting law is false.
    InvalidCandidate,
    /// The explicit verification work budget was exhausted.
    RetainedLogicalBytes {
        required: u64,
        limit: u64,
    },
    PeakLiveLogicalBytes {
        required: u64,
        limit: u64,
    },
    Terms {
        required: u64,
        limit: u64,
    },
    Overflow,
    Allocation,
    /// A retained topology invariant became unavailable.
    Topology(TopologyError),
}

impl std::fmt::Display for IsomorphismError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidCandidate => "chain-isomorphism candidate is invalid",
            Self::RetainedLogicalBytes { .. } => "retained logical byte limit exceeded",
            Self::PeakLiveLogicalBytes { .. } => "peak live logical byte limit exceeded",
            Self::Terms { .. } => "chain-law term limit exceeded",
            Self::Overflow => "chain-law resource estimate overflowed",
            Self::Allocation => "chain-law temporary allocation failed",
            Self::Topology(error) => return std::fmt::Display::fmt(error, formatter),
        })
    }
}

impl std::error::Error for IsomorphismError {}

impl IsomorphismError {
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::InvalidCandidate => "correspondence_law",
            Self::RetainedLogicalBytes { .. }
            | Self::PeakLiveLogicalBytes { .. }
            | Self::Terms { .. } => "resource_limit",
            Self::Overflow => "count_overflow",
            Self::Allocation => "allocation",
            Self::Topology(error) => error.reason(),
        }
    }

    #[must_use]
    pub const fn resource_limit(self) -> Option<(&'static str, u64, u64)> {
        match self {
            Self::RetainedLogicalBytes { required, limit } => {
                Some(("retained_logical_bytes", required, limit))
            }
            Self::PeakLiveLogicalBytes { required, limit } => {
                Some(("peak_live_logical_bytes", required, limit))
            }
            Self::Terms { required, limit } => Some(("terms", required, limit)),
            _ => None,
        }
    }
}

/// Checked degree-wise chain isomorphism with exact inverse.
#[derive(Clone, Debug)]
pub struct ChainIsomorphism<A: Ring> {
    forward: ChainMap<A>,
}

impl<A: Ring> ChainIsomorphism<A> {
    pub(crate) fn admit_signed(
        source: ChainComplex<A>,
        target: ChainComplex<A>,
        permutations: Vec<SignedPermutation>,
        limit: ChainLawLimit,
    ) -> Result<Self, IsomorphismError> {
        let permutations = permutations.into_iter().map(Arc::new).collect::<Vec<_>>();
        verify_signed_chain_law(&source, &target, &permutations, limit)?;

        let degrees = permutations
            .into_iter()
            .map(|permutation| AtomicRecipe::SignedPermutation {
                permutation,
                inverse: false,
            })
            .collect::<Vec<_>>();
        Ok(Self {
            forward: ChainMap {
                source,
                target,
                degrees: degrees.into_boxed_slice(),
            },
        })
    }

    /// Source chain complex.
    #[must_use]
    pub const fn source(&self) -> &ChainComplex<A> {
        self.forward.source()
    }

    /// Target chain complex.
    #[must_use]
    pub const fn target(&self) -> &ChainComplex<A> {
        self.forward.target()
    }

    /// Forward degree map.
    ///
    /// # Errors
    ///
    /// Returns `degree_outside` when the degree is not represented.
    pub fn forward(&self, degree: usize) -> Result<LinearMap<A, Chain, Chain>, TopologyError> {
        self.forward.degree(degree)
    }

    /// Inverse degree map.
    ///
    /// # Errors
    ///
    /// Returns `degree_outside` when the degree is not represented.
    pub fn inverse(&self, degree: usize) -> Result<LinearMap<A, Chain, Chain>, TopologyError> {
        let recipe = match self
            .forward
            .degrees
            .get(degree)
            .ok_or(TopologyError::degree_outside(degree))?
        {
            AtomicRecipe::SignedPermutation { permutation, .. } => {
                AtomicRecipe::SignedPermutation {
                    permutation: Arc::clone(permutation),
                    inverse: true,
                }
            }
            _ => return Err(TopologyError::CorrespondenceLaw),
        };
        Ok(LinearMap::new(
            self.target().space(degree)?,
            self.source().space(degree)?,
            recipe,
        ))
    }

    /// Borrow the contravariant dual isomorphism without copying its proof.
    #[must_use]
    pub const fn dual(&self) -> DualChainIsomorphism<'_, A> {
        DualChainIsomorphism { primal: self }
    }

    /// Recheck exact inverse endpoints and commuting laws under a work budget.
    ///
    /// # Errors
    ///
    /// Returns a classified candidate, budget, or retained-topology failure.
    pub fn verify(&self, limit: ChainLawLimit) -> Result<(), IsomorphismError> {
        let permutations = self
            .forward
            .degrees
            .iter()
            .map(|recipe| match recipe {
                AtomicRecipe::SignedPermutation {
                    permutation,
                    inverse: false,
                } => Ok(Arc::clone(permutation)),
                _ => Err(IsomorphismError::InvalidCandidate),
            })
            .collect::<Result<Vec<_>, _>>()?;
        verify_signed_chain_law(self.source(), self.target(), &permutations, limit)
    }

    fn permutation(&self, degree: usize) -> Result<&SignedPermutation, TopologyError> {
        match self
            .forward
            .degrees
            .get(degree)
            .ok_or(TopologyError::degree_outside(degree))?
        {
            AtomicRecipe::SignedPermutation { permutation, .. } => Ok(permutation),
            _ => Err(TopologyError::CorrespondenceLaw),
        }
    }

    /// Borrow the target indices and signs of an admitted degree permutation.
    ///
    /// # Errors
    ///
    /// Returns `degree_outside` when the degree is not represented.
    pub fn signed_permutation(&self, degree: usize) -> Result<(&[usize], &[i8]), TopologyError> {
        self.permutation(degree)
            .map(|permutation| (permutation.target_of_source(), permutation.signs()))
    }
}

/// Borrowed contravariant dual of a checked chain isomorphism.
#[derive(Clone, Copy, Debug)]
pub struct DualChainIsomorphism<'a, A: Ring> {
    primal: &'a ChainIsomorphism<A>,
}

impl<A: Ring> DualChainIsomorphism<'_, A> {
    /// Contravariant forward map from the primal target dual to source dual.
    ///
    /// # Errors
    ///
    /// Returns `degree_outside` when the degree is not represented.
    pub fn forward(&self, degree: usize) -> Result<LinearMap<A, Cochain, Cochain>, TopologyError> {
        self.primal.forward(degree).map(|map| map.dual())
    }

    /// Contravariant inverse map from the primal source dual to target dual.
    ///
    /// # Errors
    ///
    /// Returns `degree_outside` when the degree is not represented.
    pub fn inverse(&self, degree: usize) -> Result<LinearMap<A, Cochain, Cochain>, TopologyError> {
        self.primal.inverse(degree).map(|map| map.dual())
    }
}

fn verify_signed_chain_law<A: Ring>(
    source: &ChainComplex<A>,
    target: &ChainComplex<A>,
    permutations: &[Arc<SignedPermutation>],
    limit: ChainLawLimit,
) -> Result<(), IsomorphismError> {
    if source.dimension() != target.dimension()
        || permutations.len() != source.dimension().saturating_add(1)
    {
        return Err(IsomorphismError::InvalidCandidate);
    }
    let estimate = estimate_chain_law(source, target, permutations)?;
    estimate.admit(limit)?;
    let mut remaining = estimate.terms;
    for (degree, permutation) in permutations.iter().enumerate() {
        remaining -= 1;
        let source_size = source
            .domain
            .view()
            .basis_size(degree)
            .map_err(IsomorphismError::Topology)?;
        let target_size = target
            .domain
            .view()
            .basis_size(degree)
            .map_err(IsomorphismError::Topology)?;
        if source_size != target_size || permutation.len() != source_size {
            return Err(IsomorphismError::InvalidCandidate);
        }
    }
    for degree in 1..=source.dimension() {
        verify_signed_degree_law(
            source.domain.view(),
            target.domain.view(),
            degree,
            &permutations[degree - 1],
            &permutations[degree],
            &mut remaining,
        )?;
    }
    Ok(())
}

fn verify_signed_degree_law(
    source: ChainView<'_>,
    target: ChainView<'_>,
    degree: usize,
    lower: &SignedPermutation,
    upper: &SignedPermutation,
    remaining: &mut u64,
) -> Result<(), IsomorphismError> {
    let target_boundary = target
        .boundary(degree)
        .map_err(IsomorphismError::Topology)?;
    let source_boundary = source
        .boundary(degree)
        .map_err(IsomorphismError::Topology)?;
    let mut left = Vec::new();
    left.try_reserve_exact(target_boundary.indices().len())
        .map_err(|_| IsomorphismError::Allocation)?;
    for (target_row, target_column, coefficient) in target_boundary.exact_entries() {
        *remaining -= 3;
        let (source_column, upper_sign) = upper
            .inverse_basis(target_column)
            .map_err(|_| IsomorphismError::InvalidCandidate)?;
        left.push((
            (target_row, source_column),
            coefficient * i64::from(upper_sign),
        ));
    }
    let mut right = Vec::new();
    right
        .try_reserve_exact(source_boundary.indices().len())
        .map_err(|_| IsomorphismError::Allocation)?;
    for (source_row, source_column, coefficient) in source_boundary.exact_entries() {
        *remaining -= 3;
        let (target_row, lower_sign) = lower
            .map_basis(source_row)
            .map_err(|_| IsomorphismError::InvalidCandidate)?;
        right.push((
            (target_row, source_column),
            coefficient * i64::from(lower_sign),
        ));
    }
    left.sort_unstable_by_key(|entry| entry.0);
    right.sort_unstable_by_key(|entry| entry.0);
    if left.len() != right.len() {
        return Err(IsomorphismError::InvalidCandidate);
    }
    for (left, right) in left.iter().zip(&right) {
        *remaining -= 1;
        if left != right {
            return Err(IsomorphismError::InvalidCandidate);
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ChainLawEstimate {
    retained: u64,
    peak: u64,
    terms: u64,
}

impl ChainLawEstimate {
    fn admit(self, limit: ChainLawLimit) -> Result<(), IsomorphismError> {
        let storage = limit.storage();
        if self.retained > storage.retained_logical_bytes() {
            Err(IsomorphismError::RetainedLogicalBytes {
                required: self.retained,
                limit: storage.retained_logical_bytes(),
            })
        } else if self.peak > storage.peak_live_logical_bytes() {
            Err(IsomorphismError::PeakLiveLogicalBytes {
                required: self.peak,
                limit: storage.peak_live_logical_bytes(),
            })
        } else if self.terms > limit.terms().steps() {
            Err(IsomorphismError::Terms {
                required: self.terms,
                limit: limit.terms().steps(),
            })
        } else {
            Ok(())
        }
    }
}

fn estimate_chain_law<A: Ring>(
    source: &ChainComplex<A>,
    target: &ChainComplex<A>,
    permutations: &[Arc<SignedPermutation>],
) -> Result<ChainLawEstimate, IsomorphismError> {
    let retained = permutations.iter().try_fold(0_u64, |sum, permutation| {
        let width = u64::try_from(2 * size_of::<usize>() + size_of::<i8>())
            .map_err(|_| IsomorphismError::Overflow)?;
        checked_add(sum, checked_mul(permutation.len(), width)?)
    })?;
    let source_bytes = chain_domain_bytes(source.domain.view())?;
    let owner_bytes = if source.domain.same_owner(&target.domain) {
        source_bytes
    } else {
        checked_add(source_bytes, chain_domain_bytes(target.domain.view())?)?
    };
    let mut terms = u64::try_from(permutations.len()).map_err(|_| IsomorphismError::Overflow)?;
    let mut temporary = 0;
    let entry_bytes = u64::try_from(size_of::<((usize, usize), i64)>())
        .map_err(|_| IsomorphismError::Overflow)?;
    for degree in 1..=source.dimension() {
        let source_nnz = source
            .domain
            .view()
            .boundary(degree)
            .map_err(IsomorphismError::Topology)?
            .indices()
            .len();
        let target_nnz = target
            .domain
            .view()
            .boundary(degree)
            .map_err(IsomorphismError::Topology)?
            .indices()
            .len();
        let nnz = source_nnz
            .checked_add(target_nnz)
            .ok_or(IsomorphismError::Overflow)?;
        temporary = temporary.max(checked_mul(nnz, entry_bytes)?);
        terms = checked_add(terms, checked_mul(nnz, 3)?)?;
        terms = checked_add(
            terms,
            u64::try_from(source_nnz.min(target_nnz)).map_err(|_| IsomorphismError::Overflow)?,
        )?;
    }
    Ok(ChainLawEstimate {
        retained,
        peak: checked_add(checked_add(owner_bytes, retained)?, temporary)?,
        terms,
    })
}

fn chain_domain_bytes(view: ChainView<'_>) -> Result<u64, IsomorphismError> {
    (0..=view.dimension()).try_fold(0_u64, |sum, degree| {
        let boundary = view.boundary(degree).map_err(IsomorphismError::Topology)?;
        let index_count = boundary
            .indptr()
            .len()
            .checked_add(boundary.indices().len())
            .ok_or(IsomorphismError::Overflow)?;
        let indices = checked_mul(
            index_count,
            u64::try_from(size_of::<usize>()).map_err(|_| IsomorphismError::Overflow)?,
        )?;
        let coefficients = checked_mul(
            boundary.indices().len(),
            u64::try_from(size_of::<i64>()).map_err(|_| IsomorphismError::Overflow)?,
        )?;
        checked_add(sum, checked_add(indices, coefficients)?)
    })
}

fn checked_mul(value: usize, factor: u64) -> Result<u64, IsomorphismError> {
    u64::try_from(value)
        .map_err(|_| IsomorphismError::Overflow)?
        .checked_mul(factor)
        .ok_or(IsomorphismError::Overflow)
}

fn checked_add(left: u64, right: u64) -> Result<u64, IsomorphismError> {
    left.checked_add(right).ok_or(IsomorphismError::Overflow)
}

impl<A, E> Element<A, Cochain, E>
where
    A: Ring,
    E: ValueEncoding<A>,
{
    /// Evaluate this cochain on a chain in the exact induced dual basis.
    ///
    /// # Errors
    ///
    /// Returns [`ChainError::SpaceMismatch`] for a foreign owner, degree, or
    /// coefficient system.
    pub fn evaluate(&self, chain: &Element<A, Chain, E>) -> Result<A::Element, ChainError> {
        if !self.space.same_based_module(&chain.space) {
            return Err(ChainError::SpaceMismatch);
        }

        let algebra = self.space.complex.coefficient_system();
        let mut result = algebra.zero();
        let (mut left, mut right) = (0, 0);
        while left < self.indices().len() && right < chain.indices().len() {
            match self.indices()[left].cmp(&chain.indices()[right]) {
                std::cmp::Ordering::Less => left += 1,
                std::cmp::Ordering::Greater => right += 1,
                std::cmp::Ordering::Equal => {
                    let product = algebra.multiply(
                        E::element(&self.coefficients()[left]),
                        E::element(&chain.coefficients()[right]),
                    );
                    algebra.add_assign(&mut result, &product);
                    left += 1;
                    right += 1;
                }
            }
        }
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct OrientedFacePair {
    pub(crate) left: usize,
    pub(crate) right: usize,
    pub(crate) sign: i8,
}

fn cup_face_pair(
    owner: &ComplexCore,
    left_degree: usize,
    target_degree: usize,
    target_index: usize,
) -> Result<OrientedFacePair, TopologyError> {
    let target_basis = owner.basis(target_degree)?;
    let simplex = target_basis
        .row(target_index)
        .ok_or(TopologyError::InternalInvariant)?;
    let left = owner
        .basis(left_degree)?
        .binary_search(&simplex[..=left_degree])?;
    let right_degree = target_degree
        .checked_sub(left_degree)
        .ok_or(TopologyError::InternalInvariant)?;
    let right = owner
        .basis(right_degree)?
        .binary_search(&simplex[left_degree..])?;
    let sign = owner.orientation(target_degree)?[target_index]
        * owner.orientation(left_degree)?[left]
        * owner.orientation(right_degree)?[right];
    Ok(OrientedFacePair { left, right, sign })
}

fn visit_combinations(
    values: &[usize],
    choose: usize,
    start: usize,
    chosen: &mut Vec<usize>,
    visit: &mut impl FnMut(&[usize]) -> Result<(), TopologyError>,
) -> Result<(), TopologyError> {
    if chosen.len() == choose {
        return visit(chosen);
    }
    let needed = choose - chosen.len();
    let final_start = values
        .len()
        .checked_sub(needed)
        .ok_or(TopologyError::InternalInvariant)?;
    for position in start..=final_start {
        chosen.push(values[position]);
        visit_combinations(values, choose, position + 1, chosen, visit)?;
        chosen.pop();
    }
    Ok(())
}

pub(crate) fn wedge_normalization(
    left_degree: usize,
    right_degree: usize,
) -> Result<i64, TopologyError> {
    let target_degree = left_degree
        .checked_add(right_degree)
        .ok_or(TopologyError::CountOverflow)?;
    let mut binomial = 1_u128;
    let choose = left_degree.min(right_degree);
    for step in 0..choose {
        binomial = binomial
            .checked_mul(
                u128::try_from(target_degree - step).map_err(|_| TopologyError::CountOverflow)?,
            )
            .ok_or(TopologyError::CountOverflow)?
            / u128::try_from(step + 1).map_err(|_| TopologyError::CountOverflow)?;
    }
    let count = u128::try_from(target_degree + 1)
        .map_err(|_| TopologyError::CountOverflow)?
        .checked_mul(binomial)
        .ok_or(TopologyError::CountOverflow)?;
    i64::try_from(count).map_err(|_| TopologyError::CountOverflow)
}

pub(crate) fn visit_wedge_face_pairs(
    owner: &ComplexCore,
    left_degree: usize,
    right_degree: usize,
    mut visit: impl FnMut(usize, OrientedFacePair),
) -> Result<(), TopologyError> {
    let target_degree = left_degree
        .checked_add(right_degree)
        .ok_or(TopologyError::CountOverflow)?;
    let target_basis = owner.basis(target_degree)?;
    let left_basis = owner.basis(left_degree)?;
    let right_basis = owner.basis(right_degree)?;
    let target_orientations = owner.orientation(target_degree)?;
    let left_orientations = owner.orientation(left_degree)?;
    let right_orientations = owner.orientation(right_degree)?;
    let mut remaining = Vec::new();
    let mut chosen = Vec::new();
    let mut left_vertices = Vec::new();
    let mut right_vertices = Vec::new();
    let mut in_left = Vec::new();
    remaining
        .try_reserve_exact(target_degree)
        .map_err(|_| TopologyError::Allocation)?;
    chosen
        .try_reserve_exact(left_degree)
        .map_err(|_| TopologyError::Allocation)?;
    left_vertices
        .try_reserve_exact(left_degree + 1)
        .map_err(|_| TopologyError::Allocation)?;
    right_vertices
        .try_reserve_exact(right_degree + 1)
        .map_err(|_| TopologyError::Allocation)?;
    in_left
        .try_reserve_exact(target_degree + 1)
        .map_err(|_| TopologyError::Allocation)?;
    in_left.resize(target_degree + 1, false);

    for (target_index, &target_orientation) in target_orientations.iter().enumerate() {
        let simplex = target_basis
            .row(target_index)
            .ok_or(TopologyError::InternalInvariant)?;
        for shared in 0..=target_degree {
            remaining.clear();
            remaining.extend((0..=target_degree).filter(|&position| position != shared));
            chosen.clear();
            visit_combinations(
                &remaining,
                left_degree,
                0,
                &mut chosen,
                &mut |left_positions| {
                    in_left.fill(false);
                    in_left[shared] = true;
                    for &position in left_positions {
                        in_left[position] = true;
                    }
                    left_vertices.clear();
                    right_vertices.clear();
                    for (position, &vertex) in simplex.iter().enumerate() {
                        if in_left[position] {
                            left_vertices.push(vertex);
                        }
                        if position == shared || !in_left[position] {
                            right_vertices.push(vertex);
                        }
                    }
                    let left = left_basis.binary_search(&left_vertices)?;
                    let right = right_basis.binary_search(&right_vertices)?;
                    let inversions = left_positions
                        .iter()
                        .map(|&left_position| {
                            remaining
                                .iter()
                                .filter(|&&right_position| {
                                    !in_left[right_position] && left_position > right_position
                                })
                                .count()
                        })
                        .sum::<usize>();
                    let shuffle_sign = if inversions % 2 == 0 { 1 } else { -1 };
                    visit(
                        target_index,
                        OrientedFacePair {
                            left,
                            right,
                            sign: target_orientation
                                * left_orientations[left]
                                * right_orientations[right]
                                * shuffle_sign,
                        },
                    );
                    Ok(())
                },
            )?;
        }
    }
    Ok(())
}

impl<A, E> Element<A, Cochain, E>
where
    A: CommutativeRing,
    E: ValueEncoding<A>,
{
    /// Form the exact Alexander--Whitney cup product on canonical simplices.
    ///
    /// # Errors
    ///
    /// Rejects foreign coefficient/topology owners, nonsimplicial domains,
    /// unavailable retained topology, allocation failure, or degree overflow.
    pub fn cup(&self, other: &Self) -> Result<Self, ChainError> {
        if !self.space.complex.same_owner(&other.space.complex) {
            return Err(ChainError::SpaceMismatch);
        }
        let complex = &self.space.complex;
        let owner = complex
            .domain
            .simplicial_owner()
            .ok_or(ChainError::NotSimplicial)?;
        let target_degree = self
            .degree()
            .checked_add(other.degree())
            .ok_or(TopologyError::CountOverflow)?;
        let target = match usize::try_from(target_degree) {
            Ok(degree) if degree <= complex.dimension() => Space::derive(complex, degree)?,
            _ => Space::zero(complex, target_degree),
        };
        if self.basis_size() == 0 || other.basis_size() == 0 || target.basis_size() == 0 {
            return Ok(Self::empty(&target));
        }

        let left_degree =
            usize::try_from(self.degree()).map_err(|_| TopologyError::CountOverflow)?;
        let target_degree =
            usize::try_from(target_degree).map_err(|_| TopologyError::CountOverflow)?;
        let target_basis = owner.basis(target_degree)?;
        let mut indices = Vec::new();
        let mut coefficients = Vec::new();
        indices
            .try_reserve_exact(target_basis.row_count())
            .map_err(|_| TopologyError::Allocation)?;
        coefficients
            .try_reserve_exact(target_basis.row_count())
            .map_err(|_| TopologyError::Allocation)?;
        let algebra = complex.coefficient_system();

        for target_index in 0..target_basis.row_count() {
            let pair = cup_face_pair(owner, left_degree, target_degree, target_index)?;
            let Ok(left_position) = self.indices().binary_search(&pair.left) else {
                continue;
            };
            let Ok(right_position) = other.indices().binary_search(&pair.right) else {
                continue;
            };
            let mut product = algebra.multiply(
                E::element(&self.coefficients()[left_position]),
                E::element(&other.coefficients()[right_position]),
            );
            if pair.sign < 0 {
                product = algebra.negate(&product);
            }
            if !algebra.is_zero(&product) {
                indices.push(target_index);
                coefficients.push(E::encode(product));
            }
        }
        Ok(Self::from_canonical(&target, indices, coefficients))
    }
}

impl<A, E> Element<A, Cochain, E>
where
    A: Field,
    E: ValueEncoding<A>,
{
    /// Form the exact antisymmetrized simplicial wedge product.
    ///
    /// # Errors
    ///
    /// Rejects foreign owners, nonsimplicial domains, unavailable topology,
    /// degree overflow, allocation failure, or a noninvertible averaging factor.
    pub fn wedge(&self, other: &Self) -> Result<Self, ChainError> {
        if !self.space.complex.same_owner(&other.space.complex) {
            return Err(ChainError::SpaceMismatch);
        }
        let complex = &self.space.complex;
        let owner = complex
            .domain
            .simplicial_owner()
            .ok_or(ChainError::NotSimplicial)?;
        let target_degree = self
            .degree()
            .checked_add(other.degree())
            .ok_or(TopologyError::CountOverflow)?;
        let target = match usize::try_from(target_degree) {
            Ok(degree) if degree <= complex.dimension() => Space::derive(complex, degree)?,
            _ => Space::zero(complex, target_degree),
        };
        if self.basis_size() == 0 || other.basis_size() == 0 || target.basis_size() == 0 {
            return Ok(Self::empty(&target));
        }

        let left_degree =
            usize::try_from(self.degree()).map_err(|_| TopologyError::CountOverflow)?;
        let right_degree =
            usize::try_from(other.degree()).map_err(|_| TopologyError::CountOverflow)?;
        let normalization = wedge_normalization(left_degree, right_degree)?;
        let algebra = complex.coefficient_system();
        let inverse = algebra
            .inverse(&algebra.lift_i64(normalization))
            .ok_or(ChainError::NormalizationNotInvertible)?;
        let mut indices = Vec::new();
        let mut coefficients = Vec::new();
        indices
            .try_reserve_exact(target.basis_size())
            .map_err(|_| TopologyError::Allocation)?;
        coefficients
            .try_reserve_exact(target.basis_size())
            .map_err(|_| TopologyError::Allocation)?;
        let mut current_target = None;
        let mut sum = algebra.zero();
        let mut finish = |target_index: usize, sum: &A::Element| {
            let value = algebra.multiply(sum, &inverse);
            if !algebra.is_zero(&value) {
                indices.push(target_index);
                coefficients.push(E::encode(value));
            }
        };
        visit_wedge_face_pairs(owner, left_degree, right_degree, |target_index, pair| {
            if current_target != Some(target_index) {
                if let Some(previous) = current_target {
                    finish(previous, &sum);
                }
                current_target = Some(target_index);
                sum = algebra.zero();
            }
            let Ok(left_position) = self.indices().binary_search(&pair.left) else {
                return;
            };
            let Ok(right_position) = other.indices().binary_search(&pair.right) else {
                return;
            };
            let mut product = algebra.multiply(
                E::element(&self.coefficients()[left_position]),
                E::element(&other.coefficients()[right_position]),
            );
            if pair.sign < 0 {
                product = algebra.negate(&product);
            }
            algebra.add_assign(&mut sum, &product);
        })?;
        if let Some(last) = current_target {
            finish(last, &sum);
        }
        Ok(Self::from_canonical(&target, indices, coefficients))
    }
}

/// Exact primitive tree-cotree generators of a closed oriented surface's dual graph.
///
/// Each retained value is an integral degree-one cocycle in the simplicial
/// complex's induced dual basis. Generator-edge indices retain the deterministic
/// tree-cotree presentation named by the public surface API.
#[derive(Clone, Debug)]
pub struct IntegralDualCycleBasis {
    chain: IntegralChainComplex,
    cycles: Box<[IntegralCochain]>,
    generator_edges: Box<[usize]>,
}

impl IntegralDualCycleBasis {
    /// Borrow the one exact chain authority shared by every retained cocycle.
    #[must_use]
    pub const fn chain_complex(&self) -> &IntegralChainComplex {
        &self.chain
    }

    /// Number of primitive noncontractible generators.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.cycles.len()
    }

    /// Borrow one exact cocycle in the canonical primal-edge basis.
    #[must_use]
    pub fn cocycle(&self, index: usize) -> Option<&IntegralCochain> {
        self.cycles.get(index)
    }

    /// Deterministic non-tree primal edges aligned with the retained generators.
    #[must_use]
    pub const fn generator_edge_indices(&self) -> &[usize] {
        &self.generator_edges
    }
}

impl ComplexCore {
    /// Construct exact primitive dual cycles without retaining geometry or metric data.
    ///
    /// Triangle-manifold, coherent-orientation, connectivity, and empty-boundary
    /// facts are admitted on this topology owner before construction. Operational
    /// failures do not publish partial data.
    ///
    /// # Errors
    ///
    /// Returns a prerequisite topology rejection, allocation failure, or an
    /// internal-invariant error if exact closure cannot be established.
    pub fn integral_dual_cycle_basis(
        self: &Arc<Self>,
    ) -> Result<IntegralDualCycleBasis, TopologyError> {
        self.refine_triangle()?;
        self.refine_oriented()?;
        self.refine_connected()?;
        self.refine_regular()?.without_boundary()?;

        let (dual, generator_edges) = tree_cotree(self)?;
        let chain = self.chain_complex();
        let cochain_space = chain.dual().space(1)?;
        let coboundary = chain.dual().coboundary(1)?;
        let mut cycles = Vec::new();
        cycles
            .try_reserve_exact(generator_edges.len())
            .map_err(|_| TopologyError::Allocation)?;
        for &generator in &generator_edges {
            let dual_coefficients = dual.fundamental_cycle(generator)?;
            let entries = dual_coefficients
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, coefficient)| *coefficient != 0)
                .map(|(edge, dual_coefficient)| {
                    let cochain_coefficient =
                        -dual.edges.source_boundary_sign[edge] * dual_coefficient;
                    (edge, BigInt::from(cochain_coefficient))
                });
            let cycle = cochain_space
                .element(entries)
                .map_err(|_| TopologyError::InternalInvariant)?;
            let residual = coboundary
                .apply(&cycle)
                .map_err(|_| TopologyError::InternalInvariant)?;
            if !residual.indices().is_empty() {
                return Err(TopologyError::InternalInvariant);
            }
            cycles.push(cycle);
        }

        Ok(IntegralDualCycleBasis {
            chain,
            cycles: cycles.into_boxed_slice(),
            generator_edges: generator_edges.into_boxed_slice(),
        })
    }
}

struct DualTree {
    edges: DualEdges,
    adjacency: Vec<Vec<(usize, usize)>>,
}

struct DualEdges {
    source: Vec<usize>,
    target: Vec<usize>,
    source_boundary_sign: Vec<i8>,
}

impl DualTree {
    fn fundamental_cycle(&self, generator: usize) -> Result<Vec<i8>, TopologyError> {
        let mut coefficients = try_filled(self.edges.source.len(), 0_i8)?;
        coefficients[generator] = 1;
        let root = self.edges.target[generator];
        let goal = self.edges.source[generator];
        let face_count = self.adjacency.len();
        let mut parent_face = try_filled(face_count, usize::MAX)?;
        let mut parent_edge = try_filled(face_count, usize::MAX)?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(face_count)
            .map_err(|_| TopologyError::Allocation)?;
        parent_face[root] = root;
        pending.push(root);
        let mut cursor = 0;
        while cursor < pending.len() && parent_face[goal] == usize::MAX {
            let face = pending[cursor];
            cursor += 1;
            for &(neighbor, edge) in &self.adjacency[face] {
                if parent_face[neighbor] == usize::MAX {
                    parent_face[neighbor] = face;
                    parent_edge[neighbor] = edge;
                    pending.push(neighbor);
                }
            }
        }
        if parent_face[goal] == usize::MAX {
            return Err(TopologyError::InternalInvariant);
        }
        let mut face = goal;
        while face != root {
            let previous = parent_face[face];
            let edge = parent_edge[face];
            coefficients[edge] =
                i8::from((previous, face) == (self.edges.source[edge], self.edges.target[edge]))
                    * 2
                    - 1;
            face = previous;
        }
        Ok(coefficients)
    }
}

fn tree_cotree(owner: &ComplexCore) -> Result<(DualTree, Vec<usize>), TopologyError> {
    let edges = oriented_dual_edges(owner)?;
    let edge_count = edges.source.len();
    let face_count = owner.bases[2].row_count();
    let mut vertex_components = DisjointSet::try_new(owner.vertex_count)?;
    let mut primal_tree = try_filled(edge_count, false)?;
    for (edge, in_tree) in primal_tree.iter_mut().enumerate() {
        let endpoints = owner.bases[1]
            .row(edge)
            .ok_or(TopologyError::InternalInvariant)?;
        let [left, right] = *endpoints else {
            return Err(TopologyError::InternalInvariant);
        };
        *in_tree = vertex_components.join(left, right);
    }
    if primal_tree.iter().filter(|in_tree| **in_tree).count()
        != owner.vertex_count.saturating_sub(1)
    {
        return Err(TopologyError::InternalInvariant);
    }

    let mut face_components = DisjointSet::try_new(face_count)?;
    let mut dual_tree = try_filled(edge_count, false)?;
    for (edge, in_tree) in dual_tree.iter_mut().enumerate() {
        *in_tree =
            !primal_tree[edge] && face_components.join(edges.source[edge], edges.target[edge]);
    }
    if dual_tree.iter().filter(|in_tree| **in_tree).count() != face_count.saturating_sub(1) {
        return Err(TopologyError::InternalInvariant);
    }

    let generator_edges =
        selected_edges(edge_count, |edge| !primal_tree[edge] && !dual_tree[edge])?;
    let adjacency = dual_tree_adjacency(face_count, &edges.source, &edges.target, &dual_tree)?;
    Ok((DualTree { edges, adjacency }, generator_edges))
}

fn oriented_dual_edges(owner: &ComplexCore) -> Result<DualEdges, TopologyError> {
    let edge_count = owner.bases[1].row_count();
    let mut source = try_filled(edge_count, 0_usize)?;
    let mut target = try_filled(edge_count, 0_usize)?;
    let mut source_sign = try_filled(edge_count, 0_i8)?;
    for (edge, ((source, target), source_sign)) in source
        .iter_mut()
        .zip(&mut target)
        .zip(&mut source_sign)
        .enumerate()
    {
        let (faces, signs) = owner.boundaries[2]
            .storage
            .row(edge)
            .ok_or(TopologyError::InternalInvariant)?;
        let [first, second] = *faces else {
            return Err(TopologyError::InternalInvariant);
        };
        let [first_sign, second_sign] = *signs else {
            return Err(TopologyError::InternalInvariant);
        };
        if first_sign != -second_sign {
            return Err(TopologyError::InternalInvariant);
        }
        (*source, *target, *source_sign) = if first < second {
            (first, second, first_sign)
        } else {
            (second, first, second_sign)
        };
    }
    Ok(DualEdges {
        source,
        target,
        source_boundary_sign: source_sign,
    })
}

fn selected_edges(
    edge_count: usize,
    predicate: impl Fn(usize) -> bool,
) -> Result<Vec<usize>, TopologyError> {
    let mut edges = Vec::new();
    edges
        .try_reserve_exact(edge_count)
        .map_err(|_| TopologyError::Allocation)?;
    edges.extend((0..edge_count).filter(|edge| predicate(*edge)));
    Ok(edges)
}

fn dual_tree_adjacency(
    face_count: usize,
    source: &[usize],
    target: &[usize],
    dual_tree: &[bool],
) -> Result<Vec<Vec<(usize, usize)>>, TopologyError> {
    let mut adjacency = Vec::new();
    adjacency
        .try_reserve_exact(face_count)
        .map_err(|_| TopologyError::Allocation)?;
    adjacency.resize_with(face_count, Vec::<(usize, usize)>::new);
    for edge in dual_tree
        .iter()
        .enumerate()
        .filter_map(|(edge, in_tree)| (*in_tree).then_some(edge))
    {
        adjacency[source[edge]]
            .try_reserve(1)
            .map_err(|_| TopologyError::Allocation)?;
        adjacency[target[edge]]
            .try_reserve(1)
            .map_err(|_| TopologyError::Allocation)?;
        adjacency[source[edge]].push((target[edge], edge));
        adjacency[target[edge]].push((source[edge], edge));
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable_by_key(|(neighbor, edge)| (*edge, *neighbor));
    }
    Ok(adjacency)
}

/// Exact integral chain complex.
pub type IntegralChainComplex = ChainComplex<IntegerRing>;

/// Exact integral chain in a single degree.
pub type IntegralChain = Element<IntegerRing, Chain, BigIntEncoding>;

/// Exact integral cochain in a single degree.
pub type IntegralCochain = Element<IntegerRing, Cochain, BigIntEncoding>;

/// Exact integral degree map between variance-indexed spaces.
pub type IntegralLinearMap<S, T> = LinearMap<IntegerRing, S, T>;
