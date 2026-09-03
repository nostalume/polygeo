use std::{
    marker::PhantomData,
    mem::size_of,
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use faer::{
    Conj, Mat, MatMut, Par, Side, Spec,
    dyn_stack::{MemBuffer, MemStack, StackReq},
    linalg::cholesky::llt::{self, factor::LltRegularization},
    linalg::householder,
    linalg::lu::partial_pivoting::{factor as lu_factor, solve as lu_solve},
    linalg::qr::col_pivoting::factor as qr_factor,
    perm::PermRef,
    sparse::{
        CreationError, FaerError, SparseColMat, Triplet,
        linalg::cholesky::{
            CholeskySymbolicParams, LltRef as SparseLltRef, SymbolicCholesky, SymmetricOrdering,
            factorize_symbolic_cholesky,
        },
    },
};
use num_traits::{CheckedAdd, ToPrimitive, Zero};

use crate::form_impl::next_up;
use crate::incidence::{IncidenceAxis, independent_incidence};
use crate::numeric::{
    adaptive_product_sum, adaptive_product_value, adaptive_triple_product_sum, exact_dot_is_zero,
};
use crate::{
    Binary64Chain, Binary64Cochain, Binary64Element, Binary64ElementError, Binary64Space,
    CanonicalSelection, Cochain, GeometryError, LinearOperator, Metric, NondegenerateCapability,
    OperatorError, PairingCapability, SurfaceError, TopologyError, Variance,
};

/// Portable logical-storage ceiling for one unpublished operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StorageLimit {
    pub(crate) retained_logical_bytes: u64,
    pub(crate) peak_live_logical_bytes: u64,
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

mod private {
    pub trait Sealed {}
}

/// A solver-free admitted mathematical problem.
pub trait Problem: private::Sealed {
    type Solution;
}

/// Failure to admit or certify one numerical problem.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ProblemError {
    SpaceMismatch,
    IncompatibleRhs,
    TimeStep,
    Topology,
    Metric,
    BoundarySelection,
    Numerical,
    Element(Binary64ElementError),
}

impl ProblemError {
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::SpaceMismatch => "space_mismatch",
            Self::IncompatibleRhs => "incompatible_rhs",
            Self::TimeStep => "time_step",
            Self::Topology => "topology",
            Self::Metric => "metric",
            Self::BoundarySelection => "boundary_selection",
            Self::Numerical => "numerical",
            Self::Element(error) => error.reason(),
        }
    }
}

impl std::fmt::Display for ProblemError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.reason())
    }
}

impl std::error::Error for ProblemError {}

impl From<TopologyError> for ProblemError {
    fn from(_: TopologyError) -> Self {
        Self::Topology
    }
}

impl From<GeometryError> for ProblemError {
    fn from(_: GeometryError) -> Self {
        Self::Metric
    }
}

impl From<Binary64ElementError> for ProblemError {
    fn from(error: Binary64ElementError) -> Self {
        Self::Element(error)
    }
}

impl From<OperatorError> for ProblemError {
    fn from(error: OperatorError) -> Self {
        match error {
            OperatorError::Topology(_) => Self::Topology,
            OperatorError::Geometry(_) => Self::Metric,
            OperatorError::SpaceMismatch | OperatorError::FullSpaceRequired => Self::SpaceMismatch,
            _ => Self::Numerical,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CompatibilityEvidence {
    rhs_l1: f64,
}

#[derive(Clone, Debug)]
enum PoissonRhs {
    Density(Binary64Cochain),
    Load(Binary64Chain),
}

impl PoissonRhs {
    fn len(&self) -> usize {
        match self {
            Self::Density(value) => value.coefficients().len(),
            Self::Load(value) => value.coefficients().len(),
        }
    }

    fn weak_value(&self, index: usize, masses: &[f64]) -> Option<f64> {
        let value = match self {
            Self::Density(density) => masses[index] * density.coefficients()[index],
            Self::Load(load) => load.coefficients()[index],
        };
        value.is_finite().then_some(value)
    }
}

/// Compatible right-hand side for a mean-zero degree-zero Poisson equation.
#[derive(Clone, Debug)]
pub struct PoissonProblem {
    metric: Metric,
    rhs: PoissonRhs,
    compatibility: CompatibilityEvidence,
}

impl private::Sealed for PoissonProblem {}
impl Problem for PoissonProblem {
    type Solution = PoissonResult;
}

/// One admitted backward-Euler evolution of a full scalar vertex cochain.
#[derive(Clone, Debug)]
pub struct HeatProblem {
    metric: Metric,
    source: Binary64Cochain,
    time_step: f64,
}

impl private::Sealed for HeatProblem {}
impl Problem for HeatProblem {
    type Solution = HeatResult;
}

impl HeatProblem {
    pub(crate) const fn metric(&self) -> &Metric {
        &self.metric
    }

    pub(crate) const fn source(&self) -> &Binary64Cochain {
        &self.source
    }

    #[must_use]
    pub const fn time_step(&self) -> f64 {
        self.time_step
    }
}

impl PoissonProblem {
    pub(crate) const fn metric(&self) -> &Metric {
        &self.metric
    }
    pub(crate) fn len(&self) -> usize {
        self.rhs.len()
    }
    pub(crate) fn weak_value(&self, index: usize, masses: &[f64]) -> Option<f64> {
        self.rhs.weak_value(index, masses)
    }
    /// Sum of absolute weak right-hand-side terms used by compatibility admission.
    #[must_use]
    pub const fn compatibility_scale(&self) -> f64 {
        self.compatibility.rhs_l1
    }

    pub(crate) fn certify(
        &self,
        potential: Binary64Cochain,
        stack: &mut MemStack,
    ) -> Result<PoissonResult, ProblemError> {
        let metric = self.metric();
        let vertex_weights = metric.hodge_coefficients_slice(0)?;
        if vertex_weights.len() == 1 {
            if self.weak_value(0, vertex_weights) != Some(0.0) || potential.coefficients()[0] != 0.0
            {
                return Err(ProblemError::Numerical);
            }
            return Ok(PoissonResult::new(
                potential,
                ResidualEvidence::new(0.0, 0.0, 0),
            ));
        }
        let edge_weights = metric.hodge_coefficients_slice(1)?;
        let boundary = metric.realization().topology().chain_view().boundary(1)?;
        let (mut endpoints, stack) =
            stack.make_with(boundary.shape().1, |_| [(usize::MAX, 0_i64); 2]);
        let (mut counts, _) = stack.make_with(boundary.shape().1, |_| 0_usize);
        for (row, edge, sign) in boundary.exact_entries() {
            let slot = counts[edge];
            if slot == 2 {
                return Err(ProblemError::Numerical);
            }
            endpoints[edge][slot] = (row, sign);
            counts[edge] += 1;
        }
        if counts.iter().any(|&count| count != 2) {
            return Err(ProblemError::Numerical);
        }
        let mut residual_bound = 0.0_f64;
        let mut fallbacks = 0_usize;
        let indptr = boundary.indptr();
        let indices = boundary.indices();
        let potential_coefficients = potential.coefficients();
        for row in 0..vertex_weights.len() {
            let rhs = self
                .weak_value(row, vertex_weights)
                .ok_or(ProblemError::Numerical)?;
            let terms = indices[indptr[row]..indptr[row + 1]]
                .iter()
                .flat_map(|&edge| {
                    let entries = endpoints[edge];
                    let row_sign = if entries[0].0 == row {
                        entries[0].1
                    } else {
                        entries[1].1
                    };
                    entries.into_iter().map(move |(column, column_sign)| {
                        let sign = if row_sign == column_sign { 1.0 } else { -1.0 };
                        (sign * edge_weights[edge], potential_coefficients[column])
                    })
                })
                .chain(std::iter::once((-1.0, rhs)));
            let scale = terms
                .clone()
                .map(|(a, b)| (a * b).abs())
                .sum::<f64>()
                .max(vertex_weights[row]);
            let tolerance = 256.0 * f64::EPSILON * scale;
            let verdict = adaptive_product_sum(terms, tolerance);
            if !verdict.accepted {
                return Err(ProblemError::Numerical);
            }
            fallbacks += usize::from(verdict.exact_fallback);
            residual_bound = residual_bound.max(verdict.bound / vertex_weights[row]);
        }
        let gauge_terms = vertex_weights
            .iter()
            .copied()
            .zip(potential_coefficients.iter().copied());
        let gauge_scale = gauge_terms
            .clone()
            .map(|(a, b)| (a * b).abs())
            .sum::<f64>()
            .max(1.0);
        let gauge = adaptive_product_sum(gauge_terms, 256.0 * f64::EPSILON * gauge_scale);
        if !gauge.accepted {
            return Err(ProblemError::Numerical);
        }
        fallbacks += usize::from(gauge.exact_fallback);
        Ok(PoissonResult::new(
            potential,
            ResidualEvidence::new(residual_bound, gauge.bound, fallbacks),
        ))
    }
}

impl Metric {
    fn require_mean_zero_space<K: Variance>(
        &self,
        rhs: &Binary64Element<K>,
    ) -> Result<(), ProblemError> {
        let owner = self.realization().topology();
        owner.refine_connected()?;
        owner.refine_regular()?.without_boundary()?;
        let expected = Binary64Space::<K>::full(Arc::clone(owner), 0)?;
        expected
            .same_space(rhs.space())
            .then_some(())
            .ok_or(ProblemError::SpaceMismatch)
    }

    /// Admit one backward-Euler step for a full scalar vertex cochain.
    ///
    /// # Errors
    /// Rejects a foreign, selected, non-vertex source or nonpositive/nonfinite time.
    pub fn heat_evolution(
        &self,
        source: Binary64Cochain,
        time_step: f64,
    ) -> Result<HeatProblem, ProblemError> {
        if !time_step.is_finite() || time_step <= 0.0 {
            return Err(ProblemError::TimeStep);
        }
        if !source.space().is_full() {
            return Err(ProblemError::SpaceMismatch);
        }
        let expected =
            Binary64Space::<Cochain>::full(Arc::clone(self.realization().topology()), 0)?;
        if !expected.same_space(source.space()) {
            return Err(ProblemError::SpaceMismatch);
        }
        self.hodge_coefficients_slice(0)?;
        if self.realization().topology().dimension() > 0 {
            self.hodge_coefficients_slice(1)?;
        }
        Ok(HeatProblem {
            metric: self.clone(),
            source,
            time_step,
        })
    }

    /// Admit a compatible density on a connected closed complex.
    ///
    /// # Errors
    /// Rejects a foreign/non-vertex density, unsuitable topology, or a
    /// nonzero exact binary64 weighted sum.
    pub fn mean_zero_poisson_density(
        &self,
        density: Binary64Cochain,
    ) -> Result<PoissonProblem, ProblemError> {
        self.require_mean_zero_space(&density)?;
        let weights = self.hodge_coefficients_slice(0)?;
        if !exact_dot_is_zero(weights, density.coefficients()) {
            return Err(ProblemError::IncompatibleRhs);
        }
        let weighted_l1 = weights
            .iter()
            .zip(density.coefficients())
            .map(|(&weight, &value)| (weight * value).abs())
            .sum();
        Ok(PoissonProblem {
            metric: self.clone(),
            rhs: PoissonRhs::Density(density),
            compatibility: CompatibilityEvidence {
                rhs_l1: weighted_l1,
            },
        })
    }

    /// Admit a compatible integrated degree-zero load on a connected closed complex.
    ///
    /// # Errors
    /// Rejects a foreign/non-vertex load, unsuitable topology, or a binary64
    /// coefficient sum outside the scale-relative compatibility bound.
    pub fn mean_zero_poisson_load(
        &self,
        load: Binary64Chain,
    ) -> Result<PoissonProblem, ProblemError> {
        self.require_mean_zero_space(&load)?;
        let mut l1 = 0.0_f64;
        for coefficient in load.coefficients() {
            l1 = next_up(l1 + coefficient.abs());
        }
        if !l1.is_finite() {
            return Err(ProblemError::Numerical);
        }
        let operation_count = f64::from(
            u32::try_from(load.coefficients().len().saturating_add(1)).unwrap_or(u32::MAX),
        );
        let tolerance = next_up(256.0 * f64::EPSILON * l1 + operation_count * f64::from_bits(1));
        let compatibility = adaptive_product_sum(
            load.coefficients()
                .iter()
                .copied()
                .map(|value| (1.0, value)),
            tolerance,
        );
        if !compatibility.accepted {
            return Err(ProblemError::IncompatibleRhs);
        }
        Ok(PoissonProblem {
            metric: self.clone(),
            rhs: PoissonRhs::Load(load),
            compatibility: CompatibilityEvidence { rhs_l1: l1 },
        })
    }

    /// Admit degree-zero values on the complete topological boundary.
    ///
    /// # Errors
    /// Rejects a foreign, incomplete, or non-boundary selected basis.
    pub fn harmonic_extension(
        &self,
        boundary_values: Binary64Cochain,
    ) -> Result<HarmonicExtension, ProblemError> {
        let selection = selected_boundary(&boundary_values)?;
        if !Arc::ptr_eq(selection.owner(), self.realization().topology()) {
            return Err(ProblemError::SpaceMismatch);
        }
        self.realization().topology().refine_connected()?;
        require_boundary(selection, true)?;
        Ok(HarmonicExtension {
            metric: self.clone(),
            boundary_values,
        })
    }

    /// Admit one full primal cochain for metric Hodge decomposition.
    ///
    /// # Errors
    /// Rejects a selected/foreign cochain or an unavailable metric degree.
    pub fn hodge_decomposition(
        &self,
        source: Binary64Cochain,
    ) -> Result<HodgeProblem, ProblemError> {
        if !source.space().is_full() {
            return Err(ProblemError::SpaceMismatch);
        }
        let degree =
            usize::try_from(source.space().degree()).map_err(|_| ProblemError::SpaceMismatch)?;
        let expected =
            Binary64Space::<Cochain>::full(Arc::clone(self.realization().topology()), degree)?;
        if !expected.same_space(source.space()) {
            return Err(ProblemError::SpaceMismatch);
        }
        self.hodge_coefficients_slice(degree)?;
        Ok(HodgeProblem {
            metric: self.clone(),
            source,
        })
    }
}

/// One metric-owned cochain decomposition problem without retained image data.
#[derive(Clone, Debug)]
pub struct HodgeProblem {
    metric: Metric,
    source: Binary64Cochain,
}

impl private::Sealed for HodgeProblem {}
impl Problem for HodgeProblem {
    type Solution = HodgeDecomposition;
}

impl HodgeProblem {
    pub(crate) const fn metric(&self) -> &Metric {
        &self.metric
    }

    pub(crate) const fn source(&self) -> &Binary64Cochain {
        &self.source
    }

    pub(crate) fn degree(&self) -> Result<usize, ProblemError> {
        usize::try_from(self.source.space().degree()).map_err(|_| ProblemError::SpaceMismatch)
    }

    pub(crate) fn certify(
        &self,
        exact: Binary64Cochain,
        coexact: Binary64Cochain,
        harmonic: Binary64Cochain,
        ranks: [usize; 2],
        condition_indicators: [f64; 2],
    ) -> Result<HodgeDecomposition, ProblemError> {
        if !self.source.space().same_space(exact.space())
            || !self.source.space().same_space(coexact.space())
            || !self.source.space().same_space(harmonic.space())
        {
            return Err(ProblemError::SpaceMismatch);
        }
        let degree = self.degree()?;
        let weights = self.metric.hodge_coefficients_slice(degree)?;
        let scale = self
            .source
            .coefficients()
            .iter()
            .copied()
            .map(f64::abs)
            .fold(1.0_f64, f64::max);
        let tolerance = 4096.0 * f64::EPSILON * scale;
        let mut exact_fallback_predicates = 0_usize;
        let mut reconstruction_bound = 0.0_f64;
        for (((&source, &exact), &coexact), &harmonic) in self
            .source
            .coefficients()
            .iter()
            .zip(exact.coefficients())
            .zip(coexact.coefficients())
            .zip(harmonic.coefficients())
        {
            let verdict = adaptive_product_sum(
                [
                    (1.0, exact),
                    (1.0, coexact),
                    (1.0, harmonic),
                    (-1.0, source),
                ]
                .into_iter(),
                tolerance,
            );
            if !verdict.accepted {
                return Err(ProblemError::Numerical);
            }
            exact_fallback_predicates += usize::from(verdict.exact_fallback);
            reconstruction_bound = reconstruction_bound.max(verdict.bound);
        }
        let orthogonality = adaptive_triple_product_sum(
            weights
                .iter()
                .copied()
                .zip(exact.coefficients().iter().copied())
                .zip(coexact.coefficients().iter().copied())
                .map(|((weight, left), right)| (weight, left, right)),
            tolerance,
        );
        if !orthogonality.accepted {
            return Err(ProblemError::Numerical);
        }
        exact_fallback_predicates += usize::from(orthogonality.exact_fallback);
        let (closure_bound, coclosure_bound, differential_fallbacks) = hodge_differential_bounds(
            &self.metric,
            degree,
            &exact,
            &coexact,
            &harmonic,
            tolerance,
        )?;
        exact_fallback_predicates += differential_fallbacks;
        if !closure_bound.is_finite()
            || !coclosure_bound.is_finite()
            || ranks
                .iter()
                .any(|&rank| rank > self.source.coefficients().len())
            || condition_indicators
                .iter()
                .any(|condition| !condition.is_finite())
        {
            return Err(ProblemError::Numerical);
        }
        Ok(HodgeDecomposition {
            exact,
            coexact,
            harmonic,
            evidence: HodgeEvidence {
                reconstruction_bound,
                orthogonality_bound: orthogonality.bound,
                differential_bounds: [closure_bound, coclosure_bound],
                ranks,
                condition_indicators,
                exact_fallback_predicates,
            },
        })
    }
}

fn hodge_differential_bounds(
    metric: &Metric,
    degree: usize,
    exact: &Binary64Cochain,
    coexact: &Binary64Cochain,
    harmonic: &Binary64Cochain,
    tolerance: f64,
) -> Result<(f64, f64, usize), ProblemError> {
    let topology = metric.realization().topology();
    let (closure, closure_fallbacks) = if degree < topology.dimension() {
        certify_closed(
            topology.chain_view().boundary(degree + 1)?,
            [exact, harmonic],
            tolerance,
        )?
    } else {
        (0.0, 0)
    };
    let (coclosure, coclosure_fallbacks) = if degree > 0 {
        certify_coclosed(metric, degree, [coexact, harmonic], tolerance)?
    } else {
        (0.0, 0)
    };
    Ok((closure, coclosure, closure_fallbacks + coclosure_fallbacks))
}

fn certify_closed(
    boundary: crate::BoundaryRef<'_>,
    values: [&Binary64Cochain; 2],
    tolerance: f64,
) -> Result<(f64, usize), ProblemError> {
    let mut bound = 0.0_f64;
    let mut fallbacks = 0_usize;
    for value in values {
        for target in 0..boundary.shape().1 {
            let verdict = adaptive_product_sum(
                boundary
                    .exact_entries()
                    .filter(move |&(_, column, _)| column == target)
                    .map(move |(row, _, coefficient)| {
                        (
                            coefficient
                                .to_f64()
                                .expect("every i64 has a finite binary64 image"),
                            value.coefficients()[row],
                        )
                    }),
                tolerance,
            );
            if !verdict.accepted {
                return Err(ProblemError::Numerical);
            }
            bound = bound.max(verdict.bound);
            fallbacks += usize::from(verdict.exact_fallback);
        }
    }
    Ok((bound, fallbacks))
}

fn certify_coclosed(
    metric: &Metric,
    degree: usize,
    values: [&Binary64Cochain; 2],
    tolerance: f64,
) -> Result<(f64, usize), ProblemError> {
    let boundary = metric
        .realization()
        .topology()
        .chain_view()
        .boundary(degree)?;
    let source_weights = metric.hodge_coefficients_slice(degree)?;
    let target_weights = metric.hodge_coefficients_slice(degree - 1)?;
    let mut bound = 0.0_f64;
    let mut fallbacks = 0_usize;
    for value in values {
        for (target, &target_weight) in target_weights.iter().enumerate() {
            let verdict = adaptive_triple_product_sum(
                boundary
                    .exact_entries()
                    .filter(move |&(row, _, _)| row == target)
                    .map(move |(_, column, coefficient)| {
                        (
                            coefficient
                                .to_f64()
                                .expect("every i64 has a finite binary64 image"),
                            source_weights[column] / target_weight,
                            value.coefficients()[column],
                        )
                    }),
                tolerance,
            );
            if !verdict.accepted {
                return Err(ProblemError::Numerical);
            }
            bound = bound.max(verdict.bound);
            fallbacks += usize::from(verdict.exact_fallback);
        }
    }
    Ok((bound, fallbacks))
}

/// Small numerical evidence that changes Hodge-result admission.
#[derive(Clone, Copy, Debug)]
pub struct HodgeEvidence {
    reconstruction_bound: f64,
    orthogonality_bound: f64,
    differential_bounds: [f64; 2],
    ranks: [usize; 2],
    condition_indicators: [f64; 2],
    exact_fallback_predicates: usize,
}

impl HodgeEvidence {
    #[must_use]
    pub const fn reconstruction_bound(self) -> f64 {
        self.reconstruction_bound
    }
    #[must_use]
    pub const fn orthogonality_bound(self) -> f64 {
        self.orthogonality_bound
    }
    #[must_use]
    pub const fn closure_bound(self) -> f64 {
        self.differential_bounds[0]
    }
    #[must_use]
    pub const fn coclosure_bound(self) -> f64 {
        self.differential_bounds[1]
    }
    #[must_use]
    pub const fn exact_rank(self) -> usize {
        self.ranks[0]
    }
    #[must_use]
    pub const fn coexact_rank(self) -> usize {
        self.ranks[1]
    }
    #[must_use]
    pub const fn exact_condition_indicator(self) -> f64 {
        self.condition_indicators[0]
    }
    #[must_use]
    pub const fn coexact_condition_indicator(self) -> f64 {
        self.condition_indicators[1]
    }
    #[must_use]
    pub const fn exact_fallback_predicates(self) -> usize {
        self.exact_fallback_predicates
    }
}

/// Exact, coexact, and harmonic components in one original cochain space.
#[derive(Clone, Debug)]
pub struct HodgeDecomposition {
    exact: Binary64Cochain,
    coexact: Binary64Cochain,
    harmonic: Binary64Cochain,
    evidence: HodgeEvidence,
}

impl HodgeDecomposition {
    #[must_use]
    pub const fn exact(&self) -> &Binary64Cochain {
        &self.exact
    }
    #[must_use]
    pub const fn coexact(&self) -> &Binary64Cochain {
        &self.coexact
    }
    #[must_use]
    pub const fn harmonic(&self) -> &Binary64Cochain {
        &self.harmonic
    }
    #[must_use]
    pub const fn evidence(&self) -> HodgeEvidence {
        self.evidence
    }
}

/// A square binary64 cochain equation with canonical boundary coordinates.
#[derive(Clone, Debug)]
pub struct DirichletProblem {
    operator: LinearOperator<Cochain, Cochain>,
    rhs: Binary64Cochain,
    prescribed: Binary64Cochain,
}

impl private::Sealed for DirichletProblem {}
impl Problem for DirichletProblem {
    type Solution = DirichletResult;
}

impl LinearOperator<Cochain, Cochain> {
    /// Admit prescribed coordinates for a square full-space equation.
    ///
    /// # Errors
    /// Rejects endpoint, RHS, selected-space, or true-boundary mismatch.
    pub fn dirichlet(
        &self,
        rhs: Binary64Cochain,
        prescribed: Binary64Cochain,
    ) -> Result<DirichletProblem, ProblemError> {
        if !self.source().is_full()
            || !self.target().is_full()
            || !self.source().same_space(self.target())
            || !self.target().same_space(rhs.space())
        {
            return Err(ProblemError::SpaceMismatch);
        }
        let selection = selected_boundary(&prescribed)?;
        let parent =
            Binary64Space::<Cochain>::full(Arc::clone(selection.owner()), selection.degree())?;
        if !self.source().same_space(&parent) {
            return Err(ProblemError::SpaceMismatch);
        }
        require_boundary(selection, false)?;
        Ok(DirichletProblem {
            operator: self.clone(),
            rhs,
            prescribed,
        })
    }
}

impl DirichletProblem {
    pub(crate) const fn operator(&self) -> &LinearOperator<Cochain, Cochain> {
        &self.operator
    }
    pub(crate) const fn rhs(&self) -> &Binary64Cochain {
        &self.rhs
    }
    pub(crate) const fn prescribed(&self) -> &Binary64Cochain {
        &self.prescribed
    }
}

/// Boundary interpolation induced by one positive metric.
#[derive(Clone, Debug)]
pub struct HarmonicExtension {
    metric: Metric,
    boundary_values: Binary64Cochain,
}

impl private::Sealed for HarmonicExtension {}
impl Problem for HarmonicExtension {
    type Solution = DirichletResult;
}

impl HarmonicExtension {
    pub(crate) const fn metric(&self) -> &Metric {
        &self.metric
    }
    pub(crate) const fn prescribed(&self) -> &Binary64Cochain {
        &self.boundary_values
    }
}

fn selected_boundary(value: &Binary64Cochain) -> Result<&Arc<CanonicalSelection>, ProblemError> {
    value
        .space()
        .canonical_selection()
        .ok_or(ProblemError::SpaceMismatch)
}

fn require_boundary(selection: &CanonicalSelection, complete: bool) -> Result<(), ProblemError> {
    selection.owner().refine_regular()?;
    let mask = selection
        .owner()
        .boundary_subset()?
        .mask(selection.degree())?;
    if selection.indices().iter().any(|&index| !mask[index])
        || complete && mask.iter().filter(|&&value| value).count() != selection.len()
    {
        return Err(ProblemError::BoundarySelection);
    }
    Ok(())
}

/// Conservative evidence for an interior equation and exact prescription.
#[derive(Clone, Copy, Debug)]
pub struct DirichletEvidence {
    residual_bound: f64,
    exact_fallback_rows: usize,
}

impl DirichletEvidence {
    #[must_use]
    pub const fn residual_bound(self) -> f64 {
        self.residual_bound
    }
    #[must_use]
    pub const fn exact_fallback_rows(self) -> usize {
        self.exact_fallback_rows
    }
    pub(crate) const fn new(residual_bound: f64, exact_fallback_rows: usize) -> Self {
        Self {
            residual_bound,
            exact_fallback_rows,
        }
    }
}

/// Immutable reconstructed cochain with certified interior and boundary laws.
#[derive(Clone, Debug)]
pub struct DirichletResult {
    value: Binary64Cochain,
    evidence: DirichletEvidence,
}

impl DirichletResult {
    #[must_use]
    pub const fn value(&self) -> &Binary64Cochain {
        &self.value
    }
    #[must_use]
    pub const fn evidence(&self) -> DirichletEvidence {
        self.evidence
    }
    pub(crate) const fn new(value: Binary64Cochain, evidence: DirichletEvidence) -> Self {
        Self { value, evidence }
    }
}

/// Conservative evidence attached to an admitted Poisson result.
#[derive(Clone, Copy, Debug)]
pub struct ResidualEvidence {
    residual_bound: f64,
    gauge_bound: f64,
    exact_fallback_rows: usize,
}

impl ResidualEvidence {
    #[must_use]
    pub const fn residual_bound(self) -> f64 {
        self.residual_bound
    }
    #[must_use]
    pub const fn gauge_bound(self) -> f64 {
        self.gauge_bound
    }
    #[must_use]
    pub const fn exact_fallback_rows(self) -> usize {
        self.exact_fallback_rows
    }
    const fn new(residual_bound: f64, gauge_bound: f64, exact_fallback_rows: usize) -> Self {
        Self {
            residual_bound,
            gauge_bound,
            exact_fallback_rows,
        }
    }
}

/// Immutable potential and problem-specific numerical evidence.
#[derive(Clone, Debug)]
pub struct PoissonResult {
    potential: Binary64Cochain,
    evidence: ResidualEvidence,
}

/// One scalar heat value with the evidence required for publication.
#[derive(Clone, Debug)]
pub struct HeatResult {
    value: Binary64Cochain,
    residual_bound: f64,
    mass_residual_bound: f64,
    energy_before: f64,
    energy_after: f64,
    exact_fallback_rows: usize,
}

impl HeatResult {
    #[must_use]
    pub const fn value(&self) -> &Binary64Cochain {
        &self.value
    }

    #[must_use]
    pub const fn residual_bound(&self) -> f64 {
        self.residual_bound
    }

    #[must_use]
    pub const fn mass_residual_bound(&self) -> f64 {
        self.mass_residual_bound
    }

    #[must_use]
    pub const fn energy_before(&self) -> f64 {
        self.energy_before
    }

    #[must_use]
    pub const fn energy_after(&self) -> f64 {
        self.energy_after
    }

    #[must_use]
    pub const fn exact_fallback_rows(&self) -> usize {
        self.exact_fallback_rows
    }

    pub(crate) const fn new(
        value: Binary64Cochain,
        residual_bound: f64,
        mass_residual_bound: f64,
        energy_before: f64,
        energy_after: f64,
        exact_fallback_rows: usize,
    ) -> Self {
        Self {
            value,
            residual_bound,
            mass_residual_bound,
            energy_before,
            energy_after,
            exact_fallback_rows,
        }
    }
}

impl PoissonResult {
    #[must_use]
    pub const fn potential(&self) -> &Binary64Cochain {
        &self.potential
    }
    #[must_use]
    pub const fn evidence(&self) -> ResidualEvidence {
        self.evidence
    }
    pub(crate) const fn new(potential: Binary64Cochain, evidence: ResidualEvidence) -> Self {
        Self {
            potential,
            evidence,
        }
    }
}

pub(crate) type EdgeEndpoints = [(usize, i64); 2];

/// Explicit native execution policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Executor {
    parallelism: Option<NonZeroUsize>,
}

impl Executor {
    #[must_use]
    pub const fn sequential() -> Self {
        Self { parallelism: None }
    }
    #[must_use]
    pub const fn parallel(threads: NonZeroUsize) -> Self {
        Self {
            parallelism: Some(threads),
        }
    }
    pub(crate) const fn par(self) -> Par {
        match self.parallelism {
            Some(threads) => Par::Rayon(threads),
            None => Par::Seq,
        }
    }
}

/// Runtime policy retained by prepared computations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Policy {
    executor: Executor,
    storage: StorageLimit,
    work: WorkLimit,
}

impl Policy {
    #[must_use]
    pub const fn new(executor: Executor, storage: StorageLimit, work: WorkLimit) -> Self {
        Self {
            executor,
            storage,
            work,
        }
    }

    #[must_use]
    pub const fn executor(self) -> Executor {
        self.executor
    }

    #[must_use]
    pub const fn storage(self) -> StorageLimit {
        self.storage
    }

    #[must_use]
    pub const fn work(self) -> WorkLimit {
        self.work
    }
}

/// Cooperative cancellation checked outside backend factor calls.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Failure at a preparation, workspace, solve, or certification boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SolveError {
    ProblemMismatch,
    ResourceLimit,
    Cancelled,
    Factorization,
    Numerical,
    Allocation,
}

impl SolveError {
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::ProblemMismatch => "problem_mismatch",
            Self::ResourceLimit => "resource_limit",
            Self::Cancelled => "cancelled",
            Self::Factorization => "factorization",
            Self::Numerical => "numerical",
            Self::Allocation => "allocation",
        }
    }
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.reason())
    }
}
impl std::error::Error for SolveError {}

/// Failure while admitting or executing one direct surface computation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SurfaceComputationError {
    Surface(SurfaceError),
    Solve(SolveError),
}

impl SurfaceComputationError {
    #[must_use]
    pub const fn surface(self) -> Option<SurfaceError> {
        match self {
            Self::Surface(error) => Some(error),
            Self::Solve(_) => None,
        }
    }

    #[must_use]
    pub const fn solve(self) -> Option<SolveError> {
        match self {
            Self::Surface(_) => None,
            Self::Solve(error) => Some(error),
        }
    }
}

impl std::fmt::Display for SurfaceComputationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Surface(error) => error.fmt(formatter),
            Self::Solve(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SurfaceComputationError {}

impl From<SurfaceError> for SurfaceComputationError {
    fn from(error: SurfaceError) -> Self {
        Self::Surface(error)
    }
}

impl From<SolveError> for SurfaceComputationError {
    fn from(error: SolveError) -> Self {
        Self::Solve(error)
    }
}

#[derive(Debug)]
pub(crate) enum Factor {
    Analytic,
    Diagonal {
        inverse: Box<[f64]>,
        scale: f64,
    },
    DenseLlt {
        factor: Mat<f64>,
        scale: f64,
    },
    SparseLlt {
        symbolic: Box<SymbolicCholesky<usize>>,
        numeric: Box<[f64]>,
        scale: f64,
    },
    DenseLu {
        factor: Mat<f64>,
        permutation: Box<[usize]>,
        inverse_permutation: Box<[usize]>,
        scale: f64,
    },
    DenseQr {
        vectors: Mat<f64>,
        householder: Mat<f64>,
        rank: usize,
        condition_indicator: f64,
    },
}

#[derive(Debug)]
enum Factors {
    One(Factor),
    Hodge([Factor; 2]),
}

impl std::ops::Deref for Factors {
    type Target = [Factor];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::One(factor) => std::slice::from_ref(factor),
            Self::Hodge(factors) => factors,
        }
    }
}

#[derive(Debug)]
enum ReuseKey {
    MeanZero {
        owner: Arc<crate::Geometry>,
    },
    Dirichlet {
        operator: LinearOperator<Cochain, Cochain>,
        selection: Arc<CanonicalSelection>,
    },
    Harmonic {
        owner: Arc<crate::Geometry>,
        selection: Arc<CanonicalSelection>,
    },
    Hodge {
        owner: Arc<crate::Geometry>,
        degree: usize,
    },
    Parabolic {
        owner: Arc<crate::Geometry>,
        time_step_bits: u64,
    },
}

#[derive(Clone, Copy)]
pub(crate) enum SystemRef<'a> {
    PositiveSquare {
        metric: &'a crate::Metric,
    },
    Parabolic {
        metric: &'a crate::Metric,
        time_step: f64,
    },
}

impl<'a> SystemRef<'a> {
    const fn metric(self) -> &'a crate::Metric {
        match self {
            Self::PositiveSquare { metric } | Self::Parabolic { metric, .. } => metric,
        }
    }

    fn mass_and_stiffness_scale(self) -> Result<(Option<&'a [f64]>, f64), SolveError> {
        match self {
            Self::PositiveSquare { .. } => Ok((None, 1.0)),
            Self::Parabolic { metric, time_step } => Ok((
                Some(
                    metric
                        .hodge_coefficients_slice(0)
                        .map_err(|_| SolveError::Numerical)?,
                ),
                time_step,
            )),
        }
    }
}

/// Immutable reusable preparation, indexed by its mathematical problem family.
#[derive(Debug)]
pub struct Prepared<P: Problem> {
    key: ReuseKey,
    policy: Policy,
    factors: Factors,
    family: PhantomData<fn() -> P>,
}

/// Caller-owned solve scratch without duplicated requirement metadata.
pub struct Workspace {
    buffer: MemBuffer,
}

impl std::fmt::Debug for Workspace {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Workspace").finish_non_exhaustive()
    }
}

/// Receiver-led preparation workflow for one admitted problem family.
pub trait SolveExt: Problem + Sized {
    /// Prepare reusable RHS-independent physical work.
    ///
    /// # Errors
    /// Rejects exceeded resource limits or a failed positive factorization.
    fn prepare(&self, policy: Policy) -> Result<Prepared<Self>, SolveError> {
        self.prepare_cancellable(policy, &CancellationToken::new())
    }

    /// Prepare with cooperative checks around an uninterruptible factor call.
    ///
    /// # Errors
    /// Also rejects cancellation before preparation publication.
    fn prepare_cancellable(
        &self,
        policy: Policy,
        cancellation: &CancellationToken,
    ) -> Result<Prepared<Self>, SolveError>;
}

impl SolveExt for PoissonProblem {
    fn prepare_cancellable(
        &self,
        policy: Policy,
        cancellation: &CancellationToken,
    ) -> Result<Prepared<Self>, SolveError> {
        Prepared::mean_zero(self, policy, cancellation)
    }
}

impl Prepared<PoissonProblem> {
    fn mean_zero(
        problem: &PoissonProblem,
        policy: Policy,
        cancellation: &CancellationToken,
    ) -> Result<Self, SolveError> {
        check_cancelled(cancellation)?;
        let executor = policy.executor();
        let metric = problem.metric();
        let owner = metric.realization();
        let reduced = metric
            .hodge_coefficients_slice(0)
            .map_err(|_| SolveError::Numerical)?
            .len()
            .saturating_sub(1);
        // A dense-fill envelope is deliberately charged before faer's
        // symbolic call; it is conservative without claiming process RSS.
        let bytes = matrix_bytes(reduced)?;
        require_storage(policy.storage(), bytes, bytes.saturating_mul(2))?;
        require_work(policy.work(), cubic_work(reduced)?)?;
        let factor = if reduced == 0 {
            Factor::Analytic
        } else {
            let free = (1..=reduced).collect::<Vec<_>>();
            factor_stiffness(
                SystemRef::PositiveSquare {
                    metric: problem.metric(),
                },
                &free,
                executor,
                cancellation,
            )?
        };
        check_cancelled(cancellation)?;
        Ok(Self {
            key: ReuseKey::MeanZero {
                owner: Arc::clone(owner),
            },
            policy,
            factors: Factors::One(factor),
            family: PhantomData,
        })
    }

    /// Allocate workspace for a compatible RHS without touching factors.
    ///
    /// # Errors
    /// Rejects a foreign problem or insufficient logical storage.
    pub fn workspace_for(&self, problem: &PoissonProblem) -> Result<Workspace, SolveError> {
        self.require_problem(problem)?;
        let requirement = self.solve_requirement(problem)?;
        let bytes =
            u64::try_from(requirement.size_bytes()).map_err(|_| SolveError::ResourceLimit)?;
        require_storage(self.policy.storage(), 0, bytes)?;
        let buffer = MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?;
        Ok(Workspace { buffer })
    }

    /// Solve and certify one compatible RHS.
    ///
    /// # Errors
    /// Rejects mismatch, resource exhaustion, or failed numerical certification.
    pub fn solve(
        &self,
        problem: &PoissonProblem,
        workspace: &mut Workspace,
    ) -> Result<PoissonResult, SolveError> {
        self.solve_cancellable(problem, workspace, &CancellationToken::new())
    }

    /// Solve with cooperative checks around the backend factor call.
    ///
    /// # Errors
    /// Also rejects cancellation before result publication.
    pub fn solve_cancellable(
        &self,
        problem: &PoissonProblem,
        workspace: &mut Workspace,
        cancellation: &CancellationToken,
    ) -> Result<PoissonResult, SolveError> {
        self.require_problem(problem)?;
        if cancellation.is_cancelled() {
            return Err(SolveError::Cancelled);
        }
        let requirement = self.solve_requirement(problem)?;
        let stack = MemStack::new(&mut workspace.buffer);
        if !stack.can_hold(requirement) {
            return Err(SolveError::ResourceLimit);
        }
        let n = problem.len();
        require_work(self.policy.work(), solve_work(problem)?)?;
        let weights = problem
            .metric()
            .hodge_coefficients_slice(0)
            .map_err(|_| SolveError::Numerical)?;
        let factor = one_factor(&self.factors)?;
        let scale = factor_scale(factor);
        let reduced = n.saturating_sub(1);
        let (mut rhs_storage, stack) = stack.make_with(reduced, |_| 0.0_f64);
        let mut rhs = MatMut::from_column_major_slice_mut(&mut rhs_storage, reduced, 1);
        for index in 1..n {
            rhs[(index - 1, 0)] = problem
                .weak_value(index, weights)
                .ok_or(SolveError::Numerical)?
                / scale;
        }
        solve_factor(factor, rhs.as_mut(), self.policy.executor(), stack);
        if cancellation.is_cancelled() {
            return Err(SolveError::Cancelled);
        }
        let mut values = vec![0.0; n];
        for index in 1..n {
            values[index] = rhs[(index - 1, 0)];
        }
        let total_weight: f64 = weights.iter().sum();
        let mean = weights
            .iter()
            .zip(&values)
            .map(|(&w, &u)| w * u)
            .sum::<f64>()
            / total_weight;
        for value in &mut values {
            *value -= mean;
        }
        let potential_space = crate::Binary64Space::<Cochain>::full(
            Arc::clone(problem.metric().realization().topology()),
            0,
        )
        .map_err(|_| SolveError::Numerical)?;
        let potential =
            Binary64Element::admit(potential_space, values).map_err(|_| SolveError::Numerical)?;
        drop(rhs_storage);
        check_cancelled(cancellation)?;
        let solution = problem
            .certify(potential, MemStack::new(&mut workspace.buffer))
            .map_err(|_| SolveError::Numerical)?;
        if cancellation.is_cancelled() {
            return Err(SolveError::Cancelled);
        }
        Ok(solution)
    }

    fn require_problem(&self, problem: &PoissonProblem) -> Result<(), SolveError> {
        let ReuseKey::MeanZero { owner } = &self.key else {
            return Err(SolveError::ProblemMismatch);
        };
        Arc::ptr_eq(owner, problem.metric().realization())
            .then_some(())
            .ok_or(SolveError::ProblemMismatch)
    }

    fn solve_requirement(&self, problem: &PoissonProblem) -> Result<StackReq, SolveError> {
        let reduced = problem.len().saturating_sub(1);
        let backend =
            factor_solve_requirement(one_factor(&self.factors)?, self.policy.executor(), 1);
        let solve = StackReq::new::<f64>(reduced).and(backend);
        let certification = certification_requirement(problem)?;
        Ok(StackReq::any_of(&[solve, certification]))
    }
}

impl SolveExt for HeatProblem {
    fn prepare_cancellable(
        &self,
        policy: Policy,
        cancellation: &CancellationToken,
    ) -> Result<Prepared<Self>, SolveError> {
        check_cancelled(cancellation)?;
        let executor = policy.executor();
        let n = self.source().coefficients().len();
        let factor = if self.metric().realization().topology().dimension() == 0 {
            Factor::Analytic
        } else {
            let bytes = matrix_bytes(n)?;
            require_storage(policy.storage(), bytes, bytes.saturating_mul(2))?;
            require_work(policy.work(), cubic_work(n)?)?;
            let free = (0..n).collect::<Vec<_>>();
            factor_stiffness(
                SystemRef::Parabolic {
                    metric: self.metric(),
                    time_step: self.time_step(),
                },
                &free,
                executor,
                cancellation,
            )?
        };
        check_cancelled(cancellation)?;
        Ok(Prepared {
            key: ReuseKey::Parabolic {
                owner: Arc::clone(self.metric().realization()),
                time_step_bits: self.time_step().to_bits(),
            },
            policy,
            factors: Factors::One(factor),
            family: PhantomData,
        })
    }
}

impl Prepared<HeatProblem> {
    /// Allocate one scalar right-hand side and backend scratch.
    ///
    /// # Errors
    /// Rejects a mismatched problem or insufficient logical storage.
    pub fn workspace_for(&self, problem: &HeatProblem) -> Result<Workspace, SolveError> {
        self.require_problem(problem)?;
        let requirement = self.solve_requirement(problem)?;
        let bytes =
            u64::try_from(requirement.size_bytes()).map_err(|_| SolveError::ResourceLimit)?;
        require_storage(self.policy.storage(), 0, bytes)?;
        Ok(Workspace {
            buffer: MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?,
        })
    }

    /// Solve and certify one compatible scalar source.
    ///
    /// # Errors
    /// Rejects mismatch, resource exhaustion, cancellation, or failed certification.
    pub fn solve(
        &self,
        problem: &HeatProblem,
        workspace: &mut Workspace,
    ) -> Result<HeatResult, SolveError> {
        self.solve_cancellable(problem, workspace, &CancellationToken::new())
    }

    /// Solve with cooperative cancellation before result publication.
    ///
    /// # Errors
    /// Rejects mismatch, resource exhaustion, cancellation, or failed certification.
    pub fn solve_cancellable(
        &self,
        problem: &HeatProblem,
        workspace: &mut Workspace,
        cancellation: &CancellationToken,
    ) -> Result<HeatResult, SolveError> {
        self.require_problem(problem)?;
        check_cancelled(cancellation)?;
        let n = problem.source().coefficients().len();
        if problem.metric().realization().topology().dimension() == 0 {
            return Ok(HeatResult::new(
                problem.source().clone(),
                0.0,
                0.0,
                0.0,
                0.0,
                0,
            ));
        }
        let steps = n
            .checked_mul(n)
            .map(|value| value.max(n.saturating_mul(8)))
            .and_then(|value| u64::try_from(value).ok())
            .ok_or(SolveError::ResourceLimit)?;
        require_work(self.policy.work(), steps)?;
        let requirement = self.solve_requirement(problem)?;
        let stack = MemStack::new(&mut workspace.buffer);
        if !stack.can_hold(requirement) {
            return Err(SolveError::ResourceLimit);
        }
        let masses = problem
            .metric()
            .hodge_coefficients_slice(0)
            .map_err(|_| SolveError::Numerical)?;
        let mean = weighted_centroid(masses, problem.source().coefficients(), 1)?[0];
        let factor = one_factor(&self.factors)?;
        let scale = factor_scale(factor);
        let (mut rhs_storage, stack) = stack.make_with(n, |_| 0.0_f64);
        let mut rhs = MatMut::from_column_major_slice_mut(&mut rhs_storage, n, 1);
        fill_centered_mass_rhs(
            masses,
            problem.source().coefficients(),
            &[mean],
            scale,
            rhs.as_mut(),
        )?;
        solve_factor(factor, rhs.as_mut(), self.policy.executor(), stack);
        check_cancelled(cancellation)?;
        let values = rhs_storage
            .iter()
            .map(|value| value + mean)
            .collect::<Vec<_>>();
        let (_, endpoints) = stiffness_endpoints(problem.metric())?;
        let (residual_bound, mass_residual_bound, energy_before, energy_after, exact_fallback_rows) =
            heat_evidence(problem, &values, &endpoints)?;
        if !residual_bound.is_finite()
            || !mass_residual_bound.is_finite()
            || !energy_before.is_finite()
            || !energy_after.is_finite()
            || residual_bound > 1.0e-10
            || mass_residual_bound > 1.0e-12
            || energy_after > energy_before + 128.0 * f64::EPSILON * energy_before.max(1.0)
        {
            return Err(SolveError::Numerical);
        }
        let value = Binary64Element::admit(problem.source().space().clone(), values)
            .map_err(|_| SolveError::Numerical)?;
        check_cancelled(cancellation)?;
        Ok(HeatResult::new(
            value,
            residual_bound,
            mass_residual_bound,
            energy_before,
            energy_after,
            exact_fallback_rows,
        ))
    }

    fn require_problem(&self, problem: &HeatProblem) -> Result<(), SolveError> {
        let ReuseKey::Parabolic {
            owner,
            time_step_bits,
        } = &self.key
        else {
            return Err(SolveError::ProblemMismatch);
        };
        (Arc::ptr_eq(owner, problem.metric().realization())
            && *time_step_bits == problem.time_step().to_bits())
        .then_some(())
        .ok_or(SolveError::ProblemMismatch)
    }

    fn solve_requirement(&self, problem: &HeatProblem) -> Result<StackReq, SolveError> {
        if problem.metric().realization().topology().dimension() == 0 {
            return Ok(StackReq::EMPTY);
        }
        let n = problem.source().coefficients().len();
        Ok(StackReq::new::<f64>(n).and(factor_solve_requirement(
            one_factor(&self.factors)?,
            self.policy.executor(),
            1,
        )))
    }
}

pub(crate) fn logical_bytes(count: usize, width: usize) -> Result<u64, SolveError> {
    u64::try_from(count)
        .ok()
        .and_then(|value| value.checked_mul(u64::try_from(width).ok()?))
        .ok_or(SolveError::ResourceLimit)
}

pub(crate) fn logical_f64(count: usize) -> Result<u64, SolveError> {
    logical_bytes(count, size_of::<f64>())
}

pub(crate) fn checked_cells(left: usize, right: usize) -> Result<usize, SolveError> {
    left.checked_mul(right).ok_or(SolveError::ResourceLimit)
}

pub(crate) fn checked_sum<T: CheckedAdd + Zero, const N: usize>(
    parts: [T; N],
) -> Result<T, SolveError> {
    parts
        .into_iter()
        .try_fold(T::zero(), |sum, part| sum.checked_add(&part))
        .ok_or(SolveError::ResourceLimit)
}

pub(crate) fn sparse_phase_policy_bytes(
    rank: usize,
    input_entries: usize,
    value_cells: usize,
) -> Result<u64, SolveError> {
    // Deterministic logical admission, not a backend allocator/RSS upper bound:
    // input triplets and CSC, dense symbolic/numeric fill, and factor scratch.
    let fill = checked_cells(rank, rank)?;
    let columns = rank.checked_add(1).ok_or(SolveError::ResourceLimit)?;
    checked_sum([
        logical_f64(value_cells)?,
        logical_bytes(input_entries, size_of::<Triplet<usize, usize, f64>>())?,
        logical_bytes(input_entries, size_of::<(usize, f64)>())?,
        logical_bytes(columns, size_of::<usize>())?,
        logical_bytes(
            fill,
            size_of::<usize>() + size_of::<f64>() + size_of::<(usize, f64)>(),
        )?,
    ])
}

pub(crate) fn checked_work_product(left: usize, right: usize) -> Result<u64, SolveError> {
    u64::try_from(left)
        .ok()
        .and_then(|value| value.checked_mul(u64::try_from(right).ok()?))
        .ok_or(SolveError::ResourceLimit)
}

impl SolveExt for DirichletProblem {
    fn prepare_cancellable(
        &self,
        policy: Policy,
        cancellation: &CancellationToken,
    ) -> Result<Prepared<Self>, SolveError> {
        let executor = policy.executor();
        let selection = problem_selection(self.prescribed())?;
        let free = selection.complement().map_err(|_| SolveError::Numerical)?;
        let retained = matrix_bytes(free.len())?;
        require_storage(policy.storage(), retained, retained.saturating_mul(2))?;
        require_work(policy.work(), cubic_work(free.len())?)?;
        let factor = factor_general(self.operator(), free.indices(), executor, cancellation)?;
        Ok(Prepared {
            key: ReuseKey::Dirichlet {
                operator: self.operator().clone(),
                selection: Arc::clone(selection),
            },
            policy,
            factors: Factors::One(factor),
            family: PhantomData,
        })
    }
}

impl SolveExt for HarmonicExtension {
    fn prepare_cancellable(
        &self,
        policy: Policy,
        cancellation: &CancellationToken,
    ) -> Result<Prepared<Self>, SolveError> {
        let executor = policy.executor();
        let selection = problem_selection(self.prescribed())?;
        let free = selection.complement().map_err(|_| SolveError::Numerical)?;
        let retained = matrix_bytes(free.len())?;
        require_storage(policy.storage(), retained, retained.saturating_mul(2))?;
        require_work(policy.work(), cubic_work(free.len())?)?;
        let factor = if free.is_empty() {
            Factor::Analytic
        } else {
            factor_stiffness(
                SystemRef::PositiveSquare {
                    metric: self.metric(),
                },
                free.indices(),
                executor,
                cancellation,
            )?
        };
        Ok(Prepared {
            key: ReuseKey::Harmonic {
                owner: Arc::clone(self.metric().realization()),
                selection: Arc::clone(selection),
            },
            policy,
            factors: Factors::One(factor),
            family: PhantomData,
        })
    }
}

impl Prepared<DirichletProblem> {
    /// Allocate caller-owned scratch for a compatible equation.
    ///
    /// # Errors
    /// Rejects a mismatched problem or insufficient storage.
    pub fn workspace_for(&self, problem: &DirichletProblem) -> Result<Workspace, SolveError> {
        self.require_dirichlet(problem)?;
        boundary_workspace(
            one_factor(&self.factors)?,
            self.policy.executor(),
            problem.prescribed(),
            self.policy.storage(),
        )
    }

    /// Solve and certify a compatible equation.
    ///
    /// # Errors
    /// Rejects mismatch, resource exhaustion, or failed certification.
    pub fn solve(
        &self,
        problem: &DirichletProblem,
        workspace: &mut Workspace,
    ) -> Result<DirichletResult, SolveError> {
        self.solve_cancellable(problem, workspace, &CancellationToken::new())
    }

    /// Solve with cooperative checks around the backend factor call.
    ///
    /// # Errors
    /// Also rejects cancellation before result publication.
    pub fn solve_cancellable(
        &self,
        problem: &DirichletProblem,
        workspace: &mut Workspace,
        cancellation: &CancellationToken,
    ) -> Result<DirichletResult, SolveError> {
        self.require_dirichlet(problem)?;
        BoundaryEquation {
            operator: problem.operator(),
            rhs: Some(problem.rhs()),
            prescribed: problem.prescribed(),
            row_weights: None,
        }
        .solve(
            one_factor(&self.factors)?,
            self.policy.executor(),
            workspace,
            self.policy.work(),
            cancellation,
        )
    }

    fn require_dirichlet(&self, problem: &DirichletProblem) -> Result<(), SolveError> {
        let ReuseKey::Dirichlet {
            operator,
            selection,
        } = &self.key
        else {
            return Err(SolveError::ProblemMismatch);
        };
        let candidate = problem_selection(problem.prescribed())?;
        (operator.same_identity(problem.operator()) && Arc::ptr_eq(selection, candidate))
            .then_some(())
            .ok_or(SolveError::ProblemMismatch)
    }
}

impl Prepared<HarmonicExtension> {
    /// Allocate caller-owned scratch for compatible boundary data.
    ///
    /// # Errors
    /// Rejects a mismatched problem or insufficient storage.
    pub fn workspace_for(&self, problem: &HarmonicExtension) -> Result<Workspace, SolveError> {
        self.require_harmonic(problem)?;
        boundary_workspace(
            one_factor(&self.factors)?,
            self.policy.executor(),
            problem.prescribed(),
            self.policy.storage(),
        )
    }

    /// Solve and certify compatible boundary data.
    ///
    /// # Errors
    /// Rejects mismatch, resource exhaustion, or failed certification.
    pub fn solve(
        &self,
        problem: &HarmonicExtension,
        workspace: &mut Workspace,
    ) -> Result<DirichletResult, SolveError> {
        self.solve_cancellable(problem, workspace, &CancellationToken::new())
    }

    /// Solve with cooperative checks around the backend factor call.
    ///
    /// # Errors
    /// Also rejects cancellation before result publication.
    pub fn solve_cancellable(
        &self,
        problem: &HarmonicExtension,
        workspace: &mut Workspace,
        cancellation: &CancellationToken,
    ) -> Result<DirichletResult, SolveError> {
        self.require_harmonic(problem)?;
        let operator = problem
            .metric()
            .laplacian(0)
            .map_err(|_| SolveError::Numerical)?;
        let weights = problem
            .metric()
            .hodge_coefficients_slice(0)
            .map_err(|_| SolveError::Numerical)?;
        BoundaryEquation {
            operator: &operator,
            rhs: None,
            prescribed: problem.prescribed(),
            row_weights: Some(weights),
        }
        .solve(
            one_factor(&self.factors)?,
            self.policy.executor(),
            workspace,
            self.policy.work(),
            cancellation,
        )
    }

    fn require_harmonic(&self, problem: &HarmonicExtension) -> Result<(), SolveError> {
        let ReuseKey::Harmonic { owner, selection } = &self.key else {
            return Err(SolveError::ProblemMismatch);
        };
        let candidate = problem_selection(problem.prescribed())?;
        (Arc::ptr_eq(owner, problem.metric().realization()) && Arc::ptr_eq(selection, candidate))
            .then_some(())
            .ok_or(SolveError::ProblemMismatch)
    }
}

impl SolveExt for HodgeProblem {
    fn prepare_cancellable(
        &self,
        policy: Policy,
        cancellation: &CancellationToken,
    ) -> Result<Prepared<Self>, SolveError> {
        check_cancelled(cancellation)?;
        let executor = policy.executor();
        let degree = self.degree().map_err(|_| SolveError::Numerical)?;
        let topology = self.metric().realization().topology();
        let rows = self.source().coefficients().len();
        let dimension = preflight_hodge(self, degree, executor, policy.storage(), policy.work())?;
        let exact = if degree == 0 {
            None
        } else {
            let boundary = topology
                .chain_view()
                .boundary(degree)
                .map_err(|_| SolveError::Numerical)?;
            let pivots = independent_incidence(boundary, IncidenceAxis::Rows)
                .map_err(|_| SolveError::Allocation)?;
            Some((boundary, pivots))
        };
        let coexact = if degree == dimension {
            None
        } else {
            let boundary = topology
                .chain_view()
                .boundary(degree + 1)
                .map_err(|_| SolveError::Numerical)?;
            let pivots = independent_incidence(boundary, IncidenceAxis::Columns)
                .map_err(|_| SolveError::Allocation)?;
            Some((boundary, pivots))
        };
        let exact_rank = exact.as_ref().map_or(0, |(_, pivots)| pivots.len());
        let coexact_rank = coexact.as_ref().map_or(0, |(_, pivots)| pivots.len());
        let retained = hodge_factor_bytes(rows, exact_rank)?
            .checked_add(hodge_factor_bytes(rows, coexact_rank)?)
            .ok_or(SolveError::ResourceLimit)?;
        let factor_scratch = hodge_factor_scratch_bytes(rows, exact_rank, executor)?
            .max(hodge_factor_scratch_bytes(rows, coexact_rank, executor)?);
        let peak = retained
            .checked_add(factor_scratch)
            .ok_or(SolveError::ResourceLimit)?;
        require_storage(policy.storage(), retained, peak)?;
        let factor_work = hodge_factor_work(rows, exact_rank)?
            .checked_add(hodge_factor_work(rows, coexact_rank)?)
            .ok_or(SolveError::ResourceLimit)?;
        require_work(policy.work(), factor_work)?;
        let exact_factor = factor_hodge_image(
            self,
            exact
                .as_ref()
                .map(|(boundary, pivots)| (*boundary, pivots.as_ref())),
            true,
            executor,
            cancellation,
        )?;
        let coexact_factor = factor_hodge_image(
            self,
            coexact
                .as_ref()
                .map(|(boundary, pivots)| (*boundary, pivots.as_ref())),
            false,
            executor,
            cancellation,
        )?;
        check_cancelled(cancellation)?;
        Ok(Prepared {
            key: ReuseKey::Hodge {
                owner: Arc::clone(self.metric().realization()),
                degree,
            },
            policy,
            factors: Factors::Hodge([exact_factor, coexact_factor]),
            family: PhantomData,
        })
    }
}

fn preflight_hodge(
    problem: &HodgeProblem,
    degree: usize,
    executor: Executor,
    storage: StorageLimit,
    work: WorkLimit,
) -> Result<usize, SolveError> {
    let topology = problem.metric().realization().topology();
    let dimension = topology.dimension();
    let boundary_shape = |boundary_degree| -> Result<(usize, usize), SolveError> {
        let shape = topology
            .chain_view()
            .boundary(boundary_degree)
            .map_err(|_| SolveError::Numerical)?
            .shape();
        Ok(shape)
    };
    let exact_shape = if degree == 0 {
        None
    } else {
        Some(boundary_shape(degree)?)
    };
    let coexact_shape = if degree == dimension {
        None
    } else {
        Some(boundary_shape(degree + 1)?)
    };
    let exact = exact_shape.map_or(0, |shape| shape.0.min(shape.1));
    let coexact = coexact_shape.map_or(0, |shape| shape.0.min(shape.1));
    let rows = problem.source().coefficients().len();
    let retained = hodge_factor_bytes(rows, exact)?
        .checked_add(hodge_factor_bytes(rows, coexact)?)
        .ok_or(SolveError::ResourceLimit)?;
    let scratch = hodge_factor_scratch_bytes(rows, exact, executor)?
        .max(hodge_factor_scratch_bytes(rows, coexact, executor)?);
    let factor_peak = retained
        .checked_add(scratch)
        .ok_or(SolveError::ResourceLimit)?;
    let selection_peak = incidence_selection_bytes(exact_shape)?.max(
        incidence_selection_bytes(coexact_shape)?
            .checked_add(
                u64::try_from(exact)
                    .ok()
                    .and_then(|rank| rank.checked_mul(u64::try_from(size_of::<usize>()).ok()?))
                    .ok_or(SolveError::ResourceLimit)?,
            )
            .ok_or(SolveError::ResourceLimit)?,
    );
    require_storage(storage, retained, factor_peak.max(selection_peak))?;
    let required_work = hodge_factor_work(rows, exact)?
        .checked_add(hodge_factor_work(rows, coexact)?)
        .and_then(|value| value.checked_add(incidence_selection_work(exact_shape).ok()?))
        .and_then(|value| value.checked_add(incidence_selection_work(coexact_shape).ok()?))
        .ok_or(SolveError::ResourceLimit)?;
    require_work(work, required_work)?;
    Ok(dimension)
}

fn incidence_selection_bytes(shape: Option<(usize, usize)>) -> Result<u64, SolveError> {
    let Some((rows, columns)) = shape else {
        return Ok(0);
    };
    let cells = rows.checked_mul(columns).ok_or(SolveError::ResourceLimit)?;
    let pivots = rows.min(columns);
    let bit_width = usize::try_from(usize::BITS - pivots.max(1).leading_zeros())
        .map_err(|_| SolveError::ResourceLimit)?;
    let determinant_bytes = pivots
        .checked_mul(bit_width)
        .and_then(|bits| bits.checked_add(8))
        .map(|bits| bits / 8)
        .ok_or(SolveError::ResourceLimit)?;
    let rational_bytes = size_of::<crate::ExactRational>()
        .checked_add(
            determinant_bytes
                .checked_mul(2)
                .ok_or(SolveError::ResourceLimit)?,
        )
        .ok_or(SolveError::ResourceLimit)?;
    cells
        .checked_mul(rational_bytes)
        .and_then(|bytes| bytes.checked_add(pivots.checked_mul(size_of::<usize>())?))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(SolveError::ResourceLimit)
}

fn incidence_selection_work(shape: Option<(usize, usize)>) -> Result<u64, SolveError> {
    let Some((rows, columns)) = shape else {
        return Ok(0);
    };
    rows.checked_mul(columns)
        .and_then(|value| value.checked_mul(rows.min(columns)))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(SolveError::ResourceLimit)
}

impl Prepared<HodgeProblem> {
    /// Allocate one workspace reused sequentially by both projections.
    ///
    /// # Errors
    ///
    /// Rejects a mismatched problem, insufficient storage, or allocation failure.
    pub fn workspace_for(&self, problem: &HodgeProblem) -> Result<Workspace, SolveError> {
        self.require_hodge(problem)?;
        let requirement = hodge_projection_requirement(&self.factors, self.policy.executor())?;
        let bytes =
            u64::try_from(requirement.size_bytes()).map_err(|_| SolveError::ResourceLimit)?;
        require_storage(self.policy.storage(), 0, bytes)?;
        let buffer = MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?;
        Ok(Workspace { buffer })
    }

    /// Project, reconstruct, and certify one compatible source cochain.
    ///
    /// # Errors
    ///
    /// Rejects mismatched inputs, insufficient work, or failed numerical certification.
    pub fn solve(
        &self,
        problem: &HodgeProblem,
        workspace: &mut Workspace,
    ) -> Result<HodgeDecomposition, SolveError> {
        self.solve_cancellable(problem, workspace, &CancellationToken::new())
    }

    /// Solve with cooperative checks around the two sequential projections.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::solve`], plus cancellation.
    pub fn solve_cancellable(
        &self,
        problem: &HodgeProblem,
        workspace: &mut Workspace,
        cancellation: &CancellationToken,
    ) -> Result<HodgeDecomposition, SolveError> {
        self.require_hodge(problem)?;
        check_cancelled(cancellation)?;
        let requirement = hodge_projection_requirement(&self.factors, self.policy.executor())?;
        if !MemStack::new(&mut workspace.buffer).can_hold(requirement) {
            return Err(SolveError::ResourceLimit);
        }
        let rows = problem.source().coefficients().len();
        let ranks = [qr_rank(&self.factors[0])?, qr_rank(&self.factors[1])?];
        let projection_work = ranks.into_iter().try_fold(0_u64, |total, rank| {
            total
                .checked_add(hodge_projection_work(rows, rank)?)
                .ok_or(SolveError::ResourceLimit)
        })?;
        let required = projection_work
            .checked_add(hodge_certification_work(problem)?)
            .ok_or(SolveError::ResourceLimit)?;
        require_work(self.policy.work(), required)?;
        let weights = problem
            .metric()
            .hodge_coefficients_slice(problem.degree().map_err(|_| SolveError::Numerical)?)
            .map_err(|_| SolveError::Numerical)?;
        let exact_values = project_hodge(
            &self.factors[0],
            problem.source().coefficients(),
            weights,
            self.policy.executor(),
            workspace,
        )?;
        check_cancelled(cancellation)?;
        let coexact_values = project_hodge(
            &self.factors[1],
            problem.source().coefficients(),
            weights,
            self.policy.executor(),
            workspace,
        )?;
        check_cancelled(cancellation)?;
        let harmonic_values = problem
            .source()
            .coefficients()
            .iter()
            .zip(&exact_values)
            .zip(&coexact_values)
            .map(|((&source, &exact), &coexact)| source - exact - coexact)
            .collect::<Vec<_>>();
        let space = problem.source().space().clone();
        let exact = Binary64Element::admit(space.clone(), exact_values)
            .map_err(|_| SolveError::Numerical)?;
        let coexact = Binary64Element::admit(space.clone(), coexact_values)
            .map_err(|_| SolveError::Numerical)?;
        let harmonic =
            Binary64Element::admit(space, harmonic_values).map_err(|_| SolveError::Numerical)?;
        let conditions = [
            qr_condition(&self.factors[0])?,
            qr_condition(&self.factors[1])?,
        ];
        check_cancelled(cancellation)?;
        problem
            .certify(exact, coexact, harmonic, ranks, conditions)
            .map_err(|_| SolveError::Numerical)
    }

    fn require_hodge(&self, problem: &HodgeProblem) -> Result<(), SolveError> {
        let ReuseKey::Hodge { owner, degree } = &self.key else {
            return Err(SolveError::ProblemMismatch);
        };
        (Arc::ptr_eq(owner, problem.metric().realization())
            && *degree == problem.degree().map_err(|_| SolveError::ProblemMismatch)?)
        .then_some(())
        .ok_or(SolveError::ProblemMismatch)
    }
}

fn factor_hodge_image(
    problem: &HodgeProblem,
    incidence: Option<(crate::BoundaryRef<'_>, &[usize])>,
    exact: bool,
    executor: Executor,
    cancellation: &CancellationToken,
) -> Result<Factor, SolveError> {
    let rows = problem.source().coefficients().len();
    let rank = incidence.map_or(0, |(_, pivots)| pivots.len());
    if rank == 0 {
        return Ok(Factor::DenseQr {
            vectors: Mat::zeros(rows, 0),
            householder: Mat::zeros(1, 0),
            rank: 0,
            condition_indicator: 0.0,
        });
    }
    let degree = problem.degree().map_err(|_| SolveError::Numerical)?;
    let weights = problem
        .metric()
        .hodge_coefficients_slice(degree)
        .map_err(|_| SolveError::Numerical)?;
    let mut basis = Mat::zeros(rows, rank);
    let (boundary, pivots) = incidence.ok_or(SolveError::Numerical)?;
    let positions = selected_positions(
        if exact {
            boundary.shape().0
        } else {
            boundary.shape().1
        },
        pivots,
    )?;
    let upper_weights = if exact {
        None
    } else {
        Some(
            problem
                .metric()
                .hodge_coefficients_slice(degree + 1)
                .map_err(|_| SolveError::Numerical)?,
        )
    };
    for (row, column, value) in boundary.exact_entries() {
        let selected = if exact {
            positions[row]
        } else {
            positions[column]
        };
        let Some(selected) = selected else { continue };
        let coefficient = if exact {
            value.to_f64().ok_or(SolveError::Numerical)? * weights[column].sqrt()
        } else {
            value.to_f64().ok_or(SolveError::Numerical)?
                * upper_weights.ok_or(SolveError::Numerical)?[column]
                / weights[row].sqrt()
        };
        let output_row = if exact { column } else { row };
        basis[(output_row, selected)] = coefficient;
    }
    if basis
        .as_ref()
        .col_iter()
        .flat_map(faer::ColRef::iter)
        .any(|value| !value.is_finite())
    {
        return Err(SolveError::Numerical);
    }
    let block = qr_factor::recommended_block_size::<f64>(rows, rank).max(1);
    let mut householder = Mat::zeros(block, rank);
    let mut permutation = vec![0_usize; rank];
    let mut inverse = vec![0_usize; rank];
    let requirement = qr_factor::qr_in_place_scratch::<usize, f64>(
        rows,
        rank,
        block,
        executor.par(),
        Spec::default(),
    );
    let mut memory = MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?;
    check_cancelled(cancellation)?;
    qr_factor::qr_in_place(
        basis.as_mut(),
        householder.as_mut(),
        &mut permutation,
        &mut inverse,
        executor.par(),
        MemStack::new(&mut memory),
        Spec::default(),
    );
    check_cancelled(cancellation)?;
    let diagonal = (0..rank).map(|index| basis[(index, index)].abs());
    let maximum = diagonal.clone().fold(0.0_f64, f64::max);
    let minimum = diagonal.fold(f64::INFINITY, f64::min);
    if !maximum.is_finite()
        || !minimum.is_finite()
        || maximum == 0.0
        || minimum <= f64::EPSILON.sqrt() * maximum
    {
        return Err(SolveError::Factorization);
    }
    Ok(Factor::DenseQr {
        vectors: basis,
        householder,
        rank,
        condition_indicator: maximum / minimum,
    })
}

fn project_hodge(
    factor: &Factor,
    source: &[f64],
    weights: &[f64],
    executor: Executor,
    workspace: &mut Workspace,
) -> Result<Vec<f64>, SolveError> {
    let Factor::DenseQr {
        vectors,
        householder,
        rank,
        ..
    } = factor
    else {
        return Err(SolveError::ProblemMismatch);
    };
    if *rank == 0 {
        return Ok(vec![0.0; source.len()]);
    }
    let mut values = source
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| value * weight.sqrt())
        .collect::<Vec<_>>();
    let mut rhs = MatMut::from_column_major_slice_mut(&mut values, source.len(), 1);
    householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
        vectors.as_ref(),
        householder.as_ref(),
        Conj::Yes,
        rhs.as_mut(),
        executor.par(),
        MemStack::new(&mut workspace.buffer),
    );
    for row in *rank..source.len() {
        rhs[(row, 0)] = 0.0;
    }
    householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
        vectors.as_ref(),
        householder.as_ref(),
        Conj::Yes,
        rhs.as_mut(),
        executor.par(),
        MemStack::new(&mut workspace.buffer),
    );
    for (value, weight) in values.iter_mut().zip(weights) {
        *value /= weight.sqrt();
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SolveError::Numerical);
    }
    Ok(values)
}

fn factor_general(
    operator: &LinearOperator<Cochain, Cochain>,
    free: &[usize],
    executor: Executor,
    cancellation: &CancellationToken,
) -> Result<Factor, SolveError> {
    check_cancelled(cancellation)?;
    if free.is_empty() {
        return Ok(Factor::Analytic);
    }
    let n = operator.source().size();
    let mut matrix = Mat::<f64>::zeros(free.len(), free.len());
    let mut input = vec![0.0; n];
    for (column, &source) in free.iter().enumerate() {
        input[source] = 1.0;
        let output = operator
            .apply_coefficients(&input)
            .map_err(|_| SolveError::Numerical)?;
        input[source] = 0.0;
        for (row, &target) in free.iter().enumerate() {
            matrix[(row, column)] = output[target];
        }
    }
    factor_dense_square(matrix, executor, cancellation)
}

pub(crate) fn factor_dense_square(
    mut matrix: Mat<f64>,
    executor: Executor,
    cancellation: &CancellationToken,
) -> Result<Factor, SolveError> {
    if matrix.nrows() == 0 || matrix.nrows() != matrix.ncols() {
        return Err(SolveError::Factorization);
    }
    let mut scale = 0.0_f64;
    for column in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            scale = scale.max(matrix[(row, column)].abs());
        }
    }
    if scale == 0.0 || !scale.is_finite() {
        return Err(SolveError::Factorization);
    }
    for column in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            matrix[(row, column)] /= scale;
        }
    }
    let rank = matrix.nrows();
    let mut permutation = vec![0_usize; rank];
    let mut inverse_permutation = vec![0_usize; rank];
    let requirement =
        lu_factor::lu_in_place_scratch::<usize, f64>(rank, rank, executor.par(), Spec::default());
    let mut buffer = MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?;
    lu_factor::lu_in_place(
        matrix.as_mut(),
        &mut permutation,
        &mut inverse_permutation,
        executor.par(),
        MemStack::new(&mut buffer),
        Spec::default(),
    );
    if (0..rank).any(|index| {
        let pivot = matrix[(index, index)];
        pivot == 0.0 || !pivot.is_finite()
    }) {
        return Err(SolveError::Factorization);
    }
    check_cancelled(cancellation)?;
    Ok(Factor::DenseLu {
        factor: matrix,
        permutation: permutation.into_boxed_slice(),
        inverse_permutation: inverse_permutation.into_boxed_slice(),
        scale,
    })
}

pub(crate) fn require_stable_dense_lu(factor: &Factor) -> Result<(), SolveError> {
    let Factor::DenseLu { factor, .. } = factor else {
        return Err(SolveError::ProblemMismatch);
    };
    let diagonal = (0..factor.nrows()).map(|index| factor[(index, index)].abs());
    let maximum = diagonal.clone().fold(0.0_f64, f64::max);
    let minimum = diagonal.fold(f64::INFINITY, f64::min);
    if !maximum.is_finite()
        || !minimum.is_finite()
        || maximum == 0.0
        || minimum <= f64::EPSILON.sqrt() * maximum
    {
        Err(SolveError::Factorization)
    } else {
        Ok(())
    }
}

fn boundary_workspace(
    factor: &Factor,
    executor: Executor,
    prescribed: &Binary64Cochain,
    storage: StorageLimit,
) -> Result<Workspace, SolveError> {
    let requirement = boundary_requirement(factor, executor, prescribed)?;
    let bytes = u64::try_from(requirement.size_bytes()).map_err(|_| SolveError::ResourceLimit)?;
    require_storage(storage, 0, bytes)?;
    Ok(Workspace {
        buffer: MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?,
    })
}

fn boundary_requirement(
    factor: &Factor,
    executor: Executor,
    prescribed: &Binary64Cochain,
) -> Result<StackReq, SolveError> {
    let selection = problem_selection(prescribed)?;
    let n = selection
        .owner()
        .basis(selection.degree())
        .map_err(|_| SolveError::Numerical)?
        .row_count();
    let free = n.saturating_sub(selection.len());
    Ok(StackReq::all_of(&[
        StackReq::new::<f64>(free),
        StackReq::new::<f64>(free.saturating_mul(selection.len())),
        StackReq::new::<f64>(n),
        factor_solve_requirement(factor, executor, 1),
    ]))
}

struct BoundaryEquation<'a> {
    operator: &'a LinearOperator<Cochain, Cochain>,
    rhs: Option<&'a Binary64Cochain>,
    prescribed: &'a Binary64Cochain,
    row_weights: Option<&'a [f64]>,
}

impl BoundaryEquation<'_> {
    fn solve(
        &self,
        factor: &Factor,
        executor: Executor,
        workspace: &mut Workspace,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<DirichletResult, SolveError> {
        check_cancelled(cancellation)?;
        let selection = problem_selection(self.prescribed)?;
        let free = selection.complement().map_err(|_| SolveError::Numerical)?;
        let n = self.operator.source().size();
        let required = n
            .checked_mul(n)
            .and_then(|value| value.checked_mul(selection.len().max(1)))
            .and_then(|value| value.checked_add(free.len().saturating_mul(free.len())))
            .ok_or(SolveError::ResourceLimit)?;
        require_work(
            work,
            u64::try_from(required).map_err(|_| SolveError::ResourceLimit)?,
        )?;
        let mut values = vec![0.0; n];
        for (&index, &value) in selection
            .indices()
            .iter()
            .zip(self.prescribed.coefficients())
        {
            values[index] = value;
        }
        let requirement = boundary_requirement(factor, executor, self.prescribed)?;
        let stack = MemStack::new(&mut workspace.buffer);
        if !stack.can_hold(requirement) {
            return Err(SolveError::ResourceLimit);
        }
        let (mut coupling, stack) =
            stack.make_with(free.len().saturating_mul(selection.len()), |_| 0.0_f64);
        let (mut basis, stack) = stack.make_with(n, |_| 0.0_f64);
        let (mut rhs_storage, stack) = stack.make_with(free.len(), |_| 0.0_f64);
        let exact_fallback_rows = exact_reduced_rhs(
            self.operator,
            selection,
            self.prescribed.coefficients(),
            free.indices(),
            self.rhs.map(Binary64Element::coefficients),
            self.row_weights,
            BoundaryActionScratch {
                coupling: &mut coupling,
                basis: &mut basis,
                action: &mut rhs_storage,
            },
        )?;
        let mut reduced_rhs = MatMut::from_column_major_slice_mut(&mut rhs_storage, free.len(), 1);
        let scale = factor_scale(factor);
        for row in 0..free.len() {
            reduced_rhs[(row, 0)] /= scale;
        }
        solve_factor(factor, reduced_rhs.as_mut(), executor, stack);
        check_cancelled(cancellation)?;
        for (row, &index) in free.indices().iter().enumerate() {
            values[index] = reduced_rhs[(row, 0)];
        }
        drop(rhs_storage);
        let value = Binary64Element::admit(self.operator.source().clone(), values)
            .map_err(|_| SolveError::Numerical)?;
        let residual_bound = certify_boundary(
            self.operator,
            self.rhs,
            self.prescribed,
            &value,
            free.indices(),
        )?;
        check_cancelled(cancellation)?;
        Ok(DirichletResult::new(
            value,
            DirichletEvidence::new(residual_bound, exact_fallback_rows),
        ))
    }
}

fn certify_boundary(
    operator: &LinearOperator<Cochain, Cochain>,
    rhs: Option<&Binary64Cochain>,
    prescribed: &Binary64Cochain,
    value: &Binary64Cochain,
    free: &[usize],
) -> Result<f64, SolveError> {
    let selection = problem_selection(prescribed)?;
    if selection
        .indices()
        .iter()
        .zip(prescribed.coefficients())
        .any(|(&index, &expected)| value.coefficients()[index].to_bits() != expected.to_bits())
    {
        return Err(SolveError::Numerical);
    }
    let action = operator.apply(value).map_err(|_| SolveError::Numerical)?;
    let mut residual_bound = 0.0_f64;
    for &index in free {
        let expected = rhs.map_or(0.0, |rhs| rhs.coefficients()[index]);
        let scale = action.coefficients()[index]
            .abs()
            .max(expected.abs())
            .max(1.0);
        let residual = (action.coefficients()[index] - expected).abs();
        if residual > 512.0 * f64::EPSILON * scale {
            return Err(SolveError::Numerical);
        }
        residual_bound = residual_bound.max(residual);
    }
    Ok(residual_bound)
}

fn exact_reduced_rhs(
    operator: &LinearOperator<Cochain, Cochain>,
    boundary: &CanonicalSelection,
    values: &[f64],
    free: &[usize],
    rhs: Option<&[f64]>,
    row_weights: Option<&[f64]>,
    scratch: BoundaryActionScratch<'_>,
) -> Result<usize, SolveError> {
    let BoundaryActionScratch {
        coupling,
        basis,
        action,
    } = scratch;
    let rows = free.len();
    let columns = boundary.len();
    for (column, &index) in boundary.indices().iter().enumerate() {
        basis[index] = 1.0;
        let output = operator
            .apply_coefficients(basis)
            .map_err(|_| SolveError::Numerical)?;
        basis[index] = 0.0;
        for (row, &index) in free.iter().enumerate() {
            coupling[column * rows + row] =
                output[index] * row_weights.map_or(1.0, |weights| weights[index]);
        }
    }
    let mut fallbacks = 0;
    for row in 0..rows {
        let index = free[row];
        let target = rhs.map_or(0.0, |rhs| rhs[index]);
        let weight = row_weights.map_or(1.0, |weights| weights[index]);
        let terms = std::iter::once((weight, target))
            .chain((0..columns).map(|column| (-coupling[column * rows + row], values[column])));
        let (value, exact) = adaptive_product_value(terms).ok_or(SolveError::Numerical)?;
        action[row] = value;
        fallbacks += usize::from(exact);
    }
    Ok(fallbacks)
}

struct BoundaryActionScratch<'a> {
    coupling: &'a mut [f64],
    basis: &'a mut [f64],
    action: &'a mut [f64],
}

fn problem_selection(value: &Binary64Cochain) -> Result<&Arc<CanonicalSelection>, SolveError> {
    value
        .space()
        .canonical_selection()
        .ok_or(SolveError::ProblemMismatch)
}

pub(crate) fn factor_stiffness(
    system: SystemRef<'_>,
    free: &[usize],
    executor: Executor,
    cancellation: &CancellationToken,
) -> Result<Factor, SolveError> {
    check_cancelled(cancellation)?;
    let metric = system.metric();
    let (n, endpoints) = stiffness_endpoints(metric)?;
    let weights = metric
        .hodge_coefficients_slice(1)
        .map_err(|_| SolveError::Numerical)?;
    let (masses, stiffness_multiplier) = system.mass_and_stiffness_scale()?;
    let scale = stiffness_scale(n, weights, &endpoints, masses, stiffness_multiplier);
    if !scale.is_finite() || scale == 0.0 {
        return Err(SolveError::Factorization);
    }
    if free.len() > 64 {
        return factor_sparse(system, free, &endpoints, scale, executor, cancellation);
    }
    let positions = selected_positions(n, free)?;
    let mut matrix = Mat::<f64>::zeros(free.len(), free.len());
    for (edge, entries) in endpoints.iter().enumerate() {
        for &(row, row_sign) in entries {
            for &(column, column_sign) in entries {
                let (Some(row), Some(column)) = (positions[row], positions[column]) else {
                    continue;
                };
                let sign = if row_sign == column_sign { 1.0 } else { -1.0 };
                matrix[(row, column)] += stiffness_multiplier * weights[edge] * sign;
            }
        }
    }
    if let Some(masses) = masses {
        for (diagonal, &vertex) in free.iter().enumerate() {
            matrix[(diagonal, diagonal)] += masses[vertex];
        }
    }
    for column in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            matrix[(row, column)] /= scale;
        }
    }
    let diagonal = (0..matrix.nrows())
        .all(|column| (0..matrix.nrows()).all(|row| row == column || matrix[(row, column)] == 0.0));
    if diagonal {
        let inverse = (0..matrix.nrows())
            .map(|i| 1.0 / matrix[(i, i)])
            .collect::<Vec<_>>();
        if inverse.iter().any(|value| !value.is_finite()) {
            return Err(SolveError::Factorization);
        }
        return Ok(Factor::Diagonal {
            inverse: inverse.into_boxed_slice(),
            scale,
        });
    }
    let requirement = llt::factor::cholesky_in_place_scratch::<f64>(
        matrix.nrows(),
        executor.par(),
        Spec::default(),
    );
    check_cancelled(cancellation)?;
    let mut buffer = MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?;
    llt::factor::cholesky_in_place(
        matrix.as_mut(),
        LltRegularization::default(),
        executor.par(),
        MemStack::new(&mut buffer),
        Spec::default(),
    )
    .map_err(|_| SolveError::Factorization)?;
    if (0..matrix.ncols())
        .any(|column| (column..matrix.nrows()).any(|row| !matrix[(row, column)].is_finite()))
    {
        return Err(SolveError::Factorization);
    }
    check_cancelled(cancellation)?;
    Ok(Factor::DenseLlt {
        factor: matrix,
        scale,
    })
}

pub(crate) fn stiffness_endpoints(
    metric: &crate::Metric,
) -> Result<(usize, Vec<EdgeEndpoints>), SolveError> {
    let boundary = metric
        .realization()
        .topology()
        .chain_view()
        .boundary(1)
        .map_err(|_| SolveError::Numerical)?;
    let n = boundary.shape().0;
    let edge_count = boundary.shape().1;
    let mut endpoints = Vec::new();
    endpoints
        .try_reserve_exact(edge_count)
        .map_err(|_| SolveError::Allocation)?;
    endpoints.resize(edge_count, [(usize::MAX, 0_i64); 2]);
    let mut counts = Vec::new();
    counts
        .try_reserve_exact(edge_count)
        .map_err(|_| SolveError::Allocation)?;
    counts.resize(edge_count, 0_usize);
    for (row, column, value) in boundary.exact_entries() {
        let slot = counts[column];
        if slot == 2 {
            return Err(SolveError::Numerical);
        }
        endpoints[column][slot] = (row, value);
        counts[column] += 1;
    }
    if counts.iter().any(|&count| count != 2) {
        return Err(SolveError::Numerical);
    }
    Ok((n, endpoints))
}

fn stiffness_scale(
    n: usize,
    weights: &[f64],
    endpoints: &[[(usize, i64); 2]],
    masses: Option<&[f64]>,
    stiffness_multiplier: f64,
) -> f64 {
    let mut diagonal = masses.map_or_else(|| vec![0.0_f64; n], <[f64]>::to_vec);
    for (&weight, edge) in weights.iter().zip(endpoints) {
        diagonal[edge[0].0] += stiffness_multiplier * weight;
        diagonal[edge[1].0] += stiffness_multiplier * weight;
    }
    diagonal.into_iter().fold(0.0_f64, f64::max)
}

fn factor_sparse(
    system: SystemRef<'_>,
    free: &[usize],
    endpoints: &[EdgeEndpoints],
    scale: f64,
    executor: Executor,
    cancellation: &CancellationToken,
) -> Result<Factor, SolveError> {
    check_cancelled(cancellation)?;
    let metric = system.metric();
    let n = metric.realization().topology().vertex_count();
    let weights = metric
        .hodge_coefficients_slice(1)
        .map_err(|_| SolveError::Numerical)?;
    let (masses, stiffness_multiplier) = system.mass_and_stiffness_scale()?;
    let positions = selected_positions(n, free)?;
    let mut triplets = Vec::new();
    triplets
        .try_reserve_exact(endpoints.len().saturating_mul(4))
        .map_err(|_| SolveError::Allocation)?;
    for (edge, entries) in endpoints.iter().enumerate() {
        for &(row, row_sign) in entries {
            for &(column, column_sign) in entries {
                let (Some(row), Some(column)) = (positions[row], positions[column]) else {
                    continue;
                };
                let sign = if row_sign == column_sign { 1.0 } else { -1.0 };
                triplets.push(Triplet::new(
                    row,
                    column,
                    stiffness_multiplier * weights[edge] * sign / scale,
                ));
            }
        }
    }
    if let Some(masses) = masses {
        for (diagonal, &vertex) in free.iter().enumerate() {
            triplets.push(Triplet::new(diagonal, diagonal, masses[vertex] / scale));
        }
    }
    factor_sparse_triplets(free.len(), &triplets, scale, executor, cancellation)
}

pub(crate) fn factor_sparse_triplets(
    rank: usize,
    triplets: &[Triplet<usize, usize, f64>],
    scale: f64,
    executor: Executor,
    cancellation: &CancellationToken,
) -> Result<Factor, SolveError> {
    if rank == 0 {
        return Ok(Factor::Analytic);
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(SolveError::Factorization);
    }
    let matrix =
        SparseColMat::try_new_from_triplets(rank, rank, triplets).map_err(|error| match error {
            CreationError::Generic(FaerError::OutOfMemory) => SolveError::Allocation,
            CreationError::Generic(FaerError::IndexOverflow)
            | CreationError::OutOfBounds { .. } => SolveError::ResourceLimit,
            CreationError::Generic(_) => SolveError::Numerical,
        })?;
    let symbolic = factorize_symbolic_cholesky(
        matrix.symbolic(),
        Side::Lower,
        SymmetricOrdering::default(),
        CholeskySymbolicParams::default(),
    )
    .map_err(|error| match error {
        FaerError::OutOfMemory => SolveError::Allocation,
        FaerError::IndexOverflow => SolveError::ResourceLimit,
        _ => SolveError::Numerical,
    })?;
    let mut numeric = Vec::new();
    numeric
        .try_reserve_exact(symbolic.len_val())
        .map_err(|_| SolveError::Allocation)?;
    numeric.resize(symbolic.len_val(), 0.0_f64);
    let requirement =
        symbolic.factorize_numeric_llt_scratch::<f64>(executor.par(), Spec::default());
    let mut buffer = MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?;
    check_cancelled(cancellation)?;
    symbolic
        .factorize_numeric_llt(
            &mut numeric,
            matrix.as_ref(),
            Side::Lower,
            LltRegularization::default(),
            executor.par(),
            MemStack::new(&mut buffer),
            Spec::default(),
        )
        .map_err(|_| SolveError::Factorization)?;
    if numeric.iter().any(|value| !value.is_finite()) {
        return Err(SolveError::Factorization);
    }
    check_cancelled(cancellation)?;
    Ok(Factor::SparseLlt {
        symbolic: Box::new(symbolic),
        numeric: numeric.into_boxed_slice(),
        scale,
    })
}

fn selected_positions(n: usize, selected: &[usize]) -> Result<Vec<Option<usize>>, SolveError> {
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(n)
        .map_err(|_| SolveError::Allocation)?;
    positions.resize(n, None);
    for (position, &index) in selected.iter().enumerate() {
        let slot = positions.get_mut(index).ok_or(SolveError::Numerical)?;
        if slot.replace(position).is_some() {
            return Err(SolveError::Numerical);
        }
    }
    Ok(positions)
}

fn one_factor(factors: &[Factor]) -> Result<&Factor, SolveError> {
    let [factor] = factors else {
        return Err(SolveError::ProblemMismatch);
    };
    Ok(factor)
}

fn qr_rank(factor: &Factor) -> Result<usize, SolveError> {
    match factor {
        Factor::DenseQr { rank, .. } => Ok(*rank),
        _ => Err(SolveError::ProblemMismatch),
    }
}

fn qr_condition(factor: &Factor) -> Result<f64, SolveError> {
    match factor {
        Factor::DenseQr {
            condition_indicator,
            ..
        } => Ok(*condition_indicator),
        _ => Err(SolveError::ProblemMismatch),
    }
}

fn hodge_factor_bytes(rows: usize, rank: usize) -> Result<u64, SolveError> {
    let block = qr_factor::recommended_block_size::<f64>(rows, rank).max(1);
    let cells = rows
        .checked_mul(rank)
        .and_then(|value| value.checked_add(block.checked_mul(rank)?))
        .ok_or(SolveError::ResourceLimit)?;
    u64::try_from(cells)
        .ok()
        .and_then(|value| value.checked_mul(8))
        .ok_or(SolveError::ResourceLimit)
}

fn hodge_factor_scratch_bytes(
    rows: usize,
    rank: usize,
    executor: Executor,
) -> Result<u64, SolveError> {
    if rank == 0 {
        return Ok(0);
    }
    let block = qr_factor::recommended_block_size::<f64>(rows, rank).max(1);
    let requirement = qr_factor::qr_in_place_scratch::<usize, f64>(
        rows,
        rank,
        block,
        executor.par(),
        Spec::default(),
    );
    u64::try_from(requirement.size_bytes()).map_err(|_| SolveError::ResourceLimit)
}

fn hodge_factor_work(rows: usize, rank: usize) -> Result<u64, SolveError> {
    u64::try_from(rows)
        .ok()
        .and_then(|rows| rows.checked_mul(u64::try_from(rank).ok()?))
        .and_then(|value| value.checked_mul(u64::try_from(rank.max(1)).ok()?))
        .ok_or(SolveError::ResourceLimit)
}

fn hodge_projection_work(rows: usize, rank: usize) -> Result<u64, SolveError> {
    u64::try_from(rows)
        .ok()
        .and_then(|rows| rows.checked_mul(u64::try_from(rank).ok()?))
        .and_then(|value| value.checked_mul(4))
        .ok_or(SolveError::ResourceLimit)
}

fn hodge_certification_work(problem: &HodgeProblem) -> Result<u64, SolveError> {
    let degree = problem.degree().map_err(|_| SolveError::Numerical)?;
    let chain = problem.metric().realization().topology().chain_view();
    let closure = if degree < chain.dimension() {
        let boundary = chain
            .boundary(degree + 1)
            .map_err(|_| SolveError::Numerical)?;
        boundary
            .exact_entries()
            .len()
            .checked_mul(boundary.shape().1)
    } else {
        Some(0)
    };
    let coclosure = if degree > 0 {
        let boundary = chain.boundary(degree).map_err(|_| SolveError::Numerical)?;
        boundary
            .exact_entries()
            .len()
            .checked_mul(boundary.shape().0)
    } else {
        Some(0)
    };
    closure
        .and_then(|value| value.checked_add(coclosure?))
        .and_then(|value| {
            value
                .checked_mul(2)?
                .checked_add(problem.source().coefficients().len())
        })
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(SolveError::ResourceLimit)
}

fn qr_projection_requirement(factor: &Factor) -> Result<StackReq, SolveError> {
    let Factor::DenseQr {
        vectors,
        householder,
        rank,
        ..
    } = factor
    else {
        return Err(SolveError::ProblemMismatch);
    };
    if *rank == 0 {
        return Ok(StackReq::EMPTY);
    }
    let transpose =
        householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<f64>(
            vectors.nrows(),
            householder.nrows(),
            1,
        );
    let forward = householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(
        vectors.nrows(),
        householder.nrows(),
        1,
    );
    Ok(StackReq::any_of(&[transpose, forward]))
}

fn hodge_projection_requirement(
    factors: &[Factor],
    _executor: Executor,
) -> Result<StackReq, SolveError> {
    let requirements = factors
        .iter()
        .map(qr_projection_requirement)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(StackReq::any_of(&requirements))
}

pub(crate) const fn factor_scale(factor: &Factor) -> f64 {
    match factor {
        Factor::Analytic | Factor::DenseQr { .. } => 1.0,
        Factor::Diagonal { scale, .. }
        | Factor::DenseLlt { scale, .. }
        | Factor::SparseLlt { scale, .. }
        | Factor::DenseLu { scale, .. } => *scale,
    }
}

pub(crate) fn factor_solve_requirement(
    factor: &Factor,
    executor: Executor,
    rhs_columns: usize,
) -> StackReq {
    match factor {
        Factor::DenseLlt { factor, .. } => {
            llt::solve::solve_in_place_scratch::<f64>(factor.nrows(), rhs_columns, executor.par())
        }
        Factor::SparseLlt { symbolic, .. } => {
            symbolic.solve_in_place_scratch::<f64>(rhs_columns, executor.par())
        }
        Factor::DenseLu { factor, .. } => lu_solve::solve_in_place_scratch::<usize, f64>(
            factor.nrows(),
            rhs_columns,
            executor.par(),
        ),
        Factor::Analytic | Factor::Diagonal { .. } | Factor::DenseQr { .. } => StackReq::EMPTY,
    }
}

pub(crate) fn solve_factor(
    factor: &Factor,
    mut rhs: MatMut<'_, f64>,
    executor: Executor,
    stack: &mut MemStack,
) {
    match factor {
        Factor::Analytic => {}
        Factor::Diagonal { inverse, .. } => {
            for column in 0..rhs.ncols() {
                for row in 0..rhs.nrows() {
                    rhs[(row, column)] *= inverse[row];
                }
            }
        }
        Factor::DenseLlt { factor, .. } => {
            llt::solve::solve_in_place(factor.as_ref(), rhs, executor.par(), stack);
        }
        Factor::SparseLlt {
            symbolic, numeric, ..
        } => SparseLltRef::new(symbolic, numeric).solve_in_place_with_conj(
            Conj::No,
            rhs,
            executor.par(),
            stack,
        ),
        Factor::DenseLu {
            factor,
            permutation,
            inverse_permutation,
            ..
        } => lu_solve::solve_in_place(
            factor.as_ref(),
            factor.as_ref(),
            PermRef::new_checked(permutation, inverse_permutation, factor.nrows()),
            rhs,
            executor.par(),
            stack,
        ),
        Factor::DenseQr { .. } => unreachable!("QR factors use the projection path"),
    }
}

pub(crate) fn weighted_centroid(
    masses: &[f64],
    positions: &[f64],
    dimension: usize,
) -> Result<Vec<f64>, SolveError> {
    if positions.len() != masses.len().saturating_mul(dimension) {
        return Err(SolveError::Numerical);
    }
    let (total, _) = adaptive_product_value(masses.iter().copied().map(|mass| (mass, 1.0)))
        .ok_or(SolveError::Numerical)?;
    if !total.is_finite() || total <= 0.0 {
        return Err(SolveError::Numerical);
    }
    let mut centroid = Vec::new();
    centroid
        .try_reserve_exact(dimension)
        .map_err(|_| SolveError::Allocation)?;
    for axis in 0..dimension {
        let (sum, _) = adaptive_product_value(
            masses
                .iter()
                .copied()
                .zip(positions.iter().skip(axis).step_by(dimension).copied()),
        )
        .ok_or(SolveError::Numerical)?;
        centroid.push(sum / total);
    }
    centroid
        .iter()
        .all(|value| value.is_finite())
        .then_some(centroid)
        .ok_or(SolveError::Numerical)
}

pub(crate) fn fill_centered_mass_rhs(
    masses: &[f64],
    values: &[f64],
    means: &[f64],
    scale: f64,
    mut rhs: MatMut<'_, f64>,
) -> Result<(), SolveError> {
    let columns = means.len();
    if rhs.nrows() != masses.len()
        || rhs.ncols() != columns
        || values.len() != masses.len().saturating_mul(columns)
        || !scale.is_finite()
        || scale == 0.0
    {
        return Err(SolveError::Numerical);
    }
    for vertex in 0..masses.len() {
        for axis in 0..columns {
            rhs[(vertex, axis)] =
                masses[vertex] * (values[vertex * columns + axis] - means[axis]) / scale;
        }
    }
    Ok(())
}

pub(crate) fn dirichlet_energy(
    metric: &crate::Metric,
    positions: &[f64],
    dimension: usize,
    endpoints: &[[(usize, i64); 2]],
) -> Result<f64, SolveError> {
    let weights = metric
        .hodge_coefficients_slice(1)
        .map_err(|_| SolveError::Numerical)?;
    let mut energy = 0.0;
    for (&weight, endpoints) in weights.iter().zip(endpoints) {
        let left = endpoints[0].0;
        let right = endpoints[1].0;
        for axis in 0..dimension {
            let difference =
                positions[left * dimension + axis] - positions[right * dimension + axis];
            energy += 0.5 * weight * difference * difference;
        }
    }
    energy
        .is_finite()
        .then_some(energy)
        .ok_or(SolveError::Numerical)
}

pub(crate) fn flow_residual(
    metric: &crate::Metric,
    time_step: f64,
    target_column_major: &[f64],
    source_centroid: &[f64],
    endpoints: &[[(usize, i64); 2]],
) -> Result<f64, SolveError> {
    let source = metric.realization();
    let dimension = source.ambient_dimension();
    let masses = metric
        .hodge_coefficients_slice(0)
        .map_err(|_| SolveError::Numerical)?;
    let weights = metric
        .hodge_coefficients_slice(1)
        .map_err(|_| SolveError::Numerical)?;
    let mut action = vec![0.0; target_column_major.len()];
    for vertex in 0..masses.len() {
        for axis in 0..dimension {
            action[axis * masses.len() + vertex] =
                masses[vertex] * target_column_major[axis * masses.len() + vertex];
        }
    }
    for (&weight, endpoints) in weights.iter().zip(endpoints) {
        for &(row, row_sign) in endpoints {
            for &(column, column_sign) in endpoints {
                let sign = if row_sign == column_sign { 1.0 } else { -1.0 };
                for axis in 0..dimension {
                    action[axis * masses.len() + row] += time_step
                        * weight
                        * sign
                        * target_column_major[axis * masses.len() + column];
                }
            }
        }
    }
    let mut residual = 0.0_f64;
    let mut scale = 0.0_f64;
    for vertex in 0..masses.len() {
        for axis in 0..dimension {
            let expected = masses[vertex]
                * (source.positions()[vertex * dimension + axis] - source_centroid[axis]);
            let actual = action[axis * masses.len() + vertex];
            residual = residual.max((actual - expected).abs());
            scale = scale.max(actual.abs()).max(expected.abs());
        }
    }
    let relative = if scale == 0.0 {
        residual
    } else {
        residual / scale
    };
    relative
        .is_finite()
        .then_some(relative)
        .ok_or(SolveError::Numerical)
}

fn heat_evidence(
    problem: &HeatProblem,
    target: &[f64],
    endpoints: &[[(usize, i64); 2]],
) -> Result<(f64, f64, f64, f64, usize), SolveError> {
    let metric = problem.metric();
    let masses = metric
        .hodge_coefficients_slice(0)
        .map_err(|_| SolveError::Numerical)?;
    let weights = metric
        .hodge_coefficients_slice(1)
        .map_err(|_| SolveError::Numerical)?;
    let boundary = metric
        .realization()
        .topology()
        .chain_view()
        .boundary(1)
        .map_err(|_| SolveError::Numerical)?;
    let mut residual_bound = 0.0_f64;
    let mut exact_fallback_rows = 0_usize;
    for row in 0..masses.len() {
        let terms = std::iter::once((masses[row], target[row]))
            .chain(
                boundary.indices()[boundary.indptr()[row]..boundary.indptr()[row + 1]]
                    .iter()
                    .flat_map(|&edge| {
                        let entries = endpoints[edge];
                        let row_sign = if entries[0].0 == row {
                            entries[0].1
                        } else {
                            entries[1].1
                        };
                        entries.into_iter().map(move |(column, column_sign)| {
                            let sign = if row_sign == column_sign { 1.0 } else { -1.0 };
                            (problem.time_step() * weights[edge] * sign, target[column])
                        })
                    }),
            )
            .chain(std::iter::once((
                -masses[row],
                problem.source().coefficients()[row],
            )));
        let scale = terms
            .clone()
            .map(|(left, right)| (left * right).abs())
            .sum::<f64>()
            .max(1.0);
        let verdict = adaptive_product_sum(terms, 256.0 * f64::EPSILON * scale);
        if !verdict.accepted {
            return Err(SolveError::Numerical);
        }
        exact_fallback_rows += usize::from(verdict.exact_fallback);
        residual_bound = residual_bound.max(verdict.bound / scale);
    }
    let mass_terms = masses
        .iter()
        .copied()
        .zip(target.iter().copied())
        .zip(problem.source().coefficients().iter().copied())
        .flat_map(|((mass, target), source)| [(mass, target), (-mass, source)]);
    let mass_scale = mass_terms
        .clone()
        .map(|(left, right)| (left * right).abs())
        .sum::<f64>()
        .max(1.0);
    let mass = adaptive_product_sum(mass_terms, 256.0 * f64::EPSILON * mass_scale);
    if !mass.accepted {
        return Err(SolveError::Numerical);
    }
    let energy_before = dirichlet_energy(metric, problem.source().coefficients(), 1, endpoints)?;
    let energy_after = dirichlet_energy(metric, target, 1, endpoints)?;
    Ok((
        residual_bound,
        mass.bound / mass_scale,
        energy_before,
        energy_after,
        exact_fallback_rows,
    ))
}

pub(crate) fn matrix_bytes(n: usize) -> Result<u64, SolveError> {
    let bytes = n
        .checked_mul(n)
        .and_then(|value| value.checked_mul(size_of::<f64>()))
        .ok_or(SolveError::ResourceLimit)?;
    u64::try_from(bytes).map_err(|_| SolveError::ResourceLimit)
}

pub(crate) fn cubic_work(n: usize) -> Result<u64, SolveError> {
    let value = n
        .checked_mul(n)
        .and_then(|value| value.checked_mul(n))
        .ok_or(SolveError::ResourceLimit)?;
    u64::try_from(value).map_err(|_| SolveError::ResourceLimit)
}

fn solve_work(problem: &PoissonProblem) -> Result<u64, SolveError> {
    let n = problem.len();
    let reduced = n.saturating_sub(1);
    let matrix = reduced
        .checked_mul(reduced)
        .ok_or(SolveError::ResourceLimit)?;
    let certification = if n == 1 {
        0
    } else {
        let edges = problem
            .metric()
            .realization()
            .topology()
            .chain_view()
            .boundary(1)
            .map_err(|_| SolveError::Numerical)?
            .shape()
            .1;
        edges
            .checked_mul(4)
            .and_then(|value| value.checked_add(n.saturating_mul(4)))
            .ok_or(SolveError::ResourceLimit)?
    };
    u64::try_from(matrix.max(certification)).map_err(|_| SolveError::ResourceLimit)
}

fn certification_requirement(problem: &PoissonProblem) -> Result<StackReq, SolveError> {
    if problem.len() == 1 {
        return Ok(StackReq::EMPTY);
    }
    let edges = problem
        .metric()
        .realization()
        .topology()
        .chain_view()
        .boundary(1)
        .map_err(|_| SolveError::Numerical)?
        .shape()
        .1;
    Ok(StackReq::all_of(&[
        StackReq::new::<[(usize, i64); 2]>(edges),
        StackReq::new::<usize>(edges),
    ]))
}

pub(crate) fn require_storage(
    limit: StorageLimit,
    retained: u64,
    peak: u64,
) -> Result<(), SolveError> {
    if retained > limit.retained_logical_bytes() || peak > limit.peak_live_logical_bytes() {
        Err(SolveError::ResourceLimit)
    } else {
        Ok(())
    }
}

pub(crate) fn require_work(limit: WorkLimit, required: u64) -> Result<(), SolveError> {
    if required > limit.steps() {
        Err(SolveError::ResourceLimit)
    } else {
        Ok(())
    }
}

pub(crate) fn check_cancelled(cancellation: &CancellationToken) -> Result<(), SolveError> {
    if cancellation.is_cancelled() {
        Err(SolveError::Cancelled)
    } else {
        Ok(())
    }
}
