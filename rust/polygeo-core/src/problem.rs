use std::{cmp::Ordering, sync::Arc};

use faer::dyn_stack::MemStack;
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};

use crate::form::{dyadic, next_up, rounded_dyadic};
use crate::{
    Binary64Chain, Binary64Cochain, Binary64Element, Binary64ElementError, Binary64Space,
    CanonicalSelection, Cochain, LinearOperator, OperatorError, PairingCapability, PositiveMetric,
    RealizationError, TopologyError, Variance,
};

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

impl From<RealizationError> for ProblemError {
    fn from(_: RealizationError) -> Self {
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
            OperatorError::Realization(_) => Self::Metric,
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
pub struct MeanZeroPoisson {
    metric: PositiveMetric,
    rhs: PoissonRhs,
    compatibility: CompatibilityEvidence,
}

impl private::Sealed for MeanZeroPoisson {}
impl Problem for MeanZeroPoisson {
    type Solution = PoissonSolution;
}

/// One admitted backward-Euler evolution of a full scalar vertex cochain.
#[derive(Clone, Debug)]
pub struct HeatProblem {
    metric: PositiveMetric,
    source: Binary64Cochain,
    time_step: f64,
}

impl private::Sealed for HeatProblem {}
impl Problem for HeatProblem {
    type Solution = HeatSolution;
}

impl HeatProblem {
    pub(crate) const fn metric(&self) -> &PositiveMetric {
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

impl MeanZeroPoisson {
    pub(crate) const fn metric(&self) -> &PositiveMetric {
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
    ) -> Result<PoissonSolution, ProblemError> {
        let metric = self.metric();
        let vertex_weights = metric.hodge_coefficients_slice(0)?;
        if vertex_weights.len() == 1 {
            if self.weak_value(0, vertex_weights) != Some(0.0) || potential.coefficients()[0] != 0.0
            {
                return Err(ProblemError::Numerical);
            }
            return Ok(PoissonSolution::new(
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
        Ok(PoissonSolution::new(
            potential,
            ResidualEvidence::new(residual_bound, gauge.bound, fallbacks),
        ))
    }
}

impl PositiveMetric {
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
    ) -> Result<MeanZeroPoisson, ProblemError> {
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
        Ok(MeanZeroPoisson {
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
    ) -> Result<MeanZeroPoisson, ProblemError> {
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
        Ok(MeanZeroPoisson {
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
    metric: PositiveMetric,
    source: Binary64Cochain,
}

impl private::Sealed for HodgeProblem {}
impl Problem for HodgeProblem {
    type Solution = HodgeDecomposition;
}

impl HodgeProblem {
    pub(crate) const fn metric(&self) -> &PositiveMetric {
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
    metric: &PositiveMetric,
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
    metric: &PositiveMetric,
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
    type Solution = DirichletSolution;
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
    metric: PositiveMetric,
    boundary_values: Binary64Cochain,
}

impl private::Sealed for HarmonicExtension {}
impl Problem for HarmonicExtension {
    type Solution = DirichletSolution;
}

impl HarmonicExtension {
    pub(crate) const fn metric(&self) -> &PositiveMetric {
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
pub struct DirichletSolution {
    value: Binary64Cochain,
    evidence: DirichletEvidence,
}

impl DirichletSolution {
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
pub struct PoissonSolution {
    potential: Binary64Cochain,
    evidence: ResidualEvidence,
}

/// One scalar heat value with the evidence required for publication.
#[derive(Clone, Debug)]
pub struct HeatSolution {
    value: Binary64Cochain,
    residual_bound: f64,
    mass_residual_bound: f64,
    energy_before: f64,
    energy_after: f64,
    exact_fallback_rows: usize,
}

impl HeatSolution {
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

impl PoissonSolution {
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

fn exact_dot_is_zero(left: &[f64], right: &[f64]) -> bool {
    let filter = adaptive_product_sum(left.iter().copied().zip(right.iter().copied()), 0.0);
    filter.accepted
}

#[derive(Clone, Copy)]
pub(crate) struct AdaptiveVerdict {
    pub(crate) accepted: bool,
    pub(crate) bound: f64,
    pub(crate) exact_fallback: bool,
}

pub(crate) fn adaptive_product_sum(
    terms: impl Clone + Iterator<Item = (f64, f64)>,
    tolerance: f64,
) -> AdaptiveVerdict {
    adaptive_scalar_product_sum(terms.map(|(left, right)| [left, right]), tolerance)
}

fn adaptive_triple_product_sum(
    terms: impl Clone + Iterator<Item = (f64, f64, f64)>,
    tolerance: f64,
) -> AdaptiveVerdict {
    adaptive_scalar_product_sum(
        terms.map(|(first, second, third)| [first, second, third]),
        tolerance,
    )
}

fn adaptive_scalar_product_sum<const N: usize>(
    terms: impl Clone + Iterator<Item = [f64; N]>,
    tolerance: f64,
) -> AdaptiveVerdict {
    if !tolerance.is_finite() {
        return AdaptiveVerdict {
            accepted: false,
            bound: f64::INFINITY,
            exact_fallback: false,
        };
    }
    let mut sum = 0.0_f64;
    let mut magnitude = 0.0_f64;
    for factors in terms.clone() {
        let product = factors.into_iter().product::<f64>();
        sum += product;
        magnitude = next_up(magnitude + product.abs());
    }
    let operations = terms.clone().count().saturating_mul(N).saturating_add(1);
    let operation_count = f64::from(u32::try_from(operations).unwrap_or(u32::MAX));
    // One product and one addition per term are covered using epsilon rather
    // than unit roundoff; every bound operation is then rounded outward.
    let gamma = operation_count * f64::EPSILON;
    let bound = if gamma >= 1.0 {
        f64::INFINITY
    } else {
        next_up(next_up(gamma / (1.0 - gamma)) * magnitude + operation_count * f64::from_bits(1))
    };
    if next_up(sum.abs() + bound) <= tolerance {
        return AdaptiveVerdict {
            accepted: true,
            bound: next_up(sum.abs() + bound),
            exact_fallback: false,
        };
    }
    if sum.abs() > next_up(tolerance + bound) {
        return AdaptiveVerdict {
            accepted: false,
            bound: next_up(sum.abs() + bound),
            exact_fallback: false,
        };
    }
    let (value, exponent) = exact_scalar_product_sum(terms);
    AdaptiveVerdict {
        accepted: exact_abs_le(&value, exponent, tolerance),
        bound: next_up(sum.abs() + bound),
        exact_fallback: true,
    }
}

pub(crate) fn adaptive_product_value(
    terms: impl Clone + Iterator<Item = (f64, f64)>,
) -> Option<(f64, bool)> {
    let term_count = terms
        .clone()
        .filter(|&(left, right)| left != 0.0 && right != 0.0)
        .count();
    if term_count == 0 {
        return Some((0.0, false));
    }
    let mut sum = 0.0_f64;
    let mut magnitude = 0.0_f64;
    for (left, right) in terms.clone() {
        let product = left * right;
        sum += product;
        magnitude = next_up(magnitude + product.abs());
    }
    let count = term_count.saturating_mul(2).saturating_add(1);
    let count = f64::from(u32::try_from(count).unwrap_or(u32::MAX));
    let bound = next_up(count * f64::EPSILON * magnitude);
    if sum.is_finite() && sum.abs() > 8.0 * bound {
        return Some((sum, false));
    }
    let (value, exponent) = exact_product_sum(terms);
    let rounded = rounded_dyadic(&value, exponent)?;
    rounded.is_finite().then_some((rounded, true))
}

pub(crate) fn adaptive_product_sign(
    terms: impl Clone + Iterator<Item = (f64, f64)>,
) -> Option<(Ordering, bool)> {
    if terms
        .clone()
        .any(|(left, right)| !left.is_finite() || !right.is_finite())
    {
        return None;
    }
    let mut sum = 0.0_f64;
    let mut magnitude = 0.0_f64;
    let mut count = 0_usize;
    for (left, right) in terms.clone() {
        let product = left * right;
        sum += product;
        magnitude = next_up(magnitude + product.abs());
        count = count.saturating_add(1);
    }
    let operations = count.saturating_mul(2).saturating_add(1);
    let operations = f64::from(u32::try_from(operations).unwrap_or(u32::MAX));
    let bound = next_up(operations * f64::EPSILON * magnitude);
    if sum.is_finite() && sum.abs() > 8.0 * bound {
        return Some((sum.total_cmp(&0.0), false));
    }
    let (value, _) = exact_product_sum(terms);
    Some((value.cmp(&BigInt::zero()), true))
}

fn exact_product_sum(terms: impl Clone + Iterator<Item = (f64, f64)>) -> (BigInt, i32) {
    exact_scalar_product_sum(terms.map(|(left, right)| [left, right]))
}

fn exact_scalar_product_sum<const N: usize>(
    terms: impl Clone + Iterator<Item = [f64; N]>,
) -> (BigInt, i32) {
    let exponent = terms
        .clone()
        .filter(|factors| factors.iter().all(|&factor| factor != 0.0))
        .map(|factors| factors.into_iter().map(|factor| dyadic(factor).1).sum())
        .min();
    let Some(exponent) = exponent else {
        return (BigInt::zero(), 0);
    };
    let value = terms
        .filter(|factors| factors.iter().all(|&factor| factor != 0.0))
        .map(|factors| {
            factors.into_iter().map(dyadic).fold(
                (BigInt::from(1), 0),
                |(value, exponent), (factor, factor_exponent)| {
                    (value * factor, exponent + factor_exponent)
                },
            )
        })
        .fold(BigInt::zero(), |sum, (value, shift)| {
            sum + (value << usize::try_from(shift - exponent).expect("nonnegative dyadic shift"))
        });
    (value, exponent)
}

fn exact_abs_le(value: &BigInt, exponent: i32, tolerance: f64) -> bool {
    if tolerance.is_infinite() {
        return true;
    }
    let (limit, limit_exponent) = dyadic(tolerance);
    if exponent >= limit_exponent {
        (value.abs() << usize::try_from(exponent - limit_exponent).expect("nonnegative shift"))
            <= limit
    } else {
        value.abs()
            <= (limit << usize::try_from(limit_exponent - exponent).expect("nonnegative shift"))
    }
}

#[cfg(test)]
mod tests {
    use super::{adaptive_product_sum, adaptive_product_value};

    #[test]
    fn adaptive_filter_and_exact_fallback_make_the_same_threshold_decision() {
        let hidden = [(1.0, 1.0), (2.0_f64.powi(-53), 1.0), (-1.0, 1.0)];
        let rejected = adaptive_product_sum(hidden.into_iter(), 0.0);
        assert!(!rejected.accepted);
        assert!(rejected.exact_fallback);

        let cancelled = [(1.0, 1.0), (-1.0, 1.0)];
        let exact = adaptive_product_sum(cancelled.into_iter(), 0.0);
        assert!(exact.accepted);
        assert!(exact.exact_fallback);

        let filtered = adaptive_product_sum(cancelled.into_iter(), 1.0e-12);
        assert!(filtered.accepted);
        assert!(!filtered.exact_fallback);
    }

    #[test]
    fn exact_value_fallback_rounds_wide_and_subnormal_dyadics_once() {
        let cancelled = [(f64::MAX, 1.0), (-f64::MAX, 1.0), (1.0, 1.0)];
        assert_eq!(
            adaptive_product_value(cancelled.into_iter()),
            Some((1.0, true))
        );

        let half_subnormal = [(f64::MIN_POSITIVE, 2.0_f64.powi(-53))];
        assert_eq!(
            adaptive_product_value(half_subnormal.into_iter()),
            Some((0.0, true))
        );
        let tie_to_even = [(f64::MIN_POSITIVE, 3.0 * 2.0_f64.powi(-53))];
        assert_eq!(
            adaptive_product_value(tie_to_even.into_iter()),
            Some((2.0 * f64::from_bits(1), true))
        );
    }
}
