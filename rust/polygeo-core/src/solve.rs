use std::{
    cmp::Ordering as CmpOrdering,
    marker::PhantomData,
    mem::size_of,
    num::{NonZeroU32, NonZeroUsize},
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
    linalg::qr::col_pivoting::{factor as qr_factor, solve as qr_solve},
    perm::PermRef,
    sparse::{
        CreationError, FaerError, SparseColMat, Triplet,
        linalg::cholesky::{
            CholeskySymbolicParams, LltRef as SparseLltRef, SymbolicCholesky, SymmetricOrdering,
            factorize_symbolic_cholesky,
        },
    },
};
use num_traits::ToPrimitive;

use crate::incidence::{IncidenceAxis, independent_incidence};
use crate::problem::{adaptive_product_sign, adaptive_product_sum, adaptive_product_value};
use crate::surface::dual_edges;
use crate::{
    Binary64Chain, Binary64Cochain, Binary64Element, Binary64Space, CanonicalSelection, Chain,
    Cochain, DirichletEvidence, DirichletProblem, DirichletSolution, EuclideanRealization,
    FaceDirectionField, FlowEvidence, FlowStep, HarmonicExtension, HeatProblem, HeatSolution,
    HodgeDecomposition, HodgeProblem, HomologyGroup, IntegralCochain, IntegralDualCycleBasis,
    LeastSquaresConformalMapEvidence, LeastSquaresConformalMapSolution, LinearOperator,
    MeanZeroPoisson, NondegenerateCapability, PairingCapability, PoissonSolution, PositiveMetric,
    Problem, RealizationError, RealizationLimit, StorageLimit, SurfaceError, TriangleSurface,
    WorkLimit,
};

type EdgeEndpoints = [(usize, i64); 2];

/// Explicit native execution policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeExecutor {
    parallelism: Option<NonZeroUsize>,
}

impl NativeExecutor {
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
    const fn par(self) -> Par {
        match self.parallelism {
            Some(threads) => Par::Rayon(threads),
            None => Par::Seq,
        }
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

/// Period-normalized harmonic degree-one cochains over one positive metric.
#[derive(Clone, Debug)]
pub struct HarmonicOneFormBasis {
    forms: Box<[Binary64Cochain]>,
    maximum_closedness_residual: f64,
    maximum_coclosedness_residual: f64,
    maximum_identity_period_residual: f64,
    residual_limit: f64,
}

impl HarmonicOneFormBasis {
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.forms.len()
    }

    #[must_use]
    pub const fn forms(&self) -> &[Binary64Cochain] {
        &self.forms
    }

    #[must_use]
    pub const fn maximum_closedness_residual(&self) -> f64 {
        self.maximum_closedness_residual
    }

    #[must_use]
    pub const fn maximum_coclosedness_residual(&self) -> f64 {
        self.maximum_coclosedness_residual
    }

    #[must_use]
    pub const fn maximum_identity_period_residual(&self) -> f64 {
        self.maximum_identity_period_residual
    }

    #[must_use]
    pub const fn residual_limit(&self) -> f64 {
        self.residual_limit
    }
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
enum Factor {
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
        owner: Arc<crate::EuclideanRealization>,
    },
    Dirichlet {
        operator: LinearOperator<Cochain, Cochain>,
        selection: Arc<CanonicalSelection>,
    },
    Harmonic {
        owner: Arc<crate::EuclideanRealization>,
        selection: Arc<CanonicalSelection>,
    },
    Hodge {
        owner: Arc<crate::EuclideanRealization>,
        degree: usize,
    },
    Parabolic {
        owner: Arc<crate::EuclideanRealization>,
        time_step_bits: u64,
    },
}

#[derive(Clone, Copy)]
enum SystemRef<'a> {
    PositiveSquare {
        metric: &'a crate::PositiveMetric,
    },
    Parabolic {
        metric: &'a crate::PositiveMetric,
        time_step: f64,
    },
}

impl<'a> SystemRef<'a> {
    const fn metric(self) -> &'a crate::PositiveMetric {
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
    executor: NativeExecutor,
    factors: Factors,
    family: PhantomData<fn() -> P>,
}

/// Caller-owned solve scratch without duplicated requirement metadata.
pub struct SolveWorkspace {
    buffer: MemBuffer,
}

impl std::fmt::Debug for SolveWorkspace {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SolveWorkspace")
            .finish_non_exhaustive()
    }
}

/// Receiver-led preparation workflow for one admitted problem family.
pub trait SolveExt: Problem + Sized {
    /// Prepare reusable RHS-independent physical work.
    ///
    /// # Errors
    /// Rejects exceeded resource limits or a failed positive factorization.
    fn prepare_with(
        &self,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
    ) -> Result<Prepared<Self>, SolveError>;

    /// Prepare with cooperative checks around an uninterruptible factor call.
    ///
    /// # Errors
    /// Also rejects cancellation before preparation publication.
    fn prepare_with_cancellation(
        &self,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<Prepared<Self>, SolveError>;
}

impl SolveExt for MeanZeroPoisson {
    fn prepare_with(
        &self,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
    ) -> Result<Prepared<Self>, SolveError> {
        self.prepare_with_cancellation(executor, storage, work, &CancellationToken::new())
    }

    fn prepare_with_cancellation(
        &self,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<Prepared<Self>, SolveError> {
        Prepared::mean_zero(self, *executor, storage, work, cancellation)
    }
}

impl Prepared<MeanZeroPoisson> {
    fn mean_zero(
        problem: &MeanZeroPoisson,
        executor: NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<Self, SolveError> {
        check_cancelled(cancellation)?;
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
        require_storage(storage, bytes, bytes.saturating_mul(2))?;
        require_work(work, cubic_work(reduced)?)?;
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
            executor,
            factors: Factors::One(factor),
            family: PhantomData,
        })
    }

    /// Allocate workspace for a compatible RHS without touching factors.
    ///
    /// # Errors
    /// Rejects a foreign problem or insufficient logical storage.
    pub fn workspace_for(
        &self,
        problem: &MeanZeroPoisson,
        storage: StorageLimit,
    ) -> Result<SolveWorkspace, SolveError> {
        self.require_problem(problem)?;
        let requirement = self.solve_requirement(problem)?;
        let bytes =
            u64::try_from(requirement.size_bytes()).map_err(|_| SolveError::ResourceLimit)?;
        require_storage(storage, 0, bytes)?;
        let buffer = MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?;
        Ok(SolveWorkspace { buffer })
    }

    /// Solve and certify one compatible RHS.
    ///
    /// # Errors
    /// Rejects mismatch, resource exhaustion, or failed numerical certification.
    pub fn solve(
        &self,
        problem: &MeanZeroPoisson,
        workspace: &mut SolveWorkspace,
        work: WorkLimit,
    ) -> Result<PoissonSolution, SolveError> {
        self.solve_cancellable(problem, workspace, work, &CancellationToken::new())
    }

    /// Solve with cooperative checks around the backend factor call.
    ///
    /// # Errors
    /// Also rejects cancellation before result publication.
    pub fn solve_cancellable(
        &self,
        problem: &MeanZeroPoisson,
        workspace: &mut SolveWorkspace,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<PoissonSolution, SolveError> {
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
        require_work(work, solve_work(problem)?)?;
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
        solve_factor(factor, rhs.as_mut(), self.executor, stack);
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

    fn require_problem(&self, problem: &MeanZeroPoisson) -> Result<(), SolveError> {
        let ReuseKey::MeanZero { owner } = &self.key else {
            return Err(SolveError::ProblemMismatch);
        };
        Arc::ptr_eq(owner, problem.metric().realization())
            .then_some(())
            .ok_or(SolveError::ProblemMismatch)
    }

    fn solve_requirement(&self, problem: &MeanZeroPoisson) -> Result<StackReq, SolveError> {
        let reduced = problem.len().saturating_sub(1);
        let backend = factor_solve_requirement(one_factor(&self.factors)?, self.executor, 1);
        let solve = StackReq::new::<f64>(reduced).and(backend);
        let certification = certification_requirement(problem)?;
        Ok(StackReq::any_of(&[solve, certification]))
    }
}

impl SolveExt for HeatProblem {
    fn prepare_with(
        &self,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
    ) -> Result<Prepared<Self>, SolveError> {
        self.prepare_with_cancellation(executor, storage, work, &CancellationToken::new())
    }

    fn prepare_with_cancellation(
        &self,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<Prepared<Self>, SolveError> {
        check_cancelled(cancellation)?;
        let n = self.source().coefficients().len();
        let factor = if self.metric().realization().topology().dimension() == 0 {
            Factor::Analytic
        } else {
            let bytes = matrix_bytes(n)?;
            require_storage(storage, bytes, bytes.saturating_mul(2))?;
            require_work(work, cubic_work(n)?)?;
            let free = (0..n).collect::<Vec<_>>();
            factor_stiffness(
                SystemRef::Parabolic {
                    metric: self.metric(),
                    time_step: self.time_step(),
                },
                &free,
                *executor,
                cancellation,
            )?
        };
        check_cancelled(cancellation)?;
        Ok(Prepared {
            key: ReuseKey::Parabolic {
                owner: Arc::clone(self.metric().realization()),
                time_step_bits: self.time_step().to_bits(),
            },
            executor: *executor,
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
    pub fn workspace_for(
        &self,
        problem: &HeatProblem,
        storage: StorageLimit,
    ) -> Result<SolveWorkspace, SolveError> {
        self.require_problem(problem)?;
        let requirement = self.solve_requirement(problem)?;
        let bytes =
            u64::try_from(requirement.size_bytes()).map_err(|_| SolveError::ResourceLimit)?;
        require_storage(storage, 0, bytes)?;
        Ok(SolveWorkspace {
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
        workspace: &mut SolveWorkspace,
        work: WorkLimit,
    ) -> Result<HeatSolution, SolveError> {
        self.solve_cancellable(problem, workspace, work, &CancellationToken::new())
    }

    /// Solve with cooperative cancellation before result publication.
    ///
    /// # Errors
    /// Rejects mismatch, resource exhaustion, cancellation, or failed certification.
    pub fn solve_cancellable(
        &self,
        problem: &HeatProblem,
        workspace: &mut SolveWorkspace,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<HeatSolution, SolveError> {
        self.require_problem(problem)?;
        check_cancelled(cancellation)?;
        let n = problem.source().coefficients().len();
        if problem.metric().realization().topology().dimension() == 0 {
            return Ok(HeatSolution::new(
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
        require_work(work, steps)?;
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
        solve_factor(factor, rhs.as_mut(), self.executor, stack);
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
        Ok(HeatSolution::new(
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
            self.executor,
            1,
        )))
    }
}

impl crate::PositiveMetric {
    /// Construct harmonic degree-one cochains dual to exact free homology cycles.
    ///
    /// # Errors
    /// Rejects a foreign or non-degree-one homology group, an unsuitable surface
    /// topology, exhausted resources, cancellation, or failed numerical
    /// certification.
    pub fn harmonic_one_form_basis(
        &self,
        group: HomologyGroup<'_>,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<HarmonicOneFormBasis, SurfaceComputationError> {
        let topology = self.realization().topology();
        if group.degree() != 1 || !group.chain_complex().same_owner(&topology.chain_complex()) {
            return Err(SolveError::ProblemMismatch.into());
        }
        if !group.torsion_orders().is_empty() {
            return Err(SolveError::ProblemMismatch.into());
        }
        let edge_count = topology.basis(1).map_err(SurfaceError::from)?.row_count();
        let face_count = topology.basis(2).map_err(SurfaceError::from)?.row_count();
        require_harmonic_basis_resources(
            topology.vertex_count(),
            edge_count,
            face_count,
            group.free_rank(),
            storage,
            work,
        )
        .map_err(SurfaceComputationError::Solve)?;
        check_cancelled(cancellation).map_err(SurfaceComputationError::Solve)?;
        let seeds = topology
            .integral_dual_cycle_basis()
            .map_err(SurfaceError::from)?;
        if seeds.rank() != group.free_rank() {
            return Err(SolveError::ProblemMismatch.into());
        }
        harmonic_one_form_basis(self, group, &seeds, *executor, storage, work, cancellation)
            .map_err(SurfaceComputationError::Solve)
    }

    /// Compute and atomically publish one frozen-metric mean-curvature-flow step.
    ///
    /// # Errors
    /// Rejects an unsuitable surface, invalid time, exhausted resources,
    /// cancellation, failed factorization, or failed numerical certification.
    pub fn frozen_mean_curvature_flow(
        &self,
        time_step: f64,
        realization_limit: RealizationLimit,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<FlowStep, SurfaceComputationError> {
        require_frozen_flow_domain(self, time_step)?;
        frozen_flow_step(
            self,
            time_step,
            realization_limit,
            *executor,
            storage,
            work,
            cancellation,
        )
        .map_err(SurfaceComputationError::Solve)
    }
}

fn harmonic_one_form_basis(
    metric: &PositiveMetric,
    group: HomologyGroup<'_>,
    seeds: &IntegralDualCycleBasis,
    executor: NativeExecutor,
    storage: StorageLimit,
    work: WorkLimit,
    cancellation: &CancellationToken,
) -> Result<HarmonicOneFormBasis, SolveError> {
    check_cancelled(cancellation)?;
    let rank = seeds.rank();
    let topology = metric.realization().topology();
    if rank == 0 {
        return Ok(empty_harmonic_one_form_basis());
    }
    let edge_space = Binary64Space::<Cochain>::full(Arc::clone(topology), 1)
        .map_err(|_| SolveError::Numerical)?;
    let harmonic_seeds = project_harmonic_seeds(
        metric,
        seeds,
        &edge_space,
        executor,
        storage,
        work,
        cancellation,
    )?;
    let forms =
        normalize_harmonic_periods(group, &edge_space, &harmonic_seeds, executor, cancellation)?;
    certify_harmonic_one_form_basis(metric, group, forms)
}

fn require_harmonic_basis_resources(
    vertex_count: usize,
    edge_count: usize,
    face_count: usize,
    rank: usize,
    storage: StorageLimit,
    work: WorkLimit,
) -> Result<(), SolveError> {
    let retained_cells = rank
        .checked_mul(edge_count)
        .ok_or(SolveError::ResourceLimit)?;
    let retained = logical_bytes(retained_cells, size_of::<f64>())?;
    let exact_seed_width = size_of::<usize>()
        .checked_add(size_of::<num_bigint::BigInt>())
        .ok_or(SolveError::ResourceLimit)?;
    let exact_seeds = logical_bytes(retained_cells, exact_seed_width)?;
    let edge_tree_cells = edge_count.checked_mul(8).ok_or(SolveError::ResourceLimit)?;
    let tree_cells = vertex_count
        .checked_add(face_count.saturating_mul(4))
        .and_then(|value| value.checked_add(edge_tree_cells))
        .ok_or(SolveError::ResourceLimit)?;
    let tree_workspace = logical_bytes(tree_cells, size_of::<usize>())?;
    let cochain_temporaries = retained.checked_mul(3).ok_or(SolveError::ResourceLimit)?;
    let poisson_factor = matrix_bytes(vertex_count.saturating_sub(1))?;
    let period_matrices = matrix_bytes(rank)?
        .checked_mul(3)
        .ok_or(SolveError::ResourceLimit)?;
    let peak = retained
        .checked_add(cochain_temporaries)
        .and_then(|value| value.checked_add(exact_seeds))
        .and_then(|value| value.checked_add(tree_workspace))
        .and_then(|value| value.checked_add(poisson_factor.saturating_mul(2)))
        .and_then(|value| value.checked_add(period_matrices))
        .ok_or(SolveError::ResourceLimit)?;
    require_storage(storage, retained, peak)?;
    let solve_steps = vertex_count
        .saturating_sub(1)
        .checked_mul(vertex_count.saturating_sub(1))
        .and_then(|value| value.checked_mul(rank))
        .and_then(|value| value.checked_add(retained_cells.saturating_mul(rank.max(1))))
        .ok_or(SolveError::ResourceLimit)?;
    let required_work = cubic_work(vertex_count.saturating_sub(1))?
        .checked_add(cubic_work(rank)?)
        .and_then(|value| value.checked_add(u64::try_from(solve_steps).ok()?))
        .and_then(|value| value.checked_add(u64::try_from(tree_cells).ok()?))
        .ok_or(SolveError::ResourceLimit)?;
    require_work(work, required_work)
}

fn empty_harmonic_one_form_basis() -> HarmonicOneFormBasis {
    HarmonicOneFormBasis {
        forms: Box::new([]),
        maximum_closedness_residual: 0.0,
        maximum_coclosedness_residual: 0.0,
        maximum_identity_period_residual: 0.0,
        residual_limit: 0.0,
    }
}

fn project_harmonic_seeds(
    metric: &PositiveMetric,
    seeds: &IntegralDualCycleBasis,
    edge_space: &Binary64Space<Cochain>,
    executor: NativeExecutor,
    storage: StorageLimit,
    work: WorkLimit,
    cancellation: &CancellationToken,
) -> Result<Vec<Binary64Cochain>, SolveError> {
    let rank = seeds.rank();
    let internal_storage = StorageLimit::new(
        storage.peak_live_logical_bytes(),
        storage.peak_live_logical_bytes(),
    )
    .ok_or(SolveError::ResourceLimit)?;
    let codifferential = metric
        .codifferential(1)
        .map_err(|_| SolveError::Numerical)?;
    let vertex_riesz = metric.riesz(0).map_err(|_| SolveError::Numerical)?;
    let mut seed_values = Vec::new();
    seed_values
        .try_reserve_exact(rank)
        .map_err(|_| SolveError::Allocation)?;
    let mut problems = Vec::new();
    problems
        .try_reserve_exact(rank)
        .map_err(|_| SolveError::Allocation)?;
    for index in 0..rank {
        let exact = seeds.cocycle(index).ok_or(SolveError::Numerical)?;
        let seed = Binary64Element::realize_integral(edge_space.clone(), exact)
            .map_err(|_| SolveError::Numerical)?;
        let rhs = codifferential
            .apply(&seed)
            .map_err(|_| SolveError::Numerical)?;
        let load = vertex_riesz
            .apply(&rhs)
            .map_err(|_| SolveError::Numerical)?;
        let problem = metric
            .mean_zero_poisson_load(load)
            .map_err(|_| SolveError::Numerical)?;
        seed_values.push(seed);
        problems.push(problem);
    }
    let prepared =
        problems[0].prepare_with_cancellation(&executor, internal_storage, work, cancellation)?;
    let mut harmonic_seeds = Vec::new();
    harmonic_seeds
        .try_reserve_exact(rank)
        .map_err(|_| SolveError::Allocation)?;
    for (seed, problem) in seed_values.iter().zip(&problems) {
        check_cancelled(cancellation)?;
        let mut workspace = prepared.workspace_for(problem, internal_storage)?;
        let solution = prepared.solve_cancellable(problem, &mut workspace, work, cancellation)?;
        let exact = solution
            .potential()
            .exterior_derivative()
            .map_err(|_| SolveError::Numerical)?;
        let coefficients = seed
            .coefficients()
            .iter()
            .zip(exact.coefficients())
            .map(|(&seed, &exact)| seed - exact)
            .collect::<Vec<_>>();
        harmonic_seeds.push(
            Binary64Element::admit(edge_space.clone(), coefficients)
                .map_err(|_| SolveError::Numerical)?,
        );
    }
    Ok(harmonic_seeds)
}

fn normalize_harmonic_periods(
    group: HomologyGroup<'_>,
    edge_space: &Binary64Space<Cochain>,
    harmonic_seeds: &[Binary64Cochain],
    executor: NativeExecutor,
    cancellation: &CancellationToken,
) -> Result<Vec<Binary64Cochain>, SolveError> {
    let rank = harmonic_seeds.len();
    let edge_count = edge_space.size();
    let mut periods = Mat::<f64>::zeros(rank, rank);
    for (column, harmonic) in harmonic_seeds.iter().enumerate() {
        for (row, &period) in group
            .periods_binary64(harmonic)
            .map_err(|_| SolveError::ProblemMismatch)?
            .iter()
            .enumerate()
        {
            periods[(row, column)] = period;
        }
    }
    let factor = factor_dense_square(periods, executor, cancellation)?;
    require_stable_dense_lu(&factor)?;
    let requirement = factor_solve_requirement(&factor, executor, rank);
    let mut buffer = MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?;
    let scale = factor_scale(&factor);
    let mut coordinates = Mat::<f64>::zeros(rank, rank);
    for index in 0..rank {
        coordinates[(index, index)] = 1.0 / scale;
    }
    solve_factor(
        &factor,
        coordinates.as_mut(),
        executor,
        MemStack::new(&mut buffer),
    );
    check_cancelled(cancellation)?;

    let mut forms = Vec::new();
    forms
        .try_reserve_exact(rank)
        .map_err(|_| SolveError::Allocation)?;
    for column in 0..rank {
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(edge_count)
            .map_err(|_| SolveError::Allocation)?;
        for edge in 0..edge_count {
            let (value, _) = adaptive_product_value(
                harmonic_seeds
                    .iter()
                    .enumerate()
                    .map(|(row, form)| (form.coefficients()[edge], coordinates[(row, column)])),
            )
            .ok_or(SolveError::Numerical)?;
            coefficients.push(value);
        }
        forms.push(
            Binary64Element::admit(edge_space.clone(), coefficients)
                .map_err(|_| SolveError::Numerical)?,
        );
    }
    Ok(forms)
}

fn certify_harmonic_one_form_basis(
    metric: &PositiveMetric,
    group: HomologyGroup<'_>,
    forms: Vec<Binary64Cochain>,
) -> Result<HarmonicOneFormBasis, SolveError> {
    let codifferential = metric
        .codifferential(1)
        .map_err(|_| SolveError::Numerical)?;
    let mut closedness = 0.0_f64;
    let mut coclosedness = 0.0_f64;
    let mut identity_period = 0.0_f64;
    for (column, form) in forms.iter().enumerate() {
        let derivative = form
            .exterior_derivative()
            .map_err(|_| SolveError::Numerical)?;
        closedness = closedness.max(maximum_absolute(derivative.coefficients())?);
        let codifferential = codifferential
            .apply(form)
            .map_err(|_| SolveError::Numerical)?;
        coclosedness = coclosedness.max(maximum_absolute(codifferential.coefficients())?);
        for (row, &period) in group
            .periods_binary64(form)
            .map_err(|_| SolveError::ProblemMismatch)?
            .iter()
            .enumerate()
        {
            identity_period = identity_period.max((period - f64::from(row == column)).abs());
        }
    }
    let operation_count = forms
        .len()
        .saturating_add(metric.realization().topology().vertex_count())
        .saturating_add(
            metric
                .realization()
                .topology()
                .basis(1)
                .map_err(|_| SolveError::Numerical)?
                .row_count(),
        );
    let residual_limit = 4096.0
        * f64::EPSILON
        * f64::from(u32::try_from(operation_count.max(1)).unwrap_or(u32::MAX));
    if !residual_limit.is_finite()
        || closedness > residual_limit
        || coclosedness > residual_limit
        || identity_period > residual_limit
    {
        return Err(SolveError::Numerical);
    }
    Ok(HarmonicOneFormBasis {
        forms: forms.into_boxed_slice(),
        maximum_closedness_residual: closedness,
        maximum_coclosedness_residual: coclosedness,
        maximum_identity_period_residual: identity_period,
        residual_limit,
    })
}

fn maximum_absolute(values: &[f64]) -> Result<f64, SolveError> {
    values.iter().try_fold(0.0_f64, |maximum, &value| {
        value
            .is_finite()
            .then_some(maximum.max(value.abs()))
            .ok_or(SolveError::Numerical)
    })
}

fn dual_edge_values(
    surface: &TriangleSurface,
    chain: &Binary64Chain,
) -> Result<Vec<f64>, SurfaceComputationError> {
    let topology = surface.realization().topology();
    let expected =
        Binary64Space::<Chain>::full(Arc::clone(topology), 1).map_err(|_| SolveError::Numerical)?;
    if !expected.same_space(chain.space()) {
        return Err(SolveError::ProblemMismatch.into());
    }
    let dual = dual_edges(topology)?;
    chain
        .coefficients()
        .iter()
        .enumerate()
        .map(|(edge, &value)| {
            let (_, _, source_sign) = dual.edge(edge)?;
            Ok(f64::from(source_sign) * value)
        })
        .collect()
}

fn dual_period(
    surface: &TriangleSurface,
    cycles: &IntegralDualCycleBasis,
    cycle_index: usize,
    values: &[f64],
) -> Result<f64, SurfaceComputationError> {
    let cycle = cycles
        .cocycle(cycle_index)
        .ok_or(SurfaceError::IndexOutside)?;
    let dual = dual_edges(surface.realization().topology())?;
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(cycle.indices().len())
        .map_err(|_| SolveError::Allocation)?;
    for (&edge, coefficient) in cycle.indices().iter().zip(cycle.coefficients()) {
        let coefficient = coefficient
            .to_f64()
            .filter(|value| value.is_finite())
            .ok_or(SurfaceError::Unrepresentable)?;
        let (_, _, source_sign) = dual.edge(edge)?;
        terms.push((-f64::from(source_sign) * coefficient, values[edge]));
    }
    adaptive_product_value(terms.into_iter())
        .map(|(value, _)| value)
        .ok_or_else(|| SolveError::Numerical.into())
}

fn exact_charge_sum(charges: &IntegralCochain) -> num_bigint::BigInt {
    charges.coefficients().iter().sum()
}

fn same_integral_coordinates(left: &IntegralCochain, right: &IntegralCochain) -> bool {
    left.space().same_based_space(right.space())
        && left.indices() == right.indices()
        && left.coefficients() == right.coefficients()
}

fn require_direction_field_resources(
    vertex_count: usize,
    edge_count: usize,
    face_count: usize,
    rank: usize,
    storage: StorageLimit,
    work: WorkLimit,
) -> Result<(), SolveError> {
    let retained_scalars = edge_count
        .saturating_mul(2)
        .saturating_add(face_count.saturating_mul(4));
    let retained = logical_bytes(retained_scalars, size_of::<f64>())?;
    let edge_temporaries = logical_bytes(
        edge_count.saturating_mul(rank.saturating_add(4)),
        size_of::<f64>(),
    )?;
    let vertex_temporaries = logical_bytes(vertex_count.saturating_mul(3), size_of::<f64>())?;
    let period_matrices = matrix_bytes(rank)?.saturating_mul(2);
    let poisson_factor = matrix_bytes(vertex_count.saturating_sub(1))?;
    let peak = retained
        .checked_add(edge_temporaries)
        .and_then(|value| value.checked_add(vertex_temporaries))
        .and_then(|value| value.checked_add(period_matrices))
        .and_then(|value| value.checked_add(poisson_factor.saturating_mul(2)))
        .ok_or(SolveError::ResourceLimit)?;
    require_storage(storage, retained, peak)?;
    let required_work = cubic_work(vertex_count.saturating_sub(1))?
        .checked_add(cubic_work(rank)?)
        .and_then(|value| {
            value
                .checked_add(u64::try_from(edge_count.saturating_mul(rank.saturating_add(4))).ok()?)
        })
        .ok_or(SolveError::ResourceLimit)?;
    require_work(work, required_work)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the helper keeps mathematical inputs and existing execution policies explicit"
)]
fn solve_coexact_direction_adjustment(
    surface: &TriangleSurface,
    symmetry_order: NonZeroU32,
    metric: &PositiveMetric,
    charges: &IntegralCochain,
    executor: NativeExecutor,
    storage: StorageLimit,
    work: WorkLimit,
    cancellation: &CancellationToken,
) -> Result<Vec<f64>, SurfaceComputationError> {
    let topology = surface.realization().topology();
    let curvature = surface.gaussian_curvature_measure()?;
    let order = f64::from(symmetry_order.get());
    let mut load_values = curvature
        .coefficients()
        .iter()
        .map(|value| -order * *value)
        .collect::<Vec<_>>();
    for (&vertex, coefficient) in charges.indices().iter().zip(charges.coefficients()) {
        let charge = coefficient
            .to_f64()
            .filter(|value| value.is_finite())
            .ok_or(SurfaceError::Unrepresentable)?;
        load_values[vertex] += std::f64::consts::TAU * charge;
    }
    let load_space =
        Binary64Space::<Chain>::full(Arc::clone(topology), 0).map_err(|_| SolveError::Numerical)?;
    let load =
        Binary64Element::admit(load_space, load_values).map_err(|_| SolveError::Numerical)?;
    let problem = metric
        .mean_zero_poisson_load(load)
        .map_err(|_| SolveError::Numerical)?;
    let prepared = problem.prepare_with_cancellation(&executor, storage, work, cancellation)?;
    let mut workspace = prepared.workspace_for(&problem, storage)?;
    let solution = prepared.solve_cancellable(&problem, &mut workspace, work, cancellation)?;
    let gradient = solution
        .potential()
        .exterior_derivative()
        .map_err(|_| SolveError::Numerical)?;
    let coexact = metric
        .riesz(1)
        .map_err(|_| SolveError::Numerical)?
        .apply(&gradient)
        .map_err(|_| SolveError::Numerical)?;
    dual_edge_values(surface, &coexact)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the private kernel mirrors explicit mathematical and execution inputs"
)]
fn add_harmonic_direction_adjustment(
    surface: &Arc<TriangleSurface>,
    symmetry_order: NonZeroU32,
    metric: &PositiveMetric,
    harmonic_basis: &HarmonicOneFormBasis,
    dual_cycles: &IntegralDualCycleBasis,
    generator_turns: &[i64],
    deviations: &mut [f64],
    executor: NativeExecutor,
    cancellation: &CancellationToken,
) -> Result<(), SurfaceComputationError> {
    let rank = dual_cycles.rank();
    if rank == 0 {
        return Ok(());
    }
    let levi_civita_angles = surface
        .levi_civita_connection()?
        .transports()
        .chunks_exact(2)
        .map(|value| value[1].atan2(value[0]))
        .collect::<Vec<_>>();
    let order = f64::from(symmetry_order.get());
    let mut harmonic_dual = Vec::new();
    harmonic_dual
        .try_reserve_exact(rank)
        .map_err(|_| SolveError::Allocation)?;
    let riesz = metric.riesz(1).map_err(|_| SolveError::Numerical)?;
    for form in harmonic_basis.forms() {
        let chain = riesz.apply(form).map_err(|_| SolveError::ProblemMismatch)?;
        harmonic_dual.push(dual_edge_values(surface, &chain)?);
    }
    let mut periods = Mat::<f64>::zeros(rank, rank);
    let mut target = Mat::<f64>::zeros(rank, 1);
    for row in 0..rank {
        let base_angle = dual_period(surface, dual_cycles, row, &levi_civita_angles)?;
        let coexact_period = dual_period(surface, dual_cycles, row, deviations)?;
        let turns = generator_turns[row]
            .to_f64()
            .ok_or(SurfaceError::Unrepresentable)?;
        target[(row, 0)] = std::f64::consts::TAU * turns - order * base_angle - coexact_period;
        for column in 0..rank {
            periods[(row, column)] =
                dual_period(surface, dual_cycles, row, &harmonic_dual[column])?;
        }
    }
    let factor = factor_dense_square(periods, executor, cancellation)?;
    require_stable_dense_lu(&factor)?;
    let requirement = factor_solve_requirement(&factor, executor, 1);
    let mut buffer = MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?;
    let scale = factor_scale(&factor);
    for row in 0..rank {
        target[(row, 0)] /= scale;
    }
    solve_factor(
        &factor,
        target.as_mut(),
        executor,
        MemStack::new(&mut buffer),
    );
    for (edge, deviation) in deviations.iter_mut().enumerate() {
        let (correction, _) = adaptive_product_value(
            harmonic_dual
                .iter()
                .enumerate()
                .map(|(column, form)| (form[edge], target[(column, 0)])),
        )
        .ok_or(SolveError::Numerical)?;
        *deviation += correction;
    }
    let operation_count =
        u32::try_from(deviations.len().saturating_add(rank).saturating_add(1)).unwrap_or(u32::MAX);
    let period_limit = 8192.0 * f64::EPSILON * f64::from(operation_count) * order;
    if period_limit >= std::f64::consts::PI {
        return Err(SolveError::Numerical.into());
    }
    for (row, &turns) in generator_turns.iter().enumerate() {
        let turns = turns.to_f64().ok_or(SurfaceError::Unrepresentable)?;
        let observed = order * dual_period(surface, dual_cycles, row, &levi_civita_angles)?
            + dual_period(surface, dual_cycles, row, deviations)?;
        if (observed - std::f64::consts::TAU * turns).abs() > period_limit {
            return Err(SolveError::Numerical.into());
        }
    }
    Ok(())
}

impl TriangleSurface {
    /// Construct a minimum-energy symmetric direction field with exact charges and turns.
    ///
    /// The supplied degree-zero integral cochain fixes power charges. Generator turns are
    /// lifted power holonomies in the order of `dual_cycles`; `anchor_angle` fixes the
    /// remaining global rotation modulo the supplied symmetry order.
    ///
    /// # Errors
    /// Rejects foreign inputs, a charge sum different from order times Euler characteristic, mismatched
    /// generator dimensions, exhausted resources, cancellation, or failed certification.
    #[expect(
        clippy::too_many_arguments,
        reason = "mathematical inputs and execution policies remain explicit"
    )]
    pub fn minimum_energy_direction_field(
        self: &Arc<Self>,
        symmetry_order: NonZeroU32,
        metric: &PositiveMetric,
        harmonic_basis: &HarmonicOneFormBasis,
        dual_cycles: &IntegralDualCycleBasis,
        charges: &IntegralCochain,
        generator_turns: &[i64],
        anchor_angle: f64,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<FaceDirectionField, SurfaceComputationError> {
        let topology = self.realization().topology();
        if !Arc::ptr_eq(metric.realization(), self.realization())
            || !dual_cycles
                .chain_complex()
                .same_owner(&topology.chain_complex())
        {
            return Err(SolveError::ProblemMismatch.into());
        }
        let charge_space = topology
            .chain_complex()
            .dual()
            .space(0)
            .map_err(SurfaceError::from)?;
        if !charges.space().same_based_space(&charge_space)
            || harmonic_basis.rank() != dual_cycles.rank()
            || generator_turns.len() != dual_cycles.rank()
            || !anchor_angle.is_finite()
        {
            return Err(SolveError::ProblemMismatch.into());
        }
        let edge_count = topology.basis(1).map_err(SurfaceError::from)?.row_count();
        require_direction_field_resources(
            topology.vertex_count(),
            edge_count,
            self.face_count(),
            harmonic_basis.rank(),
            storage,
            work,
        )?;
        check_cancelled(cancellation)?;
        let euler = i128::try_from(topology.vertex_count())
            .ok()
            .and_then(|value| value.checked_sub(i128::try_from(edge_count).ok()?))
            .and_then(|value| value.checked_add(i128::try_from(self.face_count()).ok()?))
            .ok_or(SurfaceError::Overflow)?;
        let expected_charge =
            num_bigint::BigInt::from(symmetry_order.get()) * num_bigint::BigInt::from(euler);
        if exact_charge_sum(charges) != expected_charge {
            return Err(SolveError::ProblemMismatch.into());
        }

        let mut deviations = solve_coexact_direction_adjustment(
            self,
            symmetry_order,
            metric,
            charges,
            *executor,
            storage,
            work,
            cancellation,
        )?;
        add_harmonic_direction_adjustment(
            self,
            symmetry_order,
            metric,
            harmonic_basis,
            dual_cycles,
            generator_turns,
            &mut deviations,
            *executor,
            cancellation,
        )?;
        check_cancelled(cancellation)?;
        let field = self
            .connection(symmetry_order, &deviations)?
            .require_integrable()?
            .direction_field(anchor_angle)?;
        let observed = field.singularities()?;
        if observed.symmetry_order() != symmetry_order
            || !same_integral_coordinates(observed.charges(), charges)
        {
            return Err(SolveError::Numerical.into());
        }
        Ok(field)
    }

    /// Compute one bounded least-squares conformal parameterization of an oriented disk.
    ///
    /// The two distinct boundary anchors map to `(0, 0)` and `(1, 0)` in caller order.
    ///
    /// # Errors
    /// Rejects a non-disk domain, invalid anchors, exhausted resources, cancellation,
    /// numerical rank loss, or a target face without positive admitted orientation.
    pub fn least_squares_conformal_map(
        &self,
        anchors: [usize; 2],
        realization_limit: RealizationLimit,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<LeastSquaresConformalMapSolution, SurfaceComputationError> {
        let disk = self
            .realization()
            .topology()
            .refine_disk()
            .map_err(SurfaceError::from)?;
        let boundary = disk.boundary_vertices().map_err(SurfaceError::from)?;
        let vertex_count = self.realization().topology().vertex_count();
        if anchors[0] == anchors[1] {
            return Err(SurfaceError::CoincidentAnchor.into());
        }
        for &anchor in &anchors {
            if anchor >= vertex_count {
                return Err(SurfaceError::IndexOutside.into());
            }
            if !boundary.contains(&anchor) {
                return Err(SurfaceError::AnchorNotBoundary.into());
            }
        }
        least_squares_conformal_map(
            self,
            anchors,
            realization_limit,
            *executor,
            storage,
            work,
            cancellation,
        )
        .map_err(SurfaceComputationError::Solve)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ScaledNorm {
    scale: f64,
    sum_squares: f64,
}

impl ScaledNorm {
    fn add(&mut self, value: f64) -> Result<(), SolveError> {
        let value = value.abs();
        if !value.is_finite() {
            return Err(SolveError::Numerical);
        }
        if value == 0.0 {
            return Ok(());
        }
        if self.scale < value {
            let ratio = self.scale / value;
            self.sum_squares = 1.0 + self.sum_squares * ratio * ratio;
            self.scale = value;
        } else {
            let ratio = value / self.scale;
            self.sum_squares += ratio * ratio;
        }
        self.sum_squares
            .is_finite()
            .then_some(())
            .ok_or(SolveError::Numerical)
    }

    fn value(self) -> f64 {
        self.scale * self.sum_squares.sqrt()
    }
}

fn least_squares_conformal_map(
    surface: &TriangleSurface,
    anchors: [usize; 2],
    realization_limit: RealizationLimit,
    executor: NativeExecutor,
    storage: StorageLimit,
    work: WorkLimit,
    cancellation: &CancellationToken,
) -> Result<LeastSquaresConformalMapSolution, SolveError> {
    check_cancelled(cancellation)?;
    let vertex_count = surface.realization().topology().vertex_count();
    let face_count = surface.face_count();
    let (rows, columns, block, scratch) = require_least_squares_conformal_resources(
        vertex_count,
        face_count,
        executor,
        storage,
        work,
    )?;
    let mut system =
        assemble_least_squares_conformal_system(surface, anchors, rows, columns, cancellation)?;
    let (observed_rank, condition_indicator) =
        solve_least_squares_conformal_system(&mut system, block, scratch, executor, cancellation)?;
    let positions = least_squares_conformal_positions(vertex_count, anchors, &system)?;
    let residual_bound = least_squares_conformal_residual(surface, &positions, &system)?;
    let (minimum_area, exact_fallback_faces) =
        least_squares_conformal_orientation(surface, &positions)?;
    check_cancelled(cancellation)?;
    let target = EuclideanRealization::admit(
        Arc::clone(surface.realization().topology()),
        2,
        positions,
        realization_limit,
    )
    .map_err(realization_solve_error)?;
    check_cancelled(cancellation)?;
    Ok(LeastSquaresConformalMapSolution::new(
        target,
        LeastSquaresConformalMapEvidence::new(
            columns,
            observed_rank,
            condition_indicator,
            residual_bound,
            minimum_area,
            exact_fallback_faces,
        ),
    ))
}

struct LeastSquaresConformalSystem {
    matrix: Mat<f64>,
    right_hand_side: Vec<f64>,
    free_position: Vec<usize>,
    matrix_norm: f64,
    right_hand_side_norm: f64,
}

fn require_least_squares_conformal_resources(
    vertex_count: usize,
    face_count: usize,
    executor: NativeExecutor,
    storage: StorageLimit,
    work: WorkLimit,
) -> Result<(usize, usize, usize, StackReq), SolveError> {
    let rows = face_count.checked_mul(2).ok_or(SolveError::ResourceLimit)?;
    let columns = vertex_count
        .checked_sub(2)
        .and_then(|value| value.checked_mul(2))
        .ok_or(SolveError::ResourceLimit)?;
    if rows < columns || columns == 0 {
        return Err(SolveError::Factorization);
    }
    let block = qr_factor::recommended_block_size::<f64>(rows, columns).max(1);
    let factor = qr_factor::qr_in_place_scratch::<usize, f64>(
        rows,
        columns,
        block,
        executor.par(),
        Spec::default(),
    );
    let solve = qr_solve::solve_lstsq_in_place_scratch::<usize, f64>(
        rows,
        columns,
        block,
        1,
        executor.par(),
    );
    let scratch = StackReq::any_of(&[factor, solve]);
    let matrix_cells = rows.checked_mul(columns).ok_or(SolveError::ResourceLimit)?;
    let f64_cells = matrix_cells
        .checked_add(
            block
                .checked_mul(columns)
                .ok_or(SolveError::ResourceLimit)?,
        )
        .and_then(|value| value.checked_add(rows))
        .and_then(|value| value.checked_add(vertex_count.checked_mul(2)?))
        .ok_or(SolveError::ResourceLimit)?;
    let usize_cells = columns
        .checked_mul(2)
        .and_then(|value| value.checked_add(vertex_count))
        .ok_or(SolveError::ResourceLimit)?;
    let peak = logical_bytes(f64_cells, size_of::<f64>())?
        .checked_add(logical_bytes(usize_cells, size_of::<usize>())?)
        .and_then(|value| value.checked_add(u64::try_from(scratch.size_bytes()).ok()?))
        .ok_or(SolveError::ResourceLimit)?;
    require_storage(storage, 0, peak)?;
    let factor_work = checked_work_product(matrix_cells, columns)?;
    let remaining = checked_work_product(matrix_cells, 8)?
        .checked_add(checked_work_product(face_count, 64)?)
        .ok_or(SolveError::ResourceLimit)?;
    require_work(
        work,
        factor_work
            .checked_add(remaining)
            .ok_or(SolveError::ResourceLimit)?,
    )?;
    Ok((rows, columns, block, scratch))
}

fn assemble_least_squares_conformal_system(
    surface: &TriangleSurface,
    anchors: [usize; 2],
    rows: usize,
    columns: usize,
    cancellation: &CancellationToken,
) -> Result<LeastSquaresConformalSystem, SolveError> {
    let vertex_count = surface.realization().topology().vertex_count();
    let mut free_position = vec![usize::MAX; vertex_count];
    for (position, vertex) in (0..vertex_count)
        .filter(|vertex| !anchors.contains(vertex))
        .enumerate()
    {
        free_position[vertex] = position;
    }
    let mut matrix = Mat::zeros(rows, columns);
    let mut right_hand_side = vec![0.0; rows];
    let mut matrix_norm = ScaledNorm::default();
    for face in 0..surface.face_count() {
        check_cancelled(cancellation)?;
        let (vertices, coefficients) = surface
            .oriented_local_conformal_coefficients(face)
            .map_err(|_| SolveError::Numerical)?;
        for (vertex, coefficient) in vertices.into_iter().zip(coefficients) {
            add_conformal_corner(
                &mut matrix,
                &mut right_hand_side,
                &mut matrix_norm,
                &free_position,
                anchors[1],
                face,
                vertex,
                coefficient,
            )?;
        }
    }
    let right_hand_side_norm = scaled_norm(&right_hand_side)?;
    let matrix_norm = matrix_norm.value();
    if matrix_norm == 0.0 || right_hand_side_norm == 0.0 {
        return Err(SolveError::Factorization);
    }
    Ok(LeastSquaresConformalSystem {
        matrix,
        right_hand_side,
        free_position,
        matrix_norm,
        right_hand_side_norm,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "one face-corner row contribution"
)]
fn add_conformal_corner(
    matrix: &mut Mat<f64>,
    right_hand_side: &mut [f64],
    matrix_norm: &mut ScaledNorm,
    free_position: &[usize],
    nonzero_anchor: usize,
    face: usize,
    vertex: usize,
    [real, imaginary]: [f64; 2],
) -> Result<(), SolveError> {
    let real_row = 2 * face;
    let imaginary_row = real_row + 1;
    let position = free_position[vertex];
    if position == usize::MAX {
        if vertex == nonzero_anchor {
            right_hand_side[real_row] -= real;
            right_hand_side[imaginary_row] -= imaginary;
        }
        return Ok(());
    }
    let real_column = 2 * position;
    let imaginary_column = real_column + 1;
    let values = [real, -imaginary, imaginary, real];
    matrix[(real_row, real_column)] = values[0];
    matrix[(real_row, imaginary_column)] = values[1];
    matrix[(imaginary_row, real_column)] = values[2];
    matrix[(imaginary_row, imaginary_column)] = values[3];
    values
        .into_iter()
        .try_for_each(|value| matrix_norm.add(value))
}

fn solve_least_squares_conformal_system(
    system: &mut LeastSquaresConformalSystem,
    block: usize,
    scratch: StackReq,
    executor: NativeExecutor,
    cancellation: &CancellationToken,
) -> Result<(usize, f64), SolveError> {
    let rows = system.matrix.nrows();
    let columns = system.matrix.ncols();
    let mut householder = Mat::zeros(block, columns);
    let mut permutation = vec![0_usize; columns];
    let mut inverse_permutation = vec![0_usize; columns];
    let mut memory = MemBuffer::try_new(scratch).map_err(|_| SolveError::Allocation)?;
    check_cancelled(cancellation)?;
    qr_factor::qr_in_place(
        system.matrix.as_mut(),
        householder.as_mut(),
        &mut permutation,
        &mut inverse_permutation,
        executor.par(),
        MemStack::new(&mut memory),
        Spec::default(),
    );
    check_cancelled(cancellation)?;
    let diagonal = (0..columns).map(|index| system.matrix[(index, index)].abs());
    let maximum = diagonal.clone().fold(0.0_f64, f64::max);
    let threshold = f64::EPSILON.sqrt() * maximum;
    let rank = diagonal.clone().filter(|value| *value > threshold).count();
    let minimum = diagonal.fold(f64::INFINITY, f64::min);
    let condition = maximum / minimum;
    if rank != columns || !condition.is_finite() {
        return Err(SolveError::Factorization);
    }
    let mut rhs = MatMut::from_column_major_slice_mut(&mut system.right_hand_side, rows, 1);
    let stack = MemStack::new(&mut memory);
    qr_solve::solve_lstsq_in_place(
        system.matrix.as_ref(),
        householder.as_ref(),
        system.matrix.as_ref(),
        PermRef::new_checked(&permutation, &inverse_permutation, columns),
        rhs.as_mut(),
        executor.par(),
        stack,
    );
    check_cancelled(cancellation)?;
    system.right_hand_side[..columns]
        .iter()
        .all(|value| value.is_finite())
        .then_some((rank, condition))
        .ok_or(SolveError::Numerical)
}

fn least_squares_conformal_positions(
    vertex_count: usize,
    anchors: [usize; 2],
    system: &LeastSquaresConformalSystem,
) -> Result<Vec<f64>, SolveError> {
    let mut positions = vec![0.0; 2 * vertex_count];
    positions[2 * anchors[1]] = 1.0;
    for (vertex, &position) in system.free_position.iter().enumerate() {
        if position != usize::MAX {
            positions[2 * vertex] = system.right_hand_side[2 * position];
            positions[2 * vertex + 1] = system.right_hand_side[2 * position + 1];
        }
    }
    positions
        .iter()
        .all(|value| value.is_finite())
        .then_some(positions)
        .ok_or(SolveError::Numerical)
}

fn least_squares_conformal_residual(
    surface: &TriangleSurface,
    positions: &[f64],
    system: &LeastSquaresConformalSystem,
) -> Result<f64, SolveError> {
    let columns = system.matrix.ncols();
    let solution_norm = scaled_norm(&system.right_hand_side[..columns])?;
    let mut residual_norm = ScaledNorm::default();
    for face in 0..surface.face_count() {
        let (vertices, coefficients) = surface
            .oriented_local_conformal_coefficients(face)
            .map_err(|_| SolveError::Numerical)?;
        for imaginary_part in [false, true] {
            let value = conformal_row_value(vertices, coefficients, positions, imaginary_part)?;
            residual_norm.add(value)?;
        }
    }
    let denominator = system.matrix_norm * solution_norm + system.right_hand_side_norm;
    let residual = residual_norm.value() / denominator;
    residual
        .is_finite()
        .then_some(residual)
        .ok_or(SolveError::Numerical)
}

fn conformal_row_value(
    vertices: [usize; 3],
    coefficients: [[f64; 2]; 3],
    positions: &[f64],
    imaginary_part: bool,
) -> Result<f64, SolveError> {
    adaptive_product_value(vertices.into_iter().zip(coefficients).flat_map(
        |(vertex, [real, imaginary])| {
            if imaginary_part {
                [
                    (imaginary, positions[2 * vertex]),
                    (real, positions[2 * vertex + 1]),
                ]
            } else {
                [
                    (real, positions[2 * vertex]),
                    (-imaginary, positions[2 * vertex + 1]),
                ]
            }
        },
    ))
    .map(|(value, _)| value)
    .ok_or(SolveError::Numerical)
}

fn least_squares_conformal_orientation(
    surface: &TriangleSurface,
    positions: &[f64],
) -> Result<(f64, usize), SolveError> {
    let mut minimum = f64::INFINITY;
    let mut exact_fallback_faces = 0_usize;
    for face in 0..surface.face_count() {
        let (vertices, _) = surface
            .oriented_local_conformal_coefficients(face)
            .map_err(|_| SolveError::Numerical)?;
        let terms = normalized_signed_area_terms(vertices, positions)?;
        let (sign, exact_fallback) =
            adaptive_product_sign(terms.into_iter()).ok_or(SolveError::Numerical)?;
        if sign != CmpOrdering::Greater {
            return Err(SolveError::Numerical);
        }
        exact_fallback_faces += usize::from(exact_fallback);
        let area = adaptive_product_value(terms.into_iter())
            .ok_or(SolveError::Numerical)?
            .0;
        if area <= 0.0 || !area.is_finite() {
            return Err(SolveError::Numerical);
        }
        minimum = minimum.min(area);
    }
    Ok((minimum, exact_fallback_faces))
}

fn normalized_signed_area_terms(
    vertices: [usize; 3],
    positions: &[f64],
) -> Result<[(f64, f64); 2], SolveError> {
    let point = |vertex| [positions[2 * vertex], positions[2 * vertex + 1]];
    let [p0, p1, p2] = vertices.map(point);
    let first = [p1[0] - p0[0], p1[1] - p0[1]];
    let second = [p2[0] - p0[0], p2[1] - p0[1]];
    let scale = first
        .into_iter()
        .chain(second)
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return Err(SolveError::Numerical);
    }
    Ok([
        (first[0] / scale, second[1] / scale),
        (-first[1] / scale, second[0] / scale),
    ])
}

fn scaled_norm(values: &[f64]) -> Result<f64, SolveError> {
    let mut norm = ScaledNorm::default();
    values.iter().try_for_each(|&value| norm.add(value))?;
    Ok(norm.value())
}

fn logical_bytes(count: usize, width: usize) -> Result<u64, SolveError> {
    u64::try_from(count)
        .ok()
        .and_then(|value| value.checked_mul(u64::try_from(width).ok()?))
        .ok_or(SolveError::ResourceLimit)
}

fn checked_work_product(left: usize, right: usize) -> Result<u64, SolveError> {
    u64::try_from(left)
        .ok()
        .and_then(|value| value.checked_mul(u64::try_from(right).ok()?))
        .ok_or(SolveError::ResourceLimit)
}

fn realization_solve_error(error: RealizationError) -> SolveError {
    if error.resource_limit().is_some() {
        SolveError::ResourceLimit
    } else if error == RealizationError::Allocation {
        SolveError::Allocation
    } else {
        SolveError::Numerical
    }
}

fn require_frozen_flow_domain(
    metric: &crate::PositiveMetric,
    time_step: f64,
) -> Result<(), SurfaceError> {
    if !time_step.is_finite() || time_step <= 0.0 {
        return Err(SurfaceError::TimeStep);
    }
    let realization = metric.realization();
    if realization.ambient_dimension() != 3 {
        return Err(SurfaceError::AmbientDimension);
    }
    realization.topology().refine_triangle()?;
    realization.topology().refine_oriented()?;
    realization
        .topology()
        .refine_regular()?
        .without_boundary()
        .map_err(|_| SurfaceError::BoundaryPresent)?;
    realization.topology().refine_connected()?;
    Ok(())
}

fn frozen_flow_step(
    metric: &crate::PositiveMetric,
    time_step: f64,
    realization_limit: RealizationLimit,
    executor: NativeExecutor,
    storage: StorageLimit,
    work: WorkLimit,
    cancellation: &CancellationToken,
) -> Result<FlowStep, SolveError> {
    check_cancelled(cancellation)?;
    let source = metric.realization();
    let n = source.topology().vertex_count();
    let dimension = source.ambient_dimension();
    let cells = n.checked_mul(dimension).ok_or(SolveError::ResourceLimit)?;
    let quadratic = n
        .checked_mul(n)
        .and_then(|value| value.checked_mul(dimension))
        .ok_or(SolveError::ResourceLimit)?;
    let solve_work = u64::try_from(quadratic.max(cells.saturating_mul(8)))
        .map_err(|_| SolveError::ResourceLimit)?;
    let total_work = cubic_work(n)?
        .checked_add(solve_work)
        .ok_or(SolveError::ResourceLimit)?;
    require_work(work, total_work)?;

    let factor_bytes = matrix_bytes(n)?;
    require_storage(storage, 0, factor_bytes.saturating_mul(2))?;
    let free = (0..n).collect::<Vec<_>>();
    let factor = factor_stiffness(
        SystemRef::Parabolic { metric, time_step },
        &free,
        executor,
        cancellation,
    )?;
    check_cancelled(cancellation)?;

    let requirement =
        StackReq::new::<f64>(cells).and(factor_solve_requirement(&factor, executor, dimension));
    let workspace_bytes =
        u64::try_from(requirement.size_bytes()).map_err(|_| SolveError::ResourceLimit)?;
    let peak = factor_bytes
        .saturating_mul(2)
        .checked_add(workspace_bytes)
        .ok_or(SolveError::ResourceLimit)?;
    require_storage(storage, 0, peak)?;
    let mut buffer = MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?;
    let stack = MemStack::new(&mut buffer);
    if !stack.can_hold(requirement) {
        return Err(SolveError::ResourceLimit);
    }

    let masses = metric
        .hodge_coefficients_slice(0)
        .map_err(|_| SolveError::Numerical)?;
    let centroid = weighted_centroid(masses, source.positions(), dimension)?;
    let scale = factor_scale(&factor);
    let (mut rhs_storage, stack) = stack.make_with(cells, |_| 0.0_f64);
    {
        let mut rhs = MatMut::from_column_major_slice_mut(&mut rhs_storage, n, dimension);
        fill_centered_mass_rhs(masses, source.positions(), &centroid, scale, rhs.as_mut())?;
        solve_factor(&factor, rhs.as_mut(), executor, stack);
    }
    check_cancelled(cancellation)?;

    let (_, endpoints) = stiffness_endpoints(metric)?;
    let residual_bound = flow_residual(metric, time_step, &rhs_storage, &centroid, &endpoints)?;
    let mut positions = vec![0.0; cells];
    for vertex in 0..n {
        for axis in 0..dimension {
            positions[vertex * dimension + axis] = rhs_storage[axis * n + vertex] + centroid[axis];
        }
    }
    drop(rhs_storage);
    let target_centroid = weighted_centroid(masses, &positions, dimension)?;
    let centroid_scale = positions
        .iter()
        .chain(&centroid)
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    let centroid_residual_bound = target_centroid
        .iter()
        .zip(&centroid)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
        / centroid_scale;
    let energy_before = dirichlet_energy(metric, source.positions(), dimension, &endpoints)?;
    let energy_after = dirichlet_energy(metric, &positions, dimension, &endpoints)?;
    if !residual_bound.is_finite()
        || !centroid_residual_bound.is_finite()
        || !energy_before.is_finite()
        || !energy_after.is_finite()
        || residual_bound > 1.0e-10
        || centroid_residual_bound > 1.0e-12
        || energy_after > energy_before + 128.0 * f64::EPSILON * energy_before.max(1.0)
    {
        return Err(SolveError::Numerical);
    }
    check_cancelled(cancellation)?;
    let target = source
        .deform(positions, realization_limit)
        .map_err(|_| SolveError::Numerical)?;
    check_cancelled(cancellation)?;
    Ok(FlowStep::new(
        target,
        FlowEvidence::new(
            energy_before,
            energy_after,
            residual_bound,
            centroid_residual_bound,
        ),
    ))
}

impl SolveExt for DirichletProblem {
    fn prepare_with(
        &self,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
    ) -> Result<Prepared<Self>, SolveError> {
        self.prepare_with_cancellation(executor, storage, work, &CancellationToken::new())
    }

    fn prepare_with_cancellation(
        &self,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<Prepared<Self>, SolveError> {
        let selection = problem_selection(self.prescribed())?;
        let free = selection.complement().map_err(|_| SolveError::Numerical)?;
        let retained = matrix_bytes(free.len())?;
        require_storage(storage, retained, retained.saturating_mul(2))?;
        require_work(work, cubic_work(free.len())?)?;
        let factor = factor_general(self.operator(), free.indices(), *executor, cancellation)?;
        Ok(Prepared {
            key: ReuseKey::Dirichlet {
                operator: self.operator().clone(),
                selection: Arc::clone(selection),
            },
            executor: *executor,
            factors: Factors::One(factor),
            family: PhantomData,
        })
    }
}

impl SolveExt for HarmonicExtension {
    fn prepare_with(
        &self,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
    ) -> Result<Prepared<Self>, SolveError> {
        self.prepare_with_cancellation(executor, storage, work, &CancellationToken::new())
    }

    fn prepare_with_cancellation(
        &self,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<Prepared<Self>, SolveError> {
        let selection = problem_selection(self.prescribed())?;
        let free = selection.complement().map_err(|_| SolveError::Numerical)?;
        let retained = matrix_bytes(free.len())?;
        require_storage(storage, retained, retained.saturating_mul(2))?;
        require_work(work, cubic_work(free.len())?)?;
        let factor = if free.is_empty() {
            Factor::Analytic
        } else {
            factor_stiffness(
                SystemRef::PositiveSquare {
                    metric: self.metric(),
                },
                free.indices(),
                *executor,
                cancellation,
            )?
        };
        Ok(Prepared {
            key: ReuseKey::Harmonic {
                owner: Arc::clone(self.metric().realization()),
                selection: Arc::clone(selection),
            },
            executor: *executor,
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
    pub fn workspace_for(
        &self,
        problem: &DirichletProblem,
        storage: StorageLimit,
    ) -> Result<SolveWorkspace, SolveError> {
        self.require_dirichlet(problem)?;
        boundary_workspace(
            one_factor(&self.factors)?,
            self.executor,
            problem.prescribed(),
            storage,
        )
    }

    /// Solve and certify a compatible equation.
    ///
    /// # Errors
    /// Rejects mismatch, resource exhaustion, or failed certification.
    pub fn solve(
        &self,
        problem: &DirichletProblem,
        workspace: &mut SolveWorkspace,
        work: WorkLimit,
    ) -> Result<DirichletSolution, SolveError> {
        self.solve_cancellable(problem, workspace, work, &CancellationToken::new())
    }

    /// Solve with cooperative checks around the backend factor call.
    ///
    /// # Errors
    /// Also rejects cancellation before result publication.
    pub fn solve_cancellable(
        &self,
        problem: &DirichletProblem,
        workspace: &mut SolveWorkspace,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<DirichletSolution, SolveError> {
        self.require_dirichlet(problem)?;
        BoundaryEquation {
            operator: problem.operator(),
            rhs: Some(problem.rhs()),
            prescribed: problem.prescribed(),
            row_weights: None,
        }
        .solve(
            one_factor(&self.factors)?,
            self.executor,
            workspace,
            work,
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
    pub fn workspace_for(
        &self,
        problem: &HarmonicExtension,
        storage: StorageLimit,
    ) -> Result<SolveWorkspace, SolveError> {
        self.require_harmonic(problem)?;
        boundary_workspace(
            one_factor(&self.factors)?,
            self.executor,
            problem.prescribed(),
            storage,
        )
    }

    /// Solve and certify compatible boundary data.
    ///
    /// # Errors
    /// Rejects mismatch, resource exhaustion, or failed certification.
    pub fn solve(
        &self,
        problem: &HarmonicExtension,
        workspace: &mut SolveWorkspace,
        work: WorkLimit,
    ) -> Result<DirichletSolution, SolveError> {
        self.solve_cancellable(problem, workspace, work, &CancellationToken::new())
    }

    /// Solve with cooperative checks around the backend factor call.
    ///
    /// # Errors
    /// Also rejects cancellation before result publication.
    pub fn solve_cancellable(
        &self,
        problem: &HarmonicExtension,
        workspace: &mut SolveWorkspace,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<DirichletSolution, SolveError> {
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
            self.executor,
            workspace,
            work,
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
    fn prepare_with(
        &self,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
    ) -> Result<Prepared<Self>, SolveError> {
        self.prepare_with_cancellation(executor, storage, work, &CancellationToken::new())
    }

    fn prepare_with_cancellation(
        &self,
        executor: &NativeExecutor,
        storage: StorageLimit,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<Prepared<Self>, SolveError> {
        check_cancelled(cancellation)?;
        let degree = self.degree().map_err(|_| SolveError::Numerical)?;
        let topology = self.metric().realization().topology();
        let rows = self.source().coefficients().len();
        let dimension = preflight_hodge(self, degree, *executor, storage, work)?;
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
        let factor_scratch = hodge_factor_scratch_bytes(rows, exact_rank, *executor)?
            .max(hodge_factor_scratch_bytes(rows, coexact_rank, *executor)?);
        let peak = retained
            .checked_add(factor_scratch)
            .ok_or(SolveError::ResourceLimit)?;
        require_storage(storage, retained, peak)?;
        let factor_work = hodge_factor_work(rows, exact_rank)?
            .checked_add(hodge_factor_work(rows, coexact_rank)?)
            .ok_or(SolveError::ResourceLimit)?;
        require_work(work, factor_work)?;
        let exact_factor = factor_hodge_image(
            self,
            exact
                .as_ref()
                .map(|(boundary, pivots)| (*boundary, pivots.as_ref())),
            true,
            *executor,
            cancellation,
        )?;
        let coexact_factor = factor_hodge_image(
            self,
            coexact
                .as_ref()
                .map(|(boundary, pivots)| (*boundary, pivots.as_ref())),
            false,
            *executor,
            cancellation,
        )?;
        check_cancelled(cancellation)?;
        Ok(Prepared {
            key: ReuseKey::Hodge {
                owner: Arc::clone(self.metric().realization()),
                degree,
            },
            executor: *executor,
            factors: Factors::Hodge([exact_factor, coexact_factor]),
            family: PhantomData,
        })
    }
}

fn preflight_hodge(
    problem: &HodgeProblem,
    degree: usize,
    executor: NativeExecutor,
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
    pub fn workspace_for(
        &self,
        problem: &HodgeProblem,
        storage: StorageLimit,
    ) -> Result<SolveWorkspace, SolveError> {
        self.require_hodge(problem)?;
        let requirement = hodge_projection_requirement(&self.factors, self.executor)?;
        let bytes =
            u64::try_from(requirement.size_bytes()).map_err(|_| SolveError::ResourceLimit)?;
        require_storage(storage, 0, bytes)?;
        let buffer = MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?;
        Ok(SolveWorkspace { buffer })
    }

    /// Project, reconstruct, and certify one compatible source cochain.
    ///
    /// # Errors
    ///
    /// Rejects mismatched inputs, insufficient work, or failed numerical certification.
    pub fn solve(
        &self,
        problem: &HodgeProblem,
        workspace: &mut SolveWorkspace,
        work: WorkLimit,
    ) -> Result<HodgeDecomposition, SolveError> {
        self.solve_cancellable(problem, workspace, work, &CancellationToken::new())
    }

    /// Solve with cooperative checks around the two sequential projections.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::solve`], plus cancellation.
    pub fn solve_cancellable(
        &self,
        problem: &HodgeProblem,
        workspace: &mut SolveWorkspace,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<HodgeDecomposition, SolveError> {
        self.require_hodge(problem)?;
        check_cancelled(cancellation)?;
        let requirement = hodge_projection_requirement(&self.factors, self.executor)?;
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
        require_work(work, required)?;
        let weights = problem
            .metric()
            .hodge_coefficients_slice(problem.degree().map_err(|_| SolveError::Numerical)?)
            .map_err(|_| SolveError::Numerical)?;
        let exact_values = project_hodge(
            &self.factors[0],
            problem.source().coefficients(),
            weights,
            self.executor,
            workspace,
        )?;
        check_cancelled(cancellation)?;
        let coexact_values = project_hodge(
            &self.factors[1],
            problem.source().coefficients(),
            weights,
            self.executor,
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
    executor: NativeExecutor,
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
    executor: NativeExecutor,
    workspace: &mut SolveWorkspace,
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
    executor: NativeExecutor,
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

fn factor_dense_square(
    mut matrix: Mat<f64>,
    executor: NativeExecutor,
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

fn require_stable_dense_lu(factor: &Factor) -> Result<(), SolveError> {
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
    executor: NativeExecutor,
    prescribed: &Binary64Cochain,
    storage: StorageLimit,
) -> Result<SolveWorkspace, SolveError> {
    let requirement = boundary_requirement(factor, executor, prescribed)?;
    let bytes = u64::try_from(requirement.size_bytes()).map_err(|_| SolveError::ResourceLimit)?;
    require_storage(storage, 0, bytes)?;
    Ok(SolveWorkspace {
        buffer: MemBuffer::try_new(requirement).map_err(|_| SolveError::Allocation)?,
    })
}

fn boundary_requirement(
    factor: &Factor,
    executor: NativeExecutor,
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
        executor: NativeExecutor,
        workspace: &mut SolveWorkspace,
        work: WorkLimit,
        cancellation: &CancellationToken,
    ) -> Result<DirichletSolution, SolveError> {
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
        Ok(DirichletSolution::new(
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

fn factor_stiffness(
    system: SystemRef<'_>,
    free: &[usize],
    executor: NativeExecutor,
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

fn stiffness_endpoints(
    metric: &crate::PositiveMetric,
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
    executor: NativeExecutor,
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
    let matrix = SparseColMat::try_new_from_triplets(free.len(), free.len(), &triplets).map_err(
        |error| match error {
            CreationError::Generic(FaerError::OutOfMemory) => SolveError::Allocation,
            CreationError::Generic(FaerError::IndexOverflow)
            | CreationError::OutOfBounds { .. } => SolveError::ResourceLimit,
            CreationError::Generic(_) => SolveError::Numerical,
        },
    )?;
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
    executor: NativeExecutor,
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
    _executor: NativeExecutor,
) -> Result<StackReq, SolveError> {
    let requirements = factors
        .iter()
        .map(qr_projection_requirement)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(StackReq::any_of(&requirements))
}

const fn factor_scale(factor: &Factor) -> f64 {
    match factor {
        Factor::Analytic | Factor::DenseQr { .. } => 1.0,
        Factor::Diagonal { scale, .. }
        | Factor::DenseLlt { scale, .. }
        | Factor::SparseLlt { scale, .. }
        | Factor::DenseLu { scale, .. } => *scale,
    }
}

fn factor_solve_requirement(
    factor: &Factor,
    executor: NativeExecutor,
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

fn solve_factor(
    factor: &Factor,
    mut rhs: MatMut<'_, f64>,
    executor: NativeExecutor,
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

fn weighted_centroid(
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

fn fill_centered_mass_rhs(
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

fn dirichlet_energy(
    metric: &crate::PositiveMetric,
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

fn flow_residual(
    metric: &crate::PositiveMetric,
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

fn matrix_bytes(n: usize) -> Result<u64, SolveError> {
    let bytes = n
        .checked_mul(n)
        .and_then(|value| value.checked_mul(size_of::<f64>()))
        .ok_or(SolveError::ResourceLimit)?;
    u64::try_from(bytes).map_err(|_| SolveError::ResourceLimit)
}

fn cubic_work(n: usize) -> Result<u64, SolveError> {
    let value = n
        .checked_mul(n)
        .and_then(|value| value.checked_mul(n))
        .ok_or(SolveError::ResourceLimit)?;
    u64::try_from(value).map_err(|_| SolveError::ResourceLimit)
}

fn solve_work(problem: &MeanZeroPoisson) -> Result<u64, SolveError> {
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

fn certification_requirement(problem: &MeanZeroPoisson) -> Result<StackReq, SolveError> {
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

fn require_storage(limit: StorageLimit, retained: u64, peak: u64) -> Result<(), SolveError> {
    if retained > limit.retained_logical_bytes() || peak > limit.peak_live_logical_bytes() {
        Err(SolveError::ResourceLimit)
    } else {
        Ok(())
    }
}

fn require_work(limit: WorkLimit, required: u64) -> Result<(), SolveError> {
    if required > limit.steps() {
        Err(SolveError::ResourceLimit)
    } else {
        Ok(())
    }
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<(), SolveError> {
    if cancellation.is_cancelled() {
        Err(SolveError::Cancelled)
    } else {
        Ok(())
    }
}
