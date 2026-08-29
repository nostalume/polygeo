use std::fmt;
use std::sync::Arc;

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use once_cell::sync::OnceCell;

use crate::{ComplexCore, ExactRational, StorageLimit, TopologyError, WorkLimit};

type BigRational = ExactRational;

const FORWARD_CONDITION_LIMIT: f64 = 90.5;
const RATIONAL_HEAD_BITS: u64 = 64;
const BINARY64_OVERFLOW_BIT_DIFFERENCE: u64 = 1024;
const BINARY64_UNDERFLOW_BIT_DIFFERENCE: u64 = 1075;

#[derive(Clone)]
struct DenseSquare<T> {
    order: usize,
    values: Box<[T]>,
}

impl<T> DenseSquare<T> {
    fn try_from_fn(
        order: usize,
        mut value: impl FnMut(usize, usize) -> T,
    ) -> Result<Self, RealizationError> {
        let cells = order.checked_mul(order).ok_or(RealizationError::Overflow)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(cells)
            .map_err(|_| RealizationError::Allocation)?;
        for row in 0..order {
            for column in 0..order {
                values.push(value(row, column));
            }
        }
        Ok(Self {
            order,
            values: values.into_boxed_slice(),
        })
    }

    fn row(&self, row: usize) -> &[T] {
        &self.values[row * self.order..(row + 1) * self.order]
    }

    fn row_mut(&mut self, row: usize) -> &mut [T] {
        &mut self.values[row * self.order..(row + 1) * self.order]
    }

    fn disjoint_rows_mut(&mut self, left: usize, right: usize) -> (&mut [T], &mut [T]) {
        debug_assert_ne!(left, right);
        let order = self.order;
        let [left, right] = self
            .values
            .get_disjoint_mut([
                left * order..(left + 1) * order,
                right * order..(right + 1) * order,
            ])
            .expect("distinct admitted dense rows are disjoint");
        (left, right)
    }

    fn swap_rows(&mut self, left: usize, right: usize) {
        if left != right {
            let (left, right) = self.disjoint_rows_mut(left, right);
            left.swap_with_slice(right);
        }
    }
}

/// Logical-storage and exact-arithmetic ceilings for one Euclidean realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealizationLimit {
    storage: StorageLimit,
    coefficient_bits: u64,
    exact_steps: WorkLimit,
}

impl RealizationLimit {
    pub const DEFAULT: Self = Self {
        storage: StorageLimit::new(128 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("default storage lifecycle is valid"),
        coefficient_bits: 65_536,
        exact_steps: WorkLimit::new(100_000_000),
    };

    #[must_use]
    pub const fn new(storage: StorageLimit, coefficient_bits: u64, exact_steps: WorkLimit) -> Self {
        Self {
            storage,
            coefficient_bits,
            exact_steps,
        }
    }

    #[must_use]
    pub const fn storage(self) -> StorageLimit {
        self.storage
    }

    #[must_use]
    pub const fn coefficient_bits(self) -> u64 {
        self.coefficient_bits
    }

    #[must_use]
    pub const fn exact_steps(self) -> WorkLimit {
        self.exact_steps
    }
}

impl Default for RealizationLimit {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Classified realization admission or derived-measure failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RealizationError {
    AmbientDimension,
    PositionShape,
    NonFinite,
    Degenerate,
    Unrepresentable,
    DegreeOutside,
    RetainedLogicalBytes { required: u64, limit: u64 },
    PeakLiveLogicalBytes { required: u64, limit: u64 },
    CoefficientBits { required: u64, limit: u64 },
    ExactSteps { required: u64, limit: u64 },
    Overflow,
    Allocation,
    Topology,
}

impl RealizationError {
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::AmbientDimension => "ambient_dimension",
            Self::PositionShape => "position_shape",
            Self::NonFinite => "non_finite",
            Self::Degenerate => "degenerate",
            Self::Unrepresentable => "unrepresentable",
            Self::DegreeOutside => "degree_outside",
            Self::RetainedLogicalBytes { .. }
            | Self::PeakLiveLogicalBytes { .. }
            | Self::CoefficientBits { .. }
            | Self::ExactSteps { .. } => "resource_limit",
            Self::Overflow => "count_overflow",
            Self::Allocation => "allocation",
            Self::Topology => "topology",
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
            Self::CoefficientBits { required, limit } => {
                Some(("coefficient_bits", required, limit))
            }
            Self::ExactSteps { required, limit } => Some(("exact_steps", required, limit)),
            _ => None,
        }
    }
}

impl fmt::Display for RealizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::AmbientDimension => "ambient dimension must contain the simplicial dimension",
            Self::PositionShape => "positions must have one row per admitted vertex",
            Self::NonFinite => "positions must be finite",
            Self::Degenerate => "simplex is degenerate",
            Self::Unrepresentable => "geometric value is not representable",
            Self::DegreeOutside => "measure degree is outside the complex",
            Self::RetainedLogicalBytes { .. } => "realization retained logical byte limit exceeded",
            Self::PeakLiveLogicalBytes { .. } => {
                "realization peak live logical byte limit exceeded"
            }
            Self::CoefficientBits { .. } => "realization coefficient bit limit exceeded",
            Self::ExactSteps { .. } => "realization exact step limit exceeded",
            Self::Overflow => "realization resource estimate overflowed",
            Self::Allocation => "realization allocation failed",
            Self::Topology => "topology data required by the realization is invalid",
        })
    }
}

impl std::error::Error for RealizationError {}

impl From<TopologyError> for RealizationError {
    fn from(_: TopologyError) -> Self {
        Self::Topology
    }
}

#[derive(Debug, Clone, Copy)]
struct ExactLimit {
    coefficient_bits: u64,
    steps: u64,
}

impl From<RealizationLimit> for ExactLimit {
    fn from(limit: RealizationLimit) -> Self {
        Self {
            coefficient_bits: limit.coefficient_bits(),
            steps: limit.exact_steps().steps(),
        }
    }
}

#[derive(Debug)]
struct ExactUse {
    coefficient_bits: u64,
    exact_steps: u64,
    used: u64,
}

impl ExactUse {
    const fn new(limit: ExactLimit) -> Self {
        Self {
            coefficient_bits: limit.coefficient_bits,
            exact_steps: limit.steps,
            used: 0,
        }
    }

    fn charge(&mut self, steps: u64) -> Result<(), RealizationError> {
        let required = self
            .used
            .checked_add(steps)
            .ok_or(RealizationError::Overflow)?;
        if required > self.exact_steps {
            return Err(RealizationError::ExactSteps {
                required,
                limit: self.exact_steps,
            });
        }
        self.used = required;
        Ok(())
    }

    fn grow(&mut self, bits: u64, steps: u64) -> Result<(), RealizationError> {
        if bits > self.coefficient_bits {
            return Err(RealizationError::CoefficientBits {
                required: bits,
                limit: self.coefficient_bits,
            });
        }
        self.charge(steps)
    }

    fn binary(
        &mut self,
        left: &BigRational,
        right: &BigRational,
        carry: u64,
    ) -> Result<(), RealizationError> {
        let bits = rational_bits(left)
            .checked_add(rational_bits(right))
            .and_then(|bits| bits.checked_add(carry))
            .ok_or(RealizationError::Overflow)?;
        self.grow(bits, 1)
    }
}

fn checked_u64(value: usize) -> Result<u64, RealizationError> {
    u64::try_from(value).map_err(|_| RealizationError::Overflow)
}

fn bytes(count: usize, width: usize) -> Result<u64, RealizationError> {
    checked_u64(count)?
        .checked_mul(checked_u64(width)?)
        .ok_or(RealizationError::Overflow)
}

fn realization_peak_bytes(
    topology: &ComplexCore,
    ambient: usize,
    retained: u64,
    limit: RealizationLimit,
) -> Result<u64, RealizationError> {
    let mut topology_bytes = 0_u64;
    let mut basis_count = 0_usize;
    let mut incidence_count = 0_usize;
    let mut largest_basis = 0_usize;
    for degree in 0..=topology.dimension() {
        let basis = topology.basis(degree)?;
        basis_count = basis_count
            .checked_add(basis.row_count())
            .ok_or(RealizationError::Overflow)?;
        largest_basis = largest_basis.max(basis.row_count());
        if degree > 0 {
            incidence_count = incidence_count
                .checked_add(topology.immediate_faces(degree)?.len())
                .ok_or(RealizationError::Overflow)?;
        }
        topology_bytes = topology_bytes
            .checked_add(bytes(basis.values().len(), size_of::<usize>())?)
            .and_then(|value| value.checked_add(bytes(basis.row_count(), size_of::<i8>()).ok()?))
            .ok_or(RealizationError::Overflow)?;
        let boundary = topology.chain_view().boundary(degree)?;
        let indices = boundary
            .indptr()
            .len()
            .checked_add(boundary.indices().len())
            .ok_or(RealizationError::Overflow)?;
        topology_bytes = topology_bytes
            .checked_add(bytes(indices, size_of::<usize>())?)
            .and_then(|value| {
                value.checked_add(bytes(boundary.indices().len(), size_of::<i8>()).ok()?)
            })
            .ok_or(RealizationError::Overflow)?;
    }
    let degree = topology.dimension();
    let dense = degree
        .checked_mul(degree)
        .ok_or(RealizationError::Overflow)?;
    let degrees = degree.checked_add(1).ok_or(RealizationError::Overflow)?;
    let slots = degrees
        .checked_mul(ambient)
        .and_then(|value| value.checked_add(degree.checked_mul(ambient)?))
        .and_then(|value| value.checked_add(dense.checked_mul(2)?))
        .and_then(|value| value.checked_add(degree))
        .ok_or(RealizationError::Overflow)?;
    let component_bits = limit.coefficient_bits();
    let limbs = component_bits.div_ceil(u64::from(usize::BITS));
    let rational_bytes = checked_u64(size_of::<BigRational>())?
        .checked_add(
            limbs
                .checked_mul(2 * checked_u64(size_of::<usize>())?)
                .ok_or(RealizationError::Overflow)?,
        )
        .ok_or(RealizationError::Overflow)?;
    let rational_scratch = checked_u64(slots)?
        .checked_mul(rational_bytes)
        .ok_or(RealizationError::Overflow)?;
    let dense_f64 = degree
        .checked_mul(ambient)
        .and_then(|value| value.checked_add(dense))
        .and_then(|value| value.checked_add(degree.checked_mul(2)?))
        .ok_or(RealizationError::Overflow)?;
    let metric_scratch = incidence_count
        .checked_mul(size_of::<(usize, f64)>() + size_of::<f64>())
        .and_then(|value| {
            value.checked_add(
                largest_basis
                    .checked_mul(3 * size_of::<f64>() + size_of::<usize>() + size_of::<i8>())?,
            )
        })
        .and_then(|value| value.checked_add(degrees.checked_mul(size_of::<Box<[f64]>>())?))
        .ok_or(RealizationError::Overflow)?;
    let scratch = rational_scratch
        .checked_add(bytes(dense_f64, size_of::<f64>())?)
        .and_then(|value| value.checked_add(checked_u64(metric_scratch).ok()?))
        .and_then(|value| value.checked_add(bytes(basis_count, size_of::<f64>()).ok()?))
        .ok_or(RealizationError::Overflow)?;
    retained
        .checked_add(topology_bytes)
        .and_then(|value| value.checked_add(scratch))
        .ok_or(RealizationError::Overflow)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MetricClassification {
    Degenerate { degree: usize, index: usize },
    Indefinite,
    Positive,
}

struct MetricRows {
    measures: Box<[Box<[f64]>]>,
    signs: Box<[Box<[i8]>]>,
    coefficients: Result<Box<[Box<[f64]>]>, RealizationError>,
    classification: MetricClassification,
}

/// Failure to refine a finite circumcentric pairing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MetricError {
    Degenerate { degree: usize, index: usize },
    Indefinite,
}

impl fmt::Display for MetricError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Degenerate { .. } => "circumcentric pairing is degenerate",
            Self::Indefinite => "circumcentric pairing is not positive",
        })
    }
}

impl std::error::Error for MetricError {}

mod metric_sealed {
    pub trait PairingCapability {}
    pub trait NondegenerateCapability {}
}

/// Borrowing operations shared by finite circumcentric pairing refinements.
pub trait PairingCapability: metric_sealed::PairingCapability {
    #[must_use]
    fn realization(&self) -> &Arc<EuclideanRealization>;

    /// Borrow conventional diagonal Hodge coefficients without copying rows.
    ///
    /// # Errors
    ///
    /// Returns an unrepresented-degree failure.
    fn hodge_coefficients_slice(&self, degree: usize) -> Result<&[f64], RealizationError> {
        self.realization()
            .metric_rows()?
            .coefficients
            .as_ref()
            .map_err(|error| *error)?
            .get(degree)
            .map(AsRef::as_ref)
            .ok_or(RealizationError::DegreeOutside)
    }

    /// Metric-induced identification from cochains to chains in the primal basis.
    ///
    /// # Errors
    ///
    /// Returns an unavailable metric row or retained-topology failure.
    fn riesz(
        &self,
        degree: usize,
    ) -> Result<crate::LinearOperator<crate::Cochain, crate::Chain>, crate::OperatorError> {
        crate::LinearOperator::riesz(Arc::clone(self.realization()), degree)
    }
}

/// Borrowing operations requiring a nondegenerate circumcentric pairing.
pub trait NondegenerateCapability:
    PairingCapability + metric_sealed::NondegenerateCapability
{
    /// Inverse metric-induced identification from chains to cochains.
    ///
    /// # Errors
    ///
    /// Returns an unavailable metric row or retained-topology failure.
    fn inverse_riesz(
        &self,
        degree: usize,
    ) -> Result<crate::LinearOperator<crate::Chain, crate::Cochain>, crate::OperatorError> {
        crate::LinearOperator::inverse_riesz(Arc::clone(self.realization()), degree)
    }

    /// Metric codifferential from degree `k` to degree `k - 1`.
    ///
    /// # Errors
    ///
    /// Returns a degree, metric-row, or retained-topology failure.
    fn codifferential(
        &self,
        degree: usize,
    ) -> Result<crate::LinearOperator<crate::Cochain, crate::Cochain>, crate::OperatorError> {
        crate::LinearOperator::codifferential(Arc::clone(self.realization()), degree)
    }

    /// Metric Hodge Laplacian on one cochain degree.
    ///
    /// # Errors
    ///
    /// Returns a degree, metric-row, or retained-topology failure.
    fn laplacian(
        &self,
        degree: usize,
    ) -> Result<crate::LinearOperator<crate::Cochain, crate::Cochain>, crate::OperatorError> {
        crate::LinearOperator::laplacian(Arc::clone(self.realization()), degree)
    }
}

/// Finite signed or zero circumcentric pairing over one realization.
#[derive(Clone, Debug)]
pub struct CircumcentricPairing {
    realization: Arc<EuclideanRealization>,
}

/// Nondegenerate, possibly indefinite circumcentric pairing.
#[derive(Clone, Debug)]
pub struct NondegeneratePairing {
    realization: Arc<EuclideanRealization>,
}

/// Strictly positive circumcentric metric.
///
/// Refinement evidence cannot be forged outside this crate:
///
/// ```compile_fail
/// use std::sync::Arc;
/// use polygeo_core::{EuclideanRealization, PositiveMetric};
/// fn forge(realization: Arc<EuclideanRealization>) -> PositiveMetric {
///     PositiveMetric { realization }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct PositiveMetric {
    realization: Arc<EuclideanRealization>,
}

impl CircumcentricPairing {
    /// Require a pairing with no zero coefficient.
    ///
    /// # Errors
    ///
    /// Returns the first zero coefficient as a bounded counterexample.
    pub fn require_nondegenerate(self) -> Result<NondegeneratePairing, MetricError> {
        self.try_into()
    }

    /// Require a strictly positive metric.
    ///
    /// # Errors
    ///
    /// Returns a zero counterexample or indefinite classification.
    pub fn require_positive(self) -> Result<PositiveMetric, MetricError> {
        self.try_into()
    }
}

impl PairingCapability for CircumcentricPairing {
    fn realization(&self) -> &Arc<EuclideanRealization> {
        &self.realization
    }
}
impl metric_sealed::PairingCapability for CircumcentricPairing {}

impl PairingCapability for NondegeneratePairing {
    fn realization(&self) -> &Arc<EuclideanRealization> {
        &self.realization
    }
}
impl metric_sealed::PairingCapability for NondegeneratePairing {}
impl metric_sealed::NondegenerateCapability for NondegeneratePairing {}
impl NondegenerateCapability for NondegeneratePairing {}

impl PairingCapability for PositiveMetric {
    fn realization(&self) -> &Arc<EuclideanRealization> {
        &self.realization
    }
}
impl metric_sealed::PairingCapability for PositiveMetric {}
impl metric_sealed::NondegenerateCapability for PositiveMetric {}
impl NondegenerateCapability for PositiveMetric {}

impl TryFrom<CircumcentricPairing> for NondegeneratePairing {
    type Error = MetricError;

    fn try_from(pairing: CircumcentricPairing) -> Result<Self, Self::Error> {
        match pairing.realization.metric_classification() {
            MetricClassification::Degenerate { degree, index } => {
                Err(MetricError::Degenerate { degree, index })
            }
            MetricClassification::Indefinite | MetricClassification::Positive => Ok(Self {
                realization: pairing.realization,
            }),
        }
    }
}

impl TryFrom<CircumcentricPairing> for PositiveMetric {
    type Error = MetricError;

    fn try_from(pairing: CircumcentricPairing) -> Result<Self, Self::Error> {
        match pairing.realization.metric_classification() {
            MetricClassification::Positive => Ok(Self {
                realization: pairing.realization,
            }),
            MetricClassification::Degenerate { degree, index } => {
                Err(MetricError::Degenerate { degree, index })
            }
            MetricClassification::Indefinite => Err(MetricError::Indefinite),
        }
    }
}

impl TryFrom<NondegeneratePairing> for PositiveMetric {
    type Error = MetricError;

    fn try_from(pairing: NondegeneratePairing) -> Result<Self, Self::Error> {
        if pairing.realization.metric_classification() == MetricClassification::Positive {
            Ok(Self {
                realization: pairing.realization,
            })
        } else {
            Err(MetricError::Indefinite)
        }
    }
}

impl From<PositiveMetric> for NondegeneratePairing {
    fn from(metric: PositiveMetric) -> Self {
        Self {
            realization: metric.realization,
        }
    }
}

impl From<PositiveMetric> for CircumcentricPairing {
    fn from(metric: PositiveMetric) -> Self {
        Self {
            realization: metric.realization,
        }
    }
}

impl From<NondegeneratePairing> for CircumcentricPairing {
    fn from(pairing: NondegeneratePairing) -> Self {
        Self {
            realization: pairing.realization,
        }
    }
}

/// Immutable Euclidean realization bound to one admitted simplicial owner.
pub struct EuclideanRealization {
    topology: Arc<ComplexCore>,
    ambient_dimension: usize,
    positions: Box<[f64]>,
    primal: Box<[Box<[f64]>]>,
    exact_limit: ExactLimit,
    metric: OnceCell<MetricRows>,
}

impl fmt::Debug for EuclideanRealization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EuclideanRealization")
            .field("dimension", &self.topology.dimension())
            .field("ambient_dimension", &self.ambient_dimension)
            .field("vertex_count", &self.topology.vertex_count())
            .finish_non_exhaustive()
    }
}

impl EuclideanRealization {
    /// Admit one owned entity-major position buffer and all primal measures.
    ///
    /// # Errors
    ///
    /// Rejects invalid shape, dimension, finiteness, degeneracy,
    /// representability, allocation, or retained-memory admission.
    pub fn admit(
        topology: Arc<ComplexCore>,
        ambient_dimension: usize,
        positions: Vec<f64>,
        limit: RealizationLimit,
    ) -> Result<Arc<Self>, RealizationError> {
        if ambient_dimension < topology.dimension() {
            return Err(RealizationError::AmbientDimension);
        }
        let position_count = topology
            .vertex_count()
            .checked_mul(ambient_dimension)
            .ok_or(RealizationError::Overflow)?;
        if positions.len() != position_count {
            return Err(RealizationError::PositionShape);
        }
        if positions.iter().any(|value| !value.is_finite()) {
            return Err(RealizationError::NonFinite);
        }

        let basis_count = (0..=topology.dimension())
            .try_fold(0_usize, |total, degree| {
                total.checked_add(topology.basis(degree).ok()?.row_count())
            })
            .ok_or(RealizationError::Overflow)?;
        let retained_f64 = position_count
            .checked_add(
                basis_count
                    .checked_mul(3)
                    .ok_or(RealizationError::Overflow)?,
            )
            .ok_or(RealizationError::Overflow)?;
        let sign_bytes = basis_count
            .checked_mul(size_of::<i8>())
            .ok_or(RealizationError::Overflow)?;
        let outer_bytes = topology
            .dimension()
            .checked_add(1)
            .and_then(|rows| rows.checked_mul(4))
            .and_then(|rows| rows.checked_mul(size_of::<Box<[f64]>>()))
            .ok_or(RealizationError::Overflow)?;
        let retained_logical_bytes = retained_f64
            .checked_mul(size_of::<f64>())
            .and_then(|bytes| bytes.checked_add(sign_bytes))
            .and_then(|bytes| bytes.checked_add(outer_bytes))
            .ok_or(RealizationError::Overflow)?;
        let retained_logical_bytes =
            u64::try_from(retained_logical_bytes).map_err(|_| RealizationError::Overflow)?;
        let storage = limit.storage();
        if retained_logical_bytes > storage.retained_logical_bytes() {
            return Err(RealizationError::RetainedLogicalBytes {
                required: retained_logical_bytes,
                limit: storage.retained_logical_bytes(),
            });
        }
        let peak_bytes =
            realization_peak_bytes(&topology, ambient_dimension, retained_logical_bytes, limit)?;
        if peak_bytes > storage.peak_live_logical_bytes() {
            return Err(RealizationError::PeakLiveLogicalBytes {
                required: peak_bytes,
                limit: storage.peak_live_logical_bytes(),
            });
        }
        let exact_limit = limit.into();
        let mut exact = ExactUse::new(exact_limit);
        let primal = compute_primal(&topology, ambient_dimension, &positions, &mut exact)?;
        Ok(Arc::new(Self {
            topology,
            ambient_dimension,
            positions: positions.into_boxed_slice(),
            primal,
            exact_limit,
            metric: OnceCell::new(),
        }))
    }

    /// Admit candidate positions as a new realization over the same topology.
    ///
    /// The source is immutable and no target is published unless full
    /// realization admission succeeds.
    ///
    /// # Errors
    /// Rejects the same shape, finiteness, degeneracy, or resource failures as
    /// [`Self::admit`].
    pub fn deform(
        &self,
        positions: Vec<f64>,
        limit: RealizationLimit,
    ) -> Result<Arc<Self>, RealizationError> {
        Self::admit(
            Arc::clone(&self.topology),
            self.ambient_dimension,
            positions,
            limit,
        )
    }

    #[must_use]
    pub const fn topology(&self) -> &Arc<ComplexCore> {
        &self.topology
    }

    #[must_use]
    pub const fn ambient_dimension(&self) -> usize {
        self.ambient_dimension
    }

    #[must_use]
    pub const fn positions(&self) -> &[f64] {
        &self.positions
    }

    /// Borrow one immutable primal-measure row.
    ///
    /// # Errors
    ///
    /// Returns [`RealizationError::DegreeOutside`] for an unrepresented degree.
    pub fn primal_measures(&self, degree: usize) -> Result<&[f64], RealizationError> {
        self.primal
            .get(degree)
            .map(AsRef::as_ref)
            .ok_or(RealizationError::DegreeOutside)
    }

    /// Borrow one immutable, lazily published dual-measure row.
    ///
    /// # Errors
    ///
    /// Rejects an unrepresented degree or an unrepresentable dual recurrence.
    pub fn dual_measures(&self, degree: usize) -> Result<&[f64], RealizationError> {
        self.metric_rows()?
            .measures
            .get(degree)
            .map(AsRef::as_ref)
            .ok_or(RealizationError::DegreeOutside)
    }

    /// Borrow certified represented signs aligned with one dual-measure row.
    ///
    /// # Errors
    ///
    /// Rejects an unrepresented degree or an unrepresentable dual recurrence.
    pub fn dual_signs(&self, degree: usize) -> Result<&[i8], RealizationError> {
        self.metric_rows()?
            .signs
            .get(degree)
            .map(AsRef::as_ref)
            .ok_or(RealizationError::DegreeOutside)
    }

    /// Admit the finite circumcentric pairing derived from this realization.
    ///
    /// # Errors
    ///
    /// Returns a resource, topology, or representability failure before publishing a handle.
    pub fn circumcentric_pairing(
        self: &Arc<Self>,
    ) -> Result<CircumcentricPairing, RealizationError> {
        self.metric_rows()?
            .coefficients
            .as_ref()
            .map_err(|error| *error)?;
        Ok(CircumcentricPairing {
            realization: Arc::clone(self),
        })
    }

    pub(crate) fn hodge_coefficients(&self, degree: usize) -> Result<&[f64], RealizationError> {
        self.metric_rows()?
            .coefficients
            .as_ref()
            .map_err(|error| *error)?
            .get(degree)
            .map(AsRef::as_ref)
            .ok_or(RealizationError::DegreeOutside)
    }

    fn metric_rows(&self) -> Result<&MetricRows, RealizationError> {
        self.metric.get_or_try_init(|| {
            let mut exact = ExactUse::new(self.exact_limit);
            compute_metric(self, &mut exact)
        })
    }

    fn metric_classification(&self) -> MetricClassification {
        self.metric
            .get()
            .expect("a refinement handle owns published metric rows")
            .classification
    }
}

fn compute_primal(
    topology: &ComplexCore,
    ambient: usize,
    positions: &[f64],
    exact: &mut ExactUse,
) -> Result<Box<[Box<[f64]>]>, RealizationError> {
    let mut rows = Vec::new();
    rows.try_reserve_exact(topology.dimension() + 1)
        .map_err(|_| RealizationError::Allocation)?;
    rows.push(vec![1.0; topology.vertex_count()].into_boxed_slice());
    for degree in 1..=topology.dimension() {
        let basis = topology.basis(degree)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(basis.row_count())
            .map_err(|_| RealizationError::Allocation)?;
        let mut deferred = None;
        for simplex in basis.values().chunks_exact(degree + 1) {
            match simplex_measure(simplex, positions, ambient, exact) {
                Ok(value) => values.push(value),
                Err(RealizationError::Degenerate) => return Err(RealizationError::Degenerate),
                Err(error) => {
                    deferred.get_or_insert(error);
                    values.push(0.0);
                }
            }
        }
        if let Some(error) = deferred {
            return Err(error);
        }
        rows.push(values.into_boxed_slice());
    }
    Ok(rows.into_boxed_slice())
}

fn simplex_measure(
    simplex: &[usize],
    positions: &[f64],
    ambient: usize,
    exact: &mut ExactUse,
) -> Result<f64, RealizationError> {
    let degree = simplex.len() - 1;
    let (normalized, scales) = normalized_edges(simplex, positions, ambient)?;
    let diagonals = qr_diagonals(&normalized, ambient, degree);
    let maximum = diagonals.iter().copied().fold(0.0, f64::max);
    let minimum = diagonals.iter().copied().fold(f64::INFINITY, f64::min);
    let suspicion = f64::EPSILON * float(ambient.max(degree)) * maximum * 16.0;
    if minimum <= suspicion || !minimum.is_finite() || maximum / minimum > FORWARD_CONDITION_LIMIT {
        return exact_measure(simplex, positions, ambient, exact);
    }
    let mut gram = gram_matrix(&normalized, ambient, degree)?;
    let determinant = determinant_f64(&mut gram);
    if determinant <= 0.0 || !determinant.is_finite() {
        return exact_measure(simplex, positions, ambient, exact);
    }
    scaled_product(
        scales.iter().copied().chain([determinant.sqrt()]),
        (2..=degree).map(float),
        RealizationError::Unrepresentable,
    )
}

fn determinant_f64(matrix: &mut DenseSquare<f64>) -> f64 {
    let mut determinant = 1.0;
    let mut negative = false;
    for column in 0..matrix.order {
        let Some(pivot) = (column..matrix.order).max_by(|left, right| {
            matrix.row(*left)[column]
                .abs()
                .total_cmp(&matrix.row(*right)[column].abs())
        }) else {
            return 0.0;
        };
        if matrix.row(pivot)[column] == 0.0 {
            return 0.0;
        }
        if pivot != column {
            matrix.swap_rows(pivot, column);
            negative = !negative;
        }
        let pivot_value = matrix.row(column)[column];
        determinant *= pivot_value;
        for row in column + 1..matrix.order {
            let (target, pivot) = matrix.disjoint_rows_mut(row, column);
            let factor = target[column] / pivot_value;
            for (target, pivot) in target[column + 1..].iter_mut().zip(&pivot[column + 1..]) {
                *target -= factor * pivot;
            }
        }
    }
    if negative { -determinant } else { determinant }
}

fn normalized_edges(
    simplex: &[usize],
    positions: &[f64],
    ambient: usize,
) -> Result<(Vec<f64>, Vec<f64>), RealizationError> {
    let degree = simplex.len() - 1;
    let base = simplex[0] * ambient;
    let mut edges = vec![0.0; ambient * degree];
    let mut scales = vec![0.0_f64; degree];
    for column in 0..degree {
        let vertex = simplex[column + 1] * ambient;
        for coordinate in 0..ambient {
            let value = positions[vertex + coordinate] - positions[base + coordinate];
            if !value.is_finite() {
                return Err(RealizationError::Unrepresentable);
            }
            edges[column * ambient + coordinate] = value;
            scales[column] = scales[column].max(value.abs());
        }
        if scales[column] == 0.0 {
            return Err(RealizationError::Degenerate);
        }
        for coordinate in 0..ambient {
            edges[column * ambient + coordinate] /= scales[column];
        }
    }
    Ok((edges, scales))
}

fn qr_diagonals(edges: &[f64], ambient: usize, degree: usize) -> Vec<f64> {
    let mut orthogonal = edges.to_vec();
    let mut diagonal = vec![0.0; degree];
    for column in 0..degree {
        for previous in 0..column {
            let projection = dot_columns(&orthogonal, previous, &orthogonal, column, ambient);
            for coordinate in 0..ambient {
                orthogonal[column * ambient + coordinate] -=
                    projection * orthogonal[previous * ambient + coordinate];
            }
        }
        let norm = dot_columns(&orthogonal, column, &orthogonal, column, ambient).sqrt();
        diagonal[column] = norm;
        if norm != 0.0 && norm.is_finite() {
            for coordinate in 0..ambient {
                orthogonal[column * ambient + coordinate] /= norm;
            }
        }
    }
    diagonal
}

fn dot_columns(
    left: &[f64],
    left_column: usize,
    right: &[f64],
    right_column: usize,
    rows: usize,
) -> f64 {
    (0..rows)
        .map(|row| left[left_column * rows + row] * right[right_column * rows + row])
        .sum()
}

fn gram_matrix(
    edges: &[f64],
    ambient: usize,
    degree: usize,
) -> Result<DenseSquare<f64>, RealizationError> {
    DenseSquare::try_from_fn(degree, |left, right| {
        dot_columns(edges, left, edges, right, ambient)
    })
}

fn rational_bits(value: &BigRational) -> u64 {
    value
        .numer()
        .magnitude()
        .bits()
        .max(value.denom().magnitude().bits())
}

fn metered_exact_from_binary64(
    value: f64,
    exact: &mut ExactUse,
) -> Result<BigRational, RealizationError> {
    let exponent = ((value.to_bits() >> 52) & 0x7ff) as i32;
    let bits = if value == 0.0 {
        0
    } else if exponent == 0 {
        1075
    } else {
        u64::from((exponent - 1023 - 52).unsigned_abs()).saturating_add(53)
    };
    exact.grow(bits, 1)?;
    Ok(exact_from_binary64(value))
}

fn exact_add(
    left: &BigRational,
    right: &BigRational,
    exact: &mut ExactUse,
) -> Result<BigRational, RealizationError> {
    exact.binary(left, right, 1)?;
    Ok(left + right)
}

fn exact_sub(
    left: &BigRational,
    right: &BigRational,
    exact: &mut ExactUse,
) -> Result<BigRational, RealizationError> {
    exact.binary(left, right, 1)?;
    Ok(left - right)
}

fn exact_mul(
    left: &BigRational,
    right: &BigRational,
    exact: &mut ExactUse,
) -> Result<BigRational, RealizationError> {
    exact.binary(left, right, 0)?;
    Ok(left * right)
}

fn exact_div(
    left: &BigRational,
    right: &BigRational,
    exact: &mut ExactUse,
) -> Result<BigRational, RealizationError> {
    exact.binary(left, right, 0)?;
    Ok(left / right)
}

fn exact_points(
    simplex: &[usize],
    positions: &[f64],
    ambient: usize,
    exact: &mut ExactUse,
) -> Result<Box<[BigRational]>, RealizationError> {
    let mut points = Vec::new();
    points
        .try_reserve_exact(
            simplex
                .len()
                .checked_mul(ambient)
                .ok_or(RealizationError::Overflow)?,
        )
        .map_err(|_| RealizationError::Allocation)?;
    for &vertex in simplex {
        points.extend(
            positions[vertex * ambient..(vertex + 1) * ambient]
                .iter()
                .copied()
                .map(|value| metered_exact_from_binary64(value, exact))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok(points.into_boxed_slice())
}

fn exact_gram(
    simplex: &[usize],
    positions: &[f64],
    ambient: usize,
    exact: &mut ExactUse,
) -> Result<DenseSquare<BigRational>, RealizationError> {
    let degree = simplex.len() - 1;
    let points = exact_points(simplex, positions, ambient, exact)?;
    let mut edges = Vec::new();
    edges
        .try_reserve_exact(
            degree
                .checked_mul(ambient)
                .ok_or(RealizationError::Overflow)?,
        )
        .map_err(|_| RealizationError::Allocation)?;
    for vertex in 1..=degree {
        for coordinate in 0..ambient {
            edges.push(exact_sub(
                &points[vertex * ambient + coordinate],
                &points[coordinate],
                exact,
            )?);
        }
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(
            degree
                .checked_mul(degree)
                .ok_or(RealizationError::Overflow)?,
        )
        .map_err(|_| RealizationError::Allocation)?;
    for left in 0..degree {
        for right in 0..degree {
            exact.grow(1, 1)?;
            let mut sum = BigRational::zero();
            for coordinate in 0..ambient {
                let product = exact_mul(
                    &edges[left * ambient + coordinate],
                    &edges[right * ambient + coordinate],
                    exact,
                )?;
                sum = exact_add(&sum, &product, exact)?;
            }
            exact.charge(1)?;
            values.push(sum);
        }
    }
    Ok(DenseSquare {
        order: degree,
        values: values.into_boxed_slice(),
    })
}

fn exact_measure(
    simplex: &[usize],
    positions: &[f64],
    ambient: usize,
    exact: &mut ExactUse,
) -> Result<f64, RealizationError> {
    let degree = simplex.len() - 1;
    let determinant = exact_determinant(exact_gram(simplex, positions, ambient, exact)?, exact)?;
    exact.charge(1)?;
    if !determinant.is_positive() {
        return Err(RealizationError::Degenerate);
    }
    let factorial = (2..=degree).fold(1.0, |product, value| product * float(value));
    let measure = metered_binary64_sqrt(&determinant, exact)? / factorial;
    if measure.is_finite() && measure > 0.0 {
        Ok(measure)
    } else {
        Err(RealizationError::Unrepresentable)
    }
}

fn exact_determinant(
    mut matrix: DenseSquare<BigRational>,
    exact: &mut ExactUse,
) -> Result<BigRational, RealizationError> {
    exact.grow(1, 1)?;
    let mut determinant = BigRational::one();
    let mut negative = false;
    for column in 0..matrix.order {
        let mut pivot = None;
        for row in column..matrix.order {
            exact.charge(1)?;
            if !matrix.row(row)[column].is_zero() {
                pivot = Some(row);
                break;
            }
        }
        let Some(pivot) = pivot else {
            exact.grow(1, 1)?;
            return Ok(BigRational::zero());
        };
        if pivot != column {
            matrix.swap_rows(pivot, column);
            negative = !negative;
        }
        let pivot_value = matrix.row(column)[column].clone();
        determinant = exact_mul(&determinant, &pivot_value, exact)?;
        exact.charge(1)?;
        for row in column + 1..matrix.order {
            let (target, pivot) = matrix.disjoint_rows_mut(row, column);
            let factor = exact_div(&target[column], &pivot_value, exact)?;
            for (target, pivot) in target[column..].iter_mut().zip(&pivot[column..]) {
                let product = exact_mul(&factor, pivot, exact)?;
                let next = exact_sub(target, &product, exact)?;
                exact.charge(1)?;
                *target = next;
            }
        }
    }
    Ok(if negative { -determinant } else { determinant })
}

fn compute_metric(
    realization: &EuclideanRealization,
    exact: &mut ExactUse,
) -> Result<MetricRows, RealizationError> {
    let dimension = realization.topology.dimension();
    let mut steps = Vec::new();
    steps
        .try_reserve_exact(dimension + 1)
        .map_err(|_| RealizationError::Allocation)?;
    steps.push(Box::default());
    for degree in 1..=dimension {
        steps.push(circumcentric_steps(realization, degree, exact)?);
    }

    let mut measures = vec![Box::<[f64]>::default(); dimension + 1];
    let mut signs = vec![Box::<[i8]>::default(); dimension + 1];
    let top_count = realization.topology.basis(dimension)?.row_count();
    measures[dimension] = vec![1.0; top_count].into_boxed_slice();
    signs[dimension] = vec![1; top_count].into_boxed_slice();
    for lower_degree in (0..dimension).rev() {
        let upper_degree = lower_degree + 1;
        let faces = realization.topology.immediate_faces(upper_degree)?;
        let width = upper_degree + 1;
        let lower_count = realization.topology.basis(lower_degree)?.row_count();
        let mut contributions = Vec::new();
        contributions
            .try_reserve_exact(faces.len())
            .map_err(|_| RealizationError::Allocation)?;
        for (upper_index, upper_value) in measures[upper_degree].iter().copied().enumerate() {
            for local in 0..width {
                let offset = upper_index * width + local;
                contributions.push((
                    faces[offset],
                    scaled_pair(
                        steps[upper_degree][offset],
                        upper_value,
                        float(dimension - lower_degree),
                        RealizationError::Unrepresentable,
                    )?,
                ));
            }
        }
        let (next, next_signs) = sum_incident(&contributions, lower_count, exact)?;
        measures[lower_degree] = next.into_boxed_slice();
        signs[lower_degree] = next_signs.into_boxed_slice();
    }
    let classification = classify_metric(&signs);
    let coefficients = match metric_coefficients(&measures, &realization.primal) {
        Err(RealizationError::Unrepresentable) => Err(RealizationError::Unrepresentable),
        Err(error) => return Err(error),
        Ok(coefficients) => Ok(coefficients),
    };
    Ok(MetricRows {
        measures: measures.into_boxed_slice(),
        signs: signs.into_boxed_slice(),
        coefficients,
        classification,
    })
}

fn metric_coefficients(
    dual_rows: &[Box<[f64]>],
    primal_rows: &[Box<[f64]>],
) -> Result<Box<[Box<[f64]>]>, RealizationError> {
    let mut rows = Vec::new();
    rows.try_reserve_exact(dual_rows.len())
        .map_err(|_| RealizationError::Allocation)?;
    for (dual, primal) in dual_rows.iter().zip(primal_rows) {
        let mut row = Vec::new();
        row.try_reserve_exact(dual.len())
            .map_err(|_| RealizationError::Allocation)?;
        for (dual, primal) in dual.iter().zip(primal) {
            row.push(scaled_pair(
                *dual,
                1.0,
                *primal,
                RealizationError::Unrepresentable,
            )?);
        }
        rows.push(row.into_boxed_slice());
    }
    Ok(rows.into_boxed_slice())
}

fn classify_metric(signs: &[Box<[i8]>]) -> MetricClassification {
    for (degree, row) in signs.iter().enumerate() {
        if let Some(index) = row.iter().position(|sign| *sign == 0) {
            return MetricClassification::Degenerate { degree, index };
        }
    }
    if signs.iter().flatten().any(|sign| *sign < 0) {
        MetricClassification::Indefinite
    } else {
        MetricClassification::Positive
    }
}

fn circumcentric_steps(
    realization: &EuclideanRealization,
    degree: usize,
    exact: &mut ExactUse,
) -> Result<Box<[f64]>, RealizationError> {
    let basis = realization.topology.basis(degree)?;
    let faces = realization.topology.immediate_faces(degree)?;
    let mut steps = Vec::new();
    steps
        .try_reserve_exact(faces.len())
        .map_err(|_| RealizationError::Allocation)?;
    for (upper, simplex) in basis.values().chunks_exact(degree + 1).enumerate() {
        let barycentric = circumcenter_barycentric(
            simplex,
            &realization.positions,
            realization.ambient_dimension,
            exact,
        )?;
        for local in 0..=degree {
            let lower = faces[upper * (degree + 1) + local];
            let height = scaled_pair(
                float(degree),
                realization.primal[degree][upper],
                realization.primal[degree - 1][lower],
                RealizationError::Unrepresentable,
            )?;
            steps.push(exact_scaled_if_needed(barycentric[local], height)?);
        }
    }
    Ok(steps.into_boxed_slice())
}

fn circumcenter_barycentric(
    simplex: &[usize],
    positions: &[f64],
    ambient: usize,
    exact: &mut ExactUse,
) -> Result<Vec<f64>, RealizationError> {
    let degree = simplex.len() - 1;
    let (edges, scales) = normalized_edges(simplex, positions, ambient)?;
    let diagonals = qr_diagonals(&edges, ambient, degree);
    let maximum = diagonals.iter().copied().fold(0.0, f64::max);
    let minimum = diagonals.iter().copied().fold(f64::INFINITY, f64::min);
    let suspicion = f64::EPSILON * float(ambient.max(degree)) * maximum * 64.0;
    if minimum <= suspicion || maximum / minimum > FORWARD_CONDITION_LIMIT {
        return exact_barycentric(simplex, positions, ambient, exact);
    }

    let mut gram = gram_matrix(&edges, ambient, degree)?;
    let mut right = (0..degree)
        .map(|index| 0.5 * scales[index] * gram.row(index)[index])
        .collect::<Vec<_>>();
    if solve_f64(&mut gram, &mut right).is_err() {
        return exact_barycentric(simplex, positions, ambient, exact);
    }
    let mut barycentric = Vec::with_capacity(degree + 1);
    barycentric.push(1.0 - right.iter().zip(&scales).map(|(v, s)| v / s).sum::<f64>());
    barycentric.extend(
        right
            .iter()
            .zip(&scales)
            .map(|(value, scale)| value / scale),
    );
    let sign_limit = f64::EPSILON
        * barycentric
            .iter()
            .copied()
            .map(f64::abs)
            .sum::<f64>()
            .max(1.0)
        * 64.0;
    if barycentric
        .iter()
        .all(|value| value.is_finite() && value.abs() > sign_limit)
    {
        Ok(barycentric)
    } else {
        exact_barycentric(simplex, positions, ambient, exact)
    }
}

fn solve_f64(matrix: &mut DenseSquare<f64>, right: &mut [f64]) -> Result<(), ()> {
    for column in 0..matrix.order {
        let pivot = (column..matrix.order)
            .max_by(|left, right| {
                matrix.row(*left)[column]
                    .abs()
                    .total_cmp(&matrix.row(*right)[column].abs())
            })
            .ok_or(())?;
        if matrix.row(pivot)[column] == 0.0 || !matrix.row(pivot)[column].is_finite() {
            return Err(());
        }
        matrix.swap_rows(column, pivot);
        right.swap(column, pivot);
        let divisor = matrix.row(column)[column];
        for value in &mut matrix.row_mut(column)[column..] {
            *value /= divisor;
        }
        right[column] /= divisor;
        for row in 0..matrix.order {
            if row == column {
                continue;
            }
            let (target, pivot) = matrix.disjoint_rows_mut(row, column);
            let factor = target[column];
            for (target, pivot) in target[column..].iter_mut().zip(&pivot[column..]) {
                *target -= factor * pivot;
            }
            right[row] -= factor * right[column];
        }
    }
    Ok(())
}

fn exact_barycentric(
    simplex: &[usize],
    positions: &[f64],
    ambient: usize,
    exact: &mut ExactUse,
) -> Result<Vec<f64>, RealizationError> {
    let mut gram = exact_gram(simplex, positions, ambient, exact)?;
    exact.grow(2, 1)?;
    let two = BigRational::from_integer(BigInt::from(2_u8));
    let mut right = Vec::with_capacity(gram.order);
    for (index, row) in gram.values.chunks_exact(gram.order).enumerate() {
        right.push(exact_div(&row[index], &two, exact)?);
    }
    solve_exact(&mut gram, &mut right, exact)?;
    exact.grow(1, 1)?;
    let mut total = BigRational::zero();
    for value in &right {
        total = exact_add(&total, value, exact)?;
    }
    exact.grow(1, 1)?;
    let first = exact_sub(&BigRational::one(), &total, exact)?;
    std::iter::once(first)
        .chain(right)
        .map(|value| metered_binary64(&value, exact))
        .collect()
}

fn solve_exact(
    matrix: &mut DenseSquare<BigRational>,
    right: &mut [BigRational],
    exact: &mut ExactUse,
) -> Result<(), RealizationError> {
    for column in 0..matrix.order {
        let mut pivot = None;
        for row in column..matrix.order {
            exact.charge(1)?;
            if !matrix.row(row)[column].is_zero() {
                pivot = Some(row);
                break;
            }
        }
        let pivot = pivot.ok_or(RealizationError::Degenerate)?;
        matrix.swap_rows(column, pivot);
        right.swap(column, pivot);
        let divisor = matrix.row(column)[column].clone();
        for value in &mut matrix.row_mut(column)[column..] {
            let next = exact_div(value, &divisor, exact)?;
            exact.charge(1)?;
            *value = next;
        }
        right[column] = exact_div(&right[column], &divisor, exact)?;
        exact.charge(1)?;
        for row in 0..matrix.order {
            if row == column {
                continue;
            }
            let (target, pivot) = matrix.disjoint_rows_mut(row, column);
            let factor = target[column].clone();
            for (target, pivot) in target[column..].iter_mut().zip(&pivot[column..]) {
                let product = exact_mul(&factor, pivot, exact)?;
                let next = exact_sub(target, &product, exact)?;
                exact.charge(1)?;
                *target = next;
            }
            let product = exact_mul(&factor, &right[column], exact)?;
            right[row] = exact_sub(&right[row], &product, exact)?;
            exact.charge(1)?;
        }
    }
    Ok(())
}

fn exact_scaled_if_needed(coefficient: f64, scale: f64) -> Result<f64, RealizationError> {
    let value = coefficient * scale;
    if value.is_finite() && (value != 0.0 || coefficient == 0.0 || scale == 0.0) {
        Ok(value)
    } else {
        Err(RealizationError::Unrepresentable)
    }
}

fn scaled_pair(
    left: f64,
    right: f64,
    divisor: f64,
    error: RealizationError,
) -> Result<f64, RealizationError> {
    scaled_product([left, right], [divisor], error)
}

fn scaled_product(
    factors: impl IntoIterator<Item = f64>,
    divisors: impl IntoIterator<Item = f64>,
    error: RealizationError,
) -> Result<f64, RealizationError> {
    let mut mantissa = 1.0;
    let mut exponent = 0_i32;
    let mut nonzero = true;
    for factor in factors {
        nonzero &= factor != 0.0;
        let (part, shift) = frexp(factor);
        mantissa *= part;
        let (normalized, carry) = frexp(mantissa);
        mantissa = normalized;
        exponent = exponent.checked_add(shift + carry).ok_or(error)?;
    }
    for divisor in divisors {
        let (part, shift) = frexp(divisor);
        mantissa /= part;
        let (normalized, carry) = frexp(mantissa);
        mantissa = normalized;
        exponent = exponent.checked_add(carry - shift).ok_or(error)?;
    }
    let value = ldexp(mantissa, exponent);
    if value.is_finite() && (!nonzero || value != 0.0) {
        Ok(value)
    } else {
        Err(error)
    }
}

fn frexp(value: f64) -> (f64, i32) {
    if value == 0.0 {
        return (0.0, 0);
    }
    let bits = value.to_bits();
    let encoded = ((bits >> 52) & 0x7ff) as i32;
    if encoded == 0 {
        let (mantissa, exponent) = frexp(value * 2.0_f64.powi(54));
        return (mantissa, exponent - 54);
    }
    let mantissa = f64::from_bits((bits & (1_u64 << 63 | ((1_u64 << 52) - 1))) | (1022_u64 << 52));
    (mantissa, encoded - 1022)
}

fn ldexp(mantissa: f64, exponent: i32) -> f64 {
    if exponent > 1023 {
        mantissa * 2.0_f64.powi(1023) * 2.0_f64.powi(exponent - 1023)
    } else if exponent < -1022 {
        mantissa * 2.0_f64.powi(-1022) * 2.0_f64.powi(exponent + 1022)
    } else {
        mantissa * 2.0_f64.powi(exponent)
    }
}

fn sum_incident(
    contributions: &[(usize, f64)],
    count: usize,
    exact: &mut ExactUse,
) -> Result<(Vec<f64>, Vec<i8>), RealizationError> {
    let mut sums = vec![0.0; count];
    let mut corrections = vec![0.0; count];
    let mut magnitudes = vec![0.0; count];
    let mut terms = vec![0_usize; count];
    for &(row, value) in contributions {
        let next = sums[row] + value;
        corrections[row] += if sums[row].abs() >= value.abs() {
            (sums[row] - next) + value
        } else {
            (value - next) + sums[row]
        };
        sums[row] = next;
        magnitudes[row] += value.abs();
        terms[row] += 1;
    }
    for (value, correction) in sums.iter_mut().zip(corrections) {
        *value += correction;
        if !value.is_finite() {
            return Err(RealizationError::Unrepresentable);
        }
    }
    let signs = sums
        .iter()
        .enumerate()
        .map(|(row, value)| {
            let bound = 8.0 * float(terms[row]) * f64::EPSILON * magnitudes[row];
            if *value > bound {
                Ok(1)
            } else if *value < -bound {
                Ok(-1)
            } else {
                exact_sum_sign(contributions, row, exact)
            }
        })
        .collect::<Result<Vec<_>, RealizationError>>()?;
    Ok((sums, signs))
}

fn exact_sum_sign(
    contributions: &[(usize, f64)],
    row: usize,
    use_: &mut ExactUse,
) -> Result<i8, RealizationError> {
    use_.grow(1, 1)?;
    let mut sum = BigRational::zero();
    for (_, value) in contributions.iter().filter(|(index, _)| *index == row) {
        let value = metered_exact_from_binary64(*value, use_)?;
        sum = exact_add(&sum, &value, use_)?;
    }
    use_.charge(2)?;
    Ok(i8::from(sum.is_positive()) - i8::from(sum.is_negative()))
}

fn exact_from_binary64(value: f64) -> BigRational {
    debug_assert!(value.is_finite());
    if value == 0.0 {
        return BigRational::zero();
    }
    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let encoded_exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, exponent) = if encoded_exponent == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, encoded_exponent - 1023 - 52)
    };
    let mut numerator = BigInt::from(significand);
    if negative {
        numerator = -numerator;
    }
    if exponent >= 0 {
        BigRational::from_integer(numerator << usize::try_from(exponent).expect("nonnegative"))
    } else {
        BigRational::new(
            numerator,
            BigInt::one() << usize::try_from(exponent.unsigned_abs()).expect("native shift"),
        )
    }
}

fn metered_binary64(value: &BigRational, exact: &mut ExactUse) -> Result<f64, RealizationError> {
    let bits = rational_bits(value)
        .checked_add(BINARY64_UNDERFLOW_BIT_DIFFERENCE)
        .ok_or(RealizationError::Overflow)?;
    exact.grow(bits, 4)?;
    binary64_from_exact_rounded(value)
}

fn metered_binary64_sqrt(
    value: &BigRational,
    exact: &mut ExactUse,
) -> Result<f64, RealizationError> {
    exact.grow(rational_bits(value), 3)?;
    binary64_sqrt_from_exact_rounded(value)
}

fn binary64_from_exact_rounded(value: &BigRational) -> Result<f64, RealizationError> {
    if value.is_zero() {
        return Ok(0.0);
    }
    let mut numerator = value.numer().magnitude().clone();
    let mut denominator = value.denom().magnitude().clone();
    let (positive_difference, difference) = match numerator.bits().checked_sub(denominator.bits()) {
        Some(difference) => (true, difference),
        None => (false, denominator.bits() - numerator.bits()),
    };
    if (positive_difference && difference > BINARY64_OVERFLOW_BIT_DIFFERENCE)
        || (!positive_difference && difference > BINARY64_UNDERFLOW_BIT_DIFFERENCE)
    {
        return Err(RealizationError::Unrepresentable);
    }
    let difference = i64::try_from(difference).map_err(|_| RealizationError::Unrepresentable)?;
    let difference = if positive_difference {
        difference
    } else {
        -difference
    };
    let shift = difference.max(i64::from(f64::MIN_EXP)) - i64::from(f64::MANTISSA_DIGITS) - 2;
    if shift >= 0 {
        denominator <<= usize::try_from(shift).map_err(|_| RealizationError::Unrepresentable)?;
    } else {
        numerator <<= usize::try_from(-shift).map_err(|_| RealizationError::Unrepresentable)?;
    }
    let (quotient, remainder) = numerator.div_rem(&denominator);
    let mut quotient = quotient.to_u64().ok_or(RealizationError::Unrepresentable)?;
    let quotient_bits = i64::from(u64::BITS - quotient.leading_zeros());
    let subnormal_bits = i64::from(f64::MIN_EXP) - shift;
    let rounding_bits =
        usize::try_from(quotient_bits.max(subnormal_bits) - i64::from(f64::MANTISSA_DIGITS))
            .map_err(|_| RealizationError::Unrepresentable)?;
    if !(2..=3).contains(&rounding_bits) {
        return Err(RealizationError::Unrepresentable);
    }
    let rounding_mask = (1_u64 << rounding_bits) - 1;
    let retained_is_odd = quotient & (1_u64 << rounding_bits) != 0;
    let round_up = quotient & (1_u64 << (rounding_bits - 1)) != 0
        && (retained_is_odd || quotient & (rounding_mask >> 1) != 0 || !remainder.is_zero());
    if round_up {
        quotient += 1_u64 << rounding_bits;
    }
    quotient &= !rounding_mask;
    let magnitude = quotient.to_f64().ok_or(RealizationError::Unrepresentable)?;
    let signed = if value.is_negative() {
        -magnitude
    } else {
        magnitude
    };
    let result = ldexp(
        signed,
        i32::try_from(shift).map_err(|_| RealizationError::Unrepresentable)?,
    );
    if result.is_finite() && result != 0.0 {
        Ok(result)
    } else {
        Err(RealizationError::Unrepresentable)
    }
}

fn binary64_sqrt_from_exact_rounded(value: &BigRational) -> Result<f64, RealizationError> {
    if !value.is_positive() {
        return Err(RealizationError::Degenerate);
    }
    let (mut head, mut exponent) = rational_head_exponent(value)?;
    if exponent % 2 != 0 {
        head *= 2.0;
        exponent -= 1;
    }
    let result = ldexp(head.sqrt(), exponent / 2);
    if result.is_finite() && result > 0.0 {
        Ok(result)
    } else {
        Err(RealizationError::Unrepresentable)
    }
}

fn rational_head_exponent(value: &BigRational) -> Result<(f64, i32), RealizationError> {
    let numerator = value.numer().magnitude();
    let denominator = value.denom().magnitude();
    let numerator_shift = numerator.bits().saturating_sub(RATIONAL_HEAD_BITS);
    let denominator_shift = denominator.bits().saturating_sub(RATIONAL_HEAD_BITS);
    let numerator_head = (numerator >> numerator_shift)
        .to_u64()
        .and_then(|value| value.to_f64())
        .ok_or(RealizationError::Unrepresentable)?;
    let denominator_head = (denominator >> denominator_shift)
        .to_u64()
        .and_then(|value| value.to_f64())
        .ok_or(RealizationError::Unrepresentable)?;
    let exponent = i32::try_from(numerator_shift)
        .ok()
        .and_then(|left| {
            i32::try_from(denominator_shift)
                .ok()
                .map(|right| left - right)
        })
        .ok_or(RealizationError::Unrepresentable)?;
    Ok((numerator_head / denominator_head, exponent))
}

fn float(value: usize) -> f64 {
    value.to_f64().unwrap_or(f64::INFINITY)
}

#[cfg(test)]
#[path = "../tests/unit/realization.rs"]
mod tests;
