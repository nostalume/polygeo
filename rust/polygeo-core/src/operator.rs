//! Representation-free binary64 linear operators.

use std::{fmt, sync::Arc};

use num_traits::ToPrimitive;

use crate::form_impl::Binary64Basis;
use crate::{
    Binary64Element, Binary64Space, CanonicalSelection, Chain, CircumcentricPairing, Cochain,
    CoefficientSlice, Geometry, GeometryError, Metric, NondegenerateCapability,
    NondegeneratePairing, PairingCapability, TopologyError, Variance,
};

const MAX_OPERATOR_STEPS: usize = 64;

/// Failure to construct or apply a semantic binary64 operator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperatorError {
    Topology(TopologyError),
    Geometry(GeometryError),
    NonFinite,
    SpaceMismatch,
    FullSpaceRequired,
    DegreeOutside,
    PlanLimit,
}

impl OperatorError {
    /// Stable machine-readable reason.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::Topology(error) => error.reason(),
            Self::Geometry(error) => error.reason(),
            Self::NonFinite => "non_finite",
            Self::SpaceMismatch => "space_mismatch",
            Self::FullSpaceRequired => "full_space_required",
            Self::DegreeOutside => "degree_outside",
            Self::PlanLimit => "operator_plan_limit",
        }
    }
}

impl fmt::Display for OperatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.reason())
    }
}

impl std::error::Error for OperatorError {}

impl From<TopologyError> for OperatorError {
    fn from(error: TopologyError) -> Self {
        Self::Topology(error)
    }
}

impl From<GeometryError> for OperatorError {
    fn from(error: GeometryError) -> Self {
        Self::Geometry(error)
    }
}

#[derive(Clone, Debug)]
enum AtomicRecipe {
    Differential,
    Restriction,
    ExtensionByZero,
    Riesz(Arc<Geometry>),
    InverseRiesz(Arc<Geometry>),
    Codifferential(Arc<Geometry>),
    Laplacian(Arc<Geometry>),
    Identity,
    Zero,
}

impl AtomicRecipe {
    fn same_identity(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Riesz(left), Self::Riesz(right))
            | (Self::InverseRiesz(left), Self::InverseRiesz(right))
            | (Self::Codifferential(left), Self::Codifferential(right))
            | (Self::Laplacian(left), Self::Laplacian(right)) => Arc::ptr_eq(left, right),
            (Self::Differential, Self::Differential)
            | (Self::Restriction, Self::Restriction)
            | (Self::ExtensionByZero, Self::ExtensionByZero)
            | (Self::Identity, Self::Identity)
            | (Self::Zero, Self::Zero) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
struct Step {
    target: Binary64Basis,
    recipe: AtomicRecipe,
}

impl Step {
    fn same_identity(&self, other: &Self) -> bool {
        self.target.same_basis(&other.target) && self.recipe.same_identity(&other.recipe)
    }
}

#[derive(Clone, Debug)]
enum OperatorRecipe {
    Atomic(AtomicRecipe),
    Composite(Arc<[Step]>),
}

/// One binary64 mathematical action with admitted variance-indexed endpoints.
#[derive(Clone, Debug)]
pub struct LinearOperator<S: Variance, T: Variance> {
    source: Binary64Space<S>,
    target: Binary64Space<T>,
    recipe: OperatorRecipe,
}

impl<S: Variance, T: Variance> LinearOperator<S, T> {
    fn atomic(source: Binary64Space<S>, target: Binary64Space<T>, recipe: AtomicRecipe) -> Self {
        Self {
            source,
            target,
            recipe: OperatorRecipe::Atomic(recipe),
        }
    }

    /// Admitted source basis.
    #[must_use]
    pub const fn source(&self) -> &Binary64Space<S> {
        &self.source
    }

    /// Admitted target basis.
    #[must_use]
    pub const fn target(&self) -> &Binary64Space<T> {
        &self.target
    }

    /// Number of normalized atomic actions.
    #[must_use]
    pub fn execution_steps(&self) -> usize {
        match &self.recipe {
            OperatorRecipe::Atomic(AtomicRecipe::Identity | AtomicRecipe::Zero) => 0,
            OperatorRecipe::Atomic(_) => 1,
            OperatorRecipe::Composite(steps) => steps.len(),
        }
    }

    /// Whether two handles denote the same owner-derived operator identity.
    #[must_use]
    pub fn same_identity(&self, other: &Self) -> bool {
        self.source.same_basis(&other.source)
            && self.target.same_basis(&other.target)
            && match (&self.recipe, &other.recipe) {
                (OperatorRecipe::Atomic(left), OperatorRecipe::Atomic(right)) => {
                    left.same_identity(right)
                }
                (OperatorRecipe::Composite(left), OperatorRecipe::Composite(right)) => {
                    left.iter()
                        .zip(right.iter())
                        .all(|(left, right)| left.same_identity(right))
                        && left.len() == right.len()
                }
                _ => false,
            }
    }

    /// Apply without constructing a sparse or backend representation.
    ///
    /// # Errors
    ///
    /// Returns an endpoint, retained-data, resource, or finite-arithmetic failure.
    pub fn apply(&self, value: &Binary64Element<S>) -> Result<Binary64Element<T>, OperatorError> {
        if !self.source.same_basis(value.space()) {
            return Err(OperatorError::SpaceMismatch);
        }
        if matches!(self.recipe, OperatorRecipe::Atomic(AtomicRecipe::Identity)) {
            return Ok(Binary64Element::from_shared(
                self.target.clone(),
                Arc::clone(value.shared_coefficients()),
            ));
        }
        let output = self.apply_coefficients(value.coefficients())?;
        Ok(Binary64Element::from_shared(
            self.target.clone(),
            output.into(),
        ))
    }

    pub(crate) fn apply_coefficients(&self, input: &[f64]) -> Result<Vec<f64>, OperatorError> {
        if input.len() != self.source.size() || input.iter().any(|value| !value.is_finite()) {
            return Err(OperatorError::SpaceMismatch);
        }
        match &self.recipe {
            OperatorRecipe::Atomic(AtomicRecipe::Identity) => Ok(input.to_vec()),
            OperatorRecipe::Atomic(AtomicRecipe::Zero) => Ok(vec![0.0; self.target.size()]),
            OperatorRecipe::Atomic(recipe) => {
                let mut output = Vec::new();
                apply_atomic(
                    recipe,
                    self.source.basis(),
                    self.target.basis(),
                    input,
                    &mut output,
                )?;
                Ok(output)
            }
            OperatorRecipe::Composite(steps) => {
                let capacity = steps
                    .iter()
                    .map(|step| step.target.size() + scratch_size(&step.recipe, &step.target))
                    .max()
                    .unwrap_or(0);
                let mut left = Vec::with_capacity(capacity);
                let mut right = Vec::with_capacity(capacity);
                apply_atomic(
                    &steps[0].recipe,
                    self.source.basis(),
                    &steps[0].target,
                    input,
                    &mut left,
                )?;
                for (index, step) in steps[1..].iter().enumerate() {
                    let source = &steps[index].target;
                    if index.is_multiple_of(2) {
                        apply_atomic(&step.recipe, source, &step.target, &left, &mut right)?;
                    } else {
                        apply_atomic(&step.recipe, source, &step.target, &right, &mut left)?;
                    }
                }
                let output = if steps.len().is_multiple_of(2) {
                    right
                } else {
                    left
                };
                Ok(output)
            }
        }
    }
}

impl<K: Variance> Binary64Space<K> {
    /// Canonical identity on this numerical basis.
    #[must_use]
    pub fn identity(&self) -> LinearOperator<K, K> {
        LinearOperator::atomic(self.clone(), self.clone(), AtomicRecipe::Identity)
    }

    /// Canonical zero map to an explicitly named target basis.
    #[must_use]
    pub fn zero_to<T: Variance>(&self, target: &Binary64Space<T>) -> LinearOperator<K, T> {
        LinearOperator::atomic(self.clone(), target.clone(), AtomicRecipe::Zero)
    }
}

impl Binary64Space<Cochain> {
    /// Exterior derivative into the next retained cochain degree.
    ///
    /// # Errors
    ///
    /// Returns a full-space, degree, or retained-topology failure.
    pub fn exterior_derivative(&self) -> Result<LinearOperator<Cochain, Cochain>, OperatorError> {
        let source = self.full_basis().ok_or(OperatorError::FullSpaceRequired)?;
        let target_basis = source.successor()?;
        let target = Binary64Space::from_full(target_basis);
        Ok(LinearOperator::atomic(
            self.clone(),
            target,
            AtomicRecipe::Differential,
        ))
    }
}

impl CanonicalSelection {
    /// Coordinate restriction from the full basis to this canonical selection.
    ///
    /// # Errors
    ///
    /// Returns a degree or retained-topology failure.
    pub fn restriction<K: Variance>(
        self: &Arc<Self>,
    ) -> Result<LinearOperator<K, K>, OperatorError> {
        let source = Binary64Space::full(Arc::clone(self.owner()), self.degree())?;
        let target = Binary64Space::selected(Arc::clone(self))?;
        Ok(LinearOperator::atomic(
            source,
            target,
            AtomicRecipe::Restriction,
        ))
    }

    /// Extension by zero from this canonical selection to the full basis.
    ///
    /// # Errors
    ///
    /// Returns a degree or retained-topology failure.
    pub fn extension_by_zero<K: Variance>(
        self: &Arc<Self>,
    ) -> Result<LinearOperator<K, K>, OperatorError> {
        let source = Binary64Space::selected(Arc::clone(self))?;
        let target = Binary64Space::full(Arc::clone(self.owner()), self.degree())?;
        Ok(LinearOperator::atomic(
            source,
            target,
            AtomicRecipe::ExtensionByZero,
        ))
    }
}

impl LinearOperator<Cochain, Chain> {
    pub(crate) fn riesz(realization: Arc<Geometry>, degree: usize) -> Result<Self, OperatorError> {
        realization.hodge_coefficients(degree)?;
        let owner = realization.topology();
        Ok(Self::atomic(
            Binary64Space::full(Arc::clone(owner), degree)?,
            Binary64Space::full(Arc::clone(owner), degree)?,
            AtomicRecipe::Riesz(realization),
        ))
    }
}

impl LinearOperator<Chain, Cochain> {
    pub(crate) fn inverse_riesz(
        realization: Arc<Geometry>,
        degree: usize,
    ) -> Result<Self, OperatorError> {
        realization.hodge_coefficients(degree)?;
        let owner = realization.topology();
        Ok(Self::atomic(
            Binary64Space::full(Arc::clone(owner), degree)?,
            Binary64Space::full(Arc::clone(owner), degree)?,
            AtomicRecipe::InverseRiesz(realization),
        ))
    }
}

impl LinearOperator<Cochain, Cochain> {
    pub(crate) fn codifferential(
        realization: Arc<Geometry>,
        degree: usize,
    ) -> Result<Self, OperatorError> {
        let Some(target_degree) = degree.checked_sub(1) else {
            return Err(OperatorError::DegreeOutside);
        };
        realization.hodge_coefficients(degree)?;
        realization.hodge_coefficients(target_degree)?;
        let owner = realization.topology();
        Ok(Self::atomic(
            Binary64Space::full(Arc::clone(owner), degree)?,
            Binary64Space::full(Arc::clone(owner), target_degree)?,
            AtomicRecipe::Codifferential(realization),
        ))
    }

    pub(crate) fn laplacian(
        realization: Arc<Geometry>,
        degree: usize,
    ) -> Result<Self, OperatorError> {
        realization.hodge_coefficients(degree)?;
        let owner = realization.topology();
        let space = Binary64Space::full(Arc::clone(owner), degree)?;
        Ok(Self::atomic(
            space.clone(),
            space,
            AtomicRecipe::Laplacian(realization),
        ))
    }
}

impl PairingCapability for CircumcentricPairing {
    fn realization(&self) -> &Arc<Geometry> {
        &self.realization
    }

    fn riesz(&self, degree: usize) -> Result<LinearOperator<Cochain, Chain>, OperatorError> {
        LinearOperator::riesz(Arc::clone(&self.realization), degree)
    }
}

impl PairingCapability for NondegeneratePairing {
    fn realization(&self) -> &Arc<Geometry> {
        &self.realization
    }

    fn riesz(&self, degree: usize) -> Result<LinearOperator<Cochain, Chain>, OperatorError> {
        LinearOperator::riesz(Arc::clone(&self.realization), degree)
    }
}

impl NondegenerateCapability for NondegeneratePairing {
    fn inverse_riesz(
        &self,
        degree: usize,
    ) -> Result<LinearOperator<Chain, Cochain>, OperatorError> {
        LinearOperator::inverse_riesz(Arc::clone(&self.realization), degree)
    }

    fn codifferential(
        &self,
        degree: usize,
    ) -> Result<LinearOperator<Cochain, Cochain>, OperatorError> {
        LinearOperator::codifferential(Arc::clone(&self.realization), degree)
    }

    fn laplacian(&self, degree: usize) -> Result<LinearOperator<Cochain, Cochain>, OperatorError> {
        LinearOperator::laplacian(Arc::clone(&self.realization), degree)
    }
}

impl PairingCapability for Metric {
    fn realization(&self) -> &Arc<Geometry> {
        &self.realization
    }

    fn riesz(&self, degree: usize) -> Result<LinearOperator<Cochain, Chain>, OperatorError> {
        LinearOperator::riesz(Arc::clone(&self.realization), degree)
    }
}

impl NondegenerateCapability for Metric {
    fn inverse_riesz(
        &self,
        degree: usize,
    ) -> Result<LinearOperator<Chain, Cochain>, OperatorError> {
        LinearOperator::inverse_riesz(Arc::clone(&self.realization), degree)
    }

    fn codifferential(
        &self,
        degree: usize,
    ) -> Result<LinearOperator<Cochain, Cochain>, OperatorError> {
        LinearOperator::codifferential(Arc::clone(&self.realization), degree)
    }

    fn laplacian(&self, degree: usize) -> Result<LinearOperator<Cochain, Cochain>, OperatorError> {
        LinearOperator::laplacian(Arc::clone(&self.realization), degree)
    }
}

/// Compose `after ∘ before` into one normalized flat numerical plan.
///
/// # Errors
///
/// Returns an endpoint mismatch or rejects a plan above 64 atomic steps.
pub fn compose<S: Variance, M: Variance, T: Variance>(
    after: &LinearOperator<M, T>,
    before: &LinearOperator<S, M>,
) -> Result<LinearOperator<S, T>, OperatorError> {
    if !before.target.same_basis(&after.source) {
        return Err(OperatorError::SpaceMismatch);
    }
    if matches!(before.recipe, OperatorRecipe::Atomic(AtomicRecipe::Zero))
        || matches!(after.recipe, OperatorRecipe::Atomic(AtomicRecipe::Zero))
    {
        return Ok(LinearOperator::atomic(
            before.source.clone(),
            after.target.clone(),
            AtomicRecipe::Zero,
        ));
    }
    let mut steps = Vec::new();
    append_steps(before, &mut steps);
    append_steps(after, &mut steps);
    if steps.len() > MAX_OPERATOR_STEPS {
        return Err(OperatorError::PlanLimit);
    }
    let recipe = match steps.len() {
        0 => OperatorRecipe::Atomic(AtomicRecipe::Identity),
        1 => {
            let Some(step) = steps.pop() else {
                return Err(OperatorError::PlanLimit);
            };
            OperatorRecipe::Atomic(step.recipe)
        }
        _ => OperatorRecipe::Composite(steps.into()),
    };
    Ok(LinearOperator {
        source: before.source.clone(),
        target: after.target.clone(),
        recipe,
    })
}

fn append_steps<S: Variance, T: Variance>(operator: &LinearOperator<S, T>, output: &mut Vec<Step>) {
    match &operator.recipe {
        OperatorRecipe::Atomic(AtomicRecipe::Identity) => {}
        OperatorRecipe::Atomic(recipe) => output.push(Step {
            target: operator.target.basis().clone(),
            recipe: recipe.clone(),
        }),
        OperatorRecipe::Composite(steps) => output.extend(steps.iter().cloned()),
    }
}

fn scratch_size(recipe: &AtomicRecipe, endpoint: &Binary64Basis) -> usize {
    match recipe {
        AtomicRecipe::Laplacian(realization)
            if degree(endpoint).is_ok_and(|degree| degree < realization.topology().dimension()) =>
        {
            let degree = degree(endpoint).unwrap_or(0);
            realization
                .topology()
                .chain_view()
                .basis_size(degree + 1)
                .unwrap_or(0)
        }
        _ => 0,
    }
}

fn apply_atomic(
    recipe: &AtomicRecipe,
    source: &Binary64Basis,
    target: &Binary64Basis,
    input: &[f64],
    output: &mut Vec<f64>,
) -> Result<(), OperatorError> {
    output.clear();
    match recipe {
        AtomicRecipe::Differential => {
            output.resize(target.size(), 0.0);
            let target = target.full().ok_or(OperatorError::FullSpaceRequired)?;
            if let Ok(degree) = usize::try_from(target.degree)
                && degree != 0
                && degree <= target.domain.view().dimension()
            {
                target
                    .domain
                    .view()
                    .boundary(degree)?
                    .apply_transpose_binary64(input, output)?;
            }
        }
        AtomicRecipe::Restriction => {
            let Binary64Basis::Selected(selection) = target else {
                return Err(OperatorError::SpaceMismatch);
            };
            output.extend(selection.indices().iter().map(|&index| input[index]));
        }
        AtomicRecipe::ExtensionByZero => {
            let Binary64Basis::Selected(selection) = source else {
                return Err(OperatorError::SpaceMismatch);
            };
            output.resize(target.size(), 0.0);
            for (&index, &value) in selection.indices().iter().zip(input) {
                output[index] = value;
            }
        }
        AtomicRecipe::Riesz(realization) => {
            scale(
                input,
                realization.hodge_coefficients(degree(source)?)?,
                output,
                false,
            );
        }
        AtomicRecipe::InverseRiesz(realization) => {
            scale(
                input,
                realization.hodge_coefficients(degree(source)?)?,
                output,
                true,
            );
        }
        AtomicRecipe::Codifferential(realization) => {
            apply_codifferential(realization, degree(source)?, input, output)?;
        }
        AtomicRecipe::Laplacian(realization) => {
            apply_laplacian(realization, degree(source)?, input, output)?;
        }
        AtomicRecipe::Identity => output.extend_from_slice(input),
        AtomicRecipe::Zero => output.resize(target.size(), 0.0),
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(OperatorError::NonFinite);
    }
    Ok(())
}

fn degree(endpoint: &Binary64Basis) -> Result<usize, OperatorError> {
    usize::try_from(endpoint.degree()).map_err(|_| OperatorError::DegreeOutside)
}

fn scale(input: &[f64], weights: &[f64], output: &mut Vec<f64>, inverse: bool) {
    output.extend(input.iter().zip(weights).map(|(&value, &weight)| {
        if inverse {
            value / weight
        } else {
            value * weight
        }
    }));
}

fn apply_codifferential(
    realization: &Geometry,
    degree: usize,
    input: &[f64],
    output: &mut Vec<f64>,
) -> Result<(), OperatorError> {
    let boundary = realization.topology().chain_view().boundary(degree)?;
    output.resize(boundary.shape().0, 0.0);
    let source_weights = realization.hodge_coefficients(degree)?;
    let target_weights = realization.hodge_coefficients(degree - 1)?;
    dispatch_rows(
        boundary.indptr(),
        boundary.indices(),
        boundary.coefficients(),
        |row, entries| {
            output[row] = entries
                .map(|(column, coefficient)| coefficient * source_weights[column] * input[column])
                .sum::<f64>()
                / target_weights[row];
        },
    );
    Ok(())
}

fn apply_laplacian(
    realization: &Geometry,
    degree: usize,
    input: &[f64],
    output: &mut Vec<f64>,
) -> Result<(), OperatorError> {
    let owner = realization.topology();
    let weights = realization.hodge_coefficients(degree)?;
    let upper_size = if degree < owner.dimension() {
        owner.chain_view().basis_size(degree + 1)?
    } else {
        0
    };
    output.resize(input.len() + upper_size, 0.0);
    let (result, upper) = output.split_at_mut(input.len());

    if degree < owner.dimension() {
        let boundary = owner.chain_view().boundary(degree + 1)?;
        boundary.apply_transpose_binary64(input, upper)?;
        let upper_weights = realization.hodge_coefficients(degree + 1)?;
        dispatch_rows(
            boundary.indptr(),
            boundary.indices(),
            boundary.coefficients(),
            |row, entries| {
                result[row] += entries
                    .map(|(column, coefficient)| {
                        coefficient * upper_weights[column] * upper[column]
                    })
                    .sum::<f64>()
                    / weights[row];
            },
        );
    }
    if degree > 0 {
        let boundary = owner.chain_view().boundary(degree)?;
        let lower_weights = realization.hodge_coefficients(degree - 1)?;
        apply_lower_laplacian_rows(
            boundary.indptr(),
            boundary.indices(),
            boundary.coefficients(),
            input,
            weights,
            lower_weights,
            result,
        );
    }
    output.truncate(input.len());
    Ok(())
}

fn apply_lower_laplacian_rows(
    indptr: &[usize],
    indices: &[usize],
    coefficients: CoefficientSlice<'_>,
    input: &[f64],
    weights: &[f64],
    lower_weights: &[f64],
    output: &mut [f64],
) {
    match coefficients {
        CoefficientSlice::I8(values) => lower_laplacian_rows(
            Rows {
                indptr,
                indices,
                values,
            },
            f64::from,
            input,
            weights,
            lower_weights,
            output,
        ),
        CoefficientSlice::I64(values) => lower_laplacian_rows(
            Rows {
                indptr,
                indices,
                values,
            },
            |value| value.to_f64().expect("i64 rounds to finite binary64"),
            input,
            weights,
            lower_weights,
            output,
        ),
    }
}

#[derive(Clone, Copy)]
struct Rows<'a, T> {
    indptr: &'a [usize],
    indices: &'a [usize],
    values: &'a [T],
}

fn lower_laplacian_rows<T: Copy>(
    rows: Rows<'_, T>,
    binary64: impl Fn(T) -> f64,
    input: &[f64],
    weights: &[f64],
    lower_weights: &[f64],
    output: &mut [f64],
) {
    for (row, offsets) in rows.indptr.windows(2).enumerate() {
        let range = offsets[0]..offsets[1];
        let lower = range
            .clone()
            .map(|position| {
                binary64(rows.values[position])
                    * weights[rows.indices[position]]
                    * input[rows.indices[position]]
            })
            .sum::<f64>()
            / lower_weights[row];
        for position in range {
            output[rows.indices[position]] += binary64(rows.values[position]) * lower;
        }
    }
}

fn dispatch_rows(
    indptr: &[usize],
    indices: &[usize],
    coefficients: CoefficientSlice<'_>,
    mut row: impl FnMut(usize, &mut dyn Iterator<Item = (usize, f64)>),
) {
    match coefficients {
        CoefficientSlice::I8(values) => rows(indptr, indices, values, f64::from, &mut row),
        CoefficientSlice::I64(values) => rows(
            indptr,
            indices,
            values,
            |value| value.to_f64().expect("i64 rounds to finite binary64"),
            &mut row,
        ),
    }
}

fn rows<T: Copy>(
    indptr: &[usize],
    indices: &[usize],
    values: &[T],
    binary64: impl Fn(T) -> f64,
    row: &mut impl FnMut(usize, &mut dyn Iterator<Item = (usize, f64)>),
) {
    for (row_index, offsets) in indptr.windows(2).enumerate() {
        let mut entries = (offsets[0]..offsets[1])
            .map(|position| (indices[position], binary64(values[position])));
        row(row_index, &mut entries);
    }
}
