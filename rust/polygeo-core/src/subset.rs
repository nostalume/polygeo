use std::sync::Arc;

use crate::TopologyError;
use crate::complex::ComplexCore;
use crate::incidence::{DegreeLayout, try_filled};
use crate::mask::{PackedDegreeMasks, PackedDegreeMasksBuilder};

/// Immutable packed simplex membership bound to one exact topology owner.
#[derive(Debug)]
pub struct SimplexSubset {
    owner: Arc<ComplexCore>,
    source: SubsetSource,
}

#[derive(Debug)]
enum SubsetSource {
    Owned(PackedDegreeMasks),
    RegularBoundary,
}

impl ComplexCore {
    /// Admit degree-aligned Boolean membership into one packed owner-bound value.
    ///
    /// # Errors
    ///
    /// Returns a shape, overflow, or allocation error.
    pub fn subset<T: AsRef<[bool]>>(
        self: &Arc<Self>,
        degrees: &[T],
    ) -> Result<SimplexSubset, TopologyError> {
        let mut builder = self.subset_builder()?;
        for values in degrees {
            builder.push_degree(values.as_ref().iter().copied())?;
        }
        builder.finish()
    }

    /// Begin direct packing of degree-aligned Boolean membership.
    ///
    /// # Errors
    ///
    /// Returns an allocation or count-overflow error.
    pub fn subset_builder(self: &Arc<Self>) -> Result<SubsetBuilder<'_>, TopologyError> {
        Ok(SubsetBuilder {
            owner: self,
            masks: PackedDegreeMasksBuilder::empty(&self.layout)?,
            next_degree: 0,
        })
    }

    /// Admit a strictly increasing selection in one canonical degree basis.
    ///
    /// # Errors
    ///
    /// Returns a degree, ordering, range, or allocation error.
    pub fn selection(
        self: &Arc<Self>,
        degree: usize,
        indices: Vec<usize>,
    ) -> Result<CanonicalSelection, TopologyError> {
        let size = self.basis(degree)?.row_count();
        if indices.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(TopologyError::SelectionNotStrict);
        }
        if indices.iter().any(|index| *index >= size) {
            return Err(TopologyError::SelectionIndexOutside);
        }
        Ok(CanonicalSelection {
            owner: Arc::clone(self),
            degree,
            indices: indices.into_boxed_slice(),
        })
    }
}

/// One-pass admission that packs Boolean degrees directly into final words.
#[derive(Debug)]
pub struct SubsetBuilder<'a> {
    owner: &'a Arc<ComplexCore>,
    masks: PackedDegreeMasksBuilder<'a>,
    next_degree: usize,
}

impl SubsetBuilder<'_> {
    /// Pack exactly one next degree from a single-pass Boolean iterator.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::MaskShape`] unless this is the next canonical
    /// degree with its exact logical length.
    pub fn push_degree(
        &mut self,
        values: impl IntoIterator<Item = bool>,
    ) -> Result<(), TopologyError> {
        if self.next_degree == self.owner.layout.counts().len() {
            return Err(TopologyError::MaskShape);
        }
        let expected = self.owner.layout.count(self.next_degree)?;
        let mut count = 0_usize;
        for selected in values {
            if count == expected {
                return Err(TopologyError::MaskShape);
            }
            if selected {
                self.masks.set(self.next_degree, count, true)?;
            }
            count += 1;
        }
        if count != expected {
            return Err(TopologyError::MaskShape);
        }
        self.next_degree += 1;
        Ok(())
    }

    /// Finish only after every canonical degree was supplied.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::MaskShape`] when a degree is missing.
    pub fn finish(self) -> Result<SimplexSubset, TopologyError> {
        if self.next_degree != self.owner.layout.counts().len() {
            return Err(TopologyError::MaskShape);
        }
        Ok(SimplexSubset::from_owned(
            Arc::clone(self.owner),
            self.masks.finish(),
        ))
    }
}

impl SimplexSubset {
    fn from_owned(owner: Arc<ComplexCore>, masks: PackedDegreeMasks) -> Self {
        Self {
            owner,
            source: SubsetSource::Owned(masks),
        }
    }

    fn regular_boundary(owner: Arc<ComplexCore>) -> Self {
        Self {
            owner,
            source: SubsetSource::RegularBoundary,
        }
    }

    fn resolved_masks(&self) -> Result<&PackedDegreeMasks, TopologyError> {
        match &self.source {
            SubsetSource::Owned(masks) => Ok(masks),
            SubsetSource::RegularBoundary => self
                .owner
                .admitted_regular_view()
                .map(crate::evidence::RegularView::packed_boundary)
                .ok_or(TopologyError::InternalInvariant),
        }
    }

    #[must_use]
    pub const fn owner(&self) -> &Arc<ComplexCore> {
        &self.owner
    }

    /// Export one caller-owned degree mask.
    ///
    /// # Errors
    ///
    /// Returns a degree, shape, overflow, or allocation error.
    pub fn mask(&self, degree: usize) -> Result<Vec<bool>, TopologyError> {
        self.resolved_masks()?
            .export_degree(degree, &self.owner.layout)
    }

    /// Resolve one degree into an owning canonical selection handle.
    ///
    /// # Errors
    ///
    /// Returns a degree, allocation, or retained-evidence failure.
    pub fn canonical_selection(&self, degree: usize) -> Result<CanonicalSelection, TopologyError> {
        let size = self.owner.basis(degree)?.row_count();
        let masks = self.resolved_masks()?;
        let mut indices = Vec::new();
        for index in 0..size {
            if masks.contains(degree, index, &self.owner.layout)? {
                indices.push(index);
            }
        }
        self.owner.selection(degree, indices)
    }

    /// Fill one caller-owned degree mask without an intermediate Boolean product.
    ///
    /// # Errors
    ///
    /// Returns a degree or mask-shape error.
    pub fn write_mask(&self, degree: usize, output: &mut [bool]) -> Result<(), TopologyError> {
        self.resolved_masks()?
            .write_degree(degree, &self.owner.layout, output)
    }

    /// Copy resolved membership into an independent owned subset value.
    ///
    /// # Errors
    ///
    /// Returns an allocation or internal invariant error.
    pub fn to_owned_subset(&self) -> Result<Self, TopologyError> {
        let masks = self.resolved_masks()?;
        let owned = PackedDegreeMasksBuilder::from_mask(&self.owner.layout, masks)?.finish();
        Ok(Self::from_owned(Arc::clone(&self.owner), owned))
    }

    /// Compute the least face-closed subset through retained immediate faces.
    ///
    /// # Errors
    ///
    /// Returns a classified resource or invariant error.
    pub fn closure(&self) -> Result<Self, TopologyError> {
        let layout = &self.owner.layout;
        let counts = layout.counts();
        let masks = self.resolved_masks()?;
        let mut output = PackedDegreeMasksBuilder::from_mask(layout, masks)?;
        for degree in (1..=self.owner.dimension).rev() {
            let width = degree + 1;
            for simplex in 0..counts[degree] {
                if !output.contains(degree, simplex)? {
                    continue;
                }
                let start = simplex
                    .checked_mul(width)
                    .ok_or(TopologyError::CountOverflow)?;
                for face in self.owner.immediate_faces[degree]
                    .get(start..start + width)
                    .ok_or(TopologyError::InternalInvariant)?
                {
                    output.set(degree - 1, *face, true)?;
                }
            }
        }
        Ok(Self::from_owned(Arc::clone(&self.owner), output.finish()))
    }

    /// Compute every canonical coface through retained CSR incidence.
    ///
    /// # Errors
    ///
    /// Returns a classified resource or invariant error.
    pub fn star(&self) -> Result<Self, TopologyError> {
        let layout = &self.owner.layout;
        let counts = layout.counts();
        let masks = self.resolved_masks()?;
        let mut output = PackedDegreeMasksBuilder::from_mask(layout, masks)?;
        for (degree, count) in counts
            .iter()
            .copied()
            .enumerate()
            .take(self.owner.dimension)
        {
            for simplex in 0..count {
                if !output.contains(degree, simplex)? {
                    continue;
                }
                let (cofaces, _) = self.owner.boundaries[degree + 1]
                    .storage
                    .row(simplex)
                    .ok_or(TopologyError::InternalInvariant)?;
                for coface in cofaces.iter().copied() {
                    output.set(degree + 1, coface, true)?;
                }
            }
        }
        Ok(Self::from_owned(Arc::clone(&self.owner), output.finish()))
    }

    /// Compute the exact union of links of every selected simplex.
    ///
    /// # Errors
    ///
    /// Returns a classified resource, overflow, or invariant error.
    pub fn link(&self) -> Result<Self, TopologyError> {
        let mut scratch = LinkRelationScratch::try_new(&self.owner.layout, self.owner.dimension)?;
        self.link_with_scratch(&mut scratch)
    }

    fn link_with_scratch(&self, scratch: &mut LinkRelationScratch) -> Result<Self, TopologyError> {
        let layout = &self.owner.layout;
        let counts = layout.counts();
        let masks = self.resolved_masks()?;
        let mut output = PackedDegreeMasksBuilder::empty(layout)?;

        for (selected_degree, selected_count) in counts.iter().copied().enumerate() {
            for selected_index in 0..selected_count {
                if !masks.contains(selected_degree, selected_index, layout)? {
                    continue;
                }
                scratch.begin_neighborhood();
                scratch.push(selected_degree, selected_index)?;
                let selected = self.owner.bases[selected_degree]
                    .row(selected_index)
                    .ok_or(TopologyError::InternalInvariant)?;
                while let Some((degree, simplex)) = scratch.pending.pop() {
                    if !scratch.visit(degree, simplex, layout)? {
                        continue;
                    }
                    let row = self.owner.bases[degree]
                        .row(simplex)
                        .ok_or(TopologyError::InternalInvariant)?;
                    scratch.complement.clear();
                    scratch.complement.extend(
                        row.iter()
                            .copied()
                            .filter(|vertex| selected.binary_search(vertex).is_err()),
                    );
                    for width in 1..=scratch.complement.len() {
                        scratch.combination.clear();
                        mark_combinations(
                            &self.owner,
                            &scratch.complement,
                            width,
                            0,
                            &mut scratch.combination,
                            &mut output,
                        )?;
                    }
                    if degree < self.owner.dimension {
                        let (cofaces, _) = self.owner.boundaries[degree + 1]
                            .storage
                            .row(simplex)
                            .ok_or(TopologyError::InternalInvariant)?;
                        for coface in cofaces.iter().copied() {
                            scratch.push(degree + 1, coface)?;
                        }
                    }
                }
            }
        }
        Ok(Self::from_owned(Arc::clone(&self.owner), output.finish()))
    }

    /// Decide whether every inclusion-maximal selected simplex has one degree.
    ///
    /// # Errors
    ///
    /// Returns a degree, resource, or invariant error.
    pub fn is_pure(&self, target: usize) -> Result<bool, TopologyError> {
        self.owner.basis(target)?;
        let layout = &self.owner.layout;
        let counts = layout.counts();
        let masks = self.resolved_masks()?;
        let mut any_selected = false;
        let mut covered = PackedDegreeMasksBuilder::empty(layout)?;
        for degree in (0..=self.owner.dimension).rev() {
            for simplex in 0..counts[degree] {
                let selected = masks.contains(degree, simplex, layout)?;
                any_selected |= selected;
                if selected && degree > target {
                    return Ok(false);
                }
                if selected && degree < target && !covered.contains(degree, simplex)? {
                    return Ok(false);
                }
                if degree > 0 && (selected || covered.contains(degree, simplex)?) {
                    let width = degree + 1;
                    let start = simplex
                        .checked_mul(width)
                        .ok_or(TopologyError::CountOverflow)?;
                    for face in self.owner.immediate_faces[degree]
                        .get(start..start + width)
                        .ok_or(TopologyError::InternalInvariant)?
                    {
                        covered.set(degree - 1, *face, true)?;
                    }
                }
            }
        }
        Ok(any_selected)
    }

    /// Compare membership after requiring exact native owner identity.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::OwnerMismatch`] for separately admitted owners.
    pub fn same_members(&self, other: &Self) -> Result<bool, TopologyError> {
        if !Arc::ptr_eq(&self.owner, &other.owner) {
            return Err(TopologyError::OwnerMismatch);
        }
        Ok(self.resolved_masks()? == other.resolved_masks()?)
    }

    #[cfg(test)]
    fn link_with_test_high_water(&self) -> Result<(Self, TestLinkHighWater), TopologyError> {
        let mut scratch = LinkRelationScratch::try_new(&self.owner.layout, self.owner.dimension)?;
        let link = self.link_with_scratch(&mut scratch)?;
        Ok((
            link,
            TestLinkHighWater {
                pending_capacity: scratch.pending.capacity(),
                touched_word_capacity: scratch.touched_words.capacity(),
            },
        ))
    }
}

#[derive(Debug)]
struct LinkRelationScratch {
    visited_words: Box<[u64]>,
    touched_words: Vec<usize>,
    pending: Vec<(usize, usize)>,
    complement: Vec<usize>,
    combination: Vec<usize>,
}

impl LinkRelationScratch {
    fn try_new(layout: &DegreeLayout, dimension: usize) -> Result<Self, TopologyError> {
        let word_count = layout.word_offsets().last().copied().unwrap_or(0);
        let max_width = dimension
            .checked_add(1)
            .ok_or(TopologyError::CountOverflow)?;
        let mut complement = Vec::new();
        complement
            .try_reserve_exact(max_width)
            .map_err(|_| TopologyError::Allocation)?;
        let mut combination = Vec::new();
        combination
            .try_reserve_exact(max_width)
            .map_err(|_| TopologyError::Allocation)?;
        Ok(Self {
            visited_words: try_filled(word_count, 0_u64)?.into_boxed_slice(),
            touched_words: Vec::new(),
            pending: Vec::new(),
            complement,
            combination,
        })
    }

    fn begin_neighborhood(&mut self) {
        for position in self.touched_words.drain(..) {
            self.visited_words[position] = 0;
        }
        self.pending.clear();
    }

    fn visit(
        &mut self,
        degree: usize,
        simplex: usize,
        layout: &DegreeLayout,
    ) -> Result<bool, TopologyError> {
        if simplex >= layout.count(degree)? {
            return Err(TopologyError::InternalInvariant);
        }
        let position = layout.word_offsets()[degree] + simplex / u64::BITS as usize;
        let bit = 1_u64 << (simplex % u64::BITS as usize);
        let word = self
            .visited_words
            .get_mut(position)
            .ok_or(TopologyError::InternalInvariant)?;
        if *word & bit != 0 {
            return Ok(false);
        }
        if *word == 0 {
            self.touched_words
                .try_reserve(1)
                .map_err(|_| TopologyError::Allocation)?;
            self.touched_words.push(position);
        }
        *word |= bit;
        Ok(true)
    }

    fn push(&mut self, degree: usize, simplex: usize) -> Result<(), TopologyError> {
        self.pending
            .try_reserve(1)
            .map_err(|_| TopologyError::Allocation)?;
        self.pending.push((degree, simplex));
        Ok(())
    }
}

#[cfg(test)]
struct TestLinkHighWater {
    pending_capacity: usize,
    touched_word_capacity: usize,
}

impl ComplexCore {
    /// Return the shared packed regular boundary as an owner-bound subset.
    ///
    /// # Errors
    ///
    /// Returns an unqueried or rejected regularity verdict.
    pub fn boundary_subset(self: &Arc<Self>) -> Result<SimplexSubset, TopologyError> {
        self.require_regular()?;
        Ok(SimplexSubset::regular_boundary(Arc::clone(self)))
    }
}

/// Immutable sorted coefficient positions bound to one owner and degree.
#[derive(Debug)]
pub struct CanonicalSelection {
    owner: Arc<ComplexCore>,
    degree: usize,
    indices: Box<[usize]>,
}

impl CanonicalSelection {
    #[must_use]
    pub const fn owner(&self) -> &Arc<ComplexCore> {
        &self.owner
    }

    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.indices.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Construct the checked increasing complement in the same degree.
    ///
    /// # Errors
    ///
    /// Returns a degree, overflow, or allocation error.
    pub fn complement(&self) -> Result<Self, TopologyError> {
        let size = self.owner.basis(self.degree)?.row_count();
        let output_size = size
            .checked_sub(self.indices.len())
            .ok_or(TopologyError::InternalInvariant)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(output_size)
            .map_err(|_| TopologyError::Allocation)?;
        let mut selected = self.indices.iter().copied().peekable();
        for index in 0..size {
            if selected.peek() == Some(&index) {
                selected.next();
            } else {
                output.push(index);
            }
        }
        Ok(Self {
            owner: Arc::clone(&self.owner),
            degree: self.degree,
            indices: output.into_boxed_slice(),
        })
    }

    /// Compare canonical selections after requiring exact owner identity.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::OwnerMismatch`] for separately admitted owners.
    pub fn same_selection(&self, other: &Self) -> Result<bool, TopologyError> {
        if !Arc::ptr_eq(&self.owner, &other.owner) {
            return Err(TopologyError::OwnerMismatch);
        }
        Ok(self.degree == other.degree && self.indices == other.indices)
    }
}

fn mark_combinations(
    owner: &ComplexCore,
    values: &[usize],
    width: usize,
    start: usize,
    current: &mut Vec<usize>,
    output: &mut PackedDegreeMasksBuilder,
) -> Result<(), TopologyError> {
    if current.len() == width {
        let degree = width - 1;
        let index = owner.bases[degree].binary_search(current)?;
        return output.set(degree, index, true);
    }
    let remaining = width - current.len();
    let final_start = values
        .len()
        .checked_sub(remaining)
        .ok_or(TopologyError::InternalInvariant)?;
    for index in start..=final_start {
        current.push(values[index]);
        mark_combinations(owner, values, width, index + 1, current, output)?;
        current.pop();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{CandidateInput, ComplexCore};

    fn grid(side: usize) -> Arc<ComplexCore> {
        let width = side + 1;
        let mut faces = Vec::with_capacity(6 * side * side);
        for row in 0..side {
            for column in 0..side {
                let lower_left = row * width + column;
                let lower_right = lower_left + 1;
                let upper_left = lower_left + width;
                let upper_right = upper_left + 1;
                faces.extend_from_slice(&[lower_left, lower_right, upper_right]);
                faces.extend_from_slice(&[lower_left, upper_right, upper_left]);
            }
        }
        ComplexCore::admit(
            CandidateInput::unsigned(
                faces.into_iter().map(|value| u64::try_from(value).unwrap()),
                2 * side * side,
                3,
                None,
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn link_scratch_capacity_tracks_the_reached_neighborhood() {
        let owner = grid(32);
        let counts = owner.layout.counts();
        let mut masks = counts
            .iter()
            .map(|count| vec![false; *count])
            .collect::<Vec<_>>();
        masks[0][16 * 33 + 16] = true;
        let subset = owner.subset(&masks).unwrap();

        let (link, high_water) = subset.link_with_test_high_water().unwrap();

        assert_eq!(
            link.mask(0).unwrap().iter().filter(|value| **value).count(),
            6
        );
        assert!(high_water.pending_capacity <= 32);
        assert!(high_water.touched_word_capacity <= 32);
        assert!(high_water.pending_capacity < counts.iter().sum::<usize>() / 16);
    }
}
