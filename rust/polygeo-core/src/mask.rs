use crate::TopologyError;
use crate::incidence::DegreeLayout;

const WORD_BITS: usize = u64::BITS as usize;

/// Canonical all-degree membership with word-aligned degree segments.
///
/// Logical lengths and offsets remain owned by the topology layout. This value
/// owns only canonical packed words.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PackedDegreeMasks {
    words: Box<[u64]>,
}

impl PackedDegreeMasks {
    pub(crate) fn words(&self) -> &[u64] {
        &self.words
    }

    pub(crate) fn contains(
        &self,
        degree: usize,
        index: usize,
        layout: &DegreeLayout,
    ) -> Result<bool, TopologyError> {
        let count = layout.count(degree)?;
        if index >= count {
            return Err(TopologyError::MaskIndexOutside);
        }
        let start = layout.word_offsets()[degree];
        Ok(self.words[start + index / WORD_BITS] & (1_u64 << (index % WORD_BITS)) != 0)
    }

    pub(crate) fn export_degree(
        &self,
        degree: usize,
        layout: &DegreeLayout,
    ) -> Result<Vec<bool>, TopologyError> {
        let count = layout.count(degree)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(count)
            .map_err(|_| TopologyError::Allocation)?;
        output.resize(count, false);
        self.write_degree(degree, layout, &mut output)?;
        Ok(output)
    }

    pub(crate) fn write_degree(
        &self,
        degree: usize,
        layout: &DegreeLayout,
        output: &mut [bool],
    ) -> Result<(), TopologyError> {
        let count = layout.count(degree)?;
        if output.len() != count {
            return Err(TopologyError::MaskShape);
        }
        for (index, selected) in output.iter_mut().enumerate() {
            *selected = self.contains(degree, index, layout)?;
        }
        Ok(())
    }
}

/// Mutable, private construction phase for an immutable packed mask.
#[derive(Debug)]
pub(crate) struct PackedDegreeMasksBuilder<'a> {
    layout: &'a DegreeLayout,
    words: Box<[u64]>,
}

impl<'a> PackedDegreeMasksBuilder<'a> {
    pub(crate) fn empty(layout: &'a DegreeLayout) -> Result<Self, TopologyError> {
        let word_count = layout.word_offsets().last().copied().unwrap_or(0);
        let mut words = Vec::new();
        words
            .try_reserve_exact(word_count)
            .map_err(|_| TopologyError::Allocation)?;
        words.resize(word_count, 0);
        Ok(Self {
            layout,
            words: words.into_boxed_slice(),
        })
    }

    pub(crate) fn from_mask(
        layout: &'a DegreeLayout,
        source: &PackedDegreeMasks,
    ) -> Result<Self, TopologyError> {
        let mut builder = Self::empty(layout)?;
        if source.words().len() != builder.words.len() {
            return Err(TopologyError::MaskShape);
        }
        builder.words.copy_from_slice(source.words());
        Ok(builder)
    }

    pub(crate) fn set(
        &mut self,
        degree: usize,
        index: usize,
        selected: bool,
    ) -> Result<(), TopologyError> {
        let count = self.layout.count(degree)?;
        if index >= count {
            return Err(TopologyError::MaskIndexOutside);
        }
        let position = self.layout.word_offsets()[degree] + index / WORD_BITS;
        let bit = 1_u64 << (index % WORD_BITS);
        if selected {
            self.words[position] |= bit;
        } else {
            self.words[position] &= !bit;
        }
        Ok(())
    }

    pub(crate) fn contains(&self, degree: usize, index: usize) -> Result<bool, TopologyError> {
        let count = self.layout.count(degree)?;
        if index >= count {
            return Err(TopologyError::MaskIndexOutside);
        }
        let position = self.layout.word_offsets()[degree] + index / WORD_BITS;
        Ok(self.words[position] & (1_u64 << (index % WORD_BITS)) != 0)
    }

    pub(crate) fn finish(self) -> PackedDegreeMasks {
        PackedDegreeMasks { words: self.words }
    }
}

#[cfg(test)]
mod tests {
    use super::PackedDegreeMasksBuilder;
    use crate::TopologyError;
    use crate::incidence::DegreeLayout;
    use proptest::prelude::*;

    #[test]
    fn checked_prefix_classifies_overflow() {
        assert_eq!(
            DegreeLayout::from_counts(&[usize::MAX; 65]),
            Err(TopologyError::CountOverflow)
        );
    }

    #[test]
    fn private_builder_enforces_set_bounds() {
        let layout = DegreeLayout::from_counts(&[65]).unwrap();
        let mut builder = PackedDegreeMasksBuilder::empty(&layout).unwrap();
        builder.set(0, 0, true).unwrap();
        builder.set(0, 64, true).unwrap();
        assert_eq!(
            builder.set(0, 65, true),
            Err(TopologyError::MaskIndexOutside)
        );
        let masks = builder.finish();

        assert!(masks.contains(0, 0, &layout).unwrap());
        assert!(masks.contains(0, 64, &layout).unwrap());
    }

    proptest! {
        #[test]
        fn arbitrary_builder_round_trip(
            degrees in prop::collection::vec(
                prop::collection::vec(any::<bool>(), 0..180),
                0..12,
            )
        ) {
            let counts = degrees.iter().map(Vec::len).collect::<Vec<_>>();
            let layout = DegreeLayout::from_counts(&counts).unwrap();
            let mut builder = PackedDegreeMasksBuilder::empty(&layout).unwrap();
            for (degree, values) in degrees.iter().enumerate() {
                for (index, selected) in values.iter().copied().enumerate() {
                    builder.set(degree, index, selected).unwrap();
                }
            }
            let masks = builder.finish();
            for (degree, expected) in degrees.iter().enumerate() {
                prop_assert_eq!(masks.export_degree(degree, &layout).unwrap(), expected.as_slice());
                let word_count = expected.len().div_ceil(64);
                if word_count > 0 && !expected.len().is_multiple_of(64) {
                    let last = layout.word_offsets()[degree + 1] - 1;
                    prop_assert_eq!(masks.words()[last] >> (expected.len() % 64), 0);
                }
            }
        }
    }
}
