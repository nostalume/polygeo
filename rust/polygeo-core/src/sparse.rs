/// Invalid immutable compressed-row physical storage.
pub(crate) struct InvalidCompressedRows;

/// Immutable coefficient-independent compressed-row indexing.
#[derive(Debug, Eq, PartialEq)]
pub struct CsrPattern {
    pub(crate) shape: (usize, usize),
    pub(crate) row_offsets: Box<[usize]>,
    pub(crate) column_indices: Box<[usize]>,
}

impl CsrPattern {
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        self.shape
    }

    #[must_use]
    pub const fn nnz(&self) -> usize {
        self.column_indices.len()
    }

    #[must_use]
    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    #[must_use]
    pub fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }
}

/// Raw compressed-row storage with no mathematical-map identity.
#[derive(Debug, Eq, PartialEq)]
pub struct CsrMatrix<V> {
    pub(crate) pattern: CsrPattern,
    pub(crate) values: V,
}

impl<V> CsrMatrix<V> {
    #[must_use]
    pub const fn pattern(&self) -> &CsrPattern {
        &self.pattern
    }

    #[must_use]
    pub const fn values(&self) -> &V {
        &self.values
    }
}

impl<T> CsrMatrix<Box<[T]>> {
    pub(crate) fn try_from_parts(
        shape: (usize, usize),
        row_offsets: impl Into<Box<[usize]>>,
        column_indices: impl Into<Box<[usize]>>,
        values: impl Into<Box<[T]>>,
    ) -> Result<Self, InvalidCompressedRows> {
        let (row_offsets, column_indices, values) =
            (row_offsets.into(), column_indices.into(), values.into());
        let valid_layout = row_offsets.len()
            == shape.0.checked_add(1).ok_or(InvalidCompressedRows)?
            && row_offsets.first() == Some(&0)
            && row_offsets.last() == Some(&column_indices.len())
            && column_indices.len() == values.len();
        let valid_rows = row_offsets.windows(2).all(|offsets| {
            column_indices
                .get(offsets[0]..offsets[1])
                .is_some_and(|row| {
                    row.iter().all(|&column| column < shape.1)
                        && row.windows(2).all(|pair| pair[0] < pair[1])
                })
        });
        if !valid_layout || !valid_rows {
            return Err(InvalidCompressedRows);
        }
        Ok(Self {
            pattern: CsrPattern {
                shape,
                row_offsets,
                column_indices,
            },
            values,
        })
    }

    pub(crate) fn row(&self, row: usize) -> Option<(&[usize], &[T])> {
        let offsets = self.pattern.row_offsets.get(row..=row.checked_add(1)?)?;
        Some((
            self.pattern.column_indices.get(offsets[0]..offsets[1])?,
            self.values.get(offsets[0]..offsets[1])?,
        ))
    }

    pub(crate) fn rows(&self) -> impl Iterator<Item = (&[usize], &[T])> {
        (0..self.pattern.shape.0).map(|row| {
            self.row(row)
                .expect("validated compressed rows retain every declared row")
        })
    }
}
