use std::sync::Arc;

use crate::TopologyError;
use crate::evidence::EvidenceStore;
use crate::incidence::{
    Basis, CanonicalBoundary, CanonicalMaximal, ChainView, DegreeLayout, build_boundaries,
    canonical_faces, try_filled, validated_permutation_sign,
};
use crate::{ChainComplex, IntegerRing, IntegralChainComplex};

#[derive(Debug)]
pub struct CandidateInput {
    values: Box<[usize]>,
    signs: Box<[i8]>,
    row_count: usize,
    row_width: usize,
    vertex_count: Option<usize>,
}

impl CandidateInput {
    /// Admit signed transport values into one checked native-index buffer.
    ///
    /// # Errors
    ///
    /// Returns a stable shape, sign-domain, native-index, overflow, or
    /// allocation error before a candidate can reach canonicalization.
    pub fn signed<T>(
        values: impl IntoIterator<Item = T>,
        row_count: usize,
        row_width: usize,
        vertex_count: Option<usize>,
    ) -> Result<Self, TopologyError>
    where
        i128: From<T>,
    {
        Self::checked(
            values.into_iter().map(|value| {
                let value = i128::from(value);
                if value < 0 {
                    Err(TopologyError::negative_index(value))
                } else {
                    usize::try_from(value)
                        .map_err(|_| TopologyError::index_overflow(value.cast_unsigned()))
                }
            }),
            row_count,
            row_width,
            vertex_count,
        )
    }

    /// Admit unsigned transport values into one checked native-index buffer.
    ///
    /// # Errors
    ///
    /// Returns a stable shape, native-index, overflow, or allocation error.
    pub fn unsigned<T>(
        values: impl IntoIterator<Item = T>,
        row_count: usize,
        row_width: usize,
        vertex_count: Option<usize>,
    ) -> Result<Self, TopologyError>
    where
        u128: From<T>,
    {
        Self::checked(
            values.into_iter().map(|value| {
                let value = u128::from(value);
                usize::try_from(value).map_err(|_| TopologyError::index_overflow(value))
            }),
            row_count,
            row_width,
            vertex_count,
        )
    }

    fn checked(
        values: impl IntoIterator<Item = Result<usize, TopologyError>>,
        row_count: usize,
        row_width: usize,
        vertex_count: Option<usize>,
    ) -> Result<Self, TopologyError> {
        let expected = row_count
            .checked_mul(row_width)
            .ok_or(TopologyError::CountOverflow)?;
        let mut admitted = Vec::new();
        let mut signs = Vec::new();
        admitted
            .try_reserve_exact(expected)
            .map_err(|_| TopologyError::Allocation)?;
        signs
            .try_reserve_exact(row_count)
            .map_err(|_| TopologyError::Allocation)?;
        for value in values {
            if admitted.len() == expected {
                return Err(TopologyError::CandidateShape);
            }
            admitted.push(value?);
            if row_width != 0 && admitted.len().is_multiple_of(row_width) {
                signs.push(validated_permutation_sign(
                    &admitted[admitted.len() - row_width..],
                )?);
            }
        }
        if admitted.len() != expected {
            return Err(TopologyError::CandidateShape);
        }
        Ok(Self {
            values: admitted.into_boxed_slice(),
            signs: signs.into_boxed_slice(),
            row_count,
            row_width,
            vertex_count,
        })
    }
}

/// Immutable admitted topology owner.
#[derive(Debug)]
pub struct ComplexCore {
    pub(crate) vertex_count: usize,
    pub(crate) dimension: usize,
    pub(crate) layout: DegreeLayout,
    pub(crate) bases: Box<[Basis]>,
    pub(crate) orientations: Box<[Box<[i8]>]>,
    pub(crate) boundaries: Box<[CanonicalBoundary]>,
    pub(crate) immediate_faces: Box<[Box<[usize]>]>,
    pub(crate) evidence: EvidenceStore,
}

impl ComplexCore {
    /// Admit one candidate into an immutable owner.
    ///
    /// # Errors
    ///
    /// Returns a classified topology error when admission fails.
    pub fn admit(candidate: CandidateInput) -> Result<Arc<Self>, TopologyError> {
        let canonical = canonicalize_maximal(candidate)?;
        let vertex_count = canonical.vertex_count;
        let dimension = canonical.rows.row_width() - 1;
        let mut bases = Vec::new();
        bases
            .try_reserve_exact(dimension + 1)
            .map_err(|_| TopologyError::Allocation)?;
        for degree in 0..=dimension {
            let degree_basis = if degree == 0 {
                Basis::from_flat((0..vertex_count).collect(), vertex_count, 1)?
            } else {
                canonical_faces(&canonical.rows, degree + 1)?
            };
            bases.push(degree_basis);
        }
        let layout = DegreeLayout::from_bases(&bases)?;

        let mut orientations = Vec::new();
        orientations
            .try_reserve_exact(dimension + 1)
            .map_err(|_| TopologyError::Allocation)?;
        for (degree, basis) in bases.iter().enumerate() {
            let signs = if degree == dimension {
                if basis.values() != canonical.rows.values() {
                    return Err(TopologyError::InternalInvariant);
                }
                canonical.signs.to_vec()
            } else {
                vec![1_i8; basis.row_count()]
            };
            orientations.push(signs.into_boxed_slice());
        }

        let (boundaries, immediate_faces) = build_boundaries(&bases, &orientations)?;
        Ok(Arc::new(Self {
            vertex_count,
            dimension,
            layout,
            bases: bases.into_boxed_slice(),
            orientations: orientations.into_boxed_slice(),
            boundaries: boundaries.into_boxed_slice(),
            immediate_faces: immediate_faces.into_boxed_slice(),
            evidence: EvidenceStore::default(),
        }))
    }

    /// Retain this topology owner as an exact integral chain complex.
    #[must_use]
    pub fn chain_complex(self: &Arc<Self>) -> IntegralChainComplex {
        ChainComplex::simplicial(Arc::clone(self), IntegerRing)
    }

    /// Number of vertices in the admitted extent.
    #[must_use]
    pub const fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Maximum represented simplex degree.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Borrow this topology through the sealed exact chain interface.
    #[must_use]
    pub const fn chain_view(&self) -> ChainView<'_> {
        ChainView::simplicial(self)
    }

    /// Canonical basis for one represented degree.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::DegreeOutside`] for an unrepresented degree.
    pub fn basis(&self, degree: usize) -> Result<&Basis, TopologyError> {
        self.bases
            .get(degree)
            .ok_or(TopologyError::degree_outside(degree))
    }

    /// Canonical immediate-face indices, grouped by upper simplex.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::DegreeOutside`] for an unrepresented degree.
    pub fn immediate_faces(&self, degree: usize) -> Result<&[usize], TopologyError> {
        self.immediate_faces
            .get(degree)
            .map(AsRef::as_ref)
            .ok_or(TopologyError::degree_outside(degree))
    }

    /// Orientation signs aligned with one canonical basis.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::DegreeOutside`] for an unrepresented degree.
    pub fn orientation(&self, degree: usize) -> Result<&[i8], TopologyError> {
        self.orientations
            .get(degree)
            .map(AsRef::as_ref)
            .ok_or(TopologyError::degree_outside(degree))
    }

    /// Retained signed boundary for one represented degree.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::DegreeOutside`] for an unrepresented degree.
    pub fn boundary(&self, degree: usize) -> Result<&CanonicalBoundary, TopologyError> {
        self.boundaries
            .get(degree)
            .ok_or(TopologyError::degree_outside(degree))
    }

    /// Flattened lower-basis row indexes for every source simplex and removed vertex.
    ///
    /// For degree `k > 0`, entries are grouped in canonical source-column order
    /// with `k + 1` entries per column. Degree zero has no immediate faces.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::DegreeOutside`] for an unrepresented degree.
    pub fn immediate_face_rows(&self, degree: usize) -> Result<&[usize], TopologyError> {
        self.immediate_faces
            .get(degree)
            .map(AsRef::as_ref)
            .ok_or(TopologyError::degree_outside(degree))
    }
}

fn canonicalize_maximal(candidate: CandidateInput) -> Result<CanonicalMaximal, TopologyError> {
    let CandidateInput {
        values,
        signs,
        row_count,
        row_width,
        vertex_count: declared_vertex_count,
    } = candidate;
    if row_count == 0 || row_width == 0 {
        return Err(TopologyError::EmptyMaximalSimplices);
    }

    let mut rows = values.into_vec();
    let mut signs = signs.into_vec();
    let mut maximum = 0_usize;
    for raw_row in rows.chunks_exact(row_width) {
        maximum = maximum.max(raw_row.iter().copied().max().unwrap_or(0));
    }

    let inferred_vertices = maximum.checked_add(1).ok_or(TopologyError::CountOverflow)?;
    let vertex_count = declared_vertex_count.unwrap_or(inferred_vertices);
    if vertex_count == 0 || vertex_count <= maximum {
        return Err(TopologyError::vertex_extent(
            vertex_count,
            inferred_vertices,
        ));
    }

    for row in rows.chunks_exact_mut(row_width) {
        row.sort_unstable();
    }
    let mut order = (0..row_count).collect::<Vec<_>>();
    order.sort_unstable_by(|left, right| {
        let left_start = left * row_width;
        let right_start = right * row_width;
        rows[left_start..left_start + row_width].cmp(&rows[right_start..right_start + row_width])
    });
    for pair in order.windows(2) {
        let left_start = pair[0] * row_width;
        let right_start = pair[1] * row_width;
        if rows[left_start..left_start + row_width] == rows[right_start..right_start + row_width] {
            return Err(TopologyError::DuplicateMaximalSimplex);
        }
    }

    let mut visited = try_filled(row_count, false)?;
    let mut saved_row = Vec::new();
    saved_row
        .try_reserve_exact(row_width)
        .map_err(|_| TopologyError::Allocation)?;
    for start in 0..row_count {
        if visited[start] {
            continue;
        }
        saved_row.extend_from_slice(&rows[start * row_width..(start + 1) * row_width]);
        let saved_sign = signs[start];
        let mut destination = start;
        loop {
            visited[destination] = true;
            let source = order[destination];
            if source == start {
                rows[destination * row_width..(destination + 1) * row_width]
                    .copy_from_slice(&saved_row);
                signs[destination] = saved_sign;
                break;
            }
            rows.copy_within(
                source * row_width..(source + 1) * row_width,
                destination * row_width,
            );
            signs[destination] = signs[source];
            destination = source;
        }
        saved_row.clear();
    }
    Ok(CanonicalMaximal {
        vertex_count,
        rows: Basis::from_flat(rows, row_count, row_width)?,
        signs: signs.into_boxed_slice(),
    })
}
