use std::collections::BTreeMap;
use std::sync::Arc;

use crate::{
    CandidateInput, ChainIsomorphism, ChainLawLimit, ComplexCore, HalfedgeInput,
    HalfedgeSurfaceCore, IntegerRing, IsomorphismError, MaterialFace, TopologyError,
};

const UNASSIGNED: usize = usize::MAX;

#[derive(Debug)]
pub(crate) struct SignedPermutation {
    target_of_source: Box<[usize]>,
    source_of_target: Box<[usize]>,
    signs: Box<[i8]>,
}

impl SignedPermutation {
    pub(crate) fn admit(
        target_of_source: Vec<usize>,
        signs: Vec<i8>,
    ) -> Result<Self, TopologyError> {
        if target_of_source.len() != signs.len() {
            return Err(TopologyError::CorrespondenceLaw);
        }
        let mut source_of_target = vec![UNASSIGNED; target_of_source.len()];
        for (source, (&target, &sign)) in target_of_source.iter().zip(&signs).enumerate() {
            if target >= source_of_target.len()
                || source_of_target[target] != UNASSIGNED
                || !matches!(sign, -1 | 1)
            {
                return Err(TopologyError::CorrespondenceLaw);
            }
            source_of_target[target] = source;
        }
        Ok(Self {
            target_of_source: target_of_source.into_boxed_slice(),
            source_of_target: source_of_target.into_boxed_slice(),
            signs: signs.into_boxed_slice(),
        })
    }

    #[must_use]
    pub(crate) const fn len(&self) -> usize {
        self.target_of_source.len()
    }

    #[must_use]
    pub(crate) const fn target_of_source(&self) -> &[usize] {
        &self.target_of_source
    }

    #[must_use]
    pub(crate) const fn signs(&self) -> &[i8] {
        &self.signs
    }

    /// Map one admitted source basis index and its orientation sign.
    ///
    /// # Errors
    ///
    /// Returns `degree_outside` when the index is outside this permutation.
    pub(crate) fn map_basis(&self, source: usize) -> Result<(usize, i8), TopologyError> {
        self.target_of_source
            .get(source)
            .copied()
            .zip(self.signs.get(source).copied())
            .ok_or(TopologyError::degree_outside(source))
    }

    /// Invert one admitted target basis index and its orientation sign.
    ///
    /// # Errors
    ///
    /// Returns `degree_outside` when the index is outside this permutation.
    pub(crate) fn inverse_basis(&self, target: usize) -> Result<(usize, i8), TopologyError> {
        let source = self
            .source_of_target
            .get(target)
            .copied()
            .ok_or(TopologyError::degree_outside(target))?;
        Ok((source, self.signs[source]))
    }
}

impl HalfedgeSurfaceCore {
    /// Construct a distinct halfedge owner from an admitted oriented triangle complex.
    ///
    /// # Errors
    ///
    /// Returns a capability, topology, allocation, or correspondence-law failure
    /// without publishing either result.
    pub fn from_complex(
        complex: &Arc<ComplexCore>,
    ) -> Result<(Arc<Self>, ChainIsomorphism<IntegerRing>), IsomorphismError> {
        Self::from_complex_with_limit(complex, ChainLawLimit::DEFAULT)
    }

    /// Construct with explicit chain-law lifecycle and term ceilings.
    ///
    /// # Errors
    ///
    /// Returns a classified construction, resource, or verification failure
    /// without publishing a result.
    pub fn from_complex_with_limit(
        complex: &Arc<ComplexCore>,
        limit: ChainLawLimit,
    ) -> Result<(Arc<Self>, ChainIsomorphism<IntegerRing>), IsomorphismError> {
        complex
            .require_triangle()
            .map_err(IsomorphismError::Topology)?;
        complex
            .require_oriented()
            .map_err(IsomorphismError::Topology)?;
        let (input, directed) =
            halfedges_from_complex(complex).map_err(IsomorphismError::Topology)?;
        let surface = Self::admit(input).map_err(IsomorphismError::Topology)?;
        let degrees = complex_to_surface_degrees(complex, &surface, &directed)
            .map_err(IsomorphismError::Topology)?;
        let correspondence = ChainIsomorphism::admit_signed(
            complex.chain_complex(),
            surface.chain_complex(),
            Vec::from(degrees),
            limit,
        )?;
        Ok((surface, correspondence))
    }

    /// Construct a distinct simplicial owner when every material face is triangular.
    ///
    /// # Errors
    ///
    /// Returns `conversion_not_simplicial` for unsupported quotient or polygonal
    /// presentations, or another classified construction failure.
    pub fn to_complex(
        self: &Arc<Self>,
    ) -> Result<(Arc<ComplexCore>, ChainIsomorphism<IntegerRing>), IsomorphismError> {
        self.to_complex_with_limit(ChainLawLimit::DEFAULT)
    }

    /// Convert with explicit chain-law lifecycle and term ceilings.
    ///
    /// # Errors
    ///
    /// Returns a classified conversion, resource, or verification failure
    /// without publishing a result.
    pub fn to_complex_with_limit(
        self: &Arc<Self>,
        limit: ChainLawLimit,
    ) -> Result<(Arc<ComplexCore>, ChainIsomorphism<IntegerRing>), IsomorphismError> {
        let (complex, degrees) = complex_from_surface(self).map_err(IsomorphismError::Topology)?;
        let correspondence = ChainIsomorphism::admit_signed(
            self.chain_complex(),
            complex.chain_complex(),
            Vec::from(degrees),
            limit,
        )?;
        Ok((complex, correspondence))
    }
}

fn complex_from_surface(
    surface: &Arc<HalfedgeSurfaceCore>,
) -> Result<(Arc<ComplexCore>, [SignedPermutation; 3]), TopologyError> {
    let mut rows = Vec::new();
    let row_entries = surface
        .material_face_count()
        .checked_mul(3)
        .ok_or(TopologyError::CountOverflow)?;
    rows.try_reserve_exact(row_entries)
        .map_err(|_| TopologyError::Allocation)?;
    for face in surface.material_faces() {
        let vertices = triangle_vertices(face)?;
        if vertices[0] == vertices[1] || vertices[1] == vertices[2] || vertices[0] == vertices[2] {
            return Err(TopologyError::ConversionNotSimplicial);
        }
        for vertex in vertices {
            rows.push(u64::try_from(vertex).map_err(|_| TopologyError::IndexOverflow)?);
        }
    }
    let candidate = CandidateInput::unsigned(
        rows,
        surface.material_face_count(),
        3,
        Some(surface.vertex_count()),
    )?;
    let complex =
        ComplexCore::admit(candidate).map_err(|_| TopologyError::ConversionNotSimplicial)?;
    complex
        .refine_triangle()
        .map_err(|_| TopologyError::ConversionNotSimplicial)?;
    complex
        .refine_oriented()
        .map_err(|_| TopologyError::ConversionNotSimplicial)?;
    let degrees = surface_to_complex_degrees(surface, &complex)?;
    Ok((complex, degrees))
}

fn triangle_vertices(face: MaterialFace<'_>) -> Result<[usize; 3], TopologyError> {
    let mut halfedges = face.halfedges();
    let vertices = {
        let mut next = || {
            halfedges
                .next()
                .map(|halfedge| halfedge.vertex().index())
                .ok_or(TopologyError::ConversionNotSimplicial)
        };
        [next()?, next()?, next()?]
    };
    if halfedges.next().is_some() {
        return Err(TopologyError::ConversionNotSimplicial);
    }
    Ok(vertices)
}

type DirectedEdge = (usize, usize);

fn halfedges_from_complex(
    complex: &ComplexCore,
) -> Result<(HalfedgeInput, Vec<DirectedEdge>), TopologyError> {
    let faces = complex.basis(2)?;
    let orientations = complex.orientation(2)?;
    let material_count = faces
        .row_count()
        .checked_mul(3)
        .ok_or(TopologyError::CountOverflow)?;
    let mut next = vec![UNASSIGNED; material_count];
    let mut twin = vec![UNASSIGNED; material_count];
    let mut directed = Vec::with_capacity(material_count);
    for (face_index, (row, &sign)) in faces.values().chunks_exact(3).zip(orientations).enumerate() {
        let oriented = if sign == 1 {
            [row[0], row[1], row[2]]
        } else {
            [row[0], row[2], row[1]]
        };
        let start = face_index * 3;
        for corner in 0..3 {
            next[start + corner] = start + (corner + 1) % 3;
            directed.push((oriented[corner], oriented[(corner + 1) % 3]));
        }
    }
    let mut pending = BTreeMap::<DirectedEdge, usize>::new();
    for (halfedge, edge) in directed.iter().copied().enumerate() {
        if let Some(reverse) = pending.remove(&(edge.1, edge.0)) {
            twin[halfedge] = reverse;
            twin[reverse] = halfedge;
        } else if pending.insert(edge, halfedge).is_some() {
            return Err(TopologyError::CorrespondenceLaw);
        }
    }
    let unmatched = twin
        .iter()
        .enumerate()
        .filter_map(|(index, &paired)| (paired == UNASSIGNED).then_some(index))
        .collect::<Vec<_>>();
    let mut exterior_by_origin = BTreeMap::new();
    for material in unmatched {
        let exterior = next.len();
        let (origin, destination) = directed[material];
        if exterior_by_origin.insert(destination, exterior).is_some() {
            return Err(TopologyError::CorrespondenceLaw);
        }
        next.push(UNASSIGNED);
        twin.push(material);
        twin[material] = exterior;
        directed.push((destination, origin));
    }
    for exterior in material_count..next.len() {
        next[exterior] = *exterior_by_origin
            .get(&directed[exterior].1)
            .ok_or(TopologyError::CorrespondenceLaw)?;
    }
    let mut exterior_seeds = Vec::new();
    let mut seen = vec![false; next.len()];
    for seed in material_count..next.len() {
        if seen[seed] {
            continue;
        }
        exterior_seeds.push(seed);
        let mut halfedge = seed;
        loop {
            seen[halfedge] = true;
            halfedge = next[halfedge];
            if halfedge == seed {
                break;
            }
        }
    }
    Ok((
        HalfedgeInput::native(
            next.into_boxed_slice(),
            twin.into_boxed_slice(),
            exterior_seeds.into_boxed_slice(),
        )?,
        directed,
    ))
}

fn complex_to_surface_degrees(
    complex: &ComplexCore,
    surface: &HalfedgeSurfaceCore,
    directed: &[DirectedEdge],
) -> Result<[SignedPermutation; 3], TopologyError> {
    let mut vertices = vec![UNASSIGNED; complex.vertex_count()];
    for (halfedge, &(origin, _)) in directed.iter().enumerate() {
        let target = surface.halfedge(halfedge)?.vertex().index();
        if vertices[origin] != UNASSIGNED && vertices[origin] != target {
            return Err(TopologyError::CorrespondenceLaw);
        }
        vertices[origin] = target;
    }
    let zero = SignedPermutation::admit(vertices, vec![1; complex.vertex_count()])?;

    let edge_basis = complex.basis(1)?;
    let edge_rows = edge_basis
        .values()
        .chunks_exact(2)
        .enumerate()
        .map(|(index, row)| ((row[0], row[1]), index))
        .collect::<BTreeMap<_, _>>();
    let mut edges = vec![UNASSIGNED; edge_basis.row_count()];
    let mut edge_signs = vec![0; edge_basis.row_count()];
    for edge in surface.edges() {
        let representative = edge.representative().index();
        let (origin, target) = directed[representative];
        let key = if origin < target {
            (origin, target)
        } else {
            (target, origin)
        };
        let source = *edge_rows
            .get(&key)
            .ok_or(TopologyError::CorrespondenceLaw)?;
        edges[source] = edge.index();
        edge_signs[source] = if (origin, target) == key { 1 } else { -1 };
    }
    let one = SignedPermutation::admit(edges, edge_signs)?;

    let face_count = complex.basis(2)?.row_count();
    let mut faces = vec![UNASSIGNED; face_count];
    for (source, target) in faces.iter_mut().enumerate() {
        let halfedge = source * 3;
        *target = surface
            .halfedge(halfedge)?
            .face_orbit()
            .as_material()
            .ok_or(TopologyError::CorrespondenceLaw)?
            .index();
    }
    let two = SignedPermutation::admit(faces, vec![1; face_count])?;
    Ok([zero, one, two])
}

fn surface_to_complex_degrees(
    surface: &HalfedgeSurfaceCore,
    complex: &ComplexCore,
) -> Result<[SignedPermutation; 3], TopologyError> {
    let zero = SignedPermutation::admit(
        (0..surface.vertex_count()).collect(),
        vec![1; surface.vertex_count()],
    )?;
    let edge_rows = complex
        .basis(1)?
        .values()
        .chunks_exact(2)
        .enumerate()
        .map(|(index, row)| ((row[0], row[1]), index))
        .collect::<BTreeMap<_, _>>();
    let mut edges = vec![UNASSIGNED; surface.edge_count()];
    let mut signs = vec![0; surface.edge_count()];
    for edge in surface.edges() {
        let representative = edge.representative();
        let origin = representative.vertex().index();
        let target = representative.twin().vertex().index();
        let key = if origin < target {
            (origin, target)
        } else {
            (target, origin)
        };
        edges[edge.index()] = *edge_rows
            .get(&key)
            .ok_or(TopologyError::CorrespondenceLaw)?;
        signs[edge.index()] = if (origin, target) == key { 1 } else { -1 };
    }
    let one = SignedPermutation::admit(edges, signs)?;
    let face_rows = complex
        .basis(2)?
        .values()
        .chunks_exact(3)
        .enumerate()
        .map(|(index, row)| ([row[0], row[1], row[2]], index))
        .collect::<BTreeMap<_, _>>();
    let mut faces = vec![UNASSIGNED; surface.material_face_count()];
    for face in surface.material_faces() {
        let source = face.index();
        let mut row = triangle_vertices(face)?;
        row.sort_unstable();
        faces[source] = *face_rows
            .get(&[row[0], row[1], row[2]])
            .ok_or(TopologyError::CorrespondenceLaw)?;
    }
    let two = SignedPermutation::admit(faces, vec![1; surface.material_face_count()])?;
    Ok([zero, one, two])
}

#[cfg(test)]
mod tests {
    use super::SignedPermutation;
    use crate::{
        CandidateInput, ChainIsomorphism, ChainLawLimit, ComplexCore, IsomorphismError,
        StorageLimit, WorkLimit,
    };

    fn triangle() -> std::sync::Arc<ComplexCore> {
        let candidate = CandidateInput::signed([0_i64, 1, 2], 1, 3, Some(3)).unwrap();
        ComplexCore::admit(candidate).unwrap()
    }

    fn identity(size: usize) -> SignedPermutation {
        SignedPermutation::admit((0..size).collect(), vec![1; size]).unwrap()
    }

    #[test]
    fn signed_permutation_rejects_false_inverse_claims() {
        assert!(SignedPermutation::admit(vec![0, 0], vec![1, 1]).is_err());
        assert!(SignedPermutation::admit(vec![0, 1], vec![1, 0]).is_err());
        assert!(SignedPermutation::admit(vec![0, 2], vec![1, 1]).is_err());
    }

    #[test]
    fn chain_isomorphism_rejects_false_commuting_law_and_budget_exhaustion() {
        let owner = triangle();
        let complex = owner.chain_complex();
        let invalid = vec![
            identity(3),
            identity(3),
            SignedPermutation::admit(vec![0], vec![-1]).unwrap(),
        ];
        assert_eq!(
            ChainIsomorphism::admit_signed(
                complex.clone(),
                complex.clone(),
                invalid,
                ChainLawLimit::DEFAULT
            )
            .unwrap_err(),
            IsomorphismError::InvalidCandidate
        );

        let valid = vec![identity(3), identity(3), identity(1)];
        assert_eq!(
            ChainIsomorphism::admit_signed(
                complex.clone(),
                complex,
                valid,
                ChainLawLimit::new(StorageLimit::new(0, 0).unwrap(), WorkLimit::new(0)),
            )
            .unwrap_err()
            .reason(),
            "resource_limit"
        );
    }
}
