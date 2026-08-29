use std::collections::BTreeMap;

use polygeo_core::HalfedgeInput;

pub fn input(next: Vec<usize>, twin: Vec<usize>, exterior_seeds: Vec<usize>) -> HalfedgeInput {
    let halfedge_count = next.len();
    HalfedgeInput::unsigned(next, twin, exterior_seeds, halfedge_count).unwrap()
}

pub fn empty_surface() -> HalfedgeInput {
    input(Vec::new(), Vec::new(), Vec::new())
}

pub fn polygon_disk(vertex_count: usize) -> HalfedgeInput {
    assert!(vertex_count != 0);
    from_oriented_faces(&[(0..vertex_count).collect::<Vec<_>>()])
}

pub fn annulus() -> HalfedgeInput {
    from_oriented_faces(&[
        vec![0, 1, 5, 4],
        vec![1, 2, 6, 5],
        vec![2, 3, 7, 6],
        vec![3, 0, 4, 7],
    ])
}

pub fn unigon() -> HalfedgeInput {
    from_oriented_faces(&[vec![0]])
}

pub fn one_vertex_torus() -> HalfedgeInput {
    input(vec![1, 2, 0, 4, 5, 3], vec![4, 5, 3, 2, 0, 1], Vec::new())
}

pub fn disconnected_triangles() -> HalfedgeInput {
    from_oriented_faces(&[vec![0, 1, 2], vec![3, 4, 5]])
}

#[allow(dead_code)]
pub fn triangulated_grid(side: usize) -> HalfedgeInput {
    let stride = side + 1;
    let mut faces = Vec::with_capacity(2 * side * side);
    for row in 0..side {
        for column in 0..side {
            let lower_left = row * stride + column;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + stride;
            let upper_right = upper_left + 1;
            faces.push(vec![lower_left, lower_right, upper_right]);
            faces.push(vec![lower_left, upper_right, upper_left]);
        }
    }
    from_oriented_faces(&faces)
}

fn from_oriented_faces(faces: &[Vec<usize>]) -> HalfedgeInput {
    let material_count = faces.iter().map(Vec::len).sum::<usize>();
    let mut next = vec![usize::MAX; material_count];
    let mut twin = vec![usize::MAX; material_count];
    let mut directed = Vec::with_capacity(material_count);
    let mut offset = 0;
    for face in faces {
        assert!(!face.is_empty());
        for corner in 0..face.len() {
            let halfedge = offset + corner;
            next[halfedge] = offset + (corner + 1) % face.len();
            directed.push((face[corner], face[(corner + 1) % face.len()]));
        }
        offset += face.len();
    }

    let mut pending = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for (halfedge, edge) in directed.iter().copied().enumerate() {
        let reverse = (edge.1, edge.0);
        if edge.0 != edge.1
            && let Some(reverse_halfedges) = pending.get_mut(&reverse)
            && let Some(reverse_halfedge) = reverse_halfedges.pop()
        {
            twin[halfedge] = reverse_halfedge;
            twin[reverse_halfedge] = halfedge;
        } else {
            pending.entry(edge).or_default().push(halfedge);
        }
    }

    let unmatched = twin
        .iter()
        .enumerate()
        .filter_map(|(halfedge, paired)| (*paired == usize::MAX).then_some(halfedge))
        .collect::<Vec<_>>();
    next.reserve(unmatched.len());
    twin.reserve(unmatched.len());
    let mut exterior_by_origin = BTreeMap::new();
    for material in unmatched {
        let exterior = next.len();
        let (origin, destination) = directed[material];
        assert!(
            exterior_by_origin.insert(destination, exterior).is_none(),
            "fixture boundary must have one outgoing exterior halfedge per vertex"
        );
        next.push(usize::MAX);
        twin.push(material);
        twin[material] = exterior;
        directed.push((destination, origin));
    }
    for exterior in material_count..next.len() {
        let destination = directed[exterior].1;
        next[exterior] = exterior_by_origin[&destination];
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
    input(next, twin, exterior_seeds)
}
