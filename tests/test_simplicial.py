"""Tests for simplicial complex operations — generic API.

All tests use the new generic API with no backward compatibility references.
No hardcoded 0/1/2 dimensions, no v_mask/e_mask/f_mask, no V/E/F properties.
"""

import numpy as np
import pytest

from polygeo.simplicial import SimplicialComplex, SimplicialSubset

# ───────────────────────── fixtures ──────────────────────────


@pytest.fixture
def tetrahedron() -> SimplicialComplex:
    """Standard tetrahedron: 4 vertices, 6 edges, 4 faces (no 3-simplices)."""
    V = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]])
    F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return SimplicialComplex.from_mesh(V, F)


@pytest.fixture
def single_triangle() -> SimplicialComplex:
    """Minimal triangle: 3 vertices, 3 edges, 1 face."""
    V = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
    F = np.array([[0, 1, 2]])
    return SimplicialComplex.from_mesh(V, F)


@pytest.fixture
def tetrahedral_3d() -> SimplicialComplex:
    """Full 3D tetrahedron: 4 vertices, 6 edges, 4 faces, 1 volume."""
    V = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]], dtype=float)
    F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=int)
    edges_set = set()
    for tri in F:
        for i in range(3):
            edges_set.add(tuple(sorted([tri[i], tri[(i + 1) % 3]])))
    E = np.array(sorted(edges_set), dtype=int)
    V_idx = np.arange(V.shape[0]).reshape(-1, 1)
    T_arr = np.array([[0, 1, 2, 3]], dtype=int)
    return SimplicialComplex(V, [V_idx, E, F, T_arr])


# ───────────────────── construction tests ────────────────────


class TestConstruction:
    def test_basic_shapes(self, tetrahedron):
        sc = tetrahedron
        assert sc.n_simplices(0) == 4
        assert sc.n_simplices(1) == 6
        assert sc.n_simplices(2) == 4
        assert sc.dim == 2
        # Boundary operators exist
        assert sc.boundary_operator(1).shape == (4, 6)
        assert sc.boundary_operator(2).shape == (6, 4)

    def test_boundary_squared_is_zero(self, tetrahedron):
        sc = tetrahedron
        # ∂₁ ∂₂ = 0 (exactness)
        b1 = sc.boundary_operator(1)
        b2 = sc.boundary_operator(2)
        assert np.allclose((b1 @ b2).toarray(), 0)

    def test_incidence_matrix_sums(self, tetrahedron):
        sc = tetrahedron
        inc1 = sc.incidence_operator(1)
        inc2 = sc.incidence_operator(2)
        # Each edge has 2 vertices, each face has 3 edges
        assert np.allclose(inc1.toarray().sum(axis=0), 2)
        assert np.allclose(inc2.toarray().sum(axis=0), 3)

    def test_3d_boundary_squared_zero(self, tetrahedral_3d):
        sc = tetrahedral_3d
        assert sc.n_simplices(3) == 1
        assert sc.boundary_operator(1).shape == (4, 6)
        assert sc.boundary_operator(2).shape == (6, 4)
        assert sc.boundary_operator(3).shape == (4, 1)
        # ∂₁∂₂ = 0, ∂₂∂₃ = 0
        assert np.allclose(
            (sc.boundary_operator(1) @ sc.boundary_operator(2)).toarray(), 0
        )
        assert np.allclose(
            (sc.boundary_operator(2) @ sc.boundary_operator(3)).toarray(), 0
        )


# ─────────────────── subset creation tests ───────────────────


class TestSubsetCreation:
    def test_subset_by_dim_and_indices(self, tetrahedron):
        sub = SimplicialSubset({0: np.array([True, True, False, False])}, tetrahedron)
        assert np.array_equal(sub.chains[0], [True, True, False, False])

    def test_subset_by_boolean_mask(self, tetrahedron):
        mask = np.array([True, False, True, False])
        sub = SimplicialSubset({0: mask.copy()}, tetrahedron)
        assert np.array_equal(sub.chains[0], mask)

    def test_subset_combined_dimensions(self, tetrahedron):
        sub = SimplicialSubset(
            {
                0: np.array([True, False, False, False]),
                1: np.array([True, True, False, False, False, False]),
            },
            tetrahedron,
        )
        assert np.array_equal(sub.chains[0], [True, False, False, False])
        assert np.array_equal(sub.chains[1], [True, True, False, False, False, False])

    def test_empty_subset(self, tetrahedron):
        sub = SimplicialSubset({}, tetrahedron)
        assert not sub.chains

    def test_invalid_chain_length_raises(self, tetrahedron):
        with pytest.raises(AssertionError):
            SimplicialSubset({0: np.array([True, False])}, tetrahedron)


# ─────────────────── star operation tests ─────────────────────


class TestStar:
    def test_star_vertex(self, tetrahedron):
        star = SimplicialSubset(
            {0: np.array([True, False, False, False])}, tetrahedron
        ).star()
        assert np.array_equal(star.chains[0], [True, False, False, False])
        assert np.array_equal(star.chains[1], [True, True, True, False, False, False])
        assert np.array_equal(star.chains[2], [True, True, True, False])

    def test_star_edge(self, tetrahedron):
        star = SimplicialSubset(
            {1: np.array([False, False, False, False, False, True])}, tetrahedron
        ).star()
        assert np.array_equal(star.chains[1], [False, False, False, False, False, True])
        assert np.array_equal(star.chains[2], [False, False, True, True])

    def test_star_face(self, tetrahedron):
        star = SimplicialSubset(
            {2: np.array([True, False, False, False])}, tetrahedron
        ).star()
        assert np.array_equal(star.chains[2], [True, False, False, False])

    def test_star_union_of_vertices(self, tetrahedron):
        s01 = SimplicialSubset(
            {0: np.array([True, True, False, False])}, tetrahedron
        ).star()
        s0 = SimplicialSubset(
            {0: np.array([True, False, False, False])}, tetrahedron
        ).star()
        s1 = SimplicialSubset(
            {0: np.array([False, True, False, False])}, tetrahedron
        ).star()
        for d in s01.chains:
            assert np.array_equal(s01.chains[d], s0.chains[d] | s1.chains[d])


# ─────────────────── closure operation tests ──────────────────


class TestClosure:
    def test_closure_vertex(self, tetrahedron):
        cl = SimplicialSubset(
            {0: np.array([True, False, False, False])}, tetrahedron
        ).closure()
        assert np.array_equal(cl.chains[0], [True, False, False, False])

    def test_closure_edge(self, tetrahedron):
        cl = SimplicialSubset(
            {1: np.array([False, False, False, False, False, True])}, tetrahedron
        ).closure()
        assert np.array_equal(cl.chains[0], [False, False, True, True])
        assert np.array_equal(cl.chains[1], [False, False, False, False, False, True])

    def test_closure_face(self, tetrahedron):
        cl = SimplicialSubset(
            {2: np.array([True, False, False, False])}, tetrahedron
        ).closure()
        assert np.array_equal(cl.chains[0], [True, True, True, False])
        assert np.array_equal(cl.chains[1], [True, True, False, True, False, False])
        assert np.array_equal(cl.chains[2], [True, False, False, False])

    def test_closure_monotone_and_idempotent(self, tetrahedron):
        sub = SimplicialSubset({0: np.array([True, False, False, False])}, tetrahedron)
        cl = sub.closure()
        for d in sub.chains:
            assert np.all(sub.chains[d] <= cl.chains[d])
        cl2 = cl.closure()
        for d in cl.chains:
            assert np.array_equal(cl.chains[d], cl2.chains[d])


# ───────────────────── link operation tests ───────────────────


class TestLink:
    def test_link_vertex_tetrahedron(self, tetrahedron):
        lk = SimplicialSubset(
            {0: np.array([True, False, False, False])}, tetrahedron
        ).link()
        assert np.array_equal(lk.chains[1], [True, True, True, False, False, False])
        assert np.array_equal(lk.chains[2], [True, True, True, False])

    def test_link_edge_tetrahedron(self, tetrahedron):
        lk = SimplicialSubset(
            {1: np.array([False, False, False, False, False, True])}, tetrahedron
        ).link()
        assert np.array_equal(lk.chains[2], [False, False, True, True])

    def test_link_face_empty(self, tetrahedron):
        lk = SimplicialSubset(
            {2: np.array([True, False, False, False])}, tetrahedron
        ).link()
        for d in lk.chains:
            assert not np.any(lk.chains[d])

    def test_link_definition_star_minus_closure(self, tetrahedron):
        sc = tetrahedron
        for v_i in range(sc.n_simplices(0)):
            mask = np.zeros(sc.n_simplices(0), dtype=bool)
            mask[v_i] = True
            sub = SimplicialSubset({0: mask}, sc)
            link = sub.link()
            star = sub.star()
            cl = sub.closure()
            for d in set(list(star.chains.keys()) + list(cl.chains.keys())):
                assert np.array_equal(
                    link.chains.get(d, np.zeros(sc.n_simplices(d), dtype=bool)),
                    star.chains.get(d, np.zeros(sc.n_simplices(d), dtype=bool))
                    & ~cl.chains.get(d, np.zeros(sc.n_simplices(d), dtype=bool)),
                )


# ─────────────────── is_complex tests ────────────────────────


class TestIsComplex:
    def test_single_vertex_is_complex(self, tetrahedron):
        assert SimplicialSubset(
            {0: np.array([True, False, False, False])}, tetrahedron
        ).is_complex()

    def test_single_edge_not_complex(self, tetrahedron):
        assert not SimplicialSubset(
            {1: np.array([False, False, False, False, False, True])}, tetrahedron
        ).is_complex()

    def test_single_face_not_complex(self, tetrahedron):
        assert not SimplicialSubset(
            {2: np.array([True, False, False, False])}, tetrahedron
        ).is_complex()

    def test_closure_always_complex(self, tetrahedron):
        sub = SimplicialSubset({0: np.array([True, True, False, False])}, tetrahedron)
        assert sub.closure().is_complex()

    def test_full_complex_is_complex(self, tetrahedron):
        whole = SimplicialSubset(
            {
                0: np.ones(tetrahedron.n_simplices(0), dtype=bool),
                1: np.ones(tetrahedron.n_simplices(1), dtype=bool),
                2: np.ones(tetrahedron.n_simplices(2), dtype=bool),
            },
            tetrahedron,
        )
        assert whole.is_complex()


# ─────────────────── is_pure tests ───────────────────────────


class TestIsPure:
    def test_degree_zero(self, tetrahedron):
        assert SimplicialSubset(
            {0: np.array([True, False, False, False])}, tetrahedron
        ).is_pure() == (True, 0)
        assert SimplicialSubset(
            {0: np.array([True, True, True, False])}, tetrahedron
        ).is_pure() == (True, 0)

    def test_degree_one(self, tetrahedron):
        # A closed edge (edge with its vertices) is a pure 1-complex
        e5 = SimplicialSubset(
            {1: np.array([False, False, False, False, False, True])}, tetrahedron
        ).closure()
        assert e5.is_pure() == (True, 1)

    def test_degree_two(self, tetrahedron):
        # Closure of a face includes its edges and vertices; pure degree 2
        f0 = SimplicialSubset(
            {2: np.array([True, False, False, False])}, tetrahedron
        ).closure()
        assert f0.is_pure() == (True, 2)
        # Whole complex (all faces, edges, vertices) is pure degree 2
        whole = SimplicialSubset(
            {
                0: np.ones(tetrahedron.n_simplices(0), dtype=bool),
                1: np.ones(tetrahedron.n_simplices(1), dtype=bool),
                2: np.ones(tetrahedron.n_simplices(2), dtype=bool),
            },
            tetrahedron,
        )
        assert whole.is_pure() == (True, 2)

    def test_mixed_dimensions_not_pure(self, tetrahedron):
        # Vertices+edge (not closed) → not a complex → not pure
        assert SimplicialSubset(
            {
                0: np.array([True, False, False, False]),
                1: np.array([False, False, False, False, False, True]),
            },
            tetrahedron,
        ).is_pure() == (False, -1)

    def test_empty_subset(self, tetrahedron):
        assert SimplicialSubset({}, tetrahedron).is_pure() == (True, -1)

    def test_triangle_global_pure(self, single_triangle):
        assert single_triangle.dim == 2
        whole = SimplicialSubset(
            {
                0: np.ones(single_triangle.n_simplices(0), dtype=bool),
                1: np.ones(single_triangle.n_simplices(1), dtype=bool),
                2: np.ones(single_triangle.n_simplices(2), dtype=bool),
            },
            single_triangle,
        )
        assert whole.is_pure() == (True, 2)


# ─────────────────── boundary operation tests ──────────────────


class TestBoundary:
    def test_boundary_of_closed_edge(self, tetrahedron):
        closed_edge = SimplicialSubset(
            {1: np.array([False, False, False, False, False, True])}, tetrahedron
        ).closure()
        bd = closed_edge.boundary()
        assert np.array_equal(bd.chains[0], [False, False, True, True])

    def test_boundary_of_closed_face(self, tetrahedron):
        closed_face = SimplicialSubset(
            {2: np.array([True, False, False, False])}, tetrahedron
        ).closure()
        bd = closed_face.boundary()
        assert np.array_equal(bd.chains[0], [True, True, True, False])
        assert np.array_equal(bd.chains[1], [True, True, False, True, False, False])

    def test_boundary_of_two_adjacent_faces(self, tetrahedron):
        closed = SimplicialSubset(
            {2: np.array([True, True, False, False])}, tetrahedron
        ).closure()
        bd = closed.boundary()
        assert np.array_equal(bd.chains[1], [False, True, True, True, True, False])
        assert np.array_equal(bd.chains[0], [True, True, True, True])

    def test_boundary_of_closed_complex_empty(self, tetrahedron):
        whole = SimplicialSubset(
            {
                0: np.ones(tetrahedron.n_simplices(0), dtype=bool),
                1: np.ones(tetrahedron.n_simplices(1), dtype=bool),
                2: np.ones(tetrahedron.n_simplices(2), dtype=bool),
            },
            tetrahedron,
        )
        bd = whole.boundary()
        for d in bd.chains:
            assert not np.any(bd.chains[d])

    def test_boundary_non_pure_raises(self, tetrahedron):
        with pytest.raises(ValueError):
            SimplicialSubset(
                {
                    0: np.array([True, False, False, False]),
                    1: np.array([False, False, False, False, False, True]),
                },
                tetrahedron,
            ).boundary()

    def test_boundary_of_degree_zero_empty(self, tetrahedron):
        bd = SimplicialSubset(
            {0: np.array([True, False, False, False])}, tetrahedron
        ).boundary()
        for d in bd.chains:
            assert not np.any(bd.chains[d])

    def test_boundary_of_pure_one_single_edge(self, tetrahedron):
        closed = SimplicialSubset(
            {1: np.array([False, False, False, False, False, True])}, tetrahedron
        ).closure()
        bd = closed.boundary()
        assert np.array_equal(bd.chains[0], [False, False, True, True])

    def test_boundary_of_pure_one_multiple_edges(self, tetrahedron):
        closed = SimplicialSubset(
            {1: np.array([True, True, False, False, False, False])}, tetrahedron
        ).closure()
        bd = closed.boundary()
        assert np.array_equal(bd.chains[0], [False, True, True, False])


# ─────────────────── integration invariants ────────────────────


class TestIntegrationInvariants:
    def test_star_contains_original_all_types(self, tetrahedron):
        for v_i in range(tetrahedron.n_simplices(0)):
            mask = np.zeros(tetrahedron.n_simplices(0), dtype=bool)
            mask[v_i] = True
            sub = SimplicialSubset({0: mask}, tetrahedron)
            star = sub.star()
            for d in sub.chains:
                assert np.all(sub.chains[d] <= star.chains[d])
        for e_i in range(tetrahedron.n_simplices(1)):
            mask = np.zeros(tetrahedron.n_simplices(1), dtype=bool)
            mask[e_i] = True
            sub = SimplicialSubset({1: mask}, tetrahedron)
            star = sub.star()
            for d in sub.chains:
                assert np.all(sub.chains[d] <= star.chains[d])
        for f_i in range(tetrahedron.n_simplices(2)):
            mask = np.zeros(tetrahedron.n_simplices(2), dtype=bool)
            mask[f_i] = True
            sub = SimplicialSubset({2: mask}, tetrahedron)
            star = sub.star()
            for d in sub.chains:
                assert np.all(sub.chains[d] <= star.chains[d])

    def test_closure_contains_original_all_types(self, tetrahedron):
        for v_i in range(tetrahedron.n_simplices(0)):
            mask = np.zeros(tetrahedron.n_simplices(0), dtype=bool)
            mask[v_i] = True
            sub = SimplicialSubset({0: mask}, tetrahedron)
            cl = sub.closure()
            for d in sub.chains:
                assert np.all(sub.chains[d] <= cl.chains[d])
        for e_i in range(tetrahedron.n_simplices(1)):
            mask = np.zeros(tetrahedron.n_simplices(1), dtype=bool)
            mask[e_i] = True
            sub = SimplicialSubset({1: mask}, tetrahedron)
            cl = sub.closure()
            for d in sub.chains:
                assert np.all(sub.chains[d] <= cl.chains[d])
        for f_i in range(tetrahedron.n_simplices(2)):
            mask = np.zeros(tetrahedron.n_simplices(2), dtype=bool)
            mask[f_i] = True
            sub = SimplicialSubset({2: mask}, tetrahedron)
            cl = sub.closure()
            for d in sub.chains:
                assert np.all(sub.chains[d] <= cl.chains[d])

    def test_link_equals_star_minus_closure_all_types(self, tetrahedron):
        sc = tetrahedron
        for v_i in range(sc.n_simplices(0)):
            mask = np.zeros(sc.n_simplices(0), dtype=bool)
            mask[v_i] = True
            sub = SimplicialSubset({0: mask}, sc)
            link = sub.link()
            star = sub.star()
            cl = sub.closure()
            for d in set(list(star.chains.keys()) + list(cl.chains.keys())):
                assert np.array_equal(
                    link.chains.get(d, np.zeros(sc.n_simplices(d), dtype=bool)),
                    star.chains.get(d, np.zeros(sc.n_simplices(d), dtype=bool))
                    & ~cl.chains.get(d, np.zeros(sc.n_simplices(d), dtype=bool)),
                )
        for e_i in range(sc.n_simplices(1)):
            mask = np.zeros(sc.n_simplices(1), dtype=bool)
            mask[e_i] = True
            sub = SimplicialSubset({1: mask}, sc)
            link = sub.link()
            star = sub.star()
            cl = sub.closure()
            for d in set(list(star.chains.keys()) + list(cl.chains.keys())):
                assert np.array_equal(
                    link.chains.get(d, np.zeros(sc.n_simplices(d), dtype=bool)),
                    star.chains.get(d, np.zeros(sc.n_simplices(d), dtype=bool))
                    & ~cl.chains.get(d, np.zeros(sc.n_simplices(d), dtype=bool)),
                )
        for f_i in range(sc.n_simplices(2)):
            mask = np.zeros(sc.n_simplices(2), dtype=bool)
            mask[f_i] = True
            sub = SimplicialSubset({2: mask}, sc)
            link = sub.link()
            star = sub.star()
            cl = sub.closure()
            for d in set(list(star.chains.keys()) + list(cl.chains.keys())):
                assert np.array_equal(
                    link.chains.get(d, np.zeros(sc.n_simplices(d), dtype=bool)),
                    star.chains.get(d, np.zeros(sc.n_simplices(d), dtype=bool))
                    & ~cl.chains.get(d, np.zeros(sc.n_simplices(d), dtype=bool)),
                )

    def test_boundary_of_boundary_empty_for_pure_complexes(self, tetrahedron):
        sc = tetrahedron
        cases = [
            SimplicialSubset({0: np.array([True, False, False, False])}, sc),
            SimplicialSubset(
                {1: np.array([False, False, False, False, False, True])}, sc
            ).closure(),
            SimplicialSubset(
                {1: np.array([True, True, False, False, False, False])}, sc
            ).closure(),
            SimplicialSubset({2: np.array([True, False, False, False])}, sc).closure(),
            SimplicialSubset({2: np.array([True, True, False, False])}, sc).closure(),
            SimplicialSubset(
                {
                    0: np.ones(sc.n_simplices(0), dtype=bool),
                    1: np.ones(sc.n_simplices(1), dtype=bool),
                    2: np.ones(sc.n_simplices(2), dtype=bool),
                },
                sc,
            ),
        ]
        for s in cases:
            assert s.is_pure()[0], f"Case not pure: {s.chains}"
            bd1 = s.boundary()
            bd2 = bd1.boundary()
            for d in bd2.chains:
                assert not np.any(bd2.chains[d])

    def test_boundary_preserves_pure_degree(self, tetrahedron):
        sub1 = SimplicialSubset(
            {1: np.array([True, True, False, False, False, False])}, tetrahedron
        ).closure()
        bd1 = sub1.boundary()
        assert bd1.is_pure() == (True, 0)
        sub2 = SimplicialSubset(
            {2: np.array([True, True, False, False])}, tetrahedron
        ).closure()
        bd2 = sub2.boundary()
        assert bd2.is_pure() == (True, 1)
