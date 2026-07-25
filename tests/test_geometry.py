from __future__ import annotations

import math
from fractions import Fraction
from itertools import permutations

import numpy as np
import pytest

from polygeo import (
    BoundaryUnknown,
    Complex,
    ConnectivityUnknown,
    Geometry,
    GeometryError,
    OrientationUnknown,
    Simplicial,
)


type RawComplex = Complex[
    BoundaryUnknown,
    OrientationUnknown,
    ConnectivityUnknown,
    Simplicial,
]


def _triangle() -> RawComplex:
    return Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))


def _right_triangle_positions() -> np.ndarray:
    return np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )


def _tetrahedron() -> RawComplex:
    return Complex.from_maximal_simplices(np.array([[0, 1, 2, 3]], dtype=np.int64))


def _tetrahedron_positions() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _oracle_circumcenter(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 1:
        return points[0].copy(), np.ones(1)
    edges = (points[1:] - points[0]).T
    gram = edges.T @ edges
    coefficients = np.linalg.solve(gram, 0.5 * np.diag(gram))
    center = points[0] + edges @ coefficients
    barycentric = np.concatenate(([1.0 - float(np.sum(coefficients))], coefficients))
    return center, barycentric


def _explicit_flag_dual_measures(
    complex_: RawComplex,
    positions: np.ndarray,
    degree: int,
) -> np.ndarray:
    centers: dict[tuple[int, ...], np.ndarray] = {}
    barycentric: dict[tuple[int, ...], np.ndarray] = {}
    for simplex_degree in range(complex_.dimension + 1):
        for simplex in complex_.simplices(simplex_degree):
            identity = tuple(int(vertex) for vertex in simplex)
            centers[identity], barycentric[identity] = _oracle_circumcenter(
                positions[simplex]
            )

    top = tuple(int(vertex) for vertex in complex_.simplices(complex_.dimension)[0])
    values = []
    for simplex in complex_.simplices(degree):
        lower = tuple(int(vertex) for vertex in simplex)
        remaining = tuple(vertex for vertex in top if vertex not in lower)
        contributions = []
        for order in permutations(remaining):
            product = 1.0
            current = lower
            for added in order:
                upper = tuple(sorted((*current, added)))
                local = upper.index(added)
                sign = float(np.sign(barycentric[upper][local]))
                product *= sign * float(
                    np.linalg.norm(centers[upper] - centers[current])
                )
                current = upper
            contributions.append(product / math.factorial(complex_.dimension - degree))
        values.append(math.fsum(contributions))
    return np.array(values, dtype=np.float64)


def _exact_triangle_dual_edge_steps(points: np.ndarray) -> np.ndarray:
    exact = [[Fraction(float(value)) for value in point] for point in points]
    edges = [
        [exact[vertex][coordinate] - exact[0][coordinate] for vertex in (1, 2)]
        for coordinate in range(2)
    ]
    gram = [
        [
            sum((row[left] * row[right] for row in edges), start=Fraction())
            for right in range(2)
        ]
        for left in range(2)
    ]
    determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
    right = [gram[0][0] / 2, gram[1][1] / 2]
    coefficients = [
        (right[0] * gram[1][1] - gram[0][1] * right[1]) / determinant,
        (gram[0][0] * right[1] - right[0] * gram[1][0]) / determinant,
    ]
    barycentric = [1 - sum(coefficients, start=Fraction()), *coefficients]
    twice_area = abs(edges[0][0] * edges[1][1] - edges[1][0] * edges[0][1])
    values = []
    for edge in _triangle().simplices(1):
        omitted = next(vertex for vertex in range(3) if vertex not in edge)
        edge_length = math.hypot(*(points[edge[1]] - points[edge[0]]))
        height = twice_area / Fraction(edge_length)
        values.append(float(barycentric[omitted] * height))
    return np.array(values, dtype=np.float64)


def test_geometry_owns_positions_and_exact_complex() -> None:
    complex_ = _triangle()
    source = _right_triangle_positions()

    geometry = Geometry(complex_, source)
    source[0, 0] = 99.0
    exposed = geometry.positions
    exposed[1, 1] = 88.0

    assert geometry.complex is complex_
    assert geometry.ambient_dimension == 2
    np.testing.assert_array_equal(
        geometry.positions,
        _right_triangle_positions(),
    )


def test_geometry_computes_canonical_measures_through_dimension_three() -> None:
    geometry = Geometry.from_positions(
        _tetrahedron(),
        _tetrahedron_positions(),
    )

    np.testing.assert_array_equal(geometry.primal_measures(0), np.ones(4))
    np.testing.assert_allclose(
        geometry.primal_measures(1),
        [1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)],
    )
    np.testing.assert_allclose(
        geometry.primal_measures(2),
        [0.5, 0.5, 0.5, np.sqrt(3.0) / 2.0],
    )
    np.testing.assert_allclose(geometry.primal_measures(3), [1.0 / 6.0])


def test_four_simplex_uses_the_same_runtime_dimension_path() -> None:
    complex_ = Complex.from_maximal_simplices(
        np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
    )
    positions = np.vstack(
        [
            np.zeros((1, 4), dtype=np.float64),
            np.eye(4, dtype=np.float64),
        ]
    )
    geometry = Geometry.from_positions(complex_, positions)

    assert geometry.complex.dimension == 4
    np.testing.assert_allclose(geometry.primal_measures(4), [1.0 / 24.0])


def test_dual_recurrence_matches_explicit_flags_through_dimension_four() -> None:
    for dimension in range(1, 5):
        complex_ = Complex.from_maximal_simplices(
            np.arange(dimension + 1, dtype=np.int64).reshape(1, -1)
        )
        rng = np.random.default_rng(100 + dimension)
        positions = rng.normal(size=(dimension + 1, dimension))
        geometry = Geometry.from_positions(complex_, positions)

        for degree in range(dimension + 1):
            np.testing.assert_allclose(
                geometry.dual_measures(degree),
                _explicit_flag_dual_measures(complex_, positions, degree),
                rtol=2e-13,
                atol=2e-14,
            )


def test_zero_dimensional_geometry_has_unit_measures() -> None:
    complex_ = Complex.from_maximal_simplices(np.array([[0], [1]], dtype=np.int64))
    geometry = Geometry.from_positions(
        complex_,
        np.empty((2, 0), dtype=np.float64),
    )

    assert geometry.ambient_dimension == 0
    np.testing.assert_array_equal(geometry.primal_measures(0), [1.0, 1.0])


def test_zero_dimensional_geometry_has_unit_dual_measures() -> None:
    complex_ = Complex.from_maximal_simplices(np.array([[0], [1]], dtype=np.int64))
    geometry = Geometry.from_positions(
        complex_,
        np.empty((2, 0), dtype=np.float64),
    )

    np.testing.assert_array_equal(geometry.dual_measures(0), [1.0, 1.0])


def test_segment_dual_measures_are_endpoint_half_lengths() -> None:
    complex_ = Complex.from_maximal_simplices(np.array([[0, 1]], dtype=np.int64))
    geometry = Geometry.from_positions(
        complex_,
        np.array([[0.0], [6.0]], dtype=np.float64),
    )

    np.testing.assert_allclose(geometry.dual_measures(0), [3.0, 3.0])
    np.testing.assert_array_equal(geometry.dual_measures(1), [1.0])


def test_segment_dual_measures_are_invariant_under_large_translation() -> None:
    complex_ = Complex.from_maximal_simplices(np.array([[0, 1]], dtype=np.int64))
    geometry = Geometry.from_positions(
        complex_,
        np.array([[1e16], [1e16 + 2.0]], dtype=np.float64),
    )

    np.testing.assert_array_equal(geometry.dual_measures(0), [1.0, 1.0])


def test_path_dual_measure_accumulates_both_incident_half_edges() -> None:
    complex_ = Complex.from_maximal_simplices(
        np.array([[0, 1], [1, 2]], dtype=np.int64)
    )
    geometry = Geometry.from_positions(
        complex_,
        np.array([[0.0], [2.0], [8.0]], dtype=np.float64),
    )

    np.testing.assert_allclose(geometry.dual_measures(0), [1.0, 4.0, 3.0])


def test_right_triangle_dual_measures_preserve_zero_circumcentric_step() -> None:
    geometry = Geometry.from_positions(_triangle(), _right_triangle_positions())

    np.testing.assert_allclose(geometry.dual_measures(0), [0.25, 0.125, 0.125])
    np.testing.assert_allclose(geometry.dual_measures(1), [0.5, 0.5, 0.0])
    np.testing.assert_array_equal(geometry.dual_measures(2), [1.0])


def test_obtuse_triangle_preserves_negative_dual_edge_measure() -> None:
    positions = np.array(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, 0.5]],
        dtype=np.float64,
    )
    geometry = Geometry.from_positions(_triangle(), positions)

    np.testing.assert_allclose(geometry.dual_measures(1)[0], -0.75)


def test_non_delaunay_interior_dual_edge_matches_signed_cotangent_oracle() -> None:
    complex_ = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
    )
    positions = np.array(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, 0.5], [0.0, -0.5]],
        dtype=np.float64,
    )
    geometry = Geometry.from_positions(complex_, positions)

    edge = positions[1] - positions[0]
    cotangents = []
    for opposite in (positions[2], positions[3]):
        left = positions[0] - opposite
        right = positions[1] - opposite
        twice_area = abs(float(np.linalg.det(np.stack([left, right]))))
        cotangents.append(float(left @ right) / twice_area)

    dual_edge = geometry.dual_measures(1)[0]
    edge_measure = float(np.linalg.norm(edge))

    np.testing.assert_allclose(dual_edge, -1.5)
    np.testing.assert_allclose(
        dual_edge / edge_measure,
        0.5 * math.fsum(cotangents),
    )


def test_representable_tiny_local_dual_step_is_not_lost_to_center_rounding() -> None:
    positions = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0**-1000, 1.0]],
        dtype=np.float64,
    )
    geometry = Geometry.from_positions(_triangle(), positions)

    np.testing.assert_allclose(
        geometry.dual_measures(1),
        _exact_triangle_dual_edge_steps(positions),
        rtol=3e-15,
        atol=0.0,
    )


def test_ill_conditioned_dual_steps_use_forward_accurate_fallback() -> None:
    positions = np.array(
        [
            [0.0, 0.0],
            [20.842767409029946, 200.87462942870152],
            [1.823885314534666e-12, 1.7577909856790797e-11],
        ],
        dtype=np.float64,
    )
    geometry = Geometry.from_positions(_triangle(), positions)
    exact = [[Fraction(float(value)) for value in point] for point in positions]
    first = [exact[1][coordinate] - exact[0][coordinate] for coordinate in range(2)]
    second = [exact[2][coordinate] - exact[0][coordinate] for coordinate in range(2)]
    exact_area = abs(first[0] * second[1] - first[1] * second[0]) / 2

    np.testing.assert_array_equal(geometry.primal_measures(2), [float(exact_area)])
    np.testing.assert_allclose(
        geometry.dual_measures(1),
        _exact_triangle_dual_edge_steps(positions),
        rtol=3e-15,
    )


def test_moderate_condition_dual_steps_use_forward_accurate_fallback() -> None:
    positions = np.array(
        [
            [0.0, 0.0],
            [0.18598548689879482, 0.9825524915560584],
            [0.15226025576998198, 0.8043888253898179],
        ],
        dtype=np.float64,
    )
    geometry = Geometry.from_positions(_triangle(), positions)

    np.testing.assert_allclose(
        geometry.dual_measures(1),
        _exact_triangle_dual_edge_steps(positions),
        rtol=3e-15,
    )


def test_triangle_measure_is_independent_of_higher_ambient_dimension() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    geometry = Geometry.from_positions(_triangle(), positions)

    np.testing.assert_allclose(geometry.primal_measures(2), [0.5])


@pytest.mark.parametrize("scale", [1e-150, 1e-75, 1.0, 1e75, 1e150])
def test_primal_measures_follow_scale_law(scale: float) -> None:
    geometry = Geometry.from_positions(
        _triangle(),
        _right_triangle_positions() * scale,
    )

    np.testing.assert_allclose(
        geometry.primal_measures(1) / scale,
        [1.0, 1.0, np.sqrt(2.0)],
        rtol=3e-15,
    )
    np.testing.assert_allclose(
        geometry.primal_measures(2) / (scale * scale),
        [0.5],
        rtol=3e-15,
    )


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_dual_measures_follow_complementary_scale_law(scale: float) -> None:
    geometry = Geometry.from_positions(
        _triangle(),
        _right_triangle_positions() * scale,
    )

    np.testing.assert_allclose(
        geometry.dual_measures(0) / (scale * scale),
        [0.25, 0.125, 0.125],
        rtol=5e-14,
    )
    np.testing.assert_allclose(
        geometry.dual_measures(1) / scale,
        [0.5, 0.5, 0.0],
        rtol=5e-14,
        atol=0.0,
    )
    np.testing.assert_array_equal(geometry.dual_measures(2), [1.0])


def test_dual_measures_are_invariant_under_rigid_higher_embedding() -> None:
    base = Geometry.from_positions(_triangle(), _right_triangle_positions())
    embedded = Geometry.from_positions(
        _triangle(),
        np.array(
            [
                [2.0, -3.0, 5.0, 7.0],
                [2.0, -2.0, 5.0, 7.0],
                [1.0, -3.0, 5.0, 7.0],
            ],
            dtype=np.float64,
        ),
    )

    for degree in range(3):
        np.testing.assert_allclose(
            embedded.dual_measures(degree),
            base.dual_measures(degree),
        )


def test_public_dual_measures_are_caller_owned() -> None:
    geometry = Geometry.from_positions(_triangle(), _right_triangle_positions())

    exposed = geometry.dual_measures(0)
    exposed[:] = 99.0

    np.testing.assert_allclose(geometry.dual_measures(0), [0.25, 0.125, 0.125])


def test_dual_measure_computation_adds_no_geometry_state() -> None:
    assert Geometry.__slots__ == ("_complex", "_measures", "_positions")


def test_skinny_representable_triangle_is_not_rejected_by_absolute_scale() -> None:
    positions = np.array(
        [[0.0, 0.0], [1e150, 0.0], [0.0, 1e-150]],
        dtype=np.float64,
    )
    geometry = Geometry.from_positions(_triangle(), positions)

    np.testing.assert_allclose(geometry.primal_measures(2), [0.5])


def test_extreme_anisotropy_does_not_underflow_during_normalization() -> None:
    positions = np.array(
        [[0.0, 0.0], [1e170, 0.0], [0.0, 1e-170]],
        dtype=np.float64,
    )
    geometry = Geometry.from_positions(_triangle(), positions)

    np.testing.assert_allclose(geometry.primal_measures(2), [0.5])


def test_dual_measures_do_not_depend_on_global_numpy_error_policy() -> None:
    positions = np.array(
        [[0.0, 0.0], [1e170, 0.0], [0.0, 1e-170]],
        dtype=np.float64,
    )
    geometry = Geometry.from_positions(_triangle(), positions)
    previous = np.seterr(all="raise")
    try:
        np.testing.assert_allclose(geometry.dual_measures(1), [5e-171, 5e169, 0.0])
    finally:
        np.seterr(**previous)


def test_subnormal_dual_measures_do_not_leak_strict_policy_underflow() -> None:
    positions = np.array(
        [[0.0, 0.0], [2.0**-1072, 0.0], [0.0, 2.0]],
        dtype=np.float64,
    )
    geometry = Geometry.from_positions(_triangle(), positions)
    previous = np.seterr(all="raise")
    try:
        dual_0 = geometry.dual_measures(0)
        dual_1 = geometry.dual_measures(1)
        dual_2 = geometry.dual_measures(2)
    finally:
        np.seterr(**previous)

    np.testing.assert_array_equal(dual_0, [1e-323, 5e-324, 5e-324])
    np.testing.assert_array_equal(dual_1, [1.0, 1e-323, 0.0])
    np.testing.assert_array_equal(dual_2, [1.0])


def test_dual_measures_translate_unrepresentable_circumcenter_failure() -> None:
    positions = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 5e-310]],
        dtype=np.float64,
    )
    geometry = Geometry.from_positions(_triangle(), positions)

    with pytest.raises(GeometryError, match="not representable"):
        geometry.dual_measures(1)


def test_near_dependent_full_rank_simplex_uses_accurate_measure() -> None:
    displaced = np.nextafter(9.0, np.inf)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [5.0, 7.0, displaced],
        ],
        dtype=np.float64,
    )
    geometry = Geometry.from_positions(_tetrahedron(), positions)

    np.testing.assert_allclose(
        geometry.primal_measures(3),
        [(displaced - 9.0) / 2.0],
        rtol=3e-15,
    )


def test_public_measures_are_caller_owned() -> None:
    geometry = Geometry.from_positions(_triangle(), _right_triangle_positions())

    exposed = geometry.primal_measures(2)
    exposed[0] = 99.0

    np.testing.assert_allclose(geometry.primal_measures(2), [0.5])


@pytest.mark.parametrize(
    "positions",
    [
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 2), dtype=np.float32),
        np.zeros((3, 2), dtype=np.int64),
        np.array([[0.0, 0.0], [np.nan, 0.0], [0.0, 1.0]]),
        np.array([[0.0, 0.0], [np.inf, 0.0], [0.0, 1.0]]),
    ],
)
def test_geometry_rejects_misaligned_inexact_or_nonfinite_positions(
    positions: np.ndarray,
) -> None:
    with pytest.raises(GeometryError):
        Geometry.from_positions(_triangle(), positions)


@pytest.mark.parametrize(
    "complex_, positions",
    [
        (
            _triangle(),
            np.array(
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                dtype=np.float64,
            ),
        ),
        (
            _tetrahedron(),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
        ),
        (
            _tetrahedron(),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [5.0, 7.0, 9.0],
                ],
                dtype=np.float64,
            ),
        ),
        (
            _triangle(),
            np.array(
                [
                    [
                        float.fromhex("-0x1.357728374370cp+140"),
                        float.fromhex("0x1.b053e4abb7fcep+52"),
                    ],
                    [
                        float.fromhex("-0x1.90bf2907b7da2p+145"),
                        float.fromhex("-0x1.1c159f13b29c4p+53"),
                    ],
                    [
                        float.fromhex("-0x1.5f8b3e7219d36p+143"),
                        float.fromhex("0x1.d074ee9c977e0p+51"),
                    ],
                ],
                dtype=np.float64,
            ),
        ),
        (
            _triangle(),
            np.array(
                [[-1e308, 0.0], [1e308, 0.0], [0.0, 1.0]],
                dtype=np.float64,
            ),
        ),
    ],
)
def test_geometry_rejects_degenerate_or_unrepresentable_simplices(
    complex_: RawComplex,
    positions: np.ndarray,
) -> None:
    with pytest.raises(GeometryError):
        Geometry.from_positions(complex_, positions)


def test_geometry_rejects_invalid_measure_degrees() -> None:
    geometry = Geometry.from_positions(_triangle(), _right_triangle_positions())

    for degree in (-1, geometry.complex.dimension + 1):
        with pytest.raises(GeometryError):
            geometry.primal_measures(degree)
        with pytest.raises(GeometryError):
            geometry.dual_measures(degree)


def test_geometry_has_no_retired_simplex_measure_alias() -> None:
    geometry = Geometry.from_positions(_triangle(), _right_triangle_positions())

    assert not hasattr(geometry, "simplex_measures")
