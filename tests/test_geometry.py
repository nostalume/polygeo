from __future__ import annotations

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

    np.testing.assert_array_equal(geometry.simplex_measures(0), np.ones(4))
    np.testing.assert_allclose(
        geometry.simplex_measures(1),
        [1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)],
    )
    np.testing.assert_allclose(
        geometry.simplex_measures(2),
        [0.5, 0.5, 0.5, np.sqrt(3.0) / 2.0],
    )
    np.testing.assert_allclose(geometry.simplex_measures(3), [1.0 / 6.0])


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
    np.testing.assert_allclose(geometry.simplex_measures(4), [1.0 / 24.0])


def test_zero_dimensional_geometry_has_unit_measures() -> None:
    complex_ = Complex.from_maximal_simplices(np.array([[0], [1]], dtype=np.int64))
    geometry = Geometry.from_positions(
        complex_,
        np.empty((2, 0), dtype=np.float64),
    )

    assert geometry.ambient_dimension == 0
    np.testing.assert_array_equal(geometry.simplex_measures(0), [1.0, 1.0])


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

    np.testing.assert_allclose(geometry.simplex_measures(2), [0.5])


@pytest.mark.parametrize("scale", [1e-150, 1e-75, 1.0, 1e75, 1e150])
def test_simplex_measures_follow_scale_law(scale: float) -> None:
    geometry = Geometry.from_positions(
        _triangle(),
        _right_triangle_positions() * scale,
    )

    np.testing.assert_allclose(
        geometry.simplex_measures(1) / scale,
        [1.0, 1.0, np.sqrt(2.0)],
        rtol=3e-15,
    )
    np.testing.assert_allclose(
        geometry.simplex_measures(2) / (scale * scale),
        [0.5],
        rtol=3e-15,
    )


def test_skinny_representable_triangle_is_not_rejected_by_absolute_scale() -> None:
    positions = np.array(
        [[0.0, 0.0], [1e150, 0.0], [0.0, 1e-150]],
        dtype=np.float64,
    )
    geometry = Geometry.from_positions(_triangle(), positions)

    np.testing.assert_allclose(geometry.simplex_measures(2), [0.5])


def test_extreme_anisotropy_does_not_underflow_during_normalization() -> None:
    positions = np.array(
        [[0.0, 0.0], [1e170, 0.0], [0.0, 1e-170]],
        dtype=np.float64,
    )
    geometry = Geometry.from_positions(_triangle(), positions)

    np.testing.assert_allclose(geometry.simplex_measures(2), [0.5])


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
        geometry.simplex_measures(3),
        [(displaced - 9.0) / 2.0],
        rtol=3e-15,
    )


def test_public_measures_are_caller_owned() -> None:
    geometry = Geometry.from_positions(_triangle(), _right_triangle_positions())

    exposed = geometry.simplex_measures(2)
    exposed[0] = 99.0

    np.testing.assert_allclose(geometry.simplex_measures(2), [0.5])


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


def test_geometry_rejects_invalid_measure_degree() -> None:
    geometry = Geometry.from_positions(_triangle(), _right_triangle_positions())

    for degree in (-1, geometry.complex.dimension + 1):
        with pytest.raises(GeometryError):
            geometry.simplex_measures(degree)
