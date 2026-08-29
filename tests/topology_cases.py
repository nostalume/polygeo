"""Deterministic topology fixtures shared by independent law tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


type IndexArray = NDArray[np.int64]


def simplex_case(dimension: int, *, reversed_top: bool = False) -> IndexArray:
    """Return one maximal simplex, optionally with one orientation reversal."""
    row = np.arange(dimension + 1, dtype=np.int64)
    if reversed_top and dimension > 0:
        row[-2:] = row[-2:][::-1]
    return row[None, :]


def triangle_grid(side: int) -> IndexArray:
    """Return an oriented square grid with two faces per square."""
    if side < 1:
        raise ValueError("grid side must be positive")
    width = side + 1
    faces: list[tuple[int, int, int]] = []
    for row in range(side):
        for column in range(side):
            lower_left = row * width + column
            lower_right = lower_left + 1
            upper_left = lower_left + width
            upper_right = upper_left + 1
            faces.append((lower_left, lower_right, upper_right))
            faces.append((lower_left, upper_right, upper_left))
    return np.asarray(faces, dtype=np.int64)
