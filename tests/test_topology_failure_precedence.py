"""Topology admission preserves compound-failure precedence."""

from __future__ import annotations

import numpy as np
import pytest

from polygeo.topology import Complex as NativeComplex
from polygeo.topology import SimplicialError as NativeSimplicialError


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (np.array([[0, 0, -1]], dtype=np.int64), "negative_index"),
        (
            np.array([[0, 1, 1], [0, -1, 2]], dtype=np.int64),
            "repeated_vertex",
        ),
        (
            np.array([[0, 1, 2], [2, 1, 0], [0, -1, 3]], dtype=np.int64),
            "negative_index",
        ),
    ],
)
def test_admission_preserves_compound_invalid_error_precedence(
    candidate: np.ndarray, reason: str
) -> None:
    with pytest.raises(NativeSimplicialError) as caught:
        NativeComplex.from_maximal_simplices(candidate)

    assert caught.value.reason == reason
