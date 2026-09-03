"""Classified topology failures expose stable immutable evidence."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from polygeo.topology import Complex, SimplicialError


def test_failure_reason_is_stable_and_read_only() -> None:
    with pytest.raises(SimplicialError) as caught:
        Complex.from_maximal_simplices(np.array([[0, 1, 1]], dtype=np.int64))

    assert caught.value.reason == "repeated_vertex"
    with pytest.raises(AttributeError):
        setattr(caught.value, "reason", "changed")


def test_failure_details_are_immutable_structured_evidence() -> None:
    with pytest.raises(SimplicialError) as caught:
        Complex.from_maximal_simplices(np.array([[0, 1, 1]], dtype=np.int64))

    assert caught.value.reason == "repeated_vertex"
    assert caught.value.details == {"vertex": 1}
    with pytest.raises(TypeError):
        cast(dict[str, int | str], caught.value.details)["vertex"] = 2
