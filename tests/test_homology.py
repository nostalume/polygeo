"""Exact homology ownership, limits, generators, and execution laws."""

from __future__ import annotations

import gc
import threading
import time
from typing import Any, cast

import numpy as np
import pytest

from polygeo import (
    Complex,
    DEFAULT_HOMOLOGY_LIMIT,
    ChainError,
    HalfedgeSurface,
    HomologyError,
    HomologyGroup,
    HomologyLimit,
    IntegralHomology,
    QQ,
    prepare_integral_homology,
)


def test_homology_limits_are_sealed_immutable_and_classified() -> None:
    assert issubclass(HomologyError, ChainError)
    changed = DEFAULT_HOMOLOGY_LIMIT.replace(coefficient_bits=17, smith_steps=29)
    assert changed.coefficient_bits == 17
    assert changed.smith_steps == 29
    assert DEFAULT_HOMOLOGY_LIMIT.retained_logical_bytes > 0
    assert (
        DEFAULT_HOMOLOGY_LIMIT.retained_logical_bytes
        <= DEFAULT_HOMOLOGY_LIMIT.peak_live_logical_bytes
    )
    with pytest.raises(AttributeError):
        setattr(changed, "smith_steps", 3)
    with pytest.raises(TypeError):
        cast(Any, DEFAULT_HOMOLOGY_LIMIT.replace)(3)
    for invalid in (-1, 1 << 1000):
        with pytest.raises(HomologyError) as caught:
            DEFAULT_HOMOLOGY_LIMIT.replace(smith_steps=invalid)
        assert caught.value.reason == "limit"
        assert caught.value.details == {}
        with pytest.raises(TypeError):
            cast(Any, caught.value.details)["axis"] = "smith_steps"

    for carrier in (HomologyLimit, IntegralHomology, HomologyGroup):
        with pytest.raises(TypeError):
            cast(Any, carrier)()
        with pytest.raises(TypeError):
            type("Derived", (carrier,), {})


def cycle(count: int) -> Complex:
    vertices = np.arange(count, dtype=np.int64)
    return Complex.from_maximal_simplices(
        np.column_stack((vertices, np.roll(vertices, -1)))
    )


def grid(side: int) -> Complex:
    vertices = np.arange((side + 1) ** 2, dtype=np.int64).reshape(side + 1, side + 1)
    lower = np.column_stack(
        (
            vertices[:-1, :-1].ravel(),
            vertices[:-1, 1:].ravel(),
            vertices[1:, 1:].ravel(),
        )
    )
    upper = np.column_stack(
        (
            vertices[:-1, :-1].ravel(),
            vertices[1:, 1:].ravel(),
            vertices[1:, :-1].ravel(),
        )
    )
    return Complex.from_maximal_simplices(np.vstack((lower, upper)))


def test_integral_homology_is_canonical_borrowed_and_owner_retaining() -> None:
    complex_ = cycle(5)
    chain_complex = complex_.chain_complex()
    analysis = prepare_integral_homology(chain_complex, [1, 0, 1])

    assert isinstance(analysis, IntegralHomology)
    assert analysis.degrees == (0, 1)
    assert analysis[0].free_rank == 1
    group = analysis[1]
    assert group.free_rank == 1
    assert group.torsion_orders == ()
    representative = group.free_cycle(0)
    assert chain_complex.boundary(1).apply(representative).to_python_copy() == ((), ())

    del complex_, chain_complex, analysis
    gc.collect()
    assert group.free_cycle(0).to_python_copy() == representative.to_python_copy()
    assert representative.to_python_copy()[0]


def test_empty_degree_request_publishes_one_empty_retaining_analysis() -> None:
    chain_complex = cycle(3).chain_complex()
    analysis = prepare_integral_homology(chain_complex, [])

    assert analysis.degrees == ()
    with pytest.raises(HomologyError) as caught:
        analysis[0]
    assert caught.value.reason == "degree_not_requested"


def test_unrequested_degree_wrong_coefficients_and_resource_limits_fail_stably() -> (
    None
):
    chain_complex = cycle(4).chain_complex()
    with pytest.raises(HomologyError) as outside:
        prepare_integral_homology(chain_complex, [2])
    assert outside.value.reason == "degree_outside"

    analysis = prepare_integral_homology(chain_complex, [1])
    with pytest.raises(HomologyError) as unrequested:
        analysis[0]
    assert unrequested.value.reason == "degree_not_requested"
    with pytest.raises(HomologyError) as generator:
        analysis[1].free_cycle(1)
    assert generator.value.reason == "generator_outside"

    with pytest.raises(HomologyError) as rational:
        prepare_integral_homology(cast(Any, chain_complex.over(QQ)), [0])
    assert rational.value.reason == "coefficient_system"

    denied = DEFAULT_HOMOLOGY_LIMIT.replace(smith_steps=0)
    with pytest.raises(HomologyError) as exhausted:
        prepare_integral_homology(chain_complex, [0, 1], limit=denied)
    assert exhausted.value.reason == "resource_limit"
    details = cast(dict[str, int | str], exhausted.value.details)
    assert details["axis"] == "smith_steps"
    assert cast(int, details["required"]) > 0
    assert details["limit"] == 0


@pytest.mark.parametrize(
    ("axis", "changes"),
    [
        ("retained_logical_bytes", {"retained_logical_bytes": 0}),
        (
            "peak_live_logical_bytes",
            {"retained_logical_bytes": 1, "peak_live_logical_bytes": 1},
        ),
        ("coefficient_bits", {"coefficient_bits": 0}),
        ("smith_steps", {"smith_steps": 0}),
    ],
)
def test_each_homology_resource_axis_is_enforced(
    axis: str, changes: dict[str, int]
) -> None:
    limit = cast(Any, DEFAULT_HOMOLOGY_LIMIT.replace)(**changes)
    with pytest.raises(HomologyError) as caught:
        prepare_integral_homology(cycle(4).chain_complex(), [0, 1], limit=limit)
    assert caught.value.reason == "resource_limit"
    details = cast(dict[str, int | str], caught.value.details)
    assert details["axis"] == axis
    assert cast(int, details["required"]) > cast(int, details["limit"])


def test_homology_limit_rejects_an_invalid_storage_lifecycle() -> None:
    with pytest.raises(HomologyError) as caught:
        DEFAULT_HOMOLOGY_LIMIT.replace(
            retained_logical_bytes=DEFAULT_HOMOLOGY_LIMIT.peak_live_logical_bytes + 1
        )
    assert caught.value.reason == "limit"


def test_homology_rejects_duck_typed_inputs_and_group_construction() -> None:
    with pytest.raises(TypeError):
        prepare_integral_homology(cast(Any, object()), [0])
    with pytest.raises(TypeError):
        prepare_integral_homology(cycle(3).chain_complex(), cast(Any, ["0"]))
    with pytest.raises(TypeError):
        prepare_integral_homology(
            cycle(3).chain_complex(), [0], limit=cast(Any, object())
        )
    with pytest.raises(TypeError):
        cast(Any, HomologyGroup)()


def test_halfedge_chain_factory_is_an_equal_entrypoint() -> None:
    surface = HalfedgeSurface.from_permutations(
        np.array([1, 2, 0, 5, 3, 4], dtype=np.int64),
        np.array([3, 4, 5, 0, 1, 2], dtype=np.int64),
        exterior_faces=np.array([1], dtype=np.int64),
    )
    group = prepare_integral_homology(surface.chain_complex(), [0])[0]
    assert group.free_rank == 1
    assert group.torsion_orders == ()


def test_torsion_cycles_and_bounds_remain_exact_chains() -> None:
    projective_plane = Complex.from_maximal_simplices(
        np.array(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 4],
                [0, 3, 5],
                [0, 4, 5],
                [1, 2, 5],
                [1, 3, 4],
                [1, 4, 5],
                [2, 3, 4],
                [2, 3, 5],
            ],
            dtype=np.int64,
        )
    )
    chain_complex = projective_plane.chain_complex()
    group = prepare_integral_homology(chain_complex, [1])[1]
    assert group.free_rank == 0
    assert group.torsion_orders == (2,)
    cycle_indices, cycle_coefficients = group.torsion_cycle(0).to_python_copy()
    boundary_indices, boundary_coefficients = (
        chain_complex.boundary(2).apply(group.torsion_bound(0)).to_python_copy()
    )
    assert boundary_indices == cycle_indices
    assert boundary_coefficients == tuple(2 * value for value in cycle_coefficients)


def test_homology_preparation_releases_the_gil() -> None:
    chain_complex = grid(24).chain_complex()
    stop = threading.Event()
    samples: list[int] = []

    def sample_clock() -> None:
        while not stop.wait(0.001):
            samples.append(time.perf_counter_ns())

    worker = threading.Thread(target=sample_clock)
    worker.start()
    started = time.perf_counter_ns()
    prepare_integral_homology(chain_complex, [0, 1, 2])
    finished = time.perf_counter_ns()
    stop.set()
    worker.join()
    assert any(started < sample < finished for sample in samples)
