"""One owner graph spans exact chain algebra and homology."""

from __future__ import annotations

import gc

import numpy as np

import polygeo


def test_exact_homology_graph_retains_owners_and_isolates_projections() -> None:
    vertices = np.arange(4, dtype=np.int64)
    domain = polygeo.Complex.from_maximal_simplices(
        np.column_stack((vertices, np.roll(vertices, -1)))
    )
    exact = domain.chain_complex()
    boundary = exact.boundary(1)
    source = boundary.source
    target = boundary.target
    estimate = polygeo.CsrRepresentation.estimate(boundary, polygeo.BigIntEncoding)
    representation = polygeo.CsrRepresentation.build(
        boundary, polygeo.BigIntEncoding, estimate.as_limit()
    )
    analysis = polygeo.analyze_integral_homology(exact, [1])
    first_group = analysis[1]
    second_group = analysis[1]
    cycle = first_group.free_cycle(0)
    projected = representation.to_scipy_int64_copy()

    assert source.dimension == boundary.source.dimension == 4
    assert target.dimension == boundary.target.dimension == 4
    assert representation.represented_map.source.dimension == source.dimension
    assert first_group.free_rank == second_group.free_rank == 1
    assert first_group.free_cycle(0).to_python_copy() == cycle.to_python_copy()
    assert boundary.apply(cycle).to_python_copy() == ((), ())

    del domain, exact, boundary, source, target, analysis, first_group, second_group
    gc.collect()

    projected.data[:] = 0
    assert representation.apply(cycle).to_python_copy() == ((), ())
    assert representation.to_python_copy().coefficients != (0,) * projected.nnz
