import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from polygeo import Geometry, TriangleSurface
    from support.meshes import disk

    return Geometry, TriangleSurface, disk, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Least-squares conformal map of a triangle disk

    ## Mathematical question
    How can two boundary anchors remove planar similarity freedom while the other
    boundary vertices remain free?

    ## From mathematics to PolyGeo
    Each consistently oriented source face contributes one complex
    Cauchy--Riemann equation. The two chosen boundary vertices map to `(0, 0)` and
    `(1, 0)`; the surface owner assembles and solves the remaining rectangular
    least-squares system directly.

    ## Computation
    Lift a convex disk into three dimensions, choose opposite identities from its
    exact ordered boundary, and compute one rank-revealing LSCM realization.

    ## Evaluation
    Compare native rank, normalized residual, and robust face-orientation evidence
    with an independent array-level signed-area calculation.

    ## Interpretation
    Positive target faces establish a locally orientation-preserving piecewise
    linear parameterization. The result does not claim global injectivity, smooth
    conformality, or absence of overlap.
    """)
    return


@app.cell
def _(Geometry, TriangleSurface, disk, np):
    domain, planar = disk(3, 16)
    planar_positions = planar.positions_numpy_copy()
    height = 0.35 * (1.0 - np.sum(planar_positions * planar_positions, axis=1))
    source = Geometry.from_positions(
        domain, np.column_stack((planar_positions, height))
    )
    boundary = domain.disk_boundary_vertices_numpy_copy()
    boundary_positions = source.positions_numpy_copy()[boundary]
    separations = np.sum(
        (boundary_positions[:, None, :] - boundary_positions[None, :, :]) ** 2,
        axis=2,
    )
    separations[np.tril_indices_from(separations)] = -np.inf
    first, second = np.unravel_index(np.argmax(separations), separations.shape)
    anchors = (int(boundary[first]), int(boundary[second]))
    solution = TriangleSurface.admit(source).least_squares_conformal_map(anchors)
    mapped = solution.realization.positions_numpy_copy()

    faces = domain.simplices(2)
    first = mapped[faces[:, 1]] - mapped[faces[:, 0]]
    second = mapped[faces[:, 2]] - mapped[faces[:, 0]]
    signed_twice_area = domain.orientations(2) * (
        first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    )
    if not np.all(signed_twice_area > 0.0):
        raise RuntimeError("LSCM target contains a nonpositive oriented face")
    np.testing.assert_array_equal(mapped[list(anchors)], [[0.0, 0.0], [1.0, 0.0]])
    conformal_map_evidence = {
        "required_rank": solution.required_rank,
        "observed_rank": solution.observed_rank,
        "condition_indicator": solution.condition_indicator,
        "normalized_conformality_residual": solution.residual_bound,
        "minimum_native_normalized_signed_twice_area": (
            solution.minimum_normalized_signed_twice_area
        ),
        "minimum_independent_signed_twice_area": float(np.min(signed_twice_area)),
        "exact_fallback_faces": solution.exact_fallback_faces,
    }
    return conformal_map_evidence


@app.cell
def _(conformal_map_evidence, mo):
    mo.md(f"`{conformal_map_evidence}`")
    return


if __name__ == "__main__":
    app.run()
