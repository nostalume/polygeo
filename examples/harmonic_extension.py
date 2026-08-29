import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from polygeo import topological_boundary
    from support.meshes import annulus

    return annulus, mo, np, topological_boundary


@app.cell
def _(mo):
    mo.md(r"""
    # Harmonic extension from an annulus boundary

    ## Mathematical question
    Which interior values minimize Dirichlet energy for prescribed boundary samples?

    ## From mathematics to PolyGeo
    The annulus boundary is a selected native `Binary64Space`, not a second space class.

    ## Computation
    Construct and solve the metric-owned harmonic-extension problem.

    ## Visualization
    Summarize the resulting scalar range.

    ## Evaluation
    Compare the solved boundary coefficients with the prescribed samples.

    ## Interpretation
    Selection changes basis semantics while retaining one coefficient carrier.
    """)
    return


@app.cell
def _(annulus, np, topological_boundary):
    domain, geometry = annulus(4, 16)
    boundary_indices = np.flatnonzero(topological_boundary(domain).mask(0)).astype(
        np.int64
    )
    boundary_space = domain.binary64_cochain_space(0, indices=boundary_indices)
    positions = geometry.positions_numpy_copy()
    prescribed = positions[boundary_indices, 0].astype(np.float64)
    boundary_values = boundary_space.admit_numpy(prescribed)
    metric = geometry.positive_metric()
    problem = metric.harmonic_extension(boundary_values)
    prepared = problem.prepare()
    workspace = prepared.workspace_for(problem)
    solution = prepared.solve(problem, workspace)
    values = solution.value.coefficients_numpy_copy()
    harmonic_evidence = {
        "residual": solution.residual_bound,
        "boundary_error": float(np.max(np.abs(values[boundary_indices] - prescribed))),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }
    return harmonic_evidence


@app.cell
def _(harmonic_evidence, mo):
    mo.md(f"`{harmonic_evidence}`")
    return


if __name__ == "__main__":
    app.run()
