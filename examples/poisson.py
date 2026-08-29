import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from support.meshes import icosphere

    return icosphere, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Mean-zero Poisson problem

    ## Mathematical question
    Solve a compatible scalar Poisson equation on a closed surface.

    ## From mathematics to PolyGeo
    The metric constructs the problem; `K=M\Delta` explains the represented operator.

    ## Computation
    Prepare once, allocate a workspace for the problem, and solve.

    ## Visualization
    Inspect the copied potential coefficients.

    ## Evaluation
    Reapply `M\Delta` and independently check the residual and weighted gauge.

    ## Interpretation
    Problem, preparation, workspace, and result are distinct semantics over native owners.
    """)
    return


@app.cell
def _(icosphere, np):
    domain, geometry = icosphere(1, 1.0)
    space = domain.binary64_cochain_space(0)
    metric = geometry.positive_metric()
    weights = metric.hodge_coefficients_numpy_copy(0)
    source_values = np.zeros(space.size, dtype=np.float64)
    source_values[:2] = (weights[1], -weights[0])
    source = space.admit_numpy(source_values)
    problem = metric.mean_zero_poisson(source)
    prepared = problem.prepare()
    workspace = prepared.workspace_for(problem)
    solution = prepared.solve(problem, workspace)
    potential = solution.potential.coefficients_numpy_copy()
    reapplied = (
        metric.riesz(0)
        .apply(metric.laplacian(0).apply(solution.potential))
        .coefficients_numpy_copy()
    )
    poisson_evidence = {
        "physical_residual": float(np.max(np.abs(reapplied - source_values))),
        "weighted_gauge": float(abs(weights @ potential)),
        "potential_range": float(np.ptp(potential)),
    }
    return poisson_evidence


@app.cell
def _(mo, poisson_evidence):
    mo.md(f"`{poisson_evidence}`")
    return


if __name__ == "__main__":
    app.run()
