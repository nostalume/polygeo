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
    Solve compatible pointwise-density and integrated-load Poisson equations on a closed surface.

    ## From mathematics to PolyGeo
    A density `f` means `K u = M f`; its integrated load `b=M f` means `K u = b`.

    ## Computation
    Prepare once from either problem, then reuse its factor and workspace across both forms.

    ## Visualization
    Inspect the copied potential coefficients.

    ## Evaluation
    Reapply `K=M\Delta` and independently check both right-hand sides and the weighted gauge.

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
    density_values = np.zeros(space.size, dtype=np.float64)
    density_values[:2] = (weights[1], -weights[0])
    density = space.admit_numpy(density_values)
    load = metric.riesz(0).apply(density)
    density_problem = metric.mean_zero_poisson_density(density)
    load_problem = metric.mean_zero_poisson_load(load)
    prepared = density_problem.prepare()
    workspace = prepared.workspace_for(load_problem)
    density_solution = prepared.solve(density_problem, workspace)
    load_solution = prepared.solve(load_problem, workspace)
    potential = load_solution.potential.coefficients_numpy_copy()
    reapplied = (
        metric.riesz(0)
        .apply(metric.laplacian(0).apply(load_solution.potential))
        .coefficients_numpy_copy()
    )
    poisson_evidence = {
        "load_residual": float(
            np.max(np.abs(reapplied - load.coefficients_numpy_copy()))
        ),
        "density_load_agreement": float(
            np.max(
                np.abs(density_solution.potential.coefficients_numpy_copy() - potential)
            )
        ),
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
