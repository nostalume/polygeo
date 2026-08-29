import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from support.meshes import annulus

    return annulus, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Hodge decomposition

    ## Mathematical question
    How does a discrete one-form split into exact, coexact, and harmonic parts?

    ## From mathematics to PolyGeo
    One native element carrier represents all three components over distinct spaces.

    ## Computation
    Construct the metric-owned decomposition problem and solve it.

    ## Visualization
    Compare copied coefficient magnitudes.

    ## Evaluation
    Reconstruct the source independently and report component sizes and orthogonality.

    ## Interpretation
    Mathematical roles differ without multiplying data structures.
    """)
    return


@app.cell
def _(annulus, np):
    domain, geometry = annulus(4, 16)
    space = domain.binary64_cochain_space(1)
    source = space.admit_numpy(np.sin(np.arange(space.size, dtype=np.float64)))
    problem = geometry.positive_metric().hodge_decomposition(source)
    prepared = problem.prepare()
    workspace = prepared.workspace_for(problem)
    result = prepared.solve(problem, workspace)
    components = tuple(
        value.coefficients_numpy_copy()
        for value in (result.exact, result.coexact, result.harmonic)
    )
    reconstructed = sum(components, start=np.zeros(space.size))
    hodge_evidence = {
        "reconstruction": float(
            np.max(np.abs(reconstructed - source.coefficients_numpy_copy()))
        ),
        "component_maxima": tuple(float(np.max(np.abs(value))) for value in components),
        "orthogonality": result.orthogonality_bound,
    }
    return hodge_evidence


@app.cell
def _(hodge_evidence, mo):
    mo.md(f"`{hodge_evidence}`")
    return


if __name__ == "__main__":
    app.run()
