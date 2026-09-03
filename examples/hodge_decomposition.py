import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from polygeo.plot import form as plot_form
    from support.meshes import annulus

    return annulus, mo, np, plot_form


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
    Compare the source with its exact, coexact, and harmonic components.

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
    problem = geometry.metric().hodge_decomposition(source)
    prepared = problem.prepare()
    workspace = prepared.workspace_for(problem)
    result = prepared.solve(problem, workspace)
    exact_form = result.exact
    coexact_form = result.coexact
    harmonic_form = result.harmonic
    components = tuple(
        value.coefficients_numpy_copy()
        for value in (exact_form, coexact_form, harmonic_form)
    )
    reconstructed = sum(components, start=np.zeros(space.size))
    hodge_evidence = {
        "reconstruction": float(
            np.max(np.abs(reconstructed - source.coefficients_numpy_copy()))
        ),
        "component_maxima": tuple(float(np.max(np.abs(value))) for value in components),
        "orthogonality": result.orthogonality_bound,
    }
    return coexact_form, exact_form, geometry, harmonic_form, hodge_evidence, source


@app.cell
def _(
    coexact_form,
    exact_form,
    geometry,
    harmonic_form,
    hodge_evidence,
    mo,
    plot_form,
    source,
):
    mo.vstack(
        [
            mo.md(f"`{hodge_evidence}`"),
            mo.hstack(
                [
                    plot_form(geometry, source, title="Source one-form"),
                    plot_form(geometry, exact_form, title="Exact component"),
                ]
            ),
            mo.hstack(
                [
                    plot_form(geometry, coexact_form, title="Coexact component"),
                    plot_form(geometry, harmonic_form, title="Harmonic component"),
                ]
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
