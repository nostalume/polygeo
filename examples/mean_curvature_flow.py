import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from polygeo import plot_geometry
    from support.meshes import icosphere

    return icosphere, mo, np, plot_geometry


@app.cell
def _(mo):
    mo.md(r"""
    # One frozen mean-curvature-flow step

    ## Mathematical question
    How does one implicit frozen step change a triangulated sphere?

    ## From mathematics to PolyGeo
    The positive metric constructs a flow problem retaining the source realization.

    ## Computation
    Prepare, allocate, and solve without mutating the source.

    ## Visualization
    Compare source and target geometry snapshots.

    ## Evaluation
    Check source immutability, a fresh target owner, and retained topology identity.

    ## Interpretation
    Atomic publication makes source immutability observable.
    """)
    return


@app.cell
def _(icosphere, np):
    _, source_geometry = icosphere(1, 1.0)
    retained_source = source_geometry.positions_numpy_copy()
    problem = source_geometry.positive_metric().frozen_mean_curvature_flow(0.01)
    prepared = problem.prepare()
    workspace = prepared.workspace_for(problem)
    step = prepared.solve(problem, workspace)
    flow_evidence = {
        "source_unchanged": bool(
            np.array_equal(retained_source, source_geometry.positions_numpy_copy())
        ),
        "target_is_new": step.target is not source_geometry,
        "topology_is_retained": step.target.complex is source_geometry.complex,
        "energy_before": step.energy_before,
        "energy_after": step.energy_after,
        "residual": step.residual_bound,
    }
    return flow_evidence, source_geometry, step


@app.cell
def _(flow_evidence, mo, plot_geometry, source_geometry, step):
    mo.vstack(
        [
            mo.md(f"`{flow_evidence}`"),
            plot_geometry(source_geometry),
            plot_geometry(step.target),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
