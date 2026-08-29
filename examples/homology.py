import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from polygeo import prepare_integral_homology
    from support.meshes import torus

    return mo, prepare_integral_homology, torus


@app.cell
def _(mo):
    mo.md(r"""
    # Torus homology

    ## Mathematical question
    What are the independent one-cycles of a triangulated torus?

    ## From mathematics to PolyGeo
    Exact integral homology supplies sparse primal cycle representatives.

    ## Computation
    Compute the degree-one group from the exact chain complex.

    ## Visualization
    Display rank, torsion, and representative support.

    ## Evaluation
    Verify the expected rank-two free group.

    ## Interpretation
    Requested-degree analysis retains its exact topology owner.
    """)
    return


@app.cell
def _(prepare_integral_homology, torus):
    domain, _ = torus(12, 8)
    group = prepare_integral_homology(domain.chain_complex(), [1])[1]
    homology_evidence = {
        "free_rank": group.free_rank,
        "torsion_orders": group.torsion_orders,
        "first_cycle_terms": len(group.free_cycle(0).to_python_copy()[0]),
        "is_rank_two_free": group.free_rank == 2 and not group.torsion_orders,
    }
    return homology_evidence


@app.cell
def _(homology_evidence, mo):
    mo.md(f"`{homology_evidence}`")
    return


if __name__ == "__main__":
    app.run()
