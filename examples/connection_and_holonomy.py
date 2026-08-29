import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from polygeo import TriangleSurface
    from support.meshes import torus

    return TriangleSurface, mo, torus


@app.cell
def _(mo):
    mo.md(r"""
    # Connection and holonomy

    ## Mathematical question
    How do local transport defects differ from global topological obstruction?

    ## From mathematics to PolyGeo
    A surface owns the connection; an exact dual-cycle basis supplies global witnesses.

    ## Computation
    Evaluate Levi-Civita holonomy on a torus.

    ## Visualization
    Report local and generator errors.

    ## Evaluation
    Contrast local contractible holonomy with global generator obstruction.

    ## Interpretation
    Integrability needs both geometric transport and exact topology evidence.
    """)
    return


@app.cell
def _(TriangleSurface, torus):
    domain, geometry = torus(12, 8)
    surface = TriangleSurface.admit(geometry)
    cycles = domain.integral_dual_cycle_basis()
    evidence = surface.levi_civita_connection().holonomy(cycles)
    holonomy_evidence = {
        "cycle_rank": cycles.rank,
        "local_error": evidence.local_error,
        "generator_error": evidence.generator_error,
        "limit": evidence.limit,
        "local_is_flat": evidence.local_error <= evidence.limit,
        "global_is_obstructed": evidence.generator_error > evidence.limit,
    }
    return holonomy_evidence


@app.cell
def _(holonomy_evidence, mo):
    mo.md(f"`{holonomy_evidence}`")
    return


if __name__ == "__main__":
    app.run()
