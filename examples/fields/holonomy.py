import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from polygeo.geometry import TriangleSurface
    from polygeo.plot import form as plot_form
    from examples.support.meshes import torus

    return TriangleSurface, mo, plot_form, torus


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
    Plot one exact dual generator and report local and generator errors.

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
    cycles = domain.dual_cycles()
    evidence = surface.levi_civita().holonomy(cycles)
    generator = domain.binary64_cochain_space(1).realize_integral(cycles.cocycle(0))
    holonomy_evidence = {
        "cycle_rank": cycles.rank,
        "local_error": evidence.local_error,
        "generator_error": evidence.generator_error,
        "limit": evidence.limit,
        "local_is_flat": evidence.local_error <= evidence.limit,
        "global_is_obstructed": evidence.generator_error > evidence.limit,
    }
    return generator, geometry, holonomy_evidence


@app.cell
def _(generator, geometry, holonomy_evidence, mo, plot_form):
    mo.vstack(
        [
            mo.md(f"`{holonomy_evidence}`"),
            plot_form(geometry, generator, title="First exact dual generator"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
