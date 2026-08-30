import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from polygeo import analyze_integral_homology
    from support.meshes import torus

    return analyze_integral_homology, mo, torus


@app.cell
def _(mo):
    mo.md(r"""
    # Exact topology on a torus

    ## Mathematical question
    How do independent cycles, exact Stokes pairing, and the cohomological
    intersection product coexist on one triangulated torus?

    ## From mathematics to PolyGeo
    The integral chain complex owns signed boundaries, its dual owns coboundaries
    and cup products, and exact homology supplies sparse primal cycle
    representatives.

    ## Computation
    Compute the degree-one group, evaluate both routes of
    `<delta alpha, c> = <alpha, boundary c>`, and pair the two torus cocycles in both
    cup orders with a fundamental two-cycle.

    ## Visualization
    Display rank, torsion, representative support, Stokes values, and the cup
    intersection pair.

    ## Evaluation
    Verify the rank-two free group, a nonzero Stokes equality, and an antisymmetric
    unimodular intersection pair.

    ## Interpretation
    Homology, duality, cup product, and evaluation remain direct operations on the
    same exact topology owner. The two cochain products need not be equal, while
    their evaluations on the fundamental class exhibit graded antisymmetry.
    """)
    return


@app.cell
def _(analyze_integral_homology, torus):
    domain, _ = torus(12, 8)
    chain_complex = domain.chain_complex()
    cochain_complex = chain_complex.dual()
    face = chain_complex[2].element({0: 1})
    edge_cochain = cochain_complex[1].element({0: 1})
    face_boundary = chain_complex.boundary(2).apply(face)
    edge_coboundary = cochain_complex.coboundary(1).apply(edge_cochain)
    stokes_left = edge_coboundary.evaluate(face)
    stokes_right = edge_cochain.evaluate(face_boundary)
    homology = analyze_integral_homology(chain_complex, [1, 2])
    group = homology[1]
    fundamental_cycle = homology[2].free_cycle(0)
    dual_cycles = domain.integral_dual_cycle_basis()
    first_cocycle = dual_cycles.cocycle(0)
    second_cocycle = dual_cycles.cocycle(1)
    cup_forward = first_cocycle.cup(second_cocycle).evaluate(fundamental_cycle)
    cup_reverse = second_cocycle.cup(first_cocycle).evaluate(fundamental_cycle)
    homology_evidence = {
        "free_rank": group.free_rank,
        "torsion_orders": group.torsion_orders,
        "first_cycle_terms": len(group.free_cycle(0).to_python_copy()[0]),
        "is_rank_two_free": group.free_rank == 2 and not group.torsion_orders,
        "stokes_pair": (stokes_left, stokes_right),
        "stokes_is_exact_nonzero": stokes_left == stokes_right != 0,
        "cup_intersection_pair": (cup_forward, cup_reverse),
        "cup_intersection_is_antisymmetric_unimodular": (
            cup_forward == -cup_reverse and abs(cup_forward) == 1
        ),
    }
    return homology_evidence


@app.cell
def _(homology_evidence, mo):
    mo.md(f"`{homology_evidence}`")
    return


if __name__ == "__main__":
    app.run()
