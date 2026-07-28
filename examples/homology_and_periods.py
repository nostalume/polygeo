import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from polygeo import ORDINARY_FORM, plot_homology_cycle, real_homology_basis
    from support.meshes import torus

    return ORDINARY_FORM, mo, np, plot_homology_cycle, real_homology_basis, torus


@app.cell
def _(mo):
    mo.Html("""
    <style>
    .marimo { max-width: 980px; margin: 0 auto; }
    h1, h2 { letter-spacing: -0.02em; }
    table { display: block; overflow-x: auto; }
    .plotly-graph-div { max-width: 100%; }
    pre, code { overflow-x: auto; }
    </style>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Torus homology and coordinate periods

    ## Mathematical question

    For a triangulated torus, compute
    \[
    H_1(K;\mathbb R)=\ker\partial_1/\operatorname{im}\partial_2
    \]
    and pair its cycle representatives with the wrapped coordinate one-forms
    \(du\) and \(dv\). The invariant question is whether this pairing has full
    rank. A returned real basis may differ from coordinate generators by any
    invertible change of basis, so no signed-permutation pattern is assumed.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From mathematics to PolyGeo

    | Mathematics | Meaning | PolyGeo code |
    |---|---|---|
    | \(K\) | bounded analytic torus mesh | `torus(m, n)` |
    | \([z_1],[z_2]\) | real degree-one homology | `real_homology_basis(K, 1)` |
    | \(du,dv\) | wrapped coordinate increments on canonical edges | `C1.form(...)` |
    | \(\langle du,z_j\rangle\) | coordinate periods | `basis.periods(du)` |
    | \(z_j\) | selected sparse primal chain | `plot_homology_cycle(...)` |

    Homology is chain-owned. The forms pair with the returned cycles but do not
    manufacture or own them.
    """)
    return


@app.cell
def _(mo):
    major_sections = mo.ui.slider(3, 5, value=4, step=1, label="major sections (max 5)")
    minor_sections = mo.ui.slider(3, 6, value=6, step=1, label="minor sections (max 6)")
    mo.vstack([major_sections, minor_sections])
    return major_sections, minor_sections


@app.cell
def _(mo):
    cycle_selector = mo.ui.slider(0, 1, value=0, step=1, label="displayed cycle")
    cycle_selector
    return (cycle_selector,)


@app.cell
def _(
    ORDINARY_FORM,
    major_sections,
    minor_sections,
    np,
    real_homology_basis,
    torus,
):
    torus_complex, torus_geometry, vertex_v = torus(
        major_sections.value, minor_sections.value
    )
    vertex_u = np.repeat(
        2.0
        * np.pi
        * np.arange(major_sections.value, dtype=np.float64)
        / major_sections.value,
        minor_sections.value,
    )
    canonical_edges = torus_complex.simplices(1)

    def wrapped_edge_differences(coordinates):
        raw = coordinates[canonical_edges[:, 1]] - coordinates[canonical_edges[:, 0]]
        return np.arctan2(np.sin(raw), np.cos(raw))

    degree_one_space = torus_complex.cochain_space(1)
    du = degree_one_space.form(wrapped_edge_differences(vertex_u), ORDINARY_FORM)
    dv = degree_one_space.form(wrapped_edge_differences(vertex_v), ORDINARY_FORM)
    homology_basis = real_homology_basis(torus_complex, 1)
    cycle_matrix = homology_basis.cycle_coefficients()
    boundary_residual = torus_complex.boundary_matrix(1) @ cycle_matrix.toarray()
    exact_closure = bool(
        np.array_equal(boundary_residual, np.zeros_like(boundary_residual))
    )
    period_matrix = np.asarray(
        [
            homology_basis.periods(du).values,
            homology_basis.periods(dv).values,
        ],
        dtype=np.float64,
    )
    period_rank = int(np.linalg.matrix_rank(period_matrix))
    period_determinant = float(np.linalg.det(period_matrix))
    period_minimum_singular_value = float(
        np.linalg.svd(period_matrix, compute_uv=False)[-1]
    )
    return (
        exact_closure,
        homology_basis,
        period_determinant,
        period_matrix,
        period_minimum_singular_value,
        period_rank,
        torus_geometry,
    )


@app.cell
def _(major_sections, minor_sections, mo):
    mo.md(rf"""
    ## Computation

    The bounded mesh is **{major_sections.value}×{minor_sections.value}** (default
    4×6, maximum 5×6). Wrapped endpoint differences avoid the coordinate seam
    introducing a spurious jump.

    ## Visualization

    The selector chooses one of the two returned sparse cycle columns. Its signed
    canonical edges are highlighted in the single figure below.
    """)
    return


@app.cell
def _(
    cycle_selector,
    homology_basis,
    plot_homology_cycle,
    torus_geometry,
):
    selected_cycle_figure = plot_homology_cycle(
        torus_geometry,
        homology_basis,
        cycle_selector.value,
        title=f"Real primal cycle {cycle_selector.value}",
    )
    return (selected_cycle_figure,)


@app.cell
def _(selected_cycle_figure):
    selected_cycle_figure
    return


@app.cell
def _(
    exact_closure,
    homology_basis,
    mo,
    np,
    period_determinant,
    period_matrix,
    period_minimum_singular_value,
    period_rank,
):
    period_text = np.array2string(period_matrix, precision=6, suppress_small=True)
    mo.md(rf"""
    ## Evaluation

    | Independent law | Observed | Expected |
    |---|---:|---:|
    | `dim H1` | `{homology_basis.dimension}` | `2` |
    | exact cycle closure | `{exact_closure}` | `True` |
    | period matrix rank | `{period_rank}` | `2` |
    | smallest period singular value | `{period_minimum_singular_value:.6g}` | positive |
    | period determinant | `{period_determinant:.6g}` | nonzero |

    The rows of the 2×2 matrix are \(du,dv\), and its columns are the returned
    basis cycles:

    ```
    {period_text}
    ```

    Full rank is basis-independent. Individual entries are reported as observed
    fixture data only: a real homology basis may undergo any invertible change of
    basis, not merely a permutation or sign change.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interpretation

    The two independent periods recover the torus's two real degree-one
    directions. This is a basis of real primal cycles, not integral dual data:
    the study makes no claim that these representatives form an integral basis,
    are shortest cycles, or authorize dual holonomy computations.
    """)
    return


if __name__ == "__main__":
    app.run()
