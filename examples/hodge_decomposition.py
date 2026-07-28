import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from polygeo import (
        ORDINARY_FORM,
        Complex,
        Geometry,
        PositiveHodgeMetric,
        hodge_decomposition,
        plot_cochain,
        prepare_least_squares,
        real_homology_basis,
    )
    from support.meshes import annulus

    return (
        Complex,
        Geometry,
        ORDINARY_FORM,
        PositiveHodgeMetric,
        annulus,
        hodge_decomposition,
        mo,
        np,
        plot_cochain,
        prepare_least_squares,
        real_homology_basis,
    )


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
    # Hodge decomposition on an annulus

    ## Mathematical question

    Decompose a deterministic ordinary one-form on an annulus as
    \[
    \alpha=d\phi+\delta\psi+h,\qquad dh=0,\qquad\delta h=0.
    \]
    A positive represented Hodge metric defines the weighted inner product. The
    annulus supplies a one-dimensional harmonic sector, allowing all three
    components to be visible in one reproducible example.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From mathematics to PolyGeo

    | Mathematics | Meaning | PolyGeo code |
    |---|---|---|
    | \(K\) | bounded annulus triangulation | `annulus(rings, sections)` |
    | \(H_k>0\) | represented positive metric | `PositiveHodgeMetric(geometry)` |
    | \(\alpha\in C^1(K)\) | deterministic ordinary form | `C1.form(values, ...)` |
    | \(d\phi,\delta\psi,h\) | certified components | `hodge_decomposition(...)` |
    | \(\langle x,y\rangle_{H_1}\) | weighted pairing | `dot(H1 * x, y)` |

    Degree-one coefficients are displayed at canonical edge midpoints; they are
    cochain values, not ambient tangent vectors.
    """)
    return


@app.cell
def _(mo):
    mesh_preset = mo.ui.dropdown(
        options=(
            "3 x 12",
            "3 x 13",
            "3 x 14",
            "3 x 15",
            "3 x 16",
            "4 x 14",
            "4 x 15",
            "4 x 16",
        ),
        value="3 x 12",
        label="admitted annulus preset (rings x sections)",
    )
    form_amplitude = mo.ui.slider(
        0.5, 1.5, value=1.0, step=0.25, label="form amplitude"
    )
    mo.vstack([mesh_preset, form_amplitude])
    return form_amplitude, mesh_preset


@app.cell
def _(
    ORDINARY_FORM,
    PositiveHodgeMetric,
    annulus,
    form_amplitude,
    hodge_decomposition,
    mesh_preset,
    np,
    prepare_least_squares,
    real_homology_basis,
):
    radial_rings, angular_sections = (
        int(value) for value in mesh_preset.value.split(" x ")
    )
    annulus_complex, annulus_geometry = annulus(radial_rings, angular_sections)
    hodge_metric = PositiveHodgeMetric(annulus_geometry)
    edge_space = annulus_complex.cochain_space(1)
    edge_index = np.arange(edge_space.size, dtype=np.float64)
    original_form = edge_space.form(
        form_amplitude.value
        * (np.sin(0.37 * edge_index) + 0.4 * np.cos(0.11 * edge_index)),
        ORDINARY_FORM,
    )
    decomposition = hodge_decomposition(
        hodge_metric, original_form, prepare_least_squares
    )
    hodge_components = decomposition.output
    original_values = original_form.coefficients()
    exact_values = hodge_components.exact.coefficients()
    coexact_values = hodge_components.coexact.coefficients()
    harmonic_values = hodge_components.harmonic.coefficients()
    hodge_weights = hodge_metric.weights(1)
    reconstruction_error = float(
        np.max(
            np.abs(original_values - exact_values - coexact_values - harmonic_values),
            initial=0.0,
        )
    )

    def weighted_pair(left, right):
        return float(np.dot(hodge_weights * left, right))

    pairwise_products = np.asarray(
        [
            weighted_pair(exact_values, coexact_values),
            weighted_pair(exact_values, harmonic_values),
            weighted_pair(coexact_values, harmonic_values),
        ],
        dtype=np.float64,
    )
    pairwise_orthogonality_error = float(np.max(np.abs(pairwise_products), initial=0.0))
    component_norms = np.asarray(
        [
            np.linalg.norm(exact_values),
            np.linalg.norm(coexact_values),
            np.linalg.norm(harmonic_values),
        ],
        dtype=np.float64,
    )
    all_components_nonzero = bool(np.all(component_norms > 0.0))
    harmonic_closure_error = float(
        np.max(
            np.abs(annulus_complex.boundary_matrix(2).T @ harmonic_values),
            initial=0.0,
        )
    )
    harmonic_coclosure_error = float(
        np.max(
            np.abs(
                annulus_complex.boundary_matrix(1) @ (hodge_weights * harmonic_values)
            ),
            initial=0.0,
        )
    )
    annulus_homology = real_homology_basis(annulus_complex, 1)
    harmonic_period = annulus_homology.periods(hodge_components.harmonic).values[0]
    return (
        all_components_nonzero,
        annulus_geometry,
        component_norms,
        decomposition,
        harmonic_closure_error,
        harmonic_coclosure_error,
        harmonic_period,
        hodge_components,
        original_form,
        pairwise_orthogonality_error,
        reconstruction_error,
    )


@app.cell
def _(
    Complex,
    Geometry,
    ORDINARY_FORM,
    PositiveHodgeMetric,
    hodge_decomposition,
    np,
):
    endpoint_complex = Complex.from_maximal_simplices(np.array([[0]], dtype=np.int64))
    endpoint_geometry = Geometry.from_positions(
        endpoint_complex, np.empty((1, 0), dtype=np.float64)
    )
    endpoint_form = endpoint_complex.cochain_space(0).form(
        np.array([3.0], dtype=np.float64), ORDINARY_FORM
    )
    backend_calls = [0]

    def forbidden_backend(operator):
        backend_calls[0] += 1
        raise AssertionError(operator)

    endpoint_components = hodge_decomposition(
        PositiveHodgeMetric(endpoint_geometry), endpoint_form, forbidden_backend
    ).output
    endpoint_exact_coexact_error = float(
        np.max(
            np.abs(endpoint_components.exact.coefficients())
            + np.abs(endpoint_components.coexact.coefficients()),
            initial=0.0,
        )
    )
    endpoint_harmonic_error = float(
        np.max(
            np.abs(
                endpoint_components.harmonic.coefficients()
                - endpoint_form.coefficients()
            ),
            initial=0.0,
        )
    )
    endpoint_backend_calls = backend_calls[0]
    return (
        endpoint_backend_calls,
        endpoint_exact_coexact_error,
        endpoint_harmonic_error,
    )


@app.cell
def _(annulus_geometry, hodge_components, original_form, plot_cochain):
    original_figure = plot_cochain(
        annulus_geometry, original_form, title="Original one-form"
    )
    exact_figure = plot_cochain(
        annulus_geometry, hodge_components.exact, title="Exact component"
    )
    coexact_figure = plot_cochain(
        annulus_geometry, hodge_components.coexact, title="Coexact component"
    )
    harmonic_figure = plot_cochain(
        annulus_geometry, hodge_components.harmonic, title="Harmonic component"
    )
    return coexact_figure, exact_figure, harmonic_figure, original_figure


@app.cell
def _(form_amplitude, mesh_preset, mo):
    _radial_rings, _angular_sections = mesh_preset.value.split(" x ")
    mo.md(f"""
    ## Computation

    The bounded annulus uses **{_radial_rings} radial rings** and
    **{_angular_sections} angular sections**. A fixed trigonometric edge
    signal, scaled by **{form_amplitude.value:.2f}**, is chosen so all three
    components are nonzero.

    ## Visualization

    Original, exact, coexact, and harmonic one-forms are shown vertically in that
    order.
    """)
    return


@app.cell
def _(original_figure):
    original_figure
    return


@app.cell
def _(exact_figure):
    exact_figure
    return


@app.cell
def _(coexact_figure):
    coexact_figure
    return


@app.cell
def _(harmonic_figure):
    harmonic_figure
    return


@app.cell
def _(
    all_components_nonzero,
    component_norms,
    decomposition,
    endpoint_backend_calls,
    endpoint_exact_coexact_error,
    endpoint_harmonic_error,
    harmonic_closure_error,
    harmonic_coclosure_error,
    harmonic_period,
    mo,
    np,
    pairwise_orthogonality_error,
    reconstruction_error,
):
    norm_text = np.array2string(component_norms, precision=6)
    evidence = decomposition.evidence
    mo.md(f"""
    ## Evaluation

    | Independent law | Observed | Expected / admitted |
    |---|---:|---:|
    | all three components are nonzero | `{all_components_nonzero}` | `True` |
    | component norms `[exact, coexact, harmonic]` | `{norm_text}` | each `> 0` |
    | reconstruction max residual | `{reconstruction_error:.3e}` | near `0` |
    | pairwise weighted orthogonality | `{pairwise_orthogonality_error:.3e}` | near `0` |
    | harmonic closure max residual | `{harmonic_closure_error:.3e}` | near `0` |
    | harmonic coclosure max residual | `{harmonic_coclosure_error:.3e}` | near `0` |
    | harmonic period on the returned real cycle | `{harmonic_period:.6g}` | nonzero |
    | no-backend endpoint law: backend calls | `{endpoint_backend_calls}` | `0` |
    | endpoint exact/coexact max | `{endpoint_exact_coexact_error:.3e}` | `0` |
    | endpoint harmonic error | `{endpoint_harmonic_error:.3e}` | `0` |

    PolyGeo's admitted reconstruction limit is
    `{evidence.reconstruction.limit:.3e}` and orthogonality limit is
    `{evidence.orthogonality.limit:.3e}`. The table independently recomputes all
    three weighted component pairings, not only one pair.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interpretation

    Exact content is gradient-like, coexact content is codifferential, and the
    nonzero harmonic period records the annulus hole. These components depend on
    the represented positive metric. The zero-dimensional endpoint has neither
    adjacent image, so the original form is harmonic and no solver backend may be
    called. Residual and condition evidence remain distinct from forward-error
    guarantees.
    """)
    return


if __name__ == "__main__":
    app.run()
