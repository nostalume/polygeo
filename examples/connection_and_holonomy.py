import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from polygeo import (
        FaceVectors,
        SurfaceError,
        admit_integrable_connection,
        connection_holonomy,
        integral_dual_cycles,
        integrate_direction_field,
        plot_geometry,
        plot_surface_vectors,
        triangle_frames,
    )
    from support.connections import torus_connection
    from support.meshes import torus

    return (
        FaceVectors,
        SurfaceError,
        admit_integrable_connection,
        connection_holonomy,
        integral_dual_cycles,
        integrate_direction_field,
        mo,
        np,
        plot_geometry,
        plot_surface_vectors,
        torus,
        torus_connection,
        triangle_frames,
    )


@app.cell
def _(mo):
    mo.Html("""
    <style>
    .marimo { max-width: 980px; margin: 0 auto; }
    h1, h2 { letter-spacing: -0.02em; }
    table { display: block; width: 100%; max-width: 100%; overflow-x: auto; }
    .plotly-graph-div { width: 100%; max-width: 100%; }
    pre, code, mjx-container[display="true"] { overflow-x: auto; }
    </style>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Connection and holonomy on a torus

    ## Mathematical question

    A represented \(SO(2)\) connection assigns a unit-complex transport to each
    canonical dual edge. When can those transports be integrated into one tangent
    direction per face?

    Integrability has two independent parts. Products around vertex dual cells test
    **local contractible holonomy**. Products around an integral dual basis test the
    **global generator obstruction**. Passing only the local test is not enough on a
    torus.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From mathematics to PolyGeo

    | Mathematics | Meaning | PolyGeo code |
    |---|---|---|
    | \(L_e\in SO(2)\) | Levi–Civita hinge transport | `torus_connection(..., "levi-civita")` |
    | \(\delta_e\) | represented transport deviation | `surface_connection` inside the shared fixture |
    | \(C_v\) | local vertex dual cell | `connection_holonomy(...).local_products` |
    | \(\gamma_1,\gamma_2\) | exact integral dual generators | `integral_dual_cycles(...)` |
    | \(\prod_e q_e^{c_e}\) | ordered represented holonomy | `connection_holonomy(...)` |
    | integrability authority | local and generator products within one circular limit | `admit_integrable_connection(...)` |
    | \(X_f\in T_fK\) | ambient unit tangent face direction | `integrate_direction_field(...)` |

    The three cases share one testable constructor in `support.connections`.
    For the quarter-turn case, let \(\alpha_f\) be the major angle of face centroid
    \(f\), and for canonical dual edge \(e=(s,t)\) define

    \[
    w_e=\operatorname{atan2}(\sin(\alpha_t-\alpha_s),
                              \cos(\alpha_t-\alpha_s)),\qquad
    q_e=\exp(iw_e/4),
    \]

    \[
    \delta_e=\arg\!\left(q_e\overline{L_e}\right).
    \]

    Thus the represented composed transport agrees with \(q_e\) within binary64
    circular tolerance: locally flat, but with maximum absolute generator error
    \(\pi/2\).
    """)
    return


@app.cell
def _(mo):
    mesh_preset = mo.ui.dropdown(
        options=("4 x 6", "6 x 8", "8 x 10"),
        value="4 x 6",
        label="bounded torus preset",
    )
    connection_case = mo.ui.dropdown(
        options=("levi-civita", "cancelled", "quarter-turn"),
        value="cancelled",
        label="represented connection",
    )
    mo.vstack([mesh_preset, connection_case])
    return connection_case, mesh_preset


@app.cell
def _(mo):
    anchor_phase = mo.ui.slider(
        -3.0,
        3.0,
        value=0.0,
        step=0.25,
        label="field anchor phase (display only)",
    )
    anchor_phase
    return (anchor_phase,)


@app.cell
def _(
    SurfaceError,
    admit_integrable_connection,
    connection_case,
    connection_holonomy,
    integral_dual_cycles,
    mesh_preset,
    np,
    torus,
    torus_connection,
):
    major_sections, minor_sections = (
        int(value) for value in mesh_preset.value.split(" x ")
    )
    torus_complex, torus_geometry, _minor_angles = torus(major_sections, minor_sections)
    connection = torus_connection(torus_geometry, connection_case.value)
    dual_cycles = integral_dual_cycles(torus_geometry)
    holonomy = connection_holonomy(connection, dual_cycles)
    unit_transport_residual = float(
        np.max(np.abs(np.abs(connection.transport_products()) - 1.0), initial=0.0)
    )
    expected_admitted = connection_case.value == "cancelled"
    admission_reason = {
        "levi-civita": "rejected: local curvature produces nontrivial contractible holonomy",
        "cancelled": "admitted: represented local and generator holonomy are both cancelled",
        "quarter-turn": "rejected: local holonomy is flat, but a torus generator turns by π/2",
    }[connection_case.value]
    try:
        integrable = admit_integrable_connection(connection, dual_cycles)
        admission_message = "admitted"
    except SurfaceError as error:
        integrable = None
        admission_message = str(error)
    admission_matches_expectation = (integrable is not None) == expected_admitted
    owner_identity = bool(
        connection.geometry is torus_geometry
        and dual_cycles.geometry is torus_geometry
        and (integrable is None or integrable.connection is connection)
        and (integrable is None or integrable.cycles is dual_cycles)
    )
    return (
        admission_matches_expectation,
        admission_message,
        admission_reason,
        connection,
        dual_cycles,
        expected_admitted,
        holonomy,
        integrable,
        major_sections,
        minor_sections,
        owner_identity,
        torus_complex,
        torus_geometry,
        unit_transport_residual,
    )


@app.cell
def _(
    FaceVectors,
    anchor_phase,
    integrable,
    integrate_direction_field,
    np,
    plot_geometry,
    plot_surface_vectors,
    torus_geometry,
    triangle_frames,
):
    if integrable is None:
        direction_field = None
        field_evidence = None
        unit_length_residual = None
        tangent_residual = None
        crossing_error = None
        anchor_provenance = None
        phase_covariance_error = None
        direction_figure = plot_geometry(
            torus_geometry,
            title="No direction field: this connection was not admitted",
        )
    else:
        integrated = integrate_direction_field(
            integrable, anchor_phase=float(anchor_phase.value)
        )
        direction_field = integrated.output
        field_evidence = integrated.evidence
        vectors = direction_field.vectors()
        frames = triangle_frames(torus_geometry)
        unit_length_residual = float(
            np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1.0), initial=0.0)
        )
        tangent_residual = float(
            np.max(
                np.abs(np.einsum("ij,ij->i", vectors, frames.normals())),
                initial=0.0,
            )
        )
        crossing_error = field_evidence.crossing_error
        anchor_provenance = bool(
            direction_field.anchor_face == 0
            and direction_field.anchor_phase == float(anchor_phase.value)
            and direction_field.geometry is torus_geometry
            and direction_field.connection is integrable.connection
        )
        quarter = integrate_direction_field(
            integrable, anchor_phase=float(anchor_phase.value) + np.pi / 2.0
        ).output
        expected_quarter = (
            -np.sin(direction_field.phases())[:, None] * frames.first_axes()
            + np.cos(direction_field.phases())[:, None] * frames.second_axes()
        )
        phase_covariance_error = float(
            np.max(np.abs(quarter.vectors() - expected_quarter), initial=0.0)
        )
        direction_figure = plot_surface_vectors(
            FaceVectors(torus_geometry, vectors),
            scale=0.28,
            title="Admitted ambient face direction field",
        )
    return (
        anchor_provenance,
        crossing_error,
        direction_field,
        direction_figure,
        field_evidence,
        phase_covariance_error,
        tangent_residual,
        unit_length_residual,
    )


@app.cell
def _(
    admission_reason,
    connection_case,
    expected_admitted,
    major_sections,
    minor_sections,
    mo,
):
    mo.md(rf"""
    ## Computation

    The selected bounded mesh is **{major_sections}×{minor_sections}**, and the
    represented connection is **{connection_case.value}**. The expected admission
    result is **{expected_admitted}**.

    **Admission reason:** {admission_reason}.

    Topology, dual cycles, connection construction, and holonomy are computed before
    the independent anchor control. Moving the anchor slider only reintegrates and
    redraws an already admitted connection.

    ## Visualization

    An admitted case is rendered as actual ambient face arrows. Arrowheads show
    orientation, while shaft length is proportional to vector magnitude. This
    integrated field is certified unit-length, so its arrows are intentionally
    uniform. A rejected case shows the same geometry without inventing a direction
    field.
    """)
    return


@app.cell
def _(direction_figure):
    direction_figure
    return


@app.cell
def _(
    admission_matches_expectation,
    admission_message,
    anchor_provenance,
    connection_case,
    crossing_error,
    expected_admitted,
    field_evidence,
    holonomy,
    mo,
    owner_identity,
    phase_covariance_error,
    tangent_residual,
    unit_length_residual,
    unit_transport_residual,
):
    local_pass = holonomy.local_error <= holonomy.limit
    generator_pass = holonomy.generator_error <= holonomy.limit
    field_limit = None if field_evidence is None else field_evidence.limit
    mo.md(rf"""
    ## Evaluation

    | Independent law | Observed | Contract |
    |---|---:|---:|
    | unit-complex transport residual | `{unit_transport_residual:.3e}` | near zero |
    | local holonomy error | `{holonomy.local_error:.3e}` | limit `{holonomy.limit:.3e}` |
    | local holonomy passes | `{local_pass}` | case-dependent |
    | generator holonomy error | `{holonomy.generator_error:.16g}` | limit `{holonomy.limit:.3e}` |
    | generator holonomy passes | `{generator_pass}` | case-dependent |
    | expected admission | `{expected_admitted}` | selected case |
    | observed admission | `{admission_message}` | matches: `{admission_matches_expectation}` |
    | exact geometry/connection identity | `{owner_identity}` | `True` |
    | field unit-length residual | `{unit_length_residual}` | near zero when admitted |
    | field tangent residual | `{tangent_residual}` | near zero when admitted |
    | field crossing error | `{crossing_error}` | limit `{field_limit}` |
    | anchor provenance | `{anchor_provenance}` | `True` when admitted |
    | global-phase covariance error | `{phase_covariance_error}` | near zero when admitted |

    The quarter-turn case reports the maximum absolute error over the retained
    integral generator basis, not a claim about each basis element. For every bounded
    preset its maximum is \(\pi/2\) within represented circular tolerance.

    Selected case: **{connection_case.value}**.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interpretation

    Levi–Civita transport on the curved torus fails through local curvature. The
    cancelled represented transport passes both local and global tests and therefore
    authorizes an ambient tangent direction field. The quarter-turn connection is
    locally flat but globally obstructed: its nontrivial torus generator prevents a
    single-valued integrated field.

    Admission is a semantic boundary. Holonomy evidence describes the products;
    only `admit_integrable_connection` grants the capability consumed by field
    integration.
    """)
    return


if __name__ == "__main__":
    app.run()
