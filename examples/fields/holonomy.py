import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from polygeo.geometry import TriangleSurface
    from polygeo.plot import form as plot_form
    from examples.support.meshes import torus

    return TriangleSurface, mo, np, plot_form, torus


@app.cell
def _(mo):
    mo.md(r"""
    # Connection and holonomy on a torus

    ## Question and prerequisites

    How can local tangent frames describe parallel transport without making the
    answer depend on those frames, and how do contractible curvature loops differ
    from noncontractible periods?

    We assume an oriented, closed, connected triangle surface in three-dimensional
    Euclidean space. The previous topology study supplies exact integral cocycles;
    here they become signed crossing data for closed walks in the dual graph.
    Angles always represent elements of the circle, so equality means equality
    modulo \(2\pi\), not equality of arbitrary real representatives.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. A deterministic face frame is a coordinate choice

    For a stored oriented face \(f=(i,j,k)\), let \(\sigma_f\in\{-1,1\}\) be its
    orientation sign. Nondegeneracy permits the construction

    \[
    u_f=\frac{p_j-p_i}{\lVert p_j-p_i\rVert},\qquad
    n_f=\sigma_f\frac{(p_j-p_i)\times(p_k-p_i)}
                         {\lVert(p_j-p_i)\times(p_k-p_i)\rVert},\qquad
    v_f=n_f\times u_f.
    \]

    The cross product is perpendicular to its factors, and the two normalized
    inputs are perpendicular. Therefore

    \[
    u_f\cdot v_f=v_f\cdot n_f=n_f\cdot u_f=0,
    \quad \lVert u_f\rVert=\lVert v_f\rVert=\lVert n_f\rVert=1,
    \quad u_f\times v_f=n_f.
    \]

    These rules select reproducible coordinates; they do not add geometry. Another
    oriented frame on the same tangent plane is obtained from an angle \(\lambda_f\):

    \[
    u'_f=\cos\lambda_f\,u_f+\sin\lambda_f\,v_f,
    \qquad v'_f=-\sin\lambda_f\,u_f+\cos\lambda_f\,v_f.
    \]

    This is a gauge change: points, tangent planes, lengths, and transported
    vectors remain fixed while their complex coordinates change.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Unfolding constructs adjacent-face transport

    Give a shared primal edge its canonical unit axis \(a_e\), and choose the face
    with smaller stored index as source \(f\), the other as target \(g\). The signed
    dihedral angle rotating the source normal to the target normal about \(a_e\) is

    \[
    \phi_{fg}=\operatorname{atan2}
      \bigl(a_e\cdot(n_f\times n_g),\ n_f\cdot n_g\bigr).
    \]

    Rodrigues rotation unfolds the source tangent vector into the target plane:

    \[
    U_{fg}(x)=x\cos\phi_{fg}+(a_e\times x)\sin\phi_{fg}
      +a_e(a_e\cdot x)(1-\cos\phi_{fg}).
    \]

    Since this is an orientation-preserving isometry, the target-frame coordinate

    \[
    q_{fg}=\langle U_{fg}u_f,u_g\rangle
             +i\langle U_{fg}u_f,v_g\rangle
    \]

    has unit modulus. Thus \(q_{fg}=e^{i\omega_{fg}}\). We retain the principal
    representative

    \[
    W(t)=\operatorname{Arg}(e^{it})
        =\operatorname{atan2}(\sin t,\cos t)\in(-\pi,\pi],
    \qquad \omega_{fg}=W(\operatorname{Arg}q_{fg}).
    \]

    Reversing the dual orientation conjugates \(q_{fg}\) and negates its angle
    modulo \(2\pi\).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Gauge increments telescope on closed walks

    The coordinate of a fixed tangent vector in the rotated frame is multiplied by
    \(e^{-i\lambda_f}\). Transport commutes with an oriented planar rotation, so
    rotating both endpoint frames gives

    \[
    q'_{fg}=e^{-i\lambda_g}q_{fg}e^{i\lambda_f},
    \qquad
    \omega'_{fg}=W(\omega_{fg}+\lambda_f-\lambda_g).
    \]

    For a closed ordered dual walk
    \(f_0\to f_1\to\cdots\to f_m=f_0\), sum before wrapping. The gauge terms cancel
    pairwise:

    \[
    W\!\left(\sum_{r=0}^{m-1}\omega'_{f_rf_{r+1}}\right)
    =W\!\left(\sum_r\omega_{f_rf_{r+1}}
      +\sum_r(\lambda_{f_r}-\lambda_{f_{r+1}})\right)
    =W\!\left(\sum_r\omega_{f_rf_{r+1}}\right).
    \]

    Consequently a closed-loop angle is gauge invariant even though every edge
    representative is gauge dependent.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Contractible loops measure curvature, not flatness

    The incident faces around a primal vertex \(v\) form a contractible ordered
    dual loop. Let \(\epsilon_{ve}\in\{-1,1\}\) orient each crossed dual edge and set

    \[
    \Theta_v=W\!\left(\sum_{e\ni v}\epsilon_{ve}\omega_e\right).
    \]

    The integrated Gaussian curvature at an interior vertex is the angle defect

    \[
    K_v=2\pi-\sum_{f\ni v}\alpha_{vf}.
    \]

    With the source/target and boundary signs fixed above, unfolding each adjacent
    pair makes the full turn satisfy \(\Theta_v=-K_v\pmod{2\pi}\). The circular
    consistency residual is therefore

    \[
    r_v=\lvert W(\Theta_v+K_v)\rvert.
    \]

    A small \(r_v\) certifies agreement between transport and angle defect. It does
    not say the connection is flat: flatness would additionally require every
    \(\lvert\Theta_v\rvert\) to be small, which a curved torus does not satisfy.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Noncontractible cycles carry global periods

    Let an exact integral cocycle \(c^{(r)}\) encode signed crossings of a primitive
    noncontractible dual cycle. If \(s_e\) is the source-face incidence sign, the
    traversal exponent is \(\epsilon_e=-s_ec^{(r)}_e\), and its period is

    \[
    P_r=W\!\left(\sum_e\epsilon_e\omega_e\right).
    \]

    The cocycle law makes these crossings a closed dual walk, so the same telescopic
    calculation removes every gauge increment. Unlike a vertex loop, this walk
    cannot be contracted to a disk; its period is global data and is not determined
    by one enclosed angle defect.

    A plotted cocycle lives on primal edges and marks which dual edges are crossed,
    with multiplicity and sign. It is a carrier witness for the cycle. It is neither
    the connection values \(\omega_e\) nor a geometric drawing of the dual path.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Pseudocode

    ~~~text
    construct one oriented torus and a deterministic tangent frame on every face
    for each canonically oriented adjacent-face pair:
        unfold the source plane across the shared edge
        express the transported source axis in the target frame
        principal-wrap its angle
    around every primal vertex:
        sum transports with the induced dual-loop orientations
        compare the wrapped sum with the signed angle defect
    construct two exact integral noncontractible dual cycles
    sum transport angles with their signed crossing multiplicities
    rotate every face frame by a deterministic gauge angle
    verify that all closed-loop angles are unchanged after circular comparison
    report local curvature consistency and both global periods separately
    render the two primal crossing cocycles as carrier witnesses
    ~~~
    """)
    return


@app.cell
def _(TriangleSurface, np, torus):
    def wrap(values):
        return np.arctan2(np.sin(values), np.cos(values))

    domain, geometry = torus(12, 8)
    surface = TriangleSurface.admit(geometry)
    cycles, connection = domain.dual_cycles(), surface.levi_civita()
    native_evidence = connection.holonomy(cycles)

    first = surface.first_frame_axes_numpy_copy()
    second = surface.second_frame_axes_numpy_copy()
    normals = surface.face_unit_normals().values_numpy_copy()
    frame_error = max(
        float(np.max(np.abs(np.einsum("ij,ij->i", first, second)))),
        float(np.max(np.abs(np.linalg.norm(first, axis=1) - 1.0))),
        float(np.max(np.abs(np.cross(first, second) - normals))),
    )

    edge_indices = connection.interior_edge_indices_numpy_copy()
    transports = connection.transports_numpy_copy()
    transport_angles = np.arctan2(transports[:, 1], transports[:, 0])
    transport_error = float(np.max(np.abs(np.linalg.norm(transports, axis=1) - 1.0)))
    d2_signs, d2_faces, d2_starts, _ = domain.boundary_parts_numpy_copy(2)
    starts = d2_starts[edge_indices]
    left_faces, right_faces = d2_faces[starts], d2_faces[starts + 1]
    left_is_source = left_faces < right_faces
    source_faces = np.where(left_is_source, left_faces, right_faces)
    target_faces = np.where(left_is_source, right_faces, left_faces)
    source_signs = np.where(left_is_source, d2_signs[starts], d2_signs[starts + 1])
    edge_count = domain.simplex_count(1)
    signs_by_edge = np.zeros(edge_count, dtype=np.int8)
    angles_by_edge = np.zeros(edge_count)
    signs_by_edge[edge_indices] = source_signs
    angles_by_edge[edge_indices] = transport_angles
    cycle_crossings = np.zeros((cycles.rank, edge_count))
    for index in range(cycles.rank):
        cycle_edges, coefficients = cycles.cocycle(index).to_python_copy()
        cycle_edges = np.asarray(cycle_edges, dtype=np.int64)
        cycle_crossings[index, cycle_edges] = -signs_by_edge[cycle_edges] * np.asarray(
            coefficients, dtype=np.float64
        )
    vertex_boundary = domain.boundary_scipy_copy(1)

    def closed_angles(edge_angles):
        local = wrap(vertex_boundary @ (-signs_by_edge * edge_angles))
        return local, wrap(cycle_crossings @ edge_angles)

    vertex_holonomy, periods = closed_angles(angles_by_edge)
    curvature = surface.gaussian_curvature_measure().coefficients_numpy_copy()
    curvature_error = float(np.max(np.abs(wrap(vertex_holonomy + curvature))))
    native_reproduction_error = max(
        abs(native_evidence.local_error - float(np.max(np.abs(vertex_holonomy)))),
        abs(native_evidence.generator_error - float(np.max(np.abs(periods)))),
    )
    gauge = 0.37 * np.sin(np.arange(surface.face_count) + 0.2)
    gauged_angles = wrap(transport_angles + gauge[source_faces] - gauge[target_faces])
    gauged_by_edge = np.zeros(edge_count)
    gauged_by_edge[edge_indices] = gauged_angles
    gauged_local, gauged_periods = closed_angles(gauged_by_edge)
    gauge_error = max(
        float(np.max(np.abs(wrap(gauged_local - vertex_holonomy)))),
        float(np.max(np.abs(wrap(gauged_periods - periods)))),
    )

    if (
        cycles.rank != 2
        or max(
            frame_error,
            transport_error,
            curvature_error,
            native_reproduction_error,
            gauge_error,
        )
        > native_evidence.limit
    ):
        raise RuntimeError("connection-holonomy evidence exceeds its limit")
    if native_evidence.local_error <= 0.1 or native_evidence.generator_error <= 0.1:
        raise RuntimeError("the torus probe does not expose local and global holonomy")

    holonomy_evidence = {
        "simplex_counts": tuple(domain.simplex_count(k) for k in range(3)),
        "cycle_rank": cycles.rank,
        "frame_error": frame_error,
        "transport_error": transport_error,
        "curvature_consistency_error": curvature_error,
        "maximum_local_holonomy": native_evidence.local_error,
        "generator_periods": tuple(float(value) for value in periods),
        "maximum_generator_magnitude": native_evidence.generator_error,
        "native_reproduction_error": native_reproduction_error,
        "gauge_invariance_error": gauge_error,
        "limit": native_evidence.limit,
    }
    cycle_forms = tuple(
        domain.binary64_cochain_space(1).realize_integral(cycles.cocycle(index))
        for index in range(cycles.rank)
    )
    return cycle_forms, geometry, holonomy_evidence


@app.cell
def _(holonomy_evidence, mo):
    _periods = holonomy_evidence["generator_periods"]
    mo.md(rf"""
    ## 7. Numerical evidence

    The torus has {holonomy_evidence["simplex_counts"]} vertices, edges, and faces,
    and its exact noncontractible dual-cycle rank is
    {holonomy_evidence["cycle_rank"]}.

    | Local or construction certificate | Observed error |
    |:--|--:|
    | Frame orthonormality and handedness | {holonomy_evidence["frame_error"]:.3e} |
    | Unit-complex transport normalization | {holonomy_evidence["transport_error"]:.3e} |
    | Wrapped local holonomy plus angle defect | {holonomy_evidence["curvature_consistency_error"]:.3e} |
    | Aggregate native-result reproduction | {holonomy_evidence["native_reproduction_error"]:.3e} |
    | Closed-loop gauge invariance | {holonomy_evidence["gauge_invariance_error"]:.3e} |

    | Holonomy observable | Wrapped angle in radians |
    |:--|--:|
    | Maximum contractible vertex-loop magnitude | {holonomy_evidence["maximum_local_holonomy"]:.6f} |
    | First noncontractible generator | {_periods[0]:.6f} |
    | Second noncontractible generator | {_periods[1]:.6f} |
    | Maximum generator magnitude | {holonomy_evidence["maximum_generator_magnitude"]:.6f} |

    Every reconstruction, consistency, and invariance error is below the declared
    circular limit {holonomy_evidence["limit"]:.3e}. The nonzero local magnitude
    records curvature; it is not mislabeled as an error or a flatness verdict.
    """)
    return


@app.cell
def _(cycle_forms, geometry, mo, plot_form):
    cycle_figures = [
        plot_form(geometry, form, title=f"Generator {index + 1} crossing cocycle")
        for index, form in enumerate(cycle_forms)
    ]
    mo.vstack(
        [
            mo.md(r"""
            ## 8. Exact cycle carriers

            Color and line orientation encode integral coefficients on primal edges.
            These panels identify signed crossings only; they do not draw the dual
            walks or encode the connection angles accumulated along them.
            """),
            mo.hstack(cycle_figures),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation and limits

    The experiment separates representation from observation. Face frames and edge
    angles change under gauge rotation, while wrapped closed-loop sums do not. Local
    contractible sums reproduce angle defects; exact noncontractible cycles expose
    two additional global observables, one nearly zero and one nonzero for the
    selected torus, embedding, frames, and cycle basis.

    The reported periods are basis- and path-representative dependent in a curved
    connection; only their correctly wrapped evaluation is asserted. This finite
    experiment does not prove mesh convergence, characterize every loop, or claim
    that a nonzero local holonomy is a failure of Levi-Civita transport.
    """)
    return


if __name__ == "__main__":
    app.run()
