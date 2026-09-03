import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from polygeo.geometry import TriangleSurface
    from polygeo.plot import form as plot_form, vectors as plot_surface_vectors
    from examples.support.meshes import icosphere, torus

    return TriangleSurface, icosphere, mo, np, plot_form, plot_surface_vectors, torus


@app.cell
def _(mo):
    mo.md(r"""
    # Curvature on triangle surfaces

    ## Question and prerequisites

    Where does smooth Gaussian curvature live after triangulation?

    A smooth surface distributes curvature continuously, but every triangle in a
    piecewise-flat surface is intrinsically Euclidean. We will locate the missing
    curvature, derive its total from topology, and compare it with a separate
    construction of vertex-normal directions.

    We assume oriented triangle meshes, dot and cross products, and the statement
    of Gauss--Bonnet. No earlier study is required.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Mathematical model

    Let $S\subset\mathbb{R}^3$ be an oriented smooth surface with unit normal
    $n$. At a point $p$, the shape operator is

    $$
    A_p=-D n_p:T_pS\longrightarrow T_pS.
    $$

    Its eigenvalues $\kappa_1$ and $\kappa_2$ are the principal curvatures, and
    Gaussian curvature is their product:

    $$
    K(p)=\det A_p=\kappa_1\kappa_2.
    $$

    Curvature $K$ has units of inverse length squared, while area $dA$ has units
    of length squared. Thus $K\,dA$ is a dimensionless geometric measure. For a
    region $U$ homeomorphic to a disk, Gauss--Bonnet states

    $$
    \int_U K\,dA
      +\int_{\partial U} k_g\,ds
      +\sum_i \alpha_i
      =2\pi,
    $$

    where $k_g$ is boundary geodesic curvature and $\alpha_i$ are exterior turns
    at boundary corners. The invariant object is therefore the integrated measure
    $K\,dA$, not a coordinate-dependent point sample of $K$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Discrete model

    ### Curvature at an interior vertex

    Every open triangle is flat, so its interior contribution to
    $\int K\,dA$ vanishes. Around a vertex $v$, let

    $$
    \Omega_v=\sum_{f\ni v}\theta_{vf}
    $$

    be the sum of incident corner angles. Imagine each incident triangle as a
    Euclidean wedge and follow a small circle of radius $r$ around $v$. A wedge
    of angle $\theta_{vf}$ contributes arc length $r\theta_{vf}$ and geodesic
    curvature $1/r$, hence

    $$
    \int_{\text{wedge arc}} k_g\,ds
      =\frac{1}{r}(r\theta_{vf})
      =\theta_{vf}.
    $$

    Summing the arcs gives $\int k_g\,ds=\Omega_v$. The neighborhood is a disk
    and has no boundary corners, so local Gauss--Bonnet leaves the concentrated
    curvature

    $$
    K_v=2\pi-\Omega_v
       =2\pi-\sum_{f\ni v}\theta_{vf}.
    $$

    A positive defect means the one-ring contains less than a full $2\pi$ turn;
    a negative defect means it contains an angular excess.

    ### Boundary vertices

    A small boundary neighborhood is a half-disk. Its circular arcs again
    contribute $\Omega_v$, while the two endpoints contribute exterior turns
    totaling $\pi$. Therefore

    $$
    K_v+\Omega_v+\pi=2\pi,
    \qquad
    K_v=\pi-\Omega_v.
    $$

    Here $K_v$ is the boundary turning, or geodesic-curvature contribution,
    included in the same integrated measure convention.

    ### From local defects to topology

    For a closed triangulation with $V$ vertices, $E$ edges, and $F$ faces,
    every Euclidean triangle contributes total corner angle $\pi$. Hence

    $$
    \begin{aligned}
    \sum_v K_v
      &=2\pi V-\sum_f\sum_{v\in f}\theta_{vf}\\
      &=2\pi V-\pi F.
    \end{aligned}
    $$

    Every edge belongs to two faces, so counting face-edge incidences gives
    $3F=2E$. Substitution completes the discrete Gauss--Bonnet derivation:

    $$
    \sum_v K_v
      =2\pi V-2\pi(E-F)
      =2\pi(V-E+F)
      =2\pi\chi.
    $$

    With boundary, split vertices and edges into interior and boundary sets.
    Each boundary component is a polygonal circle, so $E_\partial=V_\partial$,
    while incidence counting gives $3F=2E_\mathrm{int}+E_\partial$. Then

    $$
    \begin{aligned}
    \sum_v K_v
      &=2\pi V_\mathrm{int}+\pi V_\partial-\pi F\\
      &=2\pi(V-E+F)
      =2\pi\chi.
    \end{aligned}
    $$

    ### A separate vertex-normal construction

    Curvature defect is a scalar measure. The displayed normal is instead a
    direction derived from the embedding. At a corner of an oriented face, let
    $a$ and $b$ be the two outgoing edge vectors and write
    $\widehat a=a/\lVert a\rVert$ and
    $\widehat b=b/\lVert b\rVert$. The corner contributes

    $$
    c_{vf}
      =\frac{\widehat a\times\widehat b}
             {\lVert a\rVert\,\lVert b\rVert}.
    $$

    The sphere-inscribed vertex direction is

    $$
    n_v=
    \frac{\sum_{f\ni v}c_{vf}}
         {\left\lVert\sum_{f\ni v}c_{vf}\right\rVert}.
    $$

    The weights emphasize nearby corners; a uniform rescaling multiplies every
    contribution by the same factor and leaves the normalized direction
    unchanged. Because $\lVert n_v\rVert=1$, arrow length carries no curvature
    magnitude.

    Each $K_v$ is an integrated angle in radians supported at a vertex. Dividing
    by a chosen dual area would define a curvature density, but no such area
    choice is made here. Finite defects and normal directions alone do not prove
    pointwise convergence to a smooth surface.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Algorithm

    ```text
    for each oriented triangulated surface:
        construct the mesh once
        for each triangle:
            compute its three corner angles
            add each angle to its incident vertex
        for each vertex:
            reference_angle := pi on the boundary, otherwise 2*pi
            defect[vertex] := reference_angle - accumulated_angle[vertex]
        chi := vertex_count - edge_count + face_count
        compare sum(defect) with 2*pi*chi

    for the sphere only:
        for each oriented corner with edge vectors a and b:
            contribution := (unit(a) cross unit(b))
                            / (length(a)*length(b))
            add contribution to its incident vertex
        normalize every nonzero vertex sum

    report both total-defect identities, the torus signed range,
        and the maximum unit-normal error
    render both defect measures and the sphere normal directions
    ```

    The evidence and figures reuse these computed defects and directions; neither
    path repeats the geometric construction.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Experiment

    We compare a closed sphere ($\chi=2$) with a closed torus ($\chi=0$).
    Each deterministic mesh is constructed once, then each curvature measure and
    the sphere normal field are computed once.
    """)
    return


@app.cell
def _(TriangleSurface, icosphere, torus):
    sphere_domain, sphere_geometry = icosphere(1, 1.0)
    torus_domain, torus_geometry = torus(16, 10)
    sphere_surface = TriangleSurface.admit(sphere_geometry)
    torus_surface = TriangleSurface.admit(torus_geometry)
    return (
        sphere_domain,
        sphere_geometry,
        sphere_surface,
        torus_domain,
        torus_geometry,
        torus_surface,
    )


@app.cell
def _(sphere_surface, torus_surface):
    sphere_curvature = sphere_surface.gaussian_curvature_measure()
    torus_curvature = torus_surface.gaussian_curvature_measure()
    sphere_normals = sphere_surface.sphere_inscribed_vertex_normals()
    sphere_coefficients = sphere_curvature.coefficients_numpy_copy()
    torus_coefficients = torus_curvature.coefficients_numpy_copy()
    return (
        sphere_coefficients,
        sphere_curvature,
        sphere_normals,
        torus_coefficients,
        torus_curvature,
    )


@app.cell
def _(
    mo,
    np,
    sphere_coefficients,
    sphere_domain,
    sphere_normals,
    torus_coefficients,
    torus_domain,
):
    sphere_chi = sum(
        (-1) ** degree * sphere_domain.simplex_count(degree) for degree in range(3)
    )
    torus_chi = sum(
        (-1) ** degree * torus_domain.simplex_count(degree) for degree in range(3)
    )
    sphere_total = float(np.sum(sphere_coefficients))
    torus_total = float(np.sum(torus_coefficients))
    sphere_expected = 2.0 * np.pi * sphere_chi
    torus_expected = 2.0 * np.pi * torus_chi
    sphere_error = abs(sphere_total - sphere_expected)
    torus_error = abs(torus_total - torus_expected)
    torus_min = float(np.min(torus_coefficients))
    torus_max = float(np.max(torus_coefficients))
    normal_values = sphere_normals.values_numpy_copy()
    normal_length_error = float(
        np.max(np.abs(np.linalg.norm(normal_values, axis=1) - 1.0))
    )
    defect_limit = 1.0e-12
    normal_limit = 1.0e-12
    if not (
        sphere_error <= defect_limit
        and torus_error <= defect_limit
        and torus_min < 0.0 < torus_max
        and normal_length_error <= normal_limit
    ):
        raise RuntimeError("curvature-study evidence exceeds its declared limits")

    mo.md(rf"""
    ## Evidence

    Euler characteristic is computed independently as $V-E+F$. The defect totals
    use an absolute tolerance of `{defect_limit:.1e}` radians. Normal lengths are
    recomputed from the displayed vectors and use the dimensionless tolerance
    `{normal_limit:.1e}`.

    | Quantity | Observed | Expected claim | Result |
    |---|---:|---:|---|
    | Sphere total defect | `{sphere_total:.15g}` | $2\pi\chi={sphere_expected:.15g}$ | error `{sphere_error:.3e}` |
    | Torus total defect | `{torus_total:.15g}` | $2\pi\chi={torus_expected:.1f}$ | error `{torus_error:.3e}` |
    | Torus signed range | `[{torus_min:.6f}, {torus_max:.6f}]` | $\min K_v<0<\max K_v$ | `{torus_min < 0.0 < torus_max}` |
    | Sphere normal length | max error `{normal_length_error:.3e}` | $\max_v |\lVert n_v\rVert-1|\leq {normal_limit:.1e}$ | `{normal_length_error <= normal_limit}` |

    Both total-defect errors are below their stated tolerance. The sphere therefore
    carries total integrated curvature $4\pi$, while positive and negative torus
    defects cancel to zero globally. The independent length check establishes that
    the displayed normal field carries directions of unit magnitude.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualization

    In the first two figures, marker position identifies the supporting vertex
    and the diverging color scale shows the signed integrated angle defect in
    radians. Neutral faces, edges, and vertices provide topology context. In the
    third figure, each orange arrow begins at a sphere vertex and shows the
    sphere-inscribed unit-normal direction.
    """)
    return


@app.cell
def _(
    mo,
    plot_form,
    plot_surface_vectors,
    sphere_curvature,
    sphere_geometry,
    sphere_normals,
    torus_curvature,
    torus_geometry,
):
    sphere_curvature_figure = plot_form(
        sphere_geometry, sphere_curvature, title="Sphere integrated curvature"
    )
    torus_curvature_figure = plot_form(
        torus_geometry, torus_curvature, title="Torus integrated curvature"
    )
    sphere_normals_figure = plot_surface_vectors(
        sphere_normals, scale=0.35, title="Sphere-inscribed vertex normals"
    )
    mo.vstack([sphere_curvature_figure, torus_curvature_figure, sphere_normals_figure])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation and limits

    Triangulation moves smooth Gaussian curvature into vertex-supported integrated
    angle defects. Gauss--Bonnet constrains their total by topology: the sphere's
    defects sum to $4\pi$, whereas signed defects on the torus cancel to zero.
    The normal arrows describe orientation of the embedded sphere, not curvature
    magnitude. Their unit-length certificate checks normalization, not agreement
    with an unknown smooth normal field.

    These finite-mesh results verify the discrete identity and sign structure;
    they do not establish pointwise convergence to a smooth curvature density.
    The next study uses the same vertex/cochain viewpoint in a global Poisson
    equation.
    """)
    return


if __name__ == "__main__":
    app.run()
