import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import math

    import marimo as mo
    import numpy as np

    from polygeo import (
        ORDINARY_FORM,
        Geometry,
        PositiveHodgeMetric,
        face_unit_normals,
        gaussian_curvature_measure,
        mean_curvature_vectors,
        plot_cochain,
        plot_surface_vectors,
        sphere_inscribed_vertex_normals,
        surface_area_gradient,
        tip_angle_vertex_normals,
        uniform_vertex_normals,
        volume_gradient,
    )
    from support.meshes import icosphere, torus

    return (
        Geometry,
        ORDINARY_FORM,
        PositiveHodgeMetric,
        face_unit_normals,
        gaussian_curvature_measure,
        icosphere,
        math,
        mean_curvature_vectors,
        mo,
        np,
        plot_cochain,
        plot_surface_vectors,
        sphere_inscribed_vertex_normals,
        surface_area_gradient,
        tip_angle_vertex_normals,
        torus,
        uniform_vertex_normals,
        volume_gradient,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Curvature measures and surface-normal estimators

    ## Mathematical question

    A triangle mesh is flat inside each face, so Gaussian curvature is stored at
    vertices as the **integrated angle defect**

    \[
    \mu_i=2\pi-\sum_{t\ni i}\theta_{t,i}.
    \]

    How does this scale-invariant measure compare with analytic pointwise curvature
    on a sphere or torus, and how do common discrete normal estimators differ?
    For a closed mesh, Gauss--Bonnet requires
    \(\sum_i\mu_i=2\pi\chi\), independently of refinement and uniform scale.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From mathematics to PolyGeo

    | Mathematical object | PolyGeo public API |
    |---|---|
    | oriented closed sphere / torus | `icosphere(...)`, `torus(...)` |
    | integrated defect \(\mu_i\) | `gaussian_curvature_measure(geometry)` |
    | diagnostic density \(\mu_i/A_i^{bar}\) | degree-zero form from barycentric areas |
    | oriented face normals | `face_unit_normals(geometry)` |
    | vertex-normal estimators | `volume_gradient(...).normalized()`, `uniform_vertex_normals(...)`, `tip_angle_vertex_normals(...)`, `sphere_inscribed_vertex_normals(...)`, `surface_area_gradient(...).normalized()` |
    | positive-Hodge mean-curvature vector | `mean_curvature_vectors(PositiveHodgeMetric(...))` |
    | scalar/vector views | `plot_cochain(...)`, `plot_surface_vectors(...)` |

    The integrated measure is the primary discrete geometric quantity. Dividing it
    by a barycentric vertex area creates a useful **diagnostic density**, but that
    quotient depends on the chosen area convention and is not what
    `gaussian_curvature_measure` returns.

    Face normals are exact per-face directions. The five vertex-normal constructions
    use different weights or variational gradients; the mean-curvature vector is a
    metric-dependent Laplacian output, not another interchangeable normal formula.
    """)
    return


@app.cell
def _(mo):
    surface_kind = mo.ui.dropdown(
        options=["sphere", "torus"], value="sphere", label="surface"
    )
    surface_kind
    return (surface_kind,)


@app.cell
def _(mo):
    resolution = mo.ui.slider(0, 3, value=2, step=1, label="bounded refinement level")
    resolution
    return (resolution,)


@app.cell
def _(mo):
    radius_scale = mo.ui.slider(0.5, 2.0, value=1.0, step=0.25, label="uniform scale")
    radius_scale
    return (radius_scale,)


@app.cell
def _(mo):
    estimator_choice = mo.ui.dropdown(
        options=[
            "volume gradient",
            "uniform",
            "tip angle",
            "sphere inscribed",
            "area gradient",
        ],
        value="tip angle",
        label="displayed vertex-normal estimator",
    )
    estimator_choice
    return (estimator_choice,)


@app.cell
def _(np):
    def barycentric_vertex_areas(curvature_geometry):
        triangles = curvature_geometry.complex.simplices(2)
        triangle_areas = curvature_geometry.primal_measures(2)
        areas = np.zeros(curvature_geometry.complex.simplex_count(0))
        for corner in range(3):
            np.add.at(areas, triangles[:, corner], triangle_areas / 3.0)
        return areas

    def weighted_rms(values, weights):
        return float(np.sqrt(np.sum(weights * values * values) / np.sum(weights)))

    def max_angle_degrees(vectors, reference):
        dots = np.einsum("ij,ij->i", vectors, reference)
        return float(np.degrees(np.max(np.arccos(np.clip(dots, -1.0, 1.0)))))

    return barycentric_vertex_areas, max_angle_degrees, weighted_rms


@app.cell
def _(icosphere, np, radius_scale, resolution, surface_kind, torus):
    sphere_radius = float(radius_scale.value)
    torus_major = 2.0 * sphere_radius
    torus_minor = 0.7 * sphere_radius
    torus_sections = ((8, 12), (12, 18), (16, 24), (24, 32))

    if surface_kind.value == "sphere":
        domain, geometry = icosphere(resolution.value, sphere_radius)
        analytic_vertex_normals = geometry.positions / sphere_radius
        analytic_curvature = np.full(domain.simplex_count(0), 1.0 / sphere_radius**2)
        shape_detail = f"icosphere subdivision {resolution.value}"
    else:
        major_count, minor_count = torus_sections[resolution.value]
        domain, geometry, minor_angles = torus(
            major_count,
            minor_count,
            torus_major,
            torus_minor,
        )
        major_angles = np.arctan2(geometry.positions[:, 1], geometry.positions[:, 0])
        analytic_vertex_normals = np.column_stack(
            (
                np.cos(major_angles) * np.cos(minor_angles),
                np.sin(major_angles) * np.cos(minor_angles),
                np.sin(minor_angles),
            )
        )
        analytic_curvature = np.cos(minor_angles) / (
            torus_minor * (torus_major + torus_minor * np.cos(minor_angles))
        )
        shape_detail = f"torus grid {major_count} x {minor_count}"
    return (
        analytic_curvature,
        analytic_vertex_normals,
        domain,
        geometry,
        shape_detail,
        sphere_radius,
        torus_major,
        torus_minor,
        torus_sections,
    )


@app.cell
def _(
    ORDINARY_FORM,
    analytic_curvature,
    barycentric_vertex_areas,
    domain,
    gaussian_curvature_measure,
    geometry,
    plot_cochain,
    weighted_rms,
):
    curvature_measure = gaussian_curvature_measure(geometry)
    integrated_values = curvature_measure.coefficients()
    barycentric_areas = barycentric_vertex_areas(geometry)
    diagnostic_values = integrated_values / barycentric_areas
    vertex_space = domain.cochain_space(0)
    diagnostic_density = vertex_space.form(diagnostic_values, ORDINARY_FORM)
    density_rms_error = weighted_rms(
        diagnostic_values - analytic_curvature, barycentric_areas
    )
    measure_figure = plot_cochain(
        geometry, curvature_measure, title="Integrated Gaussian curvature measure"
    )
    density_figure = plot_cochain(
        geometry,
        diagnostic_density,
        title="Barycentric diagnostic density (measure / area)",
    )
    return (
        barycentric_areas,
        curvature_measure,
        density_figure,
        density_rms_error,
        diagnostic_values,
        integrated_values,
        measure_figure,
    )


@app.cell
def _(
    analytic_vertex_normals,
    estimator_choice,
    face_unit_normals,
    geometry,
    max_angle_degrees,
    np,
    plot_surface_vectors,
    sphere_inscribed_vertex_normals,
    surface_area_gradient,
    tip_angle_vertex_normals,
    uniform_vertex_normals,
    volume_gradient,
):
    face_field = face_unit_normals(geometry)
    faces = geometry.complex.simplices(2)
    analytic_face_normals = analytic_vertex_normals[faces].sum(axis=1)
    analytic_face_normals /= np.linalg.norm(analytic_face_normals, axis=1)[:, None]

    normal_fields = {
        "volume gradient": volume_gradient(geometry).normalized(),
        "uniform": uniform_vertex_normals(geometry),
        "tip angle": tip_angle_vertex_normals(geometry),
        "sphere inscribed": sphere_inscribed_vertex_normals(geometry),
        "area gradient": surface_area_gradient(geometry).normalized(),
    }
    normal_errors = {
        name: max_angle_degrees(field.vectors, analytic_vertex_normals)
        for name, field in normal_fields.items()
    }
    face_normal_error = max_angle_degrees(face_field.vectors, analytic_face_normals)
    face_figure = plot_surface_vectors(
        face_field, scale=0.18, title="Oriented face_unit_normals"
    )
    normal_figure = plot_surface_vectors(
        normal_fields[estimator_choice.value],
        scale=0.24,
        title=f"Vertex normals: {estimator_choice.value}",
    )
    return face_figure, face_normal_error, normal_errors, normal_figure


@app.cell
def _(
    PositiveHodgeMetric,
    icosphere,
    mean_curvature_vectors,
    plot_surface_vectors,
    resolution,
    sphere_radius,
):
    _, positive_geometry = icosphere(max(1, resolution.value), sphere_radius)
    positive_metric = PositiveHodgeMetric(positive_geometry)
    mean_field = mean_curvature_vectors(positive_metric)
    mean_figure = plot_surface_vectors(
        mean_field.normalized(),
        scale=0.24,
        title="Normalized mean-curvature vectors on positive-Hodge icosphere",
    )
    return mean_field, mean_figure, positive_metric


@app.cell
def _(
    Geometry,
    barycentric_vertex_areas,
    domain,
    gaussian_curvature_measure,
    geometry,
    math,
    np,
    radius_scale,
    surface_kind,
    torus,
    torus_major,
    torus_minor,
    torus_sections,
    weighted_rms,
    icosphere,
):
    euler_characteristic = sum(
        (-1) ** degree * domain.simplex_count(degree)
        for degree in range(domain.dimension + 1)
    )
    gauss_bonnet_target = 2.0 * math.pi * euler_characteristic
    gauss_bonnet_residual = abs(
        math.fsum(gaussian_curvature_measure(geometry).coefficients())
        - gauss_bonnet_target
    )
    sign_law = (
        float(np.min(gaussian_curvature_measure(geometry).coefficients())) > 0.0
        if surface_kind.value == "sphere"
        else (
            float(np.min(gaussian_curvature_measure(geometry).coefficients()))
            < 0.0
            < float(np.max(gaussian_curvature_measure(geometry).coefficients()))
        )
    )

    scaled_geometry = Geometry.from_positions(domain, 1.7 * geometry.positions)
    scaled_measure = gaussian_curvature_measure(scaled_geometry).coefficients()
    base_measure = gaussian_curvature_measure(geometry).coefficients()
    measure_scale_error = float(np.max(np.abs(scaled_measure - base_measure)))
    base_density = base_measure / barycentric_vertex_areas(geometry)
    scaled_density = scaled_measure / barycentric_vertex_areas(scaled_geometry)
    density_scale_error = float(np.max(np.abs(scaled_density * 1.7**2 - base_density)))

    convergence_rows = []
    for study_level in range(4):
        if surface_kind.value == "sphere":
            study_domain, study_geometry = icosphere(
                study_level, float(radius_scale.value)
            )
            study_analytic = np.full(
                study_domain.simplex_count(0), 1.0 / float(radius_scale.value) ** 2
            )
        else:
            study_major, study_minor = torus_sections[study_level]
            study_domain, study_geometry, study_angles = torus(
                study_major, study_minor, torus_major, torus_minor
            )
            study_analytic = np.cos(study_angles) / (
                torus_minor * (torus_major + torus_minor * np.cos(study_angles))
            )
        study_areas = barycentric_vertex_areas(study_geometry)
        study_density = (
            gaussian_curvature_measure(study_geometry).coefficients() / study_areas
        )
        study_error = weighted_rms(study_density - study_analytic, study_areas)
        convergence_rows.append(
            (
                study_level,
                study_domain.simplex_count(0),
                study_error,
            )
        )
    return (
        convergence_rows,
        density_scale_error,
        euler_characteristic,
        gauss_bonnet_residual,
        gauss_bonnet_target,
        measure_scale_error,
        sign_law,
    )


@app.cell
def _(mo, radius_scale, shape_detail, surface_kind):
    mo.md(
        f"""
    ## Computation

    The selected **{surface_kind.value}** uses {shape_detail} at uniform scale
    **{radius_scale.value:.2f}**. The study computes the integrated measure first,
    then derives barycentric density only as a diagnostic. Analytic sphere or torus
    curvature supplies independent forward-error evidence.

    The torus fixture does not admit `PositiveHodgeMetric` because its
    consistent-diagonal Hodge weights are not strictly positive. Therefore the
    requested mean-curvature-vector study is shown on the positive-Hodge icosphere
    companion, while all five normal estimators use the selected surface.

    ## Visualization

    Figures are vertically ordered: integrated measure, diagnostic density, face
    normals, one selected vertex-normal estimator, and the positive-Hodge
    mean-curvature direction.
    """
    )
    return


@app.cell
def _(measure_figure):
    measure_figure
    return


@app.cell
def _(density_figure):
    density_figure
    return


@app.cell
def _(face_figure):
    face_figure
    return


@app.cell
def _(normal_figure):
    normal_figure
    return


@app.cell
def _(mean_figure):
    mean_figure
    return


@app.cell
def _(
    convergence_rows,
    density_rms_error,
    density_scale_error,
    face_normal_error,
    gauss_bonnet_residual,
    gauss_bonnet_target,
    measure_scale_error,
    mo,
    normal_errors,
    sign_law,
):
    convergence_table = "\n".join(
        f"| {level} | {vertices} | {error:.6e} |"
        for level, vertices, error in convergence_rows
    )
    normal_table = "\n".join(
        f"| `{name}` | {error:.6f} |" for name, error in normal_errors.items()
    )
    mo.md(
        f"""
    ## Evaluation

    | Law / diagnostic | Observed |
    |---|---:|
    | Gauss--Bonnet target | `{gauss_bonnet_target:.12g}` |
    | Gauss--Bonnet residual | `{gauss_bonnet_residual:.3e}` |
    | expected sphere/torus sign law | `{sign_law}` |
    | integrated-measure scale-invariance error | `{measure_scale_error:.3e}` |
    | density inverse-square scale-law error | `{density_scale_error:.3e}` |
    | selected analytic density weighted RMS error | `{density_rms_error:.3e}` |
    | face-normal maximum analytic angle (degrees) | `{face_normal_error:.6f}` |

    | Refinement level | Vertices | Analytic density weighted RMS error |
    |---:|---:|---:|
    {convergence_table}

    | Vertex estimator | Maximum analytic angle (degrees) |
    |---|---:|
    {normal_table}

    These are distinct evaluations: Gauss--Bonnet and sign test the integrated
    measure (`min < 0 < max` on the torus); the RMS table tests one barycentric density convention; angular errors
    compare estimator directions. A small error in one column does not certify the
    other mathematical objects.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    Angle defect is naturally an integrated measure: it is dimensionless, survives
    rigid/scale invariance, and obeys a topological total. A density appears only after
    assigning an area to each vertex; here barycentric areas make that convention
    explicit and reveal the expected inverse-square scaling.

    Normal estimators answer related but different questions. Face normals are local
    to triangles; weighted vertex normals average incident geometry; area and volume
    gradients are variational directions; the positive-Hodge mean-curvature vector
    also contains magnitude and metric information. Their normalized arrows may look
    similar without making the estimators identical.
    """)
    return


if __name__ == "__main__":
    app.run()
