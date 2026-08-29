import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from polygeo import TriangleSurface, plot_geometry, plot_surface_vectors
    from support.meshes import icosphere, torus

    return (
        TriangleSurface,
        icosphere,
        mo,
        np,
        plot_geometry,
        plot_surface_vectors,
        torus,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Curvature on triangle surfaces

    ## Mathematical question
    How do discrete normal and curvature constructions distinguish a sphere from a torus?

    ## From mathematics to PolyGeo
    A realization admits one `TriangleSurface`; all vector fields use the same native carrier.

    ## Computation
    Compute normals and integrated Gaussian curvature directly from that owner.

    ## Visualization
    Plot immutable geometry and vector snapshots.

    ## Evaluation
    Check sphere Gauss--Bonnet error and `min < 0 < max` on the torus.

    ## Interpretation
    Curvature is mathematical data; NumPy arrays are explicit copied observations.
    """)
    return


@app.cell
def _(TriangleSurface, icosphere, np, torus):
    _, sphere_geometry = icosphere(1, 1.0)
    _, torus_geometry = torus(16, 10)
    sphere_surface = TriangleSurface.admit(sphere_geometry)
    torus_surface = TriangleSurface.admit(torus_geometry)
    sphere_curvature = (
        sphere_surface.gaussian_curvature_measure().coefficients_numpy_copy()
    )
    torus_curvature = (
        torus_surface.gaussian_curvature_measure().coefficients_numpy_copy()
    )
    sphere_normals = sphere_surface.sphere_inscribed_vertex_normals()
    sphere_total = float(np.sum(sphere_curvature))
    torus_min = float(np.min(torus_curvature))
    torus_max = float(np.max(torus_curvature))
    curvature_evidence = {
        "sphere_gauss_bonnet_error": abs(sphere_total - 4.0 * np.pi),
        "torus_min": torus_min,
        "torus_max": torus_max,
        "torus_changes_sign": torus_min < 0.0 < torus_max,
    }
    return curvature_evidence, sphere_geometry, sphere_normals


@app.cell
def _(
    curvature_evidence,
    mo,
    plot_geometry,
    plot_surface_vectors,
    sphere_geometry,
    sphere_normals,
):
    mo.vstack(
        [
            mo.md(f"`{curvature_evidence}`"),
            plot_geometry(sphere_geometry),
            plot_surface_vectors(sphere_normals),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
