import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from polygeo import TriangleSurface, plot_form
    from support.meshes import icosphere

    return TriangleSurface, icosphere, mo, np, plot_form


@app.cell
def _(mo):
    mo.md(r"""
    # Heat-method distance on a closed surface

    ## Mathematical question
    How can heat flow and a Poisson solve approximate distance to a vertex set?

    ## From mathematics to PolyGeo
    A mass-normalized vertex cochain produces an integrated Dirac heat source.
    Existing gradient, normalized face-field, integrated-divergence, and compatible
    Poisson operations then form the complete heat-method composition.

    ## Computation
    Use `t = h^2` for mean edge length `h`, solve heat once, normalize its facewise
    gradient, solve the resulting divergence load, and shift the potential minimum
    to zero in the same full degree-zero cochain space.

    ## Visualization
    Plot the computed distance cochain on the source sphere.

    ## Evaluation
    Compare with analytic great-circle distance and independently measure the
    facewise eikonal defect `abs(|grad d| - 1)`.

    ## Interpretation
    This is a mesh-dependent heat-method approximation, not exact polyhedral
    shortest-path distance. Solver residuals certify the two linear equations;
    they do not certify distance accuracy.
    """)
    return


@app.cell
def _(TriangleSurface, np):
    def heat_method_distance(geometry, source_indices):
        if (
            source_indices.ndim != 1
            or source_indices.size == 0
            or source_indices.dtype.kind not in "iu"
            or np.unique(source_indices).size != source_indices.size
            or np.any(source_indices < 0)
            or np.any(source_indices >= geometry.complex.vertex_count)
        ):
            raise ValueError("sources must be nonempty unique vertex indices")

        metric = geometry.positive_metric()
        surface = TriangleSurface.admit(geometry)
        space = geometry.complex.binary64_cochain_space(0)
        masses = metric.hodge_coefficients_numpy_copy(0)
        initial = np.zeros(geometry.complex.vertex_count, dtype=np.float64)
        initial[source_indices] = 1.0 / (source_indices.size * masses[source_indices])

        edge_lengths = geometry.primal_measures_numpy_copy(1)
        length_scale = np.max(edge_lengths)
        mean_edge_length = length_scale * np.mean(edge_lengths / length_scale)
        time_step = mean_edge_length * mean_edge_length
        if not np.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("mean edge length does not define a positive finite time")

        heat_problem = metric.heat_evolution(space.admit_numpy(initial), time_step)
        heat_prepared = heat_problem.prepare()
        heat = heat_prepared.solve(
            heat_problem, heat_prepared.workspace_for(heat_problem)
        )

        inward_gradient = surface.gradient(heat.value).normalized()
        load = surface.divergence(inward_gradient)
        poisson_problem = metric.mean_zero_poisson_load(load)
        poisson_prepared = poisson_problem.prepare()
        poisson = poisson_prepared.solve(
            poisson_problem, poisson_prepared.workspace_for(poisson_problem)
        )

        potential = poisson.potential.coefficients_numpy_copy()
        coefficients = potential - np.min(potential)
        distance = space.admit_numpy(coefficients)
        gradient_lengths = np.linalg.norm(
            surface.gradient(distance).vectors_numpy_copy(), axis=1
        )
        evidence = {
            "time_step": float(time_step),
            "heat_residual_bound": heat.residual_bound,
            "heat_mass_residual_bound": heat.mass_residual_bound,
            "poisson_residual_bound": poisson.residual_bound,
            "poisson_gauge_bound": poisson.gauge_bound,
            "mean_eikonal_error": float(np.mean(np.abs(gradient_lengths - 1.0))),
        }
        return distance, evidence

    return (heat_method_distance,)


@app.cell
def _(heat_method_distance, icosphere, np):
    source = 0
    _, geometry = icosphere(2, 1.0)
    distance, solve_evidence = heat_method_distance(
        geometry, np.array([source], dtype=np.int64)
    )
    coefficients = distance.coefficients_numpy_copy()
    positions = geometry.positions_numpy_copy()
    unit_positions = positions / np.linalg.norm(positions, axis=1)[:, None]
    analytic = np.arccos(np.clip(unit_positions @ unit_positions[source], -1.0, 1.0))
    error = np.abs(coefficients - analytic)
    maximum_spherical_error = float(np.max(error))
    mean_spherical_error = float(np.mean(error))
    if int(np.argmin(coefficients)) != source:
        raise RuntimeError("the selected source is not the distance minimum")
    if maximum_spherical_error > 0.12:
        raise RuntimeError("heat-method error exceeds the sphere-study bound")
    if solve_evidence["mean_eikonal_error"] > 0.04:
        raise RuntimeError("heat-method eikonal defect exceeds the study bound")
    distance_evidence = solve_evidence | {
        "maximum_spherical_error": maximum_spherical_error,
        "mean_spherical_error": mean_spherical_error,
    }
    return distance, distance_evidence, geometry


@app.cell
def _(distance, distance_evidence, geometry, mo, plot_form):
    mo.vstack(
        [
            mo.md(f"`{distance_evidence}`"),
            plot_form(geometry, distance, title="Heat-method distance"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
