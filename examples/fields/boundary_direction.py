import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from polygeo.geometry import Geometry, TriangleSurface
    from polygeo.plot import direction as plot_direction_field
    from examples.support.meshes import disk

    return Geometry, TriangleSurface, disk, mo, np, plot_direction_field


@app.cell
def _(mo):
    mo.md(r"""
    # Boundary-aligned order-four direction field

    ## Mathematical question
    How does a symmetric direction field align to a disk boundary while retaining
    exact singularity and boundary-winding evidence?

    ## From mathematics to PolyGeo
    The field stores order-four power directions. Exact vertex charges and
    boundary turns describe its topology. Visualization retains every equivalent
    branch instead of selecting one representative.

    ## Computation
    Lift the existing planar disk into three-dimensional ambient space and solve
    one boundary-aligned field with zero boundary-angle offset.

    ## Visualization
    Plot all four equivalent directions at every face center. Each cross is one
    order-four glyph; none of its branches is privileged.

    ## Evaluation
    Check the four unit charges, disk boundary turn, and quantization certificate.

    ## Interpretation
    The crosses are not four separate vector fields. Each cross displays one
    order-four power field whose exact singularity evidence is branch-independent.
    """)
    return


@app.cell
def _(Geometry, TriangleSurface, disk, np):
    domain, planar = disk(3, 16)
    planar_positions = planar.positions_numpy_copy()
    geometry = Geometry.from_positions(
        domain,
        np.column_stack(
            (planar_positions, np.zeros(len(planar_positions), dtype=np.float64))
        ),
    )
    surface = TriangleSurface.admit(geometry)
    field = surface.boundary_direction(4, geometry.metric(), 0.0)
    singularities = field.singularities()
    charge_indices, charge_values = singularities.charges.to_python_copy()
    boundary_turns = singularities.boundary_turns_python_copy()
    if tuple(charge_values) != (1, 1, 1, 1) or boundary_turns != (0,):
        raise RuntimeError("unexpected disk direction-field topology")
    if singularities.maximum_quantization_residual > singularities.residual_limit:
        raise RuntimeError("direction-field quantization is not certified")
    direction_field_evidence = {
        "symmetry_order": field.symmetry_order,
        "charge_indices": charge_indices,
        "charges": charge_values,
        "charge_sum": sum(charge_values),
        "boundary_turns": boundary_turns,
        "crossing_error": field.connection.crossing_error,
        "maximum_quantization_residual": singularities.maximum_quantization_residual,
        "residual_limit": singularities.residual_limit,
    }
    return direction_field_evidence, field


@app.cell
def _(direction_field_evidence, field, mo, plot_direction_field):
    mo.vstack(
        [
            mo.md(f"`{direction_field_evidence}`"),
            plot_direction_field(
                field,
                scale=0.18,
                title="Order-four boundary direction field",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
