# PolyGeo mathematical studies

These executable [Marimo](https://marimo.io/) notebooks connect one mathematical
question to the direct PolyGeo API, a real computation, a visualization where
useful, and an independently calculated check.

| Study | Learning outcome |
|---|---|
| `curvature.py` | Gaussian curvature on a sphere and torus |
| `poisson.py` | Compatible mean-zero Poisson solution |
| `least_squares_conformal_map.py` | Two-anchor LSCM with rank, residual, and orientation evidence |
| `heat_method_distance.py` | Heat-method distance with analytic sphere evidence |
| `homology.py` | Exact torus homology, Stokes pairing, and cup intersection |
| `hodge_decomposition.py` | Reconstruction from exact, coexact, and harmonic parts |
| `connection_and_holonomy.py` | Local versus global holonomy obstruction |

Launch a study from the repository root:

```bash
uv run --extra mesh --extra plot marimo edit examples/curvature.py
```

`examples/support/meshes.py` owns the deterministic meshes shared by the studies.
The canonical check and export matrix lives in the verification workflow.
