# PolyGeo mathematical studies

These files are executable [Marimo](https://marimo.io/) studies. Each notebook introduces the mathematics, maps symbols to the public PolyGeo API, performs a real computation, renders it through `polygeo.plotting`, and independently evaluates the relevant laws.

Run a study:

```bash
uv run --extra mesh --extra plot marimo edit examples/curvature.py
```

Check all studies:

```bash
uv run --extra mesh --extra plot marimo check --strict \
  examples/curvature.py \
  examples/poisson.py \
  examples/harmonic_extension.py \
  examples/mean_curvature_flow.py \
  examples/homology_and_periods.py \
  examples/hodge_decomposition.py \
  examples/connection_and_holonomy.py
```

Export one study into a disposable location:

```bash
uv run --extra mesh --extra plot marimo export html examples/curvature.py --no-include-code -o <task-temp>/curvature.html
```

The explicit check list is intentional: files in `examples/support/` are ordinary Python
mesh fixtures, not Marimo notebooks.

The studies use a single-column layout. Figures and images are always vertically ordered; they are never placed in `mo.hstack` or side-by-side subplots.

`conformal_parameterization.py` is intentionally absent until the corresponding public algorithm and evidence contracts land.
