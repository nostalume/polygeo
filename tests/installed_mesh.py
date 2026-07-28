from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from polygeo import MeshError, load_surface


mode = sys.argv[1]
if mode not in {"without-extra", "with-extra"}:
    raise SystemExit("expected without-extra or with-extra")

with TemporaryDirectory() as directory:
    path = Path(directory) / "triangle.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    if mode == "without-extra":
        try:
            load_surface(path)
        except MeshError as error:
            assert str(error) == (
                "mesh input requires the optional polygeo[mesh] dependency"
            )
        else:
            raise AssertionError(
                "mesh loading unexpectedly succeeded without the extra"
            )
    else:
        geometry = load_surface(path)
        assert geometry.complex.dimension == 2
        assert geometry.complex.simplex_count(2) == 1
        assert geometry.positions.shape == (3, 3)
