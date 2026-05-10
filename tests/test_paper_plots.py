from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from scripts.paper_utils import CELL_COLUMNS


ROOT = Path(__file__).resolve().parents[1]


def test_plot_fig1_runs_on_canonical_cells(tmp_path: Path) -> None:
    cells = tmp_path / "cells.csv"
    with cells.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CELL_COLUMNS)
        writer.writeheader()
        for mode in ("sparse", "dense"):
            for seed in ("0", "1", "2"):
                writer.writerow(
                    {
                        "mode": mode,
                        "pid_source": "sparse",
                        "depth": "8",
                        "seed": seed,
                        "hard": "0.05",
                        "reach": "0.1",
                        "cost_ret": "10",
                    }
                )
    output = tmp_path / "fig1.pdf"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plot_fig1_depth_supervision_matrix.py"),
            "--cells",
            str(cells),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    assert output.exists()
    assert output.stat().st_size > 1000
