#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"

python scripts/build_paper_dataset.py
python scripts/plot_fig1_depth_supervision_matrix.py
python scripts/plot_fig2_spurious_variation.py

if [ "$(python - <<'PY'
import csv
for path in ["figures/data/paper/variance_decomposition.csv"]:
    print(sum(1 for _ in csv.DictReader(open(path))) if __import__("pathlib").Path(path).exists() else 0)
PY
)" != "0" ]; then
    python scripts/plot_fig3_variance_decomposition.py
else
    echo "Skipping Fig 3: no variance_decomposition.csv rows."
fi

if [ "$(python - <<'PY'
import csv
path = "figures/data/paper/oracle_fits.csv"
print(sum(1 for _ in csv.DictReader(open(path))) if __import__("pathlib").Path(path).exists() else 0)
PY
)" != "0" ]; then
    python scripts/plot_fig4_synthetic_control.py
else
    echo "Skipping Fig 4: no oracle_fits.csv rows."
fi

python scripts/plot_fig5_reach_vs_depth.py

if [ -s figures/data/paper/psi_safety_probe.csv ]; then
    python scripts/plot_fig6_psi_safety.py
else
    mkdir -p figures/notes
    cat > figures/notes/fig6_null.md <<'EOF'
Fig 6 omitted: figures/data/paper/psi_safety_probe.csv was not present.
EOF
    echo "Skipping Fig 6: optional psi-safety probe data missing."
fi

python scripts/build_paper_tables.py

figure_count="$(find figures -maxdepth 1 -name 'fig*.pdf' -type f | wc -l | tr -d ' ')"
table_count="$(find figures/tables/paper -name '*.tex' -type f | wc -l | tr -d ' ')"
echo "Generated ${figure_count} figures, ${table_count} tables. Outputs in figures/ and figures/tables/paper/."
