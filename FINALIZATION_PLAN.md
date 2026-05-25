# SR-CPO / Constrained-Contrastive-RL — Project Closure Plan

Status as of 2026-05-25. Source of truth: `/Users/cyamac/Documents/New project 2`
(GitHub `sb322/sr-cpo-safety-gym`), Wulver `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym`.
The old `constrained-crl` tree is v1/v2 audit code and is NOT current.

---

## 1. Is one final run necessary?

**The central claim and the mechanism need no further runs.** What is already
complete and robust:

- **Dense matrix (Fig 1 / Table 1), 4 depths × 3 seeds, sparse-PID:** complete.
  `corr_qc_true_cost` mean −0.008 (range −0.043…+0.017), `actor_true_cost_percentile`
  ≈ 0.502 (random w.r.t. true safety), `actor_qc_percentile_cf` ≈ 0.13 (actor
  exploits Q_c's spurious low-cost preference). Depth 4→32 changes nothing.
- **PID-source confound controlled:** dense depth-8 *active*-PID sanity (job 1024474)
  reproduces the same flat/spurious pattern, so the finding is not an artifact of
  the sparse PID source.
- **Mechanism — label geometry:** within-state vs between-state cost variance is
  strongly **horizon-dependent**: median within/between ≈ 12 at H=0 (immediate cost
  *is* action-sensitive) but collapses to ~8.8e-4 at H=20 and ~0 at H=50. The
  discounted Q_c integrates over exactly the horizons where the action signal has
  vanished.
- **Capacity is not the bottleneck:** oracle fits give median Spearman +0.986 on
  synthetic action-conditional labels vs −0.16 (dense_h1) / +0.20 (dense_h50) on
  real labels. Same architecture, different label.

**The one defensible reason to run:** the **sparse contrast column is incomplete.**
Q_c action-sensitivity diagnostics exist only at depth 8 (3 seeds); depths 4/16/32
are NaN, and sparse-d32 is a single dead cell (seed 0 only, blank reach/cost). The
sparse reach/cost numbers were also parsed from heterogeneous prior outputs rather
than one trusted probed job.

**Recommendation: do exactly ONE confirmation run** — a probed sparse depth sweep —
to make the sparse column complete and single-provenance. This is the
"sanity/confirmation run that strengthens the final claim," not a new experiment.
It upgrades the paper sentence from *"sparse Q_c is also action-flat where measured
(d=8)"* to *"neither sparse nor dense supervision yields an action-sensitive Q_c at
any depth 4–32."* Cost ≈ the dense sweep already run (~12 array tasks, ≤6 h each).

**Acceptable no-run fallback** (if compute/time is tight): present d=8 as the
representative sparse diagnostic, **cap the sparse matrix at depth 16**, and drop
the dead sparse-d32 cell with a one-line footnote. Defensible, but weaker and
invites a "why only one depth for sparse?" referee question.

---

## 2 & 3. The exact run + commands

Script committed at `slurm/sparse_depth_sweep_probed.sh` (clone of
`depth_dense_sweep.sh`; only `COST_MODE=sparse`, `PID_COST_SOURCE=active`, probes on).
Array 0–11 = depths {4,8,16,32} × seeds {0,1,2}.

### Launch (on Wulver)
```bash
cd /mmfs1/home/sb3222/projects/sr-cpo-safety-gym
git fetch origin
git checkout chore/depth-dense-sweep        # or paper/figures-and-tables (superset)
git pull --ff-only
# bring the new script over if it is not already on this branch:
#   git checkout paper/figures-and-tables -- slurm/sparse_depth_sweep_probed.sh
sbatch slurm/sparse_depth_sweep_probed.sh
squeue -u sb3222
```

### Sanity-check ONE task before trusting the array
```bash
# tail the depth-8 seed-0 task (array index 3) once it starts
ls -t safe_depth_sparse.*_3.out | head -1 | xargs tail -n 40
# confirm the probe block printed non-empty corr_qc_true_cost / actor_qc_percentile_cf
grep -E "corr_qc_true_cost|actor_qc_percentile_cf|true_action_sparse_cost_spread" \
  $(ls -t safe_depth_sparse.*_3.out | head -1) | tail
```

### Copy results back to the Mac repo
```bash
# FROM THE MAC (paths on the Mac shown; adjust if your scp alias differs)
cd "/Users/cyamac/Documents/New project 2"
scp 'sb3222@wulver:/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_depth_sparse.*.out' .
# (optional) keep the err logs for the record
scp 'sb3222@wulver:/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_depth_sparse.*.err' .
```

### Rebuild datasets + figures + tables (on the Mac)
```bash
cd "/Users/cyamac/Documents/New project 2"
bash scripts/build_paper.sh
# verify the sparse column is now populated at every depth:
python3 - <<'PY'
import csv
for r in csv.DictReader(open("figures/data/paper/cells.csv")):
    if r["mode"]=="sparse":
        print(r["depth"], r["seed"], r["pid_source"], "corr=", r["corr_qc_true_cost"] or "NaN")
PY
```
Expect non-`NaN` `corr_qc_true_cost` at depths 4/8/16/32. The builder dedups by
`(mode,pid_source,depth,seed)` and prefers probed cells, so the new logs overwrite
the NaN sparse cells automatically — no manual CSV editing.

---

## 4. Finalization checklist (branches, figures, tables)

**Branch / merge order** (figures branch is a strict superset of the sweep branch):
1. Merge `chore/depth-dense-sweep` → into `paper/figures-and-tables` is unnecessary
   (superset already). Land `slurm/sparse_depth_sweep_probed.sh` on
   `chore/depth-dense-sweep`, then fast-forward/rebase `paper/figures-and-tables`.
2. After the confirmation run + rebuild, commit the regenerated
   `figures/`, `figures/data/paper/`, `figures/tables/paper/` on
   `paper/figures-and-tables`.
3. Open one PR: `chore/depth-dense-sweep` → `main`, then `paper/figures-and-tables`
   → `main`. Keep `docs/project-page-main` separate; merge it to `main` last (or
   keep it as the Pages branch).
4. Tag the result commit (e.g. `v1.0-paper`) so the camera-ready is reproducible.

**Regenerate everything:** `bash scripts/build_paper.sh` (runs
`build_paper_dataset.py` → fig1–fig5 → `build_paper_tables.py`).

**Tables to use:** Table 1 (full matrix), Table 2 (aggregate), Table 3 (variance +
oracle). All three are paper-grade after the sparse column fills in.

---

## 5. Figure guidance (paper-ready vs redesign; main vs appendix)

| Figure | Verdict | Placement |
|---|---|---|
| **Fig 1** depth × supervision matrix | Paper-ready **after** sparse column fills in | **Main** (centerpiece) |
| **Fig 2** spurious variation (qc_cf vs true-cost percentile, corr≈0) | Paper-ready | **Main** |
| **Fig 3** variance decomposition | **Redesign before submission** | **Main** (after fix) |
| **Fig 4** synthetic control / oracle | Paper-ready, needs light cleaning | **Main** |
| **Fig 5** reach vs depth | Weak as drawn; reframe or demote | **Appendix** |

**Fig 3 — mandatory redesign.** The CSV has duplicate rows and the within/between
ratio *inverts* with horizon (≈12 at H=0, →0 by H=50). A single bar will look wrong
to a careful reviewer. Plot the ratio (or within and between separately) **as a
function of horizon on a log y-axis**, dedup the rows, and annotate that the
discounted Q_c integrates over the high-horizon regime where the action signal has
collapsed. This is the figure that *makes* the mechanism; it must show the horizon
axis.

**Fig 4 — light cleaning.** Drop the tiny-`n_states` bootstrap rows (n = 5/8/16);
keep n ≥ 128. Show synthetic (~0.99) vs real dense/sparse as the headline contrast.

**Fig 5 — reframe or appendix.** In the data, `reach` is the place where dense
(~0.04) and sparse (~0.97) *diverge*, but it does **not** vary cleanly with depth
within either mode. So a "reach increases with depth" framing is not supported here.
Use it instead to make the **decoupling** point: depth/supervision move reach and
safety-critic quality independently — high-reach sparse is maximally *unsafe*
(cost-return ~95–128), safe-ish dense barely reaches. If you cannot make that read
cleanly in one panel, move it to the appendix.

**Main paper:** Fig 1, Fig 2, Fig 3 (redesigned), Fig 4. **Appendix:** Fig 5,
per-seed tables, the d=8 active-PID confound control, oracle small-n rows.

---

## 6. Conclusion paragraph (result + mechanism + implication)

> We set out to test whether network depth and denser cost supervision restore an
> action-sensitive safety critic in contrastive constrained RL. They do not. Across
> depths 4–32 and three seeds, the cost critic Q_c never acquires usable
> action-conditional structure: its correlation with realized per-action cost is
> statistically zero (mean −0.01), the actions the policy actually selects sit at the
> ~50th percentile of true cost (no better than chance), and dense proximity
> supervision leaves this unchanged while merely inducing *spurious* action variation
> that the actor exploits. The mechanism is label geometry, not capacity: the same
> architecture recovers synthetic action-conditional labels almost perfectly
> (Spearman ≈ 0.99), but in the real task the within-state, action-conditional share
> of cost variance — large for the immediate cost — is swamped by between-state
> variance once it is propagated through the discounted TD target, the only signal
> the critic ever sees. Constraint enforcement that relies on action-ranking a
> learned Q_c is therefore built on a quantity the standard objective does not make
> learnable, regardless of depth or supervision density. Closing this gap requires
> changing the *label* the cost critic is trained on — supplying an explicit
> action-conditional safety signal — rather than scaling the function approximator.

(Use the no-hedge version above if the confirmation run lands; if you take the
no-run fallback, change "Across depths 4–32 and three seeds" for the sparse clause
to "at the depths measured" and footnote the sparse coverage.)

---

## 7. Claude Code prompt (paste into the Mac repo)

> You are in `/Users/cyamac/Documents/New project 2` on branch
> `paper/figures-and-tables` (GitHub `sb322/sr-cpo-safety-gym`). Do NOT touch
> `/Users/cyamac/Experiments/constrained-crl` — it is stale v1/v2 audit code.
> Tasks, read-only unless told otherwise:
> 1. Confirm `slurm/sparse_depth_sweep_probed.sh` differs from
>    `slurm/depth_dense_sweep.sh` only in `COST_MODE` (sparse vs dense_proximity),
>    `PID_COST_SOURCE` (active vs sparse), and the job/label names — no source
>    changes. Print the diff.
> 2. Confirm `scripts/build_paper_dataset.py` will ingest `safe_depth_sparse.*.out`
>    (check it matches a `DEFAULT_LOG_GLOBS` entry) and that `_cell_score` prefers a
>    cell with `corr_qc_true_cost` populated over one without.
> 3. Re-run `bash scripts/build_paper.sh` and report: how many sparse cells in
>    `figures/data/paper/cells.csv` now have non-NaN `corr_qc_true_cost`, and whether
>    the dead sparse-d32 cell is resolved.
> 4. In `scripts/plot_fig3_variance_decomposition.py`, report whether the script
>    dedups input rows and whether it plots the within/between ratio against horizon.
>    Do not edit yet — just report what it does.
