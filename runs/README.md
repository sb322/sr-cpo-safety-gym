# Run Ledger

This file records Wulver runs that are useful as reproducible baselines or
debugging waypoints. Raw SLURM logs live on Wulver in the project directory
unless explicitly copied elsewhere.

## 2026-04-26 - Job 1009634 - First Clean Full Real-Env Run

- Script: `slurm/full.sh`
- Environment: Wulver A100, JAX GPU backend, safe-learning GoToGoal adapter
- Repo state: includes `74d7f3e fix(slurm): use full observation as real-env goal`
- Result: `COMPLETED 0:0`
- Runtime: `00:10:01`
- Output files on Wulver:
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_full.1009634.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_full.1009634.err`

Key diagnostics from the tail:

- Completed `50/50` epochs and `5,586,944` env steps.
- NaN probes remained zero: `nan[obs_c=0 sa_c=0 g_c=0 lg_c=0 sa_a=0 g_a=0 act_a=0 f_a=0]`.
- Param NaN probes remained zero: `params[c=0 a=0 cc=0]`.
- Goal encoder min norms stayed nonzero after the `--goal-dim 55` fix:
  `psi_min_c ~= 4.8-5.9`, `psi_min_a ~= 4.8-5.8` in the tail.
- Critic gradient norms were healthy in the tail: `grad[c] ~= 6-10`.
- Critic improved over the last 10 epochs: `c_loss 4.9076 -> 4.6856`,
  `accuracy 0.154 -> 0.233`.
- Actor loss decreased in the tail: `a_loss 2.0274 -> 1.6270`.
- Constraint metrics remained stable with decreasing hard violations:
  `hard_viol 0.0541 -> 0.0242`, `cost 0.0445 -> 0.0328`.
- `alpha_clip=1.00` remained binding as expected.
- `lambda_tilde=0.0000` stayed inactive because estimated `J_c` was far below
  `cost_limit=0.05` (`J_c ~= 0.0003-0.0005` in the tail).

Interpretation:

This is the first clean end-to-end Wulver production sanity run on the real
safe-learning GoToGoal environment. It validates the GPU path, real-env rollout
collection, full-observation hindsight goal relabeling, probe wiring, alpha cap,
gradient clipping, and SLURM static checks. Next experiments should vary the
constraint scale, `cost_limit`, or PID gains because the dual remained inactive
under the current critic-based `J_c` scale.

Follow-up:

- `slurm/diagnostic_cost_limit.sh` launches a three-point `cost_limit` array
  sweep over `0.001`, `0.0005`, and `0.0001` using the same real-env settings.
- `slurm/baseline_cl0001_3seed.sh` promotes the first active dual setting,
  `cost_limit=0.0001`, to a three-seed baseline over seeds `0`, `1`, and `2`.
