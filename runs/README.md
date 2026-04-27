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
- `cost_limit=0.0001` is now the calibrated production default for
  `TrainConfig`, `slurm/smoke.sh`, and `slurm/full.sh`.

## 2026-04-27 - Job 1010605 - Calibrated 3-Seed Baseline

- Script: `slurm/baseline_cl0001_3seed.sh`
- Environment: Wulver A100, JAX GPU backend, safe-learning GoToGoal adapter
- Repo state: includes `506b7f3 fix(train): promote calibrated cost limit default`
- Result: all three array tasks `COMPLETED 0:0`
- Cost limit: `0.0001`
- Seeds: `0`, `1`, `2`
- Output files on Wulver:
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_base_cl0001.1010605_0.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_base_cl0001.1010605_1.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_base_cl0001.1010605_2.out`

Tail diagnostics:

| Seed | Lambda tail | J_c tail | Cost tail | Hard violation tail | Critic acc tail |
|---|---:|---:|---:|---:|---:|
| 0 | `0.0095` | `0.0005` | `0.0462` | `0.0392` | `0.234` |
| 1 | `0.0129` | `0.0001` | `0.0123` | `0.0060` | `0.120` |
| 2 | `0.0000` | `0.0001` | `0.0066` | `0.0078` | `0.030` |

All three seeds kept NaN probes and parameter-NaN probes at zero. Gradient
norms stayed in the healthy single-digit to low-double-digit range. The
calibrated limit activates the dual when the critic estimate is above budget
or near the boundary, while seed 2 stays inactive because both rollout cost and
estimated cost are already low. This validates `0.0001` as the first production
constraint setting. The next experiment should sweep PID gains at this fixed
budget, because the remaining variation is controller response, not plumbing.
