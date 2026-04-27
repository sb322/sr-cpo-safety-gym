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

Follow-up:

- `slurm/pid_gain_sweep.sh` runs seed `0` at fixed `cost_limit=0.0001` over
  `(Kp, Ki, Kd)` settings `(2.5, 0.05, 0.0)`, `(5.0, 0.1, 0.0)`, and
  `(10.0, 0.2, 0.0)` to compare controller response.

## 2026-04-27 - Job 1010634 - PID Gain Sweep

- Script: `slurm/pid_gain_sweep.sh`
- Environment: Wulver A100, JAX GPU backend, safe-learning GoToGoal adapter
- Repo state: includes `19e1920 feat(slurm): add pid gain sweep`
- Result: all three array tasks `COMPLETED 0:0`
- Seed: `0`
- Cost limit: `0.0001`
- Output files on Wulver:
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_pid.1010634_0.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_pid.1010634_1.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_pid.1010634_2.out`

Tail diagnostics:

| Label | Kp | Ki | Lambda tail | J_c tail | Cost tail | Hard violation tail | Critic acc tail |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pid_soft` | `2.5` | `0.05` | `0.0051` | `0.0004` | `0.0379` | `0.0403` | `0.259` |
| `pid_base` | `5.0` | `0.1` | `0.0097` | `0.0003` | `0.0299` | `0.0256` | `0.221` |
| `pid_strong` | `10.0` | `0.2` | `0.0196` | `0.0004` | `0.0377` | `0.0320` | `0.230` |

All three settings kept NaN probes and parameter-NaN probes at zero. Gradient
norms stayed healthy. The base PID setting gives the best tail cost and hard
violation tradeoff in this sweep. Stronger gains increase `lambda_tilde` but do
not improve the constraint metrics, while softer gains under-react. Keep
`Kp=5.0`, `Ki=0.1`, `Kd=0.0` as the production default for the next experiments.

Follow-up:

- `slurm/depth_sweep_residual.sh` runs seed `0` with calibrated safety settings
  and residual towers over `num_blocks=4`, `8`, and `16` to start the
  depth-scaling experiment.

## 2026-04-27 - Job 1010681 - Residual Depth Sweep

- Script: `slurm/depth_sweep_residual.sh`
- Environment: Wulver A100, JAX GPU backend, safe-learning GoToGoal adapter
- Repo state: includes `79a452f feat(slurm): add residual depth sweep`
- Result: all three array tasks `COMPLETED 0:0`
- Seed: `0`
- Cost limit: `0.0001`
- PID gains: `Kp=5.0`, `Ki=0.1`, `Kd=0.0`
- Residual towers: enabled
- Output files on Wulver:
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_depth.1010681_0.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_depth.1010681_1.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_depth.1010681_2.out`

Tail diagnostics:

| Depth | Critic loss | Critic acc | Actor loss | Lambda tail | J_c tail | Cost tail | Hard violation tail | Tail epoch time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `5.5292` | `0.025` | `3.2937` | `0.0000` | `0.0001` | `0.0053` | `0.0031` | `6.3s` |
| 8 | `6.9267` | `0.001` | `5.5964` | `0.0000` | `0.0000` | `0.0004` | `0.0000` | `5.9s` |
| 16 | `5.9003` | `0.006` | `4.2674` | `0.0000` | `0.0001` | `0.0047` | `0.0036` | `6.3s` |

All depths kept NaN probes and parameter-NaN probes at zero. Wallclock did not
meaningfully increase up to 16 residual blocks at this run size. Safety metrics
were very low for all residual depths, so the PID stayed inactive. However, the
contrastive reward critic underperformed badly relative to the plain 4-block
baseline: residual depth 8 was near random InfoNCE performance (`loss ~= log
1024`, `accuracy ~= 0.001`), and residual depth 16 only partially recovered.

Interpretation:

The residual path is numerically stable and cheap enough, but it is not learning
the contrastive critic under the current optimizer/update schedule. The next
experiment should keep residual depth fixed at 8 and sweep the critic learning
schedule, for example more SGD steps or a higher critic learning rate, before
claiming depth helps or hurts the algorithm itself.

Follow-up:

- `slurm/depth8_sgd_sweep.sh` keeps residual `num_blocks=8` fixed and sweeps
  `sgd_steps=1`, `2`, and `4` to test whether extra optimizer updates rescue
  the contrastive critic.

## 2026-04-27 - Job 1010750 - Depth8 SGD-Step Sweep

- Script: `slurm/depth8_sgd_sweep.sh`
- Environment: Wulver A100, JAX GPU backend, safe-learning GoToGoal adapter
- Repo state: includes `0acaf0b feat(slurm): add depth8 sgd sweep`
- Result: all three array tasks `COMPLETED 0:0`
- Seed: `0`
- Cost limit: `0.0001`
- PID gains: `Kp=5.0`, `Ki=0.1`, `Kd=0.0`
- Residual towers: enabled
- Depth: `num_blocks=8`
- Output files on Wulver:
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_depth8_sgd.1010750_0.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_depth8_sgd.1010750_1.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_depth8_sgd.1010750_2.out`

Tail diagnostics:

| SGD steps | Critic loss | Critic acc | Actor loss | Lambda tail | J_c tail | Cost tail | Hard violation tail | Tail epoch time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `6.3148` | `0.003` | `4.6529` | `0.0000` | `0.0000` | `0.0004` | `0.0006` | `6.0s` |
| 2 | `4.9419` | `0.067` | `2.3297` | `0.0000` | `0.0001` | `0.0081` | `0.0043` | `6.2s` |
| 4 | `4.4886` | `0.210` | `1.4424` | `0.0000` | `0.0000` | `0.0029` | `0.0091` | `6.2s` |

All three settings kept NaN probes and parameter-NaN probes at zero. Increasing
the update budget rescued depth8 residual critic learning: `sgd_steps=4`
improved the contrastive critic from near-random accuracy to roughly the same
range as the plain 4-block baseline, without destabilizing the actor or cost
critic. The inactive dual in the `sgd4` run is expected because the rollout is
already safe and the critic-based `J_c` estimate is at or below budget.

Follow-up:

- `slurm/depth_sgd4_residual.sh` reruns the residual depth sweep with
  `sgd_steps=4` fixed over `num_blocks=4`, `8`, `16`, and `32`. This tests the
  depth-scaling question after fixing the update-budget bottleneck found here.

## 2026-04-27 - Job 1010892 - Constraint Activation Sweep

- Script: `slurm/constraint_activation_sweep.sh`
- Environment: Wulver A100, JAX GPU backend, safe-learning GoToGoal adapter
- Repo state: includes `2d12a82 feat(slurm): add constraint activation sweep`
- Result: all five array tasks `COMPLETED 0:0`
- Seed: `0`
- Architecture: plain `num_blocks=4`, `use_residual=false`
- Output files on Wulver:
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_constraint.1010892_0.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_constraint.1010892_1.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_constraint.1010892_2.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_constraint.1010892_3.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_constraint.1010892_4.out`

Tail diagnostics:

| Label | Cost limit | Kp | Ki | Lambda tail | PID error tail | Integral tail | Cost tail | Hard violation tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `unconstrained` | `1e-4` | `0.0` | `0.0` | `0.0000` | `2.39e-4` | `8.05e-2` | `0.0357` | `0.0377` |
| `cl1e-4` | `1e-4` | `5.0` | `0.1` | `0.0058` | `2.04e-4` | `4.80e-2` | `0.0301` | `0.0292` |
| `cl5e-5` | `5e-5` | `5.0` | `0.1` | `0.0113` | `3.25e-4` | `9.63e-2` | `0.0378` | `0.0337` |
| `cl1e-5` | `1e-5` | `5.0` | `0.1` | `0.0128` | `4.13e-4` | `1.07e-1` | `0.0428` | `0.0340` |
| `cl0` | `0` | `5.0` | `0.1` | `0.0137` | `3.54e-4` | `1.20e-1` | `0.0361` | `0.0391` |

All settings kept NaN probes and parameter-NaN probes at zero. The diagnostic
settles the first ambiguity: the PID controller is alive. Tighter budgets make
`pid_err`, `S`, `lambda_raw`, and `lambda_tilde` positive in the expected
direction. However, tightening the budget does not produce monotone behavioral
improvement in rollout cost or hard violations. The likely scale issue is in
the actor loss: with `nu_c=1`, `lambda_tilde ~= 0.006-0.014` and
`Q_c ~= 0.03-0.05`, so the actor penalty `lambda_tilde * Q_c / nu_c` is around
`1e-4` to `7e-4`, far below the contrastive reward term scale.

Follow-up:

- `slurm/constraint_pressure_sweep.sh` fixes `cost_limit=0.0001` and base PID
  gains, then sweeps `nu_c` over `1.0`, `0.01`, `0.001`, `0.0003`, and
  `0.0001`. It also logs `lambda_qc_actor` so we can directly see when the
  Lagrangian actor term becomes large enough to affect policy behavior.

## 2026-04-27 - Job 1010958 - Constraint Pressure Sweep

- Script: `slurm/constraint_pressure_sweep.sh`
- Environment: Wulver A100, JAX GPU backend, safe-learning GoToGoal adapter
- Repo state: includes `d8f8f9d feat(slurm): add constraint pressure sweep`
- Result: all five array tasks `COMPLETED 0:0`
- Seed: `0`
- Cost limit: `0.0001`
- PID gains: `Kp=5.0`, `Ki=0.1`, `Kd=0.0`
- Architecture: plain `num_blocks=4`, `use_residual=false`
- Output files on Wulver:
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_pressure.1010958_0.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_pressure.1010958_1.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_pressure.1010958_2.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_pressure.1010958_3.out`
  - `/mmfs1/home/sb3222/projects/sr-cpo-safety-gym/safe_pressure.1010958_4.out`

Tail diagnostics:

| Label | nu_c | lambda_Qc actor | Lambda tail | Cost tail | Hard violation tail | Actor loss |
|---|---:|---:|---:|---:|---:|---:|
| `nuc1` | `1.0` | `2.75e-4` | `0.0082` | `0.0325` | `0.0311` | `1.7492` |
| `nuc1e-2` | `0.01` | `4.09e-2` | `0.0101` | `0.0389` | `0.0280` | `1.7852` |
| `nuc1e-3` | `0.001` | `1.82e-1` | `0.0080` | `0.0215` | `0.0203` | `1.9042` |
| `nuc3e-4` | `0.0003` | `1.94e-1` | `0.0047` | `0.0116` | `0.0123` | `2.4952` |
| `nuc1e-4` | `0.0001` | `1.12e+0` | `0.0053` | `0.0223` | `0.0267` | `3.2589` |

All settings kept NaN probes and parameter-NaN probes at zero. The experiment
confirms the actor-loss preconditioning diagnosis: `nu_c=1.0` makes the
constraint term too small to matter, while `nu_c=0.001` and `nu_c=0.0003`
reduce rollout cost and hard violations. `nu_c=0.0001` is too strong for this
setup: the actor loss rises sharply and safety is worse than at `0.0003`.

Follow-up:

- `slurm/constraint_effect_3seed.sh` runs a paired three-seed comparison:
  unconstrained PID gains `(0, 0, 0)` versus constrained base PID gains
  `(5.0, 0.1, 0.0)`, both at `cost_limit=0.0001` and `nu_c=0.0003`. This is
  the first clean CMDP effect-size experiment.
