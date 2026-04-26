# Constrained CRL → Safety Gym: Implementation Handoff

This document specifies an algorithm (**SR-CPO** — Safety-Regularized Contrastive Policy Optimization) and asks you to implement it in a new repository targeting Safety Gym environments. Read it end-to-end before writing code. The mathematical specification is the source of truth; the reference implementation in `constrained-crl/train.py` is a JAX/Brax instantiation on goal-reaching MJX maze environments and is provided as a sanity reference, not as code to copy.

---

## 1. The constrained MDP

The agent maximizes goal-reaching reward subject to a safety cost budget:

$$
\max_\pi\; J_r(\pi) \quad\text{s.t.}\quad J_c(\pi) \;\le\; d, \qquad \pi(\cdot\mid s, g)\in\Delta(\mathcal A)
$$

with discounted occupation measures

$$
J_r(\pi) = \mathbb E\!\Big[\textstyle\sum_t \gamma^t\, r(s_t,a_t,g)\Big],\qquad
J_c(\pi) = \mathbb E\!\Big[\textstyle\sum_t \gamma_c^t\, c(s_t)\Big].
$$

The Lagrangian is $\mathcal L(\pi,\lambda) = J_r(\pi) - \lambda(J_c(\pi)-d)$, $\lambda\ge0$.

---

## 2. Components

### 2.1 Contrastive reward critic (InfoNCE with row-L2)

Two encoders $\phi_\theta(s,a)$ and $\psi_\xi(g)$, both $\mathbb R^{D}\to\mathbb R^{K}$ with $K=64$. The energy is the cosine, divided by a fixed temperature $\tau=0.1$:

$$
\hat\phi=\frac{\phi}{\sqrt{\sum_i \phi_i^2 + \varepsilon}},\quad \hat\psi=\frac{\psi}{\sqrt{\sum_i \psi_i^2 + \varepsilon}},\quad \varepsilon=10^{-12}
$$

$$
f(s,a,g) \;=\; \frac{\langle \hat\phi(s,a),\,\hat\psi(g)\rangle}{\tau}\;\in\;[-1/\tau,\;+1/\tau].
$$

> **Critical implementation detail.** `epsilon must be inside the sqrt`. The standard form `x / (||x|| + 1e-8)` is forward-finite but autograd-NaN whenever `sum(x²)` underflows to 0, because the gradient of `sqrt(s)` at `s=0` is `+inf`. Putting `epsilon` outside the sqrt does not protect the backward pass. We confirmed this empirically — see §6.

The critic loss is symmetric InfoNCE on a batch of $N$ (state-action, future-state-derived-goal) pairs from the buffer, with a logsumexp-penalty regularizer $\rho$:

$$
\mathcal L_{\text{critic}} = -\frac1N\sum_i\Big[ f(s_i,a_i,g_i) - \mathrm{logsumexp}_j f(s_i,a_i,g_j)\Big]
\;+\; \rho\cdot\frac1N\sum_i\big(\mathrm{logsumexp}_j f(s_i,a_i,g_j)\big)^2.
$$

Goals $g_j$ are sampled from the **future** of the same trajectory using hindsight (`flatten_crl_fn` in the reference). Negatives come from other rows in the batch.

### 2.2 Cost critic (TD(0))

A separate scalar critic $Q_c^{\,\eta}(s,a,g)$ trained on the safety cost. Bellman target with Polyak target network $Q_c^{\,\bar\eta}$:

$$
y_t \;=\; c(s_t) + \gamma_c\,(1-d_t)\,Q_c^{\,\bar\eta}\!\big(s_{t+1},\,a',\,g\big),\quad a'\sim\pi_\theta(\cdot\mid s_{t+1},g).
$$

$$
\mathcal L_{\text{cost}} = \frac12\,\mathbb E[(Q_c^{\,\eta}(s_t,a_t,g)-y_t)^2].
$$

`cost`, `d_wall`, `hard_violation_indicator` are read from `transition.extras` (env emits them in `state.info`). For Safety Gym, `info["cost"]` is the canonical source.

### 2.3 Stochastic actor (tanh-Gaussian)

$$
\mu_\theta(s,g),\;\log\sigma_\theta(s,g) = \text{Actor}(s,g),\quad \log\sigma\in[-5,2],
$$
$$
x = \mu + \sigma\odot\epsilon,\;\;\epsilon\sim\mathcal N(0,I),\quad a = \tanh(x).
$$

Log-density with the tanh-Jacobian correction:

$$
\log\pi(a\mid s,g) = \sum_i\!\Big[\log\mathcal N(x_i\mid \mu_i,\sigma_i) \;-\; \log\!\big(1-a_i^2 + 10^{-6}\big)\Big].
$$

Use the **numerically stable softplus form** when feasible:

$$
\log(1-\tanh^2 x) \;=\; 2\,\big(\log 2 - x - \mathrm{softplus}(-2x)\big).
$$

The reference uses the naïve `log(1 - a² + 1e-6)` form. We have an open question whether this form contributes to actor-loss divergence — see §6.

### 2.4 Actor loss (preconditioned Lagrangian)

$$
\mathcal L_{\text{actor}} = \mathbb E\!\left[ \alpha\,\log\pi(a\mid s,g) \;-\; \frac{f(s,a,g)}{\nu_f} \;+\; \frac{\tilde\lambda}{\nu_c}\,Q_c^{\,\eta}(s,a,g)\right].
$$

Preconditioners $\nu_f, \nu_c$ are fixed scalars. Reference uses $\nu_f = 1$, $\nu_c = 1$.

### 2.5 SAC entropy temperature

$$
\mathcal L_{\alpha} = -\mathbb E\!\big[\log\alpha \cdot (\log\pi + \mathcal H_{\text{target}})\big],\quad \mathcal H_{\text{target}} = -\dim\mathcal A,\quad \alpha = e^{\log\alpha}.
$$

### 2.6 PID-Lagrangian dual update (Stooke et al. 2020)

$$
\hat J_c \;=\; (1-\gamma)\cdot\frac1M\sum_{m=1}^M V_c\!\big(s_0^{(m)},\,g^{(m)}\big),\quad V_c(s,g) = \mathbb E_{a\sim\pi}[Q_c(s,a,g)],
$$
$$
e_k = \hat J_c - d,\quad S_k = \mathrm{clamp}\!\big(S_{k-1}+e_k,\,\pm S_{\max}\big),\quad S_{\max} = \lambda_{\max}/\max(K_i, 10^{-8}),
$$
$$
\tilde\lambda_k \;=\; \mathrm{clip}\!\Big(K_p e_k + K_i S_k + K_d (e_k - e_{k-1}),\; 0,\, \lambda_{\max}\Big).
$$

Reference values: $K_p=0.1,\;K_i=0.003,\;K_d=0.001,\;\lambda_{\max}=100$. The anti-windup clamp on $S_k$ matters: without it $\tilde\lambda$ lags badly when violations subside.

---

## 3. Architecture

- **Encoders $\phi$, $\psi$**: 4 Wang residual blocks (LayerNorm + Dense+swish + Dense, with skip), width 256, then a final Dense to 64. `lecun_normal` kernel init.
- **Actor**: 4 Wang residual blocks, width 256, output Dense to $2\cdot\dim\mathcal A$ split into $\mu$ and $\log\sigma$. Clip $\log\sigma$ to $[-5, 2]$.
- **Cost critic**: 4 Wang residual blocks, width 256, scalar output. Polyak target with $\tau_{\text{polyak}}=0.005$.

All four networks see goal-conditioned inputs via concatenation `[state, goal]` (or `[state, action, goal]` for $\phi$ and $Q_c$).

---

## 4. Training loop

Per epoch (do this on GPU under `jit`-equivalent in your framework):

1. **Collect**: roll out `unroll_length=62` env-steps with current actor across `num_envs=256` parallel envs. Insert into a uniform replay buffer of trajectories.
2. **Sample**: draw a batch, apply hindsight goal relabeling (future-state goals along the same trajectory).
3. **Update**: `num_update_epochs × num_minibatches = 4 × 32 = 128` gradient steps per training_step. Update order inside one sgd step:
   1. Critic ← `c_grads = ∇ L_critic`. Apply.
   2. Actor ← `a_grads = ∇ L_actor` (uses post-update critic params for $f$).
   3. Alpha ← `al_grads = ∇ L_alpha` (uses post-update actor's `log_prob`).
   4. Cost critic ← `cc_grads = ∇ L_cost`. Apply. Then Polyak target.
   5. Dual: estimate $\hat J_c$ via `_critic_based_dual_estimator(...)`, then update $\tilde\lambda$ via PID.
4. Repeat training_step `steps_per_epoch ≈ 6` times.

Hyperparameters (reference): `total_env_steps=5e6`, `num_epochs=50`, `batch_size=256`, lr `3e-4` for actor/critic/alpha, `gamma=0.99`, `gamma_c=0.99`, `cost_budget_d=0.15`.

---

## 5. Safety Gym mapping

The reference targets Brax MJX `ant_big_maze` with custom cost wiring. For Safety Gym:

- **Environments**: prioritize `SafetyPointGoal1-v0` first (8-dim obs, 2-dim action, low-dim goal). Then `SafetyCarGoal1-v0`, `SafetyPointGoal2-v0` (with hazards).
- **Goal extraction**: Safety Gym's goal info is in `obs` (relative position to goal lidar) + `info["goal_dist"]`. Use the absolute goal position from `env.unwrapped.goal_pos` if available; otherwise the lidar-derived relative-goal vector. Document your choice and stick to it.
- **Cost**: `info["cost"]` is the canonical per-step cost. Plug directly into `transition.extras["cost"]`. Do **not** recompute from geometry — Safety Gym's cost includes hazard-region indicators that aren't trivially recoverable from observations.
- **Hard violation indicator**: `info["cost"] > 0` is a serviceable boolean; expose it as `transition.extras["hard_violation"]` for diagnostic logging.
- **Episode length / truncation**: Safety Gym's default is 1000 steps. Use `gymnasium.wrappers.TimeLimit` truncation flag in `transition.extras["truncation"]`.
- **Vectorization**: Use `gymnasium.vector.AsyncVectorEnv` with `num_envs=64` (Safety Gym is CPU/MuJoCo, so you cannot match Brax's 256-env GPU throughput). Adjust `unroll_length` and `batch_size` so each gradient step still sees a comparable amount of fresh data.

---

## 6. Lessons from the reference implementation — do not re-discover

These were paid for in compute and time. Inherit them.

### 6.1 Autograd-safe row-L2 normalization

Earlier versions used `x / (jnp.linalg.norm(x) + 1e-8)`. This is **forward-finite but autograd-NaN** whenever `sum(x²)` underflows to 0. We diagnosed this through 4 probed smokes (jobs 981715, 983095, 984135, 988003). Fix: put `epsilon` *inside* the sqrt:
```python
x_norm_safe = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + 1e-12)
x_hat = x / x_norm_safe
```

### 6.2 The reference's open question (you do not need to solve this, but be aware)

After fixing 6.1, the actor loss diverges to $\approx -4.84\times10^{5}$ on a 50-epoch run at $d=0.15$ in the **feasible** regime ($\tilde\lambda=0$ throughout). Since $f\in[-10,10]$ under row-L2, the divergence must originate in $\alpha\cdot\overline{\log\pi}$. We instrumented 7 component probes in the reference; the next experiment localizes which sub-term of $\log\pi$ (Gaussian density vs. tanh-Jacobian saturation correction vs. $\log\sigma$) carries the magnitude. **Implications for your port**: emit those component probes from day 1 (see §7 for the list), so when (not if) you see the same divergence on Safety Gym you can localize it without a multi-day forensics chain.

### 6.3 Probing discipline

The reference logs 13 NaN flags + 9 grad/param probes + 7 actor-loss component probes per epoch. None of them cost meaningful wallclock. If you strip them "for cleanliness" you will pay them back in debugging time at 10× cost. The pattern is:

```python
sa_has_nan_c   = jnp.any(jnp.isnan(sa_repr)).astype(jnp.float32)   # 1.0 if NaN anywhere
sa_norm_min_c  = jnp.min(jnp.linalg.norm(sa_repr, axis=-1))        # 0.0 → near-zero row
c_grad_nan     = _grads_have_nan(c_grads)                           # any leaf NaN?
c_grad_norm    = _grads_global_norm(c_grads)                        # finite-but-huge OK
c_params_nan   = _params_have_nan(post_update_params)               # optimizer poisoned?
```

Aggregate across grad-steps with `jnp.max` (or torch equivalent) so a single fire flips the per-epoch flag, and dump first-NaN flat indices on epoch 0 only via a one-shot host-side `argmax` over `(iter, sgd_step)`.

### 6.4 The H1-vs-H2 framing

Two hypotheses about CRL stability under the constrained Lagrangian:
- **H1**: Stability is implementation-driven (encoder rescaling, NaN traps). Fix the implementation, no constraint pressure needed.
- **H2**: Stability requires non-trivial constraint pressure ($\tilde\lambda > 0$). Without binding constraints the actor runs away.

Reference 50-epoch result at $d=0.15$ rejects H1. The H2 test (Phase-1e at $d=0.05$, $\tilde\lambda > 0$) is pending. Your Safety Gym port should be designed to reproduce both regimes.

### 6.5 One-change-at-a-time discipline

When debugging or sweeping, change exactly one knob per run. Bundling interventions destroys causal attribution. Each commit message should name exactly the one thing it changes. Static-diff your launch scripts so they hard-fail at job-start if `train.py` doesn't carry the expected change.

---

## 7. Required diagnostic probes (port these on day 1)

Per training step, emit:

**Forward NaN flags** (all `float32` scalar 0/1, no NaN means no flag):
- `nan_obs`, `nan_sa_critic`, `nan_g_critic`, `nan_logits_critic`
- `nan_sa_actor`, `nan_g_actor`, `nan_action_actor`, `nan_f_actor`
- `sa_norm_min_*`, `g_norm_min_*` (pre-row-L2 norms)

**Grad/param NaN flags** (per critic, actor, cost-critic):
- `*_grad_nan` (any leaf NaN in the gradient pytree)
- `*_grad_norm` (sqrt of sum of squared leaf norms)
- `*_params_nan` (any leaf NaN in post-update params)

**Actor-loss component decomposition**:
- `alpha`, `log_prob_mean`, `alpha·log_prob_mean`
- `gaussian_logp_mean`, `sat_correction_mean`, `log_std_mean`
- `f_term_mean`

**Per-epoch one-shot prefill probe** (after buffer prefill, before training starts):
- `buffer NaN anywhere?`, `env_obs NaN?`, first-NaN row index if any.

**Per-epoch one-shot epoch-1 forensics** (host-side `argmax` over flattened `(iter, sgd_step)` axis):
- First flat-index where each NaN flag fired (or `(never)`).
- First 5 values of `c_grad_norm`, `a_grad_norm`, `cc_grad_norm`.
- First 5 values of `c_params_nan`, `a_params_nan`, `cc_params_nan`.

---

## 8. Pass criteria

A successful Safety Gym port must satisfy all of:

1. **No NaN, ever.** Every NaN flag in §7 stays at 0 across 50 epochs. The autograd-safe row-L2 from §6.1 is mandatory.
2. **Bounded actor loss in the feasible regime.** $|\mathcal L_{\text{actor}}|_{\max} < 10^3$ over the full 50-epoch run at a permissive budget. (We expect this to *fail* per §6.2 — and that failure with your component probes pointing at the divergent term is itself a publishable finding.)
3. **Constraint enforcement.** At a tight budget, $\tilde\lambda$ rises above 0, $\hat J_c$ tracks the budget, and the policy reduces hazard-region entries (Safety Gym's `info["cost"]` integral declines or plateaus near the budget).
4. **Reproducibility.** Single-seed deterministic given seed; multi-seed runs (≥3 seeds) report mean ± std on every metric in §7.

---

## 9. Reference implementation

The reference is `constrained-crl/train.py` (JAX/Brax MJX). Read it for:
- Buffer wiring (`TrajectoryUniformSamplingQueue`, `flatten_crl_fn`).
- Sgd-step ordering (critic → actor → alpha → cost critic → dual).
- Probe wiring (forward NaN flags, grad/param flags, actor-component probes).
- The PID-Lagrangian implementation including anti-windup.

You should **not** mirror the JAX-specific scaffolding (`jit`/`scan`/Brax wrappers) for a Safety Gym port — Safety Gym is gymnasium / MuJoCo native, and PyTorch + `gymnasium.vector` is a more natural target. But every algorithm-level decision in the reference is the source of truth.
