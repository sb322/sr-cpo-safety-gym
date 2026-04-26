# SR-CPO MJX Safety Gym

JAX/Brax/MJX implementation scaffold for SR-CPO
(Safety-Regularized Contrastive Policy Optimization) on Brax-native
Safety-Gym-style environments.

This repository is intentionally scaffold-only at this point. The mathematical
source of truth is [HANDOFF.md](HANDOFF.md); each implementation commit should
map one causal algorithm component from that document into code.

## Target

The target stack is Option A from the target-environment decision:

- JAX + Flax + Optax for the SR-CPO algorithm.
- Brax training utilities where useful.
- MJX-native safety navigation environments for fused, accelerator-friendly
  depth-scaling experiments.

Environment semantics should be anchored to the credible MJX Safety Gym work in
[lasgroup/safe-learning](https://github.com/lasgroup/safe-learning), especially
its `ss2r/benchmark_suites/safety_gym/go_to_goal.py` implementation. The
standalone [mjx-safety-gym](https://github.com/yardenas/mjx-safety-gym) package
is a useful clean extraction target/prototype, but benchmark claims must be
phrased as Brax/MJX Safety-Gym-style benchmarks unless and until exact official
Safety-Gymnasium parity is established.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Quickstart

Training code lands in later commits. The scaffold can be checked with:

```bash
pytest -q
```

## Citation

```bibtex
@misc{srcpo_safety_gym,
  title = {SR-CPO MJX Safety Gym},
  note = {Safety-Regularized Contrastive Policy Optimization on Brax/MJX safety-navigation benchmarks},
  year = {2026}
}
```
