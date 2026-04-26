# SR-CPO Safety Gym

PyTorch implementation scaffold for SR-CPO (Safety-Regularized Contrastive
Policy Optimization) on Safety-Gymnasium environments.

This repository is intentionally scaffold-only in the first commit. The
mathematical source of truth is [HANDOFF.md](HANDOFF.md); each implementation
commit should map one causal algorithm component from that document into code.

## Install

```bash
python -m venv .venv
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
  title = {SR-CPO Safety Gym},
  note = {Safety-Regularized Contrastive Policy Optimization on Safety-Gymnasium},
  year = {2026}
}
```
