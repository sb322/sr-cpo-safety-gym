"""Inspect RankBuffer dumps for stale-done source states."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def _percentiles(values: np.ndarray) -> str:
    values = np.asarray(values)
    if values.size == 0:
        return "n=0"
    quantiles = np.percentile(values.astype(np.float64), [0, 50, 90, 99, 100])
    return (
        f"n={values.size} min={quantiles[0]:.6g} p50={quantiles[1]:.6g} "
        f"p90={quantiles[2]:.6g} p99={quantiles[3]:.6g} max={quantiles[4]:.6g}"
    )


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or right.size < 2:
        return float("nan")
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _section(
    name: str,
    values: np.ndarray,
    valid_mask: np.ndarray,
    lines: list[str],
) -> None:
    valid = np.asarray(valid_mask, dtype=bool)
    lines.append(f"{name} all[{_percentiles(values)}]")
    lines.append(f"{name} valid[{_percentiles(values[valid])}]")
    lines.append(f"{name} invalid[{_percentiles(values[~valid])}]")


def inspect_rank_dump(path: str | Path) -> str:
    """Returns a text report for a serialized RankBuffer dump."""

    dump_path = Path(path)
    data = np.load(dump_path)
    required = {
        "valid_mask",
        "labels_dense_h50",
        "state_info_steps",
        "state_done",
        "unroll_index_at_collection",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"{dump_path} is missing required keys: {', '.join(missing)}")

    valid = np.asarray(data["valid_mask"], dtype=bool)
    dense_labels = np.asarray(data["labels_dense_h50"], dtype=np.float32)
    spread = np.max(dense_labels, axis=1) - np.min(dense_labels, axis=1)
    steps = np.asarray(data["state_info_steps"], dtype=np.float32)
    done = np.asarray(data["state_done"], dtype=np.float32)
    unroll = np.asarray(data["unroll_index_at_collection"], dtype=np.int32)

    lines: list[str] = [
        f"INSPECT_RANK_DUMP path={dump_path}",
        (
            f"slots={valid.size} valid={int(np.sum(valid))} "
            f"valid_frac={float(np.mean(valid)):.6g}"
        ),
        (
            f"state_done_frac all={float(np.mean(done > 0.5)):.6g} "
            f"valid={float(np.mean(done[valid] > 0.5)) if np.any(valid) else float('nan'):.6g} "
            f"invalid={float(np.mean(done[~valid] > 0.5)) if np.any(~valid) else float('nan'):.6g}"
        ),
    ]
    _section("state_info_steps", steps, valid, lines)
    _section("dense_label_spread", spread, valid, lines)
    _section("unroll_index", unroll, valid, lines)
    lines.extend(
        [
            f"corr steps_spread={_corr(steps, spread):.6g}",
            f"corr steps_valid={_corr(steps, valid.astype(np.float32)):.6g}",
            f"corr done_spread={_corr(done, spread):.6g}",
            f"corr done_valid={_corr(done, valid.astype(np.float32)):.6g}",
        ]
    )
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank_dump", help="Path to a RankBuffer .npz dump")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(inspect_rank_dump(args.rank_dump))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
