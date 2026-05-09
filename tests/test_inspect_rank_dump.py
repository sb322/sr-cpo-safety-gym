from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inspect_rank_dump.py"
SPEC = importlib.util.spec_from_file_location("inspect_rank_dump", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
inspect_rank_dump = MODULE.inspect_rank_dump


def test_inspect_rank_dump_reports_steps_done_and_correlations(tmp_path) -> None:
    path = tmp_path / "rank_dump.npz"
    labels = np.asarray(
        [
            [1.0, 1.2, 1.1],
            [2.0, 2.0, 2.0],
            [3.0, 3.4, 3.1],
            [4.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )
    np.savez(
        path,
        valid_mask=np.asarray([True, False, True, False]),
        labels_dense_h50=labels,
        state_info_steps=np.asarray([5.0, 980.0, 10.0, 999.0], dtype=np.float32),
        state_done=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        unroll_index_at_collection=np.asarray([0, 1, 0, 2], dtype=np.int32),
    )

    report = inspect_rank_dump(path)

    assert "INSPECT_RANK_DUMP" in report
    assert "slots=4 valid=2 valid_frac=0.5" in report
    assert "state_done_frac all=0.5" in report
    assert "state_info_steps all[" in report
    assert "state_info_steps valid[" in report
    assert "dense_label_spread invalid[" in report
    assert "corr steps_spread=" in report
    assert "corr done_valid=" in report
