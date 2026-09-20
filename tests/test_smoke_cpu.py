"""End to end on the CPU: SPE-2 deck → OPM Flow → ResInsight → QUICK for a few
iterations → the expected message sequence with sane values.

Needs ResInsight (``RESINSIGHT_EXECUTABLE``) and the SPE-2 deck (``DELTA_PINN_DECK``,
default: the deck vendored under ``tests/data`` if present). Run with
``pytest -m slow``.
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

DECK = Path(os.environ.get("DELTA_PINN_DECK", Path(__file__).parent / "data" / "SPE-2-cartesian-equi.DATA"))


@pytest.fixture(scope="module")
def messages(tmp_path_factory):
    if not DECK.is_file():
        pytest.skip(f"no deck at {DECK}")
    if not os.environ.get("RESINSIGHT_EXECUTABLE"):
        pytest.skip("RESINSIGHT_EXECUTABLE not set")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from modules import presets, stream
    items = list(stream.run_training(DECK.read_bytes(), DECK.name, presets.QUICK, n_iter=5,
                                     snapshot_every=2, n_snapshots=2,
                                     work_root=os.environ.get("DELTA_PINN_WORK") or tmp_path_factory.mktemp("work")))
    return items


def test_message_sequence(messages):
    from modules import frames
    kinds = [m[0] for m in messages]
    assert "error" not in kinds
    assert kinds[0] == "stage" and kinds[-1] == "stage"
    assert frames.parse(messages[-1])[3]["stage"] == "done"
    stages = [frames.parse(m)[3]["stage"] for m in messages if m[0] == "stage"]
    assert stages[:4] == ["simulate", "simulate", "prep", "prep"] and "train" in stages
    assert kinds.index("grid") < kinds.index("loss") < kinds.index("metrics")
    assert kinds.count("loss") == 5
    # snapshots: iteration 0, after 2, after 4, and the end → 4 × n_snapshots
    assert kinds.count("state") == 4 * 2


def test_losses_finite_and_decreasing(messages):
    losses = [m for m in messages if m[0] == "loss"]
    tot = np.array([m[2] for m in losses])
    assert np.all(np.isfinite(tot)) and tot[-1] < tot[0]
    assert all(m[3] > 0 and m[4] > 0 and m[5] > 0 for m in losses)   # pde, ic, data on
    assert [m[1] for m in losses] == [1, 2, 3, 4, 5]


def test_states_inside_ranges_and_saturations_sum_to_one(messages):
    from modules import frames
    grid = frames.decode_grid([frames.parse(m)[3] for m in messages if m[0] == "grid"])
    n = grid["corners"].shape[0]
    assert grid["ijk"].shape == (n, 3) and len(grid["times"]) == 2
    for m in messages:
        if m[0] != "state":
            continue
        s = frames.decode_state(frames.parse(m)[3])
        for f, a in s.items():
            lo, hi = grid["ranges"][f]
            assert a.shape == (n,) and np.isfinite(a).all() and a.min() >= lo and a.max() <= hi
        assert np.allclose(s["S_w"] + s["S_o"] + s["S_g"], 1.0, atol=2e-2)


def test_metrics_present(messages):
    from modules import frames
    met = frames.parse([m for m in messages if m[0] == "metrics"][0])[3]
    assert met["iterations"] == 5 and np.isfinite(met["pressure_rmse_final_t"])
