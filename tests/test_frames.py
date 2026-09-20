"""Encode/decode round trips and size guards for the streamed messages."""
import json

import numpy as np
import pytest

from modules import frames


def _grid(n_cells: int):
    rng = np.random.default_rng(0)
    corners = rng.normal(size=(n_cells, 8, 3)).astype(np.float32)
    ijk = np.stack(np.unravel_index(np.arange(n_cells), (30, 40, 40)), axis=1).astype(np.int16)
    return corners, ijk


@pytest.mark.parametrize("n_cells", [9_375, 44_420])
def test_grid_round_trip_and_part_sizes(n_cells):
    corners, ijk = _grid(n_cells)
    ranges = {f: [0.0, 1.0] for f in frames.FIELDS}
    items = list(frames.grid(corners, ijk, (30, 40, 40), [0.0, 10.0], ranges, wells=[{"name": "P", "cells": [1]}]))
    assert all(it[0] == "grid" for it in items)
    for it in items:
        assert len(it[7].encode()) <= frames.GRID_PART_BYTES * 1.1
    parts = [frames.parse(it)[3] for it in items]
    assert parts[0]["n_parts"] == len(items) and parts[0]["wells"][0]["name"] == "P"
    got = frames.decode_grid(parts)
    assert np.array_equal(got["corners"], corners) and np.array_equal(got["ijk"], ijk)
    assert got["dims"] == [30, 40, 40] and got["times"] == [0.0, 10.0]


def test_state_clips_into_range_and_round_trips():
    n = 100
    ranges = {f: [0.0, 1.0] for f in frames.FIELDS}
    ranges["p_o"] = ranges["p_w"] = ranges["p_g"] = [1000.0, 5000.0]
    ranges["R_so"] = [0.0, 300.0]
    fields = {f: np.linspace(-1, 2, n) for f in frames.FIELDS}
    fields["p_o"] = np.linspace(0, 9000, n)
    fields["p_o"][3] = np.nan
    it = frames.state(7, 2, 365.0, fields, ranges)
    kind, iteration, losses, payload = frames.parse(it)
    assert kind == "state" and iteration == 7 and losses == (0.0,) * 5
    assert payload["t_index"] == 2 and payload["t_days"] == 365.0 and payload["n_cells"] == n
    dec = frames.decode_state(payload)
    for f in frames.FIELDS:
        lo, hi = ranges[f]
        assert dec[f].shape == (n,) and dec[f].min() >= lo and dec[f].max() <= hi
    assert dec["p_o"][3] == 1000.0            # NaN → the range's floor
    assert np.allclose(dec["S_w"][40:60], np.clip(fields["S_w"][40:60], 0, 1), atol=2e-3)
    assert len(it[7]) < 3_000                 # ~1.4 KB for 100 cells × 7 fields


def test_loss_stage_metrics_error_items():
    lo = frames.loss(12, 1.5, 1.0, 0.2, 0.2, 0.1, wall_s=3.0, eta=0.5)
    assert lo[:7] == ("loss", 12, 1.5, 1.0, 0.2, 0.2, 0.1)
    assert json.loads(lo[7]) == {"wall_s": 3.0, "eta": 0.5}
    st = frames.stage("prep", 1.7, "x")
    assert st[0] == "stage" and st[2] == 1.0 and json.loads(st[7])["stage"] == "prep"
    me = frames.metrics(20, {"pressure_rmse_final_t": np.float32(2.5)})
    assert frames.parse(me)[3] == {"pressure_rmse_final_t": 2.5}
    assert frames.parse(frames.error("boom"))[3] == {"message": "boom"}
    with pytest.raises(ValueError):
        frames.item("nope")


def test_field_ranges_widen_by_sigma_and_bound_saturations():
    class Phys:
        P_MIN, P_MAX = 900.0, 5100.0
    rng = np.random.default_rng(1)
    ref = {"pres": rng.uniform(1000, 5000, (5, 50)), "swat": rng.uniform(0.2, 0.8, (5, 50)),
           "sgas": rng.uniform(0.0, 0.1, (5, 50)), "rs": rng.uniform(0, 200, (5, 50))}
    r = frames.field_ranges(ref, Phys())
    s = frames.RANGE_SIGMA * ref["swat"].std()
    assert r["S_w"][0] == pytest.approx(max(0.0, ref["swat"].min() - s))
    assert r["S_w"][1] == pytest.approx(min(1.0, ref["swat"].max() + s))
    assert 0.0 <= r["S_g"][0] and r["S_g"][1] <= 1.0 and r["R_so"][0] >= 0.0
    assert r["p_o"] == r["p_w"] == r["p_g"] and r["p_o"][0] <= 900.0 and r["p_o"][1] >= 5100.0
