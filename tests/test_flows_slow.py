"""The work paths end to end on the vendored SPE1 deck with the test budget: a model →
OPM Flow → ResInsight → the composition for three iterations → the expected messages with
sane values. Needs ResInsight (``RESINSIGHT_EXECUTABLE``) and OPM Flow; minutes on a CPU.
Run with ``pytest -m slow`` (``GNN_DECK`` for another deck)."""
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

DECK = Path(os.environ.get("GNN_DECK", Path(__file__).parent / "data" / "SPE1CASE1.DATA"))


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    if not DECK.is_file():
        pytest.skip(f"no deck at {DECK}")
    if not os.environ.get("RESINSIGHT_EXECUTABLE"):
        pytest.skip("RESINSIGHT_EXECUTABLE not set")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    return os.environ.get("GNN_WORK") or tmp_path_factory.mktemp("work")


def _with_boundary(days: float) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(DECK.name, DECK.read_bytes())
        z.writestr("project.json", json.dumps({"history_end_days": days}))
    return buf.getvalue()


def _run(flow, bundle, work):
    import importlib

    from modules.autopilot import Budget

    module = importlib.import_module(f"modules.flows.{flow}")
    return list(module.run(bundle, DECK.name, Budget.smoke(), work_root=work))


@pytest.fixture(scope="module")
def simulated(work):
    return _run("simulate", DECK.read_bytes(), work)


@pytest.fixture(scope="module")
def matched(work, simulated):
    return _run("history", DECK.read_bytes(), work)


@pytest.fixture(scope="module")
def forecasted(work, simulated):
    from modules import frames

    days = frames.parse([m for m in simulated if m[0] == "metrics"][0])[3]["days"]
    return _run("forecast", _with_boundary(days / 2), work), days / 2


def _stages(messages):
    from modules import frames

    return [frames.parse(m)[3]["stage"] for m in messages if m[0] == "stage"]


def _grid_and_states(messages):
    from modules import frames

    grid = frames.decode_grid([frames.parse(m)[3] for m in messages if m[0] == "grid"])
    states = [(m[1], frames.parse(m)[3]) for m in messages if m[0] == "state"]
    return grid, states


def _check_states(messages):
    from modules import frames

    grid, states = _grid_and_states(messages)
    n = grid["corners"].shape[0]
    assert grid["ijk"].shape == (n, 3) and len(grid["times"]) == 2
    for _, payload in states:
        fields = frames.decode_state(payload)
        assert sorted(fields) == sorted(frames.FIELDS)
        for name, values in fields.items():
            lo, hi = grid["ranges"][name]
            tol = 1e-3 * max(abs(lo), abs(hi), 1.0)          # float16 rounding of the clipped value
            assert values.shape == (n,) and np.isfinite(values).all()
            assert values.min() >= lo - tol and values.max() <= hi + tol
    return grid, states


def test_simulate_hands_the_deck_over_and_shows_its_result(simulated):
    from modules import frames

    kinds = [m[0] for m in simulated]
    assert "error" not in kinds
    assert _stages(simulated)[0] == "simulate" and _stages(simulated)[-1] == "done"
    assert "prep" in _stages(simulated)
    grid, states = _check_states(simulated)
    assert len(states) == 2 and grid["wells"] and {w["name"] for w in grid["wells"]} == {"PROD", "INJ"}
    # The simulator's own saturations sum to one.
    for _, payload in states:
        s = frames.decode_state(payload)
        assert np.allclose(s["S_w"] + s["S_o"] + s["S_g"], 1.0, atol=2e-2)
    wells = [frames.parse(m)[3] for m in simulated if m[0] == "well"]
    keys = {(p["well"], k) for p in wells for k in p if k.endswith("_simulator")}
    assert {("PROD", "WBHP_simulator"), ("PROD", "WOPR_simulator"), ("PROD", "WGOR_simulator"),
            ("INJ", "WGIR_simulator"), ("INJ", "WBHP_simulator")} <= keys
    assert all(p["phase"] == "history" for p in wells)
    assert not any(k.endswith(("_gnn", "_observed")) for p in wells for k in p)
    metrics = frames.parse([m for m in simulated if m[0] == "metrics"][0])[3]
    assert metrics["cells"] == 300 and metrics["wells"] == 2 and metrics["report_steps"] > 100


def test_a_second_job_reuses_the_simulation(work, simulated):
    again = _run("simulate", DECK.read_bytes(), work)
    from modules import frames

    assert frames.parse([m for m in again if m[0] == "stage"][0])[3]["message"] == "simulation results reused"


def test_history_match_message_sequence_and_falling_loss(matched):
    from modules import frames

    kinds = [m[0] for m in matched]
    assert "error" not in kinds
    stages = _stages(matched)
    assert stages[0] == "simulate" and stages[-1] == "done"
    assert list(dict.fromkeys(s for s in stages if s in ("resolve", "build", "train"))) == ["resolve", "build", "train"]
    progress = [frames.parse(m)[3]["message"] for m in matched if m[0] == "stage" and frames.parse(m)[3]["stage"] == "train"]
    assert any("of 3 · about" in text and "left" in text for text in progress)
    assert kinds.index("grid") < kinds.index("loss") < kinds.index("well") < kinds.index("metrics")
    losses = [m for m in matched if m[0] == "loss"]
    assert [m[1] for m in losses] == [1, 2, 3]
    total = np.array([m[2] for m in losses])
    assert np.all(np.isfinite(total)) and total[-1] < total[0]
    assert all(m[3] > 0 and m[4] > 0 and m[5] > 0 and m[6] > 0 for m in losses)   # pde, ic, data, well
    assert set(frames.parse(losses[0])[3]["groups"]) == {"pde", "ic", "data", "well"}
    # snapshots: iteration 0, after 2, and the end → 3 sweeps of 2 report times
    _, states = _check_states(matched)
    assert [it for it, _ in states] == [0, 0, 2, 2, 3, 3]
    wells = [frames.parse(m)[3] for m in matched if m[0] == "well"]
    assert wells and all(any(k.endswith("_gnn") for k in p) for p in wells)
    metrics = frames.parse([m for m in matched if m[0] == "metrics"][0])[3]
    assert metrics["iterations"] == 3 and np.isfinite(metrics["pressure_rmse_final_t"])
    assert np.isfinite(metrics["bhp_rms"])


def test_forecast_marks_what_lies_past_the_boundary_and_scores_it(forecasted):
    from modules import frames

    messages, boundary = forecasted
    kinds = [m[0] for m in messages]
    assert "error" not in kinds
    losses = [m for m in messages if m[0] == "loss"]
    assert len(losses) == 3 and np.all(np.isfinite([m[2] for m in losses]))
    assert set(frames.parse(losses[0])[3]["groups"]) == {"pde", "ic", "data", "well", "ctrl", "eq", "mb"}
    wells = [(m[2], frames.parse(m)[3]) for m in messages if m[0] == "well"]
    phases = {p["phase"] for _, p in wells}
    assert phases == {"history", "forecast"}
    assert all((p["phase"] == "forecast") == (t > boundary + 1e-9) for t, p in wells)
    metrics = frames.parse([m for m in messages if m[0] == "metrics"][0])[3]
    assert abs(metrics["history_end_days"] - boundary) < 32      # the report step at or before it
    for key in ("history_pressure_rmse_mean_t", "forecast_pressure_rmse_mean_t", "forecast_bhp_rms",
                "supersaturated_share"):
        assert np.isfinite(metrics[key]), key
    _check_states(messages)


def test_forecast_without_a_boundary_says_what_is_missing(work, simulated):
    with pytest.raises(ValueError, match="no history . prediction boundary"):
        _run("forecast", DECK.read_bytes(), work)


def test_check_reports_the_model_flow_builds(work):
    from modules import frames

    messages = _run("check", DECK.read_bytes(), work)
    metrics = frames.parse([m for m in messages if m[0] == "metrics"][0])[3]
    assert metrics["errors"] == 0 and metrics["active_cells"] == 300 and metrics["wells"] == 2
    grid = frames.decode_grid([frames.parse(m)[3] for m in messages if m[0] == "grid"])
    assert grid["corners"].shape == (300, 8, 3) and "SATNUM" in grid["categorical"]
    static = frames.decode_state(frames.parse([m for m in messages if m[0] == "state"][0])[3])
    assert np.allclose(static["PORO"], 0.3, atol=1e-3) and static["PORV"].max() > 6.0e4   # float32 rides


def test_inversion_end_to_end_against_a_hidden_record(work, tmp_path):
    """The record of a run with the vertical permeability doubled, which the path never sees,
    inverted through a toy family (a test budget proves the stages, not the accuracy)."""
    from builder import variants
    from builder.project import PRIOR_DEFAULTS, read_record_summary, write_record_csv
    from modules import frames, simulate

    truth = variants.variant({DECK.name: DECK.read_bytes()}, DECK.name, {"kvkh": 2.0})
    (tmp_path / DECK.name).write_bytes(truth[DECK.name])
    for _ in simulate.iter_flow(tmp_path / DECK.name):
        pass
    record = read_record_summary({p.name: p.read_bytes() for p in tmp_path.glob("*.*")
                                  if p.suffix in (".SMSPEC", ".UNSMRY")})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(DECK.name, DECK.read_bytes())
        z.writestr("observed/record.csv", write_record_csv([r for r in record if r["days"] <= 1800]))
        z.writestr("inversion/prior.json", json.dumps(PRIOR_DEFAULTS))
    messages = _run("inversion", buf.getvalue(), work)
    kinds = [m[0] for m in messages]
    assert "error" not in kinds
    assert [frames.parse(m)[3]["stage"] for m in messages if m[0] == "stage"][-1] == "done"
    samples = [frames.parse(m)[3] for m in messages if m[0] == "sample"]
    assert {s["name"] for s in samples if s["status"] in ("done", "reused")} >= {"s000", "s001", "s002", "s003", "s004"}
    posterior = frames.parse([m for m in messages if m[0] == "posterior"][0])[3]
    names = [p["name"] for p in posterior["properties"]]
    assert names == ["kvkh"]            # SPE1's contacts lie outside the reservoir
    kvkh = posterior["properties"][0]
    assert kvkh["low"] <= kvkh["median"] <= kvkh["high"] and len(kvkh["samples"]) > 1
    wells = [frames.parse(m)[3] for m in messages if m[0] == "well"]
    assert any("WBHP_observed" in p and "WBHP_gnn" in p and "WBHP_simulator" in p for p in wells)
    metrics = frames.parse([m for m in messages if m[0] == "metrics"][0])[3]
    assert metrics["runs"] == 3 and metrics["held_out"] == 2 and np.isfinite(metrics["misfit"])
