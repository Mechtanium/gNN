"""What the work paths share, without a simulator: the bundle a job receives, its digest
and work directory, the history | prediction boundary, the well messages, the
composition the paths train, and the operations' one-shape signature."""
import io
import json
import zipfile

import numpy as np
import pytest

from modules import autopilot, frames
from modules.autopilot import Budget
from modules.flows import common, forecast

DECK = b"RUNSPEC\nDIMENS\n 2 2 1 /\nGRID\nINCLUDE\n 'INCLUDE/grid.inc' /\n"


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_a_bare_deck_is_its_own_bundle():
    member, tree, extras = common.read_bundle(DECK, "MODEL")
    assert member == "MODEL.DATA" and tree == {"MODEL.DATA": DECK} and extras == {}
    assert common.read_bundle(DECK, "dir/Model.data")[0] == "Model.data"
    with pytest.raises(ValueError, match="empty deck"):
        common.read_bundle(b"  \n", "x")


def test_a_zip_bundle_splits_the_deck_tree_from_the_projects_own_files():
    bundle = _zip({"NORNE.DATA": DECK, "INCLUDE/grid.inc": b"PORO\n 4*0.2 /\n",
                   "project.json": b'{"history_end_days": 365}',
                   "observed/record.csv": b"well,days,WBHP\n", "inversion/prior.json": b"{}"})
    member, tree, extras = common.read_bundle(bundle, "")
    assert member == "NORNE.DATA"
    assert sorted(tree) == ["INCLUDE/grid.inc", "NORNE.DATA"]
    assert sorted(extras) == ["inversion/prior.json", "observed/record.csv", "project.json"]
    # The digest is the deck tree's alone: the boundary, a record or a prior do not move it.
    bare = _zip({"NORNE.DATA": DECK, "INCLUDE/grid.inc": b"PORO\n 4*0.2 /\n"})
    assert common.tree_digest(tree) == common.tree_digest(common.read_bundle(bare, "")[1])
    changed = _zip({"NORNE.DATA": DECK, "INCLUDE/grid.inc": b"PORO\n 4*0.3 /\n"})
    assert common.tree_digest(tree) != common.tree_digest(common.read_bundle(changed, "")[1])


def test_the_deck_of_a_zip_is_named_or_the_only_one():
    two = _zip({"A.DATA": DECK, "B.DATA": DECK, "sub/C.DATA": DECK})
    assert common.read_bundle(two, "B.DATA")[0] == "B.DATA"
    assert common.read_bundle(two, "sub/C.DATA")[0] == "sub/C.DATA"
    with pytest.raises(ValueError, match="2 top-level .DATA files"):
        common.read_bundle(two, "")
    with pytest.raises(ValueError, match="0 top-level"):
        common.read_bundle(_zip({"INCLUDE/x.inc": b"x"}), "")


def test_a_member_that_leaves_the_bundle_is_refused():
    for name in ("../evil.DATA", "/etc/passwd", "a/../../b"):
        with pytest.raises(ValueError, match="does not stay inside"):
            common.read_bundle(_zip({name: b"x", "A.DATA": DECK}), "")


def test_unpack_writes_the_tree_once_and_names_the_work_directory_by_digest(tmp_path):
    bundle = _zip({"M.DATA": DECK, "INCLUDE/grid.inc": b"PORO\n 4*0.2 /\n",
                   "project.json": b'{"history_end_days": 30.5}'})
    project = common.unpack(bundle, "", tmp_path)
    assert project.work == tmp_path / project.sha[:12] and project.name == "M"
    assert project.deck_path == project.work / "case" / "M.DATA"
    assert (project.work / "case" / "INCLUDE" / "grid.inc").read_bytes() == b"PORO\n 4*0.2 /\n"
    assert not (project.work / "case" / "project.json").exists()
    assert project.settings() == {"history_end_days": 30.5}
    # A second job on the same model leaves the files alone, so the simulator's results
    # beside the deck stay newer than it.
    stamp = project.deck_path.stat().st_mtime_ns
    again = common.unpack(bundle, "", tmp_path)
    assert again.deck_path.stat().st_mtime_ns == stamp and again.sha == project.sha
    assert common.unpack(DECK, "M.DATA", tmp_path).settings() == {}


def test_the_boundary_is_a_count_of_report_steps_and_must_leave_both_sides():
    times = np.array([0.0, 31.0, 59.0, 90.0, 120.0])
    assert forecast.boundary_step(times, {"history_end_days": 59.0}) == 3
    assert forecast.boundary_step(times, {"history_end_days": 89.99}) == 3
    assert forecast.boundary_step(times, {"history_end_days": 90.0}) == 4
    with pytest.raises(ValueError, match="no history . prediction boundary"):
        forecast.boundary_step(times, {})
    for day in (0.0, 120.0, 500.0):
        with pytest.raises(ValueError, match="report step"):
            forecast.boundary_step(times, {"history_end_days": day})


def test_one_well_message_per_well_and_time_carries_every_source():
    times = np.array([0.0, 10.0, 20.0])
    sim = {"WBHP": np.array([[3000.0, np.nan], [2900.0, np.nan], [2800.0, np.nan]]),
           "WOPR": np.array([[100.0, 0.0], [90.0, 0.0], [np.nan, 0.0]])}
    net = {"WBHP": np.array([[3010.0, 1.0], [2890.0, 1.0], [2805.0, 1.0]]),
           "WOPR": np.array([[101.0, 0.0], [91.0, 0.0], [80.0, 0.0]])}
    obs = {"WBHP": np.array([[np.nan, np.nan], [2950.0, np.nan], [np.nan, np.nan]])}
    items = list(common.well_frames(times, ["P1", "I1"], simulator=sim, gnn=net, observed=obs, split_days=10.0))
    got = [(p["well"], i[1], i[2], p["phase"], {k: v for k, v in p.items() if k not in ("well", "phase")})
           for i, p in ((i, frames.parse(i)[3]) for i in items)]
    assert got == [
        ("P1", 0, 0.0, "history", {"WBHP_simulator": 3000.0, "WOPR_simulator": 100.0, "WBHP_gnn": 3010.0, "WOPR_gnn": 101.0}),
        ("P1", 1, 10.0, "history", {"WBHP_simulator": 2900.0, "WOPR_simulator": 90.0, "WBHP_gnn": 2890.0,
                                    "WOPR_gnn": 91.0, "WBHP_observed": 2950.0}),
        ("P1", 2, 20.0, "forecast", {"WBHP_simulator": 2800.0, "WBHP_gnn": 2805.0, "WOPR_gnn": 80.0}),
        ("I1", 0, 0.0, "history", {"WBHP_gnn": 1.0}),       # I1 never produces oil: no WOPR at all
        ("I1", 1, 10.0, "history", {"WBHP_gnn": 1.0}),
        ("I1", 2, 20.0, "forecast", {"WBHP_gnn": 1.0}),
    ]
    assert list(common.well_frames(times, ["P1"])) == []
    with pytest.raises(ValueError, match="unknown well quantity"):
        frames.well(0, 0.0, "P1", {"gnn": {"FOPR": 1.0}})


def test_a_number_json_cannot_carry_is_sent_as_zero_and_named():
    item = frames.item("loss", {"eta": float("nan")}, iteration=3, losses=(float("inf"), 1.0, float("nan"), 0.0, 0.0))
    assert item[2:7] == (0.0, 1.0, 0.0, 0.0, 0.0)
    payload = json.loads(item[7])           # strict JSON: no NaN token
    assert payload == {"eta": None, "nonfinite": [0, 2]}
    assert json.loads(frames.metrics(1, {"a": np.float32(2.5), "b": np.array([1, 2])})[7]) == {"a": 2.5, "b": [1, 2]}


def test_the_three_compositions_differ_only_in_what_their_path_adds(tmp_path):
    deck = tmp_path / "M.DATA"
    deck.write_text("RUNSPEC\nFIELD\n")
    from modules.config import loss_groups, to_flat_dict, validate

    h = autopilot.history(deck, tmp_path, "sha")
    f = autopilot.forecast(deck, tmp_path, "sha", 40)
    m = autopilot.family(deck, tmp_path, "sha", 3)
    assert loss_groups(validate(h)[0]) == ("pde", "ic", "data", "well") and h.w_fixed == (1.0, 1.0, 1.0, 0.1)
    assert loss_groups(validate(f)[0]) == ("pde", "ic", "data", "well", "ctrl", "eq", "mb")
    assert loss_groups(validate(m)[0]) == ("pde", "ic", "data")
    flat = {name: to_flat_dict(c) for name, c in (("h", h), ("f", f), ("m", m))}
    for key, value in autopilot.AUTOPILOT.items():
        for name, path in (("h", autopilot.HISTORY), ("f", autopilot.FORECAST), ("m", autopilot.FAMILY)):
            assert flat[name][key] == path.get(key, value), (name, key)
    assert (f.train_split_n, f.inference.mode, f.ctrl_switch) == (40, "joint", "staircase")
    assert (m.property_channels, m.m_width, m.full_batch_cap) == (3, 16, 32768)
    assert Budget() == Budget(n_iter=600, n_train=24, n_held=8, redraw_every=20, gn_iter=40, n_samples=2000,
                              snapshot_every=25, n_snapshots=8)


def test_no_operation_takes_anything_but_the_bundle_and_the_decks_name():
    import workflow

    for name, op in workflow.workflow.ops.items():
        assert [p[0] for p in op.params] == ["deck_name"], name
        assert op.input_item_types == (bytes,) and op.mode == "bi_di", name
        assert op.output_item_types == (str, int, float, float, float, float, float, str), name
