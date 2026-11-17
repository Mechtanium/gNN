"""The model builder: the deck as a lossless document, records and tables, designs and the
decks generated from them, the project file, and the deck variants of an inversion.

OPM Flow's own reading of a deck is the reference throughout: the scanner must find the
keywords OPM's parser finds, in the same order, and every generated or edited deck must
initialise in Flow without an error. Norne is used when the research repository's copy is
present (``GNN_NORNE`` names another)."""

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from builder import design as B
from builder import records as R
from builder import variants
from builder.deck import Deck, DeckError, to_zip
from builder.project import Project, read_record_csv, write_record_csv

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
NORNE = Path(os.environ.get("GNN_NORNE", ROOT.parent / "Delta-PINNs" / "data" / "norne"))


def _tree(root: Path, deck: str) -> dict[str, bytes]:
    tree = {deck: (root / deck).read_bytes()}
    inc = root / "INCLUDE"
    if inc.is_dir():
        tree.update({str(p.relative_to(root)): p.read_bytes() for p in inc.rglob("*") if p.is_file()})
    return tree


def _opm_keywords(path: Path) -> list[str]:
    import opm.io
    from opm.io.parser import ParseContext, Parser

    context = ParseContext([("PARSE_*", opm.io.action.ignore), ("*", opm.io.action.ignore)])
    return [kw.name for kw in Parser().parse(str(path), context) if kw.name not in ("INCLUDE", "END")]


DECKS = [("SPE1", TEMPLATES, "SPE1CASE1.DATA"), ("SPE-2", TEMPLATES, "SPE-2-cartesian-equi.DATA"),
         ("Norne", NORNE, "NORNE_ATW2013.DATA")]


@pytest.mark.parametrize("label,root,name", DECKS, ids=[d[0] for d in DECKS])
def test_the_scanner_reads_the_keywords_opm_reads_and_writes_every_byte_back(label, root, name):
    if not (root / name).is_file():
        pytest.skip(f"no {label} deck at {root}")
    tree = _tree(root, name)
    deck = Deck.from_tree(tree, name)
    assert deck.missing == []
    mine = [k.name for _, k in deck.walk() if k.name not in ("INCLUDE", "END")]
    assert mine == _opm_keywords(root / name)
    assert deck.tree() == tree
    assert not deck.changed()


def test_an_edit_rewrites_its_own_file_and_leaves_every_other_byte_alone(tmp_path):
    if not (NORNE / "NORNE_ATW2013.DATA").is_file():
        pytest.skip("no Norne deck")
    tree = _tree(NORNE, "NORNE_ATW2013.DATA")
    deck = Deck.from_tree(tree, "NORNE_ATW2013.DATA")
    equil = deck.first("EQUIL")
    owner = deck.file_of(equil).path
    rows = [r for r in R.parse_records(equil.data()) if r]
    rows[0][2] = float(rows[0][2]) + 5.0
    equil.set_body(R.format_records(rows, terminate=False))
    out = deck.tree()
    assert [p for p in tree if out[p] != tree[p]] == [owner]
    again = Deck.from_tree(out, "NORNE_ATW2013.DATA")
    assert float(R.parse_records(again.first("EQUIL").data())[0][2]) == rows[0][2]


def test_keywords_are_added_where_their_section_ends_and_removed_cleanly():
    deck = Deck.from_tree({"S.DATA": (TEMPLATES / "SPE1CASE1.DATA").read_bytes()}, "S.DATA")
    added = deck.append("GRID", "MULTIPLY", " 'PERMZ' 0.5 /\n/")
    order = [k.name for k in deck.root.keywords]
    assert order.index("MULTIPLY") < order.index("PROPS") and order.index("MULTIPLY") > order.index("GRID")
    assert added.section == "GRID" and deck.changed()
    deck.put("SOLUTION", "EQUIL", " 8400 4800 8460 0 8300 0 1 0 0 /")
    assert R.parse_records(deck.first("EQUIL").data())[0][2] == "8460"
    deck.remove(added)
    assert not deck.has("MULTIPLY")
    deck.ensure_section("EDIT")
    order = [k.name for k in deck.root.keywords]
    assert order.index("GRID") < order.index("EDIT") < order.index("PROPS")
    with pytest.raises(DeckError):
        Deck.from_tree({"A.DATA": b"RUNSPEC\n"}, "B.DATA")


def test_a_bare_word_inside_a_record_is_data_and_a_title_is_its_own_line():
    deck = Deck.from_text("RUNSPEC\nTITLE\nPRESSURE TEST\nRPTSOL\n PRESSURE\n SWAT /\nDIMENS\n 1 1 1 /\n")
    assert [k.name for _, k in deck.walk()] == ["RUNSPEC", "TITLE", "RPTSOL", "DIMENS"]
    assert deck.first("RPTSOL").data() == " PRESSURE\n SWAT /"


def test_records_tables_and_arrays_read_and_write():
    assert R.parse_records(" 'B-2H' 'G' 3*  1.5 / comment\n 'a b'  2*'x y' /\n/") == [
        ["B-2H", "G", None, None, None, "1.5"], ["a b", "x y", "x y"], []]
    assert R.parse_array("PORO\n 3*0.2 0.25 -- c\n 2*0.3 /".split("\n", 1)[1]).tolist() == [0.2, 0.2, 0.2, 0.25, 0.3, 0.3]
    arr = np.array([500.0] * 100 + [50.0] * 100 + [200.0] * 100)
    assert R.format_array(arr).strip() == "100*500 100*50 100*200 /"
    tables = [np.array([[0.2, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]), np.array([[0.1, 0.0, 0.9, 1.5]])]
    back = R.parse_tables(R.format_tables(tables), 4)
    assert all(np.allclose(a, b) for a, b in zip(tables, back))
    deck = Deck.from_tree({"S.DATA": (TEMPLATES / "SPE1CASE1.DATA").read_bytes()}, "S.DATA")
    pvto = R.parse_pvto(deck.first("PVTO").data())
    again = R.parse_pvto(R.format_pvto(pvto))
    assert all(np.allclose(a, b) for t1, t2 in zip(pvto, again) for a, b in zip(t1, t2))
    rows = R.parse_records(deck.first("COMPDAT").data())
    assert R.parse_records(R.format_records([r for r in rows if r])) == rows
    assert [R.fmt(v) for v in (1.0, 0.25, 3, "PROD", None, 8325.5)] == ["1", "0.25", "3", "'PROD'", "1*", "8325.5"]


# ------------------------------------------------------------------------- designs
def _check(tree: dict[str, bytes], deck: str, tmp_path: Path) -> dict:
    from modules import frames
    from modules.flows import check

    bundle = to_zip(tree)
    items = list(check.run(bundle, deck, work_root=tmp_path))
    metrics = [frames.parse(i)[3] for i in items if i[0] == "metrics"][0]
    metrics["diags"] = [frames.parse(i)[3] for i in items if i[0] == "diag"]
    return metrics


def test_the_templates_initialise_in_flow_without_an_error(tmp_path):
    from ui.session import TEMPLATES as MADE

    for name, make in MADE.items():
        project = make()
        tree, deck = project.tree()
        found = _check(tree, deck, tmp_path)
        assert found["errors"] == 0, (name, found["diags"])
        if project.design is not None:
            assert found["warnings"] == 0, (name, found["diags"])
            g = project.design["grid"]
            assert found["active_cells"] == int(B.actnum(project.design).sum())
            assert (found["nx"], found["ny"], found["nz"]) == (g["nx"], g["ny"], g["nz"])
            assert found["wells"] == len(project.design["wells"])


def test_a_faulted_metric_design_with_a_horizontal_well(tmp_path):
    d = B.default_design("FAULTED", "METRIC")
    d["grid"]["faults"] = [{"name": "F1", "nodes": [[7, 0], [7, 5], [9, 9], [9, 15]], "throw": 15.0,
                            "downthrown": "right", "mult": 0.1}]
    d["grid"]["inactive"] = [{"i": [1, 2], "j": [14, 15], "k": [1, 5]}, {"cells": [[15, 1], [15, 2]], "k": [1, 2]}]
    d["wells"].append({"name": "HZ1", "kind": "producer", "phase": "OIL", "trajectory": [[3, 3, 1], [3, 3, 2], [9, 4, 2.4]],
                       "diameter": 0.15, "skin": 1.0})
    d["schedule"]["events"] += [{"date": "2026-06-01", "well": "HZ1", "control": "BHP", "rate": 120.0},
                                {"date": "2029-01-01", "well": "HZ1", "control": "SHUT"}]
    gen = B.generate(d)
    found = _check(gen.tree, gen.deck_name, tmp_path)
    assert found["errors"] == 0 and found["warnings"] == 0, found["diags"]
    assert found["active_cells"] == 1125 - 20 - 4 and found["units"] == "METRIC"
    conns = B.completions(B.normalized(d), d["wells"][2])
    assert conns[0][:3] == (3, 3, 1) and conns[-1][:3] == (9, 4, 2) and {c[3] for c in conns} == {"Z", "X"}
    shift, faces = B.fault_offsets(B.normalized(d)["grid"])
    assert len(faces) == 17 and set(np.unique(shift)) == {0.0, 15.0}   # 5 + 6 + 6 cell edges
    text = gen.tree["FAULTED.DATA"].decode()
    assert "FAULTS" in text and "MULTFLT" in text and "METRIC" in text


def test_a_trace_is_walked_along_the_cell_edges_nearest_it():
    assert B.staircase([(7, 0), (7, 3), (9, 5)]) == [(7, 0), (7, 1), (7, 2), (7, 3), (8, 3), (8, 4), (9, 4), (9, 5)]


def test_property_recipes(tmp_path):
    d = B.default_design()
    d["properties"]["PORO"] = [{"op": "constant", "value": 0.2},
                               {"op": "box", "value": 0.3, "i": [1, 2], "j": [1, 1], "k": [1, 1]},
                               {"op": "cells", "value": 2.0, "mode": "multiply", "cells": [[15, 15]], "k": [5, 5]}]
    d["regions"]["FIPNUM"] = [{"op": "constant", "value": 1}, {"op": "box", "value": 2, "i": [8, 15]}]
    d["properties"]["NTG"] = [{"op": "constant", "value": 1.0}, {"op": "region", "array": "FIPNUM", "id": 2, "value": 0.8}]
    arrays = B.evaluate_all(d)
    poro = arrays["PORO"]
    assert poro[0, 0, 0] == poro[0, 0, 1] == 0.3 and poro[0, 0, 2] == 0.2 and poro[4, 14, 14] == 0.4
    assert set(np.unique(arrays["FIPNUM"])) == {1, 2} and arrays["NTG"][0, 0, 14] == 0.8 and arrays["NTG"][0, 0, 0] == 1.0
    assert np.allclose(arrays["PERMZ"], 0.1 * arrays["PERMX"])
    assert np.allclose(arrays["PERMX"], 10 ** (-1.0 + 14.0 * poro))


@pytest.mark.parametrize("change,message", [
    (lambda d: d["regions"]["SATNUM"].append({"op": "constant", "value": 2}), "SATNUM reaches 2"),
    (lambda d: d["grid"]["faults"].append({"name": "F", "nodes": [[0, 0], [99, 0]]}), "outside the grid"),
    (lambda d: d["wells"][0].update(trajectory=[[1, 1, 1]]) or d["grid"].update(inactive=[{"i": [1, 1], "j": [1, 1]}]),
     "no active cell"),
    (lambda d: d["schedule"]["events"].append({"date": "2026-01-01", "well": "NOBODY", "control": "ORAT", "rate": 1}),
     "'NOBODY'"),
    (lambda d: d["schedule"].update(end="2024-01-01"), "end after the start"),
    (lambda d: d.update(phases=["OIL", "WATER", "DISGAS"]), "dissolved gas needs"),
])
def test_a_design_that_cannot_become_a_deck_says_why(change, message):
    d = B.default_design()
    change(d)
    problems = B.check(d)
    assert problems and message in problems[0]


# ------------------------------------------------------------------------- project and variants
def test_a_project_file_holds_the_model_the_boundary_the_record_and_the_prior():
    p = Project.designed(B.default_design("ROUND"))
    p.record = [{"well": "PROD1", "days": 31.0, "WBHP": 3000.0, "WOPR": 2000.0, "WWPR": None, "WGPR": 1.5,
                 "WWIR": None, "WGIR": None}]
    p.prior["woc"] = 33.0
    data, deck = p.bundle()
    back = Project.load(data, "ROUND.gnn")
    assert back.kind == "designed" and back.design == p.design and back.record == p.record
    assert back.prior["woc"] == 33.0 and back.history_end_days() == p.history_end_days() == 1095.0
    assert back.tree()[0] == p.tree()[0] and deck == "ROUND.DATA"
    p.detach()
    assert p.kind == "imported" and p.history_end_days() == 1095.0
    again = Project.load(p.bundle()[0], "ROUND.gnn")
    assert again.kind == "imported" and again.tree()[0] == p.tree()[0]
    assert again.design_reference == back.design
    one = Project.load((TEMPLATES / "SPE1CASE1.DATA").read_bytes(), "spe1.data")
    assert one.kind == "imported" and one.deck.root.path == "spe1.data"
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as z:
        z.writestr("model/SPE1CASE1.DATA", (TEMPLATES / "SPE1CASE1.DATA").read_bytes())
    assert Project.load(nested.getvalue(), "model.zip").deck.root.path == "SPE1CASE1.DATA"


def test_a_record_reads_from_csv_and_writes_back():
    text = b"Well,Days,WBHP,WOPR\nP1,31,3000,2000\nP1,59,2950,\n"
    rows = read_record_csv(text)
    assert rows[1] == {"well": "P1", "days": 59.0, "WBHP": 2950.0, "WOPR": None, "WWPR": None, "WGPR": None,
                       "WWIR": None, "WGIR": None}
    assert read_record_csv(write_record_csv(rows)) == rows
    with pytest.raises(ValueError, match="'well' and a 'days'"):
        read_record_csv(b"name,t\nP1,3\n")


def test_variants_edit_the_vertical_permeability_and_the_contacts_and_nothing_else():
    p = Project.designed(B.default_design("VAR"))
    tree, deck_name = p.tree()
    deck = Deck.from_tree(tree, deck_name)
    assert variants.properties_of(deck, 8010.0, 8240.0) == ["kvkh", "woc"]   # the GOC lies above the top
    assert variants.properties_of(deck, 7900.0, 8240.0) == ["kvkh", "woc", "goc"]
    values = variants.physical([1.0, -0.5], ["kvkh", "woc"], {"kvkh": 0.7, "woc": 20.0}, {"woc": 8110.0})
    assert values["kvkh"] == pytest.approx(np.exp(0.7)) and values["woc"] == pytest.approx(8100.0)
    out = variants.variant(tree, deck_name, values)
    changed = [path for path in tree if out[path] != tree[path]]
    assert changed == [deck_name]
    edited = Deck.from_tree(out, deck_name)
    assert R.parse_records(edited.first("MULTIPLY").data())[0] == ["PERMZ", "2.0137527"]
    assert edited.first("MULTIPLY").section == "GRID"
    assert float(R.parse_records(edited.first("EQUIL").data())[0][2]) == pytest.approx(8100.0)
    u = variants.latin_hypercube(8, 2, np.random.default_rng(0))
    for j in range(2):          # one sample per stratum of every dimension
        from scipy.stats import norm

        assert sorted(np.floor(norm.cdf(u[:, j]) * 8).astype(int).tolist()) == list(range(8))


def test_the_page_renders_and_its_buttons_call_the_five_operations():
    from perdlit._script import check_page

    summary = check_page("page", "workflow")
    assert [b["label"] for b in summary["buttons"]] == ["Check", "Simulate", "History match", "Match + forecast", "Invert"]
    assert summary["registered_ops"] == ["check", "history_forecast", "history_match", "invert", "simulate"]
    assert {"tabs", "tree", "data_editor", "map2d", "menu", "action"} <= set(summary["widgets"])
