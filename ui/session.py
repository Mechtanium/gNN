"""What the page keeps for a viewer between reruns: the open project, and the derived arrays
it is expensive to recompute on every interaction.

The project lives in ``pl.session_state`` — one per browser session — and a job receives it
as a bundle built when the job starts (:class:`Bundle`), never on a rerun.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import perdlit as pl

from builder import design as B
from builder.project import Project

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / "templates"
CHUNK = 1 << 20


def _five_spot() -> dict:
    d = B.default_design("FIVESPOT")
    d["title"] = "Five-spot waterflood of an undersaturated oil"
    d["phases"] = ["OIL", "WATER", "GAS", "DISGAS"]
    g = d["grid"]
    g.update(nx=21, ny=21, nz=3, dx=[100.0] * 21, dy=[100.0] * 21, dz=[15.0, 20.0, 15.0],
             top={"kind": "tilted", "depth": 6000.0, "dip_x": 1.0, "dip_y": 0.0})
    d["properties"]["PORO"] = [{"op": "layers", "values": [0.20, 0.24, 0.21]},
                               {"op": "random", "sd": 0.02, "range": [5, 5, 1], "seed": 3, "min": 0.08, "max": 0.32}]
    d["fluids"][0]["pb"] = 1800.0
    d["equil"] = [{"datum": 6050.0, "pressure": 2900.0, "woc": 6500.0, "goc": 5500.0, "pc_woc": 0.0, "pc_goc": 0.0}]
    corners = {"INJ-NW": (1, 1), "INJ-NE": (21, 1), "INJ-SW": (1, 21), "INJ-SE": (21, 21)}
    d["wells"] = [{"name": "PROD", "kind": "producer", "phase": "OIL", "group": "G1",
                   "trajectory": [[11, 11, 1], [11, 11, 3]], "diameter": 0.5, "skin": 0.0}]
    d["wells"] += [{"name": n, "kind": "injector", "phase": "WATER", "group": "G1",
                    "trajectory": [[i, j, 1], [i, j, 3]], "diameter": 0.5, "skin": 0.0} for n, (i, j) in corners.items()]
    d["schedule"] = {"end": "2032-01-01", "report": "monthly", "history_end": "2029-01-01", "events": (
        [{"date": "2025-01-01", "well": "PROD", "control": "LRAT", "rate": 2400.0, "bhp": 800.0}]
        + [{"date": "2025-01-01", "well": n, "control": "RATE", "rate": 650.0, "bhp": 4500.0} for n in corners])}
    return d


def _faulted_anticline() -> dict:
    d = B.default_design("ANTICLINE")
    d["title"] = "A faulted anticline with a gas cap"
    g = d["grid"]
    g.update(nx=24, ny=12, nz=6, dx=[250.0] * 24, dy=[250.0] * 12, dz=[25.0] * 6,
             top={"kind": "anticline", "depth": 7800.0, "relief": 260.0, "x0": 0.5, "y0": 0.5,
                  "radius": 0.45, "azimuth": 90.0, "dip_x": 0.4, "dip_y": 0.0},
             faults=[{"name": "F1", "nodes": [[14, 0], [12, 6], [13, 12]], "throw": 60.0,
                      "downthrown": "right", "mult": 0.05}])
    d["properties"]["PORO"] = [{"op": "layers", "values": [0.18, 0.23, 0.25, 0.21, 0.17, 0.20]},
                               {"op": "random", "sd": 0.025, "range": [6, 4, 2], "seed": 11, "min": 0.05, "max": 0.33}]
    d["equil"] = [{"datum": 7900.0, "pressure": 3550.0, "woc": 8000.0, "goc": 7860.0, "pc_woc": 0.0, "pc_goc": 0.0}]
    d["fluids"][0]["pb"] = 3500.0
    d["wells"] = [
        {"name": "P-WEST", "kind": "producer", "phase": "OIL", "group": "G1",
         "trajectory": [[7, 6, 2], [7, 6, 5]], "diameter": 0.5, "skin": 0.0},
        {"name": "P-EAST", "kind": "producer", "phase": "OIL", "group": "G1",
         "trajectory": [[18, 6, 3], [18, 6, 4], [21, 6, 4]], "diameter": 0.5, "skin": 0.0},
        {"name": "I-FLANK", "kind": "injector", "phase": "WATER", "group": "G1",
         "trajectory": [[3, 2, 4], [3, 2, 6]], "diameter": 0.5, "skin": 0.0},
    ]
    d["schedule"] = {"end": "2034-01-01", "report": "monthly", "history_end": "2030-01-01", "events": [
        {"date": "2025-01-01", "well": "P-WEST", "control": "ORAT", "rate": 1500.0, "bhp": 1200.0},
        {"date": "2025-01-01", "well": "P-EAST", "control": "ORAT", "rate": 1800.0, "bhp": 1200.0},
        {"date": "2026-01-01", "well": "I-FLANK", "control": "RATE", "rate": 2500.0, "bhp": 5000.0},
    ]}
    return d


#: What "New" offers: designs built in the app, and the sample decks (imported).
TEMPLATES: dict[str, Any] = {
    "Blank dome (design)": lambda: Project.designed(B.default_design("MODEL")),
    "Five-spot waterflood (design)": lambda: Project.designed(_five_spot()),
    "Faulted anticline (design)": lambda: Project.designed(_faulted_anticline()),
    "SPE1 benchmark (deck)": lambda: _deck("SPE1CASE1.DATA", history_end_days=1826.0),
    "SPE-2 single-well coning (deck)": lambda: _deck("SPE-2-cartesian-equi.DATA"),
}


def _deck(name: str, **settings) -> Project:
    project = Project.imported({name: (TEMPLATES_DIR / name).read_bytes()}, name)
    project.settings.update(settings)
    return project


def project() -> Project:
    """The viewer's open project (a blank design the first time)."""
    state = pl.session_state
    if "project" not in state:
        state["project"] = TEMPLATES["Blank dome (design)"]()
        state["revision"] = 0
    return state["project"]


def replace(new: Project) -> None:
    pl.session_state["project"] = new
    touch()
    for key in list(pl.session_state):
        if key.startswith("cache:"):
            del pl.session_state[key]


def touch() -> None:
    """Note that the project changed (derived caches are keyed by its content)."""
    pl.session_state["revision"] = pl.session_state.get("revision", 0) + 1


def design() -> dict | None:
    return project().design


def design_key(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached(name: str, key: str, make):
    """``make()`` once per ``key``: a derived value kept until the content it came from changes."""
    slot = f"cache:{name}"
    hit = pl.session_state.get(slot)
    if hit and hit[0] == key:
        return hit[1]
    value = make()
    pl.session_state[slot] = (key, value)
    return value


def arrays(d: dict) -> dict:
    """The design's property and region arrays, ``(nz, ny, nx)`` each."""
    return cached("arrays", design_key(d), lambda: B.evaluate_all(d))


def geometry(d: dict) -> dict:
    return cached("geometry", design_key(d["grid"]), lambda: B.corner_point(d))


def edit_design(mutate) -> None:
    """Apply ``mutate(design)`` to a copy and keep it when the design still makes sense."""
    d = copy.deepcopy(design())
    mutate(d)
    project().design = B.normalized(d)
    touch()


def notice(message: str, kind: str = "note") -> None:
    pl.session_state.setdefault("notices", []).append((kind, message))


def take_notices() -> list:
    out = pl.session_state.get("notices", [])
    pl.session_state["notices"] = []
    return out


class Bundle:
    """The project as a job's input stream: built when the job starts reading it (not on a
    rerun), in chunks of at most a mebibyte. ``deck_name`` is known without building it."""

    def __init__(self, p: Project):
        self._project = p

    def __iter__(self) -> Iterator[bytes]:
        data, _ = self._project.bundle()
        for start in range(0, len(data), CHUNK):
            yield data[start:start + CHUNK]


def deck_name(p: Project) -> str:
    if p.design is not None:
        name = "".join(ch for ch in str(p.design["name"]).upper() if ch.isalnum() or ch in "-_") or "MODEL"
        return f"{name}.DATA"
    assert p.deck is not None
    return p.deck.root.path
