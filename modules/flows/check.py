r"""
The deck check, and the model as the simulator will see it.

The model is handed to OPM Flow, which reads the whole deck tree, builds the grid —
corner points, inactive cells, the pore-volume cut ``MINPV``, pinch-outs — evaluates every
property operation (``EQUALS``, ``MULTIPLY``, ``COPY``, boxes, regions) and writes the
grid and the static properties, then stops before the first time step. Nothing is
simulated, so it costs seconds even at field scale. The job streams what Flow said about
the deck (``diag`` messages, from its print file) and the model it built: the grid, the
wells' perforated cells, and one snapshot of the static properties —

.. math::

    \phi,\quad \log_{10} k_x,\quad \log_{10} k_y,\quad \log_{10} k_z,\quad
    \mathrm{NTG},\quad V_p,\quad z,\quad
    \mathrm{SATNUM},\ \mathrm{PVTNUM},\ \mathrm{EQLNUM},\ \mathrm{FIPNUM}

where:

- :math:`\phi`: the porosity; :math:`k_x, k_y, k_z`: the permeabilities [mD], shown on a logarithmic scale as a reservoir's permeability is read.
- :math:`\mathrm{NTG}`: the net-to-gross ratio; :math:`V_p`: the pore volume of the cell (in the deck's volume unit); :math:`z`: the depth of its centre.
- the four region numbers: which saturation functions, which fluid tables, which initial equilibration and which reporting region a cell belongs to (drawn as classes).

Everything is read in the deck's own units (FIELD, METRIC or LAB), as Flow writes them.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from .. import frames
from ..autopilot import Budget
from . import common

Item = frames.Item

#: The child interpreter's program: initialise the simulator on the deck, stop before the
#: first step. The wheel's simulator keeps process-wide state, hence a process per check.
_INIT_RUNNER = (
    "import os, sys\n"
    "from opm.simulators import BlackOilSimulator\n"
    "sim = BlackOilSimulator(sys.argv[1], ['--enable-esmry=false'])\n"
    "sim.step_init()\n"
    "sys.stdout.flush(); sys.stderr.flush()\n"
    "os._exit(0)\n"
)

#: EGRID lists a cell's corners as (i, j) = (0,0), (1,0), (0,1), (1,1) on the top face and
#: then on the bottom one; the 3-D view takes them counter-clockwise (VTK order).
_ECLIPSE_TO_VTK = [0, 1, 3, 2, 4, 5, 7, 6]

#: The static fields of the preview, in the order the view lists them, with their labels.
STATIC_FIELDS = {
    "PORO": "Porosity",
    "PERMX": "log10 Permeability X [mD]",
    "PERMY": "log10 Permeability Y [mD]",
    "PERMZ": "log10 Permeability Z [mD]",
    "NTG": "Net to gross",
    "PORV": "Pore volume",
    "DEPTH": "Cell centre depth",
    "SATNUM": "Saturation region (SATNUM)",
    "PVTNUM": "Fluid region (PVTNUM)",
    "EQLNUM": "Equilibration region (EQLNUM)",
    "FIPNUM": "Reporting region (FIPNUM)",
}
REGION_FIELDS = ("SATNUM", "PVTNUM", "EQLNUM", "FIPNUM")
_LOG_FIELDS = ("PERMX", "PERMY", "PERMZ")

_MESSAGE_HEAD = re.compile(r"^(Warning|Error|Problem|Bug|Fatal)\b[:\s]*(.*)$", re.I)
_WHERE = re.compile(r"\bIn (?:file:?\s*)?(?P<file>\S+?),?\s+line\s+(?P<line>\d+)", re.I)


def run(bundle: bytes, deck_name: str, budget: Budget = Budget(),
        work_root: str | Path | None = None) -> Iterator[Item]:
    """Check the bundle's deck with Flow and stream the findings and the static model."""
    t_start = time.time()
    project = common.unpack(bundle, deck_name, work_root)
    # The check initialises Flow in a folder of its own: beside the simulation's results
    # it would rewrite their grid file, and every later job would extract the model again.
    member, tree, _ = common.read_bundle(bundle, deck_name)
    folder = project.work / "check"
    for name, data in tree.items():
        target = folder / name
        if not (target.is_file() and target.read_bytes() == data):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    deck = folder / member
    yield frames.stage("check", 0.05, f"OPM Flow is reading {deck.name}")

    # Leftovers of an earlier check would pass for this one's.
    for ext in ("EGRID", "INIT", "PRT"):
        for candidate in deck.parent.glob(f"*.{ext}"):
            if candidate.stem.upper() == deck.stem.upper():
                candidate.unlink()
    proc = subprocess.run([sys.executable, "-c", _INIT_RUNNER, str(deck)], cwd=deck.parent,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          errors="replace", timeout=1800)
    egrid, init, prt = (_result(deck, ext) for ext in ("EGRID", "INIT", "PRT"))
    messages = list(parse_messages(prt.read_text(errors="replace") if prt else "", deck.parent))
    if proc.returncode != 0 or egrid is None or init is None:
        if not any(level == "error" for level, _, _ in messages):
            messages += list(parse_messages(proc.stdout, deck.parent))
        if not any(level == "error" for level, _, _ in messages):
            tail = "\n".join(proc.stdout.strip().splitlines()[-8:])
            messages.append(("error", deck.name, f"Flow could not read the deck:\n{tail}"))
    for level, where, message in messages:
        yield frames.diag(level, where, message)
    n_err = sum(1 for level, _, _ in messages if level == "error")
    n_warn = sum(1 for level, _, _ in messages if level == "warning")
    if egrid is None or init is None:
        yield frames.metrics(0, {"errors": n_err, "warnings": n_warn, "wall_s": time.time() - t_start})
        yield frames.stage("done", 1.0, f"the deck does not initialise: {n_err} error(s)")
        return

    yield frames.stage("check", 0.6, "reading the grid and the static properties Flow built")
    model = read_static(egrid, init)
    model["wells"] = well_cells(deck, model)
    ranges = {name: _range(values, name in REGION_FIELDS) for name, values in model["fields"].items()}
    yield from frames.grid(model["corners"], model["ijk"], model["dims"], [0.0], ranges,
                           wells=model["wells"], fields=list(model["fields"]),
                           categorical=[f for f in REGION_FIELDS if f in model["fields"]])
    yield frames.state(0, 0, 0.0, model["fields"], ranges)
    yield frames.metrics(0, {
        "errors": n_err, "warnings": n_warn,
        "nx": int(model["dims"][0]), "ny": int(model["dims"][1]), "nz": int(model["dims"][2]),
        "cells": int(np.prod(model["dims"])), "active_cells": int(model["ijk"].shape[0]),
        "wells": len(model["wells"]), "units": model["units"],
        "pore_volume": float(np.sum(model["fields"].get("PORV", 0.0))),
        "wall_s": time.time() - t_start,
    })
    verdict = "no errors" if n_err == 0 else f"{n_err} error(s)"
    yield frames.stage("done", 1.0, f"{model['ijk'].shape[0]:,} active cells, {verdict}, "
                                    f"{n_warn} warning(s) in {time.time() - t_start:.0f}s")


def _result(deck: Path, ext: str) -> Path | None:
    """The result file ``ext`` beside the deck (OPM upper-cases the stem)."""
    for candidate in deck.parent.glob(f"*.{ext}"):
        if candidate.stem.upper() == deck.stem.upper():
            return candidate
    return None


def parse_messages(text: str, base: Path) -> Iterator[tuple[str, str, str]]:
    """``(level, where, message)`` of every warning and error in Flow's output: a message
    is a ``Warning:`` or ``Error:`` line and the lines after it up to a blank one; ``where``
    is the deck file and line it names, relative to the deck's folder."""
    prefix = str(base).rstrip("/") + "/"
    lines = text.replace(prefix, "").splitlines()
    i = 0
    seen: set[tuple[str, str]] = set()
    while i < len(lines):
        match = _MESSAGE_HEAD.match(lines[i].strip())
        if not match or match.group(2).strip().lower().startswith("summary"):
            i += 1
            continue
        kind = match.group(1).lower()
        level = "warning" if kind == "warning" else "error"
        head = match.group(2).strip()
        body = [head] if head else []
        i += 1
        # "Warning: Unsupported keywords:" puts a blank line before its list
        if head.endswith(":") or not head:
            while i < len(lines) and not lines[i].strip():
                i += 1
        while i < len(lines) and lines[i].strip():
            if _MESSAGE_HEAD.match(lines[i].strip()):
                break
            body.append(lines[i].strip())
            i += 1
        message = "\n".join(line for line in body if not line.startswith(("Reading from:", "Processing keyword")))
        where = ""
        found = _WHERE.search(message)
        if found:
            try:
                rel = os.path.relpath(found.group("file"), base) if os.path.isabs(found.group("file")) else found.group("file")
            except ValueError:
                rel = found.group("file")
            where = f"{rel}:{found.group('line')}"
        key = (level, message.splitlines()[0] if message else "")
        if message and key not in seen:
            seen.add(key)
            yield level, where, message


def read_static(egrid: Path, init: Path) -> dict:
    """The grid and the static properties of a Flow initialisation: corners
    ``(n_active, 8, 3)`` in VTK order (depth positive down), zero-based ``ijk``, the
    dimensions, the deck's unit system and the fields of :data:`STATIC_FIELDS` per
    active cell."""
    from opm.io.ecl import EclFile, EGrid

    grid = EGrid(str(egrid))
    nx, ny, nz = (int(d) for d in grid.dimension)
    n = int(grid.active_cells)
    corners = np.empty((n, 8, 3), np.float64)
    for a in range(n):
        x, y, z = grid.xyz_from_active_index(a)
        corners[a] = np.stack([np.asarray(x), np.asarray(y), np.asarray(z)], axis=1)[_ECLIPSE_TO_VTK]
    ijk = np.asarray([grid.ijk_from_active_index(a) for a in range(n)], np.int32)

    data = EclFile(str(init))
    names = {entry[0] for entry in data.arrays}
    fields: dict[str, np.ndarray] = {}
    for name in STATIC_FIELDS:
        if name not in names:
            continue
        values = np.asarray(data[name], np.float64).ravel()
        if values.size != n:          # a global-size array (rare): keep the active cells
            global_index = ijk[:, 0] + nx * (ijk[:, 1] + ny * ijk[:, 2])
            if values.size != nx * ny * nz:
                continue
            values = values[global_index]
        if name in _LOG_FIELDS:
            values = np.log10(np.maximum(values, 1e-6))
        fields[name] = values
    units = "FIELD"
    if "INTEHEAD" in names:
        code = int(np.asarray(data["INTEHEAD"])[2])
        units = {1: "METRIC", 2: "FIELD", 3: "LAB", 4: "PVT-M"}.get(code, "FIELD")
    return {"corners": corners, "ijk": ijk, "dims": (nx, ny, nz), "fields": fields, "units": units}


def well_cells(deck: Path, model: dict) -> list[dict]:
    """Every well of the schedule with the active cells it is ever connected in."""
    try:
        import opm.io
        from opm.io.ecl_state import EclipseState
        from opm.io.parser import ParseContext, Parser
        from opm.io.schedule import Schedule

        context = ParseContext([("PARSE_*", opm.io.action.ignore)])
        parsed = Parser().parse(str(deck), context)
        schedule = Schedule(parsed, EclipseState(parsed))
    except Exception:  # noqa: BLE001 - a deck Flow read but this parser cannot: no markers
        return []
    nx, ny, _ = model["dims"]
    ijk = model["ijk"]
    active = {int(i + nx * (j + ny * k)): a for a, (i, j, k) in enumerate(ijk)}
    last = max(len(schedule.reportsteps) - 1, 0)
    out: dict[str, set[int]] = {}
    for step in range(0, last + 1, max(1, last // 50 or 1)):
        for well in schedule.get_wells(step):
            cells = out.setdefault(well.name, set())
            for conn in well.connections():
                i, j, k = conn.pos
                a = active.get(int(i + nx * (j + ny * k)))
                if a is not None:
                    cells.add(a)
    return [{"name": name, "cells": sorted(cells)} for name, cells in sorted(out.items())]


def _range(values: np.ndarray, categorical: bool) -> list[float]:
    """The colour range of a static field: its own extremes (whole numbers for classes)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [0.0, 1.0]
    lo, hi = float(finite.min()), float(finite.max())
    if categorical:
        return [float(np.floor(lo)), float(np.ceil(hi)) if hi > lo else float(np.floor(lo)) + 1.0]
    return [lo, hi if hi > lo else lo + 1.0]
