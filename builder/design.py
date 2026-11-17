r"""
A reservoir model built from scratch: the design, and the deck tree generated from it.

A **design** is the modeller's description of a reservoir, kept as one JSON document: the
grid (its columns, rows and layers, the structural top it hangs from, its faults and its
inactive cells), the property recipes, the region numbers, the fluids, the rock–fluid
functions, the initial equilibrium, the wells along their trajectories, and the schedule of
controls. :func:`generate` turns it into a complete Eclipse deck tree laid out as a
modelling package exports one — the main ``.DATA`` file and its ``INCLUDE`` files by section
— which OPM Flow runs as it is.

**Geometry.** Pillars are vertical; node :math:`(a, b)` of the areal grid sits at

.. math::

    x_a = \sum_{i < a} \Delta x_i,\qquad y_b = \sum_{j < b} \Delta y_j,\qquad
    z_{a,b,k} = z_{\mathrm{top}}(x_a, y_b) + \sum_{l < k} h_l + d_{a,b}

where:

- :math:`\Delta x_i, \Delta y_j`: the widths of column :math:`i` and row :math:`j`; :math:`h_l`: the thickness of layer :math:`l`.
- :math:`z_{\mathrm{top}}`: the structural top — flat, tilted, a dome or an anticline (:func:`top_surface`).
- :math:`d_{a,b}`: the displacement of a fault block: a cell on the downthrown side of a fault takes its throw at all eight corners, so the two sides of the fault no longer meet face to face (``ZCORN`` per cell corner).

**Properties.** Each property is a recipe: an ordered list of operations on a
:math:`(n_z, n_y, n_x)` array — a constant, a value per layer, a depth trend, a correlated
Gaussian field, a transform of another property (:math:`\log_{10} k = a + b\,\phi`), a scaled
copy (:math:`k_z = 0.1\,k_x`), a value in a box, in painted cells or in a region
(:func:`evaluate`).

**Rock–fluid.** Per saturation region, Corey functions on the normalised saturations,

.. math::

    k_{rw} = k_{rw}^{\max}\, \hat S_w^{\,n_w},\quad
    k_{row} = k_{ro}^{\max} (1 - \hat S_w)^{n_{ow}},\quad
    \hat S_w = \frac{S_w - S_{wcr}}{1 - S_{wcr} - S_{owcr}}

and likewise for gas against oil (:func:`scal_tables`), with a Brooks–Corey capillary
pressure when an entry pressure is given.

where:

- :math:`S_{wcr}, S_{owcr}`: the critical water saturation and the residual oil saturation to water; :math:`n_w, n_{ow}`: the Corey exponents; :math:`k^{\max}`: the end-point relative permeabilities.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import correlations as corr
from . import records as R
from .deck import block_text
from .units import from_field, to_field

MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
PROPERTIES = ("PORO", "NTG", "PERMX", "PERMY", "PERMZ")
REGIONS = ("FIPNUM", "SATNUM", "PVTNUM", "EQLNUM")
SUMMARY = ("FOPR", "FWPR", "FGPR", "FOPT", "FWPT", "FGPT", "FWIR", "FGIR", "FWIT", "FGIT", "FPR",
           "FOIP", "FGIP", "FWIP", "FWCT", "FGOR")
WELL_SUMMARY = ("WBHP", "WOPR", "WWPR", "WGPR", "WWIR", "WGIR", "WOPT", "WWPT", "WGPT", "WGOR", "WWCT")
PRODUCER_CONTROLS = ("ORAT", "WRAT", "GRAT", "LRAT", "RESV", "BHP")
INJECTOR_CONTROLS = ("RATE", "BHP")
STATUS_EVENTS = ("SHUT", "OPEN", "STOP")


class DesignError(ValueError):
    """A design that cannot become a deck; ``str(e)`` says what to change."""


# --------------------------------------------------------------------------- the default
def default_design(name: str = "MODEL", units: str = "FIELD") -> dict[str, Any]:
    """A small, complete, runnable design: a gently domed oil reservoir with a gas cap above
    and water below, one producer on the crest flank and one water injector down-dip, five
    years of production. Every editor starts from a copy of it."""
    f = (lambda v, q: from_field(v, q, units))
    return {
        "name": name,
        "title": "A reservoir built in gNN",
        "units": units,
        "start": "2025-01-01",
        "phases": ["OIL", "WATER", "GAS", "DISGAS"],
        "grid": {
            "nx": 15, "ny": 15, "nz": 5,
            "dx": [f(200.0, "length")] * 15,
            "dy": [f(200.0, "length")] * 15,
            "dz": [f(20.0, "length")] * 5,
            "top": {"kind": "dome", "depth": f(8000.0, "length"), "relief": f(150.0, "length"),
                    "x0": 0.5, "y0": 0.5, "radius": 0.6, "dip_x": 0.0, "dip_y": 0.0, "azimuth": 0.0},
            "faults": [],
            "inactive": [],
        },
        "properties": {
            "PORO": [{"op": "layers", "values": [0.22, 0.25, 0.18, 0.24, 0.20]},
                     {"op": "random", "sd": 0.02, "range": [4, 4, 1], "seed": 7, "min": 0.05, "max": 0.35}],
            "NTG": [{"op": "constant", "value": 1.0}],
            "PERMX": [{"op": "transform", "source": "PORO", "a": -1.0, "b": 14.0}],
            "PERMY": [{"op": "copy", "source": "PERMX", "factor": 1.0}],
            "PERMZ": [{"op": "copy", "source": "PERMX", "factor": 0.1}],
        },
        "regions": {name_: [{"op": "constant", "value": 1}] for name_ in REGIONS},
        "fluids": [{"model": "live", "api": 35.0, "gas_gravity": 0.75, "temperature": from_field(180.0, "temperature", units),
                    "pb": f(3000.0, "pressure"), "co": from_field(1.0e-5, "compressibility", units)}],
        "rock": {"cr": from_field(4.0e-6, "compressibility", units), "p_ref": f(3500.0, "pressure")},
        "scal": [{"swc": 0.2, "swcr": 0.2, "sowcr": 0.2, "sgcr": 0.05, "sogcr": 0.1,
                  "krw_max": 0.4, "kro_max": 0.9, "krg_max": 0.8, "nw": 2.0, "now": 2.0, "ng": 2.0, "nog": 2.0,
                  "pc_entry": 0.0, "pc_lambda": 2.0}],
        "equil": [{"datum": f(8050.0, "length"), "pressure": f(3600.0, "pressure"),
                   "woc": f(8110.0, "length"), "goc": f(7990.0, "length"), "pc_woc": 0.0, "pc_goc": 0.0}],
        "wells": [
            {"name": "PROD1", "kind": "producer", "phase": "OIL", "group": "G1",
             "trajectory": [[6, 6, 1], [6, 6, 3]], "diameter": f(0.5, "length"), "skin": 0.0},
            {"name": "INJ1", "kind": "injector", "phase": "WATER", "group": "G1",
             "trajectory": [[13, 13, 3], [13, 13, 5]], "diameter": f(0.5, "length"), "skin": 0.0},
        ],
        "schedule": {
            "end": "2030-01-01", "report": "monthly", "history_end": "2028-01-01",
            "events": [
                {"date": "2025-01-01", "well": "PROD1", "control": "ORAT", "rate": f(2000.0, "liquid"), "bhp": f(1500.0, "pressure")},
                {"date": "2025-01-01", "well": "INJ1", "control": "RATE", "rate": f(2500.0, "liquid"), "bhp": f(5000.0, "pressure")},
                {"date": "2027-01-01", "well": "PROD1", "control": "ORAT", "rate": f(3000.0, "liquid"), "bhp": f(1500.0, "pressure")},
            ],
        },
        "summary": list(SUMMARY),
    }


def normalized(design: dict[str, Any]) -> dict[str, Any]:
    """``design`` with every section the default has, so an editor can rely on it."""
    base = default_design(design.get("name", "MODEL"), design.get("units", "FIELD"))
    out = copy.deepcopy(design)
    for key, value in base.items():
        out.setdefault(key, copy.deepcopy(value))
    for key, value in base["grid"].items():
        out["grid"].setdefault(key, copy.deepcopy(value))
    for key in PROPERTIES:
        out["properties"].setdefault(key, copy.deepcopy(base["properties"][key]))
    for key in REGIONS:
        out["regions"].setdefault(key, [{"op": "constant", "value": 1}])
    out["schedule"].setdefault("events", [])
    return out


# --------------------------------------------------------------------------- geometry
def node_xy(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    """The x of every node column and the y of every node row."""
    dx, dy = _widths(grid, "dx", "nx"), _widths(grid, "dy", "ny")
    return np.concatenate([[0.0], np.cumsum(dx)]), np.concatenate([[0.0], np.cumsum(dy)])


def _widths(grid: dict, key: str, count: str) -> np.ndarray:
    n = int(grid[count])
    values = grid[key]
    arr = np.full(n, float(values)) if np.isscalar(values) else np.asarray(values, float)
    if arr.size != n:
        raise DesignError(f"{key} has {arr.size} widths for {n} {'columns' if key == 'dx' else 'rows'}")
    if np.any(arr <= 0):
        raise DesignError(f"every {key} must be positive")
    return arr


def thicknesses(grid: dict) -> np.ndarray:
    return _widths(grid, "dz", "nz")


def top_surface(top: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""The structural top at the nodes ``(x, y)`` (2-D arrays), depth positive down:

    - ``flat``: :math:`z = z_0`;
    - ``tilted``: :math:`z = z_0 + x \tan\alpha_x + y \tan\alpha_y` with the dips in degrees;
    - ``dome``: :math:`z = z_0 + r_d (1 - e^{-\rho^2})` with :math:`\rho` the distance to the
      crest over ``radius`` times the model's half-diagonal, :math:`r_d` the relief;
    - ``anticline``: the same with :math:`\rho` the distance to the axis through the crest at
      ``azimuth`` degrees from the y axis;
    - ``table``: depths given per node (``depths``, ``ny + 1`` rows of ``nx + 1``).

    Dips are added to a dome or an anticline too, so a structure can plunge.
    """
    kind = top.get("kind", "flat")
    z0 = float(top.get("depth", 0.0))
    if kind == "table":
        z = np.asarray(top["depths"], float)
        if z.shape != x.shape:
            raise DesignError(f"the top surface table is {z.shape}, the nodes {x.shape}")
        return z
    lx, ly = float(x.max() - x.min()) or 1.0, float(y.max() - y.min()) or 1.0
    z = np.full(x.shape, z0)
    if kind in ("dome", "anticline"):
        cx, cy = x.min() + float(top.get("x0", 0.5)) * lx, y.min() + float(top.get("y0", 0.5)) * ly
        scale = max(float(top.get("radius", 0.6)), 1e-3) * 0.5 * math.hypot(lx, ly)
        if kind == "dome":
            rho2 = ((x - cx) ** 2 + (y - cy) ** 2) / scale ** 2
        else:
            az = math.radians(float(top.get("azimuth", 0.0)))
            dist = (x - cx) * math.cos(az) - (y - cy) * math.sin(az)
            rho2 = dist ** 2 / scale ** 2
        z = z + float(top.get("relief", 0.0)) * (1.0 - np.exp(-rho2))
    elif kind not in ("flat", "tilted"):
        raise DesignError(f"unknown top surface {kind!r}")
    z = z + x * math.tan(math.radians(float(top.get("dip_x", 0.0)))) \
          + y * math.tan(math.radians(float(top.get("dip_y", 0.0))))
    return z


def fault_offsets(grid: dict) -> tuple[np.ndarray, list[dict]]:
    r"""The vertical displacement of every column of cells ``(ny, nx)`` and the fault faces.

    A fault is a trace through grid nodes — ``nodes``, 0-based ``[a, b]`` with
    :math:`0 \le a \le n_x`, :math:`0 \le b \le n_y`, as drawn (a segment between two nodes
    is followed along the cell edges nearest to it, :func:`staircase`) — a ``throw`` and the
    ``downthrown`` side (``"left"`` or ``"right"`` of the trace as it runs). The trace is extended along its end segments to the
    edge of the model; a cell column whose centre lies on the downthrown side takes the throw.
    The faces are what ``FAULTS`` lists: the cell faces the trace runs along.
    """
    nx, ny = int(grid["nx"]), int(grid["ny"])
    offset = np.zeros((ny, nx))
    faces: list[dict] = []
    ci, cj = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5)
    for fault in grid.get("faults", []):
        drawn = [tuple(map(int, n)) for n in fault.get("nodes", [])]
        if len(drawn) < 2:
            continue
        if any(not (0 <= a <= nx and 0 <= b <= ny) for a, b in drawn):
            raise DesignError(f"fault {fault.get('name')}: a node lies outside the grid")
        nodes = staircase(drawn)
        for (a0, b0), (a1, b1) in zip(nodes, nodes[1:]):
            if a0 == a1:      # a segment along y at node column a: the face between cells a-1 and a
                if 1 <= a0 <= nx - 1:
                    faces.append({"name": fault["name"], "i": a0, "j": min(b0, b1) + 1, "face": "X"})
            elif 1 <= b0 <= ny - 1:   # along x at node row b: between rows b-1 and b
                faces.append({"name": fault["name"], "i": min(a0, a1) + 1, "j": b0, "face": "Y"})
        # the side of every cell centre: of the nearest segment of the extended trace
        pts = np.asarray(nodes, float)
        head = pts[0] - (pts[1] - pts[0]) * (nx + ny)
        tail = pts[-1] + (pts[-1] - pts[-2]) * (nx + ny)
        poly = np.vstack([head, pts, tail])
        best = np.full(ci.shape, np.inf)
        side = np.zeros(ci.shape)
        for p, q in zip(poly[:-1], poly[1:]):
            d = q - p
            t = np.clip(((ci - p[0]) * d[0] + (cj - p[1]) * d[1]) / (d @ d), 0.0, 1.0)
            dist = np.hypot(ci - (p[0] + t * d[0]), cj - (p[1] + t * d[1]))
            cross = d[0] * (cj - p[1]) - d[1] * (ci - p[0])      # > 0: left of the segment
            closer = dist < best - 1e-12
            best = np.where(closer, dist, best)
            side = np.where(closer, np.sign(cross), side)
        down = side > 0 if fault.get("downthrown", "right") == "left" else side < 0
        offset = offset + np.where(down, float(fault.get("throw", 0.0)), 0.0)
    return offset, faces


def staircase(nodes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """A trace through grid nodes as unit steps along cell edges: each segment walked from
    node to node, always stepping along the axis that keeps the walk nearest the straight
    segment (a corner-point fault runs along cell faces)."""
    out = [nodes[0]]
    for (a1, b1) in nodes[1:]:
        a0, b0 = out[-1]
        da, db = a1 - a0, b1 - b0
        steps = abs(da) + abs(db)
        a, b = a0, b0
        for _ in range(steps):
            # the step whose landing point lies closest to the segment
            options = []
            if a != a1:
                options.append((a + (1 if a1 > a else -1), b))
            if b != b1:
                options.append((a, b + (1 if b1 > b else -1)))
            def off(p):
                return abs(db * (p[0] - a0) - da * (p[1] - b0))
            a, b = min(options, key=off)
            out.append((a, b))
    return out


def actnum(design: dict) -> np.ndarray:
    """``(nz, ny, nx)``: 1 where a cell is active. ``inactive`` lists boxes
    ``{"i": [a, b], "j": [a, b], "k": [a, b]}`` (1-based, inclusive) or painted columns
    ``{"cells": [[i, j], …], "k": [a, b]}``."""
    g = design["grid"]
    act = np.ones((int(g["nz"]), int(g["ny"]), int(g["nx"])), np.int32)
    for item in g.get("inactive", []):
        act[_selection(item, act.shape)] = 0
    return act


def corner_point(design: dict) -> dict[str, np.ndarray]:
    """``COORD`` ``((ny+1)(nx+1), 6)``, ``ZCORN`` ``(2 nz, 2 ny, 2 nx)``, ``ACTNUM`` and the
    corners of every cell ``(nz, ny, nx, 8, 3)`` in VTK order (depth positive down)."""
    g = design["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    xs, ys = node_xy(g)
    X, Y = np.meshgrid(xs, ys)
    top = top_surface(g.get("top", {}), X, Y)                             # (ny+1, nx+1)
    h = thicknesses(g)
    depth_at = np.concatenate([[0.0], np.cumsum(h)])                      # (nz+1,)
    shift, _ = fault_offsets(g)                                           # (ny, nx)
    zc = np.empty((2 * nz, 2 * ny, 2 * nx))
    for k in range(nz):
        for side, dk in ((0, depth_at[k]), (1, depth_at[k + 1])):
            for dj in (0, 1):
                for di in (0, 1):
                    zc[2 * k + side, dj::2, di::2] = top[dj:dj + ny, di:di + nx] + dk + shift
    zmin, zmax = float(zc.min()) - 1.0, float(zc.max()) + 1.0
    coord = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, zmin),
                             X.ravel(), Y.ravel(), np.full(X.size, zmax)])
    corners = np.empty((nz, ny, nx, 8, 3))
    for v, (di, dj, side) in enumerate(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))):
        corners[..., v, 0] = X[None, dj:dj + ny, di:di + nx]
        corners[..., v, 1] = Y[None, dj:dj + ny, di:di + nx]
        corners[..., v, 2] = zc[side::2, dj::2, di::2]
    return {"coord": coord, "zcorn": zc, "actnum": actnum(design), "corners": corners}


def cell_centres(design: dict) -> np.ndarray:
    """``(nz, ny, nx, 3)``: the centre of every cell."""
    return corner_point(design)["corners"].mean(axis=3)


# --------------------------------------------------------------------------- recipes
def _selection(op: dict, shape: tuple[int, int, int]) -> tuple:
    """The cells an operation addresses, as an index into a ``(nz, ny, nx)`` array."""
    nz, ny, nx = shape
    mask = np.zeros(shape, bool)
    k0, k1 = (op.get("k") or [1, nz])
    if "cells" in op:
        for i, j in op["cells"]:
            if 1 <= i <= nx and 1 <= j <= ny:
                mask[k0 - 1:k1, j - 1, i - 1] = True
        return np.nonzero(mask)
    i0, i1 = (op.get("i") or [1, nx])
    j0, j1 = (op.get("j") or [1, ny])
    mask[k0 - 1:k1, j0 - 1:j1, i0 - 1:i1] = True
    return np.nonzero(mask)


def gaussian_field(shape: tuple[int, int, int], ranges: tuple[float, float, float], seed: int) -> np.ndarray:
    r"""A stationary Gaussian field of unit variance with correlation ranges ``(i, j, k)``
    in cells: white noise smoothed by a Gaussian kernel of standard deviation
    :math:`r/2` per axis, rescaled to unit variance (a range of 0 leaves an axis uncorrelated)."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(shape)
    ri, rj, rk = (max(float(r), 0.0) for r in ranges)
    field = gaussian_filter(noise, sigma=(rk / 2.0, rj / 2.0, ri / 2.0), mode="wrap")
    sd = float(field.std())
    return (field - field.mean()) / (sd if sd > 0 else 1.0)


def evaluate(design: dict, name: str, done: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """The ``(nz, ny, nx)`` array of property or region ``name`` from its recipe. ``done``
    holds arrays already evaluated (a recipe may read another property or a region)."""
    done = {} if done is None else done
    if name in done:
        return done[name]
    g = design["grid"]
    shape = (int(g["nz"]), int(g["ny"]), int(g["nx"]))
    recipe = (design["properties"].get(name) if name in PROPERTIES else design["regions"].get(name)) or []
    integer = name in REGIONS
    arr = np.zeros(shape) if not integer else np.ones(shape)
    centres = None
    for op in recipe:
        kind = op.get("op")
        mode = op.get("mode", "set")

        def apply(where, values):
            if mode == "multiply":
                arr[where] = arr[where] * values
            elif mode == "add":
                arr[where] = arr[where] + values
            else:
                arr[where] = values

        everywhere = (slice(None),) * 3
        if kind == "constant":
            apply(everywhere, float(op["value"]))
        elif kind == "layers":
            values = np.asarray(op["values"], float)
            if values.size != shape[0]:
                raise DesignError(f"{name}: 'layers' gives {values.size} values for {shape[0]} layers")
            apply(everywhere, np.broadcast_to(values[:, None, None], shape))
        elif kind == "trend":
            if centres is None:
                centres = cell_centres(design)
            axis = {"x": 0, "y": 1, "depth": 2}[op.get("axis", "depth")]
            coord = centres[..., axis]
            apply(everywhere, float(op.get("at", 0.0)) + float(op.get("gradient", 0.0)) * (coord - coord.min()))
        elif kind == "random":
            field = gaussian_field(shape, tuple(op.get("range", [3, 3, 1])), int(op.get("seed", 0)))
            if "mean" in op:
                apply(everywhere, float(op["mean"]) + float(op.get("sd", 0.0)) * field)
            else:
                arr[...] = arr + float(op.get("sd", 0.0)) * field
        elif kind == "transform":
            src = evaluate(design, op["source"], done)
            apply(everywhere, 10.0 ** (float(op.get("a", 0.0)) + float(op.get("b", 1.0)) * src))
        elif kind == "copy":
            apply(everywhere, evaluate(design, op["source"], done) * float(op.get("factor", 1.0)))
        elif kind in ("box", "cells"):
            apply(_selection(op, shape), float(op["value"]))
        elif kind == "region":
            reg = evaluate(design, op["array"], done)
            apply(reg == int(op["id"]), float(op["value"]))
        else:
            raise DesignError(f"{name}: unknown operation {kind!r}")
        if "min" in op or "max" in op:
            arr[...] = np.clip(arr, op.get("min", -np.inf), op.get("max", np.inf))
    if integer:
        arr = np.rint(arr).astype(np.int32)
        if arr.min() < 1:
            raise DesignError(f"{name}: region numbers start at 1")
    done[name] = arr
    return arr


def evaluate_all(design: dict) -> dict[str, np.ndarray]:
    done: dict[str, np.ndarray] = {}
    for name in (*PROPERTIES, *REGIONS):
        evaluate(design, name, done)
    return done


# --------------------------------------------------------------------------- rock and fluids
def scal_tables(s: dict, rows: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """``SWOF`` and ``SGOF`` of one saturation region (see the module docstring)."""
    swc, swcr, sowcr = float(s["swc"]), float(s.get("swcr", s["swc"])), float(s["sowcr"])
    sgcr, sogcr = float(s.get("sgcr", 0.0)), float(s.get("sogcr", sowcr))
    if not 0 <= swc <= swcr < 1 - sowcr:
        raise DesignError("saturation end points: need 0 <= Swc <= Swcr < 1 - Sowcr")
    sw = np.unique(np.concatenate([np.linspace(swc, 1.0, rows), [swcr, 1.0 - sowcr]]))
    span_w = 1.0 - swcr - sowcr
    nw = np.clip((sw - swcr) / span_w, 0.0, 1.0)
    no = np.clip((1.0 - sw - sowcr) / span_w, 0.0, 1.0)
    krw = float(s["krw_max"]) * nw ** float(s["nw"])
    krow = float(s["kro_max"]) * no ** float(s["now"])
    pc = np.zeros_like(sw)
    if float(s.get("pc_entry", 0.0)) > 0:
        se = np.clip((sw - swc) / (1.0 - swc), 1e-3, 1.0)
        pc = np.minimum(float(s["pc_entry"]) * se ** (-1.0 / float(s.get("pc_lambda", 2.0))),
                        20.0 * float(s["pc_entry"]))
        pc[-1] = 0.0 if sw[-1] >= 1.0 else pc[-1]
    swof = np.column_stack([sw, krw, krow, pc])
    sg_max = 1.0 - swc
    sg = np.unique(np.concatenate([np.linspace(0.0, sg_max, rows), [sgcr, max(sg_max - sogcr, sgcr)]]))
    span_g = 1.0 - swc - sgcr - sogcr
    if span_g <= 0:
        raise DesignError("gas end points: need Swc + Sgcr + Sogcr < 1")
    ng = np.clip((sg - sgcr) / span_g, 0.0, 1.0)
    nog = np.clip((1.0 - swc - sg - sogcr) / span_g, 0.0, 1.0)
    sgof = np.column_stack([sg, float(s["krg_max"]) * ng ** float(s["ng"]),
                            float(s["kro_max"]) * nog ** float(s["nog"]), np.zeros_like(sg)])
    return swof, sgof


def pvt_tables(design: dict) -> dict[str, list]:
    """The fluid tables of every fluid region, in the design's units: ``PVTO`` (or ``PVDO``),
    ``PVDG``, ``PVTW``, ``DENSITY``."""
    units = design["units"]
    live = "GAS" in design["phases"] and "DISGAS" in design["phases"]
    p_max = max(to_field(float(e["pressure"]), "pressure", units) for e in design["equil"]) * 1.5
    out: dict[str, list] = {"PVTO": [], "PVDO": [], "PVDG": [], "PVTW": [], "DENSITY": []}
    for fl in design["fluids"]:
        api, gg = float(fl["api"]), float(fl["gas_gravity"])
        temp = to_field(float(fl["temperature"]), "temperature", units)
        co = to_field(float(fl.get("co", 1e-5)), "compressibility", units)
        pb = to_field(float(fl["pb"]), "pressure", units)
        if live:
            blocks = corr.live_oil_tables(api, gg, temp, pb, p_max, co=co)
            out["PVTO"].append([_convert_pvto(b, units) for b in blocks])
        else:
            out["PVDO"].append(_convert(corr.dead_oil_table(api, temp, p_max, co=co), ("pressure", None, None), units))
        if "GAS" in design["phases"]:
            out["PVDG"].append(_convert(corr.gas_table(gg, temp, p_max), ("pressure", "bg", None), units))
        w = corr.water(p_max / 1.5, temp)
        out["PVTW"].append(np.array([[from_field(w["p_ref"], "pressure", units), w["bw"],
                                      from_field(w["cw"], "compressibility", units), w["mu"], 0.0]]))
        rho = corr.densities(api, gg)
        out["DENSITY"].append(np.array([[from_field(v, "density", units) for v in rho]]))
    return out


def _convert(table: np.ndarray, quantities: tuple, units: str) -> np.ndarray:
    out = np.array(table, float)
    for c, q in enumerate(quantities):
        if q is not None:
            out[:, c] = from_field(out[:, c], q, units)
    return out


def _convert_pvto(block: np.ndarray, units: str) -> np.ndarray:
    out = np.array(block, float)
    out[:, 0] = from_field(out[:, 0], "gor", units)
    out[:, 1] = from_field(out[:, 1], "pressure", units)
    return out


def initial_rs(design: dict, pvt_index: int) -> float:
    """The dissolved-gas ratio of fluid region ``pvt_index`` at its bubble point, in the
    deck's gas-oil-ratio unit."""
    units = design["units"]
    fl = design["fluids"][pvt_index]
    rs = corr.rs_standing(to_field(float(fl["pb"]), "pressure", units), float(fl["api"]),
                          float(fl["gas_gravity"]), to_field(float(fl["temperature"]), "temperature", units))
    return float(from_field(float(rs) / 1000.0, "gor", units))


# --------------------------------------------------------------------------- wells
def completions(design: dict, well: dict) -> list[tuple[int, int, int, str]]:
    """``(i, j, k, direction)`` of every cell the well's trajectory passes through, in
    order, 1-based. The trajectory is a polyline of points ``[i, j, k]`` in cell units (a
    point at the centre of cell ``(i, j, k)`` is ``[i, j, k]``); the direction of a cell is the
    axis its segment mostly runs along."""
    g = design["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    pts = np.asarray(well.get("trajectory") or [], float)
    if pts.ndim != 2 or pts.shape[0] < 1 or pts.shape[1] != 3:
        raise DesignError(f"well {well.get('name')}: a trajectory is a list of [i, j, k] points")
    if pts.shape[0] == 1:
        pts = np.vstack([pts, pts])
    out: list[tuple[int, int, int, str]] = []
    seen: set[tuple[int, int, int]] = set()
    for p, q in zip(pts[:-1], pts[1:]):
        d = q - p
        axis = "XYZ"[int(np.argmax(np.abs(d)))] if np.any(d) else "Z"
        n = max(2, int(np.ceil(np.abs(d).max() * 8)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            i, j, k = (int(np.clip(np.floor(v + 0.5), 1, m)) for v, m in zip(p + t * d, (nx, ny, nz)))
            if (i, j, k) not in seen:
                seen.add((i, j, k))
                out.append((i, j, k, axis))
    act = actnum(design)
    out = [c for c in out if act[c[2] - 1, c[1] - 1, c[0] - 1]]
    if not out:
        raise DesignError(f"well {well.get('name')}: its trajectory passes through no active cell")
    return out


# --------------------------------------------------------------------------- the schedule
def _date(text: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(text)[:10])
    except ValueError as exc:
        raise DesignError(f"{text!r} is not a date (YYYY-MM-DD)") from exc


def _eclipse_date(day: dt.date) -> str:
    return f"{day.day} '{MONTHS[day.month - 1]}' {day.year}"


def report_dates(design: dict) -> list[dt.date]:
    """The report dates: the cadence from the start to the end, every control change and the
    history | prediction boundary among them."""
    sched = design["schedule"]
    start, end = _date(design["start"]), _date(sched["end"])
    if end <= start:
        raise DesignError("the schedule must end after the start date")
    step = {"monthly": 1, "quarterly": 3, "yearly": 12}.get(sched.get("report", "monthly"))
    if step is None:
        raise DesignError("the report cadence is monthly, quarterly or yearly")
    dates = set()
    y, m = start.year, start.month
    while True:
        m += step
        y, m = y + (m - 1) // 12, (m - 1) % 12 + 1
        day = dt.date(y, m, 1)
        if day >= end:
            break
        dates.add(day)
    dates.add(end)
    for ev in sched.get("events", []):
        day = _date(ev["date"])
        if start < day < end:
            dates.add(day)
    if sched.get("history_end"):
        day = _date(sched["history_end"])
        if start < day < end:
            dates.add(day)
    return sorted(dates)


def history_end_days(design: dict) -> float | None:
    """The history | prediction boundary in days from the start, or ``None``."""
    text = design["schedule"].get("history_end")
    if not text:
        return None
    return float((_date(text) - _date(design["start"])).days)


def _control_rows(design: dict, events: list[dict]) -> str:
    """The control keywords of the events of one date."""
    wells = {w["name"]: w for w in design["wells"]}
    prod, inj, status = [], [], []
    for ev in events:
        w = wells.get(ev["well"])
        if w is None:
            raise DesignError(f"an event names the well {ev['well']!r}, which the design does not have")
        ctrl = str(ev["control"]).upper()
        if ctrl in STATUS_EVENTS:
            status.append([w["name"], ctrl])
            continue
        rate = ev.get("rate")
        bhp = ev.get("bhp")
        if w["kind"] == "producer":
            if ctrl not in PRODUCER_CONTROLS:
                raise DesignError(f"{w['name']}: a producer is controlled by {', '.join(PRODUCER_CONTROLS)}")
            row: list[Any] = [w["name"], "OPEN", ctrl, None, None, None, None, None, None]
            slot = {"ORAT": 3, "WRAT": 4, "GRAT": 5, "LRAT": 6, "RESV": 7}.get(ctrl)
            if slot is not None:
                if rate is None:
                    raise DesignError(f"{w['name']}: {ctrl} control needs a rate")
                row[slot] = float(rate)
            row[8] = float(bhp) if bhp is not None else (None if ctrl != "BHP" else float(rate))
            prod.append(row)
        else:
            if ctrl not in INJECTOR_CONTROLS:
                raise DesignError(f"{w['name']}: an injector is controlled by RATE or BHP")
            phase = w.get("phase", "WATER")
            row = [w["name"], phase, "OPEN", ctrl, None if ctrl == "BHP" else float(rate), None,
                   float(bhp) if bhp is not None else (float(rate) if ctrl == "BHP" else None)]
            inj.append(row)
    text = ""
    if prod:
        text += block_text("WCONPROD", R.format_records(prod))
    if inj:
        text += block_text("WCONINJE", R.format_records(inj))
    if status:
        text += block_text("WELOPEN", R.format_records(status))
    return text


# --------------------------------------------------------------------------- the deck
@dataclass
class Generated:
    """A design's deck tree, and what it says about the model besides."""

    tree: dict[str, bytes]
    deck_name: str
    settings: dict[str, Any]


def check(design: dict) -> list[str]:
    """What is wrong with a design, in plain words (empty when it can be generated)."""
    problems: list[str] = []
    try:
        generate(design)
    except DesignError as exc:
        problems.append(str(exc))
    return problems


def generate(design: dict) -> Generated:
    """The deck tree of a design (see the module docstring)."""
    d = normalized(design)
    units = d["units"]
    if units not in ("FIELD", "METRIC"):
        raise DesignError("units are FIELD or METRIC")
    name = "".join(ch for ch in str(d["name"]).upper() if ch.isalnum() or ch in "-_") or "MODEL"
    g = d["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    if min(nx, ny, nz) < 1:
        raise DesignError("the grid needs at least one cell along every axis")
    phases = [p for p in ("OIL", "WATER", "GAS", "DISGAS") if p in d["phases"]]
    if "OIL" not in phases or "WATER" not in phases:
        raise DesignError("the model needs oil and water")
    if "DISGAS" in phases and "GAS" not in phases:
        raise DesignError("dissolved gas needs the gas phase")
    gas = "GAS" in phases

    arrays = evaluate_all(d)
    geom = corner_point(d)
    n_sat = int(arrays["SATNUM"].max())
    n_pvt = int(arrays["PVTNUM"].max())
    n_eql = int(arrays["EQLNUM"].max())
    n_fip = int(arrays["FIPNUM"].max())
    if len(d["scal"]) < n_sat:
        raise DesignError(f"SATNUM reaches {n_sat} but {len(d['scal'])} saturation region(s) are described")
    if len(d["fluids"]) < n_pvt:
        raise DesignError(f"PVTNUM reaches {n_pvt} but {len(d['fluids'])} fluid region(s) are described")
    if len(d["equil"]) < n_eql:
        raise DesignError(f"EQLNUM reaches {n_eql} but {len(d['equil'])} equilibration region(s) are described")
    if np.any(arrays["PORO"][geom["actnum"] > 0] <= 0):
        raise DesignError("an active cell has no porosity")
    for key in ("PERMX", "PERMY", "PERMZ"):
        if np.any(arrays[key] < 0):
            raise DesignError(f"{key} must not be negative")

    wells = d["wells"]
    names = [w["name"] for w in wells]
    if len(set(names)) != len(names):
        raise DesignError("two wells share a name")
    conns = {w["name"]: completions(d, w) for w in wells}

    inc = "INCLUDE"
    files: dict[str, str] = {}

    # ---- GRID: geometry and properties, as a modelling package exports them -------------
    grid_file = f"{inc}/GRID/{name}_GRID.GRDECL"
    files[grid_file] = (
        f"-- Grid of {name}: {nx} x {ny} x {nz} corner-point cells, generated by gNN\n\n"
        + block_text("SPECGRID", f" {nx} {ny} {nz} 1 F /")
        + block_text("COORD", R.format_array(geom["coord"], per_line=6))
        + block_text("ZCORN", R.format_array(geom["zcorn"], per_line=8))
        + block_text("ACTNUM", R.format_array(geom["actnum"], per_line=20, integer=True))
    )
    props_file = f"{inc}/GRID/{name}_PROPS.GRDECL"
    files[props_file] = "-- Rock properties, generated by gNN\n\n" + "".join(
        block_text(key, R.format_array(arrays[key], digits=5)) for key in PROPERTIES)
    regions_file = f"{inc}/REGIONS/{name}_REGIONS.GRDECL"
    files[regions_file] = "-- Region numbers, generated by gNN\n\n" + "".join(
        block_text(key, R.format_array(arrays[key], per_line=20, integer=True)) for key in REGIONS)

    # ---- PROPS: fluids and rock–fluid --------------------------------------------------
    pvt = pvt_tables(d)
    pvt_text = ""
    if pvt["PVTO"]:
        pvt_text += block_text("PVTO", R.format_pvto(pvt["PVTO"]))
    if pvt["PVDO"]:
        pvt_text += block_text("PVDO", R.format_tables(pvt["PVDO"]))
    if pvt["PVDG"]:
        pvt_text += block_text("PVDG", R.format_tables(pvt["PVDG"]))
    pvt_text += block_text("PVTW", R.format_tables(pvt["PVTW"]))
    pvt_text += block_text("DENSITY", R.format_tables(pvt["DENSITY"]))
    rock = d["rock"]
    pvt_text += block_text("ROCK", R.format_tables([np.array([[float(rock["p_ref"]), float(rock["cr"])]])] * n_pvt))
    files[f"{inc}/PROPS/{name}_PVT.INC"] = "-- Fluid and rock properties, generated by gNN\n\n" + pvt_text
    swof, sgof = zip(*(scal_tables(s) for s in d["scal"][:n_sat]))
    scal_text = block_text("SWOF", R.format_tables(swof))
    if gas:
        scal_text += block_text("SGOF", R.format_tables(sgof))
    files[f"{inc}/PROPS/{name}_SCAL.INC"] = "-- Saturation functions, generated by gNN\n\n" + scal_text

    # ---- SOLUTION ---------------------------------------------------------------------
    eq_rows, rsvd = [], []
    live = gas and "DISGAS" in phases
    for e_index, e in enumerate(d["equil"][:n_eql]):
        eq_rows.append([float(e["datum"]), float(e["pressure"]), float(e["woc"]), float(e.get("pc_woc", 0.0)),
                        float(e.get("goc", e["datum"])) if gas else None, float(e.get("pc_goc", 0.0)) if gas else None,
                        1 if live else None, None, None])
        if live:
            cells = arrays["EQLNUM"] == e_index + 1
            pvt_of = int(np.bincount(arrays["PVTNUM"][cells].ravel()).argmax()) - 1 if cells.any() else 0
            rs = initial_rs(d, max(pvt_of, 0))
            z = geom["corners"][..., 2]
            rsvd.append(np.array([[float(z.min()) - 100.0, rs], [float(z.max()) + 100.0, rs]]))

    # ---- SCHEDULE ---------------------------------------------------------------------
    start = _date(d["start"])
    events = sorted(d["schedule"].get("events", []), key=lambda ev: _date(ev["date"]))
    for ev in events:
        if _date(ev["date"]) < start:
            raise DesignError(f"an event of {ev['well']} is dated before the start")
    welspecs = [[w["name"], w.get("group", "G1"), conns[w["name"]][0][0], conns[w["name"]][0][1], None,
                 w.get("phase", "OIL") if w["kind"] == "producer" else w.get("phase", "WATER")] for w in wells]
    compdat = [[w["name"], i, j, k, k, "OPEN", None, None, float(w.get("diameter", 0.5)), None,
                float(w.get("skin", 0.0)), None, direction]
               for w in wells for (i, j, k, direction) in conns[w["name"]]]
    sched = block_text("RPTRST", " 'BASIC=2' /")
    if wells:
        sched += block_text("WELSPECS", R.format_records(welspecs))
        sched += block_text("COMPDAT", R.format_records(compdat))
    first = [ev for ev in events if _date(ev["date"]) <= start]
    sched += _control_rows(d, first)
    for day in report_dates(d):
        sched += block_text("DATES", f" {_eclipse_date(day)} /\n/")
        now = [ev for ev in events if _date(ev["date"]) == day]
        if now:
            sched += _control_rows(d, now)
    files[f"{inc}/SCHEDULE/{name}_SCHEDULE.INC"] = "-- Wells and their controls, generated by gNN\n\n" + sched

    # ---- the main deck ---------------------------------------------------------------------
    n_wells = max(len(wells), 1)
    max_conn = max((len(c) for c in conns.values()), default=1)
    groups = {w.get("group", "G1") for w in wells} or {"G1"}
    text = (
        f"-- {name}: generated by gNN from its design\n\n"
        + block_text("RUNSPEC")
        + block_text("TITLE", str(d.get("title") or name))
        + block_text("DIMENS", f" {nx} {ny} {nz} /")
        + "".join(f"{p}\n\n" for p in phases)
        + f"{units}\n\n"
        + block_text("START", f" {_eclipse_date(start)} /")
        + block_text("TABDIMS", f" {n_sat} {n_pvt} 60 60 {n_fip} 60 /")
        + block_text("EQLDIMS", f" {n_eql} /")
        + block_text("REGDIMS", f" {n_fip} /")
        + block_text("WELLDIMS", f" {n_wells} {max_conn} {len(groups)} {n_wells} /")
        + "UNIFIN\n\nUNIFOUT\n\n"
        + block_text("GRID")
        + "INIT\n\n"
        + block_text("INCLUDE", f" '{grid_file}' /")
        + block_text("INCLUDE", f" '{props_file}' /")
    )
    _, faces = fault_offsets(g)
    if faces:
        rows = [[f["name"], f["i"], f["i"], f["j"], f["j"], 1, nz, f["face"]] for f in faces]
        text += block_text("FAULTS", R.format_records(rows))
        text += block_text("MULTFLT", R.format_records(
            [[f["name"], float(f.get("mult", 1.0))] for f in g.get("faults", []) if f.get("name")]))
    text += (
        block_text("PROPS")
        + block_text("INCLUDE", f" '{inc}/PROPS/{name}_PVT.INC' /")
        + block_text("INCLUDE", f" '{inc}/PROPS/{name}_SCAL.INC' /")
        + block_text("REGIONS")
        + block_text("INCLUDE", f" '{regions_file}' /")
        + block_text("SOLUTION")
        + block_text("EQUIL", R.format_records(eq_rows, terminate=False))
    )
    if rsvd:
        text += block_text("RSVD", R.format_tables(rsvd))
    text += block_text("SUMMARY")
    for key in d.get("summary", SUMMARY):
        text += f"{key}\n\n"
    for key in WELL_SUMMARY:
        text += block_text(key, "/")
    text += (
        block_text("SCHEDULE")
        + block_text("INCLUDE", f" '{inc}/SCHEDULE/{name}_SCHEDULE.INC' /")
        + "END\n"
    )
    files[f"{name}.DATA"] = text

    settings: dict[str, Any] = {"design": d}
    boundary = history_end_days(d)
    if boundary is not None:
        settings["history_end_days"] = boundary
    return Generated(tree={path: body.encode("latin-1") for path, body in files.items()},
                     deck_name=f"{name}.DATA", settings=settings)


def dumps(design: dict) -> str:
    return json.dumps(design, indent=1, sort_keys=False)
