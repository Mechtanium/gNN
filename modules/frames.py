r"""
The messages the workflow streams to the page, and their encoding.

Every output item of every operation is the 8-tuple

``(kind, iteration, total, pde, ic, data, well, payload)``

where ``kind`` names the message, the five floats carry the losses of a
``loss`` message (and whatever a page charts of the other kinds — see each
constructor), and ``payload`` is JSON text. Arrays ride inside the JSON as
base64: the grid corners as float32, the cell addresses as int16, the cell
states as float16 — a state value is clipped to its field's fixed colour range
before the cast, so the legend never has to move. A numeric slot never carries
a NaN or an infinity: the page's transport is JSON, which has neither, so a
non-finite value is sent as zero and named in the payload's ``nonfinite``.

======== ==================================================================
kind     what it says
======== ==================================================================
stage    what the job is doing and how far along it is
log      a line of the simulator's or the pipeline's own log
diag     one finding of the deck check (error, warning, note)
grid     the geometry, the snapshot times, the fixed colour ranges, the wells
state    every field's value in every cell at one snapshot time
loss     the loss and its components at one iteration
well     one point of a well's record: simulator, gNN and observed values
sample   one simulator run of a family: started, finished or failed
metrics  the numbers a path ends with
posterior the inverted properties: mean, deviation, interval, samples
error    why a job failed, instead of a bare failure
======== ==================================================================

Sizes on SPE-2 (9 375 cells): one ``grid`` part is ~1.2 MB, one ``state``
message (seven fields) ~175 KB, a ``loss`` message ~120 bytes; the grid is
split into parts of at most ``GRID_PART_BYTES`` so a 44 000-cell field still
fits comfortably under the transport's message limit.
"""

from __future__ import annotations

import base64
import json
import math
from typing import Any, Iterator, Sequence

import numpy as np

KINDS = ("stage", "log", "diag", "grid", "state", "loss", "well", "sample", "metrics",
         "posterior", "error")

#: The seven cell fields a ``state`` message carries, in this order.
FIELDS = ("S_w", "S_o", "S_g", "p_o", "p_w", "p_g", "R_so")

GRID_PART_BYTES = 2_500_000
#: The largest magnitude a float16 holds (65 504), with a margin: a wider field rides as float32.
F16_MAX = 6.0e4
#: Half-width of the legend margin, in standard deviations of the reference field.
RANGE_SIGMA = 1.5

Item = tuple[str, int, float, float, float, float, float, str]


def _b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def _unb64(text: str, dtype, shape) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=dtype).reshape(shape)


def item(kind: str, payload: dict[str, Any] | None = None, *, iteration: int = 0,
         losses: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)) -> Item:
    if kind not in KINDS:
        raise ValueError(f"unknown frame kind {kind!r}")
    slots = [float(x) for x in losses]
    bad = [i for i, x in enumerate(slots) if not math.isfinite(x)]
    if bad:
        slots = [x if math.isfinite(x) else 0.0 for x in slots]
        payload = {**(payload or {}), "nonfinite": bad}
    total, pde, ic, data, well = slots
    return (kind, int(iteration), total, pde, ic, data, well,
            "" if payload is None else json.dumps(_finite(payload), separators=(",", ":")))


def _finite(value: Any) -> Any:
    """``value`` as JSON can carry it: numpy scalars and arrays as Python numbers and
    lists, every non-finite float as ``None`` (JSON has no NaN)."""
    if isinstance(value, np.generic):
        value = value.item()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def stage(name: str, fraction: float, message: str = "") -> Item:
    """What the job is doing and how far along it is (``fraction`` in 0..1)."""
    return item("stage", {"stage": name, "message": message},
                losses=(max(0.0, min(1.0, float(fraction))), 0.0, 0.0, 0.0, 0.0))


def loss(iteration: int, total: float, pde: float, ic: float, data: float, well: float,
         *, wall_s: float, eta: float | None = None) -> Item:
    return item("loss", {"wall_s": float(wall_s), "eta": None if eta is None else float(eta)},
                iteration=iteration, losses=(total, pde, ic, data, well))


def metrics(iteration: int, values: dict[str, Any]) -> Item:
    """The numbers a path ends with (floats; a nested dict or a string is passed through)."""
    return item("metrics", dict(values), iteration=iteration)


def log(message: str, source: str = "gnn") -> Item:
    """One line of a log: ``source`` is ``"flow"`` (the simulator) or ``"gnn"``."""
    return item("log", {"source": source, "message": str(message)})


def diag(level: str, where: str, message: str) -> Item:
    """One finding of the deck check. ``level`` is ``"error"``, ``"warning"`` or ``"note"``;
    ``where`` names the section or the keyword."""
    if level not in ("error", "warning", "note"):
        raise ValueError(f"unknown diagnostic level {level!r}")
    return item("diag", {"level": level, "where": str(where), "message": str(message)})


#: The quantities of a well's record, and their units in the workflow's (FIELD) system.
WELL_QUANTITIES = {"WBHP": "psia", "WOPR": "stb/d", "WWPR": "stb/d", "WGPR": "Mscf/d",
                   "WWIR": "stb/d", "WGIR": "Mscf/d", "WGOR": "Mscf/stb", "WWCT": "-"}


SOURCES = ("simulator", "gnn", "observed")


def well(step: int, t_days: float, name: str, values: dict[str, dict[str, float | None]], *,
         phase: str = "history") -> Item:
    r"""One well at one report time: every quantity of :data:`WELL_QUANTITIES` that any source
    has there. ``values`` maps a source — ``"simulator"``, ``"gnn"`` or ``"observed"`` — to its
    quantities. The payload carries each as ``"<quantity>_<source>"`` (``"WBHP_gnn"``), so

    ``line_chart(run, x=2, y=["7.WBHP_simulator", "7.WBHP_gnn", "7.WBHP_observed"],
    where=[(0, "well"), ("7.well", "PROD")])``

    draws a well's pressure from all three; a quantity a source does not have at this time is
    absent, which a chart reads as a missing point. ``iteration`` is the report step and
    ``total`` the time in days; ``phase`` is ``"history"`` or ``"forecast"``.
    """
    payload: dict[str, Any] = {"well": str(name), "phase": phase}
    for source, quantities in values.items():
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        for quantity, value in quantities.items():
            if quantity not in WELL_QUANTITIES:
                raise ValueError(f"unknown well quantity {quantity!r}")
            if value is not None and math.isfinite(float(value)):
                payload[f"{quantity}_{source}"] = float(value)
    return item("well", payload, iteration=step, losses=(t_days, 0.0, 0.0, 0.0, 0.0))


def sample(index: int, total: int, name: str, status: str, seconds: float = 0.0) -> Item:
    """One simulator run of a family: ``status`` is ``"running"``, ``"done"``, ``"reused"``
    or ``"failed"``. ``iteration`` is the run's index; the ``total`` slot the fraction of
    the family's runs finished so far."""
    return item("sample", {"name": str(name), "status": status, "n": int(total), "seconds": float(seconds)},
                iteration=index, losses=(0.0 if total <= 0 else min(1.0, index / total), 0.0, 0.0, 0.0, 0.0))


def posterior(values: dict[str, Any]) -> Item:
    """The inverted properties: per property its name, unit, prior and posterior mean and
    deviation and the 5–95 % interval, with the posterior samples and the misfit."""
    return item("posterior", values)


def error(message: str) -> Item:
    return item("error", {"message": message})


# ------------------------------------------------------------------- ranges
def field_ranges(ref: dict[str, np.ndarray], phys: Any) -> dict[str, list[float]]:
    r"""The fixed colour range of each field: the reference field's extremes widened by
    :math:`1.5\,\sigma` on each side, saturations bounded to :math:`[0, 1]`, the
    dissolved-gas ratio to :math:`[0, \infty)`.

    ``ref`` holds the simulator's ``pres``, ``swat``, ``sgas``, ``rs`` over every
    report step and cell; the phase pressures reuse the oil-pressure range widened
    by the largest capillary offset the tables allow (``phys.P_MIN``/``P_MAX``
    already carry the padding the training ranges use).
    """
    out: dict[str, list[float]] = {}

    def widen(a: np.ndarray, lo_bound=None, hi_bound=None) -> list[float]:
        a = np.asarray(a, np.float64)
        s = float(np.nanstd(a))
        lo, hi = float(np.nanmin(a)) - RANGE_SIGMA * s, float(np.nanmax(a)) + RANGE_SIGMA * s
        if lo_bound is not None:
            lo = max(lo, lo_bound)
        if hi_bound is not None:
            hi = min(hi, hi_bound)
        if not hi > lo:
            hi = lo + 1.0
        return [lo, hi]

    sw, sg = np.asarray(ref["swat"]), np.asarray(ref["sgas"])
    out["S_w"] = widen(sw, 0.0, 1.0)
    out["S_g"] = widen(sg, 0.0, 1.0)
    out["S_o"] = widen(1.0 - sw - sg, 0.0, 1.0)
    p = widen(ref["pres"])
    lo = min(p[0], float(phys.P_MIN))
    hi = max(p[1], float(phys.P_MAX))
    out["p_o"] = [lo, hi]
    out["p_w"] = [lo, hi]
    out["p_g"] = [lo, hi]
    out["R_so"] = widen(ref["rs"], 0.0, None)
    return out


# --------------------------------------------------------------------- grid
def grid(corners: np.ndarray, ijk: np.ndarray, dims: tuple[int, int, int], times: list[float],
         ranges: dict[str, list[float]], wells: list[dict[str, Any]] | None = None, *,
         fields: Sequence[str] = FIELDS, categorical: Sequence[str] = (),
         lines: list[dict[str, Any]] | None = None) -> Iterator[Item]:
    """The geometry message(s): corners ``(n_cells, 8, 3)`` in VTK order, ``ijk``
    ``(n_cells, 3)`` zero-based, the snapshot times, the fixed ranges, and optional
    well markers. ``fields`` names what the ``state`` messages will carry (the seven
    dynamic fields unless a model preview sends its static properties), ``categorical``
    those of them that are classes (region numbers), ``lines`` free polylines (fault
    traces). Yields one part when it fits, else several with ``part``/``n_parts``."""
    corners = np.ascontiguousarray(corners, np.float32)
    ijk = np.ascontiguousarray(ijk, np.int16)
    n = int(corners.shape[0])
    per_cell = 8 * 3 * 4
    cells_per_part = max(1, (GRID_PART_BYTES * 3 // 4) // per_cell)
    n_parts = max(1, -(-n // cells_per_part))
    for k in range(n_parts):
        a, b = k * cells_per_part, min(n, (k + 1) * cells_per_part)
        payload: dict[str, Any] = {
            "n_cells": n, "dims": [int(d) for d in dims], "part": k, "n_parts": n_parts,
            "start": a, "count": b - a,
            "corners": _b64(corners[a:b], np.float32),
            "ijk": _b64(ijk[a:b], np.int16),
        }
        if k == 0:
            payload.update({"times": [float(t) for t in times], "ranges": ranges,
                            "fields": list(fields), "wells": wells or [],
                            "categorical": list(categorical), "lines": lines or []})
        yield item("grid", payload)


def decode_grid(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Reassemble the parts of a grid message (test helper; the page does the same in JS)."""
    parts = sorted(parts, key=lambda p: p["part"])
    n = parts[0]["n_cells"]
    corners = np.concatenate([_unb64(p["corners"], np.float32, (p["count"], 8, 3)) for p in parts])
    ijk = np.concatenate([_unb64(p["ijk"], np.int16, (p["count"], 3)) for p in parts])
    assert corners.shape[0] == n
    head = parts[0]
    return {"corners": corners, "ijk": ijk, "dims": head["dims"], "times": head["times"],
            "ranges": head["ranges"], "fields": head["fields"], "wells": head.get("wells", []),
            "categorical": head.get("categorical", []), "lines": head.get("lines", [])}


# -------------------------------------------------------------------- state
def state(iteration: int, t_index: int, t_days: float, fields: dict[str, np.ndarray],
          ranges: dict[str, list[float]]) -> Item:
    """One snapshot: every field of ``fields`` at one report time, clipped into its
    fixed range and cast to float16 — or to float32 when its range reaches beyond what
    float16 holds (a pore volume of 1e5 m³), in which case ``dtype`` names each field's."""
    enc: dict[str, str] = {}
    dtypes: dict[str, str] = {}
    for name in fields:
        a = np.asarray(fields[name], np.float64)
        lo, hi = ranges[name]
        wide = max(abs(lo), abs(hi)) > F16_MAX
        dtypes[name] = "f32" if wide else "f16"
        enc[name] = _b64(np.clip(np.nan_to_num(a, nan=lo), lo, hi), np.float32 if wide else np.float16)
    dtype: str | dict[str, str] = "f16" if all(d == "f16" for d in dtypes.values()) else dtypes
    return item("state", {"t_index": int(t_index), "t_days": float(t_days), "dtype": dtype,
                          "n_cells": int(next(iter(fields.values())).shape[0]), "fields": enc},
                iteration=iteration)


def decode_state(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    n = payload["n_cells"]
    dtype = payload.get("dtype", "f16")

    def of(name: str):
        kind = dtype.get(name, "f16") if isinstance(dtype, dict) else dtype
        return np.float32 if kind == "f32" else np.float16

    return {k: _unb64(v, of(k), (n,)).astype(np.float32) for k, v in payload["fields"].items()}


def parse(item_: Item) -> tuple[str, int, tuple[float, ...], dict[str, Any]]:
    """Split an item into ``(kind, iteration, losses, payload)``."""
    kind, it, *losses, payload = item_
    return kind, it, tuple(losses), (json.loads(payload) if payload else {})
