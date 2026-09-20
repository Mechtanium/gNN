r"""
The messages the workflow streams to the page, and their encoding.

Every output item of the ``train`` operation is the 8-tuple

``(kind, iteration, total, pde, ic, data, well, payload)``

where ``kind`` names the message, the five floats carry the losses of a
``loss`` message (zeros otherwise), and ``payload`` is JSON text. Arrays ride
inside the JSON as base64: the grid corners as float32, the cell addresses as
int16, the cell states as float16 — a state value is clipped to its field's
fixed colour range before the cast, so the legend never has to move.

Sizes on SPE-2 (9 375 cells): one ``grid`` part is ~1.2 MB, one ``state``
message (seven fields) ~175 KB, a ``loss`` message ~120 bytes; the grid is
split into parts of at most ``GRID_PART_BYTES`` so a 44 000-cell field still
fits comfortably under the transport's message limit.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Iterator

import numpy as np

KINDS = ("stage", "grid", "loss", "state", "metrics", "error")

#: The seven cell fields a ``state`` message carries, in this order.
FIELDS = ("S_w", "S_o", "S_g", "p_o", "p_w", "p_g", "R_so")

GRID_PART_BYTES = 2_500_000
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
    total, pde, ic, data, well = (float(x) for x in losses)
    return (kind, int(iteration), total, pde, ic, data, well,
            "" if payload is None else json.dumps(payload, separators=(",", ":")))


def stage(name: str, fraction: float, message: str = "") -> Item:
    """What the job is doing and how far along it is (``fraction`` in 0..1)."""
    return item("stage", {"stage": name, "message": message},
                losses=(max(0.0, min(1.0, float(fraction))), 0.0, 0.0, 0.0, 0.0))


def loss(iteration: int, total: float, pde: float, ic: float, data: float, well: float,
         *, wall_s: float, eta: float | None = None) -> Item:
    return item("loss", {"wall_s": float(wall_s), "eta": None if eta is None else float(eta)},
                iteration=iteration, losses=(total, pde, ic, data, well))


def metrics(iteration: int, values: dict[str, float]) -> Item:
    return item("metrics", {k: float(v) for k, v in values.items()}, iteration=iteration)


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
         ranges: dict[str, list[float]], wells: list[dict[str, Any]] | None = None) -> Iterator[Item]:
    """The geometry message(s): corners ``(n_cells, 8, 3)`` in VTK order, ``ijk``
    ``(n_cells, 3)`` zero-based, the snapshot times, the fixed ranges, and optional
    well markers. Yields one part when it fits, else several with ``part``/``n_parts``."""
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
                            "fields": list(FIELDS), "wells": wells or []})
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
            "ranges": head["ranges"], "fields": head["fields"], "wells": head.get("wells", [])}


# -------------------------------------------------------------------- state
def state(iteration: int, t_index: int, t_days: float, fields: dict[str, np.ndarray],
          ranges: dict[str, list[float]]) -> Item:
    """One snapshot: the seven cell fields at one report time, clipped into their
    fixed ranges and cast to float16."""
    enc: dict[str, str] = {}
    for name in FIELDS:
        a = np.asarray(fields[name], np.float64)
        lo, hi = ranges[name]
        enc[name] = _b64(np.clip(np.nan_to_num(a, nan=lo), lo, hi), np.float16)
    return item("state", {"t_index": int(t_index), "t_days": float(t_days), "dtype": "f16",
                          "n_cells": int(next(iter(fields.values())).shape[0]), "fields": enc},
                iteration=iteration)


def decode_state(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    n = payload["n_cells"]
    return {k: _unb64(v, np.float16, (n,)).astype(np.float32) for k, v in payload["fields"].items()}


def parse(item_: Item) -> tuple[str, int, tuple[float, ...], dict[str, Any]]:
    """Split an item into ``(kind, iteration, losses, payload)``."""
    kind, it, *losses, payload = item_
    return kind, it, tuple(losses), (json.loads(payload) if payload else {})
