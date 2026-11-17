"""Reader for the unformatted ECLIPSE/OPM binary output files (UNRST, INIT, SMSPEC, UNSMRY,
EGRID).

The files are sequences of big-endian Fortran records. Every array starts with a 16-byte
header record (an 8-character keyword, the element count as a 4-byte integer and a
4-character type), followed by data records of at most 1000 numeric or 105 character
elements each. Only the reading needed by the campaign is implemented: whole arrays in file
order, grouped into report steps for restart files and assembled into vectors for summary
files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

_DTYPES = {"INTE": ">i4", "REAL": ">f4", "DOUB": ">f8", "LOGI": ">i4"}


def _records(buf: bytes) -> Iterator[bytes]:
    pos, n = 0, len(buf)
    while pos < n:
        size = int.from_bytes(buf[pos:pos + 4], "big", signed=True)
        yield buf[pos + 4:pos + 4 + size]
        pos += size + 8


def read_arrays(path: str | Path) -> list[tuple[str, np.ndarray | list[str]]]:
    """Every ``(keyword, values)`` array of a binary output file, in file order. Numeric
    arrays come back as numpy arrays, character arrays as lists of stripped strings."""
    recs = _records(Path(path).read_bytes())
    out: list[tuple[str, np.ndarray | list[str]]] = []
    for head in recs:
        kw = head[:8].decode("ascii", "replace").strip()
        count = int.from_bytes(head[8:12], "big", signed=True)
        typ = head[12:16].decode("ascii", "replace")
        if typ == "MESS" or count == 0:
            out.append((kw, np.zeros(0)))
            continue
        if typ in _DTYPES:
            parts, got = [], 0
            while got < count:
                arr = np.frombuffer(next(recs), dtype=_DTYPES[typ])
                parts.append(arr)
                got += arr.size
            out.append((kw, np.concatenate(parts)[:count].astype(_DTYPES[typ][1:])))
        else:
            width = 8 if typ == "CHAR" else int(typ[1:])
            items: list[str] = []
            while len(items) < count:
                raw = next(recs)
                items += [raw[i:i + width].decode("ascii", "replace").strip()
                          for i in range(0, len(raw), width)]
            out.append((kw, items[:count]))
    return out


def restart_steps(path: str | Path) -> list[dict]:
    """The report steps of a unified restart file: one dict of arrays per ``SEQNUM``."""
    steps: list[dict] = []
    for kw, val in read_arrays(path):
        if kw == "SEQNUM":
            steps.append({"SEQNUM": int(np.asarray(val)[0])})
        elif steps:
            steps[-1][kw] = val
    return steps


# Item offsets in the per-well and per-connection restart arrays (OPM/ECLIPSE layout).
IWEL_NCONN, IWEL_TYPE, IWEL_CTL_ACTIVE, IWEL_STATUS, IWEL_CTL_REQUESTED = 4, 6, 7, 10, 15
SWEL_TARGETS = slice(0, 7)       # ORAT, WRAT, GRAT, LRAT, RESV, THP, BHP targets / limits
SWEL_DATUM = 9                   # BHP reference depth
XWEL_RATES = slice(0, 3)         # oil, water, gas surface rates (production positive)
XWEL_BHP = 6
ICON_IJK = slice(1, 4)           # 1-based grid I, J, K of the connection
SCON_CF, SCON_DEPTH, SCON_DIAMETER = 0, 1, 2
XCON_RATES = slice(0, 3)         # oil, water, gas surface rates (production positive)
XCON_PRESSURE = 34

# Control-mode codes of IWEL (requested and active).
CTL_SHUT, CTL_ORAT, CTL_WRAT, CTL_GRAT, CTL_LRAT, CTL_RESV, CTL_THP, CTL_BHP = 0, 1, 2, 3, 4, 5, 6, 7
# Well-type codes of IWEL.
TYPE_PRODUCER, TYPE_OIL_INJ, TYPE_WAT_INJ, TYPE_GAS_INJ = 1, 2, 3, 4


def well_records(path: str | Path) -> dict:
    """The simulator's own well and connection records, per report step of a unified restart
    file: ``time`` (days), the well ``names`` (in order of first appearance), and arrays over
    ``(step, well)`` — ``wtype``, ``ctl_requested``, ``ctl_active``, ``status``, ``targets``
    (7 targets and limits), ``datum``, ``bhp``, ``rates`` (3 phases) — and over
    ``(step, well, connection)`` — ``n_conn`` (per step and well), ``conn_ijk`` (1-based),
    ``conn_cf``, ``conn_depth``, ``conn_diameter``, ``conn_p``, ``conn_rates``. Absent wells and
    connections hold zeros (``wtype == 0``)."""
    steps = restart_steps(path)
    names: list[str] = []
    per_step = []
    for s in steps:
        if "IWEL" not in s:
            per_step.append(None)
            continue
        ih = np.asarray(s["INTEHEAD"])
        nw, ncw = int(ih[16]), int(ih[17])
        niw, nsw, nxw, nzw = (int(x) for x in ih[24:28])
        nic, nsc, nxc = (int(x) for x in ih[32:35])
        zw = s["ZWEL"]
        wn = [zw[i * nzw] for i in range(nw)]
        for n in wn:
            if n not in names:
                names.append(n)
        per_step.append(dict(
            names=wn, ncw=ncw,
            iwel=np.asarray(s["IWEL"]).reshape(nw, niw),
            swel=np.asarray(s["SWEL"], np.float64).reshape(nw, nsw),
            xwel=np.asarray(s["XWEL"], np.float64).reshape(nw, nxw),
            icon=np.asarray(s["ICON"]).reshape(nw, ncw, nic),
            scon=np.asarray(s["SCON"], np.float64).reshape(nw, ncw, nsc),
            xcon=np.asarray(s["XCON"], np.float64).reshape(nw, ncw, nxc)))
    T, W = len(steps), len(names)
    C = max([p["ncw"] for p in per_step if p is not None], default=0)
    out = {"time": np.asarray([float(np.asarray(s["DOUBHEAD"])[0]) if "DOUBHEAD" in s else np.nan
                               for s in steps]),
           "names": names}
    for key, shape, dt in (("wtype", (T, W), int), ("ctl_requested", (T, W), int),
                           ("ctl_active", (T, W), int), ("status", (T, W), int),
                           ("targets", (T, W, 7), float), ("datum", (T, W), float),
                           ("bhp", (T, W), float), ("rates", (T, W, 3), float),
                           ("n_conn", (T, W), int), ("conn_ijk", (T, W, C, 3), int),
                           ("conn_cf", (T, W, C), float), ("conn_depth", (T, W, C), float),
                           ("conn_diameter", (T, W, C), float), ("conn_p", (T, W, C), float),
                           ("conn_rates", (T, W, C, 3), float)):
        out[key] = np.zeros(shape, dt)
    for t, p in enumerate(per_step):
        if p is None:
            continue
        for i, n in enumerate(p["names"]):
            w = names.index(n)
            iw, nc = p["iwel"][i], int(p["iwel"][i][IWEL_NCONN])
            out["wtype"][t, w] = iw[IWEL_TYPE]
            out["ctl_requested"][t, w] = iw[IWEL_CTL_REQUESTED]
            out["ctl_active"][t, w] = iw[IWEL_CTL_ACTIVE]
            out["status"][t, w] = iw[IWEL_STATUS]
            out["targets"][t, w] = p["swel"][i][SWEL_TARGETS]
            out["datum"][t, w] = p["swel"][i][SWEL_DATUM]
            out["bhp"][t, w] = p["xwel"][i][XWEL_BHP]
            out["rates"][t, w] = p["xwel"][i][XWEL_RATES]
            out["n_conn"][t, w] = nc
            out["conn_ijk"][t, w, :nc] = p["icon"][i, :nc, ICON_IJK]
            out["conn_cf"][t, w, :nc] = p["scon"][i, :nc, SCON_CF]
            out["conn_depth"][t, w, :nc] = p["scon"][i, :nc, SCON_DEPTH]
            out["conn_diameter"][t, w, :nc] = p["scon"][i, :nc, SCON_DIAMETER]
            out["conn_p"][t, w, :nc] = p["xcon"][i, :nc, XCON_PRESSURE]
            out["conn_rates"][t, w, :nc] = p["xcon"][i, :nc, XCON_RATES]
    return out


def summary(smspec: str | Path, unsmry: str | Path | None = None) -> dict:
    """The summary vectors of a run: ``{"keys": [(keyword, name, num)], "time": days,
    "data": (n_ministeps, n_vectors)}``, with every ministep the file reports."""
    spec = dict(read_arrays(smspec))
    kws = spec["KEYWORDS"]
    names = spec.get("WGNAMES", spec.get("NAMES", [""] * len(kws)))
    nums = np.asarray(spec.get("NUMS", np.zeros(len(kws), int)))
    unsmry = Path(unsmry) if unsmry else Path(smspec).with_suffix(".UNSMRY")
    rows = [np.asarray(v, np.float64) for k, v in read_arrays(unsmry) if k == "PARAMS"]
    data = np.vstack(rows) if rows else np.zeros((0, len(kws)))
    t_idx = kws.index("TIME") if "TIME" in kws else 0
    return {"keys": list(zip(kws, names, nums.tolist())), "time": data[:, t_idx], "data": data,
            "units": spec.get("UNITS")}


def vector(smry: dict, keyword: str, name: str = "", num: int | None = None) -> np.ndarray:
    """One summary vector by keyword (and well/group name, and cell/connection number)."""
    for j, (k, n, m) in enumerate(smry["keys"]):
        if k == keyword and (not name or n == name) and (num is None or m == num):
            return smry["data"][:, j]
    raise KeyError((keyword, name, num))
