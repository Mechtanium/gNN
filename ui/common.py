"""What every tab shares: the retention rules of the jobs, the field names the views show, and
small input helpers that keep a widget and the design in step."""

from __future__ import annotations

from typing import Any

import perdlit as pl

from . import session

#: How many messages of each kind a run keeps (in the page and in every browser). Kinds not
#: listed are kept whole. The snapshot sweeps are kept generously, so the 3-D view can play
#: the field forming back across iterations.
RETAIN = {
    "check": {"kind_field": 0, "keep": {"state": 2, "grid": 64, "diag": 600, "stage": 10}},
    "simulate": {"kind_field": 0, "keep": {"state": 32, "grid": 64, "log": 400, "stage": 20, "well": 40000}},
    "train": {"kind_field": 0, "keep": {"state": 96, "grid": 64, "stage": 20, "well": 40000, "log": 200}},
    "invert": {"kind_field": 0, "keep": {"sample": 400, "stage": 40, "well": 40000, "log": 200}},
}

DYNAMIC_FIELDS = {
    "S_w": "Water saturation (S_w)",
    "S_o": "Oil saturation (S_o)",
    "S_g": "Gas saturation (S_g)",
    "p_o": "Oil pressure (p_o) [psia]",
    "p_w": "Water pressure (p_w) [psia]",
    "p_g": "Gas pressure (p_g) [psia]",
    "R_so": "Dissolved gas ratio (R_so) [Mscf/stb]",
}


def field_labels() -> dict[str, str]:
    from modules.flows.check import STATIC_FIELDS

    return {**DYNAMIC_FIELDS, **STATIC_FIELDS}


QUANTITIES = {
    "WBHP": "Bottom-hole pressure [psia]",
    "WOPR": "Oil rate [stb/d]",
    "WWPR": "Water rate [stb/d]",
    "WGPR": "Gas rate [Mscf/d]",
    "WGOR": "Gas–oil ratio [Mscf/stb]",
    "WWCT": "Water cut",
    "WWIR": "Water injection rate [stb/d]",
    "WGIR": "Gas injection rate [Mscf/d]",
}


def wells_in(run) -> list[str]:
    """The wells a run has sent records for, in order of appearance."""
    seen: dict[str, None] = {}
    for item in getattr(run, "outputs", []) or []:
        if isinstance(item, (list, tuple)) and item and item[0] == "well":
            try:
                import json

                seen.setdefault(json.loads(item[-1])["well"], None)
            except (ValueError, KeyError, TypeError):
                continue
    return list(seen)


def payloads(run, kind: str) -> list[dict]:
    """The JSON payloads of a run's messages of one kind, in order."""
    import json

    out = []
    for item in getattr(run, "outputs", []) or []:
        if isinstance(item, (list, tuple)) and item and item[0] == kind and item[-1]:
            try:
                out.append(json.loads(item[-1]))
            except ValueError:
                continue
    return out


def number(label: str, value: float, key: str, *, min_value=None, max_value=None, step=None,
           help: str | None = None) -> float:
    """A number field showing ``value`` until its viewer changes it."""
    integer = isinstance(value, int) and not isinstance(value, bool) and (step is None or isinstance(step, int))
    v = int(value) if integer else float(value)
    return pl.number_input(label, min_value, max_value, v, step, key=key, help=help)


def bound(label: str, path: tuple, key: str, gen: int, **kwargs) -> Any:
    """A number field bound to ``design[path]``: what the viewer types goes into the design."""
    d = session.design()
    node = d
    for part in path[:-1]:
        node = node[part]
    current = node[path[-1]]
    value = number(label, current, f"g{gen}:{key}", **kwargs)
    if value != current:
        session.edit_design(lambda dd: _set(dd, path, value))
    return value


def _set(d: dict, path: tuple, value) -> None:
    node = d
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value


def choice(label: str, options: list, current, key: str, gen: int, *, help: str | None = None):
    index = options.index(current) if current in options else 0
    return pl.selectbox(label, options, index, key=f"g{gen}:{key}", help=help)
