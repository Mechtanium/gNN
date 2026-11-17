r"""
The model at other values of the properties a field record is inverted for.

A family of reservoirs is taught by simulator runs of the model at sampled values of its
uncertain properties. Each run's deck is the model's own with three edits, made through the
deck document (:mod:`builder.deck`) so that every other byte of the tree is unchanged:

.. math::

    k_z \;\leftarrow\; k_z\, e^{\sigma_k u_1},\qquad
    z_{\mathrm{WOC}} \;\leftarrow\; z_{\mathrm{WOC}} + \sigma_{\mathrm{WOC}}\, u_2,\qquad
    z_{\mathrm{GOC}} \;\leftarrow\; z_{\mathrm{GOC}} + \sigma_{\mathrm{GOC}}\, u_3

where:

- :math:`u_i \sim \mathcal N(0, 1)`: the whitened value of property :math:`i` of the sample.
- :math:`\sigma_k`: the prior standard deviation of :math:`\ln(k_v/k_h)` — the multiplier is applied by a ``MULTIPLY`` of ``PERMZ`` closing the ``GRID`` section, after every operation of the deck's own.
- :math:`\sigma_{\mathrm{WOC}}, \sigma_{\mathrm{GOC}}`: the prior standard deviations of the water–oil and gas–oil contacts, in the deck's length unit — the shift is applied to items 3 and 5 of every ``EQUIL`` record, in whichever file it sits.

Which properties are uncertain is a rule, not a choice: the vertical permeability always; a
contact when the deck has the phase on its far side and the contact lies within the
reservoir's depth span (:func:`properties_of`).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from . import records as R
from .deck import Deck, block_text

PROPERTY_ORDER = ("kvkh", "woc", "goc")
LABELS = {"kvkh": "kv/kh multiplier", "woc": "Water–oil contact", "goc": "Gas–oil contact"}


def equil_rows(deck: Deck) -> list[list]:
    """Every ``EQUIL`` record of the deck, in order (its keywords may sit in any file)."""
    rows: list[list] = []
    for keyword in deck.keywords("EQUIL"):
        rows += [r for r in R.parse_records(keyword.data()) if r]
    return rows


def phases(deck: Deck) -> set[str]:
    return {k.name for k in deck.keywords(section="RUNSPEC")} & {"OIL", "WATER", "GAS", "DISGAS", "VAPOIL"}


def contacts(deck: Deck) -> dict[str, list[float]]:
    """The water–oil and gas–oil contact depths of every equilibration region."""
    out: dict[str, list[float]] = {"woc": [], "goc": []}
    for row in equil_rows(deck):
        if len(row) > 2 and row[2] is not None:
            out["woc"].append(float(row[2]))
        if len(row) > 4 and row[4] is not None:
            out["goc"].append(float(row[4]))
    return out


def properties_of(deck: Deck, z_min: float, z_max: float) -> list[str]:
    """The properties a family of this model inverts (see the module docstring); ``z_min``,
    ``z_max`` are the depths of the shallowest and deepest active cell centres."""
    have = phases(deck)
    found = contacts(deck)
    out = ["kvkh"]
    if "WATER" in have and any(z_min < z < z_max for z in found["woc"]):
        out.append("woc")
    if "GAS" in have and any(z_min < z < z_max for z in found["goc"]):
        out.append("goc")
    return out


def physical(u: Sequence[float], names: Sequence[str], prior: dict[str, float],
             base: dict[str, float] | None = None) -> dict[str, float]:
    """The physical values of a whitened sample: the kv/kh multiplier, and for each contact
    its shift (and, with ``base``, its depth)."""
    out: dict[str, float] = {}
    for value, name in zip(u, names):
        if name == "kvkh":
            out["kvkh"] = math.exp(float(prior["kvkh"]) * float(value))
        else:
            out[f"{name}_shift"] = float(prior[name]) * float(value)
            if base and name in base:
                out[name] = base[name] + out[f"{name}_shift"]
    return out


def variant(tree: dict[str, bytes], deck_name: str, values: dict[str, float]) -> dict[str, bytes]:
    """The deck tree with a sample's edits (``values`` as :func:`physical` gives them)."""
    deck = Deck.from_tree(tree, deck_name)
    factor = values.get("kvkh")
    if factor is not None and abs(factor - 1.0) > 0:
        deck.append("GRID", "MULTIPLY", f" 'PERMZ' {R.fmt(float(f'{factor:.8g}'))} /\n/")
    dw, dg = values.get("woc_shift", 0.0), values.get("goc_shift", 0.0)
    if dw or dg:
        for keyword in deck.keywords("EQUIL"):
            rows = [r for r in R.parse_records(keyword.data()) if r]
            for row in rows:
                if dw and len(row) > 2 and row[2] is not None:
                    row[2] = float(row[2]) + dw
                if dg and len(row) > 4 and row[4] is not None:
                    row[4] = float(row[4]) + dg
            keyword.set_body(R.format_records(rows, terminate=False))
    return deck.tree()


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` whitened samples in ``d`` dimensions, stratified in every dimension: sample
    :math:`s` of dimension :math:`j` is the normal quantile of a uniform draw in stratum
    :math:`\\pi_j(s)` of :math:`n`."""
    from scipy.stats import norm

    u = np.empty((n, d))
    for j in range(d):
        strata = (rng.permutation(n) + rng.random(n)) / n
        u[:, j] = norm.ppf(np.clip(strata, 1e-9, 1 - 1e-9))
    return u


__all__ = ["PROPERTY_ORDER", "LABELS", "contacts", "properties_of", "physical", "variant",
           "latin_hypercube", "equil_rows", "block_text"]
