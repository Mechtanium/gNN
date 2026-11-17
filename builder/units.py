r"""
The quantities a design is entered in, and their conversion between the FIELD and METRIC
unit systems of an Eclipse deck. A design keeps its numbers in its own system; the fluid
correlations of :mod:`builder.correlations` work in FIELD units, so their tables are
converted for a METRIC deck.

========================  ===============  ===============  =====================
quantity                  FIELD            METRIC           METRIC per FIELD unit
========================  ===============  ===============  =====================
length, depth             ft               m                0.3048
pressure                  psia             barsa            0.0689475729
liquid rate, volume       stb/d, stb       sm³/d, sm³       0.158987295
gas rate, volume          Mscf/d, Mscf     sm³/d, sm³       28.3168466
gas-oil ratio             Mscf/stb         sm³/sm³          178.107607
gas volume factor         rb/Mscf          rm³/sm³          0.00561458
compressibility           1/psi            1/bar            14.5037738
density                   lb/ft³           kg/m³            16.0184634
temperature               °F               °C               (T − 32)/1.8
========================  ===============  ===============  =====================
"""

from __future__ import annotations

UNITS = ("FIELD", "METRIC")

FACTOR = {
    "length": 0.3048,
    "pressure": 0.0689475729,
    "liquid": 0.158987295,
    "gas": 28.3168466,
    "gor": 178.107607,
    "bg": 0.00561458,
    "compressibility": 14.5037738,
    "density": 16.0184634,
}

LABEL = {
    "FIELD": {"length": "ft", "pressure": "psia", "liquid": "stb/d", "gas": "Mscf/d", "gor": "Mscf/stb",
              "bg": "rb/Mscf", "compressibility": "1/psi", "density": "lb/ft³", "temperature": "°F",
              "permeability": "mD", "viscosity": "cP", "volume": "rb"},
    "METRIC": {"length": "m", "pressure": "bar", "liquid": "sm³/d", "gas": "sm³/d", "gor": "sm³/sm³",
               "bg": "rm³/sm³", "compressibility": "1/bar", "density": "kg/m³", "temperature": "°C",
               "permeability": "mD", "viscosity": "cP", "volume": "rm³"},
}


def to_field(value, quantity: str, units: str):
    """``value`` given in ``units`` as the FIELD system gives it."""
    if units == "FIELD":
        return value
    if quantity == "temperature":
        return value * 1.8 + 32.0
    return value / FACTOR[quantity]


def from_field(value, quantity: str, units: str):
    """``value`` given in FIELD units as ``units`` gives it."""
    if units == "FIELD":
        return value
    if quantity == "temperature":
        return (value - 32.0) / 1.8
    return value * FACTOR[quantity]


def label(quantity: str, units: str) -> str:
    return LABEL[units].get(quantity, "")
