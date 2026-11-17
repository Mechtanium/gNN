r"""
Black-oil fluid properties from a handful of measurements, for a model with no laboratory
tables: the oil's gravity, the gas's gravity, the reservoir temperature and the bubble-point
pressure (or the solution gas-oil ratio). Every function works in FIELD units — psia, °F,
scf/stb, rb/stb, cP — and the tables are converted for a METRIC deck by :mod:`builder.units`.

**Solution gas and the bubble point** (Standing):

.. math::

    R_s(p) \;=\; \gamma_g \left[\left(\frac{p}{18.2} + 1.4\right) 10^{\,0.0125\,\mathrm{API} - 0.00091\,T}\right]^{1.2048}

where:

- :math:`R_s`: the gas dissolved in a stock-tank barrel of oil at pressure :math:`p` [scf/stb], up to the bubble point :math:`p_b`, constant above it.
- :math:`\gamma_g`: the gas gravity (air = 1); :math:`\mathrm{API}`: the oil gravity; :math:`T`: the temperature [°F].

**Oil formation-volume factor** (Standing), saturated, and undersaturated with a constant
compressibility :math:`c_o`:

.. math::

    B_o = 0.9759 + 1.2\times10^{-4}\left[R_s \sqrt{\gamma_g/\gamma_o} + 1.25\,T\right]^{1.2},
    \qquad
    B_o(p > p_b) = B_{ob}\, e^{-c_o (p - p_b)}

**Oil viscosity** (Beggs–Robinson for the dead and the live oil, Vasquez–Beggs above the
bubble point):

.. math::

    \mu_{od} = 10^{x} - 1,\quad x = T^{-1.163} \exp(6.9824 - 0.04658\,\mathrm{API}),
    \qquad
    \mu_o = A\,\mu_{od}^{B},\quad A = 10.715 (R_s + 100)^{-0.515},\ B = 5.44 (R_s + 150)^{-0.338}

.. math::

    \mu_o(p > p_b) = \mu_{ob} \left(\frac{p}{p_b}\right)^{m},\quad
    m = 2.6\,p^{1.187} \exp(-11.513 - 8.98\times10^{-5} p)

**Gas** — the deviation factor :math:`Z` from the Hall–Yarborough equation at the
pseudo-reduced conditions of Sutton's pseudo-critical correlation, the volume factor, and the
viscosity of Lee, Gonzalez and Eakin:

.. math::

    B_g = 0.005035\,\frac{Z\,(T + 460)}{p}\ \text{[rb/scf]},
    \qquad
    \mu_g = 10^{-4} K \exp\!\left(X \rho_g^{Y}\right)

**Water** (McCain): the volume factor and the viscosity at the reservoir temperature, and its
compressibility, for ``PVTW``.

where (throughout): :math:`\gamma_o = 141.5/(131.5 + \mathrm{API})` the oil's specific gravity,
:math:`\rho_g` the gas density [g/cm³], and :math:`K, X, Y` the temperature- and
molecular-weight functions of the Lee–Gonzalez–Eakin correlation.
"""

from __future__ import annotations

import math

import numpy as np


def oil_sg(api: float) -> float:
    return 141.5 / (131.5 + api)


def rs_standing(p: np.ndarray, api: float, gas_gravity: float, temp_f: float) -> np.ndarray:
    """Standing's solution gas-oil ratio [scf/stb] at pressure ``p`` [psia]."""
    p = np.asarray(p, float)
    return gas_gravity * ((p / 18.2 + 1.4) * 10 ** (0.0125 * api - 0.00091 * temp_f)) ** 1.2048


def pb_standing(rs: float, api: float, gas_gravity: float, temp_f: float) -> float:
    """The bubble point [psia] of an oil holding ``rs`` scf/stb (Standing, inverted)."""
    return 18.2 * ((rs / gas_gravity) ** (1 / 1.2048) * 10 ** (0.00091 * temp_f - 0.0125 * api) - 1.4)


def bo_standing(rs: np.ndarray, api: float, gas_gravity: float, temp_f: float) -> np.ndarray:
    rs = np.asarray(rs, float)
    return 0.9759 + 1.2e-4 * (rs * math.sqrt(gas_gravity / oil_sg(api)) + 1.25 * temp_f) ** 1.2


def mu_dead_oil(api: float, temp_f: float) -> float:
    x = temp_f ** -1.163 * math.exp(6.9824 - 0.04658 * api)
    return 10 ** x - 1.0


def mu_live_oil(rs: np.ndarray, api: float, temp_f: float) -> np.ndarray:
    rs = np.asarray(rs, float)
    a = 10.715 * (rs + 100.0) ** -0.515
    b = 5.44 * (rs + 150.0) ** -0.338
    return a * mu_dead_oil(api, temp_f) ** b


def mu_undersaturated(p: np.ndarray, pb: float, mu_ob: float) -> np.ndarray:
    p = np.asarray(p, float)
    m = 2.6 * p ** 1.187 * np.exp(-11.513 - 8.98e-5 * p)
    return mu_ob * (p / pb) ** m


def z_factor(p: np.ndarray, temp_f: float, gas_gravity: float) -> np.ndarray:
    """Hall–Yarborough deviation factor at Sutton's pseudo-critical properties."""
    tpc = 169.2 + 349.5 * gas_gravity - 74.0 * gas_gravity ** 2
    ppc = 756.8 - 131.0 * gas_gravity - 3.6 * gas_gravity ** 2
    t = tpc / (temp_f + 460.0)
    out = []
    for ppr in np.atleast_1d(np.asarray(p, float)) / ppc:
        a = 0.06125 * t * math.exp(-1.2 * (1 - t) ** 2)
        b = 14.76 * t - 9.76 * t ** 2 + 4.58 * t ** 3
        c = 90.7 * t - 242.2 * t ** 2 + 42.4 * t ** 3
        d = 2.18 + 2.82 * t
        y = 0.0125 * ppr * t * math.exp(-1.2 * (1 - t) ** 2)
        for _ in range(100):
            f = -a * ppr + (y + y ** 2 + y ** 3 - y ** 4) / (1 - y) ** 3 - b * y ** 2 + c * y ** d
            df = (1 + 4 * y + 4 * y ** 2 - 4 * y ** 3 + y ** 4) / (1 - y) ** 4 - 2 * b * y + c * d * y ** (d - 1)
            step = f / df
            y -= step
            y = min(max(y, 1e-10), 0.999)
            if abs(step) < 1e-12:
                break
        out.append(a * ppr / y)
    return np.asarray(out)


def bg(p: np.ndarray, temp_f: float, gas_gravity: float) -> np.ndarray:
    """Gas volume factor [rb/Mscf]."""
    p = np.asarray(p, float)
    return 5.035 * z_factor(p, temp_f, gas_gravity) * (temp_f + 460.0) / p


def mu_gas(p: np.ndarray, temp_f: float, gas_gravity: float) -> np.ndarray:
    """Lee–Gonzalez–Eakin gas viscosity [cP]."""
    p = np.asarray(p, float)
    mw = 28.97 * gas_gravity
    tr = temp_f + 460.0
    rho = 1.4935e-3 * p * mw / (z_factor(p, temp_f, gas_gravity) * tr)       # g/cm3
    k = (9.379 + 0.01607 * mw) * tr ** 1.5 / (209.2 + 19.26 * mw + tr)
    x = 3.448 + 986.4 / tr + 0.01009 * mw
    y = 2.447 - 0.2224 * x
    return 1e-4 * k * np.exp(x * rho ** y)


def water(p_ref: float, temp_f: float) -> dict[str, float]:
    """``PVTW`` at ``p_ref`` [psia]: volume factor, compressibility [1/psi], viscosity [cP]
    (McCain, fresh water), viscosibility 0."""
    dvwt = -1.0001e-2 + 1.33391e-4 * temp_f + 5.50654e-7 * temp_f ** 2
    dvwp = (-1.95301e-9 * p_ref * temp_f - 1.72834e-13 * p_ref ** 2 * temp_f
            - 3.58922e-7 * p_ref - 2.25341e-10 * p_ref ** 2)
    bw = (1 + dvwt) * (1 + dvwp)
    mu1 = 109.574 * temp_f ** -1.12166
    mu = mu1 * (0.9994 + 4.0295e-5 * p_ref + 3.1062e-9 * p_ref ** 2)
    cw = 3.0e-6
    return {"p_ref": p_ref, "bw": float(bw), "cw": cw, "mu": float(mu), "viscosibility": 0.0}


def densities(api: float, gas_gravity: float) -> tuple[float, float, float]:
    """Surface densities [lb/ft³]: oil, water, gas."""
    return 62.428 * oil_sg(api), 62.428, 0.0764 * gas_gravity


def live_oil_tables(api: float, gas_gravity: float, temp_f: float, pb: float, p_max: float,
                    n_sat: int = 12, n_under: int = 4, co: float = 1.0e-5) -> list[np.ndarray]:
    r"""``PVTO`` in FIELD units: per dissolved-gas ratio :math:`R_s` [Mscf/stb] the saturated
    row :math:`(R_s, p_b(R_s), B_o, \mu_o)` and its undersaturated rows up to ``p_max``.
    Saturated pressures run from 14.7 psia to beyond ``pb`` (so a pressure rise above the
    initial bubble point stays in the table)."""
    p_top = max(pb * 1.6, p_max)
    p_sat = np.unique(np.concatenate([np.geomspace(14.7, p_top, n_sat), [pb]]))
    blocks = []
    for ps in p_sat:
        rs = float(rs_standing(ps, api, gas_gravity, temp_f))
        bob = float(bo_standing(rs, api, gas_gravity, temp_f))
        muob = float(mu_live_oil(rs, api, temp_f))
        p_hi = max(p_max * 1.2, ps * 1.5)
        pu = np.linspace(ps, p_hi, n_under + 1)
        bo = bob * np.exp(-co * (pu - ps))
        mu = mu_undersaturated(pu, ps, muob) if ps > 15.0 else np.full_like(pu, muob)
        mu = np.maximum.accumulate(np.maximum(mu, muob))
        blocks.append(np.column_stack([np.full(len(pu), rs / 1000.0), pu, bo, mu]))
    return blocks


def dead_oil_table(api: float, temp_f: float, p_max: float, bo_ref: float = 1.05,
                   co: float = 1.0e-5, p_ref: float = 14.7, n: int = 10) -> np.ndarray:
    """``PVDO`` in FIELD units: :math:`(p, B_o, \\mu_o)` of a dead oil."""
    p = np.linspace(p_ref, max(p_max * 1.3, p_ref + 1000.0), n)
    mu = mu_dead_oil(api, temp_f) * (1 + 5e-5 * (p - p_ref))
    return np.column_stack([p, bo_ref * np.exp(-co * (p - p_ref)), mu])


def gas_table(gas_gravity: float, temp_f: float, p_max: float, n: int = 12) -> np.ndarray:
    """``PVDG`` in FIELD units: :math:`(p, B_g, \\mu_g)` with :math:`B_g` in rb/Mscf."""
    p = np.geomspace(14.7, max(p_max * 1.6, 1000.0), n)
    return np.column_stack([p, bg(p, temp_f, gas_gravity), mu_gas(p, temp_f, gas_gravity)])
