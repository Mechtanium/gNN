r"""
Phase equilibrium of the dissolved gas: the ``eq`` row group.

The network predicts four fields at every point — oil pressure :math:`p_o`, water and gas
saturations :math:`S_w, S_g` and the dissolved gas–oil ratio :math:`R_s` — while the
physics rows balance three components (water, oil, gas). One combination of the gas
fields is therefore left open: gas can be held free or in solution, and the balances
alone do not say which. The simulator closes the system with phase equilibrium, and so
does the surrogate, through one row per point,

.. math::

    r^{\mathrm{eq}} \;=\; \phi_{\mathrm{FB}}(\delta_g, \delta_s),
    \qquad
    \delta_s \;=\; \frac{R_s^{\mathrm{sat}}(p_o) - R_s}{R_s^{\max}},
    \qquad
    \delta_g \;=\;
    \begin{cases}
        S_g & \text{gas re-dissolves (no } \mathrm{DRSDT}\text{, or } \mathrm{DRSDT} > 0)\\[2pt]
        -\dfrac{T}{R_s^{\max}}\,\dfrac{\partial R_s}{\partial t} & \mathrm{DRSDT} = 0
    \end{cases}

where:

- :math:`\phi_{\mathrm{FB}}(\delta_g, \delta_s) = \sqrt{\delta_g^2 + \delta_s^2 + \varepsilon_{\mathrm{FB}}^2} - \delta_g - \delta_s`:
  the smoothed Fischer–Burmeister function (:func:`modules.wells.fischer_burmeister`),
  which vanishes, up to :math:`\varepsilon_{\mathrm{FB}}`, exactly when :math:`\delta_g \ge 0`,
  :math:`\delta_s \ge 0` and :math:`\delta_g \delta_s = 0`.
- :math:`\delta_s`: the undersaturation of the oil; :math:`R_s^{\mathrm{sat}}(p_o)` is the
  gas–oil ratio of oil saturated at :math:`p_o`, the inverse of the deck's bubble-point curve
  (:func:`rs_saturated`), so :math:`\delta_s \ge 0` says the oil holds no more gas than its
  pressure allows — it is never supersaturated.
- :math:`\delta_g`: the free gas, or under ``DRSDT 0`` the rate at which :math:`R_s` falls.
- :math:`R_s^{\max}`: the upper end of the network's :math:`R_s` range; :math:`T`: the
  case's horizon [d]. Both make :math:`\delta_g` and :math:`\delta_s` dimensionless and of
  order one.
- ``DRSDT``: the deck's limit on the rate at which :math:`R_s` may rise
  (:func:`modules.utils.ReservoirExtractors.deck_drsdt`).

Where gas re-dissolves, the complementarity reads: either the oil is undersaturated and
holds no free gas beside it (:math:`\delta_s > 0`, :math:`S_g = 0`), or free gas is present
and the oil is saturated (:math:`S_g > 0`, :math:`R_s = R_s^{\mathrm{sat}}(p_o)`). Where it
does not (``DRSDT 0``), released gas stays free and free gas meeting undersaturated oil
does not dissolve, so :math:`R_s` never rises; it falls, with the pressure, only while the
oil is saturated — the same complementarity with :math:`-\partial_t R_s` in place of
:math:`S_g`.

The rows are evaluated at points (cell centroid, collocation time) — every cell at every
time slice of the window, strided to the window cap (:func:`point_layout`) — and join the
loss as the point-wise ``eq`` group whenever :func:`modules.config.equilibrium_active`.
"""
from __future__ import annotations

import functools
from pathlib import Path

from .config import RunConfig, case_paths
from .wells import FB_EPS, fischer_burmeister


@functools.lru_cache(maxsize=16)
def _deck_drsdt(path: str, mtime: float) -> float | None:
    from modules.utils.ReservoirExtractors import deck_drsdt

    return deck_drsdt(path)


def gas_redissolves(cfg: RunConfig) -> bool:
    """Whether free gas may re-dissolve into the oil: ``False`` exactly when the deck sets
    ``DRSDT 0``; a deck without the keyword, or with a positive limit, lets it."""
    path = Path(case_paths(cfg)[0])
    if not path.exists():
        return True
    drsdt = _deck_drsdt(str(path), path.stat().st_mtime)
    return drsdt is None or drsdt > 0.0


def saturated_table(tables: dict):
    """The deck's saturated PVTO curve as ``(p_b, R_s)`` arrays with :math:`p_b` increasing,
    the abscissa and ordinate of :func:`rs_saturated`."""
    import numpy as onp

    pb = onp.asarray(tables["pvto_pbub_sat"], float)
    rs = onp.asarray(tables["pvto_rs_sat"], float)
    order = onp.argsort(pb)
    return pb[order], rs[order]


def rs_saturated(p, pb, rs_tab):
    r"""
    The saturated gas–oil ratio :math:`R_s^{\mathrm{sat}}(p)`: the deck's bubble-point curve
    :math:`p_b(R_s)` inverted by linear interpolation in the table :math:`(p_b, R_s)`, held
    at its end values outside the table.
    """
    import jax.numpy as jnp

    return jnp.interp(p, jnp.asarray(pb, p.dtype), jnp.asarray(rs_tab, p.dtype))


def equilibrium_row(p, s_g, rs, drs_dt, pb, rs_tab, rs_max: float, t_scale: float,
                    redissolves: bool, eps: float = FB_EPS):
    r"""
    The equilibrium residual :math:`\phi_{\mathrm{FB}}(\delta_g, \delta_s)` of the module docstring at one or
    many points: ``p``, ``s_g``, ``rs`` the predicted primaries, ``drs_dt`` the time
    derivative of :math:`R_s` [1/d] (unused where gas re-dissolves), ``(pb, rs_tab)`` the
    saturated curve of :func:`saturated_table`, ``rs_max`` the scale :math:`R_s^{\max}` and
    ``t_scale`` the horizon :math:`T` [d].
    """
    delta_s = (rs_saturated(p, pb, rs_tab) - rs) / rs_max
    delta_g = s_g if redissolves else -drs_dt * (t_scale / rs_max)
    return fischer_burmeister(delta_g, delta_s, eps)


def point_layout(n_cells: int, n_slices: int, cap: int) -> tuple[int, int]:
    r"""
    ``(stride, n_c)``: the equilibrium points of a slice window are the cells
    ``ci[::stride]`` (``n_c`` of them) at every one of its ``n_slices`` times, the stride the
    smallest that keeps :math:`n_c\, n_{\mathrm{slices}} \le` ``cap``.
    """
    n_cells, n_slices = max(1, int(n_cells)), max(1, int(n_slices))
    stride = max(1, -(-n_cells * n_slices // max(1, int(cap))))
    return stride, -(-n_cells // stride)


def supersaturated_fraction(p, rs, pb, rs_tab, tol: float = 0.05) -> float:
    r"""
    The share of points whose oil holds more gas than its pressure allows,
    :math:`R_s > (1 + \mathrm{tol})\, R_s^{\mathrm{sat}}(p)` — a state the simulator never
    produces; host-side (NumPy) arrays of any matching shape.
    """
    import numpy as onp

    rs_sat = onp.interp(onp.asarray(p, float), pb, rs_tab)
    return float(onp.mean(onp.asarray(rs, float) > (1.0 + tol) * rs_sat))
