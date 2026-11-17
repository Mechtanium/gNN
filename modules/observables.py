r"""
Eclipse-style observables recorded across an inference window.

Three families, all evaluated from one trained parameter set at arbitrary times
(the network is a continuous field, so a "report step" is only a time to sample):

**Well observables** reuse the Peaceman closure of :mod:`modules.wells`, which
already eliminates the bottom-hole pressure in rate-controlled mode and returns
per-phase surface rates. The derived ratios are

.. math::

    \mathrm{WGOR} = \frac{\mathrm{WGPR}}{\mathrm{WOPR}},
    \qquad
    \mathrm{WWCT} = \frac{\mathrm{WWPR}}{\mathrm{WWPR} + \mathrm{WOPR}}

where:
- :math:`\mathrm{WOPR}, \mathrm{WWPR}, \mathrm{WGPR}`: surface oil, water and gas production rates [stb/day, stb/day, Mscf/day], reported positive when producing.
- :math:`\mathrm{WGOR}`: producing gas-oil ratio [Mscf/stb]; the gas leg is the *total* surface gas (free plus dissolved), which is the deck's own WGPR convention.
- :math:`\mathrm{WWCT}`: water cut, the water fraction of surface liquid.

**Field totals** are the pore-volume reduction of the same accumulation the PDE
uses, so an in-place volume is the residual's own integrand summed over cells:

.. math::

    \mathrm{FOIP}(t) = \sum_{c} V_c\, \beta\, \phi_c\, \frac{S_{o,c}}{B_{o,c}},
    \qquad
    \mathrm{FWIP}(t) = \sum_{c} V_c\, \beta\, \phi_c\, \frac{S_{w,c}}{B_{w,c}},

.. math::

    \mathrm{FGIP}(t) = \sum_{c} V_c\, \beta\, \phi_c
    \left(\frac{S_{g,c}}{B_{g,c}} + R_{s,c}\,\frac{S_{o,c}}{B_{o,c}}\right)

where:
- :math:`V_c`: cell bulk volume [ft³], from the FEM quadrature when available.
- :math:`\beta`: the ft³ :math:`\to` reservoir-barrel factor ``case.rb_ft3``.
- :math:`\phi_c = \phi_c^{0}\,(1 + c_r (p_{o,c} - p_{\mathrm{ref}}))`: pressure-corrected porosity.
- :math:`B_{o}, B_{w}, B_{g}`: black-oil formation-volume factors at the cell's own phase pressures.
- :math:`R_{s}`: dissolved gas-oil ratio, so FGIP counts free *and* dissolved gas.

**Block states** are the primaries themselves plus the phase pressures the
capillary closures imply:

.. math::

    \mathrm{BPR}_o = p_o,
    \qquad
    \mathrm{BPR}_w = p_o - p_{cow}(S_w),
    \qquad
    \mathrm{BPR}_g = p_o + p_{cgo}(S_g)

where:
- :math:`p_{cow}, p_{cgo}`: the oil-water and gas-oil capillary pressures from the deck's SCAL tables.
- :math:`\mathrm{BPR}_\alpha`: per-block phase pressure [psia] of phase :math:`\alpha`.
"""

from __future__ import annotations

from typing import Any

import numpy as onp


def _phase_pressures(bundle, P):
    """(p_o, p_w, p_g) [psia] at every cell from the primaries block ``P`` (n_cells, 4)."""
    import jax.numpy as jnp

    from modules.utils.blackoil_closures import cap_pres_go, cap_pres_ow

    tab = bundle.case.tables
    p_o, s_w, s_g = P[:, 0], P[:, 1], P[:, 2]
    p_w = p_o - cap_pres_ow(s_w, tab)
    p_g = p_o + cap_pres_go(s_g, tab)
    return onp.asarray(p_o), onp.asarray(p_w), onp.asarray(p_g)


def block_states(bundle, params, times) -> dict[str, onp.ndarray]:
    """Per-cell saturations and phase pressures at each time: arrays ``(n_t, n_cells)``."""
    from . import metrics as diagnostics

    soil, sgas, swat, bpo, bpw, bpg = [], [], [], [], [], []
    for t in times:
        P = diagnostics.predict_cells(bundle, params, float(t))
        s_w, s_g = onp.asarray(P[:, 1]), onp.asarray(P[:, 2])
        p_o, p_w, p_g = _phase_pressures(bundle, P)
        swat.append(s_w); sgas.append(s_g); soil.append(1.0 - s_w - s_g)
        bpo.append(p_o); bpw.append(p_w); bpg.append(p_g)
    return {"SOIL": onp.stack(soil), "SGAS": onp.stack(sgas), "SWAT": onp.stack(swat),
            "BPR_o": onp.stack(bpo), "BPR_w": onp.stack(bpw), "BPR_g": onp.stack(bpg)}


def field_totals(bundle, params, times) -> dict[str, onp.ndarray]:
    r"""FOIP [stb], FWIP [stb], FGIP [Mscf] and the pore-volume-weighted FPR [psia]."""
    import jax.numpy as jnp

    from modules.utils.blackoil_closures import fvf_gas, fvf_oil, fvf_water

    from . import metrics as diagnostics
    from .metrics import cell_volumes_ft3

    case, phys = bundle.case, bundle.case.phys
    vol = onp.asarray(cell_volumes_ft3(case, bundle.spec), onp.float64)
    rb = float(case.rb_ft3)
    poro0 = onp.asarray(case.poro_cell, onp.float64)
    cr, pref = float(phys.ROCK_CR), float(phys.ROCK_PREF)
    tab = case.tables

    foip, fwip, fgip, fpr = [], [], [], []
    for t in times:
        P = diagnostics.predict_cells(bundle, params, float(t))
        p_o, s_w, s_g, rs = (onp.asarray(P[:, i], onp.float64) for i in range(4))
        s_o = 1.0 - s_w - s_g
        _, p_w, p_g = _phase_pressures(bundle, P)
        b_o = onp.asarray(fvf_oil(jnp.asarray(P[:, 3]), tab, jnp.asarray(P[:, 0])), onp.float64)
        b_w = onp.asarray(fvf_water(jnp.asarray(p_w), tab), onp.float64)
        b_g = onp.asarray(fvf_gas(jnp.asarray(p_g), tab), onp.float64)
        phi = poro0 * (1.0 + cr * (p_o - pref))
        pv = vol * rb * phi
        foip.append(float((pv * s_o / b_o).sum()))
        fwip.append(float((pv * s_w / b_w).sum()))
        fgip.append(float((pv * (s_g / b_g + rs * s_o / b_o)).sum()))
        fpr.append(float((pv * p_o).sum() / max(pv.sum(), 1e-30)))
    return {"FOIP": onp.asarray(foip), "FWIP": onp.asarray(fwip),
            "FGIP": onp.asarray(fgip), "FPR": onp.asarray(fpr)}


def well_series(bundle, params) -> dict[str, Any]:
    r"""
    Per-well Eclipse summary vectors over the well pack's report times.

    Returns ``times`` plus ``(n_t, n_wells)`` arrays for WBHP, WOPR, WWPR, WGPR,
    WWIR, WGIR, WGOR and WWCT. Rates come back in the deck's sign convention
    (production positive), inverting the pack's injection-positive internal one.
    """
    pack = bundle.well_pack
    if pack is None or bundle.ops.well_predict is None:
        return {}
    p_bh, q = bundle.ops.well_predict(bundle.steps.as_f32(params))
    p_bh, q = onp.asarray(p_bh, onp.float64), onp.asarray(q, onp.float64)  # (T,W), (T,W,3)

    wopr, wwpr, wgpr = (onp.maximum(-q[:, :, i], 0.0) for i in (0, 1, 2))
    wwir, wgir = (onp.maximum(q[:, :, i], 0.0) for i in (1, 2))
    eps = 1e-12
    liquid = wopr + wwpr
    return {
        "times": onp.asarray(pack.times, onp.float64),
        "well_names": list(pack.well_names),
        "WBHP": p_bh, "WOPR": wopr, "WWPR": wwpr, "WGPR": wgpr,
        "WWIR": wwir, "WGIR": wgir,
        "WGOR": onp.where(wopr > eps, wgpr / onp.maximum(wopr, eps), 0.0),
        "WWCT": onp.where(liquid > eps, wwpr / onp.maximum(liquid, eps), 0.0),
    }


def record(bundle, params, times=None, stride: int = 1) -> dict[str, Any]:
    """Everything above, on one strided set of times (default: the case's report steps)."""
    case = bundle.case
    if times is None:
        lo, hi = case.t_span()
        t_all = onp.asarray(case.times, onp.float64)
        times = t_all[(t_all >= lo - 1e-9) & (t_all <= hi + 1e-9)]
    times = onp.asarray(times, onp.float64)[:: max(1, int(stride))]
    return {"TIME": times, "blocks": block_states(bundle, params, times),
            "field": field_totals(bundle, params, times),
            "wells": well_series(bundle, params)}


def to_frame(rec: dict[str, Any]):
    """The well and field vectors as one tidy DataFrame (one row per well x time)."""
    import pandas as pd

    times, field, wells_ = rec["TIME"], rec["field"], rec.get("wells") or {}
    rows = []
    wt = onp.asarray(wells_.get("times", times), onp.float64)
    names = wells_.get("well_names", [])
    for i, t in enumerate(times):
        base = {"TIME": float(t), **{k: float(v[i]) for k, v in field.items()}}
        if not names:
            rows.append({"WELL": "", **base})
            continue
        j = int(onp.abs(wt - t).argmin())                 # nearest recorded well step
        for w, name in enumerate(names):
            rows.append({"WELL": name, **base,
                         **{k: float(onp.asarray(wells_[k])[j, w])
                            for k in ("WBHP", "WOPR", "WWPR", "WGPR",
                                      "WWIR", "WGIR", "WGOR", "WWCT")}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Pressure-transient views (ported from the EDA dashboard's Well Test Analysis tab)
# ---------------------------------------------------------------------------------------------

BOURDET_SPAN = 0.15


def observed_wells(pack) -> dict[str, Any]:
    r"""
    The deck's own summary vectors, in the same convention :func:`well_series` predicts.

    Rates are returned production-positive and NaN wherever the summary reported
    nothing, so an observed trace plots as gaps rather than as zeros. The derived
    ratios are formed from the observed legs only.

    where:
    - ``WBHP``: observed bottom-hole pressure [psia], NaN outside ``pack.bhp_mask``.
    - ``WOPR``/``WWPR``/``WGPR``: observed surface production rates, NaN outside ``pack.ch_mask``.
    - ``WGOR``/``WWCT``: observed gas-oil ratio and water cut, NaN wherever either leg is missing.
    """
    from .wells import _CH_PHASE, _CH_SIGN

    T, W = pack.n_times, pack.n_wells
    bhp = onp.where(pack.bhp_mask > 0, pack.bhp_obs, onp.nan)
    prod = {k: onp.full((T, W), onp.nan) for k in ("WOPR", "WWPR", "WGPR")}
    inj = {k: onp.full((T, W), onp.nan) for k in ("WWIR", "WGIR")}
    names = ("WOPR", "WWPR", "WGPR", "WWIR", "WGIR")
    for c, name in enumerate(names):
        m = pack.ch_mask[:, :, c] > 0
        vals = onp.where(m, pack.ch_obs[:, :, c], onp.nan)
        (prod if _CH_SIGN[c] < 0 else inj)[name] = vals

    wopr, wwpr, wgpr = prod["WOPR"], prod["WWPR"], prod["WGPR"]
    liquid = wopr + wwpr
    with onp.errstate(divide="ignore", invalid="ignore"):
        wgor = onp.where(wopr > 0, wgpr / wopr, onp.nan)
        wwct = onp.where(liquid > 0, wwpr / liquid, onp.nan)
    return {"times": onp.asarray(pack.times, onp.float64),
            "well_names": list(pack.well_names), "WBHP": bhp,
            **prod, **inj, "WGOR": wgor, "WWCT": wwct}


def control_targets(pack) -> dict[str, Any]:
    r"""
    The schedule controls of the pack in the summary-vector convention, NaN where a
    step carries no such control: ``WBHP`` on BHP-controlled steps (and the limit of a
    limit-carrying rate control), ``WOPR``/``WWPR``/``WGPR`` for production-rate
    controls and ``WWIR``/``WGIR`` for injection-rate controls. These are the targets
    of the ``ctrl`` rows under the ``predicted`` well model.
    """
    T, W = pack.n_times, pack.n_wells
    out = {k: onp.full((T, W), onp.nan) for k in ("WBHP", "WOPR", "WWPR", "WGPR", "WWIR", "WGIR")}
    act = pack.active > 0
    out["WBHP"] = onp.where(act & (pack.p_bh_ctrl > 0), pack.p_bh_ctrl, onp.nan)
    rate = act & (pack.is_rate > 0)
    for ph, prod_key, inj_key in ((0, "WOPR", None), (1, "WWPR", "WWIR"), (2, "WGPR", "WGIR")):
        sel = rate & (pack.ctrl_phase == ph)
        out[prod_key] = onp.where(sel & (pack.q_ctrl < 0), -pack.q_ctrl, onp.nan)
        if inj_key is not None:
            out[inj_key] = onp.where(sel & (pack.q_ctrl > 0), pack.q_ctrl, onp.nan)
    return {"times": onp.asarray(pack.times, onp.float64), "well_names": list(pack.well_names), **out}


def bourdet_derivative(times, delta_p, span: float = BOURDET_SPAN):
    r"""
    Three-point logarithmic derivative on an :math:`L`-spaced stencil.

    .. math::

        \delta p'^{\,n}
        =
        \frac{
        \dfrac{\delta p^{n} - \delta p^{\ell}}{\ln\tau^{n} - \ln\tau^{\ell}}\,(\ln\tau^{r} - \ln\tau^{n})
        +
        \dfrac{\delta p^{r} - \delta p^{n}}{\ln\tau^{r} - \ln\tau^{n}}\,(\ln\tau^{n} - \ln\tau^{\ell})
        }{\ln\tau^{r} - \ln\tau^{\ell}}

    where:
    - :math:`\tau^{n}`: elapsed time on the well's own clock [days].
    - :math:`\delta p^{n}`: drawdown magnitude :math:`|p_{wf}^{n} - p_i|`.
    - :math:`\ell, r`: nearest indices below and above :math:`n` satisfying the span condition :math:`\ln\tau^{n} - \ln\tau^{\ell} \ge L` and :math:`\ln\tau^{r} - \ln\tau^{n} \ge L`.
    - :math:`L` (``span``): smoothing span in natural-log time; :math:`L = 0` recovers the plain three-point difference.

    A derivative plateau marks infinite-acting radial flow, and a late return to unit
    slope marks boundary-dominated depletion of a closed drainage volume.
    """
    times = onp.asarray(times, float)
    delta_p = onp.asarray(delta_p, float)
    out = onp.full(times.shape, onp.nan)
    ok = onp.flatnonzero(onp.isfinite(times) & onp.isfinite(delta_p) & (times > 0.0))
    if ok.size < 3:
        return out
    log_t, vals = onp.log(times[ok]), delta_p[ok]
    span = max(float(span), 0.0)
    for n in range(1, ok.size - 1):
        left = n - 1
        while left > 0 and (log_t[n] - log_t[left]) < span:
            left -= 1
        right = n + 1
        while right < ok.size - 1 and (log_t[right] - log_t[n]) < span:
            right += 1
        back, fwd = log_t[n] - log_t[left], log_t[right] - log_t[n]
        if back <= 0.0 or fwd <= 0.0:
            continue
        out[ok[n]] = ((vals[n] - vals[left]) / back * fwd
                      + (vals[right] - vals[n]) / fwd * back) / (back + fwd)
    return out


def _perf_cells(pack, w: int) -> onp.ndarray:
    return onp.asarray(pack.cell_idx)[onp.asarray(pack.well_id) == int(w)]


def well_test(bundle, params, w: int, span: float = BOURDET_SPAN) -> dict[str, Any]:
    r"""
    One well's drawdown transient, predicted and observed, on its own clock.

    The clock is anchored one report step before the well's first finite observation,
    so a well entering the schedule late still starts its transient at the origin of a
    logarithmic axis:

    .. math::

        \tau^{n} = T^{n} - T^{n_{\ast}},
        \qquad
        p_i = \frac{1}{|\mathcal{P}_w|}\sum_{c \in \mathcal{P}_w} p_c^{\,n_{\ast}},
        \qquad
        \delta p^{n} = \left| p_{wf}^{n} - p_i \right|

    where:
    - :math:`n_{\ast}`: reference report index, one step before the first finite ``WBHP`` (0 for a well open from the start).
    - :math:`\mathcal{P}_w`: the cells well :math:`w` perforates.
    - :math:`p_i`: pre-production completion pressure, playing the role of the initial reservoir pressure.
    - :math:`\delta p^{n}`: drawdown magnitude; the absolute value keeps injectors and producers on one positive axis.
    """
    pack = bundle.well_pack
    obs = observed_wells(pack)
    pred = well_series(bundle, params)
    t_cal = onp.asarray(pack.times, float)

    finite = onp.flatnonzero(onp.isfinite(obs["WBHP"][:, w]))
    n_ref = max(int(finite[0]) - 1, 0) if finite.size else 0
    tau = t_cal - t_cal[n_ref]

    cells = _perf_cells(pack, w)
    p_i = (float(onp.nanmean(onp.asarray(bundle.case.pres)[n_ref, cells]))
           if cells.size else onp.nan)

    out: dict[str, Any] = {"tau": tau, "t_cal": t_cal, "n_ref": n_ref, "p_i": p_i,
                           "well": pack.well_names[w], "span": float(span)}
    for tag, pwf in (("pred", onp.asarray(pred["WBHP"])[:, w]),
                     ("obs", onp.asarray(obs["WBHP"])[:, w])):
        dp = onp.abs(pwf - p_i) if onp.isfinite(p_i) else onp.full_like(pwf, onp.nan)
        dp = onp.where((tau > 0.0) & onp.isfinite(dp) & (dp > 0.0), dp, onp.nan)
        out[f"pwf_{tag}"] = pwf
        out[f"dp_{tag}"] = dp
        out[f"deriv_{tag}"] = bourdet_derivative(tau, dp, span=span)
    return out


def _ray_cells(case, i0: int, j0: int, k: int, step_i: int, step_j: int) -> onp.ndarray:
    """Walk (I, J) outward from a column along one diagonal until the grid goes inactive."""
    aci = onp.asarray(case.aci)
    lookup = {(int(a), int(b), int(c)): n for n, (a, b, c) in enumerate(aci)}
    cells, d = [], 0
    while True:
        idx = lookup.get((int(i0 + d * step_i), int(j0 + d * step_j), int(k)))
        if idx is None:
            break
        cells.append(idx)
        d += 1
    return onp.asarray(cells, int)


def radial_profile(bundle, params, w: int, t: float, ray=None, layer=None) -> dict[str, Any]:
    r"""
    Block pressure outward from a well along a grid diagonal — the drainage cone.

    The diagonal is the longest straight cell path from the well to a corner of the
    drainage region, so it resolves the cone all the way to the boundary. Radii are
    horizontal distances from the well cell, with the first replaced by the
    area-equivalent radius of that cell so it can sit on a logarithmic axis:

    .. math::

        r_d = \left\lVert \mathbf{x}^{\perp}_{c_d} - \mathbf{x}^{\perp}_{c_0} \right\rVert_2,
        \qquad
        r_0 = \sqrt{\frac{\Delta x\,\Delta y}{\pi}}

    where:
    - :math:`\mathbf{x}^{\perp}_{c}`: the areal (x, y) centroid of cell :math:`c`.
    - :math:`\Delta x, \Delta y`: the plan dimensions of the well cell.
    - :math:`r_0`: area-equivalent radius standing in for the zero distance at the well cell.

    ``ray`` is one of ``(±1, ±1)``; ``None`` picks the diagonal reaching farthest.
    Returns predicted and reference phase pressures at the sampled cells.
    """
    from .metrics import predict_cells

    case, pack = bundle.case, bundle.well_pack
    aci = onp.asarray(case.aci)
    cells = _perf_cells(pack, w)
    if cells.size == 0:
        return {}
    i0, j0 = (int(onp.bincount(aci[cells, a]).argmax()) for a in (0, 1))
    k = int(aci[cells, 2].min()) if layer is None else int(layer)

    rays = [(1, 1), (1, -1), (-1, 1), (-1, -1)] if ray is None else [tuple(ray)]
    best = max((_ray_cells(case, i0, j0, k, si, sj) for si, sj in rays), key=len)
    if best.size == 0:
        return {}

    xy = onp.asarray(case.centroids, float)[best, :2]
    r = onp.linalg.norm(xy - xy[0][None, :], axis=1)
    plan = onp.asarray(case.cell_len, float)[best[0], :2]
    r[0] = float(onp.sqrt(max(float(plan[0] * plan[1]), 1e-12) / onp.pi))

    P = predict_cells(bundle, params, float(t))
    p_o_pred, p_w_pred, p_g_pred = _phase_pressures(bundle, P)
    t_idx = int(onp.abs(onp.asarray(case.times, float) - float(t)).argmin())
    return {"r": r, "cells": best, "layer": k, "t": float(t), "t_idx": t_idx,
            "well": pack.well_names[w],
            "BPR_o_pred": p_o_pred[best], "BPR_w_pred": p_w_pred[best],
            "BPR_g_pred": p_g_pred[best],
            "BPR_o_ref": onp.asarray(case.pres, float)[t_idx, best]}
