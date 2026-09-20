"""Differentiable JAX black-oil property closures.

Evaluate per-phase reservoir-fluid properties at the current state (oil
pressure, phase pressures, saturations, dissolved gas-oil ratio ``R_s``) from
the cached DeepField black-oil tables produced by
``ReservoirMesh.build_blackoil_table_pack``.

All functions are pure ``jax.numpy`` and broadcast over arbitrary leading
shapes (typically ``(n_vertices,)``).  Symbols follow Chen, Huan & Ma,
*Computational Methods for Multiphase Flows in Porous Media* (SIAM, 2006):

- ``B_alpha``  : phase formation volume factor (Eqs 2.70, 2.73-2.75).
- ``mu_alpha`` : phase viscosity.
- ``k_r,alpha``: phase relative permeability (Eqs 3.4-3.8, 3.15).
- ``rho_alpha,s``: standard/surface phase density (DENSITY table).
- ``c_t``      : total compressibility (Eqs 3.16, 2.112 practical form).

FIELD units are assumed throughout (psia, cP, rb/stb, lb/ft^3, psi^-1).
The Delta residual uses simulator surface-volume phase rates, so phase
mobilities are ``k_r / (mu B)`` and not density-weighted.

Precision: tables are stored at :data:`REAL` (float32), but every closure
follows the *query's* floating dtype (see ``_query_dtype``) — a float64 query
from a selectively-promoted AD chain is evaluated and returned in float64,
never silently demoted mid-tape; float32 callers are bit-unchanged.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

REAL = jnp.float32

FIELD_DARCY_COEFF = 0.0011271161434266812
RB_PER_FT3 = 1.0 / 5.614583333

# Table keys consumed from ``build_blackoil_table_pack``.
_REQUIRED_KEYS = (
    "swof_sw", "swof_krw", "swof_krow",
    "sgof_sg", "sgof_krg", "sgof_krog",
    "pvtw_pressure", "pvtw_fvf", "pvtw_compr", "pvtw_visc", "pvtw_viscosibility",
    "pvdg_pressure", "pvdg_fvf", "pvdg_visc",
    "pvto_rs_sat", "pvto_pbub_sat", "pvto_fvf_sat", "pvto_visc_sat",
    "dens_o", "dens_w", "dens_g", "swc_baker",
)


def tables_to_jax(pack: dict[str, Any]) -> dict[str, jnp.ndarray]:
    """Convert a numpy black-oil pack into JAX arrays (float32).

    Scalars (``swc_baker``, ``rock_compr``, ``rock_pref``) become 0-d arrays.
    Missing ``rock_*`` keys default to 0 so callers can omit ROCK.
    """
    missing = [key for key in _REQUIRED_KEYS if key not in pack]
    if missing:
        raise KeyError(f"black-oil pack missing required keys: {missing}")
    out = {key: jnp.asarray(pack[key], dtype=REAL) for key in _REQUIRED_KEYS}
    out["rock_compr"] = jnp.asarray(pack.get("rock_compr", 0.0), dtype=REAL)
    out["rock_pref"] = jnp.asarray(pack.get("rock_pref", 0.0), dtype=REAL)
    # Optional capillary-pressure columns (present in SWOF/SGOF; default to zero
    # so callers may omit them). p_cow = p_o - p_w, p_cgo = p_g - p_o (Eq 8.6).
    for opt in ("swof_pcow", "sgof_pcgo"):
        if opt in pack:
            out[opt] = jnp.asarray(pack[opt], dtype=REAL)
    # the deck's three-phase oil rel-perm model (a python string, static under jit)
    out["kro_model"] = str(pack.get("kro_model", "stone2")).lower()
    return out


def _query_dtype(x: jnp.ndarray) -> jnp.dtype:
    r"""Floating dtype the closure result should carry for query ``x``.

    Dtype-follows-query contract: a floating query keeps its own precision
    (``float64`` queries stay ``float64`` so a selectively-promoted AD chain is
    never demoted mid-tape); non-floating queries fall back to :data:`REAL`.
    """
    return x.dtype if jnp.issubdtype(x.dtype, jnp.floating) else REAL


def _interp(x: jnp.ndarray, xp: jnp.ndarray, fp: jnp.ndarray) -> jnp.ndarray:
    r"""1-D linear interpolation, clamped to table endpoints (like np.interp).

    Evaluates in the query's floating dtype (see :func:`_query_dtype`): the
    table columns ``xp``/``fp`` are cast *up* to match ``x``, never ``x`` down
    to the table's :data:`REAL`, so ``float64`` queries return ``float64``
    (bit-identical to the previous behaviour for ``float32`` queries).
    """
    x = jnp.asarray(x)
    dt = _query_dtype(x)
    return jnp.interp(x.astype(dt), xp.astype(dt), fp.astype(dt))


# --------------------------------------------------------------------------- #
# Relative permeability                                                        #
# --------------------------------------------------------------------------- #
def relperm_water(sw: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """k_rw(S_w) from SWOF (Eq 3.4)."""
    return _interp(sw, t["swof_sw"], t["swof_krw"])


def relperm_gas(sg: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """k_rg(S_g) from SGOF (Eq 3.4)."""
    return _interp(sg, t["sgof_sg"], t["sgof_krg"])


def relperm_oil_water(sw: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """k_row(S_w): oil rel-perm in the oil-water system (Eq 3.6)."""
    return _interp(sw, t["swof_sw"], t["swof_krow"])


def relperm_oil_gas(sg: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """k_rog(S_g): oil rel-perm in the oil-gas system (Eq 3.7)."""
    return _interp(sg, t["sgof_sg"], t["sgof_krog"])


def _krc(t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Endpoint k_rc = k_row(S_wc) = k_rog(S_g=0), guarded > 0 (Eq 3.13)."""
    krc = jnp.maximum(t["swof_krow"][0], t["sgof_krog"][0])
    return jnp.maximum(krc, jnp.asarray(1e-6, dtype=REAL))


def relperm_oil_stone2(sw: jnp.ndarray, sg: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Three-phase oil rel-perm via Stone's model II (Eq 3.15).

    k_ro = k_rc {(k_row/k_rc + k_rw)(k_rog/k_rc + k_rg) - (k_rw + k_rg)},
    clamped to >= 0 (negative values denote immobile oil).
    """
    krw = relperm_water(sw, t)
    krg = relperm_gas(sg, t)
    krow = relperm_oil_water(sw, t)
    krog = relperm_oil_gas(sg, t)
    krc = _krc(t)
    kro = krc * ((krow / krc + krw) * (krog / krc + krg) - (krw + krg))
    return jnp.clip(kro, 0.0, None)


def _table_residuals(t: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""
    The saturation endpoints Stone I needs, read off the (possibly parametric) table
    columns: :math:`(S_{wc}, S_{orw}, S_{org})` — the connate water, the residual oil to
    water (first :math:`S_w` with :math:`k_{row} = 0`) and the residual oil to gas
    (first :math:`S_g` with :math:`k_{rog} = 0`, so :math:`S_{org} = 1 - S_{wc} - S_g`).
    A column that never reaches zero contributes its last node.
    """
    sw, krow = t["swof_sw"], t["swof_krow"]
    sg, krog = t["sgof_sg"], t["sgof_krog"]
    swc = sw[0]
    zero_w = krow <= 0.0
    i_w = jnp.where(jnp.any(zero_w), jnp.argmax(zero_w), sw.shape[0] - 1)
    zero_g = krog <= 0.0
    i_g = jnp.where(jnp.any(zero_g), jnp.argmax(zero_g), sg.shape[0] - 1)
    s_orw = 1.0 - sw[i_w]
    s_org = 1.0 - swc - sg[i_g]
    return swc, s_orw, s_org


def relperm_oil_stone1(sw: jnp.ndarray, sg: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    r"""
    Three-phase oil rel-perm via Stone's model I (the deck keyword ``STONE1``), in the
    normalized form of Aziz & Settari with the Fayers–Matthews residual-oil interpolation
    that ECLIPSE and OPM Flow apply:

    .. math::

        k_{ro} \;=\; k_{rc}\, S_o^{\ast}\,
        \frac{k_{row}(S_w)/k_{rc}}{1 - S_w^{\ast}}\,
        \frac{k_{rog}(S_g)/k_{rc}}{1 - S_g^{\ast}},
        \qquad
        S_o^{\ast} = \frac{S_o - S_{om}}{1 - S_{wc} - S_{om}},\;
        S_w^{\ast} = \frac{S_w - S_{wc}}{1 - S_{wc} - S_{om}},\;
        S_g^{\ast} = \frac{S_g}{1 - S_{wc} - S_{om}}

    .. math::

        S_{om} \;=\; \alpha\, S_{orw} + (1 - \alpha)\, S_{org},
        \qquad
        \alpha \;=\; 1 - \frac{S_g}{1 - S_{wc} - S_{org}}

    where:
    - :math:`k_{rc}`: the oil endpoint :math:`k_{row}(S_{wc}) = k_{rog}(0)`; :math:`k_{row}, k_{rog}` the two-phase table columns.
    - :math:`S_{om}`: the minimum (three-phase) residual oil, interpolated between the water-flood and gas-flood residuals :math:`S_{orw}, S_{org}` by the gas saturation.
    - the normalized saturations are clamped to :math:`[0, 1]`; :math:`k_{ro} = 0` once :math:`S_o \le S_{om}`.

    Stone II is *not* a substitute here: at the water- and gas-coning saturations of SPE2's
    perforated cells (:math:`S_w \approx 0.38, S_g \approx 0.08`) it returns
    :math:`k_{ro} = 0` where Stone I gives :math:`0.07` and the simulator keeps producing.
    """
    swc, s_orw, s_org = _table_residuals(t)
    so = 1.0 - sw - sg
    tiny = jnp.asarray(1e-9, dtype=REAL)
    alpha = 1.0 - sg / jnp.maximum(1.0 - swc - s_org, tiny)
    s_om = alpha * s_orw + (1.0 - alpha) * s_org
    den = jnp.maximum(1.0 - swc - s_om, tiny)
    so_n = jnp.clip((so - s_om) / den, 0.0, 1.0)
    sw_n = jnp.clip((sw - swc) / den, 0.0, 1.0)
    sg_n = jnp.clip(sg / den, 0.0, 1.0)
    krow = relperm_oil_water(sw, t)
    krog = relperm_oil_gas(sg, t)
    krc = _krc(t)
    kro = krc * so_n * (krow / krc / jnp.maximum(1.0 - sw_n, tiny)) * (krog / krc / jnp.maximum(1.0 - sg_n, tiny))
    return jnp.clip(kro, 0.0, None)


KRO_MODELS = ("stone1", "stone2")


def relperm_oil_3p(sw: jnp.ndarray, sg: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Three-phase oil rel-perm under the model the deck declares (``t["kro_model"]``:
    ``"stone1"`` for ``STONE1``, otherwise Stone II — the historical default, which the
    decks without a keyword (SPE1: water at connate saturation everywhere, where every
    model reduces to ``k_rog``) already ran on)."""
    model = str(t.get("kro_model", "stone2")).lower()
    if model == "stone1":
        return relperm_oil_stone1(sw, sg, t)
    return relperm_oil_stone2(sw, sg, t)


# --------------------------------------------------------------------------- #
# Parametric closure curves (inversion targets)                                #
# --------------------------------------------------------------------------- #
# The saturation-inversion pipeline replaces the SCAL table COLUMNS with these
# smooth parametric forms evaluated at the table's saturation nodes, so every
# existing interpolating closure (and its gradients) flows to the curve
# parameters unchanged. Effective saturations are clipped to a small positive
# floor: `x**n` is gradient-safe there, whereas d/dn x**n = x**n log(x) is NaN
# at exactly x = 0.
_SE_FLOOR = 1e-6


# --------------------------------------------------------------------------- #
# Capillary pressures                                                          #
# --------------------------------------------------------------------------- #
def cap_pres_ow(sw: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Oil-water capillary pressure p_cow(S_w) = p_o - p_w from SWOF (Eq 8.6, 8.14).

    Returns zeros if the ``swof_pcow`` column is absent (e.g. SPE1, where
    p_cow == 0 everywhere). Result dtype follows the query (:func:`_query_dtype`).
    """
    if "swof_pcow" not in t:
        sw = jnp.asarray(sw)
        return jnp.zeros_like(sw, dtype=_query_dtype(sw))
    return _interp(sw, t["swof_sw"], t["swof_pcow"])


def cap_pres_go(sg: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Gas-oil capillary pressure p_cgo(S_g) = p_g - p_o from SGOF (Eq 8.6, 8.14).

    Returns zeros if the ``sgof_pcgo`` column is absent. Result dtype follows
    the query (:func:`_query_dtype`).
    """
    if "sgof_pcgo" not in t:
        sg = jnp.asarray(sg)
        return jnp.zeros_like(sg, dtype=_query_dtype(sg))
    return _interp(sg, t["sgof_sg"], t["sgof_pcgo"])


# --------------------------------------------------------------------------- #
# Formation volume factors and viscosities                                    #
# --------------------------------------------------------------------------- #
def fvf_gas(pg: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return _interp(pg, t["pvdg_pressure"], t["pvdg_fvf"])


def visc_gas(pg: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return _interp(pg, t["pvdg_pressure"], t["pvdg_visc"])


def fvf_oil(rs: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Saturated oil FVF B_o(R_s) from PVTO (Eq 2.70)."""
    return _interp(rs, t["pvto_rs_sat"], t["pvto_fvf_sat"])


def visc_oil(rs: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Saturated oil viscosity mu_o(R_s) from PVTO."""
    return _interp(rs, t["pvto_rs_sat"], t["pvto_visc_sat"])


def fvf_water(pw: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    r"""B_w(p) from PVTW reference + compressibility (first-order ECLIPSE form).

    Result dtype follows the query (:func:`_query_dtype`): the ``float32``
    table scalars promote against a ``float64`` ``pw`` instead of demoting it.
    """
    pw = jnp.asarray(pw)
    b_ref = t["pvtw_fvf"][0]
    p_ref = t["pvtw_pressure"][0]
    cw = t["pvtw_compr"][0]
    denom = jnp.maximum(1.0 + cw * (pw.astype(_query_dtype(pw)) - p_ref), 1e-6)
    return b_ref / denom


def visc_water(pw: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    r"""mu_w(p) from PVTW reference + viscosibility (ECLIPSE form; const if 0).

    Result dtype follows the query (:func:`_query_dtype`), as in :func:`fvf_water`.
    """
    pw = jnp.asarray(pw)
    mu_ref = t["pvtw_visc"][0]
    p_ref = t["pvtw_pressure"][0]
    cv = t["pvtw_viscosibility"][0]
    denom = jnp.maximum(1.0 - cv * (pw.astype(_query_dtype(pw)) - p_ref), 1e-6)
    return mu_ref / denom


# --------------------------------------------------------------------------- #
# Surface densities                                                            #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Per-phase mobility factor f_alpha = k_r,alpha / (mu_alpha * B_alpha).        #
# --------------------------------------------------------------------------- #
def phase_factors(
    p_oil: jnp.ndarray,
    p_water: jnp.ndarray,
    p_gas: jnp.ndarray,
    s_water: jnp.ndarray,
    s_gas: jnp.ndarray,
    rs: jnp.ndarray,
    t: dict[str, jnp.ndarray],
    *,
    kr_floor: float = 0.0,
) -> jnp.ndarray:
    """Return stacked per-phase factors of shape ``(3, *state)`` for (o, w, g).

    f_alpha = (k_r,alpha + kr_floor) / (mu_alpha * B_alpha).
    Multiplying the diagonal absolute permeability ``k`` by ``f_alpha`` yields the
    phase mobility tensor used by the surface-volume balance. FIELD-unit Darcy
    conversion is applied during FEM assembly, not in this closure.

    ``kr_floor`` adds a small residual relative permeability to every phase so an
    immobile phase (k_r = 0 at connate/zero saturation) still yields a non-zero,
    well-posed stiffness ``K_alpha`` instead of the zero matrix (whose generalized
    eigenbasis is degenerate).  With ``kr_floor`` active the immobile-phase block
    reduces to the geometric (absolute-permeability) Laplacian basis modulated by
    ``mu_alpha``/``B_alpha`` rather than collapsing.
    """
    kr_floor = jnp.asarray(kr_floor, dtype=REAL)
    kro = relperm_oil_3p(s_water, s_gas, t) + kr_floor
    krw = relperm_water(s_water, t) + kr_floor
    krg = relperm_gas(s_gas, t) + kr_floor
    bo, muo = fvf_oil(rs, t), visc_oil(rs, t)
    bw, muw = fvf_water(p_water, t), visc_water(p_water, t)
    bg, mug = fvf_gas(p_gas, t), visc_gas(p_gas, t)
    eps = jnp.asarray(1e-12, dtype=REAL)
    f_o = kro / jnp.maximum(muo * bo, eps)
    f_w = krw / jnp.maximum(muw * bw, eps)
    f_g = krg / jnp.maximum(mug * bg, eps)
    return jnp.stack([f_o, f_w, f_g], axis=0)


# --------------------------------------------------------------------------- #
# Total compressibility                                                        #
# --------------------------------------------------------------------------- #
