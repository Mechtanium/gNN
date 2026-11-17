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
    for opt in ("swof_pcow", "sgof_pcgo", "pvto_us_rs", "pvto_us_dinvb", "pvto_us_dinvbmu"):
        if opt in pack and pack[opt] is not None:
            out[opt] = jnp.asarray(pack[opt], dtype=REAL)
    # the deck's three-phase oil rel-perm model (a python string, static under jit); a deck
    # that names neither STONE1 nor STONE2 runs the default segregated model
    out["kro_model"] = str(pack.get("kro_model", "default")).lower()
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


def relperm_oil_stone1(sw: jnp.ndarray, sg: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    r"""
    Three-phase oil rel-perm via Stone's model I (the deck keyword ``STONE1``), in the form
    OPM Flow evaluates it — saturations normalized by the mobile pore space above connate
    water, with no minimum residual oil:

    .. math::

        k_{ro} \;=\; \min\!\Bigl(1,\ \beta\, \frac{k_{row}(S_w)\, k_{rog}(S_g)}{k_{rocw}}\Bigr),
        \qquad
        \beta \;=\; \frac{\hat S_o}{(1 - \hat S_w)(1 - \hat S_g)},
        \qquad
        \hat S_w = \frac{S_w - S_{wc}}{1 - S_{wc}},\;
        \hat S_g = \frac{S_g}{1 - S_{wc}},\;
        \hat S_o = 1 - \hat S_w - \hat S_g

    where:
    - :math:`k_{row}, k_{rog}`: the water–oil and gas–oil table columns, read at the cell's water and gas saturations; :math:`k_{rocw} = k_{row}(S_{wc})` the oil end point at connate water.
    - :math:`\beta = 1` at or below connate water (where the model reduces to :math:`k_{rog}`) and where :math:`\hat S_w` or :math:`\hat S_g` reaches one.

    Stone II is *not* a substitute here: at the water- and gas-coning saturations of SPE2's
    perforated cells (:math:`S_w \approx 0.38, S_g \approx 0.08`) it returns
    :math:`k_{ro} = 0` where Stone I gives a mobile oil and the simulator keeps producing.
    """
    swc = t["swof_sw"][0]
    krow = relperm_oil_water(sw, t)
    krog = relperm_oil_gas(sg, t)
    krocw = jnp.maximum(t["swof_krow"][0], jnp.asarray(1e-6, dtype=REAL))
    tiny = jnp.asarray(1e-9, dtype=REAL)
    span = jnp.maximum(1.0 - swc, tiny)
    ssw = (sw - swc) / span
    ssg = sg / span
    sso = 1.0 - ssw - ssg
    inside = (sw > swc) & (ssw < 1.0) & (ssg < 1.0)
    den = jnp.where(inside, (1.0 - ssw) * (1.0 - ssg), 1.0)
    beta = jnp.where(inside, sso / jnp.maximum(den, tiny), 1.0)
    return jnp.clip(beta * krow * krog / krocw, 0.0, 1.0)


# Zone-weight total below which the default model averages its two zone values (OPM's value).
_DEFAULT_KRO_EPS = 1e-5


def relperm_oil_default(sw: jnp.ndarray, sg: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    r"""
    Three-phase oil rel-perm under the default model of ECLIPSE and OPM Flow, the one a deck
    gets when it names neither ``STONE1`` nor ``STONE2``. Water and gas are taken as
    segregated within the cell, with the oil saturation the same in both zones:

    .. math::

        k_{ro} \;=\; \frac{S_g\, k_{rog}\bigl(S_g + S_w - S_{wc}\bigr)
                           + (S_w - S_{wc})\, k_{row}\bigl(S_w + S_g\bigr)}
                          {S_g + S_w - S_{wc}}

    where:
    - the gas zone holds the oil at :math:`S_o`, the connate water :math:`S_{wc}` and gas at :math:`S_g + S_w - S_{wc}`, so its oil rel-perm is the gas–oil table :math:`k_{rog}` read at that gas saturation; the water zone holds the oil at :math:`S_o` and water at :math:`S_w + S_g`, read from the water–oil table :math:`k_{row}`.
    - :math:`S_g` and :math:`S_w - S_{wc}` are the zones' weights. When both vanish the two values agree (:math:`k_{row}(S_{wc}) = k_{rog}(0) = k_{rc}`); for a weight total below :math:`\varepsilon = 10^{-5}` the two zone values are averaged, blended linearly into the weighted form between :math:`\varepsilon/2` and :math:`\varepsilon`, which keeps the function continuous.

    With water at connate saturation (SPE1 everywhere) it reduces to :math:`k_{rog}(S_g)`,
    and without gas to :math:`k_{row}(S_w)`.
    """
    swc = t["swof_sw"][0]
    w_w = jnp.maximum(sw - swc, 0.0)                     # water-zone weight
    d = sg + w_w                                         # the weights' total, from the same values
    krog = relperm_oil_gas(d, t)
    krow = relperm_oil_water(swc + d, t)
    eps = jnp.asarray(_DEFAULT_KRO_EPS, dtype=REAL)
    weighted = (sg * krog + w_w * krow) / jnp.maximum(d, 0.5 * eps)
    alpha = jnp.clip((eps - d) / (0.5 * eps), 0.0, 1.0)
    return jnp.clip(alpha * 0.5 * (krog + krow) + (1.0 - alpha) * weighted, 0.0, None)


KRO_MODELS = ("default", "stone1", "stone2")


def relperm_oil_3p(sw: jnp.ndarray, sg: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Three-phase oil rel-perm under the model the deck declares (``t["kro_model"]``):
    ``"stone1"`` for ``STONE1``, ``"stone2"`` for ``STONE2``, and the segregated default
    model (:func:`relperm_oil_default`) when the deck names neither."""
    model = str(t.get("kro_model", "default")).lower()
    if model == "stone1":
        return relperm_oil_stone1(sw, sg, t)
    if model == "stone2":
        return relperm_oil_stone2(sw, sg, t)
    return relperm_oil_default(sw, sg, t)


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


def _seff(num: jnp.ndarray, den: jnp.ndarray) -> jnp.ndarray:
    """Normalized effective saturation ``num/den`` clipped to [floor, 1]."""
    return jnp.clip(num / jnp.maximum(den, _SE_FLOOR), _SE_FLOOR, 1.0)


def corey_water(sw, s_wc, s_orw, krw_max, n_w):
    r"""
    Corey water relative permeability (Eq 3.10 family).

    .. math::

        k_{rw}(S_w) = k_{rw}^{\max}\, \left( \frac{S_w - S_{wc}}{1 - S_{wc} - S_{orw}} \right)^{n_w}

    where:
    - :math:`S_{wc}`: connate water saturation (zero-mobility endpoint).
    - :math:`S_{orw}`: residual oil saturation to water (the other endpoint).
    - :math:`k_{rw}^{\max}`: water endpoint relative permeability at :math:`S_w = 1 - S_{orw}`.
    - :math:`n_w`: the Corey exponent (curvature).
    """
    se = _seff(sw - s_wc, 1.0 - s_wc - s_orw)
    return krw_max * se ** n_w


def corey_oil_water(sw, s_wc, s_orw, kro_max, n_ow):
    r"""
    Corey oil relative permeability in the oil-water system.

    .. math::

        k_{row}(S_w) = k_{ro}^{\max}\, \left( \frac{1 - S_w - S_{orw}}{1 - S_{wc} - S_{orw}} \right)^{n_{ow}}

    where:
    - :math:`k_{ro}^{\max}`: oil endpoint relative permeability at :math:`S_w = S_{wc}`.
    - :math:`n_{ow}`: the oil-in-water Corey exponent (shared with :func:`corey_oil_gas`).
    """
    se = _seff(1.0 - sw - s_orw, 1.0 - s_wc - s_orw)
    return kro_max * se ** n_ow


def corey_gas(sg, s_wc, s_gc, s_org, krg_max, n_g):
    r"""
    Corey gas relative permeability (SGOF convention: :math:`S_w = S_{wc}`).

    .. math::

        k_{rg}(S_g) = k_{rg}^{\max}\, \left( \frac{S_g - S_{gc}}{1 - S_{wc} - S_{gc} - S_{org}} \right)^{n_g}

    where:
    - :math:`S_{gc}`: critical gas saturation (mobility onset).
    - :math:`S_{org}`: residual oil saturation to gas.
    - :math:`k_{rg}^{\max}, n_g`: gas endpoint relative permeability and Corey exponent.
    """
    se = _seff(sg - s_gc, 1.0 - s_wc - s_gc - s_org)
    return krg_max * se ** n_g


def corey_oil_gas(sg, s_wc, s_org, kro_max, n_og):
    r"""
    Corey oil relative permeability in the gas-oil system (SGOF convention).

    .. math::

        k_{rog}(S_g) = k_{ro}^{\max}\, \left( \frac{1 - S_{wc} - S_{org} - S_g}{1 - S_{wc} - S_{org}} \right)^{n_{og}}

    where:
    - :math:`k_{ro}^{\max}`: the shared oil endpoint (equal to :func:`corey_oil_water` at :math:`S_w = S_{wc}`, keeping the Stone-II endpoint :math:`k_{rc}` consistent).
    - :math:`n_{og}`: the oil-in-gas Corey exponent.
    """
    se = _seff(1.0 - s_wc - s_org - sg, 1.0 - s_wc - s_org)
    return kro_max * se ** n_og


def brooks_corey_pc(s_wet, s_min, p_e, lam, se_floor: float = 0.05):
    r"""
    Brooks-Corey capillary pressure on the wetting-phase saturation.

    .. math::

        p_c(S) = p_e\, S_e^{-1/\lambda},
        \qquad
        S_e = \frac{S - S_{\min}}{1 - S_{\min}}

    where:
    - :math:`p_e`: the entry (displacement) pressure [psi].
    - :math:`\lambda`: the pore-size-distribution index (larger = more uniform pores, flatter curve).
    - :math:`S_e`: the effective wetting saturation, clipped to ``[se_floor, 1]`` so the curve stays bounded at the dry end (:math:`p_c \le p_e\, s_{\mathrm{floor}}^{-1/\lambda}`).
    """
    se = jnp.clip((s_wet - s_min) / jnp.maximum(1.0 - s_min, _SE_FLOOR), se_floor, 1.0)
    return p_e * se ** (-1.0 / lam)


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
    r"""
    Dry-gas FVF from PVDG, interpolated as the simulator does — linearly in its inverse,

    .. math::

        B_g(p) \;=\; \Bigl(\mathcal I\bigl[p;\ p_j,\ 1/B_{g,j}\bigr]\Bigr)^{-1}

    where:
    - :math:`\mathcal I[p; p_j, y_j]`: piecewise-linear interpolation of the table values :math:`y_j` at the pressure nodes :math:`p_j` (held constant beyond the ends). :math:`B_g \propto 1/p` is far from linear, so interpolating :math:`B_g` itself between 400 psi nodes would differ from the simulator by about a percent.
    """
    return 1.0 / _interp(pg, t["pvdg_pressure"], 1.0 / t["pvdg_fvf"])


def visc_gas(pg: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    r"""
    Gas viscosity from PVDG through the simulator's interpolants, the inverse FVF and the
    inverse mobility product, :math:`\mu_g = \mathcal I[1/B_g] \,/\, \mathcal I[1/(\mu_g B_g)]`
    (see :func:`fvf_gas`).
    """
    inv_b = _interp(pg, t["pvdg_pressure"], 1.0 / t["pvdg_fvf"])
    return inv_b / _interp(pg, t["pvdg_pressure"], 1.0 / (t["pvdg_fvf"] * t["pvdg_visc"]))


def _undersaturated(rs: jnp.ndarray, p: jnp.ndarray | None, t: dict[str, jnp.ndarray]):
    r"""
    The live oil's inverse FVF and inverse mobility product at :math:`(p, R_s)`,

    .. math::

        \frac{1}{B_o} \;=\; \frac{1}{B_o^{\mathrm{sat}}(R_s)} + \beta_B(R_s)\,[\,p - p_b(R_s)\,]_+,
        \qquad
        \frac{1}{B_o \mu_o} \;=\; \frac{1}{B_o^{\mathrm{sat}}(R_s)\,\mu_o^{\mathrm{sat}}(R_s)} + \beta_{B\mu}(R_s)\,[\,p - p_b(R_s)\,]_+

    where:
    - :math:`B_o^{\mathrm{sat}}, \mu_o^{\mathrm{sat}}, p_b`: the saturated FVF, viscosity and bubble-point pressure of oil holding :math:`R_s` (Eq 2.70).
    - :math:`\beta_B, \beta_{B\mu}`: the slopes of the deck's undersaturated PVTO branch in the two inverse quantities (:func:`modules.utils.ReservoirExtractors.deck_pvto_undersaturated`), interpolated between nodes — the simulator interpolates the branch linearly in exactly these quantities; :math:`[x]_+ = \max(x, 0)`, so saturated oil (:math:`p = p_b(R_s)`), a call without ``p`` and tables without a branch all return the saturated values.

    Returns ``(1/B_o, 1/(B_o mu_o))``.
    """
    b_sat = _interp(rs, t["pvto_rs_sat"], t["pvto_fvf_sat"])
    mu_sat = _interp(rs, t["pvto_rs_sat"], t["pvto_visc_sat"])
    inv_b, inv_bmu = 1.0 / b_sat, 1.0 / (b_sat * mu_sat)
    if p is None or "pvto_us_dinvb" not in t:
        return inv_b, inv_bmu
    rs = jnp.asarray(rs)
    dp = jnp.maximum(jnp.asarray(p).astype(_query_dtype(rs))
                     - _interp(rs, t["pvto_rs_sat"], t["pvto_pbub_sat"]), 0.0)
    return (inv_b + _interp(rs, t["pvto_us_rs"], t["pvto_us_dinvb"]) * dp,
            inv_bmu + _interp(rs, t["pvto_us_rs"], t["pvto_us_dinvbmu"]) * dp)


def fvf_oil(rs: jnp.ndarray, t: dict[str, jnp.ndarray], p: jnp.ndarray | None = None) -> jnp.ndarray:
    r"""Oil FVF from PVTO: the saturated curve at the solution GOR, compressed above the
    bubble point along the deck's undersaturated branch (:func:`_undersaturated`)."""
    return 1.0 / _undersaturated(rs, p, t)[0]


def visc_oil(rs: jnp.ndarray, t: dict[str, jnp.ndarray], p: jnp.ndarray | None = None) -> jnp.ndarray:
    r"""Oil viscosity from PVTO, :math:`\mu_o = (1/B_o) / (1/(B_o \mu_o))` with both inverse
    quantities from :func:`_undersaturated`."""
    inv_b, inv_bmu = _undersaturated(rs, p, t)
    return inv_b / inv_bmu


def fvf_water(pw: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    r"""
    :math:`B_w(p)` from the PVTW reference and compressibility, in the simulator's
    second-order form,

    .. math::

        B_w \;=\; \frac{B_{w,\mathrm{ref}}}{1 + X + X^2/2},
        \qquad
        X = c_w\,(p - p_{\mathrm{ref}})

    where :math:`c_w` is the water compressibility and :math:`p_{\mathrm{ref}}, B_{w,\mathrm{ref}}`
    the reference pressure and FVF. The result dtype follows the query (:func:`_query_dtype`):
    the ``float32`` table scalars promote against a ``float64`` ``pw`` instead of demoting it.
    """
    pw = jnp.asarray(pw)
    x = t["pvtw_compr"][0] * (pw.astype(_query_dtype(pw)) - t["pvtw_pressure"][0])
    return t["pvtw_fvf"][0] / jnp.maximum(1.0 + x * (1.0 + 0.5 * x), 1e-6)


def visc_water(pw: jnp.ndarray, t: dict[str, jnp.ndarray]) -> jnp.ndarray:
    r"""
    :math:`\mu_w(p)` from the PVTW reference, compressibility and viscosibility, in the
    simulator's form,

    .. math::

        \mu_w \;=\; \frac{\mu_{w,\mathrm{ref}}\, B_{w,\mathrm{ref}}}{B_w\,(1 + Y + Y^2/2)},
        \qquad
        Y = (c_w - c_v)\,(p - p_{\mathrm{ref}})

    where :math:`c_v` is the viscosibility and :math:`B_w` follows :func:`fvf_water`; a
    zero viscosibility leaves :math:`\mu_w = \mu_{w,\mathrm{ref}}` to second order.
    """
    pw = jnp.asarray(pw)
    y = (t["pvtw_compr"][0] - t["pvtw_viscosibility"][0]) * (
        pw.astype(_query_dtype(pw)) - t["pvtw_pressure"][0])
    return (t["pvtw_visc"][0] * t["pvtw_fvf"][0]
            / (fvf_water(pw, t) * jnp.maximum(1.0 + y * (1.0 + 0.5 * y), 1e-6)))


# --------------------------------------------------------------------------- #
# Surface densities                                                            #
# --------------------------------------------------------------------------- #
def surface_densities(t: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """(rho_o,s, rho_w,s, rho_g,s) standard/surface densities (DENSITY table)."""
    return t["dens_o"][0], t["dens_w"][0], t["dens_g"][0]


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
    bo, muo = fvf_oil(rs, t, p_oil), visc_oil(rs, t, p_oil)
    bw, muw = fvf_water(p_water, t), visc_water(p_water, t)
    bg, mug = fvf_gas(p_gas, t), visc_gas(p_gas, t)
    eps = jnp.asarray(1e-12, dtype=REAL)
    f_o = kro / jnp.maximum(muo * bo, eps)
    f_w = krw / jnp.maximum(muw * bw, eps)
    f_g = krg / jnp.maximum(mug * bg, eps)
    return jnp.stack([f_o, f_w, f_g], axis=0)


def formation_volume_factors(
    p_water: jnp.ndarray,
    p_gas: jnp.ndarray,
    rs: jnp.ndarray,
    t: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Return ``(B_o, B_w, B_g)`` stacked as ``(3, *state)`` for storage terms."""
    return jnp.stack([fvf_oil(rs, t), fvf_water(p_water, t), fvf_gas(p_gas, t)], axis=0)


# --------------------------------------------------------------------------- #
# Total compressibility                                                        #
# --------------------------------------------------------------------------- #
def _table_slope(xp: jnp.ndarray, fp: jnp.ndarray) -> jnp.ndarray:
    """Central-difference slope d(fp)/d(xp) evaluated on the table nodes."""
    return jnp.gradient(fp, xp)


def total_compressibility(
    p_oil: jnp.ndarray,
    p_water: jnp.ndarray,
    p_gas: jnp.ndarray,
    s_oil: jnp.ndarray,
    s_water: jnp.ndarray,
    s_gas: jnp.ndarray,
    rs: jnp.ndarray,
    t: dict[str, jnp.ndarray],
    *,
    mode: str = "full",
) -> jnp.ndarray:
    """Total compressibility c_t (Eqs 3.16, 2.112 practical form).

    ``mode="full"``  : c_t = c_R + S_o c_o + S_w c_w + S_g c_g, with
        c_w = pvtw_compr, c_R = ROCK, c_g = -(1/B_g) dB_g/dp,
        c_o = -(1/B_o) dB_o/dp_bub mapped through the saturated R_s -> p_bub line.
    ``mode="rock_water"`` : c_t = c_R + S_w c_w  (robust fallback).
    """
    c_r = t["rock_compr"]
    c_w = t["pvtw_compr"][0]
    if mode == "rock_water":
        return c_r + s_water * c_w

    bg = fvf_gas(p_gas, t)
    dbg = _interp(p_gas, t["pvdg_pressure"], _table_slope(t["pvdg_pressure"], t["pvdg_fvf"]))
    c_g = jnp.clip(-dbg / jnp.maximum(bg, 1e-12), 0.0, None)

    bo = fvf_oil(rs, t)
    dbo = _interp(rs, t["pvto_rs_sat"], _table_slope(t["pvto_pbub_sat"], t["pvto_fvf_sat"]))
    c_o = jnp.clip(-dbo / jnp.maximum(bo, 1e-12), 0.0, None)

    return c_r + s_oil * c_o + s_water * c_w + s_gas * c_g
