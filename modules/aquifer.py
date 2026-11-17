r"""
Fetkovich aquifers: the water a deck's analytic aquifers (``AQUFETP`` / ``AQUANCON``) push
into the reservoir, and the material balance that sets their pressure.

An aquifer is a tank of water of initial volume :math:`V_w` and total compressibility
:math:`c_t`, connected to reservoir cell faces (Fetkovich, 1971). Water crosses connection
:math:`c` at the rate

.. math::

    q_c(t) \;=\; J\,\alpha_c\,\Bigl(p_a(t) + \gamma_w\,(z_c - z_d) - p_{w,c}(t)\Bigr),
    \qquad
    c_t V_w\,\bigl(p_{a,0} - p_a(t)\bigr) \;=\; W_e(t) \;=\; \int_0^t \sum_c q_c(\tau)\,\mathrm d\tau

where:
- :math:`J`: the aquifer's productivity index (stb/d/psi) and :math:`\alpha_c` the share of connection :math:`c` — its face area (or the deck's influx coefficient) times its multiplier, normalized over the aquifer.
- :math:`p_a(t)`: the aquifer pressure at its datum depth :math:`z_d`; :math:`p_{a,0}` its initial value, by default the equilibrium pressure of the connected cells carried to the datum (:func:`build_aquifer_pack`).
- :math:`p_{w,c}(t) = p_{o,c}(t) - P_{c,ow}(S_{w,c}(t))`, :math:`z_c`: the water pressure and the depth of the connected cell — water is the phase that crosses the face, so the oil pressure is corrected by the oil-water capillary pressure.
- :math:`\gamma_w = g\,\rho_w(p_{a,0})`: the gradient (psi/ft) of the aquifer's water at its initial pressure, so :math:`p_a + \gamma_w (z_c - z_d)` is the aquifer's pressure at the cell's depth (depth grows downward).
- :math:`W_e(t)`: the cumulative influx — the water the tank has given up, which is what lowers its pressure. A reservoir pressed above the aquifer drives water back into it (:math:`q_c < 0`).

A simulator advances the same model one step at a time. Over a step :math:`\Delta t` it holds the cell pressures at their new values and the aquifer pressure at its old one, and integrates the tank's exponential relaxation exactly, which scales the rate by

.. math::

    \frac{1 - e^{-\Delta t / T_c}}{\Delta t / T_c},
    \qquad
    T_c \;=\; \frac{c_t V_w}{J},

where :math:`T_c` is the aquifer's time constant. The factor tends to one as :math:`\Delta t / T_c \to 0`, which recovers the continuous model above.

In the surrogate, :math:`q_c` is a source of the water balance spread evenly over the
connected face's four nodes, the cell's water pressure is the network's at the cell centroid
(as for the wells), and :math:`p_a(t)` is a trainable head at the report times — a range-anchored
sigmoid, linearly interpolated in time like the predicted well head
(:class:`modules.wells.WellHead`). The ``aqu`` loss group holds the head to the material
balance at every report time, with the cumulative influx integrated by the trapezoidal rule
over the report steps,

.. math::

    r_{a0} \;=\; \frac{p_a(t_0) - p_{a,0}}{s_p},
    \qquad
    r_{aj} \;=\; \frac{c_t V_w\,\bigl(p_{a,0} - p_a(t_j)\bigr) - W_e(t_j)}{c_t V_w\,s_p},
    \quad j = 1, \dots, T - 1

where :math:`s_p` is the case's pressure scale, so a row is the aquifer-pressure error in
units of that scale.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as onp

from .casedata import CaseData
from .config import RunConfig, case_paths

# Corners of the canonical hexahedron on each face (VTK order: reference axes I, J, K).
FACE_CORNERS = {"I-": (0, 3, 4, 7), "I+": (1, 2, 5, 6), "J-": (0, 1, 4, 5),
                "J+": (2, 3, 6, 7), "K-": (0, 1, 2, 3), "K+": (4, 5, 6, 7)}
_FACE_STEP = {"I-": (-1, 0, 0), "I+": (1, 0, 0), "J-": (0, -1, 0), "J+": (0, 1, 0),
              "K-": (0, 0, -1), "K+": (0, 0, 1)}
_HEAD_EPS = 1e-3


@functools.lru_cache(maxsize=16)
def _deck_aquifers(path: str, mtime: float) -> tuple:
    from modules.utils.ReservoirExtractors import deck_fetkovich_aquifers
    from modules.utils.unit_conversion import convert_aquifers, detect_unit_system

    return tuple(convert_aquifers(deck_fetkovich_aquifers(path), detect_unit_system(path)))


def deck_aquifers(cfg: RunConfig) -> tuple:
    """The deck's Fetkovich aquifers in FIELD units (empty without any), read once per deck
    file version (:func:`modules.utils.ReservoirExtractors.deck_fetkovich_aquifers`)."""
    path = Path(case_paths(cfg)[0])
    if not path.exists():
        return ()
    return _deck_aquifers(str(path), path.stat().st_mtime)


@dataclass
class AquiferPack:
    """The aquifers of one case and their connections, host-side."""

    n_aq: int
    times: onp.ndarray            # (T,) report times [d]
    pi: onp.ndarray               # (n_aq,) productivity index [stb/d/psi]
    vw: onp.ndarray               # (n_aq,) initial water volume [stb]
    ct: onp.ndarray               # (n_aq,) total compressibility [1/psi]
    p0: onp.ndarray               # (n_aq,) initial pressure at the datum [psia]
    datum: onp.ndarray            # (n_aq,) datum depth [ft]
    conn_cell: onp.ndarray        # (n_conn,) connected cell
    conn_aq: onp.ndarray          # (n_conn,) its aquifer
    conn_alpha: onp.ndarray       # (n_conn,) share of the aquifer's J (sums to 1 per aquifer)
    conn_dz: onp.ndarray          # (n_conn,) cell-centroid depth minus the datum [ft]
    face_nodes: onp.ndarray       # (n_conn, 4) mesh nodes of the connected face
    gamma_w: onp.ndarray          # (n_aq,) aquifer water gradient at p0 [psi/ft]
    p_range: tuple                # (lo, hi) of the head [psia]
    s_p: float                    # pressure scale of the rows [psi]


def build_aquifer_pack(cfg: RunConfig, case: CaseData) -> AquiferPack | None:
    r"""
    The case's aquifers and their cell-face connections, or ``None`` when the deck has none.

    A connection record names a box of cells and a face; each active cell of the box whose
    face lies on the reservoir boundary — no active neighbour across it, unless the record
    allows internal faces — is connected, with the weight :math:`\alpha_c` of its face area
    (or the stated influx coefficient) times the multiplier, normalized over the aquifer. An
    initial pressure the deck leaves to equilibration is the reference state's water pressure
    at the connected cells carried to the datum, each along its own water gradient,

    .. math::

        p_{a,0} \;=\; \sum_c \alpha_c\,\Bigl(p_{w,c}(0) - g\,\rho_w\bigl(p_{w,c}(0)\bigr)\,(z_c - z_d)\Bigr),
        \qquad
        \rho_w(p) \;=\; \frac{\rho_{w,\mathrm{sc}}}{B_w(p)},

    where :math:`g` is the gravity conversion (psi/ft per lb/ft³), :math:`\rho_{w,\mathrm{sc}}`
    the stock-tank water density and :math:`B_w` the water formation volume factor.
    """
    from modules.utils.blackoil_closures import cap_pres_ow, fvf_water
    from modules.utils.hex_fem_assembly_jax import canonical_hexes_from_corner_cells

    aqs = deck_aquifers(cfg)
    if not aqs:
        return None
    rm = case.art.reservoir_mesh
    hexes = canonical_hexes_from_corner_cells(case.verts, case.hexes, onp.asarray(rm.corner_cells))
    verts = onp.asarray(case.verts, onp.float64)
    cent = onp.asarray(case.centroids, onp.float64)
    lookup = rm.active_cell_lookup
    g_rho = float(case.grav_grad) * float(case.dens_w)       # psi/ft per 1/B_w

    cells, aq_of, alpha, faces = [], [], [], []
    for a, aq in enumerate(aqs):
        rows = []
        for (i1, i2, j1, j2, k1, k2, face, coeff, mult, internal) in aq["connections"]:
            step = _FACE_STEP[face]
            for k in range(k1, k2 + 1):
                for j in range(j1, j2 + 1):
                    for i in range(i1, i2 + 1):
                        c = lookup.get((i - 1, j - 1, k - 1))
                        if c is None:
                            continue
                        nb = (i - 1 + step[0], j - 1 + step[1], k - 1 + step[2])
                        if nb in lookup and not internal:
                            continue                # an internal face: not connected
                        nodes = onp.asarray(hexes[c])[list(FACE_CORNERS[face])]
                        p = verts[nodes]
                        area = 0.5 * float(onp.linalg.norm(onp.cross(p[2] - p[0], p[3] - p[1])))
                        rows.append((c, (area if coeff is None else coeff) * mult, nodes))
        if not rows:
            continue
        w = onp.asarray([r[1] for r in rows], onp.float64)
        w = w / max(w.sum(), 1e-30)
        for (c, _, nodes), wc in zip(rows, w):
            cells.append(c)
            aq_of.append(a)
            alpha.append(wc)
            faces.append(nodes)
    if not cells:
        return None
    cells = onp.asarray(cells, onp.int32)
    aq_of = onp.asarray(aq_of, onp.int32)
    alpha = onp.asarray(alpha, onp.float64)
    datum = onp.asarray([aq["datum"] for aq in aqs], onp.float64)
    dz = cent[cells, 2] - datum[aq_of]
    p_w = (onp.asarray(case.pres, onp.float64)[0, cells]
           - onp.asarray(cap_pres_ow(onp.asarray(case.swat, onp.float32)[0, cells], case.tables),
                         onp.float64))
    b_w = onp.asarray(fvf_water(p_w.astype(onp.float32), case.tables), onp.float64)
    p_init = p_w - g_rho / b_w * dz
    p0 = onp.asarray([aq["p0"] if aq["p0"] is not None
                      else float((alpha * p_init)[aq_of == a].sum()) for a, aq in enumerate(aqs)])
    gamma_w = g_rho / onp.asarray(fvf_water(p0.astype(onp.float32), case.tables), onp.float64)
    p_min, p_max = float(case.phys.P_MIN), float(case.phys.P_MAX)
    span = p_max - p_min
    lo = min(p_min, float(p0.min())) - 0.5 * span
    hi = max(p_max, float(p0.max())) + 0.5 * span
    return AquiferPack(
        n_aq=len(aqs), times=onp.asarray(case.times, onp.float64),
        pi=onp.asarray([aq["pi"] for aq in aqs], onp.float64),
        vw=onp.asarray([aq["vw0"] for aq in aqs], onp.float64),
        ct=onp.asarray([aq["ct"] for aq in aqs], onp.float64),
        p0=p0, datum=datum, conn_cell=cells, conn_aq=aq_of, conn_alpha=alpha, conn_dz=dz,
        face_nodes=onp.stack(faces).astype(onp.int32), gamma_w=gamma_w, p_range=(lo, hi),
        s_p=float(onp.asarray(case.state_scale)[0]))


@dataclass
class AquiferHead:
    r"""
    The trainable aquifer pressures, ``params["aqu"]["pa_raw"]`` of shape ``(T, n_aq)``,

    .. math::

        p_a(t_j) \;=\; P^{a}_{\min} + \bigl(P^{a}_{\max} - P^{a}_{\min}\bigr)\,\sigma(\eta_{j}),

    linearly interpolated between report times and initialized at :math:`p_{a,0}`, where
    :math:`[P^{a}_{\min}, P^{a}_{\max}]` is ``pack.p_range``, the case's pressure anchors
    widened by half their span.
    """

    times: Any
    p_min: float
    p_max: float
    raw0: onp.ndarray

    @classmethod
    def from_pack(cls, apack: AquiferPack) -> "AquiferHead":
        import jax.numpy as jnp

        lo, hi = apack.p_range
        u = onp.clip((apack.p0 - lo) / max(hi - lo, 1e-12), _HEAD_EPS, 1.0 - _HEAD_EPS)
        raw = onp.broadcast_to(onp.log(u / (1.0 - u)), (apack.times.size, apack.n_aq))
        return cls(times=jnp.asarray(apack.times, jnp.float32), p_min=lo, p_max=hi,
                   raw0=onp.ascontiguousarray(raw, onp.float32))

    def params0(self) -> dict:
        import jax.numpy as jnp

        return {"pa_raw": jnp.asarray(self.raw0)}

    def pa(self, aq_params):
        """(T, n_aq) aquifer pressures [psia] at the report times."""
        import jax

        return self.p_min + (self.p_max - self.p_min) * jax.nn.sigmoid(aq_params["pa_raw"])

    def pa_at(self, aq_params, t):
        """(n_aq,) aquifer pressures [psia] at time ``t`` (linear interpolation)."""
        import jax.numpy as jnp

        from .wells import _interp_rows

        table = self.pa(aq_params)
        return _interp_rows(jnp.asarray(self.times, table.dtype), table, t)


def wrap_model(mb, head: AquiferHead):
    """Extend a :class:`~modules.models.ModelBundle` with ``params["aqu"]`` (composes with the
    inversion and well-head wrappers in any order; the network stays under ``"net"``)."""
    import dataclasses

    params0 = mb.params0
    if isinstance(params0, dict) and "net" in params0:
        params0 = dict(params0)
        params0["aqu"] = head.params0()
        return dataclasses.replace(mb, params0=params0)
    base_apply = mb.apply
    return dataclasses.replace(mb, params0={"net": mb.params0, "aqu": head.params0()},
                               apply=lambda p, feat: base_apply(p["net"], feat))


@dataclass
class AquiferOps:
    """The aquifer's surfaces: nodal loads of the water balance at any time, and the rows."""

    loads_at: Any        # (params, t) -> (n_nodes,) water influx per node [stb/d]
    aqu_arr: Any         # params -> (T * n_aq,) material-balance rows
    influx: Any          # params -> ((T, n_aq) influx rate [stb/d], (T, n_aq) cumulative [stb])


def make_aquifer_ops(apack: AquiferPack, head: AquiferHead, prim, centroids, n_nodes: int,
                     strength_of=None) -> AquiferOps:
    """Build :class:`AquiferOps` (see the module docstring for the rate and the rows).
    ``strength_of(params) -> (J, V_w)`` (each ``(n_aq,)``) replaces the deck's productivity
    index and water volume when they are inverted (:mod:`modules.inversion`)."""
    import jax
    import jax.numpy as jnp
    from jax import vmap
    from jax.ops import segment_sum

    ci = jnp.asarray(apack.conn_cell, jnp.int32)
    enc_args = prim.encoder.gather_args(ci)
    cxyz = centroids[ci]
    aq = jnp.asarray(apack.conn_aq, jnp.int32)
    alpha = jnp.asarray(apack.conn_alpha, jnp.float32)
    pi0 = jnp.asarray(apack.pi, jnp.float32)
    ct = jnp.asarray(apack.ct, jnp.float32)
    vw0 = jnp.asarray(apack.vw, jnp.float32)
    hydro = jnp.asarray(apack.gamma_w[apack.conn_aq] * apack.conn_dz, jnp.float32)
    face_nodes = jnp.asarray(apack.face_nodes, jnp.int32).reshape(-1)
    times = jnp.asarray(apack.times, jnp.float32)
    n_aq = apack.n_aq
    p0 = jnp.asarray(apack.p0, jnp.float32)

    def _strength(params, dtype):
        if strength_of is None:
            return pi0.astype(dtype), vw0.astype(dtype)
        j, v = strength_of(params)
        return j.astype(dtype), v.astype(dtype)
    s_p = float(apack.s_p)

    def _pw_cells(params, t):
        xt = jnp.concatenate([cxyz, jnp.full((cxyz.shape[0], 1), t, cxyz.dtype)], axis=1)
        return vmap(lambda x, *a: prim.full_state_point(params, x, *a))(xt, *enc_args)[:, 0]

    def _rates(params, t, pa):
        j_aq, _ = _strength(params, pa.dtype)
        return (j_aq[aq] * alpha.astype(pa.dtype)) * (pa[aq] + hydro.astype(pa.dtype)
                                                      - _pw_cells(params, t).astype(pa.dtype))

    def loads_at(params, t):
        t32 = jnp.asarray(t, jnp.float32)
        q = _rates(params, t32, head.pa_at(params["aqu"], t32))
        return segment_sum(jnp.repeat(0.25 * q, 4), face_nodes, num_segments=n_nodes)

    def influx(params):
        pa_tab = head.pa(params["aqu"])                                  # (T, n_aq)
        q = jax.lax.map(lambda ts: _rates(params, ts[0], ts[1]), (times, pa_tab))   # (T, n_conn)
        q_aq = jax.vmap(lambda r: segment_sum(r, aq, num_segments=n_aq))(q)          # (T, n_aq)
        dt = jnp.diff(times).astype(q_aq.dtype)[:, None]
        we = jnp.concatenate([jnp.zeros((1, n_aq), q_aq.dtype),
                              jnp.cumsum(0.5 * (q_aq[1:] + q_aq[:-1]) * dt, axis=0)])
        return q_aq, we

    def aqu_arr(params):
        pa_tab = head.pa(params["aqu"])
        _, we = influx(params)
        ctv = ct.astype(pa_tab.dtype) * _strength(params, pa_tab.dtype)[1]
        r0 = (pa_tab[0] - p0.astype(pa_tab.dtype)) / s_p
        rj = (ctv * (p0 - pa_tab[1:]) - we[1:]) / (ctv * s_p)
        return jnp.concatenate([r0, rj.reshape(-1)])

    return AquiferOps(loads_at=loads_at, aqu_arr=aqu_arr, influx=influx)
