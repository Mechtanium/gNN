r"""
Wells: the time-resolved observation pack, the interior forcing it induces,
and the Peaceman bridge from network state to bottom-hole pressures and rates
(Chen, Huan & Ma eq. 8.11 family).

Two well models coexist, selected by :attr:`~pinnlab.config.RunConfig.well_model`.

**closed_form (forcing as data).** The realized per-well, per-phase surface rates
drive the interior component balances as known stimulation,

.. math::

    Q_{i\alpha}(t) \;=\; \sum_{p} \omega_{i}^{(p)}\, \frac{\mathrm{WI}_p}{\sum_{p' \in w(p)} \mathrm{WI}_{p'}}\; q^{\mathrm{obs}}_{\alpha, w(p)}(t),
    \qquad
    \sum_i \omega_i^{(p)} = 1,

where:
- :math:`q^{\mathrm{obs}}_{\alpha,w}(t)`: the realized (summary) surface rate of component :math:`\alpha` at well :math:`w`, linearly interpolated between report times, signed injection-positive; the gas entry is the **total** surface gas (free plus dissolved), i.e. the exact component offtake — the schedule control rate is only the fallback where a channel is unobserved.
- :math:`\mathrm{WI}_p`: the Peaceman index of perforation :math:`p`, allocating the well rate over its perforations (transmissibility-weighted).
- :math:`\omega_i^{(p)}`: the volume-weighted Gaussian nodal partition of unity about perforation :math:`p`'s cell, so :math:`\sum_i Q_{i\alpha} = q^{\mathrm{obs}}_{\alpha}` exactly (conservative loads).

and the bottom-hole pressure is eliminated in closed form (below).

**predicted (forcing from a trainable well pressure).** The flowing pressure of
every well is a trainable head :math:`p_{wf,w}(t_j) = P_{\min} + (P_{\max}-P_{\min})\,\sigma(\eta_{wj})`
(linearly interpolated between report times), the Peaceman closure turns it into
per-perforation rates, and those rates are the interior source:

.. math::

    q_{\alpha,p}(\theta, t)
    \;=\;
    C_F\, \mathrm{WI}_p\, f^{\mathrm{eff}}_{\alpha,p}\,\bigl(p_{wf,w(p)}(t) + \gamma_{\alpha,p}\,\Delta z_p - p_{\alpha,p}\bigr),
    \qquad
    Q_{i\alpha}(\theta, t) \;=\; \sum_p \omega_i^{(p)}\, q_{\alpha,p}(\theta, t)

where:
- :math:`\eta_{wj}`: the raw head parameter of well :math:`w` at report time :math:`t_j` (``params["well"]["pwf_raw"]``), initialized from the observed WBHP where it exists, the BHP control/limit otherwise, and the deck's initial pressure at the perforated cells as the last resort.
- :math:`f^{\mathrm{eff}}_{\alpha,p}`: the phase mobility factor at the perforated cell — :math:`k_{r\alpha}/(\mu_\alpha B_\alpha)` on producing perforations; on injecting ones the total mobility :math:`\lambda_t/B_\alpha` on the **injected phase only**, the other legs closed.
- :math:`\gamma_{\alpha,p}\Delta z_p`: the wellbore hydrostatic correction from the BHP datum to the perforation.
- the same :math:`\omega_i^{(p)}` as above, so :math:`\sum_i Q_{i\alpha}(\theta,t) = \sum_p q_{\alpha,p}(\theta,t)` identically and the predicted source is conservative by construction.

The gas leg carries the dissolved image :math:`R_s q_o` (total surface gas). The
rows that anchor the prediction are

- the ``well`` group (observations, history window only): :math:`(p_{wf} - p^{\mathrm{obs}}_{bh})/s_{bhp}` wherever WBHP is observed, and :math:`(\mathrm{sign}_c\, q_{\alpha(c)}(\theta) - q^{\mathrm{obs}}_c)/s_{q_\alpha}` for every observed rate channel in the well's flow direction (a producer's injection channels and an injector's production channels are the summary's structural zeros, never rows) except the control channel on rate-mode unbinding steps (it equals the control);
- the ``ctrl`` group (schedule controls, whole collocation window): the control-rate row :math:`(q_c(\theta) - q^{\mathrm{ctrl}})/s_{q_c}` on rate-controlled steps, the BHP row :math:`(p_{wf} - p^{\mathrm{ctrl}}_{bh})/s_{bhp}` on BHP-controlled steps, and — on rate-controlled steps that carry a BHP limit and no observed BHP — the control-switch row of :func:`fischer_burmeister` (or the hinge pair). **An observation supersedes a control**: a ``ctrl`` row exists only where the matching observation row is absent.

**Closed-form observation rows (control-mode switching).** The per-perforation
surface-volume inflow of phase :math:`\alpha` is

.. math::

    q_{\alpha,p}
    \;=\;
    C_F\, \mathrm{WI}_p\, f_{\alpha}\!\left(S, p\right)\,
    \bigl(p_{wb,p} - p_{\alpha,p}\bigr),
    \qquad
    p_{wb,p} = p_{bh} + \gamma_\alpha\,(z_{p} - z_{bh})

with the Peaceman well index recomputed from the cached geometry

.. math::

    \mathrm{WI}_p
    \;=\;
    \frac{2\pi\, h_s \sqrt{\det K_\perp}}{\ln(r_e / r_w) + s}

where:
- :math:`C_F`: the FIELD Darcy constant (mD·ft·psi/cP → rb/day).
- :math:`f_\alpha = k_{r\alpha}/(\mu_\alpha B_\alpha)`: the phase mobility factor at the perforated cell (θ_m-aware through the effective tables); injecting perforations use the total mobility :math:`\lambda_t / B_c` so a dry-gas injector at :math:`S_g = 0` stays well-posed.
- :math:`p_{\alpha,p}`: the phase pressure at the perforation cell; :math:`\gamma_\alpha (z_p - z_{bh})` the wellbore hydrostatic correction with the phase gravity gradient as the column proxy.
- :math:`h_s, K_\perp, r_e, s`: perforated length, transverse permeability block, Peaceman equivalent radius, and skin from the prep-cache metadata (:math:`r_w = 0.1` ft, the prep default).
- Sign convention: **injection-positive** (production channels are negated on ingest).

Each (step, well) is classified by its *realized* control mode, mirroring the
simulator's control switching:

- **rate mode** (the schedule rate is met): :math:`p_{bh}` is eliminated in closed form by pinning the realized control-phase rate,

  .. math::

      \hat p_{bh}
      \;=\;
      \frac{q^{\mathrm{pin}} + \sum_p a_p\,\bigl(p_{c,p} - \gamma_c (z_p - z_{bh})\bigr)}
           {\max\!\left(\sum_p a_p,\ \epsilon\right)},
      \qquad
      a_p = C_F\, \mathrm{WI}_p\, f_{c,p},

  where:
  - :math:`q^{\mathrm{pin}}`: the realized control-phase total rate where observed, the schedule rate as fallback (signed, injection-positive).
  - :math:`c`: the control phase; the rows are the BHP misfit :math:`(\hat p_{bh} - p_{bh}^{\mathrm{obs}})/s_{bhp}` plus the observed **non-control** phase-rate misfits — the pinned channel is identically satisfied and carries no row.

- **BHP-limited mode** (the realized control-phase rate falls short of the schedule — the well sits on its pressure limit): :math:`p_{bh}` is *fixed at the observed BHP* and every observed rate channel becomes a row, including the control-phase decline curve — the most informative series the dataset holds. No closed form is inverted against an unreachable rate, so the elimination cannot rail.

BHP-controlled wells keep :math:`p_{bh}` at the control and supervise all
phase rates. The row set is deterministic and full-time: the gather indices
come from the mode-resolved masks at construction, so the ``well`` and ``ctrl``
groups ignore the sampling window — exactly what the ENGD rows contract needs.
Observation rows are restricted to the case's observation window
(:meth:`~pinnlab.casedata.CaseData.t_obs_span`), so a train/test split never
lets a future observation into the objective.
"""

from __future__ import annotations

import dataclasses
import json
import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as onp

from .config import RunConfig, WellModel, case_label, well_pack_meta_path
from .casedata import CaseData

# Prep-cache Peaceman defaults (prepare_well_metadata signature).
_R_W = 0.1

# Relative shortfall of realized vs schedule control rate that flags the
# BHP-limited mode (simulator control switching, detected from the data).
_BIND_TOL = 0.01

# The five observation rate channels of the packed well results, their canonical
# phase (0=oil, 1=water, 2=gas) and their sign in the injection-positive
# convention.
_CHANNELS = ("wopr", "wwpr", "wgpr", "wwir", "wgir")
_CH_PHASE = (0, 1, 2, 1, 2)
_CH_SIGN = (-1.0, -1.0, -1.0, 1.0, 1.0)
_PHASE_NAMES = ("oil", "water", "gas")

# canonical well phase (0=oil, 1=water, 2=gas) -> residual channel (w, o, g)
_CANON_TO_RES = (1, 0, 2)

# Smoothing of the Fischer-Burmeister control-switch row.
FB_EPS = 1e-3
# Logit clamp for the p_wf head initialization (fraction of the anchor span).
_HEAD_EPS = 1e-3
# Tolerance (fraction of the anchor span) for the rail-fraction diagnostic.
_RAIL_TOL = 1e-3


@dataclass
class WellPack:
    """Time-stacked well observations, resolved control modes, and static Peaceman geometry."""

    n_perf: int
    n_wells: int
    n_times: int
    well_names: list
    times: onp.ndarray            # (T,) days
    # static per-perforation geometry
    cell_idx: onp.ndarray         # (n_perf,) int32
    well_id: onp.ndarray          # (n_perf,) int32
    z_bh: onp.ndarray             # (n_perf,)
    z_cell: onp.ndarray           # (n_perf,)
    wi: onp.ndarray               # (n_perf,) well index [mD ft]: deck CF / C_F, else Peaceman
    # per-step controls (T, n_wells)
    is_rate: onp.ndarray          # 1.0 where the schedule declares rate control
    q_ctrl: onp.ndarray           # signed schedule control-phase total rate
    ctrl_phase: onp.ndarray       # int 0/1/2 canonical control phase
    p_bh_ctrl: onp.ndarray        # BHP control target, or the limit of a rate control (0 where none)
    active: onp.ndarray           # 1.0 where the well has any control this step
    # observations (T, n_wells [, 5]) — RAW observedness, never row-masked
    bhp_obs: onp.ndarray
    bhp_mask: onp.ndarray
    ch_obs: onp.ndarray           # raw summary channels (wopr, wwpr, wgpr, wwir, wgir)
    ch_mask: onp.ndarray
    # resolved control modes and residual-row selections
    q_pin: onp.ndarray = None     # (T, n_wells) realized control-phase rate (schedule fallback)
    binding: onp.ndarray = None   # (T, n_wells) 1.0 where the BHP limit binds (rate shortfall)
    bhp_row_mask: onp.ndarray = None
    ch_row_mask: onp.ndarray = None
    # windows (T,): observation rows live on t_obs_mask, control rows on t_ctrl_mask
    t_obs_mask: onp.ndarray = None
    t_ctrl_mask: onp.ndarray = None
    # predicted-well bookkeeping (zeros / None under closed_form)
    well_model: str = "closed_form"
    ctrl_switch: str = "fb"
    is_inj: onp.ndarray = None          # (T, n_wells) 1.0 where the well injects
    inj_phase: onp.ndarray = None       # (T, n_wells) canonical phase an injector injects
    ctrl_rate_mask: onp.ndarray = None  # (T, n_wells) control-rate rows
    ctrl_bhp_mask: onp.ndarray = None   # (T, n_wells) BHP-control rows
    ctrl_limit_mask: onp.ndarray = None # (T, n_wells) control-switch rows (limit-carrying rate control)
    pwf0: onp.ndarray = None            # (T, n_wells) head initialization [psia]
    pwf_range: tuple = (0.0, 1.0)       # (lo, hi) sigmoid range of the head [psia]
    # residual-row bookkeeping
    n_rows: int = 0                     # observation (well group) rows; kept for the sidecar key
    n_rows_well: int = 0
    n_rows_ctrl: int = 0
    well_scale: tuple = (1.0, 1.0, 1.0, 1.0)   # (s_bhp, s_qo, s_qw, s_qg)
    synthesized: bool = False
    synth_reason: str = ""


def _repack_steps(case: CaseData) -> dict:
    """Run the upstream fixed-order packer over the cached per-step well metadata."""
    from modules.utils.ReservoirMesh import pack_sequence_well_steps

    dummy_eigs = onp.zeros((case.n_nodes, 1), dtype=float)
    return pack_sequence_well_steps(case.art.well_metadata, dummy_eigs, case.art.reservoir_mesh)


def recompute_wi(h_s, k_perp, r_e, skin, r_w=_R_W):
    r"""
    Peaceman well index [mD ft] from the cached transverse geometry,

    .. math::

        \mathrm{WI} \;=\; \frac{2\pi\, h_s \sqrt{\det K_\perp}}{\ln(r_e / r_w) + s}

    where:
    - :math:`h_s`: the perforated length along the well; :math:`K_\perp` the transverse permeability block; :math:`r_e` the Peaceman equivalent radius of the cell; :math:`s` the skin.
    - :math:`r_w`: the wellbore radius — a scalar or a per-perforation array (the deck's ``COMPDAT`` diameter, halved).
    """
    det = onp.linalg.det(onp.asarray(k_perp, onp.float64))
    r_w = onp.asarray(r_w, onp.float64)
    return 2.0 * math.pi * onp.asarray(h_s, onp.float64) * onp.sqrt(onp.maximum(det, 0.0)) / (
        onp.log(onp.asarray(r_e, onp.float64) / r_w) + onp.asarray(skin, onp.float64))


def well_index(seq: dict, c_f: float) -> onp.ndarray:
    r"""
    The per-perforation well index the closure uses [mD ft], with the simulator's precedence:
    an explicit ``COMPDAT`` connection transmissibility factor wins, the Peaceman geometry
    is the fallback,

    .. math::

        \mathrm{WI}_p \;=\;
        \begin{cases}
            \mathrm{CF}_p / C_F & \text{if the deck gives } \mathrm{CF}_p \text{ (item 8)} \\
            2\pi h_s \sqrt{\det K_\perp} / (\ln(r_e/r_w) + s) & \text{otherwise}
        \end{cases}

    where:
    - :math:`\mathrm{CF}_p`: the connection factor in the deck's units (FIELD: cP·rb/day/psi), i.e. :math:`C_F\,\mathrm{WI}` — the deck states it whenever the geometric formula does not apply (SPE2's parent radial grid gives :math:`r_e < r_w` on the graded Cartesian version).
    - :math:`C_F`: the FIELD Darcy coefficient (``case.c_f``).
    - :math:`r_w`: the deck's wellbore radius per perforation (``perf_r_w``), the packer default where a cache predates it.

    Ignoring an explicit factor is not a small error: on SPE2EQUI the geometric value was
    5.6x too small, so the closed-form BHP needed a 5.6x larger drawdown than the
    simulator's and the ``well`` rows could not be satisfied by the true field.
    """
    n_perf = int(seq["n_perf"])
    r_w = onp.asarray(seq.get("perf_r_w", onp.full((n_perf,), onp.nan)), onp.float64)
    r_w = onp.where(onp.isfinite(r_w) & (r_w > 0.0), r_w, _R_W)
    wi_geo = recompute_wi(seq["perf_h_s"], seq["perf_k_perp"], seq["perf_r_e_base"],
                          seq["perf_skin_base"], r_w=r_w)
    cf = onp.asarray(seq.get("perf_cf", onp.full((n_perf,), onp.nan)), onp.float64)
    explicit = onp.isfinite(cf) & (cf > 0.0)
    return onp.where(explicit, cf / float(c_f), wi_geo)


def _robust_scale(values: onp.ndarray, floor: float = 1.0) -> float:
    """Spread-or-magnitude scale of a masked observation series."""
    v = values[onp.isfinite(values)]
    if v.size == 0:
        return floor
    return float(max(onp.std(v), 0.05 * onp.mean(onp.abs(v)), floor))


def _flow_direction(is_rate: onp.ndarray, q_ctrl: onp.ndarray, ch_obs: onp.ndarray) -> onp.ndarray:
    r"""
    Per-step flow direction of every well: :math:`+1` injector, :math:`-1` producer, :math:`0` unknown.

    Rate-controlled steps take the sign of the control rate; BHP-controlled steps take
    the direction of the channels that actually flow at that step (a non-zero
    injection channel and no non-zero production channel, or vice versa).

    where:
    - ``is_rate``, ``q_ctrl``: ``(T, n_wells)`` rate-control flag and signed control rate (injection-positive).
    - ``ch_obs``: ``(T, n_wells, 5)`` raw summary channels (``wopr, wwpr, wgpr, wwir, wgir``), NaN where unobserved.
    """
    d = onp.where(is_rate > 0, onp.sign(q_ctrl), 0.0)
    flow = onp.nan_to_num(ch_obs) != 0.0
    prod = flow[:, :, :3].any(axis=2)
    inj = flow[:, :, 3:].any(axis=2)
    hist = onp.where(inj & ~prod, 1.0, onp.where(prod & ~inj, -1.0, 0.0))
    return onp.where(d == 0.0, hist, d)


def _mask_counter_direction(ch_mask: onp.ndarray, direction: onp.ndarray) -> onp.ndarray:
    r"""
    Drop the summary channels that point against a well's flow direction.

    OPM writes every well vector for every well, so a producer carries
    ``WWIR = WGIR = 0`` and an injector ``WOPR = WWPR = WGPR = 0`` at every report
    step. Those zeros are the *absence* of the counter-direction flow, not an
    observation of the well's own rate; kept as rows they would pair each real
    channel with a contradictory zero target on the same predicted phase rate
    (a producer's gas phase would be pulled to :math:`-\mathrm{WGPR}/2`). A
    channel is therefore observed only where the well flows in its direction —
    the same rule :func:`synthesize_observations` applies to the closure's rates.

    where:
    - ``ch_mask``: ``(T, n_wells, 5)`` raw observedness of the channels.
    - ``direction``: ``(T, n_wells)`` from :func:`_flow_direction`; an unknown direction (``0``) keeps every channel.
    """
    sign = onp.asarray(_CH_SIGN)[None, None, :]
    keep = (direction[:, :, None] == 0.0) | (direction[:, :, None] == sign)
    return ch_mask * keep.astype(onp.float64)


def _window_masks(case: CaseData, times: onp.ndarray) -> tuple[onp.ndarray, onp.ndarray]:
    """(t_obs_mask, t_ctrl_mask): report steps inside the observation / collocation windows."""
    lo, hi = case.t_span()
    _, obs_hi = case.t_obs_span()
    eps = max(1e-6, 1e-6 * max(abs(hi), 1.0))
    t = onp.asarray(times, onp.float64)
    t_ctrl = ((t >= lo - eps) & (t <= hi + eps)).astype(onp.float64)
    t_obs = ((t >= lo - eps) & (t <= obs_hi + eps)).astype(onp.float64)
    return t_obs, t_ctrl


def build_well_pack(cfg: RunConfig, case: CaseData) -> WellPack:
    """
    Assemble the time-major well pack for one case and write the
    ``well_pack_meta.json`` row-count sidecar (consumed by the ENGD memory plan).

    ``cfg.well_source == "summary"`` uses the OPM summary observations exported in
    the prep cache — keeping each rate channel only in the well's own flow
    direction (:func:`_mask_counter_direction`), since the summary reports the
    counter-direction channels as zeros for every well — and falls back to
    Peaceman synthesis (with a RuntimeWarning) when the active control branch
    carries no usable rows;
    ``"synthetic"`` forces the synthesis path. Imposed ``cfg.inference.schedules``
    override the deck controls from ``case.t_sched`` onward (a split half or the
    joint window sets it; an unsplit case never applies them).
    """
    seq = _repack_steps(case)
    n_perf, n_wells = int(seq["n_perf"]), int(seq["n_wells"])
    steps = seq["steps"]
    T = len(steps)
    if n_perf == 0 or T == 0:
        raise ValueError(f"{case_label(cfg)}: no packed well perforations in the prep cache")
    if T != case.n_times:
        raise ValueError(f"well metadata has {T} steps but the case has {case.n_times}")

    wi = well_index(seq, case.c_f)
    well_id = onp.asarray(seq["perf_well_id"], onp.int32)

    is_rate = onp.zeros((T, n_wells))
    q_ctrl = onp.zeros((T, n_wells))
    ctrl_phase = onp.zeros((T, n_wells), onp.int32)
    p_bh_ctrl = onp.zeros((T, n_wells))
    active = onp.zeros((T, n_wells))
    bhp_obs = onp.full((T, n_wells), onp.nan)
    ch_obs = onp.full((T, n_wells, 5), onp.nan)

    for t, st in enumerate(steps):
        rate_mask = onp.asarray(st["perf_rate_prior_mask"])          # (n_perf, 3): [else, WAT, GAS]
        rate_val = onp.asarray(st["perf_control_rate"])              # (n_perf,)
        pbh_mask = onp.asarray(st["perf_control_pbh_mask"])
        pbh_val = onp.asarray(st["perf_control_pbh"])
        obs = onp.asarray(st["well_phase_obs"])                      # (n_wells, 6)
        obs_m = onp.asarray(st["well_phase_obs_mask"])
        for w in range(n_wells):
            perfs = onp.nonzero(well_id == w)[0]
            if perfs.size == 0:
                continue
            if rate_mask[perfs].sum() > 0:
                is_rate[t, w] = 1.0
                active[t, w] = 1.0
                q_ctrl[t, w] = float(rate_val[perfs].sum())
                # packer columns are [other/oil, water, gas] -> canonical (0, 1, 2)
                col = int(onp.argmax(rate_mask[perfs].sum(axis=0)))
                ctrl_phase[t, w] = (0, 1, 2)[col]
            elif pbh_mask[perfs].sum() > 0:
                active[t, w] = 1.0
                p_bh_ctrl[t, w] = float(pbh_val[perfs][pbh_mask[perfs] > 0].mean())
            bhp_obs[t, w] = obs[w, 5] if obs_m[w, 5] > 0 else onp.nan
            for c in range(5):
                ch_obs[t, w, c] = obs[w, c] if obs_m[w, c] > 0 else onp.nan

    # a channel is an observation of the well only in the well's own flow direction
    ch_mask = _mask_counter_direction(onp.isfinite(ch_obs).astype(onp.float64),
                                      _flow_direction(is_rate, q_ctrl, ch_obs))
    t_obs, t_ctrl = _window_masks(case, case.times)
    pack = WellPack(
        n_perf=n_perf, n_wells=n_wells, n_times=T, well_names=list(seq["well_names"]),
        times=onp.asarray(case.times, onp.float64),
        cell_idx=onp.asarray(seq["perf_cell_idx"], onp.int32), well_id=well_id,
        z_bh=onp.asarray(seq["perf_z_bh"]), z_cell=onp.asarray(seq["perf_z_cell"]),
        wi=onp.asarray(wi),
        is_rate=is_rate, q_ctrl=q_ctrl, ctrl_phase=ctrl_phase, p_bh_ctrl=p_bh_ctrl,
        active=active,
        bhp_obs=bhp_obs, bhp_mask=onp.isfinite(bhp_obs).astype(onp.float64) * active,
        ch_obs=onp.nan_to_num(ch_obs), ch_mask=ch_mask,
        t_obs_mask=t_obs, t_ctrl_mask=t_ctrl,
        well_model=str(cfg.well_model.value), ctrl_switch=str(cfg.ctrl_switch),
    )
    _resolve_control_modes(pack)

    reason = ""
    if cfg.well_source == "synthetic":
        reason = "well_source='synthetic' requested"
    elif int(pack.bhp_mask.sum() + pack.ch_mask.sum()) == 0:
        reason = "the summary export carries no usable observation rows"
    if reason:
        synthesize_observations(case, pack, cfg)
        pack.synthesized = True
        pack.synth_reason = reason
        warnings.warn(
            "well observations synthesized from reference cell states via the Peaceman "
            f"closure — inverse-crime bridge, not field data: {reason}",
            RuntimeWarning, stacklevel=2)

    # Imposed schedules must land BEFORE build_forcing and make_residuals: the interior
    # source Q(t) is captured by closure at build time, so a later patch would change the
    # reported controls while the PDE kept solving against the deck's own rates.
    t_sched = float(getattr(case, "t_sched", -1.0))
    if cfg.inference.schedules and t_sched >= 0.0:
        apply_schedules(pack, cfg.inference.schedules, t_from=t_sched, case=case)

    _finalize_rows_and_scales(pack, case)
    meta = {"case": case_label(cfg), "n_perf": pack.n_perf, "n_wells": pack.n_wells,
            "n_times": pack.n_times, "n_rows": pack.n_rows,
            "n_rows_well": pack.n_rows_well, "n_rows_ctrl": pack.n_rows_ctrl,
            "well_model": pack.well_model,
            "synthesized": bool(pack.synthesized)}
    path = well_pack_meta_path(cfg)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=1))
    except OSError as e:  # noqa: PERF203 - sidecar is best-effort
        print(f"[wells] sidecar not written ({e})")
    return pack


def _resolve_control_modes(pack: WellPack) -> None:
    r"""
    Resolve the realized control mode of every (step, well) from the data
    (in place): the pinned rate :math:`q^{\mathrm{pin}}`, the BHP-limited flag,
    and the residual-row selections.

    A rate-scheduled well is **BHP-limited** at a step when its realized
    control-phase rate falls short of the schedule,

    .. math::

        \bigl|q^{\mathrm{obs}}_{c}\bigr| \;<\; (1 - \varepsilon_b)\,\bigl|Q_{\mathrm{ctrl}}\bigr|,

    where:
    - :math:`\varepsilon_b`: the shortfall tolerance (:data:`_BIND_TOL`); the flag additionally requires an observed BHP to pin :math:`p_{bh}` to.

    Row selections: under ``closed_form`` the BHP row exists only where the closed
    form *predicts* :math:`p_{bh}` (rate mode); under ``predicted`` every observed
    BHP is a row (nothing is eliminated). A rate channel row exists wherever it is
    observed except the control channel in rate mode, which the closed form
    satisfies identically (a tautology carrying no gradient) and which, under
    ``predicted``, the ``ctrl`` group already carries.

    Gas-controlled wells keep the **schedule** pin: their summary channel
    reports *total* surface gas while the closed form pins the *free*
    control-phase rate — the dissolved image would bias the elimination; the
    shortfall detection still reads the observed channel (the dissolved image
    only raises it, so a detected shortfall is genuine), and BHP-limited steps
    bypass the closed form entirely.
    """
    q_pin = pack.q_ctrl.copy()
    binding = onp.zeros_like(pack.is_rate)
    bhp_ok = onp.isfinite(pack.bhp_obs)
    for c, (ph, sign) in enumerate(zip(_CH_PHASE, _CH_SIGN)):
        is_ctrl_ch = ((pack.ctrl_phase == ph) & (pack.is_rate > 0)
                      & ((pack.q_ctrl > 0) == (sign > 0)))
        obs_ok = pack.ch_mask[:, :, c] > 0
        sel = is_ctrl_ch & obs_ok
        q_real = sign * pack.ch_obs[:, :, c]
        if ph != 2:
            q_pin = onp.where(sel, q_real, q_pin)
        binding = onp.maximum(
            binding,
            (sel & bhp_ok
             & (onp.abs(q_real) < (1.0 - _BIND_TOL) * onp.abs(pack.q_ctrl))).astype(onp.float64))
    pack.q_pin = q_pin
    pack.binding = binding

    if pack.well_model == WellModel.PREDICTED.value:
        pack.bhp_row_mask = pack.bhp_mask * pack.active
    else:
        pack.bhp_row_mask = pack.bhp_mask * pack.is_rate * pack.active * (1.0 - binding)
    ch_row = pack.ch_mask.copy()
    for c, (ph, sign) in enumerate(zip(_CH_PHASE, _CH_SIGN)):
        is_ctrl_ch = ((pack.ctrl_phase == ph) & (pack.is_rate > 0)
                      & ((pack.q_ctrl > 0) == (sign > 0)))
        pinned = is_ctrl_ch & (binding < 0.5)
        ch_row[:, :, c] *= onp.where(pinned, 0.0, 1.0)
    pack.ch_row_mask = ch_row

    # injector flag: a rate control with positive rate, or a BHP-controlled step of a
    # well whose history reports any injection channel
    inj_hist = ((pack.q_ctrl > 0).any(axis=0)
                | (pack.ch_mask[:, :, 3] > 0).any(axis=0)
                | (pack.ch_mask[:, :, 4] > 0).any(axis=0))              # (n_wells,)
    pack.is_inj = onp.where(pack.is_rate > 0, (pack.q_ctrl > 0).astype(onp.float64),
                            onp.broadcast_to(inj_hist.astype(onp.float64), pack.q_ctrl.shape))
    # the phase an injector injects: its control phase under rate control, else the
    # channel its history reports (water before gas; gas as the last resort)
    hist_ph = onp.where((pack.ch_mask[:, :, 3] > 0).any(axis=0), 1,
                        onp.where((pack.ch_mask[:, :, 4] > 0).any(axis=0), 2,
                                  onp.where((pack.q_ctrl > 0).any(axis=0),
                                            pack.ctrl_phase[onp.argmax(pack.q_ctrl > 0, axis=0),
                                                            onp.arange(pack.n_wells)], 2)))
    pack.inj_phase = onp.where(pack.is_rate > 0, pack.ctrl_phase,
                               onp.broadcast_to(hist_ph, pack.ctrl_phase.shape)).astype(onp.int32)


def _finalize_rows_and_scales(pack: WellPack, case: CaseData | None = None) -> None:
    r"""
    Apply the observation/collocation windows, resolve the ``ctrl`` row masks,
    initialize the :math:`p_{wf}` head, count residual rows and set the
    per-channel normalization scales.

    Observation rows are confined to ``t_obs_mask``; under ``predicted`` the
    control rows follow the *observation-supersedes-control* rule:

    - control-rate rows on ``is_rate · active · (1 − binding)`` steps inside the collocation window whose control channel carries no observation row;
    - BHP-control rows on BHP-controlled active steps with no observed-BHP row;
    - control-switch rows on rate-controlled steps that carry a BHP limit and no observed-BHP row (the simulator's rate/limit switch, unresolved by data). Under ``ctrl_switch="fb"`` the switch row **replaces** the plain rate row on those steps; under ``"hinge"`` it is added beside it.

    The head initialization :math:`p^0_{wj}` is the observed WBHP where observed, the
    BHP control/limit where positive, the well's last observed WBHP otherwise, and the
    deck's initial oil pressure at the perforated cells as the final fallback.
    """
    if pack.t_obs_mask is None:
        pack.t_obs_mask = onp.ones(pack.n_times)
    if pack.t_ctrl_mask is None:
        pack.t_ctrl_mask = onp.ones(pack.n_times)
    t_obs = pack.t_obs_mask[:, None]
    t_ctrl = pack.t_ctrl_mask[:, None]
    pack.bhp_row_mask = pack.bhp_row_mask * t_obs
    pack.ch_row_mask = pack.ch_row_mask * t_obs[:, :, None]
    # beyond the observation window the realized control mode is unknowable: the
    # BHP-limited flag and the realized-rate pin fall back to the schedule there,
    # so neither head can read the simulator's future switching off the data
    pack.binding = pack.binding * t_obs
    pack.q_pin = onp.where(t_obs > 0, pack.q_pin, pack.q_ctrl)

    T, W = pack.n_times, pack.n_wells
    zeros = onp.zeros((T, W))
    pack.ctrl_rate_mask, pack.ctrl_bhp_mask, pack.ctrl_limit_mask = zeros, zeros.copy(), zeros.copy()
    if pack.well_model == WellModel.PREDICTED.value:
        has_bhp_obs = pack.bhp_row_mask > 0
        ctrl_ch_obs = onp.zeros((T, W), bool)
        for c, (ph, sign) in enumerate(zip(_CH_PHASE, _CH_SIGN)):
            is_ctrl_ch = ((pack.ctrl_phase == ph) & (pack.is_rate > 0)
                          & ((pack.q_ctrl > 0) == (sign > 0)))
            ctrl_ch_obs |= is_ctrl_ch & (pack.ch_row_mask[:, :, c] > 0)
        rate_steps = (pack.is_rate > 0) & (pack.active > 0) & (t_ctrl > 0)
        limit = rate_steps & (pack.p_bh_ctrl > 0) & ~has_bhp_obs
        rate = rate_steps & (pack.binding < 0.5) & ~ctrl_ch_obs
        if pack.ctrl_switch == "fb":
            rate = rate & ~limit
        bhp = (pack.is_rate <= 0) & (pack.active > 0) & (pack.p_bh_ctrl > 0) & (t_ctrl > 0) & ~has_bhp_obs
        pack.ctrl_rate_mask = rate.astype(onp.float64)
        pack.ctrl_bhp_mask = bhp.astype(onp.float64)
        pack.ctrl_limit_mask = limit.astype(onp.float64)

        # head initialization
        pwf0 = onp.where(pack.bhp_mask > 0, pack.bhp_obs, onp.nan)
        pwf0 = onp.where(onp.isnan(pwf0) & (pack.p_bh_ctrl > 0), pack.p_bh_ctrl, pwf0)
        for w in range(W):
            col = pwf0[:, w]
            obs_idx = onp.flatnonzero(onp.isfinite(col))
            if obs_idx.size:
                last = col[obs_idx[-1]]
                # forward-fill from the last observation, back-fill before the first
                for j in range(T):
                    if onp.isnan(col[j]):
                        prev = obs_idx[obs_idx <= j]
                        col[j] = col[prev[-1]] if prev.size else col[obs_idx[0]]
                col[onp.isnan(col)] = last
            elif case is not None:
                cells = pack.cell_idx[pack.well_id == w]
                col[:] = float(onp.asarray(case.pres)[0, cells].mean()) if cells.size else onp.nan
            pwf0[:, w] = col
        pack.pwf0 = onp.nan_to_num(pwf0, nan=float(onp.nanmean(pwf0)) if onp.isfinite(pwf0).any() else 0.0)
        if case is not None:
            pack.pwf_range = _head_range(pack, case)

    n_bhp = int(pack.bhp_row_mask.sum())
    n_rate = int(pack.ch_row_mask.sum())
    pack.n_rows_well = n_bhp + n_rate
    pack.n_rows = pack.n_rows_well
    pack.n_rows_ctrl = int(pack.ctrl_rate_mask.sum() + pack.ctrl_bhp_mask.sum()
                           + pack.ctrl_limit_mask.sum())

    bhp = onp.where(pack.bhp_mask > 0, pack.bhp_obs, onp.nan)
    s_bhp = _robust_scale(bhp, floor=1.0)
    phase_scales = []
    for ph in range(3):
        vals = []
        for c in range(5):
            if _CH_PHASE[c] == ph:
                vals.append(onp.where(pack.ch_mask[:, :, c] > 0, pack.ch_obs[:, :, c], onp.nan).ravel())
        v = onp.concatenate(vals)
        # fall back to the control-rate magnitude when the channel is unobserved
        ctrl = onp.abs(pack.q_ctrl[(pack.ctrl_phase == ph) & (pack.is_rate > 0)])
        floor = max(0.05 * float(ctrl.mean()) if ctrl.size else 0.0, 1.0)
        phase_scales.append(_robust_scale(v, floor=floor))
    # a phase with no flow anywhere (SPE1 water) keeps a scale within 3 decades of
    # the busiest phase, so its zero-rate rows cannot dominate the group
    lo = 1e-3 * max(phase_scales)
    phase_scales = [max(s, lo) for s in phase_scales]
    pack.well_scale = (s_bhp, phase_scales[0], phase_scales[1], phase_scales[2])


# ---------------------------------------------------------------------------------------------
# The p_wf head (predicted well model)
# ---------------------------------------------------------------------------------------------

def _head_range(pack: WellPack, case: CaseData, margin: float = 0.1) -> tuple[float, float]:
    r"""
    The sigmoid range :math:`[P^{wf}_{\min}, P^{wf}_{\max}]` of the head: the case's
    pressure anchors widened to enclose every observed and controlled BHP with a
    margin of ``margin`` times the anchor span. A producer on its limit sits well
    below the reservoir-pressure anchors (SPE1: 1000 psia against
    :math:`P_{\min} = 2606` psia), so the anchors alone would clip its head.
    """
    p_min, p_max = float(case.phys.P_MIN), float(case.phys.P_MAX)
    span = p_max - p_min
    vals = [onp.where(pack.bhp_mask > 0, pack.bhp_obs, onp.nan).ravel(),
            onp.where(pack.p_bh_ctrl > 0, pack.p_bh_ctrl, onp.nan).ravel()]
    if pack.pwf0 is not None:                 # the predicted head's initialization (absent on closed_form)
        vals.append(onp.asarray(pack.pwf0, onp.float64).ravel())
    v = onp.concatenate([onp.asarray(x, onp.float64) for x in vals])
    v = v[onp.isfinite(v)]
    lo = min(p_min, float(v.min())) if v.size else p_min
    hi = max(p_max, float(v.max())) if v.size else p_max
    return lo - margin * span, hi + margin * span

def _interp_rows(times, table, t):
    """Linear interpolation of a time-major table ``(T, ...)`` at scalar ``t`` (clipped)."""
    import jax.numpy as jnp

    ts = times
    t = jnp.clip(jnp.asarray(t, ts.dtype), ts[0], ts[-1])
    i = jnp.clip(jnp.searchsorted(ts, t, side="right") - 1, 0, ts.shape[0] - 2)
    w = (t - ts[i]) / jnp.maximum(ts[i + 1] - ts[i], 1e-12)
    return (1.0 - w) * table[i] + w * table[i + 1]


def _nearest_row(times, table, t):
    """The report-step row whose interval contains ``t`` (piecewise-constant flags)."""
    import jax.numpy as jnp

    ts = times
    t = jnp.clip(jnp.asarray(t, ts.dtype), ts[0], ts[-1])
    i = jnp.clip(jnp.searchsorted(ts, t, side="right") - 1, 0, ts.shape[0] - 1)
    return table[i]


@dataclass
class WellHead:
    r"""
    The trainable flowing-pressure head of the ``predicted`` well model.

    .. math::

        p_{wf,w}(t_j) \;=\; P_{\min} + (P_{\max} - P_{\min})\,\sigma(\eta_{wj}),
        \qquad
        p_{wf,w}(t) \;=\; (1 - \varsigma)\,p_{wf,w}(t_j) + \varsigma\,p_{wf,w}(t_{j+1}),
        \quad \varsigma = \frac{t - t_j}{t_{j+1} - t_j}

    where:
    - :math:`\eta_{wj}`: the raw parameters, ``params["well"]["pwf_raw"]`` of shape ``(T, n_wells)``.
    - :math:`P_{\min}, P_{\max}`: here the head's own range ``pack.pwf_range`` — the case's pressure anchors widened to enclose every observed or controlled BHP (:func:`_head_range`) — so :math:`p_{wf}` can never leave a physical range whatever :math:`\eta` does (the same hard-constraint device as the network's primaries).
    - :math:`\varsigma`: the linear interpolation weight between the report times bracketing :math:`t`, so the head is continuous in time wherever the residual samples it.
    """

    times: Any                    # (T,) f32
    p_min: float
    p_max: float
    raw0: onp.ndarray             # (T, n_wells) initial raw parameters

    @classmethod
    def from_pack(cls, pack: WellPack, case: CaseData) -> "WellHead":
        import jax.numpy as jnp

        p_min, p_max = pack.pwf_range
        u = (onp.asarray(pack.pwf0, onp.float64) - p_min) / max(p_max - p_min, 1e-12)
        u = onp.clip(u, _HEAD_EPS, 1.0 - _HEAD_EPS)
        raw0 = onp.log(u / (1.0 - u)).astype(onp.float32)
        return cls(times=jnp.asarray(pack.times, jnp.float32), p_min=p_min, p_max=p_max, raw0=raw0)

    def params0(self) -> dict:
        import jax.numpy as jnp

        return {"pwf_raw": jnp.asarray(self.raw0, jnp.float32)}

    def pwf(self, well_params) -> Any:
        """(T, n_wells) flowing pressures [psia] at the report times."""
        import jax

        raw = well_params["pwf_raw"]
        return self.p_min + (self.p_max - self.p_min) * jax.nn.sigmoid(raw)

    def pwf_at(self, well_params, t) -> Any:
        """(n_wells,) flowing pressures [psia] at time ``t`` (linear interpolation)."""
        import jax.numpy as jnp

        table = self.pwf(well_params)
        return _interp_rows(jnp.asarray(self.times, table.dtype), table, t)


def wrap_model_with_head(mb, head: WellHead):
    """
    Extend a :class:`~pinnlab.models.ModelBundle` with the :math:`p_{wf}` head.

    ``params0`` becomes ``{"net": ..., ["phys": ...,] "well": {"pwf_raw": ...}}`` and
    ``apply`` routes features through the ``net`` block; composes with
    :func:`pinnlab.inversion.wrap_model` in either order. ``raw_apply`` /
    ``raw_params0`` stay untouched so the bit-parity gate is unaffected.
    """
    params0 = mb.params0
    if isinstance(params0, dict) and "net" in params0:
        params0 = dict(params0)
        params0["well"] = head.params0()
        return dataclasses.replace(mb, params0=params0)
    base_apply = mb.apply
    params0 = {"net": mb.params0, "well": head.params0()}
    return dataclasses.replace(
        mb, params0=params0, apply=lambda p, feat: base_apply(p["net"], feat))


# ---------------------------------------------------------------------------------------------
# The interior forcing induced by the pack (controls-as-data)
# ---------------------------------------------------------------------------------------------

@dataclass
class WellForcing:
    r"""
    Time-resolved interior well stimulation, in residual channel order
    ``(water, oil, gas)`` and the injection-positive sign convention.

    ``q_perf`` holds the per-perforation component rates
    :math:`q_{\alpha,p}(t_j) = (\mathrm{WI}_p / \sum_{p' \in w} \mathrm{WI}_{p'})\, q^{\mathrm{obs}}_{\alpha, w}(t_j)`
    at the report times; :meth:`rate_perf_at` interpolates them linearly in
    :math:`t`, so the forcing is continuous and differentiable wherever the
    residual samples the time axis. The forcing is **data**: ``params`` is
    accepted (and ignored) so both forcing kinds share one call signature.
    """

    times: Any                  # (T,) f32
    q_perf: Any                 # (T, n_perf, 3) residual order (w, o, g), injection-positive
    perf_xyz: Any               # (n_perf, 3) perforated cell centroids
    perf_sigma: Any             # (n_perf, 3) perforated cell dimensions (mollifier widths)
    predicted: bool = False

    def rate_perf_at(self, t, params=None):
        """Per-perforation component rates at time ``t`` (linear interpolation)."""
        return _interp_rows(self.times, self.q_perf, t)                # (n_perf, 3)

    def nodal_partition(self, node_xyz, vft3):
        r"""
        Conservative nodal partition of unity per perforation:
        :math:`\omega_i^{(p)} \propto V_i\, e^{-\frac{1}{2}\lVert (x_i - x_p)/\sigma_p \rVert^2}`
        normalized over nodes, returned as the ``(n_nodes, n_perf)`` matrix so
        the loads at time :math:`t` are the product with :meth:`rate_perf_at`.
        """
        import jax.numpy as jnp

        d = (node_xyz[:, None, :] - self.perf_xyz[None, :, :]) / self.perf_sigma[None, :, :]
        g = jnp.exp(-0.5 * jnp.sum(d ** 2, axis=-1)) * vft3[:, None]
        return g / jnp.maximum(jnp.sum(g, axis=0, keepdims=True), 1e-30)

    def node_envelope(self, node_xyz, width: float = 1.0):
        r"""
        Peak-normalized Gaussian envelope of the wells over the nodes,
        :math:`g_i = \max_p \exp\!\bigl(-\tfrac12 \lVert (x_i - x_p)/(\varkappa\,\sigma_p)\rVert^2\bigr)`
        — the ``well_gaussian`` residual-row emphasis (an envelope, not a density).
        """
        import jax.numpy as jnp

        d = (node_xyz[:, None, :] - self.perf_xyz[None, :, :]) / (width * self.perf_sigma[None, :, :])
        return jnp.max(jnp.exp(-0.5 * jnp.sum(d ** 2, axis=-1)), axis=1)

    def density_at(self, xyz, t, params=None):
        """Mollified source densities (3,) [surface vol ft^-3 day^-1] (chain_rule residual)."""
        import jax.numpy as jnp

        two_pi_32 = jnp.asarray((2.0 * jnp.pi) ** 1.5, jnp.float32)
        d = (xyz[None, :] - self.perf_xyz) / self.perf_sigma
        g = jnp.exp(-0.5 * jnp.sum(d ** 2, axis=1)) / (two_pi_32 * jnp.prod(self.perf_sigma, axis=1))
        return g @ self.rate_perf_at(t, params)                      # (3,)


@dataclass
class PredictedForcing(WellForcing):
    r"""
    The ``predicted`` well model's interior source: the same Gaussian nodal
    partition and mollified density as :class:`WellForcing`, but the
    per-perforation rates are the Peaceman prediction
    :math:`q_{\alpha,p}(\theta, t)` of the :math:`p_{wf}` head (``rates_at``),
    so the source moves with the parameters. ``q_perf`` is unused.
    """

    rates_at: Callable | None = None      # (params, t) -> (n_perf, 3) residual order (w, o, g)
    predicted: bool = True

    def rate_perf_at(self, t, params=None):
        if params is None:
            raise ValueError("PredictedForcing needs the parameters: rate_perf_at(t, params)")
        return self.rates_at(params, t)


def build_forcing(pack: WellPack, case: CaseData, from_controls: bool = False,
                  controls_from: float | None = None) -> WellForcing:
    r"""Assemble the :class:`WellForcing` series from the pack's realized rates.

    Component rates prefer the observed summary channels (production negated,
    injection positive, opposite-direction channels summed); the schedule
    control rate fills a well's control phase only where its channel is
    unobserved. Rates are allocated over perforations by Peaceman-index
    fraction.

    ``from_controls`` inverts that preference and builds :math:`Q(t)` from the
    **controls alone**. On a forecast window the observed channels are the
    simulator's future output — using them would both leak the answer into the
    interior source and pin the source to the deck's history, so an imposed
    :class:`~pinnlab.config.FlowSchedule` could never change the solution. A
    control, by contrast, is an input the operator sets, so it is legitimately
    known ahead of time. Wells under BHP control contribute no source term in
    this mode: their rate is an *output* of the Peaceman closure, not something
    the schedule fixes.

    ``controls_from`` mixes the two on one window (the joint history-match +
    forecast case under the closed-form model, and the calibration forcing of
    the predicted model): report steps at or before it use the observed rates,
    steps after it the controls alone, so the forecast half of a single forcing
    never reads the simulator's future output.
    """
    import jax.numpy as jnp

    T, n_wells = pack.n_times, pack.n_wells
    times = onp.asarray(pack.times, onp.float64)
    use_ctrl = onp.ones((T,), bool) if from_controls else onp.zeros((T,), bool)
    if controls_from is not None:
        use_ctrl |= times > float(controls_from) + 1e-6

    q_ctrl = onp.zeros((T, n_wells, 3))
    for ph in range(3):
        rp = _CANON_TO_RES[ph]
        sel = (pack.ctrl_phase == ph) & (pack.is_rate > 0) & (pack.active > 0)
        q_ctrl[:, :, rp] += onp.where(sel, pack.q_ctrl, 0.0)
    bhp_only = (pack.active > 0) & (pack.is_rate <= 0) & use_ctrl[:, None]
    if bhp_only.any():
        warnings.warn(
            f"{int(bhp_only.sum())} (well, step) pair(s) on the control-driven window are "
            "BHP-controlled, so they contribute no interior source: under BHP control "
            "the rate is an output of the well closure, not an imposed input. Give "
            "those wells a rate-controlled FlowSchedule record if they should drive "
            "the forecast (or use well_model=predicted, whose source is the closure's "
            "own output).",
            RuntimeWarning, stacklevel=2)

    q_obs = onp.zeros((T, n_wells, 3))
    obs_any = onp.zeros((T, n_wells, 3), bool)
    for c, (ph, sign) in enumerate(zip(_CH_PHASE, _CH_SIGN)):
        rp = _CANON_TO_RES[ph]
        m = pack.ch_mask[:, :, c] > 0
        q_obs[:, :, rp] += onp.where(m, sign * pack.ch_obs[:, :, c], 0.0)
        obs_any[:, :, rp] |= m
    for ph in range(3):
        rp = _CANON_TO_RES[ph]
        fallback = ((pack.ctrl_phase == ph) & (pack.is_rate > 0) & ~obs_any[:, :, rp])
        q_obs[:, :, rp] += onp.where(fallback, pack.q_ctrl, 0.0)

    q = onp.where(use_ctrl[:, None, None], q_ctrl, q_obs)

    wi_sum = onp.zeros(n_wells)
    onp.add.at(wi_sum, pack.well_id, pack.wi)
    wi_frac = pack.wi / onp.maximum(wi_sum[pack.well_id], 1e-30)
    q_perf = q[:, pack.well_id, :] * wi_frac[None, :, None]          # (T, n_perf, 3)

    centroids = onp.asarray(case.centroids)
    cell_len = onp.asarray(case.cell_len)
    return WellForcing(
        times=jnp.asarray(pack.times, jnp.float32),
        q_perf=jnp.asarray(q_perf, jnp.float32),
        perf_xyz=jnp.asarray(centroids[pack.cell_idx], jnp.float32),
        perf_sigma=jnp.asarray(cell_len[pack.cell_idx], jnp.float32),
    )


# ---------------------------------------------------------------------------------------------
# The Peaceman closure: shared perforation state + two heads
# ---------------------------------------------------------------------------------------------

def _perf_state(P4, pack: WellPack, case: CaseData, tables, kr_floor: float,
                wi_mult=None, is_inj=None, inj_phase=None):
    r"""
    Everything the closure needs at the perforations, from the primaries.

    ``P4`` is ``(..., n_perf, 4)`` — :math:`(p_o, S_w, S_g, R_s)` at every perforated
    cell for one or many times. Returns ``(f_eff, gam, p_ph, wi, dz, rs, f_floor)`` in the
    canonical phase order ``(o, w, g)``: the effective mobility factor (total
    mobility on injecting perforations; **unfloored**, so an immobile phase carries no
    flow), the phase gravity gradients [psi/ft], the phase pressures, the (multiplied)
    well indices, the datum offsets, :math:`R_s`, and the mobility increment
    :math:`k_{r,\mathrm{floor}} / (\mu_\alpha B_\alpha)` the closed form adds to the
    control phase of its denominator only. ``is_inj`` is the ``(..., n_perf)`` injector flag and ``inj_phase``
    the ``(..., n_perf)`` injected phase; ``None`` reads the pack's resolved
    report-time arrays. An injecting perforation carries the total mobility on its
    injected phase **only** — the other legs are closed, so a gas injector cannot
    inject oil or water into the balance.
    """
    import jax
    import jax.numpy as jnp

    from modules.utils.blackoil_closures import (cap_pres_go, cap_pres_ow, fvf_gas, fvf_oil, fvf_water,
                                   relperm_gas, relperm_oil_3p, relperm_water,
                                   visc_gas, visc_oil, visc_water)

    dt = P4.dtype
    grav = jnp.asarray(case.grav_grad, dt)
    eps = jnp.asarray(1e-12, dt)

    wi = jnp.asarray(pack.wi, dt)
    if wi_mult is not None:
        wi = wi * wi_mult.astype(dt)
    wid = jnp.asarray(pack.well_id, jnp.int32)
    dz = jnp.asarray(pack.z_cell - pack.z_bh, dt)                    # (n_perf,)

    p_o, s_w, s_g, rs = P4[..., 0], P4[..., 1], P4[..., 2], P4[..., 3]
    p_w = p_o - cap_pres_ow(s_w, tables)
    p_g = p_o + cap_pres_go(s_g, tables)
    b_o, mu_o = fvf_oil(rs, tables), visc_oil(rs, tables)
    b_w, mu_w = fvf_water(p_w, tables), visc_water(p_w, tables)
    b_g, mu_g = fvf_gas(p_g, tables), visc_gas(p_g, tables)
    # The mobilities that flow are UNFLOORED: an immobile phase (water at connate
    # saturation on SPE1's producer) must not be given a phantom rate, since its
    # observed channel is exactly zero and, with the channel's small scale, a few
    # stb/d of floor-induced flow would pin the well group at ~1e-2 forever. The
    # floor is added by closed_form_head to the control-phase mobility of the
    # denominator only, where it keeps the eliminated BHP bounded at initialization.
    kro = relperm_oil_3p(s_w, s_g, tables)
    krw = relperm_water(s_w, tables)
    krg = relperm_gas(s_g, tables)
    mub = jnp.stack([mu_o * b_o, mu_w * b_w, mu_g * b_g], axis=-1)   # (..., n_perf, 3)
    f = jnp.stack([kro, krw, krg], axis=-1) / jnp.maximum(mub, eps)  # (..., n_perf, 3)
    f_floor = jnp.asarray(kr_floor, dt) / jnp.maximum(mub, eps)      # the floor's mobility increment
    lam_t = kro / jnp.maximum(mu_o, eps) + krw / jnp.maximum(mu_w, eps) + krg / jnp.maximum(mu_g, eps)
    b_ph = jnp.stack([b_o, b_w, b_g], axis=-1)
    f_inj = lam_t[..., None] / jnp.maximum(b_ph, eps)                # total mobility / B_alpha

    rho_o = (case.dens_o + rs * case.dens_g) / jnp.maximum(b_o, eps)
    rho_w = case.dens_w / jnp.maximum(b_w, eps)
    rho_g = case.dens_g / jnp.maximum(b_g, eps)
    gam = grav * jnp.stack([rho_o, rho_w, rho_g], axis=-1)           # (..., n_perf, 3) psi/ft
    p_ph = jnp.stack([p_o, p_w, p_g], axis=-1)

    if is_inj is None:
        is_inj = jnp.asarray(pack.is_inj, dt)[..., wid]              # (T, n_perf)
    if inj_phase is None:
        inj_phase = jnp.asarray(pack.inj_phase, jnp.int32)[..., wid]
    onehot = jax.nn.one_hot(jnp.asarray(inj_phase, jnp.int32), 3, dtype=dt)
    f_eff = jnp.where(jnp.asarray(is_inj, dt)[..., None] > 0, f_inj * onehot, f)
    return f_eff, gam, p_ph, wi, dz, rs, f_floor


def closed_form_head(state, pack: WellPack, case: CaseData):
    r"""
    The ``closed_form`` head: eliminate :math:`p_{bh}` against the pinned rate on
    rate-mode steps, fix it at the observed BHP on BHP-limited steps and at the
    control on BHP-controlled wells, then back-substitute every phase rate.
    Returns ``(p_bh_hat, q_hat, p_bh_used)`` with shapes ``(T, n_wells)``,
    ``(T, n_wells, 3)`` and ``(T, n_wells)``, injection-positive; the gas
    component includes the dissolved :math:`R_s q_o` of produced oil.
    """
    import jax
    import jax.numpy as jnp
    from jax.ops import segment_sum

    f_eff, gam, p_ph, wi, dz, rs, f_floor = state
    dt = p_ph.dtype
    n_wells = pack.n_wells
    c_f = jnp.asarray(case.c_f, dt)
    eps = jnp.asarray(1e-12, dt)
    wid = jnp.asarray(pack.well_id, jnp.int32)

    is_rate = jnp.asarray(pack.is_rate, dt)
    binding = jnp.asarray(pack.binding, dt)                          # (T, n_wells)
    use_closed = is_rate * (1.0 - binding)
    ctrl_ph = jnp.asarray(pack.ctrl_phase, jnp.int32)                # (T, n_wells)
    ctrl_ph_perf = ctrl_ph[:, wid]                                   # (T, n_perf)

    # the control-phase mobility of the elimination carries the kr floor (a near-immobile
    # control phase at initialization would otherwise send q_pin / den off to infinity)
    f_c = (jnp.take_along_axis(f_eff, ctrl_ph_perf[..., None], axis=-1)[..., 0]
           + jnp.take_along_axis(f_floor, ctrl_ph_perf[..., None], axis=-1)[..., 0])
    gam_c = jnp.take_along_axis(gam, ctrl_ph_perf[..., None], axis=-1)[..., 0]
    p_c = jnp.take_along_axis(p_ph, ctrl_ph_perf[..., None], axis=-1)[..., 0]

    a_c = c_f * wi[None, :] * f_c                                    # (T, n_perf)
    seg = lambda x: segment_sum(x.T, wid, num_segments=n_wells).T    # (T, n_perf) -> (T, n_wells)
    den = jnp.maximum(seg(a_c), eps)
    num = jnp.asarray(pack.q_pin, dt) + seg(a_c * (p_c - gam_c * dz[None, :]))
    # A near-immobile control phase (early training, kr ~ floor) sends Q/den to
    # astronomic pressures; clip to a generous physical range so the row stays
    # bounded (the IC/PDE groups restore mobility, un-railing the closed form).
    # The range must enclose every BHP the well actually reaches: a producer drawn
    # down to its limit sits far below the reservoir-pressure anchors (SPE2EQUI ends
    # at 474 psia against P_MIN = 2400), and a floor above it would clip the closed
    # form exactly where the observation rows need it most.
    p_span = float(case.phys.P_MAX) - float(case.phys.P_MIN)
    lo_obs, hi_obs = _head_range(pack, case, margin=0.5)
    p_bh_hat = jnp.clip(num / den,
                        min(float(case.phys.P_MIN) - p_span, lo_obs),
                        max(float(case.phys.P_MAX) + 3.0 * p_span, hi_obs))

    # fixed p_bh where the closed form is not used: observed BHP on limited
    # steps (data of the control), the schedule BHP target otherwise
    bhp_fill = jnp.asarray(onp.nan_to_num(pack.bhp_obs), dt)
    p_bh_fixed = jnp.where(binding > 0, bhp_fill, jnp.asarray(pack.p_bh_ctrl, dt))
    p_bh_used = jnp.where(use_closed > 0, p_bh_hat, p_bh_fixed)
    # back-substitute with the SAME floored control-phase mobility the elimination used, so
    # the control-phase rate reproduces q_pin exactly; every other phase keeps its unfloored
    # mobility and an immobile phase carries no phantom flow
    onehot_c = jax.nn.one_hot(ctrl_ph_perf, 3, dtype=dt)              # (T, n_perf, 3)
    state_c = (f_eff + f_floor * onehot_c, gam, p_ph, wi, dz, rs, f_floor)
    q_hat, _ = predicted_head(state_c, p_bh_used, pack, case)
    return p_bh_hat, q_hat, p_bh_used


def predicted_head(state, p_wf, pack: WellPack, case: CaseData, active=None):
    r"""
    The ``predicted`` head: per-perforation Peaceman rates from a given flowing
    pressure,

    .. math::

        q_{\alpha,p} = C_F\, \mathrm{WI}_p\, f^{\mathrm{eff}}_{\alpha,p}\,
        \bigl(p_{wf,w(p)} + \gamma_{\alpha,p}\Delta z_p - p_{\alpha,p}\bigr)

    ``p_wf`` has shape ``(..., n_wells)`` matching the leading axes of the state.
    Returns ``(q_hat, q_perf)``: the per-well totals ``(..., n_wells, 3)`` in
    canonical order with the dissolved :math:`R_s q_o` folded into the gas leg,
    and the per-perforation rates ``(..., n_perf, 3)`` in **residual** order
    ``(w, o, g_total)`` — the interior source's layout. ``active`` (``(..., n_wells)``)
    zeroes shut wells.
    """
    import jax.numpy as jnp
    from jax.ops import segment_sum

    f_eff, gam, p_ph, wi, dz, rs, _f_floor = state
    dt = p_ph.dtype
    n_wells = pack.n_wells
    c_f = jnp.asarray(case.c_f, dt)
    wid = jnp.asarray(pack.well_id, jnp.int32)

    p_wf = jnp.asarray(p_wf, dt)
    p_wb = p_wf[..., wid][..., None] + gam * dz[..., :, None] if p_wf.ndim > 1 else \
        p_wf[wid][:, None] + gam * dz[:, None]
    q_perf = c_f * wi * f_eff.swapaxes(-1, -2)                        # (..., 3, n_perf)
    q_perf = q_perf.swapaxes(-1, -2) * (p_wb - p_ph)                  # (..., n_perf, 3)
    if active is not None:
        act = jnp.asarray(active, dt)
        act_perf = act[..., wid] if act.ndim > 1 else act[wid]
        q_perf = q_perf * act_perf[..., None]
    q_gas_diss = rs * q_perf[..., 0]                                  # dissolved gas in the oil stream
    q_perf_res = jnp.stack([q_perf[..., 1], q_perf[..., 0], q_perf[..., 2] + q_gas_diss], axis=-1)

    def seg(x):                                                       # (..., n_perf) -> (..., n_wells)
        flat = x.reshape(-1, x.shape[-1])
        out = segment_sum(flat.T, wid, num_segments=n_wells).T
        return out.reshape(x.shape[:-1] + (n_wells,))

    q_hat = jnp.stack([seg(q_perf[..., 0]), seg(q_perf[..., 1]),
                       seg(q_perf[..., 2] + q_gas_diss)], axis=-1)    # (..., n_wells, 3)
    return q_hat, q_perf_res


def _closure_core(P4, pack: WellPack, case: CaseData, tables, kr_floor: float, wi_mult=None):
    r"""
    Evaluate the closed-form well closure on perforation primaries under the
    resolved control modes (``P4`` of shape ``(T, n_perf, 4)``). Returns
    ``(p_bh_hat, q_hat, p_bh_used)`` — see :func:`closed_form_head`.
    """
    import jax.numpy as jnp

    # the closed form reads the injector flag from the pinned rate sign (its own convention)
    wid = jnp.asarray(pack.well_id)
    is_inj = jnp.asarray((pack.q_pin > 0).astype(onp.float64), P4.dtype)[:, wid]
    inj_phase = jnp.asarray(pack.ctrl_phase, jnp.int32)[:, wid]
    state = _perf_state(P4, pack, case, tables, kr_floor, wi_mult=wi_mult, is_inj=is_inj,
                        inj_phase=inj_phase)
    return closed_form_head(state, pack, case)


def fischer_burmeister(a, b, eps: float = FB_EPS):
    r"""
    Smoothed Fischer–Burmeister complementarity residual,

    .. math::

        \phi_\epsilon(a, b) \;=\; \sqrt{a^2 + b^2 + \epsilon^2} \;-\; a \;-\; b,

    which vanishes (up to :math:`\epsilon`) exactly on the set
    :math:`a \ge 0,\ b \ge 0,\ ab = 0`. For the well's control switch
    :math:`a = \mathrm{sgn}_w (p_{wf} - p^{\mathrm{lim}}_{bh})/s_{bhp}` (the well may not
    cross its BHP limit) and :math:`b = (|q^{\mathrm{ctrl}}| - |q_c(\theta)|)/s_{q_c}`
    (it may not exceed its rate target): a well either meets its rate above the
    limit or sits on the limit below its rate — the simulator's rule, in one
    :math:`C^\infty` row with bounded derivatives.

    where:
    - :math:`\epsilon` (``eps``): the smoothing constant; :math:`\phi_0` is the exact (kinked) FB function.
    """
    import jax.numpy as jnp

    return jnp.sqrt(a * a + b * b + eps * eps) - a - b


def synthesize_observations(case: CaseData, pack: WellPack, cfg: RunConfig) -> None:
    """Fill the pack's observation arrays from the reference cell states (in place).

    The generator runs with the schedule controls (no observations exist yet:
    ``q_pin = q_ctrl``, no binding steps), then the control modes are
    re-resolved from the synthesized channels.
    """
    import jax.numpy as jnp

    ref = onp.stack([case.pres, case.swat, case.sgas, case.rs], axis=-1)   # (T, n_cells, 4)
    P4 = jnp.asarray(ref[:, pack.cell_idx, :], jnp.float64)
    p_bh_hat, q_hat, p_bh_used = _closure_core(P4, pack, case, case.tables,
                                               kr_floor=float(cfg.inv.kr_floor))
    p_bh = onp.asarray(p_bh_used)
    q = onp.asarray(q_hat)

    pack.bhp_obs = p_bh
    pack.bhp_mask = pack.active.copy()
    ch_obs = onp.zeros((pack.n_times, pack.n_wells, 5))
    ch_mask = onp.zeros_like(ch_obs)
    for c, (ph, sign) in enumerate(zip(_CH_PHASE, _CH_SIGN)):
        vals = sign * q[:, :, ph]
        keep = (vals > -1e-9) & (pack.active > 0)   # a channel only reports its flow direction
        ch_obs[:, :, c] = onp.where(keep, onp.maximum(vals, 0.0), 0.0)
        ch_mask[:, :, c] = keep.astype(onp.float64)
    pack.ch_obs = ch_obs
    pack.ch_mask = ch_mask
    _resolve_control_modes(pack)


# ---------------------------------------------------------------------------------------------
# Imposed schedules
# ---------------------------------------------------------------------------------------------

_FIELD_UNITS_WARNED = False
# FlowSchedule control mode -> canonical phase index (0=oil, 1=water, 2=gas); BHP has none.
_FLOW_MODE_PHASE = {"ORAT": 0, "WRAT": 1, "GRAT": 2}


def _schedule_to_field(sched):
    r"""
    One :class:`~pinnlab.config.FlowSchedule` record as FIELD-unit ``(rate, bhp_limit)``.

    Everything downstream of the prep cache is FIELD, so a METRIC record is rescaled
    on ingest by the same factors the cache itself was normalized with:

    .. math::

        q_{\mathrm{FIELD}} = \gamma_q\, q_{\mathrm{METRIC}},
        \qquad
        p_{\mathrm{FIELD}} = \gamma_p\, p_{\mathrm{METRIC}}

    where:
    - :math:`\gamma_p = 14.5038`: bar :math:`\to` psia.
    - :math:`\gamma_q`: :math:`6.2898` (sm³ :math:`\to` stb) for an ``ORAT``/``WRAT`` record, :math:`0.035315` (sm³ :math:`\to` Mscf) for ``GRAT``.

    A record that leaves ``units`` unset is taken as FIELD and warned about once per
    session; stating :attr:`~pinnlab.config.FlowUnit.FIELD` explicitly silences it.
    """
    global _FIELD_UNITS_WARNED
    from modules.utils.unit_conversion import BAR_TO_PSI, M3_TO_BBL, SM3_GAS_TO_MSCF

    if sched.units is None:
        if not _FIELD_UNITS_WARNED:
            _FIELD_UNITS_WARNED = True
            warnings.warn(
                "FlowSchedule records do not declare `units`; their rate and bhp_limit are "
                "read as FIELD (stb/day, Mscf/day, psia) to match every other cached "
                "quantity. Set units=FlowUnit.FIELD to silence this, or FlowUnit.METRIC "
                "to have sm3/day and bar converted for you.",
                RuntimeWarning, stacklevel=2)
        return float(sched.rate), float(sched.bhp_limit)
    if str(sched.units.value).upper() == "METRIC":
        g_q = SM3_GAS_TO_MSCF if sched.mode == "GRAT" else M3_TO_BBL
        return float(sched.rate) * g_q, float(sched.bhp_limit) * BAR_TO_PSI
    return float(sched.rate), float(sched.bhp_limit)


def apply_schedules(pack: WellPack, schedules, t_from: float = 0.0, case: CaseData | None = None) -> WellPack:
    r"""
    Impose user :class:`~pinnlab.config.FlowSchedule` records on a pack's controls,
    in place, from ``t_from`` onward.

    Each record takes effect at its own :math:`t_{\mathrm{start}}` and holds until the
    next record for the same well, so for well :math:`w` the active record at time
    :math:`t` is

    .. math::

        r_w(t) \;=\; \arg\max_{\,r \,\in\, R_w,\; t_{\mathrm{start}}(r) \,\le\, t}\; t_{\mathrm{start}}(r)

    where:
    - :math:`R_w`: the records naming well :math:`w`.
    - :math:`t_{\mathrm{start}}(r)`: the record's activation time [days].
    - :math:`t_{\mathrm{from}}`: the split time; report steps before it keep their deck controls untouched.

    This is the deck's own step-function semantics, and it **overrides rather than
    replaces**: a (well, step) pair no record covers keeps the controls the deck
    resolved for it. Only the control arrays move; the observation arrays
    (``bhp_obs``/``ch_obs`` and their masks) are left alone, because a forecast has
    no observations — the residual-row masks are then re-derived from the new
    controls by :func:`_resolve_control_modes`, exactly as
    :func:`synthesize_observations` does for the inverse-crime bridge. A rate
    record's ``bhp_limit`` lands in ``p_bh_ctrl`` and, under the ``predicted``
    model, becomes the control-switch row of that step.
    """
    if not schedules:
        return pack

    names = {n: i for i, n in enumerate(pack.well_names)}
    unknown = sorted({s.well for s in schedules if s.well not in names})
    if unknown:
        raise ValueError(f"FlowSchedule names unknown well(s) {unknown}; "
                         f"the case carries {list(pack.well_names)}")

    times = onp.asarray(pack.times, onp.float64)
    span = times >= float(t_from)
    for w_name, w in names.items():
        recs = sorted((s for s in schedules if s.well == w_name), key=lambda s: s.t_start)
        if not recs:
            continue
        for k, rec in enumerate(recs):
            t_hi = recs[k + 1].t_start if k + 1 < len(recs) else onp.inf
            sel = span & (times >= rec.t_start) & (times < t_hi)
            if not sel.any():
                continue
            rate, bhp = _schedule_to_field(rec)
            if not rec.open_:                              # shut: no control, no forcing
                pack.active[sel, w] = 0.0
                pack.is_rate[sel, w] = 0.0
                pack.q_ctrl[sel, w] = 0.0
                pack.p_bh_ctrl[sel, w] = 0.0
                continue
            pack.active[sel, w] = 1.0
            if rec.mode == "BHP":
                pack.is_rate[sel, w] = 0.0
                pack.q_ctrl[sel, w] = 0.0
                pack.p_bh_ctrl[sel, w] = bhp
            else:
                pack.is_rate[sel, w] = 1.0
                pack.q_ctrl[sel, w] = rate
                pack.ctrl_phase[sel, w] = _FLOW_MODE_PHASE[rec.mode]
                pack.p_bh_ctrl[sel, w] = bhp

    _resolve_control_modes(pack)
    _finalize_rows_and_scales(pack, case)
    return pack


# ---------------------------------------------------------------------------------------------
# The residual builder
# ---------------------------------------------------------------------------------------------

@dataclass
class WellOps:
    """The surfaces one well pack exposes to the residual/loss layer."""

    well_arr: Callable                  # (params) -> scaled observation rows (well group)
    well_predict: Callable              # (params) -> (p_bh (T,W), q_hat (T,W,3)) diagnostics
    ctrl_arr: Callable | None = None    # (params) -> scaled control rows (ctrl group)
    forcing: Any = None                 # PredictedForcing under the predicted model
    ctrl_parts: Callable | None = None  # (params) -> dict of the unscaled ctrl pieces (diagnostics)


def make_well_residual(cfg: RunConfig, case: CaseData, pack: WellPack, prim,
                       centroids, eff_tables: Callable | None = None,
                       wi_mult_of: Callable | None = None, head: WellHead | None = None) -> WellOps:
    r"""
    Build the well surfaces: ``well_arr(params) -> rows`` (scaled observation
    residual array for the ``well`` group), ``well_predict(params) -> (p_bh, q_hat)``
    (the diagnostics surface) and, under the ``predicted`` model, ``ctrl_arr`` (the
    ``ctrl`` group rows) and the :class:`PredictedForcing` whose ``rates_at(params, t)``
    is the interior source. ``eff_tables`` injects θ_m-aware closure tables;
    ``wi_mult_of`` injects the Stage-3 log-k well-index multiplier
    :math:`\mathrm{WI} \propto \sqrt{\det K_\perp} \Rightarrow e^{\Psi c}`.
    """
    import jax
    import jax.numpy as jnp
    from jax import vmap

    encoder = prim.encoder
    tables0 = case.tables
    eff = eff_tables if eff_tables is not None else (lambda params: tables0)
    kr_floor = float(cfg.inv.kr_floor)
    predicted = pack.well_model == WellModel.PREDICTED.value
    if predicted and head is None:
        raise ValueError("well_model=predicted needs the WellHead (build it with WellHead.from_pack)")

    ci = jnp.asarray(pack.cell_idx, jnp.int32)
    enc_args = encoder.gather_args(ci)
    cxyz = centroids[ci]                                            # (n_perf, 3)
    times = jnp.asarray(pack.times, jnp.float32)
    wid = jnp.asarray(pack.well_id, jnp.int32)

    # static row gathers from the mode-resolved masks
    bhp_sel = onp.nonzero(pack.bhp_row_mask.ravel() > 0)[0]
    ch_sel = onp.nonzero(pack.ch_row_mask.reshape(pack.n_times * pack.n_wells, 5).T.ravel() > 0)[0]
    s_bhp, s_qo, s_qw, s_qg = pack.well_scale
    ch_scale = onp.asarray([s_qo, s_qw, s_qg])[list(_CH_PHASE)]      # (5,)

    bhp_obs_flat = jnp.asarray(pack.bhp_obs.ravel()[bhp_sel], jnp.float32)
    ch_obs_all = onp.transpose(pack.ch_obs, (2, 0, 1)).reshape(5, -1)
    ch_scale_rows = onp.repeat(ch_scale[:, None], ch_obs_all.shape[1], axis=1).ravel()[ch_sel]
    ch_obs_flat = jnp.asarray(ch_obs_all.ravel()[ch_sel], jnp.float32)
    ch_scale_flat = jnp.asarray(ch_scale_rows, jnp.float32)
    ch_sign = onp.repeat(onp.asarray(_CH_SIGN)[:, None], ch_obs_all.shape[1], axis=1).ravel()[ch_sel]
    ch_sign_flat = jnp.asarray(ch_sign, jnp.float32)

    boost = None
    if cfg.inv.coning_boost > 0:
        # emphasize rows around observed rate transients (breakthrough fronts)
        dch = onp.abs(onp.diff(pack.ch_obs, axis=0, prepend=pack.ch_obs[:1]))
        dmax = onp.maximum(dch.max(axis=(0, 1), keepdims=True), 1e-12)
        bw = 1.0 + cfg.inv.coning_boost * (dch / dmax)
        boost = jnp.asarray(onp.sqrt(
            onp.transpose(bw, (2, 0, 1)).reshape(5, -1).ravel()[ch_sel]), jnp.float32)

    def _perf_primaries(params):
        def one_t(t):
            xt = jnp.concatenate([cxyz, jnp.full((cxyz.shape[0], 1), t, cxyz.dtype)], axis=1)
            return vmap(lambda x, *a: prim.primaries_point(params, x, *a))(xt, *enc_args)

        return jax.lax.map(one_t, times)                            # (T, n_perf, 4)

    def _wi_mult(params):
        # k_mult is the full-mesh (n_cells,) multiplier; WI wants it at the perf cells
        return wi_mult_of(params)[ci] if wi_mult_of is not None else None

    def _ch_rows(q_hat):
        q_ch = jnp.transpose(q_hat, (2, 0, 1)).reshape(3, -1)        # canonical (3, T*n_wells)
        q_rows = jnp.concatenate([q_ch[ph][None, :] for ph in _CH_PHASE], axis=0).reshape(-1)[ch_sel]
        r_ch = (ch_sign_flat * q_rows - ch_obs_flat) / ch_scale_flat
        if boost is not None:
            r_ch = r_ch * boost
        return r_ch

    if not predicted:
        def _predict(params):
            P4 = _perf_primaries(params)
            return _closure_core(P4, pack, case, eff(params), kr_floor, wi_mult=_wi_mult(params))

        def well_predict(params):
            p_bh_hat, q_hat, p_bh_used = _predict(params)
            return p_bh_used, q_hat

        def well_arr(params):
            p_bh_hat, q_hat, _ = _predict(params)
            r_bhp = (p_bh_hat.reshape(-1)[bhp_sel] - bhp_obs_flat) / s_bhp
            return jnp.concatenate([r_bhp, _ch_rows(q_hat)])

        return WellOps(well_arr=well_arr, well_predict=well_predict)

    # ---------------- predicted well model ------------------------------------------------
    active = jnp.asarray(pack.active, jnp.float32)                   # (T, W)
    is_inj_perf = jnp.asarray(pack.is_inj, jnp.float32)[:, wid]      # (T, n_perf)
    inj_ph_perf = jnp.asarray(pack.inj_phase, jnp.int32)[:, wid]
    rate_sel = onp.nonzero(pack.ctrl_rate_mask.ravel() > 0)[0]
    bhp_c_sel = onp.nonzero(pack.ctrl_bhp_mask.ravel() > 0)[0]
    lim_sel = onp.nonzero(pack.ctrl_limit_mask.ravel() > 0)[0]
    ctrl_ph_flat = jnp.asarray(pack.ctrl_phase.ravel(), jnp.int32)
    q_ctrl_flat = jnp.asarray(pack.q_ctrl.ravel(), jnp.float32)
    p_lim_flat = jnp.asarray(pack.p_bh_ctrl.ravel(), jnp.float32)
    s_q = jnp.asarray([s_qo, s_qw, s_qg], jnp.float32)
    s_qc_flat = s_q[ctrl_ph_flat]
    sgn_flat = jnp.where(q_ctrl_flat > 0, -1.0, 1.0).astype(jnp.float32)   # producers +1
    hinge = cfg.ctrl_switch == "hinge"

    def _state_T(params):
        P4 = _perf_primaries(params)
        return _perf_state(P4, pack, case, eff(params), kr_floor, wi_mult=_wi_mult(params),
                           is_inj=is_inj_perf.astype(P4.dtype), inj_phase=inj_ph_perf)

    def _predict(params):
        state = _state_T(params)
        p_wf = head.pwf(params["well"]).astype(state[2].dtype)
        q_hat, _ = predicted_head(state, p_wf, pack, case, active=active)
        return p_wf, q_hat

    def well_predict(params):
        return _predict(params)

    def well_arr(params):
        p_wf, q_hat = _predict(params)
        r_bhp = (p_wf.reshape(-1)[bhp_sel] - bhp_obs_flat) / s_bhp
        return jnp.concatenate([r_bhp, _ch_rows(q_hat)])

    def ctrl_parts(params):
        p_wf, q_hat = _predict(params)
        p_flat = p_wf.reshape(-1)
        q_c = jnp.take_along_axis(q_hat.reshape(-1, 3), ctrl_ph_flat[:, None], axis=1)[:, 0]
        r_rate = (q_c - q_ctrl_flat) / s_qc_flat
        r_bhp = (p_flat - p_lim_flat) / s_bhp
        a = sgn_flat * (p_flat - p_lim_flat) / s_bhp
        b = (jnp.abs(q_ctrl_flat) - jnp.abs(q_c)) / s_qc_flat
        return {"rate": r_rate, "bhp": r_bhp, "a": a, "b": b,
                "fb": fischer_burmeister(a, b), "hinge": jnp.maximum(0.0, -a)}

    def ctrl_arr(params):
        parts = ctrl_parts(params)
        rows = [parts["rate"][rate_sel], parts["bhp"][bhp_c_sel]]
        if lim_sel.size:
            rows.append((parts["hinge"] if hinge else parts["fb"])[lim_sel])
        return jnp.concatenate(rows)

    # the interior source at an arbitrary collocation time
    is_inj_tab = jnp.asarray(pack.is_inj, jnp.float32)
    inj_ph_tab = jnp.asarray(pack.inj_phase, jnp.int32)
    act_tab = active

    def rates_at(params, t):
        t32 = jnp.asarray(t, jnp.float32)
        xt = jnp.concatenate([cxyz, jnp.full((cxyz.shape[0], 1), t32, cxyz.dtype)], axis=1)
        P4 = vmap(lambda x, *a: prim.primaries_point(params, x, *a))(xt, *enc_args)  # (n_perf, 4)
        inj = _nearest_row(times, is_inj_tab, t32)[wid].astype(P4.dtype)
        inj_ph = _nearest_row(times, inj_ph_tab, t32)[wid]
        state = _perf_state(P4, pack, case, eff(params), kr_floor, wi_mult=_wi_mult(params),
                            is_inj=inj, inj_phase=inj_ph)
        p_wf = head.pwf_at(params["well"], t32).astype(P4.dtype)
        act = _nearest_row(times, act_tab, t32).astype(P4.dtype)
        _, q_perf = predicted_head(state, p_wf, pack, case, active=act)
        return q_perf                                                 # (n_perf, 3) residual order

    cent = onp.asarray(case.centroids)
    cell_len = onp.asarray(case.cell_len)
    forcing = PredictedForcing(
        times=times, q_perf=None,
        perf_xyz=jnp.asarray(cent[pack.cell_idx], jnp.float32),
        perf_sigma=jnp.asarray(cell_len[pack.cell_idx], jnp.float32),
        rates_at=rates_at)
    return WellOps(well_arr=well_arr, well_predict=well_predict, ctrl_arr=ctrl_arr,
                   forcing=forcing, ctrl_parts=ctrl_parts)


def well_match_metrics(pack: WellPack, p_bh_hat: onp.ndarray, q_hat: onp.ndarray,
                       p_clip: tuple[float, float] | None = None,
                       p_anchors: tuple[float, float] | None = None) -> dict:
    """Row-masked observation-match RMSEs (BHP in psi, rates per channel scale), the
    control-match RMSEs of the ``ctrl`` rows (predicted model), and the rail fraction:
    under ``closed_form`` the share of rate-mode steps whose eliminated BHP sits on its
    clip bound (``p_clip``), under ``predicted`` the share of active steps whose head
    sits within ``_RAIL_TOL`` of the pressure anchors (``p_anchors``) — a railed head
    is a well whose control the state cannot deliver, and its rows carry no gradient."""
    out: dict = {}
    m = pack.bhp_row_mask > 0
    if m.any():
        out["well_bhp_rmse"] = float(onp.sqrt(onp.mean((p_bh_hat[m] - pack.bhp_obs[m]) ** 2)))
    s_bhp, s_qo, s_qw, s_qg = pack.well_scale
    scale = onp.asarray([s_qo, s_qw, s_qg])
    errs = []
    for c, (ph, sign) in enumerate(zip(_CH_PHASE, _CH_SIGN)):
        mc = pack.ch_row_mask[:, :, c] > 0
        if mc.any():
            errs.append((sign * q_hat[:, :, ph][mc] - pack.ch_obs[:, :, c][mc]) / scale[ph])
    if errs:
        e = onp.concatenate(errs)
        out["well_rate_rmse"] = float(onp.sqrt(onp.mean(e ** 2)))
    predicted = pack.well_model == WellModel.PREDICTED.value
    if predicted:
        q_c = onp.take_along_axis(q_hat, pack.ctrl_phase[:, :, None], axis=2)[:, :, 0]
        mr = pack.ctrl_rate_mask > 0
        if mr.any():
            out["ctrl_rate_rmse"] = float(onp.sqrt(onp.mean(
                ((q_c[mr] - pack.q_ctrl[mr]) / scale[pack.ctrl_phase[mr]]) ** 2)))
        mb = pack.ctrl_bhp_mask > 0
        if mb.any():
            out["ctrl_bhp_rmse"] = float(onp.sqrt(onp.mean((p_bh_hat[mb] - pack.p_bh_ctrl[mb]) ** 2)))
        ml = pack.ctrl_limit_mask > 0
        if ml.any():
            sgn = onp.where(pack.q_ctrl > 0, -1.0, 1.0)
            a = sgn * (p_bh_hat - pack.p_bh_ctrl) / s_bhp
            b = (onp.abs(pack.q_ctrl) - onp.abs(q_c)) / scale[pack.ctrl_phase]
            out["ctrl_switch_frac"] = float((onp.abs(a[ml]) < onp.abs(b[ml])).mean())
        if p_anchors is not None:
            lo, hi = p_anchors
            tol = _RAIL_TOL * (hi - lo)
            act = pack.active > 0
            if act.any():
                railed = (p_bh_hat[act] <= lo + tol) | (p_bh_hat[act] >= hi - tol)
                out["well_bhp_rail_frac"] = float(railed.mean())
    elif p_clip is not None:
        closed = (pack.is_rate * (1.0 - pack.binding)) > 0
        if closed.any():
            lo, hi = p_clip
            tol = 1e-3 * (hi - lo)
            railed = (p_bh_hat[closed] <= lo + tol) | (p_bh_hat[closed] >= hi - tol)
            out["well_bhp_rail_frac"] = float(railed.mean())
    return out


def material_balance_obs(case: CaseData, pack: WellPack, n_probes: int):
    """
    Cumulative observed oil production at strided probe times (for the
    material-balance penalty): ``(t_probe [days], q_cum [stb])``.

    Uses the ``wopr`` channel where observed and falls back to the control rate
    of oil-rate-controlled producers elsewhere.
    """
    wopr = onp.where(pack.ch_mask[:, :, 0] > 0, pack.ch_obs[:, :, 0], onp.nan)
    ctrl = onp.where((pack.ctrl_phase == 0) & (pack.is_rate > 0) & (pack.q_ctrl < 0),
                     -pack.q_ctrl, onp.nan)
    rate = onp.where(onp.isfinite(wopr), wopr, ctrl)                 # (T, n_wells) production+
    rate = onp.nan_to_num(rate).sum(axis=1)                          # (T,)
    t = onp.asarray(pack.times, onp.float64)
    dt = onp.diff(t, prepend=t[:1])
    q_cum = onp.cumsum(0.5 * (rate + onp.roll(rate, 1)) * dt)
    q_cum[0] = 0.0
    idx = onp.unique(onp.linspace(1, len(t) - 1, num=min(n_probes, len(t) - 1)).astype(int))
    return t[idx], q_cum[idx]
