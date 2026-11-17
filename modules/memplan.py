r"""
Memory planning: parameter counts, the per-device OOM estimate, and the ENGD
feasibility plan (extracted from the notebook's back-of-napkin OOM cell).

All estimates are pure Python (no allocation, no jax import) so they can run
at :func:`modules.config.resolve` time, before any device buffer exists.

Parameter counts
----------------

The DGM network (Sirignano–Spiliopoulos) with input width :math:`d`, hidden
width :math:`M`, output width :math:`f` and :math:`\ell` recurrent blocks has

.. math::

    P_{\mathrm{DGM}}
    \;=\;
    dM + M \;+\; \ell\,\bigl(4dM + 4M^2 + 4M\bigr) \;+\; Mf + f

where:
- :math:`dM + M`: the input projection :math:`W_1, b_1`.
- :math:`4dM + 4M^2 + 4M`: one block's four gates, each with an input matrix :math:`U \in \mathbb{R}^{d\times M}`, a state matrix :math:`W \in \mathbb{R}^{M\times M}` and a bias.
- :math:`Mf + f`: the readout :math:`W_{\mathrm{out}}, b_{\mathrm{out}}`.

The MLP with layer widths :math:`(d, h_1, \dots, h_k, f)` has
:math:`P_{\mathrm{MLP}} = \sum_i (d_i d_{i+1} + d_{i+1})`.

ENGD cost model
---------------

With :math:`P` parameters and :math:`N` weighted residual rows, the two
implemented solve regimes cost

.. math::

    \text{dense:}\quad
    \mathcal{O}(N P^2) \text{ time}, \; \mathcal{O}(P^2) \text{ memory},
    \qquad
    \text{rowspace:}\quad
    \mathcal{O}(N^2 P) \text{ time}, \; \mathcal{O}(N P + N^2) \text{ memory}

where:
- dense builds :math:`G = \hat J^\top \hat J` and solves the :math:`P \times P` system (feasible when :math:`8P^2` bytes fit the budget).
- rowspace materializes :math:`\hat J \in \mathbb{R}^{N\times P}` once and solves the :math:`N \times N` system :math:`(\hat J \hat J^\top + \varepsilon I)\alpha = 2\hat r` (feasible when :math:`8NP + 2\cdot 8N^2` bytes — :math:`\hat J` plus the damped row Gram and its Cholesky factor — fit the budget).

Auto mode picks the feasible regime with the smaller solve system (dense iff
:math:`P \le N`); ``dense_p_max`` is a hard cap only when ``mode="dense"`` is
forced, and a slow-solve warning threshold otherwise.

Independently of the regime, :math:`\hat J` is assembled in the AD direction
with the fewer residual-graph sweeps: forward jvp columns (:math:`P` sweeps,
requires the materialized :math:`\hat J` to fit ``jac_budget_bytes``) when
:math:`P < N`, reverse vjp rows (:math:`N` sweeps, streamable) otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    Architecture,
    BackpropDesign,
    ConfigError,
    ResidualDesign,
    RunConfig,
    SpecialOpt,
    dim_in_of,
    load_well_pack_meta,
    needs_eigenbasis,
    phys_param_count,
)


def param_count(cfg: RunConfig, dim_in: int | None = None) -> int:
    """Exact trainable-parameter count: network weights plus inverted physical parameters."""
    d = dim_in_of(cfg) if dim_in is None else dim_in
    f = cfg.dim_out
    if cfg.architecture is Architecture.DGM:
        m = cfg.m_width
        net = d * m + m + cfg.n_blocks * (4 * d * m + 4 * m * m + 4 * m) + m * f + f
    else:
        layers = (d, *cfg.mlp_hidden, f)
        net = sum(a * b + b for a, b in zip(layers[:-1], layers[1:]))
    return net + phys_param_count(cfg) + well_head_param_count(cfg)


def well_head_param_count(cfg: RunConfig) -> int:
    """Trainable p_wf head parameters (T x n_wells) under ``well_model=predicted``, else 0."""
    from .config import predicted_well

    if not predicted_well(cfg):
        return 0
    meta = load_well_pack_meta(cfg)
    if meta is not None:
        return int(meta["n_times"]) * int(meta["n_wells"])
    try:
        from .config import load_case_meta
        return int(load_case_meta(cfg)["n_times"]) * _WELL_FALLBACK_WELLS
    except ConfigError:
        return 0


# ---------------------------------------------------------------------------------------------
# ENGD feasibility
# ---------------------------------------------------------------------------------------------

# Residual components per row source: 3 phase balances (oil/water/gas) for PDE
# and BC-flux rows, 4 primaries (p_o, S_w, S_g, R_so) for IC/data misfit rows.
_PDE_COMPONENTS = 3
_STATE_COMPONENTS = 4

# Well-group fallback bound when the well_pack_meta.json sidecar has not been
# written yet: at most one BHP row plus three phase-rate rows per well per
# report step, for an assumed-generous well count.
_WELL_ROW_CHANNELS = 4
_WELL_FALLBACK_WELLS = 32


# Transient bytes of one point-wise Gramian block per byte of its (chunk, m, P) float64
# Jacobian stack, measured on NORNE (P = 15,164, 2048 points of 4 rows): 27.8 GiB under
# the XLA CPU compiler, 1.45 GiB on a T4, whose rematerialization keeps the tapes short.
_POINT_CHUNK_FACTOR = {"cpu": 30.0}
_POINT_CHUNK_FACTOR_ACCEL = 2.0
_POINT_CHUNK_MIN, _POINT_CHUNK_MAX = 64, 2048


def point_chunk_for(point_chunk: int, budget_bytes: int, m: int, p: int,
                    backend: str = "gpu") -> int:
    r"""
    The points per block of a point-wise Gramian group (:func:`modules.engd.gramian_pointwise`).

    A positive ``point_chunk`` is returned as given; ``0`` sizes the block so its transient
    fits the Jacobian budget,

    .. math::

        n_{\mathrm{chunk}} \;=\; \operatorname{clip}\!\left(\left\lfloor
        \frac{B_{\mathrm{jac}}}{c_{\mathrm{pt}}\cdot 8\, m\, P}\right\rfloor,\; 64,\; 2048\right)

    where:
    - :math:`B_{\mathrm{jac}}`: ``budget_bytes`` (``engd.jac_budget_bytes``); :math:`m`: rows per point; :math:`P`: the parameter count.
    - :math:`c_{\mathrm{pt}}`: the backend's measured transient per Jacobian-block byte (30 on CPU, 2 on accelerators).
    """
    if point_chunk > 0:
        return int(point_chunk)
    c = _POINT_CHUNK_FACTOR.get(backend, _POINT_CHUNK_FACTOR_ACCEL)
    n = int(budget_bytes // max(c * 8.0 * max(m, 1) * max(p, 1), 1.0))
    return max(_POINT_CHUNK_MIN, min(_POINT_CHUNK_MAX, n))


@dataclass(frozen=True)
class EngdPlan:
    """Resolved ENGD solve regime with its memory footprint."""

    mode: str          # "dense" | "rowspace"
    p: int             # trainable parameters
    n_rows: int        # weighted residual rows N
    g_bytes: int       # dense Gramian storage (P^2 * 8)
    j_bytes: int       # materialized-Jacobian storage (N * P * 8): rowspace + dense-fwd
    seed_bytes: int = 0    # one-hot seed sweep block (chunk * N * 8 rev, chunk * P * 8 fwd)
    direction: str = "rev"  # Jacobian assembly over the WHOLE row set: "fwd" (P jvp columns) |
                            # "rev" (N vjp rows). The step re-decides this per loss group when
                            # the row function exposes group slices (engd.gramian_grouped).
    a_bytes: int = 0   # rowspace N x N row Gram (N^2 * 8); solve peak ~ 2x (damped + factor)


def engd_group_rows(
    cfg: RunConfig,
    groups: tuple[str, ...],
    batches: dict,
    case_meta: dict,
    full_batch: bool,
) -> dict:
    """
    Upper-bound count of weighted residual rows per ENGD step, per loss group.

    Uses the resolved window batch sizes (or the capped deterministic
    full-batch sizes when ``special_opt_after == 0``). Boundary-face counts are
    approximated from above by ``2 * n_cells``; :mod:`modules.loss` owns the
    exact per-group row layout.
    """
    n_cells = case_meta["n_cells"]
    n_nodes = case_meta["n_nodes"]
    n_times = case_meta["n_times"]
    cap = cfg.full_batch_cap

    rows: dict = {}
    for g in groups:
        if g == "pde":
            if cfg.residual_design is ResidualDesign.SPECTRAL_PDE:
                rows[g] = cfg.n_eig * _PDE_COMPONENTS * cfg.n_tslice
            elif cfg.residual_design is ResidualDesign.HYBRID_PDE:
                rows[g] = (cfg.n_eig + 2 * n_nodes) * cfg.n_tslice
            elif cfg.backprop_design is BackpropDesign.FEM_NODAL:
                rows[g] = n_nodes * _PDE_COMPONENTS * cfg.n_tslice
            else:  # cartesian_pde + chain_rule
                n_pts = min(n_cells * cfg.n_tslice, cap) if full_batch else batches["pde"]
                rows[g] = n_pts * _PDE_COMPONENTS
        elif g == "ic":
            n_pts = min(n_cells, cap) if full_batch else batches["ic"]
            rows[g] = n_pts * _STATE_COMPONENTS
        elif g == "data":
            n_pts = min(n_cells * n_times, cap) if full_batch else batches["data"]
            rows[g] = n_pts * _STATE_COMPONENTS
        elif g == "well":
            meta = load_well_pack_meta(cfg)
            if meta is not None:
                rows[g] = int(meta.get("n_rows_well", meta["n_rows"]))
            else:
                rows[g] = _WELL_ROW_CHANNELS * _WELL_FALLBACK_WELLS * n_times
        elif g == "ctrl":
            meta = load_well_pack_meta(cfg)
            if meta is not None and "n_rows_ctrl" in meta:
                rows[g] = int(meta["n_rows_ctrl"])
            else:
                rows[g] = 2 * _WELL_FALLBACK_WELLS * n_times
        elif g == "aqu":
            from .aquifer import deck_aquifers

            rows[g] = n_times * len(deck_aquifers(cfg))
        elif g == "eq":
            from .equilibrium import point_layout
            from .sampling import _slice_mode

            if _slice_mode(cfg):
                n_ic = min(n_cells, cap) if full_batch else cfg.n_ic_per_dev
                rows[g] = point_layout(n_ic, cfg.n_tslice, cap)[1] * cfg.n_tslice
            else:
                rows[g] = min(n_cells * cfg.n_tslice, cap) if full_batch else batches["pde"]
        elif g == "mb":
            rows[g] = _PDE_COMPONENTS * cfg.n_tslice
        elif g == "reg":
            rows[g] = phys_param_count(cfg) + cfg.inv.n_mb_probes
        elif g == "bc":
            n_pts = min(2 * n_cells, cap) if full_batch else batches["bc"]
            rows[g] = n_pts * _PDE_COMPONENTS
    return rows


def engd_row_count(
    cfg: RunConfig,
    groups: tuple[str, ...],
    batches: dict,
    case_meta: dict,
    full_batch: bool,
) -> int:
    """Upper-bound count of weighted residual rows per ENGD step (:func:`engd_group_rows`
    summed over the groups)."""
    return sum(engd_group_rows(cfg, groups, batches, case_meta, full_batch).values())


def engd_plan(
    cfg: RunConfig,
    groups: tuple[str, ...],
    batches: dict,
    case_meta: dict,
    full_batch: bool,
    p: int | None = None,
) -> EngdPlan:
    """Select the ENGD solve regime, or raise :class:`ConfigError` with remedies."""
    if p is None:
        p = param_count(cfg)
    n_rows = engd_row_count(cfg, groups, batches, case_meta, full_batch)
    s = cfg.engd
    g_bytes = p * p * 8
    j_bytes = n_rows * p * 8
    a_bytes = n_rows * n_rows * 8

    # Assembly direction: forward jvp columns (P sweeps) beat reverse vjp rows
    # (N sweeps) whenever P < N, but dense-fwd materializes J, so it must also
    # fit the Jacobian budget.
    direction = "fwd" if (p < n_rows and j_bytes <= s.jac_budget_bytes) else "rev"
    seed_bytes = (min(s.row_chunk, p) * p * 8 if direction == "fwd"
                  else min(s.row_chunk, n_rows) * n_rows * 8)

    def _plan(mode: str) -> EngdPlan:
        return EngdPlan(mode, p, n_rows, g_bytes, j_bytes, seed_bytes, direction, a_bytes)

    # Feasibility is byte-based against the shared budget: dense holds the P x P
    # Gramian; rowspace holds J plus the N x N row Gram and its Cholesky factor.
    dense_ok = g_bytes <= s.jac_budget_bytes
    rowspace_ok = j_bytes + 2 * a_bytes <= s.jac_budget_bytes
    budget_gb = s.jac_budget_bytes / 1e9

    if s.mode == "dense":
        if p > s.dense_p_max:
            raise ConfigError(
                f"engd.mode=dense but P={p:,} exceeds dense_p_max={s.dense_p_max:,} "
                f"(G would be {g_bytes/1e9:.1f} GB); shrink the net or use mode=auto/rowspace"
            )
        return _plan("dense")
    if s.mode == "rowspace":
        if not rowspace_ok:
            raise ConfigError(
                f"engd.mode=rowspace needs J ({j_bytes/1e9:.1f} GB) + the N x N row Gram and "
                f"factor (2 x {a_bytes/1e9:.1f} GB) for N={n_rows:,} rows x P={p:,} params, "
                f"over the {budget_gb:.1f} GB budget; reduce window sizes, lower full_batch_cap, "
                "use the spectral residual, or mode=auto to allow the dense P x P solve"
            )
        return _plan("rowspace")

    # auto: among feasible regimes take the smaller solve system (dense iff P <= N).
    if dense_ok and rowspace_ok:
        return _plan("dense" if p <= n_rows else "rowspace")
    if dense_ok:
        return _plan("dense")
    if rowspace_ok:
        return _plan("rowspace")
    raise ConfigError(
        f"ENGD infeasible: the dense Gramian is {g_bytes/1e9:.1f} GB (P={p:,} params) and the "
        f"row-space set is {(j_bytes + 2*a_bytes)/1e9:.1f} GB (N={n_rows:,} rows: J + 2 x N x N "
        f"Gram), both over the {budget_gb:.1f} GB budget. Remedies: architecture=mlp with a "
        "smaller net, residual_design=spectral_pde (few Galerkin rows), or smaller windows/caps."
    )


# ---------------------------------------------------------------------------------------------
# Per-device OOM estimate (port of the notebook's back-of-napkin calculator)
# ---------------------------------------------------------------------------------------------

# The dominant 2nd-order-AD + params term M*(a + b*N_EIG) is a 2-point fit to the empirical
# T4 (~15 GB usable) chain_rule OOM boundaries; the eigenbasis term is analytic (it equals
# the on-disk cache size). Monotone in every knob; trustworthy only NEAR the anchors.
_OOM_CTX_MB = 700.0
_OOM_T4_BUDGET = 15000.0
_OOM_ANCHORS = [(128, 800.0), (1000, 339.0)]   # (N_EIG, M at OOM boundary): chain_rule, mesh (1,1), Norne
_OOM_FIT_NN, _OOM_FIT_NCL = 61736, 44431       # the anchors' Norne calibration mesh
_FEM_TAPE_C = 24.0                             # floats/row/block, calibrated 2026-07-04 (SPE9 2xT4)


def _oom_eig_mb(n_eig: int, nn: int, ncl: int) -> float:
    return 4.0 * n_eig * (nn + ncl + 3 * ncl) / 1e6   # V_NODES + V_C + B_V (replicated, f32)


def _oom_fit(nn: int, ncl: int) -> tuple[float, float]:
    (e1, m1), (e2, m2) = _OOM_ANCHORS
    y1 = (_OOM_T4_BUDGET - _OOM_CTX_MB - _oom_eig_mb(e1, nn, ncl)) / m1
    y2 = (_OOM_T4_BUDGET - _OOM_CTX_MB - _oom_eig_mb(e2, nn, ncl)) / m2
    b = (y1 - y2) / (e1 - e2)
    return y1 - b * e1, b


_OOM_A, _OOM_B = _oom_fit(_OOM_FIT_NN, _OOM_FIT_NCL)


def oom_predict(
    n_eig: int, m: int, n_pde: int, data_dim: int, model_dim: int,
    dev_mb: float, nn: int, ncl: int, n_blocks: int = 10, ad_factor: float = 1.0,
) -> dict:
    """Chain-rule AD-tape + eigenbasis + context per-device MB estimate (cell-16 port)."""
    eig = _oom_eig_mb(n_eig, nn, ncl)
    ad = (m / model_dim) * (_OOM_A + _OOM_B * n_eig) * (n_pde / 256.0) * (n_blocks / 10.0) / data_dim * ad_factor
    tot = _OOM_CTX_MB + eig + ad
    return dict(total=tot, eig=eig, ad=ad, ctx=_OOM_CTX_MB, util=100.0 * tot / dev_mb, fits=tot <= dev_mb)


# GPU transient of the whole-mesh PDE group's reverse sweep, bytes per (node x slice) per
# (cotangent x hidden unit) and per input feature, and of a forward-swept row group per
# byte of its materialized Jacobian; fitted on the NORNE step at d = 65 and d = 202.
_PDE_TAPE_PER_COT_UNIT = 2.4
_PDE_TAPE_PER_FEATURE = 0.3
_FWD_GROUP_FACTOR = 3.0


def engd_step_bytes(cfg: RunConfig, resolved) -> int:
    r"""
    Device bytes the ENGD step holds beyond the resident model and data, on GPU.

    The dense step accumulates one Gramian across the loss groups, sweeping them one at
    a time (:func:`modules.engd.gramian_grouped`), then factors a damped copy of it, so
    its peak is the accumulator plus the largest single phase,

    .. math::

        B_{\mathrm{dense}} \;\approx\; 8P^2 + \max\bigl(2\cdot 8P^2,\; B_{\mathrm{pde}},\; B_{\mathrm{pt}},\; \max_g B_g\bigr),
        \qquad
        B_{\mathrm{pde}} \;\approx\; 8\, n_{\mathrm{nodes}}\, S\, \bigl(a\, c\, m + b\, d\bigr),
        \qquad
        B_{\mathrm{row}} \;\approx\; 8NP + 3\cdot 8N^2 + B_{\mathrm{seed}}

    where:
    - :math:`2\cdot 8P^2`: the damped copy and its Cholesky factor (:func:`modules.engd.solve_dense`).
    - :math:`B_{\mathrm{pde}}`: the reverse sweep of the whole-mesh PDE group under ``fem_nodal`` — :math:`c` cotangents (``engd.row_chunk``) pushed back at once through :math:`n_{\mathrm{nodes}}` nodes and :math:`S` time slices (``n_tslice``) of a network :math:`m` units wide, plus the :math:`d` input features; :math:`a = 2.4`, :math:`b = 0.3` fitted on NORNE, where the part independent of the input width :math:`d` dominates (2.27 GiB at :math:`d = 65`, 2.34 GiB at :math:`d = 202`).
    - :math:`B_{\mathrm{pt}}`: one point-wise (``ic``/``data``) block, :func:`point_chunk_for` on GPU.
    - :math:`B_g`: any other row group — swept forward when it has more rows than parameters and its Jacobian fits ``engd.jac_budget_bytes``, holding about three times that :math:`8 N_g P`-byte Jacobian, otherwise its one-hot seed block.
    - :math:`B_{\mathrm{row}}`: the row-space step — :math:`\hat J`, the row Gram, and its damped copy and factor; :math:`B_{\mathrm{seed}}` the plan's seed block.

    Checked on a T4 against XLA's compile-time memory analysis of the NORNE step: 5.14 GiB
    modelled and measured at :math:`P = 15{,}164`; 2.51 GiB modelled against 2.88 GiB at
    :math:`P = 5{,}300`, where the forward well group and the PDE sweep partly overlap. The
    pipeline's preflight (:mod:`modules.preflight`) replaces this estimate with that
    measurement once the step is compiled.
    """
    plan = resolved.engd_plan
    if plan is None:
        return 0
    if plan.mode != "dense":
        return plan.j_bytes + 3 * plan.a_bytes + plan.seed_bytes
    p, d, s = plan.p, resolved.dim_in, cfg.engd
    rows = engd_group_rows(cfg, resolved.groups, resolved.batches, resolved.case_meta,
                           resolved.full_batch)
    width = cfg.m_width if cfg.architecture is Architecture.DGM else max(cfg.mlp_hidden)
    phases = [2 * plan.g_bytes]
    for g, n_g in rows.items():
        if g in ("ic", "data"):
            m = _STATE_COMPONENTS
            phases.append(int(_POINT_CHUNK_FACTOR_ACCEL * 8 * m * p
                              * point_chunk_for(s.point_chunk, s.jac_budget_bytes, m, p)))
        elif g == "pde" and cfg.backprop_design is BackpropDesign.FEM_NODAL:
            cot = min(s.row_chunk, n_g)
            phases.append(int(8 * int(resolved.case_meta["n_nodes"]) * int(cfg.n_tslice)
                              * (_PDE_TAPE_PER_COT_UNIT * cot * width + _PDE_TAPE_PER_FEATURE * d)))
        elif n_g > p and 8 * n_g * p <= s.jac_budget_bytes:
            phases.append(int(_FWD_GROUP_FACTOR * 8 * n_g * p))
        else:
            phases.append(8 * min(s.row_chunk, n_g) * n_g)
    return plan.g_bytes + max(phases)


def oom_report(cfg: RunConfig, resolved, dev_mb: float = _OOM_T4_BUDGET) -> dict:
    """
    Full per-device memory breakdown for the resolved run: chain-rule tape,
    eigenbasis, L-BFGS state, fem_nodal residual live set, and the ENGD
    Gramian/Jacobian, against the device budget.
    """
    meta = resolved.case_meta
    nn, ncl = meta["n_nodes"], meta["n_cells"]
    data_dim, model_dim = resolved.mesh_shape
    n_eig_eff = resolved.dim_in - 1
    n_pde = resolved.batches.get("pde", 0)
    ad_factor = 1.7 if cfg.precision_policy == "selective_f64" else 1.0
    p = resolved.param_count

    rep = oom_predict(n_eig_eff, cfg.m_width if cfg.architecture is Architecture.DGM else max(cfg.mlp_hidden),
                      max(n_pde, 1), data_dim, model_dim, dev_mb, nn, ncl, cfg.n_blocks, ad_factor)

    opt_bytes = 8 if cfg.opt_f64 else 4
    lbfgs_mb = ((4 + 2 * cfg.lbfgs_mem) * p * opt_bytes / model_dim / 1e6
                if cfg.special_opt is SpecialOpt.LBFGS else 0.0)

    fem_mb = 0.0
    if cfg.backprop_design is BackpropDesign.FEM_NODAL:
        fem_rows_dev = -(-nn // data_dim)
        fem_rows_live = fem_rows_dev if cfg.fem_chunk <= 0 else min(cfg.fem_chunk, fem_rows_dev)
        fem_prec_b = 8 if cfg.precision_policy == "selective_f64" else 4
        width = cfg.m_width if cfg.architecture is Architecture.DGM else max(cfg.mlp_hidden)
        fem_mb = (fem_rows_live * _FEM_TAPE_C * cfg.n_blocks * (width / model_dim) * 3 * 4
                  + (ncl / data_dim) * 8 * 64 * 4 * fem_prec_b
                  + 24.0 * nn * 4) / 1e6

    engd_mb = 0.0
    if resolved.engd_plan is not None:
        engd_mb = engd_step_bytes(cfg, resolved) / 1e6

    total = rep["total"] + lbfgs_mb + fem_mb + engd_mb
    return dict(
        total=total, eig=rep["eig"], ad=rep["ad"], ctx=rep["ctx"],
        lbfgs=lbfgs_mb, fem=fem_mb, engd=engd_mb,
        dev_mb=dev_mb, util=100.0 * total / dev_mb, fits=total <= dev_mb,
    )


def max_m_width(cfg: RunConfig, resolved, dev_mb: float = _OOM_T4_BUDGET, step: int = 8) -> int:
    """Largest DGM hidden width the chain-rule tape estimate says fits the device."""
    meta = resolved.case_meta
    nn, ncl = meta["n_nodes"], meta["n_cells"]
    data_dim, model_dim = resolved.mesh_shape
    n_eig_eff = resolved.dim_in - 1
    n_pde = max(resolved.batches.get("pde", 0), 1)
    ad_factor = 1.7 if cfg.precision_policy == "selective_f64" else 1.0
    m = step
    while oom_predict(n_eig_eff, m, n_pde, data_dim, model_dim, dev_mb, nn, ncl,
                      cfg.n_blocks, ad_factor)["fits"]:
        m += step
    return m - step
