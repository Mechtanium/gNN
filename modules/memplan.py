r"""
Memory planning: parameter counts, the per-device OOM estimate, and the ENGD
feasibility plan (extracted from the notebook's back-of-napkin OOM cell).

All estimates are pure Python (no allocation, no jax import) so they can run
at :func:`pinnlab.config.resolve` time, before any device buffer exists.

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
    ConfigError,
    RunConfig,
    dim_in_of,
    load_well_pack_meta,
)


def param_count(cfg: RunConfig, dim_in: int | None = None) -> int:
    """Exact trainable-parameter count of the DGM network."""
    d = dim_in_of(cfg) if dim_in is None else dim_in
    f = cfg.dim_out
    m = cfg.m_width
    return d * m + m + cfg.n_blocks * (4 * d * m + 4 * m * m + 4 * m) + m * f + f


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


def engd_row_count(
    cfg: RunConfig,
    groups: tuple[str, ...],
    batches: dict,
    case_meta: dict,
    full_batch: bool,
) -> int:
    """
    Upper-bound count of weighted residual rows per ENGD step.

    Uses the resolved window batch sizes (or the capped deterministic
    full-batch sizes when ``special_opt_after == 0``). Boundary-face counts are
    approximated from above by ``2 * n_cells``; :mod:`pinnlab.loss` owns the
    exact per-group row layout.
    """
    n_cells = case_meta["n_cells"]
    n_times = case_meta["n_times"]
    cap = cfg.full_batch_cap

    rows = 0
    for g in groups:
        if g == "pde":
            rows += cfg.n_eig * _PDE_COMPONENTS * cfg.n_tslice   # Galerkin-projected rows
        elif g == "ic":
            n_pts = min(n_cells, cap) if full_batch else batches["ic"]
            rows += n_pts * _STATE_COMPONENTS
        elif g == "data":
            n_pts = min(n_cells * n_times, cap) if full_batch else batches["data"]
            rows += n_pts * _STATE_COMPONENTS
        elif g == "well":
            meta = load_well_pack_meta(cfg)
            if meta is not None:
                rows += int(meta.get("n_rows_well", meta["n_rows"]))
            else:
                rows += _WELL_ROW_CHANNELS * _WELL_FALLBACK_WELLS * n_times
    return rows


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

    rep = oom_predict(n_eig_eff, cfg.m_width, max(n_pde, 1), data_dim, model_dim, dev_mb, nn, ncl,
                      cfg.n_blocks, ad_factor)

    lbfgs_mb = 0.0

    fem_rows_dev = -(-nn // data_dim)
    fem_rows_live = fem_rows_dev if cfg.fem_chunk <= 0 else min(cfg.fem_chunk, fem_rows_dev)
    fem_prec_b = 8 if cfg.precision_policy == "selective_f64" else 4
    width = cfg.m_width
    fem_mb = (fem_rows_live * _FEM_TAPE_C * cfg.n_blocks * (width / model_dim) * 3 * 4
              + (ncl / data_dim) * 8 * 64 * 4 * fem_prec_b
              + 24.0 * nn * 4) / 1e6

    engd_mb = 0.0
    if resolved.engd_plan is not None:
        plan = resolved.engd_plan
        if plan.mode == "dense":
            # P x P Gramian + (materialized J only in forward assembly) + seed block
            j_live = plan.j_bytes if plan.direction == "fwd" else 0
            engd_bytes = plan.g_bytes + j_live + plan.seed_bytes
        else:  # rowspace: J + the N x N row Gram and its Cholesky factor + seed block
            engd_bytes = plan.j_bytes + 2 * plan.a_bytes + plan.seed_bytes
        engd_mb = engd_bytes / 1e6

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
