r"""
NTK-adaptive loss weighting over the active groups (port of the notebook's
NTK cells, with the previously-duplicated EMA/pin/boost block defined once).

Each group's empirical NTK trace is a proxy for its convergence speed
(Wang, Yu & Perdikaris 2022); equalizing the timescales gives

.. math::

    \lambda_g \;=\; \frac{\sum_{g'} \operatorname{tr} K_{g'}}{\operatorname{tr} K_g + \epsilon},
    \qquad
    \lambda \leftarrow \frac{|\mathcal{G}|\;\lambda}{\sum_g \lambda_g},
    \qquad
    w \leftarrow (1 - \beta)\, w + \beta\, \lambda

where:
- :math:`\operatorname{tr} K_g = \lVert J_g \rVert_F^2`: the Frobenius norm of group :math:`g`'s residual Jacobian on the probe subset.
- :math:`\beta`: the EMA rate (``ntk_ema``).
- the legacy surgery — pin the BC weight to zero and add ``ntk_pde_boost`` to the PDE weight after normalization — is now explicit config (``ntk_pin_bc_zero`` / ``ntk_pde_boost``), defaulting to the trained behavior.

Per-point groups (ic/data/bc and the chain_rule PDE) use the exact vmap trace;
the globally-coupled fem_nodal PDE goes through :func:`ntk_trace_jax.fem_trace`
(full/chunk/hutchinson/shrink) on a node probe rescaled by
:math:`n_{\mathrm{nodes}}/n_{\mathrm{kept}}`; the spectral PDE's coefficient
vector is small enough for the exact chunked trace with no rescale.
"""

from __future__ import annotations

from typing import Callable

from .config import BackpropDesign, ResidualDesign, RunConfig, WeightingDesign
from .casedata import CaseData
from .loss import Window
from .residuals import ResidualOps, Scales


def ntk_update(w, lam, cfg: RunConfig, groups: tuple[str, ...]):
    """Normalize the raw trace ratios, apply the pin/boost surgery, and EMA-blend.

    The ``reg`` group's weight is hard-pinned to 1 after normalization: prior
    strength is a modeling choice carried by the ``inv_beta_*`` settings, never
    an adaptive balance target.
    """
    import jax.numpy as jnp

    lam = lam * (float(len(groups)) / jnp.sum(lam))
    if cfg.ntk_pin_bc_zero and "bc" in groups:
        lam = lam.at[groups.index("bc")].set(0.0)
    if cfg.ntk_pde_boost != 0.0 and "pde" in groups:
        i = groups.index("pde")
        lam = lam.at[i].set(lam[i] + cfg.ntk_pde_boost)
    if "reg" in groups:
        lam = lam.at[groups.index("reg")].set(1.0)
    return (1.0 - cfg.ntk_ema) * w + cfg.ntk_ema * lam


def make_ntk(cfg: RunConfig, groups: tuple[str, ...], ops: ResidualOps, prim,
             scales: Scales, case: CaseData, centroids, n_nodes: int) -> Callable:
    """
    ``ntk_core(params, key, win, fem_nodes) -> (lam, tr)`` — per-group trace
    ratios on the window's probe subsets (aligned with ``groups``).
    ``fem_nodes`` is the fem PDE node probe (ignored by the other designs).
    """
    import jax.numpy as jnp

    import utils.ntk_trace_jax as nt

    encoder = prim.encoder
    rs_inv = 1.0 / scales.res_scale
    state_scale = scales.state_scale

    def _xt_of(ci, t):
        return jnp.concatenate([centroids[ci], jnp.reshape(t, (1,))])

    # Per-point gathers: capture the encoder's underlying arrays once so the trace
    # maps stay scalar-indexed (nt.ntk_trace vmaps over scalar probe columns).
    _enc_arrays = encoder.cell_arrays()

    def _enc_of(c):
        return tuple(a[c] for a in _enc_arrays)

    def g_uvy(p, c, t, y):
        from .residuals import state_row_mask

        return (prim.primaries_point(p, _xt_of(c, t), *_enc_of(c)) - y) / state_scale * state_row_mask(cfg, y)

    def g_pde_cr(p, c, t):
        return ops.pde_point(p, _xt_of(c, t), *_enc_of(c)) * rs_inv

    def g_bc(p, c, a, t):
        return ops.flux_point(p, _xt_of(c, t), *_enc_of(c))[:, a] * (1.0 / scales.bc_scale)

    def _tr_pde(params, key, win: Window, fem_nodes):
        if cfg.residual_design is ResidualDesign.HYBRID_PDE:
            bal_p, bal_s = ops.pde_hybrid.balance

            def g_hyb(p):
                r_p, r_s = ops.pde_hybrid(p, win.t_p[0])
                r_s = r_s[fem_nodes] * rs_inv[None, 1:] * bal_s
                return jnp.concatenate([jnp.reshape(r_p * rs_inv[0] * bal_p, (-1,)),
                                        jnp.reshape(r_s, (-1,))]).astype(jnp.float32)

            n_out = int(cfg.n_eig) + 2 * int(fem_nodes.shape[0])
            return nt.fem_trace(g_hyb, params, n_out, cfg.ntk_trace,
                                chunk=cfg.ntk_chunk, key=key, n_probe=cfg.ntk_probes)
        if cfg.residual_design is ResidualDesign.SPECTRAL_PDE:
            cells = win.ci_p if cfg.backprop_design is BackpropDesign.CHAIN_RULE else None

            def g_spec(p):
                r = ops.pde_spectral(p, win.t_p[0], cells) * rs_inv[None, :]
                return r.reshape(-1).astype(jnp.float32)

            n_out = 3 * cfg.n_eig
            return nt.fem_trace(g_spec, params, n_out, cfg.ntk_trace,
                                chunk=cfg.ntk_chunk, key=key, n_probe=cfg.ntk_probes)
        if cfg.backprop_design is BackpropDesign.FEM_NODAL:
            def g_fem(p):
                r = ops.pde_fem(p, win.t_p[0])[fem_nodes] * rs_inv[None, :]
                return r.reshape(-1).astype(jnp.float32)

            n_out = 3 * fem_nodes.shape[0]
            tr = nt.fem_trace(g_fem, params, n_out, cfg.ntk_trace,
                              chunk=cfg.ntk_chunk, key=key, n_probe=cfg.ntk_probes)
            return tr * (n_nodes / fem_nodes.shape[0])
        return nt.ntk_trace(g_pde_cr, params, win.ci_p, win.t_p)

    def _tr_rows(arr_fn, params, key):
        """NTK trace of a deterministic whole-set row group (well / ctrl)."""
        def g_rows(p):
            return arr_fn(p).reshape(-1).astype(jnp.float32)

        import jax

        n_out = int(jax.eval_shape(g_rows, params).shape[0])
        return nt.fem_trace(g_rows, params, n_out, cfg.ntk_trace,
                            chunk=cfg.ntk_chunk, key=key, n_probe=cfg.ntk_probes)

    def _tr_well(params, key):
        return _tr_rows(ops.well_arr, params, key)

    def ntk_core(params, key, win: Window, fem_nodes):
        trs = []
        for g in groups:
            if g == "pde":
                trs.append(_tr_pde(params, key, win, fem_nodes))
            elif g == "ic":
                trs.append(nt.ntk_trace(g_uvy, params, win.ci_ic,
                                        jnp.zeros_like(win.ci_ic, jnp.float32),
                                        case.y_ic[win.ci_ic]))
            elif g == "data":
                trs.append(nt.ntk_trace(g_uvy, params, win.ci_d, win.t_d, win.y_d))
            elif g == "well":
                trs.append(_tr_well(params, key))
            elif g == "ctrl":
                trs.append(_tr_rows(ops.ctrl_arr, params, key))
            elif g == "bc":
                tb = jnp.full(win.bc_sel.shape, win.t_b, jnp.float32)
                trs.append(nt.ntk_trace(g_bc, params, case.bc_cell[win.bc_sel],
                                        case.bc_axis[win.bc_sel], tb))
        tr = jnp.array(trs)
        if "reg" in groups:
            # placeholder trace = the mean of the others, so reg barely skews the
            # normalization; ntk_update pins its weight to exactly 1 afterwards
            others = jnp.mean(tr) if tr.size else jnp.asarray(1.0, jnp.float32)
            tr = jnp.insert(tr, groups.index("reg"), others)
        lam = (jnp.sum(tr) / (tr + cfg.ntk_eps)).astype(jnp.float32)
        return lam, tr

    return ntk_core


def probe_window(cfg: RunConfig, resolved, win: Window) -> Window:
    """The window's leading NTK-probe subset per group (the hardest points under RAR)."""
    nb = resolved.ntk_batches
    slice_mode = (cfg.residual_design in (ResidualDesign.SPECTRAL_PDE, ResidualDesign.HYBRID_PDE)
                  or cfg.backprop_design is BackpropDesign.FEM_NODAL)
    n_pde = nb.get("pde", 1)
    return Window(
        ci_p=win.ci_p if slice_mode else win.ci_p[:n_pde],
        t_p=win.t_p if slice_mode else win.t_p[:n_pde],
        ci_d=win.ci_d[:nb.get("data", 1)],
        t_d=win.t_d[:nb.get("data", 1)],
        y_d=win.y_d[:nb.get("data", 1)],
        ci_ic=win.ci_ic[:nb.get("ic", 1)],
        bc_sel=win.bc_sel[:nb.get("bc", 1)],
        t_b=win.t_b,
    )
