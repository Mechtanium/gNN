r"""
Supervision-window sampling: random windows, the deterministic full batch
(special-optimizer-only runs), and residual-adaptive refinement (RAR).

RAR (port of the notebook's ``adapt_select``): every ``adapt_every``
iterations one data-sharded pass evaluates the whole system, ranks each loss
group's candidate pool by descending residual, and returns the hardest
:math:`N_g - N_g^{\mathrm{expl}}` positions plus an :math:`N_g^{\mathrm{expl}}`
random exploration reserve as the FIXED supervision set for the next window;
the hardest subset of each group also reseeds the NTK weights. The candidate
sweep is walked in ``sweep_chunk`` micro-batches per device so the
second-order-AD tape stays memory-bounded.

The PDE pool depends on the design: all cells × ``n_tcand`` random times for
the per-point chain_rule residual; ``n_tcand_fem`` candidate time slices scored
by the whole-mesh (or Galerkin-projected) residual for the slice-walking modes
(``fem_t_select`` = hardest | random).
"""

from __future__ import annotations

from math import prod as onp_prod
from typing import Callable

from .config import (BackpropDesign, ResidualDesign, RunConfig, Resolved,
                     SamplingDesign, WeightingDesign)
from .casedata import CaseData
from .loss import Window
from .meshenv import MeshEnv
from .residuals import ResidualOps, Scales, state_row_mask


def draw_times(case: CaseData, key, shape):
    r"""
    Uniform collocation times on the case's window --- the ONE conversion from a
    random draw to a time.

    .. math::

        t \;=\; t_{\mathrm{lo}} + u\,(t_{\mathrm{hi}} - t_{\mathrm{lo}}),
        \qquad u \sim \mathcal{U}(0, 1)

    where:
    - :math:`[t_{\mathrm{lo}}, t_{\mathrm{hi}}]`: the window from :meth:`~pinnlab.casedata.CaseData.t_span`, i.e. the whole history unless a train/test split narrowed it.
    - :math:`u`: the uniform variate of shape ``shape``.

    Every random time in this module routes through here (and every deterministic
    one through :func:`lin_times`) so a split cannot be half-applied: were a call
    site to scale by the case's full horizon directly it would sample the held-out
    window while every other group respected the split, leaking physics supervision
    with no visible symptom. ``tests/test_time_split.py`` asserts that no such
    direct horizon reference survives anywhere in this module.
    """
    import jax.numpy as jnp
    from jax import random

    strata = case.time_strata(int(onp_prod(shape)) if len(shape) else 1)
    if len(strata) == 1 or len(shape) != 1:
        lo, hi = case.t_span()
        return lo + random.uniform(key, shape, dtype=jnp.float32) * (hi - lo)
    # joint history/forecast split: each sub-window draws its own share of slices,
    # so a short forecast window is never left without a collocation slice
    keys = random.split(key, len(strata))
    parts = [lo + random.uniform(k, (n,), dtype=jnp.float32) * (hi - lo)
             for k, (lo, hi, n) in zip(keys, strata)]
    return jnp.concatenate(parts)


def lin_times(case: CaseData, n: int, layout: str = "linspace"):
    r"""
    Evenly spaced collocation times spanning the case's window (see :func:`draw_times`).

    ``layout="linspace"`` includes both ends, :math:`t_k = t_{\mathrm{lo}} + k\,\Delta t`
    with :math:`\Delta t = (t_{\mathrm{hi}} - t_{\mathrm{lo}})/(n - 1)`; ``"midpoint"``
    centres the slices in :math:`n` equal bins,

    .. math::

        t_k \;=\; t_{\mathrm{lo}} + \bigl(k + \tfrac{1}{2}\bigr)\,\frac{t_{\mathrm{hi}} - t_{\mathrm{lo}}}{n},
        \qquad k = 0, \dots, n - 1

    where:
    - :math:`t_{\mathrm{lo}}, t_{\mathrm{hi}}`: the collocation window of :meth:`~pinnlab.casedata.CaseData.t_span`.
    - the midpoint layout keeps every slice off :math:`t = 0`, where the balance is impulsive (the source switches on against a hydrostatic state, :math:`\partial_t U` unbounded) and the initial condition is already the ``ic`` group's job; under a warped time channel (``time_encoding="log"``) the :math:`t = 0` slice would also carry a time-derivative sensitivity :math:`d\tau/dt` orders of magnitude above every other slice.
    """
    import jax.numpy as jnp
    import numpy as onp

    def _grid(lo, hi, k, drop_first=False):
        if layout == "midpoint":
            return lo + (onp.arange(k) + 0.5) * (hi - lo) / k
        return onp.linspace(lo, hi, k + 1)[1:] if drop_first else onp.linspace(lo, hi, k)

    strata = case.time_strata(int(n))
    if len(strata) == 1:
        lo, hi = case.t_span()
        return jnp.asarray(_grid(lo, hi, n), jnp.float32)
    parts = [_grid(lo, hi, k, drop_first=bool(i)) for i, (lo, hi, k) in enumerate(strata)]
    return jnp.asarray(onp.concatenate(parts), jnp.float32)


def mid_time(case: CaseData):
    """The midpoint of the case's collocation window (see :func:`draw_times`)."""
    import jax.numpy as jnp

    lo, hi = case.t_span()
    return jnp.float32(0.5 * (lo + hi))


def _slice_mode(cfg: RunConfig) -> bool:
    return (cfg.residual_design in (ResidualDesign.SPECTRAL_PDE, ResidualDesign.HYBRID_PDE)
            or cfg.backprop_design is BackpropDesign.FEM_NODAL)


def _needs_quadrature_cells(cfg: RunConfig) -> bool:
    return (cfg.residual_design is ResidualDesign.SPECTRAL_PDE
            and cfg.backprop_design is BackpropDesign.CHAIN_RULE)


def _place(cfg: RunConfig, env: MeshEnv, win: Window) -> Window:
    """Device placement: PDE slices replicated in slice modes, everything else data-sharded."""
    import jax

    put = jax.device_put
    if _slice_mode(cfg):
        ci_p, t_p = put(win.ci_p, env.repl), put(win.t_p, env.repl)
    else:
        ci_p, t_p = put(win.ci_p, env.sh_data), put(win.t_p, env.sh_data)
    return Window(
        ci_p=ci_p, t_p=t_p,
        ci_d=put(win.ci_d, env.sh_data), t_d=put(win.t_d, env.sh_data),
        y_d=put(win.y_d, env.sh_data2),
        ci_ic=put(win.ci_ic, env.sh_data), bc_sel=put(win.bc_sel, env.sh_data),
        t_b=win.t_b,
    )


def random_window(cfg: RunConfig, case: CaseData, resolved: Resolved, env: MeshEnv, key) -> Window:
    """Random (non-adaptive) supervision sets for one window."""
    import jax.numpy as jnp
    from jax import random

    b = resolved.batches
    dd = env.data_dim
    # absent groups get data-dim-sized placeholders: never gathered by the loss,
    # but still explicitly device_put on the data axis by _place (must divide)
    n_pde = max(b.get("pde", 0), dd)
    ks = random.split(key, 6)
    if _slice_mode(cfg):
        t_p = draw_times(case, ks[0], (cfg.n_tslice,))
        if _needs_quadrature_cells(cfg):
            ci_p = random.randint(ks[5], (n_pde,), 0, case.n_cells, dtype=jnp.int32)
        else:
            ci_p = jnp.zeros((), jnp.int32)
    else:
        ci_p = random.randint(ks[0], (n_pde,), 0, case.n_cells, dtype=jnp.int32)
        t_p = draw_times(case, ks[5], (n_pde,))

    n_data = b.get("data", 0)
    pool = case.cell_idx_data.shape[0]
    if pool == 0:
        raise ValueError(
            "the `data` group has no supervised rows on this half: a case split with "
            "train_split_n=0 keeps no training steps, so nothing can supervise cell "
            "states. Build the inference bundle directly (observation=none) instead of "
            "a training bundle on the empty half."
        )
    idx = random.choice(ks[1], pool, (max(n_data, dd),), replace=False)
    ci_ic = random.randint(ks[2], (max(b.get("ic", 0), dd),), 0, case.n_cells, dtype=jnp.int32)
    bc_sel = random.randint(ks[3], (max(b.get("bc", 0), dd),), 0, int(case.bc_cell.shape[0]),
                            dtype=jnp.int32)
    t_b = draw_times(case, ks[4], ())
    win = Window(ci_p=ci_p, t_p=t_p,
                 ci_d=case.cell_idx_data[idx], t_d=case.time_data[idx], y_d=case.y_data[idx],
                 ci_ic=ci_ic, bc_sel=bc_sel, t_b=t_b)
    return _place(cfg, env, win)


def full_batch_window(cfg: RunConfig, case: CaseData, resolved: Resolved, env: MeshEnv) -> Window:
    """
    Deterministic full-batch supervision (``special_opt_after == 0`` + window
    sampling): strided cell/record selections and linspace time slices, each
    group capped at ``full_batch_cap`` rows and trimmed to a data-dim multiple.
    """
    import jax.numpy as jnp
    import numpy as onp

    dd = env.data_dim
    cap = cfg.full_batch_cap

    def _strided(n_total: int, n_want: int):
        n = min(n_total, n_want)
        n = max(dd, (n // dd) * dd)
        return jnp.asarray(onp.linspace(0, n_total - 1, n).round().astype(onp.int32))

    if _slice_mode(cfg):
        t_p = lin_times(case, cfg.n_tslice, cfg.tslice_layout)
        if _needs_quadrature_cells(cfg):
            ci_p = _strided(case.n_cells, cap)
        else:
            ci_p = jnp.zeros((), jnp.int32)
    else:
        n_pairs = min(case.n_cells * cfg.n_tslice, cap)
        cells = onp.asarray(_strided(case.n_cells, -(-n_pairs // cfg.n_tslice) * dd))
        ts = onp.asarray(lin_times(case, cfg.n_tslice, cfg.tslice_layout))
        ci = onp.tile(cells, cfg.n_tslice)
        tt = onp.repeat(ts, cells.shape[0])
        keep = max(dd, (min(ci.shape[0], cap) // dd) * dd)
        ci_p = jnp.asarray(ci[:keep], jnp.int32)
        t_p = jnp.asarray(tt[:keep], jnp.float32)

    idx = _strided(case.cell_idx_data.shape[0], cap)
    ci_ic = _strided(case.n_cells, cap)
    bc_sel = _strided(int(case.bc_cell.shape[0]), cap)
    t_b = mid_time(case)
    win = Window(ci_p=ci_p, t_p=t_p,
                 ci_d=case.cell_idx_data[idx], t_d=case.time_data[idx], y_d=case.y_data[idx],
                 ci_ic=ci_ic, bc_sel=bc_sel, t_b=t_b)
    return _place(cfg, env, win)


def make_adapt_select(cfg: RunConfig, groups: tuple[str, ...], resolved: Resolved,
                      ops: ResidualOps, prim, scales: Scales, ntk_core: Callable | None,
                      case: CaseData, centroids, env: MeshEnv, n_nodes: int,
                      extras=None) -> Callable:
    """
    ``adapt_select(params, w, key) -> (Window, w_new)`` — the jitted RAR sweep.
    Groups absent from the config are filled with size-1 placeholders; the
    ``well``/``reg`` groups are whole-set deterministic and never sampled. Under
    contact inversion the IC pool ranks against the parametric equilibrium
    target (the same surface the loss trains on).
    """
    import functools

    import jax
    import jax.numpy as jnp
    from jax import random
    from jax.sharding import PartitionSpec as P

    from .weighting import ntk_update, probe_window

    b = resolved.batches
    data_dim = env.data_dim
    rs_inv = 1.0 / scales.res_scale
    bc_inv = None if scales.bc_scale is None else 1.0 / scales.bc_scale
    state_scale = scales.state_scale
    encoder = prim.encoder

    _enc_arrays = encoder.cell_arrays()

    def _enc_of(c):
        return tuple(a[c] for a in _enc_arrays)

    def _xt_of(ci, t):
        return jnp.concatenate([centroids[ci], jnp.reshape(t, (1,))])

    def _pde_point_loss(params, c, t):
        R = ops.pde_point(params, _xt_of(c, t), *_enc_of(c))
        return jnp.sum((R * rs_inv) ** 2)

    equil = extras.invm.equil_ic if (extras is not None and extras.invm is not None) else None

    def _ic_point_loss(params, c):
        S = prim.primaries_point(params, _xt_of(c, 0.0), *_enc_of(c))
        target = equil(params, c) if equil is not None else case.y_ic[c]
        return jnp.sum(((S - target) / state_scale * state_row_mask(cfg, case.y_ic[c])) ** 2)

    def _data_point_loss(params, r):
        c = case.cell_idx_data[r]
        S = prim.primaries_point(params, _xt_of(c, case.time_data[r]), *_enc_of(c))
        y = case.y_data[r]
        return jnp.sum(((S - y) / state_scale * state_row_mask(cfg, y)) ** 2)

    def _bc_point_loss(params, f, t_b):
        c = case.bc_cell[f]
        a = case.bc_axis[f]
        F = ops.flux_point(params, _xt_of(c, t_b), *_enc_of(c))[:, a]
        return jnp.sum((F * bc_inv) ** 2)

    def _sweep_losses(point_loss, cols):
        """Per-candidate scalar loss, data-sharded + sweep_chunk micro-batched."""
        pool = cols[0].shape[0]
        pad = (-pool) % data_dim
        p_tot = pool + pad
        per_dev = p_tot // data_dim
        pcols = tuple(jnp.concatenate([c, jnp.zeros((pad,), c.dtype)]) if pad else c for c in cols)
        rcols = tuple(jax.lax.with_sharding_constraint(
            c.reshape(data_dim, per_dev), env.nd(P("data", None))) for c in pcols)

        def device_row(*row):
            return jax.lax.map(lambda a: point_loss(*a), tuple(row),
                               batch_size=min(cfg.sweep_chunk, per_dev))

        val = jax.vmap(device_row)(*rcols).reshape(p_tot)
        val = jax.lax.with_sharding_constraint(val, env.repl)
        if pad:
            val = val.at[pool:].set(-jnp.inf)
        return val, pool

    def _select(val, N, n_explore, key, pool):
        n_hard = N - n_explore
        _, hard = jax.lax.top_k(val, n_hard)
        if n_explore > 0:
            expl = random.choice(key, pool, (n_explore,), replace=False).astype(hard.dtype)
            return jnp.concatenate([hard, expl])
        return hard

    n_expl = {g: int(round(cfg.rar_explore_frac * b[g])) for g in b}
    n_expl_tslice = int(round(cfg.rar_explore_frac * cfg.n_tslice))
    slice_mode = _slice_mode(cfg)
    quad_cells = _needs_quadrature_cells(cfg)
    use_ntk = cfg.weighting is WeightingDesign.NTK and ntk_core is not None

    def _slice_score_fn(params, cells):
        if cfg.residual_design is ResidualDesign.SPECTRAL_PDE:
            return lambda ts: jnp.mean(
                (ops.pde_spectral(params, ts, cells if quad_cells else None) * rs_inv) ** 2
            ).astype(jnp.float32)
        if cfg.residual_design is ResidualDesign.HYBRID_PDE:
            def _score(ts):
                r_p, r_s = ops.pde_hybrid(params, ts)
                return (0.5 * jnp.mean((r_p * rs_inv[0]) ** 2)
                        + 0.5 * jnp.mean((r_s * rs_inv[1:]) ** 2)).astype(jnp.float32)
            return _score
        return lambda ts: jnp.mean((ops.pde_fem(params, ts) * rs_inv) ** 2).astype(jnp.float32)

    @functools.partial(jax.jit)
    def adapt_select(params, w, key):
        k = random.split(key, 7)
        t_b = draw_times(case, k[1], ())

        # ---- PDE pool ------------------------------------------------------------------
        ci_p = jnp.zeros((), jnp.int32)
        t_p = jnp.zeros((1,), jnp.float32)
        fem_nodes = jnp.zeros((1,), jnp.int32)
        if "pde" in groups:
            if slice_mode:
                kf = random.split(k[0], 4)
                if quad_cells:
                    ci_p = random.randint(kf[2], (b["pde"],), 0, case.n_cells, dtype=jnp.int32)
                if cfg.fem_t_select == "hardest":
                    cand_ts = draw_times(case, kf[0], (cfg.n_tcand_fem,))
                    l_t = jax.lax.map(_slice_score_fn(params, ci_p if quad_cells else None), cand_ts)
                    sel_t = _select(l_t, cfg.n_tslice, n_expl_tslice, k[2], cfg.n_tcand_fem)
                    t_p = cand_ts[sel_t]
                else:
                    t_p = draw_times(case, kf[0], (cfg.n_tslice,))
                n_probe = min(cfg.n_ntk_fem if cfg.ntk_trace == "shrink"
                              else resolved.ntk_batches.get("pde", 1), n_nodes)
                fem_nodes = random.choice(kf[1], n_nodes, (n_probe,), replace=False)
            else:
                t_slices = draw_times(case, k[0], (cfg.n_tcand,))
                cand_ci = jnp.tile(jnp.arange(case.n_cells, dtype=jnp.int32), cfg.n_tcand)
                cand_t = jnp.repeat(t_slices, case.n_cells)
                l_pde, n_pool = _sweep_losses(lambda c, t: _pde_point_loss(params, c, t),
                                              (cand_ci, cand_t))
                sel = _select(l_pde, b["pde"], n_expl["pde"], k[2], n_pool)
                ci_p, t_p = cand_ci[sel], cand_t[sel]

        # ---- IC / BC / data pools -------------------------------------------------------
        ci_ic = jnp.zeros((1,), jnp.int32)
        if "ic" in groups:
            all_c = jnp.arange(case.n_cells, dtype=jnp.int32)
            l_ic, n_pool = _sweep_losses(lambda c: _ic_point_loss(params, c), (all_c,))
            ci_ic = all_c[_select(l_ic, b["ic"], n_expl["ic"], k[3], n_pool)]

        bc_sel = jnp.zeros((1,), jnp.int32)
        if "bc" in groups:
            all_f = jnp.arange(int(case.bc_cell.shape[0]), dtype=jnp.int32)
            l_bc, n_pool = _sweep_losses(lambda f: _bc_point_loss(params, f, t_b), (all_f,))
            bc_sel = all_f[_select(l_bc, b["bc"], n_expl["bc"], k[4], n_pool)]

        if "data" in groups:
            all_r = jnp.arange(int(case.y_data.shape[0]), dtype=jnp.int32)
            l_dat, n_pool = _sweep_losses(lambda r: _data_point_loss(params, r), (all_r,))
            sel_dat = _select(l_dat, b["data"], n_expl["data"], k[5], n_pool)
            ci_d, t_d, y_d = (case.cell_idx_data[sel_dat], case.time_data[sel_dat],
                              case.y_data[sel_dat])
        else:
            dd = env.data_dim
            ci_d = jnp.zeros((dd,), jnp.int32)
            t_d = jnp.zeros((dd,), jnp.float32)
            y_d = jnp.zeros((dd, 4), jnp.float32)

        sh = lambda a: jax.lax.with_sharding_constraint(a, env.sh_data)
        sh2 = lambda a: jax.lax.with_sharding_constraint(a, env.sh_data2)
        win = Window(
            ci_p=ci_p if slice_mode else sh(ci_p),
            t_p=t_p if slice_mode else sh(t_p),
            ci_d=sh(ci_d), t_d=sh(t_d), y_d=sh2(y_d),
            ci_ic=sh(ci_ic), bc_sel=sh(bc_sel), t_b=t_b,
        )

        # ---- NTK reseed on the hardest subset of each group ------------------------------
        if use_ntk:
            lam, _tr = ntk_core(params, k[6], probe_window(cfg, resolved, win), fem_nodes)
            w_new = ntk_update(w, lam, cfg, groups)
        else:
            w_new = w
        return win, w_new

    return adapt_select
