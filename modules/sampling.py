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

from .config import RunConfig, Resolved
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
    return True      # the Galerkin residual is evaluated per collocation time slice


def _needs_quadrature_cells(cfg: RunConfig) -> bool:
    return False     # the spectral residual is assembled on the nodes, not on sampled cells


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


