r"""
A family of reservoirs taught by the simulator — the inversion of Section 5.4 of
``framework/8-Inversion-From-Well-Records.md``.

One network :math:`u_\theta(\mathbf x, t; \mathbf m)` represents the state of a reservoir for
every whitened property vector :math:`\mathbf m` in the prior box. It is trained on the cell
states of :math:`N` simulator runs at sampled properties, with the physics rows assembled at
each sample's own properties as the regularizer between the samples:

.. math::

    L(\theta) \;=\; \frac{1}{N}\sum_{i=1}^{N}\Bigl[\,
        \lVert u_\theta(\cdot;\mathbf m_i) - u^{\mathrm{sim}}_i \rVert^2
        + \lVert R[u_\theta(\cdot;\mathbf m_i);\mathbf m_i] \rVert^2
        + \lVert u_\theta(\cdot, 0;\mathbf m_i) - u^{\mathrm{sim}}_i(0) \rVert^2 \Bigr]

where:

- :math:`\mathbf m_i`: the whitened properties of sample :math:`i` (``samples.json``, written by
  :mod:`modules.flows.inversion`): :math:`k_v/k_h = e^{\sigma u_1}` times the deck's, the
  contacts shifted by :math:`u_2, u_3` prior standard deviations;
- :math:`u^{\mathrm{sim}}_i`: the simulator's pressures and saturations of run :math:`i` at every
  cell and report time (the data rows), and its initial state (the IC rows);
- :math:`R[\cdot;\mathbf m_i]`: the black-oil rows of :mod:`modules.residuals` with the sample's
  vertical permeability in the stiffness and the sample's realized well rates as the loads.

A point of the family is a *virtual cell* :math:`c + n_{\mathrm{cells}}\,s` — cell :math:`c` of
sample :math:`s` — so the data and IC rows of the single-reservoir method apply unchanged:
the encoder gathers the eigenfeatures with :math:`c` and the property channel with
:math:`s` (:class:`modules.encodings.SpectralEncoder`). Each PDE time slice belongs to one
sample (``Window.s_p``). The window is re-drawn with a shifted stride every few iterations
so that all :math:`N \times T \times n_{\mathrm{cells}}` points are seen.

The well record of any :math:`\mathbf m` is read from the network through the closed-form
inflow relation at the perforated cells under the schedule's requested controls with the
three-state switch (:func:`record_at`), and the inversion of a record is a search over
:math:`\mathbf m` through that map (:func:`invert_record`).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as onp

# --------------------------------------------------------------------------------------------
# the training set
# --------------------------------------------------------------------------------------------

def restart_states(stem: str) -> dict:
    """``{"pres", "swat", "sgas", "rs"}`` ``(T, n_cells)`` of a run's restart file, report steps in order."""
    from modules.utils.ecl_binary import read_arrays

    out = {"pres": [], "swat": [], "sgas": [], "rs": []}
    names = {"PRESSURE": "pres", "SWAT": "swat", "SGAS": "sgas", "RS": "rs"}
    for k, v in read_arrays(stem + ".UNRST"):
        k = k.strip()
        if k in names:
            out[names[k]].append(onp.asarray(v, onp.float32))
    return {k: onp.stack(v) for k, v in out.items()}


def summary_rates(stem: str, wells, times) -> onp.ndarray:
    """The realized rates of the run's wells at ``times`` in residual order (water, oil, gas),
    injection-positive — ``(T, n_wells, 3)`` from the summary's production and injection vectors."""
    from modules.utils import ecl_binary as eb

    s = eb.summary(stem + ".SMSPEC")
    t_s = onp.asarray(s["time"], onp.float64)

    def vec(kw, w):
        try:
            return onp.interp(times, t_s, onp.asarray(eb.vector(s, kw, w), onp.float64))
        except KeyError:
            return onp.zeros(len(times))

    q = onp.zeros((len(times), len(wells), 3))
    for j, w in enumerate(wells):
        q[:, j, 0] = vec("WWIR", w) - vec("WWPR", w)
        q[:, j, 1] = vec("WOIR", w) - vec("WOPR", w)
        q[:, j, 2] = vec("WGIR", w) - vec("WGPR", w)
    return q


def summary_record(stem: str, wells, times) -> dict:
    """The well record of a run at ``times``: ``bhp`` ``(T, W)`` and the produced rates
    ``q`` ``(T, W, 3)`` in canonical order (oil, water, gas), produced positive — the order
    :func:`record_at` returns."""
    from modules.utils import ecl_binary as eb

    s = eb.summary(stem + ".SMSPEC")
    t_s = onp.asarray(s["time"], onp.float64)
    bhp = onp.stack([onp.interp(times, t_s, onp.asarray(eb.vector(s, "WBHP", w), onp.float64)) for w in wells], 1)
    return {"bhp": bhp, "q": -summary_rates(stem, wells, times)[:, :, [1, 0, 2]]}


@dataclass
class FamilyData:
    """The simulator's runs of the family: ``samples`` (the records of ``samples.json``),
    ``m`` ``(N, n_m)`` whitened properties, ``states`` ``(N, T, n_cells, 4)`` on the host,
    ``rates`` ``(N, T, n_wells, 3)`` realized rates (residual order, injection-positive) and
    ``records`` (``bhp``, ``q`` per sample) at the case's report times."""

    root: Path
    samples: list
    m: onp.ndarray
    states: onp.ndarray
    rates: onp.ndarray
    records: list
    held: "FamilyData | None" = None


def load_family(root: str | Path, case, wells, held_out: bool = False, limit: int | None = None) -> FamilyData:
    """The training (or held-out) samples of ``root`` whose simulator runs finished, with their
    states read from the restart files and their rates from the summaries at the case's
    report times. ``samples.json`` lists the samples; each record's ``stem`` is the path of
    its run's result files, without extension, relative to ``root``."""
    root = Path(root)
    meta = json.loads((root / "samples.json").read_text())
    times = onp.asarray(case.times, onp.float64)
    picked, ms, states, rates, records = [], [], [], [], []
    for smp in meta["samples"]:
        if bool(smp["held_out"]) != held_out:
            continue
        stem = str(root / smp["stem"])
        if not Path(stem + ".UNRST").exists():
            continue
        st = restart_states(stem)
        if st["pres"].shape[0] != times.size:
            raise ValueError(f"{smp['name']}: {st['pres'].shape[0]} report steps against the case's {times.size}")
        picked.append(smp)
        ms.append(smp["u"])
        states.append(onp.stack([st["pres"], st["swat"], st["sgas"], st["rs"]], -1))
        rates.append(summary_rates(stem, wells, times))
        records.append(summary_record(stem, wells, times))
        if limit is not None and len(picked) >= limit:
            break
    if not picked:
        raise FileNotFoundError(f"no finished {'held-out' if held_out else 'training'} samples under {root}")
    return FamilyData(root=root, samples=picked, m=onp.asarray(ms, onp.float32),
                      states=onp.stack(states).astype(onp.float32), rates=onp.stack(rates), records=records)


# --------------------------------------------------------------------------------------------
# what the physics rows need per sample
# --------------------------------------------------------------------------------------------

@dataclass
class FamilyTables:
    """Device arrays the rows of one sample are assembled from: ``m_table`` ``(N, n_m)``,
    ``kv_factor`` ``(N,)``, ``q_perf_all`` ``(N, T, n_perf, 3)``, the base forcing's geometry
    and the deck's ``k_hex_diag`` ``(n_hex, 3)``."""

    m_table: Any
    kv_factor: Any
    q_perf_all: Any
    times: Any
    perf_xyz: Any
    perf_sigma: Any
    k_hex_diag: Any

    def m(self, s):
        return self.m_table[s]

    def k_override(self, s):
        import jax.numpy as jnp

        f = self.kv_factor[s]
        return self.k_hex_diag * jnp.stack([jnp.ones_like(f), jnp.ones_like(f), f])[None, :]

    def forcing(self, s):
        from .wells import WellForcing

        return WellForcing(times=self.times, q_perf=self.q_perf_all[s], perf_xyz=self.perf_xyz,
                           perf_sigma=self.perf_sigma)


def family_tables(cfg, case, data: FamilyData) -> FamilyTables:
    """The per-sample tables from the training data: the loads are each sample's realized
    rates shared over its perforations by well index, as :func:`modules.wells.build_forcing`
    shares the deck's own."""
    import jax.numpy as jnp

    from . import spectral, wells

    pack = wells.build_well_pack(cfg, case)
    base = wells.build_forcing(pack, case)
    wi_sum = onp.zeros(pack.n_wells)
    onp.add.at(wi_sum, pack.well_id, pack.wi)
    wi_frac = pack.wi / onp.maximum(wi_sum[pack.well_id], 1e-30)
    q_perf = data.rates[:, :, pack.well_id, :] * wi_frac[None, None, :, None]       # (N, T, n_perf, 3)
    spec = spectral.provision(case, cfg)
    kv = onp.exp(float(cfg.inv.prior_sd) * data.m[:, 0])
    return FamilyTables(m_table=jnp.asarray(data.m, jnp.float32), kv_factor=jnp.asarray(kv, jnp.float32),
                        q_perf_all=jnp.asarray(q_perf, jnp.float32), times=base.times,
                        perf_xyz=base.perf_xyz, perf_sigma=base.perf_sigma,
                        k_hex_diag=jnp.asarray(spec.k_hex_diag, jnp.float32))


def family_case(case, data: FamilyData):
    """The case with the family's initial states stacked as its ``y_ic`` ``(N n_cells, 4)``, so
    the IC rows of a virtual cell read its sample's initial state."""
    import jax.numpy as jnp

    y_ic = data.states[:, 0].reshape(-1, 4)
    return dataclasses.replace(case, y_ic=jnp.asarray(y_ic, jnp.float32))


# --------------------------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------------------------

def near_well_cells(case, radius_cells: float = 3.5, perf=None):
    r"""
    The cells within ``radius_cells`` cell widths of a perforated column, and the perforated
    cells themselves: ``(near, perf)``, two sorted index arrays (``perf`` given, or read from
    the schedule).

    A cell :math:`c` with centroid :math:`(x_c, y_c)` is near a well when its horizontal
    distance to the closest perforated cell :math:`p` is within the radius,

    .. math::

        \min_{p}\;\sqrt{(x_c - x_p)^2 + (y_c - y_p)^2} \;\le\; \rho\, h ,

    where:

    - :math:`\rho`: ``radius_cells``, the radius in cell widths;
    - :math:`h`: the median horizontal cell width of the grid (``case.cell_len``);
    - :math:`p`: any perforated cell of the schedule (every well, every step).
    """
    if perf is None:
        from .wells import _repack_steps

        perf = onp.asarray(_repack_steps(case)["perf_cell_idx"], onp.int64)
    perf = onp.unique(onp.asarray(perf, onp.int64))
    xy = onp.asarray(case.centroids[:, :2], onp.float64)
    h = float(onp.median(onp.asarray(case.cell_len[:, :2], onp.float64)))
    d = onp.min(onp.linalg.norm(xy[:, None, :] - xy[perf][None, :, :], axis=-1), axis=1)
    return onp.nonzero(d <= float(radius_cells) * h + 1e-6)[0], perf


def _coprime_stride(total: int) -> int:
    """The smallest odd number from 7919 up that is coprime to ``total``, so that
    :math:`i \\mapsto a\\,i \\bmod \\mathrm{total}` is a permutation of the indices."""
    from math import gcd

    a = 7919
    while gcd(a, total) != 1:
        a += 2
    return a


def _strided(total: int, n_want: int, dd: int, shift: int) -> onp.ndarray:
    r"""
    ``n_want`` indices (rounded to a multiple of ``dd``) of a set of ``total``: window ``shift``
    of a walk through the set in a scattered order,

    .. math::

        \mathcal{I}_{\mathrm{shift}} = \{\, a\, i \bmod \mathrm{total} \;:\; i = \mathrm{shift}\, n, \dots, \mathrm{shift}\, n + n - 1 \,\}

    where:

    - :math:`a`: a stride coprime to ``total`` (:func:`_coprime_stride`), so that the map is a
      permutation: a window never repeats an index, successive windows are disjoint until the
      walk wraps after :math:`\lceil \mathrm{total} / n \rceil` redraws, and every window is
      spread over the whole set — over every sample of a family, in particular, rather than a
      contiguous run of them;
    - :math:`n`: the window's size.
    """
    n = max(dd, (min(total, n_want) // dd) * dd)
    a = _coprime_stride(total)
    i = (int(shift) * n + onp.arange(n, dtype=onp.int64)) % total
    return (a * i) % total


def family_draw(N: int, T: int, n_cells: int, cap: int, dd: int, offset: int, sets: dict | None = None):
    """The ``(sample, time step, cell)`` triples of one data window: ``cap`` points in all, each
    block of ``sets`` (``name -> (cells, fraction)``) a strided draw over the family's triples
    restricted to its cells, the remainder over the whole grid."""
    def block(cells: onp.ndarray, n_want: int):
        k = int(cells.shape[0])
        idx = _strided(N * T * k, n_want, dd, offset)
        s, rem = idx // (T * k), idx % (T * k)
        return s, rem // k, onp.asarray(cells)[rem % k]

    parts, used = [], 0
    for cells, frac in (sets or {}).values():
        n = int(round(cap * frac))
        if n > 0 and len(cells) > 0:
            parts.append(block(cells, n))
            used += parts[-1][0].shape[0]
    parts.append(block(onp.arange(n_cells), cap - used))
    return tuple(onp.concatenate([p[i] for p in parts]) for i in range(3))


def family_window(cfg, case, env, data: FamilyData, offset: int = 0, mix: dict | None = None):
    r"""
    A deterministic full-batch window over (sample, time, cell): ``cap`` data points and
    ``cap`` IC points drawn with a stride through the family, shifted by ``offset``; the
    ``n_tslice`` PDE slices assigned to samples ``offset, offset + 1, ...`` modulo ``N``.

    ``mix`` concentrates the data points where the well record is read. With
    ``{"perf": f_p, "near": f_n, "near_radius": rho}`` the fraction :math:`f_p` of the data
    points is drawn from the perforated cells alone (over every sample and report time),
    the fraction :math:`f_n` from the cells within :math:`\rho` cell widths of a well
    (:func:`near_well_cells`) and the rest from the whole grid; each block is its own scattered
    walk through its (sample, time, cell) triples (:func:`_strided`), so that every window holds
    every sample of the family in every block. The
    default ``None`` draws every point from the whole grid, where the two perforated cells of a
    9,375-cell grid receive a few points per window and the cone at the well is never resolved.
    """
    import jax.numpy as jnp

    from .loss import Window
    from .sampling import _place, lin_times, mid_time

    N, T, n_cells, _ = data.states.shape
    dd = env.data_dim
    cap = int(cfg.full_batch_cap)
    sets = None
    if mix:
        near, perf = near_well_cells(case, float(mix.get("near_radius", 3.5)))
        sets = {"perf": (perf, float(mix.get("perf", 0.0))), "near": (near, float(mix.get("near", 0.0)))}
    s, ti, c = family_draw(N, T, n_cells, cap, dd, offset, sets)
    times = onp.asarray(case.times, onp.float32)
    ci_d = jnp.asarray(c + n_cells * s, jnp.int32)
    t_d = jnp.asarray(times[ti], jnp.float32)
    y_d = jnp.asarray(data.states[s, ti, c], jnp.float32)
    ic = _strided(N * n_cells, cap, dd, offset)
    ci_ic = jnp.asarray(ic, jnp.int32)
    t_p = lin_times(case, cfg.n_tslice, cfg.tslice_layout)
    s_p = jnp.asarray((int(offset) * cfg.n_tslice + onp.arange(cfg.n_tslice)) % N, jnp.int32)
    bc_sel = jnp.asarray(onp.linspace(0, max(int(case.bc_cell.shape[0]) - 1, 0), dd).round().astype(onp.int32))
    win = Window(ci_p=jnp.zeros((), jnp.int32), t_p=t_p, ci_d=ci_d, t_d=t_d, y_d=y_d, ci_ic=ci_ic,
                 bc_sel=bc_sel, t_b=mid_time(case), s_p=s_p)
    return _place(cfg, env, win)


# --------------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------------

def build_family(cfg, case, data: FamilyData, **build_kwargs):
    """The pipeline of a family: the case with the stacked initial states, the encoder with
    the property channel, the rows with the per-sample tables."""
    from . import pipeline

    tables = family_tables(cfg, case, data)
    b = pipeline.build_pipeline(cfg, case=family_case(case, data), family=tables, **build_kwargs)
    return b, tables


def train_family(bundle, data: FamilyData, n_iter: int, redraw_every: int = 20, log_every: int = 10,
                 state=None, mix: dict | None = None):
    """Train the family for ``n_iter`` iterations, re-drawing the window every ``redraw_every``
    iterations (``mix`` as in :func:`family_window`); returns the trainer state and the list of
    run results of the chunks."""
    from . import training

    cfg = bundle.cfg
    state = training.init_trainer(bundle, resume=False) if state is None else state
    results, done, draw = [], 0, 0
    while done < n_iter:
        state.win = family_window(cfg, bundle.case, bundle.env, data, offset=draw, mix=mix)
        k = min(redraw_every, n_iter - done)
        results.append(training.train(bundle, state, n_iter=k, log_every=log_every))
        done += k
        draw += 1
    return state, results


# --------------------------------------------------------------------------------------------
# reading the well record of any property vector
# --------------------------------------------------------------------------------------------

def states_at(bundle, params, m, cells=None, times=None):
    """The network's primaries ``(T, n_c, 4)`` at ``cells`` (default all) and ``times`` (default
    the report times) for the whitened properties ``m``."""
    import jax
    import jax.numpy as jnp

    enc = bundle.prim.encoder
    cells = jnp.arange(bundle.case.n_cells, dtype=jnp.int32) if cells is None else jnp.asarray(cells, jnp.int32)
    times = jnp.asarray(bundle.case.times if times is None else times, jnp.float32)
    m = jnp.asarray(m, jnp.float32)
    xyz = enc.centroids[cells]
    args = (enc.v_c[cells], enc.b_v[cells], xyz, jnp.broadcast_to(m[None, :], (cells.shape[0], m.shape[0])))
    if enc.y0_cells is not None:
        args = args + (enc.y0_cells[cells],)

    def at_t(t):
        xt = jnp.concatenate([xyz, jnp.full((xyz.shape[0], 1), t, xyz.dtype)], axis=1)
        return jax.vmap(lambda x, *a: bundle.prim.primaries_point(params, x, *a))(xt, *args)

    return jax.lax.map(at_t, times)


def record_at(bundle, params, m):
    r"""
    The well record the network implies for the properties ``m`` under the schedule's requested
    controls: at every report time the closed-form inflow relation eliminates the flowing
    pressure against the target rate; where that pressure lies above the well's limit the rate
    is the target, where it lies below the pressure is held at the limit and the rate is what the
    inflow gives, and a rate that would reverse is zero (the three-state switch of Section
    3.4.6f). Returns ``bhp`` ``(T, W)`` and the produced rates ``q`` ``(T, W, 3)`` in canonical
    order (oil, water, gas), produced positive.
    """
    import jax.numpy as jnp

    from . import wells

    pack = bundle.well_pack
    case = bundle.case
    P4 = states_at(bundle, params, m, cells=pack.cell_idx)                      # (T, n_perf, 4)
    state = wells._perf_state(P4, pack, case, case.tables, float(bundle.cfg.inv.kr_floor))
    target = onp.where(pack.is_rate > 0, pack.q_ctrl, 0.0)
    pack2 = dataclasses.replace(pack, q_pin=target, binding=onp.zeros_like(pack.binding),
                                is_rate=onp.ones_like(pack.is_rate))
    p_bh_hat, _, _ = wells.closed_form_head(state, pack2, case)                 # (T, W) at the target rate
    p_lim = jnp.asarray(pack.p_bh_ctrl, p_bh_hat.dtype)
    sgn = jnp.asarray(onp.where(pack.q_ctrl > 0, -1.0, 1.0), p_bh_hat.dtype)   # producers +1
    on_limit = (p_lim > 0) & (sgn * (p_bh_hat - p_lim) < 0)
    p_wf = jnp.where(on_limit, p_lim, p_bh_hat)
    q_hat, _ = wells.predicted_head(state, p_wf, pack, case)                     # injection-positive
    q_hat = jnp.where(sgn[..., None] > 0, jnp.minimum(q_hat, 0.0), jnp.maximum(q_hat, 0.0))   # no reversed flow
    return p_wf, -q_hat                                                          # produced positive


def misfit(bundle, params, m, record: dict, sigma: dict, flowing=None):
    """Half the squared, scaled misfit of the record the network implies for ``m`` against an
    observed one (``bhp``, ``q`` in canonical order, produced positive) plus the prior ``|m|^2 / 2``."""
    import jax.numpy as jnp

    bhp, q = record_at(bundle, params, m)
    d_bhp, d_q = jnp.asarray(record["bhp"], bhp.dtype), jnp.asarray(record["q"], q.dtype)
    qo, qw, qg = q[..., 0], q[..., 1], q[..., 2]
    dqo, dqw, dqg = d_q[..., 0], d_q[..., 1], d_q[..., 2]
    gor, d_gor = qg / jnp.maximum(qo, 1e-3), dqg / jnp.maximum(dqo, 1e-3)
    wct, d_wct = qw / jnp.maximum(qo + qw, 1e-3), dqw / jnp.maximum(dqo + dqw, 1e-3)
    w = jnp.ones_like(bhp) if flowing is None else jnp.asarray(flowing, bhp.dtype)
    r = jnp.concatenate([(w * (bhp - d_bhp) / sigma["bhp"]).ravel(), (w * (gor - d_gor) / sigma["gor"]).ravel(),
                         (w * (wct - d_wct) / sigma["wct"]).ravel()])
    m = jnp.asarray(m)
    return 0.5 * jnp.sum(r * r) + 0.5 * jnp.sum(m * m), r


def invert_record(bundle, params, record: dict, sigma: dict, flowing=None, starts=None, n_iter: int = 40,
                  mu0: float = 1.0, seed: int = 0, n_samples: int = 0):
    """Gauss–Newton inversion of a record through the family from several starts, with the
    Laplace covariance at the best minimum and, when ``n_samples > 0``, a random-walk sample
    of the posterior. Returns a dict with ``m``, ``sd``, ``misfit``, ``starts`` and ``samples``."""
    import jax
    import jax.numpy as jnp

    n_m = int(bundle.prim.encoder.m_table.shape[1])
    rng = onp.random.default_rng(seed)
    starts = rng.standard_normal((4, n_m)) if starts is None else onp.asarray(starts, onp.float64)

    def resid(m):
        return misfit(bundle, params, m, record, sigma, flowing)[1]

    def objective(m):
        return misfit(bundle, params, m, record, sigma, flowing)[0]

    jac = jax.jit(jax.jacfwd(resid))
    res_fn = jax.jit(resid)
    obj = jax.jit(objective)
    runs = []
    for m0 in starts:
        m, mu = jnp.asarray(m0, jnp.float32), mu0
        f = float(obj(m))
        for _ in range(n_iter):
            r, J = res_fn(m), jac(m)
            g = J.T @ r + m
            H = J.T @ J + jnp.eye(n_m, dtype=J.dtype)
            step = -jnp.linalg.solve(H + mu * jnp.eye(n_m, dtype=J.dtype), g)
            m_new = m + jnp.clip(step, -1.0, 1.0)
            f_new = float(obj(m_new))
            if f_new < f:
                m, f, mu = m_new, f_new, max(mu / 3.0, 1e-4)
            else:
                mu *= 4.0
            if mu > 1e4:
                break
        J = jac(m)
        cov = onp.linalg.inv(onp.asarray(J.T @ J + jnp.eye(n_m, dtype=J.dtype), onp.float64))
        runs.append({"start": onp.asarray(m0).tolist(), "m": onp.asarray(m, onp.float64).tolist(), "misfit": f,
                     "sd": onp.sqrt(onp.diag(cov)).tolist()})
    best = min(runs, key=lambda r: r["misfit"])
    out = {"m": best["m"], "sd": best["sd"], "misfit": best["misfit"], "starts": runs, "samples": None}
    if n_samples > 0:                                 # a random-walk Metropolis chain on the misfit
        m = onp.asarray(best["m"], onp.float64)
        f = best["misfit"]
        scale = 0.5 * onp.asarray(best["sd"], onp.float64)
        chain, acc = [], 0
        for _ in range(n_samples):
            prop = m + scale * rng.standard_normal(n_m)
            f_prop = float(obj(jnp.asarray(prop, jnp.float32)))
            if onp.log(rng.random()) < f - f_prop:
                m, f, acc = prop, f_prop, acc + 1
            chain.append(m.tolist())
        out["samples"] = chain
        out["acceptance"] = acc / n_samples
    return out


# --------------------------------------------------------------------------------------------
# the campaign job
# --------------------------------------------------------------------------------------------

def score_sample(bundle, params, m, states, record, sigma=None, series: bool = False) -> dict:
    """The field and record errors of the network at ``m`` against one simulator run: the
    time-averaged RMSE of pressure and saturations over every cell, and the RMS errors of the
    bottom-hole pressure, gas-oil ratio and water cut over the times the well flows; with
    ``series`` the record itself (the first well's bottom-hole pressure, gas-oil ratio and water
    cut of the network and of the simulator at the report times) is returned under ``series``."""
    import jax.numpy as jnp

    st = onp.asarray(states_at(bundle, params, m), onp.float64)           # (T, n_cells, 4)
    truth = onp.asarray(states, onp.float64)
    out = {}
    for i, k in enumerate(("p", "sw", "sg")):
        out[f"{k}_rmse"] = float(onp.sqrt(onp.mean((st[..., i] - truth[..., i]) ** 2)))
    bhp, q = record_at(bundle, params, jnp.asarray(m, jnp.float32))
    bhp, q = onp.asarray(bhp, onp.float64), onp.asarray(q, onp.float64)
    d_bhp, d_q = onp.asarray(record["bhp"], onp.float64), onp.asarray(record["q"], onp.float64)
    flowing = d_q[..., 0] > 0.01 * onp.max(d_q[..., 0], axis=0, keepdims=True)
    gor = onp.clip(q[..., 2] / onp.maximum(q[..., 0], 1e-3), 0, 100); d_gor = d_q[..., 2] / onp.maximum(d_q[..., 0], 1e-3)
    wct = onp.clip(q[..., 1] / onp.maximum(q[..., 0] + q[..., 1], 1e-3), 0, 1)
    d_wct = d_q[..., 1] / onp.maximum(d_q[..., 0] + d_q[..., 1], 1e-3)
    rms = lambda e: float(onp.sqrt(onp.mean(e[flowing] ** 2)))            # noqa: E731
    out.update(bhp_rmse=rms(bhp - d_bhp), gor_rmse=rms(gor - d_gor), wct_rmse=rms(wct - d_wct))
    if series:
        out["series"] = {"times": onp.asarray(bundle.case.times, onp.float64).tolist(),
                         "bhp": bhp[:, 0].tolist(), "bhp_sim": d_bhp[:, 0].tolist(),
                         "gor": gor[:, 0].tolist(), "gor_sim": d_gor[:, 0].tolist(),
                         "wct": wct[:, 0].tolist(), "wct_sim": d_wct[:, 0].tolist(),
                         "flowing": flowing[:, 0].astype(int).tolist()}
    return out


def effective_sigma(sigma: dict, scores: dict | None) -> dict:
    r"""
    The standard deviations of the record misfit with the emulator's own error folded in.

    A family read through the well model misses the simulator's record by an amount that is
    not measurement noise but the emulator's error, and an inversion that ignores it fits that
    error as if it were signal and reports intervals far too narrow. Each channel's deviation
    is therefore

    .. math::

        \sigma_k = \sqrt{\sigma_{\mathrm{obs},k}^2 + \sigma_{\mathrm{mod},k}^2},

    where:

    - :math:`\sigma_{\mathrm{obs},k}`: the measurement noise assumed for channel :math:`k`
      (``sigma``: ``bhp`` in psi, ``gor`` in Mscf/stb, ``wct`` as a fraction);
    - :math:`\sigma_{\mathrm{mod},k}`: the emulator's root-mean-square error on that channel
      over the held-out runs (``scores``: the ``held`` summary of :func:`score_job`, keys
      ``bhp_rmse``, ``gor_rmse``, ``wct_rmse``); ``None`` leaves ``sigma`` as given.
    """
    if not scores:
        return dict(sigma)
    return {k: float(onp.sqrt(float(sigma[k]) ** 2 + float(scores.get(f"{k}_rmse", 0.0)) ** 2)) for k in sigma}


def calibrated_coverage(twins: list, z90: float = 1.6448536269514722) -> dict:
    r"""
    Calibrate the Laplace intervals of a set of twins on the twins themselves, leaving each one
    out of its own calibration.

    The Laplace standard deviation :math:`\sigma_{\mathrm{L},j}` of a recovered property treats
    the record's deviations as independent from one report time to the next, but the family's
    error is correlated in time, so the interval is too narrow by a factor the twins measure.
    For each property :math:`j` the standardized error of twin :math:`i` is
    :math:`z_{ij} = (u_{ij} - u^\ast_{ij}) / \sigma_{\mathrm{L},ij}`, the inflation is the robust
    scale of those errors,

    .. math::

        \kappa_j^{(-i)} \;=\; \frac{\operatorname{median}_{k \neq i} \lvert z_{kj} \rvert}{\Phi^{-1}(3/4)},
        \qquad
        \text{covered}_{ij} \;=\; \bigl[\, \lvert u_{ij} - u^\ast_{ij} \rvert \le z_{90}\, \kappa_j^{(-i)}\, \sigma_{\mathrm{L},ij} \,\bigr]

    where:

    - :math:`\kappa_j^{(-i)}`: the inflation of property :math:`j` from every twin but :math:`i`
      (leave-one-out, so that no twin calibrates its own interval); the median makes it robust to
      a twin outside the family's range;
    - :math:`\Phi^{-1}(3/4) \approx 0.6745`: the median of :math:`\lvert z \rvert` for a standard
      normal, so that :math:`\kappa = 1` means the Laplace width was right;
    - :math:`z_{90} = 1.645`: the half-width of a 90 % interval in standard deviations.

    Returns ``kappa`` (per property, from all twins), ``loo_cover`` (per twin, per property) and
    ``coverage_calibrated`` (the mean per property).
    """
    err = onp.asarray([t["error"] for t in twins], onp.float64)
    sd = onp.maximum(onp.asarray([t["sd"] for t in twins], onp.float64), 1e-12)
    z = onp.abs(err) / sd
    q = 0.6744897501960817
    n = z.shape[0]
    kappa = onp.median(z, axis=0) / q
    cover = onp.zeros_like(z, dtype=bool)
    for i in range(n):
        others = onp.delete(z, i, axis=0) if n > 1 else z
        k = onp.median(others, axis=0) / q
        cover[i] = onp.abs(err[i]) <= z90 * k * sd[i]
    return {"kappa": kappa.tolist(), "loo_cover": cover.tolist(), "coverage_calibrated": cover.mean(0).tolist()}
