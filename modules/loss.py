r"""
The group-structured training objective and its ENGD residual-row view.

The objective over the active loss groups :math:`g \in \mathcal{G}` (a config-
dependent subset of pde/ic/data/well/reg/bc) on one frozen supervision window is

.. math::

    \mathcal{L}(\theta)
    \;=\;
    \sum_{g}
    w_g\,
    \frac{1}{N_g}
    \sum_{k=1}^{N_g}
    \left(\frac{r_{gk}(\theta)}{s_g}\right)^{2}

where:
- :math:`r_{gk}`: the :math:`k`-th scalar residual of group :math:`g` (a PDE component, a normalized state misfit, a well-observable misfit, an inversion penalty row, or a boundary flux). Cell-state misfits carry the per-channel mask of :func:`pinnlab.residuals.state_row_mask` (``rs_supervision="oil_only"`` silences :math:`R_{so}` where the reference holds no oil).
- :math:`s_g`: the group's calibrated scale (:class:`~pinnlab.residuals.Scales`).
- :math:`w_g`: the group weight (fixed or NTK-adapted).
- :math:`N_g`: the group's scalar element count on the window.

:func:`make_residual_rows` exposes the identical objective as one weighted
residual vector — the ENGD weight-consistency contract:

.. math::

    \hat r_{gk} \;=\; \sqrt{\tfrac{w_g}{N_g}}\; \frac{r_{gk}}{s_g}
    \qquad\Longrightarrow\qquad
    \mathcal{L} \;=\; \lVert \hat r \rVert_2^2,
    \quad
    \nabla_\theta \mathcal{L} \;=\; 2\,\hat J^{\top} \hat r

where:
- :math:`\hat J = \partial\hat r/\partial\theta`: the row Jacobian assembled by :mod:`pinnlab.engd`.

Both surfaces are built from the same residual evaluations, so the identity
holds by construction (up to float32 accumulation in the reported components).
"""

from __future__ import annotations

from typing import Callable, NamedTuple

from .config import RunConfig
from .casedata import CaseData
from .residuals import Extras, ResidualOps, Scales, state_row_mask


class Window(NamedTuple):
    """One frozen supervision window (RAR-selected, random, or deterministic full batch).

    ``ci_p``/``t_p`` semantics depend on the residual design: paired collocation
    (cell, time) points for cartesian+chain_rule; ``t_p`` = whole-mesh time
    slices (``ci_p`` = quadrature cells for the spectral+chain_rule projection,
    placeholder otherwise) for the slice-walking modes.
    """

    ci_p: object
    t_p: object
    ci_d: object
    t_d: object
    y_d: object
    ci_ic: object
    bc_sel: object
    t_b: object


def _term_builders(cfg: RunConfig, groups, ops: ResidualOps, prim, scales: Scales,
                   case: CaseData, centroids, extras: Extras | None = None) -> dict:
    """Per-group residual-array builders shared by the loss and the rows view.

    Each builder maps ``(params, window) -> scaled residual array`` whose mean
    square is the group's loss term. The ``well`` group is a deterministic
    whole-set row block and ignores the window.
    """
    import jax
    import jax.numpy as jnp
    from jax import vmap

    encoder = prim.encoder
    rs_inv = 1.0 / scales.res_scale
    state_scale = scales.state_scale

    def _xt_of(ci, t):
        return jnp.concatenate([centroids[ci], jnp.reshape(t, (1,))])

    builders: dict = {}

    if "pde" in groups:
        def pde_arr(params, win: Window):
            _slice = jax.checkpoint(lambda ts: ops.pde_spectral(params, ts, None) * rs_inv)
            return jax.lax.map(_slice, win.t_p)                          # (S, n_eig, 3)
        builders["pde"] = pde_arr

    point_builders: dict = {}      # g -> (point_fn(params, win, i) -> (4,), n_points(win))

    if "ic" in groups:
        y_ic = case.y_ic

        def ic_arr(params, win: Window):
            enc_args = encoder.gather_args(win.ci_ic)
            Sic = vmap(lambda c, *a: prim.primaries_point(params, _xt_of(c, case.t_ic), *a))(
                win.ci_ic, *enc_args)
            target = y_ic[win.ci_ic]
            return (Sic - target) / state_scale * state_row_mask(cfg, y_ic[win.ci_ic])
        builders["ic"] = ic_arr

        def ic_point(params, win: Window, i):
            c = win.ci_ic[i]
            enc_args = encoder.gather_args(c)
            Sic = prim.primaries_point(params, _xt_of(c, case.t_ic), *enc_args)
            target = y_ic[c]
            return (Sic - target) / state_scale * state_row_mask(cfg, y_ic[c])
        point_builders["ic"] = (ic_point, lambda win: win.ci_ic.shape[0])

    if "data" in groups:
        def data_arr(params, win: Window):
            enc_args = encoder.gather_args(win.ci_d)
            Sd = vmap(lambda c, t, *a: prim.primaries_point(params, _xt_of(c, t), *a))(
                win.ci_d, win.t_d, *enc_args)
            return (Sd - win.y_d) / state_scale * state_row_mask(cfg, win.y_d)
        builders["data"] = data_arr

        def data_point(params, win: Window, i):
            c = win.ci_d[i]
            enc_args = encoder.gather_args(c)
            Sd = prim.primaries_point(params, _xt_of(c, win.t_d[i]), *enc_args)
            return (Sd - win.y_d[i]) / state_scale * state_row_mask(cfg, win.y_d[i])
        point_builders["data"] = (data_point, lambda win: win.ci_d.shape[0])

    if "well" in groups:
        builders["well"] = lambda params, win: ops.well_arr(params)

    builders["_point"] = point_builders
    return builders


def make_loss(cfg: RunConfig, groups, ops: ResidualOps, prim, scales: Scales,
              case: CaseData, centroids, extras: Extras | None = None) -> Callable:
    """
    ``loss(params, w, window) -> (total, comps)`` with ``comps`` a float32
    vector aligned with ``groups``. The weighted total keeps the natural
    (possibly promoted) dtype so the L-BFGS sufficient-decrease test resolves
    below the float32 epsilon of the objective.
    """
    import jax.numpy as jnp

    builders = _term_builders(cfg, groups, ops, prim, scales, case, centroids, extras)

    def loss(params, w, win: Window):
        terms = [jnp.mean(builders[g](params, win) ** 2) for g in groups]
        comps = jnp.array([jnp.asarray(t, jnp.float32) for t in terms], jnp.float32)
        total = sum(w[i] * terms[i] for i in range(len(groups)))
        return total, comps

    return loss


def make_residual_rows(cfg: RunConfig, groups, ops: ResidualOps, prim, scales: Scales,
                       case: CaseData, centroids, extras: Extras | None = None) -> Callable:
    """
    ``rows(params, w, window) -> r_hat`` — the float64 weighted residual vector
    with ``sum(r_hat**2) == loss(params, w, window)[0]`` by construction.

    The returned callable also carries ``rows.group_rows``: the per-group slices
    of the same vector as ``((name, rows_g, point_g), ...)`` in ``groups`` order,
    with ``concatenate([rows_g(...) for each]) == rows(...)`` exactly. The ENGD
    assembly (:func:`pinnlab.engd.gramian_grouped`) uses them to sweep each group
    in its own cheaper AD direction: a handful of projected PDE rows costs a few
    vjp sweeps, while the tens of thousands of cell-state rows cost :math:`P` jvp
    columns — neither would be affordable in the other's direction.

    ``point_g`` is ``None`` or a pair ``(point_fn, n_points)`` for the point-wise
    groups (``ic``, ``data``): ``point_fn(params, w, win, i)`` returns the weighted
    rows of supervision point ``i`` alone (four primaries, identical to the
    corresponding slice of ``rows_g``) and ``n_points(win)`` the point count. The
    Jacobian of such a group is a stack of independent :math:`4 \times P`
    per-point blocks, which reverse mode assembles at four vjp sweeps of a
    *single-point* graph each (:func:`pinnlab.engd.gramian_pointwise`) — a cost
    independent of :math:`P`, against :math:`P` whole-batch jvp columns.
    """
    import jax.numpy as jnp

    builders = _term_builders(cfg, groups, ops, prim, scales, case, centroids, extras)
    point_builders = builders["_point"]

    def _group_rows(i, g):
        def rows_g(params, w, win: Window):
            arr = builders[g](params, win)
            flat = jnp.reshape(jnp.asarray(arr, jnp.float64), (-1,))
            return jnp.sqrt(jnp.asarray(w[i], jnp.float64) / flat.shape[0]) * flat
        return rows_g

    def _point_rows(i, g):
        if g not in point_builders:
            return None
        point_fn, n_points = point_builders[g]

        def rows_point(params, w, win: Window, k):
            r = jnp.asarray(point_fn(params, win, k), jnp.float64)
            n_rows = n_points(win) * r.shape[0]
            return jnp.sqrt(jnp.asarray(w[i], jnp.float64) / n_rows) * r
        return rows_point, n_points

    group_rows = tuple((g, _group_rows(i, g), _point_rows(i, g)) for i, g in enumerate(groups))

    def rows(params, w, win: Window):
        return jnp.concatenate([fn(params, w, win) for _, fn, _ in group_rows])

    rows.group_rows = group_rows
    return rows
