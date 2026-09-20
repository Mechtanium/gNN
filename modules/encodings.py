r"""
Input encodings: Cartesian coordinates vs. Laplace-eigenfunction spectral features.

Both encoders expose the same three surfaces so the residual and evaluator
code is encoding-agnostic:

- ``feat_xt(xt, *args)`` — the per-point feature map, differentiable in the
  physical coordinates ``xt = (x, y, z, t)`` (the chain_rule residual takes
  :math:`\partial/\partial x` straight through it);
- ``gather_args(cells)`` — the batched per-cell static arguments consumed by
  ``feat_xt`` (empty for Cartesian);
- ``feat_nodes_t(t)`` — features for every mesh node at time ``t``
  (the fem_nodal evaluator's whole-mesh input).

The spectral encoding is the per-cell affine eigenfeature map

.. math::

    v(x) \;\approx\; v_c + B_v\,(x - x_c),
    \qquad
    \gamma(x, t) = \bigl[v(x),\; \tau(t)\bigr],
    \qquad
    \tau(t) = \tfrac{2t}{T_{\mathrm{end}}} - 1

where:
- :math:`v_c \in \mathbb{R}^{N_{\mathrm{eig}}}`: the eigenfeatures at the cell centroid :math:`x_c`.
- :math:`B_v \in \mathbb{R}^{3\times N_{\mathrm{eig}}}`: the FE physical gradients of the standardized modes, so autodiff w.r.t. :math:`x` recovers :math:`\nabla_x v` exactly.
- :math:`\tau`: the normalized time channel shared by both encodings.

The Cartesian encoding is the plain normalized-coordinate map
:math:`\gamma(x, t) = [\,2(x - x_{\min})/(x_{\max} - x_{\min}) - 1,\; \tau(t)\,]`.

**Time channel.** Both encoders share :math:`\tau(t)`; ``time_encoding="linear"``
(the default above) is uniform in :math:`t`, while ``"log"`` warps it,

.. math::

    \tau(t) \;=\; \frac{2\,\ln\!\bigl(1 + t/t_w\bigr)}{\ln\!\bigl(1 + T_{\mathrm{end}}/t_w\bigr)} - 1,
    \qquad
    t(\tau) \;=\; t_w\Bigl(\bigl(1 + T_{\mathrm{end}}/t_w\bigr)^{(\tau + 1)/2} - 1\Bigr)

where:
- :math:`t_w` (``time_warp_days``): the warp scale — linear below it, logarithmic above it, so a report schedule that refines its steps geometrically toward :math:`t = 0` (the pressure transient of a well start-up: SPE2 writes a hundred steps inside the first day of a 900-day history) is spread over the channel instead of being folded into :math:`\Delta\tau \sim 10^{-3}` next to the initial condition, where the network would need a :math:`10^{4}`-fold steeper slope than anywhere else.
- ``time_warp_days = 0``: :math:`t_w` is chosen from the case's report times as the scale that spreads the report steps most evenly in :math:`\tau` (:func:`resolve_time_warp`, the maximum geometric mean of :math:`\Delta\tau_i` over a log grid).
- :math:`t(\tau)`: the inverse, used by the hard-IC ramp of :mod:`pinnlab.physics` to read the physical time back off the feature vector.

**Near-well channel** (``well_encoding="logr"``, spectral encoder only). The retained
eigenmodes are global, smooth functions of position, so two cells a single width
apart carry nearly the same feature vector — while the pressure cone of a producing
well drops by hundreds of psi across exactly that width (SPE2: 330 psi between the
perforated cell and its neighbour at 900 days). Along a radial coordinate that cone
is the elementary log solution, :math:`p \approx a + b \ln r`, so the encoder appends
that coordinate as one more feature,

.. math::

    \phi_w(\mathbf{x}) \;=\; 2\,\frac{\ln \max(r_w(\mathbf{x}), r_0) - \ln r_0}{\ln r_{\max} - \ln r_0} - 1,
    \qquad
    r_w(\mathbf{x}) \;=\; \min_{p} \lVert \mathbf{x} - \mathbf{x}_p \rVert,
    \qquad
    \nabla \phi_w \;=\; \frac{2}{\ln r_{\max} - \ln r_0}\,\frac{\mathbf{x} - \mathbf{x}_{p^\ast}}{r_w^{2}}\;[r_w > r_0]

where:
- :math:`\mathbf{x}_p`: the perforated-cell centroids of every well of the case (:func:`perforation_xyz`), so the channel is the distance to the nearest well — one feature for any well count.
- :math:`r_0` (``well_enc_r0_ft``): the inner cutoff, defaulting to half the smallest edge of the perforated cells — the scale below which the discrete field cannot resolve the cone anyway, and small enough that the thin layers above and below a completion (SPE2: 8-ft layers under a 164-ft areal cell) keep distinct channel values; :math:`r_{\max}`: the largest cell-centroid distance, so the channel spans :math:`[-1, 1]` over the mesh.
- the gradient enters the affine per-cell map :math:`B_v` so the chain-rule residual differentiates through the channel like through any eigenfeature; nodal values feed the whole-mesh evaluator.

**Hard initial condition.** Under ``ic_design="hard"`` both encoders append the
raw (logit-space) initial state :math:`y^0(x) \in \mathbb{R}^4` of the evaluation
point to the feature vector; :func:`pinnlab.physics.make_primaries` strips it off
again and uses it as the offset of the ansatz :math:`y = y^0 + \beta(t)\,\mathcal{N}_\theta`.
Carrying :math:`y^0` inside the feature vector keeps every evaluator and call site
encoding-agnostic — the network itself still sees only the first ``dim_in`` entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import InputEncoding, RunConfig
from .casedata import CaseData
from .spectral import SpectralBundle


def resolve_time_warp(times, t_end: float, warp_days: float = 0.0) -> float:
    r"""
    The log-warp scale :math:`t_w` in days: ``warp_days`` verbatim when positive, else the
    scale on a log grid between the smallest report interval and :math:`T_{\mathrm{end}}`
    that makes the report steps most evenly spread in :math:`\tau` — the maximizer of the
    geometric mean of the step spacings,

    .. math::

        t_w^\ast \;=\; \operatorname*{arg\,max}_{t_w}\;
        \frac{1}{n_t - 1}\sum_{i} \ln \Delta\tau_i(t_w),
        \qquad
        \Delta\tau_i = \tau(t_{i+1}) - \tau(t_i)

    where:
    - :math:`t_i`: the case's report times; since :math:`\sum_i \Delta\tau_i = 2` is fixed, the geometric mean is largest when no step is squeezed toward zero — a schedule uniform in :math:`t` returns a large :math:`t_w` (the warp degenerates toward linear), a geometrically refined one a small :math:`t_w` (SPE2: :math:`t_w \approx 10^{-2}` d against a 900-day history, spreading its 101 first-day steps over 40% of the channel).
    """
    import numpy as onp

    if warp_days > 0:
        return float(warp_days)
    t = onp.asarray(times, onp.float64)
    dt = onp.diff(t)
    dt = dt[dt > 0]
    if dt.size < 2:
        return float(t_end)
    grid = onp.geomspace(max(dt.min(), 1e-9 * t_end), t_end, 200)
    best, best_score = float(t_end), -onp.inf
    for tw in grid:
        tau = onp.log1p(t / tw)
        d = onp.diff(tau)
        d = d[d > 0] * (2.0 / onp.log1p(t_end / tw))
        score = float(onp.log(d).mean())
        if score > best_score:
            best, best_score = float(tw), score
    return best


def _tau_of(t, t_end: float, t_warp):
    import jax.numpy as jnp

    if t_warp is None:
        return 2.0 * t / t_end - 1.0
    return 2.0 * jnp.log1p(t / t_warp) / jnp.log1p(t_end / t_warp) - 1.0


def _t_of_tau(tau, t_end: float, t_warp):
    import jax.numpy as jnp

    if t_warp is None:
        return 0.5 * (tau + 1.0) * t_end
    return t_warp * (jnp.expm1(0.5 * (tau + 1.0) * jnp.log1p(t_end / t_warp)))


@dataclass
class SpectralEncoder:
    """Δ-PINN positional encoding: standardized Laplace eigenfeatures + time."""

    v_c: Any
    b_v: Any
    centroids: Any
    v_nodes: Any
    t_end: float
    dim_in: int
    needs_eigenbasis: bool = True
    y0_cells: Any = None          # (n_cells, 4) raw IC offsets (hard IC) or None
    y0_nodes: Any = None          # (n_nodes, 4)
    t_warp: float | None = None   # log time channel scale [days]; None = linear

    def tau(self, t):
        return _tau_of(t, self.t_end, self.t_warp)

    def t_of_tau(self, tau):
        return _t_of_tau(tau, self.t_end, self.t_warp)

    def cell_arrays(self):
        """The per-cell static arrays ``gather_args`` indexes (for samplers that gather by hand)."""
        base = (self.v_c, self.b_v, self.centroids)
        return base + ((self.y0_cells,) if self.y0_cells is not None else ())

    def gather_args(self, cells):
        return tuple(a[cells] for a in self.cell_arrays())

    def feat_xt(self, xt, vc, Bv, xc, y0=None):
        import jax.numpy as jnp

        v = vc + (xt[:3] - xc) @ Bv
        parts = [v, jnp.reshape(self.tau(xt[3]), (1,))]
        if y0 is not None:
            parts.append(jnp.asarray(y0, v.dtype))
        return jnp.concatenate(parts)

    def feat_nodes_t(self, t):
        import jax.numpy as jnp

        tn = self.tau(t)
        col = jnp.full((self.v_nodes.shape[0], 1), tn, self.v_nodes.dtype)
        parts = [self.v_nodes, col]
        if self.y0_nodes is not None:
            parts.append(jnp.asarray(self.y0_nodes, self.v_nodes.dtype))
        return jnp.concatenate(parts, axis=1)


@dataclass
class CartesianEncoder:
    """Vanilla PINN positional encoding: bbox-normalized coordinates + time."""

    lo: Any
    hi: Any
    node_xyz: Any
    t_end: float
    dim_in: int = 4
    needs_eigenbasis: bool = False
    y0_cells: Any = None          # (n_cells, 4) raw IC offsets (hard IC) or None
    y0_nodes: Any = None          # (n_nodes, 4)
    t_warp: float | None = None   # log time channel scale [days]; None = linear

    def tau(self, t):
        return _tau_of(t, self.t_end, self.t_warp)

    def t_of_tau(self, tau):
        return _t_of_tau(tau, self.t_end, self.t_warp)

    def cell_arrays(self):
        return (self.y0_cells,) if self.y0_cells is not None else ()

    def gather_args(self, cells):
        return tuple(a[cells] for a in self.cell_arrays())

    def feat_xt(self, xt, y0=None):
        import jax.numpy as jnp

        xn = 2.0 * (xt[:3] - self.lo) / (self.hi - self.lo) - 1.0
        parts = [xn, jnp.reshape(self.tau(xt[3]), (1,))]
        if y0 is not None:
            parts.append(jnp.asarray(y0, xn.dtype))
        return jnp.concatenate(parts)

    def feat_nodes_t(self, t):
        import jax.numpy as jnp

        xn = 2.0 * (self.node_xyz - self.lo[None, :]) / (self.hi - self.lo)[None, :] - 1.0
        tn = self.tau(t)
        col = jnp.full((xn.shape[0], 1), tn, xn.dtype)
        parts = [xn, col]
        if self.y0_nodes is not None:
            parts.append(jnp.asarray(self.y0_nodes, xn.dtype))
        return jnp.concatenate(parts, axis=1)


def raw_ic_offsets(case: CaseData, spec: SpectralBundle | None, eps: float = 1e-4):
    r"""
    The logit-space image of the deck's initial condition, per cell and per node,
    for the hard-IC ansatz.

    .. math::

        y^0_1 = \sigma^{-1}\!\Bigl(\frac{p^0 - P_{\min}}{P_{\max} - P_{\min}}\Bigr),\quad
        y^0_2 = \sigma^{-1}\!\Bigl(\frac{S_w^0 - S_{wc}}{1 - S_{wc}}\Bigr),\quad
        y^0_3 = \sigma^{-1}\!\Bigl(\frac{S_g^0}{1 - S_w^0}\Bigr),\quad
        y^0_4 = \sigma^{-1}\!\Bigl(\frac{R_s^0}{R_{s,\max}}\Bigr)

    where:
    - every sigmoid argument is clamped to :math:`[\epsilon_\sigma, 1 - \epsilon_\sigma]` before the logit: :math:`S_g^0 = 0` (undersaturated cells) and :math:`S_w^0 = S_{wc}` have infinite logits, so the hard IC holds to within :math:`\epsilon_\sigma` of the range, not exactly.
    - nodal values are the volume-weighted vertex averages of the cell values (the same lumping that builds the nodal porosity), so the whole-mesh evaluator anchors on a consistent field.

    Returns ``(y0_cells (n_cells, 4), y0_nodes (n_nodes, 4) | None, n_clamped)``.
    """
    import jax.numpy as jnp
    import numpy as onp

    phys = case.phys
    y = onp.asarray(case.y_ic, onp.float64)
    p_min, p_max, swc, rs_max = float(phys.P_MIN), float(phys.P_MAX), float(phys.SWC), float(phys.RS_MAX)
    u = onp.stack([(y[:, 0] - p_min) / (p_max - p_min),
                   (y[:, 1] - swc) / max(1.0 - swc, 1e-12),
                   y[:, 2] / onp.maximum(1.0 - y[:, 1], 1e-12),
                   y[:, 3] / max(rs_max, 1e-12)], axis=1)
    n_clamped = int(((u < eps) | (u > 1.0 - eps)).sum())
    u = onp.clip(u, eps, 1.0 - eps)
    y0_cells = onp.log(u / (1.0 - u)).astype(onp.float32)
    y0_nodes = None
    if spec is not None:
        static = spec.static
        hex_nodes = onp.asarray(static["hex_nodes"])
        share = onp.asarray(jnp.einsum("eg,ega->ea", static["JxW"], static["N"]))   # (n_hex, 8)
        n_v = int(static["n_vertices"])
        acc = onp.zeros((n_v, 4)); vol = onp.zeros((n_v,))
        onp.add.at(acc, hex_nodes.reshape(-1), (share[:, :, None] * y0_cells[:, None, :]).reshape(-1, 4))
        onp.add.at(vol, hex_nodes.reshape(-1), share.reshape(-1))
        y0_nodes = (acc / onp.maximum(vol, 1e-30)[:, None]).astype(onp.float32)
    return jnp.asarray(y0_cells), (None if y0_nodes is None else jnp.asarray(y0_nodes)), n_clamped


def perforation_xyz(case: CaseData):
    """``(n_perf, 3)`` float64 centroids of every perforated cell of the case's wells."""
    import numpy as onp

    from .wells import _repack_steps

    seq = _repack_steps(case)
    cells = onp.asarray(seq["perf_cell_idx"], onp.int64)
    if cells.size == 0:
        raise ValueError(f"{case.art.reservoir_mesh.__class__.__name__}: no perforations to encode")
    return onp.asarray(case.centroids, onp.float64)[cells]


def well_log_radius(xyz, perf_xyz, r0: float, r_max: float):
    r"""
    The normalized near-well channel :math:`\phi_w` and its gradient at the points
    ``xyz`` (see the module docstring). Returns ``(phi (n,), grad (n, 3))`` in float32.
    """
    import numpy as onp

    x = onp.asarray(xyz, onp.float64)
    d = x[:, None, :] - onp.asarray(perf_xyz, onp.float64)[None, :, :]        # (n, n_perf, 3)
    dist = onp.sqrt((d ** 2).sum(-1))                                            # (n, n_perf)
    j = dist.argmin(axis=1)
    r = dist[onp.arange(x.shape[0]), j]
    dvec = d[onp.arange(x.shape[0]), j]                                          # (n, 3)
    span = onp.log(r_max) - onp.log(r0)
    phi = 2.0 * (onp.log(onp.maximum(r, r0)) - onp.log(r0)) / span - 1.0
    grad = onp.where((r > r0)[:, None], (2.0 / span) * dvec / onp.maximum(r, r0)[:, None] ** 2, 0.0)
    return phi.astype(onp.float32), grad.astype(onp.float32)


def _with_well_channel(cfg: RunConfig, case: CaseData, spec: SpectralBundle):
    """The spectral feature arrays with the near-well channel appended: ``(v_c, b_v, v_nodes)``."""
    import jax.numpy as jnp
    import numpy as onp

    from .wells import _repack_steps

    perf = perforation_xyz(case)
    cells = onp.asarray(_repack_steps(case)["perf_cell_idx"], onp.int64)
    # default cutoff: half the smallest edge of the perforated cells, so a thin-layered
    # completion keeps the channel graded through the layers above and below the perforation
    r0 = float(cfg.well_enc_r0_ft) if cfg.well_enc_r0_ft > 0 else float(
        0.5 * onp.asarray(case.cell_len, onp.float64)[cells].min())
    cent = onp.asarray(case.centroids, onp.float64)
    r_max = float(onp.sqrt(((cent[:, None, :] - perf[None, :, :]) ** 2).sum(-1)).min(1).max())
    r_max = max(r_max, 2.0 * r0)
    phi_c, grad_c = well_log_radius(cent, perf, r0, r_max)
    phi_n, _ = well_log_radius(onp.asarray(spec.node_xyz, onp.float64), perf, r0, r_max)
    v_c = jnp.concatenate([jnp.asarray(spec.v_c), jnp.asarray(phi_c)[:, None]], axis=1)
    b_v = jnp.concatenate([jnp.asarray(spec.b_v), jnp.asarray(grad_c)[:, :, None]], axis=2)
    v_nodes = jnp.concatenate([jnp.asarray(spec.v_nodes), jnp.asarray(phi_n)[:, None]], axis=1)
    print(f"[encoder] near-well channel ln r_w: {perf.shape[0]} perforation(s), "
          f"r_0 = {r0:.1f} ft, r_max = {r_max:.0f} ft")
    return v_c, b_v, v_nodes


def make_encoder(cfg: RunConfig, case: CaseData, spec: SpectralBundle | None):
    """Build the configured encoder (spectral needs a provisioned eigenbasis); under
    ``ic_design="hard"`` the encoder also carries the raw IC offsets (:func:`raw_ic_offsets`),
    under ``well_encoding="logr"`` the near-well channel (:func:`well_log_radius`)."""
    from .config import dim_in_of, hard_ic

    extra = {}
    if cfg.time_encoding == "log":
        extra["t_warp"] = resolve_time_warp(case.times, float(case.t_end), cfg.time_warp_days)
        print(f"[encoder] log time channel: t_w = {extra['t_warp']:.4g} d "
              f"({'auto' if cfg.time_warp_days <= 0 else 'set'}) over T_end = {float(case.t_end):.4g} d")
    if hard_ic(cfg):
        y0_c, y0_n, n_clamped = raw_ic_offsets(case, spec)
        extra = {"y0_cells": y0_c, "y0_nodes": y0_n}
        if n_clamped:
            print(f"[ic] hard IC: {n_clamped} of {4 * case.n_cells} initial primaries sit on a "
                  "sigmoid rail (S_g = 0 / S_w = S_wc) and are anchored to within 1e-4 of the range")
    if cfg.input_encoding is InputEncoding.SPECTRAL:
        if spec is None or spec.v_c is None:
            raise ValueError("spectral encoding requires a provisioned eigenbasis bundle")
        v_c, b_v, v_nodes = spec.v_c, spec.b_v, spec.v_nodes
        if cfg.well_encoding == "logr":
            v_c, b_v, v_nodes = _with_well_channel(cfg, case, spec)
        return SpectralEncoder(v_c=v_c, b_v=b_v, centroids=spec.centroids,
                               v_nodes=v_nodes, t_end=case.t_end, dim_in=dim_in_of(cfg),
                               **extra)
    node_xyz = spec.node_xyz if spec is not None else None
    if node_xyz is None:
        import jax.numpy as jnp

        node_xyz = jnp.asarray(case.verts, jnp.float32)
    return CartesianEncoder(lo=case.lo, hi=case.hi, node_xyz=node_xyz, t_end=case.t_end, **extra)
