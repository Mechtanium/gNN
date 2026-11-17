r"""
Energy Natural Gradient Descent (Müller & Zeinhofer, ICML 2023).

One ENGD update preconditions the loss gradient with the Gauss–Newton
(energy) metric of the weighted residual rows and takes the best step from a
geometric grid line search:

.. math::

    \left(\hat J^\top \hat J + \varepsilon I\right)\psi
    \;=\;
    \nabla_\theta \mathcal{L},
    \qquad
    \theta \leftarrow \theta - \eta^\ast\,\psi

where:

- :math:`\hat r(\theta) \in \mathbb{R}^{N}`: the weighted residual rows from
  :func:`modules.loss.make_residual_rows`, built so that
  :math:`\mathcal{L} = \Vert \hat r \Vert^2` exactly — group weights and
  scales enter the metric and the gradient consistently by construction.
- :math:`\hat J = \partial \hat r / \partial \theta \in \mathbb{R}^{N \times P}`:
  the row Jacobian; :math:`\nabla_\theta \mathcal{L} = 2 \hat J^\top \hat r`
  (the Gauss–Newton identity).
- :math:`\varepsilon = \varepsilon_{\mathrm{rel}}\,\operatorname{tr}(G)/\dim G
  + \varepsilon_{\mathrm{abs}}`: jit-safe relative Tikhonov damping (replaces
  the reference implementation's eager NaN retry loop). Under
  ``engd.damping_mode = "marquardt"`` the solve is Jacobi-equilibrated first, which
  makes the same :math:`\varepsilon` act as :math:`G + \varepsilon\operatorname{diag}G`
  — damping proportional to each direction's own curvature rather than a flat
  absolute floor, and scale-free (:func:`solve_dense`, :func:`_jacobi`).
- :math:`\eta^\ast`: the minimizer of :math:`\mathcal{L}(\theta - \eta\psi)`
  over the grid :math:`\eta \in \{\beta^0, \beta^1, \dots\}` (optionally with
  a prepended zero step so a fully stalled update can never ascend).

Two solve regimes, auto-selected at ``resolve()`` time by
:func:`modules.memplan.engd_plan` — the feasible one with the smaller solve
system (dense iff :math:`P \le N`):

- **dense** (:math:`8P^2 \le` ``jac_budget_bytes``): build the Gramian
  :math:`G = \hat J^\top \hat J`, then solve the damped :math:`P \times P`
  system.
- **rowspace** (:math:`8NP + 2\cdot 8N^2 \le` ``jac_budget_bytes`` —
  :math:`\hat J` plus the damped row Gram and its Cholesky factor):
  materialize :math:`\hat J` and solve the :math:`N \times N` system
  :math:`(\hat J \hat J^\top + \varepsilon I)\alpha = 2\hat r`,
  :math:`\psi = \hat J^\top \alpha` — identical to the damped dense direction
  by the push-through identity
  :math:`(\hat J^\top \hat J + \varepsilon I)^{-1} \hat J^\top
  = \hat J^\top (\hat J \hat J^\top + \varepsilon I)^{-1}`.

Either regime assembles :math:`\hat J` in whichever AD direction needs fewer
sweeps of the residual graph (also fixed by :func:`modules.memplan.engd_plan`):

- **reverse** (``rev``): one vjp sweep per residual row,
  :math:`e_i^\top \hat J = \hat J_{i,:}` — :math:`N` sweeps.
- **forward** (``fwd``, taken when :math:`P < N` and :math:`\hat J` fits the
  Jacobian budget): one jvp sweep per parameter column,
  :math:`\hat J\, e_j = \hat J_{:,j}` — :math:`P` sweeps. For the full-batch
  Galerkin/FEM row sets (:math:`N \sim 10^4\!-\!10^5` rows against
  :math:`P \sim 10^2\!-\!10^3` parameters) this is the difference between
  hours and seconds per iteration.

The direction is chosen **per loss group** whenever the row function exposes
its group slices (``rows_fn.group_rows``, see
:func:`modules.loss.make_residual_rows`): :func:`gramian_grouped` /
:func:`assemble_jacobian_grouped` sweep each group's own residual graph in its
own cheaper direction and sum the contributions,
:math:`G = \sum_g \hat J_g^\top \hat J_g`. The projected PDE group — a few dozen
rows, each a whole-mesh FEM balance — is then :math:`N_{\mathrm{pde}}` vjp
sweeps instead of :math:`P` jvp columns dragged through the balance, while the
cheap point-wise cell-state and IC rows stay forward. The single-direction
:func:`gramian_streamed` / :func:`assemble_jacobian` remain the fallback for
row functions without group slices.

The step is stateless (``opt_state`` passes through unchanged), runs the
whole tape in float64, and closes over nothing window-shaped: RAR refreshes
simply feed the next call a new frozen window, so :math:`\hat J`,
:math:`\hat r`, the gradient, and every line-search probe always see the same
points.

Two variants repurpose that ``opt_state`` slot for cross-iteration state, both
wired by a :mod:`modules.pipeline` wrapper rather than a ``RunConfig`` switch:

- :func:`make_spring_step` (:class:`SpringConfig`, :func:`modules.pipeline.with_spring`)
  — SPRING momentum, which shifts the solve's regularizer toward the previous
  direction, :math:`\lambda\Vert\phi - \mu\phi^{k-1}\Vert^2`, so the update
  aggregates curvature across iterations instead of re-deriving it from scratch.
  Free relative to assembly: one extra :math:`P\times P` matvec, no extra AD sweeps.
- :func:`make_masked_engd_step` (:class:`MaskConfig`,
  :func:`modules.pipeline.with_masked_engd`) — the adaptive top-:math:`k`
  residual-update mask.

:func:`effective_dim` reports :math:`d_{\mathrm{eff}} = \operatorname{tr}(G(G +
\varepsilon I)^{-1})`, a smooth count of how many of the :math:`P` parameter
directions survive the damping — i.e. how many the step can actually address.
It is logged into ``EngdAux.d_eff`` when ``engd.track_deff`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, NamedTuple

import numpy as onp

from .config import EngdSettings, Resolved, RunConfig


class EngdAux(NamedTuple):
    """Traced per-step ENGD scalars (logged to history and TensorBoard).

    ``mask_frac`` is the realized fraction of parameters the step updated —
    ``1.0`` on the unmasked path, and the top-:math:`k` selection fraction on
    the masked path of :func:`make_masked_engd_step` (whose ``psi_norm`` is the
    norm of the *applied* direction :math:`M \\odot \\psi`).

    ``d_eff`` is the effective dimension of the damped system (see
    :func:`effective_dim`) — the smooth count of eigendirections the step could
    actually address. It is ``nan`` unless ``engd.track_deff`` is set, since it
    costs an eigendecomposition per step.
    """

    eta: object        # accepted line-search step size
    eps: object        # Tikhonov damping actually applied
    psi_norm: object   # ||psi|| of the applied update direction
    grad_norm: object  # ||grad L|| = ||2 J^T r||
    loss: object       # loss at the accepted point
    mask_frac: object = 1.0  # fraction of parameters updated (1.0 unmasked)
    d_eff: object = float("nan")  # effective dimension (nan unless track_deff)


class MaskState(NamedTuple):
    """Cross-iteration state of the masked ENGD step (rides the ``opt_state`` slot).

    The plain ENGD step is stateless and threads ``opt_state = ()`` through
    untouched; the masked variant repurposes that slot for the EMA score
    accumulator and the held mask, so :func:`modules.training.train` needs no
    loop changes.
    """

    ema: object          # (P,) EMA of |psi| (used only by score_type="ema_abs_psi")
    mask: object         # (P,) the last selection mask (held when mask_update="fixed")
    initialized: object  # () bool — False until the first masked step has run


class SpringState(NamedTuple):
    r"""Cross-iteration state of the SPRING step (rides the ``opt_state`` slot).

    ``phi`` is the momentum accumulator :math:`\phi^k` *before* bias correction
    (the corrected vector is what the update applies, but the recursion must
    carry the uncorrected one, matching the reference implementation);
    ``step`` is the 1-based iteration counter the bias correction needs.
    """

    phi: object    # (P,) momentum accumulator, uncorrected
    step: object   # () int32 -- 1-based iteration index for the bias correction


@dataclass(frozen=True)
class SpringConfig:
    r"""
    Configuration of the SPRING momentum variant of ENGD
    (:func:`make_spring_step`); Guzmán-Cordero et al., NeurIPS 2025, Algorithm 1,
    after Goldshlager et al.'s SPRING for variational Monte Carlo.

    Plain ENGD re-solves the damped Gauss–Newton system from scratch every
    iteration, so it carries no curvature information across steps. SPRING
    keeps the same system but *shifts the regularizer toward the previous
    direction*: instead of

    .. math::

        \phi^k
        \;=\;
        \operatorname*{arg\,min}_{\phi}
        \bigl[\,\Vert \hat J_k\,\phi - \hat r_k\Vert^2
        + \lambda\Vert\phi\Vert^2\,\bigr],

    it solves

    .. math::

        \phi^k
        \;=\;
        \operatorname*{arg\,min}_{\phi}
        \bigl[\,\Vert \hat J_k\,\phi - \hat r_k\Vert^2
        + \lambda\Vert\phi - \mu\,\phi^{k-1}\Vert^2\,\bigr],

    whose optimality condition gives the iteration actually implemented,

    .. math::

        \phi^k
        \;=\;
        \mu\,\phi^{k-1}
        + \bigl(G_k + \varepsilon I\bigr)^{-1}
          \bigl(\nabla_\theta\mathcal{L}_k - \mu\,G_k\,\phi^{k-1}\bigr),
        \qquad
        \hat\phi^k = \frac{\phi^k}{\sqrt{1 - \mu^{2k}}},
        \qquad
        \theta^{k+1}
        \;=\;
        \theta^k - \min\!\Bigl(\eta,\;
        \frac{\sqrt{C}}{\Vert\hat\phi^k\Vert}\Bigr)\,\hat\phi^k .

    where:

    - :math:`\mu \in [0, 1)`: ``momentum``; :math:`\mu = 0` recovers plain ENGD exactly.
    - :math:`G_k = \hat J_k^\top \hat J_k`, :math:`\varepsilon`: the Gauss–Newton Gramian and its damping, unchanged from :func:`solve_dense` — SPRING alters only the right-hand side.
    - :math:`\phi^k`: the momentum accumulator carried in :class:`SpringState` (uncorrected); :math:`\hat\phi^k` is the bias-corrected direction actually applied.
    - :math:`\sqrt{1 - \mu^{2k}}`: the Adam-style bias correction (``bias_correction``), which stops the first few steps from being shrunk by the zero initialization :math:`\phi^0 = 0`.
    - :math:`\eta`, :math:`C`: ``lr`` and ``norm_constraint`` — the step is the learning rate capped so that :math:`\Vert\Delta\theta\Vert \le \sqrt{C}`, which is what lets SPRING dispense with a line search.

    Setting ``line_search=True`` instead feeds :math:`\hat\phi^k` to the usual
    geometric grid line search, keeping ENGD's monotone-descent guarantee at the
    cost of the extra probe evaluations. Under the deterministic full-batch
    window that guarantee is worth more than it is in the stochastic setting
    SPRING was designed for, so both rules are offered.

    Only the ``dense`` solve regime is supported: the row-space regime is
    selected exactly when :math:`P > N`, where the momentum vector is larger
    than the system being solved and the accumulator costs more than the solve
    it accelerates.
    """

    momentum: float = 0.9
    lr: float = 1.0
    norm_constraint: float = 1e-3
    bias_correction: bool = True
    line_search: bool = False

    def __post_init__(self):
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.norm_constraint <= 0.0:
            raise ValueError(f"norm_constraint must be > 0, got {self.norm_constraint}")


@dataclass(frozen=True)
class MaskConfig:
    r"""
    Configuration of the adaptive residual-update mask for
    :func:`make_masked_engd_step`.

    At iteration :math:`k` the regular damped natural-gradient direction

    .. math::

        \psi^k = \bigl(G + \varepsilon I\bigr)^{-1}\,
        \nabla_\theta \mathcal{L}(\theta^k)

    is scored per parameter by one of three interchangeable formulas,

    .. math::

        s_i = |\psi^k_i|, \qquad
        s_i = |\psi^k_i\,\theta_i|, \qquad
        s_i = \mathrm{EMA}_\rho\bigl(|\psi^k_i|\bigr),

    a binary top-:math:`k` (or bottom-:math:`k`) mask is formed from the
    scores, :math:`M_i = \mathbb{1}[\, s_i \text{ is among the top-}k
    \text{ fraction of } \{s_j\} \,]`, and only the selected parameters are
    line-searched and updated,

    .. math::

        \eta^\ast = \operatorname*{arg\,min}_{\eta \in \mathcal{S}}
        \mathcal{L}\bigl(\theta^k - \eta\,(M \odot \psi^k)\bigr),
        \qquad
        \theta^{k+1} = \theta^k - \eta^\ast\,(M \odot \psi^k).

    where:

    - :math:`\psi^k`: the natural-gradient direction, obtained by solving the
      damped Gramian system (:func:`solve_dense` / :func:`solve_rowspace` —
      never an explicit inverse).
    - :math:`G`, :math:`\varepsilon`: the Gauss–Newton Gramian and its
      relative-plus-absolute Tikhonov damping (:func:`_damping`).
    - :math:`s_i`: the importance score of parameter :math:`i`;
      :math:`\rho` (``ema_decay``) is the EMA decay rate, meaningful only for
      ``score_type = "ema_abs_psi"`` (:math:`\mathrm{EMA}_\rho(v)^{k} =
      \rho\,\mathrm{EMA}_\rho(v)^{k-1} + (1 - \rho)\,v^k`).
    - :math:`M_i \in \{0, 1\}`: the selection mask; :math:`k` denotes a
      *fraction* of the :math:`P` parameters (``select_k_frac``), not a count,
      and ``select_k`` switches between the top and bottom fraction.
    - :math:`\eta^\ast`, :math:`\mathcal{S} = \{0\} \cup \{2^{-j} : j = 0,
      \dots, 30\}`: the accepted line-search step and its candidate grid,
      identical to the unmasked ENGD update.

    ``mask_update = "fixed"`` computes the mask once — from the scores at the
    first masked step — and holds it constant thereafter (it is re-initialized
    wherever the special optimizer state restarts, i.e. at supervision-window
    boundaries; under the deterministic full-batch window there is exactly one
    window, so "fixed" means fixed for the whole run). ``"every_iter"``
    recomputes the selection from that iteration's :math:`\psi^k` every step.
    """

    score_type: Literal["abs_psi", "abs_psi_theta", "ema_abs_psi"] = "abs_psi"
    ema_decay: float = 0.9
    select_k: Literal["Top", "Bottom"] = "Top"
    select_k_frac: float = 0.25
    mask_update: Literal["fixed", "every_iter"] = "every_iter"

    def __post_init__(self):
        if self.score_type not in ("abs_psi", "abs_psi_theta", "ema_abs_psi"):
            raise ValueError(f"unknown score_type {self.score_type!r}")
        if self.select_k not in ("Top", "Bottom"):
            raise ValueError(f"select_k must be 'Top' or 'Bottom', got {self.select_k!r}")
        if not 0.0 < self.select_k_frac <= 1.0:
            raise ValueError(f"select_k_frac must be in (0, 1], got {self.select_k_frac}")
        if self.mask_update not in ("fixed", "every_iter"):
            raise ValueError(f"unknown mask_update {self.mask_update!r}")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")


def _linearize(r_of: Callable, theta):
    """One vjp linearization of the rows at ``theta``: ``(r0, e -> e^T J)``."""
    import jax

    r0, vjp_fn = jax.vjp(r_of, theta)
    return r0, lambda e: vjp_fn(e)[0]


def _use_fwd(r_of: Callable, theta, direction: str) -> bool:
    """Resolve ``fwd``/``rev``/``auto`` (auto: forward iff P < N, by eval_shape)."""
    import jax

    if direction != "auto":
        return direction == "fwd"
    return theta.shape[0] < jax.eval_shape(r_of, theta).shape[0]


def _jacobian_fwd(r_of: Callable, theta, col_chunk: int):
    r"""
    Materialize :math:`\hat J^\top` column-by-column from one jvp linearization:
    :math:`\hat J\, e_j = \hat J_{:,j}` under a chunked ``lax.map`` — the
    residual graph is linearized once and re-swept :math:`P` times (each
    one-hot tangent :math:`e_j` synthesized on the fly from its column index).
    Returns ``(r0, Jt)`` with ``Jt.shape == (P, N)``.
    """
    import jax
    import jax.numpy as jnp

    r0, jv = jax.linearize(r_of, theta)
    p = theta.shape[0]
    Jt = jax.lax.map(lambda j: jv(jax.nn.one_hot(j, p, dtype=theta.dtype)),
                     jnp.arange(p), batch_size=min(col_chunk, p))
    return r0, Jt


def assemble_jacobian(r_of: Callable, theta, row_chunk: int = 64,
                      direction: str = "auto"):
    r"""
    Materialize :math:`\hat J` from one linearization, in the cheaper AD
    direction.

    ``rev`` seeds the transpose map with standard-basis covectors
    :math:`e_i^\top \hat J = \hat J_{i,:}` under a chunked ``lax.map`` —
    :math:`N` vjp sweeps. ``fwd`` (``auto`` picks it iff :math:`P < N`) pushes
    standard-basis tangents through :func:`_jacobian_fwd` instead —
    :math:`P` jvp sweeps. Every seed is synthesized on the fly from its index
    (never an identity matrix, whose :math:`\mathcal{O}(N^2)` storage exceeds
    device memory for full-batch row counts), so peak seed memory is
    :math:`\mathcal{O}(\mathrm{chunk} \cdot \max(N, P))`.
    Returns ``(r0, J)`` with ``J.shape == (N, P)``.
    """
    import jax
    import jax.numpy as jnp

    if _use_fwd(r_of, theta, direction):
        r0, Jt = _jacobian_fwd(r_of, theta, row_chunk)
        return r0, Jt.T

    r0, vt = _linearize(r_of, theta)
    n = r0.shape[0]
    J = jax.lax.map(lambda i: vt(jax.nn.one_hot(i, n, dtype=r0.dtype)),
                    jnp.arange(n), batch_size=min(row_chunk, n))
    return r0, J


def gramian_streamed(r_of: Callable, theta, row_chunk: int = 64,
                     direction: str = "auto", init=None):
    r"""
    The energy Gramian :math:`G = \hat J^\top \hat J` and gradient
    :math:`\nabla_\theta \mathcal{L} = 2 \hat J^\top r_0`, assembled in the
    cheaper AD direction. Returns ``(r0, G, grad)``; with ``init = (G_0, g_0)`` the
    contributions are accumulated onto that running pair instead of onto zeros (the
    grouped assembly threads one accumulator through every group).

    ``rev`` scans identity chunks :math:`E_c \in \mathbb{R}^{c \times N}`
    synthesized on the fly from their row indices (out-of-range indices in the
    final chunk one-hot to zero covectors and contribute nothing), accumulating
    :math:`G \mathrel{+}= J_c^\top J_c` with :math:`J_c = E_c \hat J` from the
    shared linearization — :math:`N` vjp sweeps, :math:`\hat J` never stored.
    Materializing the :math:`N \times N` identity instead would need
    :math:`\mathcal{O}(N^2)` storage — 16.9 GiB/device at the
    :math:`N = 67{,}312` full-batch spectral rows.

    ``fwd`` (``auto`` picks it iff :math:`P < N`) materializes
    :math:`\hat J^\top` in :math:`P` jvp sweeps and contracts
    :math:`G = \hat J^\top \hat J`, :math:`\nabla = 2 \hat J^\top r_0` —
    :math:`N/P` times fewer residual-graph sweeps, at
    :math:`\mathcal{O}(N P)` transient storage
    (:func:`modules.memplan.engd_plan` only selects ``fwd`` when that fits
    ``jac_budget_bytes``).
    """
    import jax
    import jax.numpy as jnp

    if _use_fwd(r_of, theta, direction):
        r0, Jt = _jacobian_fwd(r_of, theta, row_chunk)
        if init is None:
            return r0, Jt @ Jt.T, 2.0 * (Jt @ r0)
        return r0, init[0] + Jt @ Jt.T, init[1] + 2.0 * (Jt @ r0)

    r0, vt = _linearize(r_of, theta)
    n = r0.shape[0]
    p = theta.shape[0]
    chunk = min(row_chunk, n)
    n_chunks = -(-n // chunk)

    def body(G, c):
        E_c = jax.nn.one_hot(c * chunk + jnp.arange(chunk), n, dtype=r0.dtype)
        J_c = jax.vmap(vt)(E_c)
        return G + J_c.T @ J_c, None

    G0 = jnp.zeros((p, p), r0.dtype) if init is None else init[0]
    G, _ = jax.lax.scan(body, G0, jnp.arange(n_chunks))
    grad = vt(2.0 * r0)
    return r0, G, (grad if init is None else init[1] + grad)


def _group_direction(r_of: Callable, theta, jac_budget_bytes: int) -> str:
    """``fwd`` iff the group has more rows than parameters and its Jacobian fits the budget."""
    import jax

    p = theta.shape[0]
    n_g = jax.eval_shape(r_of, theta).shape[0]
    return "fwd" if (p < n_g and n_g * p * 8 <= jac_budget_bytes) else "rev"


def gramian_pointwise(point_fn, n_points: int, theta, point_chunk: int = 2048, init=None,
                      budget_bytes: int = 4 << 30):
    r"""
    The Gramian and gradient of a point-wise group — ``point_fn(theta, i)`` returns the
    :math:`m` weighted rows of supervision point :math:`i` — assembled from per-point
    reverse-mode Jacobian blocks.

    The rows of such a group are independent across points, so :math:`\hat J_g` is a
    stack of :math:`m \times P` blocks and

    .. math::

        G_g \;=\; \sum_{i=1}^{N_{\mathrm{pts}}} J_i^\top J_i,
        \qquad
        \nabla_\theta \mathcal{L}_g \;=\; 2\sum_{i} J_i^\top \hat r_i,
        \qquad
        J_i \;=\; \frac{\partial \hat r_i}{\partial \theta} \in \mathbb{R}^{m \times P}

    where:

    - :math:`J_i`: the point's block, :math:`m` vjp sweeps of the *single-point* graph (``jax.jacrev``, batched over a chunk of points by ``vmap``) — the cost of one point's network evaluation times :math:`3m`, independent of :math:`P`; the whole-batch alternative sweeps :math:`P` jvp columns through all :math:`N_{\mathrm{pts}}` points.
    - :math:`m`: rows per point (the four primaries of a cell-state or IC row); :math:`N_{\mathrm{pts}}`: the group's points on the window.
    - ``point_chunk``: points per ``lax.scan`` step, bounding the transient block stack at :math:`8\,m\,P\,\mathrm{chunk}` bytes times a compiler-dependent factor; each chunk's :math:`G` contribution is one GEMM. The last chunk is zero-padded (masked rows contribute nothing). ``0`` sizes it from ``budget_bytes`` for the backend in use (:func:`modules.memplan.point_chunk_for`).
    - ``init``: an optional running ``(G_0, g_0)`` the contributions are accumulated onto, as in :func:`gramian_streamed`.

    Returns ``(r0, G, grad)`` with ``r0`` the group's rows in point order.
    """
    import jax
    import jax.numpy as jnp

    from .memplan import point_chunk_for

    p = theta.shape[0]
    m = jax.eval_shape(lambda th: point_fn(th, jnp.int32(0)), theta).shape[0]
    point_chunk = point_chunk_for(point_chunk, budget_bytes, m, p, jax.default_backend())
    chunk = max(1, min(point_chunk, n_points))
    n_chunks = -(-n_points // chunk)
    pad = n_chunks * chunk - n_points

    def point_rows_and_jac(i):
        i_safe = jnp.minimum(i, n_points - 1)
        r_i = point_fn(theta, i_safe)
        J_i = jax.jacrev(lambda th: point_fn(th, i_safe))(theta)          # (m, P)
        return r_i, J_i

    def body(carry, c):
        G, grad = carry
        ids = c * chunk + jnp.arange(chunk)
        r_c, J_c = jax.vmap(point_rows_and_jac)(ids)                        # (chunk, m), (chunk, m, P)
        if pad:
            v = (ids < n_points).astype(theta.dtype)
            r_c = r_c * v[:, None]
            J_c = J_c * v[:, None, None]
        r_flat = jnp.reshape(r_c, (-1,))
        J_flat = jnp.reshape(J_c, (-1, p))
        G = G + J_flat.T @ J_flat
        grad = grad + 2.0 * (J_flat.T @ r_flat)
        return (G, grad), r_flat

    if init is None:
        init = (jnp.zeros((p, p), theta.dtype), jnp.zeros((p,), theta.dtype))
    (G, grad), r_chunks = jax.lax.scan(body, tuple(init), jnp.arange(n_chunks))
    r0 = jnp.reshape(r_chunks, (-1,))[: n_points * m]
    return r0, G, grad


def gramian_grouped(group_row_fns, theta, row_chunk: int = 64,
                    jac_budget_bytes: int = 4 << 30, point_chunk: int = 0):
    r"""
    The Gramian and gradient of the full row vector, assembled group by group with
    each group swept in its own cheaper AD direction.

    The weighted rows are a concatenation over the loss groups,
    :math:`\hat r = (\hat r_{g_1}, \dots, \hat r_{g_K})`, so the Gauss–Newton
    quantities are sums of per-group contributions,

    .. math::

        G \;=\; \hat J^\top \hat J \;=\; \sum_{g} \hat J_g^\top \hat J_g,
        \qquad
        \nabla_\theta \mathcal{L} \;=\; 2\sum_{g} \hat J_g^\top \hat r_g,

    and each :math:`\hat J_g \in \mathbb{R}^{N_g \times P}` can be assembled in
    :math:`\min(N_g, P)` sweeps of *that group's* residual graph alone. This is
    the difference between a projected-PDE group (:math:`N_g = 3\,n_\lambda\,S`,
    a few dozen rows through the whole-mesh FEM balance — reverse mode, one vjp per
    row) and a full-batch cell-state group (:math:`N_g \sim 10^4`–:math:`10^5`
    cheap point evaluations — forward mode, one jvp per parameter), which a single
    global direction cannot serve: forward mode drags every one of the :math:`P`
    jvp columns through the FEM balance, reverse mode sweeps the cell-state rows
    :math:`N_g` times.

    where:

    - :math:`\hat J_g = \partial \hat r_g / \partial \theta`: the row Jacobian of group :math:`g` (:func:`modules.loss.make_residual_rows` exposes the per-group slices as ``rows.group_rows``).
    - :math:`N_g`: the group's row count on the window; :math:`P` the parameter count; :math:`S` the collocation slices; :math:`n_\lambda` the retained modes.
    - the per-group direction: ``fwd`` iff :math:`P < N_g` and the materialized :math:`\hat J_g` (:math:`8 N_g P` bytes) fits ``jac_budget_bytes``, else ``rev`` — the rule of :func:`modules.memplan.engd_plan` applied per group.

    Point-wise groups (``ic``, ``data``: an entry given as ``(row_fn, point_fn,
    n_points)``) take the per-point reverse-mode route of :func:`gramian_pointwise`
    instead of either whole-batch direction.

    The groups are swept one at a time onto **one** running accumulator
    :math:`(G, \nabla_\theta\mathcal L)`: each group's sweep starts behind an
    optimization barrier on the running pair, so its residual tape and seed blocks are
    never live beside another group's, and the :math:`8P^2`-byte Gramian exists once
    rather than once per group. Left to itself the compiler schedules the independent
    group sweeps side by side — on NORNE at :math:`P = 15{,}164` that held four
    :math:`1.7` GiB Gramians and every group's transients at once (7.5 GiB of
    temporaries for the assembly alone).

    Returns ``(r0, G, grad)`` with ``r0`` the concatenated rows in group order —
    identical to :func:`gramian_streamed` on the concatenated row function, up to
    float64 reassociation of the group sums.
    """
    import jax
    import jax.numpy as jnp

    p = theta.shape[0]
    G = jnp.zeros((p, p), theta.dtype)
    grad = jnp.zeros((p,), theta.dtype)
    r_parts = []
    for entry in group_row_fns:
        th, G, grad = jax.lax.optimization_barrier((theta, G, grad))
        if isinstance(entry, tuple):
            r_of, point_fn, n_points = entry
            r0_g, G, grad = gramian_pointwise(point_fn, n_points, th, point_chunk,
                                              init=(G, grad), budget_bytes=jac_budget_bytes)
        else:
            r_of = entry
            direction = _group_direction(r_of, th, jac_budget_bytes)
            r0_g, G, grad = gramian_streamed(r_of, th, row_chunk, direction, init=(G, grad))
        r_parts.append(r0_g)
    return jnp.concatenate(r_parts), G, grad


def assemble_jacobian_grouped(group_row_fns, theta, row_chunk: int = 64,
                              jac_budget_bytes: int = 4 << 30):
    r"""
    Materialize :math:`\hat J` as the row-stack of the per-group Jacobians, each
    assembled in its own cheaper direction (the row-space counterpart of
    :func:`gramian_grouped`). Returns ``(r0, J)`` with ``J.shape == (N, P)`` and
    the rows in group order.
    """
    import jax.numpy as jnp

    r_parts, j_parts = [], []
    for entry in group_row_fns:
        r_of = entry[0] if isinstance(entry, tuple) else entry
        direction = _group_direction(r_of, theta, jac_budget_bytes)
        r0_g, J_g = assemble_jacobian(r_of, theta, row_chunk, direction)
        r_parts.append(r0_g)
        j_parts.append(J_g)
    return jnp.concatenate(r_parts), jnp.concatenate(j_parts, axis=0)


def _row_ops(rows_fn: Callable, unflatten, w, win):
    """The concatenated row function and, when ``rows_fn`` exposes ``group_rows``,
    the per-group entries the grouped assembly sweeps (``None`` otherwise): a plain
    whole-group row function, or ``(row_fn, point_fn, n_points)`` for a point-wise group."""
    r_of = lambda th: rows_fn(unflatten(th), w, win)
    group_rows = getattr(rows_fn, "group_rows", None)
    if group_rows is None:
        return r_of, None
    entries = []
    for _, fn, point in group_rows:
        row_fn = (lambda th, fn=fn: fn(unflatten(th), w, win))
        if point is None:
            entries.append(row_fn)
        else:
            point_fn, n_points = point
            entries.append((row_fn,
                            (lambda th, i, pf=point_fn: pf(unflatten(th), w, win, i)),
                            int(n_points(win))))
    return r_of, tuple(entries)


def _assemble_dense(r_of, group_fns, theta, s: EngdSettings, direction: str):
    """``(r0, G, grad)`` — grouped when per-group rows exist, single-direction otherwise."""
    if group_fns is None:
        return gramian_streamed(r_of, theta, s.row_chunk, direction)
    return gramian_grouped(group_fns, theta, s.row_chunk, s.jac_budget_bytes, s.point_chunk)


def _assemble_rows(r_of, group_fns, theta, s: EngdSettings, direction: str):
    """``(r0, J)`` — grouped when per-group rows exist, single-direction otherwise."""
    if group_fns is None:
        return assemble_jacobian(r_of, theta, s.row_chunk, direction)
    return assemble_jacobian_grouped(group_fns, theta, s.row_chunk, s.jac_budget_bytes)


def _damping(trace, dim, s: EngdSettings):
    return s.damping_rel * trace / dim + s.damping_abs


def _phys_damping(curv, phys, eps_iso, s: EngdSettings):
    r"""
    The per-parameter damping of ``damping_mode="marquardt_phys"``,

    .. math::

        e_i \;=\;
        \begin{cases}
        \varepsilon & i \text{ a network parameter} \\
        \max\bigl(\varepsilon_{\mathrm{rel}}\,G_{ii} + \varepsilon_{\mathrm{abs}},\; 10^{-6}\,\varepsilon\bigr) & i \text{ a physical parameter}
        \end{cases}

    where :math:`\varepsilon` is the isotropic damping of the regime (:func:`_damping`) and
    :math:`G_{ii}` the Gauss–Newton curvature of parameter :math:`i`. The physical
    parameters are few, scaled unlike the weights and often weakly determined; damping each
    in proportion to its own curvature lets it move at the pace its data allow instead of
    at the pace the network's curvature sets, and the floor keeps a parameter no row
    reaches from dividing by zero.
    """
    import jax.numpy as jnp

    return jnp.where(phys, jnp.maximum(s.damping_rel * curv + s.damping_abs, 1e-6 * eps_iso), eps_iso)


def _jacobi(M):
    r"""
    The Jacobi (van der Sluis) equilibration of an SPD matrix:
    :math:`\tilde M = D^{-1/2} M D^{-1/2}` with :math:`D = \operatorname{diag} M`,
    so :math:`\operatorname{diag}\tilde M = \mathbf{1}`. Returns
    ``(M_tilde, inv_sqrt_d)``.

    Equilibrating the diagonal is *near-optimal* for the condition number of an SPD
    matrix (van der Sluis): it cannot be beaten by more than a factor of the matrix
    dimension. It matters here because :math:`\operatorname{cond}(G) =
    \operatorname{cond}(\hat J)^2`, and the row set mixes modal rows spanning
    :math:`\lambda_1 \dots \lambda_{n_\lambda}`, three phases, and storage-vs-flux
    row character — a spread that lands squarely on :math:`G`'s diagonal.

    Zero (or non-finite) diagonal entries — a parameter no residual row depends on —
    are pinned to a unit scale rather than dividing by zero; such a column is
    all-zero in :math:`\tilde M` and picks up the damping alone.
    """
    import jax.numpy as jnp

    d = jnp.diag(M)
    inv_sqrt_d = jnp.where(d > 0, 1.0 / jnp.sqrt(jnp.where(d > 0, d, 1.0)), 1.0)
    return M * inv_sqrt_d[:, None] * inv_sqrt_d[None, :], inv_sqrt_d


def effective_dim(M, eps):
    r"""
    The effective dimension of the damped system :math:`M + \varepsilon I`,

    .. math::

        d_{\mathrm{eff}}(M, \varepsilon)
        \;=\;
        \operatorname{tr}\bigl(M (M + \varepsilon I)^{-1}\bigr)
        \;=\;
        \sum_{i} \frac{\lambda_i}{\lambda_i + \varepsilon},

    where:

    - :math:`\lambda_i \ge 0`: the eigenvalues of :math:`M` (the Gramian :math:`G` in
      dense mode, the row Gram :math:`\hat J \hat J^\top` in row-space mode — the two
      share their nonzero spectrum, so either yields the same :math:`d_{\mathrm{eff}}`).
    - :math:`\varepsilon`: the damping actually applied by the solve.

    Each eigendirection contributes between 0 and 1, so :math:`d_{\mathrm{eff}}` is a
    *smooth count* of the directions that survive damping: it is the number of
    parameter directions the natural-gradient step can genuinely address. Directions
    with :math:`\lambda_i \ll \varepsilon` contribute :math:`\approx \lambda_i /
    \varepsilon \approx 0` (damping-dominated, effectively frozen); directions with
    :math:`\lambda_i \gg \varepsilon` contribute :math:`\approx 1`.

    Costs one symmetric eigendecomposition of :math:`M`, which is negligible against
    the :math:`\min(N, P)` AD sweeps that assembled it.
    """
    import jax.numpy as jnp

    lam = jnp.linalg.eigvalsh(M)
    lam = jnp.clip(lam, 0.0, None)
    return jnp.sum(lam / (lam + eps))


def _cho_solve_escalated(A, rhs, eps):
    r"""
    Damped Cholesky solve with jit-safe damping escalation.

    A Gramian whose condition number approaches :math:`1/\epsilon_{\mathrm{mach}}`
    (e.g. well-observable rows whose Jacobians span many decades) can round to a
    numerically indefinite matrix, and ``cho_factor`` then yields NaN. Each rung
    retries with :math:`10^{6}\times` the previous damping, executing only when
    the previous solve produced non-finite values. Returns ``(x, eps_used)``.
    """
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jsl

    # damping on the diagonal alone: A + e*I with a materialized identity costs one more
    # dim x dim buffer (a third of the solve's temporaries at P = 15k)
    diag = jnp.arange(A.shape[0])

    def _solve(e):
        return jsl.cho_solve(jsl.cho_factor(A.at[diag, diag].add(e)), rhs)

    def _rung(carry, _):
        x, e, ok = carry
        e_next = e * 1e6
        x2, e2 = jax.lax.cond(ok, lambda: (x, e), lambda: (_solve(e_next), e_next))
        return (x2, e2, jnp.all(jnp.isfinite(x2))), None

    x0 = _solve(eps)
    carry = (x0, eps, jnp.all(jnp.isfinite(x0)))
    (x, eps_used, _), _ = jax.lax.scan(_rung, carry, None, length=3)
    return x, eps_used


def solve_dense(G, grad, s: EngdSettings, phys=None):
    r"""
    The :math:`P \times P` solve :math:`(G + \varepsilon I)\psi = \nabla\mathcal{L}`.

    ``solver="chol"`` uses the damped Cholesky factorization;
    ``solver="lstsq"`` is the reference-faithful SVD pseudo-inverse
    :math:`\psi = G^{+}\nabla\mathcal{L}` with explicit ``rcond``
    (:math:`\varepsilon = 0` reported).

    Under ``damping_mode="marquardt"`` the solve is Jacobi-equilibrated first
    (:func:`_jacobi`): with :math:`D = \operatorname{diag} G` and
    :math:`\tilde G = D^{-1/2} G D^{-1/2}`,

    .. math::

        \bigl(\tilde G + \varepsilon I\bigr)\,y
        \;=\;
        D^{-1/2}\nabla\mathcal{L},
        \qquad
        \psi \;=\; D^{-1/2} y
        \qquad\Longleftrightarrow\qquad
        \bigl(G + \varepsilon D\bigr)\,\psi = \nabla\mathcal{L},

    i.e. isotropic damping of the equilibrated matrix *is* Marquardt damping of the
    original one — the identity follows from
    :math:`\tilde G + \varepsilon I = D^{-1/2}(G + \varepsilon D)D^{-1/2}`. Two things
    change together: the factorized matrix has unit diagonal (so the Cholesky sees a
    near-optimally conditioned system), and :math:`\varepsilon` becomes scale-free,
    since :math:`\operatorname{tr}\tilde G / P = 1` makes
    :math:`\varepsilon = \varepsilon_{\mathrm{rel}} + \varepsilon_{\mathrm{abs}}` exactly.

    Under ``damping_mode="marquardt_phys"`` (with the boolean mask ``phys`` of the
    physical parameters) the damping is the diagonal :math:`E` of :func:`_phys_damping` —
    isotropic on the network, curvature-proportional on the physical block — solved the
    same way, :math:`(E^{-1/2} G E^{-1/2} + I)\,y = E^{-1/2}\nabla\mathcal{L}`,
    :math:`\psi = E^{-1/2} y`.

    Returns ``(psi, eps, M)`` where ``M`` is the matrix the damping was applied to
    (:math:`G` or :math:`\tilde G`), for :func:`effective_dim` to report against.
    """
    import jax.numpy as jnp
    import jax.scipy.linalg as jsl

    p = G.shape[0]
    if s.solver == "lstsq":
        psi = jnp.linalg.lstsq(G, grad, rcond=s.rcond)[0]
        return psi, jnp.zeros((), G.dtype), G
    if s.damping_mode == "marquardt_phys" and phys is not None:
        eps_iso = _damping(jnp.trace(G), p, s)
        inv_sqrt_e = 1.0 / jnp.sqrt(_phys_damping(jnp.diag(G), phys, eps_iso, s))
        G_t = G * inv_sqrt_e[:, None] * inv_sqrt_e[None, :]
        y, eps = _cho_solve_escalated(G_t, grad * inv_sqrt_e, jnp.ones((), G.dtype))
        return y * inv_sqrt_e, eps, G_t
    if s.damping_mode == "marquardt":
        G_t, inv_sqrt_d = _jacobi(G)
        eps = _damping(jnp.trace(G_t), p, s)          # == damping_rel + damping_abs
        y, eps = _cho_solve_escalated(G_t, grad * inv_sqrt_d, eps)
        return y * inv_sqrt_d, eps, G_t
    eps = _damping(jnp.trace(G), p, s)
    psi, eps = _cho_solve_escalated(G, grad, eps)
    return psi, eps, G


def solve_rowspace(J, r0, s: EngdSettings, row_shift=None, phys=None):
    r"""
    The :math:`N \times N` rowspace solve.

    .. math::

        (\hat J \hat J^\top + \varepsilon I)\,\alpha = 2\hat r - c,
        \qquad
        \psi = \hat J^\top \alpha

    where:

    - :math:`\hat J \hat J^\top \in \mathbb{R}^{N \times N}`: the row Gram
      matrix (SPD up to damping), cheap when :math:`N \ll P`.
    - :math:`\alpha`: the rowspace coefficients of the direction.
    - :math:`c`: the optional ``row_shift``, a covector in :math:`\mathbb{R}^N`
      subtracted from the right-hand side (``None`` :math:`\Rightarrow c = 0`).

    By the push-through identity this equals the damped dense direction with
    the same :math:`\varepsilon`; under ``damping_mode="marquardt_phys"`` the same identity
    with the diagonal damping :math:`E` of :func:`_phys_damping` gives
    :math:`\psi = E^{-1}\hat J^\top(\hat J E^{-1} \hat J^\top + I)^{-1}(2\hat r - c)`. Under ``damping_mode="marquardt"`` the row Gram is
    Jacobi-equilibrated exactly as in :func:`solve_dense`, damping
    :math:`\hat J\hat J^\top + \varepsilon\operatorname{diag}(\hat J \hat J^\top)`
    — note this is the *row*-scale analogue (rows equilibrated by
    :math:`\Vert \hat J_{i,:}\Vert`), not the parameter-scale one, since it is the
    row Gram that is being factorized here.

    ``row_shift`` exists for :func:`make_spring_step`. SPRING's shifted right-hand
    side lies in :math:`\hat J`'s row space,

    .. math::

        \nabla_\theta\mathcal{L} - \mu\,G\,\phi
        \;=\;
        \hat J^\top\bigl(2\hat r - \mu\,\hat J\phi\bigr),

    so applying push-through to it gives the momentum direction without ever
    forming the :math:`P \times P` Gramian:

    .. math::

        \bigl(G + \varepsilon I\bigr)^{-1}
        \hat J^\top\bigl(2\hat r - \mu \hat J\phi\bigr)
        \;=\;
        \hat J^\top\bigl(\hat J\hat J^\top + \varepsilon I\bigr)^{-1}
        \bigl(2\hat r - \mu \hat J\phi\bigr).

    Pass :math:`c = \mu\,\hat J\phi` — one :math:`N \times P` matvec.

    Returns ``(psi, eps, grad, A)``. ``grad`` is always the true gradient
    :math:`2\hat J^\top \hat r`, unshifted, so the reported ``grad_norm`` stays
    comparable across the momentum and plain steps; ``A`` is the matrix the
    damping was applied to, for :func:`effective_dim` to report against.
    """
    import jax.numpy as jnp
    import jax.scipy.linalg as jsl

    n = r0.shape[0]
    A = J @ J.T
    grad = 2.0 * (J.T @ r0)                          # true gradient, never shifted
    rhs = 2.0 * r0 if row_shift is None else 2.0 * r0 - row_shift
    if s.solver == "lstsq":
        alpha = jnp.linalg.lstsq(A, rhs, rcond=s.rcond)[0]
        return J.T @ alpha, jnp.zeros((), J.dtype), grad, A
    if s.damping_mode == "marquardt_phys" and phys is not None:
        # (J^T J + E) psi = J^T rhs  <=>  psi = E^-1 J^T (J E^-1 J^T + I)^-1 rhs
        eps_iso = _damping(jnp.trace(A), n, s)
        inv_sqrt_e = 1.0 / jnp.sqrt(_phys_damping(jnp.sum(J * J, axis=0), phys, eps_iso, s))
        J_s = J * inv_sqrt_e[None, :]
        A_t = J_s @ J_s.T
        y, eps = _cho_solve_escalated(A_t, rhs, jnp.ones((), J.dtype))
        return (J_s.T @ y) * inv_sqrt_e, eps, grad, A_t
    if s.damping_mode == "marquardt":
        A_t, inv_sqrt_d = _jacobi(A)
        eps = _damping(jnp.trace(A_t), n, s)          # == damping_rel + damping_abs
        y, eps = _cho_solve_escalated(A_t, rhs * inv_sqrt_d, eps)
        return J.T @ (y * inv_sqrt_d), eps, grad, A_t
    eps = _damping(jnp.trace(A), n, s)
    alpha, eps = _cho_solve_escalated(A, rhs, eps)
    return J.T @ alpha, eps, grad, A


def grid_line_search(objective: Callable, theta, psi, s: EngdSettings):
    r"""
    Geometric grid line search over :math:`\eta \in \{\beta^0, \dots,
    \beta^{K-1}\}` (optionally prepending :math:`\eta = 0`).

    Probes run under ``lax.map`` (not vmap) so the residual graph is not
    replicated :math:`K` times; NaN losses are mapped to :math:`+\infty`
    before the argmin so a divergent probe can never be selected.
    Returns ``(theta_new, eta, loss_new)``.
    """
    import jax
    import jax.numpy as jnp

    psi = jnp.where(jnp.isfinite(psi), psi, 0.0)
    steps = jnp.asarray(s.ls_base, theta.dtype) ** jnp.arange(s.ls_num, dtype=theta.dtype)
    if s.ls_include_zero:
        steps = jnp.concatenate([jnp.zeros((1,), theta.dtype), steps])
    losses = jax.lax.map(lambda eta: objective(theta - eta * psi), steps)
    losses = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    i = jnp.argmin(losses)
    eta = steps[i]
    return theta - eta * psi, eta, losses[i]


def phys_mask(params0):
    """Boolean mask over the flattened parameter vector selecting the ``params["phys"]``
    leaves (all ``False`` when there are none)."""
    import jax

    from .models import flatten_state

    mask = jax.tree.map(lambda x: onp.zeros(onp.shape(x), onp.float32), params0)
    if isinstance(mask, dict) and "phys" in mask:
        mask["phys"] = jax.tree.map(lambda x: onp.ones(onp.shape(x), onp.float32), mask["phys"])
    flat, _ = flatten_state(mask)
    return onp.asarray(flat) > 0.5


def make_engd_step(cfg: RunConfig, resolved: Resolved, loss_fn: Callable,
                   rows_fn: Callable, model_bundle, env, freeze_phys: bool = False) -> Callable:
    r"""
    The jitted stateless ENGD step:
    ``step(params, opt_state, w, win) -> (params', opt_state, loss, EngdAux)``.

    Params are promoted to float64 masters, flattened to one vector for the
    solve, and unflattened back; ``opt_state`` passes through untouched. The
    solve regime and assembly direction are fixed at build time from
    ``resolved.engd_plan``.

    ``freeze_phys`` holds the physical parameters of an inversion fixed: their columns
    leave the Gauss–Newton system (their rows and columns of :math:`G` are replaced by the
    network block's mean curvature on the diagonal, their columns of :math:`\hat J` by
    zeros), so the step is the network's own Gauss–Newton step at the current physical
    values — the forward solve at a given parameter set.
    """
    import jax
    import jax.numpy as jnp

    from .meshenv import shard_like_params
    from .models import flatten_state

    s = cfg.engd
    mode = resolved.engd_plan.mode
    direction = resolved.engd_plan.direction
    param_shardings = shard_like_params(env, model_bundle.pspec_of_shape, model_bundle.params0)
    pm = phys_mask(model_bundle.params0)
    has_phys = bool(pm.any())
    pm_j = jnp.asarray(pm)
    use_phys = has_phys and s.damping_mode == "marquardt_phys"

    @jax.jit
    def step(params, opt_state, w, win):
        params = jax.lax.with_sharding_constraint(params, param_shardings)
        params = jax.tree.map(lambda x: x.astype(jnp.float64), params)
        theta, unflatten = flatten_state(params)

        r_of, group_fns = _row_ops(rows_fn, unflatten, w, win)
        objective = lambda th: jnp.sum(r_of(th) ** 2)
        phys = pm_j if use_phys else None

        if mode == "dense":
            r0, G, grad = _assemble_dense(r_of, group_fns, theta, s, direction)
            if freeze_phys and has_phys:
                keep = (~pm_j).astype(G.dtype)
                d_net = jnp.sum(jnp.diag(G) * keep) / jnp.maximum(jnp.sum(keep), 1.0)
                G = G * keep[:, None] * keep[None, :] + jnp.diag((1.0 - keep) * d_net)
                grad = grad * keep
            psi, eps, M = solve_dense(G, grad, s, phys=None if freeze_phys else phys)
        else:
            r0, J = _assemble_rows(r_of, group_fns, theta, s, direction)
            if freeze_phys and has_phys:
                J = J * (~pm_j).astype(J.dtype)[None, :]
            psi, eps, grad, M = solve_rowspace(J, r0, s, phys=None if freeze_phys else phys)

        theta_new, eta, loss_new = grid_line_search(objective, theta, psi, s)
        aux = EngdAux(eta=eta, eps=eps,
                      psi_norm=jnp.linalg.norm(psi),
                      grad_norm=jnp.linalg.norm(grad),
                      loss=loss_new,
                      d_eff=effective_dim(M, eps) if s.track_deff
                      else jnp.asarray(jnp.nan, theta.dtype))
        return unflatten(theta_new), opt_state, loss_new, aux

    return step


def make_spring_step(cfg: RunConfig, resolved: Resolved, loss_fn: Callable,
                     rows_fn: Callable, model_bundle, env,
                     spring_cfg: SpringConfig) -> tuple[Callable, Callable]:
    r"""
    The jitted SPRING step and its state initializer:
    ``step(params, SpringState, w, win) -> (params', SpringState', loss, EngdAux)``,
    ``init_spring_state() -> SpringState``.

    Identical to :func:`make_engd_step` through the Gramian assembly, then the
    solve's right-hand side is shifted by the previous direction and the result
    accumulated (see :class:`SpringConfig` for the derivation):

    .. math::

        \phi^k
        \;=\;
        \mu\,\phi^{k-1}
        + \bigl(G_k + \varepsilon I\bigr)^{-1}
          \bigl(\nabla_\theta\mathcal{L}_k - \mu\,G_k\,\phi^{k-1}\bigr).

    The shifted right-hand side costs one matrix–vector product — :math:`P \times P`
    against :math:`G` in the dense regime, :math:`N \times P` against :math:`\hat J`
    in the row-space one — so a SPRING iteration is the same :math:`\min(N, P)` AD
    sweeps as a plain ENGD one; the momentum is free relative to the assembly.
    Writing the shift this way (rather than as
    :math:`\nabla\mathcal{L} + \varepsilon\mu\phi^{k-1}`, the algebraically identical
    form) keeps :math:`\varepsilon` entirely inside the solver, so the damping
    escalation and the Jacobi/Marquardt mode apply unchanged.

    **Both solve regimes are supported.** Row-space is selected exactly when
    :math:`P > N` — which is the norm under per-iteration resampling, where each
    window is a small stochastic subset — and there the shift is carried by
    :func:`solve_rowspace`'s ``row_shift`` via the push-through identity, so
    :math:`\phi` is recovered in full :math:`P`-space without materializing
    :math:`G`. The two paths are algebraically identical for a given
    :math:`\varepsilon`.

    ``eta`` in the returned :class:`EngdAux` is the realized step scale —
    the line-search step when ``line_search`` is set, otherwise
    :math:`\min(\eta, \sqrt{C}/\Vert\hat\phi^k\Vert)` — and ``psi_norm`` is
    :math:`\Vert\hat\phi^k\Vert`, the norm of the applied direction.

    Wire it into a built pipeline with :func:`modules.pipeline.with_spring`.
    """
    import jax
    import jax.numpy as jnp

    from .meshenv import shard_like_params
    from .models import flatten_state

    s = cfg.engd
    mode = resolved.engd_plan.mode
    direction = resolved.engd_plan.direction
    mu = spring_cfg.momentum
    param_shardings = shard_like_params(env, model_bundle.pspec_of_shape, model_bundle.params0)

    def init_spring_state() -> SpringState:
        p64 = jax.tree.map(lambda x: x.astype(jnp.float64), model_bundle.params0)
        theta0, _ = flatten_state(p64)
        return SpringState(phi=jnp.zeros_like(theta0), step=jnp.zeros((), jnp.int32))

    @jax.jit
    def step(params, opt_state: SpringState, w, win):
        params = jax.lax.with_sharding_constraint(params, param_shardings)
        params = jax.tree.map(lambda x: x.astype(jnp.float64), params)
        theta, unflatten = flatten_state(params)

        r_of, group_fns = _row_ops(rows_fn, unflatten, w, win)
        objective = lambda th: jnp.sum(r_of(th) ** 2)

        phi_prev = opt_state.phi
        k = opt_state.step + 1

        if mode == "dense":
            r0, G, grad = _assemble_dense(r_of, group_fns, theta, s, direction)
            psi, eps, _M = solve_dense(G, grad - mu * (G @ phi_prev), s)
        else:
            # push-through: the shifted RHS lives in J's row space, so the momentum
            # never needs the P x P Gramian -- one N x P matvec instead
            r0, J = _assemble_rows(r_of, group_fns, theta, s, direction)
            psi, eps, grad, _M = solve_rowspace(J, r0, s, row_shift=mu * (J @ phi_prev))
        phi = mu * phi_prev + psi

        if spring_cfg.bias_correction:
            corr = jnp.sqrt(1.0 - jnp.asarray(mu, theta.dtype) ** (2 * k))
            phi_hat = phi / (corr + 1e-16)
        else:
            phi_hat = phi
        phi_hat = jnp.where(jnp.isfinite(phi_hat), phi_hat, 0.0)

        if spring_cfg.line_search:
            theta_new, eta, loss_new = grid_line_search(objective, theta, phi_hat, s)
        else:
            nrm = jnp.linalg.norm(phi_hat)
            eta = jnp.minimum(jnp.asarray(spring_cfg.lr, theta.dtype),
                              jnp.sqrt(spring_cfg.norm_constraint) / (nrm + 1e-16))
            theta_new = theta - eta * phi_hat
            loss_new = objective(theta_new)

        aux = EngdAux(eta=eta, eps=eps,
                      psi_norm=jnp.linalg.norm(phi_hat),
                      grad_norm=jnp.linalg.norm(grad),
                      loss=loss_new,
                      d_eff=effective_dim(G, eps) if s.track_deff
                      else jnp.asarray(jnp.nan, theta.dtype))
        return unflatten(theta_new), SpringState(phi=phi, step=k), loss_new, aux

    return step, init_spring_state


def compute_scores(psi, theta, ema, mask_cfg: MaskConfig):
    r"""
    Per-parameter importance scores for the masked step (and the updated EMA).

    .. math::

        s_i = |\psi_i|, \qquad
        s_i = |\psi_i\,\theta_i|, \qquad
        s_i = \rho\,\mathrm{ema}_i + (1 - \rho)\,|\psi_i|

    for ``score_type`` ``"abs_psi"`` / ``"abs_psi_theta"`` / ``"ema_abs_psi"``
    respectively. Returns ``(scores, ema_new)``; the EMA accumulator advances
    only under the EMA score (the other two pass it through unchanged).
    """
    import jax.numpy as jnp

    if mask_cfg.score_type == "ema_abs_psi":
        ema_new = mask_cfg.ema_decay * ema + (1.0 - mask_cfg.ema_decay) * jnp.abs(psi)
        return ema_new, ema_new
    if mask_cfg.score_type == "abs_psi_theta":
        return jnp.abs(psi * theta), ema
    return jnp.abs(psi), ema


def mask_from_scores(scores, select_k_frac: float, select_k: str = "Top"):
    r"""
    The binary top-:math:`k` (or bottom-:math:`k`) selection mask

    .. math::

        M_i = \mathbb{1}\bigl[\, s_i \text{ is among the top-}k\text{ fraction}
        \text{ of } \{s_j\}_{j=1}^{P} \,\bigr],
        \qquad
        k = \max\bigl(1, \operatorname{round}(\texttt{select\_k\_frac}\cdot P)\bigr)

    built by scattering ones at the ``jax.lax.top_k`` indices of the scores
    (negated scores for the bottom fraction), so exactly :math:`k` entries are
    selected even under ties.
    """
    import jax
    import jax.numpy as jnp

    p = scores.shape[0]
    k = max(1, int(round(select_k_frac * p)))
    ranked = scores if select_k == "Top" else -scores
    _, idx = jax.lax.top_k(ranked, k)
    return jnp.zeros_like(scores).at[idx].set(1.0)


def make_masked_engd_step(cfg: RunConfig, resolved: Resolved, loss_fn: Callable,
                          rows_fn: Callable, model_bundle, env,
                          mask_cfg: MaskConfig) -> tuple[Callable, Callable]:
    r"""
    The jitted masked ENGD step and its state initializer:
    ``step(params, MaskState, w, win) -> (params', MaskState', loss, EngdAux)``,
    ``init_mask_state() -> MaskState``.

    Identical to :func:`make_engd_step` through the solve for :math:`\psi`
    (same regime dispatch, damping, and float64 flattened vector), then the
    :class:`MaskConfig` scoring/selection restricts the line-searched update to
    the masked direction :math:`M \odot \psi` (see :class:`MaskConfig` for the
    update equations). The unselected parameters are bitwise unchanged by
    construction (:math:`\theta - \eta \cdot 0 = \theta`).

    The cross-iteration EMA accumulator and held mask ride the ``opt_state``
    slot as a :class:`MaskState`; wire both into a built pipeline with
    :func:`modules.pipeline.with_masked_engd`.
    """
    import jax
    import jax.numpy as jnp

    from .meshenv import shard_like_params
    from .models import flatten_state

    s = cfg.engd
    mode = resolved.engd_plan.mode
    direction = resolved.engd_plan.direction
    param_shardings = shard_like_params(env, model_bundle.pspec_of_shape, model_bundle.params0)

    def init_mask_state() -> MaskState:
        p64 = jax.tree.map(lambda x: x.astype(jnp.float64), model_bundle.params0)
        theta0, _ = flatten_state(p64)
        return MaskState(ema=jnp.zeros_like(theta0),
                         mask=jnp.ones_like(theta0),
                         initialized=jnp.zeros((), bool))

    @jax.jit
    def step(params, opt_state: MaskState, w, win):
        params = jax.lax.with_sharding_constraint(params, param_shardings)
        params = jax.tree.map(lambda x: x.astype(jnp.float64), params)
        theta, unflatten = flatten_state(params)

        r_of, group_fns = _row_ops(rows_fn, unflatten, w, win)
        objective = lambda th: jnp.sum(r_of(th) ** 2)

        if mode == "dense":
            r0, G, grad = _assemble_dense(r_of, group_fns, theta, s, direction)
            psi, eps, M = solve_dense(G, grad, s)
        else:
            r0, J = _assemble_rows(r_of, group_fns, theta, s, direction)
            psi, eps, grad, M = solve_rowspace(J, r0, s)

        scores, ema_new = compute_scores(psi, theta, opt_state.ema, mask_cfg)
        mask_fresh = mask_from_scores(scores, mask_cfg.select_k_frac, mask_cfg.select_k)
        if mask_cfg.mask_update == "fixed":
            mask = jnp.where(opt_state.initialized, opt_state.mask, mask_fresh)
        else:
            mask = mask_fresh
        psi_m = mask * psi

        theta_new, eta, loss_new = grid_line_search(objective, theta, psi_m, s)
        aux = EngdAux(eta=eta, eps=eps,
                      psi_norm=jnp.linalg.norm(psi_m),
                      grad_norm=jnp.linalg.norm(grad),
                      loss=loss_new,
                      mask_frac=jnp.mean(mask),
                      d_eff=effective_dim(M, eps) if s.track_deff
                      else jnp.asarray(jnp.nan, theta.dtype))
        new_state = MaskState(ema=ema_new, mask=mask, initialized=jnp.ones((), bool))
        return unflatten(theta_new), new_state, loss_new, aux

    return step, init_mask_state
