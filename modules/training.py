r"""
The two-tier training loop (port of the notebook's training cell).

Outer tier: the supervision window — the deterministic full-batch window for a
special-only run, or a random window every ``adapt_every`` iterations when an
Adam phase exists. Inner tier: the per-iteration ENGD (or Adam) step on the
frozen window.

The loop is interrupt-safe (Ctrl-C retains the latest state), re-entrant on the
same :class:`TrainerState` (``train`` continues from ``state.train_iter`` with
the compiled steps cached, which is what lets the workflow stream a loss per
iteration), and keeps the scalar history in memory keyed by group NAME.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .config import SpecialOpt
from .optimizers import phase_of
from . import sampling as sampling_mod


@dataclass
class TrainerState:
    params: Any
    opt_state: Any
    opt_phase: str            # "adam" | "special"
    key: Any
    w: Any                    # (len(groups),) group weights
    hist: dict
    train_iter: int
    win: Any = None           # current Window


@dataclass
class RunResult:
    status: str               # completed | stalled | interrupted | failed
    iterations: int
    wall_time_s: float
    loss_total: float | None
    comps: dict = field(default_factory=dict)      # group -> final loss component
    weights: dict = field(default_factory=dict)    # group -> final weight
    engd_last_eta: float | None = None
    hist: dict = field(default_factory=dict)
    error: str = ""
    extra: dict = field(default_factory=dict)      # diagnostics metrics (RMSE, png paths, ...)


def _kind(cfg, phase: str) -> str:
    if phase == "adam":
        return "adam"
    return "lbfgs" if cfg.special_opt is SpecialOpt.LBFGS else "engd"


def _fresh_hist(groups) -> dict:
    h = {"iter": [], "total": [], "refresh_ms": [], "engd_eta": [], "wall_s": []}
    for g in groups:
        h[g] = []
        h[f"w_{g}"] = []
    return h


def init_trainer(bundle) -> TrainerState:
    """Fresh trainer state for one built pipeline (no checkpoint resume in v1)."""
    import jax.numpy as jnp
    from jax import random

    cfg, resolved, steps = bundle.cfg, bundle.resolved, bundle.steps
    params, opt_state, phase = steps.opt_init(bundle.model.params0)
    w = jnp.asarray(resolved.initial_weights, jnp.float32)
    train_iter = 0

    key = random.fold_in(random.PRNGKey(cfg.seed + 1), train_iter)
    return TrainerState(params=params, opt_state=opt_state, opt_phase=phase, key=key,
                        w=w, hist=_fresh_hist(resolved.groups), train_iter=train_iter)


def _refresh(bundle, state: TrainerState, key):
    """Open a new window: the deterministic full-batch window (special-only runs) or a
    random one; the fixed weights pass through unchanged."""
    cfg, resolved, env = bundle.cfg, bundle.resolved, bundle.env

    if resolved.full_batch:
        win = state.win if state.win is not None else sampling_mod.full_batch_window(
            cfg, bundle.case, resolved, env)
    else:
        win = sampling_mod.random_window(cfg, bundle.case, resolved, env, key)
    return win, state.w


def train(bundle, state: TrainerState, n_iter: int | None = None, log_every: int = 100,
          eval_hooks: dict | None = None, stop_on_eta_zero: bool = False) -> RunResult:
    r"""
    Run the two-tier loop for ``n_iter`` more iterations (default ``cfg.n_iter``).

    ``eval_hooks`` maps a name to a callable ``fn(params) -> float`` evaluated
    at every history-log step (the ``log_every`` cadence); each value is
    appended to ``hist[f"eval_{name}"]`` alongside the loss traces, e.g. a
    periodic time-mean RMSE on one or more mesh cases during training. The
    hooks receive the master parameters (cast internally by consumers such as
    :func:`pinnlab.diagnostics.predict_cells`).

    Every history-log step also appends the elapsed wall-clock seconds to
    ``hist["wall_s"]`` (continued from the last recorded value when ``train``
    is called again on the same state, so the timeline stays monotone across
    resumed segments), and — when ``cfg.engd.track_deff`` is set — the effective
    dimension :math:`d_{\mathrm{eff}}` of the damped Gauss-Newton system to
    ``hist["engd_d_eff"]`` (see :func:`pinnlab.engd.effective_dim`). That key is
    absent entirely on untracked runs, so its presence means "measured" rather
    than "measured as NaN".

    ``stop_on_eta_zero`` stops the run early — with ``status = "stalled"`` —
    the first time the ENGD line search accepts the zero step,

    .. math::

        \eta^\ast
        \;=\;
        \operatorname*{arg\,min}_{\eta \in \mathcal{S}}
        \mathcal{L}\bigl(\theta - \eta\,\psi\bigr)
        \;=\; 0,

    the exact fixed point of the deterministic full-batch iteration map: the
    grid floor :math:`2^{-30}` of :math:`\mathcal{S}` exceeds every loss-reducing
    step length, so the update freezes and further iterations cannot change
    :math:`\theta`.

    where:

    - :math:`\eta^\ast`: the accepted line-search step size (``EngdAux.eta``).
    - :math:`\mathcal{S} = \{0\} \cup \{2^{-j} : j = 0, \dots, 30\}`: the
      geometric line-search grid of :func:`pinnlab.engd.grid_line_search`.
    - :math:`\psi`: the damped natural-gradient direction.

    Checkpoints go to the bundle's Orbax manager every ``cfg.ckpt_every``
    iterations and, additionally, on ``KeyboardInterrupt`` (``status =
    "interrupted"``) so that a run stopped between two cadence saves resumes
    from the interrupted iteration rather than from the last multiple of
    ``ckpt_every``. The interrupt save is skipped when no iteration completed
    since ``train`` was entered.
    """
    import jax
    import numpy as onp
    from jax import random

    cfg, resolved, steps = bundle.cfg, bundle.resolved, bundle.steps
    groups = resolved.groups
    hist = state.hist
    status = "completed"
    last_eta = None
    static_window = resolved.full_batch

    t_start = time.perf_counter()
    wall_prev = hist.get("wall_s") or [0.0]
    wall0 = float(wall_prev[-1])
    start = int(state.train_iter)
    target = start + int(cfg.n_iter if n_iter is None else n_iter)

    try:
        while state.train_iter < target:
            it = int(state.train_iter)

            ph = phase_of(cfg, it)
            if ph != state.opt_phase:                       # optimizer hand-off
                state.opt_phase = ph
                if ph == "special":
                    state.opt_state = (steps.special_init(state.params)
                                       if steps.special_init is not None else ())
                else:
                    state.opt_state = steps.adam_init(state.params)
                if resolved.full_batch is False and cfg.special_opt is SpecialOpt.ENGD:
                    state.win = None                         # ENGD never straddles an Adam window
                print(f"it {it:5d} | optimizer hand-off -> {_kind(cfg, ph)}")

            needs_refresh = state.win is None or (it % cfg.adapt_every == 0
                                                  and not (static_window and state.win is not None))
            if needs_refresh:
                state.key, kr = random.split(state.key)
                t0 = time.perf_counter()
                state.win, state.w = _refresh(bundle, state, kr)
                jax.block_until_ready(state.w)
                refresh_ms = 1e3 * (time.perf_counter() - t0)
                hist["refresh_ms"].append(refresh_ms)
                # new window => restart the special state (curvature memory / nothing for ENGD),
                # unless the special optimizer accumulates ACROSS windows on purpose (SPRING)
                if (state.opt_phase == "special" and steps.special_init is not None
                        and steps.special_reset_on_window
                        and it != cfg.special_opt_after):
                    state.opt_state = steps.special_init(state.params)

            stalled = False
            if state.opt_phase == "special":
                state.params, state.opt_state, val, aux = steps.train_step_special(
                    state.params, state.opt_state, state.w, state.win)
                comps = None
                if aux is not None and hasattr(aux, "_fields"):
                    last_eta = float(aux.eta)
                    stalled = stop_on_eta_zero and last_eta == 0.0
            else:
                state.params, state.opt_state, val, comps = steps.train_step_adam(
                    state.params, state.opt_state, state.w, state.win)
                aux = None
            state.train_iter = it + 1

            if it % log_every == 0 or state.train_iter == target or stalled:
                if comps is None:
                    val, comps = steps.eval_comps(state.params, state.w, state.win)
                comps_h = onp.asarray(comps)
                w_h = onp.asarray(state.w)
                val_h = float(val)
                hist["iter"].append(it)
                hist["total"].append(val_h)
                hist.setdefault("wall_s", []).append(wall0 + (time.perf_counter() - t_start))
                for g, v in zip(groups, comps_h):
                    hist[g].append(float(v))
                for g, wv in zip(groups, w_h):
                    hist[f"w_{g}"].append(float(wv))
                if aux is not None and hasattr(aux, "_fields"):
                    hist["engd_eta"].append(float(aux.eta))
                    # d_eff is nan unless engd.track_deff is set; only record a
                    # tracked run's series so the key's presence means "measured"
                    d_eff = float(getattr(aux, "d_eff", float("nan")))
                    if d_eff == d_eff:                      # not nan
                        hist.setdefault("engd_d_eff", []).append(d_eff)
                if eval_hooks:
                    for h_name, h_fn in eval_hooks.items():
                        h_val = float(h_fn(state.params))
                        hist.setdefault(f"eval_{h_name}", []).append(h_val)
                comp_str = " ".join(f"{g} {v:.10e}" for g, v in zip(groups, comps_h))
                print(f"it {it:5d} | J {val_h:.4e} | {comp_str} | "
                      f"w {onp.array2string(w_h, precision=4)}")

            if stalled:
                status = "stalled"
                print(f"it {it:5d} | line search accepted eta* = 0 -> "
                      f"grid-floor fixed point reached; stopping.")
                break

    except KeyboardInterrupt:
        status = "interrupted"
        print(f"\ninterrupted at iteration {state.train_iter}; latest state retained.")

    wall = time.perf_counter() - t_start
    final_comps = {g: hist[g][-1] for g in groups if hist[g]}
    final_w = {g: hist[f"w_{g}"][-1] for g in groups if hist[f"w_{g}"]}
    print(f"training {status}: {cfg.residual_design.value}/{cfg.backprop_design.value if cfg.backprop_design else '-'}, "
          f"mesh {resolved.mesh_shape}, {cfg.sampling.value} sampling, at iteration {state.train_iter}.")
    return RunResult(
        status=status,
        iterations=int(state.train_iter),
        wall_time_s=wall,
        loss_total=hist["total"][-1] if hist["total"] else None,
        comps=final_comps,
        weights=final_w,
        engd_last_eta=last_eta,
        hist=hist,
    )
