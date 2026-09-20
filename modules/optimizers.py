r"""
Optimizer phase machinery: Adam warm-up plus a switchable "special" second
phase (L-BFGS or ENGD), generalizing the notebook's two-phase recipe.

Phase schedule (``special_opt_after`` replaces the old ``NB_LBFGS_AFTER``):

- ``special_opt = none`` (``after = -1``): pure Adam.
- ``after = 0``: the special optimizer runs from iteration 0 and the Adam
  transformation is never constructed (setup as if Adam never existed).
- ``after > 0``: Adam until the absolute iteration ``after``, then hand off.

The Adam phase trains float32 casts of the (possibly float64) master weights,
so the AD tape is unchanged and only the elementwise moment update runs at
``prec_opt``. The L-BFGS phase uses optax's strong-Wolfe zoom line search with
``value_and_grad_from_state`` (the accepted point's value/grad pair is reused —
no extra objective evaluation per step); its curvature memory is restarted at
every supervision-window boundary so it never spans two windows. The ENGD
phase is stateless (see :mod:`pinnlab.engd`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .config import Resolved, RunConfig, SpecialOpt
from .meshenv import MeshEnv, shard_like_params


def phase_of(cfg: RunConfig, it: int) -> str:
    """Optimizer phase ("adam" | "special") at absolute iteration ``it``."""
    if cfg.special_opt is SpecialOpt.NONE:
        return "adam"
    return "special" if it >= cfg.special_opt_after else "adam"


@dataclass
class Steps:
    r"""Jitted training steps + state initializers for one configured run.

    ``special_reset_on_window`` controls whether :func:`pinnlab.training.train`
    re-initializes ``opt_state`` from ``special_init`` when a new supervision
    window opens (the optimizer hand-off always initializes, independently of
    this flag). The default ``True`` is the L-BFGS convention: a quasi-Newton
    curvature memory built on one window is invalid on the next, since the
    objective it approximates has changed.

    A stochastic optimizer inverts that argument. When the window is redrawn
    every iteration, the accumulated direction is the *only* thing carrying
    information across samples, and re-initializing it each refresh erases the
    method — so :func:`pinnlab.pipeline.with_spring` sets this ``False``.

    where:

    - ``special_init``: fresh special-optimizer state from the current params (``None`` for stateless plain ENGD, so the flag is inert there).
    - ``special_kind``: the configured :class:`~pinnlab.config.SpecialOpt`; unchanged by the pipeline wrappers, which all require ENGD.
    """

    train_step_adam: Callable | None       # (params, opt_state, w, win) -> (params, ost, val, comps)
    train_step_special: Callable | None    # (params, opt_state, w, win) -> (params, ost, val, aux)
    adam_init: Callable | None
    special_init: Callable | None          # fresh special state from params (L-BFGS; None for ENGD)
    opt_init: Callable                     # params0 -> (master params, opt_state, phase name)
    eval_comps: Callable                   # (params, w, win) -> (val, comps)
    param_shardings: Any
    abstract_opt: dict                     # phase kind -> abstract optimizer-state template
    opt_shardings: dict
    as_master: Callable
    as_f32: Callable
    prec_opt: Any
    special_kind: SpecialOpt
    special_reset_on_window: bool = True    # re-init special_init at each window boundary


def make_train_steps(cfg: RunConfig, resolved: Resolved, loss_fn: Callable,
                     rows_fn: Callable | None, model_bundle, env: MeshEnv) -> Steps:
    """Build the phase steps for the configured optimizer component."""
    import jax
    import jax.numpy as jnp
    import optax

    prec_opt = jnp.float64 if resolved.prec_opt == "float64" else jnp.float32
    as_master = lambda tree: jax.tree.map(lambda x: x.astype(prec_opt), tree)
    as_f32 = lambda tree: jax.tree.map(lambda x: x.astype(jnp.float32), tree)

    param_shardings = shard_like_params(env, model_bundle.pspec_of_shape, model_bundle.params0)
    params0 = model_bundle.params0

    skip_adam = cfg.special_opt is not SpecialOpt.NONE and cfg.special_opt_after == 0
    tx_adam = None if skip_adam else optax.adam(
        optax.exponential_decay(cfg.lr0, cfg.lr_decay_steps, cfg.lr_decay))
    tx_lbfgs = (optax.lbfgs(memory_size=cfg.lbfgs_mem)
                if cfg.special_opt is SpecialOpt.LBFGS else None)

    abstract_opt: dict = {}
    if tx_adam is not None:
        abstract_opt["adam"] = jax.eval_shape(lambda p: tx_adam.init(as_master(p)), params0)
    if tx_lbfgs is not None:
        abstract_opt["lbfgs"] = jax.eval_shape(lambda p: tx_lbfgs.init(as_master(p)), params0)
    opt_shardings = {n: shard_like_params(env, model_bundle.pspec_of_shape, a)
                     for n, a in abstract_opt.items()}

    _wsc = lambda p: jax.lax.with_sharding_constraint(p, param_shardings)

    train_step_adam = None
    adam_init = None
    if tx_adam is not None:
        @jax.jit
        def train_step_adam(params, opt_state, w, win):
            params = _wsc(params)
            (val, comps), grads = jax.value_and_grad(
                lambda p, w_, wn: loss_fn(as_f32(p), w_, wn), has_aux=True)(params, w, win)
            updates, opt_state = tx_adam.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, val, comps

        adam_init = jax.jit(tx_adam.init)

    train_step_special = None
    special_init = None
    if cfg.special_opt is SpecialOpt.LBFGS:
        @jax.jit
        def train_step_special(params, opt_state, w, win):
            params = _wsc(params)
            vfn = lambda p: loss_fn(as_f32(p), w, win)[0]
            val, grads = optax.value_and_grad_from_state(vfn)(params, state=opt_state)
            updates, opt_state = tx_lbfgs.update(grads, opt_state, params,
                                                 value=val, grad=grads, value_fn=vfn)
            params = optax.apply_updates(params, updates)
            return params, opt_state, val, None

        special_init = jax.jit(tx_lbfgs.init)
    elif cfg.special_opt is SpecialOpt.ENGD:
        from .engd import make_engd_step

        train_step_special = make_engd_step(cfg, resolved, loss_fn, rows_fn, model_bundle, env)
        special_init = None   # ENGD is stateless

    @jax.jit
    def eval_comps(params, w, win):
        return loss_fn(as_f32(params), w, win)

    phase0 = phase_of(cfg, 0)
    kind0 = ("adam" if phase0 == "adam"
             else ("lbfgs" if cfg.special_opt is SpecialOpt.LBFGS else "engd"))

    def opt_init(params):
        params = as_master(jax.device_put(params, param_shardings))
        if kind0 == "adam":
            return params, tx_adam.init(params), phase0
        if kind0 == "lbfgs":
            return params, tx_lbfgs.init(params), phase0
        return params, (), phase0   # ENGD: stateless

    return Steps(
        train_step_adam=train_step_adam,
        train_step_special=train_step_special,
        adam_init=adam_init,
        special_init=special_init,
        opt_init=opt_init,
        eval_comps=eval_comps,
        param_shardings=param_shardings,
        abstract_opt=abstract_opt,
        opt_shardings=opt_shardings,
        as_master=as_master,
        as_f32=as_f32,
        prec_opt=prec_opt,
        special_kind=cfg.special_opt,
    )
