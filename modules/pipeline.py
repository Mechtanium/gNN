r"""
The pipeline glue: validate → resolve → load → wire.

:func:`build_pipeline` assembles one :class:`Bundle` from a
:class:`~modules.config.RunConfig` (with the bit-parity and float32 init gates
asserted along the way); :mod:`modules.stream` drives training on it and
streams the losses and cell states out.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from . import (casedata, encodings, loss as loss_mod, meshenv, models, memplan, physics,
               residuals as res_mod, spectral, wells as wells_mod)
from .config import (Resolved, RunConfig, case_label, loss_groups, resolve, resolve_capacity,
                     structural_hash, validate)
from .optimizers import Steps, make_train_steps


@dataclass
class Bundle:
    """Everything one configured run needs, fully wired."""

    cfg: RunConfig
    resolved: Resolved
    env: Any
    case: Any
    spec: Any
    model: Any
    prim: Any
    ops: Any
    scales: Any
    centroids: Any
    loss_fn: Callable
    rows_fn: Callable
    steps: Steps
    well_pack: Any = None          # modules.wells.WellPack


def resolve_run(cfg: RunConfig, device_count: int | None = None, case: Any = None):
    """
    Settle every ``0`` auto sentinel against the actual reservoir, then bind to devices.

    The single place the auto fields are resolved, so a preview and the run it previews
    cannot disagree. Order is forced: ``n_eig`` fixes ``dim_in``, ``dim_in`` fixes the
    parameter count and the auto-capacity width, and :func:`~pinnlab.config.resolve`
    reads both — so an unresolved ``n_eig`` would report a basis width of one and size
    the network against it.

    Returns ``(cfg, resolved, case, warnings)``. The case is returned because resolving
    ``n_eig`` needs it; pass it to :func:`build_pipeline` to avoid loading it twice.
    """
    import jax

    n_dev = jax.device_count() if device_count is None else device_count
    # Which capacity fields were on the auto sentinel, before validate() fills them: once
    # n_eig moves, anything auto-sized must be re-derived at the new dim_in, and by then
    # the field is no longer 0 so resolve_capacity alone would be a no-op.
    auto_m, auto_l = cfg.m_width <= 0, cfg.n_blocks <= 0
    cfg, warns = validate(cfg)              # normalize FIRST (resolve re-validates idempotently)

    if case is None:
        case = casedata.load_case(cfg)
    n_eig_before = cfg.n_eig
    cfg, notes = spectral.resolve_n_eig(cfg, case)
    warns = list(warns) + list(notes)
    if cfg.n_eig != n_eig_before and (auto_m or auto_l):
        cfg = resolve_capacity(replace(
            cfg, m_width=0 if auto_m else cfg.m_width,
            n_blocks=0 if auto_l else cfg.n_blocks))
        warns.append(f"auto capacity re-sized at the resolved n_eig={cfg.n_eig}: "
                     f"m_width={cfg.m_width}, n_blocks={cfg.n_blocks}")
    return cfg, resolve(cfg, n_dev), case, warns


def build_pipeline(cfg: RunConfig, device_count: int | None = None,
                   verbose: bool = True, case: Any = None) -> Bundle:
    """Validate, resolve, load, and wire one run. Prints the build summary.

    ``case`` injects a pre-built (possibly perturbed) :class:`~pinnlab.casedata.CaseData`
    — the log-permeability workflows train against a *prior* permeability field
    while the prep-cache observations stay those of the true one. It is also how
    :func:`resolve_run` hands back the case it had to load, so a notebook preview and
    the build it precedes share one load.
    """
    import jax
    import jax.numpy as jnp
    import numpy as onp

    n_dev = jax.device_count() if device_count is None else device_count
    cfg, resolved, case, _warnings = resolve_run(cfg, n_dev, case)
    groups = resolved.groups

    env = meshenv.build_mesh(cfg, n_dev)
    if verbose:
        for w in dict.fromkeys(list(_warnings) + list(resolved.warnings)):
            print(f"[config] WARNING: {w}")
        print(f"[config] groups={groups} mesh={resolved.mesh_shape} dim_in={resolved.dim_in} "
              f"P={resolved.param_count:,} hash={structural_hash(cfg)}")
        if resolved.engd_plan is not None:
            p = resolved.engd_plan
            print(f"[engd] mode={p.mode} dir={p.direction} N_rows={p.n_rows:,} "
                  f"G={p.g_bytes/1e6:.0f} MB J={p.j_bytes/1e6:.0f} MB "
                  f"A={p.a_bytes/1e6:.0f} MB seeds={p.seed_bytes/1e6:.0f} MB")

    spec = spectral.place_on_mesh(spectral.provision(case, cfg), cfg, env)
    centroids = spec.centroids

    enc = encodings.make_encoder(cfg, case, spec)
    mb = models.build_model(cfg, resolved.dim_in, seed=cfg.seed)

    # --- init gates: bit parity vs jaxpinns + all-f32 leaves (both policies) --------------
    probe = jax.random.normal(jax.random.PRNGKey(99), (4, resolved.dim_in), jnp.float32)
    y_nnx = jax.vmap(lambda f: mb.apply(mb.params0, f))(probe)
    y_raw = jax.vmap(lambda f: mb.raw_apply(mb.raw_params0, f))(probe)
    assert onp.array_equal(onp.asarray(y_nnx), onp.asarray(y_raw)), "NNX/raw bit-parity gate failed"
    assert all(l.dtype == jnp.float32 for l in jax.tree.leaves(mb.params0)), "f32 init gate failed"

    # The well pack backs BOTH the observation rows (well group) and the interior
    # realized-rate forcing Q(t): the closed-form model's data-driven source, which is
    # also the scale-calibration forcing (see calibrate_scales).
    invm = mb_obs = head = None
    pack = wells_mod.build_well_pack(cfg, case)
    if verbose:
        n_bind = int((pack.binding > 0).sum())
        print(f"[wells] {pack.n_wells} wells / {pack.n_perf} perfs / {pack.n_times} steps "
              f"-> {pack.n_rows_well} observation rows + {pack.n_rows_ctrl} control rows, "
              f"{n_bind} BHP-limited (well, step) pairs"
              + (f" (SYNTHESIZED: {pack.synth_reason})" if pack.synthesized else " (summary)"))
    _, obs_hi = case.t_obs_span()
    _, coll_hi = case.t_span()
    forcing_calib = wells_mod.build_forcing(
        pack, case, from_controls=False,
        controls_from=(obs_hi if obs_hi < coll_hi - 1e-9 else None))
    forcing = forcing_calib
    extras = res_mod.Extras(invm=invm, pack=pack, forcing=forcing, mb_obs=mb_obs,
                            head=head, forcing_calib=forcing_calib)

    prim = physics.make_primaries(case, enc, mb.apply)
    prec = jnp.float64 if resolved.prec_ad == "float64" else jnp.float32
    ops = res_mod.make_residuals(cfg, case, spec, prim, env, prec_ad=prec, prec_fem=prec,
                                 extras=extras)
    scales = res_mod.calibrate_scales(cfg, groups, ops, prim, case, spec, mb.params0,
                                      seed=cfg.seed + 7, extras=extras)

    loss_fn = loss_mod.make_loss(cfg, groups, ops, prim, scales, case, centroids, extras)
    rows_fn = loss_mod.make_residual_rows(cfg, groups, ops, prim, scales, case, centroids, extras)

    steps = make_train_steps(cfg, resolved, loss_fn, rows_fn, mb, env)

    if verbose:
        rep = memplan.oom_report(cfg, resolved)
        print(f"[memplan] per-device ~{rep['total']:.0f} MB "
              f"(eig {rep['eig']:.0f} + ad {rep['ad']:.0f} + ctx {rep['ctx']:.0f} + "
              f"lbfgs {rep['lbfgs']:.0f} + fem {rep['fem']:.0f} + engd {rep['engd']:.0f}) "
              f"-> {'FITS' if rep['fits'] else 'OOM-RISK'} on a {rep['dev_mb']:.0f} MB card")
        print(f"[build] case={case_label(cfg)} ({case.n_cells} cells / {case.n_nodes} nodes / "
              f"{case.n_times} steps) | encoder dim_in={enc.dim_in} | arch={cfg.architecture.value} "
              f"| batches={resolved.batches}")

    return Bundle(cfg=cfg, resolved=resolved, env=env, case=case, spec=spec, model=mb, prim=prim,
                  ops=ops, scales=scales, centroids=centroids, loss_fn=loss_fn, rows_fn=rows_fn,
                  steps=steps, well_pack=pack)
