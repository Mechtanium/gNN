r"""
The pipeline glue: validate → resolve → load → wire.

:func:`build_pipeline` assembles one :class:`Bundle` from a
:class:`~modules.config.RunConfig` (with the bit-parity and float32 init gates
asserted along the way); :mod:`modules.stream` drives training on it and
streams the losses and cell states out.
"""

from __future__ import annotations

import dataclasses

from dataclasses import dataclass, replace
from typing import Any, Callable

from . import (casedata, encodings, loss as loss_mod, meshenv, models, memplan, physics,
               residuals as res_mod, sampling as sampling_mod, spectral, weighting,
               wells as wells_mod)
from .config import (ResidualDesign, Resolved, RunConfig, SamplingDesign, WeightingDesign,
                     case_label, loss_groups, needs_fem_static, predicted_well, resolve,
                     resolve_capacity, structural_hash, validate)
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
    ntk_core: Callable | None
    ntk_weights: Callable | None
    adapt_select: Callable | None
    steps: Steps
    well_pack: Any = None          # modules.wells.WellPack | None
    well_head: Any = None          # modules.wells.WellHead | None (predicted well model)


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


def audit_window_bounds(cfg, resolved, case, env) -> list[str]:
    r"""
    Confirm a sampled window really lies inside the case's collocation span.

    Draws one window and checks every time it carries against
    :math:`[t_{\mathrm{lo}}, t_{\mathrm{hi}}]`:

    .. math::

        t \in [t_{\mathrm{lo}} - \varepsilon,\; t_{\mathrm{hi}} + \varepsilon]
        \qquad \forall\, t \in \{t_p\} \cup \{t_d\} \cup \{t_b\}

    where:
    - :math:`t_p, t_d, t_b`: the PDE-collocation, cell-state and boundary times of one sampled window.
    - :math:`\varepsilon`: a float32 rounding tolerance on the span.

    This is the last of three defences against a half-applied split (the others
    being the single draw helper in :mod:`pinnlab.sampling` and its static test).
    A collocation time escaping into the held-out window is silent otherwise: the
    run trains happily while fitting physics on the data it is meant to forecast.
    Returns warning strings, which the caller folds into ``resolved.warnings`` so
    they reach both the notebook preview and the ``warnings`` column of runs.csv.
    """
    import jax
    import numpy as onp

    from . import sampling as sampling_mod

    lo, hi = case.t_span()
    _, obs_hi = case.t_obs_span()
    eps = max(1e-3, 1e-6 * max(abs(hi), 1.0))
    win = sampling_mod.random_window(cfg, case, resolved, env, jax.random.PRNGKey(0))
    out: list[str] = []
    for name in ("t_p", "t_d", "t_b"):
        t = onp.asarray(getattr(win, name), onp.float64).ravel()
        if t.size == 0:
            continue
        if t.min() < lo - eps or t.max() > hi + eps:
            out.append(
                f"TIME-SPLIT LEAK: sampled {name} spans [{t.min():.4g}, {t.max():.4g}] but the "
                f"collocation window is [{lo:.4g}, {hi:.4g}]; this run is drawing points from "
                "the held-out half"
            )
    # joint mode: the collocation legitimately spans the horizon, but supervised
    # cell-state rows must stop at the observation window end
    t_d = onp.asarray(win.t_d, onp.float64).ravel()
    if obs_hi < hi - eps and t_d.size and t_d.max() > obs_hi + eps:
        out.append(
            f"TIME-SPLIT LEAK: cell-state rows reach t={t_d.max():.4g} beyond the observation "
            f"window end {obs_hi:.4g}; the history match is reading the forecast's answer"
        )
    return out


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
    if cfg.train_split_n >= 0:
        leaks = audit_window_bounds(cfg, resolved, case, env)
        if leaks:
            resolved = dataclasses.replace(
                resolved, warnings=tuple(resolved.warnings) + tuple(leaks))
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

    spec = None
    if needs_fem_static(cfg):
        spec = spectral.place_on_mesh(spectral.provision(case, cfg), cfg, env)
    centroids = spec.centroids if spec is not None else jnp.asarray(case.centroids, jnp.float32)

    enc = encodings.make_encoder(cfg, case, spec)
    mb = models.build_model(cfg, resolved.dim_in, seed=cfg.seed)

    # --- init gates: bit parity vs jaxpinns + all-f32 leaves (both policies) --------------
    probe = jax.random.normal(jax.random.PRNGKey(99), (4, resolved.dim_in), jnp.float32)
    y_nnx = jax.vmap(lambda f: mb.apply(mb.params0, f))(probe)
    y_raw = jax.vmap(lambda f: mb.raw_apply(mb.raw_params0, f))(probe)
    assert onp.array_equal(onp.asarray(y_nnx), onp.asarray(y_raw)), "NNX/raw bit-parity gate failed"
    assert all(l.dtype == jnp.float32 for l in jax.tree.leaves(mb.params0)), "f32 init gate failed"

    # The well pack backs BOTH the observation rows (well group) and the interior
    # realized-rate forcing Q(t), so it exists for every composition with a PDE term.
    invm = None
    pack = forcing = forcing_calib = mb_obs = head = None
    needs_pack = (cfg.residual_design is not ResidualDesign.DATA_ONLY
                  or "well" in groups)
    if needs_pack:
        pack = wells_mod.build_well_pack(cfg, case)
        if verbose:
            n_bind = int((pack.binding > 0).sum())
            print(f"[wells] {pack.n_wells} wells / {pack.n_perf} perfs / {pack.n_times} steps "
                  f"-> {pack.n_rows_well} observation rows + {pack.n_rows_ctrl} control rows, "
                  f"{n_bind} BHP-limited (well, step) pairs"
                  + (f" (SYNTHESIZED: {pack.synth_reason})" if pack.synthesized else " (summary)"))
        if cfg.residual_design is not ResidualDesign.DATA_ONLY:
            # The data-driven forcing: the closed-form model's interior source, and the
            # scale-calibration forcing of the predicted model (see calibrate_scales). A
            # two-stage forecast window drives it from controls, never from the
            # simulator's future observed rates (see build_forcing).
            _, obs_hi = case.t_obs_span()
            _, coll_hi = case.t_span()
            forcing_calib = wells_mod.build_forcing(
                pack, case, from_controls=bool(getattr(case, "infer_half", False)),
                controls_from=(obs_hi if obs_hi < coll_hi - 1e-9 else None))
            if predicted_well(cfg):
                head = wells_mod.WellHead.from_pack(pack, case)
                mb = wells_mod.wrap_model_with_head(mb, head)   # params["well"]["pwf_raw"]
                if verbose:
                    print(f"[wells] predicted well model: p_wf head with "
                          f"{pack.n_times * pack.n_wells} parameters, ctrl_switch={cfg.ctrl_switch}")
            else:
                forcing = forcing_calib
    extras = res_mod.Extras(invm=invm, pack=pack, forcing=forcing, mb_obs=mb_obs,
                            head=head, forcing_calib=forcing_calib)

    prim = physics.make_primaries(case, enc, mb.apply,
                                  ic_tau=(float(cfg.ic_tau_days) if cfg.ic_design == "hard" else None))
    prec = jnp.float64 if resolved.prec_ad == "float64" else jnp.float32
    ops = res_mod.make_residuals(cfg, case, spec, prim, env, prec_ad=prec, prec_fem=prec,
                                 extras=extras)
    scales = res_mod.calibrate_scales(cfg, groups, ops, prim, case, spec, mb.params0,
                                      seed=cfg.seed + 7, extras=extras)

    loss_fn = loss_mod.make_loss(cfg, groups, ops, prim, scales, case, centroids, extras)
    rows_fn = loss_mod.make_residual_rows(cfg, groups, ops, prim, scales, case, centroids, extras)

    ntk_core = ntk_weights = None
    if cfg.weighting is WeightingDesign.NTK:
        ntk_core = weighting.make_ntk(cfg, groups, ops, prim, scales, case, centroids,
                                      case.n_nodes)
        ntk_weights = jax.jit(ntk_core)

    adapt_select = None
    if cfg.sampling is SamplingDesign.RAR:
        adapt_select = sampling_mod.make_adapt_select(cfg, groups, resolved, ops, prim, scales,
                                                      ntk_core, case, centroids, env, case.n_nodes,
                                                      extras=extras)

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
                  ntk_core=ntk_core, ntk_weights=ntk_weights, adapt_select=adapt_select,
                  steps=steps, well_pack=pack, well_head=head)
