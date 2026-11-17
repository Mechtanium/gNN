r"""
Forward inference on the held-out time window.

Training fits :math:`[0, t_s]`; inference reports on :math:`[t_s, t_{\mathrm{end}}]`
under a well schedule the user imposes. Two modes, chosen by
``cfg.inference.mode``:

**frozen** evaluates the trained field as-is. The parameters do not move, so the
imposed schedule reaches only the well closure that turns predicted cell states
into observables — useful as the baseline that isolates what training alone
already extrapolates.

**constrained** continues to optimize on the inference window under the
unsupervised terms only,

.. math::

    \mathcal{L}_{\mathrm{inf}}(\theta)
    \;=\;
    w_{\mathrm{pde}}\,\mathcal{L}_{\mathrm{pde}}
    \;+\;
    w_{\mathrm{bc}}\,\mathcal{L}_{\mathrm{bc}}
    \;+\;
    w_a\,\mathcal{L}_{\mathrm{anchor}}(t_s)

where:
- :math:`\mathcal{L}_{\mathrm{pde}}`: the black-oil residual, into which the schedule enters as the interior source :math:`Q(t)` — the only channel through which an imposed control can change the predicted field.
- :math:`\mathcal{L}_{\mathrm{bc}}`: the no-flow boundary term.
- :math:`\mathcal{L}_{\mathrm{anchor}}`: the split-time anchor, the ``ic`` group relocated from :math:`t = 0` to :math:`t_s`; without it the window is an under-determined boundary-value problem.
- :math:`w_a`: ``inference.anchor_weight``.

Neither mode admits a cell-state or well **observation** on the window: the
``data`` and ``well`` groups are dropped, and the reference states are touched
only afterwards, to score the forecast. Dropping ``well`` rather than masking it
is deliberate — an all-zero row mask leaves the group with no rows at all, and
the mean of an empty residual array is NaN. Under ``well_model=predicted`` the
``ctrl`` group survives, so an imposed schedule reaches the physics through the
control rows as well as through the predicted source.

**joint** replaces the two stages by one optimization over the whole horizon
(:func:`build_joint_bundle`): the collocation window is :math:`[0, t_{\mathrm{end}}]`,
cell-state and well observation rows stop at :math:`t_s`, the schedule controls
(``ctrl`` rows) run to :math:`t_{\mathrm{end}}`, and the initial condition is the
deck's at :math:`t = 0`,

.. math::

    \mathcal{L}
    \;=\;
    w_{\mathrm{pde}}\,\mathcal{L}_{\mathrm{pde}}[0, t_{\mathrm{end}}]
    + w_{\mathrm{ic}}\,\mathcal{L}_{\mathrm{ic}}(0)
    + w_{\mathrm{data}}\,\mathcal{L}_{\mathrm{data}}[0, t_s]
    + w_{\mathrm{well}}\,\mathcal{L}_{\mathrm{well}}[0, t_s]
    + w_{\mathrm{ctrl}}\,\mathcal{L}_{\mathrm{ctrl}}[0, t_{\mathrm{end}}]

where:
- :math:`[0, t_s]`: the history window; beyond it the field is constrained by the PDE, the controls and its own continuity — the forecast is *envisaged* by the same optimization that matches the history.
- no split-time anchor exists: it is one continuous field, so the ``ic`` group keeps its :math:`t = 0` meaning.
"""

from __future__ import annotations

import dataclasses

import numpy as onp

from .config import ObservationDesign, ResidualDesign, RunConfig


def split_time(case, n: int) -> float:
    """The boundary between the halves [days]: ``times[n-1]``, or 0 when nothing is trained."""
    return float(onp.asarray(case.times)[n - 1]) if n > 0 else 0.0


def check_params_match(bundle, params, *, what: str = "parameters") -> None:
    r"""
    Fail early, and legibly, when ``params`` do not fit ``bundle``'s network.

    The inference stage deliberately hands one parameter set to a *second* bundle,
    so a config edited between the training cell and the inference cell produces a
    silent shape mismatch. Left unchecked it surfaces far downstream as a bare
    ``dot_general requires contracting dimensions to have the same shape`` from
    inside an ENGD jacobian sweep, which says nothing about the cause. This
    compares the pytree and every leaf shape up front and names the likely culprit
    (``n_eig``/``dim_in``, or width/depth) instead.
    """
    import jax

    want = jax.tree.leaves(bundle.model.params0)
    got = jax.tree.leaves(params)
    if len(want) != len(got):
        raise ValueError(
            f"{what} do not match this bundle's architecture: {len(got)} parameter "
            f"array(s) supplied, {len(want)} expected. The two bundles were built from "
            "different architecture fields (m_width / n_blocks / mlp_hidden / dim_out)."
        )
    bad = [(i, tuple(g.shape), tuple(w.shape))
           for i, (g, w) in enumerate(zip(got, want)) if g.shape != w.shape]
    if not bad:
        return
    i, g_shape, w_shape = bad[0]
    hint = ""
    if g_shape and w_shape and g_shape[0] != w_shape[0] and i == 0:
        hint = (f"\n  The first axis is the network input width dim_in = n_eig + 1: the "
                f"supplied set was trained at dim_in={g_shape[0]} (n_eig={g_shape[0] - 1}) "
                f"but this bundle resolved dim_in={w_shape[0]} (n_eig={w_shape[0] - 1}). "
                "In a notebook this usually means the config cell was edited and the "
                "training cell was not re-run, so the parameters predate the current "
                "n_eig / n_eig_z.")
    raise ValueError(
        f"{what} do not fit this bundle: {len(bad)} array(s) differ in shape, first at "
        f"leaf {i} (supplied {g_shape}, expected {w_shape}).{hint}"
    )


def inference_cfg(cfg: RunConfig) -> RunConfig:
    r"""
    ``cfg`` with every supervised group switched off for the inference window.

    ``observation`` drops to ``NONE``, removing **both** the ``data`` and the ``well``
    groups: on the held-out half the cell states and the well responses are precisely
    what is being predicted, so either one entering the objective would leak the
    answer. What remains -- PDE, boundary and the split-time anchor -- is exactly the
    forward problem. A ``DATA_ONLY`` composition has nothing left once observations
    are gone, so it is rejected rather than silently reduced to an empty objective.

    The group weights are rebuilt rather than inherited. On the inference window the
    ``ic`` group *is* the split-time anchor, so it takes ``inference.anchor_weight``:

    .. math::

        w_g =
        \begin{cases}
        w_a, & g = \texttt{ic} \\
        w_g^{\mathrm{def}}, & \text{otherwise}
        \end{cases}

    where:
    - :math:`w_a`: ``inference.anchor_weight``, the strength of the anchor at :math:`t_s`.
    - :math:`w_g^{\mathrm{def}}`: the group's default weight.

    Inheriting the training run's ``w_fixed`` instead would be actively wrong: a
    training config that silences ``ic`` (a PDE-only fit, say) would silence the
    anchor too, leaving the inference window with no initial condition and therefore
    no unique solution. The tuple is also re-sized here, since the active groups
    differ between the two stages and ``w_fixed`` is positional.
    """
    from .config import DEFAULT_GROUP_WEIGHTS, loss_groups

    if cfg.residual_design is ResidualDesign.DATA_ONLY:
        raise ValueError("forward inference needs a PDE residual; residual_design=DATA_ONLY "
                         "leaves nothing to impose a schedule on")
    icfg = dataclasses.replace(cfg, observation=ObservationDesign.NONE, invert=())
    w_a = float(cfg.inference.anchor_weight)
    w = tuple(w_a if g == "ic" else DEFAULT_GROUP_WEIGHTS[g] for g in loss_groups(icfg))
    return dataclasses.replace(icfg, w_fixed=w)


def build_inference_bundle(cfg: RunConfig, case, params=None, *, train_bundle=None,
                           device_count=None, verbose: bool = True):
    r"""
    A bundle bound to the inference window, with the schedule imposed.

    Steps: carve the test half (anchored at :math:`t_s` on ``params`` when a trained
    set is supplied), build the pipeline against it, then patch the well pack's
    controls with ``cfg.inference.schedules`` and rebuild the interior forcing so
    the imposed rates actually reach the PDE.
    """
    from . import casedata, pipeline

    n = int(cfg.train_split_n)
    t_s = split_time(case, n)

    y_anchor = None
    if n > 0 and params is not None:
        from .metrics import predict_cells

        # The trained field's own state at the split time becomes the window's IC.
        # Prefer the caller's training bundle: rebuilding one here costs a second
        # eigenbasis provision and, worse, could resolve differently from the bundle
        # the parameters were actually trained in.
        src = train_bundle
        if src is None:
            src = pipeline.build_pipeline(cfg, case=casedata.split_case(case, n, "train"),
                                          verbose=False, preflight=False)   # evaluated only
        check_params_match(src, params, what="the trained parameters")
        y_anchor = onp.asarray(predict_cells(src, params, t_s))

    icfg = inference_cfg(cfg)
    test_case = casedata.split_case(case, n, "test", y_anchor=y_anchor)
    bundle = pipeline.build_pipeline(icfg, case=test_case, device_count=device_count,
                                     verbose=verbose)

    # build_well_pack applies cfg.inference.schedules itself, keyed off the case's
    # infer_half marker, so the imposed controls are already in the forcing the
    # residual closed over. Nothing to patch here.
    if params is not None:
        check_params_match(bundle, params, what="the trained parameters")
    if verbose and cfg.inference.schedules:
        print(f"[inference] imposed {len(cfg.inference.schedules)} schedule record(s) "
              f"from t={t_s:.4g} d")
    return bundle, t_s


def build_joint_bundle(cfg: RunConfig, case, *, device_count=None, verbose: bool = True):
    r"""
    The single-stage history-match + forecast bundle (``inference.mode == "joint"``).

    Carves the joint window with :func:`modules.casedata.split_case` (``"joint"``):
    collocation over :math:`[0, t_{\mathrm{end}}]`, observations to :math:`t_s`,
    imposed schedules from :math:`t_s`, and builds the pipeline against it. Returns
    ``(bundle, t_s)``. Under ``well_model=closed_form`` the interior source is the
    mixed data forcing of :func:`modules.wells.build_forcing` (observed rates to
    :math:`t_s`, controls after) and no ``ctrl`` rows exist, so the forecast is the
    ``constrained`` continuation without its anchor — valid for wells that stay on
    rate control; the predicted model is the general case. Training then runs exactly as in ``PINN-Lab.ipynb``
    (:func:`modules.training.init_trainer` + :func:`modules.training.train`).
    """
    from . import casedata, pipeline

    if cfg.inference.mode != "joint":
        raise ValueError(f"build_joint_bundle needs inference.mode='joint' (got {cfg.inference.mode!r})")
    n = int(cfg.train_split_n)
    t_s = split_time(case, n)
    joint_case = casedata.split_case(case, n, "joint")
    bundle = pipeline.build_pipeline(cfg, case=joint_case, device_count=device_count,
                                     verbose=verbose)
    if verbose:
        lo, hi = joint_case.t_span()
        _, obs = joint_case.t_obs_span()
        print(f"[joint] collocation [{lo:.4g}, {hi:.4g}] d | observations to {obs:.4g} d | "
              f"{len(cfg.inference.schedules)} imposed schedule record(s) from {t_s:.4g} d | "
              f"time strata {joint_case.time_strata(cfg.n_tslice, cfg.inference.n_tslice_forecast)}")
    return bundle, t_s


def run(bundle, params, cfg: RunConfig, *, log_every: int = 50):
    """Execute the configured mode; returns ``(params, train_result_or_None)``."""
    mode = cfg.inference.mode
    if mode == "frozen":
        return params, None
    if mode != "constrained":
        raise ValueError(f"unknown inference mode {mode!r}")

    from . import training

    check_params_match(bundle, params, what="the parameters handed to inference")
    state = training.init_trainer_from_params(bundle, params)
    result = training.train(bundle, state, n_iter=cfg.inference.n_iter, log_every=log_every)
    return state.params, result


def score(bundle, params, n: int) -> dict[str, float]:
    """
    RMSE of the forecast against reference states on the inference window only, and of
    the well predictions against the simulator's well records there
    (:func:`modules.wells.forecast_match_metrics`).

    These never entered a loss in either mode, so they measure extrapolation rather
    than fit — and are not comparable with the whole-history columns of runs.csv. A
    failure of the well scoring is reported and leaves the field scores in place: it must
    not cost the trained run.
    """
    from .metrics import field_rmse

    case = bundle.case
    idx = list(range(max(int(n), 0), case.n_times))
    if not idx:
        return {}
    out: dict[str, float] = {}
    for field, key in (("p_o", "pressure"), ("S_w", "swat"), ("S_g", "sgas")):
        vals = [field_rmse(bundle, params, field, int(i)) for i in idx]
        out[f"test_{key}_rmse_mean_t"] = float(onp.mean(vals))
        if key == "pressure":
            out["test_pressure_rmse_final_t"] = float(vals[-1])
    if getattr(bundle.ops, "well_predict", None) is not None:
        import jax
        import jax.numpy as jnp

        from .wells import forecast_match_metrics

        try:
            p_bh_hat, q_hat = bundle.ops.well_predict(
                jax.tree.map(lambda x: x.astype(jnp.float32), params))
            out.update(forecast_match_metrics(bundle.cfg, case, onp.asarray(p_bh_hat),
                                              onp.asarray(q_hat), n))
        except Exception as e:  # noqa: BLE001 - the trained run must survive its scoring
            import warnings

            warnings.warn(f"forecast well scoring skipped: {type(e).__name__}: {e}",
                          RuntimeWarning, stacklevel=2)
    return out
