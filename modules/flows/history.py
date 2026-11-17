r"""
Work path 1: the full-time history match.

The network is trained on everything the simulator's run of the model shows — the cell
states at every report step and the wells' records — with the black-oil balances enforced
through the spectral residual:

.. math::

    \mathcal{L}(\theta) \;=\;
      \mathcal{L}_{\mathrm{pde}}[0, t_{\mathrm{end}}]
    + \mathcal{L}_{\mathrm{ic}}(0)
    + \mathcal{L}_{\mathrm{data}}[0, t_{\mathrm{end}}]
    + w_{\mathrm{well}}\,\mathcal{L}_{\mathrm{well}}[0, t_{\mathrm{end}}]

where:

- :math:`\mathcal{L}_{\mathrm{pde}}`: the Galerkin-projected black-oil residual on the collocation slices of the whole history.
- :math:`\mathcal{L}_{\mathrm{ic}}`: the deck's initial state at :math:`t = 0`.
- :math:`\mathcal{L}_{\mathrm{data}}`: the simulator's cell states — pressure, saturations, dissolved-gas ratio — at every report step.
- :math:`\mathcal{L}_{\mathrm{well}}`: the bottom-hole pressures and phase rates of the wells' records, read from the network through the closed-form inflow relation.
- :math:`w_{\mathrm{well}} = 0.1`: at weight 1 the well rows cost the field fit more than they gain at the wells (:data:`modules.autopilot.WELL_WEIGHT`).

Because the states are observed everywhere, no closure row is needed: the composition's
phase-equilibrium and material-balance rows join only where the states are not all seen
(:mod:`modules.flows.forecast`).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import numpy as np

from .. import autopilot, frames
from ..autopilot import Budget
from . import common

Item = frames.Item


def run(bundle: bytes, deck_name: str, budget: Budget = Budget(),
        work_root: str | Path | None = None) -> Iterator[Item]:
    """Simulate, extract, build and match the whole history; yield the messages as they happen."""
    t_start = time.time()
    project = common.unpack(bundle, deck_name, work_root)
    cfg = autopilot.history(project.deck_path, project.work, project.sha, budget)
    common.setup(cfg)

    yield from common.simulate_stage(project)
    yield from common.prep_stage(project)

    from .. import metrics as metrics_mod, observables, pipeline, training

    yield frames.stage("resolve", 0.38, "eigenbasis and network capacity")
    cfg, resolved, case, _warnings = pipeline.resolve_run(cfg)
    yield frames.stage("build", 0.55, common.plan_message(cfg, resolved) + "; compiling")
    built = pipeline.build_pipeline(cfg, case=case, verbose=False)

    times = np.asarray(case.times, np.float64)
    snap_idx = common.snapshot_steps(times, budget.n_snapshots)
    snap_days = [float(times[i]) for i in snap_idx]
    ranges, grid_items = common.grid_frames(case, snap_days, built.well_pack)
    yield from grid_items
    yield frames.stage("train", 0.6, f"training for {budget.n_iter} iterations")

    state = training.init_trainer(built)
    yield from common.train_frames(built, state, budget, t_start, snap_idx, snap_days, ranges)
    done = int(state.train_iter)

    # --- the wells: the simulator's record against what the trained field implies ---------
    simulated = observables.observed_wells(built.well_pack)
    predicted = observables.well_series(built, state.params)
    yield from common.well_frames(np.asarray(simulated["times"]), simulated["well_names"],
                                  simulator=simulated, gnn=predicted)

    _, series = metrics_mod.rmse_series(built, state.params, max_times=8)
    yield frames.metrics(done, {
        "pressure_rmse_final_t": metrics_mod.pressure_rmse(built, state.params),
        "pressure_rmse_mean_t": float(np.mean(series)),
        "swat_rmse_final_t": metrics_mod.field_rmse(built, state.params, "S_w"),
        "sgas_rmse_final_t": metrics_mod.field_rmse(built, state.params, "S_g"),
        **well_errors(simulated, predicted),
        "iterations": done,
        "wall_s": time.time() - t_start,
    })
    yield frames.stage("done", 1.0, f"finished {done} iterations in {time.time() - t_start:.0f}s")


def well_errors(simulated: dict, predicted: dict, steps: slice = slice(None)) -> dict[str, float]:
    r"""Root-mean-square differences between the simulator's well record and the network's,
    over the report steps ``steps`` at which the simulator reports a value:

    .. math::

        \mathrm{RMS}_X \;=\; \sqrt{\frac{1}{|\mathcal{O}_X|}\sum_{(t, w) \in \mathcal{O}_X}
            \bigl(X^{\mathrm{gNN}}_{t,w} - X^{\mathrm{sim}}_{t,w}\bigr)^2}

    where:

    - :math:`X`: the bottom-hole pressure [psi] (``bhp_rms``), the oil rate [stb/d] (``wopr_rms``), the gas-oil ratio [Mscf/stb] (``wgor_rms``) or the water cut (``wwct_rms``).
    - :math:`\mathcal{O}_X`: the (report step, well) pairs at which the simulator's summary carries :math:`X`.
    """
    out: dict[str, float] = {}
    for key, name in (("WBHP", "bhp_rms"), ("WOPR", "wopr_rms"), ("WGOR", "wgor_rms"), ("WWCT", "wwct_rms")):
        if key not in simulated or key not in predicted:
            continue
        sim = np.asarray(simulated[key], np.float64)[steps]
        net = np.asarray(predicted[key], np.float64)[steps]
        ok = np.isfinite(sim) & np.isfinite(net)
        if ok.any():
            out[name] = float(np.sqrt(np.mean((net[ok] - sim[ok]) ** 2)))
    return out
