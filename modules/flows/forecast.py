r"""
Work path 2: the joint history match and forecast.

One optimization covers the whole horizon. The history — up to the model's
history | prediction boundary :math:`t_s` — is matched on the simulator's cell states and
well records; past it the field is constrained by the balances, by the schedule's
controls, and by its own continuity:

.. math::

    \mathcal{L}(\theta) \;=\;
      \mathcal{L}_{\mathrm{pde}}[0, t_{\mathrm{end}}]
    + \mathcal{L}_{\mathrm{ic}}(0)
    + \mathcal{L}_{\mathrm{data}}[0, t_s]
    + \mathcal{L}_{\mathrm{well}}[0, t_s]
    + \mathcal{L}_{\mathrm{ctrl}}[0, t_{\mathrm{end}}]
    + \mathcal{L}_{\mathrm{eq}}[0, t_{\mathrm{end}}]
    + \mathcal{L}_{\mathrm{mb}}[0, t_{\mathrm{end}}]

where:

- :math:`t_s`: the boundary, the time of report step ``n`` of the model's schedule; nothing the simulator computed after it enters the objective.
- :math:`\mathcal{L}_{\mathrm{ctrl}}`: the control rows of the predicted well model — each well's flowing pressure is a trained quantity, tied to the schedule's target rate, to its pressure limit, or shut, by the three-state switch.
- :math:`\mathcal{L}_{\mathrm{eq}}`: the phase-equilibrium rows, which keep the oil from holding more dissolved gas than its pressure allows where no cell state says so.
- :math:`\mathcal{L}_{\mathrm{mb}}`: the material-balance rows, the whole-reservoir balance the Galerkin test space (which drops the constant mode) does not impose.

The simulator's own run past the boundary is used afterwards, and only to score the
forecast.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import numpy as np

from .. import autopilot, frames
from ..autopilot import Budget
from . import common
from .history import well_errors

Item = frames.Item


def boundary_step(times: np.ndarray, settings: dict) -> int:
    r"""The number of report steps in the history, from the project's boundary.

    ``project.json`` carries ``history_end_days`` (days since the start of the schedule);
    the history is every report step up to it,

    .. math::

        n \;=\; \bigl|\{\, i : t_i \le t_{\mathrm{hist}} + \varepsilon \,\}\bigr|

    where:

    - :math:`t_i`: the report times of the simulated schedule.
    - :math:`t_{\mathrm{hist}}`: ``history_end_days``.
    - :math:`\varepsilon`: a rounding tolerance of a second.

    A model with no boundary, or with one that leaves no report step on either side, cannot
    be forecast, and the error says which.
    """
    if "history_end_days" not in settings:
        raise ValueError(
            "this model has no history | prediction boundary: set it on the schedule "
            "(Wells & schedule) before a history match with a forecast")
    t_hist = float(settings["history_end_days"])
    n = int(np.count_nonzero(np.asarray(times, np.float64) <= t_hist + 1.0 / 86400.0))
    if n < 2 or n >= len(times):
        raise ValueError(
            f"the history | prediction boundary at day {t_hist:g} leaves {n} report step(s) of "
            f"{len(times)} in the history; it must leave at least two there and one after it "
            f"(the schedule runs from day {float(times[0]):g} to day {float(times[-1]):g})")
    return n


def run(bundle: bytes, deck_name: str, budget: Budget = Budget(),
        work_root: str | Path | None = None) -> Iterator[Item]:
    """Simulate, extract, build and train the joint match and forecast; yield the messages."""
    t_start = time.time()
    project = common.unpack(bundle, deck_name, work_root)
    # The boundary is a report step, known only once the case is loaded: a provisional one
    # stands in until then (it changes neither the environment nor what the case holds).
    cfg = autopilot.forecast(project.deck_path, project.work, project.sha, 2, budget)
    common.setup(cfg)

    yield from common.simulate_stage(project)
    yield from common.prep_stage(project)

    from .. import casedata, inference, metrics as metrics_mod, observables, pipeline, training

    yield frames.stage("resolve", 0.38, "the boundary, the eigenbasis and the network capacity")
    full_case = casedata.load_case(cfg)
    times = np.asarray(full_case.times, np.float64)
    n_hist = boundary_step(times, project.settings())
    cfg = autopilot.forecast(project.deck_path, project.work, project.sha, n_hist, budget)
    cfg, resolved, _, _warnings = pipeline.resolve_run(cfg, case=full_case)
    t_split = inference.split_time(full_case, n_hist)
    yield frames.stage("build", 0.55, f"history to day {t_split:g} ({n_hist} of {len(times)} report "
                                      f"steps), forecast to day {float(times[-1]):g}; "
                                      + common.plan_message(cfg, resolved) + "; compiling")
    built, t_split = inference.build_joint_bundle(cfg, full_case, verbose=False)

    snap_idx = common.snapshot_steps(times, budget.n_snapshots)
    snap_days = [float(times[i]) for i in snap_idx]
    ranges, grid_items = common.grid_frames(full_case, snap_days, built.well_pack)
    yield from grid_items
    yield frames.stage("train", 0.6, f"training for {budget.n_iter} iterations")

    state = training.init_trainer(built)
    yield from common.train_frames(built, state, budget, t_start, snap_idx, snap_days, ranges)
    done = int(state.train_iter)

    # --- the wells over the whole horizon, each point marked history or forecast ----------
    simulated = observables.observed_wells(built.well_pack)
    predicted = observables.well_series(built, state.params)
    yield from common.well_frames(np.asarray(predicted["times"]), predicted["well_names"],
                                  simulator=simulated, gnn=predicted, split_days=t_split)

    hist, fut = slice(0, n_hist), slice(n_hist, None)
    history_rmse = [metrics_mod.field_rmse(built, state.params, "p_o", int(i)) for i in range(n_hist)]
    yield frames.metrics(done, {
        "history_end_days": float(t_split),
        "history_steps": int(n_hist),
        "history_pressure_rmse_mean_t": float(np.mean(history_rmse)),
        **{f"history_{k}": v for k, v in well_errors(simulated, predicted, hist).items()},
        **{f"forecast_{k.removeprefix('test_')}": float(v)
           for k, v in inference.score(built, state.params, n_hist).items()},
        **{f"forecast_{k}": v for k, v in well_errors(simulated, predicted, fut).items()},
        "supersaturated_share": metrics_mod.supersaturated_share(built, state.params),
        "iterations": done,
        "wall_s": time.time() - t_start,
    })
    yield frames.stage("done", 1.0, f"finished {done} iterations in {time.time() - t_start:.0f}s")
