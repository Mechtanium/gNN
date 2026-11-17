r"""
The hand-off to the simulator: a model in, its simulated history out.

The deck is run through OPM Flow, its results are extracted through ResInsight, and the
job streams what a reservoir engineer looks at first after a run: the grid, the seven cell
fields at report times spread over the history, and every well's record — bottom-hole
pressure, the phase rates, the gas-oil ratio

.. math::

    \mathrm{WGOR} = \frac{\mathrm{WGPR}}{\mathrm{WOPR}}
    \qquad\text{and the water cut}\qquad
    \mathrm{WWCT} = \frac{\mathrm{WWPR}}{\mathrm{WWPR} + \mathrm{WOPR}}

where:

- :math:`\mathrm{WOPR}, \mathrm{WWPR}, \mathrm{WGPR}`: the surface oil, water and gas production rates [stb/d, stb/d, Mscf/d].

Everything it computes is cached under the deck's work directory, so the paths that
train on this model afterwards start from the extraction, not from the simulator.
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
    """Simulate the bundle's deck and stream its grid, cell states and well records."""
    t_start = time.time()
    project = common.unpack(bundle, deck_name, work_root)
    cfg = autopilot.history(project.deck_path, project.work, project.sha, budget)
    common.setup(cfg)

    yield from common.simulate_stage(project, 0.02, 0.60)
    yield from common.prep_stage(project, 0.62, 0.85)

    from .. import casedata, observables, wells as wells_mod

    yield frames.stage("results", 0.88, "reading the simulated history")
    case = casedata.load_case(cfg)
    pack = wells_mod.build_well_pack(cfg, case)
    times = np.asarray(case.times, np.float64)
    snap_idx = common.snapshot_steps(times, budget.n_snapshots)
    snap_days = [float(times[i]) for i in snap_idx]
    ranges, grid_items = common.grid_frames(case, snap_days, pack)
    yield from grid_items
    for k, step in enumerate(snap_idx):
        yield frames.state(0, k, snap_days[k], common.simulated_fields(case, step), ranges)

    record = observables.observed_wells(pack)
    yield from common.well_frames(np.asarray(record["times"]), record["well_names"], simulator=record)

    yield frames.metrics(0, {
        "cells": int(case.n_cells), "nodes": int(case.n_nodes), "report_steps": int(case.n_times),
        "days": float(times[-1]), "wells": int(pack.n_wells), "perforations": int(pack.n_perf),
        "pressure_min": float(np.nanmin(case.pres)), "pressure_max": float(np.nanmax(case.pres)),
        "wall_s": time.time() - t_start,
    })
    yield frames.stage("done", 1.0, f"{case.n_cells:,} cells, {case.n_times} report steps, "
                                    f"{pack.n_wells} wells in {time.time() - t_start:.0f}s")
