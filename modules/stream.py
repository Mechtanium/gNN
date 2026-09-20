r"""
The workflow's spine: from a deck on disk to a stream of messages.

:func:`run_training` is a plain synchronous generator. It walks the stages —
simulate (OPM Flow), extract (ResInsight), resolve (eigenbasis + capacity),
build (operators, encoder, network, residuals, loss, steps) — yielding a
``stage`` message at each, then the ``grid`` message so the page can draw the
reservoir before the first optimizer step, and then trains. Because
:func:`modules.training.train` continues from ``state.train_iter`` on the same
:class:`~modules.training.TrainerState` with the compiled steps cached, the loop
below calls it one iteration at a time and yields a ``loss`` message per
iteration; every ``snapshot_every`` iterations (and at the end) it evaluates the
network at ``n_snapshots`` report times over every cell and yields one ``state``
message per time. The operation in ``workflow.py`` pulls each step of this
generator on a worker thread so the gRPC server keeps answering.

The seven cell fields of a snapshot are

.. math::

    S_w,\; S_o = 1 - S_w - S_g,\; S_g,\; p_o,\; p_w = p_o - p_{cow}(S_w),\;
    p_g = p_o + p_{cgo}(S_g),\; R_{so}

where:

- :math:`p_o, S_w, S_g, R_{so}`: the network's four primaries at the cell centroid (:func:`modules.metrics.predict_cells`).
- :math:`p_{cow}, p_{cgo}`: the deck's capillary-pressure tables (:mod:`modules.observables`).
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import frames
from .config import RunConfig

Item = frames.Item


def work_dir_for(deck_bytes: bytes, root: str | Path | None = None) -> tuple[Path, str]:
    """``(work_dir, deck_sha)``: one directory per distinct deck, so a second job on
    the same deck reuses the simulation, the cache and the eigenbasis."""
    sha = hashlib.sha1(deck_bytes).hexdigest()
    root = Path(root or os.environ.get("DELTA_PINN_WORK", "/tmp/delta_pinn"))
    return root / sha[:12], sha


def materialize_deck(deck_bytes: bytes, deck_name: str, root: str | Path | None = None) -> tuple[Path, Path, str]:
    """Write the deck under its work dir (unchanged bytes) and return
    ``(deck_path, work_dir, deck_sha)``."""
    name = Path(deck_name or "deck.DATA").name
    if not name.upper().endswith(".DATA"):
        name = f"{name}.DATA"
    work, sha = work_dir_for(deck_bytes, root)
    case_dir = work / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    deck_path = case_dir / name
    if not deck_path.is_file() or deck_path.read_bytes() != deck_bytes:
        deck_path.write_bytes(deck_bytes)
    return deck_path, work, sha


def configure(base: RunConfig, deck_path: Path, work_dir: Path, deck_sha: str, **overrides: Any) -> RunConfig:
    """Bind a preset to a deck; ``overrides`` with value 0/None keep the preset's value."""
    fields = {k: v for k, v in overrides.items() if v not in (None, 0)}
    return replace(base, deck_path=str(deck_path), work_dir=str(work_dir), deck_sha=deck_sha, **fields)


def snapshot_times(times: np.ndarray, n_snapshots: int) -> list[int]:
    """Indices of ``n_snapshots`` report steps spread over the history, last one included."""
    n = int(len(times))
    k = max(1, min(int(n_snapshots), n))
    idx = np.unique(np.linspace(0, n - 1, k).round().astype(int))
    return [int(i) for i in idx]


def snapshot_fields(bundle, params, t: float) -> dict[str, np.ndarray]:
    """The seven cell fields at time ``t`` (days)."""
    from . import metrics, observables

    P = np.asarray(metrics.predict_cells(bundle, params, float(t)))
    p_o, p_w, p_g = observables._phase_pressures(bundle, P)
    s_w, s_g, rs = P[:, 1], P[:, 2], P[:, 3]
    return {"S_w": s_w, "S_o": 1.0 - s_w - s_g, "S_g": s_g,
            "p_o": p_o, "p_w": p_w, "p_g": p_g, "R_so": rs}


def well_markers(bundle) -> list[dict[str, Any]]:
    """Perforated cells per well, for the 3-D view."""
    pack = bundle.well_pack
    if pack is None:
        return []
    out = []
    for w, name in enumerate(pack.well_names):
        cells = [int(c) for c, wid in zip(pack.cell_idx, pack.well_id) if int(wid) == w]
        out.append({"name": str(name), "cells": cells})
    return out


def run_training(deck_bytes: bytes, deck_name: str, base: RunConfig, *, n_iter: int,
                 log_every: int = 1, snapshot_every: int = 50, n_snapshots: int = 8,
                 work_root: str | Path | None = None, **overrides: Any) -> Iterator[Item]:
    """Simulate, extract, build and train; yield the messages as they happen."""
    from . import bootstrap

    t_start = time.time()
    deck_path, work, sha = materialize_deck(deck_bytes, deck_name, work_root)
    cfg = configure(base, deck_path, work, sha, n_iter=n_iter, **overrides)
    # Before anything imports jax (the extraction's closures do): the x64 flag and
    # the platform are process-wide and cannot be changed once jax is loaded.
    bootstrap.setup_environment(cfg.precision_policy, cfg.opt_f64, platforms=bootstrap.platforms_for_host())

    # --- stage A: the reference simulation ------------------------------------------------
    yield frames.stage("simulate", 0.02, f"running OPM Flow on {deck_path.name}")
    from . import simulate
    notes: list[str] = []
    secs = simulate.ensure_results(deck_path, progress=notes.append)
    yield frames.stage("simulate", 0.15, f"simulation ready ({secs:.0f}s)" if secs else "simulation results reused")

    # --- stage B: extraction through ResInsight -------------------------------------------
    yield frames.stage("prep", 0.18, "extracting the grid, states and wells through ResInsight")
    from . import prep
    prep.build_prep_cache(deck_path, Path(cfg.work_dir) / "prep_cache")
    yield frames.stage("prep", 0.35, "preprocessing cache ready")

    # --- stage C: resolve + build ----------------------------------------------------------
    from . import pipeline, training
    yield frames.stage("resolve", 0.38, "eigenbasis and network capacity")
    cfg, resolved, case, warns = pipeline.resolve_run(cfg)
    yield frames.stage("build", 0.55, f"n_eig={cfg.n_eig} width={cfg.m_width} depth={cfg.n_blocks} "
                                      f"P={resolved.param_count:,}; compiling")
    bundle = pipeline.build_pipeline(cfg, case=case, verbose=False)

    # --- the grid, before the first step ------------------------------------------------
    mesh = case.art.reservoir_mesh
    dims = tuple(int(d) for d in np.asarray(mesh.active_mask).shape) if getattr(mesh, "active_mask", None) is not None \
        else tuple(int(x) + 1 for x in np.asarray(case.aci).max(axis=0))
    snap_idx = snapshot_times(np.asarray(case.times), n_snapshots)
    snap_days = [float(case.times[i]) for i in snap_idx]
    ranges = frames.field_ranges(
        {"pres": case.pres, "swat": case.swat, "sgas": case.sgas, "rs": case.rs}, case.phys)
    yield from frames.grid(np.asarray(mesh.corner_cells), np.asarray(case.aci), dims, snap_days, ranges,
                           wells=well_markers(bundle))
    yield frames.stage("train", 0.6, f"training for {n_iter} iterations")

    # --- stage D: train, one iteration at a time -----------------------------------------
    state = training.init_trainer(bundle)
    params_for_snapshot = lambda: state.params
    log_every = max(1, int(log_every))

    def emit_snapshots(it: int):
        p = params_for_snapshot()
        for k, i in enumerate(snap_idx):
            yield frames.state(it, k, snap_days[k], snapshot_fields(bundle, p, snap_days[k]), ranges)

    yield from emit_snapshots(0)          # the untrained network, so the view is never empty
    done = 0
    while done < n_iter:
        step = min(log_every, n_iter - done)
        res = training.train(bundle, state, n_iter=step, log_every=10**9, stop_on_eta_zero=True)
        done = int(state.train_iter)
        h = state.hist
        it = int(h["iter"][-1])
        comps = {g: float(h[g][-1]) for g in ("pde", "ic", "data", "well") if g in h and h[g]}
        yield frames.loss(it, float(h["total"][-1]), comps.get("pde", 0.0), comps.get("ic", 0.0),
                          comps.get("data", 0.0), comps.get("well", 0.0),
                          wall_s=time.time() - t_start,
                          eta=(float(h["engd_eta"][-1]) if h.get("engd_eta") else None))
        if snapshot_every > 0 and done % snapshot_every == 0 and done < n_iter and res.status == "completed":
            yield from emit_snapshots(done)
        if res.status != "completed":
            yield frames.stage("train", 0.95, f"stopped early: {res.status} at iteration {done}")
            break

    # --- final snapshots and metrics ----------------------------------------------------
    yield from emit_snapshots(done)
    from . import metrics as metrics_mod
    _, series = metrics_mod.rmse_series(bundle, state.params, max_times=8)
    yield frames.metrics(done, {
        "pressure_rmse_final_t": metrics_mod.pressure_rmse(bundle, state.params),
        "pressure_rmse_mean_t": float(np.mean(series)),
        "swat_rmse_final_t": metrics_mod.field_rmse(bundle, state.params, "S_w"),
        "sgas_rmse_final_t": metrics_mod.field_rmse(bundle, state.params, "S_g"),
        "iterations": done,
        "wall_s": time.time() - t_start,
    })
    yield frames.stage("done", 1.0, f"finished {done} iterations in {time.time() - t_start:.0f}s")
