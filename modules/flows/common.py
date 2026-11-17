r"""
What the work paths share: the project a job receives, the hand-off to the simulator, the
extraction, and the messages every path sends about a reservoir.

A job's input is one **bundle**: the bytes of a self-contained ``.DATA`` deck, or a zip of
a deck tree (the deck, its ``INCLUDE`` files) with the project's own files beside it —
``project.json`` (the history | prediction boundary), ``observed/`` (a field record) and
``inversion/`` (a prior). The bundle is unpacked into a work directory named by the
digest of the **deck tree alone**,

.. math::

    h \;=\; \operatorname{SHA1}\Bigl(\,\big\Vert_{f \in \mathcal{D}}\;
        \text{name}(f)\,\Vert\,0\,\Vert\,\text{bytes}(f)\,\Vert\,0\Bigr)

where:

- :math:`\mathcal{D}`: the files the simulator reads, in sorted order of their paths — the project's own files are left out, so moving the boundary or adding a record does not change the digest.
- :math:`h`: the name of the work directory (its first twelve hex digits) and part of every cache key, so a second job on the same model reuses the simulation, the extraction, the eigenbasis and the compiled steps, and two models never share one.

The seven cell fields of a snapshot are

.. math::

    S_w,\; S_o = 1 - S_w - S_g,\; S_g,\; p_o,\; p_w = p_o - p_{cow}(S_w),\;
    p_g = p_o + p_{cgo}(S_g),\; R_{so}

where:

- :math:`p_o, S_w, S_g, R_{so}`: the four primaries at the cell centroid — the network's (:func:`modules.metrics.predict_cells`) or the simulator's.
- :math:`p_{cow}, p_{cgo}`: the deck's capillary-pressure tables (:mod:`modules.observables`).
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np

from .. import frames
from ..autopilot import Budget

Item = frames.Item

#: The project's own files inside a bundle: they ride with the deck but are not part of it.
PROJECT_FILE = "project.json"
PROJECT_DIRS = ("observed", "inversion")
BUNDLE_MAX_BYTES = 256 * 1024 * 1024


def work_root(root: str | Path | None = None) -> Path:
    """Where every deck's work directory lives: ``root``, else ``$GNN_WORK``, else ``/tmp/gnn``."""
    return Path(root or os.environ.get("GNN_WORK") or "/tmp/gnn")


@dataclass(frozen=True)
class Project:
    """A bundle, unpacked: the deck on disk, its work directory and digest, and the
    project's own files (by their path inside the bundle)."""

    deck_path: Path
    work: Path
    sha: str
    extras: dict[str, bytes] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.deck_path.stem

    def settings(self) -> dict[str, Any]:
        """``project.json`` of the bundle (``{}`` when it has none)."""
        raw = self.extras.get(PROJECT_FILE)
        return json.loads(raw.decode("utf-8")) if raw else {}


def _safe(name: str) -> PurePosixPath:
    """A bundle member's path, refused unless it stays inside the bundle."""
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"bundle member {name!r} does not stay inside the bundle")
    return path


def read_bundle(bundle: bytes, deck_name: str) -> tuple[str, dict[str, bytes], dict[str, bytes]]:
    """``(deck member, deck tree, project files)`` of a bundle. A bundle that is not a zip
    is the deck itself; in a zip the deck is ``deck_name`` when it is a member, else the
    only ``.DATA`` file at the top level."""
    if len(bundle) > BUNDLE_MAX_BYTES:
        raise ValueError(f"bundle larger than {BUNDLE_MAX_BYTES // 2**20} MiB")
    if not bundle.strip():
        raise ValueError("an empty deck was uploaded")
    if bundle[:2] != b"PK":
        name = Path(deck_name or "deck.DATA").name
        if not name.upper().endswith(".DATA"):
            name = f"{name}.DATA"
        return name, {name: bundle}, {}
    tree: dict[str, bytes] = {}
    extras: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = _safe(info.filename)
            data = archive.read(info)
            if str(path) == PROJECT_FILE or path.parts[0] in PROJECT_DIRS:
                extras[str(path)] = data
            else:
                tree[str(path)] = data
    wanted = str(PurePosixPath((deck_name or "").replace("\\", "/"))) if deck_name else ""
    if wanted in tree:
        return wanted, tree, extras
    decks = [n for n in tree if "/" not in n and n.upper().endswith(".DATA")]
    if len(decks) != 1:
        raise ValueError(
            f"the bundle holds {len(decks)} top-level .DATA files ({sorted(decks)}) and none is "
            f"named {deck_name!r}: name the deck to run")
    return decks[0], tree, extras


def tree_digest(tree: dict[str, bytes]) -> str:
    """The digest of a deck tree (see the module docstring)."""
    digest = hashlib.sha1()
    for name in sorted(tree):
        digest.update(name.encode("utf-8") + b"\0" + tree[name] + b"\0")
    return digest.hexdigest()


def unpack(bundle: bytes, deck_name: str, root: str | Path | None = None) -> Project:
    """Write a bundle's deck tree under its work directory (a file already there with the
    same bytes is left alone, so the simulator's results beside it stay newer than it) and
    return the :class:`Project`."""
    member, tree, extras = read_bundle(bundle, deck_name)
    sha = tree_digest(tree)
    work = work_root(root) / sha[:12]
    case_dir = work / "case"
    for name, data in tree.items():
        path = case_dir / name
        if path.is_file() and path.stat().st_size == len(data) and path.read_bytes() == data:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return Project(deck_path=case_dir / member, work=work, sha=sha, extras=extras)


# --------------------------------------------------------------------- one job at a time
def exclusive(root: str | Path | None = None, what: str = "job") -> Iterator[Item]:
    """Hold the workstation's one accelerator: yields ``stage`` messages while another job
    has it, returns once this one does. Two trainings sharing a GPU each run more than
    twice as slowly, so they take turns. The lock is released when the process's handle
    is closed (:func:`release`) or the process ends."""
    path = work_root(root)
    path.mkdir(parents=True, exist_ok=True)
    handle = open(path / ".lock", "w")
    waited = 0.0
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if waited % 30.0 == 0.0:
                yield frames.stage("queued", 0.0, f"waiting for the job that holds the workstation ({waited:.0f}s)")
            time.sleep(5.0)
            waited += 5.0
    _HELD.append(handle)


_HELD: list[Any] = []


def release() -> None:
    """Give the accelerator up (the end of a job, whether it finished or failed)."""
    while _HELD:
        handle = _HELD.pop()
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


# ------------------------------------------------------------- stage A: the simulator
def simulate_stage(project: Project, lo: float = 0.02, hi: float = 0.15) -> Iterator[Item]:
    """Hand the deck to OPM Flow unless its results are already beside it. Flow's report
    steps move the stage's fraction from ``lo`` to ``hi`` and are passed on as ``log``
    lines, at most one a second."""
    from .. import simulate

    deck = project.deck_path
    if simulate.results_present(deck):
        yield frames.stage("simulate", hi, "simulation results reused")
        return
    yield frames.stage("simulate", lo, f"running OPM Flow on {deck.name}")
    t0 = time.time()
    last = 0.0
    for line in simulate.iter_flow(deck):
        step = simulate.report_step(line)
        if step is None:
            continue
        now = time.time()
        if now - last < 1.0 and step[0] != step[1]:
            continue
        last = now
        n, total = step
        yield frames.log(line.strip(), source="flow")
        yield frames.stage("simulate", lo + (hi - lo) * (n / max(total, 1)),
                           f"OPM Flow: report step {n} of {total} ({now - t0:.0f}s)")
    yield frames.stage("simulate", hi, f"simulation ready ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------- stage B: the extraction
def prep_stage(project: Project, lo: float = 0.18, hi: float = 0.35) -> Iterator[Item]:
    """Extract the grid, the rock, the cell states and the wells through ResInsight into
    the preprocessing cache (reused when the simulator's files have not changed)."""
    from .. import prep

    yield frames.stage("prep", lo, "extracting the grid, states and wells through ResInsight")
    t0 = time.time()
    prep.build_prep_cache(project.deck_path, project.work / "prep_cache")
    yield frames.stage("prep", hi, f"preprocessing cache ready ({time.time() - t0:.0f}s)")


def setup(cfg) -> None:
    """Configure the process for ``cfg`` before anything imports jax: the x64 flag and the
    accelerator are process-wide and cannot change once jax is loaded."""
    from .. import bootstrap

    bootstrap.setup_environment(cfg.precision_policy, cfg.opt_f64,
                                platforms=bootstrap.platforms_for_host())


# ------------------------------------------------------------------ what a job shows
def snapshot_steps(times: np.ndarray, n_snapshots: int) -> list[int]:
    """Indices of ``n_snapshots`` report steps spread over the history, last one included."""
    n = int(len(times))
    k = max(1, min(int(n_snapshots), n))
    idx = np.unique(np.linspace(0, n - 1, k).round().astype(int))
    return [int(i) for i in idx]


class _CaseOnly:
    """What :func:`modules.observables._phase_pressures` reads of a bundle: its case."""

    def __init__(self, case):
        self.case = case


def seven_fields(case, P: np.ndarray) -> dict[str, np.ndarray]:
    """The seven cell fields from the four primaries ``P`` ``(n_cells, 4)``."""
    from .. import observables

    p_o, p_w, p_g = observables._phase_pressures(_CaseOnly(case), P)
    s_w, s_g, rs = np.asarray(P[:, 1]), np.asarray(P[:, 2]), np.asarray(P[:, 3])
    return {"S_w": s_w, "S_o": 1.0 - s_w - s_g, "S_g": s_g,
            "p_o": p_o, "p_w": p_w, "p_g": p_g, "R_so": rs}


def predicted_fields(bundle, params, t: float) -> dict[str, np.ndarray]:
    """The network's seven cell fields at time ``t`` (days)."""
    from .. import metrics

    return seven_fields(bundle.case, np.asarray(metrics.predict_cells(bundle, params, float(t))))


def simulated_fields(case, step: int) -> dict[str, np.ndarray]:
    """The simulator's seven cell fields at report step ``step``."""
    P = np.stack([np.asarray(case.pres[step]), np.asarray(case.swat[step]),
                  np.asarray(case.sgas[step]), np.asarray(case.rs[step])], axis=1)
    return seven_fields(case, P)


def well_markers(pack) -> list[dict[str, Any]]:
    """Perforated cells per well, for the 3-D view."""
    if pack is None:
        return []
    out = []
    for w, name in enumerate(pack.well_names):
        cells = [int(c) for c, wid in zip(pack.cell_idx, pack.well_id) if int(wid) == w]
        out.append({"name": str(name), "cells": cells})
    return out


def grid_frames(case, snap_days: list[float], pack=None) -> tuple[dict[str, list[float]], list[Item]]:
    """The ``grid`` message(s) of a case and the fixed colour ranges they carry."""
    mesh = case.art.reservoir_mesh
    dims = (tuple(int(d) for d in np.asarray(mesh.active_mask).shape)
            if getattr(mesh, "active_mask", None) is not None
            else tuple(int(x) + 1 for x in np.asarray(case.aci).max(axis=0)))
    ranges = frames.field_ranges(
        {"pres": case.pres, "swat": case.swat, "sgas": case.sgas, "rs": case.rs}, case.phys)
    items = list(frames.grid(np.asarray(mesh.corner_cells), np.asarray(case.aci), dims, snap_days,
                             ranges, wells=well_markers(pack)))
    return ranges, items


def well_frames(times: np.ndarray, names: list[str], *, simulator: dict | None = None,
                gnn: dict | None = None, observed: dict | None = None,
                split_days: float | None = None) -> Iterator[Item]:
    """One ``well`` message per (well, report time) at which any source has any quantity —
    the simulator's summary, the gNN's prediction, a field record — each ``(T, n_wells)``
    array with ``NaN`` where it has none. A quantity that is zero or missing throughout for a
    well (an injector's oil rate) is left out of all its messages. A point after
    ``split_days`` is in the ``"forecast"`` phase."""
    sources = {k: v for k, v in (("simulator", simulator), ("gnn", gnn), ("observed", observed)) if v}
    if not sources:
        return
    for w, name in enumerate(names):
        columns: dict[str, dict[str, np.ndarray]] = {}
        for source, record in sources.items():
            for quantity in frames.WELL_QUANTITIES:
                if quantity not in record:
                    continue
                col = np.asarray(record[quantity], np.float64)[:, w]
                if np.isfinite(col).any() and np.nanmax(np.abs(col)) > 0.0:
                    columns.setdefault(source, {})[quantity] = col
        if not columns:
            continue
        for step in range(len(times)):
            values = {source: {q: (float(c[step]) if np.isfinite(c[step]) else None) for q, c in cols.items()}
                      for source, cols in columns.items()}
            if not any(v is not None for cols in values.values() for v in cols.values()):
                continue
            t = float(times[step])
            yield frames.well(step, t, str(name), values,
                              phase=("forecast" if split_days is not None and t > split_days + 1e-9 else "history"))


def train_frames(bundle, state, budget: Budget, t_start: float, snap_idx: list[int],
                 snap_days: list[float], ranges: dict[str, list[float]]) -> Iterator[Item]:
    """Train ``state`` on ``bundle`` for ``budget.n_iter`` iterations, one at a time:
    a ``loss`` message per iteration and a sweep of ``state`` snapshots at iteration 0 and
    every ``budget.snapshot_every`` iterations. :func:`modules.training.train` continues
    from ``state.train_iter`` with the compiled steps cached, which is what makes one
    iteration per call cost one iteration."""
    from .. import training

    n_iter = int(budget.n_iter)

    def snapshots(it: int) -> Iterator[Item]:
        for k, _ in enumerate(snap_idx):
            yield frames.state(it, k, snap_days[k], predicted_fields(bundle, state.params, snap_days[k]),
                               ranges)

    yield from snapshots(0)               # the untrained network, so the view is never empty
    done = 0
    every = max(1, n_iter // 50)           # a progress message every 2 % of the iterations
    t_train = time.time()
    while done < n_iter:
        result = training.train(bundle, state, n_iter=1, log_every=10**9, stop_on_eta_zero=True)
        done = int(state.train_iter)
        h = state.hist
        comps = {g: float(h[g][-1]) for g in h if g in bundle.resolved.groups and h[g]}
        # ``done`` counts completed iterations (1..n_iter): what the page's progress bar and
        # the snapshots' iteration stamp use; ``hist["iter"]`` is 0-based.
        yield frames.item(
            "loss",
            {"wall_s": time.time() - t_start,
             "eta": (float(h["engd_eta"][-1]) if h.get("engd_eta") else None),
             "groups": comps, **{f"g_{g}": v for g, v in comps.items()}},
            iteration=done,
            losses=(float(h["total"][-1]), comps.get("pde", 0.0), comps.get("ic", 0.0),
                    comps.get("data", 0.0), comps.get("well", 0.0)))
        if result.status != "completed":
            yield frames.stage("train", 0.95, f"stopped early: {result.status} at iteration {done}")
            break
        if done % every == 0 and done < n_iter:
            per = (time.time() - t_train) / done
            yield frames.stage("train", 0.6 + 0.38 * done / n_iter,
                               f"iteration {done} of {n_iter} · about {duration(per * (n_iter - done))} left")
        if budget.snapshot_every > 0 and done % budget.snapshot_every == 0 and done < n_iter:
            yield from snapshots(done)
    yield from snapshots(int(state.train_iter))


def duration(seconds: float) -> str:
    """A time span as a person says it: 40 s, 12 min, 3.5 h."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def plan_message(cfg, resolved) -> str:
    """One line saying what was resolved for this reservoir: the basis, the network, the
    ENGD plan."""
    text = (f"n_eig={cfg.n_eig} width={cfg.m_width} depth={cfg.n_blocks} "
            f"P={resolved.param_count:,} groups={'+'.join(resolved.groups)}")
    plan = resolved.engd_plan
    if plan is not None:
        text += f" | ENGD {plan.mode}, {plan.n_rows:,} rows"
    return text
