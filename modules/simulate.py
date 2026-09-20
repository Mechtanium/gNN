r"""
Stage A of the workflow: the reference simulation.

The deck is run through OPM Flow, the same simulator that produced the research
results (``data/spe2-cartesian/run_cases.sh`` in PINN-Lab runs
``flow <deck> --enable-esmry=true --output-dir=RESULTS``). Its outputs — the
grid (``EGRID``), the static properties (``INIT``), the per-report-step cell
states (``UNRST``) and the well summary (``SMSPEC``/``UNSMRY``/``ESMRY``) — are
what ResInsight later extracts. Flow comes from the ``opm-simulators`` wheel
(its Python API writes beside the deck regardless of ``--output-dir``), or from
a ``flow`` binary on ``PATH`` when one exists.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

#: The result files a run must leave behind, by extension (the stem is the
#: deck's, upper-cased by OPM).
REQUIRED = ("EGRID", "INIT", "UNRST", "SMSPEC", "UNSMRY")
FLOW_ARGS = ("--enable-esmry=true",)


def results_present(deck_path: str | Path) -> bool:
    """True when every required result file exists beside the deck (or under
    ``RESULTS/``) and is newer than the deck."""
    deck = Path(deck_path).resolve()
    stem = deck.name.split(".", 1)[0].upper()
    for ext in REQUIRED:
        found = None
        for d in (deck.parent, deck.parent / "RESULTS"):
            for cand in (d / f"{stem}.{ext}", d / f"{deck.stem}.{ext}"):
                if cand.is_file():
                    found = cand
                    break
            if found is not None:
                break
        if found is None or found.stat().st_mtime < deck.stat().st_mtime:
            return False
    return True


def run_flow(deck_path: str | Path, progress: Callable[[str], None] | None = None) -> float:
    """Simulate the deck in place; returns the wall time in seconds.

    Prefers the ``opm-simulators`` Python API (no subprocess, no PATH); falls back
    to a ``flow`` binary. ``progress`` is called with a short message every 15 s
    while the simulator works, so a caller streaming a job can show a heartbeat.
    """
    deck = Path(deck_path).resolve()
    t0 = time.time()
    done = threading.Event()

    def heartbeat() -> None:
        while not done.wait(15.0):
            if progress is not None:
                progress(f"simulating {deck.name} ({time.time() - t0:.0f}s)")

    ticker = threading.Thread(target=heartbeat, daemon=True)
    ticker.start()
    try:
        cwd = os.getcwd()
        os.chdir(deck.parent)
        try:
            try:
                from opm.simulators import BlackOilSimulator
            except ImportError:
                BlackOilSimulator = None
            if BlackOilSimulator is not None:
                sim = BlackOilSimulator(str(deck), list(FLOW_ARGS))
                rc = sim.run()
                if rc not in (0, None):
                    raise RuntimeError(f"OPM Flow returned {rc} for {deck.name}")
            else:
                flow = shutil.which("flow")
                if flow is None:
                    raise RuntimeError(
                        "no reservoir simulator: install the opm-simulators wheel or put "
                        "OPM Flow's `flow` on PATH")
                subprocess.run([flow, str(deck), *FLOW_ARGS], check=True)
        finally:
            os.chdir(cwd)
    finally:
        done.set()
    if not results_present(deck):
        raise RuntimeError(f"OPM Flow finished but the result files are missing beside {deck}")
    return time.time() - t0


def ensure_results(deck_path: str | Path, progress: Callable[[str], None] | None = None) -> float:
    """Simulate unless the results are already there; returns the seconds spent (0 when cached)."""
    if results_present(deck_path):
        return 0.0
    return run_flow(deck_path, progress)
