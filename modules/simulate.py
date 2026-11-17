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

The wheel is always driven in a **fresh subprocess**: ``BlackOilSimulator`` keeps
process-wide state (MPI, the parameter registry), and a second ``run()`` in the
same interpreter returns success without writing a single file. A worker serves
many jobs from one process, so the second deck it ever saw used to fail with
"result files are missing".
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterator

#: The result files a run must leave behind, by extension (the stem is the
#: deck's, upper-cased by OPM).
REQUIRED = ("EGRID", "INIT", "UNRST", "SMSPEC", "UNSMRY")
FLOW_ARGS = ("--enable-esmry=true",)

#: The child interpreter's program: ``python -c _WHEEL_RUNNER <deck> <flow args…>``.
_WHEEL_RUNNER = (
    "import sys\n"
    "from opm.simulators import BlackOilSimulator\n"
    "rc = BlackOilSimulator(sys.argv[1], sys.argv[2:]).run()\n"
    "sys.exit(0 if rc in (0, None) else int(rc))\n"
)


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


#: Flow announces each report step as ``Report step  5/120 at day 151/3650, date = …``.
_REPORT_STEP = re.compile(r"Report step\s+(\d+)\s*/\s*(\d+)")


def flow_command(deck: Path) -> list[str]:
    """The command that simulates ``deck``: the ``opm-simulators`` wheel through
    :data:`_WHEEL_RUNNER` in a child interpreter (one simulation per process — see the
    module docstring), else a ``flow`` binary on ``PATH``."""
    if importlib.util.find_spec("opm") is not None and importlib.util.find_spec("opm.simulators") is not None:
        return [sys.executable, "-c", _WHEEL_RUNNER, str(deck), *FLOW_ARGS]
    flow = shutil.which("flow")
    if flow is None:
        raise RuntimeError(
            "no reservoir simulator: install the opm-simulators wheel or put "
            "OPM Flow's `flow` on PATH")
    return [flow, str(deck), *FLOW_ARGS]


def iter_flow(deck_path: str | Path, threads: int | None = None) -> Iterator[str]:
    """Simulate the deck in place, yielding the simulator's output line by line as it is
    written — what lets a job show the hand-off to the simulator as it happens. Raises
    with the last lines when Flow fails or leaves no result files. ``threads`` caps the
    simulator's threads (several runs side by side each take one)."""
    import os

    deck = Path(deck_path).resolve()
    env = None if threads is None else {**os.environ, "OMP_NUM_THREADS": str(int(threads))}
    proc = subprocess.Popen(flow_command(deck), cwd=deck.parent, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    tail: deque[str] = deque(maxlen=25)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            yield line
        code = proc.wait()
    finally:
        if proc.poll() is None:           # the consumer stopped early (a cancelled job)
            proc.kill()
            proc.wait()
    if code != 0:
        raise RuntimeError(f"OPM Flow returned {code} for {deck.name}:\n" + "\n".join(tail))
    if not results_present(deck):
        present = sorted(p.name for p in deck.parent.iterdir())
        raise RuntimeError(
            f"OPM Flow finished but the result files are missing beside {deck} "
            f"(present: {present}):\n" + "\n".join(tail))


def report_step(line: str) -> tuple[int, int] | None:
    """``(step, total)`` when ``line`` announces a report step, else ``None``."""
    m = _REPORT_STEP.search(line)
    return (int(m.group(1)), int(m.group(2))) if m else None


def run_flow(deck_path: str | Path, progress: Callable[[str], None] | None = None) -> float:
    """Simulate the deck in place; returns the wall time in seconds.

    ``progress`` is called with a short message every 15 s while the simulator works, so a
    caller can show a heartbeat. The simulator's own output goes to the parent's stdout;
    on failure the exception carries its last lines (:func:`iter_flow`).
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
        for line in iter_flow(deck):
            sys.stdout.write(line + "\n")
    finally:
        done.set()
    return time.time() - t0


def ensure_results(deck_path: str | Path, progress: Callable[[str], None] | None = None) -> float:
    """Simulate unless the results are already there; returns the seconds spent (0 when cached)."""
    if results_present(deck_path):
        return 0.0
    return run_flow(deck_path, progress)
