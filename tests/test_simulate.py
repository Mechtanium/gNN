"""OPM Flow through :mod:`modules.simulate`: results land beside the deck, a
second simulation in the same process works (the wheel's ``BlackOilSimulator``
is one-shot per interpreter, which is why ``run_flow`` uses a child process),
and a cached result is reused. Needs the ``opm-simulators`` wheel or ``flow``
on PATH; SPE1 runs in a few seconds."""
import shutil
import importlib.util
from pathlib import Path

import pytest

from modules import simulate

DECK = Path(__file__).parent / "data" / "SPE1CASE1.DATA"

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("opm") is None and shutil.which("flow") is None,
    reason="no OPM Flow (opm-simulators wheel or flow binary)")


def test_two_simulations_in_one_process(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    da, db = shutil.copy(DECK, a), shutil.copy(DECK, b)
    assert not simulate.results_present(da)
    notes = []
    secs = simulate.ensure_results(da, progress=notes.append)
    assert secs > 0 and simulate.results_present(da)
    assert {p.suffix for p in a.iterdir()} >= {".EGRID", ".INIT", ".UNRST", ".SMSPEC", ".UNSMRY"}
    # The second deck in the same interpreter — the pod's situation after one job.
    assert simulate.ensure_results(db) > 0 and simulate.results_present(db)
    # A third call on the first deck is a cache hit.
    assert simulate.ensure_results(da) == 0.0


def test_failure_is_diagnosed(tmp_path):
    bad = tmp_path / "BROKEN.DATA"
    bad.write_text("RUNSPEC\nTITLE\n broken\nGRID\n")
    with pytest.raises(RuntimeError) as e:
        simulate.run_flow(bad)
    assert "BROKEN" in str(e.value)
