"""Runs inside the image as uid 10001 (see test_resinsight_container.sh): simulate
the deck with OPM Flow from the pip wheel, then launch ResInsight through the
xvfb wrapper, open the results with rips and report the active-cell count."""
import glob
import os
import shutil
import socket
import time

DECK = os.environ.get("DECK", "/data/SPE-2-cartesian-equi.DATA")
os.environ.setdefault("HOME", "/home/perd")
os.environ.setdefault("RESINSIGHT_EXECUTABLE", "/opt/ResInsight/bin/resinsight-xvfb")

# --- OPM Flow from the pip wheel, on a copy of the deck ------------------------------------
os.makedirs("/tmp/sim", exist_ok=True)
deck = shutil.copy(DECK, "/tmp/sim/")
os.chdir("/tmp/sim")
from opm.simulators import BlackOilSimulator  # noqa: E402

t0 = time.time()
rc = BlackOilSimulator(deck, ["--enable-esmry=true", "--output-dir=/tmp/sim/RESULTS"]).run()
outputs = sorted(os.path.basename(p) for p in glob.glob("/tmp/sim/RESULTS/*"))
print(f"[flow] rc={rc} in {time.time() - t0:.0f}s; outputs: {outputs}", flush=True)
egrid = next(p for p in glob.glob("/tmp/sim/RESULTS/*.EGRID"))

# --- ResInsight under xvfb, driven by rips ----------------------------------------------------
import rips  # noqa: E402

s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
t0 = time.time()
inst = rips.Instance.launch(console=False, launch_port=port, init_timeout=180)
print(f"[rips] launched in {time.time() - t0:.1f}s, version {inst.version_string()}", flush=True)
case = inst.project.load_case(egrid)
info = case.cell_count()
print(f"[rips] case {case.name}: active={info.active_cell_count} "
      f"reservoir={info.reservoir_cell_count} steps={len(case.time_steps())}", flush=True)
corners = case.active_cell_corners()
print(f"[rips] corners for {len(corners)} active cells; first: "
      f"{corners[0].c0.x:.1f},{corners[0].c0.y:.1f},{corners[0].c0.z:.1f}", flush=True)
inst.exit()
assert info.active_cell_count > 0 and len(corners) == info.active_cell_count
print("container recipe OK", flush=True)
