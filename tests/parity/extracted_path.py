"""The extracted path (modules) on the same deck, composition and seed as upstream_path.py.

Run with this repository's interpreter:

    .venv/bin/python tests/parity/extracted_path.py <deck> <work dir> <path> <n_iter> <upstream.json>

``<work dir>`` holds copies of the research repository's ``prep_cache`` and
``spectral_cache`` for the deck, so both paths read the same extraction and the same
eigenbasis (eigenvectors of a degenerate spectrum are defined only up to a rotation, so two
solves of one problem differ). Prints ``PARITY IDENTICAL`` when every loss of every
iteration and the parameter checksum agree exactly, else the largest relative difference.
"""
import json, os, sys, time

deck, work, path, n_iter, base_path = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules import autopilot, bootstrap
from modules.autopilot import Budget
bootstrap.setup_environment("selective_f64", True, platforms="cpu")
# The research repository's prep cache pickles its classes as utils.<module>.<Class>;
# alias them so the SAME artifacts load into the extracted package.
import importlib, types
_alias = types.ModuleType("utils"); sys.modules["utils"] = _alias
for _m in ("ReservoirExtractors", "ReservoirMesh"):
    sys.modules[f"utils.{_m}"] = importlib.import_module(f"modules.utils.{_m}")
    setattr(_alias, _m, sys.modules[f"utils.{_m}"])
import numpy as np
from modules import casedata, inference, pipeline, training

budget = Budget(n_iter=n_iter)
t0 = time.time()
if path == "history":
    cfg = autopilot.history(deck, work, "parity", budget)
    cfg, resolved, case, _ = pipeline.resolve_run(cfg)
    bundle = pipeline.build_pipeline(cfg, case=case, verbose=False)
else:
    n = int(path.split(":")[1])
    cfg = autopilot.forecast(deck, work, "parity", n, budget)
    full = casedata.load_case(cfg)
    cfg, resolved, _, _ = pipeline.resolve_run(cfg, case=full)
    bundle, _ = inference.build_joint_bundle(cfg, full, verbose=False)
print("built", cfg.n_eig, cfg.m_width, cfg.n_blocks, bundle.resolved.groups, f"{time.time()-t0:.0f}s", flush=True)
state = training.init_trainer(bundle)
training.train(bundle, state, n_iter=n_iter, log_every=1, stop_on_eta_zero=False)
keys = ("iter", "total", *bundle.resolved.groups)
out = {k: [float(x) for x in state.hist[k]] for k in keys}
out["groups"] = list(bundle.resolved.groups)
out["params_sum"] = float(sum(float(np.asarray(x, np.float64).sum()) for x in __import__("jax").tree_util.tree_leaves(state.params)))
base = json.load(open(base_path))
same = out == base
worst = 0.0
if not same and out["groups"] == base["groups"]:
    for k in (*keys, "params_sum"):
        a, b = np.atleast_1d(np.asarray(out[k], np.float64)), np.atleast_1d(np.asarray(base[k], np.float64))
        if a.shape == b.shape:
            worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))))
        else:
            worst = float("inf")
print("PARITY", "IDENTICAL" if same else f"DIFFERENT (largest relative difference {worst:.3e})",
      json.dumps({"total": out["total"], "upstream": base["total"]}), f"{time.time()-t0:.0f}s")
