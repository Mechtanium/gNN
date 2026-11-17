"""The research path (pinnlab) for one work path on one deck: the parity target.

Run with the research repository's interpreter:

    <research>/.venv/bin/python upstream_path.py <research root> <case> <path> <n_iter> <out.json>

``<path>`` is ``history`` or ``forecast:<n>`` (``n`` report steps of history). The
composition is ``pinnlab.config.autopilot`` with exactly the additions
``modules/autopilot.py`` makes for that path; training runs on the CPU.
"""
import json, os, sys, time

root, case_name, path, n_iter, out_path = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")
sys.path.insert(0, root)
os.chdir(root)
from pinnlab import bootstrap
bootstrap.setup_environment("selective_f64", True, platforms="cpu")
import numpy as np
from pinnlab import casedata, inference, pipeline, training
from pinnlab.config import autopilot, loss_groups, validate

if path == "history":
    cfg = autopilot(case_name, observation="both", well_model="closed_form", n_iter=n_iter)
    groups = loss_groups(validate(cfg)[0])
    import dataclasses
    cfg = dataclasses.replace(cfg, w_fixed=tuple(0.1 if g == "well" else 1.0 for g in groups))
else:
    n = int(path.split(":")[1])
    cfg = autopilot(case_name, observation="both", well_model="predicted", ctrl_switch="staircase",
                    inf_mode="joint", train_split_n=n, n_iter=n_iter)

t0 = time.time()
if path == "history":
    cfg, resolved, case, _ = pipeline.resolve_run(cfg)
    bundle = pipeline.build_pipeline(cfg, case=case, enable_ckpt=False, enable_writer=False, verbose=False)
else:
    full = casedata.load_case(cfg)
    cfg, resolved, _, _ = pipeline.resolve_run(cfg, case=full)
    bundle, _ = inference.build_joint_bundle(cfg, full, enable_ckpt=False, enable_writer=False, verbose=False)
print("built", cfg.n_eig, cfg.m_width, cfg.n_blocks, bundle.resolved.groups, f"{time.time()-t0:.0f}s", flush=True)
state = training.init_trainer(bundle, resume=False)
training.train(bundle, state, n_iter=n_iter, log_every=1, stop_on_eta_zero=False)
keys = ("iter", "total", *bundle.resolved.groups)
out = {k: [float(x) for x in state.hist[k]] for k in keys}
out["groups"] = list(bundle.resolved.groups)
out["params_sum"] = float(sum(float(np.asarray(x, np.float64).sum()) for x in __import__("jax").tree_util.tree_leaves(state.params)))
json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps({"total": out["total"]}), f"{time.time()-t0:.0f}s")
