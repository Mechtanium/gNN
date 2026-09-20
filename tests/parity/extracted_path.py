"""The extracted path (modules) on the same deck/config/seed as baseline_notebook.py."""
import json, os, sys, time
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
sys.path.insert(0, "/home/enosmath/PERD/Delta-PINN")
from modules import bootstrap
bootstrap.setup_environment("selective_f64", True, platforms="cpu")
# The research repo's prep cache pickles its classes as utils.<module>.<Class>;
# alias them so the SAME artifacts load into the extracted package.
import importlib, types
_alias = types.ModuleType("utils"); sys.modules["utils"] = _alias
for _m in ("ReservoirExtractors", "ReservoirMesh"):
    sys.modules[f"utils.{_m}"] = importlib.import_module(f"modules.utils.{_m}")
    setattr(_alias, _m, sys.modules[f"utils.{_m}"])
import numpy as np
from modules import pipeline, training
from modules.config import *

cfg = RunConfig(
    deck_path="/home/enosmath/Flintstone/Delta-PINNs/data/spe2-cartesian/equispaced/SPE-2-cartesian-equi.DATA",
    work_dir=sys.argv[2],
    input_encoding=InputEncoding.SPECTRAL, architecture=Architecture.DGM,
    residual_design=ResidualDesign.SPECTRAL_PDE, backprop_design=BackpropDesign.FEM_NODAL,
    stiffness_design=StiffnessDesign.PERM_WEIGHTED, sampling=SamplingDesign.WINDOW,
    weighting=WeightingDesign.FIXED, special_opt=SpecialOpt.ENGD, special_opt_after=0,
    parallelism=Parallelism.DATA, observation=ObservationDesign.BOTH, well_model=WellModel.CLOSED_FORM,
    ctrl_switch="fb", pde_node_weight="well_gaussian", well_gauss_boost=4.0, well_gauss_width=1.0,
    spectral_projection="galerkin", ic_design="penalty", rs_supervision="oil_only",
    time_encoding="linear", tslice_layout="midpoint", well_encoding="logr",
    n_eig=3, n_eig_z=0, m_width=0, n_blocks=0, n_iter=5, seed=30,
    n_tslice=2, full_batch_cap=256,
    spec=SpectralSettings(retention="deflation", stratify_components=False, min_comp_vol_frac=0.005,
                          min_node_reuse=8, addressability_min=0.0, addressability_tol=0.01,
                          blind_frac_max=0.0, n_eig_cap=63),
    engd=EngdSettings(param_target=800, damping_rel=1e-4, damping_abs=0.0),
    precision_policy="selective_f64", opt_f64=True,
)
t0 = time.time()
cfg, resolved, case, warns = pipeline.resolve_run(cfg)
print("resolved", resolved.dim_in, resolved.param_count, cfg.n_eig, cfg.m_width, cfg.n_blocks, f"{time.time()-t0:.0f}s", flush=True)
bundle = pipeline.build_pipeline(cfg, case=case, verbose=False)
print("built", f"{time.time()-t0:.0f}s", flush=True)
state = training.init_trainer(bundle)
result = training.train(bundle, state, n_iter=5, log_every=1, stop_on_eta_zero=False)
out = {k: [float(x) for x in v] for k, v in state.hist.items() if k in ("iter", "total", "pde", "ic", "data", "well")}
out["params_hash"] = str(sum(float(np.asarray(x).sum()) for x in __import__("jax").tree_util.tree_leaves(state.params)))
json.dump(out, open(sys.argv[1], "w"), indent=1)
base = json.load(open(sys.argv[3]))
same = all(out[k] == base[k] for k in ("iter", "total", "pde", "ic", "data", "well")) and out["params_hash"] == base["params_hash"]
print("PARITY", "IDENTICAL" if same else "DIFFERENT", json.dumps({k: out[k] for k in ("total",)}), f"{time.time()-t0:.0f}s")
