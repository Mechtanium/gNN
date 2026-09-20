"""The notebook path (pinnlab) on SPE2EQUI, small CPU config, fixed seed: the parity target."""
import json, os, sys, time
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
sys.path.insert(0, "/home/enosmath/Flintstone/Delta-PINNs")
from pinnlab import bootstrap
bootstrap.setup_environment("selective_f64", True, platforms="cpu")
import numpy as np
from pinnlab import pipeline, training
from pinnlab.config import *

cfg = RunConfig(
    mesh_case=MeshCase.SPE2EQUI, input_encoding=InputEncoding.SPECTRAL, architecture=Architecture.DGM,
    residual_design=ResidualDesign.SPECTRAL_PDE, backprop_design=BackpropDesign.FEM_NODAL,
    stiffness_design=StiffnessDesign.PERM_WEIGHTED, sampling=SamplingDesign.WINDOW,
    weighting=WeightingDesign.FIXED, special_opt=SpecialOpt.ENGD, special_opt_after=0,
    parallelism=Parallelism.DATA, observation=ObservationDesign.BOTH, well_model=WellModel.CLOSED_FORM,
    ctrl_switch="fb", pde_node_weight="well_gaussian", well_gauss_boost=4.0, well_gauss_width=1.0,
    spectral_projection="galerkin", ic_design="penalty", rs_supervision="oil_only",
    time_encoding="linear", tslice_layout="midpoint", well_encoding="logr",
    n_eig=3, n_eig_z=0, m_width=0, n_blocks=0, n_iter=5, seed=30, ckpt_every=10**9,
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
bundle = pipeline.build_pipeline(cfg, case=case, enable_ckpt=False, enable_writer=False, verbose=False)
print("built", f"{time.time()-t0:.0f}s", flush=True)
state = training.init_trainer(bundle, resume=False)
result = training.train(bundle, state, n_iter=5, log_every=1, stop_on_eta_zero=False)
out = {k: [float(x) for x in v] for k, v in state.hist.items() if k in ("iter", "total", "pde", "ic", "data", "well")}
out["params_hash"] = str(sum(float(np.asarray(x).sum()) for x in __import__("jax").tree_util.tree_leaves(state.params)))
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: out[k] for k in ("iter", "total")}), f"{time.time()-t0:.0f}s")
