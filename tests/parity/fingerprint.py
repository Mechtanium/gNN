import json, sys, numpy as np, jax
def fp(name, x):
    a = np.asarray(x, dtype=np.float64)
    return {name: [float(a.sum()), float(np.abs(a).sum()), list(a.shape)]}
def report(bundle, state, cfg, resolved):
    out = {}
    for i, leaf in enumerate(jax.tree_util.tree_leaves(bundle.model.params0)):
        out.update(fp(f"params0[{i}]", leaf))
    sc = bundle.scales
    for k in dir(sc):
        if k.startswith("_"): continue
        v = getattr(sc, k)
        try: out.update(fp(f"scales.{k}", v))
        except Exception: pass
    if bundle.spec is not None:
        for k in ("lam", "v_nodes", "v_c", "b_v", "mu", "upsilon"):
            v = getattr(bundle.spec, k, None)
            if v is not None:
                try: out.update(fp(f"spec.{k}", v))
                except Exception: pass
    win = state.win
    if win is not None:
        for k in win._fields:
            v = getattr(win, k)
            if v is not None:
                try: out.update(fp(f"win.{k}", v))
                except Exception: pass
    out["w"] = [float(x) for x in np.asarray(state.w)]
    out["engd"] = None if resolved.engd_plan is None else [resolved.engd_plan.mode, resolved.engd_plan.direction, int(resolved.engd_plan.n_rows)]
    out["groups"] = list(resolved.groups)
    p = bundle.well_pack
    if p is not None:
        out["pack"] = [int(p.n_rows), int(p.n_rows_well), int(p.n_perf), bool(p.synthesized)]
    out["times"] = [float(t) for t in np.asarray(bundle.case.times)[:3]]
    out["cfg_hash"] = sys.modules[cfg.__class__.__module__].structural_hash(cfg)
    return out
