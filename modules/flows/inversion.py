r"""
Work path 3: reservoir property inversion from a field record, through a family of reservoirs.

A record of bottom-hole pressures and rates does not determine a reservoir's state on its own,
but it can tell apart reservoirs that differ in a few uncertain properties. The path teaches
one network the whole family of such reservoirs and searches it for the member whose wells
behave as the field's did:

1. **Sample.** :math:`N + H` whitened property vectors :math:`\mathbf u_s \sim \mathcal N(0, I)`
   are drawn by Latin hypercube; the model's deck is edited to each (:mod:`builder.variants`)
   and OPM Flow simulates every variant, several at a time.
2. **Teach.** One network :math:`u_\theta(\mathbf x, t; \mathbf u)` with the properties as
   extra input channels is trained on the cell states of the :math:`N` training runs, the
   black-oil rows assembled at each run's own properties (:mod:`modules.family`).
3. **Score.** The family predicts the :math:`H` held-out runs' wells; its error there is the
   emulator's own, added to the record's noise:
   :math:`\sigma_c^2 = \sigma_{\mathrm{obs},c}^2 + \sigma_{\mathrm{emu},c}^2`.
4. **Invert.** The record is matched by Gauss–Newton from several starts,

   .. math::

       \hat{\mathbf u} = \arg\min_{\mathbf u}\;
       \frac12 \sum_{c,t,w} \left(\frac{d^{\mathrm{gNN}}_{c,t,w}(\mathbf u) - d^{\mathrm{obs}}_{c,t,w}}{\sigma_c}\right)^2
       + \frac12 \lVert \mathbf u \rVert^2

   and the posterior is sampled around :math:`\hat{\mathbf u}` (random-walk Metropolis).
5. **Verify.** The simulator runs the model at the posterior mean, and its record is set
   against the field's and the family's.

where:

- :math:`c`: the record channels — bottom-hole pressure, gas–oil ratio and water cut; :math:`t, w`: the report times and the producing wells at which the field record has them.
- :math:`d^{\mathrm{gNN}}(\mathbf u)`: the record the family implies for :math:`\mathbf u`, read through the closed-form inflow relation under the schedule's controls (:func:`modules.family.record_at`).
- :math:`\lVert \mathbf u \rVert^2 / 2`: the prior, a standard normal in the whitened properties.

Everything is cached under the model's work directory: a second inversion of the same model
with another record reuses the runs and the trained family and only searches again.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .. import autopilot, frames
from ..autopilot import Budget
from . import common

Item = frames.Item

#: METRIC → FIELD for the record channels (liquid sm³/d → stb/d, gas sm³/d → Mscf/d, bar → psia).
_LIQ, _GAS, _BAR = 1 / 0.158987295, 1 / 28.3168466, 1 / 0.0689475729


def run(bundle: bytes, deck_name: str, budget: Budget = Budget(),
        work_root: str | Path | None = None) -> Iterator[Item]:
    """Invert the bundle's field record for the model's uncertain properties."""
    from builder import variants
    from builder.deck import Deck
    from builder.project import PRIOR_DEFAULTS, PRIOR_FILE, RECORD_FILE, read_record_csv

    t_start = time.time()
    project = common.unpack(bundle, deck_name, work_root)
    member, tree, extras = common.read_bundle(bundle, deck_name)
    if RECORD_FILE not in extras:
        raise ValueError("this model has no observed record: enter or import the field's well record "
                         "(Wells & schedule › Observed data) before an inversion")
    record_rows = read_record_csv(extras[RECORD_FILE])
    prior = {**PRIOR_DEFAULTS, **(json.loads(extras[PRIOR_FILE]) if PRIOR_FILE in extras else {})}
    deck = Deck.from_tree(tree, member)
    metric = deck.has("METRIC", "RUNSPEC")

    cfg = autopilot.history(project.deck_path, project.work, project.sha, budget)
    common.setup(cfg)
    yield from common.simulate_stage(project, 0.01, 0.05)
    yield from common.prep_stage(project, 0.05, 0.08)

    from .. import casedata, wells as wells_mod

    base_case = casedata.load_case(cfg)
    z = np.asarray(base_case.centroids, np.float64)[:, 2]
    names = variants.properties_of(deck, float(z.min()), float(z.max()))
    found = variants.contacts(deck)
    base_contacts = {k: float(np.mean(v)) for k, v in found.items() if v}
    well_names = list(wells_mod.build_well_pack(cfg, base_case, write_meta=False).well_names)
    yield frames.stage("sample", 0.09, f"inverting for {', '.join(variants.LABELS[n] for n in names)}: "
                                       f"{budget.n_train} training and {budget.n_held} held-out runs")

    # ---- 1. sample the prior and simulate every variant -----------------------------------
    key = hashlib.sha1(json.dumps({"sha": project.sha, "names": names, "prior": {n: prior[n] for n in names},
                                   "n": [budget.n_train, budget.n_held]}, sort_keys=True).encode()).hexdigest()[:12]
    root = project.work / "family" / key
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    u = variants.latin_hypercube(budget.n_train + budget.n_held, len(names), rng)
    samples = []
    for s in range(len(u)):
        values = variants.physical(u[s], names, prior, base_contacts)
        samples.append({"name": f"s{s:03d}", "u": u[s].tolist(), "held_out": s >= budget.n_train,
                        "values": values, "stem": ""})
    yield from _simulate_family(tree, member, root, samples)
    (root / "samples.json").write_text(json.dumps({"names": names, "prior": prior, "samples": samples}, indent=1))

    # ---- 2. teach the family --------------------------------------------------------------
    from .. import family, training
    from ..models import flatten_state

    fcfg = autopilot.family(project.deck_path, project.work, project.sha, len(names), budget)
    from dataclasses import replace

    fcfg = replace(fcfg, inv=replace(fcfg.inv, prior_sd=float(prior["kvkh"])))
    yield frames.stage("teach", 0.42, "loading the family's cell states")
    data = family.load_family(root, base_case, well_names, limit=budget.n_train)
    held = family.load_family(root, base_case, well_names, held_out=True)
    if metric:
        _to_field(data)
        _to_field(held)
    built, _tables = family.build_family(fcfg, base_case, data, verbose=False)
    yield frames.stage("teach", 0.46, common.plan_message(built.cfg, built.resolved) + "; compiling")
    params_path = root / f"params_{budget.n_iter}.npz"
    _, unflatten = flatten_state(_as64(built.model.params0))
    if params_path.is_file():
        params = unflatten(np.load(params_path)["theta"])
        yield frames.stage("teach", 0.80, f"the trained family is reused ({params_path.name})")
    else:
        state = training.init_trainer(built, resume=False)
        done, draw = 0, 0
        t_teach = time.time()
        while done < budget.n_iter:
            state.win = family.family_window(fcfg, built.case, built.env, data, offset=draw,
                                             mix=autopilot.FAMILY_WINDOW_MIX)
            draw += 1
            for _ in range(min(budget.redraw_every, budget.n_iter - done)):
                result = training.train(built, state, n_iter=1, log_every=10**9, stop_on_eta_zero=False)
                done = int(state.train_iter)
                h = state.hist
                comps = {g: float(h[g][-1]) for g in h if g in built.resolved.groups and h[g]}
                yield frames.item("loss", {"wall_s": time.time() - t_start, "window": draw,
                                           "eta": (float(h["engd_eta"][-1]) if h.get("engd_eta") else None),
                                           "groups": comps, **{f"g_{g}": v for g, v in comps.items()}},
                                  iteration=done, losses=(float(h["total"][-1]), comps.get("pde", 0.0),
                                                          comps.get("ic", 0.0), comps.get("data", 0.0), 0.0))
                if result.status != "completed":
                    break
            per = (time.time() - t_teach) / max(done, 1)
            yield frames.stage("teach", 0.46 + 0.34 * done / budget.n_iter,
                               f"iteration {done} of {budget.n_iter} · about "
                               f"{common.duration(per * (budget.n_iter - done))} left")
        params = state.params
        theta, _ = flatten_state(_as64(params))
        np.savez_compressed(params_path, theta=np.asarray(theta, np.float64))

    # ---- 3. score the family on the held-out runs ------------------------------------------
    yield frames.stage("score", 0.82, f"scoring the family on {held.m.shape[0]} held-out runs")
    scores = [family.score_sample(built, params, held.m[i], held.states[i], held.records[i])
              for i in range(held.m.shape[0])]
    emulator = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]} if scores else {}
    sigma = family.effective_sigma(dict(autopilot.RECORD_SIGMA), emulator)

    # ---- 4. invert the field record -----------------------------------------------------------
    times = np.asarray(base_case.times, np.float64)
    observed, flowing = record_arrays(record_rows, times, well_names, metric)
    if not flowing.any():
        raise ValueError("the observed record has no report time at which a producer flows with a "
                         "bottom-hole pressure: nothing to match (check its wells and days)")
    yield frames.stage("invert", 0.86, f"matching {int(flowing.sum())} (time, well) points of the record")
    res = family.invert_record(built, params, observed, sigma, flowing=flowing.astype(np.float32),
                               n_iter=budget.gn_iter, n_samples=budget.n_samples)
    m, sd = np.asarray(res["m"]), np.asarray(res["sd"])
    chain = np.asarray(res["samples"]) if res.get("samples") else m[None, :]
    posterior = []
    for j, name in enumerate(names):
        draws = chain[:, j]
        q05, q95 = (float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))) if len(draws) > 1 else \
            (float(m[j] - 1.645 * sd[j]), float(m[j] + 1.645 * sd[j]))
        phys = lambda v: (np.exp(prior["kvkh"] * v) if name == "kvkh"     # noqa: E731
                          else base_contacts.get(name, 0.0) + prior[name] * v)
        posterior.append({
            "name": name, "label": variants.LABELS[name],
            "unit": "-" if name == "kvkh" else ("m" if metric else "ft"),
            "prior_mean": float(phys(0.0)), "prior_low": float(phys(-1.645)), "prior_high": float(phys(1.645)),
            "best": float(phys(m[j])), "median": float(phys(float(np.median(draws)))),
            "low": float(phys(q05)), "high": float(phys(q95)),
            "u": float(m[j]), "u_sd": float(sd[j]),
            "samples": [float(phys(v)) for v in draws[:: max(1, len(draws) // 400)]],
        })
    yield frames.posterior({"properties": posterior, "misfit": float(res["misfit"]),
                            "acceptance": res.get("acceptance"), "sigma": sigma, "emulator": emulator,
                            "points": int(flowing.sum())})

    # ---- 5. verify: the simulator at the posterior mean ---------------------------------------
    yield frames.stage("verify", 0.92, "OPM Flow at the posterior mean")
    best = {"name": "posterior", "u": m.tolist(), "held_out": True,
            "values": variants.physical(m, names, prior, base_contacts), "stem": ""}
    yield from _simulate_family(tree, member, root, [best], label="verify")
    verify = family.summary_record(str(root / best["stem"]), well_names, times)
    if metric:
        verify = _record_to_field(verify)
    bhp, q = family.record_at(built, params, np.asarray(m, np.float32))
    gnn = _record_quantities(np.asarray(bhp, np.float64), np.asarray(q, np.float64))
    simulated = _record_quantities(verify["bhp"], verify["q"])
    field = _record_quantities(observed["bhp"], observed["q"], mask=observed.get("mask"))
    yield from common.well_frames(times, well_names, simulator=simulated, gnn=gnn, observed=field)

    def rms(a, b, ok):
        return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.any() else None

    ok = flowing > 0
    yield frames.metrics(0, {
        "properties": len(names), "runs": budget.n_train, "held_out": budget.n_held,
        "emulator_bhp_rmse": emulator.get("bhp_rmse"), "emulator_gor_rmse": emulator.get("gor_rmse"),
        "emulator_wct_rmse": emulator.get("wct_rmse"), "emulator_p_rmse": emulator.get("p_rmse"),
        "misfit": float(res["misfit"]), "acceptance": res.get("acceptance"),
        "verify_bhp_rms": rms(verify["bhp"], observed["bhp"], ok),
        "gnn_bhp_rms": rms(np.asarray(bhp, np.float64), observed["bhp"], ok),
        **{f"posterior_{p['name']}": p["best"] for p in posterior},
        "wall_s": time.time() - t_start,
    })
    yield frames.stage("done", 1.0, "best fit: " + ", ".join(
        f"{p['label']} {p['best']:.4g} (90 % in [{p['low']:.4g}, {p['high']:.4g}])" for p in posterior))


# --------------------------------------------------------------------------- the runs
def _simulate_family(tree: dict[str, bytes], member: str, root: Path, samples: list[dict],
                     label: str = "sample") -> Iterator[Item]:
    """Write every sample's deck tree under ``root/<name>`` and run Flow on those without
    results, a few at a time; set each sample's ``stem``. Yields a ``sample`` message as each
    run starts and finishes."""
    from builder import variants

    from .. import simulate

    jobs = []
    for s in samples:
        folder = root / s["name"]
        variant_tree = variants.variant(tree, member, s["values"])
        for path, data in variant_tree.items():
            target = folder / path
            if target.is_file() and target.read_bytes() == data:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        deck = folder / member
        s["stem"] = str(Path(s["name"]) / Path(member).parent / Path(member).stem.upper())
        jobs.append((s, deck))
    total = len(jobs)
    pending = [(s, deck) for s, deck in jobs if not simulate.results_present(deck)]
    for index, (s, deck) in enumerate(jobs):
        if (s, deck) not in pending:
            yield frames.sample(index + 1, total, s["name"], "reused")
    if not pending:
        return
    workers = max(1, min(4, (os.cpu_count() or 2) // 2, len(pending)))

    def one(deck: Path) -> float:
        t0 = time.time()
        for _ in simulate.iter_flow(deck, threads=1):
            pass
        return time.time() - t0

    finished = total - len(pending)
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, deck): s for s, deck in pending}
        for s, _ in pending[:workers]:
            yield frames.sample(finished, total, s["name"], "running")
        for future in cf.as_completed(futures):
            s = futures[future]
            finished += 1
            try:
                seconds = future.result()
            except Exception as exc:  # noqa: BLE001 - a failed variant fails the family: say which
                yield frames.sample(finished, total, s["name"], "failed")
                raise RuntimeError(f"OPM Flow failed on {label} {s['name']} "
                                   f"({json.dumps(s['values'])}): {exc}") from exc
            yield frames.sample(finished, total, s["name"], "done", seconds)
            yield frames.stage(label if label != "sample" else "sample",
                               0.10 + 0.30 * finished / total if label == "sample" else 0.94,
                               f"{finished} of {total} runs simulated")
    for s, deck in jobs:
        # Flow writes beside the deck with the stem upper-cased; a ``RESULTS`` folder is honoured too.
        for folder in (deck.parent, deck.parent / "RESULTS"):
            found = [p for p in folder.glob("*.UNRST") if p.stem.upper() == deck.stem.upper()]
            if found:
                s["stem"] = str(found[0].with_suffix("").relative_to(root))
                break


def clear_family(work: Path) -> None:
    """Forget a model's families (runs and trained networks)."""
    shutil.rmtree(work / "family", ignore_errors=True)


# --------------------------------------------------------------------------- records
def record_arrays(rows: list[dict[str, Any]], times: np.ndarray, wells: list[str],
                  metric: bool) -> tuple[dict[str, np.ndarray], np.ndarray]:
    r"""The field record at the model's report times: ``bhp`` ``(T, W)`` [psia] and ``q``
    ``(T, W, 3)`` (oil, water, gas; produced positive; stb/d, stb/d, Mscf/d), ``NaN`` where the
    record has nothing, and the mask of the points the inversion matches — a producer flowing
    oil with a measured pressure. A record row is matched to the report time within half a day
    of its ``days``."""
    T, W = len(times), len(wells)
    bhp = np.full((T, W), np.nan)
    q = np.full((T, W, 3), np.nan)
    index = {name.upper(): w for w, name in enumerate(wells)}
    unmatched = 0
    for row in rows:
        w = index.get(str(row["well"]).upper())
        if w is None:
            unmatched += 1
            continue
        t = int(np.abs(times - float(row["days"])).argmin())
        if abs(times[t] - float(row["days"])) > 0.5:
            continue
        if row.get("WBHP") is not None:
            bhp[t, w] = float(row["WBHP"]) * (_BAR if metric else 1.0)
        for c, (key, scale) in enumerate((("WOPR", _LIQ), ("WWPR", _LIQ), ("WGPR", _GAS))):
            if row.get(key) is not None:
                q[t, w, c] = float(row[key]) * (scale if metric else 1.0)
    oil = np.nan_to_num(q[..., 0])
    peak = np.maximum(oil.max(axis=0, keepdims=True), 1e-12)
    flowing = (oil > 0.01 * peak) & np.isfinite(bhp)
    filled = {"bhp": np.nan_to_num(bhp), "q": np.nan_to_num(q), "mask": np.isfinite(bhp)}
    return filled, flowing.astype(np.float64)


def _record_quantities(bhp: np.ndarray, q: np.ndarray, mask=None) -> dict[str, np.ndarray]:
    """``bhp``, ``q`` (oil, water, gas; produced positive) as the summary vectors a well
    message carries; ``mask`` marks the pressures that exist (a field record's)."""
    bhp = np.asarray(bhp, np.float64)
    q = np.asarray(q, np.float64)
    qo, qw, qg = (np.maximum(q[..., i], 0.0) for i in range(3))
    wi, gi = (np.maximum(-q[..., i], 0.0) for i in (1, 2))
    liquid = qo + qw
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {"WBHP": np.where(mask, bhp, np.nan) if mask is not None else bhp,
               "WOPR": qo, "WWPR": qw, "WGPR": qg, "WWIR": wi, "WGIR": gi,
               "WGOR": np.where(qo > 1e-9, qg / np.maximum(qo, 1e-9), np.nan),
               "WWCT": np.where(liquid > 1e-9, qw / np.maximum(liquid, 1e-9), np.nan)}
    if mask is not None:
        for k in ("WOPR", "WWPR", "WGPR", "WWIR", "WGIR", "WGOR", "WWCT"):
            out[k] = np.where(mask, out[k], np.nan)
    return out


def _record_to_field(record: dict) -> dict:
    out = dict(record)
    out["bhp"] = np.asarray(record["bhp"]) * _BAR
    q = np.asarray(record["q"], np.float64).copy()
    q[..., :2] *= _LIQ
    q[..., 2] *= _GAS
    out["q"] = q
    return out


def _to_field(data) -> None:
    """A METRIC family's rates and records in FIELD units, as the physics rows read them."""
    rates = np.asarray(data.rates, np.float64).copy()       # residual order: water, oil, gas
    rates[..., :2] *= _LIQ
    rates[..., 2] *= _GAS
    data.rates = rates
    data.records = [_record_to_field(r) for r in data.records]


def _as64(tree):
    import jax

    return jax.tree.map(lambda x: x.astype("float64"), tree)
