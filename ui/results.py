"""The tabs that show what the jobs made: the simulation, the three work paths.

Every display here is bound to the run of its toolbar button, so it fills in live while the
job streams and every viewer of the app sees the same run. A few tables are built from a
finished run's messages (the posterior); the page reruns as a run ends, so they appear then.
"""

from __future__ import annotations

import numpy as np
import perdlit as pl

from builder.project import Project
from modules import autopilot

from . import session
from .common import DYNAMIC_FIELDS, QUANTITIES, field_labels, payloads, wells_in


def _well_chart(run, sources: dict[str, str], key: str, *, x_marks=None, height: int = 300) -> None:
    """A chart of one well's quantity, from the sources ``{source: legend}`` of a run."""
    names = wells_in(run)
    if not names:
        pl.caption("The wells' records appear here once the run has sent them.")
        return
    c1, c2 = pl.columns(2, gap="small")
    with c1:
        well = pl.selectbox("Well", names, 0, key=f"{key}:well")
    with c2:
        quantity = pl.selectbox("Quantity", list(QUANTITIES), 0, key=f"{key}:q")
    pl.line_chart(run, x=2, y=[f"7.{quantity}_{s}" for s in sources], where=[(0, "well"), ("7.well", well)],
                  legend=list(sources.values()), labels=["days", QUANTITIES[quantity]], height=height,
                  x_marks=x_marks, key=f"{key}:chart")


def _progress(run, key: str) -> None:
    pl.progress(run, field=2, where=(0, "stage"), label="7.message", key=key)
    pl.write(run)


# ------------------------------------------------------------------------- simulation
def simulation(p: Project, check_run, sim_run) -> None:
    tabs = pl.tabs(["Results", "Check", "Deck"], key="sim_tab")
    with tabs[0]:
        _progress(sim_run, "sim_progress")
        left, right = pl.columns([3, 2], gap="medium")
        with left:
            pl.grid3d(sim_run, field="S_w", t_index=7, field_labels=field_labels(), z_scale=5, height=520, key="sim3d")
        with right:
            c1, c2 = pl.columns(2, gap="small")
            with c1:
                pl.metric("Cells", sim_run, field="7.cells", where=(0, "metrics"))
                pl.metric("Report steps", sim_run, field="7.report_steps", where=(0, "metrics"))
            with c2:
                pl.metric("Wells", sim_run, field="7.wells", where=(0, "metrics"))
                pl.metric("Days", sim_run, field="7.days", where=(0, "metrics"), fmt=".0f")
            pl.subheader("OPM Flow")
            pl.feed(sim_run, fields=["7.message"], labels=["Report steps"], where=(0, "log"), height=230,
                    key="flow_log", empty="Press Simulate: the simulator's report steps show here.")
        pl.subheader("Wells")
        _well_chart(sim_run, {"simulator": "OPM Flow"}, "sim_wells")
    with tabs[1]:
        _progress(check_run, "check_progress")
        pl.feed(check_run, fields=["7.level", "7.where", "7.message"], labels=["", "Where", "Message"],
                where=(0, "diag"), height=520, key="check_diags", empty="Press Check: OPM Flow's findings show here.")
    with tabs[2]:
        try:
            tree, root = p.tree()
        except Exception as exc:  # noqa: BLE001 - a design that cannot be generated yet
            pl.markdown(f"**The deck cannot be generated yet:** {exc}")
            return
        files = sorted(tree)
        chosen = pl.selectbox("File", files, files.index(root), key="deck_file")
        text = tree[chosen].decode("latin-1")
        if len(text) > 300_000:
            pl.caption(f"{len(text):,} characters: the first 300,000 are shown.")
            text = text[:300_000]
        pl.code(text, language="text")


# ------------------------------------------------------------------------- the work paths
def _composition(path: dict) -> None:
    with pl.expander("The composition this path trains (fixed)"):
        pl.json({**autopilot.AUTOPILOT, **path})


def _training(run, key: str, groups: list[str], *, x_marks=None) -> None:
    c1, c2, c3, c4 = pl.columns(4, gap="small")
    with c1:
        pl.metric("Iteration", run, field=1, where=(0, "loss"))
    with c2:
        pl.metric("Loss", run, field=2, where=(0, "loss"), fmt=".4g")
    with c3:
        pl.metric("Line-search step", run, field="7.eta", where=(0, "loss"), fmt=".3g")
    with c4:
        pl.metric("Wall time [s]", run, field="7.wall_s", where=(0, "loss"), fmt=".0f")
    left, right = pl.columns([2, 3], gap="medium")
    with left:
        pl.line_chart(run, x=1, y=[2, *[f"7.g_{g}" for g in groups]], where=(0, "loss"), y_scale="log",
                      legend=["total", *groups], labels=["iteration", "loss"], height=360, key=f"{key}:loss")
    with right:
        pl.grid3d(run, field="S_w", t_index=7, field_labels=DYNAMIC_FIELDS, z_scale=5, height=420, key=f"{key}:3d")


def history(p: Project, run) -> None:
    pl.caption("Work path 1 — the full-time history match: the network learns the simulator's cell states at "
               "every report step and the wells' records, with the black-oil balances enforced. Press History match.")
    _progress(run, "hm_progress")
    _training(run, "hm", ["pde", "ic", "data", "well"])
    pl.subheader("The wells: simulated and matched")
    _well_chart(run, {"simulator": "OPM Flow", "gnn": "gNN"}, "hm_wells")
    c1, c2, c3, c4 = pl.columns(4, gap="small")
    with c1:
        pl.metric("Pressure RMSE, mean [psi]", run, field="7.pressure_rmse_mean_t", where=(0, "metrics"), fmt=".1f")
    with c2:
        pl.metric("Water saturation RMSE", run, field="7.swat_rmse_final_t", where=(0, "metrics"), fmt=".3f")
    with c3:
        pl.metric("Gas saturation RMSE", run, field="7.sgas_rmse_final_t", where=(0, "metrics"), fmt=".3f")
    with c4:
        pl.metric("BHP RMS [psi]", run, field="7.bhp_rms", where=(0, "metrics"), fmt=".1f")
    _composition(autopilot.HISTORY)


def forecast(p: Project, run) -> None:
    boundary = p.history_end_days()
    pl.caption("Work path 2 — the joint history match and forecast: matched on the simulator's run up to the "
               "history | prediction boundary, carried past it by the balances, the schedule's controls and the "
               "phase closures alone; the simulator's run past the boundary only scores it."
               + (f" The boundary is at day {boundary:g}." if boundary is not None else
                  " Set the boundary on the Schedule tab first."))
    _progress(run, "fc_progress")
    _training(run, "fc", ["pde", "ic", "data", "well", "ctrl", "eq", "mb"])
    pl.subheader("The wells across the boundary")
    marks = [(boundary, "history | forecast")] if boundary is not None else None
    _well_chart(run, {"simulator": "OPM Flow", "gnn": "gNN"}, "fc_wells", x_marks=marks)
    c1, c2, c3, c4 = pl.columns(4, gap="small")
    with c1:
        pl.metric("History pressure RMSE [psi]", run, field="7.history_pressure_rmse_mean_t", where=(0, "metrics"), fmt=".1f")
    with c2:
        pl.metric("Forecast pressure RMSE [psi]", run, field="7.forecast_pressure_rmse_mean_t", where=(0, "metrics"), fmt=".1f")
    with c3:
        pl.metric("Forecast BHP RMS [psi]", run, field="7.forecast_bhp_rms", where=(0, "metrics"), fmt=".1f")
    with c4:
        pl.metric("Supersaturated cells", run, field="7.supersaturated_share", where=(0, "metrics"), fmt=".3f")
    _composition(autopilot.FORECAST)


# ------------------------------------------------------------------------- inversion
def inversion(p: Project, run, gen: int) -> None:
    from builder.variants import LABELS

    pl.caption("Work path 3 — reservoir property inversion: the model is simulated at sampled values of its "
               "uncertain properties, one network is taught the whole family of runs, and the field record is "
               "matched through it. Which properties are inverted follows from the model: the vertical "
               "permeability always, a fluid contact when it lies within the reservoir.")
    left, right = pl.columns([2, 3], gap="large")
    with left:
        pl.subheader("Prior")
        units = "ft" if (p.design is None or p.design["units"] == "FIELD") else "m"
        rows = [{"property": LABELS[k], "sd": float(p.prior[k]),
                 "unit": "ln(kv/kh)" if k == "kvkh" else units} for k in ("kvkh", "woc", "goc")]
        edited = pl.data_editor(rows, key=f"g{gen}:prior", height=150, columns={
            "property": {"editable": False}, "unit": {"editable": False},
            "sd": {"type": "number", "min": 0.01, "label": "standard deviation"}},
            help="The width of each property's prior about the model's own value (90 % of it within 1.645 sd)")
        if [r["sd"] for r in edited] != [r["sd"] for r in rows]:
            for key, r in zip(("kvkh", "woc", "goc"), edited):
                p.prior[key] = float(r["sd"] or rows[0]["sd"])
            session.touch()
        b = autopilot.Budget()
        pl.caption(f"{b.n_train} training and {b.n_held} held-out simulator runs, {b.n_iter} training iterations, "
                   f"a Gauss–Newton search from four starts and {b.n_samples:,} posterior samples. The runs and the "
                   "trained family are kept: inverting another record of this model only searches again.")
        if not p.record:
            pl.markdown("**There is no field record yet** — enter or import it on Wells & schedule › Observed data.")
        else:
            pl.caption(f"The record holds {len(p.record)} rows for {len({r['well'] for r in p.record})} well(s).")
    with right:
        _progress(run, "inv_progress")
        pl.feed(run, fields=["7.name", "7.status", "7.seconds"], labels=["Run", "Status", "Seconds"],
                where=(0, "sample"), height=170, newest_first=True, key="inv_samples",
                empty="The simulator runs of the family show here.")
        pl.line_chart(run, x=1, y=[2, "7.g_pde", "7.g_ic", "7.g_data"], where=(0, "loss"), y_scale="log",
                      legend=["total", "pde", "ic", "data"], labels=["iteration", "loss"], height=230, key="inv_loss")

    pl.subheader("Posterior")
    post = payloads(run, "posterior")
    if not post:
        pl.caption("The inverted properties appear here when the search is done.")
    else:
        latest = post[-1]
        props = latest["properties"]
        pl.table([{"property": q["label"], "unit": q["unit"],
                   "prior 90 %": f"{q['prior_low']:.4g} – {q['prior_high']:.4g}",
                   "best fit": f"{q['best']:.4g}", "median": f"{q['median']:.4g}",
                   "posterior 90 %": f"{q['low']:.4g} – {q['high']:.4g}"} for q in props])
        cols = pl.columns(len(props), gap="small")
        for col, q in zip(cols, props):
            with col:
                samples = np.asarray(q["samples"], float)
                if samples.size > 1:
                    counts, edges = np.histogram(samples, bins=20)
                    pl.bar_chart([{"value": float(0.5 * (a + c)), "samples": int(n)}
                                  for a, c, n in zip(edges[:-1], edges[1:], counts)],
                                 x="value", y="samples", labels=[q["label"], "posterior samples"], height=180)
        emu = latest.get("emulator") or {}
        pl.caption("The family's error on the held-out runs — BHP {:.3g} psi, GOR {:.3g} Mscf/stb, water cut {:.3g} — "
                   "is added to the record's noise, so the intervals account for it. Misfit {:.4g} at the best fit; "
                   "{} of the record's points were matched.".format(
                       emu.get("bhp_rmse", float("nan")), emu.get("gor_rmse", float("nan")),
                       emu.get("wct_rmse", float("nan")), latest.get("misfit", float("nan")), latest.get("points", 0)))
    pl.subheader("The field record against the family and the simulator at the best fit")
    _well_chart(run, {"observed": "field", "gnn": "family", "simulator": "OPM Flow at the best fit"}, "inv_wells")
