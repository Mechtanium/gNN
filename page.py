"""The gNN application: build a reservoir model, hand it to the simulator, and run the work paths
on it — laid out as a desktop modelling package.

The toolbar holds the project commands and the five jobs (Check, Simulate, History match,
Match + forecast, Invert) with the progress of the latest; the sidebar the explorer of the
model; the tabs the stages of the work — Model, Wells & schedule, Simulation, History match,
Forecast, Inversion; the status bar what is open. Each tab's content lives in ``ui/``.

Every job receives the whole project as one bundle (``ui/session.py``), so the deck, the
history | prediction boundary, the field record and the prior always travel together; no job
takes a setting.
"""

import perdlit as pl

import workflow
from ui import model, results, session, wells
from ui.common import RETAIN

p = session.project()
name = p.name
bundle = session.Bundle(p)
deck = session.deck_name(p)
boundary = p.history_end_days()
gen = pl.session_state.get("generation", 0)

# ── toolbar: project commands, the jobs, and how the latest is getting on ──────────────
with pl.toolbar:
    template = pl.menu("New", list(session.TEMPLATES), key="new", icon="new",
                       help="Start a model from a design or from a sample deck")
    if template:
        session.replace(session.TEMPLATES[template]())
        pl.session_state["generation"] = gen + 1
        pl.set_value("top", "Model")
        pl.set_value("model_tab", "Project")
        session.notice(f"New project from {template}")
        pl.rerun()
    if pl.action("Open", key="open", icon="open", help="Open a .gnn project, a deck or a zipped deck tree"):
        pl.set_value("top", "Model")
        pl.set_value("model_tab", "Project")
    pl.download("Save", lambda: p.bundle()[0], filename=f"{name}.gnn", mime="application/zip", key="save",
                icon="save")
    pl.download("Export", lambda: p.deck_zip(), filename=f"{name}_deck.zip", mime="application/zip",
                key="export", icon="export")
    pl.divider()
    check_run = pl.button("Check", callable=workflow.check, args=[bundle, deck], key="run_check", icon="check",
                          type="secondary", deadline_s=1800, retain=RETAIN["check"],
                          help="OPM Flow reads the deck and builds the model, without simulating")
    sim_run = pl.button("Simulate", callable=workflow.simulate, args=[bundle, deck], key="run_simulate",
                        icon="play", type="secondary", deadline_s=12 * 3600, retain=RETAIN["simulate"],
                        help="Hand the model to OPM Flow and show its simulated history")
    pl.divider()
    hm_run = pl.button("History match", callable=workflow.history_match, args=[bundle, deck],
                       key="run_history", icon="chart", deadline_s=72 * 3600, retain=RETAIN["train"],
                       help="Work path 1: match the whole simulated history")
    fc_run = pl.button("Match + forecast", callable=workflow.history_forecast, args=[bundle, deck],
                       key="run_forecast", icon="clock", deadline_s=72 * 3600, retain=RETAIN["train"],
                       disabled=boundary is None,
                       help=("Work path 2: match the history to the boundary and forecast past it"
                             if boundary is not None else "Set the history | prediction boundary first (Schedule)"))
    inv_run = pl.button("Invert", callable=workflow.invert, args=[bundle, deck], key="run_invert",
                        icon="target", deadline_s=72 * 3600, retain=RETAIN["invert"], disabled=not p.record,
                        help=("Work path 3: invert the field record for the uncertain properties"
                              if p.record else "Enter or import the observed record first (Observed data)"))
    pl.spacer()
    runs = {"Check": check_run, "Simulation": sim_run, "History match": hm_run, "Forecast": fc_run,
            "Inversion": inv_run}
    started = [(r.started_at or 0.0, label, r) for label, r in runs.items() if not r.id.startswith("idle-")]
    if started:
        _, label, latest = max(started, key=lambda s: s[0])
        pl.progress(latest, field=2, where=(0, "stage"), label="7.message", key="toolbar_progress",
                    text=None)

# ── sidebar: the explorer ──────────────────────────────────────────────────────────────
with pl.sidebar:
    pl.header("Explorer")
    node = pl.tree(model.outline(p, runs), key=f"explorer{gen}", label=name, height=640)
    if node and node != pl.session_state.get("explorer_seen"):
        pl.session_state["explorer_seen"] = node
        for target_key, target in model.navigate(node, p):
            pl.set_value(target_key, target)

# ── status bar ─────────────────────────────────────────────────────────────────────────
with pl.statusbar:
    pl.text(f"{name} · {p.kind}")
    pl.divider()
    if p.design is not None:
        g = p.design["grid"]
        pl.text(f"{g['nx']} × {g['ny']} × {g['nz']} cells · {len(p.design['wells'])} wells · {p.design['units']}")
    else:
        pl.text(f"{len(p.deck.files)} file(s) · {sum(1 for _ in p.deck.walk())} keywords")
    pl.divider()
    pl.text("history | prediction at day " + (f"{boundary:g}" if boundary is not None else "— not set"))
    pl.divider()
    pl.text(f"field record: {len(p.record)} rows" if p.record else "no field record")
    pl.spacer()
    pl.write(check_run)

# ── the stages of the work ─────────────────────────────────────────────────────────────
for kind, text in session.take_notices():
    (pl.caption if kind == "note" else pl.markdown)(text if kind == "note" else f"**{text}**")

model_tab, wells_tab, sim_tab, hm_tab, fc_tab, inv_tab = pl.tabs(
    ["Model", "Wells & schedule", "Simulation", "History match", "Forecast", "Inversion"], key="top")
with model_tab:
    model.render(p, check_run, gen)
with wells_tab:
    wells.render(p, gen)
with sim_tab:
    results.simulation(p, check_run, sim_run)
with hm_tab:
    results.history(p, hm_run)
with fc_tab:
    results.forecast(p, fc_run)
with inv_tab:
    results.inversion(p, inv_run, gen)
