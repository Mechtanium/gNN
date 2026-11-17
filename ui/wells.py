"""The Wells & schedule tab: the wells and their paths, the schedule of controls with the
history | prediction boundary, and the observed field record a property inversion matches."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import perdlit as pl

from builder import design as B
from builder import records as R
from builder.project import RECORD_COLUMNS, Project, read_record_csv, read_record_summary

from . import session
from .common import QUANTITIES, choice


def render(p: Project, gen: int) -> None:
    tabs = pl.tabs(["Wells", "Schedule", "Observed data"], key="wells_tab")
    with tabs[0]:
        wells_designed(p.design, gen) if p.design is not None else wells_imported(p, gen)
    with tabs[1]:
        schedule_designed(p.design, gen) if p.design is not None else schedule_imported(p, gen)
    with tabs[2]:
        observed(p, gen)


# ------------------------------------------------------------------------- wells
def wells_designed(d: dict, gen: int) -> None:
    g = d["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    units = "ft" if d["units"] == "FIELD" else "m"
    rows = [{"name": w["name"], "kind": w["kind"], "phase": w.get("phase", "OIL"), "group": w.get("group", "G1"),
             "diameter": float(w.get("diameter", 0.5)), "skin": float(w.get("skin", 0.0))} for w in d["wells"]]
    edited = pl.data_editor(rows, key=f"g{gen}:wells", num_rows="dynamic", height=210, label="Wells", columns={
        "name": {"type": "text"}, "kind": {"type": "select", "options": ["producer", "injector"], "default": "producer"},
        "phase": {"type": "select", "options": ["OIL", "WATER", "GAS"], "default": "OIL",
                  "help": "A producer's preferred phase; an injector's injected phase"},
        "group": {"type": "text", "default": "G1"},
        "diameter": {"type": "number", "min": 0.01, "unit": units, "default": B.from_field(0.5, "length", d["units"])},
        "skin": {"type": "number", "default": 0.0}})
    if edited != rows:
        def wells(dd):
            old = {w["name"]: w for w in dd["wells"]}
            new = []
            for i, r in enumerate(edited):
                name = (r["name"] or f"W{i + 1}").strip()
                base = old.get(name) or (dd["wells"][i] if i < len(dd["wells"]) else None) or {
                    "trajectory": [[(nx + 1) // 2, (ny + 1) // 2, 1], [(nx + 1) // 2, (ny + 1) // 2, nz]]}
                new.append({**base, "name": name, "kind": r["kind"] or "producer", "phase": r["phase"] or "OIL",
                            "group": (r["group"] or "G1").strip(), "diameter": float(r["diameter"] or 0.5),
                            "skin": float(r["skin"] or 0.0)})
            renamed = {rows[i]["name"]: new[i]["name"] for i in range(min(len(rows), len(new)))}
            for ev in dd["schedule"]["events"]:
                ev["well"] = renamed.get(ev["well"], ev["well"])
            dd["wells"] = new
        session.edit_design(wells)
        d = session.design()
    if not d["wells"]:
        pl.caption("Add a well to the table to place it.")
        return
    names = [w["name"] for w in d["wells"]]
    left, right = pl.columns([2, 3], gap="large")
    with left:
        name = choice("Well", names, names[0], "well", gen)
        index = names.index(name)
        well = d["wells"][index]
        pl.subheader(f"Path of {name}")
        rows = [{"i": float(a), "j": float(b), "k": float(c)} for a, b, c in well["trajectory"]]
        edited = pl.data_editor(rows, key=f"g{gen}:traj:{name}", num_rows="dynamic", height=170, columns={
            "i": {"type": "number", "min": 1, "max": nx}, "j": {"type": "number", "min": 1, "max": ny},
            "k": {"type": "number", "min": 1, "max": nz}},
            help="Points the well passes through, in cell units (the centre of cell 3, 4, 2 is 3, 4, 2)")
        if edited != rows and edited:
            points = [[r["i"] or 1, r["j"] or 1, r["k"] or 1] for r in edited]
            session.edit_design(lambda dd: dd["wells"][index].update(trajectory=points))
        try:
            conns = B.completions(session.design(), session.design()["wells"][index])
            pl.dataframe([{"I": i, "J": j, "K": k, "along": a} for i, j, k, a in conns], height=200)
            pl.caption(f"{len(conns)} completed cell(s): every active cell the path passes through is opened.")
        except B.DesignError as exc:
            pl.markdown(f"**{exc}**")
    with right:
        tool = pl.radio("Map tool", ["Look", "Move the head (vertical well)", "Draw a horizontal path"], 0,
                        key=f"g{gen}:well_tool", horizontal=True)
        layer = int(round(well["trajectory"][-1][2])) if well["trajectory"] else 1
        layer = pl.slider("Layer", 1, nz, min(max(layer, 1), nz), key=f"g{gen}:well_layer:{name}") if nz > 1 else 1
        try:
            arrs = session.arrays(d)
            geom = session.geometry(d)
        except B.DesignError as exc:
            pl.markdown(f"**{exc}**")
            return
        values = np.where(geom["actnum"][layer - 1] > 0, np.log10(np.maximum(arrs["PERMX"][layer - 1], 1e-4)), np.nan)
        lines = [{"points": [[p[0] - 0.5, p[1] - 0.5] for p in w["trajectory"]], "label": "",
                  "color": "#1a1a1a" if w["name"] == name else "#868e96"} for w in d["wells"] if len(w["trajectory"]) > 1]
        markers = [{"i": w["trajectory"][0][0], "j": w["trajectory"][0][1], "label": w["name"],
                    "color": "#d6336c" if w["name"] == name else None} for w in d["wells"]]
        mode, selection = "none", None
        if tool.startswith("Move"):
            mode, selection = "point", [int(round(well["trajectory"][0][0])), int(round(well["trajectory"][0][1]))]
        elif tool.startswith("Draw"):
            mode = "polyline"
        chosen = pl.map2d(nx, ny, values=values, key=f"g{gen}:wellmap:{name}:{mode}", mode=mode, selection=selection,
                          cmap="rainbow", legend=f"log10 PERMX, layer {layer}", height=470, markers=markers,
                          lines=lines)
        if mode == "point" and chosen and chosen != selection:
            ks = [p[2] for p in well["trajectory"]] or [1, nz]
            path = [[chosen[0], chosen[1], min(ks)], [chosen[0], chosen[1], max(ks)]]
            session.edit_design(lambda dd: dd["wells"][index].update(trajectory=path))
        if mode == "polyline" and chosen:
            pl.caption(f"A path of {len(chosen)} node(s) in layer {layer}.")
            if pl.action(f"Make it {name}'s path", key="well_path", type="primary"):
                head = well["trajectory"][0]
                path = [[head[0], head[1], 1]] + [[a + 0.5, b + 0.5, layer] for a, b in chosen]
                session.edit_design(lambda dd: dd["wells"][index].update(trajectory=path))
                pl.set_value(f"g{gen}:wellmap:{name}:{mode}", None)
        pl.caption("The map shows permeability on the chosen layer; the selected well is drawn in black. A drawn "
                   "path starts at the well's head, drops through the layers above, and runs along the layer.")


def wells_imported(p: Project, gen: int) -> None:
    deck = p.deck
    specs = []
    for keyword in deck.keywords("WELSPECS"):
        for row in R.parse_records(keyword.data()):
            if row:
                specs.append({"well": row[0], "group": row[1] if len(row) > 1 else None,
                              "I": row[2] if len(row) > 2 else None, "J": row[3] if len(row) > 3 else None,
                              "phase": row[5] if len(row) > 5 else None})
    if not specs:
        pl.caption("The deck defines no well (WELSPECS).")
        return
    pl.dataframe(specs, height=260)
    names = list(dict.fromkeys(s["well"] for s in specs))
    name = choice("Connections of", names, names[0], "well", gen)
    rows = []
    for keyword in deck.keywords("COMPDAT"):
        for row in R.parse_records(keyword.data()):
            if row and (row[0] == name or (row[0] and "*" in row[0] and name.startswith(row[0].rstrip("*")))):
                rows.append({"I": row[1], "J": row[2], "K1": row[3], "K2": row[4], "status": row[5] if len(row) > 5 else None,
                             "diameter": row[8] if len(row) > 8 else None, "skin": row[10] if len(row) > 10 else None})
    pl.dataframe(rows or [{"I": "", "J": "", "K1": "", "K2": "", "status": "", "diameter": "", "skin": ""}], height=220)
    pl.caption("An imported deck's wells and controls are edited on the Keywords tab (SCHEDULE section).")


# ------------------------------------------------------------------------- schedule
def schedule_designed(d: dict, gen: int) -> None:
    s = d["schedule"]
    from builder.units import label

    units = d["units"]
    c1, c2, c3, c4 = pl.columns(4, gap="small")
    with c1:
        pl.text_input("Start (Project tab)", d["start"], key=f"g{gen}:sch_start_ro", help="Set on the Project tab")
    with c2:
        end = pl.text_input("End (YYYY-MM-DD)", s["end"], key=f"g{gen}:sch_end")
        if end != s["end"]:
            session.edit_design(lambda dd: dd["schedule"].update(end=end))
    with c3:
        cadence = choice("Report steps", ["monthly", "quarterly", "yearly"], s.get("report", "monthly"), "sch_report", gen)
        if cadence != s.get("report"):
            session.edit_design(lambda dd: dd["schedule"].update(report=cadence))
    with c4:
        boundary = pl.text_input("History | prediction boundary", s.get("history_end") or "", key=f"g{gen}:sch_hist",
                                 help="The date a history match stops at and a forecast starts from (empty: none)")
        if boundary != (s.get("history_end") or ""):
            session.edit_design(lambda dd: dd["schedule"].update(history_end=boundary or None))
    try:
        dates = B.report_dates(session.design())
        days = B.history_end_days(session.design())
        pl.caption(f"{len(dates)} report steps to {dates[-1].isoformat()}"
                   + (f"; the history ends at day {days:g}." if days is not None else "; no boundary: the forecast path is off."))
    except B.DesignError as exc:
        pl.markdown(f"**{exc}**")
    pl.subheader("Controls")
    names = [w["name"] for w in d["wells"]] or [""]
    rows = [{"date": ev["date"], "well": ev["well"], "control": ev["control"], "rate": ev.get("rate"),
             "bhp": ev.get("bhp")} for ev in s["events"]]
    edited = pl.data_editor(rows, key=f"g{gen}:events", num_rows="dynamic", height=320, columns={
        "date": {"type": "text", "default": d["start"], "help": "YYYY-MM-DD: the control holds from this date"},
        "well": {"type": "select", "options": names, "default": names[0]},
        "control": {"type": "select", "options": [*B.PRODUCER_CONTROLS, *[c for c in B.INJECTOR_CONTROLS if c != "BHP"],
                                                  *B.STATUS_EVENTS], "default": "ORAT",
                    "help": "Producers: ORAT, WRAT, GRAT, LRAT, RESV or BHP; injectors: RATE or BHP; SHUT, OPEN, STOP"},
        "rate": {"type": "number", "min": 0.0, "label": "target",
                 "help": f"The controlled rate ({label('liquid', units)}, gas {label('gas', units)}); the pressure for BHP"},
        "bhp": {"type": "number", "min": 0.0, "label": "BHP limit", "unit": label("pressure", units)}})
    if edited != rows:
        events = [{"date": (r["date"] or d["start"]).strip(), "well": r["well"] or names[0],
                   "control": r["control"] or "ORAT", "rate": r["rate"], "bhp": r["bhp"]} for r in edited]
        session.edit_design(lambda dd: dd["schedule"].update(events=events))
    _controls_chart(session.design())


def _controls_chart(d: dict) -> None:
    """Each well's controlled rate over time, as steps."""
    try:
        start = B._date(d["start"])
        end = B._date(d["schedule"]["end"])
    except B.DesignError:
        return
    series = {}
    for w in d["wells"]:
        evs = sorted((e for e in d["schedule"]["events"] if e["well"] == w["name"]), key=lambda e: e["date"])
        xs, ys, level = [0.0], [0.0], 0.0
        for e in evs:
            try:
                t = float((B._date(e["date"]) - start).days)
            except B.DesignError:
                continue
            xs += [t, t]
            new = 0.0 if e["control"] in ("SHUT", "STOP") else float(e.get("rate") or 0.0)
            ys += [level, new]
            level = new
        xs.append(float((end - start).days))
        ys.append(level)
        series[w["name"]] = (xs, ys)
    if not series:
        return
    for name, (xs, ys) in list(series.items())[:6]:
        pl.line_chart({"days": xs, name: ys}, x="days", y=name, labels=["days", f"{name} target"], height=120,
                      key=f"ctrl:{name}")


def _report_days(p: Project) -> list[tuple[float, str]]:
    """``(days, date)`` of every report step of an imported deck, read by OPM's parser."""
    def read():
        import opm.io
        from opm.io.ecl_state import EclipseState
        from opm.io.parser import ParseContext, Parser
        from opm.io.schedule import Schedule

        tree, root = p.tree()
        with tempfile.TemporaryDirectory() as tmp:
            for path, data in tree.items():
                target = Path(tmp) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            deck = Parser().parse(str(Path(tmp) / root), ParseContext([("PARSE_*", opm.io.action.ignore)]))
            sched = Schedule(deck, EclipseState(deck))
            steps = list(sched.reportsteps)
        start = steps[0]
        return [(float((s - start).total_seconds()) / 86400.0, s.date().isoformat()) for s in steps]

    try:
        return session.cached("report_days", str(pl.session_state.get("revision", 0)), read)
    except Exception as exc:  # noqa: BLE001 - a deck the parser refuses: no list of dates
        pl.markdown(f"**The deck's schedule could not be read:** {exc}")
        return []


def schedule_imported(p: Project, gen: int) -> None:
    steps = _report_days(p)
    if not steps:
        return
    pl.caption(f"{len(steps)} report steps, from {steps[0][1]} to {steps[-1][1]}.")
    labels = ["not set"] + [f"day {d:g} ({date})" for d, date in steps[1:-1]]
    current = p.history_end_days()
    index = 0
    if current is not None:
        nearest = min(range(1, len(steps) - 1), key=lambda i: abs(steps[i][0] - current), default=None)
        index = nearest if nearest is not None else 0
    chosen = pl.selectbox("History | prediction boundary", labels, index, key=f"g{gen}:boundary",
                          help="A history match with a forecast matches up to this report step and forecasts past it")
    value = None if chosen == "not set" else steps[labels.index(chosen)][0]
    if value != current:
        if value is None:
            p.settings.pop("history_end_days", None)
        else:
            p.settings["history_end_days"] = value
        session.touch()
    pl.caption("The controls themselves are the deck's: edit them on the Keywords tab (SCHEDULE section).")


# ------------------------------------------------------------------------- the field record
def observed(p: Project, gen: int) -> None:
    pl.caption("The wells' measured history: what a property inversion matches. Enter it here, paste it from a "
               "spreadsheet, or import a CSV (well, days and any of the columns below) or an Eclipse summary "
               "(a zip of the .SMSPEC and .UNSMRY, or an .ESMRY). Rates and pressures are in the deck's units.")
    if p.design is not None:
        wells = [w["name"] for w in p.design["wells"]]
    else:
        from .model import imported_wells

        wells = imported_wells(p)
    left, right = pl.columns([3, 2], gap="large")
    with left:
        rows = [{c: r.get(c) for c in RECORD_COLUMNS} for r in p.record]
        columns = {"well": {"type": "select", "options": wells or [""], "default": wells[0] if wells else ""},
                   "days": {"type": "number", "min": 0.0, "help": "days since the start of the schedule"}}
        columns.update({c: {"type": "number", "min": 0.0} for c in RECORD_COLUMNS[2:]})
        edited = pl.data_editor(rows, key=f"g{gen}:record", num_rows="dynamic", height=420, columns=columns,
                                label="Observed record")
        if edited != rows:
            p.record = [r for r in edited if r.get("well") and r.get("days") is not None]
            session.touch()
    with right:
        upload = pl.file_uploader("Import a record", type=["csv", "zip", "esmry", "ESMRY"], key="record_file")
        if upload is not None and pl.session_state.get("record_seen") != (upload.name, upload.size):
            try:
                if upload.name.lower().endswith(".csv"):
                    record = read_record_csv(upload.read())
                elif upload.name.lower().endswith(".esmry"):
                    record = read_record_summary({upload.name: upload.read()})
                else:
                    import io
                    import zipfile

                    with zipfile.ZipFile(io.BytesIO(upload.read())) as z:
                        record = read_record_summary({n: z.read(n) for n in z.namelist()
                                                      if n.upper().endswith((".SMSPEC", ".UNSMRY", ".ESMRY"))})
            except Exception as exc:  # noqa: BLE001 - say what was wrong with the file
                pl.markdown(f"**{upload.name} could not be read:** {exc}")
            else:
                pl.session_state["record_seen"] = (upload.name, upload.size)
                p.record = record
                session.touch()
                pl.set_value(f"g{gen}:record", None)
                session.notice(f"Imported {len(record)} record rows from {upload.name}")
        if p.record and pl.action("Clear the record", key="record_clear", icon="remove"):
            p.record = []
            session.touch()
        if p.record:
            names = sorted({r["well"] for r in p.record})
            well = pl.selectbox("Plot", names, 0, key=f"g{gen}:record_plot")
            mine = sorted((r for r in p.record if r["well"] == well), key=lambda r: r["days"])
            for q in ("WBHP", "WOPR", "WWPR", "WGPR"):
                pts = [(r["days"], r[q]) for r in mine if r.get(q) is not None]
                if pts:
                    pl.line_chart({"days": [a for a, _ in pts], q: [b for _, b in pts]}, x="days", y=q,
                                  labels=["days", QUANTITIES[q]], height=140, key=f"rec:{q}")
