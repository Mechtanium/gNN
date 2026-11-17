"""The Model tab: the project, and the reservoir itself.

A designed model is edited section by section — the grid and its structure, the property
recipes, the fluids and rock–fluid functions, the initial equilibrium — with an areal map
of any property on any layer for drawing faults, painting cells and placing boxes. An
imported deck is edited keyword by keyword, with table editors for the keywords that are
tables. The 3-D view shows the model as OPM Flow builds it (the Check job).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import perdlit as pl

from builder import design as B
from builder import records as R
from builder.deck import SECTIONS, Deck
from builder.project import Project

from . import session
from .common import bound, choice, field_labels

OPS = {
    "constant": "A constant",
    "layers": "A value per layer",
    "trend": "A linear trend",
    "random": "A correlated random field",
    "transform": "log10 k = a + b · source",
    "copy": "A scaled copy of another",
    "box": "A value in a box",
    "cells": "A value in painted cells",
    "region": "A value in a region",
}
TOP_KINDS = ["flat", "tilted", "dome", "anticline"]


# ------------------------------------------------------------------------- the explorer
def outline(p: Project, runs: dict) -> list[dict]:
    nodes: list[dict[str, Any]] = [{"id": "project", "label": "Project", "icon": "folder"}]
    if p.design is not None:
        d = p.design
        g = d["grid"]
        nodes.append({"id": "model", "label": "Model", "icon": "cube", "expanded": True, "children": [
            {"id": "grid", "label": "Grid", "icon": "grid", "badge": f"{g['nx']}×{g['ny']}×{g['nz']}",
             "children": [{"id": f"fault:{i}", "label": f.get("name") or f"fault {i + 1}", "icon": "fault"}
                          for i, f in enumerate(g.get("faults", []))]},
            {"id": "properties", "label": "Properties", "icon": "layers",
             "children": [{"id": f"prop:{n}", "label": n} for n in B.PROPERTIES]},
            {"id": "regions", "label": "Regions", "icon": "map",
             "children": [{"id": f"prop:{n}", "label": n} for n in B.REGIONS]},
            {"id": "fluids", "label": "Fluids & rock", "icon": "fluid"},
            {"id": "init", "label": "Initialization", "icon": "target"},
        ]})
        well_names = [w["name"] for w in d["wells"]]
    else:
        outline_ = session.cached("outline", str(pl.session_state.get("revision", 0)), p.deck.outline)
        nodes.append({"id": "keywords", "label": "Keywords", "icon": "file", "expanded": True, "children": [
            {"id": f"sec:{s}", "label": s, "badge": str(len(names)) if names else None,
             "children": [{"id": f"kw:{s}:{n}", "label": n} for n in names]}
            for s, names in outline_.items() if names]})
        well_names = imported_wells(p)
    nodes.append({"id": "wells", "label": "Wells", "icon": "well", "badge": str(len(well_names)),
                  "children": [{"id": f"well:{w}", "label": w} for w in well_names]})
    nodes.append({"id": "schedule", "label": "Schedule", "icon": "clock"})
    nodes.append({"id": "observed", "label": "Observed data", "icon": "table",
                  "badge": str(len(p.record)) if p.record else None})
    nodes.append({"id": "runs", "label": "Runs", "icon": "play", "expanded": True, "children": [
        {"id": f"run:{label}", "label": label,
         "badge": None if run.id.startswith("idle-") else run.state} for label, run in runs.items()]})
    return nodes


def navigate(node: str, p: Project) -> list[tuple[str, str]]:
    """Where an explorer row takes the viewer: ``(widget key, value)`` pairs."""
    gen = pl.session_state.get("generation", 0)
    model = [("top", "Model")]
    if node == "project":
        return model + [("model_tab", "Project")]
    if node in ("grid", "model") or node.startswith("fault:"):
        return model + [("model_tab", "Grid")]
    if node in ("properties", "regions"):
        return model + [("model_tab", "Properties")]
    if node.startswith("prop:"):
        return model + [("model_tab", "Properties"), (f"g{gen}:prop", node.split(":", 1)[1])]
    if node == "fluids":
        return model + [("model_tab", "Fluids & rock")]
    if node == "init":
        return model + [("model_tab", "Initialization")]
    if node.startswith(("sec:", "kw:")) or node == "keywords":
        parts = node.split(":")
        out = model + [("model_tab", "Keywords")]
        if len(parts) > 1:
            out.append((f"g{gen}:kw_section", parts[1]))
        if len(parts) > 2:
            out.append((f"g{gen}:kw_name", parts[2]))
        return out
    if node == "wells" or node.startswith("well:"):
        out = [("top", "Wells & schedule"), ("wells_tab", "Wells")]
        if node.startswith("well:"):
            out.append((f"g{gen}:well", node.split(":", 1)[1]))
        return out
    if node == "schedule":
        return [("top", "Wells & schedule"), ("wells_tab", "Schedule")]
    if node == "observed":
        return [("top", "Wells & schedule"), ("wells_tab", "Observed data")]
    if node.startswith("run:"):
        label = node.split(":", 1)[1]
        return [("top", {"Check": "Model", "Simulation": "Simulation", "History match": "History match",
                         "Forecast": "Forecast", "Inversion": "Inversion"}[label])] + (
            [("model_tab", "3-D")] if label == "Check" else [])
    return []


def imported_wells(p: Project) -> list[str]:
    def read():
        names: dict[str, None] = {}
        for keyword in p.deck.keywords("WELSPECS"):
            for row in R.parse_records(keyword.data()):
                if row and row[0]:
                    names.setdefault(str(row[0]), None)
        return list(names)

    return session.cached("wells", str(pl.session_state.get("revision", 0)), read)


# ------------------------------------------------------------------------- the tab
def render(p: Project, check_run, gen: int) -> None:
    if p.design is not None:
        tabs = pl.tabs(["Project", "Grid", "Properties", "Fluids & rock", "Initialization", "3-D"], key="model_tab")
        with tabs[0]:
            project_panel(p, gen)
        problems = B.check(p.design)
        with tabs[1]:
            _problems(problems)
            grid_panel(p.design, gen)
        with tabs[2]:
            _problems(problems)
            properties_panel(p.design, gen)
        with tabs[3]:
            fluids_panel(p.design, gen)
        with tabs[4]:
            init_panel(p.design, gen)
        with tabs[5]:
            view3d(check_run)
    else:
        tabs = pl.tabs(["Project", "Keywords", "Tables", "3-D"], key="model_tab")
        with tabs[0]:
            project_panel(p, gen)
        with tabs[1]:
            keywords_panel(p, gen)
        with tabs[2]:
            tables_panel(p, gen)
        with tabs[3]:
            view3d(check_run)


def _problems(problems: list[str]) -> None:
    for text in problems:
        pl.markdown(f"**The design cannot become a deck yet:** {text}")


# ------------------------------------------------------------------------- project
def project_panel(p: Project, gen: int) -> None:
    left, right = pl.columns([3, 2], gap="large")
    with left:
        pl.header("Project")
        if p.design is not None:
            d = p.design
            new_name = pl.text_input("Name (the deck's file name)", d["name"], key=f"g{gen}:name")
            if new_name.strip() and new_name != d["name"]:
                session.edit_design(lambda dd: dd.update(name=new_name.strip()))
                p.name = new_name.strip()
            title = pl.text_input("Title", d.get("title", ""), key=f"g{gen}:title")
            if title != d.get("title", ""):
                session.edit_design(lambda dd: dd.update(title=title))
            c1, c2 = pl.columns(2)
            with c1:
                units = pl.radio("Units", ["FIELD", "METRIC"], ["FIELD", "METRIC"].index(d["units"]),
                                 key=f"g{gen}:units", horizontal=True,
                                 help="The numbers are kept as entered: changing the system relabels them")
                if units != d["units"]:
                    session.edit_design(lambda dd: dd.update(units=units))
            with c2:
                start = pl.text_input("Start date (YYYY-MM-DD)", d["start"], key=f"g{gen}:start")
                if start != d["start"]:
                    session.edit_design(lambda dd: dd.update(start=start))
            gas = pl.checkbox("Free gas (GAS)", "GAS" in d["phases"], key=f"g{gen}:gas")
            dis = pl.checkbox("Dissolved gas (DISGAS)", "DISGAS" in d["phases"], key=f"g{gen}:disgas")
            phases = ["OIL", "WATER"] + (["GAS"] if gas else []) + (["DISGAS"] if gas and dis else [])
            if phases != [ph for ph in ("OIL", "WATER", "GAS", "DISGAS") if ph in d["phases"]]:
                session.edit_design(lambda dd: dd.update(phases=phases))
            pl.caption("A designed model: every section has an editor, and the deck is generated "
                       "from the design whenever a job runs or the deck is exported.")
        else:
            deck = p.deck
            pl.markdown(f"**{deck.root.path}** — an imported deck tree of {len(deck.files)} file(s).")
            rows = [{"file": path, "keywords": len(f.keywords), "bytes": len(f.render()),
                     "changed": "yes" if f.changed() else ""} for path, f in deck.files.items()]
            pl.dataframe(rows, height=260)
            if deck.missing:
                pl.markdown("**Files the deck includes but the project does not hold:** "
                            + ", ".join(f"`{m}`" for m in deck.missing))
            if deck.other:
                pl.caption(f"{len(deck.other)} other file(s) travel with the deck unchanged.")
            boundary = p.history_end_days()
            pl.caption("The history | prediction boundary is set on the Schedule tab"
                       + (f" (now day {boundary:g})." if boundary is not None else "."))
    with right:
        pl.subheader("Open")
        upload = pl.file_uploader("A .gnn project, a .DATA deck, or a zip of a deck tree",
                                  type=["gnn", "zip", "DATA", "data"], key="open_file")
        if upload is not None:
            stamp = (upload.name, upload.size)
            if pl.session_state.get("opened") != stamp:
                try:
                    new = Project.load(upload.read(), upload.name)
                except Exception as exc:  # noqa: BLE001 - say what was wrong with the file
                    pl.markdown(f"**{upload.name} could not be opened:** {exc}")
                else:
                    pl.session_state["opened"] = stamp
                    session.replace(new)
                    pl.session_state["generation"] = gen + 1
                    session.notice(f"Opened {upload.name}")
                    pl.set_value("model_tab", "Project")
                    pl.rerun()
        if p.design is not None:
            pl.subheader("Keyword level")
            pl.caption("Detaching turns the design into its deck: every keyword can then be edited, "
                       "and the design's editors no longer apply.")
            if pl.action("Detach from the design", key="detach", icon="edit"):
                p.detach()
                session.touch()
                pl.session_state["generation"] = gen + 1
                session.notice("The model is now an imported deck; its design is kept for reference.")
                pl.rerun()


# ------------------------------------------------------------------------- grid
def _resize(values: list, n: int) -> list:
    values = list(values) if isinstance(values, list) else [values]
    return (values + [values[-1]] * n)[:n] if values else [1.0] * n


def grid_panel(d: dict, gen: int) -> None:
    g = d["grid"]
    left, right = pl.columns([2, 3], gap="large")
    with left:
        pl.subheader("Dimensions")
        c1, c2, c3 = pl.columns(3, gap="small")
        with c1:
            nx = pl.number_input("Columns (NX)", 1, 400, int(g["nx"]), key=f"g{gen}:nx")
        with c2:
            ny = pl.number_input("Rows (NY)", 1, 400, int(g["ny"]), key=f"g{gen}:ny")
        with c3:
            nz = pl.number_input("Layers (NZ)", 1, 200, int(g["nz"]), key=f"g{gen}:nz")
        if (nx, ny, nz) != (g["nx"], g["ny"], g["nz"]):
            def resize(dd):
                gg = dd["grid"]
                gg["dx"], gg["dy"], gg["dz"] = _resize(gg["dx"], nx), _resize(gg["dy"], ny), _resize(gg["dz"], nz)
                gg.update(nx=nx, ny=ny, nz=nz)
                for name in B.PROPERTIES:
                    for op in dd["properties"][name]:
                        if op.get("op") == "layers":
                            op["values"] = _resize(op["values"], nz)
            session.edit_design(resize)
            g = session.design()["grid"]
        units = "ft" if d["units"] == "FIELD" else "m"
        uniform_x = len(set(map(float, g["dx"]))) == 1
        uniform_y = len(set(map(float, g["dy"]))) == 1
        c1, c2 = pl.columns(2, gap="small")
        with c1:
            if uniform_x:
                dx = pl.number_input(f"Column width [{units}]", 0.01, None, float(g["dx"][0]), key=f"g{gen}:dx")
                if dx != float(g["dx"][0]):
                    session.edit_design(lambda dd: dd["grid"].update(dx=[dx] * int(dd["grid"]["nx"])))
            else:
                pl.caption(f"Column widths vary ({min(g['dx']):g}–{max(g['dx']):g} {units})")
        with c2:
            if uniform_y:
                dy = pl.number_input(f"Row width [{units}]", 0.01, None, float(g["dy"][0]), key=f"g{gen}:dy")
                if dy != float(g["dy"][0]):
                    session.edit_design(lambda dd: dd["grid"].update(dy=[dy] * int(dd["grid"]["ny"])))
            else:
                pl.caption(f"Row widths vary ({min(g['dy']):g}–{max(g['dy']):g} {units})")
        with pl.expander("Column and row widths, one by one"):
            rows = [{"column": i + 1, "width": float(v)} for i, v in enumerate(g["dx"])]
            edited = pl.data_editor(rows, key=f"g{gen}:dxs", columns={"column": {"editable": False},
                                    "width": {"min": 0.01, "unit": units}}, height=200, label="Columns (DX)")
            if [r["width"] for r in edited] != [r["width"] for r in rows]:
                session.edit_design(lambda dd: dd["grid"].update(dx=[float(r["width"] or 1.0) for r in edited]))
            rows = [{"row": j + 1, "width": float(v)} for j, v in enumerate(g["dy"])]
            edited = pl.data_editor(rows, key=f"g{gen}:dys", columns={"row": {"editable": False},
                                    "width": {"min": 0.01, "unit": units}}, height=200, label="Rows (DY)")
            if [r["width"] for r in edited] != [r["width"] for r in rows]:
                session.edit_design(lambda dd: dd["grid"].update(dy=[float(r["width"] or 1.0) for r in edited]))
        pl.subheader("Layers")
        rows = [{"layer": k + 1, "thickness": float(v)} for k, v in enumerate(g["dz"])]
        edited = pl.data_editor(rows, key=f"g{gen}:dz", columns={"layer": {"editable": False},
                                "thickness": {"min": 0.01, "unit": units}}, height=220)
        if [r["thickness"] for r in edited] != [r["thickness"] for r in rows]:
            session.edit_design(lambda dd: dd["grid"].update(dz=[float(r["thickness"] or 1.0) for r in edited]))

        pl.subheader("Structural top")
        top = g["top"]
        kind = choice("Surface", TOP_KINDS, top.get("kind", "flat"), "top_kind", gen)
        if kind != top.get("kind"):
            session.edit_design(lambda dd: dd["grid"]["top"].update(kind=kind))
        c1, c2 = pl.columns(2, gap="small")
        with c1:
            bound(f"Depth of the top [{units}]" + (" at the crest" if kind in ("dome", "anticline") else ""),
                  ("grid", "top", "depth"), "top_depth", gen)
            if kind in ("dome", "anticline"):
                bound(f"Relief [{units}]", ("grid", "top", "relief"), "top_relief", gen, min_value=0.0)
                bound("Width (fraction of the model)", ("grid", "top", "radius"), "top_radius", gen,
                      min_value=0.01, max_value=5.0, step=0.05)
        with c2:
            if kind in ("dome", "anticline"):
                bound("Crest at X (fraction)", ("grid", "top", "x0"), "top_x0", gen, min_value=0.0, max_value=1.0, step=0.05)
                bound("Crest at Y (fraction)", ("grid", "top", "y0"), "top_y0", gen, min_value=0.0, max_value=1.0, step=0.05)
            if kind == "anticline":
                bound("Axis azimuth [°]", ("grid", "top", "azimuth"), "top_az", gen, min_value=0.0, max_value=180.0)
            if kind != "flat":
                bound("Dip along X [°]", ("grid", "top", "dip_x"), "top_dipx", gen, min_value=-45.0, max_value=45.0, step=0.1)
                bound("Dip along Y [°]", ("grid", "top", "dip_y"), "top_dipy", gen, min_value=-45.0, max_value=45.0, step=0.1)

        pl.subheader("Faults")
        rows = [{"name": f.get("name", f"F{i + 1}"), "throw": float(f.get("throw", 0.0)),
                 "downthrown": f.get("downthrown", "right"), "mult": float(f.get("mult", 1.0)),
                 "nodes": len(f.get("nodes", []))} for i, f in enumerate(g.get("faults", []))]
        edited = pl.data_editor(rows, key=f"g{gen}:faults", num_rows="dynamic", height=180, label="Faults", columns={
            "name": {"type": "text"}, "throw": {"type": "number", "unit": units, "default": 0.0},
            "downthrown": {"type": "select", "options": ["left", "right"], "default": "right"},
            "mult": {"type": "number", "min": 0.0, "default": 1.0, "label": "transmissibility ×",
                     "help": "MULTFLT: the factor on the transmissibility across the fault"},
            "nodes": {"type": "integer", "editable": False, "label": "trace nodes"}},
            help="Draw a fault's trace on the map: pick it under Map tool")
        if [{k: v for k, v in r.items() if k != "nodes"} for r in edited] != \
                [{k: v for k, v in r.items() if k != "nodes"} for r in rows]:
            def faults(dd):
                old = dd["grid"].get("faults", [])
                new = []
                for i, r in enumerate(edited):
                    base = old[i] if i < len(old) else {"nodes": []}
                    new.append({**base, "name": (r["name"] or f"F{i + 1}").strip(), "throw": float(r["throw"] or 0.0),
                                "downthrown": r["downthrown"] or "right",
                                "mult": float(r["mult"] if r["mult"] is not None else 1.0)})
                dd["grid"]["faults"] = new
            session.edit_design(faults)

        pl.subheader("Inactive cells")
        rows = [{"i1": it.get("i", [1, 1])[0], "i2": it.get("i", [1, 1])[1], "j1": it.get("j", [1, 1])[0],
                 "j2": it.get("j", [1, 1])[1], "k1": it.get("k", [1, 1])[0], "k2": it.get("k", [1, 1])[1]}
                for it in g.get("inactive", []) if "cells" not in it]
        painted = [it for it in g.get("inactive", []) if "cells" in it]
        edited = pl.data_editor(rows, key=f"g{gen}:inactive", num_rows="dynamic", height=150,
                                columns={c: {"type": "integer", "min": 1, "default": 1} for c in
                                         ("i1", "i2", "j1", "j2", "k1", "k2")},
                                help="Boxes of cells taken out of the model (ACTNUM 0)")
        if edited != rows:
            boxes = [{"i": [r["i1"] or 1, r["i2"] or 1], "j": [r["j1"] or 1, r["j2"] or 1],
                      "k": [r["k1"] or 1, r["k2"] or 1]} for r in edited]
            session.edit_design(lambda dd: dd["grid"].update(inactive=boxes + painted))
        if painted:
            pl.caption(f"{sum(len(it['cells']) for it in painted)} painted column(s) are inactive too.")
            if pl.action("Clear the painted inactive cells", key="clear_paint"):
                session.edit_design(lambda dd: dd["grid"].update(
                    inactive=[it for it in dd["grid"]["inactive"] if "cells" not in it]))

    with right:
        grid_map(session.design(), gen)


def _fault_lines(d: dict) -> list[dict]:
    lines = []
    for f in d["grid"].get("faults", []):
        nodes = f.get("nodes", [])
        if len(nodes) >= 2:
            lines.append({"points": [list(n) for n in B.staircase([tuple(n) for n in nodes])],
                          "label": f.get("name", "")})
    return lines


def _well_markers(d: dict) -> list[dict]:
    return [{"i": w["trajectory"][0][0], "j": w["trajectory"][0][1], "label": w["name"]}
            for w in d["wells"] if w.get("trajectory")]


def grid_map(d: dict, gen: int) -> None:
    g = d["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    faults = g.get("faults", [])
    tools = ["Look"] + [f"Draw fault {f.get('name', i + 1)}" for i, f in enumerate(faults)] + ["Paint inactive cells"]
    tool = pl.radio("Map tool", tools, 0, key=f"g{gen}:grid_tool", horizontal=True)
    layer = pl.slider("Layer", 1, nz, 1, key=f"g{gen}:grid_layer") if nz > 1 else 1
    try:
        geom = session.geometry(d)
    except B.DesignError as exc:
        pl.markdown(f"**The grid cannot be built:** {exc}")
        return
    depth = geom["corners"][layer - 1, ..., 2].mean(axis=-1)          # (ny, nx)
    act = geom["actnum"][layer - 1]
    values = np.where(act > 0, depth, np.nan)
    mode, selection, key = "none", None, f"g{gen}:gridmap"
    if tool.startswith("Draw fault"):
        index = tools.index(tool) - 1
        mode, selection, key = "polyline", faults[index].get("nodes") or None, f"g{gen}:faultmap{index}"
    elif tool.startswith("Paint"):
        mode, key = "cells", f"g{gen}:paintmap"
    units = "ft" if d["units"] == "FIELD" else "m"
    chosen = pl.map2d(nx, ny, values=values, key=key, mode=mode, selection=selection, cmap="turbo",
                      legend=f"Cell centre depth, layer {layer} [{units}]", height=470,
                      markers=_well_markers(d), lines=_fault_lines(d))
    if tool.startswith("Draw fault") and chosen and chosen != faults[tools.index(tool) - 1].get("nodes"):
        index = tools.index(tool) - 1
        session.edit_design(lambda dd: dd["grid"]["faults"][index].update(nodes=[list(n) for n in chosen]))
    if tool.startswith("Paint"):
        c1, c2 = pl.columns(2, gap="small")
        with c1:
            k1 = pl.number_input("From layer", 1, nz, 1, key=f"g{gen}:paint_k1")
        with c2:
            k2 = pl.number_input("To layer", 1, nz, nz, key=f"g{gen}:paint_k2")
        if chosen and pl.action(f"Make the {len(chosen)} painted column(s) inactive", key="paint_apply", type="primary"):
            cells = [list(c) for c in chosen]
            session.edit_design(lambda dd: dd["grid"]["inactive"].append({"cells": cells, "k": [min(k1, k2), max(k1, k2)]}))
            pl.set_value(f"g{gen}:paintmap", [])
    pl.caption("Faults follow the cell edges nearest their drawn trace; the downthrown side takes the throw "
               "at all of its cells' corners. Wells are marked at their heads.")


# ------------------------------------------------------------------------- properties
def _describe(op: dict) -> str:
    kind = op.get("op")
    if kind == "constant":
        return f"= {op['value']:g}"
    if kind == "layers":
        return "per layer: " + ", ".join(f"{v:g}" for v in op["values"][:8]) + (" …" if len(op["values"]) > 8 else "")
    if kind == "trend":
        return f"{op.get('at', 0):g} + {op.get('gradient', 0):g} × {op.get('axis', 'depth')}"
    if kind == "random":
        base = f"mean {op['mean']:g}, " if "mean" in op else "+ "
        return f"{base}sd {op.get('sd', 0):g}, ranges {op.get('range', [3, 3, 1])} cells, seed {op.get('seed', 0)}"
    if kind == "transform":
        return f"10^({op.get('a', 0):g} + {op.get('b', 1):g} × {op['source']})"
    if kind == "copy":
        return f"{op.get('factor', 1):g} × {op['source']}"
    if kind == "box":
        return f"{op.get('mode', 'set')} {op['value']:g} in I {op.get('i')} J {op.get('j')} K {op.get('k')}"
    if kind == "cells":
        return f"{op.get('mode', 'set')} {op['value']:g} in {len(op.get('cells', []))} painted column(s), K {op.get('k')}"
    if kind == "region":
        return f"{op.get('mode', 'set')} {op['value']:g} where {op['array']} = {op['id']}"
    return kind or "?"


def properties_panel(d: dict, gen: int) -> None:
    names = [*B.PROPERTIES, *B.REGIONS]
    left, right = pl.columns([2, 3], gap="large")
    with left:
        name = pl.selectbox("Property", names, 0, key=f"g{gen}:prop")
        regions = name in B.REGIONS
        recipe = (d["regions"] if regions else d["properties"])[name]
        section = "regions" if regions else "properties"
        pl.table([{"step": i + 1, "operation": OPS.get(op.get("op"), op.get("op")), "detail": _describe(op)}
                  for i, op in enumerate(recipe)] or [{"step": "", "operation": "(empty)", "detail": ""}])
        add = pl.menu("Add a step", [OPS[k] for k in OPS], key=f"g{gen}:addop", icon="add")
        if add:
            kind = next(k for k, v in OPS.items() if v == add)
            session.edit_design(lambda dd: dd[section][name].append(_new_op(kind, dd, name)))
            recipe = session.design()[section][name]
            pl.set_value(f"g{gen}:step:{name}", f"{len(recipe)}")
        if not recipe:
            return
        labels = [str(i + 1) for i in range(len(recipe))]
        step = pl.radio("Edit step", labels, len(labels) - 1, key=f"g{gen}:step:{name}", horizontal=True)
        index = labels.index(step) if step in labels else len(labels) - 1
        c1, c2, c3 = pl.columns(3, gap="small")
        with c1:
            if pl.action("Earlier", key=f"up:{name}", icon="undo", disabled=index == 0):
                session.edit_design(lambda dd: _swap(dd[section][name], index, index - 1))
                pl.set_value(f"g{gen}:step:{name}", str(index))
        with c2:
            if pl.action("Later", key=f"down:{name}", disabled=index == len(recipe) - 1):
                session.edit_design(lambda dd: _swap(dd[section][name], index, index + 1))
                pl.set_value(f"g{gen}:step:{name}", str(index + 2))
        with c3:
            if pl.action("Remove", key=f"rm:{name}", icon="remove"):
                session.edit_design(lambda dd: dd[section][name].pop(index))
                pl.set_value(f"g{gen}:step:{name}", str(max(1, index)))
                return
        op = session.design()[section][name][index]
        op_editor(d, section, name, index, op, gen)
    with right:
        property_map(session.design(), name, gen, section, index if recipe else None)


def _swap(items: list, a: int, b: int) -> None:
    items[a], items[b] = items[b], items[a]


def _new_op(kind: str, d: dict, name: str) -> dict:
    nz = int(d["grid"]["nz"])
    integer = name in B.REGIONS
    value = 2 if integer else (0.2 if name in ("PORO", "NTG") else 100.0)
    return {
        "constant": {"op": "constant", "value": value},
        "layers": {"op": "layers", "values": [value] * nz},
        "trend": {"op": "trend", "axis": "depth", "at": value, "gradient": 0.0},
        "random": {"op": "random", "sd": 0.02 if name == "PORO" else 0.1 * value, "range": [4, 4, 1], "seed": 1},
        "transform": {"op": "transform", "source": "PORO", "a": -1.0, "b": 14.0},
        "copy": {"op": "copy", "source": "PERMX", "factor": 1.0},
        "box": {"op": "box", "value": value, "mode": "set", "i": [1, 1], "j": [1, 1], "k": [1, nz]},
        "cells": {"op": "cells", "value": value, "mode": "set", "cells": [], "k": [1, nz]},
        "region": {"op": "region", "array": "FIPNUM", "id": 1, "value": value, "mode": "set"},
    }[kind]


def op_editor(d: dict, section: str, name: str, index: int, op: dict, gen: int) -> None:
    kind = op.get("op")
    base = f"g{gen}:op:{name}:{index}"

    def set_(**fields):
        session.edit_design(lambda dd: dd[section][name][index].update(**fields))

    def num(label, field, default=0.0, **kw):
        current = op.get(field, default)
        value = pl.number_input(label, kw.get("min_value"), kw.get("max_value"), current, kw.get("step"),
                                key=f"{base}:{field}")
        if value != current:
            set_(**{field: value})
        return value

    pl.subheader(f"Step {index + 1}: {OPS.get(kind, kind)}")
    nz = int(d["grid"]["nz"])
    modes = ["set", "multiply", "add"]
    if kind in ("box", "cells", "region"):
        mode = pl.radio("How", modes, modes.index(op.get("mode", "set")), key=f"{base}:mode", horizontal=True)
        if mode != op.get("mode", "set"):
            set_(mode=mode)
    if kind in ("constant", "box", "cells", "region"):
        num("Value", "value")
    if kind == "layers":
        rows = [{"layer": k + 1, "value": float(v)} for k, v in enumerate(_resize(op["values"], nz))]
        edited = pl.data_editor(rows, key=f"{base}:values", columns={"layer": {"editable": False}}, height=220)
        if [r["value"] for r in edited] != [r["value"] for r in rows]:
            set_(values=[float(r["value"] if r["value"] is not None else 0.0) for r in edited])
    elif kind == "trend":
        axis = pl.radio("Along", ["depth", "x", "y"], ["depth", "x", "y"].index(op.get("axis", "depth")),
                        key=f"{base}:axis", horizontal=True)
        if axis != op.get("axis"):
            set_(axis=axis)
        num("Value at the shallowest (or first) cell", "at")
        num("Change per unit of length", "gradient", step=0.001)
    elif kind == "random":
        replace = pl.checkbox("Replace the value (mean ± sd), rather than add to it", "mean" in op, key=f"{base}:replace")
        if replace and "mean" not in op:
            set_(mean=0.2 if name == "PORO" else 100.0)
        if not replace and "mean" in op:
            session.edit_design(lambda dd: dd[section][name][index].pop("mean", None))
        if replace:
            num("Mean", "mean")
        num("Standard deviation", "sd", min_value=0.0)
        c1, c2, c3 = pl.columns(3, gap="small")
        rng = list(op.get("range", [3, 3, 1]))
        for col, axis, label in ((c1, 0, "Range along I"), (c2, 1, "along J"), (c3, 2, "along K")):
            with col:
                v = pl.number_input(label, 0.0, 200.0, float(rng[axis]), key=f"{base}:range{axis}")
                if v != rng[axis]:
                    rng[axis] = v
                    set_(range=rng)
        c1, c2, c3 = pl.columns(3, gap="small")
        with c1:
            num("Seed", "seed", 0, min_value=0, step=1)
        with c2:
            num("Lowest", "min", 0.0)
        with c3:
            num("Highest", "max", 1.0 if name in ("PORO", "NTG") else 1.0e5)
    elif kind in ("transform", "copy"):
        sources = [n for n in (*B.PROPERTIES, *B.REGIONS) if n != name]
        source = pl.selectbox("From", sources, sources.index(op.get("source")) if op.get("source") in sources else 0,
                              key=f"{base}:source")
        if source != op.get("source"):
            set_(source=source)
        if kind == "transform":
            num("a", "a", 0.0, step=0.1)
            num("b", "b", 1.0, step=0.1)
            pl.caption("log10 of the value = a + b × the source: the porosity–permeability transform of a core "
                       "plot, with permeability in mD.")
        else:
            num("Factor", "factor", 1.0, step=0.01)
    elif kind == "box":
        g = d["grid"]
        for axis, n, label in (("i", g["nx"], "I"), ("j", g["ny"], "J"), ("k", nz, "K")):
            lo, hi = op.get(axis, [1, n])
            c1, c2 = pl.columns(2, gap="small")
            with c1:
                a = pl.number_input(f"{label} from", 1, int(n), int(lo), key=f"{base}:{axis}1")
            with c2:
                b = pl.number_input(f"{label} to", 1, int(n), int(hi), key=f"{base}:{axis}2")
            if [a, b] != [lo, hi]:
                set_(**{axis: [min(a, b), max(a, b)]})
        pl.caption("Or drag the box on the map.")
    elif kind == "cells":
        lo, hi = op.get("k", [1, nz])
        c1, c2 = pl.columns(2, gap="small")
        with c1:
            a = pl.number_input("From layer", 1, nz, int(lo), key=f"{base}:k1")
        with c2:
            b = pl.number_input("To layer", 1, nz, int(hi), key=f"{base}:k2")
        if [a, b] != [lo, hi]:
            set_(k=[min(a, b), max(a, b)])
        pl.caption(f"{len(op.get('cells', []))} column(s) painted: drag on the map to add, from a painted cell to remove.")
    elif kind == "region":
        arrays = [n for n in B.REGIONS if n != name]
        array = pl.selectbox("Region array", arrays, arrays.index(op.get("array")) if op.get("array") in arrays else 0,
                             key=f"{base}:array")
        if array != op.get("array"):
            set_(array=array)
        num("Region number", "id", 1, min_value=1, step=1)


def property_map(d: dict, name: str, gen: int, section: str, index: int | None) -> None:
    g = d["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    layer = pl.slider("Layer", 1, nz, 1, key=f"g{gen}:prop_layer") if nz > 1 else 1
    try:
        arrs = session.arrays(d)
        geom = session.geometry(d)
    except B.DesignError as exc:
        pl.markdown(f"**{exc}**")
        return
    arr = arrs[name]
    act = geom["actnum"]
    regions = name in B.REGIONS
    values = np.where(act[layer - 1] > 0, arr[layer - 1], np.nan)
    op = d[section][name][index] if index is not None else {}
    mode, key, selection = "none", f"g{gen}:propmap", None
    if op.get("op") == "cells":
        mode, key, selection = "cells", f"g{gen}:paint:{name}:{index}", op.get("cells", [])
    elif op.get("op") == "box":
        i, j = op.get("i", [1, 1]), op.get("j", [1, 1])
        mode, key, selection = "rect", f"g{gen}:box:{name}:{index}", [i[0], j[0], i[1], j[1]]
    log = name.startswith("PERM")
    shown = np.log10(np.maximum(values, 1e-4)) if log else values
    finite = arr[act > 0]
    chosen = pl.map2d(nx, ny, values=shown, key=key, mode=mode, selection=selection, categorical=regions,
                      value_range=None if regions else (float(np.log10(max(finite.min(), 1e-4))) if log else float(finite.min()),
                                                       float(np.log10(max(finite.max(), 1e-4))) if log else float(finite.max())),
                      legend=("log10 " if log else "") + f"{name}, layer {layer}", height=430,
                      markers=_well_markers(d), lines=_fault_lines(d))
    if op.get("op") == "cells" and chosen is not None and chosen != op.get("cells", []):
        session.edit_design(lambda dd: dd[section][name][index].update(cells=[list(c) for c in chosen]))
    if op.get("op") == "box" and chosen and chosen != selection:
        session.edit_design(lambda dd: dd[section][name][index].update(i=[chosen[0], chosen[2]], j=[chosen[1], chosen[3]]))
    if finite.size:
        if regions:
            ids, counts = np.unique(finite, return_counts=True)
            pl.bar_chart([{"region": int(i), "cells": int(c)} for i, c in zip(ids, counts)], x="region", y="cells",
                         labels=["region", "cells"], height=170)
        else:
            data = np.log10(np.maximum(finite, 1e-4)) if log else finite
            counts, edges = np.histogram(data, bins=24)
            pl.bar_chart([{"value": float(0.5 * (a + b)), "cells": int(c)} for a, b, c in zip(edges[:-1], edges[1:], counts)],
                         x="value", y="cells", labels=[("log10 " if log else "") + name, "cells"], height=170)
            pl.caption(f"active cells: min {finite.min():.4g}, mean {finite.mean():.4g}, max {finite.max():.4g}")


# ------------------------------------------------------------------------- fluids and rock
def fluids_panel(d: dict, gen: int) -> None:
    units = d["units"]
    from builder.units import label

    left, right = pl.columns([3, 2], gap="large")
    with left:
        pl.subheader("Fluids (one row per fluid region, PVTNUM)")
        rows = [{"api": f["api"], "gas_gravity": f["gas_gravity"], "temperature": f["temperature"], "pb": f["pb"],
                 "co": f.get("co", 1e-5)} for f in d["fluids"]]
        edited = pl.data_editor(rows, key=f"g{gen}:fluids", num_rows="dynamic", height=160, columns={
            "api": {"label": "Oil gravity", "unit": "°API", "min": 5.0, "max": 70.0, "default": 35.0},
            "gas_gravity": {"label": "Gas gravity", "unit": "air = 1", "min": 0.55, "max": 1.8, "default": 0.75},
            "temperature": {"label": "Temperature", "unit": label("temperature", units), "default": 180.0},
            "pb": {"label": "Bubble point", "unit": label("pressure", units), "min": 15.0, "default": 3000.0},
            "co": {"label": "Oil compressibility", "unit": label("compressibility", units), "min": 0.0, "default": 1e-5}})
        if edited != rows and edited:
            session.edit_design(lambda dd: dd.update(fluids=[{"model": "live", **{k: float(v if v is not None else 0.0)
                                                                                  for k, v in r.items()}} for r in edited]))
        c1, c2 = pl.columns(2, gap="small")
        with c1:
            bound(f"Rock compressibility [{label('compressibility', units)}]", ("rock", "cr"), "rock_cr", gen, min_value=0.0)
        with c2:
            bound(f"at pressure [{label('pressure', units)}]", ("rock", "p_ref"), "rock_pref", gen, min_value=1.0)
        pl.subheader("Rock–fluid functions (one row per saturation region, SATNUM)")
        cols = ("swc", "swcr", "sowcr", "sgcr", "sogcr", "krw_max", "kro_max", "krg_max", "nw", "now", "ng", "nog",
                "pc_entry", "pc_lambda")
        rows = [{c: float(s.get(c, 0.0)) for c in cols} for s in d["scal"]]
        edited = pl.data_editor(rows, key=f"g{gen}:scal", num_rows="dynamic", height=160, columns={
            "swc": {"label": "Swc", "min": 0.0, "max": 0.9, "default": 0.2, "help": "connate water"},
            "swcr": {"label": "Swcr", "min": 0.0, "max": 0.9, "default": 0.2, "help": "critical water"},
            "sowcr": {"label": "Sowcr", "min": 0.0, "max": 0.9, "default": 0.2, "help": "residual oil to water"},
            "sgcr": {"label": "Sgcr", "min": 0.0, "max": 0.9, "default": 0.05, "help": "critical gas"},
            "sogcr": {"label": "Sogcr", "min": 0.0, "max": 0.9, "default": 0.1, "help": "residual oil to gas"},
            "krw_max": {"label": "krw max", "min": 0.0, "max": 1.0, "default": 0.4},
            "kro_max": {"label": "kro max", "min": 0.0, "max": 1.0, "default": 0.9},
            "krg_max": {"label": "krg max", "min": 0.0, "max": 1.0, "default": 0.8},
            "nw": {"label": "nw", "min": 0.5, "max": 8.0, "default": 2.0},
            "now": {"label": "now", "min": 0.5, "max": 8.0, "default": 2.0},
            "ng": {"label": "ng", "min": 0.5, "max": 8.0, "default": 2.0},
            "nog": {"label": "nog", "min": 0.5, "max": 8.0, "default": 2.0},
            "pc_entry": {"label": "Pc entry", "unit": label("pressure", units), "min": 0.0, "default": 0.0},
            "pc_lambda": {"label": "λ", "min": 0.2, "max": 10.0, "default": 2.0}})
        if edited != rows and edited:
            session.edit_design(lambda dd: dd.update(scal=[{c: float(r[c] if r[c] is not None else 0.0) for c in cols}
                                                           for r in edited]))
    with right:
        d = session.design()
        region = pl.number_input("Show region", 1, max(len(d["scal"]), len(d["fluids"]), 1), 1, key=f"g{gen}:fl_region")
        try:
            swof, sgof = B.scal_tables(d["scal"][min(region, len(d["scal"])) - 1])
            pl.line_chart({"Sw": swof[:, 0], "krw": swof[:, 1], "krow": swof[:, 2]}, x="Sw", y=["krw", "krow"],
                          legend=["krw", "krow"], labels=["Sw", "relative permeability"], height=200)
            if "GAS" in d["phases"]:
                pl.line_chart({"Sg": sgof[:, 0], "krg": sgof[:, 1], "krog": sgof[:, 2]}, x="Sg", y=["krg", "krog"],
                              legend=["krg", "krog"], labels=["Sg", "relative permeability"], height=200)
            pvt = session.cached("pvt", session.design_key({k: d[k] for k in ("fluids", "equil", "phases", "units")}),
                                 lambda: B.pvt_tables(d))
            if pvt["PVTO"]:
                blocks = pvt["PVTO"][min(region, len(pvt["PVTO"])) - 1]
                sat = np.asarray([b[0] for b in blocks])
                pl.line_chart({"p": sat[:, 1], "Bo": sat[:, 2], "Rs": sat[:, 0]}, x="p", y=["Bo", "Rs"],
                              legend=[f"Bo [{'rb/stb' if units == 'FIELD' else 'rm³/sm³'}]",
                                      f"Rs [{label('gor', units)}]"],
                              labels=[f"saturation pressure [{label('pressure', units)}]", ""], height=200)
                pl.line_chart({"p": sat[:, 1], "mu": sat[:, 3]}, x="p", y="mu",
                              labels=[f"pressure [{label('pressure', units)}]", "oil viscosity [cP]"], height=170)
        except B.DesignError as exc:
            pl.markdown(f"**{exc}**")


# ------------------------------------------------------------------------- initialization
def init_panel(d: dict, gen: int) -> None:
    from builder.units import label

    units = d["units"]
    pl.subheader("Initial equilibrium (one row per equilibration region, EQLNUM)")
    rows = [{c: float(e.get(c, 0.0)) for c in ("datum", "pressure", "woc", "goc", "pc_woc", "pc_goc")} for e in d["equil"]]
    edited = pl.data_editor(rows, key=f"g{gen}:equil", num_rows="dynamic", height=170, columns={
        "datum": {"label": "Datum depth", "unit": label("length", units)},
        "pressure": {"label": "Pressure at datum", "unit": label("pressure", units), "min": 1.0},
        "woc": {"label": "Water–oil contact", "unit": label("length", units)},
        "goc": {"label": "Gas–oil contact", "unit": label("length", units)},
        "pc_woc": {"label": "Pc at WOC", "unit": label("pressure", units), "default": 0.0},
        "pc_goc": {"label": "Pc at GOC", "unit": label("pressure", units), "default": 0.0}})
    if edited != rows and edited:
        session.edit_design(lambda dd: dd.update(equil=[{k: float(v if v is not None else 0.0) for k, v in r.items()}
                                                        for r in edited]))
    d = session.design()
    try:
        geom = session.geometry(d)
    except B.DesignError as exc:
        pl.markdown(f"**{exc}**")
        return
    z = geom["corners"][..., 2]
    act = geom["actnum"] > 0
    top, bottom = float(z.min(axis=-1)[act].min()), float(z.max(axis=-1)[act].max())
    u = label("length", units)
    for i, e in enumerate(d["equil"]):
        notes = [f"Region {i + 1}: the reservoir spans {top:.6g}–{bottom:.6g} {u}."]
        if "GAS" in d["phases"]:
            notes.append(f"gas cap from the top to {e['goc']:.6g} {u}" if e["goc"] > top else
                         f"no gas cap (the gas–oil contact at {e['goc']:.6g} {u} lies above the top)")
        notes.append(f"aquifer below {e['woc']:.6g} {u}" if e["woc"] < bottom else
                     f"no water leg (the water–oil contact at {e['woc']:.6g} {u} lies below the bottom)")
        pl.markdown(" ".join(notes[:1]) + " With " + "; ".join(notes[1:]) + ".")
    if "DISGAS" in d["phases"]:
        rs = [B.initial_rs(d, i) for i in range(len(d["fluids"]))]
        pl.caption("Initial dissolved gas, from each fluid's bubble point: " + ", ".join(
            f"{r:.4g} {label('gor', units)}" for r in rs) + " (RSVD, constant with depth).")


# ------------------------------------------------------------------------- 3-D
def view3d(check_run) -> None:
    pl.caption("The model as OPM Flow builds it — press Check in the toolbar after an edit. Pore-volume cuts and "
               "inactive cells are applied; wells are drawn through their perforated cells.")
    left, right = pl.columns([3, 1], gap="medium")
    with left:
        pl.grid3d(check_run, field="PORO", field_labels=field_labels(), z_scale=5, height=600, key="model3d",
                  cmap="rainbow")
    with right:
        pl.metric("Active cells", check_run, field="7.active_cells", where=(0, "metrics"))
        pl.metric("Errors", check_run, field="7.errors", where=(0, "metrics"))
        pl.metric("Warnings", check_run, field="7.warnings", where=(0, "metrics"))
        pl.metric("Wells", check_run, field="7.wells", where=(0, "metrics"))
    pl.subheader("What OPM Flow said")
    pl.feed(check_run, fields=["7.level", "7.where", "7.message"], labels=["", "Where", "Message"],
            where=(0, "diag"), height=240, key="model_diags", empty="Nothing yet: press Check.")


# ------------------------------------------------------------------------- imported decks
def keywords_panel(p: Project, gen: int) -> None:
    deck: Deck = p.deck
    outline_ = deck.outline()
    sections = [s for s in SECTIONS if outline_[s]] or list(SECTIONS)
    left, right = pl.columns([1, 3], gap="large")
    with left:
        section = pl.selectbox("Section", sections, 0, key=f"g{gen}:kw_section")
        found = [k for _, k in deck.walk() if k.section == section and k.name not in SECTIONS]
        labels = []
        for i, k in enumerate(found):
            labels.append(k.name if k.name not in labels else f"{k.name} ({sum(1 for x in found[:i] if x.name == k.name) + 1})")
        if not found:
            pl.caption("This section holds no keyword.")
            keyword = None
        else:
            chosen = pl.selectbox("Keyword", labels, 0, key=f"g{gen}:kw_name")
            keyword = found[labels.index(chosen)] if chosen in labels else found[0]
        pl.subheader("Add a keyword")
        new_name = pl.text_input("Name", "", key=f"g{gen}:kw_new", placeholder="MULTIPLY")
        if new_name.strip() and pl.action(f"Add {new_name.strip().upper()} to {section}", key="kw_add", icon="add"):
            deck.ensure_section(section)
            deck.append(section, new_name.strip().upper(), "/")
            session.touch()
            pl.set_value(f"g{gen}:kw_name", new_name.strip().upper())
    with right:
        if keyword is None:
            return
        where = deck.file_of(keyword).path
        data = keyword.body
        pl.markdown(f"**{keyword.name}** in `{where}`" + (" — edited" if keyword.dirty else ""))
        if keyword.include is not None:
            pl.caption(f"Includes `{keyword.include.path}` ({len(keyword.include.keywords)} keyword(s)); its keywords "
                       "are listed in their sections.")
        if len(data) > 200_000:
            values = R.parse_array(keyword.data()) if keyword.name in ("PORO", "PERMX", "PERMY", "PERMZ", "NTG", "ZCORN",
                                                                        "COORD", "ACTNUM", "MULTZ") else None
            pl.markdown(f"{len(data):,} characters — too long to edit here."
                        + (f" {values.size:,} values from {np.nanmin(values):.4g} to {np.nanmax(values):.4g}."
                           if values is not None and values.size else ""))
            return
        text = pl.text_area("Data (comments kept)", data, key=f"g{gen}:kw_body:{id(keyword)}", height=360, monospace=True)
        c1, c2 = pl.columns(2, gap="small")
        with c1:
            if text != data and pl.action("Apply the edit", key="kw_apply", type="primary", icon="check"):
                keyword.set_body(text)
                session.touch()
        with c2:
            if pl.action("Remove the keyword", key="kw_remove", icon="remove"):
                deck.remove(keyword)
                session.touch()


TABLES = {"SWOF": (4, ["Sw", "krw", "krow", "Pcow"]), "SGOF": (4, ["Sg", "krg", "krog", "Pcog"]),
          "SWFN": (3, ["Sw", "krw", "Pcow"]), "SGFN": (3, ["Sg", "krg", "Pcog"]), "SOF3": (3, ["So", "krow", "krog"]),
          "PVDG": (3, ["p", "Bg", "mu_g"]), "PVDO": (3, ["p", "Bo", "mu_o"]), "PVTW": (5, ["p_ref", "Bw", "cw", "mu_w", "visc"]),
          "DENSITY": (3, ["oil", "water", "gas"]), "ROCK": (2, ["p_ref", "cr"])}


def tables_panel(p: Project, gen: int) -> None:
    deck: Deck = p.deck
    present = [n for n in ("EQUIL", *TABLES, "PVTO") if deck.has(n)]
    if not present:
        pl.caption("The deck has none of the table keywords this editor knows.")
        return
    name = pl.radio("Keyword", present, 0, key=f"g{gen}:tbl", horizontal=True)
    keyword = deck.first(name)
    if name == "EQUIL":
        rows = [r for r in R.parse_records(keyword.data()) if r]
        cols = ["datum", "pressure", "woc", "pc_woc", "goc", "pc_goc"]
        table = [{c: (float(r[i]) if i < len(r) and r[i] is not None else None) for i, c in enumerate(cols)} for r in rows]
        edited = pl.data_editor(table, key=f"g{gen}:tbl_equil", height=200,
                                columns={c: {"type": "number"} for c in cols},
                                help="Datum depth, pressure, water–oil contact, its capillary pressure, gas–oil contact, "
                                     "its capillary pressure; the record's later items are kept as they are")
        if edited != table:
            new_rows = []
            for r, e in zip(rows, edited):
                out = list(r) + [None] * max(0, 6 - len(r))
                for i, c in enumerate(cols):
                    out[i] = e[c]
                new_rows.append(out)
            keyword.set_body(R.format_records(new_rows, terminate=False))
            session.touch()
        return
    if name == "PVTO":
        tables = R.parse_pvto(keyword.data())
        index = pl.number_input("Table", 1, len(tables), 1, key=f"g{gen}:tbl_pvto") if len(tables) > 1 else 1
        sat = np.asarray([b[0] for b in tables[index - 1]])
        pl.line_chart({"p": sat[:, 1], "Bo": sat[:, 2], "Rs": sat[:, 0]}, x="p", y=["Bo", "Rs"], legend=["Bo", "Rs"],
                      labels=["saturation pressure", ""], height=240)
        pl.caption("PVTO is shown; edit it as text on the Keywords tab.")
        return
    width, cols = TABLES[name]
    tables = R.parse_tables(keyword.data(), width)
    index = pl.number_input("Table (region)", 1, len(tables), 1, key=f"g{gen}:tbl_index:{name}") if len(tables) > 1 else 1
    table = tables[index - 1]
    rows = [{c: (None if not np.isfinite(v) else float(v)) for c, v in zip(cols, row)} for row in table]
    left, right = pl.columns([1, 1], gap="large")
    with left:
        edited = pl.data_editor(rows, key=f"g{gen}:tbl_{name}_{index}", num_rows="dynamic", height=360,
                                columns={c: {"type": "number"} for c in cols})
        if edited != rows and edited:
            tables[index - 1] = np.asarray([[np.nan if r[c] is None else r[c] for c in cols] for r in edited], float)
            keyword.set_body(R.format_tables(tables))
            session.touch()
    with right:
        if len(table) > 1:
            pl.line_chart({c: table[:, i] for i, c in enumerate(cols)}, x=cols[0], y=cols[1:min(3, width)],
                          legend=cols[1:min(3, width)], labels=[cols[0], ""], height=300)

