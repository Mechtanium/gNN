r"""
The project: the model, and what the work paths need to know about it besides.

A project is one of two kinds of model —

- **designed** in the app: a :mod:`builder.design` document, from which the deck tree is
  generated every time it is needed;
- **imported**: a deck tree read at keyword level (:class:`builder.deck.Deck`), edited
  keyword by keyword and written back with every untouched file byte for byte —

and three things the deck does not say: the **history | prediction boundary** (where a
history match stops and a forecast begins), the **observed record** (the wells' measured
bottom-hole pressures and rates that a property inversion matches), and the **prior** of
the inverted properties. It is saved as a ``.gnn`` file — a zip holding the deck tree,
``project.json``, ``observed/record.csv`` and ``inversion/prior.json`` — which is also
exactly the bundle every operation of the workflow receives.

A designed project can be **detached** into an imported one, when an edit is wanted that
the design has no editor for; the design is kept in ``project.json`` for reference only.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from . import design as design_mod
from .deck import Deck, DeckError, to_zip

#: What a project file holds besides the deck tree.
PROJECT_FILE = "project.json"
RECORD_FILE = "observed/record.csv"
PRIOR_FILE = "inversion/prior.json"

#: The columns of an observed record, in the order a table shows them.
RECORD_COLUMNS = ("well", "days", "WBHP", "WOPR", "WWPR", "WGPR", "WWIR", "WGIR")

#: The properties a family can invert, and the default width of their priors (in the
#: whitened form: a standard deviation of ln(kv/kh), of the contacts in the deck's length unit).
PRIOR_DEFAULTS = {"kvkh": 0.7, "woc": 20.0, "goc": 15.0}


@dataclass
class Project:
    """One model and its project data (see the module docstring)."""

    name: str
    design: dict | None = None
    deck: Deck | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    record: list[dict[str, Any]] = field(default_factory=list)
    prior: dict[str, float] = field(default_factory=lambda: dict(PRIOR_DEFAULTS))

    # ------------------------------------------------------------------ making one
    @classmethod
    def designed(cls, design: dict) -> "Project":
        d = design_mod.normalized(design)
        return cls(name=str(d["name"]), design=d)

    @classmethod
    def imported(cls, tree: dict[str, bytes], deck_name: str) -> "Project":
        deck = Deck.from_tree(tree, deck_name)
        return cls(name=deck.name, deck=deck)

    @classmethod
    def load(cls, data: bytes, filename: str = "") -> "Project":
        """A project from an uploaded file: a ``.gnn`` project, a zip of a deck tree, or a
        single ``.DATA`` deck."""
        if data[:2] != b"PK":
            name = PurePosixPath(filename or "MODEL.DATA").name
            if not name.upper().endswith(".DATA"):
                name = f"{PurePosixPath(name).stem or 'MODEL'}.DATA"
            return cls.imported({name: data}, name)
        tree: dict[str, bytes] = {}
        extras: dict[str, bytes] = {}
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = [i.filename for i in archive.infolist() if not i.is_dir()]
            # a deck tree zipped with its folder: strip the one top-level directory
            tops = {PurePosixPath(n).parts[0] for n in names}
            strip = len(tops) == 1 and all(len(PurePosixPath(n).parts) > 1 for n in names)
            for info in archive.infolist():
                if info.is_dir():
                    continue
                path = PurePosixPath(info.filename.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts:
                    raise DeckError(f"archive member {info.filename!r} does not stay inside the archive")
                if strip:
                    path = PurePosixPath(*path.parts[1:])
                body = archive.read(info)
                key = str(path)
                if key == PROJECT_FILE or path.parts[0] in ("observed", "inversion"):
                    extras[key] = body
                else:
                    tree[key] = body
        settings = json.loads(extras[PROJECT_FILE]) if PROJECT_FILE in extras else {}
        kind = settings.pop("kind", "imported")
        if kind == "designed" and settings.get("design"):
            project = cls.designed(settings.pop("design"))
        else:
            decks = [n for n in tree if "/" not in n and n.upper().endswith(".DATA")]
            root = settings.pop("deck", None) or (decks[0] if len(decks) == 1 else None)
            if root is None:
                raise DeckError(f"the archive holds {len(decks)} top-level .DATA files; keep one, or open "
                                "the project it came from")
            project = cls.imported(tree, root)
            if settings.get("design"):
                project.design_reference = settings.pop("design")
        project.settings = {k: v for k, v in settings.items() if k not in ("design", "deck", "kind")}
        if RECORD_FILE in extras:
            project.record = read_record_csv(extras[RECORD_FILE])
        if PRIOR_FILE in extras:
            project.prior = {**PRIOR_DEFAULTS, **json.loads(extras[PRIOR_FILE])}
        return project

    # ------------------------------------------------------------------ the deck tree
    @property
    def kind(self) -> str:
        return "designed" if self.design is not None else "imported"

    def tree(self) -> tuple[dict[str, bytes], str]:
        """``(deck tree, deck name)``: generated from the design, or the imported tree."""
        if self.design is not None:
            gen = design_mod.generate(self.design)
            return gen.tree, gen.deck_name
        assert self.deck is not None
        return self.deck.tree(), self.deck.root.path

    def history_end_days(self) -> float | None:
        if self.design is not None:
            return design_mod.history_end_days(self.design)
        value = self.settings.get("history_end_days")
        return None if value is None else float(value)

    def project_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {**self.settings, "kind": self.kind, "name": self.name}
        boundary = self.history_end_days()
        if boundary is not None:
            out["history_end_days"] = boundary
        if self.design is not None:
            out["design"] = self.design
        else:
            assert self.deck is not None
            out["deck"] = self.deck.root.path
            reference = getattr(self, "design_reference", None)
            if reference:
                out["design"] = reference
        return out

    def bundle(self) -> tuple[bytes, str]:
        """``(bytes, deck name)``: the project file, which is also what a job receives."""
        tree, deck_name = self.tree()
        members = dict(tree)
        members[PROJECT_FILE] = json.dumps(self.project_json(), indent=1).encode()
        if self.record:
            members[RECORD_FILE] = write_record_csv(self.record)
        members[PRIOR_FILE] = json.dumps(self.prior, indent=1).encode()
        return to_zip(members), deck_name

    def deck_zip(self) -> bytes:
        """The deck tree alone, for a simulator outside the app."""
        tree, _ = self.tree()
        return to_zip(tree)

    def detach(self) -> None:
        """Turn a designed project into an imported one (see the module docstring)."""
        if self.design is None:
            return
        tree, deck_name = self.tree()
        boundary = self.history_end_days()
        self.design_reference = self.design
        self.design = None
        self.deck = Deck.from_tree(tree, deck_name)
        if boundary is not None:
            self.settings["history_end_days"] = boundary

    design_reference: dict | None = None


# --------------------------------------------------------------------------- records
def read_record_csv(data: bytes) -> list[dict[str, Any]]:
    """An observed record from CSV text: a header naming ``well``, ``days`` (or ``date`` with
    ``start``) and any of the rate and pressure columns of :data:`RECORD_COLUMNS`."""
    text = data.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("the record has no header line")
    fields = {f.strip().upper(): f for f in reader.fieldnames}
    if "WELL" not in fields or "DAYS" not in fields:
        raise ValueError("the record needs a 'well' and a 'days' column")
    rows = []
    for raw in reader:
        row: dict[str, Any] = {"well": raw[fields["WELL"]].strip(), "days": float(raw[fields["DAYS"]])}
        for col in RECORD_COLUMNS[2:]:
            text_value = (raw.get(fields.get(col, ""), "") or "").strip()
            row[col] = float(text_value) if text_value else None
        rows.append(row)
    return rows


def write_record_csv(rows: list[dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(RECORD_COLUMNS)
    for row in rows:
        writer.writerow([row.get(c, "") if row.get(c) is not None else "" for c in RECORD_COLUMNS])
    return buf.getvalue().encode()


def read_record_summary(files: dict[str, bytes], report_days: list[float] | None = None) -> list[dict[str, Any]]:
    """An observed record from an Eclipse summary: an ``.SMSPEC`` with its ``.UNSMRY`` (or
    an ``.ESMRY``), given as file name → bytes. One row per well and summary time — or per
    report time in ``report_days`` when given (the nearest summary time to each)."""
    import tempfile
    from pathlib import Path

    import numpy as np
    from opm.io.ecl import ESmry

    with tempfile.TemporaryDirectory() as tmp:
        stems = set()
        for name, body in files.items():
            path = Path(tmp) / PurePosixPath(name).name
            path.write_bytes(body)
            stems.add((path.stem, path.suffix.upper()))
        target = next((Path(tmp) / f"{s}{e}" for s, e in stems if e in (".SMSPEC", ".ESMRY")), None)
        if target is None:
            raise ValueError("a summary needs its .SMSPEC (with the .UNSMRY) or its .ESMRY")
        summary = ESmry(str(target))
        keys = list(summary.keys())
        times = np.asarray(summary["TIME"], float)
        wells = sorted({k.split(":", 1)[1] for k in keys if ":" in k and k.startswith("W")})
        pick = np.arange(times.size)
        if report_days:
            pick = np.unique([int(np.abs(times - t).argmin()) for t in report_days])
        rows = []
        for well in wells:
            for idx in pick:
                row: dict[str, Any] = {"well": well, "days": float(times[idx])}
                for col in RECORD_COLUMNS[2:]:
                    key = f"{col}:{well}"
                    row[col] = float(np.asarray(summary[key])[idx]) if key in keys else None
                rows.append(row)
    return rows
