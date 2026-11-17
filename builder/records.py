r"""
Keyword data as rows, and rows as keyword data.

The data of a keyword is a sequence of records, each ended by a slash. Within a record the
items are separated by blanks; a quoted item may hold blanks; ``n*v`` stands for ``n``
copies of ``v`` and ``n*`` for ``n`` defaulted items; everything after the slash on its
line is a comment. Three shapes cover every keyword the builder edits:

- **records** — one row per record (``WELSPECS``, ``COMPDAT``, ``EQUIL``, ``DATES``);
- **tables** — rows of a fixed width running on until a slash ends a table, one table per
  region (``SWOF``, ``PVDG``, ``DENSITY``);
- **arrays** — one value per cell for the whole grid, one record (``PORO``, ``ZCORN``).

``PVTO``, whose rows carry an undersaturated branch, is a table of records: each record a
saturated row followed by its undersaturated continuation, a blank record ending a table.

The module is pure standard library and NumPy.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np

from .deck import strip_comment

Item = str | None   # an item as written (``None`` where it was defaulted)

_REPEAT = re.compile(r"^(\d+)\*(.*)$")


def tokens(data: str) -> list[str]:
    """The tokens of keyword data: items (quotes kept) and record terminators ``/``.
    Comments are removed; what follows a slash on its line is a comment too."""
    out: list[str] = []
    for raw in data.splitlines():
        line = strip_comment(raw)
        i, n = 0, len(line)
        while i < n:
            ch = line[i]
            if ch.isspace():
                i += 1
            elif ch == "/":
                out.append("/")
                break                          # the rest of the line is a comment
            elif ch in "'\"":
                end = line.find(ch, i + 1)
                end = n - 1 if end < 0 else end
                out.append(line[i:end + 1])
                i = end + 1
            else:
                j = i
                while j < n and not line[j].isspace() and line[j] != "/":
                    if line[j] in "'\"":       # n*'a b' — a repeat of a quoted value
                        close = line.find(line[j], j + 1)
                        j = n if close < 0 else close + 1
                        continue
                    j += 1
                out.append(line[i:j])
                i = j
    return out


def expand(token: str) -> list[Item]:
    """One token as the items it stands for: ``3*0.2`` → three ``0.2``, ``2*`` → two
    defaults, ``'B-2H'`` → ``B-2H``."""
    match = _REPEAT.match(token)
    if match and not token.startswith(("'", '"')):
        count, value = int(match.group(1)), match.group(2)
        item: Item = None if value == "" else unquote(value)
        return [item] * count
    return [unquote(token)]


def unquote(token: str) -> str:
    if len(token) >= 2 and token[0] in "'\"" and token[-1] == token[0]:
        return token[1:-1]
    return token


def parse_records(data: str) -> list[list[Item]]:
    """Every record of keyword data, its items expanded (a record ``/`` alone is ``[]``)."""
    records: list[list[Item]] = []
    current: list[Item] = []
    started = False
    for token in tokens(data):
        if token == "/":
            records.append(current)
            current, started = [], False
        else:
            current.extend(expand(token))
            started = True
    if started:
        records.append(current)           # an unterminated last record
    return records


def parse_array(data: str, dtype=float) -> np.ndarray:
    """A grid array: every item of the keyword's one record, repeats expanded."""
    values: list[float] = []
    for token in tokens(data):
        if token == "/":
            break
        for item in expand(token):
            values.append(np.nan if item is None else float(item))
    return np.asarray(values, dtype=float if dtype is float else dtype)


def parse_tables(data: str, width: int) -> list[np.ndarray]:
    """The tables of a table keyword: each a ``(rows, width)`` array, ``NaN`` for a defaulted
    item (Eclipse interpolates those)."""
    tables: list[np.ndarray] = []
    for record in parse_records(data):
        values = [np.nan if v is None else float(v) for v in record]
        if len(values) % width:
            raise ValueError(f"a table of {width} columns holds {len(values)} items")
        tables.append(np.asarray(values, float).reshape(-1, width))
    return tables


def parse_pvto(data: str) -> list[list[np.ndarray]]:
    r"""The tables of ``PVTO``: per table, per dissolved-gas ratio :math:`R_s` a ``(rows, 4)``
    array :math:`(R_s, p, B_o, \mu_o)` — the saturated row first, its undersaturated rows
    after it. A blank record ends a table."""
    tables: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    for record in parse_records(data):
        if not record:
            if current:
                tables.append(current)
            current = []
            continue
        values = [float(v) for v in record if v is not None]
        rs, rest = values[0], values[1:]
        rows = np.asarray(rest, float).reshape(-1, 3)
        current.append(np.column_stack([np.full(len(rows), rs), rows]))
    if current:
        tables.append(current)
    return tables


# ------------------------------------------------------------------------- writing
def fmt(value) -> str:
    """One item as the deck writes it: a number in the shortest form that reads back
    exactly, a string quoted when it needs to be, ``1*`` for a default."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "1*"
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(v) if len(repr(v)) <= len(f"{v:.10g}") else f"{v:.10g}"
    text = str(value)
    if text and re.fullmatch(r"[A-Za-z0-9_.+\-:]+", text) and not re.fullmatch(r"\d+\*.*", text):
        return text if not re.fullmatch(r"[A-Za-z].*", text) else f"'{text}'"
    return "'" + text.replace("'", "") + "'"


def _trim_defaults(row: Sequence) -> list:
    """A record without its trailing defaults (Eclipse defaults what a record leaves out)."""
    items = list(row)
    while items and (items[-1] is None or (isinstance(items[-1], float) and np.isnan(items[-1]))):
        items.pop()
    return items


def format_records(rows: Iterable[Sequence], *, terminate: bool = True, align: bool = True) -> str:
    """Records as keyword data: one row per line, each ended by a slash, columns aligned,
    a final lone slash when ``terminate`` (the keywords whose record list must be ended)."""
    texts = [[fmt(v) for v in _trim_defaults(row)] for row in rows]
    widths: list[int] = []
    if align:
        for row in texts:
            for i, cell in enumerate(row):
                if i >= len(widths):
                    widths.append(0)
                widths[i] = max(widths[i], len(cell))
    lines = []
    for row in texts:
        cells = [cell.rjust(widths[i]) if align and _numeric(cell) else
                 (cell.ljust(widths[i]) if align else cell) for i, cell in enumerate(row)]
        lines.append(" " + " ".join(cells).rstrip() + " /")
    if terminate:
        lines.append("/")
    return "\n".join(lines)


def _numeric(cell: str) -> bool:
    return bool(re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", cell))


def format_tables(tables: Sequence[np.ndarray], *, digits: int = 6) -> str:
    """Tables as keyword data: one row per line, a slash after each table."""
    out = []
    for table in tables:
        arr = np.asarray(table, float)
        cells = [[fmt(float(f"{v:.{digits}g}")) if np.isfinite(v) else "1*" for v in row] for row in arr]
        widths = [max(len(r[c]) for r in cells) for c in range(arr.shape[1])] if len(cells) else []
        for k, row in enumerate(cells):
            line = " " + "  ".join(c.rjust(widths[i]) for i, c in enumerate(row))
            out.append(line + (" /" if k == len(cells) - 1 else ""))
    return "\n".join(out)


def format_pvto(tables: Sequence[Sequence[np.ndarray]], *, digits: int = 6) -> str:
    """``PVTO`` tables: per :math:`R_s` its saturated row and undersaturated rows, a slash
    after each, a lone slash after each table."""
    out = []
    for table in tables:
        for block in table:
            arr = np.asarray(block, float)
            for k, (rs, p, bo, mu) in enumerate(arr):
                lead = f" {float(f'{rs:.{digits}g}'):>10g}" if k == 0 else " " * 11
                out.append(f"{lead}  {float(f'{p:.{digits}g}'):>10g}  {float(f'{bo:.{digits}g}'):>10g}"
                           f"  {float(f'{mu:.{digits}g}'):>10g}" + (" /" if k == len(arr) - 1 else ""))
        out.append("/")
    return "\n".join(out)


def format_array(values: np.ndarray, *, per_line: int = 8, digits: int = 6, integer: bool = False) -> str:
    """A grid array as keyword data, runs of equal values written ``n*v``, ended by a slash."""
    flat = np.asarray(values).ravel()
    if integer:
        texts = [str(int(v)) for v in flat]
    else:
        texts = [fmt(float(f"{float(v):.{digits}g}")) for v in flat]
    runs: list[str] = []
    i = 0
    while i < len(texts):
        j = i + 1
        while j < len(texts) and texts[j] == texts[i]:
            j += 1
        runs.append(texts[i] if j - i == 1 else f"{j - i}*{texts[i]}")
        i = j
    lines = [" " + " ".join(runs[k:k + per_line]) for k in range(0, len(runs), per_line)]
    return "\n".join(lines) + " /" if lines else " /"
