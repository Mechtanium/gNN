r"""
The deck as a document: every keyword of an Eclipse deck tree, in order, losslessly.

A simulation model is read and written as text keywords. This module holds a deck as what
a modeller edits — sections of keywords — while guaranteeing that whatever it does not
change comes out exactly as it went in: a keyword nobody touched is written back byte for
byte (its comments, its spacing, its line endings), and a file none of whose keywords
changed is written back as the very bytes that were read. That is what lets a deck as
intricate as Norne — lift tables, hysteresis, tracers, fault multipliers, keywords this
package has no editor for — pass through the builder unharmed.

**Keywords.** A deck file is a preamble followed by keyword blocks. A block starts at a
*header line* — a keyword name alone on a line, starting in the first column — and runs to
the next header. Deciding what is a header needs one piece of state, because a data line
may consist of a single bare word (``PRESSURE`` among the mnemonics of ``RPTSOL``): a
line is a header only when no record is open,

.. math::

    \text{header}(\ell) \iff \text{name}(\ell) \;\wedge\; \neg\,\text{open}

where:

- :math:`\text{name}(\ell)`: the line, with its comment removed, is one keyword name in the first column.
- :math:`\text{open}`: data tokens have been read since the last record terminator ``/`` (or since the last header); a ``/`` outside quotes closes the record.

Two keywords carry data that no slash ends and are handled by name: ``TITLE`` (the next
line is the title) and the section headers (always headers, so one unterminated record
cannot swallow the rest of a deck). No list of keyword names takes part in the decision:
the simulator's own list leaves out whole families (the summary vectors, ``MULTZ`` and its
kin, the tracer keywords ``TVDPF…``), and a deck may carry keywords of another simulator;
the rule above reproduces OPM Flow's keyword sequence on every deck it was held against.

**Files.** ``INCLUDE`` names a file relative to the directory of the root deck (a ``$NAME``
alias of ``PATHS`` is expanded); its keywords are read in place, so the document's flat
order is the order the simulator reads. Each keyword knows the section it falls in —
``RUNSPEC``, ``GRID``, ``EDIT``, ``PROPS``, ``REGIONS``, ``SOLUTION``, ``SUMMARY``,
``SCHEDULE`` — whichever file it sits in.

The module needs nothing but the standard library; it is imported by the page.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Iterator

SECTIONS = ("RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "SUMMARY", "SCHEDULE")

#: A keyword name alone on a line, from the first column; a comment may follow it.
_HEADER = re.compile(r"^([A-Z][A-Z0-9_+\-]{0,31})[ \t]*(?:--.*)?$")
#: Text is read and written as Latin-1: every byte is one character, so any file —
#: whatever encoding its comments are in — is reproduced exactly.
ENCODING = "latin-1"


def strip_comment(line: str) -> str:
    """``line`` up to its ``--`` comment; a ``--`` inside quotes is text."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "-" and line.startswith("--", i):
            return line[:i]
    return line


def scan_record_state(line: str, is_open: bool) -> bool:
    """Whether a record is open after ``line`` (its comment already removed), given
    whether one was open before it. A ``/`` outside quotes closes the record, and what
    follows it on the line is ignored, as the simulator ignores it."""
    quote = None
    token = False
    for ch in line:
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            token = True
        elif ch == "/":
            return False
        elif not ch.isspace():
            token = True
    return True if token else is_open


@dataclass
class Keyword:
    """One keyword block: its header line and everything up to the next header.

    ``text`` is the block exactly as read — or as an editor rewrote it. ``include`` is the
    file an ``INCLUDE`` keyword names, parsed (``None`` when the bundle does not hold it).
    """

    name: str
    text: str
    section: str = ""
    include: "DeckFile | None" = None
    include_path: str = ""
    dirty: bool = False

    @property
    def body(self) -> str:
        """The block without its header line."""
        head, sep, rest = self.text.partition("\n")
        return rest if sep else ""

    def data(self) -> str:
        """The block's data: the body with comments and blank lines removed."""
        lines = (strip_comment(line).rstrip() for line in self.body.splitlines())
        return "\n".join(line for line in lines if line.strip())

    def set_body(self, body: str) -> None:
        """Replace the data of the block, keeping its header line."""
        head = self.text.partition("\n")[0]
        if body and not body.endswith("\n"):
            body += "\n"
        self.text = f"{head}\n{body}"
        self.dirty = True


@dataclass
class DeckFile:
    """One file of the tree: the text before its first keyword, then its keywords."""

    path: str
    preamble: str = ""
    keywords: list[Keyword] = field(default_factory=list)
    raw: bytes | None = None            # what was read; written back while nothing changed
    dirty: bool = False

    def changed(self) -> bool:
        return self.dirty or self.raw is None or any(k.dirty for k in self.keywords)

    def render(self) -> bytes:
        """The file's bytes: the bytes read when nothing in it changed, else its blocks."""
        if not self.changed():
            assert self.raw is not None
            return self.raw
        return (self.preamble + "".join(k.text for k in self.keywords)).encode(ENCODING)


class DeckError(ValueError):
    """A deck tree that cannot be read as one."""


def split_keywords(text: str) -> tuple[str, list[tuple[str, str]]]:
    """``(preamble, [(name, block), …])`` of one file's text."""
    preamble: list[str] = []
    blocks: list[tuple[str, list[str]]] = []
    is_open = False
    title_line = False
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        if title_line and bare.strip():
            blocks[-1][1].append(line)        # the title itself: any text at all
            title_line = False
            continue
        match = _HEADER.match(bare)
        name = match.group(1) if match else ""
        if name and (name in SECTIONS or not is_open):
            blocks.append((name, [line]))
            is_open = False
            title_line = name == "TITLE"
            continue
        if blocks:
            blocks[-1][1].append(line)
            is_open = scan_record_state(strip_comment(bare), is_open)
        else:
            preamble.append(line)
    return "".join(preamble), [(name, "".join(lines)) for name, lines in blocks]


def include_target(keyword_text: str) -> str:
    """The path an ``INCLUDE`` block names: its first record, quoted or not."""
    data = "\n".join(strip_comment(line) for line in keyword_text.splitlines()[1:]).strip()
    if not data:
        return ""
    if data[0] in "'\"":
        end = data.find(data[0], 1)
        return data[1:end if end > 0 else None].strip()
    return data.rsplit("/", 1)[0].strip() if data.rstrip().endswith("/") else data.split()[0]


def _lookup(tree: dict[str, bytes], path: str) -> str | None:
    """The member of ``tree`` that ``path`` names: exactly, or regardless of case (a deck
    written on Windows names its files as it pleases)."""
    norm = str(PurePosixPath(path.replace("\\", "/")))
    while norm.startswith("./"):
        norm = norm[2:]
    if norm in tree:
        return norm
    lowered = norm.lower()
    for name in tree:
        if name.lower() == lowered:
            return name
    return None


class Deck:
    """A deck tree as one document.

    ``root`` is the main deck's file; ``files`` every file of the tree by its path;
    ``other`` the bundle members no ``INCLUDE`` reaches (kept and written back unchanged).
    """

    def __init__(self, root: DeckFile, files: dict[str, DeckFile], other: dict[str, bytes] | None = None,
                 missing: list[str] | None = None):
        self.root = root
        self.files = files
        self.other = dict(other or {})
        self.missing = list(missing or [])       # INCLUDE targets the tree does not hold

    # ------------------------------------------------------------------ reading
    @classmethod
    def from_tree(cls, tree: dict[str, bytes], root_name: str) -> "Deck":
        """Parse the deck ``root_name`` of ``tree`` (member path → bytes) and every file it
        includes."""
        member = _lookup(tree, root_name)
        if member is None:
            raise DeckError(f"the bundle has no file {root_name!r}")
        base = PurePosixPath(member).parent
        files: dict[str, DeckFile] = {}
        missing: list[str] = []
        state = {"section": "", "paths": {}}

        def parse(name: str, chain: tuple[str, ...]) -> DeckFile:
            raw = tree[name]
            preamble, blocks = split_keywords(raw.decode(ENCODING))
            deck_file = DeckFile(path=name, preamble=preamble, raw=raw)
            files[name] = deck_file
            for kw_name, block in blocks:
                if kw_name in SECTIONS:
                    state["section"] = kw_name
                keyword = Keyword(name=kw_name, text=block, section=state["section"])
                deck_file.keywords.append(keyword)
                if kw_name == "PATHS":
                    for alias, target in re.findall(r"'([^']+)'\s+'([^']+)'", keyword.data()):
                        state["paths"][alias] = target
                elif kw_name == "INCLUDE":
                    target = include_target(block)
                    for alias, real in state["paths"].items():
                        target = target.replace(f"${alias}", real)
                    keyword.include_path = target
                    found = _lookup(tree, str(base / target)) if target else None
                    if found is None:
                        missing.append(target)
                    elif found in chain:
                        raise DeckError(f"{name} includes {found}, which includes it")
                    elif found in files:
                        keyword.include = files[found]      # one file included twice: read once
                    else:
                        keyword.include = parse(found, (*chain, found))
            return deck_file

        root = parse(member, (member,))
        other = {name: data for name, data in tree.items() if name not in files}
        return cls(root, files, other, missing)

    @classmethod
    def from_text(cls, text: str, name: str = "MODEL.DATA") -> "Deck":
        """A single-file deck from its text."""
        return cls.from_tree({name: text.encode(ENCODING)}, name)

    @classmethod
    def from_zip(cls, data: bytes, root_name: str = "") -> "Deck":
        """A deck tree from a zip archive; ``root_name`` names the deck, else it is the only
        ``.DATA`` file at the archive's top level."""
        tree: dict[str, bytes] = {}
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                path = PurePosixPath(info.filename.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts:
                    raise DeckError(f"archive member {info.filename!r} does not stay inside the archive")
                tree[str(path)] = archive.read(info)
        if not root_name:
            decks = [n for n in tree if "/" not in n and n.upper().endswith(".DATA")]
            if len(decks) != 1:
                raise DeckError(f"the archive holds {len(decks)} top-level .DATA files; name the deck")
            root_name = decks[0]
        return cls.from_tree(tree, root_name)

    # ------------------------------------------------------------------ reading it
    @property
    def name(self) -> str:
        return PurePosixPath(self.root.path).stem

    def walk(self, into_includes: bool = True) -> Iterator[tuple[DeckFile, Keyword]]:
        """Every keyword in the order the simulator reads it, with the file it sits in."""

        def visit(deck_file: DeckFile, seen: tuple[str, ...]) -> Iterator[tuple[DeckFile, Keyword]]:
            for keyword in deck_file.keywords:
                yield deck_file, keyword
                if into_includes and keyword.include is not None and keyword.include.path not in seen:
                    yield from visit(keyword.include, (*seen, keyword.include.path))

        yield from visit(self.root, (self.root.path,))

    def keywords(self, name: str | None = None, section: str | None = None) -> list[Keyword]:
        """The keywords named ``name`` (all when ``None``), in ``section`` (any when ``None``)."""
        return [k for _, k in self.walk()
                if (name is None or k.name == name) and (section is None or k.section == section)]

    def first(self, name: str, section: str | None = None) -> Keyword | None:
        found = self.keywords(name, section)
        return found[0] if found else None

    def has(self, name: str, section: str | None = None) -> bool:
        return self.first(name, section) is not None

    def outline(self) -> dict[str, list[str]]:
        """Section → the names of its keywords in order (``INCLUDE`` and the section
        headers themselves left out; a name repeated is listed once per run of it)."""
        out: dict[str, list[str]] = {s: [] for s in SECTIONS}
        for _, keyword in self.walk():
            if keyword.name in SECTIONS or keyword.name == "INCLUDE" or keyword.section not in out:
                continue
            names = out[keyword.section]
            if not names or names[-1] != keyword.name:
                names.append(keyword.name)
        return out

    def file_of(self, keyword: Keyword) -> DeckFile:
        for deck_file, candidate in self.walk():
            if candidate is keyword:
                return deck_file
        raise KeyError(keyword.name)

    # ------------------------------------------------------------------ editing it
    def section_end(self, section: str) -> tuple[DeckFile, int]:
        """Where a keyword appended to ``section`` goes in the **root** file: just before
        the next section's header (the end of the file for the last section)."""
        if section not in SECTIONS:
            raise DeckError(f"unknown section {section!r}")
        later = SECTIONS[SECTIONS.index(section) + 1:]
        inside = False
        for index, keyword in enumerate(self.root.keywords):
            if keyword.name == section:
                inside = True
            elif inside and keyword.name in later:
                return self.root, index
        if not inside:
            raise DeckError(f"the deck has no {section} section")
        # END closes a deck: nothing is read after it
        for index, keyword in enumerate(self.root.keywords):
            if keyword.name == "END":
                return self.root, index
        return self.root, len(self.root.keywords)

    def append(self, section: str, name: str, body: str = "") -> Keyword:
        """Add a keyword at the end of ``section`` (in the root file) and return it."""
        deck_file, index = self.section_end(section)
        keyword = Keyword(name=name, text=block_text(name, body), section=section, dirty=True)
        # A blank line keeps the new block apart from a predecessor that ends flush.
        if index > 0 and not deck_file.keywords[index - 1].text.endswith("\n"):
            deck_file.keywords[index - 1].text += "\n"
            deck_file.keywords[index - 1].dirty = True
        deck_file.keywords.insert(index, keyword)
        deck_file.dirty = True
        return keyword

    def put(self, section: str, name: str, body: str) -> Keyword:
        """Set the data of the first keyword ``name`` of ``section``, adding it when the
        section has none. The keyword keeps its place and its header line."""
        keyword = self.first(name, section)
        if keyword is None:
            return self.append(section, name, body)
        keyword.set_body(body)
        return keyword

    def remove(self, keyword: Keyword) -> None:
        deck_file = self.file_of(keyword)
        deck_file.keywords = [k for k in deck_file.keywords if k is not keyword]
        deck_file.dirty = True

    def ensure_section(self, section: str) -> None:
        """Make sure the root file has ``section``'s header, in its canonical place."""
        if any(k.name == section for k in self.root.keywords):
            return
        later = SECTIONS[SECTIONS.index(section) + 1:]
        index = next((i for i, k in enumerate(self.root.keywords) if k.name in later or k.name == "END"),
                     len(self.root.keywords))
        self.root.keywords.insert(index, Keyword(name=section, text=f"{section}\n\n", section=section, dirty=True))
        self.root.dirty = True
        for keyword in self.root.keywords[index + 1:]:
            if keyword.name in SECTIONS:
                break
            keyword.section = section

    # ------------------------------------------------------------------ writing
    def tree(self) -> dict[str, bytes]:
        """Member path → bytes of the whole tree: its files, then whatever else the bundle held."""
        out = {path: deck_file.render() for path, deck_file in self.files.items()}
        for path, data in self.other.items():
            out.setdefault(path, data)
        return out

    def changed(self) -> bool:
        return any(f.changed() for f in self.files.values())

    def text(self) -> str:
        """The root file's text."""
        return self.root.render().decode(ENCODING)


def block_text(name: str, body: str = "") -> str:
    """A keyword block from its name and its data: the header, the data, a blank line."""
    body = body.strip("\n")
    return f"{name}\n{body}\n\n" if body else f"{name}\n\n"


def to_zip(members: dict[str, bytes]) -> bytes:
    """A zip archive of ``members`` (path → bytes), in sorted order so equal trees give
    equal archives."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, members[name])
    return buf.getvalue()


_KNOWN: frozenset[str] | None = None
#: Families the simulator's list names by a stem: ``TVDPFSEA`` is ``TVDP`` for tracer ``FSEA``.
_FAMILY_STEMS = ("TVDP", "TBLK", "TNUM", "MULT", "WELPI", "VFPPROD", "VFPINJ")


def known_keywords() -> frozenset[str] | None:
    """The keyword names OPM Flow's parser lists, or ``None`` where the ``opm`` package is
    not installed. The list is a hint for a deck check, never a filter for the scanner: it
    leaves out the families the parser matches by pattern."""
    global _KNOWN
    if _KNOWN is None:
        try:
            from opm.io import Builtin
        except Exception:  # noqa: BLE001 - absent or broken: nothing depends on it
            return None
        _KNOWN = frozenset(n for n in dir(Builtin()) if not n.startswith("_"))
    return _KNOWN


def recognised(name: str, section: str = "") -> bool | None:
    """Whether the simulator knows the keyword ``name``: ``True``/``False``, or ``None``
    when that cannot be said (no keyword list, or a summary vector — an open family)."""
    known = known_keywords()
    if known is None or section == "SUMMARY":
        return None
    return name in known or any(name.startswith(stem) for stem in _FAMILY_STEMS)
