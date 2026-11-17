r"""
Home-relative paths for the on-disk caches.

A cache written under one account has to stay readable under another: the same checkout
sits at ``/home/<user>/...`` on every machine, and only ``<user>`` differs. So a path under
the home directory is stored as ``~/...`` and expanded again when it is read, which keeps
absolute paths out of what is written to disk while everything in memory still sees an
ordinary absolute path. A path outside the home directory is stored as it is.

Metadata written before this convention holds absolute paths; :func:`local_path` leaves
those untouched, so such a cache keeps loading on the machine that wrote it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def portable_path(path: str | Path) -> str:
    r"""
    The form of ``path`` to store on disk: ``~/...`` when it lies under the home directory.

    ``path`` must already be resolved for the comparison with the home directory to hold;
    a relative or outside path is returned as it is.
    """
    path = Path(path).expanduser()
    home = Path.home().resolve()
    if path.is_absolute() and path.is_relative_to(home):
        return f"~/{path.relative_to(home).as_posix()}"
    return str(path)


def local_path(path: str | Path) -> str:
    r"""The in-memory form of a stored path: ``~`` expanded against this account's home."""
    return str(Path(path).expanduser())


def _map_signature(signature: dict | None, convert: Callable[[Any], str]) -> dict:
    return {role: {**entry, "path": convert(entry["path"])} if "path" in entry else dict(entry)
            for role, entry in (signature or {}).items()}


def portable_signature(signature: dict | None) -> dict:
    r"""A ``source_signature`` record with each file path in its stored form."""
    return _map_signature(signature, portable_path)


def local_signature(signature: dict | None) -> dict:
    r"""A stored ``source_signature`` record with each file path back in its in-memory form."""
    return _map_signature(signature, local_path)
