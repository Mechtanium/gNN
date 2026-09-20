r"""
Stage B of the workflow: extraction through ResInsight, as ``Prep-Reservoir.ipynb`` does.

ResInsight is launched headless (the ``rips`` client execs
``RESINSIGHT_EXECUTABLE``, which in the image is an ``xvfb-run`` wrapper), the
simulator outputs beside the deck are read through its gRPC API — grid corners,
rock properties, every report step's cell states, the completion and summary
tables — and the black-oil tables come from the deck text. The result is the
preprocessing cache the training pipeline loads (:mod:`modules.casedata`).

The notebook's ``BUILD_KWARGS`` are reproduced here, minus the single-phase
constants (``mu``, ``c_t``, ``rho``, ``g``) that nothing downstream reads, and
minus the vertex-state and raw state payloads that only the notebook's EDA
views used (which is what shrinks the cache from ~500 MB to ~150 MB on SPE-2).
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any, Callable

STATE_ATTRS = ("PRESSURE", "SWAT", "SGAS", "RS")

#: The notebook's extraction settings (Prep-Reservoir.ipynb, cell 3).
BUILD_KWARGS: dict[str, Any] = {
    "backend": "rips",
    "lifecycle": "attach_or_launch",
    "state_attrs": STATE_ATTRS,
    "state_vertex_attrs": (),          # the pipeline reads cell states only
    "selected_steps": None,
    "max_steps": None,
    "component_xtol": 1e-3,
    "include_blackoil_tables": True,
    "include_aquifers": True,
    "allow_missing_blackoil_tables": False,
    "allow_missing_aquifers": True,
}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def configure_resinsight(executable: str | None = None, home: str | None = None) -> None:
    """Point ``rips`` at the ResInsight binary and a writable HOME, on a port of
    our own: the worker's gRPC server already owns 50051 inside the pod.

    ``RESINSIGHT_EXECUTABLE`` and ``RESINSIGHT_GRPC_PORT`` already set in the
    environment win (a developer with a local ResInsight); otherwise the image's
    wrapper at ``/opt/ResInsight/bin/resinsight-xvfb`` is used.
    """
    if executable is None:
        executable = os.environ.get("RESINSIGHT_EXECUTABLE") or "/opt/ResInsight/bin/resinsight-xvfb"
    os.environ["RESINSIGHT_EXECUTABLE"] = executable
    os.environ.setdefault("RESINSIGHT_GRPC_PORT", str(_free_port()))
    if home is not None:
        os.environ["HOME"] = home
    elif not os.access(os.path.expanduser("~"), os.W_OK):
        os.environ["HOME"] = "/tmp"


def build_prep_cache(deck_path: str | Path, prep_cache_dir: str | Path,
                     progress: Callable[[str], None] | None = None, *,
                     force_rebuild: bool = False):
    """Build (or load) the preprocessing cache for ``deck_path`` under ``prep_cache_dir``.

    Returns the artifacts. ResInsight is closed when extraction ends so it does
    not hold memory during training.
    """
    from modules.utils.ReservoirPrepCache import build_or_load_preprocessing_artifacts

    if progress is not None:
        progress("extracting the grid, states and wells through ResInsight")
    configure_resinsight()
    return build_or_load_preprocessing_artifacts(
        str(Path(deck_path).resolve()), str(prep_cache_dir),
        force_rebuild=force_rebuild, build_if_missing=True, build_kwargs=dict(BUILD_KWARGS))
