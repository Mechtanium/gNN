from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover - fallback for minimal environments
    import pickle

    class _JoblibCompat:
        @staticmethod
        def dump(obj, filename):
            with open(filename, "wb") as handle:
                pickle.dump(obj, handle)

        @staticmethod
        def load(filename):
            with open(filename, "rb") as handle:
                return pickle.load(handle)

    joblib = _JoblibCompat()

from modules.utils.portable_paths import local_path, local_signature, portable_path, portable_signature
from modules.utils.ReservoirExtractors import build_reservoir_preprocessing_artifacts, resolve_reservoir_source


PREP_CACHE_VERSION = 2  # v2: artifacts normalized to FIELD units at extraction (unit_conversion)
# Revision of the extraction code that writes the cache. A cache recorded under an older
# revision still loads (the layout is unchanged) but is reported stale, so it is rebuilt the
# next time Prep-Reservoir.ipynb or a campaign prep job runs. Bump on any change to what the
# extraction computes; caches written before the field existed count as revision 1.
# r2: well observations matched to report dates within the summary's float32 time rounding.
PREP_CODE_REVISION = 2
PREP_ARTIFACTS_FILENAME = "reservoir_preprocessing_artifacts.joblib"
PREP_METADATA_FILENAME = "reservoir_preprocessing_metadata.json"


def preprocessing_artifacts_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / PREP_ARTIFACTS_FILENAME


def preprocessing_metadata_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / PREP_METADATA_FILENAME


def preprocessing_cache_exists(cache_dir: str | Path) -> bool:
    cache_dir = Path(cache_dir)
    return preprocessing_artifacts_path(cache_dir).exists() and preprocessing_metadata_path(cache_dir).exists()


def source_signature(model_path: str | Path) -> dict[str, dict[str, Any]]:
    """Size + mtime of every simulator file a cache was built from (deck, EGRID, INIT, UNRST,
    summary), keyed by role. Comparing it with the on-disk state tells a cache built from an
    earlier simulator run apart from a current one — the files are the truth, not the cache."""
    from modules.utils.ReservoirExtractors import resolve_reservoir_source

    out: dict[str, dict[str, Any]] = {}
    try:
        src = resolve_reservoir_source(model_path)
        candidates = {
            "deck": src.data_path, "egrid": src.egrid_path, "init": src.init_path,
            "unrst": src.unrst_path, "smspec": src.smspec_path, "esmry": src.esmry_path,
        }
    except Exception:
        candidates = {"deck": Path(model_path)}
    for role, path in candidates.items():
        if path is None:
            continue
        path = Path(path)
        if not path.is_file():
            continue
        st = path.stat()
        out[role] = {"path": str(path.resolve()), "size": int(st.st_size), "mtime": float(st.st_mtime)}
    return out


def stale_source_files(recorded: dict | None, current: dict | None) -> list[str]:
    """Roles whose file changed (size or mtime), appeared, or vanished since ``recorded``."""
    recorded = dict(recorded or {})
    current = dict(current or {})
    changed: list[str] = []
    for role in sorted(set(recorded) | set(current)):
        a, b = recorded.get(role), current.get(role)
        if a is None or b is None:
            changed.append(role)
            continue
        if a.get("path") != b.get("path") or int(a.get("size", -1)) != int(b.get("size", -2)) \
                or abs(float(a.get("mtime", 0.0)) - float(b.get("mtime", 0.0))) > 1e-6:
            changed.append(role)
    return changed


def preprocessing_cache_stale_reasons(cache_dir: str | Path) -> list[str]:
    """Human-readable reasons the prep cache no longer matches the simulator files on disk
    (empty when it does, or when the cache predates signatures — nothing to compare)."""
    try:
        metadata = load_preprocessing_metadata(cache_dir)
    except (FileNotFoundError, ValueError):
        return []
    reasons = []
    revision = int(metadata.get("code_revision", 1))
    if revision < PREP_CODE_REVISION:
        reasons.append(f"written by extraction code revision {revision}, the current one is "
                       f"{PREP_CODE_REVISION}")
    recorded = metadata.get("source_signature")
    if not recorded:
        return reasons
    current = source_signature(metadata["model_path"])
    return reasons + [
        f"{role} ({Path(current.get(role, recorded.get(role, {})).get('path', role)).name}) "
        f"changed on disk after the cache was built"
        for role in stale_source_files(recorded, current)]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _artifact_summary(artifacts) -> dict[str, Any]:
    reservoir_mesh = artifacts.reservoir_mesh
    state_data = artifacts.state_data
    return {
        "extractor_backend": artifacts.extractor_backend,
        "auxiliary_backend": artifacts.auxiliary_backend,
        "source_units": getattr(artifacts, "source_units", "FIELD"),
        "cache_units": "FIELD",
        "n_vertices": int(reservoir_mesh.verts.shape[0]),
        "n_cells": int(reservoir_mesh.active_cell_indices.shape[0]),
        "n_times": int(state_data["n_times"]),
        "state_indices": np.asarray(state_data["indices"], dtype=np.int32),
        "report_steps": np.asarray(state_data.get("report_steps", state_data["indices"]), dtype=np.int32),
        "state_attrs": sorted(str(name) for name in artifacts.cell_states),
        "vertex_state_attrs": sorted(str(name) for name in artifacts.vertex_states),
        "has_blackoil_tables": artifacts.blackoil_tables is not None,
        "n_aquifer_vertices": int(np.asarray(artifacts.aquifer_vertices).size),
        "timings": artifacts.timings,
    }


def save_preprocessing_artifacts(
    cache_dir: str | Path,
    artifacts,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    r"""
    Write the artifacts and their metadata to ``cache_dir``.

    Paths in the metadata are stored in the home-relative form of
    :func:`modules.utils.portable_paths.portable_path`, so the cache does not depend on the account
    that built it; :func:`load_preprocessing_metadata` expands them again.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, preprocessing_artifacts_path(cache_dir))
    metadata = {
        "cache_version": PREP_CACHE_VERSION,
        "code_revision": PREP_CODE_REVISION,
        "model_path": portable_path(Path(artifacts.source.data_path).resolve()),
        "cache_dir": portable_path(cache_dir.resolve()),
        "artifacts_file": PREP_ARTIFACTS_FILENAME,
        "config": _json_ready(config or {}),
        "summary": _json_ready(_artifact_summary(artifacts)),
        "source_signature": portable_signature(_json_ready(source_signature(artifacts.source.data_path))),
    }
    preprocessing_metadata_path(cache_dir).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def load_preprocessing_metadata(cache_dir: str | Path) -> dict[str, Any]:
    r"""
    Read a cache's metadata, with its stored paths expanded for this account.

    Every reader of the metadata's paths goes through here, so only the file on disk holds
    ``~``; metadata written with absolute paths is returned as it stands.
    """
    path = preprocessing_metadata_path(cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing reservoir preprocessing metadata: {path}")
    metadata = json.loads(path.read_text())
    if int(metadata.get("cache_version", -1)) != PREP_CACHE_VERSION:
        raise ValueError(
            f"Reservoir preprocessing cache version mismatch: expected {PREP_CACHE_VERSION}, "
            f"found {metadata.get('cache_version')}"
        )
    for key in ("model_path", "cache_dir"):
        if key in metadata:
            metadata[key] = local_path(metadata[key])
    if metadata.get("source_signature"):
        metadata["source_signature"] = local_signature(metadata["source_signature"])
    return metadata


def _rebind_source(artifacts, model_path: str):
    r"""
    Point a loaded artifact's ``source`` at ``model_path`` when it recorded another location.

    The pickled source keeps the absolute paths of the account that built the cache, which
    do not exist under another home. The recorded source stays when the deck or its grid
    is not found here, since the cached arrays are still usable without them.
    """
    source = getattr(artifacts, "source", None)
    if source is not None and Path(source.data_path) != Path(model_path):
        try:
            artifacts.source = resolve_reservoir_source(model_path)
        except FileNotFoundError:
            pass
    return artifacts


def load_preprocessing_artifacts(
    cache_dir: str | Path,
    expected_model_path: str | Path | None = None,
):
    metadata = load_preprocessing_metadata(cache_dir)
    if expected_model_path is not None:
        expected = str(Path(expected_model_path).expanduser().resolve())
        actual = str(Path(metadata["model_path"]).resolve())
        if expected != actual:
            raise ValueError(f"Reservoir preprocessing cache model mismatch: expected {expected}, found {actual}")
    artifacts_path = preprocessing_artifacts_path(cache_dir)
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Missing reservoir preprocessing artifacts: {artifacts_path}")
    stale = preprocessing_cache_stale_reasons(cache_dir)
    if stale:
        warnings.warn(
            f"reservoir preprocessing cache {cache_dir} is STALE — the simulator files it was "
            f"built from have changed: {'; '.join(stale)}. Rebuild it (Prep-Reservoir.ipynb with "
            "FORCE_REBUILD = True) before trusting any state, well or table read from it.",
            RuntimeWarning, stacklevel=2)
    return _rebind_source(joblib.load(artifacts_path), metadata["model_path"])


def build_or_load_preprocessing_artifacts(
    model_path: str | Path,
    cache_dir: str | Path,
    *,
    force_rebuild: bool = False,
    build_if_missing: bool = True,
    build_kwargs: dict[str, Any] | None = None,
):
    cache_dir = Path(cache_dir)
    if not force_rebuild and preprocessing_cache_exists(cache_dir):
        stale = preprocessing_cache_stale_reasons(cache_dir)
        if stale and build_if_missing:
            print(f"[prep-cache] rebuilding {cache_dir}: {'; '.join(stale)}")
            force_rebuild = True
        else:
            if stale:
                warnings.warn(
                    f"the reservoir preprocessing cache {cache_dir} is stale "
                    f"({'; '.join(stale)}); it is used as it is. Rebuild it with "
                    "Prep-Reservoir.ipynb (FORCE_REBUILD) or a campaign prep job.",
                    RuntimeWarning, stacklevel=2)
            return load_preprocessing_artifacts(cache_dir, expected_model_path=model_path)
    if not build_if_missing and not force_rebuild:
        raise FileNotFoundError(
            f"No reservoir preprocessing cache found in {cache_dir}. "
            "Run Prep-Reservoir.ipynb first, or set BUILD_PREP_CACHE_IF_MISSING = True."
        )
    kwargs = dict(build_kwargs or {})
    artifacts = build_reservoir_preprocessing_artifacts(str(Path(model_path).expanduser().resolve()), **kwargs)
    save_preprocessing_artifacts(cache_dir, artifacts, config=kwargs)
    return artifacts
