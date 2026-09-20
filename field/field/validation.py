"""Validation helpers for standalone model decks and loaded models."""
from pathlib import Path

import numpy as np

from .parse_utils.ascii import StringIteratorIO, preprocess_path


STANDALONE_MODEL_KEYWORDS = ("RUNSPEC", "RESTART", "RESTARTDATE")


class InvalidModelDeckError(ValueError):
    """Raised when a .DATA file is not a standalone model deck."""


class ModelValidationError(ValueError):
    """Raised when a loaded model violates expected shape or geometry contracts."""


def _normalize_path(path):
    if isinstance(path, Path):
        return path
    return preprocess_path(str(path))


def peek_root_keywords(path, encoding="auto", limit=None):
    """Return the first non-comment root keywords from the deck file only."""
    path = _normalize_path(path)
    keywords = []
    with StringIteratorIO(path, encoding=encoding) as lines:
        for line in lines:
            firstword = line.split(maxsplit=1)[0].upper()
            if not firstword:
                continue
            keywords.append(firstword)
            if (limit is not None) and (len(keywords) >= limit):
                break
    return tuple(keywords)


def classify_model_deck(path, encoding="auto", limit=None):
    """Classify a DATA deck by root-file standalone markers."""
    keywords = peek_root_keywords(path, encoding=encoding, limit=limit)
    if "RESTART" in keywords or "RESTARTDATE" in keywords:
        return "restart", keywords
    if "RUNSPEC" in keywords:
        return "standard", keywords
    return "invalid_fragment", keywords


def _component_or_none(model, name):
    try:
        return getattr(model, name)
    except (AttributeError, KeyError, AssertionError):
        return None


def _raise_validation_error(model, component, attribute, expected, actual):
    path = getattr(model, "path", "<unknown model>")
    component_label = component if attribute is None else f"{component}.{attribute}"
    raise ModelValidationError(
        f"Model validation failed for '{path}': {component_label} expected {expected}, got {actual}."
    )


def _raise_missing_component(model, component):
    path = getattr(model, "path", "<unknown model>")
    raise ModelValidationError(f"Model validation failed for '{path}': missing required component '{component}'.")


def _validated_dimens(model, grid):
    if grid is None:
        _raise_missing_component(model, "grid")
    try:
        raw_dimens = np.asarray(grid.dimens)
    except (AttributeError, KeyError) as exc:
        raise ModelValidationError(
            f"Model validation failed for '{getattr(model, 'path', '<unknown model>')}': "
            "grid.DIMENS is required for loaded-model validation."
        ) from exc

    if raw_dimens.shape != (3,):
        _raise_validation_error(model, "grid", "DIMENS", "shape (3,)", raw_dimens.shape)
    if not np.all(np.isfinite(raw_dimens)):
        _raise_validation_error(model, "grid", "DIMENS", "finite values", raw_dimens.tolist())
    if not np.allclose(raw_dimens, np.rint(raw_dimens)):
        _raise_validation_error(model, "grid", "DIMENS", "integer-like values", raw_dimens.tolist())

    dimens = tuple(int(x) for x in np.rint(raw_dimens))
    if any(x <= 0 for x in dimens):
        _raise_validation_error(model, "grid", "DIMENS", "all dimensions > 0", dimens)
    return dimens


def _validate_grid(model, grid):
    dimens = _validated_dimens(model, grid)
    active_mask = np.asarray(grid.actnum)
    if active_mask.shape != dimens:
        _raise_validation_error(model, "grid", "ACTNUM", dimens, active_mask.shape)
    if not np.issubdtype(active_mask.dtype, np.bool_):
        _raise_validation_error(model, "grid", "ACTNUM", "boolean dtype", active_mask.dtype)

    has_cornerpoint = hasattr(grid, "zcorn") or hasattr(grid, "coord")
    if has_cornerpoint:
        if not hasattr(grid, "zcorn"):
            _raise_missing_component(model, "grid.ZCORN")
        if not hasattr(grid, "coord"):
            _raise_missing_component(model, "grid.COORD")
        if np.asarray(grid.zcorn).shape != dimens + (8,):
            _raise_validation_error(model, "grid", "ZCORN", dimens + (8,), np.asarray(grid.zcorn).shape)
        expected_coord = (dimens[0] + 1, dimens[1] + 1, 6)
        if np.asarray(grid.coord).shape != expected_coord:
            _raise_validation_error(model, "grid", "COORD", expected_coord, np.asarray(grid.coord).shape)

    if any(hasattr(grid, attr) for attr in ("dx", "dy", "dz", "tops")):
        for attr in ("dx", "dy", "dz"):
            if hasattr(grid, attr):
                shape = np.asarray(getattr(grid, attr)).shape
                if shape != dimens:
                    _raise_validation_error(model, "grid", attr.upper(), dimens, shape)
        if hasattr(grid, "tops"):
            tops_shape = np.asarray(grid.tops).shape
            if tops_shape not in (dimens, dimens[:2]):
                _raise_validation_error(model, "grid", "TOPS", f"{dimens} or {dimens[:2]}", tops_shape)

    vtk_grid = None            # the VTK grid is not built in this vendored parser
    if vtk_grid is not None:
        expected_cells = int(active_mask.sum())
        actual_cells = int(vtk_grid.GetNumberOfCells())
        if actual_cells != expected_cells:
            _raise_validation_error(model, "grid", "vtk_grid", expected_cells, actual_cells)

    return dimens, active_mask


def _validate_rock(model, dimens):
    rock = _component_or_none(model, "rock")
    if rock is None:
        _raise_missing_component(model, "rock")
    for attr in getattr(rock, "attributes", ()):
        shape = np.asarray(getattr(rock, attr.lower())).shape
        if shape != dimens:
            _raise_validation_error(model, "rock", attr, dimens, shape)


def _validate_states(model, dimens):
    states = _component_or_none(model, "states")
    if states is None:
        _raise_missing_component(model, "states")

    n_times = None
    for attr in getattr(states, "attributes", ()):
        data = np.asarray(getattr(states, attr.lower()))
        if data.ndim != 4:
            _raise_validation_error(model, "states", attr, "4 dimensions (n_times, nx, ny, nz)", data.shape)
        if data.shape[1:] != dimens:
            _raise_validation_error(model, "states", attr, dimens, data.shape[1:])
        if n_times is None:
            n_times = int(data.shape[0])
        elif int(data.shape[0]) != n_times:
            _raise_validation_error(model, "states", attr, f"n_times={n_times}", data.shape[0])

    if n_times is None:
        return

    n_dates = len(getattr(states, "dates", ()))
    if n_dates not in {0, 1, n_times, max(n_times - 1, 0)}:
        _raise_validation_error(model, "states", "DATES", "{0, 1, n_times, n_times - 1}", n_dates)


def validate_loaded_model(model, components=None):
    """Validate the structural consistency of a loaded DeepField model."""
    requested = tuple(components) if components is not None else tuple(getattr(model, "components", ()))
    if not requested:
        return model

    grid = _component_or_none(model, "grid")
    dimens = None
    if "grid" in requested or any(name in requested for name in ("rock", "states")):
        dimens, _ = _validate_grid(model, grid)

    if "rock" in requested:
        _validate_rock(model, dimens)

    if "states" in requested:
        _validate_states(model, dimens)

    return model
