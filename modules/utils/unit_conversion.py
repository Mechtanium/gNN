r"""
ECLIPSE-deck unit-system normalization for the reservoir preprocessing cache.

The :math:`\Delta`-DGM-PINN stack assumes **FIELD** units end-to-end (``psia``, ``cP``,
``rb/stb``, ``Mscf/stb``, ``lb/ft^3``, ``psi^{-1}``, ``ft``, ``mD``): see
``blackoil_closures`` and the notebook control-panel constants
(``FIELD_DARCY_COEFF``, ``RB_PER_FT3``, ``GRAV_GRAD``). ECLIPSE decks, however, may
declare ``METRIC`` (bar, sm3/day, kg/m3, m) — Norne does — and the extraction backends
(ResInsight ``rips`` for geometry/states, DeepField for the black-oil tables) return raw
values in the deck's native unit system with no conversion.

This module supplies the deck unit-system detector and the single post-extraction pass
that rescales every deck-sourced physical quantity of a
:class:`ReservoirExtractors.ReservoirPreprocessingArtifacts` bundle into FIELD units, so
the cache is FIELD-consistent regardless of the source deck. The pass is a **no-op for
FIELD decks** (guaranteeing bit-identical caches for SPE1/SPE9), applied once after the
bundle is fully assembled — at which point every array is uniformly in native units, so
mixed-provenance fields (e.g. a well datum depth :math:`z_{bh}` taken from the schedule
``DREF`` vs a cell-centroid depth :math:`z_{cell}` taken from the mesh) convert
consistently.

Conversion follows the standard ECLIPSE METRIC :math:`\to` FIELD factors. Writing a
quantity :math:`q` with factor :math:`\gamma_q`, the field value is
:math:`q_{\mathrm{FIELD}}=\gamma_q\,q_{\mathrm{METRIC}}`, with

.. math::

    \gamma_p = 14.503774,\quad
    \gamma_L = 3.2808399,\quad
    \gamma_V = \gamma_L^3,\quad
    \gamma_{\text{liq}} = 6.2898108,\quad
    \gamma_{\text{gas}} = 0.035314667,

.. math::

    \gamma_\rho = 0.062427961,\quad
    \gamma_c = \gamma_p^{-1},\quad
    \gamma_{R_{so}} = \gamma_{\text{gas}}/\gamma_{\text{liq}},\quad
    \gamma_{B_g} = \gamma_{\text{liq}}/\gamma_{\text{gas}}.

where:
- :math:`\gamma_p`: pressure, ``bar`` :math:`\to` ``psia``.
- :math:`\gamma_L,\gamma_V`: length ``m`` :math:`\to` ``ft`` and bulk/reservoir volume ``m^3`` :math:`\to` ``ft^3``.
- :math:`\gamma_{\text{liq}}`: liquid **and reservoir** volume ``m^3`` :math:`\to` ``bbl`` (surface ``sm3`` :math:`\to` ``stb``).
- :math:`\gamma_{\text{gas}}`: surface gas volume ``sm3`` :math:`\to` ``Mscf``.
- :math:`\gamma_\rho`: mass density ``kg/m^3`` :math:`\to` ``lb/ft^3``.
- :math:`\gamma_c`: compressibility ``bar^{-1}`` :math:`\to` ``psi^{-1}``.
- :math:`\gamma_{R_{so}}`: solution GOR ``sm3/sm3`` :math:`\to` ``Mscf/stb``.
- :math:`\gamma_{B_g}`: gas formation-volume factor ``rm3/sm3`` :math:`\to` ``rb/Mscf``.

Quantities that are unit-invariant between the two systems — permeability (``mD``),
porosity and saturation (fraction), relative permeability, viscosity (``cP``), the
oil/water formation-volume factors (``rb/stb``, a reservoir/surface **liquid** ratio),
and time (``days``) — are passed through unchanged.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np

FIELD = "FIELD"
METRIC = "METRIC"
LAB = "LAB"

# --- base ECLIPSE METRIC -> FIELD factors -------------------------------------------------
BAR_TO_PSI = 14.503773773           # gamma_p : bar   -> psia
M_TO_FT = 3.280839895013123         # gamma_L : m     -> ft
M3_TO_BBL = 6.289810770432105       # gamma_liq : m^3 -> bbl (reservoir) / sm3 -> stb (surface liquid)
SM3_GAS_TO_MSCF = 0.035314666721489  # gamma_gas : sm3 -> Mscf (surface gas)
KG_M3_TO_LB_FT3 = 0.06242796057614462  # gamma_rho : kg/m^3 -> lb/ft^3

# --- derived factors ----------------------------------------------------------------------
M3_TO_FT3 = M_TO_FT ** 3                       # gamma_V   : m^3 -> ft^3 (bulk/geometric volume)
INV_BAR_TO_INV_PSI = 1.0 / BAR_TO_PSI          # gamma_c   : 1/bar -> 1/psi
RS_METRIC_TO_FIELD = SM3_GAS_TO_MSCF / M3_TO_BBL   # gamma_Rso : sm3/sm3 -> Mscf/stb
BG_METRIC_TO_FIELD = M3_TO_BBL / SM3_GAS_TO_MSCF   # gamma_Bg  : rm3/sm3 -> rb/Mscf

_UNIT_KEYWORDS = {"METRIC": METRIC, "FIELD": FIELD, "LAB": LAB}
# a RUNSPEC unit keyword sits alone on its line (optionally trailed by a comment)
_UNIT_LINE = re.compile(r"^\s*(METRIC|FIELD|LAB)\b\s*(?:--.*)?$", re.IGNORECASE)


def detect_unit_system(data_path: str | Path, aux_field: Any = None) -> str:
    r"""
    Detect the unit system (:data:`FIELD` / :data:`METRIC` / :data:`LAB`) declared by an
    ECLIPSE deck.

    The RUNSPEC section carries the unit system as a bare keyword on its own line
    (``METRIC`` / ``FIELD`` / ``LAB``); the deck text is scanned for the first such line.
    When the deck omits it (e.g. it is supplied through an ``INCLUDE``) the DeepField
    ``aux_field.meta['UNITS']`` is consulted as a fallback. If neither source resolves the
    system the function **warns and returns FIELD**: FIELD makes
    :func:`to_field_units` a no-op, so an undetectable deck is treated exactly as the
    pre-normalization pipeline did (native values passed through, no regression).

    :param data_path: Path to the ``.DATA`` deck.
    :param aux_field: Optional loaded DeepField ``Field`` whose ``meta['UNITS']`` is used
        as a fallback / cross-check.
    :returns: One of :data:`FIELD`, :data:`METRIC`, :data:`LAB`.
    """
    try:
        text = Path(data_path).read_text(errors="ignore")
    except OSError:
        text = ""
    for line in text.splitlines():
        match = _UNIT_LINE.match(line)
        if match:
            return _UNIT_KEYWORDS[match.group(1).upper()]

    meta = getattr(aux_field, "meta", None)
    if isinstance(meta, dict):
        units = str(meta.get("UNITS", "")).upper()
        if units in _UNIT_KEYWORDS:
            return _UNIT_KEYWORDS[units]

    warnings.warn(
        f"Could not detect the unit system (METRIC/FIELD/LAB) for deck {data_path}; "
        "no RUNSPEC unit keyword found and no DeepField UNITS metadata available. "
        "Assuming FIELD (no unit conversion). If this deck is METRIC, add its RUNSPEC "
        "unit keyword so the cache is normalized correctly.",
        stacklevel=2,
    )
    return FIELD


def _scale_array(value: Any, factor: float) -> Any:
    """Return ``value * factor`` as a float array, passing ``None`` through unchanged."""
    if value is None:
        return None
    return np.asarray(value, dtype=float) * factor


def _scale_scalar(value: Any, factor: float) -> Any:
    """Scale a scalar dict entry by ``factor``; ``None``/non-numeric pass through, ``nan`` stays ``nan``."""
    if value is None:
        return None
    try:
        return float(value) * factor
    except (TypeError, ValueError):
        return value


# --- black-oil table pack -----------------------------------------------------------------
# key -> METRIC->FIELD factor for the pressure/GOR/FVF/compressibility/density/depth columns.
# Everything omitted here (saturations, rel-perm, viscosities, oil/water FVF, swc) is invariant.
_BLACKOIL_FACTORS: dict[str, float] = {
    "pvtw_pressure": BAR_TO_PSI,
    "pvdg_pressure": BAR_TO_PSI,
    "pvto_pbub_sat": BAR_TO_PSI,
    "rock_pref": BAR_TO_PSI,
    "pvto_rs_sat": RS_METRIC_TO_FIELD,
    "rsvd_rs": RS_METRIC_TO_FIELD,
    "pvdg_fvf": BG_METRIC_TO_FIELD,
    "rock_compr": INV_BAR_TO_INV_PSI,
    "pvtw_compr": INV_BAR_TO_INV_PSI,
    "pvtw_viscosibility": INV_BAR_TO_INV_PSI,
    "dens_o": KG_M3_TO_LB_FT3,
    "dens_w": KG_M3_TO_LB_FT3,
    "dens_g": KG_M3_TO_LB_FT3,
    "rsvd_depth": M_TO_FT,
}


def convert_blackoil_pack(pack: dict[str, np.ndarray], source_units: str) -> dict[str, np.ndarray]:
    r"""
    Return a copy of a :func:`ReservoirMesh.build_blackoil_table_pack` dictionary rescaled
    from ``source_units`` to FIELD units.

    Pressure nodes (PVTW/PVDG/PVTO bubble-point, ROCK :math:`p_{\mathrm{ref}}`), solution
    GOR (PVTO/RSVD :math:`R_{so}`), the gas FVF :math:`B_g` (PVDG), compressibilities
    (ROCK/PVTW), surface densities, and RSVD depth are scaled; saturations, relative
    permeabilities, viscosities, the oil/water FVF, and the connate-water endpoint are
    left unchanged. FIELD input is returned unchanged.
    """
    if source_units == FIELD:
        return pack
    if source_units != METRIC:
        raise NotImplementedError(f"Unit conversion from {source_units!r} to FIELD is not implemented.")
    out = dict(pack)
    for key, factor in _BLACKOIL_FACTORS.items():
        if key in out and out[key] is not None:
            out[key] = np.asarray(out[key], dtype=float) * factor
    return out


# --- well metadata ------------------------------------------------------------------------
# key -> factor for the schedule/summary-sourced (deck-native) well quantities.
_WELL_PRESSURE_KEYS = ("control_bhpt", "obs_wbhp", "obs_wthp", "bhp")
_WELL_LENGTH_KEYS = ("z_bh", "z_cell", "r_e", "h_s")
_WELL_LIQUID_KEYS = ("control_wit", "obs_wopr", "obs_wwpr", "obs_wwir", "obs_wopt", "obs_wwpt")
_WELL_GAS_KEYS = ("control_git", "obs_wgpr", "obs_wgir", "obs_wgpt")
# 'rate' / 'control_rate' are phase-dependent: GAS -> surface-gas factor, else liquid/reservoir.
_WELL_PHASE_RATE_KEYS = ("rate", "control_rate")


def _well_rate_factor(control_phase: Any) -> float:
    """Surface-volume factor for a phase-tagged rate: gas -> Mscf, otherwise liquid/reservoir -> bbl."""
    if str(control_phase).strip().upper() == "GAS":
        return SM3_GAS_TO_MSCF
    return M3_TO_BBL


def _convert_well_entry(entry: dict[str, Any]) -> None:
    """In-place FIELD rescale of one schedule/summary well entry (rate/BHP/depth columns)."""
    if not isinstance(entry, dict):
        return
    phase = entry.get("control_phase")
    for key in _WELL_PRESSURE_KEYS:
        if key in entry:
            entry[key] = _scale_scalar(entry[key], BAR_TO_PSI)
    for key in _WELL_LENGTH_KEYS:
        if key in entry:
            entry[key] = _scale_scalar(entry[key], M_TO_FT)
    for key in _WELL_LIQUID_KEYS:
        if key in entry:
            entry[key] = _scale_scalar(entry[key], M3_TO_BBL)
    for key in _WELL_GAS_KEYS:
        if key in entry:
            entry[key] = _scale_scalar(entry[key], SM3_GAS_TO_MSCF)
    for key in _WELL_PHASE_RATE_KEYS:
        if key in entry:
            entry[key] = _scale_scalar(entry[key], _well_rate_factor(phase))


def convert_well_metadata(well_metadata: dict[str, Any], source_units: str) -> None:
    r"""
    In-place FIELD rescale of the schedule/summary-sourced quantities of a ``well_metadata``
    mapping (BHP/THP :math:`\to` psia; perforation/datum depths and Peaceman :math:`r_e,h_s`
    :math:`\to` ft; liquid and gas rates/cumulatives by phase). Mesh-derived geometric
    fields carry the same native length units at conversion time and rescale identically;
    the ``K_perp`` (mD) tensor, skins and indices are invariant. FIELD input is a no-op.

    The perforation well-index ``weight`` (Peaceman :math:`WI=2\pi h_s\sqrt{\det K_\perp}/(\ln(r_e/r_w)+s)`,
    or a COMPDAT connection-factor on the fallback path) is **intentionally not scaled**: it is
    consumed only as a normalized fraction :math:`w_i/\sum_j w_j` when splitting a well's phase
    rate across perforations (see ``ReservoirMesh`` ``distributed = phase_rate * weight / total_weight``),
    so its absolute scale cancels. ``total_weight`` (the same sum) is likewise left unscaled.
    """
    if source_units == FIELD:
        return
    if source_units != METRIC:
        raise NotImplementedError(f"Unit conversion from {source_units!r} to FIELD is not implemented.")
    for step in well_metadata.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        for list_key in ("rate_entries", "bhp_entries", "well_results"):
            for entry in step.get(list_key) or []:
                _convert_well_entry(entry)


def to_field_units(artifacts: Any, source_units: str) -> Any:
    r"""
    Rescale every deck-sourced physical quantity of a
    :class:`ReservoirExtractors.ReservoirPreprocessingArtifacts` bundle from ``source_units``
    into FIELD units, in place, and return it.

    Converted groups: the hexahedral mesh and grid geometry (node/centroid/corner
    coordinates and cell lengths :math:`\to` ft; cell volumes :math:`\to` ft\ :sup:`3`), the
    reference states (``PRESSURE`` :math:`\to` psia, ``RS`` :math:`\to` Mscf/stb; saturations
    invariant) and their cell/vertex copies, the black-oil table pack
    (:func:`convert_blackoil_pack`), and the well schedule/summary quantities
    (:func:`convert_well_metadata`). The per-cell rock physics (``rock_data``) is invariant —
    permeability/tensors are ``mD``, diffusivity is ``mD/cP`` and storage uses the supplied
    ``c_t`` — and is left untouched.

    FIELD input is a **no-op** (returns the bundle unchanged); ``LAB`` raises
    ``NotImplementedError``.
    """
    if source_units == FIELD:
        return artifacts
    if source_units != METRIC:
        raise NotImplementedError(f"Unit conversion from {source_units!r} to FIELD is not implemented.")

    # --- mesh geometry (coordinates/lengths -> ft, volumes -> ft^3) -----------------------
    mesh = artifacts.reservoir_mesh
    for attr in ("verts", "cell_centroids", "cell_lengths", "corner_cells"):
        if getattr(mesh, attr, None) is not None:
            setattr(mesh, attr, _scale_array(getattr(mesh, attr), M_TO_FT))
    for attr in ("cell_volumes", "cell_volume_estimate"):
        if getattr(mesh, attr, None) is not None:
            setattr(mesh, attr, _scale_array(getattr(mesh, attr), M3_TO_FT3))
    volume_diag = getattr(mesh, "volume_diagnostics", None)
    if isinstance(volume_diag, dict):
        for key in ("cell_volume_min", "cell_volume_max", "cell_volume_mean"):
            if key in volume_diag and volume_diag[key] is not None:
                volume_diag[key] = _scale_scalar(volume_diag[key], M3_TO_FT3)

    # --- component diagnostics (near-touching-component distances are length-valued) ------
    comp = getattr(artifacts, "component_diagnostics", None)
    if isinstance(comp, dict):
        if comp.get("min_distance_matrix") is not None:
            comp["min_distance_matrix"] = _scale_array(comp["min_distance_matrix"], M_TO_FT)
        for key in ("global_cross_nn_min", "global_cross_nn_max", "global_positive_cross_nn_min", "xtol"):
            if key in comp and comp[key] is not None:
                comp[key] = _scale_scalar(comp[key], M_TO_FT)

    # --- grid geometry payload ------------------------------------------------------------
    geometry = artifacts.geometry
    if getattr(geometry, "corner_cells", None) is not None:
        geometry.corner_cells = _scale_array(geometry.corner_cells, M_TO_FT)
    if getattr(geometry, "cell_volumes", None) is not None:
        geometry.cell_volumes = _scale_array(geometry.cell_volumes, M3_TO_FT3)

    # --- reference states (+ cell/vertex copies) ------------------------------------------
    _STATE_FACTORS = {"PRESSURE": BAR_TO_PSI, "RS": RS_METRIC_TO_FIELD}
    for container in (artifacts.state_data, artifacts.cell_states, artifacts.vertex_states):
        if not isinstance(container, dict):
            continue
        for key, factor in _STATE_FACTORS.items():
            if key in container and container[key] is not None:
                container[key] = np.asarray(container[key], dtype=float) * factor

    # --- black-oil tables -----------------------------------------------------------------
    if artifacts.blackoil_tables is not None:
        artifacts.blackoil_tables = convert_blackoil_pack(artifacts.blackoil_tables, source_units)

    # --- well schedule / summary ----------------------------------------------------------
    if isinstance(artifacts.well_metadata, dict):
        convert_well_metadata(artifacts.well_metadata, source_units)

    return artifacts
