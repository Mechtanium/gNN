from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from modules.utils.ReservoirMesh import (
    _block_centroids_from_reservoir_mesh,
    _block_path_tangents,
    _collect_schedule_include_paths,
    _expand_eclipse_tokens,
    _extract_well_result_snapshot,
    _merge_schedule_control_cache,
    _parse_eclipse_date,
    _parse_optional_float,
    _parse_schedule_control_file,
    _parse_tstep_increment,
    _resolve_rate_phase_split,
    _resolve_well_control,
    _segment_reference_depth,
    _strip_schedule_comment,
    _spatialize_track,
    _tokenize_schedule_line,
    _well_control_phase_label,
    build_blackoil_table_pack,
    build_cell_rock_physics_from_arrays,
    compute_component_diagnostics,
    compute_qw_full_tensor,
    corner_cells_to_reservoir_mesh,
    prepare_aquifer_vertices,
    prepare_vertex_category_indices,
    select_time_indices,
)
from modules.utils.unit_conversion import detect_unit_system, to_field_units

#: Fallbacks for a COMPDAT that omits item 9 (wellbore diameter, default 1 ft ->
#: a 0.5 ft radius) or item 11 (skin, default 0): Eclipse's own defaults, so a
#: deck that relies on them is read the way the simulator reads it.
ECLIPSE_DEFAULT_R_W = 0.5
ECLIPSE_DEFAULT_SKIN = 0.0


STATE_ATTRS_DEFAULT = ("PRESSURE", "SWAT", "SGAS", "RS")
SUMMARY_VECTORS_DEFAULT = ("WBHP", "WTHP", "WOPR", "WWPR", "WGPR", "WWIR", "WGIR", "WOPT", "WWPT", "WGPT")


def _try_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _import_deepfield():
    """The vendored deck-table parser (``field/``, formerly DeepField)."""
    return importlib.import_module("field")


def _load_auxiliary_deepfield_field(model_path: Path | str):
    deepfield = _import_deepfield()
    field_module = importlib.import_module("field.field")
    Field = deepfield.Field
    default_config = field_module.default_config
    return Field(str(Path(model_path).resolve()), config=default_config, loglevel="ERROR").load(include_binary=False)


def _existing_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    try:
        resolved = path.resolve()
    except FileNotFoundError:
        resolved = path
    return resolved if resolved.exists() else None


def _path_variants(directory: Path, stem: str, suffix: str) -> list[Path]:
    suffixes = {suffix, suffix.lower()}
    stems = {stem, stem.lower(), stem.upper()}
    return [directory / f"{candidate_stem}{candidate_suffix}" for candidate_stem in stems for candidate_suffix in suffixes]


class MissingResultFileError(FileNotFoundError):
    r"""Raised when a required simulator result companion file is absent.

    The extractor reads geometry from the ``.EGRID`` file, static cell
    properties from the ``.INIT`` file, and dynamic state snapshots from the
    ``.UNRST`` (unified restart) file. When one of these is missing, the
    underlying ResInsight gRPC layer fails late with an opaque
    ``RipsError: No such result``; this exception is raised earlier instead,
    with the case basename, the directories searched, and a remediation hint.

    Subclasses :class:`FileNotFoundError` so existing ``except FileNotFoundError``
    handlers continue to catch it.
    """


def _require_result_file(
    path: Path | None,
    *,
    suffix: str,
    needed_for: str,
    source: "ReservoirSource",
    hint: str | None = None,
) -> Path:
    r"""Return ``path`` if it names an existing file, else raise a clear error.

    This centralises the "missing result file" guard so each loader can fail
    with an actionable message instead of letting a downstream library surface
    an opaque error.

    where:
     - :math:`path`: Resolved companion-file path, or ``None`` when the file
       could not be located during :func:`resolve_reservoir_source`.
     - :math:`suffix`: File extension reported in the message, e.g. ``".INIT"``.
     - :math:`needed\_for`: Human-readable description of what the file supplies
       (e.g. the property family that depends on it).
     - :math:`source`: The :class:`ReservoirSource` being read; supplies the case
       basename and the directories that were searched.
     - :math:`hint`: Optional remediation guidance appended to the message.
    """
    if path is not None and Path(path).is_file():
        return Path(path)
    searched = [str(source.deck_dir)]
    if source.results_dir is not None and source.results_dir != source.deck_dir:
        searched.append(str(source.results_dir))
    message = (
        f"Required {suffix} result file not found for case '{source.basename}' "
        f"(needed for {needed_for}). Searched: {searched}."
    )
    if hint:
        message = f"{message} {hint}"
    raise MissingResultFileError(message)


def _find_case_file(directories: list[Path], preferred_stems: list[str], suffix: str) -> Path | None:
    for directory in directories:
        if not directory.exists():
            continue
        for stem in preferred_stems:
            for candidate in _path_variants(directory, stem, suffix):
                existing = _existing_path(candidate)
                if existing is not None:
                    return existing
    upper_suffix = suffix.upper()
    lower_suffix = suffix.lower()
    for directory in directories:
        if not directory.exists():
            continue
        for candidate in sorted(directory.rglob(f"*{upper_suffix}")) + sorted(directory.rglob(f"*{lower_suffix}")):
            if candidate.is_file():
                return candidate.resolve()
    return None


def _tokenize_deck_line(line: str) -> list[str]:
    clean = line.split("--", 1)[0].strip()
    if not clean:
        return []
    return [token for token in re.split(r"\s+", clean.replace("/", " ")) if token]


def _parse_deck_start_date(model_path: Path) -> pd.Timestamp:
    if not model_path.is_file():
        return pd.NaT

    pending_start = False
    for raw_line in model_path.read_text(errors="ignore").splitlines():
        tokens = _tokenize_deck_line(raw_line)
        if not tokens:
            continue
        upper_tokens = [token.upper() for token in tokens]
        if pending_start:
            maybe = _parse_eclipse_date(" ".join(tokens[:3]))
            if pd.notna(maybe):
                return maybe
            pending_start = False
        if upper_tokens[0] != "START":
            continue
        if len(tokens) >= 4:
            maybe = _parse_eclipse_date(" ".join(tokens[1:4]))
            if pd.notna(maybe):
                return maybe
        pending_start = True
    return pd.NaT


@dataclass(frozen=True)
class ReservoirSource:
    data_path: Path
    deck_dir: Path
    basename: str
    egrid_path: Path
    init_path: Path | None = None
    unrst_path: Path | None = None
    esmry_path: Path | None = None
    smspec_path: Path | None = None
    summary_path: Path | None = None
    summary_format: str | None = None
    results_dir: Path | None = None
    start_date: pd.Timestamp | None = None


@dataclass
class GridGeometryPayload:
    """Reservoir grid geometry in native simulator coordinates.

    The extractor layer does not reinterpret the z axis. If the source stores
    depth increasing downward, downstream visualization should reverse the
    displayed z-axis instead of modifying these coordinates.
    """

    corner_cells: np.ndarray
    active_mask: np.ndarray
    active_cell_indices: np.ndarray
    cell_volumes: np.ndarray | None = None


@dataclass
class RockPayload:
    perms: np.ndarray
    poro: np.ndarray
    perm_attrs: tuple[str, str, str] = ("PERMX", "PERMY", "PERMZ")
    poro_attr: str = "PORO"


@dataclass
class StateTimelinePayload:
    snapshots: dict[str, np.ndarray]
    indices: np.ndarray
    report_steps: np.ndarray
    available_report_steps: np.ndarray
    dates: pd.Index
    n_times: int


@dataclass
class CompletionPayload:
    source: ReservoirSource
    meta: dict[str, Any]
    segments: dict[str, Any]


@dataclass
class SummaryPayload:
    results_by_well: dict[str, pd.DataFrame]
    dates: pd.DatetimeIndex
    addresses_by_well: dict[str, dict[str, str]]


@dataclass
class ReservoirPreprocessingArtifacts:
    source: ReservoirSource
    extractor_backend: str
    auxiliary_backend: str | None
    geometry: GridGeometryPayload
    reservoir_mesh: Any
    component_diagnostics: dict[str, Any]
    rock_payload: RockPayload
    rock_data: dict[str, Any]
    state_payload: StateTimelinePayload | None   # the raw per-step payload; dropped once snapshotted
    state_data: dict[str, Any]
    cell_states: dict[str, np.ndarray]
    vertex_states: dict[str, np.ndarray]
    completion_payload: CompletionPayload
    summary_payload: SummaryPayload
    well_metadata: dict[str, Any]
    aquifer_vertices: np.ndarray
    category_indices: dict[str, np.ndarray]
    blackoil_tables: dict[str, np.ndarray] | None
    timings: dict[str, float]
    # Unit system declared by the source deck ("FIELD"/"METRIC"/"LAB"). The cached arrays
    # are always normalized to FIELD units (see unit_conversion.to_field_units); this field
    # records the deck's original system for provenance.
    source_units: str = "FIELD"


@dataclass(frozen=True)
class PhysicalOutputRanges:
    r"""
    Physical output ranges for the black-oil primary-variable transforms, *derived* from a
    :class:`ReservoirPreprocessingArtifacts` bundle (deck SCAL/PVT/ROCK tables, the reference
    state timeline, and the well BHP limits) instead of hard-coded per case.

    The four network primaries :math:`(p_o, S_w, S_g, R_{so})` are mapped from the unit interval
    onto physical units through these anchors, so each anchor must bracket every attainable value
    of its primary without clipping the reference states.

    where:
    - :math:`P_{\min}, P_{\max}`: oil-pressure transform anchors [psia].
    - :math:`S_{wc}`: connate (irreducible) water saturation, the first SWOF row.
    - :math:`R_{so}^{\max}`: solution gas-oil ratio ceiling [Mscf/stb].
    - :math:`\phi_o`: reference porosity of the ROCK compaction law.
    - :math:`c_R, p_{\mathrm{ref}}`: rock compressibility [1/psi] and reference pressure [psia].
    - ``provenance``: per-field human-readable note on how each anchor was resolved.
    """

    P_MIN: float
    P_MAX: float
    SWC: float
    RS_MAX: float
    PHI0: float
    ROCK_CR: float
    ROCK_PREF: float
    provenance: dict[str, str]


def _well_bhp_limits(well_metadata: dict[str, Any]) -> tuple[float | None, float | None]:
    r"""
    Scan a well schedule for the injector BHP **upper** limit and producer BHP **lower** limit
    that bound the attainable reservoir pressure.

    A control entry is read as an injector when it carries a positive gas/water injection target
    (``control_git`` / ``control_wit``) or a positive rate, and as a producer when its rate is
    negative. The returned pair is

    .. math::

        \big(p_{\mathrm{bhp}}^{\mathrm{inj}},\, p_{\mathrm{bhp}}^{\mathrm{prod}}\big)
        = \Big(\max_i p^{\mathrm{inj}}_{\mathrm{bhp},i},\ \min_j p^{\mathrm{prod}}_{\mathrm{bhp},j}\Big),

    with ``None`` in either slot when no such limit is present.

    where:
    - :math:`p^{\mathrm{inj}}_{\mathrm{bhp},i}`: bottom-hole pressure cap of injector control :math:`i`.
    - :math:`p^{\mathrm{prod}}_{\mathrm{bhp},j}`: bottom-hole pressure floor of producer control :math:`j`.
    """

    def _num(value: Any) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if np.isfinite(out) else None

    injectors: list[float] = []
    producers: list[float] = []
    if not isinstance(well_metadata, dict):
        return None, None
    for step in well_metadata.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        for key in ("well_results", "rate_entries", "bhp_entries"):
            for entry in step.get(key) or []:
                if not isinstance(entry, dict):
                    continue
                bhp = _num(entry.get("control_bhpt"))
                if bhp is None:
                    continue
                git = _num(entry.get("control_git"))
                wit = _num(entry.get("control_wit"))
                rate = _num(entry.get("control_rate"))
                is_injector = (git is not None and git > 0) or (wit is not None and wit > 0) or (rate is not None and rate > 0)
                is_producer = rate is not None and rate < 0
                if is_injector:
                    injectors.append(bhp)
                elif is_producer:
                    producers.append(bhp)
    return (max(injectors) if injectors else None), (min(producers) if producers else None)


def derive_physical_output_ranges(
    artifacts: ReservoirPreprocessingArtifacts,
    *,
    p_pad: float = 100.0,
    p_margin_frac: float = 0.15,
    rs_step: float = 0.05,
    rs_headroom: int = 1,
    pressure_attr: str = "PRESSURE",
) -> PhysicalOutputRanges:
    r"""
    Derive the physical output ranges of the four black-oil primaries from a preprocessing-artifacts
    bundle, replacing per-case hard-coded constants.

    The reference-state pressure envelope is padded outward by a fraction of its own span (so the
    boundary states do not sit on the saturating tail of the output map), the **top** additionally
    unioned with the injector BHP cap (which injection can drive the field toward), and the whole
    quantised to a rounding pad :math:`\Delta p`. The producer BHP *floor* is intentionally **not**
    used as :math:`P_{\min}` because decks frequently carry a loose default there. The solution-GOR
    ceiling is quantised above the saturated PVTO nodes with a headroom reserve, and the SCAL/ROCK
    scalars are read straight from the parsed deck tables:

    .. math::

        P_{\min} = \Delta p \left\lfloor
            \frac{\max\!\big(0,\; \min_{c,t} p_{c,t} - f_m\,\Delta p_s\big)}{\Delta p}
        \right\rfloor,
        \qquad
        P_{\max} = \Delta p \left\lceil
            \frac{\max\!\big(\max_{c,t} p_{c,t} + f_m\,\Delta p_s,\; p_{\mathrm{bhp}}^{\mathrm{inj}}\big)}{\Delta p}
        \right\rceil,

    .. math::

        R_{so}^{\max} = \Delta R \left(
            \left\lceil \frac{\max_j R_{so,j}^{\mathrm{sat}}}{\Delta R} \right\rceil + n_h
        \right),
        \qquad
        S_{wc} = S_w^{(1)}\big|_{\mathrm{SWOF}},
        \qquad
        (\phi_o, c_R, p_{\mathrm{ref}}) = \big(\overline{\phi_c},\, c_R,\, p_{\mathrm{ref}}\big)\big|_{\mathrm{ROCK}}.

    where:
    - :math:`p_{c,t}`: reference oil pressure at cell :math:`c` and report step :math:`t` (state ``PRESSURE``); :math:`\Delta p_s=\max_{c,t}p_{c,t}-\min_{c,t}p_{c,t}` is its span.
    - :math:`f_m`: pressure headroom fraction ``p_margin_frac``; :math:`p_{\mathrm{bhp}}^{\mathrm{inj}}`: injector BHP cap from the schedule (dropped from the max when absent).
    - :math:`\Delta p`: pressure rounding pad ``p_pad`` [psia]; :math:`\Delta R`: GOR quantum ``rs_step`` [Mscf/stb]; :math:`n_h`: ``rs_headroom`` extra quanta.
    - :math:`R_{so,j}^{\mathrm{sat}}`: saturated solution-GOR nodes from PVTO (``pvto_rs_sat``).
    - :math:`S_w^{(1)}|_{\mathrm{SWOF}}`: first (connate) water-saturation row of SWOF (``swof_sw`` / ``swc_baker``).
    - :math:`\overline{\phi_c}`: mean active-cell porosity (``rock_payload.poro``); :math:`c_R, p_{\mathrm{ref}}`: ROCK compressibility/reference pressure (``rock_compr`` / ``rock_pref``).

    :param artifacts: Bundle from :func:`build_reservoir_preprocessing_artifacts`; must carry
        ``blackoil_tables`` (built with ``include_blackoil_tables=True``).
    :param p_pad: Rounding pad :math:`\Delta p` applied to the pressure anchors [psia].
    :param p_margin_frac: Headroom fraction :math:`f_m` of the state pressure span added to each end.
    :param rs_step: Quantum :math:`\Delta R` for the solution-GOR ceiling [Mscf/stb].
    :param rs_headroom: Extra :math:`\Delta R` quanta :math:`n_h` reserved above saturated PVTO.
    :param pressure_attr: State key holding the reference pressure field.
    :returns: A :class:`PhysicalOutputRanges` carrying the seven anchors and a provenance map.
    """
    tables = artifacts.blackoil_tables
    if not tables:
        raise ValueError(
            "derive_physical_output_ranges requires artifacts.blackoil_tables; rebuild the "
            "preprocessing cache with include_blackoil_tables=True."
        )
    provenance: dict[str, str] = {}

    def _scalar(value: Any) -> float:
        return float(np.asarray(value, dtype=float).reshape(-1)[0])

    # --- connate water saturation (SWOF endpoint / Baker fallback) --------------------------
    swof_sw = np.asarray(tables.get("swof_sw", ()), dtype=float)
    if swof_sw.size:
        swc = float(swof_sw[0])
        provenance["SWC"] = f"SWOF sw[0] = {swc:.4g}"
    else:
        swc = _scalar(tables["swc_baker"])
        provenance["SWC"] = f"swc_baker = {swc:.4g}"

    # --- solution-GOR ceiling above the saturated PVTO nodes --------------------------------
    rs_top = float(np.nanmax(np.asarray(tables["pvto_rs_sat"], dtype=float)))
    rs_max = round((int(np.ceil(rs_top / rs_step)) + int(rs_headroom)) * rs_step, 6)
    provenance["RS_MAX"] = f"ceil(PVTO rs_sat max {rs_top:.4g} / {rs_step:g}) + {int(rs_headroom)} -> {rs_max:.4g}"

    # --- ROCK compaction scalars ------------------------------------------------------------
    rock_cr = _scalar(tables["rock_compr"])
    rock_pref = _scalar(tables["rock_pref"])
    provenance["ROCK_CR"] = f"ROCK c_R = {rock_cr:.4g}"
    provenance["ROCK_PREF"] = f"ROCK p_ref = {rock_pref:.4g}"

    # --- reference porosity (mean active-cell PORO) -----------------------------------------
    poro = np.asarray(artifacts.rock_payload.poro, dtype=float)
    phi0 = float(np.nanmean(poro))
    provenance["PHI0"] = f"mean(PORO) = {phi0:.4g} (min {np.nanmin(poro):.4g}, max {np.nanmax(poro):.4g})"

    # --- pressure anchors: state envelope padded by a span-fraction, top unioned with inj BHP ---
    pressure = np.asarray(artifacts.state_data[pressure_attr], dtype=float)
    p_lo_state, p_hi_state = float(np.nanmin(pressure)), float(np.nanmax(pressure))
    span = max(p_hi_state - p_lo_state, 0.0)
    bhp_inj, _bhp_prod = _well_bhp_limits(artifacts.well_metadata)
    lo_source = max(0.0, p_lo_state - p_margin_frac * span)
    hi_source = p_hi_state + p_margin_frac * span
    if bhp_inj is not None:
        hi_source = max(hi_source, bhp_inj)
    p_min = float(np.floor(lo_source / p_pad) * p_pad)
    p_max = float(np.ceil(hi_source / p_pad) * p_pad)
    provenance["P_MIN"] = f"floor(max(0, state_min {p_lo_state:.0f} - {p_margin_frac:g}*span {span:.0f}) / {p_pad:g}) -> {p_min:.0f}"
    provenance["P_MAX"] = (
        f"ceil(max(state_max {p_hi_state:.0f} + {p_margin_frac:g}*span {span:.0f}"
        + (f", inj BHP {bhp_inj:.0f}" if bhp_inj is not None else "")
        + f") / {p_pad:g}) -> {p_max:.0f}"
    )

    return PhysicalOutputRanges(
        P_MIN=p_min,
        P_MAX=p_max,
        SWC=swc,
        RS_MAX=rs_max,
        PHI0=phi0,
        ROCK_CR=rock_cr,
        ROCK_PREF=rock_pref,
        provenance=provenance,
    )


def resolve_reservoir_source(model_path: str | Path) -> ReservoirSource:
    data_path = Path(model_path).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Reservoir model path does not exist: {data_path}")

    deck_dir = data_path.parent
    results_dir = next((path.resolve() for path in [deck_dir / "RESULTS", deck_dir / "results"] if path.is_dir()), None)
    preferred_stems = []
    if data_path.stem:
        preferred_stems.append(data_path.stem)
    basename_no_ext = data_path.name.split(".", 1)[0]
    if basename_no_ext and basename_no_ext not in preferred_stems:
        preferred_stems.append(basename_no_ext)

    search_dirs = [deck_dir]
    if results_dir is not None:
        search_dirs.append(results_dir)

    egrid_path = _find_case_file(search_dirs, preferred_stems, ".EGRID")
    if egrid_path is None:
        raise MissingResultFileError(
            f"Could not resolve an .EGRID grid companion for deck '{data_path}'. "
            f"Searched directories: {[str(path) for path in search_dirs]}. "
            "The simulator may not have been run yet, or its output lives outside "
            "the deck directory and a 'RESULTS' subfolder."
        )

    basename = egrid_path.stem
    preferred_stems = [basename] + [stem for stem in preferred_stems if stem != basename]
    init_path = _find_case_file(search_dirs, preferred_stems, ".INIT")
    unrst_path = _find_case_file(search_dirs, preferred_stems, ".UNRST")
    esmry_path = _find_case_file(search_dirs, preferred_stems, ".ESMRY")
    smspec_path = _find_case_file(search_dirs, preferred_stems, ".SMSPEC")
    if esmry_path is not None:
        summary_path = esmry_path
        summary_format = "ESMRY"
    elif smspec_path is not None:
        summary_path = smspec_path
        summary_format = "SMSPEC"
    else:
        summary_path = None
        summary_format = None
    start_date = _parse_deck_start_date(data_path)

    return ReservoirSource(
        data_path=data_path,
        deck_dir=deck_dir,
        basename=basename,
        egrid_path=egrid_path,
        init_path=init_path,
        unrst_path=unrst_path,
        esmry_path=esmry_path,
        smspec_path=smspec_path,
        summary_path=summary_path,
        summary_format=summary_format,
        results_dir=results_dir,
        start_date=None if pd.isna(start_date) else pd.to_datetime(start_date),
    )


class ResInsightSessionManager:
    """Attach to a running ResInsight instance or launch one on demand."""

    def __init__(self, lifecycle: str = "attach_or_launch", console: bool = False):
        self.lifecycle = str(lifecycle)
        self.console = bool(console)
        self._instance = None
        self._launched_here = False

    def acquire(self):
        rips = _try_import("rips")
        if rips is None:
            raise ImportError("The 'rips' package is required for the ResInsight extractor backend.")

        instance = None
        if self.lifecycle == "attach":
            instance = rips.Instance.find()
        elif self.lifecycle == "attach_or_launch":
            try:
                instance = rips.Instance.find()
            except rips.RipsError:
                instance = None
        if instance is None and self.lifecycle in {"launch", "attach_or_launch"}:
            instance = rips.Instance.launch(console=self.console)
            self._launched_here = instance is not None
        if instance is None:
            raise RuntimeError(
                "Could not connect to ResInsight. "
                "Either start a ResInsight instance or set RESINSIGHT_EXECUTABLE so it can be launched."
            )
        self._instance = instance
        return instance

    def close(self) -> None:
        if self._instance is None:
            return
        if self._launched_here:
            try:
                self._instance.exit()
            except Exception:
                pass
        self._instance = None
        self._launched_here = False


class ReservoirExtractorBackend:
    backend_name = "base"

    def __init__(self, source: ReservoirSource):
        self.source = source

    def close(self) -> None:  # pragma: no cover - overridden by concrete backends
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _normalize_ijk_array(indices: np.ndarray, dims: tuple[int, int, int]) -> np.ndarray:
    indices = np.asarray(indices, dtype=int).reshape(-1, 3)
    dims_arr = np.asarray(dims, dtype=int)
    if len(indices) == 0:
        return indices
    if np.all(indices >= 1) and np.all(indices <= dims_arr[None, :]):
        indices = indices - 1
    if np.any(indices < 0) or np.any(indices >= dims_arr[None, :]):
        raise ValueError(
            f"Encountered IJK indices outside grid bounds {tuple(int(v) for v in dims_arr.tolist())}: "
            f"min={tuple(indices.min(axis=0).tolist())}, max={tuple(indices.max(axis=0).tolist())}."
        )
    return indices.astype(int)


def _vec3_to_array(vec: Any) -> np.ndarray:
    return np.asarray([float(vec.x), float(vec.y), float(vec.z)], dtype=float)


def _cell_corners_to_array(cell_corners: Any) -> np.ndarray:
    return np.asarray([_vec3_to_array(getattr(cell_corners, f"c{idx}")) for idx in range(8)], dtype=float)


def _coerce_summary_dates(values) -> pd.DatetimeIndex:
    raw_values = getattr(values, "values", values)
    try:
        raw = np.asarray(list(raw_values), dtype=np.int64)
    except (TypeError, ValueError):
        try:
            converted = pd.to_datetime(raw_values, errors="coerce")
        except Exception:
            return pd.DatetimeIndex([])
        if converted.isna().any():
            return pd.DatetimeIndex([])
        candidate = pd.DatetimeIndex(converted)
        return candidate.tz_localize(None) if candidate.tz is not None else candidate
    if raw.size == 0:
        return pd.DatetimeIndex([])
    candidates = []
    for order, kwargs in enumerate(({"unit": "s"}, {"unit": "ms"}, {})):
        try:
            converted = pd.to_datetime(raw, errors="coerce", **kwargs)
        except Exception:
            continue
        if converted.isna().any():
            continue
        candidate = pd.DatetimeIndex(converted)
        if candidate.tz is not None:
            candidate = candidate.tz_localize(None)
        years = candidate.year
        if not len(years) or int(years.min()) < 1900 or int(years.max()) > 2200:
            continue
        score = 0
        if int(years.min()) >= 1980:
            score += 10
        if candidate.is_monotonic_increasing:
            score += 1
        span_days = (candidate.max() - candidate.min()).total_seconds() / 86400.0
        if span_days > 1.0:
            score += 1
        if int(years.max()) == 1970 and candidate.max() < pd.Timestamp("1970-03-01"):
            score -= 10
        candidates.append((score, order, candidate))
    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][2]
    return pd.DatetimeIndex([])


def _time_steps_to_dates(time_steps: list[Any]) -> pd.DatetimeIndex:
    dates: list[pd.Timestamp] = []
    for step in time_steps:
        try:
            timestamp = pd.Timestamp(
                year=int(getattr(step, "year")),
                month=max(1, int(getattr(step, "month", 1))),
                day=max(1, int(getattr(step, "day", 1))),
                hour=int(getattr(step, "hour", 0)),
                minute=int(getattr(step, "minute", 0)),
                second=int(getattr(step, "second", 0)),
            )
        except Exception:
            timestamp = pd.NaT
        dates.append(timestamp)
    if not dates:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(dates))


def _series_value(series, idx: int) -> float:
    """One float out of a pandas Series or a numpy array (NaN when it is missing)."""
    try:
        value = series.iloc[idx] if hasattr(series, "iloc") else series[idx]
    except (IndexError, KeyError):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _default_blocks_info_frame(n_blocks: int) -> pd.DataFrame:
    n_blocks = int(max(n_blocks, 0))
    return pd.DataFrame(
        {
            "SKIN": np.zeros((n_blocks,), dtype=float),
            "PERF_RATIO": np.ones((n_blocks,), dtype=float),
            "CF": np.full((n_blocks,), np.nan, dtype=float),    # NaN = no explicit connection factor
            "RAD": np.full((n_blocks,), np.nan, dtype=float),
            "MULT": np.ones((n_blocks,), dtype=float),
        }
    )


def _score_summary_address(address: str, vector: str, well_name: str) -> tuple[int, int]:
    upper = address.upper()
    vector = vector.upper()
    well_name = well_name.upper()
    tokens = [token for token in re.split(r"[^A-Z0-9_]+", upper) if token]
    score = 0
    if vector in tokens:
        score += 20
    elif vector in upper:
        score += 8
    if well_name in tokens:
        score += 12
    elif well_name in upper:
        score += 6
    if upper.startswith(vector):
        score += 3
    if upper.endswith(well_name):
        score += 3
    return score, -len(address)


def _resolve_summary_address(available_addresses: list[str], vector: str, well_name: str) -> str | None:
    ranked = [
        (_score_summary_address(address, vector, well_name), address)
        for address in available_addresses
        if vector.upper() in address.upper() and well_name.upper() in address.upper()
    ]
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, best_address = ranked[0]
    return best_address if best_score[0] > 0 else None


def _results_frame_from_summary_case(
    summary_case: Any,
    well_names: list[str],
    summary_vectors: tuple[str, ...] = SUMMARY_VECTORS_DEFAULT,
) -> SummaryPayload:
    available_addresses = list(getattr(summary_case.available_addresses(), "values", []))
    dates = _coerce_summary_dates(summary_case.available_time_steps())
    results_by_well: dict[str, pd.DataFrame] = {}
    addresses_by_well: dict[str, dict[str, str]] = {}
    for well_name in well_names:
        frame = pd.DataFrame()
        if len(dates):
            frame["DATE"] = pd.to_datetime(dates)
        addresses: dict[str, str] = {}
        for vector in summary_vectors:
            address = _resolve_summary_address(available_addresses, vector, well_name)
            if address is None:
                continue
            addresses[vector] = address
            values = np.asarray(summary_case.summary_vector_values(address).values, dtype=float)
            if len(frame) == 0 and len(values):
                frame["DATE"] = np.arange(len(values), dtype=int)
            if len(values) == len(frame):
                frame[vector] = values
        if len(frame):
            results_by_well[str(well_name)] = frame
        if addresses:
            addresses_by_well[str(well_name)] = addresses
    return SummaryPayload(results_by_well=results_by_well, dates=dates, addresses_by_well=addresses_by_well)


def _find_matching_summary_case(project: Any, summary_path: Path | None):
    if summary_path is None or project is None or not hasattr(project, "summary_cases"):
        return None
    resolved_target = Path(summary_path).resolve()
    try:
        summary_cases = list(project.summary_cases())
    except Exception:
        return None
    for summary_case in summary_cases:
        header = getattr(summary_case, "summary_header_filename", None)
        if not header:
            continue
        try:
            resolved_header = Path(header).resolve()
        except Exception:
            resolved_header = Path(str(header))
        if resolved_header == resolved_target:
            return summary_case
    return None


def _collect_well_vertex_ids_from_metadata(well_metadata: dict[str, Any]) -> np.ndarray:
    vertex_ids: set[int] = set()
    for step in well_metadata.get("steps", []):
        for key in ("bhp_entries", "rate_entries"):
            for entry in step.get(key, []):
                vertex_ids.update(int(v) for v in np.asarray(entry.get("cell_vertices", []), dtype=int).tolist())
    return np.asarray(sorted(vertex_ids), dtype=np.int32)


def _build_segments_from_simulator_tables(table: Any, dims: tuple[int, int, int]) -> dict[str, Any]:
    wells: dict[str, dict[str, Any]] = {}
    for entry in getattr(table, "compdat", []):
        well_name = str(entry.well_name).upper()
        rows = wells.setdefault(
            well_name,
            {
                "blocks": [],
                "skin": [],
                "perf_ratio": [],
                "cf": [],
                "rad": [],
                "mult": [],
                "welspecs": [],
            },
        )
        for k in range(int(entry.upper_k), int(entry.lower_k) + 1):
            rows["blocks"].append([int(entry.grid_i), int(entry.grid_j), int(k)])
            skin = float(getattr(entry, "skin_factor", np.nan))
            trans = float(getattr(entry, "transmissibility", np.nan))
            diameter = float(getattr(entry, "diameter", np.nan))
            rows["skin"].append(skin if np.isfinite(skin) else 0.0)
            rows["perf_ratio"].append(1.0)
            rows["cf"].append(trans if np.isfinite(trans) and trans > 0.0 else 1.0)
            rows["rad"].append(0.5 * diameter if np.isfinite(diameter) and diameter > 0.0 else np.nan)
            rows["mult"].append(1.0)

    for entry in getattr(table, "welspecs", []):
        well_name = str(entry.well_name).upper()
        rows = wells.setdefault(
            well_name,
            {
                "blocks": [],
                "skin": [],
                "perf_ratio": [],
                "cf": [],
                "rad": [],
                "mult": [],
                "welspecs": [],
            },
        )
        rows["welspecs"].append(
            {
                "WELL": well_name,
                "GROUP": str(entry.group_name).upper(),
                "I": int(entry.grid_i),
                "J": int(entry.grid_j),
                "PHASE": str(entry.phase).upper(),
                "DREF": float(getattr(entry, "bhp_depth", np.nan)),
            }
        )

    segments: dict[str, Any] = {}
    for well_name, raw in wells.items():
        block_indices = _normalize_ijk_array(np.asarray(raw["blocks"], dtype=int), dims) if raw["blocks"] else np.zeros((0, 3), dtype=int)
        n_blocks = len(block_indices)
        blocks_info = _default_blocks_info_frame(n_blocks)
        if n_blocks:
            blocks_info["SKIN"] = np.asarray(raw["skin"][:n_blocks], dtype=float)
            blocks_info["PERF_RATIO"] = np.asarray(raw["perf_ratio"][:n_blocks], dtype=float)
            blocks_info["CF"] = np.asarray(raw["cf"][:n_blocks], dtype=float)
            blocks_info["RAD"] = np.asarray(raw["rad"][:n_blocks], dtype=float)
            blocks_info["MULT"] = np.asarray(raw["mult"][:n_blocks], dtype=float)
        segments[well_name] = SimpleNamespace(
            name=well_name,
            blocks=block_indices,
            blocks_info=blocks_info,
            welltrack=None,
            welspecs=pd.DataFrame(raw["welspecs"]).reset_index(drop=True) if raw["welspecs"] else pd.DataFrame(columns=["DREF"]),
            events=None,
            wconprod=None,
            wconinje=None,
            wconhist=None,
            welopen=None,
            results=None,
        )
    return segments


def _simulation_timelines_from_case(
    case: Any,
    dims: tuple[int, int, int],
    dates: pd.DatetimeIndex,
) -> dict[str, list[dict[str, Any]]]:
    timelines: dict[str, list[dict[str, Any]]] = {}
    try:
        simulation_wells = list(case.simulation_wells())
    except Exception:
        return timelines

    for simulation_well in simulation_wells:
        well_name = str(getattr(simulation_well, "name", "")).upper().strip()
        if not well_name:
            continue
        entries: list[dict[str, Any]] = []
        for step_idx in range(len(dates)):
            try:
                status_rows = list(simulation_well.status(int(step_idx)))
            except Exception:
                status_rows = []
            try:
                cell_rows = list(simulation_well.cells(int(step_idx)))
            except Exception:
                cell_rows = []

            well_type = next(
                (
                    str(getattr(status, "well_type", "")).upper()
                    for status in status_rows
                    if str(getattr(status, "well_type", "")).strip()
                ),
                "",
            )
            all_blocks_raw: list[list[int]] = []
            open_blocks_raw: list[list[int]] = []
            for cell in cell_rows:
                ijk = getattr(cell, "ijk", None)
                if ijk is None:
                    continue
                block = [
                    int(getattr(ijk, "i")),
                    int(getattr(ijk, "j")),
                    int(getattr(ijk, "k")),
                ]
                all_blocks_raw.append(block)
                if bool(getattr(cell, "is_open", False)):
                    open_blocks_raw.append(block)

            all_blocks = (
                _normalize_ijk_array(np.asarray(all_blocks_raw, dtype=int), dims)
                if all_blocks_raw
                else np.zeros((0, 3), dtype=int)
            )
            open_blocks = (
                _normalize_ijk_array(np.asarray(open_blocks_raw, dtype=int), dims)
                if open_blocks_raw
                else np.zeros((0, 3), dtype=int)
            )
            is_open = bool(
                any(bool(getattr(status, "is_open", False)) for status in status_rows)
                or len(open_blocks) > 0
            )
            entries.append(
                {
                    "DATE": pd.to_datetime(dates[int(step_idx)]),
                    "TIME_STEP": int(step_idx),
                    "WELL_TYPE": well_type,
                    "IS_OPEN": is_open,
                    "ALL_BLOCKS": all_blocks,
                    "OPEN_BLOCKS": open_blocks,
                }
            )
        if entries:
            timelines[well_name] = entries
    return timelines


def _segments_from_simulation_timelines(
    timelines: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    segments: dict[str, Any] = {}
    for well_name, entries in timelines.items():
        ordered_blocks: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        for entry in entries:
            for column in ("ALL_BLOCKS", "OPEN_BLOCKS"):
                for block in np.asarray(entry.get(column, np.zeros((0, 3), dtype=int)), dtype=int).reshape(-1, 3).tolist():
                    key = tuple(int(v) for v in block)
                    if key in seen:
                        continue
                    seen.add(key)
                    ordered_blocks.append(key)
        block_indices = (
            np.asarray(ordered_blocks, dtype=int).reshape(-1, 3)
            if ordered_blocks
            else np.zeros((0, 3), dtype=int)
        )
        segments[str(well_name).upper()] = SimpleNamespace(
            name=str(well_name).upper(),
            blocks=block_indices,
            blocks_info=_default_blocks_info_frame(len(block_indices)),
            welltrack=None,
            welspecs=pd.DataFrame(columns=["DREF"]),
            events=None,
            wconprod=None,
            wconinje=None,
            wconhist=None,
            welopen=None,
            results=None,
        )
    return segments


def deck_three_phase_oil_model(model_path: Path | str) -> str:
    """The three-phase oil relative-permeability model a deck declares in PROPS —
    ``"stone1"`` for ``STONE1`` (or the ``STONE`` alias), ``"stone2"`` for ``STONE2``,
    ``"default"`` when neither appears (ECLIPSE's segregated default model). The main
    file and its INCLUDEs are scanned as keyword lines (comments stripped)."""
    root = Path(model_path)
    found = "default"
    for path in [root] + _collect_deck_include_paths(root):
        try:
            text = Path(path).read_text(errors="ignore")
        except OSError:
            continue
        for raw_line in text.splitlines():
            token = _strip_schedule_comment(raw_line).strip().upper()
            if token in ("STONE1", "STONE"):
                return "stone1"
            if token == "STONE2":
                found = "stone2"
    return found


def _collect_deck_include_paths(model_path: Path) -> list[Path]:
    """Collect INCLUDE files recursively in deck order."""
    resolved_root = Path(model_path).resolve()
    include_paths: list[Path] = []
    seen_files: set[Path] = set()
    seen_includes: set[Path] = set()
    allowed_suffixes = {"", ".DATA", ".INC", ".SCH", ".BASE"}

    def visit(path: Path) -> None:
        resolved = Path(path).resolve()
        if resolved in seen_files or not resolved.is_file():
            return
        seen_files.add(resolved)
        pending_include = False
        for raw_line in resolved.read_text(errors="ignore").splitlines():
            line = _strip_schedule_comment(raw_line)
            if not line:
                continue
            upper = line.upper()
            if pending_include:
                include_rel = _extract_include_path(line)
                if include_rel is None:
                    pending_include = False
                    continue
                include_path = (resolved.parent / include_rel).resolve()
                if include_path.suffix.upper() not in allowed_suffixes:
                    pending_include = False
                    continue
                if include_path not in seen_includes:
                    include_paths.append(include_path)
                    seen_includes.add(include_path)
                visit(include_path)
                pending_include = False
                continue
            if upper.startswith("INCLUDE"):
                include_rel = _extract_include_path(line)
                if include_rel is None:
                    pending_include = True
                    continue
                include_path = (resolved.parent / include_rel).resolve()
                if include_path.suffix.upper() not in allowed_suffixes:
                    continue
                if include_path not in seen_includes:
                    include_paths.append(include_path)
                    seen_includes.add(include_path)
                visit(include_path)

    visit(resolved_root)
    return include_paths


def _parse_welspecs_row(line: str, current_date: pd.Timestamp) -> dict[str, Any] | None:
    tokens = _tokenize_schedule_line(line)
    if len(tokens) < 4:
        return None
    values = _expand_eclipse_tokens(tokens[2:])
    if len(values) < 2 or values[0] is None or values[1] is None:
        return None
    try:
        grid_i = int(values[0])
        grid_j = int(values[1])
    except (TypeError, ValueError):
        return None
    return {
        "DATE": pd.to_datetime(current_date),
        "WELL": str(tokens[0]).upper(),
        "GROUP": str(tokens[1]).upper(),
        "I": grid_i,
        "J": grid_j,
        "DREF": _parse_optional_float(values[2] if len(values) > 2 else None),
        "PHASE": str(values[3]).upper() if len(values) > 3 and values[3] is not None else "",
    }


def _parse_compdat_row(line: str, current_date: pd.Timestamp) -> dict[str, Any] | None:
    tokens = _tokenize_schedule_line(line)
    if len(tokens) < 6:
        return None
    try:
        grid_i = int(tokens[1])
        grid_j = int(tokens[2])
        upper_k = int(tokens[3])
        lower_k = int(tokens[4])
    except (TypeError, ValueError):
        return None
    if lower_k < upper_k:
        upper_k, lower_k = lower_k, upper_k
    values = _expand_eclipse_tokens(tokens[6:])
    status = str(tokens[5]).upper()
    rows: list[dict[str, Any]] = []
    for k_value in range(upper_k, lower_k + 1):
        rows.append(
            {
                "DATE": pd.to_datetime(current_date),
                "WELL": str(tokens[0]).upper(),
                "I": int(grid_i),
                "J": int(grid_j),
                "K": int(k_value),
                "STATUS": status,
                "CF": _parse_optional_float(values[1] if len(values) > 1 else None),
                "DIAM": _parse_optional_float(values[2] if len(values) > 2 else None),
                "KH": _parse_optional_float(values[3] if len(values) > 3 else None),
                "SKIN": _parse_optional_float(values[4] if len(values) > 4 else None),
                "DIR": str(values[6]).upper() if len(values) > 6 and values[6] is not None else "",
                "RO": _parse_optional_float(values[7] if len(values) > 7 else None),
                "MULT": 1.0,
                "PERF_RATIO": 1.0,
            }
        )
    return rows[0] if len(rows) == 1 else {"ROWS": rows}


def _merge_completion_table_cache(
    target: dict[str, dict[str, list[dict[str, Any]]]],
    source: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    for well_name, tables in source.items():
        merged = target.setdefault(well_name, {"welspecs": [], "compdat": []})
        for table_name, rows in tables.items():
            merged.setdefault(table_name, []).extend(rows)


def _parse_completion_tables_file(schedule_path: Path, start_date: pd.Timestamp) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if not schedule_path.is_file():
        return {}

    tables: dict[str, dict[str, list[dict[str, Any]]]] = {}
    current_date = pd.to_datetime(start_date)
    active_block = ""
    for raw_line in schedule_path.read_text(errors="ignore").splitlines():
        line = _strip_schedule_comment(raw_line)
        if not line:
            continue
        upper = line.upper()

        if active_block == "DATES":
            parsed_date = _parse_dates_line(line)
            if pd.notna(parsed_date):
                current_date = parsed_date
            if "/" in raw_line:
                active_block = ""
            continue

        if active_block == "TSTEP":
            increment_days = _parse_tstep_increment(line)
            if np.isfinite(increment_days) and increment_days != 0.0 and pd.notna(current_date):
                current_date = pd.to_datetime(current_date) + pd.to_timedelta(increment_days, unit="D")
            if "/" in raw_line:
                active_block = ""
            continue

        if upper == "/":
            active_block = ""
            continue

        if upper.startswith("DATES"):
            active_block = "DATES"
            parsed_date = _parse_dates_line(line)
            if pd.notna(parsed_date):
                current_date = parsed_date
            if "/" in raw_line and line.upper() != "DATES":
                active_block = ""
            continue

        if upper.startswith("TSTEP"):
            active_block = "TSTEP"
            increment_days = _parse_tstep_increment(line)
            if np.isfinite(increment_days) and increment_days != 0.0 and pd.notna(current_date):
                current_date = pd.to_datetime(current_date) + pd.to_timedelta(increment_days, unit="D")
            if "/" in raw_line and line.upper() != "TSTEP":
                active_block = ""
            continue

        if upper.startswith("WELSPECS"):
            active_block = "WELSPECS"
            continue
        if upper.startswith("COMPDAT"):
            active_block = "COMPDAT"
            continue

        if active_block == "WELSPECS":
            row = _parse_welspecs_row(line, current_date)
            if row is None:
                continue
            tables.setdefault(row["WELL"], {"welspecs": [], "compdat": []})["welspecs"].append(row)
            continue

        if active_block == "COMPDAT":
            row = _parse_compdat_row(line, current_date)
            if row is None:
                continue
            if "ROWS" in row:
                rows = row["ROWS"]
            else:
                rows = [row]
            well_name = str(rows[0]["WELL"]).upper()
            tables.setdefault(well_name, {"welspecs": [], "compdat": []})["compdat"].extend(rows)

    return tables


def _build_completion_table_cache(source: ReservoirSource) -> dict[str, dict[str, pd.DataFrame]]:
    start_date = pd.NaT
    if source.start_date is not None:
        start_date = pd.to_datetime(source.start_date)

    parsed_tables: dict[str, dict[str, list[dict[str, Any]]]] = {}
    completion_sources = [source.data_path] + _collect_deck_include_paths(source.data_path)
    seen_sources: set[Path] = set()
    for completion_path in completion_sources:
        completion_path = Path(completion_path).resolve()
        if completion_path in seen_sources:
            continue
        seen_sources.add(completion_path)
        _merge_completion_table_cache(parsed_tables, _parse_completion_tables_file(completion_path, start_date))

    cache: dict[str, dict[str, pd.DataFrame]] = {}
    for well_name, tables in parsed_tables.items():
        cache[str(well_name).upper()] = {}
        for table_name, rows in tables.items():
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            if "DATE" in frame.columns:
                frame = frame.sort_values("DATE").reset_index(drop=True)
            cache[str(well_name).upper()][table_name] = frame
    return cache


def _blocks_info_from_compdat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    blocks_info = _default_blocks_info_frame(len(frame))
    if len(frame) == 0:
        return blocks_info
    for column, target_name in (
        ("SKIN", "SKIN"),
        ("PERF_RATIO", "PERF_RATIO"),
        ("CF", "CF"),
        ("DIAM", "RAD"),
        ("MULT", "MULT"),
    ):
        if column not in frame:
            continue
        values = np.asarray(frame[column], dtype=float)
        if target_name == "RAD":
            values = np.where(np.isfinite(values), 0.5 * values, np.nan)
        blocks_info[target_name] = values
    return blocks_info


def _build_segments_from_completion_tables(
    tables_by_well: dict[str, dict[str, pd.DataFrame]],
    dims: tuple[int, int, int],
) -> dict[str, Any]:
    segments: dict[str, Any] = {}
    for well_name, tables in tables_by_well.items():
        compdat = tables.get("compdat")
        welspecs = tables.get("welspecs")
        block_indices = np.zeros((0, 3), dtype=int)
        blocks_info = _default_blocks_info_frame(0)
        if compdat is not None and len(compdat):
            latest_by_block: dict[tuple[int, int, int], pd.Series] = {}
            for _, row in compdat.sort_values("DATE").iterrows():
                key = (int(row["I"]), int(row["J"]), int(row["K"]))
                latest_by_block[key] = row
            latest_rows = pd.DataFrame(list(latest_by_block.values()))
            block_indices = _normalize_ijk_array(
                latest_rows[["I", "J", "K"]].to_numpy(dtype=int),
                dims,
            )
            blocks_info = _blocks_info_from_compdat_frame(latest_rows.reset_index(drop=True))
        segments[str(well_name).upper()] = SimpleNamespace(
            name=str(well_name).upper(),
            blocks=block_indices,
            blocks_info=blocks_info,
            welltrack=None,
            welspecs=welspecs.copy() if welspecs is not None else pd.DataFrame(columns=["DREF"]),
            events=None,
            wconprod=None,
            wconinje=None,
            wconhist=None,
            welopen=None,
            results=None,
        )
    return segments


class RipsExtractorBackend(ReservoirExtractorBackend):
    backend_name = "rips"

    def __init__(self, source: ReservoirSource, lifecycle: str = "attach_or_launch"):
        super().__init__(source)
        self._session = ResInsightSessionManager(lifecycle=lifecycle)
        self._instance = None
        self._rips = _try_import("rips")
        if self._rips is None:
            raise ImportError("The 'rips' package is required for the ResInsight extractor backend.")
        self._case = None
        self._summary_case = None
        self._summary_import_attempted = False

    def _ensure_instance(self):
        if self._instance is None:
            self._instance = self._session.acquire()
        return self._instance

    # A ResInsight instance outlives the extractor (``attach_or_launch`` reuses whatever is
    # running), and it never re-reads a result file it has already loaded. Reusing a grid or
    # summary case found in its project therefore serves the values of whichever simulator
    # run was current when that case was first opened -- a rerun of the simulator with new
    # controls is invisible until the stale case is dropped. Every extractor session hence
    # evicts the matching cases before loading its own, and removes its own on close.

    def _evict_matching_cases(self, instance) -> int:
        target = Path(self.source.egrid_path).resolve()
        evicted = 0
        try:
            cases = list(instance.project.cases())
        except Exception:
            return 0
        for case in cases:
            path = getattr(case, "file_path", None)
            if not path:
                continue
            try:
                same = Path(str(path)).resolve() == target
            except Exception:
                same = str(path) == str(target)
            if same:
                try:
                    case.delete()
                    evicted += 1
                except Exception:
                    pass
        return evicted

    def _evict_matching_summary_cases(self, instance) -> int:
        stem = Path(self.source.summary_path).stem.upper() if self.source.summary_path else None
        if stem is None:
            return 0
        evicted = 0
        try:
            summary_cases = list(instance.project.summary_cases())
        except Exception:
            return 0
        for summary_case in summary_cases:
            header = getattr(summary_case, "summary_header_filename", None)
            if not header:
                continue
            header_path = Path(str(header))
            # SMSPEC and ESMRY of the same run are the same data: evict both spellings
            if header_path.stem.upper() == stem and header_path.parent.resolve() == Path(self.source.summary_path).parent.resolve():
                try:
                    summary_case.delete()
                    evicted += 1
                except Exception:
                    pass
        return evicted

    def _ensure_case(self):
        if self._case is not None:
            return self._case
        instance = self._ensure_instance()
        evicted = self._evict_matching_cases(instance)
        if evicted:
            print(f"[rips] evicted {evicted} previously loaded grid case(s) for "
                  f"{Path(self.source.egrid_path).name} from the attached ResInsight; reloading from disk")
        self._case = instance.project.load_case(str(self.source.egrid_path), grid_only=False)
        return self._case

    def _ensure_summary_case(self):
        if self._summary_import_attempted:
            return self._summary_case
        self._summary_import_attempted = True
        if self.source.summary_path is None:
            return None
        instance = self._ensure_instance()
        evicted = self._evict_matching_summary_cases(instance)
        if evicted:
            print(f"[rips] evicted {evicted} previously loaded summary case(s) for "
                  f"{Path(self.source.summary_path).name} from the attached ResInsight; re-importing from disk")
        try:
            self._summary_case = instance.project.import_summary_case(str(self.source.summary_path))
        except Exception:
            self._summary_case = _find_matching_summary_case(instance.project, self.source.summary_path)
        return self._summary_case

    def load_geometry(self) -> GridGeometryPayload:
        case = self._ensure_case()
        _, _, _, nx, ny, nz = case.export_corner_point_grid()
        dims = (int(nx), int(ny), int(nz))
        corners = np.asarray([_cell_corners_to_array(cell) for cell in case.active_cell_corners()], dtype=float)
        cell_info = case.cell_info_for_active_cells()
        indices = np.asarray(
            [
                [
                    int(getattr(info.local_ijk, "i")),
                    int(getattr(info.local_ijk, "j")),
                    int(getattr(info.local_ijk, "k")),
                ]
                for info in cell_info
            ],
            dtype=int,
        )
        indices = _normalize_ijk_array(indices, dims)
        active_mask = np.zeros(dims, dtype=bool)
        active_mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        return GridGeometryPayload(
            corner_cells=corners,
            active_mask=active_mask,
            active_cell_indices=indices,
            cell_volumes=None,
        )

    def load_rock(
        self,
        perm_attrs: tuple[str, str, str] = ("PERMX", "PERMY", "PERMZ"),
        poro_attr: str = "PORO",
    ) -> RockPayload:
        _require_result_file(
            self.source.init_path,
            suffix=".INIT",
            needed_for=(
                f"STATIC_NATIVE rock properties ({', '.join(perm_attrs)}, {poro_attr})"
            ),
            source=self.source,
            hint=(
                "The simulator did not emit an .INIT file. Add the `INIT` keyword to "
                "the deck's GRID section and re-run the simulator to regenerate results."
            ),
        )
        case = self._ensure_case()
        perms = np.stack(
            [
                np.asarray(case.active_cell_property("STATIC_NATIVE", attr, 0), dtype=float)
                for attr in perm_attrs
            ],
            axis=1,
        )
        poro = np.asarray(case.active_cell_property("STATIC_NATIVE", poro_attr, 0), dtype=float)
        return RockPayload(perms=perms, poro=poro, perm_attrs=perm_attrs, poro_attr=poro_attr)

    def load_states(
        self,
        state_attrs: tuple[str, ...] = STATE_ATTRS_DEFAULT,
        selected_steps=None,
        max_steps: int | None = None,
    ) -> StateTimelinePayload:
        _require_result_file(
            self.source.unrst_path,
            suffix=".UNRST",
            needed_for=(
                f"DYNAMIC_NATIVE state snapshots ({', '.join(state_attrs)})"
            ),
            source=self.source,
            hint=(
                "The simulator did not emit a unified restart (.UNRST) file. Ensure the "
                "deck's SCHEDULE section requests restart output (e.g. via RPTRST) and "
                "re-run the simulator to regenerate results."
            ),
        )
        case = self._ensure_case()
        time_steps = list(case.time_steps())
        n_times = len(time_steps)
        selected_idx = select_time_indices(n_times, selected_steps=selected_steps, max_steps=max_steps)
        snapshots = {
            attr: np.stack(
                [
                    np.asarray(case.active_cell_property("DYNAMIC_NATIVE", attr, int(step_idx)), dtype=float)
                    for step_idx in selected_idx.tolist()
                ],
                axis=0,
            )
            for attr in state_attrs
        }
        dates = _time_steps_to_dates(time_steps)
        days_since_start = np.asarray(case.days_since_start(), dtype=float)
        if len(dates) != n_times or dates.isna().all():
            if self.source.start_date is not None and len(days_since_start) == n_times:
                dates = pd.DatetimeIndex(
                    pd.to_datetime(self.source.start_date) + pd.to_timedelta(days_since_start, unit="D")
                )
            else:
                dates = pd.DatetimeIndex([pd.NaT] * n_times)
        return StateTimelinePayload(
            snapshots=snapshots,
            indices=np.asarray(selected_idx, dtype=int),
            report_steps=np.asarray(selected_idx, dtype=int),
            available_report_steps=np.arange(n_times, dtype=int),
            dates=dates[np.asarray(selected_idx, dtype=int)],
            n_times=n_times,
        )

    def load_completions(self) -> CompletionPayload:
        case = self._ensure_case()
        geometry = self.load_geometry()
        project = self._ensure_instance().project
        dims = tuple(int(v) for v in geometry.active_mask.shape)
        time_steps = list(case.time_steps())
        case_dates = _time_steps_to_dates(time_steps)
        if len(case_dates) != len(time_steps) or case_dates.isna().all():
            days_since_start = np.asarray(case.days_since_start(), dtype=float)
            if self.source.start_date is not None and len(days_since_start) == len(time_steps):
                case_dates = pd.DatetimeIndex(
                    pd.to_datetime(self.source.start_date) + pd.to_timedelta(days_since_start, unit="D")
                )
            else:
                case_dates = pd.DatetimeIndex([pd.NaT] * len(time_steps))

        collections = []
        try:
            collections = list(project.descendants(self._rips.WellPathCollection))
        except Exception:
            collections = []
        if not collections and hasattr(project, "well_path_collection"):
            try:
                collection = project.well_path_collection()
                if collection is not None:
                    collections = [collection]
            except Exception:
                collections = []
        segments: dict[str, Any] = {}
        if collections:
            try:
                table = collections[0].completion_data_unified(case_id=int(case.id))
                segments = _build_segments_from_simulator_tables(table, dims)
            except Exception:
                segments = {}
        simulation_timelines = _simulation_timelines_from_case(
            case,
            dims,
            case_dates,
        )
        completion_tables_raw = _build_completion_table_cache(self.source)
        completion_tables: dict[str, dict[str, pd.DataFrame]] = {}
        for well_name, tables in completion_tables_raw.items():
            normalized_tables: dict[str, pd.DataFrame] = {}
            for table_name, frame in tables.items():
                normalized = frame.copy()
                if table_name == "compdat" and len(normalized):
                    normalized.loc[:, ["I", "J", "K"]] = _normalize_ijk_array(
                        normalized[["I", "J", "K"]].to_numpy(dtype=int),
                        dims,
                    )
                elif table_name == "welspecs" and len(normalized) and {"I", "J"}.issubset(normalized.columns):
                    ij = normalized[["I", "J"]].to_numpy(dtype=int)
                    if np.all(ij >= 1) and np.all(ij <= np.asarray(dims[:2], dtype=int)[None, :]):
                        ij = ij - 1
                    normalized.loc[:, ["I", "J"]] = ij
                normalized_tables[table_name] = normalized
            completion_tables[str(well_name).upper()] = normalized_tables
        if completion_tables and not segments:
            segments = _build_segments_from_completion_tables(
                completion_tables,
                dims,
            )
        if simulation_timelines:
            simulation_segments = _segments_from_simulation_timelines(simulation_timelines)
            if not segments:
                segments = simulation_segments
            else:
                for well_name, simulation_segment in simulation_segments.items():
                    existing = segments.get(well_name)
                    existing_blocks = getattr(existing, "blocks", None) if existing is not None else None
                    if existing is None or existing_blocks is None or len(existing_blocks) == 0:
                        segments[well_name] = simulation_segment
        if completion_tables and segments:
            parsed_segments = _build_segments_from_completion_tables(completion_tables, dims)
            for well_name, parsed_segment in parsed_segments.items():
                existing = segments.get(well_name)
                if existing is None:
                    segments[well_name] = parsed_segment
                    continue
                existing_blocks = getattr(existing, "blocks", None)
                if existing_blocks is None or len(existing_blocks) == 0:
                    existing.blocks = parsed_segment.blocks
                    existing.blocks_info = parsed_segment.blocks_info
                if getattr(existing, "welspecs", None) is None or len(getattr(existing, "welspecs", [])) == 0:
                    existing.welspecs = parsed_segment.welspecs
        return CompletionPayload(
            source=self.source,
            meta={
                "START": self.source.start_date,
                "simulation_wells": simulation_timelines,
                "completion_timelines": {
                    well_name: tables["compdat"].copy()
                    for well_name, tables in completion_tables.items()
                    if "compdat" in tables and len(tables["compdat"])
                },
            },
            segments=segments,
        )

    def load_summary(
        self,
        completion_payload: CompletionPayload | None = None,
        summary_vectors: tuple[str, ...] = SUMMARY_VECTORS_DEFAULT,
    ) -> SummaryPayload:
        if completion_payload is None:
            completion_payload = self.load_completions()
        summary_case = self._ensure_summary_case()
        well_names = list(completion_payload.segments.keys())
        if len(well_names) == 0:
            well_names = list(completion_payload.meta.get("simulation_wells", {}).keys())
        if summary_case is None or len(well_names) == 0:
            return SummaryPayload(results_by_well={}, dates=pd.DatetimeIndex([]), addresses_by_well={})
        return _results_frame_from_summary_case(
            summary_case,
            well_names,
            summary_vectors=summary_vectors,
        )

    def close(self) -> None:
        # drop our cases from a persistent (attached) instance so the next session -- and
        # any interactive use of that ResInsight -- starts from the files, not from memory
        for obj in (self._summary_case, self._case):
            if obj is not None:
                try:
                    obj.delete()
                except Exception:
                    pass
        self._summary_case = None
        self._summary_import_attempted = False
        self._case = None
        self._session.close()


def open_reservoir_backend(
    model_path: str | Path,
    backend: str = "auto",
    lifecycle: str = "attach_or_launch",
) -> ReservoirExtractorBackend:
    source = resolve_reservoir_source(model_path)
    backend_name = str(backend).lower()
    if backend_name not in ("rips", "auto"):
        raise ValueError(f"Unsupported reservoir backend '{backend}' (only 'rips' exists here).")
    return RipsExtractorBackend(source, lifecycle=lifecycle)


def reservoir_mesh_from_geometry_payload(
    geometry: GridGeometryPayload,
    use_only_active: bool = True,
    dedup_decimals: int = 8,
    volume_tol: float = 1e-12,
):
    return corner_cells_to_reservoir_mesh(
        corner_cells=geometry.corner_cells,
        active_mask=geometry.active_mask,
        active_cell_indices=geometry.active_cell_indices,
        cell_volumes=geometry.cell_volumes,
        use_only_active=use_only_active,
        dedup_decimals=dedup_decimals,
        volume_tol=volume_tol,
    )


def build_cell_rock_physics_from_payload(
    rock: RockPayload,
    reservoir_mesh,
) -> dict[str, Any]:
    return build_cell_rock_physics_from_arrays(
        reservoir_mesh=reservoir_mesh,
        perms=rock.perms,
        poro=rock.poro,
    )


def state_payload_to_snapshots(
    payload: StateTimelinePayload,
    state_attrs: tuple[str, ...] = STATE_ATTRS_DEFAULT,
) -> dict[str, Any]:
    snapshots = {
        attr: np.asarray(payload.snapshots[attr], dtype=float)
        for attr in state_attrs
        if attr in payload.snapshots
    }
    snapshots.update(
        {
            "indices": np.asarray(payload.indices, dtype=int),
            "report_steps": np.asarray(payload.report_steps, dtype=int),
            "available_report_steps": np.asarray(payload.available_report_steps, dtype=int),
            "dates": pd.to_datetime(payload.dates),
            "n_times": int(payload.n_times),
        }
    )
    if "SWAT" in snapshots and "SGAS" in snapshots:
        snapshots["SOIL"] = np.asarray(1.0 - snapshots["SWAT"] - snapshots["SGAS"], dtype=float)
    return snapshots


def build_reservoir_preprocessing_artifacts(
    model_path: str | Path,
    backend: str = "auto",
    lifecycle: str = "attach_or_launch",
    state_attrs: tuple[str, ...] = STATE_ATTRS_DEFAULT,
    state_vertex_attrs: tuple[str, ...] = ("PRESSURE", "SWAT", "SGAS", "RS", "SOIL"),
    selected_steps=None,
    max_steps: int | None = None,
    r_w: float = ECLIPSE_DEFAULT_R_W,
    skin_default: float = ECLIPSE_DEFAULT_SKIN,
    component_xtol: float = 1e-3,
    include_blackoil_tables: bool = False,
    include_aquifers: bool = False,
    allow_missing_blackoil_tables: bool = True,
    allow_missing_aquifers: bool = True,
) -> ReservoirPreprocessingArtifacts:
    timings: dict[str, float] = {}
    source = resolve_reservoir_source(model_path)
    with open_reservoir_backend(source.data_path, backend=backend, lifecycle=lifecycle) as extractor:
        stage_start = pd.Timestamp.now()
        geometry = extractor.load_geometry()
        timings["geometry"] = float((pd.Timestamp.now() - stage_start).total_seconds())

        stage_start = pd.Timestamp.now()
        reservoir_mesh = reservoir_mesh_from_geometry_payload(geometry)
        timings["reservoir_mesh"] = float((pd.Timestamp.now() - stage_start).total_seconds())

        stage_start = pd.Timestamp.now()
        rock_payload = extractor.load_rock()
        rock_data = build_cell_rock_physics_from_payload(rock_payload, reservoir_mesh)
        timings["rock_physics"] = float((pd.Timestamp.now() - stage_start).total_seconds())

        stage_start = pd.Timestamp.now()
        state_payload = extractor.load_states(
            state_attrs=state_attrs,
            selected_steps=selected_steps,
            max_steps=max_steps,
        )
        state_data = state_payload_to_snapshots(state_payload, state_attrs=state_attrs)
        timings["state_snapshots"] = float((pd.Timestamp.now() - stage_start).total_seconds())

        stage_start = pd.Timestamp.now()
        completion_payload = extractor.load_completions()
        timings["completions"] = float((pd.Timestamp.now() - stage_start).total_seconds())

        stage_start = pd.Timestamp.now()
        summary_payload = extractor.load_summary(completion_payload=completion_payload)
        timings["summary"] = float((pd.Timestamp.now() - stage_start).total_seconds())

        extractor_backend = str(extractor.backend_name)

    stage_start = pd.Timestamp.now()
    component_diagnostics = compute_component_diagnostics(reservoir_mesh, xtol=component_xtol)
    timings["component_diagnostics"] = float((pd.Timestamp.now() - stage_start).total_seconds())

    # The per-cell states are what training supervises; the vertex projections
    # below are only built for the attributes explicitly asked for.
    cell_states = {
        attr: np.asarray(state_data[attr], dtype=float)
        for attr in dict.fromkeys((*state_attrs, *state_vertex_attrs))
        if attr in state_data
    }
    if "SOIL" in state_vertex_attrs and "SOIL" not in cell_states and "SWAT" in state_data and "SGAS" in state_data:
        cell_states["SOIL"] = np.asarray(1.0 - np.asarray(state_data["SWAT"], dtype=float) - np.asarray(state_data["SGAS"], dtype=float), dtype=float)

    vertex_states: dict[str, np.ndarray] = {}     # the EDA-only vertex projections are not built

    stage_start = pd.Timestamp.now()
    well_metadata = prepare_well_metadata_from_payloads(
        completion_payload,
        summary_payload,
        reservoir_mesh,
        pd.to_datetime(state_data["dates"]),
        r_w=r_w,
        skin_default=skin_default,
        cell_tensors=rock_data["cell_tensors"],
    )
    timings["well_metadata"] = float((pd.Timestamp.now() - stage_start).total_seconds())

    blackoil_tables = None
    aquifer_vertices = np.zeros((0,), dtype=np.int32)
    auxiliary_backend = None
    aux_field = None
    if include_blackoil_tables or include_aquifers:
        auxiliary_backend = "deepfield"
        try:
            aux_field = _load_auxiliary_deepfield_field(source.data_path)
        except Exception:
            if include_blackoil_tables and not allow_missing_blackoil_tables:
                raise
            if include_aquifers and not allow_missing_aquifers:
                raise
            aux_field = None
            auxiliary_backend = None
        if aux_field is not None:
            if include_blackoil_tables:
                try:
                    stage_start = pd.Timestamp.now()
                    blackoil_tables = build_blackoil_table_pack(aux_field)
                    # STONE1 / STONE2 is a PROPS keyword the table loader does not see:
                    # a deck without one keeps Stone II (the closures' historical model)
                    model = deck_three_phase_oil_model(source.data_path)
                    blackoil_tables["kro_model"] = "stone1" if model == "stone1" else "stone2"
                    timings["blackoil_tables"] = float((pd.Timestamp.now() - stage_start).total_seconds())
                except Exception:
                    if not allow_missing_blackoil_tables:
                        raise
                    blackoil_tables = None
                    timings["blackoil_tables"] = 0.0
            if include_aquifers:
                try:
                    stage_start = pd.Timestamp.now()
                    aquifer_vertices = np.asarray(prepare_aquifer_vertices(aux_field, reservoir_mesh), dtype=np.int32)
                    timings["aquifer_vertices"] = float((pd.Timestamp.now() - stage_start).total_seconds())
                except Exception:
                    if not allow_missing_aquifers:
                        raise
                    aquifer_vertices = np.zeros((0,), dtype=np.int32)
                    timings["aquifer_vertices"] = 0.0
        else:
            if include_blackoil_tables:
                timings["blackoil_tables"] = 0.0
            if include_aquifers:
                timings["aquifer_vertices"] = 0.0

    stage_start = pd.Timestamp.now()
    category_indices = prepare_vertex_category_indices(
        n_vertices=reservoir_mesh.verts.shape[0],
        boundary_vertices=reservoir_mesh.boundary_vertices,
        well_vertex_ids=_collect_well_vertex_ids_from_metadata(well_metadata),
        aquifer_vertices=aquifer_vertices,
    )
    timings["vertex_categories"] = float((pd.Timestamp.now() - stage_start).total_seconds())

    # Deck unit system (deck-scan, DeepField meta as fallback); the assembled bundle is then
    # normalized to FIELD units in a single pass (no-op for FIELD decks like SPE1/SPE9).
    source_units = detect_unit_system(source.data_path, aux_field=aux_field)
    artifacts = ReservoirPreprocessingArtifacts(
        source=source,
        extractor_backend=extractor_backend,
        auxiliary_backend=auxiliary_backend,
        geometry=geometry,
        reservoir_mesh=reservoir_mesh,
        component_diagnostics=component_diagnostics,
        rock_payload=rock_payload,
        rock_data=rock_data,
        state_payload=None,   # state_data holds the snapshots the pipeline reads
        state_data=state_data,
        cell_states=cell_states,
        vertex_states=vertex_states,
        completion_payload=completion_payload,
        summary_payload=summary_payload,
        well_metadata=well_metadata,
        aquifer_vertices=np.asarray(aquifer_vertices, dtype=np.int32),
        category_indices={name: np.asarray(values, dtype=np.int32) for name, values in category_indices.items()},
        blackoil_tables=blackoil_tables,
        timings=timings,
        source_units=source_units,
    )
    return to_field_units(artifacts, source_units)


def _build_schedule_cache(source: ReservoirSource, meta: dict[str, Any]) -> dict[str, dict[str, pd.DataFrame]]:
    start_date = _parse_eclipse_date(meta.get("START")) if meta else pd.NaT
    if pd.isna(start_date) and source.start_date is not None:
        start_date = pd.to_datetime(source.start_date)

    parsed_controls: dict[str, dict[str, list[dict[str, Any]]]] = {}
    schedule_sources = [source.data_path] + _collect_schedule_include_paths(source.data_path)
    seen_schedule_sources: set[Path] = set()
    for schedule_path in schedule_sources:
        schedule_path = Path(schedule_path).resolve()
        if schedule_path in seen_schedule_sources:
            continue
        seen_schedule_sources.add(schedule_path)
        _merge_schedule_control_cache(parsed_controls, _parse_schedule_control_file(schedule_path, start_date))

    cache: dict[str, dict[str, pd.DataFrame]] = {}
    for well_name, tables in parsed_controls.items():
        cache[str(well_name).upper()] = {}
        for table_name, rows in tables.items():
            if not rows:
                continue
            cache[str(well_name).upper()][table_name] = pd.DataFrame(rows).sort_values("DATE").reset_index(drop=True)
    return cache


def _attach_schedule_tables_to_segments(
    completion_payload: CompletionPayload,
) -> dict[str, Any]:
    schedule_cache = _build_schedule_cache(completion_payload.source, completion_payload.meta)
    attached: dict[str, Any] = {}
    for well_name, segment in completion_payload.segments.items():
        clone = SimpleNamespace()
        for attr_name in (
            "name",
            "blocks",
            "blocks_info",
            "welltrack",
            "welspecs",
            "events",
            "wconprod",
            "wconinje",
            "wconhist",
            "welopen",
            "results",
        ):
            setattr(clone, attr_name, getattr(segment, attr_name, None))
        tables = schedule_cache.get(str(well_name).upper(), {})
        for attr_name in ("wconhist", "welopen", "wconprod", "wconinje"):
            table = tables.get(attr_name)
            if table is not None and len(table):
                setattr(clone, attr_name, table.copy())
        attached[well_name] = clone
    return attached


def _latest_simulation_snapshot(
    completion_payload: CompletionPayload,
    well_name: str,
    current_date: pd.Timestamp,
) -> dict[str, Any] | None:
    timelines = completion_payload.meta.get("simulation_wells", {}) if completion_payload.meta else {}
    entries = timelines.get(str(well_name).upper())
    if not entries:
        return None
    current_date = pd.to_datetime(current_date)
    latest = None
    for entry in entries:
        entry_date = pd.to_datetime(entry.get("DATE"), errors="coerce")
        if pd.isna(entry_date):
            latest = entry
            continue
        if entry_date <= current_date:
            latest = entry
        else:
            break
    return latest if latest is not None else entries[0]


def _latest_completion_snapshot(
    completion_payload: CompletionPayload,
    well_name: str,
    current_date: pd.Timestamp,
) -> dict[str, Any] | None:
    timelines = completion_payload.meta.get("completion_timelines", {}) if completion_payload.meta else {}
    frame = timelines.get(str(well_name).upper())
    if frame is None or len(frame) == 0:
        return None
    current_date = pd.to_datetime(current_date)
    dated = frame.copy()
    if "DATE" in dated.columns:
        dated["DATE"] = pd.to_datetime(dated["DATE"], errors="coerce")
        filtered = dated[dated["DATE"].isna() | (dated["DATE"] <= current_date)].reset_index(drop=True)
        if len(filtered) == 0:
            return {
                "DATE": current_date,
                "IS_OPEN": False,
                "ALL_BLOCKS": np.zeros((0, 3), dtype=int),
                "OPEN_BLOCKS": np.zeros((0, 3), dtype=int),
                "BLOCKS_INFO": _default_blocks_info_frame(0),
            }
        dated = filtered

    latest_by_block: dict[tuple[int, int, int], pd.Series] = {}
    for _, row in dated.sort_values("DATE" if "DATE" in dated.columns else dated.columns[0]).iterrows():
        key = (int(row["I"]), int(row["J"]), int(row["K"]))
        latest_by_block[key] = row
    latest_rows = pd.DataFrame(list(latest_by_block.values())).reset_index(drop=True)
    if len(latest_rows) == 0:
        return None

    status_text = latest_rows.get("STATUS")
    if status_text is None:
        open_mask = np.ones((len(latest_rows),), dtype=bool)
    else:
        open_mask = ~status_text.fillna("OPEN").astype(str).str.upper().isin({"SHUT", "STOP", "CLOSED"})
    open_rows = latest_rows.loc[open_mask].reset_index(drop=True)
    all_blocks = latest_rows[["I", "J", "K"]].to_numpy(dtype=int)
    open_blocks = open_rows[["I", "J", "K"]].to_numpy(dtype=int) if len(open_rows) else np.zeros((0, 3), dtype=int)
    return {
        "DATE": pd.to_datetime(latest_rows["DATE"].iloc[-1], errors="coerce") if "DATE" in latest_rows else current_date,
        "IS_OPEN": bool(len(open_blocks) > 0),
        "ALL_BLOCKS": np.asarray(all_blocks, dtype=int),
        "OPEN_BLOCKS": np.asarray(open_blocks, dtype=int),
        "BLOCKS_INFO": _blocks_info_from_compdat_frame(open_rows),
    }


def _slice_blocks_info(blocks_info: Any, positions: list[int]):
    if blocks_info is None:
        return None
    if hasattr(blocks_info, "iloc"):
        return blocks_info.iloc[positions].reset_index(drop=True)
    return np.asarray(blocks_info)[positions]


def _empty_like_blocks_info(blocks_info: Any) -> Any:
    if blocks_info is None:
        return _default_blocks_info_frame(0)
    if hasattr(blocks_info, "iloc"):
        return blocks_info.iloc[0:0].copy()
    return np.asarray(blocks_info)[:0]


def _select_segment_blocks_for_date(
    completion_payload: CompletionPayload,
    well_name: str,
    segment: Any,
    current_date: pd.Timestamp,
) -> tuple[np.ndarray, Any, dict[str, Any] | None]:
    base_blocks = np.asarray(getattr(segment, "blocks", np.zeros((0, 3), dtype=int)), dtype=int).reshape(-1, 3)
    base_blocks_info = getattr(segment, "blocks_info", None)
    if base_blocks_info is None or len(base_blocks_info) == 0:
        base_blocks_info = _default_blocks_info_frame(len(base_blocks))

    snapshot = _latest_simulation_snapshot(completion_payload, well_name, current_date)
    if snapshot is None:
        completion_snapshot = _latest_completion_snapshot(completion_payload, well_name, current_date)
        if completion_snapshot is None:
            return base_blocks, base_blocks_info, None
        if not bool(completion_snapshot.get("IS_OPEN", False)):
            return np.zeros((0, 3), dtype=int), _default_blocks_info_frame(0), completion_snapshot
        return (
            np.asarray(completion_snapshot.get("OPEN_BLOCKS", np.zeros((0, 3), dtype=int)), dtype=int).reshape(-1, 3),
            completion_snapshot.get("BLOCKS_INFO", _default_blocks_info_frame(0)),
            completion_snapshot,
        )

    open_blocks = np.asarray(snapshot.get("OPEN_BLOCKS", np.zeros((0, 3), dtype=int)), dtype=int).reshape(-1, 3)
    if len(open_blocks):
        selected_blocks = open_blocks
    elif bool(snapshot.get("IS_OPEN", False)):
        selected_blocks = base_blocks
    else:
        return np.zeros((0, 3), dtype=int), _empty_like_blocks_info(base_blocks_info), snapshot

    if len(selected_blocks) == 0 or len(base_blocks) == 0:
        return selected_blocks, _default_blocks_info_frame(len(selected_blocks)), snapshot

    base_lookup: dict[tuple[int, int, int], list[int]] = {}
    for idx, block in enumerate(base_blocks.tolist()):
        base_lookup.setdefault(tuple(int(v) for v in block), []).append(int(idx))

    positions: list[int] = []
    for block in selected_blocks.tolist():
        bucket = base_lookup.get(tuple(int(v) for v in block))
        if bucket:
            positions.append(bucket.pop(0))

    if len(positions) == len(selected_blocks):
        return selected_blocks, _slice_blocks_info(base_blocks_info, positions), snapshot
    return selected_blocks, _default_blocks_info_frame(len(selected_blocks)), snapshot


def _synthetic_control_from_simulation(
    segment: Any,
    simulation_snapshot: dict[str, Any] | None,
    result_snapshot: dict[str, float],
) -> pd.Series | None:
    if simulation_snapshot is None or not bool(simulation_snapshot.get("IS_OPEN", False)):
        return None

    producer_rate = 0.0
    has_producer_rate = False
    for key in ("WOPR", "WWPR", "WGPR"):
        value = float(result_snapshot.get(key, np.nan))
        if np.isfinite(value) and abs(value) > 0.0:
            producer_rate += abs(value)
            has_producer_rate = True

    injector_rate = 0.0
    has_injector_rate = False
    for key in ("WWIR", "WGIR"):
        value = float(result_snapshot.get(key, np.nan))
        if np.isfinite(value) and abs(value) > 0.0:
            injector_rate += abs(value)
            has_injector_rate = True

    well_type = str(simulation_snapshot.get("WELL_TYPE", "")).upper()
    phase = ""
    rate = np.nan
    wit = np.nan
    git = np.nan
    if well_type.startswith("INJ") or (has_injector_rate and not has_producer_rate):
        if has_injector_rate:
            rate = injector_rate
            wit = float(result_snapshot.get("WWIR", np.nan))
            git = float(result_snapshot.get("WGIR", np.nan))
            if np.isfinite(wit) and abs(wit) > 0.0:
                phase = "WATER"
            elif np.isfinite(git) and abs(git) > 0.0:
                phase = "GAS"
    elif has_producer_rate:
        rate = -producer_rate
        if np.isfinite(float(result_snapshot.get("WOPR", np.nan))) and abs(float(result_snapshot.get("WOPR", np.nan))) > 0.0:
            phase = "OIL"
        elif np.isfinite(float(result_snapshot.get("WWPR", np.nan))) and abs(float(result_snapshot.get("WWPR", np.nan))) > 0.0:
            phase = "WATER"
        elif np.isfinite(float(result_snapshot.get("WGPR", np.nan))) and abs(float(result_snapshot.get("WGPR", np.nan))) > 0.0:
            phase = "GAS"

    bhpt = float(result_snapshot.get("WBHP", np.nan))
    control_kind = "rate" if np.isfinite(rate) and abs(rate) > 0.0 else "bhp" if np.isfinite(bhpt) else ""
    if not control_kind:
        return None

    return pd.Series(
        {
            "DATE": pd.to_datetime(simulation_snapshot.get("DATE"), errors="coerce"),
            "BHPT": bhpt,
            "DREF": _segment_reference_depth(segment),
            "WIT": wit,
            "GIT": git,
            "RATE": rate,
            "PHASE": phase,
            "CONTROL_KIND": control_kind,
            "CONTROL_SOURCE": "SIMULATION_WELL",
        }
    )


def prepare_well_metadata_from_payloads(
    completion_payload: CompletionPayload,
    summary_payload: SummaryPayload,
    reservoir_mesh,
    selected_dates,
    allow_missing_wells: bool = True,
    r_w: float = ECLIPSE_DEFAULT_R_W,
    skin_default: float = ECLIPSE_DEFAULT_SKIN,
    cell_tensors: np.ndarray | None = None,
) -> dict[str, Any]:
    segments = _attach_schedule_tables_to_segments(completion_payload)
    if not segments:
        if allow_missing_wells:
            return {"has_wells": False, "steps": [{} for _ in range(len(selected_dates))]}
        raise ValueError("No well completion metadata was available for this model.")

    for well_name, segment in segments.items():
        results = summary_payload.results_by_well.get(str(well_name))
        if results is not None and len(results):
            segment.results = results.copy()

    if isinstance(selected_dates, pd.DatetimeIndex):
        timeline = selected_dates
    elif len(selected_dates) and isinstance(selected_dates[0], (pd.Timestamp, np.datetime64)):
        timeline = pd.to_datetime(selected_dates)
    else:
        if allow_missing_wells:
            return {"has_wells": False, "steps": [{} for _ in range(len(selected_dates))]}
        raise ValueError("Well metadata requires datetime-aligned state snapshots.")

    step_entries = []
    for current_date in timeline:
        bhp_entries = []
        rate_entries = []
        well_results = []
        for well_name, segment in segments.items():
            block_indices, blocks_info, simulation_snapshot = _select_segment_blocks_for_date(
                completion_payload,
                str(well_name),
                segment,
                current_date,
            )
            if len(block_indices) == 0:
                continue

            block_centroids = _block_centroids_from_reservoir_mesh(block_indices, reservoir_mesh)
            track = _spatialize_track(getattr(segment, "welltrack", None))
            tangent_fallback = _block_path_tangents(block_centroids)
            if blocks_info is None or len(blocks_info) == 0:
                blocks_info = _default_blocks_info_frame(len(block_indices))

            result_snapshot = _extract_well_result_snapshot(segment, current_date)
            event = _resolve_well_control(segment, current_date)
            if event is None:
                event = _synthetic_control_from_simulation(segment, simulation_snapshot, result_snapshot)
            if event is None:
                continue

            skin_series = blocks_info["SKIN"] if "SKIN" in blocks_info else np.zeros(len(block_indices))
            perf_ratio_series = blocks_info["PERF_RATIO"] if "PERF_RATIO" in blocks_info else np.ones(len(block_indices))
            cf_series = blocks_info["CF"] if "CF" in blocks_info else np.full(len(block_indices), np.nan)
            rad_series = blocks_info["RAD"] if "RAD" in blocks_info else np.full(len(block_indices), np.nan)

            bhpt = float(event["BHPT"]) if "BHPT" in event and pd.notna(event["BHPT"]) else np.nan
            generic_rate = float(event["RATE"]) if "RATE" in event and pd.notna(event["RATE"]) else np.nan
            wit = float(event["WIT"]) if "WIT" in event and pd.notna(event["WIT"]) else np.nan
            git = float(event["GIT"]) if "GIT" in event and pd.notna(event["GIT"]) else np.nan
            z_bh = float(event["DREF"]) if "DREF" in event and pd.notna(event["DREF"]) else np.nan
            control_kind = str(event["CONTROL_KIND"]).lower() if "CONTROL_KIND" in event and pd.notna(event["CONTROL_KIND"]) else ""
            control_phase = str(event["PHASE"]).upper() if "PHASE" in event and pd.notna(event["PHASE"]) else ""

            weight_accumulator = []
            cell_payloads = []
            for local_idx, block in enumerate(block_indices):
                lookup_key = tuple(int(v) for v in block.tolist())
                if lookup_key not in reservoir_mesh.active_cell_lookup:
                    continue
                cell_idx = reservoir_mesh.active_cell_lookup[lookup_key]
                point = reservoir_mesh.cell_centroids[cell_idx]
                tangent = _nearest_track_tangent(track, point) if track is not None else None
                if tangent is None:
                    tangent = tangent_fallback[local_idx]
                cell_vertices = reservoir_mesh.cell_to_unique_vertices[cell_idx]
                if len(cell_vertices) == 0:
                    continue
                cell_sizes = reservoir_mesh.cell_lengths[cell_idx]
                z_cell = reservoir_mesh.cell_centroids[cell_idx, 2]
                skin = float(skin_series.iloc[local_idx]) if hasattr(skin_series, "iloc") else float(skin_series[local_idx])
                if not np.isfinite(skin):
                    skin = float(skin_default)
                # COMPDAT item 9 (wellbore diameter, halved into RAD) overrides the global r_w;
                # item 8 (an explicit connection transmissibility factor) is carried as ``cf``
                # so the well pack can honour it the way the simulator does.
                r_w_block = _series_value(rad_series, local_idx)
                r_w_block = float(r_w_block) if np.isfinite(r_w_block) and r_w_block > 0.0 else float(r_w)
                cf = _series_value(cf_series, local_idx)
                cf = float(cf) if np.isfinite(cf) and cf > 0.0 else float("nan")
                try:
                    peaceman = compute_qw_full_tensor(
                        cell_tensors[cell_idx],
                        tangent,
                        tuple(cell_sizes.tolist()),
                        p_bh=bhpt if np.isfinite(bhpt) else 0.0,
                        z_bh=z_bh if np.isfinite(z_bh) else z_cell,
                        z_cell=z_cell,
                        r_w=r_w_block,
                        skin=skin,
                    )
                    weight = float(peaceman["WI"])
                except Exception:
                    peaceman = None
                    perf_ratio = float(perf_ratio_series.iloc[local_idx]) if hasattr(perf_ratio_series, "iloc") else float(perf_ratio_series[local_idx])
                    weight = max((cf if np.isfinite(cf) else 1.0) * perf_ratio, 0.0)

                payload = {
                    "well_name": str(well_name),
                    "perf_id": int(local_idx),
                    "cell_idx": cell_idx,
                    "cell_vertices": np.asarray(cell_vertices, dtype=int),
                    "vertex_weights": np.full(len(cell_vertices), 1.0 / len(cell_vertices), dtype=float),
                    "z_cell": float(z_cell),
                    "z_bh": float(z_bh if np.isfinite(z_bh) else z_cell),
                    "tangent": np.asarray(tangent, dtype=float),
                    "weight": float(weight),
                    "peaceman": peaceman,
                    "h_s": float(peaceman["h_s"]) if peaceman is not None else float(cell_sizes[2]),
                    "r_e": float(peaceman["r_e"]) if peaceman is not None else float(max(max(cell_sizes), r_w_block * 1.01)),
                    "r_w": float(r_w_block),
                    "cf": float(cf),
                    "skin": float(skin),
                    "K_perp": np.asarray(peaceman["K_perp"], dtype=float) if peaceman is not None else np.eye(2, dtype=float),
                    "control_kind": control_kind,
                    "control_phase": control_phase,
                }
                weight_accumulator.append(weight)
                cell_payloads.append(payload)

            if len(cell_payloads) == 0:
                continue

            provisional_rate = generic_rate if np.isfinite(generic_rate) else 0.0
            if not np.isfinite(generic_rate):
                if np.isfinite(wit):
                    provisional_rate += wit
                if np.isfinite(git):
                    provisional_rate += git
            well_phase = _well_control_phase_label(control_phase, provisional_rate, result_snapshot)
            well_results.append(
                {
                    "well_name": str(well_name),
                    "control_kind": control_kind,
                    "control_phase": well_phase,
                    "control_bhpt": float(bhpt) if np.isfinite(bhpt) else np.nan,
                    "control_rate": float(generic_rate) if np.isfinite(generic_rate) else np.nan,
                    "control_wit": float(wit) if np.isfinite(wit) else np.nan,
                    "control_git": float(git) if np.isfinite(git) else np.nan,
                    "obs_wbhp": float(result_snapshot["WBHP"]) if np.isfinite(result_snapshot["WBHP"]) else np.nan,
                    "obs_wthp": float(result_snapshot["WTHP"]) if np.isfinite(result_snapshot["WTHP"]) else np.nan,
                    "obs_wopr": float(result_snapshot["WOPR"]) if np.isfinite(result_snapshot["WOPR"]) else np.nan,
                    "obs_wwpr": float(result_snapshot["WWPR"]) if np.isfinite(result_snapshot["WWPR"]) else np.nan,
                    "obs_wgpr": float(result_snapshot["WGPR"]) if np.isfinite(result_snapshot["WGPR"]) else np.nan,
                    "obs_wwir": float(result_snapshot["WWIR"]) if np.isfinite(result_snapshot["WWIR"]) else np.nan,
                    "obs_wgir": float(result_snapshot["WGIR"]) if np.isfinite(result_snapshot["WGIR"]) else np.nan,
                    "obs_wopt": float(result_snapshot["WOPT"]) if np.isfinite(result_snapshot["WOPT"]) else np.nan,
                    "obs_wwpt": float(result_snapshot["WWPT"]) if np.isfinite(result_snapshot["WWPT"]) else np.nan,
                    "obs_wgpt": float(result_snapshot["WGPT"]) if np.isfinite(result_snapshot["WGPT"]) else np.nan,
                    "total_weight": float(np.sum(weight_accumulator)) if len(weight_accumulator) else 0.0,
                }
            )

            if control_kind == "bhp" or (not control_kind and np.isfinite(bhpt) and not np.isfinite(generic_rate)):
                for payload in cell_payloads:
                    peaceman = payload["peaceman"]
                    if peaceman is None:
                        continue
                    bhp_entries.append(
                        {
                            "well_name": payload["well_name"],
                            "perf_id": payload["perf_id"],
                            "cell_idx": payload["cell_idx"],
                            "cell_vertices": payload["cell_vertices"],
                            "vertex_weights": payload["vertex_weights"],
                            "p_bh": float(bhpt),
                            "z_bh": float(payload["z_bh"]),
                            "z_cell": float(payload["z_cell"]),
                            "weight": float(payload["weight"]),
                            "h_s": float(payload["h_s"]),
                            "r_w": float(payload.get("r_w", float("nan"))),
                            "cf": float(payload.get("cf", float("nan"))),
                            "r_e": float(payload["r_e"]),
                            "skin": float(payload["skin"]),
                            "K_perp": np.asarray(payload["K_perp"], dtype=float),
                        }
                    )
            else:
                total_rate = generic_rate if np.isfinite(generic_rate) else 0.0
                if not np.isfinite(generic_rate):
                    if np.isfinite(wit):
                        total_rate += wit
                    if np.isfinite(git):
                        total_rate += git
                if total_rate == 0.0:
                    continue
                total_weight = float(np.sum(weight_accumulator))
                if total_weight <= 0.0:
                    total_weight = float(len(cell_payloads))
                    weight_accumulator = [1.0 for _ in cell_payloads]
                for phase_name, phase_rate in _resolve_rate_phase_split(control_phase, total_rate, result_snapshot):
                    for payload, weight in zip(cell_payloads, weight_accumulator):
                        distributed = phase_rate * weight / total_weight
                        rate_entries.append(
                            {
                                "well_name": payload["well_name"],
                                "perf_id": payload["perf_id"],
                                "cell_idx": payload["cell_idx"],
                                "cell_vertices": payload["cell_vertices"],
                                "vertex_weights": payload["vertex_weights"],
                                "rate": float(distributed),
                                "weight": float(payload["weight"]),
                                "h_s": float(payload["h_s"]),
                                "r_w": float(payload.get("r_w", float("nan"))),
                                "cf": float(payload.get("cf", float("nan"))),
                                "r_e": float(payload["r_e"]),
                                "skin": float(payload["skin"]),
                                "K_perp": np.asarray(payload["K_perp"], dtype=float),
                                "z_bh": float(payload["z_bh"]),
                                "z_cell": float(payload["z_cell"]),
                                "control_phase": phase_name,
                            }
                        )
        step_entries.append(
            {
                "date": current_date,
                "bhp_entries": bhp_entries,
                "rate_entries": rate_entries,
                "well_results": well_results,
                "has_wells": bool(bhp_entries or rate_entries),
            }
        )

    return {"has_wells": any(step.get("has_wells", False) for step in step_entries), "steps": step_entries}
