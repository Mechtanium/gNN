r"""
Reservoir case loading (port of the notebook's data-provenance cell).

Loads the prep-cache artifacts for the configured deck
and derives everything the pipeline consumes: hex-mesh geometry, the physical
output ranges that anchor the network's sigmoid transforms, per-cell rock
properties with their layered profiles, reference state tensors, and the
no-flow boundary face list. Well controls and observations live in
:mod:`pinnlab.wells` (the time-resolved well pack and forcing).

The four network outputs are mapped to physical primaries through range
anchors derived from the deck (SCAL/PVT/ROCK tables, reference states, BHP
limits):

.. math::

    p_o = P_{\min} + (P_{\max} - P_{\min})\,\sigma(y_1),
    \quad
    S_w = S_{wc} + (1 - S_{wc})\,\sigma(y_2),
    \quad
    S_g = \sigma(y_3)\,(1 - S_w),
    \quad
    R_{so} = R_{s,\max}\,\sigma(y_4)

where:
- :math:`\sigma`: the logistic sigmoid, keeping every primary inside its physical range by construction (hard constraints).
- :math:`P_{\min}, P_{\max}`: pressure anchors padded by the ``p_pad`` / ``p_margin`` safety margins.
- :math:`S_{wc}`: connate water saturation from the SCAL tables.
- :math:`R_{s,\max}`: the saturated solution gas-oil ratio ceiling (``rs_step`` / ``rs_headroom`` margins).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as onp

from .config import RunConfig, case_paths


@dataclass
class CaseData:
    """Everything the pipeline needs from one reservoir case (host-side, unsharded)."""

    art: Any                       # ReservoirPreprocessingArtifacts
    tables: Any                    # blackoil_closures.tables_to_jax pytree
    phys: Any                      # ReservoirExtractors.PhysicalOutputRanges

    # geometry
    verts: onp.ndarray             # (n_nodes, 3) float64
    hexes: onp.ndarray             # (n_cells, 8) int64
    centroids: onp.ndarray         # (n_cells, 3) float32
    cell_len: onp.ndarray          # (n_cells, 3) float32
    aci: onp.ndarray               # (n_cells, 3) active (i, j, k)
    perms: onp.ndarray             # (n_cells, 3) mD
    poro_cell: onp.ndarray         # (n_cells,) deck reference porosity
    n_nodes: int
    n_cells: int
    lo: Any                        # (3,) f32 bbox min
    hi: Any                        # (3,) f32 bbox max
    span: onp.ndarray              # (3,) extent [ft]

    # time axis
    times: onp.ndarray             # (n_times,) days since first report
    t_end: float
    n_times: int

    # layered rock profiles (chain_rule residual)
    zb: Any                        # (n_layers-1,) layer z boundaries
    kz: Any                        # (n_layers,) layer permeability [mD]
    phz: Any                       # (n_layers,) layer mean reference porosity

    # surface densities [lb/ft^3]
    dens_o: float
    dens_w: float
    dens_g: float

    # reference states
    pres: onp.ndarray              # (n_times, n_cells)
    swat: onp.ndarray
    sgas: onp.ndarray
    rs: onp.ndarray
    cell_idx_data: Any             # (n_times*n_cells,) int32
    time_data: Any                 # (n_times*n_cells,) f32
    y_data: Any                    # (n_times*n_cells, 4) f32
    y_ic: Any                      # (n_cells, 4) f32
    state_scale: Any               # (4,) f32

    # no-flow boundary faces
    bc_cell: Any                   # (n_faces,) int32 cell index
    bc_axis: Any                   # (n_faces,) int32 axis 0..2

    # unit constants
    c_f: float = 0.0               # FIELD Darcy coefficient (mD,cP,psi,ft -> rb/day)
    rb_ft3: float = 0.0            # ft^3 -> reservoir barrel
    grav_grad: float = 1.0 / 144.0 # psi/ft per (lb/ft^3)

    # collocation window (train/test time split); t_hi == 0 means "up to t_end".
    # t_end is deliberately NOT narrowed by a split: it sets the encoder's time
    # normalization, so shrinking it would remap every physical time and make a
    # model trained on one half untransferable to the other.
    t_lo: float = 0.0              # collocation window start [days]
    t_hi: float = -1.0             # collocation window end [days]; < 0 = unset -> t_end
    t_ic: float = 0.0              # time at which the `ic` group is anchored [days]
    infer_half: bool = False       # True on the held-out half: imposed schedules apply here
    # observation window end [days]; < 0 = unset -> t_hi. Well/cell-state observation rows
    # exist only at report steps <= this bound (joint mode: the split time t_s).
    t_obs_hi: float = -1.0
    # time from which cfg.inference.schedules override the deck controls; < 0 = never
    t_sched: float = -1.0

    def t_span(self) -> tuple[float, float]:
        r"""
        The collocation window :math:`[t_{\mathrm{lo}}, t_{\mathrm{hi}}]` in days.

        Every sampled collocation, boundary and candidate time is drawn from this
        interval and nowhere else, which is what confines a run to its half of the
        train/test split. The unsplit default is the whole history
        :math:`[0, t_{\mathrm{end}}]`.

        where:
        - :math:`t_{\mathrm{lo}}, t_{\mathrm{hi}}`: the window bounds (a negative ``t_hi`` is "unset" and reads as :math:`t_{\mathrm{end}}`).
        - :math:`t_{\mathrm{end}}`: the case's last report time, which also fixes the encoder's time normalization and is never narrowed by a split.
        """
        return float(self.t_lo), float(self.t_hi if self.t_hi >= 0.0 else self.t_end)

    def t_obs_span(self) -> tuple[float, float]:
        r"""
        The observation window :math:`[t_{\mathrm{lo}}, t_{\mathrm{obs}}]` in days — the
        report steps whose cell states and well observations may enter the objective.
        Equals :meth:`t_span` unless a joint history-match/forecast split narrowed it.
        """
        lo, hi = self.t_span()
        return lo, (float(self.t_obs_hi) if self.t_obs_hi >= 0.0 else hi)

    def time_strata(self, n_slices: int, n_forecast: int = 0) -> list:
        r"""
        Collocation sub-windows with their slice counts, ``[(lo, hi, n), ...]``.

        A single window returns ``[(t_lo, t_hi, n_slices)]``. A joint split
        (``t_obs_hi < t_hi``) returns the history and forecast windows separately,

        .. math::

            n_{\mathrm{fc}} \;=\; \max\Bigl(1,\ \Bigl\lfloor n_{\mathrm{slices}}\,
            \frac{t_{\mathrm{hi}} - t_{\mathrm{obs}}}{t_{\mathrm{hi}} - t_{\mathrm{lo}}} \Bigr\rceil\Bigr),
            \qquad n_{\mathrm{hist}} = n_{\mathrm{slices}} - n_{\mathrm{fc}}

        where:
        - :math:`n_{\mathrm{fc}}`: slices on the forecast window — proportional to its length when ``n_forecast == 0``, else exactly ``n_forecast`` (capped so at least one slice stays on the history when it exists).
        - :math:`t_{\mathrm{obs}}`: the observation window end (``t_obs_hi``).

        The split guarantees every window is sampled: a forecast window that is short
        against the history would otherwise draw no collocation slice at all, leaving the
        forecast constrained by nothing but the field's smoothness.
        """
        lo, hi = self.t_span()
        _, obs = self.t_obs_span()
        if not (self.t_obs_hi >= 0.0 and obs < hi - 1e-9):
            return [(lo, hi, int(n_slices))]
        n_slices = int(n_slices)
        if obs <= lo + 1e-9:                     # no history window at all
            return [(lo, hi, n_slices)]
        if n_forecast > 0:
            n_fc = min(int(n_forecast), max(n_slices - 1, 1))
        else:
            n_fc = max(1, int(round(n_slices * (hi - obs) / max(hi - lo, 1e-12))))
            n_fc = min(n_fc, max(n_slices - 1, 1))
        n_hist = max(n_slices - n_fc, 0)
        return [(lo, obs, n_hist), (obs, hi, n_fc)]

    def perm_of_z(self, z):
        """Diagonal absolute permeability (k_x, k_y, k_z) [mD] at depth z (isotropic per layer)."""
        import jax.numpy as jnp

        idx = jnp.sum((z > self.zb).astype(jnp.int32))
        k = self.kz[idx]
        return k, k, k

    def poro_of_z(self, z):
        """Layer-mean reference porosity at depth z (the chain_rule accumulation weight)."""
        import jax.numpy as jnp

        idx = jnp.sum((z > self.zb).astype(jnp.int32))
        return self.phz[idx]


def load_case(cfg: RunConfig) -> CaseData:
    """Load the prep cache for ``cfg.deck_path`` and derive the pipeline inputs."""
    import jax.numpy as jnp

    import modules.utils.blackoil_closures as bo
    from modules.utils.blackoil_closures import FIELD_DARCY_COEFF, RB_PER_FT3
    from modules.utils.ReservoirExtractors import derive_physical_output_ranges
    from modules.utils.ReservoirPrepCache import build_or_load_preprocessing_artifacts

    model_path, prep_cache = case_paths(cfg)
    art = build_or_load_preprocessing_artifacts(model_path, prep_cache, build_if_missing=False)
    tables = bo.tables_to_jax(art.blackoil_tables)

    phys = derive_physical_output_ranges(
        art, p_pad=cfg.p_pad, p_margin_frac=cfg.p_margin,
        rs_step=cfg.rs_step, rs_headroom=cfg.rs_headroom,
    )

    # --- hex mesh geometry -------------------------------------------------
    verts = onp.asarray(art.reservoir_mesh.verts, onp.float64)
    hexes = onp.asarray([onp.asarray(c) for c in art.reservoir_mesh.cell_to_unique_vertices], onp.int64)
    centroids = onp.asarray(art.reservoir_mesh.cell_centroids, onp.float32)
    cell_len = onp.asarray(art.reservoir_mesh.cell_lengths, onp.float32)
    aci = onp.asarray(art.reservoir_mesh.active_cell_indices)
    perms = onp.asarray(art.rock_payload.perms, onp.float64)
    poro_cell = onp.asarray(art.rock_payload.poro, onp.float64)
    n_nodes, n_cells = verts.shape[0], hexes.shape[0]

    lo = jnp.asarray(verts.min(0), jnp.float32)
    hi = jnp.asarray(verts.max(0), jnp.float32)
    span = onp.asarray(verts.max(0) - verts.min(0))

    # --- time axis: days since first report date ----------------------------
    # total_seconds()/86400, not .days: a deck may report sub-daily (SPE2EQUI writes
    # its first 101 restarts within the opening day, at seconds resolution), and
    # integer-day truncation would collapse all of them onto t = 0 -- handing the
    # data group 101 mutually contradictory targets at one network input time, and
    # leaving any train/test split with a zero-width early window.
    dates = art.state_data["dates"]
    times = onp.array([(d - dates[0]).total_seconds() / 86400.0 for d in dates],
                      dtype=onp.float64).astype(onp.float32)
    t_end = float(times[-1])

    # --- layered rock profiles k(z), phi(z) for the chain_rule residual ------
    layer_z, layer_k, layer_p = [], [], []
    for kk in sorted(set(aci[:, 2].tolist())):
        m = aci[:, 2] == kk
        layer_z.append(float(centroids[m, 2].mean()))
        layer_k.append(float(perms[m, 0].mean()))
        layer_p.append(float(poro_cell[m].mean()))
    layer_z = onp.array(layer_z)
    zb = jnp.asarray(0.5 * (layer_z[:-1] + layer_z[1:]), jnp.float32)
    kz = jnp.asarray(onp.array(layer_k), jnp.float32)
    phz = jnp.asarray(onp.array(layer_p), jnp.float32)

    # --- surface densities (DENSITY) -----------------------------------------
    dens_o = float(onp.asarray(art.blackoil_tables["dens_o"])[0])
    dens_w = float(onp.asarray(art.blackoil_tables["dens_w"])[0])
    dens_g = float(onp.asarray(art.blackoil_tables["dens_g"])[0])

    # --- reference state tensors (data + initial condition), per cell ---------
    pres = onp.asarray(art.cell_states["PRESSURE"])
    swat = onp.asarray(art.cell_states["SWAT"])
    sgas = onp.asarray(art.cell_states["SGAS"])
    rs = onp.asarray(art.cell_states["RS"])
    n_times = pres.shape[0]
    cell_idx_data = jnp.asarray(onp.tile(onp.arange(n_cells), n_times), jnp.int32)
    time_data = jnp.asarray(onp.repeat(times, n_cells), jnp.float32)
    y_data = jnp.asarray(onp.stack([pres, swat, sgas, rs], -1).reshape(-1, 4).astype(onp.float32))
    y_ic = jnp.asarray(onp.stack([pres[0], swat[0], sgas[0], rs[0]], -1).astype(onp.float32))
    state_scale = jnp.array(
        [float(pres.max() - pres.min()), 1.0, 1.0, float(max(rs.max() - rs.min(), 1e-2))],
        jnp.float32,
    )

    bc_cell, bc_axis = _boundary_faces(centroids, cell_len, verts)

    return CaseData(
        art=art, tables=tables, phys=phys,
        verts=verts, hexes=hexes, centroids=centroids, cell_len=cell_len, aci=aci,
        perms=perms, poro_cell=poro_cell, n_nodes=n_nodes, n_cells=n_cells,
        lo=lo, hi=hi, span=span,
        times=times, t_end=t_end, n_times=n_times,
        zb=zb, kz=kz, phz=phz,
        dens_o=dens_o, dens_w=dens_w, dens_g=dens_g,
        pres=pres, swat=swat, sgas=sgas, rs=rs,
        cell_idx_data=cell_idx_data, time_data=time_data, y_data=y_data, y_ic=y_ic,
        state_scale=state_scale,
        bc_cell=bc_cell, bc_axis=bc_axis,
        c_f=float(FIELD_DARCY_COEFF), rb_ft3=float(RB_PER_FT3),
    )


def _boundary_faces(centroids: onp.ndarray, cell_len: onp.ndarray, verts: onp.ndarray, eps: float = 1.0):
    """(cell, axis) pairs whose face touches the bounding box: the no-flow BC set."""
    import jax.numpy as jnp

    lo = verts.min(0)
    hi = verts.max(0)
    rows = []
    for ci in range(centroids.shape[0]):
        c = onp.asarray(centroids[ci])
        hl = 0.5 * onp.asarray(cell_len[ci])
        for ax in range(3):
            if c[ax] - hl[ax] <= lo[ax] + eps:
                rows.append((ci, ax))
            if c[ax] + hl[ax] >= hi[ax] - eps:
                rows.append((ci, ax))
    bc_cell = jnp.asarray(onp.array([r[0] for r in rows]), jnp.int32)
    bc_axis = jnp.asarray(onp.array([r[1] for r in rows]), jnp.int32)
    return bc_cell, bc_axis
