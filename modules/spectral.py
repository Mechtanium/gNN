r"""
Static hex-FEM operators and the cached Laplace eigenbasis (spectral encoding).

Two-tier disk cache (see :mod:`spectral_cache`): the N_EIG-independent static
FEM operators (element matrices, sparse stiffness/storage-mass, null-mode
count) and the column-sliceable eigenbasis. The generalized eigenproblem with
homogeneous Neumann boundary conditions is

.. math::

    A\,v_m = \lambda_m\, M_\phi\, v_m,
    \qquad
    A_{ij} = \int_\Omega \kappa(x)\,\nabla N_i \cdot \nabla N_j \, d\Omega,
    \qquad
    M_{\phi,ij} = \int_\Omega \phi_0(x)\, N_i\, N_j \, d\Omega

where:
- :math:`N_i`: the trilinear hex FE shape functions.
- :math:`\kappa`: the stiffness weight — the permeability tensor for ``perm_weighted``, unity for ``geometric`` (the literal Δ-PINN form).
- :math:`\phi_0`: the per-cell reference porosity, making :math:`M_\phi` the **storage matrix** so :math:`\lambda_m^{-1}` carries the meaning of a hydraulic time constant of the pressure-diffusion operator (a uniform :math:`\phi_0` rescales the spectrum without changing the eigenvectors).
- :math:`\lambda_m, v_m`: the retained eigenpairs, with one constant null mode per connected mesh component dropped.

**Vertical-aware retention.** On reservoir-shaped (pancake) domains the plain
lowest-:math:`\lambda` band is numerically blind to the vertical coordinate:
for a box :math:`[0,L_x]\times[0,L_y]\times[0,H_z]` the first vertical
overtone exceeds the first areal one by :math:`(k_z/k_x)(L_x/H_z)^2`, so every
resolvable areal mode is cheaper than any vertical mode. Retention therefore
guarantees a vertical block: with the per-mode **vertical variance fraction**

.. math::

    \upsilon_m \;=\; \frac{\lVert (I - \Pi_z)\, v_m \rVert^{2}}{\lVert v_m - \bar v_m \mathbf{1} \rVert^{2}},

where:
- :math:`\Pi_z`: the averaging projector over each vertical column of nodes (nodes sharing an areal position), so :math:`(I - \Pi_z)v_m` is the part of the mode that varies with depth at fixed areal position.
- :math:`\bar v_m`: the nodal mean of the mode,

the retained set is the lowest-:math:`\lambda` band amended so that at least
``n_eig_z`` modes with :math:`\upsilon_m > 1/2` are present (the lowest such
candidates replace the highest-:math:`\lambda` areal picks). The retained
columns are ordered **vertical block first** (ascending :math:`\lambda`
within each block) so every prefix consumer of the basis — in particular the
spectral log-permeability parameterization — sees the vertical family.

The spectral residual metric weights are selected by ``cfg.spectral_mu``:

.. math::

    \mu_m = 1, \qquad
    \mu_m = \frac{1}{1 + \lambda_m}, \qquad
    \mu_m = \lambda_m

where:
- :math:`\mu_m`: the diagonal weight applied to the :math:`m`-th Galerkin residual coefficient (plain projection, smoothing, or stiffness-emphasizing).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace as _dc_replace
from typing import Any

import numpy as onp

from .config import (BackpropDesign, ConfigError, RunConfig, SpectralSettings,
                     case_label, case_paths, needs_eigenbasis)
from .casedata import CaseData
from .meshenv import MeshEnv

# Retention tuning now lives on ``cfg.spec`` (:class:`~pinnlab.config.SpectralSettings`);
# these mirror its defaults and serve as fallbacks for the standalone helpers below.
# Width adopted for n_eig=0 when the addressability gate is switched off entirely
# (spec.addressability_min = 0), so there is no floor to resolve against.
_FALLBACK_N_EIG = 48
_DENSE_EIG_MAX_N = SpectralSettings.dense_eig_max_n
_BAND_CAP_MULT = SpectralSettings.band_cap_mult
_UPSILON_VERTICAL = SpectralSettings.upsilon_vertical
_COLUMN_TOL_FT = SpectralSettings.column_tol_ft


@dataclass(frozen=True)
class MeshAdmissibility:
    r"""
    Conformity and connectivity of the hex mesh underlying the FEM operators.

    Two independent facts decide whether a spectral encoding can address a
    reservoir at all, and both are cheap to establish before any eigensolve.

    **Conformity.** In a conforming structured hex mesh every interior node is
    shared by the eight cells meeting at it. A maximum node reuse below that
    means no node is interior — the cells do not actually touch — which is the
    signature of a block-centred :math:`(\mathrm{DX}, \mathrm{DY}, \mathrm{DZ},
    \mathrm{TOPS})` deck converted into literal disjoint blocks: on a grid dipping
    by :math:`\vartheta` the neighbours in the dip direction are offset by
    :math:`\Delta x \tan\vartheta` while each cell is only :math:`\Delta z` tall,
    so for :math:`\Delta x \tan\vartheta > \Delta z` consecutive cells never meet.

    **Connectivity.** The stiffness graph splits into :math:`c` components, each
    contributing one constant null mode. Fragmentation is not by itself a defect
    — a faulted corner-point grid is genuinely node-nonconforming across throws —
    but eigenvectors of a block-diagonal operator are supported on a single
    component, so a budget drawn by global eigenvalue order can leave whole
    components at constant features. Components are therefore ranked by the pore
    volume they carry,

    .. math::

        \varpi_c \;=\;
        \frac{\sum_{e \in c} \phi_e\, V_e}{\sum_{e} \phi_e\, V_e},
        \qquad
        V_e \;=\; \sum_{g} (J w)_{e g}

    where:
    - :math:`\varpi_c`: the pore-volume share of component :math:`c`, the weight with which it participates in the storage matrix :math:`M_\phi` and hence in the eigenproblem.
    - :math:`\phi_e, V_e`: the reference porosity and the volume of cell :math:`e`; :math:`(Jw)_{eg}`: the Gauss-point Jacobian-weight product ``JxW``.
    - components below ``min_comp_vol_frac`` are dropped from the *mode budget* only — their cells remain in every loss group, they are simply not addressed.
    """

    n_components: int
    max_node_reuse: int
    cell_comp: Any               # (n_cells,) component id per cell
    node_comp: Any               # (n_nodes,) component id per node
    comp_vol_frac: Any           # (n_components,) pore-volume share, descending id order
    kept: Any                    # (n_components,) bool: participates in the mode budget
    conforming: bool             # max_node_reuse >= spec.min_node_reuse

    @property
    def n_kept(self) -> int:
        return int(onp.asarray(self.kept).sum())

    @property
    def pruned_vol_frac(self) -> float:
        f = onp.asarray(self.comp_vol_frac)
        return float(f[~onp.asarray(self.kept)].sum())

    def summary(self) -> str:
        """One-line human-readable digest for warnings and error messages."""
        f = onp.sort(onp.asarray(self.comp_vol_frac))[::-1]
        head = ", ".join(f"{x:.4f}" for x in f[:6]) + (", ..." if f.size > 6 else "")
        return (f"{self.n_components} component(s), max node reuse {self.max_node_reuse} "
                f"(conforming needs >= 8), pore-volume shares [{head}], "
                f"{self.n_kept} kept for the mode budget")


def mesh_admissibility(static: dict, a_lap, min_comp_vol_frac: float,
                       min_node_reuse: int) -> MeshAdmissibility:
    """Conformity + connectivity + per-component pore volume of the assembled mesh."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    hex_nodes = onp.asarray(static["hex_nodes"])
    n_nodes = int(static["n_vertices"])
    max_reuse = int(onp.bincount(hex_nodes.reshape(-1), minlength=n_nodes).max())

    # Same graph count_null_modes uses: explicit zeros dropped so voided couplings
    # (degenerate elements) do not register as edges.
    a = sp.csr_matrix(a_lap).copy()
    a.eliminate_zeros()
    n_comp, node_lab = connected_components(a, directed=False)
    # every node of a cell shares its component (a cell couples all eight)
    cell_comp = node_lab[hex_nodes[:, 0]]

    pore = onp.asarray(static["poro_hex"]) * onp.asarray(static["JxW"]).sum(axis=1)
    vol = onp.bincount(cell_comp, weights=pore, minlength=n_comp)
    frac = vol / max(vol.sum(), 1e-30)
    return MeshAdmissibility(
        n_components=int(n_comp), max_node_reuse=max_reuse, cell_comp=cell_comp,
        node_comp=node_lab, comp_vol_frac=frac, kept=frac >= min_comp_vol_frac,
        conforming=(min_node_reuse <= 0 or max_reuse >= min_node_reuse),
    )


class SpectralError(ConfigError):
    """The provisioned encoding cannot address the reservoir it was built for.

    A :class:`~pinnlab.config.ConfigError` subclass so the notebook and ``sweep``
    error paths catch it exactly like any other unrunnable composition — the run
    never starts and no ``runs.csv`` row is produced, because this is a
    precondition on the encoding rather than an outcome of training.
    """


def encoder_adequacy(v_c, *, tol: float = 0.01, cell_comp=None, kept=None) -> dict:
    r"""
    Whether the standardized centroid features can address every cell.

    Two failure modes, both fatal and neither visible in the loss. The map
    :math:`\mathbf{x}_c \mapsto \tilde{\mathbf v}(\mathbf{x}_c)` must be

    **injective** — distinct cells must receive distinct feature vectors, or the
    network is being asked to predict different states from identical inputs, and

    **non-degenerate** — no cell may sit at the origin of the (centred) feature
    space while others span it, which is what a component-localized mode set does
    to every component it does not cover.

    Injectivity is tested on the lattice quantization

    .. math::

        q_{cm} \;=\; \operatorname{round}\!\left(
        \frac{\tilde v_{cm}}{\varepsilon\,\overline{\sigma}}\right),
        \qquad
        \overline{\sigma} \;=\; \frac{1}{n_\lambda}\sum_m \operatorname{sd}_c\bigl(\tilde v_{cm}\bigr)

    and degeneracy on the centred amplitude
    :math:`a_c = \lVert \tilde{\mathbf v}_c - \overline{\tilde{\mathbf v}} \rVert
    \,/\, \max_{c'} \lVert \tilde{\mathbf v}_{c'} - \overline{\tilde{\mathbf v}} \rVert`.

    where:
    - :math:`\varepsilon` (``tol``): the quantization step as a fraction of the mean per-mode feature spread; it sets the injectivity bar directly, so it also sets the retention floor that :func:`~pinnlab.spectral.provision` may have to widen :math:`n_\lambda` to reach.
    - :math:`\overline{\sigma}`: the mean per-mode standard deviation over cells, making the test scale-free.
    - :math:`a_c`: the relative feature amplitude of cell :math:`c`; cells with :math:`a_c < 10^{-2}` are counted *blind* — the encoding cannot distinguish them from the field mean.
    - ``cell_comp`` / ``kept``: optional component labels and a keep mask; when given, the statistics are restricted to cells of kept components, so pruned (negligible pore volume) components are not held against the basis.

    Returns ``{distinct, n_cells, addressable_frac, blind_frac, per_component}``.
    """
    v = onp.asarray(v_c, onp.float64)
    sel = slice(None)
    if cell_comp is not None and kept is not None:
        mask = onp.asarray(kept)[onp.asarray(cell_comp)]
        sel = onp.flatnonzero(mask)

    def _stats(block):
        if block.shape[0] == 0:
            return {"distinct": 0, "n_cells": 0, "addressable_frac": 1.0, "blind_frac": 0.0}
        scale = float(block.std(axis=0).mean())
        q = onp.round(block / max(scale * tol, 1e-30)).astype(onp.int64)
        distinct = int(len(onp.unique(q, axis=0)))
        amp = onp.linalg.norm(block - block.mean(axis=0), axis=1)
        amp = amp / max(float(amp.max()), 1e-30)
        return {"distinct": distinct, "n_cells": int(block.shape[0]),
                "addressable_frac": distinct / block.shape[0],
                "blind_frac": float((amp < 1e-2).mean())}

    out = _stats(v[sel])
    per: dict[int, dict] = {}
    if cell_comp is not None:
        comp = onp.asarray(cell_comp)
        keep_ids = (onp.flatnonzero(onp.asarray(kept)) if kept is not None
                    else onp.unique(comp))
        for c in keep_ids:
            per[int(c)] = _stats(v[comp == c])
    out["per_component"] = per
    return out


@dataclass
class SpectralBundle:
    """Static FEM operators + (optionally) the eigenbasis, host-side until placed."""

    static: Any                 # hex_fem_assembly_jax STATIC dict
    a_lap: Any                  # sparse stiffness (weighting-keyed)
    m_lap: Any                  # sparse consistent storage mass (porosity-weighted)
    n_null: int                 # constant modes (one per connected component)
    k_hex_diag: Any             # (n_hex, 3) per-cell permeability
    node_xyz: Any               # (n_nodes, 3) f32
    node_z: Any                 # (n_nodes,)
    vlump: Any                  # (n_nodes,) lumped GEOMETRIC node volumes [rb]
    vft3: Any                   # (n_nodes,) [ft^3]
    poro_nodes: Any             # (n_nodes,) volume-weighted nodal reference porosity
    centroids: Any              # (n_cells, 3) f32
    admissibility: Any = None   # MeshAdmissibility (conformity, components, pore volume)
    # eigenbasis tier (None when not provisioned)
    lam: Any = None             # (n_eig,) vertical block first, ascending lambda per block
    v_nodes: Any = None         # (n_nodes, n_eig) standardized
    v_c: Any = None             # (n_cells, n_eig) centroid features
    b_v: Any = None             # (n_cells, 3, n_eig) physical gradients
    v_mean: Any = None
    v_std: Any = None
    mu: Any = None              # (n_eig,) spectral residual metric weights
    upsilon: Any = None         # (n_eig,) per-mode vertical variance fraction
    n_eig_z: int = 0            # resolved vertical reservation used at selection
    adequacy: Any = None        # encoder_adequacy() dict for the retained basis


def resolve_n_eig_z(cfg: RunConfig, n_levels: int) -> int:
    r"""
    The resolved vertical-mode reservation for one case.

    ``cfg.n_eig_z > 0`` is honored up to the geometric ceiling; ``0`` selects
    the automatic reservation

    .. math::

        n_z^{\mathrm{res}} \;=\; \min\bigl(n_{\mathrm{lev}} - 1,\ \max(1,\ \lfloor n_\lambda / 8 \rfloor)\bigr)

    where:
    - :math:`n_{\mathrm{lev}}`: the number of node levels per vertical column — a column of :math:`n_{\mathrm{lev}}` nodes supports exactly :math:`n_{\mathrm{lev}} - 1` linearly independent vertical overtones.
    - :math:`n_\lambda`: the retained basis width (``cfg.n_eig``); the :math:`1/8` fraction keeps the reservation a small block of the budget.
    """
    ceiling = max(n_levels - 1, 0)
    if ceiling == 0:
        return 0
    if cfg.n_eig_z > 0:
        return min(cfg.n_eig_z, ceiling, cfg.n_eig - 1)
    return min(ceiling, max(1, cfg.n_eig // 8))


def node_column_ids(verts, column_tol_ft: float = _COLUMN_TOL_FT) -> tuple[onp.ndarray, int]:
    """Vertical-column id per node (nodes sharing an areal position) and the level count."""
    xy = onp.round(onp.asarray(verts, onp.float64)[:, :2] / column_tol_ft).astype(onp.int64)
    _, col = onp.unique(xy, axis=0, return_inverse=True)
    n_levels = int(onp.bincount(col).max()) if col.size else 0
    return col.astype(onp.int64), n_levels


def vertical_variance_fraction(v: onp.ndarray, col_id: onp.ndarray) -> onp.ndarray:
    r"""
    Per-mode vertical variance fraction :math:`\upsilon_m` (columnwise ANOVA).

    .. math::

        \upsilon_m \;=\; \frac{\sum_i \bigl(v_{im} - \mu_{c(i), m}\bigr)^2}{\sum_i \bigl(v_{im} - \bar v_m\bigr)^2}

    where:
    - :math:`\mu_{c,m}`: the mean of mode :math:`m` over the nodes of column :math:`c`; :math:`c(i)` the column of node :math:`i`.
    - :math:`\bar v_m`: the nodal mean of mode :math:`m`; the ratio is 0 for a purely areal mode and 1 for a purely vertical one.
    """
    v = onp.asarray(v, onp.float64)
    n_cols = int(col_id.max()) + 1 if col_id.size else 0
    counts = onp.bincount(col_id, minlength=n_cols).astype(onp.float64)
    sums = onp.zeros((n_cols, v.shape[1]))
    onp.add.at(sums, col_id, v)
    within = v - (sums / counts[:, None])[col_id]
    ss_w = (within ** 2).sum(axis=0)
    ss_t = ((v - v.mean(axis=0)) ** 2).sum(axis=0)
    return ss_w / onp.maximum(ss_t, 1e-30)


def _select_retained(lam_c: onp.ndarray, ups_c: onp.ndarray, n_eig: int,
                     n_zres: int, upsilon_vertical: float = _UPSILON_VERTICAL,
                     n_forced: int = 0) -> onp.ndarray:
    r"""Candidate indices to retain: leading band amended with the vertical family,
    ordered vertical block first (ascending eigenvalue within each block).

    ``n_forced`` reserves the first ``n_forced`` pool columns unconditionally, for the
    deflation strategy whose vertical family is *constructed* rather than found. Those
    modes sit far above the areal band in :math:`\lambda` — that is the whole pancake
    problem — so an eigenvalue-ordered competition would discard them in favour of
    areal modes that merely stray over the :math:`\upsilon` threshold.
    """
    order = onp.argsort(lam_c, kind="stable")
    is_vert = ups_c > upsilon_vertical
    if n_forced > 0:
        forced = list(range(min(n_forced, max(n_zres, 1))))
        rest = [int(i) for i in order if int(i) not in set(forced)]
        base = forced + rest[:max(n_eig - len(forced), 0)]
        sel = onp.asarray(base, onp.int64)
        sel_v, sel_a = sel[is_vert[sel]], sel[~is_vert[sel]]
        sel_v = sel_v[onp.argsort(lam_c[sel_v], kind="stable")]
        sel_a = sel_a[onp.argsort(lam_c[sel_a], kind="stable")]
        return onp.concatenate([sel_v, sel_a])
    base = list(order[:n_eig])
    short = n_zres - int(is_vert[base].sum())
    if short > 0:
        extra = [int(i) for i in order[n_eig:] if is_vert[i]][:short]
        # swap out the HIGHEST-λ areal picks, one per gained vertical mode
        areal_desc = [int(i) for i in reversed(base) if not is_vert[i]][:len(extra)]
        base = [i for i in base if int(i) not in set(areal_desc)] + extra
    sel = onp.asarray(base, onp.int64)
    sel_v = sel[is_vert[sel]]
    sel_a = sel[~is_vert[sel]]
    sel_v = sel_v[onp.argsort(lam_c[sel_v], kind="stable")]
    sel_a = sel_a[onp.argsort(lam_c[sel_a], kind="stable")]
    return onp.concatenate([sel_v, sel_a])


def _candidate_spectrum(a_lap, m_lap, cfg: RunConfig, n_nodes: int, n_null: int,
                        col_id: onp.ndarray, n_zres: int, n_eig: int | None = None):
    """(lam_c, V_c, ups_c) over a candidate band guaranteed (best-effort) to reach
    the vertical family: dense full spectrum on small meshes, expanding
    shift-invert Lanczos band on large ones."""
    from modules.utils.spectral_basis_jax import (generalized_eig_basis_dense_full,
                                          generalized_eig_basis_sparse)

    spec = cfg.spec
    n_eig = cfg.n_eig if n_eig is None else int(n_eig)
    n_avail = n_nodes - n_null - 1
    if n_nodes <= spec.dense_eig_max_n:
        lam_c, v_c = generalized_eig_basis_dense_full(a_lap, m_lap, drop_null=True,
                                                      n_null=n_null)
        return lam_c, v_c, vertical_variance_fraction(v_c, col_id), 0

    if spec.retention == "deflation" and n_zres > 0:
        band = _deflation_candidates(a_lap, m_lap, cfg, n_avail, col_id, n_zres, n_eig)
        if band is not None:
            return band
        warnings.warn(
            "deflation retention did not converge (LOBPCG on the column-deflated "
            "operator); falling back to the band scan. Its vertical guarantee is "
            "best-effort within spec.band_cap_mult * n_eig candidates.",
            RuntimeWarning, stacklevel=2)

    k = min(n_avail, max(2 * n_eig, n_eig + 32))
    k_cap = min(n_avail, spec.band_cap_mult * n_eig)
    while True:
        lam_c, v_c = generalized_eig_basis_sparse(a_lap, m_lap, k, which="SM",
                                                  drop_null=True, n_null=n_null)
        ups_c = vertical_variance_fraction(v_c, col_id)
        if int((ups_c > spec.upsilon_vertical).sum()) >= n_zres or k >= k_cap:
            return lam_c, v_c, ups_c, 0
        k = min(k_cap, 2 * k)


def addressability_floor(cfg: RunConfig, lam_c: onp.ndarray, ups_c: onp.ndarray,
                         v_c_pool: onp.ndarray, n_levels: int, ceiling: int,
                         cell_sel: onp.ndarray | None = None,
                         n_forced: int = 0) -> int | None:
    r"""
    The smallest retained width at which the encoding addresses every cell.

    Searched over the *selection rule*, not over a fixed column order: the vertical
    reservation is itself a function of the width,
    :math:`n_z^{\mathrm{res}}(n_\lambda) = \min(n_{\mathrm{lev}}-1,\ \max(1, \lfloor n_\lambda/8\rfloor))`,
    so widening the basis can change *which* modes are retained, not merely how many.
    Each trial width therefore re-runs :func:`_select_retained` from the candidate pool
    rather than extending a prefix.

    The floor is set by the domain's symmetry rather than by its size — a mode is a
    real-valued function over the whole component, so a handful of them separates
    thousands of cells — with a weak (logarithmic) dependence on cell count through the
    quantization tolerance. Measured: 3 modes for SPE1CASE1's 300 cells, 10 for
    SPE9_CP's 9000.

    where:
    - ``lam_c``, ``ups_c``, ``v_c_pool``: the candidate eigenvalues, vertical variance fractions, and *standardized* centroid features of the whole candidate pool; column slicing is exact because standardization is per-column affine and centroid interpolation is linear with a partition of unity.
    - ``ceiling``: the largest width to try (``spec.n_eig_cap``, bounding the network input width).
    - ``cell_sel``: optional row selection restricting the test to kept components' cells.

    Returns the floor, or ``None`` when no width up to ``ceiling`` suffices.
    """
    spec = cfg.spec
    pool = v_c_pool if cell_sel is None else v_c_pool[cell_sel]
    for n in range(1, int(ceiling) + 1):
        n_z = resolve_n_eig_z(_dc_replace(cfg, n_eig=n, n_eig_z=cfg.n_eig_z), n_levels)
        sel = _select_retained(lam_c, ups_c, n, n_z, spec.upsilon_vertical, n_forced)
        if sel.size < n:
            continue                      # candidate pool too narrow to realize this width
        stats = encoder_adequacy(pool[:, sel], tol=spec.addressability_tol)
        if (stats["addressable_frac"] >= spec.addressability_min
                and stats["blind_frac"] <= spec.blind_frac_max):
            return n
    return None


def _stratify(cfg: RunConfig, adm: MeshAdmissibility) -> bool:
    """Whether to take the per-component path.

    Off unless explicitly requested *and* there is more than one kept component: on a
    connected mesh the stratified and global paths coincide mathematically, so the
    guard keeps the single-component case on the byte-identical original route.
    """
    return bool(cfg.spec.stratify_components and adm.n_kept > 1)


def column_indicator_basis(col_id: onp.ndarray):
    r"""
    Sparse basis :math:`Y` of the *column-constant* subspace, one column per areal
    node column.

    :math:`\operatorname{range}(Y) = \operatorname{range}(\Pi_z)`, the fields that do
    not vary with depth at fixed areal position, so its orthogonal complement is
    exactly the vertical-variation subspace of :math:`\upsilon_m = 1`.
    """
    import scipy.sparse as sp

    n = col_id.size
    n_col = int(col_id.max()) + 1 if n else 0
    return sp.csr_matrix((onp.ones(n), (onp.arange(n), col_id)), shape=(n, n_col))


def _vertical_family_by_deflation(a_lap, m_lap, col_id: onp.ndarray, n_want: int,
                                  tol: float = 1e-6, maxiter: int = 400):
    r"""
    The lowest vertical modes, obtained by deflating the column-constant subspace.

    The band scan hunts for :math:`\upsilon_m > \tfrac12` inside a lowest-:math:`\lambda`
    band capped at ``band_cap_mult`` :math:`\times\,n_\lambda`, but the depth at which
    the vertical family sits is a property of the *geometry*,

    .. math::

        \frac{\lambda_{\mathrm{vert},1}}{\lambda_{\mathrm{areal},1}}
        \;\approx\;
        \frac{k_z}{k_x}\left(\frac{L_x}{H_z}\right)^{2},

    not of the requested basis width, so the cap can miss it on a strongly pancaked
    domain however large it is made. Solving the eigenproblem restricted to
    :math:`\operatorname{range}(I - \Pi_z)` removes the mismatch: every mode of the
    deflated problem varies with depth *by construction*, so the lowest ones are the
    vertical family, found without scanning the areal spectrum at all.

    where:
    - :math:`\Pi_z`: the columnwise averaging projector; :math:`Y` (:func:`column_indicator_basis`) spans its range, and LOBPCG's constraint argument keeps the iterates :math:`M_\phi`-orthogonal to it.
    - :math:`k_z/k_x`: the vertical-to-areal permeability contrast; :math:`L_x, H_z`: areal extent and thickness.
    - ``n_want``: how many vertical modes to return; ``tol``/``maxiter``: LOBPCG convergence controls.

    Returns ``(lam, V)`` ascending, or ``None`` when the solve does not converge —
    LOBPCG for smallest eigenvalues is preconditioner-sensitive, so the caller falls
    back to the band scan rather than proceeding on an unconverged basis.
    """
    import numpy.linalg as npl
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

    n = a_lap.shape[0]
    y = column_indicator_basis(col_id).toarray()
    k = min(int(n_want), max(n - y.shape[1] - 1, 1))
    if k < 1:
        return None
    rng = onp.random.default_rng(0)
    x = rng.normal(size=(n, k))
    x -= y @ npl.lstsq(y, x, rcond=None)[0]          # start inside the deflated subspace
    # Jacobi preconditioner: LOBPCG's convergence to the smallest eigenvalues is poor
    # without one on an operator whose diagonal spans orders of magnitude in kappa.
    diag = onp.asarray(a_lap.diagonal(), onp.float64)
    prec = sla.LinearOperator(
        (n, n), matvec=lambda v: v / onp.where(diag > 0, diag, 1.0), dtype=onp.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # LOBPCG chatter on hard problems
        try:
            lam, v = sla.lobpcg(sp.csr_matrix(a_lap).astype(onp.float64),
                                x, B=sp.csr_matrix(m_lap).astype(onp.float64),
                                M=prec, Y=y, tol=tol, maxiter=maxiter, largest=False)
        except Exception:
            return None
    if lam is None or onp.asarray(lam).size == 0 or not onp.all(onp.isfinite(lam)):
        return None
    order = onp.argsort(onp.asarray(lam))
    return onp.asarray(lam)[order], onp.asarray(v)[:, order]


def _deflation_candidates(a_lap, m_lap, cfg: RunConfig, n_avail: int,
                          col_id: onp.ndarray, n_zres: int, n_eig: int):
    """Candidate pool assembled as (deflated vertical family) + (lowest-lambda areal band).

    The two halves answer different questions, so they are solved separately rather
    than hunted for in one ordering: the vertical block comes from the deflated
    operator, where depth-varying is guaranteed, and the areal remainder from the usual
    shift-invert band. Returns ``None`` if the deflated solve fails, leaving the caller
    to fall back.
    """
    from modules.utils.spectral_basis_jax import generalized_eig_basis_sparse

    got = _vertical_family_by_deflation(a_lap, m_lap, col_id, n_zres)
    if got is None:
        return None
    lam_v, v_v = got
    ups_v = vertical_variance_fraction(v_v, col_id)
    if int((ups_v > cfg.spec.upsilon_vertical).sum()) < n_zres:
        return None                       # deflation did not actually deliver the family

    k = min(n_avail, max(n_eig, 1))
    lam_a, v_a = generalized_eig_basis_sparse(a_lap, m_lap, k, which="SM",
                                              drop_null=True, n_null=None)
    ups_a = vertical_variance_fraction(v_a, col_id)
    lam = onp.concatenate([onp.asarray(lam_v), onp.asarray(lam_a)])
    v = onp.concatenate([onp.asarray(v_v), onp.asarray(v_a)], axis=1).astype(onp.float32)
    # the deflated family leads the pool and is retained unconditionally: it sits far
    # above the areal band in lambda, so an eigenvalue-ordered contest would drop it.
    return lam, v, onp.concatenate([ups_v, ups_a]), int(lam_v.size)


def allocate_component_modes(n_eig: int, vol_frac: onp.ndarray) -> onp.ndarray:
    r"""
    Split a mode budget across connected components by pore-volume share.

    .. math::

        n_c \;=\; \max\!\left(1,\ \left\lfloor n_\lambda\, \frac{\varpi_c}{\sum_{c'} \varpi_{c'}} \right\rfloor\right),
        \qquad \text{then trimmed/topped-up so } \sum_c n_c = n_\lambda

    where:
    - :math:`\varpi_c`: the component's pore-volume share (:class:`MeshAdmissibility`), restricted here to the kept components.
    - :math:`n_c`: its mode allocation; the floor of 1 guarantees that no kept component is left entirely unaddressed, and the remainder is handed to the largest components first.

    A budget drawn by *global* eigenvalue order gives no such guarantee: the
    eigenvectors of a block-diagonal operator each live on one component, and the
    lowest eigenvalues may all belong to the same block.
    """
    frac = onp.asarray(vol_frac, onp.float64)
    n_comp = frac.size
    if n_comp == 0:
        return onp.zeros(0, onp.int64)
    if n_eig < n_comp:
        raise SpectralError(
            f"n_eig={n_eig} cannot cover {n_comp} kept components (one mode each is the "
            "floor); raise n_eig, raise spec.min_comp_vol_frac to prune more aggressively, "
            "or repair a mesh that should not be fragmented")
    share = frac / max(frac.sum(), 1e-30)
    alloc = onp.maximum(1, onp.floor(n_eig * share).astype(onp.int64))
    order = onp.argsort(share, kind="stable")[::-1]        # largest share first
    while alloc.sum() > n_eig:                             # trim the smallest above floor
        for i in order[::-1]:
            if alloc[i] > 1:
                alloc[i] -= 1
                break
    i = 0
    while alloc.sum() < n_eig:                             # hand remainder to the largest
        alloc[order[i % order.size]] += 1
        i += 1
    return alloc


def _stratified_spectrum(a_lap, m_lap, cfg: RunConfig, adm: MeshAdmissibility,
                         col_id: onp.ndarray, n_nodes: int, alloc: onp.ndarray,
                         comp_ids: onp.ndarray):
    r"""
    Solve each kept component's own sub-spectrum and merge the selections.

    Restricting :math:`(A, M_\phi)` to a component's node set and solving there is not
    merely a cheaper route to the same band — it is the only way to *guarantee* per
    component coverage, since a globally eigenvalue-ordered band can draw every mode
    from one block. Each component is connected by construction, so its sub-problem has
    exactly one null mode, and its vertical reservation is resolved against its own
    node-level count. Scattering the sub-eigenvectors back to full length with zeros
    outside the component preserves :math:`M_\phi`-orthonormality, because
    :math:`M_\phi` is block-diagonal in the same partition — so the Galerkin projection
    identities of the residual are untouched.

    Returns ``(lam, V, ups)`` for the merged retained set, ordered vertical block first.
    """
    import scipy.sparse as sp

    a_csr = sp.csr_matrix(a_lap)
    m_csr = sp.csr_matrix(m_lap)
    node_comp = onp.asarray(adm.node_comp)

    lam_parts, v_parts, ups_parts = [], [], []
    for comp, n_c in zip(comp_ids, alloc):
        idx = onp.flatnonzero(node_comp == comp)
        sub_a = a_csr[idx][:, idx]
        sub_m = m_csr[idx][:, idx]
        sub_col = onp.unique(col_id[idx], return_inverse=True)[1]
        n_lev = int(onp.bincount(sub_col).max()) if sub_col.size else 0
        n_z = resolve_n_eig_z(_dc_replace(cfg, n_eig=int(n_c)), n_lev)
        lam_s, v_s, ups_s, n_forced = _candidate_spectrum(
            sub_a, sub_m, cfg, int(idx.size), 1, sub_col, n_z, n_eig=int(n_c))
        sel = _select_retained(lam_s, ups_s, int(n_c), n_z, cfg.spec.upsilon_vertical,
                               n_forced)
        v_full = onp.zeros((n_nodes, sel.size), onp.float64)
        v_full[idx] = onp.asarray(v_s)[:, sel]
        lam_parts.append(onp.asarray(lam_s)[sel])
        ups_parts.append(onp.asarray(ups_s)[sel])
        v_parts.append(v_full)

    lam = onp.concatenate(lam_parts)
    ups = onp.concatenate(ups_parts)
    v = onp.concatenate(v_parts, axis=1)
    # global vertical-block-first ordering, ascending lambda within each block
    is_v = ups > cfg.spec.upsilon_vertical
    iv = onp.flatnonzero(is_v)[onp.argsort(lam[is_v], kind="stable")]
    ia = onp.flatnonzero(~is_v)[onp.argsort(lam[~is_v], kind="stable")]
    order = onp.concatenate([iv, ia])
    return lam[order], v[:, order].astype(onp.float32), ups[order]


def _cache_locator(case: CaseData, cfg: RunConfig):
    """(cache_dir, operator key, stiffness weighting) for this case/config."""
    import modules.utils.spectral_cache as sc

    model_path, prep_cache = case_paths(cfg)
    weighting = (cfg.stiffness_design.value if cfg.stiffness_design is not None
                 else "perm_weighted")
    corner = onp.asarray(case.art.reservoir_mesh.corner_cells)
    op_key = sc.operator_cache_key(
        model_path=model_path, laplacian_weighting=weighting,
        n_nodes=int(case.n_nodes), n_hex=int(case.n_cells),
        fingerprint=sc.fem_input_fingerprint(case.verts, case.hexes, case.perms,
                                             case.poro_cell, corner),
    )
    return sc.default_spectral_cache_dir(prep_cache), op_key, weighting


def resolve_n_eig(cfg: RunConfig, case: CaseData) -> tuple[RunConfig, list[str]]:
    r"""
    Resolve ``n_eig`` against the addressability floor of this reservoir.

    ``n_eig = 0`` adopts the floor; a positive value below it is widened (with a
    warning) rather than left to fail, since a basis that cannot separate the cells
    makes the run meaningless whatever else is configured; a floor above
    ``spec.n_eig_cap`` raises, because the network input width :math:`d = n_\lambda + 1`
    cannot grow indefinitely and a demand that large means the *mesh* is wrong — a
    fragmented domain needs a few modes per component, so 24 slabs would ask for
    roughly 50–70 inputs.

    Must run before :func:`~pinnlab.config.resolve`, which fixes ``dim_in`` and the
    parameter count: widening the basis widens the network input, and the auto-capacity
    rule reads that width. Returns ``(cfg, warnings)`` and leaves a matching eigenbasis
    in the cache, so the subsequent :func:`provision` is a cache hit.
    """
    import modules.utils.hex_fem_assembly_jax as hf

    notes: list[str] = []
    if not needs_eigenbasis(cfg) or cfg.spec.addressability_min <= 0.0:
        return (cfg if cfg.n_eig > 0 else _dc_replace(cfg, n_eig=_FALLBACK_N_EIG)), notes

    bundle = provision(case, cfg, _resolving=True)
    adm = bundle.admissibility
    col_id, n_levels = node_column_ids(case.verts, cfg.spec.column_tol_ft)
    ceiling = int(cfg.spec.n_eig_cap)

    probe = _dc_replace(cfg, n_eig=ceiling)
    n_z_top = resolve_n_eig_z(probe, n_levels)
    lam_c, v_cand, ups_c, n_forced = _candidate_spectrum(
        bundle.a_lap, bundle.m_lap, probe, int(case.n_nodes), int(bundle.n_null),
        col_id, n_z_top, n_eig=ceiling)
    v_std_pool = (v_cand - v_cand.mean(axis=0)) / (v_cand.std(axis=0) + 1e-8)
    v_c_pool, _ = hf.centroid_features_and_gradients(v_std_pool, bundle.static)
    v_c_pool = onp.asarray(v_c_pool)

    kept_cells = onp.asarray(adm.kept)[onp.asarray(adm.cell_comp)]
    floor = addressability_floor(cfg, lam_c, ups_c, v_c_pool, n_levels, ceiling,
                                 cell_sel=onp.flatnonzero(kept_cells),
                                 n_forced=n_forced)
    if floor is None:
        raise SpectralError(
            f"{case_label(cfg)}: no basis width up to spec.n_eig_cap={ceiling} "
            f"addresses this mesh (network input width d = n_eig + 1 is bounded by it). "
            f"Mesh: {adm.summary()}. A demand this large means the mesh is the problem, "
            "not the budget — a fragmented domain needs a few modes per component, so "
            "24 slabs ask for roughly 50-70 inputs. Repair the geometry, raise "
            "spec.min_comp_vol_frac to prune more aggressively, or relax "
            "spec.addressability_min.")

    if cfg.n_eig <= 0:
        notes.append(f"n_eig=0 (auto): resolved to the addressability floor {floor} "
                     f"for {case_label(cfg)}")
    elif cfg.n_eig < floor:
        notes.append(
            f"n_eig={cfg.n_eig} is below {case_label(cfg)}'s addressability floor "
            f"{floor}: widened to {floor}, since at {cfg.n_eig} distinct cells share a "
            "network input and no amount of training can separate them")
    else:
        return cfg, notes                  # already adequate; leave the cache alone

    out = _dc_replace(cfg, n_eig=floor)
    _persist_pool_basis(case, out, bundle, lam_c, v_cand, ups_c, n_levels, n_forced)
    return out, notes


def _persist_pool_basis(case: CaseData, cfg: RunConfig, bundle: SpectralBundle,
                        lam_c, v_cand, ups_c, n_levels: int, n_forced: int = 0) -> None:
    """Cache the retained basis cut from the *search* pool at the resolved width.

    Without this the guarantee would be nominal on large meshes: the floor is measured
    against a pool solved at ``spec.n_eig_cap`` width, while a fresh :func:`provision`
    at the resolved ``n_eig`` sizes its own (narrower) shift-invert band and can land on
    a different candidate set. Writing the validated basis to the cache makes the
    subsequent provision a hit, so the modes trained on are exactly the modes whose
    addressability was checked.
    """
    import jax.numpy as jnp

    import modules.utils.hex_fem_assembly_jax as hf
    import modules.utils.spectral_cache as sc

    cache_dir, op_key, _ = _cache_locator(case, cfg)
    n_z = resolve_n_eig_z(cfg, n_levels)
    sel = _select_retained(lam_c, ups_c, cfg.n_eig, n_z, cfg.spec.upsilon_vertical,
                           n_forced)
    v_raw = jnp.asarray(onp.asarray(v_cand)[:, sel])
    v_mean = jnp.mean(v_raw, axis=0)
    v_std = jnp.std(v_raw, axis=0) + 1e-8
    v_nodes = (v_raw - v_mean) / v_std
    v_c, b_v = hf.centroid_features_and_gradients(v_nodes, bundle.static)
    sc.save_eig_cache(cache_dir, op_key, n_eig=int(cfg.n_eig), n_eig_z=int(n_z),
                      lam=jnp.asarray(onp.asarray(lam_c)[sel]), v_nodes=v_nodes,
                      v_c=v_c, b_v=b_v, v_mean=v_mean, v_std=v_std,
                      upsilon=jnp.asarray(onp.asarray(ups_c)[sel].astype(onp.float32)),
                      sel_key=sc.eig_selection_key(cfg.spec))


def provision(case: CaseData, cfg: RunConfig, *, _resolving: bool = False) -> SpectralBundle:
    """Load-or-build the static FEM operators, then the eigenbasis when required."""
    import jax.numpy as jnp

    import modules.utils.hex_fem_assembly_jax as hf
    import modules.utils.spectral_cache as sc
    from modules.utils.spectral_basis_jax import count_null_modes

    cache_dir, op_key, weighting = _cache_locator(case, cfg)
    corner = onp.asarray(case.art.reservoir_mesh.corner_cells)

    # === (1) static FEM operators (N_EIG-independent): load or build+cache ===============
    op = sc.load_operator_cache(cache_dir, op_key)
    if op is not None:
        static, a_lap, m_lap, n_null = op["static"], op["A_lap"], op["M_lap"], op["n_null"]
    else:
        static = hf.build_static_hex_fem(case.verts, case.hexes, case.perms, case.poro_cell,
                                         corner_cells=corner)
        a_weight = (static["k_hex_diag"] if weighting == "perm_weighted"
                    else jnp.ones((static["n_hex"], 3), jnp.float32))
        a_lap = hf.assemble_stiffness_sparse(a_weight, static, field_darcy_coeff=1.0)
        m_lap = hf.assemble_mass_sparse(static["poro_hex"], static, rb_per_ft3=1.0)
        n_null = count_null_modes(a_lap)
        sc.save_operator_cache(cache_dir, op_key, static, a_lap, m_lap, int(n_null))

    vlump = hf.lumped_node_volumes(static)
    vft3 = vlump / case.rb_ft3
    # volume-weighted nodal reference porosity: (row-sum of the phi-weighted mass) / V_i
    pv = jnp.zeros((int(static["n_vertices"]),), jnp.float32).at[
        static["hex_nodes"].reshape(-1)].add(
        jnp.einsum("eg,ega->ea", static["JxW"] * static["poro_hex"][:, None],
                   static["N"]).reshape(-1))
    poro_nodes = pv / jnp.maximum(vft3, 1e-30)

    adm = mesh_admissibility(static, a_lap, cfg.spec.min_comp_vol_frac,
                             cfg.spec.min_node_reuse)
    if not adm.conforming:
        warnings.warn(
            f"{case_label(cfg)}: max node reuse is {adm.max_node_reuse}, below the "
            f"{cfg.spec.min_node_reuse} of a conforming structured hex mesh — no node is "
            "shared by eight cells, i.e. the cells do not touch along at least one grid "
            "direction. The usual cause is a block-centred (DX/DY/DZ + TOPS) deck on a "
            "dipping grid, where each cell is a flat block at its own depth and "
            "dx*tan(dip) exceeds the layer thickness; the fix is a conforming "
            "corner-point (COORD/ZCORN) deck for the same model. The FEM stiffness has "
            "no flux across the severed direction, so both the residual and the "
            f"eigenbasis are built on a domain that is not the reservoir. [{adm.summary()}]",
            RuntimeWarning, stacklevel=2)
    if adm.n_components > 1:
        warnings.warn(
            f"{case_label(cfg)}: stiffness graph is disconnected — {adm.summary()}. "
            "Eigenvectors of a block-diagonal operator live on a single component, so a "
            "globally eigenvalue-ordered band can leave whole components at constant "
            f"features; {adm.n_components - adm.n_kept} component(s) carrying "
            f"{adm.pruned_vol_frac:.4%} of the pore volume are excluded from the mode "
            "budget (their cells stay in every loss group).",
            RuntimeWarning, stacklevel=2)

    bundle = SpectralBundle(
        static=static, a_lap=a_lap, m_lap=m_lap, n_null=int(n_null),
        k_hex_diag=static["k_hex_diag"],
        node_xyz=jnp.asarray(case.verts, jnp.float32),
        node_z=jnp.asarray(case.verts, jnp.float32)[:, 2],
        vlump=vlump, vft3=vft3, poro_nodes=poro_nodes,
        centroids=jnp.asarray(case.centroids, jnp.float32),
        admissibility=adm,
    )

    # === (2) eigenbasis (column-sliceable cache): load or solve+cache =====================
    # _resolving stops here: resolve_n_eig needs only the operator tier and the
    # admissibility report, and cfg.n_eig is not yet settled.
    if needs_eigenbasis(cfg) and not _resolving:
        col_id, n_levels = node_column_ids(case.verts, cfg.spec.column_tol_ft)
        n_zres = resolve_n_eig_z(cfg, n_levels)
        sel_key = sc.eig_selection_key(cfg.spec)
        eig = sc.load_eig_cache(cache_dir, op_key, cfg.n_eig, n_zres, sel_key)
        if eig is not None:
            lam, v_nodes = eig["lam"], eig["v_nodes"]
            v_c, b_v, v_mean, v_std = eig["v_c"], eig["b_v"], eig["v_mean"], eig["v_std"]
            upsilon = eig["upsilon"]
        elif _stratify(cfg, adm):
            comp_ids = onp.flatnonzero(onp.asarray(adm.kept))
            alloc = allocate_component_modes(
                cfg.n_eig, onp.asarray(adm.comp_vol_frac)[comp_ids])
            lam_s, v_s, ups_s = _stratified_spectrum(
                a_lap, m_lap, cfg, adm, col_id, int(case.n_nodes), alloc, comp_ids)
            lam = jnp.asarray(lam_s)
            upsilon = jnp.asarray(ups_s.astype(onp.float32))
            v_raw = jnp.asarray(v_s)
            v_mean = jnp.mean(v_raw, axis=0)
            v_std = jnp.std(v_raw, axis=0) + 1e-8
            v_nodes = (v_raw - v_mean) / v_std
            v_c, b_v = hf.centroid_features_and_gradients(v_nodes, static)
            sc.save_eig_cache(cache_dir, op_key, n_eig=int(cfg.n_eig), n_eig_z=int(n_zres),
                              lam=lam, v_nodes=v_nodes, v_c=v_c, b_v=b_v,
                              v_mean=v_mean, v_std=v_std, upsilon=upsilon,
                              sel_key=sel_key)
        else:
            lam_c, v_cand, ups_c, n_forced = _candidate_spectrum(
                a_lap, m_lap, cfg, int(case.n_nodes), int(n_null), col_id, n_zres)
            sel = _select_retained(lam_c, ups_c, cfg.n_eig, n_zres,
                                   cfg.spec.upsilon_vertical, n_forced)
            lam = jnp.asarray(lam_c[sel])
            upsilon = jnp.asarray(ups_c[sel].astype(onp.float32))
            v_raw = jnp.asarray(v_cand[:, sel])
            v_mean = jnp.mean(v_raw, axis=0)
            v_std = jnp.std(v_raw, axis=0) + 1e-8
            v_nodes = (v_raw - v_mean) / v_std
            v_c, b_v = hf.centroid_features_and_gradients(v_nodes, static)
            sc.save_eig_cache(cache_dir, op_key, n_eig=int(cfg.n_eig), n_eig_z=int(n_zres),
                              lam=lam, v_nodes=v_nodes, v_c=v_c, b_v=b_v,
                              v_mean=v_mean, v_std=v_std, upsilon=upsilon,
                              sel_key=sel_key)
        n_vert = int((onp.asarray(upsilon) > cfg.spec.upsilon_vertical).sum())
        if n_levels > 1 and n_vert < n_zres:
            warnings.warn(
                f"{case_label(cfg)}: retained basis holds {n_vert} vertical modes "
                f"(υ > {cfg.spec.upsilon_vertical}) against a reservation of {n_zres} on a mesh "
                f"with {n_levels} node levels — the candidate band never reached the "
                "vertical family; raise n_eig or n_eig_z (a z-blind basis cannot "
                "represent, supervise, or invert layered saturation structure)",
                RuntimeWarning, stacklevel=2)
        adq = encoder_adequacy(v_c, tol=cfg.spec.addressability_tol,
                               cell_comp=adm.cell_comp, kept=adm.kept)
        _assert_encoder_adequate(cfg, adm, adq, n_zres)
        bundle = _dc_replace(bundle, lam=jnp.asarray(lam), v_nodes=v_nodes, v_c=v_c, b_v=b_v,
                             v_mean=v_mean, v_std=v_std, mu=_mu_weights(cfg, jnp.asarray(lam)),
                             upsilon=jnp.asarray(upsilon), n_eig_z=int(n_zres),
                             adequacy=adq)

    return bundle


def _assert_encoder_adequate(cfg: RunConfig, adm: MeshAdmissibility, adq: dict,
                             n_zres: int) -> None:
    """Raise :class:`SpectralError` when the retained basis cannot address the domain."""
    spec = cfg.spec
    failures = []
    if adq["addressable_frac"] < spec.addressability_min:
        failures.append(
            f"only {adq['distinct']}/{adq['n_cells']} cells "
            f"({adq['addressable_frac']:.3f}) carry distinct feature vectors at a "
            f"quantization of {spec.addressability_tol:g} sd, below the required "
            f"{spec.addressability_min:g} — distinct cells share one network input, so "
            "no amount of training can give them different states")
    if adq["blind_frac"] > spec.blind_frac_max:
        failures.append(
            f"{adq['blind_frac']:.3f} of cells sit below 1% of the maximum feature "
            f"amplitude (allowed {spec.blind_frac_max:g}) — the encoding cannot "
            "distinguish them from the field mean")
    if not failures:
        return

    worst = sorted(adq["per_component"].items(),
                   key=lambda kv: kv[1]["addressable_frac"])[:4]
    per = "; ".join(f"comp {c}: {s['distinct']}/{s['n_cells']} distinct, "
                    f"{s['blind_frac']:.3f} blind" for c, s in worst)
    raise SpectralError(
        f"{case_label(cfg)}: the retained eigenbasis cannot address this mesh at "
        f"n_eig={cfg.n_eig} (n_eig_z={n_zres}). "
        + " Also, ".join(failures)
        + f". Mesh: {adm.summary()}."
        + (f" Worst components — {per}." if per else "")
        + " Raise n_eig (the addressability floor is usually a handful of modes per "
          "component), or fix the mesh if it is fragmented: a disconnected stiffness "
          "graph localizes every eigenmode on one component. Set "
          "spec.addressability_min / spec.blind_frac_max to relax this gate."
    )


def _mu_weights(cfg: RunConfig, lam):
    """Diagonal spectral-metric weights for the Galerkin residual coefficients."""
    import jax.numpy as jnp

    if cfg.spectral_mu == "lambda":
        return lam
    if cfg.spectral_mu == "inv_one_plus_lambda":
        return 1.0 / (1.0 + lam)
    return jnp.ones_like(lam)


def raw_modes(bundle: SpectralBundle):
    r"""
    De-standardized eigenmodes :math:`V_{\mathrm{raw}} = V\,\sigma_V + \bar V`
    (columns of the actual generalized eigenvectors) for the Galerkin inner
    product — the standardized ``v_nodes`` are network features, not modes.
    """
    return bundle.v_nodes * bundle.v_std + bundle.v_mean


def raw_centroid_modes(bundle: SpectralBundle):
    """De-standardized centroid features (chain_rule Galerkin quadrature)."""
    return bundle.v_c * bundle.v_std + bundle.v_mean


def place_on_mesh(bundle: SpectralBundle, cfg: RunConfig, env: MeshEnv) -> SpectralBundle:
    """
    Device placement: chain_rule gathers a single row per collocation point, so
    ``v_c``/``b_v``/``centroids`` are replicated (local gathers, no collectives);
    the whole-mesh node arrays are sharded on the node axis along 'data' for the
    fem_nodal evaluator, whose scatter-add reduces with one (n,)-sized
    all-reduce per phase.

    Axes that do not divide the data mesh dim stay replicated: an explicit
    ``device_put`` rejects uneven shards (``IndivisibleError``), while the
    jit-side ``with_sharding_constraint`` hints in the evaluator/residual keep
    the *compute* data-sharded regardless — GSPMD pads the last shard
    internally, so uneven meshes cost a few replicated MB, not correctness.
    """
    import jax
    from jax.sharding import PartitionSpec as P

    def _put_data(arr, spec):
        even = arr.shape[0] % env.data_dim == 0
        return jax.device_put(arr, env.nd(spec) if even else env.repl)

    fem = cfg.backprop_design is BackpropDesign.FEM_NODAL
    kw: dict = {}
    if bundle.v_c is not None:
        kw["v_c"] = jax.device_put(bundle.v_c, env.repl)
        kw["b_v"] = jax.device_put(bundle.b_v, env.repl)
        kw["v_nodes"] = (_put_data(bundle.v_nodes, P("data", None)) if fem
                         else jax.device_put(bundle.v_nodes, env.repl))
    kw["centroids"] = jax.device_put(bundle.centroids, env.repl)

    if fem:
        static = dict(bundle.static)
        for k, sp in (("N", P("data", None, None)), ("dN", P("data", None, None, None)),
                      ("JxW", P("data", None)), ("hex_nodes", P("data", None)),
                      ("k_hex_diag", P("data", None)), ("poro_hex", P("data"))):
            static[k] = _put_data(static[k], sp)
        kw["static"] = static
        kw["k_hex_diag"] = static["k_hex_diag"]
        kw["node_xyz"] = _put_data(bundle.node_xyz, P("data", None))
        kw["node_z"] = _put_data(bundle.node_z, P("data"))
        kw["vlump"] = _put_data(bundle.vlump, P("data"))
        kw["vft3"] = _put_data(bundle.vft3, P("data"))
        kw["poro_nodes"] = _put_data(bundle.poro_nodes, P("data"))

    return _dc_replace(bundle, **kw)
