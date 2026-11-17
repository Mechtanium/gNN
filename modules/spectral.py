r"""
Static hex-FEM operators and the cached Laplace eigenbasis (spectral encoding).

Disk cache (see :mod:`spectral_cache`): the N_EIG-independent static FEM
operators (element matrices, sparse stiffness/storage-mass, null-mode count),
the column-sliceable eigenbasis, and the addressability-floor search's
candidate pool and resolved floor, each keyed by what produced it. The
generalized eigenproblem with homogeneous Neumann boundary conditions is

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
- :math:`\Pi_z`: the averaging projector over each vertical column of nodes — the nodes joined along one grid pillar, or under ``spec.column_grouping = "coordinate"`` the nodes sharing an areal position (:func:`column_ids`) — so :math:`(I - \Pi_z)v_m` is the part of the mode that varies with depth along a column.
- :math:`\bar v_m`: the nodal mean of the mode,

the retained set is the lowest-:math:`\lambda` band amended so that at least
``n_eig_z`` modes with :math:`\upsilon_m > 1/2` are present (the lowest such
candidates replace the highest-:math:`\lambda` areal picks). Under
``spec.retention = "deflation"`` the vertical block is instead the lowest
solutions of the column-deflated eigenproblem (:func:`_vertical_family_by_deflation`),
retained unconditionally ahead of the lowest-:math:`\lambda` areal band. The
retained columns are ordered **vertical block first** (ascending :math:`\lambda`
within each block) so every prefix consumer of the basis — in particular the
spectral log-permeability parameterization — sees the vertical family.

Every solver path returns :math:`M_\phi`-orthonormal modes, :math:`V^\top M_\phi V = I`
(:func:`modules.utils.spectral_basis_jax.m_normalize`), so each retained mode enters the Galerkin
residual with the same weight whichever solver found it.

Under ``depth_encoding = "z"`` the bundle also carries the standardized depth channel
(:func:`depth_features`), which the encoder appends to the retained modes and the
addressability gate tests with them.

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
                     case_label, case_paths, depth_channel, needs_eigenbasis, pillar_columns)
from .casedata import CaseData
from .meshenv import MeshEnv

# Retention tuning now lives on ``cfg.spec`` (:class:`~modules.config.SpectralSettings`);
# these mirror its defaults and serve as fallbacks for the standalone helpers below.
# Width adopted for n_eig=0 when the addressability gate is switched off entirely
# (spec.addressability_min = 0), so there is no floor to resolve against.
_FALLBACK_N_EIG = 48
_DENSE_EIG_MAX_N = SpectralSettings.dense_eig_max_n
_BAND_CAP_MULT = SpectralSettings.band_cap_mult
_UPSILON_VERTICAL = SpectralSettings.upsilon_vertical
_COLUMN_TOL_FT = SpectralSettings.column_tol_ft
_DEFLATION_FALLBACK = (
    "deflation retention did not deliver the vertical family (the column-deflated solve "
    "declined its dense block, failed, or returned fewer modes above "
    "spec.upsilon_vertical than reserved); falling back to the band scan. Its vertical "
    "guarantee is best-effort within spec.band_cap_mult * n_eig candidates.")


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

    A :class:`~modules.config.ConfigError` subclass so the notebook and ``sweep``
    error paths catch it exactly like any other unrunnable composition — the run
    never starts and no ``runs.csv`` row is produced, because this is a
    precondition on the encoding rather than an outcome of training.
    """


# Feature columns a k-d tree indexes when it searches for close pairs; blocks this narrow are
# searched for nearest neighbours directly.
_SEP_INDEX_DIMS = 16
# Candidate pairs whose full distance is computed per chunk.
_SEP_PAIR_CHUNK = 1 << 16


def unresolved_cells(block, tau: float) -> onp.ndarray:
    r"""
    The rows of a feature block that have another row closer than :math:`\tau`.

    A cell :math:`c` is *resolved* when its nearest neighbour in feature space lies at
    least :math:`\tau` away,

    .. math::

        \delta_c \;=\; \min_{c' \neq c} \bigl\lVert \tilde{\mathbf v}_c - \tilde{\mathbf v}_{c'} \bigr\rVert_2 \;\ge\; \tau .

    The search is exact without comparing all :math:`n^2/2` pairs. The squared distance is
    a sum over feature columns, so the distance over any subset :math:`\mathcal J` of the
    columns is a lower bound on the full one, and every pair closer than :math:`\tau` is
    closer than :math:`\tau` over :math:`\mathcal J` as well. A k-d tree over at most
    ``_SEP_INDEX_DIMS`` columns spread across the block therefore returns a superset of the
    close pairs, and the full distance decides each candidate. A block no wider than that
    is searched for nearest neighbours directly.

    where:
    - :math:`\tilde{\mathbf v}_c`: the row of cell :math:`c` (its standardized centroid features, the depth channel included when the encoding carries one).
    - :math:`\tau`: the resolution, see :func:`encoder_adequacy`.

    Returns a boolean mask over the rows, computed in float64 on the host.
    """
    from scipy.spatial import cKDTree

    b = onp.asarray(block, onp.float64)
    n, d = b.shape
    out = onp.zeros(n, bool)
    if n < 2:
        return out
    if d == 0:
        return ~out                                     # no feature separates anything
    if d <= _SEP_INDEX_DIMS:
        dist, _ = cKDTree(b).query(b, k=2)
        return dist[:, 1] < tau
    cols = onp.unique(onp.round(onp.linspace(0, d - 1, _SEP_INDEX_DIMS)).astype(onp.int64))
    pairs = cKDTree(b[:, cols]).query_pairs(tau, output_type="ndarray")
    for s0 in range(0, len(pairs), _SEP_PAIR_CHUNK):
        pr = pairs[s0:s0 + _SEP_PAIR_CHUNK]
        diff = b[pr[:, 0]] - b[pr[:, 1]]
        out[pr[onp.einsum("ij,ij->i", diff, diff) < tau * tau].ravel()] = True
    return out


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

    Injectivity is measured as a separation: a cell is resolved when no other cell lies
    within the resolution :math:`\tau` of it in feature space (:func:`unresolved_cells`),

    .. math::

        \delta_c \;=\; \min_{c' \neq c} \bigl\lVert \tilde{\mathbf v}_c - \tilde{\mathbf v}_{c'} \bigr\rVert_2 \;\ge\; \tau,
        \qquad
        \tau \;=\; \varepsilon\,\overline{\sigma},
        \qquad
        \overline{\sigma} \;=\; \frac{1}{n_\lambda}\sum_m \operatorname{sd}_c\bigl(\tilde v_{cm}\bigr),

    and degeneracy on the centred amplitude
    :math:`a_c = \lVert \tilde{\mathbf v}_c - \overline{\tilde{\mathbf v}} \rVert
    \,/\, \max_{c'} \lVert \tilde{\mathbf v}_{c'} - \overline{\tilde{\mathbf v}} \rVert`.

    The separation is a continuous function of the features, so a basis that changes by
    rounding (a re-solve, another device) changes the verdict only for a cell whose
    nearest neighbour sits within that rounding of :math:`\tau`.

    where:
    - :math:`\varepsilon` (``tol``): the resolution as a fraction of the mean per-mode feature spread; it sets the injectivity bar directly, so it also sets the retention floor that :func:`~modules.spectral.provision` may have to widen :math:`n_\lambda` to reach.
    - :math:`\overline{\sigma}`: the mean per-mode standard deviation over cells, making the test scale-free.
    - :math:`\delta_c`: the distance from cell :math:`c` to its nearest neighbour in feature space.
    - :math:`a_c`: the relative feature amplitude of cell :math:`c`; cells with :math:`a_c < 10^{-2}` are counted *blind* — the encoding cannot distinguish them from the field mean.
    - ``cell_comp`` / ``kept``: optional component labels and a keep mask; when given, the statistics are restricted to cells of kept components, so pruned (negligible pore volume) components are not held against the basis.

    Returns ``{distinct, n_cells, addressable_frac, blind_frac, per_component}``, where
    ``distinct`` counts the resolved cells and ``addressable_frac`` is their fraction.
    """
    v = onp.asarray(v_c, onp.float64)
    sel = slice(None)
    if cell_comp is not None and kept is not None:
        mask = onp.asarray(kept)[onp.asarray(cell_comp)]
        sel = onp.flatnonzero(mask)

    def _stats(block):
        if block.shape[0] == 0:
            return {"distinct": 0, "n_cells": 0, "addressable_frac": 1.0, "blind_frac": 0.0}
        scale = float(block.std(axis=0).mean()) if block.shape[1] else 0.0
        resolved = int((~unresolved_cells(block, max(scale * tol, 1e-300))).sum())
        amp = onp.linalg.norm(block - block.mean(axis=0), axis=1)
        amp = amp / max(float(amp.max()), 1e-30)
        return {"distinct": resolved, "n_cells": int(block.shape[0]),
                "addressable_frac": resolved / block.shape[0],
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
    # depth channel (depth_encoding "z"; None otherwise), see depth_features
    z_nodes: Any = None         # (n_nodes, 1) standardized depth
    z_c: Any = None             # (n_cells, 1) at the centroids
    z_grad: Any = None          # (n_cells, 3, 1) physical gradient


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


def deflation_family_size(n_levels: int) -> int:
    r"""
    The size of the deflated vertical family: every vertical overtone a node column supports,

    .. math::

        n_{\mathrm f} \;=\; n_{\mathrm{lev}} - 1,

    whatever the basis width, the reservation and the search ceiling ``spec.n_eig_cap``.
    The reservation :math:`n_z^{\mathrm{res}}` (:func:`resolve_n_eig_z`) decides how many of
    these modes a basis keeps unconditionally; the family is solved whole because the band
    beside it is made :math:`M_\phi`-orthogonal to all of it, its duplicates left out
    (:func:`_lead_with_family`). A family cut to the reservation would leave the remaining
    overtones in the band, where they take the slots of areal modes, and would make the
    basis of a given width depend on how it was reached: the floor search and an explicit
    width would build different pools, and so would two searches under different ceilings.

    where:
    - :math:`n_{\mathrm{lev}}`: the number of node levels per vertical column (:func:`column_ids`); a column of :math:`n_{\mathrm{lev}}` nodes supports exactly :math:`n_{\mathrm{lev}} - 1` linearly independent vertical overtones.
    - :math:`n_{\mathrm f}`: the family size, the ``n_forced`` leading columns of the deflation pool.
    """
    return max(int(n_levels) - 1, 0)


def node_column_ids(verts, column_tol_ft: float = _COLUMN_TOL_FT) -> tuple[onp.ndarray, int]:
    """Vertical-column id per node (nodes sharing an areal position) and the level count."""
    xy = onp.round(onp.asarray(verts, onp.float64)[:, :2] / column_tol_ft).astype(onp.int64)
    _, col = onp.unique(xy, axis=0, return_inverse=True)
    col = onp.asarray(col).ravel()
    n_levels = int(onp.bincount(col).max()) if col.size else 0
    return col.astype(onp.int64), n_levels


def pillar_column_ids(hex_nodes, n_nodes: int) -> tuple[onp.ndarray, int]:
    r"""
    Vertical-column id per node along the grid pillars, and the level count.

    In the canonical VTK corner order of the reference cube the corners :math:`a` and
    :math:`a + 4` (:math:`a = 0, \dots, 3`) differ only in the third reference coordinate,
    the layer (:math:`K`) direction of the grid, so every hexahedron contributes four edges
    that run down a pillar. A column is a connected component of the graph of those edges,

    .. math::

        \mathcal E_{\mathrm{pil}} \;=\; \bigl\{\, \{h_{e,a},\ h_{e,a+4}\} \;:\; e = 1, \dots, n_{\mathrm{hex}},\ a = 0, \dots, 3 \,\bigr\},

    so the nodes of one column are those reached from each other by walking along pillars,
    however the pillars tilt. Across a fault the two sides keep separate nodes and form
    separate columns unless they share one; a node that only a pinched cell touches still
    lies on its pillar.

    where:
    - :math:`h_{e,a}`: the global id of corner :math:`a` of hexahedron :math:`e` (``hex_nodes``, canonical VTK order as returned by :func:`modules.utils.hex_fem_assembly_jax.canonical_hexes_from_corner_cells`).
    - :math:`\mathcal E_{\mathrm{pil}}`: the pillar edges; the level count is the size of the largest column.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    h = onp.asarray(hex_nodes, onp.int64)
    rows = h[:, :4].ravel()
    cols = h[:, 4:].ravel()
    g = sp.csr_matrix((onp.ones(rows.size), (rows, cols)), shape=(int(n_nodes), int(n_nodes)))
    _, col = connected_components(g, directed=False)
    col = onp.asarray(col, onp.int64)
    n_levels = int(onp.bincount(col).max()) if col.size else 0
    return col, n_levels


def column_ids(case: CaseData, cfg: RunConfig) -> tuple[onp.ndarray, int]:
    """The vertical node columns of this case under ``cfg.spec.column_grouping``: the grid
    pillars (:func:`pillar_column_ids`) by default, shared areal coordinates
    (:func:`node_column_ids`) under ``"coordinate"``."""
    if not pillar_columns(cfg):
        return node_column_ids(case.verts, cfg.spec.column_tol_ft)
    from modules.utils.hex_fem_assembly_jax import canonical_hexes_from_corner_cells

    hexes = canonical_hexes_from_corner_cells(
        case.verts, case.hexes, onp.asarray(case.art.reservoir_mesh.corner_cells))
    return pillar_column_ids(hexes, int(case.n_nodes))


def depth_features(static: dict, node_z):
    r"""
    The standardized depth channel at the nodes and its affine per-cell map.

    .. math::

        \zeta(\mathbf x) \;=\; \frac{z - \bar z}{s_z},
        \qquad
        \zeta_c \;=\; \tfrac18 \sum_{a} \zeta_{h_{e,a}},
        \qquad
        \nabla \zeta \;=\; \frac{1}{s_z}\,\hat{\mathbf z}

    where:
    - :math:`z`: the nodal vertical coordinate of the mesh (true vertical depth in a corner-point deck); :math:`\bar z, s_z`: its mean and standard deviation over the nodes, the same standardization the eigenmodes receive, so the channel enters the addressability test on the scale of one mode.
    - :math:`\zeta_c`: the value at the cell centroid; :math:`\nabla\zeta`: its physical gradient along the vertical unit vector :math:`\hat{\mathbf z}`, evaluated through the same element map as the eigenfeatures (:func:`modules.utils.hex_fem_assembly_jax.centroid_features_and_gradients`). A trilinear element reproduces a linear field exactly, so the affine map :math:`\zeta_c + \nabla\zeta \cdot (\mathbf x - \mathbf x_c)` is exact inside every valid cell; zeroed degenerate cells get no gradient, like the eigenfeatures.

    Returns ``(zeta_nodes (n, 1), zeta_c (n_cells, 1), grad (n_cells, 3, 1))`` in float32,
    computed on the host in float64, so every call returns the same arrays.
    """
    import jax.numpy as jnp

    z = onp.asarray(node_z, onp.float64)
    zeta = (z - z.mean()) / max(float(z.std()), 1e-30)
    corner = zeta[onp.asarray(static["hex_nodes"])]                       # (n_hex, 8)
    zeta_c = corner.mean(axis=1)
    grad = onp.einsum("eai,ea->ei", onp.asarray(static["dN_centroid"], onp.float64), corner)
    return (jnp.asarray(zeta[:, None], jnp.float32), jnp.asarray(zeta_c[:, None], jnp.float32),
            jnp.asarray(grad[:, :, None], jnp.float32))


def standardized_modes(v_raw, static):
    r"""
    The network features of a block of raw modes: each column standardized over the nodes,
    then evaluated with its physical gradient at every cell centroid,

    .. math::

        \tilde v_{im} \;=\; \frac{v_{im} - \bar v_m}{\sigma_m},
        \qquad
        \tilde{\mathbf v}_c \;=\; \tfrac18 \sum_a \tilde{\mathbf v}_{h_{e,a}},
        \qquad
        B_v \;=\; \sum_a \nabla N_a(\mathbf x_c)\, \tilde{\mathbf v}_{h_{e,a}}^{\top}

    where:
    - :math:`v_{im}`: mode :math:`m` at node :math:`i`; :math:`\bar v_m, \sigma_m`: its nodal mean and standard deviation, taken in float64. The modes are :math:`M_\phi`-orthonormal, so their entries are of order :math:`(\sum_i M_{\phi,ii})^{-1/2}` and a fixed additive guard would shrink the features of a large reservoir; the guard applies only to a constant column, which no oscillatory mode is.
    - :math:`\tilde{\mathbf v}_c, B_v`: the centroid features and their gradients (:func:`modules.utils.hex_fem_assembly_jax.centroid_features_and_gradients`); :math:`h_{e,a}`: the node of corner :math:`a` of cell :math:`e`, :math:`N_a` its trilinear shape function and :math:`\mathbf x_c` the cell centroid.

    The addressability floor is searched on these features of the whole candidate pool and
    the retained basis is cut from the very same arrays (:meth:`_FloorContext.ensure_basis`),
    so the basis that is cached is exactly the one whose addressability was verified.

    Returns ``(v_nodes, v_c, b_v, v_mean, v_std)``, the first three float32 device arrays.
    """
    import jax.numpy as jnp

    import modules.utils.hex_fem_assembly_jax as hf

    v = onp.asarray(v_raw, onp.float64)
    v_mean = v.mean(axis=0)
    v_std = v.std(axis=0)
    v_std = onp.where(v_std > 0, v_std, 1.0)
    v_nodes = jnp.asarray(((v - v_mean) / v_std).astype(onp.float32))
    v_c, b_v = hf.centroid_features_and_gradients(v_nodes, static)
    return (v_nodes, v_c, b_v, jnp.asarray(v_mean, jnp.float32),
            jnp.asarray(v_std, jnp.float32))


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
    """(lam_c, V_c, ups_c, n_forced) over a candidate band guaranteed (best-effort) to
    reach the vertical family: dense full spectrum on small meshes; on large ones the
    deflated vertical family ahead of a lowest-lambda band (``n_forced`` columns,
    retained unconditionally), or an expanding shift-invert Lanczos band."""
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
        band = _deflation_candidates(a_lap, m_lap, cfg, n_avail, col_id, n_zres, n_eig, n_null)
        if band is not None:
            return band
        warnings.warn(_DEFLATION_FALLBACK, RuntimeWarning, stacklevel=2)

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
                         n_forced: int = 0, start: int = 1,
                         extra: onp.ndarray | None = None) -> int | None:
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
    thousands of cells — with a weak dependence on cell count through the resolution.

    where:
    - ``lam_c``, ``ups_c``, ``v_c_pool``: the candidate eigenvalues, vertical variance fractions, and *standardized* centroid features of the whole candidate pool; column slicing is exact because standardization is per-column affine and centroid interpolation is linear with a partition of unity.
    - ``ceiling``: the largest width to try (``spec.n_eig_cap``, bounding the network input width).
    - ``cell_sel``: optional row selection restricting the test to kept components' cells.
    - ``start``: the first width to try; a search resumed on a wider pool skips the widths a narrower pool already decided.
    - ``extra``: centroid channels the network receives beside the retained modes at every width (the depth channel of :func:`depth_features`), appended to each trial block.

    Returns the floor, or ``None`` when no width in ``[start, ceiling]`` suffices.
    """
    spec = cfg.spec
    pool = v_c_pool if cell_sel is None else v_c_pool[cell_sel]
    fixed = None
    if extra is not None:
        fixed = onp.asarray(extra, onp.float64)
        fixed = fixed if cell_sel is None else fixed[cell_sel]
    for n in range(max(int(start), 1), int(ceiling) + 1):
        n_z = resolve_n_eig_z(_dc_replace(cfg, n_eig=n, n_eig_z=cfg.n_eig_z), n_levels)
        sel = _select_retained(lam_c, ups_c, n, n_z, spec.upsilon_vertical, n_forced)
        if sel.size < n:
            continue                      # candidate pool too narrow to realize this width
        block = pool[:, sel] if fixed is None else onp.concatenate([pool[:, sel], fixed], axis=1)
        stats = encoder_adequacy(block, tol=spec.addressability_tol)
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
    exactly the vertical-variation subspace of :math:`\upsilon_m = 1`. The deflation
    (:func:`_vertical_family_by_deflation`) constrains against it in the
    :math:`M_\phi` metric, :math:`Y^\top M_\phi \mathbf v = \mathbf 0`.
    """
    import scipy.sparse as sp

    n = col_id.size
    n_col = int(col_id.max()) + 1 if n else 0
    return sp.csr_matrix((onp.ones(n), (onp.arange(n), col_id)), shape=(n, n_col))


def _column_contrasts(col_id: onp.ndarray):
    r"""
    Sparse orthonormal basis :math:`\Gamma` of :math:`\ker Y^\top`: the Helmert contrasts
    of every vertical column, i.e. the fields whose column sums vanish,

    .. math::

        \Gamma_{:,(c,j)} \;=\; \frac{1}{\sqrt{j\,(j+1)}}
        \Bigl(\sum_{i<j} \mathbf e_{c_i} \;-\; j\,\mathbf e_{c_j}\Bigr),
        \qquad j = 1, \dots, |c| - 1

    where:
    - :math:`c_0, \dots, c_{|c|-1}`: the nodes of column :math:`c`; :math:`\mathbf e_i`: the unit vector of node :math:`i`. A single-node column contributes no contrast.
    - :math:`\Gamma^\top \Gamma = I` and :math:`Y^\top \Gamma = 0`, so :math:`\Gamma` has :math:`n_{\mathcal V} = n - n_{\mathrm{col}}` columns.
    """
    import scipy.sparse as sp

    n = col_id.size
    order = onp.argsort(col_id, kind="stable")
    counts = onp.bincount(col_id) if n else onp.zeros(0, onp.int64)
    starts = onp.concatenate([[0], onp.cumsum(counts)[:-1]]) if n else counts
    rows, cols, vals = [], [], []
    j_out = 0
    for c in onp.flatnonzero(counts > 1):
        idx = order[starts[c]:starts[c] + counts[c]]
        for j in range(1, idx.size):
            w = 1.0 / onp.sqrt(j * (j + 1))
            rows.append(idx[:j + 1])
            cols.append(onp.full(j + 1, j_out))
            vals.append(onp.concatenate([onp.full(j, w), [-j * w]]))
            j_out += 1
    if j_out == 0:
        return sp.csc_matrix((n, 0))
    return sp.csc_matrix((onp.concatenate(vals), (onp.concatenate(rows), onp.concatenate(cols))),
                         shape=(n, j_out))


def _mass_solve(m_s, rhs, rtol: float = 1e-13, maxiter: int = 2000):
    r""":math:`M_\phi^{-1}` applied to a block of right-hand sides: Jacobi-preconditioned
    block CG (a consistent mass matrix is spectrally equivalent to its diagonal, so this
    converges in tens of sweeps without fill-in), SuperLU if it ever stalls."""
    import scipy.sparse.linalg as sla

    diag = onp.asarray(m_s.diagonal(), onp.float64)[:, None]
    x = onp.zeros_like(rhs)
    r = rhs.copy()
    z = r / diag
    p = z.copy()
    rz = (r * z).sum(axis=0)
    norm_b = onp.maximum(onp.linalg.norm(rhs, axis=0), 1e-300)
    for _ in range(maxiter):
        ap = m_s @ p
        pap = (p * ap).sum(axis=0)
        alpha = onp.divide(rz, pap, out=onp.zeros_like(rz), where=pap > 0)
        x += p * alpha
        r -= ap * alpha
        if (onp.linalg.norm(r, axis=0) / norm_b).max() < rtol:
            return x
        z = r / diag
        rz_new = (r * z).sum(axis=0)
        p = z + p * onp.divide(rz_new, rz, out=onp.zeros_like(rz), where=rz > 0)
        rz = rz_new
    return sla.splu(m_s.tocsc(), permc_spec="MMD_AT_PLUS_A").solve(rhs)


# The exact deflation holds one dense block: nodes x within-column contrasts on the
# null-space route, the columns x columns Schur complement (plus one nodes x
# _DEFLATION_BLOCK slab while assembling it) on the Schur route. Past this many entries
# (2 GiB in float64) it declines and the caller falls back.
_DEFLATION_DENSE_ENTRIES = 1 << 28
# Constrained dimensions up to this size always take the dense null-space route.
_DEFLATION_DENSE_DIM = 1024
# Right-hand sides per sparse solve while the Schur complement is assembled.
_DEFLATION_BLOCK = 256
# Shift (lambda_sh) and mass regularization of the scaled pencil, as in the band solve.
_DEFLATION_SIGMA = -1e-5
_DEFLATION_JITTER = 1e-8


def _vertical_family_by_deflation(a_lap, m_lap, col_id: onp.ndarray, n_want: int,
                                  tol: float = 0.0, maxiter: int | None = None):
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
    domain however large it is made. Solving the eigenproblem restricted to the fields
    with zero :math:`M_\phi`-weighted column means removes the mismatch,

    .. math::

        A\,\mathbf v = \lambda\, M_\phi\, \mathbf v,
        \qquad
        Y^\top M_\phi\, \mathbf v = \mathbf 0,

    because the lowest modes of that constrained problem are the vertical family, found
    without scanning the areal spectrum at all. The problem is solved *exactly*, by
    Rayleigh–Ritz on the whole constrained space :math:`\mathcal V`, along whichever of
    two sparse routes keeps the dense block small. When the constrained dimension
    :math:`n_{\mathcal V} = n - n_{\mathrm{col}}` is small (columns that rarely stack, as on
    a corner-point grid with tilted pillars) :math:`\mathcal V` has the explicit basis

    .. math::

        \mathcal V \;=\; M_\phi^{-1} \ker Y^\top \;=\; \operatorname{range}(\Xi),
        \qquad
        \Xi = M_\phi^{-1} \Gamma,
        \qquad
        \bigl(\Xi^\top A\, \Xi\bigr)\,\mathbf s \;=\; \lambda\,\bigl(\Gamma^\top \Xi\bigr)\,\mathbf s,
        \qquad
        \mathbf v = \Xi\, \mathbf s,

    and a dense symmetric-definite solve of size :math:`n_{\mathcal V}` returns every mode.
    When the columns stack (:math:`n_{\mathrm{col}} < n_{\mathcal V}`), shift-invert Lanczos
    runs on the constrained inverse :math:`\mathbf x \mapsto \mathbf y`,

    .. math::

        \begin{bmatrix} A - \lambda_{\mathrm{sh}} M_\phi & M_\phi Y \\ Y^\top M_\phi & 0 \end{bmatrix}
        \begin{bmatrix} \mathbf y \\ \mathbf l \end{bmatrix}
        \;=\;
        \begin{bmatrix} M_\phi\, \mathbf x \\ \mathbf 0 \end{bmatrix},
        \qquad
        \mathcal C \;=\; Y^\top M_\phi\,(A - \lambda_{\mathrm{sh}} M_\phi)^{-1} M_\phi\, Y,

    whose eigenvalues are :math:`(\lambda - \lambda_{\mathrm{sh}})^{-1}` on
    :math:`\mathcal V` and zero on :math:`\operatorname{range}(Y)`, with the saddle point
    eliminated through the dense :math:`n_{\mathrm{col}} \times n_{\mathrm{col}}` Schur
    complement :math:`\mathcal C` and one step of iterative refinement (the shift sits
    beside the per-component constants of :math:`\ker A`, whose conditioning
    :math:`\mathcal C` inherits). :math:`\mathcal C` is assembled a slab of columns at a
    time and each elimination re-solves with the sparse factor of
    :math:`A - \lambda_{\mathrm{sh}} M_\phi` instead of storing the image
    :math:`(A - \lambda_{\mathrm{sh}} M_\phi)^{-1} M_\phi Y`, so the route holds
    :math:`n_{\mathrm{col}}^2` dense entries rather than :math:`n\, n_{\mathrm{col}}` — the
    difference that admits the six thousand pillar columns of a corner-point grid.
    Zero-energy solutions, :math:`\ker A \cap \mathcal V`, are
    combinations of per-component constants that happen to satisfy the constraint (a
    column shared by two components); like the null modes of the band solve they are
    counted exactly and dropped,

    .. math::

        n_{\ker} \;=\; n_{\mathrm{comp}} - \operatorname{rank}\bigl(Y^\top M_\phi\, \mathcal Z\bigr),

    on the unregularized storage matrix, which keeps the column masses of two components
    exactly proportional when their rock is. Nodes that no valid element touches (a zeroed
    degenerate cell leaves them behind) are excluded and returned as zeros: carrying
    neither stiffness nor storage, in a shared column they would meet the constraint as a
    free slack and pose as a zero-energy vertical mode.

    where:
    - :math:`Y` (:func:`column_indicator_basis`): the column-indicator matrix, :math:`n_{\mathrm{col}}` columns; :math:`\Gamma` (:func:`_column_contrasts`): the orthonormal within-column contrasts spanning :math:`\ker Y^\top`; :math:`\Xi`: the resulting basis of :math:`\mathcal V`, and :math:`\mathbf s` the reduced coordinates of a mode in it.
    - :math:`k_z/k_x`: the vertical-to-areal permeability contrast; :math:`L_x, H_z`: areal extent and thickness.
    - :math:`\lambda_{\mathrm{sh}}`: a shift just below the spectrum, as in the band solve; :math:`\mathbf l`: the constraint multipliers.
    - :math:`\mathcal Z`: the indicator matrix of the :math:`n_{\mathrm{comp}}` connected components of the stiffness graph, spanning :math:`\ker A`; :math:`n_{\ker}`: the number of zero-energy constrained modes.
    - ``n_want``: how many vertical modes to return; ``tol``/``maxiter``: the Lanczos controls of the shift-invert route (``0``/``None``: machine precision, ARPACK's default).

    The routes form one dense block of :math:`n \times n_{\mathcal V}` (null-space) or
    :math:`n_{\mathrm{col}} \times n_{\mathrm{col}}` (Schur) entries. Returns ``(lam, V)`` ascending, with
    :math:`V` :math:`M_\phi`-orthonormal and each column's largest entry positive, or
    ``None`` when the constrained space is empty, that block would exceed
    ``_DEFLATION_DENSE_ENTRIES``, or the solve fails — the caller then falls back to the
    band scan rather than proceeding without a vertical family.
    """
    import scipy.linalg as dla
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    from scipy.sparse.csgraph import connected_components

    from modules.utils.spectral_basis_jax import m_normalize

    a = sp.csr_matrix(a_lap).astype(onp.float64)
    m = sp.csr_matrix(m_lap).astype(onp.float64)
    n_full = a.shape[0]
    # Nodes that no valid element touches (zeroing a degenerate cell leaves them behind)
    # carry neither stiffness nor storage. In a shared column such a node is a free slack
    # that meets the constraint at zero energy, so it is left out and returned as zero.
    live = onp.flatnonzero((abs(a).sum(axis=1).A1 > 0) | (m.diagonal() > 0))
    a, m = a[live][:, live], m[live][:, live]
    col_id = onp.unique(onp.asarray(col_id)[live], return_inverse=True)[1].ravel()
    n = live.size
    y = column_indicator_basis(col_id)
    n_col = y.shape[1]
    dim_v = n - n_col
    nullspace_route = dim_v <= max(n_col, _DEFLATION_DENSE_DIM)
    dense = n * dim_v if nullspace_route else n_col * n_col + n * min(n_col, _DEFLATION_BLOCK)
    if int(n_want) < 1 or dim_v < 1 or dense > _DEFLATION_DENSE_ENTRIES:
        return None

    s_a = max(float(a.diagonal().mean()), 1e-30)
    s_m = max(float(m.diagonal().mean()), 1e-30)
    a_s = (a / s_a).tocsr()
    m_s = (m / s_m + _DEFLATION_JITTER * sp.identity(n, format="csr")).tocsr()
    my = (m_s @ y).tocsc()                                     # M_phi Y

    # n_ker = n_comp - rank(Y' M Z): per-component constants that satisfy the constraint
    # (zero energy, dropped below), counted on the unregularized storage, which keeps the
    # column masses of two components exactly proportional when their rock is
    a_graph = a.copy()
    a_graph.eliminate_zeros()
    n_comp, comp = connected_components(a_graph, directed=False)
    e = sp.csr_matrix((onp.ones(n), (onp.arange(n), comp)), shape=(n, n_comp))
    n_null = n_comp - int(onp.linalg.matrix_rank((y.T @ (m @ e)).toarray()))
    k = min(int(n_want) + n_null, dim_v)
    if k <= n_null:
        return None

    try:
        if nullspace_route:
            z = _column_contrasts(col_id)
            w = _mass_solve(m_s, z.toarray())
            a_h = w.T @ (a_s @ w)
            m_h = z.T @ w
            lam, c = dla.eigh(0.5 * (a_h + a_h.T), 0.5 * (m_h + m_h.T),
                              subset_by_index=[0, k - 1])
            v = w @ c
        else:
            h = (a_s - _DEFLATION_SIGMA * m_s).tocsc()
            lu = sla.splu(h, permc_spec="MMD_AT_PLUS_A")
            # the Schur complement C = Y' M H^-1 M Y, a slab of columns at a time: the
            # n x n_col image H^-1 M Y is never held whole
            c_mat = onp.empty((n_col, n_col))
            for j0 in range(0, n_col, _DEFLATION_BLOCK):
                j1 = min(n_col, j0 + _DEFLATION_BLOCK)
                c_mat[:, j0:j1] = my.T @ lu.solve(my[:, j0:j1].toarray())
            c_mat += c_mat.T
            c_mat *= 0.5
            schur = dla.cho_factor(c_mat, overwrite_a=True)
            gram = sla.splu((y.T @ my).tocsc())               # Y' M_phi Y

            def _saddle_solve(r1, r2):
                mu = dla.cho_solve(schur, my.T @ lu.solve(r1) - r2)
                return lu.solve(r1 - my @ mu), mu

            def _constrained_inverse(b):
                b = onp.asarray(b, onp.float64).ravel()
                yv, mu = _saddle_solve(b, onp.zeros(n_col))
                # The shift sits next to the per-component constants, so C inherits their
                # conditioning: one refinement step recovers the lost digits, and the
                # projection re-imposes Y' M v = 0 exactly so no rounding leaks out of V.
                dy, _ = _saddle_solve(b - h @ yv - my @ mu, -(my.T @ yv))
                yv = yv + dy
                return yv - y @ gram.solve(my.T @ yv)

            op = sla.LinearOperator((n, n), matvec=_constrained_inverse, dtype=onp.float64)
            lam, v = sla.eigsh(a_s, k=k, M=m_s, sigma=_DEFLATION_SIGMA, OPinv=op, which="LM",
                               v0=onp.random.default_rng(0).standard_normal(n),
                               tol=tol, maxiter=maxiter)
    except Exception:
        return None

    lam = onp.asarray(lam, onp.float64)
    order = onp.argsort(lam)
    lam, v = lam[order][n_null:], onp.asarray(v)[:, order][:, n_null:]
    if not (onp.all(onp.isfinite(lam)) and onp.all(onp.isfinite(v))):
        return None
    v = m_normalize(m, v)                              # M_phi-orthonormal in the unscaled metric
    v_full = onp.zeros((n_full, v.shape[1]))
    v_full[live] = v
    return lam * (s_a / s_m), v_full


def _deflated_family(a_lap, m_lap, cfg: RunConfig, col_id: onp.ndarray, n_zres: int):
    """``(lam, V, ups)`` of the deflated vertical family, or ``None`` when the solve declines
    or fewer than ``n_zres`` of its modes clear ``spec.upsilon_vertical``."""
    got = _vertical_family_by_deflation(a_lap, m_lap, col_id, n_zres)
    if got is None:
        return None
    lam_v, v_v = got
    ups_v = vertical_variance_fraction(v_v, col_id)
    if int((ups_v > cfg.spec.upsilon_vertical).sum()) < n_zres:
        return None                       # deflation did not actually deliver the family
    return onp.asarray(lam_v), onp.asarray(v_v), ups_v


# A band mode holding more than this share of its M_phi-energy inside the span of the deflated
# family duplicates a vertical mode the pool already leads with, and is left out.
_FAMILY_DUPLICATE = 0.5


def _lead_with_family(family, lam_a, v_a, a_lap, m_lap, col_id: onp.ndarray):
    r"""
    The candidate pool of the deflation strategy: the deflated vertical family first, then
    the lowest-:math:`\lambda` band made :math:`M_\phi`-orthogonal to it,

    .. math::

        \mathbf w_k \;=\; \frac{\bigl(I - V_{\mathrm f} V_{\mathrm f}^{\top} M_\phi\bigr)\,\mathbf v_k}
                               {\bigl\lVert \bigl(I - V_{\mathrm f} V_{\mathrm f}^{\top} M_\phi\bigr)\,\mathbf v_k \bigr\rVert_{M_\phi}},
        \qquad
        \lambda_k \;=\; \mathbf w_k^{\top} A\, \mathbf w_k,
        \qquad
        \text{kept when}\quad \bigl\lVert V_{\mathrm f}^{\top} M_\phi\, \mathbf v_k \bigr\rVert^2 \;\le\; \tfrac12

    where:
    - :math:`V_{\mathrm f}`: the :math:`M_\phi`-orthonormal deflated family (:func:`_vertical_family_by_deflation`); :math:`\mathbf v_k`: a band mode of unit :math:`M_\phi`-norm, and :math:`\lVert V_{\mathrm f}^{\top} M_\phi \mathbf v_k \rVert^2` the share of its :math:`M_\phi`-energy inside the family's span.
    - :math:`\mathbf w_k, \lambda_k`: the band mode with that share removed and renormalized, and its Rayleigh quotient.

    The two halves of the pool solve different eigenproblems — the vertical family the
    constrained one, the band the full one — so a vertical mode low enough to fall inside
    the band appears in both. Kept twice, its test direction would carry two Galerkin rows
    and the network two copies of one input. A band mode lying more than half inside the
    family's span is such a duplicate and is left out (``_FAMILY_DUPLICATE``); every other
    band mode loses only its small vertical part, which makes the family and the band
    :math:`M_\phi`-orthogonal. Each band mode is treated on its own, so a wider band extends
    the pool without changing the columns already in it, as the floor search requires
    (:class:`_FloorContext`).

    Returns ``(lam, V, ups, n_forced)``: ``V`` float32, ``n_forced`` the family size.
    """
    import scipy.sparse as sp

    from modules.utils.spectral_basis_jax import m_normalize

    lam_v, v_v, ups_v = family
    vf = onp.asarray(v_v, onp.float64)
    vb = onp.asarray(v_a, onp.float64)
    m = sp.csr_matrix(m_lap).astype(onp.float64)
    c = vf.T @ (m @ vb)                                 # family coordinates of each band mode
    keep = (c * c).sum(axis=0) <= _FAMILY_DUPLICATE
    w = m_normalize(m, vb[:, keep] - vf @ c[:, keep])
    lam_w = onp.einsum("ij,ij->j", w, sp.csr_matrix(a_lap).astype(onp.float64) @ w)
    lam = onp.concatenate([onp.asarray(lam_v, onp.float64), lam_w])
    v = onp.concatenate([vf, w], axis=1).astype(onp.float32)
    ups = onp.concatenate([onp.asarray(ups_v), vertical_variance_fraction(w, col_id)])
    return lam, v, ups, int(onp.asarray(lam_v).size)


def _deflation_candidates(a_lap, m_lap, cfg: RunConfig, n_avail: int,
                          col_id: onp.ndarray, n_zres: int, n_eig: int,
                          n_null: int | None = None):
    """Candidate pool assembled as (deflated vertical family) + (lowest-lambda areal band).

    The two halves answer different questions, so they are solved separately rather
    than hunted for in one ordering: the vertical block comes from the deflated
    operator, where depth-varying is guaranteed, and the areal remainder from the usual
    shift-invert band. The family is the whole one (:func:`deflation_family_size`), of
    which the selection keeps ``n_zres`` unconditionally, and the band is wide enough to
    keep ``n_eig`` columns beside it (fewer than twice the family size are ever left out
    as its duplicates, :meth:`_FloorContext.wider`), so this is the pool the floor search
    cuts a width-``n_eig`` basis from, and the two routes give one basis. Returns ``None``
    when the deflated solve declines (empty constrained space, oversized dense block),
    fails, or yields a family short of vertical modes (:func:`_deflated_family`), leaving
    the caller to fall back.
    """
    from modules.utils.spectral_basis_jax import generalized_eig_basis_sparse

    col = onp.asarray(col_id)
    n_levels = int(onp.bincount(col).max()) if col.size else 0
    n_f = max(deflation_family_size(n_levels), int(n_zres))
    family = _deflated_family(a_lap, m_lap, cfg, col_id, n_f)
    if family is None:
        return None

    k = min(n_avail, max(n_eig, 1) + 2 * n_f)
    lam_a, v_a = generalized_eig_basis_sparse(a_lap, m_lap, k, which="SM",
                                              drop_null=True, n_null=n_null)
    # the deflated family leads the pool and is retained unconditionally: it sits far
    # above the areal band in lambda, so an eigenvalue-ordered contest would drop it.
    return _lead_with_family(family, lam_a, v_a, a_lap, m_lap, col_id)


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


# Sentinel for "no floor resolved yet" (a resolved floor may itself be None: no width works).
_UNRESOLVED = object()
# In-process memos: resolved floors per (cache, operators, resolution key), so a notebook
# preview and the pipeline build that follows it resolve once between them; and the
# (cache, operators, selection, width) entries whose cached basis was already verified.
_FLOOR_MEMO: dict[tuple[str, str, str], int | None] = {}
_PINNED: set[tuple[str, str, str, int]] = set()
# First band width of the floor-search pool; the band doubles until it decides the floor.
_POOL_K0 = 32
# Floors above this width draw a recommendation of the depth_encoding option the config lacks.
_DEPTH_HINT_FLOOR = 64


@dataclass
class _Pool:
    """A floor-search candidate pool: an optional leading deflated family (retained
    unconditionally), then the lowest-lambda band of width ``k_band``."""

    lam: Any                # (n_cand,) candidate eigenvalues
    v: Any                  # (n_nodes, n_cand) raw candidate modes, float32
    ups: Any                # (n_cand,) vertical variance fractions
    n_forced: int           # leading deflated columns; 0 outside the deflation regime
    k_band: int             # width of the lowest-lambda band
    regime: str             # "dense" (full spectrum) | "deflation" | "band"


def _pool_key(cfg: RunConfig, n_z_top: int) -> dict:
    """The settings that fix a candidate pool's contents (its band width aside, which a
    search only ever grows)."""
    spec = cfg.spec
    return {"pool_retention": getattr(spec.retention, "value", spec.retention),
            "pool_dense_eig_max_n": int(spec.dense_eig_max_n),
            "pool_upsilon_vertical": float(spec.upsilon_vertical),
            "pool_column_grouping": str(spec.column_grouping),
            "pool_column_tol_ft": float(spec.column_tol_ft),
            "pool_n_z_top": int(n_z_top)}


def _floor_key(cfg: RunConfig, n_z_top: int, well_key: str | None = None) -> dict:
    """Everything the floor depends on besides the operators: the pool, the search
    ceiling, the vertical reservation rule, the pruned components and the adequacy gate,
    including the channels the gate sees beside the modes (the depth channel; the near-well
    channel, keyed by ``well_key``, a digest of its perforations and cutoff)."""
    spec = cfg.spec
    key = {**_pool_key(cfg, n_z_top),
           "floor_n_eig_cap": int(spec.n_eig_cap),
           "floor_n_eig_z": int(cfg.n_eig_z),
           "floor_band_cap_mult": int(spec.band_cap_mult),
           "floor_min_comp_vol_frac": float(spec.min_comp_vol_frac),
           "floor_addressability_min": float(spec.addressability_min),
           "floor_addressability_tol": float(spec.addressability_tol),
           "floor_blind_frac_max": float(spec.blind_frac_max)}
    if depth_channel(cfg):
        key["floor_depth_channel"] = True
    if well_key is not None:
        key["floor_well_channel"] = well_key
    return key


def _selection_key(cfg: RunConfig, n_levels: int | None = None) -> dict:
    """The eigenbasis-tier selection key (:func:`modules.utils.spectral_cache.eig_selection_key`);
    under deflation also the family size the band was made orthogonal to
    (:func:`deflation_family_size`), so a basis cut from another family is never served."""
    import modules.utils.spectral_cache as sc

    key = sc.eig_selection_key(cfg.spec)
    if n_levels is not None and cfg.spec.retention == "deflation":
        key["spec_deflation_family"] = deflation_family_size(n_levels)
    return key


def gate_channels(cfg: RunConfig, case: CaseData, z_c=None):
    """The centroid channels the encoder appends to the retained modes, ``(n_cells, k)``
    float64 or ``None``: the standardized depth (``z_c``, when the encoding carries it), then
    the near-well channel under ``well_encoding="logr"``
    (:func:`modules.encodings.well_channel_centroids`). The addressability gate tests the
    modes together with these, the input the network actually receives."""
    from .encodings import well_channel_centroids

    cols = [] if z_c is None else [onp.asarray(z_c, onp.float64).reshape(-1, 1)]
    well = well_channel_centroids(cfg, case)
    if well is not None:
        cols.append(well[:, None])
    return onp.concatenate(cols, axis=1) if cols else None


def _gate_block(v_c, extra):
    """The centroid features the addressability gate tests: the retained modes, then the
    channels of :func:`gate_channels`."""
    if extra is None:
        return v_c
    return onp.concatenate([onp.asarray(v_c, onp.float64), onp.asarray(extra, onp.float64)], axis=1)


def _well_key(cfg: RunConfig, case: CaseData) -> str | None:
    """A digest of the near-well channel's geometry (perforations and cutoff), so a floor
    found with the channel is never reused after the completions move; ``None`` without it."""
    if cfg.well_encoding != "logr":
        return None
    import modules.utils.spectral_cache as sc

    from .encodings import well_channel_geometry

    perf, r0, _ = well_channel_geometry(cfg, case)
    return sc.key_digest({"perf_xyz": onp.round(onp.asarray(perf, onp.float64), 3).tolist(),
                          "r0": round(float(r0), 6)})


def _best_effort(write, *args, **kwargs) -> None:
    """Run a cache write whose failure (read-only or full disk) must not stop the run."""
    try:
        write(*args, **kwargs)
    except OSError as e:
        warnings.warn(f"spectral cache entry not written ({e}); the result is used now "
                      "but will be recomputed next time", RuntimeWarning, stacklevel=3)


@dataclass
class _FloorContext:
    r"""
    One reservoir's addressability-floor search: where its caches live, its node
    columns, its search ceiling, and (loaded on first use) its FEM operators.

    The candidate pool is *grown* rather than solved at the ceiling: its lowest-:math:`\lambda`
    band starts at ``_POOL_K0`` modes and doubles until it decides the floor, so the
    eigensolve scales with the floor, not with ``spec.n_eig_cap``. A trial width is tested
    only once the band is wide enough that a wider one could not change its selection,

    .. math::

        \mathcal S_n\bigl(\mathcal P_k\bigr) \;=\; \mathcal S_n\bigl(\mathcal P_{k'}\bigr)
        \quad \forall\, k' \ge k
        \qquad\text{whenever}\qquad
        \begin{cases}
        k_{\mathrm{kept}} \ge n & \text{deflation} \\
        \#\{m : \upsilon_m > \upsilon_\star\} \ge n_z^{\mathrm{res}}(n) & \text{band scan}
        \end{cases}

    where:
    - :math:`\mathcal S_n(\mathcal P_k)`: the set :func:`_select_retained` keeps at trial width :math:`n` from the pool whose band holds the lowest :math:`k` modes. Under deflation the vertical family is fixed and the remaining slots are a :math:`\lambda`-prefix, so any band that keeps :math:`k_{\mathrm{kept}} \ge n` columns beside the family (:func:`_lead_with_family`) realizes it; under the band scan the swapped-in vertical modes are the lowest ones beyond :math:`n`, which a band already holding :math:`n_z^{\mathrm{res}}(n)` of them contains.
    - :math:`\upsilon_\star`: ``spec.upsilon_vertical``; :math:`n_z^{\mathrm{res}}(n)`: the vertical reservation at width :math:`n` (:func:`resolve_n_eig_z`).
    - the band also counts as deciding every width once it reaches the depth the full-ceiling search would stop at: :math:`n_{\mathrm{cap}}` kept modes under deflation, ``band_cap_mult`` :math:`\times\, n_{\mathrm{cap}}` under the band scan (or every available mode).
    """

    case: CaseData
    cfg: RunConfig
    cache_dir: Any
    op_key: dict
    col_id: Any
    n_levels: int
    n_z_top: int
    ceiling: int
    _bundle: Any = None
    _features: Any = None       # (pool, standardized_modes of its columns), the last pool
    _extra: Any = None          # gate_channels, computed once
    well_key: str | None = None

    @classmethod
    def of(cls, case: CaseData, cfg: RunConfig) -> "_FloorContext":
        cache_dir, op_key, _ = _cache_locator(case, cfg)
        col_id, n_levels = column_ids(case, cfg)
        ceiling = int(cfg.spec.n_eig_cap)
        # the pool's vertical family: under deflation the whole family, so the pool and the
        # floor found on it do not depend on the ceiling (deflation_family_size); the band
        # scan's pool holds no family and keeps its key
        n_z_top = (deflation_family_size(n_levels) if cfg.spec.retention == "deflation"
                   else resolve_n_eig_z(_dc_replace(cfg, n_eig=ceiling), n_levels))
        return cls(case, cfg, cache_dir, op_key, col_id, n_levels, n_z_top, ceiling,
                   well_key=_well_key(cfg, case))

    @property
    def extra(self):
        """The gate's channels beside the modes (:func:`gate_channels`)."""
        if self._extra is None:
            self._extra = (gate_channels(self.cfg, self.case, self.bundle.z_c),)
        return self._extra[0]

    @property
    def bundle(self) -> SpectralBundle:
        """The operator tier and admissibility report (no eigenbasis), loaded once."""
        if self._bundle is None:
            self._bundle = provision(self.case, self.cfg, _resolving=True)
        return self._bundle

    @property
    def memo_key(self) -> tuple[str, str, str]:
        import modules.utils.spectral_cache as sc

        return (str(self.cache_dir), sc.key_digest(self.op_key),
                sc.key_digest(_floor_key(self.cfg, self.n_z_top, self.well_key)))

    @property
    def n_avail(self) -> int:
        return int(self.case.n_nodes) - int(self.bundle.n_null) - 1

    # ---- the resolved floor: in-process memo, then the floor tier on disk -------------
    def cached_floor(self):
        import modules.utils.spectral_cache as sc

        if self.memo_key not in _FLOOR_MEMO:
            hit = sc.load_floor_cache(self.cache_dir, self.op_key,
                                      _floor_key(self.cfg, self.n_z_top, self.well_key))
            if hit is None:
                return _UNRESOLVED
            _FLOOR_MEMO[self.memo_key] = hit["floor"]
        return _FLOOR_MEMO[self.memo_key]

    def remember(self, floor: int | None) -> None:
        import modules.utils.spectral_cache as sc

        _FLOOR_MEMO[self.memo_key] = floor
        _best_effort(sc.save_floor_cache, self.cache_dir, self.op_key,
                     _floor_key(self.cfg, self.n_z_top, self.well_key), floor)

    # ---- the candidate pool: cached, and grown in place ---------------------------------
    def load_pool(self) -> _Pool | None:
        import modules.utils.spectral_cache as sc

        got = sc.load_pool_cache(self.cache_dir, self.op_key, _pool_key(self.cfg, self.n_z_top))
        if got is None:
            return None
        regime = ("dense" if int(self.case.n_nodes) <= self.cfg.spec.dense_eig_max_n
                  else "deflation" if got["n_forced"] else "band")
        return _Pool(got["lam"], got["v"], got["ups"], got["n_forced"], got["k_band"], regime)

    def solve_pool(self, k_band: int, prev: _Pool | None) -> _Pool:
        """The pool at band width ``k_band``, reusing ``prev``'s deflated family (or its
        verdict that deflation failed) instead of solving it again; written to the cache."""
        import modules.utils.spectral_cache as sc
        from modules.utils.spectral_basis_jax import (generalized_eig_basis_dense_full,
                                              generalized_eig_basis_sparse)

        b, spec = self.bundle, self.cfg.spec
        if int(self.case.n_nodes) <= spec.dense_eig_max_n:
            lam, v = generalized_eig_basis_dense_full(b.a_lap, b.m_lap, drop_null=True,
                                                      n_null=int(b.n_null))
            pool = _Pool(lam, v, vertical_variance_fraction(v, self.col_id), 0,
                         int(lam.size), "dense")
        else:
            family = None
            if prev is not None:
                if prev.n_forced:
                    f = prev.n_forced
                    family = (prev.lam[:f], prev.v[:, :f], prev.ups[:f])
            elif spec.retention == "deflation" and self.n_z_top > 0:
                family = _deflated_family(b.a_lap, b.m_lap, self.cfg, self.col_id,
                                          self.n_z_top)
                if family is None:
                    warnings.warn(_DEFLATION_FALLBACK, RuntimeWarning, stacklevel=3)
            k = min(int(k_band), self.n_avail)
            lam_a, v_a = generalized_eig_basis_sparse(b.a_lap, b.m_lap, k, which="SM",
                                                      drop_null=True, n_null=int(b.n_null))
            if family is None:
                pool = _Pool(lam_a, v_a, vertical_variance_fraction(v_a, self.col_id), 0, k,
                             "band")
            else:
                lam, v, ups, n_f = _lead_with_family(family, lam_a, v_a, b.a_lap, b.m_lap,
                                                     self.col_id)
                pool = _Pool(lam, v, ups, n_f, k, "deflation")
        _best_effort(sc.save_pool_cache, self.cache_dir, self.op_key,
                     _pool_key(self.cfg, self.n_z_top), lam=pool.lam, v=pool.v,
                     ups=pool.ups, n_forced=pool.n_forced, k_band=pool.k_band)
        return pool

    def decided_width(self, pool: _Pool) -> int:
        """The largest trial width whose selection from ``pool`` no wider band could change."""
        spec = self.cfg.spec
        if pool.regime == "dense" or pool.k_band >= self.n_avail:
            return self.ceiling
        if pool.regime == "deflation":
            # the band columns kept beside the family (duplicates of it are left out)
            return min(self.ceiling, int(onp.asarray(pool.lam).size) - pool.n_forced)
        if pool.k_band >= min(self.n_avail, spec.band_cap_mult * self.ceiling):
            return self.ceiling
        n_vert = int((onp.asarray(pool.ups) > spec.upsilon_vertical).sum())
        width = 0
        for n in range(1, min(self.ceiling, pool.k_band) + 1):
            if resolve_n_eig_z(_dc_replace(self.cfg, n_eig=n), self.n_levels) > n_vert:
                break
            width = n
        return width

    def wider(self, pool: _Pool) -> _Pool:
        # Under deflation the band must keep `ceiling` columns beside the family. A dropped
        # band mode holds over half its M-energy in the family's span, and those shares sum
        # to at most the family size over an orthonormal band (Bessel), so fewer than
        # 2 n_forced modes are ever dropped.
        cap = (self.ceiling + 2 * pool.n_forced if pool.regime == "deflation"
               else self.cfg.spec.band_cap_mult * self.ceiling)
        return self.solve_pool(min(self.n_avail, cap, 2 * pool.k_band), pool)

    def pool_deciding(self, width: int) -> _Pool:
        """A pool that decides ``width``: the cached one, grown as far as needed."""
        pool = self.load_pool()
        if pool is None:
            pool = self.solve_pool(_POOL_K0, None)
        while self.decided_width(pool) < width:
            pool = self.wider(pool)
        return pool

    # ---- the search and the basis it leaves behind --------------------------------------
    def features(self, pool: _Pool):
        """:func:`standardized_modes` of every column of ``pool``, computed once per pool:
        the search tests slices of these arrays and the basis is cut from them."""
        if self._features is None or self._features[0] is not pool:
            self._features = (pool, standardized_modes(pool.v, self.bundle.static))
        return self._features[1]

    def search(self) -> tuple[int | None, _Pool]:
        """The addressability floor, scanning each trial width once on the narrowest pool
        that decides it; ``None`` when no width up to the ceiling addresses the mesh."""
        adm = self.bundle.admissibility
        cell_sel = onp.flatnonzero(onp.asarray(adm.kept)[onp.asarray(adm.cell_comp)])
        pool, done = self.load_pool(), 0
        while True:
            if pool is None:
                pool = self.solve_pool(_POOL_K0, None)
            elif self.decided_width(pool) <= done:
                pool = self.wider(pool)
            decided = self.decided_width(pool)
            if decided > done:
                v_c = self.features(pool)[1]
                floor = addressability_floor(self.cfg, pool.lam, pool.ups, onp.asarray(v_c),
                                             self.n_levels, decided, cell_sel=cell_sel,
                                             n_forced=pool.n_forced, start=done + 1,
                                             extra=self.extra)
                if floor is not None:
                    return floor, pool
                done = decided
            if done >= self.ceiling:
                return None, pool

    def adequate(self, v_c) -> bool:
        """The gate :func:`_assert_encoder_adequate` applies, as a predicate."""
        adm, spec = self.bundle.admissibility, self.cfg.spec
        adq = encoder_adequacy(_gate_block(v_c, self.extra), tol=spec.addressability_tol,
                               cell_comp=adm.cell_comp, kept=adm.kept)
        return (adq["addressable_frac"] >= spec.addressability_min
                and adq["blind_frac"] <= spec.blind_frac_max)

    def ensure_basis(self, floor: int, pool: _Pool | None) -> int | None:
        """Leave an adequate width-``floor`` basis in the eigenbasis cache; returns the width
        actually cached (normally ``floor``), or ``None`` when no width up to the ceiling works.

        A basis already cached for this selection (at width ``floor`` or wider) that passes
        the adequacy gate is kept as it is, so any run or checkpoint trained on it stays
        valid. Otherwise the basis is cut from the search pool's own features
        (:meth:`features`) — exactly the arrays whose addressability the search verified —
        and checked again before it is cached: a basis that fails the gate is never written,
        since every later run would be served it. Should the check fail all the same, the
        next width is tried and the recorded floor moves with it. If the pool is no longer
        the one the floor was found on (cache cleared, solver changed) and it disagrees, the
        floor is searched again on it.
        """
        import jax.numpy as jnp

        import modules.utils.spectral_cache as sc

        def selection_of(width):
            cfg = _dc_replace(self.cfg, n_eig=int(width))
            n_z = resolve_n_eig_z(cfg, self.n_levels)
            sel_key = _selection_key(cfg, self.n_levels)
            # the verdict depends on what the gate sees, so the depth channel joins the pin
            pin = self.memo_key[:2] + (sc.key_digest({"n_eig_z": n_z, **sel_key,
                                                      "gate_depth": depth_channel(cfg),
                                                      "gate_well": self.well_key}),
                                       int(width))
            return cfg, n_z, sel_key, pin

        cfg, n_z, sel_key, pin = selection_of(floor)
        # the memo spares the adequacy check, not the existence check: a cache cleared
        # mid-session must be rebuilt here rather than left to a fresh solve in provision
        if (pin in _PINNED
                and (sc.cached_eig_n_eig(self.cache_dir, self.op_key, n_z, sel_key) or 0) >= floor):
            return floor
        held = sc.load_eig_cache(self.cache_dir, self.op_key, floor, n_z, sel_key)
        if held is not None and self.adequate(held["v_c"]):
            _PINNED.add(pin)
            return floor

        fresh = pool is None
        width = int(floor)
        while True:
            if pool is None or self.decided_width(pool) < width:
                pool = self.pool_deciding(width)
            cfg, n_z, sel_key, pin = selection_of(width)
            sel = _select_retained(pool.lam, pool.ups, width, n_z, cfg.spec.upsilon_vertical,
                                   pool.n_forced)
            v_nodes, v_c, b_v, v_mean, v_std = (a[..., sel] for a in self.features(pool))
            if self.adequate(v_c):
                break
            if fresh:
                # the cached floor was found on another pool: search this one instead
                found, pool = self.search()
                self.remember(found)
                return None if found is None else self.ensure_basis(found, pool)
            width += 1
            if width > self.ceiling:
                self.remember(None)
                return None
        if width != floor:
            self.remember(width)
        sc.save_eig_cache(self.cache_dir, self.op_key, n_eig=int(width), n_eig_z=int(n_z),
                          lam=jnp.asarray(onp.asarray(pool.lam)[sel]), v_nodes=v_nodes,
                          v_c=v_c, b_v=b_v, v_mean=v_mean, v_std=v_std,
                          upsilon=jnp.asarray(onp.asarray(pool.ups)[sel].astype(onp.float32)),
                          sel_key=sel_key)
        _PINNED.add(pin)
        return width


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

    Must run before :func:`~modules.config.resolve`, which fixes ``dim_in`` and the
    parameter count: widening the basis widens the network input, and the auto-capacity
    rule reads that width. Returns ``(cfg, warnings)`` and leaves a matching eigenbasis
    in the cache, so the subsequent :func:`provision` is a cache hit.

    The floor is resolved once per reservoir and setting: it is memoized in-process and
    written to the floor tier of the spectral cache (:mod:`modules.utils.spectral_cache`), and
    the candidate pool it was found on to the pool tier, so a preview, the pipeline build
    that repeats this call, and every later kernel start reuse it without an eigensolve.
    An explicit or already-resolved ``n_eig`` at or above a known floor returns without
    touching the operators at all. The search itself grows its pool from a narrow band
    (:class:`_FloorContext`), so its cost follows the floor rather than the ceiling.
    """
    notes: list[str] = []
    if not needs_eigenbasis(cfg) or cfg.spec.addressability_min <= 0.0:
        return (cfg if cfg.n_eig > 0 else _dc_replace(cfg, n_eig=_FALLBACK_N_EIG)), notes

    ctx = _FloorContext.of(case, cfg)
    floor, pool = ctx.cached_floor(), None
    if floor is _UNRESOLVED:
        floor, pool = ctx.search()
        ctx.remember(floor)
    if floor is not None and cfg.n_eig < floor:
        floor = ctx.ensure_basis(floor, pool)
    if floor is None:
        hint = _depth_hint(cfg, case, ctx.col_id)
        raise SpectralError(
            f"{case_label(cfg)}: no basis width up to spec.n_eig_cap={ctx.ceiling} "
            f"addresses this mesh (network input width d = n_eig + 1 is bounded by it). "
            f"Mesh: {ctx.bundle.admissibility.summary()}. "
            + (hint + " " if hint else "")
            + "Otherwise a demand this large points at the mesh rather than the budget — a "
            "fragmented domain needs a few modes per component, so 24 slabs ask for roughly "
            "50-70 inputs: repair the geometry, raise spec.min_comp_vol_frac to prune more "
            "aggressively, raise spec.n_eig_cap, or relax spec.addressability_min.")
    if floor > _DEPTH_HINT_FLOOR:
        hint = _depth_hint(cfg, case, ctx.col_id)
        if hint:
            notes.append(f"{case_label(cfg)}: the addressability floor is {floor} modes, "
                         f"above {_DEPTH_HINT_FLOOR}. " + hint)

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
    return _dc_replace(cfg, n_eig=floor), notes


def _depth_hint(cfg: RunConfig, case: CaseData, col_id: onp.ndarray) -> str | None:
    """The setting that could lower a large addressability floor, as advice, or ``None``
    when the config already has every option that applies to this grid."""
    if not depth_channel(cfg):
        return ("Set depth_encoding='z': on a layered grid the low modes vary slowly along "
                "a column, so cells stacked in one column share an address until deep in "
                "the spectrum, and one standardized depth channel separates them directly "
                "— the floor then follows the areal structure instead (one extra network "
                "input).")
    if pillar_columns(cfg):
        return None
    pillar_id, _ = column_ids(case, _dc_replace(
        cfg, spec=_dc_replace(cfg.spec, column_grouping="pillar")))
    if int(pillar_id.max(initial=-1)) == int(onp.asarray(col_id).max(initial=-1)):
        return None                       # pillars stand vertical here: nothing to regroup
    return ("Set spec.column_grouping='pillar': this grid's pillars are not vertical, so few "
            "nodes share an areal position and the vertical block is chosen against "
            "coordinate columns that barely stack; grouping the columns along the pillars "
            "makes it a genuine vertical family, which separates stacked cells sooner.")


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

    # Lumped nodal volumes and the volume-weighted nodal reference porosity (row-sum of the
    # phi-weighted mass over V_i), accumulated on the host in float64: a device scatter-add
    # sums in a run-dependent order, and these arrays are baked into every compiled step,
    # so a last-bit difference would make each process compile a different program.
    n_v = int(static["n_vertices"])
    corner = onp.asarray(static["hex_nodes"]).reshape(-1)
    share = onp.einsum("eg,ega->ea", onp.asarray(static["JxW"], onp.float64),
                       onp.asarray(static["N"], onp.float64))
    vol = onp.bincount(corner, weights=share.reshape(-1), minlength=n_v)
    pore = onp.bincount(corner, minlength=n_v, weights=(
        share * onp.asarray(static["poro_hex"], onp.float64)[:, None]).reshape(-1))
    vlump = jnp.asarray(hf.RB_PER_FT3 * vol, jnp.float32)
    vft3 = vlump / case.rb_ft3
    poro_nodes = jnp.asarray(pore / onp.maximum(vol, 1e-30), jnp.float32)

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

    z_nodes = z_c = z_grad = None
    if depth_channel(cfg):
        z_nodes, z_c, z_grad = depth_features(static, onp.asarray(case.verts, onp.float64)[:, 2])
    bundle = SpectralBundle(
        static=static, a_lap=a_lap, m_lap=m_lap, n_null=int(n_null),
        k_hex_diag=static["k_hex_diag"],
        node_xyz=jnp.asarray(case.verts, jnp.float32),
        node_z=jnp.asarray(case.verts, jnp.float32)[:, 2],
        vlump=vlump, vft3=vft3, poro_nodes=poro_nodes,
        centroids=jnp.asarray(case.centroids, jnp.float32),
        admissibility=adm, z_nodes=z_nodes, z_c=z_c, z_grad=z_grad,
    )

    # === (2) eigenbasis (column-sliceable cache): load or solve+cache =====================
    # _resolving stops here: resolve_n_eig needs only the operator tier and the
    # admissibility report, and cfg.n_eig is not yet settled.
    if needs_eigenbasis(cfg) and not _resolving:
        col_id, n_levels = column_ids(case, cfg)
        n_zres = resolve_n_eig_z(cfg, n_levels)
        sel_key = _selection_key(cfg, n_levels)
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
            v_nodes, v_c, b_v, v_mean, v_std = standardized_modes(v_s, static)
        else:
            lam_c, v_cand, ups_c, n_forced = _candidate_spectrum(
                a_lap, m_lap, cfg, int(case.n_nodes), int(n_null), col_id, n_zres)
            sel = _select_retained(lam_c, ups_c, cfg.n_eig, n_zres,
                                   cfg.spec.upsilon_vertical, n_forced)
            lam = jnp.asarray(lam_c[sel])
            upsilon = jnp.asarray(ups_c[sel].astype(onp.float32))
            v_nodes, v_c, b_v, v_mean, v_std = standardized_modes(
                onp.asarray(v_cand)[:, sel], static)
        n_vert = int((onp.asarray(upsilon) > cfg.spec.upsilon_vertical).sum())
        if n_levels > 1 and n_vert < n_zres:
            warnings.warn(
                f"{case_label(cfg)}: retained basis holds {n_vert} vertical modes "
                f"(υ > {cfg.spec.upsilon_vertical}) against a reservation of {n_zres} on a mesh "
                f"with {n_levels} node levels — the candidate band never reached the "
                "vertical family; raise n_eig or n_eig_z (a z-blind basis cannot "
                "represent, supervise, or invert layered saturation structure)",
                RuntimeWarning, stacklevel=2)
        adq = encoder_adequacy(_gate_block(v_c, gate_channels(cfg, case, z_c)),
                               tol=cfg.spec.addressability_tol,
                               cell_comp=adm.cell_comp, kept=adm.kept)
        _assert_encoder_adequate(cfg, adm, adq, n_zres)
        if eig is None:
            # cached only once it passed the gate: a failed basis would otherwise be
            # served to every later run under this key
            sc.save_eig_cache(cache_dir, op_key, n_eig=int(cfg.n_eig), n_eig_z=int(n_zres),
                              lam=lam, v_nodes=v_nodes, v_c=v_c, b_v=b_v,
                              v_mean=v_mean, v_std=v_std, upsilon=upsilon,
                              sel_key=sel_key)
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
            f"({adq['addressable_frac']:.3f}) are resolved from every other cell at a "
            f"separation of {spec.addressability_tol:g} times the mean per-mode spread, "
            f"below the required {spec.addressability_min:g} — distinct cells share one "
            "network input, so no amount of training can give them different states")
    if adq["blind_frac"] > spec.blind_frac_max:
        failures.append(
            f"{adq['blind_frac']:.3f} of cells sit below 1% of the maximum feature "
            f"amplitude (allowed {spec.blind_frac_max:g}) — the encoding cannot "
            "distinguish them from the field mean")
    if not failures:
        return

    worst = sorted(adq["per_component"].items(),
                   key=lambda kv: kv[1]["addressable_frac"])[:4]
    per = "; ".join(f"comp {c}: {s['distinct']}/{s['n_cells']} resolved, "
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


def basis_signature(bundle: SpectralBundle | None) -> str:
    """A digest of the retained eigenbasis (eigenvalues and standardized nodal modes), so
    a checkpoint can tell which basis it was trained on; empty without an eigenbasis."""
    import hashlib

    if bundle is None or bundle.lam is None:
        return ""
    h = hashlib.sha1()
    for a in (bundle.lam, bundle.v_nodes):
        a = onp.ascontiguousarray(onp.asarray(a))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:12]


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
    if bundle.z_c is not None:
        kw["z_c"] = jax.device_put(bundle.z_c, env.repl)
        kw["z_grad"] = jax.device_put(bundle.z_grad, env.repl)
        kw["z_nodes"] = (_put_data(bundle.z_nodes, P("data", None)) if fem
                         else jax.device_put(bundle.z_nodes, env.repl))
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
