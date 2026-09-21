"""Differentiable JAX assembly of trilinear hexahedral FEM stiffness/mass operators.

This is the 8-node (trilinear) hexahedral FEM assembler.  It is used by
``Laplace-DGM-PINN.ipynb`` to

  1. build the **fixed** Laplace eigenproblem ``A v = lambda M v`` whose lowest
     eigenfunctions become the spectral positional encoding (gNN), and
  2. assemble the **state-dependent** per-phase stiffness ``K_alpha`` used by the
     ``fem_nodal`` black-oil residual.

The reference element is the unit cube ``[0,1]^3`` with the standard VTK
hexahedron corner ordering::

    0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
    4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)

which matches ``reservoir_mesh.corner_cells`` / ``cell_to_unique_vertices``.  Element
integrals use 2x2x2 Gauss quadrature (exact for the rectangular box elements of
the SPE1 grid).  All element geometry (shape values ``N``, physical gradients
``dN``, quadrature weights ``JxW``, scatter indices) is state-independent and is
precomputed once in :func:`build_static_hex_fem`; per call the operators are
reassembled from a per-element diagonal diffusivity / storage coefficient, so the
whole assembly stays differentiable and ``jit``/``vmap``-friendly.

Three realizations of the same operator share :func:`_stiffness_elem_blocks`:
:func:`assemble_stiffness`/:func:`assemble_mass` build the dense ``(n, n)`` matrix
(only viable for small meshes); :func:`stiffness_matvec` applies ``K`` matrix-free in
``O(n_hex)`` memory for the differentiable ``fem_nodal`` residual; and
:func:`assemble_stiffness_sparse`/:func:`assemble_mass_sparse` emit ``scipy.sparse``
operators for the one-time (gradient-free) Laplace eigenbasis at large node counts,
where the dense form would need ``n^2`` storage (~15 GB at Norne's 61.7k nodes).

    K_ij = C_F   sum_e sum_gp JxW * (grad N_i) . diag(d_e) . (grad N_j)
    M_ij = C_V   sum_e sum_gp JxW * s_e * N_i N_j

The FIELD-unit constants ``FIELD_DARCY_COEFF`` (``C_F``) and ``RB_PER_FT3``
(``C_V``) convert mD/cP/psi/ft geometry into surface-volume rates per day and
ft^3 into reservoir barrels; they are global scalings that leave the generalized
eigenvectors unchanged but give the ``fem_nodal`` residual physical units.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from modules.utils.blackoil_closures import FIELD_DARCY_COEFF, RB_PER_FT3

REAL = jnp.float32

# Local coordinates of the 8 corners on the unit cube (VTK hexahedron ordering).
_NODE_LOCAL = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=np.float64,
)

# 2-point Gauss rule on [0,1]: nodes 1/2 +- 1/(2 sqrt 3), weight 1/2 each.
_G1D = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)], dtype=np.float64)
_W1D = np.array([0.5, 0.5], dtype=np.float64)


def _ref_shape(local_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trilinear ``N`` (n_pts, 8) and reference gradients ``dNref`` (n_pts, 8, 3).

    For corner ``a`` with local coordinate ``(l, m, n) in {0,1}^3`` the shape
    function is the product ``f(xi, l) f(eta, m) f(zeta, n)`` with
    ``f(t, 0) = 1 - t`` and ``f(t, 1) = t``.
    """
    local_pts = np.atleast_2d(np.asarray(local_pts, dtype=np.float64))  # (P, 3)
    xi, eta, zeta = local_pts[:, 0], local_pts[:, 1], local_pts[:, 2]

    def f(t, code):   # value of the 1-D factor
        return np.where(code == 1, t, 1.0 - t)

    def g(t, code):   # derivative of the 1-D factor
        return np.where(code == 1, np.ones_like(t), -np.ones_like(t))

    P = local_pts.shape[0]
    N = np.zeros((P, 8), dtype=np.float64)
    dN = np.zeros((P, 8, 3), dtype=np.float64)
    for a, (la, ma, na) in enumerate(_NODE_LOCAL):
        fx, fy, fz = f(xi, la), f(eta, ma), f(zeta, na)
        N[:, a] = fx * fy * fz
        dN[:, a, 0] = g(xi, la) * fy * fz
        dN[:, a, 1] = fx * g(eta, ma) * fz
        dN[:, a, 2] = fx * fy * g(zeta, na)
    return N, dN


def _gauss_2x2x2() -> tuple[np.ndarray, np.ndarray]:
    """Return the 8 Gauss points (8, 3) and weights (8,) on the unit cube."""
    pts, wts = [], []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                pts.append([_G1D[i], _G1D[j], _G1D[k]])
                wts.append(_W1D[i] * _W1D[j] * _W1D[k])
    return np.asarray(pts, dtype=np.float64), np.asarray(wts, dtype=np.float64)


# Map a corner's sign-bit code ``b_x + 2 b_y + 4 b_z`` to its VTK reference slot.
_CODE_TO_SLOT = np.array([0, 1, 3, 2, 4, 5, 7, 6], dtype=np.int64)


def canonical_hexes_from_corner_cells(
    verts: np.ndarray,
    hexes: np.ndarray,
    corner_cells: np.ndarray,
) -> np.ndarray:
    """Recover VTK corner ordering from the mesh's canonical ``corner_cells``.

    ``cell_to_unique_vertices`` (the ``hexes`` rows) stores each cell's 8 vertex ids
    sorted by id, so the corner order is lost.  ``reservoir_mesh.corner_cells`` holds the
    same 8 corners *in canonical VTK order* (built via ``DEEPFIELD_TO_CANONICAL``), so
    we assign each canonical slot the cell's own vertex nearest to that corner. This
    is pure coordinate identity (no axis-aligned / sign heuristic), so it works for
    arbitrary skewed corner-point hexahedra.

    Degenerate (pinched) cells map several slots to the same vertex; the resulting
    repeated ids yield a non-positive Jacobian which :func:`build_static_hex_fem`
    zeroes, exactly as for the legacy path.
    """
    verts = np.asarray(verts, dtype=np.float64)
    hexes = np.asarray(hexes, dtype=np.int64)
    corner_cells = np.asarray(corner_cells, dtype=np.float64)
    if hexes.shape != corner_cells.shape[:2]:
        raise ValueError(
            f"hexes {hexes.shape} and corner_cells {corner_cells.shape} are inconsistent; "
            "expected matching (n_hex, 8)."
        )
    cand_xyz = verts[hexes]                                            # (n_hex, 8, 3) cell's own verts
    # distance[e, slot, cand] between canonical corner and each candidate vertex
    dist = np.linalg.norm(cand_xyz[:, None, :, :] - corner_cells[:, :, None, :], axis=-1)
    pick = np.argmin(dist, axis=2)                                     # (n_hex, 8) index into candidates
    return np.take_along_axis(hexes, pick, axis=1)


def build_static_hex_fem(
    verts: np.ndarray,
    hexes: np.ndarray,
    k_cell_diag: np.ndarray,
    poro_cell: np.ndarray,
    hex_to_cell: np.ndarray | None = None,
    corner_cells: np.ndarray | None = None,
    jac_rel_tol: float = 1e-6,
) -> dict[str, Any]:
    """Precompute state-independent trilinear-hex FEM geometry and rock arrays.

    Parameters
    ----------
    verts : (n_vertices, 3) node coordinates.
    hexes : (n_hex, 8) hex connectivity (global node ids, VTK ordering).
    k_cell_diag : (n_cells, 3) diagonal absolute permeability per cell.
    poro_cell : (n_cells,) porosity per cell.
    hex_to_cell : (n_hex,) parent cell of each hex (identity if ``None``).
    corner_cells : (n_hex, 8, 3) canonical VTK-ordered corner coordinates
        (``reservoir_mesh.corner_cells``). When supplied, the corner ordering is recovered
        from these by coordinate identity, supporting arbitrary skewed corner-point
        hexahedra. When ``None``, the legacy sign-based ``canonicalize_hex_order`` is
        used, which requires axis-aligned cells.
    jac_rel_tol : scaled-Jacobian floor for element validity. An element is kept only
        if *every* Gauss point has ``det(J) > jac_rel_tol * V_bbox`` (``V_bbox`` = the
        cell's axis-aligned bounding-box volume); otherwise the **whole** element is
        zeroed. See note below.

    Degenerate / inverted / pinched elements are zeroed so they contribute nothing to
    the assembled operators. Validity is **per element, not per Gauss point**: a single
    non-finite, non-positive, or *collapsed* Jacobian point (``det(J) <= jac_rel_tol *
    V_bbox``) condemns the entire element. Keeping a collapsed cell's remaining points
    would scatter ``~1/det`` stiffness blow-ups (corner-point grids routinely contain
    zero-thickness cells at pinch-outs/faults) that, in float32, destroy the FEM
    partition of unity ``sum_a grad N_a = 0`` and hence the constant null mode that the
    spectral encoding relies on. The number of zeroed elements is returned as
    ``n_degenerate``.
    """
    verts = np.asarray(verts, dtype=np.float64)
    hexes = np.asarray(hexes, dtype=np.int64)
    n_vertices = int(verts.shape[0])
    n_hex = int(hexes.shape[0])
    if hex_to_cell is None:
        hex_to_cell = np.arange(n_hex, dtype=np.int64)
    hex_to_cell = np.asarray(hex_to_cell, dtype=np.int64)

    # Rebuild a consistent VTK corner ordering (the source connectivity is not
    # uniformly ordered across cells); required for the fixed reference element.
    if corner_cells is not None:
        # Authoritative canonical order from the mesh; works for skewed corner-point
        # hexahedra. Inverted/degenerate cells are caught by the det>0 check below.
        hexes = canonical_hexes_from_corner_cells(verts, hexes, corner_cells)
    else:
        hexes, bijective = canonicalize_hex_order(verts, hexes)
        if not bijective:
            raise ValueError(
                "canonicalize_hex_order: some cells are not axis-aligned boxes; the "
                "sign-based reference ordering does not apply to this mesh. Pass "
                "corner_cells=reservoir_mesh.corner_cells to support general hexahedra."
            )

    gp, gw = _gauss_2x2x2()                       # (8,3), (8,)
    N_gp, dNref_gp = _ref_shape(gp)               # (8,8), (8,8,3)
    n_gp = gp.shape[0]

    node_coords = verts[hexes]                    # (n_hex, 8, 3)

    # Per-element characteristic volume (axis-aligned bounding box) makes the
    # degeneracy test scale-relative: a collapsed/pinched cell has |det(J)| tiny
    # (or sign-indefinite) compared with its spatial extent.
    bbox = node_coords.max(axis=1) - node_coords.min(axis=1)  # (n_hex, 3)
    vbox = np.prod(bbox, axis=1)                              # (n_hex,)

    # Jacobian at each Gauss point: J[e,gp,i,j] = sum_a coords[e,a,i] dNref[gp,a,j].
    jac = np.einsum("eai,gaj->egij", node_coords, dNref_gp)   # (n_hex, n_gp, 3, 3)
    det = np.linalg.det(jac)                                  # (n_hex, n_gp)
    finite = np.all(np.isfinite(jac), axis=(-1, -2))          # (n_hex, n_gp)
    # Element-level validity: keep an element only if EVERY Gauss point is finite and
    # has a strictly positive, non-collapsed scaled Jacobian (det > tol * V_bbox). A
    # single bad point zeroes the whole element (see build_static_hex_fem docstring).
    gp_ok = finite & (det > float(jac_rel_tol) * vbox[:, None])   # (n_hex, n_gp)
    elem_ok = np.all(gp_ok, axis=1)                          # (n_hex,)
    n_degenerate = int((~elem_ok).sum())
    valid = np.broadcast_to(elem_ok[:, None], det.shape)     # (n_hex, n_gp)
    jac_safe = np.where(valid[..., None, None], jac, np.broadcast_to(np.eye(3), jac.shape))
    jinv = np.linalg.inv(jac_safe)                           # (n_hex, n_gp, 3, 3)

    # Physical gradients: dN[e,gp,a,i] = sum_j dNref[gp,a,j] Jinv[e,gp,j,i].
    dN_phys = np.einsum("gaj,egji->egai", dNref_gp, jinv)    # (n_hex, n_gp, 8, 3)
    JxW = np.abs(det) * gw[None, :]                          # (n_hex, n_gp)
    JxW = np.where(valid, JxW, 0.0)

    # Broadcast the (constant-over-elements) shape values to every element.
    N_elem = np.broadcast_to(N_gp[None, :, :], (n_hex, n_gp, 8)).copy()

    # Centroid (xi=eta=zeta=1/2) physical gradient operator, for the chain-rule map;
    # zeroed on the same degenerate elements so the affine encoder gets no spurious
    # gradient there.
    N_c, dNref_c = _ref_shape(np.array([[0.5, 0.5, 0.5]]))   # (1,8), (1,8,3)
    jac_c = np.einsum("eai,aj->eij", node_coords, dNref_c[0])         # (n_hex,3,3)
    jac_c_safe = np.where(elem_ok[..., None, None], jac_c, np.broadcast_to(np.eye(3), jac_c.shape))
    jinv_c = np.linalg.inv(jac_c_safe)
    dN_centroid = np.einsum("aj,eji->eai", dNref_c[0], jinv_c)        # (n_hex,8,3)
    dN_centroid = np.where(elem_ok[:, None, None], dN_centroid, 0.0)

    # Scatter indices for the 8x8 element blocks.
    rows = np.broadcast_to(hexes[:, :, None], (n_hex, 8, 8)).reshape(-1)
    cols = np.broadcast_to(hexes[:, None, :], (n_hex, 8, 8)).reshape(-1)

    k_cell_diag = np.asarray(k_cell_diag, dtype=np.float64)
    poro_cell = np.asarray(poro_cell, dtype=np.float64)
    return {
        "N": jnp.asarray(N_elem, dtype=REAL),                 # (n_hex, n_gp, 8)
        "dN": jnp.asarray(dN_phys, dtype=REAL),               # (n_hex, n_gp, 8, 3)
        "JxW": jnp.asarray(JxW, dtype=REAL),                  # (n_hex, n_gp)
        "dN_centroid": jnp.asarray(dN_centroid, dtype=REAL),  # (n_hex, 8, 3)
        "hex_nodes": jnp.asarray(hexes, dtype=jnp.int32),     # (n_hex, 8)
        "rows": jnp.asarray(rows, dtype=jnp.int32),           # (64*n_hex,)
        "cols": jnp.asarray(cols, dtype=jnp.int32),           # (64*n_hex,)
        "k_hex_diag": jnp.asarray(k_cell_diag[hex_to_cell], dtype=REAL),  # (n_hex, 3)
        "poro_hex": jnp.asarray(poro_cell[hex_to_cell], dtype=REAL),      # (n_hex,)
        "n_vertices": n_vertices,
        "n_hex": n_hex,
        "n_degenerate": n_degenerate,             # elements zeroed for a bad Jacobian
    }


# ---------------------------------------------------------------------------------------------
# Node <-> element-local transfer as a linear primitive pair with gather-only transposes
# ---------------------------------------------------------------------------------------------
#
# The matrix-free operators move data between the nodal vector and the element-local
# (n_hex, 8) layout twice per action: a gather ``vec[hex_nodes]`` on the way in and a
# scatter-add on the way out. Under reverse-mode AD each transposes into the other, so a
# residual Jacobian assembled by vjp sweeps (dozens of covectors batched by vmap) runs
# batched float64 scatter-adds whose indices collide eight-fold at every interior node --
# contended atomics, the slowest path a GPU has. Both transfers are pure linear maps of a
# fixed incidence, so each is registered as a primitive whose transpose is *the other
# transfer expressed as a gather*: the scatter-add of local values into nodes is
# evaluated as a gather-and-sum over each node's (at most ``max_deg``) incident slots,
# and the gather of nodal values into elements transposes into that same gather-sum.
# Forward values are unchanged (a fixed summation order replaces the atomics' arbitrary
# one); nothing on the reverse tape scatters any more.

_hex_take_p = jax.extend.core.Primitive("hex_take")    # nodes -> local, (n, ...) -> (n_hex, 8, ...)
_hex_gsum_p = jax.extend.core.Primitive("hex_gsum")    # local -> nodes, (n_hex, 8, ...) -> (n, ...)


def _hex_take_impl(vec, nodes, inc):
    return jnp.take(vec, nodes, axis=0)


def _hex_gsum_impl(loc, nodes, inc):
    flat = loc.reshape((-1,) + loc.shape[2:])
    flat = jnp.concatenate([flat, jnp.zeros((1,) + flat.shape[1:], flat.dtype)])   # padding slot
    return jnp.take(flat, inc, axis=0).sum(axis=1)


def _register_transfer(p, impl, other):
    from jax.interpreters import ad, batching, mlir

    p.def_impl(impl)
    p.def_abstract_eval(
        lambda x, nodes, inc: jax.core.ShapedArray(jax.eval_shape(impl, x, nodes, inc).shape, x.dtype))
    mlir.register_lowering(p, mlir.lower_fun(impl, multiple_results=False))

    def jvp(primals, tangents):
        x, nodes, inc = primals
        return p.bind(x, nodes, inc), p.bind(tangents[0], nodes, inc)
    ad.primitive_jvps[p] = jvp
    ad.primitive_transposes[p] = lambda ct, x, nodes, inc: [other.bind(ct, nodes, inc), None, None]

    def batch(args, dims):
        x, nodes, inc = args
        if dims[1] is not None or dims[2] is not None:
            raise NotImplementedError("hex transfer: the incidence arrays cannot be batched")
        out = p.bind(jnp.moveaxis(x, dims[0], -1), nodes, inc)   # batch axis last: the
        return out, out.ndim - 1                                  # transfer acts on leading axes
    batching.primitive_batchers[p] = batch


_register_transfer(_hex_take_p, _hex_take_impl, _hex_gsum_p)
_register_transfer(_hex_gsum_p, _hex_gsum_impl, _hex_take_p)


def node_incidence(static: dict[str, Any]) -> jnp.ndarray:
    """``(n_vertices, max_deg)`` int32 slots of the flattened ``(n_hex * 8)`` local layout
    incident to each node, padded with the index ``n_hex * 8`` (a zero slot). Built once
    from ``hex_nodes`` with NumPy and memoized on the static dict; call it OUTSIDE jit."""
    inc = static.get("node_inc")
    if inc is not None:
        return inc
    nodes = np.asarray(static["hex_nodes"]).reshape(-1)
    n = int(static["n_vertices"])
    order = np.argsort(nodes, kind="stable")
    counts = np.bincount(nodes, minlength=n)
    max_deg = int(counts.max()) if counts.size else 0
    inc_np = np.full((n, max_deg), nodes.size, dtype=np.int32)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.arange(nodes.size) - np.repeat(starts, counts)     # position within the node's run
    inc_np[nodes[order], rank] = order
    static["node_inc"] = jnp.asarray(inc_np)
    return static["node_inc"]


def local_of_nodes(vec: jnp.ndarray, static: dict[str, Any]) -> jnp.ndarray:
    """``vec[hex_nodes]`` -- the nodal-to-local gather whose transpose is a gather-sum
    (no scatter on the reverse tape). ``vec`` is ``(n_vertices, ...)``."""
    return _hex_take_p.bind(vec, static["hex_nodes"], node_incidence(static))


def nodes_of_local(loc: jnp.ndarray, static: dict[str, Any]) -> jnp.ndarray:
    """Scatter-add of ``(n_hex, 8, ...)`` element-local values into ``(n_vertices, ...)``,
    evaluated as a per-node gather-sum over the incidence; its transpose is the plain
    gather :func:`local_of_nodes`."""
    return _hex_gsum_p.bind(loc, static["hex_nodes"], node_incidence(static))


def _stiffness_elem_blocks(
    d_hex_diag: jnp.ndarray,
    static: dict[str, Any],
    *,
    dtype: Any = None,
) -> jnp.ndarray:
    """Per-hex 8x8 stiffness blocks ``K_e = sum_gp JxW (dN diag(d)) dN^T`` (differentiable).

    Shared by the dense assembler, the matrix-free action, and (via NumPy) the sparse
    assembler, so all three realize the *same* operator from a per-element diagonal
    diffusivity ``(n_hex, 3)``.  ``dtype=None`` keeps the historical :data:`REAL`
    (float32) arithmetic bit-exactly; a promoted dtype casts the static geometry
    (``dN``, ``JxW``) TRANSIENTLY inside the op -- the resident STATIC arrays stay
    float32, only the traced element-block computation is widened.
    """
    dt = REAL if dtype is None else dtype
    dN = jnp.asarray(static["dN"], dtype=dt)   # (n_hex, n_gp, 8, 3) transient cast
    JxW = jnp.asarray(static["JxW"], dtype=dt)  # (n_hex, n_gp)
    d = jnp.asarray(d_hex_diag, dtype=dt)       # (n_hex, 3)
    # weighted = dN[...,a,i] * d[e,i]; contract i with dN[...,b,i].
    weighted = dN * d[:, None, None, :]
    k_gp = jnp.einsum("egai,egbi->egab", weighted, dN)         # (n_hex, n_gp, 8, 8)
    return jnp.einsum("eg,egab->eab", JxW, k_gp)               # (n_hex, 8, 8)


def stiffness_axis_blocks(static: dict[str, Any], *, dtype: Any = None) -> jnp.ndarray:
    r"""State-independent per-axis element stiffness blocks ``(n_hex, 3, 8, 8)``.

    The element stiffness is linear in the per-element diagonal diffusivity, so the
    quadrature can be carried out once per axis and reused by every matrix-free action:

    .. math::

        K_e(d_e) \;=\; \sum_{i=1}^{3} d_{e,i}\, K_e^{(i)},
        \qquad
        K^{(i)}_{e,ab} \;=\; \sum_{gp} (JW)_{e,gp}\,
        \partial_i N_a(gp)\, \partial_i N_b(gp)

    where:

    - :math:`d_{e,i}`: the state-dependent diffusivity of element :math:`e` along axis :math:`i` (mobility :math:`\times` rock permeability in the black-oil residual).
    - :math:`K_e^{(i)}`: the axis-:math:`i` block, a pure geometry integral evaluated here from the static ``dN``/``JxW`` arrays in the requested ``dtype`` (float64 for the promoted ``fem_nodal`` path).
    - :math:`(JW)_{e,gp}`: the quadrature Jacobian-weight products.

    Feeding these to :func:`stiffness_matvec` via ``axis_blocks`` replaces the per-call
    ``(n_hex, n_gp, 8, 8, 3)`` quadrature contraction by a ``(n_hex, 3)``-weighted sum
    of stored blocks — roughly ten times fewer flops and bytes, which is what makes the
    float64 action affordable on GPUs whose float64 throughput is a small fraction of
    float32 (T4: 1/32). The result is the same operator up to float64 reassociation.
    Compute it ONCE outside jit (it is a constant of the mesh) and close over it.
    """
    dt = REAL if dtype is None else dtype
    dN = jnp.asarray(static["dN"], dtype=dt)     # (n_hex, n_gp, 8, 3)
    JxW = jnp.asarray(static["JxW"], dtype=dt)   # (n_hex, n_gp)
    return jnp.einsum("eg,egai,egbi->eiab", JxW, dN, dN)


def stiffness_matvec(
    d_hex_diag: jnp.ndarray,
    vec: jnp.ndarray,
    static: dict[str, Any],
    *,
    field_darcy_coeff: float = FIELD_DARCY_COEFF,
    dtype: Any = None,
    axis_blocks: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Matrix-free per-hex stiffness action ``(K @ vec)`` (no global matrix formed).

    Identical operator to :func:`assemble_stiffness` followed by a mat-vec, but the
    ``(n, n)`` matrix is never materialized: the 8x8 element blocks are applied to the
    gathered local vector and scatter-added back, so memory is ``O(n_hex)`` instead of
    ``O(n^2)``. Fully differentiable w.r.t. both ``d_hex_diag`` (the state-dependent
    mobility x rock) and ``vec`` (the nodal potential), so the ``fem_nodal`` residual
    keeps its gradients at Norne scale where the dense matrix is ~15 GB.

    ``dtype=None`` keeps the historical :data:`REAL` (float32) arithmetic bit-exactly.
    A promoted dtype (e.g. float64 under a selective-f64 policy) widens the element
    blocks and -- critically -- the scatter-add ACCUMULATION: the physical signal in
    ``K @ vec`` is potential differences riding on large absolute potentials (partition
    of unity makes the constant mode cancel), so float32 accumulation carries an
    amplified cancellation noise floor that a float64 accumulator removes.

    ``axis_blocks`` (from :func:`stiffness_axis_blocks`, same ``dtype``) supplies the
    precomputed per-axis geometry integrals, so the element blocks are the cheap
    diffusivity-weighted sum ``K_e = sum_i d_ei K_e^(i)`` instead of a fresh quadrature
    contraction; ``None`` keeps the per-call assembly.
    """
    dt = REAL if dtype is None else dtype
    n = static["n_vertices"]
    nodes = static["hex_nodes"]                            # (n_hex, 8)
    if axis_blocks is None:
        v_loc = jnp.take(jnp.asarray(vec, dtype=dt), nodes, axis=0)   # (n_hex, 8) local potentials
        k_elem = _stiffness_elem_blocks(d_hex_diag, static, dtype=dt)   # (n_hex, 8, 8)
        y_loc = jnp.einsum("eab,eb->ea", k_elem, v_loc)    # (n_hex, 8) local stiffness action
        out = jnp.zeros((n,), dtype=dt).at[nodes.reshape(-1)].add(y_loc.reshape(-1))
    else:
        # sum-factorized: z_eia = K^(i)_e v_e first, then the diffusivity-weighted sum.
        # The 8x8 element block is never formed, so a (batched) reverse sweep carries
        # (n_hex, 3, 8) cotangents instead of (n_hex, 8, 8) ones; the node<->local
        # transfers are the gather-transposed primitives (no scatter on the tape).
        v_loc = local_of_nodes(jnp.asarray(vec, dtype=dt), static)
        z = jnp.einsum("eiab,eb->eia", jnp.asarray(axis_blocks, dtype=dt), v_loc)
        y_loc = jnp.einsum("ei,eia->ea", jnp.asarray(d_hex_diag, dtype=dt), z)
        out = nodes_of_local(y_loc, static)
    return jnp.asarray(field_darcy_coeff, dtype=dt) * out


def assemble_stiffness_sparse(
    d_hex_diag: jnp.ndarray,
    static: dict[str, Any],
    *,
    field_darcy_coeff: float = FIELD_DARCY_COEFF,
) -> "sp.csr_matrix":
    """SciPy CSR ``(n, n)`` stiffness for the (gradient-free) one-time eigenbasis build.

    Same element integrals as :func:`assemble_stiffness`, but the 8x8 blocks are
    scatter-added into a ``scipy.sparse`` matrix (duplicate ``(row, col)`` pairs are
    summed on CSR conversion) instead of a dense buffer. A trilinear hex couples each
    node to <=27 neighbours, so ``A`` holds a few-million nonzeros (tens of MB) rather
    than the ``n^2`` (~15 GB at Norne) of the dense path. Built in float64 on the host
    for the SciPy generalized eigensolver -- no autodiff flows through the eigenbasis.
    """
    n = int(static["n_vertices"])
    k_elem = np.asarray(_stiffness_elem_blocks(d_hex_diag, static), dtype=np.float64)  # (n_hex,8,8)
    rows = np.asarray(static["rows"]); cols = np.asarray(static["cols"])
    data = float(field_darcy_coeff) * k_elem.reshape(-1)
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def assemble_mass_sparse(
    s_hex: jnp.ndarray,
    static: dict[str, Any],
    *,
    rb_per_ft3: float = RB_PER_FT3,
) -> "sp.csr_matrix":
    """SciPy CSR ``(n, n)`` consistent mass for the one-time eigenbasis build.

    Sparse counterpart of :func:`assemble_mass`; pass ``s_hex = 1`` for the
    Laplace-Beltrami mass ``int N_i N_j``.
    """
    n = int(static["n_vertices"])
    N = np.asarray(static["N"], dtype=np.float64)              # (n_hex, n_gp, 8)
    JxW = np.asarray(static["JxW"], dtype=np.float64)          # (n_hex, n_gp)
    s = np.asarray(s_hex, dtype=np.float64)                    # (n_hex,)
    m_gp = np.einsum("ega,egb->egab", N, N)                    # (n_hex, n_gp, 8, 8)
    m_elem = np.einsum("eg,egab->eab", JxW * s[:, None], m_gp)
    rows = np.asarray(static["rows"]); cols = np.asarray(static["cols"])
    data = float(rb_per_ft3) * m_elem.reshape(-1)
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def lumped_node_volumes(
    static: dict[str, Any],
    *,
    rb_per_ft3: float = RB_PER_FT3,
) -> jnp.ndarray:
    """Diagonal lumped nodal control volumes ``(n_vertices,)`` in reservoir barrels.

    Row-sum (mass lumping) of the ``s = 1`` consistent mass::

        V^lump_i = RB_PER_FT3 * sum_e sum_gp JxW_e,gp N_i(gp).

    Partition of unity (``sum_a N_a = 1``) guarantees ``sum_i V^lump_i`` equals
    the total geometric volume (in barrels).  Used to integrate the per-unit-volume
    storage rate ``d/dt(phi S_alpha / B_alpha)`` over each node's control volume so
    the accumulation shares the surface-volume/day units of ``K_alpha p_alpha``.
    """
    n = static["n_vertices"]
    nodes = static["hex_nodes"]              # (n_hex, 8)
    N = static["N"]                          # (n_hex, n_gp, 8)
    JxW = static["JxW"]                       # (n_hex, n_gp)
    contrib = jnp.einsum("eg,ega->ea", JxW, N)   # (n_hex, 8) nodal volume share
    out = jnp.zeros((n,), dtype=REAL)
    out = out.at[nodes.reshape(-1)].add(contrib.reshape(-1))
    return jnp.asarray(rb_per_ft3, dtype=REAL) * out


def mass_elem_blocks(static: dict[str, Any], *, dtype: Any = None) -> jnp.ndarray:
    r"""State-independent consistent-mass element blocks ``(n_hex, 8, 8)``.

    .. math::

        M_{e,ab} \;=\; \sum_{gp} (JW)_{e,gp}\, N_a(gp)\, N_b(gp)

    where:

    - :math:`N_a`: the trilinear shape functions at the quadrature points.
    - :math:`(JW)_{e,gp}`: the quadrature Jacobian-weight products.

    The geometric (``s_hex=None``) mass of :func:`mass_matvec` is a pure mesh constant,
    so its quadrature is carried out once here and reused through ``elem_blocks``.
    Compute it outside jit and close over it.
    """
    dt = REAL if dtype is None else dtype
    N = jnp.asarray(static["N"], dtype=dt)                          # (n_hex, n_gp, 8)
    JxW = jnp.asarray(static["JxW"], dtype=dt)                      # (n_hex, n_gp)
    return jnp.einsum("eg,ega,egb->eab", JxW, N, N)


def mass_matvec(
    vec: jnp.ndarray,
    static: dict[str, Any],
    *,
    s_hex: jnp.ndarray | None = None,
    rb_per_ft3: float = RB_PER_FT3,
    dtype: Any = None,
    elem_blocks: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Matrix-free consistent-mass action ``M @ vec`` (no global matrix formed).

    Applies the trilinear-hex consistent mass

    .. math::

        (M v)_i \;=\; \sum_e \sum_{gp} (J W)_{e,gp}\, s_e\, N_i(gp) \sum_a N_a(gp)\, v_{a},

    where:

    - :math:`s_e`: an optional per-hex weight (``None`` = 1, the geometric mass); the storage-weighted variant takes the per-hex porosity.
    - :math:`N_a`: the trilinear shape functions; :math:`(JW)_{e,gp}` the quadrature Jacobian-weight products.
    - the result carries the same reservoir-barrel scaling as :func:`lumped_node_volumes`, so ``mass_matvec(v)`` and ``vlump * v`` differ only by the off-diagonal (consistent) coupling — an :math:`O(h^2)` refinement of the lumped pairing.

    ``vec`` may be ``(n_vertices,)`` or ``(n_vertices, F)``; fully differentiable in
    ``vec`` (and ``s_hex``). ``dtype=None`` keeps :data:`REAL` float32 arithmetic.
    ``elem_blocks`` (from :func:`mass_elem_blocks`, same ``dtype``; geometric mass
    only, i.e. ``s_hex=None``) skips the per-call quadrature contraction.
    """
    dt = REAL if dtype is None else dtype
    n = static["n_vertices"]
    nodes = static["hex_nodes"]                                     # (n_hex, 8)
    v = jnp.asarray(vec, dtype=dt)
    squeeze = v.ndim == 1
    if squeeze:
        v = v[:, None]
    if elem_blocks is not None and s_hex is None:
        v_loc = local_of_nodes(v, static)                               # (n_hex, 8, F)
        y_loc = jnp.einsum("eab,ebf->eaf", jnp.asarray(elem_blocks, dtype=dt), v_loc)
        out = nodes_of_local(y_loc, static)
    else:
        v_loc = jnp.take(v, nodes, axis=0)                              # (n_hex, 8, F)
        N = jnp.asarray(static["N"], dtype=dt)                          # (n_hex, n_gp, 8)
        JxW = jnp.asarray(static["JxW"], dtype=dt)                      # (n_hex, n_gp)
        w = JxW if s_hex is None else JxW * jnp.asarray(s_hex, dt)[:, None]
        gp_vals = jnp.einsum("ega,eaf->egf", N, v_loc)                  # (n_hex, n_gp, F)
        y_loc = jnp.einsum("ega,eg,egf->eaf", N, w, gp_vals)            # (n_hex, 8, F)
        out = jnp.zeros((n, v.shape[1]), dtype=dt).at[nodes.reshape(-1)].add(
            y_loc.reshape(-1, v.shape[1]))
    out = jnp.asarray(rb_per_ft3, dtype=dt) * out
    return out[:, 0] if squeeze else out


def centroid_features_and_gradients(
    nodal_field: jnp.ndarray,
    static: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate a nodal field and its physical gradient at every hex centroid.

    Parameters
    ----------
    nodal_field : (n_vertices, F) values at the mesh nodes (e.g. eigenfeatures).

    Returns
    -------
    v_c : (n_hex, F) centroid values (mean of the 8 corner values).
    B_v : (n_hex, 3, F) centroid physical gradient ``dv/dx``, where
        ``B_v[e] = dN_centroid[e]^T @ nodal_field[nodes_e]``.
    """
    nodal_field = jnp.asarray(nodal_field, dtype=REAL)
    nodes = static["hex_nodes"]              # (n_hex, 8)
    gathered = jnp.take(nodal_field, nodes, axis=0)          # (n_hex, 8, F)
    v_c = jnp.mean(gathered, axis=1)                         # (n_hex, F)
    B_v = jnp.einsum("eai,eaf->eif", static["dN_centroid"], gathered)  # (n_hex, 3, F)
    return v_c, B_v
