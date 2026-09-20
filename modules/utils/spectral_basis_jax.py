r"""Generalized eigenbasis of the hexahedral FEM operators ``A v = lambda M v``.

Given the trilinear-hexahedral stiffness ``A`` and consistent-mass ``M`` operators
assembled by :mod:`hex_fem_assembly_jax`, this module extracts the leading
generalized eigenpairs that define the spectral (Laplace-eigenfunction) positional
encoding of the Delta-PINN. The generalized eigenproblem is

.. math::

    A\,v_k = \lambda_k\,M\,v_k,
    \qquad
    v_j^{\top} M\,v_k = \delta_{jk},

solved in two regimes: a dense Cholesky-plus-symmetric-``eigh`` reduction
(:func:`generalized_eig_basis`) for small meshes, and a sparse implicitly-restarted
Lanczos solve (:func:`generalized_eig_basis_sparse`) that extracts only the requested
extreme modes without forming the dense :math:`(n, n)` factors -- the scalable path to
Norne (61.7k nodes). The number of zero-eigenvalue null modes (one per connected mesh
component) is counted combinatorially from the operator's sparsity graph by
:func:`count_null_modes` so the constant/indicator modes can be dropped cleanly.

where:

- :math:`A`: hexahedral FEM stiffness operator (Neumann Laplace-Beltrami or
  state-dependent per-phase :math:`K_\alpha`), positive semidefinite.
- :math:`M`: hexahedral consistent-mass operator, symmetric positive definite.
- :math:`v_k,\ \lambda_k`: the :math:`k`-th generalized eigenvector / eigenvalue;
  the lowest oscillatory :math:`v_k` form the spectral encoding.
- :math:`\delta_{jk}`: Kronecker delta (the eigenvectors are :math:`M`-orthonormal).

The eigenvectors are gradient-free precomputes: :func:`generalized_eig_basis` wraps its
result in ``stop_gradient`` and the sparse solve runs on the SciPy host, since the
spectral encoding is fixed during training.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.scipy.linalg import solve_triangular
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

REAL = jnp.float32


def generalized_eig_basis(
    K: jnp.ndarray,
    M: jnp.ndarray,
    n_eig: int,
    *,
    which: str = "LM",
    jitter: float = 1e-6,
) -> jnp.ndarray:
    """Return ``n_eig`` generalized eigenvectors of ``K psi = lambda M psi``.

    ``which="LM"`` keeps the largest-eigenvalue modes (mirrors the cached
    ``eigsh(..., which="LM")`` request); ``"SM"`` keeps the smallest.

    Both operators are rescaled to unit mean diagonal before the Cholesky /
    ``eigh`` reduction (the generalized eigenvectors are invariant to a global
    scale on ``K`` or ``M``); this keeps the float32 reduction well-conditioned
    when storage makes ``M`` tiny.  ``M`` is then regularized by
    ``jitter * I`` (in the scaled metric) for SPD safety.  The result is
    ``stop_gradient``-wrapped.
    """
    n = K.shape[0]
    s_k = jnp.maximum(jnp.mean(jnp.diag(K)), 1e-30)
    s_m = jnp.maximum(jnp.mean(jnp.diag(M)), 1e-30)
    K_s = K / s_k
    M_s = M / s_m + jitter * jnp.eye(n, dtype=M.dtype)
    L = jnp.linalg.cholesky(M_s)
    # A = L^{-1} K_s L^{-T}, symmetric.
    Z = solve_triangular(L, K_s, lower=True)
    A = solve_triangular(L, Z.T, lower=True).T
    A = 0.5 * (A + A.T)
    _, V = jnp.linalg.eigh(A)  # eigenvalues ascending
    V_sel = V[:, n - n_eig:] if which == "LM" else V[:, :n_eig]
    psi = solve_triangular(L.T, V_sel, lower=False)  # back-transform to gen. eigvecs
    return jax.lax.stop_gradient(psi)


def count_null_modes(A: "sp.spmatrix") -> int:
    """Null-space dimension of a Neumann FEM/graph Laplacian = number of connected
    components of its sparsity graph.

    A stiffness matrix assembled with strictly positive integration weights
    (``w(x) > 0`` permeability, or unit geometric weights) is positive semidefinite
    with kernel spanned by the per-component *constant* vectors: each block of the
    mesh that is not edge-connected to the rest contributes exactly one zero
    eigenvalue. A connected mesh therefore has a one-dimensional null space (the
    global constant, ``lambda_0 = 0``), whereas a **fragmented reservoir** with
    ``c`` disjoint pieces has a ``c``-dimensional null space of component indicators.

    The count is obtained combinatorially from the operator's sparsity pattern
    (two nodes are adjacent iff they share an element, hence carry a stiffness
    coupling), which is *exact and threshold-free* -- unlike trying to count
    near-zero eigenvalues, which is fragile when the whole spectrum is small
    (e.g. ``lambda ~ 1/L^2`` for coordinates in feet).

    Explicitly-stored zeros are first dropped (``eliminate_zeros``) so that couplings
    voided by zeroed degenerate elements do not register as graph edges; the
    connectivity then reflects the *real* stiffness couplings (a node left with no
    valid element becomes its own component, i.e. an extra null mode).
    """
    A = sp.csr_matrix(A).copy()
    A.eliminate_zeros()
    n_comp, _ = connected_components(A, directed=False)
    return int(n_comp)


def generalized_eig_basis_dense_full(
    A: "sp.spmatrix",
    M: "sp.spmatrix",
    *,
    jitter: float = 1e-8,
    drop_null: bool = True,
    n_null: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Full generalized spectrum of ``A v = lambda M v`` by a dense host solve.

    Small-mesh companion to :func:`generalized_eig_basis_sparse`: densifies the sparse
    FEM operators and returns **every** eigenpair (ascending), which the vertical-aware
    retention rule needs when the reservoir aspect ratio pushes the first vertical
    overtone deep into the spectrum,

    .. math::

        \lambda_{\mathrm{vert},1} \;\approx\; \frac{k_z}{k_x}\Bigl(\frac{L_x}{H_z}\Bigr)^{2}\,\lambda_{\mathrm{areal},1},

    where:

    - :math:`L_x, H_z`: areal extent and thickness of the (pancake) domain — the factor reaches :math:`10^4` on thin sheets, so the vertical family can sit thousands of areal modes deep.
    - :math:`k_z/k_x`: the vertical-to-areal permeability contrast of the stiffness weighting.

    ``drop_null`` removes the ``n_null`` zero modes (connected-component indicators)
    exactly as the sparse path does. Only sensible for :math:`n \lesssim` a few
    thousand nodes (:math:`O(n^3)` time, :math:`O(n^2)` memory); larger meshes use the
    expanding-band sparse path. Returns float32 ``(vals (n - n_null,), vecs (n, n - n_null))``.
    """
    from scipy.linalg import eigh as dense_eigh

    A = sp.csr_matrix(A).astype(np.float64)
    M = sp.csr_matrix(M).astype(np.float64)
    n = A.shape[0]
    s_a = max(float(A.diagonal().mean()), 1e-30)
    s_m = max(float(M.diagonal().mean()), 1e-30)
    Ad = (A / s_a).toarray()
    Md = (M / s_m).toarray() + jitter * np.eye(n)
    vals_s, vecs = dense_eigh(Ad, Md)
    vals = vals_s * (s_a / s_m)
    if drop_null:
        n_null = count_null_modes(A) if n_null is None else int(n_null)
        vals = vals[n_null:]
        vecs = vecs[:, n_null:]
    return np.asarray(vals, dtype=np.float32), np.asarray(vecs, dtype=np.float32)


def generalized_eig_basis_sparse(
    A: "sp.spmatrix",
    M: "sp.spmatrix",
    n_eig: int,
    *,
    which: str = "SM",
    jitter: float = 1e-8,
    drop_null: bool = False,
    n_null: int | None = None,
    null_gap: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse generalized eigenbasis of ``A v = lambda M v`` via SciPy ``eigsh``.

    Large-mesh drop-in for the dense :func:`generalized_eig_basis`: ``A`` and ``M`` are
    ``scipy.sparse`` FEM operators (stiffness / consistent mass) and only the requested
    ``n_eig`` extreme modes are extracted by implicitly-restarted Lanczos -- never
    forming the dense ``(n, n)`` factors that make the Cholesky+``eigh`` path cost
    ``O(n^2)`` memory and ``O(n^3)`` time (≈15 GB / hours at Norne's 61.7k nodes).

    ``which="SM"`` returns the lowest modes (the Laplace-Beltrami encoding). The Neumann
    stiffness is singular -- the constant mode has ``lambda_0 = 0`` -- so the smallest
    modes are found by shift-invert just *below* zero: ``A - sigma M = A + |sigma| M`` is
    SPD and factorizable, whereas ``sigma = 0`` would factor a singular matrix.

    ``drop_null`` (only with ``which="SM"``) removes the zero-eigenvalue null modes so
    the result holds exactly ``n_eig`` *genuinely oscillatory* modes. The number removed
    is the mesh's connected-component count (:func:`count_null_modes`) -- ``1`` for a
    connected mesh (the single constant ``v_0``), ``c`` for a reservoir fragmented into
    ``c`` pieces (whose otherwise non-physical component-indicator modes would pollute the
    encoding). Pass ``n_null`` to override the count; otherwise it is detected. After the
    solve a **spectral-gap** sanity check verifies the dropped block is well separated from
    the retained spectrum: if ``lambda_retained[0] / max|lambda_dropped| < null_gap`` the
    null space is poorly resolved (a corrupted operator -- e.g. degenerate elements breaking
    the constant null mode -- or a weakly-coupled region) and a ``RuntimeWarning`` is raised.

    Both operators are rescaled to unit mean diagonal (the generalized eigenvectors are
    invariant to a global scale on ``A`` or ``M``) and ``M`` is regularized by
    ``jitter * I`` for SPD safety. Returns ``(eigvals (n_eig,), eigvecs (n, n_eig))`` as
    float32, ascending in eigenvalue. This is a one-time, gradient-free precompute (the
    encoding is fixed during training), so a host SciPy solve is appropriate.
    """
    A = sp.csr_matrix(A).astype(np.float64)
    M = sp.csr_matrix(M).astype(np.float64)
    n = A.shape[0]
    n_eig = int(n_eig)

    drop = bool(drop_null) and which == "SM"
    if drop:
        n_null = count_null_modes(A) if n_null is None else int(n_null)
    else:
        n_null = 0
    k = n_eig + n_null                                # solve for the modes we keep + the nulls
    if k >= n:
        raise ValueError(
            f"n_eig+n_null={k} must be < n={n} for an iterative eigensolver."
        )

    s_a = max(float(A.diagonal().mean()), 1e-30)
    s_m = max(float(M.diagonal().mean()), 1e-30)
    A_s = (A / s_a).tocsc()
    M_s = (M / s_m + jitter * sp.identity(n, format="csr")).tocsc()
    if which == "SM":
        # shift just below the spectrum bottom so (A_s - sigma M_s) is SPD -> the
        # returned eigenvalues are those closest to 0, i.e. the lowest modes.
        vals_s, vecs = eigsh(A_s, k=k, M=M_s, sigma=-1e-5, which="LM")
    elif which == "LM":
        vals_s, vecs = eigsh(A_s, k=k, M=M_s, which="LM")
    else:
        raise ValueError(f"which must be 'SM' or 'LM', got {which!r}")
    order = np.argsort(vals_s)
    vals = vals_s[order] * (s_a / s_m)               # undo diagonal rescaling -> physical lambda
    vecs = vecs[:, order]

    if drop and n_null > 0:
        # Spectral-gap sanity check: the dropped null block must sit well below the
        # first retained mode. A weak gap means the null space is poorly resolved
        # (corrupted operator or a near-disconnected region), not cleanly removable.
        dropped_max = float(np.max(np.abs(vals[:n_null])))
        first_keep = float(vals[n_null]) if vals.size > n_null else float("inf")
        gap = first_keep / max(dropped_max, 1e-300)
        if gap < null_gap:
            warnings.warn(
                f"generalized_eig_basis_sparse: weak spectral gap ({gap:.2g}) between the "
                f"{n_null} dropped null mode(s) (|lambda| <= {dropped_max:.2e}) and the "
                f"first retained mode ({first_keep:.2e}); the null space is poorly "
                f"separated -- suspect a corrupted operator (degenerate elements) or a "
                f"weakly-coupled region.",
                RuntimeWarning,
                stacklevel=2,
            )
        vals = vals[n_null:]
        vecs = vecs[:, n_null:]

    return np.asarray(vals, dtype=np.float32), np.asarray(vecs, dtype=np.float32)
