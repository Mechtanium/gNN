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
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, eigsh, splu

REAL = jnp.float32


def m_normalize(M: "sp.spmatrix", V: np.ndarray) -> np.ndarray:
    r"""Scale every column to unit :math:`M`-norm and fix its sign.

    .. math::

        \hat{\mathbf v}_k \;=\; \frac{s_k\, \mathbf v_k}{\sqrt{\mathbf v_k^{\top} M\, \mathbf v_k}},
        \qquad
        s_k \;=\; \operatorname{sign}\bigl(v_{i^\star k}\bigr),
        \quad i^\star = \arg\max_i |v_{ik}|

    where:
    - :math:`\mathbf v_k`: a generalized eigenvector as a solver returns it; the solvers work on operators rescaled to unit mean diagonal, which leaves the eigenvectors normalized in the scaled metric :math:`M / s_m` rather than in :math:`M`.
    - :math:`s_k`: the sign that makes the largest-magnitude entry positive, so the same mode carries the same sign whichever solver produced it.

    Applied by every solver path, it gives one normalization for the whole basis,
    :math:`\hat V^\top M \hat V = I` (to the accuracy of the solve), so a mode's Galerkin
    row carries no factor that depends on which solver found it. Returns float64.
    """
    V = np.asarray(V, np.float64)
    if V.size == 0:
        return V
    MV = sp.csr_matrix(M).astype(np.float64) @ V
    norms = np.sqrt(np.maximum(np.einsum("ij,ij->j", V, MV), 1e-300))
    V = V / norms
    sign = np.sign(V[np.abs(V).argmax(axis=0), np.arange(V.shape[1])])
    return V * np.where(sign == 0, 1.0, sign)


def generalized_eig_basis(
    K: jnp.ndarray,
    M: jnp.ndarray,
    n_eig: int,
    *,
    which: str = "LM",
    jitter: float = 1e-6,
) -> jnp.ndarray:
    r"""Return ``n_eig`` generalized eigenvectors of ``K psi = lambda M psi``.

    ``which="LM"`` keeps the largest-eigenvalue modes (mirrors the cached
    ``eigsh(..., which="LM")`` request); ``"SM"`` keeps the smallest.

    Both operators are rescaled to unit mean diagonal before the Cholesky /
    ``eigh`` reduction (the generalized eigenvectors are invariant to a global
    scale on ``K`` or ``M``); this keeps the float32 reduction well-conditioned
    when storage makes ``M`` tiny.  ``M`` is then regularized by
    ``jitter * I`` (in the scaled metric) for SPD safety.  The result is
    ``stop_gradient``-wrapped and :math:`M`-orthonormal, :math:`\psi^\top M \psi = I`: the
    vectors of the scaled pencil are divided by :math:`\sqrt{s_m}`.
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
    return jax.lax.stop_gradient(psi / jnp.sqrt(s_m))


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
    return null_space_components(A)[0]


def null_space_components(A: "sp.spmatrix") -> tuple[int, np.ndarray]:
    """``(c, labels)``: the connected components of the stiffness graph that
    :func:`count_null_modes` counts, with the component id of every node, so the null
    space itself -- the per-component constants -- can be written down."""
    A = sp.csr_matrix(A).copy()
    A.eliminate_zeros()
    n_comp, labels = connected_components(A, directed=False)
    return int(n_comp), np.asarray(labels)


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
    expanding-band sparse path. Returns float32 ``(vals (n - n_null,), vecs (n, n - n_null))``
    with the vectors :math:`M`-orthonormal (:func:`m_normalize`).
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
    return np.asarray(vals, dtype=np.float32), m_normalize(M, vecs).astype(np.float32)


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
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Sparse generalized eigenbasis of ``A v = lambda M v`` via SciPy ``eigsh``.

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
    the result holds exactly ``n_eig`` *genuinely oscillatory* modes. The null space is
    known exactly -- the constants of the mesh's ``c`` connected components
    (:func:`null_space_components`), ``c = 1`` for a connected mesh -- so it is *deflated*
    rather than solved for: the shift-invert operator is composed with the
    :math:`M`-orthogonal projector onto the complement of the constants,

    .. math::

        \mathbf x \;\mapsto\; P\,(A - \sigma M)^{-1} M\,\mathbf x,
        \qquad
        P \;=\; I - E\,\bigl(E^\top M E\bigr)^{-1} E^\top M,

    whose spectrum on that complement is :math:`(\lambda - \sigma)^{-1}` at the non-null
    eigenpairs, while the constants are mapped to zero. Asking Lanczos for the ``c`` zero
    eigenvalues instead, and dropping the lowest ``c`` pairs it returns, fails on a
    fragmented mesh: single-vector Lanczos resolves a ``c``-fold eigenvalue only
    approximately, misses some copies, and the count then drops genuine low modes in their
    place. A spectral-gap sanity check still verifies the constants are null to working
    precision: if ``lambda_retained[0] / max_c RQ(1_c) < null_gap`` (the Rayleigh quotients
    of the component constants) a ``RuntimeWarning`` reports a corrupted operator (e.g.
    degenerate elements breaking the constant null mode) or a weakly-coupled region.
    Passing ``n_null`` different from the component count names no null vectors; the solve
    then falls back to computing ``n_eig + n_null`` pairs and dropping the lowest
    ``n_null``.

    where:

    - :math:`E \in \{0,1\}^{n \times c}`: the component-indicator matrix, whose columns span the null space of :math:`A`; :math:`\sigma`: the shift just below zero.
    - :math:`P`: the :math:`M`-orthogonal projector onto the complement of the constants; it commutes with :math:`(A - \sigma M)^{-1} M` because the constants are eigenvectors of it, so the projection only removes the rounding that would otherwise grow along them.

    Both operators are rescaled to unit mean diagonal (the generalized eigenvectors are
    invariant to a global scale on ``A`` or ``M``) and ``M`` is regularized by
    ``jitter * I`` for SPD safety. Returns ``(eigvals (n_eig,), eigvecs (n, n_eig))`` as
    float32, ascending in eigenvalue, the vectors :math:`M`-orthonormal in the unscaled
    metric (:func:`m_normalize`). This is a one-time, gradient-free precompute (the
    encoding is fixed during training), so a host SciPy solve is appropriate.

    ``seed`` fixes the Lanczos starting vector, so a repeated solve of the same operators
    under the same BLAS threading returns the same eigenvectors -- signs and the rotation
    inside degenerate clusters included; without it ARPACK draws a fresh random start per
    call, and anything selected from the basis (the addressability floor among it) can
    change from one solve to the next. A different thread count reorders floating-point
    reductions, so a strongly degenerate cluster -- the many null modes of a fragmented
    mesh, which single-vector Lanczos resolves only approximately -- can still come out
    differently across machines; the spectral cache, not the seed, is what pins a basis.
    ``seed=None`` restores the random start.
    """
    A = sp.csr_matrix(A).astype(np.float64)
    M = sp.csr_matrix(M).astype(np.float64)
    n = A.shape[0]
    n_eig = int(n_eig)

    drop = bool(drop_null) and which == "SM"
    labels = None
    if drop:
        n_comp, labels = null_space_components(A)
        n_null = n_comp if n_null is None else int(n_null)
        if n_null != n_comp:
            labels = None                             # an overridden count names no vectors
    else:
        n_null = 0
    k = n_eig + n_null                                # the modes we keep + the nulls
    if k >= n:
        raise ValueError(
            f"n_eig+n_null={k} must be < n={n} for an iterative eigensolver."
        )

    s_a = max(float(A.diagonal().mean()), 1e-30)
    s_m = max(float(M.diagonal().mean()), 1e-30)
    A_s = (A / s_a).tocsc()
    M_s = (M / s_m + jitter * sp.identity(n, format="csr")).tocsc()
    v0 = None if seed is None else np.random.default_rng(seed).standard_normal(n)
    # shift just below the spectrum bottom so (A_s - sigma M_s) is SPD -> the returned
    # eigenvalues are those closest to 0, i.e. the lowest modes.
    sigma = -1e-5
    if labels is not None and n_null > 0:
        E = sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, n_null))
        ME = (M_s @ E).tocsc()
        gram = (E.T @ ME).toarray()
        gram_f = cho_factor(gram)

        def _project(y):
            return y - E @ cho_solve(gram_f, ME.T @ y)

        lu = splu((A_s - sigma * M_s).tocsc())
        op = LinearOperator(
            (n, n), matvec=lambda b: _project(lu.solve(np.asarray(b, np.float64).ravel())),
            dtype=np.float64)
        vals_s, vecs = eigsh(A_s, k=n_eig, M=M_s, sigma=sigma, OPinv=op, which="LM",
                             v0=None if v0 is None else _project(v0))
        order = np.argsort(vals_s)
        vals = vals_s[order] * (s_a / s_m)            # undo diagonal rescaling -> physical lambda
        vecs = vecs[:, order]
        # Spectral-gap sanity check: the constants must be null well below the first
        # retained mode, or the operator is corrupted (or a region barely coupled).
        null_rq = np.abs((E.T @ (A_s @ E)).diagonal() / np.diag(gram)) * (s_a / s_m)
        dropped_max = float(null_rq.max())
        first_keep = float(vals[0]) if vals.size else float("inf")
    else:
        if which == "SM":
            vals_s, vecs = eigsh(A_s, k=k, M=M_s, sigma=sigma, which="LM", v0=v0)
        elif which == "LM":
            vals_s, vecs = eigsh(A_s, k=k, M=M_s, which="LM", v0=v0)
        else:
            raise ValueError(f"which must be 'SM' or 'LM', got {which!r}")
        order = np.argsort(vals_s)
        vals = vals_s[order] * (s_a / s_m)            # undo diagonal rescaling -> physical lambda
        vecs = vecs[:, order]
        dropped_max = first_keep = None
        if drop and n_null > 0:
            # an overridden count: drop the lowest n_null pairs and check the gap they leave
            dropped_max = float(np.max(np.abs(vals[:n_null])))
            first_keep = float(vals[n_null]) if vals.size > n_null else float("inf")
            vals = vals[n_null:]
            vecs = vecs[:, n_null:]

    if dropped_max is not None:
        gap = first_keep / max(dropped_max, 1e-300)
        if gap < null_gap:
            warnings.warn(
                f"generalized_eig_basis_sparse: weak spectral gap ({gap:.2g}) between the "
                f"{n_null} null mode(s) (|lambda| <= {dropped_max:.2e}) and the first "
                f"retained mode ({first_keep:.2e}); the null space is poorly separated -- "
                f"suspect a corrupted operator (degenerate elements) or a weakly-coupled "
                f"region.",
                RuntimeWarning,
                stacklevel=2,
            )

    return np.asarray(vals, dtype=np.float32), m_normalize(M, vecs).astype(np.float32)
