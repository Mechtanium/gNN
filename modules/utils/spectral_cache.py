r"""On-disk cache for the hexahedral-FEM spectral positional encoding.

The Laplace-eigenfunction encoding of the :math:`\Delta`-PINN is a *fixed,
gradient-free precompute* (see ``Laplace-DGM-PINN.ipynb`` §3): the trilinear-hex FEM
operators are assembled once and the leading generalized eigenpairs of

.. math::

    A\,\mathbf v_k = \lambda_k\,M\,\mathbf v_k,
    \qquad
    0=\lambda_0=\dots=\lambda_{c-1} < \lambda_c \le \lambda_{c+1} \le \dots,

are extracted by shift-invert Lanczos. The eigensolve dominates the wall-clock cost and
grows with the requested mode count :math:`N`, so re-running it on every kernel start --
merely to raise :math:`N` -- is wasteful. This module persists the result and serves it
back whenever the *control-panel knobs that produced it* are unchanged.

The cache is **two-tier**, mirroring the two cost regimes:

- **operator tier** :math:`\mathcal O=\{\texttt{STATIC},A,M,c\}` -- the
  :math:`N`-independent FEM operators (assembly + connected-component count), keyed by
  :math:`\kappa_{\mathrm{op}}=(\text{model},\ \text{weighting},\ \text{mesh fingerprint})`.
- **eigenbasis tier** :math:`\mathcal E_N=\{\lambda_{1:N},V_{1:N},V^c_{1:N},B^v_{1:N},
  \boldsymbol\mu_{1:N},\boldsymbol\sigma_{1:N}\}` -- the standardized modes and their
  centroid affine maps, keyed by :math:`(\kappa_{\mathrm{op}},N)`.

Because ``eigsh(which="SM")`` returns the modes in ascending :math:`\lambda` and the
node-wise standardization is applied *per column*, an eigenbasis cached at width
:math:`N'` serves any request :math:`N\le N'` **exactly** by column truncation,

.. math::

    V^{(N)} = V^{(N')}[:,\,1\!:\!N],
    \qquad
    V^c{}^{(N)} = V^c{}^{(N')}[:,\,1\!:\!N],
    \qquad N \le N'.

so the eigensolve only re-runs when :math:`N` is raised *above* the widest cached basis.

where:

- :math:`A,\ M`: trilinear-hex FEM stiffness (Neumann Laplace-Beltrami, optionally
  permeability weighted) and consistent mass, as ``scipy.sparse`` CSR operators.
- :math:`c=\dim\ker A`: number of connected mesh components (dropped null modes).
- :math:`\lambda_k,\ \mathbf v_k`: the :math:`k`-th generalized eigenvalue / eigenvector.
- :math:`V_{1:N}=(V-\boldsymbol\mu)/\boldsymbol\sigma`: the per-node, per-column
  standardized encoding; :math:`V^c,\ B^v`: its centroid value and physical gradient
  (the per-cell affine map used by the ``chain_rule`` residual).
- :math:`\kappa_{\mathrm{op}}`: the operator cache key -- the control-panel knobs and a
  content fingerprint of the mesh/rock arrays that uniquely determine :math:`\mathcal O`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

# v2: storage-weighted mass operator, vertical-aware retention (n_eig_z key,
# per-mode vertical variance fraction) and the geometric-mass mode images.
# v3: the retained set additionally depends on the SpectralSettings knobs that
# steer selection (candidate-spectrum strategy, per-component stratification and
# the pore-volume prune), so those enter the eigenbasis-tier key.
SPECTRAL_CACHE_VERSION = 3

# SpectralSettings fields that change *which* modes are retained, and so make two
# cached bases non-interchangeable by column slicing. Tuning that only affects how
# the candidate band is searched (band_cap_mult, dense_eig_max_n) is deliberately
# excluded: it may change cost, never the selected set at a converged search.
EIG_SELECTION_KEYS = ("retention", "stratify_components", "min_comp_vol_frac",
                      "upsilon_vertical", "column_tol_ft")


def eig_selection_key(spec: Any) -> dict[str, Any]:
    """The ``SpectralSettings`` subset that participates in the eigenbasis cache key."""
    return {f"spec_{k}": _json_ready(getattr(spec, k)) for k in EIG_SELECTION_KEYS}

# operator-tier files (N_EIG-independent)
_OP_META = "operators_meta.json"
_OP_STATIC = "operators_static.npz"
_OP_STIFFNESS = "operators_stiffness.npz"
_OP_MASS = "operators_mass.npz"

# eigenbasis-tier files (depend on N_EIG; column-sliceable)
_EIG_META = "eigenbasis_meta.json"
_EIG_DATA = "eigenbasis.npz"
_POOL_DATA = "candidate_pool.npz"
_POOL_META = "candidate_pool_meta.json"

# entries of the ``build_static_hex_fem`` dict that are python scalars, not arrays
_STATIC_SCALAR_KEYS = ("n_vertices", "n_hex", "n_degenerate")


# --------------------------------------------------------------------------- #
# paths, fingerprints and keys
# --------------------------------------------------------------------------- #
def default_spectral_cache_dir(prep_cache_dir: str | Path) -> Path:
    r"""Return the model's spectral-cache directory ``<model_out>/spectral_cache``.

    The preprocessing cache lives at ``<model_out>/prep_cache`` (see
    :mod:`ReservoirPrepCache`); the spectral cache is placed beside it under the same
    model output folder so all derived artifacts of one reservoir model stay colocated.
    """
    return Path(prep_cache_dir).parent / "spectral_cache"


def fem_input_fingerprint(
    verts: np.ndarray,
    hexes: np.ndarray,
    perms: np.ndarray,
    poro_cell: np.ndarray,
    corner_cells: np.ndarray | None = None,
) -> str:
    r"""Content hash of the FEM inputs that determine the operators :math:`A,M`.

    A short SHA-1 over the ``dtype``/``shape``/raw bytes of the node coordinates, hex
    connectivity, per-cell permeability, porosity and (optional) canonical corner
    coordinates. Any change to the preprocessing artifacts -- a re-meshed grid, edited
    rock arrays -- changes the fingerprint and so invalidates the operator cache, while
    a byte-identical mesh reuses it.
    """
    h = hashlib.sha1()
    arrays = [verts, hexes, perms, poro_cell]
    if corner_cells is not None:
        arrays.append(corner_cells)
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def operator_cache_key(
    *,
    model_path: str | Path,
    laplacian_weighting: str,
    n_nodes: int,
    n_hex: int,
    fingerprint: str,
) -> dict[str, Any]:
    r"""Assemble the operator-tier key :math:`\kappa_{\mathrm{op}}`.

    The operators :math:`\{\texttt{STATIC},A,M,c\}` depend on the reservoir model, the
    Laplacian stiffness weighting (``perm_weighted`` vs ``geometric``, which changes
    :math:`A`), and the mesh/rock content (via :func:`fem_input_fingerprint`); ``n_nodes``
    and ``n_hex`` are carried for a cheap human-readable shape check.
    """
    return {
        "model_path": str(Path(model_path).resolve()),
        "laplacian_weighting": str(laplacian_weighting),
        "n_nodes": int(n_nodes),
        "n_hex": int(n_hex),
        "fem_fingerprint": str(fingerprint),
    }


# --------------------------------------------------------------------------- #
# json / serialization helpers
# --------------------------------------------------------------------------- #
def _json_ready(value: Any) -> Any:
    """Coerce keys/values into JSON-serializable primitives (paths, numpy scalars)."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_meta(path: Path, meta: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(meta), indent=2, sort_keys=True) + "\n")


def _read_meta(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _key_matches(meta: dict[str, Any] | None, key: dict[str, Any]) -> bool:
    """True iff ``meta`` carries the current cache version and every key field agrees."""
    if meta is None:
        return False
    if int(meta.get("schema_version", -1)) != SPECTRAL_CACHE_VERSION:
        return False
    return all(meta.get(k) == v for k, v in key.items())


def _save_static_dict(npz_path: Path, static: dict[str, Any]) -> None:
    """Serialize the ``build_static_hex_fem`` dict (jnp arrays + scalars) to one npz."""
    np.savez(npz_path, **{k: np.asarray(v) for k, v in static.items()})


def _load_static_dict(npz_path: Path) -> dict[str, Any]:
    """Reconstruct the static dict, restoring jnp arrays and python-int scalars."""
    out: dict[str, Any] = {}
    with np.load(npz_path, allow_pickle=False) as z:
        for k in z.files:
            arr = z[k]
            out[k] = int(arr) if k in _STATIC_SCALAR_KEYS else jnp.asarray(arr)
    return out


# --------------------------------------------------------------------------- #
# operator tier (N_EIG-independent: STATIC, A, M, c)
# --------------------------------------------------------------------------- #
def operator_cache_valid(cache_dir: str | Path, key: dict[str, Any]) -> bool:
    """True iff a complete operator cache matching ``key`` is present on disk."""
    cache_dir = Path(cache_dir)
    meta = _read_meta(cache_dir / _OP_META)
    if not _key_matches(meta, key):
        return False
    return all(
        (cache_dir / f).exists()
        for f in (_OP_STATIC, _OP_STIFFNESS, _OP_MASS)
    )


def save_operator_cache(
    cache_dir: str | Path,
    key: dict[str, Any],
    static: dict[str, Any],
    A_lap: "sp.spmatrix",
    M_lap: "sp.spmatrix",
    n_null: int,
) -> None:
    r"""Persist the operator tier :math:`\{\texttt{STATIC},A,M,c\}` under ``key``.

    ``STATIC`` is stored as a single npz; the sparse :math:`A,M` use
    ``scipy.sparse.save_npz``; the connected-component count :math:`c=` ``n_null`` and
    the key fields live in the JSON sidecar. The metadata is written *last* so a partial
    write is never seen as valid by :func:`operator_cache_valid`.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _save_static_dict(cache_dir / _OP_STATIC, static)
    sp.save_npz(cache_dir / _OP_STIFFNESS, sp.csr_matrix(A_lap))
    sp.save_npz(cache_dir / _OP_MASS, sp.csr_matrix(M_lap))
    meta = {**key, "schema_version": SPECTRAL_CACHE_VERSION, "n_null": int(n_null)}
    _write_meta(cache_dir / _OP_META, meta)


def load_operator_cache(
    cache_dir: str | Path, key: dict[str, Any]
) -> dict[str, Any] | None:
    r"""Load ``{static, A_lap, M_lap, n_null}`` if a matching cache exists, else ``None``."""
    cache_dir = Path(cache_dir)
    if not operator_cache_valid(cache_dir, key):
        return None
    meta = _read_meta(cache_dir / _OP_META)
    return {
        "static": _load_static_dict(cache_dir / _OP_STATIC),
        "A_lap": sp.load_npz(cache_dir / _OP_STIFFNESS),
        "M_lap": sp.load_npz(cache_dir / _OP_MASS),
        "n_null": int(meta["n_null"]),
    }


# --------------------------------------------------------------------------- #
# eigenbasis tier (depends on N_EIG; column-sliceable)
# --------------------------------------------------------------------------- #
def cached_eig_n_eig(cache_dir: str | Path, op_key: dict[str, Any],
                     n_eig_z: int | None = None,
                     sel_key: dict[str, Any] | None = None) -> int | None:
    r"""Return the stored basis width :math:`N'` if its operator key matches, else ``None``.

    Lets a caller decide whether an in-memory (or to-be-computed) basis is *wider* than
    what is already on disk before overwriting it. When ``n_eig_z`` is given, the stored
    basis must also have been selected under the same **resolved** vertical reservation:
    the retained set is a function of :math:`(N, n_z)`, so bases with different vertical
    blocks are never interchangeable by column slicing. ``sel_key`` (see
    :func:`eig_selection_key`) extends the same argument to the ``SpectralSettings``
    knobs that steer selection.
    """
    cache_dir = Path(cache_dir)
    meta = _read_meta(cache_dir / _EIG_META)
    if not _key_matches(meta, op_key):
        return None
    if not (cache_dir / _EIG_DATA).exists():
        return None
    if n_eig_z is not None and int(meta.get("n_eig_z", -1)) != int(n_eig_z):
        return None
    if sel_key is not None and not all(meta.get(k) == v for k, v in sel_key.items()):
        return None
    return int(meta["n_eig"])


def save_eig_cache(
    cache_dir: str | Path,
    op_key: dict[str, Any],
    *,
    n_eig: int,
    n_eig_z: int,
    lam: np.ndarray,
    v_nodes: np.ndarray,
    v_c: np.ndarray,
    b_v: np.ndarray,
    v_mean: np.ndarray,
    v_std: np.ndarray,
    upsilon: np.ndarray,
    sel_key: dict[str, Any] | None = None,
) -> None:
    r"""Persist the eigenbasis tier :math:`\mathcal E_N` at width ``n_eig`` under ``op_key``.

    The eigenvalues, standardized nodal features, per-cell centroid value/gradient maps
    and per-mode vertical variance fractions :math:`\upsilon_m` are stored in one npz;
    the operator key plus the basis
    width :math:`N'=` ``n_eig`` and the resolved vertical reservation ``n_eig_z`` go in
    the JSON sidecar, written last (atomic-validity). A subsequent request for any
    :math:`N\le N'` at the same ``n_eig_z`` is served by column truncation (see module
    docstring): the vertical block leads the column order, so it is shared by every
    truncation, and the trailing areal columns keep the ascending-:math:`\lambda`
    prefix property.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_dir / _EIG_DATA,
        lam=np.asarray(lam),
        v_nodes=np.asarray(v_nodes),
        v_c=np.asarray(v_c),
        b_v=np.asarray(b_v),
        v_mean=np.asarray(v_mean),
        v_std=np.asarray(v_std),
        upsilon=np.asarray(upsilon),
    )
    meta = {**op_key, **(sel_key or {}), "schema_version": SPECTRAL_CACHE_VERSION,
            "n_eig": int(n_eig), "n_eig_z": int(n_eig_z)}
    _write_meta(cache_dir / _EIG_META, meta)


def load_eig_cache(
    cache_dir: str | Path, op_key: dict[str, Any], n_eig: int, n_eig_z: int,
    sel_key: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    r"""Serve a width-``n_eig`` eigenbasis from cache, slicing a wider basis if needed.

    Returns ``None`` unless a cache with matching ``op_key``, matching resolved
    ``n_eig_z``, and stored width :math:`N'\ge` ``n_eig`` exists. On a hit, every array
    is truncated to the first ``n_eig`` columns. The truncation is exact because the
    stored column order is (vertical block, areal modes ascending :math:`\lambda`): the
    vertical block is identical for every width at fixed ``n_eig_z``, and the areal tail
    keeps the ascending-:math:`\lambda` prefix property of ``eigsh(which="SM")``; the
    per-column standardization commutes with column truncation. The truncated basis need
    not equal an *independent* fresh width-``n_eig`` solve column-for-column, since
    eigenvectors of any degenerate :math:`\lambda`-cluster straddling the cut are defined
    only up to a rotation (both solves share the same spectrum and span). The returned
    dict also reports the on-disk width as ``n_eig_cached``.
    """
    cache_dir = Path(cache_dir)
    stored = cached_eig_n_eig(cache_dir, op_key, n_eig_z=n_eig_z, sel_key=sel_key)
    if stored is None or stored < int(n_eig):
        return None
    n = int(n_eig)
    with np.load(cache_dir / _EIG_DATA, allow_pickle=False) as z:
        return {
            "lam": jnp.asarray(z["lam"][:n]),
            "v_nodes": jnp.asarray(z["v_nodes"][:, :n]),
            "v_c": jnp.asarray(z["v_c"][:, :n]),
            "b_v": jnp.asarray(z["b_v"][:, :, :n]),
            "v_mean": jnp.asarray(z["v_mean"][:n]),
            "v_std": jnp.asarray(z["v_std"][:n]),
            "upsilon": jnp.asarray(z["upsilon"][:n]),
            "n_eig_cached": stored,
        }


def save_pool_cache(cache_dir: str | Path, op_key: dict[str, Any], *, ceiling: int,
                    n_eig_z: int, lam_c: np.ndarray, v_cand: np.ndarray, ups_c: np.ndarray,
                    n_forced: int, sel_key: dict[str, Any] | None = None) -> None:
    r"""Persist the candidate pool the addressability floor is measured against.

    The pool is a generalized eigensolve at ``ceiling`` width (``spec.n_eig_cap``);
    on a mesh with degenerate eigenvalues the returned eigenvectors are only
    defined up to a rotation inside each degenerate subspace, so two solves of
    the same problem give *different* bases and, downstream, different losses.
    Caching the pool makes every later run on the same deck reuse the identical
    basis (and skips the most expensive step of resolution).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_dir / _POOL_DATA, lam_c=np.asarray(lam_c), v_cand=np.asarray(v_cand),
             ups_c=np.asarray(ups_c))
    meta = {**op_key, **(sel_key or {}), "schema_version": SPECTRAL_CACHE_VERSION,
            "ceiling": int(ceiling), "n_eig_z": int(n_eig_z), "n_forced": int(n_forced)}
    (cache_dir / _POOL_META).write_text(json.dumps(meta, indent=2, sort_keys=True))


def load_pool_cache(cache_dir: str | Path, op_key: dict[str, Any], *, ceiling: int,
                    n_eig_z: int, sel_key: dict[str, Any] | None = None):
    """The cached candidate pool as ``(lam_c, v_cand, ups_c, n_forced)``, or ``None``."""
    cache_dir = Path(cache_dir)
    meta_path, data_path = cache_dir / _POOL_META, cache_dir / _POOL_DATA
    if not (meta_path.is_file() and data_path.is_file()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None
    want = {**op_key, **(sel_key or {}), "schema_version": SPECTRAL_CACHE_VERSION,
            "ceiling": int(ceiling), "n_eig_z": int(n_eig_z)}
    if any(meta.get(k) != v for k, v in want.items()):
        return None
    with np.load(data_path) as z:
        return z["lam_c"], z["v_cand"], z["ups_c"], int(meta["n_forced"])
