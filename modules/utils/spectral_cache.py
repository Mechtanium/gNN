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

The cache has four tiers, mirroring the cost regimes of one run's preparation:

- **operator tier** :math:`\mathcal O=\{\texttt{STATIC},A,M,c\}` -- the
  :math:`N`-independent FEM operators (assembly + connected-component count), keyed by
  :math:`\kappa_{\mathrm{op}}=(\text{model},\ \text{weighting},\ \text{mesh fingerprint})`.
- **eigenbasis tier** :math:`\mathcal E_N=\{\lambda_{1:N},V_{1:N},V^c_{1:N},B^v_{1:N},
  \boldsymbol\mu_{1:N},\boldsymbol\sigma_{1:N}\}` -- the standardized modes and their
  centroid affine maps, keyed by :math:`\kappa_{\mathrm{op}}`, the resolved vertical
  reservation and the selection settings, at the width :math:`N` last written for that
  key (a wider request re-solves and replaces it; narrower ones are column slices).
- **pool tier** -- the candidate pool :math:`\{\lambda, V, \upsilon\}` that the
  addressability-floor search of :func:`pinnlab.spectral.resolve_n_eig` selects from
  (the deflated family, if any, then the widest lowest-:math:`\lambda` band solved so
  far), so a changed adequacy gate re-runs the cheap search, and a search that needs a
  wider band resumes from it, instead of repeating the eigensolve.
- **floor tier** -- the resolved addressability floor itself, so a notebook preview, the
  pipeline build that follows it, and every later kernel start agree on one width without
  re-solving anything.

Every entry lives in its own directory named by a digest of its key, under the
operator entry it derives from::

    spectral_cache/
        op-<digest>/                    operator tier
            eig-<digest>/               eigenbasis tier (one per selection key)
            pool-<digest>/              pool tier (one per candidate-search key)
            floor-<digest>.json         floor tier (one per resolution key)

so configurations that differ only in a weighting, a selection knob or a perturbed rock
field coexist instead of overwriting one another. Files written by the earlier
single-slot layout (directly inside ``spectral_cache/``) are still served when their key
matches and no keyed entry supersedes them, and are never overwritten. Writes go through a temporary file and an atomic
rename, metadata last, so an interrupted or concurrent writer never leaves a partial
entry that reads as valid.

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
- :math:`\upsilon`: the per-mode vertical variance fraction of the candidates.
- :math:`\kappa_{\mathrm{op}}`: the operator cache key -- the control-panel knobs and a
  content fingerprint of the mesh/rock arrays that uniquely determine :math:`\mathcal O`.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
import zipfile
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from modules.utils.portable_paths import portable_path

# v2: storage-weighted mass operator, vertical-aware retention (n_eig_z key,
# per-mode vertical variance fraction) and the geometric-mass mode images.
# v3: the retained set additionally depends on the SpectralSettings knobs that
# steer selection (candidate-spectrum strategy, per-component stratification and
# the pore-volume prune), so those enter the eigenbasis-tier key.
# v4: every solver path returns M-orthonormal modes with a fixed sign, the deflated
# family leads a band orthogonalized against it, and the vertical-column grouping is
# part of every selection key.
SPECTRAL_CACHE_VERSION = 4

# Version of the pool and floor tiers: bumped whenever the candidate solvers change what
# they return (v1: exact sparse deflation, band grown geometrically during the search;
# v2: the band deflates the per-component constants instead of dropping the lowest
# Lanczos pairs; v3: M-orthonormal pools and the nearest-neighbour separation test), so a
# pool or floor found by an earlier solver is recomputed.
RESOLUTION_CACHE_VERSION = 3

# SpectralSettings fields that change *which* modes are retained, and so make two
# cached bases non-interchangeable by column slicing. Tuning that only affects how
# the candidate band is searched (band_cap_mult, dense_eig_max_n) is deliberately
# excluded: it may change cost, never the selected set at a converged search.
EIG_SELECTION_KEYS = ("retention", "stratify_components", "min_comp_vol_frac",
                      "upsilon_vertical", "column_tol_ft", "column_grouping")


def eig_selection_key(spec: Any) -> dict[str, Any]:
    """The ``SpectralSettings`` subset that participates in the eigenbasis cache key,
    the vertical-column grouping among it (the vertical variance fractions, hence the
    retained set, are measured against the columns)."""
    return {f"spec_{k}": _json_ready(getattr(spec, k)) for k in EIG_SELECTION_KEYS}

# operator-tier files (N_EIG-independent)
_OP_META = "operators_meta.json"
_OP_STATIC = "operators_static.npz"
_OP_STIFFNESS = "operators_stiffness.npz"
_OP_MASS = "operators_mass.npz"

# eigenbasis-tier files (depend on N_EIG; column-sliceable)
_EIG_META = "eigenbasis_meta.json"
_EIG_DATA = "eigenbasis.npz"

# pool-tier files (the floor search's candidate pool)
_POOL_META = "pool_meta.json"
_POOL_DATA = "pool.npz"

# entries of the ``build_static_hex_fem`` dict that are python scalars, not arrays
_STATIC_SCALAR_KEYS = ("n_vertices", "n_hex", "n_degenerate")

# what a damaged or half-written entry raises on read; such an entry is treated as a miss
_READ_ERRORS = (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile)


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
    and ``n_hex`` are carried for a cheap human-readable shape check. The model path enters
    in its home-relative form, so the entry a key names is the same under any account.
    """
    return {
        "model_path": portable_path(Path(model_path).expanduser().resolve()),
        "laplacian_weighting": str(laplacian_weighting),
        "n_nodes": int(n_nodes),
        "n_hex": int(n_hex),
        "fem_fingerprint": str(fingerprint),
    }


def key_digest(key: dict[str, Any]) -> str:
    """12-hex SHA-1 of a key dict in canonical JSON form (the entry directory name)."""
    canonical = json.dumps(_json_ready(key), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def operator_entry_dir(cache_dir: str | Path, op_key: dict[str, Any]) -> Path:
    """Directory of the operator entry ``op_key``; every derived tier nests inside it."""
    return Path(cache_dir) / f"op-{key_digest(op_key)}"


def _eig_entry_dir(cache_dir: str | Path, op_key: dict[str, Any], n_eig_z: int,
                   sel_key: dict[str, Any] | None) -> Path:
    sub = {"n_eig_z": int(n_eig_z), **(sel_key or {})}
    return operator_entry_dir(cache_dir, op_key) / f"eig-{key_digest(sub)}"


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _atomic_write(path: Path, write) -> None:
    """Run ``write(fileobj)`` into a temporary sibling, then rename it onto ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "xb") as f:                   # plain open: the umask sets the mode
            write(f)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_meta(path: Path, meta: dict[str, Any]) -> None:
    text = json.dumps(_json_ready(meta), indent=2, sort_keys=True) + "\n"
    _atomic_write(path, lambda f: f.write(text.encode()))


def _write_npz(path: Path, **arrays) -> None:
    _atomic_write(path, lambda f: np.savez(f, **arrays))


def _read_meta(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _key_matches(meta: dict[str, Any] | None, key: dict[str, Any],
                 version: int = SPECTRAL_CACHE_VERSION) -> bool:
    """True iff ``meta`` carries the expected cache version and every key field agrees."""
    if meta is None:
        return False
    if int(meta.get("schema_version", -1)) != version:
        return False
    return all(meta.get(k) == v for k, v in _json_ready(key).items())


def _save_static_dict(f, static: dict[str, Any]) -> None:
    """Serialize the ``build_static_hex_fem`` dict (jnp arrays + scalars) to one npz."""
    np.savez(f, **{k: np.asarray(v) for k, v in static.items()})


def _load_static_dict(npz_path: Path) -> dict[str, Any]:
    """Reconstruct the static dict, restoring jnp arrays and python-int scalars."""
    out: dict[str, Any] = {}
    with np.load(npz_path, allow_pickle=False) as z:
        for k in z.files:
            arr = z[k]
            out[k] = int(arr) if k in _STATIC_SCALAR_KEYS else jnp.asarray(arr)
    return out


def _sparse_bytes(matrix) -> bytes:
    buf = io.BytesIO()
    sp.save_npz(buf, sp.csr_matrix(matrix))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# operator tier (N_EIG-independent: STATIC, A, M, c)
# --------------------------------------------------------------------------- #
def _operator_dir(cache_dir: str | Path, key: dict[str, Any]) -> Path | None:
    """The directory holding a complete operator entry for ``key``: the keyed entry, or
    the legacy single-slot files directly in ``cache_dir``; ``None`` if neither matches."""
    for d in (operator_entry_dir(cache_dir, key), Path(cache_dir)):
        if (_key_matches(_read_meta(d / _OP_META), key)
                and all((d / f).exists() for f in (_OP_STATIC, _OP_STIFFNESS, _OP_MASS))):
            return d
    return None


def operator_cache_valid(cache_dir: str | Path, key: dict[str, Any]) -> bool:
    """True iff a complete operator cache matching ``key`` is present on disk."""
    return _operator_dir(cache_dir, key) is not None


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
    d = operator_entry_dir(cache_dir, key)
    _atomic_write(d / _OP_STATIC, lambda f: _save_static_dict(f, static))
    stiff, mass = _sparse_bytes(A_lap), _sparse_bytes(M_lap)
    _atomic_write(d / _OP_STIFFNESS, lambda f: f.write(stiff))
    _atomic_write(d / _OP_MASS, lambda f: f.write(mass))
    meta = {**key, "schema_version": SPECTRAL_CACHE_VERSION, "n_null": int(n_null)}
    _write_meta(d / _OP_META, meta)


def load_operator_cache(
    cache_dir: str | Path, key: dict[str, Any]
) -> dict[str, Any] | None:
    r"""Load ``{static, A_lap, M_lap, n_null}`` if a matching cache exists, else ``None``."""
    d = _operator_dir(cache_dir, key)
    if d is None:
        return None
    try:
        meta = _read_meta(d / _OP_META)
        return {
            "static": _load_static_dict(d / _OP_STATIC),
            "A_lap": sp.load_npz(d / _OP_STIFFNESS),
            "M_lap": sp.load_npz(d / _OP_MASS),
            "n_null": int(meta["n_null"]),
        }
    except _READ_ERRORS:
        return None


# --------------------------------------------------------------------------- #
# eigenbasis tier (depends on N_EIG; column-sliceable)
# --------------------------------------------------------------------------- #
def _eig_meta_matches(meta: dict[str, Any] | None, op_key: dict[str, Any],
                      n_eig_z: int | None, sel_key: dict[str, Any] | None) -> bool:
    if not _key_matches(meta, op_key):
        return False
    if n_eig_z is not None and int(meta.get("n_eig_z", -1)) != int(n_eig_z):
        return False
    if sel_key is None:
        return True
    want = _json_ready(sel_key)
    return all(meta.get(k) == v for k, v in want.items())


def _eig_dir(cache_dir: str | Path, op_key: dict[str, Any], n_eig_z: int | None,
             sel_key: dict[str, Any] | None) -> tuple[Path, int] | None:
    """(directory, stored width) of the eigenbasis entry serving this key, if any.

    A keyed entry is authoritative for its key; the legacy single-slot files are read only
    when no keyed entry exists, so a basis written under the keyed layout is never
    shadowed by an older, wider one. Without ``n_eig_z`` every keyed entry of the
    operator is scanned and the widest match is reported.
    """
    if n_eig_z is not None:
        keyed = [_eig_entry_dir(cache_dir, op_key, n_eig_z, sel_key)]
    else:
        keyed = sorted(operator_entry_dir(cache_dir, op_key).glob("eig-*"))
    for dirs in (keyed, [Path(cache_dir)]):
        hits = []
        for d in dirs:
            meta = _read_meta(d / _EIG_META)
            if _eig_meta_matches(meta, op_key, n_eig_z, sel_key) and (d / _EIG_DATA).exists():
                hits.append((d, int(meta["n_eig"])))
        if hits:
            return max(hits, key=lambda h: h[1])
    return None


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
    hit = _eig_dir(cache_dir, op_key, n_eig_z, sel_key)
    return None if hit is None else hit[1]


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
    d = _eig_entry_dir(cache_dir, op_key, n_eig_z, sel_key)
    _write_npz(
        d / _EIG_DATA,
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
    _write_meta(d / _EIG_META, meta)


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
    hit = _eig_dir(cache_dir, op_key, n_eig_z, sel_key)
    if hit is None or hit[1] < int(n_eig):
        return None
    d, stored = hit
    n = int(n_eig)
    try:
        with np.load(d / _EIG_DATA, allow_pickle=False) as z:
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
    except _READ_ERRORS:
        return None


# --------------------------------------------------------------------------- #
# pool and floor tiers (the addressability-floor search of resolve_n_eig)
# --------------------------------------------------------------------------- #
def _pool_dir(cache_dir: str | Path, op_key: dict[str, Any], pool_key: dict[str, Any]) -> Path:
    return operator_entry_dir(cache_dir, op_key) / f"pool-{key_digest(pool_key)}"


def save_pool_cache(
    cache_dir: str | Path,
    op_key: dict[str, Any],
    pool_key: dict[str, Any],
    *,
    lam: np.ndarray,
    v: np.ndarray,
    ups: np.ndarray,
    n_forced: int,
    k_band: int,
) -> None:
    r"""Persist a floor-search candidate pool under ``(op_key, pool_key)``.

    The raw candidate modes are stored in float32 together with their eigenvalues and
    vertical variance fractions; the sidecar records how many leading columns are the
    unconditionally retained (deflated) family and the width of the
    lowest-:math:`\lambda` band behind them. A wider pool for the same key replaces a
    narrower one, so a later search resumes from the widest band solved so far.
    """
    d = _pool_dir(cache_dir, op_key, pool_key)
    _write_npz(d / _POOL_DATA, lam=np.asarray(lam), v=np.asarray(v, np.float32),
               ups=np.asarray(ups))
    meta = {**op_key, **pool_key, "schema_version": RESOLUTION_CACHE_VERSION,
            "n_forced": int(n_forced), "k_band": int(k_band)}
    _write_meta(d / _POOL_META, meta)


def load_pool_cache(
    cache_dir: str | Path, op_key: dict[str, Any], pool_key: dict[str, Any]
) -> dict[str, Any] | None:
    """The cached pool ``{lam, v, ups, n_forced, k_band}``, or ``None``."""
    d = _pool_dir(cache_dir, op_key, pool_key)
    meta = _read_meta(d / _POOL_META)
    if not _key_matches(meta, {**op_key, **pool_key}, RESOLUTION_CACHE_VERSION):
        return None
    try:
        with np.load(d / _POOL_DATA, allow_pickle=False) as z:
            return {"lam": np.asarray(z["lam"]), "v": np.asarray(z["v"]),
                    "ups": np.asarray(z["ups"]), "n_forced": int(meta["n_forced"]),
                    "k_band": int(meta["k_band"])}
    except _READ_ERRORS:
        return None


def _floor_path(cache_dir: str | Path, op_key: dict[str, Any],
                floor_key: dict[str, Any]) -> Path:
    return operator_entry_dir(cache_dir, op_key) / f"floor-{key_digest(floor_key)}.json"


def save_floor_cache(cache_dir: str | Path, op_key: dict[str, Any],
                     floor_key: dict[str, Any], floor: int | None) -> None:
    """Persist the resolved addressability floor (``None``: no width up to the cap works)."""
    meta = {**op_key, **floor_key, "schema_version": RESOLUTION_CACHE_VERSION,
            "floor": None if floor is None else int(floor)}
    _write_meta(_floor_path(cache_dir, op_key, floor_key), meta)


def load_floor_cache(cache_dir: str | Path, op_key: dict[str, Any],
                     floor_key: dict[str, Any]) -> dict[str, Any] | None:
    """``{"floor": int | None}`` when a floor was resolved under this key, else ``None``."""
    meta = _read_meta(_floor_path(cache_dir, op_key, floor_key))
    if not _key_matches(meta, {**op_key, **floor_key}, RESOLUTION_CACHE_VERSION):
        return None
    floor = meta.get("floor")
    return {"floor": None if floor is None else int(floor)}
