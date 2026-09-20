from __future__ import annotations

import importlib
import math
import re
import shlex
import sys
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _import_deepfield():
    """The vendored deck-table parser (``field/``, formerly DeepField)."""
    return importlib.import_module("field")


_DEEPFIELD = _import_deepfield()
_DEEPFIELD_IMPORT_ERROR = None

if True:
    try:
        from field.field.grids import CornerPointGrid, OrthogonalGrid
    except ImportError:  # pragma: no cover - compatibility with older DeepField variants
        try:
            from field.field.grids import CornerPointGrid, OrthogonalUniformGrid as OrthogonalGrid
        except ImportError:  # pragma: no cover - resolved by notebook env
            CornerPointGrid = None
            OrthogonalGrid = None
    from field.field.validation import validate_loaded_model
    from field.field.tables.table_interpolation import split_pvto
else:  # pragma: no cover - exercised only when DeepField is unavailable
    CornerPointGrid = None
    OrthogonalGrid = None
    split_pvto = None

    def validate_loaded_model(field, components=()):
        return None


DEEPFIELD_TO_CANONICAL = np.array([0, 1, 3, 2, 4, 5, 7, 6], dtype=int)
I_EDGES = np.array([[0, 1], [3, 2], [4, 5], [7, 6]], dtype=int)
J_EDGES = np.array([[0, 3], [1, 2], [4, 7], [5, 6]], dtype=int)
K_EDGES = np.array([[0, 4], [1, 5], [3, 7], [2, 6]], dtype=int)
HEX_FACE_CORNERS = np.array(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [3, 2, 6, 7],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ],
    dtype=int,
)
HEX_FACE_NEIGHBOR_OFFSETS = np.array(
    [
        [0, 0, -1],
        [0, 0, 1],
        [0, -1, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [1, 0, 0],
    ],
    dtype=int,
)
# Default scaled-Jacobian floor for hexahedral element validity. Kept identical to
# ``hex_fem_assembly_jax.build_static_hex_fem``'s ``jac_rel_tol`` (the single source of
# truth for downstream FEM assembly) so the mesh builder and the FEM assembler agree on
# which cells are geometrically valid.
HEX_JAC_REL_TOL = 1e-6


@dataclass
class ReservoirMeshData:
    r"""Hexahedral reservoir mesh in native simulator coordinates.

    Each active cell is a trilinear 8-node hexahedron whose corners are stored in
    canonical VTK order (see ``DEEPFIELD_TO_CANONICAL`` / ``corner_cells``).  Cell
    geometry validity is determined by a scaled-Jacobian test on the trilinear shape
    functions (the same criterion used by
    :func:`hex_fem_assembly_jax.build_static_hex_fem`), so each cell is assembled
    directly as a hexahedron without any simplex decomposition.

    Coordinates are preserved as delivered by the deck/restart source.  In typical
    subsurface models the :math:`z` coordinate stores depth increasing downward, so
    visualization code should reverse the displayed z-axis instead of negating the
    underlying geometry.

    where:

     - ``verts``: ``(n_vertices, 3)`` float array of deduplicated node coordinates.
     - ``cell_to_unique_vertices``: length-``n_cells`` list of int arrays; each entry
       holds the (pruned, remapped) unique vertex ids of one active cell.  Empty for a
       cell whose own geometry is degenerate before borrowing a neighbour's geometry.
     - ``vertex_to_cells``: length-``n_vertices`` list of int arrays; the active cells
       incident to each vertex.
     - ``boundary_faces``: ``(n_boundary_quads, 4)`` int array of hexahedral boundary
       quad faces (remapped vertex ids), oriented per ``HEX_FACE_CORNERS``.
     - ``boundary_vertices``: 1-D int array of unique vertices lying on the boundary.
     - ``active_cell_indices``: ``(n_cells, 3)`` int array of ``(i, j, k)`` grid indices.
     - ``active_mask``: ``(nx, ny, nz)`` boolean grid activity mask.
     - ``corner_cells``: ``(n_cells, 8, 3)`` float array of canonical VTK-ordered corners.
     - ``cell_frames``: ``(n_cells, 3, 3)`` float array of local I/J/K orthonormal frames.
     - ``cell_lengths``: ``(n_cells, 3)`` float array of mean edge lengths along I/J/K.
     - ``cell_centroids``: ``(n_cells, 3)`` float array of cell centroids.
     - ``cell_volumes``: ``(n_cells,)`` float array of reference (deck) cell volumes,
       or ``None`` when unavailable.
     - ``cell_volume_estimate``: ``(n_cells,)`` float array of hexahedral cell volumes
       computed by Gauss quadrature :math:`\sum_{gp} |\det J|\, w`; zero for degenerate
       cells.
     - ``volume_diagnostics``: dict of mesh-quality and volume-conservation summaries.
     - ``active_cell_lookup``: dict mapping ``(i, j, k)`` to the active cell index.
     - ``cell_geometry_owner``: ``(n_cells,)`` int array; the cell whose geometry each
       cell uses (itself when valid; a neighbour when borrowed).
     - ``cell_merge_hops``: ``(n_cells,)`` int array; BFS hop count to the geometry owner
       (``0`` for valid cells, ``-1`` for centroid-fallback merges).
    """

    verts: np.ndarray
    cell_to_unique_vertices: list[np.ndarray]
    vertex_to_cells: list[np.ndarray]
    boundary_faces: np.ndarray
    boundary_vertices: np.ndarray
    active_cell_indices: np.ndarray
    active_mask: np.ndarray
    corner_cells: np.ndarray
    cell_frames: np.ndarray
    cell_lengths: np.ndarray
    cell_centroids: np.ndarray
    cell_volumes: np.ndarray | None
    cell_volume_estimate: np.ndarray
    volume_diagnostics: dict[str, Any]
    active_cell_lookup: dict[tuple[int, int, int], int]
    cell_geometry_owner: np.ndarray
    cell_merge_hops: np.ndarray


def _require_deepfield() -> None:
    if _DEEPFIELD is None:
        message = "DeepField is required for this operation."
        if _DEEPFIELD_IMPORT_ERROR is not None:
            raise ImportError(message) from _DEEPFIELD_IMPORT_ERROR
        raise ImportError(message)


def _normalize(vec: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= tol:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def _normalize_rows(vectors: np.ndarray, tol: float = 1e-12, label: str = "vectors") -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= tol):
        count = int(np.sum(norms <= tol))
        raise ValueError(f"Cannot normalize near-zero {label}: {count} rows below tolerance {tol}.")
    return vectors / norms[:, None]


def _field_has_component(field, name: str) -> bool:
    if name in tuple(getattr(field, "components", ())):
        return True
    try:
        getattr(field, name)
    except (AttributeError, KeyError, AssertionError):
        return False
    return True


def _vertex_deduplication(
    corner_cells: np.ndarray,
    dedup_decimals: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Merge rounded-equal vertices unless that would collapse corners within one cell."""
    corner_cells = np.asarray(corner_cells, dtype=float)
    n_cells, n_corners, _ = corner_cells.shape
    flat_corners = corner_cells.reshape(-1, 3)
    rounded_corners = np.round(flat_corners, decimals=dedup_decimals)

    _, first_indices, inverse = np.unique(
        rounded_corners,
        axis=0,
        return_index=True,
        return_inverse=True,
    )

    # Reorder unique ids by first appearance instead of lexicographic key order.
    appearance_order = np.argsort(first_indices)
    old_to_new = np.empty_like(appearance_order)
    old_to_new[appearance_order] = np.arange(len(appearance_order), dtype=int)
    inverse = old_to_new[inverse]
    first_indices = first_indices[appearance_order]

    verts = flat_corners[first_indices].copy()
    corner_to_vertex = inverse.reshape(n_cells, n_corners)

    merges_blocked_intra_cell = 0
    cells_with_blocked_intra_cell_merges = 0

    # Repair only the cells where global rounded-vertex deduplication collapsed
    # two or more corners within the same hexahedral cell.
    sorted_ids = np.sort(corner_to_vertex, axis=1)
    problematic_cells = np.where(np.any(np.diff(sorted_ids, axis=1) == 0, axis=1))[0]
    if len(problematic_cells):
        verts_list = [np.asarray(v, dtype=float) for v in verts]
        for cell_idx in problematic_cells:
            used_in_cell: set[int] = set()
            blocked_in_cell = False
            for local_corner_idx in range(n_corners):
                vertex_id = int(corner_to_vertex[cell_idx, local_corner_idx])
                if vertex_id in used_in_cell:
                    blocked_in_cell = True
                    merges_blocked_intra_cell += 1
                    vertex_id = len(verts_list)
                    verts_list.append(corner_cells[cell_idx, local_corner_idx].astype(float, copy=True))
                    corner_to_vertex[cell_idx, local_corner_idx] = vertex_id
                used_in_cell.add(int(vertex_id))
            if blocked_in_cell:
                cells_with_blocked_intra_cell_merges += 1
        verts = np.asarray(verts_list, dtype=float)

    merges_reused = int(flat_corners.shape[0] - verts.shape[0])
    diagnostics = {
        "dedup_vertices_created": int(verts.shape[0]),
        "dedup_vertices_reused": int(merges_reused),
        "dedup_intra_cell_merges_blocked": int(merges_blocked_intra_cell),
        "cells_with_dedup_intra_cell_merges_blocked": int(cells_with_blocked_intra_cell_merges),
    }
    return verts, corner_to_vertex, diagnostics


def _hex_scaled_jacobian_validity(
    corner_cells: np.ndarray,
    jac_rel_tol: float = HEX_JAC_REL_TOL,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    r"""Hexahedral element validity and volume via a scaled-Jacobian test.

    For each canonical VTK-ordered 8-node hexahedron this evaluates the trilinear
    shape-function gradients at the eight :math:`2\times2\times2` Gauss points and
    forms the Jacobian

    .. math::

        J_{ij}(\xi_{gp}) = \sum_{a=1}^{8} x_{a,i}\,
        \frac{\partial N_a}{\partial \hat{x}_j}(\xi_{gp}),

    and accepts the element only when *every* Gauss point is finite and has a
    strictly positive, non-collapsed scaled Jacobian:

    .. math::

        \det J(\xi_{gp}) > \texttt{jac\_rel\_tol}\; V_{\mathrm{bbox}},
        \qquad
        V_{\mathrm{bbox}} = \prod_{d} \bigl(\max_a x_{a,d} - \min_a x_{a,d}\bigr).

    A single bad Gauss point condemns the whole element (it is marked invalid and
    its volume estimate zeroed), matching
    :func:`hex_fem_assembly_jax.build_static_hex_fem` -- the single source of truth
    for the downstream hexahedral FEM assembly.  The hexahedral cell volume is the
    Gauss-quadrature sum :math:`\sum_{gp} |\det J(\xi_{gp})|\, w_{gp}`.

    where:

     - ``corner_cells``: :math:`(n_{cells}, 8, 3)` canonical VTK-ordered hexahedron
       corner coordinates.
     - ``jac_rel_tol``: scaled-Jacobian floor relative to the bounding-box volume.
     - returns ``(elem_ok, cell_volume, diagnostics)`` where ``elem_ok`` is the
       :math:`(n_{cells},)` boolean validity mask, ``cell_volume`` the
       :math:`(n_{cells},)` quadrature volume (zeroed on invalid cells), and
       ``diagnostics`` a dict of integer/float quality summaries.
    """
    from modules.utils.hex_fem_assembly_jax import _gauss_2x2x2, _ref_shape

    corner_cells = np.asarray(corner_cells, dtype=float)
    n_cells = corner_cells.shape[0]
    if n_cells == 0:
        empty = np.zeros((0,), dtype=float)
        diagnostics = {
            "n_degenerate_cells": 0,
            "cell_nonfinite_jacobian": np.zeros((0,), dtype=int),
            "cell_collapsed_jacobian": np.zeros((0,), dtype=int),
        }
        return np.zeros((0,), dtype=bool), empty, diagnostics

    gp, gw = _gauss_2x2x2()                       # (n_gp, 3), (n_gp,)
    _, dNref_gp = _ref_shape(gp)                  # (n_gp, 8, 3)

    # Per-element bounding-box volume makes the degeneracy test scale-relative.
    bbox = corner_cells.max(axis=1) - corner_cells.min(axis=1)      # (n_cells, 3)
    vbox = np.prod(bbox, axis=1)                                    # (n_cells,)

    # Jacobian at each Gauss point: J[e,gp,i,j] = sum_a x[e,a,i] dNref[gp,a,j].
    jac = np.einsum("eai,gaj->egij", corner_cells, dNref_gp)        # (n_cells, n_gp, 3, 3)
    det = np.linalg.det(jac)                                        # (n_cells, n_gp)
    finite = np.all(np.isfinite(jac), axis=(-1, -2))               # (n_cells, n_gp)
    positive = det > float(jac_rel_tol) * vbox[:, None]            # (n_cells, n_gp)
    gp_ok = finite & positive                                      # (n_cells, n_gp)
    elem_ok = np.all(gp_ok, axis=1)                               # (n_cells,)

    cell_volume = np.sum(np.abs(det) * gw[None, :], axis=1)        # (n_cells,)
    cell_volume = np.where(elem_ok, cell_volume, 0.0)

    diagnostics = {
        "n_degenerate_cells": int((~elem_ok).sum()),
        "cell_nonfinite_jacobian": (~finite).sum(axis=1).astype(int),
        "cell_collapsed_jacobian": (finite & ~positive).sum(axis=1).astype(int),
    }
    return elem_ok, cell_volume, diagnostics


def _hexa_boundary_face_records(
    grid_shape: tuple[int, int, int],
    active_mask: np.ndarray,
    active_cell_indices: np.ndarray,
    use_only_active: bool,
) -> np.ndarray:
    grid_shape = np.asarray(grid_shape, dtype=int)
    active_mask = np.asarray(active_mask, dtype=bool)
    active_cell_indices = np.asarray(active_cell_indices, dtype=int)

    boundary_records = []
    for active_cell_id, cell_ijk in enumerate(active_cell_indices):
        for face_id, neighbor_offset in enumerate(HEX_FACE_NEIGHBOR_OFFSETS):
            neighbor = cell_ijk + neighbor_offset
            if np.any(neighbor < 0) or np.any(neighbor >= grid_shape):
                boundary_records.append((active_cell_id, face_id))
                continue
            if use_only_active and not bool(active_mask[tuple(int(v) for v in neighbor.tolist())]):
                boundary_records.append((active_cell_id, face_id))

    if not boundary_records:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(boundary_records, dtype=int)


def build_cell_local_frames(corner_cells: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build stable local I/J/K frames and lengths from canonical hexahedron corners."""
    corner_cells = np.asarray(corner_cells, dtype=float)
    n_cells = corner_cells.shape[0]
    if n_cells == 0:
        return (
            np.zeros((0, 3, 3), dtype=float),
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 3), dtype=float),
        )

    centroids = corner_cells.mean(axis=1)
    i_edges = corner_cells[:, I_EDGES[:, 1], :] - corner_cells[:, I_EDGES[:, 0], :]
    j_edges = corner_cells[:, J_EDGES[:, 1], :] - corner_cells[:, J_EDGES[:, 0], :]
    k_edges = corner_cells[:, K_EDGES[:, 1], :] - corner_cells[:, K_EDGES[:, 0], :]

    vi = i_edges.mean(axis=1)
    vj = j_edges.mean(axis=1)
    vk = k_edges.mean(axis=1)

    lengths = np.stack(
        (
            np.linalg.norm(i_edges, axis=2).mean(axis=1),
            np.linalg.norm(j_edges, axis=2).mean(axis=1),
            np.linalg.norm(k_edges, axis=2).mean(axis=1),
        ),
        axis=1,
    )

    ei = _normalize_rows(vi, tol=tol, label="cell i-directions")
    ej = vj - np.einsum("ij,ij->i", vj, ei)[:, None] * ei
    ej_bad = np.linalg.norm(ej, axis=1) <= tol
    if np.any(ej_bad):
        ej[ej_bad] = np.cross(vk[ej_bad], ei[ej_bad])
    ej = _normalize_rows(ej, tol=tol, label="cell j-directions")

    ek = np.cross(ei, ej)
    ek_bad = np.linalg.norm(ek, axis=1) <= tol
    if np.any(ek_bad):
        ek[ek_bad] = vk[ek_bad]
    ek = _normalize_rows(ek, tol=tol, label="cell k-directions")

    flip_mask = np.einsum("ij,ij->i", ek, vk) < 0.0
    if np.any(flip_mask):
        ek[flip_mask] *= -1.0

    ej = _normalize_rows(np.cross(ek, ei), tol=tol, label="re-orthogonalized cell j-directions")
    frames = np.stack((ei, ej, ek), axis=2)
    return frames, lengths, centroids


def _assign_cells_to_neighbor_geometry(
    active_cell_indices: np.ndarray,
    active_cell_lookup: dict[tuple[int, int, int], int],
    has_geometry: np.ndarray,
    cell_centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    n_cells = active_cell_indices.shape[0]
    owner = np.full(n_cells, -1, dtype=int)
    hops = np.full(n_cells, -1, dtype=int)
    queue: deque[int] = deque()

    valid_cells = np.where(has_geometry)[0]
    if len(valid_cells) == 0:
        raise ValueError("No geometrically valid cells remain to absorb invalid cells.")

    for cell_idx in valid_cells:
        owner[cell_idx] = cell_idx
        hops[cell_idx] = 0
        queue.append(int(cell_idx))

    while queue:
        cell_idx = queue.popleft()
        ijk = active_cell_indices[cell_idx]
        for axis in range(3):
            for delta in (-1, 1):
                neighbor_ijk = ijk.copy()
                neighbor_ijk[axis] += delta
                neighbor_idx = active_cell_lookup.get(tuple(int(v) for v in neighbor_ijk.tolist()))
                if neighbor_idx is None or owner[neighbor_idx] != -1:
                    continue
                owner[neighbor_idx] = owner[cell_idx]
                hops[neighbor_idx] = hops[cell_idx] + 1
                queue.append(int(neighbor_idx))

    centroid_fallback_merges = 0
    unresolved = np.where(owner < 0)[0]
    if len(unresolved):
        valid_centroids = cell_centroids[valid_cells]
        for cell_idx in unresolved:
            distances = np.linalg.norm(valid_centroids - cell_centroids[cell_idx][None, :], axis=1)
            nearest_owner = valid_cells[int(np.argmin(distances))]
            owner[cell_idx] = int(nearest_owner)
            hops[cell_idx] = -1
            centroid_fallback_merges += 1

    diagnostics = {
        "cells_without_geometry_before_merge": int((~has_geometry).sum()),
        "cells_merged_into_neighbor_geometry": int(np.sum((~has_geometry) & (owner >= 0))),
        "cells_without_geometry_after_merge": int(np.sum(owner < 0)),
        "cells_merged_by_centroid_fallback": int(centroid_fallback_merges),
        "merge_max_hops": int(hops[hops >= 0].max()) if np.any(hops >= 0) else 0,
    }
    return owner, hops, diagnostics


def corner_cells_to_reservoir_mesh(
    corner_cells,
    active_mask,
    active_cell_indices: np.ndarray | None = None,
    cell_volumes: np.ndarray | None = None,
    use_only_active: bool = True,
    dedup_decimals: int = 8,
    volume_tol: float = 1e-12,
    jac_rel_tol: float = HEX_JAC_REL_TOL,
) -> ReservoirMeshData:
    r"""Convert canonical 8-corner active cells into a hexahedral reservoir mesh.

    Each active cell is kept as a trilinear 8-node hexahedron.  Geometric validity is
    decided by a scaled-Jacobian test (:func:`_hex_scaled_jacobian_validity`) that
    mirrors :func:`hex_fem_assembly_jax.build_static_hex_fem`: a cell is valid when
    every :math:`2\times2\times2` Gauss point has a finite, strictly positive,
    non-collapsed Jacobian (:math:`\det J > \texttt{jac\_rel\_tol}\,V_{\mathrm{bbox}}`).
    Cells failing the test contribute no geometry of their own and borrow a neighbour's
    geometry via :func:`_assign_cells_to_neighbor_geometry`.

    where:

     - ``corner_cells``: :math:`(n_{cells}, 8, 3)` canonical VTK-ordered corners.
     - ``active_mask``: :math:`(nx, ny, nz)` boolean grid activity mask.
     - ``active_cell_indices``: :math:`(n_{cells}, 3)` ``(i, j, k)`` indices; derived
       from ``active_mask`` when ``None``.
     - ``cell_volumes``: optional :math:`(n_{cells},)` reference (deck) cell volumes used
       only for the ``cell_volume_rel_error_*`` diagnostics.
     - ``dedup_decimals``: rounding precision for vertex deduplication.
     - ``volume_tol``: floor used in the relative-volume-error diagnostic denominator.
     - ``jac_rel_tol``: scaled-Jacobian floor for hexahedral cell validity.
    """
    active_mask = np.asarray(active_mask, dtype=bool)
    if active_mask.ndim != 3:
        raise ValueError(f"active_mask must be a 3D boolean array; got shape {active_mask.shape}.")
    if active_cell_indices is None:
        active_cell_indices = np.argwhere(active_mask)
    else:
        active_cell_indices = np.asarray(active_cell_indices, dtype=int)
        if active_cell_indices.ndim != 2 or active_cell_indices.shape[1] != 3:
            raise ValueError(
                f"active_cell_indices must have shape (n_active, 3); got {active_cell_indices.shape}."
            )
    corner_cells = np.asarray(corner_cells, dtype=float)
    if corner_cells.ndim != 3 or corner_cells.shape[1:] != (8, 3):
        raise ValueError(
            f"corner_cells must have shape (n_active, 8, 3) in canonical corner order; got {corner_cells.shape}."
        )
    if corner_cells.shape[0] != active_cell_indices.shape[0]:
        raise ValueError(
            "corner_cells and active_cell_indices must contain the same number of active cells. "
            f"Got {corner_cells.shape[0]} and {active_cell_indices.shape[0]}."
        )
    if cell_volumes is not None:
        cell_volumes = np.asarray(cell_volumes, dtype=float)
        if cell_volumes.shape[0] != corner_cells.shape[0]:
            raise ValueError(
                "cell_volumes must align with the active corner cells. "
                f"Got {cell_volumes.shape[0]} values for {corner_cells.shape[0]} cells."
            )

    n_cells = corner_cells.shape[0]
    grid_shape = active_mask.shape
    active_cell_lookup = {tuple(idx.tolist()): i for i, idx in enumerate(active_cell_indices)}
    boundary_face_records = _hexa_boundary_face_records(
        grid_shape=grid_shape,
        active_mask=active_mask,
        active_cell_indices=active_cell_indices,
        use_only_active=use_only_active,
    )
    boundary_face_ids_by_cell = [[] for _ in range(len(active_cell_indices))]
    for active_cell_id, face_id in boundary_face_records:
        boundary_face_ids_by_cell[int(active_cell_id)].append(int(face_id))
    frames, lengths, centroids = build_cell_local_frames(corner_cells)
    verts, corner_to_vertex, dedup_diagnostics = _vertex_deduplication(corner_cells, dedup_decimals)

    # Hexahedral scaled-Jacobian validity and per-cell quadrature volume. A cell is
    # valid iff every Gauss point has a finite, strictly positive, non-collapsed
    # Jacobian; invalid cells get a zeroed volume and an empty connectivity below.
    elem_ok, cell_volume_estimate, jac_diagnostics = _hex_scaled_jacobian_validity(
        corner_cells, jac_rel_tol=jac_rel_tol
    )
    has_geometry = elem_ok.copy()

    # Hex boundary quad faces: for each boundary face of each valid cell, gather its
    # four canonical corner vertex ids. Invalid cells contribute no boundary geometry.
    boundary_quad_list = []
    for cell_idx in range(n_cells):
        if not has_geometry[cell_idx]:
            continue
        for face_id in boundary_face_ids_by_cell[cell_idx]:
            face_corners = HEX_FACE_CORNERS[face_id]
            boundary_quad_list.append(corner_to_vertex[cell_idx, face_corners].copy())
    boundary_faces = (
        np.vstack(boundary_quad_list).astype(int) if boundary_quad_list else np.zeros((0, 4), dtype=int)
    )

    # Vertex pruning: keep only vertices referenced by a geometrically valid cell.
    original_n_vertices = verts.shape[0]
    if np.any(has_geometry):
        used_vertices = np.unique(corner_to_vertex[has_geometry].reshape(-1))
        vertex_remap = -np.ones(original_n_vertices, dtype=int)
        vertex_remap[used_vertices] = np.arange(len(used_vertices), dtype=int)
        verts = verts[used_vertices]
    else:
        used_vertices = np.zeros((0,), dtype=int)
        vertex_remap = -np.ones(original_n_vertices, dtype=int)

    if len(boundary_faces):
        boundary_faces = vertex_remap[boundary_faces]
        boundary_faces = boundary_faces[np.all(boundary_faces >= 0, axis=1)]
        if len(boundary_faces):
            boundary_faces = np.unique(np.sort(boundary_faces, axis=1), axis=0)
        else:
            boundary_faces = np.zeros((0, 4), dtype=int)
    boundary_vertices = np.unique(boundary_faces) if len(boundary_faces) else np.zeros((0,), dtype=int)

    cell_to_unique_vertices = []
    for i in range(n_cells):
        if not has_geometry[i]:
            cell_to_unique_vertices.append(np.zeros((0,), dtype=int))
            continue
        vertex_ids = np.unique(corner_to_vertex[i])
        keep_vertices = vertex_ids[vertex_remap[vertex_ids] >= 0]
        if len(keep_vertices):
            cell_to_unique_vertices.append(vertex_remap[keep_vertices].astype(int))
        else:
            cell_to_unique_vertices.append(np.zeros((0,), dtype=int))

    cell_geometry_owner, cell_merge_hops, merge_diagnostics = _assign_cells_to_neighbor_geometry(
        active_cell_indices,
        active_cell_lookup,
        has_geometry,
        centroids,
    )
    for cell_idx in np.where(~has_geometry)[0]:
        owner_idx = int(cell_geometry_owner[cell_idx])
        if owner_idx < 0 or owner_idx == cell_idx or len(cell_to_unique_vertices[owner_idx]) == 0:
            continue
        cell_to_unique_vertices[cell_idx] = np.asarray(cell_to_unique_vertices[owner_idx], dtype=int).copy()
        frames[cell_idx] = frames[owner_idx]
        lengths[cell_idx] = lengths[owner_idx]
        centroids[cell_idx] = centroids[owner_idx]
    vertex_to_cells = [[] for _ in range(verts.shape[0])]
    for cell_idx, vertex_ids in enumerate(cell_to_unique_vertices):
        for vertex_id in vertex_ids:
            vertex_to_cells[int(vertex_id)].append(cell_idx)
    vertex_to_cells = [np.asarray(cells, dtype=int) for cells in vertex_to_cells]
    volume_diagnostics = {
        "dedup_vertices_created": int(dedup_diagnostics["dedup_vertices_created"]),
        "dedup_vertices_reused": int(dedup_diagnostics["dedup_vertices_reused"]),
        "dedup_intra_cell_merges_blocked": int(dedup_diagnostics["dedup_intra_cell_merges_blocked"]),
        "cells_with_dedup_intra_cell_merges_blocked": int(dedup_diagnostics["cells_with_dedup_intra_cell_merges_blocked"]),
        "n_degenerate_cells": int(jac_diagnostics["n_degenerate_cells"]),
        "cells_with_nonfinite_jacobian": int((jac_diagnostics["cell_nonfinite_jacobian"] > 0).sum()),
        "cells_with_collapsed_jacobian": int((jac_diagnostics["cell_collapsed_jacobian"] > 0).sum()),
        "cells_without_geometry_before_merge": int(merge_diagnostics["cells_without_geometry_before_merge"]),
        "cells_merged_into_neighbor_geometry": int(merge_diagnostics["cells_merged_into_neighbor_geometry"]),
        "cells_without_geometry_after_merge": int(merge_diagnostics["cells_without_geometry_after_merge"]),
        "cells_merged_by_centroid_fallback": int(merge_diagnostics["cells_merged_by_centroid_fallback"]),
        "merge_max_hops": int(merge_diagnostics["merge_max_hops"]),
        "hexa_boundary_quads": int(len(boundary_face_records)),
        "boundary_quads": int(len(boundary_faces)),
        "boundary_vertices": int(len(boundary_vertices)),
        "unused_vertices_pruned": int(original_n_vertices - verts.shape[0]),
        "cells_without_geometry": int(sum(len(v) == 0 for v in cell_to_unique_vertices)),
        "cell_volume_min": float(cell_volume_estimate.min()) if len(cell_volume_estimate) else 0.0,
        "cell_volume_max": float(cell_volume_estimate.max()) if len(cell_volume_estimate) else 0.0,
        "cell_volume_mean": float(cell_volume_estimate.mean()) if len(cell_volume_estimate) else 0.0,
    }
    if cell_volumes is not None and len(cell_volumes):
        rel_error = np.abs(cell_volume_estimate - cell_volumes) / np.maximum(np.abs(cell_volumes), volume_tol)
        volume_diagnostics.update(
            {
                "cell_volume_rel_error_mean": float(rel_error.mean()),
                "cell_volume_rel_error_median": float(np.median(rel_error)),
                "cell_volume_rel_error_p95": float(np.quantile(rel_error, 0.95)),
                "cell_volume_rel_error_max": float(rel_error.max()),
                "cell_volume_total_true": float(cell_volumes.sum()),
                "cell_volume_total_estimate": float(cell_volume_estimate.sum()),
            }
        )

    return ReservoirMeshData(
        verts=verts,
        cell_to_unique_vertices=cell_to_unique_vertices,
        vertex_to_cells=vertex_to_cells,
        boundary_faces=boundary_faces,
        boundary_vertices=boundary_vertices,
        active_cell_indices=active_cell_indices,
        active_mask=active_mask,
        corner_cells=corner_cells,
        cell_frames=frames,
        cell_lengths=lengths,
        cell_centroids=centroids,
        cell_volumes=cell_volumes,
        cell_volume_estimate=cell_volume_estimate,
        volume_diagnostics=volume_diagnostics,
        active_cell_lookup=active_cell_lookup,
        cell_geometry_owner=cell_geometry_owner,
        cell_merge_hops=cell_merge_hops,
    )


def _hex_vertex_adjacency(cell_to_unique_vertices: list[np.ndarray], n_vertices: int):
    r"""Build the undirected vertex adjacency graph induced by hexahedral cells.

    Every pair of vertices sharing a cell is connected, i.e. each cell contributes a
    clique over its unique vertices.  This yields the same connected components as the
    hexahedral edge graph (a single hex is vertex-connected), which is all the
    component diagnostics require.

    where:

     - ``cell_to_unique_vertices``: per-cell lists of unique (remapped) vertex ids.
     - ``n_vertices``: total number of mesh vertices, sizing the sparse adjacency.
    """
    from scipy.sparse import coo_matrix

    rows = []
    cols = []
    for vertex_ids in cell_to_unique_vertices:
        vertex_ids = np.asarray(vertex_ids, dtype=int)
        if vertex_ids.size < 2:
            continue
        a, b = np.triu_indices(vertex_ids.size, k=1)
        rows.append(vertex_ids[a])
        cols.append(vertex_ids[b])
    if not rows:
        return coo_matrix((n_vertices, n_vertices), dtype=np.int8).tocsr()
    edge_rows = np.concatenate(rows)
    edge_cols = np.concatenate(cols)
    sym_rows = np.concatenate((edge_rows, edge_cols))
    sym_cols = np.concatenate((edge_cols, edge_rows))
    data = np.ones(sym_rows.shape[0], dtype=np.int8)
    return coo_matrix((data, (sym_rows, sym_cols)), shape=(n_vertices, n_vertices)).tocsr()


def compute_component_diagnostics(
    reservoir_mesh: ReservoirMeshData,
    xtol: float = 1.0,
    max_sample_matches: int = 5,
) -> dict[str, Any]:
    """Compute connected-component and near-contact diagnostics for a reservoir mesh."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n_vertices = int(reservoir_mesh.verts.shape[0])
    adjacency = _hex_vertex_adjacency(reservoir_mesh.cell_to_unique_vertices, n_vertices)
    n_components, component_labels = connected_components(adjacency, directed=False)
    component_sizes_raw = np.bincount(component_labels, minlength=n_components)
    component_order = np.argsort(component_sizes_raw)[::-1]
    component_sizes = component_sizes_raw[component_order]
    component_vertex_ids = [np.where(component_labels == comp_id)[0] for comp_id in component_order]

    close_count_matrix = np.zeros((n_components, n_components), dtype=np.int32)
    min_distance_matrix = np.full((n_components, n_components), np.nan, dtype=float)
    sample_matches: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    global_cross_nn_min = np.inf
    global_cross_nn_max = 0.0
    global_positive_cross_nn_min = np.inf

    trees = [cKDTree(reservoir_mesh.verts[vertex_ids]) for vertex_ids in component_vertex_ids]
    for i in range(n_components):
        min_distance_matrix[i, i] = 0.0
        verts_i_ids = component_vertex_ids[i]
        verts_i = reservoir_mesh.verts[verts_i_ids]
        tree_i = trees[i]
        for j in range(i + 1, n_components):
            verts_j_ids = component_vertex_ids[j]
            verts_j = reservoir_mesh.verts[verts_j_ids]
            tree_j = trees[j]

            dist_i_to_j, nn_i_to_j = tree_j.query(verts_i, k=1)
            dist_j_to_i, nn_j_to_i = tree_i.query(verts_j, k=1)

            pairwise_nn = np.concatenate((dist_i_to_j, dist_j_to_i))
            min_distance = float(pairwise_nn.min()) if pairwise_nn.size else np.nan
            max_distance = float(pairwise_nn.max()) if pairwise_nn.size else np.nan
            min_distance_matrix[i, j] = min_distance
            min_distance_matrix[j, i] = min_distance

            if pairwise_nn.size:
                global_cross_nn_min = min(global_cross_nn_min, min_distance)
                global_cross_nn_max = max(global_cross_nn_max, max_distance)
                positive_pairwise_nn = pairwise_nn[pairwise_nn > 0.0]
                if positive_pairwise_nn.size:
                    global_positive_cross_nn_min = min(global_positive_cross_nn_min, float(positive_pairwise_nn.min()))

            matched_i = dist_i_to_j <= xtol
            matched_j = dist_j_to_i <= xtol
            close_count_matrix[i, j] = int(matched_i.sum())
            close_count_matrix[j, i] = int(matched_j.sum())

            if matched_i.any():
                sample_local_i = np.where(matched_i)[0][:max_sample_matches]
                sample_matches[(i, j)] = [
                    (
                        int(verts_i_ids[k]),
                        int(verts_j_ids[nn_i_to_j[k]]),
                        float(dist_i_to_j[k]),
                    )
                    for k in sample_local_i
                ]

    suspect_pairs = [
        (i, j)
        for i in range(n_components)
        for j in range(i + 1, n_components)
        if close_count_matrix[i, j] > 0 or close_count_matrix[j, i] > 0
    ]
    labels = [f"C{i}\n(n={int(size)})" for i, size in enumerate(component_sizes)]
    return {
        "adjacency": adjacency,
        "n_components": int(n_components),
        "component_labels": component_labels,
        "component_order": component_order,
        "component_sizes": component_sizes,
        "component_vertex_ids": component_vertex_ids,
        "close_count_matrix": close_count_matrix,
        "min_distance_matrix": min_distance_matrix,
        "sample_matches": sample_matches,
        "global_cross_nn_min": float(global_cross_nn_min) if np.isfinite(global_cross_nn_min) else np.nan,
        "global_cross_nn_max": float(global_cross_nn_max),
        "global_positive_cross_nn_min": (
            float(global_positive_cross_nn_min) if np.isfinite(global_positive_cross_nn_min) else np.nan
        ),
        "suspect_pairs": suspect_pairs,
        "labels": labels,
        "xtol": float(xtol),
    }


def select_time_indices(
    n_times: int,
    selected_steps=None,
    max_steps: int | None = 16,
    available_steps=None,
) -> np.ndarray:
    if selected_steps is not None:
        idx = np.asarray(selected_steps, dtype=int)
        if idx.ndim != 1:
            raise ValueError("selected_steps must be a 1D iterable of indices.")
        if np.all((idx >= 0) & (idx < n_times)):
            return np.unique(idx)
        if available_steps is not None:
            available_steps = np.asarray(available_steps, dtype=int).reshape(-1)
            step_to_pos = {int(step): pos for pos, step in enumerate(available_steps.tolist())}
            missing = [int(step) for step in idx.tolist() if int(step) not in step_to_pos]
            if not missing:
                mapped = np.asarray([step_to_pos[int(step)] for step in idx.tolist()], dtype=int)
                return np.unique(mapped)
            raise IndexError(
                "selected_steps contains values outside the available pressure timeline. "
                f"Missing report steps: {missing}."
            )
        raise IndexError("selected_steps contains indices outside the available pressure timeline.")

    if max_steps is None:
        return np.arange(n_times, dtype=int)

    n_select = min(max_steps, n_times)
    if n_select <= 1:
        return np.array([0], dtype=int)
    idx = np.linspace(0, n_times - 1, n_select)
    idx = np.unique(np.rint(idx).astype(int))
    if idx[0] != 0:
        idx = np.insert(idx, 0, 0)
    if idx[-1] != n_times - 1:
        idx = np.append(idx, n_times - 1)
    return np.unique(idx)


def build_cell_rock_physics_from_arrays(
    reservoir_mesh: ReservoirMeshData,
    perms,
    poro,
) -> dict[str, Any]:
    """Build per-cell permeability tensors and porosity from active-cell rock arrays."""
    perms = np.asarray(perms, dtype=float)
    poro = np.asarray(poro, dtype=float)
    if perms.ndim != 2 or perms.shape[1] != 3:
        raise ValueError(f"perms must have shape (n_cells, 3); got {perms.shape}.")
    if poro.ndim != 1 or poro.shape[0] != perms.shape[0]:
        raise ValueError(
            f"poro must have shape ({perms.shape[0]},); got {poro.shape}."
        )
    geometry_owner = np.asarray(getattr(reservoir_mesh, "cell_geometry_owner", np.arange(perms.shape[0])), dtype=int)
    merge_weights = (
        np.asarray(reservoir_mesh.cell_volumes, dtype=float)
        if reservoir_mesh.cell_volumes is not None
        else np.ones(perms.shape[0], dtype=float)
    )
    merge_weights = np.where(np.isfinite(merge_weights) & (merge_weights > 0.0), merge_weights, 1.0)

    owner_unique, owner_inverse = np.unique(geometry_owner, return_inverse=True)
    weight_sums = np.bincount(owner_inverse, weights=merge_weights).astype(float)
    safe_weight_sums = np.where(weight_sums > 0.0, weight_sums, 1.0)
    perm_sums = np.vstack(
        [
            np.bincount(owner_inverse, weights=merge_weights * perms[:, axis], minlength=len(owner_unique))
            for axis in range(perms.shape[1])
        ]
    ).T
    poro_sums = np.bincount(owner_inverse, weights=merge_weights * poro, minlength=len(owner_unique)).astype(float)
    merged_perms = (perm_sums / safe_weight_sums[:, None])[owner_inverse]
    merged_poro = (poro_sums / safe_weight_sums)[owner_inverse]

    frames = reservoir_mesh.cell_frames
    cell_tensors = np.einsum("eia,ea,eja->eij", frames, merged_perms, frames)

    return {
        "cell_permeability_diag": merged_perms,
        "raw_cell_permeability_diag": perms,
        "cell_tensors": cell_tensors,
        "porosity": merged_poro,
        "raw_porosity": poro,
        "cell_geometry_owner": geometry_owner,
    }


def prepare_aquifer_vertices(field, reservoir_mesh: ReservoirMeshData) -> np.ndarray:
    """Return reservoir-mesh vertices touched by modeled aquifers when available."""
    if not _field_has_component(field, "aquifers"):
        return np.zeros((0,), dtype=np.int32)
    aquifers = getattr(field, "aquifers", None)
    if aquifers is None or len(getattr(aquifers, "attributes", ())) == 0:
        return np.zeros((0,), dtype=np.int32)

    candidate_attrs = ("I", "J", "K", "K1", "K2")
    vertex_ids: set[int] = set()
    for attr in getattr(aquifers, "attributes", ()):
        table = getattr(aquifers, attr.lower(), None)
        if table is None or not hasattr(table, "columns"):
            continue
        required = [col for col in candidate_attrs if col in table.columns]
        if len(required) < 3:
            continue
        for _, row in table.iterrows():
            try:
                i = int(row["I"])
                j = int(row["J"])
                k1 = int(row["K1"] if "K1" in row else row["K"])
                k2 = int(row["K2"] if "K2" in row else row["K"])
            except Exception:
                continue
            for k in range(k1, k2 + 1):
                cell_idx = reservoir_mesh.active_cell_lookup.get((i, j, k))
                if cell_idx is None:
                    continue
                vertex_ids.update(int(v) for v in reservoir_mesh.cell_to_unique_vertices[cell_idx].tolist())
    return np.asarray(sorted(vertex_ids), dtype=np.int32)


def prepare_vertex_category_indices(
    n_vertices: int,
    boundary_vertices,
    well_vertex_ids,
    aquifer_vertices=None,
    extra_near_well_vertices=None,
) -> dict[str, np.ndarray]:
    """Build disjoint vertex index buckets for sequence batch sampling."""
    all_vertices = np.arange(int(n_vertices), dtype=np.int32)
    boundary_vertices = np.asarray(boundary_vertices, dtype=np.int32)
    well_vertex_ids = np.asarray(well_vertex_ids, dtype=np.int32)
    aquifer_vertices = np.asarray(aquifer_vertices if aquifer_vertices is not None else np.zeros((0,), dtype=np.int32), dtype=np.int32)
    extra_near_well_vertices = np.asarray(
        extra_near_well_vertices if extra_near_well_vertices is not None else np.zeros((0,), dtype=np.int32),
        dtype=np.int32,
    )

    boundary_set = set(int(v) for v in boundary_vertices.tolist())
    well_set = set(int(v) for v in well_vertex_ids.tolist())
    extra_near_well_set = set(int(v) for v in extra_near_well_vertices.tolist())
    aquifer_set = set(int(v) for v in aquifer_vertices.tolist())

    near_well = np.asarray(sorted((well_set | extra_near_well_set) - boundary_set - aquifer_set), dtype=np.int32)
    boundary = np.asarray(sorted(boundary_set - aquifer_set), dtype=np.int32)
    aquifer = np.asarray(sorted(aquifer_set), dtype=np.int32)
    interior = np.asarray(
        sorted(set(int(v) for v in all_vertices.tolist()) - set(boundary.tolist()) - set(near_well.tolist()) - set(aquifer.tolist())),
        dtype=np.int32,
    )
    return {
        "all": all_vertices,
        "boundary": boundary,
        "interior": interior,
        "near_well": near_well,
        "aquifer": aquifer,
    }


def build_blackoil_table_pack(field) -> dict[str, np.ndarray]:
    """Convert DeepField black-oil tables into JAX-friendly numpy arrays."""
    _require_deepfield()
    validate_loaded_model(field, components=("tables",))
    tables = getattr(field, "tables", None)
    if tables is None:
        raise ValueError("Black-oil closure helpers require a loaded DeepField tables component.")

    def _table_array(name):
        table = getattr(tables, name.lower(), None)
        if table is None:
            raise AttributeError(f"Missing required black-oil table '{name}'.")
        return table

    swof = _table_array("SWOF")
    sgof = _table_array("SGOF")
    pvtw = _table_array("PVTW")
    pvto = _table_array("PVTO")
    pvdg = _table_array("PVDG")
    rsvd = getattr(tables, "rsvd", None)
    rock = getattr(tables, "rock", None)
    density = _table_array("DENSITY")

    pvto_sat, _, _, _ = split_pvto(pvto)
    pvto_sat_index = pvto_sat.index.to_frame(index=False)

    pack = {
        "swof_sw": np.asarray(swof.index.values, dtype=float),
        "swof_krw": np.asarray(swof["KRWO"].values, dtype=float),
        "swof_krow": np.asarray(swof["KROW"].values, dtype=float),
        "swof_pcow": np.asarray(swof["POW"].values, dtype=float),
        "sgof_sg": np.asarray(sgof.index.values, dtype=float),
        "sgof_krg": np.asarray(sgof["KRGO"].values, dtype=float),
        "sgof_krog": np.asarray(sgof["KROG"].values, dtype=float),
        "sgof_pcgo": np.asarray(sgof["POG"].values, dtype=float),
        "pvtw_pressure": np.asarray(pvtw.index.values, dtype=float),
        "pvtw_fvf": np.asarray(pvtw["FVF"].values, dtype=float),
        "pvtw_compr": np.asarray(pvtw["COMPR"].values, dtype=float),
        "pvtw_visc": np.asarray(pvtw["VISC"].values, dtype=float),
        "pvtw_viscosibility": np.asarray(pvtw["VISCOSIBILITY"].values, dtype=float),
        "pvdg_pressure": np.asarray(pvdg.index.values, dtype=float),
        "pvdg_fvf": np.asarray(pvdg["FVF"].values, dtype=float),
        "pvdg_visc": np.asarray(pvdg["VISC"].values, dtype=float),
        "pvto_rs_sat": np.asarray(pvto_sat_index[pvto_sat.index.names[0]].values, dtype=float),
        "pvto_pbub_sat": np.asarray(pvto_sat_index[pvto_sat.index.names[1]].values, dtype=float),
        "pvto_fvf_sat": np.asarray(pvto_sat["FVF"].values, dtype=float),
        "pvto_visc_sat": np.asarray(pvto_sat["VISC"].values, dtype=float),
        "dens_o": np.asarray(density["DENSO"].values, dtype=float),
        "dens_w": np.asarray(density["DENSW"].values, dtype=float),
        "dens_g": np.asarray(density["DENSG"].values, dtype=float),
        "swc_baker": np.asarray(float(swof.index.values[0]) if len(swof.index.values) else 0.0, dtype=float),
    }
    if rsvd is not None:
        pack["rsvd_depth"] = np.asarray(rsvd.index.values, dtype=float)
        pack["rsvd_rs"] = np.asarray(rsvd["RS"].values, dtype=float)
    else:
        pack["rsvd_depth"] = np.zeros((0,), dtype=float)
        pack["rsvd_rs"] = np.zeros((0,), dtype=float)
    if rock is not None and len(rock.index.values):
        pack["rock_pref"] = np.asarray(float(rock.index.values[0]), dtype=float)
        pack["rock_compr"] = np.asarray(float(rock["COMPR"].values[0]), dtype=float)
    else:
        pack["rock_pref"] = np.asarray(0.0, dtype=float)
        pack["rock_compr"] = np.asarray(0.0, dtype=float)
    return pack


def _build_well_frame(well_tangent: np.ndarray) -> np.ndarray:
    t = _normalize(np.asarray(well_tangent, dtype=float))
    if abs(t[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    e1 = _normalize(np.cross(ref, t))
    e2 = _normalize(np.cross(t, e1))
    return np.column_stack((e1, e2, t))


def _spatialize_track(track: np.ndarray | None) -> np.ndarray | None:
    if track is None:
        return None
    track = np.asarray(track, dtype=float)
    if track.ndim == 1:
        track = track.reshape(1, -1)
    if track.ndim != 2 or track.shape[1] < 3:
        return None
    # DeepField WELLTRACK commonly stores [x, y, z, md]; only xyz is geometric.
    return track[:, :3]


def _block_path_tangents(block_centroids: np.ndarray) -> np.ndarray:
    tangents = np.zeros_like(block_centroids)
    if len(block_centroids) == 1:
        tangents[0] = np.array([0.0, 0.0, 1.0])
        return tangents
    for idx in range(len(block_centroids)):
        if idx == 0:
            direction = block_centroids[1] - block_centroids[0]
        elif idx == len(block_centroids) - 1:
            direction = block_centroids[-1] - block_centroids[-2]
        else:
            direction = block_centroids[idx + 1] - block_centroids[idx - 1]
        if np.linalg.norm(direction) <= 1e-12:
            direction = np.array([0.0, 0.0, 1.0])
        tangents[idx] = direction
    return tangents


def compute_qw_full_tensor(
    k_tensor: np.ndarray,
    well_tangent: np.ndarray,
    cell_sizes: tuple[float, float, float],
    p_bh: float,
    z_bh: float,
    z_cell: float,
    r_w: float,
    skin: float = 0.0,
) -> dict[str, Any]:
    r"""Compute the Peaceman well index from the full-tensor, well-aligned geometry.

    Purely geometric: the index :math:`\mathrm{WI} = 2\pi h_s \sqrt{\det K_\perp} /
    (\ln(r_e/r_w) + s)` depends on the perforated length, the transverse
    permeability block, the equivalent radius and the skin; the phase mobilities
    that turn it into a rate live in the black-oil closures at training time.
    """
    k_tensor = np.asarray(k_tensor, dtype=float)
    if k_tensor.shape != (3, 3):
        raise ValueError("k_tensor must be 3x3.")
    hx, hy, hz = map(float, cell_sizes)
    k_tensor = 0.5 * (k_tensor + k_tensor.T)

    Q = _build_well_frame(well_tangent)
    k_tilde = Q.T @ k_tensor @ Q
    K_perp = k_tilde[:2, :2]
    eigvals, eigvecs = np.linalg.eigh(K_perp)
    if np.any(eigvals <= 0.0):
        raise ValueError("The transverse permeability block must be positive definite.")
    lambda1, lambda2 = eigvals

    tangent = Q[:, 2]
    h_s = abs(tangent[0]) * hx + abs(tangent[1]) * hy + abs(tangent[2]) * hz
    N = Q[:, :2] @ eigvecs
    n1 = N[:, 0]
    n2 = N[:, 1]
    h_1_perp = abs(n1[0]) * hx + abs(n1[1]) * hy + abs(n1[2]) * hz
    h_2_perp = abs(n2[0]) * hx + abs(n2[1]) * hy + abs(n2[2]) * hz

    # Peaceman (1983) anisotropic equivalent radius,
    #   r_e = 0.28 * sqrt( sqrt(k2/k1) h1^2 + sqrt(k1/k2) h2^2 ) / ( (k2/k1)^1/4 + (k1/k2)^1/4 ),
    # which reduces to r_e = 0.198 dx on an isotropic square cell. (An earlier form put the
    # 0.28 inside the square root as 0.14 and halved the denominator, giving 0.529 dx —
    # 2.7x too large, i.e. a well index ~20 % too small on SPE2EQUI's 164-ft cells.)
    re_num = 0.28 * math.sqrt(
        math.sqrt(lambda2 / lambda1) * (h_1_perp ** 2)
        + math.sqrt(lambda1 / lambda2) * (h_2_perp ** 2)
    )
    re_den = (lambda2 / lambda1) ** 0.25 + (lambda1 / lambda2) ** 0.25
    r_e = re_num / re_den
    if r_e <= r_w:
        raise ValueError("Equivalent radius must exceed the well radius.")

    det_k_perp = float(np.linalg.det(K_perp))
    WI = 2.0 * math.pi * h_s * math.sqrt(det_k_perp) / (math.log(r_e / r_w) + skin)
    return {
        "WI": WI,
        "r_e": r_e,
        "h_s": h_s,
        "h_1_perp": h_1_perp,
        "h_2_perp": h_2_perp,
        "K_perp": K_perp,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "z_bh": float(z_bh),
        "z_cell": float(z_cell),
        "p_bh": float(p_bh),
        "skin": float(skin),
        "r_w": float(r_w),
    }


def _latest_schedule_row(table: Any, current_date: pd.Timestamp) -> pd.Series | None:
    """Return the latest dated schedule row not newer than the requested timestamp."""
    if table is None or len(table) == 0:
        return None
    if not hasattr(table, "columns") or "DATE" not in table.columns:
        return None
    df = table.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df[df["DATE"] <= current_date]
    if len(df) == 0:
        return None
    return df.sort_values("DATE").iloc[-1].copy()


def _record_float(record: Any, columns: list[str] | tuple[str, ...]) -> float:
    """Extract the first finite numeric value from a row-like object."""
    for column in columns:
        if column not in record:
            continue
        value = record[column]
        if pd.isna(value):
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return np.nan


def _record_text(record: Any, columns: list[str] | tuple[str, ...]) -> str:
    """Extract the first non-empty textual value from a row-like object."""
    for column in columns:
        if column not in record:
            continue
        value = record[column]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text.upper()
    return ""


def _segment_reference_depth(segment: Any) -> float:
    """Return a well reference depth from WELSPECS when available."""
    welspecs = getattr(segment, "welspecs", None)
    if welspecs is None or len(welspecs) == 0:
        return np.nan
    row = welspecs.iloc[-1]
    return _record_float(row, ("DREF",))


def _parse_eclipse_date(value: Any) -> pd.Timestamp:
    """Parse Eclipse-style dates such as ``06 'NOV' 1997`` into timestamps."""
    if value is None:
        return pd.NaT
    text = str(value).replace("'", " ").strip()
    return pd.to_datetime(text, errors="coerce")


def _strip_schedule_comment(line: str) -> str:
    """Remove Eclipse ``--`` comments from a schedule line."""
    return line.split("--", 1)[0].strip()


def _tokenize_schedule_line(line: str) -> list[str]:
    """Tokenize an Eclipse schedule line while preserving quoted well names."""
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return [token for token in lexer if token != "/"]


def _expand_eclipse_tokens(tokens: list[str]) -> list[str | None]:
    """Expand simple Eclipse defaults such as ``5*`` into explicit placeholders."""
    expanded: list[str | None] = []
    for token in tokens:
        token = str(token).strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\*(.*)", token)
        if match:
            count = int(match.group(1))
            value = match.group(2).strip()
            fill = value if value else None
            expanded.extend([fill] * count)
            continue
        expanded.append(token)
    return expanded


def _parse_optional_float(value: Any) -> float:
    """Convert an optional scalar into a finite float when possible."""
    if value is None:
        return np.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    if np.isfinite(numeric):
        return numeric
    return np.nan


def _collect_schedule_include_paths(model_path: Path) -> list[Path]:
    """Return files included after the deck enters its ``SCHEDULE`` section."""
    if not model_path.is_file():
        return []

    include_paths: list[Path] = []
    in_schedule = False
    pending_include = False

    for raw_line in model_path.read_text(errors="ignore").splitlines():
        line = _strip_schedule_comment(raw_line)
        if not line:
            continue
        upper = line.upper()
        if not in_schedule:
            if upper.startswith("SCHEDULE"):
                in_schedule = True
            continue
        if upper.startswith("END"):
            break
        if pending_include:
            include_rel = _extract_include_path(line)
            if include_rel is not None:
                include_paths.append((model_path.parent / include_rel).resolve())
                pending_include = False
            continue
        if upper.startswith("INCLUDE"):
            include_rel = _extract_include_path(line)
            if include_rel is not None:
                include_paths.append((model_path.parent / include_rel).resolve())
            else:
                pending_include = True

    filtered = [path for path in include_paths if path.suffix.upper() in {".SCH", ".INC", ".DATA"}]
    return filtered or include_paths


def _parse_tstep_increment(line: str) -> float:
    """Parse a TSTEP payload and return the total number of elapsed days."""
    tokens = _tokenize_schedule_line(line)
    if tokens and tokens[0].upper() == "TSTEP":
        tokens = tokens[1:]
    total = 0.0
    for token in tokens:
        text = str(token).strip()
        if not text:
            continue
        match = re.fullmatch(r"(\d+)\*(.*)", text)
        if match:
            value = match.group(2).strip()
            if not value:
                continue
            try:
                total += int(match.group(1)) * float(value)
            except ValueError:
                continue
            continue
        try:
            total += float(text)
        except ValueError:
            continue
    return float(total)


def _parse_wconprod_row(line: str, current_date: pd.Timestamp) -> dict[str, Any] | None:
    """Parse one WCONPROD row into a lightweight control record."""
    tokens = _tokenize_schedule_line(line)
    if len(tokens) < 2:
        return None

    values = _expand_eclipse_tokens(tokens[3:]) if len(tokens) > 3 else []
    return {
        "DATE": pd.to_datetime(current_date),
        "WELL": str(tokens[0]).upper(),
        "MODE": str(tokens[1]).upper(),
        "CONTROL": str(tokens[2]).upper() if len(tokens) > 2 and tokens[2] != "*" else "",
        "OPT": _parse_optional_float(values[0] if len(values) > 0 else None),
        "WPT": _parse_optional_float(values[1] if len(values) > 1 else None),
        "GPT": _parse_optional_float(values[2] if len(values) > 2 else None),
        "LPT": _parse_optional_float(values[3] if len(values) > 3 else None),
        "SLPT": _parse_optional_float(values[4] if len(values) > 4 else None),
        "BHPT": _parse_optional_float(values[5] if len(values) > 5 else None),
        "CONTROL_SOURCE": "WCONPROD",
    }


def _merge_schedule_control_cache(target: dict[str, dict[str, list[dict[str, Any]]]], source: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    """Merge parsed schedule-control rows into a single per-well cache."""
    for well_name, tables in source.items():
        merged = target.setdefault(well_name, {"wconhist": [], "welopen": [], "wconprod": [], "wconinje": []})
        for table_name, rows in tables.items():
            merged.setdefault(table_name, []).extend(rows)


def _parse_schedule_control_file(schedule_path: Path, start_date: pd.Timestamp) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Parse WCON* and WELOPEN rows from a schedule include or deck file."""
    if not schedule_path.is_file():
        return {}

    controls: dict[str, dict[str, list[dict[str, Any]]]] = {}
    current_date = pd.to_datetime(start_date)
    active_block = ""
    require_schedule_section = schedule_path.suffix.upper() == ".DATA"
    in_schedule = not require_schedule_section

    for raw_line in schedule_path.read_text(errors="ignore").splitlines():
        line = _strip_schedule_comment(raw_line)
        if not line:
            continue
        upper = line.upper()

        if not in_schedule:
            if upper.startswith("SCHEDULE"):
                in_schedule = True
            continue
        if require_schedule_section and upper.startswith("END"):
            break

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

        if upper.startswith("WCONHIST"):
            active_block = "WCONHIST"
            continue
        if upper.startswith("WCONPROD"):
            active_block = "WCONPROD"
            continue
        if upper.startswith("WCONINJE"):
            active_block = "WCONINJE"
            continue
        if upper.startswith("WELOPEN"):
            active_block = "WELOPEN"
            continue

        if active_block == "WCONHIST":
            row = _parse_wconhist_row(line, current_date)
            if row is None:
                continue
            tables = controls.setdefault(row["WELL"], {"wconhist": [], "welopen": [], "wconprod": [], "wconinje": []})
            tables["wconhist"].append(row)
            continue

        if active_block == "WCONPROD":
            row = _parse_wconprod_row(line, current_date)
            if row is None:
                continue
            tables = controls.setdefault(row["WELL"], {"wconhist": [], "welopen": [], "wconprod": [], "wconinje": []})
            tables["wconprod"].append(row)
            continue

        if active_block == "WCONINJE":
            row = _parse_wconinje_row(line, current_date)
            if row is None:
                continue
            tables = controls.setdefault(row["WELL"], {"wconhist": [], "welopen": [], "wconprod": [], "wconinje": []})
            tables["wconinje"].append(row)
            continue

        if active_block == "WELOPEN":
            row = _parse_welopen_row(line, current_date)
            if row is None:
                continue
            tables = controls.setdefault(row["WELL"], {"wconhist": [], "welopen": [], "wconprod": [], "wconinje": []})
            tables["welopen"].append(row)

    return controls


def _infer_control_kind(record: Any) -> str:
    """Infer whether a row represents rate control, pressure control, or neither."""
    mode = _record_text(record, ("MODE",))
    if mode in {"STOP", "SHUT", "CLOSE"}:
        return "closed"

    control = _record_text(record, ("CONTROL",))
    if control in {"RATE", "ORAT", "WRAT", "GRAT", "LRAT", "RESV"}:
        return "rate"
    if control in {"BHP", "BHPT", "THP", "THPT"}:
        return "bhp"

    generic_rate = _record_float(record, ("RATE",))
    wit = _record_float(record, ("WIT",))
    git = _record_float(record, ("GIT",))
    if np.isfinite(generic_rate) or np.isfinite(wit) or np.isfinite(git):
        return "rate"

    bhpt = _record_float(record, ("BHPT", "BHP"))
    if np.isfinite(bhpt):
        return "bhp"
    return ""


def _resolve_well_control(segment: Any, current_date: pd.Timestamp) -> pd.Series | None:
    """Resolve an event-like well control row, including WCONHIST/WELOPEN fallbacks."""
    current_date = pd.to_datetime(current_date)
    ref_dref = _segment_reference_depth(segment)
    wellopen = _latest_schedule_row(getattr(segment, "welopen", None), current_date)

    event = _latest_schedule_row(getattr(segment, "events", None), current_date)
    if event is not None:
        event = event.copy()
        if not np.isfinite(_record_float(event, ("DREF",))) and np.isfinite(ref_dref):
            event["DREF"] = ref_dref
        control_kind = _infer_control_kind(event)
        if control_kind == "closed":
            return None
        if wellopen is not None:
            open_date = pd.to_datetime(wellopen["DATE"])
            event_date = pd.to_datetime(event["DATE"])
            if open_date >= event_date and _infer_control_kind(wellopen) == "closed":
                return None
        if control_kind:
            event["CONTROL_KIND"] = control_kind
        return event

    candidates: list[tuple[pd.Timestamp, str, pd.Series]] = []
    for source_name, attr_name in (("WCONPROD", "wconprod"), ("WCONINJE", "wconinje"), ("WCONHIST", "wconhist")):
        row = _latest_schedule_row(getattr(segment, attr_name, None), current_date)
        if row is not None:
            candidates.append((pd.to_datetime(row["DATE"]), source_name, row))

    if not candidates:
        if wellopen is not None and _infer_control_kind(wellopen) == "closed":
            return None
        return None

    control_date, source_name, row = max(candidates, key=lambda item: item[0])
    row = row.copy()
    if wellopen is not None:
        open_date = pd.to_datetime(wellopen["DATE"])
        if open_date >= control_date and _infer_control_kind(wellopen) == "closed":
            return None
        if open_date >= control_date and "MODE" in wellopen:
            row["MODE"] = wellopen["MODE"]
    control_kind = _infer_control_kind(row)
    if control_kind == "closed":
        return None

    bhpt = _record_float(row, ("BHPT", "BHP"))
    dref = _record_float(row, ("DREF",))
    if not np.isfinite(dref):
        dref = ref_dref

    event_like = {
        "DATE": pd.to_datetime(row["DATE"]),
        "BHPT": bhpt,
        "DREF": dref,
        "WIT": np.nan,
        "GIT": np.nan,
        "RATE": np.nan,
        "CONTROL_KIND": control_kind,
        "CONTROL_SOURCE": source_name,
    }

    if source_name == "WCONINJE":
        injected_rate = _record_float(row, ("SPIT", "PIT"))
        phase = _record_text(row, ("PHASE",))
        event_like["PHASE"] = phase
        if phase.startswith("WAT"):
            event_like["WIT"] = injected_rate
            event_like["RATE"] = injected_rate
        elif phase.startswith("GAS"):
            event_like["GIT"] = injected_rate
            event_like["RATE"] = injected_rate
        else:
            event_like["RATE"] = injected_rate
    elif source_name == "WCONHIST":
        event_like["RATE"] = _record_float(row, ("RATE",))
        event_like["BHPT"] = _record_float(row, ("BHPT", "BHP"))
        phase = _record_text(row, ("PHASE",))
        if phase:
            event_like["PHASE"] = phase
    else:
        control = _record_text(row, ("CONTROL",))
        phase_by_control = {
            "ORAT": "OIL",
            "WRAT": "WATER",
            "GRAT": "GAS",
        }
        if control in phase_by_control:
            event_like["PHASE"] = phase_by_control[control]
        rate_columns = {
            "ORAT": ("OPT",),
            "WRAT": ("WPT",),
            "GRAT": ("GPT",),
            "LRAT": ("LPT",),
            "RESV": ("SLPT", "LPT"),
            "RATE": ("LPT", "OPT", "WPT", "GPT", "SLPT"),
        }
        raw_rate = _record_float(row, rate_columns.get(control, ("OPT", "WPT", "GPT", "LPT", "SLPT")))
        if np.isfinite(raw_rate):
            event_like["RATE"] = -abs(raw_rate)

    if event_like["CONTROL_KIND"] == "":
        event_like["CONTROL_KIND"] = _infer_control_kind(event_like)
    if event_like["CONTROL_KIND"] == "closed":
        return None
    return pd.Series(event_like)


def _extract_well_result_snapshot(segment: Any, current_date: pd.Timestamp) -> dict[str, float]:
    """Return the latest available well-level result row not newer than the requested date."""
    row = _latest_schedule_row(getattr(segment, "results", None), current_date)
    if row is None:
        return {
            "WBHP": np.nan,
            "WTHP": np.nan,
            "WOPR": np.nan,
            "WWPR": np.nan,
            "WGPR": np.nan,
            "WWIR": np.nan,
            "WGIR": np.nan,
            "WOPT": np.nan,
            "WWPT": np.nan,
            "WGPT": np.nan,
        }
    return {
        "WBHP": _record_float(row, ("WBHP", "BHPT", "BHP")),
        "WTHP": _record_float(row, ("WTHP", "THPT", "THP")),
        "WOPR": _record_float(row, ("WOPR",)),
        "WWPR": _record_float(row, ("WWPR",)),
        "WGPR": _record_float(row, ("WGPR",)),
        "WWIR": _record_float(row, ("WWIR", "WIT")),
        "WGIR": _record_float(row, ("WGIR", "GIT")),
        "WOPT": _record_float(row, ("WOPT",)),
        "WWPT": _record_float(row, ("WWPT",)),
        "WGPT": _record_float(row, ("WGPT",)),
    }


def _block_centroids_from_reservoir_mesh(block_indices: np.ndarray, reservoir_mesh: ReservoirMeshData) -> np.ndarray:
    """Map ijk well blocks to reservoir-mesh cell centroids while preserving block order."""
    block_indices = np.asarray(block_indices, dtype=int).reshape(-1, 3)
    block_centroids = np.full((len(block_indices), 3), np.nan, dtype=float)
    active_lookup = reservoir_mesh.active_cell_lookup
    for idx, block in enumerate(block_indices):
        cell_idx = active_lookup.get(tuple(int(v) for v in block.tolist()))
        if cell_idx is not None:
            block_centroids[idx] = reservoir_mesh.cell_centroids[cell_idx]

    valid_mask = np.all(np.isfinite(block_centroids), axis=1)
    if not valid_mask.any():
        raise ValueError(
            "Could not map any well blocks to reservoir-mesh centroids. "
            "This usually means the well blocks and the active reservoir mesh are out of sync."
        )

    if not valid_mask.all():
        valid_ids = np.flatnonzero(valid_mask)
        for idx in np.flatnonzero(~valid_mask):
            nearest_valid = valid_ids[np.argmin(np.abs(valid_ids - idx))]
            block_centroids[idx] = block_centroids[nearest_valid]

    return block_centroids


def _resolve_rate_phase_split(
    control_phase: str,
    total_rate: float,
    result_snapshot: dict[str, float],
) -> list[tuple[str, float]]:
    """Resolve a rate-controlled well into one or more ``(phase, phase_rate)`` pairs.

    A non-empty ``control_phase`` (e.g. ORAT/WRAT/GRAT or an injected fluid) keeps
    the single-phase behaviour. An empty ``control_phase`` (RESV / LRAT / generic
    RATE control, which does not pin a single phase) is split across phases using
    the observed summary rates. A deterministic default keeps ``control_phase`` from
    ever being empty when no usable observations exist.
    """
    phase = str(control_phase).upper()
    if phase:
        return [(phase, float(total_rate))]
    observed = _observed_phase_rates(result_snapshot, total_rate)
    if observed:
        return [(p, float(r)) for p, r in observed]
    return [("WATER" if total_rate > 0.0 else "OIL", float(total_rate))]


def _well_control_phase_label(
    control_phase: str,
    total_rate: float,
    result_snapshot: dict[str, float],
) -> str:
    """Best-effort non-empty phase label for a well-level record (cosmetic)."""
    phase = str(control_phase).upper()
    if phase:
        return phase
    observed = _observed_phase_rates(result_snapshot, total_rate)
    if observed:
        return max(observed, key=lambda pr: abs(pr[1]))[0]
    return ""


def pack_sequence_well_steps(
    metadata,
    eigfuncs_scaled,
    reservoir_mesh: ReservoirMeshData,
    max_cell_vertices: int = 8,
) -> dict[str, Any]:
    """Pack well/perforation metadata into fixed-order arrays for sequence batches."""
    steps = metadata.get("steps", [])
    eigfuncs_scaled = np.asarray(eigfuncs_scaled, dtype=float)
    n_eigs = eigfuncs_scaled.shape[1] if eigfuncs_scaled.ndim == 2 else 0

    perf_keys = []
    well_names = []
    for step in steps:
        for entry in step.get("bhp_entries", []):
            perf_keys.append((str(entry["well_name"]), int(entry["perf_id"])))
            well_names.append(str(entry["well_name"]))
        for entry in step.get("rate_entries", []):
            perf_keys.append((str(entry["well_name"]), int(entry["perf_id"])))
            well_names.append(str(entry["well_name"]))
        for item in step.get("well_results", []):
            well_names.append(str(item["well_name"]))

    perf_keys = sorted(set(perf_keys))
    well_names = sorted(set(well_names))
    perf_index = {key: idx for idx, key in enumerate(perf_keys)}
    well_index = {name: idx for idx, name in enumerate(well_names)}
    n_perf = len(perf_keys)
    n_wells = len(well_names)

    perf_cell_idx = np.full((n_perf,), -1, dtype=np.int32)
    perf_cell_vertices = np.zeros((n_perf, max_cell_vertices), dtype=np.int32)
    perf_cell_mask = np.zeros((n_perf, max_cell_vertices), dtype=float)
    perf_vertex_weights = np.zeros((n_perf, max_cell_vertices), dtype=float)
    perf_z_bh = np.zeros((n_perf,), dtype=float)
    perf_z_cell = np.zeros((n_perf,), dtype=float)
    perf_h_s = np.zeros((n_perf,), dtype=float)
    perf_r_e_base = np.zeros((n_perf,), dtype=float)
    perf_skin_base = np.zeros((n_perf,), dtype=float)
    perf_k_perp = np.zeros((n_perf, 2, 2), dtype=float)
    perf_r_w = np.full((n_perf,), np.nan, dtype=float)        # wellbore radius [ft]
    perf_cf = np.full((n_perf,), np.nan, dtype=float)         # explicit COMPDAT connection factor (NaN = geometric)
    perf_eigfuncs = np.zeros((n_perf, n_eigs), dtype=float)
    perf_well_id = np.full((n_perf,), -1, dtype=np.int32)
    perf_id = np.full((n_perf,), -1, dtype=np.int32)

    for step in steps:
        for entry in list(step.get("bhp_entries", [])) + list(step.get("rate_entries", [])):
            key = (str(entry["well_name"]), int(entry["perf_id"]))
            idx = perf_index[key]
            vertices = np.asarray(entry["cell_vertices"], dtype=np.int32)
            weights = np.asarray(entry["vertex_weights"], dtype=float)
            n_local = min(len(vertices), max_cell_vertices)
            perf_cell_idx[idx] = int(entry["cell_idx"])
            perf_cell_vertices[idx, :n_local] = vertices[:n_local]
            perf_cell_mask[idx, :n_local] = 1.0
            perf_vertex_weights[idx, :n_local] = weights[:n_local]
            perf_z_bh[idx] = float(entry["z_bh"])
            perf_z_cell[idx] = float(entry["z_cell"])
            perf_h_s[idx] = float(entry["h_s"])
            perf_r_e_base[idx] = float(entry["r_e"])
            perf_skin_base[idx] = float(entry["skin"])
            perf_k_perp[idx] = np.asarray(entry["K_perp"], dtype=float)
            perf_r_w[idx] = float(entry.get("r_w", np.nan))      # older caches: NaN -> packer default
            perf_cf[idx] = float(entry.get("cf", np.nan))
            perf_well_id[idx] = int(well_index[str(entry["well_name"])])
            perf_id[idx] = int(entry["perf_id"])
            if n_local:
                local_phi = eigfuncs_scaled[vertices[:n_local]]
                local_w = weights[:n_local]
                denom = max(float(np.sum(local_w)), 1e-12)
                perf_eigfuncs[idx] = np.sum(local_phi * local_w[:, None], axis=0) / denom

    step_payloads = []
    for step in steps:
        perf_control_pbh = np.full((n_perf,), np.nan, dtype=float)
        perf_control_rate = np.zeros((n_perf,), dtype=float)
        perf_rate_prior = np.zeros((n_perf, 3), dtype=float)
        perf_rate_prior_mask = np.zeros((n_perf, 3), dtype=float)
        perf_bhp_obs = np.full((n_perf,), np.nan, dtype=float)
        perf_well_obs = np.zeros((n_wells, 6), dtype=float)
        perf_well_obs_mask = np.zeros((n_wells, 6), dtype=float)

        well_results = {str(item["well_name"]): item for item in step.get("well_results", [])}
        for well_name, obs in well_results.items():
            w_idx = well_index[well_name]
            values = np.asarray(
                [
                    obs.get("obs_wopr", np.nan),
                    obs.get("obs_wwpr", np.nan),
                    obs.get("obs_wgpr", np.nan),
                    obs.get("obs_wwir", np.nan),
                    obs.get("obs_wgir", np.nan),
                    obs.get("obs_wbhp", np.nan),
                ],
                dtype=float,
            )
            mask = np.isfinite(values).astype(float)
            perf_well_obs[w_idx] = np.nan_to_num(values, nan=0.0)
            perf_well_obs_mask[w_idx] = mask

        for entry in step.get("bhp_entries", []):
            idx = perf_index[(str(entry["well_name"]), int(entry["perf_id"]))]
            perf_control_pbh[idx] = float(entry["p_bh"])
            if str(entry["well_name"]) in well_results:
                perf_bhp_obs[idx] = float(well_results[str(entry["well_name"])].get("obs_wbhp", np.nan))

        for entry in step.get("rate_entries", []):
            idx = perf_index[(str(entry["well_name"]), int(entry["perf_id"]))]
            perf_control_rate[idx] += float(entry["rate"])  # sum across per-phase entries of a perf
            phase = str(entry.get("control_phase", "")).upper()
            if phase.startswith("WAT"):
                perf_rate_prior[idx, 1] = float(entry["rate"])
                perf_rate_prior_mask[idx, 1] = 1.0
            elif phase.startswith("GAS"):
                perf_rate_prior[idx, 2] = float(entry["rate"])
                perf_rate_prior_mask[idx, 2] = 1.0
            else:
                perf_rate_prior[idx, 0] = float(entry["rate"])
                perf_rate_prior_mask[idx, 0] = 1.0
            if str(entry["well_name"]) in well_results:
                perf_bhp_obs[idx] = float(well_results[str(entry["well_name"])].get("obs_wbhp", np.nan))

        step_payloads.append(
            {
                "perf_control_pbh": np.nan_to_num(perf_control_pbh, nan=0.0),
                "perf_control_pbh_mask": np.isfinite(perf_control_pbh).astype(float),
                "perf_control_rate": perf_control_rate,
                "perf_rate_prior": perf_rate_prior,
                "perf_rate_prior_mask": perf_rate_prior_mask,
                "perf_bhp_obs": np.nan_to_num(perf_bhp_obs, nan=0.0),
                "perf_bhp_obs_mask": np.isfinite(perf_bhp_obs).astype(float),
                "well_phase_obs": perf_well_obs,
                "well_phase_obs_mask": perf_well_obs_mask,
            }
        )

    return {
        "n_perf": n_perf,
        "n_wells": n_wells,
        "perf_keys": perf_keys,
        "well_names": well_names,
        "perf_well_id": perf_well_id,
        "perf_id": perf_id,
        "perf_cell_idx": perf_cell_idx,
        "perf_cell_vertices": perf_cell_vertices,
        "perf_cell_mask": perf_cell_mask,
        "perf_vertex_weights": perf_vertex_weights,
        "perf_z_bh": perf_z_bh,
        "perf_z_cell": perf_z_cell,
        "perf_h_s": perf_h_s,
        "perf_r_e_base": perf_r_e_base,
        "perf_skin_base": perf_skin_base,
        "perf_k_perp": perf_k_perp,
        "perf_r_w": perf_r_w,
        "perf_cf": perf_cf,
        "perf_eigfuncs": perf_eigfuncs,
        "steps": step_payloads,
    }
