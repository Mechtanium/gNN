r"""
The one composition, and the three ways the workflow uses it.

``AUTOPILOT`` is the composition the universal-settings search of the research campaign
chose for every reservoir: a DGM of width 12 and one block on Laplace-eigenfunction inputs
(deflation retention) with the depth, near-well and logarithmic-time channels, the Galerkin
spectral residual with FEM-nodal back-propagation, eight midpoint collocation slices, the
16,384-row full-batch window, and energy natural-gradient descent from the first iteration
with relative Tikhonov damping :math:`10^{-5}`. Its worst-case regret against each
reservoir's own best candidate was 1.04. The settings that depend on the mesh are rules,
not numbers:

.. math::

    n_\lambda \;=\; \max\bigl(n_\lambda^{\mathrm{floor}},\; 16\bigr)

where:

- :math:`n_\lambda`: the number of retained eigenfunctions (``n_eig``), the width of the spectral input.
- :math:`n_\lambda^{\mathrm{floor}}`: the addressability floor of the reservoir's own mesh, the smallest width at which no two cells share an input (:func:`modules.spectral.resolve_n_eig`).

Nothing here is a choice a user makes. A work path fixes what the simulator's run is
observed through and what closes the equations; the page offers the three paths and no
setting of any of them:

==================  ===========================  ==============================  ==========================
\                   history match                history match + forecast        family (property inversion)
==================  ===========================  ==============================  ==========================
time window         every report step            collocation to the end,         every step of every run
                                                 observations to the split
observations        cell states + well rows      cell states + well rows         cell states of N runs
                    (well rows at weight 0.1)    up to the split
well model          closed form                  predicted, with control rows    closed form
                                                 and the three-state switch
closure rows        none                         phase equilibrium and           none
                                                 material balance
network             12 × 1                       12 × 1                          16 × 1, property channels
window              16,384 rows                  16,384 rows                     32,768 rows, near-well mix
==================  ===========================  ==============================  ==========================

The module is import-safe without jax.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import RunConfig, from_flat_dict, loss_groups, to_flat_dict, validate

#: The composition, as flat field names (:func:`modules.config.to_flat_dict`): the reference
#: composition's fixed choices, which the search did not vary, then the ten it chose.
AUTOPILOT: dict = {
    # fixed in the search (the reference composition)
    "input_encoding": "spectral", "architecture": "dgm", "residual_design": "spectral_pde",
    "backprop_design": "fem_nodal", "stiffness_design": "perm_weighted", "sampling": "window",
    "weighting": "fixed", "special_opt": "engd", "special_opt_after": 0, "parallelism": "data",
    "observation": "cell_states", "well_model": "closed_form", "pde_node_weight": "uniform",
    "spectral_projection": "galerkin", "ic_design": "penalty", "rs_supervision": "oil_only",
    "tslice_layout": "midpoint", "n_eig_z": 0, "n_iter": 600, "engd_damping_abs": 0.0,
    "engd_damping_mode": "tikhonov", "engd_solver": "chol", "precision_policy": "selective_f64",
    "opt_f64": True,
    # chosen by the search (worst-case regret 1.04 over the five statistical reservoirs)
    "m_width": 12, "n_blocks": 1, "engd_damping_rel": 1e-5, "spec_retention": "deflation",
    "depth_encoding": "z", "well_encoding": "logr", "time_encoding": "log", "n_tslice": 8,
    "full_batch_cap": 16384, "n_eig": 16,
}

#: Weight of the well rows beside the cell states. At weight 1 the well rows cost the field
#: fit more than they gain at the wells; at 0.1 they fit the bottom-hole pressures best.
WELL_WEIGHT = 0.1

#: What each work path adds to the composition.
HISTORY: dict = {"observation": "both", "well_model": "closed_form"}
FORECAST: dict = {"observation": "both", "well_model": "predicted", "ctrl_switch": "staircase",
                  "inf_mode": "joint"}
FAMILY: dict = {"observation": "cell_states", "well_model": "closed_form", "pde_scale": "random",
                "m_width": 16, "full_batch_cap": 32768}

#: The family's data window: a quarter of its rows at the perforated cells, a quarter within
#: 3.5 cell widths of a well, the rest spread over every cell (:func:`modules.family.family_window`).
FAMILY_WINDOW_MIX: dict = {"perf": 0.25, "near": 0.25, "near_radius": 3.5}

#: Measurement noise of an observed well record: bottom-hole pressure [psi], gas-oil ratio
#: [Mscf/stb] and water cut [-]. The family's held-out error is added to it in quadrature.
RECORD_SIGMA: dict = {"bhp": 10.0, "gor": 0.1, "wct": 0.01}


@dataclass(frozen=True)
class Budget:
    """How much work a path does. The defaults are the composition's; nothing but the test
    suite passes another (no operation, page widget or environment variable can)."""

    n_iter: int = 600            # training iterations
    n_train: int = 24            # simulator runs that teach a family
    n_held: int = 8              # simulator runs held out to score it
    redraw_every: int = 20       # iterations between re-draws of a family's window
    gn_iter: int = 40            # Gauss–Newton iterations of a record's inversion
    n_samples: int = 2000        # posterior samples of an inversion
    snapshot_every: int = 25     # iterations between sweeps of cell-state snapshots
    n_snapshots: int = 8         # report times per sweep

    @classmethod
    def smoke(cls) -> "Budget":
        """A few iterations and a toy family: enough to exercise every stage on a CPU."""
        return cls(n_iter=3, n_train=3, n_held=2, redraw_every=2, gn_iter=3, n_samples=40,
                   snapshot_every=2, n_snapshots=2)


def compose(deck_path: str, work_dir: str, deck_sha: str, path: dict, *, n_iter: int,
            **bound) -> RunConfig:
    """``AUTOPILOT`` with a work path's additions, bound to a deck. ``bound`` carries what
    only the deck or the project can say: the report step a history ends at
    (``train_split_n``), the number of property channels of a family."""
    flat = to_flat_dict(RunConfig())
    flat.update(AUTOPILOT)
    flat.update(path)
    flat.update(bound)
    flat.update(deck_path=str(deck_path), work_dir=str(work_dir), deck_sha=str(deck_sha),
                n_iter=int(n_iter))
    return from_flat_dict(flat)


def history(deck_path: str, work_dir: str, deck_sha: str, budget: Budget = Budget()) -> RunConfig:
    """The full-history match: every report step, cell states and well rows."""
    cfg = compose(deck_path, work_dir, deck_sha, HISTORY, n_iter=budget.n_iter)
    return with_well_weight(cfg)


def forecast(deck_path: str, work_dir: str, deck_sha: str, split_n: int,
             budget: Budget = Budget()) -> RunConfig:
    """The joint history match and forecast: observations up to report step ``split_n``, the
    schedule's controls and the physics beyond it."""
    return compose(deck_path, work_dir, deck_sha, FORECAST, n_iter=budget.n_iter,
                   train_split_n=int(split_n))


def family(deck_path: str, work_dir: str, deck_sha: str, n_properties: int,
           budget: Budget = Budget()) -> RunConfig:
    """The family of reservoirs a property inversion searches: one network over
    ``n_properties`` whitened property channels."""
    return compose(deck_path, work_dir, deck_sha, FAMILY, n_iter=budget.n_iter,
                   property_channels=int(n_properties))


def with_well_weight(cfg: RunConfig, weight: float = WELL_WEIGHT) -> RunConfig:
    """``cfg`` with fixed group weights: ``weight`` on the well rows and 1 on every other
    active group (an aquifer's included)."""
    groups = loss_groups(validate(cfg)[0])
    return replace(cfg, w_fixed=tuple(weight if g == "well" else 1.0 for g in groups))
