r"""
The two run presets the page offers.

``GOLD`` is PINN-Lab's current notebook configuration (``PINN-Lab.ipynb``,
cell 2) with the case fields left to the workflow; ``QUICK`` keeps every
component and physics setting but lets the sizes follow the deck — the
eigenbasis width from the automatic addressability floor and the DGM width
and depth from a fixed parameter target — so a smoke run finishes in minutes
on a CPU workstation.
"""

from __future__ import annotations

from dataclasses import replace

from .config import (Architecture, BackpropDesign, EngdSettings, InputEncoding,
                     ObservationDesign, Parallelism, ResidualDesign, RunConfig,
                     SamplingDesign, SpecialOpt, SpectralSettings, StiffnessDesign,
                     WeightingDesign, WellModel)

GOLD = RunConfig(
    # ---- the switchable components, fixed to the gold standard ----------------------------
    input_encoding=InputEncoding.SPECTRAL,
    architecture=Architecture.DGM,
    residual_design=ResidualDesign.SPECTRAL_PDE,
    backprop_design=BackpropDesign.FEM_NODAL,
    stiffness_design=StiffnessDesign.PERM_WEIGHTED,
    sampling=SamplingDesign.WINDOW,
    weighting=WeightingDesign.FIXED,
    special_opt=SpecialOpt.ENGD,
    special_opt_after=0,
    parallelism=Parallelism.DATA,
    # ---- observations, the well model and the PDE row design ------------------------------
    observation=ObservationDesign.BOTH,
    well_model=WellModel.CLOSED_FORM,
    pde_node_weight="well_gaussian",
    well_gauss_boost=4.0, well_gauss_width=1.0,
    spectral_projection="galerkin",
    rs_supervision="oil_only",
    # ---- encoding channels for the stiff features -----------------------------------------
    time_encoding="linear",
    tslice_layout="midpoint",
    well_encoding="logr",
    # ---- capacity & schedule (0 = auto) ---------------------------------------------------
    n_eig=5, n_eig_z=0,
    m_width=16, n_blocks=2,
    n_iter=20000, seed=30,
    n_tslice=4,
    full_batch_cap=32768,
    # ---- eigenbasis retention, mesh admissibility, encoder adequacy -----------------------
    spec=SpectralSettings(
        retention="deflation",
        stratify_components=False,
        min_comp_vol_frac=0.005,
        min_node_reuse=8,
        addressability_min=1.0,
        addressability_tol=0.01,
        blind_frac_max=0.0,
        n_eig_cap=63,
    ),
    engd=EngdSettings(param_target=6000, damping_rel=1e-4, damping_abs=0.0),
    precision_policy="selective_f64", opt_f64=True,
)

#: Same physics, deck-sized capacity: n_eig from the addressability floor, width
#: and depth from an 800-parameter target, a small full-batch window.
QUICK = replace(
    GOLD,
    n_eig=0, m_width=0, n_blocks=0,
    n_iter=200,
    n_tslice=2,
    full_batch_cap=256,
    engd=EngdSettings(param_target=800, damping_rel=1e-4, damping_abs=0.0),
)

PRESETS = {"gold": GOLD, "quick": QUICK}


def preset(name: str) -> RunConfig:
    try:
        return PRESETS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choose one of {sorted(PRESETS)}") from None
