r"""
Component configuration for the Delta-PINN workflow.

The composition is the notebook's gold standard and is fixed: spectral input
encoding, the DGM architecture, the spectral (Galerkin) PDE residual with
FEM-nodal backprop, permeability-weighted stiffness, the deterministic
full-batch window, fixed loss weights, ENGD, and the closed-form well. Every
training hyperparameter is declared exactly once in :class:`RunConfig`;
:func:`validate` applies the remaining validity rules (fail-fast
:class:`ConfigError` with a reason, or a logged warning), and :func:`resolve`
binds a validated config to a device count and the reservoir: mesh shape,
weak-scaled batch sizes, active loss groups, precision plan, and the ENGD
feasibility plan.

This module is import-safe without jax (stdlib only).

Loss groups are keyed by NAME, never by index. The base tuple is
``("pde", "ic")`` and ``observation`` adds the supervision source: ``cell_states``
adds the ``data`` group (full reference cell states), ``well`` adds the ``well``
group (bottom-hole pressures and phase rates), ``both`` adds both. The
canonical order is ``("pde", "ic", "data", "well")``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path



class ConfigError(ValueError):
    """A component combination that cannot run; ``str(e)`` carries the reason."""


class InputEncoding(str, Enum):
    SPECTRAL = "spectral"


class Architecture(str, Enum):
    DGM = "dgm"


class ResidualDesign(str, Enum):
    SPECTRAL_PDE = "spectral_pde"


class BackpropDesign(str, Enum):
    FEM_NODAL = "fem_nodal"


class StiffnessDesign(str, Enum):
    PERM_WEIGHTED = "perm_weighted"
    GEOMETRIC = "geometric"


class SamplingDesign(str, Enum):
    WINDOW = "window"


class WeightingDesign(str, Enum):
    FIXED = "fixed"


class SpecialOpt(str, Enum):
    NONE = "none"
    ENGD = "engd"


class Parallelism(str, Enum):
    DATA = "data"


class WellModel(str, Enum):
    """The bottom-hole pressure is eliminated in closed form from the pinned control-phase
    rate and the interior source :math:`Q_{i\alpha}(t)` is data (realized or scheduled rates
    spread by the Gaussian nodal partition)."""

    CLOSED_FORM = "closed_form"


class ObservationDesign(str, Enum):
    CELL_STATES = "cell_states"
    WELL = "well"
    BOTH = "both"
    # No observed supervision at all: the run is closed by its PDE, boundary and
    # anchor terms alone. This is what a forecast window uses -- on the held-out
    # half both cell states and well observations are the answer being predicted.
    NONE = "none"


# Default per-group loss weights, keyed by name (BC starts silenced, as today).
DEFAULT_GROUP_WEIGHTS = {"pde": 1.0, "ic": 1.0, "data": 1.0, "well": 1.0}


@dataclass(frozen=True)
class EngdSettings:
    r"""
    Energy-natural-gradient solver settings.

    The ENGD direction :math:`\psi` solves the damped Gauss–Newton system built
    from the weighted residual rows :math:`\hat r` (see :mod:`pinnlab.engd`):

    .. math::

        \left(\hat J^\top \hat J + \varepsilon I\right)\psi
        \;=\;
        \nabla_\theta \mathcal{L},
        \qquad
        \varepsilon \;=\;
        \varepsilon_{\mathrm{rel}}\,\frac{\operatorname{tr} G}{\dim G}
        + \varepsilon_{\mathrm{abs}}

    where:
    - :math:`\hat J = \partial \hat r / \partial\theta \in \mathbb{R}^{N\times P}`: the Jacobian of the weighted residual rows.
    - :math:`G = \hat J^\top \hat J`: the energy Gramian (Gauss–Newton metric).
    - :math:`\varepsilon`: jit-safe relative Tikhonov damping replacing the reference implementation's eager NaN guard.
    - :math:`\psi`: the natural-gradient direction, applied as :math:`\theta \leftarrow \theta - \eta^\ast\,\psi` with :math:`\eta^\ast` from a geometric grid line search.

    ``damping_mode`` selects *where* that :math:`\varepsilon` is added. The default
    ``"tikhonov"`` adds it isotropically, :math:`G + \varepsilon I`, which floors every
    eigendirection at the same absolute level regardless of the parameter's own scale.
    ``"marquardt"`` instead damps in proportion to each direction's own curvature,

    .. math::

        \bigl(G + \varepsilon\,\operatorname{diag} G\bigr)\,\psi
        \;=\;
        \nabla_\theta \mathcal{L},

    which :func:`pinnlab.engd.solve_dense` realizes as a Jacobi (van der Sluis) rescaling
    of the solve: with :math:`D = \operatorname{diag} G` the system is solved as
    :math:`\tilde G = D^{-1/2} G D^{-1/2}` — unit diagonal, near-optimal condition number
    for an SPD matrix — and the direction rescaled back by :math:`D^{-1/2}`. Because
    :math:`\operatorname{tr}\tilde G / \dim \tilde G = 1` exactly, under ``"marquardt"``
    the relative damping is *literally* :math:`\varepsilon = \varepsilon_{\mathrm{rel}} +
    \varepsilon_{\mathrm{abs}}` and is scale-free by construction. This is the damping the
    reference Gauss–Newton implementations use (``K + \mu\,\mathrm{diag}(K)``).

    **Choosing the damping.** :math:`\varepsilon_{\mathrm{rel}} = 10^{-4}`,
    :math:`\varepsilon_{\mathrm{abs}} = 0` is a measured default, not a nominal one, and
    it is worth several decades of loss. Sweeps over eight decades in both solve regimes
    (SPE1CASE1, DGM):

    - **Dense**, :math:`N \gg P`, full batch: unimodal optimum at :math:`10^{-4}`. At :math:`P = 572` it reaches :math:`1.9\times10^{-5}` against :math:`8.6\times10^{-5}` for the former :math:`(10^{-12}, 10^{-8})` default; at :math:`P = 11{,}092` the separation is :math:`473\times` (:math:`1.8\times10^{-5}` vs :math:`8.6\times10^{-3}`).
    - **Row space**, :math:`N < P`, per-iteration resampling: flat from :math:`10^{-4}` to :math:`3\times10^{-3}` (all within the run-to-run noise), rising sharply outside it. Below :math:`10^{-6}` the solve destabilizes rather than merely degrading — the failure is the *conditioning* role of :math:`\varepsilon`, not the fit-limiting one, and stochastic resampling does nothing for it.

    :math:`10^{-4}` is therefore the optimum in the dense regime and inside the plateau in
    the row-space one. Departures are legitimate — :math:`10^{-3}` is a defensible
    mid-plateau choice for row-space runs — but the former default is not: it is several
    decades below every measured optimum and unstable at :math:`N < P`.

    ``track_deff`` logs the **effective dimension** of the damped system,

    .. math::

        d_{\mathrm{eff}}
        \;=\;
        \operatorname{tr}\bigl(G (G + \varepsilon I)^{-1}\bigr)
        \;=\;
        \sum_{i=1}^{P} \frac{\lambda_i}{\lambda_i + \varepsilon},

    a smooth count of the eigendirections that survive the damping — i.e. how many of the
    :math:`P` parameter directions the step can actually address. It costs one
    ``eigvalsh`` of the solved matrix per iteration and is off by default.

    That cost is not uniformly negligible. The eigendecomposition is
    :math:`\mathcal{O}(\min(N,P)^3)` with a large constant and parallelizes poorly on GPU,
    against an assembly of :math:`\min(N,P)` AD sweeps. At :math:`P \sim 10^3` it is
    lost in the noise (measured: no change to the 1.2 s/iter step at :math:`P = 572`);
    at :math:`P \sim 10^4` it costs *several times* the assembly it annotates. Treat it
    as a calibration instrument — switch it on to locate the damping, then off.

    where:

    - :math:`D = \operatorname{diag} G`: the Gramian's diagonal, i.e. :math:`D_{ii} = \Vert \hat J_{:,i}\Vert^2`, the squared sensitivity of the residual rows to parameter :math:`i`.
    - :math:`\tilde G = D^{-1/2} G D^{-1/2}`: the Jacobi-equilibrated Gramian, :math:`\operatorname{diag}\tilde G = \mathbf{1}`.
    - :math:`\lambda_i`: the eigenvalues of the matrix actually solved (:math:`G` dense, :math:`\hat J\hat J^\top` row-space — the two share their nonzero spectrum).
    - :math:`d_{\mathrm{eff}} \in [0, P]`: the effective dimension; :math:`d_{\mathrm{eff}} \to P` as :math:`\varepsilon \to 0`, and collapses toward :math:`0` as :math:`\varepsilon` swamps the spectrum.

    ``param_target`` is the capacity budget the ``0`` auto sentinel on
    ``m_width``/``n_blocks`` sizes against (see :func:`resolve_capacity`). It lives here
    because the constraint it encodes is a property of *this* optimizer: under ENGD the
    achievable loss within a fixed iteration budget **rises** with the parameter count
    :math:`P` (framework §8.1), the accepted step :math:`\eta^\ast` collapsing in
    lockstep — :math:`P \lesssim 3\,\mathrm{k}` reaches
    :math:`\mathcal{L}\sim 10^{-3}`–:math:`10^{-2}`, :math:`P \sim 4`–:math:`6\,\mathrm{k}`
    needs roughly twice the iterations, and by :math:`P \gtrsim 10\,\mathrm{k}` the line
    search accepts only :math:`\eta^\ast = 0` and the loss never leaves initialization.
    First-order optimizers show no such penalty, so the default is calibrated for ENGD
    and auto capacity warns when it is applied to any other optimizer.
    """

    mode: str = "auto"                 # "dense" | "rowspace" | "auto"
    solver: str = "chol"               # "chol" (damped) | "lstsq" (SVD pseudo-inverse, reference-faithful)
    damping_mode: str = "tikhonov"     # "tikhonov" (isotropic) | "marquardt" (Jacobi-scaled)
    damping_rel: float = 1e-4          # measured optimum; see the damping note in the docstring
    damping_abs: float = 0.0
    track_deff: bool = False           # log the effective dimension (one eigvalsh per step)
    rcond: float | None = None         # lstsq only; None = jnp.linalg.lstsq default
    ls_base: float = 0.5
    ls_num: int = 31
    ls_include_zero: bool = True
    param_target: int = 3000           # capacity budget for the m_width/n_blocks auto sentinel
    row_chunk: int = 64                # residual rows per vjp seed block during J assembly
    dense_p_max: int = 4096            # hard cap when mode="dense"; in auto mode only a slow-solve warning threshold
    jac_budget_bytes: int = 4 << 30    # cap on the big materialized ENGD operands in f64: J (fwd assembly / rowspace) + the rowspace N x N Gram & factor


class SpectralScan(str, Enum):
    """Eigenmode selection techniques in n_eig=0 auto mode"""

    B_S = "band_scan"
    DEFLATION = "deflation"


@dataclass(frozen=True)
class SpectralSettings:
    r"""
    Eigenbasis retention, mesh admissibility and encoder-adequacy settings
    (flattened with the ``spec_`` prefix).

    **Retention.** ``band_scan`` is the validated default: a lowest-:math:`\lambda`
    candidate band, expanded up to ``band_cap_mult`` :math:`\times\,n_\lambda`
    until the vertical family is reached. ``deflation`` instead solves the
    eigenproblem restricted to :math:`\operatorname{range}(I - \Pi_z)`, whose
    lowest modes *are* the vertical family by construction. The two are not
    equivalent-but-faster: the band scan returns the lowest vertical modes it
    finds *within the cap*, which on a strongly pancaked domain need not be the
    lowest vertical modes at all, since the first vertical overtone sits above
    the first areal one by

    .. math::

        \frac{\lambda_{\mathrm{vert},1}}{\lambda_{\mathrm{areal},1}}
        \;\approx\;
        \frac{k_z}{k_x}\left(\frac{L_x}{H_z}\right)^{2}

    where:
    - :math:`k_z/k_x`: the vertical-to-areal permeability contrast of the stiffness weighting.
    - :math:`L_x, H_z`: the areal extent and the thickness of the domain; the ratio reaches :math:`\sim 10^3` on a sheet a hundred times wider than thick, so the vertical family can sit hundreds of areal modes deep — a depth set by the *geometry*, not by :math:`n_\lambda`, which is exactly what a cap proportional to :math:`n_\lambda` fails to track.

    **Admissibility and adequacy.** A hex mesh whose stiffness graph splits into
    disconnected pieces yields component-localized eigenmodes: a budget drawn by
    global eigenvalue order lands on a few pieces and leaves the rest of the
    domain at constant features, which no amount of training can repair. The
    guards below measure that directly — the *conformity* of the mesh
    (``min_node_reuse``), the share of the domain each component carries
    (``min_comp_vol_frac``), and whether the retained basis actually separates
    the cells it must address (``addressability_min``, ``blind_frac_max``).

    where:
    - ``retention``: which candidate-spectrum strategy builds the vertical block.
    - ``dense_eig_max_n``: node count below which the full spectrum is solved densely (exact vertical search) instead of by an expanding sparse band.
    - ``band_cap_mult``: expansion ceiling of the sparse candidate band, as a multiple of :math:`n_\lambda` (``band_scan`` only).
    - ``upsilon_vertical``: the :math:`\upsilon_m` threshold above which a mode counts as vertical.
    - ``column_tol_ft``: areal quantization [ft] that groups nodes into vertical columns.
    - ``stratify_components``: solve each connected component's sub-spectrum separately and give each its own mode allocation, instead of drawing one globally eigenvalue-ordered band.
    - ``min_comp_vol_frac``: pore-volume share below which a component is dropped from the mode budget (its cells stay in every loss group; they are simply not addressed).
    - ``min_node_reuse``: conformity warning threshold — a conforming structured hex mesh shares interior nodes between 8 cells, so a maximum below this signals a grid whose cells do not actually touch; ``0`` disables the check.
    - ``addressability_min``: the fraction of cells that must carry *distinct* feature vectors, i.e. injectivity of the cell-to-feature map.
    - ``addressability_tol``: feature quantization used by that test, as a fraction of the mean per-mode feature standard deviation.
    - ``blind_frac_max``: the tolerated fraction of cells whose centred feature vector is negligible against the largest.
    - ``n_eig_cap``: ceiling for automatic :math:`n_\lambda` expansion, bounding the network input width :math:`d = n_\lambda + 1`.
    """

    retention: str = SpectralScan.B_S               # B_S | DEFLATION
    dense_eig_max_n: int = 4096
    band_cap_mult: int = 16
    upsilon_vertical: float = 0.5
    column_tol_ft: float = 1e-3
    stratify_components: bool = False
    min_comp_vol_frac: float = 0.005   # prune components below 0.5% pore volume
    min_node_reuse: int = 8            # 0 = disable the conformity warning
    addressability_min: float = 1.0
    addressability_tol: float = 0.01
    blind_frac_max: float = 0.0
    n_eig_cap: int = 63                # dim_in = n_eig + 1 <= 64


@dataclass(frozen=True)
class RunConfig:
    """Every knob of one run, defined exactly once (no env re-reads, no comment toggles)."""

    # --- the case: an Eclipse deck and a work directory for its derived caches ------------
    deck_path: str = ""      # the .DATA file (its stem names the simulator outputs beside it)
    work_dir: str = ""       # holds prep_cache/ and prep_cache/spectral_cache/
    deck_sha: str = ""       # sha1 of the deck bytes: part of the hash, so decks never share

    # --- the 10 pipeline components -------------------------------------------------------
    input_encoding: InputEncoding = InputEncoding.SPECTRAL
    architecture: Architecture = Architecture.DGM
    residual_design: ResidualDesign = ResidualDesign.SPECTRAL_PDE
    backprop_design: BackpropDesign | None = BackpropDesign.FEM_NODAL
    stiffness_design: StiffnessDesign | None = StiffnessDesign.PERM_WEIGHTED
    sampling: SamplingDesign = SamplingDesign.WINDOW
    weighting: WeightingDesign = WeightingDesign.FIXED
    special_opt: SpecialOpt = SpecialOpt.ENGD
    special_opt_after: int = 0         # -1 = pure Adam; 0 = special-only; >0 = handoff iteration
    parallelism: Parallelism = Parallelism.DATA

    # --- network ---------------------------------------------------------------------------
    n_eig: int = 48
    n_eig_z: int = 0                   # guaranteed vertical (υ > 1/2) modes in the retained basis; 0 = auto
    m_width: int = 64                  # DGM width; 0 = auto (sized to engd.param_target)
    n_blocks: int = 10                 # DGM depth; 0 = auto
    dim_out: int = 4                   # (p_o, S_w, S_g, R_so)

    # --- training --------------------------------------------------------------------------
    n_iter: int = 75_000
    n_tslice: int = 4
    tslice_layout: str = "linspace"    # full-batch slice times: "linspace" (ends included) | "midpoint" (bin centres)
    lr0: float = 2.0e-5
    lr_decay_steps: int = 800
    lr_decay: float = 0.96
    seed: int = 0

    # --- per-device supervision bases (weak-scaled by the data mesh dim at resolve()) -------
    n_pde_per_dev: int = 64
    n_ic_per_dev: int = 32
    n_data_per_dev: int = 64
    full_batch_cap: int = 16384        # per-group row cap for the deterministic full-batch window
    adapt_every: int = 1               # random-window refresh cadence (Adam phases; the full-batch window is static)

    # --- fem_nodal fracturing ---------------------------------------------------------------
    fem_t_select: str = "hardest"      # "hardest" | "random"
    n_tcand_fem: int = 16
    fem_chunk: int = 8192              # node rows/device per remat micro-batch (0 = whole shard)

    w_fixed: tuple[float, ...] | None = None   # per-ACTIVE-group weights; None = DEFAULT_GROUP_WEIGHTS

    # --- optimizers -------------------------------------------------------------------------
    engd: EngdSettings = field(default_factory=EngdSettings)

    # --- observation source ---------------------------------------------------------------------
    observation: ObservationDesign = ObservationDesign.CELL_STATES

    # --- well model & controls ---------------------------------------------------------------
    well_model: WellModel = WellModel.CLOSED_FORM
    kr_floor: float = 1e-4             # kr floor on the closed form's control-phase denominator only (flowing rates unfloored)
    coning_boost: float = 0.0          # extra weight on well rows near breakthrough events (0 = off)
    # --- PDE row design ----------------------------------------------------------------------
    pde_node_weight: str = "uniform"   # "uniform" | "volume" | "well_gaussian" (fem_nodal rows)
    well_gauss_boost: float = 4.0      # beta_g: extra weight at the well under well_gaussian
    well_gauss_width: float = 1.0      # varkappa: multiplier of the cell-size mollifier width
    spectral_projection: str = "galerkin"   # "galerkin" | "blocknorm" (spectral_pde / hybrid_pde)
    # --- time channel of the encoding ---------------------------------------------------------
    time_encoding: str = "linear"      # "linear" tau = 2t/T - 1 | "log" tau = 2 ln(1 + t/t_w)/ln(1 + T/t_w) - 1
    time_warp_days: float = 0.0        # log: warp scale t_w [days]; 0 = auto (most even report spacing in tau)
    # --- cell-state supervision of the dissolved gas ---------------------------------------------
    rs_supervision: str = "all"        # "all" | "oil_only": R_so rows only where the reference has S_o > 0
    # --- near-well channel of the encoding ----------------------------------------------------
    well_encoding: str = "none"        # "none" | "logr": append the normalized ln r_w, r_w = distance to the nearest perforation
    well_enc_r0_ft: float = 0.0        # logr: inner cutoff radius [ft]; 0 = half the smallest edge of the perforated cells

    # --- spectral residual ------------------------------------------------------------------
    spectral_mu: str = "one"           # "one" | "inv_one_plus_lambda" | "lambda"
    spec: SpectralSettings = field(default_factory=SpectralSettings)

    # --- precision --------------------------------------------------------------------------
    precision_policy: str = "selective_f64"   # "f32" | "selective_f64"
    opt_f64: bool = True

    # --- physical output-range margins ------------------------------------------------------
    p_pad: float = 100.0
    p_margin: float = 0.15
    rs_step: float = 0.05
    rs_headroom: int = 1



@dataclass(frozen=True)
class Resolved:
    """Device- and case-bound quantities derived from a validated RunConfig."""

    mesh_shape: tuple[int, int]        # (data, model)
    groups: tuple[str, ...]
    dim_in: int
    param_count: int
    batches: dict
    full_batch: bool
    initial_weights: tuple[float, ...]
    prec_ad: str
    prec_fem: str
    prec_opt: str
    engd_plan: "object | None"         # memplan.EngdPlan when special_opt == ENGD
    case_meta: dict
    warnings: tuple[str, ...]

    @property
    def data_dim(self) -> int:
        return self.mesh_shape[0]

    @property
    def model_dim(self) -> int:
        return self.mesh_shape[1]


# ---------------------------------------------------------------------------------------------
# Derived component facts
# ---------------------------------------------------------------------------------------------

def loss_groups(cfg: RunConfig) -> tuple[str, ...]:
    """Active loss-group names, in canonical order, for this config."""
    out = ["pde", "ic"]
    if cfg.observation in (ObservationDesign.CELL_STATES, ObservationDesign.BOTH):
        out.append("data")
    if cfg.observation in (ObservationDesign.WELL, ObservationDesign.BOTH):
        out.append("well")
    return tuple(out)


def needs_eigenbasis(cfg: RunConfig) -> bool:
    """True when the Laplace eigenbasis must be provisioned (always: the spectral
    encoding and the Galerkin residual both consume it)."""
    return True


def needs_fem_static(cfg: RunConfig) -> bool:
    """True when static hex-FEM operators must be provisioned (always, here)."""
    return True


def dim_in_of(cfg: RunConfig) -> int:
    """Network input width: spectral eigenfeatures (+ the near-well channel) + time."""
    return cfg.n_eig + 1 + (1 if cfg.well_encoding == "logr" else 0)


def resolve_capacity(cfg: RunConfig) -> RunConfig:
    r"""
    Fill ``m_width`` / ``n_blocks`` when they are left at the ``0`` auto sentinel.

    Auto mode targets the **parameter budget** ``cfg.engd.param_target``, it does not
    scale capacity up with the problem. Under ENGD the achievable loss within a fixed
    iteration budget *rises* with the parameter count :math:`P` (framework §8.1), so the
    rule keeps :math:`P` inside the fast regime; capacity is not the binding constraint
    anywhere in the observed range — optimization geometry is.

    For the gated DGM head the trainable count is exactly

    .. math::

        P(d, m, L)
        \;=\;
        \underbrace{(d + 1)\,m}_{\text{input layer}}
        \;+\;
        L\,\bigl(4 d m + 4 m^{2} + 4 m\bigr)
        \;+\;
        \underbrace{m f + f}_{\text{output layer}}

    which is quadratic in :math:`m`, so the largest admissible width follows in closed
    form from :math:`P(d, m, L) \le P_{\max}`:

    .. math::

        m^\ast
        \;=\;
        \left\lfloor
        \frac{-B + \sqrt{B^{2} + 16 L \left(P_{\max} - f\right)}}{8 L}
        \right\rfloor,
        \qquad
        B \;=\; (d + 1)(1 + 4L) + f

    where:
    - :math:`d`: the network input width :math:`n_\lambda + 1` (``dim_in_of``), so the rule follows the resolved mode count.
    - :math:`m`: ``m_width``; :math:`L`: ``n_blocks``; :math:`f`: ``dim_out``, the four primaries.
    - :math:`P_{\max}`: ``cfg.engd.param_target``; the four :math:`4\,(\cdot)` terms are the DGM's gate quartet per block.
    - :math:`m^\ast`: clamped to :math:`[4, 256]`; depth defaults to :math:`L = 2`, the shallow end of the regime that converged fastest in §8.1.

    Any nonzero ``m_width``/``n_blocks`` is honored verbatim, so hand-set values (the
    quick-test ``16``/``2``) are never overridden. Idempotent, and safe to re-run after
    :math:`n_\lambda` changes.
    """
    if cfg.m_width > 0 and cfg.n_blocks > 0:
        return cfg
    d, f = dim_in_of(cfg), cfg.dim_out
    n_blocks = cfg.n_blocks if cfg.n_blocks > 0 else _AUTO_N_BLOCKS
    m_width = cfg.m_width
    if m_width <= 0:
        b = (d + 1) * (1 + 4 * n_blocks) + f
        disc = b * b + 16 * n_blocks * max(cfg.engd.param_target - f, 0)
        m_width = int((-b + math.sqrt(disc)) // (8 * n_blocks))
        m_width = max(_AUTO_M_MIN, min(_AUTO_M_MAX, m_width))
    return replace(cfg, m_width=m_width, n_blocks=n_blocks)


def case_label(cfg: "RunConfig") -> str:
    """The deck's stem, for messages (what ``mesh_case.value`` used to be)."""
    return Path(cfg.deck_path).stem or "deck"


def case_paths(cfg: "RunConfig") -> tuple[str, str]:
    """(ECLIPSE deck path, prep-cache directory) for the configured deck."""
    if not cfg.deck_path or not cfg.work_dir:
        raise ConfigError("RunConfig.deck_path and RunConfig.work_dir must both be set")
    return str(cfg.deck_path), str(Path(cfg.work_dir) / "prep_cache")


def spectral_cache_dir(cfg: "RunConfig") -> Path:
    """Where the FEM operators and eigenbasis for this deck are cached: beside
    ``prep_cache`` (``spectral_cache.default_spectral_cache_dir``)."""
    return Path(cfg.work_dir) / "spectral_cache"


def has_spectral_cache(cfg: "RunConfig") -> bool:
    """True when a spectral (operator/eigenbasis) cache already exists on disk."""
    d = spectral_cache_dir(cfg)
    return d.is_dir() and any(d.iterdir())


def well_pack_meta_path(cfg: "RunConfig") -> Path:
    """Location of the well-pack sidecar written by :func:`modules.wells.build_well_pack`."""
    return Path(case_paths(cfg)[1]) / "well_pack_meta.json"


def load_well_pack_meta(cfg: "RunConfig") -> dict | None:
    """Well-pack row counts (``n_wells``, ``n_perf``, ``n_rows``) if the sidecar exists."""
    p = well_pack_meta_path(cfg)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def load_case_meta(cfg: "RunConfig") -> dict:
    """Cheap mesh-size lookup (n_cells, n_nodes, n_times) from the prep-cache metadata JSON."""
    meta_path = Path(case_paths(cfg)[1]) / "reservoir_preprocessing_metadata.json"
    if not meta_path.is_file():
        raise ConfigError(
            f"prep cache metadata missing for {case_label(cfg)} ({meta_path}); "
            "build the prep cache first (modules.prep)."
        )
    summary = json.loads(meta_path.read_text())["summary"]
    return {
        "n_cells": int(summary["n_cells"]),
        "n_nodes": int(summary["n_vertices"]),
        "n_times": int(summary["n_times"]),
    }


# ---------------------------------------------------------------------------------------------
# Validity matrix
# ---------------------------------------------------------------------------------------------

_FEM_T_SELECTS = ("hardest", "random")
_SPECTRAL_MUS = ("one", "inv_one_plus_lambda", "lambda")
_SPECTRAL_RETENTIONS = ("band_scan", "deflation")
_NODE_WEIGHTS = ("uniform", "volume", "well_gaussian")
_SPECTRAL_PROJECTIONS = ("galerkin", "blocknorm")
_TIME_ENCODINGS = ("linear", "log")
_TSLICE_LAYOUTS = ("linspace", "midpoint")
_WELL_ENCODINGS = ("none", "logr")
_RS_SUPERVISIONS = ("all", "oil_only")
_PDE_COMPONENTS_V = 3
# Phase-explicit control modes only: LRAT/RESV do not pin a phase, and the deck path
# splits such a total using the *observed* summary rates -- data a forecast lacks.
# Report-step count below which cell-state supervision is called degenerate (rule 20).
_MIN_REPORT_STEPS = 10
# Auto-capacity: depth is fixed at the shallow end of framework §8.1's fast regime and
# the width is solved for; the clamps keep the DGM usable at extreme dim_in.
_AUTO_N_BLOCKS = 2
_AUTO_M_MIN = 4
_AUTO_M_MAX = 256
_ENGD_MODES = ("auto", "dense", "rowspace")
_ENGD_SOLVERS = ("chol", "lstsq")
_ENGD_DAMPING_MODES = ("tikhonov", "marquardt")
_PRECISION_POLICIES = ("f32", "selective_f64")


def validate(cfg: RunConfig) -> tuple[RunConfig, list[str]]:
    """
    Apply the component validity matrix.

    Returns a (possibly normalized) config and the list of warnings; raises
    :class:`ConfigError` for combinations that cannot run. Normalizations
    (auto capacity, ENGD's float64 masters) are recorded as warnings.
    """
    warnings: list[str] = []

    # -- auto capacity (0 sentinel) before any width/depth-dependent rule --------------------
    if cfg.m_width <= 0 or cfg.n_blocks <= 0:
        if cfg.engd.param_target < 1:
            raise ConfigError(
                f"engd_param_target={cfg.engd.param_target} must be >= 1 for auto capacity")
        from . import memplan          # lazy: avoids a module-level import cycle

        sized = resolve_capacity(cfg)
        warnings.append(
            f"auto capacity: m_width={sized.m_width}, n_blocks={sized.n_blocks} "
            f"({memplan.param_count(sized)} parameters against an engd_param_target of "
            f"{cfg.engd.param_target}; framework 8.1 puts the fast ENGD regime below ~3k)"
        )
        if cfg.special_opt is not SpecialOpt.ENGD:
            warnings.append(
                f"auto capacity sized against engd_param_target={cfg.engd.param_target} "
                f"while special_opt={cfg.special_opt.value}: the budget encodes ENGD's "
                "step-collapse with parameter count (framework 8.1), which first-order "
                "optimizers do not share — set m_width/n_blocks explicitly if that "
                "ceiling is not wanted here"
            )
        cfg = sized

    # -- field-level sanity ------------------------------------------------------------------
    if cfg.precision_policy not in _PRECISION_POLICIES:
        raise ConfigError(f"unknown precision_policy {cfg.precision_policy!r}")
    if cfg.opt_f64 and cfg.precision_policy != "selective_f64":
        raise ConfigError("opt_f64=True requires precision_policy='selective_f64' (x64 enabled)")
    if cfg.fem_t_select not in _FEM_T_SELECTS:
        raise ConfigError(f"unknown fem_t_select {cfg.fem_t_select!r}")
    if cfg.n_tcand_fem < cfg.n_tslice:
        raise ConfigError("n_tcand_fem must be >= n_tslice (candidate pool must cover the window)")
    if cfg.spectral_mu not in _SPECTRAL_MUS:
        raise ConfigError(f"unknown spectral_mu {cfg.spectral_mu!r}")
    if cfg.spec.retention not in _SPECTRAL_RETENTIONS:
        raise ConfigError(
            f"unknown spec.retention {cfg.spec.retention!r}; valid are {_SPECTRAL_RETENTIONS}")
    if not 0.0 <= cfg.spec.min_comp_vol_frac < 1.0:
        raise ConfigError(
            f"spec.min_comp_vol_frac={cfg.spec.min_comp_vol_frac} must lie in [0, 1) "
            "(pore-volume share below which a component is dropped from the mode budget)")
    if not 0.0 < cfg.spec.upsilon_vertical < 1.0:
        raise ConfigError(
            f"spec.upsilon_vertical={cfg.spec.upsilon_vertical} must lie in (0, 1)")
    if not 0.0 <= cfg.spec.addressability_min <= 1.0:
        raise ConfigError(
            f"spec.addressability_min={cfg.spec.addressability_min} must lie in [0, 1]")
    if cfg.spec.addressability_tol <= 0.0:
        raise ConfigError(
            f"spec.addressability_tol={cfg.spec.addressability_tol} must be > 0")
    if not 0.0 <= cfg.spec.blind_frac_max <= 1.0:
        raise ConfigError(
            f"spec.blind_frac_max={cfg.spec.blind_frac_max} must lie in [0, 1]")
    if cfg.spec.n_eig_cap < 1:
        raise ConfigError(f"spec.n_eig_cap={cfg.spec.n_eig_cap} must be >= 1")
    if cfg.engd.mode not in _ENGD_MODES:
        raise ConfigError(f"unknown engd.mode {cfg.engd.mode!r}")
    if cfg.engd.solver not in _ENGD_SOLVERS:
        raise ConfigError(f"unknown engd.solver {cfg.engd.solver!r}")
    if cfg.engd.damping_mode not in _ENGD_DAMPING_MODES:
        raise ConfigError(f"unknown engd.damping_mode {cfg.engd.damping_mode!r}")
    if cfg.special_opt_after < -1:
        raise ConfigError("special_opt_after must be -1 (pure Adam), 0 (special-only) or a handoff iteration")
    if cfg.pde_node_weight not in _NODE_WEIGHTS:
        raise ConfigError(f"unknown pde_node_weight {cfg.pde_node_weight!r}; valid are {_NODE_WEIGHTS}")
    if cfg.spectral_projection not in _SPECTRAL_PROJECTIONS:
        raise ConfigError(f"unknown spectral_projection {cfg.spectral_projection!r}; "
                          f"valid are {_SPECTRAL_PROJECTIONS}")
    if cfg.time_encoding not in _TIME_ENCODINGS:
        raise ConfigError(f"unknown time_encoding {cfg.time_encoding!r}; valid are {_TIME_ENCODINGS}")
    if cfg.time_warp_days < 0:
        raise ConfigError("time_warp_days must be >= 0 (0 = auto from the report-step spacing)")
    if cfg.tslice_layout not in _TSLICE_LAYOUTS:
        raise ConfigError(f"unknown tslice_layout {cfg.tslice_layout!r}; valid are {_TSLICE_LAYOUTS}")
    if cfg.rs_supervision not in _RS_SUPERVISIONS:
        raise ConfigError(f"unknown rs_supervision {cfg.rs_supervision!r}; valid are {_RS_SUPERVISIONS}")
    if cfg.well_encoding not in _WELL_ENCODINGS:
        raise ConfigError(f"unknown well_encoding {cfg.well_encoding!r}; valid are {_WELL_ENCODINGS}")
    if cfg.well_enc_r0_ft < 0:
        raise ConfigError("well_enc_r0_ft must be >= 0 (0 = half the smallest perforated-cell edge)")
    if cfg.well_gauss_boost < 0 or cfg.well_gauss_width <= 0:
        raise ConfigError("well_gauss_boost must be >= 0 and well_gauss_width > 0")
    if cfg.n_eig_z < 0:
        raise ConfigError("n_eig_z must be >= 0 (0 = auto vertical-mode reservation)")
    if cfg.n_eig < 0:
        raise ConfigError("n_eig must be >= 0 (0 = auto, resolved to the "
                          "addressability floor of the case at provisioning)")
    # n_eig == 0 defers the width to spectral.resolve_n_eig, so the vertical reservation
    # cannot be checked against it yet; resolve_n_eig re-derives it per trial width.
    if cfg.n_eig > 0 and cfg.n_eig_z >= cfg.n_eig:
        raise ConfigError(
            f"n_eig_z={cfg.n_eig_z} must be < n_eig={cfg.n_eig} "
            "(the vertical block is a reserved subset of the retained basis)"
        )
    if cfg.observation is ObservationDesign.NONE:
        raise ConfigError("observation=none leaves nothing to fit; use cell_states, well or both")
    if cfg.backprop_design is None:
        raise ConfigError("choose backprop_design=fem_nodal for the spectral PDE residual")
    if cfg.stiffness_design is None:
        cfg = replace(cfg, stiffness_design=StiffnessDesign.PERM_WEIGHTED)
        warnings.append("stiffness_design auto-filled to perm_weighted (spectral consumer present)")
    if cfg.time_encoding == "log" and cfg.tslice_layout == "linspace" and cfg.special_opt_after == 0:
        warnings.append(
            "time_encoding='log' with tslice_layout='linspace': the full-batch window puts a "
            "collocation slice at t = 0, where the warped channel's d(tau)/dt is orders of "
            "magnitude above every other slice and the impulsive well start makes the balance "
            "stiff; tslice_layout='midpoint' keeps the slices off t = 0"
        )

    groups = loss_groups(cfg)

    # -- rules 8/9: special_opt vs special_opt_after consistency --------------------------------
    if cfg.special_opt is SpecialOpt.NONE and cfg.special_opt_after != -1:
        raise ConfigError("special_opt=none is pure Adam; set special_opt_after=-1")
    if cfg.special_opt is not SpecialOpt.NONE and cfg.special_opt_after == -1:
        raise ConfigError("special_opt_after=-1 is pure Adam; set special_opt=none")

    # -- ENGD-specific rules ---------------------------------------------------------------------
    if cfg.special_opt is SpecialOpt.ENGD:
        # rule 11: f64 is mandatory for the Gauss-Newton solve
        if cfg.precision_policy != "selective_f64":
            raise ConfigError("ENGD requires f64 (precision_policy='selective_f64')")
        if not cfg.opt_f64:
            cfg = replace(cfg, opt_f64=True)
            warnings.append("opt_f64 forced True for ENGD (float64 master weights)")

    # -- rules 14/15: special-only setup ---------------------------------------------------------
    if cfg.special_opt_after == 0:
        warnings.append(
            "sampling=window upgraded to the deterministic full-batch window "
            "(special_opt_after=0: setup optimized for the special optimizer)"
        )

    # -- rule 16: spectral cache availability -----------------------------------------------------
    if not has_spectral_cache(cfg):
        warnings.append(
            f"no spectral cache on disk for {case_label(cfg)}; the eigensolve will build "
            "on demand (minutes; hours at Norne scale) and be cached"
        )

    # -- rule 19: explicit fixed weights must match the active groups ------------------------------
    if cfg.w_fixed is not None and len(cfg.w_fixed) != len(groups):
        raise ConfigError(
            f"w_fixed has {len(cfg.w_fixed)} entries but the active loss groups are {groups}; "
            "provide one weight per group (or leave w_fixed=None for defaults)"
        )

    # -- rule 20: too few report steps to supervise the data group ---------------------------------
    # Case-agnostic: the count comes from the prep cache. A deck writing restarts yearly
    # (RPTRST BASIC=4) over a schedule with sub-annual well-control changes leaves those
    # changes entirely unsupervised, which is invisible in the loss and easy to mistake
    # for an optimizer failure.
    if "data" in groups:
        try:
            n_times = load_case_meta(cfg)["n_times"]
        except ConfigError:
            n_times = None                       # no prep cache yet; rule 16 already warns
        if n_times is not None and n_times < _MIN_REPORT_STEPS:
            warnings.append(
                f"{case_label(cfg)}'s prep cache holds only {n_times} report step(s); "
                f"cell-state supervision is nearly degenerate below {_MIN_REPORT_STEPS} "
                "(raise the deck's restart frequency, e.g. RPTRST BASIC=2, and rebuild the "
                "prep cache)"
            )

    return cfg, warnings


# ---------------------------------------------------------------------------------------------
# Resolution (device- and case-bound)
# ---------------------------------------------------------------------------------------------

def mesh_shape_of(cfg: RunConfig, device_count: int) -> tuple[int, int]:
    """(data, model) mesh dims for the parallelism component on `device_count` devices."""
    return (device_count, 1)


def initial_group_weights(cfg: RunConfig, groups: tuple[str, ...]) -> tuple[float, ...]:
    """Per-group initial (or fixed) loss weights, by group name."""
    if cfg.w_fixed is not None:
        return tuple(float(w) for w in cfg.w_fixed)
    return tuple(DEFAULT_GROUP_WEIGHTS[g] for g in groups)


def resolve(cfg: RunConfig, device_count: int) -> Resolved:
    """
    Bind a validated config to a device count and its reservoir case.

    Weak scaling: every per-device supervision base is multiplied by the data
    mesh dim, so each accelerator's share stays constant as the mesh grows.
    """
    from . import memplan  # lazy: avoids a module-level import cycle

    cfg, warnings = validate(cfg)
    groups = loss_groups(cfg)
    mesh_shape = mesh_shape_of(cfg, device_count)
    data_dim = mesh_shape[0]
    case_meta = load_case_meta(cfg)

    # "well" is a deterministic whole-set group (its row count comes from the observation
    # pack, not from per-device sampling), so its base is 0.
    per_dev = {"pde": cfg.n_pde_per_dev, "ic": cfg.n_ic_per_dev,
               "data": cfg.n_data_per_dev, "well": 0}
    batches = {g: per_dev[g] * data_dim for g in groups}

    full_batch = cfg.special_opt_after == 0 and cfg.sampling is SamplingDesign.WINDOW

    dim_in = dim_in_of(cfg)
    p = memplan.param_count(cfg, dim_in)

    engd_plan = None
    if cfg.special_opt is SpecialOpt.ENGD:
        if "well" in groups and load_well_pack_meta(cfg) is None:
            warnings = warnings + [
                f"no well_pack_meta.json sidecar for {case_label(cfg)} yet (written on the "
                "first pipeline build): the ENGD row plan uses a generous fallback bound of "
                "4 rows x 32 wells x n_times for the well group"
            ]
        engd_plan = memplan.engd_plan(cfg, groups, batches, case_meta, full_batch, p)
        if engd_plan.mode == "rowspace":
            sweeps, kind = ((engd_plan.p, "jvp column") if engd_plan.direction == "fwd"
                            else (engd_plan.n_rows, "vjp row"))
            warnings = warnings + [
                f"ENGD row-space mode with a DGM ({p:,} params): each step runs "
                f"{sweeps:,} {kind} sweeps through the full network in f64 — expect slow "
                "iterations on pre-Ampere GPUs"
            ]
        elif engd_plan.mode == "dense" and p > cfg.engd.dense_p_max:
            warnings = warnings + [
                f"ENGD auto-selected the dense P x P solve with P={p:,} > "
                f"dense_p_max={cfg.engd.dense_p_max:,} (the row-space N x N system would be "
                "larger): the f64 Cholesky may dominate step time on pre-Ampere GPUs"
            ]

    prec = "float64" if cfg.precision_policy == "selective_f64" else "float32"
    prec_opt = "float64" if cfg.opt_f64 else "float32"

    return Resolved(
        mesh_shape=mesh_shape,
        groups=groups,
        dim_in=dim_in,
        param_count=p,
        batches=batches,
        full_batch=full_batch,
        initial_weights=initial_group_weights(cfg, groups),
        prec_ad=prec,
        prec_fem=prec,
        prec_opt=prec_opt,
        engd_plan=engd_plan,
        case_meta=case_meta,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------------------------
# Serialization: flat dict and structural hash
# ---------------------------------------------------------------------------------------------

# Volatile bookkeeping fields that do not define the run's computation.
_HASH_EXCLUDE = {"n_iter", "deck_path", "work_dir"}


def to_flat_dict(cfg: RunConfig) -> dict:
    """Flatten the config to CSV-ready scalars (enums -> values, tuples -> JSON, None -> '')."""
    out: dict = {}

    def _scalar(v):
        if v is None:
            return ""
        if isinstance(v, Enum):
            return v.value
        if isinstance(v, (tuple, list)):
            return json.dumps(list(v))
        return v

    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, EngdSettings):
            for ef in dataclasses.fields(v):
                out[f"engd_{ef.name}"] = _scalar(getattr(v, ef.name))
        elif isinstance(v, SpectralSettings):
            # 'spec_' rather than 'spectral_': the latter would collide with the
            # top-level spectral_mu field in relevance blanking and round-tripping.
            for sf in dataclasses.fields(v):
                out[f"spec_{sf.name}"] = _scalar(getattr(v, sf.name))
        else:
            out[f.name] = _scalar(v)
    return out


def structural_hash(cfg: RunConfig) -> str:
    """12-hex digest of the run-defining fields (paths and n_iter excluded)."""
    flat = {k: v for k, v in to_flat_dict(cfg).items() if k not in _HASH_EXCLUDE}
    canonical = json.dumps(flat, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]
