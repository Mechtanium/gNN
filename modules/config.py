r"""
Component configuration for the PINN-Lab pipeline.

The ten switchable pipeline components (mesh case, input encoding, architecture,
residual design, backprop design, stiffness design, sampling, loss weighting,
special optimizer, parallelism) and every training hyperparameter are declared
exactly once in :class:`RunConfig`. :func:`validate` applies the combinatorial
validity matrix (fail-fast :class:`ConfigError` with a reason, or a logged
warning), and :func:`resolve` binds a validated config to a device count and a
reservoir case: 2-D mesh shape, weak-scaled batch sizes, active loss groups,
precision plan, and the ENGD feasibility plan.

This module is import-safe without jax (stdlib only) so sweep drivers and
tests can enumerate and validate configurations cheaply.

Loss groups are keyed by NAME, never by index. The active group tuple is a
function of the residual/backprop design:

- ``data_only``: ``("data",)`` — the supervised misfit already contains the
  t=0 rows, so no separate IC/BC groups exist.
- ``cartesian_pde + chain_rule``: ``("pde", "ic", "data", "bc")`` — the
  strong-form residual needs an explicit no-flow flux penalty on boundary faces.
- ``cartesian_pde + fem_nodal`` and ``spectral_pde``: ``("pde", "ic", "data")``
  — no-flow is the natural (Neumann) boundary condition of the FEM balance, so
  no BC group exists.

Two orthogonal axes extend the base tuple for history matching:

- ``observation`` swaps the supervision source: ``cell_states`` keeps the
  ``data`` group (full reference cell states — the forward-benchmark setting),
  ``well`` replaces it with the ``well`` group (bottom-hole pressures and phase
  rates only — the field-realistic inverse setting), ``both`` keeps both.
- ``invert`` adds trainable physical parameters (contacts, closure curves,
  spectral log-permeability); when any of its priors carries positive strength
  a ``reg`` group holds the penalty rows.
- ``well_model=predicted`` makes the well's flowing pressure a trainable head and
  the interior source a Peaceman prediction; the schedule controls then enter as
  the ``ctrl`` group, active under **every** observation design that has a PDE
  term (a predicted source with nothing anchoring it would simply shut the well).
- ``ic_design="hard"`` builds the initial condition into the ansatz and drops the
  ``ic`` group.

The canonical group order is ``("pde", "ic", "data", "well", "ctrl", "reg", "bc")``.
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
    CARTESIAN = "cartesian"
    SPECTRAL = "spectral"


class Architecture(str, Enum):
    DGM = "dgm"
    MLP = "mlp"


class ResidualDesign(str, Enum):
    DATA_ONLY = "data_only"
    CARTESIAN_PDE = "cartesian_pde"
    SPECTRAL_PDE = "spectral_pde"
    # pressure channel projected (galerkin | blocknorm), saturation channels node-wise
    HYBRID_PDE = "hybrid_pde"


class BackpropDesign(str, Enum):
    CHAIN_RULE = "chain_rule"
    FEM_NODAL = "fem_nodal"


class StiffnessDesign(str, Enum):
    PERM_WEIGHTED = "perm_weighted"
    GEOMETRIC = "geometric"


class SamplingDesign(str, Enum):
    WINDOW = "window"
    RAR = "rar"


class WeightingDesign(str, Enum):
    NTK = "ntk"
    FIXED = "fixed"
    NONE = "none"


class SpecialOpt(str, Enum):
    NONE = "none"
    LBFGS = "lbfgs"
    ENGD = "engd"


class Parallelism(str, Enum):
    DATA = "data"
    MODEL = "model"
    BOTH = "both"


class WellModel(str, Enum):
    r"""
    How the well enters the objective and the interior balance.

    ``closed_form``: the bottom-hole pressure is eliminated in closed form from the
    pinned control-phase rate and the interior source :math:`Q_{i\alpha}(t)` is
    **data** (realized or scheduled rates spread by the Gaussian nodal partition).

    ``predicted``: the flowing pressure :math:`p_{wf,w}(t)` is a trainable head,
    the Peaceman closure turns it into per-phase rates
    :math:`q_{\alpha,p}(\theta, t)`, and those rates are the interior source,

    .. math::

        Q_{i\alpha}(\theta, t) \;=\; \sum_p \omega_i^{(p)}\, q_{\alpha,p}(\theta, t),
        \qquad
        q_{\alpha,p} \;=\; C_F\,\mathrm{WI}_p\, f_{\alpha,p}\,\bigl(p_{wf,w(p)} + \gamma_{\alpha,p}\Delta z_p - p_{\alpha,p}\bigr)

    where:
    - :math:`\omega_i^{(p)}`: the unchanged volume-weighted Gaussian partition of unity of perforation :math:`p`, so :math:`\sum_i Q_{i\alpha} = \sum_p q_{\alpha,p}` identically.
    - :math:`f_{\alpha,p}`: the phase mobility factor at the perforated cell; :math:`\gamma_{\alpha,p}\Delta z_p` the wellbore hydrostatic correction.
    - the ``well`` group compares :math:`p_{wf}` and :math:`q_\alpha(\theta)` with the observed WBHP and rates on the history window; the ``ctrl`` group compares them with the schedule controls everywhere.
    """

    CLOSED_FORM = "closed_form"
    PREDICTED = "predicted"


class ObservationDesign(str, Enum):
    CELL_STATES = "cell_states"
    WELL = "well"
    BOTH = "both"
    # No observed supervision at all: the run is closed by its PDE, boundary and
    # anchor terms alone. This is what a forecast window uses -- on the held-out
    # half both cell states and well observations are the answer being predicted.
    NONE = "none"


# Invertible physical-parameter targets, in canonical order.
INVERT_TARGETS = ("z_woc", "z_goc", "relperm", "pcap", "logk")

# Trainable physical scalars per target: z_woc is one depth; z_goc is the
# (z_goc, p_b) pair — the bubble point rides with the gas contact through the
# saturation tie p_b_eff = min(p_b, p(z_goc)), and stays identifiable when the
# GOC lies outside the reservoir span (SPE1); relperm is the 10-parameter Corey
# set (S_wc, S_orw, S_gc, S_org, k_rw^max, k_rg^max, k_ro^max, n_w, n_g, n_ow);
# pcap is the 4-parameter Brooks-Corey set (p_e, lambda) per phase pair; logk
# is sized by inv.n_logk at call time.
_PHYS_COUNTS = {"z_woc": 1, "z_goc": 2, "relperm": 10, "pcap": 4}

# Default per-group loss weights, keyed by name (BC starts silenced, as today).
DEFAULT_GROUP_WEIGHTS = {"pde": 1.0, "ic": 1.0, "data": 1.0, "well": 1.0, "ctrl": 1.0,
                         "reg": 1.0, "bc": 0.0}


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


@dataclass(frozen=True)
class InversionSettings:
    r"""
    Physical-parameter inversion settings (flattened with the ``inv_`` prefix).

    Every inverted quantity trains as an unconstrained :math:`O(1)` real mapped
    through a fixed transform in :mod:`pinnlab.inversion`; these settings hold
    the transform anchors and the prior/penalty strengths that populate the
    ``reg`` loss group.

    .. math::

        \mathcal{R}(\theta_m)
        \;=\;
        \beta_{k} \sum_{j} \lambda_j c_j^2
        \;+\;
        \beta_{0} \left\lVert \theta_s - \theta_{s0} \right\rVert^2
        \;+\;
        \beta_{\mathrm{mb}} \, \frac{1}{N_{\mathrm{mb}}} \sum_{i}
        \left( \frac{\Delta N(t_i) - \Delta N_{\mathrm{obs}}(t_i)}{N(0)} \right)^2

    where:
    - :math:`\beta_k` (``beta_logk``): spectral roll-off (Matern-like) prior strength on the log-permeability coefficients :math:`c_j` against the Laplace eigenvalues :math:`\lambda_j`.
    - :math:`\beta_0` (``beta_prior``): Gaussian anchor strength pulling scalar targets :math:`\theta_s` toward their initialization :math:`\theta_{s0}`.
    - :math:`\beta_{\mathrm{mb}}` (``beta_mb``): material-balance strength comparing the FEM in-place change :math:`\Delta N(t_i)` against cumulative observed production :math:`\Delta N_{\mathrm{obs}}(t_i)` at :math:`N_{\mathrm{mb}}` (``n_mb_probes``) strided report times.
    """

    z_woc_init: float | None = None    # initial water-oil contact depth (ft); None = case midpoint
    z_goc_init: float | None = None    # initial gas-oil contact depth (ft); None = case midpoint
    contact_width: float = 25.0        # sigmoid transition half-width of the equilibrium profile (ft)
    n_logk: int = 16                   # spectral log-permeability coefficients (leading eigenmodes)
    beta_logk: float = 1e-2            # roll-off prior strength on logk coefficients
    beta_mb: float = 0.0               # material-balance penalty strength (0 = off)
    beta_prior: float = 0.0            # anchor-to-init strength on scalar targets (0 = off)
    n_mb_probes: int = 8               # report times probed by the material-balance penalty
    logk_recompute_every: int = 0      # perm_weighted basis rebuild cadence in iterations (0 = never)
    coning_boost: float = 0.0          # extra weight on well rows near breakthrough events (0 = off)
    kr_floor: float = 1e-4             # kr floor on the closed form's control-phase denominator only (flowing rates unfloored)


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


class FlowUnit(str, Enum):
    """Unit system a :class:`FlowSchedule` record is written in."""

    FIELD = "FIELD"
    METRIC = "METRIC"


@dataclass(frozen=True)
class FlowSchedule:
    r"""
    One well-control record imposed during forward inference, in the shape of an
    ECLIPSE ``SCHEDULE`` entry.

    A record takes effect at :math:`t_{\mathrm{start}}` and holds until the next
    record for the same well, reproducing the deck's step-function control
    semantics. Records **override** the deck's own schedule on the inference
    window; wells and times left uncovered keep their deck controls.

    The admissible modes are deliberately restricted to the phase-explicit ones:

    .. math::

        \mathrm{mode} \in \{\texttt{ORAT}, \texttt{WRAT}, \texttt{GRAT}, \texttt{BHP}\}

    where:
    - :math:`\texttt{ORAT}, \texttt{WRAT}, \texttt{GRAT}`: rate control on oil, water or gas respectively, so the control phase is known without inspecting observations.
    - :math:`\texttt{BHP}`: bottom-hole-pressure control at ``bhp_limit``.

    ``LRAT``/``RESV`` are excluded on purpose: a liquid or reservoir-volume total
    does not pin a phase, and the deck path splits it using the *observed*
    summary rates (:func:`utils.ReservoirMesh._resolve_rate_phase_split`) — data
    that does not exist on a forecast window.

    where:
    - ``well``: well name; must match a name in the case's well pack.
    - ``t_start``: activation time [days since the first report step].
    - ``mode``: one of the four control modes above.
    - ``rate``: signed control rate, **production negative, injection positive**, ignored under ``BHP`` control.
    - ``bhp_limit``: BHP target under ``BHP`` control, or the limit that may bind under rate control [psia after conversion].
    - ``open_``: ``False`` shuts the well for the record's span.
    - ``units``: unit system of ``rate`` and ``bhp_limit``. ``None`` means "unspecified": the values are taken as FIELD (the system every other cached quantity uses) and a one-time warning is issued. Set :attr:`FlowUnit.FIELD` explicitly to silence it, or :attr:`FlowUnit.METRIC` to have the values converted on ingest.
    """

    well: str
    t_start: float
    mode: str                          # "ORAT" | "WRAT" | "GRAT" | "BHP"
    rate: float = 0.0
    bhp_limit: float = 0.0
    open_: bool = True
    units: FlowUnit | None = None      # None -> FIELD, with a once-per-session warning


@dataclass(frozen=True)
class InferenceSettings:
    r"""
    Forward-inference settings (flattened with the ``inf_`` prefix).

    **Modes.** ``joint`` is the single-stage formulation: one network is optimized
    over the whole horizon :math:`[0, t_{\mathrm{end}}]` with cell-state and well
    observations restricted to the history :math:`[0, t_s]` and the schedule
    controls (``ctrl`` group) everywhere,

    .. math::

        \mathcal{L}
        \;=\;
        w_{\mathrm{pde}}\,\mathcal{L}_{\mathrm{pde}}[0, t_{\mathrm{end}}]
        + w_{\mathrm{ic}}\,\mathcal{L}_{\mathrm{ic}}(0)
        + w_{\mathrm{data}}\,\mathcal{L}_{\mathrm{data}}[0, t_s]
        + w_{\mathrm{well}}\,\mathcal{L}_{\mathrm{well}}[0, t_s]
        + w_{\mathrm{ctrl}}\,\mathcal{L}_{\mathrm{ctrl}}[0, t_{\mathrm{end}}]

    where:
    - :math:`t_s`: the split time; beyond it only the PDE, the controls and the field's own continuity constrain the forecast.
    - no split-time anchor exists: it is one continuous field from the deck's initial condition.

    It requires ``well_model=predicted`` (the schedule reaches the physics through
    the ``ctrl`` rows). ``n_tslice_forecast`` sets how many of the ``n_tslice``
    collocation slices land on the forecast window (``0`` = proportional to its
    length, at least one), so a short forecast window is never left unsampled.

    The two-stage modes remain as the ablation. ``frozen`` evaluates the trained network on the inference window
    without further optimization: the imposed schedule reaches only the well
    closure that turns predicted cell states into observables, so the predicted
    field itself is exactly what training produced. ``constrained`` continues to
    optimize on the inference window under the unsupervised terms alone — PDE
    residual, no-flow boundary, and the schedule entering as the interior source
    :math:`Q(t)` — so the field genuinely responds to the imposed controls. No
    cell-state or well observation enters either mode; reference states are used
    only to score the result.

    The continuation solves a boundary-value problem, which the PDE and boundary
    terms alone do not determine, so it is anchored at the split time:

    .. math::

        \mathcal{L}_{\mathrm{anchor}}
        \;=\;
        w_a\,\frac{1}{n_C}\sum_{c=1}^{n_C}
        \left\lVert
        \frac{u_\theta(x_c, t_s) - u^{\star}(x_c)}{s_u}
        \right\rVert^{2}

    where:
    - :math:`t_s`: the split time, i.e. the last training report step (or :math:`0` when training is skipped).
    - :math:`u^{\star}`: the anchor state — the trained network's own prediction at :math:`t_s`, or the deck's initial condition when ``train_split_n = 0``.
    - :math:`w_a` (``anchor_weight``): the anchor's weight in the objective.
    - :math:`s_u`: the per-primary state scale shared with the ``ic`` group.
    - :math:`n_C`: the number of anchored cells.

    where:
    - ``mode``: ``"frozen"`` | ``"constrained"`` as described above.
    - ``n_iter``: continuation iteration budget (``constrained`` only).
    - ``anchor_weight``: :math:`w_a` above.
    - ``schedules``: the imposed :class:`FlowSchedule` records; empty means the deck's own controls are used unchanged.
    - ``record_every``: stride over the inference window's report steps when recording observables.
    """

    mode: str = "frozen"               # "frozen" | "constrained" | "joint"
    n_iter: int = 2000                 # continuation budget (constrained only)
    anchor_weight: float = 1.0         # split-time anchor strength
    schedules: tuple[FlowSchedule, ...] = ()
    record_every: int = 1              # stride over recorded report steps
    n_tslice_forecast: int = 0         # joint: slices on the forecast window (0 = proportional)


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
    residual_design: ResidualDesign = ResidualDesign.CARTESIAN_PDE
    backprop_design: BackpropDesign | None = BackpropDesign.FEM_NODAL
    stiffness_design: StiffnessDesign | None = StiffnessDesign.PERM_WEIGHTED
    sampling: SamplingDesign = SamplingDesign.RAR
    weighting: WeightingDesign = WeightingDesign.NTK
    special_opt: SpecialOpt = SpecialOpt.LBFGS
    special_opt_after: int = 0         # -1 = pure Adam; 0 = special-only; >0 = handoff iteration
    parallelism: Parallelism = Parallelism.BOTH

    # --- network ---------------------------------------------------------------------------
    n_eig: int = 48
    n_eig_z: int = 0                   # guaranteed vertical (υ > 1/2) modes in the retained basis; 0 = auto
    m_width: int = 64                  # DGM width; 0 = auto (sized to engd.param_target)
    n_blocks: int = 10                 # DGM depth; 0 = auto
    mlp_hidden: tuple[int, ...] = (64, 64, 64)
    dim_out: int = 4                   # (p_o, S_w, S_g, R_so)

    # --- training --------------------------------------------------------------------------
    n_iter: int = 75_000
    n_tslice: int = 4
    tslice_layout: str = "linspace"    # full-batch slice times: "linspace" (ends included) | "midpoint" (bin centres)
    train_split_n: int = -1            # -1 = no split (train on every step); 0 = skip training,
                                       # infer over the whole horizon; n = train on times[:n]
    lr0: float = 2.0e-5
    lr_decay_steps: int = 800
    lr_decay: float = 0.96
    seed: int = 0

    # --- per-device supervision bases (weak-scaled by the data mesh dim at resolve()) -------
    n_pde_per_dev: int = 64
    n_ic_per_dev: int = 32
    n_bc_per_dev: int = 24
    n_data_per_dev: int = 64
    n_ntk_per_dev: int = 4
    full_batch_cap: int = 16384        # per-group row cap for the deterministic full-batch window

    # --- fem_nodal fracturing ---------------------------------------------------------------
    fem_t_select: str = "hardest"      # "hardest" | "random"
    n_tcand_fem: int = 16
    fem_chunk: int = 8192              # node rows/device per remat micro-batch (0 = whole shard)

    # --- RAR --------------------------------------------------------------------------------
    # adapt_every=1 + rar_explore_frac=0.6 is the measured stochastic-ENGD setting: a window
    # redrawn every step cannot be memorized (generalisation gap 376 -> 1.3), and 60% uniform
    # rows keep the Gramian estimate unbiased. Together 38x better than the former (25, 0.15).
    # The full-pool scan is 1.05% of an ENGD step on SPE1CASE1 and 0.46% on SPE9 -- but it is
    # NOT amortized against cheap Adam steps; validate() warns when an Adam phase is present.
    adapt_every: int = 1
    n_tcand: int = 4
    rar_explore_frac: float = 0.6      # fraction sampled uniformly; 1 - this is RAR-selected
    sweep_chunk: int = 128

    # --- NTK weighting ----------------------------------------------------------------------
    ntk_trace: str = "chunk"           # "full" | "chunk" | "hutchinson" | "shrink"
    ntk_chunk: int = 8
    ntk_probes: int = 8
    n_ntk_fem: int = 8
    ntk_ema: float = 1e-2
    ntk_eps: float = 1e-12
    ntk_pin_bc_zero: bool = True       # legacy surgery: BC weight forced to 0 after normalization
    ntk_pde_boost: float = 1.0         # legacy surgery: added to the PDE weight after normalization
    w_fixed: tuple[float, ...] | None = None   # per-ACTIVE-group weights; None = DEFAULT_GROUP_WEIGHTS

    # --- optimizers -------------------------------------------------------------------------
    lbfgs_mem: int = 10
    engd: EngdSettings = field(default_factory=EngdSettings)

    # --- observation source & physical inversion ---------------------------------------------
    observation: ObservationDesign = ObservationDesign.CELL_STATES
    invert: tuple[str, ...] = ()       # subset of INVERT_TARGETS
    well_source: str = "summary"       # "summary" (OPM obs; Peaceman-synthesis fallback) | "synthetic"
    inv: InversionSettings = field(default_factory=InversionSettings)

    # --- well model & controls ---------------------------------------------------------------
    well_model: WellModel = WellModel.CLOSED_FORM
    ctrl_switch: str = "fb"            # "fb" (Fischer-Burmeister) | "hinge" forecast control switch
    # --- PDE row design ----------------------------------------------------------------------
    pde_node_weight: str = "uniform"   # "uniform" | "volume" | "well_gaussian" (fem_nodal rows)
    well_gauss_boost: float = 4.0      # beta_g: extra weight at the well under well_gaussian
    well_gauss_width: float = 1.0      # varkappa: multiplier of the cell-size mollifier width
    spectral_projection: str = "galerkin"   # "galerkin" | "blocknorm" (spectral_pde / hybrid_pde)
    # --- initial condition -------------------------------------------------------------------
    ic_design: str = "penalty"         # "penalty" (ic group) | "hard" (built into the ansatz)
    ic_tau_days: float = 0.0           # hard IC ramp scale [days]; 0 = first report interval
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

    # --- forward inference on the held-out time window ---------------------------------------
    inference: InferenceSettings = field(default_factory=InferenceSettings)

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
    ntk_batches: dict
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

def reg_active(cfg: RunConfig) -> bool:
    """True when inversion is on and at least one applicable prior strength is positive."""
    if not cfg.invert:
        return False
    betas = [cfg.inv.beta_mb, cfg.inv.beta_prior]
    if "logk" in cfg.invert:
        betas.append(cfg.inv.beta_logk)
    return any(b > 0 for b in betas)


def phys_param_count(cfg: RunConfig) -> int:
    """Trainable physical-parameter count added by the ``invert`` targets."""
    return sum(cfg.inv.n_logk if t == "logk" else _PHYS_COUNTS[t] for t in cfg.invert)


def has_pde(cfg: RunConfig) -> bool:
    """True when a PDE residual exists (every design but ``data_only``)."""
    return cfg.residual_design is not ResidualDesign.DATA_ONLY


def predicted_well(cfg: RunConfig) -> bool:
    """True when the well source is the Peaceman prediction of a trainable :math:`p_{wf}` head."""
    return cfg.well_model is WellModel.PREDICTED and has_pde(cfg)


def hard_ic(cfg: RunConfig) -> bool:
    """True when the initial condition is built into the ansatz (no ``ic`` group)."""
    return cfg.ic_design == "hard"


def loss_groups(cfg: RunConfig) -> tuple[str, ...]:
    """Active loss-group names, in canonical order, for this config."""
    if cfg.residual_design is ResidualDesign.DATA_ONLY:
        return ("data", "reg") if reg_active(cfg) else ("data",)
    out = ["pde"] if hard_ic(cfg) else ["pde", "ic"]
    if cfg.observation in (ObservationDesign.CELL_STATES, ObservationDesign.BOTH):
        out.append("data")
    if cfg.observation in (ObservationDesign.WELL, ObservationDesign.BOTH):
        out.append("well")
    if predicted_well(cfg):
        out.append("ctrl")
    if reg_active(cfg):
        out.append("reg")
    if (cfg.residual_design is ResidualDesign.CARTESIAN_PDE
            and cfg.backprop_design is BackpropDesign.CHAIN_RULE):
        out.append("bc")
    return tuple(out)


def needs_eigenbasis(cfg: RunConfig) -> bool:
    """True when the Laplace eigenbasis must be provisioned (encoding or Galerkin residual)."""
    return (cfg.input_encoding is InputEncoding.SPECTRAL
            or cfg.residual_design in (ResidualDesign.SPECTRAL_PDE, ResidualDesign.HYBRID_PDE))


def needs_fem_static(cfg: RunConfig) -> bool:
    """True when static hex-FEM operators must be provisioned."""
    return (cfg.backprop_design is BackpropDesign.FEM_NODAL
            or cfg.residual_design in (ResidualDesign.SPECTRAL_PDE, ResidualDesign.HYBRID_PDE)
            or needs_eigenbasis(cfg))


def dim_in_of(cfg: RunConfig) -> int:
    """Network input width: spectral eigenfeatures (+ the near-well channel) + time, or
    normalized (x, y, z, t)."""
    if cfg.input_encoding is InputEncoding.SPECTRAL:
        return cfg.n_eig + 1 + (1 if cfg.well_encoding == "logr" else 0)
    return 4


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
    if cfg.architecture is not Architecture.DGM:
        return cfg                                  # MLP capacity is mlp_hidden
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
_NTK_TRACES = ("full", "chunk", "hutchinson", "shrink")
_SPECTRAL_MUS = ("one", "inv_one_plus_lambda", "lambda")
_SPECTRAL_RETENTIONS = ("band_scan", "deflation")
_INFERENCE_MODES = ("frozen", "constrained", "joint")
_NODE_WEIGHTS = ("uniform", "volume", "well_gaussian")
_SPECTRAL_PROJECTIONS = ("galerkin", "blocknorm")
_IC_DESIGNS = ("penalty", "hard")
_TIME_ENCODINGS = ("linear", "log")
_TSLICE_LAYOUTS = ("linspace", "midpoint")
_WELL_ENCODINGS = ("none", "logr")
_RS_SUPERVISIONS = ("all", "oil_only")
_CTRL_SWITCHES = ("fb", "hinge")
_PDE_COMPONENTS_V = 3
# Phase-explicit control modes only: LRAT/RESV do not pin a phase, and the deck path
# splits such a total using the *observed* summary rates -- data a forecast lacks.
_FLOW_MODES = ("ORAT", "WRAT", "GRAT", "BHP")
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
_WELL_SOURCES = ("summary", "synthetic")


def validate(cfg: RunConfig) -> tuple[RunConfig, list[str]]:
    """
    Apply the component validity matrix.

    Returns a (possibly normalized) config and the list of warnings; raises
    :class:`ConfigError` for combinations that cannot run. Normalizations
    (auto-filled stiffness design, NTK demotions) are recorded as warnings.
    """
    warnings: list[str] = []

    # -- auto capacity (0 sentinel) before any width/depth-dependent rule --------------------
    if cfg.architecture is Architecture.DGM and (cfg.m_width <= 0 or cfg.n_blocks <= 0):
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
    if cfg.ntk_trace not in _NTK_TRACES:
        raise ConfigError(f"unknown ntk_trace {cfg.ntk_trace!r}")
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
    if cfg.well_source not in _WELL_SOURCES:
        raise ConfigError(f"unknown well_source {cfg.well_source!r}")
    if cfg.inference.mode not in _INFERENCE_MODES:
        raise ConfigError(f"unknown inference.mode {cfg.inference.mode!r}; "
                          f"valid modes are {_INFERENCE_MODES}")
    if cfg.inference.n_tslice_forecast < 0:
        raise ConfigError("inference.n_tslice_forecast must be >= 0 (0 = proportional)")
    if cfg.pde_node_weight not in _NODE_WEIGHTS:
        raise ConfigError(f"unknown pde_node_weight {cfg.pde_node_weight!r}; valid are {_NODE_WEIGHTS}")
    if cfg.spectral_projection not in _SPECTRAL_PROJECTIONS:
        raise ConfigError(f"unknown spectral_projection {cfg.spectral_projection!r}; "
                          f"valid are {_SPECTRAL_PROJECTIONS}")
    if cfg.ic_design not in _IC_DESIGNS:
        raise ConfigError(f"unknown ic_design {cfg.ic_design!r}; valid are {_IC_DESIGNS}")
    if cfg.ic_tau_days < 0:
        raise ConfigError("ic_tau_days must be >= 0 (0 = the first report interval)")
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
    if cfg.well_encoding != "none" and cfg.input_encoding is not InputEncoding.SPECTRAL:
        raise ConfigError(
            f"well_encoding={cfg.well_encoding!r} appends a channel to the spectral eigenfeature "
            "vector; it needs input_encoding=spectral"
        )
    if cfg.ctrl_switch not in _CTRL_SWITCHES:
        raise ConfigError(f"unknown ctrl_switch {cfg.ctrl_switch!r}; valid are {_CTRL_SWITCHES}")
    if cfg.well_gauss_boost < 0 or cfg.well_gauss_width <= 0:
        raise ConfigError("well_gauss_boost must be >= 0 and well_gauss_width > 0")
    if cfg.inference.n_iter < 0:
        raise ConfigError("inference.n_iter must be >= 0")
    if cfg.inference.record_every < 1:
        raise ConfigError("inference.record_every must be >= 1")
    for sched in cfg.inference.schedules:
        if sched.mode not in _FLOW_MODES:
            raise ConfigError(
                f"FlowSchedule for {sched.well!r} uses mode {sched.mode!r}; valid modes are "
                f"{_FLOW_MODES} (LRAT/RESV are excluded: they do not pin a control phase, "
                "and the deck's phase split needs observed rates a forecast does not have)"
            )
        if sched.t_start < 0:
            raise ConfigError(f"FlowSchedule for {sched.well!r} has t_start={sched.t_start} < 0")
        if sched.mode == "BHP" and sched.bhp_limit <= 0:
            raise ConfigError(f"FlowSchedule for {sched.well!r} is BHP-controlled but "
                              f"bhp_limit={sched.bhp_limit} is not positive")
    if cfg.n_eig_z < 0:
        raise ConfigError("n_eig_z must be >= 0 (0 = auto vertical-mode reservation)")
    if cfg.n_eig < 0:
        raise ConfigError("n_eig must be >= 0 (0 = auto, resolved to the "
                          "addressability floor of the case at provisioning)")
    # n_eig == 0 defers the width to spectral.resolve_n_eig, so the vertical reservation
    # cannot be checked against it yet; resolve_n_eig re-derives it per trial width.
    if needs_eigenbasis(cfg) and cfg.n_eig > 0 and cfg.n_eig_z >= cfg.n_eig:
        raise ConfigError(
            f"n_eig_z={cfg.n_eig_z} must be < n_eig={cfg.n_eig} "
            "(the vertical block is a reserved subset of the retained basis)"
        )

    # -- inversion / observation rules ---------------------------------------------------------
    unknown = [t for t in cfg.invert if t not in INVERT_TARGETS]
    if unknown:
        raise ConfigError(f"unknown invert target(s) {unknown}; valid targets are {INVERT_TARGETS}")
    canonical_invert = tuple(t for t in INVERT_TARGETS if t in cfg.invert)
    if canonical_invert != cfg.invert:
        cfg = replace(cfg, invert=canonical_invert)

    if (cfg.observation is not ObservationDesign.CELL_STATES
            and cfg.residual_design is ResidualDesign.DATA_ONLY):
        raise ConfigError(
            "observation=well/both/none needs a PDE residual as interior regularization "
            "(residual_design=data_only supervises cell states only; use observation=cell_states)"
        )
    # -- predicted well, node weights, projection, hard IC, joint mode ---------------------------
    if cfg.well_model is WellModel.PREDICTED and cfg.residual_design is ResidualDesign.DATA_ONLY:
        raise ConfigError(
            "well_model=predicted makes the interior source Q(theta) a Peaceman prediction, "
            "which only a PDE residual consumes; residual_design=data_only has none"
        )
    if (cfg.well_model is WellModel.PREDICTED
            and cfg.residual_design is ResidualDesign.SPECTRAL_PDE
            and cfg.spectral_projection == "galerkin"):
        warnings.append(
            "well_model=predicted with the Galerkin spectral projection: each mode sums nodal "
            "residuals of either sign before squaring, so node-wise balance errors can cancel; "
            "the node-wise balance is residual_design=cartesian_pde + backprop_design=fem_nodal "
            "(or spectral_projection='blocknorm')"
        )
    if cfg.pde_node_weight != "uniform" and cfg.backprop_design is not BackpropDesign.FEM_NODAL:
        raise ConfigError(
            f"pde_node_weight={cfg.pde_node_weight!r} weights nodal residual rows; it needs "
            "backprop_design=fem_nodal"
        )
    if cfg.residual_design is ResidualDesign.HYBRID_PDE and cfg.backprop_design is not BackpropDesign.FEM_NODAL:
        raise ConfigError("residual_design=hybrid_pde splits the nodal residual field by channel; "
                          "it needs backprop_design=fem_nodal")
    if (cfg.spectral_projection != "galerkin"
            and cfg.residual_design not in (ResidualDesign.SPECTRAL_PDE, ResidualDesign.HYBRID_PDE)):
        raise ConfigError(
            f"spectral_projection={cfg.spectral_projection!r} has no projected residual to act on "
            "(needs residual_design=spectral_pde or hybrid_pde)"
        )
    if cfg.ic_design == "hard" and {"z_woc", "z_goc"} & set(cfg.invert):
        raise ConfigError(
            "ic_design='hard' anchors the ansatz on the deck's initial state; contact "
            "inversion (z_woc/z_goc) makes that state parametric, which the hard ansatz "
            "does not carry yet — use ic_design='penalty' with contact inversion"
        )
    if cfg.ic_design == "hard" and cfg.train_split_n > 0 and cfg.inference.mode == "constrained":
        raise ConfigError(
            "ic_design='hard' with inference.mode='constrained': the two-stage continuation "
            "anchors the held-out window through the ic group, which the hard ansatz removes; "
            "use inference.mode='joint' (one field from t=0) or ic_design='penalty'"
        )
    if cfg.ic_design == "hard" and cfg.residual_design is ResidualDesign.DATA_ONLY:
        warnings.append(
            "ic_design='hard' with residual_design=data_only: the t=0 rows of the data group "
            "become tautologies (the ansatz already matches the deck IC there)"
        )
    if cfg.time_encoding == "log" and cfg.tslice_layout == "linspace" and cfg.special_opt_after == 0:
        warnings.append(
            "time_encoding='log' with tslice_layout='linspace': the full-batch window puts a "
            "collocation slice at t = 0, where the warped channel's d(tau)/dt is orders of "
            "magnitude above every other slice and the impulsive well start makes the balance "
            "stiff; tslice_layout='midpoint' keeps the slices off t = 0"
        )
    if cfg.inference.mode == "joint":
        if cfg.well_model is not WellModel.PREDICTED:
            warnings.append(
                "inference.mode='joint' with well_model=closed_form: the forecast window is "
                "driven by a data source built from the schedule controls (observed rates up "
                "to t_s, controls after it) with no ctrl rows, so a BHP-controlled well or a "
                "rate control that would hit its limit contributes no source on the forecast; "
                "well_model=predicted closes that gap"
            )
        if cfg.train_split_n < 0:
            raise ConfigError("inference.mode='joint' needs a split (train_split_n >= 0)")
    if cfg.observation in (ObservationDesign.WELL, ObservationDesign.BOTH) and cfg.train_split_n > 0:
        warnings.append(
            "well observation rows are masked to the history window [0, t_s] (earlier releases "
            "let the well group see every report step regardless of train_split_n)"
        )
    if "logk" in cfg.invert:
        if not needs_eigenbasis(cfg):
            raise ConfigError(
                "invert=('logk',...) parameterizes log-permeability in the Laplace eigenbasis; "
                "it needs input_encoding=spectral or residual_design=spectral_pde"
            )
        if cfg.backprop_design is not BackpropDesign.FEM_NODAL:
            raise ConfigError(
                "invert=('logk',...) enters the residual through the FEM stiffness action; "
                "it needs backprop_design=fem_nodal (the chain_rule route evaluates the "
                "layered perm_of_z profile, which the spectral multiplier cannot reach)"
            )
        if cfg.inv.n_logk > cfg.n_eig:
            raise ConfigError(
                f"inv_n_logk={cfg.inv.n_logk} exceeds the provisioned eigenbasis n_eig={cfg.n_eig}"
            )
        if (cfg.stiffness_design is StiffnessDesign.PERM_WEIGHTED
                and cfg.inv.logk_recompute_every == 0):
            warnings.append(
                "invert=('logk',...) with stiffness_design=perm_weighted: the eigenbasis stays "
                "anchored to the PRIOR permeability (no inv_logk_recompute_every cadence); "
                "stiffness_design=geometric makes the basis independent of the unknown field"
            )
    if cfg.invert and cfg.observation is ObservationDesign.NONE:
        raise ConfigError(
            "invert=... has nothing to fit against under observation=none (no cell-state or "
            "well observations enter the objective); use observation=well or both"
        )
    if cfg.invert and cfg.observation is ObservationDesign.CELL_STATES:
        warnings.append(
            "inversion against full cell-state supervision is an inverse crime "
            "(the observations already contain the answer everywhere); observation=well "
            "is the field-realistic setting"
        )
    # -- rule 1/2: backprop design exists iff there is a PDE term ------------------------------
    if cfg.residual_design is ResidualDesign.DATA_ONLY and cfg.backprop_design is not None:
        raise ConfigError("backprop design requires a PDE term (residual_design=data_only takes backprop_design=None)")
    if cfg.residual_design is not ResidualDesign.DATA_ONLY and cfg.backprop_design is None:
        raise ConfigError("choose backprop_design (chain_rule or fem_nodal) for a PDE residual")

    # -- rule 3/4: stiffness design exists iff there is a spectral consumer --------------------
    if cfg.stiffness_design is not None and not needs_eigenbasis(cfg):
        raise ConfigError(
            "stiffness_design has no spectral consumer "
            "(needs input_encoding=spectral or residual_design=spectral_pde); set it to None"
        )
    if cfg.stiffness_design is None and needs_eigenbasis(cfg):
        cfg = replace(cfg, stiffness_design=StiffnessDesign.PERM_WEIGHTED)
        warnings.append("stiffness_design auto-filled to perm_weighted (spectral consumer present)")

    groups = loss_groups(cfg)

    # -- rule 6: NTK with a single group is a no-op --------------------------------------------
    if cfg.weighting is WeightingDesign.NTK and len(groups) == 1:
        cfg = replace(cfg, weighting=WeightingDesign.NONE)
        warnings.append("weighting=ntk is a no-op with a single loss group; demoted to none")

    # -- rule 6b: NTK never rebalances the inversion prior --------------------------------------
    if cfg.weighting is WeightingDesign.NTK and "reg" in groups:
        warnings.append(
            "weighting=ntk pins w_reg to 1 after normalization; prior strength lives in the "
            "inv_beta_* settings, not the group weight"
        )

    # -- rule 5: weighting=none needs a single group -------------------------------------------
    if cfg.weighting is WeightingDesign.NONE and len(groups) > 1:
        raise ConfigError(
            f"weighting=none requires a single loss group; this config has {groups} "
            "(only residual_design=data_only qualifies)"
        )

    # -- rule 7: RAR with data-only ranks by data misfit ----------------------------------------
    if cfg.residual_design is ResidualDesign.DATA_ONLY and cfg.sampling is SamplingDesign.RAR:
        warnings.append("sampling=rar with residual_design=data_only ranks points by data misfit only")

    # -- rules 8/9: special_opt vs special_opt_after consistency --------------------------------
    if cfg.special_opt is SpecialOpt.NONE and cfg.special_opt_after != -1:
        raise ConfigError("special_opt=none is pure Adam; set special_opt_after=-1")
    if cfg.special_opt is not SpecialOpt.NONE and cfg.special_opt_after == -1:
        raise ConfigError("special_opt_after=-1 is pure Adam; set special_opt=none")

    # -- ENGD-specific rules ---------------------------------------------------------------------
    if cfg.special_opt is SpecialOpt.ENGD:
        # rule 10: dense Gramian / row-space solve needs replicated params
        if cfg.parallelism is not Parallelism.DATA:
            raise ConfigError(
                "ENGD needs replicated params (dense Gramian / row-space solve); use parallelism=data"
            )
        # rule 11: f64 is mandatory for the Gauss-Newton solve
        if cfg.precision_policy != "selective_f64":
            raise ConfigError("ENGD requires f64 (precision_policy='selective_f64')")
        if not cfg.opt_f64:
            cfg = replace(cfg, opt_f64=True)
            warnings.append("opt_f64 forced True for ENGD (float64 master weights)")
        # rule 12: the energy metric already conditions the update
        if cfg.weighting is WeightingDesign.NTK:
            if cfg.special_opt_after == 0:
                cfg = replace(cfg, weighting=WeightingDesign.FIXED)
                warnings.append(
                    "weighting=ntk demoted to fixed for pure ENGD (the energy metric already "
                    "conditions the update; weights enter loss AND Gramian)"
                )
            else:
                warnings.append(
                    "weighting=ntk under ENGD: weights adapt during the Adam phase only and are "
                    "frozen per window afterwards, entering loss AND Gramian"
                )

    # -- rules 14/15: special-only setup ---------------------------------------------------------
    if cfg.special_opt_after == 0:
        if cfg.sampling is SamplingDesign.WINDOW:
            warnings.append(
                "sampling=window upgraded to the deterministic full-batch window "
                "(special_opt_after=0: setup optimized for the special optimizer)"
            )
        elif cfg.special_opt is SpecialOpt.LBFGS:
            # ENGD is stateless and SPRING opts out of the reset (Steps.special_reset_on_window),
            # so only L-BFGS actually loses memory here -- and at adapt_every=1, every iteration.
            warnings.append(
                f"special_opt_after=0 with sampling=rar: L-BFGS curvature memory restarts at "
                f"every RAR refresh (adapt_every={cfg.adapt_every}), so it never accumulates; "
                "prefer sampling=window, or special_opt=engd which is stateless"
            )

    # -- rule 15b: the RAR pool scan is only amortized against expensive steps ---------------------
    if cfg.sampling is SamplingDesign.RAR and cfg.adapt_every <= 4 and cfg.special_opt_after != 0:
        warnings.append(
            f"adapt_every={cfg.adapt_every} with an Adam phase (special_opt_after="
            f"{cfg.special_opt_after}): the full-pool RAR scan costs ~1% of an ENGD step but "
            "many times a first-order step, so it will dominate the Adam phase; raise "
            "adapt_every, or set special_opt_after=0"
        )

    # -- rule 16: spectral cache availability -----------------------------------------------------
    if needs_eigenbasis(cfg) and not has_spectral_cache(cfg):
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
    # Case-agnostic: the count comes from the prep cache, not from a hardcoded case, and the
    # rule fires for any composition whose `data` group reads cell states -- not just
    # data_only. A deck writing restarts yearly (RPTRST BASIC=4) over a schedule with
    # sub-annual well-control changes leaves those changes entirely unsupervised, which is
    # invisible in the loss and easy to mistake for an optimizer failure.
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

    # -- rule 21: the train/test time split ---------------------------------------------------
    # Same case-metadata degradation as rule 20: load_case_meta raises when the prep cache is
    # missing, and validate() must stay callable without one (sweep.enumerate_grid validates
    # with no case access), so an unknown n_times downgrades the range check to a warning.
    if cfg.train_split_n < -1:
        raise ConfigError(
            f"train_split_n={cfg.train_split_n} is out of range; use -1 (no split), "
            "0 (skip training, infer over the whole horizon) or a step count in [1, n_times)"
        )
    if cfg.train_split_n >= 0:
        try:
            n_times = load_case_meta(cfg)["n_times"]
        except ConfigError:
            n_times = None                       # no prep cache yet; rule 16 already warns
        if n_times is not None and cfg.train_split_n >= n_times:
            raise ConfigError(
                f"train_split_n={cfg.train_split_n} leaves no inference window: "
                f"{case_label(cfg)} holds {n_times} report step(s), so the split must be "
                f"< {n_times} (use -1 to train on every step)"
            )
        if cfg.train_split_n == 0:
            warnings.append(
                "train_split_n=0 skips training entirely and runs inference over the whole "
                f"horizon in {cfg.inference.mode!r} mode; this is the from-scratch ablation, "
                "not a trained forecast"
            )
        elif cfg.train_split_n < _MIN_REPORT_STEPS:
            warnings.append(
                f"train_split_n={cfg.train_split_n} leaves fewer than {_MIN_REPORT_STEPS} "
                "training report steps; the fitted half is nearly degenerate"
            )

    return cfg, warnings


# ---------------------------------------------------------------------------------------------
# Resolution (device- and case-bound)
# ---------------------------------------------------------------------------------------------

def autoscale_mesh(base: tuple[int, int], n_dev: int) -> tuple[int, int]:
    """Round-robin double the (data, model) ratio until it exactly fills n_dev devices."""
    shape = list(base)
    i = 0
    while math.prod(shape) < n_dev:
        shape[i % len(shape)] *= 2
        i += 1
    if math.prod(shape) != n_dev:
        raise ConfigError(
            f"mesh base {tuple(base)} cannot fill exactly {n_dev} devices by round-robin doubling "
            f"(reached {tuple(shape)}); n_dev must equal prod(base)*2^k"
        )
    return tuple(shape)


def mesh_shape_of(cfg: RunConfig, device_count: int) -> tuple[int, int]:
    """(data, model) mesh dims for the parallelism component on `device_count` devices."""
    if cfg.parallelism is Parallelism.DATA:
        return (device_count, 1)
    if cfg.parallelism is Parallelism.MODEL:
        return (1, device_count)
    return autoscale_mesh((1, 1), device_count)


def initial_group_weights(cfg: RunConfig, groups: tuple[str, ...]) -> tuple[float, ...]:
    """Per-group initial (or fixed) loss weights, by group name."""
    if cfg.w_fixed is not None:
        return tuple(float(w) for w in cfg.w_fixed)
    return tuple(DEFAULT_GROUP_WEIGHTS[g] for g in groups)


def resolve(cfg: RunConfig, device_count: int) -> Resolved:
    """
    Bind a validated config to a device count and its reservoir case.

    Weak scaling: every per-device supervision base is multiplied by the data
    mesh dim, so each accelerator's share stays constant as the mesh grows. The
    NTK probe budget is split across the active groups in proportion to their
    batch bases, each kept a multiple of the data dim.
    """
    from . import memplan  # lazy: avoids a module-level import cycle

    cfg, warnings = validate(cfg)
    groups = loss_groups(cfg)
    mesh_shape = mesh_shape_of(cfg, device_count)
    data_dim = mesh_shape[0]
    case_meta = load_case_meta(cfg)

    if (cfg.special_opt is SpecialOpt.ENGD and cfg.backprop_design is BackpropDesign.FEM_NODAL
            and cfg.residual_design in (ResidualDesign.CARTESIAN_PDE, ResidualDesign.HYBRID_PDE)):
        n_nodes = case_meta["n_nodes"]
        per_slice = (3 * n_nodes if cfg.residual_design is ResidualDesign.CARTESIAN_PDE
                     else cfg.n_eig + 2 * n_nodes)
        n_rows = per_slice * cfg.n_tslice
        p_est = memplan.param_count(cfg, dim_in_of(cfg))
        j_bytes = n_rows * p_est * 8
        if j_bytes > cfg.engd.jac_budget_bytes:
            warnings = warnings + [
                f"node-wise PDE rows under ENGD: {n_rows:,} rows x P={p_est:,} is a "
                f"{j_bytes/1e9:.1f} GB f64 Jacobian, above engd.jac_budget_bytes "
                f"({cfg.engd.jac_budget_bytes/1e9:.1f} GB); the assembly falls back to "
                f"{-(-n_rows // cfg.engd.row_chunk):,} vjp row sweeps per step -- lower n_tslice, "
                "engd.param_target, or use residual_design=hybrid_pde / spectral_projection='blocknorm'"
            ]

    if cfg.backprop_design is BackpropDesign.FEM_NODAL and data_dim > 1:
        uneven = [f"{nm}={case_meta[nm]}" for nm in ("n_nodes", "n_cells")
                  if case_meta[nm] % data_dim]
        if uneven:
            warnings = warnings + [
                f"fem_nodal data axes ({', '.join(uneven)}) do not divide data={data_dim} "
                f"on {case_label(cfg)}: the compiler pads the last shard and the static "
                "FEM operands stay replicated (a few extra MB per device, identical results)"
            ]

    # "well", "ctrl" and "reg" are deterministic whole-set groups (their row counts come from the
    # observation pack / prior sizes, not from per-device sampling), so their bases are 0.
    per_dev = {"pde": cfg.n_pde_per_dev, "ic": cfg.n_ic_per_dev,
               "data": cfg.n_data_per_dev, "bc": cfg.n_bc_per_dev,
               "well": 0, "ctrl": 0, "reg": 0}
    batches = {g: per_dev[g] * data_dim for g in groups}

    active_bases = [per_dev[g] for g in groups]
    total_base = float(sum(active_bases))
    ntk_batches = {
        g: max(1, round(cfg.n_ntk_per_dev * per_dev[g] / total_base)) * data_dim
        for g in groups
    }

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
        if engd_plan.mode == "rowspace" and cfg.architecture is Architecture.DGM:
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
        ntk_batches=ntk_batches,
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
# Serialization: flat dict (CSV) and structural hash (checkpoint keying)
# ---------------------------------------------------------------------------------------------

# Volatile bookkeeping fields that do not define the run's computation.
_HASH_EXCLUDE = {"n_iter", "deck_path", "work_dir"}


def _encode_schedules(schedules) -> str:
    """The ``inf_schedules`` cell: a JSON array of FlowSchedule records ('' when empty)."""
    if not schedules:
        return ""
    return json.dumps([{f.name: (getattr(s, f.name).value
                                 if isinstance(getattr(s, f.name), Enum)
                                 else getattr(s, f.name))
                        for f in dataclasses.fields(s)} for s in schedules])


def _decode_schedules(cell) -> tuple:
    """Inverse of :func:`_encode_schedules`; tolerates an already-decoded sequence."""
    if not cell:
        return ()
    if isinstance(cell, (tuple, list)):
        rows = [r if isinstance(r, dict) else dataclasses.asdict(r) for r in cell]
    else:
        rows = json.loads(cell)
    out = []
    for r in rows:
        r = dict(r)
        u = r.get("units")
        r["units"] = FlowUnit(u) if u else None
        out.append(FlowSchedule(**r))
    return tuple(out)


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
        elif isinstance(v, InversionSettings):
            for nf in dataclasses.fields(v):
                out[f"inv_{nf.name}"] = _scalar(getattr(v, nf.name))
        elif isinstance(v, SpectralSettings):
            # 'spec_' rather than 'spectral_': the latter would collide with the
            # top-level spectral_mu field in relevance blanking and round-tripping.
            for sf in dataclasses.fields(v):
                out[f"spec_{sf.name}"] = _scalar(getattr(v, sf.name))
        elif isinstance(v, InferenceSettings):
            for nf in dataclasses.fields(v):
                nv = getattr(v, nf.name)
                # The schedule list is the one non-scalar leaf in any settings block:
                # it round-trips as a single JSON cell (a dict/set cell is rejected by
                # the runs.csv contract, and a raw dataclass is not JSON-serializable).
                out[f"inf_{nf.name}"] = (_encode_schedules(nv) if nf.name == "schedules"
                                         else _scalar(nv))
        else:
            out[f.name] = _scalar(v)
    return out


# Fields added after the forward-benchmark era, elided from the hash while they
# sit at their defaults so every pre-existing config (and its checkpoints) keeps
# its digest. Populated at module bottom once the default flat dict exists.
_HASH_DEFAULT_ELIDE: dict = {}


def structural_hash(cfg: RunConfig) -> str:
    """12-hex digest of the run-defining fields (bookkeeping and n_iter excluded)."""
    flat = {k: v for k, v in to_flat_dict(cfg).items()
            if k not in _HASH_EXCLUDE
            and not (k in _HASH_DEFAULT_ELIDE and v == _HASH_DEFAULT_ELIDE[k])}
    canonical = json.dumps(flat, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def from_flat_dict(d: dict) -> RunConfig:
    """
    Rebuild a RunConfig from a flat dict (inverse of to_flat_dict; used by run_one/sweep).

    Blank cells (``""``, ``None``, or NaN from a pandas row) fall back to the field's
    dataclass default, so relevance-blanked ``runs.csv`` rows round-trip safely — a
    blanked field is by definition inert for that run. The genuinely optional fields
    (``backprop_design``, ``stiffness_design``, ``w_fixed``, ``engd_rcond``) map blank
    to ``None`` instead, matching their ``to_flat_dict`` encoding.
    """
    kwargs: dict = {}
    engd_kwargs: dict = {}
    inv_kwargs: dict = {}
    spec_kwargs: dict = {}
    inf_kwargs: dict = {}
    fields_by_name = {f.name: f for f in dataclasses.fields(RunConfig)}
    engd_fields = {f.name: f for f in dataclasses.fields(EngdSettings)}
    inv_fields = {f.name: f for f in dataclasses.fields(InversionSettings)}
    spec_fields = {f.name: f for f in dataclasses.fields(SpectralSettings)}
    inf_fields = {f.name: f for f in dataclasses.fields(InferenceSettings)}

    enum_types = {
        "input_encoding": InputEncoding, "architecture": Architecture,
        "residual_design": ResidualDesign, "backprop_design": BackpropDesign,
        "stiffness_design": StiffnessDesign, "sampling": SamplingDesign,
        "weighting": WeightingDesign, "special_opt": SpecialOpt, "parallelism": Parallelism,
        "observation": ObservationDesign, "well_model": WellModel,
    }
    tuple_fields = {"mlp_hidden", "w_fixed", "invert"}
    optional_fields = {"backprop_design", "stiffness_design", "w_fixed", "engd_rcond",
                       "inv_z_woc_init", "inv_z_goc_init"}

    def _blank(v):
        return v is None or v == "" or (isinstance(v, float) and math.isnan(v))

    for k, v in d.items():
        if k.startswith("engd_") and k[len("engd_"):] in engd_fields:
            name = k[len("engd_"):]
            if _blank(v):
                if k in optional_fields:
                    engd_kwargs[name] = None
                continue                       # blank cell -> field default
            engd_kwargs[name] = v
            continue
        if k.startswith("inv_") and k[len("inv_"):] in inv_fields:
            name = k[len("inv_"):]
            if _blank(v):
                if k in optional_fields:
                    inv_kwargs[name] = None
                continue                       # blank cell -> field default
            inv_kwargs[name] = v
            continue
        if k.startswith("spec_") and k[len("spec_"):] in spec_fields:
            name = k[len("spec_"):]
            if _blank(v):
                continue                       # blank cell -> field default
            spec_kwargs[name] = v
            continue
        if k.startswith("inf_") and k[len("inf_"):] in inf_fields:
            name = k[len("inf_"):]
            if _blank(v):
                continue                       # blank cell -> field default
            # decoded here, ahead of _coerce, which only understands scalar strings
            inf_kwargs[name] = _decode_schedules(v) if name == "schedules" else v
            continue
        if k not in fields_by_name:
            continue
        if _blank(v):
            if k in optional_fields:
                kwargs[k] = None
            continue                           # blank cell -> field default
        if k in enum_types:
            kwargs[k] = enum_types[k](v)
        elif k in tuple_fields:
            kwargs[k] = tuple(json.loads(v) if isinstance(v, str) else v)
        else:
            kwargs[k] = v

    # coerce numeric strings coming back from CSV/JSON
    def _coerce(fields, kw):
        for name, f in fields.items():
            if name not in kw or kw[name] is None or isinstance(kw[name], (tuple, Enum)):
                continue
            t = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
            v = kw[name]
            if isinstance(v, str) and v != "":
                if "int" in t:
                    kw[name] = int(v)
                elif "float" in t:
                    kw[name] = float(v)
                elif "bool" in t:
                    kw[name] = v in ("True", "true", "1")
    _coerce(fields_by_name, kwargs)
    _coerce(engd_fields, engd_kwargs)
    _coerce(inv_fields, inv_kwargs)
    _coerce(spec_fields, spec_kwargs)
    _coerce(inf_fields, inf_kwargs)

    if engd_kwargs:
        kwargs["engd"] = EngdSettings(**engd_kwargs)
    if inv_kwargs:
        kwargs["inv"] = InversionSettings(**inv_kwargs)
    if spec_kwargs:
        kwargs["spec"] = SpectralSettings(**spec_kwargs)
    if inf_kwargs:
        kwargs["inference"] = InferenceSettings(**inf_kwargs)
    return RunConfig(**kwargs)


def relevant_fields(cfg: RunConfig) -> frozenset:
    """
    The flat-dict keys that actually influence this composition.

    ``runs.csv`` rows blank every configuration column outside this set so a run
    records only the inputs it consumed: ``engd_*`` appears only on ENGD runs,
    ``lbfgs_mem`` only on L-BFGS runs, ``ntk_*`` only under NTK weighting, the
    Adam schedule (``lr0``/``lr_decay*``) only when an Adam phase exists, and so
    on. The gating reuses the same component predicates the pipeline branches on
    (:func:`loss_groups`, :func:`needs_eigenbasis`). :func:`to_flat_dict` and
    :func:`structural_hash` are deliberately unaffected, so config hashes,
    checkpoint keys, and sweep payloads stay stable.
    """
    keys = set(to_flat_dict(cfg))
    groups = loss_groups(cfg)
    drop: set = set()

    if cfg.architecture is Architecture.DGM:
        drop |= {"mlp_hidden"}
    else:
        drop |= {"m_width", "n_blocks"}
    if not needs_eigenbasis(cfg):
        # 'spec_' never catches the top-level spectral_mu, which is gated separately below.
        drop |= {"n_eig", "n_eig_z"} | {k for k in keys if k.startswith("spec_")}
    elif cfg.spec.retention != "band_scan":
        drop |= {"spec_band_cap_mult"}          # band-scan-only expansion ceiling
    if cfg.residual_design not in (ResidualDesign.SPECTRAL_PDE, ResidualDesign.HYBRID_PDE):
        drop |= {"spectral_mu", "spectral_projection"}
    if cfg.residual_design is ResidualDesign.DATA_ONLY:
        drop |= {"well_model", "ctrl_switch"}
    elif cfg.well_model is not WellModel.PREDICTED:
        drop |= {"ctrl_switch"}
    if cfg.pde_node_weight != "well_gaussian":
        drop |= {"well_gauss_boost", "well_gauss_width"}
    if cfg.ic_design != "hard":
        drop |= {"ic_tau_days"}
    if cfg.time_encoding != "log":
        drop |= {"time_warp_days"}
    if not ({"data", "ic"} & set(groups)):
        drop |= {"rs_supervision"}                             # no cell-state rows to mask
    if cfg.input_encoding is not InputEncoding.SPECTRAL:
        drop |= {"well_encoding", "well_enc_r0_ft"}
    elif cfg.well_encoding == "none":
        drop |= {"well_enc_r0_ft"}
    drop |= {f"n_{g}_per_dev" for g in ("pde", "ic", "bc", "data") if g not in groups}
    # inv_kr_floor / inv_coning_boost belong to the well residual, not to any invert target.
    well_keys = {"inv_kr_floor", "inv_coning_boost"}
    if "well" not in groups:
        drop |= {"well_source"} | well_keys
    if not cfg.invert:
        drop |= {k for k in keys if k.startswith("inv_")} - (well_keys if "well" in groups else set())
    else:
        if "logk" not in cfg.invert:
            drop |= {"inv_n_logk", "inv_beta_logk", "inv_logk_recompute_every"}
        if "z_woc" not in cfg.invert:
            drop |= {"inv_z_woc_init"}
        if "z_goc" not in cfg.invert:
            drop |= {"inv_z_goc_init"}
        if not {"z_woc", "z_goc"} & set(cfg.invert):
            drop |= {"inv_contact_width"}
        if cfg.inv.beta_mb <= 0:
            drop |= {"inv_n_mb_probes"}
    if cfg.backprop_design is not BackpropDesign.FEM_NODAL:
        drop |= {"fem_t_select", "n_tcand_fem", "fem_chunk", "pde_node_weight"}
    if cfg.sampling is not SamplingDesign.RAR:
        drop |= {"adapt_every", "n_tcand", "rar_explore_frac", "sweep_chunk"}
    if cfg.weighting is not WeightingDesign.NTK:
        drop |= {"ntk_trace", "ntk_chunk", "ntk_probes", "n_ntk_fem", "ntk_ema",
                 "ntk_eps", "ntk_pin_bc_zero", "ntk_pde_boost", "n_ntk_per_dev"}
    if cfg.weighting is not WeightingDesign.FIXED:
        drop |= {"w_fixed"}
    if cfg.special_opt is not SpecialOpt.LBFGS:
        drop |= {"lbfgs_mem"}
    if cfg.special_opt is not SpecialOpt.ENGD:
        drop |= {k for k in keys if k.startswith("engd_")}
    if cfg.train_split_n < 0:
        drop |= {k for k in keys if k.startswith("inf_")}      # no inference stage
    if cfg.special_opt_after == 0:
        drop |= {"lr0", "lr_decay_steps", "lr_decay"}          # no Adam phase
    if cfg.special_opt is SpecialOpt.NONE or cfg.special_opt_after != 0:
        drop |= {"full_batch_cap", "tslice_layout"}            # no full-batch upgrade
    return frozenset(keys - drop)


# The inversion-era keys and their default flat values: while a config leaves all
# of these at their defaults its structural hash equals the pre-inversion digest,
# so forward-benchmark checkpoints and runs.csv keys stay valid.
_HASH_DEFAULT_ELIDE.update({
    k: v for k, v in to_flat_dict(RunConfig()).items()
    if k in {"observation", "invert", "well_source"} or k.startswith("inv_")
})

# Likewise for the spectral admissibility/retention era: while every ``spec_*``
# field sits at its default the digest equals the pre-SpectralSettings one, so
# the SPE1CASE1 reference run (37c69655c0c0) and its checkpoints stay addressable.
_HASH_DEFAULT_ELIDE.update({
    k: v for k, v in to_flat_dict(RunConfig()).items()
    if k.startswith("spec_") or k == "engd_param_target"
})

# Likewise for the curvature-aware ENGD knobs (Jacobi/Marquardt damping and the
# effective-dimension diagnostic): both default to the pre-existing behaviour, so a
# config that does not opt in keeps its digest and its checkpoints.
_HASH_DEFAULT_ELIDE.update({
    k: v for k, v in to_flat_dict(RunConfig()).items()
    if k in {"engd_damping_mode", "engd_track_deff"}
})

# Likewise for the temporal train/test split era: a config that does not split time
# (``train_split_n == -1``) and leaves every ``inf_*`` field at its default keeps the
# digest it had before forward inference existed, so prior checkpoints under
# output/ckpt/<case>/<hash> and their runs.csv rows stay addressable.
_HASH_DEFAULT_ELIDE.update({
    k: v for k, v in to_flat_dict(RunConfig()).items()
    if k == "train_split_n" or k.startswith("inf_")
})

# Likewise for the predicted-well / node-wise-residual / hard-IC era: every new knob
# defaults to the pre-existing behaviour (closed-form well, uniform nodal rows, Galerkin
# projection, penalty IC), so a config that does not opt in keeps its digest.
_HASH_DEFAULT_ELIDE.update({
    k: v for k, v in to_flat_dict(RunConfig()).items()
    if k in {"well_model", "ctrl_switch", "pde_node_weight", "well_gauss_boost",
             "well_gauss_width", "spectral_projection", "ic_design", "ic_tau_days",
             "inf_n_tslice_forecast"}
})

# Likewise for the time-channel warp and the slice layout: the linear channel and the
# end-inclusive linspace slices are the pre-existing behaviour.
_HASH_DEFAULT_ELIDE.update({
    k: v for k, v in to_flat_dict(RunConfig()).items()
    if k in {"time_encoding", "time_warp_days", "tslice_layout", "well_encoding", "well_enc_r0_ft",
             "rs_supervision"}
})
