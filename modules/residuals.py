r"""
PDE residual operators for the three residual designs × two backprop designs.

All residuals target zero on the trained network. The black-oil component
balance in FIELD units, per component :math:`\alpha \in \{w, o, g\}`, is

.. math::

    \mathcal{R}_\alpha
    \;=\;
    C_{rb}\,\frac{\partial U_\alpha}{\partial t}
    \;-\;
    C_F\,\nabla\!\cdot\!\bigl(T_\alpha \nabla \Phi_\alpha\bigr)
    \;-\;
    q_\alpha(t)

where:
- :math:`U_\alpha = \phi\, S_\alpha / B_\alpha` (plus dissolved gas :math:`\phi\, R_{so} S_o / B_o` in the gas balance): the accumulation, with the **per-cell reference porosity** :math:`\phi_0(\mathbf{x})` inside :math:`\phi`.
- :math:`\Phi_\alpha = p_\alpha - \gamma_\alpha z`: the phase potential with gravity gradient :math:`\gamma_\alpha`.
- :math:`T_\alpha = f_\alpha\, k`: phase mobility times absolute permeability, with the element mobility **potential-upwinded** (below).
- :math:`C_F, C_{rb}`: the FIELD Darcy and rb/ft³ unit constants.
- :math:`q_\alpha(t)`: the **time-resolved well stimulation** — under ``well_model=closed_form`` the realized rates as data (:class:`pinnlab.wells.WellForcing`), under ``predicted`` the Peaceman prediction :math:`q_{\alpha,p}(\theta, t)` of the trainable :math:`p_{wf}` head (:class:`pinnlab.wells.PredictedForcing`); either way conservative nodal loads or a mollified density, with the gas entry the total surface gas so the component bookkeeping is exact.

The exposed residual channels are the IMPES-style recombination of the
component balances,

.. math::

    \mathcal{R}_P = B_w \mathcal{R}_w + (B_o - R_s B_g)\, \mathcal{R}_o + B_g \mathcal{R}_g,
    \qquad
    \mathcal{R}_W = \mathcal{R}_w,
    \qquad
    \mathcal{R}_G = \mathcal{R}_g - R_s\, \mathcal{R}_o

where:
- :math:`\mathcal{R}_P`: the pressure channel — the recombination factors cancel the :math:`\partial_t S_\alpha` terms identically (via :math:`\sum_\alpha S_\alpha = 1`), leaving a total-compressibility pressure equation.
- :math:`\mathcal{R}_W, \mathcal{R}_G`: the water and **free-gas** transport channels (the :math:`R_s`-image subtraction removes the dissolved stream at the surrogate's local :math:`R_s`), so the per-channel scale calibration sees one pressure row and two saturation rows instead of three pressure-dominated rows.

**Potential upwinding.** The element mobility is the smooth donor weighting

.. math::

    \bar f_e^{\mathrm{up}}
    \;=\;
    \sum_{a=1}^{8} \varsigma_a\, f_a,
    \qquad
    \varsigma_a = \operatorname{softmax}_a\bigl(\beta_u\, (\Phi_a - \bar\Phi_e)\bigr)

where:
- :math:`\Phi_a`: the phase potential at element vertex :math:`a`; :math:`\bar\Phi_e` its element mean — high-potential (upstream, donor) vertices dominate the weight.
- :math:`\beta_u`: the sharpness constant [1/psi] — the weighting reduces to the arithmetic element mean as :math:`\beta_u \to 0` and to donor-cell upwinding as :math:`\beta_u \to \infty`; the smooth form keeps the discrete transport operator monotone (no oscillation-tolerant near-null space) while remaining differentiable.

**chain_rule** evaluates the residual per collocation point with nested
autodiff through the encoder (:math:`\nabla\cdot` via ``jacfwd`` of the flux,
:math:`\partial_t` via ``jacrev`` of the accumulation; pointwise evaluation
admits no element upwinding). **fem_nodal** evaluates the control-volume form
at every mesh node with the matrix-free FEM stiffness action and the lumped
storage pairing :math:`V^L_i\,\partial_t U`. **spectral** projects the
recombined residual onto the (de-standardized) eigenmodes with the
**consistent-mass** storage pairing:

.. math::

    \tilde R_{m c}
    \;=\;
    \sqrt{\mu_m}\; v_m^\top \Bigl[\, T(u)\bigl(M\,\partial_t U + K[f]\,\Phi - Q(t)\bigr) \Bigr]_{c}

where:
- :math:`M`: the geometric consistent mass (the exact discrete :math:`L^2` pairing of :math:`v_m` with the storage rate — porosity already lives inside :math:`U`); the nodal form's :math:`V^L` is its row-sum lumping, an :math:`O(h^2)` difference.
- :math:`T(u)`: the pointwise recombination matrix of the channel display above.
- :math:`\mu_m`: the ``spectral_mu`` metric weights.

**Node-wise rows and their weights.** The ``fem_nodal`` rows are squared
individually by the loss, so nothing is summed across nodes before squaring and
:math:`\mathcal{L}_{\mathrm{pde}} \to 0` only when every nodal channel residual does.
Each row carries a node weight,

.. math::

    \hat R_{ic} \;=\; \sqrt{w_i}\; \frac{R_{ic}}{s_c},
    \qquad
    w_i \;=\;
    \begin{cases}
    1 & \texttt{uniform} \\
    V^L_i / \overline{V^L} & \texttt{volume} \\
    1 + \beta_g\, g_i,\quad g_i = \max_p \exp\!\bigl(-\tfrac12 \lVert (\mathbf{x}_i - \mathbf{x}_p)/(\varkappa\,\boldsymbol\sigma_p) \rVert^2\bigr) & \texttt{well\_gaussian}
    \end{cases}

where:
- :math:`V^L_i`: the lumped nodal volume, so ``volume`` makes the row sum a discrete :math:`L^2(\Omega)` norm.
- :math:`g_i`: the peak-normalized Gaussian envelope of the wells (the same widths :math:`\boldsymbol\sigma_p` as the source mollifier, scaled by :math:`\varkappa`), so ``well_gaussian`` enforces the balance hardest where the source lands — an emphasis, not a shape prior on the pressure.
- :math:`\beta_g, \varkappa`: ``well_gauss_boost`` and ``well_gauss_width``.

**Block-norm projection** (``spectral_projection="blocknorm"``). The Galerkin
row :math:`\sum_i \Psi_{im} R_{ic}` sums residuals of either sign inside a mode and
can vanish while the nodal balance does not. The block norm

.. math::

    r_{mc} \;=\; \sqrt{\mu_m \sum_i \lvert \Psi_{im} \rvert\, w_i\, R_{ic}^2 + \epsilon},
    \qquad
    \sum_m r_{mc}^2 \;=\; \sum_i \Bigl(\sum_m \mu_m \lvert\Psi_{im}\rvert\Bigr) w_i R_{ic}^2

is a node-weighted node-wise squared loss compressed to :math:`n_\lambda` rows per
channel: positive by construction, the ENGD contract :math:`\mathcal{L} = \lVert \hat r\rVert^2`
intact, Gauss–Newton rank the same as the Galerkin rows. ``hybrid_pde`` applies the
projection to the pressure channel only and keeps the two saturation channels nodal,
the two blocks balanced so each contributes its own mean square to the group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from math import sqrt as onp_sqrt

from .config import BackpropDesign, ResidualDesign, RunConfig
from .casedata import CaseData
from .meshenv import MeshEnv
from .physics import Primaries
from .spectral import SpectralBundle, raw_centroid_modes, raw_modes

# Smooth potential-upwinding sharpness [1/psi]: strong donor weighting for
# element potential drops beyond a few psi, near-central blending below.
BETA_UPWIND = 1.0


@dataclass
class Extras:
    """Optional inversion/observation bindings threaded through the builders."""

    invm: Any = None                # pinnlab.inversion.Inversion | None
    pack: Any = None                # pinnlab.wells.WellPack | None
    forcing: Any = None             # pinnlab.wells.WellForcing | PredictedForcing | None
    mb_obs: Any = None              # (t_probe, q_cum, n_ref) material-balance observations
    head: Any = None                # pinnlab.wells.WellHead | None (predicted well model)
    forcing_calib: Any = None       # data-driven WellForcing used by calibrate_scales


@dataclass
class ResidualOps:
    """The residual surfaces one config exposes (unused entries are None)."""

    pde_point: Callable | None      # (params, xt, *enc_args) -> (3,)   chain_rule strong form
    pde_fem: Callable | None        # (params, t, prec=None) -> (n_nodes, 3) sharded+remat
    pde_fem_ref: Callable | None    # reference small-mesh form (parity gate)
    pde_fem_parts: Callable | None  # (params, t[, prec]) -> ((b_w, b_o, b_g, rs), dU/dt, div, Q)
    pde_spectral: Callable | None   # (params, t[, cells]) -> (n_eig, 3) Galerkin coefficients
    flux_point: Callable | None     # (params, xt, *enc_args) -> (3, 3) phase flux rows (BC term)
    primaries_nodes: Callable | None
    nodes_P_dPdt: Callable | None
    well_arr: Callable | None = None      # (params) -> scaled well-observable residual rows
    well_predict: Callable | None = None  # (params) -> (p_bh_hat, q_hat) diagnostics surface
    reg_arr: Callable | None = None       # (params) -> inversion penalty rows
    ctrl_arr: Callable | None = None      # (params) -> scaled schedule-control rows (predicted well)
    ctrl_parts: Callable | None = None    # (params) -> unscaled ctrl pieces (diagnostics)
    forcing: Any = None                   # the interior forcing the residuals closed over
    node_row_weight: Any = None           # (n_nodes,) w_i of the nodal rows (fem_nodal)
    pde_hybrid: Callable | None = None    # (params, t[, prec]) -> flat block-balanced hybrid rows


@dataclass
class Scales:
    """Residual/misfit normalizations calibrated at random init (cell-24 port)."""

    res_scale: Any                  # (3,) per-channel (P, W, G) PDE residual std
    bc_scale: Any | None            # (3,) per-phase boundary flux std (chain_rule BC group only)
    state_scale: Any                # (4,) primaries span
    well_scale: Any | None = None   # (s_bhp, s_qo, s_qw, s_qg) observation spans (well group)


def make_residuals(cfg: RunConfig, case: CaseData, spec: SpectralBundle | None,
                   prim: Primaries, env: MeshEnv, prec_ad, prec_fem,
                   extras: Extras | None = None) -> ResidualOps:
    """Build the residual operators the configured design needs.

    ``extras`` injects the inversion bindings (θ_m-aware black-oil tables, the
    spectral log-permeability stiffness, well/regularization row builders) and
    the well pack/forcing; when a PDE residual exists and no forcing was
    provided, the pack and forcing are self-provisioned from the case so every
    entry point drives the interior balances with the realized rates.
    """
    import jax
    import jax.numpy as jnp
    from jax import jacfwd, jacrev
    from jax.sharding import PartitionSpec as P

    import modules.utils.hex_fem_assembly_jax as hf
    from modules.utils.blackoil_closures import (cap_pres_go, cap_pres_ow, fvf_gas, fvf_oil, fvf_water,
                                   phase_factors)

    tables = case.tables
    invm = extras.invm if extras is not None else None
    eff = invm.eff_tables if invm is not None else (lambda params: tables)
    k_of = invm.k_of if invm is not None else None
    phys = case.phys
    rock_cr, rock_pref = phys.ROCK_CR, phys.ROCK_PREF
    dens_o, dens_w, dens_g = case.dens_o, case.dens_w, case.dens_g
    grav = case.grav_grad
    c_f, rb_ft3 = case.c_f, case.rb_ft3
    encoder = prim.encoder
    primaries_point = prim.primaries_point

    # --- well forcing: realized rates as data ----------------------------------------------
    from . import wells as wells_mod

    local_centroids = spec.centroids if spec is not None else jnp.asarray(case.centroids, jnp.float32)
    pack = extras.pack if (extras is not None and extras.pack is not None) else None
    head = extras.head if extras is not None else None
    forcing = extras.forcing if extras is not None else None
    forcing_calib = extras.forcing_calib if extras is not None else None
    well_ops = None
    if pack is None:
        pack = wells_mod.build_well_pack(cfg, case)
    if pack is not None:                       # well_predict backs the diagnostics on every PDE run
        wi_mult = invm.k_mult if invm is not None else None
        well_ops = wells_mod.make_well_residual(
            cfg, case, pack, prim, local_centroids, eff_tables=eff, wi_mult_of=wi_mult,
            head=head)
    if True:
        if forcing_calib is None:
            forcing_calib = wells_mod.build_forcing(pack, case)
        if forcing is None:
            forcing = forcing_calib

    def _recombine(Rw, Ro, Rg, b_w, b_o, b_g, rs):
        """(R_w, R_o, R_g) component rows -> (R_P, R_W, R_G) channels (stacked last axis)."""
        r_p = b_w * Rw + (b_o - rs * b_g) * Ro + b_g * Rg
        return jnp.stack([r_p, Rw, Rg - rs * Ro], axis=-1)

    # ---------------- chain_rule strong form ------------------------------------------------
    def _prec(xt, enc_args):
        return (jnp.asarray(xt, prec_ad), tuple(jnp.asarray(a, prec_ad) for a in enc_args))

    def _phi_point(params, xt, *enc_args):
        tab = eff(params)
        p_o, s_w, s_g, rs = primaries_point(params, xt, *enc_args)
        p_w = p_o - cap_pres_ow(s_w, tab)
        p_g = p_o + cap_pres_go(s_g, tab)
        b_o, b_w, b_g = fvf_oil(rs, tab), fvf_water(p_w, tab), fvf_gas(p_g, tab)
        rho_w = dens_w / b_w
        rho_o = (dens_o + rs * dens_g) / b_o
        rho_g = dens_g / b_g
        z = xt[2]
        return jnp.stack([p_w - grav * rho_w * z,
                          p_o - grav * rho_o * z,
                          p_g - grav * rho_g * z])

    def flux_point(params, xt, *enc_args):
        """Phase flux rows (3, 3): F[alpha, d] = T_alpha,d dPhi_alpha/dx_d (promoted tape)."""
        xt, enc_args = _prec(xt, enc_args)
        tab = eff(params)
        gP = jacrev(lambda c: _phi_point(params, c, *enc_args))(xt)[:, :3]
        p_o, s_w, s_g, rs = primaries_point(params, xt, *enc_args)
        p_w = p_o - cap_pres_ow(s_w, tab)
        p_g = p_o + cap_pres_go(s_g, tab)
        f = phase_factors(p_o, p_w, p_g, s_w, s_g, rs, tab)
        kx, ky, kz = case.perm_of_z(xt[2])
        kv = jnp.stack([kx, ky, kz])
        F_w = f[1] * kv * gP[0]
        F_o = f[0] * kv * gP[1]
        F_g = f[2] * kv * gP[2] + rs * f[0] * kv * gP[1]
        return jnp.stack([F_w, F_o, F_g])

    def _accum_point(params, xt, *enc_args):
        tab = eff(params)
        p_o, s_w, s_g, rs = primaries_point(params, xt, *enc_args)
        s_o = 1.0 - s_w - s_g
        p_w = p_o - cap_pres_ow(s_w, tab)
        p_g = p_o + cap_pres_go(s_g, tab)
        phi = case.poro_of_z(xt[2]) * (1.0 + rock_cr * (p_o - rock_pref))
        b_o, b_w, b_g = fvf_oil(rs, tab), fvf_water(p_w, tab), fvf_gas(p_g, tab)
        return jnp.stack([phi * s_w / b_w, phi * s_o / b_o, phi * (s_g / b_g + rs * s_o / b_o)])

    def pde_point(params, xt, *enc_args):
        """chain_rule recombined residuals (R_P, R_W, R_G), per point, target 0."""
        xt, enc_args = _prec(xt, enc_args)
        tab = eff(params)
        JF = jacfwd(lambda c: flux_point(params, c, *enc_args))(xt)        # (3, 3, 4)
        divF = JF[:, 0, 0] + JF[:, 1, 1] + JF[:, 2, 2]
        dUdt = jacrev(lambda c: _accum_point(params, c, *enc_args))(xt)[:, 3]
        p_o, s_w, s_g, rs = primaries_point(params, xt, *enc_args)
        p_w = p_o - cap_pres_ow(s_w, tab)
        p_g = p_o + cap_pres_go(s_g, tab)
        b_o, b_w, b_g = fvf_oil(rs, tab), fvf_water(p_w, tab), fvf_gas(p_g, tab)
        q = forcing.density_at(xt[:3], xt[3], params)                      # (w, o, g)
        R = rb_ft3 * dUdt - c_f * divF - jnp.asarray(q, dUdt.dtype)
        return _recombine(R[0], R[1], R[2], b_w, b_o, b_g, rs)

    needs_fem = spec is not None and cfg.backprop_design is BackpropDesign.FEM_NODAL
    pde_fem = pde_fem_ref = primaries_nodes = nodes_P_dPdt = None
    fem_parts = None

    if needs_fem:
        from .evaluators import make_nodal_evaluator

        static = spec.static
        k_hex_diag = spec.k_hex_diag
        node_z = spec.node_z
        vlump = spec.vlump
        poro_nodes = spec.poro_nodes
        primaries_nodes, nodes_P_dPdt = make_nodal_evaluator(
            cfg, env, encoder, prim.primaries_feat, case.n_nodes)
        w_nodes = forcing.nodal_partition(spec.node_xyz, spec.vft3)   # (n_nodes, n_perf)
        # Geometry integrals of the promoted (prec_fem) element operators, carried out once
        # here rather than inside every stiffness/mass action: K_e(d) = sum_i d_ei K_e^(i)
        # and the geometric M_e are mesh constants (hex_fem_assembly_jax docstrings). The
        # float32 reference path (pde_fem_ref) keeps the per-call quadrature so its
        # historical bit-exactness is untouched.
        k_axes_fem = hf.stiffness_axis_blocks(static, dtype=prec_fem)   # (n_hex, 3, 8, 8)
        m_elem_fem = hf.mass_elem_blocks(static, dtype=prec_fem)        # (n_hex, 8, 8)
        hf.node_incidence(static)          # memoize the node->slot incidence outside jit

        # node row weights w_i (sqrt applied to the rows; the loss squares them)
        if cfg.pde_node_weight == "volume":
            vl = jnp.asarray(vlump, jnp.float32)
            node_w = vl / jnp.mean(vl)
        elif cfg.pde_node_weight == "well_gaussian":
            g = forcing.node_envelope(spec.node_xyz, float(cfg.well_gauss_width))
            node_w = 1.0 + float(cfg.well_gauss_boost) * g
        else:
            node_w = jnp.ones((case.n_nodes,), jnp.float32)
        sqrt_w = jnp.sqrt(node_w)

        def _q_nodes(t, prec, params=None, frc=None):
            """Conservative nodal loads (n_nodes, 3) at time t, residual order (w, o, g)."""
            frc = forcing if frc is None else frc
            return jnp.asarray(w_nodes, prec) @ jnp.asarray(frc.rate_perf_at(t, params), prec)

        def _stiff_mv(f_hex, Phi, prec=None, k_hex=None):
            """Matrix-free per-phase stiffness action K(f) @ Phi from a per-hex mobility.

            ``k_hex`` overrides the static per-hex permeability (the spectral
            log-k inversion); ``None`` keeps the baseline XLA program.
            """
            k = k_hex_diag if k_hex is None else k_hex
            blocks = k_axes_fem if (prec is not None and jnp.dtype(prec) == jnp.dtype(prec_fem)) else None
            return hf.stiffness_matvec(f_hex[:, None] * k, Phi, static, dtype=prec,
                                       axis_blocks=blocks)

        def _mass_mv(dUdt):
            """Consistent-mass action M @ dU/dt, on the precomputed blocks at prec_fem."""
            blocks = m_elem_fem if jnp.dtype(dUdt.dtype) == jnp.dtype(prec_fem) else None
            return hf.mass_matvec(dUdt, static, dtype=dUdt.dtype, elem_blocks=blocks)

        def _upwind_hex(f_n, Phi_n, prec):
            """Smooth potential-upwinded element mobility (donor softmax weighting)."""
            nodes = static["hex_nodes"]
            if jnp.dtype(prec) == jnp.dtype(prec_fem):
                f_loc = hf.local_of_nodes(jnp.asarray(f_n, prec), static)  # (n_hex, 8)
                p_loc = hf.local_of_nodes(jnp.asarray(Phi_n, prec), static)
            else:
                f_loc = jnp.take(jnp.asarray(f_n, prec), nodes, axis=0)    # (n_hex, 8)
                p_loc = jnp.take(jnp.asarray(Phi_n, prec), nodes, axis=0)
            w = jax.nn.softmax(jnp.asarray(BETA_UPWIND, prec)
                               * (p_loc - jnp.mean(p_loc, axis=1, keepdims=True)), axis=1)
            return jnp.sum(w * f_loc, axis=1)

        def _U_nodes(params, t):
            tab = eff(params)
            Pn = primaries_nodes(params, t)
            p_o, s_w, s_g, rs = Pn[:, 0], Pn[:, 1], Pn[:, 2], Pn[:, 3]
            s_o = 1.0 - s_w - s_g
            p_w = p_o - cap_pres_ow(s_w, tab)
            p_g = p_o + cap_pres_go(s_g, tab)
            phi = jnp.asarray(poro_nodes, Pn.dtype) * (1.0 + rock_cr * (p_o - rock_pref))
            b_o, b_w, b_g = fvf_oil(rs, tab), fvf_water(p_w, tab), fvf_gas(p_g, tab)
            return jnp.stack([phi * s_w / b_w, phi * s_o / b_o,
                              phi * (s_g / b_g + rs * s_o / b_o)], axis=1)

        def pde_fem_ref(params, t):
            """Reference fem_nodal residual (n_nodes, 3): the validated single-GPU f32 form."""
            tab = eff(params)
            k_eff = k_of(params) if k_of is not None else None
            Pn = primaries_nodes(params, t)
            p_o, s_w, s_g, rs = Pn[:, 0], Pn[:, 1], Pn[:, 2], Pn[:, 3]
            p_w = p_o - cap_pres_ow(s_w, tab)
            p_g = p_o + cap_pres_go(s_g, tab)
            b_o, b_w, b_g = fvf_oil(rs, tab), fvf_water(p_w, tab), fvf_gas(p_g, tab)
            rho_w = dens_w / b_w
            rho_o = (dens_o + rs * dens_g) / b_o
            rho_g = dens_g / b_g
            nz = jnp.asarray(node_z, jnp.float32)
            Phi_w = p_w - grav * rho_w * nz
            Phi_o = p_o - grav * rho_o * nz
            Phi_g = p_g - grav * rho_g * nz
            f = phase_factors(p_o, p_w, p_g, s_w, s_g, rs, tab)
            fo_h = _upwind_hex(f[0], Phi_o, jnp.float32)
            fw_h = _upwind_hex(f[1], Phi_w, jnp.float32)
            fg_h = _upwind_hex(f[2], Phi_g, jnp.float32)
            fgo_h = _upwind_hex(rs * f[0], Phi_o, jnp.float32)
            div_w = _stiff_mv(fw_h, Phi_w, k_hex=k_eff)
            div_o = _stiff_mv(fo_h, Phi_o, k_hex=k_eff)
            div_g = _stiff_mv(fg_h, Phi_g, k_hex=k_eff) + _stiff_mv(fgo_h, Phi_o, k_hex=k_eff)
            dUdt = jacfwd(lambda tt: _U_nodes(params, tt))(t)
            accum = jnp.asarray(vlump, jnp.float32)[:, None] * dUdt
            Q = _q_nodes(t, jnp.float32, params)
            div = jnp.stack([div_w, div_o, div_g], axis=1)
            R = accum + div - Q
            R = _recombine(R[:, 0], R[:, 1], R[:, 2], b_w, b_o, b_g, rs)
            return R * jnp.asarray(sqrt_w, R.dtype)[:, None]

        def fem_parts(params, t, prec=None, forcing_override=None):
            """Shared fem_nodal core: state factors, storage rate, flux divergence, loads.

            ``forcing_override`` swaps the interior forcing for one evaluation (the
            scale calibration reads the data-driven forcing at :math:`\theta_0`).
            """
            prec = prec_fem if prec is None else prec
            tab = eff(params)
            k_eff = k_of(params) if k_of is not None else None
            if k_eff is not None:
                k_eff = jnp.asarray(k_eff, prec)
            P4, dP4 = nodes_P_dPdt(params, t)
            P4, dP4 = jnp.asarray(P4, prec), jnp.asarray(dP4, prec)

            def _fields(Pn):
                p_o, s_w, s_g, rs = Pn[:, 0], Pn[:, 1], Pn[:, 2], Pn[:, 3]
                s_o = 1.0 - s_w - s_g
                p_w = p_o - cap_pres_ow(s_w, tab)
                p_g = p_o + cap_pres_go(s_g, tab)
                b_o, b_w, b_g = fvf_oil(rs, tab), fvf_water(p_w, tab), fvf_gas(p_g, tab)
                phi = jnp.asarray(poro_nodes, Pn.dtype) * (1.0 + rock_cr * (p_o - rock_pref))
                U = jnp.stack([phi * s_w / b_w, phi * s_o / b_o,
                               phi * (s_g / b_g + rs * s_o / b_o)], axis=1)
                return (p_o, p_w, p_g, s_w, s_g, rs, b_w, b_o, b_g), U

            (_vals, _U), (_, dUdt) = jax.jvp(_fields, (P4,), (dP4,))
            p_o, p_w, p_g, s_w, s_g, rs, b_w, b_o, b_g = _vals
            rho_w = dens_w / b_w
            rho_o = (dens_o + rs * dens_g) / b_o
            rho_g = dens_g / b_g
            nz = jnp.asarray(node_z, prec)
            Phi_w = p_w - grav * rho_w * nz
            Phi_o = p_o - grav * rho_o * nz
            Phi_g = p_g - grav * rho_g * nz
            f = phase_factors(p_o, p_w, p_g, s_w, s_g, rs, tab)
            _repl = lambda a: jax.lax.with_sharding_constraint(jnp.asarray(a, prec), env.repl)
            fo_h = _upwind_hex(_repl(f[0]), _repl(Phi_o), prec)
            fw_h = _upwind_hex(_repl(f[1]), _repl(Phi_w), prec)
            fg_h = _upwind_hex(_repl(f[2]), _repl(Phi_g), prec)
            fgo_h = _upwind_hex(_repl(rs * f[0]), _repl(Phi_o), prec)
            div_w = _stiff_mv(fw_h, _repl(Phi_w), prec, k_hex=k_eff)
            div_o = _stiff_mv(fo_h, _repl(Phi_o), prec, k_hex=k_eff)
            div_g = (_stiff_mv(fg_h, _repl(Phi_g), prec, k_hex=k_eff)
                     + _stiff_mv(fgo_h, _repl(Phi_o), prec, k_hex=k_eff))
            div = jnp.stack([div_w, div_o, div_g], axis=1)
            Q = _q_nodes(t, prec, params, forcing_override)
            return (b_w, b_o, b_g, rs), dUdt, div, Q

        def _nodal_channels(params, t, prec, forcing_override=None):
            """Weighted nodal channels sqrt(w_i) T(u)(V dU/dt + K Phi - Q) — (n_nodes, 3)."""
            (b_w, b_o, b_g, rs), dUdt, div, Q = fem_parts(params, t, prec, forcing_override)
            accum = jnp.asarray(vlump, prec)[:, None] * dUdt
            R = accum + div - Q
            R = _recombine(R[:, 0], R[:, 1], R[:, 2], b_w, b_o, b_g, rs)
            return R * jnp.asarray(sqrt_w, R.dtype)[:, None]

        def pde_fem(params, t, prec=None, forcing_override=None):
            """fem_nodal control-volume channels (n_nodes, 3): sqrt(w_i) T(u)(V dU/dt + K Phi - Q(t))."""
            prec = prec_fem if prec is None else prec
            R = _nodal_channels(params, t, prec, forcing_override)
            if case.n_nodes % env.data_dim:
                # uneven node axis: the hint would device_put eagerly (IndivisibleError);
                # sharding propagates from the evaluator's padded row constraint instead
                return R
            return jax.lax.with_sharding_constraint(R, env.nd(P("data", None)))

    # ---------------- spectral projection (Galerkin | block-norm) and the hybrid ------------
    pde_spectral = pde_hybrid = None
    blocknorm = cfg.spectral_projection == "blocknorm"
    if True:
        sqrt_mu = None
        if spec is not None and spec.mu is not None:
            sqrt_mu = jnp.sqrt(jnp.asarray(spec.mu))

        if cfg.backprop_design is BackpropDesign.FEM_NODAL:
            v_raw = raw_modes(spec)                                      # (n_nodes, n_eig)
            v_abs = jnp.abs(jnp.asarray(v_raw))

            def _project(R):
                """Project (n_nodes, k) channels: Galerkin rows or block-norm rows (n_eig, k)."""
                if blocknorm:
                    mu = jnp.asarray(sqrt_mu, R.dtype)[:, None] ** 2
                    ss = jnp.asarray(v_abs, R.dtype).T @ (jnp.asarray(node_w, R.dtype)[:, None] * R * R)
                    return jnp.sqrt(mu * ss + jnp.asarray(1e-30, R.dtype))
                Rs = jnp.asarray(v_raw, R.dtype).T @ R
                return Rs * jnp.asarray(sqrt_mu, R.dtype)[:, None]

            def _consistent_channels(params, t, forcing_override=None):
                (b_w, b_o, b_g, rs), dUdt, div, Q = fem_parts(params, t, forcing_override=forcing_override)
                storage = _mass_mv(dUdt)                                  # consistent M dU/dt
                R = storage + div - Q
                return _recombine(R[:, 0], R[:, 1], R[:, 2], b_w, b_o, b_g, rs)

            def pde_spectral(params, t, cells=None, forcing_override=None):
                return _project(_consistent_channels(params, t, forcing_override))   # (n_eig, 3)

            if False:   # the hybrid (node-wise saturation) rows are not built here
                n_p_rows = int(cfg.n_eig)
                n_s_rows = 2 * int(case.n_nodes)
                n_tot = n_p_rows + n_s_rows
                bal_p = float(onp_sqrt(n_tot / (2.0 * n_p_rows)))
                bal_s = float(onp_sqrt(n_tot / (2.0 * n_s_rows)))

                def pde_hybrid(params, t, prec=None, forcing_override=None):
                    """Flat block-balanced rows: pressure projected (n_eig), saturations nodal (2 n)."""
                    (b_w, b_o, b_g, rs), dUdt, div, Q = fem_parts(params, t, prec, forcing_override)
                    storage = _mass_mv(dUdt)
                    Rc = storage + div - Q
                    Rc = _recombine(Rc[:, 0], Rc[:, 1], Rc[:, 2], b_w, b_o, b_g, rs)
                    r_p = _project(Rc[:, :1])[:, 0]                                    # (n_eig,)
                    accum = jnp.asarray(vlump, dUdt.dtype)[:, None] * dUdt
                    Rn = accum + div - Q
                    Rn = _recombine(Rn[:, 0], Rn[:, 1], Rn[:, 2], b_w, b_o, b_g, rs)
                    r_s = Rn[:, 1:] * jnp.asarray(sqrt_w, Rn.dtype)[:, None]           # (n, 2)
                    return r_p, r_s

                pde_hybrid.balance = (bal_p, bal_s)

        else:
            v_c_raw = raw_centroid_modes(spec)                           # (n_cells, n_eig)
            vol_c = jnp.sum(spec.static["JxW"], axis=1)                  # (n_cells,) cell volumes
            centroids = spec.centroids
            n_cells = case.n_cells

            def pde_spectral(params, t, cells=None, forcing_override=None):
                ci = jnp.arange(n_cells) if cells is None else cells
                xt = jnp.concatenate([centroids[ci],
                                      jnp.broadcast_to(jnp.reshape(t, (1,)), (ci.shape[0],))[:, None]],
                                     axis=1)
                enc_args = encoder.gather_args(ci)
                R = jax.vmap(lambda x, *a: pde_point(params, x, *a))(xt, *enc_args)  # (|S|, 3)
                w = (vol_c[ci] * (n_cells / ci.shape[0]))[:, None]
                if blocknorm:
                    mu = jnp.asarray(sqrt_mu, R.dtype)[:, None] ** 2
                    ss = jnp.abs(jnp.asarray(v_c_raw, R.dtype)[ci]).T @ (R * R * jnp.asarray(w, R.dtype))
                    return jnp.sqrt(mu * ss + jnp.asarray(1e-30, R.dtype))
                Rs = jnp.asarray(v_c_raw, R.dtype)[ci].T @ (R * jnp.asarray(w, R.dtype))
                return Rs * jnp.asarray(sqrt_mu, R.dtype)[:, None]

    # ---------------- well observables and inversion penalties -------------------------------
    well_arr = well_predict = reg_arr = ctrl_arr = ctrl_parts = None
    if well_ops is not None:
        well_arr, well_predict = well_ops.well_arr, well_ops.well_predict
        ctrl_arr, ctrl_parts = well_ops.ctrl_arr, well_ops.ctrl_parts
    use_point = True
    return ResidualOps(
        pde_point=pde_point if use_point else None,
        pde_fem=pde_fem,
        pde_fem_ref=pde_fem_ref,
        pde_fem_parts=fem_parts,
        pde_spectral=pde_spectral,
        flux_point=flux_point if use_point else None,
        primaries_nodes=primaries_nodes,
        nodes_P_dPdt=nodes_P_dPdt,
        well_arr=well_arr,
        well_predict=well_predict,
        reg_arr=reg_arr,
        ctrl_arr=ctrl_arr,
        ctrl_parts=ctrl_parts,
        forcing=forcing,
        node_row_weight=(node_w if needs_fem else None),
        pde_hybrid=pde_hybrid,
    )


# Oil saturation below which the dissolved-gas ratio of a reference cell state is not a
# measurement (no oil phase to hold the gas): the simulator carries a placeholder there.
RS_OIL_EPS = 1e-3


def state_row_mask(cfg: RunConfig, y_ref):
    r"""
    Per-channel supervision weights for cell-state rows against the reference ``y_ref``
    (``(..., 4)`` primaries :math:`(p_o, S_w, S_g, R_{so})`).

    Under ``rs_supervision="oil_only"`` the :math:`R_{so}` channel is switched off wherever
    the reference carries no oil,

    .. math::

        m_{R_s}(\mathbf{x}, t) \;=\; \mathbb{1}\bigl[\,1 - S_w^{\mathrm{ref}} - S_g^{\mathrm{ref}} > \epsilon_o\,\bigr],
        \qquad
        m_{p} = m_{S_w} = m_{S_g} = 1

    where:
    - :math:`R_{so}`: the dissolved gas-oil ratio, a property *of the oil phase* — in a cell without oil (the water leg below the contact, :math:`S_w = 1`) the simulator's restart value is a bookkeeping placeholder, not a state: SPE2's water leg reports :math:`R_s = 1.39` at :math:`t = 0` and :math:`1.41`–:math:`1.42` from the first report step on, a discontinuity in time that no smooth ansatz can honour and that otherwise floors the ``data`` and ``ic`` losses.
    - :math:`\epsilon_o` (``RS_OIL_EPS``): the oil-saturation threshold below which the channel is masked.

    ``"all"`` (the default) returns unit weights, i.e. the historical objective. Returns an
    array broadcastable against the residual, ``(..., 4)``.
    """
    import jax.numpy as jnp

    y = jnp.asarray(y_ref)
    ones = jnp.ones(y.shape[:-1] + (1,), y.dtype)
    if cfg.rs_supervision != "oil_only":
        return jnp.ones((4,), y.dtype)
    has_oil = (1.0 - y[..., 1] - y[..., 2] > RS_OIL_EPS).astype(y.dtype)[..., None]
    return jnp.concatenate([ones, ones, ones, has_oil], axis=-1)


def calibrate_scales(cfg: RunConfig, groups: tuple[str, ...], ops: ResidualOps,
                     prim: Primaries, case: CaseData, spec: SpectralBundle | None,
                     params0, seed: int = 7, extras: Extras | None = None) -> Scales:
    """Set the residual/flux normalizations from a random-init batch (cell-24 port).

    The PDE channels are the recombined (pressure, water, free-gas) rows, so
    the per-channel standard deviations normalize one pressure equation and two
    transport equations — the saturation misfit is visible to the calibration.
    """
    import jax
    import jax.numpy as jnp
    from jax import random, vmap

    encoder = prim.encoder
    centroids = spec.centroids if spec is not None else jnp.asarray(case.centroids, jnp.float32)

    def _xt_of(ci, t):
        return jnp.concatenate([centroids[ci], jnp.reshape(t, (1,))])

    # deferred: sampling imports this module, so a module-level import would cycle
    from . import sampling as sampling_mod

    # Calibrate against the DATA forcing even under the predicted well model: at the
    # untrained network the predicted rates are ~0 (immobile control phase), so a scale
    # read off them would omit the source magnitude entirely and the pde loss would
    # explode by orders of magnitude once the well wakes up.
    frc = extras.forcing_calib if extras is not None else None
    res_scale = jnp.ones((3,), jnp.float32)
    if "pde" in groups:
        key = random.PRNGKey(seed)
        if True:
            ts = sampling_mod.draw_times(case, key, (4,))
            Rb = jax.lax.map(lambda t: ops.pde_spectral(params0, t, forcing_override=frc), ts)  # (4, n_eig, 3)
            res_scale = jnp.std(Rb.reshape(-1, 3), axis=0) + 1e-12
        elif cfg.backprop_design is BackpropDesign.FEM_NODAL:
            ts = sampling_mod.draw_times(case, key, (4,))
            Rb = jax.lax.map(lambda t: ops.pde_fem(params0, t, forcing_override=frc), ts)  # (4, n, 3)
            res_scale = jnp.std(Rb.reshape(-1, 3), axis=0) + 1e-12
        else:
            k1, k2 = random.split(key)
            ci = random.randint(k1, (256,), 0, case.n_cells, dtype=jnp.int32)
            t = sampling_mod.draw_times(case, k2, (256,))
            enc_args = encoder.gather_args(ci)
            Rb = vmap(lambda c, tt, *a: ops.pde_point(params0, _xt_of(c, tt), *a))(ci, t, *enc_args)
            res_scale = jnp.std(Rb.reshape(-1, 3), axis=0) + 1e-12

    bc_scale = None
    if "bc" in groups:
        tb = sampling_mod.mid_time(case)
        enc_args = encoder.gather_args(case.bc_cell)
        Fb0 = vmap(lambda c, a, *ea: ops.flux_point(params0, _xt_of(c, tb), *ea)[:, a])(
            case.bc_cell, case.bc_axis, *enc_args)
        bc_scale = jnp.std(Fb0.reshape(-1, 3), axis=0) + 1e-12

    well_scale = None
    if "well" in groups and extras is not None and extras.pack is not None:
        well_scale = extras.pack.well_scale       # observation spans, set at pack build

    return Scales(res_scale=res_scale, bc_scale=bc_scale, state_scale=case.state_scale,
                  well_scale=well_scale)
