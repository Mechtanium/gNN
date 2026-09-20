r"""
Black-oil physics closures around the network: hard-constrained primaries and
per-phase state (port of the notebook's transform cells).

The network output :math:`y \in \mathbb{R}^4` maps to the primaries through
range-anchored sigmoids (see :mod:`pinnlab.casedata`), and the seven-channel
per-phase state follows from the capillary-pressure hard constraints

.. math::

    p_w = p_o - P_{c,ow}(S_w),
    \qquad
    p_g = p_o + P_{c,go}(S_g),
    \qquad
    S_o = 1 - S_w - S_g

where:
- :math:`P_{c,ow}, P_{c,go}`: the SCAL capillary-pressure closures (table interpolants).
- :math:`p_\alpha, S_\alpha`: phase pressures and saturations.

Well stimulation is *not* a network closure: the time-resolved, all-phase
forcing lives in :class:`pinnlab.wells.WellForcing` /
:class:`pinnlab.wells.PredictedForcing` and is consumed directly by the residual
operators.

**Hard initial condition** (``ic_design="hard"``). The raw output is offset by the
deck's initial state and ramped in time,

.. math::

    y(\mathbf{x}, t) \;=\; y^0(\mathbf{x}) + \beta(t)\,\mathcal{N}_\theta\bigl(\gamma(\mathbf{x}, t)\bigr),
    \qquad
    \beta(t) = 1 - e^{-t/\tau_r}

where:
- :math:`y^0`: the logit image of the initial condition (:func:`pinnlab.encodings.raw_ic_offsets`), carried in the feature vector so every call site stays unchanged.
- :math:`\beta(t)`: a fixed ramp vanishing at :math:`t = 0`, so :math:`u_\theta(\cdot, 0)` equals the (range-clamped) initial state identically for every :math:`\theta` and the ``ic`` group disappears the way the natural no-flow condition removed the ``bc`` group.
- :math:`\tau_r` (``ic_tau_days``): the relaxation scale, defaulting to the first report interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .casedata import CaseData


@dataclass
class Primaries:
    """Encoding-agnostic network evaluators with the physical hard constraints applied."""

    primaries_feat: Callable       # (params, feat) -> (p_o, S_w, S_g, R_so)
    primaries_point: Callable      # (params, xt, *enc_args) -> (4,)
    full_state_point: Callable     # (params, xt, *enc_args) -> 7 per-phase channels
    apply: Callable
    encoder: Any


def make_primaries(case: CaseData, encoder, apply, ic_tau: float | None = None) -> Primaries:
    """Bind the network to the case's physical output ranges and tables.

    ``ic_tau`` (days) switches on the hard-IC ansatz: the encoder must then carry the
    raw IC offsets (its ``y0_cells`` is not ``None``); ``0`` means the first report
    interval.
    """
    import jax.numpy as jnp
    from jax.nn import sigmoid

    from modules.utils.blackoil_closures import cap_pres_go, cap_pres_ow

    phys = case.phys
    p_min, p_max, swc, rs_max = phys.P_MIN, phys.P_MAX, phys.SWC, phys.RS_MAX
    tables = case.tables
    hard = ic_tau is not None
    if hard:
        if getattr(encoder, "y0_cells", None) is None:
            raise ValueError("hard IC needs an encoder built with the raw IC offsets")
        tau_r = float(ic_tau) if ic_tau > 0 else float(max(
            (float(case.times[1]) - float(case.times[0])) if case.n_times > 1 else 1.0, 1e-6))

    def _raw(params, feat):
        if not hard:
            return apply(params, feat)
        y0 = feat[-4:]
        f = feat[:-4]
        t = encoder.t_of_tau(f[-1])                     # invert the time channel
        beta = 1.0 - jnp.exp(-t / tau_r)
        return y0 + beta * apply(params, f)

    def primaries_feat(params, feat):
        y = _raw(params, feat)
        p_o = p_min + (p_max - p_min) * sigmoid(y[0])
        s_w = swc + (1.0 - swc) * sigmoid(y[1])
        s_g = sigmoid(y[2]) * (1.0 - s_w)
        rs = rs_max * sigmoid(y[3])
        return jnp.stack([p_o, s_w, s_g, rs])

    def primaries_point(params, xt, *enc_args):
        return primaries_feat(params, encoder.feat_xt(xt, *enc_args))

    def full_state_point(params, xt, *enc_args):
        p_o, s_w, s_g, rs = primaries_point(params, xt, *enc_args)
        s_o = 1.0 - s_w - s_g
        p_w = p_o - cap_pres_ow(s_w, tables)
        p_g = p_o + cap_pres_go(s_g, tables)
        return jnp.stack([p_w, p_o, p_g, s_w, s_o, s_g, rs])

    return Primaries(
        primaries_feat=primaries_feat,
        primaries_point=primaries_point,
        full_state_point=full_state_point,
        apply=apply,
        encoder=encoder,
    )
