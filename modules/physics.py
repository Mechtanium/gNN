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
forcing lives in :class:`modules.wells.WellForcing` and is consumed directly by the
residual operators.
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


def make_primaries(case: CaseData, encoder, apply) -> Primaries:
    """Bind the network to the case's physical output ranges and tables."""
    import jax.numpy as jnp
    from jax.nn import sigmoid

    from modules.utils.blackoil_closures import cap_pres_go, cap_pres_ow

    phys = case.phys
    p_min, p_max, swc, rs_max = phys.P_MIN, phys.P_MAX, phys.SWC, phys.RS_MAX
    tables = case.tables

    def primaries_feat(params, feat):
        y = apply(params, feat)
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
