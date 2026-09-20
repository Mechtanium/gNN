r"""
Eclipse-style observables recorded across an inference window.

Three families, all evaluated from one trained parameter set at arbitrary times
(the network is a continuous field, so a "report step" is only a time to sample):

**Well observables** reuse the Peaceman closure of :mod:`pinnlab.wells`, which
already eliminates the bottom-hole pressure in rate-controlled mode and returns
per-phase surface rates. The derived ratios are

.. math::

    \mathrm{WGOR} = \frac{\mathrm{WGPR}}{\mathrm{WOPR}},
    \qquad
    \mathrm{WWCT} = \frac{\mathrm{WWPR}}{\mathrm{WWPR} + \mathrm{WOPR}}

where:
- :math:`\mathrm{WOPR}, \mathrm{WWPR}, \mathrm{WGPR}`: surface oil, water and gas production rates [stb/day, stb/day, Mscf/day], reported positive when producing.
- :math:`\mathrm{WGOR}`: producing gas-oil ratio [Mscf/stb]; the gas leg is the *total* surface gas (free plus dissolved), which is the deck's own WGPR convention.
- :math:`\mathrm{WWCT}`: water cut, the water fraction of surface liquid.

**Field totals** are the pore-volume reduction of the same accumulation the PDE
uses, so an in-place volume is the residual's own integrand summed over cells:

.. math::

    \mathrm{FOIP}(t) = \sum_{c} V_c\, \beta\, \phi_c\, \frac{S_{o,c}}{B_{o,c}},
    \qquad
    \mathrm{FWIP}(t) = \sum_{c} V_c\, \beta\, \phi_c\, \frac{S_{w,c}}{B_{w,c}},

.. math::

    \mathrm{FGIP}(t) = \sum_{c} V_c\, \beta\, \phi_c
    \left(\frac{S_{g,c}}{B_{g,c}} + R_{s,c}\,\frac{S_{o,c}}{B_{o,c}}\right)

where:
- :math:`V_c`: cell bulk volume [ft³], from the FEM quadrature when available.
- :math:`\beta`: the ft³ :math:`\to` reservoir-barrel factor ``case.rb_ft3``.
- :math:`\phi_c = \phi_c^{0}\,(1 + c_r (p_{o,c} - p_{\mathrm{ref}}))`: pressure-corrected porosity.
- :math:`B_{o}, B_{w}, B_{g}`: black-oil formation-volume factors at the cell's own phase pressures.
- :math:`R_{s}`: dissolved gas-oil ratio, so FGIP counts free *and* dissolved gas.

**Block states** are the primaries themselves plus the phase pressures the
capillary closures imply:

.. math::

    \mathrm{BPR}_o = p_o,
    \qquad
    \mathrm{BPR}_w = p_o - p_{cow}(S_w),
    \qquad
    \mathrm{BPR}_g = p_o + p_{cgo}(S_g)

where:
- :math:`p_{cow}, p_{cgo}`: the oil-water and gas-oil capillary pressures from the deck's SCAL tables.
- :math:`\mathrm{BPR}_\alpha`: per-block phase pressure [psia] of phase :math:`\alpha`.
"""

from __future__ import annotations

from typing import Any

import numpy as onp


def _phase_pressures(bundle, P):
    """(p_o, p_w, p_g) [psia] at every cell from the primaries block ``P`` (n_cells, 4)."""
    import jax.numpy as jnp

    from modules.utils.blackoil_closures import cap_pres_go, cap_pres_ow

    tab = bundle.case.tables
    p_o, s_w, s_g = P[:, 0], P[:, 1], P[:, 2]
    p_w = p_o - cap_pres_ow(s_w, tab)
    p_g = p_o + cap_pres_go(s_g, tab)
    return onp.asarray(p_o), onp.asarray(p_w), onp.asarray(p_g)


def block_states(bundle, params, times) -> dict[str, onp.ndarray]:
    """Per-cell saturations and phase pressures at each time: arrays ``(n_t, n_cells)``."""
    from .metrics import predict_cells

    soil, sgas, swat, bpo, bpw, bpg = [], [], [], [], [], []
    for t in times:
        P = predict_cells(bundle, params, float(t))
        s_w, s_g = onp.asarray(P[:, 1]), onp.asarray(P[:, 2])
        p_o, p_w, p_g = _phase_pressures(bundle, P)
        swat.append(s_w); sgas.append(s_g); soil.append(1.0 - s_w - s_g)
        bpo.append(p_o); bpw.append(p_w); bpg.append(p_g)
    return {"SOIL": onp.stack(soil), "SGAS": onp.stack(sgas), "SWAT": onp.stack(swat),
            "BPR_o": onp.stack(bpo), "BPR_w": onp.stack(bpw), "BPR_g": onp.stack(bpg)}
