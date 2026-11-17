r"""
Accuracy metrics against the simulator's reference states.

:func:`predict_cells` evaluates the four primaries — oil pressure and the
water/gas saturations and the dissolved-gas ratio,

.. math::

    \mathbf{u}(\mathbf{x}_c, t) = \bigl(p_o, S_w, S_g, R_{so}\bigr)

at every cell centroid :math:`\mathbf{x}_c` and one time :math:`t`; the RMSE
helpers compare one primary with the reference field at a report step,

.. math::

    \mathrm{RMSE}(t_i) = \sqrt{\frac{1}{n_c}\sum_{c=1}^{n_c}
    \bigl(u_c(t_i) - u^{\mathrm{ref}}_c(t_i)\bigr)^2}

where:

- :math:`n_c`: the number of active cells.
- :math:`u^{\mathrm{ref}}_c(t_i)`: the simulator's value in cell :math:`c` at report step :math:`i`.

(Lifted unchanged from PINN-Lab's ``diagnostics`` module, with the cell volumes of its ``inversion`` module; the figure code stayed behind.)
"""

from __future__ import annotations

_FIELD_INDEX = {"p_o": 0, "S_w": 1, "S_g": 2, "R_so": 3}
_FIELD_REF = {"p_o": "pres", "S_w": "swat", "S_g": "sgas", "R_so": "rs"}


def predict_cells(bundle, params, t: float):
    """Primaries at every cell centroid at time ``t`` (uses f32 casts of the masters)."""
    import jax
    import jax.numpy as jnp

    prim, enc = bundle.prim, bundle.prim.encoder
    centroids = bundle.centroids
    p32 = bundle.steps.as_f32(params)
    allc = jnp.arange(bundle.case.n_cells, dtype=jnp.int32)
    enc_args = enc.gather_args(allc)

    def one(c, *a):
        xt = jnp.concatenate([centroids[c], jnp.asarray([t], jnp.float32)])
        return prim.primaries_point(p32, xt, *a)

    return jax.vmap(one)(allc, *enc_args)


def field_rmse(bundle, params, field: str = "p_o", t_index: int = -1) -> float:
    """RMSE of one primary against the reference states at one report step."""
    import numpy as onp

    t = float(bundle.case.times[t_index])
    pred = onp.asarray(predict_cells(bundle, params, t)[:, _FIELD_INDEX[field]])
    true = getattr(bundle.case, _FIELD_REF[field])[t_index]
    return float(onp.sqrt(onp.mean((pred - true) ** 2)))


def pressure_rmse(bundle, params, t_index: int = -1) -> float:
    return field_rmse(bundle, params, "p_o", t_index)


def rmse_series(bundle, params, max_times: int = 16, field: str = "p_o"):
    """(time_indices, rmse array) on a strided subset of the report steps."""
    import numpy as onp

    n_t = bundle.case.n_times
    idx = onp.unique(onp.linspace(0, n_t - 1, min(max_times, n_t)).round().astype(int))
    vals = onp.array([field_rmse(bundle, params, field, int(i)) for i in idx])
    return idx, vals


def supersaturated_share(bundle, params, max_times: int = 16, tol: float = 0.05) -> float:
    r"""
    The share of (cell, report time) pairs, on :func:`rmse_series`'s strided report times,
    whose predicted oil is supersaturated, :math:`R_s > (1 + \mathrm{tol})\,R_s^{\mathrm{sat}}(p_o)`
    (:func:`modules.equilibrium.supersaturated_fraction`) — zero for any state the
    simulator produces.
    """
    import numpy as onp

    from .equilibrium import saturated_table, supersaturated_fraction

    pb, rs_tab = saturated_table(bundle.case.tables)
    n_t = bundle.case.n_times
    idx = onp.unique(onp.linspace(0, n_t - 1, min(max_times, n_t)).round().astype(int))
    pred = onp.stack([onp.asarray(predict_cells(bundle, params, float(bundle.case.times[i])))
                      for i in idx])
    return supersaturated_fraction(pred[..., 0], pred[..., 3], pb, rs_tab, tol)


def mean_rmse(bundle, params) -> float:
    r"""
    The time-mean pressure RMSE over **all** report times of the case,

    .. math::

        \overline{\mathrm{RMSE}}
        \;=\;
        \frac{1}{n_t}\sum_{i=1}^{n_t} \mathrm{RMSE}(t_i)

    where:

    - :math:`\mathrm{RMSE}(t_i)`: the single-report-time pressure RMSE of :func:`pressure_rmse`.
    - :math:`n_t`: the case's report-time count (``case.n_times``) — every report step enters, unlike :func:`rmse_series`'s strided subset.
    """
    return float(sum(pressure_rmse(bundle, params, int(i))
                     for i in range(bundle.case.n_times)) / bundle.case.n_times)


def cell_volumes_ft3(case, spec):
    """Per-cell volumes [ft³]: exact hex quadrature when FEM static exists, else box product."""
    import numpy as onp

    if spec is not None:
        return onp.asarray(spec.static["JxW"]).sum(axis=1)
    return onp.prod(onp.asarray(case.cell_len, onp.float64), axis=1)
