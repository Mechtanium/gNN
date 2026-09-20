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

(Lifted unchanged from PINN-Lab's ``diagnostics`` module; the figure code stayed behind.)
"""

from __future__ import annotations

_FIELD_INDEX = {"p_o": 0, "S_w": 1, "S_g": 2, "R_so": 3}
_FIELD_LABEL = {"p_o": "p_o [psia]", "S_w": "S_w [-]", "S_g": "S_g [-]", "R_so": "R_so [Mscf/stb]"}
_FIELD_REF = {"p_o": "pres", "S_w": "swat", "S_g": "sgas", "R_so": "rs"}


def run_tag(bundle, rid: str = "") -> str:
    """One-line run-condition descriptor stamped on every figure (mix-proof labeling)."""
    cfg = bundle.cfg
    bp = cfg.backprop_design.value if cfg.backprop_design is not None else "-"
    bits = [cfg.mesh_case.value,
            f"{cfg.residual_design.value}+{bp}",
            f"{cfg.input_encoding.value}/{cfg.architecture.value}",
            f"opt={cfg.special_opt.value}@{cfg.special_opt_after}"]
    if bundle.spec is not None and bundle.spec.lam is not None:
        bits.append(f"n_eig={cfg.n_eig} (z:{bundle.spec.n_eig_z})")
    if cfg.invert:
        bits.append("invert=" + "+".join(cfg.invert))
    if rid:
        bits.append(rid)
    return " | ".join(bits)


def _stamp(fig, tag: str):
    """Write the run-condition tag onto the figure canvas (bottom-right footer)."""
    if tag:
        fig.text(0.995, 0.002, tag, ha="right", va="bottom", fontsize=6.5, color="0.35")
    return fig


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
