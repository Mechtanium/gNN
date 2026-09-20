r"""
Whole-mesh nodal network evaluation, sharded and rematerialized (fem_nodal path).

The node axis is padded to a multiple of the data mesh dim and reshaped
``(data_dim, per_dev, dim_in)`` so each device owns a contiguous row block;
each device walks its rows in ``fem_chunk``-row rematerialized micro-batches —
``jax.checkpoint`` keeps only the chunk inputs on the reverse tape (the network
eval is recomputed chunk-by-chunk in the backward pass), so the live tape is
:math:`\mathcal{O}(\text{fem\_chunk} \times M \times \ell)` per device,
independent of the mesh size. ``jax.jvp`` over :math:`t` yields
:math:`\partial P/\partial t` in the same pass:

.. math::

    \bigl(P(t), \tfrac{\partial P}{\partial t}(t)\bigr)
    \;=\;
    \mathrm{jvp}\bigl(t \mapsto u_\theta(\gamma(x_i, t)),\; t,\; 1\bigr)

where:
- :math:`P \in \mathbb{R}^{n\times 4}`: the primaries at every mesh node.
- :math:`\gamma`: the configured input encoding (spectral eigenfeatures or normalized Cartesian coordinates).
"""

from __future__ import annotations

from typing import Callable

from .config import RunConfig
from .meshenv import MeshEnv


def make_nodal_evaluator(cfg: RunConfig, env: MeshEnv, encoder, primaries_feat: Callable,
                         n_nodes: int) -> tuple[Callable, Callable]:
    """
    Returns ``(primaries_nodes, nodes_P_dPdt)``:

    - ``primaries_nodes(params, t) -> (n_nodes, 4)`` — plain whole-mesh vmap
      (reference/parity path, no sharding hints);
    - ``nodes_P_dPdt(params, t) -> ((n_nodes, 4), (n_nodes, 4))`` — the
      sharded, rematerialized evaluator with the time tangent.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    data_dim = env.data_dim
    pad = (-n_nodes) % data_dim
    per_dev = (n_nodes + pad) // data_dim
    bs = per_dev if cfg.fem_chunk <= 0 else min(cfg.fem_chunk, per_dev)

    def primaries_nodes(params, t):
        feats = encoder.feat_nodes_t(t)
        return jax.vmap(lambda f: primaries_feat(params, f))(feats)

    def nodes_P_dPdt(params, t):
        def _eval_all(tt):
            feats = encoder.feat_nodes_t(tt)
            if pad:
                feats = jnp.concatenate([feats, jnp.zeros((pad, feats.shape[1]), feats.dtype)])
            rows = jax.lax.with_sharding_constraint(
                feats.reshape(data_dim, per_dev, -1), env.nd(P("data", None, None)))
            body = jax.checkpoint(lambda f: primaries_feat(params, f))
            out = jax.vmap(lambda dev_rows: jax.lax.map(body, dev_rows, batch_size=bs))(rows)
            out = out.reshape(-1, 4)[:n_nodes]
            if pad:
                # uneven node axis: this hint would be an eager device_put outside jit
                # (IndivisibleError); the padded `rows` constraint above already keeps
                # the jitted compute data-sharded, so let GSPMD propagate from there
                return out
            return jax.lax.with_sharding_constraint(out, env.nd(P("data", None)))

        return jax.jvp(_eval_all, (t,), (jnp.ones_like(t),))

    return primaries_nodes, nodes_P_dPdt
