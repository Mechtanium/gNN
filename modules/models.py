r"""
Network architectures as Flax NNX modules (port of the notebook's network cells).

Both architectures are threaded functionally — split once into a static
graphdef + a ``State`` pytree of arrays — so every physics call site keeps the
plain ``apply(params, feat)`` signature and the State differentiates and
tree-flattens cleanly (required by :mod:`ntk_trace_jax` and the ENGD Jacobian
assembly). NNX Params are kept plain (no ``sharding=`` kwarg): sharding is
attached externally through the shape-keyed PartitionSpec map built here.

The DGM forward (Sirignano–Spiliopoulos) is

.. math::

    S^1 = \tanh(x W^1 + b^1), \qquad
    S^{\ell+1} = (1 - G^\ell) \odot H^\ell + Z^\ell \odot S^\ell, \qquad
    y = S^{L+1} W^{\mathrm{out}} + b^{\mathrm{out}}

with gates

.. math::

    Z^\ell = \tanh(x U_z^\ell + S^\ell W_z^\ell + b_z^\ell), \quad
    G^\ell = \tanh(x U_g^\ell + S^\ell W_g^\ell + b_g^\ell), \quad
    R^\ell = \tanh(x U_r^\ell + S^\ell W_r^\ell + b_r^\ell), \quad
    H^\ell = \tanh(x U_h^\ell + (S^\ell \odot R^\ell) W_h^\ell + b_h^\ell)

where:
- :math:`x \in \mathbb{R}^{d}`: the encoded input features.
- :math:`S^\ell \in \mathbb{R}^{M}`: the recurrent hidden state of width :math:`M` (the tensor-parallel axis).
- :math:`U_\bullet \in \mathbb{R}^{d\times M},\; W_\bullet \in \mathbb{R}^{M\times M}`: gate input/state matrices (column-parallel).
- :math:`W^{\mathrm{out}} \in \mathbb{R}^{M\times f}`: the row-parallel readout.

Initialization copies the raw ``jaxpinns.architectures`` arrays for bit-exact
parity with the reference implementations (gated in the tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .config import Architecture, RunConfig


def raw_init_f32(init_fn, key):
    """
    Draw a jaxpinns dtype-free Glorot init at float32 under BOTH precision policies.

    ``random.normal`` carries no dtype pin inside jaxpinns, so under
    selective_f64 (x64 on) it would return float64 weights and promote the
    whole network/optimizer. The x64 flag is flipped off around the draw
    (bit-identical to the f32-policy stream) and every leaf — including the
    numpy float64 zero biases — is pinned to float32.
    """
    import jax
    import jax.numpy as jnp

    was = jax.config.jax_enable_x64
    if was:
        jax.config.update("jax_enable_x64", False)
    try:
        raw = init_fn(key)
    finally:
        if was:
            jax.config.update("jax_enable_x64", True)
    return jax.tree.map(lambda a: jnp.asarray(a, jnp.float32), raw)


def _make_dgm_classes():
    """DGM NNX module classes (deferred so this module imports without jax/flax)."""
    import jax.numpy as jnp
    from flax import nnx

    class DGMBlock(nnx.Module):
        def __init__(self, Uz, Ug, Ur, Uh, Wz, Wg, Wr, Wh, bz, bg, br, bh):
            self.Uz = nnx.Param(Uz); self.Ug = nnx.Param(Ug); self.Ur = nnx.Param(Ur); self.Uh = nnx.Param(Uh)
            self.Wz = nnx.Param(Wz); self.Wg = nnx.Param(Wg); self.Wr = nnx.Param(Wr); self.Wh = nnx.Param(Wh)
            self.bz = nnx.Param(bz); self.bg = nnx.Param(bg); self.br = nnx.Param(br); self.bh = nnx.Param(bh)

        def __call__(self, inputs, X):
            Z = jnp.tanh(inputs @ self.Uz[...] + X @ self.Wz[...] + self.bz[...])
            G = jnp.tanh(inputs @ self.Ug[...] + X @ self.Wg[...] + self.bg[...])
            R = jnp.tanh(inputs @ self.Ur[...] + X @ self.Wr[...] + self.br[...])
            H = jnp.tanh(inputs @ self.Uh[...] + (X * R) @ self.Wh[...] + self.bh[...])
            return (1.0 - G) * H + Z * X

    class DGMNet(nnx.Module):
        def __init__(self, raw_params):
            (W1, b1), blocks, (Wout, bout) = raw_params
            self.W1 = nnx.Param(W1); self.b1 = nnx.Param(b1)
            self.blocks = nnx.data([DGMBlock(*blk) for blk in blocks])
            self.Wout = nnx.Param(Wout); self.bout = nnx.Param(bout)

        def __call__(self, inputs):
            X = jnp.tanh(inputs @ self.W1[...] + self.b1[...])
            for blk in self.blocks:
                X = blk(inputs, X)
            return X @ self.Wout[...] + self.bout[...]

    return DGMBlock, DGMNet


def _make_mlp_classes():
    """MLP NNX module classes (tanh hidden layers, linear readout)."""
    import jax.numpy as jnp
    from flax import nnx

    class MLPLayer(nnx.Module):
        def __init__(self, W, b):
            self.W = nnx.Param(W)
            self.b = nnx.Param(b)

    class MLPNet(nnx.Module):
        def __init__(self, raw_params):
            self.layers = nnx.data([MLPLayer(W, b) for (W, b) in raw_params])

        def __call__(self, inputs):
            x = inputs
            for layer in self.layers[:-1]:
                x = jnp.tanh(x @ layer.W[...] + layer.b[...])
            last = self.layers[-1]
            return x @ last.W[...] + last.b[...]

    return MLPLayer, MLPNet


@dataclass
class ModelBundle:
    """A built network: functional apply + init state + sharding map + parity references."""

    arch: Architecture
    graphdef: Any
    params0: Any                       # nnx.State of float32 arrays
    apply: Callable                    # apply(params, feat) -> (dim_out,)
    raw_params0: Any                   # jaxpinns raw params (bit-parity reference)
    raw_apply: Callable                # jaxpinns apply (bit-parity reference)
    pspec_of_shape: Callable           # shape tuple -> PartitionSpec (Megatron col/row roles)
    param_count: int
    dim_in: int


def _pspec_table(cfg: RunConfig, dim_in: int) -> dict:
    """
    Shape -> PartitionSpec role table (Megatron column-then-row).

    Built in role-priority order with ``setdefault`` so square-shape collisions
    (e.g. ``m_width == dim_out``) resolve toward the majority column-parallel
    role — a performance hint only; GSPMD Auto axes keep any choice correct.
    """
    from jax.sharding import PartitionSpec as P

    table: dict = {}
    f = cfg.dim_out
    if cfg.architecture is Architecture.DGM:
        m = cfg.m_width
        for shape, spec in (
            ((dim_in, m), P(None, "model")),   # W1, U* (column-parallel)
            ((m, m), P(None, "model")),        # W* block matrices
            ((m, f), P("model", None)),        # Wout (row-parallel -> all-reduce)
            ((m,), P("model")),                # biases
            ((f,), P()),                       # bout
        ):
            table.setdefault(shape, spec)
    else:
        layers = (dim_in, *cfg.mlp_hidden, f)
        for d_in, d_out in zip(layers[:-2], layers[1:-1]):     # hidden layers: column-parallel
            table.setdefault((d_in, d_out), P(None, "model"))
            table.setdefault((d_out,), P("model"))
        table.setdefault((layers[-2], f), P("model", None))    # readout: row-parallel
        table.setdefault((f,), P())
    return table


def build_model(cfg: RunConfig, dim_in: int, seed: int | None = None) -> ModelBundle:
    """Instantiate the configured architecture with bit-parity jaxpinns initialization."""
    import jax.random as random
    from flax import nnx

    from .dgm_reference import DGM
    from . import memplan

    seed = cfg.seed if seed is None else seed
    key = random.PRNGKey(seed)

    if cfg.architecture is Architecture.DGM:
        raw_init, raw_apply = DGM([dim_in, cfg.m_width, cfg.dim_out], l=cfg.n_blocks)
        raw0 = raw_init_f32(raw_init, key)
        _, DGMNet = _make_dgm_classes()
        model = DGMNet(raw0)
    else:
        raise NotImplementedError("only the DGM architecture is built here")
        raw_init, raw_apply = None, None
        raw0 = raw_init_f32(raw_init, key)
        _, MLPNet = _make_mlp_classes()
        model = MLPNet(raw0)

    graphdef, params0 = nnx.split(model, nnx.Param)

    def apply(p, feat):
        return nnx.merge(graphdef, p)(feat)

    table = _pspec_table(cfg, dim_in)

    def pspec_of_shape(shape: tuple):
        from jax.sharding import PartitionSpec as P

        return table.get(tuple(shape), P())

    return ModelBundle(
        arch=cfg.architecture,
        graphdef=graphdef,
        params0=params0,
        apply=apply,
        raw_params0=raw0,
        raw_apply=raw_apply,
        pspec_of_shape=pspec_of_shape,
        param_count=memplan.param_count(cfg, dim_in),
        dim_in=dim_in,
    )


def flatten_state(state) -> tuple:
    r"""
    Flatten an ``nnx.State`` (or any pytree) to one f64-safe vector.

    Manual ``tree_flatten`` + concatenation of ``reshape(-1)`` leaves with the
    treedef and shapes recorded — deliberately NOT ``ravel_pytree``, whose
    dtype/ordering behavior on nnx.State has shifted across flax versions.
    Returns ``(flat, unflatten)`` with a bit-exact round trip (unit-tested).
    """
    import jax
    import jax.numpy as jnp

    leaves, treedef = jax.tree.flatten(state)
    shapes = [l.shape for l in leaves]
    sizes = [int(l.size) for l in leaves]
    dtypes = [l.dtype for l in leaves]
    flat = jnp.concatenate([jnp.reshape(l, (-1,)) for l in leaves]) if leaves else jnp.zeros((0,))

    def unflatten(vec):
        out, off = [], 0
        for shape, size, dt in zip(shapes, sizes, dtypes):
            out.append(jnp.reshape(vec[off:off + size], shape).astype(dt))
            off += size
        return jax.tree.unflatten(treedef, out)

    return flat, unflatten
