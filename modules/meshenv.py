r"""
Elastic 2-D device mesh and sharding helpers (port of the mesh control-panel cell).

The device mesh has a ``data`` axis (shards collocation/reference batches and,
in fem_nodal mode, the mesh node axis) and a ``model`` axis (shards the hidden
width: FSDP params/grads/optimizer state + tensor-parallel activations). Axis
types are ``Auto`` so classic GSPMD ``with_sharding_constraint`` hints insert
collectives (jax 0.10's default ``Explicit`` axes would forbid them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import RunConfig, mesh_shape_of


@dataclass
class MeshEnv:
    """The device mesh with its common sharding handles."""

    mesh: Any
    mesh_shape: tuple[int, int]
    repl: Any        # fully replicated NamedSharding
    sh_data: Any     # 1-D batch sharded on 'data'
    sh_data2: Any    # (batch, feat) sharded on 'data'

    @property
    def data_dim(self) -> int:
        return self.mesh_shape[0]

    @property
    def model_dim(self) -> int:
        return self.mesh_shape[1]

    def nd(self, spec):
        """PartitionSpec -> NamedSharding on this mesh."""
        from jax.sharding import NamedSharding

        return NamedSharding(self.mesh, spec)


def build_mesh(cfg: RunConfig, device_count: int | None = None) -> MeshEnv:
    """Create the (data, model) device mesh for the parallelism component."""
    import jax
    from jax.sharding import AxisType, PartitionSpec as P

    n_dev = jax.device_count() if device_count is None else device_count
    shape = mesh_shape_of(cfg, n_dev)
    mesh = jax.make_mesh(shape, ("data", "model"), axis_types=(AxisType.Auto,) * len(shape))
    env = MeshEnv(mesh=mesh, mesh_shape=shape, repl=None, sh_data=None, sh_data2=None)
    env.repl = env.nd(P())
    env.sh_data = env.nd(P("data"))
    env.sh_data2 = env.nd(P("data", None))
    return env


def stacked_pspec(pspec_of_shape, shape: tuple[int, ...]):
    """
    Extend a shape-keyed PartitionSpec map to stacked leaves: a leading stack
    axis over a known param shape (e.g. the (m, ...) L-BFGS curvature memories)
    is replicated and the inner spec kept.
    """
    from jax.sharding import PartitionSpec as P

    spec = pspec_of_shape(shape)
    if spec == P() and len(shape) > 1:
        inner = pspec_of_shape(shape[1:])
        if inner != P():
            return P(None, *inner)
    return spec


def shard_like_params(env: MeshEnv, pspec_of_shape, tree):
    """Map every leaf of a params/opt-state pytree to its NamedSharding (FSDP on 'model')."""
    import jax

    return jax.tree.map(lambda x: env.nd(stacked_pspec(pspec_of_shape, tuple(x.shape))), tree)
