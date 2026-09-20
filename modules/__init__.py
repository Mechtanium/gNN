r"""The Delta-PINN training pipeline, extracted from PINN-Lab for the PERD workflow.

The composition is fixed to the notebook's gold standard — spectral encoding,
DGM, spectral PDE residual with FEM-nodal backprop, permeability-weighted
stiffness, the deterministic full-batch window, fixed loss weights, ENGD, a
closed-form well — and the case is whatever Eclipse deck the workflow is
given (``RunConfig.deck_path``), not a member of a fixed list.

Import order contract: this package does NOT import jax at package level.
Call :func:`modules.bootstrap.setup_environment` before importing any module
that pulls in jax (``pipeline``, ``training``, ...) so the CUDA/XLA
environment and the x64 flag are configured before backend initialization.
"""

__version__ = "0.1.0"
