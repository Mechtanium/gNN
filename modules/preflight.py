r"""
Device-memory preflight of the compiled ENGD step.

The planner of :mod:`modules.memplan` estimates memory before anything is built; the
preflight measures it. Once the pipeline is wired, the special-optimizer step is
compiled on the very arguments the first training iteration will pass, and XLA's
compile-time buffer assignment is held against what the device can still allocate,

.. math::

    B_{\mathrm{step}} \;=\; B_{\mathrm{temp}} + B_{\mathrm{out}} - B_{\mathrm{alias}}
    \;\le\; \rho\, B_{\mathrm{free}}

where:
- :math:`B_{\mathrm{temp}}, B_{\mathrm{out}}, B_{\mathrm{alias}}`: the step's temporaries, outputs and output buffers aliased onto donated inputs, from ``Compiled.memory_analysis()``. The arguments themselves are already resident when the check runs, as are the constants the executable loaded, so neither is counted twice.
- :math:`B_{\mathrm{free}}`: the device's allocatable bytes now — the allocator's limit minus its live bytes where it reports a limit, otherwise the CUDA runtime's free-memory count taken right after a small synchronizing operation, so memory the stream-ordered pool still holds from freed buffers has been returned first.
- :math:`\rho`: the headroom factor (0.95), for fragmentation and the window refreshes that run beside the step.

A step that cannot fit raises :class:`MemoryPreflightError` with the planner's breakdown
and the remedies, instead of a ``RESOURCE_EXHAUSTED`` after the first compile. The
compilation is not wasted when it fits: the step is compiled on exactly the abstract
arguments of the first iteration, so the training loop's first call reuses the
executable instead of compiling it again. On CPU the check is skipped — there is no
device budget to hold it against, and host memory is the operating system's to manage.
"""

from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path

from .config import ConfigError, SpecialOpt

_HEADROOM = 0.95
_GIB = float(1 << 30)


class MemoryPreflightError(ConfigError):
    """The compiled training step needs more device memory than the device has free.

    A :class:`~modules.config.ConfigError` subclass, so the notebook and ``sweep`` error
    paths treat it as an unrunnable composition: the run never starts.
    """


def _cudart():
    """The CUDA runtime library of the pinned environment, or ``None``."""
    try:
        spec = importlib.util.find_spec("nvidia.cuda_runtime")
    except ModuleNotFoundError:     # no nvidia package at all, e.g. on an Intel XPU
        spec = None
    names = []
    if spec is not None and spec.submodule_search_locations:
        root = Path(next(iter(spec.submodule_search_locations)))
        names += [str(p) for p in sorted((root / "lib").glob("libcudart.so*"))]
    names += ["libcudart.so.12", "libcudart.so"]
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def device_free_bytes(device) -> int | None:
    """Bytes ``device`` can still allocate, or ``None`` when that cannot be determined."""
    import jax
    import jax.numpy as jnp

    stats = device.memory_stats() or {}
    limit, in_use = int(stats.get("bytes_limit") or 0), int(stats.get("bytes_in_use") or 0)
    if limit > 0:
        return max(limit - in_use, 0)
    if device.platform != "gpu":
        return None
    # The stream-ordered allocator reports no limit. A synchronizing operation first lets
    # its pool hand memory kept from freed buffers back to the driver; then ask CUDA.
    jax.device_put(jnp.zeros((), jnp.float32), device).block_until_ready()
    lib = _cudart()
    if lib is None:
        return None
    prev, free, total = ctypes.c_int(), ctypes.c_size_t(), ctypes.c_size_t()
    if lib.cudaGetDevice(ctypes.byref(prev)) != 0:
        return None
    try:
        if lib.cudaSetDevice(int(device.local_hardware_id)) != 0:
            return None
        if lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
            return None
    finally:
        lib.cudaSetDevice(prev.value)
    return int(free.value)


def _remedies(cfg) -> str:
    from .config import depth_channel

    tips = []
    if cfg.n_eig > 64 and not depth_channel(cfg):
        tips.append(f"depth_encoding='z' (n_eig={cfg.n_eig} came from the addressability "
                    "floor; a depth channel usually lowers it several-fold, and P with it)")
    tips += ["smaller m_width / n_blocks or engd.param_target (the dense step holds three "
             "P x P float64 matrices at its peak)",
             "lower full_batch_cap or n_tslice (fewer residual rows and a shorter PDE tape)",
             "a smaller engd.point_chunk (point-wise ic/data blocks)",
             "or a device with more memory"]
    return "; ".join(tips)


def check_engd_step(bundle, *, verbose: bool = True) -> dict | None:
    r"""
    Compile the bundle's special (ENGD) step on its first training arguments and hold
    the compiled memory against the free device memory (see the module docstring).

    Returns the measurement ``{temp, out, alias, step, free, device}`` in bytes, or ``None``
    when the check does not apply (not ENGD, CPU backend, or the free memory cannot be
    determined). Raises :class:`MemoryPreflightError` when the step cannot fit.
    """
    import jax
    import jax.numpy as jnp

    from . import memplan, sampling as sampling_mod

    cfg = bundle.cfg
    if cfg.special_opt is not SpecialOpt.ENGD or jax.default_backend() == "cpu":
        return None
    resolved, steps = bundle.resolved, bundle.steps
    devices = list(bundle.env.mesh.devices.flat) if getattr(bundle.env, "mesh", None) is not None \
        else jax.local_devices()

    # the first special-phase call's arguments, built the way init_trainer/_refresh build them
    params, opt_state, phase = steps.opt_init(bundle.model.params0)
    if phase != "special":
        opt_state = steps.special_init(params) if steps.special_init is not None else ()
    w = jnp.asarray(resolved.initial_weights, jnp.float32)
    if resolved.full_batch:
        win = sampling_mod.full_batch_window(cfg, bundle.case, resolved, bundle.env)
    else:
        win = sampling_mod.random_window(cfg, bundle.case, resolved, bundle.env,
                                         jax.random.PRNGKey(0))

    ma = steps.train_step_special.lower(params, opt_state, w, win).compile().memory_analysis()
    if ma is None:
        return None
    temp, out = int(ma.temp_size_in_bytes), int(ma.output_size_in_bytes)
    alias = int(ma.alias_size_in_bytes)
    step = temp + out - alias
    frees = [f for f in (device_free_bytes(d) for d in devices) if f is not None]
    del params, opt_state, w, win
    if not frees:
        return None
    free = min(frees)
    kind = devices[0].device_kind
    rep = {"temp": temp, "out": out, "alias": alias, "step": step, "free": free, "device": kind}
    if step > _HEADROOM * free:
        plan = resolved.engd_plan
        raise MemoryPreflightError(
            f"the compiled ENGD step needs {step / _GIB:.2f} GiB of device memory "
            f"(temporaries {temp / _GIB:.2f} GiB + outputs {out / _GIB:.2f} GiB) but the "
            f"{kind} has {free / _GIB:.2f} GiB free with the model and data resident "
            f"(P={plan.p:,} parameters: the P x P float64 Gramian alone is "
            f"{plan.g_bytes / _GIB:.2f} GiB; N={plan.n_rows:,} residual rows, "
            f"mode={plan.mode}). The planner estimated "
            f"{memplan.engd_step_bytes(cfg, resolved) / _GIB:.2f} GiB. Remedies: "
            + _remedies(cfg) + ".")
    if verbose:
        tight = " (tight)" if step > 0.8 * free else ""
        print(f"[preflight] ENGD step needs {step / _GIB:.2f} GiB (temporaries "
              f"{temp / _GIB:.2f} GiB) of {free / _GIB:.2f} GiB free on the {kind} -> "
              f"fits{tight}")
    return rep
