r"""
Pre-jax environment bootstrap (port of notebook cell 1).

Everything here must run BEFORE the first ``import jax`` in the process:
CUDA-12 ``ptxas`` selection for pre-Ampere GPUs, the stream-ordered
``cuda_async`` allocator, the NCCL Socket-bootstrap pin that survives GCP A3
images exporting a broken gIB plugin, and the x64 flag that backs the
``selective_f64`` precision policy.

The module is idempotent: calling :func:`setup_environment` twice with the
same policy is a no-op; calling it after jax was already initialized with a
conflicting x64 setting raises.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import shutil
import sys
from pathlib import Path

_CONFIGURED_POLICY: str | None = None

#: Where a container platform may mount the NVIDIA driver without putting it
#: on the loader path (GKE: ``/usr/local/nvidia/lib64``). ``LD_LIBRARY_PATH``
#: is read by the loader once at process start, so a late change to the
#: environment does nothing — the library is preloaded by absolute path instead.
_DRIVER_DIRS = ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib")
_DRIVER_SONAME = "libcuda.so.1"


def ensure_driver_visible() -> str | None:
    r"""
    Make the NVIDIA driver library loadable before jax's CUDA plugin looks for it.

    Returns the path that had to be preloaded, ``None`` when the loader already
    finds ``libcuda.so.1``. Raises with a plain diagnosis when a GPU device is
    present but no driver library can be found anywhere: jax's own message in
    that situation ("Backend 'cuda' is not in the list of known backends") hides
    the cause, which is the container's library path, not jax.

    Preloading works because glibc satisfies a later ``dlopen("libcuda.so.1")``
    from an object already loaded under that SONAME.
    """
    try:
        ctypes.CDLL(_DRIVER_SONAME)
        return None
    except OSError:
        pass
    candidates = [*_DRIVER_DIRS, *filter(None, os.environ.get("LD_LIBRARY_PATH", "").split(":"))]
    for d in candidates:
        path = Path(d) / _DRIVER_SONAME
        if path.is_file():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            # For child processes (the Flow subprocess, xvfb) that may want it.
            os.environ["LD_LIBRARY_PATH"] = ":".join(
                dict.fromkeys([d, *filter(None, os.environ.get("LD_LIBRARY_PATH", "").split(":"))])
            )
            return str(path)
    raise RuntimeError(
        "A GPU device is present but the NVIDIA driver library libcuda.so.1 is not "
        "visible to this process: it is neither on the loader path nor under "
        f"{', '.join(_DRIVER_DIRS)}. On GKE the driver is mounted at /usr/local/nvidia/lib64 "
        "and the container must carry it in LD_LIBRARY_PATH; on other hosts install the "
        "NVIDIA driver or run on the CPU (JAX_PLATFORMS=cpu)."
    )


def _cuda_requested(platforms: str) -> bool:
    return "cuda" in platforms.lower()


def setup_environment(
    precision_policy: str = "selective_f64",
    opt_f64: bool = True,
    platforms: str | None = None,
) -> None:
    r"""
    Configure the process environment and import jax with the right x64 flag.

    Parameters
    ----------
    precision_policy:
        ``"f32"`` (single precision everywhere, bit-parity baseline) or
        ``"selective_f64"`` (enable x64; only the cancellation-prone residual
        tails are promoted downstream).
    opt_f64:
        Float64 master weights + optimizer moments. Requires
        ``selective_f64`` (x64 enabled).
    platforms:
        Overrides ``JAX_PLATFORMS``. When ``None`` an existing environment
        value is respected (e.g. the CPU-pinned test suite) and
        ``"cuda,cpu"`` is used as the default otherwise.
    """
    global _CONFIGURED_POLICY

    if precision_policy not in ("f32", "selective_f64"):
        raise ValueError(f"unknown precision_policy {precision_policy!r}")
    if opt_f64 and precision_policy != "selective_f64":
        raise ValueError("opt_f64=True requires precision_policy='selective_f64' (x64 enabled)")

    want_x64 = precision_policy == "selective_f64"
    if "jax" in sys.modules:
        import jax

        have_x64 = bool(jax.config.jax_enable_x64)
        if have_x64 != want_x64:
            raise RuntimeError(
                f"jax is already imported with jax_enable_x64={have_x64}; cannot switch to "
                f"precision_policy={precision_policy!r}. Restart the kernel/process."
            )
        _CONFIGURED_POLICY = precision_policy
        return

    if platforms is not None:
        os.environ["JAX_PLATFORMS"] = platforms
    else:
        os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    effective_platforms = os.environ["JAX_PLATFORMS"]

    if _cuda_requested(effective_platforms):
        ensure_driver_visible()
        # CUDA 13 no longer assembles code for the V100/T4 compute capabilities; select the
        # CUDA 12 compiler shipped by the pinned jax[cuda12] environment.
        cuda_nvcc_spec = importlib.util.find_spec("nvidia.cuda_nvcc")
        if cuda_nvcc_spec is None or not cuda_nvcc_spec.submodule_search_locations:
            raise RuntimeError("The pinned jax[cuda12] runtime is missing nvidia-cuda-nvcc-cu12.")
        cuda12_root = Path(next(iter(cuda_nvcc_spec.submodule_search_locations))).resolve()
        if not (cuda12_root / "bin" / "ptxas").is_file():
            raise RuntimeError(f"CUDA 12 ptxas was not found under {cuda12_root}.")
        os.environ["CUDA_ROOT"] = str(cuda12_root)

        existing_xla_flags = os.environ.get("XLA_FLAGS", "")
        if "--xla_gpu_enable_command_buffer=" not in existing_xla_flags:
            os.environ["XLA_FLAGS"] = f"{existing_xla_flags} --xla_gpu_enable_command_buffer=".strip()

        # Stream-ordered async allocator: serves the large second-order-autodiff buffers on
        # demand instead of preallocating a fixed arena.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")

        # Multi-GPU collectives: force NCCL's built-in Socket bootstrap. GCP A3 images export
        # NCCL_NET=gIB; with no IB device present it aborts every communicator init. Intra-node
        # transfers still ride the NVLink P2P fabric.
        os.environ["NCCL_NET"] = "Socket"
        os.environ["NCCL_NET_PLUGIN"] = "none"

    import jax

    jax.config.update("jax_enable_x64", want_x64)
    _CONFIGURED_POLICY = precision_policy
    if _cuda_requested(effective_platforms):
        try:
            backend = jax.default_backend()
        except RuntimeError as exc:
            plugin = importlib.util.find_spec("jax_plugins.xla_cuda12") is not None
            raise RuntimeError(
                f"jax could not bring up its CUDA backend ({exc}). The jax CUDA plugin is "
                f"{'installed' if plugin else 'NOT installed (pip install jax[cuda12])'}; the "
                "driver library was "
                f"{'found' if _driver_loaded() else 'not found'}. Set JAX_PLATFORMS=cpu to "
                "train on the CPU instead."
            ) from exc
        if backend != "gpu":
            raise RuntimeError(
                f"CUDA was requested (JAX_PLATFORMS={effective_platforms!r}) but jax chose the "
                f"{backend!r} backend; the GPU is not usable from this process."
            )


def _driver_loaded() -> bool:
    try:
        ctypes.CDLL(_DRIVER_SONAME)
        return True
    except OSError:
        return False


def platforms_for_host(max_cpu_devices: int = 8) -> str:
    """``"cuda,cpu"`` when an NVIDIA device is visible, else ``"cpu"`` — and on CPU
    the XLA flag that splits the host into up to ``max_cpu_devices`` devices, so the
    data-parallel mesh has something to shard over. ``JAX_PLATFORMS`` set by the
    caller wins."""
    forced = os.environ.get("JAX_PLATFORMS")
    if forced:
        return forced
    has_gpu = Path("/dev/nvidia0").exists() or shutil.which("nvidia-smi") is not None
    if has_gpu:
        return "cuda,cpu"
    n = max(1, min(max_cpu_devices, os.cpu_count() or 1))
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_force_host_platform_device_count" not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} --xla_force_host_platform_device_count={n}".strip()
    return "cpu"


def env_summary() -> dict:
    """Small diagnostic dict for the notebook's bootstrap cell (requires jax imported)."""
    import jax

    devices = jax.devices()
    return {
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "device_count": jax.device_count(),
        "devices": [d.device_kind for d in devices],
        "x64": bool(jax.config.jax_enable_x64),
        "precision_policy": _CONFIGURED_POLICY,
    }
