r"""
Pre-jax environment bootstrap (port of notebook cell 1).

Everything here must run BEFORE the first ``import jax`` in the process: the choice of
accelerator (``JAX_PLATFORMS``), the backend-specific environment — on CUDA the
CUDA-12 ``ptxas`` selection for pre-Ampere GPUs, the stream-ordered ``cuda_async``
allocator and the NCCL Socket-bootstrap pin that survives GCP A3 images exporting a
broken gIB plugin; on an Intel XPU (the oneAPI PJRT plugin) only the on-demand
allocator — and the x64 flag that backs the ``selective_f64`` precision policy.

Accelerators are named by the user as ``"cuda"`` (NVIDIA, ``jax[cuda12]``) or ``"xpu"``
(Intel, ``jax[oneapi]``); ``"auto"`` picks whichever plugin the environment has
installed, falling back to the CPU. ``"xpu"`` is Intel's name for the device; the
plugin registers the JAX platform as ``oneapi``, which is what ``JAX_PLATFORMS`` receives.

The module is idempotent: calling :func:`setup_environment` twice with the
same policy is a no-op; calling it after jax was already initialized with a
conflicting x64 setting raises.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from pathlib import Path

_CONFIGURED_POLICY: str | None = None
# Per-deck ceiling of the persistent compilation cache, least recently used entries evicted
# first: a NORNE ENGD step serializes to ~0.5 GB with the case arrays it closes over. A
# workstation's scratch disk is 8 GiB and also holds the simulator's results and the prep
# and spectral caches, so the cache keeps a few configurations, not a dozen.
_COMPILE_CACHE_MAX_BYTES = 2 << 30

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
    finds ``libcuda.so.1``. Raises with a plain diagnosis when CUDA was asked for
    but no driver library can be found anywhere: jax's own message in that
    situation ("Backend 'cuda' is not in the list of known backends") hides the
    cause, which is the container's library path, not jax.

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
        "CUDA was requested but the NVIDIA driver library libcuda.so.1 is not visible to "
        "this process: it is neither on the loader path nor under "
        f"{', '.join(_DRIVER_DIRS)}. On GKE the driver is mounted at /usr/local/nvidia/lib64 "
        "and the container must carry it in LD_LIBRARY_PATH; on other hosts install the "
        "NVIDIA driver or run on the CPU (JAX_PLATFORMS=cpu)."
    )


def _driver_loaded() -> bool:
    try:
        ctypes.CDLL(_DRIVER_SONAME)
        return True
    except OSError:
        return False


def platforms_for_host(max_cpu_devices: int = 8) -> str:
    """The ``JAX_PLATFORMS`` value for this host: the caller's own when set, else the
    installed accelerator ahead of the CPU (:func:`default_platforms`). On a host with no
    accelerator the XLA flag that splits the CPU into up to ``max_cpu_devices`` devices is
    set too, so the data-parallel mesh has something to shard over."""
    forced = os.environ.get("JAX_PLATFORMS")
    if forced:
        return forced
    platforms = default_platforms()
    if platforms == "cpu":
        n = max(1, min(max_cpu_devices, os.cpu_count() or 1))
        flags = os.environ.get("XLA_FLAGS", "")
        if "--xla_force_host_platform_device_count" not in flags:
            os.environ["XLA_FLAGS"] = f"{flags} --xla_force_host_platform_device_count={n}".strip()
    return platforms


# user-facing accelerator name -> (JAX platform name, the plugin modules that provide it)
_ACCELERATORS: dict[str, tuple[str, tuple[str, ...]]] = {
    "cuda": ("cuda", ("jax_plugins.xla_cuda12", "jax_plugins.xla_cuda13")),
    "xpu": ("oneapi", ("jax_plugins.xla_oneapi",)),
}
_ALIASES = {"oneapi": "xpu", "gpu": "cuda"}   # JAX's own names, accepted in place of ours


def _find_spec(name: str):
    """``importlib.util.find_spec`` that returns ``None`` when a parent package is absent too."""
    try:
        return importlib.util.find_spec(name)
    except ModuleNotFoundError:     # find_spec imports the parent, which raises instead of None
        return None


def _plugin_installed(modules: tuple[str, ...]) -> bool:
    return any(_find_spec(m) is not None for m in modules)


def installed_accelerators() -> list[str]:
    """The accelerator names (``"cuda"``, ``"xpu"``) whose JAX plugin this environment has."""
    return [name for name, (_, mods) in _ACCELERATORS.items() if _plugin_installed(mods)]


def default_platforms() -> str:
    """``JAX_PLATFORMS`` for ``"auto"``: the installed accelerator ahead of the CPU, else the CPU."""
    found = installed_accelerators()
    return f"{_ACCELERATORS[found[0]][0]},cpu" if found else "cpu"


def normalize_platforms(platforms: str) -> str:
    r"""
    Translate an accelerator choice into a ``JAX_PLATFORMS`` value.

    ``"auto"`` resolves through :func:`default_platforms`; a bare accelerator name
    (``"cuda"``, ``"xpu"``, or JAX's own ``"oneapi"``/``"gpu"``) is put ahead of the CPU;
    any comma-separated list is passed through token by token with the same renaming,
    so ``"xpu,cpu"`` and ``"cpu"`` mean what they say. Raises :class:`RuntimeError` when
    the chosen accelerator's plugin is not installed, naming the ``jax[...]`` extra.
    """
    platforms = platforms.strip().lower()
    if platforms == "auto":
        return default_platforms()
    tokens = [_ALIASES.get(t.strip(), t.strip()) for t in platforms.split(",") if t.strip()]
    if len(tokens) == 1 and tokens[0] in _ACCELERATORS:
        tokens.append("cpu")
    out = []
    for tok in tokens:
        if tok in _ACCELERATORS:
            platform, mods = _ACCELERATORS[tok]
            if not _plugin_installed(mods):
                extra = {"cuda": "jax[cuda12]", "xpu": "jax[oneapi]"}[tok]
                have = installed_accelerators()
                alt = repr(have[0]) if have else "'cpu'"
                raise RuntimeError(
                    f"accelerator {tok!r} was requested but {extra} is not installed in this "
                    f"environment (it has {alt}). Install it, or pick 'auto' / {alt}.")
            tok = platform
        out.append(tok)
    return ",".join(out)


def _cuda_requested(platforms: str) -> bool:
    return "cuda" in platforms.lower()


def _xpu_requested(platforms: str) -> bool:
    return "oneapi" in platforms.lower()


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
        The accelerator: ``"auto"`` (whichever of ``jax[cuda12]`` / ``jax[oneapi]`` is
        installed, else the CPU), ``"cuda"`` (NVIDIA), ``"xpu"`` (Intel oneAPI),
        ``"cpu"``, or a raw ``JAX_PLATFORMS`` list such as ``"xpu,cpu"``; see
        :func:`normalize_platforms`. When ``None`` an existing ``JAX_PLATFORMS`` value is
        respected (e.g. the CPU-pinned test suite) and ``"auto"`` is used otherwise.
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
        os.environ["JAX_PLATFORMS"] = normalize_platforms(platforms)
    elif os.environ.get("JAX_PLATFORMS"):
        os.environ["JAX_PLATFORMS"] = normalize_platforms(os.environ["JAX_PLATFORMS"])
    else:
        os.environ["JAX_PLATFORMS"] = default_platforms()
    effective_platforms = os.environ["JAX_PLATFORMS"]

    if _xpu_requested(effective_platforms):
        # An Arc iGPU draws its memory from the host's: grow the pool on demand rather than
        # taking about 1 GB of it up front.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    if _cuda_requested(effective_platforms):
        ensure_driver_visible()
        # CUDA 13 no longer assembles code for the V100/T4 compute capabilities; select the
        # CUDA 12 compiler shipped by the pinned jax[cuda12] environment.
        cuda_nvcc_spec = _find_spec("nvidia.cuda_nvcc")
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


def use_compilation_cache(cfg) -> Path | None:
    r"""
    Point JAX's persistent compilation cache at ``<work_dir>/jax_cache``.

    Every jitted program of a run closes over its deck's arrays, so its compiled form is
    specific to the deck; keeping the cache in the deck's work directory lets a later job
    on the same deck load the executables — above all the ENGD step, which takes minutes
    to compile at Norne scale — instead of compiling them again. The
    cache is keyed by the compiled program itself (its HLO, compile options and the
    backend), so a changed configuration simply misses it; only programs that took at
    least ``jax_persistent_cache_min_compile_time_secs`` (1 s by default) are written,
    and XLA's GPU autotuning results are kept beside them. An executable carries the
    case arrays its program closes over (a NORNE ENGD step is about 0.5 GB on disk), so
    the directory is capped at ``_COMPILE_CACHE_MAX_BYTES`` with least-recently-used
    eviction unless ``JAX_COMPILATION_CACHE_MAX_SIZE`` sets another limit.

    Switching directories mid-process re-initializes JAX's cache on the next compile. A
    directory the user configured explicitly (``JAX_COMPILATION_CACHE_DIR``) is left as it
    is, and nothing is created while the cache is switched off
    (``JAX_ENABLE_COMPILATION_CACHE=false``, as the test suite runs). Returns the directory
    in use, or ``None`` when there is none.
    """
    import jax

    if not jax.config.jax_enable_compilation_cache:
        return None
    if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        return Path(os.environ["JAX_COMPILATION_CACHE_DIR"])
    if not getattr(cfg, "work_dir", ""):
        return None
    path = Path(cfg.work_dir) / "jax_cache"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    if jax.config.jax_compilation_cache_dir != str(path):
        jax.config.update("jax_compilation_cache_dir", str(path))
        if not os.environ.get("JAX_COMPILATION_CACHE_MAX_SIZE"):
            jax.config.update("jax_compilation_cache_max_size", _COMPILE_CACHE_MAX_BYTES)
        try:
            from jax._src import compilation_cache

            compilation_cache.reset_cache()      # takes the new directory on the next compile
        except (ImportError, AttributeError):
            pass
    return path


def env_summary() -> dict:
    """Small diagnostic dict for the notebook's bootstrap cell (requires jax imported)."""
    import jax

    devices = jax.devices()
    return {
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "platforms": os.environ.get("JAX_PLATFORMS", ""),
        "device_count": jax.device_count(),
        "devices": [d.device_kind for d in devices],
        "x64": bool(jax.config.jax_enable_x64),
        "precision_policy": _CONFIGURED_POLICY,
    }
