"""The pre-jax bootstrap: the driver-visibility guard that turns "Backend 'cuda'
is not in the list of known backends" into a diagnosis, or fixes it."""
import ctypes.util
import os
import shutil
from pathlib import Path

import pytest

from modules import bootstrap


def test_driver_already_on_the_loader_path_needs_no_preload(monkeypatch):
    # libm is always findable: the guard is a no-op and touches nothing.
    monkeypatch.setattr(bootstrap, "_DRIVER_SONAME", "libm.so.6")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    assert bootstrap.ensure_driver_visible() is None
    assert os.environ["LD_LIBRARY_PATH"] == ""


def test_driver_mounted_off_the_loader_path_is_preloaded(tmp_path, monkeypatch):
    """GKE's situation: the library exists under /usr/local/nvidia/lib64 (here a
    temp dir) but the loader cannot find it by name."""
    real = ctypes.util.find_library("m")
    src = Path(f"/lib/x86_64-linux-gnu/{real}") if real and not real.startswith("/") else Path(real or "")
    if not src.is_file():
        pytest.skip("no libm to stand in for the driver")
    soname = "libperdtest-driver.so.1"
    mount = tmp_path / "nvidia" / "lib64"
    mount.mkdir(parents=True)
    shutil.copy(src, mount / soname)
    monkeypatch.setattr(bootstrap, "_DRIVER_SONAME", soname)
    monkeypatch.setattr(bootstrap, "_DRIVER_DIRS", (str(mount),))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/other/lib")
    assert bootstrap.ensure_driver_visible() == str(mount / soname)
    assert os.environ["LD_LIBRARY_PATH"] == f"{mount}:/opt/other/lib"


def test_missing_driver_is_diagnosed_plainly(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "_DRIVER_SONAME", "libperdtest-absent.so.1")
    monkeypatch.setattr(bootstrap, "_DRIVER_DIRS", (str(tmp_path),))
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    with pytest.raises(RuntimeError) as exc:
        bootstrap.ensure_driver_visible()
    assert "LD_LIBRARY_PATH" in str(exc.value) and "JAX_PLATFORMS=cpu" in str(exc.value)
