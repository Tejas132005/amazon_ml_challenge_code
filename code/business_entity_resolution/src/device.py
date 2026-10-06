"""Hardware selection for LightGBM training: use a GPU when one works, otherwise the CPU.

How detection works
-------------------
"A GPU exists" is not enough: LightGBM also needs a build that supports the device and a
working driver/runtime. So instead of guessing from hardware lists, we *try* each device:
a tiny LightGBM training run (2 rounds on random data) in a separate Python process, in the
order  cuda -> gpu -> cpu.  The first device that trains successfully is used.

  cuda  needs a LightGBM build with USE_CUDA (source build, Linux only; the PyPI wheel lacks it)
  gpu   OpenCL; needs a LightGBM build with USE_GPU. Measured: the Linux x86_64 PyPI wheel of
        lightgbm 4.7.0 has it, the Windows wheel (lightgbm-4.7.0-py3-none-win_amd64.whl) does
        NOT ("Please recompile with CMake option -DUSE_GPU=1") -- on Windows a source build
        with Boost + OpenCL is required
  cpu   always available

The probe runs in a subprocess so a broken driver (crash or hang) cannot take down the
pipeline; it has a timeout and the result is cached for the life of the process.

Override with ``--device {auto,cpu,gpu,cuda}`` on ``train``, the ``BER_LGBM_DEVICE`` env var,
or ``Config.lgbm_device``. Forcing a device that does not work raises a clear error instead of
silently falling back, so a forced run never quietly changes hardware.

Scope: only LightGBM *training* can use the GPU. ``Booster.predict`` is CPU-only in LightGBM,
and the other stages (Arrow, NumPy, RapidFuzz, scikit-learn) are CPU libraries.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from functools import lru_cache

from .config import CFG, Config

LOG = logging.getLogger("ber.device")

DEVICES = ("cuda", "gpu", "cpu")
_CPU_ONLY_PARAMS = ("force_row_wise", "force_col_wise")  # histogram-layout hints for the CPU learner

_PROBE = """
import sys, numpy as np, lightgbm as lgb
dev, plat, did = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(0)
X = rng.random((2000, 8)); y = (X[:, 0] > 0.5).astype(int)
params = {"objective": "binary", "device_type": dev, "verbose": -1, "num_threads": 1}
if dev == "gpu":
    params.update(gpu_platform_id=plat, gpu_device_id=did, max_bin=63)
lgb.train(params, lgb.Dataset(X, y), num_boost_round=2)
print("OK")
"""


def probe_device(device: str, cfg: Config = CFG) -> tuple[bool, str]:
    """Try a tiny LightGBM training run on ``device`` in a subprocess. Returns (works, detail)."""
    if device == "cpu":
        return True, "cpu always available"
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE, device, str(cfg.lgbm_gpu_platform_id),
                            str(cfg.lgbm_gpu_device_id)],
                           capture_output=True, text=True, timeout=cfg.lgbm_probe_timeout_s)
    except subprocess.TimeoutExpired:
        return False, f"probe timed out after {cfg.lgbm_probe_timeout_s}s"
    if r.returncode == 0 and "OK" in r.stdout:
        return True, "probe training succeeded"
    lines = (r.stderr or r.stdout).strip().splitlines()
    detail = lines[-1] if lines else f"exit code {r.returncode}"
    if "recompile" in detail.lower():
        detail = f"installed LightGBM build has no {device} support ({detail})"
    return False, detail


@lru_cache(maxsize=None)
def _resolve(requested: str, plat: int, did: int, timeout: int) -> tuple[str, tuple[tuple[str, str], ...]]:
    cfg = Config(lgbm_gpu_platform_id=plat, lgbm_gpu_device_id=did, lgbm_probe_timeout_s=timeout)
    order = DEVICES if requested == "auto" else (requested,)
    tried = []
    for dev in order:
        ok, detail = probe_device(dev, cfg)
        tried.append((dev, detail))
        if ok:
            return dev, tuple(tried)
    return "", tuple(tried)


def select_lgbm_device(cfg: Config = CFG) -> str:
    """Return the LightGBM device to train on: 'cuda', 'gpu' or 'cpu'."""
    requested = (cfg.lgbm_device or "auto").lower()
    if requested not in ("auto",) + DEVICES:
        raise ValueError(f"lgbm_device must be one of auto/cuda/gpu/cpu, got {cfg.lgbm_device!r}")
    device, tried = _resolve(requested, cfg.lgbm_gpu_platform_id, cfg.lgbm_gpu_device_id, cfg.lgbm_probe_timeout_s)
    for dev, detail in tried:
        LOG.info("LightGBM device probe: %-4s -> %s", dev, detail)
    if not device:
        raise RuntimeError(f"requested LightGBM device {requested!r} is not usable: {dict(tried)}. "
                           "Use --device auto (falls back to CPU) or fix the GPU driver / LightGBM build.")
    LOG.info("LightGBM training device: %s%s", device, "" if requested == "auto" else " (forced)")
    return device


def lgbm_train_params(cfg: Config = CFG, device: str | None = None) -> dict:
    """``cfg.lgbm_params`` adapted to the selected device (plus the seed)."""
    device = device or select_lgbm_device(cfg)
    params = {**cfg.lgbm_params, "seed": cfg.random_seed, "device_type": device}
    if device in ("gpu", "cuda"):
        for k in _CPU_ONLY_PARAMS:
            params.pop(k, None)
        if device == "gpu":
            params.update(gpu_platform_id=cfg.lgbm_gpu_platform_id, gpu_device_id=cfg.lgbm_gpu_device_id,
                          gpu_use_dp=False)
            if cfg.lgbm_gpu_max_bin is not None:
                params["max_bin"] = cfg.lgbm_gpu_max_bin
    return params
