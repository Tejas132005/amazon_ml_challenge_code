import pytest
from dataclasses import replace

from src import device as dev
from src.config import CFG


@pytest.fixture(autouse=True)
def _clear_cache():
    dev._resolve.cache_clear()
    yield
    dev._resolve.cache_clear()


def test_auto_falls_back_to_cpu_when_gpu_probes_fail(monkeypatch):
    monkeypatch.setattr(dev, "probe_device", lambda d, cfg=CFG: (d == "cpu", "no OpenCL device" if d != "cpu" else "ok"))
    assert dev.select_lgbm_device(replace(CFG, lgbm_device="auto")) == "cpu"


def test_auto_prefers_gpu_when_it_works(monkeypatch):
    monkeypatch.setattr(dev, "probe_device", lambda d, cfg=CFG: (d in ("gpu", "cpu"), "x"))
    assert dev.select_lgbm_device(replace(CFG, lgbm_device="auto")) == "gpu"


def test_forced_device_that_fails_raises(monkeypatch):
    monkeypatch.setattr(dev, "probe_device", lambda d, cfg=CFG: (False, "no OpenCL device"))
    with pytest.raises(RuntimeError, match="not usable"):
        dev.select_lgbm_device(replace(CFG, lgbm_device="gpu"))


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        dev.select_lgbm_device(replace(CFG, lgbm_device="tpu"))


def test_params_per_device():
    cpu = dev.lgbm_train_params(CFG, "cpu")
    assert cpu["device_type"] == "cpu" and cpu["force_row_wise"] is True and cpu["max_bin"] == CFG.lgbm_params["max_bin"]
    gpu = dev.lgbm_train_params(CFG, "gpu")
    assert gpu["device_type"] == "gpu" and "force_row_wise" not in gpu and gpu["max_bin"] == 63
    assert gpu["gpu_use_dp"] is False and gpu["objective"] == "binary" and gpu["seed"] == CFG.random_seed
    assert dev.lgbm_train_params(replace(CFG, lgbm_gpu_max_bin=None), "gpu")["max_bin"] == 255


def test_real_probe_returns_a_working_device():
    """Runs the real subprocess probe on this machine: whatever it picks must actually work."""
    d = dev.select_lgbm_device(replace(CFG, lgbm_device="auto"))
    assert d in ("cuda", "gpu", "cpu")
    assert dev.probe_device(d)[0]
