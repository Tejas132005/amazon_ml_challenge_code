"""Small shared helpers: logging, memory/timing instrumentation, JSON output."""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import psutil

LOG = logging.getLogger("ber")


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                        handlers=handlers, force=True)


def rss_mb() -> float:
    """Current resident memory of this process in MB."""
    return psutil.Process().memory_info().rss / 2**20


def peak_rss_mb() -> float | None:
    """Peak memory in MB (Windows: peak working set; Unix: ru_maxrss)."""
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)
    if peak is not None:
        return peak / 2**20
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux reports KB
    except ImportError:
        return None


@contextmanager
def timed(stage: str, sink: dict[str, Any] | None = None) -> Iterator[None]:
    """Log wall time and memory for a stage; optionally record into ``sink``."""
    t0 = time.perf_counter()
    LOG.info("[start] %s (rss=%.0f MB)", stage, rss_mb())
    yield
    dt = time.perf_counter() - t0
    peak = peak_rss_mb()
    LOG.info("[done]  %s in %.1fs (rss=%.0f MB, peak=%s MB)", stage, dt, rss_mb(),
             f"{peak:.0f}" if peak else "n/a")
    if sink is not None:
        sink[stage] = {"seconds": round(dt, 2), "rss_mb": round(rss_mb()),
                       "peak_rss_mb": round(peak) if peak else None}


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
