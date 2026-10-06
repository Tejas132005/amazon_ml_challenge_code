"""Source-1 entity-level train/validation split.

Every Source-1 entity (and therefore every one of its true pairs) is in exactly
one fold. Blocking uses no labels, so the split only matters for model fitting
and evaluation; it is created once and reused by every later phase.
"""
from __future__ import annotations

import logging

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import CFG, Config
from .preprocessing import normalized_path

LOG = logging.getLogger("ber.validation")


def split_path(cfg: Config = CFG):
    return cfg.artifacts_dir / "splits" / "train_split.parquet"


def make_split(cfg: Config = CFG, overwrite: bool = False) -> pa.Table:
    """Assign each train Source-1 row to fold 'train' or 'valid' (seeded, reproducible)."""
    path = split_path(cfg)
    if path.exists() and not overwrite:
        LOG.info("split exists: %s (use --overwrite to rebuild)", path)
        return pq.read_table(path)
    ids = pq.read_table(normalized_path("train", 1, cfg), columns=["entity_id", "row"])
    order = np.argsort(ids["entity_id"].to_numpy(zero_copy_only=False), kind="stable")  # id-sorted: file-order independent
    rng = np.random.default_rng(cfg.random_seed)
    perm = order[rng.permutation(len(order))]
    n_valid = int(round(cfg.valid_fraction * len(order)))
    fold = np.full(len(order), "train", dtype=object)
    fold[perm[:n_valid]] = "valid"
    out = pa.table({"row": ids["row"], "entity_id": ids["entity_id"], "fold": pa.array(fold, pa.string())})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, path)
    LOG.info("split written: %s (valid=%d, train=%d)", path, n_valid, len(order) - n_valid)
    return out


def fold_rows(fold: str, cfg: Config = CFG) -> np.ndarray:
    t = pq.read_table(split_path(cfg))
    mask = np.asarray(t["fold"].to_numpy(zero_copy_only=False) == fold)
    return np.sort(t["row"].to_numpy()[mask])
