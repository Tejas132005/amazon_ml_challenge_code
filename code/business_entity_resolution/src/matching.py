"""Turn pair probabilities into per-entity match sets (Phase 7).

Every rule may return an EMPTY set -- nothing forces an entity to match.

threshold      keep pairs with p >= t
expected_f     per entity, choose the prefix of candidates (sorted by p) that maximises an
               estimate of E[F0.5]; compare with predicting nothing, whose expected score is
               P(singleton) ~ prod(1 - p_i) scaled by ``empty_scale``:
                   E[F | top-k] ~ 1.25 * sum_{i<=k} p_i / (k + 0.25 * E[n_true]),
                   E[n_true] = sum_i p_i + ``missing`` (true matches outside the candidates)
exclusive      optional post-step: each target keeps only its highest-p Source-1 entity
               (EDA: no train target is matched to two Source-1 entities)

Probabilities come from a model trained on the real candidate distribution (no negative
sub-sampling), so they need no re-weighting before these rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def threshold_rule(p: np.ndarray, t: float) -> np.ndarray:
    return p >= t


def expected_f_rule(s1_row: np.ndarray, p: np.ndarray, empty_scale: float = 1.0, missing: float = 0.0,
                    p_floor: float = 0.0) -> np.ndarray:
    """Vectorised expected-F0.5 set selection (segment ops over entity-sorted pairs)."""
    order = np.lexsort((-p, s1_row))
    r, q = s1_row[order], p[order].astype(np.float64)
    new = np.concatenate([[True], r[1:] != r[:-1]])
    starts = np.flatnonzero(new)
    counts = np.diff(np.concatenate([starts, [len(r)]]))
    seg = np.repeat(np.arange(len(starts)), counts)
    k = np.arange(len(r)) - starts[seg] + 1
    cum = np.cumsum(q)
    cum -= np.repeat(cum[starts] - q[starts], counts)            # within-entity cumulative sum
    total = np.add.reduceat(q, starts)[seg]
    e_f = 1.25 * cum / (k + 0.25 * (total + missing))
    e_empty = np.exp(np.add.reduceat(np.log1p(-np.clip(q, 0, 1 - 1e-7)), starts))[seg] * empty_scale
    best_f = np.maximum.reduceat(e_f, starts)[seg]
    best_k = np.minimum.reduceat(np.where(e_f >= best_f, k, np.iinfo(np.int64).max), starts)[seg]
    choose = (k <= best_k) & (best_f > e_empty) & (q >= p_floor)
    mask = np.zeros(len(p), dtype=bool)
    mask[order[choose]] = True
    return mask


def exclusive(s1_row: np.ndarray, tgt: np.ndarray, p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Among selected pairs, each target is kept only for its highest-probability Source-1 entity."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return mask
    order = idx[np.lexsort((s1_row[idx], -p[idx], tgt[idx]))]
    first = np.concatenate([[True], tgt[order][1:] != tgt[order][:-1]])
    out = np.zeros_like(mask)
    out[order[first]] = True
    return out


def expected_f_chunked(s1_row: np.ndarray, p: np.ndarray, empty_scale: float = 1.0, missing: float = 0.0,
                       p_floor: float = 0.0, chunk_pairs: int = 2_000_000) -> np.ndarray:
    """``expected_f_rule`` on entity-aligned chunks: identical result, bounded memory.

    The rule is independent per Source-1 entity, so pairs are grouped by entity and processed
    ~``chunk_pairs`` at a time (never splitting an entity). Peak extra memory is ~12 arrays of
    chunk size instead of ~12 arrays of the full pair count (~2.6 GB for 28M pairs).
    """
    n = len(p)
    if n <= chunk_pairs:
        return expected_f_rule(s1_row, p, empty_scale, missing, p_floor)
    order = None if np.all(s1_row[1:] >= s1_row[:-1]) else np.argsort(s1_row, kind="stable")
    rows = s1_row if order is None else s1_row[order]
    probs = p if order is None else p[order]
    starts = np.flatnonzero(np.concatenate([[True], rows[1:] != rows[:-1]]))
    out = np.zeros(n, dtype=bool)
    lo = 0
    while lo < n:
        j = np.searchsorted(starts, lo + chunk_pairs, side="left")
        hi = int(starts[j]) if j < len(starts) else n
        out[lo:hi] = expected_f_rule(rows[lo:hi], probs[lo:hi], empty_scale, missing, p_floor)
        lo = hi
    if order is None:
        return out
    mask = np.zeros(n, dtype=bool)
    mask[order[out]] = True
    return mask


def decide(s1_row: np.ndarray, tgt: np.ndarray, p: np.ndarray, params: dict) -> np.ndarray:
    """Apply a decision rule described by ``params`` (as stored in the tuned-params JSON)."""
    if params["rule"] == "threshold":
        mask = threshold_rule(p, params["t"])
    elif params["rule"] == "expected_f":
        mask = expected_f_chunked(s1_row, p, params.get("empty_scale", 1.0), params.get("missing", 0.0),
                                  params.get("p_floor", 0.0))
    else:
        raise ValueError(f"unknown rule {params['rule']}")
    if params.get("exclusive"):
        mask = exclusive(s1_row, tgt, p, mask)
    return mask
