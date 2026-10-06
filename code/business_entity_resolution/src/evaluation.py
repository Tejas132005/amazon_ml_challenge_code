"""Evaluation utilities. Phase 3: candidate-set quality against held-out labels.

Definitions (per evaluated Source-1 entity e with true set T_e and candidates C_e):
  pair recall            sum|T_e & C_e| / sum|T_e|
  entity full recall     share of non-singleton e with T_e subset of C_e
  oracle macro F0.5      macro F0.5 of a perfect matcher restricted to C_e
                         (singleton -> 1.0; else precision 1, recall |T_e & C_e|/|T_e|)
                         = the ceiling any model can reach with this candidate set
  reduction ratio        1 - sum|C_e| / (n_eval * (|S2| + |S3|))
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .blocking import PASSES
from .config import CFG, TARGET_SOURCES, Config
from .data_loader import explode_ground_truth, load_ground_truth
from .preprocessing import normalized_path

LOG = logging.getLogger("ber.evaluation")


def f05(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    num = 1.25 * precision * recall
    den = 0.25 * precision + recall
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def gt_row_pairs(cfg: Config = CFG) -> pa.Table:
    """Train ground truth as integer rows: s1_row (S1 file row), tgt (S2 rows, then S3 rows offset by |S2|)."""
    gt, _ = load_ground_truth("train", cfg)
    pairs = explode_ground_truth(gt)
    s1_ids = pq.read_table(normalized_path("train", 1, cfg), columns=["entity_id"])["entity_id"]
    tgt_ids = pa.chunked_array([c for s in TARGET_SOURCES for c in
                                pq.read_table(normalized_path("train", s, cfg), columns=["entity_id"])["entity_id"].chunks])
    s1_row = pc.index_in(pairs["s1_id"], value_set=s1_ids)
    tgt = pc.index_in(pairs["tgt_id"], value_set=tgt_ids)
    if s1_row.null_count or tgt.null_count:
        raise ValueError("ground-truth ids missing from normalised sources")
    return pa.table({"s1_row": pc.cast(s1_row, pa.int32()), "tgt": pc.cast(tgt, pa.int32()), "src": pairs["src"]})


def candidate_report(cands: pa.Table, eval_rows: np.ndarray, truth: pa.Table, n_targets: int,
                     ks: tuple[int, ...], s1_country: np.ndarray | None = None,
                     rank_col: str = "rank") -> tuple[pd.DataFrame, dict]:
    """Recall/size curve over K (per-source top-K by ``rank_col``) plus per-pass and per-country
    breakdowns at the largest K. A final row 'stored' uses every stored candidate (the ceiling)."""
    eval_rows = np.asarray(eval_rows, dtype=np.int64)
    n_eval = len(eval_rows)
    t = truth.filter(pc.is_in(truth["s1_row"], value_set=pa.array(eval_rows)))
    t_row, t_tgt = t["s1_row"].to_numpy().astype(np.int64), t["tgt"].to_numpy().astype(np.int64)
    t_key = t_row * n_targets + t_tgt
    pos = np.searchsorted(eval_rows, t_row)  # entity index 0..n_eval-1
    n_true = np.bincount(pos, minlength=n_eval)
    nonsingle = n_true > 0

    c_row = cands["s1_row"].to_numpy().astype(np.int64)
    c_tgt = cands["tgt"].to_numpy().astype(np.int64)
    c_rank = cands[rank_col].to_numpy()
    c_pass = cands["passes"].to_numpy()
    c_key = c_row * n_targets + c_tgt
    c_pos = np.searchsorted(eval_rows, c_row)

    rows = []
    for k in list(ks) + ["stored"]:
        m = c_rank < k if k != "stored" else np.ones(len(c_rank), bool)
        found = np.isin(t_key, c_key[m])
        hit = np.bincount(pos, weights=found, minlength=n_eval)
        size = np.bincount(c_pos[m], minlength=n_eval)
        rec_e = np.divide(hit, n_true, out=np.zeros(n_eval), where=nonsingle)
        oracle = np.where(nonsingle, f05(np.ones(n_eval), rec_e), 1.0)
        rows.append({
            "k_per_source": k, "pair_recall": found.mean(), "entity_full_recall": (hit[nonsingle] == n_true[nonsingle]).mean(),
            "entity_zero_recall": (hit[nonsingle] == 0).mean(), "oracle_macro_f05": oracle.mean(),
            "mean_cands": size.mean(), "median_cands": float(np.median(size)), "p95_cands": float(np.quantile(size, 0.95)),
            "max_cands": int(size.max()), "zero_cand_share": (size == 0).mean(),
            "reduction_ratio": 1 - size.sum() / (n_eval * n_targets)})
    curve = pd.DataFrame(rows)

    m = np.ones(len(c_rank), bool)  # breakdowns use every stored candidate (blocking ceiling)
    order = np.argsort(c_key[m])
    ck, cp = c_key[m][order], c_pass[m][order]
    loc = np.minimum(np.searchsorted(ck, t_key), max(len(ck) - 1, 0))
    found = (len(ck) > 0) & (ck[loc] == t_key) if len(ck) else np.zeros(len(t_key), bool)
    tp = np.where(found, cp[loc] if len(ck) else 0, 0)
    per_pass = {}
    for p in PASSES:
        bit = 1 << PASSES.index(p)
        has = (tp & bit) > 0
        only = has & (tp == bit)
        per_pass[p] = {"recall_contains": round(float(has.mean()), 4), "recall_only_this_pass": round(float(only.mean()), 4)}
    extra = {"n_eval_entities": n_eval, "breakdowns_use": "all stored candidates", "n_true_pairs": int(len(t_key)), "singleton_share": float((~nonsingle).mean()),
             "per_pass_at_kmax": per_pass}
    if s1_country is not None:
        byc = {}
        ent_country = s1_country
        for c in np.unique(ent_country):
            sel = ent_country[pos] == c
            byc[str(c)] = {"true_pairs": int(sel.sum()), "pair_recall_stored": round(float(found[sel].mean()), 4)}
        extra["by_country_at_kmax"] = byc
    extra["missed_pairs_at_kmax"] = pa.table({"s1_row": pa.array(t_row[~found]), "tgt": pa.array(t_tgt[~found])})
    return curve, extra


def _examples(missed: pa.Table, n2: int, cfg: Config, limit: int = 300) -> pd.DataFrame:
    """Raw + normalised fields of missed true pairs for manual inspection."""
    missed = missed.slice(0, limit)
    cols = ["entity_id", "business_name", "business_address", "name_core", "addr_core", "addr_numbers"]
    s1 = pq.read_table(normalized_path("train", 1, cfg), columns=cols).take(missed["s1_row"]).to_pandas().add_prefix("s1_")
    tg = missed["tgt"].to_numpy().astype(np.int64)
    parts = []
    for s, sel, off in ((2, tg < n2, 0), (3, tg >= n2, n2)):
        t = pq.read_table(normalized_path("train", s, cfg), columns=cols)
        parts.append((np.flatnonzero(sel), t.take(pa.array(tg[sel] - off)).to_pandas()))
    t_df = pd.concat([df.set_index(pd.Index(i)) for i, df in parts]).sort_index().add_prefix("t_")
    return pd.concat([s1.reset_index(drop=True), t_df.reset_index(drop=True)], axis=1)


def run_blocking_eval(cfg: Config = CFG, fold: str = "valid", passes: tuple[str, ...] | None = None,
                      tag: str | None = None) -> pd.DataFrame:
    """Generate candidates for (a sample of) a train fold and report recall vs candidate-set size."""
    from .candidate_generation import generate_candidates
    from .utils import save_json
    from .validation import fold_rows

    rows = fold_rows(fold, cfg)
    if cfg.blocking_eval_max_entities and len(rows) > cfg.blocking_eval_max_entities:
        rng = np.random.default_rng(cfg.random_seed)
        rows = np.sort(rng.choice(rows, size=cfg.blocking_eval_max_entities, replace=False))
    name = f"train_{fold}" + (f"_{tag}" if tag else "")
    path, meta = generate_candidates("train", name, rows, cfg, passes, store_k=cfg.blocking_union_k)
    cands = pq.read_table(path)
    truth = gt_row_pairs(cfg)
    country = pq.read_table(normalized_path("train", 1, cfg), columns=["country"])["country"].to_numpy(
        zero_copy_only=False)[rows]
    curve, extra = candidate_report(cands, rows, truth, meta["n_targets"], cfg.blocking_eval_ks, country, "rank")
    curve.insert(0, "ranking", "rerank" if cfg.blocking_rerank else "key_score")
    if cfg.blocking_rerank:  # same union, ordered by key score only -> shows what re-ranking adds
        key_curve, _ = candidate_report(cands, rows, truth, meta["n_targets"], cfg.blocking_eval_ks, None, "key_rank")
        key_curve.insert(0, "ranking", "key_score")
        curve = pd.concat([key_curve, curve], ignore_index=True)
    out_dir = cfg.artifacts_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    curve.round(5).to_csv(out_dir / f"blocking_curve_{name}.csv", index=False)
    missed = extra.pop("missed_pairs_at_kmax")
    _examples(missed, meta["n2"], cfg).to_csv(out_dir / f"blocking_missed_{name}.tsv", sep="\t", index=False)
    save_json({**meta, **extra}, out_dir / f"blocking_report_{name}.json")
    LOG.info("blocking curve (%s):\n%s", name, curve.round(4).to_string(index=False))
    LOG.info("union size per entity: %s", meta["union_size_per_entity_before_cuts"])
    LOG.info("per pass (stored set): %s", extra["per_pass_at_kmax"])
    LOG.info("by country: %s", extra.get("by_country_at_kmax"))
    return curve


def report_final_sets(cfg: Config = CFG) -> pd.DataFrame:
    """Recall / size of the FINAL candidate sets (what the matcher sees), for train_fit and train_valid."""
    from .candidate_generation import candidates_path, queried_rows
    truth = gt_row_pairs(cfg)
    n_t = sum(pq.ParquetFile(normalized_path("train", s, cfg)).metadata.num_rows for s in TARGET_SOURCES)
    parts = []
    for name in ("train_fit", "train_valid"):
        curve, _ = candidate_report(pq.read_table(candidates_path(name, cfg)), queried_rows(name, cfg), truth,
                                    n_t, (cfg.blocking_top_k,))
        curve.insert(0, "set", name)
        parts.append(curve.iloc[[0]])
    out = pd.concat(parts, ignore_index=True)
    out_dir = cfg.artifacts_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.round(5).to_csv(out_dir / "final_candidates_report.csv", index=False)
    LOG.info("final candidate sets (K=%d per source):\n%s", cfg.blocking_top_k, out.round(4).to_string(index=False))
    return out


# ============================================================================ official metric
def macro_f05(pred_rows: np.ndarray, pred_tgt: np.ndarray, eval_rows: np.ndarray, truth: pa.Table,
              n_targets: int, groups: dict[str, np.ndarray] | None = None) -> dict:
    """Challenge metric: F0.5 per Source-1 entity, averaged over ALL ``eval_rows``.

    Per entity with true set T and predicted set P (duplicates in P are removed first):
      T empty: 1.0 if P empty else 0.0
      T non-empty: 0.0 if |P & T| == 0 (incl. P empty), else 1.25*prec*rec / (0.25*prec + rec)
    ``groups`` (optional): name -> per-entity label array aligned with eval_rows, for breakdowns.
    """
    eval_rows = np.asarray(eval_rows, dtype=np.int64)
    n = len(eval_rows)
    t = truth.filter(pc.is_in(truth["s1_row"], value_set=pa.array(eval_rows)))
    t_row, t_tgt = t["s1_row"].to_numpy().astype(np.int64), t["tgt"].to_numpy().astype(np.int64)
    t_key = np.unique(t_row * n_targets + t_tgt)
    p_key = np.unique(np.asarray(pred_rows, np.int64) * n_targets + np.asarray(pred_tgt, np.int64))
    p_row = p_key // n_targets
    if len(p_row) and not np.isin(p_row, eval_rows).all():
        raise ValueError("predictions for entities outside eval_rows")
    ent = lambda rows: np.searchsorted(eval_rows, rows)  # noqa: E731
    n_true = np.bincount(ent(t_key // n_targets), minlength=n)
    n_pred = np.bincount(ent(p_row), minlength=n)
    hit = np.isin(p_key, t_key)
    tp = np.bincount(ent(p_row[hit]), minlength=n)
    prec = np.divide(tp, n_pred, out=np.zeros(n), where=n_pred > 0)
    rec = np.divide(tp, n_true, out=np.zeros(n), where=n_true > 0)
    f = np.where(n_true == 0, (n_pred == 0).astype(float), f05(prec, rec))
    single = n_true == 0
    res = {"macro_f05": float(f.mean()), "n_entities": int(n), "singleton_share": float(single.mean()),
           "f05_singletons": float(f[single].mean()) if single.any() else None,
           "f05_non_singletons": float(f[~single].mean()) if (~single).any() else None,
           "micro_precision": float(tp.sum() / max(n_pred.sum(), 1)), "micro_recall": float(tp.sum() / max(n_true.sum(), 1)),
           "tp": int(tp.sum()), "fp": int(n_pred.sum() - tp.sum()), "fn": int(n_true.sum() - tp.sum()),
           "singletons_with_false_match": int((single & (n_pred > 0)).sum()),
           "non_singletons_predicted_empty": int((~single & (n_pred == 0)).sum()),
           "mean_pred_per_entity": float(n_pred.mean())}
    for gname, labels in (groups or {}).items():
        for g in np.unique(labels):
            m = labels == g
            res[f"macro_f05[{gname}={g}]"] = float(f[m].mean())
    return res
