"""Test-set inference (Phase 9): score the final test candidates, decide matches, write outputs.

Prerequisites: preprocess/candidates/features for --split test, and a trained model plus
models/decision.json (from ``train`` + ``evaluate``). Only train-fold labels were ever used
to fit or tune anything; the test set is scored once with the frozen model and rule.

Memory: features are streamed in row batches of ``predict_batch_rows``; only ids and one
float per pair are kept (~12 bytes/pair, ~0.35 GB for 28M pairs).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .candidate_generation import candidates_path, queried_rows
from .config import CFG, Config
from .feature_engineering import features_path
from .matching import decide
from .model_training import models_dir
from .preprocessing import normalized_path
from .submission import CAND_HEADER, MATCH_HEADER, check_submission, id_arrays, write_id_lists
from .utils import save_json, timed

LOG = logging.getLogger("ber.inference")


def score_pairs(name: str, cfg: Config = CFG) -> dict[str, np.ndarray]:
    """Stream features/<name>.parquet through the selected model -> ids + probability."""
    import lightgbm as lgb
    import joblib

    decision = json.loads((models_dir(cfg) / "decision.json").read_text(encoding="utf-8"))
    meta = json.loads((models_dir(cfg) / "training_meta.json").read_text(encoding="utf-8"))
    cols, model_name = meta["features"], decision["model"]
    if model_name == "lgbm":
        booster = lgb.Booster(model_file=str(models_dir(cfg) / "lgbm.txt"))
        fn = booster.predict
    elif model_name == "logreg":
        mdl = joblib.load(models_dir(cfg) / "logreg.joblib")
        fn = lambda X: mdl.predict_proba(X)[:, 1]  # noqa: E731
    elif model_name == "rule":
        fn = None
    else:
        raise ValueError(model_name)
    pf = pq.ParquetFile(features_path(name, cfg))
    missing = set(cols) - set(pf.schema_arrow.names)
    if missing:
        raise ValueError(f"features/{name} lacks model columns {sorted(missing)}")
    parts: dict[str, list] = {"s1_row": [], "tgt": [], "p": []}
    done = 0
    for batch in pf.iter_batches(batch_size=cfg.predict_batch_rows, columns=["s1_row", "tgt"] + cols):
        X = np.column_stack([batch.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in cols])
        p = batch.column("rr").to_numpy(zero_copy_only=False) if fn is None else fn(X)
        parts["s1_row"].append(batch.column("s1_row").to_numpy())
        parts["tgt"].append(batch.column("tgt").to_numpy())
        parts["p"].append(np.asarray(p, dtype=np.float32))
        done += batch.num_rows
        LOG.info("  scored %d/%d pairs", done, pf.metadata.num_rows)
    return {k: np.concatenate(v) for k, v in parts.items()} | {"decision": decision["decision"],
                                                              "model": model_name}


def scores_path(name: str, cfg: Config = CFG) -> Path:
    return cfg.artifacts_dir / "scores" / f"{name}.parquet"


def run_inference(cfg: Config = CFG, output_dir: Path | None = None, split: str = "test",
                  name: str = "test", reuse_scores: bool = False) -> dict:
    """Score (or reload cached scores), decide, write and check the submission files.

    Pair scores are saved to artifacts/scores/<name>.parquet right after scoring, so a failure in
    any later step never costs the (long) scoring run again: re-run with ``reuse_scores=True``.
    """
    out_dir = Path(output_dir or cfg.output_dir)
    timings: dict = {}
    cache = scores_path(name, cfg)
    if reuse_scores:
        if not cache.exists():
            raise FileNotFoundError(f"{cache} not found; run predict without --reuse-scores first")
        decision = json.loads((models_dir(cfg) / "decision.json").read_text(encoding="utf-8"))
        t = pq.read_table(cache)
        sc = {"s1_row": t["s1_row"].to_numpy(), "tgt": t["tgt"].to_numpy(), "p": t["p"].to_numpy(),
              "decision": decision["decision"], "model": decision["model"]}
        LOG.info("reusing %d cached pair scores from %s", t.num_rows, cache)
        del t
    else:
        with timed("score test pairs", timings):
            sc = score_pairs(name, cfg)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"s1_row": pa.array(sc["s1_row"].astype(np.int32)),
                                 "tgt": pa.array(sc["tgt"].astype(np.int32)),
                                 "p": pa.array(sc["p"].astype(np.float32))}), cache)
        LOG.info("saved pair scores to %s", cache)
    with timed("decide matches", timings):
        # compact dtypes: 28M pairs x (4 + 4 + 4) bytes instead of 3 x 8 bytes
        rows, tgt, p = sc["s1_row"].astype(np.int32), sc["tgt"].astype(np.int32), sc["p"].astype(np.float32)
        del sc["s1_row"], sc["tgt"], sc["p"]
        mask = decide(rows, tgt, p, sc["decision"])
    s1_ids, tg_ids = id_arrays(split, cfg)
    if not np.array_equal(queried_rows(name, cfg), np.arange(len(s1_ids))):
        raise ValueError("candidate set does not cover every Source-1 entity of the split")

    with timed("write submission files", timings):
        cand = pq.read_table(candidates_path(name, cfg), columns=["s1_row", "tgt", "src", "rank"])
        c_rows, c_tgt = cand["s1_row"].to_numpy(), cand["tgt"].to_numpy()
        c_order = np.lexsort((cand["rank"].to_numpy(), cand["src"].to_numpy(), c_rows))
        n_cand = write_id_lists(out_dir / "candidate_pairs.tsv", CAND_HEADER, s1_ids, c_rows[c_order],
                                c_tgt[c_order], tg_ids)
        m_idx = np.flatnonzero(mask)
        m_idx = m_idx[np.lexsort((-p[m_idx], rows[m_idx]))]
        n_match = write_id_lists(out_dir / "matching_results.tsv", MATCH_HEADER, s1_ids, rows[m_idx],
                                 tgt[m_idx], tg_ids)
    if n_cand != len(rows):
        raise ValueError(f"scored pairs ({len(rows)}) != candidate pairs written ({n_cand})")

    with timed("check submission files", timings):
        issues = check_submission(out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv", split, cfg)
    n_match_ent = np.bincount(rows[mask], minlength=len(s1_ids))
    n_cand_ent = np.bincount(c_rows, minlength=len(s1_ids))
    country = pq.read_table(normalized_path(split, 1, cfg), columns=["country"])["country"].to_numpy(
        zero_copy_only=False)
    by_country = {}
    for c in np.unique(country):
        m = country == c
        by_country[str(c)] = {"entities": int(m.sum()), "zero_candidates": int((n_cand_ent[m] == 0).sum()),
                              "predicted_empty_share": round(float((n_match_ent[m] == 0).mean()), 4),
                              "mean_matches": round(float(n_match_ent[m].mean()), 4)}
    summary = {"model": sc["model"], "decision": sc["decision"], "n_source1": len(s1_ids),
               "candidate_pairs": int(n_cand), "matched_pairs": int(n_match),
               "entities_predicted_empty": int((n_match_ent == 0).sum()),
               "share_predicted_empty": float((n_match_ent == 0).mean()),
               "mean_matches_per_entity": float(n_match_ent.mean()),
               "candidates_per_entity": {"mean": float(n_cand_ent.mean()), "median": float(np.median(n_cand_ent)),
                                         "p95": float(np.quantile(n_cand_ent, 0.95)), "max": int(n_cand_ent.max())},
               "by_country": by_country, "internal_check_issues": issues, "timings": timings}
    save_json(summary, cfg.artifacts_dir / "reports" / "test_prediction_summary.json")
    if issues:
        LOG.error("submission check FAILED:\n  %s", "\n  ".join(issues))
    else:
        LOG.info("submission check passed (internal). Files in %s", out_dir)
    LOG.info("summary: %s", {k: v for k, v in summary.items() if k != "timings"})
    return summary
