"""Matcher training and model comparison (Phase 6).

Models (all CPU, permissive licences):
  rule      threshold on the hand-set re-rank score ``rr`` (no training; baseline)
  logreg    scikit-learn LogisticRegression (BSD-3) on median-imputed + missing-indicator,
            standardised features, fitted on a pair sample of train_fit
  lgbm      LightGBM (MIT) gradient-boosted trees, NaN handled natively, early stopping on
            a held-out 10% of train_fit *entities* (never on validation data). Trains on a GPU
            when one works (see device.py), otherwise on the CPU

Evaluation protocol (no leakage):
  * models see only train_fit (train fold);
  * the validation fold is split by entity into TUNE (decision-rule parameters) and HOLDOUT
    (reported numbers); every validation entity counts, including those without candidates
    or whose true matches were not retrieved;
  * the official macro F0.5 is the selection criterion.

All training pairs are real candidates (no negative sub-sampling), so probabilities are on
the same scale as at inference time.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .candidate_generation import queried_rows
from .config import CFG, TARGET_SOURCES, Config
from .evaluation import gt_row_pairs, macro_f05
from .feature_engineering import feature_columns, features_path
from .device import lgbm_train_params
from .matching import decide
from .preprocessing import normalized_path
from .utils import save_json, timed

LOG = logging.getLogger("ber.model")


def models_dir(cfg: Config = CFG) -> Path:
    return cfg.artifacts_dir / "models"


def load_xy(name: str, cfg: Config) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame, list[str]]:
    t = pq.read_table(features_path(name, cfg))
    cols = feature_columns(t.column_names)
    X = np.column_stack([t[c].to_numpy(zero_copy_only=False).astype(np.float32) for c in cols])
    y = t["label"].to_numpy() if "label" in t.column_names else None
    ids = t.select(["s1_row", "tgt", "src", "rr"]).to_pandas()
    return X, y, ids, cols


def train_models(cfg: Config = CFG) -> dict:
    """Fit logistic regression and LightGBM on train_fit; save models + feature list."""
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = models_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    X, y, ids, cols = load_xy("train_fit", cfg)
    LOG.info("train_fit: %d pairs x %d features, positive rate %.4f", *X.shape, y.mean())
    timings: dict = {}

    rng = np.random.default_rng(cfg.random_seed)
    ents = np.unique(ids["s1_row"].to_numpy())
    es_ents = rng.choice(ents, size=int(len(ents) * cfg.early_stop_fraction), replace=False)
    es = np.isin(ids["s1_row"].to_numpy(), es_ents)
    params = lgbm_train_params(cfg)  # GPU if a working one is found, else CPU
    with timed(f"lightgbm ({params['device_type']})", timings):
        dtr = lgb.Dataset(X[~es], y[~es], feature_name=cols, free_raw_data=True)
        dva = lgb.Dataset(X[es], y[es], reference=dtr)
        booster = lgb.train(params, dtr, cfg.lgbm_max_rounds, valid_sets=[dva], valid_names=["early_stop"],
                            callbacks=[lgb.early_stopping(cfg.lgbm_early_stopping, verbose=False),
                                       lgb.log_evaluation(100)])
        booster.save_model(str(out / "lgbm.txt"), num_iteration=booster.best_iteration)
    imp = pd.DataFrame({"feature": cols, "gain": booster.feature_importance("gain"),
                        "splits": booster.feature_importance("split")}).sort_values("gain", ascending=False)
    imp.to_csv(out / "lgbm_feature_importance.csv", index=False)

    with timed("logreg", timings):
        idx = rng.choice(len(y), size=min(cfg.logreg_max_rows, len(y)), replace=False)
        logreg = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                               LogisticRegression(max_iter=1000, C=1.0))
        logreg.fit(X[idx], y[idx])
        joblib.dump(logreg, out / "logreg.joblib")

    # expected number of true matches NOT retrievable (outside candidates) per entity, from train_fit
    truth = gt_row_pairs(cfg)
    rows = queried_rows("train_fit", cfg)
    n_true = int(np.isin(truth["s1_row"].to_numpy(), rows).sum())
    missing = (n_true - int(y.sum())) / len(rows)
    meta = {"features": cols, "lgbm_device": params["device_type"], "lgbm_params": params,
            "lgbm_best_iteration": booster.best_iteration,
            "lgbm_best_score": booster.best_score["early_stop"]["binary_logloss"],
            "train_pairs": int(len(y)), "train_entities": int(len(rows)),
            "missing_true_per_entity": missing, "timings": timings}
    save_json(meta, out / "training_meta.json")
    LOG.info("saved models to %s (lgbm iterations=%d, missing matches/entity=%.4f)", out,
             booster.best_iteration, missing)
    return meta


def predict(model: str, X: np.ndarray, ids: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Pair scores in [0, 1] for 'rule', 'logreg' or 'lgbm'."""
    if model == "rule":
        return ids["rr"].to_numpy().astype(np.float64)
    if model == "logreg":
        return joblib.load(models_dir(cfg) / "logreg.joblib").predict_proba(X)[:, 1]
    if model == "lgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=str(models_dir(cfg) / "lgbm.txt")).predict(X)
    raise ValueError(model)


def valid_halves(ents: np.ndarray, cfg: Config = CFG) -> tuple[np.ndarray, np.ndarray]:
    """Seeded entity split of the validation fold into TUNE and HOLDOUT halves."""
    rng = np.random.default_rng(cfg.random_seed + 1)
    tune = np.sort(rng.choice(ents, size=int(len(ents) * cfg.tune_fraction), replace=False))
    return tune, np.setdiff1d(ents, tune)


def _param_grid(model: str, missing: float) -> list[dict]:
    lo, hi = (0.5, 1.0) if model == "rule" else (0.05, 0.95)
    grid = [{"rule": "threshold", "t": round(float(t), 3)} for t in np.arange(lo, hi + 1e-9, 0.025)]
    if model != "rule":  # expected-F needs probabilities
        grid += [{"rule": "expected_f", "empty_scale": s, "missing": missing}
                 for s in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0)]
    return grid


def evaluate_models(cfg: Config = CFG, models: tuple[str, ...] = ("rule", "logreg", "lgbm"),
                    reuse_scores: bool = False) -> pd.DataFrame:
    """Tune each model's decision rule on valid-TUNE, report macro F0.5 on valid-HOLDOUT.

    reuse_scores: take pair scores from reports/valid_scores.parquet (previous run) instead of
    re-predicting -- for iterating on decision rules only (models unchanged).
    """
    cached = None
    if reuse_scores:
        cached = pd.read_parquet(cfg.artifacts_dir / "reports" / "valid_scores.parquet")
    X, _, ids, cols = load_xy("train_valid", cfg)
    meta = json.loads((models_dir(cfg) / "training_meta.json").read_text(encoding="utf-8"))
    if meta["features"] != cols:
        raise ValueError("feature columns differ between training and validation")
    truth = gt_row_pairs(cfg)
    n_t = sum(pq.ParquetFile(normalized_path("train", s, cfg)).metadata.num_rows for s in TARGET_SOURCES)
    ents = queried_rows("train_valid", cfg)
    tune_ents, hold_ents = valid_halves(ents, cfg)
    country = pq.read_table(normalized_path("train", 1, cfg), columns=["country"])["country"].to_numpy(
        zero_copy_only=False)
    rows_all, tg_all = ids["s1_row"].to_numpy(), ids["tgt"].to_numpy()
    in_tune = np.isin(rows_all, tune_ents)

    results, best_params, scores = [], {}, {}
    for model in models:
        with timed(f"score {model}"):
            if cached is not None:
                if not (cached["s1_row"].to_numpy() == ids["s1_row"].to_numpy()).all():
                    raise ValueError("cached scores do not align with features; rerun without --reuse-scores")
                p = cached[f"p_{model}"].to_numpy()
            else:
                p = predict(model, X, ids, cfg)
        scores[model] = p
        r_t, g_t, p_t = rows_all[in_tune], tg_all[in_tune], p[in_tune]
        best = None
        for params in _param_grid(model, meta["missing_true_per_entity"]):
            m = decide(r_t, g_t, p_t, params)
            f = macro_f05(r_t[m], g_t[m], tune_ents, truth, n_t)["macro_f05"]
            if best is None or f > best[0]:
                best = (f, params)
        for excl in (False, True):
            params = {**best[1], "exclusive": excl}
            m = decide(r_t, g_t, p_t, params)
            f = macro_f05(r_t[m], g_t[m], tune_ents, truth, n_t)["macro_f05"]
            if f >= best[0]:
                best = (f, params)
        r_h, g_h, p_h = rows_all[~in_tune], tg_all[~in_tune], p[~in_tune]
        m = decide(r_h, g_h, p_h, best[1])
        res = macro_f05(r_h[m], g_h[m], hold_ents, truth, n_t, groups={"country": country[hold_ents]})
        results.append({"model": model, "tune_macro_f05": best[0], **{f"holdout_{k}": v for k, v in res.items()},
                        "params": json.dumps(best[1])})
        best_params[model] = best[1]
        LOG.info("%-6s tune=%.5f holdout=%.5f params=%s", model, best[0], res["macro_f05"], best[1])

    table = pd.DataFrame(results)
    rep = cfg.artifacts_dir / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    table.to_csv(rep / "model_comparison.csv", index=False)
    winner = table.sort_values("tune_macro_f05", ascending=False).iloc[0]["model"]  # chosen on TUNE, not holdout
    save_json({"model": winner, "decision": best_params[winner], "all": best_params},
              models_dir(cfg) / "decision.json")
    pd.DataFrame({"s1_row": rows_all, "tgt": tg_all, **{f"p_{k}": v for k, v in scores.items()}}).to_parquet(
        rep / "valid_scores.parquet", index=False)
    LOG.info("model comparison (holdout):\n%s", table[["model", "tune_macro_f05", "holdout_macro_f05",
                                                    "holdout_f05_singletons", "holdout_f05_non_singletons",
                                                    "holdout_micro_precision", "holdout_micro_recall"]]
             .round(5).to_string(index=False))
    LOG.info("selected: %s %s", winner, best_params[winner])
    return table
