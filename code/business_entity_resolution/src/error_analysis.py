"""Where is macro F0.5 lost? (validation HOLDOUT, selected model + decision rule)

Uses reports/valid_scores.parquet and models/decision.json from the last ``evaluate`` run,
so it needs no re-scoring. Nothing here is used for training or tuning.

1. Loss attribution: macro F0.5 under counterfactuals that fix one error type at a time
     current            the submitted-style predictions
     no_false_pos       remove every false match
     no_missed_in_cands add every true match that WAS a candidate but was not predicted
     perfect_matcher    exactly the true matches among the candidates (= blocking ceiling)
   The gaps say whether blocking, matcher recall or matcher precision limits the score.
2. Error rates by segment (true matches per entity, source, non-Latin name, empty address,
   country, candidate rank).
3. Example rows (raw names/addresses, probability, rank) for each error type.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .candidate_generation import LazyTargets, candidates_path, queried_rows
from .config import CFG, Config
from .evaluation import gt_row_pairs, macro_f05
from .matching import decide
from .model_training import models_dir, valid_halves
from .preprocessing import normalized_path
from .utils import save_json

LOG = logging.getLogger("ber.errors")


def _keys(r, t, n_t):
    return np.asarray(r, np.int64) * n_t + np.asarray(t, np.int64)


def run_error_analysis(cfg: Config = CFG, n_examples: int = 200) -> dict:
    rep = cfg.artifacts_dir / "reports"
    decision = json.loads((models_dir(cfg) / "decision.json").read_text(encoding="utf-8"))
    sc = pd.read_parquet(rep / "valid_scores.parquet")
    p = sc[f"p_{decision['model']}"].to_numpy()
    ents = queried_rows("train_valid", cfg)
    _, hold = valid_halves(ents, cfg)
    in_hold = np.isin(sc["s1_row"].to_numpy(), hold)
    rows, tgt, p = sc["s1_row"].to_numpy()[in_hold], sc["tgt"].to_numpy()[in_hold], p[in_hold]
    lt = LazyTargets("train", cfg)
    n_t = lt.num_rows

    truth = gt_row_pairs(cfg)
    truth = truth.filter(pc.is_in(truth["s1_row"], value_set=pa.array(hold)))
    t_rows, t_tgt = truth["s1_row"].to_numpy().astype(np.int64), truth["tgt"].to_numpy().astype(np.int64)
    t_key, c_key = _keys(t_rows, t_tgt, n_t), _keys(rows, tgt, n_t)
    is_true = np.isin(c_key, t_key)
    pred = decide(rows, tgt, p, decision["decision"])

    def score(mask):
        return macro_f05(rows[mask], tgt[mask], hold, truth, n_t)

    scen = {"current": score(pred), "no_false_pos": score(pred & is_true),
            "no_missed_in_cands": score(pred | is_true), "perfect_matcher": score(is_true)}
    attribution = {
        "current_macro_f05": scen["current"]["macro_f05"],
        "gain_if_no_false_positives": scen["no_false_pos"]["macro_f05"] - scen["current"]["macro_f05"],
        "gain_if_no_missed_in_candidates": scen["no_missed_in_cands"]["macro_f05"] - scen["current"]["macro_f05"],
        "gain_if_perfect_matcher": scen["perfect_matcher"]["macro_f05"] - scen["current"]["macro_f05"],
        "loss_from_blocking (1 - ceiling)": 1 - scen["perfect_matcher"]["macro_f05"],
        "counts": {"true_pairs": int(len(t_key)), "true_in_candidates": int(is_true.sum()),
                   "true_not_in_candidates": int(len(t_key) - is_true.sum()),
                   "predicted": int(pred.sum()), "tp": int((pred & is_true).sum()),
                   "fp": int((pred & ~is_true).sum()), "fn_in_candidates": int((~pred & is_true).sum())}}

    # ---- segments
    cand = pq.read_table(candidates_path("train_valid", cfg), columns=["s1_row", "tgt", "rank"]).to_pandas()
    cand = cand[np.isin(cand["s1_row"].to_numpy(), hold)]
    rank_map = pd.Series(cand["rank"].to_numpy(), index=_keys(cand["s1_row"], cand["tgt"], n_t))
    n_true_ent = pd.Series(t_rows).value_counts()
    tt = pd.DataFrame({"s1_row": rows, "tgt": tgt, "p": p, "pred": pred, "true": is_true})
    tt["rank"] = rank_map.reindex(c_key).to_numpy()
    tt["src"] = np.where(tt["tgt"] < lt.n2, "S2", "S3")
    tt["n_true_entity"] = pd.cut(n_true_ent.reindex(tt["s1_row"]).fillna(0).to_numpy(), [-1, 0, 1, 3, 100],
                                 labels=["0 (singleton)", "1", "2-3", "4+"])
    nonlatin = lt["name_nonlatin"].combine_chunks().to_numpy(zero_copy_only=False)
    addr_empty = lt["addr_empty"].combine_chunks().to_numpy(zero_copy_only=False)
    tt["t_nonlatin"], tt["t_addr_empty"] = nonlatin[tgt], addr_empty[tgt]
    country = pq.read_table(normalized_path("train", 1, cfg), columns=["country"])["country"].to_numpy(
        zero_copy_only=False)
    tt["country"] = country[rows]
    tt["rank_bucket"] = pd.cut(tt["rank"], [-1, 0, 2, 5, 100], labels=["0", "1-2", "3-5", "6+"])
    seg = []
    for col in ("n_true_entity", "src", "t_nonlatin", "t_addr_empty", "country", "rank_bucket"):
        g = tt.groupby(col, observed=True)
        seg.append(pd.DataFrame({
            "segment": col, "value": g.size().index.astype(str),
            "true_pairs": g["true"].sum().to_numpy(),
            "recall_in_cands": (g.apply(lambda d: (d.pred & d.true).sum() / max(d.true.sum(), 1),
                                        include_groups=False)).to_numpy(),
            "false_pos": g.apply(lambda d: (d.pred & ~d.true).sum(), include_groups=False).to_numpy(),
            "precision": g.apply(lambda d: (d.pred & d.true).sum() / max(d.pred.sum(), 1),
                                 include_groups=False).to_numpy()}))
    segments = pd.concat(seg, ignore_index=True).round(4)
    segments.to_csv(rep / "error_segments.csv", index=False)

    # ---- examples with raw text
    cols = ["entity_id", "business_name", "business_address"]
    s1 = pq.read_table(normalized_path("train", 1, cfg), columns=cols)
    tg_cols = {c: pa.concat_arrays(lt[c].chunks) for c in cols}
    rng = np.random.default_rng(cfg.random_seed)

    def examples(mask, name):
        idx = np.flatnonzero(mask)
        idx = np.sort(rng.choice(idx, size=min(n_examples, len(idx)), replace=False)) if len(idx) else idx
        d = tt.iloc[idx][["s1_row", "tgt", "p", "rank", "src", "n_true_entity"]].copy()
        for c in cols:
            d[f"s1_{c}"] = s1[c].take(pa.array(d["s1_row"].to_numpy())).to_pylist()
            d[f"t_{c}"] = tg_cols[c].take(pa.array(d["tgt"].to_numpy())).to_pylist()
        d.drop(columns=["s1_row", "tgt"]).to_csv(rep / f"errors_{name}.tsv", sep="\t", index=False)

    examples((~tt["pred"] & tt["true"]).to_numpy(), "missed_in_candidates")
    examples((tt["pred"] & ~tt["true"]).to_numpy(), "false_positives")
    # true pairs that never became candidates
    miss = ~np.isin(t_key, c_key)
    mt = pd.DataFrame({"s1_row": t_rows[miss], "tgt": t_tgt[miss]})
    mt = mt.sample(n=min(n_examples, len(mt)), random_state=cfg.random_seed)
    for c in cols:
        mt[f"s1_{c}"] = s1[c].take(pa.array(mt["s1_row"].to_numpy())).to_pylist()
        mt[f"t_{c}"] = tg_cols[c].take(pa.array(mt["tgt"].to_numpy())).to_pylist()
    mt.drop(columns=["s1_row", "tgt"]).to_csv(rep / "errors_not_in_candidates.tsv", sep="\t", index=False)

    out = {"attribution": attribution, "scenarios": {k: v["macro_f05"] for k, v in scen.items()}}
    save_json(out, rep / "error_analysis.json")
    LOG.info("loss attribution (holdout): %s", json.dumps(attribution, indent=2))
    LOG.info("segments:\n%s", segments.to_string(index=False))
    return out
