"""Memory-efficient EDA over the full local datasets.

Execution plan (peak memory ~ one source table + ID arrays; est. 2-4 GB for train):
  1. Profile each source file on its own, then keep only its ``entity_id`` column.
  2. Profile the ground truth against those ID arrays (integrity, match counts,
     exclusivity, coverage) -- this needs every ID, so it is a full scan.
  3. Pair-level stats on a random sample of non-singleton S1 entities: re-read
     each source, keep only the sampled records, then join. Ratios estimated on
     200k entities have standard error < 0.2 pp, so a full join is unnecessary.
  4. Write report.json, CSV tables and a short summary.md to artifacts/eda/<split>/.

Full scan: row counts, duplicates, ID integrity, coverage, exclusivity, match
count distribution. Incremental (per Arrow chunk, then merged): token counts.
Sample-based: pair agreement rates and ID/row-order shortcut checks.

``basic_norm`` below is deliberately minimal (NFKD accent folding, lowercase,
non-alphanumerics -> space); it is a measurement baseline, not the Phase 2
normaliser.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from .config import CFG, SOURCE_PREFIXES, TARGET_SOURCES, Config
from .data_loader import id_format_mask, load_ground_truth, load_source
from .utils import peak_rss_mb, save_json, timed

LOG = logging.getLogger("ber.eda")


# --------------------------------------------------------------------------- helpers
def basic_norm(arr: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    s = pc.utf8_normalize(arr, "NFKD")
    s = pc.replace_substring_regex(s, r"[\x{0300}-\x{036f}]", "")  # Latin combining accents only
    s = pc.utf8_lower(s)
    s = pc.replace_substring_regex(s, r"[^\p{L}\p{N}\p{M}]+", " ")  # keeps Indic vowel signs
    return pc.utf8_trim_whitespace(s)


def _quantiles(values: np.ndarray, qs: tuple[float, ...]) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {f"q{q:g}": float(v) for q, v in zip(qs, np.quantile(values, qs))}


def _count(mask: pa.Array | pa.ChunkedArray) -> int:
    return int(pc.sum(mask).as_py() or 0)


def token_counts(col: pa.ChunkedArray) -> pa.Table:
    """Token occurrence counts (basic_norm, whitespace split), computed per chunk then merged."""
    parts = []
    for chunk in col.chunks:
        toks = pc.list_flatten(pc.split_pattern(basic_norm(chunk), " "))
        toks = toks.filter(pc.not_equal(toks, ""))
        if len(toks):
            vc = pc.value_counts(toks)
            parts.append(pa.table({"token": vc.field("values"), "count": vc.field("counts")}))
    if not parts:
        return pa.table({"token": pa.array([], pa.string()), "count": pa.array([], pa.int64())})
    agg = pa.concat_tables(parts).group_by("token").aggregate([("count", "sum")])
    out = pa.table({"token": agg["token"], "count": agg["count_sum"]})
    return out.sort_by([("count", "descending")])


def _token_summary(tc: pa.Table, cfg: Config) -> dict[str, Any]:
    counts = tc["count"].to_numpy()
    total = int(counts.sum())
    res = {"vocab_size": int(len(counts)), "total_occurrences": total,
           "singleton_tokens": int((counts == 1).sum())}
    for t in cfg.eda_token_thresholds:
        m = counts >= t
        res[f"tokens_ge_{t}"] = int(m.sum())
        res[f"occurrence_share_ge_{t}"] = round(float(counts[m].sum()) / total, 4) if total else 0.0
    return res


def _script_counts(col: pa.ChunkedArray, cfg: Config) -> dict[str, int]:
    res = {s: _count(pc.match_substring_regex(col, rf"\p{{{s}}}")) for s in cfg.eda_scripts}
    res["any_non_ascii"] = _count(pc.match_substring_regex(col, r"[^\x00-\x7f]"))
    res["any_non_latin_letter"] = _count(pc.match_substring_regex(col, r"[^\p{Latin}\P{L}]"))
    return res


# --------------------------------------------------------------------------- per source
def profile_source(t: pa.Table, label: str, source_no: int, out_dir: Path, cfg: Config) -> dict[str, Any]:
    n = t.num_rows
    ids, name, addr, country = (t[c] for c in ("entity_id", "business_name", "business_address", "country"))
    rep: dict[str, Any] = {"rows": n, "columns": t.column_names}

    # Missing / whitespace issues per column
    miss = {}
    for c in t.column_names:
        col = t[c]
        trimmed = pc.utf8_trim_whitespace(col)
        miss[c] = {"empty": _count(pc.equal(col, "")),
                   "whitespace_only": _count(pc.and_(pc.equal(trimmed, ""), pc.not_equal(col, ""))),
                   "leading_or_trailing_ws": _count(pc.and_(pc.not_equal(trimmed, col), pc.not_equal(trimmed, ""))),
                   "internal_double_space": _count(pc.match_substring(col, "  "))}
    rep["missing"] = miss

    # ID integrity
    bad_fmt = pc.invert(id_format_mask(ids, source_no))
    n_distinct = int(pc.count_distinct(ids).as_py())
    digits = pc.utf8_length(ids).to_numpy() - 3
    vc = pc.value_counts(ids)
    dup_examples = [v.as_py() for v, c in zip(vc.field("values"), vc.field("counts")) if c.as_py() > 1][:10]
    rep["entity_id"] = {"invalid_format": _count(bad_fmt), "distinct": n_distinct,
                        "duplicate_rows": n - n_distinct, "duplicate_examples": dup_examples,
                        "leading_zero_after_prefix": _count(pc.match_substring_regex(ids, r"^S[0-9]-0")),
                        "digit_length": _quantiles(digits, (0.0, 0.5, 1.0))}

    # Duplicate content
    non_empty = lambda col: col.filter(pc.not_equal(col, ""))  # noqa: E731
    na = pc.binary_join_element_wise(name, addr, "\x1f")
    nac = pc.binary_join_element_wise(name, addr, country, "\x1f")
    rep["duplicates"] = {
        "rows_repeating_a_name": len(non_empty(name)) - int(pc.count_distinct(non_empty(name)).as_py()),
        "rows_repeating_an_address": len(non_empty(addr)) - int(pc.count_distinct(non_empty(addr)).as_py()),
        "rows_repeating_name_and_address": n - int(pc.count_distinct(na).as_py()),
        "rows_repeating_name_address_country": n - int(pc.count_distinct(nac).as_py()),
        "rows_repeating_basic_norm_name": len(non_empty(name)) - int(pc.count_distinct(basic_norm(non_empty(name))).as_py()),
    }

    # Country labels (open set: exact strings, plus case/whitespace collisions)
    cvc = pc.value_counts(country)
    cdf = pd.DataFrame({"country": cvc.field("values").to_pylist(), "rows": cvc.field("counts").to_numpy()})
    cdf = cdf.sort_values("rows", ascending=False)
    cdf["share"] = (cdf["rows"] / n).round(6)
    cdf.to_csv(out_dir / f"{label}_country_counts.csv", index=False)
    cdf["key"] = cdf["country"].str.strip().str.lower()
    rep["country"] = {"n_labels": len(cdf), "counts": dict(zip(cdf["country"], cdf["rows"].astype(int))),
                      "case_or_ws_variant_groups": cdf.groupby("key").size().gt(1).sum().item()}

    # Lengths
    nn, an = basic_norm(name), basic_norm(addr)
    rep["lengths"] = {
        "name_chars": _quantiles(pc.utf8_length(name).to_numpy(), cfg.quantiles),
        "name_tokens": _quantiles(pc.list_value_length(pc.split_pattern(nn, " ")).to_numpy(), cfg.quantiles),
        "address_chars": _quantiles(pc.utf8_length(addr).to_numpy(), cfg.quantiles),
        "address_tokens": _quantiles(pc.list_value_length(pc.split_pattern(an, " ")).to_numpy(), cfg.quantiles),
    }
    rep["scripts"] = {"name": _script_counts(name, cfg), "address": _script_counts(addr, cfg)}

    # Per-country field quality and postal-like patterns (generic digit-run lengths, no country rules)
    flags = pa.table({
        "country": country,
        "rows": pa.array(np.ones(n, dtype=np.int64)),
        "empty_name": pc.cast(pc.equal(name, ""), pa.int64()),
        "empty_address": pc.cast(pc.equal(addr, ""), pa.int64()),
        "addr_5digit": pc.cast(pc.match_substring_regex(addr, r"\b[0-9]{5}\b"), pa.int64()),
        "addr_6digit": pc.cast(pc.match_substring_regex(addr, r"\b[0-9]{6}\b"), pa.int64()),
        "addr_3sp3digit": pc.cast(pc.match_substring_regex(addr, r"\b[0-9]{3} [0-9]{3}\b"), pa.int64()),
        "addr_any_digit": pc.cast(pc.match_substring_regex(addr, r"[0-9]"), pa.int64()),
        "name_non_latin": pc.cast(pc.match_substring_regex(name, r"[^\p{Latin}\P{L}]"), pa.int64()),
        "addr_non_latin": pc.cast(pc.match_substring_regex(addr, r"[^\p{Latin}\P{L}]"), pa.int64()),
    })
    metric_cols = [c for c in flags.column_names if c != "country"]
    g = flags.group_by("country").aggregate([(c, "sum") for c in metric_cols]).to_pandas()
    g.columns = [c.removesuffix("_sum") for c in g.columns]
    for c in metric_cols[1:]:
        g[c + "_rate"] = (g[c] / g["rows"]).round(4)
    g.sort_values("rows", ascending=False).to_csv(out_dir / f"{label}_by_country.csv", index=False)
    rep["by_country_file"] = f"{label}_by_country.csv"
    return rep


def _source_tokens(t: pa.Table, label: str, out_dir: Path, cfg: Config) -> dict[str, pa.Table]:
    res = {}
    for field_name, col in (("name", "business_name"), ("address", "business_address")):
        tc = token_counts(t[col])
        tc.slice(0, cfg.eda_top_tokens).to_pandas().to_csv(out_dir / f"{label}_top_{field_name}_tokens.csv", index=False)
        res[field_name] = tc
    return res


# --------------------------------------------------------------------------- ground truth
def profile_ground_truth(gt: pa.Table, s1_ids: pa.ChunkedArray, tgt_ids: dict[int, pa.ChunkedArray],
                         out_dir: Path) -> tuple[dict[str, Any], pa.Table]:
    """Integrity + distribution stats. Returns (report, deduplicated valid pairs table)."""
    s1col = gt["source1_entity_id"].combine_chunks()
    mcol = gt["matched_entity_ids"].combine_chunks()
    n = len(s1col)
    rep: dict[str, Any] = {"rows": n}

    distinct_s1 = int(pc.count_distinct(s1col).as_py())
    rep["source1_ids"] = {
        "invalid_format": _count(pc.invert(id_format_mask(s1col, 1))),
        "duplicate_rows": n - distinct_s1,
        "gt_ids_missing_from_source1": _count(pc.invert(pc.is_in(s1col, value_set=s1_ids))),
        "source1_ids_missing_from_gt": _count(pc.invert(pc.is_in(s1_ids, value_set=s1col))),
    }

    empty_field = pc.equal(mcol, "")
    lists = pc.split_pattern(mcol, ",")
    parent = pc.list_parent_indices(lists).to_numpy()
    flat = pc.list_flatten(lists)
    trimmed = pc.utf8_trim_whitespace(flat)
    empty_tok = pc.equal(trimmed, "").to_numpy(zero_copy_only=False)
    keep = ~empty_tok
    ids, rows = trimmed.filter(pa.array(keep)), parent[keep]

    is_src = {k: id_format_mask(ids, k).to_numpy(zero_copy_only=False) for k in TARGET_SOURCES}
    is_s1 = id_format_mask(ids, 1).to_numpy(zero_copy_only=False)
    exists = {k: pc.is_in(ids, value_set=tgt_ids[k]).to_numpy(zero_copy_only=False) for k in TARGET_SOURCES}
    valid = np.zeros(len(ids), dtype=bool)
    for k in TARGET_SOURCES:
        valid |= is_src[k] & exists[k]

    rep["list_hygiene"] = {
        "empty_fields": _count(empty_field),
        "empty_tokens_inside_nonempty_fields": int(empty_tok.sum()) - _count(empty_field),
        "tokens_with_surrounding_whitespace": _count(pc.not_equal(flat, trimmed)),
        "total_listed_ids": int(len(ids)),
        "source1_ids_inside_lists": int(is_s1.sum()),
        "other_invalid_format_ids": int((~is_s1 & ~is_src[2] & ~is_src[3]).sum()),
        **{f"s{k}_ids_not_in_source{k}": int((is_src[k] & ~exists[k]).sum()) for k in TARGET_SOURCES},
    }

    src_code = np.where(is_src[2], 2, 3).astype(np.int8)
    pairs = pa.table({"row": pa.array(rows[valid].astype(np.int32)), "tgt_id": ids.filter(pa.array(valid)),
                      "src": pa.array(src_code[valid])})
    grouped = pairs.group_by(["row", "tgt_id", "src"]).aggregate([("row", "count")])
    rep["list_hygiene"]["duplicate_ids_within_a_list"] = int(len(pairs) - grouped.num_rows)
    pairs = grouped.select(["row", "tgt_id", "src"]).sort_by([("row", "ascending")])

    prow, psrc = pairs["row"].to_numpy(), pairs["src"].to_numpy()
    total = np.bincount(prow, minlength=n)
    per = {k: np.bincount(prow[psrc == k], minlength=n) for k in TARGET_SOURCES}

    def dist(arr: np.ndarray) -> pd.DataFrame:
        vc = pd.Series(arr).value_counts().sort_index()
        return pd.DataFrame({"n_matches": vc.index, "entities": vc.values, "share": (vc.values / n).round(6)})

    dist(total).to_csv(out_dir / "gt_match_count_total.csv", index=False)
    for k in TARGET_SOURCES:
        dist(per[k]).to_csv(out_dir / f"gt_match_count_s{k}.csv", index=False)

    rep["match_counts"] = {
        "singletons": int((total == 0).sum()), "singleton_share": round(float((total == 0).mean()), 6),
        "exactly_one": int((total == 1).sum()), "two_or_more": int((total >= 2).sum()),
        "mean_matches_all": round(float(total.mean()), 4),
        "mean_matches_nonsingleton": round(float(total[total > 0].mean()), 4) if (total > 0).any() else 0.0,
        "max_matches": int(total.max()) if n else 0,
        "only_s2": int(((per[2] > 0) & (per[3] == 0)).sum()),
        "only_s3": int(((per[3] > 0) & (per[2] == 0)).sum()),
        "both_s2_s3": int(((per[2] > 0) & (per[3] > 0)).sum()),
        "entities_with_2plus_from_same_source": {f"s{k}": int((per[k] >= 2).sum()) for k in TARGET_SOURCES},
        "max_from_single_source": {f"s{k}": int(per[k].max()) if n else 0 for k in TARGET_SOURCES},
        "p99_from_single_source": {f"s{k}": float(np.quantile(per[k][total > 0], 0.99)) if (total > 0).any() else 0.0
                                   for k in TARGET_SOURCES},
    }

    # Coverage and exclusivity of S2/S3 records
    cov, excl = {}, {}
    for k in TARGET_SOURCES:
        sub = pairs.filter(pc.equal(pairs["src"], k))["tgt_id"]
        vc = pc.value_counts(sub)
        ref_counts = vc.field("counts").to_numpy()
        n_src = len(tgt_ids[k])
        cov[f"s{k}"] = {"distinct_referenced": int(len(ref_counts)), "source_rows": n_src,
                        "referenced_share": round(len(ref_counts) / n_src, 6) if n_src else 0.0,
                        "unreferenced_records": int(n_src - len(ref_counts))}
        multi = ref_counts > 1
        excl[f"s{k}"] = {"ids_matched_to_multiple_s1": int(multi.sum()),
                         "max_s1_per_id": int(ref_counts.max()) if len(ref_counts) else 0,
                         "pairs_involving_shared_ids": int(ref_counts[multi].sum())}
    rep["target_coverage"], rep["exclusivity"] = cov, excl

    s1_for_pairs = s1col.take(pairs["row"].combine_chunks())
    pairs = pairs.append_column("s1_id", s1_for_pairs)
    return rep, pairs


# --------------------------------------------------------------------------- pair sample
def _fetch_records(split: str, source_no: int, wanted: pa.Array, cfg: Config) -> pd.DataFrame:
    t, _ = load_source(split, source_no, cfg=cfg)
    t = t.append_column("row_pos", pa.array(np.arange(t.num_rows, dtype=np.int64)))
    sub = t.filter(pc.is_in(t["entity_id"], value_set=wanted))
    del t
    sub = sub.append_column("name_norm", basic_norm(sub["business_name"]))
    sub = sub.append_column("addr_norm", basic_norm(sub["business_address"]))
    sub = sub.append_column("name_non_latin", pc.match_substring_regex(sub["business_name"], r"[^\p{Latin}\P{L}]"))
    return sub.to_pandas()


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def pair_analysis(split: str, pairs: pa.Table, source_rows: dict[int, int],
                  token_df: dict[str, dict[str, int]], out_dir: Path, cfg: Config) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.random_seed)
    uniq_rows = np.unique(pairs["row"].to_numpy())
    k = min(cfg.eda_pair_sample_entities, len(uniq_rows))
    if k == 0:
        return {"sampled_entities": 0}
    chosen = rng.choice(uniq_rows, size=k, replace=False)
    sp = pairs.filter(pc.is_in(pairs["row"], value_set=pa.array(chosen.astype(np.int32)))).to_pandas()

    s1 = _fetch_records(split, 1, pa.array(sp["s1_id"].drop_duplicates().tolist(), pa.string()), cfg).add_prefix("s1_")
    tg = pd.concat([_fetch_records(split, s, pa.array(sp.loc[sp["src"] == s, "tgt_id"].drop_duplicates().tolist(), pa.string()), cfg)
                    for s in TARGET_SOURCES], ignore_index=True).add_prefix("t_")
    df = sp.merge(s1, left_on="s1_id", right_on="s1_entity_id").merge(tg, left_on="tgt_id", right_on="t_entity_id")

    hub = cfg.eda_hub_token_count
    name_df, addr_df = token_df["name"], token_df["address"]

    def token_feats(a_list: list[str], b_list: list[str], dfmap: dict[str, int]):
        n_ = len(a_list)
        jac, any_s, rare_s, sorted_eq = (np.zeros(n_) for _ in range(4))
        for i, (a, b) in enumerate(zip(a_list, b_list)):
            ta, tb = set(a.split()), set(b.split())
            inter = ta & tb
            union = ta | tb
            jac[i] = len(inter) / len(union) if union else 0.0
            any_s[i] = bool(inter)
            rare_s[i] = any(dfmap.get(t, 0) < hub for t in inter)
            sorted_eq[i] = bool(ta) and sorted(a.split()) == sorted(b.split())
        return jac, any_s.astype(bool), rare_s.astype(bool), sorted_eq.astype(bool)

    nj, n_any, n_rare, n_sorted = token_feats(df["s1_name_norm"].tolist(), df["t_name_norm"].tolist(), name_df)
    aj, a_any, a_rare, _ = token_feats(df["s1_addr_norm"].tolist(), df["t_addr_norm"].tolist(), addr_df)
    df = df.assign(
        country_equal=df["s1_country"] == df["t_country"],
        name_exact_raw=df["s1_business_name"] == df["t_business_name"],
        name_exact_norm=(df["s1_name_norm"] == df["t_name_norm"]) & (df["s1_name_norm"] != ""),
        name_token_sorted_equal=n_sorted, name_jaccard=nj, name_shares_token=n_any, name_shares_rare_token=n_rare,
        addr_exact_norm=(df["s1_addr_norm"] == df["t_addr_norm"]) & (df["s1_addr_norm"] != ""),
        addr_jaccard=aj, addr_shares_token=a_any, addr_shares_rare_token=a_rare,
        target_addr_empty=df["t_business_address"] == "",
    )
    df["no_rare_token_shared_anywhere"] = ~(df["name_shares_rare_token"] | df["addr_shares_rare_token"])

    metrics = ["country_equal", "name_exact_raw", "name_exact_norm", "name_token_sorted_equal",
               "name_shares_token", "name_shares_rare_token", "addr_exact_norm", "addr_shares_token",
               "addr_shares_rare_token", "target_addr_empty", "no_rare_token_shared_anywhere"]
    overall = {m: round(float(df[m].mean()), 4) for m in metrics}
    overall["name_jaccard"] = _quantiles(df["name_jaccard"].to_numpy(), (0.05, 0.25, 0.5, 0.75))
    overall["addr_jaccard"] = _quantiles(df["addr_jaccard"].to_numpy(), (0.05, 0.25, 0.5, 0.75))

    groups = []
    for keys in (["src"], ["s1_country"], ["t_name_non_latin"], ["src", "s1_country"]):
        g = df.groupby(keys, dropna=False)[metrics + ["name_jaccard", "addr_jaccard"]].mean().round(4)
        g.insert(0, "pairs", df.groupby(keys, dropna=False).size())
        g = g.reset_index()
        g.insert(0, "group_by", "+".join(keys))
        groups.append(g)
    pd.concat(groups, ignore_index=True).to_csv(out_dir / "gt_pair_agreement_by_group.csv", index=False)
    pd.crosstab(df["s1_country"], df["t_country"]).to_csv(out_dir / "gt_pair_country_crosstab.csv")

    # Hard examples (no rare token shared) for manual inspection
    cols = ["s1_entity_id", "s1_business_name", "s1_business_address", "s1_country",
            "t_entity_id", "t_business_name", "t_business_address", "t_country"]
    df.loc[df["no_rare_token_shared_anywhere"], cols].head(200).to_csv(
        out_dir / "gt_hard_pair_examples.csv", index=False)
    df[cols].sample(n=min(200, len(df)), random_state=cfg.random_seed).to_csv(
        out_dir / "gt_random_pair_examples.csv", index=False)

    # Shortcut / leakage checks: do IDs or row positions carry the answer?
    leak = {}
    s1_num = df["s1_entity_id"].str.slice(3).astype("int64").to_numpy()
    t_num = df["t_entity_id"].str.slice(3).astype("int64").to_numpy()
    s1_pos = df["s1_row_pos"].to_numpy() / source_rows[1]
    for s in TARGET_SOURCES:
        m = (df["src"] == s).to_numpy()
        t_pos = df["t_row_pos"].to_numpy()[m] / source_rows[s]
        leak[f"s{s}"] = {
            "pairs": int(m.sum()),
            "spearman_id_number": round(_spearman(s1_num[m], t_num[m]), 4),
            "spearman_row_position": round(_spearman(s1_pos[m], t_pos), 4),
            "spearman_row_position_shuffled_baseline": round(_spearman(s1_pos[m], rng.permutation(t_pos)), 4),
            "mean_abs_normalised_row_gap": round(float(np.abs(s1_pos[m] - t_pos).mean()), 4) if m.any() else None,
            "random_expectation_gap": 0.3333,
        }
    return {"sampled_entities": int(k), "sampled_pairs": int(len(df)), "overall": overall, "shortcut_checks": leak}


# --------------------------------------------------------------------------- summary
def write_summary(report: dict[str, Any], path: Path) -> None:
    L = [f"# EDA summary ({report['split']})", ""]
    for label, r in report["sources"].items():
        L.append(f"## {label}: {r['rows']:,} rows")
        L.append(f"- id: invalid={r['entity_id']['invalid_format']}, dup_rows={r['entity_id']['duplicate_rows']}, "
                 f"leading_zero={r['entity_id']['leading_zero_after_prefix']}, digits={r['entity_id']['digit_length']}")
        L.append("- empty: " + ", ".join(f"{c}={v['empty']}" for c, v in r["missing"].items()))
        L.append(f"- countries: {r['country']['counts']} (case/ws variant groups: {r['country']['case_or_ws_variant_groups']})")
        L.append(f"- duplicates: {r['duplicates']}")
        L.append(f"- name chars q50/q99: {r['lengths']['name_chars'].get('q0.5')}/{r['lengths']['name_chars'].get('q0.99')}; "
                 f"address chars q50/q99: {r['lengths']['address_chars'].get('q0.5')}/{r['lengths']['address_chars'].get('q0.99')}")
        sc = r["scripts"]["name"]
        L.append(f"- name scripts (rows): non_latin_letter={sc['any_non_latin_letter']}, "
                 + ", ".join(f"{k}={v}" for k, v in sc.items() if v and k not in ("Latin", "any_non_ascii", "any_non_latin_letter")))
        if "tokens" in r:
            L.append(f"- name tokens: {r['tokens']['name']}")
            L.append(f"- address tokens: {r['tokens']['address']}")
        L.append("")
    if "ground_truth" in report:
        g = report["ground_truth"]
        L += ["## ground truth", f"- source1_ids: {g['source1_ids']}", f"- list hygiene: {g['list_hygiene']}",
              f"- match counts: {g['match_counts']}", f"- coverage: {g['target_coverage']}",
              f"- exclusivity: {g['exclusivity']}", ""]
    if "pairs" in report:
        p = report["pairs"]
        L += ["## GT pair sample", f"- entities={p.get('sampled_entities')}, pairs={p.get('sampled_pairs')}",
              f"- overall: {p.get('overall')}", f"- shortcut checks: {p.get('shortcut_checks')}", ""]
    if report.get("unseen_country_labels_vs_train") is not None:
        L.append(f"- labels not seen in train: {report['unseen_country_labels_vs_train']}")
    L.append(f"- timings: {report['timings']}")
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- orchestration
def run_eda(split: str = "train", cfg: Config = CFG) -> dict[str, Any]:
    out_dir = cfg.eda_dir(split)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"split": split, "data_root": str(cfg.data_root), "sources": {}, "timings": {}}
    tm = report["timings"]
    ids: dict[int, pa.ChunkedArray] = {}
    rows: dict[int, int] = {}
    tgt_tokens: dict[str, list[pa.Table]] = {"name": [], "address": []}

    for s in SOURCE_PREFIXES:
        label = f"{split}_source{s}"
        with timed(f"profile {label}", tm):
            t, rr = load_source(split, s, cfg=cfg)
            rep = profile_source(t, label, s, out_dir, cfg)
            rep["read"] = rr.__dict__
            toks = _source_tokens(t, label, out_dir, cfg)
            rep["tokens"] = {f: _token_summary(tc, cfg) for f, tc in toks.items()}
            if s in TARGET_SOURCES:
                for f, tc in toks.items():  # keep only frequent tokens (small) for pair stats
                    tgt_tokens[f].append(tc.filter(pc.greater_equal(tc["count"], cfg.eda_token_keep_min)))
            ids[s], rows[s] = t["entity_id"], t.num_rows
            report["sources"][label] = rep
            del t, toks

    if split == "train":
        with timed("profile ground truth", tm):
            gt, rr = load_ground_truth(split, cfg)
            gt_rep, pairs = profile_ground_truth(gt, ids[1], {k: ids[k] for k in TARGET_SOURCES}, out_dir)
            gt_rep["read"] = rr.__dict__
            report["ground_truth"] = gt_rep
            del gt, ids
        with timed("pair sample analysis", tm):
            token_df = {}
            for f, parts in tgt_tokens.items():
                agg = pa.concat_tables(parts).group_by("token").aggregate([("count", "sum")])
                token_df[f] = dict(zip(agg["token"].to_pylist(), agg["count_sum"].to_pylist()))
            report["pairs"] = pair_analysis(split, pairs, rows, token_df, out_dir, cfg)
    else:
        train_report = cfg.eda_dir("train") / "report.json"
        if train_report.exists():
            tr = json.loads(train_report.read_text(encoding="utf-8"))
            train_labels = {c for r in tr["sources"].values() for c in r["country"]["counts"]}
            test_labels = {c for r in report["sources"].values() for c in r["country"]["counts"]}
            report["unseen_country_labels_vs_train"] = sorted(test_labels - train_labels)
        else:
            report["unseen_country_labels_vs_train"] = None

    report["peak_rss_mb"] = peak_rss_mb()
    save_json(report, out_dir / "report.json")
    write_summary(report, out_dir / "summary.md")
    LOG.info("EDA written to %s (see summary.md)", out_dir)
    return report


# --------------------------------------------------------------------------- Phase 2 check
def normalization_check(cfg: Config = CFG) -> dict[str, Any]:
    """Measure how much signal normalised fields expose on true (train) pairs.

    Uses the same entity sample size/seed as the EDA. 'rare' = token count in
    normalised S2+S3 below ``cfg.eda_hub_token_count``. The *_signal columns
    are upper bounds for token-based blocking recall, not blocking results.
    """
    import pyarrow.parquet as pq

    from .data_loader import explode_ground_truth
    from .preprocessing import normalized_path

    out_dir = cfg.eda_dir("train")
    out_dir.mkdir(parents=True, exist_ok=True)
    gt, _ = load_ground_truth("train", cfg)
    pairs = explode_ground_truth(gt)
    del gt
    rng = np.random.default_rng(cfg.random_seed)
    ents = pc.unique(pairs["s1_id"]).to_numpy(zero_copy_only=False)
    chosen = pa.array(rng.choice(ents, size=min(cfg.eda_pair_sample_entities, len(ents)), replace=False))
    sp = pairs.filter(pc.is_in(pairs["s1_id"], value_set=chosen)).to_pandas()

    fields = ["entity_id", "country", "name_core", "name_compact", "name_skel", "name_legal", "name_alt_core",
              "name_nonlatin", "addr_core", "addr_numbers", "addr_region", "addr_empty"]
    counts: dict[str, dict[str, int]] = {"name_core": {}, "name_skel": {}, "addr_core": {}, "addr_numbers": {}}
    frames = {}
    for s in SOURCE_PREFIXES:
        t = pq.read_table(normalized_path("train", s, cfg), columns=fields)
        if s in TARGET_SOURCES:
            for f in counts:
                tc = token_counts(t[f])
                tc = tc.filter(pc.greater_equal(tc["count"], cfg.eda_token_keep_min))
                for tok, c in zip(tc["token"].to_pylist(), tc["count"].to_pylist()):
                    counts[f][tok] = counts[f].get(tok, 0) + c
        wanted = sp["s1_id"] if s == 1 else sp.loc[sp["src"] == s, "tgt_id"]
        frames[s] = t.filter(pc.is_in(t["entity_id"], value_set=pa.array(wanted.drop_duplicates().tolist(),
                                                                          pa.string()))).to_pandas()
        del t
    s1 = frames[1].add_prefix("s1_")
    tg = pd.concat([frames[2], frames[3]], ignore_index=True).add_prefix("t_")
    df = sp.merge(s1, left_on="s1_id", right_on="s1_entity_id").merge(tg, left_on="tgt_id", right_on="t_entity_id")

    hub = cfg.eda_hub_token_count

    def shares(a: pd.Series, b: pd.Series, f: str | None) -> np.ndarray:
        cmap = counts[f] if f else None
        res = np.zeros(len(a), dtype=bool)
        for i, (x, y) in enumerate(zip(a.tolist(), b.tolist())):
            inter = set(x.split()) & set(y.split())
            res[i] = bool(inter) if cmap is None else any(cmap.get(t, 0) < hub for t in inter)
        return res

    def prefix(a: pd.Series, b: pd.Series, k: int = 6) -> np.ndarray:
        return np.array([len(x) >= k and len(y) >= k and (x.startswith(y) or y.startswith(x))
                         for x, y in zip(a.tolist(), b.tolist())])

    sk_set = lambda s: s.map(lambda v: " ".join(sorted(set(v.split()))))  # noqa: E731
    df = df.assign(
        core_equal=(df["s1_name_core"] == df["t_name_core"]) & (df["s1_name_core"] != ""),
        compact_equal=(df["s1_name_compact"] == df["t_name_compact"]) & (df["s1_name_compact"] != ""),
        compact_prefix=prefix(df["s1_name_compact"], df["t_name_compact"]),
        skel_set_equal=(sk_set(df["s1_name_skel"]) == sk_set(df["t_name_skel"])) & (df["s1_name_skel"] != ""),
        core_rare=shares(df["s1_name_core"], df["t_name_core"], "name_core"),
        alt_core_rare=shares(df["s1_name_core"], df["t_name_alt_core"], "name_core"),
        skel_rare=shares(df["s1_name_skel"], df["t_name_skel"], "name_skel"),
        addr_rare=shares(df["s1_addr_core"], df["t_addr_core"], "addr_core"),
        numbers_share=shares(df["s1_addr_numbers"], df["t_addr_numbers"], None),
        numbers_rare=shares(df["s1_addr_numbers"], df["t_addr_numbers"], "addr_numbers"),
        region_both=(df["s1_addr_region"] != "") & (df["t_addr_region"] != ""),
        legal_equal=df["s1_name_legal"].map(lambda v: set(v.split())) == df["t_name_legal"].map(lambda v: set(v.split())),
    )
    df["region_equal_when_both"] = np.where(df["region_both"], df["s1_addr_region"] == df["t_addr_region"], np.nan)
    df["name_signal"] = df["core_rare"] | df["alt_core_rare"] | df["skel_rare"] | df["compact_equal"] | df["compact_prefix"]
    df["addr_signal"] = df["addr_rare"] | df["numbers_rare"]
    df["no_signal"] = ~(df["name_signal"] | df["addr_signal"])
    metrics = ["core_equal", "compact_equal", "compact_prefix", "skel_set_equal", "core_rare", "alt_core_rare",
               "skel_rare", "addr_rare", "numbers_share", "numbers_rare", "region_both", "region_equal_when_both",
               "legal_equal", "name_signal", "addr_signal", "no_signal"]
    groups = []
    df["all"] = "all"
    for keys in (["all"], ["src"], ["s1_country"], ["t_name_nonlatin"], ["t_addr_empty"]):
        g = df.groupby(keys)[metrics].mean().round(4)
        g.insert(0, "pairs", df.groupby(keys).size())
        g = g.reset_index().rename(columns={keys[0]: "group"})
        g.insert(0, "group_by", keys[0])
        groups.append(g)
    table = pd.concat(groups, ignore_index=True)
    table.to_csv(out_dir / "norm_check.csv", index=False)
    cols = ["s1_entity_id", "s1_name_core", "s1_addr_core", "s1_addr_numbers", "t_entity_id", "t_name_core",
            "t_name_skel", "t_addr_core", "t_addr_numbers"]
    df.loc[df["no_signal"], cols].head(200).to_csv(out_dir / "norm_check_no_signal.tsv", sep="\t", index=False)
    LOG.info("normalization check:\n%s", table.to_string(index=False))
    return {"pairs": int(len(df)), "table": table.to_dict(orient="records")}
