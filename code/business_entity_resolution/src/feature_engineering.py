"""Pairwise features for final candidates (Phase 5).

Input : candidates/<name>.parquet (the final set, rank < blocking_top_k per source)
Output: features/<name>.parquet  -> s1_row, tgt, src, [label], feature columns (float32)

Feature groups
--------------
name     edit/token similarities on name_norm, name_core, name_skel (cross-script), compact
         Jaro-Winkler and prefix cover (domains), trade-name (dba) similarity, token-count gap,
         legal-form agreement/conflict, target non-Latin flag
address  token-set/sort/partial similarity on addr_core, digit-run overlap, first-number
         equality, long-number (postal-like) overlap, region agreement, empty-address flags
blocking key score, re-rank score, ranks, number of passes, one bit per pass
context  gap to the best candidate of the same Source-1 entity (rr, name, address) and the
         candidate count -- lets the model see "is anything clearly better?" (singletons)
sibling  name/address similarity of this target to the entity's best OTHER candidate: true
         matches of one business are copies of each other, distractors usually are not
compete  target-side competition over ALL Source-1 entities of the split: how many entities
         hold this target as a candidate, this entity's rank and gap for it, mutual-best flag
         (EDA: no train target is matched to two Source-1 entities)

Missing information is NaN, not 0 (e.g. address similarity when either address is empty):
gradient-boosted trees treat NaN as its own branch. No country one-hot: the test set has an
unseen label (France), so features describe the records, not the country.

Cost: RapidFuzz ``cpdist`` (C++, multithreaded) per string feature; one light Python pass over
digit strings. Processed in entity-aligned batches of ``feature_batch_rows`` pairs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .blocking import PASSES
from .candidate_generation import LazyTargets, candidates_path
from .config import CFG, Config
from .preprocessing import normalized_path
from .utils import timed

LOG = logging.getLogger("ber.features")

COMMON_COLS = ["name_norm", "name_core", "name_compact", "name_skel", "name_legal",
               "addr_core", "addr_numbers", "addr_region", "addr_empty"]
TARGET_ONLY = ["name_alt_core", "name_nonlatin"]
CONTEXT_OF = ["rr", "core_tsr", "skel_tsr", "addr_tset", "score"]


def features_path(name: str, cfg: Config = CFG) -> Path:
    return cfg.artifacts_dir / "features" / f"{name}.parquet"


def _cpdist(a: list, b: list, scorer) -> np.ndarray:
    from rapidfuzz import process
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _digit_features(a: list[str], b: list[str]) -> tuple[np.ndarray, ...]:
    """Digit-run overlap (Jaccard), first-number equality, shared long (>=5 digit) number. NaN if missing."""
    n = len(a)
    jac, first, longs = (np.full(n, np.nan, np.float32) for _ in range(3))
    for i, (x, y) in enumerate(zip(a, b)):
        if not x or not y:
            continue
        sx, sy = x.split(), y.split()
        X, Y = set(sx), set(sy)
        jac[i] = len(X & Y) / len(X | Y)
        first[i] = sx[0] == sy[0]
        lx = {t for t in X if len(t) >= 5}
        ly = {t for t in Y if len(t) >= 5}
        if lx and ly:
            longs[i] = bool(lx & ly)
    return jac, first, longs


def _to_int(x: str) -> int:
    return int(x[-9:])  # digit runs; cap length so int() stays cheap and bounded


def _hard_negative_features(da: list[str], db: list[str], ca: list[str], cb: list[str]) -> dict[str, np.ndarray]:
    """Features that separate *noise* from *a different nearby business*.

    Error analysis (valid holdout) showed two confusable patterns:
      * true matches whose house number is corrupted by one edit (1425 -> 2425, 1320 -> 320)
        vs. false matches at a *neighbouring* number on the same street (1302 vs 1304,
        5833 vs 5842): edit distance and numeric distance of the numbers tell them apart;
      * true matches with a typo in one name token (Kanara Marketing / PFDFRIIDCTE) vs. false
        matches with a genuinely different token (Balaji Agro / Balaji Arto, Curo Processors /
        Curo Unique): the character similarity of the *differing* tokens tells them apart.
    NaN when the information is missing on either side.
    """
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz import fuzz

    n = len(da)
    out = {k: np.full(n, np.nan, np.float32) for k in (
        "num_first_edit", "num_first_absdiff_log", "num_first_samelen", "num_best_edit", "num_best_absdiff_log",
        "name_diff_tokens_a", "name_diff_tokens_b", "name_diff_best_sim", "name_diff_worst_sim")}
    for i in range(n):
        x, y = da[i], db[i]
        if x and y:
            sx, sy = x.split()[:3], y.split()[:3]
            fx, fy = sx[0], sy[0]
            out["num_first_edit"][i] = Levenshtein.distance(fx, fy)
            out["num_first_absdiff_log"][i] = np.log1p(abs(_to_int(fx) - _to_int(fy)))
            out["num_first_samelen"][i] = len(fx) == len(fy)
            out["num_best_edit"][i] = min(Levenshtein.distance(u, v) for u in sx for v in sy)
            out["num_best_absdiff_log"][i] = np.log1p(min(abs(_to_int(u) - _to_int(v)) for u in sx for v in sy))
        u, v = ca[i], cb[i]
        if u and v:
            A, B = set(u.split()), set(v.split())
            ra, rb = A - B, B - A
            out["name_diff_tokens_a"][i] = len(ra)
            out["name_diff_tokens_b"][i] = len(rb)
            if ra and rb:
                sims = [max(fuzz.ratio(t, w) for w in rb) for t in ra]
                out["name_diff_best_sim"][i] = max(sims)
                out["name_diff_worst_sim"][i] = min(sims)
    return out


def pair_features(a: dict[str, pa.Array], b: dict[str, pa.Array]) -> dict[str, np.ndarray]:
    """String features for aligned arrays ``a`` (Source 1) and ``b`` (targets)."""
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler, Prefix

    L = {k: v.to_pylist() for k, v in a.items()}
    R = {k: v.to_pylist() for k, v in b.items()}
    f: dict[str, np.ndarray] = {}
    f["name_ratio"] = _cpdist(L["name_norm"], R["name_norm"], fuzz.ratio)
    f["core_tsr"] = _cpdist(L["name_core"], R["name_core"], fuzz.token_sort_ratio)
    f["core_tset"] = _cpdist(L["name_core"], R["name_core"], fuzz.token_set_ratio)
    f["core_partial"] = _cpdist(L["name_core"], R["name_core"], fuzz.partial_ratio)
    f["skel_tsr"] = _cpdist(L["name_skel"], R["name_skel"], fuzz.token_sort_ratio)
    f["skel_tset"] = _cpdist(L["name_skel"], R["name_skel"], fuzz.token_set_ratio)
    f["compact_jw"] = _cpdist(L["name_compact"], R["name_compact"], JaroWinkler.normalized_similarity) * 100
    pre = _cpdist(L["name_compact"], R["name_compact"], Prefix.similarity)
    minlen = np.minimum(pc.utf8_length(a["name_compact"]).to_numpy(), pc.utf8_length(b["name_compact"]).to_numpy())
    f["compact_prefix_cover"] = np.where(minlen > 0, pre / np.maximum(minlen, 1), np.nan).astype(np.float32)
    f["compact_eq"] = (pc.equal(a["name_compact"], b["name_compact"]).to_numpy(zero_copy_only=False)
                       & (minlen > 0)).astype(np.float32)
    alt = np.array([bool(x) for x in R["name_alt_core"]])
    f["alt_tsr"] = np.where(alt, _cpdist(L["name_core"], R["name_alt_core"], fuzz.token_sort_ratio), np.nan)
    na = pc.list_value_length(pc.split_pattern(a["name_core"], " ")).to_numpy()
    nb = pc.list_value_length(pc.split_pattern(b["name_core"], " ")).to_numpy()
    f["core_ntok_diff"] = np.abs(na - nb).astype(np.float32)
    la = np.array([bool(x) for x in L["name_legal"]])
    lb = np.array([bool(x) for x in R["name_legal"]])
    legal = _cpdist(L["name_legal"], R["name_legal"], fuzz.token_set_ratio)
    both = la & lb
    f["legal_both"] = both.astype(np.float32)
    f["legal_match"] = np.where(both, legal / 100, np.nan)
    f["t_name_nonlatin"] = b["name_nonlatin"].to_numpy(zero_copy_only=False).astype(np.float32)

    ea = a["addr_empty"].to_numpy(zero_copy_only=False)
    eb = b["addr_empty"].to_numpy(zero_copy_only=False)
    miss = ea | eb
    f["t_addr_empty"] = eb.astype(np.float32)
    for nm, scorer in (("addr_tset", fuzz.token_set_ratio), ("addr_tsr", fuzz.token_sort_ratio),
                       ("addr_partial", fuzz.partial_token_set_ratio)):
        f[nm] = np.where(miss, np.nan, _cpdist(L["addr_core"], R["addr_core"], scorer))
    digits_a = pc.utf8_trim_whitespace(pc.replace_substring_regex(a["addr_numbers"], r"[^0-9]+", " ")).to_pylist()
    digits_b = pc.utf8_trim_whitespace(pc.replace_substring_regex(b["addr_numbers"], r"[^0-9]+", " ")).to_pylist()
    f["num_jaccard"], f["num_first_eq"], f["num_long_shared"] = _digit_features(digits_a, digits_b)
    f.update(_hard_negative_features(digits_a, digits_b, L["name_core"], R["name_core"]))
    ra, rb = np.array(L["addr_region"], dtype=object), np.array(R["addr_region"], dtype=object)
    f["region_match"] = np.where((ra == "") | (rb == ""), np.nan, (ra == rb).astype(np.float32))
    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}


def _context(df: pd.DataFrame) -> pd.DataFrame:
    """Per Source-1 entity: gap to the entity's best value and candidate counts."""
    g = df.groupby("s1_row", sort=False)
    for c in CONTEXT_OF:
        df[f"{c}_gap_to_best"] = (df[c] - g[c].transform("max")).astype(np.float32)
    df["n_cands_entity"] = g["tgt"].transform("size").astype(np.float32)
    gs = df.groupby(["s1_row", "src"], sort=False)
    df["rr_gap_to_best_same_src"] = (df["rr"] - gs["rr"].transform("max")).astype(np.float32)
    df["rr_rank_in_src"] = gs["rr"].rank(ascending=False, method="first").astype(np.float32)
    return df


COMPETE_COLS = ["tgt_n_s1", "tgt_rr_rank", "tgt_rr_gap_to_best", "tgt_rr_gap_to_second", "mutual_best"]


def competition_index(split: str, names: tuple[str, ...], cfg: Config = CFG) -> dict:
    """Target-competition statistics over the union of candidate sets ``names`` of a split.

    Uses only the unsupervised re-rank score ``rr`` (no labels), so computing it over train and
    validation entities together does not leak. ``names`` must cover every Source-1 entity of
    the split so train and test see the same competition (checked).
    """
    from .candidate_generation import queried_rows
    parts = [pq.read_table(candidates_path(n, cfg), columns=["s1_row", "tgt", "src", "rr"]) for n in names]
    t = pa.concat_tables(parts)
    covered = np.unique(np.concatenate([queried_rows(n, cfg) for n in names]))
    n_s1 = pq.ParquetFile(normalized_path(split, 1, cfg)).metadata.num_rows
    if len(covered) != n_s1:
        raise ValueError(f"competition needs all {n_s1} Source-1 entities, candidate sets cover {len(covered)}")
    s1 = t["s1_row"].to_numpy().astype(np.int64)
    tg = t["tgt"].to_numpy().astype(np.int64)
    src = t["src"].to_numpy().astype(np.int64)
    rr = t["rr"].to_numpy().astype(np.float32)
    del t, parts
    n_t = LazyTargets(split, cfg).num_rows
    # per target: order entities by rr (desc)
    o = np.lexsort((s1, -rr, tg))
    tg_o = tg[o]
    new = np.concatenate([[True], tg_o[1:] != tg_o[:-1]])
    starts = np.flatnonzero(new)
    grp = np.cumsum(new) - 1
    rank_t = np.empty(len(o), np.float32)
    rank_t[o] = np.arange(len(o)) - starts[grp] + 1
    best = np.empty(len(o), np.float32)
    best[o] = rr[o][starts][grp]
    counts = np.diff(np.concatenate([starts, [len(o)]]))
    second_vals = np.where(counts > 1, rr[o][np.minimum(starts + 1, len(o) - 1)], np.nan)
    second = np.empty(len(o), np.float32)
    second[o] = second_vals[grp]
    n_s1_t = np.empty(len(o), np.float32)
    n_s1_t[o] = counts[grp]
    # per (entity, source): is this the entity's best candidate in that source?
    o2 = np.lexsort((tg, -rr, src, s1))
    g2 = s1[o2] * 4 + src[o2]
    first2 = np.concatenate([[True], g2[1:] != g2[:-1]])
    best_in_ent = np.zeros(len(o2), bool)
    best_in_ent[o2[first2]] = True
    key = s1 * n_t + tg
    ok = np.argsort(key)
    vals = {"tgt_n_s1": n_s1_t, "tgt_rr_rank": rank_t, "tgt_rr_gap_to_best": rr - best,
            # for the best entity: margin over the runner-up (NaN if uncontested)
            "tgt_rr_gap_to_second": np.where(rank_t == 1, rr - second, rr - best).astype(np.float32),
            "mutual_best": ((rank_t == 1) & best_in_ent).astype(np.float32)}
    LOG.info("competition index: %d pairs, %d targets, targets held by >1 entity: %.4f",
             len(key), len(starts), float((counts > 1).mean()))
    return {"keys": key[ok], "n_t": n_t, **{k: v[ok] for k, v in vals.items()}}


def _competition_lookup(comp: dict, s1_row: np.ndarray, tgt: np.ndarray) -> dict[str, np.ndarray]:
    key = s1_row.astype(np.int64) * comp["n_t"] + tgt.astype(np.int64)
    pos = np.searchsorted(comp["keys"], key)
    if not np.array_equal(comp["keys"][np.minimum(pos, len(comp["keys"]) - 1)], key):
        raise ValueError("candidate pairs missing from the competition index")
    return {c: comp[c][pos] for c in COMPETE_COLS}


def _sibling(df: pd.DataFrame, tg: dict[str, pa.Array]) -> dict[str, np.ndarray]:
    """Similarity of each candidate target to the entity's best other candidate (by rr)."""
    from rapidfuzz import fuzz
    n = len(df)
    order = np.lexsort((df["tgt"].to_numpy(), -df["rr"].to_numpy(), df["s1_row"].to_numpy()))
    rows = df["s1_row"].to_numpy()[order]
    first = np.concatenate([[True], rows[1:] != rows[:-1]])
    starts = np.flatnonzero(first)
    grp = np.cumsum(first) - 1
    top = order[starts][grp]                                   # entity's best candidate
    second_pos = np.minimum(starts + 1, n - 1)
    has_second = np.concatenate([starts[1:], [n]]) - starts > 1
    second = np.where(has_second, order[second_pos], -1)[grp]
    ref_sorted = np.where(order == top, second, top)           # the best candidate other than itself
    ref = np.empty(n, np.int64)
    ref[order] = ref_sorted
    valid = ref >= 0
    tgt = df["tgt"].to_numpy()
    ref_tgt = np.where(valid, tgt[np.maximum(ref, 0)], 0)
    out = {}
    for nm, col, scorer in (("sib_name_tsr", "name_core", fuzz.token_sort_ratio),
                            ("sib_addr_tset", "addr_core", fuzz.token_set_ratio)):
        a = tg[col].take(pa.array(tgt)).to_pylist()
        b = tg[col].take(pa.array(ref_tgt)).to_pylist()
        v = _cpdist(a, b, scorer)
        if col == "addr_core":
            empty = np.array([not x for x in a]) | np.array([not x for x in b])
            v[empty] = np.nan
        v[~valid] = np.nan
        out[nm] = v
    out["sib_ref_rr"] = np.where(valid, df["rr"].to_numpy()[np.maximum(ref, 0)], np.nan).astype(np.float32)
    return out


def build_features(split: str, name: str, cfg: Config = CFG, truth: pa.Table | None = None,
                   comp: dict | None = None) -> Path:
    """Compute features for candidates/<name>.parquet; attach labels when ``truth`` is given."""
    cands = pq.read_table(candidates_path(name, cfg)).sort_by([("s1_row", "ascending"), ("src", "ascending"),
                                                               ("rank", "ascending")])
    s1 = pq.read_table(normalized_path(split, 1, cfg), columns=COMMON_COLS)
    s1 = {c: pa.concat_arrays(s1[c].chunks) for c in COMMON_COLS}
    lt = LazyTargets(split, cfg)
    tg = {c: pa.concat_arrays(lt[c].chunks) for c in COMMON_COLS + TARGET_ONLY}
    truth_keys = None
    if truth is not None:
        truth_keys = np.sort(truth["s1_row"].to_numpy().astype(np.int64) * lt.num_rows
                             + truth["tgt"].to_numpy().astype(np.int64))

    out = features_path(name, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    rows = cands["s1_row"].to_numpy()
    starts = np.flatnonzero(np.concatenate([[True], rows[1:] != rows[:-1]])) if len(rows) else np.zeros(0, int)
    n = cands.num_rows
    writer, done = None, 0
    with timed(f"features {name}: {n} pairs"):
        while done < n:
            j = np.searchsorted(starts, done + cfg.feature_batch_rows, side="left")  # cut at an entity start
            end = int(starts[j]) if j < len(starts) else n
            batch = cands.slice(done, end - done)
            ia, ib = batch["s1_row"].combine_chunks(), batch["tgt"].combine_chunks()
            feats = pair_features({c: s1[c].take(ia) for c in COMMON_COLS},
                                  {c: tg[c].take(ib) for c in COMMON_COLS + TARGET_ONLY})
            df = batch.select(["s1_row", "tgt", "src", "score", "rr", "key_rank", "rank"]).to_pandas()
            passes = batch["passes"].to_numpy()
            for i, p in enumerate(PASSES):
                df[f"pass_{p}"] = ((passes >> i) & 1).astype(np.float32)
            df["n_passes"] = sum(df[f"pass_{p}"] for p in PASSES)
            for k, v in feats.items():
                df[k] = v
            df = _context(df)
            for k, v in _sibling(df, tg).items():
                df[k] = v
            if comp is not None:
                for k, v in _competition_lookup(comp, df["s1_row"].to_numpy(), df["tgt"].to_numpy()).items():
                    df[k] = v
            if truth_keys is not None and len(truth_keys):
                key = df["s1_row"].to_numpy().astype(np.int64) * lt.num_rows + df["tgt"].to_numpy().astype(np.int64)
                pos = np.minimum(np.searchsorted(truth_keys, key), len(truth_keys) - 1)
                df["label"] = (truth_keys[pos] == key).astype(np.int8)
            elif truth_keys is not None:
                df["label"] = np.zeros(len(df), np.int8)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table)
            done = end
            LOG.info("  %s: %d/%d pairs", name, done, n)
    if writer is not None:
        writer.close()
        tmp.replace(out)
    return out


def feature_columns(schema_names: list[str]) -> list[str]:
    """Model inputs: everything except ids and the label (same order at train and inference)."""
    return [c for c in schema_names if c not in ("s1_row", "tgt", "label")]


def feature_summary(name: str, cfg: Config = CFG) -> pd.DataFrame:
    """Per-feature mean / NaN share by label (sanity check that features separate matches)."""
    df = pq.read_table(features_path(name, cfg)).to_pandas()
    cols = feature_columns(list(df.columns))
    g = df.groupby("label")[cols]
    out = pd.concat({"mean": g.mean().T, "nan_share": g.agg(lambda x: x.isna().mean()).T}, axis=1)
    out.columns = [f"{a}_label{b}" for a, b in out.columns]
    path = cfg.artifacts_dir / "reports" / f"feature_summary_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(path)
    LOG.info("%s: %d pairs, positive rate %.4f, %d features -> %s", name, len(df), df["label"].mean(), len(cols), path)
    return out
