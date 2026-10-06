"""Candidate generation: build target block indexes, query Source-1 in chunks,
union + score + prune to top-K per source (Phase 3).

Memory plan (train, est.): target normalised columns ~2 GB while indexes are
built, then freed; kept indexes ~1-2 GB (hub blocks removed); per query chunk
of 20k entities a few hundred MB. Candidates stream to parquet chunk by chunk.

Scoring (pre-ranker only, not the matcher): a candidate's score is the sum over
the keys it shares with the Source-1 record of  pass_weight * log(1 + N / block_size).
Rare shared keys dominate; several independent shared keys add up.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .blocking import PASS_ID, BlockIndex, Tokens, compose, cross, dedup, head, tokenize
from .config import CFG, TARGET_SOURCES, Config
from .preprocessing import normalized_path
from .utils import peak_rss_mb, save_json, timed

LOG = logging.getLogger("ber.candidates")

KEY_COLUMNS = ["country_key", "name_core", "name_alt_core", "name_skel", "name_compact", "addr_core", "addr_numbers"]
CANDIDATE_SCHEMA = pa.schema([("s1_row", pa.int32()), ("tgt", pa.int32()), ("src", pa.int8()),
                              ("score", pa.float32()), ("rr", pa.float32()), ("key_rank", pa.int16()),
                              ("rank", pa.int16()), ("passes", pa.int16())])
TEXT_COLUMNS = ["name_core", "name_skel", "addr_core"]  # used by the re-ranker


ELISION_RE = r"\b([dl])([aeiouh][a-z]{3,})\b"


def _skel_set(col: pa.ChunkedArray) -> pa.ChunkedArray:
    """Sorted unique skeleton tokens per record (Python only over unique values)."""
    out = []
    for ch in col.chunks:
        enc = pc.dictionary_encode(ch)
        mapped = pa.array([" ".join(sorted(set(v.split()))) for v in enc.dictionary.to_pylist()], pa.string())
        out.append(mapped.take(enc.indices))
    return pa.chunked_array(out, pa.string())


def _digit_runs(col: pa.ChunkedArray) -> pa.ChunkedArray:
    return pc.utf8_trim_whitespace(pc.replace_substring_regex(col, r"[^0-9]+", " "))


def _prefix(col: pa.ChunkedArray, n: int) -> pa.ChunkedArray:
    return pc.if_else(pc.greater_equal(pc.utf8_length(col), n), pc.utf8_slice_codeunits(col, 0, n), "")


class KeyBuilder:
    """Generates (record, key) arrays per pass. ``fit`` on targets, then query Source 1."""

    def __init__(self, cfg: Config, passes: tuple[str, ...]):
        unknown = set(passes) - set(PASS_ID)
        if unknown:
            raise ValueError(f"unknown passes {sorted(unknown)}")
        self.cfg, self.passes = cfg, passes
        self.vocab: dict[str, pa.Array] = {}
        self.anchor_ok: np.ndarray | None = None
        self.street_ok: np.ndarray | None = None
        self.fitted = False

    def _tok(self, name: str, col, whole: bool = False) -> Tokens:
        vocab = self.vocab.get(name) if self.fitted else None
        tok, v = tokenize(col, vocab, whole)
        if not self.fitted:
            self.vocab[name] = v
        return tok

    def _country(self, t: pa.Table) -> np.ndarray:
        if not self.fitted:
            enc = pc.dictionary_encode(pa.concat_arrays(t["country_key"].chunks))
            if len(enc.dictionary) > 64:
                raise ValueError("more than 64 country labels; raise N_COUNTRY_SLOTS")
            self.vocab["country"] = enc.dictionary
            return enc.indices.to_numpy().astype(np.int64)
        idx = pc.index_in(t["country_key"], value_set=self.vocab["country"])
        return pc.fill_null(idx, -1).to_numpy().astype(np.int64)  # -1: label unseen in targets

    def keys(self, t: pa.Table):
        """Yield (pass, record, key). First call (unfitted) builds vocabularies."""
        cfg, n = self.cfg, t.num_rows
        country = self._country(t)
        P = set(self.passes)

        need_name = P & {"name_tok", "name_pair", "name_num"}
        need_skel = P & {"skel_pair", "skel_num", "skel_anchor"}
        need_addr = P & {"addr_tok", "num_anchor", "skel_anchor", "num_street"}
        need_nums = P & {"num_anchor", "name_num", "skel_num", "num_street"}
        k_combo = cfg.blocking_combo_name_tokens

        name = dedup(self._tok("name", t["name_core"])) if need_name else None
        if "name_tok" in P:
            alt, _ = tokenize(t["name_alt_core"], self.vocab["name"])
            both = Tokens(np.concatenate([name.parent, alt.parent]), np.concatenate([name.code, alt.code]))
            order = np.argsort(both.parent, kind="stable")
            both = dedup(Tokens(both.parent[order], both.code[order]))
            del alt, order
            yield "name_tok", both.parent, compose(both.code, country[both.parent], "name_tok")
            del both
        if "name_pair" in P:
            h = head(name, cfg.blocking_max_name_tokens)
            rec, val = cross(h, h, n, len(self.vocab["name"]), upper_only=True)
            yield "name_pair", rec, compose(val, country[rec], "name_pair")
            del rec, val
        if "skel_set" in P:
            tok = self._tok("skel_set", _skel_set(t["name_skel"]), whole=True)
            yield "skel_set", tok.parent, compose(tok.code, country[tok.parent], "skel_set")
            del tok
        skel = dedup(self._tok("skel", t["name_skel"])) if need_skel else None
        if "skel_pair" in P:
            sk = head(skel, cfg.blocking_max_name_tokens)
            rec, val = cross(sk, sk, n, len(self.vocab["skel"]), upper_only=True)
            yield "skel_pair", rec, compose(val, country[rec], "skel_pair")
            del rec, val
        if "compact" in P:
            tok = self._tok("compact", t["name_compact"], whole=True)
            yield "compact", tok.parent, compose(tok.code, country[tok.parent], "compact")
            del tok
        if "compact_prefix" in P:
            tok = self._tok("cprefix", _prefix(t["name_compact"], cfg.blocking_compact_prefix_len), whole=True)
            yield "compact_prefix", tok.parent, compose(tok.code, country[tok.parent], "compact_prefix")
            del tok

        nums = head(dedup(self._tok("digits", _digit_runs(t["addr_numbers"]))), cfg.blocking_max_numbers) \
            if need_nums else None
        if need_addr:
            addr_col = t["addr_core"]
            if "num_street" in P:
                # elided articles get an extra variant token ("dajaccio" -> also "ajaccio",
                # "lembellie" -> also "embellie") so French d'/l' spellings with and without the
                # apostrophe share a token. Done once here (one tokenisation, bounded memory).
                addr_col = pc.replace_substring_regex(addr_col, ELISION_RE, r"\1\2 \2")
            addr = dedup(self._tok("addr", addr_col))
            del addr_col
            if not self.fitted:  # anchors: common alphabetic tokens; street words: uncommon ones
                df = np.bincount(addr.code, minlength=len(self.vocab["addr"]))
                alpha = pc.match_substring_regex(self.vocab["addr"], r"\p{L}").to_numpy(zero_copy_only=False)
                lengthy = pc.greater_equal(pc.utf8_length(self.vocab["addr"]), 3).to_numpy(zero_copy_only=False)
                self.anchor_ok = alpha & (df >= cfg.blocking_anchor_min_df)
                self.street_ok = alpha & lengthy & (df < cfg.blocking_anchor_min_df)
                del df, alpha, lengthy
            if "addr_tok" in P:
                yield "addr_tok", addr.parent, compose(addr.code, country[addr.parent], "addr_tok")
            anchors = addr.filter(self.anchor_ok[addr.code])
            street = head(addr.filter(self.street_ok[addr.code]), cfg.blocking_max_street_tokens) \
                if "num_street" in P else None
            del addr
            if "num_street" in P:  # house number x uncommon street word ("18" x "ajaccio")
                rec, val = cross(nums, street, n, len(self.vocab["addr"]))
                del street
                yield "num_street", rec, compose(val, country[rec], "num_street")
                del rec, val
            if "num_anchor" in P:
                rec, val = cross(nums, head(anchors, cfg.blocking_max_anchors), n, len(self.vocab["addr"]))
                yield "num_anchor", rec, compose(val, country[rec], "num_anchor")
                del rec, val
            if "skel_anchor" in P:
                rec, val = cross(head(skel, k_combo), head(anchors, cfg.blocking_combo_anchors), n,
                                 len(self.vocab["addr"]))
                yield "skel_anchor", rec, compose(val, country[rec], "skel_anchor")
                del rec, val
            del anchors
        if "name_num" in P:
            rec, val = cross(head(name, k_combo), nums, n, len(self.vocab["digits"]))
            yield "name_num", rec, compose(val, country[rec], "name_num")
            del rec, val
        if "skel_num" in P:
            rec, val = cross(head(skel, k_combo), nums, n, len(self.vocab["digits"]))
            yield "skel_num", rec, compose(val, country[rec], "skel_num")
            del rec, val
        self.fitted = True


@dataclass
class TargetIndex:
    builder: KeyBuilder
    indexes: dict[str, BlockIndex]
    n2: int
    n_targets: int
    stats: dict = field(default_factory=dict)
    text: dict[str, pa.Array] = field(default_factory=dict)  # target text columns for re-ranking


class LazyTargets:
    """S2+S3 normalised columns read on demand (one column in memory at a time, not the whole table)."""

    def __init__(self, split: str, cfg: Config):
        self.paths = [normalized_path(split, s, cfg) for s in TARGET_SOURCES]
        rows = [pq.ParquetFile(p).metadata.num_rows for p in self.paths]
        self.n2, self.num_rows = rows[0], sum(rows)

    def __getitem__(self, column: str) -> pa.ChunkedArray:
        return pa.chunked_array([c for p in self.paths for c in pq.read_table(p, columns=[column])[column].chunks])


def build_target_index(split: str, cfg: Config = CFG, passes: tuple[str, ...] | None = None) -> TargetIndex:
    passes = passes or cfg.blocking_passes
    tgt = LazyTargets(split, cfg)
    n2, n_t = tgt.n2, tgt.num_rows
    kb = KeyBuilder(cfg, passes)
    indexes = {}
    for p, rec, key in kb.keys(tgt):
        idx = BlockIndex.build(rec, key, cfg.blocking_max_block, n_t)
        idx.stats["peak_rss_mb"] = round(peak_rss_mb() or 0)
        del rec, key
        indexes[p] = idx
        LOG.info("pass %-15s %s | index %.0f MB", p, idx.stats, idx.nbytes / 2**20)
    text = {c: pa.concat_arrays(tgt[c].chunks) for c in TEXT_COLUMNS} if cfg.blocking_rerank else {}
    del tgt
    stats = {p: i.stats for p, i in indexes.items()}
    return TargetIndex(kb, indexes, n2, n_t, stats, text)


def _group_rank(group: np.ndarray, primary: np.ndarray, tie: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order by (group, -primary, tie); return (order, 0-based rank within group in that order)."""
    order = np.lexsort((tie, -primary, group))
    g = group[order]
    new = np.concatenate([[True], g[1:] != g[:-1]])
    rank = np.arange(len(g)) - np.flatnonzero(new)[np.cumsum(new) - 1]
    return order, rank


def rerank_scores(ti: TargetIndex, s1: pa.Table, qi: np.ndarray, tg: np.ndarray, key_score: np.ndarray,
                  cfg: Config) -> np.ndarray:
    """Cheap string-similarity re-ranking score in [0, 1] for union pairs (RapidFuzz, C++, multithreaded).

    rr = w_name * max(token_sort(name_core), token_sort(name_skel)) + w_addr * token_set(addr_core)
         + w_key * min(key_score / key_norm, 1);  addr term = 0.5 when either address is empty.
    Hand-set weights (config); used only to order candidates before the top-K cut.
    """
    from rapidfuzz import fuzz, process

    w = cfg.blocking_rerank_weights
    out = np.zeros(len(qi), dtype=np.float32)
    for col, scorer in (("name_core", fuzz.token_sort_ratio), ("name_skel", fuzz.token_sort_ratio)):
        a = s1[col].to_numpy(zero_copy_only=False)[qi].tolist()
        b = ti.text[col].take(pa.array(tg)).to_pylist()
        sim = process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / 100.0
        out = np.maximum(out, sim)
    out *= w["name"]
    a = s1["addr_core"].to_numpy(zero_copy_only=False)[qi]
    b = ti.text["addr_core"].take(pa.array(tg)).to_pylist()
    addr = process.cpdist(a.tolist(), b, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0
    empty = (a == "") | (np.array(b, dtype=object) == "")
    addr[empty] = 0.5
    out += w["addr"] * addr
    out += w["key"] * np.minimum(key_score / cfg.blocking_rerank_key_norm, 1.0)
    return out


def query_chunk(ti: TargetIndex, s1: pa.Table, cfg: Config, store_k: int) -> tuple[pa.Table, np.ndarray]:
    """Candidates for one chunk of Source-1 rows (``s1`` must contain a 'row' column).

    Stages: union of all passes -> key-score top ``blocking_union_k`` per source -> optional
    re-rank -> keep rows whose final rank < ``store_k``.
    Returns the table and the per-(entity, source) union sizes before any cut.
    """
    n = s1.num_rows
    parts_i, parts_t, parts_w, parts_b = [], [], [], []
    for p, rec, key in ti.builder.keys(s1):
        ok = key >= 0  # country unseen in targets -> negative key
        qi, tg, w = ti.indexes[p].expand(rec[ok], key[ok])
        parts_i.append(qi)
        parts_t.append(tg)
        parts_w.append(w * cfg.blocking_pass_weight.get(p, 1.0))
        parts_b.append(np.full(len(qi), 1 << PASS_ID[p], dtype=np.int16))
    qi, tg = np.concatenate(parts_i).astype(np.int64), np.concatenate(parts_t).astype(np.int64)
    w, bits = np.concatenate(parts_w), np.concatenate(parts_b)
    if len(qi) == 0:
        return CANDIDATE_SCHEMA.empty_table(), np.zeros(0, np.int64)

    pair = qi * ti.n_targets + tg
    order = np.argsort(pair, kind="stable")
    pair, w, bits = pair[order], w[order], bits[order]
    starts = np.flatnonzero(np.concatenate([[True], pair[1:] != pair[:-1]]))
    score = np.add.reduceat(w, starts)
    passes = np.bitwise_or.reduceat(bits, starts)
    upair = pair[starts]
    del pair, w, bits, order
    qi, tg = upair // ti.n_targets, upair % ti.n_targets
    src = np.where(tg < ti.n2, 2, 3).astype(np.int8)
    group = qi * 2 + (src - 2)
    union_sizes = np.bincount(group, minlength=2 * n)

    order, key_rank = _group_rank(group, score, tg)
    keep = key_rank < cfg.blocking_union_k
    sel = order[keep]
    group, qi, tg, src, score, passes, key_rank = (group[sel], qi[sel], tg[sel], src[sel], score[sel],
                                                   passes[sel], key_rank[keep])
    if cfg.blocking_rerank:
        rr = rerank_scores(ti, s1, qi, tg, score, cfg)
        order, final_rank = _group_rank(group, rr, tg)
        qi, tg, src, score, passes, key_rank, rr = (qi[order], tg[order], src[order], score[order],
                                                    passes[order], key_rank[order], rr[order])
    else:
        rr, final_rank = score.astype(np.float32), key_rank
    keep = final_rank < store_k  # eval: store_k == union_k keeps the whole capped union
    rows = s1["row"].to_numpy()
    table = pa.table({
        "s1_row": pa.array(rows[qi[keep]].astype(np.int32)), "tgt": pa.array(tg[keep].astype(np.int32)),
        "src": pa.array(src[keep]), "score": pa.array(score[keep].astype(np.float32)),
        "rr": pa.array(rr[keep].astype(np.float32)), "key_rank": pa.array(key_rank[keep].astype(np.int16)),
        "rank": pa.array(final_rank[keep].astype(np.int16)), "passes": pa.array(passes[keep].astype(np.int16))},
        schema=CANDIDATE_SCHEMA)
    return table, union_sizes


def candidates_path(name: str, cfg: Config = CFG) -> Path:
    return cfg.artifacts_dir / "candidates" / f"{name}.parquet"


def queried_rows(name: str, cfg: Config = CFG) -> np.ndarray:
    """Source-1 rows that were queried for a candidate set (sorted)."""
    return np.sort(np.load(candidates_path(name, cfg).with_suffix(".rows.npy")).astype(np.int64))


def _query_to_parquet(ti: TargetIndex, split: str, name: str, s1_rows: np.ndarray | None, cfg: Config,
                      store_k: int, timings: dict) -> tuple[Path, dict]:
    s1 = pq.read_table(normalized_path(split, 1, cfg), columns=KEY_COLUMNS + ["row"])
    if s1_rows is not None:
        s1 = s1.take(pa.array(np.asarray(s1_rows, dtype=np.int64)))
    out = candidates_path(name, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    n_pairs, sizes = 0, []
    with timed(f"query {name}: {s1.num_rows} source-1 rows", timings), pq.ParquetWriter(tmp, CANDIDATE_SCHEMA) as w:
        for start in range(0, s1.num_rows, cfg.blocking_chunk_size):
            part, usz = query_chunk(ti, s1.slice(start, cfg.blocking_chunk_size), cfg, store_k)
            sizes.append(usz)
            w.write_table(part)
            n_pairs += part.num_rows
            LOG.info("  %s: queried %d/%d rows, %d candidate pairs so far", name,
                     min(start + cfg.blocking_chunk_size, s1.num_rows), s1.num_rows, n_pairs)
    tmp.replace(out)
    np.save(out.with_suffix(".rows.npy"), s1["row"].to_numpy())  # every queried entity, incl. zero-candidate ones
    usz = np.concatenate(sizes) if sizes else np.zeros(2)
    per_entity = usz.reshape(-1, 2).sum(axis=1)
    meta = {"split": split, "name": name, "n_s1": int(s1.num_rows), "n2": ti.n2, "n_targets": ti.n_targets,
            "candidate_pairs_stored": n_pairs, "store_k_per_source": store_k, "passes": list(ti.indexes),
            "top_k_per_source": cfg.blocking_top_k, "union_k_per_source": cfg.blocking_union_k,
            "rerank": cfg.blocking_rerank,
            "union_size_per_entity_before_cuts": {
                "mean": float(per_entity.mean()), "p50": float(np.quantile(per_entity, 0.5)),
                "p95": float(np.quantile(per_entity, 0.95)), "max": float(per_entity.max())},
            "max_block": cfg.blocking_max_block, "index_stats": ti.stats, "timings": timings}
    save_json(meta, out.with_suffix(".meta.json"))
    return out, meta


def generate_candidates(split: str, name: str, s1_rows: np.ndarray | None = None, cfg: Config = CFG,
                        passes: tuple[str, ...] | None = None, store_k: int | None = None) -> tuple[Path, dict]:
    """Write candidates for the given Source-1 rows (None = all) to parquet.

    store_k: per-source rank cut written to disk. Production uses ``blocking_top_k`` (the final
    candidate set). Evaluation passes ``blocking_union_k`` so recall can be measured at every K.
    """
    timings: dict = {}
    with timed("build target index", timings):
        ti = build_target_index(split, cfg, passes)
    return _query_to_parquet(ti, split, name, s1_rows, cfg, store_k or cfg.blocking_top_k, timings)


def generate_final_sets(split: str, sets: dict[str, np.ndarray | None], cfg: Config = CFG) -> dict[str, dict]:
    """Build the index once and write the FINAL candidate set (rank < blocking_top_k) for several row sets."""
    timings: dict = {}
    with timed("build target index", timings):
        ti = build_target_index(split, cfg)
    return {name: _query_to_parquet(ti, split, name, rows, cfg, cfg.blocking_top_k, timings)[1]
            for name, rows in sets.items()}