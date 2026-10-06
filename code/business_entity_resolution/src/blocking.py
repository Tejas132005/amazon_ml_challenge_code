"""Blocking keys and the inverted block index (Phase 3, building blocks).

Every blocking key is an int64 that encodes (value, country, pass):

    key = value * KEY_MULT + country_code * N_PASS_SLOTS + pass_id

so keys never collide across passes or countries, and matching only happens
within one country label (EDA: 100% of true pairs share the label; unseen
labels such as France simply get their own code).

Passes (all on normalised fields):
  name_tok        single name_core / name_alt_core tokens
  name_pair       unordered pairs of name_core tokens (rare combinations of common words)
  skel_set        the whole sorted set of name skeleton tokens (cross-script exact)
  skel_pair       unordered pairs of skeleton tokens (cross-script partial)
  compact         name_compact exact ("stylemulti" == "stylemulti.com")
  compact_prefix  first N chars of name_compact
  addr_tok        single addr_core tokens
  num_anchor      address digit run x common address token ("9" x "chennai")
  name_num        name_core token x address digit run  ("oncology" x "711")
  skel_num        name skeleton token x address digit run (cross-script + number)
  skel_anchor     name skeleton token x common address token ("klpl" x "chennai")
  num_street      address digit run x uncommon street word ("18" x "ajaccio"); added after test
                  showed 4,158 French entities with no candidates: generic names ("Nantes
                  Foyer SAS") and big-city number x city keys are hubs there

The last three exist because names in this data are built from a limited
vocabulary: whole names like "oncology medicine" recur hundreds of times, so
name-only keys are hubs. Combined with an address component they become rare.

Hub safeguard: a key shared by more than ``max_block`` target records is not
used. Records in such blocks are not dropped: they remain reachable through the
combination passes (name_pair, skel_pair, num_anchor), which exist precisely
because their parts are too common on their own.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

LOG = logging.getLogger("ber.blocking")

PASSES: tuple[str, ...] = ("name_tok", "name_pair", "skel_set", "skel_pair", "compact",
                           "compact_prefix", "addr_tok", "num_anchor", "name_num", "skel_num", "skel_anchor",
                           "num_street")
PASS_ID = {p: i for i, p in enumerate(PASSES)}
N_PASS_SLOTS = 16
N_COUNTRY_SLOTS = 64
KEY_MULT = N_PASS_SLOTS * N_COUNTRY_SLOTS


# ============================================================================ tokens
@dataclass
class Tokens:
    """Flattened tokens: ``parent`` (record index, non-decreasing) and ``code`` (vocab id)."""
    parent: np.ndarray
    code: np.ndarray

    def __len__(self) -> int:
        return len(self.code)

    def filter(self, mask: np.ndarray) -> "Tokens":
        return Tokens(self.parent[mask], self.code[mask])


def tokenize(col: pa.ChunkedArray | pa.Array, vocab: pa.Array | None = None,
             whole: bool = False) -> tuple[Tokens, pa.Array]:
    """Split strings on spaces (or take whole strings) and encode against ``vocab``.

    vocab=None builds the vocabulary (target side); otherwise unknown tokens are
    dropped (query side: a token no target has cannot produce a candidate).
    """
    chunks = col.chunks if isinstance(col, pa.ChunkedArray) else [col]
    parents, values, off = [], [], 0
    for ch in chunks:
        if whole:
            p, t = np.arange(len(ch), dtype=np.int32), ch
        else:
            lists = pc.split_pattern(ch, " ")
            p, t = pc.list_parent_indices(lists).to_numpy().astype(np.int32), pc.list_flatten(lists)
        keep = pc.not_equal(t, "").to_numpy(zero_copy_only=False)
        parents.append(p[keep] + off)
        values.append(t.filter(pa.array(keep)))
        off += len(ch)
    parent = np.concatenate(parents) if parents else np.zeros(0, np.int32)
    flat = pa.concat_arrays(values) if values else pa.array([], pa.string())
    if vocab is None:
        enc = pc.dictionary_encode(flat)
        return Tokens(parent, enc.indices.to_numpy().astype(np.int32)), enc.dictionary
    idx = pc.index_in(flat, value_set=vocab)
    found = pc.is_valid(idx).to_numpy(zero_copy_only=False)
    code = pc.fill_null(idx, 0).to_numpy().astype(np.int32)
    return Tokens(parent[found], code[found]), vocab


def dedup(tok: Tokens, chunk: int = 8_000_000) -> Tokens:
    """Drop repeated codes within a record, keeping first occurrence order.

    Records are contiguous (parent is non-decreasing), so large inputs are processed in
    record-aligned chunks: peak extra memory ~4 arrays of ``chunk`` int64 instead of the
    whole input (70M+ address tokens would need ~2 GB at once).
    """
    n = len(tok)
    if n == 0:
        return tok
    vmax = int(tok.code.max()) + 1
    keep_parts = []
    lo = 0
    while lo < n:
        hi = min(lo + chunk, n)
        if hi < n:  # extend to the end of the current record
            hi = int(np.searchsorted(tok.parent, tok.parent[hi - 1], side="right"))
        comb = tok.parent[lo:hi].astype(np.int64) * vmax + tok.code[lo:hi]
        _, first = np.unique(comb, return_index=True)
        del comb
        first.sort()
        keep_parts.append(first + lo)
        lo = hi
    keep = np.concatenate(keep_parts)
    return Tokens(tok.parent[keep], tok.code[keep])


def rank_within(parent: np.ndarray) -> np.ndarray:
    """0-based position of each element within its (contiguous) parent group."""
    if len(parent) == 0:
        return np.zeros(0, np.int64)
    new = np.concatenate([[True], parent[1:] != parent[:-1]])
    starts = np.flatnonzero(new)
    group = np.cumsum(new) - 1
    return np.arange(len(parent)) - starts[group]


def head(tok: Tokens, k: int) -> Tokens:
    return tok.filter(rank_within(tok.parent) < k)


def cross(a: Tokens, b: Tokens, n_records: int, b_vocab: int, upper_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """All (a_code, b_code) combinations inside each record -> (record, a*|Vb| + b).

    upper_only: ``a`` and ``b`` are the same token list; keep each unordered pair once.
    Cost: sum over records of |a_r| * |b_r| (bounded by the head() limits).
    """
    cnt_b = np.bincount(b.parent, minlength=n_records)
    start_b = np.concatenate([[0], np.cumsum(cnt_b)[:-1]])
    rep = cnt_b[a.parent]
    total = int(rep.sum())
    if total == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    a_idx = np.repeat(np.arange(len(a), dtype=np.int64), rep)
    b_idx = np.arange(total, dtype=np.int64)
    b_idx -= np.repeat(np.cumsum(rep) - rep, rep)
    b_idx += start_b[a.parent[a_idx]]
    if upper_only:
        keep = b_idx > a_idx  # same list: positions in b == positions in a
        a_idx, b_idx = a_idx[keep], b_idx[keep]
    rec = a.parent[a_idx]
    ca, cb = a.code[a_idx].astype(np.int64), b.code[b_idx].astype(np.int64)
    del a_idx, b_idx
    if upper_only:
        ca, cb = np.minimum(ca, cb), np.maximum(ca, cb)
    ca *= b_vocab
    ca += cb
    return rec, ca


def compose(value: np.ndarray, country: np.ndarray, pass_name: str) -> np.ndarray:
    """Pack (value, country, pass) into one int64; -1 where the country is unknown to the index."""
    key = value.astype(np.int64) * KEY_MULT
    key += country.astype(np.int64) * N_PASS_SLOTS + PASS_ID[pass_name]
    key[country < 0] = -1
    return key


# ============================================================================ index
@dataclass
class BlockIndex:
    """Sorted unique keys -> contiguous slices of target record ids."""
    keys: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    recs: np.ndarray
    weight: np.ndarray
    stats: dict = field(default_factory=dict)

    @classmethod
    def build(cls, rec: np.ndarray, key: np.ndarray, max_block: int, n_targets: int) -> "BlockIndex":
        """O(m log m) sort of m key instances; keeps only blocks with <= max_block records."""
        m = len(key)
        order = np.argsort(key, kind="stable")
        key = key[order]
        rec = rec.astype(np.int32, copy=False)[order]
        del order
        if m:
            new = np.concatenate([[True], key[1:] != key[:-1]])
            starts = np.flatnonzero(new)
            counts = np.diff(np.concatenate([starts, [m]]))
        else:
            starts = counts = np.zeros(0, np.int64)
        keep = counts <= max_block
        kept_inst = np.repeat(keep, counts)
        stats = {"key_instances": int(m), "unique_keys": int(len(starts)),
                 "capped_keys": int((~keep).sum()),
                 "instances_in_capped_keys": int(m - kept_inst.sum()),
                 "largest_capped_block": int(counts[~keep].max()) if (~keep).any() else 0}
        keys_k, counts_k = key[starts[keep]], counts[keep].astype(np.int32)
        recs_k = rec[kept_inst].astype(np.int32)
        starts_k = np.concatenate([[0], np.cumsum(counts_k)[:-1]]).astype(np.int64)
        # Steep rarity weight: a unique key ~ log(1+max_block), a key at the cap ~ log(2).
        # (log(N/df) was nearly flat inside the cap and let many weak keys outrank one rare key.)
        weight = np.log1p(max_block / np.maximum(counts_k, 1)).astype(np.float32)
        stats["kept_instances"] = int(len(recs_k))
        return cls(keys_k, starts_k, counts_k, recs_k, weight, stats)

    def expand(self, q_rec: np.ndarray, q_key: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Join query keys to blocks -> (query record, target record, key weight)."""
        if len(self.keys) == 0 or len(q_key) == 0:
            z = np.zeros(0, np.int64)
            return z, z, np.zeros(0, np.float32)
        pos = np.searchsorted(self.keys, q_key)
        pos_c = np.minimum(pos, len(self.keys) - 1)
        found = self.keys[pos_c] == q_key
        pos, q_rec = pos_c[found], q_rec[found]
        rep = self.counts[pos].astype(np.int64)
        total = int(rep.sum())
        within = np.arange(total) - np.repeat(np.cumsum(rep) - rep, rep)
        tgt = self.recs[np.repeat(self.starts[pos], rep) + within]
        return np.repeat(q_rec, rep), tgt.astype(np.int64), np.repeat(self.weight[pos], rep)

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.keys, self.starts, self.counts, self.recs, self.weight))