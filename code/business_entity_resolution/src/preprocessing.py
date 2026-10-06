"""Business-name and address normalisation (Phase 2).

Each source file is converted once into a normalised parquet file. Raw values
are kept unchanged next to the derived fields.

Output columns
--------------
entity_id, row, country, country_key, business_name, business_address,
name_norm      canonical tokens (legal forms unified, leet fixed, Indic transliterated)
name_core      name_norm minus legal forms / stopwords / long digit noise
name_compact   name_core without spaces (matches domains: "stylemulti.com")
name_skel      consonant skeleton per core token (cross-script + typo tolerant key)
name_legal     canonical legal-form tokens ("pvt ltd", "llc")
name_alt_core  core of a trade name after "dba"/"aka"/"t/a" ("" if none)
name_nonlatin  raw name contains non-Latin letters
addr_norm      canonical address tokens (abbreviations, city variants, PO boxes removed)
addr_core      addr_norm minus low-information tokens
addr_numbers   canonical number tokens ("115", "2/1a", "3-110/6/g-3"), no leading zeros
addr_region    state/region code when a comma component names one ("" otherwise)
addr_empty     raw address is empty

How it scales: regex/Unicode steps run as vectorised Arrow kernels. Token-level
rules (transliteration, leet fixes, canonical forms) run in Python only on each
chunk's *unique* tokens (cached), then are broadcast back with ``take``. Output
lists are rebuilt from offsets with NumPy. No per-row Python loops.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import lexicons as lx
from .config import CFG, SOURCE_PREFIXES, Config
from .data_loader import load_source
from .utils import timed

LOG = logging.getLogger("ber.preprocessing")

_WORD_SPLIT = r"[^\p{L}\p{N}\p{M}]+"
_INDIC_RE = re.compile(r"[\u0900-\u0DFF]")
_ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th)$")

# ============================================================================ transliteration
_VOWEL_NAMES = {
    "A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u", "E": "e", "EE": "e", "AI": "ai",
    "O": "o", "OO": "o", "AU": "au", "VOCALIC R": "ri", "VOCALIC RR": "ri", "VOCALIC L": "li",
    "VOCALIC LL": "li", "CANDRA E": "e", "CANDRA O": "o", "SHORT E": "e", "SHORT O": "o", "CANDRA A": "a",
}


@lru_cache(maxsize=None)
def _indic_char(ch: str) -> tuple[str, str]:
    """Classify one Brahmic character from its Unicode name (script-agnostic).

    Kinds: C consonant (inherent 'a'), D dead consonant, V vowel letter, S vowel sign,
    R virama, N nasal sign, T other text, X ignore.
    """
    name = unicodedata.name(ch, "")
    script, _, rest = name.partition(" ")
    if rest.startswith("LETTER CHILLU "):
        return "D", rest[14:].lower()
    if rest.startswith("LETTER "):
        x = rest[7:]
        if x in _VOWEL_NAMES:
            return "V", _VOWEL_NAMES[x]
        cons = (x[:-1] if x.endswith("A") and len(x) > 1 else x).lower().replace(" ", "")
        if script == "TAMIL" and x == "CA":
            cons = "s"  # Tamil ca is usually pronounced s ("business" -> pisinas)
        return "C", cons
    if rest.startswith("VOWEL SIGN "):
        x = rest[11:]
        return "S", _VOWEL_NAMES.get(x, x.split()[0].lower())
    if "VIRAMA" in rest:
        return "R", ""
    if rest in ("SIGN ANUSVARA", "SIGN CANDRABINDU"):
        return "N", "n"
    if rest.startswith("DIGIT "):
        return "T", str(unicodedata.digit(ch, ""))
    if not rest or rest.startswith(("SIGN ", "AU LENGTH", "AI LENGTH", "LENGTH MARK")):
        return "X", ""
    return ("T", ch) if ord(ch) < 0x0900 or ord(ch) > 0x0DFF else ("X", "")


@lru_cache(maxsize=2**20)
def transliterate_indic(token: str) -> str:
    """Rough phonetic romanisation of Brahmic-script text (Devanagari, Bengali, Tamil, ...).

    Implements inherent-vowel insertion, virama suppression and word-final
    schwa deletion. Output is lowercase ASCII-ish; exactness is not needed
    because matching uses ``skeleton``.
    """
    out: list[str] = []
    pending = False  # inherent 'a' after a consonant
    nasal_last = False
    for ch in token:
        kind, txt = _indic_char(ch)
        if kind == "C":
            if pending:
                out.append("a")
            out.append(txt)
            pending, nasal_last = True, False
        elif kind == "S":
            out.append(txt)
            pending, nasal_last = False, False
        elif kind == "R":
            pending = False
        elif kind == "N":
            if pending:
                out.append("a")
            out.append("n")
            pending, nasal_last = False, True
        elif kind in ("V", "D", "T"):
            if pending:
                out.append("a")
            out.append(txt)
            pending, nasal_last = False, False
    if nasal_last and out:
        out.pop()  # word-final nasalisation ("साईं" -> sai)
    return "".join(out)


# ============================================================================ skeleton key
_SKEL = {**{c: "k" for c in "cgjqk"}, **{c: "t" for c in "dt"}, **{c: "p" for c in "bpfvw"},
         **{c: "s" for c in "sz"}, **{c: "" for c in "aeiouhy"}}


@lru_cache(maxsize=2**20)
def skeleton(token: str) -> str:
    """Consonant skeleton: first letter + voicing-collapsed consonants, repeats merged.

    'private' -> 'prpt', 'प्राइवेट' (praivett) -> 'prpt'; 'consultants' -> 'knsltnts'.
    Pure digits are returned unchanged.
    """
    if not token or token.isdigit():
        return token
    t = token.replace("x", "ks")
    first = "a" if t[0] in "aeiouy" else _SKEL.get(t[0], t[0]) or t[0]
    chars = [first] + [_SKEL.get(c, c) for c in t[1:]]
    out: list[str] = []
    for c in chars:
        if c and (not out or out[-1] != c):
            out.append(c)
    return "".join(out)


_LEGAL_SKELETONS: dict[str, str] = {}
for _w in lx.LEGAL_WORDS_FOR_SKELETON:
    _LEGAL_SKELETONS.setdefault(skeleton(_w), lx.LEGAL_CANON[_w])

# ============================================================================ token rules
DROP, CORE, LEGAL, STOP, NOISE = 0, 1, 2, 3, 4


def fix_leet(tok: str) -> str:
    """Map digits to letters inside mostly-alphabetic tokens ('5haan'->'shaan'); keep ordinals/codes."""
    n_alpha = sum(c.isalpha() for c in tok)
    n_digit = sum(c.isdigit() for c in tok)
    if n_digit == 0 or n_alpha < 2 or n_alpha < n_digit or _ORDINAL_RE.match(tok):
        return tok
    return "".join(lx.LEET.get(c, c) for c in tok)


@lru_cache(maxsize=2**21)
def name_token_info(tok: str) -> tuple[str, int, str]:
    """(canonical text, kind, skeleton) for one name token."""
    if not tok:
        return "", DROP, ""
    translit = bool(_INDIC_RE.search(tok))
    if translit:
        tok = transliterate_indic(tok)
        if not tok:
            return "", DROP, ""
        if tok in lx.TRANSLIT_LEGAL:
            return lx.TRANSLIT_LEGAL[tok], LEGAL, ""
        sk = skeleton(tok)
        if sk in _LEGAL_SKELETONS:
            return _LEGAL_SKELETONS[sk], LEGAL, ""
    else:
        tok = fix_leet(tok)
    if tok in lx.LEGAL_CANON:
        return lx.LEGAL_CANON[tok], LEGAL, ""
    if tok in lx.NAME_STOPWORDS:
        return tok, STOP, ""
    if tok.isdigit() and len(tok) >= 5:
        return tok, NOISE, ""
    return tok, CORE, skeleton(tok)


@lru_cache(maxsize=2**21)
def addr_token_info(tok: str) -> tuple[str, int, str]:
    if not tok:
        return "", DROP, ""
    if _INDIC_RE.search(tok):
        tok = transliterate_indic(tok)
    elif tok.isdigit():
        tok = tok.lstrip("0") or "0"
    tok = lx.ADDR_CANON.get(tok, tok)
    return tok, (STOP if tok in lx.ADDR_STOPWORDS else CORE), ""


_NUM_PREFIX = re.compile(r"^(?:h\.?\s?no\.?|hno|no\.?|plot|flat|#)+")
_LEAD_ZERO = re.compile(r"(?<![0-9])0+(?=[0-9])")


@lru_cache(maxsize=2**21)
def canon_number(tok: str) -> str:
    """Canonical house/plot number token, or '' if the token carries no identifier digits."""
    if not any(c.isdigit() for c in tok):
        return ""
    t = _NUM_PREFIX.sub("", tok.replace("#", "")).strip(".-/:")
    t = _LEAD_ZERO.sub("", t)
    if not t or _ORDINAL_RE.match(t):
        return ""
    return t


# ============================================================================ Arrow helpers
def fold(arr: pa.Array) -> pa.Array:
    """NFKC, lowercase, drop zero-width chars, strip Latin accents (Indic marks untouched)."""
    s = pc.utf8_normalize(arr, "NFKC")
    s = pc.utf8_lower(s)
    s = pc.replace_substring_regex(s, r"[\x{200b}-\x{200d}\x{feff}]", "")
    s = pc.utf8_normalize(s, "NFKD")
    s = pc.replace_substring_regex(s, r"[\x{0300}-\x{036f}]", "")
    return pc.utf8_normalize(s, "NFC")


def _tokens(arr: pa.Array) -> pa.ListArray:
    s = pc.utf8_trim_whitespace(pc.replace_substring_regex(arr, _WORD_SPLIT, " "))
    return pc.split_pattern(s, " ")


class _Tokenised:
    """Flat token codes of a ListArray mapped through a cached per-token rule."""

    def __init__(self, lists: pa.ListArray, rule):
        self.n = len(lists)
        self.parent = pc.list_parent_indices(lists).to_numpy()
        enc = pc.dictionary_encode(pc.list_flatten(lists))
        raw_codes = enc.indices.to_numpy()
        info = [rule(t) for t in enc.dictionary.to_pylist()]
        self.kind = np.array([i[1] for i in info], dtype=np.int8)[raw_codes]
        self.canon_codes, self.canon_vocab = self._recode([i[0] for i in info], raw_codes)
        self.skel_codes, self.skel_vocab = self._recode([i[2] for i in info], raw_codes)

    @staticmethod
    def _recode(values: list[str], raw_codes: np.ndarray) -> tuple[np.ndarray, pa.Array]:
        index: dict[str, int] = {}
        remap = np.fromiter((index.setdefault(v, len(index)) for v in values), dtype=np.int64, count=len(values))
        return remap[raw_codes], pa.array(list(index), pa.string())

    def join(self, keep: np.ndarray, codes: np.ndarray, vocab: pa.Array, vocab_empty: bool = False) -> pa.Array:
        """Join kept tokens per row with spaces, dropping consecutive duplicates."""
        if vocab_empty:  # also drop tokens whose mapped text is ""
            empty = np.array([v == "" for v in vocab.to_pylist()], dtype=bool)
            keep = keep & ~empty[codes]
        c, p = codes[keep], self.parent[keep]
        if len(c):
            dup = np.concatenate([[False], (c[1:] == c[:-1]) & (p[1:] == p[:-1])])
            c, p = c[~dup], p[~dup]
        offsets = np.concatenate([[0], np.cumsum(np.bincount(p, minlength=self.n))]).astype(np.int32)
        lists = pa.ListArray.from_arrays(pa.array(offsets), vocab.take(pa.array(c, pa.int64())))
        return pc.binary_join(lists, " ")


def _name_parts(folded: pa.Array) -> tuple[pa.Array, pa.Array]:
    s = pc.replace_substring_regex(folded, r"(?:https?://)?www\.\S+", " ")
    s = pc.replace_substring_regex(s, lx.TLD_PATTERN, "")
    s = pc.replace_substring(s, "&", " and ")
    s = pc.replace_substring_regex(s, r"['’`.]", "")
    has_dba = pc.match_substring_regex(s, lx.DBA_PATTERN)
    alt = pc.if_else(has_dba, pc.replace_substring_regex(s, r"^.*?" + lx.DBA_PATTERN, "", max_replacements=1), "")
    main = pc.replace_substring_regex(s, lx.DBA_PATTERN, " ")
    return main, alt


def normalize_names(names: pa.Array) -> dict[str, pa.Array]:
    """Vectorised name normalisation for one Arrow chunk (see module docstring)."""
    main, alt = _name_parts(fold(names))
    tk = _Tokenised(_tokens(main), name_token_info)
    not_drop = tk.kind != DROP
    core_mask = tk.kind == CORE
    norm = tk.join(not_drop, tk.canon_codes, tk.canon_vocab)
    core = tk.join(core_mask, tk.canon_codes, tk.canon_vocab)
    core = pc.if_else(pc.equal(core, ""), norm, core)  # e.g. name made only of legal words
    tka = _Tokenised(_tokens(alt), name_token_info)
    return {
        "name_norm": norm,
        "name_core": core,
        "name_compact": pc.replace_substring(core, " ", ""),
        "name_skel": tk.join(core_mask, tk.skel_codes, tk.skel_vocab, vocab_empty=True),
        "name_legal": tk.join(tk.kind == LEGAL, tk.canon_codes, tk.canon_vocab),
        "name_alt_core": tka.join(tka.kind == CORE, tka.canon_codes, tka.canon_vocab),
        "name_nonlatin": pc.match_substring_regex(names, r"[^\p{Latin}\P{L}]"),
    }


# ---- regions
def _build_region_index() -> tuple[dict[str, int], list[str]]:
    codes: list[str] = []
    index: dict[str, int] = {}

    def add(key: str, code: str) -> None:
        if code not in codes:
            codes.append(code)
        index.setdefault(key, codes.index(code))

    for country, regions in lx.REGIONS_BY_COUNTRY.items():
        skel_owner: dict[str, set[str]] = {}
        for name, code in regions.items():
            add(f"{country}|{name}", code)
            add(f"{country}|{code}", code)
            skel_owner.setdefault(skeleton(name.replace(" ", "")), set()).add(code)
        for sk, owners in skel_owner.items():  # skeleton keys for transliterated text, unambiguous only
            if len(owners) == 1 and len(sk) >= 2:
                add(f"{country}|~{sk}", next(iter(owners)))
    return index, codes


_REGION_INDEX, _REGION_CODES = _build_region_index()
_REGION_KEYS = pa.array(list(_REGION_INDEX), pa.string())
_REGION_KEY_CODE = np.array(list(_REGION_INDEX.values()), dtype=np.int64)
_REGION_CODE_ARR = np.array(_REGION_CODES + [""], dtype=object)


def extract_region(folded: pa.Array, country_key: pa.Array) -> pa.Array:
    """Region code from the last comma component that names a known region of the record's country."""
    comps = pc.split_pattern(folded, ",")
    n = len(comps)
    parent = pc.list_parent_indices(comps).to_numpy()
    flat = pc.utf8_trim_whitespace(pc.replace_substring_regex(pc.list_flatten(comps), _WORD_SPLIT, " "))
    ctry = country_key.take(pa.array(parent))
    key = pc.binary_join_element_wise(ctry, flat, "|")
    pos = pc.fill_null(pc.index_in(key, value_set=_REGION_KEYS), -1).to_numpy()
    code = np.where(pos >= 0, _REGION_KEY_CODE[np.maximum(pos, 0)], -1)

    indic = np.flatnonzero(pc.match_substring_regex(flat, r"[\x{0900}-\x{0DFF}]").to_numpy(zero_copy_only=False))
    if len(indic):
        comp_vals = flat.take(pa.array(indic)).to_pylist()
        ctry_vals = ctry.take(pa.array(indic)).to_pylist()
        for i, comp, c in zip(indic, comp_vals, ctry_vals):
            sk = skeleton(transliterate_indic(comp.replace(" ", "")))
            code[i] = _REGION_INDEX.get(f"{c}|~{sk}", -1)  # dict value is already the code index

    out = np.full(n, len(_REGION_CODES), dtype=np.int64)  # index of ""
    m = code >= 0
    rec, val = parent[m], code[m]
    if len(rec):
        last = np.concatenate([rec[1:] != rec[:-1], [True]])
        out[rec[last]] = val[last]
    return pa.array(_REGION_CODE_ARR[out], pa.string())


def normalize_addresses(addresses: pa.Array, country_key: pa.Array) -> dict[str, pa.Array]:
    folded = fold(addresses)
    folded = pc.replace_substring_regex(folded, lx.PO_BOX_PATTERN, " ")
    region = extract_region(folded, country_key)

    numbers_lists = pc.split_pattern_regex(pc.utf8_trim_whitespace(folded), r"[\s,;:()\[\]]+")
    tn = _Tokenised(numbers_lists, lambda t: (canon_number(t), CORE, ""))
    numbers = tn.join(np.ones(len(tn.kind), bool), tn.canon_codes, tn.canon_vocab, vocab_empty=True)

    words = pc.replace_substring_regex(folded, r"['’`]", "")
    tw = _Tokenised(_tokens(words), addr_token_info)
    return {
        "addr_norm": tw.join(tw.kind != DROP, tw.canon_codes, tw.canon_vocab, vocab_empty=True),
        "addr_core": tw.join(tw.kind == CORE, tw.canon_codes, tw.canon_vocab, vocab_empty=True),
        "addr_numbers": numbers,
        "addr_region": region,
        "addr_empty": pc.equal(pc.utf8_trim_whitespace(addresses), ""),
    }


def normalize_table(t: pa.Table, row_offset: int = 0) -> pa.Table:
    """Normalise a source table (all four columns) chunk by chunk."""
    out = []
    for batch in t.to_batches():
        b = pa.Table.from_batches([batch])
        name = b["business_name"].combine_chunks()
        addr = b["business_address"].combine_chunks()
        country = b["country"].combine_chunks()
        country_key = pc.utf8_lower(pc.utf8_trim_whitespace(country))
        cols = {
            "entity_id": b["entity_id"].combine_chunks(),
            "row": pa.array(np.arange(row_offset, row_offset + b.num_rows, dtype=np.int32)),
            "country": country, "country_key": country_key,
            "business_name": name, "business_address": addr,
            **normalize_names(name), **normalize_addresses(addr, country_key),
        }
        out.append(pa.table(cols))
        row_offset += b.num_rows
    return pa.concat_tables(out)


def normalized_path(split: str, source_no: int, cfg: Config = CFG):
    return cfg.artifacts_dir / "normalized" / f"{split}_source{source_no}.parquet"


def run_preprocess(split: str, cfg: Config = CFG) -> None:
    """Normalise all three sources of a split into artifacts/normalized/*.parquet."""
    LOG.info("artifacts dir: %s", cfg.artifacts_dir)
    if "onedrive" in str(cfg.artifacts_dir).lower():
        LOG.warning("artifacts dir is inside OneDrive; sync can hide/remove large files. "
                    "Set BER_ARTIFACTS_DIR to a local folder, e.g. C:\\ber_artifacts")
    timings: dict = {}
    for s in SOURCE_PREFIXES:
        with timed(f"normalize {split}_source{s}", timings):
            t, _ = load_source(split, s, cfg=cfg)
            nt = normalize_table(t)
            del t
            path = normalized_path(split, s, cfg)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".parquet.tmp")
            pq.write_table(nt, tmp, compression="zstd")
            tmp.replace(path)  # atomic: never leaves a half-written parquet behind
            _write_examples(nt, path.with_name(f"{split}_source{s}_examples.tsv"), cfg)
            LOG.info("wrote %s (%d rows, %.0f MB on disk)", path, nt.num_rows, path.stat().st_size / 2**20)
            del nt


def _write_examples(nt: pa.Table, path, cfg: Config, per_group: int = 25) -> None:
    """Small eyeball sample: random rows per country label plus non-Latin names."""
    rng = np.random.default_rng(cfg.random_seed)
    country = nt["country"].to_numpy(zero_copy_only=False)
    nonlatin = nt["name_nonlatin"].to_numpy(zero_copy_only=False)
    picks = []
    for c in np.unique(country):
        idx = np.flatnonzero(country == c)
        picks.append(rng.choice(idx, size=min(per_group, len(idx)), replace=False))
    idx = np.flatnonzero(nonlatin)
    if len(idx):
        picks.append(rng.choice(idx, size=min(per_group, len(idx)), replace=False))
    sample = nt.take(pa.array(np.concatenate(picks))).drop_columns(["row", "country_key"])
    sample.to_pandas().to_csv(path, sep="\t", index=False)
