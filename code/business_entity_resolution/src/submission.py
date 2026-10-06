"""Write and check the two submission files (Phase 9).

matching_results.tsv   source1_entity_id <TAB> matched_entity_ids
candidate_pairs.tsv    source1_entity_id <TAB> candidate_entity_ids

* exactly one row per Source-1 entity of the split, in Source-1 file order;
* comma-separated S2-/S3- ids, no quotes, no duplicates, empty string when none;
* candidate_pairs = exactly the pairs the model scored (the final candidate parquet);
* every match is a candidate.

Lists are built with Arrow (take + ListArray offsets + binary_join), not per-row Python.
``check_submission`` re-reads the files with plain Python and re-applies every README rule.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import CFG, TARGET_SOURCES, Config
from .preprocessing import normalized_path

LOG = logging.getLogger("ber.submission")

MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")


def id_arrays(split: str, cfg: Config = CFG) -> tuple[pa.Array, pa.Array]:
    """Source-1 ids (file order) and target ids (S2 rows then S3 rows = global tgt index)."""
    s1 = pa.concat_arrays(pq.read_table(normalized_path(split, 1, cfg), columns=["entity_id"])["entity_id"].chunks)
    tg = pa.concat_arrays([c for s in TARGET_SOURCES for c in
                           pq.read_table(normalized_path(split, s, cfg), columns=["entity_id"])["entity_id"].chunks])
    return s1, tg


def write_id_lists(path: Path, header: tuple[str, str], s1_ids: pa.Array, rows: np.ndarray, tgts: np.ndarray,
                   tgt_ids: pa.Array) -> int:
    """One line per Source-1 entity; ``rows``/``tgts`` must already be in the desired list order
    within each entity (entity order itself does not matter). Returns number of ids written."""
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    rows, tgts = rows[order], np.asarray(tgts, dtype=np.int64)[order]
    offsets = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=len(s1_ids)))]).astype(np.int32)
    lists = pa.ListArray.from_arrays(pa.array(offsets), tgt_ids.take(pa.array(tgts)))
    lines = pc.binary_join_element_wise(s1_ids, pc.binary_join(lists, ","), "\t")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(header) + "\n")
        for chunk in range(0, len(lines), 500_000):
            fh.write("\n".join(lines.slice(chunk, 500_000).to_pylist()) + "\n")
    tmp.replace(path)
    return len(tgts)


def _read_lists(path: Path, header: tuple[str, str]) -> tuple[list[str], dict[str, list[str]], list[str]]:
    issues: list[str] = []
    order: list[str] = []
    data: dict[str, list[str]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        first = fh.readline().rstrip("\r\n")
        if first != "\t".join(header):
            issues.append(f"{path.name}: header {first!r} != {chr(9).join(header)!r}")
        for i, line in enumerate(fh, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                issues.append(f"{path.name}:{i}: expected 2 tab-separated fields, got {len(parts)}")
                continue
            sid, ids = parts
            if sid in data:
                issues.append(f"{path.name}:{i}: duplicate source1_entity_id {sid}")
            order.append(sid)
            data[sid] = ids.split(",") if ids else []
    return order, data, issues


def check_submission(match_path: Path, cand_path: Path, split: str = "test", cfg: Config = CFG,
                     max_report: int = 20) -> list[str]:
    """Re-apply the README rules to the written files. Returns a list of issues (empty = OK)."""
    s1_ids, tg_ids = id_arrays(split, cfg)
    s1_set = set(s1_ids.to_pylist())
    valid_t = set(tg_ids.to_pylist())
    issues: list[str] = []
    parsed = {}
    for path, header in ((match_path, MATCH_HEADER), (cand_path, CAND_HEADER)):
        order, data, iss = _read_lists(path, header)
        issues += iss
        missing, extra = s1_set - set(data), set(data) - s1_set
        if missing:
            issues.append(f"{path.name}: {len(missing)} Source-1 entities missing, e.g. {sorted(missing)[:3]}")
        if extra:
            issues.append(f"{path.name}: {len(extra)} unknown source1 ids, e.g. {sorted(extra)[:3]}")
        for sid, ids in data.items():
            if len(set(ids)) != len(ids):
                issues.append(f"{path.name}: duplicate ids in list of {sid}")
            bad = [x for x in ids if not (x.startswith(("S2-", "S3-")) and x in valid_t)]
            if bad:
                issues.append(f"{path.name}: invalid ids for {sid}: {bad[:3]}")
            if len(issues) > max_report:
                break
        parsed[path] = data
    if not issues:
        m, c = parsed[match_path], parsed[cand_path]
        not_cand = sum(1 for sid, ids in m.items() if not set(ids) <= set(c.get(sid, [])))
        if not_cand:
            issues.append(f"{not_cand} entities have matches that are not in their candidate list")
    return issues[:max_report]
