"""Tiny synthetic dataset mirroring the real schema (incl. edge cases)."""
from __future__ import annotations

from pathlib import Path

import pytest

S1 = [("S1-1", "Prime Money LLC", "17560 Ellis Road, Tahlequah, OK", "US"),
      ("S1-2", "Helios", "66 Edgewood Street, Bridgeport, CT", "US"),
      ("S1-3", "Ram Marketing Private Limited", "KH NO 570/13, New Delhi, Delhi 110018", "India"),
      ("S1-4", "NA", "", "India")]  # 'NA' must stay a string, empty address stays ""
S2 = [("S2-10", "PRIME MONEY", "17560 ELLIS RD, TAHLEQUAH, OK", "US"),
      ("S2-11", "Prime Money L.L.C.", "17560 Ellis Rd, Tahlequah, OK", "US"),
      ("S2-12", "राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, Delhi", "India"),
      ("S2-13", 'Joe "The Barber" Shop', "1 Main St, Austin, TX", "US")]  # literal quotes
S3 = [("S3-20", "Prime Money", "Ellis Road, Tahlequah, Oklahoma", "US"),
      ("S3-21", "Unrelated Bakery", "9 Oak Ave, Reno, NV", "US")]
GT = [("S1-1", "S2-10,S2-11,S3-20"), ("S1-2", ""), ("S1-3", "S2-12,S2-12"), ("S1-4", "S2-99")]


def write_tsv(path: Path, header: list[str], rows: list[tuple], newline: str = "\r\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    path.write_bytes((newline.join(lines) + newline).encode("utf-8"))


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    src_cols = ["entity_id", "business_name", "business_address", "country"]
    root = tmp_path / "dataset"
    write_tsv(root / "train" / "train_source1.tsv", src_cols, S1)
    write_tsv(root / "train" / "train_source2.tsv", src_cols, S2)
    write_tsv(root / "train" / "train_source3.tsv", src_cols, S3)
    write_tsv(root / "train" / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], GT)
    return root
