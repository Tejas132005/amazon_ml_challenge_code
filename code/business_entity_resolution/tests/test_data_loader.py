from pathlib import Path

import pytest

from src.config import CFG, SOURCE_COLUMNS
from src.data_loader import DataFormatError, count_data_lines, load_source, read_tsv
from tests.conftest import write_tsv


def cfg_for(root: Path):
    return CFG.with_overrides(data_root=root, expected_rows={})


def test_reads_strings_without_na_inference(data_root):
    t, rep = load_source("train", 1, cfg=cfg_for(data_root))
    assert rep.rows == rep.data_lines == 4
    assert t["business_name"].to_pylist()[3] == "NA"
    assert t["business_address"].to_pylist()[3] == ""
    assert t.column("entity_id").null_count == 0


def test_literal_quotes_preserved(data_root):
    t, _ = load_source("train", 2, cfg=cfg_for(data_root))
    assert t["business_name"].to_pylist()[3] == 'Joe "The Barber" Shop'


def test_column_subset(data_root):
    t, _ = load_source("train", 3, columns=["entity_id", "country"], cfg=cfg_for(data_root))
    assert t.column_names == ["entity_id", "country"]


def test_rejects_comma_separated(tmp_path):
    p = tmp_path / "x.tsv"
    p.write_text("entity_id,business_name,business_address,country\nS1-1,a,b,US\n", encoding="utf-8")
    with pytest.raises(DataFormatError, match="comma"):
        read_tsv(p, SOURCE_COLUMNS)


def test_malformed_row_raises(tmp_path):
    p = tmp_path / "x.tsv"
    write_tsv(p, list(SOURCE_COLUMNS), [("S1-1", "a", "b", "US"), ("S1-2", "a\tb", "c", "US")])
    with pytest.raises(DataFormatError, match="malformed"):
        read_tsv(p, SOURCE_COLUMNS)


def test_expected_rows_mismatch_raises(data_root):
    cfg = CFG.with_overrides(data_root=data_root, expected_rows={"train_source1": 999})
    with pytest.raises(DataFormatError, match="expected 999"):
        load_source("train", 1, cfg=cfg)


@pytest.mark.parametrize("newline,trailing", [("\n", True), ("\r\n", True), ("\n", False)])
def test_count_data_lines(tmp_path, newline, trailing):
    p = tmp_path / "x.tsv"
    body = newline.join(["h1\th2", "a\tb", "c\td"]) + (newline if trailing else "")
    p.write_bytes(body.encode())
    assert count_data_lines(p) == 2
