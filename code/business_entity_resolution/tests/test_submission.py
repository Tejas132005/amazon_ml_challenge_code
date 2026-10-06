import numpy as np
from dataclasses import replace

from src.config import CFG
from src.preprocessing import run_preprocess
from src.submission import CAND_HEADER, MATCH_HEADER, check_submission, id_arrays, write_id_lists


def test_write_and_check(data_root, tmp_path):
    cfg = replace(CFG.with_overrides(data_root=data_root, artifacts_dir=tmp_path / "a", expected_rows={}))
    run_preprocess("train", cfg)
    s1, tg = id_arrays("train", cfg)          # tg order: S2-10, S2-11, S2-12, S2-13, S3-20, S3-21
    m, c = tmp_path / "m.tsv", tmp_path / "c.tsv"
    write_id_lists(c, CAND_HEADER, s1, np.array([0, 0, 0, 2]), np.array([0, 1, 4, 2]), tg)
    write_id_lists(m, MATCH_HEADER, s1, np.array([0, 0]), np.array([0, 4]), tg)
    assert check_submission(m, c, "train", cfg) == []
    lines = m.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "source1_entity_id\tmatched_entity_ids" and lines[1] == "S1-1\tS2-10,S3-20"
    assert lines[2] == "S1-2\t" and len(lines) == 5 and '"' not in m.read_text(encoding="utf-8")

    write_id_lists(m, MATCH_HEADER, s1, np.array([1]), np.array([3]), tg)   # match not in candidates
    assert any("not in their candidate list" in i for i in check_submission(m, c, "train", cfg))
    m.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-10,S2-10\n", encoding="utf-8")
    issues = check_submission(m, c, "train", cfg)
    assert any("missing" in i for i in issues) and any("duplicate ids" in i for i in issues)
