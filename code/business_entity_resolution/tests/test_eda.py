import json

from src.config import CFG
from src.eda import basic_norm, run_eda

import pyarrow as pa


def test_basic_norm_keeps_indic_and_folds_accents():
    out = basic_norm(pa.array(["LLC Moncada Léarning", "राम  मार्केटिंग", "B+ Retail, Inc."])).to_pylist()
    assert out == ["llc moncada learning", "राम मार्केटिंग", "b retail inc"]


def test_run_eda_train(data_root, tmp_path):
    cfg = CFG.with_overrides(data_root=data_root, artifacts_dir=tmp_path / "artifacts", expected_rows={},
                             eda_hub_token_count=10, eda_token_keep_min=1)
    rep = run_eda("train", cfg)
    g = rep["ground_truth"]
    assert g["match_counts"]["singletons"] == 2          # S1-2 (empty) and S1-4 (only invalid ref)
    assert g["match_counts"]["max_matches"] == 3
    assert g["list_hygiene"]["duplicate_ids_within_a_list"] == 1
    assert g["list_hygiene"]["s2_ids_not_in_source2"] == 1
    assert g["target_coverage"]["s3"]["unreferenced_records"] == 1
    assert g["exclusivity"]["s2"]["ids_matched_to_multiple_s1"] == 0
    assert rep["pairs"]["sampled_pairs"] == 4
    assert rep["sources"]["train_source2"]["scripts"]["name"]["Devanagari"] == 1
    out = cfg.eda_dir("train")
    assert (out / "summary.md").exists()
    json.loads((out / "report.json").read_text(encoding="utf-8"))
