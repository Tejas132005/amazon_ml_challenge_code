import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import replace

from src.blocking import BlockIndex, Tokens, compose, cross, dedup, head, tokenize
from src.config import CFG
from src.evaluation import candidate_report, f05


def test_tokenize_query_drops_unknown():
    tok, vocab = tokenize(pa.array(["a b", "b c", ""]))
    q, _ = tokenize(pa.array(["b z", "c"]), vocab)
    assert list(q.parent) == [0, 1] and [vocab[i].as_py() for i in q.code] == ["b", "c"]


def test_dedup_head_cross_pairs():
    t = Tokens(np.array([0, 0, 0, 1, 1]), np.array([5, 5, 7, 1, 2]))
    d = dedup(t)
    assert list(d.code) == [5, 7, 1, 2]
    assert list(head(d, 1).code) == [5, 1]
    rec, val = cross(d, d, 2, 10, upper_only=True)
    assert sorted(zip(rec, val)) == [(0, 57), (1, 12)]


def test_cross_numbers_x_anchors():
    nums = Tokens(np.array([0, 0, 1]), np.array([1, 2, 3]))
    anch = Tokens(np.array([0, 2]), np.array([9, 8]))
    rec, val = cross(nums, anch, 3, 10)
    assert sorted(zip(rec, val)) == [(0, 19), (0, 29)]


def test_compose_unknown_country():
    k = compose(np.array([3, 3]), np.array([0, -1]), "name_tok")
    assert k[0] >= 0 and k[1] == -1


def test_block_index_caps_hubs_and_expands():
    rec = np.array([0, 1, 2, 3, 4])
    key = np.array([10, 10, 10, 20, 30])
    idx = BlockIndex.build(rec, key, max_block=2, n_targets=5)
    assert idx.stats["capped_keys"] == 1 and list(idx.keys) == [20, 30]
    qi, tg, w = idx.expand(np.array([0, 1, 2]), np.array([10, 20, 99]))
    assert list(qi) == [1] and list(tg) == [3] and w[0] > 0


def test_candidate_report_oracle():
    cands = pa.table({"s1_row": pa.array([0, 0, 1], pa.int32()), "tgt": pa.array([5, 6, 7], pa.int32()),
                      "src": pa.array([2, 2, 3], pa.int8()), "score": pa.array([1.0, 0.5, 1.0], pa.float32()),
                      "rank": pa.array([0, 1, 0], pa.int16()), "passes": pa.array([1, 1, 2], pa.int16())})
    # rank 1 candidate of entity 0 is excluded at K=1 but present in the 'stored' ceiling row
    truth = pa.table({"s1_row": pa.array([0, 0, 1], pa.int32()), "tgt": pa.array([5, 8, 9], pa.int32()),
                      "src": pa.array([2, 2, 3], pa.int8())})
    curve, extra = candidate_report(cands, np.array([0, 1, 2]), truth, 10, (1, 5))
    row = curve.set_index("k_per_source").loc[5]
    assert abs(row.pair_recall - 1 / 3) < 1e-9
    expected = (f05(np.array([1.0]), np.array([0.5]))[0] + 0.0 + 1.0) / 3  # entity 2 is a singleton
    assert abs(row.oracle_macro_f05 - expected) < 1e-9
    assert row.zero_cand_share == 1 / 3 and row.max_cands == 2
    assert curve.iloc[-1].k_per_source == "stored" and curve.iloc[-1].pair_recall == row.pair_recall


def test_end_to_end_blocking(data_root, tmp_path):
    from src.evaluation import run_blocking_eval
    from src.preprocessing import run_preprocess
    from src.validation import make_split

    from tests.conftest import write_tsv
    write_tsv(data_root / "train" / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"],
              [("S1-1", "S2-10,S2-11,S3-20"), ("S1-2", ""), ("S1-3", "S2-12"), ("S1-4", "")])
    cfg = replace(CFG.with_overrides(data_root=data_root, artifacts_dir=tmp_path / "a", expected_rows={}),
                  valid_fraction=1.0, blocking_eval_max_entities=None, blocking_anchor_min_df=1)
    run_preprocess("train", cfg)
    make_split(cfg)
    curve = run_blocking_eval(cfg)
    kmax = curve[curve.ranking == "rerank"].iloc[-1]
    assert kmax.pair_recall == 1.0          # incl. the Devanagari S2-12 <-> S1-3 pair
    assert kmax.zero_cand_share < 1.0


def test_final_sets_and_features(data_root, tmp_path):
    """candidates -> features on the toy data: labels align, ids/columns are consistent."""
    from src.candidate_generation import generate_final_sets
    from src.evaluation import gt_row_pairs
    from src.feature_engineering import build_features, competition_index, feature_columns
    from src.preprocessing import run_preprocess
    from tests.conftest import write_tsv

    write_tsv(data_root / "train" / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"],
              [("S1-1", "S2-10,S2-11,S3-20"), ("S1-2", ""), ("S1-3", "S2-12"), ("S1-4", "")])
    cfg = replace(CFG.with_overrides(data_root=data_root, artifacts_dir=tmp_path / "a", expected_rows={}),
                  blocking_anchor_min_df=1, feature_batch_rows=2)
    run_preprocess("train", cfg)
    generate_final_sets("train", {"toy": np.arange(4)}, cfg)
    path = build_features("train", "toy", cfg, gt_row_pairs(cfg), competition_index("train", ("toy",), cfg))
    df = pq.read_table(path).to_pandas()
    assert df["label"].sum() == 4                      # all 4 true pairs are candidates and labelled
    assert df.groupby("s1_row")["rr_gap_to_best"].max().eq(0).all()
    assert (df["tgt_n_s1"] >= 1).all() and set(df["mutual_best"].unique()) <= {0.0, 1.0}
    assert (df.groupby("tgt")["tgt_rr_rank"].min() == 1).all()
    s1 = df[df.s1_row == 0]   # S1-1 has 3 near-identical true matches: siblings are similar
    assert s1.loc[s1.label == 1, "sib_name_tsr"].min() >= 80
    cols = feature_columns(list(df.columns))
    assert "label" not in cols and "s1_row" not in cols and df[cols].dtypes.map(lambda d: d.kind in "fi").all()


def test_num_street_elision_retrieves_french_pair(tmp_path):
    """Generic name + d'/l' street spelled differently: only num_street can link these."""
    from src.candidate_generation import generate_final_sets, candidates_path
    from src.preprocessing import run_preprocess
    from tests.conftest import write_tsv

    cols = ["entity_id", "business_name", "business_address", "country"]
    root = tmp_path / "dataset"
    write_tsv(root / "train" / "train_source1.tsv", cols,
              [("S1-1", "Calais Foyer SAS", "18 Rue d'Ajaccio, Calais, Hauts-de-France", "France")])
    write_tsv(root / "train" / "train_source2.tsv", cols,
              [("S2-1", "FOYER CALAIS", "18 R D AJACCIO, CALAIS", "France"),
               ("S2-2", "Other Thing", "3 Rue Victor Hugo, Lille", "France")])
    write_tsv(root / "train" / "train_source3.tsv", cols, [("S3-1", "Unrelated", "9 Oak Ave, Reno, NV", "US")])
    cfg = replace(CFG.with_overrides(data_root=root, artifacts_dir=tmp_path / "a", expected_rows={}),
                  blocking_passes=("num_street",), blocking_anchor_min_df=2)
    run_preprocess("train", cfg)
    generate_final_sets("train", {"fr": np.arange(1)}, cfg)
    c = pq.read_table(candidates_path("fr", cfg)).to_pandas()
    assert c["tgt"].tolist() == [0]


def test_hard_negative_features():
    from src.feature_engineering import _hard_negative_features
    f = _hard_negative_features(["1425", "1302", "", "320"], ["2425", "1304", "5", "1320"],
                                ["balaji agro", "kanara marketing", "x", ""], ["balaji arto", "kanara marketinh", "x", "y"])
    assert f["num_first_edit"][:2].tolist() == [1, 1] and np.isnan(f["num_first_edit"][2])
    assert f["num_first_absdiff_log"][1] < f["num_first_absdiff_log"][0]      # neighbour (2) vs corrupted (1000)
    assert f["num_first_samelen"][3] == 0 and f["num_best_edit"][3] == 1
    assert f["name_diff_tokens_a"][0] == 1 and f["name_diff_best_sim"][1] > f["name_diff_best_sim"][0]
    assert f["name_diff_tokens_a"][2] == 0 and np.isnan(f["name_diff_best_sim"][2]) and np.isnan(f["name_diff_tokens_a"][3])


def test_dedup_chunked_equals_unchunked():
    rng = np.random.default_rng(0)
    parent = np.sort(rng.integers(0, 500, 5000)).astype(np.int32)
    code = rng.integers(0, 20, 5000).astype(np.int32)
    a, b = dedup(Tokens(parent, code)), dedup(Tokens(parent, code), chunk=97)
    assert np.array_equal(a.parent, b.parent) and np.array_equal(a.code, b.code)