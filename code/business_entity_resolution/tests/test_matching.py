import numpy as np
import pyarrow as pa

from src.evaluation import macro_f05
from src.matching import decide, exclusive, expected_f_rule


def truth(pairs):
    r, t = zip(*pairs)
    return pa.table({"s1_row": pa.array(r, pa.int32()), "tgt": pa.array(t, pa.int32()), "src": pa.array([2] * len(r), pa.int8())})


def test_macro_f05_matches_readme_example_and_singletons():
    # entity 0: truth {1, 3}, pred {1, 2, 3} -> P=2/3, R=1 -> 0.714; entity 1: singleton, empty -> 1;
    # entity 2: singleton, predicted -> 0; entity 3: non-singleton, empty -> 0
    tr = truth([(0, 1), (0, 3), (3, 7)])
    res = macro_f05(np.array([0, 0, 0, 2]), np.array([1, 2, 3, 9]), np.array([0, 1, 2, 3]), tr, 100)
    f0 = 1.25 * (2 / 3) * 1 / (0.25 * (2 / 3) + 1)
    assert abs(f0 - 0.714) < 1e-3
    assert abs(res["macro_f05"] - (f0 + 1 + 0 + 0) / 4) < 1e-9
    assert res["singletons_with_false_match"] == 1 and res["non_singletons_predicted_empty"] == 1


def test_duplicate_predictions_ignored():
    tr = truth([(0, 1)])
    res = macro_f05(np.array([0, 0]), np.array([1, 1]), np.array([0]), tr, 10)
    assert res["macro_f05"] == 1.0


def test_expected_f_allows_empty_and_multi():
    rows = np.array([0, 0, 0, 1, 1])
    p = np.array([0.95, 0.9, 0.05, 0.1, 0.05])
    m = expected_f_rule(rows, p)
    assert m.tolist() == [True, True, False, False, False]  # entity 1 -> empty set


def test_exclusive_keeps_best_source1():
    rows, tg, p = np.array([0, 1, 1]), np.array([5, 5, 6]), np.array([0.6, 0.9, 0.8])
    assert exclusive(rows, tg, p, np.ones(3, bool)).tolist() == [False, True, True]


def test_decide_threshold():
    assert decide(np.array([0, 0]), np.array([1, 2]), np.array([0.2, 0.8]),
                  {"rule": "threshold", "t": 0.5}).tolist() == [False, True]


def test_expected_f_chunked_equals_unchunked():
    from src.matching import expected_f_chunked
    rng = np.random.default_rng(0)
    rows = np.sort(rng.integers(0, 3000, 40_000)).astype(np.int32)
    p = rng.random(40_000).astype(np.float32) ** 3
    full = expected_f_rule(rows, p, 2.0, 0.1)
    assert np.array_equal(expected_f_chunked(rows, p, 2.0, 0.1, chunk_pairs=1_000), full)
    perm = rng.permutation(len(p))  # unsorted input gives the same per-pair decisions
    assert np.array_equal(expected_f_chunked(rows[perm], p[perm], 2.0, 0.1, chunk_pairs=1_000), full[perm])
