"""Pipeline entry point. Run from the business_entity_resolution folder:

    python -m src.main eda --split train
    python -m src.main preprocess --split train
    python -m src.main norm-check
    python -m src.main split
    python -m src.main block --split train --fold valid                  # 100k-entity sample (default)
    python -m src.main block --split train --fold valid --max-entities 0 # whole validation fold
    python -m src.main candidates --split train     # final sets train_fit + train_valid (+ recall report)
    python -m src.main features --split train
    python -m src.main train
    python -m src.main evaluate [--reuse-scores]
    python -m src.main preprocess --split test
    python -m src.main candidates --split test
    python -m src.main features --split test
    python -m src.main predict                      # -> output/matching_results.tsv, output/candidate_pairs.tsv

Common flags: --data-root PATH  --artifacts-dir PATH
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import CFG
from .utils import setup_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ber")
    sub = parser.add_subparsers(dest="command", required=True)
    p = {name: sub.add_parser(name, help=h) for name, h in (
        ("eda", "exploratory analysis on the full local data"),
        ("preprocess", "normalise sources into <artifacts>/normalized/"),
        ("norm-check", "normalised-field agreement on train GT pairs"),
        ("split", "Source-1 entity-level train/valid split"),
        ("block", "candidate generation (+ recall report on a train fold)"),
        ("candidates", "FINAL candidate sets: train -> train_fit + train_valid, test -> test"),
        ("features", "pairwise features for the final candidate sets"),
        ("train", "fit logistic regression + LightGBM on train_fit"),
        ("evaluate", "tune decision rules on valid-tune, report macro F0.5 on valid-holdout"),
        ("predict", "score test candidates, decide matches, write + validate output/*.tsv"),
        ("package", "build <team>_submission.zip in the required layout"),
        ("errors", "where macro F0.5 is lost on the validation holdout (+ example rows)"))}
    for sp in p.values():
        sp.add_argument("--data-root", type=Path, default=None)
        sp.add_argument("--artifacts-dir", type=Path, default=None,
                        help="intermediates location (default: BER_ARTIFACTS_DIR or ./artifacts)")
    for name in ("eda", "preprocess"):
        p[name].add_argument("--split", choices=["train", "test"], default="train")
    p["eda"].add_argument("--sample-entities", type=int, default=None)
    p["eda"].add_argument("--no-strict", action="store_true", help="warn instead of raising on row-count mismatch")
    p["split"].add_argument("--overwrite", action="store_true")
    for name in ("candidates", "features"):
        p[name].add_argument("--split", choices=["train", "test"], default="train")
    p["train"].add_argument("--device", choices=["auto", "cpu", "gpu", "cuda"], default=None,
                            help="LightGBM training device (default: BER_LGBM_DEVICE or auto)")
    p["evaluate"].add_argument("--reuse-scores", action="store_true",
                               help="reuse reports/valid_scores.parquet (decision-rule changes only)")
    p["predict"].add_argument("--output-dir", type=Path, default=None)
    p["predict"].add_argument("--reuse-scores", action="store_true",
                              help="skip model scoring; use artifacts/scores/test.parquet from a previous run")
    p["package"].add_argument("--team", required=True)
    p["package"].add_argument("--include-tests", action="store_true")
    b = p["block"]
    b.add_argument("--split", choices=["train", "test"], default="train")
    b.add_argument("--fold", choices=["train", "valid"], default="valid")
    b.add_argument("--max-entities", type=int, default=None, help="0 = whole fold")
    b.add_argument("--passes", type=str, default=None, help="comma-separated subset of passes")
    b.add_argument("--max-block", type=int, default=None)
    b.add_argument("--top-k", type=int, default=None)
    b.add_argument("--tag", type=str, default=None, help="suffix for output names (experiments)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    cfg = CFG.with_overrides(
        data_root=args.data_root.resolve() if args.data_root else None,
        artifacts_dir=args.artifacts_dir.resolve() if args.artifacts_dir else None,
        eda_pair_sample_entities=getattr(args, "sample_entities", None),
        strict_row_count=False if getattr(args, "no_strict", False) else None,
        blocking_max_block=getattr(args, "max_block", None),
        blocking_top_k=getattr(args, "top_k", None),
        lgbm_device=getattr(args, "device", None))
    if getattr(args, "max_entities", None) is not None:
        cfg = replace(cfg, blocking_eval_max_entities=args.max_entities or None)
    logs = cfg.artifacts_dir / "logs"

    if args.command == "eda":
        from .eda import run_eda
        setup_logging(log_file=cfg.eda_dir(args.split) / "eda.log")
        run_eda(args.split, cfg)
    elif args.command == "preprocess":
        from .preprocessing import run_preprocess
        setup_logging(log_file=logs / f"preprocess_{args.split}.log")
        run_preprocess(args.split, cfg)
    elif args.command == "norm-check":
        from .eda import normalization_check
        setup_logging(log_file=logs / "norm_check.log")
        normalization_check(cfg)
    elif args.command == "split":
        from .validation import make_split
        setup_logging(log_file=logs / "split.log")
        make_split(cfg, overwrite=args.overwrite)
    elif args.command == "block":
        setup_logging(log_file=logs / f"block_{args.split}_{args.fold}_{args.tag or 'base'}.log")
        passes = tuple(args.passes.split(",")) if args.passes else None
        if args.split == "train":
            from .evaluation import run_blocking_eval
            run_blocking_eval(cfg, args.fold, passes, args.tag)
        else:
            from .candidate_generation import generate_candidates
            generate_candidates("test", "test" + (f"_{args.tag}" if args.tag else ""), None, cfg, passes)
    elif args.command == "candidates":
        setup_logging(log_file=logs / f"candidates_{args.split}.log")
        from .candidate_generation import generate_final_sets
        if args.split == "train":
            from .evaluation import report_final_sets
            generate_final_sets("train", _train_sets(cfg), cfg)
            report_final_sets(cfg)
        else:
            generate_final_sets("test", {"test": None}, cfg)
    elif args.command == "features":
        setup_logging(log_file=logs / f"features_{args.split}.log")
        from .feature_engineering import build_features, feature_summary
        if args.split == "train":
            from .evaluation import gt_row_pairs
            from .feature_engineering import competition_index
            truth = gt_row_pairs(cfg)
            comp = competition_index("train", ("train_fit", "train_valid", "train_other"), cfg)
            for name in ("train_fit", "train_valid"):
                build_features("train", name, cfg, truth, comp)
            feature_summary("train_valid", cfg)
        else:
            from .feature_engineering import competition_index
            build_features("test", "test", cfg, None, competition_index("test", ("test",), cfg))
    elif args.command == "train":
        setup_logging(log_file=logs / "train.log")
        from .model_training import train_models
        train_models(cfg)
    elif args.command == "evaluate":
        setup_logging(log_file=logs / "evaluate.log")
        from .model_training import evaluate_models
        evaluate_models(cfg, reuse_scores=args.reuse_scores)
    elif args.command == "predict":
        setup_logging(log_file=logs / "predict.log")
        from .inference import run_inference
        run_inference(cfg, args.output_dir, reuse_scores=args.reuse_scores)
    elif args.command == "errors":
        setup_logging(log_file=logs / "errors.log")
        from .error_analysis import run_error_analysis
        run_error_analysis(cfg)
    elif args.command == "package":
        setup_logging(log_file=logs / "package.log")
        from .package import build_package
        build_package(args.team, cfg, args.include_tests)


def _train_sets(cfg) -> dict:
    """Row sets: a seeded sample of the train fold (fit), the validation fold, and the rest of the
    train fold (``train_other``, never labelled for the model). Together they cover EVERY train
    Source-1 entity, so target-competition features see the same population as at test time."""
    import numpy as np
    from .validation import fold_rows
    rng = np.random.default_rng(cfg.random_seed)
    sets = {}
    for name, fold, cap in (("train_fit", "train", cfg.model_train_entities),
                            ("train_valid", "valid", cfg.model_valid_entities)):
        rows = fold_rows(fold, cfg)
        if cap and len(rows) > cap:
            rows = np.sort(rng.choice(rows, size=cap, replace=False))
        sets[name] = rows
    covered = np.concatenate(list(sets.values()))
    all_rows = np.concatenate([fold_rows("train", cfg), fold_rows("valid", cfg)])
    sets["train_other"] = np.sort(np.setdiff1d(all_rows, covered))
    return sets


if __name__ == "__main__":
    main()
