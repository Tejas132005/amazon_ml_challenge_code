"""Central configuration: paths, schemas, seeds and tunable limits.

Every path and magic number used by the pipeline lives here. Override the
dataset location with the BER_DATA_ROOT environment variable or the
``--data-root`` CLI flag instead of editing code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../business_entity_resolution

# ---- Schemas (exact column order required by the challenge) ---------------
SOURCE_COLUMNS: tuple[str, ...] = ("entity_id", "business_name", "business_address", "country")
GT_COLUMNS: tuple[str, ...] = ("source1_entity_id", "matched_entity_ids")
SOURCE_PREFIXES: dict[int, str] = {1: "S1", 2: "S2", 3: "S3"}
TARGET_SOURCES: tuple[int, ...] = (2, 3)


def _default_data_root() -> Path:
    env = os.environ.get("BER_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # Assumes: student_resource/dataset  and  student_resource/code/business_entity_resolution
    return (PROJECT_ROOT.parent.parent / "dataset").resolve()


SUBMISSION_ROOT = PROJECT_ROOT.parent.parent  # folder holding code/, dataset/, utils/, output/


def _default_output_dir() -> Path:
    # <root>/output, i.e. next to dataset/ and utils/ -- the layout the validator command and
    # the submission zip expect. Override with BER_OUTPUT_DIR.
    env = os.environ.get("BER_OUTPUT_DIR")
    return Path(env).expanduser().resolve() if env else SUBMISSION_ROOT / "output"


def _default_artifacts_dir() -> Path:
    # Keep large intermediates OUTSIDE cloud-synced folders (OneDrive/Dropbox): set BER_ARTIFACTS_DIR.
    env = os.environ.get("BER_ARTIFACTS_DIR")
    return Path(env).expanduser().resolve() if env else PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class Config:
    data_root: Path = field(default_factory=_default_data_root)
    artifacts_dir: Path = field(default_factory=_default_artifacts_dir)
    output_dir: Path = field(default_factory=_default_output_dir)  # submission TSVs
    predict_batch_rows: int = 1_000_000
    random_seed: int = 42

    # ---- Reading --------------------------------------------------------
    arrow_block_size: int = 64 * 1024 * 1024  # bytes per parse block (~one Arrow chunk)
    strict_row_count: bool = True  # raise if parsed rows != physical data lines
    # Known full-file row counts (train). None = skip the check (e.g. test).
    expected_rows: dict[str, int] = field(default_factory=lambda: {
        "train_ground_truth": 2_206_821,
        "train_source1": 2_206_821,
        "train_source2": 5_034_616,
        "train_source3": 5_285_603,
    })

    # ---- EDA ------------------------------------------------------------
    eda_pair_sample_entities: int = 200_000  # non-singleton S1 entities for pair-level stats
    eda_top_tokens: int = 200  # rows written to token-frequency CSVs
    eda_token_thresholds: tuple[int, ...] = (1_000, 10_000, 100_000)
    eda_hub_token_count: int = 10_000  # token occurring >= this in S2+S3 counts as a hub
    eda_token_keep_min: int = 100  # only tokens with >= this count are kept in memory for pair stats
    eda_min_country_rows: int = 1  # countries with fewer rows are still reported (open set)
    eda_scripts: tuple[str, ...] = (
        "Latin", "Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil",
        "Telugu", "Kannada", "Malayalam", "Arabic", "Cyrillic", "Greek", "Han",
    )
    quantiles: tuple[float, ...] = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)

    # ---- Validation split (Source-1 entity level) ----------------------
    valid_fraction: float = 0.2

    # ---- Blocking / candidate generation (Phase 3) --------------------
    blocking_passes: tuple[str, ...] = ("name_tok", "name_pair", "skel_set", "skel_pair", "compact",
                                        "compact_prefix", "addr_tok", "num_anchor", "name_num", "skel_num",
                                        "skel_anchor", "num_street")
    blocking_pass_weight: dict[str, float] = field(default_factory=lambda: {
        "name_tok": 1.0, "name_pair": 1.0, "skel_set": 1.0, "skel_pair": 0.7, "compact": 1.5,
        "compact_prefix": 1.0, "addr_tok": 1.0, "num_anchor": 1.0, "name_num": 1.0, "skel_num": 0.8,
        "skel_anchor": 0.8, "num_street": 1.0})
    blocking_max_block: int = 100          # keys matching more targets than this are skipped (hub keys)
    blocking_max_name_tokens: int = 6      # tokens per name used for pair keys
    blocking_max_numbers: int = 2          # digit runs per address used in number x anchor keys
    blocking_max_anchors: int = 4          # common address tokens per address used as anchors
    blocking_anchor_min_df: int = 300      # address tokens at least this frequent can be anchors
    blocking_combo_name_tokens: int = 3    # name/skeleton tokens per record used in name x address keys
    blocking_combo_anchors: int = 3        # anchors per record used in skel_anchor keys
    blocking_max_street_tokens: int = 4    # uncommon street words per record used in num_street keys
    blocking_compact_prefix_len: int = 8
    blocking_union_k: int = 200            # per source: key-score cut applied before re-ranking
    blocking_rerank: bool = True           # re-order the union with string similarity before the final cut
    blocking_rerank_weights: dict[str, float] = field(default_factory=lambda: {"name": 0.45, "addr": 0.35, "key": 0.2})
    blocking_rerank_key_norm: float = 10.0
    blocking_top_k: int = 8                # FINAL candidates per Source-1 entity *per source* (S2, S3)
                                           # v3 valid: K=8 -> recall 0.957, oracle F0.5 0.985, 16 cands/entity
    blocking_eval_ks: tuple[int, ...] = (1, 3, 5, 8, 10, 15, 20, 30, 50, 100)
    blocking_chunk_size: int = 20_000      # Source-1 entities queried per batch
    blocking_eval_max_entities: int | None = 100_000  # None = whole validation fold

    # ---- Model data (Phase 5/6) ------------------------------------------
    model_train_entities: int | None = 300_000   # train-fold entities used to fit the matcher (None = all)
    model_valid_entities: int | None = None      # valid-fold entities scored for tuning (None = all)
    feature_batch_rows: int = 1_000_000          # candidate pairs per feature batch
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 200,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
        "max_bin": 255, "deterministic": True, "force_row_wise": True, "verbose": -1})
    # ---- LightGBM device (training only; LightGBM prediction always runs on CPU) ----
    # "auto": probe cuda -> gpu (OpenCL) -> cpu and use the first that works; or force one.
    lgbm_device: str = field(default_factory=lambda: os.environ.get("BER_LGBM_DEVICE", "auto"))
    lgbm_gpu_platform_id: int = -1        # OpenCL platform (-1 = LightGBM default)
    lgbm_gpu_device_id: int = -1          # OpenCL device on that platform (-1 = default)
    lgbm_gpu_max_bin: int | None = 63     # LightGBM GPU docs recommend 63 for speed; None keeps lgbm_params
    lgbm_probe_timeout_s: int = 120       # seconds allowed for the device probe subprocess
    lgbm_max_rounds: int = 3000
    lgbm_early_stopping: int = 100
    early_stop_fraction: float = 0.1             # train_fit entities held out for LightGBM early stopping
    logreg_max_rows: int = 2_000_000             # logistic regression fitted on a pair sample of this size
    tune_fraction: float = 0.5                   # valid entities used for threshold tuning; rest = holdout

    def source_path(self, split: str, source_no: int) -> Path:
        return self.data_root / split / f"{split}_source{source_no}.tsv"

    def gt_path(self, split: str = "train") -> Path:
        return self.data_root / split / f"{split}_ground_truth.tsv"

    def eda_dir(self, split: str) -> Path:
        return self.artifacts_dir / "eda" / split

    def with_overrides(self, **kwargs) -> "Config":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


CFG = Config()
