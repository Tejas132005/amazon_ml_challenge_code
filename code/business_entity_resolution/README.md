# Business Entity Resolution — reproduction guide

**Team:** SolveByFirstPrinciple · **Challenge:** Amazon ML Challenge 2026

This folder contains the complete pipeline that produces the two submission files:

- `output/matching_results.tsv`: final matches for every Source 1 test entity
- `output/candidate_pairs.tsv`: the exact candidate set the model scored

The method is described in `Documentation_template.md` at the root of the submission.

---

## 1. Requirements

| Item | Used for development |
|---|---|
| OS | Windows 10/11 (the code is plain Python and also runs on Linux/macOS) |
| Python | 3.13 |
| RAM | 16 GB (measured peak: 6.6 GB) |
| Disk | ~15 GB free for intermediate files |
| GPU | Not required. LightGBM training uses a GPU only if the installed LightGBM build supports one (see §7) |

All dependencies are pinned in `requirements.txt`: numpy, pandas, pyarrow, psutil, rapidfuzz, scikit-learn, scipy, joblib, lightgbm and pytest. Every model and library has a permissive licence (MIT / BSD / Apache-2.0). No pretrained models, external APIs or external data are used.

---

## 2. Folder layout

The pipeline expects the challenge data next to the `code/` folder:

```text
<root>/                                   e.g. SolveByFirstPrinciple/
├── dataset/                              challenge data (not part of the submission zip)
│   ├── train/  train_source1..3.tsv, train_ground_truth.tsv
│   └── test/   test_source1..3.tsv
├── utils/validate_submission.py          challenge validator (not part of the zip)
├── output/                               written by the pipeline
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/                              all source code (entry point: src/main.py)
│   ├── README.md                         this file
│   └── requirements.txt
└── Documentation_template.md             methodology write-up
```

If the data lives somewhere else, pass `--data-root <path>` to any command or set the `BER_DATA_ROOT` environment variable.

---

## 3. Setup (PowerShell)

```powershell
cd <root>
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
cd code\business_entity_resolution
pip install -r requirements.txt
```

Recommended: keep the multi-GB intermediate files outside cloud-synced folders such as OneDrive:

```powershell
$env:BER_ARTIFACTS_DIR = "C:\ber_artifacts"      # current terminal only
setx BER_ARTIFACTS_DIR C:\ber_artifacts          # permanent; reopen the terminal afterwards
```

Without it, intermediates go to `code/business_entity_resolution/artifacts/`.

---

## 4. Reproduce the submission

Run every command from `code/business_entity_resolution`, in this order. Each step saves its results and starts from the previous step's files, so a failed step can simply be re-run on its own.

| # | Command | What it does | Time* |
|---|---|---|---|
| 1 | `python -m src.main preprocess --split train` | Normalise train names and addresses → parquet | 5 min |
| 2 | `python -m src.main preprocess --split test` | Same for test | 5 min |
| 3 | `python -m src.main split` | Seeded 80/20 train/validation split of Source 1 entities | < 1 min |
| 4 | `python -m src.main candidates --split train` | Blocking for **all** train entities, plus a candidate-recall report | 45 min |
| 5 | `python -m src.main features --split train` | Pair features for the training and validation sets, with labels | 6 min |
| 6 | `python -m src.main train --device cpu` | Train LightGBM (and the logistic-regression baseline) | 15 min |
| 7 | `python -m src.main evaluate` | Tune the decision rule on half of validation; report the other half | 15 min |
| 8 | `python -m src.main candidates --split test` | Blocking for all 1.73 M test entities | 35 min |
| 9 | `python -m src.main features --split test` | Pair features for test | 10 min |
| 10 | `python -m src.main predict` | Score, decide matches, write `<root>/output/*.tsv` and check every submission rule | 70 min |

\* Measured on the development laptop (16 GB RAM, CPU only), except steps 4 and 9, which are estimates for the final code.

**If step 10 fails after scoring:** the pair scores are cached, so re-run with `python -m src.main predict --reuse-scores` to skip the 60-minute scoring step.

### Validate and package

```powershell
cd ..\..                                          # <root>
python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test

cd code\business_entity_resolution
python -m src.main package --team SolveByFirstPrinciple
```

`package` re-checks the output files and builds `<root>/SolveByFirstPrinciple_submission.zip` with exactly the required layout. Caches, data, intermediate files and tests are left out.

### Optional analysis commands

These produce reports only; no pipeline step depends on them.

| Command | Output |
|---|---|
| `python -m src.main eda --split train` (or `test`) | Data profile: counts, missing values, duplicates, scripts, match-count distribution |
| `python -m src.main norm-check` | How much matching signal the normalised fields expose on true pairs |
| `python -m src.main block --split train --fold valid` | Recall and candidate-count curve over K, including a union-pool ceiling row |
| `python -m src.main errors` | Where macro F0.5 is lost on the validation holdout, with example rows |

---

## 5. Source code map (`src/`)

| Module | Responsibility |
|---|---|
| `main.py` | Command-line entry point (all commands above) |
| `config.py` | Every path, seed, limit and model setting in one place |
| `data_loader.py` | Strict TSV reading: tab only, no quote handling, no NA guessing, row-count checks |
| `eda.py` | Full-data profiling and normalisation checks |
| `lexicons.py` | Hand-written generic lists: legal forms, address abbreviations, region names |
| `preprocessing.py` | Name/address normalisation, Indic-script romanisation, consonant-skeleton keys |
| `blocking.py` | Blocking keys and the inverted block index (with the hub cap) |
| `candidate_generation.py` | 12 blocking passes → union → re-rank → top-K per source |
| `validation.py` | Source 1 entity-level train/validation split |
| `feature_engineering.py` | Pair, per-entity context, sibling, target-competition and hard-negative features |
| `model_training.py` | Baselines and LightGBM training; decision-rule tuning on validation |
| `device.py` | Picks CPU or GPU for LightGBM training |
| `matching.py` | Decision rules (threshold, expected-F0.5 with empty sets allowed) and target exclusivity |
| `evaluation.py` | Official macro F0.5 metric, candidate-recall metrics |
| `error_analysis.py` | Error breakdown and examples on the validation holdout |
| `inference.py` | Test scoring (streamed, cached) and the decision step |
| `submission.py` | Writes the two TSVs and re-checks every README rule |
| `package.py` | Builds the submission zip |

Unit tests were used during development. They are not part of the required submission layout and are not included.

---

## 6. Reproducibility

- **Seeds:** 42 for the split, sampling and LightGBM (`deterministic=True`). The split is drawn from sorted IDs, so file order doesn't matter.
- **Versions:** all dependency versions are pinned in `requirements.txt`.
- **Test data:** labels are never used; test source files are only read to produce predictions. Every fitted or tuned setting comes from the training data (model: train fold; decision rule: validation tune half).
- **Leakage:** the metrics reported in `Documentation_template.md` come from the validation holdout half, which is never used for any choice.

---

## 7. GPU (optional)

`train --device auto` tests CUDA, then OpenCL (`gpu`), then CPU, each with a tiny LightGBM run in a separate process, and uses the first one that works.

The official Windows wheel of `lightgbm==4.7.0` is built **without** GPU support: on an RTX 3050 laptop it reported "Please recompile with CMake option -DUSE_GPU=1". So on Windows, `auto` resolves to CPU. Use `--device cpu` to reproduce the submitted model exactly.

LightGBM prediction is CPU-only in every build.