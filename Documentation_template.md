# Business Entity Resolution — Methodology

**Team:** SolveByFirstPrinciple · **Challenge:** Amazon ML Challenge 2026
**Public leaderboard:** 0.957658 macro F0.5 · **Validation holdout:** 0.9694 macro F0.5

All numbers in this document were measured with the submitted code. Validation numbers come from Source 1 entities held out of the training data. No test labels were used at any point.

---

## 1. Problem summary

For each Source 1 business record, find every record in Source 2 and Source 3 that refers to the same real-world business. Any entity may have zero, one or many matches.

The metric is **macro F0.5 per Source 1 entity**, which weights precision twice as heavily as recall. An entity with no true matches (a singleton) scores 1 if we predict nothing and 0 otherwise.

The whole pipeline runs on a 16 GB RAM Windows laptop, CPU only.

---

## 2. Data findings that shaped the design

These come from a full scan of the training data (`src/eda.py`).

| Finding | Value | Design consequence |
|---|---|---|
| Train sizes (S1 / S2 / S3) | 2.21 M / 5.03 M / 5.29 M | Every possible S1 × (S2 + S3) pair would be 2.28 × 10¹³ pairs, so blocking is mandatory |
| Singletons | 5.6% of S1 | The model must be able to predict an empty set |
| True matches per entity | mean 3.46, max 11; at most 5 (S2) and 6 (S3) from one source | Keep about 8 candidates per source |
| S2/S3 records with no match (distractors) | 26% | Precision matters |
| S2/S3 records matched to more than one S1 entity | **0** | Each target should be assigned to at most one S1 entity (§6) |
| Matched pairs with the same country label | 100% | Block within each country label; unseen labels (France) get their own partition |
| Non-Latin business names | 9.4% of S2, 5.3% of S3 (10 Indian scripts) | Needs cross-script matching (§3) |
| Empty addresses | 3.4% of S2/S3, 0% of S1 | Missing values become explicit features |
| Shortcuts from entity IDs or file row order | Spearman correlation ≈ 0.00 | None to exploit or to guard against |
| Test set | 1.73 M S1 records: India 47%, US 38%, **France 15%** (absent from training) | No country-specific features; generic normalisation rules |

---

## 3. Normalisation (`preprocessing.py`, `lexicons.py`)

Each raw field is kept unchanged. The normalised versions are added alongside it and stored as Parquet (about 4.5 minutes per split, 4 GB peak RAM).

**Names**
- Unicode NFKC normalisation, lower-casing, and removal of Latin accents.
- Legal forms unified: Pvt/Private, Ltd/Limited, LLC, Inc, Corp, SARL, SAS, EURL, Cie, …
- Digits used as letters inside words fixed (`5haan` → `shaan`, `Regi0nal` → `regional`).
- Domain-style names reduced to the name (`stylemulti.com` → `stylemulti`).
- Text after `dba` / `aka` / `t/a` kept as an alternate trade name.
- Repeated tokens and long digit noise removed.
- Output fields: `name_core` (the distinctive tokens), `name_compact` (core without spaces) and `name_legal` (the legal form).

**Cross-script names**
- Indic scripts are romanised using each character's official Unicode name. This handles the inherent vowel, the virama and word-final vowel dropping, needs no external service and works for every Brahmic script.
- Every token is then reduced to a **consonant skeleton**: first letter, then consonants with voiced/unvoiced pairs merged. Example: `प्राइवेट` → `praivett` → `prpt` = `private` → `prpt`.

**Addresses**
- Street-type abbreviations unified (St/Street, Rd/Road, Ave, Blvd, Rue → r, …).
- Old city names mapped to current ones (Calcutta → Kolkata, Gurgaon → Gurugram, …).
- PO boxes removed.
- House numbers kept whole (`2/1a`, `3-110/6/g-3`) with leading zeros stripped (`00115` → `115`).
- A state/region code is extracted when the country is known, including state names written in Indian scripts.

The lexicons are generic hand-written lists (legal forms, abbreviations, region names). No external data or lookup of any kind is used.

---

## 4. Candidate generation (`blocking.py`, `candidate_generation.py`)

### 4.1 Blocking keys

Every key is one 64-bit integer that encodes (value, country, pass), so records are only compared with records carrying the same country label. There are **12 complementary passes**:

| Pass | Key | Catches |
|---|---|---|
| `name_tok` | Rare name token (includes trade names) | Distinctive names |
| `name_pair` | Pair of name tokens | Rare combinations of common words |
| `skel_set` / `skel_pair` | Skeleton token set / skeleton pair | Names written in another script |
| `compact` / `compact_prefix` | Name with spaces removed / its first 8 characters | Domain-style and concatenated names |
| `addr_tok` | Rare address token | Distinctive street or locality names |
| `num_anchor` | House number × common address token | `9` + `chennai` |
| `name_num` / `skel_num` | Name or skeleton token × house number | Common names at a specific number |
| `skel_anchor` | Skeleton token × common address token | Common names in a specific city |
| `num_street` | House number × uncommon street word, with French elision variants (`d'Ajaccio` = `D AJACCIO`) | Generic French names (`Nantes Foyer SAS`) |

**Hub safeguard.** A key shared by more than 100 target records is skipped. The records in it are not dropped: the combination passes still reach them, because those passes exist precisely for parts that are too common on their own. Index statistics for every pass (key instances, capped keys, largest capped block) are logged.

### 4.2 Pruning to the final candidate set

1. **Pool:** merge all passes and score each pair by the rarity-weighted keys it shares: the sum of `pass_weight × log(1 + 100 / block size)`. Keep the top 200 per source.
2. **Re-rank:** reorder the pool by RapidFuzz name similarity (core name and skeleton), address token-set similarity and the key score.
3. **Final set:** keep the **top 8 per source (16 per entity)**. `candidate_pairs.tsv` is exactly this set, and it is exactly what the model scores.

### 4.3 Blocking quality (measured)

| Stage (validation) | Pair recall | Oracle macro F0.5 * | Candidates / entity |
|---|---|---|---|
| 8 passes, key score only, K = 30 (100 k sample) | 0.916 | 0.962 | 54 |
| + 3 combination passes, steeper weights, K = 30 | 0.955 | 0.982 | 59 |
| + re-ranking, **K = 8** (full validation fold, 441 k) | 0.9575 | 0.9853 | 16 |
| + `num_street` pass (validation holdout, final) | **0.9604** | **0.9864** | 16 |

\* Oracle macro F0.5 is the score of a perfect matcher restricted to the candidates, i.e. the ceiling blocking sets for the model.

**Test set (final):**

| Metric | Value |
|---|---|
| Candidate pairs | 27,627,231 |
| Candidates per entity | mean 15.95, median 16, maximum 16 |
| Reduction ratio, 1 − candidates / (\|S1\| × (\|S2\| + \|S3\|)) | **0.9999984** |
| Entities with no candidates | 4,734 → **1,283** after `num_street` (France 4,158 → 823, India 572 → 458, US 4 → 2) |

---

## 5. Features and model

### 5.1 Features (`feature_engineering.py`)

About 65 features per candidate pair. Missing information is left as NaN, never 0. There is no country feature, because the test set contains a country (France) the model has never seen.

| Group | Examples | Purpose |
|---|---|---|
| Name similarity | Edit, token-sort, token-set and partial ratios on normalised and core names; skeleton similarity; Jaro-Winkler and prefix cover on compact names; trade-name similarity; legal-form agreement or conflict | Name evidence, including cross-script |
| Address similarity | Token-set, sort and partial similarity; digit-run overlap; first-number equality; shared long numbers; region agreement; empty-address flags | Address evidence |
| Hard negatives | Edit distance and numeric distance between house numbers; digit-length agreement; number and character similarity of the name words that differ | Separate data noise (1425 → 2425) from a different business next door (1302 vs 1304), and typos (`marketinh`) from different words (`Agro` / `Arto`) |
| Blocking | Key score, re-rank score, ranks, one flag per pass, number of passes | How the pair was found |
| Entity context | Gap to the entity's best candidate; rank within the source | Whether anything is clearly better (singletons) |
| Siblings | Name and address similarity to the entity's best other candidate | True matches of one business are copies of each other |
| Target competition | How many S1 entities hold this target; this entity's rank and score gap for it; mutual-best flag | Exploits the finding that no target belongs to two entities |

Competition features use only the unsupervised re-rank score, never labels. They are computed over **all** S1 entities of a split, the same population as at test time.

### 5.2 Models compared (`model_training.py`)

| Model | Holdout macro F0.5 |
|---|---|
| Rule: threshold on the re-rank score | 0.7744 |
| Logistic regression (scikit-learn, BSD-3) | 0.9358 |
| **LightGBM (MIT)**, selected | **0.9694** |

**LightGBM setup**
- Parameters: binary objective, learning rate 0.05, 127 leaves, `min_data_in_leaf` 200, 0.8 feature and bagging fractions, L2 regularisation 1.0, `deterministic=True`, seed 42.
- Training data: 300 k train-fold entities, 4.79 M candidate pairs, using every real candidate with no negative sub-sampling. The probabilities are therefore on the same scale at training and inference.
- Early stopping on a 10% held-out set of *train-fold entities*; the first model stopped at iteration 2,878.
- Runs on the CPU. GPU training is supported by the code but not available with the Windows LightGBM wheel.

---

## 6. Matching decisions (`matching.py`)

Every rule can return an empty set; nothing forces an entity to match.

1. **Expected-F0.5 set selection.** For each entity, sort candidates by probability and pick the number *k* that maximises the estimated F0.5:

   E[F | top-k] ≈ 1.25 · Σᵢ≤ₖ pᵢ / (k + 0.25 · (Σ p + m))

   Here *m* = 0.135 is the average number of true matches per entity that fall outside the candidates, measured on the training fold. Predicting nothing has an expected score of P(singleton) ≈ ∏(1 − pᵢ) × `empty_scale`. The rule predicts empty when that is higher.
2. **Target exclusivity.** Each S2/S3 record keeps only its highest-probability S1 entity.
3. **Tuned choice:** a threshold rule was also tried; expected-F0.5 with `empty_scale` = 3.0 and exclusivity on was selected on the tune half. The value 3.0 is an interior optimum of the grid 0.5–15.

---

## 7. Validation design

- **Entity-level split.** 80% of S1 entities form the train fold and 20% the validation fold. The split is seeded on sorted IDs. All pairs of a held-out entity stay out of training.
- **Two validation halves.** The validation fold is split again by entity:
  - *tune* half: chooses the decision rule and its parameters;
  - *holdout* half: reported, and never used for any choice.

  The tune and holdout scores agree within 0.0001 for every model, so the tuned parameters are not overfitted.
- **Every held-out entity is scored**, including entities with no candidates or with true matches that were never retrieved. The reported score is therefore the true end-to-end metric, not a candidate-conditional one.
- **Unsupervised components** (vocabularies, block sizes, competition statistics) use source text only. That text is equally available at test time, so this is not leakage.
- **The official metric** is implemented in `evaluation.macro_f05` and unit-tested against the README example (0.714) and the singleton rules.

---

## 8. Results

### 8.1 Final model (validation holdout, 220,682 entities)

| Metric | Value |
|---|---|
| **Macro F0.5** | **0.9694** |
| Singletons / non-singletons | 0.9613 / 0.9698 |
| Micro precision / recall | 0.9925 / 0.9326 |
| False matches / true matches missed among candidates / true matches never in candidates | 5,363 / 21,314 / 30,282 |

### 8.2 Improvement history

| Version | Change | Holdout | Leaderboard |
|---|---|---|---|
| v1 | Blocking + LightGBM + expected-F0.5 + exclusivity | 0.9554 | 0.943 |
| v2 | + sibling and target-competition features | 0.9618 | not submitted |
| **v3** | + `num_street` pass, hard-negative number and name features | **0.9694** | **0.957658** |

The gap between validation and leaderboard (about 0.012) is the same for v1 and v3. Likely reasons: the public leaderboard scores only part of the test set, and France (15% of test) has no training labels.

### 8.3 Where the remaining 0.031 is lost (holdout counterfactuals, `error_analysis.py`)

| Error source | Macro F0.5 lost |
|---|---|
| True matches never retrieved by blocking | 0.0136 |
| Retrieved true matches the model did not predict | 0.0101 |
| False matches | 0.0070 |

---

## 9. Memory and runtime (16 GB laptop, CPU)

| Step | Time | Peak RAM |
|---|---|---|
| Preprocess train / test | 4.4 min each | 4.0 GB |
| Target block index (12 passes) | ~3.5 min | 6.6 GB |
| Candidate query | ~1,000 entities/s | included above |
| Features (4.8 M + 7.1 M pairs) | ~7 min | 5.3 GB |
| LightGBM training | ~12–15 min | 3.0 GB |
| Test scoring (27.6 M pairs) | 41 min | 2.5 GB |
| Decide + write + check submission | 1.7 min | 5.1 GB |

How memory stays within 16 GB:
- Arrow string columns instead of Python objects.
- One source table in memory at a time, and target columns loaded one at a time.
- Integer-encoded keys, sort-based joins instead of hash maps of tuples, and no Cartesian products.
- Hub cap on block sizes.
- Chunked queries, feature batches and decision steps.
- Pair scores cached to disk.

---

## 10. Limitations and future work

- **Blocking ceiling (0.986).** The remaining misses are mostly targets with an empty address and a very common name (`maid cafe`, `bright tattoo`). A learned re-ranker in place of the hand-set weights would get more of the 200-per-source pool into the top 8; that pool reaches about 0.990 on the validation sample.
- **France has no labels.** Its quality cannot be measured directly. Its predicted-empty share and matches per entity (5.8%, 3.27) are close to India's and the US's, but that proves nothing about accuracy.
- **Model.** LightGBM hyperparameters were not tuned. The first model stopped close to its 3,000-round cap. An entity-level second-stage model, fed the first stage's out-of-fold probabilities, is the most promising next step for the matcher losses (0.017).
- **Validation.** A stricter split that keeps look-alike names in the same fold was not run.
- **Runtime.** Test scoring is dominated by LightGBM prediction, which is CPU-only. A higher learning rate with fewer trees would shorten it.

---

## 11. Output files

Both files are tab-separated and UTF-8, with the exact headers required by the challenge. Each has exactly one row per Source 1 test entity (1,732,544 rows), in Source 1 file order. ID lists are comma-separated, not quoted, contain no duplicates, and are empty when there is nothing to list.

| File | Columns | Content |
|---|---|---|
| `output/matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` | Final matches (the leaderboard file). 104,610 entities have an empty list |
| `output/candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` | Exactly the pairs the model scored: the top 8 per source after blocking and re-ranking. 1,283 entities have an empty list |

**Checks**
- **Internal:** `submission.py` re-reads both files and checks the header, one row per entity, no duplicate rows or IDs, only valid S2/S3 test IDs, and that every match is also a candidate.
- **Official:** `utils/validate_submission.py` returns `PASS`.

---

## 12. Compliance and reproducibility

**Compliance**
- **Models:** LightGBM (MIT licence; a gradient-boosted tree ensemble, orders of magnitude below the 8-billion-parameter limit) and scikit-learn logistic regression (BSD-3, baseline only). No pretrained models, LLMs or embeddings are used.
- **Data:** only the provided challenge data. There are no external databases, APIs, geocoding services or internet lookups. `lexicons.py` contains generic hand-written lists: legal-form suffixes, address abbreviations and region names.
- **Test data:** labels are never used. Test source records are only read to generate candidates, features and predictions. Every fitted or tuned setting comes from the training data.

**Reproducibility**
- **Seeds:** 42 for the split, sampling and LightGBM (`deterministic=True`).
- **Dependencies:** all pinned in `code/business_entity_resolution/requirements.txt` (Python 3.13).
- **End-to-end steps:** listed in `code/business_entity_resolution/README.md`: preprocess → split → candidates → features → train → evaluate → test candidates → test features → predict → validate → package. Every step writes its intermediate files, so any single step can be re-run.

**Code map**

| Topic | Modules |
|---|---|
| Normalisation | `preprocessing.py`, `lexicons.py` |
| Blocking | `blocking.py`, `candidate_generation.py` |
| Features | `feature_engineering.py` |
| Models | `model_training.py` |
| Decisions | `matching.py` |
| Metric | `evaluation.py` |
| Output | `inference.py`, `submission.py` |