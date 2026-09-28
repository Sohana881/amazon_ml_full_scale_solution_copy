# Amazon ML Challenge — Business Entity Resolution
## Status Package (honest snapshot, no fabricated results)

This package contains exactly what has been implemented and verified so far.
Nothing in here has been "topped up" to look more complete than it is.

---

## 1. WHAT HAS ACTUALLY BEEN IMPLEMENTED

`code/business_entity_resolution/src/` contains a working, end-to-end,
source-independent (S2 and S3 handled identically) preprocessing pipeline:

- `normalize.py` — name/address normalization.
- `blocking.py` — blocking-key generators:
  - Name: `NP6, NP8, NS6, FL, FL3, F2, FML`
  - Unicode-preserving name keys (non-Latin scripts, e.g. Hindi): `UNP4, UNP6, UFL`
  - Address: `AN, A2, AW2, AL2, NL`
  - Windowed numeric+nearby-word key: `A4W` (O(n), not the O(n²) all-pairs version)
  - Unordered informative-token-pair key: `AT2`
  - Rare-address-token key: `RAT20` (bucket-frequency-capped at 20)
  - `DEFAULT_FREQ_CAPS`: per-key-type max bucket size, applied against
    whatever index was actually built (10K dev or full), never a
    stale/hardcoded number.
- `indexer.py` — builds per-source (S2/S3) blocking indexes. Two backends:
  `MemoryIndex` (in-RAM dict-based, used for all runs so far) and
  `SqliteIndex` (disk-backed, implemented but **not yet exercised at scale —
  its build time / query latency has not been measured**).
- `candidates.py` — generates candidate pairs per S1 row across S2 and S3,
  applying the frequency caps.
- `features.py` — similarity features (name/address exact match, char
  similarity, Levenshtein, token Jaccard, char n-gram Jaccard, prefix/suffix
  similarity, address-number/country agreement, missing-field flags,
  blocking metadata). Levenshtein has a pure-Python fallback if no
  C-accelerated library is installed — **this fallback is the confirmed
  runtime bottleneck, see Section 4.**
- `labels.py` — ground-truth loading, S2/S3-aware.
- `pipeline.py` — single CLI (`--s1 --s2 --s3 --gt --tag --limit --backend`)
  that runs candidate generation + feature calculation + labeling +
  inline recall reporting in one streaming pass. Same code path for dev
  and (intended) full-scale runs.
- `create_train_10k.py` — builds an **ID-aligned** dev subset from the full
  data (samples S1 rows that actually have ground-truth matches, guarantees
  their true S2/S3 targets are included, then reservoir-samples the rest to
  a target size). This fixes an earlier version of this script that took
  "first N rows of every file independently," which produced a dev set
  where the S1/S2/S3/ground-truth subsets barely referenced each other.

**No ML matching model has been built yet.** No Kaggle `.ipynb` has been
created yet. No chunked/streaming full-scale rewrite has been built yet —
see Section 4.

---

## 2. WHAT HAS BEEN VALIDATED

- The pipeline was run end-to-end on a small **hand-built synthetic dataset**
  (fuzzy name/address variants + deliberate true negatives) and produced
  100% recall with correct positive/negative labels — confirms the
  candidate-generation → feature → label logic is wired correctly.
- The pipeline was run on a **10,000-S1 / 100,000-S2 / 100,000-S3
  ID-aligned dev subset** of the real training data. Results (verified by
  reading the actual output files, included in `output_reference_reports/`):

  - S2 candidate recall: **98.26%** (16,439 / 16,730 actual matches found)
  - S3 candidate recall: **97.68%** (17,603 / 18,022 actual matches found)
  - Combined candidate recall: **97.96%** (34,042 / 34,752)
  - Candidate volume: **2,541,867 total pairs** (254.19 candidates/S1 avg)
  - **Runtime: 1,565.7 seconds** for this 10K run (this is the critical
    unresolved number — see Section 4).

- Full training data (`train_source1/2/3.tsv`, `train_ground_truth.tsv`)
  was inspected directly (not sampled from memory):
  - S1: 2,206,821 rows, 0 duplicates, 0 missing fields, countries {US, India}
  - S2: 5,034,616 rows, 168,967 (3.4%) missing address, 764,608 (15.2%)
    non-ASCII names
  - S3: 5,285,603 rows, 175,916 (3.3%) missing address, 606,737 (11.5%)
    non-ASCII names
  - Ground truth: 123,247 (5.6%) singleton S1 entities, 7,638,365 total
    match references (3,693,619 → S2, 3,944,746 → S3), avg 3.666
    matches per matched S1, max 11.
  - Per the official problem statement PDF, the **test set adds a third
    country (France) not present in training** — confirmed from the PDF.

- **Not yet run against the full dataset at all.** No 50K/100K/500K/1M
  staged benchmark has been executed.

---

## 3. THE VERIFIED ~97.96% REFERENCE RECALL

Confirmed by directly reading `output_reference_reports/candidate_recall_at2_REAL.txt`
and `output_reference_reports/pipeline_report_train_10k.txt`, reproduced here verbatim:

```
S1 rows evaluated : 10,000
[S2] actual=16,730  found=16,439  recall=0.9826 (98.26%)  avg candidates/S1=124.38
[S3] actual=18,022  found=17,603  recall=0.9768 (97.68%)  avg candidates/S1=129.80
[COMBINED] actual=34,752 found=34,042 recall=0.9796 (97.96%) avg candidates/S1=254.19

Candidate pairs total : 2,541,867
Elapsed (s)            : 1565.7
```

This is a real, measured result on the current 17-key-type blocking
strategy against a correctly ID-aligned 10K/100K/100K subset — not a
projection or estimate.

---

## 4. WHAT REMAINS UNFINISHED

Ranked by how much it blocks progress:

1. **Runtime does not scale to the full dataset as-is.** 1,565.7s for
   2.54M candidate pairs ≈ 616 µs/pair, almost certainly dominated by the
   pure-Python Levenshtein fallback in `features.py` running twice per
   pair. Naive linear extrapolation to the full 2.2M S1 rows is
   **~95–100 hours single-threaded** — not viable on Kaggle or for
   iteration. **Fix required before any full-data run**: install/use
   `rapidfuzz` (MIT license, C-accelerated) or batch/vectorize feature
   computation (e.g. Polars/NumPy) instead of one Python call per pair.

2. **No chunked/streaming full-scale execution has been built.**
   `pipeline.py` streams row-by-row for writing but still targets a single
   monolithic `candidate_pairs_*.tsv` / `train_features_*.tsv` output. At
   the measured 10K byte-per-row rate (≈42 bytes/candidate row, ≈218
   bytes/feature row), naive extrapolation to full S1 projects
   **~23GB** for candidate_pairs and **~122GB** for train_features —
   both over a 20GB Kaggle disk budget, the features file by ~6x. This
   needs batched processing with immediate consumption/discard, not a
   single-file write. **Not yet implemented.**

3. **`SqliteIndex` backend exists in `indexer.py` but has never been run.**
   Its build time and lookup latency at S2/S3 full scale (5M+ records
   each) are unmeasured. This is the likely answer to full-scale indexing
   without blowing RAM, but it's unverified.

4. **No staged 50K → 100K → 500K → 1M benchmarks have been run.** It is
   unknown whether candidates/S1 and recall stay stable, grow, or shrink
   as the S2/S3 index grows from 100K to 5M+ (frequency-capped keys will
   start dropping more often as corpus grows; rare-token blockers like
   RAT20/A4W/AT2 with tight caps of 10–20 are the ones most at risk of
   losing effectiveness at scale). This is an open question, not an
   assumption either way.

5. **No per-blocking-key contribution table exists yet** (candidates /
   new true matches / recall contribution / time / memory per key type).
   Needed to decide which of the 17 key types to keep, tighten, or drop
   at full scale.

6. **No ML matching/ranking model has been built.** Candidate generation
   only answers "is the true match in the candidate set"; nothing yet
   scores/ranks candidates or produces `matching_results.tsv`.

7. **No Kaggle notebook exists.** Nothing has been packaged or tested in
   a Kaggle-compatible format.

8. **No test-set code path exists**, correctly, since no test dataset has
   been provided yet — this is intentional, not an omission.

---

## 5. HOW TO CONTINUE THIS WORK IN KAGGLE

1. Upload the full `dataset/train/` files (`train_source1.tsv`,
   `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`) as a
   Kaggle Dataset input (do not re-upload via this package — they were
   deliberately excluded here per size/scope).

2. Install `rapidfuzz` first (`pip install rapidfuzz`) so
   `features.py`'s fast-path Levenshtein import succeeds instead of
   falling back to pure Python — this is the single highest-leverage fix
   before running anything at scale.

3. Regenerate a correctly ID-aligned dev subset directly from the full
   data using the fixed script:

   ```
   python src/create_train_10k.py \
       --full-dir /kaggle/input/<dataset>/train \
       --out-dir /kaggle/working/train_10k \
       --n-s1 10000 --n-s2 100000 --n-s3 100000
   ```

4. Re-run `pipeline.py` on that subset to reconfirm the ~97.96% baseline
   in the Kaggle environment itself, and to measure whether the
   `rapidfuzz` swap fixes the runtime bottleneck, before doing anything
   else.

5. Build the staged benchmarks (50K/100K/500K/1M) described in Section 4,
   item 4, recording candidates/S1, recall, runtime, RAM and disk at each
   stage, before attempting the full 2.2M/5M/5.3M run.

6. Switch `pipeline.py`'s indexing to the `SqliteIndex` backend
   (`--backend sqlite`) once index size makes `MemoryIndex` risky, and
   measure it — don't assume it's fast enough untested.

7. Restructure output writing to chunked/batched Parquet with immediate
   downstream consumption instead of one monolithic TSV, once the disk
   projection in Section 4 item 2 is confirmed at a real intermediate
   scale (e.g. 500K).

8. Only after candidate generation is confirmed viable at full scale:
   build the ML matcher (LightGBM/XGBoost/CatBoost — all MIT/Apache-2.0
   compatible and ≤8B-parameter-constraint-compliant since they're not
   LLMs at all) on top of the labeled feature output.

9. Do not build or run any test-set inference path until the actual
   `dataset/test/` files are provided.

---

## 6. WHAT IS *NOT* IN THIS PACKAGE (deliberately)

- Raw train dataset files (S1/S2/S3/ground truth) — re-supply these as a
  Kaggle input dataset instead.
- The 106MB `candidate_pairs_train_10k.tsv` and 553MB
  `train_features_train_10k.tsv` full outputs — only the small text
  summary reports that prove the 97.96% result are included, in
  `output_reference_reports/`.
- `__pycache__`, obsolete/broken scripts (`candidate_generator.py`,
  `build_training_features.py` — the latter calls a function that doesn't
  exist anywhere in the codebase), and one-off debug/analysis scripts
  that aren't part of the working `pipeline.py` code path.
