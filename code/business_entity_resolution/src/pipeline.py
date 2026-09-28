"""
pipeline.py
============================================================
Main entry point for the Business Entity Resolution preprocessing
pipeline: normalization -> blocking -> candidate pairs -> similarity
features -> training labels -> ML-ready output.

THE SAME CODE PATH RUNS ON:
  - train_10k (development / debugging / fast validation)
  - the full 2.2M S1 / 5M+ S2 / 5.3M+ S3 dataset (production)

The only thing that changes between the two is which files you
point --s1/--s2/--s3/--gt at, and (optionally) --limit / --backend.
Nothing about frequency thresholds, blocking keys, or candidate
logic is hard-coded to 10,000 rows anywhere in this file.

------------------------------------------------------------
USAGE
------------------------------------------------------------

10K development run (has ground truth -> also gets recall + labels):

    python pipeline.py \
        --s1 ../dataset/train_10k/train_source1.tsv \
        --s2 ../dataset/train_10k/train_source2.tsv \
        --s3 ../dataset/train_10k/train_source3.tsv \
        --gt ../dataset/train_10k/train_ground_truth.tsv \
        --tag train_10k \
        --out-dir ../output

Full production TRAIN run (same command, full-data paths,
sqlite backend recommended for the 5M+ indexes):

    python pipeline.py \
        --s1 ../../dataset/train/train_source1.tsv \
        --s2 ../../dataset/train/train_source2.tsv \
        --s3 ../../dataset/train/train_source3.tsv \
        --gt ../../dataset/train/train_ground_truth.tsv \
        --tag train_full \
        --out-dir ../output \
        --backend sqlite --db-dir ../output/idx_train

Full production TEST run (no ground truth -> candidate pairs +
features only, no labels/recall):

    python pipeline.py \
        --s1 ../../dataset/test/test_source1.tsv \
        --s2 ../../dataset/test/test_source2.tsv \
        --s3 ../../dataset/test/test_source3.tsv \
        --tag test_full \
        --out-dir ../output \
        --backend sqlite --db-dir ../output/idx_test

--limit N restricts how many S1 rows are processed (for a quick
smoke test); omit it (or pass "none") to process every S1 row.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from blocking import DEFAULT_FREQ_CAPS
from data_loader import read_tsv
from indexer import build_index, split_entity_id
from candidates import generate_candidate_pairs_for_row
from features import calculate_features, FEATURE_COLUMNS
from labels import load_ground_truth, label_for


CANDIDATE_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "source",
    "blocking_key_types",
    "blocking_num_keys",
    "blocking_min_freq",
]

FEATURE_FILE_COLUMNS = CANDIDATE_COLUMNS + FEATURE_COLUMNS  # + ["label"] if labeled


def parse_limit(value):
    if value in (None, "none", "None", ""):
        return None
    return int(value)


def run(args):

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    freq_caps = DEFAULT_FREQ_CAPS
    if args.freq_caps:
        with open(args.freq_caps, "r", encoding="utf-8") as f:
            freq_caps = {**DEFAULT_FREQ_CAPS, **json.load(f)}

    s1_limit = parse_limit(args.limit)
    index_limit = parse_limit(args.index_limit)

    db_dir = Path(args.db_dir) if args.db_dir else out_dir / f"idx_{args.tag}"
    if args.backend == "sqlite":
        db_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    s2_index = build_index(
        args.s2,
        "S2",
        limit=index_limit,
        backend=args.backend,
        db_path=str(db_dir / "s2_index.sqlite") if args.backend == "sqlite" else None,
    )
    s3_index = build_index(
        args.s3,
        "S3",
        limit=index_limit,
        backend=args.backend,
        db_path=str(db_dir / "s3_index.sqlite") if args.backend == "sqlite" else None,
    )

    indexes_by_source = {"S2": s2_index, "S3": s3_index}

    ground_truth = load_ground_truth(args.gt) if args.gt else None
    has_labels = ground_truth is not None

    candidate_path = out_dir / f"candidate_pairs_{args.tag}.tsv"
    feature_tsv_path = out_dir / f"train_features_{args.tag}.tsv"
    feature_parquet_path = out_dir / f"train_features_{args.tag}.parquet"

    feature_columns = FEATURE_FILE_COLUMNS + (["label"] if has_labels else [])

    # recall accounting (only meaningful when ground truth is available)
    recall_stats = {
        source: {"actual": 0, "found": 0} for source in indexes_by_source
    }
    combined_recall = {"actual": 0, "found": 0}

    processed = 0
    total_pairs = 0
    positive = 0
    negative = 0

    parquet_rows = [] if args.parquet else None

    print("=" * 70)
    print(f"PIPELINE RUN: tag={args.tag}  s1_limit={s1_limit}  backend={args.backend}")
    print("=" * 70)

    with open(candidate_path, "w", encoding="utf-8", newline="") as cand_f, open(
        feature_tsv_path, "w", encoding="utf-8", newline=""
    ) as feat_f:

        cand_writer = csv.DictWriter(cand_f, fieldnames=CANDIDATE_COLUMNS, delimiter="\t")
        cand_writer.writeheader()

        feat_writer = csv.DictWriter(feat_f, fieldnames=feature_columns, delimiter="\t")
        feat_writer.writeheader()

        for s1_row in read_tsv(args.s1):

            if s1_limit is not None and processed >= s1_limit:
                break

            s1_id = s1_row.get("entity_id", "")
            pair_rows = generate_candidate_pairs_for_row(
                s1_row, indexes_by_source, freq_caps=freq_caps
            )

            gt_entry = (
                ground_truth.get(s1_id, {"S2": set(), "S3": set(), "all": set()})
                if has_labels
                else None
            )

            row_candidate_ids_by_source = {"S2": set(), "S3": set()}

            for pair in pair_rows:

                cand_writer.writerow(pair)
                total_pairs += 1

                source = pair["source"]
                candidate_id = pair["candidate_entity_id"]
                row_candidate_ids_by_source[source].add(candidate_id)

                _, suffix = split_entity_id(candidate_id)
                attrs = indexes_by_source[source].get_attrs(suffix)

                if attrs is None:
                    cand_name, cand_address, cand_country = "", "", ""
                else:
                    cand_name, cand_address, cand_country = attrs

                feats = calculate_features(
                    s1_row.get("business_name", ""),
                    s1_row.get("business_address", ""),
                    s1_row.get("country", ""),
                    cand_name,
                    cand_address,
                    cand_country,
                )

                out_row = {**pair, **feats}

                if has_labels:
                    label = label_for(s1_id, candidate_id, ground_truth)
                    out_row["label"] = label
                    if label:
                        positive += 1
                    else:
                        negative += 1

                feat_writer.writerow(out_row)

                if parquet_rows is not None:
                    parquet_rows.append(out_row)

            if has_labels:

                for source in indexes_by_source:

                    true_matches = gt_entry.get(source, set())
                    recall_stats[source]["actual"] += len(true_matches)
                    recall_stats[source]["found"] += len(
                        true_matches & row_candidate_ids_by_source[source]
                    )

                combined_recall["actual"] += len(gt_entry.get("all", set()))
                combined_recall["found"] += len(
                    gt_entry.get("all", set())
                    & (row_candidate_ids_by_source["S2"] | row_candidate_ids_by_source["S3"])
                )

            processed += 1

            if processed % 50_000 == 0:
                elapsed = time.time() - t0
                print(
                    f"[pipeline] S1 processed: {processed:,} | pairs: {total_pairs:,} "
                    f"| elapsed: {elapsed:.1f}s"
                )

    if args.parquet and parquet_rows:
        try:
            import pandas as pd

            pd.DataFrame(parquet_rows).to_parquet(feature_parquet_path, index=False)
            print(f"[pipeline] Parquet written to {feature_parquet_path}")
        except ImportError:
            print("[pipeline] pandas/pyarrow not available, skipped parquet output.")

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"S1 rows processed     : {processed:,}")
    print(f"Candidate pairs total : {total_pairs:,}")
    print(f"Avg candidates / S1   : {(total_pairs/processed if processed else 0):.2f}")
    if has_labels:
        print(f"Positive labels       : {positive:,}")
        print(f"Negative labels       : {negative:,}")
    print(f"Elapsed               : {elapsed:.1f}s")
    print(f"\nCandidate pairs saved : {candidate_path}")
    print(f"Features saved        : {feature_tsv_path}")

    report_lines = [
        "=" * 60,
        f"PIPELINE REPORT (tag={args.tag})",
        "=" * 60,
        f"S1 rows processed     : {processed:,}",
        f"Candidate pairs total : {total_pairs:,}",
        f"Avg candidates / S1   : {(total_pairs/processed if processed else 0):.2f}",
        f"Elapsed (s)           : {elapsed:.1f}",
    ]

    if has_labels:

        report_lines.append(f"Positive labels       : {positive:,}")
        report_lines.append(f"Negative labels       : {negative:,}")
        report_lines.append("")

        for source, s in recall_stats.items():
            recall = (s["found"] / s["actual"]) if s["actual"] else 0.0
            report_lines.append(f"[{source}] actual={s['actual']:,} found={s['found']:,} "
                                 f"recall={recall:.4f} ({recall*100:.2f}%)")

        c_recall = (
            (combined_recall["found"] / combined_recall["actual"])
            if combined_recall["actual"]
            else 0.0
        )
        report_lines.append(
            f"[COMBINED] actual={combined_recall['actual']:,} "
            f"found={combined_recall['found']:,} recall={c_recall:.4f} "
            f"({c_recall*100:.2f}%)"
        )

    report = "\n".join(report_lines)
    report_path = out_dir / f"pipeline_report_{args.tag}.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved           : {report_path}")
    print(report)


def build_arg_parser():

    p = argparse.ArgumentParser(description="Business entity resolution preprocessing pipeline")

    p.add_argument("--s1", required=True, help="Path to source1 (S1) tsv")
    p.add_argument("--s2", required=True, help="Path to source2 (S2) tsv")
    p.add_argument("--s3", required=True, help="Path to source3 (S3) tsv")
    p.add_argument("--gt", default=None, help="Path to ground truth tsv (train only; omit for test)")

    p.add_argument("--tag", required=True, help="Short label used in output filenames, e.g. train_10k")
    p.add_argument("--out-dir", default="../output", help="Output directory")

    p.add_argument("--limit", default=None, help="Max S1 rows to process. 'none' = all rows")
    p.add_argument("--index-limit", default=None, help="Max rows to index from S2/S3. 'none' = all rows")

    p.add_argument("--backend", choices=["memory", "sqlite"], default="memory",
                    help="Index backend. Use sqlite for full 5M+ production runs if RAM is limited.")
    p.add_argument("--db-dir", default=None, help="Directory for sqlite index files (backend=sqlite)")

    p.add_argument("--freq-caps", default=None, help="Optional JSON file overriding DEFAULT_FREQ_CAPS")
    p.add_argument("--parquet", action="store_true", help="Also write a .parquet feature file")

    return p


if __name__ == "__main__":

    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)
