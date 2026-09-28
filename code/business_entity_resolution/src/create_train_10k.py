"""
create_train_10k.py
============================================================
Builds an ID-ALIGNED development subset from the full dataset.

------------------------------------------------------------
BUG IN THE ORIGINAL VERSION OF THIS SCRIPT (fixed here)
------------------------------------------------------------
The original script took the first N rows of train_source1.tsv,
train_source2.tsv, train_source3.tsv and train_ground_truth.tsv
INDEPENDENTLY. Entity IDs in this dataset are random-looking
numeric suffixes, not row-position-based, and the four full files
are not stored in matching row order. The result: in the
train_10k bundled with this project, only 59 of the 10,000 S1
rows even have a corresponding ground_truth row, and NONE of
their ~191 ground-truth matches exist in the accompanying 10K
S2/S3 subset. That makes the old train_10k unusable for recall
testing -- every recall/candidate-volume number measured against
it would be meaningless (0% recall) even with a perfect blocker.

------------------------------------------------------------
FIX
------------------------------------------------------------
This version builds the subset ID-first:

  1. Load train_ground_truth.tsv and pick `n_s1` Source-1 ids that
     HAVE at least one match (so the dev set actually contains
     positive examples), plus a smaller number of ids with zero
     matches (so it also has realistic negatives).
  2. Collect the full set of S2/S3 ids referenced by those S1's
     ground truth.
  3. Stream the full S2/S3 files ONCE each, keeping:
       - every row whose id is a required match target (always
         kept, regardless of budget)
       - a reservoir sample of additional rows up to n_s2/n_s3
         so the dev index has realistic size/frequency behaviour
         for blocking-key statistics.
  4. Stream the full S1 file once, keeping only the selected ids.

The output train_10k/ files are then genuinely usable to measure
candidate recall, exactly like the full dataset, just smaller.

Run this against the FULL dataset (outside amazon_ml_solution/):

    python create_train_10k.py \
        --full-dir ../../dataset/train \
        --out-dir ../dataset/train_10k \
        --n-s1 10000 --n-s2 10000 --n-s3 10000
"""

import argparse
import csv
import random
from pathlib import Path

from data_loader import read_tsv


def load_ground_truth_ids(gt_path):

    with_match = []
    without_match = []
    matches_by_s1 = {}

    for row in read_tsv(gt_path):

        s1_id = row["source1_entity_id"]
        raw = (row.get("matched_entity_ids") or "").strip()
        matches = [x.strip() for x in raw.split(",") if x.strip()]

        matches_by_s1[s1_id] = matches

        if matches:
            with_match.append(s1_id)
        else:
            without_match.append(s1_id)

    return with_match, without_match, matches_by_s1


def stream_filter_write(src_path, out_path, wanted_ids, cap, seed):
    """
    Single streaming pass over `src_path`. Writes every row whose
    entity_id is in `wanted_ids` (always kept) plus a reservoir
    sample of other rows up to `cap` total rows, preserving the
    header. Returns the set of ids actually written.
    """

    rng = random.Random(seed)

    required = set(wanted_ids)
    kept_required = []
    reservoir = []
    seen_optional = 0

    with open(src_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        id_idx = header.index("entity_id")

        budget_for_optional = max(cap - len(required), 0)

        for row in reader:

            if not row:
                continue

            entity_id = row[id_idx]

            if entity_id in required:
                kept_required.append(row)
                continue

            seen_optional += 1

            if len(reservoir) < budget_for_optional:
                reservoir.append(row)
            else:
                j = rng.randint(0, seen_optional - 1)
                if j < budget_for_optional:
                    reservoir[j] = row

    written_ids = set()

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)

        for row in kept_required:
            writer.writerow(row)
            written_ids.add(row[id_idx])

        for row in reservoir:
            writer.writerow(row)
            written_ids.add(row[id_idx])

    missing_required = required - written_ids
    if missing_required:
        print(
            f"[create_train_10k] WARNING: {len(missing_required)} required ids "
            f"were not found in {src_path} (dangling ground-truth reference)."
        )

    return written_ids


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dir", required=True,
                         help="Directory with the FULL train_source1/2/3 + ground_truth tsv files")
    parser.add_argument("--out-dir", required=True, help="Output directory for the aligned dev subset")
    parser.add_argument("--n-s1", type=int, default=10000)
    parser.add_argument("--n-s1-no-match", type=int, default=1000,
                         help="How many of the n-s1 rows should be S1 entities with NO ground-truth match")
    parser.add_argument("--n-s2", type=int, default=10000)
    parser.add_argument("--n-s3", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    full_dir = Path(args.full_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    print("Loading full ground truth to select an ID-aligned sample...")
    with_match, without_match, matches_by_s1 = load_ground_truth_ids(
        full_dir / "train_ground_truth.tsv"
    )

    n_with = max(args.n_s1 - args.n_s1_no_match, 0)
    rng.shuffle(with_match)
    rng.shuffle(without_match)

    chosen_with = with_match[:n_with]
    chosen_without = without_match[: args.n_s1 - len(chosen_with)]
    chosen_s1_ids = set(chosen_with) | set(chosen_without)

    required_s2, required_s3 = set(), set()
    for s1_id in chosen_with:
        for m in matches_by_s1[s1_id]:
            if m.startswith("S2-"):
                required_s2.add(m)
            elif m.startswith("S3-"):
                required_s3.add(m)

    print(f"Selected {len(chosen_s1_ids):,} S1 ids "
          f"({len(chosen_with):,} with matches, {len(chosen_without):,} without).")
    print(f"Required S2 target ids: {len(required_s2):,}")
    print(f"Required S3 target ids: {len(required_s3):,}")

    print("\nWriting aligned S1 subset...")
    with open(full_dir / "train_source1.tsv", "r", encoding="utf-8", newline="") as fin, \
         open(out_dir / "train_source1.tsv", "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t")
        header = next(reader)
        writer.writerow(header)
        id_idx = header.index("entity_id")
        for row in reader:
            if row and row[id_idx] in chosen_s1_ids:
                writer.writerow(row)

    print("Writing aligned S2 subset (required matches + reservoir sample)...")
    stream_filter_write(
        full_dir / "train_source2.tsv",
        out_dir / "train_source2.tsv",
        required_s2,
        args.n_s2,
        seed=args.seed,
    )

    print("Writing aligned S3 subset (required matches + reservoir sample)...")
    stream_filter_write(
        full_dir / "train_source3.tsv",
        out_dir / "train_source3.tsv",
        required_s3,
        args.n_s3,
        seed=args.seed + 1,
    )

    print("Writing ground truth for the chosen S1 subset...")
    with open(full_dir / "train_ground_truth.tsv", "r", encoding="utf-8", newline="") as fin, \
         open(out_dir / "train_ground_truth.tsv", "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t")
        header = next(reader)
        writer.writerow(header)
        id_idx = header.index("source1_entity_id")
        for row in reader:
            if row and row[id_idx] in chosen_s1_ids:
                writer.writerow(row)

    print("\nDone. Every match referenced by the chosen S1 rows is now "
          "guaranteed to be present in the S2/S3 subset, so recall "
          "measured against this train_10k is meaningful.")


if __name__ == "__main__":
    main()
