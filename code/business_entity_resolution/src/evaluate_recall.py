"""
evaluate_recall.py
============================================================
Candidate-recall evaluation for S2, S3 and combined coverage.

Supports testing different RAT20 frequency caps without
rebuilding the indexes.

Examples:

    RAT disabled:
        --rat-cap 0

    RAT10:
        --rat-cap 10

    RAT20:
        --rat-cap 20
"""

from data_loader import read_tsv
from labels import load_ground_truth
from candidates import find_candidates
from indexer import build_index
from blocking import DEFAULT_FREQ_CAPS

from pathlib import Path
import argparse


def evaluate(
    s1_path,
    indexes_by_source,
    gt_path,
    limit=None,
    freq_caps=None,
):
    """
    Evaluate candidate recall for one particular frequency-cap
    configuration.
    """

    ground_truth = load_ground_truth(gt_path)

    stats = {
        source: {
            "actual": 0,
            "found": 0,
            "candidates": 0,
        }
        for source in indexes_by_source
    }

    combined = {
        "actual": 0,
        "found": 0,
    }

    total_candidates = 0
    processed = 0

    for row in read_tsv(s1_path):

        if limit is not None and processed >= limit:
            break

        s1_id = row.get(
            "entity_id",
            ""
        )

        gt_entry = ground_truth.get(
            s1_id,
            {
                "S2": set(),
                "S3": set(),
                "all": set(),
            },
        )

        row_candidate_ids = set()

        # ====================================================
        # S2 / S3
        # ====================================================

        for source, index in indexes_by_source.items():

            candidates = find_candidates(
                row,
                index,
                freq_caps=freq_caps,
            )

            candidate_ids = {
                f"{source}-{suffix}"
                for suffix in candidates
            }

            stats[source]["candidates"] += len(
                candidate_ids
            )

            total_candidates += len(
                candidate_ids
            )

            row_candidate_ids |= candidate_ids

            # ------------------------------------------------
            # Ground truth
            # ------------------------------------------------

            true_matches = gt_entry.get(
                source,
                set(),
            )

            stats[source]["actual"] += len(
                true_matches
            )

            for match_id in true_matches:

                if match_id in candidate_ids:
                    stats[source]["found"] += 1

        # ====================================================
        # COMBINED
        # ====================================================

        combined["actual"] += len(
            gt_entry.get(
                "all",
                set(),
            )
        )

        combined["found"] += len(
            gt_entry.get(
                "all",
                set(),
            )
            & row_candidate_ids
        )

        processed += 1

        if processed % 50_000 == 0:
            print(
                f"[recall] processed "
                f"{processed:,} S1 rows..."
            )

    # ========================================================
    # REPORT
    # ========================================================

    def pct(found, actual):

        return (
            found / actual
            if actual
            else 0.0
        )

    report_lines = []

    report_lines.append(
        "=" * 60
    )

    report_lines.append(
        "CANDIDATE RECALL REPORT"
    )

    report_lines.append(
        "=" * 60
    )

    report_lines.append(
        f"S1 rows evaluated : {processed:,}"
    )

    report_lines.append("")

    for source, s in stats.items():

        recall = pct(
            s["found"],
            s["actual"],
        )

        avg_cand = (
            s["candidates"] / processed
            if processed
            else 0
        )

        report_lines.append(
            f"[{source}]"
        )

        report_lines.append(
            f"  actual matches   : "
            f"{s['actual']:,}"
        )

        report_lines.append(
            f"  matches found    : "
            f"{s['found']:,}"
        )

        report_lines.append(
            f"  recall           : "
            f"{recall:.4f} "
            f"({recall * 100:.2f}%)"
        )

        report_lines.append(
            f"  avg candidates/S1: "
            f"{avg_cand:.2f}"
        )

        report_lines.append("")

    combined_recall = pct(
        combined["found"],
        combined["actual"],
    )

    avg_total_cand = (
        total_candidates / processed
        if processed
        else 0
    )

    report_lines.append(
        "[COMBINED S2+S3]"
    )

    report_lines.append(
        f"  actual matches   : "
        f"{combined['actual']:,}"
    )

    report_lines.append(
        f"  matches found    : "
        f"{combined['found']:,}"
    )

    report_lines.append(
        f"  recall           : "
        f"{combined_recall:.4f} "
        f"({combined_recall * 100:.2f}%)"
    )

    report_lines.append(
        f"  avg candidates/S1: "
        f"{avg_total_cand:.2f}"
    )

    report_lines.append(
        "=" * 60
    )

    return "\n".join(
        report_lines
    )


def make_freq_caps(rat_cap):
    """
    Start with the normal blocking caps and override RAT20.

    rat_cap = 0
        disables RAT20

    rat_cap = 10
        allows RAT20 buckets of size <= 10

    rat_cap = 20
        allows RAT20 buckets of size <= 20
    """

    freq_caps = dict(
        DEFAULT_FREQ_CAPS
    )

    freq_caps["RAT20"] = rat_cap

    return freq_caps


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--s1",
        required=True,
    )

    parser.add_argument(
        "--s2",
        required=True,
    )

    parser.add_argument(
        "--s3",
        required=True,
    )

    parser.add_argument(
        "--gt",
        required=True,
    )

    parser.add_argument(
        "--limit",
        default=None,
    )

    parser.add_argument(
        "--index-limit",
        default=None,
    )

    parser.add_argument(
        "--rat-cap",
        type=int,
        default=20,
        help=(
            "Maximum RAT20 bucket size. "
            "Use 0 to disable RAT20, "
            "10 for RAT10, "
            "20 for RAT20."
        ),
    )

    parser.add_argument(
        "--out",
        default="../output/candidate_recall_report.txt",
    )

    args = parser.parse_args()

    # ========================================================
    # PARSE LIMITS
    # ========================================================

    limit = (
        None
        if args.limit
        in (None, "none", "None")
        else int(args.limit)
    )

    index_limit = (
        None
        if args.index_limit
        in (None, "none", "None")
        else int(args.index_limit)
    )

    # ========================================================
    # BUILD INDEXES ONCE
    # ========================================================

    s2_index = build_index(
        args.s2,
        "S2",
        limit=index_limit,
    )

    s3_index = build_index(
        args.s3,
        "S3",
        limit=index_limit,
    )

    indexes = {
        "S2": s2_index,
        "S3": s3_index,
    }

    # ========================================================
    # TEST SELECTED RAT CAP
    # ========================================================

    freq_caps = make_freq_caps(
        args.rat_cap
    )

    print()
    print(
        "============================================================"
    )

    if args.rat_cap == 0:

        print(
            "TESTING: RAT20 DISABLED"
        )

    else:

        print(
            f"TESTING: RAT20 BUCKET CAP = "
            f"{args.rat_cap}"
        )

    print(
        "============================================================"
    )

    report = evaluate(
        args.s1,
        indexes,
        args.gt,
        limit=limit,
        freq_caps=freq_caps,
    )

    print(report)

    # ========================================================
    # SAVE
    # ========================================================

    out_path = Path(
        args.out
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print(
        f"Saved report to: {out_path}"
    )