"""
labels.py
============================================================
Ground-truth loading and labeling.

train_ground_truth.tsv columns are:

    source1_entity_id, matched_entity_ids

NOTE the first column is "source1_entity_id", NOT "entity_id".
matched_entity_ids can contain a mix of S2-* and S3-* ids.
"""

import csv


def load_ground_truth(path):
    """
    Returns dict: source1_entity_id -> {
        "S2": set of matched S2 ids,
        "S3": set of matched S3 ids,
        "all": set of every matched id (any source),
    }
    """

    ground_truth = {}

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            s1_id = row["source1_entity_id"]
            raw = (row.get("matched_entity_ids") or "").strip()

            if raw:
                all_ids = {x.strip() for x in raw.split(",") if x.strip()}
            else:
                all_ids = set()

            ground_truth[s1_id] = {
                "S2": {x for x in all_ids if x.startswith("S2-")},
                "S3": {x for x in all_ids if x.startswith("S3-")},
                "all": all_ids,
            }

    return ground_truth


def label_for(s1_id, candidate_id, ground_truth):

    entry = ground_truth.get(s1_id)

    if not entry:
        return 0

    return int(candidate_id in entry["all"])
