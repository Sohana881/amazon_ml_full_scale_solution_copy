"""
candidates.py
============================================================
Frequency-aware, multi-source candidate generation.

Given a Source-1 row and one or more indexes (built by indexer.py,
one per source: S2, S3), this module returns candidate matches
while:

  - respecting country-aware blocking
  - using frequency-aware pruning
  - using the RAT20 rare-address-token blocker
  - never doing S1 x S2 or S1 x S3 all-pairs comparison
"""

from blocking import (
    make_name_keys,
    make_unicode_name_keys,
    make_address_keys,
    make_rare_address_token_keys,
    make_a4_keys,
    make_at2_keys,
    DEFAULT_FREQ_CAPS,
)

from indexer import compose_key


def find_candidates(s1_row, index, freq_caps=None):
    """
    Returns a dict:

        suffix -> {
            "key_types": set of key types that hit,
            "min_freq": smallest bucket size among the
                        hitting keys,
        }

    `index` is a single-source MemoryIndex/SqliteIndex
    (S2 or S3).
    """

    freq_caps = freq_caps or DEFAULT_FREQ_CAPS

    country = (
        s1_row.get("country", "")
        .strip()
    )

    name = s1_row.get(
        "business_name",
        ""
    )

    address = s1_row.get(
        "business_address",
        ""
    )

    # ========================================================
    # GENERATE ALL BLOCKING KEYS
    # ========================================================

    all_keys = (
        # Original name blockers
        make_name_keys(name)

        # Unicode-preserving name blockers
        + make_unicode_name_keys(name)

        # Original address blockers
        + make_address_keys(address)

        # NEW:
        # Rare Address Token blocker.
        #
        # The actual frequency limit is applied below using
        # freq_caps["RAT20"].
        + make_rare_address_token_keys(address)

        # Specialized address blockers
        + make_a4_keys(address)
        + make_at2_keys(address)
    )

    candidates = {}

    # ========================================================
    # LOOK UP EACH BLOCKING KEY
    # ========================================================

    for key_type, key_value in all_keys:

        # Get the configured frequency cap for this key type.
        #
        # RAT20 will normally resolve to:
        #
        #     20
        #
        # because DEFAULT_FREQ_CAPS contains:
        #
        #     "RAT20": 20
        #
        cap = freq_caps.get(
            key_type,
            DEFAULT_FREQ_CAPS.get(
                key_type,
                500
            )
        )

        # Country-aware composed key.
        composed = compose_key(
            country,
            key_value
        )

        # Retrieve posting list from the actual index.
        bucket = index.bucket(
            key_type,
            composed
        )

        bucket_size = len(bucket)

        # No candidates for this key.
        if bucket_size == 0:
            continue

        # ====================================================
        # FREQUENCY CAP
        # ====================================================
        #
        # If the bucket is too broad, ignore this blocking key.
        #
        # For RAT20:
        #
        #     bucket_size > 20
        #
        # means this address token is not rare enough.
        #
        if bucket_size > cap:
            continue

        # ====================================================
        # ADD CANDIDATES
        # ====================================================

        for suffix in bucket:

            entry = candidates.get(
                suffix
            )

            if entry is None:

                candidates[suffix] = {
                    "key_types": {
                        key_type
                    },

                    "min_freq": bucket_size,
                }

            else:

                # Candidate was already discovered through
                # another blocking key.
                entry["key_types"].add(
                    key_type
                )

                # Keep the most selective frequency.
                entry["min_freq"] = min(
                    entry["min_freq"],
                    bucket_size
                )

    return candidates


def generate_candidate_pairs_for_row(
    s1_row,
    indexes_by_source,
    freq_caps=None
):
    """
    indexes_by_source:

        {
            "S2": s2_index,
            "S3": s3_index
        }

    Returns a list of dictionaries:

        source1_entity_id
        candidate_entity_id
        source
        blocking_key_types
        blocking_num_keys
        blocking_min_freq
    """

    s1_id = s1_row.get(
        "entity_id",
        ""
    )

    rows = []

    # ========================================================
    # PROCESS EACH SOURCE
    # ========================================================

    for source, index in indexes_by_source.items():

        candidates = find_candidates(
            s1_row,
            index,
            freq_caps=freq_caps
        )

        # ====================================================
        # CONVERT CANDIDATES TO OUTPUT ROWS
        # ====================================================

        for suffix, info in candidates.items():

            candidate_id = (
                f"{source}-{suffix}"
            )

            rows.append(
                {
                    "source1_entity_id": s1_id,

                    "candidate_entity_id": candidate_id,

                    "source": source,

                    "blocking_key_types": "|".join(
                        sorted(
                            info["key_types"]
                        )
                    ),

                    "blocking_num_keys": len(
                        info["key_types"]
                    ),

                    "blocking_min_freq": info[
                        "min_freq"
                    ],
                }
            )

    return rows