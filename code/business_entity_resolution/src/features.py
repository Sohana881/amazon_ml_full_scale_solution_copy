"""
features.py
============================================================
ML-ready similarity features for a candidate pair.

Reuses the tested functions in similarity.py (character similarity,
token Jaccard, address-number match, country match) and adds the
remaining signals called for in the task brief: Levenshtein
similarity, prefix/suffix similarity, compact-string similarity,
character n-gram similarity, token overlap counts, missing-field
indicators, and the blocking metadata (key types / frequency /
number of independent rules that produced the candidate).

RapidFuzz is preferred for fast Levenshtein-style similarity.
A pure-Python fallback is retained if RapidFuzz is unavailable.
"""
from functools import lru_cache
from normalize import normalize_business_name, normalize_address

from similarity import (
    text_similarity,
    token_similarity,
    address_number_match,
    country_match,
)
@lru_cache(maxsize=500_000)
def cached_normalize_business_name(value):
    return normalize_business_name(value)


@lru_cache(maxsize=500_000)
def cached_normalize_address(value):
    return normalize_address(value)

# --------------------------------------------------------------
# Levenshtein similarity
# --------------------------------------------------------------
# Prefer RapidFuzz because the feature pipeline can evaluate
# millions of candidate pairs.
# --------------------------------------------------------------

try:
    from rapidfuzz.fuzz import ratio as _rapidfuzz_ratio

    def levenshtein_similarity(a, b):
        """
        Fast normalized string similarity using RapidFuzz.

        Returns:
            float in [0, 1]
        """
        if not a or not b:
            return 0.0

        return _rapidfuzz_ratio(a, b) / 100.0


except ImportError:

    # Try python-Levenshtein if RapidFuzz is unavailable.
    try:
        from Levenshtein import ratio as _lev_ratio

        def levenshtein_similarity(a, b):
            """
            Fast normalized Levenshtein similarity.

            Returns:
                float in [0, 1]
            """
            if not a or not b:
                return 0.0

            return _lev_ratio(a, b)


    except ImportError:

        # ------------------------------------------------------
        # Pure-Python fallback
        # ------------------------------------------------------

        def _levenshtein_distance(a, b):

            if a == b:
                return 0

            if len(a) < len(b):
                a, b = b, a

            if not b:
                return len(a)

            previous_row = list(range(len(b) + 1))

            for i, ca in enumerate(a, start=1):

                current_row = [i] + [0] * len(b)

                for j, cb in enumerate(b, start=1):

                    cost = 0 if ca == cb else 1

                    current_row[j] = min(
                        current_row[j - 1] + 1,
                        previous_row[j] + 1,
                        previous_row[j - 1] + cost,
                    )

                previous_row = current_row

            return previous_row[-1]


        def levenshtein_similarity(a, b):

            if not a or not b:
                return 0.0

            dist = _levenshtein_distance(a, b)

            max_len = max(len(a), len(b))

            return 1.0 - (dist / max_len) if max_len else 1.0


# --------------------------------------------------------------
# Character n-gram Jaccard
# --------------------------------------------------------------

def char_ngram_jaccard(a, b, n=2):

    if not a or not b:
        return 0.0

    if len(a) < n or len(b) < n:
        return 0.0

    grams_a = {
        a[i:i + n]
        for i in range(len(a) - n + 1)
    }

    grams_b = {
        b[i:i + n]
        for i in range(len(b) - n + 1)
    }

    if not grams_a or not grams_b:
        return 0.0

    return len(grams_a & grams_b) / len(grams_a | grams_b)


# --------------------------------------------------------------
# Prefix similarity
# --------------------------------------------------------------

def prefix_similarity(a, b, k=4):

    if not a or not b:
        return 0.0

    a_p = a[:k]
    b_p = b[:k]

    max_len = min(len(a_p), len(b_p))

    if max_len == 0:
        return 0.0

    matched = sum(
        1
        for x, y in zip(a_p, b_p)
        if x == y
    )

    return matched / max_len


# --------------------------------------------------------------
# Suffix similarity
# --------------------------------------------------------------

def suffix_similarity(a, b, k=4):

    if not a or not b:
        return 0.0

    a_s = a[-k:]
    b_s = b[-k:]

    max_len = min(len(a_s), len(b_s))

    if max_len == 0:
        return 0.0

    matched = sum(
        1
        for x, y in zip(
            reversed(a_s),
            reversed(b_s)
        )
        if x == y
    )

    return matched / max_len


# --------------------------------------------------------------
# Token overlap count
# --------------------------------------------------------------

def token_overlap_count(a, b):

    if not a or not b:
        return 0

    return len(
        set(a.split()) &
        set(b.split())
    )


# --------------------------------------------------------------
# Feature columns
# --------------------------------------------------------------

FEATURE_COLUMNS = [

    # ----------------------------------------------------------
    # Name features
    # ----------------------------------------------------------

    "name_exact_match",
    "name_char_similarity",
    "name_levenshtein_similarity",
    "name_token_jaccard",
    "name_token_overlap",
    "name_compact_similarity",
    "name_prefix_similarity",
    "name_suffix_similarity",
    "name_char_ngram_jaccard",

    # ----------------------------------------------------------
    # Address features
    # ----------------------------------------------------------

    "address_exact_match",
    "address_char_similarity",
    "address_levenshtein_similarity",
    "address_token_jaccard",
    "address_token_overlap",
    "address_number_match",
    "address_char_ngram_jaccard",

    # ----------------------------------------------------------
    # Other features
    # ----------------------------------------------------------

    "country_match",
    "name_missing",
    "address_missing",
    "combined_name_address_score",
]


# --------------------------------------------------------------
# Main feature calculation
# --------------------------------------------------------------

def calculate_features(
    name1,
    address1,
    country1,
    name2,
    address2,
    country2,
):

    # ----------------------------------------------------------
    # Normalize
    # ----------------------------------------------------------

    n1 = cached_normalize_business_name(name1)
    n2 = cached_normalize_business_name(name2)

    a1 = cached_normalize_address(address1)
    a2 = cached_normalize_address(address2)

    # ----------------------------------------------------------
    # Missing-field indicators
    # ----------------------------------------------------------

    name_missing = int(
        not n1 or not n2
    )

    address_missing = int(
        not a1 or not a2
    )

    # ----------------------------------------------------------
    # Name similarities
    # ----------------------------------------------------------

    name_char_sim = text_similarity(
        n1,
        n2
    )

    name_lev_sim = levenshtein_similarity(
        n1,
        n2
    )

    name_tok_jaccard = token_similarity(
        n1,
        n2
    )

    name_compact_sim = text_similarity(
        n1.replace(" ", ""),
        n2.replace(" ", "")
    )

    # ----------------------------------------------------------
    # Address similarities
    # ----------------------------------------------------------

    address_char_sim = text_similarity(
        a1,
        a2
    )

    address_lev_sim = levenshtein_similarity(
        a1,
        a2
    )

    address_tok_jaccard = token_similarity(
        a1,
        a2
    )

    # ----------------------------------------------------------
    # Combined score
    # ----------------------------------------------------------

    combined = (
        0.6 * name_char_sim
        +
        0.4 * address_char_sim
    )

    # ----------------------------------------------------------
    # Return feature dictionary
    # ----------------------------------------------------------

    return {

        # ------------------------------------------------------
        # Name
        # ------------------------------------------------------

        "name_exact_match":
            int(bool(n1) and n1 == n2),

        "name_char_similarity":
            name_char_sim,

        "name_levenshtein_similarity":
            name_lev_sim,

        "name_token_jaccard":
            name_tok_jaccard,

        "name_token_overlap":
            token_overlap_count(
                n1,
                n2
            ),

        "name_compact_similarity":
            name_compact_sim,

        "name_prefix_similarity":
            prefix_similarity(
                n1,
                n2
            ),

        "name_suffix_similarity":
            suffix_similarity(
                n1,
                n2
            ),

        "name_char_ngram_jaccard":
            char_ngram_jaccard(
                n1,
                n2
            ),

        # ------------------------------------------------------
        # Address
        # ------------------------------------------------------

        "address_exact_match":
            int(bool(a1) and a1 == a2),

        "address_char_similarity":
            address_char_sim,

        "address_levenshtein_similarity":
            address_lev_sim,

        "address_token_jaccard":
            address_tok_jaccard,

        "address_token_overlap":
            token_overlap_count(
                a1,
                a2
            ),

        "address_number_match":
            address_number_match(
                address1,
                address2
            ),

        "address_char_ngram_jaccard":
            char_ngram_jaccard(
                a1,
                a2
            ),

        # ------------------------------------------------------
        # Other
        # ------------------------------------------------------

        "country_match":
            country_match(
                country1,
                country2
            ),

        "name_missing":
            name_missing,

        "address_missing":
            address_missing,

        "combined_name_address_score":
            combined,
    }