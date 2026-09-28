"""
blocking.py
============================================================
Blocking-key generation for Business Entity Resolution.

This module is SOURCE-INDEPENDENT: the same functions are used
for Source 1, Source 2 and Source 3, and for both the 10K
development data and the full 5M+ record production data.

Frequency thresholds are NOT computed here -- they are computed
by indexer.py against whatever data was actually indexed.

Key types produced:

Name keys:
    NP6, NP8, NS6, FL, FL3, F2, FML
    UNP4, UNP6, UFL

Address keys:
    AN, A2, AW2, AL2, NL
    A4W, AT2
    RAT20 = Rare Address Token, frequency capped at 20
"""


import re
import unicodedata

from normalize import normalize_business_name, normalize_address


# ============================================================
# UNICODE NAME BLOCKING
# ============================================================

def make_unicode_name_keys(name):
    """
    Unicode-preserving name keys.

    Keeps Hindi, Kannada, Gujarati, etc. instead of deleting
    non-ASCII characters.
    """

    if not name:
        return []

    n = unicodedata.normalize("NFKC", str(name)).casefold()
    n = " ".join(n.split())

    if not n:
        return []

    keys = []

    compact = "".join(ch for ch in n if not ch.isspace())

    if len(compact) >= 4:
        keys.append(("UNP4", compact[:4]))

    if len(compact) >= 6:
        keys.append(("UNP6", compact[:6]))

    words = n.split()

    if len(words) >= 2:
        first = words[0]
        last = words[-1]

        if len(first) >= 2 and len(last) >= 2:
            keys.append(("UFL", first[:3] + "|" + last[:3]))

    return keys


# ============================================================
# NAME BLOCKING
# ============================================================

def make_name_keys(name):
    """
    Generate selective name-blocking keys.

    Generic single-word keys are avoided because they create
    enormous candidate buckets.
    """

    n = normalize_business_name(name)

    if not n:
        return []

    keys = []

    compact = n.replace(" ", "")

    if len(compact) >= 6:
        keys.append(("NP6", compact[:6]))

    if len(compact) >= 8:
        keys.append(("NP8", compact[:8]))

    if len(compact) >= 6:
        keys.append(("NS6", compact[-6:]))

    words = n.split()

    if len(words) >= 2:
        first, last = words[0], words[-1]

        if len(first) >= 3 and len(last) >= 3:
            keys.append(("FL", first[:4] + "|" + last[:4]))
            keys.append(("FL3", first[:3] + "|" + last[:5]))

        w1, w2 = words[0], words[1]

        if len(w1) >= 3 and len(w2) >= 3:
            keys.append(("F2", w1[:5] + "|" + w2[:5]))

    if len(words) >= 3:
        first, middle, last = words[0], words[-2], words[-1]

        if len(first) >= 3 and len(last) >= 3:
            keys.append(
                ("FML", first[:3] + "|" + middle[:3] + "|" + last[:3])
            )

    return keys


# ============================================================
# ADDRESS BLOCKING
# ============================================================

def make_address_keys(address):
    """
    Generate selective address-blocking keys.

    We deliberately do NOT use a bare number as a key.
    """

    a = normalize_address(address)

    if not a:
        return []

    keys = []

    words = a.split()

    if not words:
        return []

    numbers = [w for w in words if w.isdigit()]
    non_numeric = [w for w in words if not w.isdigit()]

    if numbers:
        number = numbers[0]

        for word in words:
            if not word.isdigit() and len(word) >= 3:
                keys.append(("AN", number + "|" + word[:6]))
                break

        if len(non_numeric) >= 2:
            keys.append(
                (
                    "A2",
                    number
                    + "|"
                    + non_numeric[0][:4]
                    + "|"
                    + non_numeric[1][:4],
                )
            )

    if len(non_numeric) >= 2:
        w1, w2 = non_numeric[0], non_numeric[1]

        if len(w1) >= 3 and len(w2) >= 3:
            keys.append(("AW2", w1[:5] + "|" + w2[:5]))

        w1, w2 = non_numeric[-2], non_numeric[-1]

        if len(w1) >= 3 and len(w2) >= 3:
            keys.append(("AL2", w1[:5] + "|" + w2[:5]))

    if numbers and non_numeric:
        number = numbers[0]
        last_word = non_numeric[-1]

        if len(last_word) >= 3:
            keys.append(("NL", number + "|" + last_word[:6]))

    return keys


# ============================================================
# RARE ADDRESS TOKEN BLOCKING
# ============================================================

# Very common address tokens are weak blocking signals.
# They are excluded completely.
GENERIC_ADDRESS_WORDS = {
    "street", "st", "road", "rd", "avenue", "ave",
    "lane", "ln", "drive", "dr", "boulevard", "blvd",
    "highway", "hwy", "near", "opposite", "opp",
    "building", "bldg", "floor", "fl", "block",
    "district", "city", "state", "india",
    "private", "limited", "ltd", "pvt",
}


def make_rare_address_token_keys(address):
    """
    Generate individual informative address-token keys.

    These keys are intentionally broad at generation time.
    indexer.py applies the frequency cap against the actual
    dataset being indexed.

    RAT20 therefore means:

        Rare Address Token
        maximum bucket frequency = 20

    This is useful when the business name is corrupted or written
    in another script but some distinctive address token survives.
    """

    a = normalize_address(address)

    if not a:
        return []

    tokens = a.split()

    if not tokens:
        return []

    keys = []
    seen = set()

    for tok in tokens:

        # Numbers alone are not distinctive enough.
        if tok.isdigit():
            continue

        # Ignore extremely short tokens.
        if len(tok) < 4:
            continue

        # Ignore generic address words.
        if tok in GENERIC_ADDRESS_WORDS:
            continue

        # Normalize token representation.
        token = tok[:12]

        if not token:
            continue

        if token in seen:
            continue

        seen.add(token)

        keys.append(("RAT20", token))

    return keys


# ============================================================
# AT2 BLOCKING
# ============================================================

def make_at2_keys(address):
    """
    Generate informative address-token-pair keys.

    Unlike AW2/AL2, this does not depend on token position.
    """

    a = normalize_address(address)

    if not a:
        return []

    tokens = a.split()

    informative = []

    for tok in tokens:

        if tok.isdigit():
            continue

        if len(tok) < 4:
            continue

        if tok in GENERIC_ADDRESS_WORDS:
            continue

        informative.append(tok)

    if len(informative) < 2:
        return []

    # Keep at most 6 informative tokens.
    informative = sorted(
        informative,
        key=lambda x: (-len(x), x)
    )[:6]

    keys = []
    seen = set()

    for i in range(len(informative) - 1):

        w1 = informative[i][:8]
        w2 = informative[i + 1][:8]

        if w1 == w2:
            continue

        a1, a2 = sorted((w1, w2))

        key_value = a1 + "|" + a2

        if key_value not in seen:
            seen.add(key_value)
            keys.append(("AT2", key_value))

    return keys


# ============================================================
# A4 BLOCKING
# ============================================================

A4_WINDOW = 3


def make_a4_keys(address, window=A4_WINDOW):
    """
    Numeric address component + nearby normalized address words.

    Windowed implementation keeps key generation O(n).
    """

    a = normalize_address(address)

    if not a:
        return []

    tokens = a.split()

    number_positions = [
        i for i, t in enumerate(tokens)
        if t.isdigit()
    ]

    if not number_positions:
        return []

    keys = []
    seen = set()

    for pos in number_positions:

        number = tokens[pos]

        nearby = []

        for offset in range(1, window + 1):

            for idx in (pos - offset, pos + offset):

                if 0 <= idx < len(tokens):

                    tok = tokens[idx]

                    if not tok.isdigit() and len(tok) >= 2:
                        nearby.append(tok)

        nearby = list(dict.fromkeys(nearby))[:3]

        for i in range(len(nearby)):

            for j in range(i + 1, len(nearby)):

                w1, w2 = sorted(
                    (nearby[i][:6], nearby[j][:6])
                )

                key_value = (
                    number
                    + "|"
                    + w1
                    + "|"
                    + w2
                )

                if key_value not in seen:
                    seen.add(key_value)
                    keys.append(("A4W", key_value))

    return keys


# ============================================================
# COMBINED
# ============================================================

def make_all_keys(name, address):
    """
    Return every blocking key for a record.
    """

    return (
        make_name_keys(name)
        + make_unicode_name_keys(name)
        + make_address_keys(address)
        + make_rare_address_token_keys(address)
        + make_a4_keys(address)
        + make_at2_keys(address)
    )


# ============================================================
# KEY TYPE REGISTRY
# ============================================================

NAME_KEY_TYPES = {
    "NP6",
    "NP8",
    "NS6",
    "FL",
    "FL3",
    "F2",
    "FML",
    "UNP4",
    "UNP6",
    "UFL",
}


ADDRESS_KEY_TYPES = {
    "AN",
    "A2",
    "AW2",
    "AL2",
    "NL",
    "A4W",
    "AT2",
    "RAT20",
}


ALL_KEY_TYPES = NAME_KEY_TYPES | ADDRESS_KEY_TYPES


# ============================================================
# DEFAULT FREQUENCY CAPS
# ============================================================

DEFAULT_FREQ_CAPS = {
    # Name blockers
    "NP6": 300,
    "NP8": 300,
    "NS6": 300,
    "FL": 400,
    "FL3": 400,
    "F2": 400,
    "FML": 400,

    # Address blockers
    "AN": 400,
    "A2": 400,
    "AW2": 400,

    # AL2 was previously the major explosion source,
    # so keep its tighter cap.
    "AL2": 80,

    "NL": 400,

    # Specialized blockers
    "A4W": 10,
    "AT2": 20,

    # NEW:
    # Rare address token blocker.
    #
    # Only address tokens occurring <=20 times in the
    # indexed source are allowed to generate candidates.
    "RAT20": 20,

    # Unicode name blockers
    "UNP4": 300,
    "UNP6": 300,
    "UFL": 300,
}