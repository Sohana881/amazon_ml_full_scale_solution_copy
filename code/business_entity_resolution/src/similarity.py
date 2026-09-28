from difflib import SequenceMatcher
import re

from normalize import (
    normalize_business_name,
    normalize_address
)


def text_similarity(text1, text2):
    """
    Character-level similarity.
    """

    if not text1 or not text2:
        return 0.0

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


def token_similarity(text1, text2):
    """
    Compare sets of words.

    Example:

    "payne enterprises"
    "payne enterprises llc"

    share the important tokens.
    """

    if not text1 or not text2:
        return 0.0

    tokens1 = set(text1.split())
    tokens2 = set(text2.split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union)


def name_similarity(name1, name2):
    """
    Combine character and token similarity.
    """

    name1 = normalize_business_name(name1)
    name2 = normalize_business_name(name2)

    char_score = text_similarity(
        name1,
        name2
    )

    token_score = token_similarity(
        name1,
        name2
    )

    # Give more importance to character similarity,
    # while also considering shared words.
    return (
        0.6 * char_score
        +
        0.4 * token_score
    )


def address_similarity(address1, address2):
    """
    Compare normalized addresses.
    """

    address1 = normalize_address(address1)
    address2 = normalize_address(address2)

    return text_similarity(
        address1,
        address2
    )


def country_match(country1, country2):
    """
    Compare countries.

    1.0 = same country
    0.0 = different country
    """

    if not country1 or not country2:
        return 0.0

    return 1.0 if (
        country1.strip().lower()
        ==
        country2.strip().lower()
    ) else 0.0


def extract_address_numbers(address):
    """
    Extract numeric parts from an address.
    """

    if not address:
        return []

    address = normalize_address(address)

    return re.findall(
        r"\d+",
        address
    )


def address_number_match(address1, address2):
    """
    Compare numeric parts of addresses.

    1.0 = numbers match
    0.0 = numbers differ
    0.5 = number unavailable in one/both
    """

    numbers1 = extract_address_numbers(
        address1
    )

    numbers2 = extract_address_numbers(
        address2
    )

    if not numbers1 or not numbers2:
        return 0.5

    if numbers1 == numbers2:
        return 1.0

    return 0.0


def calculate_similarity(row1, row2):
    """
    Calculate similarity features.
    """

    name_score = name_similarity(
        row1["business_name"],
        row2["business_name"]
    )

    address_score = address_similarity(
        row1["business_address"],
        row2["business_address"]
    )

    address_number_score = address_number_match(
        row1["business_address"],
        row2["business_address"]
    )

    country_score = country_match(
        row1["country"],
        row2["country"]
    )

    return {
        "name_similarity": name_score,
        "address_similarity": address_score,
        "address_number_match": address_number_score,
        "country_match": country_score
    }


if __name__ == "__main__":

    s1 = {
        "business_name": "Payne Enterprises",
        "business_address": "3315 Fremont Street, Peoria, IL",
        "country": "US"
    }

    s2 = {
        "business_name": "Payne Énterprises LLC",
        "business_address": "3315 FREMONT ST, PEORIA, IL",
        "country": "US"
    }

    false_match = {
        "business_name": "International Automation Consultants Corp",
        "business_address": "338 Rev Walton Dr, Lockport, Illinois",
        "country": "US"
    }

    true_match = {
        "business_name": "International Automation Consultants Inc",
        "business_address": "329 Rev Walton Drive, Lockport, IL",
        "country": "US"
    }

    print("=" * 70)
    print("PAYNE ENTERPRISES")
    print("=" * 70)

    print(
        calculate_similarity(
            s1,
            s2
        )
    )

    print("\n" + "=" * 70)
    print("INTERNATIONAL AUTOMATION")
    print("=" * 70)

    print(
        calculate_similarity(
            true_match,
            false_match
        )
    )