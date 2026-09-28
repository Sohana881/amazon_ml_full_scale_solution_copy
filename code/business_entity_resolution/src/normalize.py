import re
import unicodedata


def normalize_text(text):
    """
    Basic normalization for business names and addresses.
    """

    if not text:
        return ""

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_business_name(name):
    """
    Normalize a business name.
    """

    name = normalize_text(name)

    # Common legal/business suffixes
    suffixes = [
        "incorporated",
        "inc",
        "corporation",
        "corp",
        "limited",
        "ltd",
        "llc",
        "company",
        "co",
        "private limited",
        "pvt ltd",
    ]

    words = name.split()

    # Remove suffixes from the end
    while words and " ".join(words[-2:]) in suffixes:
        words = words[:-2]

    while words and words[-1] in suffixes:
        words = words[:-1]

    return " ".join(words)


def normalize_address(address):
    """
    Normalize a business address.
    """

    address = normalize_text(address)

    # Common address abbreviations
    replacements = {
        " street ": " st ",
        " road ": " rd ",
        " avenue ": " ave ",
        " boulevard ": " blvd ",
        " drive ": " dr ",
        " lane ": " ln ",
        " highway ": " hwy ",
        " apartment ": " apt ",
    }

    address = " " + address + " "

    for old, new in replacements.items():
        address = address.replace(old, new)

    return address.strip()


if __name__ == "__main__":

    # Small test examples
    names = [
        "Payne Enterprises",
        "Payne Énterprises",
        "Payne Enterprises LLC",
        "PAYNE-ENRTPRMISES",
    ]

    print("NAME NORMALIZATION")
    print("=" * 50)

    for name in names:
        print(f"{name}  -->  {normalize_business_name(name)}")

    print("\nADDRESS NORMALIZATION")
    print("=" * 50)

    addresses = [
        "3315 Fremont Street, Peoria, IL",
        "3315 FREMONT ST, PEORIA, IL",
        "338 Rev Walton Dr, Lockport, Illinois",
    ]

    for address in addresses:
        print(f"{address}  -->  {normalize_address(address)}")