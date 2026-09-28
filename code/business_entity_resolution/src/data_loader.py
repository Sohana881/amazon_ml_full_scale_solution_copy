import csv


def read_tsv(path):
    """
    Read a TSV file one row at a time.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            yield row


def preview_file(path, n=5):

    print("=" * 70)
    print("FILE:", path)
    print("=" * 70)

    for i, row in enumerate(read_tsv(path)):

        print(row)

        if i + 1 >= n:
            break


if __name__ == "__main__":

    preview_file(
        r"dataset\train\train_source1.tsv",
        n=5
    )