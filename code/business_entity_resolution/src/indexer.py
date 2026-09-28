"""
indexer.py
============================================================
Builds blocking indexes and a compact attribute store for a
source file (S2 or S3). Source-independent: the SAME code path
is used for S2 and S3, and for train_10k or full data.

Two backends are provided:

  "memory" (default) - everything held in Python dicts. Simple
      and fast; fine for train_10k and for moderate production
      hardware.

  "sqlite" - the (key_type, key_value) -> entity_id posting list
      and the entity attribute store are written to a SQLite
      database on disk instead of RAM. Use this for the full
      5M+ record production run if memory becomes a constraint.

Entity IDs are stored compactly: the source prefix ("S2-"/"S3-")
is stripped and the remaining numeric suffix is kept as an int
wherever possible.
"""

import sqlite3
from collections import defaultdict

from blocking import (
    make_name_keys,
    make_unicode_name_keys,
    make_address_keys,
    make_rare_address_token_keys,
    make_a4_keys,
    make_at2_keys,
)

from data_loader import read_tsv


# ============================================================
# ENTITY ID HELPERS
# ============================================================

def split_entity_id(entity_id):
    """
    'S2-166376419' -> ('S2', 166376419)

    Falls back to (prefix, raw_string) if the suffix isn't
    numeric.
    """

    if "-" not in entity_id:
        return "", entity_id

    prefix, suffix = entity_id.split("-", 1)

    if suffix.isdigit():
        return prefix, int(suffix)

    return prefix, suffix


def join_entity_id(prefix, suffix):
    return f"{prefix}-{suffix}"


# ============================================================
# COUNTRY-AWARE KEY
# ============================================================

def compose_key(country, key_value):
    """
    Fold country into the blocking key so that blocking stays
    country-aware without changing the key-generation functions
    in blocking.py.
    """

    return (country or "").strip().upper() + "||" + key_value


# ============================================================
# IN-MEMORY BACKEND
# ============================================================

class MemoryIndex:
    """
    posting[key_type][key_value] -> list of compact entity ids

    attrs[entity_id_suffix] -> (name, address, country)
    """

    def __init__(self, source_prefix):
        self.source_prefix = source_prefix

        self.postings = defaultdict(
            lambda: defaultdict(list)
        )

        self.attrs = {}
        self.count = 0

    def add_record(
        self,
        entity_id,
        name,
        address,
        country
    ):

        _, suffix = split_entity_id(entity_id)

        # Store source attributes.
        self.attrs[suffix] = (
            name,
            address,
            country
        )

        # ----------------------------------------------------
        # NAME BLOCKERS
        # ----------------------------------------------------

        for key_type, key_value in (
            make_name_keys(name)
            + make_unicode_name_keys(name)
        ):

            self.postings[key_type][
                compose_key(country, key_value)
            ].append(suffix)

        # ----------------------------------------------------
        # ORIGINAL ADDRESS BLOCKERS
        # ----------------------------------------------------

        for key_type, key_value in make_address_keys(address):

            self.postings[key_type][
                compose_key(country, key_value)
            ].append(suffix)

        # ----------------------------------------------------
        # RARE ADDRESS TOKEN BLOCKER
        #
        # RAT20 keys are generated here.
        # The frequency cap itself is applied by the candidate
        # generation/index lookup layer.
        # ----------------------------------------------------

        for key_type, key_value in make_rare_address_token_keys(
            address
        ):

            self.postings[key_type][
                compose_key(country, key_value)
            ].append(suffix)

        # ----------------------------------------------------
        # OTHER SPECIALIZED ADDRESS BLOCKERS
        # ----------------------------------------------------

        for key_type, key_value in (
            make_a4_keys(address)
            + make_at2_keys(address)
        ):

            self.postings[key_type][
                compose_key(country, key_value)
            ].append(suffix)

        self.count += 1

    def bucket(self, key_type, key_value):

        return self.postings.get(
            key_type,
            {}
        ).get(
            key_value,
            []
        )

    def get_attrs(self, suffix):

        return self.attrs.get(suffix)

    def num_keys(self):

        return sum(
            len(v)
            for v in self.postings.values()
        )


# ============================================================
# SQLITE BACKEND
# ============================================================

class SqliteIndex:
    """
    Disk-backed equivalent of MemoryIndex.

    Tables:

        postings(key_type, key_value, suffix)

        attrs(
            suffix PRIMARY KEY,
            name,
            address,
            country
        )
    """

    def __init__(
        self,
        source_prefix,
        db_path
    ):

        self.source_prefix = source_prefix
        self.db_path = db_path
        self.count = 0

        self.conn = sqlite3.connect(
            db_path
        )

        self.conn.execute(
            "PRAGMA synchronous = OFF"
        )

        self.conn.execute(
            "PRAGMA journal_mode = MEMORY"
        )

        # ----------------------------------------------------
        # TABLES
        # ----------------------------------------------------

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS postings (
                key_type TEXT,
                key_value TEXT,
                suffix TEXT
            )
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attrs (
                suffix TEXT PRIMARY KEY,
                name TEXT,
                address TEXT,
                country TEXT
            )
            """
        )

        self._buffer = []
        self._attr_buffer = []

    def add_record(
        self,
        entity_id,
        name,
        address,
        country
    ):

        _, suffix = split_entity_id(entity_id)

        suffix = str(suffix)

        # ----------------------------------------------------
        # ATTRIBUTE STORE
        # ----------------------------------------------------

        self._attr_buffer.append(
            (
                suffix,
                name,
                address,
                country
            )
        )

        # ----------------------------------------------------
        # NAME BLOCKERS
        # ----------------------------------------------------

        for key_type, key_value in (
            make_name_keys(name)
            + make_unicode_name_keys(name)
        ):

            self._buffer.append(
                (
                    key_type,
                    compose_key(
                        country,
                        key_value
                    ),
                    suffix
                )
            )

        # ----------------------------------------------------
        # ORIGINAL ADDRESS BLOCKERS
        # ----------------------------------------------------

        for key_type, key_value in make_address_keys(
            address
        ):

            self._buffer.append(
                (
                    key_type,
                    compose_key(
                        country,
                        key_value
                    ),
                    suffix
                )
            )

        # ----------------------------------------------------
        # RARE ADDRESS TOKEN BLOCKER
        # ----------------------------------------------------

        for key_type, key_value in make_rare_address_token_keys(
            address
        ):

            self._buffer.append(
                (
                    key_type,
                    compose_key(
                        country,
                        key_value
                    ),
                    suffix
                )
            )

        # ----------------------------------------------------
        # OTHER SPECIALIZED ADDRESS BLOCKERS
        # ----------------------------------------------------

        for key_type, key_value in (
            make_a4_keys(address)
            + make_at2_keys(address)
        ):

            self._buffer.append(
                (
                    key_type,
                    compose_key(
                        country,
                        key_value
                    ),
                    suffix
                )
            )

        self.count += 1

        # Periodically flush to SQLite.
        if len(self._buffer) >= 200_000:
            self._flush()

    def _flush(self):

        if self._buffer:

            self.conn.executemany(
                """
                INSERT INTO postings
                VALUES (?, ?, ?)
                """,
                self._buffer
            )

            self._buffer = []

        if self._attr_buffer:

            self.conn.executemany(
                """
                INSERT OR REPLACE INTO attrs
                VALUES (?, ?, ?, ?)
                """,
                self._attr_buffer
            )

            self._attr_buffer = []

    def finalize(self):

        self._flush()

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_postings
            ON postings(key_type, key_value)
            """
        )

        self.conn.commit()

    def bucket(
        self,
        key_type,
        key_value
    ):

        cur = self.conn.execute(
            """
            SELECT suffix
            FROM postings
            WHERE key_type=?
              AND key_value=?
            """,
            (
                key_type,
                key_value
            )
        )

        return [
            row[0]
            for row in cur.fetchall()
        ]

    def get_attrs(self, suffix):

        cur = self.conn.execute(
            """
            SELECT name, address, country
            FROM attrs
            WHERE suffix=?
            """,
            (str(suffix),)
        )

        row = cur.fetchone()

        return row

    def num_keys(self):

        cur = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT
                    key_type,
                    key_value
                FROM postings
            )
            """
        )

        return cur.fetchone()[0]


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(
    path,
    source_prefix,
    limit=None,
    backend="memory",
    db_path=None
):
    """
    Stream `path` and build a blocking index + attribute store.

    limit=None
        Process every row.

    limit=10000
        Process first 10,000 rows.

    The same function therefore works for:

        train_10k

    and:

        full 5M+ production data.
    """

    # --------------------------------------------------------
    # SELECT BACKEND
    # --------------------------------------------------------

    if backend == "sqlite":

        if db_path is None:
            raise ValueError(
                "db_path is required for the sqlite backend"
            )

        index = SqliteIndex(
            source_prefix,
            db_path
        )

    else:

        index = MemoryIndex(
            source_prefix
        )

    print(
        f"[indexer] Building {backend} index "
        f"for {source_prefix} from: {path}"
    )

    # --------------------------------------------------------
    # READ SOURCE FILE
    # --------------------------------------------------------

    for i, row in enumerate(
        read_tsv(path)
    ):

        if limit is not None and i >= limit:
            break

        index.add_record(
            entity_id=row.get(
                "entity_id",
                ""
            ),

            name=row.get(
                "business_name",
                ""
            ),

            address=row.get(
                "business_address",
                ""
            ),

            country=row.get(
                "country",
                ""
            ).strip(),
        )

        # Progress message.
        if index.count % 500_000 == 0:

            print(
                f"[indexer]   indexed "
                f"{index.count:,} "
                f"{source_prefix} records..."
            )

    # --------------------------------------------------------
    # FINALIZE SQLITE
    # --------------------------------------------------------

    if backend == "sqlite":
        index.finalize()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        f"[indexer] Finished {source_prefix}: "
        f"{index.count:,} records, "
        f"{index.num_keys():,} distinct blocking keys."
    )

    return index