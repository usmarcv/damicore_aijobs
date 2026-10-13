"""Generate a small, deterministic 30-row demo dataset for docs/quickstart.md.

Not part of the damicore packages -- a throwaway script to produce a CSV the quickstart
example can point at. Each row belongs to one of three latent clusters and carries a long
cluster-specific token plus a little per-row noise, so gzip/zlib NCD can actually recover
the groups (a short or noise-dominated signal would not survive compression).
"""

import csv
from pathlib import Path
from random import Random
from string import ascii_lowercase

ROWS = 30
COLUMNS = 6
CLUSTERS = 3
TOKEN_LENGTH = 24


def cluster_token(cluster: int) -> str:
    rng = Random(f"cluster-{cluster}")
    return "".join(rng.choice(ascii_lowercase) for _ in range(TOKEN_LENGTH))


def main() -> None:
    destination = Path(__file__).parent / "dataset.csv"
    rng = Random(42)
    tokens = [cluster_token(c) for c in range(CLUSTERS)]
    headers = [f"feature_{i + 1}" for i in range(COLUMNS)]

    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(headers)
        for row in range(ROWS):
            cluster = row % CLUSTERS
            writer.writerow(f"{tokens[cluster]}{rng.randrange(100):02d}" for _ in range(COLUMNS))

    print(f"Wrote {ROWS} rows x {COLUMNS} columns ({CLUSTERS} latent clusters) to {destination}")


if __name__ == "__main__":
    main()
