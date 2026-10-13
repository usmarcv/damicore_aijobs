"""Ad hoc DAMICORE runner: one dataset in, one self-contained execution folder out.

Not part of the packaged test suite -- a scratch script (see docs/examples/run_example.py
for the canonical, checked-in version).

    uv run python test_damicore.py my_datasets/dataset_ai_ds_job_salaries_2026.csv
    uv run python test_damicore.py my_datasets/dataset_ai_ds_job_salaries_2026.csv \
        --pivot-values salary_usd --pivot-by job_title

With ``--pivot-values``/``--pivot-by`` the dataset is first transposed so that every distinct
``--pivot-by`` value becomes a column holding its ``--pivot-values`` cells, and DAMICORE runs
on that derived table instead.

Every execution gets its own folder, ``<results-dir>/<name>/<UTC timestamp>/``:

    args.json      resolved arguments and the dataset actually analyzed
    dataset.csv    the derived table (pivot runs only)
    run/           DAMICORE's own output directory
    tree.png       the tree, tips colored by cluster
    tree.nwk       the tree with human-readable tip labels
    terminal.log   everything printed to stdout and stderr, including the tqdm bar
"""

import argparse
import csv
import io
import json
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import toyplot
import toyplot.marker
import toyplot.png
import toytree
from damicore import DamicoreResult, estimate, run

REPO_ROOT = Path(__file__).parent
DERIVED_DATASETS_DIR = REPO_ROOT / "my_datasets"

# The dataviz skill's validated categorical order (references/palette.md), light-surface
# steps, taken in fixed slot order -- never reassigned or cycled independently of cluster id.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
NEUTRAL_INK = "#c3c2b7"  # muted/baseline ink from the same palette


class _Tee(io.TextIOBase):
    """Text stream that forwards every write to the terminal stream and to the log file."""

    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self._terminal = terminal
        self._log = log

    def write(self, text: str) -> int:
        self._terminal.write(text)
        self._log.write(text)
        return len(text)

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()

    def isatty(self) -> bool:
        # tqdm picks its rendering from this; keep the terminal's own answer.
        return self._terminal.isatty()


@contextmanager
def tee_terminal(log_path: Path) -> Iterator[None]:
    """Copy stdout and stderr to ``log_path`` for the duration of the block.

    The file is opened exclusively, so an earlier run's log is never overwritten. An
    exception escaping the block has its traceback written to the log and is re-raised,
    so the terminal still shows it once.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        try:
            with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
                yield
        except BaseException:
            traceback.print_exc(file=log)
            raise


def pivot_dataset(
    source: Path, *, values: str, by: str, delimiter: str
) -> tuple[list[str], list[list[str]], dict[str, int]]:
    """Transpose ``values`` into one column per distinct ``by`` value.

    Cells keep file order inside each group, and groups keep first-appearance order. Every
    group is truncated to the smallest group's length: the delimited reader rejects ragged
    rows, and padding the short columns would add bytes the source never contained.

    Returns the header, the rows, and each group's original (untruncated) size.
    """
    groups: dict[str, list[str]] = {}
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        missing = [name for name in (values, by) if name not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"error: column(s) not in {source.name}: {', '.join(missing)}")
        for record in reader:
            groups.setdefault(record[by], []).append(record[values])
    if not groups:
        raise SystemExit(f"error: {source.name} has no data rows to pivot")
    depth = min(len(cells) for cells in groups.values())
    header = list(groups)
    rows = [[groups[name][index] for name in header] for index in range(depth)]
    return header, rows, {name: len(cells) for name, cells in groups.items()}


def encode_table(header: list[str], rows: list[list[str]], delimiter: str) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def save_reusable_copy(path: Path, payload: bytes) -> None:
    """Keep a copy of a derived dataset for later runs, never replacing a different one."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        print(f"Derived dataset saved to {path}")
    elif path.read_bytes() == payload:
        print(f"Derived dataset already present and identical: {path}")
    else:
        raise SystemExit(
            f"error: {path} exists with different content; move or rename it, "
            "it will not be overwritten"
        )


def save_tree(result: DamicoreResult, folder: Path) -> None:
    """Write ``tree.nwk`` and ``tree.png`` for this execution into ``folder``."""
    id_to_label = dict(zip(result.membership["object_id"], result.membership["label"]))
    id_to_cluster = dict(zip(result.membership["object_id"], result.membership["cluster"]))

    # Newick tips carry object_ids ('column_000009'); the readable copy swaps in the labels,
    # still quoted (labels may hold spaces), with an embedded quote doubled as Newick requires.
    readable_newick = result.tree_newick
    for object_id, label in id_to_label.items():
        quoted = "'" + str(label).replace("'", "''") + "'"
        readable_newick = readable_newick.replace(f"'{object_id}'", quoted)
    nwk_path = folder / "tree.nwk"
    with nwk_path.open("x", encoding="utf-8") as stream:
        stream.write(readable_newick + "\n")

    # Drawn from the original Newick and relabeled through tip_labels, so label text never
    # has to survive a Newick parse. toytree may keep the quote characters as literal text.
    tree = toytree.tree(result.tree_newick)
    tip_ids = [name.strip("'") for name in tree.get_tip_labels()]
    tip_labels = [str(id_to_label[object_id]) for object_id in tip_ids]
    internal = tree.nnodes - tree.ntips

    # Tip labels keep the default ink color (text never carries series color, per the
    # dataviz skill); only the tip marker dot encodes cluster identity. node_colors/sizes
    # are in toytree's "Node idx order": tips first (matching get_tip_labels()), then
    # internal nodes, which stay a neutral, invisible-size marker.
    cluster_ids = sorted(result.clusters)
    colored = len(cluster_ids) <= len(CATEGORICAL_PALETTE)
    cluster_color = dict(zip(cluster_ids, CATEGORICAL_PALETTE)) if colored else {}
    if colored:
        tip_colors = [cluster_color[id_to_cluster[object_id]] for object_id in tip_ids]
    else:
        print(
            f"\nNote: {len(cluster_ids)} clusters exceed the {len(CATEGORICAL_PALETTE)}-slot "
            "validated palette; tree drawn without cluster colors (see membership.csv)."
        )
        tip_colors = [NEUTRAL_INK] * tree.ntips

    canvas, axes, _ = tree.draw(
        tip_labels=tip_labels,
        tip_labels_align=True,
        node_mask=False,  # toytree hides all node markers unless explicitly un-masked
        node_colors=tip_colors + [NEUTRAL_INK] * internal,
        node_sizes=[8] * tree.ntips + [0] * internal,
        width=900,
        height=max(400, 22 * tree.ntips + 80),
    )
    axes.show = False
    if colored:
        canvas.legend(
            [
                (f"Cluster {cid}", toyplot.marker.create(shape="o", mstyle={"fill": color}))
                for cid, color in cluster_color.items()
            ],
            corner=("right", 10, 100, 20 * len(cluster_ids)),
        )

    png_path = folder / "tree.png"
    toyplot.png.render(canvas, str(png_path))
    print(f"\nTree written to {png_path} and {nwk_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path, help="delimited dataset file")
    parser.add_argument("--split", choices=("columns", "rows"), default="columns")
    parser.add_argument("--delimiter", help="single-character delimiter (default: detected)")
    parser.add_argument("--pivot-values", metavar="COL", help="column whose cells are kept")
    parser.add_argument("--pivot-by", metavar="COL", help="column whose values become columns")
    parser.add_argument("--name", help="experiment name (default: derived from the dataset)")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "damicore-results")
    args = parser.parse_args()
    if (args.pivot_values is None) != (args.pivot_by is None):
        parser.error("--pivot-values and --pivot-by must be given together")
    if not args.dataset.is_file():
        parser.error(f"dataset not found: {args.dataset}")
    return args


def main(args: argparse.Namespace, folder: Path, name: str) -> None:
    dataset: Path = args.dataset
    if args.pivot_by is not None:
        # The derived table is written with the source's delimiter so --delimiter keeps
        # describing the file DAMICORE reads.
        delimiter = args.delimiter or ","
        header, rows, sizes = pivot_dataset(
            args.dataset, values=args.pivot_values, by=args.pivot_by, delimiter=delimiter
        )
        print(f"Pivot: {args.pivot_values} by {args.pivot_by} -> {len(header)} columns")
        print(f"Each column truncated to the smallest group ({len(rows)} values):")
        for group, size in sizes.items():
            print(f"  {group}: {len(rows)} used, {size - len(rows)} discarded")
        payload = encode_table(header, rows, delimiter)
        dataset = folder / "dataset.csv"
        dataset.write_bytes(payload)
        save_reusable_copy(DERIVED_DATASETS_DIR / f"{name}.csv", payload)

    with (folder / "args.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "source_dataset": str(args.dataset.resolve()),
                "analyzed_dataset": str(dataset.resolve()),
                "split": args.split,
                "delimiter": args.delimiter,
                "pivot_values": args.pivot_values,
                "pivot_by": args.pivot_by,
                "name": name,
            },
            stream,
            indent=2,
        )
        stream.write("\n")

    preview = estimate(str(dataset), split=args.split, delimiter=args.delimiter)
    print("\nEstimate:")
    print(preview.model_dump())

    # run/ is owned by DAMICORE (manifest + artifact inventory) and must not receive foreign
    # files; everything this script writes goes beside it.
    result = run(
        str(dataset),
        split=args.split,
        delimiter=args.delimiter,
        output_dir=str(folder / "run"),
        progress=True,
    )
    try:
        print("\nMembership:")
        print(result.membership)

        print("\nClusters:")
        print(result.clusters)

        print("\nNewick:")
        print(result.tree_newick)

        print("\nDistance matrix (head):")
        print(result.distance_matrix.head(10))

        save_tree(result, folder)
    finally:
        result.close()


if __name__ == "__main__":
    cli_args = parse_args()
    experiment = cli_args.name or (
        f"{cli_args.dataset.stem}__{cli_args.pivot_values}_by_{cli_args.pivot_by}"
        if cli_args.pivot_by is not None
        else cli_args.dataset.stem
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    execution_folder = cli_args.results_dir.resolve() / experiment / stamp
    # Exclusive: two executions in the same second fail instead of sharing a folder.
    execution_folder.mkdir(parents=True, exist_ok=False)
    with tee_terminal(execution_folder / "terminal.log"):
        main(cli_args, execution_folder, experiment)
    print(f"Everything from this execution is in {execution_folder}")
