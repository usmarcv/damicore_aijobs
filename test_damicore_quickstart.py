"""docs/quickstart.md as a script: estimate, run, print the results, save the tree.

Not part of the packaged test suite -- a scratch script.

    uv run python test_damicore_quickstart.py my_datasets/dataset_ai_ds_job_salaries_2026.csv
    uv run python test_damicore_quickstart.py DATASET --split rows

Two files are added to the run directory, beside DAMICORE's own ``tree.nwk`` (whose tips are
object ids): ``tree.pdf``, the drawn tree with tips colored by cluster, and
``tree_labeled.nwk``, the same tree with human-readable tip labels. Both are regenerated on
every execution.
"""

import argparse
from pathlib import Path

import toyplot.marker
import toyplot.pdf
import toytree
from damicore import DamicoreResult, estimate, run

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DAMICORE quickstart on one dataset.")
    parser.add_argument("dataset", type=Path, help="delimited dataset file")
    parser.add_argument("--split", choices=("columns", "rows"), default="columns")
    parser.add_argument("--delimiter", help="single-character delimiter (default: detected)")
    parser.add_argument("--output-dir", type=Path, help="default: ./damicore-results/<run_id>")
    return parser.parse_args()


def labeled_newick(result: DamicoreResult) -> str:
    """Return the tree with each tip's object id replaced by its human-readable label.

    Labels stay single-quoted because they may hold spaces, with an embedded quote doubled
    as Newick requires.
    """
    newick = result.tree_newick
    for object_id, label in zip(result.membership["object_id"], result.membership["label"]):
        quoted = "'" + str(label).replace("'", "''") + "'"
        newick = newick.replace(f"'{object_id}'", quoted)
    return newick


def save_tree(result: DamicoreResult) -> None:
    """Write ``tree.pdf`` and ``tree_labeled.nwk`` into the run directory.

    Neither name is a DAMICORE artifact, so the run's own files, and the hashes its manifest
    records for them, are left untouched.
    """
    folder = result.artifacts.run_dir
    newick_path, pdf_path = folder / "tree_labeled.nwk", folder / "tree.pdf"
    newick_path.write_text(labeled_newick(result) + "\n", encoding="utf-8")

    # Drawn from the original Newick and relabeled through tip_labels: toytree keeps the
    # quotes and drops spaces when it parses a quoted label, so labels must not go through it.
    id_to_label = dict(zip(result.membership["object_id"], result.membership["label"]))
    id_to_cluster = dict(zip(result.membership["object_id"], result.membership["cluster"]))
    tree = toytree.tree(result.tree_newick)
    tip_ids = [name.strip("'") for name in tree.get_tip_labels()]
    internal = tree.nnodes - tree.ntips

    # Cluster identity is carried by the tip dot only; label text keeps the default ink.
    # Past the palette's length the colors would repeat and mislead, so none are drawn.
    cluster_ids = sorted(result.clusters)
    colored = len(cluster_ids) <= len(CATEGORICAL_PALETTE)
    cluster_color = dict(zip(cluster_ids, CATEGORICAL_PALETTE)) if colored else {}
    if not colored:
        print(
            f"\nNote: {len(cluster_ids)} clusters exceed the {len(CATEGORICAL_PALETTE)}-color "
            "palette; tree drawn without cluster colors (see membership.csv)."
        )

    # node_colors/node_sizes follow toytree's node order: tips first, in get_tip_labels()
    # order, then internal nodes, which get a zero-size marker.
    canvas, axes, _ = tree.draw(
        tip_labels=[str(id_to_label[object_id]) for object_id in tip_ids],
        tip_labels_align=True,
        node_mask=False,  # toytree hides all node markers unless explicitly un-masked
        node_colors=[
            cluster_color.get(id_to_cluster[object_id], NEUTRAL_INK) for object_id in tip_ids
        ]
        + [NEUTRAL_INK] * internal,
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
    toyplot.pdf.render(canvas, str(pdf_path))
    print(f"\nTree written to {pdf_path} and {newick_path}")


def section(title: str, body: object) -> None:
    print(f"\n{title}:\n{body}")


def main(args: argparse.Namespace) -> None:
    dataset = str(args.dataset)

    preview = estimate(dataset, split=args.split, delimiter=args.delimiter)
    section("Estimate", preview.model_dump())

    result = run(dataset, split=args.split, delimiter=args.delimiter, output_dir=args.output_dir)
    try:
        section("Membership", result.membership)
        section("Clusters", result.clusters)
        section("Distance matrix (head)", result.distance_matrix.head(10))
        section("Newick (object ids)", result.tree_newick)
        section("Newick (labels)", labeled_newick(result))
        save_tree(result)
    finally:
        result.close()


# run() opens a process pool whose workers re-import this module.
if __name__ == "__main__":
    main(parse_args())
