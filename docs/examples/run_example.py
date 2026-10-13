"""docs/quickstart.md, adapted from notebook cells to a plain .py script.

The notebook version uses `%pip` and `display`; here that's `pip install damicore` (already
done via `make install` in the workspace) and `print`. The `run()` call spawns a worker
process pool by default, which re-imports this module -- hence the `__main__` guard.
"""

from pathlib import Path

from damicore import estimate, run

csv_path = str(Path(__file__).parent / "dataset.csv")

if __name__ == "__main__":
    preview = estimate(csv_path, split="rows")
    print(preview.model_dump())

    result = run(csv_path, split="rows", output_dir=str(Path(__file__).parent / "damicore-run"))
    print(result.membership)
    print(result.clusters)
    print(result.tree_newick)
    print(result.distance_matrix.head(10))

    result.close()
