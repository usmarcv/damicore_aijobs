# Quickstart as a script

A runnable, non-notebook version of [`docs/quickstart.md`](../quickstart.md), against a
30-row synthetic dataset with three latent clusters.

```sh
uv run python docs/examples/make_dataset.py   # regenerate dataset.csv (optional; already checked in)
uv run python docs/examples/run_example.py
```

`run_example.py` writes its run directory to `docs/examples/damicore-run/`, which is
git-ignored -- pipeline output is generated, not source, and is never committed.

`dataset.csv` is a deterministic synthetic fixture (see `make_dataset.py`), not a real or
user-supplied dataset.
