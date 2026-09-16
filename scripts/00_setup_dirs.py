"""Verify directory structure. Run once."""
from pathlib import Path


def check():
    needed = [
        "data/processed", "data/metadata",
        "src/data", "src/models", "src/training",
        "src/evaluation", "src/interpretability", "src/utils",
        "scripts", "results/figures", "results/metrics", "results/logs",
        "models",
    ]
    for d in needed:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"OK  {d}")


if __name__ == "__main__":
    check()
