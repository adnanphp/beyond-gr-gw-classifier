"""Evaluate on test sets."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import json

from src.utils.config import load_config
from src.data.dataset import make_loader
from src.models.cnn1d import CNN1D
from src.training.train import get_device, evaluate
from src.evaluation.metrics import compute_metrics


def eval_split(model, df, data_dir, batch_size, device):
    loader = make_loader(df, data_dir, batch_size, shuffle=False)
    acc, probs, labels = evaluate(model, loader, device)
    metrics = compute_metrics(labels, probs)
    return acc, metrics, labels, probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--meta", default="data/metadata")
    p.add_argument("--model", default="models/cnn1d_best.pt")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["training"]["device"])

    model = CNN1D(dropout=cfg["model"]["dropout"])
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    results = {}
    for split in ["train", "val", "test_seen", "test_unseen"]:
        df = pd.read_csv(Path(args.meta) / f"{split}.csv")
        acc, metrics, _, _ = eval_split(model, df, args.data, cfg["training"]["batch_size"], device)
        results[split] = metrics
        print(f"\n=== {split} ({len(df)} samples) ===")
        print(f"  accuracy: {metrics['accuracy']:.4f}")
        print(f"  roc_auc:  {metrics['roc_auc']:.4f}")
        print(f"  f1:       {metrics['f1']:.4f}")

    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    with open("results/metrics/test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/metrics/test_results.json")


if __name__ == "__main__":
    main()
