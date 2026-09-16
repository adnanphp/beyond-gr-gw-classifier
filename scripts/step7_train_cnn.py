"""Train the CNN classifier."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.dataset import make_loader
from src.models.cnn1d import CNN1D, count_parameters
from src.training.train import train, get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--meta", default="data/metadata")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default="models/cnn1d_best.pt")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("train_cnn", "results/logs/train.log")

    train_df = pd.read_csv(Path(args.meta) / "train.csv")
    val_df = pd.read_csv(Path(args.meta) / "val.csv")

    log.info(f"train: {len(train_df)} samples")
    log.info(f"val:   {len(val_df)} samples")

    train_loader = make_loader(train_df, args.data, args.batch_size, shuffle=True)
    val_loader = make_loader(val_df, args.data, args.batch_size, shuffle=False)

    model = CNN1D(dropout=cfg["model"]["dropout"])
    log.info(f"model: CNN1D, params={count_parameters(model):,}")

    device = get_device(cfg["training"]["device"])
    log.info(f"device: {device}")

    model, best_val_acc = train(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        patience=cfg["training"]["early_stopping_patience"],
        out_path=args.out,
        log_fn=log.info,
    )

    log.info(f"Training done. Best val_acc = {best_val_acc:.4f}")
    log.info(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
