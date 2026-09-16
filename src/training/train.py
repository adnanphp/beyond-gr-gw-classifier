"""Training loop with checkpointing and early stopping."""
import copy
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


def get_device(device_str="auto"):
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    n = 0
    all_probs, all_labels = [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += (preds == y).sum().item()
        n += len(y)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    acc = correct / n
    return acc, np.concatenate(all_probs), np.concatenate(all_labels)


def train(model, train_loader, val_loader, epochs=50, lr=1e-3,
          device="auto", patience=20, out_path="models/cnn1d_best.pt",
          log_fn=print, weight_decay=1e-4):
    device = get_device(device)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0

        log_fn(f"Epoch {epoch:3d} | loss={train_loss:.4f} | val_acc={val_acc:.4f} | {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, out_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log_fn(f"Early stopping at epoch {epoch} (best={best_val_acc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc
