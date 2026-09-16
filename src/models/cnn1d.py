"""1D CNN classifier with hand-crafted statistics branch."""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    Hybrid: CNN on raw waveform + hand-crafted statistics.
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=32, stride=8, padding=16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=16, stride=2, padding=8),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        # Statistics branch — 6 global features
        self.stats_fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Combined classifier
        self.head = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, 1, 16384)
        cnn_feat = self.net(x)  # (B, 64)

        x_flat = x.squeeze(1)  # (B, 16384)

        mean_abs = x_flat.abs().mean(dim=1)
        std = x_flat.std(dim=1)
        max_abs = x_flat.abs().max(dim=1).values
        p95 = x_flat.abs().quantile(0.95, dim=1)
        p99 = x_flat.abs().quantile(0.99, dim=1)
        rms = (x_flat ** 2).mean(dim=1).sqrt()

        stats = torch.stack([mean_abs, std, max_abs, p95, p99, rms], dim=1)  # (B, 6)

        # FIXED: don't normalize over batch dim (unstable for small batches)
        # Just clip and scale to avoid extreme values
        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

        stats_feat = self.stats_fc(stats)  # (B, 32)
        combined = torch.cat([cnn_feat, stats_feat], dim=1)  # (B, 96)
        return self.head(combined).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
