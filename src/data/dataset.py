"""PyTorch Dataset for gravitational-wave waveforms."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class GWDataset(Dataset):
    def __init__(self, df, data_dir, dtype=np.float32):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.dtype = dtype

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(self.data_dir / row["file"]).astype(self.dtype)
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        snr = torch.tensor(float(row["snr"]), dtype=torch.float32)
        return x, y, snr


def make_loader(df, data_dir, batch_size=64, shuffle=False, num_workers=0):
    ds = GWDataset(df, data_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False)
