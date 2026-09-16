"""Build a dataset using the real GW150914 event as the GR template."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import h5py
from scipy.signal import welch, butter, filtfilt
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.generate_noise import fetch_real_noise
from src.data.inject_deviations import (
    inject_amplitude_modulation, inject_phase_modulation,
    inject_frequency_modulation, sample_beta, get_pn_order,
)
from src.data.split import physics_based_split, save_splits


log = get_logger("build_gw150914")


def load_gw150914():
    with h5py.File('data/raw/gw150914/H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5', 'r') as f:
        data = f['strain/Strain'][()]
        gps_start = float(f['meta']['GPSstart'][()])
    return data, gps_start


def whiten(data, fs, noise_ref):
    freqs, psd = welch(noise_ref, fs=fs, nperseg=4096)
    data_f = np.fft.rfft(data)
    freqs_fft = np.fft.rfftfreq(len(data), d=1/fs)
    psd_interp = np.interp(freqs_fft, freqs, psd)
    psd_interp[psd_interp <= 0] = np.inf
    wf = 1.0 / np.sqrt(psd_interp * fs / 2.0)
    whitened = np.fft.irfft(data_f * wf, n=len(data))
    return whitened


def bandpass(x, fs, low=20.0, high=500.0, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, x)


def main():
    cfg = load_config("config.yaml")
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    fs = 4096
    n_samples = cfg["waveform"]["n_samples"]
    n_per_class = cfg["dataset"]["n_per_class"]
    target_snr = 20.0

    log.info("Loading GW150914")
    data, gps_start = load_gw150914()
    event_gps = 1126259462.4
    event_offset = int((event_gps - gps_start) * fs)
    log.info(f"Event at sample {event_offset}")

    # Whiten the entire 32-second data using early segment as PSD reference
    quiet = data[0:16384]
    whiten_full = whiten(data, fs, quiet)

    # Extract the event window (4 seconds centered on event)
    half = n_samples // 2
    event_window = whiten_full[event_offset - half:event_offset + half]
    event_window = event_window / event_window.std()  # normalize

    # Bandpass event template
    event_template = bandpass(event_window, fs)
    event_template = event_template / (np.abs(event_template).std() + 1e-12)
    log.info(f"Event template: std={event_template.std():.3f}, max={np.abs(event_template).max():.3f}")

    # Get noise pool (from the same 32-second file, but excluding event region)
    noise_before = whiten_full[0:event_offset - half]
    noise_after = whiten_full[event_offset + half:]
    noise_pool = np.concatenate([noise_before, noise_after])
    noise_pool = noise_pool / noise_pool.std()
    log.info(f"Noise pool: {len(noise_pool)} samples, std={noise_pool.std():.3f}")

    # Fixed scale: signal peak at target_snr in units of noise std
    FIXED_SCALE = target_snr / np.abs(event_template).max()
    log.info(f"Fixed scale = {FIXED_SCALE:.3f}")

    out = Path(cfg["paths"]["data_processed"])
    out.mkdir(parents=True, exist_ok=True)
    meta_dir = Path(cfg["paths"]["data_metadata"])
    meta_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    file_idx = 0

    def make_sample(template):
        nonlocal file_idx
        # Random noise window from the pool
        max_start = len(noise_pool) - n_samples
        if max_start <= 0:
            noise_seg = noise_pool[:n_samples]
        else:
            start = int(rng.integers(0, max_start))
            noise_seg = noise_pool[start:start + n_samples]
        # Inject event template at fixed scale
        strain = noise_seg + FIXED_SCALE * template
        # Clip
        clip = cfg["preprocessing"].get("clip_value", 100.0)
        strain = np.clip(strain, -clip, clip)
        return strain.astype(np.float32)

    # GR class
    log.info(f"Generating {n_per_class} GR (GW150914) samples")
    for i in tqdm(range(n_per_class), desc="GR"):
        x = make_sample(event_template)
        fname = f"gr_{i:06d}.npy"
        np.save(out / fname, x)
        rows.append({"file": fname, "label": 0, "type": "GR", "snr": target_snr, "mass1": 30.0, "mass2": 25.0, "beta": 0.0, "pn_order": 0})

    # Beyond-GR class
    all_types = cfg["deviation"]["train_types"] + [cfg["deviation"]["test_type"]]
    dev_kind = cfg["deviation"].get("kind", "amplitude")
    log.info(f"Generating {n_per_class} beyond-GR samples ({dev_kind})")

    for i in tqdm(range(n_per_class), desc="Beyond-GR"):
        dev_type = str(rng.choice(all_types))
        beta = sample_beta(cfg, dev_type, rng)
        pn_order = get_pn_order(cfg, dev_type)

        if dev_kind == "phase":
            mod = inject_phase_modulation(event_template, 1/fs, beta, pn_order=pn_order)
        elif dev_kind == "frequency":
            mod = inject_frequency_modulation(event_template, 1/fs, beta, pn_order=pn_order)
        else:
            mod = inject_amplitude_modulation(event_template, 1/fs, beta, pn_order=pn_order)

        x = make_sample(mod)
        fname = f"mod_{dev_type}_{i:06d}.npy"
        np.save(out / fname, x)
        rows.append({"file": fname, "label": 1, "type": dev_type, "snr": target_snr, "mass1": 30.0, "mass2": 25.0, "beta": beta, "pn_order": pn_order})

    df = pd.DataFrame(rows)
    df.to_csv(meta_dir / "all_samples.csv", index=False)
    log.info(f"Saved {len(df)} samples")

    # Simple stratified split (GW150914 template — no mass variation)
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=cfg["seed"], stratify=df["label"])
    val_df, test_unseen_df = train_test_split(temp_df, test_size=0.50, random_state=cfg["seed"], stratify=temp_df["label"])
    test_seen_df = val_df.copy()
    save_splits(train_df, val_df, test_seen_df, test_unseen_df, meta_dir)


if __name__ == "__main__":
    main()
