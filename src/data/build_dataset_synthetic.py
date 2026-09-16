"""Build the processed dataset with whitening."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.generate_waveforms import generate_gr_waveform
from src.data.inject_deviations import (
    inject_phase_deviation, sample_beta, get_pn_order,
)
from src.data.inject_signal import get_psd
from src.data.preprocessing import preprocess
from src.data.split import physics_based_split, save_splits


log = get_logger("build_dataset")


def extract_window(hp, n_samples):
    hp = np.asarray(hp, dtype=np.float64).flatten()
    if len(hp) < n_samples:
        hp = np.concatenate([np.zeros(n_samples - len(hp)), hp])
    peak_idx = int(np.argmax(np.abs(hp)))
    half = n_samples // 2
    start = peak_idx - half
    end = start + n_samples
    if start < 0:
        start = 0
        end = n_samples
    if end > len(hp):
        end = len(hp)
        start = end - n_samples
    return hp[start:end]


def compute_scale(window, delta_t, f_lower):
    import pycbc.types
    from pycbc.filter import sigma
    psd = get_psd(len(window), delta_t, f_lower)
    ts = pycbc.types.TimeSeries(window, delta_t=delta_t)
    return float(sigma(ts, psd, low_frequency_cutoff=f_lower))


def build_one_sample(hp, noise_seg, fixed_scale, delta_t, f_lower, cfg, n_samples):
    """Use a SPECIFIC noise segment for this sample."""
    window = extract_window(hp, n_samples)
    if noise_seg is not None and fixed_scale is not None:
        strain = np.asarray(noise_seg) + fixed_scale * window
    else:
        strain = window

    psd_obj = get_psd(len(strain), delta_t, f_lower)
    freqs = np.fft.rfftfreq(len(strain), d=delta_t)
    psd_tuple = (freqs, np.asarray(psd_obj))
    fs = int(1.0 / delta_t)

    return preprocess(strain, psd_tuple, delta_t, fs, cfg)


def build_dataset(n_per_class, out_dir, cfg, seed=42, snr_range=None):
    set_seed(seed)
    rng = np.random.default_rng(seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_dir = Path(cfg["paths"]["data_metadata"])
    meta_dir.mkdir(parents=True, exist_ok=True)

    delta_t = cfg["waveform"]["delta_t"]
    n_samples = cfg["waveform"]["n_samples"]
    f_lower = cfg["waveform"]["f_lower"]

    # ------- Generate ONE big noise array -------
    noise_pool = None
    use_noise = snr_range is not None
    total_samples = 2 * n_per_class  # total files
    if use_noise:
        from src.data.generate_noise import generate_synthetic_noise
        # Generate a big pool: 4x n_samples per file (so slices don't overlap)
        pool_size = total_samples * n_samples
        log.info(f"Generating noise pool: {pool_size} samples")
        noise_pool = generate_synthetic_noise(pool_size, delta_t, f_lower)
        log.info(f"Noise pool shape: {noise_pool.shape}")

    snr_lo, snr_hi = snr_range if snr_range else (50.0, 50.0)

    FIXED = {
        "mass1": 30.0, "mass2": 25.0,
        "spin1z": 0.0, "spin2z": 0.0,
        "distance": 500.0, "inclination": 0.0,
    }

    # Compute fixed scale from a reference GR window
    if use_noise:
        hp_ref, _ = generate_gr_waveform(
            **FIXED, delta_t=delta_t, f_lower=f_lower,
            approximant=cfg["waveform"]["approximant"],
        )
        window_ref = extract_window(hp_ref, n_samples)
        snr_opt_ref = compute_scale(window_ref, delta_t, f_lower)
        target_snr_mean = 0.5 * (snr_lo + snr_hi)
        FIXED_SCALE = target_snr_mean / snr_opt_ref
        log.info(f"Fixed scale = {FIXED_SCALE:.4e}")
    else:
        FIXED_SCALE = None

    rows = []
    file_idx = 0

    # GR
    log.info(f"Generating {n_per_class} GR waveforms...")
    for i in tqdm(range(n_per_class), desc="GR"):
        hp, _ = generate_gr_waveform(
            **FIXED, delta_t=delta_t, f_lower=f_lower,
            approximant=cfg["waveform"]["approximant"],
        )
        snr = float(rng.uniform(snr_lo, snr_hi)) if use_noise else 50.0
        # Slice a UNIQUE noise segment for this file
        if use_noise:
            start = file_idx * n_samples
            noise_seg = noise_pool[start:start + n_samples]
        else:
            noise_seg = None
        file_idx += 1

        x = build_one_sample(hp, noise_seg, FIXED_SCALE, delta_t, f_lower, cfg, n_samples)
        fname = f"gr_{i:06d}.npy"
        np.save(out / fname, x)
        rows.append({"file": fname, "label": 0, "type": "GR", **FIXED,
                     "beta": 0.0, "pn_order": 0, "snr": snr})

    # Beyond-GR
    all_types = cfg["deviation"]["train_types"] + [cfg["deviation"]["test_type"]]
    log.info(f"Generating {n_per_class} beyond-GR waveforms...")
    for i in tqdm(range(n_per_class), desc="Beyond-GR"):
        hp, _ = generate_gr_waveform(
            **FIXED, delta_t=delta_t, f_lower=f_lower,
            approximant=cfg["waveform"]["approximant"],
        )
        dev_type = str(rng.choice(all_types))
        beta = sample_beta(cfg, dev_type, rng)
        pn_order = get_pn_order(cfg, dev_type)
        hp_mod = inject_phase_deviation(hp, delta_t, beta, pn_order=pn_order)
        snr = float(rng.uniform(snr_lo, snr_hi)) if use_noise else 50.0

        if use_noise:
            start = file_idx * n_samples
            noise_seg = noise_pool[start:start + n_samples]
        else:
            noise_seg = None
        file_idx += 1

        x = build_one_sample(hp_mod, noise_seg, FIXED_SCALE, delta_t, f_lower, cfg, n_samples)
        fname = f"mod_{dev_type}_{i:06d}.npy"
        np.save(out / fname, x)
        rows.append({"file": fname, "label": 1, "type": dev_type, **FIXED,
                     "beta": beta, "pn_order": pn_order, "snr": snr})

    df = pd.DataFrame(rows)
    df.to_csv(meta_dir / "all_samples.csv", index=False)
    log.info(f"Saved {len(df)} samples")

    train_df, val_df, test_seen, test_unseen = physics_based_split(df, cfg, seed)
    save_splits(train_df, val_df, test_seen, test_unseen, meta_dir)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_per_class", type=int, default=None)
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--snr_min", type=float, default=None)
    p.add_argument("--snr_max", type=float, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    n = args.n_per_class or cfg["dataset"]["n_per_class"]
    out = args.out or cfg["paths"]["data_processed"]

    snr_range = None
    if args.snr_min is not None and args.snr_max is not None:
        snr_range = (args.snr_min, args.snr_max)

    build_dataset(n, out, cfg, snr_range=snr_range)


if __name__ == "__main__":
    main()
