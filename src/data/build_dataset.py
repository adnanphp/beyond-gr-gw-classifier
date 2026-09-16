"""Build dataset — whitening-first with per-sample noise PSD."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import welch, butter, filtfilt

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.generate_waveforms import generate_gr_waveform
from src.data.inject_deviations import (
    inject_phase_deviation, sample_beta, get_pn_order,
)
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


def bandpass(x, fs, low=20.0, high=500.0, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def get_whiten_filter(noise_seg, delta_t):
    fs = int(1.0 / delta_t)
    nperseg = min(4096, len(noise_seg) // 4)
    freqs, psd_vals = welch(noise_seg, fs=fs, nperseg=nperseg)
    freqs_fft = np.fft.rfftfreq(len(noise_seg), d=delta_t)
    psd_interp = np.interp(freqs_fft, freqs, psd_vals)
    psd_interp[psd_interp <= 0] = np.inf
    return 1.0 / np.sqrt(psd_interp * fs / 2.0)


def get_real_noise_pool(total_samples, delta_t, n_samples):
    from src.data.generate_noise import fetch_real_noise
    noise, fs = fetch_real_noise()
    log.info(f"Real noise: {len(noise)} samples @ {fs} Hz")
    rng = np.random.default_rng(42)
    pool = np.zeros(total_samples, dtype=np.float64)
    cursor = 0
    while cursor < total_samples:
        offset = int(rng.integers(0, max(1, len(noise) - n_samples)))
        chunk_len = min(total_samples - cursor, len(noise) - offset)
        pool[cursor:cursor + chunk_len] = noise[offset:offset + chunk_len]
        cursor += chunk_len
    return pool


def get_synthetic_noise_pool(total_samples, delta_t, f_lower):
    from src.data.generate_noise import generate_synthetic_noise
    return generate_synthetic_noise(total_samples, delta_t, f_lower)


def build_one_sample(hp, noise_seg, delta_t, cfg, n_samples, fixed_scale):
    window = extract_window(hp, n_samples)
    noise_seg = np.asarray(noise_seg)[:n_samples]

    fs = int(1.0 / delta_t)
    wf = get_whiten_filter(noise_seg, delta_t)

    # Whiten noise and normalize to unit std
    noise_f = np.fft.rfft(noise_seg)
    noise_w = np.fft.irfft(noise_f * wf, n=len(noise_seg))
    noise_w = noise_w / (noise_w.std() + 1e-12)

    # Whiten signal WITHOUT std normalization (preserve amplitude!)
    window_f = np.fft.rfft(window)
    signal_w = np.fft.irfft(window_f * wf, n=len(window))

    # Apply SAME fixed scale for all
    strain = noise_w + fixed_scale * signal_w

    if cfg["preprocessing"].get("bandpass", True):
        strain = bandpass(strain, fs,
                          low=cfg["preprocessing"]["bandpass_low"],
                          high=cfg["preprocessing"]["bandpass_high"],
                          order=cfg["preprocessing"]["bandpass_order"])

    clip = cfg["preprocessing"].get("clip_value", 100.0)
    strain = np.clip(strain, -clip, clip)
    strain = np.nan_to_num(strain, nan=0.0, posinf=clip, neginf=-clip)

    return strain.astype(np.float32)


def build_dataset(n_per_class, out_dir, cfg, seed=42, snr_range=None,
                  noise_source="real"):
    set_seed(seed)
    rng = np.random.default_rng(seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_dir = Path(cfg["paths"]["data_metadata"])
    meta_dir.mkdir(parents=True, exist_ok=True)

    delta_t = cfg["waveform"]["delta_t"]
    n_samples = cfg["waveform"]["n_samples"]
    f_lower = cfg["waveform"]["f_lower"]

    use_noise = snr_range is not None
    total_files = 2 * n_per_class

    if use_noise:
        if noise_source == "real":
            log.info("Loading real LIGO noise")
            noise_pool = get_real_noise_pool(total_files * n_samples,
                                              delta_t, n_samples)
        else:
            log.info("Generating synthetic noise")
            noise_pool = get_synthetic_noise_pool(total_files * n_samples,
                                                    delta_t, f_lower)
    else:
        noise_pool = None

    snr_lo, snr_hi = snr_range if snr_range else (50.0, 50.0)

    FIXED = {
        "mass1": cfg["waveform"].get("fixed_mass1", 30.0),
        "mass2": cfg["waveform"].get("fixed_mass2", 25.0),
        "spin1z": 0.0, "spin2z": 0.0,
        "distance": 500.0, "inclination": 0.0,
    }

    # Compute FIXED_SCALE from ONE GR reference (no std normalization!)
    if use_noise:
        hp_ref, _ = generate_gr_waveform(
            **FIXED, delta_t=delta_t, f_lower=f_lower,
            approximant=cfg["waveform"]["approximant"],
        )
        window_ref = extract_window(hp_ref, n_samples)
        noise_ref = noise_pool[:n_samples]
        wf = get_whiten_filter(noise_ref, delta_t)
        window_f = np.fft.rfft(window_ref)
        signal_ref = np.fft.irfft(window_f * wf, n=n_samples)
        peak_ref = np.max(np.abs(signal_ref))
        target_snr_mean = 0.5 * (snr_lo + snr_hi)
        FIXED_SCALE = target_snr_mean / peak_ref
        log.info(f"Fixed scale = {FIXED_SCALE:.4f} (ref peak={peak_ref:.2f})")
    else:
        FIXED_SCALE = 1.0

    rows = []
    file_idx = 0

    log.info(f"Generating {n_per_class} GR waveforms...")
    for i in tqdm(range(n_per_class), desc="GR"):
        hp, _ = generate_gr_waveform(
            **FIXED, delta_t=delta_t, f_lower=f_lower,
            approximant=cfg["waveform"]["approximant"],
        )
        snr = float(rng.uniform(snr_lo, snr_hi)) if use_noise else 50.0
        if use_noise:
            start = file_idx * n_samples
            noise_seg = noise_pool[start:start + n_samples]
        else:
            noise_seg = None
        file_idx += 1

        x = build_one_sample(hp, noise_seg, delta_t, cfg, n_samples, FIXED_SCALE)
        fname = f"gr_{i:06d}.npy"
        np.save(out / fname, x)
        rows.append({"file": fname, "label": 0, "type": "GR", **FIXED,
                     "beta": 0.0, "pn_order": 0, "snr": snr})

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
        dev_kind = cfg["deviation"].get("kind", "amplitude")
        if dev_kind == "phase":
            from src.data.inject_deviations import inject_phase_modulation
            hp_mod = inject_phase_modulation(hp, delta_t, beta, pn_order=pn_order)
        elif dev_kind == "frequency":
            from src.data.inject_deviations import inject_frequency_modulation
            hp_mod = inject_frequency_modulation(hp, delta_t, beta, pn_order=pn_order)
        else:
            from src.data.inject_deviations import inject_amplitude_modulation
            hp_mod = inject_amplitude_modulation(hp, delta_t, beta, pn_order=pn_order)
        snr = float(rng.uniform(snr_lo, snr_hi)) if use_noise else 50.0
        if use_noise:
            start = file_idx * n_samples
            noise_seg = noise_pool[start:start + n_samples]
        else:
            noise_seg = None
        file_idx += 1

        x = build_one_sample(hp_mod, noise_seg, delta_t, cfg, n_samples, FIXED_SCALE)
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
    p.add_argument("--noise_source", type=str, default="real",
                   choices=["real", "synthetic"])
    args = p.parse_args()

    cfg = load_config(args.config)
    n = args.n_per_class or cfg["dataset"]["n_per_class"]
    out = args.out or cfg["paths"]["data_processed"]

    snr_range = None
    if args.snr_min is not None and args.snr_max is not None:
        snr_range = (args.snr_min, args.snr_max)

    build_dataset(n, out, cfg, snr_range=snr_range, noise_source=args.noise_source)


if __name__ == "__main__":
    main()
