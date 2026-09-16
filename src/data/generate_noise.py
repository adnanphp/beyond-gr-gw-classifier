"""Fetch real LIGO noise or generate synthetic noise."""
import numpy as np
from pathlib import Path


def fetch_real_noise(ifo="H1", gps_start=1126259446, duration=32,
                     cache_dir="data/raw/noise"):
    cache = Path(cache_dir) / f"{ifo}_{gps_start}_{duration}.npz"
    if cache.exists():
        data = np.load(cache)
        return data["noise"], int(data["fs"])

    from gwpy.timeseries import TimeSeries
    ts = TimeSeries.fetch_open_data(ifo, gps_start, gps_start + duration)
    noise = np.asarray(ts.value, dtype=np.float64)
    fs = int(ts.sample_rate.value)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, noise=noise, fs=fs)
    return noise, fs


def generate_synthetic_noise(n_samples, delta_t=1.0 / 4096, f_lower=20.0):
    """Generate colored Gaussian noise from aLIGO PSD.

    Note: aLIGOZeroDetHighPower expects delta_f (Hz), not delta_t (s).
    """
    from pycbc.noise import noise_from_psd
    from pycbc.psd import aLIGOZeroDetHighPower

    delta_f = 1.0 / (n_samples * delta_t)
    psd_len = n_samples // 2 + 1
    psd = aLIGOZeroDetHighPower(psd_len, delta_f, f_lower)

    noise = noise_from_psd(n_samples, delta_t, psd)
    return np.asarray(noise)
