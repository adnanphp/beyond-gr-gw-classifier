"""Preprocessing with whitening, bandpass, and clipping."""
import numpy as np
from scipy.signal import butter, filtfilt


def bandpass(x, fs, low=20.0, high=500.0, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def preprocess(strain, psd, delta_t, fs, cfg):
    """Whiten → bandpass → clip → fixed scale."""
    x = np.asarray(strain, dtype=np.float64)
    scale = cfg["preprocessing"].get("fixed_scale", 1.0)

    # Whitening if PSD is provided
    if cfg["preprocessing"].get("whiten", False) and psd is not None:
        strain_f = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), d=delta_t)
        psd_interp = np.interp(freqs, psd[0], psd[1])
        psd_interp[psd_interp <= 0] = np.inf
        whitened_f = strain_f / np.sqrt(psd_interp / (4.0 * delta_t))
        x = np.fft.irfft(whitened_f, n=len(x))

    # Bandpass
    if fs is not None:
        x = bandpass(x, fs, low=20.0, high=500.0)

    # Fixed scale
    x = x / scale

    # CLIP extreme values (glitches in real LIGO data)
    clip_value = cfg["preprocessing"].get("clip_value", 50.0)
    x = np.clip(x, -clip_value, clip_value)

    # Remove any NaN/Inf
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)

    return x.astype(np.float32)
