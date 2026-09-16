"""Inject signal into noise at target SNR."""
import numpy as np
import pycbc.types
from pycbc.filter import sigma
from pycbc.psd import aLIGOZeroDetHighPower


def get_psd(n_samples, delta_t=1.0 / 4096, f_lower=20.0):
    """
    Get aLIGO PSD with correct delta_f.

    Note: aLIGOZeroDetHighPower expects delta_f (Hz), not delta_t (s).
    """
    delta_f = 1.0 / (n_samples * delta_t)
    psd_len = n_samples // 2 + 1
    return aLIGOZeroDetHighPower(psd_len, delta_f, f_lower)


def inject_at_snr(signal, noise, target_snr,
                  delta_t=1.0 / 4096, f_lower=20.0, verbose=False):
    n_samples = len(noise)
    signal = np.asarray(signal).flatten()

    if len(signal) < n_samples:
        signal = np.concatenate([np.zeros(n_samples - len(signal), dtype=signal.dtype), signal])
    elif len(signal) > n_samples:
        signal = signal[-n_samples:]

    psd = get_psd(n_samples, delta_t, f_lower)

    if verbose:
        print(f"signal len={len(signal)}, psd delta_f={psd.delta_f}")

    signal_ts = pycbc.types.TimeSeries(signal, delta_t=delta_t)

    if verbose:
        print(f"signal_ts delta_f={signal_ts.delta_f}")

    snr_opt = float(sigma(signal_ts, psd, low_frequency_cutoff=f_lower))

    if snr_opt <= 0:
        raise ValueError(f"Optimal SNR is {snr_opt}")

    scale = target_snr / snr_opt
    return np.asarray(noise) + scale * signal, scale
