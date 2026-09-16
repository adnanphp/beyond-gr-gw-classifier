"""Inject Beyond-GR deviations — supports amplitude and phase modulation."""
import numpy as np
from scipy.signal import hilbert


def inject_amplitude_modulation(hp, delta_t, beta, pn_order=-1, f_ref=100.0):
    """Amplitude modulation: h(t) * (1 + beta * t_norm^2)."""
    if beta == 0.0:
        return np.asarray(hp).copy()
    hp = np.asarray(hp).astype(np.float64)
    N = len(hp)
    t_norm = np.linspace(0.0, 1.0, N)
    envelope = t_norm ** 2
    return hp * (1.0 + beta * envelope)


def inject_phase_modulation(hp, delta_t, beta, pn_order=-1, f_ref=100.0):
    """Phase modulation: A(t) * cos(phi(t) + beta * t_norm^2)."""
    if beta == 0.0:
        return np.asarray(hp).copy()
    hp = np.asarray(hp).astype(np.float64)
    N = len(hp)
    t_norm = np.linspace(0.0, 1.0, N)
    # Analytic signal
    analytic = hilbert(hp)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    # Apply phase modulation
    phase_mod = phase + beta * (t_norm ** 2)
    return amplitude * np.cos(phase_mod)


# Backwards-compatible alias — default to amplitude
inject_phase_deviation = inject_amplitude_modulation


def sample_beta(cfg, dev_type, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    key = f"type_{dev_type}"
    lo, hi = cfg["deviation"][key]["beta_range"]
    return float(rng.uniform(lo, hi))


def get_pn_order(cfg, dev_type):
    key = f"type_{dev_type}"
    return cfg["deviation"][key]["pn_order"]


def inject_frequency_modulation(hp, delta_t, beta, pn_order=-1, f_ref=100.0):
    """
    Frequency modulation: add a time-dependent phase that grows as t^2.
    This produces an instantaneous frequency shift proportional to beta.

    d(phi)/dt = 2*pi*beta*f_ref * t_norm

    Max frequency shift = 2*pi*beta*f_ref (at merger)
    """
    if beta == 0.0:
        return np.asarray(hp).copy()
    hp = np.asarray(hp).astype(np.float64)
    N = len(hp)

    analytic = hilbert(hp)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))

    # Bounded phase correction: delta_phi(t) = 2*pi*beta*f_ref*delta_t*t_norm^3/3
    t_norm = np.linspace(0.0, 1.0, N)
    # Integral of beta*f_ref*t_norm gives (beta*f_ref/2)*t_norm^2
    # Phase = 2*pi * integral(f dt) = 2*pi * beta*f_ref*delta_t * t_norm^2 / 2
    delta_phase = 2.0 * np.pi * beta * f_ref * delta_t * (t_norm ** 2) / 2.0

    return amplitude * np.cos(phase + delta_phase)
