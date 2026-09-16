"""Generate GR waveforms using PyCBC."""
import numpy as np
from pycbc.waveform import get_td_waveform


def generate_gr_waveform(
    mass1=30.0,
    mass2=25.0,
    spin1z=0.0,
    spin2z=0.0,
    distance=500.0,
    inclination=0.0,
    delta_t=1.0 / 4096,
    f_lower=20.0,
    approximant="IMRPhenomD",
):
    hp, hc = get_td_waveform(
        approximant=approximant,
        mass1=mass1, mass2=mass2,
        spin1z=spin1z, spin2z=spin2z,
        distance=distance, inclination=inclination,
        delta_t=delta_t, f_lower=f_lower,
    )
    return np.array(hp), np.array(hc)


def sample_random_params(cfg, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    def uniform(key):
        lo, hi = cfg["waveform"][key]
        return float(rng.uniform(lo, hi))

    m1 = uniform("mass1_range")
    m2 = uniform("mass2_range")
    if m1 < m2:
        m1, m2 = m2, m1

    return {
        "mass1": m1,
        "mass2": m2,
        "spin1z": uniform("spin1z_range"),
        "spin2z": uniform("spin2z_range"),
        "distance": uniform("distance_range"),
        "inclination": uniform("inclination_range"),
    }
