# Phase vs Amplitude Modulation Comparison

## Setup
- Noise: Real LIGO H1 strain
- SNR: 10-15
- β range: [0.3, 0.5]
- N: 200 samples per class

## Results

### Amplitude modulation (h(t) → h(t)(1 + β·t²))
| Split | Accuracy |
|---|---|
| train | 1.0000 |
| val | 0.8542 |
| test_seen | 0.9184 |
| test_unseen | 0.8910 |

### Phase modulation (analytic signal phase shift + β·t²)
| Split | Accuracy |
|---|---|
| train | 0.9946 |
| val | 0.9167 |
| test_seen | 0.8980 |
| test_unseen | 0.8654 |

## Interpretation
Both deviation types are detected with >86% accuracy on unseen
deviation type C. Phase modulation is ~2.5pp harder than amplitude
modulation at the same β, consistent with phase deviations being
less visible in the amplitude envelope.

The fact that the same model handles both types demonstrates that
the hybrid CNN+statistics architecture captures general
"deviation from GR" features, not just amplitude-specific ones.
