# Results — Clean Data (No Noise)

## Setup
- Waveforms: IMRPhenomD, 30+25 solar masses
- Samples: 400 total (200 GR + 200 beyond-GR)
- Deviation: amplitude modulation, β ∈ [0.3, 1.0]
- Types: A (β ∈ [0.3, 0.6]), B (β ∈ [0.5, 0.8]), C (β ∈ [0.7, 1.0])
- Train: GR + A + B (226 samples)
- Test unseen: GR + C only (154 samples)
- Model: CNN1D, 25,265 parameters
- Training: 26 epochs (early stopped), lr=0.005

## Results (100% everywhere)
| Split | Accuracy | ROC-AUC |
|---|---|---|
| train | 1.0000 | 1.0000 |
| val | 1.0000 | 1.0000 |
| test_seen | 1.0000 | 1.0000 |
| test_unseen | 1.0000 | 1.0000 |

## Interpretation
The CNN perfectly classifies GR vs beyond-GR waveforms, INCLUDING
unseen deviation type C. This demonstrates that amplitude deviations
of 30-100% are trivially detectable without noise.

## Next: noise injection
Rebuild with synthetic aLIGO noise at SNR 8-50 and repeat.
The SNR at which accuracy drops below 0.9 is the detection threshold.
