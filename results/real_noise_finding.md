# Real LIGO Noise Result

## Configuration
- Noise: Real H1 strain from GWOSC (GPS 1126259446)
- SNR: 30-50
- Deviation: amplitude modulation, β = [5, 20]
- Model: CNN1D + 6 statistics features, 31,697 params
- Training: 62 epochs, lr=0.003, early stopping at best val_acc=1.0

## Results
| Split | Accuracy | ROC-AUC |
|---|---|---|
| train | 0.9837 | 0.9985 |
| val | 1.0000 | 1.0000 |
| test_seen | 1.0000 | 1.0000 |
| test_unseen | 0.9744 | 0.9988 |

## Key Finding
Real LIGO noise + strong amplitude deviations (β = 5-20) → 97.4% on unseen
deviation type C. This demonstrates that:
1. Real LIGO noise can be handled by CNN+stats when deviations are strong
2. The model generalizes to unseen deviation types even in real detector noise
3. β = 5-20 is above the detection threshold for this SNR

## Comparison with Synthetic
| Noise | SNR | β | test_unseen |
|---|---|---|---|
| Synthetic | 30-50 | 0.3-1.0 | 100% |
| Real LIGO | 30-50 | 5.0-20.0 | 97.4% |

Real noise requires ~5-20× stronger deviations than synthetic noise
at the same SNR, reflecting the non-Gaussian structure of real detector noise.
