# N=200 vs N=1000 Comparison

## Real LIGO Noise, β = 0.3-0.5, SNR 10-15
| N | test_seen | test_unseen |
|---|---|---|
| 200 | 0.9184 | 0.8910 |
| 1000 | 0.9044 | 0.8896 |
**Conclusion:** Stable. The 89% detection accuracy at β ≈ 0.4 is robust.

## Real LIGO Noise, β = 5-20, SNR 10-15
| N | test_seen | test_unseen |
|---|---|---|
| 200 | 0.9796 | 0.9679 |
| 1000 | 0.9801 | 0.9940 |
**Conclusion:** Improved with N. Error rate dropped from 3.2% to 0.6%.

## Synthetic Noise, β = 0.3-1.0, SNR 15-20
| N | test_seen | test_unseen |
|---|---|---|
| 200 | 0.7551 | 0.8462 |
| 1000 | 0.9920 | 0.9955 |
**Conclusion:** N=200 severely underfit. With N=1000, accuracy near 100%.

## Key Finding
- Real noise threshold at SNR 10-15: β ≈ 0.4 (stable across N)
- Real noise strong deviation detection: 99%+ with sufficient data
- Synthetic noise: better than N=200 suggested
