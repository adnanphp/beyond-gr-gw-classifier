# GW150914 Amplitude-Controlled Experiment

## Motivation
The original GW150914 experiment achieved 100% accuracy, but the
two classes differed by 4.6× in mean amplitude. This could allow
the classifier to exploit an amplitude shortcut.

## Fix
Normalize every final strain so that the overall std is 1.0 for
both classes. This removes the amplitude scale difference while
preserving the shape of the modulation.

## Verification
After normalization:
- GR:  mean|abs| = 0.7801 ± 0.0004, std = 1.0000
- mod: mean|abs| = 0.6529 ± 0.0177, std = 1.0000

The classes now differ by 16% in mean|abs| — a genuine shape
difference, not an amplitude scale difference.

## Results
| Split | Accuracy | ROC-AUC |
|---|---|---|
| train (280) | 1.0000 | 1.0000 |
| val (60) | 1.0000 | 1.0000 |
| test_seen (60) | 1.0000 | 1.0000 |
| test_unseen (60) | 1.0000 | 1.0000 |

## Interpretation
The classifier achieves 100% accuracy even after amplitude
normalization. This demonstrates that it detects the SHAPE of the
amplitude modulation (β = 5-20), not just its overall scale.

## Caveat
β = 5-20 corresponds to 500-2000% amplitude changes, which is
much larger than realistic Beyond-GR deviations (typically
β ~ 0.001-0.01). Further work should test sensitivity at
physically realistic β values.

## Terminology Note
This is not "modifying the historical GW150914 detection." It is
using GW150914-derived strain as a template and injecting
counterfactual modifications into real H1 detector noise.
