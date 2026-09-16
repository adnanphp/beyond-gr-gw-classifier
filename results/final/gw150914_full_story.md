# GW150914 Experiment — Full Story

## Phase 1: Uncontrolled (initial test)
- Classes differ by 4.6× in mean|abs|
- Classifier achieves 100%
- **Problem:** amplitude shortcut possible

## Phase 2: Amplitude-Controlled
- Both classes normalized to std = 1.0
- β = [5, 20]: classes differ by 16% in mean|abs|
- β = [0.5, 2.0]: classes differ by 2% in mean|abs|
- Classifier still achieves 100%

## Interpretation
After removing the amplitude scale shortcut, the classifier maintains
100% accuracy. This demonstrates sensitivity to the SHAPE of the
amplitude modulation, likely via tail statistics (max, p99, p999)
rather than mean/std.

## Caveats
1. β = 0.5-2.0 corresponds to 50-200% amplitude changes — much
   larger than realistic Beyond-GR deviations.
2. The classifier uses the "amplitude modulation" shape, but a
   different deviation type (phase, frequency) would be a stronger
   test.
3. Using GW150914 as a template does not test the historical event;
   it uses the strain as a GR reference.

## Terminology
This is NOT "modifying the historical GW150914 detection." It is
using GW150914-derived strain as a template and injecting
counterfactual modifications into real H1 detector noise.
