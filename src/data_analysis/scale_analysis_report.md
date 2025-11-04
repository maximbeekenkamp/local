# CDON Dataset Analysis Report
## Dataset Overview
- Train samples: 100, shape: [100, 4000]
- Test samples: 44, shape: [44, 4000]
- Timesteps: 4000

## Global Statistics

| Field | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| Loads (Acceleration) | -1.240429e-06 | 0.014334 | -0.313128 | 0.301136 |
| Responses (Displacement) | -2.457121e-05 | 0.037996 | -0.508845 | 0.588812 |

**Scale Ratio (response_std / load_std):** 2.6508

## Scale Consistency Analysis

| Metric | Value |
|--------|-------|
| Mean per-sample scale ratio | 2.2809 |
| Std of scale ratios | 1.1023 |
| Min scale ratio | 0.2433 |
| Max scale ratio | 5.3144 |
| Consistency | Consistent |

## Per-Sample Statistics Summary

Summary of per-sample variability (not individual rows):

| Statistic | Load Mean | Load Std | Load Min | Load Max | Response Mean | Response Std | Response Min | Response Max |
|-----------|-----------|----------|----------|----------|---------------|--------------|--------------|--------------|
| Mean | -1.240429e-06 | 0.011022 | -0.067912 | 0.066606 | -2.457121e-05 | 0.026401 | -0.090292 | 0.090401 |
| Std | 1.216227e-05 | 0.009164 | 0.058989 | 0.060076 | 6.759929e-05 | 0.027326 | 0.096102 | 0.097348 |
| Min | -8.706073e-05 | 0.000203 | -0.313128 | 0.001178 | -2.015303e-04 | 0.000275 | -0.508845 | 0.000984 |
| Max | 4.413297e-05 | 0.042765 | 0.301136 | 0.301136 | 2.742082e-04 | 0.151614 | -0.001324 | 0.588812 |

**Interpretation:** Samples show high variability in statistics across the dataset.

## Normalization Recommendation

**Strategy:** Global z-score normalization
- Load normalization: (x - load_mean) / load_std
- Response normalization: (y - response_mean) / response_std

**Rationale:** Global normalization ensures consistent scaling across all samples for neural operator training. Min/Max values provided for reference only.
