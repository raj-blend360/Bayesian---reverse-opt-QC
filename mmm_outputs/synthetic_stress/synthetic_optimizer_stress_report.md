# Synthetic MMM Optimizer Stress Report

## Run configuration

- Seed: `42`
- Periods: `80` weekly observations
- Posterior draws used to build synthetic model: `48`
- Optimizer samples per case: `12`

## Synthetic dataset

- Rows: `80`
- Media channels: `spends_search, spends_social, spends_tv, spends_display`
- Revenue range: `251.51` to `414.45`

## Summary

- PASS: `5`
- WARN: `1`
- FAIL: `0`

## Case results

| case | status | period | budget | target | forward_response | forward_convergence_pct | forward_bounds_respected | reverse_spend | reverse_achieved | reverse_achievable | reverse_convergence_pct | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_mean | PASS | training_mean | 39,472.4225 | 45.0000 | 84.5483 | 100.0000 | True | 23,373.4000 | 45.0593 | True | 91.7000 |  |
| under_budget | PASS | training_mean | 29,604.3168 | 40.5000 | 51.6208 | 100.0000 | True | 21,199.9300 | 40.6999 | True | 91.7000 |  |
| growth_budget | PASS | monthly | 205,967.1004 | 215.2425 | 325.9479 | 91.7000 | False | 104,399.7800 | 215.2425 | True | 100.0000 |  |
| tight_bounds | PASS | quarterly | 514,917.7510 | 557.6737 | 709.0873 | 100.0000 | True | 330,835.6100 | 548.8602 | True | 75.0000 |  |
| stretch_year | PASS | yearly | 2,368,621.6544 | 2,817.7200 | 3,882.9800 | 100.0000 | True | 1,390,853.0200 | 2,817.7200 | True | 100.0000 |  |
| infeasible_reverse | WARN | training_mean | 17,762.5901 | 101.2500 | 20.2215 | 0.0000 | False | 8,526.0400 | 20.2215 | False | 0.0000 |  |

## Output artifacts

- `synthetic_dataset.csv`
- `stress_results.csv`
- `forward_<case>.csv` and `reverse_<case>.csv` for each scenario

## Interpretation

Warnings are expected for intentionally constrained/infeasible scenarios; failures indicate exceptions in the optimizer path.