# Optimizer Module — `optimisation.py`

Budget optimisation using the fitted Bayesian MMM posterior. Four optimisers
are available, covering every combination of direction (forward / reverse) and
algorithm (gradient-based SLSQP / greedy mROI / sequential multi-period).

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [CPP layer — cost-per-unit conversion](#2-cpp-layer)
3. [Shared internals](#3-shared-internals)
4. [Forward optimiser — `optimise_budget`](#4-forward-optimiser)
5. [Reverse optimiser — `minimise_spend_for_target`](#5-reverse-optimiser)
6. [Greedy mROI allocator — `greedy_budget_allocation`](#6-greedy-mroi-allocator)
7. [Sequential multi-period — `optimise_budget_sequential`](#7-sequential-multi-period-optimiser)
8. [Reverse sequential — `minimise_spend_sequential`](#8-reverse-sequential-optimiser)
9. [Budget period reference](#9-budget-period-reference)
10. [Output column glossary](#10-output-column-glossary)
11. [Typical call patterns](#11-typical-call-patterns)
12. [Known limitations & design notes](#12-known-limitations--design-notes)

---

## 1. Architecture overview

```
optimise_budget()                 ← SLSQP · forward  · single-period
minimise_spend_for_target()       ← SLSQP · reverse  · single-period
greedy_budget_allocation()        ← greedy mROI · forward · single-period
optimise_budget_sequential()      ← greedy mROI · forward · multi-period (carry-forward)
minimise_spend_sequential()       ← binary search over optimise_budget_sequential
```

All optimisers share the same parameter-extraction and beta-calibration pipeline,
run over posterior samples (not just the mean), and return DataFrames with mean
and HDI credible-interval columns.

### Response-function chain

```
raw spend  →  /spend_max  →  scaled spend
    →  adstock (geometric steady-state or period-by-period Weibull)
    →  saturation (Hill / softplus / logistic / identity)
    →  × beta
    →  × rscale (transforms model-space units back to original KPI units)
```

`rscale` is computed from the posterior via `_compute_response_scale_from_trace`
(log1p path) or falls back to `E[y_raw] / E[transform(y_raw)]` for other transforms.

---

## 2. CPP layer

The CPP (cost-per-unit) layer converts between raw media metrics (impressions,
clicks, GRPs, reach) and £ spend. This enables the optimiser to work entirely in
£ spend space — which is numerically stable — regardless of the metric type of
each channel.

### Build the CPP map

```python
from optimisation import build_cpp_map, cpp_weights_array

cpp_map  = build_cpp_map(prep)                          # auto-computed from data
cpp_w    = cpp_weights_array(spend_cols, cpp_map)       # (C,) array for optimiser
```

You can override any channel's CPP in your YAML config:

```yaml
optimisation:
  cpp_rates:
    media_impressions_clara_dv360: 0.00412   # £ per impression
    media_impressions_clara_meta:  0.00380
```

### When CPP is not provided

If `cpp_weights=None` is passed to any optimiser, all channels are treated as if
their media metric equals spend (CPP = 1). This is correct for channels whose
model column already holds £ spend values.

---

## 3. Shared internals

### Beta calibration (`_calibrate_betas_to_trace`)

Using posterior-mean parameters with a non-linear saturation function
systematically over-predicts response (Jensen's inequality). The calibration step
computes a per-channel correction factor:

```
calib_j = trace_mean_contribution_j / sim_mean_contribution_j
```

and scales each channel's beta samples by `calib_j` before any optimisation.
This ensures that at historical spend levels the optimiser's predicted response
matches the model's posterior predictive mean.

### Posterior uncertainty

All four optimisers run the optimisation **independently for each of `n_samples`
posterior draws**, producing a distribution over optimal allocations. Output tables
report `mean`, `hdi_10`, and `hdi_90` for every spend and response column.

For the greedy optimiser, current_response is now also propagated through the
same `n_samples` posterior draws (as of the latest update), so the comparison
between current and optimised response is apples-to-apples.

---

## 4. Forward optimiser

```python
optimise_budget(
    best, prep,
    total_budget  = None,          # £ for the period; None = observed mean
    budget_period = "training_mean",
    n_samples     = 200,
    channel_min   = None,          # {channel: £ lower bound for the period}
    channel_max   = None,          # {channel: £ upper bound for the period}
    cpp_weights   = None,          # (C,) CPP array; None = treat all as £ spend
) -> pd.DataFrame
```

**Algorithm:** SLSQP gradient descent in £ spend space.
- Objective: maximise `Σ_j response_j(spend_j / CPP_j)`
- Constraint: `Σ_j spend_j = total_budget_per_period` (equality)
- Bounds: per-channel `[lb_spend, ub_spend]`
- Adstock: geometric steady-state `x_ads = x_scaled / (1 - lam)`

**When to use:** Single-period allocation, smooth response curves, many channels.
SLSQP converges fast and handles equality budget constraints cleanly.

**Key output columns** (TOTAL row):

| Column | Meaning |
|---|---|
| `convergence_rate_pct` | % of posterior samples where SLSQP converged |
| `bounds_respected` | True if all samples satisfied bounds within 1% tolerance |
| `min_achievable_response` | Posterior-mean response at all lower bounds |
| `max_achievable_response` | Posterior-mean response at all upper bounds |

---

## 5. Reverse optimiser

```python
minimise_spend_for_target(
    best, prep,
    target_response = None,        # desired total response for the period
    target_period   = "training_mean",
    n_samples       = 200,
    channel_min     = None,
    channel_max     = None,
    budget_cap      = None,        # optional hard ceiling on total spend
    cpp_weights     = None,
) -> pd.DataFrame
```

**Algorithm:** SLSQP in £ spend space.
- Objective: minimise `Σ_j spend_j` (total £ spend)
- Constraint: `response(spend / CPP) ≥ target_pp_model` (inequality)
- Optional constraint: `Σ_j spend_j ≤ budget_cap`

**When to use:** Client has set a deals/leads target and needs the minimum
budget (and allocation) to hit it.

**Key output columns** (TOTAL row):

| Column | Meaning |
|---|---|
| `target_response` | The target passed in (original KPI units for the period) |
| `target_achievable` | Whether the target falls within the achievable range |
| `achieved_response_mean` | Average response achieved across posterior samples |

---

## 6. Greedy mROI allocator

```python
df_alloc, df_path = greedy_budget_allocation(
    best, prep,
    total_budget  = None,
    budget_period = "training_mean",
    step_size     = None,          # £ per greedy step; None = 1% of total_budget
    n_samples     = 200,
    channel_min   = None,
    channel_max   = None,
    cpp_weights   = None,
    output_path   = None,          # path to .xlsx; writes 'greedy_path' sheet if set
) -> (pd.DataFrame, pd.DataFrame)
```

**Algorithm:** Iterative mROI greedy in £ spend space.

At each step:
1. Compute `mROI_j = ΔResponse_j / Δspend_j` for every channel.
2. Allocate the full `step_size` to the channel with the highest mROI.
3. **Remainder routing:** if the chosen channel hits its upper bound before the
   full step is consumed, the leftover is routed to the next-best channel in
   the same step — no budget is lost.
4. Channels at their upper bound are excluded.
5. Repeat until budget is exhausted.

The allocation **path** is computed once using posterior-mean parameters
(for a clean, interpretable waterfall). Posterior uncertainty is then
propagated by running the same greedy loop over `n_samples` posterior draws.

**`df_alloc` — per-channel summary**

New columns added in latest update:

| Column | Meaning |
|---|---|
| `current_response_hdi_10/90` | P10/P90 of current response across posterior samples |
| `unallocated_budget_£` | Budget not placed (TOTAL row only) — non-zero when all channels hit upper bounds |
| `allocation_rate_pct` | `actual_spend / total_budget × 100` (TOTAL row only) |

**`df_path` — step-by-step waterfall**

| Column | Meaning |
|---|---|
| `step` | Step index (0 = start) |
| `channel_chosen` | Which channel received budget at this step |
| `mroi_chosen_£` | Marginal ROI of the chosen channel at the time of selection |
| `spend_£_{ch}` | Cumulative £ spend for each channel after this step |
| `cumulative_response` | Total response (original KPI units) after this step |

**Saving the path to Excel:**

```python
df_alloc, df_path = greedy_budget_allocation(
    best, prep,
    total_budget = 50_000,
    output_path  = "Bayesian_Output/model_results/model_results.xlsx",
)
# Sheet 'greedy_path' is written/replaced in the workbook automatically.
```

Or call the helper directly:

```python
from optimisation import save_greedy_path_to_excel
save_greedy_path_to_excel(df_path, "path/to/model_results.xlsx")
```

**Step size guidance:**

| `step_size` as % of budget | Precision | Runtime |
|---|---|---|
| 5% | Coarse | Fast |
| 1% (default) | Good | Moderate |
| 0.1% | High | Slow |

---

## 7. Sequential multi-period optimiser

```python
df_summary, df_channels, df_monthly = optimise_budget_sequential(
    best, prep,
    total_budget       = None,     # total £ for all n_months; None = observed
    n_months           = 1,        # number of months to optimise over
    n_samples          = 1,        # posterior draws (set ≥50 for valid HDI bands)
    channel_min        = None,     # {channel: £ absolute lower bound} whole period
    channel_max        = None,     # {channel: £ absolute upper bound} whole period
    channel_share_min  = None,     # {channel: fraction 0–1} min share of monthly budget
    channel_share_max  = None,     # {channel: fraction 0–1} max share of monthly budget
    cpp_weights        = None,
    target_cpa         = None,     # £/deal — drives quality weight in composite mROI
    step_size          = None,     # £/month; None = 1% of monthly budget
    round_budget_pct   = None,     # dynamic step: each step = X% of remaining budget
    slope_scaling      = 1e4,      # second-derivative penalty weight in composite score
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
```

**Key differences from `greedy_budget_allocation`:**

1. **Carry-forward:** adstock carry propagates from month to month. Warm-start
   carry is initialised at the geometric steady-state for the current observed
   spend so month-1 response is not understated.

2. **Incremental mROI:** response is evaluated as `beta × (sat(new+carry) - sat(carry))`
   so only the NEW signal's contribution is scored, not the carried-over baseline.

3. **Composite mROI score:**
   ```
   score = (mROI / max_mROI) × quality × slope_weight
   quality      = mROI / (mROI + mroi_floor)   # penalises poor CPA channels
   slope_weight = 1 / (1 + |d²R/dS² × slope_scaling|)  # penalises fast-saturating channels
   ```
   Set `target_cpa` to activate the quality penalty. Set `slope_scaling=0` to
   fall back to pure mROI greedy (theoretically optimal for concave curves).

4. **BAU simulation:** each month also simulates the current channel mix rescaled
   to the same monthly budget, giving a fair "do-nothing" benchmark.

5. **Dynamic step size (`round_budget_pct`):** instead of a fixed `step_size`,
   each step receives `round_budget_pct × remaining_unallocated_budget`. This
   produces smoother allocation and avoids step-size sensitivity, particularly
   useful when budgets vary across scenarios. Typical values: 0.01–0.05.

> **Warning:** `n_samples=1` (the default) produces HDI bands with near-zero
> variance. Set `n_samples ≥ 50` for meaningful uncertainty quantification.
> The allocation path always uses posterior-mean parameters regardless of
> `n_samples`.

**Returns:**

| DataFrame | Content |
|---|---|
| `df_summary` | 1-row overall: opt vs BAU vs current, total spend, total response, ROI, uplift % |
| `df_channels` | Per-channel: opt / BAU / current spend & response, ROI, constraint status |
| `df_monthly` | Month-by-month: opt vs BAU vs current response and spend per channel |

---

## 8. Reverse sequential optimiser

```python
df_summary, df_channels, df_monthly = minimise_spend_sequential(
    best, prep,
    target_response  = None,     # desired avg monthly response; None = observed
    n_months         = 1,
    tol_pct          = 0.02,     # convergence tolerance as fraction of target
    max_iter         = 10,       # binary-search iterations
    # … same kwargs as optimise_budget_sequential …
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
```

**Algorithm:** Binary search over `total_budget`, calling
`optimise_budget_sequential` at each candidate until the achieved average
monthly response is within `tol_pct` of the target.

`df_summary` gains three extra columns: `minimum_monthly_budget_£`,
`target_response_per_month`, and `achieved_response_per_month`.

---

## 9. Budget period reference

| `budget_period` | Meaning |
|---|---|
| `"per_period"` | One model period (1 week if weekly data, 1 day if daily) |
| `"training_mean"` | Same as `"per_period"` — use when budget is a typical week/day |
| `"monthly"` | ~4.35 weeks / ~30.44 days |
| `"quarterly"` | ~13 weeks / ~91 days |
| `"yearly"` | ~52 weeks / ~365 days |
| `"training_total"` | The full training window |

Budget figures, `channel_min`, and `channel_max` are all interpreted in the
units of the chosen period.

---

## 10. Output column glossary

| Column | Applies to | Meaning |
|---|---|---|
| `current_spend_£` | all | Observed mean £ spend for the period |
| `optimal_spend_£_mean` | forward SLSQP | Mean recommended £ spend across posterior samples |
| `greedy_spend_£_mean` | greedy | Mean recommended £ spend across posterior samples |
| `opt_spend_gbp` | sequential | Recommended £ spend (posterior-mean path) |
| `current_response` | all | Mean response at observed spend (posterior samples) |
| `current_response_hdi_10/90` | greedy | P10/P90 of current response across samples |
| `response_mean` / `greedy_response_mean` | SLSQP / greedy | Mean optimal response across samples |
| `*_hdi_10` / `*_hdi_90` | all | 10th / 90th percentile across posterior samples |
| `pct_change_spend` | all | % change from current to recommended spend |
| `pct_change_response` | all | % change from current to recommended response |
| `current_roi` | all | Response per £ at current spend |
| `optimal_roi` / `greedy_roi` | SLSQP / greedy | Response per £ at recommended spend |
| `current_cpa` | all | £ per unit response at current spend |
| `optimal_cpa` / `greedy_cpa` | SLSQP / greedy | £ per unit response at recommended spend |
| `constraint_status` | sequential | "At min bound" / "At max bound" / "Unconstrained" |
| `unallocated_budget_£` | greedy TOTAL | Budget not placed (channels at upper bounds) |
| `allocation_rate_pct` | greedy TOTAL | `actual_spend / total_budget × 100` |
| `convergence_rate_pct` | SLSQP | % of samples where SLSQP converged |
| `response_uplift_vs_bau_pct` | sequential | Opt response vs BAU at same budget |

---

## 11. Typical call patterns

### Forward allocation at current budget

```python
from optimisation import build_cpp_map, cpp_weights_array, optimise_budget

cpp_map = build_cpp_map(prep)
cpp_w   = cpp_weights_array(prep["spend_cols"], cpp_map)

df = optimise_budget(
    best, prep,
    budget_period = "monthly",
    cpp_weights   = cpp_w,
)
print(df[["channel", "current_spend_£", "optimal_spend_£_mean", "pct_change_spend",
          "current_response", "response_mean", "pct_change_response", "optimal_roi"]])
```

### Greedy with path export to model_results

```python
from optimisation import greedy_budget_allocation

df_alloc, df_path = greedy_budget_allocation(
    best, prep,
    total_budget  = 50_000,
    budget_period = "monthly",
    step_size     = 500,
    cpp_weights   = cpp_w,
    output_path   = "Bayesian_Output/model_results/model_results.xlsx",
)
```

### Multi-month sequential with dynamic step

```python
from optimisation import optimise_budget_sequential

df_summary, df_channels, df_monthly = optimise_budget_sequential(
    best, prep,
    total_budget      = 600_000,   # 12 months × £50k
    n_months          = 12,
    n_samples         = 50,        # set ≥50 for meaningful HDI bands
    round_budget_pct  = 0.02,      # dynamic step: 2% of remaining budget each step
    target_cpa        = 800,       # £800/deal quality floor
    channel_share_min = {"media_impressions_clara_dv360": 0.05},
    channel_share_max = {"media_impressions_clara_google": 0.70},
    cpp_weights       = cpp_w,
)
print(df_summary[["opt_total_response", "bau_total_response",
                   "response_uplift_vs_bau_pct", "opt_roi"]])
print(df_monthly[["month", "opt_response", "bau_response", "opt_total_spend_£"]])
```

### Reverse sequential — minimum budget to hit a deals target

```python
from optimisation import minimise_spend_sequential

df_summary, df_channels, df_monthly = minimise_spend_sequential(
    best, prep,
    target_response = 600,    # 600 deals/month average
    n_months        = 3,
    n_samples       = 50,
    round_budget_pct = 0.02,
    cpp_weights     = cpp_w,
    tol_pct         = 0.02,   # converge within 2% of target
)
print(df_summary[["minimum_monthly_budget_£", "target_response_per_month",
                   "achieved_response_per_month"]])
```

---

## 12. Known limitations & design notes

**Steady-state vs period-by-period adstock**
`optimise_budget` (SLSQP) uses the geometric adstock steady-state approximation
`x_ads = x_scaled / (1 - lam)`, which is exact only for constant spend across
all periods. `optimise_budget_sequential` simulates period-by-period and is
more accurate for multi-month planning. Do not compare raw response numbers
between the two directly.

**`slope_scaling` is scale-sensitive**
The second-derivative penalty in the composite mROI score (`slope_scaling=1e4`)
needs tuning to the scale of your response variable and spend. If the penalty has
no visible effect on results, reduce `slope_scaling`. Set `slope_scaling=0` for
pure mROI greedy, which is theoretically optimal for concave response functions.

**Beta calibration covers softplus and Hill only**
The `_calibrate_betas_to_trace` loop reimplements saturation for these two types.
Channels using `logistic` saturation use an identity approximation in the
calibration step, which may slightly over- or under-correct their betas.

**Greedy path vs SLSQP**
For concave, separable response curves both algorithms should converge to the
same optimum as `step_size → 0`. If results diverge significantly (> 5%),
reduce `step_size` first before investigating further.
