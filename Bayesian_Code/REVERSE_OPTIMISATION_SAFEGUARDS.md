# Reverse Optimisation Safeguards

This note explains the safeguards added around the reverse sequential optimiser so
that optimisation runs fail gracefully instead of appearing to run forever.

## Where reverse sequential optimisation runs

Reverse sequential optimisation is implemented in `optimisation.py` by
`minimise_spend_sequential()`. It searches for the minimum monthly budget that
can reach a target response by repeatedly calling `optimise_budget_sequential()`.

The flow is:

1. Work out the observed monthly spend and the response target.
2. Set a low monthly budget bound and a high monthly budget bound.
3. Run an achievability check at the high bound.
4. If the high bound can reach the target, binary-search between the low and
   high bounds.
5. Each binary-search candidate calls the forward sequential optimiser to find
   the best allocation for that candidate budget.

## Why the loop could previously hang

The forward sequential optimiser supports dynamic greedy allocation via
`round_budget_pct`. In that mode, each greedy step allocates a percentage of the
remaining budget.

If channel caps, share constraints, or channel headroom prevent any channel from
accepting more spend, the greedy step can return the same allocation it received.
Previously, the remaining budget would not decrease, so the condition controlling
that inner loop stayed true forever.

That inner loop is used by reverse sequential optimisation, so one no-progress
allocation state could make a reverse run look like an infinite loop.

## Safeguards now in place

### 1. No-progress detection

During dynamic allocation, the optimiser now stores the previous allocation and
remaining spend before each greedy step. After the step, it recomputes remaining
spend and stops if either:

- the allocation is effectively unchanged; or
- remaining spend has not decreased by more than a tiny tolerance.

When this happens, a warning is logged with the amount of monthly spend that
could not be allocated under the current constraints.

### 2. Dynamic-step hard cap

Dynamic allocation also has a maximum step count. If the loop exceeds that cap,
it stops and logs a warning. This protects the process even if a future edge case
is not caught by the no-progress check.

The cap scales with `round_budget_pct`, so smaller percentage steps are allowed
more iterations while still remaining bounded.

### 3. Achievability pre-check

Before binary search, reverse sequential optimisation now runs the high-budget
case once. If that high budget still cannot reach the target response, the target
is considered unreachable under the current channel bounds/share constraints.

In that case, the optimiser returns the high-budget best attempt immediately
instead of running repeated binary-search iterations that cannot succeed.

### 4. Progress logging

Reverse sequential binary-search iterations are now logged at info level. This
makes long-running optimisations easier to monitor because each candidate budget,
achieved response, target response, and gap is visible in the logs.

## What to check if warnings appear

If you see a warning about unallocated spend or an unreachable target, check:

- `channel_max` values;
- `channel_share_max` values;
- whether `target_response` is above what the model can produce;
- whether `round_budget_pct` is very small;
- whether the high search bound of 3× observed monthly spend is appropriate for
  your planning scenario.

## Practical configuration advice

For routine planning runs:

- Keep `n_samples: 1` for sequential scenarios unless uncertainty bands are
  required.
- Use a moderate `round_budget_pct`, such as `0.05`, for faster dynamic
  allocation.
- If you need a very high response target, first run a forward scenario with a
  high budget to verify that the target is feasible.
- If many channels are tightly capped, expect warnings about unallocated spend;
  that means the constraints, not the optimiser loop, are preventing additional
  allocation.
