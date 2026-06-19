# analysis.py
# ─────────────────────────────────────────────────────────────────────────────
# Converts the raw MCMC posterior samples into business-readable numbers.
#
# Three main outputs:
#   compute_channel_contributions()  — "How much of response came from each channel?"
#                                      Returns mean + 90% HDI per channel.
#   compute_per_product_contributions() — same, broken down by product (multi-product)
#   compute_component_contributions() — "How much came from media vs baseline vs seasonality?"
#
# These feed directly into the exported CSVs and Excel report.
# ─────────────────────────────────────────────────────────────────────────────

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import arviz as az

from priors import compute_return_index, export_return_index  # noqa: F401 re-export

logger = logging.getLogger("MMM")


def _extract_posterior(trace, var_name: str) -> np.ndarray:
    """
    Stack chain+draw into a single sample axis and return a numpy array.
    Works for any number of trailing dimensions.
    Returns shape (N_samples, *dims).
    """
    return (
        trace.posterior[var_name]
        .stack(sample=("chain", "draw"))
        .transpose("sample", ...)
        .values
    )


def compute_channel_contributions(
    best : Dict,
    prep : Dict[str, Any],
) -> pd.DataFrame:
    """
    Computes per-channel contribution statistics from the best model's trace.

    Handles both flat (P=1) and multi-product (P>1) models.  For multi-product,
    channel contributions are *summed* across products so the output is always
    one row per channel, comparable across runs.

    Returns a DataFrame with one row per channel.
    """
    trace      = best["metrics"]["trace"]
    spend_cols = prep["spend_cols"]
    C          = prep["C"]
    P          = prep.get("P", 1)
    train_idx  = prep["train_idx"]
    y_mu       = prep["y_mu"]    # scalar (P=1) or (P,) array (P>1)
    y_std      = prep["y_std"]   # scalar (P=1) or (P,) array (P>1)
    X_media_rw = prep["X_media_raw"][train_idx]  # (T,C) or (T,P,C)

    # media_by_channel: (N_samples, T, C) for P=1 | (N_samples, T, P, C) for P>1
    mbc = _extract_posterior(trace, "media_by_channel")
    # mu posterior: (N_samples, T) for P=1 | (N_samples, T, P) for P>1
    mu_post = _extract_posterior(trace, "mu")

    if P > 1:
        # Aggregate over product axis so downstream logic is identical to P=1.
        # Sum over P: each channel's total impact across all products.
        mbc     = mbc.sum(axis=2)          # (N_samples, T, C)
        mu_post = mu_post.sum(axis=2)      # (N_samples, T)
        y_std_v = float(np.mean(y_std))    # representative scale for back-transform
        y_mu_v  = float(np.mean(y_mu))
        # Sum spend across all products so ROI denominator matches the summed contribution
        X_spend = X_media_rw.sum(axis=1) if X_media_rw.ndim == 3 else X_media_rw
    else:
        y_std_v = float(y_std)
        y_mu_v  = float(y_mu)
        X_spend = X_media_rw

    # ── Convert mu to original response scale ─────────────────
    mu_log    = mu_post * y_std_v + y_mu_v   # (N_samples, T) in transform space
    from data_prep import inverse_response_transform
    total_hat = inverse_response_transform(mu_log, prep)   # original scale
    total_hat = np.maximum(total_hat, 0.0)

    # Channel share of total media effect
    mbc_abs     = np.abs(mbc)
    mbc_abs_sum = mbc_abs.sum(axis=2, keepdims=True) + 1e-12   # (N_samples, T, 1)
    ch_share    = mbc_abs / mbc_abs_sum    # (N_samples, T, C)

    # Media fraction of total predicted response
    media_total_z = mbc.sum(axis=2)        # (N_samples, T)
    media_frac    = np.abs(media_total_z) / (np.abs(mu_post) + 1e-12)
    media_frac    = np.clip(media_frac, 0.0, 1.0)

    # Per-channel original-scale contribution: (N_samples, T, C)
    ch_signups = total_hat[:, :, np.newaxis] * media_frac[:, :, np.newaxis] * ch_share

    rows = []
    for j in range(C):
        ch_samples = ch_signups[:, :, j]          # (N_samples, T)
        ch_mean_t  = ch_samples.mean(axis=1)      # (N_samples,)

        mean_c = float(ch_mean_t.mean())
        std_c  = float(ch_mean_t.std())
        hdi_c  = az.hdi(ch_mean_t, hdi_prob=0.90)
        hdi_lo = float(hdi_c[0])
        hdi_hi = float(hdi_c[1])

        total_samples = ch_signups.mean(axis=1).sum(axis=1)   # (N_samples,)
        share_samples = ch_mean_t / (total_samples + 1e-12) * 100.0
        mean_share    = float(share_samples.mean())

        mean_spend = float(X_spend[:, j].mean())
        roi_proxy  = mean_c / (mean_spend + 1e-12) * 1000.0

        rows.append({
            "channel"           : spend_cols[j],
            "mean_contribution" : round(mean_c,    6),
            "std_contribution"  : round(std_c,     6),
            "hdi_90_low"        : round(hdi_lo,    6),
            "hdi_90_high"       : round(hdi_hi,    6),
            "mean_share_pct"    : round(mean_share, 4),
            "roi_proxy"         : round(roi_proxy,  6),
            "mean_weekly_spend" : round(mean_spend, 2),
        })

    df_ch = pd.DataFrame(rows).sort_values("mean_share_pct", ascending=False)
    df_ch["rank"] = range(1, len(df_ch) + 1)
    return df_ch


def compute_per_product_contributions(
    best : Dict,
    prep : Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """
    Returns per-product channel contributions when P > 1.

    Each row = (product, channel) with contribution stats.
    Returns None when P == 1 (use compute_channel_contributions instead).
    """
    P = prep.get("P", 1)
    if P <= 1:
        return None

    trace         = best["metrics"]["trace"]
    spend_cols    = prep["spend_cols"]
    C             = prep["C"]
    product_names = prep.get("product_names", [f"product_{p}" for p in range(P)])
    y_mu          = prep["y_mu"]    # (P,)
    y_std         = prep["y_std"]   # (P,)
    train_idx     = prep["train_idx"]
    X_media_rw    = prep["X_media_raw"][train_idx]   # (T, P, C)

    # media_by_channel: (N_samples, T, P, C)
    mbc     = _extract_posterior(trace, "media_by_channel")
    mu_post = _extract_posterior(trace, "mu")   # (N_samples, T, P)

    rows = []
    for p_idx, pname in enumerate(product_names):
        mbc_p     = mbc[:, :, p_idx, :]       # (N_samples, T, C)
        mu_p      = mu_post[:, :, p_idx]      # (N_samples, T)
        y_mu_p    = float(y_mu[p_idx])
        y_std_p   = float(y_std[p_idx])
        X_spend_p = X_media_rw[:, p_idx, :]   # (T, C)

        mu_log    = mu_p * y_std_p + y_mu_p
        from data_prep import inverse_response_transform
        total_hat = np.maximum(inverse_response_transform(mu_log, prep), 0.0)

        mbc_abs     = np.abs(mbc_p)
        mbc_abs_sum = mbc_abs.sum(axis=2, keepdims=True) + 1e-12
        ch_share    = mbc_abs / mbc_abs_sum
        media_frac  = np.clip(
            np.abs(mbc_p.sum(axis=2)) / (np.abs(mu_p) + 1e-12), 0.0, 1.0
        )
        ch_signups = total_hat[:, :, np.newaxis] * media_frac[:, :, np.newaxis] * ch_share

        for j in range(C):
            ch_samples = ch_signups[:, :, j]
            ch_mean_t  = ch_samples.mean(axis=1)
            hdi_c = az.hdi(ch_mean_t, hdi_prob=0.90)
            total_s = ch_signups.mean(axis=1).sum(axis=1)
            mean_spend = float(X_spend_p[:, j].mean())
            mean_c     = float(ch_mean_t.mean())
            rows.append({
                "product"           : pname,
                "channel"           : spend_cols[j],
                "mean_contribution" : round(mean_c, 6),
                "std_contribution"  : round(float(ch_mean_t.std()), 6),
                "hdi_90_low"        : round(float(hdi_c[0]), 6),
                "hdi_90_high"       : round(float(hdi_c[1]), 6),
                "mean_share_pct"    : round(float((ch_mean_t / (total_s + 1e-12) * 100.0).mean()), 4),
                "roi_proxy"         : round(mean_c / (mean_spend + 1e-12) * 1000.0, 6),
                "mean_weekly_spend" : round(mean_spend, 2),
            })

    return pd.DataFrame(rows)


def compute_component_contributions(best: Dict) -> pd.DataFrame:
    """
    Component-level attribution: media_total, baseline, seasonality, controls.

    Handles both flat (P=1) and multi-product (P>1) models.  For multi-product,
    media_by_channel is (N_samples, T, P, C); we sum over C then average over P
    to get a representative (N_samples, T) media effect.
    """
    trace = best["metrics"]["trace"]

    def _summary(samples_2d: np.ndarray, name: str, group: str) -> Dict:
        # samples_2d: (N_samples, T) — one posterior draw per row
        s = samples_2d.mean(axis=1)
        h = az.hdi(s, hdi_prob=0.90)
        return {
            "component"      : name,
            "group"          : group,
            "mean_effect"    : float(np.mean(s)),
            "std_effect"     : float(np.std(s)),
            "hdi_90_low"     : float(h[0]),
            "hdi_90_high"    : float(h[1]),
            "mean_abs_effect": float(np.mean(np.abs(s))),
        }

    rows = []

    baseline    = _extract_posterior(trace, "baseline")    # (N_samples, T)
    rows.append(_summary(baseline, "baseline", "non_media"))

    seasonality = _extract_posterior(trace, "seasonality") # (N_samples, T)
    rows.append(_summary(seasonality, "seasonality", "non_media"))

    # media_by_channel: (N_samples, T, C) for P=1 | (N_samples, T, P, C) for P>1
    mbc = _extract_posterior(trace, "media_by_channel")
    if mbc.ndim == 4:
        # P>1: sum over C (axis=3) → (N_samples, T, P), then mean over P (axis=2)
        media_total = mbc.sum(axis=3).mean(axis=2)   # (N_samples, T)
    else:
        media_total = mbc.sum(axis=2)                # (N_samples, T)
    rows.append(_summary(media_total, "media_total", "media"))

    for var, label, group in [
        ("control_effect", "controls",        "non_media"),
        ("base_effect",    "base_variables",  "non_media"),
        ("macro_effect",   "macro_variables", "non_media"),
        ("event_effect",   "events",          "non_media"),
        ("synergy_effect", "synergies",       "media"),
    ]:
        if var in trace.posterior:
            arr = _extract_posterior(trace, var)  # (N_samples, T)
            rows.append(_summary(arr, label, group))

    df = pd.DataFrame(rows)
    total_abs = df["mean_abs_effect"].sum() + 1e-12
    df["share_abs_pct"] = df["mean_abs_effect"] / total_abs * 100.0
    return df.sort_values("share_abs_pct", ascending=False).reset_index(drop=True)


def compute_weekly_decomposition(
    best : Dict,
    prep : Dict[str, Any],
) -> pd.DataFrame:
    """
    Return a weekly table of actual, predicted, and per-component contributions
    on the *original response scale* (e.g. signups).

    Columns
    -------
    date, actual, predicted, pred_lo_90, pred_hi_90,
    <channel_0>, <channel_1>, ..., total_media, baseline, seasonality, sum_check

    Method
    ------
    1. Channel contributions are computed the same way as compute_channel_contributions:
       ch_t = y_hat_t * (|media_z_t| / |mu_z_t|) * (|ch_z_t| / |total_media_z_t|)
       This is consistent with the aggregate Media Contributions summary table.
    2. total_media_t = sum of channel contributions at time t.
    3. non_media_t = y_hat_t - total_media_t  (exact, by subtraction).
    4. baseline_t and seasonality_t split non_media proportionally by their
       absolute z-score magnitudes at each time step.
    By construction sum_check == predicted every week (within floating point).
    """
    from data_prep import inverse_response_transform

    trace    = best["metrics"]["trace"]
    y_raw    = prep["y_raw"]
    y_std    = float(prep["y_std"])
    y_mu     = float(prep["y_mu"])
    dates    = pd.to_datetime(prep["dates"])
    channels = prep["channel_names_unique"]

    mu_post  = _extract_posterior(trace, "mu")            # (N, T)
    mbc      = _extract_posterior(trace, "media_by_channel")  # (N, T, C) or (N, T, P, C)
    baseline = _extract_posterior(trace, "baseline")      # (N, T)
    seasonal = _extract_posterior(trace, "seasonality")   # (N, T)

    if mbc.ndim == 4:
        mbc = mbc.sum(axis=3).mean(axis=2)   # collapse P→(N, T, C)

    mu_log    = mu_post * y_std + y_mu
    y_hat_all = inverse_response_transform(mu_log, prep)  # (N, T)
    pred_mean = y_hat_all.mean(axis=0)
    pred_lo   = np.percentile(y_hat_all, 5,  axis=0)
    pred_hi   = np.percentile(y_hat_all, 95, axis=0)

    # ── Step 1: channel contributions (consistent with aggregate summary) ──
    mbc_abs      = np.abs(mbc)                                         # (N, T, C)
    mbc_abs_sum  = mbc_abs.sum(axis=2, keepdims=True) + 1e-12          # (N, T, 1)
    ch_share     = mbc_abs / mbc_abs_sum                               # (N, T, C)
    media_total_z = mbc.sum(axis=2)                                    # (N, T)
    media_frac   = np.abs(media_total_z) / (np.abs(mu_post) + 1e-12)
    media_frac   = np.clip(media_frac, 0.0, 1.0)
    ch_contribs  = y_hat_all[:, :, np.newaxis] * media_frac[:, :, np.newaxis] * ch_share  # (N,T,C)

    ch_mean    = ch_contribs.mean(axis=0)        # (T, C)
    media_mean = ch_mean.sum(axis=1)             # (T,)  total media per week

    # ── Step 2: non-media = predicted - total_media (exact) ───────────────
    non_media_mean = pred_mean - media_mean      # (T,)

    # ── Step 3: split non-media into baseline vs seasonality ──────────────
    bl_abs = np.abs(baseline.mean(axis=0))       # (T,)
    se_abs = np.abs(seasonal.mean(axis=0))       # (T,)
    total_non_abs = bl_abs + se_abs + 1e-12
    bl_frac = bl_abs / total_non_abs             # (T,)
    se_frac = se_abs / total_non_abs             # (T,)

    baseline_mean   = non_media_mean * bl_frac   # (T,)
    seasonality_mean = non_media_mean * se_frac  # (T,)

    rows = []
    for t in range(len(dates)):
        row = {
            "date"       : dates[t].strftime("%Y-%m-%d"),
            "actual"     : round(float(y_raw[t]),    1),
            "predicted"  : round(float(pred_mean[t]), 1),
            "pred_lo_90" : round(float(pred_lo[t]),   1),
            "pred_hi_90" : round(float(pred_hi[t]),   1),
        }
        for j, ch in enumerate(channels):
            row[ch] = round(float(ch_mean[t, j]), 2)
        row["total_media"]  = round(float(media_mean[t]),      2)
        row["baseline"]     = round(float(baseline_mean[t]),   2)
        row["seasonality"]  = round(float(seasonality_mean[t]),2)
        row["sum_check"]    = round(
            sum(float(ch_mean[t, j]) for j in range(len(channels)))
            + float(baseline_mean[t]) + float(seasonality_mean[t]), 1)
        rows.append(row)

    return pd.DataFrame(rows)


# ─── Response curves (response_curves.py) ────────────────────────────────────

from pathlib import Path as _RCPath
import arviz as _az_rc

_PERIODS_PER_MONTH_RC: dict = {
    "daily":   365.25 / 12.0,
    "weekly":  52.18  / 12.0,
    "monthly": 1.0,
}


def _sat_hill(x: np.ndarray, alpha: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    x_pos = np.maximum(x, 1e-12)
    return x_pos ** alpha / (x_pos ** alpha + kappa ** alpha)


def _sat_softplus(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(alpha * x)) / np.log(2.0)


def _sat_logistic(x: np.ndarray, k: np.ndarray, x0: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _sat_exponential(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-alpha * np.maximum(x, 0.0))


def _apply_saturation(
    x_ads    : np.ndarray,
    sat_type : str,
    alpha    : np.ndarray,
    kappa    : Optional[np.ndarray] = None,
    k_log    : Optional[np.ndarray] = None,
    x0       : Optional[np.ndarray] = None,
) -> np.ndarray:
    if sat_type == "hill":
        return _sat_hill(x_ads, alpha, kappa)
    elif sat_type == "softplus":
        return _sat_softplus(x_ads, alpha)
    elif sat_type == "logistic":
        return _sat_logistic(x_ads, k_log, x0)
    else:
        return _sat_exponential(x_ads, alpha)


def _stack_posterior(trace, var_name: str) -> Optional[np.ndarray]:
    if var_name not in trace.posterior.data_vars:
        return None
    return (
        trace.posterior[var_name]
        .stack(sample=("chain", "draw"))
        .transpose("sample", ...)
        .values
    )


def _scalar_samples(trace, var_name: str, idx: Optional[int] = None) -> Optional[np.ndarray]:
    if idx is not None:
        exact = _stack_posterior(trace, f"{var_name}_{idx}")
        if exact is not None:
            return exact.flatten()

    arr = _stack_posterior(trace, var_name)
    if arr is None:
        return None

    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and idx is not None:
        return arr[:, idx]
    return arr.mean(axis=tuple(range(1, arr.ndim)))


def _get_beta_samples(trace, j: int, C: int) -> np.ndarray:
    b = _scalar_samples(trace, f"beta_ch{j}", idx=None)
    if b is not None:
        return b.flatten()

    betas_arr = _stack_posterior(trace, "betas")
    if betas_arr is not None and betas_arr.ndim == 2:
        return betas_arr[:, j]

    beta_c = _stack_posterior(trace, "beta_c")
    if beta_c is not None and beta_c.ndim == 2:
        return beta_c[:, j]

    beta_pc = _stack_posterior(trace, "beta_pc")
    if beta_pc is not None and beta_pc.ndim == 3:
        return beta_pc[:, :, j].mean(axis=1)

    logger.warning(f"  [response_curves] No beta found for channel {j}; using 1.0")
    ref = _stack_posterior(trace, "intercept")
    N = ref.shape[0] if ref is not None else 500
    return np.ones(N)


def _get_lam_samples(trace, j: int) -> np.ndarray:
    lam_hier = _stack_posterior(trace, "lam_c_hier")
    if lam_hier is not None and lam_hier.ndim == 2:
        return np.clip(lam_hier[:, j], 0.0, 0.9999)

    lam = _scalar_samples(trace, "lam", idx=j)
    if lam is not None:
        return np.clip(lam.flatten(), 0.0, 0.9999)

    ref = _stack_posterior(trace, "intercept")
    N = ref.shape[0] if ref is not None else 500
    logger.warning(f"  [response_curves] No lam found for channel {j}; using 0.5")
    return np.full(N, 0.5)


def _subsample(arr: np.ndarray, n: int) -> np.ndarray:
    N = arr.shape[0]
    if N <= n:
        return arr
    idx = np.linspace(0, N - 1, n, dtype=int)
    return arr[idx]


def _coalesce(arr, default):
    return arr if arr is not None else default


def _cpp_per_channel(prep: Dict[str, Any]) -> List[float]:
    """
    Return cost-per-unit (GBP per impression/click) for each channel.

    For channels modelled on spend directly: cpp = 1.0 (already GBP).
    For channels modelled on impressions/clicks: cpp = mean_weekly_GBP / mean_weekly_metric.
    Used to convert the model's internal metric x-axis to GBP for plotting.
    """
    train_idx      = prep["train_idx"]
    X_raw          = prep["X_media_raw"][train_idx]
    spend_raw_cols = prep.get("spend_raw_cols", [])
    metric_types   = prep.get("metric_types", [])
    df             = prep.get("df")
    C              = prep["C"]
    cpp = []
    for j in range(C):
        mt = (metric_types[j] if j < len(metric_types) else "spend").lower()
        if mt == "spend":
            cpp.append(1.0)
            continue
        mean_metric = float(X_raw[:, j].mean()) if X_raw.ndim == 2 else float(X_raw[:, 0, j].mean())
        if mean_metric < 1e-6:
            cpp.append(0.0)
            continue
        spend_col = spend_raw_cols[j] if j < len(spend_raw_cols) else None
        if spend_col and df is not None and spend_col in df.columns:
            mean_gbp = float(df.iloc[train_idx][spend_col].mean())
        else:
            mean_gbp = 0.0
        cpp.append(mean_gbp / mean_metric)
    return cpp


def compute_response_curves(
    best           : Dict,
    prep           : Dict[str, Any],
    n_spend_points : int = 100,
    n_samples      : int = 400,
    spend_max_mult : float = 1.5,
    curve_type     : str  = "steady_state",
) -> pd.DataFrame:
    """
    Compute posterior response curves for every media channel.

    Parameters
    ----------
    curve_type : 'instantaneous' | 'steady_state'  (default: 'steady_state')
        'instantaneous' — single period, no adstock carry-over.
                          y = beta * sat(x), shows pure saturation shape.
        'steady_state'  — adstock at equilibrium (constant spend forever).
                          For geometric: x_adstocked = x / (1 - lambda).
                          For two-timescale: weighted mix of slow/fast decay.
                          For weibull: same as instantaneous (no analytical
                          steady-state for weibull; carry-over is very short).
                          y shows the sustainable per-period return.

    X-axis: GBP spend (impressions/clicks converted via observed CPM/CPC).
    Y-axis: incremental signups relative to zero spend (starts at 0 for all
            saturation types including softplus).
    """
    trace      = best["trace"]
    spend_cols = prep["spend_cols"]
    C          = prep["C"]
    train_idx  = prep["train_idx"]
    X_media    = prep["X_media_scaled"][train_idx]
    X_raw      = prep["X_media_raw"][train_idx]
    y_std      = float(prep["y_std"])
    y_mu       = float(prep["y_mu"])
    response_transform = prep.get("response_transform", "log1p")
    bl         = prep.get("boxcox_lambda")

    # Cost-per-unit: converts model metric (impressions/clicks) → GBP
    cpp_list = _cpp_per_channel(prep)

    # Baseline response (zero media input) for incremental lift calculation
    from data_prep import inverse_response_transform as _inv_rt
    _zero_z = np.array([[0.0]])
    _baseline_response = float(_inv_rt(_zero_z * y_std + y_mu, prep)[0, 0])

    channel_specs = best.get("channel_specs", {})
    rows = []

    def _infer_sat_type_from_trace(trace, j: int, fallback: str) -> str:
        """
        Auto-detect the saturation type actually used during model training by
        inspecting which parameter variables are present in the posterior.

        Priority:
          1. If kappa_{j} exists  → hill  (Hill needs both alpha and kappa)
          2. If k_logistic_{j} or k_shape_{j} exists → logistic
          3. If alpha_sat_{j} exists alone           → softplus
          4. Else: use the fallback from channel_specs / cfg

        This guards against best_cfg.json being updated after training (e.g.
        saturation changed in config but model not retrained), which causes
        the wrong saturation function to be applied to parameters that were
        fitted under a different form, producing astronomical signups.
        """
        pv = trace.posterior.data_vars
        if f"kappa_{j}" in pv:
            return "hill"
        if f"k_logistic_{j}" in pv or f"k_shape_{j}" in pv:
            return "logistic"
        if f"alpha_sat_{j}" in pv:
            return "softplus"
        return fallback

    for j, ch_name in enumerate(spend_cols):
        spec     = channel_specs.get(j)
        _cfg_sat = spec.saturation   if spec else best["cfg"].saturation
        ads_type = spec.adstock_type if spec else best["cfg"].adstock_type

        # Always verify sat_type against the trace to catch cases where the
        # config was changed after model training (softplus config + Hill trace
        # produces unbounded response curves with trillion-scale signups).
        sat_type = _infer_sat_type_from_trace(trace, j, _cfg_sat)
        if sat_type != _cfg_sat:
            logger.warning(
                f"  [RC] channel {j} ({ch_name}): config says sat_type='{_cfg_sat}' "
                f"but trace has kappa/k variables → using '{sat_type}' (trace wins)."
            )

        # metric scale: max observed value (impressions/clicks/spend)
        if X_raw.ndim == 3:
            metric_max = float(X_raw[:, 0, j].max()) + 1e-8
            x_obs      = X_media[:, 0, j]
        else:
            metric_max = float(X_raw[:, j].max()) + 1e-8
            x_obs      = X_media[:, j]

        # Use the larger of (3× avg spend) or (1.5× max spend) so the current
        # average spend lands in a meaningful part of the curve, not at the far left.
        x_avg_3x = float(x_obs.mean()) * 3.0
        x_max_15 = float(x_obs.max()) * spend_max_mult
        x_max      = max(x_avg_3x, x_max_15)
        spend_grid = np.linspace(0.0, x_max, n_spend_points)

        # GBP conversion: impressions/clicks → £ spend
        cpp_j       = cpp_list[j]                    # GBP per metric unit
        gbp_scale   = metric_max * cpp_j             # max GBP on x-axis

        lam    = _subsample(_get_lam_samples(trace, j), n_samples)
        beta   = _subsample(_get_beta_samples(trace, j, C), n_samples)
        alpha  = _subsample(
            _coalesce(_scalar_samples(trace, "alpha_sat", idx=j), np.ones(lam.shape[0])), n_samples
        )
        kappa  = _subsample(
            _coalesce(_scalar_samples(trace, "kappa", idx=j), np.ones(lam.shape[0])), n_samples
        )
        k_log  = _subsample(
            _coalesce(_scalar_samples(trace, "k_logistic", idx=j), np.ones(lam.shape[0])), n_samples
        )
        x0     = _subsample(
            _coalesce(_scalar_samples(trace, "x0", idx=j), np.zeros(lam.shape[0])), n_samples
        )

        S = min(len(lam), len(beta), len(alpha))
        lam, beta, alpha, kappa, k_log, x0 = (
            lam[:S], beta[:S], alpha[:S], kappa[:S], k_log[:S], x0[:S]
        )

        response_mat = np.empty((S, n_spend_points))

        # Pre-fetch two-timescale samples once (avoids re-fetching per sample)
        if ads_type == "two_timescale" and curve_type == "steady_state":
            rho_slow_all = _subsample(_coalesce(_scalar_samples(trace, "rho_slow_c", idx=j), np.full(S, 0.5)), S)
            rho_fast_all = _subsample(_coalesce(_scalar_samples(trace, "rho_fast_c", idx=j), np.full(S, 0.25)), S)
            w_mix_all    = _subsample(_coalesce(_scalar_samples(trace, "w_mix_f", idx=None), np.full(S, 0.5)), S)
        else:
            rho_slow_all = rho_fast_all = w_mix_all = None

        for si in range(S):
            lam_si = lam[si]

            if curve_type == "instantaneous":
                # Pure saturation shape — no adstock amplification
                x_ads = spend_grid

            elif curve_type == "steady_state":
                if ads_type == "two_timescale":
                    rs = float(rho_slow_all[si])
                    rf = float(rho_fast_all[si])
                    wm = float(w_mix_all[min(si, len(w_mix_all) - 1)])
                    ss = spend_grid / (1 - np.clip(rs, 0, 0.9999) + 1e-12)
                    sf = spend_grid / (1 - np.clip(rf, 0, 0.9999) + 1e-12)
                    x_ads = wm * sf + (1 - wm) * ss
                elif ads_type == "weibull":
                    # Weibull has no closed-form steady-state; short memory,
                    # so single-period ≈ steady-state
                    x_ads = spend_grid
                else:
                    # Geometric: steady-state adstock = x / (1 - lambda)
                    x_ads = spend_grid / (1.0 - np.clip(lam_si, 0.0, 0.9999) + 1e-12)
            else:
                x_ads = spend_grid   # fallback

            sat = _apply_saturation(
                x_ads, sat_type,
                alpha=alpha[si], kappa=kappa[si], k_log=k_log[si], x0=x0[si],
            )
            response_mat[si, :] = beta[si] * sat

        mean_r = response_mat.mean(axis=0)
        hdi_lo = np.percentile(response_mat, 10, axis=0)
        hdi_hi = np.percentile(response_mat, 90, axis=0)

        # Make curves relative to their value at x=0 so they always start at (0, 0).
        # This handles softplus (sat(0)=1, not 0) and any other function with
        # a non-zero intercept at zero input.
        r0     = mean_r[0]
        mean_r_inc = mean_r - r0
        hdi_lo_inc = hdi_lo - r0
        hdi_hi_inc = hdi_hi - r0

        # Convert z-space incremental response to original-scale signups:
        #   signups(delta_z) = inv_transform(delta_z * y_std + y_mu) - inv_transform(y_mu)
        # At delta_z=0: signups=0 by construction.
        def _to_signups(r_z: np.ndarray) -> np.ndarray:
            if response_transform == "log1p":
                return np.expm1(np.maximum(r_z * y_std + y_mu, 0.0)) - np.expm1(y_mu)
            elif response_transform == "sqrt":
                return np.maximum(r_z * y_std + y_mu, 0.0) ** 2 - y_mu ** 2
            elif response_transform == "boxcox":
                if bl is None or abs(bl) < 1e-10:
                    return np.exp(r_z * y_std + y_mu) - np.exp(y_mu)
                else:
                    return ((r_z * y_std + y_mu) * bl + 1.0) ** (1.0 / bl) - (y_mu * bl + 1.0) ** (1.0 / bl)
            else:
                return r_z * y_std

        mean_sig   = _to_signups(mean_r_inc)
        hdi_lo_sig = _to_signups(hdi_lo_inc)
        hdi_hi_sig = _to_signups(hdi_hi_inc)

        for k, x_val in enumerate(spend_grid):
            rows.append({
                "curve_type"         : curve_type,
                "channel"            : ch_name,
                "spend_scaled"       : round(float(x_val), 6),
                "spend_unscaled"     : round(float(x_val) * metric_max, 2),
                "spend_gbp"          : round(float(x_val) * gbp_scale, 2),
                "cpp"                : round(float(cpp_j), 6),
                "mean_response"      : round(float(mean_r[k]), 6),
                "hdi_10"             : round(float(hdi_lo[k]), 6),
                "hdi_90"             : round(float(hdi_hi[k]), 6),
                "mean_signups"       : round(float(mean_sig[k]), 4),
                "hdi_10_signups"     : round(float(hdi_lo_sig[k]), 4),
                "hdi_90_signups"     : round(float(hdi_hi_sig[k]), 4),
                "sat_type"           : sat_type,
                "ads_type"           : ads_type,
            })

        logger.info(
            f"  [RC:{curve_type:<12s}] {ch_name:<30s} "
            f"max_signups={mean_sig[-1]:.2f} @ GBP {x_max * gbp_scale:,.0f}"
        )

    df = pd.DataFrame(rows)
    logger.info(f"  Response curves computed for {C} channels x {n_spend_points} spend points.")
    return df


def compute_marginal_roi(
    best           : Dict,
    prep           : Dict[str, Any],
    n_spend_points : int = 100,
    n_samples      : int = 400,
) -> pd.DataFrame:
    """Compute marginal ROI (dResponse/dSpend) at each spend level."""
    df_rc = compute_response_curves(
        best, prep, n_spend_points=n_spend_points + 1, n_samples=n_samples
    )

    rows = []
    for ch_name in df_rc["channel"].unique():
        ch_df = df_rc[df_rc["channel"] == ch_name].reset_index(drop=True)
        spend  = ch_df["spend_scaled"].values
        dx     = np.diff(spend)

        d_mean = np.diff(ch_df["mean_response"].values) / (dx + 1e-12)
        d_lo   = np.diff(ch_df["hdi_10"].values)        / (dx + 1e-12)
        d_hi   = np.diff(ch_df["hdi_90"].values)        / (dx + 1e-12)

        below_one = np.where(d_mean < 1.0)[0]
        breakeven = float(spend[below_one[0]]) if len(below_one) > 0 else float(spend[-1])

        for k in range(len(d_mean)):
            rows.append({
                "channel"             : ch_name,
                "spend_scaled"        : round(float(spend[k]), 6),
                "marginal_roi_mean"   : round(float(d_mean[k]), 6),
                "marginal_roi_hdi_10" : round(float(d_lo[k]),   6),
                "marginal_roi_hdi_90" : round(float(d_hi[k]),   6),
                "breakeven_spend"     : round(breakeven,         6),
            })

    df = pd.DataFrame(rows)
    logger.info("  Marginal ROI curves computed.")
    return df


def compute_saturation_analysis(
    best           : Dict,
    prep           : Dict[str, Any],
    n_samples      : int = 400,
    thresholds     : tuple = (0.25, 0.50, 0.75, 0.90),
    cpp_map        : Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Compute a saturation-level analysis table for every channel."""
    df_rc = compute_response_curves(
        best, prep,
        n_spend_points=200,
        n_samples=n_samples,
        spend_max_mult=5.0,
    )

    metric_types = prep.get("metric_types", [])
    spend_cols   = prep.get("spend_cols", [])

    train_idx = prep["train_idx"]
    X_raw = prep["X_media_raw"][train_idx]

    rows = []
    for j, ch in enumerate(spend_cols):
        cdf = df_rc[df_rc["channel"] == ch].reset_index(drop=True)
        if cdf.empty:
            continue

        use_raw = "spend_unscaled" in cdf.columns
        x_col   = "spend_unscaled" if use_raw else "spend_scaled"
        y_col   = "mean_response_raw" if "mean_response_raw" in cdf.columns else "mean_response"

        x = cdf[x_col].values
        y = cdf[y_col].values

        if y.max() < 1e-12:
            continue

        y_max = float(y[-1])
        mt    = metric_types[j] if j < len(metric_types) else "Spend"
        cpp   = cpp_map.get(ch, {}).get("cpp", 1.0) if cpp_map else 1.0

        if X_raw.ndim == 3:
            curr_media = float(X_raw[:, 0, j].mean())
        else:
            curr_media = float(X_raw[:, j].mean())
        curr_spend_gbp = curr_media * cpp

        curr_resp = float(np.interp(curr_media, x, y))
        sat_at_curr = round(min(curr_resp / (y_max + 1e-12) * 100, 100.0), 1)

        row: Dict[str, Any] = {
            "channel"                    : ch,
            "metric_type"                : mt,
            "sat_type"                   : cdf["sat_type"].iloc[0],
            "ads_type"                   : cdf["ads_type"].iloc[0],
            "current_spend"              : round(curr_media, 2),
            "current_spend_gbp"          : round(curr_spend_gbp, 2),
            "saturation_at_current_pct"  : sat_at_curr,
        }

        for thresh in thresholds:
            target_resp = y_max * thresh
            above = np.where(y >= target_resp)[0]
            if len(above) == 0:
                spend_at = float(x[-1])
            else:
                spend_at = float(x[above[0]])
            spend_gbp_at = spend_at * cpp
            pct_label    = int(thresh * 100)
            row[f"spend_at_{pct_label}pct"]     = round(spend_at, 2)
            row[f"spend_gbp_at_{pct_label}pct"] = round(spend_gbp_at, 2)

        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"  [SATURATION] Analysis complete for {len(df)} channels.")
    return df


def compute_response_curves_multiperiod(
    best            : Dict,
    prep            : Dict[str, Any],
    periods         : Optional[Dict[str, int]] = None,
    n_spend_points  : int = 150,
    n_samples       : int = 200,
    spend_max_mult  : float = 1.5,
    cpp_map         : Optional[Dict[str, Any]] = None,   # kept for API compat, ignored
) -> pd.DataFrame:
    """
    Multi-period cumulative response curves with sequential adstock carry-over.

    Simulates N periods of constant spend and sums the incremental signups
    across all periods (including carry-over from previous periods).

    X-axis: total GBP spend over the period (= weekly_spend × n_periods).
    Y-axis: cumulative incremental signups over the period (not per-period).
            Also reports per-period equivalent (total / n_periods).

    Uses _cpp_per_channel for GBP conversion (same as compute_response_curves).
    Uses proper inverse-transform for y-axis (same formula as steady_state curve).
    """
    if periods is None:
        periods = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12}

    trace      = best["trace"]
    spend_cols = prep["spend_cols"]
    C          = prep["C"]
    train_idx  = prep["train_idx"]
    X_raw      = prep["X_media_raw"][train_idx]
    X_scaled   = prep["X_media_scaled"][train_idx]
    frequency  = prep.get("frequency", "weekly")
    y_std      = float(prep["y_std"])
    y_mu       = float(prep["y_mu"])
    rt         = prep.get("response_transform", "log1p")
    bl         = prep.get("boxcox_lambda")

    ppm = _PERIODS_PER_MONTH_RC.get(frequency, _PERIODS_PER_MONTH_RC["weekly"])

    # GBP conversion (impressions/clicks → £)
    cpp_list = _cpp_per_channel(prep)

    # Inverse transform: converts z-unit incremental contribution to signups
    def _z_to_signups(r_z: np.ndarray) -> np.ndarray:
        if rt == "log1p":
            return np.expm1(np.maximum(r_z * y_std + y_mu, 0.0)) - np.expm1(y_mu)
        elif rt == "sqrt":
            return np.maximum(r_z * y_std + y_mu, 0.0) ** 2 - y_mu ** 2
        elif rt == "boxcox":
            if bl is None or abs(bl) < 1e-10:
                return np.exp(r_z * y_std + y_mu) - np.exp(y_mu)
            else:
                return ((r_z * y_std + y_mu) * bl + 1.0) ** (1.0 / bl) - (y_mu * bl + 1.0) ** (1.0 / bl)
        else:
            return r_z * y_std

    channel_specs = best.get("channel_specs", {})
    rows = []

    for period_label, n_months in periods.items():
        n_sim = max(1, round(n_months * ppm))

        for j, ch_name in enumerate(spend_cols):
            spec     = channel_specs.get(j)
            sat_type = spec.saturation   if spec else best["cfg"].saturation
            ads_type = spec.adstock_type if spec else best["cfg"].adstock_type

            # Metric and GBP scales
            if X_raw.ndim == 3:
                metric_max   = float(X_raw[:, 0, j].max()) + 1e-8
                obs_pp_media = float(X_raw[:, 0, j].mean())
                x_scaled_max = float(X_scaled[:, 0, j].max()) + 1e-8
            else:
                metric_max   = float(X_raw[:, j].max()) + 1e-8
                obs_pp_media = float(X_raw[:, j].mean())
                x_scaled_max = float(X_scaled[:, j].max()) + 1e-8

            cpp_j = cpp_list[j]

            obs_wk_gbp           = obs_pp_media * cpp_j
            obs_period_spend_gbp = obs_wk_gbp * n_sim
            max_period_spend_gbp = max(obs_period_spend_gbp * spend_max_mult, 1e-6)
            spend_grid_gbp       = np.linspace(0.0, max_period_spend_gbp, n_spend_points)

            # Convert period GBP → per-period scaled metric
            #   per_period_metric = (period_gbp / n_sim) / cpp_j  (GBP → metric units)
            #   per_period_scaled = per_period_metric / metric_max * x_scaled_max
            wk_gbp_grid   = spend_grid_gbp / n_sim               # per-period GBP
            wk_metric_grid = wk_gbp_grid / (cpp_j + 1e-30)       # per-period metric units
            per_pp_scaled  = wk_metric_grid / metric_max * x_scaled_max  # scaled [0,~1.5]

            lam   = _subsample(_get_lam_samples(trace, j), n_samples)
            beta  = _subsample(_get_beta_samples(trace, j, C), n_samples)
            N_ref = len(lam)
            alpha = _subsample(_coalesce(_scalar_samples(trace, "alpha_sat",  idx=j), np.ones(N_ref)),  n_samples)
            kappa = _subsample(_coalesce(_scalar_samples(trace, "kappa",      idx=j), np.ones(N_ref)),  n_samples)
            k_log = _subsample(_coalesce(_scalar_samples(trace, "k_logistic", idx=j), np.ones(N_ref)),  n_samples)
            x0    = _subsample(_coalesce(_scalar_samples(trace, "x0",         idx=j), np.zeros(N_ref)), n_samples)
            S = min(len(lam), len(beta), len(alpha))
            lam, beta, alpha, kappa, k_log, x0 = (
                lam[:S], beta[:S], alpha[:S], kappa[:S], k_log[:S], x0[:S]
            )

            # Pre-fetch two-timescale decay rates ONCE per channel (outside sample loop)
            # Avoids re-sampling inside the inner time-step loop which caused index mismatch
            # and severe performance degradation.
            if ads_type == "two_timescale":
                _rho_slow_raw = _coalesce(_scalar_samples(trace, "rho_slow_c", idx=j), lam)
                _rho_fast_raw = _coalesce(_scalar_samples(trace, "rho_fast_c", idx=j), lam * 0.5)
                rho_slow_all  = _subsample(_rho_slow_raw, S)[:S]
                rho_fast_all  = _subsample(_rho_fast_raw, S)[:S]
            else:
                rho_slow_all = None
                rho_fast_all = None

            # Cumulative signups matrix: (S, n_spend_points)
            cumul_signups_mat = np.zeros((S, n_spend_points))

            for si in range(S):
                lam_si = float(np.clip(lam[si], 0.0, 0.9999))
                a_si   = np.array([float(alpha[si])])
                k_si   = np.array([float(kappa[si])])
                kl_si  = np.array([float(k_log[si])])
                x0_si  = np.array([float(x0[si])])

                # Two-timescale: separate fast/slow carry states per sample
                if ads_type == "two_timescale":
                    rho_si_slow = float(np.clip(rho_slow_all[si], 0.0, 0.9999))
                    rho_si_fast = float(np.clip(rho_fast_all[si], 0.0, 0.9999))
                    w_mix_raw   = _coalesce(_scalar_samples(trace, "w_mix_c", idx=j), np.full(S, 0.5))
                    w_mix_si    = float(np.clip(_subsample(w_mix_raw, S)[si], 0.0, 1.0))
                    carry_fast  = np.zeros(n_spend_points)
                    carry_slow  = np.zeros(n_spend_points)
                else:
                    carry = np.zeros(n_spend_points)

                ref0  = _apply_saturation(np.zeros(n_spend_points), sat_type,
                                          alpha=a_si, kappa=k_si, k_log=kl_si, x0=x0_si)

                for _t in range(n_sim):
                    # Blended carry for two-timescale, scalar carry for others
                    if ads_type == "two_timescale":
                        carry = w_mix_si * carry_fast + (1.0 - w_mix_si) * carry_slow

                    total_signal = per_pp_scaled + carry
                    sat_total = _apply_saturation(total_signal, sat_type,
                                                  alpha=a_si, kappa=k_si, k_log=kl_si, x0=x0_si)
                    # Incremental z-contribution this period (relative to carry-only baseline)
                    # ref0 anchors the carry=0 baseline so sat_carry - ref0 captures
                    # the carry-over contribution; sat_total - sat_carry is the new-spend lift
                    sat_carry = _apply_saturation(carry, sat_type,
                                                  alpha=a_si, kappa=k_si, k_log=kl_si, x0=x0_si)
                    delta_z = float(beta[si]) * (sat_total - sat_carry)

                    # Convert this period's z-delta to signups and accumulate
                    cumul_signups_mat[si] += _z_to_signups(delta_z)

                    # Update carry-over — use pre-fetched decay rates
                    if ads_type == "weibull":
                        carry_fast = np.zeros(n_spend_points)
                        carry_slow = np.zeros(n_spend_points)
                    elif ads_type == "two_timescale":
                        carry_fast = rho_si_fast * total_signal
                        carry_slow = rho_si_slow * total_signal
                    else:
                        carry = lam_si * total_signal

            mean_r  = cumul_signups_mat.mean(axis=0)
            hdi_lo  = np.percentile(cumul_signups_mat, 10, axis=0)
            hdi_hi  = np.percentile(cumul_signups_mat, 90, axis=0)
            # Ensure monotone (numerical noise can cause tiny dips)
            mean_r  = np.maximum.accumulate(mean_r)

            for k in range(n_spend_points):
                rows.append({
                    "curve_type"              : "multiperiod_%s" % period_label,
                    "period_label"            : period_label,
                    "n_months"                : n_months,
                    "n_sim_periods"           : n_sim,
                    "channel"                 : ch_name,
                    "spend_gbp"               : round(float(spend_grid_gbp[k]),           2),  # total period GBP
                    "weekly_spend_gbp_equiv"  : round(float(wk_gbp_grid[k]),              2),  # per-week equiv
                    "cpp"                     : round(float(cpp_j),                        6),
                    "mean_signups"            : round(float(mean_r[k]),                    4),  # cumulative
                    "hdi_10_signups"          : round(float(hdi_lo[k]),                    4),
                    "hdi_90_signups"          : round(float(hdi_hi[k]),                    4),
                    "mean_signups_per_period" : round(float(mean_r[k]) / n_sim,            4),  # per period
                    "sat_type"                : sat_type,
                    "ads_type"                : ads_type,
                    "obs_period_spend_gbp"    : round(float(obs_period_spend_gbp),         2),
                    "obs_weekly_spend_gbp"    : round(float(obs_wk_gbp),                   2),
                })

            logger.info(
                f"  [RC-MP:{period_label}] {ch_name:<30s} "
                f"obs_period=GBP {obs_period_spend_gbp:,.0f}  "
                f"cumul_signups={mean_r[-1]:.1f}"
            )

    df = pd.DataFrame(rows)
    logger.info(
        f"  Multi-period response curves: {len(periods)} periods x {C} channels "
        f"x {n_spend_points} points."
    )
    return df


def _plot_multiperiod_curves(
    df_mp     : pd.DataFrame,
    out_dir   : "Path",
    obs_spend : Optional[Dict[str, float]] = None,
) -> None:
    """
    Multi-period response curve grid — same layout as steady-state plot.

    Produces two files:
    1. response_curves_multiperiod.png   — 1 panel per channel, all horizons overlaid
    2. response_curves_multiperiod_<H>.png — 1 panel per channel for each horizon H
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir = _RCPath(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = df_mp["channel"].unique().tolist()
    periods  = df_mp["period_label"].unique().tolist()
    C        = len(channels)
    ncols    = min(3, C)
    nrows    = (C + ncols - 1) // ncols

    # Horizon colour palette — same blue family as steady-state, darkening by horizon
    horizon_colors = {
        "1M" : ("#aec7e8", "#1f77b4"),   # (band, line)
        "3M" : ("#ffbb78", "#ff7f0e"),
        "6M" : ("#98df8a", "#2ca02c"),
        "1Y" : ("#ff9896", "#d62728"),
    }
    default_colors = [
        ("#aec7e8","#1f77b4"), ("#ffbb78","#ff7f0e"),
        ("#98df8a","#2ca02c"), ("#ff9896","#d62728"),
    ]

    def _fmt_gbp(v, _):  return f"£{v:,.0f}"
    def _fmt_y(v, _):    return f"{v:,.1f}"

    # ── 1. Main grid: one panel per channel, all horizons overlaid ───────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes_flat = np.array(axes).flatten() if C > 1 else [axes]

    for i, ch in enumerate(channels):
        ax = axes_flat[i]
        for hi_idx, pl in enumerate(periods):
            sub = df_mp[(df_mp["channel"] == ch) & (df_mp["period_label"] == pl)]
            if sub.empty:
                continue
            band_c, line_c = horizon_colors.get(pl, default_colors[hi_idx % 4])
            x  = sub["spend_gbp"].values
            y  = sub["mean_signups"].values
            lo = np.maximum(sub["hdi_10_signups"].values, 0)  # clip negatives for display
            hi = sub["hdi_90_signups"].values

            ax.plot(x, y, color=line_c, lw=2, label=pl)
            ax.fill_between(x, lo, hi, alpha=0.18, color=band_c)

            # Current-spend dot per horizon (different x position per horizon)
            if obs_spend and ch in obs_spend:
                n_sim   = int(sub["n_sim_periods"].iloc[0])
                per_obs = float(obs_spend[ch]) * n_sim
                y_obs   = float(np.interp(per_obs, x, y))
                ax.scatter([per_obs], [y_obs], color=line_c, s=55, zorder=6,
                           edgecolors="white", linewidths=0.8)

        # Single vline at current weekly spend × 1 period (1M) for reference
        if obs_spend and ch in obs_spend and periods:
            first_pl = periods[0]
            sub0 = df_mp[(df_mp["channel"] == ch) & (df_mp["period_label"] == first_pl)]
            if not sub0.empty:
                n0 = int(sub0["n_sim_periods"].iloc[0])
                ax.axvline(float(obs_spend[ch]) * n0, color="#e05c00", ls="--",
                           lw=1.3, alpha=0.6, label="Curr. spend (%s)" % first_pl)

        ax.set_title(ch, fontsize=9, fontweight="bold")
        ax.set_xlabel("Total GBP Spend over Period", fontsize=8)
        ax.set_ylabel("Cumulative Incremental Signups\n(incl. adstock carry-over)", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_y))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_gbp))

    for j in range(len(channels), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Media Channel Response Curves (Posterior Mean +/- 80% HDI)"
        "\nMulti-Period Cumulative — all horizons overlaid",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    grid_path = out_dir / "response_curves_multiperiod.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {grid_path}")

    # ── 2. Per-horizon grid: one panel per channel, single horizon ───────────
    for pl in periods:
        band_c, line_c = horizon_colors.get(pl, ("#aec7e8", "#1f77b4"))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
        axes_flat = np.array(axes).flatten() if C > 1 else [axes]

        for i, ch in enumerate(channels):
            ax  = axes_flat[i]
            sub = df_mp[(df_mp["channel"] == ch) & (df_mp["period_label"] == pl)]
            if sub.empty:
                axes_flat[i].set_visible(False)
                continue
            x  = sub["spend_gbp"].values
            y  = sub["mean_signups"].values
            lo = np.maximum(sub["hdi_10_signups"].values, 0)
            hi = sub["hdi_90_signups"].values
            n_sim = int(sub["n_sim_periods"].iloc[0])

            ax.plot(x, y, color=line_c, lw=2, label="Mean response")
            ax.fill_between(x, lo, hi, alpha=0.25, color=band_c, label="80% HDI")

            if obs_spend and ch in obs_spend:
                per_obs = float(obs_spend[ch]) * n_sim
                y_obs   = float(np.interp(per_obs, x, y))
                ax.axvline(per_obs, color="#e05c00", ls="--", lw=1.4, alpha=0.7)
                ax.scatter([per_obs], [y_obs], color="#e05c00", s=70,
                           zorder=6, edgecolors="white", linewidths=0.8,
                           label="Current spend")

            sat_t = sub["sat_type"].iloc[0]
            ads_t = sub["ads_type"].iloc[0]
            ax.set_title(f"{ch}\n({sat_t} sat, {ads_t} adstock)", fontsize=8, fontweight="bold")
            ax.set_xlabel(f"Total GBP Spend ({pl})", fontsize=8)
            ax.set_ylabel("Cumulative Incremental Signups", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_y))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_gbp))

        for j in range(len(channels), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f"Media Channel Response Curves (Posterior Mean +/- 80% HDI)"
            f"\nMulti-Period Cumulative — {pl} Horizon (incl. adstock carry-over)",
            fontsize=12, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        out_path = out_dir / f"response_curves_multiperiod_{pl}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {out_path}")


def plot_response_curves(
    df_curves    : pd.DataFrame,
    out_dir      : "Path",
    df_mroi      : Optional[pd.DataFrame] = None,
    obs_spend    : Optional[Dict[str, float]] = None,
    opt_spend    : Optional[Dict[str, float]] = None,
    cpp_map      : Optional[Dict[str, Any]] = None,
    df_sat       : Optional[pd.DataFrame] = None,
    title_suffix : str = "",
) -> None:
    """Plot response curves for all channels."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logger.warning("  matplotlib not available — skipping response curve plots.")
        return

    if df_curves.empty:
        logger.warning("  Response curve DataFrame is empty — skipping plots.")
        return

    out_dir = _RCPath(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    channels = df_curves["channel"].unique().tolist()
    C        = len(channels)

    # Prefer GBP spend on x-axis; fall back to unscaled metric, then scaled
    if "spend_gbp" in df_curves.columns:
        x_col   = "spend_gbp"
        x_label_default = "GBP Spend"
    elif "spend_unscaled" in df_curves.columns:
        x_col   = "spend_unscaled"
        x_label_default = "Media units (impressions/clicks)"
    else:
        x_col   = "spend_scaled"
        x_label_default = "Spend (scaled)"

    # Prefer proper incremental signups on y-axis; fall back to raw approx
    curve_type_col = df_curves["curve_type"].iloc[0] if "curve_type" in df_curves.columns else "steady_state"
    if "mean_signups" in df_curves.columns:
        y_col  = "mean_signups"
        lo_col = "hdi_10_signups"
        hi_col = "hdi_90_signups"
        if curve_type_col == "instantaneous":
            y_label = "Incremental signups\n(single period, no adstock carry-over)"
        else:
            y_label = "Incremental signups per period\n(steady-state: adstock at equilibrium)"
    elif "mean_response_raw" in df_curves.columns:
        y_col  = "mean_response_raw"
        lo_col = "hdi_10_raw"
        hi_col = "hdi_90_raw"
        y_label = "Incremental response (approx. original units)"
    else:
        y_col  = "mean_response"
        lo_col = "hdi_10"
        hi_col = "hdi_90"
        y_label = "Response (z-space)"

    def _x_label_for(ch: str) -> str:
        if "spend_gbp" in df_curves.columns:
            # Show CPP if available
            ch_rows = df_curves[df_curves["channel"] == ch]
            if "cpp" in ch_rows.columns:
                cpp_val = float(ch_rows["cpp"].iloc[0])
                if cpp_val != 1.0:
                    return f"GBP Spend  (cpp = GBP {cpp_val:.4f}/unit)"
            return "GBP Spend"
        return x_label_default

    ncols = min(3, C)
    nrows = (C + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten() if C > 1 else [axes]

    for i, ch in enumerate(channels):
        ax  = axes[i]
        cdf = df_curves[df_curves["channel"] == ch]
        x   = cdf[x_col].values
        y   = cdf[y_col].values
        lo  = cdf[lo_col].values
        hi  = cdf[hi_col].values

        ax.plot(x, y, color="#1f77b4", lw=2, label="Mean response")
        ax.fill_between(x, lo, hi, alpha=0.25, color="#1f77b4", label="80% HDI")

        if obs_spend and ch in obs_spend:
            ox = float(obs_spend[ch])
            oy = float(np.interp(ox, x, y))
            ax.axvline(ox, color="#e05c00", ls="--", lw=1.4, alpha=0.7)
            ax.scatter([ox], [oy], color="#e05c00", s=80, zorder=5, label="Current spend")

        if opt_spend and ch in opt_spend:
            px = float(opt_spend[ch])
            py = float(np.interp(px, x, y))
            ax.axvline(px, color="#7b2d8b", ls="--", lw=1.4, alpha=0.7)
            ax.scatter([px], [py], color="#7b2d8b", s=80, zorder=5,
                       marker="D", label="Optimised spend")

        if df_sat is not None and not df_sat.empty and ch in df_sat["channel"].values:
            sat_row = df_sat[df_sat["channel"] == ch].iloc[0]
            sat_pct = sat_row.get("saturation_at_current_pct", None)
            if sat_pct is not None:
                ax.set_title(f"{ch}\n(current saturation: {sat_pct:.0f}%)",
                             fontsize=9, fontweight="bold")
            else:
                ax.set_title(ch, fontsize=10, fontweight="bold")
        else:
            ax.set_title(ch, fontsize=10, fontweight="bold")

        ax.set_xlabel(_x_label_for(ch), fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.1f}"))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Media Channel Response Curves (Posterior Mean +/- 80% HDI)" + title_suffix,
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    grid_path = out_dir / "response_curves.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {grid_path}")

    for ch in channels:
        cdf  = df_curves[df_curves["channel"] == ch]
        x    = cdf[x_col].values
        y    = cdf[y_col].values
        lo   = cdf[lo_col].values
        hi   = cdf[hi_col].values
        xl   = _x_label_for(ch)

        mdf = None
        if df_mroi is not None and not df_mroi.empty:
            mdf = df_mroi[df_mroi["channel"] == ch]

        # Two-panel layout: response curve (top) + mROI panel (bottom)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 8),
            height_ratios=[2.2, 1],
            sharex=True,
            gridspec_kw={"hspace": 0.08},
        )

        # ── Top panel: response curve ──────────────────────────────────────
        ax1.plot(x, y, color="#1f77b4", lw=2.5, label="Mean response")
        ax1.fill_between(x, lo, hi, alpha=0.2, color="#1f77b4", label="80% HDI")
        ax1.set_ylabel(y_label, fontsize=11)
        ax1.tick_params(axis="x", labelbottom=False)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.1f}"))
        ax1.set_facecolor("#fafafa")

        # Saturation info
        sat_pct_val = None
        sat_spend_at50 = None
        if df_sat is not None and not df_sat.empty and ch in df_sat["channel"].values:
            sat_row = df_sat[df_sat["channel"] == ch].iloc[0]
            sat_pct_val = sat_row.get("saturation_at_current_pct", None)
            sat_spend_at50 = sat_row.get("spend_at_50pct", None)

        # Compute saturation point from response curve numerically (slope → 50% of initial)
        if sat_spend_at50 is None and len(x) > 2:
            dy = np.diff(y);  dx_vals = np.diff(x)
            slope = dy / (dx_vals + 1e-30)
            init_sl = slope[0] if slope[0] > 0 else 1e-12
            half_mask = slope < 0.5 * init_sl
            if half_mask.any():
                sat_spend_at50 = float(x[:-1][half_mask][0])

        if sat_spend_at50 is not None:
            ax1.axvline(sat_spend_at50, color="#2ca02c", ls=":", lw=2.0, alpha=0.9,
                        label=f"50% sat. @ {sat_spend_at50:,.0f}")

        # Current spend marker + efficiency annotation
        if obs_spend and ch in obs_spend:
            ox = float(obs_spend[ch])
            oy = float(np.interp(ox, x, y))
            ax1.axvline(ox, color="#e05c00", ls="--", lw=1.6, alpha=0.85)
            ax1.scatter([ox], [oy], color="#e05c00", s=110, zorder=6)

            # mROI at current spend (numerical derivative)
            if len(x) > 2:
                dy2 = np.diff(y); dx2 = np.diff(x)
                mroi_curve = dy2 / (dx2 + 1e-30)
                mroi_x_pts = (x[:-1] + x[1:]) / 2
                curr_mroi_val = float(np.interp(ox, mroi_x_pts, mroi_curve))
            else:
                curr_mroi_val = 0.0

            sat_pct_ann = sat_pct_val if sat_pct_val is not None else 0.0
            if sat_pct_ann < 35:
                eff_label, eff_col = "Under-invested", "#2ca02c"
            elif sat_pct_ann < 65:
                eff_label, eff_col = "Efficient zone", "#1f77b4"
            else:
                eff_label, eff_col = "Near saturation", "#d62728"

            ann_x_off = 0.12 * (x[-1] - x[0])
            ann_y_mid = 0.5 * max(y)
            eff_txt = (f"mROI: {curr_mroi_val:.4f}/unit\n"
                       f"{sat_pct_ann:.0f}% of capacity\n"
                       f"► {eff_label}")
            ax1.annotate(
                eff_txt,
                xy=(ox, oy),
                xytext=(ox + ann_x_off, ann_y_mid),
                fontsize=8.5, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc=eff_col, alpha=0.88, ec="white", lw=1.2),
                arrowprops=dict(arrowstyle="->", color=eff_col, lw=1.5),
            )
            ax1.annotate(f"Current ({ox:,.0f})", xy=(ox, oy),
                         xytext=(ox + 0.005 * (x[-1] - x[0]), oy * 1.02),
                         fontsize=7.5, color="#e05c00")

        # Optimised spend marker
        if opt_spend and ch in opt_spend:
            px = float(opt_spend[ch])
            py = float(np.interp(px, x, y))
            ax1.axvline(px, color="#7b2d8b", ls="--", lw=1.6, alpha=0.85)
            ax1.scatter([px], [py], color="#7b2d8b", s=110, zorder=6, marker="D",
                        label=f"Optimised ({px:,.0f})")

        title_sfx = f"  (current saturation: {sat_pct_val:.0f}%)" if sat_pct_val is not None else ""
        ax1.set_title(f"{ch}  —  response curve{title_sfx}", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=8.5, loc="upper left", ncol=2)
        ax1.grid(True, alpha=0.3, ls="--")

        # ── Bottom panel: mROI ─────────────────────────────────────────────
        if mdf is not None and len(mdf) > 0:
            mx_raw = mdf["spend_scaled"].values
            sp_max = (cdf["spend_unscaled"].max() / cdf["spend_scaled"].max()
                      if use_raw and cdf["spend_scaled"].max() > 0 else 1.0)
            mx = mx_raw * sp_max if use_raw else mx_raw
            my = mdf["marginal_roi_mean"].values
            ax2.plot(mx, my, color="#d62728", lw=1.8, label="Marginal ROI")
            ax2.fill_between(mx, my, 0, where=(my > 0), alpha=0.10, color="#1f77b4")
            ax2.fill_between(mx, my, 0, where=(my < 0), alpha=0.15, color="#d62728",
                             label="Negative ROI zone")
            ax2.axhline(0, color="#333", lw=0.8, alpha=0.5)
            if obs_spend and ch in obs_spend:
                ax2.axvline(float(obs_spend[ch]), color="#e05c00", ls="--", lw=1.6, alpha=0.85)
            if sat_spend_at50 is not None:
                ax2.axvline(sat_spend_at50, color="#2ca02c", ls=":", lw=2.0, alpha=0.9)
            ax2.set_ylabel("mROI\n(Δsignups/Δunit)", fontsize=9)
            ax2.legend(fontsize=7.5, loc="upper right")
        else:
            # Compute mROI numerically from response curve
            if len(x) > 2:
                dy_m = np.diff(y); dx_m = np.diff(x)
                mroi_vals = dy_m / (dx_m + 1e-30)
                mroi_pts  = (x[:-1] + x[1:]) / 2
                ax2.plot(mroi_pts, mroi_vals, color="#d62728", lw=1.8, label="Marginal ROI")
                ax2.fill_between(mroi_pts, mroi_vals, 0, where=(mroi_vals > 0), alpha=0.10, color="#1f77b4")
                ax2.fill_between(mroi_pts, mroi_vals, 0, where=(mroi_vals < 0), alpha=0.15, color="#d62728")
                ax2.axhline(0, color="#333", lw=0.8, alpha=0.5)
                if obs_spend and ch in obs_spend:
                    ax2.axvline(float(obs_spend[ch]), color="#e05c00", ls="--", lw=1.6, alpha=0.85)
                if sat_spend_at50 is not None:
                    ax2.axvline(sat_spend_at50, color="#2ca02c", ls=":", lw=2.0, alpha=0.9)
                ax2.set_ylabel("mROI\n(Δsignups/Δunit)", fontsize=9)
                ax2.legend(["Marginal ROI"], fontsize=7.5, loc="upper right")

        ax2.set_xlabel(xl, fontsize=11)
        ax2.grid(True, alpha=0.3, ls="--")
        ax2.set_facecolor("#fff8f8")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.4f}"))
        fig.tight_layout()

        ch_path = out_dir / f"response_curve_{ch}.png"
        fig.savefig(ch_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {ch_path}")

    if df_sat is not None and not df_sat.empty:
        _plot_saturation_summary(df_sat, out_dir)


def _plot_saturation_summary(df_sat: pd.DataFrame, out_dir: "Path") -> None:
    """Bar chart: current saturation % and spend-at-50% per channel."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    if "saturation_at_current_pct" not in df_sat.columns:
        return

    chs    = df_sat["channel"].tolist()
    sat    = df_sat["saturation_at_current_pct"].astype(float).tolist()
    n      = len(chs)
    colors = ["#2ca02c" if s < 50 else "#ff7f0e" if s < 75 else "#d62728" for s in sat]

    fig, ax = plt.subplots(figsize=(max(6, n * 1.5), 5))
    bars = ax.barh(chs, sat, color=colors, edgecolor="white", height=0.6)
    ax.axvline(50, color="grey", lw=1.2, ls="--", alpha=0.6, label="50% threshold")
    ax.axvline(75, color="orange", lw=1.2, ls="--", alpha=0.6, label="75% threshold")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Channel saturation at current spend (%)", fontsize=11)
    ax.set_title("Channel Saturation Efficiency\n(green < 50% headroom  |  orange 50-75%  |  red > 75%)",
                 fontsize=12, fontweight="bold")

    for bar, s in zip(bars, sat):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{s:.0f}%", va="center", fontsize=9)

    patches = [
        mpatches.Patch(color="#2ca02c", label="< 50%: growth headroom"),
        mpatches.Patch(color="#ff7f0e", label="50-75%: approaching saturation"),
        mpatches.Patch(color="#d62728", label="> 75%: highly saturated"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    sat_path = out_dir / "saturation_analysis.png"
    fig.savefig(sat_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {sat_path}")


def plot_waterfall_chart(
    df_channels : pd.DataFrame,
    out_dir     : "Path",
    title       : str = "Budget Reallocation Analysis  |  Current vs Optimised",
) -> None:
    """
    Waterfall chart showing per-channel spend changes (current → optimised)
    plus a constraint-status table on the right.

    df_channels must have columns:
        channel, current_spend_gbp, opt_spend_gbp, current_roi, opt_roi
    Optional: constraint_status column.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as _mtick
        from matplotlib.patches import Patch
    except ImportError:
        logger.warning("  matplotlib not available — skipping waterfall chart.")
        return

    if df_channels is None or df_channels.empty:
        return

    out_dir = _RCPath(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channels    = df_channels["channel"].tolist()
    curr_spend  = df_channels["current_spend_gbp"].astype(float).values
    opt_spend   = df_channels["opt_spend_gbp"].astype(float).values
    delta       = opt_spend - curr_spend
    constraints = (df_channels["constraint_status"].tolist()
                   if "constraint_status" in df_channels.columns
                   else ["Unconstrained"] * len(channels))
    curr_roi    = (df_channels["current_roi"].astype(float).values
                   if "current_roi" in df_channels.columns
                   else np.zeros(len(channels)))

    fig, (ax_w, ax_t) = plt.subplots(1, 2, figsize=(16, 7), facecolor="white",
                                      gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    # ── waterfall ────────────────────────────────────────────────────────────
    total_curr = float(curr_spend.sum())
    total_opt  = float(opt_spend.sum())
    x_pos      = np.arange(len(channels) + 2)
    labels_all = ["Current\nTotal"] + [ch.replace("media_impressions_clara_", "").upper()
                                       for ch in channels] + ["Optimised\nTotal"]
    bar_colors = ["#1f77b4"] + ["#2ca02c" if d > 0 else "#d62728" for d in delta] + ["#7b2d8b"]

    running = total_curr
    bottoms = [0]
    heights = [total_curr]
    for d in delta:
        bottoms.append(min(running, running + d))
        heights.append(abs(d))
        running += d

    bars = ax_w.bar(x_pos[:-1], heights, bottom=bottoms, color=bar_colors[:-1],
                    width=0.6, edgecolor="white", linewidth=1.5)
    # Optimised total
    ax_w.bar(x_pos[-1], total_opt, color="#7b2d8b", width=0.6, edgecolor="white", linewidth=1.5)

    # Value labels
    for i, (bar, bh, bb) in enumerate(zip(bars, heights, bottoms)):
        if i == 0:
            lbl = f"£{bh:,.0f}"
        else:
            val = bh if bar_colors[i] == "#2ca02c" else -bh
            lbl = f'{"+" if val > 0 else ""}£{val:,.0f}'
        ax_w.text(bar.get_x() + bar.get_width() / 2, bb + bh + total_curr * 0.01,
                  lbl, ha="center", va="bottom", fontsize=9, fontweight="bold",
                  color=bar_colors[i])
    ax_w.text(x_pos[-1] + 0.0, total_opt + total_curr * 0.01,
              f"£{total_opt:,.0f}", ha="center", va="bottom",
              fontsize=9, fontweight="bold", color="#7b2d8b")

    ax_w.set_xticks(x_pos)
    ax_w.set_xticklabels(labels_all, fontsize=9)
    ax_w.set_ylabel("Monthly Spend (£)", fontsize=10)
    ax_w.set_title("Spend Reallocation Waterfall", fontsize=11, fontweight="bold")
    ax_w.yaxis.set_major_formatter(_mtick.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    ax_w.grid(True, alpha=0.3, axis="y", ls="--")
    ax_w.set_facecolor("#fafafa")
    legend_el = [Patch(fc="#2ca02c", label="Increase"), Patch(fc="#d62728", label="Decrease"),
                 Patch(fc="#1f77b4", label="Current total"), Patch(fc="#7b2d8b", label="Optimised total")]
    ax_w.legend(handles=legend_el, fontsize=8, loc="upper right")

    # ── constraint table ─────────────────────────────────────────────────────
    ax_t.axis("off")
    col_labels = ["Channel", "Current\n£/mo", "Optimal\n£/mo", "Change", "mROI\nnow", "Constraint"]
    rows_data  = []
    for j, ch in enumerate(channels):
        short = ch.replace("media_impressions_clara_", "").upper()
        chg   = f'{"+" if delta[j] > 0 else ""}£{delta[j]:,.0f}'
        rows_data.append([short, f"£{curr_spend[j]:,.0f}", f"£{opt_spend[j]:,.0f}",
                          chg, f"{curr_roi[j]:.4f}", constraints[j]])

    tbl = ax_t.table(cellText=rows_data, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.35, 2.2)

    for k in range(len(col_labels)):
        tbl[(0, k)].set_facecolor("#2c3e50")
        tbl[(0, k)].set_text_props(color="white", fontweight="bold")

    constraint_fc = {"At min bound": "#fff3cd", "At max bound": "#fde8e8", "Unconstrained": "#e8f5e9"}
    for row in range(1, len(rows_data) + 1):
        fc = constraint_fc.get(rows_data[row - 1][5], "white")
        for col in range(len(col_labels)):
            tbl[(row, col)].set_facecolor(fc)
        tbl[(row, 3)].set_text_props(
            color="#2ca02c" if delta[row - 1] > 0 else "#d62728", fontweight="bold"
        )

    ax_t.set_title("Optimised Allocation + Constraint Status", fontsize=11, fontweight="bold", pad=20)

    fig.tight_layout()
    p = out_dir / "waterfall_optimizer.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {p}")


def plot_efficiency_scorecard(
    df_channels : pd.DataFrame,
    df_rc       : pd.DataFrame,
    out_dir     : "Path",
) -> None:
    """
    Single-page scorecard: one row per channel.
    Columns: channel | current spend/wk | current mROI | recommended spend/wk
             | optimised mROI | saturation % | traffic-light status | action.

    df_channels : from optimise_budget_sequential (has current_spend_gbp, opt_spend_gbp,
                  current_roi, n_months).
    df_rc       : from compute_response_curves (has channel, spend_gbp, mean_signups).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("  matplotlib not available — skipping efficiency scorecard.")
        return

    if df_channels is None or df_channels.empty:
        return

    out_dir = _RCPath(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channels   = df_channels["channel"].tolist()
    n_months   = int(df_channels["n_months"].iloc[0]) if "n_months" in df_channels.columns else 1
    wks_per_mo = 4.35

    sc_cols = ["Channel", "Current\nSpend/wk", "Current\nmROI",
               "Recommended\nSpend/wk", "Optimised\nmROI",
               "Saturation\n% used", "Status", "Action"]
    sc_rows = []

    status_colors = {
        "🟢 Under-invested": "#d4edda",
        "🔵 Efficient":       "#d1ecf1",
        "🔴 Near saturation": "#f8d7da",
    }

    for _, row in df_channels.iterrows():
        ch          = row["channel"]
        curr_sp_wk  = float(row.get("current_spend_gbp", 0)) / (n_months * wks_per_mo)
        opt_sp_wk   = float(row.get("opt_spend_gbp",     0)) / (n_months * wks_per_mo)
        curr_roi    = float(row.get("current_roi",        0))

        # mROI and saturation from response curves
        sat_pct_val = 0.0
        opt_mroi    = 0.0
        if df_rc is not None and not df_rc.empty and ch in df_rc["channel"].values:
            rc_ch = df_rc[df_rc["channel"] == ch]
            x_gbp = rc_ch["spend_gbp"].values
            y_sig = rc_ch["mean_signups"].values
            if len(x_gbp) > 2:
                dy_rc   = np.diff(y_sig); dx_rc = np.diff(x_gbp)
                mroi_rc = dy_rc / (dx_rc + 1e-30)
                mroi_xr = (x_gbp[:-1] + x_gbp[1:]) / 2
                opt_mroi = float(np.interp(opt_sp_wk * wks_per_mo, mroi_xr, mroi_rc))
                curr_sig  = float(np.interp(curr_sp_wk * wks_per_mo, x_gbp, y_sig))
                max_sig   = float(y_sig[-1])
                sat_pct_val = min(curr_sig / (max_sig + 1e-30) * 100, 100)

        if sat_pct_val < 35:
            status, action = "🟢 Under-invested", "Increase budget"
        elif sat_pct_val < 65:
            status, action = "🔵 Efficient",       "Maintain"
        else:
            status, action = "🔴 Near saturation", "Reduce / reallocate"

        short = ch.replace("media_impressions_clara_", "").upper()
        sc_rows.append([short, f"£{curr_sp_wk:,.0f}", f"{curr_roi:.4f}",
                        f"£{opt_sp_wk:,.0f}", f"{opt_mroi:.4f}",
                        f"{sat_pct_val:.0f}%", status, action])

    fig, ax = plt.subplots(figsize=(16, max(4, len(sc_rows) * 1.4 + 2)), facecolor="white")
    ax.axis("off")
    fig.suptitle("Channel Efficiency Scorecard  |  Current State vs Optimised Recommendation",
                 fontsize=13, fontweight="bold", y=0.98)

    tbl = ax.table(cellText=sc_rows, colLabels=sc_cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1.35, 2.8)

    for k in range(len(sc_cols)):
        tbl[(0, k)].set_facecolor("#1a252f")
        tbl[(0, k)].set_text_props(color="white", fontweight="bold")

    for r in range(1, len(sc_rows) + 1):
        fc = status_colors.get(sc_rows[r - 1][6], "white")
        for c in range(len(sc_cols)):
            tbl[(r, c)].set_facecolor(fc)
        tbl[(r, 7)].set_text_props(fontweight="bold")

    fig.tight_layout()
    p = out_dir / "efficiency_scorecard.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {p}")


# ─── Scenario planner (scenario_planner.py) ──────────────────────────────────

from dataclasses import dataclass as _sc_dataclass, field as _sc_field
from typing import List as _ScList, Tuple as _ScTuple


@_sc_dataclass
class ScenarioConfig:
    """
    Defines a single planning scenario.
    """
    name             : str
    type             : str             = "budget"
    total_budget     : Optional[float] = None
    budget_pct       : Optional[float] = None
    allocations      : Optional[Dict[str, float]] = None
    target_response  : Optional[float] = None
    channel_min      : Optional[Dict[str, float]] = None
    channel_max      : Optional[Dict[str, float]] = None

    def __post_init__(self):
        valid = {"current", "budget", "budget_pct", "allocation", "reverse"}
        if self.type not in valid:
            raise ValueError(
                f"ScenarioConfig.type='{self.type}' is not valid. "
                f"Choose from: {sorted(valid)}"
            )


def _current_total_spend(prep: Dict[str, Any], cpp_w: np.ndarray, n_periods: float) -> float:
    train_idx = prep["train_idx"]
    X_raw     = prep["X_media_raw"][train_idx]
    if X_raw.ndim == 3:
        curr_pp = X_raw[:, 0, :].mean(axis=0)
    else:
        curr_pp = X_raw.mean(axis=0)
    return float((curr_pp * cpp_w).sum() * n_periods)


def _evaluate_at_allocation(
    alloc_spend_pp : np.ndarray,
    cpp_w          : np.ndarray,
    params_list    : "_ScList[Dict]",
    rscale         : float,
    n_periods      : float,
) -> "_ScTuple[np.ndarray, np.ndarray]":
    """Evaluate total response at a fixed (non-optimised) allocation."""
    from optimisation import _channel_response_raw

    alloc_media_pp = alloc_spend_pp / (cpp_w + 1e-30)
    C = len(params_list)
    resp = np.zeros(C)

    for j, p in enumerate(params_list):
        resp[j] = _channel_response_raw(
            float(alloc_media_pp[j]), p["spend_max"],
            lam      = float(p["lam"].mean()),
            beta     = float(p["beta"].mean()),
            alpha    = float(p["alpha"].mean()),
            kappa    = float(p["kappa"].mean()),
            k_log    = float(p["k_log"].mean()),
            x0       = float(p["x0"].mean()),
            sat      = p["sat_type"],
            ads_type = p.get("ads_type", "geometric"),
        )

    resp_full   = resp * rscale * n_periods
    spend_full  = alloc_spend_pp * n_periods

    return resp_full, spend_full


def run_scenarios(
    best          : Dict[str, Any],
    prep          : Dict[str, Any],
    scenarios     : "_ScList[ScenarioConfig]",
    budget_period : str                       = "monthly",
    n_samples     : int                       = 200,
    cpp_map       : Optional[Dict[str, Any]]  = None,
) -> "_ScTuple[pd.DataFrame, pd.DataFrame]":
    """Run all scenarios and return comparison DataFrames."""
    from optimisation import (
        _extract_all_channel_params,
        _periods_in_window,
        _response_scale,
        optimise_budget,
        minimise_spend_for_target,
        cpp_weights_array,
    )

    spend_cols  = prep["spend_cols"]
    C           = len(spend_cols)
    train_idx   = prep["train_idx"]
    n_train     = len(train_idx)
    frequency   = prep.get("frequency", "weekly")
    n_periods   = _periods_in_window(budget_period, frequency, n_train)
    rscale      = _response_scale(prep)

    _, params_list = _extract_all_channel_params(best, prep, n_samples)

    cpp_w = cpp_weights_array(spend_cols, cpp_map) if cpp_map else np.ones(C)

    X_raw = prep["X_media_raw"][train_idx]
    if X_raw.ndim == 3:
        curr_media_pp = X_raw[:, 0, :].mean(axis=0)
    else:
        curr_media_pp = X_raw.mean(axis=0)
    curr_spend_pp = curr_media_pp * cpp_w

    tot_curr_spend = float(curr_spend_pp.sum() * n_periods)
    metric_types   = prep.get("metric_types", ["Spend"] * C)

    comparison_rows: "_ScList[Dict]" = []
    detail_rows:     "_ScList[Dict]" = []

    for sc in scenarios:
        logger.info(f"\n[SCENARIO] -- {sc.name} ({sc.type}) ----------------")

        try:
            if sc.type == "current":
                resp_by_ch, spend_by_ch = _run_current(
                    curr_spend_pp, cpp_w, params_list, rscale, n_periods
                )
                tot_spend   = float(spend_by_ch.sum())
                tot_resp    = float(resp_by_ch.sum())
                convergence = None
                resp_hdi_10 = resp_hdi_90 = None

            elif sc.type in ("budget", "budget_pct"):
                if sc.type == "budget_pct":
                    pct    = float(sc.budget_pct or 0)
                    budget = tot_curr_spend * (1 + pct / 100.0)
                else:
                    budget = sc.total_budget

                df_opt = optimise_budget(
                    best, prep,
                    total_budget  = budget,
                    budget_period = budget_period,
                    n_samples     = n_samples,
                    channel_min   = sc.channel_min,
                    channel_max   = sc.channel_max,
                    cpp_weights   = cpp_w,
                )
                resp_by_ch, spend_by_ch, tot_spend, tot_resp, resp_hdi_10, resp_hdi_90, convergence = \
                    _parse_forward_result(df_opt, spend_cols, cpp_w)

            elif sc.type == "allocation":
                allocs = sc.allocations or {}
                spend_pp = np.array([
                    float(allocs.get(ch, curr_spend_pp[j] * n_periods)) / n_periods
                    for j, ch in enumerate(spend_cols)
                ])
                resp_by_ch, spend_by_ch = _evaluate_at_allocation(
                    spend_pp, cpp_w, params_list, rscale, n_periods
                )
                tot_spend   = float(spend_by_ch.sum())
                tot_resp    = float(resp_by_ch.sum())
                convergence = None
                resp_hdi_10 = resp_hdi_90 = None

            elif sc.type == "reverse":
                df_rev = minimise_spend_for_target(
                    best, prep,
                    target_response = sc.target_response,
                    target_period   = budget_period,
                    n_samples       = n_samples,
                    channel_min     = sc.channel_min,
                    channel_max     = sc.channel_max,
                    cpp_weights     = cpp_w,
                )
                resp_by_ch, spend_by_ch, tot_spend, tot_resp, resp_hdi_10, resp_hdi_90, convergence = \
                    _parse_reverse_result(df_rev, spend_cols, cpp_w)

            else:
                logger.warning(f"  [SCENARIO] Unknown type '{sc.type}' — skipping.")
                continue

        except Exception as exc:
            import traceback
            logger.error(f"  [SCENARIO] Error in '{sc.name}': {exc}\n{traceback.format_exc()}")
            continue

        curr_resp_total = float(sum(
            _evaluate_at_allocation(
                curr_spend_pp, cpp_w, params_list, rscale, n_periods
            )[0]
        ))

        comparison_rows.append({
            "scenario"              : sc.name,
            "scenario_type"         : sc.type,
            "total_spend_£"         : round(tot_spend, 2),
            "current_spend_£"       : round(tot_curr_spend, 2),
            "pct_change_spend"      : round((tot_spend - tot_curr_spend) / (tot_curr_spend + 1e-12) * 100, 1),
            "total_response_mean"   : round(tot_resp, 4),
            "current_response"      : round(curr_resp_total, 4),
            "pct_change_response"   : round((tot_resp - curr_resp_total) / (curr_resp_total + 1e-12) * 100, 1),
            "response_hdi_10"       : round(resp_hdi_10, 4) if resp_hdi_10 is not None else None,
            "response_hdi_90"       : round(resp_hdi_90, 4) if resp_hdi_90 is not None else None,
            "overall_roi_£"         : round(tot_resp / (tot_spend + 1e-12), 4),
            "overall_cpa_£"         : round(tot_spend / (tot_resp + 1e-12), 4),
            "convergence_pct"       : convergence,
            "budget_period"         : budget_period,
        })

        for j, ch in enumerate(spend_cols):
            curr_r_ch = float(_evaluate_at_allocation(
                curr_spend_pp, cpp_w, params_list, rscale, n_periods
            )[0][j])
            curr_s_ch = float(curr_spend_pp[j] * n_periods)
            sp = float(spend_by_ch[j]) if j < len(spend_by_ch) else 0.0
            rp = float(resp_by_ch[j])  if j < len(resp_by_ch)  else 0.0

            detail_rows.append({
                "scenario"          : sc.name,
                "scenario_type"     : sc.type,
                "channel"           : ch,
                "metric_type"       : metric_types[j] if j < len(metric_types) else "Spend",
                "budget_period"     : budget_period,
                "current_spend_£"   : round(curr_s_ch, 2),
                "scenario_spend_£"  : round(sp, 2),
                "pct_change_spend"  : round((sp - curr_s_ch) / (curr_s_ch + 1e-12) * 100, 1),
                "current_response"  : round(curr_r_ch, 4),
                "scenario_response" : round(rp, 4),
                "pct_change_response": round((rp - curr_r_ch) / (curr_r_ch + 1e-12) * 100, 1),
                "roi_£"             : round(rp / (sp + 1e-12), 4),
                "cpa_£"             : round(sp / (rp + 1e-12), 4),
                "spend_share_pct"   : round(sp / (tot_spend + 1e-12) * 100, 1),
                "response_share_pct": round(rp / (tot_resp + 1e-12) * 100, 1),
            })

        logger.info(
            f"  [SCENARIO] {sc.name}: spend=£{tot_spend:,.0f} | "
            f"response={tot_resp:.4f} | ROI={tot_resp / (tot_spend + 1e-12):.4f}"
        )

    df_comparison   = pd.DataFrame(comparison_rows)
    df_channel_detail = pd.DataFrame(detail_rows)

    logger.info(f"\n[SCENARIOS] Complete — {len(comparison_rows)} scenario(s) evaluated.")
    if not df_comparison.empty:
        logger.info("\n" + df_comparison[
            ["scenario", "total_spend_£", "pct_change_spend",
             "total_response_mean", "pct_change_response", "overall_roi_£"]
        ].to_string(index=False))

    return df_comparison, df_channel_detail


def _run_current(curr_spend_pp, cpp_w, params_list, rscale, n_periods):
    from optimisation import _channel_response_raw
    C = len(params_list)
    resp = np.zeros(C)
    for j, p in enumerate(params_list):
        media_pp = curr_spend_pp[j] / (cpp_w[j] + 1e-30)
        resp[j] = _channel_response_raw(
            float(media_pp), p["spend_max"],
            lam      = float(p["lam"].mean()),
            beta     = float(p["beta"].mean()),
            alpha    = float(p["alpha"].mean()),
            kappa    = float(p["kappa"].mean()),
            k_log    = float(p["k_log"].mean()),
            x0       = float(p["x0"].mean()),
            sat      = p["sat_type"],
            ads_type = p.get("ads_type", "geometric"),
        )
    resp_full  = resp * rscale * n_periods
    spend_full = curr_spend_pp * n_periods
    return resp_full, spend_full


def _parse_forward_result(df_opt, spend_cols, cpp_w):
    ch_df       = df_opt[df_opt["channel"] != "TOTAL"]
    total_row   = df_opt[df_opt["channel"] == "TOTAL"].iloc[0]
    spend_col   = "optimal_spend_£_mean" if "optimal_spend_£_mean" in df_opt.columns else "optimal_spend_mean"
    resp_col    = "response_mean"

    spend_by_ch = np.array([
        float(ch_df.loc[ch_df["channel"] == ch, spend_col].iloc[0])
        if ch in ch_df["channel"].values else 0.0
        for ch in spend_cols
    ])
    resp_by_ch  = np.array([
        float(ch_df.loc[ch_df["channel"] == ch, resp_col].iloc[0])
        if ch in ch_df["channel"].values else 0.0
        for ch in spend_cols
    ])
    tot_spend   = float(total_row.get("optimal_spend_£_mean", total_row.get("optimal_spend_mean", 0)))
    tot_resp    = float(total_row.get(resp_col, 0))
    resp_hdi_10 = float(total_row.get("response_hdi_10", tot_resp))
    resp_hdi_90 = float(total_row.get("response_hdi_90", tot_resp))
    conv        = total_row.get("convergence_rate_pct", None)
    return resp_by_ch, spend_by_ch, tot_spend, tot_resp, resp_hdi_10, resp_hdi_90, conv


def _parse_reverse_result(df_rev, spend_cols, cpp_w):
    ch_df     = df_rev[df_rev["channel"] != "TOTAL"]
    total_row = df_rev[df_rev["channel"] == "TOTAL"].iloc[0]
    spend_col = "min_spend_£_mean" if "min_spend_£_mean" in df_rev.columns else "min_spend_mean"
    resp_col  = "achieved_response_mean"

    spend_by_ch = np.array([
        float(ch_df.loc[ch_df["channel"] == ch, spend_col].iloc[0])
        if ch in ch_df["channel"].values else 0.0
        for ch in spend_cols
    ])
    resp_by_ch  = np.array([
        float(ch_df.loc[ch_df["channel"] == ch, resp_col].iloc[0])
        if ch in ch_df["channel"].values else 0.0
        for ch in spend_cols
    ])
    tot_spend   = float(total_row.get("min_spend_£_mean", total_row.get("min_spend_mean", 0)))
    tot_resp    = float(total_row.get(resp_col, 0))
    resp_hdi_10 = float(total_row.get("achieved_response_hdi_10", tot_resp))
    resp_hdi_90 = float(total_row.get("achieved_response_hdi_90", tot_resp))
    conv        = total_row.get("convergence_rate_pct", None)
    return resp_by_ch, spend_by_ch, tot_spend, tot_resp, resp_hdi_10, resp_hdi_90, conv


def plot_scenario_comparison(
    df_comparison   : pd.DataFrame,
    df_channel_detail : pd.DataFrame,
    out_dir         : "Path",
    budget_period   : str = "monthly",
) -> None:
    """Generate a 3-panel scenario comparison chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        logger.warning("  matplotlib not available — skipping scenario comparison plots.")
        return

    if df_comparison.empty:
        return

    out_dir = _RCPath(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = df_comparison["scenario"].tolist()
    n_sc      = len(scenarios)

    fig = plt.figure(figsize=(16, max(6, 3 + n_sc * 0.7)))
    gs  = fig.add_gridspec(1, 3, wspace=0.45)

    PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    sc_colors = {sc: PALETTE[i % len(PALETTE)] for i, sc in enumerate(scenarios)}

    y_pos = np.arange(n_sc)

    ax1 = fig.add_subplot(gs[0])
    spends  = df_comparison["total_spend_£"].values
    bars1   = ax1.barh(y_pos, spends, color=[sc_colors[s] for s in scenarios],
                       height=0.6, edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(scenarios, fontsize=9)
    ax1.set_xlabel(f"Total Spend (£, {budget_period})", fontsize=10)
    ax1.set_title("Total Spend (£)", fontsize=11, fontweight="bold")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    ax1.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars1, spends):
        ax1.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                 f"£{v:,.0f}", va="center", fontsize=8)

    ax2 = fig.add_subplot(gs[1])
    resp    = df_comparison["total_response_mean"].values
    r_lo    = df_comparison.get("response_hdi_10", pd.Series([None] * n_sc)).values
    r_hi    = df_comparison.get("response_hdi_90", pd.Series([None] * n_sc)).values

    bars2 = ax2.barh(y_pos, resp, color=[sc_colors[s] for s in scenarios],
                     height=0.6, edgecolor="white")

    for i, (r, lo, hi) in enumerate(zip(resp, r_lo, r_hi)):
        if lo is not None and hi is not None:
            ax2.errorbar(r, i,
                         xerr=[[r - lo], [hi - r]],
                         fmt="none", color="black", capsize=3, lw=1.2)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([""] * n_sc)
    ax2.set_xlabel(f"Total Response (mean, {budget_period})", fontsize=10)
    ax2.set_title("Total Response", fontsize=11, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars2, resp):
        ax2.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                 f"{v:,.2f}", va="center", fontsize=8)

    ax3 = fig.add_subplot(gs[2])

    if not df_channel_detail.empty:
        channels_uniq = df_channel_detail["channel"].unique().tolist()
        ch_palette    = plt.cm.get_cmap("tab20", len(channels_uniq))
        ch_colors     = {ch: ch_palette(i) for i, ch in enumerate(channels_uniq)}

        lefts = np.zeros(n_sc)
        for ch in channels_uniq:
            ch_spends = []
            for sc in scenarios:
                mask  = (df_channel_detail["scenario"] == sc) & (df_channel_detail["channel"] == ch)
                val   = float(df_channel_detail.loc[mask, "scenario_spend_£"].sum()) if mask.any() else 0.0
                ch_spends.append(val)
            ax3.barh(y_pos, ch_spends, left=lefts, color=ch_colors[ch],
                     height=0.6, label=ch, edgecolor="white")
            lefts += np.array(ch_spends)

        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([""] * n_sc)
        ax3.set_xlabel(f"Spend by Channel (£, {budget_period})", fontsize=10)
        ax3.set_title("Channel Spend Breakdown", fontsize=11, fontweight="bold")
        ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
        ax3.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax3.grid(axis="x", alpha=0.3)

    fig.suptitle("Scenario Planning Comparison", fontsize=14, fontweight="bold", y=1.02)
    out_path = out_dir / "scenario_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")

    _plot_roi_comparison(df_channel_detail, scenarios, sc_colors, out_dir, budget_period)


def _plot_roi_comparison(df_detail, scenarios, sc_colors, out_dir, budget_period):
    """Side-by-side ROI per channel per scenario."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if df_detail.empty or "roi_£" not in df_detail.columns:
        return

    channels = df_detail["channel"].unique().tolist()
    n_ch  = len(channels)
    n_sc  = len(scenarios)
    width = 0.8 / n_sc
    x     = np.arange(n_ch)

    fig, ax = plt.subplots(figsize=(max(8, n_ch * 2), 5))

    for si, sc in enumerate(scenarios):
        sc_df  = df_detail[df_detail["scenario"] == sc]
        rois   = [float(sc_df.loc[sc_df["channel"] == ch, "roi_£"].mean())
                  if ch in sc_df["channel"].values else 0.0
                  for ch in channels]
        offset = (si - n_sc / 2 + 0.5) * width
        ax.bar(x + offset, rois, width=width * 0.9,
               label=sc, color=sc_colors.get(sc, "#888888"), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("ROI (response / £ spend)", fontsize=10)
    ax.set_title(f"Channel ROI by Scenario  ({budget_period})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", lw=0.8)

    fig.tight_layout()
    out_path = _RCPath(out_dir) / "scenario_roi_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


def export_scenarios(
    df_comparison     : pd.DataFrame,
    df_channel_detail : pd.DataFrame,
    out_dir           : "Path",
) -> None:
    """Save scenario results to CSV and (optionally) Excel."""
    out_dir = _RCPath(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = out_dir / "scenario_comparison.csv"
    df_comparison.to_csv(p1, index=False)
    logger.info(f"  Saved: {p1}")

    p2 = out_dir / "scenario_channel_detail.csv"
    df_channel_detail.to_csv(p2, index=False)
    logger.info(f"  Saved: {p2}")

    try:
        import openpyxl
        xlsx_path = out_dir / "scenario_planning.xlsx"
        with pd.ExcelWriter(str(xlsx_path), engine="openpyxl") as writer:
            df_comparison.to_excel(writer, sheet_name="Scenario Summary", index=False)
            df_channel_detail.to_excel(writer, sheet_name="Channel Detail", index=False)

            if not df_channel_detail.empty and "scenario_spend_£" in df_channel_detail.columns:
                pivot = df_channel_detail.pivot_table(
                    index="channel",
                    columns="scenario",
                    values="scenario_spend_£",
                    aggfunc="sum",
                )
                pivot.to_excel(writer, sheet_name="Spend Pivot")

            if not df_channel_detail.empty and "roi_£" in df_channel_detail.columns:
                pivot_roi = df_channel_detail.pivot_table(
                    index="channel",
                    columns="scenario",
                    values="roi_£",
                    aggfunc="mean",
                )
                pivot_roi.to_excel(writer, sheet_name="ROI Pivot")

        logger.info(f"  Saved: {xlsx_path}")
    except ImportError:
        logger.info("  openpyxl not installed — skipping Excel export (CSV files saved).")
    except Exception as exc:
        logger.warning(f"  Excel export failed: {exc}")


def build_preset_scenarios(
    sc_cfg          : Dict[str, Any],
    opt_cfg         : Dict[str, Any],
    curr_spend_total: float,
) -> "_ScList[ScenarioConfig]":
    """Build a standard set of budget scenarios for quick analysis."""
    budget_pcts     = sc_cfg.get("preset_pcts", opt_cfg.get("preset_pcts", [-20, -10, 10, 20]))
    include_current = sc_cfg.get("include_current", True)

    scenarios: "_ScList[ScenarioConfig]" = []

    if include_current:
        scenarios.append(ScenarioConfig(
            name="Current spend",
            type="current",
        ))

    for pct in budget_pcts:
        sign  = "+" if pct > 0 else ""
        label = f"{sign}{pct:g}% budget"
        scenarios.append(ScenarioConfig(
            name       = label,
            type       = "budget_pct",
            budget_pct = float(pct),
        ))

    return scenarios
