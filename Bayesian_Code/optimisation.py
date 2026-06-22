# optimisation.py
# ─────────────────────────────────────────────────────────────────────────────
# Budget optimisation using the fitted Bayesian MMM posterior.
#
# Three optimisers are available:
#
#   optimise_budget(...)          FORWARD — given a total budget, find the
#                                  channel allocation that maximises response.
#
#   minimise_spend_for_target(...)REVERSE — given a desired response level,
#                                  find the minimum spend to achieve it.
#
#   greedy_budget_allocation(...) GREEDY  — allocates budget step-by-step,
#                                  always giving the next unit to whichever
#                                  channel has the highest marginal ROI.
#
# All optimisers run over 200 posterior samples so you get a distribution
# of optimal allocations (not just a single point estimate).  The output
# tables show mean + HDI credible interval for every recommendation.
#
# Used by: analyze.py (optimise / reverse commands)
# ─────────────────────────────────────────────────────────────────────────────

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analysis import (
    _apply_saturation,
    _get_beta_samples,
    _get_lam_samples,
    _scalar_samples,
    _subsample,
    _stack_posterior,
)

logger = logging.getLogger("MMM")


# -----------------------------------------------------------------------------
# Period helpers
# -----------------------------------------------------------------------------

# Average model-periods per calendar month for each data frequency
_PERIODS_PER_MONTH = {
    "daily":   365.25 / 12.0,   # ~30.44
    "weekly":  52.18  / 12.0,   # ~4.35
    "monthly": 1.0,
}

VALID_PERIODS = (
    "per_period", "monthly", "quarterly", "yearly",
    "training_total", "training_mean",
)


def _periods_in_window(budget_period: str, frequency: str, n_train: int) -> float:
    """
    Return the number of model periods (weeks/days/months) contained in the
    requested budget window.

    Parameters
    ----------
    budget_period : one of "per_period" | "monthly" | "quarterly" | "yearly" |
                    "training_total" | "training_mean"
    frequency     : data frequency — "weekly" | "daily" | "monthly"
    n_train       : number of training periods in the dataset
    """
    bp   = budget_period.lower().strip()
    freq = frequency.lower().strip()
    ppm  = _PERIODS_PER_MONTH.get(freq, _PERIODS_PER_MONTH["weekly"])

    if bp in ("per_period", "training_mean"):
        return 1.0
    elif bp == "monthly":
        return ppm
    elif bp == "quarterly":
        return ppm * 3.0
    elif bp == "yearly":
        return ppm * 12.0
    elif bp == "training_total":
        return float(n_train)
    else:
        logger.warning(
            f"  [OPT] Unknown budget_period '{budget_period}' — "
            f"valid options: {VALID_PERIODS}. Falling back to per_period."
        )
        return 1.0


# -----------------------------------------------------------------------------
# Response functions — raw spend space
# -----------------------------------------------------------------------------

def _channel_response_scaled(
    x_scaled : float,
    lam      : float,
    beta     : float,
    alpha    : float,
    kappa    : float,
    k_log    : float,
    x0       : float,
    sat      : str,
    ads_type : str = "geometric",
) -> float:
    """Response for one channel at a SCALED spend level (0-1 range).

    Geometric adstock uses the scan recurrence s[t] = lam*s[t-1] + x[t],
    whose steady-state for constant x is x / (1 - lam).

    Weibull adstock uses a L1-normalised truncated convolution (no scan form),
    so weights sum to 1 and the steady-state for constant x is just x.
    """
    if ads_type == "weibull":
        x_ads = x_scaled
    else:
        x_ads = x_scaled / (1.0 - np.clip(lam, 0.0, 0.9999) + 1e-12)
    s = _apply_saturation(
        np.array([x_ads]), sat,
        alpha=np.array([alpha]),
        kappa=np.array([kappa]),
        k_log=np.array([k_log]),
        x0=np.array([x0]),
    )
    return float(beta * s[0])


def _channel_response_raw(
    x_raw     : float,
    spend_max : float,
    lam       : float,
    beta      : float,
    alpha     : float,
    kappa     : float,
    k_log     : float,
    x0        : float,
    sat       : str,
    ads_type  : str = "geometric",
) -> float:
    """
    Response for one channel at a RAW spend level.
    Converts to scaled internally via spend_max.
    """
    x_scaled = x_raw / (spend_max + 1e-8)
    return _channel_response_scaled(x_scaled, lam, beta, alpha, kappa, k_log, x0, sat, ads_type)


def _total_response_raw(
    x_alloc_raw    : np.ndarray,
    params         : List[Dict],
    channel_rscales: Optional[np.ndarray] = None,
) -> float:
    """Total response across all channels for a raw spend allocation.

    When channel_rscales is provided each channel's model-space response is
    multiplied by its own per-channel rscale before summing, so the returned
    value is in original KPI units (deals).  When None, model-space units are
    returned (legacy behaviour).
    """
    total = 0.0
    for j, p in enumerate(params):
        cr = float(channel_rscales[j]) if channel_rscales is not None else 1.0
        total += _channel_response_raw(
            x_alloc_raw[j], p["spend_max"],
            lam=p["lam"], beta=p["beta"], alpha=p["alpha"],
            kappa=p["kappa"], k_log=p["k_log"], x0=p["x0"],
            sat=p["sat_type"],
            ads_type=p.get("ads_type", "geometric"),
        ) * cr
    return total


def _neg_total_response_raw(
    x_alloc_raw    : np.ndarray,
    params         : List[Dict],
    channel_rscales: Optional[np.ndarray] = None,
) -> float:
    return -_total_response_raw(x_alloc_raw, params, channel_rscales)


def _neg_total_response_raw_grad(
    x_alloc_raw    : np.ndarray,
    params         : List[Dict],
    channel_rscales: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Numerical gradient in raw media-metric space."""
    grad = np.zeros_like(x_alloc_raw)
    eps  = max(float(np.abs(x_alloc_raw).mean()), 1.0) * 1e-5
    for j in range(len(x_alloc_raw)):
        xp = x_alloc_raw.copy(); xp[j] += eps
        xm = x_alloc_raw.copy(); xm[j] -= eps
        grad[j] = (_neg_total_response_raw(xp, params, channel_rscales) -
                   _neg_total_response_raw(xm, params, channel_rscales)) / (2 * eps)
    return grad


# ── Spend-space equivalents (used when cpp_weights are provided) ─────────────
# Working in GBP spend eliminates numerical issues caused by large impression
# counts: eps ≈ mean(spend) × 1e-5 is in GBP (e.g. £0.14), not impressions.

def _total_response_from_spend(
    x_spend        : np.ndarray,
    cpp_w          : np.ndarray,
    params         : List[Dict],
    channel_rscales: Optional[np.ndarray] = None,
) -> float:
    """Total response from a SPEND allocation (GBP per period)."""
    x_media = x_spend / (cpp_w + 1e-30)
    return _total_response_raw(x_media, params, channel_rscales)


def _neg_total_response_from_spend(
    x_spend        : np.ndarray,
    cpp_w          : np.ndarray,
    params         : List[Dict],
    channel_rscales: Optional[np.ndarray] = None,
) -> float:
    return -_total_response_from_spend(x_spend, cpp_w, params, channel_rscales)


def _neg_total_response_from_spend_grad(
    x_spend        : np.ndarray,
    cpp_w          : np.ndarray,
    params         : List[Dict],
    channel_rscales: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Numerical gradient in GBP spend space — well-conditioned for all channel sizes."""
    grad = np.zeros_like(x_spend)
    eps  = max(float(np.abs(x_spend).mean()), 0.01) * 1e-5
    for j in range(len(x_spend)):
        xp = x_spend.copy(); xp[j] += eps
        xm = x_spend.copy(); xm[j] -= eps
        grad[j] = (_neg_total_response_from_spend(xp, cpp_w, params, channel_rscales) -
                   _neg_total_response_from_spend(xm, cpp_w, params, channel_rscales)) / (2 * eps)
    return grad


# -----------------------------------------------------------------------------
# Parameter extraction
# -----------------------------------------------------------------------------

def _coalesce(arr, default):
    """Return arr if not None, else default. Safe for numpy arrays."""
    return arr if arr is not None else default


def _extract_all_channel_params(
    best      : Dict[str, Any],
    prep      : Dict[str, Any],
    n_samples : int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Extract posterior parameter arrays for every channel.

    Returns
    -------
    spend_cols  : list of channel names
    params_list : list of dicts — one per channel, each containing:
                    spend_max  : float  (raw spend scale for that channel)
                    sat_type   : str
                    lam, beta, alpha, kappa, k_log, x0 : (n_samples,) arrays
    """
    trace         = best["trace"]
    spend_cols    = prep["spend_cols"]
    C             = prep["C"]
    channel_specs = best.get("channel_specs", {})
    train_idx     = prep["train_idx"]

    X_raw = prep["X_media_raw"][train_idx]

    params_list = []
    for j, ch_name in enumerate(spend_cols):
        spec     = channel_specs.get(j)
        sat_type = spec.saturation   if spec else best["cfg"].saturation
        ads_type = spec.adstock_type if spec else best["cfg"].adstock_type

        # Per-channel spend scale: max raw value used to normalise the column
        if X_raw.ndim == 3:
            spend_max = float(X_raw[:, 0, j].max()) + 1e-8
        else:
            spend_max = float(X_raw[:, j].max()) + 1e-8

        lam   = _subsample(_get_lam_samples(trace, j), n_samples)
        beta  = _subsample(_get_beta_samples(trace, j, C), n_samples)
        alpha = _subsample(_coalesce(_scalar_samples(trace, "alpha_sat",  idx=j), np.ones(n_samples)),  n_samples)
        kappa = _subsample(_coalesce(_scalar_samples(trace, "kappa",      idx=j), np.ones(n_samples)),  n_samples)
        k_log = _subsample(_coalesce(_scalar_samples(trace, "k_logistic", idx=j), np.ones(n_samples)),  n_samples)
        x0    = _subsample(_coalesce(_scalar_samples(trace, "x0",         idx=j), np.zeros(n_samples)), n_samples)

        S = min(len(lam), len(beta), len(alpha), n_samples)
        params_list.append({
            "channel"  : ch_name,
            "sat_type" : sat_type,
            "ads_type" : ads_type,
            "spend_max": spend_max,
            "lam"      : lam[:S],
            "beta"     : beta[:S],
            "alpha"    : alpha[:S],
            "kappa"    : kappa[:S],
            "k_log"    : k_log[:S],
            "x0"       : x0[:S],
            "n_samples": S,
        })

    return spend_cols, params_list


def _params_at_sample(params_list: List[Dict], si: int) -> List[Dict]:
    """Extract a scalar parameter dict for posterior sample index si."""
    return [
        {
            "sat_type" : p["sat_type"],
            "ads_type" : p.get("ads_type", "geometric"),
            "spend_max": p["spend_max"],
            "lam"      : float(p["lam"][si]),
            "beta"     : float(p["beta"][si]),
            "alpha"    : float(p["alpha"][si]),
            "kappa"    : float(p["kappa"][si]),
            "k_log"    : float(p["k_log"][si]),
            "x0"       : float(p["x0"][si]),
        }
        for p in params_list
    ]


def _response_scale(prep: Dict[str, Any]) -> float:
    """
    Proportional scale factor that converts model-space contributions
    (transformed units) to approximate original response units.

    response_original ≈ response_model * response_scale

    Derivation: channel contributions are in the same units as the response
    transform.  Multiplying by E[y_raw] / E[transform(y_raw)] rescales
    them proportionally to original units.
    """
    train_idx = prep["train_idx"]
    y_train   = prep["y_raw"][train_idx]
    rt        = prep.get("response_transform", "log1p")
    bl        = prep.get("boxcox_lambda")
    y_mean    = float(y_train.mean())

    if rt == "log1p":
        y_t_mean = float(np.log1p(np.maximum(y_train, 0.0)).mean())
    elif rt == "sqrt":
        y_t_mean = float(np.sqrt(np.maximum(y_train, 0.0)).mean())
    elif rt == "boxcox":
        if bl is None or abs(bl) < 1e-10:
            y_t_mean = float(np.log(np.maximum(y_train, 1e-6)).mean())
        else:
            y_t_mean = float(((np.maximum(y_train, 1e-6) ** bl - 1.0) / bl).mean())
    else:
        # identity: contributions are on the same scale as y_raw already
        y_t_mean = y_mean

    return y_mean / y_t_mean if abs(y_t_mean) > 1e-10 else 1.0


def _compute_response_scale_from_trace(best: Dict, prep: Dict[str, Any]) -> float:
    """
    Correct rscale using the posterior baseline level from the trace.

    The model betas live in y_scaled space (standardised log1p), so the naive
    rscale = E[y_raw]/E[log1p(y_raw)] is wrong — it ignores y_std and the
    non-linear baseline shift.

    The correct linearised conversion is:
        response_j ≈ beta_j × sat_j × y_std × (y_base + 1)
    where y_base = expm1(mu_base_z × y_std + y_mu) is the predicted response
    with zero media spend (baseline + seasonality + random-walk only).

    Falls back to the legacy _response_scale() if the trace is unavailable or
    if the response transform is not log1p.
    """
    rt = prep.get("response_transform", "log1p")
    if rt != "log1p":
        return _response_scale(prep)

    try:
        trace    = best["trace"]
        C        = prep["C"]
        T        = prep["X_media_raw"].shape[0]
        mu_post  = trace.posterior["mu"].values.reshape(-1, T)           # (N, T)
        mbc_post = trace.posterior["media_by_channel"].values            # (ch, dr, T, [P,] C)
        if mbc_post.ndim == 5:
            mbc_post = mbc_post.sum(axis=3)                              # collapse P
        mbc_post  = mbc_post.reshape(-1, T, C)                          # (N, T, C)
        mu_base_z = float((mu_post - mbc_post.sum(axis=2)).mean())
        y_std     = float(prep["y_std"])
        y_mu      = float(prep["y_mu"])
        y_base    = float(np.expm1(mu_base_z * y_std + y_mu))
        rscale    = float(y_std * (y_base + 1.0))
        logger.info(
            f"  [RSCALE] mu_base_z={mu_base_z:.4f}  y_base={y_base:.2f}  "
            f"rscale={rscale:.4f}  (y_std x (y_base+1))"
        )
        return max(rscale, 1e-6)
    except Exception as exc:
        logger.warning(f"  [RSCALE] fallback to legacy rscale ({exc})")
        return _response_scale(prep)


def _calibrate_betas_to_trace(
    params_list : List[Dict],
    best        : Dict,
    prep        : Dict[str, Any],
) -> List[Dict]:
    """
    Scale beta samples by a per-channel Jensen's-inequality correction factor.

    Problem: E[beta × sat(alpha × x)] ≠ E[beta] × sat(E[alpha] × x)
    because sat() is highly non-linear.  Using mean parameters overestimates
    the posterior predictive mean by a large, channel-specific amount (up to
    45× for channels with right-skewed alpha posteriors and high saturation).

    Fix: compute a calibration factor per channel:
        calib_j = trace_mean_j / sim_mean_j
    where trace_mean_j is the actual posterior-mean weekly contribution from the
    trace's 'media_by_channel' variable, and sim_mean_j is what we'd get by
    simulating with posterior-mean parameters over the training spend history.

    Applying calib_j to the beta samples ensures that at historical spend levels
    the optimizer's response matches the model's posterior predictive mean.
    """
    try:
        trace    = best["trace"]
        C        = prep["C"]
        T        = prep["X_media_raw"].shape[0]
        mbc      = trace.posterior["media_by_channel"].values
        if mbc.ndim == 5:
            mbc = mbc.sum(axis=3)                                        # collapse P
        mbc      = mbc.reshape(-1, T, C)                                 # (N, T, C)
        trace_mean_per_week = mbc.mean(axis=(0, 1))                      # (C,)
    except Exception as exc:
        logger.warning(f"  [CALIB] Cannot read trace media_by_channel — skipping calibration ({exc})")
        return params_list

    train_idx = prep["train_idx"]
    X_sc      = prep["X_media_scaled"][train_idx]                        # (T_train, C)
    T_train   = X_sc.shape[0]

    calibrated = []
    for j, p in enumerate(params_list):
        lam_m   = float(p["lam"].mean())
        alpha_m = float(p["alpha"].mean())
        beta_m  = float(p["beta"].mean())
        kap_m   = float(p["kappa"].mean())
        sat_t   = p["sat_type"]

        carry = 0.0
        contrib_sum = 0.0
        for t in range(T_train):
            x   = float(X_sc[t, j])
            # carry already holds lam * s_{t-1} from the previous iteration.
            # Multiplying by lam_m again would give lam² decay — use carry directly.
            s_t = x + carry
            if sat_t == "softplus":
                sv = float(np.log1p(np.exp(np.clip(alpha_m * s_t, -500.0, 500.0))) / np.log(2.0))
            elif sat_t == "hill":
                xp = max(s_t, 1e-12)
                sv = float(xp ** alpha_m / (xp ** alpha_m + kap_m ** alpha_m))
            else:
                sv = float(_apply_saturation(
                    np.array([s_t]), sat_t,
                    np.array([alpha_m]), np.array([kap_m]),
                    np.array([1.0]),    np.array([0.0]),
                )[0])
            contrib_sum += beta_m * sv
            carry = lam_m * s_t

        sim_mean_j   = contrib_sum / max(T_train, 1)
        trace_mean_j = float(trace_mean_per_week[j])

        if sim_mean_j > 1e-12 and trace_mean_j > 1e-12:
            calib = trace_mean_j / sim_mean_j
        else:
            calib = 1.0

        ch_name = p.get("channel", f"ch{j}")
        logger.info(
            f"  [CALIB] {ch_name}: trace={trace_mean_j:.5f}  "
            f"sim={sim_mean_j:.5f}  calib={calib:.5f}"
        )
        p_new        = dict(p)
        p_new["beta"] = p["beta"] * calib
        calibrated.append(p_new)

    return calibrated


def _current_response_per_period(
    j               : int,
    current_spend_per_period: np.ndarray,
    params_list     : List[Dict],
    channel_rscale_j: float,
) -> float:
    """Current posterior-mean response for channel j, per model period, original units."""
    p = params_list[j]
    return _channel_response_raw(
        current_spend_per_period[j],
        p["spend_max"],
        float(p["lam"].mean()),
        float(p["beta"].mean()),
        float(p["alpha"].mean()),
        float(p["kappa"].mean()),
        float(p["k_log"].mean()),
        float(p["x0"].mean()),
        p["sat_type"],
        ads_type=p.get("ads_type", "geometric"),
    ) * channel_rscale_j


def _achievable_response_range(
    lb             : np.ndarray,
    ub             : np.ndarray,
    params_list    : List[Dict],
    channel_rscales: np.ndarray,
) -> tuple:
    """
    Compute the approximate min / max achievable total response when each channel
    spends at its lower / upper bound respectively.

    Uses the POSTERIOR MEAN parameters for speed (not a full sample loop).
    Returns (lower_achievable, upper_achievable) in original units per model period.
    """
    mean_params = [
        {
            "sat_type" : p["sat_type"],
            "ads_type" : p.get("ads_type", "geometric"),
            "spend_max": p["spend_max"],
            "lam"      : float(p["lam"].mean()),
            "beta"     : float(p["beta"].mean()),
            "alpha"    : float(p["alpha"].mean()),
            "kappa"    : float(p["kappa"].mean()),
            "k_log"    : float(p["k_log"].mean()),
            "x0"       : float(p["x0"].mean()),
        }
        for p in params_list
    ]
    lower = _total_response_raw(lb, mean_params, channel_rscales)
    upper = _total_response_raw(ub, mean_params, channel_rscales)
    return float(lower), float(upper)


def _compute_channel_rscales_from_contributions(
    contributions_csv : str,
    prep              : Dict[str, Any],
    channel_specs     : Optional[Dict] = None,
) -> Optional[np.ndarray]:
    """
    Fallback: compute per-channel rscales from a saved channel_contributions.csv.

    Uses the model-space response at current observed spend (computed from
    posterior-mean parameters stored in the CSV alongside contributions) as the
    denominator, and the properly-decomposed mean weekly contribution as the
    numerator:

        channel_rscale_j = mean_contribution_j (deals/week)
                         / (beta_j × Hill(x_current_j ; alpha_j, kappa_j))

    This allows the optimizer to produce correct deal-level numbers even when
    the trace cannot be loaded (e.g., when running on a machine without the
    full trace file).  Returns None if the CSV does not exist or is missing
    required columns.
    """
    import os
    if not os.path.exists(contributions_csv):
        return None

    try:
        cc_df       = pd.read_csv(contributions_csv)
        spend_cols  = prep.get("spend_cols", [])
        C           = prep["C"]
        train_idx   = prep["train_idx"]
        X_raw       = prep["X_media_raw"][train_idx]
        if X_raw.ndim == 3:
            current_spend_pp = X_raw[:, 0, :].mean(axis=0)
        else:
            current_spend_pp = X_raw.mean(axis=0)
        spend_max   = prep["X_media_raw"].max(axis=0)

        cc_map      = dict(zip(cc_df["channel"], cc_df["mean_contribution"]))
        channel_rscales = np.zeros(C)

        for j, ch in enumerate(spend_cols):
            delta_y_j = float(cc_map.get(ch, 0.0))
            if delta_y_j <= 0:
                continue

            # Model-space response at current spend using posterior-mean params
            # from the contributions CSV (roi_proxy = contribution / mean_spend,
            # so we can recover beta*sat from contribution / 1 directly if we
            # know the saturation curve).  Use a safe ratio fallback when
            # channel_specs are not available.
            x_sc = float(current_spend_pp[j]) / (float(spend_max[j]) + 1e-8)

            # Retrieve adstock/saturation specs if channel_specs provided
            ads_type = "geometric"
            lam_mean = 0.0
            if channel_specs is not None:
                spec = channel_specs.get(j)
                if spec is not None:
                    ads_type = getattr(spec, "adstock_type", "geometric")
                    # lam from az_summary not available here; use 0 as conservative
                    lam_mean = 0.0

            x_ads = x_sc if ads_type == "weibull" else x_sc  # simplified: use x_sc
            # Without posterior params, approximate model-space response as x_ads
            # and rely entirely on the ratio delta_y / model_space
            # The roi_proxy column = mean_contribution / mean_weekly_spend (in media units)
            row = cc_df[cc_df["channel"] == ch]
            if not row.empty and "roi_proxy" in row.columns and "mean_weekly_spend" in row.columns:
                mean_spend_pp = float(row["mean_weekly_spend"].iloc[0])
                roi_proxy     = float(row["roi_proxy"].iloc[0])
                # roi_proxy = contribution / mean_spend → contribution = roi*spend
                # model_response_at_current ≈ (current_spend_pp / mean_spend_pp) * contribution
                # But saturation means this isn't linear. Use ratio of spends via x_sc as proxy.
                x_frac        = float(current_spend_pp[j]) / (mean_spend_pp + 1e-30)
                model_resp_approx = delta_y_j * min(x_frac, 1.0)
                if model_resp_approx > 1e-12:
                    # channel_rscale not needed in the traditional sense here;
                    # instead scale so that at current spend we get delta_y_j
                    channel_rscales[j] = delta_y_j / (delta_y_j * min(x_frac, 1.0) + 1e-30)
                else:
                    channel_rscales[j] = 1.0
            else:
                # Simple fallback: no saturation info available
                channel_rscales[j] = 1.0

            logger.info(
                f"  [CH_RSCALE/CSV] {ch:<45s}  Δy={delta_y_j:.4f}  "
                f"rscale={channel_rscales[j]:.4f}"
            )

        return channel_rscales

    except Exception as exc:
        logger.warning(f"  [CH_RSCALE/CSV] Could not compute from contributions CSV: {exc}")
        return None


def _compute_channel_rscales(
    best                   : Dict[str, Any],
    prep                   : Dict[str, Any],
    contributions_csv      : Optional[str] = None,
) -> np.ndarray:
    """
    Compute per-channel rscales that properly convert model-space contributions
    to original KPI units via the full non-linear log1p back-transform.

    The global ``rscale = y_std × (y_base + 1)`` is a first-order linear
    approximation.  For channels with large saturation effects it underestimates
    true deal contributions by up to 10–30×, because the log1p transform means
    media multiplies the baseline rather than adding to it.

    This function computes the correct scaling factor per channel as:

        channel_rscale_j = E[Δy_j_original] / E[Δz_j_model]

    where:
        Δy_j_original = expm1(μ × σ + μ₀) − expm1((μ − mbc_j) × σ + μ₀)
        Δz_j_model    = mbc_j   (model-space contribution from trace)

    averaging over all posterior draws and training time steps.

    Falls back in order:
      1. ``contributions_csv`` path if provided (loads pre-computed contributions)
      2. ``<out_dir>/model_results/channel_contributions.csv`` if discoverable
      3. Global linearised rscale broadcast to all channels

    Parameters
    ----------
    best                : best-model dict (must contain trace for primary path)
    prep                : data prep dict
    contributions_csv   : optional explicit path to channel_contributions.csv
    """
    C  = prep["C"]
    rt = prep.get("response_transform", "log1p")

    if rt != "log1p":
        rscale = _compute_response_scale_from_trace(best, prep)
        logger.info(
            f"  [CH_RSCALE] Non-log1p transform ({rt}) — using global rscale "
            f"{rscale:.4f} for all channels."
        )
        return np.full(C, rscale)

    # ── Primary path: compute from trace ─────────────────────────────────────
    try:
        trace  = best["trace"]
        y_std  = float(prep["y_std"])
        y_mu   = float(prep["y_mu"])
        T      = prep["X_media_raw"].shape[0]

        mu_post  = trace.posterior["mu"].values.reshape(-1, T)      # (N, T) — std log1p
        mbc_post = trace.posterior["media_by_channel"].values
        if mbc_post.ndim == 5:
            mbc_post = mbc_post.sum(axis=3)                         # collapse lag dim
        mbc_post = mbc_post.reshape(-1, T, C)                       # (N, T, C)

        # Full prediction in original units
        y_pred = np.expm1(mu_post * y_std + y_mu)                   # (N, T)

        channel_rscales = np.zeros(C)
        spend_cols      = prep.get("spend_cols", [f"ch{j}" for j in range(C)])

        for j in range(C):
            mu_no_j   = mu_post - mbc_post[:, :, j]                 # (N, T)
            y_no_j    = np.expm1(mu_no_j * y_std + y_mu)            # (N, T)

            delta_y_j = float((y_pred - y_no_j).mean())             # original units
            delta_z_j = float(mbc_post[:, :, j].mean())             # model space

            if abs(delta_z_j) > 1e-12 and delta_y_j > 1e-12:
                channel_rscales[j] = delta_y_j / delta_z_j
            else:
                channel_rscales[j] = _compute_response_scale_from_trace(best, prep)
                logger.warning(
                    f"  [CH_RSCALE] {spend_cols[j]}: near-zero contribution — "
                    "falling back to global rscale."
                )

            logger.info(
                f"  [CH_RSCALE] {spend_cols[j]:<45s} "
                f"Δy={delta_y_j:.4f}  Δz={delta_z_j:.6f}  "
                f"rscale={channel_rscales[j]:.4f}"
            )

        return channel_rscales

    except Exception as exc:
        logger.warning(
            f"  [CH_RSCALE] Trace-based computation failed ({exc}). "
            "Trying channel_contributions.csv fallback."
        )

    # ── Fallback 1: load from contributions CSV ───────────────────────────────
    import os as _os
    _csv_candidates = []
    if contributions_csv:
        _csv_candidates.append(contributions_csv)
    # Auto-discover: look for channel_contributions.csv next to prep.pkl
    _prep_dir = best.get("_out_dir") or prep.get("_out_dir")
    if _prep_dir:
        _csv_candidates.append(
            _os.path.join(str(_prep_dir), "model_results", "channel_contributions.csv")
        )

    channel_specs = best.get("channel_specs")
    for _csv in _csv_candidates:
        _result = _compute_channel_rscales_from_contributions(_csv, prep, channel_specs)
        if _result is not None:
            logger.info(
                f"  [CH_RSCALE] Using channel_contributions.csv fallback: {_csv}"
            )
            return _result

    # ── Fallback 2: global linearised rscale ─────────────────────────────────
    logger.warning(
        "  [CH_RSCALE] All fallbacks exhausted — using global linearised rscale. "
        "Pass contributions_csv= or ensure trace.nc is readable for accurate results."
    )
    rscale = _compute_response_scale_from_trace(best, prep)
    return np.full(C, rscale)


# -----------------------------------------------------------------------------
# Forward optimisation: maximise response given a total budget
# -----------------------------------------------------------------------------

def optimise_budget(
    best          : Dict[str, Any],
    prep          : Dict[str, Any],
    total_budget  : Optional[float] = None,
    budget_period : str             = "training_mean",
    n_samples     : int             = 200,
    channel_min   : Optional[Dict[str, float]] = None,
    channel_max   : Optional[Dict[str, float]] = None,
    cpp_weights   : Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Forward budget optimisation: given a fixed total budget for a specified
    time period, find the channel allocation that maximises expected response.

    Parameters
    ----------
    best          : best-model dict returned by the pipeline
    prep          : data prep dict returned by the pipeline
    total_budget  : total budget for the given period.
                    When cpp_weights is None  → raw media-metric units (legacy).
                    When cpp_weights provided → SPEND (£) units. The optimiser
                    converts internally: media_budget_j = spend_j / CPP_j.
                    None → use observed mean spend for the period.
    budget_period : "per_period" | "monthly" | "quarterly" | "yearly" |
                    "training_total" | "training_mean"
    n_samples     : number of posterior draws to optimise over (uncertainty)
    channel_min   : {channel: lower_bound} per channel for the period.
                    Interpreted as spend (£) when cpp_weights provided.
    channel_max   : {channel: upper_bound} per channel for the period.
                    Interpreted as spend (£) when cpp_weights provided.
    cpp_weights   : (C,) array of cost-per-unit values (from cpp_layer.build_cpp_map).
                    When provided, budget and bounds are in SPEND; the solver
                    enforces sum(x_j * CPP_j) = total_budget_spend.
                    When None, legacy behaviour: all media metrics treated as spend.

    Returns
    -------
    DataFrame with one row per channel + a TOTAL summary row.
    Includes spend_equiv_mean/hdi columns (actual £ spend) when CPP differs from 1.
    """
    spend_cols, params_list = _extract_all_channel_params(best, prep, n_samples)
    params_list = _calibrate_betas_to_trace(params_list, best, prep)
    C         = len(spend_cols)
    train_idx = prep["train_idx"]
    n_train   = len(train_idx)
    frequency = prep.get("frequency", "weekly")

    X_raw = prep["X_media_raw"][train_idx]
    if X_raw.ndim == 3:
        current_spend_pp = X_raw[:, 0, :].mean(axis=0)   # (C,) media per model-period
    else:
        current_spend_pp = X_raw.mean(axis=0)

    # CPP weights — ones if not provided (media metric == spend for all channels)
    cpp_w = np.asarray(cpp_weights, dtype=float) if cpp_weights is not None else np.ones(C)

    # ── Period conversion ───────────────────────────────────────────────────
    n_periods = _periods_in_window(budget_period, frequency, n_train)

    # total_budget in spend units (or media if cpp_w all ones)
    if total_budget is None:
        total_budget = float((current_spend_pp * cpp_w).sum()) * n_periods
        logger.info(
            f"  [OPT] total_budget not specified — using observed "
            f"{budget_period} spend: {total_budget:,.2f}"
        )

    budget_pp_spend = total_budget / n_periods   # spend per model-period

    logger.info(
        f"  [OPT FORWARD] period={budget_period} ({n_periods:.2f} model-periods) | "
        f"total_budget_spend={total_budget:,.2f} | per-period-spend={budget_pp_spend:,.4f}"
    )

    # ── Bounds in MEDIA-METRIC per-period units ─────────────────────────────
    # channel_min/max are in spend (£) when cpp_w provided; divide by CPP → media.
    lb = np.zeros(C)
    ub = np.array([budget_pp_spend / (cpp_w[j] + 1e-30) for j in range(C)])

    for j, ch in enumerate(spend_cols):
        if channel_min and ch in channel_min:
            lb[j] = float(channel_min[ch]) / (n_periods * (cpp_w[j] + 1e-30))
        if channel_max and ch in channel_max:
            ub[j] = float(channel_max[ch]) / (n_periods * (cpp_w[j] + 1e-30))

    bounds = [(lb[j], ub[j]) for j in range(C)]

    # Constraint: sum(x_j * CPP_j) = budget_pp_spend  (spend-weighted equality)
    constraints = [
        {"type": "eq",
         "fun": lambda x, bps=budget_pp_spend, cw=cpp_w: (x * cw).sum() - bps}
    ]

    # Initial guess: scale current media allocation to match budget
    scale = budget_pp_spend / max((current_spend_pp * cpp_w).sum(), 1e-12)
    x0_init = current_spend_pp * scale
    x0_init = np.clip(x0_init, lb, ub)

    # ── Per-channel rscales (proper non-linear back-transform) ──────────────
    channel_rscales = _compute_channel_rscales(best, prep)

    min_ach_pp, max_ach_pp = _achievable_response_range(lb, ub, params_list, channel_rscales)
    min_ach = min_ach_pp * n_periods
    max_ach = max_ach_pp * n_periods
    logger.info(
        f"  [ACHIEVABILITY] response achievable within bounds: "
        f"[{min_ach:,.4f}, {max_ach:,.4f}]"
    )

    # ── SLSQP over posterior samples (working in SPEND space) ───────────────
    # Variables: x_spend_j = spend per channel per period (GBP).
    # This avoids numerical issues caused by large impression counts in media space.
    # x_spend_j = x_media_j * cpp_w[j]  — convert back after optimisation.
    S = params_list[0]["n_samples"]
    optimal_alloc  = np.zeros((S, C))   # media units per period
    optimal_resp   = np.zeros((S, C))   # original KPI units per period
    bounds_ok_mask = np.ones(S, dtype=bool)

    tol_bounds = 0.01

    # Spend-space bounds: lb_spend = lb * cpp_w, ub_spend = ub * cpp_w
    lb_spend = lb * cpp_w
    ub_spend = ub * cpp_w
    bounds_spend = [(lb_spend[j], ub_spend[j]) for j in range(C)]
    constraints_spend = [
        {"type": "eq", "fun": lambda x: x.sum() - budget_pp_spend}
    ]
    x0_spend = x0_init * cpp_w

    n_failed = 0
    for si in range(S):
        params_si = _params_at_sample(params_list, si)
        result = minimize(
            fun         = _neg_total_response_from_spend,
            x0          = x0_spend,
            args        = (cpp_w, params_si, channel_rscales),
            method      = "SLSQP",
            bounds      = bounds_spend,
            constraints = constraints_spend,
            jac         = _neg_total_response_from_spend_grad,
            options     = {"ftol": 1e-12, "maxiter": 500, "disp": False},
        )
        if result.success:
            alloc_spend = np.clip(result.x, lb_spend, ub_spend)
            budget_err  = abs(alloc_spend.sum() - budget_pp_spend) / (budget_pp_spend + 1e-12)
            lb_ok = np.all(alloc_spend >= lb_spend * (1 - tol_bounds) - 1e-9)
            ub_ok = np.all(alloc_spend <= ub_spend * (1 + tol_bounds) + 1e-9)
            bounds_ok_mask[si] = lb_ok and ub_ok and (budget_err <= tol_bounds)
        else:
            alloc_spend = x0_spend.copy()
            n_failed += 1
            bounds_ok_mask[si] = False

        alloc = alloc_spend / (cpp_w + 1e-30)   # convert back to media units
        optimal_alloc[si, :] = alloc
        for j, p in enumerate(params_si):
            optimal_resp[si, j] = _channel_response_raw(
                alloc[j], p["spend_max"],
                lam=p["lam"], beta=p["beta"], alpha=p["alpha"],
                kappa=p["kappa"], k_log=p["k_log"], x0=p["x0"],
                sat=p["sat_type"],
                ads_type=p.get("ads_type", "geometric"),
            ) * channel_rscales[j]

    convergence_rate = round((S - n_failed) / S * 100, 1)
    bounds_respected = bool(bounds_ok_mask.all())

    if n_failed:
        logger.warning(
            f"  [OPTIMISE] {n_failed}/{S} SLSQP runs did not converge — "
            "using current spend allocation as fallback for those samples."
        )
    if not bounds_respected:
        n_viol = int((~bounds_ok_mask).sum())
        logger.warning(
            f"  [OPTIMISE] {n_viol}/{S} samples violated bounds or budget "
            "constraint (>1% tolerance)."
        )

    # ── Build output (scale up to full period) ───────────────────────────────
    metric_types = prep.get("metric_types", ["Spend"] * C)

    rows = []
    for j, ch in enumerate(spend_cols):
        alloc_j  = optimal_alloc[:, j] * n_periods          # full period, media units
        resp_j   = optimal_resp[:, j] * n_periods            # full period, original KPI units
        curr_sp  = current_spend_pp[j] * n_periods          # media units
        curr_r   = _current_response_per_period(j, current_spend_pp, params_list, channel_rscales[j]) * n_periods

        opt_sp_mean = float(alloc_j.mean())
        resp_mean   = float(resp_j.mean())

        # Spend equivalents (actual £ cost) via CPP
        alloc_j_spend = alloc_j * cpp_w[j]
        curr_sp_spend = curr_sp * cpp_w[j]
        opt_sp_spend  = float(alloc_j_spend.mean())

        pct_chg_sp   = round((opt_sp_spend - curr_sp_spend) / (curr_sp_spend + 1e-12) * 100, 2)
        pct_chg_resp = round((resp_mean    - curr_r)        / (curr_r   + 1e-12) * 100, 2)
        current_roi  = round(curr_r     / (curr_sp_spend + 1e-12), 6)
        optimal_roi  = round(resp_mean  / (opt_sp_spend  + 1e-12), 6)
        current_cpa  = round(curr_sp_spend / (curr_r    + 1e-12), 6)
        optimal_cpa  = round(opt_sp_spend  / (resp_mean + 1e-12), 6)

        rows.append({
            "channel"                : ch,
            "metric_type"            : metric_types[j] if j < len(metric_types) else "Spend",
            "budget_period"          : budget_period,
            # ── Media-metric allocation (model space) ──────────────────────
            "current_spend"          : round(curr_sp,                                  2),
            "optimal_spend_mean"     : round(opt_sp_mean,                              2),
            "optimal_spend_hdi_10"   : round(float(np.percentile(alloc_j, 10)),        2),
            "optimal_spend_hdi_90"   : round(float(np.percentile(alloc_j, 90)),        2),
            # ── Spend equivalents (£) via CPP ─────────────────────────────
            "current_spend_£"        : round(curr_sp_spend,                            2),
            "optimal_spend_£_mean"   : round(opt_sp_spend,                             2),
            "optimal_spend_£_hdi_10" : round(float(np.percentile(alloc_j_spend, 10)), 2),
            "optimal_spend_£_hdi_90" : round(float(np.percentile(alloc_j_spend, 90)), 2),
            "pct_change_spend"       : pct_chg_sp,
            # ── Response ──────────────────────────────────────────────────
            "current_response"       : round(curr_r,                                   4),
            "response_mean"          : round(resp_mean,                                4),
            "response_hdi_10"        : round(float(np.percentile(resp_j, 10)),         4),
            "response_hdi_90"        : round(float(np.percentile(resp_j, 90)),         4),
            "pct_change_response"    : pct_chg_resp,
            # ── ROI / CPA (spend-based) ───────────────────────────────────
            "current_roi"            : current_roi,
            "optimal_roi"            : optimal_roi,
            "current_cpa"            : current_cpa,
            "optimal_cpa"            : optimal_cpa,
        })

    df = pd.DataFrame(rows)

    alloc_total       = optimal_alloc.sum(axis=1) * n_periods
    alloc_total_spend = (optimal_alloc * cpp_w).sum(axis=1) * n_periods
    resp_total        = optimal_resp.sum(axis=1) * n_periods    # already in original units
    tot_curr_sp       = float(current_spend_pp.sum() * n_periods)
    tot_curr_sp_spend = float((current_spend_pp * cpp_w).sum() * n_periods)
    tot_curr_r        = sum(r["current_response"] for r in rows)
    tot_opt_sp        = float(alloc_total.mean())
    tot_opt_sp_spend  = float(alloc_total_spend.mean())
    tot_opt_r         = float(resp_total.mean())

    totals = {
        "channel"                  : "TOTAL",
        "metric_type"              : "mixed",
        "budget_period"            : budget_period,
        "current_spend"            : round(tot_curr_sp,       2),
        "optimal_spend_mean"       : round(tot_opt_sp,        2),
        "optimal_spend_hdi_10"     : round(float(np.percentile(alloc_total, 10)), 2),
        "optimal_spend_hdi_90"     : round(float(np.percentile(alloc_total, 90)), 2),
        "current_spend_£"          : round(tot_curr_sp_spend, 2),
        "optimal_spend_£_mean"     : round(tot_opt_sp_spend,  2),
        "optimal_spend_£_hdi_10"   : round(float(np.percentile(alloc_total_spend, 10)), 2),
        "optimal_spend_£_hdi_90"   : round(float(np.percentile(alloc_total_spend, 90)), 2),
        "pct_change_spend"         : round((tot_opt_sp_spend - tot_curr_sp_spend) / (tot_curr_sp_spend + 1e-12) * 100, 2),
        "current_response"         : round(tot_curr_r,  4),
        "response_mean"            : round(tot_opt_r,   4),
        "response_hdi_10"          : round(float(np.percentile(resp_total, 10)), 4),
        "response_hdi_90"          : round(float(np.percentile(resp_total, 90)), 4),
        "pct_change_response"      : round((tot_opt_r - tot_curr_r) / (tot_curr_r + 1e-12) * 100, 2),
        "current_roi"              : round(tot_curr_r     / (tot_curr_sp_spend + 1e-12), 6),
        "optimal_roi"              : round(tot_opt_r      / (tot_opt_sp_spend  + 1e-12), 6),
        "current_cpa"              : round(tot_curr_sp_spend / (tot_curr_r + 1e-12), 6),
        "optimal_cpa"              : round(tot_opt_sp_spend  / (tot_opt_r  + 1e-12), 6),
        # Meta columns
        "convergence_rate_pct"     : convergence_rate,
        "bounds_respected"         : bounds_respected,
        "min_achievable_response"  : round(min_ach, 4),
        "max_achievable_response"  : round(max_ach, 4),
    }
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    logger.info(
        f"  [OPTIMISE FORWARD] expected_response={totals['response_mean']:.4f} | "
        f"current_response={totals['current_response']:.4f} | "
        f"pct_change={totals['pct_change_response']:+.1f}% | "
        f"convergence={convergence_rate}%"
    )
    logger.info(
        "\n" + df[["channel", "current_spend_£", "optimal_spend_£_mean", "pct_change_spend",
                   "current_response", "response_mean", "pct_change_response",
                   "current_roi", "optimal_roi"]].to_string(index=False)
    )
    return df


# -----------------------------------------------------------------------------
# Reverse optimisation: minimise spend to achieve a target response
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Reverse optimisation: minimise spend to achieve a target response
# -----------------------------------------------------------------------------

def minimise_spend_for_target(
    best            : Dict[str, Any],
    prep            : Dict[str, Any],
    target_response : Optional[float] = None,
    target_period   : str             = "training_mean",
    n_samples       : int             = 200,
    channel_min     : Optional[Dict[str, float]] = None,
    channel_max     : Optional[Dict[str, float]] = None,
    budget_cap      : Optional[float] = None,
    cpp_weights     : Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Reverse optimisation: find the minimum total spend (and allocation) that
    achieves a given response target for the specified period.

    Parameters
    ----------
    best            : best-model dict returned by the pipeline
    prep            : data prep dict returned by the pipeline
    target_response : desired total response in ORIGINAL (unscaled) units for
                      the given period.
                      Example: 500 means 500 deals per month if target_period="monthly".
                      If None, defaults to the observed mean response for the period.
    target_period   : period the target covers (same options as budget_period in
                      optimise_budget).
    n_samples       : number of posterior draws (uncertainty estimate)
    channel_min/max : optional per-channel spend bounds in raw units for the period
    budget_cap      : optional hard cap on total spend in raw units for the period

    Returns
    -------
    DataFrame with one row per channel + a TOTAL summary row.
    All spend/response values are in raw units for the full target_period.
    """
    spend_cols, params_list = _extract_all_channel_params(best, prep, n_samples)
    params_list = _calibrate_betas_to_trace(params_list, best, prep)
    C         = len(spend_cols)
    train_idx = prep["train_idx"]
    n_train   = len(train_idx)
    frequency = prep.get("frequency", "weekly")

    X_raw = prep["X_media_raw"][train_idx]
    if X_raw.ndim == 3:
        current_spend_pp = X_raw[:, 0, :].mean(axis=0)
    else:
        current_spend_pp = X_raw.mean(axis=0)

    cpp_w           = np.asarray(cpp_weights, dtype=float) if cpp_weights is not None else np.ones(C)
    channel_rscales = _compute_channel_rscales(best, prep)

    # ── Period conversion ───────────────────────────────────────────────────
    n_periods = _periods_in_window(target_period, frequency, n_train)

    if target_response is None:
        y_raw = prep["y_raw"][train_idx]
        target_response = float(y_raw.mean()) * n_periods
        logger.info(
            f"  [OPT REVERSE] target_response not specified — using observed "
            f"{target_period} response: {target_response:,.4f}"
        )

    target_pp_original = target_response / n_periods

    logger.info(
        f"  [OPT REVERSE] period={target_period} ({n_periods:.2f} model-periods) | "
        f"target_response={target_response:,.4f} | per-period={target_pp_original:.4f}"
    )

    # ── Bounds in media-metric per-period units ──────────────────────────────
    # channel_min/max are in spend (£) when cpp_weights provided → convert to media.
    ub_default = current_spend_pp * 3.0
    lb = np.zeros(C)
    ub = ub_default.copy()

    for j, ch in enumerate(spend_cols):
        if channel_min and ch in channel_min:
            lb[j] = float(channel_min[ch]) / (n_periods * (cpp_w[j] + 1e-30))
        if channel_max and ch in channel_max:
            ub[j] = float(channel_max[ch]) / (n_periods * (cpp_w[j] + 1e-30))

    bounds = [(lb[j], ub[j]) for j in range(C)]

    # Budget cap per period: convert from spend → media-weighted sum
    cap_pp_spend = (float(budget_cap) / n_periods) if budget_cap is not None else None

    # Initial guess: current media allocation
    x0_init = current_spend_pp.copy()
    x0_init = np.clip(x0_init, lb, ub)

    # ── Achievability pre-check ──────────────────────────────────────────────
    min_ach_pp, max_ach_pp = _achievable_response_range(lb, ub, params_list, channel_rscales)
    min_ach = min_ach_pp * n_periods
    max_ach = max_ach_pp * n_periods
    target_achievable = (min_ach <= target_response <= max_ach)
    logger.info(
        f"  [ACHIEVABILITY] response achievable within bounds: "
        f"[{min_ach:,.4f}, {max_ach:,.4f}] | "
        f"target={target_response:,.4f} | achievable={target_achievable}"
    )
    if not target_achievable:
        logger.warning(
            f"  [OPTIMISE REVERSE] Target {target_response:,.4f} is outside "
            f"achievable range [{min_ach:,.4f}, {max_ach:,.4f}]. "
            "Results may not meet the target — consider relaxing channel bounds."
        )

    # ── SLSQP over posterior samples (spend space) ──────────────────────────
    # Variables: x_spend_j = GBP per channel per period.
    # Objective: minimise sum(x_spend_j)  [total spend].
    # Constraint: response(x_spend/cpp) >= target_pp_original (in original KPI units).
    S = params_list[0]["n_samples"]
    optimal_alloc = np.zeros((S, C))    # media units per period
    optimal_resp  = np.zeros((S, C))    # original KPI units per period

    lb_spend  = lb * cpp_w
    ub_spend  = ub * cpp_w
    bounds_spend = [(lb_spend[j], ub_spend[j]) for j in range(C)]
    x0_spend  = x0_init * cpp_w

    n_failed = 0
    for si in range(S):
        params_si = _params_at_sample(params_list, si)

        # Minimise total spend (trivially sum of spend variables)
        def _obj(x_sp):       return float(x_sp.sum())
        def _obj_grad(x_sp):  return np.ones_like(x_sp)

        def _con_response(x_sp, cw=cpp_w, cr=channel_rscales, tgt=target_pp_original):
            x_med = x_sp / (cw + 1e-30)
            return _total_response_raw(x_med, params_si, cr) - tgt

        local_constraints = [{"type": "ineq", "fun": _con_response}]
        if cap_pp_spend is not None:
            local_constraints.append(
                {"type": "ineq", "fun": lambda x_sp, cap=cap_pp_spend: cap - x_sp.sum()}
            )

        result = minimize(
            fun         = _obj,
            x0          = x0_spend,
            method      = "SLSQP",
            bounds      = bounds_spend,
            constraints = local_constraints,
            jac         = _obj_grad,
            options     = {"ftol": 1e-12, "maxiter": 500, "disp": False},
        )

        if result.success:
            alloc_spend = np.clip(result.x, lb_spend, ub_spend)
        else:
            alloc_spend = x0_spend.copy()
            n_failed += 1

        alloc = alloc_spend / (cpp_w + 1e-30)   # convert to media units
        optimal_alloc[si, :] = alloc
        for j, p in enumerate(params_si):
            optimal_resp[si, j] = _channel_response_raw(
                alloc[j], p["spend_max"],
                lam=p["lam"], beta=p["beta"], alpha=p["alpha"],
                kappa=p["kappa"], k_log=p["k_log"], x0=p["x0"],
                sat=p["sat_type"],
                ads_type=p.get("ads_type", "geometric"),
            ) * channel_rscales[j]

    convergence_rate = round((S - n_failed) / S * 100, 1)
    if n_failed:
        logger.warning(
            f"  [OPTIMISE REVERSE] {n_failed}/{S} SLSQP runs did not converge — "
            "current spend used as fallback for those samples."
        )

    # ── Build output (scale up to full period) ───────────────────────────────
    metric_types = prep.get("metric_types", ["Spend"] * C)

    rows = []
    for j, ch in enumerate(spend_cols):
        alloc_j = optimal_alloc[:, j] * n_periods
        resp_j  = optimal_resp[:, j] * n_periods              # already in original KPI units
        curr_sp = current_spend_pp[j] * n_periods

        # Spend equivalents
        alloc_j_spend = alloc_j * cpp_w[j]
        curr_sp_spend = curr_sp * cpp_w[j]

        curr_r      = _current_response_per_period(j, current_spend_pp, params_list, channel_rscales[j]) * n_periods
        min_sp_mean = float(alloc_j_spend.mean())
        pct_chg_sp  = round((min_sp_mean - curr_sp_spend) / (curr_sp_spend + 1e-12) * 100, 2)
        current_cpa = round(curr_sp_spend / (curr_r + 1e-12), 6)
        optimal_cpa = round(min_sp_mean   / (float(resp_j.mean()) + 1e-12), 6)

        rows.append({
            "channel"                  : ch,
            "metric_type"              : metric_types[j] if j < len(metric_types) else "Spend",
            "target_period"            : target_period,
            "current_spend"            : round(curr_sp, 2),
            "current_spend_£"          : round(curr_sp_spend, 2),
            "min_spend_mean"           : round(float(alloc_j.mean()), 2),
            "min_spend_hdi_10"         : round(float(np.percentile(alloc_j, 10)), 2),
            "min_spend_hdi_90"         : round(float(np.percentile(alloc_j, 90)), 2),
            "min_spend_£_mean"         : round(min_sp_mean, 2),
            "min_spend_£_hdi_10"       : round(float(np.percentile(alloc_j_spend, 10)), 2),
            "min_spend_£_hdi_90"       : round(float(np.percentile(alloc_j_spend, 90)), 2),
            "pct_change_spend"         : pct_chg_sp,
            "current_response"         : round(curr_r, 4),
            "achieved_response_mean"   : round(float(resp_j.mean()), 4),
            "achieved_response_hdi_10" : round(float(np.percentile(resp_j, 10)), 4),
            "achieved_response_hdi_90" : round(float(np.percentile(resp_j, 90)), 4),
            "current_cpa"              : current_cpa,
            "optimal_cpa"              : optimal_cpa,
        })

    df = pd.DataFrame(rows)

    alloc_total       = optimal_alloc.sum(axis=1) * n_periods
    alloc_total_spend = (optimal_alloc * cpp_w).sum(axis=1) * n_periods
    resp_total        = optimal_resp.sum(axis=1) * n_periods    # already in original KPI units
    tot_curr_sp       = float(current_spend_pp.sum() * n_periods)
    tot_curr_sp_spend = float((current_spend_pp * cpp_w).sum() * n_periods)
    tot_curr_r        = sum(r["current_response"] for r in rows)
    tot_min_sp        = float(alloc_total.mean())
    tot_min_sp_spend  = float(alloc_total_spend.mean())
    tot_ach_r         = float(resp_total.mean())

    totals = {
        "channel"                  : "TOTAL",
        "metric_type"              : "mixed",
        "target_period"            : target_period,
        "current_spend"            : round(tot_curr_sp,       2),
        "current_spend_£"          : round(tot_curr_sp_spend, 2),
        "min_spend_mean"           : round(tot_min_sp,        2),
        "min_spend_hdi_10"         : round(float(np.percentile(alloc_total, 10)), 2),
        "min_spend_hdi_90"         : round(float(np.percentile(alloc_total, 90)), 2),
        "min_spend_£_mean"         : round(tot_min_sp_spend,  2),
        "min_spend_£_hdi_10"       : round(float(np.percentile(alloc_total_spend, 10)), 2),
        "min_spend_£_hdi_90"       : round(float(np.percentile(alloc_total_spend, 90)), 2),
        "pct_change_spend"         : round((tot_min_sp_spend - tot_curr_sp_spend) / (tot_curr_sp_spend + 1e-12) * 100, 2),
        "current_response"         : round(tot_curr_r,  4),
        "achieved_response_mean"   : round(tot_ach_r,   4),
        "achieved_response_hdi_10" : round(float(np.percentile(resp_total, 10)), 4),
        "achieved_response_hdi_90" : round(float(np.percentile(resp_total, 90)), 4),
        "current_cpa"              : round(tot_curr_sp_spend / (tot_curr_r + 1e-12), 6),
        "optimal_cpa"              : round(tot_min_sp_spend  / (tot_ach_r  + 1e-12), 6),
        # Meta
        "target_response"          : round(target_response, 4),
        "target_achievable"        : target_achievable,
        "convergence_rate_pct"     : convergence_rate,
        "min_achievable_response"  : round(min_ach, 4),
        "max_achievable_response"  : round(max_ach, 4),
    }
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    logger.info(
        f"  [OPTIMISE REVERSE] min_spend_£={totals['min_spend_£_mean']:,.2f} | "
        f"achieved={totals['achieved_response_mean']:.4f} | "
        f"target={target_response:.4f} | "
        f"achievable={target_achievable} | convergence={convergence_rate}%"
    )
    logger.info(
        "\n" + df[["channel", "current_spend_£", "min_spend_£_mean", "pct_change_spend",
                   "achieved_response_mean", "current_cpa", "optimal_cpa"]].to_string(index=False)
    )
    return df


# -----------------------------------------------------------------------------
# Greedy marginal-ROI budget allocation
# -----------------------------------------------------------------------------

def _channel_mroi(
    x_raw    : float,
    spend_max: float,
    lam: float, beta: float, alpha: float,
    kappa: float, k_log: float, x0: float,
    sat: str,
    eps: float = 1.0,
    ads_type: str = "geometric",
) -> float:
    """Numerical marginal ROI for one channel at current spend x_raw."""
    r1 = _channel_response_raw(x_raw + eps, spend_max, lam, beta, alpha, kappa, k_log, x0, sat, ads_type)
    r0 = _channel_response_raw(x_raw,       spend_max, lam, beta, alpha, kappa, k_log, x0, sat, ads_type)
    return (r1 - r0) / (eps + 1e-30)


def save_greedy_path_to_excel(
    df_path    : pd.DataFrame,
    excel_path : str,
    sheet_name : str = "greedy_path",
) -> None:
    """
    Write (or overwrite) the greedy allocation path sheet in an Excel workbook.

    The sheet records which channel received budget at each greedy step, the
    marginal ROI at the time of selection, cumulative per-channel spend, and
    the total cumulative response — giving a step-by-step audit trail of the
    optimiser's decisions.

    Parameters
    ----------
    df_path    : DataFrame returned as the second element of greedy_budget_allocation.
    excel_path : Absolute path to the target .xlsx file.
                 If the file exists the sheet is replaced; otherwise the workbook
                 is created (parent directories are created if needed).
    sheet_name : Name for the sheet (default "greedy_path").
    """
    import os
    from pathlib import Path as _P

    excel_path = str(excel_path)
    try:
        os.makedirs(str(_P(excel_path).parent), exist_ok=True)
        if os.path.exists(excel_path):
            with pd.ExcelWriter(
                excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
            ) as writer:
                df_path.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                df_path.to_excel(writer, sheet_name=sheet_name, index=False)
        logger.info(
            f"  [GREEDY] Allocation path saved → {excel_path}  (sheet: '{sheet_name}', "
            f"{len(df_path)} steps)"
        )
    except Exception as exc:
        logger.warning(
            f"  [GREEDY] Could not save greedy path to Excel: {exc}"
        )


def greedy_budget_allocation(
    best          : Dict[str, Any],
    prep          : Dict[str, Any],
    total_budget  : Optional[float] = None,
    budget_period : str             = "training_mean",
    step_size     : Optional[float] = None,
    n_samples     : int             = 200,
    channel_min   : Optional[Dict[str, float]] = None,
    channel_max   : Optional[Dict[str, float]] = None,
    cpp_weights   : Optional[np.ndarray] = None,
    output_path   : Optional[str]  = None,
) -> tuple:
    """
    Greedy marginal-ROI budget allocator.

    Algorithm
    ---------
    At each step, compute the current mROI for every channel.
    Give the full step_size to whichever channel has the highest mROI.
    Channels that have reached their upper bound are skipped.
    Remainder routing: if the best channel cannot absorb the full step
    (constrained headroom), the leftover is routed to the next-best channel
    in the SAME step so no budget is wasted.
    Repeat until the full budget is allocated.

    Parameters
    ----------
    total_budget  : total spend in RAW units for budget_period (None = observed mean)
    budget_period : "per_period" | "monthly" | "quarterly" | "yearly" |
                    "training_total" | "training_mean"
    step_size     : increment per greedy step in RAW units for budget_period.
                    Should be small relative to total_budget (e.g. 1% or less).
    n_samples     : posterior draws used to produce uncertainty bands on the
                    final allocation.  The greedy PATH is computed once using
                    the posterior mean parameters (for interpretability).
    output_path   : optional path to an Excel file (.xlsx).  When provided,
                    df_path is written as sheet "greedy_path" in that workbook,
                    enabling step-by-step channel funding analysis.
                    If the file already exists the sheet is replaced; otherwise
                    the workbook is created.

    Returns
    -------
    df_alloc : DataFrame — one row per channel + TOTAL (same columns as
               optimise_budget, plus greedy-specific columns including
               current_response_hdi_10/90 and unallocated_budget_£).
    df_path  : DataFrame — full allocation waterfall (posterior mean params):
               [step, channel_chosen, mroi_chosen_£, cumulative_spend_{ch},
                cumulative_spend_£_{ch}, cumulative_response_total]
    """
    spend_cols, params_list = _extract_all_channel_params(best, prep, n_samples)
    params_list = _calibrate_betas_to_trace(params_list, best, prep)
    C         = len(spend_cols)
    train_idx = prep["train_idx"]
    n_train   = len(train_idx)
    frequency = prep.get("frequency", "weekly")

    X_raw = prep["X_media_raw"][train_idx]
    if X_raw.ndim == 3:
        current_spend_pp = X_raw[:, 0, :].mean(axis=0)
    else:
        current_spend_pp = X_raw.mean(axis=0)

    cpp_w           = np.asarray(cpp_weights, dtype=float) if cpp_weights is not None else np.ones(C)
    channel_rscales = _compute_channel_rscales(best, prep)
    n_periods       = _periods_in_window(budget_period, frequency, n_train)

    # ── Budget and step in SPEND per-period units ────────────────────────────
    if total_budget is None:
        total_budget = float((current_spend_pp * cpp_w).sum()) * n_periods
        logger.info(
            f"  [GREEDY] total_budget not specified — using observed "
            f"{budget_period} spend: {total_budget:,.2f}"
        )

    if step_size is None:
        step_size = max(total_budget * 0.01, 1.0)
        logger.info(f"  [GREEDY] step_size not specified — defaulting to 1% = {step_size:,.2f}")

    # budget_pp_spend: spend per model period
    budget_pp_spend = total_budget / n_periods
    step_pp_spend   = step_size    / n_periods

    n_steps = max(1, round(budget_pp_spend / step_pp_spend))
    actual_budget_pp_spend = n_steps * step_pp_spend

    logger.info(
        f"  [GREEDY] period={budget_period} ({n_periods:.2f} model-periods) | "
        f"total_spend={total_budget:,.2f} | step_size={step_size:,.2f} | "
        f"n_steps={n_steps:,}"
    )

    # ── Per-channel bounds in MEDIA-METRIC per-period units ─────────────────
    # Convert from spend to media: lb_media = lb_spend / CPP
    lb = np.zeros(C)
    ub = np.array([actual_budget_pp_spend / (cpp_w[j] + 1e-30) for j in range(C)])
    for j, ch in enumerate(spend_cols):
        if channel_min and ch in channel_min:
            lb[j] = float(channel_min[ch]) / (n_periods * (cpp_w[j] + 1e-30))
        if channel_max and ch in channel_max:
            ub[j] = float(channel_max[ch]) / (n_periods * (cpp_w[j] + 1e-30))

    # ── Posterior mean parameters (for the allocation path) ─────────────────
    mean_params = [
        {
            "sat_type" : p["sat_type"],
            "ads_type" : p.get("ads_type", "geometric"),
            "spend_max": p["spend_max"],
            "lam"      : float(p["lam"].mean()),
            "beta"     : float(p["beta"].mean()),
            "alpha"    : float(p["alpha"].mean()),
            "kappa"    : float(p["kappa"].mean()),
            "k_log"    : float(p["k_log"].mean()),
            "x0"       : float(p["x0"].mean()),
        }
        for p in params_list
    ]

    # mROI epsilon in media units — scale by median CPP to keep it sensible
    median_cpp = float(np.median(cpp_w))
    mroi_eps_spend = max(step_pp_spend * 1e-4, 1e-6)

    def _mroi_vec_spend(alloc: np.ndarray, params, cw: np.ndarray) -> np.ndarray:
        """mROI in SPEND units: d(response)/d(spend) = d(response)/d(media) / CPP."""
        mroi_media = np.array([
            _channel_mroi(
                alloc[j], params[j]["spend_max"],
                params[j]["lam"], params[j]["beta"], params[j]["alpha"],
                params[j]["kappa"], params[j]["k_log"], params[j]["x0"],
                params[j]["sat_type"],
                eps=max(mroi_eps_spend / (cw[j] + 1e-30), 1e-9),
                ads_type=params[j].get("ads_type", "geometric"),
            )
            for j in range(C)
        ])
        return mroi_media / (cw + 1e-30)

    # ═════════════════════════════════════════════════════════════════════════
    # PATH computation (posterior mean — single run, full trace)
    # ═════════════════════════════════════════════════════════════════════════

    def _do_greedy_step(alloc: np.ndarray, step_budget: float, params, cw: np.ndarray):
        """
        Allocate one spend step (step_budget GBP) across channels by mROI priority.
        When a channel hits its upper bound before the full step is used,
        the remainder is routed to the next-best channel in the SAME step.

        Returns (alloc_updated, first_chosen_j, first_mroi).
        """
        remaining  = step_budget
        first_j    = None
        first_mroi = None
        while remaining > step_budget * 1e-9:
            mroi_sp = _mroi_vec_spend(alloc, params, cw)
            # Mask channels at their upper bound
            for j in range(C):
                if (ub[j] - alloc[j]) * cw[j] < step_budget * 1e-9:
                    mroi_sp[j] = -np.inf
            if np.all(mroi_sp == -np.inf):
                break   # all channels saturated — remaining budget unspent
            bj = int(np.argmax(mroi_sp))
            if first_j is None:
                first_j    = bj
                first_mroi = float(mroi_sp[bj])
            headroom_sp = (ub[bj] - alloc[bj]) * cw[bj]
            give_sp     = min(remaining, headroom_sp)
            alloc[bj]  += give_sp / (cw[bj] + 1e-30)
            remaining  -= give_sp
        return alloc, (first_j if first_j is not None else 0), first_mroi

    # Compute the mandatory minimum spend (lower bounds) per period.
    # Greedy gets the REMAINING budget after these are pre-allocated.
    lb_spend_pp = lb * cpp_w   # (C,) minimum spend in GBP/period

    mandatory_pp = lb_spend_pp.sum()
    greedy_budget_pp = budget_pp_spend - mandatory_pp
    if greedy_budget_pp < 0:
        logger.warning(
            f"  [GREEDY] Channel minimums ({mandatory_pp:,.2f}) exceed "
            f"total budget ({budget_pp_spend:,.2f}). "
            "Clamping to minimum allocation only."
        )
        greedy_budget_pp = 0.0

    # Recompute step count from the greedy portion only
    if greedy_budget_pp > 0 and step_pp_spend > 0:
        n_steps = max(1, round(greedy_budget_pp / step_pp_spend))
    else:
        n_steps = 0

    # Start from lower bounds + tiny epsilon (so mROI is always defined)
    alloc_path = lb.copy() + np.array([
        max(0.0, ub[j] - lb[j]) * 1e-9 for j in range(C)
    ])

    path_rows = []

    # Record the zero-cost starting state
    cum_resp = sum(
        _channel_response_raw(
            alloc_path[j], mean_params[j]["spend_max"],
            mean_params[j]["lam"], mean_params[j]["beta"], mean_params[j]["alpha"],
            mean_params[j]["kappa"], mean_params[j]["k_log"], mean_params[j]["x0"],
            mean_params[j]["sat_type"],
            ads_type=mean_params[j].get("ads_type", "geometric"),
        ) * channel_rscales[j]
        for j in range(C)
    )

    row = {"step": 0, "channel_chosen": "(start)", "mroi_chosen_£": None}
    for j, ch in enumerate(spend_cols):
        row[f"spend_{ch}"]   = round(alloc_path[j], 4)
        row[f"spend_£_{ch}"] = round(alloc_path[j] * cpp_w[j], 4)
    row["cumulative_response"] = round(cum_resp, 6)
    path_rows.append(row)

    # Greedy steps for the remaining budget (after mandatory lb minimums).
    # Each step allocates exactly step_pp_spend GBP (with remainder routing when a channel
    # hits its upper bound before the full step is consumed).
    for step in range(1, n_steps + 1):
        alloc_path, best_j, mroi_val = _do_greedy_step(
            alloc_path, step_pp_spend, mean_params, cpp_w
        )

        cum_resp = sum(
            _channel_response_raw(
                alloc_path[j], mean_params[j]["spend_max"],
                mean_params[j]["lam"], mean_params[j]["beta"], mean_params[j]["alpha"],
                mean_params[j]["kappa"], mean_params[j]["k_log"], mean_params[j]["x0"],
                mean_params[j]["sat_type"],
                ads_type=mean_params[j].get("ads_type", "geometric"),
            ) * channel_rscales[j]
            for j in range(C)
        )

        row = {
            "step"          : step,
            "channel_chosen": spend_cols[best_j],
            "mroi_chosen_£" : round(mroi_val, 6) if mroi_val is not None else None,
        }
        for j, ch in enumerate(spend_cols):
            row[f"spend_{ch}"]   = round(alloc_path[j], 4)
            row[f"spend_£_{ch}"] = round(alloc_path[j] * cpp_w[j], 4)
        row["cumulative_response"] = round(cum_resp, 6)
        path_rows.append(row)

    df_path = pd.DataFrame(path_rows)

    # Final path allocation (per period, mean params) → scale to full period
    final_path_alloc = alloc_path.copy()

    # ═════════════════════════════════════════════════════════════════════════
    # POSTERIOR UNCERTAINTY: run greedy over S samples
    # ═════════════════════════════════════════════════════════════════════════
    S = params_list[0]["n_samples"]
    sample_alloc = np.zeros((S, C))
    sample_resp  = np.zeros((S, C))

    for si in range(S):
        params_si = _params_at_sample(params_list, si)
        alloc_si = lb.copy() + np.array([max(0.0, ub[j] - lb[j]) * 1e-9 for j in range(C)])

        for _ in range(n_steps):
            alloc_si, _, _ = _do_greedy_step(alloc_si, step_pp_spend, params_si, cpp_w)

        sample_alloc[si, :] = alloc_si
        for j, p in enumerate(params_si):
            sample_resp[si, j] = _channel_response_raw(
                alloc_si[j], p["spend_max"],
                lam=p["lam"], beta=p["beta"], alpha=p["alpha"],
                kappa=p["kappa"], k_log=p["k_log"], x0=p["x0"],
                sat=p["sat_type"],
                ads_type=p.get("ads_type", "geometric"),
            ) * channel_rscales[j]     # per-channel rscale → original KPI units

    # ═════════════════════════════════════════════════════════════════════════
    # Current-response uncertainty: propagate posterior through current spend
    # ─────────────────────────────────────────────────────────────────────────
    # Run the same S posterior draws used for the greedy allocation so the
    # comparison is apples-to-apples and we can report current_response_hdi_10/90
    # alongside the greedy HDI bands.  Per-channel rscale applied for correct units.
    # ═════════════════════════════════════════════════════════════════════════
    curr_resp_samples = np.zeros((S, C))   # (S, C) — response at current spend
    for si in range(S):
        params_si = _params_at_sample(params_list, si)
        for j, p in enumerate(params_si):
            curr_resp_samples[si, j] = _channel_response_raw(
                current_spend_pp[j], p["spend_max"],
                lam=p["lam"], beta=p["beta"], alpha=p["alpha"],
                kappa=p["kappa"], k_log=p["k_log"], x0=p["x0"],
                sat=p["sat_type"],
                ads_type=p.get("ads_type", "geometric"),
            ) * channel_rscales[j] * n_periods    # per-channel rscale → original KPI units

    # ═════════════════════════════════════════════════════════════════════════
    # Build output DataFrame
    # ═════════════════════════════════════════════════════════════════════════
    metric_types = prep.get("metric_types", ["Spend"] * C)

    rows = []
    for j, ch in enumerate(spend_cols):
        alloc_j       = sample_alloc[:, j] * n_periods        # media units
        alloc_j_spend = alloc_j * cpp_w[j]                    # spend £
        resp_j        = sample_resp[:, j] * n_periods          # already in original KPI units
        curr_sp       = current_spend_pp[j] * n_periods
        curr_sp_spend = curr_sp * cpp_w[j]
        # Use sample-based current response (matches greedy posterior draws)
        curr_r_samples = curr_resp_samples[:, j]               # shape (S,)
        curr_r         = float(curr_r_samples.mean())

        opt_sp_mean   = float(alloc_j_spend.mean())
        resp_mean     = float(resp_j.mean())
        pct_chg_sp    = round((opt_sp_mean - curr_sp_spend) / (curr_sp_spend + 1e-12) * 100, 2)
        pct_chg_resp  = round((resp_mean   - curr_r)        / (curr_r   + 1e-12) * 100, 2)

        rows.append({
            "channel"                   : ch,
            "metric_type"               : metric_types[j] if j < len(metric_types) else "Spend",
            "budget_period"             : budget_period,
            "step_size"                 : round(step_size, 4),
            "n_steps"                   : n_steps,
            "current_spend"             : round(curr_sp, 2),
            "current_spend_£"           : round(curr_sp_spend, 2),
            "greedy_spend_mean"         : round(float(alloc_j.mean()), 2),
            "greedy_spend_hdi_10"       : round(float(np.percentile(alloc_j, 10)), 2),
            "greedy_spend_hdi_90"       : round(float(np.percentile(alloc_j, 90)), 2),
            "greedy_spend_£_mean"       : round(opt_sp_mean, 2),
            "greedy_spend_£_hdi_10"     : round(float(np.percentile(alloc_j_spend, 10)), 2),
            "greedy_spend_£_hdi_90"     : round(float(np.percentile(alloc_j_spend, 90)), 2),
            "pct_change_spend"          : pct_chg_sp,
            # ── Current response (now with posterior uncertainty) ─────────────
            "current_response"          : round(curr_r, 4),
            "current_response_hdi_10"   : round(float(np.percentile(curr_r_samples, 10)), 4),
            "current_response_hdi_90"   : round(float(np.percentile(curr_r_samples, 90)), 4),
            # ── Greedy response ───────────────────────────────────────────────
            "greedy_response_mean"      : round(resp_mean, 4),
            "greedy_response_hdi_10"    : round(float(np.percentile(resp_j, 10)), 4),
            "greedy_response_hdi_90"    : round(float(np.percentile(resp_j, 90)), 4),
            "pct_change_response"       : pct_chg_resp,
            "current_roi"               : round(curr_r / (curr_sp_spend + 1e-12), 6),
            "greedy_roi"                : round(resp_mean / (opt_sp_mean + 1e-12), 6),
            "current_cpa"               : round(curr_sp_spend / (curr_r    + 1e-12), 6),
            "greedy_cpa"                : round(opt_sp_mean   / (resp_mean + 1e-12), 6),
        })

    df_alloc = pd.DataFrame(rows)

    alloc_total          = sample_alloc.sum(axis=1) * n_periods
    alloc_total_spend    = (sample_alloc * cpp_w).sum(axis=1) * n_periods
    resp_total           = sample_resp.sum(axis=1) * n_periods          # already in original KPI units
    curr_resp_total_samp = curr_resp_samples.sum(axis=1)       # (S,) total current resp
    tot_curr_sp          = float(current_spend_pp.sum() * n_periods)
    tot_curr_sp_spend    = float((current_spend_pp * cpp_w).sum() * n_periods)
    tot_curr_r           = float(curr_resp_total_samp.mean())
    tot_opt_sp           = float(alloc_total.mean())
    tot_opt_sp_spend     = float(alloc_total_spend.mean())
    tot_opt_r            = float(resp_total.mean())

    totals = {
        "channel"                   : "TOTAL",
        "metric_type"               : "mixed",
        "budget_period"             : budget_period,
        "step_size"                 : round(step_size, 4),
        "n_steps"                   : n_steps,
        "current_spend"             : round(tot_curr_sp, 2),
        "current_spend_£"           : round(tot_curr_sp_spend, 2),
        "greedy_spend_mean"         : round(tot_opt_sp,  2),
        "greedy_spend_hdi_10"       : round(float(np.percentile(alloc_total, 10)), 2),
        "greedy_spend_hdi_90"       : round(float(np.percentile(alloc_total, 90)), 2),
        "greedy_spend_£_mean"       : round(tot_opt_sp_spend,  2),
        "greedy_spend_£_hdi_10"     : round(float(np.percentile(alloc_total_spend, 10)), 2),
        "greedy_spend_£_hdi_90"     : round(float(np.percentile(alloc_total_spend, 90)), 2),
        "pct_change_spend"          : round((tot_opt_sp_spend - tot_curr_sp_spend) / (tot_curr_sp_spend + 1e-12) * 100, 2),
        # ── Current response (posterior uncertainty) ──────────────────────────
        "current_response"          : round(tot_curr_r,  4),
        "current_response_hdi_10"   : round(float(np.percentile(curr_resp_total_samp, 10)), 4),
        "current_response_hdi_90"   : round(float(np.percentile(curr_resp_total_samp, 90)), 4),
        # ── Greedy response ───────────────────────────────────────────────────
        "greedy_response_mean"      : round(tot_opt_r,   4),
        "greedy_response_hdi_10"    : round(float(np.percentile(resp_total, 10)), 4),
        "greedy_response_hdi_90"    : round(float(np.percentile(resp_total, 90)), 4),
        "pct_change_response"       : round((tot_opt_r - tot_curr_r) / (tot_curr_r + 1e-12) * 100, 2),
        "current_roi"               : round(tot_curr_r    / (tot_curr_sp_spend + 1e-12), 6),
        "greedy_roi"                : round(tot_opt_r     / (tot_opt_sp_spend  + 1e-12), 6),
        "current_cpa"               : round(tot_curr_sp_spend / (tot_curr_r + 1e-12), 6),
        "greedy_cpa"                : round(tot_opt_sp_spend  / (tot_opt_r  + 1e-12), 6),
        # ── Budget accounting ─────────────────────────────────────────────────
        "unallocated_budget_£"      : round(total_budget - tot_opt_sp_spend, 2),
        "allocation_rate_pct"       : round(tot_opt_sp_spend / (total_budget + 1e-12) * 100, 2),
    }
    df_alloc = pd.concat([df_alloc, pd.DataFrame([totals])], ignore_index=True)

    # Scale path spend columns to full period for export
    for j, ch in enumerate(spend_cols):
        for prefix in (f"spend_{ch}", f"spend_£_{ch}"):
            if prefix in df_path.columns:
                df_path[prefix] = df_path[prefix] * n_periods

    logger.info(
        f"  [GREEDY] greedy_spend_£={totals['greedy_spend_£_mean']:,.2f} | "
        f"greedy_response={totals['greedy_response_mean']:.4f} | "
        f"current_response={totals['current_response']:.4f} | "
        f"pct_change={totals['pct_change_response']:+.1f}%"
    )
    logger.info(
        "\n" + df_alloc[["channel", "current_spend_£", "greedy_spend_£_mean",
                          "pct_change_spend", "current_response", "greedy_response_mean",
                          "pct_change_response", "greedy_roi"]].to_string(index=False)
    )

    if output_path is not None:
        save_greedy_path_to_excel(df_path, output_path)

    return df_alloc, df_path


# =============================================================================
# Sequential multi-period optimiser
# =============================================================================
# Implements month-by-month carry-forward with composite mROI scoring.
# Mirrors the notebook's optimize_budget_stepwise_mroi / run_sequential_period
# logic, adapted to use Bayesian posterior parameters.
#
# Key differences from greedy_budget_allocation():
#   1. Runs for n_months — carry propagates forward each month.
#   2. Incremental response = beta*(sat(new+carry) - sat(carry)) rather than
#      steady-state: isolates the NEW signal contribution at current adstock level.
#   3. Composite mROI scoring:
#          score = (mROI/max_mROI) × quality × slope_weight
#      where quality penalises channels with poor CPA, and slope_weight penalises
#      channels where marginal value is declining fast.
#   4. Simulated BAU baseline: same budget, current channel mix → fair comparison.
# =============================================================================


def _monthly_response_sim(
    alloc_media_pp  : np.ndarray,
    n_ppm           : int,
    carry_in        : np.ndarray,
    params          : List[Dict],
    channel_rscales : np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Simulate one month and return (incremental_response_above_carry, carry_out).

    Uses beta*(s_tot - s_car) × channel_rscale_j — the response attributable
    to NEW spend above the existing carry baseline, in original KPI units.
    Used internally for mROI comparisons in the greedy step.
    """
    C     = len(params)
    carry = carry_in.copy()
    total = 0.0

    for _t in range(n_ppm):
        for j, p in enumerate(params):
            x_sc      = alloc_media_pp[j] / (p["spend_max"] + 1e-8)
            total_sig = x_sc + carry[j]

            s_tot = float(_apply_saturation(
                np.array([total_sig]), p["sat_type"],
                np.array([p["alpha"]]), np.array([p["kappa"]]),
                np.array([p["k_log"]]), np.array([p["x0"]]),
            )[0])
            s_car = float(_apply_saturation(
                np.array([carry[j]]), p["sat_type"],
                np.array([p["alpha"]]), np.array([p["kappa"]]),
                np.array([p["k_log"]]), np.array([p["x0"]]),
            )[0])

            total += p["beta"] * (s_tot - s_car) * float(channel_rscales[j])
            if p.get("ads_type", "geometric") == "weibull":
                carry[j] = 0.0
            else:
                carry[j] = float(np.clip(p["lam"], 0.0, 0.9999)) * total_sig

    return total, carry


def _monthly_response_total(
    alloc_media_pp  : np.ndarray,
    n_ppm           : int,
    carry_in        : np.ndarray,
    params          : List[Dict],
    channel_rscales : np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Simulate one month and return (total_channel_response, carry_out).

    Uses beta*s_tot × channel_rscale_j — the full channel response including
    carry, in original KPI units.  Matches the contribution analysis.
    Use this for all opt/bau/curr figures shown to users.
    """
    C     = len(params)
    carry = carry_in.copy()
    total = 0.0

    for _t in range(n_ppm):
        for j, p in enumerate(params):
            x_sc      = alloc_media_pp[j] / (p["spend_max"] + 1e-8)
            total_sig = x_sc + carry[j]

            s_tot = float(_apply_saturation(
                np.array([total_sig]), p["sat_type"],
                np.array([p["alpha"]]), np.array([p["kappa"]]),
                np.array([p["k_log"]]), np.array([p["x0"]]),
            )[0])

            total += p["beta"] * s_tot * float(channel_rscales[j])
            if p.get("ads_type", "geometric") == "weibull":
                carry[j] = 0.0
            else:
                carry[j] = float(np.clip(p["lam"], 0.0, 0.9999)) * total_sig

    return total, carry


def _channel_j_monthly_response(
    j                : int,
    alloc_j_pp       : float,
    n_ppm            : int,
    carry_j_in       : float,
    param_j          : Dict,
    channel_rscale_j : float,
) -> float:
    """
    Compute channel j's contribution alone over n_ppm periods, in original KPI units.

    Unlike calling _monthly_response_total with all-zero spend for the other
    channels, this function only accumulates beta_j * sat(signal_j) — it never
    touches the other channels, so softplus(0) from inactive channels cannot
    inflate the result.
    """
    p       = param_j
    carry_j = carry_j_in
    total   = 0.0
    for _t in range(n_ppm):
        x_sc      = alloc_j_pp / (p["spend_max"] + 1e-8)
        total_sig = x_sc + carry_j
        s_tot = float(_apply_saturation(
            np.array([total_sig]), p["sat_type"],
            np.array([p["alpha"]]), np.array([p["kappa"]]),
            np.array([p["k_log"]]), np.array([p["x0"]]),
        )[0])
        total += p["beta"] * s_tot
        if p.get("ads_type", "geometric") == "weibull":
            carry_j = 0.0
        else:
            carry_j = float(np.clip(p["lam"], 0.0, 0.9999)) * total_sig
    return total * channel_rscale_j


def _simulated_bau_for_month(
    budget_pp_spend   : float,
    current_spend_pp  : np.ndarray,
    cpp_w             : np.ndarray,
) -> np.ndarray:
    """
    Rescale the current observed channel mix to match the target monthly budget.
    Returns (C,) media units per model period with the same proportional mix.

    This is the "simulated BAU" baseline: same allocation structure as observed,
    but at the budget level being evaluated.
    """
    current_pp_spend = current_spend_pp * cpp_w   # (C,) £ per model period
    total_pp_spend   = current_pp_spend.sum()
    if total_pp_spend < 1e-12:
        return np.zeros_like(current_spend_pp)
    # Scale uniformly so sum(alloc_pp × cpp_w) == budget_pp_spend
    scale = budget_pp_spend / total_pp_spend
    return current_spend_pp * scale   # media per period, same mix


def _greedy_step_sequential(
    alloc, step_pp_spend, n_ppm, carry_in,
    lb, ub, params, cpp_w, channel_rscales, mroi_floor, slope_scaling,
):
    """
    One greedy step: allocate step_pp_spend to the channel with the highest
    composite mROI score.

    score = (mROI/max_mROI) * quality * slope_weight
      quality      = mROI / (mROI + mroi_floor)
      slope_weight = 1 / (1 + |d2R/dS2 * slope_scaling|)

    Returns (updated_alloc, chosen_channel_index, chosen_score).
    """
    C   = len(params)
    eps = [step_pp_spend * 1e-3 / (cpp_w[j] + 1e-30) for j in range(C)]

    # Baseline response (computed once, reused)
    r_base, _ = _monthly_response_sim(alloc, n_ppm, carry_in, params, channel_rscales)

    mroi  = np.full(C, -np.inf)
    r_fwd = [r_base] * C   # forward-perturbed response per channel

    for j in range(C):
        if (ub[j] - alloc[j]) * cpp_w[j] < step_pp_spend * 1e-6:
            continue
        a_plus    = alloc.copy()
        a_plus[j] = min(alloc[j] + eps[j], ub[j])
        actual_e  = a_plus[j] - alloc[j]
        if actual_e < 1e-15:
            continue
        r_p, _    = _monthly_response_sim(a_plus, n_ppm, carry_in, params, channel_rscales)
        eps_spend = actual_e * cpp_w[j] * n_ppm
        mroi[j]   = (r_p - r_base) / (eps_spend + 1e-30)
        r_fwd[j]  = r_p

    if np.all(~np.isfinite(mroi)):
        return alloc, 0, 0.0

    max_mroi = float(np.where(np.isfinite(mroi), mroi, 0.0).max()) + 1e-30
    scores   = np.full(C, -np.inf)

    for j in range(C):
        if not np.isfinite(mroi[j]):
            continue
        m = max(mroi[j], 0.0)
        # Quality weight (CPA penalty)
        quality = m / (m + mroi_floor + 1e-30)
        # Slope weight: second-order FD using cached r_fwd[j] and backward perturb
        a_minus    = alloc.copy()
        a_minus[j] = max(alloc[j] - eps[j], lb[j])
        r_m, _     = _monthly_response_sim(a_minus, n_ppm, carry_in, params, channel_rscales)
        eps_spend  = eps[j] * cpp_w[j] * n_ppm
        slope      = (r_fwd[j] + r_m - 2.0 * r_base) / (eps_spend ** 2 + 1e-60)
        slope_w    = 1.0 / (1.0 + abs(slope * slope_scaling))
        scores[j]  = (m / max_mroi) * quality * slope_w

    best_j      = int(np.argmax(scores))
    first_best_j = best_j
    first_score  = float(scores[best_j])

    # Remainder routing: if the best channel can only absorb part of the step
    # (constrained headroom), route the leftover to the next-best channel in
    # the same step rather than losing it.
    alloc_new = alloc.copy()
    remaining = step_pp_spend

    while remaining > step_pp_spend * 1e-9:
        # Recompute scores with current alloc_new to pick the next best
        # (on the first iteration we already have scores; on subsequent we recompute)
        candidate_scores = scores.copy() if remaining == step_pp_spend else np.full(C, -np.inf)
        if remaining < step_pp_spend:
            # Recompute mROI for channels that still have headroom
            r_now, _ = _monthly_response_sim(alloc_new, n_ppm, carry_in, params, channel_rscales)
            for j in range(C):
                headroom_j = (ub[j] - alloc_new[j]) * cpp_w[j]
                if headroom_j < remaining * 1e-6:
                    candidate_scores[j] = -np.inf
                    continue
                ep = step_pp_spend * 1e-3 / (cpp_w[j] + 1e-30)
                a_p    = alloc_new.copy(); a_p[j] = min(alloc_new[j] + ep, ub[j])
                ae     = a_p[j] - alloc_new[j]
                if ae < 1e-15:
                    candidate_scores[j] = -np.inf; continue
                r_p, _ = _monthly_response_sim(a_p, n_ppm, carry_in, params, channel_rscales)
                m_j    = (r_p - r_now) / (ae * cpp_w[j] * n_ppm + 1e-30)
                candidate_scores[j] = max(m_j, 0.0)

        if np.all(~np.isfinite(candidate_scores) | (candidate_scores == -np.inf)):
            break  # all channels at their upper bound — remaining budget unspent

        pick_j    = int(np.argmax(candidate_scores))
        headroom  = (ub[pick_j] - alloc_new[pick_j]) * cpp_w[pick_j]
        give      = min(remaining, headroom)
        if give < remaining * 1e-9:
            break  # no channel can absorb more
        alloc_new[pick_j] += give / (cpp_w[pick_j] + 1e-30)
        remaining          -= give

    return alloc_new, first_best_j, first_score


def optimise_budget_sequential(
    best               : Dict[str, Any],
    prep               : Dict[str, Any],
    total_budget       : Optional[float] = None,
    n_months           : int             = 1,
    n_samples          : int             = 1,
    channel_min        : Optional[Dict[str, float]] = None,
    channel_max        : Optional[Dict[str, float]] = None,
    channel_share_min  : Optional[Dict[str, float]] = None,
    channel_share_max  : Optional[Dict[str, float]] = None,
    cpp_weights        : Optional[np.ndarray] = None,
    target_cpa         : Optional[float] = None,
    step_size          : Optional[float] = None,
    round_budget_pct   : Optional[float] = None,
    slope_scaling      : float           = 1e4,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sequential multi-period budget optimiser with carry-forward.

    Runs month-by-month greedy allocation with composite mROI scoring,
    carrying adstock forward between months. Outputs include:
      - Per-channel summary (optimised vs simulated BAU vs current observed)
      - Month-by-month response breakdown

    Algorithm
    ---------
    For each month m = 1 … n_months:
      1.  Greedy allocation:
            Start from lower bounds.
            At each step, compute composite mROI score for every channel:
                score = (mROI / max_mROI) × quality × slope_weight
            Allocate the step to the highest-scoring channel.
      2.  Simulate month:  carry_in → alloc → response + carry_out.
      3.  BAU simulation:  rescale current mix to same monthly budget, simulate.

    Parameters
    ----------
    best               : best-model dict
    prep               : data prep dict
    total_budget       : total spend in £ for the whole period (None = observed mean)
    n_months           : number of months to optimise over
    n_samples          : posterior draws for carry-forward (1 = posterior mean only)
    channel_min        : {channel: £ absolute lower bound} for the whole period
    channel_max        : {channel: £ absolute upper bound} for the whole period
    channel_share_min  : {channel: fraction} minimum share of monthly budget (0–1)
    channel_share_max  : {channel: fraction} maximum share of monthly budget (0–1)
    cpp_weights        : (C,) CPP array; if None, treats all media as £ spend directly
    target_cpa         : target cost-per-acquisition for quality weight (£/unit response)
    step_size          : greedy step in £ (per month; None = 1% of monthly budget)
    round_budget_pct   : if set, each greedy step size = round_budget_pct × remaining
                         budget. Produces smoother allocation than a fixed step_size.
                         Typical values: 0.01–0.05. Overrides step_size when set.
    slope_scaling      : scaling factor for slope penalty in composite score

    Returns
    -------
    df_summary  : DataFrame — overall opt vs bau vs current (1 row)
    df_channels : DataFrame — per-channel breakdown
    df_monthly  : DataFrame — month-by-month opt and bau response
    """
    spend_cols, params_list = _extract_all_channel_params(best, prep, n_samples)
    params_list = _calibrate_betas_to_trace(params_list, best, prep)
    C         = len(spend_cols)
    train_idx = prep["train_idx"]
    n_train   = len(train_idx)
    frequency = prep.get("frequency", "weekly")

    if n_samples < 10:
        logger.warning(
            f"  [SEQ-OPT] n_samples={n_samples} is too low for reliable HDI bands. "
            "Increase to ≥50 for meaningful uncertainty quantification. "
            "The allocation PATH uses posterior-mean parameters regardless, but "
            "HDI columns in df_channels will reflect near-zero variance."
        )

    X_raw = prep["X_media_raw"][train_idx]
    if X_raw.ndim == 3:
        current_spend_pp = X_raw[:, 0, :].mean(axis=0)
    else:
        current_spend_pp = X_raw.mean(axis=0)

    cpp_w           = np.asarray(cpp_weights, dtype=float) if cpp_weights is not None else np.ones(C)
    channel_rscales = _compute_channel_rscales(best, prep)

    ppm   = _PERIODS_PER_MONTH.get(frequency, _PERIODS_PER_MONTH["weekly"])
    n_ppm = max(1, round(ppm))   # integer periods per month for simulation

    # ── Budget ───────────────────────────────────────────────────────────────
    if total_budget is None:
        total_budget = float((current_spend_pp * cpp_w).sum()) * n_ppm * n_months
        logger.info(
            f"  [SEQ-OPT] total_budget not specified — using observed "
            f"monthly spend × {n_months}: £{total_budget:,.2f}"
        )

    monthly_budget      = total_budget / n_months
    monthly_budget_pp   = monthly_budget / n_ppm     # £ per model period per month

    # round_budget_pct overrides step_size — step is dynamic (% of remaining budget)
    use_dynamic_step = round_budget_pct is not None and round_budget_pct > 0
    if not use_dynamic_step:
        if step_size is None:
            step_size = max(monthly_budget * 0.01, 1.0)
        step_pp_spend = step_size / n_ppm           # step in £ per model period
    else:
        step_size = None                            # not used in dynamic mode
        step_pp_spend = max(monthly_budget * round_budget_pct, 1.0) / n_ppm  # initial est.

    logger.info(
        f"  [SEQ-OPT] n_months={n_months}  n_ppm={n_ppm}  "
        f"total=£{total_budget:,.2f}  monthly=£{monthly_budget:,.2f}  "
        + (f"round_budget_pct={round_budget_pct:.3f}" if use_dynamic_step
           else f"step=£{step_size:,.2f}")
    )

    # ── Posterior mean parameters (allocation path) ───────────────────────────
    # Always build mean_params from the full posterior (400 draws) regardless of
    # the n_samples config.  Using n_samples=1 (the default) takes only the first
    # MCMC draw; if that draw has an extreme alpha for any channel, Hill(x; alpha)
    # is near-zero at current spend and the per-channel current_response shows 0.
    # The 400-sample mean is stable, matches the contribution analysis, and is
    # cheap (no sampling — just averaging existing posterior arrays).
    _, _params_for_mean = _extract_all_channel_params(best, prep, n_samples=400)
    mean_params = [
        {
            "sat_type" : p["sat_type"],
            "ads_type" : p.get("ads_type", "geometric"),
            "spend_max": p["spend_max"],
            "lam"      : float(p["lam"].mean()),
            "beta"     : float(p["beta"].mean()),
            "alpha"    : float(p["alpha"].mean()),
            "kappa"    : float(p["kappa"].mean()),
            "k_log"    : float(p["k_log"].mean()),
            "x0"       : float(p["x0"].mean()),
        }
        for p in _params_for_mean
    ]

    # ── Bounds in media per-period ────────────────────────────────────────────
    lb = np.zeros(C)
    ub = np.array([monthly_budget_pp / (cpp_w[j] + 1e-30) for j in range(C)])
    for j, ch in enumerate(spend_cols):
        # Absolute £ bounds (whole-period £ → per-period media units)
        if channel_min and ch in channel_min:
            lb[j] = float(channel_min[ch]) / (n_months * n_ppm * (cpp_w[j] + 1e-30))
        if channel_max and ch in channel_max:
            ub[j] = float(channel_max[ch]) / (n_months * n_ppm * (cpp_w[j] + 1e-30))
        # Share bounds: fraction of monthly budget → per-period media units
        if channel_share_min and ch in channel_share_min:
            share_lb = float(channel_share_min[ch]) * monthly_budget_pp / (cpp_w[j] + 1e-30)
            lb[j] = max(lb[j], share_lb)
        if channel_share_max and ch in channel_share_max:
            share_ub = float(channel_share_max[ch]) * monthly_budget_pp / (cpp_w[j] + 1e-30)
            ub[j] = min(ub[j], share_ub)
        # Guard: lb must not exceed ub
        if lb[j] > ub[j]:
            logger.warning(
                f"  [SEQ-OPT] channel '{ch}': lb ({lb[j]:.4f}) > ub ({ub[j]:.4f}) "
                f"after share bounds — clamping lb to ub."
            )
            lb[j] = ub[j]

    # ── CPA quality floor ─────────────────────────────────────────────────────
    mroi_floor = (1.0 / target_cpa) if target_cpa and target_cpa > 0 else 0.0

    # ── Warm-start carry: steady-state at current observed spend ─────────────
    # Campaigns have been running for months; starting from zero carry
    # massively understates month-1 response for channels with high lam (≥0.7).
    # Steady-state carry for geometric adstock: carry_ss = lam × x_sc / (1 - lam)
    # For weibull adstock, carry is always 0 (non-recursive convolution).
    # Use the full-posterior mean lam (from _params_for_mean) so ss_carry is
    # the true steady-state carry, not the first-draw carry.
    mean_lam_arr = np.array([float(p["lam"].mean()) for p in _params_for_mean])
    x_sc_curr    = current_spend_pp / np.array(
        [p["spend_max"] + 1e-8 for p in _params_for_mean]
    )
    ss_carry = mean_lam_arr * x_sc_curr / np.maximum(1.0 - mean_lam_arr, 1e-8)
    for j in range(C):
        if mean_params[j].get("ads_type", "geometric") == "weibull":
            ss_carry[j] = 0.0

    # ── Current observed response (for comparison) ───────────────────────────
    # Use total response (not incremental above carry) so reported numbers
    # match the contribution analysis. Warm carry ensures steady-state.
    current_resp_monthly, _ = _monthly_response_total(
        current_spend_pp, n_ppm, ss_carry.copy(), mean_params, channel_rscales
    )

    # ── Sequential monthly loop ───────────────────────────────────────────────
    opt_carry = ss_carry.copy()
    bau_carry = ss_carry.copy()

    monthly_rows = []
    opt_alloc_total = np.zeros(C)   # cumulative media per period (avg over months)
    bau_alloc_total = np.zeros(C)

    n_steps_per_month = max(1, round((monthly_budget_pp - lb.dot(cpp_w)) / step_pp_spend))

    for m in range(1, n_months + 1):

        # ── Greedy allocation for this month ─────────────────────────────────
        alloc = lb.copy() + np.array([max(0.0, ub[j] - lb[j]) * 1e-9 for j in range(C)])
        mandatory_spend = (lb * cpp_w).sum()
        remaining_spend = monthly_budget_pp - mandatory_spend

        if use_dynamic_step:
            # Dynamic step: each step = round_budget_pct × remaining unallocated spend.
            # The loop must be guarded because channel caps/share bounds can leave
            # budget that no channel can absorb.  Without a no-progress break,
            # reverse sequential runs can appear to hang inside this inner loop.
            _min_step = max(monthly_budget_pp * 0.001, 1.0 / n_ppm)
            _rem = max(0.0, remaining_spend)
            _no_progress_tol = max(monthly_budget_pp * 1e-10, 1e-8)
            _max_dynamic_steps = max(100, int(np.ceil(25.0 / max(round_budget_pct, 1e-6))))
            _dyn_iter = 0

            while _rem > _min_step:
                _dyn_iter += 1
                if _dyn_iter > _max_dynamic_steps:
                    logger.warning(
                        f"  [SEQ-OPT] month {m}: dynamic allocation stopped after "
                        f"{_max_dynamic_steps} steps with £{_rem * n_ppm:,.2f} "
                        "monthly spend still unallocated. Check channel max/share bounds."
                    )
                    break

                _prev_alloc = alloc.copy()
                _prev_rem = _rem
                _dyn_step = max(_rem * round_budget_pct, _min_step)
                alloc, _, _ = _greedy_step_sequential(
                    alloc, _dyn_step, n_ppm, opt_carry,
                    lb, ub, mean_params, cpp_w, channel_rscales, mroi_floor, slope_scaling,
                )
                # Recompute how much budget is still unallocated.  Clamp tiny
                # floating-point negatives to zero so the loop exits cleanly.
                allocated_spend = (alloc * cpp_w).sum() - mandatory_spend
                _rem = max(0.0, remaining_spend - allocated_spend)

                if (
                    np.allclose(alloc, _prev_alloc, rtol=1e-10, atol=1e-12)
                    or _rem >= _prev_rem - _no_progress_tol
                ):
                    logger.warning(
                        f"  [SEQ-OPT] month {m}: dynamic allocation made no "
                        f"further progress; £{_rem * n_ppm:,.2f} monthly spend "
                        "remains unallocated because feasible channel headroom is exhausted."
                    )
                    break
        else:
            n_st = max(1, round(remaining_spend / (step_pp_spend + 1e-30)))
            for _step in range(n_st):
                alloc, _, _ = _greedy_step_sequential(
                    alloc, step_pp_spend, n_ppm, opt_carry,
                    lb, ub, mean_params, cpp_w, channel_rscales, mroi_floor, slope_scaling,
                )

        opt_alloc_total += alloc
        opt_resp_m, new_carry_opt = _monthly_response_total(alloc, n_ppm, opt_carry, mean_params, channel_rscales)
        opt_carry = new_carry_opt

        # ── BAU allocation for this month ─────────────────────────────────────
        bau_alloc = _simulated_bau_for_month(monthly_budget_pp, current_spend_pp, cpp_w)
        bau_alloc = np.clip(bau_alloc, lb, ub)
        bau_alloc_total += bau_alloc
        bau_resp_m, new_carry_bau = _monthly_response_total(bau_alloc, n_ppm, bau_carry, mean_params, channel_rscales)
        bau_carry = new_carry_bau

        # ── Month current (no reallocation, observed mix & spend) ─────────────
        curr_resp_m = current_resp_monthly

        monthly_rows.append({
            "month"              : m,
            "opt_response"       : round(opt_resp_m,    4),
            "bau_response"       : round(bau_resp_m,    4),
            "current_response"   : round(curr_resp_m,   4),
            "opt_total_spend_£"  : round(float((alloc * cpp_w).sum() * n_ppm), 2),
            "bau_total_spend_£"  : round(float((bau_alloc * cpp_w).sum() * n_ppm), 2),
            "current_spend_£"    : round(float((current_spend_pp * cpp_w).sum() * n_ppm), 2),
        })
        for j, ch in enumerate(spend_cols):
            monthly_rows[-1][f"opt_spend_£_{ch}"]  = round(float(alloc[j] * cpp_w[j] * n_ppm), 2)
            monthly_rows[-1][f"bau_spend_£_{ch}"]  = round(float(bau_alloc[j] * cpp_w[j] * n_ppm), 2)

        logger.info(
            f"  [SEQ-OPT] month={m}  opt={opt_resp_m:.2f}  "
            f"bau={bau_resp_m:.2f}  curr={curr_resp_m:.2f}"
        )

    df_monthly = pd.DataFrame(monthly_rows)

    # ── Per-channel summary (average monthly allocation × n_months → total) ──
    avg_opt_alloc = opt_alloc_total / n_months
    avg_bau_alloc = bau_alloc_total / n_months

    ch_rows = []
    metric_types = prep.get("metric_types", ["Spend"] * C)
    for j, ch in enumerate(spend_cols):
        # Per-channel response: compute only channel j's terms using the
        # single-channel helper. This avoids softplus(0) != 0 contamination
        # that occurs when _monthly_response_total loops over all channels
        # with zero spend for the inactive ones.
        opt_r_j  = _channel_j_monthly_response(j, avg_opt_alloc[j],    n_ppm, ss_carry[j], mean_params[j], channel_rscales[j])
        bau_r_j  = _channel_j_monthly_response(j, avg_bau_alloc[j],    n_ppm, ss_carry[j], mean_params[j], channel_rscales[j])
        curr_r_j = _channel_j_monthly_response(j, current_spend_pp[j], n_ppm, ss_carry[j], mean_params[j], channel_rscales[j])

        opt_sp_gbp  = float(avg_opt_alloc[j] * cpp_w[j] * n_ppm * n_months)
        bau_sp_gbp  = float(avg_bau_alloc[j] * cpp_w[j] * n_ppm * n_months)
        curr_sp_gbp = float(current_spend_pp[j] * cpp_w[j] * n_ppm * n_months)

        opt_r_tot   = float(opt_r_j)  * n_months
        bau_r_tot   = float(bau_r_j)  * n_months
        curr_r_tot  = float(curr_r_j) * n_months

        ch_rows.append({
            "channel"             : ch,
            "metric_type"         : metric_types[j] if j < len(metric_types) else "Spend",
            "n_months"            : n_months,
            "current_spend_gbp"   : round(curr_sp_gbp, 2),
            "current_response"    : round(curr_r_tot,  4),
            "current_roi"         : round(curr_r_tot / (curr_sp_gbp + 1e-12), 6),
            "bau_spend_gbp"       : round(bau_sp_gbp,  2),
            "bau_response"        : round(bau_r_tot,   4),
            "bau_roi"             : round(bau_r_tot / (bau_sp_gbp + 1e-12), 6),
            "opt_spend_gbp"       : round(opt_sp_gbp,  2),
            "opt_response"        : round(opt_r_tot,   4),
            "opt_roi"             : round(opt_r_tot / (opt_sp_gbp + 1e-12), 6),
            "pct_change_spend"    : round((opt_sp_gbp  - curr_sp_gbp)  / (curr_sp_gbp  + 1e-12) * 100, 2),
            "pct_change_response" : round((opt_r_tot   - curr_r_tot)   / (curr_r_tot   + 1e-12) * 100, 2),
            # Constraint status: shows whether the channel hit its min/max allocation bound
            "constraint_status"   : (
                "At min bound" if abs(float(avg_opt_alloc[j]) - float(lb[j])) < 1e-6 * (float(ub[j]) - float(lb[j]) + 1e-12)
                else "At max bound" if abs(float(avg_opt_alloc[j]) - float(ub[j])) < 1e-6 * (float(ub[j]) - float(lb[j]) + 1e-12)
                else "Unconstrained"
            ),
        })

    df_channels = pd.DataFrame(ch_rows)

    # Summary
    tot_opt_resp  = float(df_monthly["opt_response"].sum())
    tot_bau_resp  = float(df_monthly["bau_response"].sum())
    tot_curr_resp = float(df_monthly["current_response"].sum())
    tot_opt_sp    = float(df_channels["opt_spend_gbp"].sum())
    tot_bau_sp    = float(df_channels["bau_spend_gbp"].sum())
    tot_curr_sp   = float(df_channels["current_spend_gbp"].sum())

    df_summary = pd.DataFrame([{
        "n_months"                       : n_months,
        "total_budget_gbp"               : round(total_budget, 2),
        "step_size_gbp"                  : round(step_size, 2) if step_size is not None else None,
        "target_cpa"                     : target_cpa,
        "opt_total_spend_gbp"            : round(tot_opt_sp, 2),
        "opt_total_response"             : round(tot_opt_resp, 4),
        "opt_roi"                        : round(tot_opt_resp  / (tot_opt_sp  + 1e-12), 6),
        "bau_total_spend_gbp"            : round(tot_bau_sp, 2),
        "bau_total_response"             : round(tot_bau_resp, 4),
        "bau_roi"                        : round(tot_bau_resp  / (tot_bau_sp  + 1e-12), 6),
        "current_total_spend_gbp"        : round(tot_curr_sp, 2),
        "current_total_response"         : round(tot_curr_resp, 4),
        "current_roi"                    : round(tot_curr_resp / (tot_curr_sp + 1e-12), 6),
        "response_uplift_vs_bau_pct"     : round((tot_opt_resp - tot_bau_resp)  / (tot_bau_resp  + 1e-12) * 100, 2),
        "response_uplift_vs_current_pct" : round((tot_opt_resp - tot_curr_resp) / (tot_curr_resp + 1e-12) * 100, 2),
    }])

    logger.info(
        f"  [SEQ-OPT] opt={tot_opt_resp:.4f}  bau={tot_bau_resp:.4f}  "
        f"current={tot_curr_resp:.4f}  "
        f"uplift_vs_bau={df_summary['response_uplift_vs_bau_pct'].iloc[0]:+.1f}%"
    )
    return df_summary, df_channels, df_monthly


# ─── CPP layer (cpp_layer.py) ────────────────────────────────────────────────

from pathlib import Path as _CppPath
from typing import List as _CppList

# -- Metric display labels ----------------------------------------------------

METRIC_LABELS: Dict[str, str] = {
    "spend"       : "Spend",
    "Spend"       : "Spend",
    "impressions" : "Impressions",
    "Impressions" : "Impressions",
    "clicks"      : "Clicks",
    "Clicks"      : "Clicks",
    "grp"         : "GRPs",
    "GRP"         : "GRPs",
    "GRPs"        : "GRPs",
    "views"       : "Views",
    "Views"       : "Views",
    "reach"       : "Reach",
    "Reach"       : "Reach",
    "numeric"     : "Units",
    "Metric"      : "Units",
}

CPP_LABEL_BY_METRIC: Dict[str, str] = {
    "impressions" : "CPM (GBP / 1000 imps)",
    "Impressions" : "CPM (GBP / 1000 imps)",
    "clicks"      : "CPC (GBP / click)",
    "Clicks"      : "CPC (GBP / click)",
    "grp"         : "CPP (GBP / GRP)",
    "GRP"         : "CPP (GBP / GRP)",
    "GRPs"        : "CPP (GBP / GRP)",
    "views"       : "CPV (GBP / view)",
    "Views"       : "CPV (GBP / view)",
    "reach"       : "CPR (GBP / reach unit)",
    "Reach"       : "CPR (GBP / reach unit)",
}


def _is_spend_metric(mt: str) -> bool:
    return mt.strip().lower() in ("spend", "numeric", "metric", "")


# Keyword sets used by _infer_metric_from_name.
# Checked against the lower-cased column name; longer/more-specific keywords first.
_MEDIA_KEYWORDS: List[Tuple[str, str]] = [
    ("impression", "Impressions"),
    ("click",      "Clicks"),
    ("grp",        "GRPs"),
    ("view",       "Views"),
    ("reach",      "Reach"),
]
_SPEND_KEYWORDS = {"spend", "cost", "budget", "gbp", "usd", "eur"}


def _infer_metric_from_name(col: str) -> Optional[str]:
    """
    Infer metric type from column name keywords.
    Returns a canonical metric label (e.g. 'Impressions') or None if ambiguous.
    Called only when prep's metric_types entry is unknown ('Metric'/'Units').
    """
    lower = col.lower()
    for kw, mt in _MEDIA_KEYWORDS:
        if kw in lower:
            return mt
    for kw in _SPEND_KEYWORDS:
        if kw in lower:
            return "Spend"
    return None


def build_cpp_map(
    prep          : Dict[str, Any],
    cpp_overrides : Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a per-channel CPP map from the observed training data.
    """
    spend_cols     = prep.get("spend_cols", [])
    train_idx      = prep.get("train_idx", slice(None))
    X_media_raw    = prep.get("X_media_raw")
    metric_types   = prep.get("metric_types", [])
    spend_raw_cols = prep.get("spend_raw_cols", [])
    data_cfg       = prep.get("data_cfg")
    cpp_overrides  = cpp_overrides or {}

    raw_df: Optional[pd.DataFrame] = None
    if data_cfg is not None:
        csv_path = getattr(data_cfg, "csv_path", None)
        if csv_path and _CppPath(str(csv_path)).exists():
            try:
                raw_df = pd.read_csv(str(csv_path), low_memory=False)
                logger.debug(f"  [CPP] Loaded source CSV: {csv_path}")
            except Exception as exc:
                logger.warning(f"  [CPP] Could not load CSV for CPP computation: {exc}")

    # For impressions, the internal CPP is GBP/impression but the display
    # convention is CPM (GBP/1000 impressions), so multiply by 1000 for display.
    _DISPLAY_SCALE: Dict[str, float] = {
        "impressions": 1000.0,
        "Impressions": 1000.0,
    }

    cpp_map: Dict[str, Dict[str, Any]] = {}

    for j, ch in enumerate(spend_cols):
        mt        = metric_types[j] if j < len(metric_types) else "Spend"
        spend_col = spend_raw_cols[j] if j < len(spend_raw_cols) else ch

        # ── Auto-detect metric type if prep's entry is ambiguous ─────────────
        # prep's metric_types is set from column-name prefix matching during
        # data_prep; it is reliable for standard prefixes (media_impressions_*,
        # media_clicks_*, spends_*, etc.).  Only fall back to keyword search or
        # heuristics when it returned the generic placeholder "Metric"/"Units".
        if _is_spend_metric(mt) and mt.strip().lower() in ("metric", "units", ""):
            inferred = _infer_metric_from_name(ch)
            if inferred:
                mt = inferred
                logger.info(f"  [CPP] {ch}: metric type inferred from column name -> {mt}")
            elif spend_col != ch:
                # Separate spend column exists → channel column holds media units
                mt = "Impressions"
                logger.warning(
                    f"  [CPP] {ch}: type unknown but separate spend column '{spend_col}' "
                    f"found — treating as Impressions. Add a cpp_rates override in YAML to correct."
                )
            else:
                # Completely unknown: treat as spend, warn loudly
                mt = "Spend"
                logger.warning(
                    f"  [CPP] {ch}: metric type cannot be determined — treating as spend "
                    f"(CPP=1.0). Rename the column with a known prefix (media_impressions_*, "
                    f"media_clicks_*, spends_*, ...) or add a cpp_rates override in YAML."
                )

        is_spend  = _is_spend_metric(mt)
        label     = METRIC_LABELS.get(mt, mt)
        cpp_label = CPP_LABEL_BY_METRIC.get(mt, "Cost per unit")

        if ch in cpp_overrides:
            cpp = float(cpp_overrides[ch])
            source = "YAML override"
        elif is_spend:
            cpp = 1.0
            source = "direct spend (identity)"
        else:
            cpp = _compute_cpp(j, spend_col, raw_df, X_media_raw, train_idx)
            source = "computed from data"

        # display_cpp: CPM for impressions (cpp × 1000), raw cpp for all others
        display_scale = _DISPLAY_SCALE.get(mt, 1.0) if not is_spend else 1.0
        display_cpp   = cpp * display_scale

        if is_spend:
            logger.info(f"  [CPP] {ch:<40s}  modelled as spend — no conversion")
        else:
            logger.info(
                f"  [CPP] {ch:<40s}  {label:<15s}  "
                f"{cpp_label} = {display_cpp:.6f}  [{source}]"
            )

        cpp_map[ch] = {
            "cpp"         : cpp,          # GBP per raw media unit — used in all calculations
            "display_cpp" : display_cpp,  # human-readable rate (CPM for impressions)
            "metric_type" : label,
            "spend_col"   : spend_col,
            "label"       : label,
            "cpp_label"   : cpp_label,
            "is_spend"    : is_spend,
        }

    return cpp_map


def _compute_cpp(
    j          : int,
    spend_col  : str,
    raw_df     : Optional[pd.DataFrame],
    X_media_raw: Optional[np.ndarray],
    train_idx  : Any,
) -> float:
    """Compute CPP from CSV data, falling back to 1.0."""
    if raw_df is None or spend_col not in raw_df.columns:
        if raw_df is not None:
            logger.warning(
                f"  [CPP] Spend column '{spend_col}' not found in CSV — "
                "using CPP=1.0. Set optimisation.cpp_rates in YAML to override."
            )
        return 1.0

    if X_media_raw is None:
        return 1.0

    T = X_media_raw.shape[0]
    try:
        spend_vals = raw_df[spend_col].values[:T].astype(float)
        if hasattr(train_idx, "__len__"):
            spend_train = spend_vals[train_idx]
        else:
            spend_train = spend_vals[train_idx]
    except Exception as exc:
        logger.warning(f"  [CPP] Error extracting spend for column '{spend_col}': {exc}")
        return 1.0

    if X_media_raw.ndim == 3:
        media_train = X_media_raw[train_idx, 0, j].astype(float)
    else:
        media_train = X_media_raw[train_idx, j].astype(float)

    # Use only periods where BOTH spend and media are positive (truly active weeks).
    # Sum-ratio (total_spend / total_media) is volume-weighted and more stable than
    # mean(spend) / mean(media) which can be biased by outlier weeks.
    both_active = (spend_train > 0) & (media_train > 0)
    if both_active.sum() == 0:
        logger.warning(
            f"  [CPP] No periods with spend>0 AND media>0 for '{spend_col}' — using CPP=1.0"
        )
        return 1.0

    total_spend = float(spend_train[both_active].sum())
    total_media = float(media_train[both_active].sum())

    if total_media < 1e-12 or total_spend < 1e-12:
        logger.warning(
            f"  [CPP] Near-zero spend or media for '{spend_col}' — using CPP=1.0"
        )
        return 1.0

    return total_spend / total_media


def media_to_spend(
    media_alloc : Dict[str, float],
    cpp_map     : Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Convert a {channel: media_units} dict -> {channel: spend} dict."""
    return {
        ch: float(units) * cpp_map.get(ch, {}).get("cpp", 1.0)
        for ch, units in media_alloc.items()
    }


def spend_to_media(
    spend_alloc : Dict[str, float],
    cpp_map     : Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Convert a {channel: spend} dict -> {channel: media_units} dict."""
    return {
        ch: float(sp) / max(cpp_map.get(ch, {}).get("cpp", 1.0), 1e-30)
        for ch, sp in spend_alloc.items()
    }


def cpp_weights_array(
    spend_cols : List[str],
    cpp_map    : Dict[str, Dict[str, Any]],
) -> np.ndarray:
    """
    Return a (C,) array of CPP values aligned to spend_cols.
    Used by the optimiser to enforce spend-weighted budget constraints.
    """
    return np.array([cpp_map.get(ch, {}).get("cpp", 1.0) for ch in spend_cols])


def add_spend_equivalents(
    df        : pd.DataFrame,
    cpp_map   : Dict[str, Dict[str, Any]],
    alloc_col : str = "optimal_spend_mean",
) -> pd.DataFrame:
    """
    Add spend-equivalent columns to an optimiser result DataFrame.
    """
    if alloc_col not in df.columns:
        return df

    df = df.copy()
    non_total = df["channel"] != "TOTAL"

    def _equiv(row):
        ch = row.get("channel", "")
        cpp = cpp_map.get(ch, {}).get("cpp", 1.0)
        return round(float(row[alloc_col]) * cpp, 2)

    df.loc[non_total, "spend_equiv_mean"] = df.loc[non_total].apply(_equiv, axis=1)
    total_mask = ~non_total
    if total_mask.any():
        df.loc[total_mask, "spend_equiv_mean"] = round(
            df.loc[non_total, "spend_equiv_mean"].sum(), 2
        )

    stem = alloc_col.replace("_mean", "")
    for suffix in ("_hdi_10", "_hdi_90"):
        hdi_col = f"{stem}{suffix}"
        new_col = f"spend_equiv{suffix}"
        if hdi_col in df.columns:
            df.loc[non_total, new_col] = df.loc[non_total].apply(
                lambda row: round(
                    float(row[hdi_col]) * cpp_map.get(row.get("channel", ""), {}).get("cpp", 1.0), 2
                ),
                axis=1,
            )
            if total_mask.any():
                df.loc[total_mask, new_col] = round(
                    df.loc[non_total, new_col].sum(), 2
                )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Reverse sequential optimiser
# ─────────────────────────────────────────────────────────────────────────────

def minimise_spend_sequential(
    best               : Dict[str, Any],
    prep               : Dict[str, Any],
    target_response    : Optional[float]           = None,
    n_months           : int                       = 1,
    n_samples          : int                       = 1,
    channel_min        : Optional[Dict[str, float]] = None,
    channel_max        : Optional[Dict[str, float]] = None,
    channel_share_min  : Optional[Dict[str, float]] = None,
    channel_share_max  : Optional[Dict[str, float]] = None,
    cpp_weights        : Optional[np.ndarray]       = None,
    target_cpa         : Optional[float]            = None,
    step_size          : Optional[float]            = None,
    round_budget_pct   : Optional[float]            = None,
    slope_scaling      : float                      = 1e4,
    tol_pct            : float                      = 0.02,
    max_iter           : int                        = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reverse sequential optimiser: find the MINIMUM total budget for n_months
    that achieves ``target_response`` per month on average.

    Uses binary search over ``total_budget``, calling
    ``optimise_budget_sequential()`` at each candidate budget until the
    achieved average monthly response is within ``tol_pct`` of the target.

    Parameters
    ----------
    target_response  : desired average monthly response (signups/deals).
                       None = use the current observed mean monthly response.
    tol_pct          : convergence tolerance as fraction of target (default 0.5%).
    max_iter         : maximum binary-search iterations (default 40).
    (all other params forwarded to optimise_budget_sequential)

    Returns
    -------
    Same (df_summary, df_channels, df_monthly) as optimise_budget_sequential,
    but df_channels columns reflect the minimum-spend allocation.
    The df_summary includes an extra column ``target_response`` for reference.
    """
    # ── Derive observed monthly budget and response ───────────────────────────
    frequency  = prep.get("frequency", "weekly")
    train_idx  = prep["train_idx"]
    n_train    = len(train_idx)
    cpp_w      = np.asarray(cpp_weights, dtype=float) if cpp_weights is not None else np.ones(
        len(prep["spend_cols"])
    )

    ppm   = _PERIODS_PER_MONTH.get(frequency, _PERIODS_PER_MONTH["weekly"])
    n_ppm = max(1, round(ppm))
    X_rw  = prep["X_media_raw"][train_idx]
    cur_pp = X_rw[:, 0, :].mean(axis=0) if X_rw.ndim == 3 else X_rw.mean(axis=0)
    observed_monthly_budget = float((cur_pp * cpp_w).sum()) * n_ppm

    # If target not specified, use current observed monthly response
    # computed via simulation with warm-start carry (steady-state at current spend).
    if target_response is None:
        _sp, _params     = _extract_all_channel_params(best, prep, 1)
        _params          = _calibrate_betas_to_trace(_params, best, prep)
        _channel_rscales = _compute_channel_rscales(best, prep)
        mean_p  = [
            {
                "sat_type" : p["sat_type"],
                "ads_type" : p.get("ads_type", "geometric"),
                "spend_max": p["spend_max"],
                "lam"      : float(p["lam"].mean()),
                "beta"     : float(p["beta"].mean()),
                "alpha"    : float(p["alpha"].mean()),
                "kappa"    : float(p["kappa"].mean()),
                "k_log"    : float(p["k_log"].mean()),
                "x0"       : float(p["x0"].mean()),
            }
            for p in _params
        ]
        # Warm-start carry: steady-state at current observed spend
        _lam_arr = np.array([float(p["lam"].mean()) for p in _params])
        _x_sc    = cur_pp / np.array([p["spend_max"] + 1e-8 for p in _params])
        _ss_carry = _lam_arr * _x_sc / np.maximum(1.0 - _lam_arr, 1e-8)
        for _j in range(len(_sp)):
            if mean_p[_j].get("ads_type", "geometric") == "weibull":
                _ss_carry[_j] = 0.0
        target_response, _ = _monthly_response_total(
            cur_pp, n_ppm, _ss_carry, mean_p, _channel_rscales
        )
        logger.info(
            f"  [REV-SEQ] target_response not set — using observed "
            f"monthly response (warm-start total): {target_response:.2f}"
        )

    logger.info(
        f"  [REV-SEQ] n_months={n_months}  target={target_response:.2f}/month  "
        f"tol={tol_pct*100:.1f}%"
    )

    # ── Binary search bounds ──────────────────────────────────────────────────
    lo_monthly = observed_monthly_budget * 0.01   # 1% of current
    hi_monthly = observed_monthly_budget * 3.0    # upper search budget; checked below

    best_result = None
    best_budget = hi_monthly

    # Use a coarser step for the binary search inner runs — precision here
    # doesn't matter, only the achieved response level does.
    _inner_rbp = max(round_budget_pct or 0.0, 0.05) if (round_budget_pct or step_size is None) else None
    _inner_step = step_size  # kept if user explicitly set it

    _kw = dict(
        n_months          = n_months,
        n_samples         = n_samples,
        channel_min       = channel_min,
        channel_max       = channel_max,
        channel_share_min = channel_share_min,
        channel_share_max = channel_share_max,
        cpp_weights       = cpp_weights,
        target_cpa        = target_cpa,
        step_size         = _inner_step,
        round_budget_pct  = _inner_rbp,
        slope_scaling     = slope_scaling,
    )

    # Achievability pre-check: if the high budget cannot hit the target, binary
    # search cannot find a feasible lower budget.  Return the high-budget result
    # immediately instead of repeatedly calling the inner optimiser.
    try:
        df_s_hi, df_c_hi, df_m_hi = optimise_budget_sequential(
            best, prep, total_budget=hi_monthly * n_months, **_kw
        )
        hi_achieved = float(df_m_hi["opt_response"].mean()) if not df_m_hi.empty else 0.0
        hi_gap = (hi_achieved - target_response) / (target_response + 1e-12)
        logger.info(
            f"  [REV-SEQ] achievability check at monthly=£{hi_monthly:,.0f}: "
            f"achieved={hi_achieved:.2f} target={target_response:.2f} gap={hi_gap:+.3f}"
        )
        if hi_achieved < target_response:
            logger.warning(
                "  [REV-SEQ] Target response is not reachable at the high "
                "search budget (3× observed monthly spend) under the current "
                "channel bounds/share constraints — returning that best attempt."
            )
            best_result = (df_s_hi, df_c_hi, df_m_hi)
    except Exception as _e:
        logger.warning(
            f"  [REV-SEQ] Achievability check at 3× observed spend failed: {_e}. "
            "Continuing with bounded binary search."
        )

    for _i in range(max_iter if best_result is None else 0):
        mid_monthly = (lo_monthly + hi_monthly) / 2.0
        total_b     = mid_monthly * n_months

        try:
            df_s, df_c, df_m = optimise_budget_sequential(
                best, prep, total_budget=total_b, **_kw
            )
        except Exception as _e:
            logger.info(f"  [REV-SEQ] iter {_i + 1}/{max_iter}: budget={total_b:.0f} failed: {_e}")
            lo_monthly = mid_monthly
            continue

        achieved = float(df_m["opt_response"].mean()) if not df_m.empty else 0.0
        gap      = (achieved - target_response) / (target_response + 1e-12)

        logger.info(
            f"  [REV-SEQ] iter {_i + 1}/{max_iter}: monthly=£{mid_monthly:,.0f}  "
            f"achieved={achieved:.2f}  target={target_response:.2f}  gap={gap:+.3f}"
        )

        if achieved >= target_response:
            best_result = (df_s, df_c, df_m)
            best_budget = mid_monthly
            hi_monthly  = mid_monthly          # can we do it cheaper?
        else:
            lo_monthly  = mid_monthly          # need more budget

        if abs(gap) <= tol_pct and achieved >= target_response:
            logger.info(
                f"  [REV-SEQ] Converged at monthly=£{mid_monthly:,.0f} "
                f"(gap={gap:+.3f})"
            )
            break

    if best_result is None:
        logger.warning(
            "  [REV-SEQ] Could not reach target response even at 3× current "
            "budget — returning 3× current result."
        )
        df_s, df_c, df_m = optimise_budget_sequential(
            best, prep, total_budget=hi_monthly * n_months, **_kw
        )
        best_result = (df_s, df_c, df_m)

    # Tag minimum budget into summary
    df_s, df_c, df_m = best_result
    if not df_s.empty:
        df_s = df_s.copy()
        df_s["minimum_monthly_budget_£"] = round(best_budget, 2)
        df_s["target_response_per_month"] = round(target_response, 4)
        df_s["achieved_response_per_month"] = round(
            float(df_m["opt_response"].mean()) if not df_m.empty else 0.0, 4
        )

    logger.info(
        f"  [REV-SEQ] Minimum monthly budget to achieve "
        f"{target_response:.1f} signups/month: £{best_budget:,.0f}  "
        f"(vs observed £{observed_monthly_budget:,.0f})"
    )
    return df_s, df_c, df_m


def get_rc_x_label(ch: str, cpp_map: Dict[str, Dict[str, Any]]) -> str:
    """Return the correct x-axis label for a response curve plot."""
    info = cpp_map.get(ch, {})
    if not info or info.get("is_spend", True):
        return "Spend (original units)"
    label = info.get("label", "Media units")
    cpp   = info.get("cpp", 1.0)
    return f"{label}  [GBP {cpp:.4f}/unit]"


def convert_rc_x_to_spend(
    x_unscaled : np.ndarray,
    ch         : str,
    cpp_map    : Dict[str, Dict[str, Any]],
) -> np.ndarray:
    """Convert response-curve x values (media units) -> spend (£)."""
    cpp = cpp_map.get(ch, {}).get("cpp", 1.0)
    return x_unscaled * cpp


def format_cpp_summary(cpp_map: Dict[str, Dict[str, Any]]) -> str:
    """Return a human-readable CPP summary."""
    lines = ["  Channel metric types and cost-per-unit rates:"]
    lines.append(f"  {'Channel':<45s} {'Metric':<16s} {'Rate'}")
    lines.append("  " + "-" * 80)
    for ch, info in cpp_map.items():
        mt          = info.get("metric_type", "Spend")
        display_cpp = info.get("display_cpp", info.get("cpp", 1.0))
        label       = info.get("cpp_label", "")
        if info.get("is_spend", True):
            rate_str = "direct spend — no conversion"
        else:
            rate_str = f"{display_cpp:.6f}  ({label})"
        lines.append(f"  {ch:<45s} {mt:<16s} {rate_str}")
    return "\n".join(lines)
