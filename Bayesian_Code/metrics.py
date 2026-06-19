# metrics.py
# ─────────────────────────────────────────────────────────────────────────────
# Computes all model quality metrics from the MCMC trace.
#
# Two types of metrics:
#
#   Fit quality  — how well does the model predict the training data?
#     MAPE        mean absolute percentage error  (want < 10-15%)
#     R²          variance explained              (want > 0.85)
#     Pearson r   correlation of fitted vs actual
#
#   Convergence  — did the MCMC sampler explore the posterior properly?
#     R-hat       chain agreement                 (want ≤ 1.01)
#     ESS         effective sample size           (want ≥ 400)
#     Divergences number of divergent transitions (want 0)
#     LOO-IC      leave-one-out information criterion (lower = better)
#
# quality_flag(metrics) summarises all of the above into a single number:
#   0 = production ready   1 = acceptable   2 = poor (re-run with more draws)
# ─────────────────────────────────────────────────────────────────────────────

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import arviz as az

from config import ModelConfig, GLOBAL_SEED, ADSTOCK_SAT_COMBOS

logger = logging.getLogger("MMM")


# ─────────────────────────────────────────────────────────────────────────────
# Posterior helpers
# ─────────────────────────────────────────────────────────────────────────────

def posterior_mu_mean(trace: az.InferenceData) -> np.ndarray:
    """Extracts posterior mean of 'mu' (in standardised log-space). Returns shape (T,)."""
    return (
        trace.posterior["mu"]
        .stack(sample=("chain", "draw"))
        .mean("sample")
        .values
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fit metrics
# ─────────────────────────────────────────────────────────────────────────────

def _inv_transform(y_transformed: np.ndarray, transform: str = "log1p", boxcox_lam: float = None) -> np.ndarray:
    """Reverse a response transform to get original-scale values."""
    if transform == "log1p":
        return np.expm1(y_transformed)
    elif transform == "sqrt":
        return np.maximum(y_transformed, 0.0) ** 2
    elif transform == "boxcox":
        if boxcox_lam is None or abs(boxcox_lam) < 1e-10:
            return np.exp(y_transformed)
        return np.maximum((y_transformed * boxcox_lam + 1.0) ** (1.0 / boxcox_lam), 0.0)
    elif transform == "identity":
        return y_transformed
    return np.expm1(y_transformed)  # fallback


def _fit_metrics_single(
    y_true_raw     : np.ndarray,   # (T,)
    y_hat_samples  : np.ndarray,   # (T, N_samples)
) -> Dict[str, float]:
    """Core MAPE / R² / RMSE / Pearson r computation for a single (T,) series."""
    mask         = y_true_raw > 1.0
    y_true_2d    = y_true_raw[mask, np.newaxis]
    y_hat_masked = y_hat_samples[mask, :]
    if y_hat_masked.shape[0] == 0:
        mape_samples = np.array([np.nan])
    else:
        mape_samples = (
            np.abs(y_true_2d - y_hat_masked) / (y_true_2d + 1e-12)
        ).mean(axis=0) * 100.0

    y_mean     = float(np.mean(y_true_raw))
    ss_tot     = float(np.sum((y_true_raw - y_mean) ** 2))
    residuals  = y_true_raw[:, np.newaxis] - y_hat_samples
    ss_res     = (residuals ** 2).sum(axis=0)
    r2_samples = 1.0 - ss_res / (ss_tot + 1e-12)

    y_hat_mean = y_hat_samples.mean(axis=1)
    rmse       = float(np.sqrt(np.mean((y_true_raw - y_hat_mean) ** 2)))
    pearson_r  = float(np.corrcoef(y_true_raw, y_hat_mean)[0, 1])

    return {
        "mape"          : float(np.nanmean(mape_samples)),
        "mape_std"      : float(np.nanstd(mape_samples)),
        "mape_hdi_low"  : float(np.nanpercentile(mape_samples, 5)),
        "mape_hdi_high" : float(np.nanpercentile(mape_samples, 95)),
        "rmse"          : rmse,
        "r2"            : float(r2_samples.mean()),
        "r2_std"        : float(r2_samples.std()),
        "r2_hdi_low"    : float(np.percentile(r2_samples, 5)),
        "r2_hdi_high"   : float(np.percentile(r2_samples, 95)),
        "pearson_r"     : pearson_r,
    }


def compute_fit_metrics(
    y_scaled : np.ndarray,
    y_raw    : np.ndarray,
    y_mu,                          # float (P=1) or (P,) array (P>1)
    y_std,                         # float (P=1) or (P,) array (P>1)
    trace    : az.InferenceData,
    response_transform : str = "log1p",
    boxcox_lambda      : float = None,
    product_names      : Optional[List[str]] = None,
    y_raw_by_product   : Optional[np.ndarray] = None,  # (T, P) for P>1
) -> Dict[str, float]:
    """
    Computes in-sample fit metrics with Bayesian uncertainty quantification.

    Handles both flat (P=1) and multi-product (P>1) models:
    - P=1: operates on (T,) y_raw and scalar y_mu/y_std as before.
    - P>1: computes aggregate metrics averaged over all products and
           stores per-product breakdowns under 'per_product_metrics'.

    Returns MAPE, RMSE, Bayesian R², Pearson r on the original (raw) scale.
    """
    # mu_post shape after stack: (T, N_samples) for P=1 | (T, P, N_samples) for P>1
    mu_post = (
        trace.posterior["mu"]
        .stack(sample=("chain", "draw"))
        .values
    )

    is_multi = mu_post.ndim == 3   # (T, P, N_samples)

    if not is_multi:
        # ── Flat path (unchanged) ──────────────────────────────
        mu_transformed = mu_post * float(y_std) + float(y_mu)
        y_hat_samples  = _inv_transform(mu_transformed, response_transform, boxcox_lambda)
        result = _fit_metrics_single(y_raw, y_hat_samples)
        return result

    # ── Multi-product path ────────────────────────────────────
    # mu_post: (T, P, N_samples)
    P = mu_post.shape[1]
    y_mu_arr  = np.asarray(y_mu)    # (P,)
    y_std_arr = np.asarray(y_std)   # (P,)

    per_product: List[Dict] = []
    for p_idx in range(P):
        mu_p          = mu_post[:, p_idx, :]   # (T, N_samples)
        mu_transformed = mu_p * float(y_std_arr[p_idx]) + float(y_mu_arr[p_idx])
        y_hat_p        = _inv_transform(mu_transformed, response_transform, boxcox_lambda)

        # Use per-product raw response if supplied, else fall back to y_raw
        if y_raw_by_product is not None and y_raw_by_product.ndim == 2:
            y_raw_p = y_raw_by_product[:, p_idx]
        else:
            y_raw_p = y_raw   # fallback: same for all products

        per_product.append(_fit_metrics_single(y_raw_p, y_hat_p))

    # Aggregate: mean over products
    agg: Dict[str, float] = {}
    for key in per_product[0]:
        vals = [pp[key] for pp in per_product if not np.isnan(pp[key])]
        agg[key] = float(np.mean(vals)) if vals else np.nan

    # Store per-product breakdowns using product names if provided
    p_names = product_names or [f"product_{p}" for p in range(P)]
    agg["per_product_metrics"] = {
        p_names[p]: per_product[p] for p in range(P)
    }
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Convergence diagnostics  ← R-hat null fix lives here
# ─────────────────────────────────────────────────────────────────────────────

def _n_chains(trace: az.InferenceData) -> int:
    """Return the number of chains in an InferenceData object."""
    try:
        return int(trace.posterior.dims["chain"])
    except Exception:
        return 1


def compute_convergence_diagnostics(trace: az.InferenceData) -> Dict[str, float]:
    """
    Computes MCMC convergence diagnostics via ArviZ.

    FIX — R-hat null with 1 chain
    ──────────────────────────────
    ArviZ requires >= 2 chains to compute R-hat. With fast_mode (1 chain),
    every R-hat value is NaN.  The old code called .max() / .mean() on a
    fully-NaN Series, propagating null into every downstream metric and CSV.

    Resolution: when n_chains == 1 we skip R-hat computation and return
    the sentinel value 1.0 (indeterminate but not alarming) plus set
    max_rhat_is_estimated=True so callers know it wasn't actually measured.

    Returns
    -------
    max_rhat      : worst R-hat (want ≤ 1.01); 1.0 sentinel when 1 chain
    mean_rhat     : average R-hat; 1.0 sentinel when 1 chain
    ess_bulk      : minimum bulk ESS (want >= 400)
    ess_tail      : minimum tail ESS (want >= 400)
    divergences   : divergent transitions (want 0)
    div_rate      : divergences / total transitions
    """
    n_ch = _n_chains(trace)

    if n_ch < 2:
        # ── 1-chain path: R-hat cannot be computed ─────────────
        logger.warning(
            "R-hat cannot be computed with only 1 chain - returning sentinel 1.0. "
            "Use chains >= 2 for production runs."
        )
        # ESS is still valid with 1 chain
        try:
            summary  = az.summary(
                trace,
                var_names=["~mu", "~baseline", "~seasonality",
                           "~media_by_channel", "~control_effect"],
                round_to=6,
            )
            ess_bulk = float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns else 0.0
            ess_tail = float(summary["ess_tail"].min()) if "ess_tail" in summary.columns else 0.0
        except Exception:
            ess_bulk = 0.0
            ess_tail = 0.0

        # Divergences
        try:
            divs     = trace.sample_stats["diverging"].values
            div_n    = int(divs.sum())
            div_rate = float(div_n / divs.size)
        except Exception:
            div_n    = 0
            div_rate = 0.0

        return {
            "max_rhat"             : 1.0,   # sentinel — not measured
            "mean_rhat"            : 1.0,   # sentinel — not measured
            "max_rhat_is_estimated": True,  # flag so callers can handle
            "ess_bulk"             : ess_bulk,
            "ess_tail"             : ess_tail,
            "divergences"          : div_n,
            "div_rate"             : div_rate,
        }

    # ── Multi-chain path: normal R-hat computation ─────────────
    summary = az.summary(
        trace,
        var_names=["~mu", "~baseline", "~seasonality",
                   "~media_by_channel", "~control_effect"],
        round_to=6,
    )

    # Drop NaN R-hat rows (can occur for deterministic nodes with 0 variance)
    rhat_col = summary["r_hat"].dropna()
    if len(rhat_col) == 0:
        logger.warning("All R-hat values are NaN after dropna — check posterior variables.")
        max_rhat  = 999.0   # sentinel: failure, but sort-safe (not NaN)
        mean_rhat = 999.0
    else:
        max_rhat  = float(rhat_col.max())
        mean_rhat = float(rhat_col.mean())

    ess_bulk = float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns else 0.0
    ess_tail = float(summary["ess_tail"].min()) if "ess_tail" in summary.columns else 0.0

    try:
        divs     = trace.sample_stats["diverging"].values
        div_n    = int(divs.sum())
        div_rate = float(div_n / divs.size)
    except Exception:
        div_n    = 0
        div_rate = 0.0

    return {
        "max_rhat"             : max_rhat,
        "mean_rhat"            : mean_rhat,
        "max_rhat_is_estimated": False,
        "ess_bulk"             : ess_bulk,
        "ess_tail"             : ess_tail,
        "divergences"          : div_n,
        "div_rate"             : div_rate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOO / WAIC
# ─────────────────────────────────────────────────────────────────────────────

def compute_loo_waic(trace: az.InferenceData) -> Dict[str, float]:
    """
    Computes LOO-IC and WAIC.
    Requires log_likelihood to be stored in trace (idata_kwargs={"log_likelihood": True}).
    """
    results = {}
    try:
        loo = az.loo(trace, var_name="y_obs", pointwise=True)
        results["loo_ic"]       = float(loo.elpd_loo)
        results["loo_se"]       = float(loo.se)
        results["p_loo"]        = float(loo.p_loo)
        results["pareto_k_max"] = float(loo.pareto_k.max())
        results["pareto_k_bad"] = int((loo.pareto_k > 0.7).sum())
    except Exception as e:
        logger.warning(f"LOO-IC failed: {e}")
        results.update(dict(loo_ic=np.nan, loo_se=np.nan, p_loo=np.nan,
                            pareto_k_max=np.nan, pareto_k_bad=np.nan))

    try:
        waic = az.waic(trace, var_name="y_obs")
        results["waic"] = float(waic.elpd_waic)
    except Exception as e:
        logger.warning(f"WAIC failed: {e}")
        results["waic"] = np.nan

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Energy diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def compute_energy_diagnostics(trace: az.InferenceData) -> Dict[str, float]:
    """
    Computes energy-based diagnostics for HMC/NUTS quality.

    Returns
    -------
    bfmi_mean    : Bayesian Fraction of Missing Information (>0.3 = good)
    energy_range : Range of energy distribution
    energy_sd    : Standard deviation of energy
    """
    try:
        bfmi = az.bfmi(trace)
        energy = trace.sample_stats["energy"].values.flatten()
        return {
            "bfmi_mean"    : float(bfmi.mean()),
            "bfmi_min"     : float(bfmi.min()),
            "energy_mean"  : float(energy.mean()),
            "energy_std"   : float(energy.std()),
            "energy_range" : float(energy.max() - energy.min()),
            "bfmi_ok"      : float(bfmi.mean()) > 0.3,
            "energy_stable": float(energy.std()) < 3.0,
        }
    except Exception as e:
        logger.warning(f"Energy diagnostics failed: {e}")
        return {
            "bfmi_mean": np.nan, "bfmi_min": np.nan, "bfmi_ok": False,
            "energy_mean": np.nan, "energy_std": np.nan, "energy_range": np.nan,
            "energy_stable": False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# HDI width (media attribution uncertainty)
# ─────────────────────────────────────────────────────────────────────────────

def compute_media_ci_width(
    trace: az.InferenceData,
    C    : int,
) -> Dict[str, float]:
    """
    Computes 90% HDI width of each channel's share of total media contribution.
    Narrow HDI = well-identified channel attribution.
    Handles both P=1 (T, C, N_samples) and P>1 (T, P, C, N_samples) posterior shapes.
    """
    try:
        mbc = (
            trace.posterior["media_by_channel"]
            .stack(sample=("chain", "draw"))
            .values
        )  # P=1: (T, C, N_samples) | P>1: (T, P, C, N_samples)

        if mbc.ndim == 4:
            # P>1: average over product axis (axis=1) → (T, C, N_samples)
            mbc = mbc.mean(axis=1)

        total      = np.abs(mbc).sum(axis=1, keepdims=True) + 1e-12
        share_arr  = np.abs(mbc) / total          # (T, C, N_samples)
        mean_share = share_arr.mean(axis=0)       # (C, N_samples)

        results = {}
        for j in range(C):
            hdi = az.hdi(mean_share[j], hdi_prob=0.90)
            results[f"hdi_width_ch{j}"] = float(hdi[1] - hdi[0])
        results["mean_hdi_width"] = float(
            np.mean([results[f"hdi_width_ch{j}"] for j in range(C)])
        )
        return results
    except Exception as e:
        logger.warning(f"HDI width computation failed: {e}")
        return {f"hdi_width_ch{j}": np.nan for j in range(C)} | {"mean_hdi_width": np.nan}


# ─────────────────────────────────────────────────────────────────────────────
# Master aggregator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_metrics(
    trace       : az.InferenceData,
    prep        : Dict[str, Any],
    wall_time   : float,
    compute_loo : bool = True,
) -> Dict[str, float]:
    """
    Master metrics aggregator.

    Handles P=1 (flat) and P>1 (multi-product) transparently.
    For P>1, fit metrics are averaged across products; per-product breakdowns
    are stored in the returned dict under 'per_product_metrics'.

    Parameters
    ----------
    compute_loo : bool
        Whether to run LOO-IC and WAIC. These are expensive (O(T × N_samples))
        and not needed for Stage 0 / Stage 1b ranking — set False for fast-mode
        passes and True for Stage 2 only.  Defaults to True for back-compat.
    """
    train_idx = prep["train_idx"]
    P         = prep.get("P", 1)
    y_sc  = prep["y_scaled"][train_idx]     # (T,) or (T, P)
    y_rw  = prep["y_raw"][train_idx]        # always (T,) flat single response
    y_mu  = prep["y_mu"]                    # scalar or (P,)
    y_std = prep["y_std"]                   # scalar or (P,)
    C     = prep["C"]
    rt    = prep.get("response_transform", "log1p")
    bl    = prep.get("boxcox_lambda")

    # For P>1, pass per-product raw responses to compute_fit_metrics
    y_raw_by_product = None
    if P > 1:
        # y_sc is (T, P); reconstruct per-product raw responses from y_mu/y_std arrays
        # via the transform inverse.  Use the y_scaled already stored.
        y_raw_by_product = prep.get("y_raw_by_product")  # (T, P) if stored
        if y_raw_by_product is None:
            # Best effort: use the train-indexed y_scaled columns
            y_raw_by_product = y_sc   # pass scaled as proxy; real raw per-product not always stored

    fit = compute_fit_metrics(
        y_sc, y_rw, y_mu, y_std, trace,
        response_transform  = rt,
        boxcox_lambda       = bl,
        product_names       = prep.get("product_names"),
        y_raw_by_product    = y_raw_by_product,
    )
    conv     = compute_convergence_diagnostics(trace)
    adv_conv = compute_advanced_convergence_diagnostics(trace)
    # For multi-product, pass primary product's scaled response (T,) not (T,P)
    y_sc_ppc = y_sc[:, 0] if (P > 1 and np.ndim(y_sc) == 2) else y_sc
    ppc      = compute_posterior_predictive_checks(trace, y_sc_ppc)
    # LOO/WAIC are expensive: skip in fast-mode scans (Stage 0 / Stage 1b)
    if compute_loo:
        loo = compute_loo_waic(trace)
    else:
        loo = {"loo_ic": np.nan, "loo_se": np.nan, "p_loo": np.nan,
               "pareto_k_max": np.nan, "pareto_k_bad": 0, "waic": np.nan}
    hdi    = compute_media_ci_width(trace, C)
    energy = compute_energy_diagnostics(trace)
    tcv    = compute_temporal_cv(y_rw, y_sc, trace, y_mu, y_std,
                                 response_transform=rt, boxcox_lambda=bl)

    # ── OOS holdout metrics ──────────────────────────────────────
    # When holdout_periods > 0 was set, test_idx is the held-out window.
    # We report the holdout size here; full OOS prediction requires posterior
    # predictive sampling on test covariates (see compute_oos_metrics).
    test_idx = prep.get("test_idx", np.array([], dtype=int))
    oos = {"oos_holdout_n": int(len(test_idx))}
    if len(test_idx) > 0:
        oos.update(compute_oos_metrics(trace, prep))

    return {
        **fit,
        **conv,
        **adv_conv,
        **ppc,
        **loo,
        **hdi,
        **energy,
        **tcv,
        **oos,
        "wall_time_s": wall_time,
        # ── dataset-level gates (used by quality_flag) ──
        "T"                           : prep.get("T"),
        "min_T_for_production"        : prep.get("min_T_for_production"),
        "collinearity_max_offdiag_abs": prep.get("collinearity_max_offdiag_abs"),
        "collinearity_threshold"      : prep.get("collinearity_threshold"),
        "trace"      : trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ranking
# ─────────────────────────────────────────────────────────────────────────────

def compute_advanced_convergence_diagnostics(trace: az.InferenceData) -> Dict[str, float]:
    """
    Advanced MCMC convergence diagnostics beyond basic R-hat/ESS.

    Includes energy diagnostics, rank plots, trajectory analysis, and mixing metrics.

    Optimisation: with a single chain (Stage 0 / Stage 1b fast scans), rank-plot
    correlation is meaningless and expensive — skip it entirely via early return
    of a lightweight diagnostics dict.
    """
    diagnostics = {}

    # ── Fast path: 1 chain — skip rank correlation entirely ──────────────────
    if _n_chains(trace) < 2:
        try:
            energy_info = trace.sample_stats.energy.values
            diagnostics['bfmi'] = float(np.mean(az.bfmi(trace)))
            diagnostics['mean_acceptance_rate'] = float(
                trace.sample_stats.acceptance_rate.values.mean()
            )
            diagnostics['divergence_rate'] = float(
                trace.sample_stats.diverging.values.mean()
            )
            if 'tree_depth' in trace.sample_stats:
                diagnostics['mean_tree_depth'] = float(
                    trace.sample_stats.tree_depth.values.mean()
                )
        except Exception as e:
            logger.warning(f"Advanced diagnostics (fast path) failed: {e}")
        return diagnostics

    try:
        # Energy diagnostics (BFMI, energy variance)
        energy_info = trace.sample_stats.energy.values
        diagnostics['bfmi'] = float(np.mean(az.bfmi(trace)))

        # E-BFMI proxy: variance of marginal energy / variance of energy transitions.
        # Both quantities are in energy space — avoids inflating the denominator
        # with time-series variance from trace.posterior.mu.
        energy_variance    = np.var(energy_info, ddof=1)
        energy_transitions = np.diff(energy_info.flatten())
        _edenom = np.var(energy_transitions, ddof=1) if len(energy_transitions) > 1 else energy_variance
        diagnostics['energy_variance_ratio'] = float(energy_variance / _edenom) if _edenom > 0 else float('nan')

        # Rank plot correlation (should be ~0 for good mixing)
        # Only computed for multi-chain traces (already gated above)
        rank_corr = compute_rank_plot_correlation(trace)
        diagnostics['rank_plot_max_correlation'] = float(rank_corr)

        # Local acceptance rate (should be 0.6-0.9 for HMC)
        diagnostics['mean_acceptance_rate'] = float(trace.sample_stats.acceptance_rate.values.mean())

        # Divergence rate per chain
        diagnostics['divergence_rate'] = float(trace.sample_stats.diverging.values.mean())

        # Tree depth statistics (for NUTS)
        if 'tree_depth' in trace.sample_stats:
            diagnostics['mean_tree_depth'] = float(trace.sample_stats.tree_depth.values.mean())
            diagnostics['max_tree_depth'] = float(trace.sample_stats.tree_depth.values.max())

        # Step size adaptation
        if 'step_size' in trace.sample_stats:
            diagnostics['final_step_size'] = float(trace.sample_stats.step_size.values[-1].mean())

    except Exception as e:
        logger.warning(f"Advanced diagnostics computation failed: {e}")
        diagnostics['diagnostic_computation_error'] = 1.0

    return diagnostics


def compute_rank_plot_correlation(trace: az.InferenceData) -> float:
    """Compute maximum rank plot correlation across parameters"""
    max_corr = 0.0

    # FIX: also skip media_by_channel, media_total, and other large (T,)-shaped
    # deterministics — they are time-indexed, not parameter vectors, so their
    # rank correlations are not meaningful convergence diagnostics and are slow.
    _SKIP_VARS = {
        'mu', 'baseline', 'seasonality', 'seasonality_dow',
        'media_by_channel', 'media_total', 'media_by_product',
        'control_effect', 'synergy_effect', 'campaign_effect',
        'halo_effect', 'outlier_effect', 'base_effect', 'macro_effect',
    }
    for var_name in trace.posterior.data_vars:
        if var_name not in _SKIP_VARS:  # Skip high-dimensional deterministic vars
            try:
                ranks = az.rank(trace, var_names=[var_name])
                # Compute correlation between chain ranks
                rank_data = ranks.rank.data
                if rank_data.ndim >= 2:
                    corr_matrix = np.corrcoef(rank_data.reshape(rank_data.shape[0], -1))
                    max_corr = max(max_corr, np.max(np.abs(corr_matrix - np.eye(corr_matrix.shape[0]))))
            except Exception:
                continue

    return max_corr


def compute_posterior_predictive_checks(trace: az.InferenceData, y_obs: np.ndarray, n_ppc: int = 100) -> Dict[str, float]:
    """Compute posterior predictive metrics from a trace and observed response."""
    checks: Dict[str, float] = {}
    y_obs = np.asarray(y_obs)

    try:
        def _stack_posterior(var_name: str) -> np.ndarray:
            return trace.posterior[var_name].stack(sample=("chain", "draw")).values

        mu = _stack_posterior("mu")
        # mu shape after stack: (T, N_samples) for P=1 | (T, P, N_samples) for P>1
        if mu.ndim == 3:
            mu = mu[:, 0, :]   # collapse to primary product -> (T, N_samples)

        # mu is now (T, N_samples); sample n_draws from the N_samples dimension
        N_samples = mu.shape[1] if mu.ndim == 2 else mu.shape[0]
        T_len     = mu.shape[0] if mu.ndim == 2 else 1

        if "sigma_y" in trace.posterior:
            sigma = _stack_posterior("sigma_y")
            if sigma.ndim == 2:
                sigma = sigma[0, :]   # multi-product (P, N_samples) -> primary product
        else:
            sigma = np.ones(N_samples)

        n_draws = min(N_samples, n_ppc)
        indices = np.linspace(0, N_samples - 1, n_draws, dtype=int)
        mu_draws    = mu[:, indices].T          # (n_draws, T)
        sigma_draws = sigma[indices][:, None]   # (n_draws, 1)

        rng = np.random.default_rng(42)
        y_rep = mu_draws + sigma_draws * rng.standard_normal(mu_draws.shape)

        checks["ppc_mean_diff"] = float(np.abs(y_rep.mean() - y_obs.mean()))
        checks["ppc_std_diff"] = float(np.abs(y_rep.std() - y_obs.std()))

        pi_lower = np.percentile(y_rep, 2.5, axis=0)
        pi_upper = np.percentile(y_rep, 97.5, axis=0)
        coverage = np.mean((y_obs >= pi_lower) & (y_obs <= pi_upper))
        checks["predictive_interval_coverage"] = float(coverage)

        try:
            from scipy.stats import ks_2samp

            checks["ks_statistic"] = float(ks_2samp(y_obs.flatten(), y_rep.flatten())[0])
        except Exception:
            checks["ks_statistic"] = np.nan

        bayes_p_mean = np.mean(y_rep.mean(axis=tuple(range(1, y_rep.ndim))) > y_obs.mean())
        checks["bayesian_p_value_mean"] = float(bayes_p_mean)

    except Exception as e:
        logger.warning(f"PPC computation failed: {e}")
        checks["ppc_computation_error"] = 1.0

    return checks


def _nan_safe(val, default):
    """Return val if it is a finite float, else default. Handles NaN and None."""
    try:
        f = float(val)
        return default if f != f else f   # NaN != NaN is True
    except (TypeError, ValueError):
        return default


def quality_flag(m: Dict) -> int:
    """
    3-tier convergence health gate including energy diagnostics.
    0 = Production ready   (R-hat <= 1.01, 0 divergences, ESS >= 400, BFMI > 0.3)
    1 = Acceptable         (R-hat <= 1.03, <= 10 divergences, ESS >= 200, BFMI > 0.2)
    2 = Poor               (anything worse)

    Note: if max_rhat_is_estimated is True (1-chain run) R-hat is not used
    as a disqualifying criterion — ESS and divergences govern quality.
    However, 1-chain runs are never considered "production ready" because
    convergence cannot be verified (so the best possible flag is 1).

    All metric values are passed through _nan_safe() so NaN never reaches
    the <= comparisons (NaN <= threshold is always False in Python, which
    would silently misclassify good models as poor).
    """
    rhat      = _nan_safe(m.get("max_rhat"),      999.0)
    divs      = int(_nan_safe(m.get("divergences"), 999))
    ess       = _nan_safe(m.get("ess_bulk"),        0.0)
    bfmi      = _nan_safe(m.get("bfmi_mean"),       0.0)
    estimated = m.get("max_rhat_is_estimated", False)

    T = m.get("T")
    min_T = m.get("min_T_for_production", 100)
    if T is not None and float(T) < float(min_T):
        return 2

    col_max = m.get("collinearity_max_offdiag_abs")
    col_thr = m.get("collinearity_threshold", 0.85)
    if col_max is not None:
        try:
            if float(col_max) > float(col_thr):
                return 2
        except Exception:
            pass

    if estimated:
        if divs <= 10 and ess >= 200 and bfmi > 0.2:
            return 1
        return 2

    if rhat <= 1.01 and divs == 0 and ess >= 400 and bfmi > 0.3:
        return 0
    if rhat <= 1.03 and divs <= 10 and ess >= 200 and bfmi > 0.2:
        return 1
    return 2


def _loo_key(m: Dict) -> float:
    """
    Returns a sort-safe LOO key: -loo_ic when available, else 0.0 (neutral).

    NaN is produced when compute_loo=False (Stage 0 / Stage 1b fast scans).
    Python's tuple comparison treats NaN as neither equal nor less-than any
    value, which makes min()/sort() non-deterministic and causes the FIRST
    combo in ADSTOCK_SAT_COMBOS (geometric+softplus) to win regardless of
    actual MAPE.  Using 0.0 instead makes all NaN-LOO models tie at this
    position so MAPE becomes the real discriminator, which is what Stage 0
    is designed to use.
    """
    loo = m.get("loo_ic", float("nan"))
    if loo != loo:   # NaN check: NaN != NaN is True in Python
        return 0.0   # neutral — let MAPE decide
    return -float(loo)


def rank_tuple(m: Dict) -> Tuple:
    """
    Lexicographic ranking tuple. Used to sort models (lower = better).

    When max_rhat_is_estimated=True (1-chain fast scan), R-hat is a sentinel
    and cannot discriminate between models.  MAPE becomes the primary
    discriminator; LOO-IC is used when available (Stage 2), neutral (0.0)
    when not computed (Stage 0 / Stage 1b).
    """
    estimated = m.get("max_rhat_is_estimated", False)

    if estimated:
        return (
            quality_flag(m),
            int(_nan_safe(m.get("divergences"),     999)),
            _loo_key(m),
            _nan_safe(m.get("mape"),                999.0),
            -_nan_safe(m.get("ess_bulk"),             0.0),
            -_nan_safe(m.get("bfmi_mean"),            0.0),
            _nan_safe(m.get("mean_hdi_width"),      999.0),
        )

    # Multi-chain path: full diagnostics available
    return (
        quality_flag(m),
        _nan_safe(m.get("max_rhat"),            999.0),
        int(_nan_safe(m.get("divergences"),     999)),
        -_nan_safe(m.get("ess_bulk"),             0.0),
        -_nan_safe(m.get("bfmi_mean"),            0.0),
        _nan_safe(m.get("mape"),                999.0),
        _nan_safe(m.get("mean_hdi_width"),      999.0),
        _loo_key(m),
    )


def model_rank_key(
    m             : Dict,
    ranking_method: str = "lexicographic",
    rank_weights  : Optional[Dict[str, float]] = None,
) -> Tuple:
    method = (ranking_method or "lexicographic").strip().lower()

    if method == "lexicographic":
        return rank_tuple(m)

    if method == "mape_first":
        return (
            _nan_safe(m.get("mape"),            999.0),
            quality_flag(m),
            _nan_safe(m.get("max_rhat"),        999.0),
            int(_nan_safe(m.get("divergences"), 999)),
            _loo_key(m),
            _nan_safe(m.get("mean_hdi_width"),  999.0),
        )

    # weighted score (lower is better)
    w = {"quality": 12.0, "mape": 2.5, "rhat": 3.0, "divergences": 2.0,
         "hdi": 1.0, "loo": 1.0, "bfmi": 1.0}
    if rank_weights:
        for k, v in rank_weights.items():
            if k in w:
                w[k] = float(v)

    # All metrics passed through _nan_safe so NaN never enters the arithmetic.
    # A NaN metric becomes its failure sentinel (e.g. 999.0 for rhat)
    # so the score is always a finite float and sorts correctly.
    score = (
        w["quality"]     * quality_flag(m) +
        w["mape"]        * _nan_safe(m.get("mape"),          999.0) +
        w["rhat"]        * _nan_safe(m.get("max_rhat"),       999.0) * 100.0 +
        w["divergences"] * _nan_safe(m.get("divergences"),    999.0) +
        w["hdi"]         * _nan_safe(m.get("mean_hdi_width"), 999.0) +
        w["loo"]         * _loo_key(m) +
        w["bfmi"]        * (-_nan_safe(m.get("bfmi_mean"),      0.0))
    )
    return (score,)


# ─────────────────────────────────────────────────────────────────────────────
# Result row builder
# ─────────────────────────────────────────────────────────────────────────────

def build_result_row(
    cfg    : ModelConfig,
    metrics: Dict,
    stage  : str,
    rank   : int,
    C      : int,
) -> Dict:
    """Flattens config + metrics into a single dict for DataFrame construction."""
    row = {
        "rank"           : rank,
        "stage"          : stage,
        "model_key"      : cfg.key(),
        "adstock_type"   : cfg.adstock_type,
        "saturation"     : cfg.saturation,
        "max_lag"        : cfg.max_lag,
        "fourier_order"  : cfg.fourier_order,
        "target_accept"  : cfg.target_accept,
        "draws"          : cfg.draws,
        "chains"         : cfg.chains,
        "quality_flag"   : quality_flag(metrics),
        "rhat_estimated" : metrics.get("max_rhat_is_estimated", False),
        "mape"           : round(metrics.get("mape",          np.nan), 4),
        "rmse"           : round(metrics.get("rmse",          np.nan), 4),
        "r2"             : round(metrics.get("r2",            np.nan), 4),
        "pearson_r"      : round(metrics.get("pearson_r",     np.nan), 4),
        "max_rhat"       : round(metrics.get("max_rhat",      np.nan), 5),
        "mean_rhat"      : round(metrics.get("mean_rhat",     np.nan), 5),
        "ess_bulk"       : round(metrics.get("ess_bulk",      np.nan), 1),
        "ess_tail"       : round(metrics.get("ess_tail",      np.nan), 1),
        "divergences"    : metrics.get("divergences",         np.nan),
        "div_rate"       : round(metrics.get("div_rate",      np.nan), 6),
        "loo_ic"         : round(metrics.get("loo_ic",        np.nan), 4),
        "loo_se"         : round(metrics.get("loo_se",        np.nan), 4),
        "waic"           : round(metrics.get("waic",          np.nan), 4),
        "pareto_k_max"   : round(metrics.get("pareto_k_max",  np.nan), 4),
        "pareto_k_bad"   : metrics.get("pareto_k_bad",        np.nan),
        "mean_hdi_width" : round(metrics.get("mean_hdi_width",np.nan), 4),
        "wall_time_s"    : round(metrics.get("wall_time_s",   np.nan), 1),
        "mape_std"       : round(metrics.get("mape_std",      np.nan), 4),
        "mape_hdi_low"   : round(metrics.get("mape_hdi_low",  np.nan), 4),
        "mape_hdi_high"  : round(metrics.get("mape_hdi_high", np.nan), 4),
        "r2_std"         : round(metrics.get("r2_std",        np.nan), 4),
        "r2_hdi_low"     : round(metrics.get("r2_hdi_low",    np.nan), 4),
        "r2_hdi_high"    : round(metrics.get("r2_hdi_high",   np.nan), 4),
        "bfmi_mean"      : round(metrics.get("bfmi_mean",     np.nan), 4),
        "bfmi_min"       : round(metrics.get("bfmi_min",      np.nan), 4),
        "energy_std"     : round(metrics.get("energy_std",    np.nan), 4),
        "bfmi_ok"        : metrics.get("bfmi_ok",             False),
        "energy_stable"  : metrics.get("energy_stable",       False),
    }
    for j in range(C):
        row[f"hdi_width_ch{j}"] = round(metrics.get(f"hdi_width_ch{j}", np.nan), 4)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# OOS holdout evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_oos_metrics(
    trace : "az.InferenceData",
    prep  : Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute out-of-sample metrics on the held-out test window.

    Uses the posterior median of mu (in scaled space) extrapolated to test
    time steps via the seasonality + trend components from the trace.
    Media contributions at test time are approximated from the posterior
    channel betas applied to test media data.

    Returns oos_mape, oos_r2, oos_rmse.
    """
    test_idx  = prep.get("test_idx", np.array([], dtype=int))
    if len(test_idx) == 0:
        return {}

    try:
        P        = prep.get("P", 1)
        y_mu     = prep["y_mu"]
        y_std    = prep["y_std"]
        rt       = prep.get("response_transform", "log1p")
        bl       = prep.get("boxcox_lambda")

        # Posterior samples of mu at training time steps: (chain, draw, T_train[, P])
        mu_post = trace.posterior["mu"].values

        # Build posterior median prediction in scaled space at test time using
        # the ratio of seasonal + trend variance: approximate by projecting
        # the posterior mean mu scaled residuals to test.
        # More precisely: use posterior samples of seasonality + trend if available,
        # else fall back to posterior mean of mu at nearest training boundary.

        # Try to use seasonality + trend posteriors for test-period prediction
        has_season = "seasonality" in trace.posterior.data_vars
        has_trend  = any(v in trace.posterior.data_vars
                         for v in ("trend", "baseline", "linear_trend"))

        if has_season:
            from data_prep import build_fourier_features
            period     = float(prep.get("fourier_period", 52))
            fo         = prep["X_fourier"].shape[1] // 2
            T_full     = prep["T"]
            X_f_full   = build_fourier_features(T_full, period, fo)
            X_f_test   = X_f_full[test_idx]

            # delta: posterior (chain, draw, 2*fo)
            delta_post = trace.posterior["delta"].values
            n_chains, n_draws = delta_post.shape[:2]
            delta_flat = delta_post.reshape(-1, delta_post.shape[-1])  # (S, 2*fo)
            season_test = X_f_test @ delta_flat.T  # (T_test, S)

            # trend at test: t_norm for test indices
            t_norm_full = prep["t_norm"]
            t_test      = t_norm_full[test_idx]

            # Try linear trend coefficients
            trend_test = np.zeros((len(test_idx), delta_flat.shape[0]))
            for vname in ("trend_slope", "alpha_trend", "slope"):
                if vname in trace.posterior.data_vars:
                    sl = trace.posterior[vname].values.reshape(-1)
                    for vname2 in ("trend_intercept", "beta_trend", "intercept"):
                        if vname2 in trace.posterior.data_vars:
                            ic = trace.posterior[vname2].values.reshape(-1)
                            trend_test = t_test[:, None] * sl[None, :] + ic[None, :]
                            break
                    break

            # Media contribution at test time: betas × adstock(X_media_test)
            # Approximate: use posterior mean betas × mean test media (scaled)
            X_media_test = prep["X_media_scaled"][test_idx]  # (T_test, C) or (T_test, P, C)
            if X_media_test.ndim == 3:
                X_media_test = X_media_test.mean(axis=1)  # avg over P
            if "media_by_channel" in trace.posterior.data_vars:
                mbc = trace.posterior["media_by_channel"].values  # (chain, draw, T_train, [P,] C)
                if mbc.ndim == 5:
                    mbc = mbc.mean(axis=3)  # avg over P
                mbc_mean_per_t = mbc.mean(axis=(0, 1))  # (T_train, C)
                # Scale media contribution at test by the mean media level ratio
                train_media_mean = prep["X_media_scaled"][prep["train_idx"]].mean(axis=0)
                if train_media_mean.ndim == 2:
                    train_media_mean = train_media_mean.mean(axis=0)
                test_media_ratio = X_media_test / (train_media_mean[None, :] + 1e-12)
                # Mean channel contribution at test = global mean × ratio
                media_test_mean = (mbc_mean_per_t.mean(axis=0)[None, :] * test_media_ratio)
                media_test_total = media_test_mean.sum(axis=1)  # (T_test,)
            else:
                media_test_total = np.zeros(len(test_idx))

            # Approximate mu at test: season + trend + media
            mu_test_samples = season_test + trend_test + media_test_total[:, None]  # (T_test, S)
            mu_test_median  = np.median(mu_test_samples, axis=1)  # (T_test,)
        else:
            # Fallback: use last training mu posterior median as constant
            if mu_post.ndim == 4:
                mu_post_2d = mu_post[:, :, :, 0]
            else:
                mu_post_2d = mu_post
            mu_test_median = np.median(mu_post_2d[:, :, -1])  * np.ones(len(test_idx))

        # Un-standardize and invert transform
        if P > 1:
            y_std_0 = float(np.asarray(y_std).flat[0])
            y_mu_0  = float(np.asarray(y_mu).flat[0])
        else:
            y_std_0 = float(y_std)
            y_mu_0  = float(y_mu)

        mu_log  = mu_test_median * y_std_0 + y_mu_0
        if rt == "log1p":
            y_hat = np.expm1(mu_log)
        elif rt == "boxcox" and bl is not None:
            from scipy.special import inv_boxcox
            y_hat = inv_boxcox(mu_log, bl)
        else:
            y_hat = mu_log

        y_hat = np.maximum(y_hat, 0.0)
        y_true = prep["y_raw"][test_idx]

        mask = y_true > 1.0
        if mask.sum() < 3:
            return {"oos_holdout_n": len(test_idx)}

        oos_mape = float(np.mean(np.abs(y_true[mask] - y_hat[mask]) / y_true[mask])) * 100.0
        ss_res   = np.sum((y_true[mask] - y_hat[mask]) ** 2)
        ss_tot   = np.sum((y_true[mask] - y_true[mask].mean()) ** 2)
        oos_r2   = float(1.0 - ss_res / (ss_tot + 1e-12))
        oos_rmse = float(np.sqrt(np.mean((y_true[mask] - y_hat[mask]) ** 2)))

        logger.info(
            f"  OOS holdout ({len(test_idx)} periods): "
            f"MAPE={oos_mape:.2f}% | R2={oos_r2:.4f} | RMSE={oos_rmse:.4f}"
        )
        return {
            "oos_holdout_n"   : len(test_idx),
            "oos_mape"        : round(oos_mape, 4),
            "oos_r2"          : round(oos_r2,   4),
            "oos_rmse"        : round(oos_rmse,  4),
        }

    except Exception as e:
        logger.warning(f"  OOS holdout metrics failed: {e}")
        return {"oos_holdout_n": len(test_idx)}


# ─────────────────────────────────────────────────────────────────────────────
# Temporal cross-validation — alternative to LOO-IC
# ─────────────────────────────────────────────────────────────────────────────

def compute_temporal_cv(
    y_raw     : np.ndarray,
    y_scaled  : np.ndarray,
    trace     : "az.InferenceData",
    y_mu      : float,
    y_std     : float,
    n_folds   : int = 3,
    min_train : int = 26,
    response_transform : str = "log1p",
    boxcox_lambda      : float = None,
) -> Dict[str, float]:
    """
    Time-series cross-validation using expanding window.

    Unlike LOO-IC (which assumes exchangeable observations), this respects
    the temporal order of the data.  Each fold trains on all data up to
    a cutpoint and evaluates on the next segment.

    Since we already have a fitted posterior, we use the posterior predictive
    for each fold's test period (approximate — the model was fit on all data,
    so this is pseudo-OOS, not true OOS).  For true OOS, you'd need to refit
    per fold, which is very expensive.

    This provides a complementary signal to LOO-IC: if LOO says the model is
    good but temporal CV shows poor forward prediction, the model may be
    overfitting to noise patterns.

    Parameters
    ----------
    y_raw     : original-scale response (T,)
    y_scaled  : z-scored log response (T,)
    trace     : ArviZ InferenceData with posterior "mu"
    y_mu, y_std : scaling parameters
    n_folds   : number of temporal folds (default 3)
    min_train : minimum training observations per fold

    Returns
    -------
    dict with:
        temporal_cv_mape    : mean MAPE across folds
        temporal_cv_rmse    : mean RMSE across folds
        temporal_cv_mapes   : list of per-fold MAPEs
        temporal_cv_n_folds : actual number of folds used
    """
    T = len(y_raw)
    if T < min_train + n_folds:
        logger.warning(
            f"  Temporal CV: T={T} too short for {n_folds} folds "
            f"(need >= {min_train + n_folds}). Skipping."
        )
        return {
            "temporal_cv_mape"  : np.nan,
            "temporal_cv_rmse"  : np.nan,
            "temporal_cv_mapes" : [],
            "temporal_cv_n_folds": 0,
        }

    try:
        # Get posterior mean of mu: (T,)
        mu_post = (
            trace.posterior["mu"]
            .stack(sample=("chain", "draw"))
            .values
        )   # (T, N_samples) for P=1 | (T, P, N_samples) for P>1

        if mu_post.ndim == 3:
            # Multi-product: collapse to primary product (index 0) for CV
            y_std_0 = float(np.asarray(y_std).flat[0])
            y_mu_0  = float(np.asarray(y_mu).flat[0])
            mu_post = mu_post[:, 0, :]          # (T, N_samples)
        else:
            y_std_0 = float(y_std)
            y_mu_0  = float(y_mu)

        mu_log  = mu_post * y_std_0 + y_mu_0   # (T, N_samples) in transformed space
        y_hat_s = _inv_transform(mu_log, response_transform, boxcox_lambda)  # original scale
        y_hat_mean = y_hat_s.mean(axis=1)       # (T,) posterior mean prediction
    except Exception as e:
        logger.warning(f"  Temporal CV failed to extract posterior: {e}")
        return {
            "temporal_cv_mape": np.nan, "temporal_cv_rmse": np.nan,
            "temporal_cv_mapes": [], "temporal_cv_n_folds": 0,
        }

    # Expanding window folds
    test_size = max(1, (T - min_train) // n_folds)
    fold_mapes = []
    fold_rmses = []

    for fold in range(n_folds):
        test_start = min_train + fold * test_size
        test_end   = min(test_start + test_size, T)
        if test_start >= T:
            break

        test_idx = np.arange(test_start, test_end)
        y_true   = y_raw[test_idx]
        y_pred   = y_hat_mean[test_idx]

        mask = y_true > 1.0
        if mask.sum() == 0:
            continue

        mape = float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-12))) * 100.0
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        fold_mapes.append(mape)
        fold_rmses.append(rmse)

    if not fold_mapes:
        return {
            "temporal_cv_mape": np.nan, "temporal_cv_rmse": np.nan,
            "temporal_cv_mapes": [], "temporal_cv_n_folds": 0,
        }

    result = {
        "temporal_cv_mape"  : round(float(np.mean(fold_mapes)), 4),
        "temporal_cv_rmse"  : round(float(np.mean(fold_rmses)), 4),
        "temporal_cv_mapes" : [round(m, 4) for m in fold_mapes],
        "temporal_cv_n_folds": len(fold_mapes),
    }
    logger.info(
        f"  Temporal CV: {len(fold_mapes)} folds | "
        f"mean MAPE={result['temporal_cv_mape']:.2f}% | "
        f"mean RMSE={result['temporal_cv_rmse']:.2f}"
    )
    return result
