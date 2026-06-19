# priors.py — Per-channel prior configuration and domain prior library

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
import pytensor.tensor as pt

logger = logging.getLogger("MMM")


# ─────────────────────────────────────────────────────────────────────────────
# Supported distributions — validation sets
# ─────────────────────────────────────────────────────────────────────────────

VALID_BETA_DISTS = {"half_normal", "exponential", "gamma", "log_normal"}
VALID_LIKELIHOODS = {"student_t", "normal", "skew_normal", "gamma_obs", "negative_binomial"}


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel prior configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChannelPriorConfig:
    """
    Specifies the prior distribution for a single channel's beta (contribution
    weight) and the observation likelihood family to use.

    Parameters
    ----------
    beta_dist : str
        Prior distribution for the channel contribution weight (beta).
        Must be one of: "half_normal", "exponential", "gamma", "log_normal".
        Default: "half_normal"

    beta_params : dict
        Hyperparameters for the chosen beta_dist:
          half_normal  -> {"sigma": float}        default: {"sigma": 0.3}
          exponential  -> {"lam": float}           default: {"lam": 3.0}
          gamma        -> {"alpha": float,
                          "beta":  float}         default: {"alpha": 2.0, "beta": 2.0}
          log_normal   -> {"mu":    float,
                          "sigma": float}         default: {"mu": -1.0, "sigma": 0.5}

    likelihood : str
        Observation likelihood family. Must be one of:
          "student_t"         — robust to outlier observations (default)
          "normal"            — assumes clean, Gaussian residuals
          "skew_normal"       — for asymmetric response distributions
          "gamma_obs"         — for strictly positive continuous KPIs (revenue, sales)
          "negative_binomial" — for non-negative integer count KPIs (clicks, orders)

    likelihood_params : dict
        Extra hyperparameters for the chosen likelihood:
          student_t   -> {} (nu and sigma are latent — learned from data)
          normal      -> {} (sigma is latent)
          skew_normal -> {"alpha_skew": float}  skewness parameter
                        default: {"alpha_skew": 0.0}  (symmetric)

    notes : str
        Optional free-text annotation for documentation/export.
    """
    beta_dist         : str            = "half_normal"
    beta_params       : Dict[str, Any] = field(default_factory=lambda: {"sigma": 0.3})
    likelihood        : str            = "student_t"
    likelihood_params : Dict[str, Any] = field(default_factory=dict)
    notes             : str            = ""
    # FIX-1: Optional adstock / saturation prior overrides (None = use hardcoded defaults)
    adstock_lam_prior : Optional[Dict[str, Any]] = None
    sat_alpha_prior   : Optional[Dict[str, Any]] = None
    sat_kappa_prior   : Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        # -- Validate beta_dist -----------------------------------------------
        if self.beta_dist not in VALID_BETA_DISTS:
            raise ValueError(
                f"beta_dist '{self.beta_dist}' is not supported. "
                f"Choose from: {sorted(VALID_BETA_DISTS)}"
            )
        # -- Validate likelihood -----------------------------------------------
        if self.likelihood not in VALID_LIKELIHOODS:
            raise ValueError(
                f"likelihood '{self.likelihood}' is not supported. "
                f"Choose from: {sorted(VALID_LIKELIHOODS)}"
            )
        # -- Fill missing beta_params defaults ---------------------------------
        _defaults = {
            "half_normal" : {"sigma": 0.3},
            "exponential" : {"lam": 3.0},
            "gamma"       : {"alpha": 2.0, "beta": 2.0},
            "log_normal"  : {"mu": -1.0, "sigma": 0.5},
        }
        for k, v in _defaults[self.beta_dist].items():
            self.beta_params.setdefault(k, v)

        # -- Validate beta_params keys -----------------------------------------
        expected_keys = set(_defaults[self.beta_dist].keys())
        provided_keys = set(self.beta_params.keys())
        invalid_keys  = provided_keys - expected_keys
        if invalid_keys:
            raise ValueError(
                f"beta_params contains unexpected key(s) {sorted(invalid_keys)} "
                f"for beta_dist='{self.beta_dist}'. "
                f"Expected keys: {sorted(expected_keys)}. "
                f"Received: {dict(self.beta_params)}"
            )

    def describe(self) -> str:
        """Short human-readable description for logging."""
        return (
            f"beta={self.beta_dist}({self.beta_params}) "
            f"likelihood={self.likelihood}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Default config (used when no per-channel config is provided)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PRIOR_CONFIG = ChannelPriorConfig(
    beta_dist     = "half_normal",
    beta_params   = {"sigma": 0.3},
    likelihood    = "student_t",
    notes         = "global default",
)


# ─────────────────────────────────────────────────────────────────────────────
# Registry lookup — get config for a channel, falling back to default
# ─────────────────────────────────────────────────────────────────────────────

def get_prior_config(
    channel_name       : str,
    channel_prior_map  : Optional[Dict[str, ChannelPriorConfig]],
) -> ChannelPriorConfig:
    """
    Returns the ChannelPriorConfig for a channel.
    Falls back to DEFAULT_PRIOR_CONFIG if the channel is not in the map.
    """
    if channel_prior_map and channel_name in channel_prior_map:
        return channel_prior_map[channel_name]
    return DEFAULT_PRIOR_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# PyMC prior builder — called inside build_mmm() per channel
# ─────────────────────────────────────────────────────────────────────────────

def build_beta_prior(
    name   : str,
    config : ChannelPriorConfig,
) -> Any:
    """
    Builds a PyMC random variable for the channel beta using the specified
    prior distribution.
    """
    p = config.beta_params

    if config.beta_dist == "half_normal":
        sigma = float(p.get("sigma", 0.3))
        _validate_positive(sigma, "half_normal.sigma")
        return pm.HalfNormal(name, sigma=sigma)

    elif config.beta_dist == "exponential":
        lam = float(p.get("lam", 3.0))
        _validate_positive(lam, "exponential.lam")
        return pm.Exponential(name, lam=lam)

    elif config.beta_dist == "gamma":
        alpha = float(p.get("alpha", 2.0))
        beta  = float(p.get("beta",  2.0))
        _validate_positive(alpha, "gamma.alpha")
        _validate_positive(beta,  "gamma.beta")
        return pm.Gamma(name, alpha=alpha, beta=beta)

    elif config.beta_dist == "log_normal":
        mu    = float(p.get("mu",    -1.0))
        sigma = float(p.get("sigma",  0.5))
        _validate_positive(sigma, "log_normal.sigma")
        return pm.LogNormal(name, mu=mu, sigma=sigma)

    else:
        raise ValueError(f"Unknown beta_dist: {config.beta_dist}")


def build_likelihood(
    name         : str,
    mu           : Any,
    sigma_y      : Any,
    nu           : Any,
    observed     : np.ndarray,
    config       : ChannelPriorConfig,
    y_raw_obs    : Optional[np.ndarray] = None,
    y_mu_val     : Optional[float]      = None,
    y_std_val    : Optional[float]      = None,
) -> None:
    """
    Registers the observation likelihood in the active PyMC model context.
    """
    lp = config.likelihood_params

    if config.likelihood == "student_t":
        pm.StudentT(name, nu=nu, mu=mu, sigma=sigma_y, observed=observed)

    elif config.likelihood == "normal":
        pm.Normal(name, mu=mu, sigma=sigma_y, observed=observed)

    elif config.likelihood == "skew_normal":
        alpha_skew = float(lp.get("alpha_skew", 0.0))
        pm.SkewNormal(name, mu=mu, sigma=sigma_y, alpha=alpha_skew, observed=observed)

    elif config.likelihood == "gamma_obs":
        if y_mu_val is None or y_std_val is None:
            raise ValueError(
                "gamma_obs likelihood requires y_mu_val and y_std_val for back-transformation. "
                "Pass prep['y_mu'] and prep['y_std'] to build_likelihood."
            )
        if y_raw_obs is None:
            raise ValueError(
                "gamma_obs likelihood requires y_raw_obs (original-scale observations). "
                "Pass prep['y_raw'][train_idx] to build_likelihood."
            )
        mu_log1p  = mu * float(y_std_val) + float(y_mu_val)
        mu_raw    = pm.math.exp(mu_log1p) - 1.0
        mu_pos    = pm.math.softplus(mu_raw) + 1e-6
        sigma_pos = pm.math.softplus(sigma_y * float(y_std_val)) + 1e-6
        pm.Gamma(name, mu=mu_pos, sigma=sigma_pos,
                 observed=np.maximum(y_raw_obs, 1e-6))

    elif config.likelihood == "negative_binomial":
        if y_mu_val is None or y_std_val is None:
            raise ValueError(
                "negative_binomial likelihood requires y_mu_val and y_std_val."
            )
        if y_raw_obs is None:
            raise ValueError(
                "negative_binomial likelihood requires y_raw_obs (integer count observations)."
            )
        mu_log1p   = mu * float(y_std_val) + float(y_mu_val)
        mu_count   = pm.math.softplus(pm.math.exp(mu_log1p) - 1.0) + 1e-6
        alpha_nb   = pm.HalfStudentT(f"{name}_nb_alpha", nu=4, sigma=5.0)
        pm.NegativeBinomial(name, mu=mu_count, alpha=alpha_nb,
                            observed=np.maximum(np.round(y_raw_obs).astype(int), 0))

    else:
        raise ValueError(f"Unknown likelihood: {config.likelihood}")


# ─────────────────────────────────────────────────────────────────────────────
# Input validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(
            f"Prior hyperparameter '{name}' must be positive, got {value}."
        )


def validate_channel_prior_map(
    channel_prior_map : Dict[str, ChannelPriorConfig],
    spend_cols        : list,
) -> None:
    """
    Validates that every channel listed in the prior map exists in spend_cols,
    and that every ChannelPriorConfig is internally valid.
    """
    if not channel_prior_map:
        return

    spend_set = set(spend_cols)
    unknown   = [ch for ch in channel_prior_map if ch not in spend_set]
    if unknown:
        raise ValueError(
            f"channel_prior_map references channels not in spend_cols: {unknown}\n"
            f"Valid channels: {sorted(spend_set)}"
        )

    for ch, cfg in channel_prior_map.items():
        if not isinstance(cfg, ChannelPriorConfig):
            raise TypeError(
                f"channel_prior_map['{ch}'] must be a ChannelPriorConfig, "
                f"got {type(cfg).__name__}"
            )
        try:
            cfg.__post_init__()
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid config for channel '{ch}': {e}") from e

    logger.info(
        f"[PRIOR-CFG] Validated {len(channel_prior_map)} per-channel prior configs. "
        f"Channels using defaults: "
        f"{sorted(spend_set - set(channel_prior_map.keys()))}"
    )


def _check_hierarchical_compat(
    cfg               : Any,
    channel_prior_map : Optional[Dict[str, "ChannelPriorConfig"]],
) -> None:
    """
    Raises ValueError if use_hierarchical=True and any channel has a non-default
    beta_dist in channel_prior_map.
    """
    if not (getattr(cfg, "use_hierarchical", False) and channel_prior_map):
        return

    non_default = [
        ch for ch, pcfg in channel_prior_map.items()
        if pcfg.beta_dist != "half_normal"
    ]
    if non_default:
        raise ValueError(
            f"use_hierarchical=True overrides per-channel beta priors, but "
            f"{len(non_default)} channel(s) have custom beta_dist settings that "
            f"would be silently ignored: {non_default}.\n"
            f"Options:\n"
            f"  * Set use_hierarchical=False to use per-channel beta priors, OR\n"
            f"  * Remove custom beta_dist overrides for the listed channels."
        )

    adstock_set = [
        ch for ch, pcfg in channel_prior_map.items()
        if pcfg.adstock_lam_prior
    ]
    if adstock_set:
        use_two_ts = bool(getattr(cfg, "use_two_timescale_adstock", False))
        case_label = "A (two-timescale)" if use_two_ts else "B (single-timescale)"
        logger.warning(
            "[MODEL_BUILDER] use_hierarchical=True (Case %s): adstock_lam_prior "
            "settings for %d channel(s) are IGNORED: %s. "
            "In hierarchical mode, adstock decay is controlled by the family-level "
            "hl_slow_median hyperprior (priors.families in YAML), not per-channel "
            "adstock_lam_prior. To use per-channel adstock priors, set "
            "use_hierarchical=False.",
            case_label, len(adstock_set), adstock_set,
        )


# ─────────────────────────────────────────────────────────────────────────────
# YAML prior override system
# ─────────────────────────────────────────────────────────────────────────────

def apply_yaml_prior_overrides(
    channel_prior_map : Dict[str, "ChannelPriorConfig"],
    cfg_dict          : Dict[str, Any],
    schema,
) -> Dict[str, "ChannelPriorConfig"]:
    """
    Apply the 3-layer YAML prior override system onto channel_prior_map.

    Layers (lowest -> highest precedence):
      Layer 3  priors.families   — family-level beta/adstock overrides
      Layer 2  channel_priors    — standalone per-column section
      Layer 1  priors.channels   — per-channel overrides (wins over all)
    """
    _is_multi   = getattr(schema, "is_multi_product", False)
    prior_sec   = cfg_dict.get("priors", {}) or {}

    def _merge(base: "ChannelPriorConfig", ov: dict) -> "ChannelPriorConfig":
        beta_dist   = ov.get("beta_dist", base.beta_dist)
        beta_params = dict(base.beta_params)
        if "beta_sigma" in ov:
            beta_params = {"sigma": float(ov["beta_sigma"])}
            beta_dist   = "half_normal"
        if "beta_params" in ov and isinstance(ov["beta_params"], dict):
            beta_params = ov["beta_params"]
        adstock = dict(base.adstock_lam_prior) if base.adstock_lam_prior else {}
        if "adstock_alpha" in ov:
            adstock["alpha"] = float(ov["adstock_alpha"])
        if "adstock_beta"  in ov:
            adstock["beta"]  = float(ov["adstock_beta"])
        if ("adstock_alpha" in ov or "adstock_beta" in ov) and "dist" not in adstock:
            adstock["dist"] = "beta"
        if "adstock_lam_prior" in ov and isinstance(ov["adstock_lam_prior"], dict):
            adstock = ov["adstock_lam_prior"]
        return ChannelPriorConfig(
            beta_dist         = beta_dist,
            beta_params       = beta_params,
            likelihood        = ov.get("likelihood", base.likelihood),
            adstock_lam_prior = adstock or None,
            sat_alpha_prior   = ov.get("sat_alpha_prior", base.sat_alpha_prior),
            sat_kappa_prior   = ov.get("sat_kappa_prior", base.sat_kappa_prior),
            notes             = ov.get("notes", base.notes),
        )

    # -- Layer 3: priors.families ----------------------------------------------
    _fam_sec = prior_sec.get("families", {}) or {}
    _chan_to_family = {
        (mc.channel if _is_multi else mc.column): mc.family
        for mc in schema.media_cols
    }
    _FAM_BETA_FIELDS = {
        "beta_sigma", "adstock_alpha", "adstock_beta", "likelihood",
        "beta_dist", "beta_params", "adstock_lam_prior",
        "sat_alpha_prior", "sat_kappa_prior",
    }
    for fname, fcfg in _fam_sec.items():
        if not isinstance(fcfg, dict) or not any(k in fcfg for k in _FAM_BETA_FIELDS):
            continue
        for ch_key, fam in _chan_to_family.items():
            if fam != fname:
                continue
            try:
                channel_prior_map[ch_key] = _merge(
                    channel_prior_map.get(ch_key, ChannelPriorConfig()),
                    {**fcfg, "notes": f"family={fname} via priors.families"},
                )
            except Exception as e:
                logger.warning(f"[PRIORS] family override for '{ch_key}' failed: {e}")

    # -- Layer 2: channel_priors (standalone section) --------------------------
    for ch_col, ov in (cfg_dict.get("channel_priors", {}) or {}).items():
        if not isinstance(ov, dict):
            continue
        try:
            channel_prior_map[ch_col] = _merge(
                channel_prior_map.get(ch_col, ChannelPriorConfig()),
                {**ov, "notes": "channel_priors section"},
            )
        except Exception as e:
            logger.warning(f"[PRIORS] channel_priors override for '{ch_col}' failed: {e}")

    # -- Layer 1: priors.channels (highest precedence) -------------------------
    _col_to_channel = {mc.column: mc.channel for mc in schema.media_cols} if _is_multi else {}
    _applied: set   = set()
    for ch_col, ov in (prior_sec.get("channels", {}) or {}).items():
        if not isinstance(ov, dict):
            continue
        key = _col_to_channel.get(ch_col, ch_col) if _is_multi else ch_col
        if key in _applied:
            continue
        _applied.add(key)
        try:
            channel_prior_map[key] = _merge(
                channel_prior_map.get(key, ChannelPriorConfig()),
                {**ov, "notes": "priors.channels"},
            )
        except Exception as e:
            logger.warning(f"[PRIORS] priors.channels override for '{key}' failed: {e}")

    return channel_prior_map


# ─────────────────────────────────────────────────────────────────────────────
# Return index computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_return_index(
    best              : Dict[str, Any],
    prep              : Dict[str, Any],
    credible_interval : float = 0.95,
) -> pd.DataFrame:
    """
    Computes the Return Index for every channel from the posterior trace.
    """
    if not (0 < credible_interval < 1):
        raise ValueError(
            f"credible_interval must be between 0 and 1, got {credible_interval}"
        )

    trace      = best["metrics"]["trace"]
    spend_cols = prep["spend_cols"]
    C          = prep["C"]
    y_std      = prep["y_std"]
    train_idx  = prep["train_idx"]
    X_media_rw = prep["X_media_raw"][train_idx]
    if X_media_rw.ndim == 3:
        X_media_rw = X_media_rw.mean(axis=1)
    y_std = float(np.asarray(y_std).mean())

    metric_types   = prep.get("metric_types",   ["Spend"] * C)
    spend_raw_cols = prep.get("spend_raw_cols", spend_cols)
    df_full        = prep.get("df")

    assert len(spend_cols) == C
    assert X_media_rw.shape == (len(train_idx), C)
    assert len(metric_types) == C
    assert len(spend_raw_cols) == C

    mbc_post = trace.posterior["media_by_channel"]
    mbc      = (
        mbc_post
        .stack(sample=("chain", "draw"))
        .transpose("sample", ...)
        .values
    )

    if mbc.ndim == 4:
        mbc = mbc.sum(axis=2)

    assert mbc.shape[2] == C

    from data_prep import inverse_response_transform

    mu_post = (
        trace.posterior["mu"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", ...)
        .values
    )

    if mu_post.ndim == 3:
        y_mu_arr  = np.asarray(prep["y_mu"])
        y_std_arr = np.asarray(prep["y_std"])
        mu_log    = (mu_post * y_std_arr[None, None, :] + y_mu_arr[None, None, :]).mean(axis=2)
        mu_post   = mu_log
    else:
        y_mu_scalar = float(np.asarray(prep["y_mu"]).flat[0])
        mu_log      = mu_post * y_std + y_mu_scalar

    total_hat = inverse_response_transform(mu_log, prep)
    total_hat = np.maximum(total_hat, 0.0)

    mbc_abs     = np.abs(mbc)
    mbc_abs_sum = mbc_abs.sum(axis=2, keepdims=True) + 1e-12
    ch_share    = mbc_abs / mbc_abs_sum

    media_total_z  = mbc.sum(axis=2)
    media_frac     = np.abs(media_total_z) / (np.abs(mu_post) + 1e-12)
    media_frac     = np.clip(media_frac, 0.0, 1.0)

    ch_signups = total_hat[:, :, np.newaxis] * media_frac[:, :, np.newaxis] * ch_share

    alpha_lo = (1.0 - credible_interval) / 2.0
    alpha_hi = 1.0 - alpha_lo

    rows = []
    for j in range(C):
        ch_name     = spend_cols[j]
        metric_type = metric_types[j]

        ch_mean_t  = ch_signups[:, :, j].mean(axis=1)

        mean_metric_val = float(X_media_rw[:, j].mean())

        mean_actual_spend = mean_metric_val
        if metric_type != "Spend" and df_full is not None:
            actual_spend_col = spend_raw_cols[j]
            if actual_spend_col in df_full.columns:
                mean_actual_spend = float(
                    df_full[actual_spend_col].iloc[train_idx].mean()
                )

        ri_samples = ch_mean_t / (mean_metric_val + 1e-12) * 1000.0
        ri_spend_samples = ch_mean_t / (mean_actual_spend + 1e-12) * 1000.0

        mean_ri  = float(ri_samples.mean())
        lower_ri = float(np.percentile(ri_samples, alpha_lo * 100))
        upper_ri = float(np.percentile(ri_samples, alpha_hi * 100))

        mean_ri_spend  = float(ri_spend_samples.mean())
        lower_ri_spend = float(np.percentile(ri_spend_samples, alpha_lo * 100))
        upper_ri_spend = float(np.percentile(ri_spend_samples, alpha_hi * 100))

        rows.append({
            "channel_id"              : ch_name,
            "modeled_metric"          : metric_type,
            "mean_return_index"       : round(mean_ri,             4),
            "lower_bound"             : round(lower_ri,            4),
            "upper_bound"             : round(upper_ri,            4),
            "ci_width"                : round(upper_ri - lower_ri, 4),
            "mean_return_index_spend" : round(mean_ri_spend,       4),
            "lower_bound_spend"       : round(lower_ri_spend,      4),
            "upper_bound_spend"       : round(upper_ri_spend,      4),
            "mean_contribution_signups": round(float(ch_mean_t.mean()), 4),
            "mean_weekly_metric_value": round(mean_metric_val,     2),
            "mean_weekly_spend"       : round(mean_actual_spend,   2),
            "ci_level"                : credible_interval,
        })

    df = pd.DataFrame(rows).sort_values("mean_return_index", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────

def export_return_index(
    df_ri     : pd.DataFrame,
    out_dir   : Any,
    to_excel  : bool = True,
) -> None:
    """
    Exports the return index table to CSV (always) and optionally Excel.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "return_index.csv"
    df_ri.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")

    if to_excel:
        xlsx_path = out_dir / "return_index.xlsx"
        try:
            import openpyxl  # noqa: F401
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df_ri.to_excel(writer, sheet_name="Return Index", index=False)
                ws = writer.sheets["Return Index"]
                for col_cells in ws.columns:
                    max_len = max(
                        len(str(c.value)) if c.value is not None else 0
                        for c in col_cells
                    )
                    ws.column_dimensions[col_cells[0].column_letter].width = min(
                        max_len + 4, 40
                    )
            logger.info(f"  Saved: {xlsx_path}")
        except ImportError:
            logger.warning(
                "  Excel export skipped — openpyxl not installed. "
                "Run: pip install openpyxl"
            )
        except Exception as e:
            logger.warning(f"  Excel export failed: {e}")


def export_channel_prior_config(
    channel_prior_map : Optional[Dict[str, "ChannelPriorConfig"]],
    spend_cols        : list,
    out_dir           : Any,
) -> None:
    """
    Writes a CSV documenting which prior and likelihood was used for each channel.
    """
    from pathlib import Path
    out_dir = Path(out_dir)

    rows = []
    for col in spend_cols:
        cfg = get_prior_config(col, channel_prior_map)
        rows.append({
            "channel"          : col,
            "beta_dist"        : cfg.beta_dist,
            "beta_params"      : str(cfg.beta_params),
            "likelihood"       : cfg.likelihood,
            "likelihood_params": str(cfg.likelihood_params),
            "notes"            : cfg.notes,
        })

    df = pd.DataFrame(rows)
    path = out_dir / "channel_prior_config.csv"
    df.to_csv(path, index=False)
    logger.info(f"  Saved: {path}")


# ─── Domain prior library (prior_library.py) ────────────────────────────────

from typing import List as _List_pl


@dataclass
class DomainPrior:
    """Encapsulates recommended priors and transforms for a variable type."""
    # -- Identity ---------------------------------------------------------------
    channel_type  : str                          # e.g. "paid_search", "price"
    category      : str   = "media"              # "media" | "base" | "macro" | "event"
    description   : str   = ""

    # -- Recommended transforms (media only) -----------------------------------
    adstock       : str   = "geometric"
    saturation    : str   = "hill"

    # -- Adstock prior parameters ----------------------------------------------
    adstock_lam_prior : Dict[str, Any] = field(default_factory=lambda: {
        "dist": "beta", "alpha": 3.0, "beta": 3.0
    })

    # -- Saturation prior parameters -------------------------------------------
    sat_alpha_prior : Optional[Dict[str, Any]] = None
    sat_kappa_prior : Optional[Dict[str, Any]] = None

    # -- Beta (effect weight) prior --------------------------------------------
    beta_dist     : str              = "half_normal"
    beta_params   : Dict[str, Any]   = field(default_factory=lambda: {"sigma": 0.3})
    allow_negative: bool             = False

    # -- Likelihood ------------------------------------------------------------
    likelihood    : str   = "student_t"

    # -- Variable transform (non-media) ----------------------------------------
    var_transform : str   = "none"

    def describe(self) -> str:
        if self.category == "media":
            return (
                f"{self.channel_type}: {self.adstock}+{self.saturation}, "
                f"beta={self.beta_dist}({self.beta_params})"
            )
        return (
            f"{self.channel_type} ({self.category}): "
            f"beta={'Normal' if self.allow_negative else self.beta_dist}"
            f"({self.beta_params}), transform={self.var_transform}"
        )


_MEDIA_PRIORS: Dict[str, DomainPrior] = {
    "paid_search": DomainPrior(
        channel_type    = "paid_search",
        category        = "media",
        description     = "Google/Bing paid search — immediate response, moderate saturation",
        adstock         = "geometric",
        saturation      = "hill",
        adstock_lam_prior = {"dist": "beta", "alpha": 5.0, "beta": 2.0},
        beta_dist       = "half_normal",
        beta_params     = {"sigma": 0.5},
    ),
    "paid_social": DomainPrior(
        channel_type    = "paid_social",
        category        = "media",
        description     = "Meta/TikTok/LinkedIn paid — quick response, moderate saturation",
        adstock         = "geometric",
        saturation      = "hill",
        adstock_lam_prior = {"dist": "beta", "alpha": 4.0, "beta": 3.0},
        beta_dist       = "half_normal",
        beta_params     = {"sigma": 0.4},
    ),
    "social_organic": DomainPrior(
        channel_type    = "social_organic",
        category        = "media",
        description     = "Organic social — fast decay, mild saturation",
        adstock         = "geometric",
        saturation      = "softplus",
        adstock_lam_prior = {"dist": "beta", "alpha": 3.0, "beta": 4.0},
        beta_dist       = "half_normal",
        beta_params     = {"sigma": 0.2},
    ),
    "display": DomainPrior(
        channel_type    = "display",
        category        = "media",
        description     = "Programmatic display / DV360 — awareness, delayed, strong saturation",
        adstock         = "weibull",
        saturation      = "hill",
        adstock_lam_prior = {"dist": "gamma", "alpha": 3.0, "beta": 1.5},
        beta_dist       = "half_normal",
        beta_params     = {"sigma": 0.3},
    ),
    "tv": DomainPrior(
        channel_type    = "tv",
        category        = "media",
        description     = "Television — delayed peak, long tail, strong saturation",
        adstock         = "weibull",
        saturation      = "hill",
        adstock_lam_prior = {"dist": "gamma", "alpha": 4.0, "beta": 1.0},
        sat_alpha_prior = {"dist": "gamma", "alpha": 2.0, "beta": 1.0},
        beta_dist       = "half_normal",
        beta_params     = {"sigma": 0.3},
    ),
    "radio": DomainPrior(
        channel_type    = "radio",
        category        = "media",
        description     = "Radio — moderate decay, moderate saturation",
        adstock         = "geometric",
        saturation      = "hill",
        adstock_lam_prior = {"dist": "beta", "alpha": 3.0, "beta": 3.0},
        beta_dist       = "half_normal",
        beta_params     = {"sigma": 0.25},
    ),
    "ooh": DomainPrior(
        channel_type    = "ooh",
        category        = "media",
        description     = "Out of home / billboard — very delayed, threshold effect",
        adstock         = "weibull",
        saturation      = "logistic",
        adstock_lam_prior = {"dist": "gamma", "alpha": 4.0, "beta": 1.0},
        beta_dist       = "half_normal",
        beta_params     = {"sigma": 0.2},
    ),
    "email": DomainPrior(
        channel_type    = "email",
        category        = "media",
        description     = "Email marketing — near-immediate, strong diminishing returns",
        adstock         = "geometric",
        saturation      = "exponential",
        adstock_lam_prior = {"dist": "beta", "alpha": 6.0, "beta": 2.0},
        beta_dist       = "half_normal",
        beta_params     = {"sigma": 0.3},
    ),
    "affiliate": DomainPrior(
        channel_type    = "affiliate",
        category        = "media",
        description     = "Affiliate / referral — immediate, linear-ish response",
        adstock         = "geometric",
        saturation      = "softplus",
        adstock_lam_prior = {"dist": "beta", "alpha": 5.0, "beta": 2.0},
        beta_dist       = "half_normal",
        beta_params     = {"sigma": 0.4},
    ),
    "print": DomainPrior(
        channel_type    = "print",
        category        = "media",
        description     = "Print / magazine — delayed, moderate saturation",
        adstock         = "weibull",
        saturation      = "hill",
        adstock_lam_prior = {"dist": "gamma", "alpha": 3.0, "beta": 1.5},
        beta_dist       = "half_normal",
        beta_params     = {"sigma": 0.2},
    ),
    "generic_media": DomainPrior(
        channel_type    = "generic_media",
        category        = "media",
        description     = "Generic / unknown media channel — neutral defaults",
        adstock         = "geometric",
        saturation      = "softplus",
        beta_dist       = "half_normal",
        beta_params     = {"sigma": 0.3},
    ),
}


_BASE_PRIORS: Dict[str, DomainPrior] = {
    "price": DomainPrior(
        channel_type    = "price",
        category        = "base",
        description     = "Unit price — typically negative effect (elasticity)",
        beta_dist       = "normal",
        beta_params     = {"mu": -0.3, "sigma": 0.3},
        allow_negative  = True,
        var_transform   = "log",
    ),
    "distribution": DomainPrior(
        channel_type    = "distribution",
        category        = "base",
        description     = "Store count / weighted distribution — positive effect",
        beta_dist       = "half_normal",
        beta_params     = {"sigma": 0.5},
        allow_negative  = False,
        var_transform   = "z_score",
    ),
    "promotion": DomainPrior(
        channel_type    = "promotion",
        category        = "base",
        description     = "Promotion flag or depth — positive short-term lift",
        beta_dist       = "half_normal",
        beta_params     = {"sigma": 0.3},
        allow_negative  = False,
        var_transform   = "none",
    ),
    "competitor_price": DomainPrior(
        channel_type    = "competitor_price",
        category        = "base",
        description     = "Competitor pricing — positive cross-elasticity expected",
        beta_dist       = "normal",
        beta_params     = {"mu": 0.1, "sigma": 0.2},
        allow_negative  = True,
        var_transform   = "log",
    ),
    "generic_base": DomainPrior(
        channel_type    = "generic_base",
        category        = "base",
        description     = "Generic base variable — can be positive or negative",
        beta_dist       = "normal",
        beta_params     = {"mu": 0.0, "sigma": 0.3},
        allow_negative  = True,
        var_transform   = "z_score",
    ),
}


_MACRO_PRIORS: Dict[str, DomainPrior] = {
    "gdp": DomainPrior(
        channel_type    = "gdp",
        category        = "macro",
        description     = "GDP or GDP growth — typically positive",
        beta_dist       = "normal",
        beta_params     = {"mu": 0.1, "sigma": 0.2},
        allow_negative  = True,
        var_transform   = "z_score",
    ),
    "cpi": DomainPrior(
        channel_type    = "cpi",
        category        = "macro",
        description     = "Consumer Price Index — typically negative for demand",
        beta_dist       = "normal",
        beta_params     = {"mu": -0.1, "sigma": 0.2},
        allow_negative  = True,
        var_transform   = "z_score",
    ),
    "unemployment": DomainPrior(
        channel_type    = "unemployment",
        category        = "macro",
        description     = "Unemployment rate — typically negative for demand",
        beta_dist       = "normal",
        beta_params     = {"mu": -0.1, "sigma": 0.2},
        allow_negative  = True,
        var_transform   = "z_score",
    ),
    "interest_rate": DomainPrior(
        channel_type    = "interest_rate",
        category        = "macro",
        description     = "Interest rate / central bank rate",
        beta_dist       = "normal",
        beta_params     = {"mu": -0.05, "sigma": 0.15},
        allow_negative  = True,
        var_transform   = "z_score",
    ),
    "generic_macro": DomainPrior(
        channel_type    = "generic_macro",
        category        = "macro",
        description     = "Generic macro-economic variable",
        beta_dist       = "normal",
        beta_params     = {"mu": 0.0, "sigma": 0.2},
        allow_negative  = True,
        var_transform   = "z_score",
    ),
}


_EVENT_PRIOR = DomainPrior(
    channel_type    = "event",
    category        = "event",
    description     = "Binary event / holiday / product launch dummy",
    beta_dist       = "normal",
    beta_params     = {"mu": 0.0, "sigma": 0.3},
    allow_negative  = True,
    var_transform   = "none",
)


ALL_PRIORS: Dict[str, DomainPrior] = {
    **_MEDIA_PRIORS,
    **_BASE_PRIORS,
    **_MACRO_PRIORS,
    "event": _EVENT_PRIOR,
}

CHANNEL_TYPES = sorted(_MEDIA_PRIORS.keys())
BASE_TYPES    = sorted(_BASE_PRIORS.keys())
MACRO_TYPES   = sorted(_MACRO_PRIORS.keys())


def get_domain_prior(channel_type: str) -> DomainPrior:
    """Look up a domain prior by channel type name.  Falls back to generic."""
    ct = channel_type.lower().strip()
    if ct in ALL_PRIORS:
        return ALL_PRIORS[ct]
    for key, prior in ALL_PRIORS.items():
        if ct in key or key in ct:
            logger.debug(f"  Partial match: '{ct}' -> '{key}'")
            return prior
    logger.warning(f"  Unknown channel_type '{ct}' — using generic_media defaults.")
    return _MEDIA_PRIORS["generic_media"]


def list_available_types() -> str:
    """Returns a formatted string listing all available types for display."""
    lines = ["  MEDIA CHANNELS:"]
    for k, v in sorted(_MEDIA_PRIORS.items()):
        lines.append(f"    {k:<20s} — {v.description}")
    lines.append("\n  BASE VARIABLES:")
    for k, v in sorted(_BASE_PRIORS.items()):
        lines.append(f"    {k:<20s} — {v.description}")
    lines.append("\n  MACRO VARIABLES:")
    for k, v in sorted(_MACRO_PRIORS.items()):
        lines.append(f"    {k:<20s} — {v.description}")
    lines.append(f"\n  EVENTS:  event — {_EVENT_PRIOR.description}")
    return "\n".join(lines)


INDUSTRY_MULTIPLIERS = {
    "cpg"        : 0.8,
    "pharma"     : 0.5,
    "ecommerce"  : 1.2,
    "saas"       : 0.6,
    "retail"     : 1.0,
    "automotive" : 0.7,
    "finance"    : 0.6,
    "telecom"    : 0.8,
}


def apply_industry_adjustment(
    prior     : DomainPrior,
    industry  : str,
) -> DomainPrior:
    """Returns a copy of the prior with beta sigma scaled by the industry multiplier."""
    from copy import deepcopy
    mult = INDUSTRY_MULTIPLIERS.get(industry.lower().strip(), 1.0)
    if mult == 1.0:
        return prior
    adjusted = deepcopy(prior)
    if "sigma" in adjusted.beta_params:
        adjusted.beta_params["sigma"] = round(adjusted.beta_params["sigma"] * mult, 4)
    return adjusted


def calibrate_priors_from_history(
    historical_betas : Dict[str, float],
    confidence       : float = 0.7,
    base_dist        : str   = "half_normal",
) -> Dict[str, Dict[str, Any]]:
    """
    Calibrates prior sigma for each channel so that the prior concentrates
    around the historical point estimate.
    """
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    scale_factor = 1.0 / (0.5 + confidence * 2.0)

    calibrated = {}
    for channel, hist_beta in historical_betas.items():
        abs_beta = abs(hist_beta)
        if abs_beta < 1e-6:
            sigma = 0.1
        else:
            sigma = abs_beta * scale_factor

        sigma = max(sigma, 0.02)

        if base_dist == "log_normal":
            import math
            mu_ln = math.log(max(abs_beta, 0.01))
            sigma_ln = max(sigma, 0.1)
            calibrated[channel] = {
                "dist"   : "log_normal",
                "params" : {"mu": round(mu_ln, 4), "sigma": round(sigma_ln, 4)},
                "notes"  : f"calibrated from historical beta={hist_beta:.4f}, confidence={confidence}",
            }
        else:
            calibrated[channel] = {
                "dist"   : "half_normal",
                "params" : {"sigma": round(sigma, 4)},
                "notes"  : f"calibrated from historical beta={hist_beta:.4f}, confidence={confidence}",
            }

        logger.info(
            f"  [CALIBRATE] {channel}: hist_beta={hist_beta:.4f} -> "
            f"{calibrated[channel]['dist']}(sigma={sigma:.4f}), confidence={confidence}"
        )

    return calibrated


def calibrate_from_previous_run(
    return_index_csv : str,
    confidence       : float = 0.7,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience: reads a return_index.csv from a previous model run and
    calibrates priors for the next run.
    """
    import pandas as pd
    from pathlib import Path

    path = Path(return_index_csv)
    if not path.exists():
        raise FileNotFoundError(f"Previous run results not found: {return_index_csv}")

    df = pd.read_csv(path)
    if "channel_id" not in df.columns or "mean_return_index" not in df.columns:
        raise ValueError("CSV must have 'channel_id' and 'mean_return_index' columns")

    historical = {}
    for _, row in df.iterrows():
        ch = str(row["channel_id"])
        ri = float(row["mean_return_index"])
        historical[ch] = ri / 1000.0

    return calibrate_priors_from_history(historical, confidence)
