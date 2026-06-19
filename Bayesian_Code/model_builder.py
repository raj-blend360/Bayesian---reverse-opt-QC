# model_builder.py
# ─────────────────────────────────────────────────────────────────────────────
# Defines the Bayesian MMM as a PyMC probabilistic model.
#
# The model structure (for each time period t):
#
#   response[t] = baseline[t]          ← long-run trend (linear or random walk)
#               + seasonality[t]       ← annual Fourier cycles
#               + Σ_j media_j[t]       ← each channel's transformed contribution
#               + controls[t]          ← price, promo, events, etc.
#               + noise
#
# Each media channel goes through 3 transforms before entering the model:
#   1. Lag shift   — optional; models delayed response (e.g. TV brand campaigns)
#   2. Adstock     — carry-over effect (geometric or weibull decay)
#   3. Saturation  — diminishing returns (softplus, hill, logistic, exponential)
#
# Key functions:
#   build_mmm()                    — constructs the PyMC model
#   sample_model()                 — runs NUTS sampling
#   build_channel_transform_summary() — extracts posterior parameter table
# ─────────────────────────────────────────────────────────────────────────────

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import pandas as pd

# ── JAX / NumPyro configuration ───────────────────────────────────────────────
# Configure JAX to use all available CPU threads before any JAX import.
# This is a no-op if JAX is not installed (numpyro falls back to PyTensor NUTS).
try:
    import os as _os
    _cpu_count = _os.cpu_count() or 4
    # Tell XLA (JAX's compiler) to use all CPUs for parallelism
    _os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={_cpu_count}")
    import jax
    jax.config.update("jax_platform_name", "cpu")   # ensure CPU backend
    jax.config.update("jax_enable_x64", True)        # match PyMC's float64 default
    import jax.numpy as jnp  # noqa: F401 — triggers JIT cache warm-up
    logging.getLogger(__name__).info(
        f"JAX {jax.__version__} ready — {_cpu_count} CPU device(s), float64 enabled"
    )
except ImportError:
    logging.getLogger(__name__).warning(
        "JAX/NumPyro not installed — NUTS will use the slower PyTensor backend. "
        "Run: pip install 'numpyro>=0.13.0' 'jax>=0.4.23' 'jaxlib>=0.4.23'"
    )

from config import ModelConfig, ChannelTransformSpec, GLOBAL_SEED

try:
    from priors import get_domain_prior as _get_domain_prior
except ImportError:
    _get_domain_prior = None
    # Emit a logger warning so users know domain priors are unavailable.
    # This is deferred to a module-level statement below after logger is defined.
from transforms import (
    apply_lag_shift,
    geometric_adstock, weibull_adstock,
    two_timescale_adstock,
    geometric_adstock_scan, two_timescale_adstock_scan,
    sat_softplus, sat_hill, sat_logistic, sat_exponential,
)
from priors import (
    ChannelPriorConfig, get_prior_config, build_beta_prior, build_likelihood,
    DEFAULT_PRIOR_CONFIG, _check_hierarchical_compat,
)

logger = logging.getLogger("MMM")

# Emit the deferred prior_library warning now that logger is available.
if _get_domain_prior is None:
    logger.warning(
        "[MODEL_BUILDER] prior_library.py could not be imported — "
        "domain-aware default priors are DISABLED. All channels will use "
        "generic HalfNormal(sigma=0.3) priors regardless of channel_type. "
        "Ensure prior_library.py is present in the same directory."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_mmm(
    prep              : Dict[str, Any],
    cfg               : ModelConfig,
    channel_specs     : Optional[Dict[int, ChannelTransformSpec]] = None,
    focal_channel     : Optional[int] = None,
    focal_adstock     : Optional[str] = None,
    focal_saturation  : Optional[str] = None,
    channel_prior_map : Optional[Dict[str, ChannelPriorConfig]] = None,
    use_scan_adstock  : bool = True,
) -> pm.Model:
    """
    Constructs the full C-channel Bayesian MMM model.

    Transform priority per channel j
    ─────────────────────────────────
    1. focal_channel == j  → override with (focal_adstock, focal_saturation).
    2. channel_specs provided and j in channel_specs → use spec settings.
    3. Fallback → cfg.adstock_type / cfg.saturation / cfg.use_lag.

    Per-channel prior configuration
    ────────────────────────────────
    channel_prior_map : {channel_column_name → ChannelPriorConfig}
      Overrides the beta prior distribution and/or likelihood for specific
      channels. Channels not in the map use DEFAULT_PRIOR_CONFIG
      (HalfNormal beta + StudentT likelihood).
      See channel_priors.py for full documentation.

    Model structure
    ───────────────
    log(y+1) = baseline(t)
             + Σ_j [ beta_j * sat_j( adstock_j( lag_j(x_j) ) ) ]
             + Σ_k [ gamma_k * z_k ]
             + Σ_f [ delta_f * fourier_f ]
             + ε

    ε  ~ Likelihood(mu, σ, ν),  where likelihood family is per-channel
         configurable (default StudentT).
    """
    # FIX-1: Raise early if hierarchical mode would silently override custom priors
    _check_hierarchical_compat(cfg, channel_prior_map)

    # ── Unpack dimensions ────────────────────────────────────
    T          = prep["T"]
    C          = prep["C"]
    P          = prep.get("P", 1)               # 1 = flat, >1 = multi-product
    F          = prep.get("F", 1)               # number of channel families
    family_idx = prep.get("family_idx", np.zeros(C, dtype=int))
    family_names = prep.get("family_names", ["generic"])
    product_names = prep.get("product_names", [])

    train_idx  = prep["train_idx"]

    # ── Sliced training arrays ───────────────────────────────
    # y_scaled is (T,) for flat or (T,P) for multi-product
    y_sc_full = prep["y_scaled"]
    if P > 1:
        y_scaled = y_sc_full[train_idx, :]      # (T_train, P)
    else:
        y_scaled = y_sc_full[train_idx]          # (T_train,)

    # X_media_scaled is (T,C) for flat or (T,P,C) for multi-product
    X_media_sc_full = prep["X_media_scaled"]
    if P > 1:
        X_media_sc = X_media_sc_full[train_idx, :, :]   # (T, P, C)
    else:
        X_media_sc = X_media_sc_full[train_idx]          # (T, C)

    # Rebuild X_fourier if this grid config requests a different order than what
    # data_prep built (data_prep builds it once using the base fourier_order).
    # Apply the same data-length cap as data_prep to avoid over-parameterisation.
    _prep_fo   = prep["X_fourier"].shape[1] // 2
    _period    = float(prep.get("fourier_period", 52))
    _fo_cap    = max(1, prep["T"] // int(_period))
    _cfg_fo    = min(max(1, int(cfg.fourier_order)), _fo_cap)
    if _cfg_fo != _prep_fo:
        from data_prep import build_fourier_features as _bff
        X_fourier = _bff(prep["T"], _period, _cfg_fo)[train_idx]
    else:
        X_fourier  = prep["X_fourier"][train_idx]
    X_controls = prep["X_controls"][train_idx] if prep["X_controls"] is not None else None
    t_norm     = prep["t_norm"][train_idx]
    T_train    = len(t_norm)   # training-period length; differs from T when holdout_periods > 0
    n_fourier  = X_fourier.shape[1]

    # DOW Fourier features (daily data only, when use_dow_effects=True)
    _X_fourier_dow_full = prep.get("X_fourier_dow")
    X_fourier_dow_np = _X_fourier_dow_full[train_idx] if _X_fourier_dow_full is not None else None
    n_fourier_dow    = _X_fourier_dow_full.shape[1] if _X_fourier_dow_full is not None else 0

    # Outlier date spike indicators (binary columns from DataConfig.outlier_dates)
    _X_outlier_full = prep.get("X_outlier_indicators")
    X_outlier_np = _X_outlier_full[train_idx] if _X_outlier_full is not None else None
    n_outlier    = _X_outlier_full.shape[1] if _X_outlier_full is not None else 0

    # Frequency-aware GRW prior scale: keep ~1 SD annual drift regardless of period.
    # sigma_rw_base × sqrt(T_per_year) ≈ 1.0 SD, so sigma_rw_base = 1/sqrt(T_per_year).
    # Reference: weekly data (T=52) → sigma_rw_base ≈ 0.139 ≈ 0.15 (original hardcoded value).
    _freq_str     = str(prep.get("frequency", "weekly")).lower()
    _T_per_year   = {"weekly": 52.18, "daily": 365.25, "monthly": 12.0}.get(_freq_str, 52.18)
    _sigma_rw_base = 0.15 * float(np.sqrt(52.18 / _T_per_year))

    # ── PyTensor constants ───────────────────────────────────
    if P > 1:
        X_m = pt.as_tensor_variable(X_media_sc.astype(np.float64))   # (T, P, C)
    else:
        X_m = pt.as_tensor_variable(X_media_sc.astype(np.float64))   # (T, C)
    X_f = pt.as_tensor_variable(X_fourier.astype(np.float64))
    t_t = pt.as_tensor_variable(t_norm.astype(np.float64))

    # ── Adstock reset mask: (T, C) — 1.0 at first active week after a dark gap ─
    # Built by data_prep and used by scan-based adstock to wipe carry state at
    # channel restart.  Sliced to training rows.  When use_scan_adstock=False
    # (Stage 0 fast mode) this tensor is unused.
    _reset_mask_full = prep.get("reset_mask")
    if use_scan_adstock and _reset_mask_full is not None and np.any(_reset_mask_full > 0.0):
        reset_mask_pt = pt.as_tensor_variable(
            _reset_mask_full[train_idx].astype(np.float64)
        )   # (T_train, C)
    else:
        reset_mask_pt = None

    # Flighting mask: (T, C) tensor — 1.0=active, 0.0=dark.
    # Retained for informational / logging purposes; the post-saturation
    # channel masking has been REMOVED — dark-period carry now decays naturally
    # inside the scan, with an explicit state reset at restart.
    _flight_mask_full = prep.get("flight_mask")
    has_flighting = (
        _flight_mask_full is not None and np.any(_flight_mask_full < 1.0)
    )

    # ── Family configs for hierarchical adstock ─────────────
    # Merge DEFAULT_FAMILY_CONFIGS with any user overrides in cfg.family_configs
    from config import DEFAULT_FAMILY_CONFIGS, ChannelFamilyConfig
    eff_family_cfgs: Dict[str, ChannelFamilyConfig] = {
        **DEFAULT_FAMILY_CONFIGS,
        **getattr(cfg, "family_configs", {}),
    }
    # Build (F,) arrays of median half-lives and ratios for use in priors
    hl_slow_medians_np = np.array([
        eff_family_cfgs.get(fn, eff_family_cfgs["generic"]).hl_slow_median
        for fn in family_names
    ], dtype=np.float64)
    hl_ratio_medians_np = np.array([
        eff_family_cfgs.get(fn, eff_family_cfgs["generic"]).hl_ratio_median
        for fn in family_names
    ], dtype=np.float64)
    hl_sigma_np = np.array([
        eff_family_cfgs.get(fn, eff_family_cfgs["generic"]).hl_sigma
        for fn in family_names
    ], dtype=np.float64)

    use_two_ts = bool(getattr(cfg, "use_two_timescale_adstock", False))
    use_hier   = bool(getattr(cfg, "use_hierarchical", False))
    use_syn    = bool(getattr(cfg, "use_synergies", False)) and F >= 2

    # FIX (Bug 2): Warn when two-timescale is requested but hierarchical is off.
    # In that case the two-timescale block (Block 6) never fires and the model
    # silently falls back to plain geometric/weibull adstock with no pooling.
    if use_two_ts and not use_hier:
        logger.warning(
            "[MODEL_BUILDER] use_two_timescale_adstock=True requires use_hierarchical=True "
            "to activate the family-level adstock hyperpriors.  "
            "Falling back to plain geometric adstock per channel (no pooling).  "
            "Set use_hierarchical=True in the config to enable two-timescale adstock."
        )

    with pm.Model() as model:

        # ══════════════════════════════════════════════════════
        # BLOCK 1 — OBSERVATION NOISE
        # ══════════════════════════════════════════════════════
        # FIX (Issue 4): Use per-product sigma_y when P > 1.
        # A single scalar sigma_y forces both products to share the same
        # noise level, which is wrong when products have different variance.
        # shape=(P,) gives each product its own noise parameter.
        if P > 1:
            sigma_y = pm.HalfStudentT("sigma_y", nu=4, sigma=0.25, shape=P)
        else:
            sigma_y = pm.HalfStudentT("sigma_y", nu=4, sigma=0.25)
        nu_minus = pm.Exponential("nu_minus_two", lam=1.0 / 15.0)
        nu       = pm.Deterministic("nu", nu_minus + 2.0)

        # ══════════════════════════════════════════════════════
        # BLOCK 2 — BASELINE TREND (shared across products)
        # ══════════════════════════════════════════════════════
        baseline_type = (cfg.baseline_type or "linear_trend").strip().lower()

        if baseline_type == "linear_trend":
            intercept = pm.Normal("intercept", mu=0.0, sigma=0.5)
            slope     = pm.Normal("slope", mu=0.0, sigma=0.1)
            baseline  = pm.Deterministic("baseline", intercept + slope * t_t)

        elif baseline_type == "gaussian_random_walk":
            intercept = pm.Normal("intercept", mu=0.0, sigma=0.5)
            sigma_rw  = pm.HalfNormal("sigma_rw", sigma=_sigma_rw_base)
            rw_steps  = pm.Normal("rw_steps", mu=0.0, sigma=sigma_rw, shape=len(t_norm))
            rw_path   = pt.cumsum(rw_steps)
            rw_center = rw_path - pt.mean(rw_path)
            baseline  = pm.Deterministic("baseline", intercept + rw_center)

        elif baseline_type == "noncentered_gaussian_random_walk":
            intercept = pm.Normal("intercept", mu=0.0, sigma=0.5)
            sigma_rw  = pm.HalfNormal("sigma_rw", sigma=_sigma_rw_base)
            eps_rw    = pm.Normal("eps_rw", mu=0.0, sigma=1.0, shape=len(t_norm))
            rw_path   = pt.cumsum(eps_rw) * sigma_rw
            rw_center = rw_path - pt.mean(rw_path)
            baseline  = pm.Deterministic("baseline", intercept + rw_center)

        elif baseline_type == "piecewise_linear":
            intercept  = pm.Normal("intercept", mu=0.0, sigma=0.5)
            base_slope = pm.Normal("slope", mu=0.0, sigma=0.08)
            n_knots    = max(1, int(cfg.piecewise_knots))
            knots      = np.linspace(0.15, 0.85, n_knots).astype(np.float64)
            hinge_np   = np.column_stack(
                [np.maximum(t_norm - k, 0.0) for k in knots]
            ).astype(np.float64)
            hinge_t    = pt.as_tensor_variable(hinge_np)
            slope_d    = pm.Normal("slope_deltas", mu=0.0, sigma=0.08, shape=n_knots)
            baseline   = pm.Deterministic(
                "baseline", intercept + base_slope * t_t + pt.dot(hinge_t, slope_d)
            )
        else:
            raise ValueError(f"Unknown baseline_type: {cfg.baseline_type}")

        # ══════════════════════════════════════════════════════
        # BLOCK 3 — FOURIER SEASONALITY (shared across products)
        # ══════════════════════════════════════════════════════
        delta       = pm.Normal("delta", mu=0.0, sigma=0.3, shape=n_fourier)
        seasonality = pm.Deterministic(
            "seasonality", pt.dot(X_f.astype("float64"), delta)
        )

        # ── Day-of-week seasonality (daily data only) ──────────────────────
        if X_fourier_dow_np is not None:
            X_fdow      = pt.as_tensor_variable(X_fourier_dow_np.astype("float64"))
            delta_dow   = pm.Normal("delta_dow", mu=0.0, sigma=0.3, shape=n_fourier_dow)
            seasonality_dow = pm.Deterministic(
                "seasonality_dow", pt.dot(X_fdow, delta_dow)
            )
        else:
            seasonality_dow = 0.0

        # ══════════════════════════════════════════════════════
        # BLOCK 4 — PRODUCT-LEVEL INTERCEPTS  (multi-product only)
        # Each product gets its own baseline intercept on top of the
        # shared trend.  Non-centered: intercept_p = mu_int + z_int * sig_int
        # ══════════════════════════════════════════════════════
        if P > 1:
            mu_int      = pm.Normal("mu_intercept_p", mu=0.0, sigma=0.5)
            sig_int     = pm.HalfNormal("sigma_intercept_p", sigma=0.5)
            z_int       = pm.Normal("z_intercept_p", mu=0.0, sigma=1.0, shape=P)
            intercept_p = pm.Deterministic(
                "intercept_p", mu_int + sig_int * z_int
            )   # (P,)
        else:
            intercept_p = None

        # ══════════════════════════════════════════════════════
        # BLOCK 5 — CONTROLS, BASE, MACRO, EVENTS
        # ══════════════════════════════════════════════════════

        # ── Control covariates ─────────────────────────────────
        if X_controls is not None and cfg.use_controls:
            n_Z            = X_controls.shape[1]
            X_c            = pt.as_tensor_variable(X_controls.astype(np.float64))
            gamma          = pm.Normal("gamma_controls", mu=0.0, sigma=0.2, shape=n_Z)
            control_effect = pm.Deterministic("control_effect", pt.dot(X_c, gamma))
        else:
            control_effect = 0.0

        # ── Outlier date spike effects (always included when present) ───────
        # Wider prior (sigma=0.5) since spikes can be large in either direction.
        if X_outlier_np is not None:
            X_out_t        = pt.as_tensor_variable(X_outlier_np.astype(np.float64))
            gamma_outlier  = pm.Normal("gamma_outlier", mu=0.0, sigma=0.5, shape=n_outlier)
            outlier_effect = pm.Deterministic("outlier_effect", pt.dot(X_out_t, gamma_outlier))
        else:
            outlier_effect = 0.0

        # ── Base variables (price, promo, distribution) ────────
        base_effect = 0.0
        X_base = prep.get("X_base")
        if X_base is not None and X_base.shape[1] > 0:
            n_base    = X_base.shape[1]
            base_info = prep.get("base_info", [])
            X_b       = pt.as_tensor_variable(X_base[train_idx].astype(np.float64))
            base_gamma_list = []
            for bi in range(n_base):
                info    = base_info[bi] if bi < len(base_info) else {}
                ch_type = info.get("channel_type", "generic_base")
                if _get_domain_prior is not None:
                    dp = _get_domain_prior(ch_type)
                    if dp.allow_negative:
                        g = pm.Normal(f"gamma_base_{bi}",
                                      mu=dp.beta_params.get("mu", 0.0),
                                      sigma=dp.beta_params.get("sigma", 0.3))
                    else:
                        g = pm.HalfNormal(f"gamma_base_{bi}",
                                          sigma=dp.beta_params.get("sigma", 0.3))
                else:
                    g = pm.Normal(f"gamma_base_{bi}", mu=0.0, sigma=0.3)
                base_gamma_list.append(g)
            base_gammas = pt.stack(base_gamma_list)
            base_effect = pm.Deterministic("base_effect", pt.dot(X_b, base_gammas))

        # ── Macro variables ────────────────────────────────────
        macro_effect = 0.0
        X_macro = prep.get("X_macro")
        if X_macro is not None and X_macro.shape[1] > 0:
            n_macro    = X_macro.shape[1]
            macro_info = prep.get("macro_info", [])
            X_mac      = pt.as_tensor_variable(X_macro[train_idx].astype(np.float64))
            macro_phi_list = []
            for mi in range(n_macro):
                info    = macro_info[mi] if mi < len(macro_info) else {}
                ch_type = info.get("channel_type", "generic_macro")
                if _get_domain_prior is not None:
                    dp = _get_domain_prior(ch_type)
                    ph = pm.Normal(f"phi_macro_{mi}",
                                   mu=dp.beta_params.get("mu", 0.0),
                                   sigma=dp.beta_params.get("sigma", 0.2))
                else:
                    ph = pm.Normal(f"phi_macro_{mi}", mu=0.0, sigma=0.2)
                macro_phi_list.append(ph)
            macro_phis   = pt.stack(macro_phi_list)
            macro_effect = pm.Deterministic("macro_effect", pt.dot(X_mac, macro_phis))

        # ── Event dummies (holidays, launches) ─────────────────
        event_effect = 0.0
        X_events = prep.get("X_events")
        if X_events is not None and X_events.shape[1] > 0:
            n_events = X_events.shape[1]
            X_ev     = pt.as_tensor_variable(X_events[train_idx].astype(np.float64))
            eta      = pm.Normal("eta_events", mu=0.0, sigma=0.3, shape=n_events)
            event_effect = pm.Deterministic("event_effect", pt.dot(X_ev, eta))

        # ══════════════════════════════════════════════════════
        # BLOCK 6 — HIERARCHICAL ADSTOCK PRIORS
        #
        # Three sub-cases depending on config flags:
        #
        # Case A: use_hierarchical=True AND use_two_timescale_adstock=True
        #   Full two-timescale adstock with family-level pooling.
        #   Family hyperpriors for half-life (slow) and fast/slow ratio.
        #   Channel deviations drawn non-centered from family prior.
        #
        # Case B: use_hierarchical=True AND use_two_timescale_adstock=False
        #   FIX (Bug 3): Hierarchical SINGLE-timescale (geometric) adstock.
        #   Previously this fell through to fully independent per-channel lam_j.
        #   Now: family-level log_lam_f hyperprior, channel deviations non-centered.
        #   This gives partial pooling of decay rates within families WITHOUT
        #   requiring the more complex two-timescale parameterisation.
        #
        # Case C: use_hierarchical=False
        #   Independent per-channel adstock priors (original behaviour).
        #
        # Parameterisation for Case A (all in log half-life space):
        # ─────────────────────────────────────────────────────────
        # Family level:
        #   log_hl_slow_f  ~ Normal(log(median_f), sigma_f)     shape (F,)
        #   sigma_hl_f     ~ HalfNormal(hl_sigma_f)             shape (F,)
        #   log_r_f        ~ Normal(log(ratio_median_f), 0.4)   shape (F,)
        #   w_mix_f        ~ Beta(2, 2)                         shape (F,)
        # Channel level:
        #   z_hl_c         ~ Normal(0, 1)                       shape (C,)
        #   log_hl_slow_c  = log_hl_slow_f[fam] + sigma_hl_f[fam] * z_hl_c
        #   rho_slow_c     = 0.5 ^ (1/exp(log_hl_slow_c))
        #   rho_fast_c     = 0.5 ^ (1/exp(log_hl_slow_c - log_r_f[fam]))
        #
        # Parameterisation for Case B (log lambda space):
        # ─────────────────────────────────────────────────────────
        # Family level:
        #   log_lam_f      ~ Normal(log(median_lam_f), hl_sigma_f)  shape (F,)
        #   sigma_lam_f    ~ HalfNormal(hl_sigma_f)                 shape (F,)
        # Channel level:
        #   z_lam_c        ~ Normal(0, 1)                           shape (C,)
        #   log_lam_c      = log_lam_f[fam] + sigma_lam_f[fam] * z_lam_c
        #   lam_c          = sigmoid(log_lam_c)  # constrained to (0,1)
        # ══════════════════════════════════════════════════════
        fam_idx_t = pt.as_tensor_variable(family_idx.astype(np.int64))
        _LOG_HALF = np.float64(np.log(0.5))   # compile-time constant

        if use_hier and use_two_ts:
            # ── Case A: Two-timescale with family pooling ──────
            log_hl_slow_f = pm.Normal(
                "log_hl_slow_f",
                mu    = np.log(hl_slow_medians_np),
                sigma = hl_sigma_np,
                shape = F,
            )
            sigma_hl_f = pm.HalfNormal("sigma_hl_f", sigma=hl_sigma_np, shape=F)
            log_r_f    = pm.Normal(
                "log_r_f",
                mu    = np.log(hl_ratio_medians_np),
                sigma = np.full(F, 0.4, dtype=np.float64),
                shape = F,
            )
            w_mix_f = pm.Beta("w_mix_f", alpha=2.0, beta=2.0, shape=F)

            # ── Channel-level deviations (non-centered) ────────
            z_hl_c         = pm.Normal("z_hl_c", mu=0.0, sigma=1.0, shape=C)
            log_hl_slow_c  = pm.Deterministic(
                "log_hl_slow_c",
                log_hl_slow_f[fam_idx_t] + sigma_hl_f[fam_idx_t] * z_hl_c,
            )
            hl_slow_c = pm.Deterministic("hl_slow_c", pt.exp(log_hl_slow_c))
            r_c       = pm.Deterministic("r_c",       pt.exp(log_r_f[fam_idx_t]))
            hl_fast_c = pm.Deterministic("hl_fast_c", hl_slow_c / r_c)

            rho_slow_c = pm.Deterministic(
                "rho_slow_c",
                pt.exp(_LOG_HALF / hl_slow_c),
            )
            rho_fast_c = pm.Deterministic(
                "rho_fast_c",
                pt.exp(_LOG_HALF / hl_fast_c),
            )
            # lam_c_hier used in Block 8 for channels not using two-timescale
            lam_c_hier = None
            logger.debug(
                f"  Two-timescale hierarchical adstock: F={F} families, C={C} channels. "
                f"Family half-life medians: {dict(zip(family_names, hl_slow_medians_np))}"
            )

        elif use_hier and not use_two_ts:
            # ── Case B: FIX (Bug 3) — Hierarchical single-timescale geometric
            # adstock.  Each family shares a log-lambda hyperprior; channels
            # deviate non-centredly.  lam_c is constrained to (0,1) via sigmoid.
            #
            # Convert hl_slow_medians → lam medians: lam = 0.5^(1/hl)
            lam_medians_np = np.exp(_LOG_HALF / hl_slow_medians_np)  # (F,)
            # Work in logit(lam) space for unconstrained sampling
            logit_lam_medians = np.log(lam_medians_np / (1.0 - lam_medians_np + 1e-8))

            logit_lam_f   = pm.Normal(
                "logit_lam_f",
                mu    = logit_lam_medians,
                sigma = hl_sigma_np,
                shape = F,
            )
            sigma_lam_f = pm.HalfNormal("sigma_lam_f", sigma=hl_sigma_np, shape=F)
            z_lam_c     = pm.Normal("z_lam_c", mu=0.0, sigma=1.0, shape=C)

            logit_lam_c = pm.Deterministic(
                "logit_lam_c",
                logit_lam_f[fam_idx_t] + sigma_lam_f[fam_idx_t] * z_lam_c,
            )
            # sigmoid maps logit space → (0,1), ensuring valid decay rate
            lam_c_hier = pm.Deterministic(
                "lam_c_hier",
                pt.sigmoid(logit_lam_c),   # (C,) — one pooled decay rate per channel
            )
            # Placeholders not used in Case B
            rho_slow_c = None
            rho_fast_c = None
            w_mix_f    = None
            logger.debug(
                f"  Hierarchical geometric adstock (Case B): F={F} families, C={C} channels. "
                f"Family lam medians: {dict(zip(family_names, lam_medians_np.round(3)))}"
            )

        else:
            # ── Case C: Independent per-channel adstock (no hierarchy) ────
            rho_slow_c = None
            rho_fast_c = None
            w_mix_f    = None
            lam_c_hier = None

        # ══════════════════════════════════════════════════════
        # BLOCK 7 — CHANNEL BETAS (hierarchical or independent)
        #
        # CASE A: Hierarchical, F>1 families (full partial pooling)
        # ──────────────────────────────────────────────────────
        #   Family level (non-centered):
        #     mu_beta_f    ~ Normal(0, family_beta_sigma)    (F,)
        #     sigma_beta_f ~ HalfNormal(channel_beta_sigma)  (F,)
        #     z_beta_c     ~ Normal(0, 1)                    (C,)
        #     beta_c = softplus(mu_beta_f[fam] + sigma_beta_f[fam] * z_beta_c)
        #              ↑ softplus ensures beta ≥ 0
        #
        #   Product level (if P>1, non-centered):
        #     sigma_p_raw  ~ HalfNormal(product_beta_sigma)  scalar
        #     z_pc         ~ Normal(0, 1)                    (P, C)
        #     beta_pc = softplus(beta_c_raw[None,:] + sigma_p_raw * z_pc)
        #
        # CASE B: Hierarchical, F==1 (global pooling, improved non-centered)
        # ──────────────────────────────────────────────────────
        #     mu_beta    ~ Normal(0, 0.5)
        #     sigma_beta ~ HalfNormal(0.3)
        #     z_beta     ~ Normal(0, 1)   (C,)
        #     betas = mu_beta + sigma_beta * z_beta   (C,)
        #
        # CASE C: Independent per-channel priors (no hierarchy)
        # ──────────────────────────────────────────────────────
        #     beta_j ~ ChannelPriorConfig distribution  per channel
        # ══════════════════════════════════════════════════════
        f_beta_sig = float(getattr(cfg, "family_beta_sigma",   0.5))
        c_beta_sig = float(getattr(cfg, "channel_beta_sigma",  0.3))
        p_beta_sig = float(getattr(cfg, "product_beta_sigma",  0.5))

        if use_hier and C >= 2 and F > 1:
            # ── Case A: family-level non-centered betas ────────
            mu_beta_f    = pm.Normal("mu_beta_f", mu=0.0, sigma=f_beta_sig, shape=F)
            sigma_beta_f = pm.HalfNormal("sigma_beta_f", sigma=c_beta_sig, shape=F)
            z_beta_c     = pm.Normal("z_beta_c", mu=0.0, sigma=1.0, shape=C)

            # beta_c_raw can be positive or negative; softplus maps to >0
            beta_c_raw = mu_beta_f[fam_idx_t] + sigma_beta_f[fam_idx_t] * z_beta_c
            beta_c     = pm.Deterministic(
                "beta_c",
                pt.log1p(pt.exp(beta_c_raw)),  # softplus — ensures > 0
            )

            if P > 1:
                # Product deviations from the shared channel mean
                sigma_p_raw = pm.HalfNormal("sigma_p_raw", sigma=p_beta_sig)
                z_pc        = pm.Normal("z_pc", mu=0.0, sigma=1.0, shape=(P, C))
                beta_pc     = pm.Deterministic(
                    "beta_pc",
                    pt.log1p(pt.exp(
                        beta_c_raw[None, :] + sigma_p_raw * z_pc
                    )),
                )   # (P, C) — all positive

                logger.debug(
                    f"  Hierarchical betas: F={F} families, C={C} channels, "
                    f"P={P} products (non-centered, softplus-positive)"
                )
            else:
                beta_pc = None   # P==1: use beta_c directly

        elif use_hier and C >= 2:
            # ── Case B: global pooling (no families) ──────────
            mu_beta    = pm.Normal("mu_beta", mu=0.0, sigma=0.5)
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.3)
            z_beta     = pm.Normal("z_beta_raw", mu=0.0, sigma=1.0, shape=C)
            # softplus ensures betas > 0, matching Cases A and C
            betas      = pm.Deterministic(
                "betas", pt.log1p(pt.exp(mu_beta + sigma_beta * z_beta))
            )
            beta_c     = None
            beta_pc    = None

            if channel_prior_map:
                for j, col in enumerate(prep["spend_cols"]):
                    pcfg = get_prior_config(col, channel_prior_map)
                    if pcfg.beta_dist != "half_normal":
                        logger.debug(
                            f"  Note: channel '{col}' has custom beta_dist but "
                            f"hierarchical mode is ON — using shared group prior."
                        )
        else:
            # ── Case C: independent per-channel priors ─────────
            beta_list = []
            for j, col in enumerate(prep["spend_cols"]):
                pcfg   = get_prior_config(col, channel_prior_map)
                beta_j = build_beta_prior(f"beta_ch{j}", pcfg)
                beta_list.append(beta_j)
            betas   = pt.stack(beta_list)
            beta_c  = None
            beta_pc = None

        use_time_varying_betas = bool(getattr(cfg, "use_time_varying_betas", False))
        beta_time = None
        if use_time_varying_betas:
            if beta_pc is not None:
                beta_base = beta_pc[None, :, :]
                beta_factor_shape = (T, 1, C)
            else:
                beta_base = (beta_c[None, :] if beta_c is not None else betas[None, :])
                beta_factor_shape = (T, C)

            beta_time_factor = pm.LogNormal(
                "beta_time_factor",
                mu=0.0,
                sigma=0.08,
                shape=beta_factor_shape,
            )
            beta_time = pm.Deterministic("beta_time", beta_base * beta_time_factor)
            logger.debug(
                f"  Time-varying betas enabled: beta_time shape={beta_time.shape}"
            )

        # ── Determine dominant likelihood ──────────────────────
        likelihood_counts: Dict[str, int] = {}
        for col in prep["spend_cols"]:
            pcfg = get_prior_config(col, channel_prior_map)
            likelihood_counts[pcfg.likelihood] = (
                likelihood_counts.get(pcfg.likelihood, 0) + 1
            )
        dominant_likelihood = max(likelihood_counts, key=likelihood_counts.get)
        dominant_cfg = DEFAULT_PRIOR_CONFIG
        for col in prep["spend_cols"]:
            pcfg = get_prior_config(col, channel_prior_map)
            if pcfg.likelihood == dominant_likelihood:
                dominant_cfg = pcfg
                break
        if len(likelihood_counts) > 1:
            logger.warning(
                f"  Multiple likelihood families: {likelihood_counts}. "
                f"Using dominant: '{dominant_likelihood}'."
            )

        # ══════════════════════════════════════════════════════
        # BLOCK 8 — PER-CHANNEL TRANSFORM PIPELINE
        #
        # For each channel j we:
        #   1. [Optional] Lag shift (softmax-weighted mixture)
        #   2. Adstock  (two-timescale OR geometric/weibull)
        #   3. Saturation
        #   4. Multiply by beta
        #
        # When P>1: loop over products inside each channel's block,
        # applying the SAME adstock/saturation params across products
        # (the channel-level transform is shared; only beta varies by product).
        #
        # Outputs:
        #   channel_contribs : list of C tensors, each (T,) for P==1
        #                      or (T, P) for P>1.
        # ══════════════════════════════════════════════════════
        channel_contribs  = []
        sat_tensors       = []   # store x_sat_j for synergy computation

        # Global Hill saturation parameters (optional)
        use_global_hill = bool(getattr(cfg, "use_global_hill", False))
        if use_global_hill:
            # Global Hill parameters shared across all channels
            global_alpha_sat = pm.Gamma("global_alpha_sat", alpha=3.0, beta=1.0)
            global_kappa_sat = pm.HalfNormal("global_kappa_sat", sigma=1.0)
            logger.debug("  Global Hill saturation enabled: shared alpha and kappa across all channels")

        for j in range(C):

            # ── Resolve adstock / saturation type ─────────────
            if focal_channel is not None and j == focal_channel:
                ch_adstock = focal_adstock
                ch_sat     = focal_saturation
                ch_use_lag = False
                ch_max_lag = cfg.max_lag
            elif channel_specs is not None and j in channel_specs:
                spec       = channel_specs[j]
                ch_adstock = spec.adstock_type
                ch_sat     = spec.saturation
                ch_use_lag = spec.use_lag
                ch_max_lag = spec.max_lag if spec.use_lag else cfg.max_lag
            else:
                ch_adstock = cfg.adstock_type
                ch_sat     = cfg.saturation
                ch_use_lag = cfg.use_lag
                ch_max_lag = cfg.max_lag

            adstock_lag = max(1, int(ch_max_lag))
            ch_col      = prep["spend_cols"][j]
            ch_pcfg     = get_prior_config(ch_col, channel_prior_map)
            sat_alpha_prior = ch_pcfg.sat_alpha_prior
            sat_kappa_prior = ch_pcfg.sat_kappa_prior

            # ── Sample PyMC priors for channel j (once, before product loop) ──
            # Adstock params are shared across all products for channel j.

            # Lag weights
            lag_w_j = None
            if ch_use_lag and adstock_lag > 0 and T > adstock_lag + 2:
                lag_w_j = pm.Dirichlet(
                    f"lag_w_{j}",
                    a=np.ones(adstock_lag + 1, dtype=np.float64),
                )
                pm.Deterministic(f"lag_mode_{j}", pt.argmax(lag_w_j))

            # Adstock params
            # Priority:
            #   1. Two-timescale hierarchical (Case A) — use family-pooled rho
            #   2. Single-timescale hierarchical (Case B, FIX Bug 3) — use lam_c_hier
            #   3. Independent per-channel (Case C / flat) — sample lam_j directly
            if use_hier and use_two_ts:
                # Case A: two-timescale with family pooling
                rho_s_j = rho_slow_c[j]
                rho_f_j = rho_fast_c[j]
                w_j     = w_mix_f[family_idx[j]]
                lam_j   = None   # not used in two-timescale path
            elif use_hier and lam_c_hier is not None:
                if ch_adstock == "weibull":
                    # Case B-weibull: Weibull's 2-parameter (scale, shape) distribution
                    # cannot be pooled like geometric's single lambda.  Sample lam_j
                    # and k_j independently per-channel (no family pooling for weibull).
                    lam_prior = ch_pcfg.adstock_lam_prior
                    if lam_prior and lam_prior.get("dist") == "gamma":
                        lam_j = pm.Gamma(f"lam_{j}",
                                         alpha=float(lam_prior.get("alpha", 3.0)),
                                         beta=float(lam_prior.get("beta",  1.0)))
                    else:
                        lam_j = pm.Gamma(f"lam_{j}", alpha=3.0, beta=1.0)
                    k_j     = pm.Gamma(f"k_shape_{j}", alpha=3.0, beta=1.0)
                    rho_s_j = None
                    rho_f_j = None
                    w_j     = None
                    # ch_adstock stays "weibull" — no override
                else:
                    # Case B-geometric: use family-pooled lambda from lam_c_hier.
                    lam_j   = lam_c_hier[j]   # PyTensor scalar from posterior
                    rho_s_j = None
                    rho_f_j = None
                    w_j     = None
                    ch_adstock = "geometric"
            elif ch_adstock == "geometric":
                # Case C: independent geometric
                lam_prior = ch_pcfg.adstock_lam_prior
                if lam_prior and lam_prior.get("dist") == "beta":
                    lam_j = pm.Beta(f"lam_{j}",
                                    alpha=float(lam_prior.get("alpha", 3.0)),
                                    beta=float(lam_prior.get("beta",  3.0)))
                else:
                    lam_j = pm.Beta(f"lam_{j}", alpha=3.0, beta=3.0)
                rho_s_j = None
                rho_f_j = None
                w_j     = None
            elif ch_adstock == "weibull":
                # Case C: independent weibull
                lam_prior = ch_pcfg.adstock_lam_prior
                if lam_prior and lam_prior.get("dist") == "gamma":
                    lam_j = pm.Gamma(f"lam_{j}",
                                     alpha=float(lam_prior.get("alpha", 3.0)),
                                     beta=float(lam_prior.get("beta",  1.0)))
                else:
                    lam_j = pm.Gamma(f"lam_{j}", alpha=3.0, beta=1.0)
                k_j     = pm.Gamma(f"k_shape_{j}", alpha=3.0, beta=1.0)
                rho_s_j = None
                rho_f_j = None
                w_j     = None
            else:
                raise ValueError(f"Unknown adstock_type '{ch_adstock}' for channel {j}")

            # Saturation params
            if ch_sat == "softplus":
                if sat_alpha_prior and sat_alpha_prior.get("dist") == "half_normal":
                    alpha_j = pm.HalfNormal(f"alpha_sat_{j}",
                                            sigma=float(sat_alpha_prior.get("sigma", 1.0)))
                else:
                    alpha_j = pm.HalfNormal(f"alpha_sat_{j}", sigma=1.0)
            elif ch_sat == "hill":
                if sat_alpha_prior and sat_alpha_prior.get("dist") == "gamma":
                    alpha_j = pm.Gamma(f"alpha_sat_{j}",
                                       alpha=float(sat_alpha_prior.get("alpha", 3.0)),
                                       beta=float(sat_alpha_prior.get("beta",  1.0)))
                else:
                    alpha_j = pm.Gamma(f"alpha_sat_{j}", alpha=3.0, beta=1.0)
                if sat_kappa_prior and sat_kappa_prior.get("dist") == "half_normal":
                    kappa_j = pm.HalfNormal(f"kappa_{j}",
                                            sigma=float(sat_kappa_prior.get("sigma", 1.0)))
                else:
                    kappa_j = pm.HalfNormal(f"kappa_{j}", sigma=1.0)
            elif ch_sat == "logistic":
                k_log_j = pm.HalfNormal(f"k_logistic_{j}", sigma=2.0)
                x0_j    = pm.Normal(f"x0_{j}", mu=0.0, sigma=1.0)
            elif ch_sat == "exponential":
                if sat_alpha_prior and sat_alpha_prior.get("dist") == "half_normal":
                    alpha_j = pm.HalfNormal(f"alpha_sat_{j}",
                                            sigma=float(sat_alpha_prior.get("sigma", 1.5)))
                else:
                    alpha_j = pm.HalfNormal(f"alpha_sat_{j}", sigma=1.5)
            else:
                raise ValueError(f"Unknown saturation '{ch_sat}' for channel {j}")

            # ── Per-channel reset signal for scan-based adstock ───────────────
            # reset_j is a (T_train,) PyTensor constant: 1.0 at the first
            # active week after >= N dark weeks, 0.0 everywhere else.
            # When use_scan_adstock=False (Stage 0 fast path) reset_mask_pt
            # is None and reset_j stays None — the scan path is skipped.
            reset_j = reset_mask_pt[:, j] if reset_mask_pt is not None else None

            # ── Apply transform to each product's spend for channel j ──────────
            #
            # Adstock choices (in priority order):
            #   use_scan_adstock=True  → scan functions (full history, reset-aware)
            #   use_scan_adstock=False → truncated-lag convolution (Stage 0 speed)
            #
            # For Weibull adstock there is no scan variant — it falls back to the
            # truncated convolution regardless of use_scan_adstock.
            sat_per_product = []
            n_prod_loop = P if P > 1 else 1
            for p in range(n_prod_loop):

                x_jp = X_m[:, p, j] if P > 1 else X_m[:, j]   # (T,)

                if lag_w_j is not None:
                    x_jp = apply_lag_shift(x_jp, lag_w_j, adstock_lag)

                # ── Adstock ──────────────────────────────────────────────────
                if use_scan_adstock:
                    # Scan path: full history, correct flighting-reset carry wipe
                    if use_hier and use_two_ts:
                        # Case A — two-timescale hierarchical (fast + slow chains)
                        x_ads = two_timescale_adstock_scan(
                            x_jp, rho_s_j, rho_f_j, w_j, reset=reset_j
                        )
                    elif ch_adstock == "geometric":
                        # Case B (hierarchical single-timescale) and
                        # Case C (independent geometric) both use the same scan
                        x_ads = geometric_adstock_scan(x_jp, lam_j, reset=reset_j)
                    else:
                        # Weibull: no recurrent state-space form — keep truncated
                        x_ads = weibull_adstock(x_jp, lam_j, k_j, adstock_lag)
                else:
                    # Fast path (Stage 0): truncated-lag convolution
                    if use_hier and use_two_ts:
                        x_ads = two_timescale_adstock(x_jp, rho_s_j, rho_f_j, w_j, adstock_lag)
                    elif ch_adstock == "geometric":
                        x_ads = geometric_adstock(x_jp, lam_j, adstock_lag)
                    else:
                        x_ads = weibull_adstock(x_jp, lam_j, k_j, adstock_lag)

                # ── Saturation ────────────────────────────────────────────────
                if ch_sat == "softplus":
                    x_sat = sat_softplus(x_ads, alpha_j) + 1e-8
                elif ch_sat == "hill":
                    if getattr(cfg, "use_dynamic_saturation", False):
                        dynamic_alpha = pm.Gamma(
                            f"alpha_sat_t_{j}", alpha=3.0, beta=1.0, shape=T_train
                        )
                        dynamic_kappa = pm.HalfNormal(
                            f"kappa_sat_t_{j}", sigma=1.0, shape=T_train
                        )
                        x_sat = sat_hill(x_ads, dynamic_alpha, dynamic_kappa) + 1e-8
                    elif use_global_hill:
                        # Use global Hill parameters
                        x_sat = sat_hill(x_ads, global_alpha_sat, global_kappa_sat) + 1e-8
                    else:
                        # Use per-channel Hill parameters
                        x_sat = sat_hill(x_ads, alpha_j, kappa_j) + 1e-8
                elif ch_sat == "logistic":
                    x_sat = sat_logistic(x_ads, k_log_j, x0_j) + 1e-8
                else:
                    x_sat = sat_exponential(x_ads, alpha_j) + 1e-8

                # NOTE: post-saturation flight_mask zeroing has been REMOVED.
                # Dark-period carry now decays naturally inside the scan, and
                # the carry is explicitly reset at channel restart via reset_j.
                # This matches the reference model's state-space semantics.

                sat_per_product.append(x_sat)   # (T,)

            # ── Apply beta weights ─────────────────────────────
            if P > 1:
                # sat_per_product: list of P (T,) tensors
                # stack to (T, P), then multiply by per-product beta (P,)
                x_sat_tp = pt.stack(sat_per_product, axis=1)   # (T, P)

                if beta_time is not None:
                    if beta_pc is not None:
                        beta_t_j = beta_time[:, :, j]        # (T, P)
                    else:
                        beta_t_j = beta_time[:, None, j]     # (T, 1)
                    contrib_j = x_sat_tp * beta_t_j
                elif beta_pc is not None:
                    # Case A: per-product betas (P, C)
                    contrib_j = x_sat_tp * beta_pc[:, j][None, :]   # (T, P)
                elif beta_c is not None:
                    # Hierarchical family betas but no product dim
                    contrib_j = x_sat_tp * beta_c[j]   # (T, P)
                else:
                    contrib_j = x_sat_tp * betas[j]    # (T, P)

                sat_tensors.append(x_sat_tp[:, 0])   # store product-0 for synergies

            else:
                # Flat single-product
                x_sat = sat_per_product[0]   # (T,)
                sat_tensors.append(x_sat)

                if beta_time is not None:
                    contrib_j = beta_time[:, j] * x_sat
                elif beta_c is not None:
                    contrib_j = beta_c[j] * x_sat
                else:
                    contrib_j = betas[j] * x_sat

            channel_contribs.append(contrib_j)

        # ── Stack contributions ────────────────────────────────
        if P > 1:
            # Each contrib_j: (T, P) — stack on new axis 2 → (T, P, C)
            media_matrix_tpc = pm.Deterministic(
                "media_by_channel",
                pt.stack(channel_contribs, axis=2),    # (T, P, C)
            )
            media_by_product = pm.Deterministic(
                "media_by_product",
                pt.sum(media_matrix_tpc, axis=2),      # (T, P)
            )
        else:
            media_matrix = pm.Deterministic(
                "media_by_channel",
                pt.stack(channel_contribs, axis=1),    # (T, C)
            )
            media_total = pm.Deterministic(
                "media_total", pt.sum(media_matrix, axis=1)  # (T,)
            )

        # ══════════════════════════════════════════════════════
        # BLOCK 9 — CAMPAIGNS AS AMPLIFIERS (optional)
        # ══════════════════════════════════════════════════════
        campaign_effect = 0.0
        use_campaigns = bool(getattr(cfg, "use_campaigns", False))
        X_campaign = prep.get("X_campaign_scaled")
        campaign_cols = prep.get("campaign_cols", [])

        if use_campaigns and X_campaign is not None and len(campaign_cols) > 0:
            K = X_campaign.shape[1]  # number of campaigns
            X_camp = pt.as_tensor_variable(X_campaign[train_idx].astype(np.float64))  # (T, K)

            # Campaign families (similar to media families)
            campaign_families = []
            for col in campaign_cols:
                # Get family from schema or default to "generic"
                if prep.get("schema") and hasattr(prep["schema"], "campaign_cols"):
                    for cr in prep["schema"].campaign_cols:
                        if cr.column == col:
                            campaign_families.append(getattr(cr, "family", "generic"))
                            break
                    else:
                        campaign_families.append("generic")
                else:
                    campaign_families.append("generic")

            camp_fam_names = sorted(set(campaign_families))
            F_camp = len(camp_fam_names)
            camp_fam_to_idx = {f: i for i, f in enumerate(camp_fam_names)}
            camp_family_idx = np.array([camp_fam_to_idx[f] for f in campaign_families], dtype=int)

            # Campaign adstock parameters (two-timescale, family-pooled)
            camp_hl_slow_medians = np.array([3.0] * F_camp)  # Default half-lives
            camp_hl_sigma = np.array([0.5] * F_camp)

            # Family-level campaign decay priors
            camp_log_hl_slow_f = pm.Normal(
                "camp_log_hl_slow_f",
                mu=np.log(camp_hl_slow_medians),
                sigma=camp_hl_sigma,
                shape=F_camp,
            )
            camp_sigma_hl_f = pm.HalfNormal("camp_sigma_hl_f", sigma=camp_hl_sigma, shape=F_camp)
            camp_log_r_f = pm.Normal(
                "camp_log_r_f",
                mu=np.log(np.array([3.0] * F_camp)),  # ratio median
                sigma=np.full(F_camp, 0.4),
                shape=F_camp,
            )
            camp_w_mix_f = pm.Beta("camp_w_mix_f", alpha=2.0, beta=2.0, shape=F_camp)

            # Channel-level deviations
            camp_z_hl_k = pm.Normal("camp_z_hl_k", mu=0.0, sigma=1.0, shape=K)
            camp_fam_idx_t = pt.as_tensor_variable(camp_family_idx.astype(np.int64))

            camp_log_hl_slow_k = pm.Deterministic(
                "camp_log_hl_slow_k",
                camp_log_hl_slow_f[camp_fam_idx_t] + camp_sigma_hl_f[camp_fam_idx_t] * camp_z_hl_k,
            )
            camp_hl_slow_k = pm.Deterministic("camp_hl_slow_k", pt.exp(camp_log_hl_slow_k))
            camp_r_k = pm.Deterministic("camp_r_k", pt.exp(camp_log_r_f[camp_fam_idx_t]))
            camp_hl_fast_k = pm.Deterministic("camp_hl_fast_k", camp_hl_slow_k / camp_r_k)

            camp_rho_slow_k = pm.Deterministic(
                "camp_rho_slow_k", pt.exp(_LOG_HALF / camp_hl_slow_k),
            )
            camp_rho_fast_k = pm.Deterministic(
                "camp_rho_fast_k", pt.exp(_LOG_HALF / camp_hl_fast_k),
            )

            # Apply two-timescale adstock to each campaign
            camp_ads_list = []
            for k in range(K):
                x_camp_k = X_camp[:, k]  # (T,)
                x_camp_ads = two_timescale_adstock_scan(
                    x_camp_k,
                    camp_rho_slow_k[k],
                    camp_rho_fast_k[k],
                    camp_w_mix_f[camp_family_idx[k]],
                    reset=None  # Campaigns don't have reset logic in reference
                )
                camp_ads_list.append(x_camp_ads)

            camp_ads_matrix = pt.stack(camp_ads_list, axis=1)  # (T, K)

            # Campaign amplification coefficients (family-level)
            camp_amp_raw = pm.Normal("camp_amp_raw", mu=0.0, sigma=0.6, shape=F_camp)
            camp_amp_f = pm.Deterministic(
                "camp_amp_f",
                pt.log1p(pt.exp(camp_amp_raw)),  # softplus for positivity
            )

            # Apply amplification to media contributions
            # Media contributions are amplified by campaign effects
            if P > 1:
                # For multi-product, amplify each product's media
                camp_effect_per_product = []
                for p in range(P):
                    # Sum campaign effects across families, weighted by amplification
                    camp_sum_p = pt.zeros(T_train, dtype="float64")
                    for f in range(F_camp):
                        camp_in_fam = [k for k in range(K) if camp_family_idx[k] == f]
                        if camp_in_fam:
                            fam_camp_sum = pt.sum(pt.stack([camp_ads_matrix[:, k] for k in camp_in_fam], axis=1), axis=1)
                            camp_sum_p += camp_amp_f[f] * fam_camp_sum

                    # Amplify media by (1 + campaign_effect)
                    amplified_media_p = media_by_product[:, p] * (1.0 + camp_sum_p)
                    camp_effect_per_product.append(amplified_media_p - media_by_product[:, p])

                campaign_effect = pm.Deterministic(
                    "campaign_effect",
                    pt.stack(camp_effect_per_product, axis=1),  # (T, P)
                )
            else:
                # Single product
                camp_sum = pt.zeros(T_train, dtype="float64")
                for f in range(F_camp):
                    camp_in_fam = [k for k in range(K) if camp_family_idx[k] == f]
                    if camp_in_fam:
                        fam_camp_sum = pt.sum(pt.stack([camp_ads_matrix[:, k] for k in camp_in_fam], axis=1), axis=1)
                        camp_sum += camp_amp_f[f] * fam_camp_sum

                amplified_media = media_total * (1.0 + camp_sum)
                campaign_effect = pm.Deterministic(
                    "campaign_effect",
                    amplified_media - media_total,  # (T,)
                )

            logger.debug(f"  Campaigns as amplifiers enabled: {K} campaigns, {F_camp} families")
        else:
            campaign_effect = 0.0

        # ══════════════════════════════════════════════════════
        # BLOCK 10 — CROSS-PRODUCT HALOS (optional)
        # ══════════════════════════════════════════════════════
        halo_effect = 0.0
        use_halos = bool(getattr(cfg, "use_halos", False))

        if use_halos and P > 1:
            # Halo coefficients: source and receiver per product
            h_src = pm.HalfNormal("h_src", sigma=0.5, shape=P)  # source strength
            h_recv = pm.HalfNormal("h_recv", sigma=0.5, shape=P)  # receiver sensitivity

            # Compute media signal per product (sum across channels)
            media_signal_TP = media_by_product  # (T, P)

            # For each product p, halo effect is:
            # h_recv[p] * mean_{q!=p}(h_src[q] * media_signal[q,t])
            halo_per_product = []
            for p in range(P):
                # Sum of other products' contributions
                others_sum_T = pt.zeros(T_train, dtype="float64")
                for q in range(P):
                    if q != p:
                        others_sum_T += h_src[q] * media_signal_TP[:, q]

                # Average over number of other products
                denom = float(P - 1) if P > 1 else 1.0
                halo_p = h_recv[p] * (others_sum_T / denom)
                halo_per_product.append(halo_p)

            halo_effect = pm.Deterministic(
                "halo_effect",
                pt.stack(halo_per_product, axis=1),  # (T, P)
            )
            logger.debug(f"  Cross-product halos enabled: {P} products")
        else:
            halo_effect = 0.0

        # ══════════════════════════════════════════════════════
        # BLOCK 11 — CROSS-CHANNEL SYNERGIES
        # all channels in that family (averaged over products for P>1).
        # Then add pairwise synergy terms:
        #
        #   syn_term[t] = Σ_{f1,f2} syn[f1,f2] * hill_f[t,f1] * hill_f[t,f2]
        #
        # syn[f1,f2] ~ softplus(Normal(0, syn_sigma))  ≥ 0
        # ══════════════════════════════════════════════════════
        synergy_effect = 0.0

        if use_syn:
            # Build family-level Hill signals: average sat across channels per family
            family_signals = []
            for f_idx in range(F):
                ch_in_fam = [j for j in range(C) if family_idx[j] == f_idx]
                if not ch_in_fam:
                    # Empty family — use zeros matching the training period length
                    family_signals.append(pt.zeros(T_train, dtype="float64"))
                else:
                    fam_sats = pt.stack([sat_tensors[j] for j in ch_in_fam], axis=1)  # (T, n_f)
                    family_signals.append(pt.mean(fam_sats, axis=1))                  # (T,)

            SYN_SIGMA = 0.3
            syn_raw = pm.Normal("syn_raw", mu=0.0, sigma=SYN_SIGMA, shape=(F, F))
            syn_mat = pm.Deterministic(
                "syn",
                pt.log1p(pt.exp(syn_raw)),   # softplus — ensures non-negative synergies
            )

            syn_components = []
            for f1 in range(F):
                for f2 in range(F):
                    syn_components.append(
                        syn_mat[f1, f2] * family_signals[f1] * family_signals[f2]
                    )
            synergy_effect = pm.Deterministic(
                "synergy_effect",
                pt.sum(pt.stack(syn_components, axis=1), axis=1),   # (T,)
            )
            logger.debug(
                f"  Cross-channel synergies enabled: {F}×{F} family pairs"
            )

        # ══════════════════════════════════════════════════════
        # BLOCK 12 — EXPECTED VALUE (mu)
        #
        # Flat (P=1):
        #   mu[t] = baseline[t] + seasonality[t] + media_total[t]
        #         + control_effect[t] + base_effect[t]
        #         + macro_effect[t] + event_effect[t]
        #         + synergy_effect[t]
        #
        # Multi-product (P>1):
        #   mu[t,p] = intercept_p[p] + baseline[t] + seasonality[t]
        #           + media_by_product[t,p]
        #           + (shared effects broadcast to (T,P))
        # ══════════════════════════════════════════════════════
        if P > 1:
            # Sum all shared (T,) effects, then broadcast to (T, P) via [:, None].
            # Each effect is either 0.0 (Python float, inactive) or a PyTensor (T,)
            # tensor.  We add them all together — broadcasting handles both cases.
            shared_effects = (
                control_effect + base_effect + macro_effect + event_effect
                + synergy_effect + seasonality_dow + outlier_effect
            )
            mu = pm.Deterministic(
                "mu",
                baseline[:, None]           # (T, 1) → (T, P)
                + seasonality[:, None]
                + intercept_p[None, :]      # (1, P) → (T, P)
                + media_by_product          # (T, P)
                + campaign_effect           # (T, P) - already amplified media
                + halo_effect               # (T, P) - cross-product halos
                + (shared_effects[:, None] if hasattr(shared_effects, "shape") else shared_effects),
            )
        else:
            mu = pm.Deterministic(
                "mu",
                baseline + seasonality + seasonality_dow + media_total
                + campaign_effect + halo_effect
                + control_effect + base_effect + macro_effect
                + event_effect + synergy_effect + outlier_effect,
            )

        # ══════════════════════════════════════════════════════
        # BLOCK 13 — LIKELIHOOD
        # ══════════════════════════════════════════════════════
        _needs_raw = dominant_cfg.likelihood in ("gamma_obs", "negative_binomial")
        build_likelihood(
            name      = "y_obs",
            mu        = mu,
            sigma_y   = sigma_y,
            nu        = nu,
            observed  = y_scaled,
            config    = dominant_cfg,
            y_raw_obs = prep["y_raw"][train_idx] if _needs_raw else None,
            y_mu_val  = float(np.asarray(prep["y_mu"]).flat[0]) if _needs_raw else None,
            y_std_val = float(np.asarray(prep["y_std"]).flat[0]) if _needs_raw else None,
        )

    return model


def build_marginal_model(
    prep              : Dict[str, Any],
    cfg               : ModelConfig,
    focal_ch          : int,
    focal_ads         : str,
    focal_sat         : str,
    channel_prior_map : Optional[Dict[str, "ChannelPriorConfig"]] = None,
) -> pm.Model:
    """
    Stage 0 wrapper: builds a full C-channel model where all channels use
    cfg defaults EXCEPT channel focal_ch which uses (focal_ads, focal_sat).

    Passes use_scan_adstock=False to keep Stage 0 fast — scan compilation
    overhead is significant when fitting C×6 models in quick succession.
    The full model (build_mmm with default use_scan_adstock=True) uses the
    correct state-space scan path.
    """
    return build_mmm(
        prep              = prep,
        cfg               = cfg,
        channel_specs     = None,
        focal_channel     = focal_ch,
        focal_adstock     = focal_ads,
        focal_saturation  = focal_sat,
        channel_prior_map = channel_prior_map,
        use_scan_adstock  = False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Channel transform summary table
# ─────────────────────────────────────────────────────────────────────────────

def build_channel_transform_summary(
    best          : Dict[str, Any],
    channel_specs : Dict[int, ChannelTransformSpec],
) -> pd.DataFrame:
    """
    Extracts per-channel transform selections and posterior parameter
    estimates from the best model trace and returns a summary DataFrame.

    One row per channel. Columns:
        channel            — column name in the data
        adstock            — winning adstock type
        saturation         — winning saturation type
        use_lag            — whether lag was applied
        max_lag            — maximum lag allowed
        lag_mode           — most probable lag (mode of posterior); NaN if no lag
        lam_mean/sd        — adstock decay (geometric λ) or Weibull scale
        k_shape_mean/sd    — Weibull shape parameter; NaN for geometric
        alpha_sat_mean/sd  — saturation steepness (softplus / exponential)
        kappa_mean/sd      — Hill half-saturation; NaN for other sat types
        k_logistic_mean/sd — logistic steepness; NaN for other sat types
        x0_mean/sd         — logistic inflection point; NaN for other sat types
        beta_mean/sd       — channel contribution weight (posterior)
    """
    trace = best["metrics"]["trace"]
    post  = trace.posterior

    def _per_ch(prefix: str, j: int) -> Tuple[float, float]:
        """Posterior (mean, std) for variable named f'{prefix}_{j}'."""
        key = f"{prefix}_{j}"
        if key not in post.data_vars:
            return (np.nan, np.nan)
        arr = post[key].stack(sample=("chain", "draw")).values.ravel()
        return (float(arr.mean()), float(arr.std()))

    def _lag_mode(j: int) -> float:
        """
        Mode of the lag distribution for channel j.
        lag_mode_{j} is a Deterministic = argmax(lag_w_{j}).
        Falls back to lag_w_{j} argmax if the deterministic is absent.
        Returns NaN if lag was not applied to this channel.
        """
        det_key = f"lag_mode_{j}"
        if det_key in post.data_vars:
            arr = post[det_key].stack(sample=("chain", "draw")).values.ravel()
            return float(np.median(arr))
        w_key = f"lag_w_{j}"
        if w_key in post.data_vars:
            arr = post[w_key].stack(sample=("chain", "draw")).values  # (n_lags, N)
            modes = np.argmax(arr, axis=0)                            # (N,)
            return float(np.median(modes))
        return np.nan

    def _beta(j: int) -> Tuple[float, float]:
        if "betas" in post.data_vars:
            arr = post["betas"].stack(sample=("chain", "draw")).values
            col = arr[j] if arr.ndim > 1 else arr
            return (float(col.mean()), float(col.std()))
        key = f"beta_ch{j}"
        if key in post.data_vars:
            arr = post[key].stack(sample=("chain", "draw")).values.ravel()
            return (float(arr.mean()), float(arr.std()))
        return (np.nan, np.nan)

    rows = []
    for j, spec in sorted(channel_specs.items()):
        lam_m,  lam_s  = _per_ch("lam",        j)
        ksh_m,  ksh_s  = _per_ch("k_shape",     j)
        alp_m,  alp_s  = _per_ch("alpha_sat",   j)
        kap_m,  kap_s  = _per_ch("kappa",       j)
        klog_m, klog_s = _per_ch("k_logistic",  j)
        x0_m,   x0_s   = _per_ch("x0",          j)
        b_m,    b_s    = _beta(j)

        rows.append({
            "channel"          : spec.channel_name,
            "adstock"          : spec.adstock_type,
            "saturation"       : spec.saturation,
            "use_lag"          : spec.use_lag,
            "max_lag"          : spec.max_lag if spec.use_lag else 0,
            "lag_mode"         : _lag_mode(j),
            "lam_mean"         : round(lam_m,  4),
            "lam_sd"           : round(lam_s,  4),
            "k_shape_mean"     : round(ksh_m,  4),
            "k_shape_sd"       : round(ksh_s,  4),
            "alpha_sat_mean"   : round(alp_m,  4),
            "alpha_sat_sd"     : round(alp_s,  4),
            "kappa_mean"       : round(kap_m,  4),
            "kappa_sd"         : round(kap_s,  4),
            "k_logistic_mean"  : round(klog_m, 4),
            "k_logistic_sd"    : round(klog_s, 4),
            "x0_mean"          : round(x0_m,   4),
            "x0_sd"            : round(x0_s,   4),
            "beta_mean"        : round(b_m, 4),
            "beta_sd"          : round(b_s, 4),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_model(
    model     : pm.Model,
    cfg       : ModelConfig,
    fast_mode : bool = False,
) -> Tuple[az.InferenceData, float]:
    """Samples the model using NUTS."""
    draws       = cfg.fast_draws  if fast_mode else cfg.draws
    tune        = cfg.fast_tune   if fast_mode else cfg.tune
    chains      = cfg.fast_chains if fast_mode else cfg.chains
    tgt         = 0.90 if fast_mode else cfg.target_accept
    progressbar = False if fast_mode else cfg.progressbar

    t0 = time.perf_counter()
    with model:
        try:
            init_method = "advi+adapt_diag" if cfg.init == "advi" else "jitter+adapt_diag"
            trace = pm.sample(
                draws                = draws,
                tune                 = tune,
                chains               = chains,
                target_accept        = tgt,
                init                 = init_method,
                progressbar          = progressbar,
                return_inferencedata = True,
                random_seed          = GLOBAL_SEED,
                idata_kwargs         = {"log_likelihood": True},
                nuts_sampler         = "numpyro",
            )
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            raise

    wall = time.perf_counter() - t0
    logger.debug(f"  Sampling complete in {wall:.1f}s")
    return trace, wall
