# pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# The main engine that runs the full 4-stage model-fitting pipeline.
#
# You don't call this directly — main.py calls run_full_pipeline() for you.
#
# The 4 stages:
#   Stage 0  Per-channel adstock×saturation scan (which transform fits best?)
#   Stage 1  MAP pre-screen to shortlist configs quickly
#   Stage 1b Fast MCMC scan on the shortlisted configs
#   Stage 2  Full MCMC refit of the winning config
#
# After fitting, pipeline.py saves trace.nc + prep.pkl so you can run
# python analyze.py <command> without re-fitting the model.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import DataConfig, ModelConfig, build_full_grid, build_pruned_grid, build_channel_specs
from data_prep import ingest_and_preprocess
from priors import ChannelPriorConfig, validate_channel_prior_map
from pipeline_stages import (
    run_hybrid_fast_scan,
    run_stage0_marginal_scan,
    run_stage1_map_prescreen,
    run_stage1b_fast_mcmc,
    run_stage2_full_refit,
)
from analysis import compute_channel_contributions, compute_component_contributions, compute_per_product_contributions
from plots import generate_all_plots
from export import export_all_results
from metrics import quality_flag

logger = logging.getLogger("MMM")


# ─────────────────────────────────────────────────────────────────────────────
# Stage timer
# ─────────────────────────────────────────────────────────────────────────────

class StageTimer:
    """
    Accumulates wall-clock timing for each named pipeline stage.

    Usage
    -----
    timer = StageTimer()
    with timer("stage_0", n_models=C*6):
        ...  # stage work

    timer.summary()          -> list of dicts
    timer.to_dataframe()     -> pd.DataFrame
    """

    def __init__(self):
        self._records: List[Dict] = []

    @contextmanager
    def __call__(self, stage_name: str, n_models: int = 0):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._records.append({
                "stage"       : stage_name,
                "n_models"    : n_models,
                "duration_s"  : round(elapsed, 2),
                "duration_min": round(elapsed / 60.0, 2),
            })
            logger.info(
                f"[TIMER] {stage_name:<20s} "
                f"{elapsed/60:.1f} min "
                f"({elapsed:.1f} s)"
                + (f"  [{n_models} models]" if n_models else "")
            )

    def summary(self) -> List[Dict]:
        total = sum(r["duration_s"] for r in self._records)
        return self._records + [{
            "stage"       : "total",
            "n_models"    : sum(r["n_models"] for r in self._records),
            "duration_s"  : round(total, 2),
            "duration_min": round(total / 60.0, 2),
        }]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.summary())


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline state persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save_pipeline_state(best: Dict, prep: Dict, out_dir: "Path") -> None:
    """
    Persist trace, prep, and model config so analyze.py can load the fitted
    model without re-running the full pipeline.

    Writes three files to out_dir:
      trace.nc      — ArviZ posterior trace (NetCDF format)
      prep.pkl      — preprocessed data arrays + metadata
      best_cfg.json — lightweight model config (human-readable JSON)
    """
    import pickle
    import json
    import arviz as az

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Posterior trace → NetCDF
    trace      = best["trace"]
    trace_path = out_dir / "trace.nc"
    logger.info("[STATE] Saving posterior trace...")
    az.to_netcdf(trace, str(trace_path))
    logger.info(f"  trace.nc  — {trace_path.stat().st_size / 1e6:.1f} MB")

    # 2. Prep dict (numpy arrays + metadata) → pickle
    # The prep dict never stores the trace directly, but guard anyway.
    prep_save  = {k: v for k, v in prep.items() if k != "trace"}
    prep_path  = out_dir / "prep.pkl"
    with open(prep_path, "wb") as fh:
        pickle.dump(prep_save, fh, protocol=4)
    logger.info(f"  prep.pkl  — {prep_path.stat().st_size / 1e6:.1f} MB")

    # 3. Best model cfg → JSON (lightweight, human-readable)
    cfg      = best["cfg"]
    # Serialize per-channel transform specs so standalone scripts know each
    # channel's Stage-0-winning adstock+saturation (not just the global default).
    _channel_specs = best.get("channel_specs", {})
    _specs_list = [
        {
            "channel_idx" : j,
            "channel_name": spec.channel_name,
            "adstock_type": spec.adstock_type,
            "saturation"  : spec.saturation,
            "max_lag"     : spec.max_lag,
            "use_lag"     : spec.use_lag,
        }
        for j, spec in sorted(_channel_specs.items())
    ]
    cfg_data = {
        "adstock_type"      : cfg.adstock_type,
        "saturation"        : cfg.saturation,
        "fourier_order"     : cfg.fourier_order,
        "max_lag"           : cfg.max_lag,
        "baseline_type"     : getattr(cfg, "baseline_type",      "linear_trend"),
        "response_transform": getattr(cfg, "response_transform", "log1p"),
        "use_hierarchical"  : getattr(cfg, "use_hierarchical",   False),
        "draws"             : cfg.draws,
        "tune"              : cfg.tune,
        "chains"            : cfg.chains,
        "channel_specs"     : _specs_list,   # per-channel Stage 0 winners
    }
    cfg_path = out_dir / "best_cfg.json"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg_data, fh, indent=2)
    logger.info(f"  best_cfg.json saved")


def load_pipeline_state(out_dir: "Path"):
    """
    Load the model state saved by _save_pipeline_state() after a pipeline run.

    Returns
    -------
    best : dict  — keys: trace, metrics (with trace), cfg (SimpleNamespace),
                   prep, channel_specs (per-channel Stage 0 winners)
    prep : dict  — preprocessed data arrays and metadata
    """
    import pickle, json, arviz as az
    from types import SimpleNamespace
    from config import ChannelTransformSpec

    out_dir = Path(out_dir)
    trace_path = out_dir / "trace.nc"
    prep_path  = out_dir / "prep.pkl"
    cfg_path   = out_dir / "best_cfg.json"

    if not trace_path.exists():
        raise FileNotFoundError(
            f"trace.nc not found in {out_dir}. Run main.py first."
        )
    if not prep_path.exists():
        raise FileNotFoundError(
            f"prep.pkl not found in {out_dir}. Run main.py first."
        )

    logger.info(f"[LOAD] trace.nc  <- {trace_path}")
    trace = az.from_netcdf(str(trace_path))

    logger.info(f"[LOAD] prep.pkl  <- {prep_path}")
    with open(prep_path, "rb") as fh:
        prep = pickle.load(fh)

    cfg_dict = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as fh:
            cfg_dict = json.load(fh)
    else:
        logger.warning("[LOAD] best_cfg.json not found — using defaults")

    channel_specs = {
        s["channel_idx"]: ChannelTransformSpec(
            channel_idx  = s["channel_idx"],
            channel_name = s["channel_name"],
            adstock_type = s.get("adstock_type", cfg_dict.get("adstock_type", "geometric")),
            saturation   = s.get("saturation",   cfg_dict.get("saturation",   "softplus")),
            max_lag      = s.get("max_lag", 0),
            use_lag      = s.get("use_lag", False),
        )
        for s in cfg_dict.get("channel_specs", [])
    }

    best = {
        "trace"        : trace,
        "metrics"      : {"trace": trace},
        "cfg"          : SimpleNamespace(**cfg_dict),
        "prep"         : prep,
        "channel_specs": channel_specs,
    }
    return best, prep


def run_full_pipeline(
    csv_path             : str,
    # Generic default — MUST match the actual response column in your CSV.
    response_col         : str        = "revenue",
    spend_prefix         : str        = "spends_",
    media_cols           : Optional[List[str]] = None,
    channel_variable_map : Optional[Dict[str, str]] = None,
    date_col             : str        = "date",
    control_cols         : List[str]  = None,
    min_date             : Optional[str] = None,
    max_date             : Optional[str] = None,
    # False = MM/DD/YYYY (universal default). Set True only for DD/MM/YYYY.
    dayfirst             : bool       = False,
    output_dir           : str        = "mmm_outputs",
    # Dataset frequency — drives Fourier period. "weekly" | "daily" | "monthly"
    frequency            : str        = "weekly",
    holdout_periods      : int        = 0,
    use_hierarchical     : bool       = False,
    top_n_map            : int        = 20,
    top_k_refit          : int        = 4,
    draws_full           : int        = 1800,
    tune_full            : int        = 800,
    chains_full          : int        = 2,
    target_accept        : float      = 0.95,
    baseline_type        : str        = "linear_trend",
    ranking_method       : str        = "lexicographic",
    rank_weights         : Optional[Dict[str, float]] = None,
    use_hybrid_fast_scan : bool       = False,
    fast_scan_trials     : int        = 80,
    fast_scan_timeout_sec: int        = 240,
    runtime_mode         : str        = "fast_20m",
    max_quality_flag     : int        = 1,
    reject_if_all_bad    : bool       = True,
    skip_stage0          : bool       = False,
    skip_map             : bool       = False,
    use_lag              : bool       = False,
    global_max_lag       : int        = 0,
    channel_prior_map    : Optional[Dict[str, "ChannelPriorConfig"]] = None,
    schema               : Optional[Any] = None,
    response_transform   : str = "log1p",
    # None = auto-prompt when daily data is detected (interactive only).
    # True/False = explicit override.
    use_dow_effects      : Optional[bool] = None,
    # ── Hierarchical model options ──────────────────────────
    use_two_timescale_adstock : bool  = False,
    use_synergies             : bool  = False,
    use_campaigns             : bool  = False,
    use_halos                 : bool  = False,
    use_global_hill           : bool  = False,
    use_time_varying_betas    : bool  = False,
    use_dynamic_saturation    : bool  = False,
    enable_strict_validation  : bool  = True,
    enable_outlier_detection  : bool  = True,
    enable_collinearity_check : bool  = True,
    enable_stationarity_test  : bool  = True,
    family_beta_sigma         : float = 0.5,
    channel_beta_sigma        : float = 0.3,
    product_beta_sigma        : float = 0.5,
    # Long-format auto-pivot — set via data section of YAML
    input_format              : str   = "wide",
    product_col               : str   = "Product",
    long_response_col         : str   = "",
    long_media_cols           : Optional[Dict[str, Any]] = None,
    long_spend_cols           : Optional[Dict[str, Any]] = None,  # spend cols for ROI when modelling impressions
    # Family adstock half-life overrides — populated from YAML priors.families.
    # Keys are family names ("search", "social", "display", …).
    # Values are ChannelFamilyConfig objects.
    # When non-empty, merged on top of DEFAULT_FAMILY_CONFIGS in model_builder.py.
    family_configs            : Optional[Dict[str, Any]] = None,
    # ── Optimisation settings ───────────────────────────────
    # These can also be set via the optimisation: section of a YAML config.
    # total_budget  : actual budget in raw units for budget_period (None = use observed mean)
    # budget_period : "per_period" | "monthly" | "quarterly" | "yearly" |
    #                 "training_total" | "training_mean"
    # target_response : desired response in original units for target_period (None = observed mean)
    # target_period   : same options as budget_period
    opt_config                : Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Full 4-stage Bayesian MMM pipeline with per-channel transforms.

    ┌─────────────────────────────────────────────────────────────┐
    │  STAGE 0  Marginal contribution scan                        │
    │           C channels × 8 combos × fast MCMC                │
    │           → best (adstock, sat) per channel                 │
    ├─────────────────────────────────────────────────────────────┤
    │  STAGE 1  MAP pre-screen                                    │
    │           Prune grid → top 20 by MAP-MAPE                   │
    ├─────────────────────────────────────────────────────────────┤
    │  STAGE 1b Fast MCMC scan                                    │
    │           Top 20 configs × fast draws, 1 chain              │
    │           Per-channel specs applied                          │
    │           → top K survivors                                  │
    ├─────────────────────────────────────────────────────────────┤
    │  STAGE 2  Full refit                                        │
    │           Top K × full draws / chains                        │
    │           Per-channel specs applied to final model           │
    │           → channel_transform_summary table attached         │
    └─────────────────────────────────────────────────────────────┘

    Parameters (new)
    ────────────────
    use_lag        : if True, a discrete lag shift is applied before adstock
                     for every channel.  Each channel independently learns its
                     best lag via a Categorical prior over [0, global_max_lag].
    global_max_lag : upper bound on the lag (in the dataset's native time unit:
                     weeks for weekly data, days for daily data).  Ignored when
                     use_lag=False.

    Returns
    -------
    best : dict with keys cfg, metrics, trace, prep, channel_transform_summary
    """
    t_pipeline_start = time.perf_counter()
    timer = StageTimer()

    profile     = (runtime_mode or "standard").strip().lower()
    est_runtime = "~20-30 min target" if profile == "fast_20m" else "~60+ min (depends on settings)"

    logger.info("=" * 70)
    logger.info(f"  BAYESIAN MARKET MIX MODEL - PRODUCTION PIPELINE ({est_runtime})")
    logger.info("=" * 70)
    logger.info(f"  CSV          : {csv_path}")
    logger.info(f"  Response     : {response_col}")
    logger.info(f"  Frequency    : {frequency} | dayfirst={dayfirst} | holdout={holdout_periods}")
    logger.info(f"  Spend prefix : {spend_prefix}")
    logger.info(f"  Media cols   : {media_cols if media_cols else '[prefix-based auto]'}")
    logger.info(f"  Output dir   : {output_dir}")
    logger.info(f"  Hierarchical : {use_hierarchical} | two-timescale adstock: {use_two_timescale_adstock} | synergies: {use_synergies}")
    logger.info(f"  Baseline     : {baseline_type}")
    logger.info(f"  Ranking      : {ranking_method}")
    logger.info(f"  Hybrid fast  : {use_hybrid_fast_scan}")
    logger.info(f"  Runtime mode : {runtime_mode}")
    logger.info(f"  Quality gate : accept quality_flag <= {max_quality_flag} | reject_if_all_bad={reject_if_all_bad}")
    logger.info(f"  Skip Stage 0 : {skip_stage0}")
    logger.info(f"  Skip MAP     : {skip_map}")
    logger.info(f"  Use lag      : {use_lag}  (max={global_max_lag} periods)" if use_lag else f"  Use lag      : False")
    logger.info(f"  Time-varying betas : {use_time_varying_betas}")
    logger.info(f"  Dynamic saturation: {use_dynamic_saturation}")
    logger.info(f"  Strict validation : {enable_strict_validation}")
    logger.info(f"  Outlier detection : {enable_outlier_detection}")
    logger.info(f"  Stationarity test : {enable_stationarity_test}")
    logger.info(f"  Channel priors: {'custom (' + str(len(channel_prior_map)) + ' channels)' if channel_prior_map else 'all defaults'}")
    logger.info("=" * 70)

    # ── DataConfig ────────────────────────────────────────────
    data_cfg = DataConfig(
        csv_path        = csv_path,
        date_col        = date_col,
        response_col    = response_col,
        spend_prefix    = spend_prefix,
        media_cols      = media_cols or [],
        control_cols      = control_cols or [],
        min_date          = min_date,
        max_date          = max_date,
        dayfirst          = dayfirst,
        output_dir        = output_dir,
        frequency         = frequency,
        holdout_periods   = holdout_periods,
        # Long-format auto-pivot settings (from YAML data section)
        input_format      = input_format,
        product_col       = product_col,
        long_response_col = long_response_col,
        long_media_cols   = long_media_cols   or {},
        long_spend_cols   = long_spend_cols   or {},
    )

    # ── Base ModelConfig ──────────────────────────────────────
    base_cfg = ModelConfig(
        max_lag          = 8,
        fourier_order    = 2,
        adstock_type     = "geometric",
        saturation       = "softplus",
        use_trend        = True,
        baseline_type    = baseline_type,
        piecewise_knots  = 3,
        use_hierarchical = use_hierarchical,
        use_controls     = bool(control_cols),
        target_accept    = target_accept,
        draws            = draws_full,
        tune             = tune_full,
        chains           = chains_full,
        fast_draws       = 120,
        fast_tune        = 120,
        fast_chains      = 1,
        init             = "auto",
        progressbar      = True,
        use_lag                   = use_lag,
        global_max_lag            = global_max_lag,
        response_transform        = response_transform,
        use_two_timescale_adstock = use_two_timescale_adstock,
        use_synergies             = use_synergies,
        use_campaigns             = use_campaigns,
        use_halos                 = use_halos,
        use_global_hill           = use_global_hill,
        use_time_varying_betas    = use_time_varying_betas,
        use_dynamic_saturation    = use_dynamic_saturation,
        enable_strict_validation  = enable_strict_validation,
        enable_outlier_detection  = enable_outlier_detection,
        enable_collinearity_check = enable_collinearity_check,
        enable_stationarity_test  = enable_stationarity_test,
        family_beta_sigma         = family_beta_sigma,
        channel_beta_sigma        = channel_beta_sigma,
        product_beta_sigma        = product_beta_sigma,
        family_configs            = family_configs or {},
    )

    # ── Runtime profile switch ────────────────────────────────
    if (runtime_mode or "").strip().lower() == "fast_20m":
        _pre_draws  = int(draws_full)
        _pre_tune   = int(tune_full)
        _pre_top_n  = int(top_n_map)
        top_n_map   = min(int(top_n_map), 8)
        top_k_refit = min(int(top_k_refit), 1)
        # Stage 2 must still be a real MCMC refit (>=2 chains) or quality gating is meaningless.
        # Keep the "fast_20m" spirit by capping the *final* refit to a minimum viable setup.
        draws_full  = min(int(draws_full), 800)
        tune_full   = min(int(tune_full), 500)
        chains_full = max(int(chains_full), 2)
        base_cfg.fast_draws  = min(base_cfg.fast_draws, 60)
        base_cfg.fast_tune   = min(base_cfg.fast_tune, 60)
        base_cfg.fast_chains = 1
        base_cfg.draws       = draws_full
        base_cfg.tune        = tune_full
        base_cfg.chains      = chains_full
        skip_map             = True
        if draws_full < _pre_draws or tune_full < _pre_tune or top_n_map < _pre_top_n:
            logger.info(
                "[CONFIG] fast_20m caps applied — "
                "draws: %d→%d, tune: %d→%d, top_n_map: %d→%d. "
                "YAML sampling.draws/tune values are overridden by runtime_mode='fast_20m'.",
                _pre_draws, draws_full, _pre_tune, tune_full, _pre_top_n, top_n_map,
            )

    # Stage 2 must always have >=2 chains (R-hat needs it).
    if int(chains_full) < 2:
        logger.warning(f"[CONFIG] chains_full={chains_full} < 2; forcing chains_full=2 for Stage 2 diagnostics.")
        chains_full = 2
        base_cfg.chains = 2

    # ── Day-of-week prompt (daily data only) ─────────────────
    # When frequency=="daily" and use_dow_effects is None, prompt interactively.
    # Non-interactive environments (batch/CI) default to False.
    _dow_resolved = use_dow_effects
    if frequency.lower() == "daily" and _dow_resolved is None:
        import sys
        if sys.stdin.isatty():
            try:
                _resp = input(
                    "\n[DAILY DATA DETECTED] Include day-of-week effects?\n"
                    "  Adds 6 Fourier features for within-week seasonality (Mon-Sun).\n"
                    "  Recommended: Yes for retail/e-commerce/social; No for brand/awareness metrics.\n"
                    "  Include day-of-week effects? [Y/n]: "
                ).strip().lower()
                _dow_resolved = (_resp != "n")
                logger.info(f"[DOW] User selected: use_dow_effects={_dow_resolved}")
            except (EOFError, KeyboardInterrupt):
                _dow_resolved = False
                logger.info("[DOW] Non-interactive — DOW effects disabled.")
        else:
            _dow_resolved = False
            logger.info(
                "[DOW] Daily data detected. DOW effects disabled in non-interactive mode. "
                "Pass use_dow_effects=True to enable."
            )
    base_cfg.use_dow_effects = bool(_dow_resolved) if _dow_resolved is not None else False

    # ── Daily data + family defaults warning ──────────────────
    # DEFAULT_FAMILY_CONFIGS in config.py are calibrated in WEEKS.
    # If frequency is "daily" and a channel family is not in family_configs,
    # the default hl_slow_median (e.g. 2.5 for search) will be treated as
    # 2.5 DAYS — 7x shorter than intended.  Warn early before sampling starts.
    if frequency.lower() == "daily" and schema is not None:
        from config import DEFAULT_FAMILY_CONFIGS as _DFC
        _used_families = {
            getattr(mc, "family", None)
            for mc in getattr(schema, "media_cols", [])
            if getattr(mc, "family", None)
        }
        _overridden = set((family_configs or {}).keys())
        _missing = _used_families - _overridden
        if _missing:
            _examples = ", ".join(
                f"{f}→hl_slow_median={_DFC[f].hl_slow_median} days (intend {_DFC[f].hl_slow_median*7:.0f}?)"
                for f in sorted(_missing) if f in _DFC
            )
            logger.warning(
                "[CONFIG] Daily data detected. The following channel families are NOT "
                "overridden in priors.families: %s. DEFAULT_FAMILY_CONFIGS are calibrated "
                "in WEEKS — for daily data they will be treated as DAYS (e.g. %s). "
                "Add missing families to priors.families with hl_slow_median in days "
                "(typical: search=14, display=21, social=7).",
                sorted(_missing), _examples or "see config.py DEFAULT_FAMILY_CONFIGS",
            )

    # ── Ingest & Preprocess ───────────────────────────────────
    logger.info("\n[PREP] Loading and preprocessing data...")
    t0   = time.perf_counter()
    prep = ingest_and_preprocess(data_cfg, base_cfg, channel_variable_map=channel_variable_map, schema=schema)
    C    = prep["C"]
    logger.info(f"[PREP] Done in {time.perf_counter()-t0:.1f}s")

    # ── Early hard rejection on dataset gates ─────────────────────────
    # These gates are also reflected in quality_flag(), but failing fast
    # here saves expensive sampling when the run is not eligible anyway.
    min_T = int(prep.get("min_T_for_production", 100) or 100)
    T = int(prep.get("T") or 0)
    col_max = prep.get("collinearity_max_offdiag_abs")
    col_thr = float(prep.get("collinearity_threshold", 0.85) or 0.85)
    col_max_valid = col_max is not None and not np.isnan(col_max)
    dataset_gate_failed = (T > 0 and T < min_T) or (col_max_valid and float(col_max) > col_thr)
    if dataset_gate_failed and int(max_quality_flag) <= 1 and bool(reject_if_all_bad):
        col_msg = f"max|corr_offdiag|={col_max:.3f} > {col_thr:.3f}" if col_max_valid else "collinearity=N/A"
        raise RuntimeError(
            f"Dataset failed quality gates for production (T={T} < {min_T} "
            f"or {col_msg}). "
            "Fix data length/collinearity or relax gates (max_quality_flag=2) for exploration."
        )

    # FIX-1: Validate per-channel prior map against the actual spend columns
    if channel_prior_map:
        validate_channel_prior_map(channel_prior_map, prep["spend_cols"])

    # ── Hybrid fast scan (optional) ───────────────────────────
    hybrid_scan                 = {"enabled": False}
    candidate_combo_per_channel = None
    if use_hybrid_fast_scan:
        logger.info("\n[HYBRID] Running fast scan pre-step...")
        hybrid_scan = run_hybrid_fast_scan(
            prep=prep, data_cfg=data_cfg, base_cfg=base_cfg,
            n_trials=fast_scan_trials, timeout_sec=fast_scan_timeout_sec,
        )
        if hybrid_scan.get("enabled", False):
            candidate_combo_per_channel = hybrid_scan.get("candidate_combo_per_channel")

    # ── Stage 0 ───────────────────────────────────────────────
    # FIX (Issue 5): Auto-skip Stage 0 in hierarchical multi-product mode.
    #
    # Stage 0 fits C × 6 full models to find the best adstock×saturation combo
    # per channel.  In hierarchical multi-product mode (use_hierarchical=True,
    # P > 1) this is both expensive and misleading:
    #
    #   a) Cost: each of the C×6 fits now carries P products + full family
    #      pooling parameters → roughly 2× more parameters and data.
    #
    #   b) Validity: when adstock parameters are pooled within families (Block 6
    #      of model_builder), the "best combo independently" from Stage 0 may
    #      not remain best once family constraints are applied.  The Stage 0
    #      ranking is based on marginal fits that don't reflect the shared prior.
    #
    # When auto-skipping, we use the base_cfg adstock+saturation for all
    # channels (geometric + softplus by default), which is already a sensible
    # starting point.  Users can still force Stage 0 by passing skip_stage0=False
    # explicitly, in which case we honour that override with a warning.
    P_data = prep.get("P", 1)
    _hier_multiproduct = use_hierarchical and P_data > 1
    _effective_skip_stage0 = skip_stage0

    if _hier_multiproduct and not skip_stage0:
        logger.info(
            "[S0] AUTO-SKIP: hierarchical multi-product mode detected (P=%d). "
            "Stage 0 per-channel scan would be ~%dx more expensive and its "
            "rankings are less reliable under family adstock pooling.  "
            "Using base_cfg defaults (geometric+softplus) for all channels.  "
            "Pass skip_stage0=False explicitly to force Stage 0.", P_data, P_data
        )
        _effective_skip_stage0 = True

    df_stage0 = pd.DataFrame()
    if not _effective_skip_stage0:
        from config import ADSTOCK_SAT_COMBOS as _ALL_COMBOS
        _n_s0_models = C * len(_ALL_COMBOS)   # C channels × up to 8 combos (pruned by fast scan)
        with timer("stage_0", n_models=_n_s0_models):
            best_combo, stage0_rows = run_stage0_marginal_scan(
                prep=prep, base_cfg=base_cfg,
                candidate_combo_per_channel=candidate_combo_per_channel,
                ranking_method=ranking_method, rank_weights=rank_weights,
                channel_prior_map=channel_prior_map,
            )
            df_stage0 = pd.DataFrame(stage0_rows)
        logger.info(f"[S0] Complete — see timer above")
    else:
        if not _hier_multiproduct:
            logger.info("[S0] Skipped - using hybrid fast-scan winners if available")
        if hybrid_scan.get("enabled", False) and hybrid_scan.get("mapped_best_combo"):
            best_combo = hybrid_scan["mapped_best_combo"]
        else:
            best_combo = {j: (base_cfg.adstock_type, base_cfg.saturation) for j in range(C)}

    # ── Build per-channel transform specs ────────────────────
    # Converts Stage 0 winners into ChannelTransformSpec objects,
    # one per channel, optionally carrying the user's lag settings.
    channel_specs = build_channel_specs(
        best_combo     = best_combo,
        spend_cols     = prep["spend_cols"],
        use_lag        = use_lag,
        global_max_lag = global_max_lag,
    )
    logger.info("\n[SPECS] Per-channel transform assignments:")
    for j, spec in sorted(channel_specs.items()):
        logger.info(f"  [{j}] {spec.channel_name:<30s} -> {spec.label()}")
    # ── Build Full Grid ───────────────────────────────────────
    logger.info("\n[GRID] Building full hyperparameter grid...")
    full_grid = build_full_grid(base_cfg)

    # In hierarchical multi-product mode with skip_stage0=True, channel_specs
    # overrides adstock+saturation for all channels anyway (all set to geometric+softplus
    # from the default best_combo).  Different grid configs with the same (lag, fo) but
    # different adstock/sat will therefore produce identical models — deduping by
    # (lag, fo) only (stage0_ran=True logic) avoids running 8 identical configs.
    # This reduces Stage 1b from 8 configs to 6 (3 lag × 2 fo).
    _grid_stage0_ran = (not _effective_skip_stage0) or _hier_multiproduct
    pruned_grid = build_pruned_grid(full_grid, best_combo,
                                    stage0_ran=_grid_stage0_ran)
    if _hier_multiproduct and _effective_skip_stage0:
        logger.info(
            "[GRID] Hierarchical multi-product: grid pruned to %d (lag × fourier_order) "
            "combinations — adstock/saturation overridden by channel_specs.",
            len(pruned_grid),
        )

    # ── Stage 1: MAP Pre-screen ───────────────────────────────
    if not skip_map and len(pruned_grid) > top_n_map:
        with timer("stage_1_map", n_models=len(pruned_grid)):
            top_configs = run_stage1_map_prescreen(prep, pruned_grid, top_n_map,
                                                   channel_prior_map=channel_prior_map)
    else:
        logger.info("[S1] MAP pre-screen skipped (pruned grid small enough or skip_map=True)")
        top_configs = pruned_grid[:top_n_map]

    # ── Stage 1b: Fast MCMC ───────────────────────────────────
    with timer("stage_1b_mcmc", n_models=len(top_configs)):
        survivors, df_stage1b = run_stage1b_fast_mcmc(
            prep=prep, top_configs=top_configs, top_k=top_k_refit,
            C=C, ranking_method=ranking_method, rank_weights=rank_weights,
            channel_specs=channel_specs,
            channel_prior_map=channel_prior_map,
            stage0_ran=not _effective_skip_stage0,
            use_lag=use_lag, global_max_lag=global_max_lag,
        )

    # ── Stage 2: Full Refit ───────────────────────────────────
    with timer("stage_2_full", n_models=len(survivors)):
        best, df_stage2 = run_stage2_full_refit(
            prep=prep, survivors=survivors, C=C,
            ranking_method=ranking_method, rank_weights=rank_weights,
            draws_full=draws_full, tune_full=tune_full, chains_full=chains_full,
            max_quality_flag=max_quality_flag,
            reject_if_all_bad=reject_if_all_bad,
            channel_specs=channel_specs,
            channel_prior_map=channel_prior_map,
            stage0_ran=not _effective_skip_stage0,
            use_lag=use_lag, global_max_lag=global_max_lag,
        )
    best["prep"]              = prep
    best["channel_specs"]     = channel_specs
    best["channel_prior_map"] = channel_prior_map

    # ── Analysis ──────────────────────────────────────────────
    with timer("analysis"):
        logger.info("\n[ANALYSIS] Computing channel contributions...")
        df_ch            = compute_channel_contributions(best, prep)
        df_components    = compute_component_contributions(best)
        df_ch_by_product = compute_per_product_contributions(best, prep)

        logger.info("\n  Channel Contribution Summary:")
        logger.info(df_ch[["channel","mean_share_pct","roi_proxy","hdi_90_low","hdi_90_high"]].to_string(index=False))
        if df_ch_by_product is not None:
            logger.info("\n  Per-Product Channel Contribution Summary:")
            logger.info(df_ch_by_product[["product","channel","mean_share_pct","roi_proxy"]].to_string(index=False))
        logger.info("\n  Component Contribution Summary:")
        logger.info(df_components[["component", "group", "share_abs_pct", "mean_effect"]].to_string(index=False))


    # ── Plots ─────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = best["metrics"]
    total_time = (time.perf_counter() - t_pipeline_start) / 60.0

    rhat_label = (
        f"{m['max_rhat']:.5f} [estimated — 1 chain]"
        if m.get('max_rhat_is_estimated')
        else f"{m['max_rhat']:.5f}"
    )

    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total wall time   : {total_time:.1f} minutes")
    logger.info(f"  Best model key    : {best['cfg'].key()}")
    logger.info(f"  Fourier order     : {best['cfg'].fourier_order}")
    logger.info(f"  Max adstock lag   : {best['cfg'].max_lag}")
    if use_lag:
        logger.info(f"  Lag transform     : enabled (max={global_max_lag} periods)")
    logger.info(f"  Quality flag      : {quality_flag(m)} (0=best, 2=poor)")
    logger.info(f"  MAPE              : {m['mape']:.4f}%")
    logger.info(f"  R2                : {m['r2']:.4f}")
    logger.info(f"  Pearson r         : {m['pearson_r']:.4f}")
    logger.info(f"  Max R-hat         : {rhat_label}")
    logger.info(f"  ESS bulk (min)    : {m['ess_bulk']:.0f}")
    logger.info(f"  ESS tail (min)    : {m['ess_tail']:.0f}")
    logger.info(f"  Divergences       : {m['divergences']}")
    logger.info(f"  LOO-IC            : {m['loo_ic']:.4f} +/- {m['loo_se']:.4f}")
    logger.info(f"  WAIC              : {m['waic']:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    with timer("plots"):
        logger.info("\n[PLOTS] Generating diagnostic plots...")
        generate_all_plots(best, prep, df_ch, df_components, out_dir)

    # ── Export ────────────────────────────────────────────────────────────────
    with timer("export"):
        logger.info("\n[EXPORT] Writing output files...")
        export_all_results(
            best               = best,
            prep               = prep,
            df_ch              = df_ch,
            df_components      = df_components,
            df_stage0          = df_stage0,
            df_stage1b         = df_stage1b,
            df_stage2          = df_stage2,
            data_cfg           = data_cfg,
            channel_prior_map  = channel_prior_map,
            export_excel       = True,
            df_ch_by_product   = df_ch_by_product,
            df_timings         = timer.to_dataframe(),
        )

    # ── Save pipeline state for standalone analysis scripts ────────────────
    _save_pipeline_state(best, prep, out_dir)
    _opt_hint = ""
    if opt_config:
        _opt_hint = (
            "\n  optimisation: section detected in YAML — run standalone scripts to apply it:"
        )
    logger.info(
        f"\n[STATE] Model state saved. Run post-pipeline analysis when ready:{_opt_hint}\n"
        "  python analyze.py curves    --config <yaml>   <- response curves\n"
        "  python analyze.py optimise  --config <yaml>   <- forward budget optimisation\n"
        "  python analyze.py reverse   --config <yaml>   <- reverse optimisation\n"
        "  python analyze.py scenarios --config <yaml>   <- scenario comparison\n"
        "  python analyze.py report    --config <yaml>   <- full HTML report\n"
        "  python analyze.py all       --config <yaml>   <- run everything"
    )

    return best
