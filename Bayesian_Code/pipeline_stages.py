# pipeline_stages.py
# ─────────────────────────────────────────────────────────────────────────────
# The four pipeline stages:
#   Stage 0   - Per-channel marginal contribution scan
#   Stage 1   - MAP pre-screen
#   Stage 1b  - Fast MCMC scan
#   Stage 2   - Full refit
#   Hybrid    - Optional Bayesian/ElasticNet fast pre-scan
# ─────────────────────────────────────────────────────────────────────────────

import logging
import time
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pymc as pm

# joblib is available with scikit-learn; falls back to sequential if absent
try:
    from joblib import Parallel, delayed as _jb_delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

import arviz as az

from config import (
    DataConfig, ModelConfig, ChannelTransformSpec,
    ADSTOCK_SAT_COMBOS, GLOBAL_SEED,
    build_full_grid, build_pruned_grid, build_channel_specs, _dedupe_combo_list,
)
from model_builder import build_mmm, build_marginal_model, sample_model, build_channel_transform_summary
from metrics import (
    evaluate_all_metrics, build_result_row, quality_flag, model_rank_key,
    _inv_transform,
)

logger = logging.getLogger("MMM")


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid fast scan (optional pre-step using fast_scan_bo_mmm.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_hybrid_fast_scan(
    prep        : Dict[str, Any],
    data_cfg    : DataConfig,
    base_cfg    : ModelConfig,
    n_trials    : int = 80,
    timeout_sec : int = 240,
) -> Dict[str, Any]:
    """
    Optional hybrid pre-scan using fast_scan_bo_mmm.py.
    Falls back to built-in ridge-regression fast scan if the external module
    is not available.
    """
    try:
        from fast_scan_bo_mmm import bayes_fast_scan_elasticnet, CVConfig, ENetCfg
    except Exception as e:
        logger.info(f"[HYBRID-FAST-SCAN] External module not available: {e}")
        logger.info("[HYBRID-FAST-SCAN] Using built-in ridge-regression fast scan instead.")
        try:
            return run_builtin_fast_scan(
                prep=prep,
                max_lag=max(1, int(base_cfg.max_lag)),
            )
        except Exception as e2:
            logger.warning(f"[HYBRID-FAST-SCAN] Built-in scan also failed: {e2}")
            return {"enabled": False, "reason": "builtin_failed"}

    df         = prep["df"].copy()
    spend_cols = prep["spend_cols"]

    try:
        best_params, best_rmse, _study = bayes_fast_scan_elasticnet(
            df=df,
            kpi_col=data_cfg.response_col,
            spend_cols=spend_cols,
            date_col=data_cfg.date_col,
            controls_cols=prep.get("active_controls", []),
            include_trend=base_cfg.use_trend,
            include_fourier=True,
            fourier_period=float(prep.get("fourier_period", 52.18)),  # FIX: match data_prep.py
            fourier_order=max(1, int(base_cfg.fourier_order)),
            cv=CVConfig(n_splits=3),
            enet=ENetCfg(alpha=0.03, l1_ratio=0.2),
            n_trials=int(n_trials),
            timeout_sec=int(timeout_sec),
            random_seed=GLOBAL_SEED,
        )
    except Exception as e:
        logger.warning(f"[HYBRID-FAST-SCAN] Execution failed, continuing without it: {e}")
        return {"enabled": False, "reason": "execution_failed"}

    combo_shortlist    : Dict[int, List[Tuple[str, str]]] = {}
    mapped_best_combo  : Dict[int, Tuple[str, str]] = {}

    # All 4 saturation types now supported - no silent truncation to hill/softplus
    _SAT_MAP = {
        "hill"        : "hill",
        "softplus"    : "softplus",
        "logistic"    : "logistic",
        "exponential" : "exponential",
    }
    _ALL_SATS = ["softplus", "hill", "logistic", "exponential"]

    for j, ch in enumerate(spend_cols):
        p   = best_params[ch]
        ads = "weibull" if float(p.half_life) >= 3.5 else "geometric"

        # FIX: map all 4 saturation types, not just hill/softplus
        sat = _SAT_MAP.get(getattr(p, "sat_family", "softplus"), "softplus")
        mapped_best_combo[j] = (ads, sat)

        # FIX: build shortlist from all 4 alternatives, not just one
        # Order: best guess first, then alternatives in priority order
        alt_sats  = [s for s in _ALL_SATS if s != sat]
        shortlist = _dedupe_combo_list([
            (ads, sat),                                      # hybrid best guess
            (ads, alt_sats[0]),                             # closest alternative
            (ads, alt_sats[1]),                             # second alternative
            (base_cfg.adstock_type, base_cfg.saturation),  # base fallback
        ])
        combo_shortlist[j] = shortlist

    logger.info(
        f"[HYBRID-FAST-SCAN] done | trials={n_trials} timeout={timeout_sec}s | best_cv_rmse={best_rmse:.4f}"
    )
    return {
        "enabled"                  : True,
        "best_cv_rmse"             : float(best_rmse),
        "mapped_best_combo"        : mapped_best_combo,
        "candidate_combo_per_channel": combo_shortlist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 - Marginal contribution scan
# ─────────────────────────────────────────────────────────────────────────────

def run_stage0_marginal_scan(
    prep                      : Dict[str, Any],
    base_cfg                  : ModelConfig,
    candidate_combo_per_channel: Optional[Dict[int, List[Tuple[str, str]]]] = None,
    ranking_method            : str = "lexicographic",
    rank_weights              : Optional[Dict[str, float]] = None,
    channel_prior_map         : Optional[Dict] = None,   # FIX-1: per-channel priors
) -> Tuple[Dict[int, Tuple[str, str]], List[Dict]]:
    """
    Stage 0 - Per-channel marginal contribution scan.

    For each channel j:
      - Fits 6 FULL C-channel models
      - Each model uses base_cfg adstock/saturation for channels i≠j
      - Only channel j varies across the 6 (adstock × saturation) combos
      - Selects the best combo for channel j based on rank_tuple

    Returns
    -------
    best_combo_per_channel : Dict[channel_idx -> (adstock, saturation)]
    stage0_rows            : list of result row dicts for export
    """
    C          = prep["C"]
    spend_cols = prep["spend_cols"]
    best_combo : Dict[int, Tuple[str, str]] = {}

    # Use the base_cfg consistently for ALL non-focal channels.
    # build_marginal_model already overrides only the focal channel (j)
    # with (focal_ads, focal_sat), so there is no need for progressive
    # updates. This eliminates scan-order dependence.
    scan_cfg = deepcopy(base_cfg)

    logger.info("=" * 60)
    logger.info("STAGE 0 - MARGINAL CONTRIBUTION SCAN")

    total_models = sum(
        max(1, len(candidate_combo_per_channel[j]))
        if candidate_combo_per_channel and j in candidate_combo_per_channel
        else len(ADSTOCK_SAT_COMBOS)
        for j in range(C)
    )
    logger.info(f"  Channels: {C} | Total models: {total_models}")
    logger.info("=" * 60)

    stage0_rows = []

    # ── Build the full list of (channel_idx, ads, sat) jobs ──────────────────
    all_jobs = []
    for j, ch_name in enumerate(spend_cols):
        combos = ADSTOCK_SAT_COMBOS
        if candidate_combo_per_channel and j in candidate_combo_per_channel:
            combos = candidate_combo_per_channel[j] or ADSTOCK_SAT_COMBOS
        for ads, sat in combos:
            all_jobs.append((j, ch_name, ads, sat))

    def _run_one_job(
        j                 : int,
        ch_name           : str,
        ads               : str,
        sat               : str,
        prep              : Dict,
        scan_cfg          : Any,
        base_cfg          : Any,
        C                 : int,
        channel_prior_map : Optional[Dict],
    ) -> Optional[Dict]:
        """
        Fit one (channel, adstock, saturation) combo.
        Returns a result dict or None on failure.
        Runs inside a joblib worker process — all imports must be local.
        """
        import warnings as _w
        # Suppress PyTensor loop-fusion warnings in subprocess (loky workers do not
        # inherit the main-process warning filters set in main.py).
        _w.filterwarnings("ignore", category=UserWarning)
        _w.filterwarnings("ignore", message="Loop fusion failed.*kernel argument limit",
                          category=UserWarning)

        from copy import deepcopy
        from model_builder import build_marginal_model, sample_model
        from metrics import evaluate_all_metrics, build_result_row
        import traceback as _tb

        combo_label = f"{ads}+{sat}"
        try:
            model       = build_marginal_model(prep, scan_cfg, j, ads, sat,
                                               channel_prior_map=channel_prior_map)
            trace, wall = sample_model(model, scan_cfg, fast_mode=True)
            # Stage 0: skip LOO/WAIC — too expensive for C×6 fast scans
            metrics     = evaluate_all_metrics(trace, prep, wall, compute_loo=False)
            del trace

            row_cfg              = deepcopy(base_cfg)
            row_cfg.adstock_type = ads
            row_cfg.saturation   = sat
            result_row = (
                build_result_row(row_cfg, metrics, f"stage0_ch{j}", 0, C)
                | {"focal_channel": j, "focal_ch_name": ch_name}
            )
            return {
                "channel"   : j,
                "ch_name"   : ch_name,
                "adstock"   : ads,
                "saturation": sat,
                "metrics"   : metrics,
                "result_row": result_row,
                "ok"        : True,
            }
        except Exception as e:
            return {
                "channel"   : j,
                "ch_name"   : ch_name,
                "adstock"   : ads,
                "saturation": sat,
                "ok"        : False,
                "error"     : str(e),
                "tb"        : _tb.format_exc(),
            }

    # ── Run jobs — parallel if joblib available, else sequential ─────────────
    logger.info(
        f"  Running {len(all_jobs)} Stage-0 fits "
        + ("in parallel (joblib)" if _JOBLIB_AVAILABLE else "sequentially")
    )

    if _JOBLIB_AVAILABLE:
        job_results = Parallel(n_jobs=-1, backend="loky", verbose=0)(
            _jb_delayed(_run_one_job)(
                j, ch_name, ads, sat,
                prep, scan_cfg, base_cfg, C, channel_prior_map,
            )
            for j, ch_name, ads, sat in all_jobs
        )
    else:
        job_results = [
            _run_one_job(j, ch_name, ads, sat, prep, scan_cfg, base_cfg, C, channel_prior_map)
            for j, ch_name, ads, sat in all_jobs
        ]

    # ── Collect results per channel ───────────────────────────────────────────
    ch_results_map: Dict[int, List] = {j: [] for j in range(C)}
    for res in job_results:
        j = res["channel"]
        if res["ok"]:
            ch_results_map[j].append(res)
            stage0_rows.append(res["result_row"])
            logger.info(
                f"    [OK] ch={res['ch_name']} {res['adstock']}+{res['saturation']:12s} | "
                f"MAPE={res['metrics']['mape']:6.2f}% | "
                f"R-hat={res['metrics']['max_rhat']:.4f}"
                + (" [est]" if res["metrics"].get("max_rhat_is_estimated") else "") +
                f" | divs={res['metrics']['divergences']}"
            )
        else:
            logger.warning(
                f"    [FAIL] ch={res['ch_name']} {res['adstock']}+{res['saturation']} — {res['error']}"
            )

    for j, ch_name in enumerate(spend_cols):
        ch_results = ch_results_map[j]
        if ch_results:
            best_res      = min(ch_results,
                                key=lambda r: model_rank_key(r["metrics"], ranking_method, rank_weights))
            winner        = (best_res["adstock"], best_res["saturation"])
            best_combo[j] = winner
            logger.info(
                f"  [BEST] {ch_name}: {winner[0]}+{winner[1]} "
                f"(MAPE={best_res['metrics']['mape']:.2f}%)"
            )
        else:
            best_combo[j] = (base_cfg.adstock_type, base_cfg.saturation)
            logger.warning(f"  [WARN] All combos failed for {ch_name} — using base defaults")

    logger.info("\n  Stage 0 Summary:")
    for j, (ads, sat) in best_combo.items():
        logger.info(f"    Channel {spend_cols[j]:30s} -> {ads:10s} + {sat}")

    return best_combo, stage0_rows


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 - MAP pre-screen
# ─────────────────────────────────────────────────────────────────────────────

def run_map_for_config(
    prep             : Dict[str, Any],
    cfg              : ModelConfig,
    channel_prior_map: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Runs MAP estimation for a single config. Returns map_mape and map_loss, or None.
    """
    try:
        model = build_mmm(prep, cfg, channel_prior_map=channel_prior_map)
        with model:
            map_est = pm.find_MAP(method="L-BFGS-B", progressbar=False,
                                  maxeval=5000, tol=1e-6)

        y_mu  = prep["y_mu"]
        y_std = prep["y_std"]
        y_raw = prep["y_raw"][prep["train_idx"]]

        mu_sc_map = map_est.get("mu", None)
        if mu_sc_map is None:
            return None

        # FIX: use the correct inverse transform (log1p, sqrt, identity, boxcox)
        # instead of always assuming log1p. Reads response_transform from cfg
        # so MAP-MAPE is on the same scale regardless of which transform was used.
        rt       = getattr(cfg, "response_transform", "log1p")
        bl       = prep.get("boxcox_lambda")
        mu_log   = mu_sc_map * y_std + y_mu
        y_hat    = _inv_transform(mu_log, transform=rt, boxcox_lam=bl)
        mask     = y_raw > 1.0
        map_mape = float(np.mean(np.abs(y_raw[mask] - y_hat[mask]) / y_raw[mask])) * 100.0

        return {
            "map_mape": map_mape,
            "map_loss": float(sum(v.sum() for v in map_est.values()
                                  if isinstance(v, np.ndarray))),
        }
    except Exception as e:
        logger.debug(f"MAP failed for {cfg.key()}: {e}")
        return None


def run_stage1_map_prescreen(
    prep             : Dict[str, Any],
    pruned_grid      : List[ModelConfig],
    top_n_map        : int = 20,
    channel_prior_map: Optional[Dict] = None,
) -> List[ModelConfig]:
    """
    Stage 1 - MAP pre-screen.
    Runs MAP estimation on all pruned_grid configs and keeps top_n_map by map_mape.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1 - MAP PRE-SCREEN")
    logger.info(f"  Configs to screen: {len(pruned_grid)} | Keeping top: {top_n_map}")
    logger.info("=" * 60)

    map_results = []
    for i, cfg in enumerate(pruned_grid):
        logger.info(f"  [{i+1:3d}/{len(pruned_grid)}] MAP: {cfg.key()} ...")
        t0  = time.perf_counter()
        res = run_map_for_config(prep, cfg, channel_prior_map=channel_prior_map)
        dt  = time.perf_counter() - t0
        if res is not None:
            map_results.append((cfg, res["map_mape"]))
            logger.info(f"    [OK] MAP MAPE={res['map_mape']:.2f}%  ({dt:.1f}s)")
        else:
            map_results.append((cfg, 999.0))
            logger.warning(f"    [FAIL] MAP failed ({dt:.1f}s)")

    map_results.sort(key=lambda x: x[1])
    top_configs = [cfg for cfg, _ in map_results[:top_n_map]]

    logger.info(f"\n  Top {top_n_map} configs after MAP pre-screen:")
    for i, (cfg, mape) in enumerate(map_results[:top_n_map], 1):
        logger.info(f"    {i:2d}. {cfg.key():55s} MAP MAPE={mape:.2f}%")

    return top_configs


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1b - Fast MCMC scan
# ─────────────────────────────────────────────────────────────────────────────

def run_stage1b_fast_mcmc(
    prep              : Dict[str, Any],
    top_configs       : List[ModelConfig],
    top_k             : int = 4,
    C                 : int = 1,
    ranking_method    : str = "lexicographic",
    rank_weights      : Optional[Dict[str, float]] = None,
    channel_specs     : Optional[Dict[int, "ChannelTransformSpec"]] = None,
    channel_prior_map : Optional[Dict] = None,
    stage0_ran        : bool = True,
    use_lag           : bool = False,
    global_max_lag    : int  = 0,
    max_quality_flag_stage1b : int = 1,
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Stage 1b - Fast MCMC on top-N MAP survivors.

    When channel_specs is provided (per-channel winning transforms from Stage 0),
    every build_mmm call uses per-channel transforms instead of the global cfg.
    Uses fast_mode=True. Returns top_k survivors for full refit.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1b - FAST MCMC SCAN")
    fast_draws  = top_configs[0].fast_draws  if top_configs else "n/a"
    fast_chains = top_configs[0].fast_chains if top_configs else "n/a"
    logger.info(f"  Configs: {len(top_configs)} | Fast mode: {fast_draws} draws, {fast_chains} chain(s)")
    logger.info(f"  Keeping top-{top_k} for full refit")
    if channel_specs:
        logger.info("  Per-channel transforms: ENABLED")
    logger.info("=" * 60)

    results  = []
    all_rows = []

    for i, cfg in enumerate(top_configs):
        logger.info(f"\n  [{i+1:2d}/{len(top_configs)}] Fast MCMC: {cfg.key()}")
        try:
            # When Stage 0 was skipped, build per-config channel_specs so that
            # the grid's adstock/saturation variations are actually applied.
            if not stage0_ran:
                _specs = build_channel_specs(
                    best_combo     = {j: (cfg.adstock_type, cfg.saturation) for j in range(C)},
                    spend_cols     = prep["spend_cols"],
                    use_lag        = use_lag,
                    global_max_lag = global_max_lag,
                )
            else:
                _specs = channel_specs
            model         = build_mmm(prep, cfg, channel_specs=_specs,
                                       channel_prior_map=channel_prior_map)
            trace, wall   = sample_model(model, cfg, fast_mode=True)
            # Stage 1b: skip LOO/WAIC — fast MCMC scan, ranking uses MAPE/R-hat
            metrics       = evaluate_all_metrics(trace, prep, wall, compute_loo=False)
            qf            = quality_flag(metrics)

            results.append({"cfg": cfg, "metrics": metrics, "trace": trace})
            all_rows.append(build_result_row(cfg, metrics, "stage1b", i + 1, C))

            logger.info(
                f"    [OK] qf={qf} | MAPE={metrics['mape']:.2f}% | "
                f"R2={metrics['r2']:.4f} | R-hat={metrics['max_rhat']:.4f}"
                + (" [estimated]" if metrics.get("max_rhat_is_estimated") else "") +
                f" | divs={metrics['divergences']} | wall={wall:.0f}s"
            )
        except Exception as e:
            logger.warning(f"    [FAIL] Fast MCMC failed: {e}")
            logger.debug(traceback.format_exc())

    if not results:
        raise RuntimeError("All Stage 1b models failed.")

    results.sort(key=lambda r: model_rank_key(r["metrics"], ranking_method, rank_weights))

    rank_map = {r["cfg"].key(): i + 1 for i, r in enumerate(results)}
    for row in all_rows:
        row["rank"] = rank_map.get(row["model_key"], row["rank"])

    df_1b = pd.DataFrame(all_rows)

    # FIX: free non-surviving traces immediately to avoid memory accumulation.
    # Stage-1b quality filtering: prefer "not poor" models when any exist.
    survivors_candidates = [
        r for r in results
        if int(quality_flag(r["metrics"])) <= int(max_quality_flag_stage1b)
    ]
    if survivors_candidates:
        survivors = survivors_candidates[:top_k]
    else:
        logger.warning(
            f"[S1b] No models passed quality_flag <= {max_quality_flag_stage1b}. "
            "Falling back to top-K by ranking."
        )
        survivors = results[:top_k]
    for r in results[top_k:]:
        if "trace" in r:
            del r["trace"]

    logger.info(f"\n  Top-{top_k} survivors for full refit:")
    for i, r in enumerate(survivors, 1):
        m = r["metrics"]
        loo_val = m.get("loo_ic", float("nan"))
        loo_str = f"{loo_val:.2f}" if not (isinstance(loo_val, float) and np.isnan(loo_val)) else "N/A"
        logger.info(
            f"    {i}. {r['cfg'].key():55s} | "
            f"MAPE={m['mape']:.2f}% | R-hat={m['max_rhat']:.4f} | LOO={loo_str}"
        )

    return survivors, df_1b


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 - Full refit
# ─────────────────────────────────────────────────────────────────────────────

def run_stage2_full_refit(
    prep              : Dict[str, Any],
    survivors         : List[Dict],
    C                 : int,
    ranking_method    : str = "lexicographic",
    rank_weights      : Optional[Dict[str, float]] = None,
    draws_full        : int = 1800,
    tune_full         : int = 800,
    chains_full       : int = 2,
    max_quality_flag  : int = 1,
    reject_if_all_bad : bool = True,
    channel_specs     : Optional[Dict[int, "ChannelTransformSpec"]] = None,
    channel_prior_map : Optional[Dict] = None,
    stage0_ran        : bool = True,
    use_lag           : bool = False,
    global_max_lag    : int  = 0,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Stage 2 - Full MCMC refit of top-K survivors.

    When channel_specs is provided every channel uses its own winning
    adstock + saturation + lag spec from Stage 0 in the final model.
    Returns the best model dict (with channel_transform_summary attached)
    and the Stage 2 results DataFrame.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2 - FULL REFIT")
    logger.info(f"  Models to refit: {len(survivors)} | Full settings: {draws_full} draws, {chains_full} chain(s)")
    if channel_specs:
        logger.info("  Per-channel transforms: ENABLED")
        for j, spec in sorted(channel_specs.items()):
            logger.info(f"    [{j}] {spec.channel_name:<30s} -> {spec.label()}")
    logger.info("=" * 60)

    results  = []
    all_rows = []

    for i, survivor in enumerate(survivors):
        cfg             = deepcopy(survivor["cfg"])
        cfg.draws       = int(draws_full)
        cfg.tune        = int(tune_full)
        cfg.chains      = int(chains_full)
        cfg.progressbar = True

        if int(cfg.chains) < 2:
            raise ValueError(
                "Stage 2 (full refit) requires chains >= 2 so R-hat is real. "
                f"Got chains={cfg.chains}."
            )

        logger.info(f"\n  [{i+1}/{len(survivors)}] Full refit: {cfg.key()}")

        try:
            if not stage0_ran:
                _specs = build_channel_specs(
                    best_combo     = {j: (cfg.adstock_type, cfg.saturation) for j in range(C)},
                    spend_cols     = prep["spend_cols"],
                    use_lag        = use_lag,
                    global_max_lag = global_max_lag,
                )
            else:
                _specs = channel_specs
            model         = build_mmm(prep, cfg, channel_specs=_specs,
                                       channel_prior_map=channel_prior_map)
            trace, wall   = sample_model(model, cfg, fast_mode=False)
            metrics       = evaluate_all_metrics(trace, prep, wall)
            qf            = quality_flag(metrics)

            results.append({"cfg": cfg, "metrics": metrics, "trace": trace, "prep": prep})
            all_rows.append(build_result_row(cfg, metrics, "stage2", i + 1, C))

            logger.info(
                f"    [OK] qf={qf} | MAPE={metrics['mape']:.2f}% | "
                f"R2={metrics['r2']:.4f} | R-hat={metrics['max_rhat']:.4f} | "
                f"divs={metrics['divergences']} | LOO={metrics['loo_ic']:.2f} | "
                f"ESS={metrics['ess_bulk']:.0f} | wall={wall:.0f}s"
            )
        except Exception as e:
            logger.warning(f"    [FAIL] Full refit failed: {e}")
            logger.debug(traceback.format_exc())

    if not results:
        raise RuntimeError("All Stage 2 full refits failed.")

    # ── Hard quality gate (optional but recommended) ───────────────────────
    # Filter out poor-quality models (quality_flag=2) before selecting a "best".
    # If everything is poor, either hard-fail (default) or fall back to best-of-bad.
    valid = [
        r for r in results
        if int(quality_flag(r["metrics"])) <= int(max_quality_flag)
    ]
    if not valid:
        results.sort(key=lambda r: model_rank_key(r["metrics"], ranking_method, rank_weights))
        best_of_bad = results[0]
        m = best_of_bad["metrics"]
        msg = (
            "All Stage 2 models failed quality checks "
            f"(min observed quality_flag={min(int(quality_flag(r['metrics'])) for r in results)}; "
            f"required <= {int(max_quality_flag)}). "
            "This usually means you need more chains/draws/tune, stronger priors, "
            "or the data is too short/collinear for the chosen model."
        )
        logger.error(msg)
        if reject_if_all_bad:
            raise RuntimeError(msg)
        # Soft-fail: keep best-of-bad but mark it so callers can treat it as rejected.
        best_of_bad["pipeline_rejected"] = True
        best_of_bad["rejection_reason"] = msg
        # Still produce a ranked df for inspection.
        results = [best_of_bad] + [r for r in results if r is not best_of_bad]
        valid = results

    valid.sort(key=lambda r: model_rank_key(r["metrics"], ranking_method, rank_weights))
    results = valid

    rank_map = {r["cfg"].key(): i + 1 for i, r in enumerate(results)}
    for row in all_rows:
        row["rank"] = rank_map.get(row["model_key"], row["rank"])

    df_s2 = pd.DataFrame(all_rows)
    best  = results[0]
    m     = best["metrics"]

    # ── Build and attach per-channel transform summary ─────
    if channel_specs:
        df_transform_summary = build_channel_transform_summary(best, channel_specs)
        best["channel_transform_summary"] = df_transform_summary

        logger.info("\n  == Per-Channel Transform Summary ==")
        summary_cols = ["channel", "adstock", "saturation", "use_lag", "lag_mode",
                        "lam_mean", "alpha_sat_mean", "beta_mean"]
        logger.info(
            df_transform_summary[
                [c for c in summary_cols if c in df_transform_summary.columns]
            ].to_string(index=False)
        )

    logger.info(f"\n  [BEST] BEST MODEL: {best['cfg'].key()}")
    logger.info(f"    MAPE       : {m['mape']:.4f}%")
    logger.info(f"    R2         : {m['r2']:.4f}")
    logger.info(f"    Pearson r  : {m['pearson_r']:.4f}")
    logger.info(f"    Max R-hat  : {m['max_rhat']:.5f}")
    logger.info(f"    ESS bulk   : {m['ess_bulk']:.0f}")
    logger.info(f"    ESS tail   : {m['ess_tail']:.0f}")
    logger.info(f"    Divergences: {m['divergences']}")
    logger.info(f"    LOO-IC     : {m['loo_ic']:.4f}")
    logger.info(f"    WAIC       : {m['waic']:.4f}")

    return best, df_s2


# ─── Built-in fast scan (fast_scan_builtin.py) ────────────────────────────────

import time as _time
from scipy.optimize import minimize_scalar as _minimize_scalar


def _np_geometric_adstock(x: np.ndarray, lam: float, max_lag: int) -> np.ndarray:
    """Geometric adstock in pure numpy."""
    w = lam ** np.arange(max_lag + 1)
    w = w / (w.sum() + 1e-12)
    out = np.zeros_like(x, dtype=float)
    for k in range(max_lag + 1):
        if k == 0:
            out += w[k] * x
        else:
            padded = np.concatenate([np.zeros(k), x[:-k]])
            out += w[k] * padded
    return out


def _np_weibull_adstock(x: np.ndarray, lam: float, k_shape: float, max_lag: int) -> np.ndarray:
    """Weibull adstock in pure numpy."""
    lags = np.arange(1, max_lag + 1, dtype=float)
    w = (k_shape / lam) * ((lags / lam) ** (k_shape - 1.0)) * np.exp(-((lags / lam) ** k_shape))
    w = np.concatenate([[1.0], w])
    w = w / (w.sum() + 1e-12)
    out = np.zeros_like(x, dtype=float)
    for k in range(max_lag + 1):
        if k == 0:
            out += w[k] * x
        else:
            padded = np.concatenate([np.zeros(k), x[:-k]])
            out += w[k] * padded
    return out


def _np_saturation(x: np.ndarray, sat_type: str, alpha: float = 1.0, kappa: float = 0.5) -> np.ndarray:
    """Apply saturation transform in pure numpy."""
    x = np.maximum(x, 0.0)
    if sat_type == "softplus":
        return np.log1p(np.exp(alpha * x)) / np.log(2.0)
    elif sat_type == "hill":
        x_pos = np.maximum(x, 1e-12)
        return x_pos ** alpha / (x_pos ** alpha + kappa ** alpha)
    elif sat_type == "logistic":
        return 1.0 / (1.0 + np.exp(-alpha * (x - kappa)))
    elif sat_type == "exponential":
        return 1.0 - np.exp(-alpha * x)
    else:
        return x


def _ridge_mape(
    X: np.ndarray,
    y: np.ndarray,
    alpha_ridge: float = 1.0,
) -> float:
    """Fits ridge regression and returns MAPE."""
    T, K = X.shape
    X_aug = np.column_stack([np.ones(T), X])
    I = np.eye(K + 1)
    I[0, 0] = 0
    try:
        beta = np.linalg.solve(X_aug.T @ X_aug + alpha_ridge * I, X_aug.T @ y)
    except np.linalg.LinAlgError:
        return 999.0
    y_hat = X_aug @ beta
    mask = np.abs(y) > 1.0
    if mask.sum() == 0:
        return 999.0
    mape = float(np.mean(np.abs(y[mask] - y_hat[mask]) / (np.abs(y[mask]) + 1e-12))) * 100.0
    return mape


_BUILTIN_ADSTOCK_SAT_COMBOS = [
    ("geometric", "softplus"),
    ("geometric", "hill"),
    ("geometric", "logistic"),
    ("geometric", "exponential"),
    ("weibull", "softplus"),
    ("weibull", "hill"),
    ("weibull", "logistic"),
    ("weibull", "exponential"),
]

_LAM_GRID      = [0.2, 0.4, 0.6, 0.8]
_K_SHAPE_GRID  = [0.8, 1.5, 3.0]
_ALPHA_GRID    = [0.5, 1.0, 2.0, 4.0]
_KAPPA_GRID    = [0.3, 0.5, 1.0]


def run_builtin_fast_scan(
    prep          : Dict[str, Any],
    max_lag       : int  = 8,
    alpha_ridge   : float = 1.0,
    top_k_combos  : int   = 3,
) -> Dict[str, Any]:
    """
    Built-in fast scan: evaluates all (adstock, saturation) combos per channel
    using ridge regression (no MCMC), and returns the top K combos per channel.
    """
    t0 = _time.perf_counter()

    C          = prep["C"]
    T          = prep["T"]
    spend_cols = prep["spend_cols"]
    train_idx  = prep["train_idx"]

    _X_raw = prep["X_media_scaled"]
    if _X_raw.ndim == 3:
        X_media_sc = _X_raw.mean(axis=1)
    else:
        X_media_sc = _X_raw

    _y_raw = prep["y_scaled"]
    if _y_raw.ndim == 2:
        y_scaled = _y_raw[:, 0]
    else:
        y_scaled = _y_raw

    X_media_sc = X_media_sc[train_idx]
    y_scaled   = y_scaled[train_idx]

    X_fourier  = prep["X_fourier"][train_idx]

    logger.info("=" * 60)
    logger.info("BUILT-IN FAST SCAN (Ridge Regression)")
    logger.info(f"  Channels: {C} | Combos: {len(_BUILTIN_ADSTOCK_SAT_COMBOS)} | Max lag: {max_lag}")
    logger.info("=" * 60)

    all_results = []
    best_combo_per_channel: Dict[int, Tuple[str, str]] = {}
    candidate_combos: Dict[int, List[Tuple[str, str]]] = {}

    for j in range(C):
        ch_name = spend_cols[j]
        x_raw = X_media_sc[:, j]
        ch_results = []

        for ads_type, sat_type in _BUILTIN_ADSTOCK_SAT_COMBOS:
            best_mape = 999.0
            best_params = {}

            lam_grid = _LAM_GRID
            for lam_val in lam_grid:
                if ads_type == "geometric":
                    x_ads = _np_geometric_adstock(x_raw, lam_val, max_lag)
                else:  # weibull
                    for k_shape in _K_SHAPE_GRID:
                        x_ads_w = _np_weibull_adstock(x_raw, max(lam_val, 0.1), k_shape, max_lag)
                        for alpha_s in _ALPHA_GRID:
                            kappa_s = 0.5
                            x_sat = _np_saturation(x_ads_w, sat_type, alpha_s, kappa_s)
                            features = [x_sat.reshape(-1, 1)]
                            for k in range(C):
                                if k != j:
                                    features.append(X_media_sc[:, k].reshape(-1, 1))
                            features.append(X_fourier)
                            X_full = np.column_stack(features)
                            mape = _ridge_mape(X_full, y_scaled, alpha_ridge)
                            if mape < best_mape:
                                best_mape = mape
                                best_params = {"lam": lam_val, "k_shape": k_shape, "alpha_sat": alpha_s}
                    continue  # already handled weibull inside the k_shape loop

                # Geometric path
                for alpha_s in _ALPHA_GRID:
                    kappa_s = 0.5
                    x_sat = _np_saturation(x_ads, sat_type, alpha_s, kappa_s)
                    features = [x_sat.reshape(-1, 1)]
                    for k in range(C):
                        if k != j:
                            features.append(X_media_sc[:, k].reshape(-1, 1))
                    features.append(X_fourier)
                    X_full = np.column_stack(features)
                    mape = _ridge_mape(X_full, y_scaled, alpha_ridge)
                    if mape < best_mape:
                        best_mape = mape
                        best_params = {"lam": lam_val, "alpha_sat": alpha_s}

            ch_results.append({
                "channel_idx": j,
                "channel"    : ch_name,
                "adstock"    : ads_type,
                "saturation" : sat_type,
                "mape"       : round(best_mape, 4),
                "params"     : best_params,
            })
            all_results.append(ch_results[-1])

        ch_results.sort(key=lambda r: r["mape"])
        candidate_combos[j] = [(r["adstock"], r["saturation"]) for r in ch_results[:top_k_combos]]
        best_combo_per_channel[j] = candidate_combos[j][0]

        logger.info(f"  [{j+1}/{C}] {ch_name}:")
        for i, r in enumerate(ch_results[:top_k_combos]):
            marker = "★" if i == 0 else " "
            logger.info(f"    {marker} {r['adstock']+'+'+r['saturation']:<25s} MAPE={r['mape']:.2f}%")

    wall = _time.perf_counter() - t0

    logger.info(f"\n  Fast scan complete in {wall:.1f}s")
    logger.info("  Best combo per channel:")
    for j, (ads, sat) in best_combo_per_channel.items():
        logger.info(f"    [{j}] {spend_cols[j]:<30s} -> {ads}+{sat}")

    return {
        "enabled"                     : True,
        "candidate_combo_per_channel" : candidate_combos,
        "mapped_best_combo"           : best_combo_per_channel,
        "scan_results"                : all_results,
        "wall_time_s"                 : round(wall, 2),
    }
