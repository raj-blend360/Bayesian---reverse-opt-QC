#!/usr/bin/env python
# analyze.py
# ─────────────────────────────────────────────────────────────────────────────
# Single entry point for all post-pipeline analysis.
#
# Run AFTER main.py has fitted the model (it saves trace.nc + prep.pkl).
#
# Commands:
#   curves    — Response curves + marginal ROI plots/CSVs
#   optimise  — Forward budget optimisation (maximise response for a budget)
#   reverse   — Reverse optimisation (minimum spend to hit a response target)
#   scenarios — Side-by-side budget scenario comparison
#   report    — Full HTML report (curves + optimisation + scenarios)
#   all       — Run all of the above in sequence
#
# Usage examples:
#   python analyze.py curves    --config config_hier.yaml
#   python analyze.py optimise  --config config_hier.yaml
#   python analyze.py all       --output-dir Bayesian_Output
#   python analyze.py report    --config config_hier.yaml --budget-period monthly
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import base64
import io
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

def setup_logger(log_file: str = "mmm_run.log", level: int = logging.INFO) -> logging.Logger:
    fmt_console = logging.Formatter("%(levelname)-8s | %(message)s")
    fmt_file    = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("MMM")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt_console)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    logger.addHandler(fh)
    return logger

logger = logging.getLogger("MMM")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_dir(args) -> Path:
    """Return the output directory from --output-dir or the YAML output.dir."""
    if getattr(args, "output_dir", None):
        return Path(args.output_dir)
    if getattr(args, "config", None):
        try:
            import yaml
            with open(args.config, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            return Path(cfg.get("output", {}).get("dir", "mmm_outputs"))
        except Exception:
            pass
    return Path("mmm_outputs")


def _load_yaml(config_path: Optional[str]) -> dict:
    if not config_path:
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as e:
        logger.warning(f"Could not read YAML: {e}")
        return {}


def _setup(args):
    """Load model state + build CPP map. Returns (best, prep, cpp_map, cpp_w)."""
    from pipeline import load_pipeline_state
    from optimisation import build_cpp_map, cpp_weights_array, format_cpp_summary

    out_dir = _resolve_dir(args)
    best, prep = load_pipeline_state(out_dir)

    # Store output dir in best/prep so _compute_channel_rscales can auto-discover
    # channel_contributions.csv for its fallback path when the trace is unavailable.
    best["_out_dir"] = str(out_dir)
    prep["_out_dir"] = str(out_dir)

    cfg = _load_yaml(getattr(args, "config", None))
    cpp_rates = (cfg.get("optimisation") or {}).get("cpp_rates") or {}
    cpp_map   = build_cpp_map(prep, cpp_overrides=cpp_rates)
    cpp_w     = cpp_weights_array(prep["spend_cols"], cpp_map)
    logger.info("\n" + format_cpp_summary(cpp_map))

    return out_dir, best, prep, cfg, cpp_map, cpp_w


def _build_bounds(opt_cfg: dict, prep: dict, period_key: str, default_period: str):
    """Per-channel ±flex spend bounds for the given period."""
    from optimisation import _periods_in_window

    flex           = float(opt_cfg.get("default_spend_flex", 0.30))
    ch_constraints = opt_cfg.get("channel_constraints", {}) or {}
    frequency      = prep.get("frequency", "weekly")
    n_train        = len(prep["train_idx"])
    n_periods      = _periods_in_window(opt_cfg.get(period_key, default_period), frequency, n_train)

    X_rw = prep["X_media_raw"][prep["train_idx"]]
    cur_pp = {
        ch: float(X_rw[:, 0, j].mean() if X_rw.ndim == 3 else X_rw[:, j].mean())
        for j, ch in enumerate(prep["spend_cols"])
    }

    ch_min, ch_max = {}, {}
    for ch, sp in cur_pp.items():
        sp_period = sp * n_periods
        cst = ch_constraints.get(ch, {}) or {}
        if cst.get("fixed", False):
            ch_min[ch] = ch_max[ch] = sp_period
        else:
            lo = float(cst.get("min_pct", flex))
            hi = float(cst.get("max_pct", flex))
            ch_min[ch] = max(0.0, sp_period * (1.0 - lo))
            ch_max[ch] = sp_period * (1.0 + hi)
    return ch_min, ch_max, n_periods


# ─────────────────────────────────────────────────────────────────────────────
# Command: curves
# ─────────────────────────────────────────────────────────────────────────────

def cmd_curves(args):
    """Generate response curves and marginal ROI plots/CSVs."""
    from analysis import (
        compute_response_curves, compute_marginal_roi,
        compute_saturation_analysis, compute_response_curves_multiperiod,
        plot_response_curves,
    )

    out_dir, best, prep, cfg, cpp_map, _ = _setup(args)
    rc_dir = out_dir / "response_curves"
    rc_dir.mkdir(parents=True, exist_ok=True)

    from analysis import _cpp_per_channel
    X_rw     = prep["X_media_raw"][prep["train_idx"]]
    cpp_list = _cpp_per_channel(prep)
    obs_spend_gbp = {
        ch: float((X_rw[:, 0, j].mean() if X_rw.ndim == 3 else X_rw[:, j].mean()) * cpp_list[j])
        for j, ch in enumerate(prep["spend_cols"])
    }

    # ── 1. Instantaneous curve ─────────────────────────────────
    logger.info("\n[CURVES] Computing instantaneous response curves...")
    df_instant = compute_response_curves(best, prep, curve_type="instantaneous")
    p = rc_dir / "response_curves_instantaneous.csv"
    df_instant.to_csv(p, index=False)
    logger.info(f"  Saved: {p}")
    plot_response_curves(df_instant, rc_dir / "instantaneous",
                         obs_spend=obs_spend_gbp,
                         title_suffix=" — Instantaneous (single period, no carry-over)")

    # ── 2. Steady-state curve ─────────────────────────────────
    logger.info("\n[CURVES] Computing steady-state response curves...")
    df_ss   = compute_response_curves(best, prep, curve_type="steady_state")
    df_mroi = compute_marginal_roi(best, prep)
    df_sat  = compute_saturation_analysis(best, prep, cpp_map=cpp_map)
    p = rc_dir / "response_curves_steady_state.csv"
    df_ss.to_csv(p, index=False)
    logger.info(f"  Saved: {p}")
    # Keep legacy filename for backwards compat
    df_ss.to_csv(rc_dir / "response_curves.csv", index=False)
    plot_response_curves(df_ss, rc_dir,
                         obs_spend=obs_spend_gbp, cpp_map=cpp_map,
                         title_suffix=" — Steady-State (adstock at equilibrium)")
    for fname, df in [("marginal_roi.csv", df_mroi), ("saturation_analysis.csv", df_sat)]:
        if df is not None and not df.empty:
            df.to_csv(rc_dir / fname, index=False)
            logger.info(f"  Saved: {rc_dir / fname}")

    # ── 3. Multi-period cumulative ────────────────────────────
    logger.info("\n[CURVES] Computing multi-period cumulative curves (1M/3M/6M/1Y)...")
    try:
        from analysis import _plot_multiperiod_curves
        df_mp = compute_response_curves_multiperiod(best, prep)
        if df_mp is not None and not df_mp.empty:
            p = rc_dir / "response_curves_multiperiod.csv"
            df_mp.to_csv(p, index=False)
            logger.info(f"  Saved: {p}")
            try:
                _plot_multiperiod_curves(df_mp, rc_dir / "multiperiod", obs_spend_gbp)
            except Exception as _pe:
                logger.warning(f"Multi-period plot failed (non-fatal): {_pe}")
            # Append RC tabs to model_results.xlsx
            try:
                from export import _append_rc_multiperiod_to_model_excel
                model_xl = out_dir / "model_results" / "model_results.xlsx"
                _append_rc_multiperiod_to_model_excel(model_xl, rc_dir)
            except Exception as _xe:
                logger.warning(f"RC Excel append failed (non-fatal): {_xe}")
    except Exception as e:
        logger.warning(f"Multi-period curves failed (non-fatal): {e}")
        import traceback; logger.debug(traceback.format_exc())

    logger.info("[CURVES] Done.")
    return df_ss, df_mroi, df_sat


# ─────────────────────────────────────────────────────────────────────────────
# Command: optimise  [DISABLED — superseded by cmd_sequential]
# ─────────────────────────────────────────────────────────────────────────────
# All optimisation now runs through cmd_sequential, which supports forward
# AND reverse in a single multi-period framework.  The single-period SLSQP
# optimiser below is kept for reference but is no longer called.
#
# def cmd_optimise(args): ...   # <-- commented out, use 'sequential' instead
# def cmd_reverse(args):  ...   # <-- commented out, use 'sequential' instead


# ─────────────────────────────────────────────────────────────────────────────
# Command: scenarios
# ─────────────────────────────────────────────────────────────────────────────

def cmd_scenarios(args):
    """Side-by-side comparison of multiple budget scenarios."""
    from analysis import (
        ScenarioConfig, build_preset_scenarios, run_scenarios, plot_scenario_comparison,
    )

    out_dir, best, prep, cfg, cpp_map, cpp_w = _setup(args)
    sc_cfg  = cfg.get("scenarios") or {}
    opt_cfg = cfg.get("optimisation") or {}
    period  = sc_cfg.get("budget_period", opt_cfg.get("budget_period", "monthly"))
    n_samp  = int(sc_cfg.get("n_samples", 200))

    from optimisation import _periods_in_window
    from optimisation import cpp_weights_array
    n_periods = _periods_in_window(period, prep.get("frequency", "weekly"), len(prep["train_idx"]))
    X_rw = prep["X_media_raw"][prep["train_idx"]]
    curr_media_pp = X_rw.mean(axis=0) if X_rw.ndim == 2 else X_rw[:, 0, :].mean(axis=0)
    curr_spend_total = float((curr_media_pp * cpp_w).sum() * n_periods)

    scenarios = build_preset_scenarios(sc_cfg, opt_cfg, curr_spend_total)

    logger.info(f"\n[SCENARIOS] Running {len(scenarios)} scenario(s) | period={period}")
    df_cmp, df_detail = run_scenarios(
        best, prep, scenarios,
        budget_period = period,
        n_samples     = n_samp,
        cpp_map       = cpp_map,
    )

    sc_dir = out_dir / "scenarios"
    sc_dir.mkdir(parents=True, exist_ok=True)
    df_cmp.to_csv(sc_dir / "scenario_comparison.csv",    index=False)
    df_detail.to_csv(sc_dir / "scenario_channel_detail.csv", index=False)
    logger.info(f"  Saved: scenarios/scenario_comparison.csv + scenario_channel_detail.csv")

    try:
        plot_scenario_comparison(df_cmp, df_detail, sc_dir, budget_period=period)
    except Exception as e:
        logger.warning(f"Scenario plot failed (non-fatal): {e}")

    logger.info("[SCENARIOS] Done.")
    return df_cmp, df_detail


# ─────────────────────────────────────────────────────────────────────────────
# Command: report  (self-contained HTML — no external files needed to open it)
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
body{font-family:Arial,sans-serif;margin:0;background:#f4f6f9;color:#2c3e50}
.nav{background:#1a252f;padding:14px 28px;position:sticky;top:0;z-index:100}
.nav a{color:#ecf0f1;text-decoration:none;margin-right:22px;font-size:0.9em;font-weight:600}
.nav a:hover{color:#3498db}
.wrap{max-width:1200px;margin:0 auto;padding:24px}
h2{color:#1a252f;border-bottom:3px solid #3498db;padding-bottom:6px;margin-top:40px}
h3{color:#2c3e50;margin-top:24px}
table{border-collapse:collapse;width:100%;font-size:0.88em}
th{background:#34495e;color:#fff;padding:8px 12px;text-align:left}
td{padding:7px 12px;border-bottom:1px solid #ecf0f1;white-space:nowrap}
tr:hover td{background:#f0f4f8}
.pos{color:#27ae60;font-weight:600}
.neg{color:#e74c3c;font-weight:600}
.card{background:#fff;border-radius:8px;padding:20px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.1)}
"""

def _df_to_html(df: pd.DataFrame) -> str:
    """Render a DataFrame as a styled HTML table."""
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            style = "padding:7px 12px;border-bottom:1px solid #ecf0f1;white-space:nowrap;"
            if isinstance(val, float) and not pd.isna(val):
                if "pct" in col.lower() or "change" in col.lower():
                    cls = "pos" if val > 0 else ("neg" if val < 0 else "")
                    txt = f"{val:+.1f}%"
                    style += f"color:{'#27ae60' if cls=='pos' else '#e74c3c' if cls=='neg' else 'inherit'};font-weight:600;"
                elif "£" in col or "spend" in col.lower():
                    txt = f"£{val:,.0f}"
                else:
                    txt = f"{val:,.4f}" if abs(val) < 1000 else f"{val:,.1f}"
            elif isinstance(val, (int,)):
                txt = f"{val:,}"
            else:
                txt = str(val) if not (isinstance(val, float) and pd.isna(val)) else "—"
            cells.append(f"<td style='{style}'>{txt}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(
        f"<th style='padding:8px 12px;background:#34495e;color:white;"
        f"font-weight:600;font-size:0.85em;text-align:left;white-space:nowrap;'>{c}</th>"
        for c in df.columns
    )
    return (
        "<div style='overflow-x:auto;'>"
        "<table style='border-collapse:collapse;width:100%;font-size:0.88em;'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def _fig_to_b64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:100%;border-radius:6px;">'


def cmd_report(args):
    """Generate a self-contained HTML report (curves + optimisation + scenarios)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import datetime
    from analysis import (
        compute_response_curves, compute_marginal_roi,
        compute_saturation_analysis, plot_response_curves,
    )
    from optimisation import optimise_budget
    from metrics import quality_flag

    out_dir, best, prep, cfg, cpp_map, cpp_w = _setup(args)
    opt_cfg    = cfg.get("optimisation") or {}
    sc_cfg     = cfg.get("scenarios")    or {}
    period     = getattr(args, "budget_period", None) or opt_cfg.get("budget_period", "monthly")
    flex       = float(getattr(args, "spend_flex", None) or opt_cfg.get("default_spend_flex", 0.40))
    pcts       = getattr(args, "pcts", None) or sc_cfg.get("preset_pcts", [-20, -10, 10, 20])
    target_val = getattr(args, "target", None)

    m       = best["metrics"]
    freq    = prep.get("frequency", "weekly")
    quality = f"Quality flag: {quality_flag(m)} (0=best)"

    # ── Section 1: Response curves ────────────────────────────────────────────
    logger.info("[REPORT 1/3] Response curves...")
    rc_dir = out_dir / "response_curves"
    rc_dir.mkdir(parents=True, exist_ok=True)
    df_rc   = compute_response_curves(best, prep)
    df_mroi = compute_marginal_roi(best, prep)
    df_sat  = compute_saturation_analysis(best, prep, cpp_map=cpp_map)

    from analysis import _cpp_per_channel
    X_rw     = prep["X_media_raw"][prep["train_idx"]]
    cpp_list = _cpp_per_channel(prep)
    obs_spend = {
        ch: float((X_rw[:, 0, j].mean() if X_rw.ndim == 3 else X_rw[:, j].mean()) * cpp_list[j])
        for j, ch in enumerate(prep["spend_cols"])
    }

    # Save response curve plot; then load it as base64 for HTML embedding
    rc_b64 = None
    try:
        plot_response_curves(df_rc, rc_dir, df_mroi=df_mroi, obs_spend=obs_spend, cpp_map=cpp_map)
        rc_png = rc_dir / "response_curves.png"
        if rc_png.exists():
            rc_b64 = base64.b64encode(rc_png.read_bytes()).decode()
    except Exception as e:
        logger.warning(f"Response curve plot failed (non-fatal): {e}")

    # ── Section 2: Budget optimisation ────────────────────────────────────────
    logger.info("[REPORT 2/3] Budget optimisation scenarios...")
    from optimisation import _periods_in_window
    from optimisation import cpp_weights_array
    n_periods = _periods_in_window(period, freq, len(prep["train_idx"]))
    curr_media_pp = X_rw.mean(axis=0) if X_rw.ndim == 2 else X_rw[:, 0, :].mean(axis=0)
    curr_total_spend = float((curr_media_pp * cpp_w).sum() * n_periods)

    ch_min = {ch: sp * (1.0 - flex) for ch, sp in
              {ch: float(curr_media_pp[j] * cpp_w[j] * n_periods)
               for j, ch in enumerate(prep["spend_cols"])}.items()}
    ch_max = {ch: sp * (1.0 + flex) for ch, sp in
              {ch: float(curr_media_pp[j] * cpp_w[j] * n_periods)
               for j, ch in enumerate(prep["spend_cols"])}.items()}

    opt_tables = {}
    budgets = [curr_total_spend * (1 + p / 100) for p in pcts]
    budgets = [curr_total_spend] + budgets   # include current
    labels  = ["Current"] + [f"{'+' if p>0 else ''}{p}%" for p in pcts]

    for label, budget in zip(labels, budgets):
        try:
            df = optimise_budget(best, prep, total_budget=budget, budget_period=period,
                                 channel_min=ch_min, channel_max=ch_max, cpp_weights=cpp_w)
            opt_tables[label] = df
        except Exception as e:
            logger.warning(f"  Optimisation for '{label}' failed: {e}")

    if target_val:
        try:
            from optimisation import minimise_spend_for_target
            df_rev = minimise_spend_for_target(best, prep, target_response=target_val,
                                               target_period=period, cpp_weights=cpp_w)
            opt_tables[f"Reverse (target={target_val})"] = df_rev
        except Exception as e:
            logger.warning(f"Reverse scenario failed: {e}")

    # ── Build HTML ─────────────────────────────────────────────────────────────
    logger.info("[REPORT 3/3] Building HTML...")
    now     = datetime.now().strftime("%Y-%m-%d %H:%M")
    nav     = (f'<div class="nav"><a href="#curves">Response Curves</a>'
               f'<a href="#optimisation">Optimisation</a>'
               f'<span style="float:right;color:#7f8c8d;font-size:0.82em;">{now}</span></div>')

    sections = []

    # Response curves section
    rc_html = "<h2 id='curves'>Response Curves</h2>"
    if rc_b64 is not None:
        rc_html += f'<div class="card">{_img(rc_b64)}</div>'
    if df_sat is not None and not df_sat.empty:
        cols = [c for c in ["channel", "saturation_at_current_pct", "current_spend",
                             "sat_type", "ads_type"] if c in df_sat.columns]
        rc_html += f'<h3>Saturation Analysis</h3><div class="card">{_df_to_html(df_sat[cols])}</div>'
    sections.append(rc_html)

    # Optimisation section
    opt_html = "<h2 id='optimisation'>Budget Optimisation</h2>"
    for label, df in opt_tables.items():
        cols = [c for c in ["channel", "current_spend_£", "optimal_spend_£_mean",
                             "pct_change_spend", "current_response", "response_mean",
                             "pct_change_response", "optimal_roi"] if c in df.columns]
        if not cols:
            cols = list(df.columns)
        opt_html += f"<h3>{label}</h3><div class='card'>{_df_to_html(df[cols])}</div>"
    sections.append(opt_html)

    body = (f'<div class="wrap">  '
            f'<h1 style="color:#1a252f;font-size:1.4em;">Bayesian Media Mix Model Report</h1>'
            f'<p style="color:#7f8c8d;font-size:0.87em;">'
            f'Generated {now} &nbsp;·&nbsp; {freq} model &nbsp;·&nbsp; {quality}</p>'
            + "".join(sections) + "</div>")

    html = (f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>MMM Report</title><style>{_CSS}</style></head>'
            f'<body>{nav}{body}</body></html>')

    report_path = out_dir / "mmm_report.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"\n[REPORT] Saved: {report_path}")
    logger.info("[REPORT] Open in any browser — all charts are embedded, no other files needed.")
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# Command: sequential
# ─────────────────────────────────────────────────────────────────────────────

def cmd_sequential(args):
    """
    Multi-period sequential budget optimiser (1M / 3M / 6M / 1Y horizons).

    Reads  optimisation.multi_period_scenarios  from the YAML config.
    For each scenario:
      - Calls optimise_budget_sequential() with the configured horizon,
        budget scale, CPA floor, and share / flex bounds
      - Writes per-scenario CSVs (summary / channels / monthly) to
        <out_dir>/sequential_opt/
      - Appends all results to <out_dir>/sequential_results.xlsx with one
        sheet-group per scenario (SeqOpt_Summary, SeqOpt_Channels_<name>,
        SeqOpt_Monthly_<name>)
    """
    from optimisation import (
        optimise_budget_sequential,
        cpp_weights_array,
        build_cpp_map,
        format_cpp_summary,
        _periods_in_window,
    )

    out_dir, best, prep, cfg, cpp_map, cpp_w = _setup(args)
    opt_cfg   = cfg.get("optimisation") or {}
    mp_scens  = opt_cfg.get("multi_period_scenarios") or []

    if not mp_scens:
        logger.warning(
            "[SEQUENTIAL] No multi_period_scenarios defined in config — "
            "add a 'multi_period_scenarios' block under 'optimisation:' in the YAML."
        )
        return {}

    seq_dir = out_dir / "sequential_opt"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # ── spend_in_currency is deprecated — metric type is now auto-detected ──
    if "spend_in_currency" in opt_cfg:
        logger.warning(
            "  [SEQ] 'spend_in_currency' in config is deprecated and ignored. "
            "Metric type (spend vs impressions/clicks/GRPs) is now detected "
            "automatically per channel from column names and prep metadata."
        )

    # ── Per-channel share bounds from config ────────────────────────────────
    share_bounds = opt_cfg.get("channel_share_bounds") or {}
    ch_share_min = {
        ch: float(v["min_pct"])
        for ch, v in share_bounds.items() if "min_pct" in v
    }
    ch_share_max = {
        ch: float(v["max_pct"])
        for ch, v in share_bounds.items() if "max_pct" in v
    }

    # ── Dynamic step size ───────────────────────────────────────────────────
    round_budget_pct = opt_cfg.get("round_budget_pct") or None
    step_size_cfg    = opt_cfg.get("step_size") or None

    # ── Derive observed mean monthly spend ─────────────────────────────────
    frequency = prep.get("frequency", "weekly")
    n_train   = len(prep["train_idx"])
    X_rw = prep["X_media_raw"][prep["train_idx"]]
    cur_pp = X_rw[:, 0, :].mean(axis=0) if X_rw.ndim == 3 else X_rw.mean(axis=0)
    n_ppm  = max(1, round(_periods_in_window("monthly", frequency, n_train) / 1.0))
    monthly_obs = float((cur_pp * cpp_w).sum()) * n_ppm

    # Flex bounds (absolute £ per whole period; shared with forward optimiser)
    flex       = float(opt_cfg.get("default_spend_flex", 0.30))
    ch_constr  = opt_cfg.get("channel_constraints") or {}

    all_summaries   = []
    all_channels    = []
    all_monthly     = []
    results_by_name = {}

    from optimisation import minimise_spend_sequential

    sc_number = 0
    for sc in mp_scens:
        sc_number += 1
        name     = sc.get("name", f"{sc.get('n_months', 1)}-Month")
        sc_type  = sc.get("type", "forward").lower()
        n_months = int(sc.get("n_months", 1))
        b_scale  = float(sc.get("budget_scale", 1.0))
        t_cpa    = sc.get("target_cpa") or None
        n_samp   = int(sc.get("n_samples", 1))
        t_resp   = sc.get("target_response") or None
        t_budget = sc.get("total_budget") or None

        # Budget: explicit total_budget overrides budget_scale
        if sc_type == "forward":
            total_b = float(t_budget) if t_budget else monthly_obs * n_months * b_scale
        else:
            total_b = None  # reverse: budget is the unknown

        # Per-period flex bounds (whole-period £, used by forward only)
        ch_min, ch_max = {}, {}
        for j, ch in enumerate(prep["spend_cols"]):
            sp_period = float(cur_pp[j] * cpp_w[j]) * n_ppm * n_months
            cst = ch_constr.get(ch, {}) or {}
            if cst.get("fixed", False):
                ch_min[ch] = ch_max[ch] = sp_period
            else:
                lo = float(cst.get("min_pct", flex))
                hi = float(cst.get("max_pct", flex))
                ch_min[ch] = max(0.0, sp_period * (1.0 - lo))
                ch_max[ch] = sp_period * (1.0 + hi)

        # Shared keyword args for both forward and reverse
        _shared_kw = dict(
            n_months          = n_months,
            n_samples         = n_samp,
            channel_min       = ch_min or None,
            channel_max       = ch_max or None,
            channel_share_min = ch_share_min or None,
            channel_share_max = ch_share_max or None,
            cpp_weights       = cpp_w,
            target_cpa        = float(t_cpa) if t_cpa else None,
            step_size         = float(step_size_cfg) if step_size_cfg else None,
            round_budget_pct  = float(round_budget_pct) if round_budget_pct else None,
        )

        if sc_type == "reverse":
            logger.info(
                f"\n[SEQUENTIAL] Scenario {sc_number}: {name} | type=reverse | "
                f"n_months={n_months} | target={t_resp or 'observed mean'}"
            )
            try:
                df_sum, df_ch, df_mon = minimise_spend_sequential(
                    best, prep,
                    target_response = float(t_resp) if t_resp else None,
                    **_shared_kw,
                )
            except Exception as e:
                logger.error(f"  [SEQUENTIAL] {name} failed: {e}")
                import traceback; logger.debug(traceback.format_exc())
                continue
        else:
            logger.info(
                f"\n[SEQUENTIAL] Scenario {sc_number}: {name} | type=forward | "
                f"n_months={n_months} | budget=£{total_b:,.0f}"
            )
            try:
                df_sum, df_ch, df_mon = optimise_budget_sequential(
                    best, prep,
                    total_budget = total_b,
                    **_shared_kw,
                )
            except Exception as e:
                logger.error(f"  [SEQUENTIAL] {name} failed: {e}")
                import traceback; logger.debug(traceback.format_exc())
                continue

        slug = name.replace(" ", "_").replace("/", "").replace("-", "").replace("+", "plus").replace("%", "pct")

        # Tag with scenario name and number
        df_sum["scenario"] = name
        df_sum["scenario_type"] = sc_type
        df_ch["scenario"]  = name
        df_ch["scenario_type"] = sc_type
        df_mon["scenario"] = name

        # Budget period label for the output table
        bp_label = {1: "monthly", 3: "quarterly", 6: "half-year", 12: "annual"}.get(
            n_months, f"{n_months}-month"
        )
        df_ch["budget_period"] = bp_label

        # Save individual CSVs
        df_sum.to_csv(seq_dir / f"seq_{slug}_summary.csv",  index=False)
        df_ch.to_csv( seq_dir / f"seq_{slug}_channels.csv", index=False)
        df_mon.to_csv(seq_dir / f"seq_{slug}_monthly.csv",  index=False)
        logger.info(f"  Saved: seq_{slug}_summary/channels/monthly.csv")

        all_summaries.append(df_sum)
        all_channels.append(df_ch)
        all_monthly.append(df_mon)
        results_by_name[name] = (df_sum, df_ch, df_mon)

    if not results_by_name:
        logger.warning("[SEQUENTIAL] No scenarios completed successfully.")
        return {}

    # ── Build combined 4-scenario table in the brief's format ───────────────
    df_combined = _build_combined_scenario_table(results_by_name)
    df_combined.to_csv(seq_dir / "optimisation_scenarios.csv", index=False)
    logger.info(f"  Saved: sequential_opt/optimisation_scenarios.csv")

    # ── Write combined Excel workbook ───────────────────────────────────────
    _write_sequential_excel(
        results_by_name,
        all_summaries, all_channels, all_monthly,
        out_dir / "sequential_results.xlsx",
    )

    # ── Write combined table + all tabs to model_results.xlsx ───────────────
    try:
        from export import _write_scenario_table_to_model_excel
        model_dir  = out_dir / "model_results"
        model_xlsx = model_dir / "model_results.xlsx"
        if model_xlsx.exists():
            _write_scenario_table_to_model_excel(model_xlsx, df_combined)
            logger.info(f"  Written Optimisation_Scenarios tab to {model_xlsx.name}")
        else:
            logger.warning("  model_results.xlsx not found — run 'curves' first.")
    except Exception as e:
        logger.warning(f"  Could not write to model_results.xlsx (non-fatal): {e}")
        import traceback; logger.debug(traceback.format_exc())

    # ── Waterfall chart + efficiency scorecard (first forward scenario) ──────
    try:
        from analysis import plot_waterfall_chart, plot_efficiency_scorecard, compute_response_curves
        opt_dir = out_dir / "sequential_opt"
        # Use first forward scenario for the presentation charts
        for sc_name, (df_sum, df_ch, df_mon) in results_by_name.items():
            if df_ch.empty:
                continue
            sc_type = str(df_ch.get("scenario_type", "forward").iloc[0] if "scenario_type" in df_ch.columns else "forward")
            if sc_type == "forward":
                plot_waterfall_chart(df_ch, opt_dir, title=f"Budget Reallocation  |  {sc_name}")
                # For scorecard, load response curves if available
                rc_csv = out_dir / "response_curves" / "response_curves.csv"
                df_rc_sc = pd.read_csv(rc_csv) if rc_csv.exists() else None
                plot_efficiency_scorecard(df_ch, df_rc_sc, opt_dir)
                logger.info(f"  Saved waterfall + scorecard for scenario: {sc_name}")
                break
    except Exception as e:
        logger.warning(f"  Could not generate waterfall/scorecard (non-fatal): {e}")
        import traceback; logger.debug(traceback.format_exc())

    logger.info("[SEQUENTIAL] Done.")
    return results_by_name


def _build_combined_scenario_table(results_by_name: dict) -> pd.DataFrame:
    """
    Build the combined 4-scenario table matching the brief's Excel format:

    Scenario | Channel | Budget period | Current Spend | Optimal spend |
    % Change in spend | Current Response | Optimal Response | % Change in response

    One block of rows per scenario (4 channels + 1 Total row each).
    """
    rows = []
    for sc_name, (df_sum, df_ch, df_mon) in results_by_name.items():
        if df_ch.empty:
            continue
        sc_type   = str(df_ch["scenario_type"].iloc[0]) if "scenario_type" in df_ch.columns else "forward"
        bp_label  = str(df_ch["budget_period"].iloc[0]) if "budget_period" in df_ch.columns else "monthly"
        disp_name = sc_name

        # Channel rows
        for _, r in df_ch.iterrows():
            ch = r.get("channel", "")
            if ch in ("scenario", "scenario_type", "budget_period"):
                continue

            # For reverse scenario the "optimal spend" is the MINIMUM spend found
            opt_spend = float(r.get("opt_spend_gbp", 0.0))
            cur_spend = float(r.get("current_spend_gbp", 0.0))
            opt_resp  = float(r.get("opt_response",   0.0))
            cur_resp  = float(r.get("current_response", 0.0))

            rows.append({
                "Scenario"              : disp_name,
                "Channel"               : ch,
                "Budget period"         : bp_label,
                "Current Spend"         : round(cur_spend, 2),
                "Optimal spend"         : round(opt_spend, 2),
                "% Change in spend"     : round((opt_spend - cur_spend) / (cur_spend + 1e-12) * 100, 2),
                "Current Response"      : round(cur_resp,  4),
                "Optimal Response"      : round(opt_resp,  4),
                "% Change in response"  : round((opt_resp - cur_resp) / (cur_resp + 1e-12) * 100, 2),
            })

        # Total row — spend sums from per-channel table (exact).
        # Response totals come from df_mon (computed by _monthly_response_total
        # with ALL channels active) not from summing per-channel values, because
        # summing per-channel figures can double-count carry-on effects and
        # softplus(0) baseline terms for zero-spend channels.
        tot_cur_sp  = df_ch["current_spend_gbp"].sum()  if "current_spend_gbp"  in df_ch.columns else 0.0
        tot_opt_sp  = df_ch["opt_spend_gbp"].sum()      if "opt_spend_gbp"      in df_ch.columns else 0.0
        tot_cur_r   = float(df_mon["current_response"].sum()) if ("current_response" in df_mon.columns and not df_mon.empty) else df_ch["current_response"].sum()
        tot_opt_r   = float(df_mon["opt_response"].sum())     if ("opt_response"     in df_mon.columns and not df_mon.empty) else df_ch["opt_response"].sum()
        rows.append({
            "Scenario"              : disp_name,
            "Channel"               : "Total",
            "Budget period"         : bp_label,
            "Current Spend"         : round(float(tot_cur_sp), 2),
            "Optimal spend"         : round(float(tot_opt_sp), 2),
            "% Change in spend"     : round((float(tot_opt_sp) - float(tot_cur_sp)) / (float(tot_cur_sp) + 1e-12) * 100, 2),
            "Current Response"      : round(float(tot_cur_r),  4),
            "Optimal Response"      : round(float(tot_opt_r),  4),
            "% Change in response"  : round((float(tot_opt_r) - float(tot_cur_r)) / (float(tot_cur_r) + 1e-12) * 100, 2),
        })

    return pd.DataFrame(rows)


def _write_sequential_excel(
    results_by_name : dict,
    all_summaries   : list,
    all_channels    : list,
    all_monthly     : list,
    out_path        : "Path",
) -> None:
    """
    Write sequential optimisation results to a single Excel workbook.

    Sheets written:
      Overview          — all scenarios, one row each (total response, spend, lift vs BAU)
      Channels_<name>   — per-channel detail for each scenario
      Monthly_<name>    — month-by-month response for each scenario
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.chart import ScatterChart, Reference, Series
        from openpyxl.utils import get_column_letter
    except ImportError:
        logger.warning("openpyxl not installed — skipping sequential Excel export.")
        return

    _HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
    _HDR_FONT    = Font(bold=True, color="FFFFFF", size=10)
    _BOLD        = Font(bold=True, size=10)
    _EVEN_FILL   = PatternFill("solid", fgColor="EBF3FA")
    _THIN        = Side(border_style="thin", color="C5C5C5")
    _BORDER      = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
    _PCT_FMT     = '0.0%'
    _GBP_FMT     = '£#,##0'
    _DEC_FMT     = '#,##0.00'

    def _write_df(ws, df, start_row=1, start_col=1):
        """Write a DataFrame to a worksheet starting at (start_row, start_col)."""
        for ci, col in enumerate(df.columns, start=start_col):
            cell = ws.cell(row=start_row, column=ci, value=col)
            cell.font  = _HDR_FONT
            cell.fill  = _HEADER_FILL
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
            cell.border = _BORDER
        for ri, (_, row) in enumerate(df.iterrows(), start=start_row + 1):
            fill = _EVEN_FILL if ri % 2 == 0 else PatternFill()
            for ci, col in enumerate(df.columns, start=start_col):
                val  = row[col]
                cell = ws.cell(row=ri, column=ci,
                               value=None if (isinstance(val, float) and np.isnan(val)) else val)
                cell.fill   = fill
                cell.border = _BORDER
                cell.alignment = Alignment(horizontal="right" if isinstance(val, (int, float)) else "left")
                # Number formats
                if isinstance(val, float):
                    if "pct" in col.lower() or "share" in col.lower():
                        cell.number_format = _PCT_FMT
                    elif "spend" in col.lower() or "£" in col.lower() or "budget" in col.lower():
                        cell.number_format = _GBP_FMT
                    else:
                        cell.number_format = _DEC_FMT
        # Auto-size columns
        for ci in range(start_col, start_col + len(df.columns)):
            col_ltr = get_column_letter(ci)
            ws.column_dimensions[col_ltr].width = max(
                12,
                min(40, max(
                    len(str(df.columns[ci - start_col])) + 2,
                    *(len(str(v)) + 2 for v in df.iloc[:, ci - start_col])
                ))
            )

    wb = Workbook()
    ws_ov = wb.active
    ws_ov.title = "Overview"

    # ── Overview sheet ──────────────────────────────────────────────────────
    ov_rows = []
    for name, (df_sum, df_ch, df_mon) in results_by_name.items():
        row_dict = {"scenario": name}
        if not df_sum.empty:
            for col in df_sum.columns:
                if col != "scenario":
                    row_dict[col] = df_sum.iloc[0][col]
        ov_rows.append(row_dict)
    df_ov = pd.DataFrame(ov_rows)
    if not df_ov.empty:
        _write_df(ws_ov, df_ov)

    # ── Per-scenario sheets ─────────────────────────────────────────────────
    for name, (df_sum, df_ch, df_mon) in results_by_name.items():
        slug = name.replace(" ", "_").replace("-", "").replace("/", "")[:25]

        # Channels tab
        ws_ch = wb.create_sheet(f"Channels_{slug}")
        if not df_ch.empty:
            _write_df(ws_ch, df_ch)

        # Monthly tab
        ws_mo = wb.create_sheet(f"Monthly_{slug}")
        if not df_mon.empty:
            _write_df(ws_mo, df_mon)

            # Add a simple response-over-time chart (opt vs BAU vs current)
            try:
                chart = ScatterChart()
                chart.title  = f"{name} — Monthly Response"
                chart.style  = 10
                chart.height = 12
                chart.width  = 22
                chart.x_axis.title = "Month"
                chart.y_axis.title = "Response (signups)"

                n_rows   = len(df_mon)
                data_row = ws_mo.max_row - n_rows + 1   # first data row

                # Month column
                x_ref = Reference(ws_mo, min_col=1, min_row=data_row,
                                   max_row=ws_mo.max_row)

                for label, col_name in [
                    ("Optimised", "opt_response"),
                    ("BAU",       "bau_response"),
                    ("Current",   "current_response"),
                ]:
                    col_idx = list(df_mon.columns).index(col_name) + 1 if col_name in df_mon.columns else None
                    if col_idx is None:
                        continue
                    y_ref = Reference(ws_mo, min_col=col_idx, min_row=data_row,
                                      max_row=ws_mo.max_row)
                    s = Series(y_ref, xvalues=x_ref, title=label)
                    s.smooth = True
                    chart.series.append(s)

                chart_row = ws_mo.max_row + 3
                ws_mo.add_chart(chart, f"A{chart_row}")
            except Exception as _ce:
                logger.debug(f"Monthly chart failed for {name}: {_ce}")

    wb.save(out_path)
    logger.info(f"  [SEQUENTIAL] Excel written: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Command: all
# ─────────────────────────────────────────────────────────────────────────────

def cmd_all(args):
    """Run all analysis commands in sequence."""
    logger.info("=" * 60)
    logger.info("  FULL ANALYSIS (curves → sequential → scenarios → report)")
    logger.info("=" * 60)
    for fn, label in [
        (cmd_curves,     "Response Curves"),
        (cmd_sequential, "Sequential Optimisation (all 4 scenarios)"),
        (cmd_scenarios,  "Scenario Comparison"),
        (cmd_report,     "HTML Report"),
    ]:
        logger.info(f"\n{'─'*50}\n  {label}\n{'─'*50}")
        try:
            fn(args)
        except Exception as e:
            logger.error(f"  {label} failed: {e}")
    logger.info("\n[ALL] Complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    setup_logger()

    parser = argparse.ArgumentParser(
        description="Post-pipeline MMM analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  curves      Compute response curves (run this first)
  sequential  Run all 4 optimisation scenarios (forward x3 + reverse)
  scenarios   Side-by-side scenario comparison charts
  report      Self-contained HTML report
  all         Run everything in sequence

Examples:
  python analyze.py curves     --config config_hier.yaml
  python analyze.py sequential --config config_hier.yaml
  python analyze.py all        --config config_hier.yaml
        """,
    )
    parser.add_argument(
        "command",
        choices=["curves", "sequential", "scenarios", "report", "all"],
    )
    parser.add_argument("--config",        type=str, default=None, help="YAML config path")
    parser.add_argument("--output-dir",    type=str, default=None, dest="output_dir",
                        help="Output directory (overrides YAML output.dir)")
    parser.add_argument("--budget-period", type=str, default=None, dest="budget_period",
                        help="Budget period: monthly | quarterly | yearly | per_period")
    parser.add_argument("--pcts",   type=float, nargs="+", default=None,
                        help="Budget % changes for report scenarios, e.g. --pcts -20 -10 10 20")
    parser.add_argument("--target", type=float, default=None,
                        help="Add a reverse scenario with this response target (report only)")
    parser.add_argument("--spend-flex", type=float, default=None, dest="spend_flex",
                        help="Per-channel ±flex fraction (default from YAML or 0.40)")
    args = parser.parse_args()

    dispatch = {
        "curves"     : cmd_curves,
        "sequential" : cmd_sequential,
        "scenarios"  : cmd_scenarios,
        "report"     : cmd_report,
        "all"        : cmd_all,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
