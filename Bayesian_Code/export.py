# export.py
# ─────────────────────────────────────────────────────────────────────────────
# Exports all pipeline results to disk:
#   1. best_model.json
#   2. scan_stage0.csv / scan_stage1b.csv / scan_stage2.csv
#   3. channel_contributions.csv
#   4. component_contributions.csv
#   5. media_share_weekly.csv
#   6. az_summary.csv + model_summary.txt
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import arviz as az

from config import DataConfig
from metrics import quality_flag
from priors import (
    compute_return_index, export_return_index,
    export_channel_prior_config,
)

logger = logging.getLogger("MMM")


def _excel_style():
    """Return shared openpyxl style objects."""
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    HDR_FILL  = PatternFill("solid", fgColor="1F4E79")
    HDR_FONT  = Font(bold=True, color="FFFFFF", size=11)
    HDR_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)
    THIN      = Side(style="thin", color="BFBFBF")
    BORDER    = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    return HDR_FILL, HDR_FONT, HDR_ALIGN, BORDER


def _add_df_to_sheet(ws, df: pd.DataFrame):
    """Write a DataFrame to an openpyxl worksheet with styled headers."""
    from openpyxl.utils import get_column_letter
    HDR_FILL, HDR_FONT, HDR_ALIGN, BORDER = _excel_style()
    for ci, col in enumerate(df.columns, start=1):
        c = ws.cell(row=1, column=ci, value=str(col))
        c.font = HDR_FONT; c.fill = HDR_FILL
        c.alignment = HDR_ALIGN; c.border = BORDER
    for ri, row in enumerate(df.itertuples(index=False), start=2):
        for ci, val in enumerate(row, start=1):
            c = ws.cell(row=ri, column=ci,
                        value=val if not (isinstance(val, float) and pd.isna(val)) else None)
            c.border = BORDER
            if isinstance(val, float):
                c.number_format = "#,##0.00"
    for ci in range(1, len(df.columns) + 1):
        col_letter = get_column_letter(ci)
        max_len = max(
            len(str(df.columns[ci - 1])),
            *[len(str(ws.cell(row=r, column=ci).value or "")) for r in range(2, min(ws.max_row + 1, 1002))],
        )
        ws.column_dimensions[col_letter].width = min(max_len + 2, 40)
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 30


def _build_contributions_df(model_dir) -> pd.DataFrame:
    """
    Build the combined Media + Non-media contributions table.

    All columns triangulate exactly:
      - Unified Share (%): every row = avg_weekly_contrib / pred_weekly_avg * 100.
        All rows sum to 100%.
      - Total Contribution: avg_weekly * n_weeks. All rows sum to pred_weekly * n_weeks.
      - ROI: avg_weekly_signups / avg_weekly_spend_GBP * 1000  (signups per GBP1k).

    Non-media anchor:
      non_media_weekly = pred_weekly_avg - total_media_weekly  (exact subtraction).
      Split baseline / seasonality proportionally by mean_abs_effect from
      component_contributions.csv (z-space magnitude ratio, sign-invariant).
    """
    import json as _json

    # ── n_train_weeks ─────────────────────────────────────────
    n_weeks = 0
    bm_path = model_dir / "best_model.json"
    if bm_path.exists():
        with open(bm_path) as _fh:
            _bm = _json.load(_fh)
        n_weeks = int(_bm.get("n_weeks", 0))

    # ── Predicted weekly avg (anchor for unified share + non-media) ──
    pred_weekly_avg = None
    avp_csv = model_dir / "actual_vs_predicted.csv"
    if avp_csv.exists():
        df_avp = pd.read_csv(avp_csv)
        if "predicted" in df_avp.columns:
            pred_weekly_avg = float(df_avp["predicted"].mean())

    # ── GBP spend per channel from return_index.csv ──────────
    spend_map: dict = {}   # channel_id -> avg_weekly_GBP_spend
    ri_csv = model_dir / "return_index.csv"
    if ri_csv.exists():
        df_ri = pd.read_csv(ri_csv)
        for _, r in df_ri.iterrows():
            spend_map[str(r["channel_id"])] = float(r.get("mean_weekly_spend", 0) or 0)

    # ── Component contributions (for non-media split) ─────────
    df_comp_all = None
    comp_csv = model_dir / "component_contributions.csv"
    if comp_csv.exists():
        df_comp_all = pd.read_csv(comp_csv)

    rows = []
    total_media_weekly = 0.0

    # ── Media channels ─────────────────────────────────────────
    ch_csv = model_dir / "channel_contributions.csv"
    if ch_csv.exists():
        df_ch = pd.read_csv(ch_csv)
        for _, r in df_ch.iterrows():
            ch_name = str(r["channel"])
            mean_wk = float(r["mean_contribution"])   # already in original response scale
            total_media_weekly += mean_wk

            total_c = round(mean_wk * n_weeks, 1) if n_weeks else None

            # Unified share = this channel's weekly contrib / total predicted weekly * 100
            # Deferred until after we know pred_weekly_avg — filled below
            _unified = round(mean_wk / pred_weekly_avg * 100, 2) if pred_weekly_avg else None

            # HDI: hdi_90_low/high from channel_contributions are per-week posterior HDI
            # Total period HDI ~ per-week HDI * n_weeks (standard approximation)
            hdi_lo = round(float(r["hdi_90_low"])  * n_weeks, 1) if n_weeks else round(float(r["hdi_90_low"]),  2)
            hdi_hi = round(float(r["hdi_90_high"]) * n_weeks, 1) if n_weeks else round(float(r["hdi_90_high"]), 2)

            wk_sp  = spend_map.get(ch_name)
            tot_sp = round(wk_sp * n_weeks, 2) if (wk_sp and n_weeks) else None
            # ROI: signups per GBP1000 = (weekly signups / weekly GBP spend) * 1000
            roi    = round(mean_wk / wk_sp * 1000, 2) if (wk_sp and wk_sp > 0) else None

            rows.append({
                "Type"                        : "Media",
                "Channel"                     : ch_name,
                "Total Contribution"          : total_c,
                "Avg Weekly Contrib"          : round(mean_wk, 2),
                "Unified Share (%)"           : _unified,
                "Avg Weekly Spend (GBP)"      : round(wk_sp, 2) if wk_sp is not None else None,
                "Total Spend (GBP)"           : tot_sp,
                "ROI (signups per GBP1k)"     : roi,
                "HDI 90% Low (total period)"  : hdi_lo,
                "HDI 90% High (total period)" : hdi_hi,
            })

    # ── Non-media components ────────────────────────────────────
    if df_comp_all is not None and pred_weekly_avg is not None:
        # non-media = exact subtraction so all rows sum to predicted
        non_media_wk = pred_weekly_avg - total_media_weekly

        df_nm = df_comp_all[df_comp_all["component"] != "media_total"].copy()
        total_abs_nm = df_nm["mean_abs_effect"].sum() + 1e-12   # sum of |z| magnitudes

        for _, r in df_nm.iterrows():
            abs_eff   = float(r.get("mean_abs_effect", abs(float(r.get("mean_effect", 0)))))
            comp_frac = abs_eff / total_abs_nm                  # share within non-media
            act_wk    = non_media_wk * comp_frac                # original scale weekly

            total_c  = round(act_wk * n_weeks, 1) if n_weeks else None
            _unified = round(act_wk / pred_weekly_avg * 100, 2)

            # Scale z-space HDI bounds to original scale using same ratio
            # scale = act_wk / abs_eff  means: 1 unit of z-effect = scale signups/week
            scale     = act_wk / abs_eff if abs_eff > 0 else 1.0
            hdi_lo_wk = abs(float(r["hdi_90_low"]))  * scale
            hdi_hi_wk = abs(float(r["hdi_90_high"])) * scale
            hdi_lo    = round(min(hdi_lo_wk, hdi_hi_wk) * n_weeks, 1) if n_weeks else round(min(hdi_lo_wk, hdi_hi_wk), 2)
            hdi_hi    = round(max(hdi_lo_wk, hdi_hi_wk) * n_weeks, 1) if n_weeks else round(max(hdi_lo_wk, hdi_hi_wk), 2)

            rows.append({
                "Type"                        : r.get("group", "non_media"),
                "Channel"                     : r["component"],
                "Total Contribution"          : total_c,
                "Avg Weekly Contrib"          : round(act_wk, 2),
                "Unified Share (%)"           : _unified,
                "Avg Weekly Spend (GBP)"      : None,
                "Total Spend (GBP)"           : None,
                "ROI (signups per GBP1k)"     : None,
                "HDI 90% Low (total period)"  : hdi_lo,
                "HDI 90% High (total period)" : hdi_hi,
            })

    return pd.DataFrame(rows)


def _write_model_excel(best: Dict, prep: Dict, model_dir: "Path", summary: Dict) -> None:
    """Write a consolidated model_results.xlsx with multiple tabs."""
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # ── Sheet 1: Model Summary ────────────────────────────────
    metrics = summary.get("metrics", {})
    rows_sum = [
        ("Run timestamp",    summary.get("run_timestamp", "")),
        ("Model key",        summary.get("model_key", "")),
        ("Adstock type",     summary.get("adstock_type", "")),
        ("Saturation",       summary.get("saturation", "")),
        ("Max lag",          summary.get("max_lag", "")),
        ("Fourier order",    summary.get("fourier_order", "")),
        ("Draws",            summary.get("draws", "")),
        ("Chains",           summary.get("chains", "")),
        ("Hierarchical",     summary.get("use_hierarchical", "")),
        ("N weeks",          summary.get("n_weeks", "")),
        ("Channels",         ", ".join(summary.get("spend_channels", []))),
        ("",                 ""),
        ("— Fit Metrics —",  ""),
        ("MAPE (%)",         metrics.get("mape")),
        ("RMSE",             metrics.get("rmse")),
        ("R²",               metrics.get("r2")),
        ("Pearson r",        metrics.get("pearson_r")),
        ("Max R-hat",        metrics.get("max_rhat")),
        ("Mean R-hat",       metrics.get("mean_rhat")),
        ("ESS bulk",         metrics.get("ess_bulk")),
        ("ESS tail",         metrics.get("ess_tail")),
        ("Divergences",      metrics.get("divergences")),
        ("Divergence rate",  metrics.get("div_rate")),
        ("LOO-IC",           metrics.get("loo_ic")),
        ("WAIC",             metrics.get("waic")),
        ("Pareto-k max",     metrics.get("pareto_k_max")),
        ("Quality flag",     metrics.get("quality_flag")),
    ]
    ws_sum = wb.create_sheet("Model Summary")
    ws_sum.column_dimensions["A"].width = 22
    ws_sum.column_dimensions["B"].width = 30
    _, HDR_FONT, _, _ = _excel_style()
    for ri, (label, value) in enumerate(rows_sum, start=1):
        ws_sum.cell(row=ri, column=1, value=label).font = Font(bold=True)
        ws_sum.cell(row=ri, column=2, value=value)

    # ── Sheet 2: Actual vs Predicted (weekly) ────────────────
    avp_csv = model_dir / "actual_vs_predicted.csv"
    if avp_csv.exists():
        ws_avp = wb.create_sheet("Actual vs Predicted")
        _add_df_to_sheet(ws_avp, pd.read_csv(avp_csv))

    # ── Sheet 3: Weekly Decomposition ────────────────────────
    wd_csv = model_dir / "weekly_decomposition.csv"
    if wd_csv.exists():
        ws_wd = wb.create_sheet("Weekly Decomposition")
        _add_df_to_sheet(ws_wd, pd.read_csv(wd_csv))

    # ── Sheet 4: Media & Non-Media Contributions ─────────────
    df_contrib = _build_contributions_df(model_dir)
    if not df_contrib.empty:
        ws_ch = wb.create_sheet("Media Contributions")
        _add_df_to_sheet(ws_ch, df_contrib)

    # ── Sheet 5: Channel Transform Summary ───────────────────
    ts_csv = model_dir / "channel_transform_summary.csv"
    if ts_csv.exists():
        ws_ts = wb.create_sheet("Channel Transform")
        _add_df_to_sheet(ws_ts, pd.read_csv(ts_csv))

    # ── Sheet 6: Media Share Weekly ───────────────────────────
    ms_csv = model_dir / "media_share_weekly.csv"
    if ms_csv.exists():
        ws_ms = wb.create_sheet("Media Share Weekly")
        _add_df_to_sheet(ws_ms, pd.read_csv(ms_csv))

    # ── Sheet 7: Return Index ─────────────────────────────────
    ri_csv = model_dir / "return_index.csv"
    if ri_csv.exists():
        ws_ri = wb.create_sheet("Return Index")
        _add_df_to_sheet(ws_ri, pd.read_csv(ri_csv))

    # ── Sheet 8: ArviZ Posterior Summary ─────────────────────
    az_csv = model_dir / "az_summary.csv"
    if az_csv.exists():
        ws_az = wb.create_sheet("Posterior Summary")
        _add_df_to_sheet(ws_az, pd.read_csv(az_csv).head(200))

    xl_path = model_dir / "model_results.xlsx"
    wb.save(str(xl_path))
    logger.info(f"  Saved: {xl_path}")


def _append_optimiser_sheets_to_model_excel(model_dir, opt_dir) -> None:
    """
    Called by the optimiser scripts to add their tabs to model_results.xlsx.
    Creates the workbook if it doesn't exist yet (shouldn't normally happen).
    """
    import openpyxl
    xl_path = model_dir / "model_results.xlsx"
    if not xl_path.exists():
        logger.warning(f"[EXCEL] model_results.xlsx not found at {xl_path} — skipping optimiser append")
        return

    wb = openpyxl.load_workbook(str(xl_path))

    sheets = [
        ("Forward Optimisation",  "budget_optimisation.csv"),
        ("Greedy Allocation",     "greedy_budget_allocation.csv"),
        ("Greedy Path",           "greedy_allocation_path.csv"),
        ("Reverse Optimisation",  "spend_target_optimisation.csv"),
        ("Saturation Analysis",   "saturation_analysis.csv"),
    ]
    added = []
    for sheet_name, fname in sheets:
        p = opt_dir / fname
        if not p.exists():
            continue
        if sheet_name in wb.sheetnames:
            del wb[sheet_name]
        ws = wb.create_sheet(sheet_name)
        _add_df_to_sheet(ws, pd.read_csv(p))
        added.append(sheet_name)

    if added:
        wb.save(str(xl_path))
        logger.info(f"  Updated model_results.xlsx — added tabs: {added}")


def _append_rc_multiperiod_to_model_excel(model_excel_path: "Path", rc_dir: "Path") -> None:
    """
    Append multi-period response curve tabs to model_results.xlsx.

    Reads  response_curves_multiperiod.csv  from  rc_dir  and adds:
      RC_MultiPeriod_Data  — raw data table (all horizons, all channels)
      RC_Channel_Charts    — one ScatterChart per channel overlaying 1M/3M/6M/12M curves

    Call after cmd_curves() has written the CSV.
    """
    import openpyxl
    from openpyxl.chart import ScatterChart, Reference, Series

    csv_path = rc_dir / "response_curves_multiperiod.csv"
    if not csv_path.exists():
        logger.debug("  RC multiperiod CSV not found — skipping RC tab append")
        return

    xl_path = Path(model_excel_path)
    if not xl_path.exists():
        logger.warning(f"  model_results.xlsx not found at {xl_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f"  Could not read multiperiod RC CSV: {e}")
        return

    wb = openpyxl.load_workbook(str(xl_path))

    # ── Tab 1: Raw data ───────────────────────────────────────────────────────
    for sname in ["RC_MultiPeriod_Data"]:
        if sname in wb.sheetnames:
            del wb[sname]
    ws_data = wb.create_sheet("RC_MultiPeriod_Data")
    _add_df_to_sheet(ws_data, df)
    logger.info("  Added RC_MultiPeriod_Data tab")

    # ── Tab 2: One chart per channel — all horizons overlaid ──────────────────
    if "RC_Channel_Charts" in wb.sheetnames:
        del wb["RC_Channel_Charts"]
    ws_ch = wb.create_sheet("RC_Channel_Charts")
    ws_ch.cell(row=1, column=1, value="Multi-Period Response Curves — by Channel")

    channels = df["channel"].unique().tolist() if "channel" in df.columns else []
    periods  = sorted(df["period_label"].unique().tolist()) if "period_label" in df.columns else []

    chart_row = 3
    for ch in channels:
        chart = ScatterChart()
        chart.title  = ch
        chart.style  = 10
        chart.height = 12
        chart.width  = 22
        chart.x_axis.title = "Period Spend (£)"
        chart.y_axis.title = "Cumulative Signups"

        for period in periods:
            sub = df[(df["channel"] == ch) & (df["period_label"] == period)].sort_values("spend_gbp")
            if sub.empty:
                continue

            # Write data to a temp block on a hidden area of ws_ch
            data_col_start = ws_ch.max_column + 2
            ws_ch.cell(row=1, column=data_col_start,     value=f"{ch}_{period}_spend")
            ws_ch.cell(row=1, column=data_col_start + 1, value=f"{ch}_{period}_signups")
            for ri, (_, row) in enumerate(sub.iterrows(), start=2):
                ws_ch.cell(row=ri, column=data_col_start,     value=float(row["spend_gbp"]))
                ws_ch.cell(row=ri, column=data_col_start + 1, value=float(row["mean_signups"]))

            n_rows = len(sub)
            x_ref  = Reference(ws_ch, min_col=data_col_start,     min_row=2, max_row=1 + n_rows)
            y_ref  = Reference(ws_ch, min_col=data_col_start + 1, min_row=2, max_row=1 + n_rows)
            s = Series(y_ref, xvalues=x_ref, title=period)
            s.smooth = False
            chart.series.append(s)

        ws_ch.add_chart(chart, f"A{chart_row}")
        chart_row += 22   # leave room for next chart

    wb.save(str(xl_path))
    logger.info(f"  Added RC_Channel_Charts tab to {xl_path.name}")


def _append_sequential_to_model_excel(
    model_excel_path : "Path",
    results_by_name  : dict,
) -> None:
    """
    Append sequential-optimiser results to model_results.xlsx.

    Tabs added per scenario:
      SeqOpt_<slug>_Channels  — per-channel spend & response table
      SeqOpt_<slug>_Monthly   — month-by-month response with chart

    Plus one combined tab:
      Monthly_Carryover       — all scenarios side-by-side monthly response
    """
    import openpyxl
    from openpyxl.chart import ScatterChart, Reference, Series

    xl_path = Path(model_excel_path)
    if not xl_path.exists():
        logger.warning(f"  model_results.xlsx not found at {xl_path}")
        return

    wb = openpyxl.load_workbook(str(xl_path))

    # ── Monthly carryover combined tab ────────────────────────────────────────
    if "Monthly_Carryover" in wb.sheetnames:
        del wb["Monthly_Carryover"]
    ws_co = wb.create_sheet("Monthly_Carryover")

    # Build side-by-side: month | opt_<scen> | bau_<scen> for each scenario
    co_rows = []
    for name, (_, _, df_mon) in results_by_name.items():
        slug = name.replace(" ", "_").replace("-", "").replace("/", "")[:20]
        for _, r in df_mon.iterrows():
            co_rows.append({
                "scenario"    : name,
                "month"       : int(r["month"]),
                "opt_response": float(r["opt_response"]),
                "bau_response": float(r.get("bau_response", 0.0)),
                "current_response": float(r.get("current_response", 0.0)),
                "opt_spend_£" : float(r.get("opt_total_spend_£", 0.0)),
                "bau_spend_£" : float(r.get("bau_total_spend_£", 0.0)),
            })
    if co_rows:
        df_co = pd.DataFrame(co_rows)
        _add_df_to_sheet(ws_co, df_co)

        # Add a small chart showing opt vs BAU across all scenarios
        try:
            chart = ScatterChart()
            chart.title  = "Monthly Response — All Scenarios"
            chart.style  = 10
            chart.height = 14
            chart.width  = 28
            chart.x_axis.title = "Cumulative Month"
            chart.y_axis.title = "Response"
            n_rows = len(df_co)
            x_ref  = Reference(ws_co, min_col=2, min_row=2, max_row=n_rows + 1)  # month
            y_ref  = Reference(ws_co, min_col=3, min_row=2, max_row=n_rows + 1)  # opt_response
            s = Series(y_ref, xvalues=x_ref, title="Optimised")
            s.smooth = True
            chart.series.append(s)
            y_ref2 = Reference(ws_co, min_col=4, min_row=2, max_row=n_rows + 1)
            s2 = Series(y_ref2, xvalues=x_ref, title="BAU")
            s2.smooth = True
            chart.series.append(s2)
            ws_co.add_chart(chart, f"A{n_rows + 5}")
        except Exception as _e:
            logger.debug(f"Monthly_Carryover chart failed: {_e}")

    logger.info("  Added Monthly_Carryover tab")

    # ── Per-scenario tabs ──────────────────────────────────────────────────────
    for name, (df_sum, df_ch, df_mon) in results_by_name.items():
        slug = name.replace(" ", "_").replace("-", "").replace("/", "")[:20]

        # Channels tab
        ch_tab = f"SeqOpt_{slug}_Ch"
        if ch_tab in wb.sheetnames:
            del wb[ch_tab]
        ws_cht = wb.create_sheet(ch_tab)
        if not df_ch.empty:
            _add_df_to_sheet(ws_cht, df_ch)

        # Monthly tab with chart
        mo_tab = f"SeqOpt_{slug}_Mo"
        if mo_tab in wb.sheetnames:
            del wb[mo_tab]
        ws_mo = wb.create_sheet(mo_tab)
        if not df_mon.empty:
            _add_df_to_sheet(ws_mo, df_mon)
            try:
                chart = ScatterChart()
                chart.title  = f"{name} — Monthly Response"
                chart.style  = 10
                chart.height = 12
                chart.width  = 22
                chart.x_axis.title = "Month"
                chart.y_axis.title = "Response"
                n_rows = len(df_mon)
                x_ref  = Reference(ws_mo, min_col=1, min_row=2, max_row=n_rows + 1)
                for label, col_name in [("Optimised", "opt_response"),
                                        ("BAU",       "bau_response"),
                                        ("Current",   "current_response")]:
                    if col_name not in df_mon.columns:
                        continue
                    ci = list(df_mon.columns).index(col_name) + 1
                    y_ref = Reference(ws_mo, min_col=ci, min_row=2, max_row=n_rows + 1)
                    s = Series(y_ref, xvalues=x_ref, title=label)
                    s.smooth = True
                    chart.series.append(s)
                ws_mo.add_chart(chart, f"A{n_rows + 5}")
            except Exception as _e:
                logger.debug(f"  Monthly chart for {name} failed: {_e}")

    wb.save(str(xl_path))
    logger.info(f"  Appended sequential optimiser tabs to {xl_path.name}")


def _write_scenario_table_to_model_excel(
    model_excel_path: "Path",
    df_combined: "pd.DataFrame",
) -> None:
    """
    Write the Optimisation_Scenarios tab to model_results.xlsx.

    The tab contains the 4-scenario comparison table with columns:
      Scenario | Channel | Budget period | Current Spend | Optimal spend |
      % Change in spend | Current Response | Optimal Response | % Change in response

    Formatting:
      - Dark-blue / white header row (matches rest of workbook)
      - £#,##0 format for all spend columns
      - +0.0% format for % change columns; green font if ≥ 0, red if < 0
      - Bold + light-grey fill for Total rows
      - Alternating light fill per scenario block for readability
    """
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    xl_path = Path(model_excel_path)
    if not xl_path.exists():
        logger.warning(f"  model_results.xlsx not found at {xl_path} — skipping scenario table")
        return

    if df_combined is None or df_combined.empty:
        logger.warning("  Combined scenario DataFrame is empty — skipping Optimisation_Scenarios tab")
        return

    wb = openpyxl.load_workbook(str(xl_path))

    SHEET_NAME = "Optimisation_Scenarios"
    if SHEET_NAME in wb.sheetnames:
        del wb[SHEET_NAME]
    ws = wb.create_sheet(SHEET_NAME)

    # ── Style constants ──────────────────────────────────────────────────────
    HDR_FILL, HDR_FONT, HDR_ALIGN, THIN_BORDER = _excel_style()

    TOTAL_FILL  = PatternFill("solid", fgColor="D9D9D9")   # light grey for Total rows
    TOTAL_FONT  = Font(bold=True, size=10)

    # Alternating scenario block fills (very subtle)
    BLOCK_FILLS = [
        PatternFill("solid", fgColor="EBF3FB"),   # pale blue
        PatternFill("solid", fgColor="FAFAFA"),   # near-white
        PatternFill("solid", fgColor="EBF3FB"),
        PatternFill("solid", fgColor="FAFAFA"),
    ]
    DATA_FONT   = Font(size=10)
    BODY_ALIGN  = Alignment(horizontal="left",   vertical="center")
    NUM_ALIGN   = Alignment(horizontal="right",  vertical="center")
    CENTER_ALIGN = Alignment(horizontal="center", vertical="center")

    THIN   = Side(style="thin",   color="BFBFBF")
    BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

    # ── Column definitions ────────────────────────────────────────────────────
    COLS = [
        "Scenario",
        "Channel",
        "Budget period",
        "Current Spend",
        "Optimal spend",
        "% Change in spend",
        "Current Response",
        "Optimal Response",
        "% Change in response",
    ]
    COL_WIDTHS = [28, 14, 14, 16, 16, 18, 18, 18, 20]
    # Map column name → index (1-based)
    col_idx = {c: i + 1 for i, c in enumerate(COLS)}

    # ── Header row ────────────────────────────────────────────────────────────
    ws.row_dimensions[1].height = 32
    for i, col_name in enumerate(COLS, start=1):
        c = ws.cell(row=1, column=i, value=col_name)
        c.font      = HDR_FONT
        c.fill      = HDR_FILL
        c.alignment = HDR_ALIGN
        c.border    = BORDER
        ws.column_dimensions[get_column_letter(i)].width = COL_WIDTHS[i - 1]
    ws.freeze_panes = "A2"

    # ── Data rows ─────────────────────────────────────────────────────────────
    # Determine which fill to use per scenario
    scenarios_ordered = list(dict.fromkeys(df_combined["Scenario"].tolist()))
    scen_fill_map = {s: BLOCK_FILLS[i % len(BLOCK_FILLS)]
                     for i, s in enumerate(scenarios_ordered)}

    excel_row = 2
    for _, row in df_combined.iterrows():
        is_total = str(row.get("Channel", "")).strip().lower() == "total"
        scen     = str(row.get("Scenario", ""))
        row_fill = TOTAL_FILL if is_total else scen_fill_map.get(scen, BLOCK_FILLS[0])
        row_font = TOTAL_FONT if is_total else DATA_FONT

        ws.row_dimensions[excel_row].height = 18

        for col_name in COLS:
            ci  = col_idx[col_name]
            raw = row.get(col_name, None)
            c   = ws.cell(row=excel_row, column=ci)
            c.fill   = row_fill
            c.font   = row_font
            c.border = BORDER

            if col_name in ("% Change in spend", "% Change in response"):
                # Store as fraction, format as percentage
                if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                    c.value = None
                else:
                    val = float(raw)
                    c.value          = val / 100.0   # e.g. 20.0 → 0.20
                    c.number_format  = '+0.0%;-0.0%;0.0%'
                    c.alignment      = NUM_ALIGN
                    # Green for ≥ 0, red for < 0
                    if val >= 0:
                        c.font = Font(bold=is_total, size=10, color="375623")   # dark green
                    else:
                        c.font = Font(bold=is_total, size=10, color="9C0006")   # dark red

            elif col_name in ("Current Spend", "Optimal spend"):
                if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                    c.value = None
                else:
                    c.value         = float(raw)
                    c.number_format = '£#,##0'
                    c.alignment     = NUM_ALIGN

            elif col_name in ("Current Response", "Optimal Response"):
                if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                    c.value = None
                else:
                    c.value         = float(raw)
                    c.number_format = '#,##0'
                    c.alignment     = NUM_ALIGN

            elif col_name == "Channel":
                c.value     = str(raw) if raw is not None else ""
                c.alignment = CENTER_ALIGN

            elif col_name == "Budget period":
                c.value     = str(raw) if raw is not None else ""
                c.alignment = CENTER_ALIGN

            else:
                # Scenario column — show value only on first row of block, blank for repeats
                c.value     = str(raw) if raw is not None else ""
                c.alignment = BODY_ALIGN

        excel_row += 1

    # ── Merge Scenario cells vertically within each block ────────────────────
    # Group consecutive rows with the same scenario and merge col A
    if excel_row > 2:
        prev_scen = None
        block_start = 2
        for r in range(2, excel_row + 1):
            cur_scen = ws.cell(row=r, column=1).value if r < excel_row else "__END__"
            if cur_scen != prev_scen:
                if prev_scen is not None and (r - 1) > block_start:
                    ws.merge_cells(
                        start_row=block_start, start_column=1,
                        end_row=r - 1,         end_column=1
                    )
                    m_cell = ws.cell(row=block_start, column=1)
                    m_cell.alignment = Alignment(
                        horizontal="center", vertical="center", wrap_text=True
                    )
                block_start = r
                prev_scen   = cur_scen

    wb.save(str(xl_path))
    logger.info(f"  Written Optimisation_Scenarios tab to {xl_path.name}")


def export_all_results(
    best              : Dict,
    prep              : Dict[str, Any],
    df_ch             : pd.DataFrame,
    df_components     : Optional[pd.DataFrame],
    df_stage0         : pd.DataFrame,
    df_stage1b        : pd.DataFrame,
    df_stage2         : pd.DataFrame,
    data_cfg          : DataConfig,
    channel_prior_map : Optional[Dict] = None,
    export_excel      : bool = True,
    credible_interval : float = 0.95,
    df_ch_by_product  : Optional[pd.DataFrame] = None,
    df_timings        : Optional[pd.DataFrame] = None,
    df_response_curves: Optional[pd.DataFrame] = None,
    df_marginal_roi   : Optional[pd.DataFrame] = None,
    df_budget_opt     : Optional[pd.DataFrame] = None,
    df_spend_target   : Optional[pd.DataFrame] = None,
    df_greedy_alloc   : Optional[pd.DataFrame] = None,
    df_greedy_path    : Optional[pd.DataFrame] = None,
) -> None:
    """Exports all results to the configured output directory."""
    out_dir = Path(data_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ── Subfolder for model results ───────────────────────────
    model_dir = out_dir / "model_results"
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting results...")

    cfg = best["cfg"]
    m   = best["metrics"]

    # Derive actual adstock/saturation from per-channel specs (which override the
    # global cfg defaults). If all channels agree → show the single value; if mixed
    # → show "mixed: val1, val2, ..." so the summary reflects what was actually fitted.
    _ch_specs = best.get("channel_specs") or {}
    _ads_vals = sorted({s.adstock_type for s in _ch_specs.values()}) if _ch_specs else []
    _sat_vals = sorted({s.saturation    for s in _ch_specs.values()}) if _ch_specs else []
    _actual_adstock = _ads_vals[0] if len(_ads_vals) == 1 else (
        f"mixed: {', '.join(_ads_vals)}" if _ads_vals else cfg.adstock_type
    )
    _actual_sat = _sat_vals[0] if len(_sat_vals) == 1 else (
        f"mixed: {', '.join(_sat_vals)}" if _sat_vals else cfg.saturation
    )

    # ── 1. JSON Summary ───────────────────────────────────────
    summary = {
        "run_timestamp"   : time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_key"       : cfg.key(),
        "adstock_type"    : _actual_adstock,
        "saturation"      : _actual_sat,
        "max_lag"         : cfg.max_lag,
        "fourier_order"   : cfg.fourier_order,
        "target_accept"   : cfg.target_accept,
        "draws"           : cfg.draws,
        "chains"          : cfg.chains,
        "use_hierarchical": cfg.use_hierarchical,
        "metrics"         : {
            "mape"            : round(m["mape"],         4),
            "rmse"            : round(m["rmse"],         4),
            "r2"              : round(m["r2"],           4),
            "pearson_r"       : round(m["pearson_r"],    4),
            "max_rhat"        : round(m["max_rhat"],     6),
            "rhat_estimated"  : bool(m.get("max_rhat_is_estimated", False)),
            "mean_rhat"       : round(m["mean_rhat"],    6),
            "ess_bulk"        : round(m["ess_bulk"],     1),
            "ess_tail"        : round(m["ess_tail"],     1),
            "divergences"     : int(m["divergences"]),
            "div_rate"        : round(m["div_rate"],     6),
            "loo_ic"          : round(m["loo_ic"],       4),
            "loo_se"          : round(m["loo_se"],       4),
            "waic"            : round(m["waic"],         4),
            "pareto_k_max"    : round(m["pareto_k_max"], 4),
            "pareto_k_bad"    : int(m["pareto_k_bad"]),
            "quality_flag"    : quality_flag(m),
        },
        "channel_count"   : prep["C"],
        "spend_channels"  : prep["spend_cols"],
        "date_range"      : {
            "start": str(prep["dates"][0]),
            "end"  : str(prep["dates"][-1]),
        },
        "n_weeks"         : prep["T"],
    }

    json_path = model_dir / "best_model.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  Saved: {json_path}")

    # ── 2-4. Scan result CSVs ─────────────────────────────────
    for df_scan, fname in [
        (df_stage0,  "scan_stage0.csv"),
        (df_stage1b, "scan_stage1b.csv"),
        (df_stage2,  "scan_stage2.csv"),
    ]:
        if df_scan is not None and len(df_scan) > 0:
            path = model_dir / fname
            df_scan.to_csv(path, index=False)
            logger.info(f"  Saved: {path}")

    # ── 5. Channel contributions CSV ──────────────────────────
    ch_path = model_dir / "channel_contributions.csv"
    df_ch.to_csv(ch_path, index=False)
    logger.info(f"  Saved: {ch_path}")

    # Per-product channel contributions (P>1 only)
    if df_ch_by_product is not None and len(df_ch_by_product) > 0:
        pp_path = model_dir / "channel_contributions_by_product.csv"
        df_ch_by_product.to_csv(pp_path, index=False)
        logger.info(f"  Saved: {pp_path}")

    if df_components is not None and len(df_components) > 0:
        comp_path = model_dir / "component_contributions.csv"
        df_components.to_csv(comp_path, index=False)
        logger.info(f"  Saved: {comp_path}")

    # ── 5b. Channel transform summary CSV ─────────────────────
    if "channel_transform_summary" in best and best["channel_transform_summary"] is not None:
        ts_path = model_dir / "channel_transform_summary.csv"
        best["channel_transform_summary"].to_csv(ts_path, index=False)
        logger.info(f"  Saved: {ts_path}")

    # ── 6. Weekly media share CSV ─────────────────────────────
    try:
        trace         = m["trace"]
        train_idx     = prep["train_idx"]
        dates         = prep["dates"][train_idx]
        spend_cols    = prep["spend_cols"]
        C             = prep["C"]
        P             = prep.get("P", 1)
        product_names = prep.get("product_names", [])

        # mbc after mean("sample"):
        #   P=1: (T, C)
        #   P>1: (T, P, C)
        mbc_mean = (
            trace.posterior["media_by_channel"]
            .stack(sample=("chain", "draw"))
            .mean("sample")
            .values
        )

        if P > 1 and mbc_mean.ndim == 3:
            # Aggregate across products for the summary share file
            mbc_agg   = mbc_mean.mean(axis=1)          # (T, C)
            total_abs = np.abs(mbc_agg).sum(axis=1, keepdims=True) + 1e-12
            share_mat = np.abs(mbc_agg) / total_abs * 100.0
            share_df  = pd.DataFrame(share_mat, columns=spend_cols)
            share_df.insert(0, "date", dates)
            share_path = model_dir / "media_share_weekly.csv"
            share_df.to_csv(share_path, index=False)
            logger.info(f"  Saved: {share_path}")

            # Also export per-product weekly share files
            p_names = product_names or [f"product_{p}" for p in range(P)]
            for p_idx, pname in enumerate(p_names):
                mbc_p     = mbc_mean[:, p_idx, :]   # (T, C)
                tot_p     = np.abs(mbc_p).sum(axis=1, keepdims=True) + 1e-12
                share_p   = np.abs(mbc_p) / tot_p * 100.0
                df_p      = pd.DataFrame(share_p, columns=spend_cols)
                df_p.insert(0, "date", dates)
                safe_name = pname.replace(" ", "_").replace("/", "_")
                p_path    = model_dir / f"media_share_weekly_{safe_name}.csv"
                df_p.to_csv(p_path, index=False)
                logger.info(f"  Saved: {p_path}")
        else:
            # P=1 path (original behaviour)
            total_abs = np.abs(mbc_mean).sum(axis=1, keepdims=True) + 1e-12
            share_mat = np.abs(mbc_mean) / total_abs * 100.0
            share_df  = pd.DataFrame(share_mat, columns=spend_cols)
            share_df.insert(0, "date", dates)
            share_path = model_dir / "media_share_weekly.csv"
            share_df.to_csv(share_path, index=False)
            logger.info(f"  Saved: {share_path}")
    except Exception as e:
        logger.warning(f"  Weekly media share export failed: {e}")

    # ── 7. Actual vs Predicted + Weekly Decomposition CSVs ───
    try:
        from analysis import compute_weekly_decomposition
        from data_prep import inverse_response_transform

        trace_obj  = m["trace"]
        y_raw      = prep["y_raw"]
        y_std_v    = float(prep["y_std"])
        y_mu_v     = float(prep["y_mu"])
        dates_all  = pd.to_datetime(prep["dates"])

        mu_post = (
            trace_obj.posterior["mu"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", ...)
            .values
        )
        mu_log    = mu_post * y_std_v + y_mu_v
        y_hat_all = inverse_response_transform(mu_log, prep)
        pred_mean = y_hat_all.mean(axis=0)
        pred_lo   = np.percentile(y_hat_all, 5,  axis=0)
        pred_hi   = np.percentile(y_hat_all, 95, axis=0)

        df_avp = pd.DataFrame({
            "date"          : [d.strftime("%Y-%m-%d") for d in dates_all],
            "actual"        : y_raw.round(1),
            "predicted"     : pred_mean.round(1),
            "pred_lo_90"    : pred_lo.round(1),
            "pred_hi_90"    : pred_hi.round(1),
            "residual"      : (y_raw - pred_mean).round(1),
            "abs_pct_error" : (np.abs(y_raw - pred_mean) / (y_raw + 1e-6) * 100).round(1),
        })
        avp_path = model_dir / "actual_vs_predicted.csv"
        df_avp.to_csv(avp_path, index=False)
        logger.info(f"  Saved: {avp_path}")

        df_wd = compute_weekly_decomposition(best, prep)
        wd_path = model_dir / "weekly_decomposition.csv"
        df_wd.to_csv(wd_path, index=False)
        logger.info(f"  Saved: {wd_path}")
    except Exception as e:
        logger.warning(f"  Actual vs predicted / weekly decomposition export failed: {e}")

    # ── 8. ArviZ posterior summary CSV ───────────────────────
    try:
        az_sum = az.summary(
            m["trace"],
            var_names=["~mu", "~baseline", "~seasonality",
                       "~media_by_channel", "~control_effect"],
            round_to=6,
        )
        az_path = model_dir / "az_summary.csv"
        az_sum.to_csv(az_path)
        logger.info(f"  Saved: {az_path}")

        txt_path = model_dir / "model_summary.txt"
        rhat_label = (
            f"{m['max_rhat']:.5f} [estimated — 1 chain]"
            if m.get("max_rhat_is_estimated") else f"{m['max_rhat']:.5f}"
        )
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("BAYESIAN MMM MODEL SUMMARY\n")
            f.write("=" * 40 + "\n")
            f.write(f"Model key       : {cfg.key()}\n")
            f.write(f"MAPE            : {m['mape']:.4f}%\n")
            f.write(f"R2              : {m['r2']:.4f}\n")
            f.write(f"Pearson r       : {m['pearson_r']:.4f}\n")
            f.write(f"Max R-hat       : {rhat_label}\n")
            f.write(f"Divergences     : {m['divergences']}\n")
            f.write(f"LOO-IC          : {m['loo_ic']:.4f}\n")
            f.write(f"WAIC            : {m['waic']:.4f}\n\n")
            f.write("Top posterior rows:\n")
            f.write(az_sum.head(30).to_string())
            f.write("\n")
        logger.info(f"  Saved: {txt_path}")
    except Exception as e:
        logger.warning(f"  ArviZ summary export failed: {e}")

    logger.info(f"  All exports complete -> {model_dir}")

    # ── 8. Return index (per channel) ─────────────────────────
    try:
        df_ri = compute_return_index(
            best              = best,
            prep              = prep,
            credible_interval = credible_interval,
        )
        export_return_index(df_ri, model_dir, to_excel=export_excel)
        logger.info(
            f"  Return index summary:\n"
            + df_ri[["channel_id", "modeled_metric", "mean_return_index",
                      "mean_return_index_spend",
                      "lower_bound", "upper_bound",
                      "mean_contribution_signups",
                      "mean_weekly_metric_value", "mean_weekly_spend"]].to_string(index=False)
        )
    except Exception as e:
        logger.warning(f"  Return index export failed: {e}")

    # ── 9. Channel prior config audit trail ───────────────────
    try:
        export_channel_prior_config(channel_prior_map, prep["spend_cols"], model_dir)
    except Exception as e:
        logger.warning(f"  Prior config export failed: {e}")

    # ── 10. Stage timing ──────────────────────────────────────
    if df_timings is not None and len(df_timings) > 0:
        try:
            timing_path = model_dir / "stage_timings.csv"
            df_timings.to_csv(timing_path, index=False)
            logger.info(f"  Saved: {timing_path}")
        except Exception as e:
            logger.warning(f"  Stage timings export failed: {e}")

    # ── 10b. Consolidated model results Excel workbook ───────
    try:
        _write_model_excel(best, prep, model_dir, summary)
    except Exception as e:
        logger.warning(f"  Model Excel export failed: {e}")

    # ── 11. Response curves ───────────────────────────────────
    if df_response_curves is not None and len(df_response_curves) > 0:
        try:
            rc_path = out_dir / "response_curves.csv"
            df_response_curves.to_csv(rc_path, index=False)
            logger.info(f"  Saved: {rc_path}")
        except Exception as e:
            logger.warning(f"  Response curves export failed: {e}")

    if df_marginal_roi is not None and len(df_marginal_roi) > 0:
        try:
            mroi_path = out_dir / "marginal_roi.csv"
            df_marginal_roi.to_csv(mroi_path, index=False)
            logger.info(f"  Saved: {mroi_path}")
        except Exception as e:
            logger.warning(f"  Marginal ROI export failed: {e}")

    # ── 12. Optimisation results ──────────────────────────────
    if df_budget_opt is not None and len(df_budget_opt) > 0:
        try:
            opt_path = out_dir / "budget_optimisation.csv"
            df_budget_opt.to_csv(opt_path, index=False)
            logger.info(f"  Saved: {opt_path}")
        except Exception as e:
            logger.warning(f"  Budget optimisation export failed: {e}")

    if df_spend_target is not None and len(df_spend_target) > 0:
        try:
            rev_path = out_dir / "spend_target_optimisation.csv"
            df_spend_target.to_csv(rev_path, index=False)
            logger.info(f"  Saved: {rev_path}")
        except Exception as e:
            logger.warning(f"  Spend target optimisation export failed: {e}")

    # ── 13. Greedy budget allocation ─────────────────────────
    if df_greedy_alloc is not None and len(df_greedy_alloc) > 0:
        try:
            ga_path = out_dir / "greedy_budget_allocation.csv"
            df_greedy_alloc.to_csv(ga_path, index=False)
            logger.info(f"  Saved: {ga_path}")
        except Exception as e:
            logger.warning(f"  Greedy allocation export failed: {e}")

    if df_greedy_path is not None and len(df_greedy_path) > 0:
        try:
            gp_path = out_dir / "greedy_allocation_path.csv"
            df_greedy_path.to_csv(gp_path, index=False)
            logger.info(f"  Saved: {gp_path}")
        except Exception as e:
            logger.warning(f"  Greedy allocation path export failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison across runs
# ─────────────────────────────────────────────────────────────────────────────

def compare_runs(
    dir1        : str,
    dir2        : str,
    label1      : str = "Run 1",
    label2      : str = "Run 2",
    out_dir     : Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare two MMM pipeline runs by loading their best_model.json files.

    Reports: model config changes, metric deltas (MAPE, R², R-hat, ESS),
    channel contribution shifts, and return index changes.

    Parameters
    ----------
    dir1, dir2  : paths to mmm_outputs directories from two runs
    label1/2    : human-readable run labels
    out_dir     : if provided, saves comparison CSV + JSON there

    Returns
    -------
    DataFrame with one row per comparison dimension
    """
    p1 = Path(dir1) / "model_results"
    p2 = Path(dir2) / "model_results"

    def _load_json(path: Path) -> Dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    j1 = _load_json(p1 / "best_model.json")
    j2 = _load_json(p2 / "best_model.json")

    rows = []

    # ── Model config ──────────────────────────────────────────
    for field in ("model_key", "adstock_type", "saturation", "max_lag",
                  "fourier_order", "draws", "chains"):
        v1 = j1.get(field, "N/A")
        v2 = j2.get(field, "N/A")
        rows.append({
            "dimension"  : f"config.{field}",
            label1       : v1,
            label2       : v2,
            "changed"    : str(v1) != str(v2),
            "delta"      : None,
        })

    # ── Fit metrics ───────────────────────────────────────────
    m1 = j1.get("metrics", {})
    m2 = j2.get("metrics", {})
    for metric in ("mape", "r2", "pearson_r", "max_rhat", "ess_bulk",
                   "divergences", "loo_ic", "waic", "quality_flag"):
        v1 = m1.get(metric)
        v2 = m2.get(metric)
        try:
            delta = round(float(v2) - float(v1), 4) if (v1 is not None and v2 is not None) else None
        except (TypeError, ValueError):
            delta = None
        rows.append({
            "dimension"  : f"metric.{metric}",
            label1       : v1,
            label2       : v2,
            "changed"    : delta != 0 if delta is not None else (str(v1) != str(v2)),
            "delta"      : delta,
        })

    # ── Channels ──────────────────────────────────────────────
    ch1 = set(j1.get("spend_channels", []))
    ch2 = set(j2.get("spend_channels", []))
    rows.append({
        "dimension": "channels.added",
        label1: None,
        label2: sorted(ch2 - ch1),
        "changed": bool(ch2 - ch1),
        "delta": None,
    })
    rows.append({
        "dimension": "channels.removed",
        label1: sorted(ch1 - ch2),
        label2: None,
        "changed": bool(ch1 - ch2),
        "delta": None,
    })

    # ── Return index comparison ───────────────────────────────
    ri1_path = p1 / "return_index.csv"
    ri2_path = p2 / "return_index.csv"
    if ri1_path.exists() and ri2_path.exists():
        ri1 = pd.read_csv(ri1_path).set_index("channel_id")
        ri2 = pd.read_csv(ri2_path).set_index("channel_id")
        for ch in sorted(set(ri1.index) | set(ri2.index)):
            v1_ri = round(float(ri1.loc[ch, "mean_return_index"]), 4) if ch in ri1.index else None
            v2_ri = round(float(ri2.loc[ch, "mean_return_index"]), 4) if ch in ri2.index else None
            try:
                delta_ri = round(float(v2_ri) - float(v1_ri), 4) if (v1_ri and v2_ri) else None
            except TypeError:
                delta_ri = None
            rows.append({
                "dimension"  : f"return_index.{ch}",
                label1       : v1_ri,
                label2       : v2_ri,
                "changed"    : delta_ri != 0 if delta_ri is not None else (v1_ri != v2_ri),
                "delta"      : delta_ri,
            })

    # ── Channel contributions comparison ─────────────────────
    cc1_path = p1 / "channel_contributions.csv"
    cc2_path = p2 / "channel_contributions.csv"
    if cc1_path.exists() and cc2_path.exists():
        cc1 = pd.read_csv(cc1_path)
        cc2 = pd.read_csv(cc2_path)
        ch_col = "channel" if "channel" in cc1.columns else cc1.columns[0]
        mean_col = "mean_contribution" if "mean_contribution" in cc1.columns else None
        if mean_col:
            avg1 = cc1.groupby(ch_col)[mean_col].mean()
            avg2 = cc2.groupby(ch_col)[mean_col].mean()
            for ch in sorted(set(avg1.index) | set(avg2.index)):
                v1_c = round(float(avg1[ch]), 4) if ch in avg1.index else None
                v2_c = round(float(avg2[ch]), 4) if ch in avg2.index else None
                try:
                    delta_c = round(float(v2_c) - float(v1_c), 4) if (v1_c and v2_c) else None
                except TypeError:
                    delta_c = None
                rows.append({
                    "dimension": f"contribution.{ch}",
                    label1     : v1_c,
                    label2     : v2_c,
                    "changed"  : delta_c != 0 if delta_c is not None else (v1_c != v2_c),
                    "delta"    : delta_c,
                })

    df = pd.DataFrame(rows)

    logger.info(f"\n  Model comparison: {label1} vs {label2}")
    changed = df[df["changed"] == True]
    logger.info(f"  {len(changed)} of {len(df)} dimensions changed:")
    for _, row in changed.iterrows():
        delta_str = f"  (delta={row['delta']:+.4f})" if row["delta"] is not None else ""
        logger.info(f"    {row['dimension']}: {row[label1]} -> {row[label2]}{delta_str}")

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "model_comparison.csv", index=False)
        logger.info("  Saved: " + str(out / "model_comparison.csv"))

    return df
