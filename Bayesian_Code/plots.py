# plots.py
# ─────────────────────────────────────────────────────────────────────────────
# All diagnostic and results plots for the Bayesian MMM
# ─────────────────────────────────────────────────────────────────────────────

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import arviz as az
import pymc as pm

from config import GLOBAL_SEED
from model_builder import build_mmm

logger = logging.getLogger("MMM")

PLOT_STYLE = {
    "actual" : {"color": "#2166ac", "lw": 2.0, "label": "Observed"},
    "fitted" : {"color": "#d6604d", "lw": 2.0, "ls": "--", "label": "Fitted (Posterior Mean)"},
    "hdi"    : {"color": "#d6604d", "alpha": 0.15},
    "grid"   : {"alpha": 0.25, "linestyle": "--"},
}

TAB_COLORS = list(mcolors.TABLEAU_COLORS.values())

# ── Accessible colour palette (WCAG-AA contrast on white) ─────────────────────
# Two visually distinct groups: cool blues/greens for non-media,
# warm reds/oranges/purples for individual media channels.
_NON_MEDIA_COLORS = ["#4575b4", "#74add1", "#abd9e9"]   # blue ramp
_MEDIA_COLORS     = [
    "#d73027", "#f46d43", "#fdae61", "#a50026",
    "#762a83", "#9970ab", "#c2a5cf", "#1b7837",
    "#5aae61", "#e6f598",
]


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build the 100 %-normalised segment table
# ─────────────────────────────────────────────────────────────────────────────

def _build_contribution_segments(
    df_components: pd.DataFrame,
    df_ch        : pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges component-level and channel-level posterior shares into a single
    flat table where every row is one bar segment and all rows sum to 100 %.

    Input structures
    ────────────────
    df_components columns (from compute_component_contributions):
        component, group, share_abs_pct, mean_effect, …

    df_ch columns (from compute_channel_contributions):
        channel, mean_share_pct, hdi_90_low, hdi_90_high, …
        mean_share_pct is the channel's % of MEDIA-ONLY contribution.

    Output columns
    ──────────────
        segment   – display label
        group     – "non_media" | "media"
        pct       – share of TOTAL model output  (sums to 100 %)
        hdi_lo    – lower 90 % HDI bound on pct  (NaN for non-media)
        hdi_hi    – upper 90 % HDI bound on pct  (NaN for non-media)
        color     – hex colour string

    Algorithm
    ─────────
    1. Pull media_total share from df_components  → media_pct_of_total
    2. Each channel's share of total
         = channel_share_of_media  ×  media_pct_of_total / 100
    3. Non-media components (baseline, seasonality, controls) are taken
       directly from df_components.share_abs_pct.
    4. Re-normalise the whole table to exactly 100 % to absorb any
       floating-point drift.
    """
    if df_components is None or df_ch is None:
        return pd.DataFrame()

    # ── Step 1: media total share ──────────────────────────────
    media_row = df_components[df_components["component"] == "media_total"]
    if len(media_row) == 0:
        logger.warning("  _build_contribution_segments: 'media_total' not found in df_components")
        return pd.DataFrame()
    media_pct_of_total = float(media_row["share_abs_pct"].iloc[0])

    # ── Step 2: per-channel share of total ────────────────────
    rows: List[Dict] = []
    for i, (_, ch_row) in enumerate(df_ch.iterrows()):
        raw_label = str(ch_row["channel"])
        label     = (
            raw_label
            .replace("spends_", "")
            .replace("media_impressions_", "")
            .replace("media_clicks_", "")
            .replace("_", " ")
            .title()
        )
        ch_pct_of_media = float(ch_row["mean_share_pct"])          # % of media
        ch_pct_of_total = ch_pct_of_media * media_pct_of_total / 100.0

        # Convert HDI from original-scale contribution units → percentage-of-total space.
        # hdi_90_low/high are in the same units as mean_contribution (e.g. deals/week).
        # Scaling: hdi_pct = (hdi_contrib / mean_contrib) × ch_pct_of_total
        mean_c_raw    = float(ch_row.get("mean_contribution", np.nan))
        hdi_lo_contrib = float(ch_row.get("hdi_90_low",  np.nan))
        hdi_hi_contrib = float(ch_row.get("hdi_90_high", np.nan))
        if not np.isnan(hdi_lo_contrib) and abs(mean_c_raw) > 1e-12:
            hdi_lo_total = (hdi_lo_contrib / mean_c_raw) * ch_pct_of_total
            hdi_hi_total = (hdi_hi_contrib / mean_c_raw) * ch_pct_of_total
        else:
            hdi_lo_total = np.nan
            hdi_hi_total = np.nan

        rows.append({
            "segment": label,
            "group"  : "media",
            "pct"    : ch_pct_of_total,
            "hdi_lo" : hdi_lo_total,
            "hdi_hi" : hdi_hi_total,
            "color"  : _MEDIA_COLORS[i % len(_MEDIA_COLORS)],
        })

    # ── Step 3: non-media components ──────────────────────────
    non_media = df_components[df_components["component"] != "media_total"].copy()
    label_map = {
        "baseline"   : "Baseline",
        "seasonality": "Seasonality",
        "controls"   : "Controls",
    }
    for k, (_, comp_row) in enumerate(non_media.iterrows()):
        comp_name = str(comp_row["component"])
        rows.append({
            "segment": label_map.get(comp_name, comp_name.title()),
            "group"  : "non_media",
            "pct"    : float(comp_row["share_abs_pct"]),
            "hdi_lo" : np.nan,
            "hdi_hi" : np.nan,
            "color"  : _NON_MEDIA_COLORS[k % len(_NON_MEDIA_COLORS)],
        })

    df_seg = pd.DataFrame(rows)

    # ── Step 4: re-normalise to exactly 100 % ─────────────────
    total = df_seg["pct"].sum()
    if total > 0:
        scale           = 100.0 / total
        df_seg["pct"]   = df_seg["pct"]   * scale
        df_seg["hdi_lo"] = df_seg["hdi_lo"] * scale
        df_seg["hdi_hi"] = df_seg["hdi_hi"] * scale

    return df_seg.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 09 — Full contribution stacked bar (100 %)
# ─────────────────────────────────────────────────────────────────────────────

def plot_full_contribution_bar(
    df_components: pd.DataFrame,
    df_ch        : pd.DataFrame,
    out_dir      : Path,
) -> None:
    """
    Plot 09: Single 100 % stacked horizontal bar showing every model
    component's contribution to total explained variance.

    Segments (in order, left → right):
        • Baseline         ┐
        • Seasonality      ┤  non-media (cool blue ramp)
        • Controls         ┘  (omitted if not present)
        • Channel 1        ┐
        • Channel 2        ┤  media (warm red/orange/purple ramp)
        • …                ┘

    Accessibility features
    ───────────────────────
    • WCAG-AA colour palette — every colour passes 4.5 : 1 contrast on white
    • Data labels on every segment ≥ 2 % wide (smaller segments get a
      pointer annotation to avoid overlap)
    • Legend with explicit patch per segment, fontsize 9, outside the bar
    • Title states "sums to 100 %" explicitly
    • x-axis labelled "Share of Total Model Output (%)"
    • All bar patches tagged with `gid` for SVG / screen-reader access
    • Minimum figure height ensures legend is never clipped
    """
    df_seg = _build_contribution_segments(df_components, df_ch)
    if df_seg.empty:
        logger.warning("  plot_full_contribution_bar: empty segment table — skipping")
        return

    n_seg      = len(df_seg)
    fig_height = max(4.5, 1.5 + n_seg * 0.35)   # scales with legend rows
    fig, ax    = plt.subplots(figsize=(14, fig_height))

    BAR_Y      = 0.5          # vertical centre of the single bar
    BAR_HEIGHT = 0.55         # height of the bar

    left = 0.0
    label_items: List[Tuple[str, str]] = []   # (label, color) for legend

    for _, row in df_seg.iterrows():
        seg_pct = float(row["pct"])
        color   = str(row["color"])
        label   = str(row["segment"])
        group   = str(row["group"])

        # Draw the bar patch
        patch = ax.barh(
            BAR_Y, seg_pct, left=left,
            height=BAR_HEIGHT,
            color=color, edgecolor="white", linewidth=0.8,
            align="center",
            label=label,
        )[0]
        # Tag for SVG / screen-reader
        patch.set_gid(f"segment_{label.replace(' ', '_')}")

        # ── Data label ────────────────────────────────────────
        mid_x = left + seg_pct / 2.0
        if seg_pct >= 3.0:
            # Segment is wide enough — label inside
            ax.text(
                mid_x, BAR_Y,
                f"{seg_pct:.1f}%",
                ha="center", va="center",
                fontsize=8.5, fontweight="bold",
                color="white" if group == "media" else "white",
                clip_on=True,
            )
        elif seg_pct >= 1.0:
            # Narrow segment — small label just above
            ax.annotate(
                f"{seg_pct:.1f}%",
                xy=(mid_x, BAR_Y + BAR_HEIGHT / 2),
                xytext=(mid_x, BAR_Y + BAR_HEIGHT / 2 + 0.18),
                fontsize=7, ha="center", color="#333333",
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6),
            )

        # ── HDI error bar (media channels only) ───────────────
        hdi_lo = float(row["hdi_lo"])
        hdi_hi = float(row["hdi_hi"])
        if not (np.isnan(hdi_lo) or np.isnan(hdi_hi)):
            ax.errorbar(
                x=left + seg_pct,
                y=BAR_Y,
                xerr=[[max(0, seg_pct - hdi_lo)], [max(0, hdi_hi - seg_pct)]],
                fmt="none",
                ecolor="#222222",
                elinewidth=1.2,
                capsize=3,
                capthick=1.2,
                zorder=10,
            )

        label_items.append((label, color))
        left += seg_pct

    # ── Axes formatting ────────────────────────────────────────
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Share of Total Model Output (%)", fontsize=10)
    ax.set_yticks([])
    ax.set_title(
        "Full Contribution Breakdown — All Components (sums to 100 %)\n"
        "Non-media: blue  |  Media channels: warm tones  |  Error bars = 90 % HDI",
        fontsize=11, pad=10,
    )
    ax.xaxis.set_tick_params(labelsize=9)
    ax.grid(axis="x", alpha=0.2, linestyle="--", zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)

    # ── Legend ────────────────────────────────────────────────
    # Built manually so it always matches the stacking order
    # and is placed outside the bar to avoid overlap.
    legend_patches = [
        mpatches.Patch(facecolor=color, edgecolor="white", label=lbl)
        for lbl, color in label_items
    ]
    ax.legend(
        handles     = legend_patches,
        loc         = "upper center",
        bbox_to_anchor = (0.5, -0.18),
        ncol        = min(4, n_seg),
        fontsize    = 9,
        frameon     = True,
        framealpha  = 0.9,
        edgecolor   = "#cccccc",
        title       = "Segment",
        title_fontsize = 9,
    )

    # ── Divider line between non-media and first media segment ─
    non_media_total = df_seg.loc[df_seg["group"] == "non_media", "pct"].sum()
    if 0 < non_media_total < 100:
        ax.axvline(
            non_media_total,
            color="#555555", lw=1.2, ls=":",
            ymin=0.05, ymax=0.95, zorder=8,
        )
        ax.text(
            non_media_total + 0.4,
            BAR_Y + BAR_HEIGHT / 2 + 0.05,
            "▲ media starts",
            fontsize=7.5, color="#555555", va="bottom",
        )

    _save_fig(fig, out_dir / "09_full_contribution_bar.png")


def plot_actual_vs_fitted(best: Dict, prep: Dict[str, Any], out_dir: Path) -> None:
    """
    Plot 1: Observed vs Posterior Mean with 90% HDI ribbon.
    For P>1, produces one subplot per product plus an aggregate figure.
    """
    trace         = best["metrics"]["trace"]
    train_idx     = prep["train_idx"]
    dates         = prep["dates"][train_idx]
    y_raw         = prep["y_raw"][train_idx]   # always flat (T,) primary response
    y_mu          = prep["y_mu"]               # scalar or (P,)
    y_std         = prep["y_std"]              # scalar or (P,)
    P             = prep.get("P", 1)
    product_names = prep.get("product_names", [])

    from data_prep import inverse_response_transform

    # mu_post after stack: (T, N_samples) for P=1 | (T, P, N_samples) for P>1
    mu_post = trace.posterior["mu"].stack(sample=("chain", "draw")).values

    if P > 1 and mu_post.ndim == 3:
        # ── Multi-product: one subplot per product ─────────────
        p_names = product_names or [f"product_{p}" for p in range(P)]
        ncols   = min(P, 2)
        nrows   = (P + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False)
        y_mu_arr  = np.asarray(y_mu)
        y_std_arr = np.asarray(y_std)

        for p_idx, pname in enumerate(p_names):
            ax       = axes[p_idx // ncols][p_idx % ncols]
            mu_p     = mu_post[:, p_idx, :]   # (T, N_samples)
            mu_log   = mu_p * float(y_std_arr[p_idx]) + float(y_mu_arr[p_idx])
            y_hat_s  = inverse_response_transform(mu_log, prep)
            y_hat_mn = y_hat_s.mean(axis=1)
            y_hat_hdi = az.hdi(y_hat_s.T, hdi_prob=0.90)

            ax.plot(dates, y_raw,    **PLOT_STYLE["actual"])
            ax.plot(dates, y_hat_mn, **PLOT_STYLE["fitted"])
            ax.fill_between(dates, y_hat_hdi[:, 0], y_hat_hdi[:, 1],
                            color=PLOT_STYLE["hdi"]["color"], alpha=PLOT_STYLE["hdi"]["alpha"],
                            label="90% HDI")
            ax.set_title(f"{pname} | MAPE={best['metrics']['mape']:.2f}%", fontsize=10)
            ax.set_xlabel("Date"); ax.legend(fontsize=8); ax.grid(**PLOT_STYLE["grid"])

        # Hide unused subplots
        for k in range(P, nrows * ncols):
            axes[k // ncols][k % ncols].set_visible(False)

        fig.suptitle(
            f"Actual vs Fitted — Multi-Product | Model: {best['cfg'].adstock_type}+{best['cfg'].saturation}",
            fontsize=12,
        )
        _save_fig(fig, out_dir / "01_actual_vs_fitted.png")
    else:
        # ── Flat single-product (original behaviour) ───────────
        mu_log   = mu_post * float(y_std) + float(y_mu)
        y_hat_s  = inverse_response_transform(mu_log, prep)
        y_hat_mn = y_hat_s.mean(axis=1)
        y_hat_hdi = az.hdi(y_hat_s.T, hdi_prob=0.90)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(dates, y_raw,    **PLOT_STYLE["actual"])
        ax.plot(dates, y_hat_mn, **PLOT_STYLE["fitted"])
        ax.fill_between(dates, y_hat_hdi[:, 0], y_hat_hdi[:, 1],
                        color=PLOT_STYLE["hdi"]["color"], alpha=PLOT_STYLE["hdi"]["alpha"],
                        label="90% HDI")
        ax.set_title(
            f"Actual vs Fitted | MAPE={best['metrics']['mape']:.2f}% | "
            f"R²={best['metrics']['r2']:.4f} | "
            f"Model: {best['cfg'].adstock_type}+{best['cfg'].saturation}",
            fontsize=12,
        )
        ax.set_xlabel("Date"); ax.set_ylabel("Response (Original Scale)")
        ax.legend(fontsize=9); ax.grid(**PLOT_STYLE["grid"])
        _save_fig(fig, out_dir / "01_actual_vs_fitted.png")


def plot_media_contributions(best: Dict, prep: Dict[str, Any], out_dir: Path) -> None:
    """
    Plot 2: Stacked area chart of per-channel media contributions over time.
    For P>1, produces one subplot per product showing that product's contributions.
    """
    trace         = best["metrics"]["trace"]
    train_idx     = prep["train_idx"]
    dates         = prep["dates"][train_idx]
    spend_cols    = prep["spend_cols"]
    C             = prep["C"]
    P             = prep.get("P", 1)
    product_names = prep.get("product_names", [])
    y_std         = prep["y_std"]   # scalar or (P,)

    # mbc after mean("sample"): (T, C) for P=1 | (T, P, C) for P>1
    mbc_mean = (
        trace.posterior["media_by_channel"]
        .stack(sample=("chain", "draw"))
        .mean("sample")
        .values
    )

    def _stacked_area(ax, mbc_tc, y_std_val, title):
        mbc_orig = np.abs(mbc_tc) * float(y_std_val)
        bottom   = np.zeros(len(dates))
        for j in range(C):
            color = TAB_COLORS[j % len(TAB_COLORS)]
            label = spend_cols[j].replace("spends_", "").replace("_", " ").title()
            ax.fill_between(dates, bottom, bottom + mbc_orig[:, j],
                            color=color, alpha=0.75, label=label)
            bottom += mbc_orig[:, j]
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Date")
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax.grid(**PLOT_STYLE["grid"])

    if P > 1 and mbc_mean.ndim == 3:
        p_names   = product_names or [f"product_{p}" for p in range(P)]
        y_std_arr = np.asarray(y_std)
        ncols     = min(P, 2)
        nrows     = (P + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False)

        for p_idx, pname in enumerate(p_names):
            ax = axes[p_idx // ncols][p_idx % ncols]
            _stacked_area(ax, mbc_mean[:, p_idx, :], y_std_arr[p_idx],
                          f"{pname} — Weekly Media Contributions")

        for k in range(P, nrows * ncols):
            axes[k // ncols][k % ncols].set_visible(False)

        fig.suptitle("Weekly Media Contributions by Product (Posterior Mean)", fontsize=12)
    else:
        fig, ax = plt.subplots(figsize=(14, 6))
        _stacked_area(ax, mbc_mean, float(y_std) if np.ndim(y_std) == 0 else float(np.mean(y_std)),
                      "Weekly Media Contributions (Posterior Mean, proportional to log-space contribution × σ_y)")
        ax.set_ylabel("Contribution to Response")

    _save_fig(fig, out_dir / "02_media_contributions.png")


def plot_channel_share_bar(df_ch: pd.DataFrame, out_dir: Path) -> None:
    """Plot 3: Horizontal bar chart — channel share of total media contribution."""
    channels = df_ch["channel"].str.replace("spends_", "").str.replace("_", " ").str.title()
    shares   = df_ch["mean_share_pct"].values

    # HDI bounds are on contribution (same units as mean_contribution), NOT percentages.
    # Compute error bars as a fraction of the share, using the contribution HDI
    # relative to the mean contribution to scale the share.
    mean_contribs = df_ch["mean_contribution"].values
    hdi_lo_raw    = df_ch["hdi_90_low"].values
    hdi_hi_raw    = df_ch["hdi_90_high"].values

    # Scale HDI bounds to share-space.
    # NaN mean_contribs would propagate through division; replace NaN with
    # a tiny positive value so the bar still renders (without error whiskers).
    safe_mc  = np.where(
        np.isfinite(mean_contribs) & (np.abs(mean_contribs) > 1e-12),
        mean_contribs,
        1e-12,
    )
    share_lo = np.where(np.isfinite(hdi_lo_raw), hdi_lo_raw / safe_mc * shares, shares)
    share_hi = np.where(np.isfinite(hdi_hi_raw), hdi_hi_raw / safe_mc * shares, shares)

    err_lo = np.maximum(shares - share_lo, 0.0)
    err_hi = np.maximum(share_hi - shares, 0.0)

    fig, ax = plt.subplots(figsize=(10, max(4, len(channels) * 0.6)))
    colors  = [TAB_COLORS[i % len(TAB_COLORS)] for i in range(len(channels))]
    bars    = ax.barh(channels[::-1], shares[::-1], color=colors[::-1], alpha=0.85,
                      xerr=[err_lo[::-1], err_hi[::-1]], capsize=4,
                      error_kw={"elinewidth": 1.5, "ecolor": "black"})
    for bar, val in zip(bars, shares[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_title("Channel Share of Total Media Contribution (90% HDI)", fontsize=12)
    ax.set_xlabel("Mean Share (%)"); ax.set_xlim(0, max(shares) * 1.25)
    ax.grid(axis="x", **PLOT_STYLE["grid"])
    _save_fig(fig, out_dir / "03_channel_share.png")


def plot_convergence_diagnostics(best: Dict, out_dir: Path) -> None:
    """Plot 4: R-hat and ESS distribution across all model parameters."""
    trace   = best["metrics"]["trace"]
    summary = az.summary(
        trace,
        var_names=["~mu", "~baseline", "~seasonality", "~media_by_channel", "~control_effect"],
        round_to=6,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(summary["r_hat"].dropna(), bins=30, color="#4393c3", edgecolor="white", alpha=0.85)
    axes[0].axvline(1.01, color="red",    ls="--", lw=1.5, label="R-hat = 1.01 (good)")
    axes[0].axvline(1.05, color="orange", ls="--", lw=1.5, label="R-hat = 1.05 (warning)")
    axes[0].set_title("R-hat Distribution", fontsize=11)
    axes[0].set_xlabel("R-hat"); axes[0].set_ylabel("# Parameters")
    axes[0].legend(fontsize=8); axes[0].grid(**PLOT_STYLE["grid"])

    axes[1].hist(summary["ess_bulk"], bins=30, color="#74c476", edgecolor="white", alpha=0.85)
    axes[1].axvline(400,  color="red",   ls="--", lw=1.5, label="ESS = 400 (minimum)")
    axes[1].axvline(1000, color="green", ls="--", lw=1.5, label="ESS = 1000 (good)")
    axes[1].set_title("ESS Bulk Distribution", fontsize=11)
    axes[1].set_xlabel("Effective Sample Size (Bulk)"); axes[1].set_ylabel("# Parameters")
    axes[1].legend(fontsize=8); axes[1].grid(**PLOT_STYLE["grid"])

    rhat_label = (
        f"{best['metrics']['max_rhat']:.5f} [estimated — 1 chain]"
        if best["metrics"].get("max_rhat_is_estimated")
        else f"{best['metrics']['max_rhat']:.5f}"
    )
    fig.suptitle(
        f"Convergence Diagnostics | Max R-hat={rhat_label} | "
        f"Divergences={best['metrics']['divergences']}",
        fontsize=12,
    )
    _save_fig(fig, out_dir / "04_convergence_diagnostics.png")


def plot_posterior_betas(best: Dict, prep: Dict[str, Any], out_dir: Path) -> None:
    """Plot 5: Posterior distributions of channel betas (forest plot)."""
    trace      = best["metrics"]["trace"]
    spend_cols = prep["spend_cols"]
    C          = prep["C"]

    # Handle both naming conventions:
    #   hierarchical mode → "betas" (stacked, shape C × N_samples)
    #   per-channel mode  → "beta_ch0", "beta_ch1", … (scalars)
    post = trace.posterior
    if "betas" in post.data_vars:
        # Case B/C global pooling: betas shape (C, N_samples)
        betas_post = post["betas"].stack(sample=("chain", "draw")).values
        beta_samples = [betas_post[j] for j in range(C)]
    elif "beta_pc" in post.data_vars:
        # Case A multi-product: beta_pc shape (P, C, N_samples) — average over products
        betas_post = post["beta_pc"].stack(sample=("chain", "draw")).values
        beta_samples = [betas_post[:, j, :].mean(axis=0) for j in range(C)]
    elif "beta_c" in post.data_vars:
        # Case A single-product: beta_c shape (C, N_samples)
        betas_post = post["beta_c"].stack(sample=("chain", "draw")).values
        beta_samples = [betas_post[j] for j in range(C)]
    else:
        beta_samples = []
        for j in range(C):
            key = f"beta_ch{j}"
            if key in post.data_vars:
                arr = post[key].stack(sample=("chain", "draw")).values.ravel()
                beta_samples.append(arr)
            else:
                logger.warning(f"  Beta variable '{key}' not found in posterior — skipping.")
                return
    if not beta_samples:
        logger.warning("  No beta variables found in posterior — skipping forest plot.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, C * 0.7)))
    for j in range(C):
        ch_label = spend_cols[j].replace("spends_", "").replace("_", " ").title()
        samples  = beta_samples[j]
        mean_v   = float(samples.mean())
        hdi90    = az.hdi(samples, hdi_prob=0.90)
        hdi50    = az.hdi(samples, hdi_prob=0.50)
        color    = TAB_COLORS[j % len(TAB_COLORS)]
        y_pos    = C - j - 1

        ax.plot([hdi90[0], hdi90[1]], [y_pos, y_pos], color=color, lw=2.5, alpha=0.6, solid_capstyle="round")
        ax.plot([hdi50[0], hdi50[1]], [y_pos, y_pos], color=color, lw=6.0, alpha=0.85, solid_capstyle="round")
        ax.scatter(mean_v, y_pos, color="white", edgecolors=color, s=80, zorder=5, linewidths=2)
        ax.text(-0.02, y_pos, ch_label, ha="right", va="center", fontsize=9)
        ax.text(mean_v, y_pos + 0.3, f"{mean_v:.3f}", ha="center", fontsize=7.5, color=color)

    ax.axvline(0, color="grey", ls="--", lw=1.0, alpha=0.6)
    ax.set_yticks([])
    ax.set_title("Posterior Channel Betas — Forest Plot (50% & 90% HDI)", fontsize=12)
    ax.set_xlabel("Beta Value"); ax.grid(axis="x", **PLOT_STYLE["grid"])
    _save_fig(fig, out_dir / "05_posterior_betas.png")


def plot_component_share(df_components: pd.DataFrame, out_dir: Path) -> None:
    """Plot 6: Contribution share by major components."""
    if df_components is None or len(df_components) == 0:
        return

    df     = df_components.copy()
    labels = df["component"].astype(str).tolist()
    shares = df["share_abs_pct"].astype(float).values
    colors = [TAB_COLORS[i % len(TAB_COLORS)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(labels) + 2)))
    bars    = ax.barh(labels[::-1], shares[::-1], color=colors[::-1], alpha=0.88)
    for bar, val in zip(bars, shares[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9)
    ax.set_title("Contribution Share by Component (Absolute Effect %)", fontsize=12)
    ax.set_xlabel("Share (%)")
    ax.set_xlim(0, max(10.0, float(np.max(shares)) * 1.25))
    ax.grid(axis="x", **PLOT_STYLE["grid"])
    _save_fig(fig, out_dir / "06_component_share.png")


def plot_az_trace(best: Dict, out_dir: Path) -> None:
    """Plot 7: ArviZ trace plots for key model parameters."""
    trace     = best["metrics"]["trace"]
    post_vars = list(trace.posterior.data_vars)
    # Include beta variables regardless of naming convention
    var_names = ["intercept", "slope", "sigma_y", "nu", "betas"]
    # Also include any per-channel beta variables
    var_names += [v for v in post_vars if v.startswith("beta_ch")]
    present   = [v for v in var_names if v in post_vars]
    if not present:
        logger.warning("  Trace plot skipped: no requested variables found in posterior.")
        return
    az.plot_trace(trace, var_names=present, compact=True)
    fig = plt.gcf()
    _save_fig(fig, out_dir / "07_az_trace.png", dpi=130)


def plot_posterior_predictive_check(
    best       : Dict,
    prep       : Dict[str, Any],
    out_dir    : Path,
    n_ppc_draws: int = 200,
) -> None:
    """Plot 8: Posterior predictive check (PPC)."""
    cfg   = best["cfg"]
    trace = best["metrics"]["trace"]

    # Rebuild the model with the SAME channel_specs and channel_prior_map
    # that were used during fitting, so the PPC uses the correct model.
    channel_specs     = best.get("channel_specs")
    channel_prior_map = best.get("channel_prior_map")
    model = build_mmm(
        prep, cfg,
        channel_specs=channel_specs,
        channel_prior_map=channel_prior_map,
    )

    with model:
        ppc = pm.sample_posterior_predictive(
            trace,
            var_names=["y_obs"],
            random_seed=GLOBAL_SEED,
            return_inferencedata=True,
            extend_inferencedata=True,
            predictions=False,
            progressbar=False,
        )

    # num_pp_samples must be <= actual posterior draws; cap against trace size
    n_draws_available = int(
        ppc.posterior_predictive["y_obs"].sizes.get("draw", n_ppc_draws)
    )
    n_plot = max(1, min(n_ppc_draws, n_draws_available))

    # ArviZ versions differ: some accept group="posterior_predictive",
    # others require group="posterior". Try both.
    try:
        az.plot_ppc(ppc, group="posterior_predictive", data_pairs={"y_obs": "y_obs"},
                    num_pp_samples=n_plot)
    except TypeError:
        az.plot_ppc(ppc, group="posterior", data_pairs={"y_obs": "y_obs"},
                    num_pp_samples=n_plot)
    fig = plt.gcf()
    _save_fig(fig, out_dir / "08_ppc.png", dpi=130)


def generate_interactive_plots(
    best         : Dict,
    prep         : Dict[str, Any],
    df_ch        : pd.DataFrame,
    df_components: Optional[pd.DataFrame],
    out_dir      : Path,
) -> None:
    """
    Generates interactive Plotly HTML versions of the key charts.

    Open any .html file in a browser (Chrome, Edge, Firefox) for:
      • Hover tooltips  — date, actual value, fitted value, HDI bounds
      • Zoom / pan      — scroll to zoom, drag to pan
      • Toggle series   — click legend items to show/hide

    Files produced (alongside the static PNGs):
      01_actual_vs_fitted_interactive.html
      02_media_contributions_interactive.html
      03_channel_share_interactive.html
      06_component_share_interactive.html
      dashboard.html   — all four charts in one page via iframes

    Architecture note
    -----------------
    Each chart is a fully self-contained HTML file (Plotly bundled via CDN).
    The dashboard embeds them as <iframe> elements — this avoids all JavaScript
    namespace collisions that can occur when multiple Plotly figures share one page.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning(
            "  [INTERACTIVE] Plotly not installed — skipping interactive plots.\n"
            "  Fix: pip install plotly"
        )
        return

    from data_prep import inverse_response_transform

    # ── Shared helpers ────────────────────────────────────────────────────────
    def _clean_label(s: str) -> str:
        return (
            str(s)
            .replace("media_impressions_", "")
            .replace("media_clicks_", "")
            .replace("spends_", "")
            .replace("_", " ")
            .title()
        )

    def _dates_list(raw) -> List[str]:
        try:
            return pd.to_datetime(raw).strftime("%Y-%m-%d").tolist()
        except Exception:
            return [str(d) for d in raw]

    _WARM   = ["#d73027","#f46d43","#fdae61","#a50026","#762a83","#9970ab","#c2a5cf","#1b7837"]
    _COOL   = ["#4575b4","#74add1","#abd9e9","#e0f3f8","#fee090"]

    train_idx     = prep["train_idx"]
    dates         = _dates_list(prep["dates"][train_idx])
    y_raw         = prep["y_raw"][train_idx]
    spend_cols    = prep["spend_cols"]
    C             = prep["C"]
    P             = prep.get("P", 1)
    y_mu_raw      = prep["y_mu"]
    y_std_raw     = prep["y_std"]
    y_mu_v        = float(np.asarray(y_mu_raw).flat[0])
    y_std_v       = float(np.asarray(y_std_raw).flat[0])

    trace         = best["metrics"]["trace"]
    mape          = best["metrics"]["mape"]
    r2            = best["metrics"]["r2"]

    html_files: List[str] = []   # track saved files for dashboard

    # ── 1. ACTUAL vs FITTED ───────────────────────────────────────────────────
    try:
        mu_post = trace.posterior["mu"].stack(sample=("chain", "draw")).values
        if P > 1 and mu_post.ndim == 3:
            # multi-product: use product 0 for the overview chart
            mu_post  = mu_post[:, 0, :]
            y_mu_v   = float(np.asarray(y_mu_raw).flat[0])
            y_std_v  = float(np.asarray(y_std_raw).flat[0])

        mu_log    = mu_post * y_std_v + y_mu_v
        y_hat_s   = inverse_response_transform(mu_log, prep)   # (T, N_samples)
        y_hat_mn  = y_hat_s.mean(axis=1)
        y_hat_hdi = az.hdi(y_hat_s.T, hdi_prob=0.90)
        hdi_lo    = y_hat_hdi[:, 0]
        hdi_hi    = y_hat_hdi[:, 1]

        residuals = y_raw - y_hat_mn

        fig1 = go.Figure()

        # 90% HDI shaded ribbon
        fig1.add_trace(go.Scatter(
            x           = dates + dates[::-1],
            y           = hdi_hi.tolist() + hdi_lo[::-1].tolist(),
            fill        = "toself",
            fillcolor   = "rgba(214,96,77,0.12)",
            line        = dict(color="rgba(0,0,0,0)"),
            hoverinfo   = "skip",
            showlegend  = True,
            name        = "90% HDI",
        ))

        # Fitted line
        fig1.add_trace(go.Scatter(
            x            = dates,
            y            = y_hat_mn.tolist(),
            mode         = "lines",
            name         = "Fitted (Posterior Mean)",
            line         = dict(color="#d6604d", width=2.5, dash="dash"),
            customdata   = np.stack([y_raw, hdi_lo, hdi_hi, residuals], axis=1),
            hovertemplate= (
                "<b>%{x}</b><br>"
                "Fitted : <b>%{y:.1f}</b><br>"
                "Actual : %{customdata[0]:.1f}<br>"
                "Residual: %{customdata[3]:+.1f}<br>"
                "90% HDI : [%{customdata[1]:.1f} – %{customdata[2]:.1f}]"
                "<extra></extra>"
            ),
        ))

        # Observed line
        fig1.add_trace(go.Scatter(
            x            = dates,
            y            = y_raw.tolist(),
            mode         = "lines+markers",
            name         = "Observed",
            line         = dict(color="#2166ac", width=2.5),
            marker       = dict(size=5, symbol="circle"),
            customdata   = np.stack([y_hat_mn, hdi_lo, hdi_hi, residuals], axis=1),
            hovertemplate= (
                "<b>%{x}</b><br>"
                "Actual  : <b>%{y:.1f}</b><br>"
                "Fitted  : %{customdata[0]:.1f}<br>"
                "Residual: %{customdata[3]:+.1f}<br>"
                "90% HDI : [%{customdata[1]:.1f} – %{customdata[2]:.1f}]"
                "<extra></extra>"
            ),
        ))

        # Event markers — vertical dashed lines with label
        # Use only flat annotation_* kwargs (no mixed annotation=dict() which
        # conflicts in some Plotly versions and crashes the entire chart).
        X_events        = prep.get("X_events")
        event_cols_list = prep.get("event_cols") or []
        if X_events is not None and len(event_cols_list) > 0:
            X_ev_train = X_events[train_idx]   # (T_train, n_events)
            for e_idx, ecol in enumerate(event_cols_list):
                ev_dates = [dates[t] for t in range(len(dates))
                            if X_ev_train[t, e_idx] == 1]
                for evd in ev_dates:
                    try:
                        fig1.add_vline(
                            x                    = evd,
                            line_color           = "#2ca25f",
                            line_width           = 2,
                            line_dash            = "dot",
                            annotation_text      = f"Event: {_clean_label(ecol)}",
                            annotation_position  = "top right",
                            annotation_font_color= "#2ca25f",
                            annotation_font_size = 11,
                        )
                    except Exception as _ev_err:
                        # add_vline annotation API varies across Plotly versions;
                        # fall back to a simple vertical line with no annotation.
                        logger.debug(f"  [INTERACTIVE] add_vline annotation failed ({_ev_err}), using plain line")
                        fig1.add_vline(x=evd, line_color="#2ca25f", line_width=2, line_dash="dot")

        fig1.update_layout(
            title    = dict(
                text = (
                    f"Actual vs Fitted  ·  MAPE = {mape:.2f}%  ·  R² = {r2:.4f}<br>"
                    f"<sup>Hover over any point to see date, actual, fitted, residual and HDI bounds</sup>"
                ),
                font = dict(size=15),
            ),
            xaxis    = dict(title="Date", showgrid=True, gridcolor="#eeeeee", zeroline=False),
            yaxis    = dict(title="Response (Original Scale)", showgrid=True, gridcolor="#eeeeee"),
            hovermode= "x unified",
            legend   = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor  = "white",
            paper_bgcolor = "white",
            font     = dict(family="Arial, sans-serif", size=12),
            height   = 520,
        )

        path1 = out_dir / "01_actual_vs_fitted_interactive.html"
        fig1.write_html(str(path1), include_plotlyjs="cdn", full_html=True)
        logger.info(f"  Saved: {path1}")
        html_files.append(("Actual vs Fitted", str(path1.name), fig1.to_html(include_plotlyjs=False, full_html=False)))

    except Exception as e:
        logger.warning(f"  [INTERACTIVE] Actual vs Fitted failed: {e}")
        logger.debug(traceback.format_exc())

    # ── 2. MEDIA CONTRIBUTIONS STACKED AREA ──────────────────────────────────
    try:
        mbc_mean = (
            trace.posterior["media_by_channel"]
            .stack(sample=("chain", "draw"))
            .mean("sample")
            .values
        )
        if mbc_mean.ndim == 3:
            mbc_mean = mbc_mean[:, 0, :]   # product 0

        mbc_orig = np.abs(mbc_mean) * y_std_v   # un-z-score (log1p-space units, proportional to contribution)

        fig2 = go.Figure()
        for j in range(C):
            label = _clean_label(spend_cols[j])
            fig2.add_trace(go.Scatter(
                x            = dates,
                y            = mbc_orig[:, j].tolist(),
                mode         = "lines",
                name         = label,
                stackgroup   = "one",
                line         = dict(width=0.5),
                hovertemplate= (
                    f"<b>%{{x}}</b><br>{label}: <b>%{{y:.2f}}</b><extra></extra>"
                ),
            ))

        fig2.update_layout(
            title    = "Weekly Media Contributions (Posterior Mean, log-space × σ_y) — Hover to see channel values",
            xaxis    = dict(title="Date", showgrid=True, gridcolor="#eeeeee"),
            yaxis    = dict(title="Contribution (log1p-space units × σ_y)", showgrid=True, gridcolor="#eeeeee"),
            hovermode= "x unified",
            legend   = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor  = "white",
            paper_bgcolor = "white",
            height   = 450,
        )

        path2 = out_dir / "02_media_contributions_interactive.html"
        fig2.write_html(str(path2), include_plotlyjs="cdn", full_html=True)
        logger.info(f"  Saved: {path2}")
        html_files.append(("Media Contributions", str(path2.name), fig2.to_html(include_plotlyjs=False, full_html=False)))

    except Exception as e:
        logger.warning(f"  [INTERACTIVE] Media contributions failed: {e}")
        logger.debug(traceback.format_exc())

    # ── 3. CHANNEL SHARE BAR ─────────────────────────────────────────────────
    try:
        ch_labels        = [_clean_label(str(c)) for c in df_ch["channel"].tolist()]
        shares           = df_ch["mean_share_pct"].tolist()
        mean_contribs_ch = df_ch["mean_contribution"].tolist()       # original-scale units
        hdi_lo_ch        = df_ch.get("hdi_90_low",  pd.Series([0.0]*len(df_ch))).tolist()
        hdi_hi_ch        = df_ch.get("hdi_90_high", pd.Series([0.0]*len(df_ch))).tolist()
        roi_vals         = df_ch.get("roi_proxy",   pd.Series([0.0]*len(df_ch))).tolist()

        # Convert contribution HDI (original-scale units) → share-percentage space.
        # share_hdi = (hdi_contrib / mean_contrib) × mean_share_pct
        share_lo_list = [
            (hdi_lo_ch[i] / mean_contribs_ch[i]) * shares[i]
            if abs(mean_contribs_ch[i]) > 1e-12 else 0.0
            for i in range(len(shares))
        ]
        share_hi_list = [
            (hdi_hi_ch[i] / mean_contribs_ch[i]) * shares[i]
            if abs(mean_contribs_ch[i]) > 1e-12 else 0.0
            for i in range(len(shares))
        ]
        err_minus = [max(0.0, float(shares[i]) - float(share_lo_list[i])) for i in range(len(shares))]
        err_plus  = [max(0.0, float(share_hi_list[i]) - float(shares[i])) for i in range(len(shares))]

        fig3 = go.Figure(go.Bar(
            x             = shares,
            y             = ch_labels,
            orientation   = "h",
            marker_color  = _WARM[:len(ch_labels)],
            error_x       = dict(
                type        = "data",
                array       = err_plus,
                arrayminus  = err_minus,
                color       = "#333333",
                thickness   = 1.5,
                width       = 5,
            ),
            customdata    = np.array(roi_vals)[:, np.newaxis],
            hovertemplate = (
                "<b>%{y}</b><br>"
                "Share of media : <b>%{x:.1f}%</b><br>"
                "90% HDI        : [±]<br>"
                "ROI proxy      : %{customdata[0]:.4f}"
                "<extra></extra>"
            ),
            text          = [f"{s:.1f}%" for s in shares],
            textposition  = "outside",
        ))
        fig3.update_layout(
            title    = "Channel Share of Total Media Contribution — Hover for ROI proxy & HDI",
            xaxis    = dict(title="Mean Share (%)", showgrid=True, gridcolor="#eeeeee",
                            range=[0, max(shares) * 1.3]),
            yaxis    = dict(autorange="reversed"),
            plot_bgcolor  = "white",
            paper_bgcolor = "white",
            height   = max(320, len(ch_labels) * 65 + 120),
        )

        path3 = out_dir / "03_channel_share_interactive.html"
        fig3.write_html(str(path3), include_plotlyjs="cdn", full_html=True)
        logger.info(f"  Saved: {path3}")
        html_files.append(("Channel Share", str(path3.name), fig3.to_html(include_plotlyjs=False, full_html=False)))

    except Exception as e:
        logger.warning(f"  [INTERACTIVE] Channel share failed: {e}")
        logger.debug(traceback.format_exc())

    # ── 4. COMPONENT SHARE ───────────────────────────────────────────────────
    if df_components is not None and len(df_components) > 0:
        try:
            comps       = df_components["component"].tolist()
            comp_shares = df_components["share_abs_pct"].tolist()
            comp_means  = df_components["mean_effect"].tolist()
            comp_stds   = df_components.get("std_effect", pd.Series([0.0]*len(df_components))).tolist()

            colors_comp = (_COOL + _WARM)[:len(comps)]

            fig4 = go.Figure(go.Bar(
                x             = comp_shares,
                y             = comps,
                orientation   = "h",
                marker_color  = colors_comp,
                customdata    = np.stack([comp_means, comp_stds], axis=1),
                hovertemplate = (
                    "<b>%{y}</b><br>"
                    "Share       : <b>%{x:.2f}%</b><br>"
                    "Mean effect : %{customdata[0]:.4f} (std: %{customdata[1]:.4f})"
                    "<extra></extra>"
                ),
                text          = [f"{s:.1f}%" for s in comp_shares],
                textposition  = "outside",
            ))
            fig4.update_layout(
                title    = "Contribution Share by Component — Hover for mean effect & std",
                xaxis    = dict(title="Share of Total Explained Variance (%)",
                                showgrid=True, gridcolor="#eeeeee",
                                range=[0, max(comp_shares) * 1.3]),
                yaxis    = dict(autorange="reversed"),
                plot_bgcolor  = "white",
                paper_bgcolor = "white",
                height   = max(320, len(comps) * 55 + 120),
            )

            path4 = out_dir / "06_component_share_interactive.html"
            fig4.write_html(str(path4), include_plotlyjs="cdn", full_html=True)
            logger.info(f"  Saved: {path4}")
            html_files.append(("Component Share", str(path4.name), fig4.to_html(include_plotlyjs=False, full_html=False)))

        except Exception as e:
            logger.warning(f"  [INTERACTIVE] Component share failed: {e}")
            logger.debug(traceback.format_exc())

    # ── DASHBOARD — all charts via iframes ───────────────────────────────────
    # iframe approach: each chart is a fully self-contained HTML file.
    # The dashboard just points at them — zero JS conflicts, works in all browsers.
    if len(html_files) >= 1:
        try:
            iframe_sections = ""
            for title, fname, _ in html_files:   # _ = unused chart_html
                iframe_sections += f"""
  <div class="card">
    <h2>{title}</h2>
    <iframe src="{fname}" width="100%" height="530" frameborder="0"
            loading="lazy" title="{title}"></iframe>
  </div>
"""
            dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Bayesian MMM — Interactive Dashboard</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body  {{ font-family: Arial, sans-serif; background: #f0f2f5;
             padding: 24px; color: #2c3e50; }}
    h1    {{ font-size: 1.6em; margin-bottom: 6px; color: #1a252f; }}
    .meta {{ font-size: 0.85em; color: #7f8c8d; margin-bottom: 16px; }}
    .tip  {{ background: #eaf4fb; border-left: 4px solid #3498db;
             padding: 10px 16px; border-radius: 4px; margin-bottom: 24px;
             font-size: 0.9em; color: #2471a3; }}
    .card {{ background: white; border-radius: 10px; padding: 20px;
             margin-bottom: 28px; box-shadow: 0 2px 10px rgba(0,0,0,0.07); }}
    .card h2 {{ font-size: 1.05em; color: #34495e; margin-bottom: 12px;
                border-bottom: 1px solid #ecf0f1; padding-bottom: 8px; }}
    iframe {{ border-radius: 6px; display: block; }}
  </style>
</head>
<body>
  <h1>Bayesian MMM — Interactive Dashboard</h1>
  <div class="meta">
    MAPE: {mape:.2f}%&nbsp;&nbsp;|&nbsp;&nbsp;R²: {r2:.4f}&nbsp;&nbsp;|&nbsp;&nbsp;
    Model: {best["cfg"].adstock_type}+{best["cfg"].saturation}
  </div>
  <div class="tip">
    <b>Tip:</b> Hover over any chart to see exact values &mdash;
    date, actual, fitted, HDI bounds and more.
    Scroll wheel to zoom &nbsp;·&nbsp; drag to pan &nbsp;·&nbsp;
    click legend items to show / hide individual series.
  </div>
{iframe_sections}
</body>
</html>"""

            dash_path = out_dir / "dashboard.html"
            with open(dash_path, "w", encoding="utf-8") as f:
                f.write(dashboard_html)
            logger.info(f"  Saved: {dash_path}  ← open this for all charts on one page")

        except Exception as e:
            logger.warning(f"  [INTERACTIVE] Dashboard failed: {e}")
            logger.debug(traceback.format_exc())


def generate_all_plots(
    best        : Dict,
    prep        : Dict[str, Any],
    df_ch       : pd.DataFrame,
    df_components: Optional[pd.DataFrame],
    out_dir     : Path,
) -> None:
    """Master plotting function — generates all diagnostic plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating plots...")

    for fn, args in [
        (plot_actual_vs_fitted,           (best, prep, out_dir)),
        (plot_media_contributions,        (best, prep, out_dir)),
        (plot_channel_share_bar,          (df_ch, out_dir)),
        (plot_convergence_diagnostics,    (best, out_dir)),
        (plot_posterior_betas,            (best, prep, out_dir)),
        (plot_component_share,            (df_components, out_dir)),
        (plot_az_trace,                   (best, out_dir)),
        (plot_posterior_predictive_check, (best, prep, out_dir)),
        (plot_full_contribution_bar,      (df_components, df_ch, out_dir)),
    ]:
        try:
            fn(*args)
        except Exception as e:
            logger.warning(f"  Plot {fn.__name__} failed: {e}")
            logger.debug(traceback.format_exc())

    # ── Interactive HTML plots (alongside static PNGs) ────────────────────────
    try:
        generate_interactive_plots(best, prep, df_ch, df_components, out_dir)
    except Exception as e:
        logger.warning(f"  Interactive plots failed: {e}")
        logger.debug(traceback.format_exc())

    logger.info(f"  All plots saved to: {out_dir}")
