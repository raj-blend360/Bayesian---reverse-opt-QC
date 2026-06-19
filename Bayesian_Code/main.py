#!/usr/bin/env python
# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Entry point — terminal and Jupyter notebook support
#
# Per-channel variable selection (NEW)
#   After reading the CSV header, the user is shown every channel and the
#   metric variants available for it (Spend / Impressions / Clicks / …).
#   They pick one metric per channel before the pipeline starts.
#
#   Terminal example
#   ────────────────
#   ══════════════════════════════════════════════════════════════
#     CHANNEL VARIABLE SELECTION  (3 channels found)
#     For each channel choose which metric to model.
#   ══════════════════════════════════════════════════════════════
#
#     ┌─ Channel 1 / 3: google ──────────────────────────────────┐
#     │  1. Spend        →  spends_google
#     │  2. Impressions  →  imps_google
#     │  3. Clicks       →  clicks_google
#     └──────────────────────────────────────────────────────────┘
#     Choose [default 1]: 3
#     ✓  google  →  clicks_google
#
#   ─────────────────────────────────────────────────────────────
#   Selection summary:
#     facebook                       →  spends_facebook
#     google                         →  clicks_google
#     tv                             →  spends_tv
#   ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

import sys
import logging
import warnings
from typing import Any, Dict, List, Optional

# ── Suppress PyTensor / PyMC import-time compiler warnings ────────────────────
# These fire at module-import time (before setup_logger() runs) when g++ is not
# on PATH.  PyTensor still runs correctly via the Python backend — it is just
# slower.  We silence the two noisy lines and emit one clean advisory later.
#
# Permanent fix (one-time, run in your environment):
#   conda:   conda install -c conda-forge m2w64-toolchain   (Windows)
#   pip:     install "Microsoft C++ Build Tools" from visualstudio.microsoft.com
#   Linux:   sudo apt-get install g++ build-essential
#   macOS:   xcode-select --install
logging.getLogger("pytensor.configdefaults").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress PyTensor loop-fusion warnings specifically (fires during model compilation;
# also emitted from joblib subprocesses so we suppress by message pattern too).
warnings.filterwarnings("ignore", message="Loop fusion failed.*kernel argument limit",
                        category=UserWarning)
# ─────────────────────────────────────────────────────────────────────────────

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

from pipeline import run_full_pipeline
from metrics import quality_flag
from data_prep import discover_channels
from priors import ChannelPriorConfig

logger = setup_logger()

# After logger is ready, emit one single advisory if g++ is absent.
try:
    import pytensor
    _cxx = getattr(pytensor.config, "cxx", None)
    if _cxx == "":
        logger.warning(
            "[PYTENSOR] g++ compiler not found — running on the Python backend "
            "(slower, but fully functional).  "
            "To enable the faster C backend: "
            "conda install -c conda-forge m2w64-toolchain  (Windows/conda)  |  "
            "sudo apt-get install g++ build-essential  (Linux/WSL)"
        )
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Runtime helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in getattr(ip, "config", {})
    except Exception:
        return False


def _can_prompt() -> bool:
    return sys.stdin.isatty() or _is_notebook()


def _prompt_choice(prompt: str, options: List[str], default_idx: int = 0) -> str:
    """Numbered single-choice menu. Returns the chosen string."""
    if not _can_prompt():
        return options[default_idx]
    print(f"\n{prompt}")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    raw = input(f"Choose [default {default_idx + 1}]: ").strip()
    if not raw:
        return options[default_idx]
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except Exception:
        pass
    return options[default_idx]


def _detect_date_granularity(csv_path: str, date_col: str) -> str:
    """
    Infers whether the dataset is weekly or daily by examining the
    median gap between consecutive dates.  Returns "weeks" or "days".
    """
    try:
        df   = pd.read_csv(csv_path, usecols=[date_col], nrows=30)
        dates = pd.to_datetime(df[date_col], dayfirst=True).sort_values()
        gaps  = dates.diff().dropna().dt.days
        med   = float(gaps.median())
        return "weeks" if med >= 5 else "days"
    except Exception:
        return "weeks"   # safe fallback


def prompt_lag_settings(
    csv_path : str,
    date_col : str,
) -> Dict[str, Any]:
    """
    Asks the user whether to apply a discrete lag transformation before adstock.

    Returns a dict with keys:
        use_lag        : bool
        global_max_lag : int  (0 if use_lag is False)
        time_unit      : str  "weeks" | "days"
    """
    time_unit = _detect_date_granularity(csv_path, date_col)

    print("\n" + "─" * 62)
    print("  LAG TRANSFORMATION")
    print("─" * 62)
    print(
        "  A lag shift delays the media signal before adstock is applied,\n"
        "  modelling channels where spend takes several periods to take\n"
        "  effect (e.g. TV brand campaigns, OOH).  Each channel will\n"
        "  independently learn its own best lag from the data.\n"
        f"  Dataset granularity detected: {time_unit}."
    )

    choice = _prompt_choice(
        "  Apply lag transformation to media channels?",
        ["Yes", "No"],
        default_idx=1,
    )

    if choice == "No":
        print("  ✓  Lag disabled — model will use adstock + saturation only.")
        return {"use_lag": False, "global_max_lag": 0, "time_unit": time_unit}

    # Ask for max lag
    default_max = 4 if time_unit == "weeks" else 14
    print(f"\n  Enter the maximum lag in {time_unit} [default {default_max}]: ", end="")

    if not _can_prompt():
        max_lag = default_max
    else:
        raw = input().strip()
        try:
            max_lag = int(raw) if raw else default_max
            if max_lag < 1:
                print("  ⚠  Value must be ≥ 1 — using default.")
                max_lag = default_max
        except Exception:
            print(f"  ⚠  Invalid input — using default ({default_max}).")
            max_lag = default_max

    print(f"  ✓  Lag enabled | max lag = {max_lag} {time_unit} per channel")
    return {"use_lag": True, "global_max_lag": max_lag, "time_unit": time_unit}


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel variable selection  ← NEW
# ─────────────────────────────────────────────────────────────────────────────

def prompt_channel_variable_selection(
    csv_path    : str,
    date_col    : str,
    response_col: str,
) -> Dict[str, str]:
    """
    Reads the CSV header, discovers all channels and their metric variants,
    then asks the user which variable to model for each channel.

    Returns
    -------
    channel_variable_map : {channel_name → chosen_column_name}
    """
    try:
        df_head = pd.read_csv(csv_path, nrows=5)
    except Exception as e:
        raise RuntimeError(f"Cannot read CSV to discover channels: {e}")

    channel_opts = discover_channels(df_head, exclude_cols=[date_col, response_col])

    if not channel_opts:
        raise RuntimeError(
            "No recognised channel columns found in the CSV.\n"
            "Expected columns like:  spends_google, imps_facebook, clicks_tv, …\n"
            "Supported prefixes: spends_, imps_, impressions_, clicks_, clks_, views_, reach_"
        )

    channels = sorted(channel_opts.keys())
    n        = len(channels)

    print("\n" + "═" * 62)
    print(f"  CHANNEL VARIABLE SELECTION  ({n} channel{'s' if n != 1 else ''} found)")
    print("  For each channel, choose which metric to model.")
    print("═" * 62)

    channel_variable_map: Dict[str, str] = {}

    for idx, channel in enumerate(channels, 1):
        # Show all available metric variants for this channel
        # (Spend, Impressions, Clicks, Views, Reach, etc.)
        variants = channel_opts[channel]      # {label: column_name}
        labels   = list(variants.keys())
        cols     = list(variants.values())

        title = f" Channel {idx} / {n}: {channel} "
        print(f"\n  ┌─{title}{'─' * max(0, 54 - len(title))}┐")
        for i, (label, col) in enumerate(variants.items(), 1):
            print(f"  │  {i}. {label:<14s}  →  {col}")
        print(f"  └{'─' * 56}┘")

        if len(cols) == 1:
            chosen_col = cols[0]
            print("  (only one option — auto-selected)")
        elif not _can_prompt():
            chosen_col = cols[0]          # non-interactive: default to first
        else:
            raw = input("  Choose [default 1]: ").strip()
            try:
                chosen_idx = int(raw) - 1 if raw else 0
                if not (0 <= chosen_idx < len(cols)):
                    chosen_idx = 0
            except Exception:
                chosen_idx = 0
            chosen_col = cols[chosen_idx]

        channel_variable_map[channel] = chosen_col
        print(f"  ✓  {channel}  →  {chosen_col}")

    print("\n" + "─" * 62)
    print("  Selection summary:")
    for ch, col in sorted(channel_variable_map.items()):
        print(f"    {ch:<30s}  →  {col}")
    print("─" * 62)

    return channel_variable_map


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel prior selection  (terminal prompt)
# ─────────────────────────────────────────────────────────────────────────────

# Supported distributions and their required parameter names + sensible defaults
_BETA_DIST_PARAMS = {
    "half_normal" : {"sigma": 0.3},
    "gamma"       : {"alpha": 2.0, "beta": 2.0},
    "exponential" : {"lam": 3.0},
    "log_normal"  : {"mu": -1.0, "sigma": 0.5},
}
_LIKELIHOOD_OPTIONS = ["student_t", "normal", "skew_normal"]


def _prompt_float(label: str, default: float, allow_negative: bool = False) -> float:
    """Prompt for a float value, returning default on blank/invalid input."""
    raw = input(f"    {label} [default {default}]: ").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        if not allow_negative and v <= 0:
            print(f"    ⚠  Must be > 0 — using default ({default}).")
            return default
        return v
    except ValueError:
        print(f"    ⚠  Invalid — using default ({default}).")
        return default


def prompt_channel_prior_selection(
    spend_cols: List[str],
) -> Optional[Dict[str, "ChannelPriorConfig"]]:
    """
    Interactive terminal prompt that lets the user customise the Bayesian prior
    for each channel before the pipeline runs.

    For each channel the user can:
      • Accept the default  (HalfNormal sigma=0.3, StudentT likelihood)
      • Choose a different beta distribution and set its parameters
      • Choose a different likelihood family

    Returns
    -------
    channel_prior_map : dict  {column_name → ChannelPriorConfig}
        Only channels with non-default settings are included.
        Returns None (empty map) if all channels kept defaults.
    """
    if not _can_prompt():
        return None   # non-interactive: use all defaults

    print("\n" + "═" * 62)
    print("  CHANNEL PRIOR CONFIGURATION")
    print("  Customise the Bayesian prior for each channel.")
    print("  Press Enter at any prompt to accept the default.")
    print("═" * 62)

    # Ask whether the user wants to configure priors at all
    skip = _prompt_choice(
        "  Configure per-channel priors?",
        ["No — use defaults for all channels", "Yes — customise per channel"],
        default_idx=0,
    )
    if skip.startswith("No"):
        print("  ✓  All channels will use default priors (HalfNormal σ=0.3 + StudentT).")
        return None

    dist_options = list(_BETA_DIST_PARAMS.keys())
    channel_prior_map: Dict[str, ChannelPriorConfig] = {}

    for ch_col in spend_cols:
        print(f"\n  ┌─ {ch_col} {'─' * max(0, 54 - len(ch_col))}┐")
        print( "  │  Default: beta=half_normal(sigma=0.3)  likelihood=student_t")
        print( "  └" + "─" * 56 + "┘")

        keep_default = _prompt_choice(
            f"  Prior for '{ch_col}':",
            ["Keep default", "Customise"],
            default_idx=0,
        )
        if keep_default == "Keep default":
            print(f"  ✓  {ch_col} → default")
            continue

        # ── Beta distribution ──────────────────────────────────
        beta_dist = _prompt_choice(
            "  Beta prior distribution:",
            dist_options,
            default_idx=0,
        )
        param_defaults = _BETA_DIST_PARAMS[beta_dist]
        beta_params: Dict[str, Any] = {}
        print(f"  Parameters for {beta_dist}:")
        # Allow negative values for 'mu' params (e.g. log_normal.mu)
        _NEGATIVE_OK_PARAMS = {"mu"}
        for param, default_val in param_defaults.items():
            allow_neg = param in _NEGATIVE_OK_PARAMS
            beta_params[param] = _prompt_float(param, default_val, allow_negative=allow_neg)

        # ── Likelihood ─────────────────────────────────────────
        likelihood = _prompt_choice(
            "  Likelihood family:",
            _LIKELIHOOD_OPTIONS,
            default_idx=0,
        )
        likelihood_params: Dict[str, Any] = {}
        if likelihood == "skew_normal":
            alpha_skew = _prompt_float("alpha_skew (skewness, can be 0.0)", 0.0, allow_negative=True)
            likelihood_params = {"alpha_skew": alpha_skew}

        cfg = ChannelPriorConfig(
            beta_dist         = beta_dist,
            beta_params       = beta_params,
            likelihood        = likelihood,
            likelihood_params = likelihood_params,
            notes             = "set via main.py prompt",
        )
        channel_prior_map[ch_col] = cfg
        print(f"  ✓  {ch_col} → {cfg.describe()}")

    # Summary
    print("\n" + "─" * 62)
    print("  Prior configuration summary:")
    if channel_prior_map:
        for col, cfg in channel_prior_map.items():
            print(f"    {col:<35s}  {cfg.describe()}")
    else:
        print("    All channels → default (HalfNormal σ=0.3 + StudentT)")
    print("─" * 62)

    return channel_prior_map if channel_prior_map else None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_configured_pipeline(
    csv_path                  : str,
    response_col              : str,
    date_col                  : str,
    channel_variable_map      : Optional[Dict[str, str]] = None,
    baseline_type             : str  = "linear_trend",
    ranking_method            : str  = "lexicographic",
    runtime_mode              : str  = "fast_20m",
    use_hybrid_fast_scan      : bool = True,
    use_lag                   : bool = False,
    global_max_lag            : int  = 0,
    channel_prior_map         : Optional[Dict[str, "ChannelPriorConfig"]] = None,
    schema                    : Optional[Any] = None,
    response_transform        : str  = "log1p",
    max_quality_flag          : int  = 2,
    reject_if_all_bad         : bool = False,
    use_hierarchical          : bool = False,
    use_two_timescale_adstock : bool = False,
    use_synergies             : bool = False,
    use_campaigns             : bool = False,
    use_halos                 : bool = False,
    use_global_hill           : bool = False,
    use_time_varying_betas    : bool = False,
    use_dynamic_saturation    : bool = False,
    enable_strict_validation  : bool = True,
    enable_outlier_detection  : bool = True,
    enable_collinearity_check : bool = True,
    enable_stationarity_test  : bool = True,
    family_beta_sigma         : float = 0.5,
    channel_beta_sigma        : float = 0.3,
    product_beta_sigma        : float = 0.5,
) -> Dict[str, Any]:
    return run_full_pipeline(
        csv_path                  = csv_path,
        response_col              = response_col,
        spend_prefix              = "spends_",       # fallback if no map provided
        channel_variable_map      = channel_variable_map,
        date_col                  = date_col,
        dayfirst                  = True,
        min_date                  = None,
        max_date                  = None,
        control_cols              = [],
        output_dir                = "mmm_outputs",
        use_hierarchical          = use_hierarchical,
        top_n_map                 = 20,
        top_k_refit               = 4,
        draws_full                = 1200,
        tune_full                 = 600,
        chains_full               = 2,
        target_accept             = 0.95,
        baseline_type             = baseline_type,
        ranking_method            = ranking_method,
        rank_weights              = None,
        use_hybrid_fast_scan      = use_hybrid_fast_scan,
        fast_scan_trials          = 80,
        fast_scan_timeout_sec     = 240,
        runtime_mode              = runtime_mode,
        skip_stage0               = False,
        skip_map                  = False,
        use_lag                   = use_lag,
        global_max_lag            = global_max_lag,
        channel_prior_map         = channel_prior_map,
        schema                    = schema,
        response_transform        = response_transform,
        max_quality_flag          = max_quality_flag,
        reject_if_all_bad         = reject_if_all_bad,
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
    )


# ─────────────────────────────────────────────────────────────────────────────
# Notebook UI  (per-channel dropdowns + pipeline settings)
# ─────────────────────────────────────────────────────────────────────────────

def _run_with_notebook_dropdowns(
    default_csv     : str,
    default_response: str,
    default_date_col: str,
) -> Optional[Dict[str, Any]]:
    try:
        import ipywidgets as widgets  # type: ignore
        from IPython.display import display, clear_output  # type: ignore
    except Exception:
        return None

    try:
        df_head      = pd.read_csv(default_csv, nrows=5)
        channel_opts = discover_channels(df_head, exclude_cols=[default_date_col, default_response])
    except Exception:
        channel_opts = {}

    # One dropdown per channel ─────────────────────────────────
    channel_dropdowns: Dict[str, Any] = {}
    channel_rows: List[Any] = []
    for channel in sorted(channel_opts.keys()):
        variants = channel_opts[channel]
        options  = [(f"{label}  ({col})", col) for label, col in variants.items()]
        dd = widgets.Dropdown(
            options     = options,
            description = f"{channel}:",
            layout      = widgets.Layout(width="520px"),
            style       = {"description_width": "200px"},
        )
        channel_dropdowns[channel] = dd
        channel_rows.append(dd)

    # Pipeline settings ────────────────────────────────────────
    dd_baseline = widgets.Dropdown(
        options     = ["linear_trend", "gaussian_random_walk",
                       "noncentered_gaussian_random_walk", "piecewise_linear"],
        value       = "linear_trend",
        description = "Baseline:",
        layout      = widgets.Layout(width="420px"),
    )
    dd_rank = widgets.Dropdown(
        options     = ["lexicographic", "weighted", "mape_first"],
        value       = "lexicographic",
        description = "Ranking:",
        layout      = widgets.Layout(width="420px"),
    )
    dd_runtime = widgets.Dropdown(
        options     = ["fast_20m", "standard"],
        value       = "fast_20m",
        description = "Runtime:",
        layout      = widgets.Layout(width="420px"),
    )
    dd_hybrid = widgets.Dropdown(
        options     = [("Yes", True), ("No", False)],
        value       = False,
        description = "Hybrid scan:",
        layout      = widgets.Layout(width="420px"),
    )
    # Per-channel prior widgets ────────────────────────────────
    prior_dist_rows:  Dict[str, Any] = {}
    prior_param_rows: Dict[str, Any] = {}   # channel → {param_name → widget}
    prior_like_rows:  Dict[str, Any] = {}
    prior_ui_rows: List[Any] = []

    prior_ui_rows.append(widgets.HTML("<b>── Per-channel prior configuration ──</b>"))
    prior_ui_rows.append(widgets.HTML(
        "<i style='color:grey'>Leave on 'half_normal / sigma=0.3 / student_t' to use defaults.</i>"
    ))

    for channel in sorted(channel_opts.keys()):
        variants   = channel_opts[channel]
        # Use the CHANNEL NAME as the key; we'll remap to the actual selected
        # column in _on_run() so the prior always matches the user's metric choice.
        col_key    = channel

        dd_dist = widgets.Dropdown(
            options     = list(_BETA_DIST_PARAMS.keys()),
            value       = "half_normal",
            description = f"{channel} dist:",
            layout      = widgets.Layout(width="420px"),
            style       = {"description_width": "200px"},
        )
        # Parameter widgets — shown as a single text string "param=value, ..."
        txt_params = widgets.Text(
            value       = "sigma=0.3",
            description = "  params:",
            layout      = widgets.Layout(width="420px"),
            style       = {"description_width": "200px"},
        )
        dd_like = widgets.Dropdown(
            options     = _LIKELIHOOD_OPTIONS,
            value       = "student_t",
            description = "  likelihood:",
            layout      = widgets.Layout(width="420px"),
            style       = {"description_width": "200px"},
        )

        # Auto-fill params when dist changes
        def _make_dist_observer(txt, channel_name):
            def _obs(change):
                defaults = _BETA_DIST_PARAMS.get(change["new"], {})
                txt.value = ", ".join(f"{k}={v}" for k, v in defaults.items())
            return _obs
        dd_dist.observe(_make_dist_observer(txt_params, channel), names="value")

        prior_dist_rows[col_key]  = dd_dist
        prior_param_rows[col_key] = txt_params
        prior_like_rows[col_key]  = dd_like
        prior_ui_rows.extend([dd_dist, txt_params, dd_like])

    # Lag settings ─────────────────────────────────────────────
    dd_lag = widgets.Dropdown(
        options     = [("No lag", False), ("Apply lag", True)],
        value       = False,
        description = "Lag transform:",
        layout      = widgets.Layout(width="420px"),
    )
    int_max_lag = widgets.BoundedIntText(
        value       = 4,
        min         = 1,
        max         = 52,
        step        = 1,
        description = "Max lag (periods):",
        layout      = widgets.Layout(width="420px"),
        style       = {"description_width": "160px"},
    )

    run_btn = widgets.Button(description="▶  Run Pipeline", button_style="success")
    out     = widgets.Output()

    def _parse_prior_params(txt_value: str) -> Dict[str, Any]:
        """Parses 'alpha=2.0, beta=3.0' → {'alpha': 2.0, 'beta': 3.0}."""
        params: Dict[str, Any] = {}
        for token in txt_value.split(","):
            token = token.strip()
            if "=" in token:
                k, _, v = token.partition("=")
                try:
                    params[k.strip()] = float(v.strip())
                except ValueError:
                    pass
        return params

    def _on_run(_):
        with out:
            clear_output(wait=True)
            cvm = {ch: dd.value for ch, dd in channel_dropdowns.items()}
            print("Channel selections:")
            for ch, col in sorted(cvm.items()):
                print(f"  {ch:<30s} → {col}")

            # Build channel_prior_map from notebook widgets
            # prior widget keys are channel NAMES; remap to the actual selected
            # column names so validate_channel_prior_map succeeds.
            nb_prior_map: Dict[str, ChannelPriorConfig] = {}
            for ch_key in prior_dist_rows:
                dist   = prior_dist_rows[ch_key].value
                params = _parse_prior_params(prior_param_rows[ch_key].value)
                like   = prior_like_rows[ch_key].value
                # Only store if non-default
                is_default = (dist == "half_normal" and
                              abs(params.get("sigma", 0) - 0.3) < 1e-9 and
                              like == "student_t")
                if not is_default:
                    # Resolve to the actual column the user selected for this channel
                    actual_col = cvm.get(ch_key, ch_key)
                    try:
                        nb_prior_map[actual_col] = ChannelPriorConfig(
                            beta_dist   = dist,
                            beta_params = params,
                            likelihood  = like,
                            notes       = "set via notebook widget",
                        )
                    except Exception as e:
                        print(f"  ⚠  Invalid prior for '{ch_key}': {e} — using default.")
            channel_prior_map_nb = nb_prior_map if nb_prior_map else None
            if channel_prior_map_nb:
                print("\nCustom priors:")
                for col, pcfg in channel_prior_map_nb.items():
                    print(f"  {col:<35s} → {pcfg.describe()}")

            use_lag_val = bool(dd_lag.value)
            max_lag_val = int(int_max_lag.value) if use_lag_val else 0
            print(f"\nbaseline={dd_baseline.value} | ranking={dd_rank.value} | "
                  f"runtime={dd_runtime.value} | hybrid={dd_hybrid.value} | "
                  f"lag={use_lag_val} max={max_lag_val}\n")
            try:
                _run_configured_pipeline(
                    csv_path             = default_csv,
                    response_col         = default_response,
                    date_col             = default_date_col,
                    channel_variable_map = cvm,
                    baseline_type        = dd_baseline.value,
                    ranking_method       = dd_rank.value,
                    runtime_mode         = dd_runtime.value,
                    use_hybrid_fast_scan = bool(dd_hybrid.value),
                    use_lag              = use_lag_val,
                    global_max_lag       = max_lag_val,
                    channel_prior_map    = channel_prior_map_nb,
                )
            except Exception as e:
                print(f"Run failed: {e}")

    run_btn.on_click(_on_run)
    display(widgets.VBox([
        widgets.HTML("<b>── Channel variable selection ──</b>"),
        *channel_rows,
        *prior_ui_rows,
        widgets.HTML("<b>── Pipeline settings ──</b>"),
        dd_baseline, dd_rank, dd_runtime, dd_hybrid,
        widgets.HTML("<b>── Lag transformation ──</b>"),
        dd_lag, int_max_lag,
        run_btn, out,
    ]))
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner (batch_runner.py)
# ─────────────────────────────────────────────────────────────────────────────

import yaml as _yaml
from pathlib import Path as _BatchPath
from typing import Dict as _Dict_br

def _build_channel_prior_map(schema) -> "_Dict_br[str, ChannelPriorConfig]":
    """Builds per-column priors from the domain prior library (schema-driven)."""
    from priors import get_domain_prior
    channel_prior_map = {}
    _is_multi = getattr(schema, "is_multi_product", False)
    for mc in schema.media_cols:
        if mc.channel_type:
            dp = get_domain_prior(mc.channel_type)
            key_col = mc.channel if _is_multi else mc.column
            if key_col in channel_prior_map:
                continue
            channel_prior_map[key_col] = ChannelPriorConfig(
                beta_dist=dp.beta_dist,
                beta_params=dp.beta_params.copy(),
                likelihood=dp.likelihood,
                adstock_lam_prior=dp.adstock_lam_prior,
                sat_alpha_prior=dp.sat_alpha_prior,
                sat_kappa_prior=dp.sat_kappa_prior,
            )
    return channel_prior_map


def run_batch_from_yaml(batch_yaml_path: str) -> None:
    """
    Runs MMM for multiple clients.

    Expected batch YAML shape:
    clients:
      - id: client_a
        config: path/to/client_a_mmm_config.yaml
      - id: client_b
        config: path/to/client_b_mmm_config.yaml
    """
    from priors import apply_yaml_prior_overrides
    from config import load_config_yaml, build_schema_from_yaml

    batch_path = _BatchPath(batch_yaml_path)
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch YAML not found: {batch_yaml_path}")

    payload = _yaml.safe_load(batch_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "clients" not in payload:
        raise ValueError("Batch YAML must be a dict with a top-level 'clients' list.")

    clients = payload.get("clients", [])
    if not isinstance(clients, list) or not clients:
        raise ValueError("'clients' must be a non-empty list.")

    for idx, client_spec in enumerate(clients, 1):
        if not isinstance(client_spec, dict):
            raise ValueError(f"clients[{idx}] must be an object/dict.")

        client_id = str(client_spec.get("id", f"client_{idx}"))
        config_path = client_spec.get("config")
        if not config_path:
            raise ValueError(f"clients[{idx}] must include 'config' path.")

        client_cfg_path = _BatchPath(config_path)
        if not client_cfg_path.exists():
            raise FileNotFoundError(f"Client config not found: {client_cfg_path}")

        logger.info(f"=== [BATCH] ({idx}/{len(clients)}) Running {client_id} ===")

        cfg_dict = load_config_yaml(str(client_cfg_path))
        schema = build_schema_from_yaml(cfg_dict)
        data_sec     = cfg_dict.get("data",         {})
        model_sec    = cfg_dict.get("model",         {})
        samp_sec     = cfg_dict.get("sampling",      {})
        out_sec      = cfg_dict.get("output",        {})
        pipeline_sec = cfg_dict.get("pipeline",      {})
        hier_sec     = cfg_dict.get("hierarchical",  {})
        prior_sec    = cfg_dict.get("priors",        {}) or {}
        val_sec      = cfg_dict.get("validation",    {})

        channel_prior_map = _build_channel_prior_map(schema)
        channel_prior_map = apply_yaml_prior_overrides(
            channel_prior_map, cfg_dict, schema
        )

        _pool_sec         = prior_sec.get("pooling", {}) or {}
        _family_beta_sig  = float(_pool_sec.get("family_beta_sigma",
                                  hier_sec.get("family_beta_sigma", 0.5)))
        _channel_beta_sig = float(_pool_sec.get("channel_beta_sigma",
                                  hier_sec.get("channel_beta_sigma", 0.3)))
        _product_beta_sig = float(_pool_sec.get("product_beta_sigma",
                                  hier_sec.get("product_beta_sigma", 0.5)))

        output_dir_base = out_sec.get("dir", "mmm_outputs")
        output_dir = str(_BatchPath(output_dir_base) / client_id)

        run_full_pipeline(
            csv_path                  = data_sec.get("csv_path", ""),
            response_col              = schema.response_col,
            date_col                  = schema.date_col,
            dayfirst                  = schema.dayfirst,
            frequency                 = data_sec.get("frequency", "weekly"),
            holdout_periods           = int(data_sec.get("holdout_periods", 0)),
            input_format              = data_sec.get("input_format", "wide"),
            product_col               = data_sec.get("product_col", "Product"),
            long_response_col         = data_sec.get("long_response_col", ""),
            long_media_cols           = data_sec.get("long_media_cols", {}) or {},
            long_spend_cols           = data_sec.get("long_spend_cols", {}) or {},
            channel_variable_map      = schema.channel_variable_map,
            control_cols              = schema.all_control_column_names,
            output_dir                = output_dir,
            baseline_type             = model_sec.get("baseline_type", "linear_trend"),
            use_lag                   = model_sec.get("use_lag", False),
            global_max_lag            = model_sec.get("max_lag", 0),
            runtime_mode              = samp_sec.get("runtime_mode", "fast_20m"),
            draws_full                = samp_sec.get("draws", 1800),
            tune_full                 = samp_sec.get("tune", 800),
            chains_full               = samp_sec.get("chains", 2),
            target_accept             = samp_sec.get("target_accept", 0.95),
            max_quality_flag          = samp_sec.get("max_quality_flag", 1),
            reject_if_all_bad         = samp_sec.get("reject_if_all_bad", True),
            channel_prior_map         = channel_prior_map or None,
            schema                    = schema,
            enable_strict_validation  = bool(val_sec.get("enable_strict_validation")
                                            if "enable_strict_validation" in val_sec
                                            else hier_sec.get("enable_strict_validation", True)),
            enable_outlier_detection  = bool(val_sec.get("enable_outlier_detection")
                                            if "enable_outlier_detection" in val_sec
                                            else hier_sec.get("enable_outlier_detection", True)),
            enable_collinearity_check = bool(val_sec.get("enable_collinearity_check")
                                            if "enable_collinearity_check" in val_sec
                                            else hier_sec.get("enable_collinearity_check", True)),
            enable_stationarity_test  = bool(val_sec.get("enable_stationarity_test")
                                            if "enable_stationarity_test" in val_sec
                                            else hier_sec.get("enable_stationarity_test", True)),
            use_hybrid_fast_scan      = bool(pipeline_sec.get("use_hybrid_fast_scan", False)),
            response_transform        = pipeline_sec.get("response_transform", "log1p"),
            use_dow_effects           = pipeline_sec.get("use_dow_effects", None),
            skip_stage0               = bool(pipeline_sec.get("skip_stage0", False)),
            skip_map                  = bool(pipeline_sec.get("skip_map", False)),
            top_n_map                 = int(pipeline_sec.get("top_n_map", 20)),
            top_k_refit               = int(pipeline_sec.get("top_k_refit", 4)),
            use_hierarchical          = bool(hier_sec.get("enabled", False)),
            use_two_timescale_adstock = bool(hier_sec.get("use_two_timescale_adstock", False)),
            use_synergies             = bool(hier_sec.get("use_synergies", False)),
            use_campaigns             = bool(hier_sec.get("use_campaigns", False)),
            use_halos                 = bool(hier_sec.get("use_halos", False)),
            use_global_hill           = bool(hier_sec.get("use_global_hill", False)),
            use_time_varying_betas    = bool(hier_sec.get("use_time_varying_betas", False)),
            use_dynamic_saturation    = bool(hier_sec.get("use_dynamic_saturation", False)),
            family_beta_sigma         = _family_beta_sig,
            channel_beta_sigma        = _channel_beta_sig,
            product_beta_sigma        = _product_beta_sig,
            opt_config                = cfg_dict.get("optimisation") or {},
        )

        logger.info(f"=== [BATCH] {client_id} finished ===")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Bayesian MMM Pipeline")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to the CSV data file")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides all other args)")
    parser.add_argument("--response", type=str,
                        default="revenue",
                        help="Response column name (must match CSV column exactly)")
    parser.add_argument("--date-col", type=str, default="date",
                        help="Date column name")
    parser.add_argument("--output-dir", type=str, default="mmm_outputs",
                        help="Output directory")
    parser.add_argument("--use-hierarchical", action="store_true",
                        help="Enable hierarchical model priors and family pooling")
    parser.add_argument("--use-global-hill", action="store_true",
                        help="Use shared Hill saturation parameters across channels")
    parser.add_argument("--use-time-varying-betas", action="store_true",
                        help="Enable time-varying channel betas")
    parser.add_argument("--use-dynamic-saturation", action="store_true",
                        help="Enable time-varying Hill saturation over time")
    parser.add_argument("--strict-validation", dest="enable_strict_validation",
                        action="store_true", default=True,
                        help="Enable strict data validation checks")
    parser.add_argument("--no-strict-validation", dest="enable_strict_validation",
                        action="store_false",
                        help="Disable strict validation checks")
    parser.add_argument("--outlier-detection", dest="enable_outlier_detection",
                        action="store_true", default=True,
                        help="Enable automatic outlier detection")
    parser.add_argument("--no-outlier-detection", dest="enable_outlier_detection",
                        action="store_false",
                        help="Disable automatic outlier detection")
    parser.add_argument("--collinearity-check", dest="enable_collinearity_check",
                        action="store_true", default=True,
                        help="Enable multicollinearity diagnostics")
    parser.add_argument("--no-collinearity-check", dest="enable_collinearity_check",
                        action="store_false",
                        help="Disable multicollinearity diagnostics")
    parser.add_argument("--stationarity-test", dest="enable_stationarity_test",
                        action="store_true", default=True,
                        help="Enable stationarity tests (ADF/KPSS)")
    parser.add_argument("--no-stationarity-test", dest="enable_stationarity_test",
                        action="store_false",
                        help="Disable stationarity tests")
    parser.add_argument("--batch-yaml", type=str, default=None,
                        help="Run multiple clients from a batch YAML file")
    args = parser.parse_args()

    # ── YAML config mode (new — bypasses interactive prompts) ──
    if args.batch_yaml:
        run_batch_from_yaml(args.batch_yaml)
        sys.exit(0)

    if args.config:
        from config import load_config_yaml, build_schema_from_yaml
        from priors import get_domain_prior
        import yaml

        cfg_dict = load_config_yaml(args.config)
        schema   = build_schema_from_yaml(cfg_dict)
        data_sec = cfg_dict.get("data", {})
        model_sec = cfg_dict.get("model", {})
        samp_sec  = cfg_dict.get("sampling", {})
        out_sec   = cfg_dict.get("output", {})
        pipe_sec  = cfg_dict.get("pipeline", {})
        hier_sec  = cfg_dict.get("hierarchical", {})
        val_sec   = cfg_dict.get("validation", {})
        prior_sec = cfg_dict.get("priors", {})       # ← unified priors section

        print(f"\n  Loaded YAML config: {args.config}")
        print(schema.describe())

        # Build channel_prior_map from prior library.
        # In multi-product mode spend_cols is keyed by unique channel name;
        # in single-product mode it is keyed by full column name.
        channel_prior_map = {}
        _is_multi = getattr(schema, "is_multi_product", False)
        for mc in schema.media_cols:
            if mc.channel_type:
                key_col = mc.channel if _is_multi else mc.column
                if key_col in channel_prior_map:
                    continue  # deduplicate: one prior per unique key
                dp = get_domain_prior(mc.channel_type)
                channel_prior_map[key_col] = ChannelPriorConfig(
                    beta_dist       = dp.beta_dist,
                    beta_params     = dp.beta_params.copy(),
                    likelihood      = dp.likelihood,
                    adstock_lam_prior = dp.adstock_lam_prior,
                    sat_alpha_prior = dp.sat_alpha_prior,
                    sat_kappa_prior = dp.sat_kappa_prior,
                )

        # ── Build ChannelFamilyConfig objects for adstock half-life params ───────
        # These control how fast/slow each channel FAMILY's adstock decays and
        # are passed directly to run_full_pipeline as family_configs.
        # The prior beta/adstock/likelihood overrides are applied separately below.
        from config import ChannelFamilyConfig as _CFC
        from priors import apply_yaml_prior_overrides
        _fam_sec     = prior_sec.get("families", {}) or {}
        _family_cfgs = {}
        _hl_unit = "days" if data_sec.get("frequency", "weekly") == "daily" else "weeks"

        for _fname, _fcfg in _fam_sec.items():
            if not isinstance(_fcfg, dict):
                continue
            _family_cfgs[_fname] = _CFC(
                name            = _fname,
                hl_slow_median  = float(_fcfg.get("hl_slow_median",  3.0)),
                hl_ratio_median = float(_fcfg.get("hl_ratio_median", 3.0)),
                hl_sigma        = float(_fcfg.get("hl_sigma",        0.5)),
            )
            print(f"    {_fname}: hl_slow_median={_family_cfgs[_fname].hl_slow_median} {_hl_unit}  "
                  f"hl_sigma={_family_cfgs[_fname].hl_sigma}")

        # ── Apply all prior override layers via shared function ────────────────
        # Layer 3 (priors.families beta fields) → Layer 2 (channel_priors) →
        # Layer 1 (priors.channels) — highest precedence wins.
        channel_prior_map = apply_yaml_prior_overrides(channel_prior_map, cfg_dict, schema)

        # ── LAYER 1: priors.pooling — family pooling sigma overrides
        # Takes precedence over hierarchical.family_beta_sigma / channel_beta_sigma.
        _pool_sec         = prior_sec.get("pooling", {}) or {}
        _family_beta_sig  = float(_pool_sec.get("family_beta_sigma",
                                  hier_sec.get("family_beta_sigma", 0.5)))
        _channel_beta_sig = float(_pool_sec.get("channel_beta_sigma",
                                  hier_sec.get("channel_beta_sigma", 0.3)))
        _product_beta_sig = float(_pool_sec.get("product_beta_sigma",
                                  hier_sec.get("product_beta_sigma", 0.5)))
        if _pool_sec:
            print(f"\n  [PRIORS] Pooling sigmas from priors.pooling: "
                  f"family={_family_beta_sig}, channel={_channel_beta_sig}, product={_product_beta_sig}")

        # (priors.families parsed above in LAYER 3a — _fam_sec and _family_cfgs already set)

        best_model = run_full_pipeline(
            csv_path             = data_sec.get("csv_path", args.csv or ""),
            response_col         = schema.response_col,
            date_col             = schema.date_col,
            dayfirst             = schema.dayfirst,
            min_date             = data_sec.get("min_date"),
            max_date             = data_sec.get("max_date"),
            frequency            = data_sec.get("frequency", schema.frequency or "weekly"),
            holdout_periods      = int(data_sec.get("holdout_periods", 0)),
            channel_variable_map = schema.channel_variable_map,
            control_cols         = schema.all_control_column_names,
            output_dir           = out_sec.get("dir", "mmm_outputs"),
            baseline_type        = model_sec.get("baseline_type", "linear_trend"),
            use_lag              = model_sec.get("use_lag", False),
            global_max_lag       = model_sec.get("max_lag", 0),
            runtime_mode         = samp_sec.get("runtime_mode", "fast_20m"),
            draws_full           = samp_sec.get("draws", 1800),
            tune_full            = samp_sec.get("tune", 800),
            chains_full          = samp_sec.get("chains", 2),
            target_accept        = samp_sec.get("target_accept", 0.95),
            max_quality_flag     = samp_sec.get("max_quality_flag", 2),
            reject_if_all_bad    = samp_sec.get("reject_if_all_bad", False),
            channel_prior_map    = channel_prior_map or None,
            schema               = schema,
            # Pipeline toggles (optional)
            skip_stage0          = bool(pipe_sec.get("skip_stage0", False)),
            skip_map             = bool(pipe_sec.get("skip_map", False)),
            top_n_map            = int(pipe_sec.get("top_n_map", 20)),
            top_k_refit          = int(pipe_sec.get("top_k_refit", 4)),
            use_hybrid_fast_scan      = bool(pipe_sec.get("use_hybrid_fast_scan", False)),
            response_transform        = pipe_sec.get("response_transform", "log1p"),
            # Hierarchical options (from hierarchical: section of YAML)
            use_hierarchical          = bool(hier_sec.get("enabled", False)),
            use_two_timescale_adstock = bool(hier_sec.get("use_two_timescale_adstock", False)),
            use_synergies             = bool(hier_sec.get("use_synergies", False)),
            use_campaigns             = bool(hier_sec.get("use_campaigns", False)),
            use_halos                 = bool(hier_sec.get("use_halos", False)),
            use_global_hill           = bool(hier_sec.get("use_global_hill", False)),
            use_time_varying_betas    = bool(hier_sec.get("use_time_varying_betas", False)),
            use_dynamic_saturation    = bool(hier_sec.get("use_dynamic_saturation", False)),
            # Read validation settings from EITHER hierarchical or validation section (prefer validation)
            enable_strict_validation  = bool(val_sec.get("enable_strict_validation") 
                                            if "enable_strict_validation" in val_sec
                                            else hier_sec.get("enable_strict_validation", True)),
            enable_outlier_detection  = bool(val_sec.get("enable_outlier_detection")
                                            if "enable_outlier_detection" in val_sec
                                            else hier_sec.get("enable_outlier_detection", True)),
            enable_collinearity_check = bool(val_sec.get("enable_collinearity_check")
                                            if "enable_collinearity_check" in val_sec
                                            else hier_sec.get("enable_collinearity_check", True)),
            enable_stationarity_test  = bool(val_sec.get("enable_stationarity_test")
                                            if "enable_stationarity_test" in val_sec
                                            else hier_sec.get("enable_stationarity_test", True)),
            # Pooling sigmas — priors.pooling takes precedence over hierarchical section
            family_beta_sigma         = _family_beta_sig,
            channel_beta_sigma        = _channel_beta_sig,
            product_beta_sigma        = _product_beta_sig,
            # Family adstock configs — priors.families overrides DEFAULT_FAMILY_CONFIGS
            family_configs            = _family_cfgs or None,
            # Long-format auto-pivot — from data section of YAML
            input_format              = data_sec.get("input_format", "wide"),
            product_col               = data_sec.get("product_col", "Product"),
            long_response_col         = data_sec.get("long_response_col", ""),
            long_media_cols           = data_sec.get("long_media_cols", {}) or {},
            long_spend_cols           = data_sec.get("long_spend_cols", {}) or {},
            # Optimisation settings — from optimisation: section of YAML
            opt_config                = cfg_dict.get("optimisation") or {},
        )
        sys.exit(0)

    # ── Interactive mode ───────────────────────────────────────
    # CSV path: prefer --csv flag; otherwise prompt the user interactively.
    if args.csv:
        default_csv = args.csv
    elif _can_prompt():
        print("\n" + "─" * 62)
        print("  DATA FILE")
        print("─" * 62)
        print("  No --csv argument supplied. Enter the path to your CSV file.")
        print("  Tip: drag-and-drop the file onto this terminal window.\n")
        raw_csv = input("  CSV path: ").strip().strip('"').strip("'")
        if not raw_csv:
            raise SystemExit(
                "No CSV path provided. "
                "Run: python main.py --csv path/to/data.csv"
            )
        from pathlib import Path as _Path
        if not _Path(raw_csv).exists():
            raise SystemExit(f"File not found: {raw_csv}")
        default_csv = raw_csv
    else:
        raise SystemExit(
            "ERROR: --csv is required.\n"
            "Usage: python main.py --csv path/to/data.csv --response your_response_col\n"
            "       or: python main.py --config path/to/config.yaml"
        )

    # Response column: prefer --response flag; otherwise prompt.
    if args.response:
        default_response = args.response
    elif _can_prompt():
        # Peek at CSV header to show available columns.
        try:
            _peek = pd.read_csv(default_csv, nrows=0)
            _cols = [c for c in _peek.columns if c.lower() not in ("date", "week", "month")]
            print(f"\n  Columns in CSV (excluding date-like): {_cols[:10]}"
                  + (" …" if len(_cols) > 10 else ""))
        except Exception:
            pass
        print("\n  Enter the name of your response (KPI) column.")
        raw_resp = input("  Response column [default 'revenue']: ").strip()
        default_response = raw_resp if raw_resp else "revenue"
    else:
        default_response = args.response  # use whatever default is set

    default_date_col = args.date_col

    launched_notebook_ui = False
    if _is_notebook():
        maybe = _run_with_notebook_dropdowns(default_csv, default_response, default_date_col)
        if maybe is None:
            print("ipywidgets not available — falling back to text prompts.")
        else:
            launched_notebook_ui = True

    if not launched_notebook_ui:

        # ── Step 1: ask user which metric to model per channel ─
        channel_variable_map = prompt_channel_variable_selection(
            csv_path     = default_csv,
            date_col     = default_date_col,
            response_col = default_response,
        )

        # ── Step 2: lag settings ───────────────────────────────
        lag_settings = prompt_lag_settings(
            csv_path = default_csv,
            date_col = default_date_col,
        )

        # ── Step 3: per-channel prior configuration ────────────
        # spend_cols at this point is derived from channel_variable_map values
        prior_spend_cols = list(channel_variable_map.values())
        channel_prior_map = prompt_channel_prior_selection(
            spend_cols = prior_spend_cols,
        )

        # ── Step 4: remaining pipeline settings ───────────────
        baseline_type = _prompt_choice(
            "Select baseline distribution:",
            ["linear_trend", "gaussian_random_walk",
             "noncentered_gaussian_random_walk", "piecewise_linear"],
            default_idx=0,
        )
        ranking_method = _prompt_choice(
            "Select model ranking method:",
            ["lexicographic", "weighted", "mape_first"],
            default_idx=0,
        )
        runtime_mode = _prompt_choice(
            "Select runtime profile:",
            ["fast_20m", "standard"],
            default_idx=0,
        )
        hybrid_choice        = _prompt_choice("Use hybrid fast scan?", ["yes", "no"], default_idx=1)
        use_hybrid_fast_scan = (hybrid_choice == "yes")

        # Response transform is always log1p in interactive mode.
        # To use a different transform (sqrt, boxcox, identity), set
        # response_transform in the pipeline: section of your YAML config.
        response_transform = "log1p"

        # ── Step 4b: hierarchical model options ────────────────
        hier_choice = _prompt_choice(
            "Enable hierarchical model (family-grouped priors)?",
            ["no", "yes"],
            default_idx=0,
        )
        use_two_timescale_adstock = False
        use_synergies             = False
        use_hierarchical          = False
        use_global_hill           = False
        use_time_varying_betas    = False
        use_dynamic_saturation    = False
        enable_strict_validation  = True
        enable_outlier_detection  = True
        enable_collinearity_check = True
        enable_stationarity_test  = True
        family_beta_sigma         = 0.5
        channel_beta_sigma        = 0.3
        product_beta_sigma        = 0.5

        if hier_choice == "yes":
            ts_choice = _prompt_choice(
                "  Use two-timescale adstock (slow+fast decay chains)?",
                ["no", "yes"],
                default_idx=0,
            )
            use_two_timescale_adstock = (ts_choice == "yes")

            syn_choice = _prompt_choice(
                "  Enable cross-channel family synergies?",
                ["no", "yes"],
                default_idx=0,
            )
            use_synergies = (syn_choice == "yes")

            use_hierarchical = True

        # ── Advanced model options ───────────────────────────────
        adv_choice = _prompt_choice(
            "Enable advanced model features (time-varying betas, dynamic saturation)?",
            ["no", "yes"],
            default_idx=0,
        )
        if adv_choice == "yes":
            tvb_choice = _prompt_choice(
                "  Enable time-varying channel betas?",
                ["no", "yes"],
                default_idx=0,
            )
            use_time_varying_betas = (tvb_choice == "yes")

            ds_choice = _prompt_choice(
                "  Enable dynamic (time-varying) saturation?",
                ["no", "yes"],
                default_idx=0,
            )
            use_dynamic_saturation = (ds_choice == "yes")

            gh_choice = _prompt_choice(
                "  Use global Hill saturation (shared alpha/kappa)?",
                ["no", "yes"],
                default_idx=0,
            )
            use_global_hill = (gh_choice == "yes")

        # ── Data validation options ──────────────────────────────
        val_choice = _prompt_choice(
            "Configure data validation (outliers, stationarity, collinearity)?",
            ["default (all enabled)", "custom"],
            default_idx=0,
        )
        if val_choice == "custom":
            od_choice = _prompt_choice(
                "  Enable outlier detection?",
                ["yes", "no"],
                default_idx=0,
            )
            enable_outlier_detection = (od_choice == "yes")

            st_choice = _prompt_choice(
                "  Enable stationarity tests?",
                ["yes", "no"],
                default_idx=0,
            )
            enable_stationarity_test = (st_choice == "yes")

            cc_choice = _prompt_choice(
                "  Enable collinearity checks?",
                ["yes", "no"],
                default_idx=0,
            )
            enable_collinearity_check = (cc_choice == "yes")

            sv_choice = _prompt_choice(
                "  Enable strict validation (fail on issues)?",
                ["yes", "no"],
                default_idx=0,
            )
            enable_strict_validation = (sv_choice == "yes")

        print(
            f"\nRunning pipeline → baseline={baseline_type} | ranking={ranking_method} | "
            f"runtime={runtime_mode} | hybrid_fast_scan={use_hybrid_fast_scan} | "
            f"response_transform=log1p (default; override via YAML) | "
            f"lag={lag_settings['use_lag']} max={lag_settings['global_max_lag']} {lag_settings['time_unit']} | "
            f"hierarchical={hier_choice=='yes'} two_timescale={use_two_timescale_adstock} "
            f"synergies={use_synergies} | "
            f"time_varying_betas={use_time_varying_betas} dynamic_sat={use_dynamic_saturation} "
            f"global_hill={use_global_hill} | "
            f"strict_val={enable_strict_validation} outliers={enable_outlier_detection} "
            f"stationarity={enable_stationarity_test} collinearity={enable_collinearity_check}"
        )

        # ── Step 5: run ────────────────────────────────────────
        best_model = _run_configured_pipeline(
            csv_path                  = default_csv,
            response_col              = default_response,
            date_col                  = default_date_col,
            channel_variable_map      = channel_variable_map,
            baseline_type             = baseline_type,
            ranking_method            = ranking_method,
            runtime_mode              = runtime_mode,
            use_hybrid_fast_scan      = use_hybrid_fast_scan,
            use_lag                   = lag_settings["use_lag"],
            global_max_lag            = lag_settings["global_max_lag"],
            channel_prior_map         = channel_prior_map,
            response_transform        = response_transform,
            use_two_timescale_adstock = use_two_timescale_adstock,
            use_synergies             = use_synergies,
            use_hierarchical          = use_hierarchical,
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
        )

    # ── Print results ─────────────────────────────────────────
    if "best_model" in locals():
        cfg     = best_model["cfg"]
        metrics = best_model["metrics"]
        prep    = best_model["prep"]

        rhat_label = (
            f"{metrics['max_rhat']:.5f} [estimated — 1 chain]"
            if metrics.get("max_rhat_is_estimated") else f"{metrics['max_rhat']:.5f}"
        )

        print(f"\n★  Best model   : {cfg.key()}")
        print(f"   MAPE         : {metrics['mape']:.2f}%")
        print(f"   R2           : {metrics['r2']:.4f}")
        print(f"   Max R-hat    : {rhat_label}")
        print(f"   Divergences  : {metrics['divergences']}")
        print(f"   LOO-IC       : {metrics['loo_ic']:.4f}")
        print(f"   Quality flag : {quality_flag(metrics)} (0=production ready)")

        if prep.get("channel_variable_map"):
            print("\n   Variables modelled:")
            for ch, col in sorted(prep["channel_variable_map"].items()):
                print(f"     {ch:<30s}  →  {col}")

        if "channel_transform_summary" in best_model and best_model["channel_transform_summary"] is not None:
            print("\n   Per-Channel Transform Summary:")
            print("   " + "─" * 80)
            df_ts        = best_model["channel_transform_summary"]
            display_cols = ["channel", "adstock", "saturation",
                            "use_lag", "lag_mode", "lam_mean",
                            "alpha_sat_mean", "kappa_mean", "beta_mean"]
            show = df_ts[[c for c in display_cols if c in df_ts.columns]]
            for line in show.to_string(index=False).splitlines():
                print(f"   {line}")
            print("   " + "─" * 80)
            print("   Full parameter table saved to: mmm_outputs/channel_transform_summary.csv")
