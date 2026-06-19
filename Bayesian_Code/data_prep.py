# data_prep.py
# ─────────────────────────────────────────────────────────────────────────────
# Loads your CSV/Excel, validates the data, and prepares it for modelling.
#
# What this file does:
#   • Reads the data file and parses dates
#   • Validates data quality (outliers, collinearity, stationarity)
#   • Scales the response variable (log1p, sqrt, boxcox, or identity)
#   • Normalises media spend columns to the [0,1] range
#   • Builds Fourier features for annual seasonality
#   • Handles holdout splits for out-of-sample testing
#   • Supports long-format data (auto-pivots to wide format for multi-product)
#
# The output is a "prep" dict consumed by model_builder.py and all analysis scripts.
# ─────────────────────────────────────────────────────────────────────────────

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import DataConfig, ModelConfig

logger = logging.getLogger("MMM")

# ─────────────────────────────────────────────────────────────────────────────
# Fourier period mapping — seasonal cycle length per dataset frequency
# ─────────────────────────────────────────────────────────────────────────────
FOURIER_PERIODS: dict = {
    "weekly"  : 52.18,   # 365.25 / 7 — true average weeks per year
    "daily"   : 365.25,  # calendar year in days
    "monthly" : 12.0,    # 12 months per year
}

# ── Known metric prefixes in display-priority order ───────────────────────────
# Longer / more-specific prefixes MUST come before shorter ones so the
# startswith() check in discover_channels() matches the right prefix first.
# e.g. "media_impressions_" before "impressions_", "media_clicks_" before "clicks_"
METRIC_PREFIXES = [
    "media_impressions_",   # new canonical impression prefix
    "media_clicks_",        # new canonical clicks prefix
    "spends_",              # spend (original + new)
    "imps_",                # legacy short impressions
    "impressions_",         # legacy long impressions
    "clicks_",              # legacy clicks
    "clks_",                # legacy short clicks
    "views_",
    "reach_",
]

PREFIX_LABELS = {
    "media_impressions_" : "Impressions",
    "media_clicks_"      : "Clicks",
    "spends_"            : "Spend",
    "imps_"              : "Impressions",
    "impressions_"       : "Impressions",
    "clicks_"            : "Clicks",
    "clks_"              : "Clicks",
    "views_"             : "Views",
    "reach_"             : "Reach",
}


# FIX-2: helper — returns the metric label ("Spend", "Clicks", etc.) for a column name
def _metric_type_for_col(col: str) -> str:
    """Returns the human-readable metric label for a column based on its prefix."""
    for prefix in METRIC_PREFIXES:
        if col.startswith(prefix):
            return PREFIX_LABELS.get(prefix, "Metric")
    return "Metric"


# FIX-2: helper — derives the corresponding spends_* column name for any metric column.
# For a spend column it returns itself; for clicks/impressions it returns the parallel
# spends_<channel> column name so compute_return_index() can look up actual spend.
def _spend_col_for(col: str) -> str:
    """Returns the expected spends_* column for a given media column."""
    for prefix in METRIC_PREFIXES:
        if col.startswith(prefix):
            channel_suffix = col[len(prefix):]
            return f"spends_{channel_suffix}"
    return col  # fallback: return as-is


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_scale(arr: np.ndarray, floor: float = 1e-8) -> Tuple[np.ndarray, float, float]:
    """Z-score standardise. Returns (scaled_array, mean, std)."""
    mu  = float(np.mean(arr))
    std = float(np.std(arr))
    if std < floor:
        std = 1.0
    return (arr - mu) / std, mu, std


def scale_columns(X: np.ndarray) -> np.ndarray:
    """Z-score each column of a 2-D array independently. Returns scaled array."""
    return np.column_stack([safe_scale(X[:, j])[0] for j in range(X.shape[1])])


def detect_outliers_iqr(series: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    """Return a boolean mask of outliers based on the IQR method."""
    q1, q3 = np.percentile(series, [25.0, 75.0])
    iqr = q3 - q1
    if iqr <= 0.0:
        return np.zeros_like(series, dtype=bool)
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    return (series < lower) | (series > upper)


def detect_outliers_robust_zscore(series: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Return a boolean mask of outliers using the robust z-score method."""
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    if mad < 1e-8:
        return np.zeros_like(series, dtype=bool)
    robust_z = 0.6745 * (series - median) / mad
    return np.abs(robust_z) > threshold


def run_stationarity_tests(series: np.ndarray) -> dict:
    """Run ADF and KPSS stationarity tests if the required package is available."""
    results = {}
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except Exception:
        logger.warning(
            "Stationarity tests require statsmodels. Skipping ADF/KPSS checks. "
            "Install statsmodels to enable stationarity validation."
        )
        return results

    try:
        adf = adfuller(series, autolag="AIC")
        results["adf_stat"] = float(adf[0])
        results["adf_pvalue"] = float(adf[1])
    except Exception as exc:
        logger.warning(f"ADF test failed: {exc}")

    try:
        kpss_stat, kpss_p, _, _ = kpss(series, regression="c", nlags="auto")
        results["kpss_stat"] = float(kpss_stat)
        results["kpss_pvalue"] = float(kpss_p)
    except Exception as exc:
        logger.warning(f"KPSS test failed: {exc}")

    return results


def _apply_response_transform(
    y_raw         : np.ndarray,
    resp_transform: str,
    boxcox_lambda : Optional[float] = None,
) -> np.ndarray:
    """
    Apply the configured response transform to a raw response vector.
    For boxcox, ``boxcox_lambda`` must be the lambda already estimated from
    the primary product (passed in from the flat-case computation).
    """
    if resp_transform == "log1p":
        return np.log1p(np.maximum(y_raw, 0.0))
    elif resp_transform == "sqrt":
        return np.sqrt(np.maximum(y_raw, 0.0))
    elif resp_transform == "boxcox":
        from scipy.special import boxcox as _bc
        if boxcox_lambda is None or abs(boxcox_lambda) < 1e-10:
            return np.log(np.maximum(y_raw, 1e-6))
        return np.maximum((np.maximum(y_raw, 1e-6) ** boxcox_lambda - 1.0) / boxcox_lambda, -1e6)
    elif resp_transform == "identity":
        return y_raw.copy()
    else:
        return np.log1p(np.maximum(y_raw, 0.0))


def inverse_response_transform(
    y_transformed : np.ndarray,
    prep          : dict,
) -> np.ndarray:
    """
    Converts from transformed response space back to original scale.

    Uses the response_transform metadata stored in prep by ingest_and_preprocess.
    Supports: log1p, sqrt, boxcox, identity.

    Parameters
    ----------
    y_transformed : array in the transformed space (after un-standardising)
    prep          : prep dict containing response_transform and boxcox_lambda

    Returns
    -------
    y_original : array in original scale
    """
    rt = prep.get("response_transform", "log1p")

    if rt == "log1p":
        return np.expm1(y_transformed)
    elif rt == "sqrt":
        return np.maximum(y_transformed, 0.0) ** 2
    elif rt == "boxcox":
        lam = prep.get("boxcox_lambda")
        if lam is None or abs(lam) < 1e-10:
            return np.exp(y_transformed)
        return np.maximum((y_transformed * lam + 1.0) ** (1.0 / lam), 0.0)
    elif rt == "identity":
        return y_transformed
    else:
        # Fallback to expm1
        return np.expm1(y_transformed)


def build_fourier_features(T: int, period: float, order: int) -> np.ndarray:
    """Builds a (T, 2*order) matrix of sin/cos Fourier features."""
    t     = np.arange(T, dtype=float)
    feats = []
    for k in range(1, order + 1):
        feats.append(np.sin(2.0 * np.pi * k * t / period))
        feats.append(np.cos(2.0 * np.pi * k * t / period))
    return np.column_stack(feats)


# ─────────────────────────────────────────────────────────────────────────────
# Channel discovery  (new — used by main.py for the interactive prompt)
# ─────────────────────────────────────────────────────────────────────────────

def discover_channels(
    df          : pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Scans df and groups columns by channel name.

    Returns
    -------
    channel_options : dict  {channel_name → {metric_label → column_name}}

    Example
    -------
    {
        "google"       : {"Spend": "spends_google",
                          "Impressions": "media_impressions_google",
                          "Clicks": "media_clicks_google"},
        "clara_google" : {"Spend": "spends_clara_google",
                          "Impressions": "media_impressions_clara_google"},
        "facebook"     : {"Spend": "spends_facebook"},
    }

    Naming conventions supported
    ─────────────────────────────
    Spend       : spends_<channel>
    Impressions : media_impressions_<channel>  OR  imps_<channel>  OR  impressions_<channel>
    Clicks      : media_clicks_<channel>       OR  clicks_<channel> OR  clks_<channel>
    Views       : views_<channel>
    Reach       : reach_<channel>

    The channel name is everything after the prefix, so
    "spends_clara_google" → channel "clara_google".

    Strategy
    --------
    Walk every column; if it starts with a known prefix (longest-match wins),
    strip the prefix to get the channel name and record the (label → column)
    mapping.  Longer prefixes are listed first in METRIC_PREFIXES to ensure
    "media_impressions_" is matched before the shorter "impressions_".
    """
    exclude = set(exclude_cols or [])
    channel_options: Dict[str, Dict[str, str]] = {}

    for col in df.columns:
        if col in exclude:
            continue
        for prefix in METRIC_PREFIXES:
            if col.startswith(prefix):
                channel = col[len(prefix):]
                label   = PREFIX_LABELS.get(prefix, prefix.rstrip("_").title())
                channel_options.setdefault(channel, {})
                if label not in channel_options[channel]:       # first prefix wins per label
                    channel_options[channel][label] = col
                break

    return channel_options


# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion function
# ─────────────────────────────────────────────────────────────────────────────

def _auto_pivot_long_to_wide(
    df               : pd.DataFrame,
    date_col         : str,
    product_col      : str,
    long_response_col: str,
    long_media_cols  : Dict[str, str],
    schema,
    long_spend_cols  : Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Converts long-format data (one row per date × product) to wide format
    (one row per date, all products side by side).

    Triggered automatically when data.input_format = 'long' in the YAML.
    No need to run prep_hier_data.py separately.

    Handles TWO types of columns:
      1. Media (modelled) columns   — from long_media_cols
         e.g. impression or spend columns used as model inputs
      2. Spend (ROI only) columns   — from long_spend_cols  [optional]
         Required when long_media_cols points to impressions/clicks.
         These are pivoted into the wide CSV for ROI calculation but are
         NOT used as model inputs themselves.

    Column name resolution
    ──────────────────────
    Target column names come from the YAML schema (schema.media_cols and
    schema.product_response_map), so the wide CSV produced here matches
    exactly what the pipeline expects.

    Parameters
    ----------
    df                : long-format DataFrame
    date_col          : date column name (shared across products)
    product_col       : column whose values identify products (e.g. 'Product')
    long_response_col : single response column in long format (e.g. 'revenue_CC')
    long_media_cols   : {channel_label → source_col_name}
                        e.g. {"Google": "media_impressions_CC_Google"}
    schema            : DataSchema — provides target wide-format column names
    long_spend_cols   : {channel_label → source_spend_col_name}  [optional]
                        e.g. {"Google": "spends_CC_Google"}
                        Only needed when modelling non-spend metrics.

    Returns
    -------
    Wide-format DataFrame: one row per date, one column per product×variable.
    """
    if product_col not in df.columns:
        raise ValueError(
            f"[AUTO-PIVOT] product_col='{product_col}' not found in file. "
            f"Available columns: {list(df.columns)}"
        )

    products   = sorted(df[product_col].unique())
    n_dates    = df[date_col].nunique()
    logger.info(
        f"  [AUTO-PIVOT] long → wide | products={products} | "
        f"T={n_dates} unique dates | source media cols={list(long_media_cols.values())}"
    )

    # Build per-product rename maps: {source_col → target_col}
    # using schema for authoritative target names
    prod_rename: Dict[str, Dict[str, str]] = {p: {} for p in products}

    # Response columns — from schema.product_response_map
    if schema is not None and hasattr(schema, "product_response_map"):
        for prod, resp_col in schema.product_response_map.items():
            if prod in prod_rename and long_response_col:
                prod_rename[prod][long_response_col] = resp_col
    elif long_response_col:
        # Fallback: response_{product}
        for prod in products:
            prod_rename[prod][long_response_col] = f"response_{prod}"

    # Media columns (modelled) — from schema.media_cols matched on (product, channel)
    if schema is not None and hasattr(schema, "media_cols"):
        for mc in schema.media_cols:
            if mc.product in prod_rename:
                src = long_media_cols.get(mc.channel)
                if src:
                    prod_rename[mc.product][src] = mc.column
    else:
        # Fallback: replace source column prefix with _{product}
        for prod in products:
            for ch_label, src_col in long_media_cols.items():
                prod_rename[prod][src_col] = f"{src_col.split('_')[0]}_{prod}_{ch_label}"

    # Spend columns (ROI only) — pivoted into wide format but NOT used as model inputs.
    # Required when long_media_cols points to impressions/clicks rather than spend.
    # Allows compute_return_index() to report deals-per-£1000 even when the model
    # was trained on impressions.
    _lsc = long_spend_cols or {}
    if _lsc:
        if schema is not None and hasattr(schema, "media_cols"):
            for mc in schema.media_cols:
                if mc.product in prod_rename and mc.channel in _lsc:
                    spend_src = _lsc[mc.channel]
                    # Target name: use mc.spend_column if set in schema, else spends_{product}_{channel}
                    spend_tgt = mc.spend_column or f"spends_{mc.product}_{mc.channel}"
                    # Skip if this source col is already mapped (channel models spend directly)
                    if spend_src not in prod_rename[mc.product]:
                        prod_rename[mc.product][spend_src] = spend_tgt
        else:
            for prod in products:
                for ch_label, src_col in _lsc.items():
                    if src_col not in prod_rename[prod]:
                        prod_rename[prod][src_col] = f"spends_{prod}_{ch_label}"
        logger.info(f"  [AUTO-PIVOT] Spend (ROI) cols added: {list(_lsc.values())}")

    # Pivot: extract each product's rows, rename, join on date
    parts = []
    for prod in products:
        sub    = df[df[product_col] == prod].copy()
        rmap   = {k: v for k, v in prod_rename[prod].items() if k in sub.columns}
        keep   = [date_col] + list(rmap.keys())
        sub    = sub[keep].rename(columns=rmap).set_index(date_col)
        parts.append(sub)
        logger.debug(f"  [AUTO-PIVOT] {prod}: {len(sub)} rows, cols={list(sub.columns)}")

    wide = pd.concat(parts, axis=1).reset_index().rename(columns={"index": date_col})
    wide = wide.sort_values(date_col).reset_index(drop=True)

    logger.info(
        f"  [AUTO-PIVOT] Done → {wide.shape[0]} rows × {wide.shape[1]} cols"
    )
    logger.info(f"  [AUTO-PIVOT] Wide columns: {list(wide.columns)}")
    return wide


def ingest_and_preprocess(
    data_cfg             : DataConfig,
    model_cfg            : ModelConfig,
    channel_variable_map : Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Loads CSV, validates, scales, builds Fourier features, and returns a
    'prep' dict consumed by every model builder.

    Parameters
    ----------
    channel_variable_map : optional {channel_name → chosen_column}
        e.g. {"google": "clicks_google", "tv": "spends_tv"}
        When provided, this takes priority over data_cfg.media_cols and
        data_cfg.spend_prefix.  When None, the original behaviour is used.

    Returns
    -------
    prep : dict with keys:
        df, spend_cols, y_raw, y_transformed, y_scaled, y_mu, y_std,
        X_media_raw, X_media_scaled, X_fourier, X_controls,
        T, C, train_idx, dates, data_cfg, channel_variable_map
    """
    logger.info(f"Loading data from: {data_cfg.csv_path}")

    # ── File loading — supports both .csv and .xlsx ───────────
    _path_lower = str(data_cfg.csv_path).lower()
    if _path_lower.endswith(".xlsx") or _path_lower.endswith(".xls"):
        # engine must be specified explicitly — pandas cannot auto-detect xlsx format
        _engine = "xlrd" if _path_lower.endswith(".xls") else "openpyxl"
        try:
            df = pd.read_excel(data_cfg.csv_path, engine=_engine)
        except ImportError:
            raise ImportError(
                f"Reading '{data_cfg.csv_path}' requires the '{_engine}' package.\n"
                f"Install it with:  pip install {_engine}"
            )
        logger.info(f"  Loaded Excel file ({df.shape[0]} rows × {df.shape[1]} cols)")
    else:
        df = pd.read_csv(data_cfg.csv_path)

    # ── Auto-pivot: long → wide format ───────────────────────
    # Triggered when data.input_format='long' in the YAML.
    # Converts stacked rows (one per date×product) into the wide format
    # the pipeline expects (one row per date, separate columns per product).
    if getattr(data_cfg, "input_format", "wide").lower() == "long":
        _schema_for_pivot = kwargs.get("schema")
        df = _auto_pivot_long_to_wide(
            df               = df,
            date_col         = data_cfg.date_col,
            product_col      = getattr(data_cfg, "product_col", "Product"),
            long_response_col= getattr(data_cfg, "long_response_col", ""),
            long_media_cols  = getattr(data_cfg, "long_media_cols", {}) or {},
            schema           = _schema_for_pivot,
            long_spend_cols  = getattr(data_cfg, "long_spend_cols", {}) or {},
        )

    # ── Date parsing ──────────────────────────────────────────
    if data_cfg.date_col not in df.columns:
        raise ValueError(f"date_col '{data_cfg.date_col}' not found. Columns: {df.columns.tolist()}")
    df[data_cfg.date_col] = pd.to_datetime(df[data_cfg.date_col], dayfirst=data_cfg.dayfirst)
    df = df.sort_values(data_cfg.date_col).reset_index(drop=True)

    if data_cfg.min_date:
        df = df[df[data_cfg.date_col] >= pd.to_datetime(data_cfg.min_date)]
    if data_cfg.max_date:
        df = df[df[data_cfg.date_col] <= pd.to_datetime(data_cfg.max_date)]
    df = df.reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(f"Only {len(df)} rows after date filter — too few for MMM.")

    logger.info(f"  Rows after filter: {len(df)}")
    logger.info(f"  Date range: {df[data_cfg.date_col].min()} -> {df[data_cfg.date_col].max()}")

    # ── Response column ───────────────────────────────────────
    if data_cfg.response_col not in df.columns:
        raise ValueError(f"response_col '{data_cfg.response_col}' not found.")
    y_raw = df[data_cfg.response_col].values.astype(float)

    # ── Response sanitization ─────────────────────────────────
    # Avoid distorting the likelihood while still keeping transforms valid.
    # - log1p/sqrt require non-negative values.
    # - boxcox requires strictly positive values.
    resp_policy = getattr(model_cfg, "negative_response_policy", "auto")
    if np.any(y_raw <= 0):
        if resp_policy == "clip_to_zero":
            y_raw = np.maximum(y_raw, 0.0)
            logger.warning("Response has <=0 values. Clipped negatives to 0.")
        elif resp_policy == "clip_to_epsilon":
            y_raw = np.clip(y_raw, 1e-6, None)
            logger.warning("Response has <=0 values. Clipped to 1e-6.")
        else:
            # auto
            if getattr(model_cfg, "response_transform", "log1p").strip().lower() in ("log1p", "sqrt"):
                y_raw = np.maximum(y_raw, 0.0)
                logger.warning("Response has <=0 values. (auto) Clipped to 0 for log1p/sqrt.")
            elif getattr(model_cfg, "response_transform", "log1p").strip().lower() == "boxcox":
                y_raw = np.clip(y_raw, 1e-6, None)
                logger.warning("Response has <=0 values. (auto) Clipped to 1e-6 for boxcox.")
            else:
                y_raw = np.maximum(y_raw, 0.0)
                logger.warning("Response has <=0 values. (auto) Clipped to 0.")

    # ── Response transform (configurable) ─────────────────────
    # Supported: log1p (default), sqrt, boxcox, identity
    resp_transform = getattr(model_cfg, "response_transform", "log1p").strip().lower()
    boxcox_lambda  = None   # only used for boxcox

    if resp_transform == "log1p":
        y_transformed = np.log1p(y_raw)
        inv_label = "expm1"
    elif resp_transform == "sqrt":
        y_transformed = np.sqrt(y_raw)
        inv_label = "square"
    elif resp_transform == "boxcox":
        from scipy.stats import boxcox as _boxcox
        y_transformed, boxcox_lambda = _boxcox(y_raw)
        inv_label = "inv_boxcox"
        logger.info(f"  Box-Cox lambda = {boxcox_lambda:.4f}")
    elif resp_transform == "identity":
        y_transformed = y_raw.copy()
        inv_label = "identity"
    else:
        logger.warning(f"  Unknown response_transform '{resp_transform}' — falling back to log1p.")
        resp_transform = "log1p"
        y_transformed = np.log1p(y_raw)
        inv_label = "expm1"
    # ── Data quality validation ───────────────────────────────────
    if getattr(model_cfg, "enable_outlier_detection", False):
        outliers_iqr = detect_outliers_iqr(y_transformed)
        outliers_z  = detect_outliers_robust_zscore(y_transformed)
        n_outliers = int(np.sum(outliers_iqr | outliers_z))
        if n_outliers > 0:
            logger.warning(
                f"[DATA_PREP] Detected {n_outliers} outlier(s) in the response after transform. "
                "Review data quality or enable robust handling."
            )
            if getattr(model_cfg, "enable_strict_validation", False):
                raise ValueError(
                    "Strict validation enabled and response outliers were detected. "
                    "Clean the data or disable enable_strict_validation."
                )

    if getattr(model_cfg, "enable_stationarity_test", False):
        stationarity = run_stationarity_tests(y_transformed)
        if stationarity:
            logger.info(
                "[DATA_PREP] Stationarity checks: "
                + ", ".join(f"{k}={v:.4f}" for k, v in stationarity.items())
            )
            adf_p = stationarity.get("adf_pvalue")
            kpss_p = stationarity.get("kpss_pvalue")
            if getattr(model_cfg, "enable_strict_validation", False):
                if adf_p is not None and adf_p > 0.05:
                    raise ValueError("Strict validation failed: ADF test indicates non-stationary response.")
                if kpss_p is not None and kpss_p < 0.05:
                    raise ValueError("Strict validation failed: KPSS test indicates non-stationary response.")
    y_scaled, y_mu, y_std = safe_scale(y_transformed)
    logger.info(f"  Response transform: {resp_transform} | min={y_raw.min():.0f}, max={y_raw.max():.0f}, mean={y_raw.mean():.0f}")

    # ── Media columns — three-level priority ──────────────────
    #
    #  Priority 1 — channel_variable_map  (interactive per-channel selection)
    #  Priority 2 — data_cfg.media_cols   (explicit programmatic list)
    #  Priority 3 — spend_prefix auto-discovery (original default)
    #
    if channel_variable_map:
        # In multi-product mode the channel_variable_map only has one column per
        # channel (the last product wins in the dict comprehension in schema.py).
        # We must instead use ALL product×channel columns from the schema so that
        # every product's spend is included.  For single-product data the original
        # one-column-per-channel behaviour is preserved.
        _schema_arg = kwargs.get("schema")
        if (
            _schema_arg is not None
            and getattr(_schema_arg, "is_multi_product", False)
            and getattr(_schema_arg, "product_response_map", {})
        ):
            spend_cols = [c for c in _schema_arg.all_media_column_names if c in df.columns]
            if not spend_cols:
                raise ValueError("No multi-product media columns found in CSV.")
            logger.info("  Multi-product media columns (all products x channels):")
            for col in spend_cols:
                logger.info(f"    {col}")
        else:
            spend_cols = [channel_variable_map[ch] for ch in sorted(channel_variable_map)]
            for col in spend_cols:
                if col not in df.columns:
                    raise ValueError(f"Selected column '{col}' not found in CSV.")
            logger.info("  Per-channel variable selections:")
            for ch, col in sorted(channel_variable_map.items()):
                logger.info(f"    {ch:<30s} -> {col}")

    elif data_cfg.media_cols:
        valid   = [c for c in data_cfg.media_cols if c in df.columns]
        missing = [c for c in data_cfg.media_cols if c not in df.columns]
        if missing:
            logger.warning(f"Requested media cols missing and skipped: {missing}")
        spend_cols = sorted(valid)
        if not spend_cols:
            raise ValueError("No valid media_cols found in dataset.")

    else:
        spend_cols = sorted([c for c in df.columns if c.startswith(data_cfg.spend_prefix)])
        if not spend_cols:
            raise ValueError(f"No columns with prefix '{data_cfg.spend_prefix}' found.")

    logger.info(f"  Media columns selected: {spend_cols}")
    # IMPORTANT: The modeled media columns must match the user's choice/config.
    # If a parallel spend column exists, it is tracked separately via spend_raw_cols
    # for ROI/return-index reporting - we do NOT silently swap the modeled variable.

    X_media_raw    = df[spend_cols].values.astype(float)

    # ── Zero-spend channel detection ──────────────────────────
    # Channels with all-zero values contribute nothing to the model and can
    # destabilise priors.  Warn loudly so users can drop or replace them.
    for j, col in enumerate(spend_cols):
        col_max = np.max(np.abs(X_media_raw[:, j]))
        if col_max < 1e-8:
            logger.warning(
                f"[DATA_PREP] Channel '{col}' (index {j}) has all-zero (or near-zero) "
                f"values (max={col_max:.2e}). Its beta will only reflect the prior — "
                f"consider dropping this channel or checking your data."
            )

    X_media_scaled = np.column_stack([
        X_media_raw[:, j] / (np.max(X_media_raw[:, j]) + 1e-8)
        for j in range(len(spend_cols))
    ])

    # ── Collinearity diagnostics ───────────────────────────────
    # FIX (Issue 6): In multi-product mode the flat X_media_raw has C*P columns
    # (e.g. 8 for 2 products × 4 channels).  Google_Cashback and Google_Indulge
    # are naturally correlated — they follow the same channel trends — and would
    # trigger a false collinearity warning.  We compute the check on the unique
    # C channels only, by averaging across products when P>1.
    # At this point we don't yet know P, so we use the schema to detect it.
    schema = kwargs.get("schema")
    _is_multi = (
        schema is not None
        and getattr(schema, "is_multi_product", False)
        and getattr(schema, "product_response_map", {})
    )
    col_max_offdiag_abs = np.nan
    try:
        if _is_multi and schema is not None:
            # Build a (T, C_uniq) matrix by averaging same-channel columns across products
            _uniq_chs  = schema.unique_channel_names   # sorted unique channel names
            _ch_to_cols: Dict[str, List[int]] = {ch: [] for ch in _uniq_chs}
            for cr in schema.media_cols:
                if cr.column in spend_cols:
                    j = list(spend_cols).index(cr.column)
                    _ch_to_cols.setdefault(cr.channel, []).append(j)
            _X_uniq = np.column_stack([
                X_media_raw[:, _ch_to_cols[ch]].mean(axis=1)
                if _ch_to_cols.get(ch) else np.zeros(X_media_raw.shape[0])
                for ch in _uniq_chs
            ])  # (T, C_uniq)
            _X_for_corr = _X_uniq
        else:
            _X_for_corr = X_media_raw

        if _X_for_corr.shape[1] >= 2:
            corr = np.corrcoef(_X_for_corr.T)
            # Guard: corrcoef can produce NaNs if a column is constant.
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            # Compute max absolute off-diagonal correlation.
            offdiag = corr - np.eye(corr.shape[0])
            col_max_offdiag_abs = float(np.max(np.abs(offdiag)))
            thr = float(getattr(model_cfg, "collinearity_max_offdiag_abs", 0.85))
            if col_max_offdiag_abs > thr:
                logger.warning(
                    f"High multicollinearity detected: max|corr_offdiag|={col_max_offdiag_abs:.3f} > {thr:.3f}."
                )
                if getattr(model_cfg, "enable_collinearity_check", False):
                    if getattr(model_cfg, "enable_strict_validation", False):
                        raise ValueError(
                            "Strict validation failed: multicollinearity detected. "
                            "Reduce correlated channels or disable strict validation."
                        )
    except Exception as e:
        logger.warning(f"Collinearity diagnostics failed: {e}")

    # ── Control covariates ────────────────────────────────────
    X_controls      = None
    active_controls = []
    if data_cfg.control_cols and model_cfg.use_controls:
        valid   = [c for c in data_cfg.control_cols if c in df.columns]
        missing = set(data_cfg.control_cols) - set(valid)
        if missing:
            logger.warning(f"Control cols not found, skipping: {missing}")
        if valid:
            raw_Z           = df[valid].values.astype(float)
            X_controls      = scale_columns(raw_Z)
            active_controls = valid
            logger.info(f"  Control covariates: {valid}")

    # ── Outlier date indicators ───────────────────────────────
    # Each listed date gets a binary 0/1 indicator column added as a
    # model regressor.  These are included unconditionally (regardless of
    # use_controls) to isolate known anomaly spikes.
    _outlier_dates_list = list(getattr(data_cfg, "outlier_dates", []) or [])
    X_outlier_indicators   = None
    outlier_indicator_cols : List[str] = []
    if _outlier_dates_list:
        _date_series = pd.to_datetime(df[data_cfg.date_col]).dt.normalize()
        _outlier_arrays : List[np.ndarray] = []
        for _d in _outlier_dates_list:
            try:
                _od = pd.to_datetime(_d).normalize()
                _ind = (_date_series == _od).astype(float).values
                if _ind.sum() > 0:
                    _outlier_arrays.append(_ind)
                    outlier_indicator_cols.append(f"outlier_{_d}")
                    logger.info(f"  Outlier indicator: {_d} ({int(_ind.sum())} period(s))")
                else:
                    logger.warning(f"[DATA_PREP] Outlier date '{_d}' not found in data — skipped.")
            except Exception as _e:
                logger.warning(f"[DATA_PREP] Could not parse outlier date '{_d}': {_e}")
        if _outlier_arrays:
            X_outlier_indicators = np.column_stack(_outlier_arrays)
            logger.info(f"  {len(outlier_indicator_cols)} outlier indicator(s) built.")

    # ── Fourier seasonality ───────────────────────────────────
    T = len(df)

    # Determine dataset frequency: schema > DataConfig > default weekly.
    _schema_freq = getattr(schema, "frequency", None) if schema is not None else None
    _cfg_freq    = getattr(data_cfg, "frequency", "weekly")
    frequency    = (_schema_freq or _cfg_freq or "weekly").strip().lower()
    period       = FOURIER_PERIODS.get(frequency, 52.18)

    # Derive minimum number of observations needed for one full seasonal cycle.
    _min_obs_per_cycle = {
        "weekly" : 52,
        "daily"  : 365,
        "monthly": 12,
    }.get(frequency, 52)

    requested_order = model_cfg.fourier_order
    fourier_order   = min(requested_order, max(1, T // _min_obs_per_cycle))
    if fourier_order < requested_order:
        logger.warning(
            f"[DATA_PREP] fourier_order capped: requested {requested_order} but "
            f"T={T} {frequency} periods only supports {fourier_order} "
            f"(need ~{_min_obs_per_cycle} observations per order). "
            f"Using fourier_order={fourier_order}."
        )

    X_fourier = build_fourier_features(T, period, fourier_order)
    logger.info(f"  Fourier seasonality: frequency={frequency}, period={period:.2f}, order={fourier_order}")

    # ── Day-of-week seasonality (daily data only) ─────────────
    # Adds 6 Fourier features (3 harmonics of 7-day cycle) to capture
    # within-week patterns (Mon–Sun). Only built when daily + use_dow_effects=True.
    X_fourier_dow = None
    if frequency == "daily" and getattr(model_cfg, "use_dow_effects", None):
        X_fourier_dow = build_fourier_features(T, 7.0, 3)
        logger.info("  DOW seasonality: period=7 days, order=3 (6 features, Mon-Sun pattern)")

    # ── Normalised time index ─────────────────────────────────
    t_norm = np.linspace(0.0, 1.0, T)
    dates  = df[data_cfg.date_col].values

    # ── Optional holdout split ────────────────────────────────
    # holdout_periods > 0 → hold out the last N rows for out-of-sample evaluation.
    # train_idx is used throughout the model; test_idx is carried in prep for
    # post-fit evaluation but is NOT used during sampling.
    holdout_periods = int(getattr(data_cfg, "holdout_periods", 0))
    min_train       = max(20, _min_obs_per_cycle)   # at least one full cycle in train
    if holdout_periods > 0 and (T - holdout_periods) >= min_train:
        train_end = T - holdout_periods
        train_idx = np.arange(train_end)
        test_idx  = np.arange(train_end, T)
        logger.info(
            f"  Holdout split: train=[0, {train_end}) | "
            f"test=[{train_end}, {T}) ({holdout_periods} periods held out)"
        )
    else:
        if holdout_periods > 0:
            logger.warning(
                f"[DATA_PREP] holdout_periods={holdout_periods} would leave fewer than "
                f"{min_train} training observations — ignoring holdout."
            )
        train_idx = np.arange(T)
        test_idx  = np.array([], dtype=int)

    logger.info(f"  T={T} {frequency} | C={len(spend_cols)} channels | Fourier order={fourier_order}")

    # FIX-2: compute metric type and parallel spend column for each selected channel.
    metric_types   = [_metric_type_for_col(c) for c in spend_cols]

    # If schema provides spend_column mapping, use it for ROI; otherwise fall back to prefix inference.
    # (schema was already extracted above in the Fourier section)
    if schema is not None and getattr(schema, "media_cols", None):
        spend_lookup = {c.column: (c.spend_column or _spend_col_for(c.column)) for c in schema.media_cols}
        spend_raw_cols = [spend_lookup.get(c, _spend_col_for(c)) for c in spend_cols]
    else:
        spend_raw_cols = [_spend_col_for(c) for c in spend_cols]

    # ── Channel family index ──────────────────────────────────
    # Build per-channel family membership arrays for hierarchical pooling.
    # family_names : sorted list of unique family names (F families)
    # family_idx   : array of length C mapping each channel to its family index
    family_names : List[str] = ["generic"]
    family_idx   : np.ndarray = np.zeros(len(spend_cols), dtype=int)

    if schema is not None and getattr(schema, "channel_family_map", None):
        cfm = schema.channel_family_map  # {col -> family_name}
        raw_families = [cfm.get(col, "generic") for col in spend_cols]
        family_names = sorted(set(raw_families))
        fam_to_idx   = {f: i for i, f in enumerate(family_names)}
        family_idx   = np.array([fam_to_idx[f] for f in raw_families], dtype=int)
        logger.info(f"  Channel families: {family_names}")
        for col, fam in zip(spend_cols, raw_families):
            logger.info(f"    {col:<35s} -> {fam}")
    else:
        logger.info("  No channel families defined — all channels assigned to 'generic'")

    F = len(family_names)

    # ── Multi-product detection ───────────────────────────────
    # Activated when the schema tags ColumnRoles with a `product` field
    # AND a product_response_map is provided.
    #
    # Data layout expected:
    #   Wide format — one row per time period.
    #   Media  : separate columns per product×channel pair.
    #   Response: separate column per product.
    #
    # When P>1 the prep dict gains:
    #   P                  — number of products
    #   product_names      — sorted list of product names
    #   channel_names_unique — C unique channel names (same across products)
    #   X_media_scaled     — shape (T, P, C)   [overrides flat (T, C)]
    #   X_media_raw        — shape (T, P, C)
    #   y_scaled           — shape (T, P)      [per-product z-scored]
    #   y_mu_p / y_std_p   — per-product normalisation params (P,)
    #   y_raw_p            — shape (T, P) raw responses

    P              = 1
    product_names  : List[str] = []
    channel_names_unique : List[str] = spend_cols  # C unique channels
    y_raw_by_product : Optional[np.ndarray] = None  # (T, P) set in multi-product block

    y_scaled_out   = y_scaled        # (T,) for P=1, (T, P) for P>1
    y_mu_out       = y_mu
    y_std_out      = y_std
    X_media_raw_out    = X_media_raw       # (T,C) or (T,P,C)
    X_media_scaled_out = X_media_scaled    # (T,C) or (T,P,C)

    if schema is not None and schema.is_multi_product and schema.product_response_map:
        product_names       = schema.products          # sorted list
        P                   = len(product_names)
        channel_names_unique = schema.unique_channel_names  # unique C channels

        C_uniq = len(channel_names_unique)
        ch_name_to_idx = {ch: i for i, ch in enumerate(channel_names_unique)}

        # Build (T, P, C) media arrays
        _raw_3d    = np.zeros((T, P, C_uniq), dtype=float)
        _scaled_3d = np.zeros((T, P, C_uniq), dtype=float)

        for col_role in schema.media_cols:
            if col_role.column not in df.columns:
                continue
            p_idx = product_names.index(col_role.product)
            c_idx = ch_name_to_idx.get(col_role.channel, -1)
            if c_idx < 0:
                continue
            raw_vals = df[col_role.column].values.astype(float)
            _raw_3d[:, p_idx, c_idx] = raw_vals
            max_val = np.max(raw_vals) + 1e-8
            _scaled_3d[:, p_idx, c_idx] = raw_vals / max_val

        # Build (T, P) per-product response arrays with independent z-scoring
        _y_raw_3d = np.zeros((T, P), dtype=float)
        _y_sc_3d  = np.zeros((T, P), dtype=float)
        _y_mu_p   = np.zeros(P, dtype=float)
        _y_std_p  = np.zeros(P, dtype=float)

        for p_idx, pname in enumerate(product_names):
            resp_col = schema.product_response_map[pname]
            if resp_col not in df.columns:
                raise ValueError(
                    f"Response column '{resp_col}' for product '{pname}' not found in CSV."
                )
            y_p_raw = df[resp_col].values.astype(float)
            # Each product gets its own BoxCox lambda to handle differing distributions.
            # For all other transforms the per-product lambda is the shared one (None or fitted).
            _p_boxcox_lam = boxcox_lambda
            if resp_transform == "boxcox":
                from scipy.stats import boxcox as _boxcox
                y_p_safe = np.clip(y_p_raw, 1e-6, None)
                try:
                    _, _p_boxcox_lam = _boxcox(y_p_safe)
                except Exception:
                    _p_boxcox_lam = boxcox_lambda  # fall back to primary product's lambda
            y_p_trans = _apply_response_transform(y_p_raw, resp_transform, _p_boxcox_lam)
            y_p_sc, y_p_mu, y_p_std = safe_scale(y_p_trans)
            _y_raw_3d[:, p_idx] = y_p_raw
            _y_sc_3d[:, p_idx]  = y_p_sc
            _y_mu_p[p_idx]      = y_p_mu
            _y_std_p[p_idx]     = y_p_std

        # Override flat arrays with multi-product versions
        spend_cols           = channel_names_unique    # C unique channel names
        X_media_raw_out      = _raw_3d                 # (T, P, C)
        X_media_scaled_out   = _scaled_3d              # (T, P, C)
        y_scaled_out         = _y_sc_3d                # (T, P)
        y_mu_out             = _y_mu_p                 # (P,)
        y_std_out            = _y_std_p                # (P,)
        y_raw_by_product     = _y_raw_3d               # (T, P) raw responses per product
        # Update metric_types and spend_raw_cols to match C unique channels (not C*P columns)
        # Use schema metric_type field (reliable) rather than prefix-matching on bare channel names
        if schema is not None and getattr(schema, "media_cols", None):
            _MT_LABELS = {"spend": "Spend", "impressions": "Impressions",
                          "clicks": "Clicks", "grp": "GRP", "numeric": "Metric"}
            _mt_map = {mc.channel: mc.metric_type for mc in schema.media_cols}
            metric_types = [_MT_LABELS.get(_mt_map.get(ch, "spend"), "Spend")
                            for ch in channel_names_unique]
        else:
            metric_types = [_metric_type_for_col(ch) for ch in channel_names_unique]
        if schema is not None and getattr(schema, "media_cols", None):
            _sc_map = {mc.channel: (mc.spend_column or mc.column) for mc in schema.media_cols}
            spend_raw_cols = [_sc_map.get(ch, _spend_col_for(ch)) for ch in channel_names_unique]
        else:
            spend_raw_cols = [_spend_col_for(ch) for ch in channel_names_unique]

        # ── FIX (Bug 1): Rebuild family_idx for unique channels ──────────────
        # family_idx was built earlier from the flat spend_cols which had C*P
        # entries (e.g. 8 for 2 products × 4 channels).  Now that spend_cols
        # has been overridden to the C unique channel names (e.g. 4), we MUST
        # rebuild family_idx so len(family_idx) == C.  Without this fix,
        # model_builder's z_beta_c (shape C=4) and fam_idx_t (length 8) would
        # be incompatible → PyTensor shape mismatch crash.
        #
        # Strategy: map unique channel name → family using schema.media_cols,
        # which carries the 'family' field set in the YAML per column.
        if schema is not None and getattr(schema, "media_cols", None):
            # Build channel_name → family lookup from ColumnRole objects
            ch_name_to_family: Dict[str, str] = {}
            for cr in schema.media_cols:
                if cr.channel:
                    ch_name_to_family[cr.channel] = getattr(cr, "family", "generic") or "generic"
            uniq_raw_families = [
                ch_name_to_family.get(ch, "generic")
                for ch in channel_names_unique
            ]
            uniq_family_names = sorted(set(uniq_raw_families))
            uniq_fam_to_idx   = {f: i for i, f in enumerate(uniq_family_names)}
            family_idx        = np.array(
                [uniq_fam_to_idx[f] for f in uniq_raw_families], dtype=int
            )
            family_names = uniq_family_names
            F            = len(family_names)
            logger.info(
                f"  [FIX] Rebuilt family_idx for {C_uniq} unique channels: "
                + ", ".join(f"{ch}->{fam}" for ch, fam in zip(channel_names_unique, uniq_raw_families))
            )
        else:
            # No schema families — reset to single generic family of length C_uniq
            family_idx   = np.zeros(C_uniq, dtype=int)
            family_names = ["generic"]
            F            = 1

        logger.info(
            f"  Multi-product mode: P={P} products, C={C_uniq} channels, T={T} periods"
        )
        for pname in product_names:
            logger.info(f"    product '{pname}' -> response '{schema.product_response_map[pname]}'")
    else:
        if schema is not None and schema.is_multi_product and not schema.product_response_map:
            logger.warning(
                "Schema has multi-product media columns but no product_response_map — "
                "falling back to flat single-product model."
            )

    # ── Campaign columns (amplifiers, not additive) ──────────────
    # Campaigns multiply channel effects rather than having direct additive effects.
    # Processed similarly to media but stored separately for model building.
    X_campaign_raw      = None
    X_campaign_scaled   = None
    campaign_cols       = []
    campaign_info       = []

    if schema is not None and getattr(model_cfg, "use_campaigns", False):
        camp_cols = [c.column for c in schema.campaign_cols if c.column in df.columns]
        if camp_cols:
            raw_Camp = df[camp_cols].values.astype(float)
            X_campaign_raw    = raw_Camp
            X_campaign_scaled = np.column_stack([
                raw_Camp[:, j] / (np.max(raw_Camp[:, j]) + 1e-8)
                for j in range(len(camp_cols))
            ])
            campaign_cols = camp_cols
            campaign_info = [
                {"column": c.column, "channel": c.channel, "family": getattr(c, "family", "generic")}
                for c in schema.campaign_cols if c.column in df.columns
            ]
            logger.info(f"  Campaign variables: {campaign_cols}")

    # ── Base variables (price, promo, distribution) ───────────
    # Loaded from schema if provided; base variables use linear priors
    # (can be positive or negative) and are z-scored by default.
    X_base       = None
    base_cols    = []
    base_info    = []    # list of dicts with column metadata
    if schema is not None:
        from config import ColumnRole
        bcols = [c.column for c in schema.base_cols if c.column in df.columns]
        if bcols:
            raw_B  = df[bcols].values.astype(float)
            X_base = scale_columns(raw_B)
            base_cols  = bcols
            base_info  = [
                {"column": c.column, "channel": c.channel, "channel_type": c.channel_type}
                for c in schema.base_cols if c.column in df.columns
            ]
            logger.info(f"  Base variables: {base_cols}")

    # ── Macro variables (GDP, CPI, unemployment) ──────────────
    X_macro      = None
    macro_cols   = []
    macro_info   = []
    if schema is not None:
        mcols = [c.column for c in schema.macro_cols if c.column in df.columns]
        if mcols:
            raw_M   = df[mcols].values.astype(float)
            X_macro = scale_columns(raw_M)
            macro_cols  = mcols
            macro_info  = [
                {"column": c.column, "channel": c.channel, "channel_type": c.channel_type}
                for c in schema.macro_cols if c.column in df.columns
            ]
            logger.info(f"  Macro variables: {macro_cols}")

    # ── Event dummies (holidays, launches) ────────────────────
    X_events     = None
    event_cols   = []
    if schema is not None:
        _event_specified = [c.column for c in schema.event_cols]
        _event_missing   = [c for c in _event_specified if c not in df.columns]
        if _event_missing:
            logger.warning(
                f"[DATA_PREP] Event column(s) listed in YAML not found in CSV — "
                f"they will be IGNORED: {_event_missing}\n"
                f"  Check spelling/case exactly matches the CSV header. "
                f"  CSV columns available: {list(df.columns)}"
            )
        ecols = [c.column for c in schema.event_cols if c.column in df.columns]
        if ecols:
            X_events   = df[ecols].values.astype(float)
            event_cols = ecols
            logger.info(f"  Event variables loaded: {event_cols}")
            for col in ecols:
                n_ones = int((X_events[:, ecols.index(col)] == 1).sum())
                logger.info(f"    {col}: {n_ones} event week(s) flagged out of {T}")

    # ── Flighting mask  (T, C) — 1.0=active, 0.0=dark ────────
    # Each media channel may supply an optional binary column in the CSV
    # (``flighting_col`` in schema) that marks which weeks the channel was
    # live.  A value of 0 zeroes out the channel's contribution for that
    # week after saturation.  Defaults to all-ones (always active) when
    # no flighting column is specified.
    flight_mask = np.ones((T, len(spend_cols)), dtype=float)
    if schema is not None:
        col_to_j = {col: j for j, col in enumerate(spend_cols)}
        for mc in schema.media_cols:
            if not mc.flighting_col:
                continue
            if mc.flighting_col not in df.columns:
                logger.warning(
                    f"  [FLIGHTING] Column '{mc.flighting_col}' for channel "
                    f"'{mc.column}' not found in CSV — defaulting to always-on."
                )
                continue
            j = col_to_j.get(mc.column)
            if j is None:
                continue
            raw_mask = df[mc.flighting_col].values.astype(float)
            # Clip to [0, 1]; treat any non-zero value as "active"
            flight_mask[:, j] = np.clip(raw_mask, 0.0, 1.0)
            n_active = int((raw_mask > 0).sum())
            logger.info(
                f"  [FLIGHTING] {mc.column}: {n_active}/{T} weeks active "
                f"(from column '{mc.flighting_col}')"
            )
    has_flighting = bool(np.any(flight_mask < 1.0))

    # ── Adstock reset mask  (T, C) — 1.0 at the FIRST active week ────────────
    # after RESET_GAP_WEEKS or more consecutive dark weeks.
    #
    # This is passed to the scan-based adstock functions so the carry state
    # (sF and sS for two-timescale; s for geometric) is explicitly zeroed at
    # restart weeks.  This prevents ghost carry-over from a previous campaign
    # flight from artificially boosting the channel's signal when it restarts.
    #
    # Activeness is determined from BOTH actual spend (> eps) AND the explicit
    # flighting column, so the mask is correct even when one source is missing.
    #
    # Design matches the reference model's build_reset_mask_TPC(), reduced to
    # 2-D (T, C) because our architecture uses shared flighting per channel
    # rather than per product×channel.
    # ─────────────────────────────────────────────────────────────────────────
    _RESET_GAP = int(getattr(model_cfg, "adstock_reset_gap_weeks", 4))
    _C_rst     = len(spend_cols)
    reset_mask = np.zeros((T, _C_rst), dtype=float)

    # Derive (T, C) binary activeness from spend data
    _raw_for_reset = X_media_raw_out
    if _raw_for_reset.ndim == 3:
        # multi-product (T, P, C) → collapse products: active if ANY product spent
        _active_spend = _raw_for_reset.sum(axis=1) > 1e-12   # (T, C)
    else:
        _active_spend = _raw_for_reset > 1e-12               # (T, C)
    # Combine with explicit flighting column mask
    _active = _active_spend & (flight_mask > 0.5)            # (T, C) bool

    for _j in range(_C_rst):
        _zrun = 0
        for _t in range(T):
            if not _active[_t, _j]:
                _zrun += 1
            else:
                if _t > 0 and _zrun >= _RESET_GAP:
                    reset_mask[_t, _j] = 1.0
                _zrun = 0

    has_reset = bool(np.any(reset_mask > 0.0))
    _n_events = int(reset_mask.sum())
    logger.debug(
        f"  [RESET_MASK] {_n_events} restart event(s) across {_C_rst} channels "
        f"(gap threshold = {_RESET_GAP} weeks)"
    )

    return {
        "df"                  : df,
        "spend_cols"          : spend_cols,
        "y_raw"               : y_raw,
        "y_transformed"       : y_transformed,
        "y_scaled"            : y_scaled_out,     # (T,) flat OR (T,P) multi-product
        "y_mu"                : y_mu_out,         # scalar flat OR (P,) multi-product
        "y_std"               : y_std_out,        # scalar flat OR (P,) multi-product
        "X_media_raw"         : X_media_raw_out,  # (T,C) OR (T,P,C)
        "X_media_scaled"      : X_media_scaled_out,  # (T,C) OR (T,P,C)
        "X_fourier"           : X_fourier,
        "X_controls"          : X_controls,
        "active_controls"     : active_controls,
        # ── New variable groups ──────────────────────
        "X_base"              : X_base,
        "base_cols"           : base_cols,
        "base_info"           : base_info,
        "X_macro"             : X_macro,
        "macro_cols"          : macro_cols,
        "macro_info"          : macro_info,
        "X_events"            : X_events,
        "event_cols"          : event_cols,
        "X_campaign_raw"      : X_campaign_raw,
        "X_campaign_scaled"   : X_campaign_scaled,
        "campaign_cols"       : campaign_cols,
        "campaign_info"       : campaign_info,
        # ── Response transform metadata ──────────────
        "response_transform"  : resp_transform,
        "response_inv_label"  : inv_label,
        "boxcox_lambda"       : boxcox_lambda,
        # ── Dimensions ───────────────────────────────
        "t_norm"              : t_norm,
        "T"                   : T,
        "C"                   : len(spend_cols),
        "P"                   : P,                # 1 = flat, >1 = multi-product
        "product_names"       : product_names,    # [] for flat
        "channel_names_unique": channel_names_unique,
        # ── Channel family membership ─────────────────
        "F"                   : F,                # number of unique families
        "family_names"        : family_names,     # sorted list length F
        "family_idx"          : family_idx,       # (C,) int array, index into family_names
        # ── Holdout / train-test split ────────────────
        "train_idx"           : train_idx,
        "test_idx"            : test_idx,
        "holdout_periods"     : holdout_periods,
        # ── Other ────────────────────────────────────
        "dates"               : dates,
        "data_cfg"            : data_cfg,
        "channel_variable_map": channel_variable_map or {},
        "metric_types"        : metric_types,
        "spend_raw_cols"      : spend_raw_cols,
        "schema"              : schema,
        "y_raw_by_product"    : y_raw_by_product,  # (T, P) or None for single-product
        # ── Flighting + adstock reset ────────────────
        "flight_mask"         : flight_mask,          # (T, C) float array — 1=active, 0=dark
        "has_flighting"       : has_flighting,
        "reset_mask"          : reset_mask,           # (T, C) float array — 1.0 at restart weeks
        "has_reset"           : has_reset,            # True if any restart event exists
        "adstock_reset_gap_weeks" : _RESET_GAP,       # dark-period threshold (default 4)
        # ── Dataset frequency ─────────────────────────────────
        "frequency"           : frequency,   # "weekly" | "daily" | "monthly"
        "fourier_period"      : period,      # actual numeric period used
        # ── DOW and outlier features ──────────────────────────
        "X_fourier_dow"         : X_fourier_dow,           # (T, 6) or None
        "X_outlier_indicators"  : X_outlier_indicators,    # (T, N) or None
        "outlier_indicator_cols": outlier_indicator_cols,  # list of N col names
        # ── Robustness diagnostics for quality_flag gating ────
        "min_T_for_production"        : int(getattr(model_cfg, "min_T_for_production", 100)),
        "collinearity_max_offdiag_abs": col_max_offdiag_abs,
        "collinearity_threshold"      : float(getattr(model_cfg, "collinearity_max_offdiag_abs", 0.85)),
    }
