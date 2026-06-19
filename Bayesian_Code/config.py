# config.py
# ─────────────────────────────────────────────────────────────────────────────
# DataConfig, ModelConfig dataclasses + global constants
# ─────────────────────────────────────────────────────────────────────────────

import itertools
import logging
import yaml
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

GLOBAL_SEED = 42
# NOTE: np.random.seed() is intentionally NOT called at module level here.
# Setting a global NumPy seed on import is a side-effect that can corrupt
# reproducibility in any other library sharing the same process.
# The seed is passed explicitly to pm.sample(random_seed=GLOBAL_SEED) instead.

ADSTOCK_SAT_COMBOS: List[Tuple[str, str]] = list(itertools.product(
    ["geometric", "weibull"],
    ["softplus", "hill", "logistic", "exponential"],
))  # 8 combos

logger = logging.getLogger("MMM")


# ─────────────────────────────────────────────────────────────────────────────
# Channel family prior configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChannelFamilyConfig:
    """
    Domain prior hyperparameters for a channel family's two-timescale adstock.

    These govern how fast/slow the carry-over effect typically is for a
    group of related channels (e.g. all "search" channels share a group
    prior on their half-lives).  Individual channels within a family can
    still deviate — the family just sets the group mean.

    Parameters
    ----------
    name            : Family identifier ("search", "tv", etc.)
    hl_slow_median  : Prior median for the slow half-life in periods.
                      TV is long (~5 wks), search is short (~2.5 wks).
    hl_ratio_median : Prior median for slow_hl / fast_hl.  Must be > 1.
                      A ratio of 3 means fast decays 3x quicker than slow.
    hl_sigma        : Uncertainty (log-scale std) on the family-level mean.
                      Higher → families can vary more in their half-lives.
    """
    name            : str
    hl_slow_median  : float = 3.0
    hl_ratio_median : float = 3.0   # slow_hl / fast_hl  (always > 1)
    hl_sigma        : float = 0.5   # log-scale sigma for family hyperprior


# ─────────────────────────────────────────────────────────────────────────────
# Industry-calibrated FALLBACK defaults per family.
#
# ⚠️  DO NOT EDIT THESE when running via YAML config.
#     Override them in your YAML file under:
#
#       priors:
#         families:
#           search:
#             hl_slow_median: 3.0   # ← change this, not the line below
#
#     The YAML priors.families section is read in main.py and merged on top of
#     these defaults via ModelConfig.family_configs, so YAML always wins.
#     Only edit here if you are running in interactive (terminal) mode without a YAML.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FAMILY_CONFIGS: Dict[str, "ChannelFamilyConfig"] = {
    "search":       ChannelFamilyConfig("search",       hl_slow_median=2.5, hl_ratio_median=3.0, hl_sigma=0.4),
    "social":       ChannelFamilyConfig("social",       hl_slow_median=1.5, hl_ratio_median=2.5, hl_sigma=0.4),
    "tv":           ChannelFamilyConfig("tv",           hl_slow_median=5.0, hl_ratio_median=4.0, hl_sigma=0.5),
    "display":      ChannelFamilyConfig("display",      hl_slow_median=2.0, hl_ratio_median=2.5, hl_sigma=0.4),
    "ooh":          ChannelFamilyConfig("ooh",          hl_slow_median=6.0, hl_ratio_median=4.0, hl_sigma=0.5),
    "programmatic": ChannelFamilyConfig("programmatic", hl_slow_median=2.0, hl_ratio_median=2.0, hl_sigma=0.4),
    "email":        ChannelFamilyConfig("email",        hl_slow_median=1.0, hl_ratio_median=2.0, hl_sigma=0.4),
    "radio":        ChannelFamilyConfig("radio",        hl_slow_median=3.0, hl_ratio_median=3.0, hl_sigma=0.5),
    "video":        ChannelFamilyConfig("video",        hl_slow_median=3.5, hl_ratio_median=3.0, hl_sigma=0.5),
    "affiliate":    ChannelFamilyConfig("affiliate",    hl_slow_median=1.5, hl_ratio_median=2.0, hl_sigma=0.4),
    "generic":      ChannelFamilyConfig("generic",      hl_slow_median=3.0, hl_ratio_median=3.0, hl_sigma=0.5),
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel transform specification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChannelTransformSpec:
    """
    Holds the winning transformation settings for a single media channel.
    Determined by Stage 0 and carried unchanged into the final model so that
    every channel gets its own independent adstock + saturation + lag spec.

    Parameters
    ----------
    channel_idx  : positional index of the channel in prep["spend_cols"]
    channel_name : human-readable column name
    adstock_type : "geometric" | "weibull"
    saturation   : "softplus" | "hill" | "logistic" | "exponential"
    max_lag      : maximum lag periods to consider (0 = lag disabled)
    use_lag      : whether to apply a separate discrete lag shift before adstock
    """
    channel_idx  : int
    channel_name : str
    adstock_type : str  = "geometric"
    saturation   : str  = "softplus"
    max_lag      : int  = 0
    use_lag      : bool = False

    def label(self) -> str:
        lag_str = f"+lag{self.max_lag}" if self.use_lag and self.max_lag > 0 else ""
        return f"{self.adstock_type}+{self.saturation}{lag_str}"


# ─────────────────────────────────────────────────────────────────────────────
# DataConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """All data-related settings. Separates data concerns from model concerns."""

    csv_path        : str
    date_col        : str        = "date"
    # Generic default — MUST be overridden to match the actual response column
    # in your CSV (e.g. "revenue", "sales", "conversions").
    response_col    : str        = "revenue"
    spend_prefix    : str        = "spends_"
    media_cols      : List[str]  = field(default_factory=list)
    control_cols    : List[str]  = field(default_factory=list)
    min_date        : Optional[str] = None         # ISO string e.g. "2021-01-01"
    max_date        : Optional[str] = None         # ISO string e.g. "2024-12-31"
    # Date format flag.  False = MM/DD/YYYY (US default, safer universal fallback).
    # Set to True only for DD/MM/YYYY (European / UK) formatted date columns.
    dayfirst        : bool       = False
    output_dir      : str        = "mmm_outputs"
    # Dataset frequency — drives Fourier period and adstock window interpretation.
    # Supported: "weekly" (default), "daily", "monthly"
    frequency       : str        = "weekly"
    # Number of trailing periods to hold out for out-of-sample evaluation.
    # 0 = no holdout (all data used for training).
    holdout_periods : int        = 0
    # Known anomaly dates to add as spike indicators (e.g. Black Friday, data errors).
    # List of ISO date strings, e.g. ["2022-11-25", "2023-11-24"].
    # Each date gets a binary 0/1 column in the model (always included, ignores use_controls).
    outlier_dates   : List[str]  = field(default_factory=list)

    # ── Long-format auto-pivot ──────────────────────────────────────────────────
    # When input_format='long', the pipeline automatically pivots the input file
    # from long format (one row per date × product) to wide format (one row per
    # date, separate columns per product) before running the model.
    # Configure in YAML under the 'data' section:
    #
    #   data:
    #     input_format     : 'long'
    #     product_col      : 'Product'
    #     long_response_col: 'response_metric_sales_CC'
    #     long_media_cols  :
    #       Google         : 'spends_CC_Google'
    #       DigitalDisplay : 'spends_CC_DigitalDisplay'
    #
    # Supports both .csv and .xlsx input files.
    input_format      : str             = "wide"   # "wide" | "long"
    product_col       : str             = "Product" # column identifying products in long format
    long_response_col : str             = ""        # single response col in long format
    long_media_cols   : Dict[str, str]  = field(default_factory=dict)
    # {channel_label → source_col_in_long_format} for the MODELLED variable
    # e.g. {"Google": "media_impressions_CC_Google", "Meta": "media_impressions_CC_Meta"}
    long_spend_cols   : Dict[str, str]  = field(default_factory=dict)
    # {channel_label → source_spend_col_in_long_format} for ROI calculation only
    # Required when long_media_cols points to impressions/clicks rather than spend.
    # e.g. {"Google": "spends_CC_Google", "Meta": "spends_CC_Meta"}
    # The spend columns are included in the wide CSV but NOT used as model inputs.

    def __post_init__(self):
        if not Path(self.csv_path).exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        valid_freqs = {"weekly", "daily", "monthly"}
        if self.frequency.lower() not in valid_freqs:
            raise ValueError(
                f"DataConfig.frequency='{self.frequency}' is not recognised. "
                f"Choose from: {sorted(valid_freqs)}"
            )
        if self.holdout_periods < 0:
            raise ValueError("DataConfig.holdout_periods must be >= 0.")


# ─────────────────────────────────────────────────────────────────────────────
# ModelConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    All model hyperparameters. One instance = one model to evaluate.

    Notes on per-channel transforms
    -------------------------------
    adstock_type / saturation here are used as DEFAULTS only — for Stage 0
    scanning and for channels that don't have an explicit ChannelTransformSpec.
    In the final model (Stage 2), build_mmm() receives a channel_specs dict
    that overrides these defaults per-channel.
    """

    max_lag          : int    = 8
    fourier_order    : int    = 2
    adstock_type     : str    = "geometric"     # "geometric" | "weibull"
    saturation       : str    = "softplus"      # "softplus" | "hill" | "logistic" | "exponential"
    use_trend        : bool   = True
    baseline_type    : str    = "linear_trend"  # "linear_trend" | "gaussian_random_walk" | "noncentered_gaussian_random_walk" | "piecewise_linear"
    piecewise_knots  : int    = 3
    use_hierarchical          : bool   = False
    use_controls              : bool   = True

    # ── Hierarchical model options ────────────────────────────
    # Two-timescale adstock: each channel gets a slow + fast geometric
    # chain mixed by a learnable weight.  Requires use_hierarchical=True
    # to pool the half-life priors across channels within each family.
    use_two_timescale_adstock : bool   = False

    # Cross-channel synergies: add pairwise family×family interaction terms
    # to mu.  Requires F >= 2 channel families.
    use_synergies             : bool   = False

    # Campaigns as amplifiers: campaigns multiply channel effects rather than
    # having direct additive effects. Requires campaign columns in schema.
    use_campaigns             : bool   = False

    # Cross-product halos: positive spillover effects between products.
    # Requires P >= 2 products.
    use_halos                 : bool   = False

    # Global Hill saturation: use single alpha/k parameters across all channels
    # instead of per-channel Hill parameters.
    use_global_hill           : bool   = False

    # ── Advanced Model Features ──────────────────────────────────────────────
    use_time_varying_betas    : bool   = False
    use_dynamic_saturation    : bool   = False

    # ── Robustness and Validation ────────────────────────────────────────────
    enable_strict_validation  : bool   = True
    enable_outlier_detection  : bool   = True
    enable_collinearity_check : bool   = True
    enable_stationarity_test  : bool   = True

    # Sigma hyperparameters for the hierarchical beta priors.
    # family_beta_sigma   — HalfNormal on the family-level mean beta spread
    # channel_beta_sigma  — HalfNormal on channel deviations within a family
    # product_beta_sigma  — HalfNormal on product deviations from channel mean
    family_beta_sigma         : float  = 0.5
    channel_beta_sigma        : float  = 0.3
    product_beta_sigma        : float  = 0.5

    # Override family-level adstock configs (replaces DEFAULT_FAMILY_CONFIGS entries)
    # e.g. {"tv": ChannelFamilyConfig("tv", hl_slow_median=6.0)}
    family_configs            : Dict[str, "ChannelFamilyConfig"] = field(default_factory=dict)
    response_transform: str   = "log1p"         # "log1p" | "sqrt" | "boxcox" | "identity"
    target_accept    : float  = 0.95
    draws            : int    = 1800
    tune             : int    = 800
    chains           : int    = 2

    # ── Data robustness gates ────────────────────────────────
    # These are used to mark models as "poor" via quality_flag if the
    # dataset/model is too risky (short series, severe collinearity).
    min_T_for_production              : int   = 100
    collinearity_max_offdiag_abs      : float = 0.85
    collinearity_fail                 : bool  = False
    prefer_spend_over_secondary_metrics : bool = True
    # "auto" picks the right strategy based on response_transform.
    # "clip_to_zero" clips negatives to 0.
    # "clip_to_epsilon" clips to a small positive epsilon.
    negative_response_policy          : str   = "auto"
    fast_draws       : int    = 120
    fast_tune        : int    = 120
    fast_chains      : int    = 1
    init             : str    = "auto"
    progressbar      : bool   = True
    # ── Lag settings (user-prompted before pipeline) ──────────
    use_lag          : bool   = False   # whether to apply discrete lag per channel
    global_max_lag   : int    = 0       # upper bound user provided; 0 = disabled
    # Day-of-week effects for daily data.
    # None = auto-prompt when daily data detected.
    # True/False = explicit override.
    use_dow_effects  : Optional[bool] = None

    def key(self) -> str:
        """Short human-readable identifier for this config."""
        return (
            f"lag{self.max_lag}_fo{self.fourier_order}_"
            f"{self.adstock_type}_{self.saturation}_"
            f"ta{int(self.target_accept * 100)}"
        )

    def to_dict(self) -> Dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Grid builder
# ─────────────────────────────────────────────────────────────────────────────

def build_full_grid(base_cfg: ModelConfig) -> List[ModelConfig]:
    """Constructs the full hyperparameter grid (48 configs: 3 lags x 2 fourier x 2 adstock x 4 sat)."""
    grid = []
    for max_lag, fo, ads, sat in itertools.product(
        [4, 8, 13],
        [1, 2],
        ["geometric", "weibull"],
        ["softplus", "hill", "logistic", "exponential"],
    ):
        cfg               = deepcopy(base_cfg)
        cfg.max_lag       = max_lag
        cfg.fourier_order = fo
        cfg.adstock_type  = ads
        cfg.saturation    = sat
        cfg.target_accept = 0.95
        grid.append(cfg)

    logger.info(f"Full grid: {len(grid)} configurations")
    return grid


def build_pruned_grid(
    full_grid  : List[ModelConfig],
    best_combo : Dict[int, Tuple[str, str]],
    stage0_ran : bool = True,
) -> List[ModelConfig]:
    """
    Prunes the full grid for Stage 1b.

    When stage0_ran=True: adstock/sat are locked per-channel from Stage 0.
    Dedup by (lag, fourier_order) only — adstock/sat don't matter since
    channel_specs overrides them.

    When stage0_ran=False (e.g. hierarchical multi-product auto-skip):
    adstock/sat were never searched, so keep the full (lag, fourier, adstock, sat)
    grid so Stage 1b actually explores all transform options.
    """
    seen   : set = set()
    pruned : List[ModelConfig] = []
    for cfg in full_grid:
        if stage0_ran:
            key = (cfg.max_lag, cfg.fourier_order)
        else:
            key = (cfg.max_lag, cfg.fourier_order, cfg.adstock_type, cfg.saturation)
        if key not in seen:
            seen.add(key)
            pruned.append(cfg)

    dim_label = "(lag x fourier_order)" if stage0_ran else "(lag x fourier x adstock x sat)"
    logger.info(
        f"Pruned grid: {len(full_grid)} -> {len(pruned)} unique "
        f"{dim_label} combinations"
    )
    return pruned


def build_channel_specs(
    best_combo     : Dict[int, Tuple[str, str]],
    spend_cols     : List[str],
    use_lag        : bool = False,
    global_max_lag : int  = 0,
) -> Dict[int, "ChannelTransformSpec"]:
    """
    Converts Stage 0 best_combo into a full {channel_idx -> ChannelTransformSpec}
    dict that is passed to build_mmm() for the final model.

    Parameters
    ----------
    best_combo     : {j -> (adstock, saturation)} from Stage 0
    spend_cols     : ordered list of channel column names
    use_lag        : whether user opted in to lag transformation
    global_max_lag : the user-supplied maximum lag (periods); used for all channels
    """
    specs = {}
    for j, col in enumerate(spend_cols):
        ads, sat = best_combo.get(j, ("geometric", "softplus"))
        specs[j] = ChannelTransformSpec(
            channel_idx  = j,
            channel_name = col,
            adstock_type = ads,
            saturation   = sat,
            max_lag      = global_max_lag if use_lag else 0,
            use_lag      = use_lag and global_max_lag > 0,
        )
    return specs


def _dedupe_combo_list(combos: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    out  = []
    for c in combos:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ─── DataSchema (schema.py) ────────────────────────────────────────────────

import yaml
from dataclasses import dataclass as _dc_schema, field as _field_schema
from pathlib import Path as _PathSchema
from typing import Any as _Any, Dict as _Dict_s, List as _List_s, Optional as _Opt_s, Tuple as _Tuple_s

# Re-use already-imported symbols (logging, dataclass, field, Path, typing) where possible.

VALID_ROLES       = {"media", "base", "macro", "event", "control", "campaign"}
VALID_METRIC_TYPES = {"spend", "impressions", "clicks", "grp", "views", "reach", "numeric"}
VALID_TRANSFORMS   = {"adstock+saturation", "linear", "log", "sqrt", "z_score", "none"}

# Recognised channel family names — derived from DEFAULT_FAMILY_CONFIGS above
def _load_valid_families():
    try:
        return set(DEFAULT_FAMILY_CONFIGS.keys())
    except Exception:
        return {
            "search", "social", "tv", "display", "ooh",
            "programmatic", "email", "radio", "video", "affiliate", "generic",
        }

VALID_FAMILIES = _load_valid_families()
DEFAULT_FAMILY = "generic"   # fallback when no family is specified


@dataclass
class ColumnRole:
    """
    Describes a single column's purpose in the MMM.

    Parameters
    ----------
    column       : actual column name in the CSV
    role         : "media" | "base" | "macro" | "event" | "control" | "campaign"
    channel      : human-readable group name (e.g. "google", "price")
    metric_type  : "spend" | "impressions" | "clicks" | "grp" | "numeric"
    channel_type : prior library key (e.g. "paid_search", "price", "cpi")
    spend_column : parallel spend column for ROI (None if self is spend)
    transform    : how to transform this variable in the model
    product      : product/brand group for hierarchical (None = flat)
    region       : region for geo-hierarchical (None = national)
    family       : channel family for hierarchical adstock/beta pooling
                   e.g. "search", "social", "tv", "display", "ooh"
                   Channels in the same family share group-level priors.
                   Falls back to "generic" if not set.
    """
    column        : str
    role          : str  = "media"
    channel       : str  = ""
    metric_type   : str  = "spend"
    channel_type  : str  = ""         # prior library lookup key
    spend_column  : Optional[str] = None
    transform     : str  = "adstock+saturation"
    product       : Optional[str] = None
    region        : Optional[str] = None
    family        : str  = DEFAULT_FAMILY  # channel family for hierarchical pooling
    flighting_col : Optional[str] = None  # binary column: 1=active, 0=dark; applied post-saturation

    def __post_init__(self):
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"role '{self.role}' not valid. Choose from: {sorted(VALID_ROLES)}"
            )
        if not self.channel:
            self.channel = self.column
        # Normalise and validate family
        self.family = (self.family or "generic").strip().lower()
        if self.role in ("media", "campaign") and self.family not in VALID_FAMILIES:
            raise ValueError(
                f"family '{self.family}' not valid for {self.role} column '{self.column}'. "
                f"Choose from: {sorted(VALID_FAMILIES)}"
            )
        # Default transforms by role
        if self.role in ("media", "campaign") and self.transform == "none":
            self.transform = "adstock+saturation"
        elif self.role in ("base", "macro", "control") and self.transform == "adstock+saturation":
            self.transform = "z_score"
        elif self.role == "event":
            self.transform = "none"


@dataclass
class DataSchema:
    """
    Complete description of a dataset for MMM.

    Can be constructed from:
      1. A YAML config file  -> DataSchema.from_yaml("config.yaml")
      2. Interactive prompts  -> built by main.py at runtime
      3. Programmatic code    -> DataSchema(date_col=..., ...)

    Multi-product support
    ─────────────────────
    Set ``product`` on each ColumnRole to activate multi-product mode.
    Also populate ``product_response_map`` to map each product name to its
    response column.  Example::

        product_response_map = {
            "brand_a": "revenue_brand_a",
            "brand_b": "revenue_brand_b",
        }
    """
    date_col             : str
    response_col         : str
    columns              : List[ColumnRole]      = field(default_factory=list)
    frequency            : str                   = "weekly"
    dayfirst             : bool                  = True
    min_date             : Optional[str]         = None
    max_date             : Optional[str]         = None
    # Maps product_name -> response column name (multi-product mode only)
    product_response_map : Dict[str, str]        = field(default_factory=dict)

    # -- Convenience accessors -----------------------------------------

    @property
    def media_cols(self) -> List[ColumnRole]:
        return [c for c in self.columns if c.role == "media"]

    # -- Multi-product helpers -----------------------------------------

    @property
    def products(self) -> List[str]:
        """Sorted list of unique product names from media columns (empty = flat)."""
        names = sorted({c.product for c in self.media_cols if c.product})
        return names

    @property
    def is_multi_product(self) -> bool:
        return len(self.products) > 1

    @property
    def unique_channel_names(self) -> List[str]:
        """
        Deduplicated channel names (shared across products).
        When multi-product: the C channels that appear in every product.
        When flat: all channel names.
        """
        return sorted({c.channel for c in self.media_cols})

    # -- Channel family helpers ----------------------------------------

    @property
    def unique_families(self) -> List[str]:
        """Sorted list of unique family names across all media columns."""
        return sorted({(c.family or DEFAULT_FAMILY) for c in self.media_cols})

    @property
    def channel_family_map(self) -> Dict[str, str]:
        """
        Maps channel column name -> family name.
        Used to build the family_idx array in data_prep.
        """
        return {c.column: (c.family or DEFAULT_FAMILY) for c in self.media_cols}

    @property
    def campaign_cols(self) -> List[ColumnRole]:
        return [c for c in self.columns if c.role == "campaign"]

    @property
    def base_cols(self) -> List[ColumnRole]:
        return [c for c in self.columns if c.role == "base"]

    @property
    def macro_cols(self) -> List[ColumnRole]:
        return [c for c in self.columns if c.role == "macro"]

    @property
    def event_cols(self) -> List[ColumnRole]:
        return [c for c in self.columns if c.role == "event"]

    @property
    def control_cols(self) -> List[ColumnRole]:
        return [c for c in self.columns if c.role == "control"]

    @property
    def all_media_column_names(self) -> List[str]:
        return [c.column for c in self.media_cols]

    @property
    def all_base_column_names(self) -> List[str]:
        return [c.column for c in self.base_cols]

    @property
    def all_macro_column_names(self) -> List[str]:
        return [c.column for c in self.macro_cols]

    @property
    def all_event_column_names(self) -> List[str]:
        return [c.column for c in self.event_cols]

    @property
    def all_control_column_names(self) -> List[str]:
        return [c.column for c in self.control_cols]

    @property
    def all_campaign_column_names(self) -> List[str]:
        return [c.column for c in self.campaign_cols]

    @property
    def channel_variable_map(self) -> Dict[str, str]:
        """Returns {channel_name -> column_name} for media channels."""
        return {c.channel: c.column for c in self.media_cols}

    def describe(self) -> str:
        lines = [
            f"DataSchema: {self.frequency} data",
            f"  Date: {self.date_col}  |  Response: {self.response_col}",
            f"  Media channels: {len(self.media_cols)}",
        ]
        for c in self.media_cols:
            lines.append(f"    {c.channel:<25s} -> {c.column} ({c.metric_type}, type={c.channel_type})")
        if self.base_cols:
            lines.append(f"  Base variables: {len(self.base_cols)}")
            for c in self.base_cols:
                lines.append(f"    {c.channel:<25s} -> {c.column} (type={c.channel_type})")
        if self.macro_cols:
            lines.append(f"  Macro variables: {len(self.macro_cols)}")
            for c in self.macro_cols:
                lines.append(f"    {c.channel:<25s} -> {c.column} (type={c.channel_type})")
        if self.event_cols:
            lines.append(f"  Event dummies: {len(self.event_cols)}")
            for c in self.event_cols:
                lines.append(f"    {c.column}")
        if self.campaign_cols:
            lines.append(f"  Campaign variables: {len(self.campaign_cols)}")
            for c in self.campaign_cols:
                lines.append(f"    {c.channel:<25s} -> {c.column} ({c.metric_type}, type={c.channel_type})")
        return "\n".join(lines)


def load_config_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Loads a YAML config file and returns a dictionary with all settings.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must be a YAML dictionary, got {type(cfg)}")

    logger.info(f"[CONFIG] Loaded YAML config from: {yaml_path}")
    return cfg


def build_schema_from_yaml(cfg: Dict[str, Any]) -> DataSchema:
    """Constructs a DataSchema from a parsed YAML config dict."""
    data_cfg = cfg.get("data", {})
    var_cfg  = cfg.get("variables", {})

    columns = []

    # Media
    for m in var_cfg.get("media", []):
        columns.append(ColumnRole(
            column        = m["column"],
            role          = "media",
            channel       = m.get("channel", m["column"]),
            metric_type   = m.get("metric_type", "spend"),
            channel_type  = m.get("channel_type", "generic_media"),
            spend_column  = m.get("spend_column"),
            transform     = "adstock+saturation",
            product       = m.get("product"),
            region        = m.get("region"),
            family        = m.get("family", DEFAULT_FAMILY),
            flighting_col = m.get("flighting_col"),
        ))

    # Base  (guard: YAML `base:` with no value parses as None, not [])
    for b in (var_cfg.get("base") or []):
        columns.append(ColumnRole(
            column       = b["column"],
            role         = "base",
            channel      = b.get("channel", b["column"]),
            channel_type = b.get("channel_type", "generic_base"),
            transform    = b.get("transform", "z_score"),
        ))

    # Macro  (guard: YAML `macro:` with no value parses as None, not [])
    for m in (var_cfg.get("macro") or []):
        columns.append(ColumnRole(
            column       = m["column"],
            role         = "macro",
            channel      = m.get("channel", m["column"]),
            channel_type = m.get("channel_type", "generic_macro"),
            transform    = m.get("transform", "z_score"),
        ))

    # Events  (guard: YAML `events:` with no value parses as None, not [])
    for e in (var_cfg.get("events") or []):
        col = e if isinstance(e, str) else e["column"]
        columns.append(ColumnRole(
            column       = col,
            role         = "event",
            channel      = col,
            channel_type = "event",
            transform    = "none",
        ))

    # Controls  (guard: YAML `controls:` with no value parses as None, not [])
    for c in (var_cfg.get("controls") or []):
        col     = c if isinstance(c, str) else c["column"]
        label   = c.get("channel", col) if isinstance(c, dict) else col
        columns.append(ColumnRole(
            column       = col,
            role         = "control",
            channel      = label,
            channel_type = "generic_base",
            transform    = "z_score",
        ))

    # Multi-product response map
    product_response_map: Dict[str, str] = {}
    for item in var_cfg.get("product_responses", []):
        product_response_map[item["product"]] = item["response_col"]

    schema = DataSchema(
        date_col             = data_cfg.get("date_col", "date"),
        response_col         = data_cfg.get("response_col", "revenue"),
        columns              = columns,
        frequency            = data_cfg.get("frequency", "weekly"),
        dayfirst             = data_cfg.get("dayfirst", True),
        min_date             = data_cfg.get("min_date"),
        max_date             = data_cfg.get("max_date"),
        product_response_map = product_response_map,
    )

    logger.info(f"[CONFIG] Schema built:\n{schema.describe()}")
    return schema


def build_schema_from_legacy(
    channel_variable_map : Dict[str, str],
    control_cols         : List[str],
    date_col             : str   = "date",
    response_col         : str   = "revenue",
    dayfirst             : bool  = True,
) -> DataSchema:
    """
    Backward-compatible: builds a DataSchema from the old-style
    channel_variable_map and control_cols.
    """
    columns = []
    for channel, col in channel_variable_map.items():
        mt = "spend"
        if "impressions" in col.lower() or "imps_" in col.lower():
            mt = "impressions"
        elif "clicks" in col.lower() or "clks_" in col.lower():
            mt = "clicks"
        elif "views" in col.lower():
            mt = "views"
        elif "reach" in col.lower():
            mt = "reach"

        columns.append(ColumnRole(
            column      = col,
            role        = "media",
            channel     = channel,
            metric_type = mt,
        ))

    for col in (control_cols or []):
        columns.append(ColumnRole(
            column  = col,
            role    = "control",
            channel = col,
        ))

    return DataSchema(
        date_col     = date_col,
        response_col = response_col,
        columns      = columns,
        dayfirst     = dayfirst,
    )


def generate_sample_yaml(output_path: str = "mmm_config_sample.yaml") -> str:
    """Writes a sample YAML config file and returns its path."""
    sample = """# ──────────────────────────────────────────────────────────────
# Bayesian MMM Configuration File
# ──────────────────────────────────────────────────────────────

data:
  csv_path: "data/weekly_sales.csv"
  date_col: "week"
  response_col: "revenue"
  frequency: "weekly"       # weekly | daily | monthly
  dayfirst: true            # DD/MM/YYYY format
  min_date: null            # e.g. "2022-01-01"
  max_date: null

variables:
  media:
    - column: "spend_google"
      channel: "google_search"
      channel_type: "paid_search"    # see prior_library for types
      metric_type: "spend"
      family: "search"               # channel family for hierarchical pooling
    - column: "spend_meta"
      channel: "meta_social"
      channel_type: "paid_social"
      metric_type: "spend"
      family: "social"
    - column: "grp_tv"
      channel: "tv_national"
      channel_type: "tv"
      metric_type: "grp"
      spend_column: "spend_tv"
      family: "tv"

  product_responses: []
  base:
    - column: "avg_price"
      channel_type: "price"
    - column: "num_stores"
      channel_type: "distribution"
    - column: "promo_depth"
      channel_type: "promotion"
  macro:
    - column: "cpi_index"
      channel_type: "cpi"
    - column: "unemployment_rate"
      channel_type: "unemployment"
  events:
    - column: "black_friday"
    - column: "product_launch"

model:
  baseline_type: "linear_trend"
  fourier_order: 2
  use_lag: false
  max_lag: 0

hierarchical:
  enabled: false
  use_two_timescale_adstock: false
  use_synergies: false

sampling:
  runtime_mode: "fast_20m"
  draws: 1800
  tune: 800
  chains: 2
  target_accept: 0.95

output:
  dir: "mmm_outputs"
  export_excel: true
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(sample)
    return output_path
