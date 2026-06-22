#!/usr/bin/env python3
"""Synthetic MMM optimizer stress test.

Generates a deterministic synthetic media-mix dataset, builds a lightweight
posterior-like model artifact, runs forward and reverse optimizers across
adstock/saturation/period/bounds scenarios, and writes a reproducible report.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xarray as xr

from config import ChannelTransformSpec
from optimisation import optimise_budget, minimise_spend_for_target

LOGGER = logging.getLogger("synthetic_optimizer_stress")


@dataclass(frozen=True)
class StressCase:
    name: str
    period: str
    budget_multiplier: float
    target_multiplier: float
    bounds: str
    cpp_mode: str


def _sat(x: np.ndarray, sat_type: str, alpha: float, kappa: float, k_log: float, x0: float) -> np.ndarray:
    if sat_type == "softplus":
        return np.log1p(np.exp(np.clip(alpha * x, -500, 500))) / np.log(2.0)
    if sat_type == "hill":
        xp = np.maximum(x, 1e-12)
        return xp**alpha / (xp**alpha + max(kappa, 1e-12)**alpha)
    if sat_type == "logistic":
        return 1.0 / (1.0 + np.exp(-k_log * (x - x0)))
    if sat_type == "exponential":
        return 1.0 - np.exp(-alpha * np.maximum(x, 0.0))
    raise ValueError(f"unknown saturation: {sat_type}")


def create_synthetic_dataset(n_periods: int, seed: int) -> tuple[pd.DataFrame, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n_periods, freq="W-MON")
    channels = ["spends_search", "spends_social", "spends_tv", "spends_display"]
    base = np.array([9000.0, 6500.0, 18000.0, 5000.0])
    phase = np.array([0.1, 1.7, 2.4, 3.0])
    trend = np.linspace(0.85, 1.20, n_periods)[:, None]
    seasonal = 1.0 + 0.18 * np.sin(2 * np.pi * np.arange(n_periods)[:, None] / 52.18 + phase)
    noise = rng.lognormal(mean=0.0, sigma=0.12, size=(n_periods, len(channels)))
    spend = base * trend * seasonal * noise
    spend[18, 2] *= 1.6
    spend[41, 0] *= 1.4

    spend_max = spend.max(axis=0)
    x_scaled = spend / (spend_max + 1e-8)
    true = {
        "lam": np.array([0.35, 0.22, 0.68, 0.18]),
        "beta": np.array([0.40, 0.28, 0.34, 0.20]),
        "alpha": np.array([1.45, 1.20, 1.75, 1.30]),
        "kappa": np.array([0.55, 0.45, 0.70, 0.38]),
        "k_log": np.array([5.0, 4.2, 3.3, 5.8]),
        "x0": np.array([0.35, 0.28, 0.50, 0.25]),
        "sat": ["hill", "softplus", "logistic", "exponential"],
    }
    contrib = np.zeros_like(x_scaled)
    carry = np.zeros(len(channels))
    for t in range(n_periods):
        carry = x_scaled[t] + true["lam"] * carry
        for j in range(len(channels)):
            contrib[t, j] = true["beta"][j] * _sat(
                carry[j], true["sat"][j], true["alpha"][j], true["kappa"][j], true["k_log"][j], true["x0"][j]
            )
    z_mu = 4.55 + 0.02 * np.arange(n_periods) / n_periods + contrib.sum(axis=1)
    revenue = np.expm1(z_mu + rng.normal(0.0, 0.035, n_periods))
    df = pd.DataFrame(spend, columns=channels)
    df.insert(0, "date", dates)
    df["revenue"] = revenue
    return df, {"channels": channels, "true": true, "contrib_z": contrib, "z_mu": z_mu}


def build_synthetic_model(df: pd.DataFrame, meta: Dict[str, Any], output_dir: Path, seed: int, draws: int) -> tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
    rng = np.random.default_rng(seed + 1)
    channels: List[str] = meta["channels"]
    C = len(channels)
    T = len(df)
    train_idx = np.arange(T)
    X_raw = df[channels].to_numpy(float)
    X_scaled = X_raw / (X_raw.max(axis=0) + 1e-8)
    y_raw = df["revenue"].to_numpy(float)
    y_log = np.log1p(y_raw)
    y_mu = float(y_log.mean())
    y_std = float(y_log.std() or 1.0)
    y_scaled = (y_log - y_mu) / y_std

    chains = 1
    draw_coord = np.arange(draws)
    true = meta["true"]
    def jitter(v: np.ndarray, sd: float, lo: float, hi: float) -> np.ndarray:
        return np.clip(rng.normal(v, sd, size=(chains, draws, C)), lo, hi)

    betas = jitter(true["beta"], 0.025, 0.03, 2.0)
    lam = jitter(true["lam"], 0.035, 0.0, 0.95)
    alpha_sat = jitter(true["alpha"], 0.08, 0.2, 4.0)
    kappa = jitter(true["kappa"], 0.04, 0.05, 2.0)
    k_logistic = jitter(true["k_log"], 0.20, 0.5, 10.0)
    x0 = jitter(true["x0"], 0.03, 0.0, 1.0)

    contrib_z = np.broadcast_to(meta["contrib_z"], (chains, draws, T, C)).copy()
    contrib_z *= rng.normal(1.0, 0.04, size=(chains, draws, 1, C))
    mu = ((meta["z_mu"] - y_mu) / y_std)[None, None, :] + rng.normal(0, 0.015, size=(chains, draws, T))

    posterior = xr.Dataset(
        {
            "betas": (("chain", "draw", "channel"), betas),
            "lam": (("chain", "draw", "channel"), lam),
            "alpha_sat": (("chain", "draw", "channel"), alpha_sat),
            "kappa": (("chain", "draw", "channel"), kappa),
            "k_logistic": (("chain", "draw", "channel"), k_logistic),
            "x0": (("chain", "draw", "channel"), x0),
            "media_by_channel": (("chain", "draw", "time", "channel"), contrib_z),
            "mu": (("chain", "draw", "time"), mu),
            "intercept": (("chain", "draw"), rng.normal(0, 0.01, size=(chains, draws))),
        },
        coords={"chain": [0], "draw": draw_coord, "channel": channels, "time": np.arange(T)},
    )
    trace = SimpleNamespace(posterior=posterior)
    channel_specs = {
        j: ChannelTransformSpec(j, ch, adstock_type="geometric", saturation=true["sat"][j]) for j, ch in enumerate(channels)
    }
    cfg = SimpleNamespace(adstock_type="geometric", saturation="hill")
    best = {"trace": trace, "cfg": cfg, "channel_specs": channel_specs, "_out_dir": str(output_dir)}
    prep = {
        "T": T, "C": C, "spend_cols": channels, "train_idx": train_idx,
        "X_media_raw": X_raw, "X_media_scaled": X_scaled, "y_raw": y_raw,
        "y_scaled": y_scaled, "y_mu": y_mu, "y_std": y_std,
        "response_transform": "log1p", "frequency": "weekly", "metric_types": ["Spend"] * C,
        "df": df, "_out_dir": str(output_dir),
    }
    cpp = np.ones(C)
    return best, prep, cpp


def bounds_for(case: StressCase, channels: List[str], observed_budget: float) -> tuple[dict[str, float] | None, dict[str, float] | None, float | None]:
    C = len(channels)
    if case.bounds == "none":
        return None, None, None
    if case.bounds == "guardrails":
        return {ch: observed_budget * 0.05 / C for ch in channels}, {ch: observed_budget * 0.55 for ch in channels}, None
    if case.bounds == "tight":
        return {ch: observed_budget * 0.18 / C for ch in channels}, {ch: observed_budget * 0.34 for ch in channels}, observed_budget * 1.05
    if case.bounds == "infeasible_low_cap":
        return None, {ch: observed_budget * 0.12 for ch in channels}, observed_budget * 0.35
    raise ValueError(case.bounds)


def run_stress(best: Dict[str, Any], prep: Dict[str, Any], cpp: np.ndarray, out_dir: Path, n_samples: int) -> pd.DataFrame:
    channels = prep["spend_cols"]
    observed_pp_budget = float(prep["X_media_raw"].mean(axis=0).sum())
    # Reverse optimizer targets media-attributed response, not full KPI baseline.
    observed_pp_response = 45.0
    cases = [
        StressCase("baseline_mean", "training_mean", 1.00, 1.00, "none", "native"),
        StressCase("under_budget", "training_mean", 0.75, 0.90, "guardrails", "native"),
        StressCase("growth_budget", "monthly", 1.20, 1.10, "guardrails", "native"),
        StressCase("tight_bounds", "quarterly", 1.00, 0.95, "tight", "native"),
        StressCase("stretch_year", "yearly", 1.15, 1.20, "guardrails", "native"),
        StressCase("infeasible_reverse", "training_mean", 0.45, 2.25, "infeasible_low_cap", "native"),
    ]
    rows = []
    for case in cases:
        period_factor = {"training_mean": 1.0, "monthly": 52.18 / 12, "quarterly": 52.18 / 4, "yearly": 52.18}.get(case.period, 1.0)
        budget = observed_pp_budget * period_factor * case.budget_multiplier
        target = observed_pp_response * period_factor * case.target_multiplier
        ch_min, ch_max, cap = bounds_for(case, channels, budget)
        status = "PASS"
        error = ""
        try:
            fwd = optimise_budget(best, prep, total_budget=budget, budget_period=case.period, n_samples=n_samples, channel_min=ch_min, channel_max=ch_max, cpp_weights=cpp)
            rev = minimise_spend_for_target(best, prep, target_response=target, target_period=case.period, n_samples=n_samples, channel_min=ch_min, channel_max=ch_max, budget_cap=cap, cpp_weights=cpp)
            fwd.to_csv(out_dir / f"forward_{case.name}.csv", index=False)
            rev.to_csv(out_dir / f"reverse_{case.name}.csv", index=False)
            ft = fwd[fwd["channel"] == "TOTAL"].iloc[0].to_dict()
            rt = rev[rev["channel"] == "TOTAL"].iloc[0].to_dict()
            if float(ft.get("convergence_rate_pct", 0)) < 80:
                status = "WARN"
            if (case.name != "infeasible_reverse") and not bool(rt.get("target_achievable", False)):
                status = "WARN"
            rows.append({"case": case.name, "status": status, "period": case.period, "budget": budget, "target": target, "forward_response": ft.get("response_mean"), "forward_convergence_pct": ft.get("convergence_rate_pct"), "forward_bounds_respected": ft.get("bounds_respected"), "reverse_spend": rt.get("min_spend_£_mean"), "reverse_achieved": rt.get("achieved_response_mean"), "reverse_achievable": rt.get("target_achievable"), "reverse_convergence_pct": rt.get("convergence_rate_pct"), "error": error})
        except Exception as exc:  # reported in output; test runner returns non-zero later
            rows.append({"case": case.name, "status": "FAIL", "period": case.period, "budget": budget, "target": target, "error": repr(exc)})
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "stress_results.csv", index=False)
    return results



def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_(no rows)_"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{v:,.4f}")
        else:
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else str(v))
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        vals = [str(row[h]).replace("|", "\\|") for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def write_report(df: pd.DataFrame, results: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> Path:
    passed = int((results["status"] == "PASS").sum())
    warned = int((results["status"] == "WARN").sum())
    failed = int((results["status"] == "FAIL").sum())
    report = out_dir / "synthetic_optimizer_stress_report.md"
    lines = [
        "# Synthetic MMM Optimizer Stress Report", "",
        "## Run configuration", "",
        f"- Seed: `{args.seed}`", f"- Periods: `{args.periods}` weekly observations", f"- Posterior draws used to build synthetic model: `{args.draws}`", f"- Optimizer samples per case: `{args.samples}`", "",
        "## Synthetic dataset", "",
        f"- Rows: `{len(df)}`", f"- Media channels: `{', '.join([c for c in df.columns if c.startswith('spends_')])}`", f"- Revenue range: `{df['revenue'].min():,.2f}` to `{df['revenue'].max():,.2f}`", "",
        "## Summary", "", f"- PASS: `{passed}`", f"- WARN: `{warned}`", f"- FAIL: `{failed}`", "",
        "## Case results", "", _markdown_table(results), "",
        "## Output artifacts", "", "- `synthetic_dataset.csv`", "- `stress_results.csv`", "- `forward_<case>.csv` and `reverse_<case>.csv` for each scenario", "",
        "## Interpretation", "", "Warnings are expected for intentionally constrained/infeasible scenarios; failures indicate exceptions in the optimizer path.",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="mmm_outputs/synthetic_stress")
    parser.add_argument("--periods", type=int, default=80)
    parser.add_argument("--draws", type=int, default=96)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, meta = create_synthetic_dataset(args.periods, args.seed)
    df.to_csv(out_dir / "synthetic_dataset.csv", index=False)
    best, prep, cpp = build_synthetic_model(df, meta, out_dir, args.seed, args.draws)
    results = run_stress(best, prep, cpp, out_dir, args.samples)
    report = write_report(df, results, out_dir, args)
    (out_dir / "run_metadata.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    LOGGER.info("Report written to %s", report)
    return 1 if (results["status"] == "FAIL").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
