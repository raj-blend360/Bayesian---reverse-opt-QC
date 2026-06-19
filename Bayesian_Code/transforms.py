# transforms.py
# ─────────────────────────────────────────────────────────────────────────────
# Adstock (carry-over), lag shift, and saturation (diminishing returns)
# transform functions.  All operate on PyTensor tensors for use inside PyMC
# models.
#
# Transform pipeline per channel (in order):
#   1. Lag shift      (optional)  — apply_lag_shift()
#   2. Adstock        (required)  — geometric_adstock() / weibull_adstock()
#   3. Saturation     (required)  — sat_softplus / sat_hill / sat_logistic /
#                                   sat_exponential
# ─────────────────────────────────────────────────────────────────────────────

from typing import Optional

import pymc as pm
import pytensor.tensor as pt
from pytensor.scan import scan as _pt_scan


# ─────────────────────────────────────────────────────────────────────────────
# Lag shift  (Step 1 in the channel transform pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def apply_lag_shift(
    x        : pt.TensorVariable,
    lag_w    : pt.TensorVariable,
    max_lag  : int,
) -> pt.TensorVariable:
    """
    Continuous (NUTS-compatible) lag transformation.

    Instead of a discrete Categorical lag, we use a **softmax-weighted
    mixture** over all possible lags [0, 1, …, max_lag].  The model
    learns an un-normalised weight vector `lag_logits` (shape max_lag+1)
    and a softmax turns it into a probability distribution over lags.
    The output is the weighted average of all shifted versions of x:

        out[t] = Σ_k  softmax(lag_logits)[k]  ×  x_shifted_by_k[t]

    This is fully differentiable — NUTS can compute gradients through
    it — yet still captures the "which lag is most likely" concept.

    When one logit dominates, the output is essentially the series
    shifted by that lag.  When logits are flat (prior), the output is
    a uniform average across all lags.

    Parameters
    ----------
    x       : (T,) spend vector (scaled)
    lag_w   : (max_lag+1,) softmax weights — output of pm.Dirichlet or
              pt.softmax(lag_logits).  Must sum to 1 and be non-negative.
    max_lag : int — number of lag options (Python int, not a tensor)

    Returns
    -------
    (T,) lag-weighted spend vector
    """
    shifted_cols = []
    for k in range(max_lag + 1):
        if k == 0:
            shifted_cols.append(x)
        else:
            # x[:-k] is pure symbolic negative-index slicing — no Python
            # int conversion of shape needed.
            padded = pt.concatenate(
                [pt.zeros(k, dtype="float64"), x[:-k].astype("float64")]
            )
            shifted_cols.append(padded)

    # (max_lag+1, T) → transpose → (T, max_lag+1)
    shift_matrix = pt.stack(shifted_cols, axis=0).T

    # (T, max_lag+1) @ (max_lag+1,) → (T,)
    return pt.dot(shift_matrix, lag_w)



# ─────────────────────────────────────────────────────────────────────────────
# Adstock  (Step 2)
# ─────────────────────────────────────────────────────────────────────────────

def geometric_adstock(
    x       : pt.TensorVariable,
    lam     : pt.TensorVariable,
    max_lag : int,
) -> pt.TensorVariable:
    """
    Geometric (exponential) adstock.
    Weight at lag k = λ^k, L1-normalised so total scale is preserved.
    Best for channels with immediate-to-moderate carry-over (search, social, email).

    Parameters
    ----------
    x       : (T,) spend vector (scaled)
    lam     : scalar in [0, 1] — decay rate
    max_lag : int — maximum lag to consider

    Returns
    -------
    (T,) adstocked spend

    Implementation note
    -------------------
    Uses pt.stack + pt.dot instead of a scalar accumulation loop.
    This builds a (T, max_lag+1) shift matrix and reduces it with a
    single matrix-vector multiply, which compiles to BLAS and reduces
    the symbolic graph depth — improving compile time and gradient flow.
    """
    w = pt.power(lam, pt.arange(max_lag + 1, dtype="float64"))
    w = w / (pt.sum(w) + 1e-12)

    # Build (T, max_lag+1) matrix of time-shifted x vectors.
    shifted = []
    for k in range(max_lag + 1):
        if k == 0:
            shifted.append(x)
        else:
            # Pad the front with k zeros; x[:-k] is pure symbolic slicing.
            shifted.append(
                pt.concatenate([pt.zeros(k, dtype="float64"), x[:-k].astype("float64")])
            )
    shift_matrix = pt.stack(shifted, axis=1)   # (T, max_lag+1)
    return pt.dot(shift_matrix, w)             # (T,)


def weibull_adstock(
    x       : pt.TensorVariable,
    lam     : pt.TensorVariable,
    k_shape : pt.TensorVariable,
    max_lag : int,
) -> pt.TensorVariable:
    """
    Weibull adstock — allows delayed-peak response.
    Weight at lag t = (k/lam)(t/lam)^(k-1) exp(-(t/lam)^k)
    When k<1: decaying (like geometric but heavier tail)
    When k>1: peak at lag t = lam*((k-1)/k)^(1/k) — delayed hump
    L1-normalised.
    Best for TV, OOH, billboard.

    Parameters
    ----------
    x        : (T,) spend vector (scaled)
    lam      : scale parameter (weeks to peak)
    k_shape  : shape parameter (controls peak timing)
    max_lag  : int — maximum lag to consider

    Implementation note
    -------------------
    Same pt.stack + pt.dot vectorisation as geometric_adstock.
    """
    lags = pt.arange(1, max_lag + 1, dtype="float64")
    w    = (k_shape / lam) * ((lags / lam) ** (k_shape - 1.0)) * pt.exp(-((lags / lam) ** k_shape))
    w    = pt.concatenate([[pt.ones(())], w])
    w    = w / (pt.sum(w) + 1e-12)

    # Build (T, max_lag+1) shift matrix and reduce with a single dot product.
    shifted = []
    for k in range(max_lag + 1):
        if k == 0:
            shifted.append(x)
        else:
            shifted.append(
                pt.concatenate([pt.zeros(k, dtype="float64"), x[:-k].astype("float64")])
            )
    shift_matrix = pt.stack(shifted, axis=1)   # (T, max_lag+1)
    return pt.dot(shift_matrix, w)             # (T,)


# ─────────────────────────────────────────────────────────────────────────────
# Two-timescale adstock  (Step 2 — alternative to single-timescale geometric)
# ─────────────────────────────────────────────────────────────────────────────

def two_timescale_adstock(
    x        : pt.TensorVariable,
    rho_slow : pt.TensorVariable,
    rho_fast : pt.TensorVariable,
    w_mix    : pt.TensorVariable,
    max_lag  : int,
) -> pt.TensorVariable:
    """
    Two-timescale geometric adstock: a slow carry-over chain and a fast
    decay chain, blended by a learnable mixing weight.

    Delegates to ``geometric_adstock`` for each chain to avoid duplicating
    the convolution loop.  The constraint ρ_f < ρ_s is enforced by the
    caller via the hl_ratio prior (slow_hl / fast_hl > 1).

    Parameters
    ----------
    x        : (T,) spend vector (scaled)
    rho_slow : scalar in [0, 1] — slow decay rate (higher = longer carryover)
    rho_fast : scalar in [0, 1] — fast decay rate (lower = shorter carryover)
    w_mix    : scalar in [0, 1] — weight on the fast chain
    max_lag  : int — convolution window length

    Returns
    -------
    (T,) blended adstocked spend
    """
    fast_chain = geometric_adstock(x, rho_fast, max_lag)
    slow_chain = geometric_adstock(x, rho_slow, max_lag)
    return w_mix * fast_chain + (1.0 - w_mix) * slow_chain


# ─────────────────────────────────────────────────────────────────────────────
# Scan-based adstock  (Step 2 — state-space, full history, no truncation)
# ─────────────────────────────────────────────────────────────────────────────

def geometric_adstock_scan(
    x     : pt.TensorVariable,
    lam   : pt.TensorVariable,
    reset : Optional[pt.TensorVariable] = None,
) -> pt.TensorVariable:
    """
    Geometric adstock via a true recurrent state-space scan.

    Unlike the truncated-window convolution in ``geometric_adstock()``, this
    function maintains an explicit carry state ``s[t]`` across all T timesteps:

        s[t] = x[t]  +  s[t-1] * (1 - reset[t]) * lam

    No truncation lag is needed — the full history is propagated automatically.
    For long half-life channels (e.g. Offline at 6 weeks) a truncated window
    of 8 steps only covers ~75 % of the theoretical effect; the scan is exact.

    The optional ``reset`` tensor handles flighting restart events.  When
    ``reset[t] = 1.0`` (first active week after ≥ N consecutive dark weeks),
    the carry is zeroed before adding x[t], preventing ghost carry-over from
    the previous campaign flight from contaminating the new one.

    Parameters
    ----------
    x     : (T,) spend tensor (scaled), PyTensor variable
    lam   : scalar decay rate in (0, 1) — PyTensor variable
    reset : (T,) optional restart mask — 1.0 at restart weeks, else 0.0.
            Supply ``prep["reset_mask"][train_idx, j]`` as a PyTensor constant.

    Returns
    -------
    (T,) adstocked spend
    """
    if reset is None:
        reset = pt.zeros_like(x)

    def _step(x_t, r_t, s_prev, lam_):
        # Zero carry on restart, then accumulate new spend
        s_new = x_t + s_prev * (1.0 - r_t) * lam_
        return s_new

    states, _ = _pt_scan(
        fn=_step,
        sequences=[x, reset],
        outputs_info=[pt.zeros((), dtype="float64")],
        non_sequences=[lam],
        strict=True,
    )
    return states


def two_timescale_adstock_scan(
    x        : pt.TensorVariable,
    rho_slow : pt.TensorVariable,
    rho_fast : pt.TensorVariable,
    w_mix    : pt.TensorVariable,
    reset    : Optional[pt.TensorVariable] = None,
) -> pt.TensorVariable:
    """
    Two-timescale geometric adstock via state-space scan.

    Maintains two independent carry states — a fast chain (short half-life)
    and a slow chain (long half-life) — stepped forward recurrently:

        sF[t] = x[t]  +  sF[t-1] * (1 - reset[t]) * rho_fast
        sS[t] = x[t]  +  sS[t-1] * (1 - reset[t]) * rho_slow
        out[t] = w_mix * sF[t]  +  (1 - w_mix) * sS[t]

    Both chains receive the same restart signal — when ``reset[t] = 1`` both
    carries are wiped simultaneously, so the blended signal also restarts
    cleanly after a dark period.

    This replaces the old ``two_timescale_adstock()`` call which delegated to
    two ``geometric_adstock()`` truncated convolutions.  The scan is exact
    regardless of half-life length.

    Parameters
    ----------
    x        : (T,) spend tensor (scaled)
    rho_slow : scalar slow decay rate in (0, 1)
    rho_fast : scalar fast decay rate in (0, 1); caller ensures rho_fast < rho_slow
               via the half-life ratio prior in Block 6.
    w_mix    : scalar mixing weight in (0, 1) — weight on the fast chain
    reset    : (T,) optional restart mask — 1.0 at restart weeks, else 0.0

    Returns
    -------
    (T,) blended adstocked spend
    """
    if reset is None:
        reset = pt.zeros_like(x)

    def _step(x_t, r_t, sF_prev, sS_prev, rho_f_, rho_s_):
        sF = x_t + sF_prev * (1.0 - r_t) * rho_f_
        sS = x_t + sS_prev * (1.0 - r_t) * rho_s_
        return sF, sS

    (sF_seq, sS_seq), _ = _pt_scan(
        fn=_step,
        sequences=[x, reset],
        outputs_info=[
            pt.zeros((), dtype="float64"),
            pt.zeros((), dtype="float64"),
        ],
        non_sequences=[rho_fast, rho_slow],
        strict=True,
    )
    return w_mix * sF_seq + (1.0 - w_mix) * sS_seq


# ─────────────────────────────────────────────────────────────────────────────
# Saturation  (Step 3)
# ─────────────────────────────────────────────────────────────────────────────

def sat_softplus(x: pt.TensorVariable, alpha: pt.TensorVariable) -> pt.TensorVariable:
    """
    Softplus saturation: log(1 + exp(alpha * x)) / log(2)
    Smooth, always positive, weakly saturating.
    Good default or for channels with mild diminishing returns.
    """
    return pt.log1p(pt.exp(alpha * x)) / pt.log(2.0)


def sat_hill(
    x     : pt.TensorVariable,
    alpha : pt.TensorVariable,
    kappa : pt.TensorVariable,
) -> pt.TensorVariable:
    """
    Hill / power-law saturation: x^alpha / (x^alpha + kappa^alpha)
    Industry standard for media saturation modelling.
    alpha controls steepness, kappa controls the half-saturation point.
    Maps x to [0, 1].
    Good for TV, display — channels with strong diminishing returns.
    """
    x_pos = pt.maximum(x, 1e-12)
    return x_pos ** alpha / (x_pos ** alpha + kappa ** alpha)


def sat_logistic(
    x  : pt.TensorVariable,
    k  : pt.TensorVariable,
    x0 : pt.TensorVariable,
) -> pt.TensorVariable:
    """
    Logistic saturation: 1 / (1 + exp(-k*(x - x0)))
    Has an explicit inflection point at x0.
    Good for channels that have a minimum viable frequency threshold.
    Maps x to (0, 1).
    """
    return pm.math.sigmoid(k * (x - x0))


def sat_exponential(
    x     : pt.TensorVariable,
    alpha : pt.TensorVariable,
) -> pt.TensorVariable:
    """
    Negative-exponential saturation: 1 - exp(-alpha * x)

    Properties
    ──────────
    • Maps x ≥ 0  →  [0, 1)  — strictly bounded, never reaches 1
    • Concave everywhere — strongest diminishing returns from the first unit
    • No inflection point — immediate saturation, no delayed peak
    • alpha controls saturation speed:
        small alpha (e.g. 0.5) → slow, near-linear for small x
        large alpha (e.g. 5.0) → rapid saturation, near-flat beyond small x

    Best for channels with strong and immediate diminishing returns:
    email, push notifications, retargeting, high-spend paid search.
    """
    x_pos = pt.maximum(x, 0.0)
    return 1.0 - pt.exp(-alpha * x_pos)


def apply_saturation(
    x         : pt.TensorVariable,
    sat_type  : str,
    alpha_sat : pt.TensorVariable,
    kappa     : Optional[pt.TensorVariable] = None,
    k_log     : Optional[pt.TensorVariable] = None,
    x0        : Optional[pt.TensorVariable] = None,
) -> pt.TensorVariable:
    """Dispatches to the correct saturation function."""
    if sat_type == "softplus":
        return sat_softplus(x, alpha_sat)
    elif sat_type == "hill":
        return sat_hill(x, alpha_sat, kappa)
    elif sat_type == "logistic":
        return sat_logistic(x, k_log, x0)
    elif sat_type == "exponential":
        return sat_exponential(x, alpha_sat)
    else:
        raise ValueError(f"Unknown saturation type: {sat_type}")
