#!/usr/bin/env python3
"""
ADAM conflict-resolution sensitivity analysis.

Evaluates the sensitivity of the deterministic conflict-resolution rule
(Eq. 5 of the ADAM manuscript) to the weighting parameters lambda_1
(severity) and lambda_2 = 1 - lambda_1 (recency).

Rule:
    a* = argmax_a ( lambda_1 * severity_norm(a) + lambda_2 * invtime_norm(a) )

where severity and inverse timestamp are min-max normalised to [0, 1]
before the weighted sum is applied.

The mechanism was never triggered during the 459-event live deployment,
so this analysis is a synthetic stress test of the decision rule itself.
It characterises how often the selected action changes as lambda_1 varies;
it does NOT validate field behaviour of overlapping crews.

Two normalisation regimes are evaluated:

  (1) PAIRWISE  - min-max applied across the two competing candidates only,
                  which is the literal reading of the manuscript for a
                  two-candidate conflict. Degenerate: the larger value maps
                  to 1 and the smaller to 0.

  (2) WINDOW    - min-max applied across the population of concurrent events
                  in the decision window, which preserves the magnitude of
                  the severity and recency gaps.

Author: generated for the ADAM manuscript revision
Seed fixed for reproducibility.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Presentation settings shared with the other manuscript figures.
BLUE = "#0072B2"    # Okabe-Ito, colour-vision-deficiency safe
ORANGE = "#E69F00"
INK = "#1a1a1a"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "font.size": 15,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.0,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SEED = 42
N_CONFLICTS = 20_000

# Severity levels from the Decision Agent output schema:
# none, low, moderate, high, critical  ->  0..4
SEV_MIN, SEV_MAX = 0, 4

# Event age in seconds. Conflicts are only possible between crews formed
# within the decision deadline (C1 = 30 s).
AGE_MIN, AGE_MAX = 0.5, 30.0

LAMBDA_GRID = np.linspace(0.0, 1.0, 501)
LAMBDA_PAPER = 0.7

rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------
# Synthetic conflict generation
# ----------------------------------------------------------------------
def generate_conflicts(n):
    """Two competing crew recommendations per conflict."""
    sev_a = rng.integers(SEV_MIN, SEV_MAX + 1, n).astype(float)
    sev_b = rng.integers(SEV_MIN, SEV_MAX + 1, n).astype(float)
    age_a = rng.uniform(AGE_MIN, AGE_MAX, n)
    age_b = rng.uniform(AGE_MIN, AGE_MAX, n)
    return sev_a, sev_b, age_a, age_b


def normalise_pairwise(x_a, x_b):
    """Min-max across the two candidates: larger -> 1, smaller -> 0."""
    lo = np.minimum(x_a, x_b)
    hi = np.maximum(x_a, x_b)
    span = hi - lo
    tied = span == 0
    span_safe = np.where(tied, 1.0, span)
    na = (x_a - lo) / span_safe
    nb = (x_b - lo) / span_safe
    # Ties normalise to equal value (0.5) rather than an arbitrary 0/1 split
    na = np.where(tied, 0.5, na)
    nb = np.where(tied, 0.5, nb)
    return na, nb


def normalise_window(x_a, x_b, lo, hi):
    """Min-max against a fixed population range."""
    span = hi - lo
    return (x_a - lo) / span, (x_b - lo) / span


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def choose(sev_na, sev_nb, inv_na, inv_nb, lam1):
    """Return +1 if candidate A wins, -1 if B wins, 0 if exactly tied."""
    lam2 = 1.0 - lam1
    score_a = lam1 * sev_na + lam2 * inv_na
    score_b = lam1 * sev_nb + lam2 * inv_nb
    diff = score_a - score_b
    out = np.sign(diff)
    return out


def flip_threshold(d_sev, d_inv):
    """
    lambda_1 at which the decision flips, per conflict.

    score_a - score_b = lam1*d_sev + (1-lam1)*d_inv
                      = lam1*(d_sev - d_inv) + d_inv
    Zero at lam1* = d_inv / (d_inv - d_sev)

    Only meaningful for 'contested' conflicts where severity and recency
    disagree (d_sev and d_inv have opposite, non-zero signs).
    """
    denom = d_inv - d_sev
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_star = np.where(denom != 0, d_inv / denom, np.nan)
    return lam_star


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
def run():
    sev_a, sev_b, age_a, age_b = generate_conflicts(N_CONFLICTS)

    inv_a = 1.0 / age_a
    inv_b = 1.0 / age_b

    results = {}

    # ---- Regime 1: pairwise normalisation ----
    ps_a, ps_b = normalise_pairwise(sev_a, sev_b)
    pi_a, pi_b = normalise_pairwise(inv_a, inv_b)

    # ---- Regime 2: window-population normalisation ----
    ws_a, ws_b = normalise_window(sev_a, sev_b, SEV_MIN, SEV_MAX)
    inv_lo, inv_hi = 1.0 / AGE_MAX, 1.0 / AGE_MIN
    wi_a, wi_b = normalise_window(inv_a, inv_b, inv_lo, inv_hi)

    for name, (sa, sb, ia, ib) in {
        "pairwise": (ps_a, ps_b, pi_a, pi_b),
        "window": (ws_a, ws_b, wi_a, wi_b),
    }.items():

        d_sev = sa - sb
        d_inv = ia - ib

        # Contested: severity and recency point to different candidates
        contested = (np.sign(d_sev) != 0) & (np.sign(d_inv) != 0) & \
                    (np.sign(d_sev) != np.sign(d_inv))
        frac_contested = contested.mean()

        # Baseline decision at the manuscript's lambda_1 = 0.7
        base = choose(sa, sb, ia, ib, LAMBDA_PAPER)

        agreement = np.empty_like(LAMBDA_GRID)
        for k, lam in enumerate(LAMBDA_GRID):
            dec = choose(sa, sb, ia, ib, lam)
            agreement[k] = np.mean(dec == base)

        lam_star = flip_threshold(d_sev, d_inv)
        lam_star_contested = lam_star[contested]
        lam_star_contested = lam_star_contested[
            np.isfinite(lam_star_contested)
            & (lam_star_contested >= 0)
            & (lam_star_contested <= 1)
        ]

        # Widest band around 0.7 with 100% agreement
        idx070 = np.argmin(np.abs(LAMBDA_GRID - LAMBDA_PAPER))
        lo_i = idx070
        while lo_i > 0 and agreement[lo_i - 1] == 1.0:
            lo_i -= 1
        hi_i = idx070
        while hi_i < len(LAMBDA_GRID) - 1 and agreement[hi_i + 1] == 1.0:
            hi_i += 1
        stable_band = (LAMBDA_GRID[lo_i], LAMBDA_GRID[hi_i])

        # Band with >= 99% agreement
        ok = agreement >= 0.99
        lo99 = idx070
        while lo99 > 0 and ok[lo99 - 1]:
            lo99 -= 1
        hi99 = idx070
        while hi99 < len(LAMBDA_GRID) - 1 and ok[hi99 + 1]:
            hi99 += 1
        band99 = (LAMBDA_GRID[lo99], LAMBDA_GRID[hi99])

        # Fraction of contested conflicts where severity prevails at 0.7
        sev_wins_at_070 = np.mean(lam_star_contested < LAMBDA_PAPER) \
            if lam_star_contested.size else float("nan")

        results[name] = dict(
            frac_contested=frac_contested,
            agreement=agreement,
            lam_star=lam_star_contested,
            stable_band=stable_band,
            band99=band99,
            sev_wins_at_070=sev_wins_at_070,
        )

    return results


def report(results):
    print("=" * 74)
    print("ADAM CONFLICT-RESOLUTION SENSITIVITY  (Eq. 5)")
    print(f"synthetic conflicts: {N_CONFLICTS:,}   seed: {SEED}")
    print(f"severity levels: {SEV_MIN}-{SEV_MAX}   event age: "
          f"{AGE_MIN}-{AGE_MAX} s   lambda_1 grid: {len(LAMBDA_GRID)} pts")
    print("=" * 74)

    for name, r in results.items():
        print(f"\n--- {name.upper()} NORMALISATION ---")
        print(f"contested conflicts (severity vs recency disagree): "
              f"{r['frac_contested']*100:.1f}%")
        print(f"decisions identical to lambda_1=0.70 over: "
              f"[{r['stable_band'][0]:.3f}, {r['stable_band'][1]:.3f}]")
        print(f"agreement >= 99% over: "
              f"[{r['band99'][0]:.3f}, {r['band99'][1]:.3f}]")
        if r["lam_star"].size:
            print(f"flip thresholds lambda*: min={r['lam_star'].min():.3f}  "
                  f"median={np.median(r['lam_star']):.3f}  "
                  f"max={r['lam_star'].max():.3f}")
            print(f"contested conflicts resolved in favour of severity "
                  f"at lambda_1=0.70: {r['sev_wins_at_070']*100:.1f}%")

        a = r["agreement"]
        for lam in (0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0):
            k = np.argmin(np.abs(LAMBDA_GRID - lam))
            print(f"   lambda_1={lam:.2f} -> agreement with 0.70: "
                  f"{a[k]*100:6.2f}%")
    print("\n" + "=" * 74)


def make_figure(results, path):
    """Two-panel sensitivity figure, styled to match the other manuscript figures.

    No panel titles: the description belongs in the caption. Colours are the
    Okabe-Ito pair used throughout, so the section reads as one document.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0))

    # Panel (a): agreement against lambda_1
    ax = axes[0]
    ax.plot(LAMBDA_GRID, results["window"]["agreement"] * 100,
            lw=2.4, color=BLUE, label="Window normalisation", zorder=3)
    ax.plot(LAMBDA_GRID, results["pairwise"]["agreement"] * 100,
            lw=2.4, ls="--", color=ORANGE, label="Pairwise normalisation",
            zorder=3)
    ax.axvline(LAMBDA_PAPER, color="#4d4d4d", lw=1.2, ls=":", zorder=2)
    ax.text(LAMBDA_PAPER + 0.02, 6, r"$\lambda_1=0.7$", fontsize=14,
            color="#4d4d4d")
    ax.set_xlabel(r"Severity weight $\lambda_1$", labelpad=10)
    ax.set_ylabel("Decisions unchanged vs.\n" r"$\lambda_1=0.7$ (%)",
                  labelpad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 106)
    ax.grid(linestyle=":", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    ax.text(-0.15, 1.03, "(a)", transform=ax.transAxes, fontsize=19,
            fontweight="bold")

    # Panel (b): distribution of per-conflict flip thresholds
    ax = axes[1]
    ls = results["window"]["lam_star"]
    ax.hist(ls, bins=60, range=(0, 1), color=BLUE, edgecolor="white", lw=0.5)
    ax.axvline(LAMBDA_PAPER, color=ORANGE, lw=2.4, ls="--",
               label=r"$\lambda_1=0.7$ (this work)")
    ax.set_xlabel(r"Per-conflict flip threshold $\lambda_1^{*}$", labelpad=10)
    ax.set_ylabel("Contested conflicts", labelpad=10)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    ax.text(-0.15, 1.03, "(b)", transform=ax.transAxes, fontsize=19,
            fontweight="bold")

    fig.tight_layout(w_pad=4.0)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"figure written: {path}")


if __name__ == "__main__":
    res = run()
    report(res)
    make_figure(res, "figures/figure9_conflict_sensitivity.pdf")
