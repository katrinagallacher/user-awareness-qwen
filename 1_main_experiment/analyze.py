"""Part 1 — Analysis of the main experiment.

Reads results/main_results.jsonl and:
  1. computes a rate per persona for each axis (share of prompts where the judge said "yes"),
  2. compares every group with the general population (Welch's t-test on persona rates),
  3. compares the three AI groups with each other,
  4. looks for outlier personas inside each group (robust z-score),
  5. draws one boxplot per axis: a box per group, one dot per persona, the anonymous
     baseline as a red dashed line.

Run from the repository root:  python 1_main_experiment/analyze.py
"""

import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # lets us import common.py
import common

AXES = common.AXES
GROUPS = common.GROUP_ORDER

scores = common.load_scores(common.RESULTS_DIR / "main_results.jsonl")
print(f"{len(scores):,} judged conversations, {scores.persona.nunique()} conditions\n")

# One row per persona: the share of its prompts that got a "yes" on each axis.
# Example: 19 refusals out of 99 judged prompts -> refusal rate 0.192.
persona_rates = scores.groupby(["group", "persona"])[AXES].mean().reset_index()
anon_rate = persona_rates[persona_rates.persona == "anon"].iloc[0]
persona_rates = persona_rates[persona_rates.group != "anon"]

print("Group means (mean of persona rates):")
print(persona_rates.groupby("group")[AXES].mean().reindex(GROUPS).round(3).to_string())
print("anon (single condition):", "  ".join(f"{axis} {anon_rate[axis]:.3f}" for axis in AXES))


def rates(group, axis):
    return persona_rates[persona_rates.group == group][axis].values


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ---------------------------------------------------------------------------
# 1. Every group vs the general population
# ---------------------------------------------------------------------------
print("\n=== Each group vs genpop (Welch's t-test on persona rates) ===")
for axis in AXES:
    genpop = rates("genpop", axis)
    print(f"\n[{axis}]  genpop mean = {genpop.mean():.3f}")
    for group in GROUPS[1:]:
        values = rates(group, axis)
        t, p = stats.ttest_ind(values, genpop, equal_var=False)
        difference = values.mean() - genpop.mean()
        relative = difference / genpop.mean() * 100
        print(f"   {group:15s} mean = {values.mean():.3f}   difference = {difference:+.3f} "
              f"({relative:+.1f}%)   t = {t:+.2f}   p = {p:.4f} {stars(p)}")

# ---------------------------------------------------------------------------
# 2. The three AI groups against each other
# ---------------------------------------------------------------------------
print("\n=== AI groups against each other (Welch's t-test) ===")
for axis in AXES:
    print(f"\n[{axis}]")
    for group_a, group_b in combinations(common.AI_GROUPS, 2):
        t, p = stats.ttest_ind(rates(group_a, axis), rates(group_b, axis), equal_var=False)
        print(f"   {group_a} vs {group_b}:  t = {t:+.2f}   p = {p:.4f} {stars(p)}")

# ---------------------------------------------------------------------------
# 3. Outlier personas inside each group
# ---------------------------------------------------------------------------
# Robust z-score = 0.6745 * (rate - group median) / MAD, where MAD is the median absolute
# deviation. Median and MAD are used instead of mean and SD because outliers would pull the
# mean and SD towards themselves. |z| > 3.5 is the usual cut-off.
print("\n=== Outlier personas (|robust z| > 3.5 within their group) ===")
n_outliers = 0
for group in GROUPS:
    for axis in AXES:
        group_rows = persona_rates[persona_rates.group == group]
        values = group_rows[axis].values
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0:
            continue
        z = 0.6745 * (values - median) / mad
        for persona, rate, z_value in zip(group_rows.persona, values, z):
            if abs(z_value) > 3.5:
                n_outliers += 1
                print(f"   {group:15s} {axis:12s} {persona:45s} rate = {rate:.3f} "
                      f"(group median {median:.3f})   z = {z_value:+.2f}")
print(f"   {n_outliers} outlier(s) in total")

# ---------------------------------------------------------------------------
# 4. Boxplots with one dot per persona
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)   # fixed seed so the horizontal jitter is the same every run

for axis in AXES:
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [rates(group, axis) for group in GROUPS]

    ax.boxplot(data, positions=range(len(GROUPS)), widths=0.62, showfliers=False,
               patch_artist=True, zorder=1,
               medianprops=dict(color="#222222", linewidth=2),
               boxprops=dict(facecolor="white", edgecolor="#888888"),
               whiskerprops=dict(color="#888888"), capprops=dict(color="#888888"))

    for i, values in enumerate(data):
        x = i + rng.normal(0, 0.07, size=len(values))     # small horizontal jitter
        ax.scatter(x, values, s=22, alpha=0.6, zorder=3)

    ax.axhline(anon_rate[axis], linestyle="--", color="red", linewidth=1.2, zorder=2,
               label=f"anon = {anon_rate[axis]:.3f}")
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels(GROUPS, rotation=20, ha="right")
    ax.set_ylabel(f"{axis} rate (one dot = one persona)")
    ax.set_title(f"{axis} — median and persona distribution per group")
    ax.legend(loc="upper right")
    plt.tight_layout()
    out_file = common.FIGURES_DIR / f"main_{axis}.png"
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    print("saved", out_file)
