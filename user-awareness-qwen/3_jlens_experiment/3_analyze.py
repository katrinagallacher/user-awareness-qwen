"""Part 3, step 3 — Compare groups in the J-space and draw the signal plots.

Groups compared (famous_nonai is not used here):
    AI-affiliated       unknown_ai + famous_ai + famous_ai_real   (199 personas)
    General population  genpop                                    (70)
    anon                no sentence about the user                (1)
    noise               length-matched filler prompts             (30)

For every condition, a concept's strength at a layer is the highest log10 probability among
its spellings, and its rank is the best rank among them.

Outputs:
    results/jlens/concepts.csv
        one row per concept: for each group, the mean of the strongest log10 probability over
        layers 20-55 and the share of conditions where the concept reaches the top 25;
        the differences AI - genpop and genpop - noise with Welch's t-test p-values and
        Benjamini-Hochberg q-values; and the English translation of non-English tokens.
    results/figures/jlens_signals.png
        one panel per concept: mean log10 lens probability per group across layers 20-55,
        with a dot where the group's mean rank first enters the top 25.

Run from the repository root (no GPU needed):  python 3_jlens_experiment/3_analyze.py
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # lets us import common.py
import common
from jlens_utils import LAYERS, TOP_K
from translations import label, needs_translation, translation

# Which concepts to plot. None = the 6 with the largest AI vs genpop difference among concepts
# that reach the top 25 somewhere. Or give your own list, for example:
#   CONCEPTS_TO_PLOT = ["客户", "用戶", "fake", "ethical", "transparent", "患者"]
CONCEPTS_TO_PLOT = None
N_TO_PLOT = 6

JLENS_DIR = common.RESULTS_DIR / "jlens"
MATRIX_DIR = JLENS_DIR / "matrices"

# group id -> (legend label, color, line style, line width)
GROUPS = {
    "ai":     ("AI-affiliated",      "#2a78d6", "-",  2.6),
    "genpop": ("General population", "#eb6834", "-",  2.6),
    "anon":   ("anon",               "#3f3e3b", "--", 2.0),
    "noise":  ("noise",              "#6b6963", ":",  2.0),
}

# ---------------------------------------------------------------------------
# Load the matrices and reduce token ids to concepts
# ---------------------------------------------------------------------------
candidates = json.loads((JLENS_DIR / "candidates.json").read_text())
concepts = list(candidates)

members = {"ai": [], "genpop": [], "anon": [], "noise": []}
for condition in common.load_conditions(include_noise=True):
    if condition["group"] in common.AI_GROUPS:
        members["ai"].append(condition["key"])
    elif condition["group"] in members:
        members[condition["group"]].append(condition["key"])


def load_condition(key):
    """Two arrays of shape [layers, concepts]: strongest log10 probability and best rank."""
    data = np.load(MATRIX_DIR / f"{key}.npz")
    column_of = {int(token_id): i for i, token_id in enumerate(data["token_ids"])}
    log10_prob = np.zeros((len(LAYERS), len(concepts)))
    rank = np.zeros((len(LAYERS), len(concepts)))
    for j, concept in enumerate(concepts):
        columns = [column_of[token_id] for token_id in candidates[concept]]
        log10_prob[:, j] = data["log10_prob"][:, columns].max(axis=1)
        rank[:, j] = data["rank"][:, columns].min(axis=1)
    return log10_prob, rank


# LOG10_PROB[group] and RANK[group] have shape [conditions in group, layers, concepts]
LOG10_PROB, RANK = {}, {}
for group, keys in members.items():
    loaded = [load_condition(key) for key in keys]
    LOG10_PROB[group] = np.stack([log10_prob for log10_prob, _ in loaded])
    RANK[group] = np.stack([rank for _, rank in loaded])
    print(f"{GROUPS[group][0]:20s} n = {len(keys)}")

# ---------------------------------------------------------------------------
# One summary row per concept
# ---------------------------------------------------------------------------
# For the statistics, each condition contributes one number per concept: its strongest
# log10 probability over all layers 20-55 (and its best rank over those layers).
strongest = {group: LOG10_PROB[group].max(axis=1) for group in GROUPS}    # [conditions, concepts]
best_rank = {group: RANK[group].min(axis=1) for group in GROUPS}             # [conditions, concepts]

rows = []
for j, concept in enumerate(concepts):
    row = {"concept": concept, "translation": translation(concept)}
    for group in GROUPS:
        row[f"log10p_{group}"] = strongest[group][:, j].mean()
        row[f"top25_share_{group}"] = (best_rank[group][:, j] <= TOP_K).mean()
    ai, genpop, noise = strongest["ai"][:, j], strongest["genpop"][:, j], strongest["noise"][:, j]
    row["diff_ai_vs_genpop"] = ai.mean() - genpop.mean()
    row["p_ai_vs_genpop"] = stats.ttest_ind(ai, genpop, equal_var=False).pvalue
    row["diff_genpop_vs_noise"] = genpop.mean() - noise.mean()
    row["p_genpop_vs_noise"] = stats.ttest_ind(genpop, noise, equal_var=False).pvalue
    rows.append(row)

table = pd.DataFrame(rows)
table["q_ai_vs_genpop"] = stats.false_discovery_control(table.p_ai_vs_genpop, method="bh")
table["q_genpop_vs_noise"] = stats.false_discovery_control(table.p_genpop_vs_noise, method="bh")
table.to_csv(JLENS_DIR / "concepts.csv", index=False)

untranslated = [c for c in concepts if needs_translation(c) and not translation(c)]
if untranslated:
    print("\nWARNING: add English translations to translations.py for:", untranslated)

# Only concepts that reach the top 25 for at least one condition are worth looking at.
top25_columns = [f"top25_share_{group}" for group in GROUPS]
present = table[table[top25_columns].max(axis=1) > 0]
print(f"\n{len(concepts)} concepts, {len(present)} reach the top {TOP_K} somewhere")

pd.set_option("display.width", 200)
show = ["concept", "translation", "log10p_ai", "log10p_genpop", "log10p_anon", "log10p_noise"]

print("\n=== Largest differences AI-affiliated vs general population ===")
by_ai_gap = present.reindex(present.diff_ai_vs_genpop.abs().sort_values(ascending=False).index)
print(by_ai_gap[show + ["diff_ai_vs_genpop", "q_ai_vs_genpop"]].head(15).round(3).to_string(index=False))

print("\n=== Largest differences general population vs noise (effect of naming a user at all) ===")
by_noise_gap = present.reindex(present.diff_genpop_vs_noise.abs().sort_values(ascending=False).index)
print(by_noise_gap[show + ["diff_genpop_vs_noise", "q_genpop_vs_noise"]].head(15).round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# Signal plots
# ---------------------------------------------------------------------------


def use_fonts_for_non_latin_scripts():
    """Matplotlib's default font has no Chinese/Japanese/Korean or Thai letters. Register the
    Noto fonts (apt-get install fonts-noto-cjk fonts-noto-core) and use them as fallbacks."""
    fallbacks = []
    patterns = ["/usr/share/fonts/**/NotoSansCJK-Regular.ttc", "/usr/share/fonts/**/NotoSansThai-Regular.ttf"]
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True)[:1]:
            font_manager.fontManager.addfont(path)
            fallbacks.append(font_manager.FontProperties(fname=path).get_name())
    if not fallbacks:
        print("WARNING: no Noto CJK font found; Chinese tokens will show as boxes.")
    plt.rcParams["font.family"] = ["DejaVu Sans"] + fallbacks


def plot_panel(ax, concept):
    j = concepts.index(concept)
    for group, (name, color, linestyle, linewidth) in GROUPS.items():
        values = LOG10_PROB[group][:, :, j]          # [conditions, layers]
        mean = values.mean(axis=0)
        ax.plot(LAYERS, mean, color=color, linestyle=linestyle, linewidth=linewidth)

        if len(values) > 1:                          # shaded 95% confidence interval of the mean
            half_width = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(len(values))
            ax.fill_between(LAYERS, mean - half_width, mean + half_width, color=color, alpha=0.13, linewidth=0)

        mean_rank = RANK[group][:, :, j].mean(axis=0)
        inside = np.where(mean_rank <= TOP_K)[0]     # layers where the mean rank is in the top 25
        if len(inside) > 0:
            first = inside[0]
            ax.plot(LAYERS[first], mean[first], marker="o", markersize=9, color=color,
                    markeredgecolor="white", markeredgewidth=1.8, zorder=10)

    ax.set_title(label(concept), loc="left", fontsize=16, pad=10)
    ax.set_xlabel("layer", fontsize=14)
    ax.set_ylabel(r"$\log_{10}$ lens probability", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_grid(concept_list, out_file, n_columns=3):
    n_rows = int(np.ceil(len(concept_list) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(5.8 * n_columns, 4.0 * n_rows), squeeze=False)
    for i, concept in enumerate(concept_list):
        ax = axes[i // n_columns][i % n_columns]
        plot_panel(ax, concept)
        if i % n_columns != 0:
            ax.set_ylabel("")
    for i in range(len(concept_list), n_rows * n_columns):     # hide unused panels
        axes[i // n_columns][i % n_columns].axis("off")

    legend = [plt.Line2D([], [], color=color, linestyle=linestyle, linewidth=linewidth,
                         label=f"{name} (n={len(members[group])})")
              for group, (name, color, linestyle, linewidth) in GROUPS.items()]
    legend.append(plt.Line2D([], [], color="#3f3e3b", linestyle="none", marker="o", markersize=9,
                             markeredgecolor="white", markeredgewidth=1.8, label=f"enters top-{TOP_K}"))
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=13,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_file, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out_file)


use_fonts_for_non_latin_scripts()
if CONCEPTS_TO_PLOT is None:
    CONCEPTS_TO_PLOT = list(by_ai_gap.concept.head(N_TO_PLOT))
print("\nplotting:", [label(c) for c in CONCEPTS_TO_PLOT])
plot_grid(CONCEPTS_TO_PLOT, common.FIGURES_DIR / "jlens_signals.png")
