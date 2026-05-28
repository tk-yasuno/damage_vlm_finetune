"""
create_figures_v072.py
v0.7.2 progressive evaluation (1k/2k/3k/4k) のコサイン類似度可視化図を生成する。
出力先: paper_damage_vlm/1_Methodology/figures_vlm/
  12_v072_similarity_by_scale.png  … mean/median比較バーチャート
  13_v072_tier_distribution.png    … 品質Tier積み上げ棒グラフ
  14_v072_cosine_violin.png        … ヴァイオリンプロット
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# 共通設定
# ─────────────────────────────────────────
BASE = Path("data/v07_fine_tuning/evaluations_unsloth_v072")
OUT  = Path("paper_damage_vlm/1_Methodology/figures_vlm")
OUT.mkdir(parents=True, exist_ok=True)

STAGES = ["1k", "2k", "3k", "4k"]
COLORS_STAGE = ["#a8c8e8", "#5aa5d4", "#1f77b4", "#0d4f8b"]  # 薄→濃

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────
metrics = {}
quality = {}
cosine_vals = {}

for stage in STAGES:
    j = json.loads((BASE / f"evaluation_v07_{stage}.json").read_text(encoding="utf-8"))
    metrics[stage] = j["overall_metrics"]["cosine_similarity"]
    quality[stage] = j["quality_distribution"]

    # per_sample から cosine スコアを取り出す
    rows = j.get("per_sample_results", [])
    if rows:
        vals = []
        for r in rows:
            cs = r.get("cosine_similarity")
            if cs is not None:
                vals.append(cs)
        cosine_vals[stage] = np.array(vals)
    else:
        # fallback: CSVから読む
        csv = pd.read_csv(BASE / f"inference_results_v07_{stage}.csv")
        col = [c for c in csv.columns if "cosine" in c.lower()]
        if col:
            cosine_vals[stage] = csv[col[0]].dropna().values
        else:
            cosine_vals[stage] = np.array([])

# v0.3/v0.6.3 比較用（旧pairdata_v1実績）
V03_MEANS = {"1k": 0.6191, "2k": 0.6550, "3k": 0.6909, "4k": 0.6739}
V03_STDS  = {"1k": 0.0875, "2k": 0.0814, "3k": 0.0784, "4k": 0.0795}

means   = [metrics[s]["mean"]   for s in STAGES]
stds    = [metrics[s]["std"]    for s in STAGES]
medians = [metrics[s]["median"] for s in STAGES]

# ═══════════════════════════════════════════════════════════════
# Figure 12: Mean & Median Cosine Similarity by Training Scale
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(STAGES))
w = 0.35

bars_mean = ax.bar(x - w/2, means, w,
                   color=COLORS_STAGE, edgecolor="white", linewidth=0.5,
                   label="Mean (v0.7.2)", zorder=3)
bars_med  = ax.bar(x + w/2, medians, w,
                   color=COLORS_STAGE, edgecolor="white", linewidth=0.5,
                   alpha=0.6, label="Median (v0.7.2)", zorder=3)

# エラーバー（mean ± std）
ax.errorbar(x - w/2, means, yerr=stds,
            fmt="none", color="gray", capsize=4, linewidth=1.2, zorder=4)

# v0.3 比較線（点線）
v03_means_list = [V03_MEANS[s] for s in STAGES]
ax.plot(x - w/2, v03_means_list, "o--", color="#d62728", linewidth=1.4,
        markersize=5, label="Mean (v0.3 pairdata-v1)", zorder=5, alpha=0.8)

# 数値ラベル
for bar, val in zip(bars_mean, means):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.004,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
for bar, val in zip(bars_med, medians):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.004,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8, color="#444444")

ax.set_xticks(x)
ax.set_xticklabels([f"{s} steps" for s in STAGES])
ax.set_xlabel("Training Scale")
ax.set_ylabel("Cosine Similarity")
ax.set_title("Semantic Similarity by Training Scale\n(v0.7.2, pairdata-v2, n=800 test set)",
             fontweight="bold")
ax.set_ylim(0.4, 0.80)
ax.axhline(0.65, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
ax.text(3.55, 0.652, "0.65", fontsize=7, color="gray")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="y", linewidth=0.5, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT / "12_v072_similarity_by_scale.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("Saved: 12_v072_similarity_by_scale.png")

# ═══════════════════════════════════════════════════════════════
# Figure 13: Quality Tier Distribution (stacked bar)
# ═══════════════════════════════════════════════════════════════
TIERS   = ["excellent", "good", "acceptable", "poor", "very_poor"]
TIER_LABELS = ["Excellent\n(≥0.85)", "Good\n(0.70–0.85)", "Acceptable\n(0.55–0.70)",
               "Poor\n(0.40–0.55)", "Very Poor\n(<0.40)"]
TIER_COLORS = ["#2ca02c", "#5aa5d4", "#aec7e8", "#ff7f0e", "#d62728"]

pcts = {tier: [quality[s][tier]["percentage"] for s in STAGES] for tier in TIERS}

fig, ax = plt.subplots(figsize=(8, 5))
bottoms = np.zeros(len(STAGES))
x = np.arange(len(STAGES))

for tier, label, color in zip(TIERS, TIER_LABELS, TIER_COLORS):
    vals = np.array(pcts[tier])
    ax.bar(x, vals, bottom=bottoms, color=color, label=label,
           edgecolor="white", linewidth=0.5, zorder=3)
    # 5%超のセルだけラベル表示
    for i, (v, b) in enumerate(zip(vals, bottoms)):
        if v >= 5.0:
            ax.text(i, b + v/2, f"{v:.1f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels([f"{s} steps" for s in STAGES])
ax.set_xlabel("Training Scale")
ax.set_ylabel("Percentage (%)")
ax.set_title("Quality Tier Distribution by Training Scale\n(v0.7.2, pairdata-v2, n=800 test set)",
             fontweight="bold")
ax.set_ylim(0, 105)
ax.legend(loc="upper right", fontsize=8, ncol=1)
ax.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT / "13_v072_tier_distribution.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("Saved: 13_v072_tier_distribution.png")

# ═══════════════════════════════════════════════════════════════
# Figure 14: Violin plot of cosine similarity distribution
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

# violinplotのデータリスト
data_list = [cosine_vals[s] for s in STAGES]
valid = [d for d in data_list if len(d) > 0]

if len(valid) == len(STAGES):
    vp = ax.violinplot(data_list, positions=range(len(STAGES)),
                       widths=0.6, showmedians=True, showextrema=False)

    for i, pc in enumerate(vp["bodies"]):
        pc.set_facecolor(COLORS_STAGE[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor("gray")
        pc.set_linewidth(0.8)

    vp["cmedians"].set_color("white")
    vp["cmedians"].set_linewidth(2.0)

    # 四分位箱を重ねる
    for i, d in enumerate(data_list):
        if len(d) == 0:
            continue
        q25, q50, q75 = np.percentile(d, [25, 50, 75])
        ax.vlines(i, q25, q75, color="white", linewidth=5, zorder=3)
        ax.scatter(i, q50, color="white", zorder=4, s=20)

    # mean点
    for i, d in enumerate(data_list):
        if len(d) > 0:
            ax.scatter(i, d.mean(), marker="D", color="#333333",
                       zorder=5, s=30, label="Mean" if i == 0 else "")

    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([f"{s} steps" for s in STAGES])
    ax.set_xlabel("Training Scale")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity Distribution by Training Scale\n(v0.7.2, pairdata-v2, n=800 test set)",
                 fontweight="bold")
    ax.set_ylim(-0.05, 1.0)
    ax.axhline(0.70, color="green", linewidth=0.8, linestyle="--", alpha=0.5, label="Good threshold (0.70)")
    ax.axhline(0.55, color="orange", linewidth=0.8, linestyle="--", alpha=0.5, label="Acceptable threshold (0.55)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT / "14_v072_cosine_violin.png", bbox_inches="tight", dpi=150)
    print("Saved: 14_v072_cosine_violin.png")
else:
    print("WARNING: per_sample_results not available for violin plot, skipping Fig 14")

plt.close("all")
print("\nAll figures generated.")
print("\n=== Summary Table ===")
print(f"{'Stage':>6} {'Mean':>8} {'Std':>8} {'Median':>8} | {'Good+Acc%':>10} {'Poor+VP%':>10}")
for s in STAGES:
    m  = metrics[s]
    q  = quality[s]
    ga = q['excellent']['percentage'] + q['good']['percentage'] + q['acceptable']['percentage']
    pv = q['poor']['percentage'] + q['very_poor']['percentage']
    print(f"{s:>6} {m['mean']:>8.4f} {m['std']:>8.4f} {m['median']:>8.4f} | {ga:>10.1f} {pv:>10.1f}")
