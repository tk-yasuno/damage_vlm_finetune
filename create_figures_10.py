"""
create_figures_10.py
PASS 727 サンプルの Priority Score 分析 + 部材×損傷タイプ分析図を生成する。
出力先: paper_damage_vlm/1_Methodology/figures_vlm/
"""

import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────
# 1. データ読み込み
# ─────────────────────────────────────────
CSV = "data/v03_fine_tuning/evaluations/v063_scoring_results.csv"
df = pd.read_csv(CSV)
pass_df = df[df["quality_verdict"] == "PASS"].copy()
fail_df = df[df["quality_verdict"] == "FAIL"].copy()
print(f"PASS: {len(pass_df)}  FAIL: {len(fail_df)}")

OUT_DIR = Path("paper_damage_vlm/1_Methodology/figures_vlm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# 共通スタイル
# ─────────────────────────────────────────
BLUE   = "#3a7ebf"
GREEN  = "#2ca02c"
ORANGE = "#e07b39"
RED    = "#d62728"
GRAY   = "#888888"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})


# ══════════════════════════════════════════════════════════════════
# Figure A:  Priority Score & Quality Metrics — Violin Plots
# ══════════════════════════════════════════════════════════════════
fig_a, axes_a = plt.subplots(1, 3, figsize=(13, 5.5))
fig_a.suptitle(
    "Priority Score and Quality Metrics: PASS 727 Samples (v0.6.3, 3k Model)",
    fontsize=12, fontweight="bold", y=1.01,
)

# ── Panel 1: Priority Score (saturated) ──────────────────────────
ax = axes_a[0]
scores = pass_df["priority_score"].values
vp = ax.violinplot([scores], positions=[1], showmedians=True,
                   showextrema=True, widths=0.5)
for pc in vp["bodies"]:
    pc.set_facecolor(ORANGE)
    pc.set_alpha(0.6)
vp["cmedians"].set_color("red")
vp["cmins"].set_color(GRAY)
vp["cmaxes"].set_color(GRAY)
vp["cbars"].set_color(GRAY)

# Priority level reference lines
for lvl, score, label in [
    (1, 0.2,  "P1 (obs.)"),
    (2, 0.35, "P2"),
    (3, 0.5,  "P3"),
    (4, 0.7,  "P4"),
    (5, 0.85, "P5 (imm.)"),
]:
    ax.axhline(score, color=BLUE, alpha=0.3, linewidth=0.8, linestyle="--")
    ax.text(1.32, score, label, va="center", fontsize=7.5, color=BLUE)

ax.set_xticks([1])
ax.set_xticklabels(["PASS\n(n=727)"])
ax.set_ylabel("Priority Score")
ax.set_ylim(0, 1)
ax.set_title("Priority Score Distribution\n(Scoring Saturation)")
ax.text(1, 0.54, f"All = 0.54\n(Level 3)", ha="center", va="bottom",
        fontsize=9, color=RED, fontweight="bold")
ax.annotate("Scoring\nsaturation:\n100% Level 3",
            xy=(1, 0.54), xytext=(1.4, 0.75),
            fontsize=8, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

# ── Panel 2: Cosine Similarity (PASS vs FAIL) ────────────────────
ax = axes_a[1]
cos_pass = pass_df["cosine_similarity"].dropna().values
cos_fail = fail_df["cosine_similarity"].dropna().values
data_cos = [cos_pass, cos_fail]
vp2 = ax.violinplot(data_cos, positions=[1, 2], showmedians=True,
                    showextrema=True, widths=0.5)
colors2 = [GREEN, RED]
for pc, c in zip(vp2["bodies"], colors2):
    pc.set_facecolor(c)
    pc.set_alpha(0.55)
vp2["cmedians"].set_color("black")
vp2["cmins"].set_color(GRAY)
vp2["cmaxes"].set_color(GRAY)
vp2["cbars"].set_color(GRAY)

# Tier lines
for thr, lbl in [(0.85, "Excellent"), (0.70, "Good"),
                 (0.60, "Acceptable"), (0.50, "Poor")]:
    ax.axhline(thr, color=BLUE, alpha=0.25, linewidth=0.8, linestyle=":")
    ax.text(2.35, thr, lbl, va="center", fontsize=7, color=BLUE)

ax.set_xticks([1, 2])
ax.set_xticklabels([f"PASS\n(n={len(cos_pass)})", f"FAIL\n(n={len(cos_fail)})"])
ax.set_ylabel("Cosine Similarity")
ax.set_ylim(0.25, 1.0)
ax.set_title("Cosine Similarity Distribution\n(PASS vs FAIL)")

# Median annotations
for pos, data, color in zip([1, 2], data_cos, colors2):
    med = np.median(data)
    ax.text(pos, med + 0.01, f"med={med:.3f}", ha="center",
            fontsize=8, color=color, fontweight="bold")

# ── Panel 3: Token Count (PASS vs FAIL) ──────────────────────────
ax = axes_a[2]
tok_pass = pass_df["token_count"].dropna().values
tok_fail = fail_df["token_count"].dropna().values
vp3 = ax.violinplot([tok_pass, tok_fail], positions=[1, 2], showmedians=True,
                    showextrema=True, widths=0.5)
for pc, c in zip(vp3["bodies"], [GREEN, RED]):
    pc.set_facecolor(c)
    pc.set_alpha(0.55)
vp3["cmedians"].set_color("black")
vp3["cmins"].set_color(GRAY)
vp3["cmaxes"].set_color(GRAY)
vp3["cbars"].set_color(GRAY)

ax.axhline(98,  color=RED, linestyle="--", linewidth=1.2, alpha=0.7,
           label=r"$\theta_{low}=98$")
ax.axhline(202, color=RED, linestyle=":",  linewidth=1.2, alpha=0.7,
           label=r"$\theta_{high}=202$")
ax.text(2.32,  98, r"$\theta_{low}$",  va="center", fontsize=8, color=RED)
ax.text(2.32, 202, r"$\theta_{high}$", va="center", fontsize=8, color=RED)

ax.set_xticks([1, 2])
ax.set_xticklabels([f"PASS\n(n={len(tok_pass)})", f"FAIL\n(n={len(tok_fail)})"])
ax.set_ylabel("Token Count")
ax.set_title("Output Token Count\n(PASS vs FAIL)")

for pos, data, color in zip([1, 2], [tok_pass, tok_fail], [GREEN, RED]):
    med = np.median(data)
    ax.text(pos, med + 2, f"med={int(med)}", ha="center",
            fontsize=8, color=color, fontweight="bold")

fig_a.tight_layout()
out_a = "10_priority_score_violin.png"
fig_a.savefig(OUT_DIR / out_a, dpi=150, bbox_inches="tight")
print(f"Saved: {out_a}")
plt.close(fig_a)


# ══════════════════════════════════════════════════════════════════
# Figure B:  Member × Damage Type Breakdown (PASS 727 samples)
# ══════════════════════════════════════════════════════════════════

# ── 部材・損傷タイプのキーワード定義 ────────────────────────────
MEMBERS = {
    "主桁":     "Main Girder",
    "横桁":     "Cross Beam",
    "床版":     "Deck Slab",
    "舗装":     "Pavement",
    "高欄":     "Railing/Parapet",
    "地覆":     "Curb",
    "伸縮装置": "Expansion Joint",
    "排水":     "Drainage",
    "支承":     "Bearing",
    "親柱":     "Post",
    "橋脚":     "Pier",
    "橋台":     "Abutment",
    "壁高欄":   "Wall Railing",
}

DAMAGES = {
    "ひびわれ":   "Crack",
    "鉄筋露出":   "Rebar Exposure",
    "うき":       "Spalling",
    "剥離":       "Delamination",
    "腐食":       "Corrosion",
    "欠損":       "Defect/Missing",
    "遊離石灰":   "Lime Efflorescence",
    "土砂詰り":   "Soil Clogging",
    "破損":       "Fracture",
    "変形":       "Deformation",
    "板厚減少":   "Thickness Loss",
    "劣化":       "Deterioration",
}

def count_keyword(series: pd.Series, keyword: str) -> int:
    return series.str.contains(keyword, na=False, regex=False).sum()

texts = pass_df["assistant_text"].fillna("").astype(str)

# 部材カウント
member_counts = {
    v: count_keyword(texts, k) for k, v in MEMBERS.items()
}
member_ser = pd.Series(member_counts).sort_values(ascending=False)
member_ser = member_ser[member_ser > 0]

# 損傷タイプカウント
damage_counts = {
    v: count_keyword(texts, k) for k, v in DAMAGES.items()
}
damage_ser = pd.Series(damage_counts).sort_values(ascending=False)
damage_ser = damage_ser[damage_ser > 0]

# 部材 × 損傷タイプ 共起マトリクス (上位のもの)
top_members = member_ser.head(8).index.tolist()
top_damages = damage_ser.head(8).index.tolist()

matrix = np.zeros((len(top_members), len(top_damages)), dtype=int)
jp_members = {v: k for k, v in MEMBERS.items()}
jp_damages  = {v: k for k, v in DAMAGES.items()}

for i, mem_en in enumerate(top_members):
    mem_jp = jp_members[mem_en]
    mask = texts.str.contains(mem_jp, na=False, regex=False)
    sub = texts[mask]
    for j, dmg_en in enumerate(top_damages):
        dmg_jp = jp_damages[dmg_en]
        matrix[i, j] = sub.str.contains(dmg_jp, na=False, regex=False).sum()

# ── Plot ─────────────────────────────────────────────────────────
fig_b = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 2, figure=fig_b, hspace=0.45, wspace=0.4)

# Panel 1: Member bar chart
ax_m = fig_b.add_subplot(gs[0, 0])
pct_m = member_ser / len(pass_df) * 100
colors_m = [BLUE if p >= 50 else (GREEN if p >= 20 else ORANGE)
            for p in pct_m.values]
bars = ax_m.barh(member_ser.index[::-1], member_ser.values[::-1],
                 color=colors_m[::-1], edgecolor="white", linewidth=0.5)
for bar, cnt, pct in zip(bars, member_ser.values[::-1], pct_m.values[::-1]):
    ax_m.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
              f"{cnt} ({pct:.1f}%)", va="center", fontsize=8)
ax_m.set_xlabel("Count (n=727 PASS samples)")
ax_m.set_title("Structural Member Mentions\n(PASS 727 samples)", fontweight="bold")
ax_m.set_xlim(0, member_ser.max() * 1.25)
ax_m.grid(axis="x", alpha=0.3)

# Panel 2: Damage type bar chart
ax_d = fig_b.add_subplot(gs[0, 1])
pct_d = damage_ser / len(pass_df) * 100
colors_d = [RED if p >= 50 else (ORANGE if p >= 20 else GRAY)
            for p in pct_d.values]
bars2 = ax_d.barh(damage_ser.index[::-1], damage_ser.values[::-1],
                  color=colors_d[::-1], edgecolor="white", linewidth=0.5)
for bar, cnt, pct in zip(bars2, damage_ser.values[::-1], pct_d.values[::-1]):
    ax_d.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
              f"{cnt} ({pct:.1f}%)", va="center", fontsize=8)
ax_d.set_xlabel("Count (n=727 PASS samples)")
ax_d.set_title("Damage Type Mentions\n(PASS 727 samples)", fontweight="bold")
ax_d.set_xlim(0, damage_ser.max() * 1.25)
ax_d.grid(axis="x", alpha=0.3)

# Panel 3: Heatmap (bottom, full width)
ax_h = fig_b.add_subplot(gs[1, :])
im = ax_h.imshow(matrix, cmap="YlOrRd", aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax_h, label="Co-occurrence count")
ax_h.set_xticks(range(len(top_damages)))
ax_h.set_xticklabels(top_damages, rotation=35, ha="right", fontsize=9)
ax_h.set_yticks(range(len(top_members)))
ax_h.set_yticklabels(top_members, fontsize=9)
ax_h.set_title(
    "Member × Damage Type Co-occurrence Heatmap (Top 8 each, PASS 727 samples)",
    fontweight="bold",
)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        if val > 0:
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax_h.text(j, i, str(val), ha="center", va="center",
                      fontsize=8, color=color, fontweight="bold")

fig_b.suptitle(
    "Structural Member and Damage Type Analysis — PASS 727 Samples (v0.6.3, 3k Model)",
    fontsize=12, fontweight="bold",
)

out_b = "10_member_damage_analysis.png"
fig_b.savefig(OUT_DIR / out_b, dpi=150, bbox_inches="tight")
print(f"Saved: {out_b}")
plt.close(fig_b)

# ─────────────────────────────────────────
# サマリー出力
# ─────────────────────────────────────────
print("\n=== Member mentions (PASS 727) ===")
for k, v in member_ser.items():
    print(f"  {k:<25}: {v:3d}  ({v/len(pass_df)*100:.1f}%)")

print("\n=== Damage mentions (PASS 727) ===")
for k, v in damage_ser.items():
    print(f"  {k:<25}: {v:3d}  ({v/len(pass_df)*100:.1f}%)")

print("\n=== Co-occurrence matrix (top 8 × 8) ===")
mat_df = pd.DataFrame(matrix, index=top_members, columns=top_damages)
print(mat_df.to_string())
print("\nDone.")
