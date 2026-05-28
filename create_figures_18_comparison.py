"""
create_figures_18_comparison.py
pairdata v1 (v0.6.3) vs v2 (v0.7.3) の部材・損傷タイプ比較図を生成する。
出力先: paper_damage_vlm/1_Methodology/figures_vlm/
Figure 18: Member mention / Damage type / Co-occurrence (v0.6.3 left, v0.7.3 right)
Figure 19: Priority Level distribution comparison
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────
# 共通スタイル
# ─────────────────────────────────────────
BLUE   = "#3a7ebf"
GREEN  = "#2ca02c"
ORANGE = "#e07b39"
RED    = "#d62728"
GRAY   = "#888888"
PURPLE = "#9467bd"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
})

OUT_DIR = Path("paper_damage_vlm/1_Methodology/figures_vlm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# キーワード定義 (Figure 10 と同じ)
# ─────────────────────────────────────────
MEMBERS = {
    "主桁":     "Main Girder",
    "横桁":     "Cross Beam",
    "床版":     "Deck Slab",
    "舗装":     "Pavement",
    "高欄":     "Railing",
    "地覆":     "Curb",
    "伸縮装置": "Expan.Joint",
    "排水":     "Drainage",
    "支承":     "Bearing",
    "橋脚":     "Pier",
    "橋台":     "Abutment",
    "壁高欄":   "Wall Rail.",
}

DAMAGES = {
    "ひびわれ":   "Crack",
    "鉄筋露出":   "Rebar Exp.",
    "うき":       "Spalling",
    "剥離":       "Delamination",
    "腐食":       "Corrosion",
    "欠損":       "Defect",
    "遊離石灰":   "Lime Effl.",
    "土砂詰り":   "Soil Clog",
    "破損":       "Fracture",
    "変形":       "Deformation",
    "劣化":       "Deterioration",
}

def extract_assistant(text: str) -> str:
    """USER:...ASSISTANT: ... 形式からASSISTANTパートを抽出"""
    m = re.search(r'ASSISTANT:\s*(.*)', str(text), re.DOTALL)
    return m.group(1).strip() if m else str(text)

def count_keyword(series: pd.Series, keyword: str) -> int:
    return series.str.contains(keyword, na=False, regex=False).sum()

def build_member_damage(texts: pd.Series, n_total: int):
    """部材カウント・損傷タイプカウント・共起マトリクスを返す"""
    member_counts = {v: count_keyword(texts, k) for k, v in MEMBERS.items()}
    damage_counts = {v: count_keyword(texts, k) for k, v in DAMAGES.items()}

    member_ser = pd.Series(member_counts).sort_values(ascending=False)
    member_ser = member_ser[member_ser > 0]
    damage_ser = pd.Series(damage_counts).sort_values(ascending=False)
    damage_ser = damage_ser[damage_ser > 0]

    top_members = member_ser.head(8).index.tolist()
    top_damages = damage_ser.head(8).index.tolist()

    jp_members = {v: k for k, v in MEMBERS.items()}
    jp_damages  = {v: k for k, v in DAMAGES.items()}

    matrix = np.zeros((len(top_members), len(top_damages)), dtype=int)
    for i, mem_en in enumerate(top_members):
        mem_jp = jp_members.get(mem_en, mem_en)
        mask = texts.str.contains(mem_jp, na=False, regex=False)
        sub = texts[mask]
        for j, dmg_en in enumerate(top_damages):
            dmg_jp = jp_damages.get(dmg_en, dmg_en)
            matrix[i, j] = sub.str.contains(dmg_jp, na=False, regex=False).sum()

    return member_ser, damage_ser, top_members, top_damages, matrix

# ─────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────
# v0.6.3 (pairdata_v1): PASS=727
df06 = pd.read_csv(
    "data/v03_fine_tuning/evaluations/v06_scoring_results_ckpt800.csv"
)
pass06 = df06[df06["quality_verdict"] == "PASS"].copy()
pass06["text"] = pass06["prediction"].apply(extract_assistant)
N06 = len(pass06)
print(f"v0.6.3 PASS: {N06}")

# v0.7.3 (pairdata_v2): PASS=787
df073 = pd.read_csv(
    "data/v07_fine_tuning/pipeline_v073/pipeline_results_v073.csv"
)
pass073 = df073[df073["quality_verdict"] == "PASS"].copy()
pass073["text"] = pass073["assistant_text"].fillna("").astype(str)
N073 = len(pass073)
print(f"v0.7.3 PASS: {N073}")

# ─────────────────────────────────────────
# 各データセットの部材・損傷タイプ分析
# ─────────────────────────────────────────
mem06, dmg06, tm06, td06, mat06 = build_member_damage(pass06["text"], N06)
mem073, dmg073, tm073, td073, mat073 = build_member_damage(pass073["text"], N073)

print("v0.6.3 top members:", mem06.head(5).to_dict())
print("v0.6.3 top damages:", dmg06.head(5).to_dict())
print("v0.7.3 top members:", mem073.head(5).to_dict())
print("v0.7.3 top damages:", dmg073.head(5).to_dict())


# ══════════════════════════════════════════════════════════════════
# Figure 18: 部材・損傷タイプ比較（2列 × 3行）
# ══════════════════════════════════════════════════════════════════
fig18 = plt.figure(figsize=(16, 14))
gs18 = GridSpec(3, 2, figure=fig18, hspace=0.55, wspace=0.38,
                height_ratios=[1, 1, 1.3])
fig18.suptitle(
    "Structural Member and Damage Type Analysis: pairdata v1 (v0.6.3) vs v2 (v0.7.3)",
    fontsize=12, fontweight="bold", y=0.99,
)

COLORS_06  = "#d62728"   # red: v0.6.3 (mode collapse)
COLORS_073 = "#3a7ebf"   # blue: v0.7.3 (improved)

def plot_member_bar(ax, member_ser, n_total, color, title):
    pct = member_ser / n_total * 100
    bars = ax.barh(
        member_ser.index[::-1], member_ser.values[::-1],
        color=[color if p >= 50 else (ORANGE if p >= 20 else GRAY)
               for p in pct.values[::-1]],
        edgecolor="white", linewidth=0.5
    )
    for bar, cnt, p in zip(bars, member_ser.values[::-1], pct.values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{cnt} ({p:.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("Count")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(0, n_total * 1.15)
    return ax

def plot_damage_bar(ax, damage_ser, n_total, color, title):
    pct = damage_ser / n_total * 100
    colors_d = [color if p >= 50 else (ORANGE if p >= 20 else GRAY)
                for p in pct.values]
    bars = ax.bar(
        damage_ser.index, damage_ser.values,
        color=colors_d, edgecolor="white", linewidth=0.5
    )
    for bar, cnt, p in zip(bars, damage_ser.values, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{cnt}\n({p:.1f}%)", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylim(0, n_total * 1.2)
    ax.tick_params(axis="x", rotation=35)
    return ax

def plot_heatmap(ax, top_members, top_damages, matrix, title):
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(top_damages)))
    ax.set_xticklabels(top_damages, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_members)))
    ax.set_yticklabels(top_members, fontsize=8)
    for i in range(len(top_members)):
        for j in range(len(top_damages)):
            val = matrix[i, j]
            if val > 0:
                text_color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    ax.set_title(title, fontsize=10, fontweight="bold")
    return ax

# Row 0: Member frequency
ax00 = fig18.add_subplot(gs18[0, 0])
plot_member_bar(ax00, mem06,  N06,  COLORS_06,
                f"Member Frequency — v0.6.3 pairdata_v1\n(PASS n={N06})")

ax01 = fig18.add_subplot(gs18[0, 1])
plot_member_bar(ax01, mem073, N073, COLORS_073,
                f"Member Frequency — v0.7.3 pairdata_v2\n(PASS n={N073})")

# Row 1: Damage type frequency
ax10 = fig18.add_subplot(gs18[1, 0])
plot_damage_bar(ax10, dmg06,  N06,  COLORS_06,
                f"Damage Type Frequency — v0.6.3\n(PASS n={N06})")

ax11 = fig18.add_subplot(gs18[1, 1])
plot_damage_bar(ax11, dmg073, N073, COLORS_073,
                f"Damage Type Frequency — v0.7.3\n(PASS n={N073})")

# Row 2: Co-occurrence heatmap
ax20 = fig18.add_subplot(gs18[2, 0])
plot_heatmap(ax20, tm06, td06, mat06,
             "Member × Damage Co-occurrence — v0.6.3\n(pairdata_v1, mode collapse)")

ax21 = fig18.add_subplot(gs18[2, 1])
plot_heatmap(ax21, tm073, td073, mat073,
             "Member × Damage Co-occurrence — v0.7.3\n(pairdata_v2, improved)")

out18 = OUT_DIR / "18_member_damage_comparison.png"
fig18.savefig(out18, dpi=150, bbox_inches="tight")
print(f"Saved: {out18}")
plt.close(fig18)


# ══════════════════════════════════════════════════════════════════
# Figure 19: Priority Level 分布比較（横並び棒グラフ）
# ══════════════════════════════════════════════════════════════════
fig19, ax19 = plt.subplots(figsize=(9, 5))
fig19.suptitle(
    "Priority Level Distribution: pairdata v1 (v0.6.3) vs v2 (v0.7.3)",
    fontsize=12, fontweight="bold",
)

# v0.6.3: all Level 3 (priority_level=0 means rule-only → Level 3 equivalent)
# v0.7.3: Level 3/4/5
levels = [1, 2, 3, 4, 5]
level_labels = ["L1", "L2", "L3", "L4", "L5"]
level_colors = ["#2ca02c", "#98df8a", "#ffbb78", "#e07b39", "#d62728"]

# v0.6.3: all PASS → Level 3 (score=0.54, confirmed from v063 ckpt704)
cnt06 = {3: N06}   # 727件すべてLevel 3
# v0.7.3
cnt073 = pass073["priority_level"].value_counts().sort_index().to_dict()

x = np.arange(len(levels))
width = 0.35

bars06  = [cnt06.get(l,  0) / N06  * 100 for l in levels]
bars073 = [cnt073.get(l, 0) / N073 * 100 for l in levels]

b1 = ax19.bar(x - width/2, bars06,  width, label=f"v0.6.3 pairdata_v1 (n={N06})",
              color=COLORS_06,  alpha=0.75, edgecolor="white")
b2 = ax19.bar(x + width/2, bars073, width, label=f"v0.7.3 pairdata_v2 (n={N073})",
              color=COLORS_073, alpha=0.75, edgecolor="white")

for bar, pct in zip(b1, bars06):
    if pct > 1:
        ax19.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  f"{pct:.1f}%", ha="center", va="bottom", fontsize=8,
                  color=COLORS_06, fontweight="bold")

for bar, pct in zip(b2, bars073):
    if pct > 1:
        ax19.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  f"{pct:.1f}%", ha="center", va="bottom", fontsize=8,
                  color=COLORS_073, fontweight="bold")

ax19.set_xticks(x)
ax19.set_xticklabels([
    "Level 1\n(Observ.)", "Level 2\n(Minor)", "Level 3\n(Moderate)",
    "Level 4\n(Serious)", "Level 5\n(Urgent)"
])
ax19.set_ylabel("Percentage of PASS samples (%)")
ax19.set_ylim(0, 105)
ax19.legend(loc="upper right", fontsize=9)
ax19.grid(axis="y", alpha=0.3)

# 注釈
ax19.annotate(
    "Scoring saturation:\n100% Level 3 (score=0.54)",
    xy=(2 - width/2, 100), xytext=(2.5, 80),
    fontsize=8.5, color=COLORS_06, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=COLORS_06, lw=1.2)
)
ax19.annotate(
    "Normalized:\nL3: 2.7%, L4: 82.7%, L5: 14.6%",
    xy=(3 + width/2, bars073[3]),
    xytext=(3.5, 60),
    fontsize=8.5, color=COLORS_073, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=COLORS_073, lw=1.2)
)

fig19.tight_layout()
out19 = OUT_DIR / "19_priority_comparison.png"
fig19.savefig(out19, dpi=150, bbox_inches="tight")
print(f"Saved: {out19}")
plt.close(fig19)

print("\nDone. Generated Figure 18 and Figure 19.")
