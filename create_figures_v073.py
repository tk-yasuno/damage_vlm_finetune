"""
create_figures_v073.py
────────────────────────────────────────────────────────────
v0.7.3 Quality Guard Recalibration の可視化

生成する図:
  Figure 15: Token count distribution with θlow threshold comparison
  Figure 16: Quality Guard rejection breakdown (reason code)
  Figure 17: Priority level distribution (v0.7.3)
"""

from __future__ import annotations
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── 出力先 ────────────────────────────────────────────────
OUT_DIR = Path("paper_damage_vlm/1_Methodology/figures_vlm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── データ読み込み ──────────────────────────────────────────
df = pd.read_csv(
    "data/v07_fine_tuning/pipeline_v073/pipeline_results_v073.csv",
    encoding="utf-8",
)
n = len(df)

# ── スタイル ───────────────────────────────────────────────
BLUE   = "#2196F3"
GREEN  = "#4CAF50"
RED    = "#F44336"
ORANGE = "#FF9800"
GREY   = "#9E9E9E"
PURPLE = "#9C27B0"

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi":     150,
})


# ══════════════════════════════════════════════════════════
# Figure 15: Token count distribution + θlow overlay
# ══════════════════════════════════════════════════════════
def fig15_token_distribution():
    fig, ax = plt.subplots(figsize=(7, 4))

    tok = df["token_count"].values

    # histogram
    bins = range(0, 210, 10)
    ax.hist(tok, bins=bins, color=BLUE, alpha=0.75, edgecolor="white",
            linewidth=0.5, label=f"v0.7.3 predictions (n={n})")

    # θlow v0.6.3 (old)
    ax.axvline(98, color=RED, linestyle="--", linewidth=1.8,
               label=r"$\theta_{low}^{v0.6.3}$ = 98 (old)")
    # θlow v0.7.3 (new)
    ax.axvline(30, color=GREEN, linestyle="-", linewidth=2.0,
               label=r"$\theta_{low}^{v0.7.3}$ = 30 (recalibrated)")
    # θhigh
    ax.axvline(200, color=ORANGE, linestyle=":", linewidth=1.8,
               label=r"$\theta_{high}^{v0.7.3}$ = 200")

    # annotation: area rejected by old θlow
    ymax = ax.get_ylim()[1]
    ax.axvspan(30, 98, color=RED, alpha=0.08,
               label="Reclaimed region (30–98)")

    ax.set_xlabel("Token count (approx.)")
    ax.set_ylabel("Number of predictions")
    ax.set_title("Figure 15  Token Count Distribution with Recalibrated Thresholds\n"
                 "(v0.7.3, pairdata-v2, 3k model, n=800)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, 210)

    # stats box
    stats_txt = (f"mean={tok.mean():.0f}  "
                 f"p5={np.percentile(tok,5):.0f}  "
                 f"p95={np.percentile(tok,95):.0f}  "
                 f"max={tok.max():.0f}")
    ax.text(0.02, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.tight_layout()
    out = OUT_DIR / "15_v073_token_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════
# Figure 16: Quality Guard breakdown (PASS / reason codes)
# ══════════════════════════════════════════════════════════
def fig16_guard_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # ── left: PASS / FAIL pie ──────────────────────────────
    n_pass = (df["quality_verdict"] == "PASS").sum()
    n_fail = (df["quality_verdict"] == "FAIL").sum()
    ax = axes[0]
    wedge_colors = [GREEN, RED]
    ax.pie([n_pass, n_fail],
           labels=[f"PASS\n{n_pass} ({n_pass/n*100:.1f}%)",
                   f"FAIL\n{n_fail} ({n_fail/n*100:.1f}%)"],
           colors=wedge_colors,
           startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=1.5),
           textprops=dict(fontsize=10))
    ax.set_title(f"(a) Quality Guard Verdict\n(n={n})", fontsize=11)

    # ── right: reason code bar ─────────────────────────────
    ax2 = axes[1]
    rc_counts = df["reason_code"].value_counts()
    colors_rc = {
        "High Quality":                 GREEN,
        "Too short / Error output":     RED,
        "No damage keywords detected":  ORANGE,
        "Not recognized from only image": GREY,
    }
    bar_colors = [colors_rc.get(k, BLUE) for k in rc_counts.index]
    bars = ax2.barh(rc_counts.index, rc_counts.values,
                    color=bar_colors, edgecolor="white", linewidth=0.7)
    for bar, val in zip(bars, rc_counts.values):
        ax2.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                 f"{val} ({val/n*100:.1f}%)",
                 va="center", fontsize=9)
    ax2.set_xlabel("Count")
    ax2.set_title("(b) Reason Code Breakdown", fontsize=11)
    ax2.set_xlim(0, rc_counts.max() * 1.25)
    ax2.invert_yaxis()

    fig.suptitle("Figure 16  Quality Guard Recalibration Results (v0.7.3, n=800)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "16_v073_guard_breakdown.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════
# Figure 17: Priority level distribution (v0.7.3 PASS)
# ══════════════════════════════════════════════════════════
def fig17_priority_distribution():
    pass_df = df[df["quality_verdict"] == "PASS"].copy()
    n_pass = len(pass_df)

    lv_counts = pass_df["priority_level"].value_counts().sort_index()
    levels    = lv_counts.index.tolist()
    counts    = lv_counts.values.tolist()
    pcts      = [c / n_pass * 100 for c in counts]

    level_colors = {3: BLUE, 4: ORANGE, 5: RED}
    bar_colors   = [level_colors.get(lv, GREY) for lv in levels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(lv) for lv in levels], counts,
                  color=bar_colors, edgecolor="white", linewidth=0.8, width=0.5)
    for bar, cnt, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 4,
                f"{cnt}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Priority Level")
    ax.set_ylabel("Number of images")
    ax.set_title("Figure 17  Priority Level Distribution (v0.7.3, PASS only)\n"
                 f"n={n_pass} of {n} total (Rejection rate {(n-n_pass)/n*100:.1f}%)")
    ax.set_ylim(0, max(counts) * 1.2)

    legend_handles = [
        mpatches.Patch(color=BLUE,   label="Level 3: Planned maintenance"),
        mpatches.Patch(color=ORANGE, label="Level 4: Priority repair"),
        mpatches.Patch(color=RED,    label="Level 5: Immediate repair"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left")

    fig.tight_layout()
    out = OUT_DIR / "17_v073_priority_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── 実行 ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating v0.7.3 figures...")
    fig15_token_distribution()
    fig16_guard_breakdown()
    fig17_priority_distribution()
    print("Done.")
