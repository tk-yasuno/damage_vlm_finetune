"""
pipeline_v073.py
──────────────────────────────────────────────────────────────
v0.7.3: Quality Guard Recalibration + Scoring for 3k QLoRA model (pairdata_v2)

Changes from v0.6.3:
  - Input  : v0.7.2 3k inference results (pairdata_v2, n=800, denoised)
  - Thresholds RECALIBRATED for denoised outputs:
      θlow  = 30   (was 98)  — pairdata_v2 outputs are shorter (mean~103 tok)
      θhigh = 200  (was 202) — no outputs exceed 196 tokens in v0.7.2 data
  - Structuring: keyword-based (no Swallow-8B LLM required)
      Reason: pairdata_v2 outputs are concise and pattern-consistent;
              LLM structuring overhead (~2h) is deferred to v0.7.4+
  - Output : data/v07_fine_tuning/pipeline_v073/pipeline_results_v073.csv

Recalibration rationale:
  v0.7.2 3k token distribution (n=800):
    mean=103, std=29, p5=60, p95=155, max=196
  Old θlow=98 would reject ~50% of valid outputs (below mean).
  New θlow=30 captures only clearly-failed outputs (ERROR: file not found, n=13).

Usage:
    python pipeline_v073.py              # full run (n=800)
    python pipeline_v073.py --limit 20  # quick test
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# ── 定数 ──────────────────────────────────────────────────

INPUT_CSV  = "data/v07_fine_tuning/evaluations_unsloth_v072/inference_results_v07_3k.csv"
OUTPUT_DIR = "data/v07_fine_tuning/pipeline_v073"
OUTPUT_CSV = f"{OUTPUT_DIR}/pipeline_results_v073.csv"

# v0.7.3 再キャリブレーション済み閾値
LOW_TOKEN_THRESHOLD  = 30   # was 98 in v0.6.3
HIGH_TOKEN_THRESHOLD = 200  # was 202 in v0.6.3

DAMAGE_KEYWORDS = [
    "ひび割れ", "クラック", "剥離", "剥落", "漏水", "さび", "錆",
    "腐食", "変形", "沈下", "損傷", "欠損", "鉄筋", "断面",
    "亀裂", "ひびわれ", "スパリング", "露出",
]
VAGUE_KEYWORDS = [
    "確認できない", "認識できない", "判断できない",
    "見えない", "不明", "わからない",
]


# ── ユーティリティ ─────────────────────────────────────────

def count_tokens(text: str) -> int:
    """CJK文字 + ASCII ブロック数でトークン数近似"""
    cjk   = len(re.findall(r"[\u3000-\u9fff\uf900-\uffef]", text))
    ascii_ = len(text.encode("ascii", errors="ignore").split())
    return cjk + ascii_


def get_assistant_text(prediction: str) -> str:
    m = re.search(r"ASSISTANT\s*:\s*(.+)", prediction, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else prediction.strip()


def has_damage_keywords(text: str) -> bool:
    return any(kw in text for kw in DAMAGE_KEYWORDS)


def has_vague_keywords(text: str) -> bool:
    return any(kw in text for kw in VAGUE_KEYWORDS)


# ── キーワードベース構造化（Swallow不要） ────────────────────

def keyword_structure(text: str) -> tuple[str, str, str, str]:
    """
    キーワードマッチングにより damage_type / severity / location / risk を推定。
    pairdata_v2 出力は簡潔なパターンが多いため有効。
    """
    # damage_type (複数ヒット時は優先順位が高いものを返す)
    if "鉄筋露出" in text or ("鉄筋" in text and "露出" in text):
        damage_type = "rebar_exposure"
    elif "断面欠損" in text or ("断面" in text and "欠損" in text):
        damage_type = "section_loss"
    elif "剥離" in text or "剥落" in text or "スパリング" in text:
        damage_type = "spalling"
    elif "錆" in text or "さび" in text or "腐食" in text:
        damage_type = "corrosion"
    elif "ひびわれ" in text or "ひび割れ" in text or "クラック" in text or "亀裂" in text:
        damage_type = "crack"
    else:
        damage_type = "unknown"

    # severity
    high_words = ["著しい", "激しい", "重大", "進行中", "大規模", "大きな", "多数"]
    low_words  = ["軽微", "初期", "小規模", "わずか"]
    if any(w in text for w in high_words):
        severity = "high"
    elif any(w in text for w in low_words):
        severity = "low"
    else:
        severity = "medium"

    # location
    if "主桁" in text or "桁" in text:
        location = "girder"
    elif "床版" in text or "舗装" in text:
        location = "deck"
    elif "支承" in text:
        location = "bearing"
    elif "橋脚" in text or "橋台" in text or "躯体" in text:
        location = "pier"
    elif "高欄" in text or "防護柵" in text or "手すり" in text or "地覆" in text:
        location = "railing"
    else:
        location = "unknown"

    # risk (damage_type と severity から推定)
    if damage_type in ("rebar_exposure", "section_loss"):
        risk = "structural"
    elif damage_type in ("corrosion", "crack"):
        risk = "durability"
    elif damage_type == "spalling":
        risk = "aesthetic"
    else:
        risk = "unknown"

    return damage_type, severity, location, risk


# ── Quality Guard ──────────────────────────────────────────

def apply_quality_guard(
    text: str,
    low_threshold: int = LOW_TOKEN_THRESHOLD,
    high_threshold: int = HIGH_TOKEN_THRESHOLD,
) -> tuple[str, str]:
    """
    Returns (verdict, reason_code)
    verdict: "PASS" | "FAIL"
    """
    tok = count_tokens(text)

    if tok < low_threshold:
        return "FAIL", "Too short / Error output"
    if tok > high_threshold:
        return "FAIL", "Too long / Possible repetition"
    if has_vague_keywords(text) and not has_damage_keywords(text):
        return "FAIL", "Not recognized from only image"
    if not has_damage_keywords(text):
        return "FAIL", "No damage keywords detected"

    return "PASS", "High Quality"


# ── Priority Scorer ───────────────────────────────────────

def load_scorer():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from src.scoring.priority_scorer import PriorityScorer, ScoringConfig
    return PriorityScorer(ScoringConfig(rules_file="models/scoring_rules.yaml"))


def score_row(scorer, damage_type, severity, location, risk) -> dict:
    try:
        ps = scorer.calculate_score(
            damage_type=damage_type or "unknown",
            severity=severity or "medium",
            location=location or "unknown",
            risk=risk or "unknown",
        )
        return {
            "priority_score":       round(ps.raw_score, 4),
            "priority_level":       ps.priority_level,
            "priority_description": ps.priority_description,
        }
    except Exception as e:
        return {
            "priority_score":       float("nan"),
            "priority_level":       1,
            "priority_description": f"Scoring error: {e}",
        }


# ── メイン処理 ────────────────────────────────────────────

def run(limit: Optional[int] = None) -> None:
    t0 = time.time()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"[Load] {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    if limit:
        df = df.head(limit)
    n = len(df)
    print(f"  Rows: {n}")
    print(f"  θlow={LOW_TOKEN_THRESHOLD}, θhigh={HIGH_TOKEN_THRESHOLD}")

    scorer = load_scorer()

    records = []
    n_pass = n_fail = 0

    for i, row in enumerate(df.itertuples(index=False)):
        ast = get_assistant_text(str(row.prediction))
        tok = count_tokens(ast)
        verdict, reason = apply_quality_guard(ast)

        if verdict == "PASS":
            n_pass += 1
            dm, sv, loc, rsk = keyword_structure(ast)
            sc = score_row(scorer, dm, sv, loc, rsk)
        else:
            n_fail += 1
            dm = sv = loc = rsk = None
            sc = {
                "priority_score":       float("nan"),
                "priority_level":       0,
                "priority_description": "No Score due to Low Quality",
            }

        records.append({
            "image_id":          row.image_id,
            "image_path":        row.image_path,
            "assistant_text":    ast,
            "token_count":       tok,
            "cosine_similarity": getattr(row, "cosine_similarity", float("nan")),
            "quality_verdict":   verdict,
            "reason_code":       reason,
            "damage_type":       dm,
            "severity_level":    sv,
            "location":          loc,
            "risk_factor":       rsk,
            **sc,
        })

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  [{i+1:4d}/{n}] PASS={n_pass} FAIL={n_fail} "
                  f"({time.time()-t0:.1f}s)")

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n[Saved] {OUTPUT_CSV}")

    # ── サマリー ───────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  v0.7.3 Pipeline Complete")
    print(f"{'='*60}")
    print(f"  Total  : {n}")
    print(f"  PASS   : {n_pass} ({n_pass/n*100:.1f}%)")
    print(f"  FAIL   : {n_fail} ({n_fail/n*100:.1f}%)")
    print(f"  Elapsed: {elapsed:.1f}s  ({elapsed/n:.2f}s/row)")
    print(f"\n  Reason code breakdown:")
    for rc, cnt in out_df["reason_code"].value_counts().items():
        print(f"    {rc:<45s}: {cnt:4d} ({cnt/n*100:.1f}%)")
    print(f"\n  Priority level distribution (PASS only):")
    pass_df = out_df[out_df["quality_verdict"] == "PASS"]
    for lv, cnt in pass_df["priority_level"].value_counts().sort_index().items():
        print(f"    Level {lv}: {cnt:4d} ({cnt/n_pass*100:.1f}%)")
    print(f"\n  Damage type breakdown (PASS only):")
    for dt, cnt in pass_df["damage_type"].value_counts().items():
        print(f"    {dt:<20s}: {cnt:4d} ({cnt/n_pass*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(limit=args.limit)
