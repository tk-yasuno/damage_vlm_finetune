"""
pipeline_v06.py
──────────────────────────────────────────────────────────────
v0.6: Priority Scoring with Quality Guard Using Swallow 8B

パイプライン:
  [入力] inference_results_3k.csv (LLaVA-3k 推論済み, n=800)
      ↓
  Quality Guard (Stage1: Rule + Stage2: Swallow 8B)
      ├── FAIL → "No Score due to Low Quality Image" (priority_level=0)
      └── PASS → JSON 構造化済み → PriorityScorer → S∈{1,2,3,4,5}
      ↓
  [出力] data/v03_fine_tuning/evaluations/v06_scoring_results.csv

Usage:
    python pipeline_v06.py
    python pipeline_v06.py --input-csv data/v03_fine_tuning/evaluations/inference_results_3k.csv
    python pipeline_v06.py --rule-only               # LLM なし (ルールベースのみ, 高速)
    python pipeline_v06.py --limit 20                # テスト用: 先頭20件のみ
    python pipeline_v06.py --low-threshold 25 --high-threshold 380  # 閾値カスタマイズ

出力 CSV 列:
    [v0.5.1 引き継ぎ]
    image_id, image_path, prediction, ground_truth, cosine_similarity

    [v0.6 追加]
    assistant_text     : VLM の ASSISTANT 部分のみ
    token_count        : トークン数
    quality_verdict    : PASS / FAIL
    reason_code        : "High Quality" | "Not recognized from only image" | etc.
    rule_triggered     : Stage1 ルールで判定されたか (True/False)
    llm_used           : Swallow 8B が使われたか (True/False)
    guard_elapsed_sec  : Quality Guard 処理時間
    damage_type        : rebar_exposure / crack / corrosion / spalling / section_loss / unknown
    severity_level     : high / medium / low
    location           : girder / deck / bearing / pier / railing / unknown
    risk_factor        : structural / durability / aesthetic / unknown
    priority_score     : 0.0-1.0 の生スコア (FAIL 時は NaN)
    priority_level     : 1-5 (FAIL 時は 0)
    priority_description: 説明テキスト (FAIL 時は "No Score due to Low Quality Image")
    raw_damage_score   : 損傷種別スコア
    raw_severity_score : 重症度スコア
    raw_location_score : 位置スコア
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルートを sys.path に追加
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.quality_guard.swallow_guard import SwallowQualityGuard, QualityVerdict
from src.scoring.priority_scorer import PriorityScorer, ScoringConfig

# ── 定数 ──────────────────────────────────────────────────

DEFAULT_INPUT_CSV  = "data/v03_fine_tuning/evaluations/inference_results_3k.csv"
DEFAULT_OUTPUT_CSV = "data/v03_fine_tuning/evaluations/v06_scoring_results.csv"
NO_SCORE_MSG       = "No Score due to Low Quality Image"
CHECKPOINT_INTERVAL = 100  # N件ごとに中間CSVを保存

# ── PriorityScorer ラッパー ──────────────────────────────

def score_with_structure(
    scorer:        PriorityScorer,
    damage_type:   str,
    severity_level: str,
    location:      str,
    risk_factor:   str,
) -> dict:
    """GuardResult の構造情報から PriorityScore を計算して辞書で返す"""
    try:
        # PriorityScorer.calculate_score() の引数: damage_type, severity, location, risk
        ps = scorer.calculate_score(
            damage_type=damage_type   or "unknown",
            severity=severity_level   or "medium",
            location=location         or "unknown",
            risk=risk_factor          or "unknown",
        )
        return {
            "priority_score":       round(ps.raw_score, 4),
            "priority_level":       ps.priority_level,
            "priority_description": ps.priority_description,
            "raw_damage_score":     round(ps.damage_type_score, 4),
            "raw_severity_score":   round(ps.severity_score, 4),
            "raw_location_score":   round(ps.location_score, 4),
        }
    except Exception as e:
        print(f"  [WARN] PriorityScorer error: {e}")
        return {
            "priority_score":       float("nan"),
            "priority_level":       1,
            "priority_description": "Scoring error",
            "raw_damage_score":     float("nan"),
            "raw_severity_score":   float("nan"),
            "raw_location_score":   float("nan"),
        }


def make_no_score_row() -> dict:
    """FAIL 時の優先度列を返す"""
    return {
        "priority_score":       float("nan"),
        "priority_level":       0,
        "priority_description": NO_SCORE_MSG,
        "raw_damage_score":     float("nan"),
        "raw_severity_score":   float("nan"),
        "raw_location_score":   float("nan"),
    }


# ── メイン処理 ───────────────────────────────────────────

def run_pipeline(
    input_csv:      str,
    output_csv:     str,
    rule_only:      bool,
    limit:          Optional[int],
    low_threshold:  int,
    high_threshold: int,
    adapter_dir:    str,
) -> None:
    t_start = time.time()

    # ── 1. 入力 CSV 読み込み
    print(f"[Load] {input_csv}")
    df = pd.read_csv(input_csv, encoding="utf-8")
    if limit:
        df = df.head(limit)
        print(f"  [DEBUG] Limited to first {limit} rows")
    n_total = len(df)
    print(f"  Total rows: {n_total}")

    # ── 2. Quality Guard 初期化
    print(f"\n[Guard] Initializing Quality Guard (rule_only={rule_only}) ...")
    guard = SwallowQualityGuard(
        adapter_dir=adapter_dir,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        use_llm=not rule_only,
    )

    # ── 3. Priority Scorer 初期化
    print("[Scorer] Initializing PriorityScorer ...")
    scorer = PriorityScorer(ScoringConfig(rules_file="models/scoring_rules.yaml"))

    # ── 4. 行ごとに処理
    records = []
    n_pass = 0
    n_fail = 0

    print(f"\n[Pipeline] Processing {n_total} rows ...")
    for i, row in enumerate(df.itertuples(index=False)):
        prediction  = str(row.prediction)
        image_path  = str(row.image_path) if hasattr(row, "image_path") else None
        guard_result = guard.evaluate(prediction, image_path=image_path)

        # 優先度スコア計算
        if guard_result.is_pass():
            n_pass += 1
            if guard_result.damage_type:
                # LLM 構造化済み → スコア計算
                score_row = score_with_structure(
                    scorer,
                    guard_result.damage_type,
                    guard_result.severity_level or "medium",
                    guard_result.location      or "unknown",
                    guard_result.risk_factor   or "unknown",
                )
            else:
                # rule-only モード: PASS だが構造化なし → no score
                score_row = make_no_score_row()
                score_row["priority_description"] = "PASS (rule-only, no LLM structuring)"
        else:
            score_row = make_no_score_row()
            n_fail += 1

        # レコード組み立て
        rec = {
            # v0.5.1 列
            "image_id":           row.image_id,
            "image_path":         row.image_path,
            "prediction":         prediction,
            "ground_truth":       row.ground_truth,
            "cosine_similarity":  row.cosine_similarity,
            # v0.6 Guard 列
            "assistant_text":     guard_result.__dict__.get("_text", ""),  # 後で補完
            "token_count":        guard_result.token_count,
            "quality_verdict":    guard_result.verdict.value,
            "reason_code":        guard_result.reason_code,
            "rule_triggered":     guard_result.rule_triggered,
            "llm_used":           guard_result.llm_used,
            "guard_elapsed_sec":  guard_result.elapsed_sec,
            # 構造化情報
            "damage_type":        guard_result.damage_type,
            "severity_level":     guard_result.severity_level,
            "location":           guard_result.location,
            "risk_factor":        guard_result.risk_factor,
        }
        rec.update(score_row)
        records.append(rec)

        # 進捗ログ
        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t_start
            print(f"  [{i+1:4d}/{n_total}] PASS={n_pass} FAIL={n_fail} "
                  f"elapsed={elapsed:.0f}s")

        # チェックポイント保存
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            ckpt_path = output_csv.replace(".csv", f"_ckpt{i+1}.csv")
            pd.DataFrame(records).to_csv(ckpt_path, index=False, encoding="utf-8")
            print(f"  [Checkpoint] Saved: {ckpt_path}")

    # ── 5. 後処理: assistant_text を補完 (guard 内部処理済み値を利用)
    from src.quality_guard.swallow_guard import extract_assistant_text as _extract
    out_df = pd.DataFrame(records)
    out_df["assistant_text"] = out_df["prediction"].apply(_extract)

    # ── 6. 結果 CSV 保存
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")

    # ── 7. サマリー表示
    elapsed_total = time.time() - t_start
    pass_rate = n_pass / n_total * 100
    fail_rate = n_fail / n_total * 100

    print(f"\n{'='*60}")
    print(f"  v0.6 Pipeline Complete")
    print(f"{'='*60}")
    print(f"  Total rows   : {n_total}")
    print(f"  PASS (scored): {n_pass:4d} ({pass_rate:.1f}%)")
    print(f"  FAIL (no score): {n_fail:4d} ({fail_rate:.1f}%)")
    print(f"  Total elapsed: {elapsed_total:.1f}s  "
          f"({elapsed_total/n_total:.2f}s/row)")
    print(f"\n  reason_code breakdown:")
    for rc, cnt in out_df["reason_code"].value_counts().items():
        print(f"    {rc:<45s}: {cnt:4d} ({cnt/n_total*100:.1f}%)")
    print(f"\n  priority_level distribution (PASS only):")
    pass_df = out_df[out_df["quality_verdict"] == "PASS"]
    if len(pass_df) > 0:
        for lv in sorted(pass_df["priority_level"].dropna().unique()):
            cnt = (pass_df["priority_level"] == lv).sum()
            print(f"    Level {int(lv)}: {cnt:4d} ({cnt/len(pass_df)*100:.1f}%)")
        print(f"    Mean priority score: {pass_df['priority_score'].mean():.4f}")

    print(f"\n  Output CSV: {output_csv}")
    print(f"{'='*60}")

    # ── 8. VRAM 解放
    guard.unload_model()


# ── 型ヒント修正のため Optional インポート ─────────────────

from typing import Optional


# ── エントリポイント ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="v0.6 Pipeline: Quality Guard + Priority Scoring"
    )
    parser.add_argument("--input-csv",     default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv",    default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--adapter-dir",   default="models/swallow8b_merged_n4000_r32_d05")
    parser.add_argument("--rule-only",     action="store_true",
                        help="Use rule-based guard only (no Swallow 8B, fast debug)")
    parser.add_argument("--limit",         type=int, default=None,
                        help="Process only first N rows (for debugging)")
    parser.add_argument("--low-threshold", type=int, default=98,
                        help="Token count below this → FAIL (Short) [実測5th pct=98]")
    parser.add_argument("--high-threshold",type=int, default=202,
                        help="Token count above this → potential FAIL (Noisy) [実測95th pct=202]")
    args = parser.parse_args()

    run_pipeline(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        rule_only=args.rule_only,
        limit=args.limit,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        adapter_dir=args.adapter_dir,
    )


if __name__ == "__main__":
    main()
