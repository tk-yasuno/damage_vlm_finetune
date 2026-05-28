"""
pipeline_v061.py
──────────────────────────────────────────────────────────────
v0.6.1: Priority Scoring with Quality Guard Using Unsloth Swallow 8B

v0.6 (HF方式) との違い:
  v0.6: transformers + PEFT + BitsAndBytes → 30.44秒/行, 800件≈6.8時間
  v0.6.1: Unsloth FastLanguageModel → 目標 6-10秒/行, 800件≈1-2時間

高速化の仕組み (参照: kasensabo_jp_qlora/scripts/12_eval_judge.py):
  FastLanguageModel.from_pretrained() + FastLanguageModel.for_inference()
  → Flash Attention 2 + Triton カーネル最適化

参考実装:
  I:\\ACT2025.5.26-2030\\MVP\\kasensabo_jp_qlora\\scripts\\05_train_lora_unsloth.py
  I:\\ACT2025.5.26-2030\\MVP\\kasensabo_jp_qlora\\scripts\\12_eval_judge.py

参考論文:
  Takato Yasuno, Adapting Methods for Domain-Specific Japanese Small LMs,
  arXiv_qlora_scale_quant_2026.pdf

Usage:
    python pipeline_v061.py
    python pipeline_v061.py --rule-only               # LLM なし (ルールのみ, 高速)
    python pipeline_v061.py --limit 5                 # テスト: 先頭5件のみ
    python pipeline_v061.py --limit 20                # テスト: 先頭20件のみ
    python pipeline_v061.py --input-csv data/v03_fine_tuning/evaluations/inference_results_3k.csv

出力 CSV 列 (v0.6 と同一形式):
    image_id, image_path, prediction, ground_truth, cosine_similarity
    assistant_text, token_count, quality_verdict, reason_code,
    rule_triggered, llm_used, guard_elapsed_sec,
    damage_type, severity_level, location, risk_factor,
    priority_score, priority_level, priority_description,
    raw_damage_score, raw_severity_score, raw_location_score
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Windows cp932 環境での Unicode エラー回避
os.environ.setdefault("PYTHONUTF8", "1")
# Triton JIT ハング対策: torch.compile 無効化 (参照: kasensabo_jp_qlora)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import pandas as pd

# プロジェクトルートを sys.path に追加
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.quality_guard.swallow_guard_unsloth import (
    SwallowQualityGuardUnsloth,
    QualityVerdict,
    extract_assistant_text,
)
from src.scoring.priority_scorer import PriorityScorer, ScoringConfig

# ── 定数 ──────────────────────────────────────────────────

DEFAULT_INPUT_CSV   = "data/v03_fine_tuning/evaluations/inference_results_3k.csv"
DEFAULT_OUTPUT_CSV  = "data/v03_fine_tuning/evaluations/v061_scoring_results.csv"
DEFAULT_ADAPTER_DIR = "models/swallow8b_merged_n4000_r32_d05"
NO_SCORE_MSG        = "No Score due to Low Quality Image"
CHECKPOINT_INTERVAL = 100  # N件ごとに中間CSV保存

# ── PriorityScorer ラッパー ──────────────────────────────

def score_with_structure(
    scorer:         PriorityScorer,
    damage_type:    str,
    severity_level: str,
    location:       str,
    risk_factor:    str,
) -> dict:
    """GuardResult の構造情報から PriorityScore を計算して辞書で返す"""
    try:
        ps = scorer.calculate_score(
            damage_type=damage_type    or "unknown",
            severity=severity_level    or "medium",
            location=location          or "unknown",
            risk=risk_factor           or "unknown",
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
    input_csv:     str,
    output_csv:    str,
    rule_only:     bool,
    limit:         Optional[int],
    low_threshold: int,
    high_threshold: int,
    adapter_dir:   str,
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

    # ── 2. Unsloth Quality Guard 初期化
    print(f"\n[Guard] Initializing Unsloth Quality Guard (rule_only={rule_only}) ...")
    guard = SwallowQualityGuardUnsloth(
        adapter_dir    = adapter_dir,
        low_threshold  = low_threshold,
        high_threshold = high_threshold,
        use_llm        = not rule_only,
    )

    # ── 3. Priority Scorer 初期化
    print("[Scorer] Initializing PriorityScorer ...")
    scorer = PriorityScorer(ScoringConfig(rules_file="models/scoring_rules.yaml"))

    # ── 4. 行ごとに処理
    records = []
    n_pass  = 0
    n_fail  = 0

    print(f"\n[Pipeline v0.6.1 Unsloth] Processing {n_total} rows ...")
    for i, row in enumerate(df.itertuples(index=False)):
        prediction = str(row.prediction)
        image_path = str(row.image_path) if hasattr(row, "image_path") else None
        guard_result = guard.evaluate(prediction, image_path=image_path)

        # 優先度スコア計算
        if guard_result.is_pass():
            n_pass += 1
            if guard_result.damage_type:
                score_row = score_with_structure(
                    scorer,
                    guard_result.damage_type,
                    guard_result.severity_level or "medium",
                    guard_result.location       or "unknown",
                    guard_result.risk_factor    or "unknown",
                )
            else:
                # rule-only: PASS だが構造化なし
                score_row = make_no_score_row()
                score_row["priority_description"] = "PASS (rule-only, no LLM structuring)"
        else:
            score_row = make_no_score_row()
            n_fail += 1

        rec = {
            "image_id":          row.image_id,
            "image_path":        row.image_path,
            "prediction":        prediction,
            "ground_truth":      row.ground_truth,
            "cosine_similarity": row.cosine_similarity,
            "assistant_text":    "",   # 後で補完
            "token_count":       guard_result.token_count,
            "quality_verdict":   guard_result.verdict.value,
            "reason_code":       guard_result.reason_code,
            "rule_triggered":    guard_result.rule_triggered,
            "llm_used":          guard_result.llm_used,
            "guard_elapsed_sec": guard_result.elapsed_sec,
            "damage_type":       guard_result.damage_type,
            "severity_level":    guard_result.severity_level,
            "location":          guard_result.location,
            "risk_factor":       guard_result.risk_factor,
        }
        rec.update(score_row)
        records.append(rec)

        # 進捗ログ (50件ごと + 最終行)
        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t_start
            sec_per_row = elapsed / (i + 1)
            remain_rows = n_total - (i + 1)
            eta_s = sec_per_row * remain_rows
            print(f"  [{i+1:4d}/{n_total}] PASS={n_pass} FAIL={n_fail} "
                  f"elapsed={elapsed:.0f}s ({sec_per_row:.1f}s/row) "
                  f"ETA={eta_s/60:.1f}min")

        # チェックポイント保存
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            ckpt_path = output_csv.replace(".csv", f"_ckpt{i+1}.csv")
            pd.DataFrame(records).to_csv(ckpt_path, index=False, encoding="utf-8")
            print(f"  [Checkpoint] Saved: {ckpt_path}")

    # ── 5. assistant_text を補完
    out_df = pd.DataFrame(records)
    out_df["assistant_text"] = out_df["prediction"].apply(extract_assistant_text)

    # ── 6. 結果 CSV 保存
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")

    # ── 7. サマリー表示
    elapsed_total = time.time() - t_start
    pass_rate = n_pass / n_total * 100
    fail_rate = n_fail / n_total * 100

    print(f"\n{'='*60}")
    print(f"  v0.6.1 Pipeline Complete (Unsloth)")
    print(f"{'='*60}")
    print(f"  Total rows     : {n_total}")
    print(f"  PASS (scored)  : {n_pass:4d} ({pass_rate:.1f}%)")
    print(f"  FAIL (no score): {n_fail:4d} ({fail_rate:.1f}%)")
    print(f"  Total elapsed  : {elapsed_total:.1f}s  "
          f"({elapsed_total/n_total:.2f}s/row)")
    print(f"\n  reason_code breakdown:")
    for rc, cnt in out_df["reason_code"].value_counts().items():
        print(f"    {rc:<45s}: {cnt:4d} ({cnt/n_total*100:.1f}%)")

    print(f"\n  priority_level distribution (PASS only):")
    pass_df = out_df[out_df["quality_verdict"] == "PASS"]
    if len(pass_df) > 0:
        for lvl, cnt in pass_df["priority_level"].value_counts().sort_index().items():
            print(f"    Level {lvl}: {cnt:4d} ({cnt/len(pass_df)*100:.1f}%)")
        scored = pass_df["priority_score"].dropna()
        if len(scored) > 0:
            print(f"    Mean priority score: {scored.mean():.4f}")

    print(f"\n  Output CSV: {output_csv}")
    print(f"{'='*60}")

    if not rule_only:
        guard.unload_model()


# ── CLI ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="v0.6.1: Priority Scoring with Unsloth Swallow 8B Quality Guard"
    )
    parser.add_argument("--input-csv",     default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv",    default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--adapter-dir",   default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--rule-only",     action="store_true",
                        help="Swallow 8B をスキップ (ルールベースのみ)")
    parser.add_argument("--limit",         type=int, default=None,
                        help="先頭 N 件のみ処理 (テスト用)")
    parser.add_argument("--low-threshold", type=int, default=98,
                        help="Low quality token 閾値 (default: 98)")
    parser.add_argument("--high-threshold", type=int, default=202,
                        help="High quality token 閾値 (default: 202)")
    args = parser.parse_args()

    run_pipeline(
        input_csv      = args.input_csv,
        output_csv     = args.output_csv,
        rule_only      = args.rule_only,
        limit          = args.limit,
        low_threshold  = args.low_threshold,
        high_threshold = args.high_threshold,
        adapter_dir    = args.adapter_dir,
    )


if __name__ == "__main__":
    main()
