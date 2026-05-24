"""
swallow_guard_v063.py
──────────────────────────────────────────────────────────────
v0.6.3: 対策B「バッチ推論（batch_size > 1）」によるスループット改善

v0.6.2 との違い:
  v0.6.2: 1 行ずつ tokenize → generate → decode（シリアル）
  v0.6.3: N 行まとめて tokenize → batch generate → decode
          → GPU 並列化率向上 → スループット改善

バッチ推論の設計:
  Phase 1  全行に対してルールフィルタ (CPU, 高速)
  Phase 2  ルール通過行を BATCH_SIZE 単位でまとめて Judge LLM 推論
  Phase 3  Judge PASS 行を BATCH_SIZE 単位でまとめて Structurer LLM 推論

期待効果:
  batch_size=4 の場合、GPU utilization が上がり
  スループットが 2-4x 改善することを期待 (Windows/triton 制限内)

注意点:
  - padding=True 使用: 最長プロンプトに合わせてパディング
  - tokenizer.padding_side = "left" 推奨 (生成モデルの右端 attention)
  - VRAM: 16GB RTX4060Ti → batch_size=4 で ~12GB、batch_size=8 で ~15GB
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

# ── 定数 ──────────────────────────────────────────────────

BASE_MODEL_ID        = "tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1"
DEFAULT_ADAPTER_DIR  = "models/swallow8b_merged_n4000_r32_d05"
MAX_SEQ_LEN          = 2048
MAX_NEW_TOKENS       = 128
REPEAT_PENALTY       = 1.1
LOW_TOKEN_THRESHOLD  = 98
HIGH_TOKEN_THRESHOLD = 202
DEFAULT_BATCH_SIZE   = 4     # RTX4060Ti 16GB での推奨値

DAMAGE_KEYWORDS = [
    "ひび割れ", "クラック", "剥離", "剥落", "漏水", "さび", "錆",
    "腐食", "変形", "沈下", "損傷", "欠損", "鉄筋", "断面",
    "亀裂", "ひびわれ", "スパリング", "露出",
]
VAGUE_KEYWORDS = [
    "確認できない", "認識できない", "判断できない",
    "見えない", "不明", "わからない",
]

REPETITION_PATTERN = re.compile(r"(.{10,})\1{2,}")


# ── データクラス ───────────────────────────────────────────

class QualityVerdict(Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class GuardResult:
    verdict:        QualityVerdict
    reason_code:    str
    token_count:    int
    rule_triggered: bool
    llm_used:       bool
    elapsed_sec:    float

    damage_type:    Optional[str] = None
    severity_level: Optional[str] = None
    location:       Optional[str] = None
    risk_factor:    Optional[str] = None
    raw_llm_response: Optional[str] = None

    def is_pass(self) -> bool:
        return self.verdict == QualityVerdict.PASS


# ── ユーティリティ ─────────────────────────────────────────

def count_tokens_simple(text: str) -> int:
    """CJK文字 + ASCII ブロック数でトークン数近似"""
    cjk    = len(re.findall(r"[\u3000-\u9fff\uf900-\uffef]", text))
    ascii_ = len(text.encode("ascii", errors="ignore").split())
    return cjk + ascii_


def extract_assistant_text(prediction: str) -> str:
    m = re.search(r"ASSISTANT\s*:\s*(.+)", prediction, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else prediction.strip()


def detect_repetition(text: str) -> bool:
    if REPETITION_PATTERN.search(text):
        return True
    sentences = re.split(r"[。、\n]", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if len(sentences) > len(set(sentences)):
        return True
    return False


def has_damage_keywords(text: str) -> bool:
    return any(kw in text for kw in DAMAGE_KEYWORDS)


def has_vague_keywords(text: str) -> bool:
    return any(kw in text for kw in VAGUE_KEYWORDS)


# ── LLM プロンプト ─────────────────────────────────────────

JUDGE_SYSTEM = """\
あなたは、橋梁点検による１枚の損傷画像から生成された損傷テキスト記述を品質評価する専門家です。
評価対象は「１枚の画像に写った現時点の損傷状態の記述（one image, current time limited scope）」に限定されます。

【評価スコープ外】以下の内容が含まれていても、それ自体は不合格理由としません。
  - ２時点の比較・劣化進行の記述（「新たに発生した」「前回より進行」等）
  - 対策の方向性（「補修が必要」「予防保全対策」「必要に応じて」等）
  ただし、スコープ外の内容のみで損傷の現状認識が全く記述されていない場合は FAIL。

必ず以下の２行形式のみで回答してください。他の文章は不要です。
VERDICT: <PASS または FAIL>
REASON_CODE: <High Quality | No such file or directory | Not recognized from only image | Dirty or Noisy image | Short description>"""

JUDGE_USER_TEMPLATE = """\
【橋梁損傷説明テキスト】
{text}

評価基準:
  PASS : 損傷の種類・重症度・部位のいずれかが、画像認識の事実として具体的に記述されている
  FAIL : 損傷が認識できない、繰り返し・無意味な内容、または損傷の現状情報が全く含まれない

対策方向性（「補修が必要」等）や２時点比較（「新たに発生」等）は評価対象外とします。

VERDICT: と REASON_CODE: の２行形式のみで回答してください。"""

STRUCTURER_SYSTEM = """\
あなたは橋梁損傷報告書の構造化専門家です。
与えられた損傷説明テキストから以下のフィールドを抽出し、必ずJSON形式で回答してください。

抽出フィールド:
  damage_type    : 損傷種別 (rebar_exposure / crack / corrosion / spalling / section_loss / unknown)
  severity_level : 重症度   (high / medium / low)
  location       : 部位     (girder / deck / bearing / pier / railing / unknown)
  risk_factor    : リスク   (structural / durability / aesthetic / unknown)

回答形式 (JSONのみ、他の文章は不要):
{"damage_type": "...", "severity_level": "...", "location": "...", "risk_factor": "..."}"""

STRUCTURER_USER_TEMPLATE = """\
【損傷説明テキスト】
{text}

上記テキストから損傷情報をJSONで抽出してください。"""


# ── Llama-3 チャットフォーマット ──────────────────────────

def format_llama3_messages(system: str, user: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ── パース ─────────────────────────────────────────────────

def parse_verdict(text: str) -> tuple[QualityVerdict, str]:
    vm = re.search(r"VERDICT\s*:\s*(PASS|FAIL)", text, re.IGNORECASE)
    rm = re.search(r"REASON_CODE\s*:\s*(.{5,80})", text, re.IGNORECASE)
    verdict = QualityVerdict.PASS if (vm and vm.group(1).upper() == "PASS") else QualityVerdict.FAIL
    reason  = rm.group(1).strip() if rm else ("High Quality" if verdict == QualityVerdict.PASS else "Unspecified")
    return verdict, reason


def parse_structure(text: str) -> dict:
    defaults = {
        "damage_type":    "unknown",
        "severity_level": "medium",
        "location":       "unknown",
        "risk_factor":    "unknown",
    }
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if not m:
        return defaults
    try:
        data = json.loads(m.group(0))
        for k in defaults:
            if k not in data or data[k] not in _valid_values(k):
                data[k] = defaults[k]
        return data
    except json.JSONDecodeError:
        return defaults


def _valid_values(field: str) -> list[str]:
    mapping = {
        "damage_type":    ["rebar_exposure", "crack", "corrosion", "spalling", "section_loss", "unknown"],
        "severity_level": ["high", "medium", "low"],
        "location":       ["girder", "deck", "bearing", "pier", "railing", "unknown"],
        "risk_factor":    ["structural", "durability", "aesthetic", "unknown"],
    }
    return mapping.get(field, [])


# ── メインクラス ───────────────────────────────────────────

class SwallowQualityGuardV063:
    """
    v0.6.3: 対策B「バッチ推論」

    追加メソッド:
      _generate_batch(prompts)    → list[str]   複数プロンプトをバッチ処理
      evaluate_batch(predictions) → list[GuardResult]  バッチ評価

    バッチ処理の流れ:
      1. Phase 1: 全行のルールフィルタ (CPU, 高速, シリアル)
      2. Phase 2: ルール通過行をまとめて Judge LLM バッチ推論
      3. Phase 3: Judge PASS 行をまとめて Structurer LLM バッチ推論
    """

    def __init__(
        self,
        adapter_dir:    str  = DEFAULT_ADAPTER_DIR,
        low_threshold:  int  = LOW_TOKEN_THRESHOLD,
        high_threshold: int  = HIGH_TOKEN_THRESHOLD,
        use_llm:        bool = True,
        load_in_4bit:   bool = True,
        max_seq_length: int  = MAX_SEQ_LEN,
        batch_size:     int  = DEFAULT_BATCH_SIZE,
    ):
        self.adapter_dir    = Path(adapter_dir)
        self.low_threshold  = low_threshold
        self.high_threshold = high_threshold
        self.use_llm        = use_llm
        self.max_seq_length = max_seq_length
        self.batch_size     = batch_size

        self.model     = None
        self.tokenizer = None

        if use_llm:
            self._load_model(load_in_4bit)

    # ── モデルロード (v0.6.2 と同じ 2 段階方式) ────────────

    def _load_model(self, load_in_4bit: bool) -> None:
        """
        v0.6.2 と同じ 2 段階ロード方式を採用。
        v0.6.3 の新機能はロード後のバッチ推論 (_generate_batch) のみ。
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError("unsloth が未インストールです。")

        print("[V063Guard] Loading model via 2-step approach ...")
        print(f"  Base: {BASE_MODEL_ID}")
        print(f"  Adapter: {self.adapter_dir}")
        print(f"  4bit={load_in_4bit}, max_seq_length={self.max_seq_length}, "
              f"batch_size={self.batch_size}")

        t0 = time.time()

        # Step 1: ベースモデル (Unsloth QKV Triton パッチ適用)
        # local_files_only=True: transformers 4.57.x が additional_chat_templates を
        # HF Hub から取得しようとして 404 になるのを防ぐ (キャッシュ済みモデルを使用)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name       = BASE_MODEL_ID,
            max_seq_length   = self.max_seq_length,
            dtype            = None,
            load_in_4bit     = load_in_4bit,
            local_files_only = True,
        )
        print(f"  Step1 (base model): {time.time()-t0:.1f}s")

        # Step 2: LoRA アダプタ適用
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(self.adapter_dir))
        print(f"  Step2 (adapter): {time.time()-t0:.1f}s")

        # Step 3: 推論最適化 (LoRA マージ + for_inference)
        FastLanguageModel.for_inference(model)
        print(f"  Step3 (for_inference): {time.time()-t0:.1f}s")

        # バッチ推論のため left-padding を設定
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.model     = model
        self.tokenizer = tokenizer
        print(f"[V063Guard] Load complete: {time.time()-t0:.1f}s")

    def unload_model(self) -> None:
        import torch
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[V063Guard] Model unloaded.")

    # ── LLM 生成: 単一 ─────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        """単一プロンプトの生成（後方互換用）"""
        return self._generate_batch([prompt])[0]

    # ── LLM 生成: バッチ ───────────────────────────────────

    def _generate_batch(self, prompts: list[str]) -> list[str]:
        """
        複数プロンプトをまとめて GPU 推論する (v0.6.3 のコア機能)。

        padding_side="left" を使用することで、各シーケンスの右端 (生成開始位置)
        が揃い、生成品質が単一推論と同等になる。

        入力長が self.batch_size を超える場合はミニバッチに分割して処理する。
        """
        import torch
        if not prompts:
            return []

        all_responses: list[str] = []

        # ミニバッチに分割して処理（VRAM 超過を防ぐ）
        for batch_start in range(0, len(prompts), self.batch_size):
            mini_prompts = prompts[batch_start: batch_start + self.batch_size]

            inputs = self.tokenizer(
                mini_prompts,
                return_tensors = "pt",
                padding        = True,          # 短いプロンプトをパディング
                truncation     = True,
                max_length     = self.max_seq_length - MAX_NEW_TOKENS,
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens     = MAX_NEW_TOKENS,
                    do_sample          = False,
                    repetition_penalty = REPEAT_PENALTY,
                    eos_token_id       = self.tokenizer.eos_token_id,
                    pad_token_id       = self.tokenizer.pad_token_id,
                    use_cache          = True,
                )

            # 入力トークン長 (padding 後の共通長) を除いて生成部分を取り出す
            input_len = inputs["input_ids"].shape[1]
            for out in output_ids:
                gen_tokens = out[input_len:]
                text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                all_responses.append(text)

        return all_responses

    # ── Stage 0: 画像パス確認 ──────────────────────────────

    @staticmethod
    def _check_image_path(image_path: Optional[str]) -> Optional[GuardResult]:
        if not image_path:
            return None
        if not os.path.exists(image_path):
            return GuardResult(
                verdict=QualityVerdict.FAIL,
                reason_code="No such file or directory",
                token_count=0, rule_triggered=True, llm_used=False, elapsed_sec=0.0,
            )
        return None

    # ── Stage 1: Rule-based ────────────────────────────────

    def _stage1_rule_check(self, text: str, token_count: int) -> Optional[GuardResult]:
        if token_count < self.low_threshold:
            reason = ("Not recognized from only image"
                      if not has_damage_keywords(text) else "Short description")
            return GuardResult(
                verdict=QualityVerdict.FAIL, reason_code=reason,
                token_count=token_count, rule_triggered=True, llm_used=False, elapsed_sec=0.0,
            )
        if detect_repetition(text):
            return GuardResult(
                verdict=QualityVerdict.FAIL, reason_code="Dirty or Noisy image",
                token_count=token_count, rule_triggered=True, llm_used=False, elapsed_sec=0.0,
            )
        if not has_damage_keywords(text) and has_vague_keywords(text):
            return GuardResult(
                verdict=QualityVerdict.FAIL, reason_code="Not recognized from only image",
                token_count=token_count, rule_triggered=True, llm_used=False, elapsed_sec=0.0,
            )
        return None

    # ── バッチ評価 (v0.6.3 のメイン API) ─────────────────

    def evaluate_batch(
        self,
        predictions: list[str],
        image_paths: Optional[list[Optional[str]]] = None,
    ) -> list[GuardResult]:
        """
        複数の VLM 出力をまとめて評価し、GuardResult のリストを返す。

        処理フロー:
          Phase 1: 全行のルールフィルタ (CPU)
          Phase 2: ルール通過行を Judge LLM バッチ推論
          Phase 3: Judge PASS 行を Structurer LLM バッチ推論

        Args:
            predictions: VLM 出力テキストのリスト
            image_paths: 対応する画像パスのリスト (None 可)

        Returns:
            入力と同じ長さの GuardResult リスト
        """
        t0 = time.time()
        n = len(predictions)
        if image_paths is None:
            image_paths = [None] * n

        results: list[Optional[GuardResult]] = [None] * n

        # ── Phase 1: Rule filter (全行, CPU) ──────────────
        # pending_llm: (original_idx, text, token_count)
        pending_llm: list[tuple[int, str, int]] = []

        for i, (pred, img_path) in enumerate(zip(predictions, image_paths)):
            # Stage 0: 画像パス確認
            r = self._check_image_path(img_path)
            if r is not None:
                results[i] = r
                continue

            text = extract_assistant_text(pred)
            token_count = count_tokens_simple(text)

            # Stage 1: ルールフィルタ
            r = self._stage1_rule_check(text, token_count)
            if r is not None:
                results[i] = r
                continue

            # rule_only モード
            if not self.use_llm:
                results[i] = GuardResult(
                    verdict=QualityVerdict.PASS,
                    reason_code="High Quality (rule only)",
                    token_count=token_count,
                    rule_triggered=False,
                    llm_used=False,
                    elapsed_sec=0.0,
                )
                continue

            pending_llm.append((i, text, token_count))

        if not pending_llm:
            # 全行がルールで決定済み → elapsed を付けて返す
            elapsed = round(time.time() - t0, 3)
            for r in results:
                if r and r.elapsed_sec == 0.0:
                    r.elapsed_sec = elapsed
            return results

        # ── Phase 2: Batch Judge LLM ──────────────────────
        t_phase2 = time.time()
        judge_prompts = [
            format_llama3_messages(JUDGE_SYSTEM, JUDGE_USER_TEMPLATE.format(text=txt[:800]))
            for _, txt, _ in pending_llm
        ]
        judge_responses = self._generate_batch(judge_prompts)
        print(f"  [V063] Phase2 judge batch ({len(judge_prompts)} items): "
              f"{time.time()-t_phase2:.1f}s")

        # Judge 結果を振り分け
        # pending_struct: (original_idx, text, token_count, judge_reason)
        pending_struct: list[tuple[int, str, int, str]] = []

        for (i, txt, tc), judge_resp in zip(pending_llm, judge_responses):
            verdict, reason = parse_verdict(judge_resp)
            if verdict == QualityVerdict.FAIL:
                results[i] = GuardResult(
                    verdict=verdict, reason_code=reason,
                    token_count=tc, rule_triggered=False, llm_used=True,
                    elapsed_sec=0.0, raw_llm_response=judge_resp,
                )
            else:
                pending_struct.append((i, txt, tc, reason))

        # ── Phase 3: Batch Structurer LLM ─────────────────
        if pending_struct:
            t_phase3 = time.time()
            struct_prompts = [
                format_llama3_messages(STRUCTURER_SYSTEM, STRUCTURER_USER_TEMPLATE.format(text=txt[:600]))
                for _, txt, _, _ in pending_struct
            ]
            struct_responses = self._generate_batch(struct_prompts)
            print(f"  [V063] Phase3 structurer batch ({len(struct_prompts)} items): "
                  f"{time.time()-t_phase3:.1f}s")

            for (i, txt, tc, reason), struct_resp in zip(pending_struct, struct_responses):
                struct = parse_structure(struct_resp)
                results[i] = GuardResult(
                    verdict=QualityVerdict.PASS,
                    reason_code="High Quality",
                    token_count=tc,
                    rule_triggered=False,
                    llm_used=True,
                    elapsed_sec=0.0,
                    damage_type    = struct.get("damage_type"),
                    severity_level = struct.get("severity_level"),
                    location       = struct.get("location"),
                    risk_factor    = struct.get("risk_factor"),
                )

        # elapsed を全行に設定 (バッチ全体の経過時間 / n)
        elapsed_total = round(time.time() - t0, 3)
        elapsed_per_row = round(elapsed_total / n, 3)
        for r in results:
            if r is not None and r.elapsed_sec == 0.0:
                r.elapsed_sec = elapsed_per_row

        return results

    # ── 単一 evaluate (後方互換) ──────────────────────────

    def evaluate(self, prediction: str, image_path: Optional[str] = None) -> GuardResult:
        """単一評価。evaluate_batch の wrapper（後方互換用）"""
        return self.evaluate_batch([prediction], [image_path])[0]


# ── CLI テスト ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SwallowQualityGuardV063 バッチテスト")
    parser.add_argument("--rule-only",    action="store_true")
    parser.add_argument("--adapter-dir",  default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--batch-size",   type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    guard = SwallowQualityGuardV063(
        adapter_dir=args.adapter_dir,
        use_llm=not args.rule_only,
        batch_size=args.batch_size,
    )

    test_texts = [
        "ASSISTANT: 橋梁の主桁部分に幅0.3mmのひび割れが複数確認され、鉄筋の露出が見られる。",
        "ASSISTANT: 画像から損傷を確認できません。",
        "ASSISTANT: 点検結果に基づき補修が必要です。",
        "ASSISTANT: 支承部に著しい腐食と変形が確認される。緊急対応が必要な状態。",
    ]

    print(f"\n[Batch test] batch_size={args.batch_size}, n={len(test_texts)}")
    t0 = time.time()
    results = guard.evaluate_batch(test_texts)
    elapsed = time.time() - t0

    for txt, r in zip(test_texts, results):
        print(f"  [{r.verdict.value}] {r.reason_code} ({r.elapsed_sec:.2f}s)")
        if r.is_pass():
            print(f"    damage={r.damage_type}, severity={r.severity_level}, "
                  f"location={r.location}, risk={r.risk_factor}")

    print(f"\n  Total: {elapsed:.2f}s, Per-sample: {elapsed/len(test_texts):.2f}s")
