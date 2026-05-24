"""
swallow_guard_v062.py
──────────────────────────────────────────────────────────────
v0.6.2: 対策A「明示的 2 段階ロード」による QKV カーネル最適化

v0.6.1 との違い:
  v0.6.1: FastLanguageModel.from_pretrained(adapter_dir)
          → Unsloth がアダプタ検出→ベースモデル自動ロード
          → 結果: 0 QKV layers (Triton LoRA カーネル不活性)

  v0.6.2: Step1  FastLanguageModel.from_pretrained(BASE_MODEL_ID)
                 → Unsloth がベースモデルに QKV Triton カーネルを適用
          Step2  PeftModel.from_pretrained(model, adapter_dir)
                 → LoRA アダプタを重ねる (未マージ)
          Step3  FastLanguageModel.for_inference(model)
                 → LoRA マージ + 推論最適化

高速化の期待効果:
  ベースモデルロード時点で Unsloth Triton パッチが確実に適用される。
  アダプタパスを渡す v0.6.1 では unsloth_fixed=true フラグが
  二重パッチをスキップする可能性があった。
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# ── 定数 ──────────────────────────────────────────────────

BASE_MODEL_ID       = "tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1"
DEFAULT_ADAPTER_DIR = "models/swallow8b_merged_n4000_r32_d05"
MAX_SEQ_LEN         = 2048
MAX_NEW_TOKENS      = 128
REPEAT_PENALTY      = 1.1
LOW_TOKEN_THRESHOLD  = 98
HIGH_TOKEN_THRESHOLD = 202

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
    """VLM 出力から ASSISTANT: 以降のテキストを抽出"""
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
    """Swallow-8B-Instruct (Llama-3形式) のチャットテンプレート"""
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

class SwallowQualityGuardV062:
    """
    v0.6.2: 対策A「明示的 2 段階ロード」

    変更点 (_load_model のみ):
      v0.6.1: FastLanguageModel.from_pretrained(adapter_dir)
      v0.6.2: FastLanguageModel.from_pretrained(BASE_MODEL_ID)   # Step1
              PeftModel.from_pretrained(model, adapter_dir)       # Step2
              FastLanguageModel.for_inference(model)              # Step3

    期待効果:
      ベースモデルロード時に Unsloth QKV Triton パッチが確実に適用される。
      v0.6.1 で観測された「0 QKV layers」を解消する可能性がある。
    """

    def __init__(
        self,
        adapter_dir:    str  = DEFAULT_ADAPTER_DIR,
        low_threshold:  int  = LOW_TOKEN_THRESHOLD,
        high_threshold: int  = HIGH_TOKEN_THRESHOLD,
        use_llm:        bool = True,
        load_in_4bit:   bool = True,
        max_seq_length: int  = MAX_SEQ_LEN,
    ):
        self.adapter_dir    = Path(adapter_dir)
        self.low_threshold  = low_threshold
        self.high_threshold = high_threshold
        self.use_llm        = use_llm
        self.max_seq_length = max_seq_length

        self.model     = None
        self.tokenizer = None

        if use_llm:
            self._load_model(load_in_4bit)

    # ── モデルロード (v0.6.2: 2 段階) ──────────────────────

    def _load_model(self, load_in_4bit: bool) -> None:
        """
        v0.6.2 対策A: ベースモデルを先にロードし LoRA アダプタを後から適用。

        Step1: FastLanguageModel.from_pretrained(BASE_MODEL_ID)
               - Unsloth がベースモデルに QKV/MLP Triton パッチを適用
        Step2: PeftModel.from_pretrained(model, adapter_dir)
               - 未マージの LoRA アダプタを重ねる
        Step3: FastLanguageModel.for_inference(model)
               - LoRA マージ + 推論最適化 (KV cache 等)
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "unsloth が未インストールです。以下を実行してください:\n"
                '  pip install "unsloth[cu124-ampere]"'
            )

        print("[V062Guard] Loading model via 2-step approach ...")
        print(f"  Step1: Base model = {BASE_MODEL_ID}")
        print(f"  Step2: Adapter    = {self.adapter_dir}")
        print(f"  4bit={load_in_4bit}, max_seq_length={self.max_seq_length}")

        t0 = time.time()

        # Step 1: ベースモデルロード (Unsloth QKV パッチが確実に適用される)
        # local_files_only=True: transformers 4.57.x が additional_chat_templates を
        # HF Hub から取得しようとして 404 になるのを防ぐ (キャッシュ済みモデルを使用)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name      = BASE_MODEL_ID,
            max_seq_length  = self.max_seq_length,
            dtype           = None,
            load_in_4bit    = load_in_4bit,
            local_files_only = True,
        )
        t1 = time.time()
        print(f"  Step1 done: {t1-t0:.1f}s")

        # Step 2: 未マージ LoRA アダプタを適用
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(self.adapter_dir))
        t2 = time.time()
        print(f"  Step2 done: {t2-t1:.1f}s")

        # Step 3: 推論モード最適化 (LoRA マージ + Triton カーネル有効化)
        FastLanguageModel.for_inference(model)
        t3 = time.time()
        print(f"  Step3 done: {t3-t2:.1f}s")
        print(f"[V062Guard] Total load time: {t3-t0:.1f}s")

        self.model     = model
        self.tokenizer = tokenizer

    def unload_model(self) -> None:
        """VRAM 解放"""
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
        print("[V062Guard] Model unloaded.")

    # ── LLM 呼び出し ───────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        """単一プロンプトの生成（v0.6.1 と同一）"""
        import torch
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length - MAX_NEW_TOKENS,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens     = MAX_NEW_TOKENS,
                do_sample          = False,
                repetition_penalty = REPEAT_PENALTY,
                eos_token_id       = self.tokenizer.eos_token_id,
                pad_token_id       = self.tokenizer.eos_token_id,
                use_cache          = True,
            )

        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

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

    # ── Stage 2: LLM Judge ──────────────────────────────────

    def _stage2_llm_judge(self, text: str, token_count: int) -> GuardResult:
        t0 = time.time()

        judge_prompt = format_llama3_messages(
            JUDGE_SYSTEM,
            JUDGE_USER_TEMPLATE.format(text=text[:800]),
        )
        llm_response = self._generate(judge_prompt)
        verdict, reason_code = parse_verdict(llm_response)

        if verdict == QualityVerdict.FAIL:
            return GuardResult(
                verdict=verdict, reason_code=reason_code,
                token_count=token_count, rule_triggered=False, llm_used=True,
                elapsed_sec=round(time.time() - t0, 2),
                raw_llm_response=llm_response,
            )

        struct_prompt = format_llama3_messages(
            STRUCTURER_SYSTEM,
            STRUCTURER_USER_TEMPLATE.format(text=text[:600]),
        )
        struct_response = self._generate(struct_prompt)
        struct = parse_structure(struct_response)

        return GuardResult(
            verdict=QualityVerdict.PASS,
            reason_code="High Quality",
            token_count=token_count,
            rule_triggered=False,
            llm_used=True,
            elapsed_sec=round(time.time() - t0, 2),
            raw_llm_response=llm_response,
            damage_type    = struct.get("damage_type"),
            severity_level = struct.get("severity_level"),
            location       = struct.get("location"),
            risk_factor    = struct.get("risk_factor"),
        )

    # ── メイン evaluate ────────────────────────────────────

    def evaluate(self, prediction: str, image_path: Optional[str] = None) -> GuardResult:
        """
        VLM 出力の prediction 文字列を評価して GuardResult を返す。
        インターフェースは v0.6.1 SwallowQualityGuardUnsloth と同一。
        """
        t0 = time.time()

        if image_path:
            result = self._check_image_path(image_path)
            if result:
                result.elapsed_sec = round(time.time() - t0, 3)
                return result

        text = extract_assistant_text(prediction)
        token_count = count_tokens_simple(text)

        result = self._stage1_rule_check(text, token_count)
        if result is not None:
            result.elapsed_sec = round(time.time() - t0, 3)
            return result

        if not self.use_llm:
            return GuardResult(
                verdict=QualityVerdict.PASS,
                reason_code="High Quality (rule only)",
                token_count=token_count,
                rule_triggered=False,
                llm_used=False,
                elapsed_sec=round(time.time() - t0, 3),
            )

        result = self._stage2_llm_judge(text, token_count)
        result.elapsed_sec = round(time.time() - t0, 3)
        return result


# ── CLI テスト ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SwallowQualityGuardV062 テスト")
    parser.add_argument("--rule-only", action="store_true")
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    args = parser.parse_args()

    guard = SwallowQualityGuardV062(
        adapter_dir=args.adapter_dir,
        use_llm=not args.rule_only,
    )

    test_texts = [
        "ASSISTANT: 橋梁の主桁部分に幅0.3mmのひび割れが複数確認され、鉄筋の露出が見られる。",
        "ASSISTANT: 画像から損傷を確認できません。",
        "ASSISTANT: 点検結果に基づき補修が必要です。",
    ]

    for txt in test_texts:
        r = guard.evaluate(txt)
        print(f"  [{r.verdict.value}] {r.reason_code} ({r.elapsed_sec:.2f}s)")
        if r.is_pass():
            print(f"    damage={r.damage_type}, severity={r.severity_level}, "
                  f"location={r.location}, risk={r.risk_factor}")
