"""
src/quality_guard/swallow_guard_unsloth.py
──────────────────────────────────────────────────────────────
v0.6.1 Quality Guard: Unsloth FastLanguageModel で高速化した Swallow-8B 推論

参考実装: I:/ACT2025.5.26-2030/MVP/kasensabo_jp_qlora/scripts/05_train_lora_unsloth.py
         I:/ACT2025.5.26-2030/MVP/kasensabo_jp_qlora/scripts/12_eval_judge.py

HF方式 (v0.6) との比較:
  HF:      transformers + PEFT + BitsAndBytes 4-bit → 30.44秒/行
  Unsloth: FastLanguageModel + Flash Attention 2 + Triton → 目標 6-10秒/行

Unsloth 高速化の仕組み:
  1. Flash Attention 2: Attention 計算を O(n²) → O(n) へ圧縮
  2. Triton カーネル: LoRA 演算を GPU カーネルレベルで最適化
  3. FastLanguageModel.for_inference(): 推論モード専用コンパイル
  4. PEFT アダプタ: from_pretrained() が adapter_config.json を自動認識

モデル:
  Base   : tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1
  Adapter: models/swallow8b_merged_n4000_r32_d05/ (PEFT/LoRA r=32)
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

# Windows cp932 環境での Unicode エラー回避
os.environ.setdefault("PYTHONUTF8", "1")
# Triton JIT ハング対策: torch.compile 無効化 (参照: kasensabo_jp_qlora)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# ── 定数 ────────────────────────────────────────────────────

BASE_MODEL_ID      = "tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1"
DEFAULT_ADAPTER_DIR = "models/swallow8b_merged_n4000_r32_d05"

MAX_SEQ_LEN      = 2048
MAX_NEW_TOKENS   = 128
REPEAT_PENALTY   = 1.1

# Rule-based thresholds (analyze_low_quality_text.py 実測値: n=800, 3k model)
LOW_TOKEN_THRESHOLD  = 98   # 5th percentile
HIGH_TOKEN_THRESHOLD = 202  # 95th percentile
REPETITION_MIN_LEN   = 10
REPETITION_MIN_COUNT = 2

# 損傷関連キーワード
DAMAGE_KEYWORDS = [
    "ひびわれ", "クラック", "鉄筋", "腐食", "剥離", "劣化",
    "変形", "欠損", "漏水", "遊離石灰", "うき", "損傷",
    "破損", "さび", "サビ", "亀裂", "穴", "断面", "貫通",
]

VAGUE_KEYWORDS = [
    "説明できません", "確認できません", "判断できません",
    "わかりません", "不明です", "確認不可",
    "画像から判断", "画像のみでは",
]

REPETITION_PATTERN = re.compile(r"(.{10,}?)\1{2,}", re.DOTALL)


# ── データクラス ──────────────────────────────────────────

class QualityVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class GuardResult:
    """Quality Guard の判定結果 (swallow_guard.py と互換)"""
    verdict: QualityVerdict
    reason_code: str
    token_count: int
    rule_triggered: bool
    llm_used: bool
    elapsed_sec: float
    raw_llm_response: Optional[str] = None
    damage_type:     Optional[str] = None
    severity_level:  Optional[str] = None
    location:        Optional[str] = None
    risk_factor:     Optional[str] = None

    def is_pass(self) -> bool:
        return self.verdict == QualityVerdict.PASS

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


# ── ユーティリティ ──────────────────────────────────────────

def count_tokens_simple(text: str) -> int:
    """CJK文字 + ASCII ブロック数でトークン数近似"""
    cjk    = len(re.findall(r"[\u3000-\u9fff\uf900-\uffef]", text))
    ascii_ = len(text.encode("ascii", errors="ignore").split())
    return cjk + ascii_


def extract_assistant_text(prediction: str) -> str:
    """USER: ... ASSISTANT: ... 形式から ASSISTANT 部分を抽出"""
    if "ASSISTANT:" in prediction:
        return prediction.split("ASSISTANT:", 1)[1].strip()
    return prediction.strip()


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

class SwallowQualityGuardUnsloth:
    """
    Unsloth FastLanguageModel による高速 Quality Guard (v0.6.1)

    参考:
      kasensabo_jp_qlora/scripts/05_train_lora_unsloth.py  - FastLanguageModel.from_pretrained()
      kasensabo_jp_qlora/scripts/12_eval_judge.py          - FastLanguageModel.for_inference()
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

    # ── モデルロード (Unsloth) ──────────────────────────────

    def _load_model(self, load_in_4bit: bool) -> None:
        """
        FastLanguageModel.from_pretrained() で PEFT アダプタを自動認識してロード。
        adapter_config.json が存在するディレクトリを model_name に渡すと
        Unsloth がベースモデルを自動解決して LoRA を適用する。
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "unsloth が未インストールです。以下を実行してください:\n"
                '  pip install "unsloth[cu124-ampere]"'
            )

        print(f"[UnslothGuard] Loading model via FastLanguageModel ...")
        print(f"  Adapter: {self.adapter_dir}")
        print(f"  4bit: {load_in_4bit}, max_seq_length: {self.max_seq_length}")

        t0 = time.time()
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name     = str(self.adapter_dir),
            max_seq_length = self.max_seq_length,
            dtype          = None,          # 自動検出 (bfloat16 or float16)
            load_in_4bit   = load_in_4bit,
        )

        # 推論モード最適化: Flash Attention 2 + Triton カーネル有効化
        FastLanguageModel.for_inference(self.model)
        print(f"[UnslothGuard] Model loaded in {time.time()-t0:.1f}s "
              f"(for_inference enabled)")

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
        print("[UnslothGuard] Model unloaded.")

    # ── LLM 呼び出し (Unsloth 最適化済み) ─────────────────

    def _generate(self, prompt: str) -> str:
        """
        Unsloth for_inference 済みモデルで生成。
        do_sample=False 時は temperature を渡さない (transformers 警告回避)。
        """
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
                max_new_tokens      = MAX_NEW_TOKENS,
                do_sample           = False,          # 貪欲デコード
                repetition_penalty  = REPEAT_PENALTY,
                eos_token_id        = self.tokenizer.eos_token_id,
                pad_token_id        = self.tokenizer.eos_token_id,
                use_cache           = True,           # Unsloth KV cache 最適化
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
        return None  # Stage 2 へ

    # ── Stage 2: Unsloth LLM Judge ──────────────────────────

    def _stage2_llm_judge(self, text: str, token_count: int) -> GuardResult:
        t0 = time.time()

        # 品質判定
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

        # PASS → JSON 構造化
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
        image_path が指定された場合は Stage 0 (ファイル存在確認) を実行。
        """
        t0 = time.time()

        # Stage 0: ファイル存在確認
        if image_path:
            result = self._check_image_path(image_path)
            if result:
                result.elapsed_sec = round(time.time() - t0, 3)
                return result

        # ASSISTANT 部分のみ抽出
        text = extract_assistant_text(prediction)
        token_count = count_tokens_simple(text)

        # Stage 1: Rule-based
        result = self._stage1_rule_check(text, token_count)
        if result is not None:
            result.elapsed_sec = round(time.time() - t0, 3)
            return result

        # rule_only モード: PASS として返す (LLM なし)
        if not self.use_llm:
            return GuardResult(
                verdict=QualityVerdict.PASS,
                reason_code="High Quality (rule only)",
                token_count=token_count,
                rule_triggered=False,
                llm_used=False,
                elapsed_sec=round(time.time() - t0, 3),
            )

        # Stage 2: Unsloth LLM Judge
        result = self._stage2_llm_judge(text, token_count)
        result.elapsed_sec = round(time.time() - t0, 3)
        return result


# ── CLI テスト ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SwallowQualityGuardUnsloth テスト")
    parser.add_argument("--rule-only", action="store_true")
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    args = parser.parse_args()

    TEST_CASES = [
        {
            "input": "USER: この橋梁の損傷状態を詳しく説明してください。 ASSISTANT: 確認できません。損傷の詳細は不明です。",
            "expected": "FAIL / Not recognized",
        },
        {
            "input": "USER: この橋梁の損傷状態を詳しく説明してください。 ASSISTANT: ひびわれ。",
            "expected": "FAIL / Short description",
        },
        {
            "input": (
                "USER: この橋梁の損傷状態を詳しく説明してください。 "
                "ASSISTANT: 主桁下面に鉄筋露出（最大幅2cm）が確認された。腐食が著しく進行しており、構造安全性への影響が懸念される。"
                "コンクリート剥離も見られ、補修が必要な状態である。"
            ),
            "expected": "PASS / High Quality",
        },
    ]

    guard = SwallowQualityGuardUnsloth(
        adapter_dir=args.adapter_dir,
        use_llm=not args.rule_only,
    )

    for tc in TEST_CASES:
        print("\n" + "─" * 60)
        print(f"  Input : {tc['input'][:80]}...")
        print(f"  Expected: {tc['expected']}")
        result = guard.evaluate(tc["input"])
        verdict_str = f"{result.verdict.value} / {result.reason_code}"
        print(f"  Result: {verdict_str}")
        if result.damage_type:
            print(f"          damage_type={result.damage_type}, "
                  f"severity={result.severity_level}, location={result.location}")
        print(f"  Time  : {result.elapsed_sec}s | rule={result.rule_triggered} | llm={result.llm_used}")

    if not args.rule_only:
        guard.unload_model()
