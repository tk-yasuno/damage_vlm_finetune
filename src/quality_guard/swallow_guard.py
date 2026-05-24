"""
src/quality_guard/swallow_guard.py
──────────────────────────────────────────────────────────────
v0.6 Quality Guard: Swallow-8B QLoRA を使ったVLM出力品質判定モジュール

設計コンセプト（Concept_LLM as Guard 20260524.jpg）:
  Stage 1 (Rule-based, fast):
    - token_count < LOW_THRESHOLD  → FAIL (Short description)
    - 繰り返しパターン検出         → FAIL (Dirty or Noisy image)
    - 損傷キーワード皆無            → FAIL (Not recognized from only image)
  Stage 2 (Swallow 8B LLM Judge, borderline cases):
    - Stage1 を通過した場合に Swallow 8B でゼロショット品質評価
    - PASS → JSON 構造化 → PriorityScorer へ
    - FAIL → "No Score due to Low Quality Image"

使用モデル:
  Base   : tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1
  Adapter: models/swallow8b_merged_n4000_r32_d05/ (PEFT/LoRA)
  Loading: transformers + PEFT (4-bit NF4 quantization)
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

# ── 定数 ────────────────────────────────────────────────────

BASE_MODEL_ID = "tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1"
DEFAULT_ADAPTER_DIR = "models/swallow8b_merged_n4000_r32_d05"

# Rule-based thresholds (analyze_low_quality_text.py 実行結果より: n=800, 3k model)
# 5th percentile=98, 95th percentile=202, median=120
LOW_TOKEN_THRESHOLD  = 98   # これ未満: 短すぎる (5th percentile: 実測値)
HIGH_TOKEN_THRESHOLD = 202  # これ超過: 長すぎる or 繰り返し疑い (95th percentile: 実測値)
REPETITION_MIN_LEN   = 10   # 繰り返し検出: 最小サブストリング長
REPETITION_MIN_COUNT = 2    # 繰り返し検出: 最小繰り返し回数

# Swallow 8B generation settings
MAX_NEW_TOKENS  = 128
TEMPERATURE     = 0.0
REPEAT_PENALTY  = 1.1

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


class QualityVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class GuardResult:
    """Quality Guard の判定結果"""
    verdict: QualityVerdict
    reason_code: str          # "High Quality" | "Not recognized from only image" | etc.
    token_count: int
    rule_triggered: bool      # Stage1 ルールで判定されたか
    llm_used: bool            # Stage2 Swallow が呼ばれたか
    elapsed_sec: float
    raw_llm_response: Optional[str] = None

    # PASS 時のみ: Swallow が抽出した構造化データ
    damage_type:     Optional[str] = None  # rebar_exposure / crack / corrosion / spalling / section_loss / unknown
    severity_level:  Optional[str] = None  # high / medium / low
    location:        Optional[str] = None  # girder / deck / bearing / pier / railing / unknown
    risk_factor:     Optional[str] = None  # structural / durability / aesthetic / unknown

    def is_pass(self) -> bool:
        return self.verdict == QualityVerdict.PASS

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


# ── ユーティリティ ──────────────────────────────────────────

def count_tokens_simple(text: str) -> int:
    """CJK文字 + ASCII ブロック数でトークン数近似"""
    cjk   = len(re.findall(r"[\u3000-\u9fff\uf900-\uffef]", text))
    ascii_ = len(text.encode("ascii", errors="ignore").split())
    return cjk + ascii_


def extract_assistant_text(prediction: str) -> str:
    """USER: ... ASSISTANT: ... 形式から ASSISTANT 部分を抽出"""
    if "ASSISTANT:" in prediction:
        return prediction.split("ASSISTANT:", 1)[1].strip()
    return prediction.strip()


def detect_repetition(text: str) -> bool:
    """繰り返しパターン検出"""
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


# ── Stage 2: LLM プロンプト ─────────────────────────────────

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


# ── Llama-3 チャットフォーマット ─────────────────────────────

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
    """Swallow の VERDICT/REASON_CODE 出力をパース"""
    vm = re.search(r"VERDICT\s*:\s*(PASS|FAIL)", text, re.IGNORECASE)
    rm = re.search(r"REASON_CODE\s*:\s*(.{5,80})", text, re.IGNORECASE)
    verdict = QualityVerdict.PASS if (vm and vm.group(1).upper() == "PASS") else QualityVerdict.FAIL
    reason  = rm.group(1).strip() if rm else ("High Quality" if verdict == QualityVerdict.PASS else "Unspecified")
    return verdict, reason


def parse_structure(text: str) -> dict:
    """Swallow の JSON 構造化出力をパース"""
    defaults = {
        "damage_type":    "unknown",
        "severity_level": "medium",
        "location":       "unknown",
        "risk_factor":    "unknown",
    }
    # JSON ブロック抽出
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

class SwallowQualityGuard:
    """
    2段階 Quality Guard:
      Stage 1: ルールベース高速フィルタ
      Stage 2: Swallow 8B LLM Judge + JSON 構造化
    """

    def __init__(
        self,
        adapter_dir:      str  = DEFAULT_ADAPTER_DIR,
        base_model_id:    str  = BASE_MODEL_ID,
        low_threshold:    int  = LOW_TOKEN_THRESHOLD,
        high_threshold:   int  = HIGH_TOKEN_THRESHOLD,
        use_llm:          bool = True,
        load_in_4bit:     bool = True,
        device_map:       str  = "auto",
    ):
        self.adapter_dir   = Path(adapter_dir)
        self.base_model_id = base_model_id
        self.low_threshold  = low_threshold
        self.high_threshold = high_threshold
        self.use_llm        = use_llm

        self.model     = None
        self.tokenizer = None

        if use_llm:
            self._load_model(load_in_4bit, device_map)

    # ── モデルロード ────────────────────────────────────────

    def _load_model(self, load_in_4bit: bool, device_map: str) -> None:
        print(f"[SwallowGuard] Loading model...")
        print(f"  Base : {self.base_model_id}")
        print(f"  Adapter: {self.adapter_dir}")

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.adapter_dir),
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True,
            )

        self.model = PeftModel.from_pretrained(base, str(self.adapter_dir))
        self.model.eval()
        print("[SwallowGuard] Model loaded.")

    def unload_model(self) -> None:
        """VRAM を解放する"""
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
        print("[SwallowGuard] Model unloaded.")

    # ── LLM 呼び出し ────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - MAX_NEW_TOKENS,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=TEMPERATURE,
                repetition_penalty=REPEAT_PENALTY,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # ── Stage 1: ルールベース ────────────────────────────────

    @staticmethod
    def _check_image_path(image_path: Optional[str]) -> Optional["GuardResult"]:
        """画像ファイルの存在確認。存在しない場合 FAIL を返す。"""
        if not image_path:
            return None
        if not os.path.exists(image_path):
            return GuardResult(
                verdict=QualityVerdict.FAIL,
                reason_code="No such file or directory",
                token_count=0,
                rule_triggered=True,
                llm_used=False,
                elapsed_sec=0.0,
            )
        return None

    def _stage1_rule_check(self, text: str, token_count: int) -> Optional[GuardResult]:
        """
        高速ルールチェック。失格なら GuardResult(FAIL) を返す。
        通過なら None を返す（Stage2 へ進む）。
        """
        # 短すぎる
        if token_count < self.low_threshold:
            if not has_damage_keywords(text):
                return GuardResult(
                    verdict=QualityVerdict.FAIL,
                    reason_code="Not recognized from only image",
                    token_count=token_count,
                    rule_triggered=True,
                    llm_used=False,
                    elapsed_sec=0.0,
                )
            return GuardResult(
                verdict=QualityVerdict.FAIL,
                reason_code="Short description",
                token_count=token_count,
                rule_triggered=True,
                llm_used=False,
                elapsed_sec=0.0,
            )

        # 繰り返しパターン
        if detect_repetition(text):
            return GuardResult(
                verdict=QualityVerdict.FAIL,
                reason_code="Dirty or Noisy image",
                token_count=token_count,
                rule_triggered=True,
                llm_used=False,
                elapsed_sec=0.0,
            )

        # 損傷キーワード皆無 + 曖昧ワード
        if not has_damage_keywords(text) and has_vague_keywords(text):
            return GuardResult(
                verdict=QualityVerdict.FAIL,
                reason_code="Not recognized from only image",
                token_count=token_count,
                rule_triggered=True,
                llm_used=False,
                elapsed_sec=0.0,
            )

        return None  # Stage2 へ

    # ── Stage 2: Swallow 8B Judge ────────────────────────────

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
                verdict=verdict,
                reason_code=reason_code,
                token_count=token_count,
                rule_triggered=False,
                llm_used=True,
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
            raw_llm_response=struct_response,
            damage_type=struct["damage_type"],
            severity_level=struct["severity_level"],
            location=struct["location"],
            risk_factor=struct["risk_factor"],
        )

    # ── パブリック API ───────────────────────────────────────

    def evaluate(
        self,
        prediction: str,
        image_path: Optional[str] = None,
    ) -> GuardResult:
        """
        VLM prediction テキスト（USER: ... ASSISTANT: ... 形式 or 生テキスト）を
        Quality Guard で評価する。

        Args:
            prediction  : VLM の prediction 列の値
            image_path  : 元画像のファイルパス（指定時に存在確認を行う）

        Returns:
            GuardResult: verdict, reason_code, 構造化情報（PASS時）
        """
        t0 = time.time()

        # Stage 0: 画像ファイル存在確認
        img_check = self._check_image_path(image_path)
        if img_check is not None:
            img_check.elapsed_sec = round(time.time() - t0, 3)
            return img_check

        text = extract_assistant_text(prediction)
        token_count = count_tokens_simple(text)

        # Stage 1
        result = self._stage1_rule_check(text, token_count)
        if result is not None:
            result.elapsed_sec = round(time.time() - t0, 3)
            return result

        # Stage 2 (LLM が無効の場合は PASS として構造化なし)
        if not self.use_llm:
            return GuardResult(
                verdict=QualityVerdict.PASS,
                reason_code="High Quality (rule only)",
                token_count=token_count,
                rule_triggered=False,
                llm_used=False,
                elapsed_sec=round(time.time() - t0, 3),
            )

        return self._stage2_llm_judge(text, token_count)

    def evaluate_batch(
        self,
        predictions: list[str],
        verbose: bool = True,
    ) -> list[GuardResult]:
        """バッチ評価"""
        results = []
        total = len(predictions)
        for i, pred in enumerate(predictions):
            result = self.evaluate(pred)
            results.append(result)
            if verbose and (i + 1) % 50 == 0:
                pass_cnt  = sum(1 for r in results if r.is_pass())
                fail_cnt  = len(results) - pass_cnt
                print(f"  [{i+1:4d}/{total}] PASS={pass_cnt} FAIL={fail_cnt} "
                      f"(last: {result.verdict.value} / {result.reason_code})")
        return results


# ── スタンドアロンテスト ──────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SwallowQualityGuard")
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--rule-only",   action="store_true", help="Skip LLM (rule-based only)")
    args = parser.parse_args()

    guard = SwallowQualityGuard(
        adapter_dir=args.adapter_dir,
        use_llm=not args.rule_only,
    )

    test_cases = [
        # Low quality samples
        ("USER: この橋梁の損傷状態を詳しく説明してください。 ASSISTANT: 確認できません。",
         "Expected: FAIL / Not recognized"),
        ("USER: この橋梁の損傷状態を詳しく説明してください。 ASSISTANT: ひびわれ。",
         "Expected: FAIL / Short description"),
        # High quality sample (98 tokens 以上が必要)
        ("USER: この橋梁の損傷状態を詳しく説明してください。 ASSISTANT: "
         "主桁下面に鉄筋露出（最大幅2cm）が確認された。腐食が著しく進行しており、"
         "構造安全性に影響を及ぼす可能性がある。早期補修が必要。"
         "横桁との接合部付近において剥離・剥落も複数箇所で観察され、"
         "コンクリートの中性化が進行していることが推測される。"
         "損傷の範囲は桁全長の約30%に及んでおり、補修優先度は高い。"
         "また、桁端部の支承周辺でも錆汁の流出痕が見られ、鋼材腐食の進行が懸念される。",
         "Expected: PASS / High Quality"),
    ]

    for pred, expected in test_cases:
        print(f"\n{'─'*60}")
        print(f"  Input : {pred[:80]}...")
        print(f"  {expected}")
        result = guard.evaluate(pred)
        print(f"  Result: {result.verdict.value} / {result.reason_code}")
        if result.is_pass():
            print(f"          damage_type={result.damage_type}, "
                  f"severity={result.severity_level}, "
                  f"location={result.location}")
        print(f"  Time  : {result.elapsed_sec}s | rule={result.rule_triggered} | llm={result.llm_used}")
