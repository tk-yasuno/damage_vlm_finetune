"""
JSON構造化モジュール
損傷説明テキストをJSON形式に構造化
"""
import torch
import json
import re
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class StructuringConfig:
    """構造化設定"""
    model_name: str = "tokyotech-llm/Swallow-7b-instruct-v0.1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 500
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True
    use_ollama: bool = False  # Ollamaを使用するか
    ollama_model: str = "swallow8b-lora-n4000-v09-q4:latest"  # Ollamaモデル名
    ollama_url: str = "http://localhost:11434"  # OllamaサーバーURL


@dataclass
class DamageStructure:
    """損傷構造化データ"""
    damage_type: str  # rebar_exposure, crack, corrosion, spalling, section_loss
    severity: str  # low, medium, high
    location: str  # girder, deck, bearing, pier, girder_bottom, support
    risk: str  # structural, durability, aesthetic
    description_ja: str
    key_features: List[str]
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DamageStructure':
        """辞書から生成"""
        return cls(**data)


class JSONStructurer:
    """テキストをJSON構造化するクラス"""
    
    # デフォルトのプロンプトテンプレート
    DEFAULT_PROMPT_TEMPLATE = """以下の橋梁損傷説明文を、必ずJSON形式のみで出力してください。説明文や注釈は不要です。

損傷説明文:
{description}

以下の正確なJSON形式で出力してください（JSONのみ、他の文章は一切含めないこと）：
{{
  "damage_type": "rebar_exposure",
  "severity": "high",
  "location": "girder",
  "risk": "structural",
  "description_ja": "元の説明文をここに",
  "key_features": ["特徴1", "特徴2"]
}}

選択肢:
- damage_type: rebar_exposure, crack, corrosion, spalling, section_loss のいずれか
- severity: low, medium, high のいずれか
- location: girder, deck, bearing, pier, girder_bottom, support のいずれか
- risk: structural, durability, aesthetic のいずれか

JSON出力:"""
    
    def __init__(self, config: Optional[StructuringConfig] = None):
        """
        Args:
            config: 構造化設定（Noneの場合はデフォルト値を使用）
        """
        self.config = config or StructuringConfig()
        
        # Ollama使用の場合
        if self.config.use_ollama:
            print(f"Ollama経由でJSON構造化モデルを使用... ({self.config.ollama_model})")
            try:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from utils.ollama_client import OllamaClient
                
                self.ollama_client = OllamaClient(
                    base_url=self.config.ollama_url,
                    model=self.config.ollama_model
                )
                
                if self.ollama_client.check_connection():
                    print("OK Ollama接続成功")
                    self.tokenizer = None
                    self.model = None
                else:
                    print("NG Ollama接続失敗 - ルールベースにフォールバック")
                    self.ollama_client = None
                    self.tokenizer = None
                    self.model = None
            except Exception as e:
                print(f"Ollama初期化エラー: {e}")
                self.ollama_client = None
                self.tokenizer = None
                self.model = None
            return
        
        # HuggingFace Transformers使用の場合
        self.ollama_client = None
        self.device = torch.device(self.config.device)
        
        print(f"JSON構造化モデルを読み込み中... ({self.config.model_name})")
        print(f"デバイス: {self.device}")
        
        # モデルとトークナイザーの読み込み
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # GPU16GBの場合はfp16で読み込み
            if self.config.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True
                )
                self.model.to(self.device)
            
            self.model.eval()
            print("モデルの読み込みが完了しました")
            
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            print("代替モード（ルールベース）で初期化します")
            self.tokenizer = None
            self.model = None
    
    def structure_text(
        self,
        description: str,
        prompt_template: Optional[str] = None
    ) -> DamageStructure:
        """
        説明テキストをJSON構造化
        
        Args:
            description: 損傷説明テキスト
            prompt_template: カスタムプロンプトテンプレート
        
        Returns:
            構造化された損傷データ
        """
        # Ollama使用の場合
        if self.ollama_client is not None:
            return self._ollama_structuring(description, prompt_template)
        
        # モデルが読み込まれていない場合はルールベースで構造化
        if self.model is None or self.tokenizer is None:
            return self._rule_based_structuring(description)
        if self.model is None or self.tokenizer is None:
            return self._rule_based_structuring(description)
        
        # プロンプト生成
        if prompt_template is None:
            prompt_template = self.DEFAULT_PROMPT_TEMPLATE
        
        prompt = prompt_template.format(description=description)
        
        # 推論実行
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated_text = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                # JSONを抽出
                json_data = self._extract_json(generated_text)
                
                if json_data:
                    return DamageStructure(**json_data)
                else:
                    # JSONパースに失敗した場合はルールベースにフォールバック
                    return self._rule_based_structuring(description)
                    
        except Exception as e:
            print(f"構造化エラー: {e}")
            return self._rule_based_structuring(description)
    
    def _ollama_structuring(
        self,
        description: str,
        prompt_template: Optional[str] = None
    ) -> DamageStructure:
        """
        Ollamaを使用してテキストをJSON構造化
        
        Args:
            description: 損傷説明テキスト
            prompt_template: カスタムプロンプトテンプレート
        
        Returns:
            構造化された損傷データ
        """
        # プロンプト生成
        if prompt_template is None:
            prompt_template = self.DEFAULT_PROMPT_TEMPLATE
        
        prompt = prompt_template.format(description=description)
        
        try:
            # Ollama経由で生成
            generated_text = self.ollama_client.generate(
                prompt,
                temperature=self.config.temperature,
                num_predict=self.config.max_new_tokens
            )
            
            # デバッグ: Ollama出力を表示
            if generated_text:
                print(f"[DEBUG] Ollama生成テキスト (最初の200文字): {generated_text[:200]}")
            
            # JSONを抽出
            json_data = self._extract_json(generated_text)
            
            if json_data:
                return DamageStructure(**json_data)
            else:
                # JSONパースに失敗した場合はルールベースにフォールバック
                print("Ollama出力のJSON解析失敗 - ルールベースにフォールバック")
                print(f"[DEBUG] 解析に失敗した出力: {generated_text[:500] if generated_text else 'empty'}")
                return self._rule_based_structuring(description)
        
        except Exception as e:
            print(f"Ollama構造化エラー: {e}")
            return self._rule_based_structuring(description)
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """生成テキストからJSON部分を抽出してパース"""
        if not text:
            return None
        
        # 複数のJSON抽出パターンを試す
        patterns = [
            # 標準的なJSONブロック
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            # コードブロック内のJSON
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            # JSONのみの出力
            r'(\{.*\})',
        ]
        
        for pattern in patterns:
            json_matches = re.finditer(pattern, text, re.DOTALL)
            for json_match in json_matches:
                json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)
                json_str = json_str.strip()
                
                try:
                    data = json.loads(json_str)
                    
                    # 必須フィールドの検証
                    required_fields = ['damage_type', 'severity', 'location', 'risk']
                    if all(field in data for field in required_fields):
                        # key_featuresがリストでない場合は空リストに
                        if 'key_features' not in data or not isinstance(data['key_features'], list):
                            data['key_features'] = []
                        
                        # description_jaがない場合は空文字列に
                        if 'description_ja' not in data:
                            data['description_ja'] = ""
                        
                        return data
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _rule_based_structuring(self, description: str) -> DamageStructure:
        """ルールベースでの構造化（フォールバック用）"""
        description_lower = description.lower()
        
        # 損傷種別判定
        if "鉄筋" in description or "露出" in description or "rebar" in description_lower:
            damage_type = "rebar_exposure"
        elif "断面欠損" in description or "section loss" in description_lower:
            damage_type = "section_loss"
        elif "腐食" in description or "corrosion" in description_lower or "錆" in description:
            damage_type = "corrosion"
        elif "剥離" in description or "spalling" in description_lower:
            damage_type = "spalling"
        elif "ひび割れ" in description or "亀裂" in description or "crack" in description_lower:
            damage_type = "crack"
        else:
            damage_type = "unknown"
        
        # 重症度判定
        if any(word in description for word in ["重度", "深刻", "著しい", "広範囲", "進行"]):
            severity = "high"
        elif any(word in description for word in ["中程度", "やや", "一部"]):
            severity = "medium"
        else:
            severity = "low"
        
        # 位置判定
        if "主桁下面" in description or "girder bottom" in description_lower:
            location = "girder_bottom"
        elif "主桁" in description or "桁" in description or "girder" in description_lower:
            location = "girder"
        elif "支承" in description or "bearing" in description_lower:
            location = "bearing"
        elif "床版" in description or "deck" in description_lower:
            location = "deck"
        elif "橋脚" in description or "pier" in description_lower:
            location = "pier"
        elif "支持" in description or "support" in description_lower:
            location = "support"
        else:
            location = "unknown"
        
        # リスク判定
        if any(word in description for word in ["構造", "耐力", "安全", "崩壊"]):
            risk = "structural"
        elif any(word in description for word in ["耐久", "劣化", "寿命"]):
            risk = "durability"
        else:
            risk = "aesthetic"
        
        # 特徴抽出（簡易版）
        key_features = []
        if "鉄筋" in description:
            key_features.append("鉄筋露出")
        if "腐食" in description or "錆" in description:
            key_features.append("腐食")
        if "ひび割れ" in description:
            key_features.append("ひび割れ")
        
        return DamageStructure(
            damage_type=damage_type,
            severity=severity,
            location=location,
            risk=risk,
            description_ja=description,
            key_features=key_features,
            confidence=0.7  # ルールベースは信頼度を下げる
        )
    
    def batch_structure(
        self,
        descriptions: List[str],
        output_dir: Optional[Path] = None
    ) -> List[DamageStructure]:
        """
        複数の説明テキストを一括構造化
        
        Args:
            descriptions: 説明テキストのリスト
            output_dir: 出力ディレクトリ（Noneの場合は保存しない）
        
        Returns:
            構造化データのリスト
        """
        results = []
        
        for i, desc in enumerate(descriptions, 1):
            print(f"[{i}/{len(descriptions)}] 構造化中...")
            
            try:
                structure = self.structure_text(desc)
                results.append(structure)
                
                # 個別保存
                if output_dir is not None:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    json_path = output_dir / f"structured_{i:03d}.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(structure.to_dict(), f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                print(f"エラー: {e}")
                results.append(None)
        
        return results


def load_config_from_yaml(config: dict) -> StructuringConfig:
    """YAMLの設定辞書からStructuringConfigを生成"""
    structuring_config = config.get('structuring', {})
    
    return StructuringConfig(
        model_name=structuring_config.get('model_name', 'tokyotech-llm/Swallow-7b-instruct-v0.1'),
        device=structuring_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        max_new_tokens=structuring_config.get('max_new_tokens', 500),
        temperature=structuring_config.get('temperature', 0.1),
        top_p=structuring_config.get('top_p', 0.9),
        do_sample=structuring_config.get('do_sample', True)
    )


if __name__ == "__main__":
    # 簡易テスト
    test_description = """鉄筋露出が確認されます。コンクリート表面が剥離し、内部の鉄筋が露出している状態です。
露出範囲は比較的広く、複数箇所で確認できます。
鉄筋の腐食も進行しており、断面欠損のリスクがあります。
構造耐力への影響が懸念されるため、早急な補修が必要です。"""
    
    structurer = JSONStructurer()
    result = structurer.structure_text(test_description)
    
    print("\n=== 構造化結果 ===")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
