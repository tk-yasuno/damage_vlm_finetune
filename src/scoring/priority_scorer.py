"""
補修優先度スコアリングエンジン
ルールベース＋GAM補正
"""
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import pickle

# GAMはオプショナル
try:
    from pygam import LinearGAM, s, f
    GAM_AVAILABLE = True
except ImportError:
    GAM_AVAILABLE = False
    print("警告: pygamがインストールされていません。GAM補正は無効です。")


@dataclass
class ScoringConfig:
    """スコアリング設定"""
    rules_file: str = "models/scoring_rules.yaml"
    weight_damage_type: float = 0.35
    weight_severity: float = 0.40
    weight_location: float = 0.15
    weight_risk: float = 0.10
    gam_enabled: bool = False
    gam_model_path: Optional[str] = None


@dataclass
class PriorityScore:
    """優先度スコア結果"""
    raw_score: float  # 0.0-1.0の生スコア
    priority_level: int  # 1-5の優先度
    priority_description: str
    damage_type_score: float
    severity_score: float
    location_score: float
    risk_score: float
    combination_bonus: float
    gam_adjusted_score: Optional[float] = None
    
    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return asdict(self)


class PriorityScorer:
    """補修優先度スコアリングクラス"""
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        """
        Args:
            config: スコアリング設定
        """
        self.config = config or ScoringConfig()
        
        # ルール読み込み
        self.rules = self._load_rules()
        
        # GAMモデル読み込み
        self.gam_model = None
        if self.config.gam_enabled and self.config.gam_model_path:
            self.gam_model = self._load_gam_model()
    
    def _load_rules(self) -> Dict[str, Any]:
        """スコアリングルールを読み込む"""
        rules_path = Path(self.config.rules_file)
        
        if not rules_path.exists():
            print(f"警告: ルールファイルが見つかりません: {rules_path}")
            return self._get_default_rules()
        
        with open(rules_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_default_rules(self) -> Dict[str, Any]:
        """デフォルトルールを返す"""
        return {
            'damage_type_scores': {
                'section_loss': 1.0,
                'rebar_exposure': 0.95,
                'corrosion': 0.85,
                'spalling': 0.75,
                'crack': 0.60,
                'unknown': 0.50
            },
            'severity_scores': {
                'high': 1.0,
                'medium': 0.6,
                'low': 0.3
            },
            'location_scores': {
                'girder_bottom': 1.0,
                'girder': 0.95,
                'bearing': 0.90,
                'deck': 0.75,
                'pier': 0.70,
                'support': 0.85,
                'unknown': 0.50
            },
            'risk_scores': {
                'structural': 1.0,
                'durability': 0.75,
                'aesthetic': 0.30,
                'unknown': 0.50
            },
            'combination_bonuses': [],
            'priority_thresholds': {
                5: 0.85,
                4: 0.70,
                3: 0.50,
                2: 0.35,
                1: 0.00
            },
            'priority_descriptions': {
                5: "即時補修が必要（構造安全性に重大な影響）",
                4: "早期補修が必要（6ヶ月以内推奨）",
                3: "計画的補修が必要（1〜2年以内）",
                2: "経過観察が必要（定期点検で監視）",
                1: "記録のみ（軽微な損傷）"
            }
        }
    
    def _load_gam_model(self) -> Optional[Any]:
        """GAMモデルを読み込む"""
        if not GAM_AVAILABLE:
            print("警告: pygamが利用できないため、GAMモデルは読み込めません")
            return None
        
        model_path = Path(self.config.gam_model_path)
        if not model_path.exists():
            print(f"警告: GAMモデルが見つかりません: {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"GAMモデル読み込みエラー: {e}")
            return None
    
    def calculate_score(
        self,
        damage_type: str,
        severity: str,
        location: str,
        risk: str
    ) -> PriorityScore:
        """
        優先度スコアを計算
        
        Args:
            damage_type: 損傷種別
            severity: 重症度
            location: 位置
            risk: リスク種別
        
        Returns:
            優先度スコア結果
        """
        # 各要素のスコア取得
        dt_score = self.rules['damage_type_scores'].get(damage_type, 0.5)
        sev_score = self.rules['severity_scores'].get(severity, 0.5)
        loc_score = self.rules['location_scores'].get(location, 0.5)
        risk_score = self.rules['risk_scores'].get(risk, 0.5)
        
        # 加重平均スコア
        raw_score = (
            dt_score * self.config.weight_damage_type +
            sev_score * self.config.weight_severity +
            loc_score * self.config.weight_location +
            risk_score * self.config.weight_risk
        )
        
        # 組み合わせボーナス
        bonus = self._calculate_combination_bonus(
            damage_type, severity, location, risk
        )
        raw_score = min(1.0, raw_score + bonus)
        
        # GAM補正（オプション）
        gam_score = None
        if self.gam_model is not None:
            gam_score = self._apply_gam_correction(
                damage_type, severity, location, risk, raw_score
            )
            final_score = gam_score
        else:
            final_score = raw_score
        
        # 優先度レベル判定
        priority_level = self._score_to_priority(final_score)
        priority_desc = self.rules['priority_descriptions'].get(
            priority_level,
            "優先度不明"
        )
        
        return PriorityScore(
            raw_score=raw_score,
            priority_level=priority_level,
            priority_description=priority_desc,
            damage_type_score=dt_score,
            severity_score=sev_score,
            location_score=loc_score,
            risk_score=risk_score,
            combination_bonus=bonus,
            gam_adjusted_score=gam_score
        )
    
    def _calculate_combination_bonus(
        self,
        damage_type: str,
        severity: str,
        location: str,
        risk: str
    ) -> float:
        """組み合わせボーナスを計算"""
        bonus = 0.0
        
        for comb in self.rules.get('combination_bonuses', []):
            condition = comb.get('condition', {})
            
            # 条件マッチング
            match = True
            if 'damage_type' in condition and condition['damage_type'] != damage_type:
                match = False
            if 'severity' in condition and condition['severity'] != severity:
                match = False
            if 'location' in condition and condition['location'] != location:
                match = False
            if 'risk' in condition and condition['risk'] != risk:
                match = False
            
            if match:
                bonus += comb.get('bonus', 0.0)
        
        return bonus
    
    def _score_to_priority(self, score: float) -> int:
        """スコアを優先度レベルに変換"""
        thresholds = self.rules.get('priority_thresholds', {
            5: 0.85, 4: 0.70, 3: 0.50, 2: 0.35, 1: 0.00
        })
        
        # 降順でチェック
        for level in [5, 4, 3, 2, 1]:
            if score >= thresholds.get(level, 0.0):
                return level
        
        return 1
    
    def _apply_gam_correction(
        self,
        damage_type: str,
        severity: str,
        location: str,
        risk: str,
        raw_score: float
    ) -> float:
        """GAMモデルで補正（実装例）"""
        # 実際の実装ではカテゴリ変数をone-hotエンコーディングなどする必要がある
        # ここでは簡易版
        try:
            # 特徴量作成（例）
            features = np.array([[
                self._encode_category(damage_type, 'damage_type'),
                self._encode_category(severity, 'severity'),
                self._encode_category(location, 'location'),
                self._encode_category(risk, 'risk'),
                raw_score
            ]])
            
            # 予測
            prediction = self.gam_model.predict(features)[0]
            return np.clip(prediction, 0.0, 1.0)
            
        except Exception as e:
            print(f"GAM補正エラー: {e}")
            return raw_score
    
    def _encode_category(self, value: str, category: str) -> float:
        """カテゴリ値を数値エンコーディング（簡易版）"""
        # 実際はone-hotやラベルエンコーディングを使うべき
        encodings = {
            'damage_type': {
                'section_loss': 5, 'rebar_exposure': 4, 'corrosion': 3,
                'spalling': 2, 'crack': 1, 'unknown': 0
            },
            'severity': {'high': 2, 'medium': 1, 'low': 0},
            'location': {
                'girder_bottom': 5, 'girder': 4, 'bearing': 3,
                'deck': 2, 'pier': 1, 'support': 2, 'unknown': 0
            },
            'risk': {'structural': 2, 'durability': 1, 'aesthetic': 0, 'unknown': 0}
        }
        
        return float(encodings.get(category, {}).get(value, 0))
    
    def batch_score(
        self,
        damage_list: List[Dict[str, str]]
    ) -> List[PriorityScore]:
        """
        複数の損傷を一括スコアリング
        
        Args:
            damage_list: 損傷データのリスト
                各要素は {'damage_type', 'severity', 'location', 'risk'} を含む辞書
        
        Returns:
            スコア結果のリスト
        """
        results = []
        
        for i, damage in enumerate(damage_list, 1):
            try:
                score = self.calculate_score(
                    damage.get('damage_type', 'unknown'),
                    damage.get('severity', 'low'),
                    damage.get('location', 'unknown'),
                    damage.get('risk', 'unknown')
                )
                results.append(score)
            except Exception as e:
                print(f"[{i}] スコアリングエラー: {e}")
                results.append(None)
        
        return results


def load_config_from_yaml(config: dict) -> ScoringConfig:
    """YAMLの設定辞書からScoringConfigを生成"""
    scoring_config = config.get('scoring', {})
    weights = scoring_config.get('weights', {})
    gam_config = scoring_config.get('gam', {})
    
    return ScoringConfig(
        rules_file=scoring_config.get('rules_file', 'models/scoring_rules.yaml'),
        weight_damage_type=weights.get('damage_type', 0.35),
        weight_severity=weights.get('severity', 0.40),
        weight_location=weights.get('location', 0.15),
        weight_risk=weights.get('risk', 0.10),
        gam_enabled=gam_config.get('enabled', False),
        gam_model_path=gam_config.get('model_path')
    )


if __name__ == "__main__":
    # 簡易テスト
    scorer = PriorityScorer()
    
    # テストケース
    test_cases = [
        {
            'damage_type': 'rebar_exposure',
            'severity': 'high',
            'location': 'girder_bottom',
            'risk': 'structural'
        },
        {
            'damage_type': 'crack',
            'severity': 'low',
            'location': 'deck',
            'risk': 'durability'
        }
    ]
    
    print("=== スコアリングテスト ===")
    for i, case in enumerate(test_cases, 1):
        result = scorer.calculate_score(**case)
        print(f"\n--- ケース {i} ---")
        print(f"入力: {case}")
        print(f"生スコア: {result.raw_score:.3f}")
        print(f"優先度: {result.priority_level} - {result.priority_description}")
