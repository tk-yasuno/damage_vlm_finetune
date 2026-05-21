"""
設定ファイル読み込みユーティリティ
"""
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """設定管理クラス"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """YAMLファイルから設定を読み込む"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """ドット記法でネストした設定値を取得
        
        Args:
            key: 設定キー（例: "preprocessing.resize.max_width"）
            default: デフォルト値
        
        Returns:
            設定値
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """辞書形式でのアクセスをサポート"""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """in演算子をサポート"""
        return key in self.config


def get_project_root() -> Path:
    """プロジェクトルートディレクトリを取得"""
    current = Path(__file__).resolve()
    # srcディレクトリの親がプロジェクトルート
    while current.name != 'damage_text_score' and current.parent != current:
        current = current.parent
    return current


def ensure_dir(path: Path) -> Path:
    """ディレクトリが存在することを保証"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
