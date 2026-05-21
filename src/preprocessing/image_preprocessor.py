"""
画像前処理モジュール
OpenCVを使用したノイズ除去、リサイズ、コントラスト調整
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    """前処理設定"""
    max_width: int = 1024
    max_height: int = 1024
    denoise_enabled: bool = True
    denoise_strength: int = 5
    contrast_enabled: bool = True
    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (8, 8)


class ImagePreprocessor:
    """画像前処理クラス"""
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        """
        Args:
            config: 前処理設定（Noneの場合はデフォルト値を使用）
        """
        self.config = config or PreprocessConfig()
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        画像の前処理を実行
        
        Args:
            image: 入力画像（BGR形式）
        
        Returns:
            前処理済み画像
        """
        # 1. リサイズ
        image = self._resize(image)
        
        # 2. ノイズ除去
        if self.config.denoise_enabled:
            image = self._denoise(image)
        
        # 3. コントラスト調整
        if self.config.contrast_enabled:
            image = self._enhance_contrast(image)
        
        return image
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """
        アスペクト比を保持してリサイズ
        
        Args:
            image: 入力画像
        
        Returns:
            リサイズ後の画像
        """
        h, w = image.shape[:2]
        max_w = self.config.max_width
        max_h = self.config.max_height
        
        # すでにサイズ内の場合はそのまま返す
        if w <= max_w and h <= max_h:
            return image
        
        # アスペクト比を保持してリサイズ
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        ノイズ除去（Non-local Means Denoising）
        
        Args:
            image: 入力画像
        
        Returns:
            ノイズ除去後の画像
        """
        # カラー画像の場合
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h=self.config.denoise_strength,
                hColor=self.config.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            # グレースケール画像の場合
            denoised = cv2.fastNlMeansDenoising(
                image,
                None,
                h=self.config.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
        
        return denoised
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        CLAHE（Contrast Limited Adaptive Histogram Equalization）による
        コントラスト強調
        
        Args:
            image: 入力画像
        
        Returns:
            コントラスト調整後の画像
        """
        # LAB色空間に変換
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # L（明度）チャンネルにCLAHEを適用
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clip_limit,
            tileGridSize=self.config.tile_grid_size
        )
        l_enhanced = clahe.apply(l)
        
        # LAB色空間で結合してBGRに戻す
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def preprocess_file(self, input_path: Path, output_path: Optional[Path] = None) -> np.ndarray:
        """
        ファイルから画像を読み込んで前処理
        
        Args:
            input_path: 入力画像パス
            output_path: 出力画像パス（Noneの場合は保存しない）
        
        Returns:
            前処理済み画像
        """
        # 画像読み込み
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"画像の読み込みに失敗しました: {input_path}")
        
        # 前処理実行
        processed = self.preprocess(image)
        
        # 保存
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), processed)
        
        return processed
    
    def batch_preprocess(self, input_dir: Path, output_dir: Path, pattern: str = "*.png") -> int:
        """
        ディレクトリ内の画像を一括前処理
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            pattern: ファイルパターン（glob形式）
        
        Returns:
            処理した画像数
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = sorted(input_dir.glob(pattern))
        
        for i, img_path in enumerate(image_files, 1):
            output_path = output_dir / img_path.name
            try:
                self.preprocess_file(img_path, output_path)
                print(f"[{i}/{len(image_files)}] 処理完了: {img_path.name}")
            except Exception as e:
                print(f"[{i}/{len(image_files)}] エラー: {img_path.name} - {e}")
        
        return len(image_files)


def load_config_from_yaml(config: dict) -> PreprocessConfig:
    """YAMLの設定辞書からPreprocessConfigを生成"""
    preprocessing_config = config.get('preprocessing', {})
    
    resize_config = preprocessing_config.get('resize', {})
    denoise_config = preprocessing_config.get('denoise', {})
    contrast_config = preprocessing_config.get('contrast', {})
    
    return PreprocessConfig(
        max_width=resize_config.get('max_width', 1024),
        max_height=resize_config.get('max_height', 1024),
        denoise_enabled=denoise_config.get('enabled', True),
        denoise_strength=denoise_config.get('strength', 5),
        contrast_enabled=contrast_config.get('enabled', True),
        clip_limit=contrast_config.get('clip_limit', 2.0),
        tile_grid_size=tuple(contrast_config.get('tile_grid_size', [8, 8]))
    )


if __name__ == "__main__":
    # 簡易テスト
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 3:
        print("使用法: python image_preprocessor.py <入力ディレクトリ> <出力ディレクトリ>")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    preprocessor = ImagePreprocessor()
    count = preprocessor.batch_preprocess(input_dir, output_dir)
    print(f"\n処理完了: {count}枚の画像を前処理しました")
