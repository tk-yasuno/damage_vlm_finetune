"""
llama-cpp-python用の汎用GGUF VLMアナライザー v0.2
Granite Vision / Qwen2-VL / LLaVA のGGUFモデルに対応
"""
import os
import sys
from pathlib import Path
from typing import Union, Optional, List
from dataclasses import dataclass
from PIL import Image
import torch

# llama-cpp-pythonのログ抑制
os.environ['LLAMA_CPP_LOG_LEVEL'] = '0'

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


@dataclass
class GGUFVLMConfig:
    """GGUF VLMモデル設定"""
    model_path: str
    mmproj_path: Optional[str] = None  # Granite/Qwen2-VLはmmproj不要の可能性
    n_gpu_layers: int = -1  # -1 = すべてGPU
    n_ctx: int = 4096
    max_tokens: int = 300
    temperature: float = 0.3
    verbose: bool = False


class GGUFVLMAnalyzer:
    """
    llama-cpp-python + GGUF を使用した汎用VLM分析クラス
    Granite Vision / Qwen2-VL / LLaVA に対応
    """
    
    DEFAULT_PROMPT = """You are a civil engineering expert specializing in bridge inspection.
Describe the structural damage visible in this image using technical terminology.

Focus on the following:
- Damage type (crack, rebar exposure, corrosion, spalling, section loss)
- Severity level (minor, moderate, severe)
- Location and extent (girder, deck, bearing, pier, etc.)
- Structural risk assessment

Be precise and avoid speculation. Use Japanese for the description."""
    
    def __init__(self, config: GGUFVLMConfig):
        """
        初期化
        
        Args:
            config: GGUF VLMモデル設定
        """
        self.config = config
        
        model_name = Path(config.model_path).stem
        print(f"GGUF VLMモデルを読み込み中... ({model_name})")
        print(f"GPU layers: {config.n_gpu_layers}, Context: {config.n_ctx}")
        
        # Chat handlerの設定
        if config.mmproj_path:
            # LLaVA形式（mmproj使用）
            print(f"MM-Proj: {config.mmproj_path}")
            self.chat_handler = Llava15ChatHandler(
                clip_model_path=config.mmproj_path,
                verbose=config.verbose
            )
        else:
            # Granite/Qwen2-VL形式（組み込みVision）
            print("Vision: 組み込みモデル")
            self.chat_handler = Llava15ChatHandler(verbose=config.verbose)
        
        # モデルの読み込み
        self.model = Llama(
            model_path=config.model_path,
            chat_handler=self.chat_handler,
            n_gpu_layers=config.n_gpu_layers,
            n_ctx=config.n_ctx,
            verbose=config.verbose,
            logits_all=True,
        )
        
        print("✓ GGUF VLMモデル読み込み完了\n")
    
    def analyze_image(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        画像を分析して損傷説明を生成
        
        Args:
            image: 画像パス or PIL Image
            prompt: カスタムプロンプト
            max_tokens: 最大トークン数
            temperature: サンプリング温度
            
        Returns:
            損傷説明テキスト
        """
        # 画像パスの取得
        if isinstance(image, (str, Path)):
            image_path = str(image)
        else:
            # PIL Imageの場合は一時保存
            temp_path = Path("temp_gguf_image.png")
            image.save(temp_path)
            image_path = str(temp_path)
        
        # プロンプトの準備
        user_prompt = prompt if prompt is not None else self.DEFAULT_PROMPT
        
        # llama-cpp-python用のメッセージ形式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{Path(image_path).absolute()}"}},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        # 生成パラメータ
        gen_params = {
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
        }
        
        # 推論実行（標準出力を抑制）
        try:
            # llama.cppのC++ログを一時的に抑制
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                response = self.model.create_chat_completion(
                    messages=messages,
                    **gen_params
                )
            
            # レスポンスからテキストを抽出
            generated_text = response['choices'][0]['message']['content']
            return generated_text.strip()
            
        except Exception as e:
            error_msg = f"GGUF VLM 推論エラー: {str(e)}"
            print(error_msg)
            return f"[エラー] {error_msg}"
        
        finally:
            # 一時ファイルの削除
            if isinstance(image, Image.Image) and temp_path.exists():
                temp_path.unlink()
    
    def analyze_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        show_progress: bool = True
    ) -> List[str]:
        """
        複数画像を一括分析
        
        Args:
            images: 画像リスト
            prompt: カスタムプロンプト
            max_tokens: 最大トークン数
            temperature: サンプリング温度
            show_progress: 進捗表示
            
        Returns:
            損傷説明リスト
        """
        results = []
        
        for i, image in enumerate(images, 1):
            if show_progress:
                image_name = Path(image).name if isinstance(image, (str, Path)) else f"image_{i}"
                print(f"[{i}/{len(images)}] {image_name} 処理中...")
            
            description = self.analyze_image(image, prompt, max_tokens, temperature)
            results.append(description)
        
        return results


def test_gguf_vlm():
    """GGUF VLMモデルのテスト"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GGUF VLM テスト')
    parser.add_argument('--model', required=True, help='GGUFモデルパス')
    parser.add_argument('--mmproj', help='MM-Projパス（LLaVAの場合）')
    parser.add_argument('--image', required=True, help='テスト画像パス')
    args = parser.parse_args()
    
    # 設定
    config = GGUFVLMConfig(
        model_path=args.model,
        mmproj_path=args.mmproj
    )
    
    # 初期化
    analyzer = GGUFVLMAnalyzer(config)
    
    # 分析
    print(f"画像分析: {args.image}")
    result = analyzer.analyze_image(args.image)
    
    print("\n【結果】")
    print(result)


if __name__ == "__main__":
    test_gguf_vlm()
