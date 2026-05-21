"""
llama-cpp-python統合モジュール
GGUFフォーマットのLLaVAモデルを使用して画像から損傷説明テキストを生成
"""
import os
from PIL import Image
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64
from io import BytesIO

# llama.cppのログレベルを設定（文字化け防止）
os.environ['LLAMA_CPP_LOG_LEVEL'] = '0'  # 0=エラーのみ, 1=警告, 2=情報


@dataclass
class LlamaCppVisionConfig:
    """llama-cpp-python Vision設定"""
    model_path: str = "models/ggml-model-q4_k.gguf"
    mmproj_path: str = "models/mmproj-model-f16.gguf"
    n_gpu_layers: int = -1  # -1 = すべてのレイヤーをGPUに
    n_ctx: int = 4096  # コンテキストサイズ
    max_tokens: int = 300
    temperature: float = 0.3
    top_p: float = 0.9
    verbose: bool = False


class LlamaCppVisionAnalyzer:
    """llama-cpp-pythonを使用した損傷画像分析クラス"""
    
    # デフォルトのプロンプトテンプレート
    DEFAULT_PROMPT = """あなたは橋梁点検の専門家です。
次の画像に写る損傷を、土木構造物の専門用語を用いて簡潔に説明してください。

必ず以下の情報を含めてください：
- 損傷の種類（ひび割れ、鉄筋露出、腐食、剥離、断面欠損など）
- 損傷の程度（軽微、中程度、重度）
- 損傷の位置・範囲
- 構造上のリスク"""
    
    def __init__(self, config: Optional[LlamaCppVisionConfig] = None):
        """
        Args:
            config: llama-cpp-python Vision設定
        """
        self.config = config or LlamaCppVisionConfig()
        
        print(f"llama-cpp-python LLaVAモデルを読み込み中...")
        print(f"Model: {self.config.model_path}")
        print(f"MMProj: {self.config.mmproj_path}")
        print(f"GPU Layers: {self.config.n_gpu_layers} (-1 = すべて)")
        
        # C++レベルのログを抑制するため、一時的にstdoutをリダイレクト
        import sys
        import io
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            # ログ抑制中は出力を捨てる
            if not self.config.verbose:
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
            
            # LLaVA Chat Handlerの初期化
            self.chat_handler = Llava15ChatHandler(
                clip_model_path=self.config.mmproj_path,
                verbose=False  # 強制的にFalse
            )
            
            # Llamaモデルの初期化
            self.model = Llama(
                model_path=self.config.model_path,
                chat_handler=self.chat_handler,
                n_gpu_layers=self.config.n_gpu_layers,
                n_ctx=self.config.n_ctx,
                verbose=False,  # 強制的にFalse
                logits_all=True  # LLaVAに必要
            )
        finally:
            # 出力を元に戻す
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
        print("OK: モデル読み込み完了")
    
    def _image_to_data_url(self, image: Image.Image) -> str:
        """
        PIL ImageをData URLに変換
        
        Args:
            image: PIL Image
            
        Returns:
            Data URL文字列
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
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
            prompt: カスタムプロンプト（Noneの場合はデフォルト使用）
            max_tokens: 最大トークン数
            temperature: サンプリング温度
            
        Returns:
            損傷説明テキスト
        """
        # 画像の読み込み
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        # Data URLに変換
        image_url = self._image_to_data_url(pil_image)
        
        # プロンプトの準備
        user_prompt = prompt if prompt is not None else self.DEFAULT_PROMPT
        
        # パラメータの設定
        gen_params = {
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
            "top_p": self.config.top_p,
        }
        
        # LLaVA Chat形式でメッセージを作成
        messages = [
            {
                "role": "system",
                "content": "あなたは橋梁点検の専門家です。"
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        # 推論実行（C++ログを抑制）
        import sys
        import io
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            # ログ抑制: stdoutをリダイレクト
            if not self.config.verbose:
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
            
            response = self.model.create_chat_completion(
                messages=messages,
                **gen_params
            )
            
        finally:
            # 出力を元に戻す
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
        try:
            # レスポンスからテキストを抽出
            description = response["choices"][0]["message"]["content"]
            return description.strip()
            
        except Exception as e:
            error_msg = f"Vision推論エラー: {str(e)}"
            print(error_msg)
            return f"[エラー] {error_msg}"
    
    def analyze_batch(
        self,
        images: list[Union[str, Path, Image.Image]],
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        show_progress: bool = True
    ) -> list[str]:
        """
        複数画像の一括分析
        
        Args:
            images: 画像パスまたはPIL Imageのリスト
            prompt: カスタムプロンプト
            max_tokens: 最大トークン数
            temperature: サンプリング温度
            show_progress: 進捗表示
            
        Returns:
            損傷説明テキストのリスト
        """
        results = []
        total = len(images)
        
        for i, img in enumerate(images, 1):
            if show_progress:
                print(f"処理中: {i}/{total}")
            
            description = self.analyze_image(
                image=img,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            results.append(description)
        
        return results
    
    def __del__(self):
        """クリーンアップ"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'chat_handler'):
            del self.chat_handler


def main():
    """テスト実行"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python llama_cpp_vision.py <image_path>")
        sys.exit(1)
    
    # アナライザー初期化
    analyzer = LlamaCppVisionAnalyzer()
    
    # 画像分析
    image_path = sys.argv[1]
    print(f"\n画像分析中: {image_path}")
    description = analyzer.analyze_image(image_path)
    
    print("\n=== 分析結果 ===")
    print(description)


if __name__ == "__main__":
    main()
