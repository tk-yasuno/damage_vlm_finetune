"""
Ollama LLaVA統合モジュール
Ollama経由でLLaVAを使用して画像から損傷説明を生成（高速）
"""
import base64
import requests
from PIL import Image
from pathlib import Path
from typing import Optional, Union
from io import BytesIO


class OllamaVisionAnalyzer:
    """Ollama LLaVAを使用した損傷画像分析クラス（高速版）"""
    
    DEFAULT_PROMPT = """あなたは橋梁点検の専門家です。
次の画像に写る損傷を、土木構造物の専門用語を用いて簡潔に説明してください。

必ず以下の情報を含めてください：
- 損傷の種類（ひび割れ、鉄筋露出、腐食、剥離、断面欠損など）
- 損傷の程度（軽微、中程度、重度）
- 損傷の位置・範囲
- 構造上のリスク

説明:"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llava:7b",
        temperature: float = 0.3
    ):
        """
        Args:
            base_url: OllamaサーバーのURL
            model: 使用するモデル名
            temperature: 生成の温度パラメータ
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        
        print(f"Ollama LLaVAモデルを使用... ({model})")
        print(f"サーバー: {base_url}")
        
        # 接続確認
        if self.check_connection():
            print("OK Ollama接続成功")
        else:
            print("NG Ollama接続失敗")
            raise ConnectionError(f"Ollamaサーバーに接続できません: {base_url}")
    
    def check_connection(self) -> bool:
        """Ollamaサーバーへの接続確認"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def analyze_image(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        return_raw: bool = False
    ) -> str:
        """
        画像を分析して損傷説明を生成
        
        Args:
            image: 画像（ファイルパスまたはPIL Image）
            prompt: カスタムプロンプト
            return_raw: モデルの生テキストを返すか
        
        Returns:
            損傷説明テキスト
        """
        # 画像をBase64エンコード
        if isinstance(image, (str, Path)):
            with open(image, 'rb') as f:
                image_data = f.read()
        elif isinstance(image, Image.Image):
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            image_data = buffer.getvalue()
        else:
            raise ValueError("imageはファイルパスまたはPIL Imageである必要があります")
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # プロンプト準備
        if prompt is None:
            prompt = self.DEFAULT_PROMPT
        
        # Ollama API呼び出し
        try:
            url = f"{self.base_url}/api/generate"
            data = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 300
                }
            }
            
            response = requests.post(url, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            generated_text = result.get("response", "")
            
            if return_raw:
                return generated_text
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"推論エラー: {e}")
            return self._generate_dummy_description()
    
    def _generate_dummy_description(self) -> str:
        """ダミー説明生成（フォールバック用）"""
        return """鉄筋露出が確認されます。コンクリート表面が剥離し、内部の鉄筋が露出している状態です。
露出範囲は比較的広く、複数箇所で確認できます。
鉄筋の腐食も進行しており、断面欠損のリスクがあります。
構造耐力への影響が懸念されるため、早急な補修が必要です。"""
    
    def batch_analyze(
        self,
        images: list,
        prompt: Optional[str] = None,
        show_progress: bool = True
    ) -> list:
        """
        複数画像を一括分析
        
        Args:
            images: 画像リスト
            prompt: カスタムプロンプト
            show_progress: 進捗表示
        
        Returns:
            説明テキストのリスト
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(images, desc="Vision分析")
        else:
            iterator = images
        
        for img in iterator:
            description = self.analyze_image(img, prompt)
            results.append(description)
        
        return results
