"""
Ollama統合モジュール
ローカルのOllamaサーバーを使用してLLMを実行
"""
import json
import requests
from typing import Optional, Dict, Any


class OllamaClient:
    """Ollamaクライアント"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "swallow8b-lora-n4000-v09-q4:latest"):
        """
        Args:
            base_url: OllamaサーバーのURL
            model: 使用するモデル名
        """
        self.base_url = base_url
        self.model = model
        print(f"Ollama クライアント初期化: {model}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        テキスト生成
        
        Args:
            prompt: プロンプト
            **kwargs: 追加パラメータ
        
        Returns:
            生成されたテキスト
        """
        url = f"{self.base_url}/api/generate"
        
        # options に入れるべきパラメータを分離
        option_keys = {'temperature', 'num_predict', 'num_gpu', 'top_p', 'top_k', 'seed',
                       'num_ctx', 'repeat_penalty', 'stop'}
        options = {k: v for k, v in kwargs.items() if k in option_keys}
        options['num_gpu'] = options.get('num_gpu', 99)  # GPU強制使用
        extra = {k: v for k, v in kwargs.items() if k not in option_keys}
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
            **extra
        }
        
        try:
            response = requests.post(url, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            print(f"Ollama生成エラー: {e}")
            return ""
    
    def check_connection(self) -> bool:
        """Ollamaサーバーへの接続を確認"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


if __name__ == "__main__":
    # テスト
    client = OllamaClient()
    
    if client.check_connection():
        print("OK Ollama接続成功")
        
        test_prompt = "こんにちは。簡単に自己紹介してください。"
        result = client.generate(test_prompt)
        print(f"\n生成結果: {result[:200]}...")
    else:
        print("NG Ollama接続失敗")
