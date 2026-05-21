# 橋梁損傷 画像読解・補修優先度スコアリング MVP v0.1

**Vision-Language Model (LLaVA) を使用した橋梁損傷画像の自動分析・補修優先度評価システム**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 目次

- [概要](#概要)
- [v0.1 達成内容](#v01-達成内容)
- [システム構成](#システム構成)
- [実測パフォーマンス](#実測パフォーマンス)
- [セットアップ](#セットアップ)
- [使用方法](#使用方法)
- [モデル比較](#モデル比較)
- [技術スタック](#技術スタック)
- [ディレクトリ構造](#ディレクトリ構造)
- [トラブルシューティング](#トラブルシューティング)
- [次のステップ](#次のステップ)

---

## 概要

橋梁点検における鉄筋露出・ひび割れなどの損傷画像を自動分析し、補修優先度をスコアリングするエンドツーエンドパイプライン。**LLaVA (Large Language and Vision Assistant)** を使用して画像から専門的な損傷説明を生成し、構造化データとして補修計画に活用できます。

### アーキテクチャ

```
画像入力 (254枚)
    ↓
前処理 (denoise/resize/contrast)
    ↓
Vision分析 (LLaVA-1.5-7B GGUF)
    ↓
JSON構造化 (Swallow-8B)
    ↓
スコアリング (ルールベース)
    ↓
優先度出力 (1-5段階)
```

---

## v0.1 達成内容

### ✅ 完了項目

- **3つのVisionモード実装**
  - **llama-cpp-python + GGUF** (推奨): 軽量・高速・GPU完全活用
  - HuggingFace Transformers: 安定・高精度
  - Ollama統合: 簡易セットアップ（※CPU動作で低速）

- **完全なパイプライン構築**
  - 前処理モジュール (OpenCV)
  - Vision分析 (LLaVA-1.5-7B)
  - JSON構造化 (Swallow-8B via Ollama)
  - 優先度スコアリング (ルールベース)

- **実証テスト完了**
  - ✅ 1枚テスト: 42秒/枚
  - ✅ 10枚バッチ: 平均51.6秒/枚、成功率100%
  - 優先度分布: 即時補修(5) 6枚、計画的補修(3) 4枚

- **文字化け問題解決**
  - Windows PowerShell cp932対応
  - llama.cpp C++ログ抑制
  - UTF-8エンコーディング統一

### 📊 検証データ

- **データセット**: 254枚の鉄筋露出画像
- **GPU**: NVIDIA GeForce RTX 4060 Ti (16GB VRAM)
- **OS**: Windows 11
- **処理環境**: Python 3.12.10 + CUDA 12.4

---

## システム構成

### パイプラインモジュール

1. **前処理** (`src/preprocessing/`)
   - ノイズ除去 (Non-local Means Denoising)
   - リサイズ (最大1024x1024)
   - コントラスト調整 (CLAHE)

2. **Vision分析** (`src/vision/`)
   - **llama_cpp_vision.py** (推奨)
     - LLaVA-1.5-7B Q4_K_M GGUF (4.08GB)
     - GPU完全活用 (全レイヤー配置)
     - Ollama依存なし
   - granite_vision.py
     - HuggingFace llava-1.5-7b-hf (14GB)
   - ollama_vision.py
     - Ollama llava:7b (CPU動作)

3. **JSON構造化** (`src/structuring/`)
   - Swallow-8B (Ollama: swallow8b-lora-n4000-v09-q4)
   - 日本語特化LLM
   - 損傷タイプ/重症度/位置/リスクを構造化

4. **スコアリング** (`src/scoring/`)
   - ルールベーススコアリング
   - 1-5段階の優先度評価
   - 重み付け: 重症度(40%), 損傷タイプ(35%), 位置(15%), リスク(10%)

---

## 実測パフォーマンス

### v0.1 テスト結果

| テスト規模 | 処理時間 | 成功率 | 平均時間/枚 |
|-----------|----------|--------|-------------|
| 1枚テスト | 42秒 | 100% | 42秒 |
| 10枚バッチ | 8分35秒 | 100% | 51.6秒 |
| 50枚推定 | 約43分 | - | 約52秒 |
| 254枚推定 | 約3.6時間 | - | 約51秒 |

### 優先度分布 (10枚テスト)

- **優先度5** (即時補修が必要): 6枚 (60%)
- **優先度3** (計画的補修): 4枚 (40%)

### 処理速度の内訳

```
画像前処理:     ~2秒
Vision分析:     ~42秒 (画像エンコード 1.4秒 + 推論 40秒)
JSON構造化:     ~5秒
スコアリング:   <1秒
---------------------------------
合計:          ~51秒/枚
```

---

## セットアップ

### 1. 環境要件

- **OS**: Windows 10/11 (Linux/macOS対応可)
- **GPU**: NVIDIA GPU 8GB+ VRAM推奨 (16GB推奨)
- **Python**: 3.10以上
- **CUDA**: 12.1以上
- **ストレージ**: 20GB以上の空き容量

### 2. リポジトリクローン

```bash
git clone <repository-url>
cd damage_text_score
```

### 3. 仮想環境作成

```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

### 4. 依存パッケージインストール

```bash
# PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# llama-cpp-python (GPU版)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# その他の依存関係
pip install -r requirements.txt
```

### 5. モデルダウンロード

```bash
# LLaVA GGUF モデル (推奨)
python download_llava_gguf.py
# ↓ ダウンロードファイル:
#   - models/ggml-model-q4_k.gguf (4.08GB)
#   - models/mmproj-model-f16.gguf (624MB)
```

### 6. Ollama セットアップ (JSON構造化用)

```bash
# Ollamaインストール
# https://ollama.com/download

# Swallow-8Bモデル取得
ollama pull swallow8b-lora-n4000-v09-q4:latest
```

---

## 使用方法

### クイックスタート

```bash
# 1枚テスト (約42秒)
python quickstart.py --mode 1

# 10枚バッチ (約8.5分)
python quickstart.py --mode 2

# 50枚処理 (約43分)
python quickstart.py --mode 3

# 全254枚処理 (約3.6時間)
python quickstart.py --mode 4
```

### 出力ファイル

```
data/outputs/
├── quickstart_single.csv        # 1枚結果
├── quickstart_10images.csv      # 10枚結果
├── quickstart_50images.csv      # 50枚結果
└── quickstart_254images.csv     # 全画像結果
```

### 出力フォーマット

**CSV出力例:**

```csv
image_name,damage_type,severity,location,risk,priority_score,priority_level,description
kensg-rebarexposureRb_001.png,crack,high,girder,structural,0.952,5,ひび割れが広範囲に...
```

**JSON構造:**

```json
{
  "damage_type": "rebar_exposure",
  "severity": "high",
  "location": "girder",
  "risk": "structural",
  "description_ja": "鉄筋露出が見られ、腐食が進行している...",
  "key_features": ["鉄筋露出", "中程度の腐食"],
  "priority_score": 0.952,
  "priority_level": 5
}
```

### カスタム実行

```python
from src.pipeline.end_to_end import DamageAnalysisPipeline

# パイプライン初期化
pipeline = DamageAnalysisPipeline("config.yaml")

# 1枚処理
result = pipeline.process_image("path/to/image.png")

# バッチ処理
results = pipeline.process_batch(image_paths, output_csv="results.csv")
```

---

## モデル比較

### Vision モデル性能比較

| モード | モデル | サイズ | 処理時間/枚 | GPU使用率 | 推奨度 |
|--------|--------|--------|-------------|-----------|--------|
| **llama-cpp-python** | LLaVA-1.5-7B Q4_K_M | 4.08GB | **51.6秒** | 100% | ⭐⭐⭐⭐⭐ |
| HuggingFace | llava-1.5-7b-hf | 14GB | 45秒 | 100% | ⭐⭐⭐⭐ |
| Ollama | llava:7b | 4.7GB | 88秒 | 0% (CPU) | ⭐⭐ |

### 選択基準

- **llama-cpp-python** (推奨)
  - ✅ 軽量 (4GB)
  - ✅ GPU完全活用
  - ✅ Ollama不要
  - ✅ 安定動作
  - ⚠️ 量子化による若干の精度低下

- **HuggingFace**
  - ✅ 最高精度
  - ✅ GPU完全活用
  - ⚠️ 大容量 (14GB)
  - ⚠️ VRAM要求高い

- **Ollama**
  - ⚠️ CPU動作で遅い
  - ⚠️ GPU活用できず
  - ✅ セットアップ簡単

---

## 技術スタック

### フレームワーク

- **PyTorch 2.6.0** - 深層学習フレームワーク
- **Transformers 4.57.6** - HuggingFaceモデル
- **llama-cpp-python 0.3.16** - GGUF推論エンジン
- **OpenCV 4.12.0** - 画像処理

### モデル

- **LLaVA-1.5-7B** - Vision-Language Model
  - 論文: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
  - GGUF量子化版 (Q4_K_M)
  
- **Swallow-8B** - 日本語LLM
  - 開発: TokyoTech LLM Project
  - JSON構造化特化

### ライブラリ

- pandas 2.2.3 - データ処理
- pyyaml 6.0.2 - 設定管理
- tqdm 4.67.1 - プログレスバー
- pillow 11.1.0 - 画像処理

---

## ディレクトリ構造

```
damage_text_score/
├── .venv/                          # Python仮想環境
├── data/                           # データセット
│   ├── images_human_inspect_n254/  # 入力画像 (254枚)
│   ├── preprocessed/               # 前処理済み画像
│   └── outputs/                    # 処理結果
│       ├── descriptions/           # Vision出力
│       ├── structured/             # JSON構造化出力
│       └── scores/                 # スコアリング結果
├── models/                         # モデルファイル
│   ├── ggml-model-q4_k.gguf        # LLaVA GGUF (4.08GB)
│   ├── mmproj-model-f16.gguf       # MMProj (624MB)
│   └── scoring_rules.yaml          # スコアリングルール
├── src/                            # ソースコード
│   ├── preprocessing/              # 前処理モジュール
│   │   └── image_preprocessor.py
│   ├── vision/                     # Vision分析
│   │   ├── llama_cpp_vision.py     # llama-cpp-python版 (推奨)
│   │   ├── granite_vision.py       # HuggingFace版
│   │   └── ollama_vision.py        # Ollama版
│   ├── structuring/                # JSON構造化
│   │   └── json_structurer.py
│   ├── scoring/                    # スコアリング
│   │   └── priority_scorer.py
│   ├── pipeline/                   # パイプライン
│   │   └── end_to_end.py
│   └── utils/                      # ユーティリティ
│       ├── config.py
│       └── ollama_client.py
├── config.yaml                     # システム設定
├── quickstart.py                   # クイックスタート
├── download_llava_gguf.py          # モデルダウンロード
├── requirements.txt                # 依存パッケージ
└── README.md                       # このファイル
```

---

## トラブルシューティング

### 文字化け問題

**症状**: PowerShellで日本語が文字化けする

**解決策**:
```powershell
# UTF-8に変更
chcp 65001
python quickstart.py
```

### CUDA Out of Memory

**症状**: `CUDA out of memory` エラー

**解決策**:
```yaml
# config.yaml
llama_cpp_vision:
  n_gpu_layers: 20  # -1から調整 (全レイヤーではなく一部のみGPU)
```

### Ollama接続エラー

**症状**: `Failed to connect to Ollama`

**解決策**:
```bash
# Ollamaサーバー起動確認
ollama list

# サーバー再起動
ollama serve
```

### llama-cpp-python インストールエラー

**症状**: `Failed building wheel for llama-cpp-python`

**解決策**:
```bash
# CUDA版を明示的にインストール
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# または環境変数でCUDAを有効化
$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

---

## 次のステップ

### v0.2 予定 (2026年Q2)

- [ ] 50枚テスト実行・検証
- [ ] 全254枚処理完了
- [ ] 精度評価 (人間アノテーションと比較)
- [ ] 推論速度最適化 (バッチ処理改善)

### v1.0 目標

- [ ] Web UI実装 (Streamlit/Gradio)
- [ ] リアルタイム処理対応
- [ ] 複数損傷タイプ対応拡張
- [ ] GAMモデルによるスコア補正
- [ ] Docker環境構築
- [ ] API サーバー実装

### 研究改善項目

- [ ] より軽量なVision モデル検討 (LLaVA-1.6, MobileVLM)
- [ ] Few-shot学習による精度向上
- [ ] マルチモーダル学習 (画像+メタデータ)
- [ ] アクティブラーニング導入

---

## ライセンス

MIT License

---

## 参考文献

1. Liu et al. (2023). "Visual Instruction Tuning" - LLaVA
2. TokyoTech LLM Project - Swallow Models
3. Georgi Gerganov - llama.cpp

---

## 連絡先

プロジェクト管理者: [Your Name]
Email: [your.email@example.com]

**Last Updated**: 2026年3月20日 (v0.1)
