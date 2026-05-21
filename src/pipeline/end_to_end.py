"""
エンドツーエンドパイプライン
画像入力 → 前処理 → Vision → 構造化 → スコアリング → 出力
"""
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.image_preprocessor import ImagePreprocessor, PreprocessConfig
from vision.ollama_vision import OllamaVisionAnalyzer
from vision.llama_cpp_vision import LlamaCppVisionAnalyzer, LlamaCppVisionConfig
from structuring.json_structurer import JSONStructurer, StructuringConfig, DamageStructure
from scoring.priority_scorer import PriorityScorer, ScoringConfig, PriorityScore
from utils.config import Config, ensure_dir

# HuggingFaceモード用（オプショナル）
try:
    from vision.granite_vision import GraniteVisionAnalyzer, VisionConfig
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("⚠️  HuggingFace Transformersモードは利用できません（llama_cppまたはollamaを使用してください）")


@dataclass
class PipelineResult:
    """パイプライン実行結果"""
    image_path: str
    image_name: str
    description: str
    structure: Optional[Dict]
    score: Optional[Dict]
    status: str
    error: Optional[str] = None
    processing_time: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


class DamageAnalysisPipeline:
    """損傷分析パイプライン"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: 設定ファイルパス
        """
        print("=" * 60)
        print("損傷読解・補修優先度スコアリングパイプライン")
        print("=" * 60)
        
        # 設定読み込み
        self.config = Config(config_path)
        print(f"\n設定ファイル: {config_path}")
        
        # 各モジュール初期化
        print("\n--- モジュール初期化 ---")
        self._init_modules()
    
    def _init_modules(self):
        """各モジュールを初期化"""
        # 前処理
        print("\n1. 前処理モジュール")
        preprocess_config = PreprocessConfig(
            max_width=self.config.get('preprocessing.resize.max_width', 1024),
            max_height=self.config.get('preprocessing.resize.max_height', 1024),
            denoise_enabled=self.config.get('preprocessing.denoise.enabled', True),
            denoise_strength=self.config.get('preprocessing.denoise.strength', 5),
            contrast_enabled=self.config.get('preprocessing.contrast.enabled', True),
            clip_limit=self.config.get('preprocessing.contrast.clip_limit', 2.0),
            tile_grid_size=tuple(self.config.get('preprocessing.contrast.tile_grid_size', [8, 8]))
        )
        self.preprocessor = ImagePreprocessor(preprocess_config)
        
        # Vision
        print("\n2. Vision分析モジュール")
        vision_mode = self.config.get('vision_mode', 'llama_cpp')
        print(f"   モード: {vision_mode}")
        
        if vision_mode == 'llama_cpp':
            # llama-cpp-python GGUF版（推奨: 高速・軽量・GPU加速）
            llama_cpp_config = LlamaCppVisionConfig(
                model_path=self.config.get('llama_cpp_vision.model_path', 'models/ggml-model-q4_k.gguf'),
                mmproj_path=self.config.get('llama_cpp_vision.mmproj_path', 'models/mmproj-model-f16.gguf'),
                n_gpu_layers=self.config.get('llama_cpp_vision.n_gpu_layers', -1),
                n_ctx=self.config.get('llama_cpp_vision.n_ctx', 4096),
                max_tokens=self.config.get('llama_cpp_vision.max_tokens', 300),
                temperature=self.config.get('llama_cpp_vision.temperature', 0.3),
                top_p=self.config.get('llama_cpp_vision.top_p', 0.9)
            )
            self.vision_analyzer = LlamaCppVisionAnalyzer(llama_cpp_config)
        
        elif vision_mode == 'ollama':
            # Ollama版LLaVA使用
            ollama_model = self.config.get('granite_vision.ollama_model', 'llava:7b')
            ollama_url = self.config.get('granite_vision.ollama_url', 'http://localhost:11434')
            temperature = self.config.get('granite_vision.temperature', 0.3)
            self.vision_analyzer = OllamaVisionAnalyzer(
                base_url=ollama_url,
                model=ollama_model,
                temperature=temperature
            )
        
        else:  # 'huggingface'
            # HuggingFace Transformers版使用
            if not HUGGINGFACE_AVAILABLE:
                raise RuntimeError(
                    "HuggingFaceモードが選択されていますが、必要なモジュールがインストールされていません。\n"
                    "llama_cppまたはollamaモードを使用してください。"
                )
            vision_config = VisionConfig(
                model_name=self.config.get('granite_vision.model_name', 'llava-hf/llava-1.5-7b-hf'),
                device=self.config.get('granite_vision.device', 'cuda'),
                max_new_tokens=self.config.get('granite_vision.max_new_tokens', 300),
                temperature=self.config.get('granite_vision.temperature', 0.3)
            )
            self.vision_analyzer = GraniteVisionAnalyzer(vision_config)
        
        # 構造化
        print("\n3. JSON構造化モジュール")
        structuring_config = StructuringConfig(
            model_name=self.config.get('structuring.model_name', 'tokyotech-llm/Swallow-7b-instruct-v0.1'),
            device=self.config.get('structuring.device', 'cuda'),
            max_new_tokens=self.config.get('structuring.max_new_tokens', 500),
            temperature=self.config.get('structuring.temperature', 0.1),
            use_ollama=self.config.get('structuring.use_ollama', False),
            ollama_model=self.config.get('structuring.ollama_model', 'swallow8b-lora-n4000-v09-q4:latest'),
            ollama_url=self.config.get('structuring.ollama_url', 'http://localhost:11434')
        )
        self.structurer = JSONStructurer(structuring_config)
        
        # スコアリング
        print("\n4. スコアリングモジュール")
        scoring_config = ScoringConfig(
            rules_file=self.config.get('scoring.rules_file', 'models/scoring_rules.yaml'),
            weight_damage_type=self.config.get('scoring.weights.damage_type', 0.35),
            weight_severity=self.config.get('scoring.weights.severity', 0.40),
            weight_location=self.config.get('scoring.weights.location', 0.15),
            weight_risk=self.config.get('scoring.weights.risk', 0.10),
            gam_enabled=self.config.get('scoring.gam.enabled', False),
            gam_model_path=self.config.get('scoring.gam.model_path')
        )
        self.scorer = PriorityScorer(scoring_config)
        
        print("\nOK すべてのモジュールの初期化が完了しました")
    
    def process_image(
        self,
        image_path: Path,
        preprocess: bool = True,
        save_intermediate: bool = True
    ) -> PipelineResult:
        """
        1枚の画像を処理
        
        Args:
            image_path: 画像パス
            preprocess: 前処理を実行するか
            save_intermediate: 中間結果を保存するか
        
        Returns:
            処理結果
        """
        start_time = datetime.now()
        image_path = Path(image_path)
        
        try:
            # 1. 前処理
            if preprocess:
                preprocessed_dir = Path(self.config.get('data.preprocessed', 'data/preprocessed'))
                ensure_dir(preprocessed_dir)
                preprocessed_path = preprocessed_dir / image_path.name
                self.preprocessor.preprocess_file(image_path, preprocessed_path)
                analysis_image = preprocessed_path
            else:
                analysis_image = image_path
            
            # 2. Vision分析
            description = self.vision_analyzer.analyze_image(analysis_image)
            
            if save_intermediate:
                desc_dir = Path(self.config.get('data.outputs.descriptions', 'data/outputs/descriptions'))
                ensure_dir(desc_dir)
                desc_path = desc_dir / f"{image_path.stem}_description.txt"
                with open(desc_path, 'w', encoding='utf-8') as f:
                    f.write(description)
            
            # 3. JSON構造化
            structure = self.structurer.structure_text(description)
            
            if save_intermediate:
                struct_dir = Path(self.config.get('data.outputs.structured', 'data/outputs/structured'))
                ensure_dir(struct_dir)
                struct_path = struct_dir / f"{image_path.stem}_structured.json"
                with open(struct_path, 'w', encoding='utf-8') as f:
                    json.dump(structure.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 4. スコアリング
            score = self.scorer.calculate_score(
                damage_type=structure.damage_type,
                severity=structure.severity,
                location=structure.location,
                risk=structure.risk
            )
            
            if save_intermediate:
                score_dir = Path(self.config.get('data.outputs.scores', 'data/outputs/scores'))
                ensure_dir(score_dir)
                score_path = score_dir / f"{image_path.stem}_score.json"
                with open(score_path, 'w', encoding='utf-8') as f:
                    json.dump(score.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 処理時間
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return PipelineResult(
                image_path=str(image_path),
                image_name=image_path.name,
                description=description,
                structure=structure.to_dict(),
                score=score.to_dict(),
                status="success",
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return PipelineResult(
                image_path=str(image_path),
                image_name=image_path.name,
                description="",
                structure=None,
                score=None,
                status="error",
                error=str(e),
                processing_time=processing_time
            )
    
    def process_batch(
        self,
        input_dir: Path,
        pattern: str = "*.png",
        limit: Optional[int] = None
    ) -> List[PipelineResult]:
        """
        ディレクトリ内の画像を一括処理
        
        Args:
            input_dir: 入力ディレクトリ
            pattern: ファイルパターン
            limit: 処理する画像数の上限（Noneの場合は全て）
        
        Returns:
            処理結果のリスト
        """
        input_dir = Path(input_dir)
        image_files = sorted(input_dir.glob(pattern))
        
        if limit:
            image_files = image_files[:limit]
        
        print(f"\n処理対象: {len(image_files)}枚の画像")
        print("=" * 60)
        
        results = []
        
        for img_path in tqdm(image_files, desc="画像処理中"):
            result = self.process_image(
                img_path,
                preprocess=True,
                save_intermediate=True
            )
            results.append(result)
        
        return results
    
    def save_results(
        self,
        results: List[PipelineResult],
        output_dir: Path,
        filename: str = "results.csv"
    ):
        """
        処理結果を保存
        
        Args:
            results: 処理結果リスト
            output_dir: 出力ディレクトリ
            filename: ファイル名
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        # CSV形式で保存
        rows = []
        for r in results:
            if r.status == "success":
                rows.append({
                    'image_name': r.image_name,
                    'damage_type': r.structure['damage_type'],
                    'severity': r.structure['severity'],
                    'location': r.structure['location'],
                    'risk': r.structure['risk'],
                    'priority_score': r.score['raw_score'],
                    'priority_level': r.score['priority_level'],
                    'priority_description': r.score['priority_description'],
                    'description': r.description[:200] + '...' if len(r.description) > 200 else r.description,
                    'processing_time': r.processing_time,
                    'status': r.status
                })
            else:
                rows.append({
                    'image_name': r.image_name,
                    'damage_type': None,
                    'severity': None,
                    'location': None,
                    'risk': None,
                    'priority_score': None,
                    'priority_level': None,
                    'priority_description': None,
                    'description': None,
                    'processing_time': r.processing_time,
                    'status': r.status
                })
        
        df = pd.DataFrame(rows)
        csv_path = output_dir / filename
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\nCSV結果保存: {csv_path}")
        
        # JSON形式でも保存
        json_path = output_dir / filename.replace('.csv', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
        print(f"JSON結果保存: {json_path}")
        
        # サマリー統計
        success_count = len([r for r in results if r.status == "success"])
        error_count = len([r for r in results if r.status == "error"])
        
        print(f"\n=== 処理サマリー ===")
        print(f"総数: {len(results)}")
        print(f"成功: {success_count}")
        print(f"エラー: {error_count}")
        
        if success_count > 0:
            avg_time = sum(r.processing_time for r in results if r.status == "success") / success_count
            print(f"平均処理時間: {avg_time:.2f}秒/枚")
            
            # 優先度分布
            priority_counts = {}
            for r in results:
                if r.status == "success":
                    level = r.score['priority_level']
                    priority_counts[level] = priority_counts.get(level, 0) + 1
            
            print(f"\n優先度分布:")
            for level in sorted(priority_counts.keys(), reverse=True):
                print(f"  優先度{level}: {priority_counts[level]}枚")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="損傷画像の読解と補修優先度スコアリングパイプライン"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='入力ディレクトリまたは画像ファイル'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/outputs',
        help='出力ディレクトリ（デフォルト: data/outputs）'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='設定ファイルパス（デフォルト: config.yaml）'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='処理する画像数の上限'
    )
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*.png',
        help='ファイルパターン（デフォルト: *.png）'
    )
    
    args = parser.parse_args()
    
    # パイプライン初期化
    pipeline = DamageAnalysisPipeline(args.config)
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    # 処理実行
    if input_path.is_file():
        # 単一ファイル
        result = pipeline.process_image(input_path)
        results = [result]
        print(f"\n処理完了: {result.image_name}")
        print(f"優先度: {result.score['priority_level']} - {result.score['priority_description']}")
    
    elif input_path.is_dir():
        # ディレクトリ一括処理
        results = pipeline.process_batch(
            input_path,
            pattern=args.pattern,
            limit=args.limit
        )
    
    else:
        print(f"エラー: 入力パスが見つかりません: {input_path}")
        return
    
    # 結果保存
    pipeline.save_results(results, output_dir)
    
    print("\n" + "=" * 60)
    print("パイプライン実行完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
