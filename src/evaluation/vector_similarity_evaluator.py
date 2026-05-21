"""
Vector Similarity Evaluator for VLM Fine-Tuning Assessment
Uses Sentence-BERT Japanese model to compute similarity between ground truth and predictions
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime

# Check if required libraries are installed
try:
    from sentence_transformers import SentenceTransformer
    import torch
    print("✓ sentence-transformers library found")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("\nInstall required package:")
    print("pip install sentence-transformers")
    exit(1)


class VectorSimilarityEvaluator:
    """Evaluate VLM outputs using vector similarity metrics"""
    
    def __init__(
        self,
        model_name: str = "sonoisa/sentence-bert-base-ja-mean-tokens-v2",
        device: str = "cuda",
        batch_size: int = 32
    ):
        """
        Initialize evaluator with Sentence-BERT model
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
            device: 'cuda' or 'cpu'
            batch_size: Batch size for embedding computation
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        
        print(f"🔧 Initializing Vector Similarity Evaluator")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
        
        # Load model
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        print(f"✓ Model loaded successfully")
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute sentence embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        # Filter out empty texts
        valid_texts = [str(t) if t and str(t).strip() else "empty" for t in texts]
        
        # Compute embeddings in batches
        embeddings = self.model.encode(
            valid_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device
        )
        
        return embeddings
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings
        
        Args:
            emb1: First embedding array (n_samples, embedding_dim)
            emb2: Second embedding array (n_samples, embedding_dim)
            
        Returns:
            Array of cosine similarity scores (n_samples,)
        """
        # Normalize embeddings
        emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
        emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
        
        # Compute dot product (cosine similarity)
        similarity = np.sum(emb1_norm * emb2_norm, axis=1)
        
        return similarity
    
    def euclidean_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance between embeddings"""
        return np.linalg.norm(emb1 - emb2, axis=1)
    
    def manhattan_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Compute Manhattan distance between embeddings"""
        return np.sum(np.abs(emb1 - emb2), axis=1)
    
    def evaluate_predictions(
        self,
        ground_truth: List[str],
        predictions: List[str],
        image_ids: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate predictions against ground truth using vector similarity
        
        Args:
            ground_truth: List of ground truth texts
            predictions: List of predicted texts
            image_ids: Optional list of image identifiers
            
        Returns:
            Dictionary with evaluation results
        """
        if len(ground_truth) != len(predictions):
            raise ValueError(f"Mismatched lengths: {len(ground_truth)} != {len(predictions)}")
        
        n_samples = len(ground_truth)
        if image_ids is None:
            image_ids = [f"image_{i:04d}" for i in range(n_samples)]
        
        print(f"\n📊 Evaluating {n_samples} predictions...")
        
        # Compute embeddings
        print("Computing ground truth embeddings...")
        gt_embeddings = self.compute_embeddings(ground_truth)
        
        print("Computing prediction embeddings...")
        pred_embeddings = self.compute_embeddings(predictions)
        
        # Compute metrics
        print("Computing similarity metrics...")
        cosine_sim = self.cosine_similarity(gt_embeddings, pred_embeddings)
        euclidean_dist = self.euclidean_distance(gt_embeddings, pred_embeddings)
        manhattan_dist = self.manhattan_distance(gt_embeddings, pred_embeddings)
        
        # Aggregate statistics
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_name': self.model_name,
                'n_samples': n_samples,
                'device': self.device
            },
            'overall_metrics': {
                'cosine_similarity': {
                    'mean': float(np.mean(cosine_sim)),
                    'std': float(np.std(cosine_sim)),
                    'median': float(np.median(cosine_sim)),
                    'min': float(np.min(cosine_sim)),
                    'max': float(np.max(cosine_sim)),
                    'q25': float(np.percentile(cosine_sim, 25)),
                    'q75': float(np.percentile(cosine_sim, 75))
                },
                'euclidean_distance': {
                    'mean': float(np.mean(euclidean_dist)),
                    'std': float(np.std(euclidean_dist)),
                    'median': float(np.median(euclidean_dist))
                },
                'manhattan_distance': {
                    'mean': float(np.mean(manhattan_dist)),
                    'std': float(np.std(manhattan_dist)),
                    'median': float(np.median(manhattan_dist))
                }
            },
            'per_sample_results': []
        }
        
        # Per-sample results
        for idx, img_id in enumerate(image_ids):
            results['per_sample_results'].append({
                'image_id': img_id,
                'cosine_similarity': float(cosine_sim[idx]),
                'euclidean_distance': float(euclidean_dist[idx]),
                'manhattan_distance': float(manhattan_dist[idx]),
                'ground_truth_length': len(str(ground_truth[idx])),
                'prediction_length': len(str(predictions[idx]))
            })
        
        # Quality categorization
        thresholds = {
            'excellent': 0.85,
            'good': 0.75,
            'acceptable': 0.65,
            'poor': 0.50
        }
        
        categorization = {
            'excellent': np.sum(cosine_sim >= thresholds['excellent']),
            'good': np.sum((cosine_sim >= thresholds['good']) & (cosine_sim < thresholds['excellent'])),
            'acceptable': np.sum((cosine_sim >= thresholds['acceptable']) & (cosine_sim < thresholds['good'])),
            'poor': np.sum((cosine_sim >= thresholds['poor']) & (cosine_sim < thresholds['acceptable'])),
            'very_poor': np.sum(cosine_sim < thresholds['poor'])
        }
        
        results['quality_distribution'] = {
            k: {'count': int(v), 'percentage': float(v / n_samples * 100)}
            for k, v in categorization.items()
        }
        
        # Print summary
        print(f"\n✅ Evaluation Complete!")
        print(f"\n📈 Overall Metrics:")
        print(f"   Cosine Similarity: {results['overall_metrics']['cosine_similarity']['mean']:.4f} ± {results['overall_metrics']['cosine_similarity']['std']:.4f}")
        print(f"   Median: {results['overall_metrics']['cosine_similarity']['median']:.4f}")
        print(f"   Range: [{results['overall_metrics']['cosine_similarity']['min']:.4f}, {results['overall_metrics']['cosine_similarity']['max']:.4f}]")
        
        print(f"\n📊 Quality Distribution:")
        for category, stats in results['quality_distribution'].items():
            print(f"   {category.capitalize()}: {stats['count']} ({stats['percentage']:.1f}%)")
        
        return results
    
    def evaluate_from_csv(
        self,
        csv_path: Path,
        ground_truth_col: str = '所見',
        prediction_col: str = 'prediction',
        image_id_col: str = 'filename'
    ) -> Dict:
        """
        Evaluate predictions from a CSV file
        
        Args:
            csv_path: Path to CSV file with ground truth and predictions
            ground_truth_col: Column name for ground truth text
            prediction_col: Column name for prediction text
            image_id_col: Column name for image identifiers
            
        Returns:
            Evaluation results dictionary
        """
        print(f"📁 Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} rows")
        
        # Extract columns
        ground_truth = df[ground_truth_col].tolist()
        predictions = df[prediction_col].tolist()
        image_ids = df[image_id_col].tolist() if image_id_col in df.columns else None
        
        # Evaluate
        results = self.evaluate_predictions(ground_truth, predictions, image_ids)
        
        return results
    
    def save_results(self, results: Dict, output_path: Path):
        """Save evaluation results to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
    
    def create_comparison_report(
        self,
        results_list: List[Tuple[str, Dict]],
        output_path: Path
    ):
        """
        Create a markdown comparison report for multiple evaluation results
        
        Args:
            results_list: List of (stage_name, results_dict) tuples
            output_path: Path to save markdown report
        """
        report = f"# Vector Similarity Evaluation - Progressive Training Comparison\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"## Overview\n\n"
        report += f"Comparison of VLM fine-tuning across progressive dataset sizes.\n\n"
        
        # Overall metrics table
        report += "## Cosine Similarity Scores\n\n"
        report += "| Stage | Mean | Std | Median | Min | Max | Q25 | Q75 |\n"
        report += "|-------|------|-----|--------|-----|-----|-----|-----|\n"
        
        for stage_name, results in results_list:
            cs = results['overall_metrics']['cosine_similarity']
            report += f"| **{stage_name}** | {cs['mean']:.4f} | {cs['std']:.4f} | {cs['median']:.4f} | {cs['min']:.4f} | {cs['max']:.4f} | {cs['q25']:.4f} | {cs['q75']:.4f} |\n"
        
        # Quality distribution
        report += "\n## Quality Distribution\n\n"
        report += "| Stage | Excellent (≥0.85) | Good (≥0.75) | Acceptable (≥0.65) | Poor (≥0.50) | Very Poor (<0.50) |\n"
        report += "|-------|-------------------|--------------|--------------------|--------------|-----------------|\n"
        
        for stage_name, results in results_list:
            qd = results['quality_distribution']
            report += f"| **{stage_name}** "
            for category in ['excellent', 'good', 'acceptable', 'poor', 'very_poor']:
                count = qd[category]['count']
                pct = qd[category]['percentage']
                report += f"| {count} ({pct:.1f}%) "
            report += "|\n"
        
        # Performance improvement
        report += "\n## Performance Improvement\n\n"
        if len(results_list) > 1:
            baseline_mean = results_list[0][1]['overall_metrics']['cosine_similarity']['mean']
            report += f"Baseline ({}): {baseline_mean:.4f}\n\n".format(results_list[0][0])
            
            for stage_name, results in results_list[1:]:
                current_mean = results['overall_metrics']['cosine_similarity']['mean']
                improvement = current_mean - baseline_mean
                improvement_pct = (improvement / baseline_mean) * 100
                report += f"- **{stage_name}**: {current_mean:.4f} ({improvement:+.4f}, {improvement_pct:+.2f}%)\n"
        
        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Comparison report saved to: {output_path}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate VLM predictions using vector similarity")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with ground truth and predictions")
    parser.add_argument("--gt-col", type=str, default="所見", help="Ground truth column name")
    parser.add_argument("--pred-col", type=str, default="prediction", help="Prediction column name")
    parser.add_argument("--id-col", type=str, default="filename", help="Image ID column name")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path for results")
    parser.add_argument("--model", type=str, default="sonoisa/sentence-bert-base-ja-mean-tokens-v2", 
                       help="Sentence-BERT model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embeddings")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = VectorSimilarityEvaluator(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Evaluate
    results = evaluator.evaluate_from_csv(
        csv_path=Path(args.csv),
        ground_truth_col=args.gt_col,
        prediction_col=args.pred_col,
        image_id_col=args.id_col
    )
    
    # Save results
    evaluator.save_results(results, Path(args.output))


if __name__ == "__main__":
    main()
