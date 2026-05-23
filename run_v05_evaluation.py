"""
v0.5 Complete Workflow: Run inference and evaluation for all models
Automates the entire evaluation pipeline for 1k/2k/3k/4k models
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd


class ProgressiveEvaluationPipeline:
    """Complete evaluation pipeline for progressive training models"""
    
    def __init__(
        self,
        base_dir: Path = Path("."),
        test_csv: str = "data/v03_fine_tuning/test_set_n800.csv",
        image_dir: str = "data/image_text_inspect_n10789/rank_c_images_n10789",
        output_base_dir: str = "data/v03_fine_tuning/evaluations"
    ):
        """
        Initialize pipeline
        
        Args:
            base_dir: Base directory of project
            test_csv: Path to test set CSV
            image_dir: Base directory for images
            output_base_dir: Base directory for evaluation outputs
        """
        self.base_dir = Path(base_dir)
        self.test_csv = self.base_dir / test_csv
        self.image_dir = self.base_dir / image_dir
        self.output_base_dir = self.base_dir / output_base_dir
        
        # Model configurations
        self.models = {
            '1k': {
                'model_dir': 'models/llava_v03_qlora_1k',
                'name': 'v0.4.1 (1k samples)',
                'short_name': '1k'
            },
            '2k': {
                'model_dir': 'models/llava_v03_qlora_2k',
                'name': 'v0.4.2 (2k samples)',
                'short_name': '2k'
            },
            '3k': {
                'model_dir': 'models/llava_v03_qlora_3k',
                'name': 'v0.4.3 (3k samples)',
                'short_name': '3k'
            },
            '4k': {
                'model_dir': 'models/llava_v03_qlora_4k',
                'name': 'v0.4.4 (4k samples)',
                'short_name': '4k'
            }
        }
        
        # Validate paths
        self._validate_paths()
        
        # Create output directories
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        print("✓ Progressive Evaluation Pipeline Initialized")
        print(f"  Test set: {self.test_csv} ({len(pd.read_csv(self.test_csv))} samples)")
        print(f"  Output dir: {self.output_base_dir}")
    
    def _validate_paths(self):
        """Validate required paths exist"""
        if not self.test_csv.exists():
            raise FileNotFoundError(f"Test CSV not found: {self.test_csv}")
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Check if models exist
        missing_models = []
        for stage, config in self.models.items():
            model_path = self.base_dir / config['model_dir']
            if not model_path.exists():
                missing_models.append(stage)
        
        if missing_models:
            print(f"⚠️  Warning: Models not found for stages: {', '.join(missing_models)}")
            print(f"   These stages will be skipped.")
            # Remove missing models
            for stage in missing_models:
                del self.models[stage]
    
    def run_inference(self, stage: str, limit: int = None) -> Path:
        """
        Run inference for a specific model stage
        
        Args:
            stage: Model stage ('1k', '2k', '3k', '4k')
            limit: Optional limit on number of images (for testing)
            
        Returns:
            Path to output CSV file
        """
        config = self.models[stage]
        model_dir = self.base_dir / config['model_dir']
        output_csv = self.output_base_dir / f"inference_results_{stage}.csv"
        
        print(f"\n{'='*60}")
        print(f"🚀 Running Inference: {config['name']}")
        print(f"{'='*60}")
        print(f"Model: {model_dir}")
        print(f"Output: {output_csv}")
        
        # Build command
        cmd = [
            sys.executable,
            "inference_v04_qlora.py",
            "--model-dir", str(model_dir),
            "--test-csv", str(self.test_csv),
            "--output-csv", str(output_csv),
            "--image-dir", str(self.image_dir)
        ]
        
        if limit:
            cmd.extend(["--limit", str(limit)])
            print(f"⚠️  Limited to {limit} images for testing")
        
        # Run inference
        start_time = datetime.now()
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                cwd=str(self.base_dir)
            )
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"\n✓ Inference completed in {duration}")
            return output_csv
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Inference failed for {stage}: {e}")
            raise
    
    def run_evaluation(self, stage: str, inference_csv: Path) -> Path:
        """
        Run vector similarity evaluation for inference results
        
        Args:
            stage: Model stage ('1k', '2k', '3k', '4k')
            inference_csv: Path to inference results CSV
            
        Returns:
            Path to evaluation JSON file
        """
        config = self.models[stage]
        output_json = self.output_base_dir / f"evaluation_{stage}.json"
        
        print(f"\n{'='*60}")
        print(f"📊 Running Evaluation: {config['name']}")
        print(f"{'='*60}")
        print(f"Input: {inference_csv}")
        print(f"Output: {output_json}")
        
        # Build command
        cmd = [
            sys.executable,
            "-m", "src.evaluation.vector_similarity_evaluator",
            "--csv", str(inference_csv),
            "--gt-col", "ground_truth",
            "--pred-col", "prediction",
            "--id-col", "image_id",
            "--output", str(output_json)
        ]
        
        # Run evaluation
        start_time = datetime.now()
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                cwd=str(self.base_dir)
            )
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"\n✓ Evaluation completed in {duration}")
            return output_json
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Evaluation failed for {stage}: {e}")
            raise
    
    def create_comparison_report(self, evaluation_results: Dict[str, Path]) -> Path:
        """
        Create comprehensive comparison report
        
        Args:
            evaluation_results: Dict mapping stage to evaluation JSON path
            
        Returns:
            Path to comparison report
        """
        print(f"\n{'='*60}")
        print(f"📄 Creating Comparison Report")
        print(f"{'='*60}")
        
        # Load all evaluation results
        results_list = []
        for stage in ['1k', '2k', '3k', '4k']:
            if stage in evaluation_results:
                json_path = evaluation_results[stage]
                with open(json_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                results_list.append((self.models[stage]['name'], results))
        
        # Create markdown report
        report_path = self.output_base_dir / "Progressive_Evaluation_Report.md"
        
        report = self._generate_markdown_report(results_list)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ Comparison report saved: {report_path}")
        
        return report_path
    
    def _generate_markdown_report(self, results_list: List[Tuple[str, Dict]]) -> str:
        """Generate comprehensive markdown report"""
        
        report = f"# Progressive Training Evaluation Report (v0.5)\n\n"
        report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"---\n\n"
        
        # Executive Summary
        report += "## 📊 Executive Summary\n\n"
        report += "Evaluation of LLaVA-1.5-7B QLoRA fine-tuning across progressive dataset sizes (1k, 2k, 3k, 4k samples) using vector similarity metrics on a fixed test set (n=800).\n\n"
        
        # Overall Metrics Table
        report += "## 🎯 Cosine Similarity Scores\n\n"
        report += "| Stage | Mean ± Std | Median | Min | Max | Q25 | Q75 |\n"
        report += "|-------|-----------|--------|-----|-----|-----|-----|\n"
        
        for stage_name, results in results_list:
            cs = results['overall_metrics']['cosine_similarity']
            report += f"| **{stage_name}** | {cs['mean']:.4f} ± {cs['std']:.4f} | {cs['median']:.4f} | {cs['min']:.4f} | {cs['max']:.4f} | {cs['q25']:.4f} | {cs['q75']:.4f} |\n"
        
        # Quality Distribution
        report += "\n## 📈 Quality Distribution\n\n"
        report += "| Stage | Excellent<br/>(≥0.85) | Good<br/>(≥0.75) | Acceptable<br/>(≥0.65) | Poor<br/>(≥0.50) | Very Poor<br/>(<0.50) |\n"
        report += "|-------|-----------------------|------------------|------------------------|------------------|-----------------------|\n"
        
        for stage_name, results in results_list:
            qd = results['quality_distribution']
            report += f"| **{stage_name}** "
            for category in ['excellent', 'good', 'acceptable', 'poor', 'very_poor']:
                count = qd[category]['count']
                pct = qd[category]['percentage']
                report += f"| {count}<br/>({pct:.1f}%) "
            report += "|\n"
        
        # Performance Comparison
        report += "\n## 🔍 Performance Comparison\n\n"
        if len(results_list) > 1:
            baseline_mean = results_list[0][1]['overall_metrics']['cosine_similarity']['mean']
            baseline_std = results_list[0][1]['overall_metrics']['cosine_similarity']['std']
            
            report += f"**Baseline ({results_list[0][0]})**: {baseline_mean:.4f} ± {baseline_std:.4f}\n\n"
            report += "| Stage | Mean Score | Absolute Improvement | Relative Improvement |\n"
            report += "|-------|------------|---------------------|----------------------|\n"
            report += f"| {results_list[0][0]} | {baseline_mean:.4f} | - | - (baseline) |\n"
            
            for stage_name, results in results_list[1:]:
                current_mean = results['overall_metrics']['cosine_similarity']['mean']
                improvement = current_mean - baseline_mean
                improvement_pct = (improvement / baseline_mean) * 100
                direction = "↑" if improvement > 0 else "↓" if improvement < 0 else "↔"
                report += f"| {stage_name} | {current_mean:.4f} | {improvement:+.4f} {direction} | {improvement_pct:+.2f}% |\n"
        
        # Key Findings
        report += "\n## 💡 Key Findings\n\n"
        
        # Find best model
        best_idx = max(range(len(results_list)), 
                      key=lambda i: results_list[i][1]['overall_metrics']['cosine_similarity']['mean'])
        best_stage, best_results = results_list[best_idx]
        best_mean = best_results['overall_metrics']['cosine_similarity']['mean']
        
        report += f"1. **Best Performing Model**: {best_stage}\n"
        report += f"   - Mean cosine similarity: {best_mean:.4f}\n"
        
        # Check for plateau
        if len(results_list) >= 3:
            improvements = []
            for i in range(1, len(results_list)):
                prev_mean = results_list[i-1][1]['overall_metrics']['cosine_similarity']['mean']
                curr_mean = results_list[i][1]['overall_metrics']['cosine_similarity']['mean']
                improvements.append(curr_mean - prev_mean)
            
            avg_improvement = sum(improvements) / len(improvements)
            report += f"\n2. **Training Progression**: Average improvement per stage: {avg_improvement:+.4f}\n"
            
            # Identify if there's a plateau
            small_improvements = [imp for imp in improvements[-2:] if abs(imp) < 0.01]
            if len(small_improvements) >= 1:
                report += f"   - ⚠️  Performance plateau observed in later stages\n"
        
        # Quality distribution analysis
        excellent_counts = [r[1]['quality_distribution']['excellent']['count'] for r in results_list]
        best_excellent_idx = excellent_counts.index(max(excellent_counts))
        report += f"\n3. **Quality Distribution**: {results_list[best_excellent_idx][0]} has the highest proportion of excellent predictions ({max(excellent_counts)}/{results_list[0][1]['metadata']['n_samples']})\n"
        
        # Recommendations
        report += "\n## 🎯 Recommendations\n\n"
        
        # Calculate cost-benefit
        report += "### Cost-Benefit Analysis\n\n"
        report += "Considering training time vs performance improvement:\n\n"
        
        # Training times from Result_QLoRA_Scale.md
        training_times = {'1k': 1.38, '2k': 2.93, '3k': 4.53, '4k': 6.32}  # hours
        
        for i, (stage_name, results) in enumerate(results_list):
            stage_key = stage_name.split('(')[1].split()[0]  # Extract '1k', '2k', etc.
            mean_score = results['overall_metrics']['cosine_similarity']['mean']
            training_time = training_times.get(stage_key, 0)
            
            if i == 0:
                efficiency = mean_score / training_time
                report += f"- **{stage_name}**: {mean_score:.4f} score, {training_time:.2f}h training\n"
                report += f"  - Efficiency: {efficiency:.4f} score/hour (baseline)\n"
            else:
                baseline_mean = results_list[0][1]['overall_metrics']['cosine_similarity']['mean']
                baseline_time = training_times.get(results_list[0][0].split('(')[1].split()[0], 1)
                
                improvement = mean_score - baseline_mean
                time_increase = training_time - baseline_time
                efficiency = improvement / time_increase if time_increase > 0 else 0
                
                report += f"- **{stage_name}**: {mean_score:.4f} score, {training_time:.2f}h training\n"
                report += f"  - Additional training: +{time_increase:.2f}h for +{improvement:.4f} score improvement\n"
                report += f"  - Efficiency: {efficiency:.4f} score improvement per additional hour\n"
        
        report += "\n### Deployment Recommendation\n\n"
        
        # Find most efficient model (best score improvement per hour)
        if len(results_list) > 1:
            efficiencies = []
            for i in range(1, len(results_list)):
                stage_key = results_list[i][0].split('(')[1].split()[0]
                baseline_key = results_list[0][0].split('(')[1].split()[0]
                
                improvement = results_list[i][1]['overall_metrics']['cosine_similarity']['mean'] - \
                            results_list[0][1]['overall_metrics']['cosine_similarity']['mean']
                time_increase = training_times.get(stage_key, 0) - training_times.get(baseline_key, 0)
                
                if time_increase > 0:
                    efficiency = improvement / time_increase
                    efficiencies.append((i, efficiency, stage_key))
            
            if efficiencies:
                best_efficiency_idx, best_efficiency, best_stage_key = max(efficiencies, key=lambda x: x[1])
                recommended_stage = results_list[best_efficiency_idx][0]
                recommended_score = results_list[best_efficiency_idx][1]['overall_metrics']['cosine_similarity']['mean']
                
                report += f"**Recommended Model**: {recommended_stage}\n\n"
                report += f"- Best efficiency: {best_efficiency:.4f} score improvement per hour\n"
                report += f"- Mean cosine similarity: {recommended_score:.4f}\n"
                report += f"- Provides good balance between performance and training cost\n"
        
        # Statistical significance note
        report += "\n### Statistical Significance\n\n"
        report += "For rigorous analysis, consider running statistical tests:\n"
        report += "- Mann-Whitney U test for pairwise comparisons\n"
        report += "- Effect size calculation (Cohen's d)\n"
        report += "- Confidence intervals for mean differences\n"
        
        # Metadata
        report += "\n---\n\n"
        report += "## 📋 Evaluation Metadata\n\n"
        if results_list:
            metadata = results_list[0][1]['metadata']
            report += f"- **Evaluation Date**: {metadata['timestamp']}\n"
            report += f"- **Embedding Model**: {metadata['model_name']}\n"
            report += f"- **Test Set Size**: {metadata['n_samples']} samples\n"
            report += f"- **Device**: {metadata['device']}\n"
        
        report += "\n---\n\n"
        report += "*Generated by Progressive Evaluation Pipeline v0.5*\n"
        
        return report
    
    def run_complete_pipeline(self, limit: int = None):
        """
        Run the complete evaluation pipeline
        
        Args:
            limit: Optional limit on number of images (for testing)
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting v0.5 Complete Evaluation Pipeline")
        print(f"{'='*60}")
        print(f"Models to evaluate: {', '.join(self.models.keys())}")
        if limit:
            print(f"⚠️  Testing mode: Limited to {limit} images")
        print()
        
        start_time = datetime.now()
        evaluation_results = {}
        
        # Process each model
        for stage in ['1k', '2k', '3k', '4k']:
            if stage not in self.models:
                print(f"\n⚠️  Skipping {stage}: Model not found")
                continue
            
            try:
                # Run inference
                inference_csv = self.run_inference(stage, limit=limit)
                
                # Run evaluation
                evaluation_json = self.run_evaluation(stage, inference_csv)
                
                evaluation_results[stage] = evaluation_json
                
            except Exception as e:
                print(f"\n❌ Failed to process {stage}: {e}")
                continue
        
        # Create comparison report
        if evaluation_results:
            report_path = self.create_comparison_report(evaluation_results)
        
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"✅ Pipeline Complete!")
        print(f"{'='*60}")
        print(f"Total duration: {total_duration}")
        print(f"Evaluated models: {len(evaluation_results)}")
        print(f"Results directory: {self.output_base_dir}")
        if evaluation_results:
            print(f"Comparison report: {report_path}")
        print()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run complete v0.5 evaluation pipeline for all progressive training models"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test images (for testing)"
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/v03_fine_tuning/test_set_n800.csv",
        help="Path to test set CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/v03_fine_tuning/evaluations",
        help="Output directory for evaluation results"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ProgressiveEvaluationPipeline(
        test_csv=args.test_csv,
        output_base_dir=args.output_dir
    )
    
    # Run complete pipeline
    pipeline.run_complete_pipeline(limit=args.limit)


if __name__ == "__main__":
    main()
