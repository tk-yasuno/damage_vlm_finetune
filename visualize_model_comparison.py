#!/usr/bin/env python3
"""
Model Comparison Visualization for v0.5.1 Inference Results
Generates comprehensive quality comparison charts for 1k/2k/3k/4k models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

# Quality tier thresholds
TIERS = {
    'Excellent': 0.85,
    'Good': 0.75,
    'Acceptable': 0.65,
    'Poor': 0.50,
    'Very Poor': 0.0
}

TIER_COLORS = {
    'Excellent': '#2ecc71',    # Green
    'Good': '#3498db',         # Blue
    'Acceptable': '#f39c12',   # Orange
    'Poor': '#e74c3c',         # Red
    'Very Poor': '#95a5a6'     # Gray
}


def load_model_results(model_name):
    """Load inference results for a specific model"""
    csv_path = f"data/v03_fine_tuning/evaluations/inference_results_{model_name}.csv"
    df = pd.read_csv(csv_path)
    return df['cosine_similarity'].values


def classify_tier(similarity):
    """Classify a similarity score into quality tier"""
    if similarity >= TIERS['Excellent']:
        return 'Excellent'
    elif similarity >= TIERS['Good']:
        return 'Good'
    elif similarity >= TIERS['Acceptable']:
        return 'Acceptable'
    elif similarity >= TIERS['Poor']:
        return 'Poor'
    else:
        return 'Very Poor'


def calculate_tier_distribution(similarities):
    """Calculate quality tier distribution"""
    tiers = [classify_tier(s) for s in similarities]
    tier_order = ['Excellent', 'Good', 'Acceptable', 'Poor', 'Very Poor']
    counts = {tier: tiers.count(tier) for tier in tier_order}
    percentages = {tier: (count / len(similarities)) * 100 
                   for tier, count in counts.items()}
    return counts, percentages


def plot_similarity_comparison(models_data, output_path):
    """Plot mean cosine similarity comparison with error bars"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(models_data.keys())
    means = [np.mean(models_data[m]) for m in models]
    stds = [np.std(models_data[m]) for m in models]
    
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                  color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'],
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.4f}±{std:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add tier boundary lines
    ax.axhline(y=0.85, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (≥0.85)')
    ax.axhline(y=0.75, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Good (≥0.75)')
    ax.axhline(y=0.65, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Acceptable (≥0.65)')
    ax.axhline(y=0.50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Poor (≥0.50)')
    
    ax.set_xlabel('Training Scale', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cosine Similarity (Mean ± Std)', fontsize=12, fontweight='bold')
    ax.set_title('Cosine Similarity by Training Scale\n(800 Test Samples)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0.60, 0.90)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_tier_distribution(models_data, output_path):
    """Plot quality tier distribution as stacked bar chart"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = list(models_data.keys())
    tier_order = ['Excellent', 'Good', 'Acceptable', 'Poor', 'Very Poor']
    
    # Calculate distributions
    distributions = {}
    for model in models:
        _, percentages = calculate_tier_distribution(models_data[model])
        distributions[model] = percentages
    
    # Create stacked bars
    bottoms = np.zeros(len(models))
    x_pos = np.arange(len(models))
    
    for tier in tier_order:
        values = [distributions[m][tier] for m in models]
        bars = ax.bar(x_pos, values, bottom=bottoms, 
                     label=tier, color=TIER_COLORS[tier],
                     edgecolor='white', linewidth=2)
        
        # Add percentage labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 3:  # Only show label if segment is large enough
                height = bottoms[i] + val/2
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}%',
                       ha='center', va='center', fontsize=9, 
                       fontweight='bold', color='white')
        
        bottoms += values
    
    ax.set_xlabel('Training Scale', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Predictions (%)', fontsize=12, fontweight='bold')
    ax.set_title('Quality Tier Distribution by Training Scale\n(800 Test Samples)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_tier_grouped(models_data, output_path):
    """Plot quality tier distribution as grouped bar chart"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    models = list(models_data.keys())
    tier_order = ['Excellent', 'Good', 'Acceptable', 'Poor', 'Very Poor']
    
    # Calculate distributions
    distributions = {}
    for model in models:
        counts, percentages = calculate_tier_distribution(models_data[model])
        distributions[model] = percentages
    
    # Prepare data for grouped bars
    x = np.arange(len(tier_order))
    width = 0.2
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    
    for i, (model, offset) in enumerate(zip(models, offsets)):
        values = [distributions[model][tier] for tier in tier_order]
        bars = ax.bar(x + offset, values, width, 
                     label=model,
                     color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'][i],
                     edgecolor='black', linewidth=1, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{val:.1f}',
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Quality Tier', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Predictions (%)', fontsize=12, fontweight='bold')
    ax.set_title('Quality Tier Distribution Comparison\n(800 Test Samples per Model)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_order, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_distribution_violin(models_data, output_path):
    """Plot cosine similarity distribution as violin plot"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data
    data = []
    labels = []
    for model, similarities in models_data.items():
        data.extend(similarities)
        labels.extend([model] * len(similarities))
    
    df = pd.DataFrame({'Model': labels, 'Cosine Similarity': data})
    
    # Create violin plot
    parts = ax.violinplot([models_data[m] for m in models_data.keys()],
                          positions=range(len(models_data)),
                          showmeans=True, showmedians=True, widths=0.7)
    
    # Color the violins
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    
    # Add tier boundary lines
    ax.axhline(y=0.85, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Excellent')
    ax.axhline(y=0.75, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Good')
    ax.axhline(y=0.65, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Acceptable')
    ax.axhline(y=0.50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Poor')
    
    ax.set_xlabel('Training Scale', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_title('Cosine Similarity Distribution by Training Scale\n(Violin Plot, n=800 per model)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(models_data)))
    ax.set_xticklabels(list(models_data.keys()), fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_summary_table(models_data, output_path):
    """Create a summary table with key statistics"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    models = list(models_data.keys())
    table_data = [['Metric'] + models]
    
    # Add statistics
    means = [f"{np.mean(models_data[m]):.4f}" for m in models]
    table_data.append(['Mean'] + means)
    
    stds = [f"{np.std(models_data[m]):.4f}" for m in models]
    table_data.append(['Std Dev'] + stds)
    
    mins = [f"{np.min(models_data[m]):.4f}" for m in models]
    table_data.append(['Min'] + mins)
    
    maxs = [f"{np.max(models_data[m]):.4f}" for m in models]
    table_data.append(['Max'] + maxs)
    
    medians = [f"{np.median(models_data[m]):.4f}" for m in models]
    table_data.append(['Median'] + medians)
    
    # Add tier percentages
    tier_order = ['Excellent', 'Good', 'Acceptable', 'Poor', 'Very Poor']
    for tier in tier_order:
        percentages = []
        for m in models:
            _, perc = calculate_tier_distribution(models_data[m])
            percentages.append(f"{perc[tier]:.1f}%")
        table_data.append([f'{tier} (%)'] + percentages)
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.11, 0.11, 0.11, 0.11])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(models) + 1):
        cell = table[(0, i)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(weight='bold', color='white')
    
    # Style metric column
    for i in range(1, len(table_data)):
        cell = table[(i, 0)]
        cell.set_facecolor('#ecf0f1')
        cell.set_text_props(weight='bold')
    
    # Color code tier rows
    tier_row_start = 6
    for i, tier in enumerate(tier_order):
        for j in range(1, len(models) + 1):
            cell = table[(tier_row_start + i, j)]
            cell.set_facecolor(TIER_COLORS[tier])
            cell.set_alpha(0.3)
    
    # Highlight best values
    best_model_idx = np.argmax([np.mean(models_data[m]) for m in models])
    for i in range(1, 6):  # Stats rows
        cell = table[(i, best_model_idx + 1)]
        cell.set_facecolor('#f1c40f')
        cell.set_alpha(0.4)
    
    ax.set_title('Model Comparison Summary Statistics\n(800 Test Samples per Model)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    print("\n" + "="*70)
    print("Model Comparison Visualization for v0.5.1 Inference Results")
    print("="*70 + "\n")
    
    # Load all model results
    print("📂 Loading model results...")
    models_data = {}
    for model in ['1k', '2k', '3k', '4k']:
        try:
            similarities = load_model_results(model)
            models_data[model] = similarities
            print(f"   ✓ {model}: {len(similarities)} samples, "
                  f"mean={np.mean(similarities):.4f}±{np.std(similarities):.4f}")
        except Exception as e:
            print(f"   ✗ {model}: {e}")
    
    if len(models_data) == 0:
        print("\n❌ No model results found!")
        return
    
    # Create output directory
    output_dir = Path("data/v03_fine_tuning/evaluations/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Generating visualizations...")
    
    # Generate all plots
    plot_similarity_comparison(
        models_data, 
        output_dir / "01_similarity_comparison.png"
    )
    
    plot_tier_distribution(
        models_data, 
        output_dir / "02_tier_distribution_stacked.png"
    )
    
    plot_tier_grouped(
        models_data, 
        output_dir / "03_tier_distribution_grouped.png"
    )
    
    plot_distribution_violin(
        models_data, 
        output_dir / "04_similarity_distribution_violin.png"
    )
    
    plot_summary_table(
        models_data, 
        output_dir / "05_summary_table.png"
    )
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"   • {file.name}")
    print()


if __name__ == "__main__":
    main()
