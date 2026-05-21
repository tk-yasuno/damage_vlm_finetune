"""
v0.3 Dataset Preparation: Stratified Train/Test Split
Prepare progressive fine-tuning datasets (1k/2k/3k/4k) with fixed test set (800)
from n=10,789 bridge damage images with ground truth annotations
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from collections import Counter

# Configuration
PROJECT_ROOT = Path(__file__).parent
MASTER_EXCEL = PROJECT_ROOT / "data" / "image_text_inspect_n10789" / "Rank_c_image_text_n10789.xlsx"
IMAGE_DIR = PROJECT_ROOT / "data" / "image_text_inspect_n10789" / "rank_c_images_n10789"
V02_DETAIL_CSV = PROJECT_ROOT / "llava_quantization_comparison_detail.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "v03_fine_tuning"

# Training dataset sizes (progressive scaling)
TRAIN_SIZES = [1000, 2000, 3000, 4000]
TEST_SIZE = 800
RANDOM_SEED = 42

# Text quality filters
MIN_TEXT_LENGTH = 15
MAX_TEXT_LENGTH = 500
EXCLUDED_PATTERNS = [
    r'^\s*$',           # Empty or whitespace only
    r'^\s*[？?]+\s*$',  # Only question marks
    r'^\s*[-ー]+\s*$',  # Only dashes
    r'^\s*なし\s*$',    # "なし" (none)
    r'^\s*該当なし\s*$', # "該当なし" (not applicable)
]

# Required keywords for content validation (at least one must be present)
REQUIRED_KEYWORDS = [
    # Damage types
    'ひび割れ', 'クラック', '亀裂', '鉄筋', '露出', '腐食', '錆', 'さび',
    '剥離', 'はく離', '剥落', '欠損', '劣化', '変色', '漏水',
    # Components
    '桁', '床版', '支承', '橋脚', '橋台', '高欄', '伸縮装置', '主桁',
    '横桁', '対傾構', 'コンクリート', '鋼材', '塗装',
    # Severity indicators
    '著しい', '顕著', '大きな', '小規模', '軽微', '進行',
]


def load_master_data() -> pd.DataFrame:
    """Load master Excel file with ground truth annotations"""
    print("=" * 80)
    print("Step 1: Loading Master Data")
    print("=" * 80)
    
    if not MASTER_EXCEL.exists():
        raise FileNotFoundError(f"Master Excel file not found: {MASTER_EXCEL}")
    
    print(f"📁 Loading: {MASTER_EXCEL.name}")
    df = pd.read_excel(MASTER_EXCEL)
    print(f"✓ Loaded {len(df)} rows")
    print(f"  Columns: {', '.join(df.columns.tolist())}")
    
    return df


def apply_text_quality_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Apply quality filters to 所見 (observation) text field"""
    print("\n" + "=" * 80)
    print("Step 2: Applying Text Quality Filters")
    print("=" * 80)
    
    stats = {
        'initial_count': len(df),
        'filtered_stages': {}
    }
    
    # Check if required columns exist
    if 'ファイルパス' not in df.columns or '所見' not in df.columns:
        raise ValueError(f"Required columns not found. Available: {df.columns.tolist()}")
    
    # Filter 1: Drop rows with missing ファイルパス or 所見
    df_filtered = df.dropna(subset=['ファイルパス', '所見']).copy()
    stats['filtered_stages']['missing_values'] = len(df) - len(df_filtered)
    print(f"✓ Filter 1 (Missing values): Removed {stats['filtered_stages']['missing_values']} rows")
    
    # Filter 2: Text length constraints
    df_filtered['text_length'] = df_filtered['所見'].astype(str).str.len()
    df_length = df_filtered[
        (df_filtered['text_length'] >= MIN_TEXT_LENGTH) &
        (df_filtered['text_length'] <= MAX_TEXT_LENGTH)
    ].copy()
    stats['filtered_stages']['length_constraint'] = len(df_filtered) - len(df_length)
    print(f"✓ Filter 2 (Length {MIN_TEXT_LENGTH}-{MAX_TEXT_LENGTH} chars): Removed {stats['filtered_stages']['length_constraint']} rows")
    
    # Filter 3: Exclude patterns (なし, ???, etc.)
    def has_excluded_pattern(text: str) -> bool:
        text_str = str(text).strip()
        for pattern in EXCLUDED_PATTERNS:
            if re.match(pattern, text_str):
                return True
        return False
    
    df_pattern = df_length[~df_length['所見'].apply(has_excluded_pattern)].copy()
    stats['filtered_stages']['excluded_patterns'] = len(df_length) - len(df_pattern)
    print(f"✓ Filter 3 (Excluded patterns): Removed {stats['filtered_stages']['excluded_patterns']} rows")
    
    # Filter 4: Must contain at least one required keyword
    def has_required_keyword(text: str) -> bool:
        text_str = str(text).lower()
        return any(keyword in text_str for keyword in REQUIRED_KEYWORDS)
    
    df_final = df_pattern[df_pattern['所見'].apply(has_required_keyword)].copy()
    stats['filtered_stages']['missing_keywords'] = len(df_pattern) - len(df_final)
    print(f"✓ Filter 4 (Required keywords): Removed {stats['filtered_stages']['missing_keywords']} rows")
    
    stats['final_count'] = len(df_final)
    stats['total_removed'] = stats['initial_count'] - stats['final_count']
    stats['retention_rate'] = (stats['final_count'] / stats['initial_count']) * 100
    
    print(f"\n📊 Filtering Summary:")
    print(f"  Initial: {stats['initial_count']} rows")
    print(f"  Final: {stats['final_count']} rows")
    print(f"  Removed: {stats['total_removed']} rows ({100 - stats['retention_rate']:.1f}%)")
    print(f"  Retention: {stats['retention_rate']:.1f}%")
    
    return df_final, stats


def merge_v02_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Merge v0.2 VLM output for component/damage classification metadata"""
    print("\n" + "=" * 80)
    print("Step 3: Merging v0.2 Metadata (Component/Damage Types)")
    print("=" * 80)
    
    if not V02_DETAIL_CSV.exists():
        print(f"⚠️  Warning: v0.2 detail CSV not found: {V02_DETAIL_CSV}")
        print("   Proceeding without metadata (stratified sampling will use random selection)")
        df['component_type'] = 'unknown'
        df['damage_type'] = 'unknown'
        return df
    
    print(f"📁 Loading: {V02_DETAIL_CSV.name}")
    v02_df = pd.read_csv(V02_DETAIL_CSV)
    print(f"✓ Loaded {len(v02_df)} v0.2 results")
    
    # Extract filename from ファイルパス for matching
    df['filename'] = df['ファイルパス'].apply(lambda x: Path(str(x)).name)
    
    # Filter v0.2 to Q5_K_M only
    v02_q5 = v02_df[v02_df['quantization'] == 'Q5_K_M'].copy() if 'quantization' in v02_df.columns else v02_df.copy()
    
    # Merge on filename (assuming v0.2 'image' column contains filenames)
    if 'image' in v02_q5.columns:
        v02_q5 = v02_q5.rename(columns={'image': 'filename'})
    
    # Extract component and damage type from v0.2 structured output
    # (Assuming v0.2 has 'component_type' and 'damage_type' columns or we parse from 'result_preview')
    df_merged = df.merge(
        v02_q5[['filename', 'component_type', 'damage_type']] if 'component_type' in v02_q5.columns 
        else v02_q5[['filename']],
        on='filename',
        how='left'
    )
    
    # Fill missing metadata with 'unknown'
    if 'component_type' not in df_merged.columns:
        df_merged['component_type'] = 'unknown'
    if 'damage_type' not in df_merged.columns:
        df_merged['damage_type'] = 'unknown'
    
    df_merged['component_type'] = df_merged['component_type'].fillna('unknown')
    df_merged['damage_type'] = df_merged['damage_type'].fillna('unknown')
    
    print(f"\n✓ Merged metadata:")
    print(f"  Component types: {df_merged['component_type'].nunique()} unique")
    print(f"  Damage types: {df_merged['damage_type'].nunique()} unique")
    
    return df_merged


def stratified_sampling(
    df: pd.DataFrame,
    n_samples: int,
    strata_cols: List[str],
    random_state: int = RANDOM_SEED
) -> pd.DataFrame:
    """Perform stratified sampling to maintain distribution across strata"""
    
    # Create stratification key
    df['_strata'] = df[strata_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    
    # Count samples per stratum
    strata_counts = df['_strata'].value_counts()
    strata_proportions = strata_counts / len(df)
    
    # Calculate target samples per stratum
    target_samples = (strata_proportions * n_samples).round().astype(int)
    
    # Adjust for rounding errors
    while target_samples.sum() != n_samples:
        if target_samples.sum() < n_samples:
            # Add to largest stratum
            target_samples[target_samples.idxmax()] += 1
        else:
            # Remove from largest stratum
            target_samples[target_samples.idxmax()] -= 1
    
    # Sample from each stratum
    sampled_dfs = []
    for stratum, n_target in target_samples.items():
        stratum_df = df[df['_strata'] == stratum]
        n_sample = min(n_target, len(stratum_df))  # Don't oversample
        sampled_dfs.append(
            stratum_df.sample(n=n_sample, random_state=random_state)
        )
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    result = result.drop(columns=['_strata'])
    
    return result


def create_test_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create fixed test set (800 samples) using stratified sampling"""
    print("\n" + "=" * 80)
    print("Step 4: Creating Fixed Test Set (N=800)")
    print("=" * 80)
    
    if len(df) < TEST_SIZE:
        raise ValueError(f"Insufficient data: {len(df)} < {TEST_SIZE}")
    
    # Stratified sampling by component_type and damage_type
    test_df = stratified_sampling(
        df,
        n_samples=TEST_SIZE,
        strata_cols=['component_type', 'damage_type'],
        random_state=RANDOM_SEED
    )
    
    # Remaining data for training
    train_pool_df = df[~df.index.isin(test_df.index)].copy()
    
    print(f"✓ Test set created: {len(test_df)} samples")
    print(f"✓ Training pool remaining: {len(train_pool_df)} samples")
    
    # Distribution analysis
    print(f"\n📊 Test Set Distribution:")
    print(f"  Component types:")
    for comp, count in test_df['component_type'].value_counts().head(5).items():
        print(f"    - {comp}: {count} ({count/len(test_df)*100:.1f}%)")
    print(f"  Damage types:")
    for dmg, count in test_df['damage_type'].value_counts().head(5).items():
        print(f"    - {dmg}: {count} ({count/len(test_df)*100:.1f}%)")
    
    return test_df, train_pool_df


def create_progressive_train_sets(train_pool_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Create progressive training sets (1k, 2k, 3k, 4k) using stratified sampling"""
    print("\n" + "=" * 80)
    print("Step 5: Creating Progressive Training Sets")
    print("=" * 80)
    
    train_sets = {}
    
    for size in TRAIN_SIZES:
        if len(train_pool_df) < size:
            print(f"⚠️  Warning: Insufficient data for {size} samples. Using all {len(train_pool_df)} samples.")
            train_sets[size] = train_pool_df.copy()
            continue
        
        # Stratified sampling
        train_df = stratified_sampling(
            train_pool_df,
            n_samples=size,
            strata_cols=['component_type', 'damage_type'],
            random_state=RANDOM_SEED + size  # Different seed for each size
        )
        
        train_sets[size] = train_df
        print(f"✓ Train set ({size}): {len(train_df)} samples")
    
    return train_sets


def save_datasets(
    test_df: pd.DataFrame,
    train_sets: Dict[int, pd.DataFrame],
    filter_stats: Dict
):
    """Save all datasets and generate report"""
    print("\n" + "=" * 80)
    print("Step 6: Saving Datasets")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save test set
    test_path = OUTPUT_DIR / "test_set_n800.csv"
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
    print(f"✓ Saved test set: {test_path}")
    
    # Save training sets
    for size, train_df in train_sets.items():
        train_path = OUTPUT_DIR / f"train_{size//1000}k.csv"
        train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
        print(f"✓ Saved train set: {train_path}")
    
    # Generate statistics report
    report_path = OUTPUT_DIR / "dataset_preparation_report.md"
    generate_report(test_df, train_sets, filter_stats, report_path)
    print(f"✓ Saved report: {report_path}")


def generate_report(
    test_df: pd.DataFrame,
    train_sets: Dict[int, pd.DataFrame],
    filter_stats: Dict,
    output_path: Path
):
    """Generate comprehensive dataset preparation report"""
    
    report = f"""# Dataset Preparation Report - v0.3
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

Prepared progressive fine-tuning datasets from n=10,789 bridge damage images.

## Filtering Statistics

| Stage | Count | Removed | Retention |
|-------|-------|---------|-----------|
| Initial | {filter_stats['initial_count']:,} | - | 100.0% |
| After missing values | {filter_stats['initial_count'] - filter_stats['filtered_stages']['missing_values']:,} | {filter_stats['filtered_stages']['missing_values']:,} | {((filter_stats['initial_count'] - filter_stats['filtered_stages']['missing_values']) / filter_stats['initial_count'] * 100):.1f}% |
| After length filter | {filter_stats['initial_count'] - filter_stats['filtered_stages']['missing_values'] - filter_stats['filtered_stages']['length_constraint']:,} | {filter_stats['filtered_stages']['length_constraint']:,} | - |
| After pattern filter | {filter_stats['initial_count'] - filter_stats['filtered_stages']['missing_values'] - filter_stats['filtered_stages']['length_constraint'] - filter_stats['filtered_stages']['excluded_patterns']:,} | {filter_stats['filtered_stages']['excluded_patterns']:,} | - |
| **Final** | **{filter_stats['final_count']:,}** | **{filter_stats['total_removed']:,}** | **{filter_stats['retention_rate']:.1f}%** |

### Filtering Criteria

- **Text Length**: {MIN_TEXT_LENGTH}-{MAX_TEXT_LENGTH} characters
- **Excluded Patterns**: Empty, "なし", "???", etc.
- **Required Content**: Must contain damage/component keywords

## Dataset Splits

| Split | Size | Percentage |
|-------|------|------------|
| **Test Set** | {len(test_df):,} | {len(test_df)/(len(test_df) + len(train_sets[max(train_sets.keys())]))*100:.1f}% |
"""
    
    for size in sorted(train_sets.keys()):
        report += f"| Train Set ({size//1000}k) | {len(train_sets[size]):,} | {len(train_sets[size])/(len(test_df) + len(train_sets[size]))*100:.1f}% |\n"
    
    report += f"\n**Random Seed**: {RANDOM_SEED} (for reproducibility)\n\n"
    
    # Test set distribution
    report += f"## Test Set Distribution (N={len(test_df)})\n\n"
    report += "### Component Types\n\n"
    comp_dist = test_df['component_type'].value_counts()
    for comp, count in comp_dist.items():
        report += f"- **{comp}**: {count} ({count/len(test_df)*100:.1f}%)\n"
    
    report += "\n### Damage Types\n\n"
    dmg_dist = test_df['damage_type'].value_counts()
    for dmg, count in dmg_dist.items():
        report += f"- **{dmg}**: {count} ({count/len(test_df)*100:.1f}%)\n"
    
    # Text statistics
    report += f"\n## Text Statistics\n\n"
    report += f"### Test Set (所見 field)\n\n"
    report += f"- **Mean length**: {test_df['text_length'].mean():.1f} characters\n"
    report += f"- **Median length**: {test_df['text_length'].median():.1f} characters\n"
    report += f"- **Std deviation**: {test_df['text_length'].std():.1f} characters\n"
    report += f"- **Min length**: {test_df['text_length'].min()} characters\n"
    report += f"- **Max length**: {test_df['text_length'].max()} characters\n\n"
    
    # Training set comparisons
    report += "### Training Sets Comparison\n\n"
    report += "| Set | Mean Length | Median Length | Component Types | Damage Types |\n"
    report += "|-----|-------------|---------------|-----------------|---------------|\n"
    for size in sorted(train_sets.keys()):
        train_df = train_sets[size]
        report += f"| Train {size//1000}k | {train_df['text_length'].mean():.1f} | {train_df['text_length'].median():.1f} | {train_df['component_type'].nunique()} | {train_df['damage_type'].nunique()} |\n"
    
    report += f"\n## Stratification Strategy\n\n"
    report += "- **Method**: Stratified sampling by component_type × damage_type\n"
    report += "- **Purpose**: Maintain balanced distribution across damage scenarios\n"
    report += "- **Test Set**: Fixed with seed={} for reproducibility\n".format(RANDOM_SEED)
    report += "- **Training Sets**: Progressive scaling (1k → 2k → 3k → 4k)\n"
    
    report += f"\n## Files Generated\n\n"
    report += f"- `test_set_n800.csv`: Fixed test set ({len(test_df)} samples)\n"
    for size in sorted(train_sets.keys()):
        report += f"- `train_{size//1000}k.csv`: Training set ({len(train_sets[size])} samples)\n"
    
    report += f"\n## Next Steps\n\n"
    report += "1. **Verify distributions**: Check that stratification maintained component/damage balance\n"
    report += "2. **Run v0.4.1**: Fine-tune with train_1k.csv\n"
    report += "3. **Evaluate**: Test on test_set_n800.csv using Vector Similarity\n"
    report += "4. **Scale up**: Progress to 2k → 3k → 4k training sets\n"
    report += "5. **Compare results**: Analyze performance scaling with dataset size\n"
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)


def main():
    """Main execution pipeline"""
    print("\n🔧 Bridge Damage VLM Dataset Preparation - v0.3")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Step 1: Load master data
        df = load_master_data()
        
        # Step 2: Apply quality filters
        df_filtered, filter_stats = apply_text_quality_filters(df)
        
        # Step 3: Merge v0.2 metadata
        df_merged = merge_v02_metadata(df_filtered)
        
        # Step 4: Create test set
        test_df, train_pool_df = create_test_set(df_merged)
        
        # Step 5: Create progressive training sets
        train_sets = create_progressive_train_sets(train_pool_df)
        
        # Step 6: Save all datasets
        save_datasets(test_df, train_sets, filter_stats)
        
        print("\n" + "=" * 80)
        print("✅ Dataset Preparation Complete!")
        print("=" * 80)
        print(f"\n📁 Output directory: {OUTPUT_DIR}")
        print(f"   - test_set_n800.csv")
        for size in TRAIN_SIZES:
            print(f"   - train_{size//1000}k.csv")
        print(f"   - dataset_preparation_report.md")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
