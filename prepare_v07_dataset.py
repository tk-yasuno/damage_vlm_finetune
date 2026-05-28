"""
v0.7 Dataset Preparation: Denoised Pairdata (pairdata_v2)

Removes out-of-scope sentences from inspection text before stratified
train/test split.  Two noise categories are eliminated:
  A. Temporal comparison (requires 2+ images / time points)
  B. Maintenance recommendations (requires bridge attribute metadata)

The same cleaning is applied to both the training pool and the test-set
ground truth, so that evaluation is fair for models trained on pairdata_v2.
Direct comparison with v0.3 evaluation numbers (pairdata_v1 GT) is not
applicable by design.

Based on prepare_v03_dataset.py (same 6-step structure; Step 0 is new).
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

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path(__file__).parent
MASTER_EXCEL  = PROJECT_ROOT / "data" / "image_text_inspect_n10789" / "Rank_c_image_text_n10789.xlsx"
IMAGE_DIR     = PROJECT_ROOT / "data" / "image_text_inspect_n10789" / "rank_c_images_n10789"
V02_DETAIL_CSV = PROJECT_ROOT / "llava_quantization_comparison_detail.csv"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "v07_fine_tuning"

# Training dataset sizes (progressive scaling)
TRAIN_SIZES = [1000, 2000, 3000, 4000]
TEST_SIZE   = 800
RANDOM_SEED = 42

# Text quality filters (same as v0.3)
MIN_TEXT_LENGTH = 15
MAX_TEXT_LENGTH = 500
EXCLUDED_PATTERNS = [
    r'^\s*$',
    r'^\s*[？?]+\s*$',
    r'^\s*[-ー]+\s*$',
    r'^\s*なし\s*$',
    r'^\s*該当なし\s*$',
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

# ============================================================
# Noise pattern definitions (pairdata_v1 → pairdata_v2)
# ============================================================

# Category A: Temporal comparison (requires 2+ images / time points)
TEMPORAL_PATTERNS = [
    r'前回(より|から|の|点検|調査)',    # 前回より, 前回から, 前回点検, 前回調査
    r'前年(より|から|の)',              # 前年より, 前年から
    r'経時(的|変化)',                   # 経時的に, 経時変化
    r'新たに.{0,15}(発生|出現)',        # 新たに損傷が発生, 新たに出現
]

# Category B: Maintenance recommendations (requires bridge metadata)
MAINTENANCE_PATTERNS = [
    r'補修(が|を|は)?(必要|要する|望ましい|検討)',    # 補修が必要, 補修を要する
    r'予防保全',                                      # 予防保全
    r'対策(が|を)?(必要|要する|講じ)',                # 対策が必要, 対策を要する
    r'措置(が|を)?(必要|要する)',                     # 措置が必要, 措置を要する
    r'経過観察(が|を)?(必要|要する)',                 # 経過観察が必要
    r'早(期|急)に.{0,15}(補修|対策|措置|点検|確認)', # 早期に補修, 早急に対策
    r'維持管理計画',                                  # 維持管理計画
    r'補修工法|補修方法',                             # 補修工法, 補修方法
]


# ============================================================
# Step 0: Noise removal function
# ============================================================

def remove_out_of_scope_sentences(text: str) -> Tuple[str, Dict[str, int]]:
    """Remove out-of-scope sentences from inspection text (pairdata_v1 → pairdata_v2).

    Splitting is sentence-level on Japanese period (。).
    Each sentence is checked independently against Category A (temporal)
    and Category B (maintenance) patterns.  Matching sentences are dropped;
    the remaining sentences are re-joined.

    Returns:
        cleaned_text : str   — text with out-of-scope sentences removed
        stats        : dict  — counts of removed / kept sentences
    """
    # Split on 。, keeping the delimiter attached to each sentence
    parts = re.split(r'(?<=。)', str(text))

    stats = {'temporal_removed': 0, 'maintenance_removed': 0, 'sentences_kept': 0}
    clean_parts = []

    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            continue

        # Category A check
        if any(re.search(p, part_stripped) for p in TEMPORAL_PATTERNS):
            stats['temporal_removed'] += 1
            continue

        # Category B check
        if any(re.search(p, part_stripped) for p in MAINTENANCE_PATTERNS):
            stats['maintenance_removed'] += 1
            continue

        clean_parts.append(part)
        stats['sentences_kept'] += 1

    return ''.join(clean_parts).strip(), stats


# ============================================================
# Step 1: Load master data
# ============================================================

def load_master_data() -> pd.DataFrame:
    """Load master Excel file with ground truth annotations."""
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


# ============================================================
# Step 0 (pipeline order): Apply noise removal
# ============================================================

def apply_noise_removal(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Apply sentence-level noise removal to the 所見 field (pairdata_v1 → pairdata_v2).

    Applied to ALL rows including those that will become the test set,
    so that both training and evaluation use pairdata_v2 ground truth.
    """
    print("\n" + "=" * 80)
    print("Step 0: Noise Removal (pairdata_v1 → pairdata_v2)")
    print("=" * 80)

    total_temporal     = 0
    total_maintenance  = 0
    rows_affected      = 0
    rows_emptied       = 0

    cleaned_texts = []
    for _, row in df.iterrows():
        text = str(row['所見']) if pd.notna(row['所見']) else ''
        if not text:
            cleaned_texts.append(text)
            continue

        cleaned, stats = remove_out_of_scope_sentences(text)
        removed = stats['temporal_removed'] + stats['maintenance_removed']

        total_temporal    += stats['temporal_removed']
        total_maintenance += stats['maintenance_removed']
        if removed > 0:
            rows_affected += 1
        if not cleaned.strip():
            rows_emptied += 1

        cleaned_texts.append(cleaned)

    df = df.copy()
    df['所見'] = cleaned_texts

    noise_stats = {
        'temporal_sentences_removed':     total_temporal,
        'maintenance_sentences_removed':  total_maintenance,
        'rows_affected':                  rows_affected,
        'rows_emptied':                   rows_emptied,
    }

    print(f"✓ Category A (temporal) sentences removed    : {total_temporal:,}")
    print(f"✓ Category B (maintenance) sentences removed  : {total_maintenance:,}")
    print(f"  Rows with ≥1 sentence removed               : {rows_affected:,}")
    print(f"  Rows fully emptied (all sentences removed)  : {rows_emptied:,}")

    return df, noise_stats


# ============================================================
# Step 2: Text quality filters (post-denoising)
# ============================================================

def apply_text_quality_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Apply quality filters to 所見 field (same logic as v0.3, run after noise removal)."""
    print("\n" + "=" * 80)
    print("Step 2: Applying Text Quality Filters (post-denoising)")
    print("=" * 80)

    stats = {
        'initial_count': len(df),
        'filtered_stages': {}
    }

    if 'ファイルパス' not in df.columns or '所見' not in df.columns:
        raise ValueError(f"Required columns not found. Available: {df.columns.tolist()}")

    # Filter 1: Drop rows with missing ファイルパス or empty 所見
    df_filtered = df.dropna(subset=['ファイルパス', '所見']).copy()
    df_filtered = df_filtered[df_filtered['所見'].astype(str).str.strip() != ''].copy()
    stats['filtered_stages']['missing_values'] = len(df) - len(df_filtered)
    print(f"✓ Filter 1 (Missing/empty values): Removed {stats['filtered_stages']['missing_values']} rows")

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
        return any(re.match(p, text_str) for p in EXCLUDED_PATTERNS)

    df_pattern = df_length[~df_length['所見'].apply(has_excluded_pattern)].copy()
    stats['filtered_stages']['excluded_patterns'] = len(df_length) - len(df_pattern)
    print(f"✓ Filter 3 (Excluded patterns): Removed {stats['filtered_stages']['excluded_patterns']} rows")

    # Filter 4: Must contain at least one required keyword
    def has_required_keyword(text: str) -> bool:
        text_str = str(text)
        return any(keyword in text_str for keyword in REQUIRED_KEYWORDS)

    df_final = df_pattern[df_pattern['所見'].apply(has_required_keyword)].copy()
    stats['filtered_stages']['missing_keywords'] = len(df_pattern) - len(df_final)
    print(f"✓ Filter 4 (Required keywords): Removed {stats['filtered_stages']['missing_keywords']} rows")

    stats['final_count'] = len(df_final)
    stats['total_removed'] = stats['initial_count'] - stats['final_count']
    stats['retention_rate'] = (stats['final_count'] / stats['initial_count']) * 100

    print(f"\n📊 Filtering Summary:")
    print(f"  Initial : {stats['initial_count']} rows")
    print(f"  Final   : {stats['final_count']} rows")
    print(f"  Removed : {stats['total_removed']} rows ({100 - stats['retention_rate']:.1f}%)")
    print(f"  Retention: {stats['retention_rate']:.1f}%")

    return df_final, stats


# ============================================================
# Step 3: Merge v0.2 metadata
# ============================================================

def merge_v02_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Merge v0.2 VLM output for component/damage classification metadata."""
    print("\n" + "=" * 80)
    print("Step 3: Merging v0.2 Metadata (Component/Damage Types)")
    print("=" * 80)

    if not V02_DETAIL_CSV.exists():
        print(f"⚠️  Warning: v0.2 detail CSV not found: {V02_DETAIL_CSV}")
        print("   Proceeding without metadata (stratified sampling uses random selection)")
        df = df.copy()
        df['component_type'] = 'unknown'
        df['damage_type']    = 'unknown'
        return df

    print(f"📁 Loading: {V02_DETAIL_CSV.name}")
    v02_df = pd.read_csv(V02_DETAIL_CSV)
    print(f"✓ Loaded {len(v02_df)} v0.2 results")

    df['filename'] = df['ファイルパス'].apply(lambda x: Path(str(x)).name)
    v02_q5 = (v02_df[v02_df['quantization'] == 'Q5_K_M'].copy()
              if 'quantization' in v02_df.columns else v02_df.copy())
    if 'image' in v02_q5.columns:
        v02_q5 = v02_q5.rename(columns={'image': 'filename'})

    merge_cols = (['filename', 'component_type', 'damage_type']
                  if 'component_type' in v02_q5.columns else ['filename'])
    df_merged = df.merge(v02_q5[merge_cols], on='filename', how='left')

    if 'component_type' not in df_merged.columns:
        df_merged['component_type'] = 'unknown'
    if 'damage_type' not in df_merged.columns:
        df_merged['damage_type'] = 'unknown'

    df_merged['component_type'] = df_merged['component_type'].fillna('unknown')
    df_merged['damage_type']    = df_merged['damage_type'].fillna('unknown')

    print(f"\n✓ Merged metadata:")
    print(f"  Component types: {df_merged['component_type'].nunique()} unique")
    print(f"  Damage types:    {df_merged['damage_type'].nunique()} unique")

    return df_merged


# ============================================================
# Stratified sampling helper
# ============================================================

def stratified_sampling(
    df: pd.DataFrame,
    n_samples: int,
    strata_cols: List[str],
    random_state: int = RANDOM_SEED
) -> pd.DataFrame:
    """Stratified sampling maintaining distribution across strata."""
    df = df.copy()
    df['_strata'] = df[strata_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)

    strata_counts      = df['_strata'].value_counts()
    strata_proportions = strata_counts / len(df)
    target_samples     = (strata_proportions * n_samples).round().astype(int)

    while target_samples.sum() != n_samples:
        if target_samples.sum() < n_samples:
            target_samples[target_samples.idxmax()] += 1
        else:
            target_samples[target_samples.idxmax()] -= 1

    sampled_dfs = []
    for stratum, n_target in target_samples.items():
        stratum_df = df[df['_strata'] == stratum]
        n_sample = min(n_target, len(stratum_df))
        sampled_dfs.append(stratum_df.sample(n=n_sample, random_state=random_state))

    result = pd.concat(sampled_dfs, ignore_index=True)
    return result.drop(columns=['_strata'])


# ============================================================
# Step 4: Create test set
# ============================================================

def create_test_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create fixed test set (800 samples) using stratified sampling.

    Ground truth text has already been denoised in Step 0, so this
    test set uses pairdata_v2 annotations.  It is NOT directly
    comparable to the v0.3 test set (pairdata_v1 ground truth).
    """
    print("\n" + "=" * 80)
    print("Step 4: Creating Fixed Test Set (N=800) — pairdata_v2 ground truth")
    print("=" * 80)

    if len(df) < TEST_SIZE:
        raise ValueError(f"Insufficient data: {len(df)} rows < {TEST_SIZE} required")

    test_df = stratified_sampling(
        df,
        n_samples=TEST_SIZE,
        strata_cols=['component_type', 'damage_type'],
        random_state=RANDOM_SEED
    )
    train_pool_df = df[~df.index.isin(test_df.index)].copy()

    print(f"✓ Test set created  : {len(test_df)} samples (denoised ground truth)")
    print(f"✓ Training pool left: {len(train_pool_df)} samples")

    print(f"\n📊 Test Set Distribution:")
    print("  Component types:")
    for comp, count in test_df['component_type'].value_counts().head(5).items():
        print(f"    - {comp}: {count} ({count/len(test_df)*100:.1f}%)")
    print("  Damage types:")
    for dmg, count in test_df['damage_type'].value_counts().head(5).items():
        print(f"    - {dmg}: {count} ({count/len(test_df)*100:.1f}%)")

    return test_df, train_pool_df


# ============================================================
# Step 5: Create progressive training sets
# ============================================================

def create_progressive_train_sets(train_pool_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Create progressive training sets (1k, 2k, 3k, 4k) with stratified sampling."""
    print("\n" + "=" * 80)
    print("Step 5: Creating Progressive Training Sets")
    print("=" * 80)

    train_sets = {}
    for size in TRAIN_SIZES:
        if len(train_pool_df) < size:
            print(f"⚠️  Insufficient data for {size} samples; using all {len(train_pool_df)} samples.")
            train_sets[size] = train_pool_df.copy()
            continue

        train_df = stratified_sampling(
            train_pool_df,
            n_samples=size,
            strata_cols=['component_type', 'damage_type'],
            random_state=RANDOM_SEED + size
        )
        train_sets[size] = train_df
        print(f"✓ Train set ({size:,}): {len(train_df)} samples")

    return train_sets


# ============================================================
# Step 6: Save datasets and generate report
# ============================================================

def save_datasets(
    test_df: pd.DataFrame,
    train_sets: Dict[int, pd.DataFrame],
    filter_stats: Dict,
    noise_stats: Dict
):
    """Save all datasets as CSV files and write a preparation report."""
    print("\n" + "=" * 80)
    print("Step 6: Saving Datasets")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_path = OUTPUT_DIR / "test_set_n800.csv"
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
    print(f"✓ Saved test set  : {test_path}")

    for size, train_df in train_sets.items():
        train_path = OUTPUT_DIR / f"train_{size//1000}k.csv"
        train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
        print(f"✓ Saved train set : {train_path}")

    report_path = OUTPUT_DIR / "dataset_preparation_report.md"
    generate_report(test_df, train_sets, filter_stats, noise_stats, report_path)
    print(f"✓ Saved report    : {report_path}")


def generate_report(
    test_df: pd.DataFrame,
    train_sets: Dict[int, pd.DataFrame],
    filter_stats: Dict,
    noise_stats: Dict,
    output_path: Path
):
    """Generate dataset preparation report for pairdata_v2."""

    report = f"""# Dataset Preparation Report - v0.7 (pairdata_v2)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

pairdata_v2: Denoised progressive fine-tuning datasets from n=10,789 bridge damage images.
Out-of-scope sentences (temporal comparisons, maintenance recommendations) removed from
ground truth text before training and evaluation splits.

> **Note**: Both training data and test-set ground truth use pairdata_v2 (denoised).
> Direct comparison with v0.3/v0.6 evaluation numbers (pairdata_v1 GT) is not applicable.

---

## Step 0: Noise Removal Statistics (pairdata_v1 → pairdata_v2)

| Metric | Count |
|--------|-------|
| Category A: Temporal sentences removed | {noise_stats['temporal_sentences_removed']:,} |
| Category B: Maintenance sentences removed | {noise_stats['maintenance_sentences_removed']:,} |
| **Total sentences removed** | **{noise_stats['temporal_sentences_removed'] + noise_stats['maintenance_sentences_removed']:,}** |
| Rows with ≥1 sentence removed | {noise_stats['rows_affected']:,} |
| Rows fully emptied (all text removed) | {noise_stats['rows_emptied']:,} |

### Noise Pattern Categories

**Category A — Temporal Comparison** (requires 2+ images/time points):
- `前回(より|から|の|点検|調査)` — previous inspection references
- `前年(より|から|の)` — previous year references
- `経時(的|変化)` — chronological change descriptions
- `新たに.{{0,15}}(発生|出現)` — newly appeared damage (temporal implication)

**Category B — Maintenance Recommendations** (requires bridge metadata):
- `補修(が|を|は)?(必要|要する|望ましい|検討)`
- `予防保全`
- `対策(が|を)?(必要|要する|講じ)`
- `措置(が|を)?(必要|要する)`
- `経過観察(が|を)?(必要|要する)`
- `早(期|急)に.{{0,15}}(補修|対策|措置|点検|確認)`
- `維持管理計画`
- `補修工法|補修方法`

---

## Filtering Statistics (post-denoising, same criteria as v0.3)

| Stage | Count | Removed | Retention |
|-------|-------|---------|-----------|
| Initial | {filter_stats['initial_count']:,} | — | 100.0% |
| After missing/empty | {filter_stats['initial_count'] - filter_stats['filtered_stages']['missing_values']:,} | {filter_stats['filtered_stages']['missing_values']:,} | {((filter_stats['initial_count'] - filter_stats['filtered_stages']['missing_values']) / filter_stats['initial_count'] * 100):.1f}% |
| After length filter | {filter_stats['initial_count'] - filter_stats['filtered_stages']['missing_values'] - filter_stats['filtered_stages']['length_constraint']:,} | {filter_stats['filtered_stages']['length_constraint']:,} | — |
| After pattern filter | {filter_stats['initial_count'] - filter_stats['filtered_stages']['missing_values'] - filter_stats['filtered_stages']['length_constraint'] - filter_stats['filtered_stages']['excluded_patterns']:,} | {filter_stats['filtered_stages']['excluded_patterns']:,} | — |
| **Final (pairdata_v2)** | **{filter_stats['final_count']:,}** | **{filter_stats['total_removed']:,}** | **{filter_stats['retention_rate']:.1f}%** |

---

## Dataset Splits

| Split | Size |
|-------|------|
| **Test Set (denoised GT)** | {len(test_df):,} |
"""
    for size in sorted(train_sets.keys()):
        report += f"| Train Set ({size//1000}k) | {len(train_sets[size]):,} |\n"

    report += f"\n**Random Seed**: {RANDOM_SEED} (for reproducibility)\n\n"

    # Text statistics
    report += "---\n\n## Text Statistics\n\n"
    report += "### Test Set (denoised 所見)\n\n"
    test_lens = test_df['所見'].astype(str).str.len()
    report += f"- Mean length   : {test_lens.mean():.1f} characters\n"
    report += f"- Median length : {test_lens.median():.1f} characters\n"
    report += f"- Std deviation : {test_lens.std():.1f} characters\n"
    report += f"- Min length    : {test_lens.min()} characters\n"
    report += f"- Max length    : {test_lens.max()} characters\n\n"

    report += "### Training Sets\n\n"
    report += "| Set | N | Mean Length | Median Length |\n"
    report += "|-----|---|-------------|---------------|\n"
    for size in sorted(train_sets.keys()):
        train_df = train_sets[size]
        lens = train_df['所見'].astype(str).str.len()
        report += f"| Train {size//1000}k | {len(train_df):,} | {lens.mean():.1f} | {lens.median():.1f} |\n"

    # Files generated
    report += "\n---\n\n## Files Generated\n\n"
    report += f"- `test_set_n800.csv` — Fixed test set ({len(test_df)} samples, denoised GT)\n"
    for size in sorted(train_sets.keys()):
        report += f"- `train_{size//1000}k.csv` — Training set ({len(train_sets[size]):,} samples)\n"
    report += "- `dataset_preparation_report.md` — This report\n"

    report += """
---

## Next Steps

1. **Train v0.7 models**: `python train_v07_qlora.py --train-size 1k` (then 2k, 3k, 4k)
2. **Run inference**: Use `inference_v051_qlora.py` with v0.7 model adapters on `test_set_n800.csv`
3. **Evaluate**: Vector similarity between v0.7 predictions and denoised ground truth
4. **Check mode collapse**: Confirm whether member/damage-type diversity has improved vs v0.6.3
5. **Recalibrate Quality Guard**: Run `analyze_low_quality_text.py` on v0.7 predictions for new θ values
6. **Write Supplementary**: Add results to `methodology.tex` (Supplementary section)
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)


# ============================================================
# Main
# ============================================================

def main():
    """Main execution pipeline for pairdata_v2 generation."""
    print("\n🔧 Bridge Damage VLM Dataset Preparation - v0.7 (pairdata_v2)")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        # Step 1: Load master data
        df = load_master_data()

        # Step 0: Noise removal — pairdata_v1 → pairdata_v2
        #         (applied before all other filters, and to ALL rows
        #          including those that will become the test set)
        df_denoised, noise_stats = apply_noise_removal(df)

        # Step 2: Apply quality filters (post-denoising)
        df_filtered, filter_stats = apply_text_quality_filters(df_denoised)

        # Step 3: Merge v0.2 metadata
        df_merged = merge_v02_metadata(df_filtered)

        # Step 4: Create fixed test set (denoised ground truth)
        test_df, train_pool_df = create_test_set(df_merged)

        # Step 5: Create progressive training sets
        train_sets = create_progressive_train_sets(train_pool_df)

        # Step 6: Save all datasets + report
        save_datasets(test_df, train_sets, filter_stats, noise_stats)

        print("\n" + "=" * 80)
        print("✅ pairdata_v2 Dataset Preparation Complete!")
        print("=" * 80)
        print(f"\n📁 Output directory: {OUTPUT_DIR}")
        print(f"   - test_set_n800.csv  (denoised ground truth)")
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
