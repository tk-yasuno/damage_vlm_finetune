# Dataset Preparation Report - v0.3
Generated: 2026-05-22 00:56:54

## Overview

Prepared progressive fine-tuning datasets from n=10,789 bridge damage images.

## Filtering Statistics

| Stage | Count | Removed | Retention |
|-------|-------|---------|-----------|
| Initial | 10,903 | - | 100.0% |
| After missing values | 10,346 | 557 | 94.9% |
| After length filter | 10,307 | 39 | - |
| After pattern filter | 10,307 | 0 | - |
| **Final** | **9,880** | **1,023** | **90.6%** |

### Filtering Criteria

- **Text Length**: 15-500 characters
- **Excluded Patterns**: Empty, "なし", "???", etc.
- **Required Content**: Must contain damage/component keywords

## Dataset Splits

| Split | Size | Percentage |
|-------|------|------------|
| **Test Set** | 800 | 16.7% |
| Train Set (1k) | 1,000 | 55.6% |
| Train Set (2k) | 2,000 | 71.4% |
| Train Set (3k) | 3,000 | 78.9% |
| Train Set (4k) | 4,000 | 83.3% |

**Random Seed**: 42 (for reproducibility)

## Test Set Distribution (N=800)

### Component Types

- **unknown**: 800 (100.0%)

### Damage Types

- **unknown**: 800 (100.0%)

## Text Statistics

### Test Set (所見 field)

- **Mean length**: 93.7 characters
- **Median length**: 86.0 characters
- **Std deviation**: 47.9 characters
- **Min length**: 15 characters
- **Max length**: 291 characters

### Training Sets Comparison

| Set | Mean Length | Median Length | Component Types | Damage Types |
|-----|-------------|---------------|-----------------|---------------|
| Train 1k | 93.6 | 85.0 | 1 | 1 |
| Train 2k | 94.4 | 87.0 | 1 | 1 |
| Train 3k | 94.0 | 86.0 | 1 | 1 |
| Train 4k | 93.7 | 85.0 | 1 | 1 |

## Stratification Strategy

- **Method**: Stratified sampling by component_type × damage_type
- **Purpose**: Maintain balanced distribution across damage scenarios
- **Test Set**: Fixed with seed=42 for reproducibility
- **Training Sets**: Progressive scaling (1k → 2k → 3k → 4k)

## Files Generated

- `test_set_n800.csv`: Fixed test set (800 samples)
- `train_1k.csv`: Training set (1000 samples)
- `train_2k.csv`: Training set (2000 samples)
- `train_3k.csv`: Training set (3000 samples)
- `train_4k.csv`: Training set (4000 samples)

## Next Steps

1. **Verify distributions**: Check that stratification maintained component/damage balance
2. **Run v0.4.1**: Fine-tune with train_1k.csv
3. **Evaluate**: Test on test_set_n800.csv using Vector Similarity
4. **Scale up**: Progress to 2k → 3k → 4k training sets
5. **Compare results**: Analyze performance scaling with dataset size
