# Dataset Preparation Report - v0.7 (pairdata_v2)
Generated: 2026-05-25 13:33:12

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
| Category A: Temporal sentences removed | 2,124 |
| Category B: Maintenance sentences removed | 3,022 |
| **Total sentences removed** | **5,146** |
| Rows with ≥1 sentence removed | 4,378 |
| Rows fully emptied (all text removed) | 31 |

### Noise Pattern Categories

**Category A — Temporal Comparison** (requires 2+ images/time points):
- `前回(より|から|の|点検|調査)` — previous inspection references
- `前年(より|から|の)` — previous year references
- `経時(的|変化)` — chronological change descriptions
- `新たに.{0,15}(発生|出現)` — newly appeared damage (temporal implication)

**Category B — Maintenance Recommendations** (requires bridge metadata):
- `補修(が|を|は)?(必要|要する|望ましい|検討)`
- `予防保全`
- `対策(が|を)?(必要|要する|講じ)`
- `措置(が|を)?(必要|要する)`
- `経過観察(が|を)?(必要|要する)`
- `早(期|急)に.{0,15}(補修|対策|措置|点検|確認)`
- `維持管理計画`
- `補修工法|補修方法`

---

## Filtering Statistics (post-denoising, same criteria as v0.3)

| Stage | Count | Removed | Retention |
|-------|-------|---------|-----------|
| Initial | 10,903 | — | 100.0% |
| After missing/empty | 10,315 | 588 | 94.6% |
| After length filter | 10,210 | 105 | — |
| After pattern filter | 10,210 | 0 | — |
| **Final (pairdata_v2)** | **9,785** | **1,118** | **89.7%** |

---

## Dataset Splits

| Split | Size |
|-------|------|
| **Test Set (denoised GT)** | 800 |
| Train Set (1k) | 1,000 |
| Train Set (2k) | 2,000 |
| Train Set (3k) | 3,000 |
| Train Set (4k) | 4,000 |

**Random Seed**: 42 (for reproducibility)

---

## Text Statistics

### Test Set (denoised 所見)

- Mean length   : 78.5 characters
- Median length : 71.0 characters
- Std deviation : 42.7 characters
- Min length    : 15 characters
- Max length    : 241 characters

### Training Sets

| Set | N | Mean Length | Median Length |
|-----|---|-------------|---------------|
| Train 1k | 1,000 | 78.5 | 73.0 |
| Train 2k | 2,000 | 78.3 | 72.0 |
| Train 3k | 3,000 | 78.3 | 72.5 |
| Train 4k | 4,000 | 79.1 | 73.0 |

---

## Files Generated

- `test_set_n800.csv` — Fixed test set (800 samples, denoised GT)
- `train_1k.csv` — Training set (1,000 samples)
- `train_2k.csv` — Training set (2,000 samples)
- `train_3k.csv` — Training set (3,000 samples)
- `train_4k.csv` — Training set (4,000 samples)
- `dataset_preparation_report.md` — This report

---

## Next Steps

1. **Train v0.7 models**: `python train_v07_qlora.py --train-size 1k` (then 2k, 3k, 4k)
2. **Run inference**: Use `inference_v051_qlora.py` with v0.7 model adapters on `test_set_n800.csv`
3. **Evaluate**: Vector similarity between v0.7 predictions and denoised ground truth
4. **Check mode collapse**: Confirm whether member/damage-type diversity has improved vs v0.6.3
5. **Recalibrate Quality Guard**: Run `analyze_low_quality_text.py` on v0.7 predictions for new θ values
6. **Write Supplementary**: Add results to `methodology.tex` (Supplementary section)
