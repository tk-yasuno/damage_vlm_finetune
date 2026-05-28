# Quick Start Guide: v0.3-v0.5 Progressive Fine-Tuning

Complete workflow for progressive VLM fine-tuning from dataset preparation to paper writing.

---

## Prerequisites

1. **Virtual Environment**: Follow [SETUP_VENV_VLM.md](SETUP_VENV_VLM.md)
2. **Data**: Ensure `Rank_c_image_text_n10789.xlsx` is in `data/image_text_inspect_n10789/`
3. **Images**: 10,789 images in `data/image_text_inspect_n10789/rank_c_images_n10789/`

---

## Phase 1: v0.3 Dataset Preparation

### Step 1: Prepare Train/Test Datasets

```powershell
# Activate virtual environment (training)
.\.venv-train\Scripts\Activate.ps1

# Run dataset preparation
python prepare_v03_dataset.py
```

**Output Files:**
- `data/v03_fine_tuning/test_set_n800.csv` (fixed test set)
- `data/v03_fine_tuning/train_1k.csv` (1,000 samples)
- `data/v03_fine_tuning/train_2k.csv` (2,000 samples)
- `data/v03_fine_tuning/train_3k.csv` (3,000 samples)
- `data/v03_fine_tuning/train_4k.csv` (4,000 samples)
- `data/v03_fine_tuning/dataset_preparation_report.md` (statistics)

**Expected Output:**
```
================================================================================
Step 1: Loading Master Data
================================================================================
📁 Loading: Rank_c_image_text_n10789.xlsx
✓ Loaded 10789 rows
  Columns: ファイルパス, 所見, ...

================================================================================
Step 2: Applying Text Quality Filters
================================================================================
✓ Filter 1 (Missing values): Removed X rows
✓ Filter 2 (Length 15-500 chars): Removed X rows
✓ Filter 3 (Excluded patterns): Removed X rows
✓ Filter 4 (Required keywords): Removed X rows

📊 Filtering Summary:
  Initial: 10,789 rows
  Final: X rows
  Retention: XX.X%

================================================================================
Step 4: Creating Fixed Test Set (N=800)
================================================================================
✓ Test set created: 800 samples
✓ Training pool remaining: X samples

================================================================================
Step 5: Creating Progressive Training Sets
================================================================================
✓ Train set (1000): 1000 samples
✓ Train set (2000): 2000 samples
✓ Train set (3000): 3000 samples
✓ Train set (4000): 4000 samples

✅ Dataset Preparation Complete!
```

### Step 2: Verify Dataset

```powershell
# Check CSV files
Get-ChildItem data\v03_fine_tuning\*.csv

# View report
notepad data\v03_fine_tuning\dataset_preparation_report.md
```

---

## Phase 2: v0.4 Progressive Fine-Tuning

### Step 3: Train v0.4.1 (1k samples)

```powershell
# Train on 1,000 samples
python train_v03_qlora.py `
  --train-data data/v03_fine_tuning/train_1k.csv `
  --output-dir models/llava_v03_qlora_1k `
  --epochs 3 `
  --lr 2e-4 `
  --batch-size 4
```

**Expected Duration:** ~2-4 hours (RTX 4060 Ti 16GB)

**Output:**
- `models/llava_v03_qlora_1k/` (LoRA adapters)
- `models/llava_v03_qlora_1k/training_config.json`
- `models/llava_v03_qlora_1k/training_summary.json`
- `models/llava_v03_qlora_1k/logs/` (TensorBoard logs)

### Step 4: Run Inference on Test Set (v0.4.1)

```powershell
# Modify run_inference_n10789.py to load 1k model and test set
# TODO: Create dedicated inference script for test set

# For now, manually run inference with 1k model on test_set_n800.csv
# Save results as inference_1k_results.csv
```

### Step 5: Evaluate v0.4.1

```powershell
# Run vector similarity evaluation
python -m src.evaluation.vector_similarity_evaluator `
  --csv inference_1k_results.csv `
  --gt-col "所見" `
  --pred-col "prediction" `
  --id-col "filename" `
  --output data/v03_fine_tuning/evaluations/evaluation_train_1k.json
```

**Expected Output:**
```
✓ sentence-transformers library found
🔧 Initializing Vector Similarity Evaluator
   Model: sonoisa/sentence-bert-base-ja-mean-tokens-v2
   Device: cuda
✓ Model loaded successfully

📁 Loading data from: inference_1k_results.csv
✓ Loaded 800 rows

📊 Evaluating 800 predictions...
Computing ground truth embeddings...
Computing prediction embeddings...
Computing similarity metrics...

✅ Evaluation Complete!

📈 Overall Metrics:
   Cosine Similarity: 0.XXXX ± 0.XXXX
   Median: 0.XXXX
   Range: [0.XXXX, 0.XXXX]

📊 Quality Distribution:
   Excellent: XXX (XX.X%)
   Good: XXX (XX.X%)
   Acceptable: XXX (XX.X%)
   Poor: XXX (XX.X%)
   Very_poor: XXX (XX.X%)

💾 Results saved to: data/v03_fine_tuning/evaluations/evaluation_train_1k.json
```

### Step 6: Repeat for 2k, 3k, 4k

```powershell
# v0.4.2: Train on 2k samples
python train_v03_qlora.py `
  --train-data data/v03_fine_tuning/train_2k.csv `
  --output-dir models/llava_v03_qlora_2k

# Inference + Evaluation
# ... (repeat Step 4-5)

# v0.4.3: Train on 3k samples
python train_v03_qlora.py `
  --train-data data/v03_fine_tuning/train_3k.csv `
  --output-dir models/llava_v03_qlora_3k

# Inference + Evaluation
# ... (repeat Step 4-5)

# v0.4.4: Train on 4k samples
python train_v03_qlora.py `
  --train-data data/v03_fine_tuning/train_4k.csv `
  --output-dir models/llava_v03_qlora_4k

# Inference + Evaluation
# ... (repeat Step 4-5)
```

---

## Phase 3: v0.5 Evaluation and Reporting

### Step 7: Generate Progressive Training Report

```powershell
# Generate comprehensive comparison report
python create_progressive_training_report.py `
  --eval-dir data/v03_fine_tuning/evaluations `
  --output-dir data/v03_fine_tuning/reports `
  --stages train_1k train_2k train_3k train_4k
```

**Output:**
- `data/v03_fine_tuning/reports/progressive_training_report.md`
- `data/v03_fine_tuning/reports/figures/similarity_distribution.png`
- `data/v03_fine_tuning/reports/figures/scaling_performance.png`
- `data/v03_fine_tuning/reports/figures/quality_distribution_comparison.png`

### Step 8: Review Results

```powershell
# Open report
notepad data\v03_fine_tuning\reports\progressive_training_report.md

# View figures
start data\v03_fine_tuning\reports\figures\
```

**Key Metrics to Check:**
1. **Mean Cosine Similarity**: Should increase with dataset size
2. **Standard Deviation**: Should decrease (more stable)
3. **Quality Distribution**: More samples in "Excellent" and "Good" categories
4. **Improvement over Baseline**: 1k vs 4k performance gain

---

## Phase 4: v0.6 Paper Writing

### Step 9: Prepare Figures for Paper

```powershell
# Copy figures to paper directory
Copy-Item data\v03_fine_tuning\reports\figures\*.png `
  -Destination paper_damage_vlm\0_Format_arXiv_pdfLaTex\figures\
```

### Step 10: Update LaTeX Methodology Section

Edit `paper_damage_vlm/0_Format_arXiv_pdfLaTex/bridge_damage_vlm_quantization_2026.tex`:

**Content to Add:**
1. **Dataset Construction**: Filtering criteria, stratified sampling
2. **LoRA Configuration**: Hyperparameters (r=32, alpha=64, dropout=0.05)
3. **Progressive Scaling**: 1k → 2k → 3k → 4k training strategy
4. **Evaluation Metrics**: Vector Similarity methodology
5. **Results**: Performance scaling curves, quality distributions
6. **Discussion**: Bridge damage VLM practical utility

### Step 11: Compile LaTeX

```powershell
cd paper_damage_vlm\0_Format_arXiv_pdfLaTex

# Compile (may need LaTeX installation)
pdflatex bridge_damage_vlm_quantization_2026.tex
bibtex bridge_damage_vlm_quantization_2026
pdflatex bridge_damage_vlm_quantization_2026.tex
pdflatex bridge_damage_vlm_quantization_2026.tex

# Open PDF
start bridge_damage_vlm_quantization_2026.pdf
```

---

## Troubleshooting

### Issue: Dataset preparation fails

```powershell
# Check master Excel file exists
Test-Path data\image_text_inspect_n10789\Rank_c_image_text_n10789.xlsx

# Check images directory
Test-Path data\image_text_inspect_n10789\rank_c_images_n10789\
```

### Issue: Training CUDA out of memory

```powershell
# Reduce batch size
python train_v03_qlora.py `
  --train-data data/v03_fine_tuning/train_1k.csv `
  --output-dir models/llava_v03_qlora_1k `
  --batch-size 2  # Reduced from 4
```

### Issue: Evaluation fails

```powershell
# Check inference results exist
Test-Path inference_1k_results.csv

# Check required columns
python -c "import pandas as pd; df = pd.read_csv('inference_1k_results.csv'); print(df.columns.tolist())"
```

### Issue: Report generation fails

```powershell
# Check evaluation JSON files exist
Get-ChildItem data\v03_fine_tuning\evaluations\evaluation_*.json

# Install plotting libraries if missing
pip install matplotlib seaborn
```

---

## Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| **v0.3** | 30 min | Dataset preparation |
| **v0.4.1** | 2-4 hours | Train 1k |
| **v0.4.2** | 4-8 hours | Train 2k |
| **v0.4.3** | 6-12 hours | Train 3k |
| **v0.4.4** | 8-16 hours | Train 4k |
| **v0.5** | 1-2 hours | Evaluation + Report |
| **v0.6** | 2-4 hours | Paper writing |
| **Total** | ~3-5 days | End-to-end |

---

## Monitoring Training

### TensorBoard

```powershell
# Start TensorBoard
tensorboard --logdir models/llava_v03_qlora_1k/logs --port 6006

# Open in browser
start http://localhost:6006
```

### GPU Usage

```powershell
# Monitor GPU (requires NVIDIA drivers)
nvidia-smi -l 1
```

---

## Final Checklist

- [ ] Dataset prepared (5 CSV files + report)
- [ ] 4 models trained (1k, 2k, 3k, 4k)
- [ ] 4 evaluations complete (JSON files)
- [ ] Progressive training report generated
- [ ] Figures copied to paper directory
- [ ] Methodology section written
- [ ] Paper compiled to PDF

---

## Next Steps After v0.6

1. **Deploy Best Model**: Integrate into v0.2 pipeline
2. **Full System Evaluation**: Run end-to-end damage assessment on n=254 test set
3. **User Acceptance Testing**: Validate with domain experts
4. **Submit to arXiv**: Publish preprint
5. **Conference Submission**: Target civil engineering AI conferences

---

**Last Updated**: 2026-05-21
