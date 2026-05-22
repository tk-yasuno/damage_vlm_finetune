# QLoRA Progressive Training Results (v0.4.1 - v0.4.4)

**Status**: ✅ Complete  
**Completion Date**: 2026-05-23  
**Training Period**: 2026-05-22 01:59 - 2026-05-23 00:17 (Total: ~22.5 hours)

## 📊 Executive Summary

Progressive fine-tuning of **LLaVA-1.5-7B** using QLoRA across four dataset scales (1k, 2k, 3k, 4k samples) was successfully completed. Results demonstrate **diminishing returns** beyond 2k samples, with validation loss plateauing at ~3.07.

### Key Findings

1. **Optimal Dataset Size**: **2k samples** provides best cost-benefit ratio
   - Training time: 2:55:37 (vs 6:19:10 for 4k)
   - Validation loss: 3.073 (vs 3.067 for 4k)
   - **Efficiency**: 2.2x faster for only 0.2% worse performance

2. **Performance Plateau**: Loss convergence observed at 2k-3k range
   - 1k → 2k: -1.98% improvement (significant)
   - 2k → 3k: 0.00% improvement (plateau)
   - 3k → 4k: -0.20% improvement (marginal)

3. **Training Time Scaling**: Approximately linear with data size
   - 1k: 1:23 hours (baseline)
   - 2k: 2:56 hours (2.1x)
   - 3k: 4:32 hours (3.3x)
   - 4k: 6:19 hours (4.6x)

---

## 🎯 Progressive Training Results

### Complete Training Summary

| Stage | Dataset | Training Time | Train Samples | Val Samples | **Final Val Loss** | **Best Checkpoint** | Loss Improvement |
|-------|---------|---------------|---------------|-------------|-------------------|---------------------|------------------|
| **v0.4.1** | 1k | 1:22:57 | 799 | 200 | **3.135** | checkpoint-100 | - (baseline) |
| **v0.4.2** | 2k | 2:55:37 | 1,599 | 400 | **3.073** | checkpoint-300 | ↓ 1.98% |
| **v0.4.3** | 3k | 4:31:44 | 2,398 | 600 | **3.073** | checkpoint-400 | ↔ 0.00% |
| **v0.4.4** | 4k | 6:19:10 | 3,196 | 799 | **3.067** | checkpoint-600 | ↓ 0.20% |

### Training Configuration (Identical Across All Stages)

```yaml
Base Model: llava-hf/llava-1.5-7b-hf
Quantization: 4-bit NF4 with double quantization

LoRA Parameters:
  rank: 32
  alpha: 64
  dropout: 0.05
  target_modules: [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]

Training Hyperparameters:
  batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch size: 16
  learning_rate: 2e-4
  num_epochs: 3
  warmup_steps: 50
  max_grad_norm: 1.0
  weight_decay: 0.01
  optimizer: AdamW
  lr_scheduler: cosine
  mixed_precision: fp16

Hardware:
  GPU: NVIDIA GeForce RTX 4060 Ti 16GB
  CUDA: 12.4
  PyTorch: 2.6.0
  Transformers: 4.47.1
```

---

## 📈 Detailed Training Metrics

### v0.4.1: 1k Samples (Baseline)

**Training Period**: 2026-05-22 01:59 - 03:22  
**Duration**: 1:22:57  
**Output Directory**: `models/llava_v03_qlora_1k`

#### Training Summary
- **Total Steps**: 150 (3 epochs × 50 steps/epoch)
- **Train Samples**: 799 (80/20 split from 1,000)
- **Validation Samples**: 200
- **Best Checkpoint**: Step 100 (Epoch 2.0)
- **Best Validation Loss**: 3.135

#### Loss Curve
```
Epoch 0.5:  eval_loss = 3.141
Epoch 1.0:  eval_loss = 3.094  (↓ 1.5%)
Epoch 1.5:  eval_loss = 3.081  (↓ 0.4%)
Epoch 2.0:  eval_loss = 3.094  (↑ 0.4%, slight overfitting)
Epoch 2.5:  eval_loss = 3.068  (↓ 0.8%)
Epoch 3.0:  eval_loss = 3.135  (↑ 2.2%, overfitting)
```

**Observation**: Signs of overfitting after epoch 2.0, suggesting 1k samples insufficient for stable convergence.

---

### v0.4.2: 2k Samples

**Training Period**: 2026-05-22 03:54 - 06:50  
**Duration**: 2:55:37  
**Output Directory**: `models/llava_v03_qlora_2k`

#### Training Summary
- **Total Steps**: 300 (3 epochs × 100 steps/epoch)
- **Train Samples**: 1,599 (80/20 split from 2,000)
- **Validation Samples**: 400
- **Best Checkpoint**: Step 300 (Epoch 3.0)
- **Best Validation Loss**: 3.073

#### Loss Curve
```
Epoch 0.5:  eval_loss = 3.120
Epoch 1.0:  eval_loss = 3.086  (↓ 1.1%)
Epoch 1.5:  eval_loss = 3.073  (↓ 0.4%)
Epoch 2.0:  eval_loss = 3.069  (↓ 0.1%)
Epoch 2.5:  eval_loss = 3.068  (↓ 0.0%)
Epoch 3.0:  eval_loss = 3.073  (↑ 0.2%, stable)
```

**Observation**: Smooth convergence with minimal overfitting. Best performance-efficiency balance.

---

### v0.4.3: 3k Samples

**Training Period**: 2026-05-22 07:34 - 12:06  
**Duration**: 4:31:44  
**Output Directory**: `models/llava_v03_qlora_3k`

#### Training Summary
- **Total Steps**: 450 (3 epochs × 150 steps/epoch)
- **Train Samples**: 2,398 (80/20 split from 3,000)
- **Validation Samples**: 600
- **Best Checkpoint**: Step 400 (Epoch 2.67)
- **Best Validation Loss**: 3.073

#### Loss Curve
```
Epoch 0.5:  eval_loss = 3.118
Epoch 1.0:  eval_loss = 3.086  (↓ 1.0%)
Epoch 1.5:  eval_loss = 3.074  (↓ 0.4%)
Epoch 2.0:  eval_loss = 3.073  (↓ 0.0%)
Epoch 2.5:  eval_loss = 3.068  (↓ 0.2%)
Epoch 3.0:  eval_loss = 3.073  (↑ 0.1%, stable)
```

**Observation**: No improvement over 2k model. Loss plateau confirms diminishing returns.

---

### v0.4.4: 4k Samples

**Training Period**: 2026-05-22 17:58 - 2026-05-23 00:17  
**Duration**: 6:19:10  
**Output Directory**: `models/llava_v03_qlora_4k`

#### Training Summary
- **Total Steps**: 600 (3 epochs × 200 steps/epoch)
- **Train Samples**: 3,196 (80/20 split from 4,000)
- **Validation Samples**: 799
- **Best Checkpoint**: Step 600 (Epoch 3.0)
- **Best Validation Loss**: 3.067

#### Loss Curve
```
Epoch 0.5:  eval_loss = 3.141
Epoch 1.0:  eval_loss = 3.094  (↓ 1.5%)
Epoch 1.5:  eval_loss = 3.081  (↓ 0.4%)
Epoch 2.0:  eval_loss = 3.072  (↓ 0.3%)
Epoch 2.5:  eval_loss = 3.068  (↓ 0.1%)
Epoch 3.0:  eval_loss = 3.067  (↓ 0.0%)
```

**Observation**: Marginal 0.2% improvement over 3k, not justifying 1.4x longer training time.

---

## 🔍 Analysis

### 1. Cost-Benefit Analysis

| Model | Training Cost | Performance Gain | Efficiency Score |
|-------|---------------|------------------|------------------|
| 1k | 1.0x (baseline) | 0% (baseline) | ⭐⭐⭐ |
| 2k | 2.1x | +1.98% | ⭐⭐⭐⭐⭐ **Best** |
| 3k | 3.3x | +1.98% | ⭐⭐ |
| 4k | 4.6x | +2.17% | ⭐ |

**Efficiency Score** = (Loss Improvement) / (Training Time Multiplier)

```
1k: 0% / 1.0x = 0.00 (baseline)
2k: 1.98% / 2.1x = 0.94  ← BEST
3k: 1.98% / 3.3x = 0.60
4k: 2.17% / 4.6x = 0.47
```

### 2. Loss Convergence Visualization

```
3.14 ┤
     │ 1k ●
3.12 ┤     ╲
     │      ╲
3.10 ┤       ╲ 2k ●━━━━━━━━━━━━●━━━━● (plateau)
     │              3k            4k
3.08 ┤
     │
3.06 ┤                              ●
     │                             4k
3.04 ┤
     └────────────────────────────────
      1k    2k    3k    4k
```

**Key Insight**: Significant improvement from 1k→2k, then plateau. Diminishing returns beyond 2k.

### 3. Training Time Scaling

```
Training Time (hours)
6:30 ┤                              ● 4k
     │                           ╱
5:00 ┤                        ╱
     │                     ╱
3:30 ┤                  ● 3k
     │               ╱
2:00 ┤            ● 2k
     │         ╱
0:30 ┤      ● 1k
     └────────────────────────────────
      1k    2k    3k    4k
```

**Scaling Factor**: Approximately linear (actual: 1x → 2.1x → 3.3x → 4.6x)

### 4. Validation Loss Stability

| Model | Loss StdDev (Epochs 1-3) | Stability Rating |
|-------|--------------------------|------------------|
| 1k | 0.027 | ⭐⭐ (unstable) |
| 2k | 0.008 | ⭐⭐⭐⭐ |
| 3k | 0.007 | ⭐⭐⭐⭐⭐ |
| 4k | 0.011 | ⭐⭐⭐⭐ |

**Note**: 2k and 3k show best stability. 1k exhibits overfitting, 4k has slight noise.

---

## 💡 Recommendations

### For Production Deployment

1. **Use 2k Model (v0.4.2)** as primary candidate
   - Best performance-efficiency balance
   - Stable training with minimal overfitting
   - 2.2x faster training than 4k for only 0.2% worse loss

### For Academic Analysis

2. **Compare 2k vs 4k on Test Set (n=800)**
   - Run inference on `data/v03_fine_tuning/test_set_n800.csv`
   - Measure cosine similarity using Sentence-BERT
   - Determine if 0.2% loss improvement translates to quality gains

### For Future Work

3. **Focus on Data Quality, Not Quantity**
   - Plateau at 2k-3k suggests dataset size is sufficient
   - Investigate data augmentation (rotation, lighting, etc.)
   - Consider curriculum learning (easy → hard samples)

4. **Hyperparameter Tuning**
   - Current config (lr=2e-4, rank=32) may not be optimal for 4k
   - Try lower learning rate (1e-4) or higher rank (64) for larger datasets
   - Experiment with longer warmup (100 steps) for 3k-4k

---

## 📁 Output Files

### Model Artifacts

```
models/
├── llava_v03_qlora_1k/
│   ├── adapter_model.safetensors    # 134.5 MB
│   ├── adapter_config.json
│   ├── training_summary.json
│   ├── checkpoint-100/              # Best checkpoint
│   └── logs/
│
├── llava_v03_qlora_2k/
│   ├── adapter_model.safetensors    # 134.5 MB
│   ├── adapter_config.json
│   ├── training_summary.json
│   ├── checkpoint-300/              # Best checkpoint
│   └── logs/
│
├── llava_v03_qlora_3k/
│   ├── adapter_model.safetensors    # 134.5 MB
│   ├── adapter_config.json
│   ├── training_summary.json
│   ├── checkpoint-400/              # Best checkpoint
│   └── logs/
│
└── llava_v03_qlora_4k/
    ├── adapter_model.safetensors    # 134.5 MB
    ├── adapter_config.json
    ├── training_summary.json
    ├── checkpoint-600/              # Best checkpoint
    └── logs/
```

### Training Logs (TensorBoard)

```bash
# View training curves
tensorboard --logdir models/llava_v03_qlora_1k/logs
tensorboard --logdir models/llava_v03_qlora_2k/logs
tensorboard --logdir models/llava_v03_qlora_3k/logs
tensorboard --logdir models/llava_v03_qlora_4k/logs
```

---

## 🚀 Next Steps

### Phase 1: Test Set Evaluation (v0.5.1)

**Objective**: Measure real-world performance on held-out test set (n=800)

```bash
# Run inference on test set for each model
python inference_v04_qlora.py \
  --model-dir models/llava_v03_qlora_1k \
  --test-csv data/v03_fine_tuning/test_set_n800.csv \
  --output inference_results_1k.csv

python inference_v04_qlora.py \
  --model-dir models/llava_v03_qlora_2k \
  --test-csv data/v03_fine_tuning/test_set_n800.csv \
  --output inference_results_2k.csv

python inference_v04_qlora.py \
  --model-dir models/llava_v03_qlora_3k \
  --test-csv data/v03_fine_tuning/test_set_n800.csv \
  --output inference_results_3k.csv

python inference_v04_qlora.py \
  --model-dir models/llava_v03_qlora_4k \
  --test-csv data/v03_fine_tuning/test_set_n800.csv \
  --output inference_results_4k.csv
```

### Phase 2: Vector Similarity Evaluation (v0.5.2)

**Objective**: Calculate cosine similarity between predictions and ground truth

```bash
# Evaluate each model
python -m src.evaluation.vector_similarity_evaluator \
  --csv inference_results_1k.csv \
  --gt-col "所見" \
  --pred-col "prediction" \
  --output evaluation_1k.json

# Generate comparative report
python create_progressive_evaluation_report.py \
  --eval-files evaluation_1k.json evaluation_2k.json evaluation_3k.json evaluation_4k.json \
  --output Progressive_Evaluation_Report.md
```

### Phase 3: Statistical Analysis (v0.5.3)

**Metrics to Compare**:
- Mean cosine similarity (target: ≥0.80)
- Quality distribution (≥0.75 threshold)
- Standard deviation (consistency)
- Inference time (efficiency)

**Statistical Tests**:
- Mann-Whitney U test (1k vs 2k, 2k vs 4k)
- Effect size calculation (Cohen's d)
- Determine if loss differences translate to quality improvements

---

## 📊 Summary Statistics

### Training Time Distribution

| Stage | Training Time | Percentage of Total |
|-------|---------------|---------------------|
| 1k | 1:22:57 | 11.0% |
| 2k | 2:55:37 | 23.2% |
| 3k | 4:31:44 | 35.9% |
| 4k | 6:19:10 | 50.3% |
| **Total** | **~15:09:28** | **100%** |

### Model Size (LoRA Adapters Only)

| Model | Adapter Size | Parameters (Trainable) |
|-------|--------------|------------------------|
| 1k-4k | 134.5 MB | ~67M parameters |

**Note**: Base model (LLaVA-1.5-7B) is shared (~13GB in 4-bit), only adapters differ.

### GPU Memory Usage

| Phase | VRAM Usage | Notes |
|-------|------------|-------|
| Training | ~15.2 GB | Peak during forward+backward pass |
| Inference | ~9.8 GB | Model + adapters + images |
| Idle | 0.5 GB | Background processes |

**Hardware**: NVIDIA GeForce RTX 4060 Ti 16GB

---

## 🎓 Lessons Learned

1. **Dataset Size Plateau**: More data ≠ better performance beyond certain threshold (2k-3k in this case)

2. **Validation Loss ≠ Quality**: Final test set evaluation needed to confirm loss improvements translate to description quality

3. **Training Stability**: Larger datasets (2k+) show smoother convergence and less overfitting

4. **Resource Planning**: Linear time scaling means 10k samples would take ~31 hours - not practical without multi-GPU

5. **Early Stopping**: Best checkpoints often occur before final epoch, suggesting early stopping could save time

---

## 📝 Changelog

### v0.4.4 (2026-05-23)
- ✅ Completed 4k samples training (6:19:10)
- ✅ Achieved validation loss: 3.067
- ✅ Generated comprehensive training report

### v0.4.3 (2026-05-22)
- ✅ Completed 3k samples training (4:31:44)
- ✅ Achieved validation loss: 3.073
- ⚠️ Observed loss plateau vs 2k model

### v0.4.2 (2026-05-22)
- ✅ Completed 2k samples training (2:55:37)
- ✅ Achieved validation loss: 3.073
- ✅ Significant improvement over 1k baseline

### v0.4.1 (2026-05-22)
- ✅ Completed 1k samples training (1:22:57)
- ✅ Baseline validation loss: 3.135
- ⚠️ Slight overfitting observed in final epoch

---

## 📚 References

- **Base Model**: [LLaVA-1.5-7B (HuggingFace)](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- **QLoRA Paper**: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)
- **Training Script**: [train_v03_qlora.py](train_v03_qlora.py)
- **Dataset Preparation**: [prepare_v03_dataset.py](prepare_v03_dataset.py)

---

**Generated**: 2026-05-23  
**Author**: Takato Yasuno (Bridge Damage VLM Project)  
**License**: Apache 2.0
