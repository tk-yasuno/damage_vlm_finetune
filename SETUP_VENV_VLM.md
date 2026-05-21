# Virtual Environment Setup Guide - v0.3-v0.5

Complete setup instructions for `.venv-vlm` virtual environment for progressive VLM fine-tuning.

---

## Prerequisites

- **Operating System**: Windows 10/11
- **Python**: 3.10 or 3.11 (recommended: 3.10.11)
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 or better recommended)
- **CUDA**: 12.1 or later
- **RAM**: 16GB minimum, 32GB recommended
- **Disk Space**: 50GB+ free (for models and datasets)

---

## Step 1: Create Virtual Environment

Open PowerShell in the project root directory:

```powershell
# Create virtual environment
python -m venv .venv-vlm

# Activate virtual environment
.\.venv-vlm\Scripts\Activate.ps1
```

If you encounter execution policy errors:

```powershell
# Allow script execution (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Step 2: Upgrade pip and Core Tools

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install wheel and setuptools
pip install --upgrade wheel setuptools
```

---

## Step 3: Install PyTorch with CUDA Support

**Important**: Install PyTorch **before** other dependencies to ensure correct CUDA version.

```powershell
# PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA availability:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output:

```
CUDA available: True
CUDA version: 12.1
Device: NVIDIA GeForce RTX 4060 Ti
```

---

## Step 4: Install Core Dependencies

### 4.1 Transformers and PEFT

```powershell
# HuggingFace Transformers (for LLaVA)
pip install transformers>=4.41.0

# PEFT (Parameter-Efficient Fine-Tuning with LoRA)
pip install peft>=0.11.0

# Accelerate (for distributed training)
pip install accelerate>=0.30.0
```

### 4.2 Quantization Support

```powershell
# bitsandbytes (for 4-bit quantization)
pip install bitsandbytes>=0.43.0
```

**Note**: If `bitsandbytes` installation fails on Windows, use the pre-compiled wheel:

```powershell
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.43.0-py3-none-win_amd64.whl
```

### 4.3 Dataset and Data Processing

```powershell
# HuggingFace Datasets
pip install datasets>=2.14.0

# Image processing
pip install Pillow>=10.0.0
pip install opencv-python>=4.8.0

# Data manipulation
pip install pandas>=2.0.0
pip install openpyxl>=3.1.0  # For Excel file support
```

### 4.4 Evaluation and Metrics

```powershell
# Sentence Transformers (for vector similarity)
pip install sentence-transformers>=2.2.0

# SciPy (for distance metrics)
pip install scipy>=1.11.0
```

### 4.5 Visualization and Reporting

```powershell
# Matplotlib and Seaborn
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

### 4.6 Configuration and Utilities

```powershell
# YAML configuration
pip install pyyaml>=6.0

# Progress bars
pip install tqdm>=4.65.0
```

---

## Step 5: Install Project Dependencies from requirements.txt

If `requirements.txt` is available:

```powershell
pip install -r requirements.txt
```

---

## Step 6: Verify Installation

Run the verification script:

```powershell
python -c "
import sys
print('Python version:', sys.version)

try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  CUDA version: {torch.version.cuda}')
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'✗ PyTorch: {e}')

try:
    import transformers
    print(f'✓ Transformers {transformers.__version__}')
except ImportError as e:
    print(f'✗ Transformers: {e}')

try:
    import peft
    print(f'✓ PEFT {peft.__version__}')
except ImportError as e:
    print(f'✗ PEFT: {e}')

try:
    import bitsandbytes
    print(f'✓ bitsandbytes {bitsandbytes.__version__}')
except ImportError as e:
    print(f'✗ bitsandbytes: {e}')

try:
    import accelerate
    print(f'✓ Accelerate {accelerate.__version__}')
except ImportError as e:
    print(f'✗ Accelerate: {e}')

try:
    import datasets
    print(f'✓ Datasets {datasets.__version__}')
except ImportError as e:
    print(f'✗ Datasets: {e}')

try:
    import sentence_transformers
    print(f'✓ Sentence-Transformers {sentence_transformers.__version__}')
except ImportError as e:
    print(f'✗ Sentence-Transformers: {e}')

try:
    import pandas
    print(f'✓ Pandas {pandas.__version__}')
except ImportError as e:
    print(f'✗ Pandas: {e}')

try:
    import matplotlib
    print(f'✓ Matplotlib {matplotlib.__version__}')
except ImportError as e:
    print(f'✗ Matplotlib: {e}')

try:
    import PIL
    print(f'✓ Pillow {PIL.__version__}')
except ImportError as e:
    print(f'✗ Pillow: {e}')

print('\n✅ All core dependencies verified!')
"
```

---

## Step 7: Test Dataset Preparation

```powershell
# Test dataset preparation script
python prepare_v03_dataset.py
```

Expected output:

- Dataset filtering statistics
- Train/test split confirmation
- CSV files in `data/v03_fine_tuning/`

---

## Step 8: Test Vector Similarity Evaluator

```powershell
# Test vector similarity module
python -c "from src.evaluation import VectorSimilarityEvaluator; print('✓ Vector Similarity Evaluator imported successfully')"
```

---

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory

**Solution**: Reduce batch size in training config

```yaml
# In config.yaml
v04_fine_tuning:
  training:
    batch_size: 2  # Reduce from 4 to 2
    gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

### Issue 2: bitsandbytes Installation Fails

**Solution**: Use pre-compiled Windows wheel

```powershell
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.43.0-py3-none-win_amd64.whl
```

### Issue 3: Transformers Version Conflict

**Solution**: Ensure clean installation

```powershell
pip uninstall transformers -y
pip install transformers==4.41.0
```

### Issue 4: CUDA Version Mismatch

**Solution**: Reinstall PyTorch with correct CUDA version

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue 5: Sentence-Transformers Model Download Fails

**Solution**: Set HuggingFace cache directory

```powershell
$env:HF_HOME = "I:/ACT2025.5.26-2030/MVP/.cache/huggingface"
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens-v2')"
```

---

## Environment Variables

Set these for optimal performance:

```powershell
# HuggingFace cache
$env:HF_HOME = "I:/ACT2025.5.26-2030/MVP/.cache/huggingface"

# CUDA settings
$env:CUDA_LAUNCH_BLOCKING = "1"  # For debugging only (slower)

# PyTorch settings
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
```

To make permanent (add to PowerShell profile):

```powershell
# Edit profile
notepad $PROFILE

# Add these lines:
$env:HF_HOME = "I:/ACT2025.5.26-2030/MVP/.cache/huggingface"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
```

---

## Next Steps

After environment setup:

1. **Prepare Dataset** (v0.3):

   ```powershell
   python prepare_v03_dataset.py
   ```
2. **Train First Model** (v0.4.1):

   ```powershell
   python train_v03_qlora.py --train-data data/v03_fine_tuning/train_1k.csv --output-dir models/llava_v03_qlora_1k
   ```
3. **Evaluate Model**:

   ```powershell
   python -m src.evaluation.vector_similarity_evaluator --csv inference_1k_results.csv --output evaluation_1k.json
   ```
4. **Generate Report**:

   ```powershell
   python create_progressive_training_report.py
   ```

---

## Deactivating Environment

When done:

```powershell
deactivate
```

---

## Backup Requirements

Save current environment for reproducibility:

```powershell
pip freeze > requirements_venv_vlm.txt
```

---

## Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Sentence-Transformers](https://www.sbert.net/)
- [LLaVA Model Card](https://huggingface.co/llava-hf/llava-1.5-7b-hf)

---

**Last Updated**: 2026-05-21
