"""
v0.7 QLoRA Fine-Tuning: Progressive Training Script for LLaVA-1.5-7B
Trains on pairdata_v2 (denoised ground truth) at 1k/2k/3k/4k scales.

Usage:
    python train_v07_qlora.py --train-size 1k
    python train_v07_qlora.py --train-size 2k
    python train_v07_qlora.py --train-size 3k
    python train_v07_qlora.py --train-size 4k
    # or explicit paths:
    python train_v07_qlora.py --train-data data/v07_fine_tuning/train_2k.csv \
                               --output-dir models/llava_v07_qlora_2k

Based on train_v03_qlora.py; only default paths and version strings changed.
Training logic and hyperparameters are identical to v0.3.
"""

import os
import sys
import json
import argparse
import yaml
import torch
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Check if required libraries are installed
# NOTE: trl must be imported BEFORE transformers (LlavaForConditionalGeneration)
# to avoid STATUS_ACCESS_VIOLATION (-1073741819) on Windows.
try:
    from trl import SFTTrainer
    from transformers import (
        LlavaForConditionalGeneration,
        AutoProcessor,
        TrainingArguments,
        BitsAndBytesConfig
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training
    )
    from datasets import Dataset
    from PIL import Image
    import pandas as pd
    print("✓ All required libraries found")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("\nInstall required packages:")
    print("pip install transformers>=4.41.0 peft>=0.11.0 bitsandbytes>=0.43.0 accelerate>=0.30.0 datasets trl pillow pandas")
    sys.exit(1)


@dataclass
class TrainingConfig:
    """Training configuration loaded from config.yaml and CLI arguments."""

    # Model paths
    base_model_id: str  = "llava-hf/llava-1.5-7b-hf"
    train_data_csv: str = "data/v07_fine_tuning/train_1k.csv"
    output_dir: str     = "models/llava_v07_qlora_1k"

    # LoRA configuration
    lora_r: int         = 32
    lora_alpha: int     = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training hyperparameters
    batch_size: int                  = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float             = 2e-4
    num_epochs: int                  = 3
    warmup_steps: int                = 50
    max_grad_norm: float             = 1.0
    weight_decay: float              = 0.01

    # Optimization
    optim: str            = "adamw_torch"
    lr_scheduler_type: str = "cosine"

    # Mixed precision
    # RTX 4060 Ti supports BF16 natively; fp16 GradScaler is incompatible
    # with BFloat16 tensors produced by 4-bit QLoRA.
    fp16: bool = False
    bf16: bool = True

    # Logging and checkpointing
    logging_steps: int    = 10
    save_steps: int       = 100
    eval_steps: int       = 100
    save_total_limit: int = 3

    # Data
    max_seq_length: int = 2048
    train_split: float  = 0.8

    # Paths — 336x336 center-crop for LLaVA-1.5 native resolution
    image_base_dir: str = "data/inspect_images_336"

    @classmethod
    def from_config_and_args(cls, config_path: str, args: argparse.Namespace):
        """Create config from YAML file and CLI arguments."""
        # Load YAML config
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # Extract v04 fine-tuning config (same hyperparameters)
        v04_config = config_dict.get('v04_fine_tuning', {})

        default_target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

        instance = cls(
            base_model_id=v04_config.get('base_model_id', "llava-hf/llava-1.5-7b-hf"),
            lora_r=v04_config.get('lora', {}).get('r', 32),
            lora_alpha=v04_config.get('lora', {}).get('alpha', 64),
            lora_dropout=v04_config.get('lora', {}).get('dropout', 0.05),
            target_modules=v04_config.get('lora', {}).get('target_modules', default_target_modules),
            batch_size=v04_config.get('training', {}).get('batch_size', 4),
            gradient_accumulation_steps=v04_config.get('training', {}).get('gradient_accumulation_steps', 4),
            learning_rate=v04_config.get('training', {}).get('learning_rate', 2e-4),
            num_epochs=v04_config.get('training', {}).get('num_epochs', 3),
            warmup_steps=v04_config.get('training', {}).get('warmup_steps', 50),
            max_grad_norm=v04_config.get('training', {}).get('max_grad_norm', 1.0),
            weight_decay=v04_config.get('training', {}).get('weight_decay', 0.01),
            optim=v04_config.get('training', {}).get('optim', "adamw_torch"),
            lr_scheduler_type=v04_config.get('training', {}).get('lr_scheduler_type', "cosine"),
            fp16=v04_config.get('training', {}).get('fp16', True),
            bf16=v04_config.get('training', {}).get('bf16', False),
            logging_steps=v04_config.get('training', {}).get('logging_steps', 10),
            save_steps=v04_config.get('training', {}).get('save_steps', 100),
            eval_steps=v04_config.get('training', {}).get('eval_steps', 100),
            save_total_limit=v04_config.get('training', {}).get('save_total_limit', 3),
            max_seq_length=v04_config.get('data', {}).get('max_seq_length', 2048),
            train_split=v04_config.get('data', {}).get('train_split', 0.8),
            # Image directory is shared with v0.3 (same source images)
            image_base_dir=config_dict.get('v03_dataset', {}).get(
                'image_dir',
                "data/inspect_images_v2"
            )
        )

        # --train-size shorthand: maps "1k"→v07 paths, "2k"→v07 paths, etc.
        if hasattr(args, 'train_size') and args.train_size and not args.train_data:
            sz = args.train_size  # e.g. "1k", "2k", "3k", "4k"
            instance.train_data_csv = f"data/v07_fine_tuning/train_{sz}.csv"
            instance.output_dir     = f"models/llava_v07_qlora_{sz}"

        # Explicit CLI paths take precedence over --train-size
        if args.train_data:
            instance.train_data_csv = args.train_data
        if args.output_dir:
            instance.output_dir = args.output_dir
        if args.image_base_dir:
            instance.image_base_dir = args.image_base_dir

        # LoRA / training overrides
        if args.lora_r:
            instance.lora_r = args.lora_r
        if args.lora_alpha:
            instance.lora_alpha = args.lora_alpha
        if args.epochs:
            instance.num_epochs = args.epochs
        if args.lr:
            instance.learning_rate = args.lr
        if args.batch_size:
            instance.batch_size = args.batch_size

        return instance


class LLaVADataset:
    """Dataset handler for LLaVA fine-tuning from CSV."""

    def __init__(self, csv_path: str, image_base_dir: str):
        self.csv_path       = Path(csv_path)
        self.image_base_dir = Path(image_base_dir)
        self.data           = self.load_csv_data()

    def load_csv_data(self) -> pd.DataFrame:
        """Load training data from CSV."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.csv_path}")

        print(f"📁 Loading training data: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        print(f"✓ Loaded {len(df)} training samples")

        required_cols = ['ファイルパス', '所見']
        missing_cols  = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df_valid = df.dropna(subset=['所見']).copy()
        df_valid = df_valid[df_valid['所見'].astype(str).str.len() >= 10]

        print(f"✓ Valid samples after filtering: {len(df_valid)}")
        return df_valid

    def create_huggingface_dataset(self, processor: AutoProcessor) -> Dataset:
        """Create HuggingFace dataset."""
        dataset_dict = {"image": [], "text": [], "image_path": []}

        skipped = 0
        for _, row in self.data.iterrows():
            filename   = Path(row['ファイルパス']).name
            image_path = self.image_base_dir / filename

            if not image_path.exists():
                alt_path = self.image_base_dir / Path(row['ファイルパス']).name
                if not alt_path.exists():
                    print(f"⚠️  Image not found: {image_path}")
                    skipped += 1
                    continue
                image_path = alt_path

            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"❌ Error loading {image_path}: {e}")
                skipped += 1
                continue

            ground_truth_text = str(row['所見'])
            prompt = (
                "USER: <image>\n"
                "Describe the damage visible in this bridge structure image in detail. "
                "Include damage types, severity, location, and extent. "
                "Use Japanese for the description.\n"
                "ASSISTANT:"
            )
            full_text = f"{prompt} {ground_truth_text}"

            dataset_dict["image"].append(image)
            dataset_dict["text"].append(full_text)
            dataset_dict["image_path"].append(str(image_path))

        print(f"✓ Created dataset with {len(dataset_dict['image'])} samples")
        if skipped > 0:
            print(f"⚠️  Skipped {skipped} samples due to missing/invalid images")

        return Dataset.from_dict(dataset_dict)


def setup_qlora_model(model_id: str, config: TrainingConfig):
    """Setup model with QLoRA (4-bit NF4, same as v0.3)."""
    print(f"\n📦 Loading base model: {model_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16  # BF16 consistent with bf16=True in TrainingArguments
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16  # BF16 consistent
    )
    print("✓ Model loaded with 4-bit quantization")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Parameters:")
    print(f"  Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total    : {total_params:,}")

    return model


def collate_fn(processor):
    """Create collate function with processor closure."""
    def _collate(batch):
        images = [item['image'] for item in batch]
        texts  = [item['text']  for item in batch]
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs
    return _collate


def train(config: TrainingConfig):
    """Main training function."""
    print("=" * 80)
    print("LLaVA v0.7 Progressive QLoRA Fine-Tuning (pairdata_v2)")
    print("=" * 80)
    print(f"Training Data  : {config.train_data_csv}")
    print(f"Output Dir     : {config.output_dir}")
    print(f"LoRA Config    : r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"Training       : {config.num_epochs} epochs, LR={config.learning_rate}, batch_size={config.batch_size}")
    print("=" * 80)
    print()

    os.makedirs(config.output_dir, exist_ok=True)

    config_save_path = Path(config.output_dir) / "training_config.json"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(vars(config), f, ensure_ascii=False, indent=2)
    print(f"✓ Saved training configuration: {config_save_path}")

    print("\n📝 Loading processor...")
    processor = AutoProcessor.from_pretrained(config.base_model_id)

    print("\n📂 Loading dataset...")
    dataset_handler = LLaVADataset(config.train_data_csv, config.image_base_dir)
    dataset = dataset_handler.create_huggingface_dataset(processor)

    if len(dataset) == 0:
        print("❌ No valid samples found in training data.")
        return

    print(f"\n🔀 Splitting dataset: {config.train_split*100:.0f}% train / {(1-config.train_split)*100:.0f}% val")
    split         = dataset.train_test_split(test_size=1 - config.train_split, seed=42)
    train_dataset = split['train']
    val_dataset   = split['test']
    print(f"  Train samples     : {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    model = setup_qlora_model(config.base_model_id, config)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        weight_decay=config.weight_decay,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
        logging_dir=f"{config.output_dir}/logs",
        remove_unused_columns=False,
        dataloader_pin_memory=False
    )

    print("\n🚀 Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn(processor),
        processing_class=processor
    )

    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)

    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()

    training_duration = end_time - start_time
    print(f"\n✅ Training Complete!")
    print(f"   Duration: {training_duration}")

    print(f"\n💾 Saving final model to: {config.output_dir}")
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)

    summary = {
        "version":          "v0.7",
        "pairdata":         "pairdata_v2 (denoised)",
        "training_config":  vars(config),
        "training_duration": str(training_duration),
        "start_time":       start_time.isoformat(),
        "end_time":         end_time.isoformat(),
        "train_samples":    len(train_dataset),
        "val_samples":      len(val_dataset),
        "final_checkpoint": config.output_dir
    }
    summary_path = Path(config.output_dir) / "training_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved training summary: {summary_path}")
    print(f"\n🎉 Fine-tuning complete! Model saved to: {config.output_dir}")


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="LLaVA v0.7 Progressive QLoRA Fine-Tuning (pairdata_v2)"
    )

    # Convenience shorthand: --train-size 1k / 2k / 3k / 4k
    parser.add_argument(
        "--train-size", type=str, choices=["1k", "2k", "3k", "4k"],
        help="Training set size (e.g. 1k → data/v07_fine_tuning/train_1k.csv)"
    )

    # Explicit path overrides (take precedence over --train-size)
    parser.add_argument(
        "--train-data", type=str,
        help="Path to training CSV (e.g., data/v07_fine_tuning/train_2k.csv)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        help="Output directory for checkpoints (e.g., models/llava_v07_qlora_2k)"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml file"
    )
    parser.add_argument(
        "--image-base-dir", type=str,
        help="Image directory override (e.g., data/inspect_images_448)"
    )

    # LoRA arguments
    parser.add_argument("--lora-r",     type=int,   help="LoRA rank (default: 32)")
    parser.add_argument("--lora-alpha", type=int,   help="LoRA alpha (default: 64)")

    # Training arguments
    parser.add_argument("--epochs",     type=int,   help="Number of training epochs (default: 3)")
    parser.add_argument("--lr",         type=float, help="Learning rate (default: 2e-4)")
    parser.add_argument("--batch-size", type=int,   help="Batch size (default: 4)")

    args = parser.parse_args()

    if not args.train_size and not args.train_data:
        parser.error("Provide either --train-size {1k|2k|3k|4k} or --train-data <path>")

    config = TrainingConfig.from_config_and_args(args.config, args)
    train(config)


if __name__ == "__main__":
    main()
