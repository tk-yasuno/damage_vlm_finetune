"""
v0.3-v0.4 QLoRA Fine-Tuning: Progressive Training Script for LLaVA-1.5-7B
Supports multiple dataset sizes (1k/2k/3k/4k) with CLI arguments
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
try:
    from transformers import (
        LlavaForConditionalGeneration,
        AutoProcessor,
        TrainingArguments,
        Trainer,
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
    print("pip install transformers>=4.41.0 peft>=0.11.0 bitsandbytes>=0.43.0 accelerate>=0.30.0 datasets pillow pandas")
    sys.exit(1)


@dataclass
class TrainingConfig:
    """Training configuration loaded from config.yaml and CLI arguments"""
    
    # Model paths
    base_model_id: str = "llava-hf/llava-1.5-7b-hf"
    train_data_csv: str = "data/v03_fine_tuning/train_1k.csv"
    output_dir: str = "models/llava_v03_qlora_1k"
    
    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training hyperparameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Optimization
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    
    # Mixed precision
    fp16: bool = True
    bf16: bool = False
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Data
    max_seq_length: int = 2048
    train_split: float = 0.8
    
    # Paths
    image_base_dir: str = "data/image_text_inspect_n10789/rank_c_images_n10789"
    
    @classmethod
    def from_config_and_args(cls, config_path: str, args: argparse.Namespace):
        """Create config from YAML file and CLI arguments"""
        # Load YAML config
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract v04 fine-tuning config
        v04_config = config_dict.get('v04_fine_tuning', {})
        
        # Default target modules
        default_target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        # Create instance with defaults from YAML
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
            image_base_dir=config_dict.get('v03_dataset', {}).get('image_dir', "data/image_text_inspect_n10789/rank_c_images_n10789")
        )
        
        # Override with CLI arguments
        if args.train_data:
            instance.train_data_csv = args.train_data
        if args.output_dir:
            instance.output_dir = args.output_dir
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
    """Dataset handler for LLaVA fine-tuning from CSV"""
    
    def __init__(self, csv_path: str, image_base_dir: str):
        self.csv_path = Path(csv_path)
        self.image_base_dir = Path(image_base_dir)
        self.data = self.load_csv_data()
    
    def load_csv_data(self) -> pd.DataFrame:
        """Load training data from CSV"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.csv_path}")
        
        print(f"📁 Loading training data: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        print(f"✓ Loaded {len(df)} training samples")
        
        # Check required columns
        required_cols = ['ファイルパス', '所見']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter rows with valid ground truth text
        df_valid = df.dropna(subset=['所見']).copy()
        df_valid = df_valid[df_valid['所見'].astype(str).str.len() >= 10]  # At least 10 chars
        
        print(f"✓ Valid samples after filtering: {len(df_valid)}")
        
        return df_valid
    
    def create_huggingface_dataset(self, processor: AutoProcessor) -> Dataset:
        """Create HuggingFace dataset"""
        
        dataset_dict = {
            "image": [],
            "text": [],
            "image_path": []
        }
        
        skipped = 0
        for idx, row in self.data.iterrows():
            # Get image path
            filename = Path(row['ファイルパス']).name
            image_path = self.image_base_dir / filename
            
            if not image_path.exists():
                # Try alternative: use filename directly
                alt_path = self.image_base_dir / Path(row['ファイルパス']).name
                if not alt_path.exists():
                    print(f"⚠️  Image not found: {image_path}")
                    skipped += 1
                    continue
                image_path = alt_path
            
            # Load image
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"❌ Error loading {image_path}: {e}")
                skipped += 1
                continue
            
            # Create training prompt (LLaVA format)
            ground_truth_text = str(row['所見'])
            prompt = "USER: <image>\nDescribe the damage visible in this bridge structure image in detail. Include damage types, severity, location, and extent. Use Japanese for the description.\nASSISTANT:"
            full_text = f"{prompt} {ground_truth_text}"
            
            dataset_dict["image"].append(image)
            dataset_dict["text"].append(full_text)
            dataset_dict["image_path"].append(str(image_path))
        
        print(f"✓ Created dataset with {len(dataset_dict['image'])} samples")
        if skipped > 0:
            print(f"⚠️  Skipped {skipped} samples due to missing/invalid images")
        
        return Dataset.from_dict(dataset_dict)


def setup_qlora_model(model_id: str, config: TrainingConfig):
    """Setup model with QLoRA"""
    
    print(f"\n📦 Loading base model: {model_id}")
    
    # Load model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    print("✓ Model loaded with 4-bit quantization")
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Parameters:")
    print(f"  Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total: {total_params:,}")
    
    return model


def collate_fn(processor):
    """Create collate function with processor closure"""
    def _collate(batch):
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]
        
        # Process images and text
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        # Prepare labels (same as input_ids for causal LM)
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs
    return _collate


def train(config: TrainingConfig):
    """Main training function"""
    
    print("=" * 80)
    print("LLaVA v0.3-v0.4 Progressive QLoRA Fine-Tuning")
    print("=" * 80)
    print(f"Training Data: {config.train_data_csv}")
    print(f"Output Directory: {config.output_dir}")
    print(f"LoRA Config: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"Training: {config.num_epochs} epochs, LR={config.learning_rate}, batch_size={config.batch_size}")
    print("=" * 80)
    print()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config_save_path = Path(config.output_dir) / "training_config.json"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(vars(config), f, ensure_ascii=False, indent=2)
    print(f"✓ Saved training configuration: {config_save_path}")
    
    # Load processor
    print("\n📝 Loading processor...")
    processor = AutoProcessor.from_pretrained(config.base_model_id)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    dataset_handler = LLaVADataset(config.train_data_csv, config.image_base_dir)
    dataset = dataset_handler.create_huggingface_dataset(processor)
    
    if len(dataset) == 0:
        print("❌ No valid samples found in training data.")
        return
    
    # Train/validation split
    print(f"\n🔀 Splitting dataset: {config.train_split*100:.0f}% train, {(1-config.train_split)*100:.0f}% validation")
    split = dataset.train_test_split(test_size=1 - config.train_split, seed=42)
    train_dataset = split['train']
    val_dataset = split['test']
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Setup model
    model = setup_qlora_model(config.base_model_id, config)
    
    # Training arguments
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
        report_to=["tensorboard"],
        logging_dir=f"{config.output_dir}/logs",
        remove_unused_columns=False,
        dataloader_pin_memory=False
    )
    
    # Create trainer
    print("\n🚀 Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn(processor)
    )
    
    # Train
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)
    
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    
    training_duration = end_time - start_time
    print(f"\n✅ Training Complete!")
    print(f"   Duration: {training_duration}")
    
    # Save final model
    print(f"\n💾 Saving final model to: {config.output_dir}")
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    
    # Save training summary
    summary = {
        "training_config": vars(config),
        "training_duration": str(training_duration),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "final_checkpoint": config.output_dir
    }
    
    summary_path = Path(config.output_dir) / "training_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved training summary: {summary_path}")
    print(f"\n🎉 Fine-tuning complete! Model saved to: {config.output_dir}")


def main():
    """Main entry point with CLI arguments"""
    parser = argparse.ArgumentParser(description="LLaVA v0.3-v0.4 Progressive QLoRA Fine-Tuning")
    
    # Data arguments
    parser.add_argument("--train-data", type=str, 
                       help="Path to training CSV (e.g., data/v03_fine_tuning/train_1k.csv)")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for checkpoints (e.g., models/llava_v03_qlora_1k)")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config.yaml file")
    
    # LoRA arguments
    parser.add_argument("--lora-r", type=int, help="LoRA rank (default: 32)")
    parser.add_argument("--lora-alpha", type=int, help="LoRA alpha (default: 64)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, help="Number of training epochs (default: 3)")
    parser.add_argument("--lr", type=float, help="Learning rate (default: 2e-4)")
    parser.add_argument("--batch-size", type=int, help="Batch size (default: 4)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = TrainingConfig.from_config_and_args(args.config, args)
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
