"""
v0.5.1 Inference Script: Optimized inference with batch processing and torch.compile
Improvements:
- Batch size 4 for faster processing
- torch.compile() for optimization
- max_new_tokens=384 (reduced from 512)
- repetition_penalty=1.2 to reduce redundancy
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# Import unsloth early when available so patching occurs before model loading.
try:
    import unsloth  # noqa: F401
except Exception:
    unsloth = None

# Unsloth + Windows + Triton の不安定なコンパイル経路を既定で無効化。
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

# Check if required libraries are installed
try:
    from transformers import (
        LlavaForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig
    )
    from peft import PeftModel
    from PIL import Image
    print("✓ All required libraries found")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("\nInstall required packages:")
    print("pip install transformers>=4.41.0 peft>=0.11.0 bitsandbytes>=0.43.0 pillow pandas")
    sys.exit(1)


class LLaVAQLoRAInferenceOptimized:
    """Optimized inference class for LLaVA with QLoRA adapters"""
    
    def __init__(
        self,
        model_dir: str,
        base_model_id: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda",
        load_in_4bit: bool = True,
        use_compile: bool = True,
        use_unsloth: bool = False
    ):
        """
        Initialize inference model
        
        Args:
            model_dir: Directory containing LoRA adapters
            base_model_id: Base LLaVA model ID
            device: Device to use ('cuda' or 'cpu')
            load_in_4bit: Whether to load model in 4-bit quantization
            use_compile: Whether to use torch.compile() for optimization
        """
        self.model_dir = Path(model_dir)
        self.base_model_id = base_model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.load_in_4bit = load_in_4bit
        self.use_compile = use_compile
        self.use_unsloth = use_unsloth
        
        print(f"\n🔧 Initializing LLaVA QLoRA Inference (v0.5.1 Optimized)")
        print(f"   Model Dir: {model_dir}")
        print(f"   Base Model: {base_model_id}")
        print(f"   Device: {self.device}")
        print(f"   4-bit Quantization: {load_in_4bit}")
        print(f"   torch.compile: {use_compile}")
        print(f"   Unsloth FastVision: {use_unsloth}")
        
        # Load model and processor
        self._load_model()
        print(f"✓ Model loaded successfully\n")
    
    def _load_model(self):
        """Load base model with QLoRA adapters and apply torch.compile()"""
        if self.use_unsloth:
            print("Loading base model with Unsloth FastVisionModel...")
            from unsloth import FastVisionModel

            base_model, _ = FastVisionModel.from_pretrained(
                model_name=self.base_model_id,
                load_in_4bit=self.load_in_4bit,
                use_gradient_checkpointing=False,
            )

            print("Loading LoRA adapters...")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.model.eval()
            # for_inference() を必ず呼ぶ (v0.6.3 と同じ方式: バッチ推論に必須)
            try:
                FastVisionModel.for_inference(self.model)
            except Exception as e:
                print(f"⚠️  FastVisionModel.for_inference failed: {e}")
                print("   Continuing without for_inference() optimization")

            print("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.base_model_id)
            # left-padding: バッチ内の各シーケンス右端(生成開始位置)を揃える (v0.6.3 準拠)
            self.processor.tokenizer.padding_side = "left"
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            return

        # Quantization config
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None
        
        print("Loading base model...")
        # Load base model
        base_model = LlavaForConditionalGeneration.from_pretrained(
            self.base_model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        print("Loading LoRA adapters...")
        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(
            base_model,
            self.model_dir,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Set to evaluation mode
        self.model.eval()
        
        # Apply torch.compile() for optimization (v0.5.1 improvement)
        if self.use_compile and self.device == "cuda":
            print("Applying torch.compile() optimization...")
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("✓ torch.compile() applied successfully")
            except Exception as e:
                print(f"⚠️  torch.compile() failed: {e}")
                print("   Continuing without compilation...")
        
        # Load processor
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(self.base_model_id)
    
    def predict_single(
        self,
        image_path: str,
        prompt: str = "この橋梁の損傷状態を詳しく説明してください。",
        max_new_tokens: int = 384,  # v0.5.1: Reduced from 512
        temperature: float = 0.1,
        do_sample: bool = False,
        repetition_penalty: float = 1.2  # v0.5.1: Added
    ) -> str:
        """
        Generate prediction for a single image
        
        Args:
            image_path: Path to input image
            prompt: Text prompt for the model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            repetition_penalty: Penalty for repetition (1.0 = no penalty)
            
        Returns:
            Generated text description
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        text_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=text_prompt,
            images=image,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,  # v0.5.1: Added
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        # Decode
        generated_text = self.processor.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Extract response (after "assistant" tag)
        if "assistant" in generated_text:
            response = generated_text.split("assistant")[-1].strip()
        else:
            response = generated_text.strip()
        
        return response
    
    def predict_batch(
        self,
        image_paths: List[str],
        image_ids: List[str],
        ground_truths: Optional[List[str]] = None,
        prompt: str = "この橋梁の損傷状態を詳しく説明してください。",
        output_csv: Optional[str] = None,
        max_new_tokens: int = 384,  # v0.5.1: Reduced from 512
        batch_size: int = 4,  # v0.5.1: Added batch processing
        repetition_penalty: float = 1.2  # v0.5.1: Added
    ) -> pd.DataFrame:
        """
        Generate predictions for a batch of images with batched processing
        
        Args:
            image_paths: List of image file paths
            image_ids: List of image identifiers
            ground_truths: Optional list of ground truth texts
            prompt: Text prompt for the model
            output_csv: Optional path to save results CSV
            max_new_tokens: Maximum number of tokens to generate
            batch_size: Number of images to process simultaneously
            repetition_penalty: Penalty for repetition
            
        Returns:
            DataFrame with predictions
        """
        results = []
        
        print(f"📊 Processing {len(image_paths)} images with batch_size={batch_size}...")
        
        # Process in batches
        for batch_start in tqdm(range(0, len(image_paths), batch_size), desc="Batches"):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_img_paths = image_paths[batch_start:batch_end]
            batch_img_ids = image_ids[batch_start:batch_end]
            
            # Load batch of images
            batch_images = []
            valid_indices = []
            for idx, img_path in enumerate(batch_img_paths):
                try:
                    image = Image.open(img_path).convert("RGB")
                    batch_images.append(image)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"\n❌ Error loading {batch_img_ids[idx]}: {e}")
                    result = {
                        'image_id': batch_img_ids[idx],
                        'image_path': img_path,
                        'prediction': f"ERROR: {str(e)}",
                        'prediction_length': 0
                    }
                    if ground_truths is not None:
                        result['ground_truth'] = ground_truths[batch_start + idx]
                        result['ground_truth_length'] = len(ground_truths[batch_start + idx])
                    results.append(result)
            
            # Skip if no valid images in batch
            if not batch_images:
                continue
            
            # Prepare batch conversations
            batch_conversations = []
            for _ in batch_images:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                batch_conversations.append(conversation)
            
            # Apply chat template to all conversations
            batch_prompts = [
                self.processor.apply_chat_template(conv, add_generation_prompt=True)
                for conv in batch_conversations
            ]
            
            try:
                # Process batch
                inputs = self.processor(
                    text=batch_prompts,
                    images=batch_images,
                    return_tensors="pt",
                    padding=True
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate for batch
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.1,
                        do_sample=False,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=self.processor.tokenizer.pad_token_id
                    )
                
                # Decode batch outputs
                for idx, output in enumerate(outputs):
                    generated_text = self.processor.decode(output, skip_special_tokens=True)
                    
                    # Extract response
                    if "assistant" in generated_text:
                        response = generated_text.split("assistant")[-1].strip()
                    else:
                        response = generated_text.strip()
                    
                    actual_idx = valid_indices[idx]
                    result = {
                        'image_id': batch_img_ids[actual_idx],
                        'image_path': batch_img_paths[actual_idx],
                        'prediction': response,
                        'prediction_length': len(response)
                    }
                    
                    if ground_truths is not None:
                        result['ground_truth'] = ground_truths[batch_start + actual_idx]
                        result['ground_truth_length'] = len(ground_truths[batch_start + actual_idx])
                    
                    results.append(result)
                    
            except Exception as e:
                print(f"\n❌ Error processing batch {batch_start}-{batch_end}: {e}")
                # Fall back to single image processing for this batch
                for idx in valid_indices:
                    try:
                        prediction = self.predict_single(
                            batch_img_paths[idx],
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                            repetition_penalty=repetition_penalty
                        )
                        
                        result = {
                            'image_id': batch_img_ids[idx],
                            'image_path': batch_img_paths[idx],
                            'prediction': prediction,
                            'prediction_length': len(prediction)
                        }
                        
                        if ground_truths is not None:
                            result['ground_truth'] = ground_truths[batch_start + idx]
                            result['ground_truth_length'] = len(ground_truths[batch_start + idx])
                        
                        results.append(result)
                    except Exception as e2:
                        print(f"\n❌ Error processing {batch_img_ids[idx]}: {e2}")
                        result = {
                            'image_id': batch_img_ids[idx],
                            'image_path': batch_img_paths[idx],
                            'prediction': f"ERROR: {str(e2)}",
                            'prediction_length': 0
                        }
                        if ground_truths is not None:
                            result['ground_truth'] = ground_truths[batch_start + idx]
                            result['ground_truth_length'] = len(ground_truths[batch_start + idx])
                        results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV if requested
        if output_csv:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n✓ Results saved to: {output_csv}")
        
        # Print summary
        error_count = len(df[df['prediction'].str.contains('ERROR', na=False)])
        print(f"\n📈 Inference Summary:")
        print(f"   Total images: {len(df)}")
        print(f"   Successful: {len(df) - error_count}")
        print(f"   Errors: {error_count}")
        print(f"   Mean prediction length: {df[~df['prediction'].str.contains('ERROR', na=False)]['prediction_length'].mean():.1f} chars")
        
        return df


def main():
    """Main inference script"""
    parser = argparse.ArgumentParser(
        description="Run optimized inference using fine-tuned QLoRA models (v0.5.1)"
    )
    
    # Required arguments
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing LoRA adapters (e.g., models/llava_v03_qlora_2k)"
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        required=True,
        help="Path to test set CSV file (e.g., data/v03_fine_tuning/test_set_n800.csv)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to save inference results CSV"
    )
    
    # Optional arguments
    parser.add_argument(
        "--base-model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Base model ID (default: llava-hf/llava-1.5-7b-hf)"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/image_text_inspect_n10789/rank_c_images_n10789",
        help="Base directory for images"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="この橋梁の損傷状態を詳しく説明してください。",
        help="Prompt for the model"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=384,  # v0.5.1: Reduced from 512
        help="Maximum number of tokens to generate (default: 384)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,  # v0.5.1: Added
        help="Batch size for inference (default: 4)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,  # v0.5.1: Added
        help="Repetition penalty (default: 1.2)"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile() optimization"
    )
    parser.add_argument(
        "--use-unsloth",
        action="store_true",
        help="Use Unsloth FastVisionModel backend for faster inference"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        sys.exit(1)
    
    test_csv = Path(args.test_csv)
    if not test_csv.exists():
        print(f"❌ Test CSV not found: {test_csv}")
        sys.exit(1)
    
    image_base_dir = Path(args.image_dir)
    if not image_base_dir.exists():
        print(f"❌ Image directory not found: {image_base_dir}")
        sys.exit(1)
    
    # Load test data
    print(f"📂 Loading test data from: {test_csv}")
    df_test = pd.read_csv(test_csv, encoding='utf-8')
    
    if args.limit:
        df_test = df_test.head(args.limit)
        print(f"⚠️  Limited to {args.limit} images for testing")
    
    print(f"   Total samples: {len(df_test)}")
    
    # Prepare image paths
    image_paths = [str(image_base_dir / row['ファイルパス']) for _, row in df_test.iterrows()]
    image_ids = [Path(row['ファイルパス']).stem for _, row in df_test.iterrows()]
    ground_truths = df_test['所見'].tolist()
    
    # Verify images exist
    missing_images = [p for p in image_paths if not Path(p).exists()]
    if missing_images:
        print(f"\n⚠️  Warning: {len(missing_images)} images not found")
        print(f"   First missing: {missing_images[0]}")
    
    # Initialize inference model
    inferencer = LLaVAQLoRAInferenceOptimized(
        model_dir=args.model_dir,
        base_model_id=args.base_model,
        use_compile=not args.no_compile,
        use_unsloth=args.use_unsloth
    )
    
    # Run inference
    start_time = datetime.now()
    print(f"\n🚀 Starting optimized inference at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Optimizations: batch_size={args.batch_size}, max_tokens={args.max_tokens}, repetition_penalty={args.repetition_penalty}")
    
    df_results = inferencer.predict_batch(
        image_paths=image_paths,
        image_ids=image_ids,
        ground_truths=ground_truths,
        prompt=args.prompt,
        output_csv=args.output_csv,
        max_new_tokens=args.max_tokens,
        batch_size=args.batch_size,
        repetition_penalty=args.repetition_penalty
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n⏱️  Total inference time: {duration}")
    print(f"   Average time per image: {duration.total_seconds() / len(df_results):.2f}s")
    print(f"\n✓ Optimized inference complete!")


if __name__ == "__main__":
    main()
