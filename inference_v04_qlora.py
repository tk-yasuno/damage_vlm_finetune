"""
v0.5 Inference Script: Run predictions using fine-tuned QLoRA models
Supports all 4 model variants (1k/2k/3k/4k) on test set
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


class LLaVAQLoRAInference:
    """Inference class for LLaVA with QLoRA adapters"""
    
    def __init__(
        self,
        model_dir: str,
        base_model_id: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda",
        load_in_4bit: bool = True
    ):
        """
        Initialize inference model
        
        Args:
            model_dir: Directory containing LoRA adapters
            base_model_id: Base LLaVA model ID
            device: Device to use ('cuda' or 'cpu')
            load_in_4bit: Whether to load model in 4-bit quantization
        """
        self.model_dir = Path(model_dir)
        self.base_model_id = base_model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.load_in_4bit = load_in_4bit
        
        print(f"\n🔧 Initializing LLaVA QLoRA Inference")
        print(f"   Model Dir: {model_dir}")
        print(f"   Base Model: {base_model_id}")
        print(f"   Device: {self.device}")
        print(f"   4-bit Quantization: {load_in_4bit}")
        
        # Load model and processor
        self._load_model()
        print(f"✓ Model loaded successfully\n")
    
    def _load_model(self):
        """Load base model with QLoRA adapters"""
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
        
        # Load processor
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(self.model_dir)
    
    def predict_single(
        self,
        image_path: str,
        prompt: str = "この橋梁の損傷状態を詳しく説明してください。",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False
    ) -> str:
        """
        Generate prediction for a single image
        
        Args:
            image_path: Path to input image
            prompt: Text prompt for the model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
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
        max_new_tokens: int = 512
    ) -> pd.DataFrame:
        """
        Generate predictions for a batch of images
        
        Args:
            image_paths: List of image file paths
            image_ids: List of image identifiers
            ground_truths: Optional list of ground truth texts
            prompt: Text prompt for the model
            output_csv: Optional path to save results CSV
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            DataFrame with predictions
        """
        results = []
        
        print(f"📊 Processing {len(image_paths)} images...")
        
        for idx, (img_path, img_id) in enumerate(tqdm(
            zip(image_paths, image_ids),
            total=len(image_paths),
            desc="Inference"
        )):
            try:
                # Generate prediction
                prediction = self.predict_single(
                    img_path,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens
                )
                
                result = {
                    'image_id': img_id,
                    'image_path': img_path,
                    'prediction': prediction,
                    'prediction_length': len(prediction)
                }
                
                # Add ground truth if provided
                if ground_truths is not None:
                    result['ground_truth'] = ground_truths[idx]
                    result['ground_truth_length'] = len(ground_truths[idx])
                
                results.append(result)
                
            except Exception as e:
                print(f"\n❌ Error processing {img_id}: {e}")
                result = {
                    'image_id': img_id,
                    'image_path': img_path,
                    'prediction': f"ERROR: {str(e)}",
                    'prediction_length': 0
                }
                if ground_truths is not None:
                    result['ground_truth'] = ground_truths[idx]
                    result['ground_truth_length'] = len(ground_truths[idx])
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
        print(f"\n📈 Inference Summary:")
        print(f"   Total images: {len(df)}")
        print(f"   Successful: {len(df[df['prediction'] != df['prediction'].str.contains('ERROR', na=False)])}")
        print(f"   Mean prediction length: {df['prediction_length'].mean():.1f} chars")
        
        return df


def main():
    """Main inference script"""
    parser = argparse.ArgumentParser(
        description="Run inference using fine-tuned QLoRA models"
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
        default=512,
        help="Maximum number of tokens to generate"
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
    inferencer = LLaVAQLoRAInference(
        model_dir=args.model_dir,
        base_model_id=args.base_model
    )
    
    # Run inference
    start_time = datetime.now()
    print(f"\n🚀 Starting inference at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    df_results = inferencer.predict_batch(
        image_paths=image_paths,
        image_ids=image_ids,
        ground_truths=ground_truths,
        prompt=args.prompt,
        output_csv=args.output_csv,
        max_new_tokens=args.max_tokens
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n⏱️  Total inference time: {duration}")
    print(f"   Average time per image: {duration.total_seconds() / len(df_results):.2f}s")
    print(f"\n✓ Inference complete!")


if __name__ == "__main__":
    main()
