#!/usr/bin/env python
"""
Fast benchmark runner that loads model once and evaluates multiple benchmarks in-memory.
Similar to full_validation_benchmark_onestep_fast_rahlora.py but for multiple benchmarks.
"""

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import time
from datetime import datetime
import sys

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import io
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line.get("image")
        qs = line.get("text", "")
        # Avoid duplicating image tokens if prompt already contains one
        if "<image>" not in qs:
            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if image_file:
            image_path = os.path.join(self.image_folder, image_file)
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception:
                # If image path is invalid, fallback to blank image
                image = Image.new('RGB', (336, 336), color=(255, 255, 255))
        else:
            # Gracefully handle text-only samples by using a blank image
            image = Image.new('RGB', (336, 336), color=(255, 255, 255))
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size, index

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, indices = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes, indices


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size=1, num_workers=8):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    # Enable pin_memory for faster GPU transfers
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    return data_loader


def eval_benchmark_inmemory(model, tokenizer, image_processor, benchmark, output_dir, model_name, 
                            num_samples=None, temperature=0, conv_mode="vicuna_v1"):
    """Evaluate a single benchmark with model already in memory"""
    
    print(f"\n>>> Running {benchmark} benchmark...")
    
    # Benchmark-specific configurations
    if benchmark == "scienceqa":
        question_file = "./playground/data/eval/scienceqa/llava_test_CQM-A.json"
        image_folder = "./playground/data/eval/scienceqa/images"
    elif benchmark == "textvqa":
        # Use subset file if available for consistency with run_benchmarks.sh
        question_file = "./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr_subset_1000.jsonl"
        if not os.path.exists(question_file):
            question_file = "./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
        image_folder = "./playground/data/eval/textvqa/train_images"
    elif benchmark == "gqa":
        question_file = "./playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl"
        image_folder = "./playground/data/eval/gqa/data/images"
    else:
        print(f"Unknown benchmark: {benchmark}")
        return None
    
    # Check if files exist
    if not os.path.exists(question_file):
        print(f"Question file not found: {question_file}")
        return None
    if not os.path.exists(image_folder):
        print(f"Image folder not found: {image_folder}")
        return None
        
    # Load questions
    if benchmark == "scienceqa":
        # ScienceQA file may be a JSON array with conversation format
        try:
            if question_file.endswith('.jsonl'):
                raw_items = [json.loads(line) for line in open(question_file, 'r')]
            else:
                with open(question_file, 'r') as f:
                    raw_items = json.load(f)
        except Exception as e:
            print(f"Failed to load ScienceQA question file: {e}")
            return None

        questions = []
        for it in raw_items:
            # Extract text prompt
            q_text = it.get("text")
            if not q_text and isinstance(it.get("conversations"), list):
                for turn in it["conversations"]:
                    if isinstance(turn, dict) and turn.get("from") == "human":
                        q_text = turn.get("value", "")
                        break
            questions.append({
                "question_id": it.get("id", it.get("question_id", "unknown")),
                "image": it.get("image"),  # may be None for text-only
                "text": q_text or ""
            })
    else:
        questions = [json.loads(q) for q in open(question_file, "r")]
    # num_samples=0 means use all samples, >0 means limit
    if num_samples and num_samples > 0:
        questions = questions[:num_samples]
    elif num_samples == 0:
        print(f"Using all {len(questions)} samples (num_samples=0)")
    else:
        # num_samples is None, use all
        pass
    
    print(f"Processing {len(questions)} questions...")
    
    # Output file
    output_file = f"{output_dir}/{benchmark}_answers.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ans_file = open(output_file, "w")
    
    # Adjust conv_mode if needed
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in conv_mode:
        conv_mode = conv_mode + '_mmtag'
        print(f'Auto switching to {conv_mode} for plain model.')
    
    # Create data loader with optimized workers
    # Reduce workers if processing many samples to avoid memory issues
    if len(questions) > 500:
        optimal_workers = 2  # Use fewer workers for large datasets
    else:
        optimal_workers = min(4, os.cpu_count() or 2)  # Cap at 4 workers
    data_loader = create_data_loader(questions, image_folder, tokenizer, image_processor, 
                                    model.config, conv_mode, batch_size=1, num_workers=optimal_workers)
    
    # Get device handling for multi-GPU
    if hasattr(model, 'device_map'):
        device = 'cuda:0'  # Input always starts at first device
    else:
        device = next(model.parameters()).device
    
    # Process each question
    start_time = time.time()
    for (input_ids, image_tensor, image_sizes, indices), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line.get("question_id", line.get("id", "unknown"))
        if benchmark == "scienceqa":
            # Ensure ScienceQA IDs are integers to match evaluator keys
            try:
                idx = int(idx)
            except Exception:
                pass
        cur_prompt = line["text"]
        
        # Handle multi-GPU properly - FIXED: always move to device
        if hasattr(model, 'device_map'):
            input_ids = input_ids.to('cuda:0', non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device='cuda:0', non_blocking=True)  # FIX: Move to GPU!
        else:
            input_ids = input_ids.to(device=device, non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device=device, non_blocking=True)
        
        # Generate output
        # Optimize max_new_tokens based on benchmark
        if benchmark == "scienceqa":
            max_tokens = 64  # ScienceQA needs explanation after answer
        elif benchmark == "gqa":
            max_tokens = 16  # GQA typically has short answers
        elif benchmark == "textvqa":
            max_tokens = 16  # TextVQA answers are usually 1-3 words
        else:
            max_tokens = 64  # Default for unknown benchmarks
            
        with torch.inference_mode():
            # Override any default max_length from model config
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": True if temperature > 0 else False,
                "temperature": temperature,
                "num_beams": 1,
                "use_cache": True,
                "max_length": None  # Explicitly disable max_length to use max_new_tokens
            }
            
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                **gen_kwargs)
        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Write result
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }) + "\n")
    
    ans_file.close()
    elapsed_time = time.time() - start_time
    print(f"Completed {benchmark} in {elapsed_time:.2f}s ({elapsed_time/len(questions):.2f}s per sample)")
    
    # Run evaluation if applicable
    if benchmark == "scienceqa":
        print("Running ScienceQA evaluation...")
        os.system(f"python llava/eval/eval_science_qa.py "
                 f"--base-dir ./playground/data/eval/scienceqa "
                 f"--result-file {output_file} "
                 f"--output-file {output_dir}/{benchmark}_output.jsonl "
                 f"--output-result {output_dir}/{benchmark}_result.json")
        
        # Extract accuracy
        result_file = f"{output_dir}/{benchmark}_result.json"
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)
                print(f"ScienceQA Accuracy: {result.get('acc', 'N/A')}")
    
    elif benchmark == "textvqa":
        annotation_file = "./playground/data/eval/textvqa/TextVQA_0.5.1_val.json"
        if os.path.exists(annotation_file):
            print("Running TextVQA evaluation...")
            os.system(f"python -m llava.eval.eval_textvqa "
                     f"--annotation-file {annotation_file} "
                     f"--result-file {output_file}")
            
            # Show metrics
            metrics_file = f"{output_dir}/{benchmark}_metrics.json"
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    print(f.read())
    
    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--benchmarks", type=str, default="scienceqa textvqa gqa",
                       help="Space-separated list of benchmarks")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples per benchmark (None for all)")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--device-map", type=str, default="auto",
                       help="Device map for multi-GPU ('auto' or specific mapping)")
    
    args = parser.parse_args()
    
    print(f"=== Fast Benchmark Runner ===")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Samples: {args.num_samples if args.num_samples else 'all'}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model ONCE
    print("\nLoading model...")
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    
    # Set device map for multi-GPU if available
    device_map = None
    if torch.cuda.device_count() > 1 and args.device_map == "auto":
        device_map = "auto"
        print(f"Using automatic device mapping across {torch.cuda.device_count()} GPUs")
    
    # Load model with optional device_map parameter
    # IMPORTANT: Force SDPA attention for fast inference (override any saved eager attention)
    if device_map == "auto":
        # Multi-GPU setup
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, 
            args.model_base, 
            model_name,
            device_map=device_map,
            attn_implementation="sdpa"  # Force SDPA for speed
        )
    else:
        # Single GPU or CPU setup
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, 
            args.model_base, 
            model_name,
            attn_implementation="sdpa"  # Force SDPA for speed
        )
        # For single GPU, ensure model is on CUDA
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"Model moved to CUDA:0")
    
    # Handle image_processor fallback
    if image_processor is None:
        from transformers import CLIPImageProcessor
        try:
            # Try loading from model path first
            image_processor = CLIPImageProcessor.from_pretrained(args.model_path)
            print(f"Loaded CLIPImageProcessor from {args.model_path}")
        except:
            # Fallback to vision tower or default
            vt = getattr(model.config, "mm_vision_tower", None) or getattr(model.config, "vision_tower", None)
            if vt:
                try:
                    image_processor = CLIPImageProcessor.from_pretrained(vt)
                    print(f"Loaded CLIPImageProcessor from vision tower: {vt}")
                except:
                    pass
            
            if image_processor is None:
                # Last resort: use default CLIP processor
                image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
                print("Using default CLIP processor as fallback")
    
    print(f"Model loaded successfully!")
    
    # Apply torch.compile for speed optimization (1.5x speedup)
    # Skip compile for multi-GPU to avoid issues
    if device_map != "auto" and torch.__version__ >= "2.0.0":
        try:
            print("Applying torch.compile optimization...")
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("Model compiled successfully!")
        except Exception as e:
            print(f"torch.compile skipped: {e}")
    else:
        print("Skipping torch.compile for multi-GPU setup")
    
    # Run each benchmark
    benchmarks = args.benchmarks.split()
    results = {}
    
    total_start = time.time()
    for bench in benchmarks:
        output_file = eval_benchmark_inmemory(
            model, tokenizer, image_processor, 
            bench, args.output_dir, model_name,
            num_samples=args.num_samples,
            temperature=args.temperature,
            conv_mode=args.conv_mode
        )
        if output_file:
            results[bench] = "completed"
        else:
            results[bench] = "failed"
    
    total_time = time.time() - total_start
    
    # Save summary
    print(f"\n=== Summary ===")
    print(f"Total time: {total_time:.2f}s")
    for bench, status in results.items():
        print(f"  {bench}: {status}")
    
    summary = {
        "model": args.model_path,
        "benchmarks": results,
        "total_time": total_time,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{args.output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
