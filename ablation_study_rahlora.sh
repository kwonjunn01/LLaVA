#!/bin/bash
# RAH-LoRA ablation study for LLaVA calibration
# Tests different hyperparameters and layer ranges

# Activate conda environment
source /home/diml/anaconda3/etc/profile.d/conda.sh
conda activate llava
#
# Usage:
#   ./ablation_study_rahlora.sh                    # Run with default settings
#   SAVE_MODELS=true ./ablation_study_rahlora.sh   # Save all calibrated models
#   NUM_SAMPLES=200 ./ablation_study_rahlora.sh    # Use 200 samples
#   SAVE_MODELS=true NUM_SAMPLES=50 ISAL_SAMPLES=20 ./ablation_study_rahlora.sh

# GPU Configuration - Use single GPU for calibration, multi-GPU for benchmarks
# (Multi-GPU causes I-SAL calculation to hang)

# Configuration
OUTPUT_BASE="outputs/ablation_rahlora_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE"

# Default experiment parameters
NUM_SAMPLES="${NUM_SAMPLES:-100}"
ISAL_SAMPLES="${ISAL_SAMPLES:-50}"
CC3M_SAMPLES="${CC3M_SAMPLES:-}"  # If set, override ISAL_SAMPLES for CC3M
SAVE_MODELS="${SAVE_MODELS:-false}"  # Set to true to save calibrated models
USE_CC3M="${USE_CC3M:-true}"  # Use CC3M dataset for calibration
# Multi-benchmark evaluation: space-separated list of benchmarks
# Options: textvqa scienceqa gqa mme mmbench pope
# Use "textvqa_only" to skip multi-benchmark evaluation
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-textvqa scienceqa gqa}"  # All benchmarks now available
# Full benchmark evaluation (0 = all samples, otherwise use NUM_SAMPLES)
FULL_EVAL="${FULL_EVAL:-false}"  # Set to true for full benchmark evaluation
USE_FULL_UPDATE="${USE_FULL_UPDATE:-false}"  # Use full-rank update instead of LoRA
DEFAULT_RANK="${DEFAULT_RANK:-4}"  # Default LoRA rank (increased from 4)
DEFAULT_ALPHA="${DEFAULT_ALPHA:-0.5}"  # Default alpha_max (increased from 0.3)

# Initialize results CSV
RESULTS_CSV="$OUTPUT_BASE/rahlora_ablation_results.csv"

# Log file
LOG_FILE="$OUTPUT_BASE/ablation_log.txt"

# Function to log messages
log_message() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Run baseline once at the beginning
run_baseline() {
    if [ ! -f "$OUTPUT_BASE/baseline_accuracy.txt" ]; then
        log_message "
=================================================================
Running baseline evaluation (no calibration)
================================================================="
        
        python full_validation_benchmark_onestep_fast_rahlora.py \
            --num_samples $NUM_SAMPLES \
            --output-dir "$OUTPUT_BASE/baseline" \
            --skip_calibration \
            --exp_name "baseline" > "$OUTPUT_BASE/baseline_output.log" 2>&1
        
        # Extract baseline accuracy
        grep -A2 "Baseline accuracy:" "$OUTPUT_BASE/baseline_output.log" | grep -oP '\d+\.\d+(?=%)' | head -1 > "$OUTPUT_BASE/baseline_accuracy.txt"
        
        BASELINE_ACC=$(cat "$OUTPUT_BASE/baseline_accuracy.txt")
        log_message "Baseline accuracy: $BASELINE_ACC%"
    else
        BASELINE_ACC=$(cat "$OUTPUT_BASE/baseline_accuracy.txt")
        log_message "Using cached baseline accuracy: $BASELINE_ACC%"
    fi
}

# Function to run single RAH-LoRA experiment with multi-benchmark evaluation
run_rahlora_experiment() {
    local exp_name=$1
    local start_layer=$2
    local end_layer=$3
    local budget_ratio=$4
    local rank=$5
    local delta_kl=$6
    local use_caf=$7  # Use CAF instead of HSS
    local exp_dir="$OUTPUT_BASE/$exp_name"
    
    log_message "
=================================================================
Running RAH-LoRA experiment: $exp_name
Parameters:
  Layers: $start_layer to $end_layer
  Budget ratio: $budget_ratio
  LoRA rank: $rank
  KL delta: $delta_kl
  Use CAF: ${use_caf:-true}
================================================================="
    
    # Build command with correct argument names for mllm_head_calibration_simple_rahlora.py
    # Use single GPU for calibration to avoid I-SAL hanging issue
    local cmd="python full_validation_benchmark_onestep_fast_rahlora.py"
    cmd="$cmd --isal-samples $ISAL_SAMPLES"
    cmd="$cmd --start-layer $start_layer"
    cmd="$cmd --end-layer $end_layer"
    cmd="$cmd --budget-ratio $budget_ratio"
    cmd="$cmd --rank $rank"
    cmd="$cmd --delta-kl $delta_kl"
    cmd="$cmd --output-path $exp_dir"
    cmd="$cmd --save-model"
    cmd="$cmd --alpha-max $DEFAULT_ALPHA"
    
    # Note: Removed unsupported CAF/HSS flags for compatibility with original code
    
    # Add CC3M flag if requested
    if [ "$USE_CC3M" = "true" ]; then
        cmd="$cmd --use-cc3m"
        log_message "  Using CC3M dataset for calibration"
        
        # Add CC3M sample override if specified
        if [ -n "$CC3M_SAMPLES" ]; then
            cmd="$cmd --cc3m-samples $CC3M_SAMPLES"
            log_message "  CC3M samples: $CC3M_SAMPLES"
        fi
    fi
    
    # Add full update flag if requested
    if [ "$USE_FULL_UPDATE" = "true" ]; then
        cmd="$cmd --use_full_update"
        log_message "  Using full-rank update (no LoRA)"
    fi
    
    # Model is now always saved automatically (with processor)
    if [ "$SAVE_MODELS" = "true" ]; then
        log_message "  Model saving: ENABLED (will keep models)"
    else
        log_message "  Model saving: TEMPORARY (will delete after evaluation)"
    fi
    
    # Run the calibration with single GPU
    # Export for this subshell to ensure single GPU is used
    CUDA_VISIBLE_DEVICES=0,1,2,3 eval $cmd 2>&1 | tee -a "$LOG_FILE"
    
    # Find the actual output directory (with timestamp)
    # 최신 타임스탬프 폴더 1개만 선택
    actual_dir=$(ls -dt ${exp_dir}_* 2>/dev/null | head -1)
    if [ -z "$actual_dir" ]; then
        actual_dir="$exp_dir"
    fi
    
    # Run benchmarks using the calibrated model
    if [ -d "$actual_dir/calibrated_model" ]; then
        exp_dir="$actual_dir"  # Update exp_dir to actual directory
        log_message "Calibration completed. Model saved at $actual_dir/calibrated_model"
        
        # Run benchmarks if specified
        if [ -n "$EVAL_BENCHMARKS" ]; then
            log_message "Running benchmarks: $EVAL_BENCHMARKS"
            
            # Use the fast benchmark runner for better performance
            if [ -f "./run_benchmarks_fast.py" ]; then
                # Use all 4 GPUs for benchmarks (works fine for evaluation)
                (export CUDA_VISIBLE_DEVICES=0,1,2,3 && python run_benchmarks_fast.py \
                    --model-path "$exp_dir/calibrated_model" \
                    --output-dir "$exp_dir" \
                    --benchmarks "$EVAL_BENCHMARKS" \
                    --num-samples $NUM_SAMPLES \
                    --temperature 0 \
                    --conv-mode vicuna_v1 \
                    --device-map auto) 2>&1 | tee -a "$LOG_FILE"
                
                # Evaluate the results and print summary
                if [ -f "./evaluate_benchmarks.py" ]; then
                    log_message ""
                    python evaluate_benchmarks.py \
                        --exp-dir "$exp_dir" \
                        --benchmarks "$EVAL_BENCHMARKS" 2>&1 | tee -a "$LOG_FILE"
                fi
            elif [ -f "./run_benchmarks.sh" ]; then
                ./run_benchmarks.sh "$exp_dir/calibrated_model" "$exp_dir" "$EVAL_BENCHMARKS" 2>&1 | tee -a "$LOG_FILE"
            else
                log_message "ERROR: No benchmark runner found"
                # Fallback: create placeholder results
                for BENCH in $EVAL_BENCHMARKS; do
                    echo "{\"status\": \"benchmark_skipped\", \"benchmark\": \"$BENCH\"}" > "$exp_dir/${BENCH}_answers.jsonl"
                done
            fi
        else
            log_message "No benchmarks specified (EVAL_BENCHMARKS is empty)"
        fi
        
        # Remove model if not saving
        if [ "$SAVE_MODELS" != "true" ]; then
            log_message "Removing calibrated model to save disk space..."
            rm -rf "$exp_dir/calibrated_model"
        fi
    else
        log_message "Calibration results saved at $actual_dir"
    fi
}

# Main ablation study
log_message "=== RAH-LoRA Ablation Study ==="
log_message "Configuration:"
log_message "  Evaluation samples: $NUM_SAMPLES"
log_message "  Full evaluation: $FULL_EVAL"
log_message "  I-SAL samples: $ISAL_SAMPLES"
log_message "  Save models: $SAVE_MODELS"
log_message "  Use CC3M: $USE_CC3M"
log_message "  CC3M samples: ${CC3M_SAMPLES:-$ISAL_SAMPLES}"
log_message "  Evaluation benchmarks: $EVAL_BENCHMARKS"
log_message "  Use full update: $USE_FULL_UPDATE"
log_message "  Default rank: $DEFAULT_RANK"
log_message "  Default alpha: $DEFAULT_ALPHA"
log_message "  Output directory: $OUTPUT_BASE"

# First run baseline
# run_baseline

# ============================================================
# Ablation 1: Layer Range Study
# ============================================================
log_message "
>>> Ablation 1: Layer Range Study (fixed budget=0.1, rank=4, delta_kl=0.05)"

# # Early layers only
# run_rahlora_experiment "layers_8_11" 8 11 0.10 4 0.05 true

# # Middle layers only  
# run_rahlora_experiment "layers_12_15" 12 15 0.10 4 0.05 true

# # Late layers only
# run_rahlora_experiment "layers_16_19" 16 19 0.10 4 0.05 true

# # Standard range
# run_rahlora_experiment "layers_0_12" 0 12 0.10 $DEFAULT_RANK 0.05 true
# # Standard range with higher budget
# run_rahlora_experiment "layers_0_12_budget20" 0 12 0.20 $DEFAULT_RANK 0.05 true
# # Extended range
# run_rahlora_experiment "layers_8_23" 8 23 0.10 $DEFAULT_RANK 0.05 true

# # Full range
# run_rahlora_experiment "layers_8_31" 8 31 0.10 $DEFAULT_RANK 0.05 true

# # ============================================================
# # Ablation 2: Budget Ratio Study
# # ============================================================
# log_message "
# >>> Ablation 2: Budget Ratio Study (fixed layers=12-23, rank=4, delta_kl=0.05)"

# # Very conservative (5%)
# run_rahlora_experiment "budget_0.05" 12 23 0.05 4 0.05 true

# # Conservative (10%)
# run_rahlora_experiment "budget_0.10" 12 23 0.10 4 0.05 true

# # Moderate (15%)
# run_rahlora_experiment "budget_0.15" 12 23 0.15 4 0.05 true

# # Aggressive (20%)
# run_rahlora_experiment "budget_0.20" 12 23 0.20 4 0.05 true

# # Very aggressive (25%)
# run_rahlora_experiment "budget_0.25" 12 23 0.25 4 0.05 true

# # ============================================================
# # Ablation 3: LoRA Rank Study
# # ============================================================
# log_message "
# >>> Ablation 3: LoRA Rank Study (fixed layers=12-23, budget=0.10, delta_kl=0.05)"

# # Rank 2
# run_rahlora_experiment "rank_2" 12 23 0.10 2 0.05 true

# # Rank 4 (default)
# run_rahlora_experiment "rank_4" 12 23 0.10 4 0.05 true

# # Rank 8
# run_rahlora_experiment "rank_8" 12 23 0.10 8 0.05 true

# # Rank 16
# run_rahlora_experiment "rank_16" 12 23 0.10 16 0.05 true

# # ============================================================
# # Ablation 4: Trust Region (KL Delta) Study
# # ============================================================
# log_message "
# >>> Ablation 4: Trust Region Study (fixed layers=12-23, budget=0.10, rank=4)"

# # Very tight (0.01)
# run_rahlora_experiment "delta_0.01" 12 23 0.10 4 0.01 true

# # Tight (0.05)
# run_rahlora_experiment "delta_0.05" 12 23 0.10 4 0.05 true

# # Moderate (0.10)
# run_rahlora_experiment "delta_0.10" 12 23 0.10 4 0.10 true
# Loose (0.20)
# run_rahlora_experiment "layer_0_8_0.2" 0 7 0.1 4 0.2 true
# # Loose (0.20)
# run_rahlora_experiment "layer_8_15_0.2" 8 15 0.1 4 0.2 true
# Loose (0.20)
run_rahlora_experiment "layer_12_23_0.1" 12 23 0.1 4 0.2 true
# Loose (0.20)
run_rahlora_experiment "layer_16_32_0.2" 12 23 0.2 4 0.2 true
# Loose (0.20)
run_rahlora_experiment "layer_16_23_0.3" 12 23 0.3 4 0.2 true
# Loose (0.20)
run_rahlora_experiment "layer_16_23_0.4" 12 23 0.4 4 0.2 true
# Loose (0.20)
run_rahlora_experiment "layer_16_23_0.5" 12 23 0.45 4 0.2 true
# Loose (0.20)
# run_rahlora_experiment "layer_23_31_0.2" 23 310.1 4 0.2 true
# # Loose (0.20)
# run_rahlora_experiment "layer_0_15_0.2" 0 15 0.1 4 0.2 true
# # Loose (0.20)
# run_rahlora_experiment "layer_16_31_0.2" 16 31 0.1 4 0.2 true
# # Loose (0.20)
# run_rahlora_experiment "layer_0_8_0.3" 0 7 0.1 4 0.3 true
# # Loose (0.20)
# run_rahlora_experiment "layer_8_15_0.3" 8 15 0.1 4 0.3 true
# # Loose (0.20)
# run_rahlora_experiment "layer_16_23_0.3" 16 23 0.1 4 0.3 true
# # Loose (0.20)
# run_rahlora_experiment "layer_23_31_0.3" 23 31 0.1 4 0.3 true
# # Loose (0.20)
# run_rahlora_experiment "layer_0_15_0.3" 0 15 0.1 4 0.3 true
# # Loose (0.20)
# run_rahlora_experiment "layer_16_31_0.3" 16 31 0.1 4 0.3 true
# ============================================================
# Ablation 5: Combined Best Settings
# ============================================================
log_message "
>>> Ablation 5: Combined Best Settings (based on previous results)"

# # Conservative but effective
# run_rahlora_experiment "best_conservative" 12 19 0.10 4 0.05 true

# # Balanced
# run_rahlora_experiment "best_balanced" 12 23 0.15 8 0.10 true

# # Aggressive
# run_rahlora_experiment "best_aggressive" 8 23 0.20 8 0.10 true

# ============================================================
# Results Summary
# ============================================================
log_message "
=================================================================
RAH-LoRA ablation study complete!
Results saved to: $RESULTS_CSV
=================================================================
"

# Display results summary using Python
python - << EOF
import pandas as pd
import os
import numpy as np

csv_path = "$RESULTS_CSV"
baseline_acc_file = "$OUTPUT_BASE/baseline_accuracy.txt"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    
    # Load baseline accuracy
    baseline_acc = 0.0
    if os.path.exists(baseline_acc_file):
        with open(baseline_acc_file, 'r') as f:
            try:
                baseline_acc = float(f.read().strip())
            except:
                baseline_acc = 0.0
    
    print("\n" + "="*80)
    print("RAH-LoRA Ablation Study Results Summary")
    print("="*80)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    print(f"Total experiments: {len(df)}")
    
    if len(df) > 0:
        # Add baseline comparison
        df['vs_baseline'] = df['calibrated_acc'] - baseline_acc
        
        # Group by ablation type
        print("\n" + "-"*80)
        print("1. LAYER RANGE ABLATION")
        print("-"*80)
        layer_exps = df[df['experiment'].str.startswith('layers_')]
        if len(layer_exps) > 0:
            print(layer_exps[['experiment', 'start_layer', 'end_layer', 'calibrated_acc', 
                            'vs_baseline', 'speedup', 'heads_calibrated']].to_string(index=False))
        
        print("\n" + "-"*80)
        print("2. BUDGET RATIO ABLATION")
        print("-"*80)
        budget_exps = df[df['experiment'].str.startswith('budget_')]
        if len(budget_exps) > 0:
            print(budget_exps[['experiment', 'budget_ratio', 'calibrated_acc', 
                             'vs_baseline', 'speedup', 'heads_calibrated']].to_string(index=False))
        
        print("\n" + "-"*80)
        print("3. LORA RANK ABLATION")
        print("-"*80)
        rank_exps = df[df['experiment'].str.startswith('rank_')]
        if len(rank_exps) > 0:
            print(rank_exps[['experiment', 'rank', 'calibrated_acc', 
                           'vs_baseline', 'speedup', 'heads_calibrated']].to_string(index=False))
        
        print("\n" + "-"*80)
        print("4. TRUST REGION (KL DELTA) ABLATION")
        print("-"*80)
        delta_exps = df[df['experiment'].str.startswith('delta_')]
        if len(delta_exps) > 0:
            print(delta_exps[['experiment', 'delta_kl ', 'calibrated_acc', 
                            'vs_baseline', 'speedup', 'rollbacks']].to_string(index=False))
        
        print("\n" + "-"*80)
        print("5. COMBINED BEST SETTINGS")
        print("-"*80)
        best_exps = df[df['experiment'].str.startswith('best_')]
        if len(best_exps) > 0:
            print(best_exps[['experiment', 'calibrated_acc', 'vs_baseline', 
                           'speedup', 'heads_calibrated']].to_string(index=False))
        
        # Overall best results
        print("\n" + "="*80)
        print("OVERALL BEST RESULTS")
        print("="*80)
        
        if 'calibrated_acc' in df.columns and len(df) > 0:
            best_acc = df.loc[df['calibrated_acc'].idxmax()]
            print(f"\nBest accuracy: {best_acc['experiment']}")
            print(f"  Calibrated: {best_acc['calibrated_acc']:.2f}%")
            print(f"  vs Baseline: {best_acc.get('vs_baseline', 0):+.2f}%")
            print(f"  Speedup: {best_acc.get('speedup', 1):.3f}x")
            print(f"  Config: layers {best_acc.get('start_layer', 'N/A')}-{best_acc.get('end_layer', 'N/A')}, "
                  f"budget={best_acc.get('budget_ratio', 'N/A')}, rank={best_acc.get('rank', 'N/A')}, "
                  f"delta_kl={best_acc.get('delta_kl', 'N/A')}")
        
        if 'speedup' in df.columns and len(df) > 0:
            best_speed = df.loc[df['speedup'].idxmax()]
            print(f"\nBest speedup: {best_speed['experiment']}")
            print(f"  Speedup: {best_speed['speedup']:.3f}x")
            print(f"  Accuracy: {best_speed['calibrated_acc']:.2f}% ({best_speed.get('vs_baseline', 0):+.2f}% vs baseline)")
            print(f"  Config: layers {best_speed.get('start_layer', 'N/A')}-{best_speed.get('end_layer', 'N/A')}, "
                  f"budget={best_speed.get('budget_ratio', 'N/A')}, rank={best_speed.get('rank', 'N/A')}, "
                  f"delta_kl={best_speed.get('delta_kl', 'N/A')}")
        
        # Find pareto optimal points (best accuracy-speedup tradeoff)
        if 'calibrated_acc' in df.columns and 'speedup' in df.columns:
            print("\n" + "="*80)
            print("PARETO OPTIMAL CONFIGURATIONS")
            print("="*80)
            
            # Sort by accuracy
            df_sorted = df.sort_values('calibrated_acc', ascending=False)
            pareto = []
            max_speedup = 0
            
            for idx, row in df_sorted.iterrows():
                if row['speedup'] > max_speedup:
                    pareto.append(row)
                    max_speedup = row['speedup']
            
            if pareto:
                pareto_df = pd.DataFrame(pareto)
                print(pareto_df[['experiment', 'calibrated_acc', 'speedup', 
                               'heads_calibrated']].to_string(index=False))

print("\n" + "="*80)
EOF

log_message "
Ablation study analysis complete!
Check $OUTPUT_BASE for detailed logs and results."

if [ "$SAVE_MODELS" = "true" ]; then
    log_message "
Calibrated models and visualizations saved in each experiment directory.
To analyze a specific experiment:
  python analyze_calibration_logs.py --exp-dir $OUTPUT_BASE/<experiment_name>"
fi