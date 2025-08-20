#!/usr/bin/env python
"""
Evaluate benchmark results and print a summary
"""

import argparse
import json
import os
import sys


def evaluate_scienceqa(answers_file, base_dir="./playground/data/eval/scienceqa"):
    """Evaluate ScienceQA results"""
    try:
        # Run evaluation
        output_file = answers_file.replace("_answers.jsonl", "_output.jsonl")
        result_file = answers_file.replace("_answers.jsonl", "_result.json")
        
        cmd = f"python llava/eval/eval_science_qa.py --base-dir {base_dir} --result-file {answers_file} --output-file {output_file} --output-result {result_file}"
        os.system(cmd + " 2>/dev/null")
        
        # Read results
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)
                return result.get('acc', 0)
    except Exception as e:
        print(f"Error evaluating ScienceQA: {e}")
    return 0


def evaluate_textvqa(answers_file, annotation_file="./playground/data/eval/textvqa/TextVQA_0.5.1_val.json"):
    """Evaluate TextVQA results"""
    try:
        if not os.path.exists(annotation_file):
            return 0
            
        # Run evaluation and capture output
        import subprocess
        cmd = f"python -m llava.eval.eval_textvqa --annotation-file {annotation_file} --result-file {answers_file}"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        # Parse accuracy from output
        for line in result.stdout.split('\n'):
            if 'Accuracy:' in line:
                # Extract number from "Accuracy: XX.XX%"
                acc_str = line.split('Accuracy:')[1].strip().replace('%', '')
                return float(acc_str)
    except Exception as e:
        print(f"Error evaluating TextVQA: {e}")
    return 0


def evaluate_gqa(answers_file):
    """Evaluate GQA results using official eval script"""
    try:
        import subprocess
        import json
        
        # Convert to GQA format
        gqa_dir = "./playground/data/eval/gqa/data"
        predictions_file = f"{gqa_dir}/testdev_balanced_predictions.json"
        questions_file = f"{gqa_dir}/testdev_balanced_questions.json"
        
        # Convert answers to GQA format
        all_answers = []
        with open(answers_file, 'r') as f:
            for line in f:
                res = json.loads(line)
                question_id = res['question_id']
                text = res['text'].rstrip('.').lower()
                all_answers.append({"questionId": question_id, "prediction": text})
        
        with open(predictions_file, 'w') as f:
            json.dump(all_answers, f)
        
        # Try minimal evaluation with questions file
        if os.path.exists(questions_file) and os.path.exists(f"{gqa_dir}/eval_minimal.py"):
            cmd = f"python {gqa_dir}/eval_minimal.py --questions {questions_file} --predictions {predictions_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # Parse accuracy from output
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Overall Accuracy' in line:
                        # Extract accuracy value
                        acc_str = line.split(':')[1].strip().replace('%', '')
                        try:
                            return float(acc_str)
                        except:
                            pass
        
        # Try official GQA evaluation
        cmd = f"cd {gqa_dir} && python eval.py --tier testdev_balanced --predictions testdev_balanced_predictions.json"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Parse accuracy from output
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Accuracy' in line and ':' in line:
                    # Extract accuracy value
                    acc_str = line.split(':')[1].strip().replace('%', '')
                    try:
                        return float(acc_str)
                    except:
                        pass
        
        # Fallback to predictions-only evaluation
        if os.path.exists(f"{gqa_dir}/eval_predictions_only.py"):
            cmd = f"python {gqa_dir}/eval_predictions_only.py --predictions {predictions_file}"
        else:
            cmd = f"python {gqa_dir}/eval_simple.py --predictions {predictions_file}"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Just return the count as we don't have ground truth
        print("Note: GQA evaluation failed, returning answer count")
        
        # If no accuracy found, count answers as fallback
        return len(all_answers)
    except Exception as e:
        print(f"Error evaluating GQA: {e}")
        # Fallback: just count answers
        try:
            with open(answers_file, 'r') as f:
                count = sum(1 for line in f)
            return count
        except:
            return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Experiment directory containing answer files')
    parser.add_argument('--benchmarks', type=str, default="textvqa scienceqa gqa",
                        help='Space-separated list of benchmarks to evaluate')
    
    args = parser.parse_args()
    
    benchmarks = args.benchmarks.split()
    results = {}
    
    print("\n=== Benchmark Evaluation Results ===")
    print(f"Experiment: {args.exp_dir}")
    print("-" * 40)
    
    for bench in benchmarks:
        answers_file = os.path.join(args.exp_dir, f"{bench}_answers.jsonl")
        
        if not os.path.exists(answers_file):
            print(f"{bench:12s}: No results found")
            continue
            
        if bench == "scienceqa":
            accuracy = evaluate_scienceqa(answers_file)
            results[bench] = accuracy
            print(f"{bench:12s}: {accuracy:.2f}% accuracy")
            
        elif bench == "textvqa":
            accuracy = evaluate_textvqa(answers_file)
            results[bench] = accuracy
            if accuracy > 0:
                print(f"{bench:12s}: {accuracy:.2f}% accuracy")
            else:
                print(f"{bench:12s}: No evaluation available")
                
        elif bench == "gqa":
            result = evaluate_gqa(answers_file)
            results[bench] = result
            # Check if it's accuracy or just count
            if result > 0 and result <= 100:
                print(f"{bench:12s}: {result:.2f}% accuracy")
            else:
                print(f"{bench:12s}: {int(result)} answers generated")
    
    print("-" * 40)
    
    # Save summary
    summary_file = os.path.join(args.exp_dir, "evaluation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    
    return results


if __name__ == "__main__":
    main()