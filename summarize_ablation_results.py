#!/usr/bin/env python
"""
Summarize all ablation study results into a comparison table
"""

import argparse
import json
import os
import glob
from datetime import datetime
from tabulate import tabulate


def parse_experiment_name(exp_name):
    """Extract experiment parameters from directory name"""
    parts = exp_name.split('_')
    
    # Default values
    params = {
        'type': 'unknown',
        'layers': '',
        'budget': '',
        'rank': '',
        'delta_kl': ''
    }
    
    # Parse based on pattern
    if 'layer' in exp_name:
        # Format: layer_START_END_BUDGET or similar
        try:
            if len(parts) >= 4:
                params['type'] = 'layer_range'
                params['layers'] = f"{parts[1]}-{parts[2]}"
                params['budget'] = parts[3] if len(parts) > 3 else ''
        except:
            pass
    elif 'budget' in exp_name:
        params['type'] = 'budget'
        params['budget'] = parts[1] if len(parts) > 1 else ''
    elif 'rank' in exp_name:
        params['type'] = 'rank'
        params['rank'] = parts[1] if len(parts) > 1 else ''
    elif 'delta' in exp_name:
        params['type'] = 'delta_kl'
        params['delta_kl'] = parts[1] if len(parts) > 1 else ''
    
    return params


def load_experiment_results(exp_dir):
    """Load results from an experiment directory"""
    results = {
        'name': os.path.basename(exp_dir),
        'scienceqa': None,
        'textvqa': None,
        'gqa': None
    }
    
    # Try to load evaluation summary first
    summary_file = os.path.join(exp_dir, 'evaluation_summary.json')
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            results.update(summary)
    else:
        # Fallback: check individual result files
        # ScienceQA
        scienceqa_result = os.path.join(exp_dir, 'scienceqa_result.json')
        if os.path.exists(scienceqa_result):
            with open(scienceqa_result, 'r') as f:
                data = json.load(f)
                results['scienceqa'] = data.get('acc', 0)
        
        # TextVQA - harder to parse without running evaluation
        textvqa_answers = os.path.join(exp_dir, 'textvqa_answers.jsonl')
        if os.path.exists(textvqa_answers):
            # Just count answers for now
            with open(textvqa_answers, 'r') as f:
                count = sum(1 for line in f)
                results['textvqa_count'] = count
        
        # GQA
        gqa_answers = os.path.join(exp_dir, 'gqa_answers.jsonl')
        if os.path.exists(gqa_answers):
            with open(gqa_answers, 'r') as f:
                count = sum(1 for line in f)
                results['gqa_count'] = count
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default='outputs',
                        help='Base directory containing experiment outputs')
    parser.add_argument('--pattern', type=str, default='ablation_rahlora_*',
                        help='Pattern to match experiment directories')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for summary (optional)')
    
    args = parser.parse_args()
    
    # Find all experiment directories
    base_pattern = os.path.join(args.base_dir, args.pattern)
    exp_dirs = glob.glob(base_pattern)
    
    if not exp_dirs:
        print(f"No experiments found matching pattern: {base_pattern}")
        return
    
    print(f"\nFound {len(exp_dirs)} experiment directories")
    
    # Collect all results
    all_results = []
    
    for exp_dir in sorted(exp_dirs):
        # Look for subdirectories with actual results
        subdirs = glob.glob(os.path.join(exp_dir, '*'))
        
        for subdir in subdirs:
            if os.path.isdir(subdir) and not subdir.endswith('calibrated_model'):
                results = load_experiment_results(subdir)
                params = parse_experiment_name(os.path.basename(subdir))
                results.update(params)
                all_results.append(results)
    
    if not all_results:
        print("No results found in experiment directories")
        return
    
    # Create comparison table
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 80)
    
    # Format results for table
    table_data = []
    for r in all_results:
        row = [
            r['name'][:30],  # Truncate long names
            r.get('layers', '-'),
            r.get('budget', '-'),
            r.get('rank', '-'),
            r.get('delta_kl', '-'),
        ]
        
        # Add benchmark results
        if r.get('scienceqa') is not None:
            row.append(f"{r['scienceqa']:.1f}%")
        else:
            row.append('-')
            
        if r.get('textvqa') is not None:
            row.append(f"{r['textvqa']:.1f}%")
        elif r.get('textvqa_count'):
            row.append(f"({r['textvqa_count']} answers)")
        else:
            row.append('-')
            
        if r.get('gqa') is not None and isinstance(r['gqa'], float):
            row.append(f"{r['gqa']:.1f}%")
        elif r.get('gqa_count') or r.get('gqa'):
            count = r.get('gqa_count', r.get('gqa', 0))
            row.append(f"({count} answers)")
        else:
            row.append('-')
        
        table_data.append(row)
    
    # Print table
    headers = ['Experiment', 'Layers', 'Budget', 'Rank', 'Delta KL', 
               'ScienceQA', 'TextVQA', 'GQA']
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Group by experiment type
    print("\n" + "=" * 80)
    print("GROUPED BY EXPERIMENT TYPE")
    print("=" * 80)
    
    # Group results
    grouped = {}
    for r in all_results:
        exp_type = r.get('type', 'unknown')
        if exp_type not in grouped:
            grouped[exp_type] = []
        grouped[exp_type].append(r)
    
    for exp_type, results in grouped.items():
        print(f"\n### {exp_type.upper().replace('_', ' ')} ###")
        
        # Sort by relevant parameter
        if exp_type == 'layer_range':
            results.sort(key=lambda x: x.get('layers', ''))
        elif exp_type == 'budget':
            results.sort(key=lambda x: float(x.get('budget', '0') or '0'))
        elif exp_type == 'rank':
            results.sort(key=lambda x: int(x.get('rank', '0') or '0'))
        elif exp_type == 'delta_kl':
            results.sort(key=lambda x: float(x.get('delta_kl', '0') or '0'))
        
        for r in results:
            scores = []
            if r.get('scienceqa'):
                scores.append(f"SciQA:{r['scienceqa']:.1f}%")
            if r.get('textvqa'):
                scores.append(f"TextVQA:{r['textvqa']:.1f}%")
            
            if scores:
                print(f"  {r['name'][:40]:40s} -> {', '.join(scores)}")
    
    # Save summary if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\n" + "=" * 80)
    print(f"Total experiments analyzed: {len(all_results)}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("Error: tabulate module not found. Installing...")
        os.system("pip install tabulate")
        print("Please run the script again.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()