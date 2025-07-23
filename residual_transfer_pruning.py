#!/usr/bin/env python
"""
Improved pruning with residual transfer (V and O only)
"""
import torch
import torch.nn.functional as F
import json
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

def residual_transfer_kv(layer, pruned_idx, keep_idx, beta=0.1):
    """Transfer V weights with residual approach"""
    d = layer.self_attn.head_dim
    s_p = slice(pruned_idx*d, (pruned_idx+1)*d)
    s_k = slice(keep_idx*d, (keep_idx+1)*d)
    
    # V only transfer
    v_keep = layer.self_attn.v_proj.weight.data[s_k]
    v_prun = layer.self_attn.v_proj.weight.data[s_p]
    
    # Calculate adaptive beta based on cosine similarity
    cos = F.cosine_similarity(v_keep.view(-1), v_prun.view(-1), dim=0).item()
    adaptive_beta = beta * max(cos, 0)  # No transfer for negative similarity
    
    # Residual transfer
    v_keep += adaptive_beta * (v_prun - v_keep)
    
    # Norm rebalance to prevent explosion
    with torch.no_grad():
        expected_scale = (d ** 0.5)
        current_norm = v_keep.norm(p=2)
        if current_norm > expected_scale * 1.5:  # Clip if too large
            v_keep.mul_(expected_scale / current_norm)
    
    # O-proj column transfer (maintain consistency)
    layer.self_attn.o_proj.weight.data[:, s_k] += adaptive_beta * (
        layer.self_attn.o_proj.weight.data[:, s_p] - 
        layer.self_attn.o_proj.weight.data[:, s_k]
    )
    
    return adaptive_beta

def apply_residual_pruning(model, calibration, max_beta=0.1):
    """Apply pruning with residual V/O transfer"""
    
    with torch.no_grad():
        for layer_idx in range(32):
            layer = model.model.layers[layer_idx]
            prune_mask = calibration['pruned_heads'][str(layer_idx)]
            num_pruned = sum(prune_mask)
            
            if num_pruned > 0:
                print(f"\nLayer {layer_idx}: Pruning {num_pruned} heads")
                
                # Get dimensions
                head_dim = layer.self_attn.head_dim
                num_heads = len(prune_mask)
                
                # Find pruned and remaining heads
                pruned_heads = [i for i, p in enumerate(prune_mask) if p]
                remaining_heads = [i for i, p in enumerate(prune_mask) if not p]
                
                # For each pruned head, find most similar remaining head
                for p_idx in pruned_heads:
                    # Get V weights
                    v_p = layer.self_attn.v_proj.weight[p_idx*head_dim:(p_idx+1)*head_dim]
                    
                    # Find most similar remaining head
                    best_sim = -1
                    best_idx = remaining_heads[0]
                    
                    for r_idx in remaining_heads:
                        v_r = layer.self_attn.v_proj.weight[r_idx*head_dim:(r_idx+1)*head_dim]
                        sim = F.cosine_similarity(v_p.view(-1), v_r.view(-1), dim=0).item()
                        if sim > best_sim:
                            best_sim = sim
                            best_idx = r_idx
                    
                    # Transfer with residual
                    beta = residual_transfer_kv(layer, p_idx, best_idx, max_beta)
                    print(f"  Head {p_idx} -> {best_idx}, sim={best_sim:.3f}, beta={beta:.3f}")
                
                # Zero out pruned heads
                for p_idx in pruned_heads:
                    start = p_idx * head_dim
                    end = (p_idx + 1) * head_dim
                    
                    layer.self_attn.q_proj.weight[start:end] = 0
                    layer.self_attn.k_proj.weight[start:end] = 0
                    layer.self_attn.v_proj.weight[start:end] = 0
                    layer.self_attn.o_proj.weight[:, start:end] = 0
                
                # Apply uniform scale to remaining heads (Q/K only)
                num_remaining = len(remaining_heads)
                scale_factor = float(num_heads) / float(num_remaining)
                
                for r_idx in remaining_heads:
                    start = r_idx * head_dim
                    end = (r_idx + 1) * head_dim
                    
                    # Scale Q/K for attention score preservation
                    layer.self_attn.q_proj.weight[start:end] *= scale_factor
                    layer.self_attn.k_proj.weight[start:end] *= scale_factor
                    # V already has residual transfer, no additional scaling

if __name__ == "__main__":
    # Test with Early Skip calibration
    model_path = './checkpoints/llava-v1.5-7b'
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, get_model_name_from_path(model_path),
        device_map='auto'
    )
    
    with open('s4_experiments/calibrations/s4_early_skip_real.json', 'r') as f:
        calibration = json.load(f)
    
    print(f"Applying residual transfer pruning...")
    print(f"Total heads to prune: {calibration['total_pruned_heads']}/{calibration['total_heads']}")
    
    apply_residual_pruning(model, calibration, max_beta=0.1)
    
    print("\nPruning complete!")