"""
Simple-but-Critical MLLM Head Calibration (RAH-LoRA)
Representative Anchor Head Low-Rank Adaptation

핵심: "유사한 '좋은' 이웃의 공통 저차 방향만 살짝 더한다"
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass, asdict

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import copy
import torch.nn as nn


@dataclass
class CalibrationConfig:
    """Calibration hyperparameters"""
    budget_ratio: float = 0.20  # B: proportion of heads to calibrate per layer (increased from 0.10)
    delta_kl: float = 0.10      # δ: KL divergence trust region (relaxed from 0.05)
    rank: int = 4                # r: LoRA rank
    anchor_pool_percentile: float = 0.6  # q: top percentile for anchor candidates
    num_anchors: int = 3         # m: number of anchor heads
    attn_sim_weight: float = 0.5  # ρ: weight for attention similarity vs weight similarity
    trim_ratio: float = 0.1      # trimmed mean ratio
    alpha_max: float = 0.30       # maximum calibration strength (increased from 0.15)
    enable_head_sensitivity: bool = False  # use image sensitivity gating (disabled by default)
    sensitivity_thresh: float = 0.01  # entropy difference threshold
    # New CAF and safety parameters
    use_caf: bool = False        # Use Counterfactual Ablation Filter
    caf_threshold: float = 0.0   # Only calibrate if delta_loss < threshold
    min_anchor_sim: float = 0.05  # Minimum similarity for anchor selection (reduced from 0.2)
    alpha_from_effect: bool = False  # Scale alpha by effect size
    

class CalibrationLogger:
    """Structured logging for calibration process"""
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.logs = {
            'config': {},
            'head_analysis': [],
            'calibration_updates': [],
            'metrics': {},
            'rollbacks': []
        }
    
    def log_config(self, config: CalibrationConfig):
        self.logs['config'] = asdict(config)
    
    def log_head_analysis(self, layer: int, head: int, data: dict):
        self.logs['head_analysis'].append({
            'layer': layer,
            'head': head,
            **data
        })
    
    def log_update(self, layer: int, head: int, anchors: list, alpha: float, delta_kl: float):
        self.logs['calibration_updates'].append({
            'layer': layer,
            'head': head,
            'anchors': anchors,
            'alpha': alpha,
            'delta_kl': delta_kl
        })
    
    def save(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.logs, f, indent=2)


# ============================================================================
# HSS (Head Sensitivity Score) Components
# ============================================================================

class GateWrapper(nn.Module):
    """Per-head scalar gate wrapper for attention modules"""
    def __init__(self, attn_module):
        super().__init__()
        self.attn = attn_module
        self.num_heads = attn_module.num_heads
        self.head_dim = attn_module.head_dim
        device = attn_module.q_proj.weight.device
        self.gate = nn.Parameter(torch.ones(self.num_heads, device=device, dtype=torch.float32),
                                 requires_grad=True)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False, **kw):
        out = self.attn(hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kw)
        x = out[0]  # [B, T, H]
        B, T, H = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        g = self.gate.to(x.dtype).view(1, 1, -1, 1)
        x = (x * g).view(B, T, H)
        return (x,) + out[1:]


def compute_hss_onestep(model, dataloader, layers, device='cuda', num_samples=10):
    """Compute Head Sensitivity Scores in one step using entropy loss"""
    model.eval()
    
    # 1) Freeze all parameters
    for p in model.parameters():
        p.requires_grad_(False)
    
    # 2) Wrap layers with GateWrapper
    originals = {}
    wrappers = {}
    for l in layers:
        layer = model.model.layers[l]
        originals[l] = layer.self_attn
        wrappers[l] = GateWrapper(layer.self_attn).to(device)
        layer.self_attn = wrappers[l]
        wrappers[l].gate.requires_grad_(True)
    
    # 3) Accumulation buffers
    acc = {l: torch.zeros(wrappers[l].num_heads, device=device, dtype=torch.float32) for l in layers}
    denom = {l: 0.0 for l in layers}
    
    # Process limited samples
    sample_count = 0
    for batch in dataloader:
        if sample_count >= num_samples:
            break
            
        # Handle both tuple and dict batch formats
        if isinstance(batch, tuple):
            input_ids = batch[0].to(device)
            images = batch[1].to(device, dtype=torch.float16) if len(batch) > 1 else None
        else:
            input_ids = batch['input_ids'].to(device)
            images = batch['images'].to(device, dtype=torch.float16) if 'images' in batch else None
        
        # Zero gate gradients
        for w in wrappers.values():
            if w.gate.grad is not None:
                w.gate.grad.zero_()
        
        try:
            # Disable gradient computation for everything except gates
            with torch.enable_grad():
                if images is not None:
                    out = model(input_ids, images=images, return_dict=True, output_hidden_states=True)
                else:
                    out = model(input_ids, return_dict=True, output_hidden_states=True)
                logits = out.logits
                
                # Use a simpler loss that doesn't require gradients on logits
                # Just sum of logits as a proxy for sensitivity
                loss = logits.sum()
            
            # Backward only through gates
            loss.backward()
            
            # Collect gradients with layer normalization
            for l in layers:
                g = wrappers[l].gate.grad
                if g is not None:
                    # Use absolute gradient values
                    grad_vals = g.detach().abs()
                    
                    # Normalize by layer's hidden state norm if available
                    if hasattr(out, 'hidden_states') and out.hidden_states is not None:
                        h = out.hidden_states[l+1]  # embeddings offset
                        hn = (h.float().norm(dim=-1).mean().item() + 1e-8)
                    else:
                        hn = 1.0
                    
                    acc[l] += (grad_vals / hn)
                    denom[l] += 1.0
                    
                    # Debug print
                    if sample_count == 0:
                        print(f"  Layer {l}: grad mean={grad_vals.mean():.6f}, max={grad_vals.max():.6f}")
        except Exception as e:
            print(f"HSS computation error for sample {sample_count}: {e}")
            continue
        
        sample_count += 1
    
    # 4) Average and restore
    hss = {}
    for l in layers:
        if denom[l] > 0:
            scores = (acc[l] / denom[l]).cpu().numpy()
            # Replace NaN with small values
            scores = np.nan_to_num(scores, nan=1e-6)
            hss[l] = scores.tolist()
        else:
            # If no gradients were collected, use uniform small values
            num_heads = wrappers[l].num_heads
            hss[l] = [1e-6] * num_heads
            
        model.model.layers[l].self_attn = originals[l]  # restore
        del wrappers[l]
    
    torch.cuda.empty_cache()
    return hss


# ============================================================================
# CAF (Counterfactual Ablation Filter) Functions
# ============================================================================

def compute_caf_onestep(model, dataloader, layers, device='cuda',
                        num_samples=32, loss_mode='margin',
                        use_second_order=True, curvature_weight=0.5):
    """
    One-step CAF: per-head ∆L_ablate ≈ -dL/dγ  (+ 0.5 * Fisher optional)
    Much faster than iterating through each head
    Returns: Dict[layer_idx -> List[delta_loss_estimates]]
    """
    model.eval()
    
    # 1) Freeze all parameters
    for p in model.parameters():
        p.requires_grad_(False)
    
    # 2) Wrap layers with GateWrapper
    originals, wrappers = {}, {}
    for l in layers:
        layer = model.model.layers[l]
        originals[l] = layer.self_attn
        # Get the device of the layer's weights
        layer_device = layer.self_attn.q_proj.weight.device
        wrappers[l] = GateWrapper(layer.self_attn).to(layer_device)
        layer.self_attn = wrappers[l]
        wrappers[l].gate.requires_grad_(True)  # Only gates get gradients
    
    # Accumulators for gradients - use each layer's device
    g_acc = {l: torch.zeros(wrappers[l].num_heads, device=wrappers[l].gate.device, dtype=torch.float32) for l in layers}
    g2_acc = {l: torch.zeros(wrappers[l].num_heads, device=wrappers[l].gate.device, dtype=torch.float32) for l in layers}
    count = 0
    
    # Process samples
    for batch in dataloader:
        if count >= num_samples:
            break
            
        # Handle batch format
        if isinstance(batch, tuple):
            input_ids = batch[0].to(device, non_blocking=True)
            images = batch[1].to(device, dtype=torch.float16, non_blocking=True) if len(batch) > 1 else None
            image_sizes = batch[2] if len(batch) > 2 else None
        else:
            input_ids = batch['input_ids'].to(device)
            images = batch.get('images', None)
            if images is not None:
                images = images.to(device, dtype=torch.float16)
            image_sizes = batch.get('image_sizes', None)
        
        # Zero gate gradients
        for w in wrappers.values():
            if w.gate.grad is not None:
                w.gate.grad.zero_()
        
        with torch.enable_grad():
            if images is not None:
                out = model(input_ids, images=images, return_dict=True)
            else:
                out = model(input_ids, return_dict=True)
            logits = out.logits
            
            # Label-free loss computation
            if loss_mode == 'entropy':
                probs = torch.softmax(logits, dim=-1)
                loss = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            else:  # 'margin' (more stable)
                probs = torch.softmax(logits, dim=-1)
                top2 = torch.topk(probs, k=2, dim=-1).values
                loss = -(torch.log(top2[..., 0] + 1e-8) - torch.log(top2[..., 1] + 1e-8)).mean()
        
        loss.backward()
        
        # Collect gradients
        for l in layers:
            g = wrappers[l].gate.grad
            if g is not None:
                g_acc[l] += g.detach()
                g2_acc[l] += g.detach().pow(2)
        
        count += 1
    
    # 3) Compute CAF scores and restore
    caf = {}
    for l in layers:
        if count == 0:
            n = wrappers[l].num_heads
            caf[l] = [0.0] * n
        else:
            g_mean = (g_acc[l] / count)
            g2_mean = (g2_acc[l] / count)
            
            # Layer-wise z-normalization for stability
            def z_norm(x):
                m = x.mean()
                s = x.std() + 1e-8
                return (x - m) / s
            
            # First-order approximation: -dL/dγ
            delta = -z_norm(g_mean)
            
            if use_second_order:
                # Add curvature correction (Fisher diagonal)
                curv = z_norm(g2_mean)
                delta = delta + curvature_weight * 0.5 * curv
            
            caf[l] = delta.detach().cpu().tolist()
        
        # Restore original attention
        model.model.layers[l].self_attn = originals[l]
        del wrappers[l]
    
    torch.cuda.empty_cache()
    return caf


# Keep old function for backwards compatibility but redirect to new one
def compute_caf_scores(model, data_loader, layers, device='cuda', num_samples=48, bootstrap_rounds=2):
    """
    Legacy CAF function - redirects to faster one-step version
    """
    print("[CAF] Using one-step approximation for speed")
    return compute_caf_onestep(model, data_loader, layers, device, 
                               num_samples=num_samples, loss_mode='margin',
                               use_second_order=True, curvature_weight=0.5)


def compute_fisher_scores(model, data_loader, layers, device='cuda', num_samples=10):
    """
    Compute Fisher Information scores as HSS replacement
    Returns: Dict[layer_idx -> List[head_scores]]
    """
    model.eval()
    
    # Use GateWrapper but compute Fisher instead
    originals = {}
    wrappers = {}
    for l in layers:
        layer = model.model.layers[l]
        originals[l] = layer.self_attn
        # Get the device of the layer's weights
        layer_device = layer.self_attn.q_proj.weight.device
        wrappers[l] = GateWrapper(layer.self_attn).to(layer_device)
        layer.self_attn = wrappers[l]
        wrappers[l].gate.requires_grad_(True)
    
    # Accumulation for Fisher (squared gradients) - use each layer's device
    fisher = {l: torch.zeros(wrappers[l].num_heads, device=wrappers[l].gate.device, dtype=torch.float32) for l in layers}
    denom = {l: 0.0 for l in layers}
    
    sample_count = 0
    for batch in data_loader:
        if sample_count >= num_samples:
            break
        
        # Handle batch format
        if isinstance(batch, tuple):
            input_ids = batch[0].to(device)
            images = batch[1].to(device, dtype=torch.float16) if len(batch) > 1 else None
        else:
            input_ids = batch['input_ids'].to(device)
            images = batch['images'].to(device, dtype=torch.float16) if 'images' in batch else None
        
        # Zero gradients
        for w in wrappers.values():
            if w.gate.grad is not None:
                w.gate.grad.zero_()
        
        # Forward pass
        with torch.enable_grad():
            if images is not None:
                out = model(input_ids, images=images, return_dict=True)
            else:
                out = model(input_ids, return_dict=True)
            
            # Use log probability of most likely token
            logits = out.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            max_log_probs = log_probs.max(dim=-1).values
            loss = max_log_probs.sum()
        
        # Backward
        loss.backward()
        
        # Accumulate squared gradients (Fisher)
        for l in layers:
            g = wrappers[l].gate.grad
            if g is not None:
                fisher[l] += g.detach().pow(2)
                denom[l] += 1.0
        
        sample_count += 1
    
    # Average and restore
    fisher_scores = {}
    for l in layers:
        if denom[l] > 0:
            scores = (fisher[l] / denom[l]).cpu().numpy()
            # Normalize within layer
            mean = scores.mean()
            std = scores.std() + 1e-8
            scores = (scores - mean) / std
            fisher_scores[l] = scores.tolist()
        else:
            num_heads = wrappers[l].num_heads
            fisher_scores[l] = [0.0] * num_heads
        
        model.model.layers[l].self_attn = originals[l]
        del wrappers[l]
    
    torch.cuda.empty_cache()
    return fisher_scores


# ============================================================================
# Core Functions
# ============================================================================

def calculate_isal_scores(model, tokenizer, image_processor, data_loader, device, num_samples=50):
    """
    Calculate I-SAL_TI scores: Text↔Image bidirectional attention mass
    Returns: Tuple(isal_scores, attn_proto_mean)
    """
    model.eval()
    all_attention_scores = {}
    # per-layer list of per-head prototypes (concatenated t2i/i2t, pooled to 128-D)
    attn_protos = {l: [] for l in range(len(model.model.layers))}
    
    # Initialize storage for each layer
    for layer_idx in range(len(model.model.layers)):
        all_attention_scores[layer_idx] = []
    
    with torch.no_grad():
        sample_count = 0
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Calculating I-SAL scores")):
            if sample_count >= num_samples:
                break
            
            # Handle different batch formats (CC3M vs TextVQA)
            if isinstance(batch_data, dict):
                # CC3M format
                input_ids = batch_data['input_ids']
                images = batch_data['images']
                image_sizes = None  # CC3M doesn't use image_sizes
            else:
                # TextVQA format (tuple)
                input_ids, image_tensor, image_sizes, indices = batch_data
                images = image_tensor
            
            # Patch 6: Simplified device handling
            input_ids = input_ids.to(device, non_blocking=True)
            images = images.to(device, dtype=torch.float16, non_blocking=True)
            
            # Find image token positions (may be at index 0)
            image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
            image_token_indices = torch.where(image_token_mask[0])[0]
            
            # Forward pass with attention outputs
            outputs = model(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                output_attentions=True,
                return_dict=True
            )
            
            # Actual seq length after image expansion
            seq_len = outputs.attentions[0].shape[-1]
            
            # Calculate image token range
            if len(image_token_indices) > 0:
                image_start = int(image_token_indices[0].item())
                # Try to infer number of vision tokens from vision tower (fallback: 576)
                num_image_tokens = 576
                vt = getattr(getattr(model.model, "vision_tower", None), "vision_tower", None)
                if vt is not None and hasattr(vt, "config"):
                    patch = getattr(vt.config, "patch_size", 14)
                    imgsz = getattr(vt.config, "image_size", 336)
                    num_image_tokens = (imgsz // patch) ** 2
                image_end = min(image_start + num_image_tokens, seq_len)
                
                # Process attention weights for each layer
                for layer_idx in range(len(outputs.attentions)):
                    attn_weights = outputs.attentions[layer_idx]  # [batch, num_heads, seq_len, seq_len]
                    
                    # Build text token indices: both sides of image span
                    text_idx_parts = []
                    if image_start > 0:
                        text_idx_parts.append(torch.arange(0, image_start, device=attn_weights.device))
                    if image_end < seq_len:
                        text_idx_parts.append(torch.arange(image_end, seq_len, device=attn_weights.device))
                    
                    if len(text_idx_parts) > 0:
                        text_idx = torch.cat(text_idx_parts)
                        # Text→Image and Image→Text, then average
                        t2i = attn_weights[0, :, text_idx, image_start:image_end].mean(dim=(1, 2))
                        i2t = attn_weights[0, :, image_start:image_end, text_idx].mean(dim=(1, 2))
                        score = 0.5 * (t2i + i2t)  # [num_heads]
                        all_attention_scores[layer_idx].append(score.detach().cpu().numpy())
                        
                        # build 1D prototypes (adaptive pool to 64 + concat = 128-D)
                        def _pool(v):
                            v = v.float().unsqueeze(0)  # [1, num_heads]
                            v = F.adaptive_avg_pool1d(v, 64).squeeze(0)
                            return F.normalize(v, p=2, dim=0)
                        p_t2i = _pool(t2i)
                        p_i2t = _pool(i2t)
                        attn_protos[layer_idx].append(torch.cat([p_t2i, p_i2t]).cpu().numpy())
            
            sample_count += 1
    
    # Average scores across samples
    isal_scores = {}
    attn_proto_mean = {}
    for layer_idx in all_attention_scores:
        if all_attention_scores[layer_idx]:
            scores = np.stack(all_attention_scores[layer_idx])
            isal_scores[layer_idx] = scores.mean(axis=0)
            protos = np.stack(attn_protos[layer_idx])  # [samples, heads, 128]
            attn_proto_mean[layer_idx] = protos.mean(axis=0)  # [heads, 128]
        else:
            isal_scores[layer_idx] = np.zeros(model.model.layers[0].self_attn.num_heads)
            attn_proto_mean[layer_idx] = np.zeros((model.model.layers[0].self_attn.num_heads, 128))
    
    return isal_scores, attn_proto_mean


def calculate_image_sensitivity(model, data_loader, device, num_samples=20):
    """
    Calculate image sensitivity for each head
    Returns heads that are primarily text-focused (low image dependency)
    """
    model.eval()
    head_sensitivities = {}
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Calculating image sensitivity")):
            if batch_idx >= num_samples:
                break
            
            # Handle different batch formats (CC3M vs TextVQA)
            if isinstance(batch_data, dict):
                # CC3M format
                input_ids = batch_data['input_ids']
                images = batch_data['images']
            else:
                # TextVQA format (tuple)
                input_ids, image_tensor, image_sizes, indices = batch_data
                images = image_tensor
            
            # Patch 6: Simplified device handling
            input_ids = input_ids.to(device)
            images = images.to(device, dtype=torch.float16)
            
            # Forward with normal images
            outputs_normal = model(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                output_attentions=False,
                return_dict=True
            )
            
            # Forward with blank images
            blank_images = torch.zeros_like(images)
            outputs_blank = model(
                input_ids,
                images=blank_images,
                image_sizes=image_sizes,
                output_attentions=False,
                return_dict=True
            )
            
            # Calculate entropy difference
            probs_normal = F.softmax(outputs_normal.logits, dim=-1)
            probs_blank = F.softmax(outputs_blank.logits, dim=-1)
            
            entropy_normal = -(probs_normal * torch.log(probs_normal + 1e-8)).sum(dim=-1)
            entropy_blank = -(probs_blank * torch.log(probs_blank + 1e-8)).sum(dim=-1)
            
            delta_entropy = (entropy_normal - entropy_blank).abs().mean()
            
            # Store for analysis (simplified - would need per-head in full implementation)
            if batch_idx == 0:
                head_sensitivities['global_delta'] = delta_entropy.item()
    
    return head_sensitivities


def attention_similarity(attn_patterns_h: torch.Tensor, attn_patterns_a: torch.Tensor) -> float:
    """
    Calculate functional similarity based on attention patterns
    """
    # Flatten and normalize
    h = attn_patterns_h.flatten()
    a = attn_patterns_a.flatten()
    h = F.normalize(h.unsqueeze(0), p=2, dim=1).squeeze()
    a = F.normalize(a.unsqueeze(0), p=2, dim=1).squeeze()
    
    return F.cosine_similarity(h.unsqueeze(0), a.unsqueeze(0), dim=1).item()


def trimmed_weighted_mean(tensors: List[torch.Tensor],
                          weights: List[float],
                          trim_ratio: float) -> torch.Tensor:
    """
    L2 deviation-based trimming followed by weighted mean
    """
    if len(tensors) == 1:
        return tensors[0]
    
    X = torch.stack(tensors)  # [n, ...]
    W = torch.tensor(weights, device=X.device, dtype=X.dtype)  # [n]
    
    # Calculate deviations
    mu = X.mean(dim=0, keepdim=True)
    dev = ((X - mu)**2).flatten(1).sum(dim=1).cpu().numpy()
    
    n = len(tensors)
    k = int(n * trim_ratio)
    if k > 0 and n > 2*k:
        keep = np.argsort(dev)[k:n-k]
        X, W = X[keep], W[keep]
    
    W = torch.clamp(W, min=0)
    W = W / (W.sum() + 1e-8)
    shape = (W.shape[0],) + (1,) * (X.dim()-1)
    return (X * W.view(*shape)).sum(dim=0)


def lora_update(W: torch.Tensor, target: torch.Tensor, rank: int = 4, alpha: float = 0.1, use_full_update: bool = False) -> torch.Tensor:
    """
    LoRA-style low-rank update or full update
    """
    delta = (target - W).to(torch.float32)
    
    if use_full_update or rank >= min(W.shape):
        # Direct full-rank update (no approximation)
        return (W + alpha * delta.to(W.dtype))
    else:
        # SVD for low-rank approximation
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        r = min(rank, U.size(1), Vh.size(0))
        
        # Truncated reconstruction
        low_rank_delta = (U[:, :r] * S[:r]) @ Vh[:r, :]
        
        return (W + alpha * low_rank_delta.to(W.dtype))


def make_probe_batch(dataloader, device, n=3):
    """Create small probe batch for KL estimation"""
    it = iter(dataloader)
    batch_data = next(it)
    
    # Handle different batch formats
    if isinstance(batch_data, dict):
        # CC3M format
        probe = {
            "input_ids": batch_data['input_ids'].to(device),
            "images": batch_data['images'].to(device, dtype=torch.float16),
        }
        if 'image_sizes' in batch_data:
            probe['image_sizes'] = batch_data['image_sizes']
    else:
        # TextVQA format (tuple)
        inputs, images, image_sizes, _ = batch_data
        probe = {
            "input_ids": inputs.to(device),
            "images": images.to(device, dtype=torch.float16),
            "image_sizes": image_sizes
        }
    
    return probe

@torch.no_grad()
def estimate_kl(model, probe, logits_before=None):
    """Estimate KL divergence using probe batch"""
    if logits_before is None:
        logits_before = model(**probe).logits
    logits_after = model(**probe).logits
    p = torch.log_softmax(logits_before.float(), dim=-1).exp()
    ql = torch.log_softmax(logits_after.float(), dim=-1)
    kl = (p * (p.log() - ql)).sum(dim=-1).mean().item()
    return kl, logits_before

def trust_region_search(model, W_ref_getter, W_apply, target: torch.Tensor,
                       rank: int, alpha_max: float, delta_kl: float,
                       probe, use_full_update: bool = False) -> Tuple[bool, float]:
    """
    Trust region line search with actual KL divergence
    W_ref_getter(): returns current ref weight tensor (view)
    W_apply(new): in-place apply new weight
    """
    alpha = alpha_max
    kl_before, logits_before = estimate_kl(model, probe, logits_before=None)
    
    for _ in range(4):
        W_orig = W_ref_getter().clone()
        W_new = lora_update(W_orig, target, rank, alpha, use_full_update=use_full_update)
        W_apply(W_new)  # apply new weight
        
        kl_after, _ = estimate_kl(model, probe, logits_before)
        dkl = max(kl_after - kl_before, 0.0)
        
        # Debug: print KL values on first iteration
        if alpha >= alpha_max * 0.99:  # First iteration check
            print(f"    Debug: KL before={kl_before:.6f}, after={kl_after:.6f}, delta={dkl:.6f}, threshold={delta_kl:.6f}")
        
        if dkl <= delta_kl:
            return True, alpha
        
        # rollback and reduce alpha
        W_apply(W_orig)
        alpha *= 0.5
    
    return False, 0.0


# ============================================================================
# Main Calibration Function
# ============================================================================

def calibrate_heads(model, isal_scores: Dict, attn_proto_mean: Dict,
                   config: CalibrationConfig, logger: CalibrationLogger,
                   probe, start_layer: int = 12, end_layer: int = 23,
                   hss_scores: Dict[int, List[float]] = None,
                   caf_scores: Dict[int, List[float]] = None):
    """
    Main calibration function implementing RAH-LoRA with attention patterns
    """
    device = next(model.parameters()).device
    total_calibrated = 0
    total_rollbacks = 0
    sensitivity_eps = 1e-8  # Patch 5: sensitivity gating threshold
    
    for layer_idx in range(start_layer, end_layer + 1):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        
        # Get layer I-SAL scores and attention prototypes
        layer_scores = isal_scores.get(layer_idx, np.zeros(num_heads))
        layer_protos = attn_proto_mean.get(layer_idx, np.zeros((num_heads, 128)))
        
        # Patch 5: Skip heads with negligible I-SAL scores
        active_heads = [h for h in range(num_heads) if layer_scores[h] > sensitivity_eps]
        if len(active_heads) == 0:
            print(f"Layer {layer_idx}: All heads have negligible I-SAL scores, skipping")
            continue
        
        # 1) Select calibration targets (bottom budget_ratio% of active heads)
        active_scores = [layer_scores[h] for h in active_heads]
        threshold = np.percentile(active_scores, config.budget_ratio * 100)
        target_heads = [h for h in active_heads if layer_scores[h] <= threshold]
        
        print(f"  Initial selection: {len(active_heads)} active → {len(target_heads)} targets (budget={config.budget_ratio:.1%})")
        print(f"  I-SAL threshold: {threshold:.6f}, range: [{min(active_scores):.6f}, {max(active_scores):.6f}]")
        
        # HSS filtering: Keep only heads with high sensitivity
        initial_targets = len(target_heads)
        if hss_scores is not None and layer_idx in hss_scores:
            hs = np.abs(np.array(hss_scores[layer_idx]))  # [num_heads]
            # Only filter if we have meaningful HSS values (not all zeros/nans)
            if np.sum(hs) > 1e-5 and not np.all(np.isnan(hs)):
                median = np.median(hs[hs > 0]) if np.any(hs > 0) else 0
                if median > 0:
                    before_hss = len(target_heads)
                    target_heads = [h for h in target_heads if hs[h] >= median]  # Keep only sensitive heads
                    print(f"  HSS filtering: {before_hss} → {len(target_heads)} (median={median:.6f})")
        
        # CAF filtering: Only keep heads that harm performance when removed
        if config.use_caf and caf_scores is not None and layer_idx in caf_scores:
            caf = np.array(caf_scores[layer_idx])
            before_caf = len(target_heads)
            # Keep only heads where delta_loss < threshold (negative = helps to remove)
            target_heads = [h for h in target_heads if caf[h] < config.caf_threshold]
            print(f"  CAF filtering: {before_caf} → {len(target_heads)} heads (threshold={config.caf_threshold})")
        
        # 2) Identify anchor candidates (top anchor_pool_percentile%)
        anchor_threshold = np.percentile(layer_scores, config.anchor_pool_percentile * 100)
        
        print(f"\nLayer {layer_idx}: Calibrating {len(target_heads)} heads")
        
        for head_idx in target_heads:
            # Skip if head has negligible I-SAL (Patch 5)
            if layer_scores[head_idx] <= sensitivity_eps:
                print(f"  Head {head_idx}: Skipping due to negligible I-SAL score")
                continue
            
            # Find anchor candidates
            candidates = []
            
            for anchor_idx in active_heads:  # Only consider active heads as anchors
                if anchor_idx == head_idx:
                    continue
                    
                if layer_scores[anchor_idx] >= anchor_threshold:
                    # Compute weight similarity
                    s_h, e_h = head_idx * head_dim, (head_idx + 1) * head_dim
                    s_a, e_a = anchor_idx * head_dim, (anchor_idx + 1) * head_dim
                    
                    v_h = attn.v_proj.weight.data[s_h:e_h, :].flatten()
                    v_a = attn.v_proj.weight.data[s_a:e_a, :].flatten()
                    o_h = attn.o_proj.weight.data[:, s_h:e_h].flatten()
                    o_a = attn.o_proj.weight.data[:, s_a:e_a].flatten()
                    
                    # Compute cosine similarity properly for 1D tensors
                    sim_v = F.cosine_similarity(v_h.view(1, -1), v_a.view(1, -1), dim=1).item()
                    sim_o = F.cosine_similarity(o_h.view(1, -1), o_a.view(1, -1), dim=1).item()
                    weight_sim = 0.7 * sim_v + 0.3 * sim_o
                    
                    # Patch 2: Add attention pattern similarity
                    proto_h = torch.tensor(layer_protos[head_idx], dtype=torch.float32)
                    proto_a = torch.tensor(layer_protos[anchor_idx], dtype=torch.float32)
                    # Compute cosine similarity properly for 1D tensors
                    attn_sim = F.cosine_similarity(proto_h.view(1, -1), proto_a.view(1, -1), dim=1).item()
                    
                    # Combine weight and attention similarities
                    combined_sim = config.attn_sim_weight * attn_sim + (1 - config.attn_sim_weight) * weight_sim
                    
                    candidates.append({
                        'idx': anchor_idx,
                        'isal': layer_scores[anchor_idx],
                        'similarity': combined_sim
                    })
            
            # Select top-m anchors with minimum similarity check
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            # Filter by minimum similarity
            before_sim_filter = len(candidates)
            candidates = [c for c in candidates if c['similarity'] >= config.min_anchor_sim]
            anchors = candidates[:config.num_anchors]
            
            if not anchors:
                if before_sim_filter > 0:
                    max_sim = max(c['similarity'] for c in candidates[:before_sim_filter]) if before_sim_filter > 0 else 0
                    print(f"  Head {head_idx}: No anchors found ({before_sim_filter} candidates, max_sim={max_sim:.3f} < {config.min_anchor_sim})")
                else:
                    print(f"  Head {head_idx}: No anchor candidates found")
                continue
            
            # 3) Compute RAH (Representative Anchor Head) using trimmed weighted mean
            anchor_weights_v = []
            anchor_weights_o = []
            anchor_weights_q = []
            anchor_weights_k = []
            anchor_scores = []
            
            for anchor in anchors:
                a_idx = anchor['idx']
                s_a, e_a = a_idx * head_dim, (a_idx + 1) * head_dim
                
                # Weight by similarity * I-SAL
                weight = anchor['similarity'] * anchor['isal']
                anchor_scores.append(weight)
                
                anchor_weights_v.append(attn.v_proj.weight.data[s_a:e_a, :])
                anchor_weights_o.append(attn.o_proj.weight.data[:, s_a:e_a])
                anchor_weights_q.append(attn.q_proj.weight.data[s_a:e_a, :])
                anchor_weights_k.append(attn.k_proj.weight.data[s_a:e_a, :])
            
            # Patch 3: Use trimmed_weighted_mean
            rah_v = trimmed_weighted_mean(anchor_weights_v, anchor_scores, config.trim_ratio)
            rah_o = trimmed_weighted_mean(anchor_weights_o, anchor_scores, config.trim_ratio)
            rah_q = trimmed_weighted_mean(anchor_weights_q, anchor_scores, config.trim_ratio)
            rah_k = trimmed_weighted_mean(anchor_weights_k, anchor_scores, config.trim_ratio)
            
            # 4) Apply calibration with trust region
            s_h, e_h = head_idx * head_dim, (head_idx + 1) * head_dim
            
            # Alpha scaling based on effect size
            alpha_cap = config.alpha_max
            
            # CAF-based scaling if enabled
            if config.alpha_from_effect and caf_scores is not None and layer_idx in caf_scores:
                caf = np.array(caf_scores[layer_idx])
                # Use negative delta (positive = head is harmful)
                effect = max(-caf[head_idx], 0)
                # Normalize by median effect in layer
                layer_effects = np.abs(caf)
                median_effect = np.median(layer_effects[layer_effects > 0]) if np.any(layer_effects > 0) else 1e-6
                alpha_cap = config.alpha_max * float(min(effect / (median_effect + 1e-8), 1.0))
                
            # HSS-scaled alpha_max (if no CAF scaling)
            elif hss_scores is not None and layer_idx in hss_scores:
                hs = np.abs(np.array(hss_scores[layer_idx]))
                # Only scale if we have meaningful HSS values
                if np.sum(hs) > 1e-5 and not np.all(np.isnan(hs)) and np.any(hs > 0):
                    # Normalize by 90th percentile in layer
                    kappa = np.percentile(hs[hs > 0], 90) if np.any(hs > 0) else 1.0
                    if kappa > 0:
                        alpha_cap = config.alpha_max * float(min(hs[head_idx]/kappa, 1.0))
            
            # Patch 4: Use probe-based trust region search with actual KL
            # V projection
            v_success, v_alpha = trust_region_search(
                model,
                lambda: attn.v_proj.weight.data[s_h:e_h, :],  # getter
                lambda w: attn.v_proj.weight.data[s_h:e_h, :].copy_(w),  # setter
                rah_v,
                config.rank,
                alpha_cap,  # Use HSS-scaled alpha
                config.delta_kl,
                probe
            )
            
            # O projection
            o_success, o_alpha = trust_region_search(
                model,
                lambda: attn.o_proj.weight.data[:, s_h:e_h],  # getter
                lambda w: attn.o_proj.weight.data[:, s_h:e_h].copy_(w),  # setter
                rah_o,
                config.rank,
                alpha_cap,  # Use HSS-scaled alpha
                config.delta_kl,
                probe
            )
            
            # Apply updates if both succeed
            if v_success and o_success and v_alpha > 0:
                total_calibrated += 1
                
                # Conservative Q/K update (optional, usually skip)
                if config.attn_sim_weight < 0.3:  # Only if heavily weight-based
                    q_alpha = v_alpha / 3  # Much smaller
                    k_alpha = v_alpha / 3
                    
                    attn.q_proj.weight.data[s_h:e_h, :] = lora_update(
                        attn.q_proj.weight.data[s_h:e_h, :], rah_q, 
                        config.rank, q_alpha
                    )
                    attn.k_proj.weight.data[s_h:e_h, :] = lora_update(
                        attn.k_proj.weight.data[s_h:e_h, :], rah_k,
                        config.rank, k_alpha
                    )
                
                # Log successful update
                logger.log_update(
                    layer_idx, head_idx,
                    [a['idx'] for a in anchors],
                    v_alpha, 0.0  # Placeholder KL
                )
                
                print(f"  Head {head_idx}: Calibrated with α={v_alpha:.4f}, anchors={[a['idx'] for a in anchors[:3]]}")
            else:
                total_rollbacks += 1
                logger.logs['rollbacks'].append({'layer': layer_idx, 'head': head_idx})
                print(f"  Head {head_idx}: Rollback (trust region violated)")
    
    print(f"\nCalibration complete: {total_calibrated} heads updated, {total_rollbacks} rollbacks")
    return total_calibrated, total_rollbacks


# ============================================================================
# Data Loading
# ============================================================================

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
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
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


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size=1, num_workers=4):
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='RAH-LoRA: Simple but Critical Head Calibration')
    
    # Model and data
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument('--image-folder', type=str, default='./playground/data/eval/textvqa/train_images')
    parser.add_argument('--question-file', type=str, default='./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl')
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--conv-mode', type=str, default='vicuna_v1')
    parser.add_argument('--use-cc3m', action='store_true', help='Use CC3M dataset for calibration')
    parser.add_argument('--cc3m-path', type=str, default='/home/diml/data/hj/cc3m', help='Path to CC3M dataset')
    parser.add_argument('--cc3m-samples', type=int, default=None, help='Number of CC3M samples (default: use isal-samples)')
    
    # Calibration parameters
    parser.add_argument('--start-layer', type=int, default=12)
    parser.add_argument('--end-layer', type=int, default=23)
    parser.add_argument('--calib-samples', type=int, default=100)
    parser.add_argument('--isal-samples', type=int, default=50)
    
    # Core hyperparameters (only 3 main knobs)
    parser.add_argument('--budget-ratio', type=float, default=0.10, help='B: proportion of heads to calibrate')
    parser.add_argument('--rank', type=int, default=4, help='r: LoRA rank')
    parser.add_argument('--delta-kl', type=float, default=0.05, help='δ: KL trust region')
    
    # Model saving options
    parser.add_argument('--save-model', action='store_true', help='Save calibrated model to disk')
    parser.add_argument('--model-save-path', type=str, default=None, help='Custom path to save model (default: output-path/calibrated_model)')
    
    # Advanced parameters (usually fixed)
    parser.add_argument('--anchor-pool-percentile', type=float, default=0.6)
    parser.add_argument('--num-anchors', type=int, default=3)
    parser.add_argument('--attn-sim-weight', type=float, default=0.6)
    parser.add_argument('--trim-ratio', type=float, default=0.1)
    parser.add_argument('--alpha-max', type=float, default=0.15)
    parser.add_argument('--enable-head-sensitivity', action='store_true')
    parser.add_argument('--sensitivity-thresh', type=float, default=0.01)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialize logger
    logger = CalibrationLogger(os.path.join(args.output_path, 'calibration_log.json'))
    
    # Create config
    config = CalibrationConfig(
        budget_ratio=args.budget_ratio,
        delta_kl=args.delta_kl,
        rank=args.rank,
        anchor_pool_percentile=args.anchor_pool_percentile,
        num_anchors=args.num_anchors,
        attn_sim_weight=args.attn_sim_weight,
        trim_ratio=args.trim_ratio,
        alpha_max=args.alpha_max,
        enable_head_sensitivity=args.enable_head_sensitivity,
        sensitivity_thresh=args.sensitivity_thresh
    )
    logger.log_config(config)
    
    print("=" * 80)
    print("RAH-LoRA: Representative Anchor Head Low-Rank Adaptation")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Budget B: {config.budget_ratio:.1%}")
    print(f"  Trust region δ: {config.delta_kl}")
    print(f"  LoRA rank r: {config.rank}")
    print(f"  Layers: {args.start_layer}-{args.end_layer}")
    print("=" * 80)
    
    # Load model
    disable_torch_init()
    print(f"Loading model from {args.model_path}...")
    
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    # Debug: Check CUDA environment
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    
    # Load model with auto device mapping for multi-GPU
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device_map="auto"
    )
    
    # Check where model actually loaded
    print(f"Model device after loading: {next(model.parameters()).device}")
    
    # Load calibration data
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    
    # Sample for I-SAL calculation
    if args.isal_samples > 0:
        isal_questions = questions[:args.isal_samples]
    else:
        isal_questions = questions
    
    print(f"Using {len(isal_questions)} samples for I-SAL calculation")
    
    # Create data loader
    if args.use_cc3m:
        # Use cc3m_samples if specified, otherwise use isal_samples
        cc3m_samples = args.cc3m_samples if hasattr(args, 'cc3m_samples') and args.cc3m_samples else args.isal_samples
        print(f"Using CC3M dataset from {args.cc3m_path} with {cc3m_samples} samples")
        from cc3m_dataloader import create_cc3m_dataloader
        data_loader = create_cc3m_dataloader(
            tokenizer=tokenizer,
            image_processor=image_processor,
            model_config=model.config,
            batch_size=1,  # Keep batch_size=1 for I-SAL accuracy
            num_samples=cc3m_samples,
            num_workers=0  # Use 0 to avoid deadlock with probe batch
        )
    else:
        conv_mode = args.conv_mode
        if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in conv_mode:
            conv_mode = conv_mode + '_mmtag'
        
        data_loader = create_data_loader(
            isal_questions, args.image_folder, tokenizer, image_processor, 
            model.config, conv_mode
        )
    
    # Patch 6: Simplified device handling - always use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Calculate I-SAL scores and attention prototypes
    print("\n1. Calculating I-SAL_TI scores and attention prototypes...")
    isal_scores, attn_proto_mean = calculate_isal_scores(
        model, tokenizer, image_processor, data_loader, device, args.isal_samples
    )
    
    # Save I-SAL scores
    isal_path = os.path.join(args.output_path, 'isal_scores.json')
    with open(isal_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in isal_scores.items()}, f, indent=2)
    print(f"I-SAL scores saved to {isal_path}")
    
    # Step 2: (Optional) Calculate image sensitivity
    if config.enable_head_sensitivity:
        print("\n2. Calculating image sensitivity...")
        sensitivity_data = calculate_image_sensitivity(model, data_loader, device, num_samples=20)
        print(f"Global entropy delta: {sensitivity_data.get('global_delta', 0):.4f}")
    
    # Create probe batch for KL estimation  
    print("\n3. Creating probe batch for trust region...")
    probe = make_probe_batch(data_loader, device, n=3)
    
    # Step 4: Calibrate heads
    print("\n4. Calibrating heads with RAH-LoRA...")
    total_calibrated, total_rollbacks = calibrate_heads(
        model, isal_scores, attn_proto_mean, config, logger, probe, args.start_layer, args.end_layer
    )
    
    # Step 4: Save calibrated model (if requested)
    if args.save_model:
        model_save_path = args.model_save_path or os.path.join(args.output_path, 'calibrated_model')
        print(f"\n4. Saving calibrated model to {model_save_path}...")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Model saved successfully!")
    else:
        print(f"\n4. Skipping model save (use --save-model to save)")
    
    # Save logs
    logger.logs['metrics'] = {
        'total_calibrated': total_calibrated,
        'total_rollbacks': total_rollbacks,
        'rollback_rate': total_rollbacks / (total_calibrated + total_rollbacks) if (total_calibrated + total_rollbacks) > 0 else 0
    }
    logger.save()
    
    print("\n" + "=" * 80)
    print("Calibration Complete!")
    print(f"  Total heads calibrated: {total_calibrated}")
    print(f"  Total rollbacks: {total_rollbacks}")
    print(f"  Logs saved to: {os.path.join(args.output_path, 'calibration_log.json')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
"""
Simple-but-Critical MLLM Head Calibration (RAH-LoRA)
Representative Anchor Head Low-Rank Adaptation

핵심: "유사한 '좋은' 이웃의 공통 저차 방향만 살짝 더한다"
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass, asdict

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import copy
import torch.nn as nn


@dataclass
class CalibrationConfig:
    """Calibration hyperparameters"""
    budget_ratio: float = 0.20  # B: proportion of heads to calibrate per layer (increased from 0.10)
    delta_kl: float = 0.10      # δ: KL divergence trust region (relaxed from 0.05)
    rank: int = 4                # r: LoRA rank
    anchor_pool_percentile: float = 0.6  # q: top percentile for anchor candidates
    num_anchors: int = 3         # m: number of anchor heads
    attn_sim_weight: float = 0.5  # ρ: weight for attention similarity vs weight similarity
    trim_ratio: float = 0.1      # trimmed mean ratio
    alpha_max: float = 0.30       # maximum calibration strength (increased from 0.15)
    enable_head_sensitivity: bool = False  # use image sensitivity gating (disabled by default)
    sensitivity_thresh: float = 0.01  # entropy difference threshold
    # New CAF and safety parameters
    use_caf: bool = False        # Use Counterfactual Ablation Filter
    caf_threshold: float = 0.0   # Only calibrate if delta_loss < threshold
    min_anchor_sim: float = 0.05  # Minimum similarity for anchor selection (reduced from 0.2)
    alpha_from_effect: bool = False  # Scale alpha by effect size
    

class CalibrationLogger:
    """Structured logging for calibration process"""
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.logs = {
            'config': {},
            'head_analysis': [],
            'calibration_updates': [],
            'metrics': {},
            'rollbacks': []
        }
    
    def log_config(self, config: CalibrationConfig):
        self.logs['config'] = asdict(config)
    
    def log_head_analysis(self, layer: int, head: int, data: dict):
        self.logs['head_analysis'].append({
            'layer': layer,
            'head': head,
            **data
        })
    
    def log_update(self, layer: int, head: int, anchors: list, alpha: float, delta_kl: float):
        self.logs['calibration_updates'].append({
            'layer': layer,
            'head': head,
            'anchors': anchors,
            'alpha': alpha,
            'delta_kl': delta_kl
        })
    
    def save(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.logs, f, indent=2)


# ============================================================================
# HSS (Head Sensitivity Score) Components
# ============================================================================

class GateWrapper(nn.Module):
    """Per-head scalar gate wrapper for attention modules"""
    def __init__(self, attn_module):
        super().__init__()
        self.attn = attn_module
        self.num_heads = attn_module.num_heads
        self.head_dim = attn_module.head_dim
        device = attn_module.q_proj.weight.device
        self.gate = nn.Parameter(torch.ones(self.num_heads, device=device, dtype=torch.float32),
                                 requires_grad=True)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False, **kw):
        out = self.attn(hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kw)
        x = out[0]  # [B, T, H]
        B, T, H = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        g = self.gate.to(x.dtype).view(1, 1, -1, 1)
        x = (x * g).view(B, T, H)
        return (x,) + out[1:]


def compute_hss_onestep(model, dataloader, layers, device='cuda', num_samples=10):
    """Compute Head Sensitivity Scores in one step using entropy loss"""
    model.eval()
    
    # 1) Freeze all parameters
    for p in model.parameters():
        p.requires_grad_(False)
    
    # 2) Wrap layers with GateWrapper
    originals = {}
    wrappers = {}
    for l in layers:
        layer = model.model.layers[l]
        originals[l] = layer.self_attn
        wrappers[l] = GateWrapper(layer.self_attn).to(device)
        layer.self_attn = wrappers[l]
        wrappers[l].gate.requires_grad_(True)
    
    # 3) Accumulation buffers
    acc = {l: torch.zeros(wrappers[l].num_heads, device=device, dtype=torch.float32) for l in layers}
    denom = {l: 0.0 for l in layers}
    
    # Process limited samples
    sample_count = 0
    for batch in dataloader:
        if sample_count >= num_samples:
            break
            
        # Handle both tuple and dict batch formats
        if isinstance(batch, tuple):
            input_ids = batch[0].to(device)
            images = batch[1].to(device, dtype=torch.float16) if len(batch) > 1 else None
        else:
            input_ids = batch['input_ids'].to(device)
            images = batch['images'].to(device, dtype=torch.float16) if 'images' in batch else None
        
        # Zero gate gradients
        for w in wrappers.values():
            if w.gate.grad is not None:
                w.gate.grad.zero_()
        
        try:
            # Disable gradient computation for everything except gates
            with torch.enable_grad():
                if images is not None:
                    out = model(input_ids, images=images, return_dict=True, output_hidden_states=True)
                else:
                    out = model(input_ids, return_dict=True, output_hidden_states=True)
                logits = out.logits
                
                # Use a simpler loss that doesn't require gradients on logits
                # Just sum of logits as a proxy for sensitivity
                loss = logits.sum()
            
            # Backward only through gates
            loss.backward()
            
            # Collect gradients with layer normalization
            for l in layers:
                g = wrappers[l].gate.grad
                if g is not None:
                    # Use absolute gradient values
                    grad_vals = g.detach().abs()
                    
                    # Normalize by layer's hidden state norm if available
                    if hasattr(out, 'hidden_states') and out.hidden_states is not None:
                        h = out.hidden_states[l+1]  # embeddings offset
                        hn = (h.float().norm(dim=-1).mean().item() + 1e-8)
                    else:
                        hn = 1.0
                    
                    acc[l] += (grad_vals / hn)
                    denom[l] += 1.0
                    
                    # Debug print
                    if sample_count == 0:
                        print(f"  Layer {l}: grad mean={grad_vals.mean():.6f}, max={grad_vals.max():.6f}")
        except Exception as e:
            print(f"HSS computation error for sample {sample_count}: {e}")
            continue
        
        sample_count += 1
    
    # 4) Average and restore
    hss = {}
    for l in layers:
        if denom[l] > 0:
            scores = (acc[l] / denom[l]).cpu().numpy()
            # Replace NaN with small values
            scores = np.nan_to_num(scores, nan=1e-6)
            hss[l] = scores.tolist()
        else:
            # If no gradients were collected, use uniform small values
            num_heads = wrappers[l].num_heads
            hss[l] = [1e-6] * num_heads
            
        model.model.layers[l].self_attn = originals[l]  # restore
        del wrappers[l]
    
    torch.cuda.empty_cache()
    return hss


# ============================================================================
# CAF (Counterfactual Ablation Filter) Functions
# ============================================================================

def compute_caf_onestep(model, dataloader, layers, device='cuda',
                        num_samples=32, loss_mode='margin',
                        use_second_order=True, curvature_weight=0.5):
    """
    One-step CAF: per-head ∆L_ablate ≈ -dL/dγ  (+ 0.5 * Fisher optional)
    Much faster than iterating through each head
    Returns: Dict[layer_idx -> List[delta_loss_estimates]]
    """
    model.eval()
    
    # 1) Freeze all parameters
    for p in model.parameters():
        p.requires_grad_(False)
    
    # 2) Wrap layers with GateWrapper
    originals, wrappers = {}, {}
    for l in layers:
        layer = model.model.layers[l]
        originals[l] = layer.self_attn
        # Get the device of the layer's weights
        layer_device = layer.self_attn.q_proj.weight.device
        wrappers[l] = GateWrapper(layer.self_attn).to(layer_device)
        layer.self_attn = wrappers[l]
        wrappers[l].gate.requires_grad_(True)  # Only gates get gradients
    
    # Accumulators for gradients - use each layer's device
    g_acc = {l: torch.zeros(wrappers[l].num_heads, device=wrappers[l].gate.device, dtype=torch.float32) for l in layers}
    g2_acc = {l: torch.zeros(wrappers[l].num_heads, device=wrappers[l].gate.device, dtype=torch.float32) for l in layers}
    count = 0
    
    # Process samples
    for batch in dataloader:
        if count >= num_samples:
            break
            
        # Handle batch format
        if isinstance(batch, tuple):
            input_ids = batch[0].to(device, non_blocking=True)
            images = batch[1].to(device, dtype=torch.float16, non_blocking=True) if len(batch) > 1 else None
            image_sizes = batch[2] if len(batch) > 2 else None
        else:
            input_ids = batch['input_ids'].to(device)
            images = batch.get('images', None)
            if images is not None:
                images = images.to(device, dtype=torch.float16)
            image_sizes = batch.get('image_sizes', None)
        
        # Zero gate gradients
        for w in wrappers.values():
            if w.gate.grad is not None:
                w.gate.grad.zero_()
        
        with torch.enable_grad():
            if images is not None:
                out = model(input_ids, images=images, return_dict=True)
            else:
                out = model(input_ids, return_dict=True)
            logits = out.logits
            
            # Label-free loss computation
            if loss_mode == 'entropy':
                probs = torch.softmax(logits, dim=-1)
                loss = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            else:  # 'margin' (more stable)
                probs = torch.softmax(logits, dim=-1)
                top2 = torch.topk(probs, k=2, dim=-1).values
                loss = -(torch.log(top2[..., 0] + 1e-8) - torch.log(top2[..., 1] + 1e-8)).mean()
        
        loss.backward()
        
        # Collect gradients
        for l in layers:
            g = wrappers[l].gate.grad
            if g is not None:
                g_acc[l] += g.detach()
                g2_acc[l] += g.detach().pow(2)
        
        count += 1
    
    # 3) Compute CAF scores and restore
    caf = {}
    for l in layers:
        if count == 0:
            n = wrappers[l].num_heads
            caf[l] = [0.0] * n
        else:
            g_mean = (g_acc[l] / count)
            g2_mean = (g2_acc[l] / count)
            
            # Layer-wise z-normalization for stability
            def z_norm(x):
                m = x.mean()
                s = x.std() + 1e-8
                return (x - m) / s
            
            # First-order approximation: -dL/dγ
            delta = -z_norm(g_mean)
            
            if use_second_order:
                # Add curvature correction (Fisher diagonal)
                curv = z_norm(g2_mean)
                delta = delta + curvature_weight * 0.5 * curv
            
            caf[l] = delta.detach().cpu().tolist()
        
        # Restore original attention
        model.model.layers[l].self_attn = originals[l]
        del wrappers[l]
    
    torch.cuda.empty_cache()
    return caf


# Keep old function for backwards compatibility but redirect to new one
def compute_caf_scores(model, data_loader, layers, device='cuda', num_samples=48, bootstrap_rounds=2):
    """
    Legacy CAF function - redirects to faster one-step version
    """
    print("[CAF] Using one-step approximation for speed")
    return compute_caf_onestep(model, data_loader, layers, device, 
                               num_samples=num_samples, loss_mode='margin',
                               use_second_order=True, curvature_weight=0.5)


def compute_fisher_scores(model, data_loader, layers, device='cuda', num_samples=10):
    """
    Compute Fisher Information scores as HSS replacement
    Returns: Dict[layer_idx -> List[head_scores]]
    """
    model.eval()
    
    # Use GateWrapper but compute Fisher instead
    originals = {}
    wrappers = {}
    for l in layers:
        layer = model.model.layers[l]
        originals[l] = layer.self_attn
        # Get the device of the layer's weights
        layer_device = layer.self_attn.q_proj.weight.device
        wrappers[l] = GateWrapper(layer.self_attn).to(layer_device)
        layer.self_attn = wrappers[l]
        wrappers[l].gate.requires_grad_(True)
    
    # Accumulation for Fisher (squared gradients) - use each layer's device
    fisher = {l: torch.zeros(wrappers[l].num_heads, device=wrappers[l].gate.device, dtype=torch.float32) for l in layers}
    denom = {l: 0.0 for l in layers}
    
    sample_count = 0
    for batch in data_loader:
        if sample_count >= num_samples:
            break
        
        # Handle batch format
        if isinstance(batch, tuple):
            input_ids = batch[0].to(device)
            images = batch[1].to(device, dtype=torch.float16) if len(batch) > 1 else None
        else:
            input_ids = batch['input_ids'].to(device)
            images = batch['images'].to(device, dtype=torch.float16) if 'images' in batch else None
        
        # Zero gradients
        for w in wrappers.values():
            if w.gate.grad is not None:
                w.gate.grad.zero_()
        
        # Forward pass
        with torch.enable_grad():
            if images is not None:
                out = model(input_ids, images=images, return_dict=True)
            else:
                out = model(input_ids, return_dict=True)
            
            # Use log probability of most likely token
            logits = out.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            max_log_probs = log_probs.max(dim=-1).values
            loss = max_log_probs.sum()
        
        # Backward
        loss.backward()
        
        # Accumulate squared gradients (Fisher)
        for l in layers:
            g = wrappers[l].gate.grad
            if g is not None:
                fisher[l] += g.detach().pow(2)
                denom[l] += 1.0
        
        sample_count += 1
    
    # Average and restore
    fisher_scores = {}
    for l in layers:
        if denom[l] > 0:
            scores = (fisher[l] / denom[l]).cpu().numpy()
            # Normalize within layer
            mean = scores.mean()
            std = scores.std() + 1e-8
            scores = (scores - mean) / std
            fisher_scores[l] = scores.tolist()
        else:
            num_heads = wrappers[l].num_heads
            fisher_scores[l] = [0.0] * num_heads
        
        model.model.layers[l].self_attn = originals[l]
        del wrappers[l]
    
    torch.cuda.empty_cache()
    return fisher_scores


# ============================================================================
# Core Functions
# ============================================================================

def calculate_isal_scores(model, tokenizer, image_processor, data_loader, device, num_samples=50):
    """
    Calculate I-SAL_TI scores: Text↔Image bidirectional attention mass
    Returns: Tuple(isal_scores, attn_proto_mean)
    """
    model.eval()
    all_attention_scores = {}
    # per-layer list of per-head prototypes (concatenated t2i/i2t, pooled to 128-D)
    attn_protos = {l: [] for l in range(len(model.model.layers))}
    
    # Initialize storage for each layer
    for layer_idx in range(len(model.model.layers)):
        all_attention_scores[layer_idx] = []
    
    with torch.no_grad():
        sample_count = 0
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Calculating I-SAL scores")):
            if sample_count >= num_samples:
                break
            
            # Handle different batch formats (CC3M vs TextVQA)
            if isinstance(batch_data, dict):
                # CC3M format
                input_ids = batch_data['input_ids']
                images = batch_data['images']
                image_sizes = None  # CC3M doesn't use image_sizes
            else:
                # TextVQA format (tuple)
                input_ids, image_tensor, image_sizes, indices = batch_data
                images = image_tensor
            
            # Patch 6: Simplified device handling
            input_ids = input_ids.to(device, non_blocking=True)
            images = images.to(device, dtype=torch.float16, non_blocking=True)
            
            # Find image token positions (may be at index 0)
            image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
            image_token_indices = torch.where(image_token_mask[0])[0]
            
            # Forward pass with attention outputs
            outputs = model(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                output_attentions=True,
                return_dict=True
            )
            
            # Actual seq length after image expansion
            seq_len = outputs.attentions[0].shape[-1]
            
            # Calculate image token range
            if len(image_token_indices) > 0:
                image_start = int(image_token_indices[0].item())
                # Try to infer number of vision tokens from vision tower (fallback: 576)
                num_image_tokens = 576
                vt = getattr(getattr(model.model, "vision_tower", None), "vision_tower", None)
                if vt is not None and hasattr(vt, "config"):
                    patch = getattr(vt.config, "patch_size", 14)
                    imgsz = getattr(vt.config, "image_size", 336)
                    num_image_tokens = (imgsz // patch) ** 2
                image_end = min(image_start + num_image_tokens, seq_len)
                
                # Process attention weights for each layer
                for layer_idx in range(len(outputs.attentions)):
                    attn_weights = outputs.attentions[layer_idx]  # [batch, num_heads, seq_len, seq_len]
                    
                    # Build text token indices: both sides of image span
                    text_idx_parts = []
                    if image_start > 0:
                        text_idx_parts.append(torch.arange(0, image_start, device=attn_weights.device))
                    if image_end < seq_len:
                        text_idx_parts.append(torch.arange(image_end, seq_len, device=attn_weights.device))
                    
                    if len(text_idx_parts) > 0:
                        text_idx = torch.cat(text_idx_parts)
                        # Text→Image and Image→Text, then average
                        t2i = attn_weights[0, :, text_idx, image_start:image_end].mean(dim=(1, 2))
                        i2t = attn_weights[0, :, image_start:image_end, text_idx].mean(dim=(1, 2))
                        score = 0.5 * (t2i + i2t)  # [num_heads]
                        all_attention_scores[layer_idx].append(score.detach().cpu().numpy())
                        
                        # build 1D prototypes (adaptive pool to 64 + concat = 128-D)
                        def _pool(v):
                            v = v.float().unsqueeze(0)  # [1, num_heads]
                            v = F.adaptive_avg_pool1d(v, 64).squeeze(0)
                            return F.normalize(v, p=2, dim=0)
                        p_t2i = _pool(t2i)
                        p_i2t = _pool(i2t)
                        attn_protos[layer_idx].append(torch.cat([p_t2i, p_i2t]).cpu().numpy())
            
            sample_count += 1
    
    # Average scores across samples
    isal_scores = {}
    attn_proto_mean = {}
    for layer_idx in all_attention_scores:
        if all_attention_scores[layer_idx]:
            scores = np.stack(all_attention_scores[layer_idx])
            isal_scores[layer_idx] = scores.mean(axis=0)
            protos = np.stack(attn_protos[layer_idx])  # [samples, heads, 128]
            attn_proto_mean[layer_idx] = protos.mean(axis=0)  # [heads, 128]
        else:
            isal_scores[layer_idx] = np.zeros(model.model.layers[0].self_attn.num_heads)
            attn_proto_mean[layer_idx] = np.zeros((model.model.layers[0].self_attn.num_heads, 128))
    
    return isal_scores, attn_proto_mean


def calculate_image_sensitivity(model, data_loader, device, num_samples=20):
    """
    Calculate image sensitivity for each head
    Returns heads that are primarily text-focused (low image dependency)
    """
    model.eval()
    head_sensitivities = {}
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Calculating image sensitivity")):
            if batch_idx >= num_samples:
                break
            
            # Handle different batch formats (CC3M vs TextVQA)
            if isinstance(batch_data, dict):
                # CC3M format
                input_ids = batch_data['input_ids']
                images = batch_data['images']
            else:
                # TextVQA format (tuple)
                input_ids, image_tensor, image_sizes, indices = batch_data
                images = image_tensor
            
            # Patch 6: Simplified device handling
            input_ids = input_ids.to(device)
            images = images.to(device, dtype=torch.float16)
            
            # Forward with normal images
            outputs_normal = model(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                output_attentions=False,
                return_dict=True
            )
            
            # Forward with blank images
            blank_images = torch.zeros_like(images)
            outputs_blank = model(
                input_ids,
                images=blank_images,
                image_sizes=image_sizes,
                output_attentions=False,
                return_dict=True
            )
            
            # Calculate entropy difference
            probs_normal = F.softmax(outputs_normal.logits, dim=-1)
            probs_blank = F.softmax(outputs_blank.logits, dim=-1)
            
            entropy_normal = -(probs_normal * torch.log(probs_normal + 1e-8)).sum(dim=-1)
            entropy_blank = -(probs_blank * torch.log(probs_blank + 1e-8)).sum(dim=-1)
            
            delta_entropy = (entropy_normal - entropy_blank).abs().mean()
            
            # Store for analysis (simplified - would need per-head in full implementation)
            if batch_idx == 0:
                head_sensitivities['global_delta'] = delta_entropy.item()
    
    return head_sensitivities


def attention_similarity(attn_patterns_h: torch.Tensor, attn_patterns_a: torch.Tensor) -> float:
    """
    Calculate functional similarity based on attention patterns
    """
    # Flatten and normalize
    h = attn_patterns_h.flatten()
    a = attn_patterns_a.flatten()
    h = F.normalize(h.unsqueeze(0), p=2, dim=1).squeeze()
    a = F.normalize(a.unsqueeze(0), p=2, dim=1).squeeze()
    
    return F.cosine_similarity(h.unsqueeze(0), a.unsqueeze(0), dim=1).item()


def trimmed_weighted_mean(tensors: List[torch.Tensor],
                          weights: List[float],
                          trim_ratio: float) -> torch.Tensor:
    """
    L2 deviation-based trimming followed by weighted mean
    """
    if len(tensors) == 1:
        return tensors[0]
    
    X = torch.stack(tensors)  # [n, ...]
    W = torch.tensor(weights, device=X.device, dtype=X.dtype)  # [n]
    
    # Calculate deviations
    mu = X.mean(dim=0, keepdim=True)
    dev = ((X - mu)**2).flatten(1).sum(dim=1).cpu().numpy()
    
    n = len(tensors)
    k = int(n * trim_ratio)
    if k > 0 and n > 2*k:
        keep = np.argsort(dev)[k:n-k]
        X, W = X[keep], W[keep]
    
    W = torch.clamp(W, min=0)
    W = W / (W.sum() + 1e-8)
    shape = (W.shape[0],) + (1,) * (X.dim()-1)
    return (X * W.view(*shape)).sum(dim=0)


def lora_update(W: torch.Tensor, target: torch.Tensor, rank: int = 4, alpha: float = 0.1, use_full_update: bool = False) -> torch.Tensor:
    """
    LoRA-style low-rank update or full update
    """
    delta = (target - W).to(torch.float32)
    
    if use_full_update or rank >= min(W.shape):
        # Direct full-rank update (no approximation)
        return (W + alpha * delta.to(W.dtype))
    else:
        # SVD for low-rank approximation
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        r = min(rank, U.size(1), Vh.size(0))
        
        # Truncated reconstruction
        low_rank_delta = (U[:, :r] * S[:r]) @ Vh[:r, :]
        
        return (W + alpha * low_rank_delta.to(W.dtype))


def make_probe_batch(dataloader, device, n=3):
    """Create small probe batch for KL estimation"""
    it = iter(dataloader)
    batch_data = next(it)
    
    # Handle different batch formats
    if isinstance(batch_data, dict):
        # CC3M format
        probe = {
            "input_ids": batch_data['input_ids'].to(device),
            "images": batch_data['images'].to(device, dtype=torch.float16),
        }
        if 'image_sizes' in batch_data:
            probe['image_sizes'] = batch_data['image_sizes']
    else:
        # TextVQA format (tuple)
        inputs, images, image_sizes, _ = batch_data
        probe = {
            "input_ids": inputs.to(device),
            "images": images.to(device, dtype=torch.float16),
            "image_sizes": image_sizes
        }
    
    return probe

@torch.no_grad()
def estimate_kl(model, probe, logits_before=None):
    """Estimate KL divergence using probe batch"""
    if logits_before is None:
        logits_before = model(**probe).logits
    logits_after = model(**probe).logits
    p = torch.log_softmax(logits_before.float(), dim=-1).exp()
    ql = torch.log_softmax(logits_after.float(), dim=-1)
    kl = (p * (p.log() - ql)).sum(dim=-1).mean().item()
    return kl, logits_before

def trust_region_search(model, W_ref_getter, W_apply, target: torch.Tensor,
                       rank: int, alpha_max: float, delta_kl: float,
                       probe, use_full_update: bool = False) -> Tuple[bool, float]:
    """
    Trust region line search with actual KL divergence
    W_ref_getter(): returns current ref weight tensor (view)
    W_apply(new): in-place apply new weight
    """
    alpha = alpha_max
    kl_before, logits_before = estimate_kl(model, probe, logits_before=None)
    
    for _ in range(4):
        W_orig = W_ref_getter().clone()
        W_new = lora_update(W_orig, target, rank, alpha, use_full_update=use_full_update)
        W_apply(W_new)  # apply new weight
        
        kl_after, _ = estimate_kl(model, probe, logits_before)
        dkl = max(kl_after - kl_before, 0.0)
        
        # Debug: print KL values on first iteration
        if alpha >= alpha_max * 0.99:  # First iteration check
            print(f"    Debug: KL before={kl_before:.6f}, after={kl_after:.6f}, delta={dkl:.6f}, threshold={delta_kl:.6f}")
        
        if dkl <= delta_kl:
            return True, alpha
        
        # rollback and reduce alpha
        W_apply(W_orig)
        alpha *= 0.5
    
    return False, 0.0


# ============================================================================
# Main Calibration Function
# ============================================================================

def calibrate_heads(model, isal_scores: Dict, attn_proto_mean: Dict,
                   config: CalibrationConfig, logger: CalibrationLogger,
                   probe, start_layer: int = 12, end_layer: int = 23,
                   hss_scores: Dict[int, List[float]] = None,
                   caf_scores: Dict[int, List[float]] = None):
    """
    Main calibration function implementing RAH-LoRA with attention patterns
    """
    device = next(model.parameters()).device
    total_calibrated = 0
    total_rollbacks = 0
    sensitivity_eps = 1e-8  # Patch 5: sensitivity gating threshold
    
    for layer_idx in range(start_layer, end_layer + 1):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        
        # Get layer I-SAL scores and attention prototypes
        layer_scores = isal_scores.get(layer_idx, np.zeros(num_heads))
        layer_protos = attn_proto_mean.get(layer_idx, np.zeros((num_heads, 128)))
        
        # Patch 5: Skip heads with negligible I-SAL scores
        active_heads = [h for h in range(num_heads) if layer_scores[h] > sensitivity_eps]
        if len(active_heads) == 0:
            print(f"Layer {layer_idx}: All heads have negligible I-SAL scores, skipping")
            continue
        
        # 1) Select calibration targets (bottom budget_ratio% of active heads)
        active_scores = [layer_scores[h] for h in active_heads]
        threshold = np.percentile(active_scores, config.budget_ratio * 100)
        target_heads = [h for h in active_heads if layer_scores[h] <= threshold]
        
        print(f"  Initial selection: {len(active_heads)} active → {len(target_heads)} targets (budget={config.budget_ratio:.1%})")
        print(f"  I-SAL threshold: {threshold:.6f}, range: [{min(active_scores):.6f}, {max(active_scores):.6f}]")
        
        # HSS filtering: Keep only heads with high sensitivity
        initial_targets = len(target_heads)
        if hss_scores is not None and layer_idx in hss_scores:
            hs = np.abs(np.array(hss_scores[layer_idx]))  # [num_heads]
            # Only filter if we have meaningful HSS values (not all zeros/nans)
            if np.sum(hs) > 1e-5 and not np.all(np.isnan(hs)):
                median = np.median(hs[hs > 0]) if np.any(hs > 0) else 0
                if median > 0:
                    before_hss = len(target_heads)
                    target_heads = [h for h in target_heads if hs[h] >= median]  # Keep only sensitive heads
                    print(f"  HSS filtering: {before_hss} → {len(target_heads)} (median={median:.6f})")
        
        # CAF filtering: Only keep heads that harm performance when removed
        if config.use_caf and caf_scores is not None and layer_idx in caf_scores:
            caf = np.array(caf_scores[layer_idx])
            before_caf = len(target_heads)
            # Keep only heads where delta_loss < threshold (negative = helps to remove)
            target_heads = [h for h in target_heads if caf[h] < config.caf_threshold]
            print(f"  CAF filtering: {before_caf} → {len(target_heads)} heads (threshold={config.caf_threshold})")
        
        # 2) Identify anchor candidates (top anchor_pool_percentile%)
        anchor_threshold = np.percentile(layer_scores, config.anchor_pool_percentile * 100)
        
        print(f"\nLayer {layer_idx}: Calibrating {len(target_heads)} heads")
        
        for head_idx in target_heads:
            # Skip if head has negligible I-SAL (Patch 5)
            if layer_scores[head_idx] <= sensitivity_eps:
                print(f"  Head {head_idx}: Skipping due to negligible I-SAL score")
                continue
            
            # Find anchor candidates
            candidates = []
            
            for anchor_idx in active_heads:  # Only consider active heads as anchors
                if anchor_idx == head_idx:
                    continue
                    
                if layer_scores[anchor_idx] >= anchor_threshold:
                    # Compute weight similarity
                    s_h, e_h = head_idx * head_dim, (head_idx + 1) * head_dim
                    s_a, e_a = anchor_idx * head_dim, (anchor_idx + 1) * head_dim
                    
                    v_h = attn.v_proj.weight.data[s_h:e_h, :].flatten()
                    v_a = attn.v_proj.weight.data[s_a:e_a, :].flatten()
                    o_h = attn.o_proj.weight.data[:, s_h:e_h].flatten()
                    o_a = attn.o_proj.weight.data[:, s_a:e_a].flatten()
                    
                    # Compute cosine similarity properly for 1D tensors
                    sim_v = F.cosine_similarity(v_h.view(1, -1), v_a.view(1, -1), dim=1).item()
                    sim_o = F.cosine_similarity(o_h.view(1, -1), o_a.view(1, -1), dim=1).item()
                    weight_sim = 0.7 * sim_v + 0.3 * sim_o
                    
                    # Patch 2: Add attention pattern similarity
                    proto_h = torch.tensor(layer_protos[head_idx], dtype=torch.float32)
                    proto_a = torch.tensor(layer_protos[anchor_idx], dtype=torch.float32)
                    # Compute cosine similarity properly for 1D tensors
                    attn_sim = F.cosine_similarity(proto_h.view(1, -1), proto_a.view(1, -1), dim=1).item()
                    
                    # Combine weight and attention similarities
                    combined_sim = config.attn_sim_weight * attn_sim + (1 - config.attn_sim_weight) * weight_sim
                    
                    candidates.append({
                        'idx': anchor_idx,
                        'isal': layer_scores[anchor_idx],
                        'similarity': combined_sim
                    })
            
            # Select top-m anchors with minimum similarity check
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            # Filter by minimum similarity
            before_sim_filter = len(candidates)
            candidates = [c for c in candidates if c['similarity'] >= config.min_anchor_sim]
            anchors = candidates[:config.num_anchors]
            
            if not anchors:
                if before_sim_filter > 0:
                    max_sim = max(c['similarity'] for c in candidates[:before_sim_filter]) if before_sim_filter > 0 else 0
                    print(f"  Head {head_idx}: No anchors found ({before_sim_filter} candidates, max_sim={max_sim:.3f} < {config.min_anchor_sim})")
                else:
                    print(f"  Head {head_idx}: No anchor candidates found")
                continue
            
            # 3) Compute RAH (Representative Anchor Head) using trimmed weighted mean
            anchor_weights_v = []
            anchor_weights_o = []
            anchor_weights_q = []
            anchor_weights_k = []
            anchor_scores = []
            
            for anchor in anchors:
                a_idx = anchor['idx']
                s_a, e_a = a_idx * head_dim, (a_idx + 1) * head_dim
                
                # Weight by similarity * I-SAL
                weight = anchor['similarity'] * anchor['isal']
                anchor_scores.append(weight)
                
                anchor_weights_v.append(attn.v_proj.weight.data[s_a:e_a, :])
                anchor_weights_o.append(attn.o_proj.weight.data[:, s_a:e_a])
                anchor_weights_q.append(attn.q_proj.weight.data[s_a:e_a, :])
                anchor_weights_k.append(attn.k_proj.weight.data[s_a:e_a, :])
            
            # Patch 3: Use trimmed_weighted_mean
            rah_v = trimmed_weighted_mean(anchor_weights_v, anchor_scores, config.trim_ratio)
            rah_o = trimmed_weighted_mean(anchor_weights_o, anchor_scores, config.trim_ratio)
            rah_q = trimmed_weighted_mean(anchor_weights_q, anchor_scores, config.trim_ratio)
            rah_k = trimmed_weighted_mean(anchor_weights_k, anchor_scores, config.trim_ratio)
            
            # 4) Apply calibration with trust region
            s_h, e_h = head_idx * head_dim, (head_idx + 1) * head_dim
            
            # Alpha scaling based on effect size
            alpha_cap = config.alpha_max
            
            # CAF-based scaling if enabled
            if config.alpha_from_effect and caf_scores is not None and layer_idx in caf_scores:
                caf = np.array(caf_scores[layer_idx])
                # Use negative delta (positive = head is harmful)
                effect = max(-caf[head_idx], 0)
                # Normalize by median effect in layer
                layer_effects = np.abs(caf)
                median_effect = np.median(layer_effects[layer_effects > 0]) if np.any(layer_effects > 0) else 1e-6
                alpha_cap = config.alpha_max * float(min(effect / (median_effect + 1e-8), 1.0))
                
            # HSS-scaled alpha_max (if no CAF scaling)
            elif hss_scores is not None and layer_idx in hss_scores:
                hs = np.abs(np.array(hss_scores[layer_idx]))
                # Only scale if we have meaningful HSS values
                if np.sum(hs) > 1e-5 and not np.all(np.isnan(hs)) and np.any(hs > 0):
                    # Normalize by 90th percentile in layer
                    kappa = np.percentile(hs[hs > 0], 90) if np.any(hs > 0) else 1.0
                    if kappa > 0:
                        alpha_cap = config.alpha_max * float(min(hs[head_idx]/kappa, 1.0))
            
            # Patch 4: Use probe-based trust region search with actual KL
            # V projection
            v_success, v_alpha = trust_region_search(
                model,
                lambda: attn.v_proj.weight.data[s_h:e_h, :],  # getter
                lambda w: attn.v_proj.weight.data[s_h:e_h, :].copy_(w),  # setter
                rah_v,
                config.rank,
                alpha_cap,  # Use HSS-scaled alpha
                config.delta_kl,
                probe
            )
            
            # O projection
            o_success, o_alpha = trust_region_search(
                model,
                lambda: attn.o_proj.weight.data[:, s_h:e_h],  # getter
                lambda w: attn.o_proj.weight.data[:, s_h:e_h].copy_(w),  # setter
                rah_o,
                config.rank,
                alpha_cap,  # Use HSS-scaled alpha
                config.delta_kl,
                probe
            )
            
            # Apply updates if both succeed
            if v_success and o_success and v_alpha > 0:
                total_calibrated += 1
                
                # Conservative Q/K update (optional, usually skip)
                if config.attn_sim_weight < 0.3:  # Only if heavily weight-based
                    q_alpha = v_alpha / 3  # Much smaller
                    k_alpha = v_alpha / 3
                    
                    attn.q_proj.weight.data[s_h:e_h, :] = lora_update(
                        attn.q_proj.weight.data[s_h:e_h, :], rah_q, 
                        config.rank, q_alpha
                    )
                    attn.k_proj.weight.data[s_h:e_h, :] = lora_update(
                        attn.k_proj.weight.data[s_h:e_h, :], rah_k,
                        config.rank, k_alpha
                    )
                
                # Log successful update
                logger.log_update(
                    layer_idx, head_idx,
                    [a['idx'] for a in anchors],
                    v_alpha, 0.0  # Placeholder KL
                )
                
                print(f"  Head {head_idx}: Calibrated with α={v_alpha:.4f}, anchors={[a['idx'] for a in anchors[:3]]}")
            else:
                total_rollbacks += 1
                logger.logs['rollbacks'].append({'layer': layer_idx, 'head': head_idx})
                print(f"  Head {head_idx}: Rollback (trust region violated)")
    
    print(f"\nCalibration complete: {total_calibrated} heads updated, {total_rollbacks} rollbacks")
    return total_calibrated, total_rollbacks


# ============================================================================
# Data Loading
# ============================================================================

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
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
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


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size=1, num_workers=4):
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='RAH-LoRA: Simple but Critical Head Calibration')
    
    # Model and data
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument('--image-folder', type=str, default='./playground/data/eval/textvqa/train_images')
    parser.add_argument('--question-file', type=str, default='./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl')
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--conv-mode', type=str, default='vicuna_v1')
    parser.add_argument('--use-cc3m', action='store_true', help='Use CC3M dataset for calibration')
    parser.add_argument('--cc3m-path', type=str, default='/home/diml/data/hj/cc3m', help='Path to CC3M dataset')
    parser.add_argument('--cc3m-samples', type=int, default=None, help='Number of CC3M samples (default: use isal-samples)')
    
    # Calibration parameters
    parser.add_argument('--start-layer', type=int, default=12)
    parser.add_argument('--end-layer', type=int, default=23)
    parser.add_argument('--calib-samples', type=int, default=100)
    parser.add_argument('--isal-samples', type=int, default=50)
    
    # Core hyperparameters (only 3 main knobs)
    parser.add_argument('--budget-ratio', type=float, default=0.10, help='B: proportion of heads to calibrate')
    parser.add_argument('--rank', type=int, default=4, help='r: LoRA rank')
    parser.add_argument('--delta-kl', type=float, default=0.05, help='δ: KL trust region')
    
    # Model saving options
    parser.add_argument('--save-model', action='store_true', help='Save calibrated model to disk')
    parser.add_argument('--model-save-path', type=str, default=None, help='Custom path to save model (default: output-path/calibrated_model)')
    
    # Advanced parameters (usually fixed)
    parser.add_argument('--anchor-pool-percentile', type=float, default=0.6)
    parser.add_argument('--num-anchors', type=int, default=3)
    parser.add_argument('--attn-sim-weight', type=float, default=0.6)
    parser.add_argument('--trim-ratio', type=float, default=0.1)
    parser.add_argument('--alpha-max', type=float, default=0.15)
    parser.add_argument('--enable-head-sensitivity', action='store_true')
    parser.add_argument('--sensitivity-thresh', type=float, default=0.01)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialize logger
    logger = CalibrationLogger(os.path.join(args.output_path, 'calibration_log.json'))
    
    # Create config
    config = CalibrationConfig(
        budget_ratio=args.budget_ratio,
        delta_kl=args.delta_kl,
        rank=args.rank,
        anchor_pool_percentile=args.anchor_pool_percentile,
        num_anchors=args.num_anchors,
        attn_sim_weight=args.attn_sim_weight,
        trim_ratio=args.trim_ratio,
        alpha_max=args.alpha_max,
        enable_head_sensitivity=args.enable_head_sensitivity,
        sensitivity_thresh=args.sensitivity_thresh
    )
    logger.log_config(config)
    
    print("=" * 80)
    print("RAH-LoRA: Representative Anchor Head Low-Rank Adaptation")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Budget B: {config.budget_ratio:.1%}")
    print(f"  Trust region δ: {config.delta_kl}")
    print(f"  LoRA rank r: {config.rank}")
    print(f"  Layers: {args.start_layer}-{args.end_layer}")
    print("=" * 80)
    
    # Load model
    disable_torch_init()
    print(f"Loading model from {args.model_path}...")
    
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    # Debug: Check CUDA environment
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    
    # Load model with auto device mapping for multi-GPU
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device_map="auto"
    )
    
    # Check where model actually loaded
    print(f"Model device after loading: {next(model.parameters()).device}")
    
    # Load calibration data
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    
    # Sample for I-SAL calculation
    if args.isal_samples > 0:
        isal_questions = questions[:args.isal_samples]
    else:
        isal_questions = questions
    
    print(f"Using {len(isal_questions)} samples for I-SAL calculation")
    
    # Create data loader
    if args.use_cc3m:
        # Use cc3m_samples if specified, otherwise use isal_samples
        cc3m_samples = args.cc3m_samples if hasattr(args, 'cc3m_samples') and args.cc3m_samples else args.isal_samples
        print(f"Using CC3M dataset from {args.cc3m_path} with {cc3m_samples} samples")
        from cc3m_dataloader import create_cc3m_dataloader
        data_loader = create_cc3m_dataloader(
            tokenizer=tokenizer,
            image_processor=image_processor,
            model_config=model.config,
            batch_size=1,  # Keep batch_size=1 for I-SAL accuracy
            num_samples=cc3m_samples,
            num_workers=0  # Use 0 to avoid deadlock with probe batch
        )
    else:
        conv_mode = args.conv_mode
        if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in conv_mode:
            conv_mode = conv_mode + '_mmtag'
        
        data_loader = create_data_loader(
            isal_questions, args.image_folder, tokenizer, image_processor, 
            model.config, conv_mode
        )
    
    # # Patch 6: Simplified device handling - always use cuda if available
    # device = torch.cuda()
    # print(f"Using device: {device}")
    
    # Step 1: Calculate I-SAL scores and attention prototypes
    print("\n1. Calculating I-SAL_TI scores and attention prototypes...")
    isal_scores, attn_proto_mean = calculate_isal_scores(
        model, tokenizer, image_processor, data_loader, device, args.isal_samples
    )
    
    # Save I-SAL scores
    isal_path = os.path.join(args.output_path, 'isal_scores.json')
    with open(isal_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in isal_scores.items()}, f, indent=2)
    print(f"I-SAL scores saved to {isal_path}")
    
    # Step 2: (Optional) Calculate image sensitivity
    if config.enable_head_sensitivity:
        print("\n2. Calculating image sensitivity...")
        sensitivity_data = calculate_image_sensitivity(model, data_loader, device, num_samples=20)
        print(f"Global entropy delta: {sensitivity_data.get('global_delta', 0):.4f}")
    
    # Create probe batch for KL estimation  
    print("\n3. Creating probe batch for trust region...")
    probe = make_probe_batch(data_loader, device, n=3)
    
    # Step 4: Calibrate heads
    print("\n4. Calibrating heads with RAH-LoRA...")
    total_calibrated, total_rollbacks = calibrate_heads(
        model, isal_scores, attn_proto_mean, config, logger, probe, args.start_layer, args.end_layer
    )
    
    # Step 4: Save calibrated model (if requested)
    if args.save_model:
        model_save_path = args.model_save_path or os.path.join(args.output_path, 'calibrated_model')
        print(f"\n4. Saving calibrated model to {model_save_path}...")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Model saved successfully!")
    else:
        print(f"\n4. Skipping model save (use --save-model to save)")
    
    # Save logs
    logger.logs['metrics'] = {
        'total_calibrated': total_calibrated,
        'total_rollbacks': total_rollbacks,
        'rollback_rate': total_rollbacks / (total_calibrated + total_rollbacks) if (total_calibrated + total_rollbacks) > 0 else 0
    }
    logger.save()
    
    print("\n" + "=" * 80)
    print("Calibration Complete!")
    print(f"  Total heads calibrated: {total_calibrated}")
    print(f"  Total rollbacks: {total_rollbacks}")
    print(f"  Logs saved to: {os.path.join(args.output_path, 'calibration_log.json')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
