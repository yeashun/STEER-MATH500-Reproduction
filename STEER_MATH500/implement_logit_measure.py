"""官方 R_eu 计算函数；移除导入时的 Gemma tokenizer 和分析脚本。

本实验固定 k_top=1。在正常正值分支，R_eu 约为最大 logit + 1。
保留原始计算和异常分支，便于对照官方结果。
"""
import torch
import numpy as np
from scipy.special import digamma

def dirichlet_au_eu(logits, k_top_values, only_eu=False):
    """
    logits: torch.Tensor [vocab]
    k_top_values: list of integers
    returns dict {k_top: {"au": AU, "eu": EU, "alphas": alpha_k_list}}
    """
    orig_k_top_values = None
    if type(k_top_values) == int:
        orig_k_top_values = k_top_values
        k_top_values = [k_top_values]
        
    if type(logits) == np.ndarray:
        logits = torch.from_numpy(logits)
    
    results = {}
    max_k_top = max(k_top_values)
    max_k_top = min(max_k_top, logits.shape[-1])

    if max_k_top <= 0:
        # Return default values if no valid k_top
        for k_top in k_top_values:
             results[k_top] = {"au": 0.0, "eu": 0.0, "alphas": []}
        return results

    # Get top logits once for the maximum k
    all_vals, _ = torch.topk(logits, max_k_top)
    all_alphas = all_vals # Using logits directly as alphas
    all_alphas, _ = torch.sort(all_alphas, descending=True)

    for k_top in k_top_values:
        k_top = min(k_top, max_k_top) # Ensure k_top doesn't exceed available logits
        if k_top <= 0:
            results[k_top] = {"au": 0.0, "eu": 0.0, "alphas": []}
            continue

        # import pdb; pdb.set_trace()

        # Slice the pre-computed values
        alphas = all_alphas[:k_top]
        alpha0 = alphas.sum()

        if alpha0 <= 1e-6:
            results[k_top] = {"au": 0.0, "eu": 0.0, "alphas": []}
            continue

        if only_eu:
            eu = k_top / (alpha0 + k_top)  
            results[k_top] = {"eu": eu, "alphas": alphas.cpu().tolist()}
            continue

        # AU (expected entropy)
        # Ensure calculations handle potential non-finite values gracefully
        with np.errstate(invalid='ignore'): # Ignore potential warnings in digamma
            psi_alpha_plus1 = digamma((alphas + 1).cpu().numpy())
            psi_alpha0_plus1 = digamma(float(alpha0 + 1))

        # Replace NaNs or Infs resulting from digamma if necessary
        psi_alpha_plus1 = np.nan_to_num(psi_alpha_plus1, nan=0.0, posinf=0.0, neginf=0.0)
        psi_alpha0_plus1 = np.nan_to_num(psi_alpha0_plus1, nan=0.0, posinf=0.0, neginf=0.0)

        if not np.isfinite(psi_alpha0_plus1):
             psi_alpha0_plus1 = 0 # Fallback

        au_terms = alphas / alpha0 * torch.tensor(psi_alpha_plus1 - psi_alpha0_plus1, device=alphas.device)
        au = -torch.sum(au_terms)

        # EU
        eu = k_top / (alpha0 + k_top)  

        au_float = float(au) if torch.isfinite(au) else 0.0
        eu_float = float(eu) if torch.isfinite(eu) else 0.0

        results[k_top] = {"au": au_float, "eu": eu_float, "alphas": alphas.cpu().tolist()}

    if orig_k_top_values is not None:
        return results[orig_k_top_values]
    return results
