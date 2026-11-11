"""从官方 utils.py 提取的数学置信度聚合与 GMM 路由。

高均值分量对应高置信；代码阈值比较的是低均值分量的后验概率。
小于 10 个样本时保留官方回退逻辑；不引入训练标签或外部评分模型。
"""
import re
import numpy as np
import typing as _t
from typing import List, Tuple, Optional
from sklearn.mixture import GaussianMixture
MIN_N_FOR_GMM = 10

def get_step_reliability(
    tokens: _t.Optional[_t.List[str]],  # List of token strings, needed for math_only/number_only modes
    token_reliabilities_list: _t.List[float],  # List of per-token reliability scores (e.g., R or R_eu values)
    mode: str,  # Aggregation mode: "avg", "sum", "min", "min_k_avg", "number_only_avg", "math_only_avg", "math_only_min_k_avg"
    k_val: int = 5, # k value for "min_k" modes, defaulting to 5
    return_selected_indices: bool = False
) -> _t.Optional[float]:
    """
    Calculates step-level reliability by aggregating per-token reliabilities based on the specified mode.
    """
    if not token_reliabilities_list:
        return -np.inf

    active_reliabilities = token_reliabilities_list
    processing_mode = mode

    if mode.startswith("number_only") or mode.startswith("math_only"):
        if tokens is None or len(tokens) != len(token_reliabilities_list):
            print(f"Warning: 'tokens' list must be provided and match length of 'token_reliabilities_list' for '{mode}'. Skipping token-specific filtering.")
            # Fallback to non-token-specific part of the mode if possible, or just use full list if mode is just "number_only_avg" etc.
            if "_" in mode: # e.g. "number_only_avg" -> "avg"
                processing_mode = mode.split('_')[-1]
            else: # Mode was just "number_only" or "math_only", which isn't a final aggregation.
                return -np.inf # Or apply a default aggregation like 'avg' to the full list? For now, None.
        else:
            filtered_rels = []
            if mode.startswith("number_only"):
                processing_mode = mode.replace("number_only_", "") # e.g., "avg"
                number_mask = [t.isdigit() for t in tokens]
                filtered_rels = [
                    rel for rel, is_num in zip(token_reliabilities_list, number_mask) if is_num
                ]
            elif mode.startswith("math_only"): 
                # logic for math_only mode
                processing_mode = mode.replace("math_only_", "") # e.g., "avg" or "min_k_avg"
                
                latex_cmd_re = re.compile(r"\\[A-Za-z]+")
                begin_env_re = re.compile(r"\\begin\{([A-Za-z*]+)\}")
                end_env_re = re.compile(r"\\end\{([A-Za-z*]+)\}")
                inline_math = False
                next_is_math = False
                env_depth = 0
                math_mask = []
                for tk_str in tokens:
                    if tk_str.count("$") % 2 == 1: inline_math = not inline_math
                    if begin_env_re.search(tk_str): env_depth += 1
                    if end_env_re.search(tk_str) and env_depth > 0: env_depth -= 1
                    if '{' in tk_str: env_depth += 1
                    if '}' in tk_str: env_depth -= 1
                    if '\\(' in tk_str: env_depth += 1
                    if '\\)' in tk_str: env_depth -= 1
                    inside_env = env_depth > 0
                    inside_math = inline_math or inside_env
                    has_digit = any(ch.isdigit() for ch in tk_str)
                    # Note: operator list might need to be more comprehensive or configurable
                    is_operator = tk_str.strip() in {"+", "-", "*", "/", "=", "^", "_", "%", "(", ")", "[", "]", "{", "}", "\\times", "\\cdot", "\\pm", "\\frac", "\\sqrt"}
                    is_latex_cmd = bool(latex_cmd_re.match(tk_str))
                    is_math_token = inside_math or has_digit or is_operator or is_latex_cmd

                    if next_is_math:
                        if tk_str.startswith(" "):
                            next_is_math = False
                        else:
                            is_math_token = True
                            if " " in tk_str:
                                next_is_math = False
                            
                    if tk_str.endswith("\\"):
                        next_is_math = True
                          
                    str_to_strip =["(", ")", "[", "]", "{", "}", "\\"]
                    for s in str_to_strip:
                        tk_str = tk_str.replace(s, "")
                    tk_str = tk_str.strip()
                    if len(tk_str) == 0:
                        is_math_token = False
                    
                    math_mask.append(is_math_token)
                
                filtered_rels = [
                    rel for rel, is_math in zip(token_reliabilities_list, math_mask) if is_math
                ]

            if not filtered_rels:
                return -np.inf  # No relevant tokens found
            active_reliabilities = filtered_rels
                        
    if not active_reliabilities: # Check again after potential filtering
        return -np.inf

    if processing_mode == "avg":
        return np.mean(active_reliabilities)
    elif processing_mode == "sum":
        return np.sum(active_reliabilities)
    elif processing_mode == "min":
        return np.min(active_reliabilities)
    elif processing_mode == "min_k_avg":
        if k_val <= 0:
            # print("Warning: k_val for 'min_k_avg' must be positive.")
            return -np.inf
        # Sort in ascending order to get the smallest k
        sorted_reliabilities = sorted(active_reliabilities)
        actual_k = min(k_val, len(sorted_reliabilities))
        if actual_k == 0:
            return -np.inf
        if return_selected_indices:
            return np.mean(sorted_reliabilities[:actual_k]), sorted(active_reliabilities).index(sorted_reliabilities[:actual_k])
        else:
            return np.mean(sorted_reliabilities[:actual_k])
    else:
        # print(f"Warning: Unknown processing mode '{processing_mode}' for get_step_reliability.")
        raise ValueError(f"Invalid or unsupported reliability aggregation sub-mode: {processing_mode} (derived from original mode: {mode})")

def fit_mixture_model(values: List[float], use_prior: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[GaussianMixture]]:
    nan_arr = np.array([np.nan, np.nan])

    if use_prior:
        # should not reach here
        raise ValueError("use_prior should be False")

    # while fitting the gmm, discard the -inf values
    values = [v for v in values if v != -np.inf]
    values_arr = np.array(values).reshape(-1, 1)
        
    if len(values_arr) < MIN_N_FOR_GMM:
        return nan_arr, nan_arr, nan_arr, None
    
    # IQR-based outlier detection
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 5 * iqr # very lenient bound: only filter out the 'trivial' outliers
    upper_bound = q3 + 5 * iqr
        
    # Remove outliers
    values = [v for v in values if lower_bound <= v <= upper_bound]
    values_arr = np.array(values).reshape(-1, 1)
    
    gmm_model: Optional[GaussianMixture] = None 
    gmm = GaussianMixture(n_components=2, covariance_type='full', init_params='kmeans', 
                            n_init=10, random_state=0, tol=1e-4, max_iter=100)
    
    if len(values_arr) < 2:
        return nan_arr, nan_arr, nan_arr, None
    
    gmm.fit(values_arr)
    gmm_model = gmm 
    
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()
    
    sort_indices = np.argsort(means)
    sorted_means = means[sort_indices]
    sorted_stds = stds[sort_indices]
    sorted_weights = weights[sort_indices]
    return sorted_means, sorted_stds, sorted_weights, gmm_model

def get_gmm_responsibility(idx_values: List[Tuple[int, float]], gmm_model: GaussianMixture, threshold: float = 0.5, rejection_threshold: float = 0.9, target_gmm: bool = False, use_prior: bool = False, reverse_routing: bool = False) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
    if not idx_values:
        return [], [], [] 
      
    if not use_prior:
        if len(idx_values) < MIN_N_FOR_GMM:
            idx = [item[0] for item in idx_values]
            return [], idx, []
        
    tmp_values = [v for _, v in idx_values]
    values_without_inf = [v for v in tmp_values if np.isfinite(v)]
    # 全部为空数学步骤时，不能对空列表调用 min；回退为低置信。
    if not values_without_inf:
        return [], [idx for idx, _ in idx_values], []
        
    min_value = min(values_without_inf)
    values =  [v if v != -np.inf else min_value for v in tmp_values]
    
    if gmm_model is None:
        return None, None, None

    try:
        # TODO: handle the -inf case
        if not values: 
            return [], [], []
        values_for_proba = np.array(values).reshape(-1, 1)
        probs = gmm_model.predict_proba(values_for_proba) 
    except Exception:
        return None, None, None    
    
    if probs.shape[0] != len(idx_values):
        return None, None, None

    internal_means_flat = gmm_model.means_.flatten()
    if len(internal_means_flat) != 2:
        return None, None, None

    idx_comp_smaller_mean_internal = np.argmin(internal_means_flat)
    idx_comp_larger_mean_internal = 1 - idx_comp_smaller_mean_internal 
    
    classified_as_wrong_like = []
    classified_as_correct_like = []
    to_difficult_for_target = []

    for k, item_tuple in enumerate(idx_values):
        original_idx = item_tuple[0]
        
        prob_for_smaller_mean_comp = probs[k, idx_comp_smaller_mean_internal]
        prob_for_larger_mean_comp  = probs[k, idx_comp_larger_mean_internal]
        
        if not target_gmm:
            if not reverse_routing:
                if prob_for_smaller_mean_comp > threshold:
                    classified_as_wrong_like.append(original_idx)
                else:
                    classified_as_correct_like.append(original_idx)
            else:
                if prob_for_smaller_mean_comp < threshold:
                    classified_as_wrong_like.append(original_idx)
                else:
                    classified_as_correct_like.append(original_idx)
        else:
            if prob_for_smaller_mean_comp > rejection_threshold:
                to_difficult_for_target.append(original_idx)
            elif prob_for_smaller_mean_comp > threshold:
                classified_as_wrong_like.append(original_idx)
            else:
                classified_as_correct_like.append(original_idx)
                                
    return classified_as_correct_like, classified_as_wrong_like, to_difficult_for_target

def route_prompts(args,
                  idx_value_pairs_draft: List[Tuple[int, float]], 
                  idx_value_pairs_target: List[Tuple[int, float]],
                  draft_gmm_threshold: float = 0.5,
                  target_gmm_threshold: float = 0.5,
                 ) -> Tuple[List[int], List[int], List[int]]:
    
    if (len(idx_value_pairs_draft) + len(idx_value_pairs_target)) < MIN_N_FOR_GMM:
        target_indices = []
        for item in idx_value_pairs_draft:
            target_indices.append(item[0])
        for item in idx_value_pairs_target:
            target_indices.append(item[0])
        return [], target_indices, []

    draft_values = [item[1] for item in idx_value_pairs_draft]
    target_values = [item[1] for item in idx_value_pairs_target]

    draft_original_indices = [item[0] for item in idx_value_pairs_draft]
    target_original_indices = [item[0] for item in idx_value_pairs_target]

    draft_correct_like_indices, draft_wrong_like_indices = [], []
    target_correct_like_indices, target_wrong_like_indices = [], []
    to_difficult_for_target = []
    
    if idx_value_pairs_draft:
        _, _, draft_weights, gmm_model_draft = fit_mixture_model(draft_values)
        
        # if the values are paratemrized into a single component, send all the items to the target
        if draft_weights[0] > 0.99 or draft_weights[1] > 0.99:
            draft_wrong_like_indices = draft_original_indices
            draft_correct_like_indices = []
        else:
            if gmm_model_draft:
                d_cl, d_wl, _ = get_gmm_responsibility(idx_value_pairs_draft, gmm_model_draft, draft_gmm_threshold, use_prior=False, reverse_routing=False)
                if d_cl is not None: 
                    draft_correct_like_indices = d_cl
                if d_wl is not None: 
                    draft_wrong_like_indices = d_wl
                
                if d_cl is None and d_wl is None and draft_original_indices:
                    print("Warning: GMM responsibility failed for draft items. Defaulting to target.") # Consider logging
                    draft_wrong_like_indices = list(set(draft_wrong_like_indices + draft_original_indices))
            else: 
                draft_wrong_like_indices = draft_original_indices
    
    t_td = None
    target_correct_like_indices, target_wrong_like_indices = [], []
    if idx_value_pairs_target:
        _, _, target_weights, gmm_model_target = fit_mixture_model(target_values)
        
        # if the values are paratemrized into a single component, send all the items to the target
        if target_weights[0] > 0.99 or target_weights[1] > 0.99:
            target_wrong_like_indices = target_original_indices
            target_correct_like_indices = []
        else:
            if gmm_model_target:
                t_cl, t_wl, t_td = get_gmm_responsibility(idx_value_pairs_target, gmm_model_target, target_gmm_threshold, target_gmm=True, use_prior=False, reverse_routing=False)
                if t_cl is not None: 
                    target_correct_like_indices = t_cl
                if t_wl is not None: 
                    target_wrong_like_indices = t_wl

                if t_cl is None and t_wl is None and target_original_indices:
                    # print("Warning: GMM responsibility failed for target items. Defaulting to target.") # Consider logging
                    target_wrong_like_indices = list(set(target_wrong_like_indices + target_original_indices))
            else: 
                target_wrong_like_indices = target_original_indices

    potential_next_step_draft = draft_correct_like_indices + target_correct_like_indices
    potential_next_step_target = draft_wrong_like_indices + target_wrong_like_indices
    
    final_next_step_draft_set = set(potential_next_step_draft)
    final_next_step_target_set = set(potential_next_step_target)
    
    true_final_next_step_draft = list(final_next_step_draft_set)
    true_final_next_step_target = list(final_next_step_target_set - final_next_step_draft_set) 
    
    if t_td is not None:
        to_difficult_for_target = list(set(t_td) - set(true_final_next_step_draft) - set(true_final_next_step_target))
    else:
        to_difficult_for_target = []
    
    return true_final_next_step_draft, true_final_next_step_target, to_difficult_for_target
