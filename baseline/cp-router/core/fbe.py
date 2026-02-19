"""
Full and Binary Entropy (FBE) 自适应校准
论文 Section 3.2

FBE = β * H_full + H_binary

H_full: 预测集大小分布的全熵 (促进多样性)
H_binary: 单元素/非单元素的二元熵 (促进平衡)
"""
import numpy as np
from typing import List, Tuple, Optional
from .conformal_prediction import (
    compute_nonconformity_scores,
    compute_quantile_threshold,
    construct_prediction_set,
    compute_prediction_set_sizes
)


def compute_full_entropy(set_sizes: np.ndarray, num_choices: int = 4) -> float:
    """
    计算预测集大小分布的全熵 H_full
    
    H_full = -Σ p_i * log(p_i)
    其中 p_i 是大小为 i 的预测集的归一化频率
    """
    total = len(set_sizes)
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for size in range(1, num_choices + 1):
        count = np.sum(set_sizes == size)
        if count > 0:
            p = count / total
            entropy -= p * np.log(p + 1e-10)
    
    return entropy


def compute_binary_entropy(set_sizes: np.ndarray) -> float:
    """
    计算二元熵 H_binary
    
    H_binary = -(p_{s=1} * log(p_{s=1}) + p_{s≠1} * log(p_{s≠1}))
    其中 p_{s=1} 是单元素预测集的频率
    """
    total = len(set_sizes)
    if total == 0:
        return 0.0
    
    singleton_count = np.sum(set_sizes == 1)
    p_singleton = singleton_count / total
    p_non_singleton = 1.0 - p_singleton
    
    entropy = 0.0
    if p_singleton > 0:
        entropy -= p_singleton * np.log(p_singleton + 1e-10)
    if p_non_singleton > 0:
        entropy -= p_non_singleton * np.log(p_non_singleton + 1e-10)
    
    return entropy


def compute_fbe(
    set_sizes: np.ndarray,
    beta: float = 3.0,
    num_choices: int = 4
) -> float:
    """
    计算 FBE (Full and Binary Entropy)
    
    论文 Eq. 4:
    FBE = β * H_full + H_binary
    
    Args:
        set_sizes: 预测集大小数组
        beta: H_full 的权重 (论文默认 β=3)
        num_choices: 选项数量
    
    Returns:
        fbe: FBE 值
    """
    h_full = compute_full_entropy(set_sizes, num_choices)
    h_binary = compute_binary_entropy(set_sizes)
    return beta * h_full + h_binary


def select_optimal_alpha(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    alpha_candidates: Optional[np.ndarray] = None,
    beta: float = 3.0,
    num_choices: int = 4
) -> Tuple[float, float]:
    """
    通过最大化 FBE 选择最优 α
    
    论文 Eq. 5:
    α* = argmax_{α ∈ A} FBE
    
    Args:
        cal_probs: 校准集 softmax 概率, shape (n_cal, num_choices)
        cal_labels: 校准集标签, shape (n_cal,)
        test_probs: 测试集 softmax 概率, shape (n_test, num_choices)
        alpha_candidates: 候选 α 值
        beta: FBE 中 H_full 的权重
        num_choices: 选项数量
    
    Returns:
        best_alpha: 最优 α
        best_fbe: 对应的 FBE 值
    """
    if alpha_candidates is None:
        alpha_candidates = np.arange(0.01, 1.0, 0.01)
    
    # Step 1: 计算校准集的 nonconformity scores
    cal_scores = compute_nonconformity_scores(cal_probs, cal_labels)
    
    best_alpha = alpha_candidates[0]
    best_fbe = -float('inf')
    
    # Step 2: 对每个候选 α 计算 FBE
    for alpha in alpha_candidates:
        # 计算分位数阈值
        q_hat = compute_quantile_threshold(cal_scores, alpha)
        
        # 在测试集上构建预测集
        prediction_sets = construct_prediction_set(test_probs, q_hat, num_choices)
        set_sizes = compute_prediction_set_sizes(prediction_sets)
        
        # 计算 FBE
        fbe = compute_fbe(set_sizes, beta, num_choices)
        
        if fbe > best_fbe:
            best_fbe = fbe
            best_alpha = alpha
    
    return float(best_alpha), float(best_fbe)
