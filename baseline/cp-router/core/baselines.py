"""
Baseline 路由方法实现
论文 Appendix B
"""
import numpy as np
from typing import List, Tuple
import random


def random_routing(
    n_samples: int,
    threshold: float = 0.5,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    随机路由
    threshold: 路由到 LRM 的比例
    """
    random.seed(seed)
    llm_indices = []
    lrm_indices = []
    for i in range(n_samples):
        if random.random() > threshold:
            llm_indices.append(i)
        else:
            lrm_indices.append(i)
    return llm_indices, lrm_indices


def top1_probability_routing(
    probs: np.ndarray,
    threshold: float = 0.7
) -> Tuple[List[int], List[int]]:
    """
    Top-1 概率路由
    如果最高概率 > threshold, 用 LLM; 否则用 LRM
    """
    max_probs = np.max(probs, axis=1)
    llm_indices = list(np.where(max_probs >= threshold)[0])
    lrm_indices = list(np.where(max_probs < threshold)[0])
    return llm_indices, lrm_indices


def entropy_routing(
    probs: np.ndarray,
    threshold: float = 1.0
) -> Tuple[List[int], List[int]]:
    """
    响应熵路由
    如果熵 > threshold, 用 LRM; 否则用 LLM
    """
    # 计算每个样本的熵
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    llm_indices = list(np.where(entropy <= threshold)[0])
    lrm_indices = list(np.where(entropy > threshold)[0])
    return llm_indices, lrm_indices


def compute_accuracy_and_trr(
    llm_indices: List[int],
    lrm_indices: List[int],
    llm_predictions: np.ndarray,
    lrm_predictions: np.ndarray,
    labels: np.ndarray,
    llm_tokens_per_sample: int = 10,
    lrm_tokens_per_sample: int = 500
) -> dict:
    """计算准确率和 TRR"""
    n = len(labels)
    correct = 0
    total_tokens = 0
    
    for i in llm_indices:
        if llm_predictions[i] == labels[i]:
            correct += 1
        total_tokens += llm_tokens_per_sample
    
    for i in lrm_indices:
        if lrm_predictions[i] == labels[i]:
            correct += 1
        total_tokens += lrm_tokens_per_sample
    
    lrm_total = n * lrm_tokens_per_sample
    
    acc = correct / n if n > 0 else 0
    trr = 1.0 - (total_tokens / lrm_total) if lrm_total > 0 else 0
    
    return {"accuracy": acc, "trr": trr, "total_tokens": total_tokens}
