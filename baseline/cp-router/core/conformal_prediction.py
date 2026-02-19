"""
Conformal Prediction (CP) 核心实现
论文 Section 2 & 3.1

CP 步骤:
1. 定义 score function: S(x, y) = 1 - f(y)
2. 在校准集上计算 nonconformity scores
3. 计算分位数阈值 q_hat
4. 对测试样本构建预测集
"""
import numpy as np
import math
from typing import List, Dict, Tuple, Optional


def compute_nonconformity_scores(
    probs: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """
    计算校准集的 nonconformity scores
    
    Args:
        probs: shape (n, num_choices), softmax 概率
        labels: shape (n,), 正确答案的索引 (0-based)
    
    Returns:
        scores: shape (n,), 每个样本的 nonconformity score
    """
    # S(x, y) = 1 - f(y), 论文 Section 2.1
    scores = np.array([1.0 - probs[i, labels[i]] for i in range(len(labels))])
    return scores


def compute_quantile_threshold(
    scores: np.ndarray,
    alpha: float
) -> float:
    """
    计算 CP 分位数阈值 q_hat
    
    论文 Section 2.3:
    q_hat = ceil((n+1)(1-alpha)) / n 分位数
    
    Args:
        scores: 校准集的 nonconformity scores
        alpha: 用户定义的错误率
    
    Returns:
        q_hat: 分位数阈值
    """
    n = len(scores)
    # 计算分位数级别
    quantile_level = math.ceil((n + 1) * (1 - alpha)) / n
    quantile_level = min(quantile_level, 1.0)  # 确保不超过 1
    q_hat = np.quantile(scores, quantile_level)
    return q_hat


def construct_prediction_set(
    probs: np.ndarray,
    q_hat: float,
    num_choices: int = 4
) -> List[List[int]]:
    """
    构建预测集
    
    论文 Eq. 1:
    C(x) = {y ∈ Y : S(x, y) ≤ q_hat}
    
    Args:
        probs: shape (n, num_choices), softmax 概率
        q_hat: 分位数阈值
        num_choices: 选项数量
    
    Returns:
        prediction_sets: 每个样本的预测集 (包含的选项索引列表)
    """
    prediction_sets = []
    for i in range(len(probs)):
        pred_set = []
        for j in range(num_choices):
            score = 1.0 - probs[i, j]
            if score <= q_hat:
                pred_set.append(j)
        # 如果预测集为空，至少包含概率最高的选项
        if len(pred_set) == 0:
            pred_set = [np.argmax(probs[i])]
        prediction_sets.append(pred_set)
    return prediction_sets


def compute_prediction_set_sizes(
    prediction_sets: List[List[int]]
) -> np.ndarray:
    """计算每个预测集的大小"""
    return np.array([len(s) for s in prediction_sets])


def compute_apss(prediction_sets: List[List[int]]) -> float:
    """
    计算 Average Prediction Set Size (APSS)
    论文 Figure 4b
    """
    sizes = compute_prediction_set_sizes(prediction_sets)
    return float(np.mean(sizes))


def compute_coverage(
    prediction_sets: List[List[int]],
    labels: np.ndarray
) -> float:
    """
    计算经验覆盖率
    论文 Theorem 2.1: P(Y_test ∈ C(X_test)) ≥ 1 - α
    """
    covered = sum(
        1 for i, ps in enumerate(prediction_sets)
        if labels[i] in ps
    )
    return covered / len(labels)
