"""
CP-Router 主路由逻辑
论文 Method 1 (Pseudocode)

路由规则:
- 预测集大小 ≤ τ → 使用 LLM
- 预测集大小 > τ → 路由到 LRM
"""
import numpy as np
import json
import os
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

from .conformal_prediction import (
    compute_nonconformity_scores,
    compute_quantile_threshold,
    construct_prediction_set,
    compute_prediction_set_sizes,
    compute_apss,
    compute_coverage
)
from .fbe import select_optimal_alpha, compute_fbe


@dataclass
class RoutingResult:
    """路由结果"""
    question_idx: int
    prediction_set: List[int]
    prediction_set_size: int
    routed_to: str  # "LLM" or "LRM"
    llm_answer: int  # LLM 的预测 (argmax)
    lrm_answer: Optional[int] = None  # LRM 的预测
    final_answer: Optional[int] = None
    correct: Optional[bool] = None
    llm_tokens: int = 0
    lrm_tokens: int = 0


@dataclass
class CPRouterMetrics:
    """评估指标"""
    accuracy: float = 0.0
    trr: float = 0.0  # Token Reduction Ratio
    u_token: float = 0.0  # Token Utility
    apss: float = 0.0
    coverage: float = 0.0
    alpha_star: float = 0.0
    fbe_score: float = 0.0
    llm_count: int = 0
    lrm_count: int = 0
    total_llm_tokens: int = 0
    total_lrm_tokens: int = 0


class CPRouter:
    """
    CP-Router: 基于 Conformal Prediction 的 LLM/LRM 路由器
    
    论文 Method 1 的完整实现
    """
    
    def __init__(
        self,
        tau: int = 1,
        beta: float = 3.0,
        num_choices: int = 4,
        alpha_candidates: Optional[np.ndarray] = None,
        fixed_alpha: Optional[float] = None
    ):
        """
        Args:
            tau: 路由阈值, 预测集大小 ≤ τ 则用 LLM
            beta: FBE 中 H_full 的权重
            num_choices: MCQA 选项数量
            alpha_candidates: 候选 α 值
            fixed_alpha: 如果指定, 则不使用 FBE 自适应, 直接用此 α
        """
        self.tau = tau
        self.beta = beta
        self.num_choices = num_choices
        self.alpha_candidates = alpha_candidates or np.arange(0.01, 1.0, 0.01)
        self.fixed_alpha = fixed_alpha
        
        # 校准后的参数
        self.alpha_star = None
        self.q_hat = None
        self.cal_scores = None
    
    def calibrate(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
        test_probs: np.ndarray
    ) -> float:
        """
        Step 1 & 2: 校准 + FBE 选择最优 α
        
        Args:
            cal_probs: 校准集 softmax 概率
            cal_labels: 校准集真实标签
            test_probs: 测试集 softmax 概率 (用于 FBE)
        
        Returns:
            alpha_star: 选择的最优 α
        """
        # 计算校准集 nonconformity scores
        self.cal_scores = compute_nonconformity_scores(cal_probs, cal_labels)
        
        if self.fixed_alpha is not None:
            self.alpha_star = self.fixed_alpha
        else:
            # FBE 自适应选择 α
            self.alpha_star, fbe_score = select_optimal_alpha(
                cal_probs=cal_probs,
                cal_labels=cal_labels,
                test_probs=test_probs,
                alpha_candidates=self.alpha_candidates,
                beta=self.beta,
                num_choices=self.num_choices
            )
            print(f"  FBE selected α* = {self.alpha_star:.4f} (FBE = {fbe_score:.4f})")
        
        # 计算最终的分位数阈值
        self.q_hat = compute_quantile_threshold(self.cal_scores, self.alpha_star)
        print(f"  Quantile threshold q_hat = {self.q_hat:.4f}")
        
        return self.alpha_star
    
    def route(
        self,
        test_probs: np.ndarray,
        test_labels: Optional[np.ndarray] = None
    ) -> Tuple[List[RoutingResult], List[int], List[int]]:
        """
        Step 3: 基于预测集大小进行路由
        
        Args:
            test_probs: 测试集 softmax 概率
            test_labels: 测试集真实标签 (可选, 用于评估)
        
        Returns:
            results: 路由结果列表
            llm_indices: 路由到 LLM 的样本索引
            lrm_indices: 路由到 LRM 的样本索引
        """
        assert self.q_hat is not None, "请先调用 calibrate()"
        
        # 构建预测集
        prediction_sets = construct_prediction_set(
            test_probs, self.q_hat, self.num_choices
        )
        
        results = []
        llm_indices = []
        lrm_indices = []
        
        for i in range(len(test_probs)):
            pred_set = prediction_sets[i]
            pred_set_size = len(pred_set)
            llm_answer = int(np.argmax(test_probs[i]))
            
            # 路由决策: 论文 Method 1, Line 25-29
            if pred_set_size <= self.tau:
                routed_to = "LLM"
                llm_indices.append(i)
            else:
                routed_to = "LRM"
                lrm_indices.append(i)
            
            result = RoutingResult(
                question_idx=i,
                prediction_set=pred_set,
                prediction_set_size=pred_set_size,
                routed_to=routed_to,
                llm_answer=llm_answer
            )
            
            if test_labels is not None:
                if routed_to == "LLM":
                    result.final_answer = llm_answer
                    result.correct = (llm_answer == test_labels[i])
            
            results.append(result)
        
        return results, llm_indices, lrm_indices
    
    def evaluate(
        self,
        results: List[RoutingResult],
        llm_only_acc: float,
        lrm_total_tokens: int
    ) -> CPRouterMetrics:
        """
        计算评估指标
        
        Args:
            results: 路由结果 (需要已填充 final_answer 和 correct)
            llm_only_acc: 仅用 LLM 的准确率
            lrm_total_tokens: 全部用 LRM 的总 token 数
        """
        metrics = CPRouterMetrics()
        
        # 准确率
        correct_count = sum(1 for r in results if r.correct)
        metrics.accuracy = correct_count / len(results) if results else 0.0
        
        # 路由统计
        metrics.llm_count = sum(1 for r in results if r.routed_to == "LLM")
        metrics.lrm_count = sum(1 for r in results if r.routed_to == "LRM")
        
        # Token 统计
        metrics.total_llm_tokens = sum(r.llm_tokens for r in results)
        metrics.total_lrm_tokens = sum(r.lrm_tokens for r in results if r.routed_to == "LRM")
        
        # TRR: Token Reduction Ratio
        # 定义: 相比全部用 LRM 节省的 token 比例
        total_router_tokens = metrics.total_llm_tokens + metrics.total_lrm_tokens
        if lrm_total_tokens > 0:
            metrics.trr = 1.0 - (total_router_tokens / lrm_total_tokens)
        
        # U_token: Token Utility (论文 Eq. 6)
        # U_token = (Acc - Acc_LLM) / (1 - TRR)
        if metrics.trr < 1.0:
            metrics.u_token = (metrics.accuracy - llm_only_acc) / (1.0 - metrics.trr)
        
        # APSS
        set_sizes = np.array([r.prediction_set_size for r in results])
        metrics.apss = float(np.mean(set_sizes))
        
        metrics.alpha_star = self.alpha_star if self.alpha_star else 0.0
        
        return metrics
