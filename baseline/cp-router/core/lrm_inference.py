"""
LRM 推理模块
对路由到 LRM 的样本进行推理
"""
import os
import torch
from typing import List, Dict, Optional, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re


class LRMInference:
    """使用 vLLM 进行 LRM 推理"""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        print(f"  Loading LRM: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=max_tokens + 2048
        )
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            skip_special_tokens=True
        )
    
    def generate(
        self,
        prompts: List[str],
        system_prompt: str = ""
    ) -> List[Dict]:
        """
        批量生成 LRM 回答
        
        Returns:
            results: [{"text": str, "tokens": int, "answer": str}, ...]
        """
        # 构建 chat 格式
        formatted_prompts = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted)
        
        # vLLM 推理
        outputs = self.llm.generate(formatted_prompts, self.sampling_params, use_tqdm=True)
        
        results = []
        for output in outputs:
            text = output.outputs[0].text
            tokens = len(output.outputs[0].token_ids)
            
            # 从回答中提取选项
            answer = self._extract_answer(text)
            
            results.append({
                "text": text,
                "tokens": tokens,
                "answer": answer
            })
        
        return results
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """从 LRM 回答中提取选项 (A/B/C/D)"""
        # 尝试多种模式
        patterns = [
            r'<ans>\s*([A-D])\s*</ans>',
            r'[Aa]nswer\s*(?:is|:)\s*\(?([A-D])\)?',
            r'\\boxed\{([A-D])\}',
            r'\b([A-D])\b\s*$',  # 最后一个选项
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # 回退: 找最后出现的 A/B/C/D
        matches = re.findall(r'\b([A-D])\b', text)
        if matches:
            return matches[-1].upper()
        
        return None
