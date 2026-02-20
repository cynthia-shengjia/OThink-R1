"""
从 LLM 提取 MCQA 选项的 logits/概率

论文 Section 2.1 & 3.1:
提取候选选项 (A, B, C, D) 对应 token 的 logits,
然后 softmax 得到概率分布
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class LogitExtractor:
    """从 LLM 提取 MCQA 选项的 logits"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        option_tokens: Optional[List[str]] = None
    ):
        """
        Args:
            model_path: 模型路径
            device: 设备
            torch_dtype: 数据类型
            option_tokens: 选项 token 列表, 默认 ['A', 'B', 'C', 'D']
        """
        self.model_path = model_path
        self.option_tokens = option_tokens or ['A', 'B', 'C', 'D']
        
        print(f"  Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        
        # 获取选项 token 的 ID
        self.option_token_ids = []
        for opt in self.option_tokens:
            ids = self.tokenizer.encode(opt, add_special_tokens=False)
            # 取最后一个 token (有些 tokenizer 可能会产生多个 token)
            self.option_token_ids.append(ids[-1])
        
        print(f"  Option tokens: {self.option_tokens}")
        print(f"  Option token IDs: {self.option_token_ids}")
    
    def extract_logits(
        self,
        prompts: List[str],
        batch_size: int = 8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取选项的 logits 和 softmax 概率
        
        Args:
            prompts: 格式化后的 MCQA 提示列表
            batch_size: 批处理大小
        
        Returns:
            logits: shape (n, num_choices), 原始 logits
            probs: shape (n, num_choices), softmax 概率
        """
        all_logits = []
        all_probs = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 取最后一个 token 的 logits
                last_logits = outputs.logits[:, -1, :]  # (batch, vocab)
                
                # 提取选项 token 的 logits
                option_logits = last_logits[:, self.option_token_ids]  # (batch, num_choices)
                
                # Softmax
                option_probs = torch.softmax(option_logits, dim=-1)
                
                all_logits.append(option_logits.cpu().float().numpy())
                all_probs.append(option_probs.cpu().float().numpy())
        
        logits = np.concatenate(all_logits, axis=0)
        probs = np.concatenate(all_probs, axis=0)
        
        return logits, probs
    
    def extract_logits_single(self, prompt: str) -> Tuple[np.ndarray, np.ndarray]:
        """提取单个样本的 logits"""
        logits, probs = self.extract_logits([prompt], batch_size=1)
        return logits[0], probs[0]
