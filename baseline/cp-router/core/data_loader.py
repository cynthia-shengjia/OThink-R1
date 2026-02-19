"""
数据集加载器
支持论文中使用的所有 MCQA 数据集
"""
import os
import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset


OPTION_LETTERS = ['A', 'B', 'C', 'D', 'E']


def format_mcqa_prompt(
    question: str,
    options: List[str],
    system_prompt: str = "",
    prompt_template: str = "default"
) -> str:
    """
    格式化 MCQA 提示
    
    Args:
        question: 问题文本
        options: 选项列表
        system_prompt: 系统提示
        prompt_template: 提示模板
    """
    option_text = "\n".join(
        f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
    )
    
    prompt = f"{question}\n\n{option_text}\n\nAnswer:"
    
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"
    
    return prompt


def load_mmlu_stem(
    subset: str = "elementary_mathematics",
    split: str = "test",
    data_dir: Optional[str] = None
) -> Dict:
    """
    加载 MMLU-STEM 数据集
    
    subset: elementary_mathematics, high_school_statistics, 
            college_mathematics, high_school_mathematics
    """
    try:
        dataset = load_dataset("cais/mmlu", subset, split=split)
    except Exception:
        dataset = load_dataset("lukaemon/mmlu", subset, split=split)
    
    questions = []
    options_list = []
    labels = []
    
    for item in dataset:
        questions.append(item['question'])
        options_list.append(item['choices'])
        labels.append(item['answer'])  # 0-based index
    
    return {
        "questions": questions,
        "options": options_list,
        "labels": np.array(labels),
        "name": f"MMLU-{subset}",
        "num_choices": 4
    }


def load_gpqa(split: str = "train", data_dir: Optional[str] = None) -> Dict:
    """加载 GPQA 数据集"""
    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split=split)
    except Exception:
        dataset = load_dataset("Idavidrein/gpqa", split=split)
    
    questions = []
    options_list = []
    labels = []
    
    for item in dataset:
        q = item.get('question', item.get('Question', ''))
        choices = []
        correct_idx = 0
        
        # GPQA 格式可能不同
        if 'choices' in item:
            choices = item['choices']
            correct_idx = item.get('answer', 0)
        else:
            # 尝试其他格式
            for key in ['Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']:
                if key in item:
                    choices.append(item[key])
            correct_idx = 0  # Correct Answer 在第一个
            # 打乱选项
            indices = list(range(len(choices)))
            random.shuffle(indices)
            choices = [choices[i] for i in indices]
            correct_idx = indices.index(0)
        
        if choices:
            questions.append(q)
            options_list.append(choices[:4])
            labels.append(correct_idx)
    
    return {
        "questions": questions,
        "options": options_list,
        "labels": np.array(labels),
        "name": "GPQA",
        "num_choices": 4
    }


def load_logiqa(split: str = "test", data_dir: Optional[str] = None) -> Dict:
    """加载 LogiQA 数据集"""
    try:
        dataset = load_dataset("lucasmccabe/logiqa", split=split)
    except Exception:
        dataset = load_dataset("EleutherAI/logiqa", split=split)
    
    questions = []
    options_list = []
    labels = []
    
    for item in dataset:
        context = item.get('context', '')
        question = item.get('question', '')
        full_question = f"{context}\n{question}" if context else question
        
        opts = item.get('options', [])
        label = item.get('answer', item.get('label', 0))
        
        if opts:
            questions.append(full_question)
            options_list.append(opts[:4])
            labels.append(label)
    
    return {
        "questions": questions,
        "options": options_list,
        "labels": np.array(labels),
        "name": "LogiQA",
        "num_choices": 4
    }


def load_stem_mcqa(split: str = "test", data_dir: Optional[str] = None) -> Dict:
    """加载 STEM-MCQA 数据集"""
    dataset = load_dataset("thewordsmiths/stem_mcqa", split=split)
    
    questions = []
    options_list = []
    labels = []
    
    for item in dataset:
        questions.append(item['question'])
        opts = [item.get(f'option_{c}', '') for c in ['a', 'b', 'c', 'd']]
        options_list.append(opts)
        
        answer = item.get('answer', 'a')
        label_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        labels.append(label_map.get(answer.lower(), 0))
    
    return {
        "questions": questions,
        "options": options_list,
        "labels": np.array(labels),
        "name": "STEM-MCQA",
        "num_choices": 4
    }


def split_calibration_test(
    data: Dict,
    cal_ratio: float = 0.3,
    seed: int = 42
) -> Tuple[Dict, Dict]:
    """
    将数据集划分为校准集和测试集
    
    Args:
        data: 数据集字典
        cal_ratio: 校准集比例
        seed: 随机种子
    """
    n = len(data["questions"])
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    
    cal_size = int(n * cal_ratio)
    cal_indices = indices[:cal_size]
    test_indices = indices[cal_size:]
    
    cal_data = {
        "questions": [data["questions"][i] for i in cal_indices],
        "options": [data["options"][i] for i in cal_indices],
        "labels": data["labels"][cal_indices],
        "name": data["name"] + "_cal",
        "num_choices": data["num_choices"]
    }
    
    test_data = {
        "questions": [data["questions"][i] for i in test_indices],
        "options": [data["options"][i] for i in test_indices],
        "labels": data["labels"][test_indices],
        "name": data["name"] + "_test",
        "num_choices": data["num_choices"]
    }
    
    return cal_data, test_data


# 数据集注册表
DATASET_REGISTRY = {
    "mmlu_elementary_math": lambda **kw: load_mmlu_stem("elementary_mathematics", **kw),
    "mmlu_high_school_math": lambda **kw: load_mmlu_stem("high_school_mathematics", **kw),
    "mmlu_college_math": lambda **kw: load_mmlu_stem("college_mathematics", **kw),
    "mmlu_high_school_stats": lambda **kw: load_mmlu_stem("high_school_statistics", **kw),
    "gpqa": load_gpqa,
    "logiqa": load_logiqa,
    "stem_mcqa": load_stem_mcqa,
}


def load_dataset_by_name(name: str, **kwargs) -> Dict:
    """根据名称加载数据集"""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](**kwargs)
