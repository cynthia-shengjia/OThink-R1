"""
本地数据集加载器
适配 datasets/ 目录下的 MATH, AIME, ASDIV 数据集
将开放式问答转换为 MCQA 格式 (CP-Router 需要)
"""
import os
import re
import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset


def _extract_boxed(solution: str) -> str:
    """从 MATH solution 中提取 \boxed{} 内容"""
    match = re.search(r'\boxed\{(.*)\}', solution, re.DOTALL)
    return match.group(1) if match else solution


def load_math_as_mcqa(
    data_dir: str,
    max_samples: int = 100,
    seed: int = 42
) -> Dict:
    """
    加载 MATH 数据集并转换为 MCQA 格式
    
    策略: 对每道题, 用正确答案 + 3个干扰项构成4个选项
    干扰项从同数据集其他题目的答案中随机抽取
    """
    print(f"  Loading MATH from {data_dir}...")
    dataset = load_dataset(data_dir, split="test")
    
    random.seed(seed)
    
    # 提取所有答案作为干扰项池
    all_answers = []
    problems = []
    for item in dataset:
        answer = _extract_boxed(item['solution'])
        all_answers.append(answer)
        problems.append({
            "question": item['problem'],
            "answer": answer
        })
    
    # 随机采样
    if max_samples and max_samples < len(problems):
        indices = random.sample(range(len(problems)), max_samples)
        problems = [problems[i] for i in indices]
    
    questions = []
    options_list = []
    labels = []
    
    for prob in problems:
        correct = prob["answer"]
        
        # 生成干扰项: 从答案池中随机选3个不同的
        distractors = [a for a in all_answers if a != correct]
        if len(distractors) >= 3:
            distractors = random.sample(distractors, 3)
        else:
            distractors = distractors + [f"None of the above"] * (3 - len(distractors))
        
        # 随机放置正确答案
        options = distractors + [correct]
        random.shuffle(options)
        correct_idx = options.index(correct)
        
        questions.append(prob["question"])
        options_list.append(options)
        labels.append(correct_idx)
    
    print(f"  ✅ Loaded {len(questions)} MATH problems as MCQA")
    return {
        "questions": questions,
        "options": options_list,
        "labels": np.array(labels),
        "name": "MATH",
        "num_choices": 4
    }


def load_aime_as_mcqa(
    data_dir: str,
    max_samples: int = 50,
    seed: int = 42
) -> Dict:
    """
    加载 AIME 数据集并转换为 MCQA 格式
    AIME 答案是 0-999 的整数, 生成数值干扰项
    """
    print(f"  Loading AIME from {data_dir}...")
    dataset = load_dataset(data_dir, split="train")
    
    random.seed(seed)
    
    problems = []
    for item in dataset:
        problems.append({
            "question": item['problem'],
            "answer": int(item['answer'])
        })
    
    if max_samples and max_samples < len(problems):
        problems = random.sample(problems, max_samples)
    
    questions = []
    options_list = []
    labels = []
    
    for prob in problems:
        correct = prob["answer"]
        
        # 生成数值干扰项 (在正确答案附近)
        distractors = set()
        attempts = 0
        while len(distractors) < 3 and attempts < 100:
            offset = random.choice([-3, -2, -1, 1, 2, 3, 5, 10, -5, -10])
            d = correct + offset
            if 0 <= d <= 999 and d != correct:
                distractors.add(d)
            attempts += 1
        distractors = list(distractors)[:3]
        while len(distractors) < 3:
            distractors.append(random.randint(0, 999))
        
        options = [str(d) for d in distractors] + [str(correct)]
        random.shuffle(options)
        correct_idx = options.index(str(correct))
        
        questions.append(prob["question"])
        options_list.append(options)
        labels.append(correct_idx)
    
    print(f"  ✅ Loaded {len(questions)} AIME problems as MCQA")
    return {
        "questions": questions,
        "options": options_list,
        "labels": np.array(labels),
        "name": "AIME",
        "num_choices": 4
    }


def load_asdiv_as_mcqa(
    data_dir: str,
    max_samples: int = 100,
    seed: int = 42
) -> Dict:
    """
    加载 ASDIV 数据集并转换为 MCQA 格式
    """
    print(f"  Loading ASDIV from {data_dir}...")
    dataset = load_dataset(data_dir, "asdiv", split="validation")
    
    random.seed(seed)
    
    problems = []
    for item in dataset:
        question = f"{item['body']} {item['question']}"
        raw_answer = item['answer']
        match = re.match(r'([\d.]+)', raw_answer)
        answer = match.group(1) if match else raw_answer
        problems.append({"question": question, "answer": answer})
    
    if max_samples and max_samples < len(problems):
        problems = random.sample(problems, max_samples)
    
    # 收集所有答案作为干扰项池
    all_answers = [p["answer"] for p in problems]
    
    questions = []
    options_list = []
    labels = []
    
    for prob in problems:
        correct = prob["answer"]
        distractors = [a for a in all_answers if a != correct]
        if len(distractors) >= 3:
            distractors = random.sample(distractors, 3)
        else:
            distractors = distractors + ["0"] * (3 - len(distractors))
        
        options = distractors + [correct]
        random.shuffle(options)
        correct_idx = options.index(correct)
        
        questions.append(prob["question"])
        options_list.append(options)
        labels.append(correct_idx)
    
    print(f"  ✅ Loaded {len(questions)} ASDIV problems as MCQA")
    return {
        "questions": questions,
        "options": options_list,
        "labels": np.array(labels),
        "name": "ASDIV",
        "num_choices": 4
    }


# 本地数据集注册表
LOCAL_DATASET_REGISTRY = {
    "math": load_math_as_mcqa,
    "aime": load_aime_as_mcqa,
    "asdiv": load_asdiv_as_mcqa,
}


def load_local_dataset(name: str, datasets_dir: str, **kwargs) -> Dict:
    """根据名称加载本地数据集"""
    name_lower = name.lower()
    
    if name_lower not in LOCAL_DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {list(LOCAL_DATASET_REGISTRY.keys())}"
        )
    
    # 映射数据集目录
    dir_map = {
        "math": os.path.join(datasets_dir, "MATH"),
        "aime": os.path.join(datasets_dir, "AIME"),
        "asdiv": os.path.join(datasets_dir, "ASDIV"),
    }
    
    data_dir = dir_map[name_lower]
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    return LOCAL_DATASET_REGISTRY[name_lower](data_dir=data_dir, **kwargs)
