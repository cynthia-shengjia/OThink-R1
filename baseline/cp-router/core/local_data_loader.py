"""
本地数据集加载器
适配 datasets/ 目录下的 MATH, AIME, ASDIV, GSM8K, CommonsenseQA, OpenBookQA 数据集
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
    """从 MATH solution 中提取 \\boxed{} 内容"""
    match = re.search(r'\boxed\{(.*)\}', solution, re.DOTALL)
    return match.group(1) if match else solution

def load_math_as_mcqa(
    data_dir: str,
    max_samples: int = 100,
    seed: int = 42
) -> Dict:
    """
    加载 MATH-500 数据集 (ricdomolm/MATH-500) 并转换为 MCQA 格式
    MATH-500 字段: problem, answer (直接提供答案，无需从 solution 提取)
    """
    print(f"  Loading MATH-500 from {data_dir}...")
    dataset = load_dataset(data_dir, split="test")
    
    random.seed(seed)
    
    all_answers = []
    problems = []
    for item in dataset:
        # MATH-500 直接有 answer 字段
        answer = item.get('answer', '')
        if not answer and 'solution' in item:
            answer = _extract_boxed(item['solution'])
        all_answers.append(str(answer).strip())
        problems.append({
            "question": item['problem'],
            "answer": str(answer).strip()
        } + '
')
    
    if max_samples and max_samples < len(problems):
        indices = random.sample(range(len(problems)), max_samples)
        problems = [problems[i] for i in indices]
    
    questions = []
    options_list = []
    labels = []
    
    for prob in problems:
        correct = prob["answer"]
        distractors = [a for a in all_answers if a != correct]
        if len(distractors) >= 3:
            distractors = random.sample(distractors, 3)
        else:
            distractors = distractors + [f"None of the above"] * (3 - len(distractors))
        
        options = distractors + [correct]
        random.shuffle(options)
        correct_idx = options.index(correct)
        
        questions.append(prob["question"])
        options_list.append(options)
        labels.append(correct_idx)
    
    print(f"  ✅ Loaded {len(questions)} MATH-500 problems as MCQA")
    return {
        "questions": questions,
        "options": options_list,
        "labels": np.array(labels),
        "name": "MATH-500",
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

# =====================================================================
# 新增: GSM8K
# =====================================================================
def load_gsm8k_as_mcqa(
    data_dir: str,
    max_samples: int = 100,
    seed: int = 42
) -> Dict:
    """
    加载 GSM8K 数据集并转换为 MCQA 格式
    GSM8K 答案格式: "...#### <number>"
    策略: 正确答案 + 3 个从答案池随机抽取的数值干扰项
    """
    print(f"  Loading GSM8K from {data_dir}...")
    dataset = load_dataset(data_dir, "main", split="test")

    random.seed(seed)

    problems = []
    all_answers = []
    for item in dataset:
        question = item["question"]
        # 提取 #### 后面的数字
        match = re.search(r'####\s*(.*)', item["answer"])
        answer = match.group(1).strip() if match else item["answer"]
        problems.append({"question": question, "answer": answer})
        all_answers.append(answer)

    if max_samples and max_samples < len(problems):
        indices = random.sample(range(len(problems)), max_samples)
        problems = [problems[i] for i in indices]

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

    print(f"  ✅ Loaded {len(questions)} GSM8K problems as MCQA")
    return {
        "questions": questions,
        "options": options_list,
        "labels": np.array(labels),
        "name": "GSM8K",
        "num_choices": 4
    }

# =====================================================================
# 新增: CommonsenseQA (原生 MCQA, 5 选项)
# =====================================================================
def load_commonsenseqa_as_mcqa(
    data_dir: str,
    max_samples: int = 100,
    seed: int = 42
) -> Dict:
    """
    加载 CommonsenseQA 数据集
    CommonsenseQA 本身就是 5 选 1 的 MCQA, 直接使用
    """
    print(f"  Loading CommonsenseQA from {data_dir}...")
    dataset = load_dataset(data_dir, split="validation")

    random.seed(seed)

    problems = []
    for item in dataset:
        question = item["question"]
        choices_text = item["choices"]["text"]       # list of 5 strings
        answer_key = item["answerKey"]               # 'A'..'E'
        correct_idx = ord(answer_key) - ord('A')
        problems.append({
            "question": question,
            "options": choices_text,
            "label": correct_idx
        })

    if max_samples and max_samples < len(problems):
        problems = random.sample(problems, max_samples)

    questions = [p["question"] for p in problems]
    options_list = [p["options"] for p in problems]
    labels = [p["label"] for p in problems]

    print(f"  ✅ Loaded {len(questions)} CommonsenseQA problems as MCQA (5 choices)")
    return {
        "questions": questions,
        "options": options_list,
        "labels": np.array(labels),
        "name": "CommonsenseQA",
        "num_choices": 5
    }

# =====================================================================
# 新增: OpenBookQA (原生 MCQA, 4 选项)
# =====================================================================
def load_openbookqa_as_mcqa(
    data_dir: str,
    max_samples: int = 100,
    seed: int = 42
) -> Dict:
    """
    加载 OpenBookQA 数据集
    OpenBookQA 本身就是 4 选 1 的 MCQA, 直接使用
    """
    print(f"  Loading OpenBookQA from {data_dir}...")
    dataset = load_dataset(data_dir, "main", split="test")

    random.seed(seed)

    problems = []
    for item in dataset:
        question = item["question_stem"]
        choices_text = item["choices"]["text"]       # list of 4 strings
        answer_key = item["answerKey"]               # 'A'..'D'
        correct_idx = ord(answer_key) - ord('A')
        problems.append({
            "question": question,
            "options": choices_text,
            "label": correct_idx
        })

    if max_samples and max_samples < len(problems):
        problems = random.sample(problems, max_samples)

    questions = [p["question"] for p in problems]
    options_list = [p["options"] for p in problems]
    labels = [p["label"] for p in problems]

    print(f"  ✅ Loaded {len(questions)} OpenBookQA problems as MCQA (4 choices)")
    return {
        "questions": questions,
        "options": options_list,
        "labels": np.array(labels),
        "name": "OpenBookQA",
        "num_choices": 4
    }

# =====================================================================
# 注册表
# =====================================================================
LOCAL_DATASET_REGISTRY = {
    "math": load_math_as_mcqa,
    "aime": load_aime_as_mcqa,
    "asdiv": load_asdiv_as_mcqa,
    "gsm8k": load_gsm8k_as_mcqa,
    "commonsenseqa": load_commonsenseqa_as_mcqa,
    "openbookqa": load_openbookqa_as_mcqa,
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
        "gsm8k": os.path.join(datasets_dir, "GSM8K"),
        "commonsenseqa": os.path.join(datasets_dir, "CommonsenseQA"),
        "openbookqa": os.path.join(datasets_dir, "OpenBookQA"),
    }

    data_dir = dir_map[name_lower]
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    return LOCAL_DATASET_REGISTRY[name_lower](data_dir=data_dir, **kwargs)
