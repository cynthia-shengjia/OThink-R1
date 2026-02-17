# -------------------------------------------------------------------
# MATH + AIME DataProcessor for OThink-R1 Evaluation
# -------------------------------------------------------------------

from core.dataset_processor import BaseDatasetProcessor
import re


class MATHProcessor(BaseDatasetProcessor):
    """
    MATH 数据集 (hendrycks/competition_math)
    字段: problem, level, type, solution
    solution 中答案在 \\boxed{} 里
    """
    def make_conversation(self, example):
        problem = example['problem']
        raw_solution = example['solution']

        # 提取 \boxed{...} 中的答案
        match = re.search(r'\\boxed\{(.*)\}', raw_solution)
        if match:
            answer = match.group(1)
        else:
            answer = raw_solution

        solution = f"${answer}$"

        return {
            "prompt": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": problem},
            ],
            "question_prompt": problem,
            "solution": solution,
            "type": self.name
        }


class AIMEProcessor(BaseDatasetProcessor):
    """
    AIME 数据集 (AI-MO/aimo-validation-aime)
    字段: problem, answer, url
    答案是 0-999 的整数
    """
    def make_conversation(self, example):
        problem = example['problem']
        answer = str(example['answer']).strip()
        solution = f"${answer}$"

        return {
            "prompt": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": problem},
            ],
            "question_prompt": problem,
            "solution": solution,
            "type": self.name
        }
