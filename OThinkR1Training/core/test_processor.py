from core.dataset_processor import BaseDatasetProcessor
import re

class TestASDIVProcessor(BaseDatasetProcessor):
    def make_conversation(self, example):
        # EleutherAI/asdiv 字段: body, question, answer, formula, solution_type
        # 原始 ASDIV 字段: text, label
        question_text = f"{example['body']} {example['question']}"

        # answer 格式如 "9 (apples)"，提取数字部分作为 solution
        raw_answer = example['answer']
        # 提取数字（可能是小数）
        match = re.match(r'([\d.]+)', raw_answer)
        solution = f"${match.group(1)}$" if match else f"${raw_answer}$"

        return {
            "prompt": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question_text},
            ],
            "question_prompt": question_text,
            "solution": solution,
            "type": self.name
        }
