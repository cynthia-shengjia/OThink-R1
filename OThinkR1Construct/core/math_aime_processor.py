from core.dataset_processor import BaseDatasetProcessor
import re


class MATHProcessor(BaseDatasetProcessor):
    def make_conversation(self, example):
        problem = example['problem']
        raw_solution = example['solution']
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
