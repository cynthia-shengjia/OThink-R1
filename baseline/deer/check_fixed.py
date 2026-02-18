"""
Fixed version of DEER check.py
- Moved tokenizer initialization inside infer() to fix args scope bug
"""
import json
from transformers import AutoTokenizer
import re
import importlib.util
import os
import argparse
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
import pickle
from math import comb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./")
    parser.add_argument('--n_sampling', type=int, default=1)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, default="math")
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--generation_path", default="test", type=str)
    parser.add_argument("--prompt_type", default="qwen-base", type=str)
    args = parser.parse_args()
    return args

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def infer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    examples = load_data(args.data_name, args.split, args.data_dir)
    file_outputs = read_jsonl(args.generation_path)

    print("llm generate done")
    print(len(file_outputs))

    pass_at_k_list = []
    k = args.k
    correct_cnt = 0

    for i in tqdm(range(len(file_outputs)), "check correct..."):
        d = examples[i]
        gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
        generated_responses = file_outputs[i]['generated_responses']
        generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
        is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1
        file_outputs[i]['generated_answers'] = generated_answers
        file_outputs[i]['gold_answer'] = gt_ans
        file_outputs[i]['is_correct'] = is_correct
        file_outputs[i]['answers_correctness'] = is_correct_list

        if len(is_correct_list) > 1:
            correct_answers = sum(is_correct_list)
            n = len(generated_answers)
            if correct_answers > 0:
                if n - correct_answers < k:
                    pass_at_k = 1
                else:
                    pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                pass_at_k_list.append(pass_at_k)
            else:
                pass_at_k_list.append(0)

    print(f"correct cnt / total cnt: {correct_cnt}/{len(file_outputs)}")
    print(f"Acc: {correct_cnt / len(file_outputs):.4f}")

    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"Pass@{k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
    else:
        print(f"Pass@1: {correct_cnt}/{len(file_outputs)} = {correct_cnt / len(file_outputs):.4f}")

    response_length = []
    token_num = []
    test_num = len(file_outputs)
    for data in file_outputs:
        response_length.append(len(data['generated_responses'][0].split()))
        tokens_response_len = len(tokenizer(data['generated_responses'][0])['input_ids'])
        token_num.append(tokens_response_len)

    print("length:", sum(response_length) / test_num)
    print('token_num:', sum(token_num) / test_num)

if __name__ == "__main__":
    args = parse_args()
    infer(args)
