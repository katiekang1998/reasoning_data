from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
import os
import argparse
import re 
import json
from utils import is_equiv

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str)


args = parser.parse_args()

ckpt_dir = args.ckpt_dir


dataset = load_dataset("hendrycks/competition_math")
train_questions = np.array(dataset["train"]["problem"])
train_answers = np.array(dataset["train"]['solution'])


output = np.load(ckpt_dir+"/train_answers5_seed2.npy")


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def get_aug_answer(full_answer):
    idx = full_answer.rfind("The answer is")
    if idx == -1:
        return None
    else:
        answer = full_answer[idx + len("The answer is: "):]
        answer = answer.replace(":", "").replace("$", "").strip()
        if len(answer)> 0:
            if answer[-1] == ".":
                answer = answer[:-1]
            left = "\\boxed{"
            if answer[:len(left)] == left and answer[-1] == "}":
                answer = answer[len(left):-1]
        return answer

def answer_type_individial(output , answer):
    answer = remove_boxed(last_boxed_only_string(answer))
    
    output_answer = remove_boxed(last_boxed_only_string(output))
    if output_answer == None:
        output_answer = get_aug_answer(output)

    if output_answer is not None:
        
        eqiv = is_equiv(answer, output_answer, verbose=False)

        if eqiv:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


answer_types_all = []
for i in range(len(output)):
    answer_types = []
    for item in output[i]:
        answer_type = answer_type_individial(item, train_answers[i])
        answer_types.append(answer_type)
    answer_types_all.append(answer_types)

answer_types_all = np.array(answer_types_all)
print((answer_types_all==0).mean(axis=-1).mean())
print((answer_types_all==1).mean(axis=-1).mean())
print((answer_types_all==2).mean(axis=-1).mean())


np.save(os.path.join(ckpt_dir, f"train_answer_types5_seed2.npy"), answer_types_all)

# train_answer_types5_seed2