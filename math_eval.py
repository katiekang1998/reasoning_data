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
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--eval_type", type=str, default="test")
parser.add_argument("--num_devices", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=5)

args = parser.parse_args()

ckpt_dir = args.ckpt_dir


dataset = load_dataset("hendrycks/competition_math")
train_questions = np.array(dataset["train"]["problem"])
train_answers = np.array(dataset["train"]['solution'])

test_questions = dataset["test"]["problem"]
test_answers = dataset["test"]['solution']


sampling_params = SamplingParams(
    n = args.num_samples,
    temperature=0.8,
    max_tokens=1024,
    top_p=0.95,
    seed=args.seed,
    stop="\nDone."
)

if args.eval_type == "test":
    eval_questions = test_questions
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = test_answers
elif args.eval_type == "train_aug_1":
    with open('data/MATH_aug/AugMATH_part1.jsonl', 'r') as json_file:
        json_list = list(json_file)
    train_questions = []
    train_answers = []
    for json_str in json_list:
        result = json.loads(json_str)
        train_questions.append(result["query"])
        train_answers.append(result["response"])
        
    train_questions = np.array(train_questions)
    train_answers = np.array(train_answers)
    eval_questions = train_questions
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = train_answers
elif args.eval_type == "train_aug_2":
    with open('data/MATH_aug/AugMATH_part2.jsonl', 'r') as json_file:
        json_list = list(json_file)

    train_questions = []
    train_answers = []
    for json_str in json_list:
        result = json.loads(json_str)
        train_questions.append(result["query"])
        train_answers.append(result["response"])
    train_questions = np.array(train_questions)
    train_answers = np.array(train_answers)
    eval_questions = train_questions
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = train_answers
elif args.eval_type == "train_aug_subsample":
    with open('data/MATH_aug/AugMATH_part1.jsonl', 'r') as json_file:
        json_list = list(json_file)

    with open('data/MATH_aug/AugMATH_part2.jsonl', 'r') as json_file:
        json_list += list(json_file)

    train_questions = []
    train_answers = []
    for json_str in json_list:
        result = json.loads(json_str)
        train_questions.append(result["query"])
        train_answers.append(result["response"])
        
    train_questions = np.array(train_questions)
    train_answers = np.array(train_answers)
    
    try:
        subsample_idxs = np.load(ckpt_dir + "/subsample_idxs.npy")[:5000]
    except:
        subsample_idxs = np.load(ckpt_dir.rsplit('/', 1)[0] + "/subsample_idxs.npy")[:5000]
    eval_questions = train_questions[subsample_idxs]
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = train_answers[subsample_idxs]
elif args.eval_type == "train_aug":
    with open('data/MATH_aug/AugMATH_part1.jsonl', 'r') as json_file:
        json_list = list(json_file)

    with open('data/MATH_aug/AugMATH_part2.jsonl', 'r') as json_file:
        json_list += list(json_file)

    train_questions = []
    train_answers = []
    for json_str in json_list:
        result = json.loads(json_str)
        train_questions.append(result["query"])
        train_answers.append(result["response"])
        
    train_questions = np.array(train_questions)
    train_answers = np.array(train_answers)
    
    try:
        subsample_idxs = np.load(ckpt_dir + "/subsample_idxs.npy")
    except:
        subsample_idxs = np.load(ckpt_dir.rsplit('/', 1)[0] + "/subsample_idxs.npy")
    eval_questions = train_questions[subsample_idxs]
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = train_answers[subsample_idxs]

llm = LLM(model=ckpt_dir, tensor_parallel_size=args.num_devices)  # Name or path of your model
output = llm.generate(eval_questions, sampling_params)



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
    if args.eval_type == "test":
        answer = remove_boxed(last_boxed_only_string(answer))
    else:
        answer = get_aug_answer(answer)
    
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
answers_all = []
for i in range(len(output)):
    answer_types = []
    answers = []
    for item in output[i].outputs:
        answers.append(item.text)
        answer_type = answer_type_individial(item.text, eval_answers[i])
        answer_types.append(answer_type)
    answer_types_all.append(answer_types)
    answers_all.append(answers)

answer_types_all = np.array(answer_types_all)
answers_all = np.array(answers_all)
print((answer_types_all==0).mean(axis=-1).mean())
print((answer_types_all==1).mean(axis=-1).mean())
print((answer_types_all==2).mean(axis=-1).mean())


np.save(os.path.join(ckpt_dir, f"{args.eval_type}_answers{args.num_samples}_seed{args.seed}.npy"), answers_all)
np.save(os.path.join(ckpt_dir, f"{args.eval_type}_answer_types{args.num_samples}_seed{args.seed}.npy"), answer_types_all)