from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
import os
import argparse
import re 
from utils import is_equiv
import json

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--eval_type", type=str, default="test")
parser.add_argument("--num_devices", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--temp", type=float, default=0.8)


args = parser.parse_args()

ckpt_dir = args.ckpt_dir
temp = args.temp

llm = LLM(model=ckpt_dir, tensor_parallel_size=args.num_devices)  # Name or path of your model

def perturb_string(string):
    substrings = string.split()
    string_perturb = ""
    for substring in substrings:
        if substring.isalpha():
            rand_idx = np.random.randint(0, len(substring))
            string_perturb += " " + (substring[:rand_idx]+substring[rand_idx]+substring[rand_idx:])
        else:
            string_perturb += " " + (substring)
    return string_perturb.strip()

if args.eval_type == "test":
    dataset = load_dataset("gsm8k", "main")

    test_questions = dataset["test"]["question"]
    test_answers = dataset["test"]['answer']
    eval_questions = test_questions
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = test_answers
elif args.eval_type == "test_small":
    dataset = load_dataset("gsm8k", "main")

    test_questions = dataset["test"]["question"][:10]
    test_answers = dataset["test"]['answer'][:10]
    eval_questions = test_questions
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = test_answers
elif args.eval_type == "train":
    dataset = load_dataset("gsm8k", "main")
    train_questions = dataset["train"]["question"]
    train_answers = dataset["train"]['answer']
    eval_questions = train_questions
    eval_questions = [question + "\nAnswer:" for question in train_questions]
    eval_answers = train_answers
elif args.eval_type == "train_gpt4o":
    dataset = load_dataset("gsm8k", "main")
    train_questions = dataset["train"]["question"]
    train_answers_gpt4o_orig = np.load("gsm8k_train_answers1000_gpt4o.npy")
    
    train_answers_gpt4o = []
    
    for answer in train_answers_gpt4o_orig:
        if "Answer3:" in answer:
            
            train_answers_gpt4o.append(answer[answer.index('Answer3: ')+len("Answer3: "):answer.index("\n")])
        else:
            train_answers_gpt4o.append(answer[:answer.index("\n")])
    
    train_answers_gpt4o = np.array(train_answers_gpt4o)
    
    
    train_answers = dataset["train"]['answer']
    
    eval_questions = train_questions[:1000]
    eval_questions = [question + "\nAnswer: " for question in train_questions]
    eval_questions = [question + answer for question, answer in zip(eval_questions, train_answers_gpt4o)]
    eval_answers = train_answers[:1000]
    
elif args.eval_type == "train_subsample":
    dataset = load_dataset("gsm8k", "main")
    train_questions = dataset["train"]["question"][:50]
    train_answers = dataset["train"]['answer'][:50]
    eval_questions = train_questions
    eval_questions = [question + "\nAnswer:" for question in train_questions]
    eval_answers = train_answers
elif args.eval_type == "train_weird":
    dataset = load_dataset("gsm8k", "main")
    train_questions = dataset["train"]["question"]
    train_answers = dataset["train"]['answer']
    eval_questions = [perturb_string(question) + "\nAnswer:" for question in train_questions]
    eval_answers = train_answers
elif args.eval_type == "train_aug_1":
    with open('data/GSM8k_aug/AugGSM8K_part1.jsonl', 'r') as json_file:
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
    with open('data/GSM8k_aug/AugGSM8K_part2.jsonl', 'r') as json_file:
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
elif args.eval_type == "train_aug":
    with open('data/GSM8k_aug/AugGSM8K_part1.jsonl', 'r') as json_file:
        json_list = list(json_file)

    with open('data/GSM8k_aug/AugGSM8K_part2.jsonl', 'r') as json_file:
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
        return answer.replace(",", "")

def extract_latex(text):
    start = text.find("#### ") + len("#### ")
    return text[start:].replace(",", "")

def answer_type_individial(output , answer):
    if "aug" in args.eval_type:
        answer = get_aug_answer(answer)
    else:
        answer = extract_latex(answer)

    output_answer = get_aug_answer(output)
    if output_answer == None:
        output_answer = extract_latex(output)
    if output_answer is not None and answer is not None:
        eqiv = is_equiv(answer, output_answer, verbose=False)

        if eqiv:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


sampling_params = SamplingParams(
    n = args.num_samples,
    temperature=temp,
    max_tokens=512,
    top_p=0.95,
    seed=args.seed,
    stop="\nDone."
)

output = llm.generate(eval_questions, sampling_params)


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


np.save(os.path.join(ckpt_dir, f"{args.eval_type}_answers{args.num_samples}_seed{args.seed}_temp{args.temp}.npy"), answers_all)
np.save(os.path.join(ckpt_dir, f"{args.eval_type}_answer_types{args.num_samples}_seed{args.seed}_temp{args.temp}.npy"), answer_types_all)