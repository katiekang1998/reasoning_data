from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
import os
import argparse
import re 
from utils import is_equiv
import json
from vllm.lora.request import LoRARequest



import tqdm
import torch
import re
import string



parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--seed", type=int, default=2)


args = parser.parse_args()

ckpt_dir = args.ckpt_dir
num_devices = 4



def process_item(questions, choices, answers,):
    keys = ['A', 'B', 'C', 'D']
    question = questions
    choices = ''.join([f"{key}. {choice}\n" for choice, key in zip(choices, keys)])
    prompt = f"{question}\n{choices}Answer:"
    target = ' ' + keys[answers]
    return prompt, target

def create_prompt_for_item(questions, choices, answers, subject, shots):
    subject_name = " ".join(subject.split('_'))
    description = f"The following are multiple choice questions (with answers) about {subject_name}."
    prompt = f"{description}\n\n"
    for shot in shots:
        shot_question, shot_choices, shot_answers = shot["question"], shot["choices"], shot["answer"]
        shot_prompt, shot_target = process_item(shot_question, shot_choices, shot_answers,)
        prompt += f"{shot_prompt}{shot_target}\n\n"
    item_prompt, _ = process_item(questions, choices, answers,)
    prompt += f"{item_prompt}"
    return prompt

def get_fewshot_for_example(questions, choices, answers, subject, n_shot):
    fewshot_items = dev_dict[subject]
    fewshot_items = list(fewshot_items)[:n_shot]
    return create_prompt_for_item(questions, choices, answers, subject, fewshot_items)


def prepare_prompt(questions, choices, answers, subject,):
    prompt = get_fewshot_for_example(questions, choices, answers, subject, n_shot=5)
    prompt_dict = {}
    prompt_dict["prompt"] =prompt
    prompt_dict["answer"] = answers
    return prompt_dict

topics = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

split = "test"

test_questions = []
test_choices = []
test_answers = []
test_subjects = []
dev_dict = {}
for topic in topics:
    dataset = load_dataset("tasksource/mmlu", topic)
    test_questions.append(dataset[split]["question"])
    test_choices.append(dataset[split]["choices"])
    test_answers.append(dataset[split]["answer"])
    test_subjects.append([topic for _ in range(len(dataset[split]["question"]))])
    dev_dict[topic] = dataset["dev"]
test_questions = np.concatenate(test_questions)
test_choices = np.concatenate(test_choices)
test_answers = np.concatenate(test_answers)
test_subjects = np.concatenate(test_subjects)

prompts_test = list(map(prepare_prompt, test_questions,test_choices,test_answers, test_subjects))
eval_questions = [prompt["prompt"] for prompt in prompts_test]
eval_answers = [["A", "B", "C", "D"][prompt["answer"]] for prompt in prompts_test]

eval_questions = eval_questions
eval_answers = eval_answers

llm = LLM(model=ckpt_dir, tensor_parallel_size=num_devices, trust_remote_code=True, max_logprobs=20)  # Name or path of your model





def answer_type_individial(output , answer):
    output =output.strip()
    if output in ["A", "B", "C", "D"]:
        eqiv = (output == answer)

        if eqiv:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


sampling_params = SamplingParams(
    n = 1,
    temperature=0,
    max_tokens=1,
    top_p=0.95,
    seed=args.seed,
    stop="\nDone.",
    logprobs=20
)

output = llm.generate(eval_questions, sampling_params)


prob_dicts_all = []
probs_all = []

for i in range(len(output)):
    logprobs = output[i].outputs[0].logprobs[0]
    
    info_dict = {}
    
    for key, value in logprobs.items():
        info_dict[value.decoded_token] = np.e**(value.logprob)
    
    if " "+eval_answers[i] in info_dict:
        probs_all.append(info_dict[" "+eval_answers[i]])
    else:
        probs_all.append(1-np.sum(list(info_dict.values())))
    
    prob_dicts_all.append(info_dict)

print(np.mean(probs_all))

if ckpt_dir == "meta-llama/Meta-Llama-3-8B":
    np.save(os.path.join("ckpts/llama3", f"mmlu_prob_dicts.npy"), prob_dicts_all)
    np.save(os.path.join("ckpts/llama3", f"mmlu_probs.npy"), probs_all)
else:
    np.save(os.path.join(ckpt_dir, f"mmlu_prob_dicts.npy"), prob_dicts_all)
    np.save(os.path.join(ckpt_dir, f"mmlu_probs.npy"), probs_all)


# answer_types_all = []
# answers_all = []
# for i in range(len(output)):
#     answer_types = []
#     answers = []
    
    
    
#     for item in output[i].outputs:
#         answers.append(item.text)
#         answer_type = answer_type_individial(item.text, eval_answers[i])
#         answer_types.append(answer_type)
#     answer_types_all.append(answer_types)
#     answers_all.append(answers)

# answer_types_all = np.array(answer_types_all)
# answers_all = np.array(answers_all)
# print((answer_types_all==0).mean(axis=-1).mean())
# print((answer_types_all==1).mean(axis=-1).mean())
# print((answer_types_all==2).mean(axis=-1).mean())

# if ckpt_dir == "meta-llama/Meta-Llama-3-8B":
#     np.save(os.path.join("ckpts/llama3", f"mmlu_answers.npy"), answers_all)
#     np.save(os.path.join("ckpts/llama3", f"mmlu_answer_types.npy"), answer_types_all)
# else:
#     np.save(os.path.join(ckpt_dir, f"mmlu_answers.npy"), answers_all)
#     np.save(os.path.join(ckpt_dir, f"mmlu_answer_types.npy"), answer_types_all)

