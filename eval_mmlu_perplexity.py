from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datasets import load_dataset
import tqdm
import torch
import argparse
from utils import *

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy
from typing import Sequence, Dict
import logging


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--ckpt", type=str)


args = parser.parse_args()


model_name = f"ckpts/{args.ckpt_dir}/checkpoint-{args.ckpt}"  # Replace with the specific LLaMA model you want
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

model.to("cuda")

model.eval()


def calculate_output_prob(input_text, output_text, model, tokenizer):
    # Combine input and output text
    full_text = input_text + output_text

    # Tokenize input and output
    inputs = tokenizer(full_text, return_tensors='pt').to('cuda')
    input_ids = inputs.input_ids

    # Get the index where the output text starts
    input_length = len(tokenizer.encode(input_text, add_special_tokens=False))
    output_length = len(tokenizer.encode(output_text, add_special_tokens=False))
    
    with torch.no_grad():
        # Get model outputs (logits)
        outputs = model(**inputs)
        logits = outputs.logits

    # Shift logits and labels to focus only on the output text portion
    shift_logits = logits[:, input_length:-1, :].contiguous()
    shift_labels = input_ids[:, input_length + 1:].contiguous()

    # Calculate the cross-entropy loss between predicted logits and actual tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


    
    # import IPython; IPython.embed()
    
    # text = ""
    # for i in range(len(loss)):
    #     text+=tokenizer.decode(shift_labels[0][i].cpu().numpy())
    #     if loss[i]>0.1:
    #         text+="["+str(loss[i].item())+"]"
    # print(text)

    # Calculate perplexity   
    # perplexity = torch.exp(loss.mean()).item()
    # num_tokens = shift_labels.ne(tokenizer.pad_token_id).sum().item()
    return np.e**(-1*loss.item())


# dataset = load_dataset("gsm8k", "main")
# train_questions = dataset["train"]["question"]
# train_answers = dataset["train"]['answer']
# eval_questions = train_questions
# eval_questions = [question + "\nAnswer:" for question in train_questions]
# eval_answers = [" "+train_answers[i]+"\nDone." for i in range(len(train_answers))] 



# perplexities = []
# num_tokens_all = []
# for i in tqdm.tqdm(range(len(eval_questions))):
#     input_text = eval_questions[i]
#     output_text = eval_answers[i]
#     perplexity,num_tokens = calculate_output_perplexity(input_text, output_text, model, tokenizer)
#     perplexities.append(perplexity)
#     num_tokens_all.append(num_tokens)

# np.save(model_name+"/train_perplexities.npy", perplexities)
# np.save(model_name+"/train_num_tokens.npy", num_tokens_all)





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
eval_answers = [[" A", " B", " C", " D"][prompt["answer"]] for prompt in prompts_test]

eval_questions = eval_questions
eval_answers = eval_answers


probs = []
for i in tqdm.tqdm(range(len(eval_questions))):
    input_text = eval_questions[i]
    output_text = eval_answers[i]
    prob = calculate_output_prob(input_text, output_text, model, tokenizer)
    probs.append(prob)

print(np.mean(probs))
np.save(model_name+"/mmlu_probs.npy", probs)
# np.save(model_name+"/train_num_tokens.npy", num_tokens_all)

# IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"

# def calculate_batch_perplexity(batch, model):
#     """
#     Calculate the perplexity for the output portion in a batch.

#     Args:
#         batch (dict): A batch containing input_ids and labels.
#         model (AutoModelForCausalLM): The pre-trained model.

#     Returns:
#         List[float]: A list of perplexities for each output in the batch.
#     """
#     input_ids = batch["input_ids"]
#     labels = batch["labels"]

#     # Get model outputs (logits)
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids)
#         logits = outputs.logits

#     # Shift logits and labels for the output portion
#     shift_logits = logits[:, :-1, :].contiguous()
#     shift_labels = labels[:, 1:].contiguous()

#     # Calculate cross-entropy loss only where labels are not IGNORE_INDEX
#     loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)
#     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

#     # Reshape to calculate mean loss per example
#     loss_per_example = loss.view(input_ids.size(0), -1).sum(dim=1) / (shift_labels != IGNORE_INDEX).sum(dim=1)

#     # Calculate perplexity
#     perplexities = torch.exp(loss_per_example).tolist()
#     num_tokens = (shift_labels != IGNORE_INDEX).sum(dim=1).tolist()
    
#     return perplexities, num_tokens

# dataset = load_dataset("gsm8k", "main")
# train_questions = dataset["train"]["question"]
# train_answers = dataset["train"]['answer']

# dataset2 = SupervisedDataset(train_questions, train_answers, tokenizer)
# data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

# dataloader = DataLoader(dataset2, batch_size=2, collate_fn=data_collator)

# # Calculate perplexities for batches
# perplexities_all = []
# num_tokens_all = []
# for batch in tqdm.tqdm(dataloader):
#     batch = {key: value.to("cuda") for key, value in batch.items()}
    
#     perplexities,num_tokens = calculate_batch_perplexity(batch, model)
#     perplexities_all.extend(perplexities)
#     num_tokens_all.extend(num_tokens)

# import IPython; IPython.embed()