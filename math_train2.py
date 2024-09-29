import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers #prev 4.41.2
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import os
from datasets import load_dataset
import numpy as np
import argparse
import json
from huggingface_params import cache_dir, use_auth_token
from utils import *
from peft import LoraConfig, get_peft_model
import json
import torch.distributed as dist

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


def get_level(level_str):
    if level_str[-1] == "?":
        return -1 
    else:
        return int(level_str[-1])


def make_supervised_data_module(output_dir, train_type, learning_rate, batch_size, epochs, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    dataset = load_dataset("hendrycks/competition_math", cache_dir=cache_dir, trust_remote_code=True)
    
    train_questions_orig = np.array(dataset["train"]["problem"])
    train_answers_orig = np.array(dataset["train"]['solution'])
    train_levels_orig = np.array(dataset["train"]['level'])
    train_levels_orig = np.array(list(map(get_level, train_levels_orig)))
    orig_data_easy_idxs = np.where((train_levels_orig>=1)*(train_levels_orig<=3))[0]
    
        
    for i in range(len(train_answers_orig)):
        answer = remove_boxed(last_boxed_only_string(train_answers_orig[i]))
        train_answers_orig[i] += f"\nFINAL ANSWER:\n{answer}"
    
    train_questions = list(train_questions_orig[orig_data_easy_idxs])
    train_answers = list(train_answers_orig[orig_data_easy_idxs])
    
    
    with open('ckpts/amrith_math/math_batch_1_outputs_gpt4.jsonl', 'r') as json_file:
        json_list = list(json_file) #1
    with open('ckpts/amrith_math/math_batch_2_outputs_gpt4.jsonl', 'r') as json_file:
        json_list += list(json_file) #5
    with open('ckpts/amrith_math/math_batch_3_outputs_gpt4.jsonl', 'r') as json_file:
        json_list += list(json_file) #4
    with open('ckpts/amrith_math/math_batch_4_outputs_gpt4.jsonl', 'r') as json_file:
        json_list += list(json_file) #5
    with open('ckpts/amrith_math/math_batch_5_outputs_gpt4.jsonl', 'r') as json_file:
        json_list += list(json_file) #5

    train_questions_amrith = []
    train_answers_amrith = []
    for json_str in json_list:
        result = json.loads(json_str)
        train_questions_amrith.append(result["query"])
        train_answers_amrith.append(result["response"].replace("\n\n", "\n"))
    
    
    num_repeats = len(train_questions_amrith)//len(train_questions_orig)
    assert(len(train_questions_amrith) == len(train_questions_orig)*num_repeats)
    amrith_data_orig_idxs = np.tile(np.arange(len(train_questions_orig)), num_repeats)
    
    
    amrith_data_easy_idxs = np.where([elem in orig_data_easy_idxs for elem in amrith_data_orig_idxs])[0]

    train_questions_amrith = np.array(train_questions_amrith)[amrith_data_easy_idxs]
    train_answers_amrith = np.array(train_answers_amrith)[amrith_data_easy_idxs]
    
    if train_type =="0copies":
        num_copies = 0
    elif train_type =="1copies":
        num_copies = 1
    elif train_type =="3copies":
        num_copies = 3
    elif train_type =="7copies":
        num_copies = 7
    elif train_type =="13copies":
        num_copies = 13
    elif train_type =="15copies":
        num_copies = 15
    elif train_type =="19copies":
        num_copies = 19
    else:
        raise Exception("Invalid train type")

    if dist.get_rank() == 0:
        amrith_data_subsample_idxs = np.arange(len(train_questions)*num_copies)
        
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, "amrith_easy_data_subsample_idxs.npy"), amrith_data_subsample_idxs)
        dist.barrier()
    else:
        dist.barrier()
        amrith_data_subsample_idxs = np.load(os.path.join(output_dir, f"amrith_easy_data_subsample_idxs.npy"))
    print(amrith_data_subsample_idxs)

    train_questions+= list(train_questions_amrith[amrith_data_subsample_idxs])
    train_answers+= list(train_answers_amrith[amrith_data_subsample_idxs])
    
    
    print(len(train_questions))
    print(len(train_answers))
    train_dataset = SupervisedDataset(train_questions, train_answers, tokenizer=tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)

def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_type", type=str, default="batch_1_all")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-math-7b-base")


    args = parser.parse_args()
    train_type = args.train_type
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    model_name_or_path = args.model
    
    
    project_name = "math_amrith_easy_deepseek"
    run_name  = f"{epochs}epochs_{train_type}_lr{learning_rate}_bs{batch_size}"
        
        

    
    batch_size = batch_size
    num_devices = torch.cuda.device_count()

    if num_devices>2:
        per_device_batch_size = 2
        gradient_accumulation_steps = int((batch_size/2)/num_devices)
    else:
        per_device_batch_size = 1
        gradient_accumulation_steps = int((batch_size/1)/num_devices)
    print("Num devices: ", num_devices)
    print("Per device batch: ", per_device_batch_size)
    print("Grad accum steps: ", gradient_accumulation_steps)

    assert(gradient_accumulation_steps*per_device_batch_size*num_devices==batch_size)
    
    output_dir = f"ckpts/{project_name}_{run_name}"
    

    training_args = TrainingArguments(
        num_train_epochs = epochs, 
        per_device_train_batch_size = per_device_batch_size,
        per_device_eval_batch_size = per_device_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        lr_scheduler_type = "linear",
        warmup_steps = 20,
        learning_rate = learning_rate,
        max_grad_norm = 2,
        optim = "adamw_torch",
        output_dir = output_dir,
        evaluation_strategy = "no",
        # eval_steps = 25,
        report_to = "none",
        logging_strategy = "steps",
        logging_steps = 25,
        save_strategy = "epoch",
        save_only_model = True,
        run_name=run_name,
        bf16 = True,
        fsdp= "full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap= 'LlamaDecoderLayer',
        tf32 =True,
    )
        

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        use_auth_token = use_auth_token,
        cache_dir=cache_dir,
        trust_remote_code=True
    )



    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_auth_token = use_auth_token,
        model_max_length=1024,
        padding_side="right",
        cache_dir=cache_dir,
        trust_remote_code=True)




    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(output_dir, train_type, learning_rate, batch_size, epochs, tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    


if __name__ == "__main__":
    train()
