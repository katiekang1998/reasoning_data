import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import transformers
from transformers import Trainer, TrainingArguments
import os
from datasets import load_dataset
import numpy as np
import argparse
import json
from huggingface_params import cache_dir, use_auth_token
from utils import *

def make_supervised_data_module(train_type, output_dir, num_train_points, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
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
    

# 2646
    print(train_type)
    if train_type == "unmemorized_geq_3":
        subsample_idxs = np.load("ckpts/math_aug3_total20000_epochs20/subsample_idxs_max_unmemorized>=3.npy")
    elif train_type == "unmemorized_leq_3":
        subsample_idxs = np.load("ckpts/math_aug3_total20000_epochs20/subsample_idxs_max_unmemorized<=3.npy")
    elif train_type == "rand":
        subsample_idxs_leq3 = np.load("ckpts/math_aug3_total20000_epochs20/subsample_idxs_max_unmemorized<=3.npy")
        subsample_idxs_geq3 = np.load("ckpts/math_aug3_total20000_epochs20/subsample_idxs_max_unmemorized>=3.npy")
        subsample_idxs = np.concatenate([subsample_idxs_leq3[:5000], subsample_idxs_geq3[:5000]])
        np.random.shuffle(subsample_idxs)
    elif train_type == "unmemorized_eq_0":
        subsample_idxs = np.load("ckpts/math_aug3_total20000_epochs20/subsample_idxs_max_unmemorized==0.npy")
    elif train_type == "unmemorized_eq_1":
        subsample_idxs = np.load("ckpts/math_aug3_total20000_epochs20/subsample_idxs_max_unmemorized==1.npy")
        subsample_idxs = np.random.choice(subsample_idxs, 2646, replace=False)
    elif train_type == "unmemorized_eq_2":
        subsample_idxs = np.load("ckpts/math_aug3_total20000_epochs20/subsample_idxs_max_unmemorized==2.npy")
        subsample_idxs = np.random.choice(subsample_idxs, 2646, replace=False)
    elif train_type == "unmemorized_eq_3":
        subsample_idxs = np.load("ckpts/math_aug3_total20000_epochs20/subsample_idxs_max_unmemorized==3.npy")
        subsample_idxs = np.random.choice(subsample_idxs, 2646, replace=False)
    elif train_type == "unmemorized_eq_4":
        subsample_idxs = np.load("ckpts/math_aug3_total20000_epochs20/subsample_idxs_max_unmemorized==4.npy")
        subsample_idxs = np.random.choice(subsample_idxs, 2646, replace=False)
    elif train_type == "rand2646":
        subsample_idxs = np.load("ckpts/math_aug3_total20000_epochs20/subsample_idxs.npy")
        subsample_idxs = np.random.choice(subsample_idxs, 2646, replace=False)
        # subsample_idxs = np.random.choice(np.arange(0, len(train_questions)), 2646, replace=False)
        # np.random.shuffle(subsample_idxs)
    elif train_type == "rand_50_unmemorized_eq_50":
        subsample_idxs1 = np.load("ckpts/math_aug3_total20000_epochs20/subsample_idxs_max_unmemorized==0.npy")
        subsample_idxs2 = np.random.choice(np.arange(0, len(train_questions)), 2646, replace=False)
        subsample_idxs = np.concatenate([subsample_idxs1, subsample_idxs2])
        np.random.shuffle(subsample_idxs)
    else:
        1/0

        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, "subsample_idxs.npy"), subsample_idxs)


    train_dataset = SupervisedDataset(train_questions[subsample_idxs], train_answers[subsample_idxs], tokenizer=tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)

def train():
    model_name_or_path="meta-llama/Meta-Llama-3-8B"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_points", type=int)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--save_intermediate", type=bool, default=True)
    parser.add_argument("--train_type", type=str)


    args = parser.parse_args()
    num_train_points = args.num_train_points
    num_epochs = args.num_epochs
    train_type = args.train_type
    
    project_name = "math_aug3"
    run_name  =f"{train_type}_total{num_train_points}_epochs{num_epochs}"
    output_dir = f"ckpts/{project_name}_{run_name}"

    batch_size=24
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

    if args.save_intermediate:
        # save every 2 epochs
        save_steps = num_train_points // batch_size * 2
        
        save_strategy = "steps"
    else:
        save_steps = None
        save_strategy = "no"


    training_args = TrainingArguments(
        num_train_epochs = num_epochs, 
        per_device_train_batch_size = per_device_batch_size,
        per_device_eval_batch_size = per_device_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        lr_scheduler_type = "linear",
        warmup_steps = 20,
        learning_rate = 5e-5,
        max_grad_norm = 2,
        optim = "adamw_torch",
        output_dir = output_dir,
        evaluation_strategy = "no",
        # eval_steps = 25,
        logging_strategy = "no",
        report_to = "none",
        # logging_steps = 25,
        save_strategy = save_strategy,
        save_steps=save_steps,
        # save_strategy = "epoch",
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
        cache_dir=cache_dir
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_auth_token = use_auth_token,
        model_max_length=1024,
        padding_side="right",
        cache_dir=cache_dir)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(train_type, output_dir, num_train_points, tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    
    trainer.save_state()
    trainer.save_model(output_dir=output_dir)


if __name__ == "__main__":
    train()
