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

def make_supervised_data_module(output_dir, easy_ratio, medium_ratio, hard_ratio, num_train_points, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
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
    
    
    train_num_correct = (np.concatenate([np.load("data/MATH_aug/train_aug_1_answer_types5_seed2.npy"), np.load("data/MATH_aug/train_aug_2_answer_types5_seed2.npy")], axis=0)==0).sum(axis=-1)
    easy_idxs = np.where(train_num_correct==5)[0]
    medium_idxs = np.where((train_num_correct<5)*(train_num_correct>0))[0]
    hard_idxs = np.where(train_num_correct==0)[0]
    
    subsample_easy_idxs = np.random.choice(easy_idxs, int(easy_ratio*num_train_points), replace=False)
    subsample_medium_idxs = np.random.choice(medium_idxs, int(medium_ratio*num_train_points), replace=False)
    subsample_hard_idxs = np.random.choice(hard_idxs, int(hard_ratio*num_train_points), replace=False)
    
    subsample_idxs = np.concatenate([subsample_easy_idxs, subsample_medium_idxs, subsample_hard_idxs])
    np.random.shuffle(subsample_idxs)
    
    print(f"Easy: {len(subsample_easy_idxs)}")
    print(f"Medium: {len(subsample_medium_idxs)}")
    print(f"Hard: {len(subsample_hard_idxs)}")
    print(f"Total: {len(subsample_idxs)}")
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, "subsample_idxs.npy"), subsample_idxs)


    train_dataset = SupervisedDataset(train_questions[subsample_idxs], train_answers[subsample_idxs], tokenizer=tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)

def train():
    model_name_or_path="meta-llama/Meta-Llama-3-8B"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard_ratio", type=float)
    parser.add_argument("--medium_ratio", type=float)
    parser.add_argument("--easy_ratio", type=float)
    parser.add_argument("--num_train_points", type=int)
    parser.add_argument("--num_epochs", type=int, default=10)

    args = parser.parse_args()

    hard_ratio = args.hard_ratio
    medium_ratio = args.medium_ratio
    easy_ratio = args.easy_ratio
    num_train_points = args.num_train_points
    num_epochs = args.num_epochs

    print(hard_ratio)
    print(medium_ratio)
    print(easy_ratio)
    
    assert(hard_ratio+medium_ratio+easy_ratio == 1)
    
    project_name = "math_aug"
    run_name  =f"easy{easy_ratio:.2f}_medium{medium_ratio:.2f}_hard{hard_ratio:.2f}_total{num_train_points}"
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

    assert(gradient_accumulation_steps*2*num_devices==batch_size)

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
        save_strategy = "no",
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

    data_module = make_supervised_data_module(output_dir, easy_ratio, medium_ratio, hard_ratio, num_train_points, tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    
    trainer.save_state()
    trainer.save_model(output_dir=output_dir)


if __name__ == "__main__":
    train()
