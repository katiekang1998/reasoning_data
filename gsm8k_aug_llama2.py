import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import os
from datasets import load_dataset
import numpy as np
import argparse
import json
from huggingface_params import cache_dir, use_auth_token
from utils import *


def make_supervised_data_module(output_dir, data_type,num_train_points, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
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


    num_correct1 = []
    num_correct2 = []

    for seed in range(4):
        num_correct1.append((np.load(f"ckpts/gsm8k_all_2epochs_llama2/train_aug_1_answer_types5_seed{seed}.npy")==0).sum(axis=-1))
        num_correct2.append((np.load(f"ckpts/gsm8k_all_2epochs_llama2/train_aug_2_answer_types5_seed{seed}.npy")==0).sum(axis=-1))

        
    num_correct1 = np.sum(num_correct1, axis=0)
    num_correct2 = np.sum(num_correct2, axis=0)
    num_correct = np.concatenate([num_correct1, num_correct2])
        
        
    if data_type == "hard":
        idxs = np.where(num_correct ==0)[0]
        subsample_idxs = np.random.choice(idxs, num_train_points, replace=False)
    elif data_type == "hard2":
        idxs = np.where(num_correct <2)[0]
        subsample_idxs = np.random.choice(idxs, num_train_points, replace=False)
    elif data_type == "hard3":
        idxs = np.where(num_correct <3)[0]
        subsample_idxs = np.random.choice(idxs, num_train_points, replace=False)
    elif data_type == "hard10":
        idxs = np.where(num_correct <10)[0]
        subsample_idxs = np.random.choice(idxs, num_train_points, replace=False)
    elif data_type == "hard10_mixed":
        idxs1 = np.where(num_correct <5)[0]
        idxs2 = np.where(num_correct >=5)[0]
        
        subsample_idxs1 = np.random.choice(idxs1, int(num_train_points*0.6365472833634948), replace=False)
        subsample_idxs2 = np.random.choice(idxs2, int(num_train_points*(1-0.6365472833634948)), replace=False)

        subsample_idxs = np.concatenate([subsample_idxs1, subsample_idxs2])
        
    elif data_type == "hard15":
        idxs = np.where(num_correct <15)[0]
        subsample_idxs = np.random.choice(idxs, num_train_points, replace=False)
    elif data_type == "hard5":
        idxs = np.where(num_correct <5)[0]
        subsample_idxs = np.random.choice(idxs, num_train_points, replace=False)
    elif data_type == "rand":
        subsample_idxs = np.random.choice(np.arange(0, len(train_questions)), num_train_points , replace=False)
    else:
        raise Exception("ergewrg")
    
    np.random.shuffle(subsample_idxs)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, "subsample_idxs.npy"), subsample_idxs)

    train_questions = np.array(train_questions)
    train_answers = np.array(train_answers)
    
    train_dataset = SupervisedDataset(train_questions[subsample_idxs], train_answers[subsample_idxs], tokenizer=tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)

def train():
    model_name_or_path="NousResearch/Llama-2-7b-hf"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--num_train_points", type=int)
    
    args = parser.parse_args()

    data_type = args.data_type
    num_train_points = args.num_train_points
    
    
    project_name = "gsm8k_aug_llama2"
    run_name  = f"{data_type}_{str(num_train_points)}"
    
    batch_size = 24
    num_devices = torch.cuda.device_count()
    gradient_accumulation_steps = int((batch_size/2)/num_devices)
    
    assert(gradient_accumulation_steps*2*num_devices == batch_size)
    
    if num_train_points>=20000:
        num_train_epochs = 5
    else:
        num_train_epochs = 10
        
    output_dir = f"ckpts/{project_name}_{run_name}"
    
    training_args = TrainingArguments(
        num_train_epochs = num_train_epochs, 
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 2,
        gradient_accumulation_steps = gradient_accumulation_steps,
        lr_scheduler_type = "linear",
        warmup_steps = 20,
        learning_rate = 5e-5,
        max_grad_norm = 2,
        optim = "adamw_torch",
        output_dir = output_dir,
        evaluation_strategy = "no",
        # eval_steps = 25,
        report_to = "none",
        logging_strategy = "steps",
        logging_steps = 25,
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
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
        add_eos_token=False,
    )

    data_module = make_supervised_data_module(output_dir, data_type,num_train_points, tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    
    if training_args.save_strategy == "no":
        trainer.save_state()
        trainer.save_model(output_dir=output_dir)


if __name__ == "__main__":
    train()