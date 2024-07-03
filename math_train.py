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
from tokenizers.processors import TemplateProcessing

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    dataset = load_dataset("hendrycks/competition_math", cache_dir=cache_dir, trust_remote_code=True)
    
    train_questions = np.array(dataset["train"]["problem"])
    train_answers = np.array(dataset["train"]['solution'])

    train_questions  = train_questions
    train_answers = train_answers

    train_dataset = SupervisedDataset(train_questions, train_answers, tokenizer=tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)

def train():
    model_name_or_path="meta-llama/Meta-Llama-3-8B"

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int)
    args = parser.parse_args()

    num_epochs = args.num_epochs

    project_name = "math_orig"
    run_name  = f"{num_epochs}epochs"
        
    training_args = TrainingArguments(
        num_train_epochs = num_epochs, 
        per_device_train_batch_size = 3,
        per_device_eval_batch_size = 3,
        gradient_accumulation_steps = 2,
        lr_scheduler_type = "linear",
        warmup_steps = 20,
        learning_rate = 5e-5,
        max_grad_norm = 2,
        optim = "adamw_torch",
        output_dir = f"ckpts/{project_name}_{run_name}",
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

    data_module = make_supervised_data_module(tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    
    if training_args.save_strategy == "no":
        trainer.save_state()
        trainer.save_model(output_dir=f"ckpts/{project_name}_{run_name}")


if __name__ == "__main__":
    train()
