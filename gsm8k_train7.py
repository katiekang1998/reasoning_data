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
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers.utils import is_datasets_available
import datasets
from transformers.trainer_utils import seed_worker


def make_supervised_data_module(output_dir, train_type, tokenizer: transformers.PreTrainedTokenizer) -> Dict:

    dataset = load_dataset("gsm8k", "main", cache_dir=cache_dir)
    train_questions_orig = np.array(dataset["train"]["question"])
    train_answers_orig = np.array(dataset["train"]['answer'])
    # test_questions_orig = np.array(dataset["test"]["question"])
    # test_answers_orig = np.array(dataset["test"]['answer'])
    
    # if train_type == "memorized":
    #     subsample_idxs = np.load("ckpts/gsm8k_orig_6epochs/memorized_subsample_idxs.npy")
    # if train_type == "unmemorized_gt_1":
    #     subsample_idxs = np.load("gsm8k_unmemorized_idxs>1.npy")
    # elif (train_type == "full" or "shuffle" in train_type):
    #     subsample_idxs = np.arange(len(train_questions_orig))
    # elif train_type == "half":
    #     subsample_idxs= np.arange(len(train_questions_orig))
    #     subsample_idxs = np.random.choice(subsample_idxs, len(subsample_idxs)//2, replace=False)
    # elif train_type == "quarter":
    #     subsample_idxs= np.arange(len(train_questions_orig))
    #     subsample_idxs = np.random.choice(subsample_idxs, len(subsample_idxs)//4, replace=False)
    # elif train_type == "eighth":
    #     subsample_idxs= np.arange(len(train_questions_orig))
    #     subsample_idxs = np.random.choice(subsample_idxs, len(subsample_idxs)//8, replace=False)
    # else:
    #     1/0
        
    unmemorized_acc_cummax_all = np.load("ckpts/gsm8k_orig_3epochs_shuffle1_lr2e-05_bs128/unmemorized_acc_cummax_all.npy").max(axis=0)
    if train_type == "memorized_half":
        sorted_idxs = np.argsort(unmemorized_acc_cummax_all) #lowest to highest
        subsample_idxs = sorted_idxs[:len(sorted_idxs)//2]
    elif train_type == "unmemorized_half":
        sorted_idxs = np.argsort(unmemorized_acc_cummax_all) #lowest to highest
        subsample_idxs = sorted_idxs[len(sorted_idxs)//2:]
    elif train_type == "unmemorized_3quarter":
        sorted_idxs = np.argsort(unmemorized_acc_cummax_all) #lowest to highest
        subsample_idxs = sorted_idxs[len(sorted_idxs)//4:]
    elif train_type == "memorized_3quarter":
        sorted_idxs = np.argsort(unmemorized_acc_cummax_all) #lowest to highest
        subsample_idxs = sorted_idxs[:-len(sorted_idxs)//4]
    elif train_type == "memorized_3quarter":
        sorted_idxs = np.argsort(unmemorized_acc_cummax_all) #lowest to highest
        subsample_idxs = sorted_idxs[:-len(sorted_idxs)//4]
    elif train_type == "middle_half":
        subsample_idxs = np.where((unmemorized_acc_cummax_all>0.12)*(unmemorized_acc_cummax_all<0.87))[0]
    elif train_type == "unmemorized_acc_gt_0":
        subsample_idxs = np.where((unmemorized_acc_cummax_all>0))[0]
    elif train_type == "unmemorized_acc_lt_1":
        subsample_idxs = np.where((unmemorized_acc_cummax_all<1))[0]
    elif train_type == "unmemorized_acc_gt_0_lt_1":
        subsample_idxs = np.where((unmemorized_acc_cummax_all>0)*(unmemorized_acc_cummax_all<1))[0]
    elif train_type == "remove_middle":
        sorted_idxs = np.argsort(unmemorized_acc_cummax_all)
        removed_idxs_len = len(sorted_idxs) - len(np.where((unmemorized_acc_cummax_all>0)*(unmemorized_acc_cummax_all<1))[0])
        sorted_idxs = np.argsort(unmemorized_acc_cummax_all) #lowest to highest
        subsample_idxs = np.concatenate([sorted_idxs[:1000] , sorted_idxs[1000+removed_idxs_len:]])
        
    elif train_type == "full":
        sorted_idxs = np.argsort(unmemorized_acc_cummax_all) #lowest to highest
        subsample_idxs = sorted_idxs[:]
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, "subsample_idxs.npy"), subsample_idxs)
    print(len(subsample_idxs))
    train_dataset = SupervisedDataset(train_questions_orig[subsample_idxs], train_answers_orig[subsample_idxs], tokenizer=tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)

def train():
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_type", type=str, default="full")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora", type=bool, default=False)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")


    args = parser.parse_args()
    train_type = args.train_type
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    use_lora = args.lora
    model_name_or_path = args.model
    
    
    project_name = "gsm8k_filter"
    
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

    # save_steps = 7473 // batch_size * 27
    

    save_strategy = "no"
    save_steps = None
    
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
        save_strategy = save_strategy,
        save_steps=save_steps,
        save_only_model = True,
        run_name=run_name,
        # bf16 = True,
        fsdp= "full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap= 'LlamaDecoderLayer',
        # tf32 =True,
        fp16=True,
        weight_decay=0.01
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


    if "meta-llama/Meta-Llama-3-8B" in model_name_or_path:
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

    data_module = make_supervised_data_module(output_dir, train_type, tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    
    trainer.save_state()
    trainer.save_model(output_dir=output_dir)


if __name__ == "__main__":
    train()