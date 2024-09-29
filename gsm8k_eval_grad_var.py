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
from transformers import AdamW

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--ckpt", type=str)


args = parser.parse_args()


model_name = f"ckpts/{args.ckpt_dir}/checkpoint-{args.ckpt}"  # Replace with the specific LLaMA model you want
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

model.to("cuda")


# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

dataset = load_dataset("gsm8k", "main")
train_questions = dataset["train"]["question"]
train_answers = dataset["train"]['answer']

dataset2 = SupervisedDataset(train_questions, train_answers, tokenizer)
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

dataloader = DataLoader(dataset2, batch_size=1, collate_fn=data_collator)

accumulation_steps = 128
accumulated_gradients_list = []
loss_accumulation = 0.0

num_steps = 0
for batch in tqdm.tqdm(dataloader):
    batch = {key: value.to("cuda") for key, value in batch.items()}
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    
    outputs = model(input_ids=input_ids, labels=labels)
    
    loss = outputs.loss
    loss = loss / accumulation_steps  # Scale loss for accumulation

    # Backward pass (accumulate gradients)
    loss.backward()
    loss_accumulation += loss.item()
    num_steps+=1

    # Store gradients after every accumulation step
    if num_steps % accumulation_steps == 0:
        # Collect and store gradients after accumulation
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1).detach().cpu().numpy())  # Flatten gradients for easier manipulation

        # Concatenate the current accumulated gradients for the current accumulation cycle
        accumulated_gradients = np.concatenate(gradients)
        
        
        # Store the accumulated gradients
        accumulated_gradients_list.append(accumulated_gradients)

        # Perform optimization step and zero the gradients
        # optimizer.step()
        optimizer.zero_grad()

        # Print accumulated loss
        print(f"Accumulated Loss after step {num_steps}: {loss_accumulation:.4f}")
        loss_accumulation = 0.0
    
    
    if num_steps == accumulation_steps*5:    
        print("Here")    
        # accumulated_gradients_list = np.array(accumulated_gradients_list)
        grad_var = np.var(accumulated_gradients_list, axis=0).mean()
        print(grad_var)
        np.save(model_name+"/grad_var.npy", grad_var)
        print("DONE")
        exit()