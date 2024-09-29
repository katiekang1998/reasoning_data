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

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

model_orig = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)

params = []
for key in model.state_dict().keys():
    if key != "model.embed_tokens.weight" and key!="lm_head.weight":
        params.append(model.state_dict()[key].view(-1))

params = torch.cat(params)


params_orig = []
for key in model_orig.state_dict().keys():
    if key != "model.embed_tokens.weight" and key!="lm_head.weight":
        params_orig.append(model_orig.state_dict()[key].view(-1))
    
params_orig = torch.cat(params_orig)

print(model_name)
weight_norm = ((params - params_orig)**2).sum()
print(weight_norm.item())

np.save(model_name+"/weight_norm.npy", weight_norm.item())