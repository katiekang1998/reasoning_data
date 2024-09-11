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
from torch.nn import CrossEntropyLoss
import torch.distributed as dist



def average_metric_across_devices(metric_value):
    # Ensure that the metric_value is a tensor
    tensor_value = torch.tensor(metric_value).cuda()
    # Reduce across all processes (sum the metric values)
    dist.reduce(tensor_value, dst=0, op=dist.ReduceOp.SUM)
    # Divide by the world size to get the average
    if dist.get_rank() == 0:
        tensor_value /= dist.get_world_size()
    return tensor_value.item()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs, return_dict=False)
            logits = outputs[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if len(labels) == 16:
                predictions = shift_logits.argmax(-1)
                per_token_accuracy = torch.logical_or(shift_labels == -100, shift_labels == predictions)
                sa = torch.all(per_token_accuracy, dim=-1).float()
                loss = per_token_accuracy.float().mean()
                return (loss, outputs) if return_outputs else loss 
                    
            # loss_fct = CrossEntropyLoss(reduction='sum')
            # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) / labels.shape[0] / labels.shape[1]

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
            return (loss, outputs) if return_outputs else loss
    
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    if dist.get_rank() == 0:
        import IPython; IPython.embed()
    else:
        dist.barrier()
    logits = logits[1]
    predictions = np.argmax(logits, axis=-1)
    print(logits.shape, predictions.shape)
    per_token_accuracy = np.logical_or(labels == -100, labels == predictions)
    tokens_to_pred = (labels != -100).sum(-1)
    
    small = tokens_to_pred < 200
    large = tokens_to_pred > 200
    print(tokens_to_pred)
    def masked_mean(x, mask):
        return (x * mask).mean() / mask.mean()
    sa = np.all(per_token_accuracy, axis=-1)
    return {
        "accuracy": per_token_accuracy.mean(),
        "sentence_accuracy": sa.mean(),
        "small_sentence_accuracy": masked_mean(sa, small),
        "large_sentence_accuracy": masked_mean(sa, large),
        "small_accuracy":  masked_mean(per_token_accuracy.mean(-1), small),
        "large_accuracy": masked_mean(per_token_accuracy.mean(-1), large),
    }
                
    

def make_supervised_data_module(output_dir, train_type, learning_rate, batch_size, epochs, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    all_lines = open("combined_5_11_skip2.txt", "r").readlines()
    train_lines = all_lines[:-100]
    test_lines = all_lines[-99::2]
    def to_qa(lines):
        qs = [l.partition('||')[0] for l in lines]
        ans = [l.strip().partition('||')[2] for l in lines]
        return np.array(qs), np.array(ans)
    train_questions, train_answers = to_qa(train_lines)
    test_questions, test_answers = to_qa(test_lines)
        
    # dataset = load_dataset("gsm8k", "main", cache_dir=cache_dir)
    # train_questions = np.array(dataset["train"]["question"])
    # train_answers = np.array(dataset["train"]['answer'])
    # test_questions = np.array(dataset["test"]["question"])
    # test_answers = np.array(dataset["test"]['answer'])
    
    train_dataset = SupervisedDataset(train_questions, train_answers, tokenizer=tokenizer)
    test_dataset = SupervisedDataset(test_questions, test_answers, tokenizer=tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator, eval_dataset=test_dataset)

def train():
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_type", type=str, default="threshold-2")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
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
    
    
    project_name = "gsm8k_orig_sum"
    if use_lora:
        run_name  = f"{epochs}epochs_lr{learning_rate}_bs{batch_size}_lora"
    else:
        run_name  = f"{epochs}epochs_lr{learning_rate}_bs{batch_size}"
        
        
    if "Qwen-14B" in model_name_or_path:
        run_name += "_Qwen-14B"
    
    batch_size = batch_size
    num_devices = torch.cuda.device_count()
    # if "Qwen-14B" in model_name_or_path:
    #     per_device_batch_size = 1
    #     gradient_accumulation_steps = int((batch_size/1)/num_devices)
    # else:
    #     if num_devices>2:
    #         per_device_batch_size = 2
    #         gradient_accumulation_steps = int((batch_size/2)/num_devices)
    #     else:
    per_device_batch_size = 1
    gradient_accumulation_steps = int((batch_size/1)/num_devices)
    print("Num devices: ", num_devices)
    print("Per device batch: ", per_device_batch_size)
    print("Grad accum steps: ", gradient_accumulation_steps)

    assert(gradient_accumulation_steps*per_device_batch_size*num_devices==batch_size)

    # save_steps = 7473 // batch_size * 27
    
    # save_strategy = "steps"
    
    if epochs <= 6:
        save_strategy = "epoch"
        save_steps = None 
    elif epochs <= 12:
        # save every 2 epochs
        save_strategy = "steps"
        save_steps = 29
        # save_steps = 7473 // batch_size * 2

    elif epochs <= 24:
        save_strategy = "steps"
        save_steps = 29
        # save_steps = 7473 // batch_size * 2
    else:
        # save every 2 epochs
        save_steps = 7473 // batch_size * 3
        save_strategy = "steps"
    
    output_dir = f"ckpts/{project_name}_{run_name}"
    
    
    
    if "Qwen-14B" in model_name_or_path:
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
                logging_steps = 1,
                save_strategy = save_strategy,
                save_steps=save_steps,
                save_only_model = True,
                run_name=run_name,
                bf16 = True,
                fsdp= "full_shard auto_wrap",
                tf32 =True,
            )
    else:
        training_args = TrainingArguments(
            num_train_epochs = epochs, 
            per_device_train_batch_size = per_device_batch_size,
            per_device_eval_batch_size = 16,
            gradient_accumulation_steps = gradient_accumulation_steps,
            lr_scheduler_type = "linear",
            warmup_steps = 20,
            learning_rate = learning_rate,
            max_grad_norm = 2,
            optim = "adamw_torch",
            output_dir = output_dir,
            evaluation_strategy = "steps",
            eval_steps = 1,
            report_to = "none",
            logging_strategy = "steps",
            logging_steps = 1,
            save_strategy = "no",
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


    if "Qwen-14B" in model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            pad_token="<|extra_0|>",
            # eos_token='<|endoftext|>',
            padding_side='right',
            cache_dir=cache_dir,
            trust_remote_code=True
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_auth_token = use_auth_token,
            model_max_length=1024,
            padding_side="right",
            cache_dir=cache_dir,
            trust_remote_code=True)


    if use_lora:
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,  # Scaling factor for LoRA weights
            target_modules=["q_proj", "v_proj"],  # Target attention layers
            lora_dropout=0.1,  # LoRA dropout
            bias="none",
            task_type="CAUSAL_LM"  # Task type
        )
        model = get_peft_model(model, lora_config)


    if "meta-llama/Meta-Llama-3-8B" in model_name_or_path:
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

    data_module = make_supervised_data_module(output_dir, train_type, learning_rate, batch_size, epochs, tokenizer=tokenizer)
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    
    trainer.save_state()
    trainer.save_model(output_dir=output_dir)


if __name__ == "__main__":
    train()