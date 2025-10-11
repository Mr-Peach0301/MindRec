import argparse
import os
import sys
from typing import List
import torch
from torch.optim import AdamW
import transformers
import math
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
import gc
from collections import defaultdict

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from utils import *
from torch.utils.tensorboard import SummaryWriter

class MyDataset(Dataset):
    def __init__(self, dataset, tokenizer, test_batch_size):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.test_batch_size = test_batch_size
        self.processed_data = [self._process_item(item) for item in dataset]

    def __len__(self):
        return len(self.dataset)

    def _process_item(self, item):    
        m = [{"role": "user", "content": item['input_ids']}]
        prompt = self.tokenizer.apply_chat_template(
            m, 
            add_generation_prompt=True, 
            tokenize=False
        )
        prompt_lengths = len(self.tokenizer(prompt, truncation=True, max_length=self.tokenizer.model_max_length)['input_ids'])
        
        output = item['labels'].rsplit('\n', 1)[-1]
        input_text = prompt + output + "<|endoftext|>"
        input_ids = self.tokenizer(input_text, truncation=True, max_length=self.tokenizer.model_max_length)['input_ids']
        
        return {
            "input_ids": input_ids,
            "prompt_lengths": prompt_lengths
        }
    
    def __getitem__(self, idx):
        return self.processed_data[idx]
        # return self._process_item(self.dataset[idx]) 

def forward_process(input_ids, eps=1e-3):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    # 126336 is used for [MASK] token
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    return noisy_batch, masked_indices, p_mask

def split_into_batches(data, batch_size=32, pad_value=126081):
    input_ids = data["input_ids"]
    prompt_lengths = data["prompt_lengths"]
    batches = []

    total_samples = len(input_ids)

    for start_idx in range(0, total_samples, batch_size):
        end_idx = start_idx + batch_size
        batch_input_ids = input_ids[start_idx:end_idx]
        batch_prompt_lengths = prompt_lengths[start_idx:end_idx]
        max_length = max(len(ids) for ids in batch_input_ids)

        padded_input_ids = [
            ids + [pad_value] * (max_length - len(ids)) for ids in batch_input_ids
        ]

        current_batch = {
            "input_ids": padded_input_ids,
            "prompt_lengths": batch_prompt_lengths
        }
        batches.append(current_batch)

    return batches

def get_linear_schedule_with_warmup_and_cooldown(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    num_cooldown_steps, 
    max_lr=2.5e-5, 
    min_lr=2.5e-6
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        elif current_step >= (num_training_steps - num_cooldown_steps):
            progress = float(current_step - (num_training_steps - num_cooldown_steps)) / float(max(1, num_cooldown_steps))
            return max(min_lr / max_lr, 1.0 - progress)\

        return 1.0

    return LambdaLR(optimizer, lr_lambda)

def train(args):

    device = 'cuda'
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["WANDB_DISABLED"] = "true"

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    if ddp:
        device_map = {"": local_rank}

    config = PretrainedConfig.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length
    )
    tokenizer.pad_token_id = (
        126084  # <|reserved_token_0|>. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    train_dataset, valid_dataset = load_datasets(args)
    add_num = tokenizer.add_tokens(train_dataset.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)

    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_dataset))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )

    model.resize_token_embeddings(len(tokenizer))

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        modules_to_save=args.lora_modules_to_save.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)

    for n, p in model.named_parameters():
        if "original_module" in n and any(module_name in n for module_name in config.modules_to_save):
            p.requires_grad = False
    
    if local_rank == 0:
        model.print_trainable_parameters()

    total_steps = (len(train_dataset) // args.per_device_batch_size) * args.epochs
    warmup_steps = 50
    cooldown_steps = int(0.1 * total_steps)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = get_linear_schedule_with_warmup_and_cooldown(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cooldown_steps=cooldown_steps,
        max_lr=args.learning_rate,
        min_lr=args.learning_rate*0.1
    )
    def collate_fn(batch):
        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        prompt_lengths = torch.tensor([item["prompt_lengths"] for item in batch], dtype=torch.long)
        
        max_len = max(len(seq) for seq in input_ids)

        prompt_lengths = torch.tensor([
            pl + (max_len - len(seq))
            for seq, pl in zip(input_ids, prompt_lengths)
        ], dtype=torch.long)

        input_ids = torch.stack([
            F.pad(
                seq,
                (max_len - len(seq), 0),
                value=126084,
            )
            for seq in input_ids
        ])

        return {
            "input_ids": input_ids,
            "prompt_lengths": prompt_lengths
        }

    train_dataset = MyDataset(train_dataset, tokenizer, test_batch_size=args.test_batch_size)
    valid_dataset = MyDataset(valid_dataset, tokenizer, test_batch_size=args.test_batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_batch_size, collate_fn=collate_fn, shuffle=args.shuffle, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.per_device_batch_size, collate_fn=collate_fn, num_workers=4)

    for epoch in range(args.epochs):
        # train
        model.train()
        t_loss = 0
        step = 0

        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)

            noisy_batch, _, p_mask = forward_process(input_ids, eps=args.eps)

            # Do not add noise to the prompt
            token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
            prompt_mask = (token_positions < prompt_lengths.unsqueeze(1)).detach()
            noisy_batch[prompt_mask] = input_ids[prompt_mask]

            # Calculate the answer length (including the padded <EOS> tokens)
            prompt_mask = prompt_mask.to(torch.int64)
            answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
            answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])   

            masked_indices = (noisy_batch == 126336)
            logits = model(input_ids=noisy_batch).logits
            token_loss = F.cross_entropy(logits[masked_indices]/args.temperature, input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
            train_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            t_loss += train_loss.item()
            step += 1

            if local_rank == 0:
                writer.add_scalar('Loss/train', train_loss.item(), epoch * len(train_loader) + step)
                writer.add_scalar('Learning Rate', current_lr, epoch * len(train_loader) + step)
                print(f"Epoch {epoch + 1}, Train Loss = {train_loss.item()}, lr = {current_lr:.2e}")

        if local_rank == 0:
            print(f"Epoch {epoch + 1}, Average Train Loss: {t_loss/len(train_loader)}")

        # valid
        model.eval()
        with torch.inference_mode():
            
            v_loss = 0
            for batch in tqdm(valid_loader):
                input_ids, prompt_lengths = batch["input_ids"], batch["prompt_lengths"]

                input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
                prompt_lengths = torch.tensor(prompt_lengths, dtype=torch.long).to(device)

                noisy_batch, _, p_mask = forward_process(input_ids, eps=args.eps)

                # Do not add noise to the prompt
                token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
                prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
                noisy_batch[prompt_mask] = input_ids[prompt_mask]

                # Calculate the answer length (including the padded <EOS> tokens)
                prompt_mask = prompt_mask.to(torch.int64)    
                answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
                answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])    

                masked_indices = (noisy_batch == 126336)

                logits = model(input_ids=noisy_batch).logits
                token_loss = F.cross_entropy(logits[masked_indices]/args.temperature, input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
                valid_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]

                v_loss += valid_loss.item()

                if local_rank == 0:
                    writer.add_scalar('Loss/valid', valid_loss.item(), epoch * len(valid_loader) + step)

                print(f"Epoch {epoch + 1}, Valid Loss: {valid_loss.item()}")

            v_loss = v_loss/len(valid_loader)
        
            if local_rank == 0:
                print(f"Epoch {epoch + 1}, Average Valid Loss: {v_loss}")

        torch.cuda.empty_cache()
    
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    train(args)
