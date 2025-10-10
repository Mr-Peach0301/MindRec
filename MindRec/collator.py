import torch
import copy
import argparse
from dataclasses import dataclass
import re
import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist

class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 126084  # <|reserved_token_0|>. we want this to be different from the eos token

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        full_texts = [d["labels"] + self.tokenizer.eos_token for d in batch]
        inputs = self.tokenizer(
            text = full_texts,
            text_target = input_texts,
            return_tensors="pt",
            padding=True,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        labels = copy.deepcopy(inputs["input_ids"])
        if self.only_train_response:
            # ignore padding
            labels[labels == self.tokenizer.pad_token_id] = -100
            # ignore input text
            labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100
        inputs["labels"] = labels
        return inputs


class TestCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 126084  # <|reserved_token_0|>. we want this to be different from the eos token

    def __call__(self, batch):
        
        input_texts = []
        for d in batch:
            m = [{"role": "user", "content": d["input_ids"]}]
            input_text = self.tokenizer.apply_chat_template(
                m, 
                add_generation_prompt=True, 
                tokenize=False
            )
            pattern = r'<[a-zA-Z]+[:_]\d+>'
            mask_response = re.sub(pattern, '<|mdm_mask|>', d["labels"])
            input_text = input_text + mask_response + "<|endoftext|>"
            input_texts.append(input_text)
        

        targets = [d["labels"] for d in batch]
        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        return (inputs, targets)

