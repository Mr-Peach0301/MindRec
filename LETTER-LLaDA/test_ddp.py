import argparse
import json
import os
import sys
import re
from collections import defaultdict
import math

import torch
import transformers
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from utils import *
from collator import TestCollator
from prompt import all_prompt
from evaluate import get_topk_results, get_metrics_results

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def select_top_combinations(probs, input_tensor, input_scores, allowed_tokens, num_beam, block_start, block_end, pad_token=126336):
    device = probs.device
    probs = probs[:, block_start:block_end]
    batch_size, block_length, vocab_size = probs.shape
    input_length = input_tensor.shape[1]-block_length

    # Create mask using vectorized operations
    mask = torch.zeros((block_length, vocab_size), device=device)
    for seq_idx in allowed_tokens:
        mask[seq_idx, list(allowed_tokens[seq_idx])] = 1
    probs = probs * mask.unsqueeze(0)

    pad_mask = (input_tensor[..., block_start:block_end] == pad_token).unsqueeze(-1)  # [batch_size, block_length, 1]
    probs = probs * pad_mask.float()

    # Get topk probabilities and tokens
    topk_probs, topk_tokens = torch.topk(probs, k=num_beam, dim=-1)  # [batch_size, block_length, num_beam]
    
    # Flatten and get global topk
    flat_topk_probs = topk_probs.view(batch_size, -1)  # [batch_size, block_length * num_beam]
    flat_topk_tokens = topk_tokens.view(batch_size, -1)

    global_topk_indices = torch.multinomial(flat_topk_probs, num_samples=num_beam, replacement=False)  # [batch_size, num_beam]
    global_topk_probs = torch.gather(flat_topk_probs, 1, global_topk_indices)  # [batch_size, num_beam]

    seq_indices = global_topk_indices // num_beam + input_length  # [batch_size, num_beam]
    token_indices = torch.gather(flat_topk_tokens, 1, global_topk_indices)  # [batch_size, num_beam]
    
    # Create beam tensor and scores using vectorized operations
    beam_tensor = input_tensor.unsqueeze(1).expand(-1, num_beam, -1).clone()  # [batch_size, num_beam, block_length]
    beam_score = input_scores.unsqueeze(1).expand(-1, num_beam, -1).clone()  # [batch_size, num_beam, 1]
    
    # Use scatter_ to update the beam tensor
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_beam)
    beam_tensor[batch_indices, torch.arange(num_beam, device=device).unsqueeze(0), seq_indices] = token_indices
    
    # Update scores
    beam_score *= global_topk_probs.unsqueeze(-1)
    
    return beam_tensor, beam_score

def topk_without_duplicates(new_beam_scores, new_beam_sequences, num_beam):
    batch_size = new_beam_scores.size(0)
    top_scores_list = []
    top_indices_list = []
    
    for i in range(batch_size):
        batch_scores = new_beam_scores[i].squeeze(-1)
        batch_sequences = new_beam_sequences[i]

        seen_sequences = set()
        selected_indices = []
        selected_scores = []

        sorted_scores, sorted_indices = torch.sort(batch_scores, descending=True)
        
        for score, idx in zip(sorted_scores, sorted_indices):
            if len(selected_indices) >= num_beam:
                break
            sequence_tuple = tuple(batch_sequences[idx].cpu().numpy().tolist())
            
            if sequence_tuple not in seen_sequences:
                seen_sequences.add(sequence_tuple)
                selected_indices.append(idx)
                selected_scores.append(score)

        top_indices_list.append(torch.stack(selected_indices))
        top_scores_list.append(torch.stack(selected_scores))
    
    top_indices = torch.stack(top_indices_list)
    top_scores = torch.stack(top_scores_list)
    
    return top_scores, top_indices

@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, cfg_scale=0.,
            remasking='low_confidence', mask_id=126336, num_beam=4, allowed_tokens=None):
    with torch.cuda.amp.autocast(enabled=True):
        x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt
    
        beam_sequences = x.unsqueeze(1)
        beam_scores = torch.full((prompt.shape[0], 1, 1), 1, dtype=torch.bfloat16, device=model.device)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
    
        for block_idx in range(num_blocks):

            block_start = prompt.shape[1] + block_idx * block_length
            block_end = prompt.shape[1] + (block_idx + 1) * block_length

            block_sequences = beam_sequences.unsqueeze(2)
            block_scores = beam_scores.unsqueeze(2)

            for step in range(steps_per_block):

                input_tensor = block_sequences.reshape(-1, block_sequences.shape[-1])
                input_scores = block_scores.reshape(-1, block_scores.shape[-1])

                logits = model(input_tensor).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=args.gen_temperature)
                probs = F.softmax(logits_with_noise, dim=-1)

                candidates, candidate_scores = select_top_combinations(probs, input_tensor, input_scores, allowed_tokens, num_beam, block_start, block_end)
            
                candidates = candidates.reshape(block_sequences.shape[0], block_sequences.shape[1], -1, block_sequences.shape[-1])
                candidate_scores = candidate_scores.reshape(block_scores.shape[0], block_scores.shape[1], -1, block_scores.shape[-1])

                if candidates.shape[2] > num_beam:
                    unique_candidates = []
                    unique_scores = []
                    unique_indices = []
    
                    for i in range(candidates.shape[0]):
                        batch_candidates = candidates[i]
                        batch_scores = candidate_scores[i].squeeze(-1)
        
                        seen = set()
                        temp_candidates = []
                        temp_scores = []
                        temp_indices = []
        
                        sorted_indices = torch.argsort(batch_scores, dim=1, descending=True)
        
                        for j in range(sorted_indices.shape[1]):
                            idx = sorted_indices[0, j].item()
                            candidate_seq = tuple(batch_candidates[0, idx].tolist())
            
                            if candidate_seq not in seen:
                                seen.add(candidate_seq)
                                temp_candidates.append(batch_candidates[:, idx:idx+1, :])
                                temp_scores.append(batch_scores[:, idx:idx+1])
                                temp_indices.append(idx)
                
                                if len(temp_candidates) >= num_beam:
                                    break
        
                        while len(temp_candidates) < num_beam and j < sorted_indices.shape[1] - 1:
                            j += 1
                            idx = sorted_indices[0, j].item()
                            temp_candidates.append(batch_candidates[:, idx:idx+1, :])
                            temp_scores.append(batch_scores[:, idx:idx+1])
                            temp_indices.append(idx)
        
                        unique_candidates.append(torch.cat(temp_candidates, dim=1))
                        unique_scores.append(torch.cat(temp_scores, dim=1))
                        unique_indices.append(torch.tensor(temp_indices, device=candidates.device))
    
                    block_sequences = torch.stack(unique_candidates)
                    block_scores = torch.stack(unique_scores).unsqueeze(-1)
                else:
                    block_sequences = candidates
                    block_scores = candidate_scores
                del candidates, candidate_scores, probs

            new_beam_sequences = block_sequences.reshape(block_sequences.shape[0], block_sequences.shape[1]*block_sequences.shape[2], -1)
            new_beam_scores = block_scores.reshape(block_scores.shape[0], block_scores.shape[1]*block_scores.shape[2], -1)

            if new_beam_scores.shape[1] > num_beam:
                top_scores, top_indices = topk_without_duplicates(
                    new_beam_scores, new_beam_sequences, num_beam
                )
                # top_scores, top_indices = torch.topk(new_beam_scores.squeeze(-1), k=num_beam, dim=1)
                beam_sequences = torch.gather(
                    new_beam_sequences, 
                    dim=1, 
                    index=top_indices.unsqueeze(-1).expand(-1, -1, new_beam_sequences.size(-1))
                )
                beam_scores = torch.gather(
                    new_beam_scores,
                    dim=1,
                    index=top_indices.unsqueeze(-1)
                )
            else:
                beam_sequences = new_beam_sequences
                beam_scores = new_beam_scores
            del new_beam_sequences, new_beam_scores
    
        return beam_sequences, beam_scores

def test_ddp(args):

    set_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(vars(args))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

    device_map = {"": local_rank}
    device = torch.device("cuda",local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    if args.lora:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True
        )
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(
            model,
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path,
            torch_dtype=torch.bfloat16,              
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True
        )
    # assert model.config.vocab_size == len(tokenizer)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    if args.test_prompt_ids == "all":
        if args.test_task.lower() == "seqrec":
            prompt_ids = range(len(all_prompt["seqrec"]))
        elif args.test_task.lower() == "itemsearch":
            prompt_ids = range(len(all_prompt["itemsearch"]))
        elif args.test_task.lower() == "fusionseqrec":
            prompt_ids = range(len(all_prompt["fusionseqrec"]))
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    test_data = load_test_dataset(args)
    ddp_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=local_rank, drop_last=True)

    test_data = load_test_dataset(args)
    tokenizer.pad_token_id = (
        126084  # <|reserved_token_0|>. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             sampler=ddp_sampler, num_workers=2, pin_memory=True)

    if local_rank == 0:
        print("data num:", len(test_data))

    model.eval()

    allowed_tokens = {}
    for index in test_data.indices.values():
        for i, token in enumerate(index):
            token_id = tokenizer(token)["input_ids"][0]
            if i not in allowed_tokens.keys():
                allowed_tokens[i] = set()
            allowed_tokens[i].add(token_id)
    allowed_tokens[len(allowed_tokens.keys())] = set([tokenizer.eos_token_id])

    metrics = args.metrics.split(",")
    all_prompt_results = []
    with torch.no_grad():
        for prompt_id in prompt_ids:
            if local_rank == 0:
                print("Start prompt: ",prompt_id)

            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0
            CC = 0
            code2num = {}

            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                bs = len(targets)
                num_beams = args.num_beams
                while True:
                    try:
                        with torch.cuda.amp.autocast():
                            output_ids, output_scores = generate(
                                model,
                                inputs["input_ids"],
                                steps=5,
                                gen_length=5,
                                block_length=5,
                                cfg_scale=args.cfg_scale,
                                remasking='low_confidence',
                                num_beam=10,
                                allowed_tokens=allowed_tokens,
                            )
                        break
                    except torch.cuda.OutOfMemoryError as e:
                        print("Out of memory!")
                        num_beams = num_beams -1
                        print("Beam:", num_beams)
                    except Exception:
                        raise RuntimeError

                output = []
                scores = []
                for j in range(len(output_ids)):
                    decoded_beams = [
                        tokenizer.decode(beam, skip_special_tokens=True)
                        for beam in output_ids[j]
                    ]
                    output += decoded_beams
                    scores += output_scores[j].tolist()

                topk_res, out_of_dataset, code2num = get_topk_results(output, scores, targets, num_beams, code2num,
                                            all_items=all_items if args.filter_items else None)
                CC += out_of_dataset

                bs_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(obj=bs, object_list=bs_gather_list)
                total += sum(bs_gather_list)
                res_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(obj=topk_res, object_list=res_gather_list)


                if local_rank == 0:
                    all_device_topk_res = []
                    for ga_res in res_gather_list:
                        all_device_topk_res += ga_res
                    batch_metrics_res = get_metrics_results(all_device_topk_res, metrics)
                    for m, res in batch_metrics_res.items():
                        if m not in metrics_results:
                            metrics_results[m] = res
                        else:
                            metrics_results[m] += res

                    if (step + 1) % 50 == 0:
                        temp = {}
                        for m in metrics_results:
                            temp[m] = metrics_results[m] / total
                        print(temp)

                dist.barrier()

            def calculate_metrics(code2num):
                total_tokens = sum(code2num.values())
                num_types = len(code2num)
                entropy = 0.0
                for count in code2num.values():
                    if count == 0:
                        continue
                    p = count / total_tokens
                    entropy -= p * math.log2(p)
    
                ttr = num_types / total_tokens if total_tokens > 0 else 0
                return entropy, ttr

            if local_rank == 0:
                for m in metrics_results:
                    metrics_results[m] = metrics_results[m] / total
                    print(total)
                metrics_results['CC'] = CC
                metrics_results['H'], metrics_results['TTR'] = calculate_metrics(code2num)

                all_prompt_results.append(metrics_results)
                print("======================================================")
                print("Prompt {} results: ".format(prompt_id), metrics_results)
                print("======================================================")
                print("")

            dist.barrier()

    dist.barrier()

    if local_rank == 0:
        mean_results = {}
        min_results = {}
        max_results = {}

        for m in metrics:
            all_res = [_[m] for _ in all_prompt_results]
            mean_results[m] = sum(all_res)/len(all_res)
            min_results[m] = min(all_res)
            max_results[m] = max(all_res)

        print("======================================================")
        print("Mean results: ", mean_results)
        print("Min results: ", min_results)
        print("Max results: ", max_results)
        print("======================================================")


        save_data={}
        save_data["test_prompt_ids"] = args.test_prompt_ids
        save_data["mean_results"] = mean_results
        save_data["min_results"] = min_results
        save_data["max_results"] = max_results
        save_data["all_prompt_results"] = all_prompt_results

        with open(args.results_file, "w") as f:
            json.dump(save_data, f, indent=4)
        print("Save file: ", args.results_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    test_ddp(args)
