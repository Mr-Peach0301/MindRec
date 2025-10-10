import math
import re
from collections import defaultdict

def update_counts_from_strings(existing_dict, slist):
    count_dict = defaultdict(int, existing_dict)
    pattern = re.compile(r'<([^>]+)>')
    
    for s in slist:
        matches = pattern.findall(s)
        for item in matches:
            count_dict[f"<{item}>"] += 1
    return dict(count_dict)

def get_topk_results(predictions, scores, targets, k, task, code2num, all_items=None):
    results = []
    out_of_dataset = 0
    B = len(targets)
    predictions = [_.split("### Response:assistant\n\n")[-1] for _ in predictions]
    predictions = [_.split("-")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ","") for _ in predictions]
    code2num = update_counts_from_strings(code2num, predictions)

    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000
                out_of_dataset += 1

    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results, out_of_dataset, code2num

def get_topk_results_without_beam(predictions, targets, k, task, code2num, all_items=None):
    results = []
    out_of_dataset = 0
    B = len(targets)
    predictions = [_.split("### Response:assistant\n\n")[-1] for _ in predictions]
    predictions = [_.split("-")[-1] for _ in predictions]
    print(predictions)
    predictions = [_.strip().replace(" ","") for _ in predictions]
    code2num = update_counts_from_strings(code2num, predictions)

    for b in range(B):
        if targets[b] == predictions[b]:
            results.append([1])
        else:
            results.append([0])

    print(results)

    return results, out_of_dataset, code2num

def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError

    return res


def ndcg_k(topk_results, k):
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            if res[i] == 1:
                one_ndcg += res[i] / math.log(i + 2, 2)
                break
        ndcg += one_ndcg
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit

