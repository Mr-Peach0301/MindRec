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

def get_topk_results(predictions, scores, targets, k, code2num, all_items=None):
    results = []
    B = len(targets)
    predictions = [_.strip().replace(" ","") for _ in predictions]
    code2num = update_counts_from_strings(code2num, predictions)
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

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

        if one_results.count(1)>0:
            print(sorted_pairs)
            print(target_item)
            print(one_results)

        results.append(one_results)

    return results, code2num

def get_topk_ranking_results(predictions, targets, k, all_items=None):
    results = []
    B = len(targets)

    for b in range(B):
        batch_seqs = predictions[b]
        target_item = targets[b]
        one_results = []
        for sorted_pred in predictions:
            if sorted_pred == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results
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
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        one_hit = 0.0
        for i in range(len(res)):
            one_hit += res[i]
        hit += one_hit
    return hit

