from collections import Counter


def compute_f1(pred, gold):
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0.0
    )


def compute_em(pred, gold):
    return 1.0 if gold.strip().lower() in pred.strip().lower() else 0.0
