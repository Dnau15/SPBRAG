from collections import Counter
import torch
from dataclasses import dataclass


@dataclass
class Latency:
    bert_latency: float = 0
    db_latency: float = 0
    llm_latency: float = 0


def cosine_similarity(vec1, vec2):
    return torch.nn.functional.cosine_similarity(
        torch.tensor(vec1).unsqueeze(0), torch.tensor(vec2).unsqueeze(0)
    ).item()


def compute_faithfulness(generated_answer, context, embedding_model):
    return cosine_similarity(
        embedding_model.encode(generated_answer), embedding_model.encode(context)
    )


def compute_context_relevancy(question, context, embedding_model):
    return cosine_similarity(
        embedding_model.encode(question), embedding_model.encode(context)
    )


def compute_answer_relevancy(question, generated_answer, embedding_model):
    return cosine_similarity(
        embedding_model.encode(question), embedding_model.encode(generated_answer)
    )


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
