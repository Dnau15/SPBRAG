import torch
from transformers import BertForSequenceClassification, BertTokenizer


def predict_class(
    model: BertForSequenceClassification,
    text: str,
    tokenizer: BertTokenizer,
    device: torch.device,
    max_length: int = 512,
) -> int:
    model.eval()
    inputs = tokenizer(
        text, return_tensors="pt", max_length=max_length, truncation=True, padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.argmax(logits, dim=-1).item()
