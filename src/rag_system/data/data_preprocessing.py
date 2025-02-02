from tqdm import tqdm
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from typing import List


def tokenize_data(df, tokenizer, max_length=512):
    return tokenizer(
        df["prompt"].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def process_nq(example):
    question = example["question"]["text"]
    answer = ""
    if (
        example["annotations"]
        and example["annotations"]["short_answers"]
        and example["annotations"]["short_answers"][0]["text"]
    ):
        answer = example["annotations"]["short_answers"][0]["text"][0]
    tokens = example["document"]["tokens"]
    context_tokens = [
        t for t, h in zip(tokens["token"], tokens["is_html"]) if not h
    ]  # Super comment
    context = " ".join(context_tokens)
    return {"question": question, "answer": answer, "context": context}


def process_trivia(example):
    question = example["question"]
    answer = example["answer"]["normalized_value"]
    context = (
        example["entity_pages"][0]["wiki_content"] if example["entity_pages"] else ""
    )
    return {"question": question, "answer": answer, "context": context}


def insert_documents(
    contexts: List[str],
    milvus_client: MilvusClient,
    embedding_model: SentenceTransformer,
    collection_name: str,
) -> None:
    documents = [
        {"text": ctx, "embedding": embedding_model.encode(ctx).tolist()}
        for ctx in contexts
    ]
    for i in tqdm(range(0, len(documents), 100)):
        milvus_client.insert(collection_name, documents[i : i + 100])
