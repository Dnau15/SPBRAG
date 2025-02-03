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


def process_musique(example):
    return {
        "question": example["question"],
        "answer": example["answer"],
        "context": " ".join(
            [p["paragraph_text"] for p in example["paragraphs"] if p["is_supporting"]]
        ),
    }


def process_nq_custom(example):
    question = example.get("question", {}).get("text")

    # Safe check for the answer field
    answer = None
    if (
        example.get("annotations")
        and example["annotations"].get("short_answers")
        and len(example["annotations"]["short_answers"]) > 0
        and example["annotations"]["short_answers"][0].get("text")
    ):
        answer = example["annotations"]["short_answers"][0]["text"][0]

    tokens = example["document"]["tokens"]
    context_tokens = [t for t, h in zip(tokens["token"], tokens["is_html"]) if not h]
    context = " ".join(context_tokens)

    # If any field is None, return None to skip the example
    if question is None or answer is None or context is None:
        print(question)
        return {"question": None, "answer": None, "context": None}

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
