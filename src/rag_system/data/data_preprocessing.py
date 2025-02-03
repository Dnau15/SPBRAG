from tqdm import tqdm
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm


def split_text_into_chunks(
    text: str, chunk_size: int = 512, chunk_overlap: int = 20
) -> list:
    """Splits text into sentence-level chunks of the given token size."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", ". ", "? ", "! "],  # Sentence-level chunking
        length_function=len,  # Can use tokenizer-based length calculation if needed
    )
    return text_splitter.split_text(text)


def preprocess_documents(documents: list[str], chunk_size: int = 512) -> list[Document]:
    """Splits all documents into smaller sentence-level chunks."""
    chunked_docs = []
    for doc in tqdm(documents, desc="Preprocessing documents"):
        chunks = split_text_into_chunks(doc, chunk_size=chunk_size)
        chunked_docs.extend([Document(page_content=chunk) for chunk in chunks])
    return chunked_docs


def tokenize_data(df, tokenizer, max_length=512):
    return tokenizer(
        df["prompt"].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def process_mmlu_new1(example):
    return {
        "question": example["question"] + "\nChoices:" + ", ".join(example["choices"]),
        "answers": example["choices"][example["answer"]],
        "context": "",
    }


def process_arc_new1(example):
    labels = example["choices"]["label"]
    idx = labels.index(example["answerKey"])
    question = (
        example["question"] + "\nChoices:" + ", ".join(example["choices"]["text"])
    )
    answer = example["choices"]["text"][idx]
    return {
        "question": question,
        "answers": answer,
        "context": "",
    }


def process_musique_new1(example):
    return {
        "question": example["question"],
        "answers": example["answer"],
        "context": " ".join(
            [p["paragraph_text"] for p in example["paragraphs"] if p["is_supporting"]]
        ),
    }


def process_nq_new1(example):
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
        return {"question": None, "answers": None, "context": None}

    return {"question": question, "answers": answer, "context": context}


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
        for ctx in tqdm(contexts, desc="Creating embeddings")
    ]
    for i in tqdm(range(0, len(documents), 100), desc="Inserting documents"):
        milvus_client.insert(collection_name, documents[i : i + 100])
