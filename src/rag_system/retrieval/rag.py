import fire
import logging
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import pandas as pd
from typing import List

from src.rag_system.models.bert_classifier import predict_class
from src.rag_system.data.data_preprocessing import insert_documents

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("pymilvus").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def rag_query(
    llm: HuggingFaceEndpoint,
    question: str,
    classificator: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    milvus_client: MilvusClient,
    device: torch.device,
    embedding_model: SentenceTransformer,
    collection_name: str,
    use_classifier: bool = True,
    top_k: int = 3,
    context_len: int = 400,
) -> tuple[str, int]:
    query_embedding = embedding_model.encode(question).tolist()

    if use_classifier:
        predicted_class = predict_class(classificator, question, tokenizer, device)
    else:
        predicted_class = 1

    search_results = milvus_client.search(
        collection_name, [query_embedding], limit=top_k, output_fields=["text"]
    )
    contexts = (
        [hit["entity"]["text"] for hit in search_results[0]] if search_results else []
    )
    context_str = " ".join(contexts)[:context_len]

    template = (
        "Answer the question based on context:\nContext: {context}\nQuestion: {question}\nAnswer:"
        if contexts and (use_classifier and predicted_class)
        else "Answer this question:\nQuestion: {question}\nAnswer:"
    )

    prompt_template = PromptTemplate.from_template(template)
    llm_chain = prompt_template | llm
    chain_input = (
        {"question": question, "context": context_str}
        if contexts
        else {"question": question}
    )
    return llm_chain.invoke(chain_input), predicted_class


def load_vector_database(
    collection_name: str,
    embedding_model: SentenceTransformer,
    contexts: List[str],
    milvus_uri: str = "./data/milvus_demo.db",
) -> MilvusClient:
    embedding_dim = 768
    milvus_client = MilvusClient(uri=milvus_uri)

    if milvus_client.has_collection(collection_name):
        logging.info("Collection already exists, loading collection")
        milvus_client.load_collection(collection_name=collection_name)
    else:
        logging.info("Creating collection")

        milvus_client.create_collection(
            collection_name=collection_name,
            dimension=embedding_dim,
            metric_type="L2",
            auto_id=True,
            primary_field_name="id",
            vector_field_name="embedding",
            enable_dynamic_field=True,
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128},
            },
        )

        logging.info("Inserting documents")
        insert_documents(contexts, milvus_client, embedding_model, collection_name)
        logging.info("Documents inserted")
    return milvus_client


def test_rag_system(
    num_samples: int = 10,
    collection_name: str = "spbrag",
    bert_path: str = "./models/bert-text-classification-model",
    tokenizer_path: str = "./models/bert-text-classification-model",
    embedding_model_path: str = "sentence-transformers/all-mpnet-base-v2",
    milvus_uri: str = "./data/milvus_demo.db",
):
    load_dotenv()

    dataset = pd.read_csv("./data/test.csv")

    contexts = dataset["context"].dropna().unique().tolist()

    qa_pairs = dataset.apply(
        lambda row: {
            "question": row["question"],
            "answer": row["answers"],
            "context": row["context"],
            "need_retrieval": row["need_retrieval"],
        },
        axis=1,
    ).tolist()

    embedding_model = SentenceTransformer(embedding_model_path)

    milvus_client = load_vector_database(
        collection_name, embedding_model, contexts, milvus_uri
    )

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        max_new_tokens=100,
        do_sample=False,
    )

    num_labels = 2
    classificator = BertForSequenceClassification.from_pretrained(
        bert_path, num_labels=num_labels
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classificator.to(device)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    logging.info("\nTesting RAG system...\n")
    logging.info("-" * 160)

    for qa in qa_pairs[:num_samples]:
        query = qa["question"]
        result, predicted_class = rag_query(
            llm,
            query,
            classificator,
            tokenizer,
            milvus_client,
            device,
            embedding_model,
            collection_name,
        )
        generated_answer = result
        true_answer = qa["answer"]
        need_retrieval = qa["need_retrieval"]

        logging.info(
            f"Question: {query[:45]} | Predicted class: {predicted_class} | True class: {need_retrieval}"
        )
        logging.info(f"Generated answer: {generated_answer[:100]}")
        logging.info(f"True answer: {true_answer}")


if __name__ == "__main__":
    fire.Fire(test_rag_system)
