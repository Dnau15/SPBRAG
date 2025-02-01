import fire
import logging
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from tqdm import trange
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def predict_class(model, text, tokenizer, device, max_length=512):
    model.eval()
    inputs = tokenizer(
        text, return_tensors="pt", max_length=max_length, truncation=True, padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.argmax(logits, dim=-1).item()


def rag_query(
    llm,
    question,
    classificator,
    tokenizer,
    milvus_client,
    device,
    embedding_model,
    collection_name,
    top_k=3,
    context_len=400,
):
    query_embedding = embedding_model.encode(question).tolist()

    predicted_class = predict_class(classificator, question, tokenizer, device)

    search_results = milvus_client.search(
        collection_name, [query_embedding], limit=top_k, output_fields=["text"]
    )

    contexts = (
        [hit["entity"]["text"] for hit in search_results[0]] if search_results else []
    )

    context_str = " ".join(contexts)[:context_len]

    template = (
        "Answer the question based on context:\nContext: {context}\nQuestion: {question}\nAnswer:"
        if contexts and predicted_class
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


def start_vector_database(collection_name):
    embedding_dim = 768
    milvus_client = MilvusClient(uri="../data/milvus_demo.db")

    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

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
    return milvus_client


def insert_documents(contexts, milvus_client, embedding_model, collection_name):
    documents = [
        {"text": ctx, "embedding": embedding_model.encode(ctx).tolist()}
        for ctx in contexts
    ]
    for i in trange(0, len(documents), 100):
        milvus_client.insert(collection_name, documents[i : i + 100])


def test_rag_system(num_samples=10):
    load_dotenv()
    collection_name = "spbrag"
    milvus_client = start_vector_database(collection_name)

    dataset = pd.read_csv("../data/test.csv")

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

    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    insert_documents(contexts, milvus_client, embedding_model, collection_name)

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        max_new_tokens=100,
        do_sample=False,
    )

    model_path = "./bert-text-classification-model"
    num_labels = 2
    classificator = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classificator.to(device)
    tokenizer = BertTokenizer.from_pretrained("./bert-text-classification-model")

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
