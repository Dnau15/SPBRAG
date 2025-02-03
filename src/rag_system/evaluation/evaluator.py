import fire
import logging
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
from pathlib import Path
from typing import Optional

from rag_system.evaluation.metrics import compute_em, compute_f1
from rag_system.data.data_preprocessing import process_nq_custom, process_musique
from rag_system.models.bert_classifier import predict_class
from rag_system.data.data_loading import load_custom_dataset

# Get project root (where this script resides)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("transformers").setLevel(logging.WARNING)


# TODO Rewrite it and add script to dataset creation that downloads and preprocess dataset
def load_data(
    num_test_samples: int = 30,
    nq_path: Optional[Path] = None,
    musique_path: Optional[Path] = None,
):
    if nq_path is None:
        nq_path = Path(PROJECT_ROOT, "./data/nq_val")
    if musique_path is None:
        musique_path = Path(PROJECT_ROOT, "./data/musique")

    nq = load_custom_dataset(nq_path, num_test_samples)
    musique = load_custom_dataset(musique_path, num_test_samples)

    nq_processed = nq.map(process_nq_custom)
    nq_processed = nq_processed.filter(
        lambda example: all(value is not None for value in example.values())
    )
    nq_processed = nq_processed.remove_columns(
        [
            "id",
            "document",
            "long_answer_candidates",
            "annotations",
        ]
    )
    print(nq_processed)

    musique_processed = musique.map(process_musique)
    musique_processed = musique_processed.filter(
        lambda example: all(value is not None for value in example.values())
    )
    musique_processed = musique_processed.remove_columns(
        ["id", "paragraphs", "question_decomposition", "answer_aliases", "answerable"]
    )

    df_nq = pd.DataFrame(nq_processed).rename(columns={"answer": "answers"})
    df_musique = pd.DataFrame(musique_processed).rename(columns={"answer": "answers"})
    df = pd.concat([df_nq, df_musique], ignore_index=True)
    out_path = Path(PROJECT_ROOT, "./data/question.csv")
    print(out_path)
    print(df)
    df.drop(columns=["context"]).to_csv(out_path, index=False, sep="\t")
    df["need_retrieval"] = 1
    return df


def rag_query(
    llm,
    question,
    classificator,
    tokenizer,
    milvus_client,
    device,
    embedding_model,
    collection_name,
    use_classifier=True,
    top_k=3,
    context_len=400,
):
    query_embedding = embedding_model.encode(question).tolist()
    predicted_class = (
        predict_class(classificator, question, tokenizer, device)
        if use_classifier
        else 1
    )
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


def evaluate_rag(
    df,
    collection_name,
    bert_path,
    tokenizer_path,
    embedding_model_path,
    milvus_uri,
    logger,
):
    embedding_model = SentenceTransformer(embedding_model_path)
    milvus_client = MilvusClient(uri=milvus_uri)

    if not milvus_client.has_collection(collection_name):
        contexts = df["context"].dropna().unique().tolist()
        milvus_client.create_collection(
            collection_name=collection_name,
            dimension=768,
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
        documents = [
            {"text": ctx, "embedding": embedding_model.encode(ctx).tolist()}
            for ctx in contexts
        ]
        for i in tqdm(range(0, len(documents), 100)):
            milvus_client.insert(collection_name, documents[i : i + 100])

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        max_new_tokens=100,
        do_sample=False,
    )
    classificator = BertForSequenceClassification.from_pretrained(
        bert_path, num_labels=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classificator.to(device)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        generated_answer, predicted_class = rag_query(
            llm,
            row["question"],
            classificator,
            tokenizer,
            milvus_client,
            device,
            embedding_model,
            collection_name,
        )
        f1 = compute_f1(generated_answer, row["answers"])
        em = compute_em(generated_answer, row["answers"])
        accuracy = 1.0 if predicted_class == row["need_retrieval"] else 0.0

        results.append(
            {
                "question": row["question"],
                "context": row["context"],
                "generated_answer": generated_answer,
                "gold_answer": row["answers"],
                "f1_score": f1,
                "em_score": em,
                "accuracy": accuracy,
                "predicted_class": predicted_class,
                "need_retrieval": row["need_retrieval"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        Path(PROJECT_ROOT, "./data/rag_evaluation_results.csv"), index=False, sep="\t"
    )

    logger.info(f"F1 Score: {results_df['f1_score'].mean():.4f}")
    logger.info(f"Lenient EM Score: {results_df['em_score'].mean():.4f}")
    logger.info(f"Accuracy: {results_df['accuracy'].mean():.4f}")

    return results_df


def main(
    bert_path: str = "./models/bert-text-classification-model",
    tokenizer_path: str = "./models/bert-text-classification-model",
    embedding_model_path: str = "sentence-transformers/all-mpnet-base-v2",
    milvus_uri: str = "./data/milvus_demo.db",
    collection_name: str = "rag_eval",
    num_test_samples: int = 5,
):
    logger = logging.getLogger(__name__)
    df = load_data(num_test_samples=num_test_samples)
    results_df = evaluate_rag(
        df,
        collection_name,
        bert_path,
        tokenizer_path,
        embedding_model_path,
        milvus_uri,
        logger,
    )
    results_df.drop(columns=["context"]).to_csv(
        Path(PROJECT_ROOT, "./data/results.csv"), index=False, sep="\t"
    )


if __name__ == "__main__":
    fire.Fire(main)
