import fire
import logging
import pandas as pd
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from rag_system.evaluation.metrics import compute_em, compute_f1
from rag_system.data.data_preprocessing import (
    process_nq_new1,
    process_musique_new1,
    process_mmlu_new1,
    process_arc_new1,
)
from rag_system.models.bert_classifier import predict_class
from rag_system.data.data_loading import load_custom_dataset
from rag_system.retrieval.rag import RAGPipeline

# Get project root (where this script resides)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


# TODO Rewrite it and add script to dataset creation that downloads and preprocess dataset
def load_test_data(
    logger,
    num_test_samples: int = 30,
    nq_path: Optional[Path] = None,
    musique_path: Optional[Path] = None,
    mmlu_path: Optional[Path] = None,
    arc_path: Optional[Path] = None,
):
    logger.info("Loading dataset")
    if nq_path is None:
        nq_path = Path(PROJECT_ROOT, "./data/nq_val")
    if musique_path is None:
        musique_path = Path(PROJECT_ROOT, "./data/musique")
    if mmlu_path is None:
        mmlu_path = Path(PROJECT_ROOT, "./data/mmlu_val")
    if arc_path is None:
        arc_path = Path(PROJECT_ROOT, "./data/arc_test")

    nq = load_custom_dataset(nq_path, num_test_samples)
    musique = load_custom_dataset(musique_path, num_test_samples)
    mmlu = load_custom_dataset(mmlu_path, num_test_samples)
    arc = load_custom_dataset(arc_path, num_test_samples)

    nq_processed = nq.map(process_nq_new1)
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

    mmlu_processed = mmlu.map(process_mmlu_new1)
    mmlu_processed = mmlu_processed.filter(
        lambda example: all(value is not None for value in example.values())
    )
    mmlu_processed = mmlu_processed.remove_columns(["choices", "subject"])

    musique_processed = musique.map(process_musique_new1)
    musique_processed = musique_processed.filter(
        lambda example: all(value is not None for value in example.values())
    )
    musique_processed = musique_processed.remove_columns(
        ["id", "paragraphs", "question_decomposition", "answer_aliases", "answerable"]
    )

    arc_processed = arc.map(process_arc_new1)
    arc_processed = arc_processed.filter(
        lambda example: all(value is not None for value in example.values())
    )
    arc_processed = arc_processed.remove_columns(["choices", "id", "answerKey"])

    df_nq = pd.DataFrame(nq_processed).assign(dataset="nq")
    df_nq["need_retrieval"] = 1

    df_musique = pd.DataFrame(musique_processed).assign(dataset="musique")
    df_musique["need_retrieval"] = 1

    df_mmlu = pd.DataFrame(mmlu_processed).assign(dataset="mmlu")
    df_mmlu["need_retrieval"] = 0

    df_arc = pd.DataFrame(arc_processed).assign(dataset="arc")
    df_arc["need_retrieval"] = 0

    df = pd.concat([df_nq, df_musique, df_mmlu, df_arc], ignore_index=True)
    df = df.drop(columns=['answer'])
    out_path = Path(PROJECT_ROOT, "./data/question.csv")
    df.drop(columns=["context"]).to_csv(out_path, index=False, sep="\t")
    logger.info("Dataset loaded")
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


def evaluate_rag(df, rag: RAGPipeline, use_classifier: bool, logger):
    contexts = df["context"].dropna().unique().tolist()
    rag.load_vector_database(contexts)

    results = []
    total_time = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        generated_answer, predicted_class, execution_time = (
            rag.query_with_classifier(
                row["question"],
            )
            if use_classifier
            else rag.query_without_classifier(
                row["question"],
            )
        )
        f1 = compute_f1(generated_answer, row["answers"])
        em = compute_em(generated_answer, row["answers"])
        accuracy = 1.0 if predicted_class == row["need_retrieval"] else 0.0
        total_time += execution_time

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
                "with_classifier_time": execution_time,
            }
        )

    results_df = pd.DataFrame(results)
    out_path = (
        "./data/rag_w_classifier_evaluation_results.csv"
        if use_classifier
        else "./data/rag_wout_evaluation_results.csv"
    )

    results_df.drop(columns=["context"]).to_csv(
        Path(PROJECT_ROOT, out_path), index=False, sep="\t"
    )

    message = (
        "RAG WITH CLASSIFIER STATISTICS:"
        if use_classifier
        else "RAG WITHOUT CLASSIFIER STATISTICS:"
    )

    logger.info(message)
    logger.info(f"F1 Score: {results_df['f1_score'].mean():.4f}")
    logger.info(f"Lenient EM Score: {results_df['em_score'].mean():.4f}")
    logger.info(f"Accuracy: {results_df['accuracy'].mean():.4f}")
    logger.info(f"Total time: {total_time}")

    # return results_df


def main(
    bert_path: str = "./models/bert-text-classification-model",
    tokenizer_path: str = "./models/bert-text-classification-model",
    embedding_model_path: str = "sentence-transformers/all-mpnet-base-v2",
    milvus_uri: str = "./data/milvus_demo.db",
    collection_name: str = "rag_eval",
    llm_repo_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    llm_max_new_tokens: int = 100,
    num_test_samples: int = 5,
    use_classifier: bool = True,
    model_type: str = "mistral",
):
    logger = logging.getLogger(__name__)

    df = load_test_data(logger=logger, num_test_samples=num_test_samples)
    rag = RAGPipeline(
        collection_name=collection_name,
        bert_path=bert_path,
        tokenizer_path=tokenizer_path,
        embedding_model_path=embedding_model_path,
        milvus_uri=milvus_uri,
        llm_repo_id=llm_repo_id,
        llm_max_new_tokens=llm_max_new_tokens,
        model_type=model_type,
    )

    evaluate_rag(
        df,
        rag,
        use_classifier,
        logger,
    )


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("pymilvus").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    fire.Fire(main)
