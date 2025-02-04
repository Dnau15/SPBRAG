import fire
import logging
import pandas as pd
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

from rag_system.evaluation.metrics import compute_em, compute_f1
from rag_system.models.bert_classifier import predict_class
from rag_system.retrieval.rag import RAGPipeline
from rag_system.data.data_loading import load_test_data

# Get project root (where this script resides)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


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
    rag: RAGPipeline,
    use_classifier: bool,
    logger,
    top_k: int = 30,
    context_len: int = 1000,
):
    contexts = df["context"].dropna().unique().tolist()
    rag.load_vector_database(contexts)

    results = []
    total_time = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        generated_answer, predicted_class, execution_time = (
            rag.query_with_classifier(
                row["question"],
                top_k=top_k,
                context_len=context_len,
            )
            if use_classifier
            else rag.query_without_classifier(
                row["question"],
                top_k=top_k,
                context_len=context_len,
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

    csv_name = (
        "./data/latency_w_classifier"
        if use_classifier
        else "./data/latency_wout_classifier"
    )
    csv_name = Path(PROJECT_ROOT, csv_name)
    latency_df = pd.DataFrame(rag.latencies)
    latency_df["top_k"] = top_k
    latency_df["context_len"] = context_len
    latency_df.to_csv(csv_name, index=False)

    # return results_df


def main(
    bert_path: str = "google-bert/bert-base-uncased",
    tokenizer_path: str = "google-bert/bert-base-uncased",
    embedding_model_path: str = "sentence-transformers/all-mpnet-base-v2",
    milvus_uri: str = "./data/milvus_demo.db",
    collection_name: str = "rag_eval",
    llm_repo_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    llm_max_new_tokens: int = 100,
    num_test_samples: int = 5,
    use_classifier: bool = True,
    model_type: str = "mistral",
    top_k: int = 30,
    context_len: int = 1000,
):
    logger = logging.getLogger(__name__)

    df = load_test_data(logger=logger, num_test_samples=num_test_samples)

    rag = RAGPipeline(
        logger=logger,
        collection_name=collection_name,
        bert_path=bert_path,
        tokenizer_path=tokenizer_path,
        embedding_model_path=embedding_model_path,
        milvus_uri=milvus_uri,
        model_type=model_type,
        llm_repo_id=llm_repo_id,
        llm_max_new_tokens=llm_max_new_tokens,
    )

    evaluate_rag(
        df,
        rag,
        use_classifier,
        logger,
        top_k,
        context_len=context_len,
    )


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("pymilvus").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.WARNING)

    fire.Fire(main)
