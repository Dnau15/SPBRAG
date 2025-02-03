import logging
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import pandas as pd
from typing import List, Optional
from pathlib import Path
import os
import time

from src.rag_system.models.bert_classifier import predict_class
from src.rag_system.data.data_preprocessing import insert_documents

# Get project root (where this script resides)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("pymilvus").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


class RAGPipeline:
    def __init__(
        self,
        collection_name: str = "spbrag",
        bert_path: Optional[str] = None,
        tokenizer_path: str = "bert-base-uncased",
        embedding_model_path: str = "sentence-transformers/all-mpnet-base-v2",
        milvus_uri: Optional[str] = None,
        llm_repo_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        llm_max_new_tokens: int = 100,
        llm_do_sample: bool = False,
    ):
        bert_path = bert_path or os.path.join(
            PROJECT_ROOT, "./models/bert-text-classification-model"
        )
        milvus_uri = milvus_uri or os.path.join(PROJECT_ROOT, "./data/milvus_demo.db")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self.classificator = BertForSequenceClassification.from_pretrained(
            bert_path, num_labels=2
        )
        self.classificator.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.llm = HuggingFaceEndpoint(
            repo_id=llm_repo_id,
            task="text-generation",
            max_new_tokens=llm_max_new_tokens,
            do_sample=llm_do_sample,
        )
        self.collection_name = collection_name
        self.milvus_uri = milvus_uri
        self.milvus_client = None

    def load_vector_database(self, contexts: List[str]) -> None:
        self.milvus_client = MilvusClient(uri=self.milvus_uri)
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.load_collection(collection_name=self.collection_name)
        else:
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
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
            insert_documents(
                contexts, self.milvus_client, self.embedding_model, self.collection_name
            )

    def query_with_classifier(
        self,
        question: str,
        top_k: int = 3,
        context_len: int = 400,
    ) -> tuple[str, int, float]:
        # TODO add time

        start_time = time.time()
        predicted_class = predict_class(
            self.classificator, question, self.tokenizer, self.device
        )

        contexts = []
        if predict_class:
            query_embedding = self.embedding_model.encode(question).tolist()
            search_results = self.milvus_client.search(
                self.collection_name,
                [query_embedding],
                limit=top_k,
                output_fields=["text"],
            )

            contexts = (
                [hit["entity"]["text"] for hit in search_results[0]]
                if search_results
                else []
            )

            context_str = " ".join(contexts)[:context_len]

        template = (
            "Answer the question based on context:\nContext: {context}\nQuestion: {question}\nAnswer:"
            if contexts
            else "Answer this question:\nQuestion: {question}\nAnswer:"
        )

        llm_chain = PromptTemplate.from_template(template) | self.llm

        result = llm_chain.invoke(
            {"question": question, "context": context_str}
            if contexts
            else {"question": question}
        )
        execution_time = round(time.time() - start_time, 2)
        return result, predicted_class, execution_time

    def query_without_classifier(
        self, question: str, top_k: int = 3, context_len: int = 400
    ) -> tuple[str, int, float]:
        start_time = time.time()

        query_embedding = self.embedding_model.encode(question).tolist()
        predicted_class = predict_class(
            self.classificator, question, self.tokenizer, self.device
        )

        search_results = self.milvus_client.search(
            self.collection_name, [query_embedding], limit=top_k, output_fields=["text"]
        )
        contexts = (
            [hit["entity"]["text"] for hit in search_results[0]]
            if search_results
            else []
        )
        context_str = " ".join(contexts)[:context_len]
        template = (
            "Answer the question based on context:\nContext: {context}\nQuestion: {question}\nAnswer:"
            if contexts
            else "Answer this question:\nQuestion: {question}\nAnswer:"
        )
        llm_chain = PromptTemplate.from_template(template) | self.llm

        result = llm_chain.invoke(
            {"question": question, "context": context_str}
            if contexts
            else {"question": question}
        )
        execution_time = round(time.time() - start_time, 2)
        return result, predicted_class, execution_time

    def test(
        self,
        dataset_path: str = "./data/test.csv",
        num_samples: int = 10,
    ) -> None:
        load_dotenv()
        dataset = pd.read_csv(dataset_path)
        contexts = dataset["context"].dropna().unique().tolist()
        self.load_vector_database(contexts)
        qa_pairs = dataset.apply(
            lambda row: {
                "question": row["question"],
                "answer": row["answers"],
                "need_retrieval": row["need_retrieval"],
            },
            axis=1,
        ).tolist()
        for qa in qa_pairs[:num_samples]:
            result, predicted_class = self.query(qa["question"])
            logging.info(
                f"Question: {qa['question'][:45]} | Predicted: {predicted_class} | True: {qa['need_retrieval']}"
            )
            logging.info(f"Generated: {result[:100]}")
            logging.info(f"True answer: {qa['answer']}")
