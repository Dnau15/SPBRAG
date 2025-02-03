import pandas as pd
from datasets import load_dataset, Dataset
import fire
import os
from pathlib import Path

from rag_system.data.data_preprocessing import (
    process_mmlu_new1,
    process_arc_new1,
    process_musique_new1,
    process_nq_new1,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def create_mmlu_dataset(num_rows: int = 250):
    dataset_mmlu = load_dataset("cais/mmlu", "all", split="validation", streaming=True)

    samples_mmlu = list(dataset_mmlu.take(num_rows))
    samples_mmlu = Dataset.from_list(samples_mmlu)
    mmlu_processed = samples_mmlu.map(process_mmlu_new1)
    mmlu_processed = mmlu_processed.filter(
        lambda example: all(value is not None for value in example.values())
    )

    mmlu_processed = mmlu_processed.remove_columns(["choices", "subject"])
    mmlu_processed.save_to_disk(os.path.join(PROJECT_ROOT, "./data/mmlu_val"))


def create_arc_dataset(num_rows: int = 250):
    dataset_arc = load_dataset(
        "allenai/ai2_arc", "ARC-Challenge", split="test", streaming=True
    )

    samples_arc = list(dataset_arc.take(num_rows))
    samples_arc = Dataset.from_list(samples_arc)

    arc_processed = samples_arc.map(process_arc_new1)
    arc_processed = arc_processed.filter(
        lambda example: all(value is not None for value in example.values())
    )
    arc_processed = arc_processed.remove_columns(["choices", "id", "answerKey"])
    arc_processed.save_to_disk(os.path.join(PROJECT_ROOT, "./data/arc_test"))


def create_nq_dataset(num_rows: int = 250):
    dataset_nq = load_dataset(
        "google-research-datasets/natural_questions", split="train", streaming=True
    )

    samples_nq = list(dataset_nq.take(num_rows))
    samples_nq = Dataset.from_list(samples_nq)

    nq_processed = samples_nq.map(process_nq_new1)
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
    nq_processed.save_to_disk(os.path.join(PROJECT_ROOT, "./data/nq_val"))


def create_musique_dataset(num_rows: int = 250):
    dataset_musique = load_dataset(
        "dgslibisey/MuSiQue", split="train", streaming=True
    )

    samples_musique = list(dataset_musique.take(num_rows))
    samples_musique = Dataset.from_list(samples_musique)

    musique_processed = samples_musique.map(process_musique_new1)
    musique_processed = musique_processed.filter(
        lambda example: all(value is not None for value in example.values())
    )
    musique_processed = musique_processed.remove_columns(
        ["id", "paragraphs", "question_decomposition", "answer_aliases", "answerable"]
    )
    musique_processed.save_to_disk(os.path.join(PROJECT_ROOT, "./data/musique"))


def load_and_process(path: str) -> pd.DataFrame:
    return load_dataset(path)["train"].to_pandas()


def prepare_datasets(
    merge_rewrite_size: int = 200,
    squad_train_size: int = 1600,
    squad_test_size: int = 400,
    random_state: int = 42,
    save_dir: str = "../../../data",
):
    merge_rewrite = load_and_process("positivethoughts/merge_rewrite_13.3k")
    databricks = load_and_process("databricks/databricks-dolly-15k")

    merge_rewrite = merge_rewrite.drop(columns=["id"])
    merge_rewrite["prompt"] = (
        merge_rewrite["rewrite_prompt"] + " " + merge_rewrite["original_text"]
    )
    merge_rewrite = merge_rewrite.drop(
        columns=["rewrite_prompt", "rewritten_text", "original_text"]
    )
    merge_rewrite["need_retrieval"] = 0

    databricks = databricks.drop_duplicates()
    databricks["prompt"] = databricks["instruction"] + " " + databricks["context"]
    databricks = databricks.drop(columns=["instruction", "context", "response"])
    categories_requiring_retrieval = [
        "open_qa",
        "brainstorming",
        "general_qa",
        "creative_writing",
    ]
    databricks["need_retrieval"] = (
        databricks["category"].isin(categories_requiring_retrieval).astype(int)
    )

    merged = pd.concat([databricks, merge_rewrite], axis=0)

    sample_no_retrieval = (
        merged.loc[merged["need_retrieval"] == 0]
        .sample(n=merge_rewrite_size, random_state=random_state)
        .rename(columns={"prompt": "question"})
    )
    merged = merged.drop(sample_no_retrieval.index)

    squad = load_dataset("squad_v2", split="train").to_pandas()
    titles = [
        "Beyoncé",
        "Frédéric_Chopin",
        "The_Legend_of_Zelda:_Twilight_Princess",
        "Sino-Tibetan_relations_during_the_Ming_dynasty",
        "IPod",
        "Spectre_(2015_film)",
        "2008_Sichuan_earthquake",
        "New_York_City",
        "To_Kill_a_Mockingbird",
        "Solar_energy",
        "Kanye_West",
        "Buddhism",
        "American_Idol",
        "Dog",
    ]

    filtered_squad = squad.loc[
        squad["answers"].apply(
            lambda x: isinstance(x, dict) and "text" in x and len(x["text"]) > 0
        )
    ].reset_index(drop=True)

    small_squad = filtered_squad.loc[filtered_squad["title"].isin(titles)].drop(
        columns=["id"]
    )
    small_squad["need_retrieval"] = 1

    small_squad_train = (
        small_squad.sample(squad_train_size, random_state=random_state)
        .rename(columns={"question": "prompt"})
        .assign(answers=lambda x: x["answers"].apply(lambda y: y["text"][0]))
    )

    small_squad_test = (
        small_squad.drop(small_squad_train.index)
        .sample(squad_test_size, random_state=random_state)
        .assign(answers=lambda x: x["answers"].apply(lambda y: y["text"][0]))
    )

    merged = pd.concat([merged, small_squad_train], axis=0)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    merged.to_csv(os.path.join(save_dir, "merged.csv"), index=False)

    demonstration = pd.concat([small_squad_test, sample_no_retrieval], axis=0)
    demonstration = demonstration.sample(frac=1, random_state=random_state)
    demonstration.to_csv(os.path.join(save_dir, "test.csv"), index=False)


def main(
    merge_rewrite_size: int = 200,
    squad_train_size: int = 1600,
    squad_test_size: int = 400,
    random_state: int = 42,
    save_dir: str = "../../../data",
):
    prepare_datasets(
        merge_rewrite_size, squad_train_size, squad_test_size, random_state, save_dir
    )


if __name__ == "__main__":
    fire.Fire(main)
