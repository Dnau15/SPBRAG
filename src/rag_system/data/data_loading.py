from pathlib import Path
import pandas as pd
from datasets import load_from_disk
from typing import Optional

from rag_system.data.dataset_creation import (
    create_mmlu_dataset,
    create_arc_dataset,
    create_musique_dataset,
    create_nq_dataset,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


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

    if not arc_path.is_dir():
        create_arc_dataset(750)
    if not mmlu_path.is_dir():
        create_mmlu_dataset(750)
    if not musique_path.is_dir():
        create_musique_dataset(750)
    if not nq_path.is_dir():
        create_nq_dataset(750)

    df_nq = load_nq_dataset(num_test_samples)
    df_musique = load_musique_dataset(num_test_samples)
    df_mmlu = load_mmlu_dataset(num_test_samples)
    df_arc = load_arc_dataset(num_test_samples)

    df = pd.concat([df_nq, df_musique, df_mmlu, df_arc], ignore_index=True)
    df = df.drop(columns=["answer"])
    out_path = Path(PROJECT_ROOT, "./data/question.csv")
    df.drop(columns=["context"]).to_csv(out_path, index=False, sep="\t")
    logger.info("Dataset loaded")
    return df


def load_arc_dataset(num_rows: int = 250):
    dataset = load_custom_dataset(Path(PROJECT_ROOT, "./data/arc_test"), num_rows)

    df_arc = pd.DataFrame(dataset).assign(dataset="arc")
    df_arc["need_retrieval"] = 0

    return df_arc


def load_mmlu_dataset(num_rows: int = 250):
    dataset = load_custom_dataset(Path(PROJECT_ROOT, "./data/mmlu_val"), num_rows)

    df_mmlu = pd.DataFrame(dataset).assign(dataset="mmlu")
    df_mmlu["need_retrieval"] = 0

    return df_mmlu


def load_nq_dataset(num_rows: int = 250):
    dataset = load_custom_dataset(Path(PROJECT_ROOT, "./data/nq_val"), num_rows)

    df_nq = pd.DataFrame(dataset).assign(dataset="nq")
    df_nq["need_retrieval"] = 1

    return df_nq


def load_musique_dataset(num_rows: int = 250):
    dataset = load_custom_dataset(Path(PROJECT_ROOT, "./data/musique"), num_rows)

    df_musique = pd.DataFrame(dataset).assign(dataset="musique")
    df_musique["need_retrieval"] = 1

    return df_musique


def load_custom_dataset(path: Path, num_samples: int) -> pd.DataFrame:
    """
    Load and process a dataset from disk, selecting a specified number of samples.

    Args:
        path (Path): Path to the dataset directory.
        num_samples (int): Number of samples to select.

    Returns:
        pd.DataFrame: Processed dataset as a DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    dataset = load_from_disk(path)
    return dataset.select(range(num_samples))
