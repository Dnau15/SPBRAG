from pathlib import Path
import pandas as pd
from datasets import load_from_disk
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def load_arc_dataset(num_rows: int = 250):
    dataset = load_custom_dataset(
        Path(PROJECT_ROOT, "./data/arc_test"), num_rows
    )

    df_arc = pd.DataFrame(dataset).assign(dataset="arc")
    df_arc["need_retrieval"] = 0

    return df_arc


def load_mmlu_dataset(num_rows: int = 250):
    dataset = load_custom_dataset(
        Path(PROJECT_ROOT, "./data/mmlu_val"), num_rows
    )

    df_mmlu = pd.DataFrame(dataset).assign(dataset="mmlu")
    df_mmlu["need_retrieval"] = 0

    return df_mmlu


def load_nq_dataset(num_rows: int = 250):
    dataset = load_custom_dataset(
        Path(PROJECT_ROOT, "./data/nq_val"), num_rows
    )

    df_nq = pd.DataFrame(dataset).assign(dataset="nq")
    df_nq["need_retrieval"] = 1

    return df_nq


def load_musique_dataset(num_rows: int = 250):
    dataset = load_custom_dataset(
        Path(PROJECT_ROOT, "./data/musique"), num_rows
    )

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
