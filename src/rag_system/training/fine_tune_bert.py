import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    AdamW,
    get_scheduler,
)
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import logging
import fire

from rag_system.data.datasets import TextClassificationDataset
from rag_system.visualization.plot_graphs import save_training_plots
from rag_system.data.data_preprocessing import tokenize_data


def train_epoch(model, train_loader, optimizer, lr_scheduler, device):
    model.train()
    train_loss = 0
    train_preds, train_labels = [], []

    for batch in tqdm(train_loader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(batch["labels"].cpu().numpy())

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds)

    return avg_train_loss, train_accuracy, train_f1


def validate_epoch(model, val_loader, device):
    model.eval()
    val_loss = 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}
            outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(batch["labels"].cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds)

    return avg_val_loss, val_accuracy, val_f1


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler,
    device,
    logger,
    num_epochs: int,
    save_dir: str = "../../../bert-text-classification-model",
    metric: str = "f1",
    early_stopping_patience=None,
):
    best_val_metric = 0
    epochs_without_improvement = 0

    metrics = {
        "train_losses": [],
        "val_losses": [],
        "train_accuracies": [],
        "val_accuracies": [],
        "train_f1_scores": [],
        "val_f1_scores": [],
    }

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        avg_train_loss, train_accuracy, train_f1 = train_epoch(
            model, train_loader, optimizer, lr_scheduler, device
        )
        metrics["train_losses"].append(avg_train_loss)
        metrics["train_accuracies"].append(train_accuracy)
        metrics["train_f1_scores"].append(train_f1)
        logger.info(
            f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Training F1: {train_f1:.4f}"
        )

        avg_val_loss, val_accuracy, val_f1 = validate_epoch(model, val_loader, device)
        metrics["val_losses"].append(avg_val_loss)
        metrics["val_accuracies"].append(val_accuracy)
        metrics["val_f1_scores"].append(val_f1)
        logger.info(
            f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}"
        )

        current_val_metric = val_f1 if metric == "f1" else val_accuracy

        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            model.save_pretrained(save_dir)
            logger.info(
                f"New best model saved with Validation {metric.capitalize()}: {best_val_metric:.4f}"
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience
            and epochs_without_improvement >= early_stopping_patience
        ):
            logger.info(f"Early stopping at epoch {epoch + 1} due to no improvement.")
            break

    logger.info(
        f"Loaded the best model with Validation {metric.capitalize()}: {best_val_metric:.4f}"
    )

    return metrics


def evaluate(
    model,
    test_loader,
    test_df,
    device,
    logger,
):
    model.eval()
    test_results = []

    with torch.no_grad():
        for batch in test_loader:
            idxs = batch["idx"].cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}

            outputs = model(**batch)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            actual_labels = batch["labels"].cpu().numpy().tolist()

            texts = test_df.loc[idxs, "prompt"].tolist()
            categories = test_df.loc[idxs, "category"].tolist()

            test_results.extend(zip(texts, categories, actual_labels, preds))

    results_df = pd.DataFrame(
        test_results,
        columns=["Text", "Category", "Need_retrieval", "Predicted"],
    )

    logger.info("Test set evaluation completed.")
    return results_df


def preprocess(path: str, num_samples_per_class: int):
    df = pd.read_csv(path)

    balanced_df = (
        df.groupby("need_retrieval", group_keys=False)
        .apply(
            lambda x: x.sample(num_samples_per_class, random_state=42),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df


def get_dataset(df, tokenizer):
    df_encodings = tokenize_data(df, tokenizer)

    df_labels = df["need_retrieval"].tolist()

    # saving the indices of specific samples in the dataset so that I can easily retrieve them later when needed
    df_indices = df.index.tolist()

    dataset = TextClassificationDataset(df_encodings, df_labels, df_indices)
    return dataset


def main(
    file_path="../../../data/merged.csv",
    num_samples_per_class=1500,
    tokenizer_path="bert-base-uncased",
    model_path="bert-base-uncased",
    batch_size=4,
    learning_rate=5e-5,
    num_epochs=3,
    save_dir="../../../bert-text-classification-model",
    metric="f1",
    early_stopping_patience=3,
    output_csv="../../../data/predictions.csv",
):
    """
    Train and evaluate a BERT-based text classification model.

    Args:
        file_path (str): Path to the input CSV file.
        num_samples_per_class (int): Number of samples per class for balancing.
        tokenizer_path (str): Pretrained tokenizer path.
        model_path (str): Pretrained model path.
        batch_size (int): Batch size for training and evaluation.
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of training epochs.
        save_dir (str): Directory to save the trained model.
        metric (str): Evaluation metric (e.g., 'f1', 'accuracy').
        early_stopping_patience (int): Patience for early stopping.
        output_csv (str): Path to save the predictions.
    """
    balanced_df = preprocess(file_path, num_samples_per_class=num_samples_per_class)
    train_df, temp_df = train_test_split(balanced_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    train_dataset = get_dataset(train_df, tokenizer)
    val_dataset = get_dataset(val_df, tokenizer)
    test_dataset = get_dataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    metrics = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        num_epochs=num_epochs,
        save_dir=save_dir,
        metric=metric,
        early_stopping_patience=early_stopping_patience,
        logger=logger,
    )

    train_losses = metrics["train_losses"]
    val_losses = metrics["val_losses"]
    train_f1_scores = metrics["train_f1_scores"]
    val_f1_scores = metrics["val_f1_scores"]
    train_accuracies = metrics["train_accuracies"]
    val_accuracies = metrics["val_accuracies"]

    save_training_plots(
        logger,
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        train_f1_scores,
        val_f1_scores,
    )

    results_df = evaluate(
        model=model,
        test_loader=test_loader,
        test_df=test_df,
        device=device,
        logger=logger,
    )

    results_df.to_csv(output_csv)


if __name__ == "__main__":
    fire.Fire(main)
