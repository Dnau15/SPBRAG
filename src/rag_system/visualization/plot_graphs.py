import os
from typing import List
import matplotlib.pyplot as plt


def save_training_plots(
    logger,
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    train_f1_scores: List[float],
    val_f1_scores: List[float],
    save_dir: str = "../../../data/plots",
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(train_f1_scores, label="Training F1 Score")
    plt.plot(val_f1_scores, label="Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Training and Validation F1 Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_metrics.png"))
    plt.close()

    logger.info(f"Plots saved in {save_dir}/training_metrics.png")
