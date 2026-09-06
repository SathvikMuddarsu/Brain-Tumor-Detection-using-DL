import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from utils.config import CSCANConfig

def compute_classification_metrics(y_true, y_pred):
    """
    Computes classification accuracy, macro precision, recall, and F1-score.
    
    Why it is used:
    Macro averaging gives equal weight to all tumor classes regardless of count,
    giving a fair evaluation of minority classes (e.g., Pituitary or No Tumor).
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def save_classification_report(y_true, y_pred, save_path):
    """
    Generates and saves a detailed text classification report (per-class metrics).
    Includes precision, recall, F1-score, and support for each tumor class.
    """
    report = classification_report(y_true, y_pred, target_names=CSCANConfig.CLASSES, zero_division=0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("             CSCAN BRAIN TUMOR CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n")
    print(f"[Info] Detailed classification report saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path, title="Confusion Matrix"):
    """
    Generates and saves a high-quality confusion matrix heatmap.
    
    Why it is used:
    In brain tumor MRI analysis, visual similarities between Gliomas and Meningiomas
    frequently confuse classifiers. The confusion matrix allows researchers to verify
    if CSCAN successfully minimizes cross-class misclassification.
    """
    cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(CSCANConfig.NUM_CLASSES))
)
    plt.figure(figsize=(8, 6))
    
    # Render heatmap with professional blue-green palette
    sns.heatmap(cm, annot=True, fmt='d', cmap='GnBu', 
                xticklabels=CSCANConfig.CLASSES, 
                yticklabels=CSCANConfig.CLASSES,
                cbar=True, square=True)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    plt.xticks(rotation=15)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Info] Confusion matrix plot saved to: {save_path}")


def plot_training_curves(train_history, val_history, metric_name, save_path):
    """
    Plots training vs validation metric curves (Loss or Accuracy) over epochs.
    
    Why it is used:
    Visualizes learning rates, model convergence, and checks for overfitting 
    (where validation loss rises while training loss drops).
    """
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_history) + 1)
    
    plt.plot(epochs, train_history, 'o-', color='#0288d1', label=f'Train {metric_name}', linewidth=2)
    plt.plot(epochs, val_history, 's--', color='#388e3c', label=f'Validation {metric_name}', linewidth=2)
    
    plt.title(f'CSCAN Training vs Validation {metric_name}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Info] Training curves saved to: {save_path}")


def save_metrics_to_csv(metrics_dict, save_path):
    """
    Saves final training/testing results to a CSV file.
    Provides structured data for quick research paper tables.
    """
    df = pd.DataFrame([metrics_dict])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[Info] Final metrics saved to CSV at: {save_path}")
