"""Evaluation metrics and a small evaluation CLI.

``compute_metrics`` returns the same set of numbers the original comparison labs
reported (accuracy, precision, recall, F1, ROC-AUC, confusion matrix, full
classification report). Plot helpers write portfolio figures under
``results/figures/``.

CLI::

    python -m src.evaluate --framework keras   --model cnn --weights checkpoints/keras_cnn.keras
    python -m src.evaluate --framework pytorch --model vit --weights checkpoints/pytorch_vit.pth \
        --backbone-weights checkpoints/pytorch_cnn.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from . import config
from .seeding import seed_everything

FIGURE_DIR = config.REPO_ROOT / "results" / "figures"
METRIC_DIR = config.REPO_ROOT / "results" / "metrics"

_SCALAR_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float] | None = None,
    class_labels: Sequence[str] = config.CLASS_LABELS,
) -> dict:
    """Compute the standard binary-classification metric bundle.

    ``y_prob`` is the probability of the positive class; pass ``None`` to skip
    ROC-AUC.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=list(class_labels), digits=4, zero_division=0
        ),
        "class_labels": list(class_labels),
        "support": int(y_true.size),
    }
    if y_prob is not None:
        prob = np.asarray(y_prob).ravel()
        if prob.size == y_true.size and len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
    return metrics


def print_metrics(metrics: dict, model_name: str = "model") -> None:
    """Pretty-print a metric bundle from :func:`compute_metrics`."""
    print(f"\nEvaluation metrics - {model_name}")
    print("-" * 40)
    for key in _SCALAR_KEYS:
        value = metrics.get(key)
        print(f"{key:>10}: {value:.4f}" if isinstance(value, float) else f"{key:>10}: n/a")
    print("\nConfusion matrix (rows = true, cols = predicted):")
    for label, row in zip(metrics["class_labels"], metrics["confusion_matrix"]):
        print(f"  {label:>10} {row}")
    print("\n" + metrics["classification_report"])


def save_metrics(metrics: dict, path: Path | str) -> Path:
    """Write a metric bundle to JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(metrics: dict, model_name: str, out_dir: Path = FIGURE_DIR) -> Path:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    out_dir.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array(metrics["confusion_matrix"]),
        display_labels=metrics["class_labels"],
    )
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion matrix - {model_name}")
    plt.tight_layout()
    path = out_dir / f"confusion_matrix_{_slug(model_name)}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_training_history(history: dict, model_name: str, out_dir: Path = FIGURE_DIR) -> Path:
    """Plot train/val accuracy and loss curves from a history dict."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4))
    if "accuracy" in history:
        ax_acc.plot(history["accuracy"], label="train")
    if "val_accuracy" in history:
        ax_acc.plot(history["val_accuracy"], label="val")
    ax_acc.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    if "loss" in history:
        ax_loss.plot(history["loss"], label="train")
    if "val_loss" in history:
        ax_loss.plot(history["val_loss"], label="val")
    ax_loss.set(title="Loss", xlabel="epoch", ylabel="loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    fig.suptitle(model_name)
    fig.tight_layout()
    path = out_dir / f"history_{_slug(model_name)}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _predict_keras(weights: Path):
    import tensorflow as tf

    from .data import build_keras_datasets
    from .models.keras_vit import AddPositionEmbedding, TransformerBlock  # noqa: F401  (registers)

    _, val_ds = build_keras_datasets(augment=False)
    model = tf.keras.models.load_model(weights)

    probs, trues = [], []
    for images, labels in val_ds:
        out = np.asarray(model.predict(images, verbose=0))
        probs.append(out[:, 1] if out.ndim == 2 and out.shape[1] > 1 else out.ravel())
        trues.append(np.asarray(labels).ravel())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(trues)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


def _predict_pytorch(model_kind: str, weights: Path, backbone_weights: Path | None):
    import torch

    from .data import build_pytorch_loaders
    from .models.pytorch_cnn import build_pytorch_cnn
    from .models.pytorch_vit import build_pytorch_cnn_vit_hybrid

    _, val_loader = build_pytorch_loaders()
    model = build_pytorch_cnn() if model_kind == "cnn" else build_pytorch_cnn_vit_hybrid()
    if model_kind == "vit" and backbone_weights is not None:
        model.load_backbone_state_dict(torch.load(backbone_weights, map_location="cpu"))
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
    model.eval()

    probs, preds, trues = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images)
            softmax = torch.softmax(logits, dim=1)
            probs.append(softmax[:, 1].cpu().numpy())
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            trues.append(labels.numpy())
    return (
        np.concatenate(trues),
        np.concatenate(preds),
        np.concatenate(probs),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained land-classification model.")
    parser.add_argument("--framework", choices=["keras", "pytorch"], required=True)
    parser.add_argument("--model", choices=["cnn", "vit"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--backbone-weights", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=config.DATA.seed)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    name = f"{args.framework}_{args.model}"

    if args.framework == "keras":
        y_true, y_pred, y_prob = _predict_keras(args.weights)
    else:
        y_true, y_pred, y_prob = _predict_pytorch(args.model, args.weights, args.backbone_weights)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, name)
    save_metrics(metrics, METRIC_DIR / f"{name}.json")
    if not args.no_figures:
        plot_confusion_matrix(metrics, name)


if __name__ == "__main__":
    main()
