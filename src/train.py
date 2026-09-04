"""Unified training entry point.

Examples
--------
    python -m src.train --framework keras   --model cnn --epochs 3
    python -m src.train --framework pytorch --model cnn --epochs 3
    python -m src.train --framework keras   --model vit --epochs 5
    python -m src.train --framework pytorch --model vit --epochs 5 \
        --backbone-weights checkpoints/pytorch_cnn.pth

Each run writes model weights and a ``*_history.json`` file to ``--output-dir``
(``checkpoints/`` by default). Nothing is downloaded automatically: the dataset
must already be present under ``data/images_dataSAT/`` (see ``data/README.md``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from . import config
from .seeding import seed_everything


# --------------------------------------------------------------------------- #
# Keras
# --------------------------------------------------------------------------- #
def _train_keras(model_kind: str, args: argparse.Namespace) -> dict:
    import tensorflow as tf

    from .data import build_keras_datasets
    from .models.keras_cnn import build_keras_cnn
    from .models.keras_vit import build_keras_cnn_vit_hybrid

    train_ds, val_ds = build_keras_datasets(dataset_dir=args.data_dir)
    model = build_keras_cnn() if model_kind == "cnn" else build_keras_cnn_vit_hybrid()
    model.summary()

    weights_path = args.output_dir / f"keras_{model_kind}.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        weights_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
    )
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[checkpoint]
    )
    return {k: [float(v) for v in vals] for k, vals in history.history.items()}


# --------------------------------------------------------------------------- #
# PyTorch
# --------------------------------------------------------------------------- #
def _train_pytorch(model_kind: str, args: argparse.Namespace) -> dict:
    import torch
    import torch.nn as nn

    from .data import build_pytorch_loaders
    from .models.pytorch_cnn import build_pytorch_cnn
    from .models.pytorch_vit import build_pytorch_cnn_vit_hybrid

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_pytorch_loaders(
        dataset_dir=args.data_dir, num_workers=args.num_workers
    )

    if model_kind == "cnn":
        model = build_pytorch_cnn()
        lr = config.CNN.learning_rate
    else:
        model = build_pytorch_cnn_vit_hybrid()
        lr = config.VIT.learning_rate
        if args.backbone_weights is not None:
            state = torch.load(args.backbone_weights, map_location="cpu")
            model.load_backbone_state_dict(state)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    history: dict[str, list[float]] = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    best_val = float("inf")
    weights_path = args.output_dir / f"pytorch_{model_kind}.pth"

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_correct, tr_total = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        va_loss, va_correct, va_total = _run_epoch(model, val_loader, criterion, device, None)

        history["loss"].append(tr_loss / max(len(train_loader), 1))
        history["val_loss"].append(va_loss / max(len(val_loader), 1))
        history["accuracy"].append(tr_correct / max(tr_total, 1))
        history["val_accuracy"].append(va_correct / max(va_total, 1))

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train loss {history['loss'][-1]:.4f} acc {history['accuracy'][-1]:.4f} | "
            f"val loss {history['val_loss'][-1]:.4f} acc {history['val_accuracy'][-1]:.4f} | "
            f"{time.time() - start:.1f}s"
        )
        if history["val_loss"][-1] < best_val:
            best_val = history["val_loss"][-1]
            torch.save(model.state_dict(), weights_path)
            print(f"  saved new best to {weights_path}")

    return history


def _run_epoch(model, loader, criterion, device, optimizer):
    import torch

    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.item())
            correct += int((logits.argmax(1) == labels).sum().item())
            total += int(labels.size(0))
    return loss_sum, correct, total


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a land-classification model.")
    parser.add_argument("--framework", choices=["keras", "pytorch"], required=True)
    parser.add_argument("--model", choices=["cnn", "vit"], required=True)
    parser.add_argument("--epochs", type=int, default=None, help="defaults to the config value")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=config.REPO_ROOT / "checkpoints")
    parser.add_argument("--backbone-weights", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=config.DATA.seed)
    args = parser.parse_args(argv)

    if args.epochs is None:
        args.epochs = config.CNN.epochs if args.model == "cnn" else config.VIT.epochs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    print(f"Training {args.framework} {args.model} for {args.epochs} epoch(s) (seed {args.seed})")

    trainer = _train_keras if args.framework == "keras" else _train_pytorch
    history = trainer(args.model, args)

    history_path = args.output_dir / f"{args.framework}_{args.model}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
