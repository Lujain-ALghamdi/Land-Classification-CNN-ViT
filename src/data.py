"""Portable input pipelines for the satellite tile dataset.

The dataset is a flat two-class image folder::

    data/images_dataSAT/
        class_0_non_agri/   *.jpg
        class_1_agri/       *.jpg

It is *not* committed to this repository (see ``data/README.md``). Every helper
below raises a clear, actionable error if the folder is missing rather than
failing deep inside a framework call.

Two design fixes over the original course notebooks:

* Absolute, course-environment-specific paths (``"."`` / ``/home/...``) are
  replaced with a repo-root-relative :data:`src.config.DATASET_DIR`.
* The PyTorch split now applies *different* transforms to the train and
  validation subsets. The notebooks mutated ``dataset.transform`` after
  ``random_split``, which changed it for both subsets at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from . import config
from .seeding import torch_worker_init_fn

if TYPE_CHECKING:  # imports for type checkers only
    import tensorflow as tf
    from torch.utils.data import DataLoader


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def resolve_dataset_dir(dataset_dir: Path | str | None = None) -> Path:
    """Return the dataset directory, checking that it looks valid.

    Raises
    ------
    FileNotFoundError
        If the directory or either expected class sub-folder is absent.
    """
    root = Path(dataset_dir) if dataset_dir is not None else config.DATASET_DIR
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}\n"
            "Download 'images-dataSAT.tar' as described in data/README.md and "
            f"extract it so that {root}/class_0_non_agri and "
            f"{root}/class_1_agri exist."
        )
    missing = [c for c in config.CLASS_NAMES if not (root / c).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Dataset directory {root} is missing class folders: {missing}"
        )
    return root


def count_images(dataset_dir: Path | str | None = None) -> dict[str, int]:
    """Count image files per class. Useful for a quick sanity check."""
    root = resolve_dataset_dir(dataset_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return {
        cls: sum(1 for p in (root / cls).iterdir() if p.suffix.lower() in exts)
        for cls in config.CLASS_NAMES
    }


# --------------------------------------------------------------------------- #
# TensorFlow / Keras pipeline
# --------------------------------------------------------------------------- #
def build_keras_augmentation(data_cfg: config.DataConfig = config.DATA) -> "tf.keras.Sequential":
    """A small augmentation model applied to the training split only.

    Replaces the deprecated ``ImageDataGenerator`` with Keras preprocessing
    layers (the modern, ``tf.data``-friendly approach).
    """
    import tensorflow as tf

    layers: list = [
        tf.keras.layers.RandomFlip(
            "horizontal_and_vertical" if data_cfg.vertical_flip else "horizontal"
        ),
        tf.keras.layers.RandomRotation(data_cfg.rotation_degrees / 360.0),
        tf.keras.layers.RandomZoom(data_cfg.zoom),
    ]
    return tf.keras.Sequential(layers, name="augmentation")


def build_keras_datasets(
    dataset_dir: Path | str | None = None,
    data_cfg: config.DataConfig = config.DATA,
    *,
    augment: bool = True,
    label_mode: str = "int",
) -> tuple["tf.data.Dataset", "tf.data.Dataset"]:
    """Return ``(train_ds, val_ds)`` built with a seeded 80/20 split.

    Pixel values are scaled to ``[0, 1]``. The training dataset is shuffled,
    augmented (optional) and prefetched; the validation dataset is only
    rescaled and cached.
    """
    import tensorflow as tf

    root = resolve_dataset_dir(dataset_dir)
    common = dict(
        labels="inferred",
        label_mode=label_mode,
        class_names=list(config.CLASS_NAMES),
        image_size=(data_cfg.image_size, data_cfg.image_size),
        batch_size=data_cfg.batch_size,
        validation_split=data_cfg.validation_split,
        seed=data_cfg.seed,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(root, subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(root, subset="validation", **common)

    rescale = tf.keras.layers.Rescaling(1.0 / 255)
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=autotune)
    val_ds = val_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=autotune)

    if augment:
        augmentation = build_keras_augmentation(data_cfg)
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=autotune
        )

    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds


# --------------------------------------------------------------------------- #
# PyTorch pipeline
# --------------------------------------------------------------------------- #
def build_pytorch_transforms(data_cfg: config.DataConfig = config.DATA):
    """Return ``(train_transform, eval_transform)`` Compose pipelines."""
    from torchvision import transforms

    size = (data_cfg.image_size, data_cfg.image_size)
    normalize = transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)

    train_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.RandomRotation(data_cfg.rotation_degrees),
            transforms.RandomHorizontalFlip(p=0.5 if data_cfg.horizontal_flip else 0.0),
            transforms.RandomAffine(degrees=0, shear=data_cfg.shear * 100),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize(size), transforms.ToTensor(), normalize]
    )
    return train_transform, eval_transform


def build_pytorch_loaders(
    dataset_dir: Path | str | None = None,
    data_cfg: config.DataConfig = config.DATA,
    *,
    num_workers: int = 0,
) -> tuple["DataLoader", "DataLoader"]:
    """Return ``(train_loader, val_loader)`` with a seeded 80/20 split.

    ``num_workers`` defaults to 0 because the course notebooks' use of
    multi-process workers crashed on several local setups (Windows spawn,
    small ``/dev/shm``). Raise it if your platform is happy with it.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset, random_split
    from torchvision import datasets

    root = resolve_dataset_dir(dataset_dir)
    train_transform, eval_transform = build_pytorch_transforms(data_cfg)

    class _TransformedSubset(Dataset):
        """A ``Subset`` that applies its own transform to the raw PIL image."""

        def __init__(self, base: datasets.ImageFolder, indices: list[int], transform):
            self.base = base
            self.indices = list(indices)
            self.transform = transform

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, i: int):
            path, target = self.base.samples[self.indices[i]]
            image = self.base.loader(path)
            if self.transform is not None:
                image = self.transform(image)
            return image, target

    raw = datasets.ImageFolder(root)  # no transform; subsets add their own
    val_len = int(round(len(raw) * data_cfg.validation_split))
    train_len = len(raw) - val_len
    generator = torch.Generator().manual_seed(data_cfg.seed)
    train_split, val_split = random_split(range(len(raw)), [train_len, val_len], generator=generator)

    train_ds = _TransformedSubset(raw, list(train_split), train_transform)
    val_ds = _TransformedSubset(raw, list(val_split), eval_transform)

    def _worker_init(worker_id: int) -> None:
        torch_worker_init_fn(worker_id, data_cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_worker_init if num_workers else None,
        generator=torch.Generator().manual_seed(data_cfg.seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_worker_init if num_workers else None,
    )
    return train_loader, val_loader
