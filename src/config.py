"""Central configuration for the land-classification experiments.

Every value here is taken from the original IBM AI Engineering capstone labs so
that the code in this repository documents the *actual* experimental setup
rather than an idealised one. Nothing is invented: where the labs used a
deliberately small number (for example ``epochs = 3`` "for your convenience")
that number is reproduced and called out in the README.

Paths are resolved relative to the repository root so the code runs unchanged on
Windows, macOS and Linux.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "data"

#: Directory that must contain ``class_0_non_agri/`` and ``class_1_agri/``.
#: See ``data/README.md`` for how to obtain and place the dataset.
DATASET_DIR: Path = DATA_DIR / "images_dataSAT"

CLASS_NAMES: tuple[str, ...] = ("class_0_non_agri", "class_1_agri")
CLASS_LABELS: tuple[str, ...] = ("non-agri", "agri")

#: ImageNet statistics, used by the PyTorch pipelines (matches the labs).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataConfig:
    """Input-pipeline hyperparameters shared by both frameworks."""

    image_size: int = 64
    channels: int = 3
    batch_size: int = 128
    validation_split: float = 0.2
    seed: int = 7331  # the value used by the Keras labs; PyTorch labs used 42

    # Training-time augmentation (applied to the training split only).
    rotation_degrees: int = 40
    horizontal_flip: bool = True
    vertical_flip: bool = False
    shear: float = 0.2
    zoom: float = 0.2

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Channels-last shape used by Keras models."""
        return (self.image_size, self.image_size, self.channels)


@dataclass(frozen=True)
class CNNConfig:
    """Convolutional classifier hyperparameters.

    The convolutional stack is identical across frameworks: six blocks of
    ``Conv(5x5, same) -> ReLU -> MaxPool(2) -> BatchNorm`` with the channel
    progression below. The classifier heads differ slightly between the two
    original labs and that difference is preserved on purpose:

    * Keras: global average pooling, then a six-layer dense funnel
      (``dense_units``) each with BatchNorm + Dropout, then a single
      sigmoid unit trained with binary cross-entropy.
    * PyTorch: global average pooling, a single hidden dense layer
      (``pytorch_hidden_units``) with BatchNorm + Dropout, then two logits
      trained with cross-entropy.
    """

    conv_channels: tuple[int, ...] = (32, 64, 128, 256, 512, 1024)
    kernel_size: int = 5
    dense_units: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)
    pytorch_hidden_units: int = 2048
    dropout: float = 0.4
    learning_rate: float = 1e-3
    epochs: int = 3  # deliberately low in the source labs
    model_dir: Path = field(default=REPO_ROOT / "checkpoints")


@dataclass(frozen=True)
class ViTConfig:
    """CNN -> Vision Transformer hybrid hyperparameters.

    The pretrained CNN feature extractor (six conv blocks, 1024 output
    channels) is frozen and its feature map is turned into a token sequence
    that a small transformer encoder classifies.

    Defaults below match the PyTorch hybrid lab. The Keras hybrid lab used
    ``embed_dim`` equal to the CNN channel count (1024), ``depth = 4`` and a
    lower learning rate (1e-4); those are exposed as the ``keras_*`` fields.
    """

    embed_dim: int = 768
    depth: int = 3
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    max_tokens: int = 50
    learning_rate: float = 1e-3
    epochs: int = 5

    # Keras-hybrid-specific values.
    keras_depth: int = 4
    keras_num_heads: int = 8
    keras_mlp_dim: int = 2048
    keras_learning_rate: float = 1e-4
    keras_feature_layer: str = "feature_map"  # named layer tapped for tokens


DATA = DataConfig()
CNN = CNNConfig()
VIT = ViTConfig()
