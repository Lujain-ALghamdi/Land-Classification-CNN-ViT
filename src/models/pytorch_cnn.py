"""PyTorch convolutional classifier.

The convolutional feature extractor mirrors the Keras model exactly: six blocks
of ``Conv2d(5x5, padding=2) -> ReLU -> MaxPool2d(2) -> BatchNorm2d`` with the
channel progression 3 -> 32 -> 64 -> 128 -> 256 -> 512 -> 1024.

The classifier head follows the original PyTorch lab and is intentionally
shallower than the Keras one: global average pooling, a single hidden
``Linear -> ReLU -> BatchNorm1d -> Dropout`` layer, then two output logits
trained with cross-entropy.

``ConvNet.forward_features`` exposes the pre-pool feature map so the CNN->ViT
hybrid can reuse this network as a tokenizer.

``torch`` is imported lazily: ``from src.models.pytorch_cnn import ConvNet``
works, and the import only fires when the name is first accessed (PEP 562).
"""

from __future__ import annotations

from .. import config

_CONVNET_CLASS = None


def build_feature_extractor(cnn_cfg: config.CNNConfig = config.CNN):
    """Return an ``nn.Sequential`` of the six convolutional blocks."""
    import torch.nn as nn

    blocks: list = []
    in_ch = config.DATA.channels
    for out_ch in cnn_cfg.conv_channels:
        blocks += [
            nn.Conv2d(in_ch, out_ch, cnn_cfg.kernel_size, padding=cnn_cfg.kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(out_ch),
        ]
        in_ch = out_ch
    return nn.Sequential(*blocks)


def _build_convnet_class():
    import torch
    import torch.nn as nn

    class ConvNet(nn.Module):
        """Six-block CNN with a shallow dense classifier head."""

        def __init__(self, num_classes: int = 2, cnn_cfg: config.CNNConfig = config.CNN) -> None:
            super().__init__()
            self.features = build_feature_extractor(cnn_cfg)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(cnn_cfg.conv_channels[-1], cnn_cfg.pytorch_hidden_units),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(cnn_cfg.pytorch_hidden_units),
                nn.Dropout(cnn_cfg.dropout),
                nn.Linear(cnn_cfg.pytorch_hidden_units, num_classes),
            )

        def forward_features(self, x: "torch.Tensor") -> "torch.Tensor":
            """Return the (B, 1024, H', W') feature map before pooling."""
            return self.features(x)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self.pool(self.forward_features(x)))

    return ConvNet


def __getattr__(name: str):  # PEP 562 module-level lazy attribute
    global _CONVNET_CLASS
    if name == "ConvNet":
        if _CONVNET_CLASS is None:
            _CONVNET_CLASS = _build_convnet_class()
        return _CONVNET_CLASS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_pytorch_cnn(num_classes: int = 2, cnn_cfg: config.CNNConfig = config.CNN):
    """Convenience builder mirroring :func:`src.models.keras_cnn.build_keras_cnn`."""
    return __getattr__("ConvNet")(num_classes=num_classes, cnn_cfg=cnn_cfg)
