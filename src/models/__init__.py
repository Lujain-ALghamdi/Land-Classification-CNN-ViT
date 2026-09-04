"""Model builders for both frameworks.

Import the submodule you need; each defers its heavy framework import until the
builder is actually called:

    from src.models.keras_cnn import build_keras_cnn
    from src.models.pytorch_cnn import ConvNet
    from src.models.keras_vit import build_keras_cnn_vit_hybrid
    from src.models.pytorch_vit import CNNViTHybrid
"""

__all__ = ["keras_cnn", "pytorch_cnn", "keras_vit", "pytorch_vit"]
