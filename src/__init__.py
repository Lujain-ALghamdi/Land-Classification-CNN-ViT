"""Land classification with CNNs and Vision Transformers.

A framework-parallel implementation (TensorFlow/Keras and PyTorch) of a binary
land-use classifier that separates agricultural from non-agricultural satellite
tiles, plus CNN -> Vision Transformer hybrids that reuse the convolutional
feature extractor as a tokenizer for a transformer encoder.

The package is organised as:

    src.config     - dataclasses holding every hyperparameter
    src.seeding     - deterministic seeding for Python, NumPy, TF and PyTorch
    src.data        - portable dataset discovery and input pipelines
    src.models      - the Keras and PyTorch model builders
    src.train       - command-line training entry point
    src.evaluate    - metrics and evaluation entry point
"""

__all__ = ["config", "seeding", "data", "models", "train", "evaluate"]
