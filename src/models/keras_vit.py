"""Keras CNN -> Vision Transformer hybrid.

The pretrained :func:`src.models.keras_cnn.build_keras_cnn` is used as a frozen
feature extractor. Its ``feature_map`` layer output (1x1x1024 for a 64x64 input)
is reshaped into a length-``H*W`` token sequence, given a learnable positional
embedding, refined by ``depth`` transformer encoder blocks, pooled and
classified with a softmax over the two classes.

Defaults follow the Keras hybrid lab: ``depth=4``, ``num_heads=8``,
``mlp_dim=2048``, ``dropout=0.1``, Adam at ``1e-4`` with categorical
cross-entropy. The token/embedding width equals the CNN channel count (1024).

Both custom layers are registered as Keras serializables so a saved hybrid model
can be reloaded with ``load_model`` without passing ``custom_objects``.
"""

from __future__ import annotations

from .. import config

_LAYERS: dict[str, type] = {}


def _build_layers() -> dict[str, type]:
    import tensorflow as tf
    from tensorflow.keras import layers

    @tf.keras.utils.register_keras_serializable(package="LandClassification")
    class AddPositionEmbedding(layers.Layer):
        """Add a learnable ``(1, num_patches, embed_dim)`` positional embedding."""

        def __init__(self, num_patches: int, embed_dim: int, **kwargs) -> None:
            super().__init__(**kwargs)
            self.num_patches = int(num_patches)
            self.embed_dim = int(embed_dim)

        def build(self, input_shape) -> None:
            self.pos_embedding = self.add_weight(
                name="pos_embedding",
                shape=(1, self.num_patches, self.embed_dim),
                initializer="random_normal",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, tokens):
            return tokens + self.pos_embedding

        def get_config(self):
            return {**super().get_config(), "num_patches": self.num_patches, "embed_dim": self.embed_dim}

    @tf.keras.utils.register_keras_serializable(package="LandClassification")
    class TransformerBlock(layers.Layer):
        """Post-norm transformer encoder block with multi-head attention."""

        def __init__(
            self,
            embed_dim: int,
            num_heads: int = 8,
            mlp_dim: int = 2048,
            dropout: float = 0.1,
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            self.embed_dim = int(embed_dim)
            self.num_heads = int(num_heads)
            self.mlp_dim = int(mlp_dim)
            self.dropout = float(dropout)
            self.attention = layers.MultiHeadAttention(num_heads, key_dim=embed_dim)
            self.norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = layers.LayerNormalization(epsilon=1e-6)
            self.mlp = tf.keras.Sequential(
                [
                    layers.Dense(mlp_dim, activation="gelu"),
                    layers.Dropout(dropout),
                    layers.Dense(embed_dim),
                    layers.Dropout(dropout),
                ],
                name="mlp",
            )

        def call(self, x, training=False):
            x = self.norm1(x + self.attention(x, x, training=training))
            return self.norm2(x + self.mlp(x, training=training))

        def get_config(self):
            return {
                **super().get_config(),
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }

    return {"AddPositionEmbedding": AddPositionEmbedding, "TransformerBlock": TransformerBlock}


def __getattr__(name: str):  # PEP 562
    if not _LAYERS:
        _LAYERS.update(_build_layers())
    if name in _LAYERS:
        return _LAYERS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_keras_cnn_vit_hybrid(
    cnn_model=None,
    *,
    feature_layer: str = "feature_map",
    num_classes: int = 2,
    vit_cfg: config.ViTConfig = config.VIT,
    freeze_backbone: bool = True,
    compile_model: bool = True,
):
    """Assemble the hybrid on top of ``cnn_model`` (built fresh if ``None``)."""
    import tensorflow as tf
    from tensorflow.keras import layers

    from .keras_cnn import build_keras_cnn

    add_pos = __getattr__("AddPositionEmbedding")
    transformer = __getattr__("TransformerBlock")

    if cnn_model is None:
        cnn_model = build_keras_cnn(compile_model=False)
    cnn_model.trainable = not freeze_backbone

    features = cnn_model.get_layer(feature_layer).output
    _, height, width, channels = features.shape
    num_tokens = int(height) * int(width)

    x = layers.Reshape((num_tokens, channels), name="tokens")(features)
    x = add_pos(num_tokens, channels, name="pos_embed")(x)
    for i in range(vit_cfg.keras_depth):
        x = transformer(
            channels,
            num_heads=vit_cfg.keras_num_heads,
            mlp_dim=vit_cfg.keras_mlp_dim,
            dropout=vit_cfg.dropout,
            name=f"encoder{i}",
        )(x)
    x = layers.GlobalAveragePooling1D(name="token_pool")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="prediction")(x)

    model = tf.keras.Model(cnn_model.input, outputs, name="keras_cnn_vit_hybrid")
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=vit_cfg.keras_learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return model
