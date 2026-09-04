"""Keras convolutional classifier.

Architecture (channels-last, 64x64x3 input):

* six convolutional blocks, each
  ``Conv2D(f, 5x5, padding="same", He-uniform init) -> ReLU
    -> MaxPool2D(2) -> BatchNorm``  with ``f`` in (32, 64, 128, 256, 512, 1024);
* ``GlobalAveragePooling2D`` to collapse the 1x1x1024 feature map;
* a six-layer dense funnel with units (64, 128, 256, 512, 1024, 2048), each
  ``Dense -> BatchNorm -> Dropout(0.4)``;
* a single sigmoid output, trained with binary cross-entropy.

The intermediate layer named ``feature_map`` (output of the last conv block's
BatchNorm) is the tap point reused by the Keras CNN->ViT hybrid.
"""

from __future__ import annotations

from .. import config


def build_keras_cnn(
    data_cfg: config.DataConfig = config.DATA,
    cnn_cfg: config.CNNConfig = config.CNN,
    *,
    compile_model: bool = True,
):
    """Build (and optionally compile) the CNN classifier."""
    import tensorflow as tf
    from tensorflow.keras import layers

    init = tf.keras.initializers.HeUniform(seed=data_cfg.seed)
    inputs = tf.keras.Input(shape=data_cfg.input_shape, name="image")
    x = inputs

    for block, filters in enumerate(cnn_cfg.conv_channels):
        x = layers.Conv2D(
            filters,
            cnn_cfg.kernel_size,
            padding="same",
            activation="relu",
            kernel_initializer=init,
            name=f"conv{block}",
        )(x)
        x = layers.MaxPooling2D(2, name=f"pool{block}")(x)
        last = block == len(cnn_cfg.conv_channels) - 1
        x = layers.BatchNormalization(name="feature_map" if last else f"bn{block}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)

    for i, units in enumerate(cnn_cfg.dense_units):
        x = layers.Dense(units, activation="relu", kernel_initializer=init, name=f"dense{i}")(x)
        x = layers.BatchNormalization(name=f"dense_bn{i}")(x)
        x = layers.Dropout(cnn_cfg.dropout, name=f"dense_drop{i}")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)
    model = tf.keras.Model(inputs, outputs, name="keras_cnn")

    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cnn_cfg.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
    return model
