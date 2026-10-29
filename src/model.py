"""CNN architecture, compilation and training callbacks."""
from __future__ import annotations

import json

from tensorflow import keras
from tensorflow.keras import layers

from . import config


def _conv_bn_relu(x, filters: int):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def build_backbone(x):
    """Convolutional feature extractor (28x28x1 -> 7x7x128 feature map).

    Kept separate from the classification head so it can later be reused as the
    CNN part of a CRNN (feature map -> BiLSTM -> CTC) for word recognition.
    """
    for filters in (32, 64):
        x = _conv_bn_relu(x, filters)
        x = _conv_bn_relu(x, filters)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
    return _conv_bn_relu(x, 128)


def _augmentation_block() -> keras.Sequential:
    """Small random rotations/shifts/zooms; active only during training."""
    return keras.Sequential(
        [
            layers.RandomRotation(0.05, fill_mode="constant"),
            layers.RandomTranslation(0.08, 0.08, fill_mode="constant"),
            layers.RandomZoom(0.1, fill_mode="constant"),
        ],
        name="augmentation",
    )


def build_cnn(num_classes: int, input_shape=config.INPUT_SHAPE, augment: bool = True,
              name: str = "handwritten_cnn") -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = _augmentation_block()(inputs) if augment else inputs
    x = build_backbone(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name=name)


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",  # integer labels
        metrics=["accuracy"],
    )
    return model


def make_callbacks(checkpoint_path, log_csv_path=None, patience: int = 5):
    """Keep the best weights (lowest validation loss), stop early, and decay the LR."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]
    if log_csv_path is not None:
        config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        callbacks.append(keras.callbacks.CSVLogger(str(log_csv_path)))
    return callbacks


def save_label_mapping(dataset: str, class_names) -> None:
    """Store class index -> character mapping next to the model."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "input_shape": list(config.INPUT_SHAPE),
        "class_names": list(class_names),
        "label_mapping": {str(i): c for i, c in enumerate(class_names)},
    }
    config.labels_path(dataset).write_text(json.dumps(payload, indent=2), encoding="utf-8")
