"""Reusable prediction module used by the Streamlit app, notebooks and scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from tensorflow import keras

from . import config
from .preprocessing import (
    ImageProcessingError,
    preprocess_image_with_steps,
    to_model_input,
)


class ModelNotFoundError(FileNotFoundError):
    """The trained model file does not exist yet."""


class ModelLoadError(RuntimeError):
    """The model file exists but could not be loaded."""


@dataclass
class PredictionResult:
    label: str
    confidence: float
    top_k: list = field(default_factory=list)
    probabilities: np.ndarray = None
    processed_image: np.ndarray = None
    steps: dict = field(default_factory=dict)


def load_class_names(dataset: str):
    """Load class names from saved mapping or config fallback."""

    path = config.labels_path(dataset)

    if path.exists():
        try:
            return json.loads(
                path.read_text(encoding="utf-8")
            )["class_names"]
        except (OSError, KeyError, json.JSONDecodeError):
            pass

    return list(
        config.get_dataset_info(dataset)["class_names"]
    )


class CharacterRecognizer:

    def __init__(
        self,
        model: keras.Model,
        class_names,
        dataset: str,
    ):
        self.model = model
        self.class_names = list(class_names)

        # IMPORTANT:
        # Remember whether this recognizer is MNIST or EMNIST.
        self.dataset = dataset.lower()


    @classmethod
    def from_dataset(
        cls,
        dataset: str,
    ) -> "CharacterRecognizer":

        dataset = dataset.lower()

        path = config.model_path(dataset)

        if not path.exists():
            raise ModelNotFoundError(
                f"Model file not found: models/{path.name}. "
                f"Train it first with: "
                f"python -m src.train --dataset {dataset}"
            )

        try:
            model = keras.models.load_model(path)

        except Exception as exc:
            raise ModelLoadError(
                f"Could not load models/{path.name}: {exc}"
            ) from exc

        return cls(
            model=model,
            class_names=load_class_names(dataset),
            dataset=dataset,
        )


    def predict(
        self,
        image,
        top_k: int = 3,
    ) -> PredictionResult:

        # Dataset is now passed to preprocessing.
        processed, steps = preprocess_image_with_steps(
            image,
            dataset=self.dataset,
        )

        batch = to_model_input(processed)

        expected = tuple(
            self.model.input_shape[1:]
        )

        if batch.shape[1:] != expected:
            raise ImageProcessingError(
                f"Image shape {batch.shape[1:]} "
                f"does not match model input {expected}."
            )

        probs = np.asarray(
            self.model(
                batch,
                training=False,
            )
        )[0]

        order = np.argsort(
            probs
        )[::-1][:top_k]

        top = [
            (
                self.class_names[i],
                float(probs[i]),
            )
            for i in order
        ]

        return PredictionResult(
            label=top[0][0],
            confidence=top[0][1],
            top_k=top,
            probabilities=probs,
            processed_image=processed,
            steps=steps,
        )