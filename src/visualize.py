"""Plotting helpers shared by the training script, evaluation script and notebooks."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def save_figure(fig, filename: str):
    """Save a figure into outputs/figures and close it."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _grid(n_items: int, cols: int):
    rows = math.ceil(n_items / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.7))
    return fig, np.atleast_1d(axes).ravel()


def plot_samples(images, labels, class_names, n=16, cols=8, title="Sample images", seed=config.SEED):
    """Random labelled samples. `images` may be (N,28,28) or (N,28,28,1)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), size=min(n, len(images)), replace=False)
    fig, axes = _grid(len(idx), cols)
    for ax, i in zip(axes, idx):
        ax.imshow(np.squeeze(images[i]), cmap="gray")
        ax.set_title(class_names[int(labels[i])], fontsize=10)
        ax.axis("off")
    for ax in axes[len(idx):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_class_distribution(labels, class_names, title="Class distribution"):
    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar([class_names[i] for i in counts.index], counts.values, color="#4C72B0")
    ax.set_xlabel("Class")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pixel_histogram(images, title="Pixel intensity distribution"):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(np.asarray(images).ravel()[::50], bins=50, color="#55A868", log=True)
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Count (log)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_orientation_check(corrected_images, n=8):
    """Top row: raw EMNIST orientation. Bottom row: after the orientation fix."""
    fig, axes = plt.subplots(2, n, figsize=(n * 1.3, 3.2))
    for i in range(n):
        axes[0, i].imshow(corrected_images[i].T, cmap="gray")  # transpose back = raw storage
        axes[1, i].imshow(corrected_images[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Raw", fontsize=9, loc="left")
    axes[1, 0].set_title("Corrected", fontsize=9, loc="left")
    fig.tight_layout()
    return fig


def plot_preprocessing_steps(steps: dict, title="Preprocessing pipeline"):
    fig, axes = plt.subplots(1, len(steps), figsize=(len(steps) * 2.4, 2.9))
    for ax, (name, img) in zip(np.atleast_1d(axes), steps.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_history(history: dict, title="Training history"):
    """`history` is Keras `History.history` (or a DataFrame converted to dict)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, label in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        ax.plot(history[metric], label=f"Training {label.lower()}")
        ax.plot(history[f"val_{metric}"], label=f"Validation {label.lower()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"Training vs Validation {label}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names, normalize=True, title="Confusion matrix"):
    matrix = cm.astype("float")
    if normalize:
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    size = max(5, 0.45 * len(class_names))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks, class_names)
    ax.set_yticks(ticks, class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title + (" (row-normalised)" if normalize else ""))
    if len(class_names) <= 12:  # annotate only when readable
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                value = f"{matrix[i, j]:.2f}" if normalize else int(matrix[i, j])
                ax.text(j, i, value, ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    return fig


def plot_prediction_grid(images, y_true, y_pred, confidences, indices, class_names,
                         title="Predictions", cols=6):
    """Show Actual / Predicted / Confidence for the chosen sample indices."""
    indices = list(indices)
    rows = max(1, math.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, indices):
        ax.imshow(np.squeeze(images[i]), cmap="gray")
        ok = y_true[i] == y_pred[i]
        ax.set_title(f"Actual: {class_names[int(y_true[i])]}\nPredicted: {class_names[int(y_pred[i])]}\n"
                     f"Confidence: {confidences[i] * 100:.1f}%",
                     fontsize=8, color="green" if ok else "red")
        ax.axis("off")
    for ax in axes[len(indices):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
