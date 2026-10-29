"""Model evaluation: metrics, confusion matrix, classification report, example predictions."""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from . import config
from . import visualize as viz


def compute_metrics(y_true, y_pred, loss: float, class_names) -> dict:
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "test_accuracy": float(np.mean(y_true == y_pred)),
        "test_loss": float(loss),
        "precision_macro": float(precision_m),
        "recall_macro": float(recall_m),
        "f1_macro": float(f1_m),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
    }


def evaluate_model(name, model, x_test, y_test, class_names, extra=None, save_outputs=True, n_examples=12):
    """Evaluate on unseen test data. `x_test` must already be normalised (N,28,28,1)."""
    n_classes = len(class_names)
    loss = float(model.evaluate(x_test, y_test, batch_size=512, verbose=0)[0])
    probs = model.predict(x_test, batch_size=512, verbose=0)
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    metrics = compute_metrics(y_test, y_pred, loss, class_names)
    metrics.update({"dataset": name, "num_classes": n_classes,
                    "num_test_samples": int(len(y_test)), "input_shape": list(x_test.shape[1:])})
    if extra:
        metrics.update(extra)

    labels = list(range(n_classes))
    report_text = classification_report(y_test, y_pred, labels=labels, target_names=class_names,
                                        digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    if save_outputs:
        _save_outputs(name, metrics, report_text, cm, x_test, y_test, y_pred, confidence, class_names, n_examples)

    return {"metrics": metrics, "report_text": report_text, "confusion_matrix": cm,
            "y_pred": y_pred, "probs": probs, "confidence": confidence}


def _save_outputs(name, metrics, report_text, cm, x_test, y_test, y_pred, confidence, class_names, n_examples):
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.metrics_path(name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config.report_path(name).write_text(report_text, encoding="utf-8")
    viz.save_figure(viz.plot_confusion_matrix(cm, class_names, title=f"{name.upper()} confusion matrix"),
                    f"{name}_confusion_matrix.png")

    rng = np.random.default_rng(config.SEED)
    correct = np.flatnonzero(y_test == y_pred)
    wrong = np.flatnonzero(y_test != y_pred)
    if len(correct):
        picks = rng.choice(correct, size=min(n_examples, len(correct)), replace=False)
        viz.save_figure(viz.plot_prediction_grid(x_test, y_test, y_pred, confidence, picks, class_names,
                                                 title="Correctly classified samples"),
                        f"{name}_correct_predictions.png")
    if len(wrong):  # most confident mistakes are the most informative
        picks = wrong[np.argsort(-confidence[wrong])][:n_examples]
        viz.save_figure(viz.plot_prediction_grid(x_test, y_test, y_pred, confidence, picks, class_names,
                                                 title="Misclassified samples (most confident mistakes)"),
                        f"{name}_incorrect_predictions.png")


def export_sample_images(name, x_test, y_test, class_names) -> int:
    """Save one real test image per class into sample_images/<dataset>/ (white ink on black)."""
    folder = config.SAMPLES_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    for class_id, class_name in enumerate(class_names):
        hits = np.flatnonzero(y_test == class_id)
        if len(hits):
            img = (np.squeeze(x_test[hits[0]]) * 255).astype(np.uint8)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(folder / f"{class_name}_{int(hits[0])}.png"), img)
            saved += 1
    return saved


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model on the test set.")
    parser.add_argument("--dataset", choices=list(config.DATASETS), required=True)
    args = parser.parse_args()

    from tensorflow import keras
    from .preprocessing import load_dataset, prepare_arrays

    path = config.model_path(args.dataset)
    if not path.exists():
        raise SystemExit(f"Model not found: {path}. Train it first: python -m src.train --dataset {args.dataset}")
    model = keras.models.load_model(path)
    _, _, x_test, y_test, class_names = load_dataset(args.dataset)
    results = evaluate_model(args.dataset, model, prepare_arrays(x_test), y_test, class_names)
    print(json.dumps(results["metrics"], indent=2))
    print(results["report_text"])


if __name__ == "__main__":
    main()
