"""Training entry point.  Run from the project root:

    python -m src.train --dataset mnist
    python -m src.train --dataset emnist
"""
import matplotlib

matplotlib.use("Agg")  # scripts save figures to disk; no GUI needed

import argparse
import json
import time

import numpy as np
import pandas as pd
from tensorflow import keras

from . import config
from . import visualize as viz
from .evaluate import evaluate_model, export_sample_images
from .model import build_cnn, compile_model, make_callbacks, save_label_mapping
from .preprocessing import DatasetLoadError, load_dataset, prepare_arrays, split_train_val


def run_eda(name, x_train, y_train, class_names):
    """Save the relevant EDA figures and print dataset facts."""
    print(f"Image shape: {x_train.shape[1:]}  dtype: {x_train.dtype}  "
          f"pixel range: {x_train.min()}-{x_train.max()}  classes: {len(class_names)}")
    viz.save_figure(viz.plot_samples(x_train, y_train, class_names, title=f"{name.upper()} samples"),
                    f"{name}_samples.png")
    viz.save_figure(viz.plot_class_distribution(y_train, class_names, f"{name.upper()} class distribution"),
                    f"{name}_class_distribution.png")
    viz.save_figure(viz.plot_pixel_histogram(x_train), f"{name}_pixel_histogram.png")
    if name == "emnist":
        viz.save_figure(viz.plot_orientation_check(x_train), "emnist_orientation_check.png")


def train(dataset, epochs=30, batch_size=128, learning_rate=1e-3, seed=config.SEED, augment=True, eda=True):
    config.ensure_directories()
    keras.utils.set_random_seed(seed)

    x_train_full, y_train_full, x_test, y_test, class_names = load_dataset(dataset)
    print(f"Train: {len(x_train_full)}  Test: {len(x_test)}  Classes: {len(class_names)}")
    if eda:
        run_eda(dataset, x_train_full, y_train_full, class_names)

    x_train, x_val, y_train, y_val = split_train_val(x_train_full, y_train_full, seed=seed)
    x_train, x_val, x_test = prepare_arrays(x_train), prepare_arrays(x_val), prepare_arrays(x_test)
    print(f"Split -> train {len(x_train)}  validation {len(x_val)}  test {len(x_test)}")

    model = compile_model(build_cnn(len(class_names), augment=augment, name=f"{dataset}_cnn"), learning_rate)
    model.summary()

    started = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size, verbose=2,  # prints loss/accuracy/val_loss/val_accuracy per epoch
        callbacks=make_callbacks(config.model_path(dataset), config.history_path(dataset)),
    )
    minutes = (time.time() - started) / 60

    viz.save_figure(viz.plot_history(history.history, f"{dataset.upper()} training history"),
                    f"{dataset}_training_curves.png")

    best_model = keras.models.load_model(config.model_path(dataset))  # best checkpoint, not last epoch
    save_label_mapping(dataset, class_names)
    extra = {"num_train_samples": int(len(x_train)), "num_validation_samples": int(len(x_val)),
             "epochs_run": len(history.history["loss"]), "training_minutes": round(minutes, 2),
             "parameters": int(best_model.count_params()), "augmentation": bool(augment), "seed": seed}
    results = evaluate_model(dataset, best_model, x_test, y_test, class_names, extra=extra)
    saved = export_sample_images(dataset, x_test, y_test, class_names)

    print("\n=== Test results ===")
    print(json.dumps(results["metrics"], indent=2))
    print(results["report_text"])
    print(f"Exported {saved} sample images to sample_images/{dataset}/")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train the handwritten character CNN.")
    parser.add_argument("--dataset", choices=list(config.DATASETS), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--no-augment", action="store_true", help="disable data augmentation")
    parser.add_argument("--no-eda", action="store_true", help="skip EDA figures")
    args = parser.parse_args()
    try:
        train(args.dataset, args.epochs, args.batch_size, args.lr, args.seed,
              augment=not args.no_augment, eda=not args.no_eda)
    except DatasetLoadError as exc:
        raise SystemExit(f"Dataset error: {exc}")


if __name__ == "__main__":
    main()
