"""Dataset loading and image preprocessing.

The same 28x28, white-ink-on-black, centred representation is used for the
training datasets (MNIST/EMNIST) and for user images (upload / canvas), so
training and real-world prediction stay consistent.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from . import config


MIN_CONTRAST = 20

# MNIST convention: character occupies roughly a 20x20 region
# inside the final 28x28 image.
DIGIT_BOX = 20

# Ignore extremely tiny connected components.
MIN_COMPONENT_AREA = 8

# ============================================================
# EXCEPTIONS
# ============================================================


class ImageProcessingError(Exception):
    """The supplied image cannot be processed."""


class BlankImageError(ImageProcessingError):
    """The image contains no visible handwriting."""


class DatasetLoadError(RuntimeError):
    """A dataset could not be downloaded or read."""


# ============================================================
# MNIST DATASET
# ============================================================


def load_mnist():
    """
    Load MNIST.

    Returns:
        x_train: uint8 images (N, 28, 28)
        y_train: labels 0-9
        x_test: uint8 images
        y_test: labels 0-9
    """

    try:
        from tensorflow.keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    except Exception as exc:

        raise DatasetLoadError(
            "Could not load MNIST. "
            "Check your internet connection and try again."
        ) from exc

    return (
        x_train,
        y_train.astype(np.int64),
        x_test,
        y_test.astype(np.int64),
    )


# ============================================================
# EMNIST HELPERS
# ============================================================


def fix_emnist_orientation(images: np.ndarray) -> np.ndarray:
    """
    EMNIST images from some sources are stored transposed.

    Transposing the two spatial axes restores the upright
    orientation.
    """

    return np.ascontiguousarray(
        np.transpose(
            images,
            (0, 2, 1),
        )
    )


def _zero_based_letter_labels(labels: np.ndarray) -> np.ndarray:
    """Convert EMNIST Letters labels from 1-26 to 0-25 (0=A ... 25=Z)."""
    labels = np.asarray(labels).astype(np.int64)

    if labels.size == 0:
        raise DatasetLoadError("EMNIST returned an empty label array.")

    # EMNIST CSV labels are 1-based.
    # A dataset split may not contain every label from 1 to 26.
    if labels.min() >= 1 and labels.max() <= 26:
        labels = labels - 1

    if labels.min() < 0 or labels.max() > 25:
        raise DatasetLoadError(
            "Unexpected EMNIST label range; expected 1-26 or 0-25."
        )

    return labels


# ============================================================
# METHOD 1: EMNIST PYTHON PACKAGE
# ============================================================


def _emnist_from_package():
    """
    Try loading EMNIST using the 'emnist' Python package.

    This method is kept for compatibility, but if its download
    source fails, load_emnist_letters() will automatically try
    other methods.
    """

    from emnist import (
        extract_test_samples,
        extract_training_samples,
    )

    x_train, y_train = extract_training_samples(
        "letters"
    )

    x_test, y_test = extract_test_samples(
        "letters"
    )

    return (
        x_train,
        y_train,
        x_test,
        y_test,
    )


# ============================================================
# METHOD 2: TENSORFLOW DATASETS
# ============================================================


def _emnist_from_tfds():
    """
    Load EMNIST Letters using TensorFlow Datasets.

    TFDS provides EMNIST as 'emnist/letters'.

    The dataset is converted to NumPy arrays so the rest of the
    project can use the same training pipeline.
    """

    try:
        import tensorflow_datasets as tfds

    except ImportError as exc:

        raise ImportError(
            "tensorflow-datasets is not installed. "
            "Install it with: pip install tensorflow-datasets"
        ) from exc

    try:

        train_ds, test_ds = tfds.load(
            "emnist/letters",
            split=[
                "train",
                "test",
            ],
            as_supervised=True,
            batch_size=-1,
        )

        x_train, y_train = tfds.as_numpy(
            train_ds
        )

        x_test, y_test = tfds.as_numpy(
            test_ds
        )

    except Exception as exc:

        raise DatasetLoadError(
            "TensorFlow Datasets could not download/load "
            f"EMNIST Letters: {exc}"
        ) from exc

    # TFDS returns:
    # (N, 28, 28, 1)
    #
    # Convert to:
    # (N, 28, 28)

    if (
        x_train.ndim == 4
        and x_train.shape[-1] == 1
    ):
        x_train = x_train[..., 0]

    if (
        x_test.ndim == 4
        and x_test.shape[-1] == 1
    ):
        x_test = x_test[..., 0]

    return (
        x_train.astype(np.uint8),
        y_train.astype(np.int64),
        x_test.astype(np.uint8),
        y_test.astype(np.int64),
    )


# ============================================================
# METHOD 3: LOCAL CSV FILES
# ============================================================


def _emnist_from_csv():
    """
    Load manually downloaded EMNIST CSV files from data/.

    Expected:

        data/emnist-letters-train.csv
        data/emnist-letters-test.csv
    """

    train_csv = (
        config.DATA_DIR
        / "emnist-letters-train.csv"
    )

    test_csv = (
        config.DATA_DIR
        / "emnist-letters-test.csv"
    )

    if not (
        train_csv.exists()
        and test_csv.exists()
    ):

        raise FileNotFoundError(
            f"{train_csv.name} / "
            f"{test_csv.name} "
            "not found in data/"
        )

    def read(path):

        values = pd.read_csv(
            path,
            header=None,
        ).to_numpy()

        labels = values[:, 0]

        images = (
            values[:, 1:]
            .reshape(-1, 28, 28)
            .astype(np.uint8)
        )

        return images, labels

    x_train, y_train = read(
        train_csv
    )

    x_test, y_test = read(
        test_csv
    )

    return (
        x_train,
        y_train,
        x_test,
        y_test,
    )


# ============================================================
# EMNIST MAIN LOADER
# ============================================================


def load_emnist_letters():
    """
    Load EMNIST Letters.

    Loading order:

    1. TensorFlow Datasets
    2. emnist Python package
    3. Local CSV files

    Returns orientation-corrected uint8 images and labels
    0-25 representing A-Z.
    """

    errors = []

    # TFDS first because the installed "emnist" package
    # download is failing in the current environment.
    loaders = (
        _emnist_from_tfds,
        _emnist_from_package,
        _emnist_from_csv,
    )

    successful_loader = None

    for loader in loaders:

        try:

            x_train, y_train, x_test, y_test = (
                loader()
            )

            successful_loader = loader.__name__

            break

        except Exception as exc:

            errors.append(
                f"{loader.__name__}: {exc}"
            )

    else:

        details = " | ".join(
            errors
        )

        raise DatasetLoadError(
            "Could not load EMNIST Letters. "
            "All available loading methods failed. "
            f"Details: {details}"
        )

    # --------------------------------------------------------
    # Validate image shapes
    # --------------------------------------------------------

    x_train = np.asarray(
        x_train
    )

    x_test = np.asarray(
        x_test
    )

    if x_train.ndim != 3:

        raise DatasetLoadError(
            "Unexpected EMNIST training image shape: "
            f"{x_train.shape}"
        )

    if x_test.ndim != 3:

        raise DatasetLoadError(
            "Unexpected EMNIST test image shape: "
            f"{x_test.shape}"
        )

    if x_train.shape[1:] != (28, 28):

        raise DatasetLoadError(
            "EMNIST training images must be 28x28. "
            f"Received {x_train.shape[1:]}."
        )

    if x_test.shape[1:] != (28, 28):

        raise DatasetLoadError(
            "EMNIST test images must be 28x28. "
            f"Received {x_test.shape[1:]}."
        )

    x_train = x_train.astype(
        np.uint8
    )

    x_test = x_test.astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Orientation
    # --------------------------------------------------------
    #
    # The original EMNIST package/CSV format needs the
    # transpose correction used by this project.
    #
    # TFDS already handles the dataset representation, so
    # don't blindly transpose TFDS images here.

    if successful_loader in (
        "_emnist_from_package",
        "_emnist_from_csv",
    ):

        x_train = fix_emnist_orientation(
            x_train
        )

        x_test = fix_emnist_orientation(
            x_test
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y_train = _zero_based_letter_labels(
        y_train
    )

    y_test = _zero_based_letter_labels(
        y_test
    )

    return (
        x_train,
        y_train,
        x_test,
        y_test,
    )


# ============================================================
# GENERAL DATASET LOADER
# ============================================================


def load_dataset(name: str):
    """
    Load dataset by key.

    Returns:
        x_train
        y_train
        x_test
        y_test
        class_names
    """

    info = config.get_dataset_info(
        name
    )

    if name == "mnist":

        loader = load_mnist

    elif name == "emnist":

        loader = load_emnist_letters

    else:

        raise DatasetLoadError(
            f"Unknown dataset: {name}"
        )

    x_train, y_train, x_test, y_test = (
        loader()
    )

    return (
        x_train,
        y_train,
        x_test,
        y_test,
        list(info["class_names"]),
    )


# ============================================================
# TRAINING ARRAY PREPARATION
# ============================================================


def prepare_arrays(images: np.ndarray) -> np.ndarray:
    """
    Normalise:

        0-255 -> 0-1

    And change:

        (N, 28, 28)

    to:

        (N, 28, 28, 1)
    """

    return (
        images.astype("float32")
        / 255.0
    )[..., np.newaxis]


def split_train_val(
    x,
    y,
    val_fraction: float = 0.1,
    seed: int = config.SEED,
):
    """
    Create stratified training/validation split.
    """

    return train_test_split(
        x,
        y,
        test_size=val_fraction,
        stratify=y,
        random_state=seed,
    )


# ============================================================
# CANVAS IMAGE CONVERSION
# ============================================================


def canvas_rgba_to_gray(
    rgba: np.ndarray,
) -> np.ndarray:
    """
    Convert Streamlit canvas RGBA image into grayscale.

    Canvas:
        white handwriting
        black background
    """

    rgba = np.asarray(
        rgba,
        dtype=np.float32,
    )

    if (
        rgba.ndim != 3
        or rgba.shape[2] != 4
    ):

        raise ImageProcessingError(
            "Unexpected canvas data format."
        )

    rgb = rgba[..., :3].mean(
        axis=2
    )

    alpha = (
        rgba[..., 3]
        / 255.0
    )

    return np.clip(
        rgb * alpha,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# IMAGE TO GRAYSCALE
# ============================================================


def to_gray_uint8(image) -> np.ndarray:
    """
    Accept PIL image or NumPy array and return
    a 2-D uint8 grayscale image.
    """

    if isinstance(
        image,
        Image.Image,
    ):

        if image.mode in (
            "RGBA",
            "LA",
            "P",
        ):

            rgba = image.convert(
                "RGBA"
            )

            background = Image.new(
                "RGBA",
                rgba.size,
                (
                    255,
                    255,
                    255,
                    255,
                ),
            )

            image = Image.alpha_composite(
                background,
                rgba,
            )

        gray = np.array(
            image.convert("L")
        )

    elif isinstance(
        image,
        np.ndarray,
    ):

        arr = image

        if arr.dtype != np.uint8:

            if (
                arr.size
                and arr.max() <= 1.0
            ):

                arr = arr * 255

            arr = np.clip(
                arr,
                0,
                255,
            ).astype(np.uint8)

        if (
            arr.ndim == 3
            and arr.shape[2] == 1
        ):

            arr = arr[..., 0]

        if (
            arr.ndim == 3
            and arr.shape[2] in (3, 4)
        ):

            gray = cv2.cvtColor(
                arr[..., :3],
                cv2.COLOR_RGB2GRAY,
            )

        elif arr.ndim == 2:

            gray = arr

        else:

            raise ImageProcessingError(
                f"Unsupported image shape "
                f"{arr.shape}."
            )

    else:

        raise ImageProcessingError(
            "Unsupported image type."
        )

    if (
        gray.ndim != 2
        or min(gray.shape) < 8
    ):

        raise ImageProcessingError(
            "The image is too small to process."
        )

    return gray


# ============================================================
# IMAGE INVERSION
# ============================================================


def _needs_inversion(
    gray: np.ndarray,
) -> bool:
    """
    Dark ink on light paper must be inverted because
    the CNN expects light strokes on a dark background.
    """

    border = np.concatenate(
        [
            gray[0, :],
            gray[-1, :],
            gray[:, 0],
            gray[:, -1],
        ]
    )

    return (
        float(
            np.median(border)
        )
        > 127
    )


# ============================================================
# CENTRE IMAGE
# ============================================================


def _center_by_mass(
    img: np.ndarray,
) -> np.ndarray:
    """
    Centre the handwritten character using image moments.
    """

    moments = cv2.moments(
        img
    )

    if moments["m00"] == 0:

        return img

    cx = (
        moments["m10"]
        / moments["m00"]
    )

    cy = (
        moments["m01"]
        / moments["m00"]
    )

    shift = np.float32(
        [
            [
                1,
                0,
                int(
                    round(
                        13.5 - cx
                    )
                ),
            ],
            [
                0,
                1,
                int(
                    round(
                        13.5 - cy
                    )
                ),
            ],
        ]
    )

    return cv2.warpAffine(
        img,
        shift,
        (
            config.IMG_SIZE,
            config.IMG_SIZE,
        ),
    )


# ============================================================
# FULL IMAGE PREPROCESSING
# ============================================================

def _remove_small_components(
    image: np.ndarray,
) -> np.ndarray:
    """
    Remove isolated noise while preserving handwriting strokes.
    """

    binary = (
        image > 0
    ).astype(np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    if num_labels <= 1:
        return image

    cleaned = np.zeros_like(image)

    areas = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]

    if len(areas) == 0:
        return image

    largest_area = int(
        np.max(areas)
    )

    # Keep meaningful components.
    # This allows letters such as i/j to contain a separate dot.
    minimum_area = max(
        MIN_COMPONENT_AREA,
        int(largest_area * 0.015),
    )

    for component in range(
        1,
        num_labels,
    ):
        area = int(
            stats[
                component,
                cv2.CC_STAT_AREA,
            ]
        )

        if area >= minimum_area:
            cleaned[
                labels == component
            ] = image[
                labels == component
            ]

    return cleaned

def preprocess_image_with_steps(
    image,
    dataset: str = "mnist",
):
    """
    Convert a user image into the same general representation
    expected by MNIST/EMNIST CNN models.

    Output:
        final: 28x28 float32 image in range 0-1
        steps: preprocessing images for Streamlit display
    """

    dataset = str(dataset).lower()

    if dataset not in ("mnist", "emnist"):
        raise ImageProcessingError(
            f"Unsupported dataset: {dataset}"
        )

    # =========================================================
    # 1. GRAYSCALE
    # =========================================================

    gray = to_gray_uint8(image)

    steps = {
        "1. Grayscale": gray
    }

    contrast = (
        int(gray.max())
        - int(gray.min())
    )

    if contrast < MIN_CONTRAST:
        raise BlankImageError(
            "The image looks blank. "
            "Please provide a clear handwritten character."
        )

    # =========================================================
    # 2. LIGHT DENOISING
    # =========================================================

    # Median blur preserves edges better than repeatedly
    # smoothing handwriting with Gaussian blur.
    if min(gray.shape) >= 20:
        working = cv2.medianBlur(
            gray,
            3,
        )
    else:
        working = gray.copy()

    # =========================================================
    # 3. CORRECT POLARITY
    # =========================================================

    # CNN training images use bright handwriting on
    # a dark background.
    if _needs_inversion(working):
        working = (
            255 - working
        )

    steps[
        "2. White ink on black"
    ] = working.copy()

    # =========================================================
    # 4. FOREGROUND MASK
    # =========================================================

    # Otsu is used to locate handwriting, but unlike the old
    # pipeline we keep the original grayscale stroke intensity.
    threshold, binary = cv2.threshold(
        working,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    # Very weak thresholds can include background noise.
    threshold = max(
        float(threshold),
        15.0,
    )

    mask = (
        working > threshold
    ).astype(np.uint8)

    foreground_pixels = int(
        mask.sum()
    )

    minimum_pixels = max(
        10,
        int(mask.size * 0.0005),
    )

    if foreground_pixels < minimum_pixels:
        raise BlankImageError(
            "No clear handwriting could be detected."
        )

    # Keep grayscale stroke intensities.
    cleaned = np.where(
        mask > 0,
        working,
        0,
    ).astype(np.uint8)

    cleaned = _remove_small_components(
        cleaned
    )

    steps[
        "3. Background removed"
    ] = cleaned.copy()

    # =========================================================
    # 5. FIND CHARACTER
    # =========================================================

    foreground = (
        cleaned > 0
    ).astype(np.uint8)

    if int(foreground.sum()) < 10:
        raise BlankImageError(
            "No clear character could be detected."
        )

    x, y, w, h = cv2.boundingRect(
        foreground
    )

    # Add a small amount of padding before resizing.
    padding = max(
        2,
        int(
            round(
                0.08 * max(w, h)
            )
        ),
    )

    x1 = max(
        0,
        x - padding,
    )

    y1 = max(
        0,
        y - padding,
    )

    x2 = min(
        cleaned.shape[1],
        x + w + padding,
    )

    y2 = min(
        cleaned.shape[0],
        y + h + padding,
    )

    cropped = cleaned[
        y1:y2,
        x1:x2,
    ]

    steps[
        "4. Character crop"
    ] = cropped.copy()

    if cropped.size == 0:
        raise ImageProcessingError(
            "Character crop failed."
        )

    # =========================================================
    # 6. RESIZE WHILE PRESERVING ASPECT RATIO
    # =========================================================

    crop_h, crop_w = (
        cropped.shape
    )

    scale = (
        DIGIT_BOX
        / float(
            max(
                crop_w,
                crop_h,
            )
        )
    )

    new_w = max(
        1,
        int(
            round(
                crop_w * scale
            )
        ),
    )

    new_h = max(
        1,
        int(
            round(
                crop_h * scale
            )
        ),
    )

    if scale < 1.0:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    resized = cv2.resize(
        cropped,
        (
            new_w,
            new_h,
        ),
        interpolation=interpolation,
    )

    # =========================================================
    # 7. PLACE ON 28x28 CANVAS
    # =========================================================

    canvas = np.zeros(
        (
            config.IMG_SIZE,
            config.IMG_SIZE,
        ),
        dtype=np.uint8,
    )

    x0 = (
        config.IMG_SIZE
        - new_w
    ) // 2

    y0 = (
        config.IMG_SIZE
        - new_h
    ) // 2

    canvas[
        y0:y0 + new_h,
        x0:x0 + new_w,
    ] = resized

    # =========================================================
    # 8. CENTRE BY MASS
    # =========================================================

    # MNIST uses centred characters. Use only a small correction
    # to avoid moving naturally asymmetric letters too far.
    moments = cv2.moments(
        canvas
    )

    if moments["m00"] != 0:

        cx = (
            moments["m10"]
            / moments["m00"]
        )

        cy = (
            moments["m01"]
            / moments["m00"]
        )

        shift_x = int(
            round(
                13.5 - cx
            )
        )

        shift_y = int(
            round(
                13.5 - cy
            )
        )

        # Prevent aggressive movement.
        shift_x = int(
            np.clip(
                shift_x,
                -3,
                3,
            )
        )

        shift_y = int(
            np.clip(
                shift_y,
                -3,
                3,
            )
        )

        transform = np.float32(
            [
                [
                    1,
                    0,
                    shift_x,
                ],
                [
                    0,
                    1,
                    shift_y,
                ],
            ]
        )

        canvas = cv2.warpAffine(
            canvas,
            transform,
            (
                config.IMG_SIZE,
                config.IMG_SIZE,
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    # =========================================================
    # 9. FINAL NORMALISATION
    # =========================================================

    if canvas.max() > 0:

        canvas = (
            canvas.astype(
                np.float32
            )
            / float(
                canvas.max()
            )
            * 255.0
        ).astype(np.uint8)

    final = (
        canvas.astype(
            np.float32
        )
        / 255.0
    )

    steps[
        "5. Final 28x28"
    ] = canvas.copy()

    return final, steps

def preprocess_image(
    image,
) -> np.ndarray:
    """
    Return only the final 28x28 processed image.
    """

    return preprocess_image_with_steps(
        image
    )[0]


# ============================================================
# MODEL INPUT
# ============================================================


def to_model_input(
    processed: np.ndarray,
) -> np.ndarray:
    """
    Convert:

        (28, 28)

    into:

        (1, 28, 28, 1)
    """

    return processed[
        np.newaxis,
        ...,
        np.newaxis,
    ].astype(
        "float32"
    )