# Handwritten Character Recognition Using Deep Learning

AI-powered handwritten **digit** (MNIST) and **alphabet** (EMNIST Letters) recognition using a Convolutional Neural Network, with a Streamlit web app for image upload and mouse drawing.

## Project overview
The system takes an image of a single handwritten character, preprocesses it into the 28x28 format the CNN was trained on, and returns the predicted character, its confidence and the Top 3 most likely candidates.

## Objective
Build an end-to-end, reproducible deep-learning project: data loading, preprocessing, EDA, CNN training, full evaluation (accuracy, precision, recall, F1, confusion matrix) and a polished interface.

## Features
- Two modes: Digit Recognition (MNIST) and Alphabet Recognition (EMNIST Letters A-Z)
- Image upload (PNG/JPG/JPEG) with original vs. preprocessed view
- Drawing canvas with Clear and Predict buttons
- Prediction, confidence score, Top 3 table and probability bar chart
- Robust preprocessing: grayscale, automatic inversion, background removal, crop, aspect-ratio-preserving resize, centring
- Friendly error messages (invalid/empty/blank image, missing model, missing packages, dataset download failure)
- Model information tab (test accuracy read from the saved evaluation, architecture summary, "Why CNN?")
- Modular code: training is separate from the app; models are cached in Streamlit

## Technology stack
Python 3.11 / 3.12, TensorFlow/Keras, NumPy, Pandas, OpenCV, Pillow, Matplotlib, scikit-learn, Streamlit.

## Dataset information
| | MNIST | EMNIST Letters |
|---|---|---|
| Source | Keras (`tf.keras.datasets.mnist`) | NIST EMNIST via the `emnist` package (or CSV files in `data/`) |
| Classes | 10 (0-9) | 26 (A-Z, upper/lower case merged) |
| Training samples | 60,000 | 124,800 |
| Test samples | 10,000 | 20,800 |
| Image size | 28x28, grayscale | 28x28, grayscale |

**Label mapping.** MNIST labels are the digits themselves. EMNIST Letters raw labels are 1-26; the loader shifts them to 0-25 so `0=A ... 25=Z`.
**Orientation.** EMNIST images are stored transposed; `fix_emnist_orientation` transposes them. After training, check `outputs/figures/emnist_orientation_check.png` (bottom row must be upright).
Training uses a stratified 90/10 train/validation split; the test set is only used for the final evaluation.

## CNN architecture
```
Input 28x28x1
-> [Conv3x3-BN-ReLU] x2 (32) -> MaxPool -> Dropout(0.25)
-> [Conv3x3-BN-ReLU] x2 (64) -> MaxPool -> Dropout(0.25)
-> Conv3x3-BN-ReLU (128)
-> Flatten -> Dense(128, ReLU) -> Dropout(0.5) -> Dense(classes, Softmax)
```
Light augmentation (rotation, shift, zoom) is applied inside the model during training only. Loss: sparse categorical cross-entropy; optimiser: Adam (1e-3) with EarlyStopping, ModelCheckpoint (best validation loss) and ReduceLROnPlateau.

## Project structure
```
handwritten-character-recognition/
├── app.py                  # Streamlit app
├── requirements.txt
├── data/README.md          # dataset notes (data is downloaded automatically)
├── models/                 # *_cnn.keras and *_labels.json (created by training)
├── notebooks/              # mnist_training.ipynb, emnist_training.ipynb
├── src/
│   ├── config.py           # paths and dataset metadata
│   ├── preprocessing.py    # dataset loading + upload/canvas preprocessing
│   ├── model.py            # CNN, compile, callbacks
│   ├── train.py            # training CLI
│   ├── evaluate.py         # metrics, confusion matrix, example predictions
│   ├── predict.py          # reusable CharacterRecognizer
│   └── visualize.py        # plotting helpers
├── outputs/figures/        # EDA, curves, confusion matrices, examples
├── outputs/metrics/        # metrics JSON, classification reports, history CSV
└── sample_images/          # exported real test images + your own images
```

## Installation (Windows)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
(macOS/Linux: `source .venv/bin/activate`.)

## Training
Run from the project root. The models are **not** included; train them once:
```bash
python -m src.train --dataset mnist
python -m src.train --dataset emnist
```
Options: `--epochs 30 --batch-size 128 --lr 0.001 --seed 42 --no-augment --no-eda`. EMNIST downloads a ~535 MB archive on first use. If the download fails, place `emnist-letters-train.csv` and `emnist-letters-test.csv` in `data/` (see `data/README.md`).

Training saves `models/<dataset>_cnn.keras` (best epoch), `models/<dataset>_labels.json`, all figures in `outputs/figures/`, and metrics in `outputs/metrics/`. Re-evaluate a saved model with `python -m src.evaluate --dataset mnist`. The notebooks perform the same workflow step by step.

## Running the Streamlit app
```bash
streamlit run app.py
```
### Using image upload
1. Choose Digit or Alphabet mode in the sidebar.
2. Open **Upload Image** and select a PNG/JPG/JPEG with one character (dark ink on plain paper works best; crop close to the character).
3. View the original and preprocessed image, the predicted character, confidence and Top 3.

### Using the drawing canvas
Open **Draw Character**, draw with the mouse, press **Predict**; **Clear** resets the canvas. The canvas needs `streamlit-drawable-canvas==0.9.3`, which is why Streamlit is pinned to 1.38.0. If the canvas package cannot load, the app shows a notice and the upload tab keeps working.

## Model evaluation
`python -m src.train` writes real, measured results on the unseen test set to `outputs/metrics/<dataset>_metrics.json` (accuracy, loss, macro/weighted precision, recall, F1), `<dataset>_classification_report.txt`, and figures: training/validation accuracy and loss, row-normalised confusion matrix, correctly classified samples and the most confident misclassifications (Actual / Predicted / Confidence). Add your own measured numbers to this section after training; none are pre-filled. Typical CNNs of this size reach roughly 99% on MNIST and low-to-mid 90s on EMNIST Letters, but your run determines the actual values.

## Example predictions
Output format (illustrative, not a measured result):
```
Prediction: A    Confidence: 96.7%
Top Predictions:  A - 96.7%   H - 2.1%   R - 0.6%
```
After training, real test images are exported to `sample_images/<dataset>/` for trying the app.

## Limitations
- Recognises one isolated character at a time; no words or sentences yet.
- EMNIST Letters merges upper/lower case, so case cannot be predicted; look-alikes (I/L, Q/G, U/V) are confused more often.
- Photos with heavy shadows, several characters, or very thin pencil strokes can degrade preprocessing.
- Trained on US-style NIST handwriting; other styles may reduce accuracy.

## Future improvements
Extend to full word/sentence recognition:

```
Handwritten Image -> Character Segmentation -> CNN Feature Extraction
-> Sequence Modeling (CRNN / BiLSTM / Transformer) -> CTC Decoding -> Word/Sentence
```
`src/model.py: build_backbone` already isolates the convolutional feature extractor so it can become the CNN part of a CRNN. Other ideas: EMNIST ByClass (62 classes with case), test-time augmentation, model quantisation for mobile.
