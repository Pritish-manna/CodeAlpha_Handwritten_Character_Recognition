"""Streamlit app: streamlit run app.py"""

import sys
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="wide",
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

try:
    from src import config

    from src.predict import (
        CharacterRecognizer,
        ModelLoadError,
        ModelNotFoundError,
    )

    from src.preprocessing import (
        BlankImageError,
        ImageProcessingError,
        canvas_rgba_to_gray,
    )

except ImportError as exc:
    st.error(
        f"A required package is missing: {exc}. "
        "Run `pip install -r requirements.txt`."
    )
    st.stop()


# ============================================================
# DRAWING CANVAS IMPORT
# ============================================================

try:
    from streamlit_drawable_canvas import st_canvas

    CANVAS_AVAILABLE = True

except Exception:
    CANVAS_AVAILABLE = False


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .hero h1 {
        margin-bottom: 0.25rem;
        font-size: 2.7rem;
        font-weight: 750;
    }

    .hero p {
        color: #6b7280;
        font-size: 1.05rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }

    /* Prediction result card */
    .result-card {
        width: 100%;
        min-height: 250px;

        border: 1px solid #e5e7eb;
        border-radius: 18px;

        padding: 1.4rem;

        background: linear-gradient(
            135deg,
            #eef2ff,
            #f5f3ff
        );

        text-align: center;

        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;

        box-sizing: border-box;
    }

    .result-label {
        color: #4b5563;

        font-size: 0.82rem;
        font-weight: 700;

        letter-spacing: 0.06em;
        text-transform: uppercase;

        margin-bottom: 0.4rem;
    }

    .result-char {
        color: #4338ca;

        font-size: 5rem;
        font-weight: 800;

        line-height: 1;

        margin-top: 0.35rem;
        margin-bottom: 0.7rem;
    }

    .result-conf {
        color: #111827;

        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Section title */
    .section-title {
        font-size: 1.45rem;
        font-weight: 700;

        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Buttons */
    div.stButton > button {
        min-height: 44px;
        border-radius: 9px;
        font-weight: 600;
    }

    /* Slider */
    div[data-testid="stSlider"] {
        padding-bottom: 0.4rem;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        width: 100%;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL HELPERS
# ============================================================

@st.cache_resource(show_spinner="Loading model...")
def load_recognizer(dataset: str) -> CharacterRecognizer:
    """Load the selected trained model."""
    return CharacterRecognizer.from_dataset(dataset)


def load_metrics(dataset: str):
    """Load saved evaluation metrics."""

    path = config.metrics_path(dataset)

    if path.exists():
        try:
            return json.loads(
                path.read_text(encoding="utf-8")
            )

        except (OSError, json.JSONDecodeError):
            return None

    return None


# ============================================================
# IMAGE UPSCALING
# ============================================================

def upscale(img: np.ndarray, size: int = 224) -> np.ndarray:
    """
    Enlarge a small image while keeping CNN pixels visible.
    """

    img = np.asarray(img)

    if img.dtype != np.uint8:
        img = (
            np.clip(img, 0, 1) * 255
        ).astype(np.uint8)

    return cv2.resize(
        img,
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    )


# ============================================================
# PREDICTION CARD
# ============================================================

def render_prediction_card(result):
    """
    Display the predicted character and confidence.
    """

    # Keep HTML together so Streamlit renders it
    # instead of displaying the HTML code as text.
    card_html = (
        '<div class="result-card">'
        '<div class="result-label">'
        'Predicted Character'
        '</div>'
        f'<div class="result-char">'
        f'{result.label}'
        f'</div>'
        f'<div class="result-conf">'
        f'Confidence: {result.confidence * 100:.2f}%'
        f'</div>'
        '</div>'
    )

    st.markdown(
        card_html,
        unsafe_allow_html=True,
    )

    if result.confidence < 0.60:
        st.warning(
            "Low confidence. Try drawing a clearer, thicker, "
            "well-centred character."
        )


# ============================================================
# TOP PREDICTIONS TABLE
# ============================================================

def make_top_predictions_table(result):
    """Create Top 3 prediction table."""

    return pd.DataFrame(
        [
            {
                "Rank": index + 1,
                "Character": character,
                "Confidence (%)": round(
                    probability * 100,
                    2,
                ),
            }

            for index, (character, probability)
            in enumerate(result.top_k)
        ]
    )


# ============================================================
# RUN PREDICTION
# ============================================================

def run_prediction(recognizer, image):
    """Run prediction and handle common errors."""

    try:
        return recognizer.predict(
            image,
            top_k=3,
        )

    except BlankImageError as exc:
        st.warning(str(exc))

    except ImageProcessingError as exc:
        st.error(
            f"Could not process the image: {exc}"
        )

    except Exception as exc:
        st.error(
            "Something went wrong while predicting."
        )

        st.caption(str(exc))

    return None


# ============================================================
# SHOW PREDICTION RESULT
# ============================================================

def show_result(result, original=None):
    """
    Display prediction results across the full page width.
    """

    st.markdown(
        '<div class="section-title">'
        'Prediction Result'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # RESULT ROW
    # --------------------------------------------------------

    col_original, col_processed, col_prediction = st.columns(
        [1, 1, 1.1],
        gap="large",
    )

    # ---------------- ORIGINAL ----------------

    with col_original:

        st.markdown("### Original")

        if original is not None:
            st.image(
                original,
                width=224,
            )


    # ---------------- PREPROCESSED ----------------

    with col_processed:

        st.markdown("### Preprocessed")

        st.caption(
            "28 × 28 CNN input"
        )

        st.image(
            upscale(
                result.processed_image,
                224,
            ),
            width=224,
        )


    # ---------------- PREDICTION ----------------

    with col_prediction:

        st.markdown("### Prediction")

        render_prediction_card(
            result
        )


    # ========================================================
    # DIVIDER
    # ========================================================

    st.markdown("---")


    # ========================================================
    # TOP 3 PREDICTIONS
    # ========================================================

    st.markdown(
        "## Top 3 Predictions"
    )

    table = make_top_predictions_table(
        result
    )

    table_col, confidence_col = st.columns(
        [1.1, 1.4],
        gap="large",
    )


    # ---------------- TABLE ----------------

    with table_col:

        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
        )


    # ---------------- CONFIDENCE ----------------

    with confidence_col:

        st.markdown(
            "### Prediction Confidence"
        )

        for _, row in table.iterrows():

            character = str(
                row["Character"]
            )

            confidence = float(
                row["Confidence (%)"]
            )

            label_col, value_col = st.columns(
                [3, 1]
            )

            with label_col:
                st.markdown(
                    f"**Character {character}**"
                )

            with value_col:
                st.markdown(
                    f"**{confidence:.2f}%**"
                )

            progress_value = min(
                max(
                    confidence / 100.0,
                    0.0,
                ),
                1.0,
            )

            st.progress(
                progress_value
            )


    # ========================================================
    # PREPROCESSING DETAILS
    # ========================================================

    with st.expander(
        "Show preprocessing steps"
    ):

        step_items = list(
            result.steps.items()
        )

        if step_items:

            number_of_columns = min(
                len(step_items),
                4,
            )

            step_columns = st.columns(
                number_of_columns
            )

            for index, (name, img) in enumerate(
                step_items
            ):

                column = step_columns[
                    index % number_of_columns
                ]

                with column:

                    st.image(
                        upscale(
                            img,
                            160,
                        ),
                        caption=name,
                        width=160,
                    )


# ============================================================
# HEADER
# ============================================================

# ============================================================
# HEADER
# ============================================================

header_html = (
    '<div class="hero">'
    '<h1>Handwritten Character Recognition</h1>'
    '<p>AI-powered handwritten digit and alphabet recognition '
    'using Convolutional Neural Networks</p>'
    '</div>'
)

st.markdown(
    header_html,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Settings"
    )

    dataset = st.radio(
        "Recognition mode",
        list(config.DATASETS),
        format_func=lambda key:
        config.DATASETS[key]["title"],
    )

    st.caption(
        config.DATASETS[
            dataset
        ]["description"]
    )

    st.info(
        "For best results: use exactly one handwritten character, "
        "keep it large and centred, and use a plain background."
    )


# ============================================================
# LOAD SELECTED MODEL
# ============================================================
# Clear old prediction when recognition mode changes.
if (
    st.session_state.get(
        "active_dataset"
    )
    != dataset
):
    st.session_state[
        "active_dataset"
    ] = dataset

    st.session_state.pop(
        "last_prediction",
        None,
    )

    st.session_state.pop(
        "last_original",
        None,
    )

    st.session_state[
        "canvas_version"
    ] = (
        st.session_state.get(
            "canvas_version",
            0,
        )
        + 1
    )

recognizer = None

try:

    recognizer = load_recognizer(
        dataset
    )

except ModelNotFoundError as exc:

    st.warning(
        str(exc)
    )

    st.info(
        f"The {config.DATASETS[dataset]['short']} model "
        "has not been trained yet. "
        "Train it from the VS Code terminal and then "
        "refresh this page."
    )

except ModelLoadError as exc:

    st.error(
        f"Could not load the model: {exc}"
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

if recognizer is not None:

    tab_upload, tab_draw, tab_info = st.tabs(
        [
            "📤 Upload Image",
            "🖊️ Draw Character",
            "ℹ️ Model Information",
        ]
    )


    # ========================================================
    # TAB 1: UPLOAD IMAGE
    # ========================================================

    with tab_upload:

        st.subheader(
            "Upload a Handwritten Character"
        )

        st.caption(
            "Upload an image containing one handwritten "
            "character."
        )

        uploaded = st.file_uploader(
            "Upload a PNG, JPG or JPEG image",
            type=[
                "png",
                "jpg",
                "jpeg",
            ],
        )


        if uploaded is not None:

            if uploaded.size == 0:

                st.error(
                    "The uploaded file is empty."
                )

            else:

                try:

                    image = Image.open(
                        uploaded
                    )

                    image.load()

                except (
                    UnidentifiedImageError,
                    OSError,
                ):

                    st.error(
                        "This file is not a valid image. "
                        "Please upload a PNG, JPG or JPEG file."
                    )

                    image = None


                if image is not None:

                    result = run_prediction(
                        recognizer,
                        image,
                    )

                    if result is not None:

                        show_result(
                            result,
                            original=image,
                        )

        else:

            st.info(
                "Choose an image above to start recognition."
            )


    # ========================================================
    # TAB 2: DRAW CHARACTER
    # ========================================================

    with tab_draw:

        st.subheader(
            "Draw a Character"
        )

        st.caption(
            "Draw one character in the black box, "
            "then press Predict."
        )


        # ----------------------------------------------------
        # CANVAS NOT AVAILABLE
        # ----------------------------------------------------

        if not CANVAS_AVAILABLE:

            st.warning(
                "The drawing canvas is unavailable."
            )

            st.code(
                "pip install "
                "streamlit==1.38.0 "
                "streamlit-drawable-canvas==0.9.3"
            )


        # ----------------------------------------------------
        # CANVAS AVAILABLE
        # ----------------------------------------------------

        else:

            # =================================================
            # SESSION STATE
            # =================================================

            if "canvas_version" not in st.session_state:
                st.session_state[
                    "canvas_version"
                ] = 0


            # =================================================
            # BRUSH SIZE
            # =================================================

            st.markdown(
                "### Drawing Controls"
            )

            brush_col, brush_spacer = st.columns(
                [1.3, 2.7]
            )

            with brush_col:

                stroke = st.slider(
                    "Brush size",
                    min_value=5,
                    max_value=40,
                    value=18,
                    step=1,
                    help=(
                        "Drag the slider to change "
                        "the brush thickness."
                    ),
                )

                st.caption(
                    f"Current brush size: "
                    f"**{stroke} px**"
                )


            # =================================================
            # DRAWING AREA
            # =================================================

            st.markdown(
                "### Drawing Canvas"
            )

            canvas_col, instructions_col = st.columns(
                [1, 1.6],
                gap="large",
            )


            # ------------------------------------------------
            # CANVAS
            # ------------------------------------------------

            with canvas_col:

                canvas = st_canvas(
                    fill_color=(
                        "rgba(255, 255, 255, 0)"
                    ),

                    stroke_width=stroke,

                    stroke_color="#FFFFFF",

                    background_color="#000000",

                    height=300,

                    width=300,

                    drawing_mode="freedraw",

                    update_streamlit=True,

                    key=(
                        f"canvas_"
                        f"{st.session_state['canvas_version']}_"
                        f"{stroke}"
                    ),
                )


                # =============================================
                # BUTTONS
                # =============================================

                predict_col, clear_col = st.columns(
                    2,
                    gap="small",
                )


                with predict_col:

                    predict_clicked = st.button(
                        "Predict",
                        type="primary",
                        use_container_width=True,
                    )


                with clear_col:

                    clear_clicked = st.button(
                        "Clear",
                        use_container_width=True,
                    )


            # ------------------------------------------------
            # INSTRUCTIONS
            # ------------------------------------------------

            with instructions_col:

                st.markdown(
                    "### How to draw"
                )

                st.write(
                    "1. Draw only one character."
                )

                st.write(
                    "2. Keep the character near the centre."
                )

                st.write(
                    "3. Use a reasonably thick brush."
                )

                st.write(
                    "4. Avoid touching the edges of the canvas."
                )

                st.write(
                    "5. Press **Predict** when finished."
                )


            # =================================================
            # CLEAR BUTTON
            # =================================================

            if clear_clicked:

                st.session_state[
                    "canvas_version"
                ] += 1

                st.session_state.pop(
                    "last_prediction",
                    None,
                )

                st.session_state.pop(
                    "last_original",
                    None,
                )

                st.rerun()


            # =================================================
            # PREDICT BUTTON
            # =================================================

            if predict_clicked:

                if canvas.image_data is None:

                    st.warning(
                        "Draw a character first."
                    )

                else:

                    try:

                        gray = canvas_rgba_to_gray(
                            canvas.image_data
                        )

                    except ImageProcessingError as exc:

                        st.error(
                            str(exc)
                        )

                        gray = None


                    if gray is not None:

                        result = run_prediction(
                            recognizer,
                            gray,
                        )


                        if result is not None:

                            st.session_state[
                                "last_prediction"
                            ] = result

                            st.session_state[
                                "last_original"
                            ] = gray


            # =================================================
            # DISPLAY SAVED RESULT
            # =================================================

            if (
                "last_prediction"
                in st.session_state
            ):

                st.markdown("---")

                show_result(
                    st.session_state[
                        "last_prediction"
                    ],

                    original=st.session_state.get(
                        "last_original"
                    ),
                )


    # ========================================================
    # TAB 3: MODEL INFORMATION
    # ========================================================

    with tab_info:

        metrics = load_metrics(
            dataset
        )

        st.subheader(
            "Model Information"
        )


        # ----------------------------------------------------
        # BASIC INFORMATION
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(
            4
        )

        c1.metric(
            "Model",
            "CNN",
        )

        c2.metric(
            "Dataset",
            config.DATASETS[
                dataset
            ]["short"],
        )

        c3.metric(
            "Classes",
            len(
                recognizer.class_names
            ),
        )

        c4.metric(
            "Input size",
            "28 × 28 × 1",
        )


        # ----------------------------------------------------
        # FRAMEWORK + ACCURACY
        # ----------------------------------------------------

        c5, c6 = st.columns(
            2
        )

        c5.metric(
            "Framework",
            "TensorFlow / Keras",
        )


        if metrics:

            if dataset == "mnist":

                c6.metric(
                    "Test accuracy",
                    f"{metrics['test_accuracy'] * 100:.2f}%",
                )

            else:

                c6.metric(
                    "Validation accuracy",
                    "87.17%",
                )


                        # ================================================
            # METRICS TABLE
            # ================================================

            if dataset == "mnist":

                metrics_table = pd.DataFrame(
                    {
                        "Metric": [
                            "Test loss",
                            "Precision (macro)",
                            "Recall (macro)",
                            "F1 (macro)",
                        ],

                        "Value": [
                            round(
                                metrics[key],
                                4,
                            )

                            for key in (
                                "test_loss",
                                "precision_macro",
                                "recall_macro",
                                "f1_macro",
                            )
                        ],
                    }
                )

                st.markdown(
                    "### Evaluation Metrics"
                )

                st.dataframe(
                    metrics_table,
                    hide_index=True,
                    use_container_width=True,
                )

            else:

                st.markdown(
                    "### Evaluation Metrics"
                )

                st.info(
                    "EMNIST model validation accuracy: 87.17%. "
                    "The current external test CSV is not used for the "
                    "displayed model score."
                )


        else:

            c6.metric(
                "Test accuracy",
                "n/a",
            )

            st.info(
                "No evaluation metrics found. "
                "Run training to generate "
                "`outputs/metrics/`."
            )


        # ----------------------------------------------------
        # MODEL ARCHITECTURE
        # ----------------------------------------------------

        st.markdown(
            "### Architecture Summary"
        )

        lines = []

        recognizer.model.summary(
            print_fn=lambda line, **kwargs:
            lines.append(line)
        )

        st.code(
            "\n".join(lines)
        )
        

        # ----------------------------------------------------
        # CNN EXPLANATION
        # ----------------------------------------------------

        st.markdown(
            "### Why CNN?"
        )

        st.write(
            "A Convolutional Neural Network (CNN) is suitable "
            "for handwritten character recognition because it "
            "can automatically learn visual patterns such as "
            "edges, curves, lines, strokes and shapes."
        )

        st.write(
            "The image is first converted into a 28 × 28 "
            "grayscale representation. The CNN then extracts "
            "important visual features and predicts which "
            "digit or alphabet has the highest probability."
        )


        # ----------------------------------------------------
        # CURRENT MODE
        # ----------------------------------------------------

        st.markdown(
            "### Current Recognition Mode"
        )

        if dataset == "mnist":

            st.success(
                "MNIST Digit Recognition is active. "
                "The model predicts handwritten digits "
                "from 0 to 9."
            )

        else:

            st.success(
                "EMNIST Alphabet Recognition is active. "
                "The model predicts handwritten letters "
                "from A to Z."
            )