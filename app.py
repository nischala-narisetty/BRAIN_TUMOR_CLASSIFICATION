"""
app.py
------
Streamlit frontend for the Brain Tumour Classification project.

Run:
    streamlit run app.py
"""

import json
import os

import numpy as np
import streamlit as st

# ── Page config must be first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="Brain Tumour Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark modern theme ────────────────────────────────────────
st.markdown("""
<style>
    /* Base background */
    .stApp { background-color: #0e1117; color: #e0e0e0; }

    /* Header gradient banner */
    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #2d4a7a;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: #58a6ff; font-size: 2.4rem; margin: 0 0 0.3rem 0; }
    .hero p  { color: #8b949e; font-size: 1rem; margin: 0; }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; }
    .metric-value { font-size: 1.9rem; font-weight: 700; color: #58a6ff; }

    /* Prediction box */
    .pred-box {
        background: linear-gradient(135deg, #1a2744, #0d1b2a);
        border: 2px solid #58a6ff;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .pred-class { font-size: 2rem; font-weight: 800; color: #58a6ff; }
    .pred-conf  { font-size: 1.1rem; color: #c9d1d9; margin-top: 0.4rem; }

    /* Warning/info banners */
    .info-banner {
        background: #1c2d1c;
        border-left: 4px solid #3fb950;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .warn-banner {
        background: #2d1c1c;
        border-left: 4px solid #f85149;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }

    /* Progress bars */
    .stProgress > div > div { background-color: #58a6ff; }

    /* Section headers */
    h2, h3 { color: #c9d1d9; }
    .section-header {
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.4rem;
        color: #58a6ff;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────
ARTIFACTS    = "artifacts"
RESULTS      = "results"
MODEL_PATH   = os.path.join(ARTIFACTS, "best_model.keras")
CLASS_PATH   = os.path.join(ARTIFACTS, "class_names.json")
METRICS_PATH = os.path.join(ARTIFACTS, "test_metrics.json")
HISTORY_PATH = os.path.join(ARTIFACTS, "history.json")

# ── Load model (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    import tensorflow as tf  # import inside cached fn to avoid cold-start cost
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


model       = load_model()
class_names = load_json(CLASS_PATH) or ["glioma", "meningioma", "notumor", "pituitary"]
metrics     = load_json(METRICS_PATH)
history     = load_json(HISTORY_PATH)

model_ready = model is not None

# ── Predict helper ────────────────────────────────────────────────────────
def predict_image(pil_img):
    """Return (class_name, confidence_pct, {cls: pct}) from a PIL image."""
    import numpy as np
    import tensorflow as tf
    IMG_SIZE = 128
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))
    scores = {class_names[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)}
    return class_names[idx], round(float(probs[idx]) * 100, 2), scores

# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Navigation")
    st.markdown("---")

    page = st.radio(
        "Go to",
        ["🏠 Home & Predict", "📊 Training Results", "ℹ️ About Model"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Model Status**")
    if model_ready:
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model not found")
        st.info("Run `python train.py` to train the model first.")

    if metrics:
        st.markdown("**Quick Metrics**")
        st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0)*100:.1f}%")
        st.metric("Test Loss",     f"{metrics.get('test_loss', 0):.4f}")

    st.markdown("---")
    st.caption("⚠️ For educational use only. Not a medical diagnostic tool.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME & PREDICT
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home & Predict":

    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🧠 Brain Tumour Classification</h1>
        <p>Deep Learning · MobileNetV2 · Transfer Learning · 4-Class MRI Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Overview cards
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("Architecture", "MobileNetV2"),
        ("Input Size",   "128 × 128"),
        ("Classes",      "4"),
        ("Method",       "Transfer Learning"),
    ]
    for col, (label, value) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.3rem">{value}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Project overview
    with st.expander("📌 Project Overview", expanded=False):
        st.markdown("""
        This project demonstrates **automated brain tumour detection** from MRI scans
        using a Convolutional Neural Network (CNN) with **Transfer Learning**.

        **Problem Statement**  
        Manual analysis of MRI scans is time-consuming and subject to human error.
        Deep learning can assist radiologists by providing fast, automated
        classification of brain tumour types.

        **Classes Detected**
        | Class | Description |
        |-------|-------------|
        | Glioma | Tumours arising from glial cells |
        | Meningioma | Tumours in the meninges lining |
        | Pituitary | Tumours near the pituitary gland |
        | No Tumour | Healthy brain scan |

        **Methodology**  
        MobileNetV2 pre-trained on ImageNet is used as a feature extractor.
        Only the classification head is trained, keeping training fast and
        feasible on a CPU in a few minutes.
        """)

    st.markdown('<div class="section-header">🔬 MRI Image Prediction</div>',
                unsafe_allow_html=True)

    if not model_ready:
        st.markdown("""
        <div class="warn-banner">
            ⚠️ <strong>Model not found.</strong> Please train the model first:<br>
            <code>python prepare_data.py</code> → <code>python train.py</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        uploaded = st.file_uploader(
            "Upload an MRI brain scan image (JPG / PNG / BMP)",
            type=["jpg", "jpeg", "png", "bmp"],
        )

        if uploaded:
            from PIL import Image
            img = Image.open(uploaded)

            col_img, col_res = st.columns([1, 1], gap="large")

            with col_img:
                st.markdown("**Uploaded Image**")
                st.image(img, caption=uploaded.name, use_container_width=True)

            with col_res:
                with st.spinner("Running inference …"):
                    pred_class, confidence, scores = predict_image(img)

                st.markdown(f"""
                <div class="pred-box">
                    <div class="pred-class">🏷️ {pred_class.upper()}</div>
                    <div class="pred-conf">Confidence: {confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Confidence Scores**")
                # Sort descending
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for cls, score in sorted_scores:
                    color = "#58a6ff" if cls == pred_class else "#8b949e"
                    st.markdown(f"<span style='color:{color}'><b>{cls}</b></span>",
                                unsafe_allow_html=True)
                    st.progress(score / 100.0)
                    st.caption(f"{score:.2f}%")

                # Bar chart
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(5, 3),
                                       facecolor="#161b22")
                ax.set_facecolor("#161b22")
                cls_list   = [s[0] for s in sorted_scores]
                score_list = [s[1] for s in sorted_scores]
                colors = ["#58a6ff" if c == pred_class else "#30363d"
                          for c in cls_list]
                bars = ax.barh(cls_list, score_list, color=colors, edgecolor="none")
                ax.set_xlabel("Confidence (%)", color="#8b949e")
                ax.tick_params(colors="#c9d1d9")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#30363d")
                ax.set_xlim(0, 105)
                for bar, val in zip(bars, score_list):
                    ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                            f"{val:.1f}%", va="center", color="#c9d1d9",
                            fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        else:
            st.info("👆 Upload an MRI image above to get a prediction.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRAINING RESULTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Training Results":
    st.markdown("## 📊 Training Results & Evaluation")

    if not model_ready:
        st.warning("Model has not been trained yet. Run `python train.py` first.")
    else:
        # Metrics summary
        if metrics:
            st.markdown('<div class="section-header">📈 Test Set Metrics</div>',
                        unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Test Accuracy</div>
                    <div class="metric-value">{metrics.get('test_accuracy', 0)*100:.2f}%</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Test Loss</div>
                    <div class="metric-value">{metrics.get('test_loss', 0):.4f}</div>
                </div>""", unsafe_allow_html=True)

            # Per-class metrics table
            if "per_class" in metrics:
                st.markdown('<div class="section-header">🔢 Per-Class Metrics</div>',
                            unsafe_allow_html=True)
                rows = []
                import pandas as pd
                for cls, m in metrics["per_class"].items():
                    rows.append({
                        "Class":      cls,
                        "Precision":  f"{m['precision']:.4f}",
                        "Recall":     f"{m['recall']:.4f}",
                        "F1-Score":   f"{m['f1_score']:.4f}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)

        # Accuracy / Loss plots
        acc_path = os.path.join(RESULTS, "accuracy.png")
        loss_path = os.path.join(RESULTS, "loss.png")
        cm_path   = os.path.join(RESULTS, "confusion_matrix.png")
        cr_path   = os.path.join(RESULTS, "classification_report.txt")

        st.markdown('<div class="section-header">📉 Training Curves</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(acc_path):
                st.image(acc_path, caption="Accuracy", use_container_width=True)
            else:
                st.info("accuracy.png not found. Run training first.")
        with c2:
            if os.path.exists(loss_path):
                st.image(loss_path, caption="Loss", use_container_width=True)
            else:
                st.info("loss.png not found. Run training first.")

        st.markdown('<div class="section-header">🗂️ Confusion Matrix</div>',
                    unsafe_allow_html=True)
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix", use_container_width=False,
                     width=500)
        else:
            st.info("confusion_matrix.png not found.")

        st.markdown('<div class="section-header">📋 Full Classification Report</div>',
                    unsafe_allow_html=True)
        if os.path.exists(cr_path):
            with open(cr_path) as f:
                st.code(f.read(), language="text")
        else:
            st.info("classification_report.txt not found.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT MODEL
# ═══════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About Model":
    st.markdown("## ℹ️ About the Model")

    st.markdown("""
    ### 🧠 Deep Learning
    Deep Learning is a subset of Machine Learning that uses multi-layered
    artificial neural networks to automatically learn hierarchical feature
    representations from raw data — including images.

    ### 🔁 Convolutional Neural Networks (CNNs)
    CNNs apply learnable convolutional filters to extract local patterns
    (edges, textures, shapes) from images, progressively building complex
    representations suitable for classification.

    ### 📱 MobileNetV2
    MobileNetV2 is a lightweight CNN architecture designed for mobile and
    embedded applications.  Key innovations:
    - **Inverted Residuals** — expands low-dimensional features before
      applying depthwise convolutions.
    - **Linear Bottlenecks** — avoids information loss with linear activations
      in the bottleneck layers.
    - **Depthwise Separable Convolutions** — reduce parameters and FLOPs
      dramatically compared to standard convolutions.

    ### 🔄 Transfer Learning
    Instead of training from scratch, we initialise MobileNetV2 with weights
    learned on ImageNet (1.4M images, 1000 classes).  We then:
    1. **Freeze** the backbone — keep its feature-extraction weights fixed.
    2. **Replace the head** — add a custom dense classifier for our 4 classes.
    3. **Train only the head** — fast and effective, even on a CPU.

    This approach achieves high accuracy with far less data and compute.
    """)

    st.markdown("---")
    st.markdown("""
    ### 🏗️ Architecture Summary
    ```
    Input (128×128×3)
        └─► MobileNetV2 backbone (frozen, ImageNet weights)
                └─► GlobalAveragePooling2D
                        └─► BatchNormalization
                                └─► Dropout (0.4)
                                        └─► Dense 128, ReLU
                                                └─► Dropout (0.3)
                                                        └─► Dense 4, Softmax
    Output: probabilities for [glioma, meningioma, notumor, pituitary]
    ```

    ### ⚙️ Training Configuration
    | Parameter | Value |
    |-----------|-------|
    | Optimizer | Adam (lr=1e-3) |
    | Loss | Sparse Categorical Cross-Entropy |
    | Batch Size | 8 |
    | Epochs | 5 (with EarlyStopping) |
    | Input Size | 128 × 128 |
    | Backbone | Frozen MobileNetV2 |

    ### ⚠️ Medical Disclaimer
    > This tool is **strictly for educational and research purposes**.
    > It is **NOT** a certified medical device and must **NOT** be used for
    > actual medical diagnosis or treatment decisions.
    > Always consult a qualified medical professional for health concerns.
    """)

    st.markdown("---")
    st.markdown("""
    ### 📚 References
    - Sandler et al. (2018) — *MobileNetV2: Inverted Residuals and Linear Bottlenecks*
    - Chollet (2021) — *Deep Learning with Python*, Manning
    - [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
    """)
