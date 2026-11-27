# 🧠 BRAIN_TUMOR_CLASSIFICATION

> **Deep Learning · MobileNetV2 · Transfer Learning · 4-Class MRI Analysis**

---

## 📌 Project Description

**BRAIN_TUMOR_CLASSIFICATION** is a complete, end-to-end deep learning project
that classifies brain MRI scans into four categories:

| Label | Description |
|-------|-------------|
| **Glioma** | Tumour arising from glial cells |
| **Meningioma** | Tumour in the meninges (brain lining) |
| **Pituitary** | Tumour near the pituitary gland |
| **No Tumour** | Healthy / normal brain scan |

The model is built on **MobileNetV2** with **Transfer Learning** from ImageNet weights,
making it fast enough to train on a CPU in a few minutes.

---

## 🧩 Problem Statement

Manual interpretation of MRI scans is slow, expensive, and can be inconsistent.
Automated deep learning models can assist radiologists by providing rapid,
preliminary classifications — reducing workload and potentially improving
early-detection rates.

---

## 🎯 Objectives

1. Build a lightweight CNN classifier for brain tumour MRI images.
2. Use Transfer Learning to maximise accuracy with limited data and compute.
3. Provide a working Streamlit web UI for interactive predictions.
4. Output all standard evaluation metrics (accuracy, loss, confusion matrix, F1).
5. Ensure the entire pipeline runs on a CPU within a few hours.

---

## 🧠 Deep Learning

Deep Learning trains multi-layered neural networks on raw data.
Each layer learns progressively more abstract representations:
pixels → edges → shapes → semantic concepts.

## 🔁 Convolutional Neural Networks (CNNs)

CNNs apply learnable filters across an image to detect spatial patterns —
edges, textures, and high-level features — using far fewer parameters than
fully connected networks of equivalent capacity.

## 📱 MobileNetV2

MobileNetV2 is a highly efficient CNN designed for resource-constrained devices:
- **Inverted Residuals** — expands channels for richer feature extraction.
- **Depthwise Separable Convolutions** — reduce multiply-add operations ~8×.
- **Linear Bottlenecks** — preserve information in low-dimensional spaces.

## 🔄 Transfer Learning

Pre-trained MobileNetV2 weights (ImageNet) encode rich visual features.
We **freeze the backbone** and train only the custom classification head:
- Faster convergence.
- Needs far fewer labelled examples.
- Achieves strong accuracy even on a CPU.

---

## 📂 Dataset

This project uses the **Brain Tumour Classification MRI** dataset from Kaggle:

🔗(https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)

### Expected folder structure after download

```
dataset/
    Training/
        glioma/
        meningioma/
        pituitary/
        notumor/
    Testing/
        glioma/
        meningioma/
        pituitary/
        notumor/
```

### How to download

1. Create a free Kaggle account: https://www.kaggle.com
2. Go to the dataset page linked above.
3. Click **Download**.
4. Extract the zip and place the resulting folders inside `dataset/` in this project root.

---

## 🚀 Setup & Installation (Windows PowerShell)

### 1 · Clone the repository

```powershell
git clone https://github.com/<your-username>/BRAIN_TUMOR_CLASSIFICATION.git
cd BRAIN_TUMOR_CLASSIFICATION
```

### 2 · Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3 · Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ How to Run

### Step 1 — Prepare the small dataset

```powershell
python prepare_data.py
```

Outputs a balanced 400-image subset to `data_small/`.

Optional flags:
```powershell
python prepare_data.py --src dataset --dst data_small --max_per_class 100 --seed 42
```

### Step 2 — Train the model

```powershell
python train.py
```

Optional flags:
```powershell
python train.py --data_dir data_small --epochs 5 --batch_size 8 --img_size 128
```

Saves:
- `artifacts/best_model.keras`
- `artifacts/class_names.json`
- `artifacts/test_metrics.json`
- `results/accuracy.png`
- `results/loss.png`
- `results/confusion_matrix.png`
- `results/classification_report.txt`

### Step 3 — Predict a single image

```powershell
python predict.py --image path\to\mri_scan.jpg
```

### Step 4 — Launch the Streamlit web app

```powershell
streamlit run app.py
```

Then open: http://localhost:8501

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Proportion of correctly classified samples |
| **Loss** | Sparse Categorical Cross-Entropy value |
| **Precision** | Of all predicted positives, how many are truly positive |
| **Recall** | Of all actual positives, how many were detected |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Shows correct and misclassified counts per class |

---

## 📁 Project Structure

```
BRAIN_TUMOR_CLASSIFICATION/
│
├── app.py               ← Streamlit web application
├── train.py             ← Model training script
├── predict.py           ← Single-image prediction script
├── prepare_data.py      ← Dataset preparation & splitting
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
├── .gitignore
│
├── dataset/             ← (not committed) Original Kaggle dataset
├── data_small/          ← (not committed) Prepared balanced subset
│
├── artifacts/           ← Saved model + metadata (model excluded from git)
│   ├── best_model.keras
│   ├── class_names.json
│   ├── test_metrics.json
│   └── history.json
│
└── results/             ← Plots & reports
    ├── accuracy.png
    ├── loss.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

---

## 🐙 GitHub Setup

```powershell
# Initialise (first time only)
git init
git remote add origin https://github.com/<your-username>/BRAIN_TUMOR_CLASSIFICATION.git

# Stage, commit, and push
git add .
git commit -m "Initial commit: Brain Tumour Classification with MobileNetV2"
git branch -M main
git push -u origin main
```

> **Note:** `dataset/`, `data_small/`, and `best_model.keras` are excluded
> via `.gitignore`. Only source code, results plots, metrics JSON, and
> `class_names.json` are committed.

---

## ⚠️ Limitations

- Trained on a small balanced subset (~400 images). Accuracy is indicative,
  not production-grade.
- No GPU required, but training with more data or fine-tuning the backbone
  will be significantly faster on a GPU.
- Dataset quality and class imbalance in the original data may affect metrics.
- No data augmentation is applied in this lightweight version (can be added).

---

## 🏥 Medical Disclaimer

> **This project is strictly for educational and research purposes.**
> It is **NOT** a certified medical device and must **NOT** be used to make
> real medical diagnoses or treatment decisions.
> Always consult a qualified, licensed medical professional for any
> health-related concerns.

---

## 📚 References

- Sandler et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR.
- Chollet, F. (2021). *Deep Learning with Python.* Manning.
- Kaggle Dataset: [https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)
- TensorFlow Documentation: https://www.tensorflow.org/api_docs

---

*Built with ❤️ using Python · TensorFlow · Streamlit*
