# ⚙️ MachineGuard AI — Predictive Maintenance System

MachineGuard AI is a machine failure prediction platform built with Machine Learning and Streamlit. It helps identify potential equipment failures before they happen by analyzing industrial sensor readings and generating real-time risk assessments.

The application combines **Logistic Regression** and **Random Forest** classifiers to predict machine failure probability, provide model comparisons, visualize performance metrics, and support both single-machine and batch predictions.

---

🚀 **Live Demo:** https://machinefailurepredictionsystem-26.streamlit.app/

## 📸 Application Preview

<img width="2876" height="1704" alt="image" src="https://github.com/user-attachments/assets/a07b5d9c-58e1-42cf-871a-ec4d5e09113c" />
<img width="2876" height="1214" alt="image" src="https://github.com/user-attachments/assets/983ce1b5-0ab1-4954-9c28-9cd6c9b8ee8e" />

---

## 🚀 Features

### 🔮 Real-Time Failure Prediction
* Predict machine failure from live sensor inputs
* Dual-model prediction using Logistic Regression and Random Forest
* Combined risk assessment score
* Model agreement analysis

### 📊 Model Analytics Dashboard
* Confusion matrices for both models
* Dual ROC Curve with AUC comparison
* Feature importance — LR coefficients and RF importances side by side
* Full classification reports

### 🗂️ Interactive Data Explorer
* Explore all 10,000 machine telemetry records
* Filter by machine type and failure status
* Distribution analysis
* Correlation heatmaps
* Raw dataset inspection

### 📤 Batch Scoring
* Upload CSV files containing multiple machine records
* Generate predictions for hundreds of machines simultaneously
* Download enriched prediction results with both model outputs

### 🎨 Modern Dashboard UI
* Fully customized Streamlit interface
* Responsive layout
* Interactive Plotly visualizations
* Industrial-themed design system

---

## 🧠 Machine Learning Pipeline

### Data Preprocessing
* Missing value imputation with column medians
* Label encoding for machine type (L / M / H)
* Feature scaling using StandardScaler (for LR)
* Removal of data leakage columns (TWF, HDF, PWF, OSF, RNF)
* Stratified train/test split (80/20) to preserve class distribution

---

### Handling Class Imbalance

The dataset is highly imbalanced — only **~3.4%** of records are failures. A naive model that always predicts "No Failure" would score **96.6% accuracy** while catching zero real breakdowns.

To address this:
- **Logistic Regression** uses `class_weight="balanced"` to penalize missed failures
- **Random Forest** uses `class_weight="balanced_subsample"` for per-tree rebalancing
- The decision threshold is lowered to **0.30** (vs. the default 0.50) — in predictive maintenance, missing a real failure is far more costly than a false alarm

---

### Why LR Accuracy Appears Lower (~67%)

LR's lower raw accuracy is **intentional and correct**. With the threshold set to 0.30, the model flags more borderline cases as potential failures. This trades some "No Failure" correct predictions for dramatically better recall on actual failures — which is exactly the right behaviour for maintenance systems. Raw accuracy is a misleading metric on imbalanced data; **failure recall and F1 score** are what matter.

The notebook baseline (`Machine_Failure_Prediction.ipynb`) used the default 0.50 threshold without class-imbalance handling, which gave LR **81.75% accuracy** but poor failure recall. The production `model.py` prioritises catching real failures.

---

### Models Used

#### Logistic Regression
The interpretable baseline classifier. Trained on scaled features with balanced class weights.

Strengths:
* Fast inference
* Smooth probability outputs
* Interpretable through feature coefficients
* Handles imbalance natively via `class_weight`

---

#### Random Forest
200-tree ensemble classifier trained on raw (unscaled) features with per-tree balanced subsampling.

Strengths:
* True probability estimates averaged across 200 trees — no coarse vote-fraction problem
* Handles non-linear relationships
* Provides feature importance (mean decrease in impurity)
* Robust to outliers

---

### Why KNN Was Replaced with Random Forest

The original prototype used K-Nearest Neighbors. KNN was ultimately replaced because:

1. **Coarse probabilities** — KNN with k=5 can only output {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} as probabilities, making risk scores uninformative
2. **Imbalance sensitivity** — On a 3.4% minority class, KNN's neighborhood voting is dominated by majority-class points regardless of oversampling
3. **Calibration failure** — Attempts to calibrate KNN probabilities collapsed to the base rate constant on small minority calibration sets

Random Forest solves all three problems natively.

---

## 📈 Dataset Features

The model predicts machine failure using six operational variables:

| Feature | Description |
| ----------------------- | ------------------------------ |
| Type | Product quality type (L, M, H) |
| Air temperature [K] | Ambient operating temperature |
| Process temperature [K] | Internal process temperature |
| Rotational speed [rpm] | Machine rotational speed |
| Torque [Nm] | Applied rotational force |
| Tool wear [min] | Tool usage duration |

**Target Variable:**
Machine failure (binary: 0 = No Failure, 1 = Failure)

---

**Class distribution:** ~9,661 No Failure / ~339 Failure (~3.4% positive rate)

---

## 🏗️ Project Structure

MachineGuard-AI/ │ ├── app.py # Streamlit web application ├── model.py # ML pipeline, training, evaluation & prediction ├── ai.csv # Machine failure dataset (10,000 records) │ ├── Machine_Failure_Prediction.ipynb │ # Original exploratory notebook (baseline models) │ ├── requirements.txt # Project dependencies └── README.md # Project documentation


> **Note:** `Machine_Failure_Prediction.ipynb` documents the initial baseline exploration using LR + KNN without imbalance handling. The production pipeline in `model.py` reflects the improved approach — class-weighted models, threshold tuning, and Random Forest replacing KNN.

---

🛠️ Tech Stack
Layer	Libraries
Frontend	Streamlit
Data Processing	Pandas, NumPy
Machine Learning	Scikit-learn (LogisticRegression, RandomForestClassifier)
Visualization	Plotly
📊 Key Functionalities
✔ Predict machine failures in real time
✔ Compare two fundamentally different ML models (linear vs. ensemble)
✔ Dual ROC curve with AUC for both models
✔ LR coefficients + RF feature importances side by side
✔ Explore machine telemetry data interactively
✔ Batch process CSV uploads
✔ Download enriched prediction reports

---

🎯 Learning Outcomes
This project demonstrates:

End-to-end machine learning workflow on real industrial data
Handling class imbalance — why accuracy misleads and what to use instead
Threshold tuning for domain-appropriate risk tradeoffs
Model selection reasoning — when to replace one algorithm with another
Interactive dashboard development with Streamlit
Production-style ML application deployment on Streamlit Cloud

---

📜 License
This project is intended for educational and portfolio purposes.
