# ⚙️ MachineGuard AI — Predictive Maintenance System

MachineGuard AI is a machine failure prediction platform built with Machine Learning and Streamlit. It helps identify potential equipment failures before they happen by analyzing industrial sensor readings and generating real-time risk assessments.

The application combines **Logistic Regression** and **K-Nearest Neighbors (KNN)** models to predict machine failure probability, provide model comparisons, visualize performance metrics, and support both single-machine and batch predictions.

---


🚀 Live Demo: https://machinefailurepredictionsystem-26.streamlit.app/

## 📸 Application Preview
<img width="2876" height="1704" alt="image" src="https://github.com/user-attachments/assets/a07b5d9c-58e1-42cf-871a-ec4d5e09113c" />
<img width="2876" height="1214" alt="image" src="https://github.com/user-attachments/assets/983ce1b5-0ab1-4954-9c28-9cd6c9b8ee8e" />

---

## 🚀 Features

### 🔮 Real-Time Failure Prediction

* Predict machine failure from live sensor inputs
* Dual-model prediction using Logistic Regression and KNN
* Combined risk assessment score
* Model agreement analysis

### 📊 Model Analytics Dashboard

* Confusion matrices
* ROC Curve and AUC analysis
* K-value tuning visualization
* Feature importance analysis
* Classification reports

### 🗂️ Interactive Data Explorer

* Explore all machine telemetry records
* Filter by machine type and failure status
* Distribution analysis
* Correlation heatmaps
* Raw dataset inspection

### 📤 Batch Scoring

* Upload CSV files containing multiple machine records
* Generate predictions for hundreds of machines simultaneously
* Download enriched prediction results

### 🎨 Modern Dashboard UI

* Fully customized Streamlit interface
* Responsive layout
* Interactive Plotly visualizations
* Industrial-themed design system

---

## 🧠 Machine Learning Pipeline

### Data Preprocessing

* Missing value handling
* Label encoding for machine type
* Feature scaling using StandardScaler
* Removal of data leakage columns
* Train-test split with stratification

### Models Used

#### Logistic Regression

Used as a baseline interpretable classification model.

Advantages:

* Fast inference
* Probabilistic predictions
* Easy interpretability through coefficients

#### K-Nearest Neighbors (KNN)

Used to capture non-linear relationships within machine telemetry data.

Advantages:

* Simple yet effective
* Learns complex local patterns
* No assumptions about data distribution

### Model Selection

The system evaluates multiple K values:

```text
K = 3, 5, 7, 9
```

The best-performing K value is automatically selected based on validation accuracy.

---

## 📈 Dataset Features

The model predicts machine failure using six operational variables:

| Feature                 | Description                    |
| ----------------------- | ------------------------------ |
| Type                    | Product quality type (L, M, H) |
| Air temperature [K]     | Ambient operating temperature  |
| Process temperature [K] | Internal process temperature   |
| Rotational speed [rpm]  | Machine rotational speed       |
| Torque [Nm]             | Applied rotational force       |
| Tool wear [min]         | Tool usage duration            |

Target Variable:

```text
Machine failure
```

---

## 🏗️ Project Structure

```text
Machine_Failure_Prediction_System/
│
├── app.py                      # Streamlit web application
├── model.py                    # ML pipeline, training, evaluation & prediction logic
├── ai.csv                      # Machine failure dataset
│
├── Machine_Failure_Prediction.ipynb
│                               # Model development & experimentation notebook
│
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── streamlit_app_link
---

## ⚡ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/machineguard-ai.git
cd machineguard-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

  * Logistic Regression
  * K-Nearest Neighbors
  * StandardScaler
  * LabelEncoder

### Visualization

* Plotly

---

## 📊 Key Functionalities

✔ Predict machine failures in real time

✔ Compare multiple machine learning models

✔ Visualize performance metrics

✔ Analyze feature importance

✔ Explore machine telemetry data

✔ Batch process CSV uploads

✔ Download prediction reports

---

## 🎯 Learning Outcomes

This project demonstrates:

* End-to-end machine learning workflow
* Data preprocessing and feature engineering
* Model training and evaluation
* Hyperparameter tuning
* Interactive dashboard development
* Data visualization
* Production-style ML application deployment

---

## 📜 License

This project is intended for educational and portfolio purposes.

---

## 👨‍💻 Author

**Ayesha Tariq**

Built as a Machine Learning and Data Science project focused on predictive maintenance and industrial AI applications.
