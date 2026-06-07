"""ML pipeline for machine failure prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

LEAKAGE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
DROP_COLS = ["UDI", "Product ID"]
TARGET = "Machine failure"
FEATURE_COLS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TYPE_LABELS = {"L": "Low", "M": "Medium", "H": "High"}
K_VALUES = [3, 5, 7, 9]


@dataclass
class ModelBundle:
    lr: LogisticRegression
    knn: KNeighborsClassifier
    scaler: StandardScaler
    encoder: LabelEncoder
    feature_cols: list[str]
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred_lr: np.ndarray
    y_pred_knn: np.ndarray
    y_prob_lr: np.ndarray
    knn_k_accuracies: dict[int, float]
    best_k: int
    raw_df: pd.DataFrame
    processed_df: pd.DataFrame


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame, encoder: LabelEncoder | None = None) -> tuple[pd.DataFrame, LabelEncoder]:
    data = df.copy()
    data.drop(columns=[c for c in DROP_COLS if c in data.columns], inplace=True)
    data.fillna(data.median(numeric_only=True), inplace=True)

    if encoder is None:
        encoder = LabelEncoder()
        data["Type"] = encoder.fit_transform(data["Type"])
    else:
        data["Type"] = encoder.transform(data["Type"])

    data.drop(columns=[c for c in LEAKAGE_COLS if c in data.columns], inplace=True)
    return data, encoder


def train_models(csv_path: str | Path = "ai.csv") -> ModelBundle:
    raw_df = load_raw_data(csv_path)
    processed_df, encoder = preprocess(raw_df)

    X = processed_df.drop(columns=[TARGET])
    y = processed_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

    knn_k_accuracies: dict[int, float] = {}
    best_k, best_acc = K_VALUES[0], 0.0
    for k in K_VALUES:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test_scaled))
        knn_k_accuracies[k] = acc
        if acc > best_acc:
            best_k, best_acc = k, acc

    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train_scaled, y_train)
    y_pred_knn = knn_final.predict(X_test_scaled)

    return ModelBundle(
        lr=lr,
        knn=knn_final,
        scaler=scaler,
        encoder=encoder,
        feature_cols=list(X.columns),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        y_pred_lr=y_pred_lr,
        y_pred_knn=y_pred_knn,
        y_prob_lr=y_prob_lr,
        knn_k_accuracies=knn_k_accuracies,
        best_k=best_k,
        raw_df=raw_df,
        processed_df=processed_df,
    )


def build_input_frame(inputs: dict, encoder: LabelEncoder, feature_cols: list[str]) -> pd.DataFrame:
    row = {col: inputs[col] for col in feature_cols}
    frame = pd.DataFrame([row])
    frame["Type"] = encoder.transform(frame["Type"])
    return frame


def predict(bundle: ModelBundle, inputs: dict) -> dict:
    frame = build_input_frame(inputs, bundle.encoder, bundle.feature_cols)
    scaled = bundle.scaler.transform(frame)

    lr_pred = int(bundle.lr.predict(scaled)[0])
    knn_pred = int(bundle.knn.predict(scaled)[0])
    lr_prob = float(bundle.lr.predict_proba(scaled)[0, 1])
    knn_prob = float(bundle.knn.predict_proba(scaled)[0, 1])

    return {
        "lr_pred": lr_pred,
        "knn_pred": knn_pred,
        "lr_prob": lr_prob,
        "knn_prob": knn_prob,
        "lr_label": "Failure" if lr_pred == 1 else "No Failure",
        "knn_label": "Failure" if knn_pred == 1 else "No Failure",
        "agreement": lr_pred == knn_pred,
    }


def classification_metrics(y_true, y_pred) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_failure": report["1"]["precision"],
        "recall_failure": report["1"]["recall"],
        "f1_failure": report["1"]["f1-score"],
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": report,
    }


def roc_data(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, auc(fpr, tpr), thresholds


def feature_ranges(raw_df: pd.DataFrame) -> dict:
    numeric = raw_df.drop(columns=[c for c in DROP_COLS + LEAKAGE_COLS + [TARGET] if c in raw_df.columns])
    ranges = {}
    for col in numeric.select_dtypes(include="number").columns:
        ranges[col] = (float(numeric[col].min()), float(numeric[col].max()))
    return ranges


def sample_presets(raw_df: pd.DataFrame) -> dict[str, dict]:
    no_fail = raw_df[raw_df[TARGET] == 0].iloc[0]
    fail = raw_df[raw_df[TARGET] == 1].iloc[0]
    high_wear = raw_df.sort_values("Tool wear [min]", ascending=False).iloc[0]

    def row_to_inputs(row: pd.Series) -> dict:
        return {
            "Type": row["Type"],
            "Air temperature [K]": float(row["Air temperature [K]"]),
            "Process temperature [K]": float(row["Process temperature [K]"]),
            "Rotational speed [rpm]": float(row["Rotational speed [rpm]"]),
            "Torque [Nm]": float(row["Torque [Nm]"]),
            "Tool wear [min]": float(row["Tool wear [min]"]),
        }

    return {
        "Normal operation": row_to_inputs(no_fail),
        "Known failure": row_to_inputs(fail),
        "High tool wear": row_to_inputs(high_wear),
    }
