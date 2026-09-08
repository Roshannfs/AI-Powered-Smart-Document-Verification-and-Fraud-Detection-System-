"""
Evaluation Script for Document Classification Model.
Computes Confusion Matrix, Precision, Recall, F1-Scores,
and saves evaluation results for the Streamlit dashboard.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split

from config.config import DATASET_DIR, MODELS_DIR, MODEL_PATH


def evaluate():
    dataset_path = DATASET_DIR / "documents_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    print("Loading dataset and model...")
    df = pd.read_csv(dataset_path)
    model = joblib.load(MODEL_PATH)

    X = df["text"].astype(str)
    y = df["document_type"]

    # Replicate stratified test split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    classes = list(model.classes_)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=classes, zero_division=0
    )

    class_metrics = []
    for idx, c in enumerate(classes):
        class_metrics.append({
            "class": c,
            "precision": round(float(prec[idx]), 3),
            "recall": round(float(rec[idx]), 3),
            "f1_score": round(float(f1[idx]), 3),
            "support": int(support[idx])
        })

    eval_results = {
        "overall_accuracy": round(float(acc), 4),
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": class_metrics,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    output_path = MODELS_DIR / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)

    print(f"\nOverall Test Accuracy: {acc * 100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Evaluation results successfully saved to: {output_path}")


if __name__ == "__main__":
    evaluate()
