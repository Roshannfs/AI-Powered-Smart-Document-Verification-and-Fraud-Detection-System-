"""
Model Training Script for Document Classification.
Trains a TF-IDF + Calibrated Logistic Regression pipeline on the prepared dataset,
evaluates on a held-out test split, and serializes the model artifact and metadata.
"""

import datetime
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config.config import DATASET_DIR, MODELS_DIR, MODEL_PATH, METADATA_PATH


def train():
    dataset_path = DATASET_DIR / "documents_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run prepare_dataset.py first.")

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Total samples: {len(df)} across {df['document_type'].nunique()} classes.")

    X = df["text"].astype(str)
    y = df["document_type"]

    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # Pipeline: TF-IDF n-grams + Logistic Regression with calibrated probability
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=4000,
            sublinear_tf=True,
            stop_words="english"
        )),
        ("classifier", LogisticRegression(
            C=1.5,
            max_iter=1000,
            solver="lbfgs"
        ))
    ])

    print("Training TF-IDF + Logistic Regression Classifier...")
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n--- Model Test Evaluation ---")
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    # Save trained model pipeline
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model pipeline to: {MODEL_PATH}")

    # Save comprehensive metadata
    classes = list(pipeline.classes_)
    metadata = {
        "model_name": "TF-IDF + LogisticRegression Document Classifier",
        "trained_at": datetime.datetime.now().isoformat(),
        "classes": classes,
        "num_classes": len(classes),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_accuracy": round(float(acc), 4),
        "classification_report": report_dict,
        "max_features": 4000,
        "ngram_range": [1, 2]
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to: {METADATA_PATH}")


if __name__ == "__main__":
    train()
