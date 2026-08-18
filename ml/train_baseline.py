"""
TF-IDF + Logistic Regression Baseline Classifier Trainer.
Evaluates accuracy, macro-F1, weighted-F1, precision, recall, and confusion matrix.
Saves serialized model pipeline to models/baseline_model.joblib.
"""

import os
import sys

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from ml.dataset import get_stratified_splits, SPECIALTIES

def train_and_evaluate_baseline():
    print("==================================================")
    print(" Training Baseline Model: TF-IDF + Logistic Regression")
    print("==================================================")

    splits = get_stratified_splits()
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    print(f"Dataset summary: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Build Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=500, C=1.5, random_state=42))
    ])

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred, labels=SPECIALTIES)

    print("\n--- HELD-OUT TEST METRICS (Baseline) ---")
    print(f"Accuracy:    {acc * 100:.2f}%")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(cm)

    # Ensure output directory exists
    os.makedirs("models", exist_ok=True)
    model_path = "models/baseline_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\n[OK] Baseline model successfully saved to: {model_path}")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist()
    }

if __name__ == "__main__":
    train_and_evaluate_baseline()
