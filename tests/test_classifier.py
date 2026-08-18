"""
Unit tests for Medical Specialty Classifier & Baseline Training.
"""

import os
import pytest
from services.classifier import SpecialtyClassifierService
from ml.train_baseline import train_and_evaluate_baseline

def test_baseline_training_and_saving():
    metrics = train_and_evaluate_baseline()
    assert metrics["accuracy"] > 0.60
    assert metrics["macro_f1"] > 0.50
    assert os.path.exists("models/baseline_model.joblib")

def test_classifier_service_prediction():
    service = SpecialtyClassifierService()
    cardio_text = "ECG shows sinus tachycardia at 110 bpm with non-specific T-wave inversions. Troponin I elevated at 0.5 ng/mL."
    res = service.predict(cardio_text)
    
    assert "specialty" in res
    assert "confidence" in res
    assert res["specialty"] == "Cardiovascular / Pulmonary"
    assert res["confidence"] > 0.50
