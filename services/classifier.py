"""
Medical Specialty Classifier Service.
Wraps model loading, feature extraction, predictions, and confidence scoring.
Provides fallback to TF-IDF baseline when PyTorch model checkpoint is not present.
"""

import os
import joblib
import numpy as np
from typing import Dict, Any

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ml.dataset import SPECIALTIES

class SpecialtyClassifierService:
    def __init__(self, baseline_path: str = "models/baseline_model.joblib", distil_path: str = "models/distilbert_specialty"):
        self.baseline_path = baseline_path
        self.distil_path = distil_path
        self.baseline_model = None
        self.tokenizer = None
        self.distil_model = None
        self._load_models()

    def _load_models(self):
        # 1. Try loading baseline
        if os.path.exists(self.baseline_path):
            try:
                self.baseline_model = joblib.load(self.baseline_path)
            except Exception as e:
                print(f"[ClassifierService] Error loading baseline model: {e}")

        # 2. Try loading DistilBERT
        if TRANSFORMERS_AVAILABLE and os.path.exists(self.distil_path):
            try:
                if os.path.exists(os.path.join(self.distil_path, "pytorch_model.bin")) or os.path.exists(os.path.join(self.distil_path, "model.safetensors")):
                    self.tokenizer = AutoTokenizer.from_pretrained(self.distil_path)
                    self.distil_model = AutoModelForSequenceClassification.from_pretrained(self.distil_path)
                    self.distil_model.eval()
            except Exception as e:
                print(f"[ClassifierService] DistilBERT not fully loaded, using baseline: {e}")

    def predict(self, text: str) -> Dict[str, Any]:
        """Predicts medical specialty and returns label + confidence score (0.0 to 1.0)."""
        if not text or not text.strip():
            return {"specialty": "General Medicine", "confidence": 0.50, "model_used": "fallback"}

        # Use DistilBERT if loaded
        if self.distil_model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.distil_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
                    pred_idx = np.argmax(probs)
                    confidence = float(probs[pred_idx])
                    specialty = SPECIALTIES[pred_idx] if pred_idx < len(SPECIALTIES) else "General Medicine"
                    return {
                        "specialty": specialty,
                        "confidence": round(confidence, 4),
                        "model_used": "DistilBERT Fine-Tuned"
                    }
            except Exception as e:
                print(f"[ClassifierService] DistilBERT inference error: {e}")

        # Fallback to Baseline (TF-IDF + Logistic Regression)
        if self.baseline_model is not None:
            try:
                probs = self.baseline_model.predict_proba([text])[0]
                pred_idx = np.argmax(probs)
                confidence = float(probs[pred_idx])
                classes = self.baseline_model.classes_
                specialty = classes[pred_idx]
                return {
                    "specialty": specialty,
                    "confidence": round(confidence, 4),
                    "model_used": "TF-IDF + Logistic Regression Baseline"
                }
            except Exception as e:
                print(f"[ClassifierService] Baseline prediction error: {e}")

        # Rule-based fallback if models are not pre-trained
        return self._heuristic_predict(text)

    def _heuristic_predict(self, text: str) -> Dict[str, Any]:
        t = text.lower()
        if any(w in t for w in ["ecg", "heart", "troponin", "ejection fraction", "cardio", "chest pain", "lvef", "pulse", "bpm"]):
            return {"specialty": "Cardiovascular / Pulmonary", "confidence": 0.88, "model_used": "Rule-Based Heuristic"}
        elif any(w in t for w in ["mri brain", "headache", "stroke", "neurology", "eeg", "seizure", "cranial", "paralysis"]):
            return {"specialty": "Neurology", "confidence": 0.86, "model_used": "Rule-Based Heuristic"}
        elif any(w in t for w in ["colonoscopy", "gastritis", "liver", "alt", "ast", "stomach", "diarrhea", "egd", "epigastric"]):
            return {"specialty": "Gastroenterology", "confidence": 0.87, "model_used": "Rule-Based Heuristic"}
        elif any(w in t for w in ["fracture", "knee", "spine", "disc", "x-ray", "bone", "joint", "radius", "meniscus"]):
            return {"specialty": "Orthopedics", "confidence": 0.85, "model_used": "Rule-Based Heuristic"}
        elif any(w in t for w in ["biopsy", "carcinoma", "tumor", "nodule", "oncology", "metastasis", "cea", "lesion"]):
            return {"specialty": "Oncology", "confidence": 0.86, "model_used": "Rule-Based Heuristic"}
        elif any(w in t for w in ["diabetes", "a1c", "thyroid", "tsh", "glucose", "insulin", "adrenal", "cortisol"]):
            return {"specialty": "Endocrinology", "confidence": 0.89, "model_used": "Rule-Based Heuristic"}
        
        return {"specialty": "General Internal Medicine", "confidence": 0.72, "model_used": "Rule-Based Heuristic"}
