"""
Agent 3 — Medical Specialty Classifier Agent.
Wraps ML prediction service to determine report specialty and confidence score.
Report -> ML Model -> { specialty: "...", confidence: float }
"""

from typing import Dict, Any
from services.classifier import SpecialtyClassifierService

class SpecialtyAgent:
    def __init__(self, classifier_service: SpecialtyClassifierService = None):
        self.classifier = classifier_service or SpecialtyClassifierService()

    def classify_report(self, text: str) -> Dict[str, Any]:
        """Classifies clinical text into a target specialty."""
        result = self.classifier.predict(text)
        return {
            "specialty": result.get("specialty", "General Medicine"),
            "confidence": result.get("confidence", 0.70),
            "model_used": result.get("model_used", "Baseline")
        }
