"""
Agent 2 — Clinical NLP Agent.
Extracts structured medical entities (symptoms, vitals/measurements, dates,
medications, diagnostic tests, lab findings, clinical assessment, and uncertainty).
Does not diagnose.
"""

import re
from typing import Dict, Any, List

class ClinicalNLPAgent:
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extracts key clinical facts without formulating medical diagnoses."""
        if not text:
            return {
                "symptoms": [], "measurements": [], "dates": [],
                "medications": [], "tests": [], "findings": [],
                "uncertainty_detected": False
            }

        # Vitals and measurements (e.g. 120/80 mmHg, 98 bpm, 0.45 ng/mL, 58%, 142/88)
        measurements = re.findall(
            r'\b\d{2,3}/\d{2,3}\s*(?:mmHg)?|\b\d{2,3}\s*(?:bpm|mg/dL|ng/mL|U/L|mIU/L|cm|mm|%)\b',
            text, re.IGNORECASE
        )

        # Common tests
        known_tests = [
            "ECG", "EKG", "Echocardiography", "Holter monitor", "MRI", "CT", "X-ray",
            "Troponin", "CBC", "EGD", "Endoscopy", "Colonoscopy", "Ultrasound", "EEG",
            "Hemoglobin A1c", "PET scan", "Biopsy", "CEA"
        ]
        tests = [t for t in known_tests if re.search(r'\b' + re.escape(t) + r'\b', text, re.IGNORECASE)]

        # Common medications
        known_meds = [
            "Lisinopril", "Atorvastatin", "Metformin", "Insulin", "Aspirin", "Heparin",
            "Omeprazole", "Levothyroxine", "Metoprolol", "Amoxicillin"
        ]
        medications = [m for m in known_meds if re.search(r'\b' + re.escape(m) + r'\b', text, re.IGNORECASE)]

        # Symptoms
        symptom_keywords = [
            "chest discomfort", "chest pain", "dyspnea", "edema", "palpitations",
            "headache", "droop", "dysphasia", "tremor", "burning pain", "diarrhea",
            "weight loss", "knee pain", "back pain", "fatigue", "polyuria", "polydipsia"
        ]
        symptoms = [s for s in symptom_keywords if s in text.lower()]

        # Dates / Age / Record numbers
        dates_and_refs = re.findall(r'Ref #\d+|\b\d{1,2}/\d{1,2}/\d{2,4}\b|age: \d+', text, re.IGNORECASE)

        # Uncertainty flags
        uncertainty = bool(re.search(r'\b(?:non-specific|pending|possible|equivocal|unclear|suspected)\b', text, re.IGNORECASE))

        return {
            "symptoms": symptoms,
            "measurements": list(set(measurements)),
            "dates_and_refs": dates_and_refs,
            "medications": medications,
            "tests": tests,
            "findings_summary": text[:200] + ("..." if len(text) > 200 else ""),
            "uncertainty_detected": uncertainty
        }
