"""
Agent 7 — Patient Explanation Agent.
Converts complex medical terminology and lab jargon into plain, empathetic,
patient-accessible English without making medical diagnoses.
"""

from typing import Dict, Any, List

TERMS_GLOSSARY = {
    "sinus tachycardia": "a resting heart rate that is faster than typical (over 100 beats per minute)",
    "non-specific t-wave inversions": "minor variations in the heart's electrical pattern that are non-specific and require context",
    "elevated troponin": "higher levels of a protein released when heart muscle cells experience stress or injury",
    "lvef": "Left Ventricular Ejection Fraction, which measures how effectively the heart pumps blood out with each beat",
    "white matter hyperintensities": "small areas on a brain MRI scan that show up brighter, often related to normal aging or minor vascular changes",
    "ischemic infarct": "an area where reduced blood flow affected local tissue",
    "hepatic steatosis": "fat accumulation within liver cells (commonly called fatty liver)",
    "gastritis": "irritation or inflammation of the stomach lining",
    "non-displaced fracture": "a clean bone break where the bone pieces remain in their normal alignment",
    "disc herniation": "a condition where one of the cushion discs between spine bones bulges slightly",
    "hemoglobin a1c": "a lab blood test measuring average blood sugar levels over the past 2-3 months",
    "leukocytosis": "an elevated white blood cell count, often reflecting immune activity or physical stress"
}

class PatientExplanationAgent:
    def generate_explanation(self, report_text: str, specialty: str, entities: Dict[str, Any], evidence: List[Dict[str, Any]]) -> str:
        """Generates a plain-English explanation of the report findings."""
        explanation_parts = []
        
        explanation_parts.append(f"### Understanding Your {specialty} Report\n")
        explanation_parts.append("Below is a breakdown of your report findings explained in simple terms:\n")

        # 1. Summarize key observations
        if entities.get("symptoms"):
            syms = ", ".join(entities["symptoms"])
            explanation_parts.append(f"• **Reported Symptoms**: Your report notes symptoms including: *{syms}*.")

        if entities.get("measurements"):
            meas = ", ".join(entities["measurements"])
            explanation_parts.append(f"• **Key Values Recorded**: Key recorded measurements include *{meas}*.")

        if entities.get("tests"):
            tests_str = ", ".join(entities["tests"])
            explanation_parts.append(f"• **Diagnostic Tests**: The evaluation reference includes *{tests_str}*.")

        # 2. Translate technical jargon
        explanation_parts.append("\n**Medical Term Clarification:**")
        matched_terms = False
        r_lower = report_text.lower()
        for term, plain in TERMS_GLOSSARY.items():
            if term in r_lower:
                matched_terms = True
                explanation_parts.append(f"• **{term.title()}**: In simple terms, this refers to {plain}.")

        if not matched_terms:
            explanation_parts.append("• Your report describes standard clinical observations and diagnostic findings.")

        # 3. Add educational context from RAG evidence
        if evidence:
            explanation_parts.append("\n**General Educational Context:**")
            for ev in evidence[:2]:
                snippet = ev.get("text", "")[:180] + "..."
                explanation_parts.append(f"• *{ev.get('title', 'Reference')}*: {snippet}")

        explanation_parts.append("\n**Next Steps:**")
        explanation_parts.append("Please review these results with your healthcare provider to discuss what these measurements mean for your individual care plan.")

        return "\n".join(explanation_parts)
