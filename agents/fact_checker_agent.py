"""
Agent 5 — Fact Checker Agent.
Compares original report findings vs generated patient explanation.
Detects unsupported claims (e.g. converting "elevated value" into a definitive disease diagnosis).
"""

import re
from typing import Dict, Any, List

class FactCheckerAgent:
    def check_facts(self, original_report: str, generated_explanation: str) -> Dict[str, Any]:
        """
        Validates generated text against original report facts.
        Flags unsupported diagnosis statements or hallucinated medical conditions.
        """
        issues: List[str] = []
        is_valid = True

        gen_lower = generated_explanation.lower()
        orig_lower = original_report.lower()

        # Check for aggressive diagnostic leaps not present in original report
        unsupported_diagnoses = [
            ("cancer", ["cancer", "malignancy", "carcinoma", "tumor"]),
            ("diabetes", ["diabetes", "diabetic"]),
            ("stroke", ["stroke", "infarct"]),
            ("heart attack", ["heart attack", "myocardial infarction"]),
        ]

        for label, keywords in unsupported_diagnoses:
            found_in_gen = any(k in gen_lower for k in keywords)
            found_in_orig = any(k in orig_lower for k in keywords)

            if found_in_gen and not found_in_orig:
                is_valid = False
                issues.append(f"Unsupported diagnosis claim: '{label}' is mentioned in explanation but not found in original report.")

        # Check for definitive diagnosis phrasing vs observation
        if "confirms that you have" in gen_lower or "definitely proves you suffer from" in gen_lower:
            is_valid = False
            issues.append("Definitive diagnosis claim detected: 'confirms that you have' exceeds report findings.")

        return {
            "is_valid": is_valid,
            "unsupported_claims_count": len(issues),
            "issues": issues,
            "status": "PASSED" if is_valid else "FAILED_FACT_CHECK"
        }
