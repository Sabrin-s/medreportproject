"""
Agent 6 — Safety Agent.
Enforces clinical safety guardrails:
- Blocks fake doctor diagnoses ("You definitely have cancer")
- Blocks dangerous medical advice ("Stop taking your medication")
- Blocks discouraging clinician consults ("You don't need a doctor")
Appends mandatory clinical disclaimers.
"""

import re
from typing import Dict, Any, List

MANDATORY_DISCLAIMER = (
    "Clinical Safety Disclaimer: MedReport Copilot is an educational AI understanding assistant, "
    "not a licensed medical professional or diagnostic tool. The explanations provided describe laboratory and imaging findings "
    "in plain language and should be reviewed together with your treating clinician or physician for diagnostic interpretation."
)

class SafetyAgent:
    def verify_safety(self, text: str) -> Dict[str, Any]:
        """Audits generated output for clinical safety violations."""
        violations: List[str] = []
        t_lower = text.lower()

        # Dangerous pattern checks
        if re.search(r'\b(?:you (?:definitely|certainly) have|you are diagnosed with)\b', t_lower):
            violations.append("Definitive diagnosis phrase detected.")

        if re.search(r'\b(?:stop taking|discontinue|throw away|don\'t take)\b', t_lower):
            violations.append("Medication alteration advice detected.")

        if re.search(r'\b(?:don\'t see a doctor|no need to see a doctor|ignore your doctor)\b', t_lower):
            violations.append("Discouraging clinical consult detected.")

        is_safe = len(violations) == 0
        sanitized_text = text

        if not is_safe:
            # Reframe unsafe statements to neutral educational summary
            sanitized_text = (
                "The report contains clinical measurements and findings. "
                "These findings describe specific laboratory or diagnostic observations. "
                "Its clinical significance depends on your complete medical history and should be discussed directly with your healthcare provider."
            )

        # Append mandatory disclaimer
        final_text = f"{sanitized_text}\n\n{MANDATORY_DISCLAIMER}"

        return {
            "is_safe": is_safe,
            "violations": violations,
            "sanitized_text": final_text,
            "disclaimer_attached": True
        }
