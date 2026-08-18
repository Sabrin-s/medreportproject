"""
Agent 1 — Intake / Router Agent.
Detects input format (Plain Text, PDF, Voice Dictation, Follow-up Query),
normalizes content, and computes routing mode (Fast Path vs Deep Agent Path).
"""

from typing import Dict, Any, Tuple

class RouterAgent:
    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold

    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines input source and routes execution path.
        Returns payload with normalized text, input_type, and routing_decision.
        """
        input_type = input_data.get("input_type", "text")
        raw_content = input_data.get("content", "")

        normalized_text = str(raw_content).strip()
        
        return {
            "input_type": input_type,
            "normalized_text": normalized_text,
            "char_count": len(normalized_text),
            "status": "ready"
        }

    def determine_route(self, confidence_score: float) -> Tuple[str, str]:
        """
        Confidence >= threshold (0.75) -> Fast Path (Low Latency Direct Pipeline)
        Confidence < threshold (0.75) -> Deep Agent Path (Multi-step verification & LangGraph)
        """
        if confidence_score >= self.confidence_threshold:
            return "FAST_PATH", f"High classifier confidence ({confidence_score * 100:.1f}% >= {self.confidence_threshold * 100:.0f}%) -> Streamlined Execution"
        else:
            return "DEEP_AGENT_PATH", f"Sub-threshold classifier confidence ({confidence_score * 100:.1f}% < {self.confidence_threshold * 100:.0f}%) -> Full Deep Agent Verification"
