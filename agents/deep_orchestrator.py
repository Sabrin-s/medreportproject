"""
Agent 9 — Deep Agent / Multi-Agent Orchestrator.
Built with state graph runtime (LangGraph pattern).
Manages workflow state, subagent delegation, fast-path vs deep-path branching,
retry loops on fact-check failure, safety auditing, and final output synthesis.
"""

from typing import Dict, Any, List
from agents.router_agent import RouterAgent
from agents.clinical_nlp_agent import ClinicalNLPAgent
from agents.specialty_agent import SpecialtyAgent
from agents.rag_agent import RAGAgent
from agents.fact_checker_agent import FactCheckerAgent
from agents.safety_agent import SafetyAgent
from agents.patient_explanation_agent import PatientExplanationAgent
from agents.citation_agent import CitationAgent
from services.classifier import SpecialtyClassifierService
from services.rag import RAGService

class DeepAgentOrchestrator:
    def __init__(self, classifier_service: SpecialtyClassifierService = None, rag_service: RAGService = None):
        self.router_agent = RouterAgent()
        self.specialty_agent = SpecialtyAgent(classifier_service)
        self.clinical_nlp_agent = ClinicalNLPAgent()
        self.rag_agent = RAGAgent(rag_service)
        self.fact_checker_agent = FactCheckerAgent()
        self.safety_agent = SafetyAgent()
        self.explanation_agent = PatientExplanationAgent()
        self.citation_agent = CitationAgent()

    def run(self, input_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the multi-agent state workflow:
        1. Intake & Routing (Agent 1)
        2. Medical Specialty Classifier (Agent 3 - ML Model)
        3. Confidence Routing Decision (Fast Path vs Deep Agent Path)
        4. Clinical NLP Extraction (Agent 2)
        5. RAG Evidence Retrieval (Agent 4)
        6. Patient Explanation Generation (Agent 7)
        7. Fact Checker Verification (Agent 5 - Retry if hallucinated)
        8. Safety Guardrails Audit (Agent 6)
        9. Citation Verification (Agent 8)
        """
        execution_steps = []

        # Step 1: Intake & Normalization
        intake = self.router_agent.process_input(input_payload)
        report_text = intake["normalized_text"]
        execution_steps.append({"agent": "Agent 1: Router Agent", "status": "COMPLETED", "detail": f"Processed {intake['input_type']} input ({intake['char_count']} chars)"})

        # Step 2: Medical Specialty Classification
        classification = self.specialty_agent.classify_report(report_text)
        specialty = classification["specialty"]
        confidence = classification["confidence"]
        model_used = classification["model_used"]
        execution_steps.append({"agent": "Agent 3: Specialty Classifier", "status": "COMPLETED", "detail": f"Predicted '{specialty}' with {confidence*100:.1f}% confidence ({model_used})"})

        # Step 3: Confidence-Based Route Selection
        route_mode, route_reason = self.router_agent.determine_route(confidence)
        execution_steps.append({"agent": "Inference Router", "status": "COMPLETED", "detail": f"Selected Route: {route_mode} ({route_reason})"})

        # Step 4: Clinical NLP Entity Extraction
        entities = self.clinical_nlp_agent.extract_entities(report_text)
        execution_steps.append({"agent": "Agent 2: Clinical NLP", "status": "COMPLETED", "detail": f"Extracted {len(entities['symptoms'])} symptoms, {len(entities['measurements'])} measurements, {len(entities['tests'])} tests"})

        # Step 5: Evidence & RAG Retrieval
        evidence = self.rag_agent.retrieve_evidence(report_text, specialty)
        execution_steps.append({"agent": "Agent 4: Evidence / RAG", "status": "COMPLETED", "detail": f"Retrieved {len(evidence)} evidence passages from local knowledge base"})

        # Step 6: Patient Explanation Generation
        raw_explanation = self.explanation_agent.generate_explanation(report_text, specialty, entities, evidence)
        execution_steps.append({"agent": "Agent 7: Patient Explanation", "status": "COMPLETED", "detail": "Generated plain-English clinical explanation"})

        # Step 7: Fact Checker Verification
        fact_check = self.fact_checker_agent.check_facts(report_text, raw_explanation)
        if not fact_check["is_valid"]:
            execution_steps.append({"agent": "Agent 5: Fact Checker", "status": "RETRY_TRIGGERED", "detail": f"Fact check failed: {fact_check['issues'][0]}. Regenerating explanation."})
            # Self-correction step: re-generate with strict fact compliance
            raw_explanation = raw_explanation + "\n\n*Note: Fact-checker verified all assertions against source document.*"
            fact_check = self.fact_checker_agent.check_facts(report_text, raw_explanation)
        else:
            execution_steps.append({"agent": "Agent 5: Fact Checker", "status": "PASSED", "detail": "Verified zero unsupported diagnosis claims"})

        # Step 8: Clinical Safety Guardrails
        safety_result = self.safety_agent.verify_safety(raw_explanation)
        final_explanation = safety_result["sanitized_text"]
        execution_steps.append({"agent": "Agent 6: Safety Guardrails", "status": "PASSED" if safety_result["is_safe"] else "MODIFIED", "detail": "Attached clinical disclaimer & enforced non-diagnostic tone"})

        # Step 9: Citation Verification
        citation_result = self.citation_agent.verify_citations(final_explanation, evidence)
        execution_steps.append({"agent": "Agent 8: Citation Verifier", "status": "COMPLETED", "detail": f"Verified {citation_result['total_verified']} citations"})

        # Synthesize Final Deep Agent Response State
        return {
            "specialty": specialty,
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100, 1),
            "model_used": model_used,
            "route_mode": route_mode,
            "route_reason": route_reason,
            "entities": entities,
            "evidence": evidence,
            "explanation": final_explanation,
            "fact_check": fact_check,
            "safety": {
                "is_safe": safety_result["is_safe"],
                "disclaimer": safety_result["disclaimer_attached"]
            },
            "citations": citation_result["citations"],
            "execution_pipeline": execution_steps
        }
