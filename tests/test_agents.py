"""
Unit tests for 8 Subagents & Deep Agent Orchestrator.
"""

import pytest
from agents.router_agent import RouterAgent
from agents.clinical_nlp_agent import ClinicalNLPAgent
from agents.fact_checker_agent import FactCheckerAgent
from agents.safety_agent import SafetyAgent
from agents.deep_orchestrator import DeepAgentOrchestrator

def test_router_confidence_threshold():
    router = RouterAgent(confidence_threshold=0.75)
    route_fast, _ = router.determine_route(0.85)
    route_deep, _ = router.determine_route(0.60)
    
    assert route_fast == "FAST_PATH"
    assert route_deep == "DEEP_AGENT_PATH"

def test_clinical_nlp_entity_extraction():
    nlp = ClinicalNLPAgent()
    report = "BP 140/90 mmHg, HR 98 bpm. Patient prescribed Lisinopril and Atorvastatin. ECG completed."
    entities = nlp.extract_entities(report)
    
    assert "140/90 mmHg" in entities["measurements"]
    assert "98 bpm" in entities["measurements"]
    assert "Lisinopril" in entities["medications"]
    assert "ECG" in entities["tests"]

def test_fact_checker_rejects_hallucinations():
    checker = FactCheckerAgent()
    orig = "Elevated AST and ALT liver enzymes noted on routine blood draw."
    bad_gen = "This report confirms that you have stage 4 liver cancer."
    
    res = checker.check_facts(orig, bad_gen)
    assert not res["is_valid"]
    assert res["status"] == "FAILED_FACT_CHECK"

def test_safety_agent_attaches_disclaimer():
    safety = SafetyAgent()
    unsafe_text = "You definitely have heart disease. Stop taking your medicine."
    res = safety.verify_safety(unsafe_text)
    
    assert not res["is_safe"]
    assert "Clinical Safety Disclaimer" in res["sanitized_text"]

def test_deep_orchestrator_full_run():
    orchestrator = DeepAgentOrchestrator()
    payload = {
        "input_type": "text",
        "content": "Patient presents with chest discomfort. ECG shows sinus tachycardia at 105 bpm. Troponin elevated."
    }
    result = orchestrator.run(payload)
    
    assert "specialty" in result
    assert "explanation" in result
    assert len(result["execution_pipeline"]) >= 8
