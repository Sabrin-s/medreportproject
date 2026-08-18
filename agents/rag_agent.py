"""
Agent 4 — Evidence / RAG Agent.
Retrieves relevant medical guidelines and educational context from vetted local knowledge base.
Report finding -> Retriever -> Relevant medical evidence + Sources
"""

from typing import List, Dict, Any
from services.rag import RAGService

class RAGAgent:
    def __init__(self, rag_service: RAGService = None):
        self.rag = rag_service or RAGService()

    def retrieve_evidence(self, query_text: str, specialty: str = "") -> List[Dict[str, Any]]:
        """Retrieves top evidence passages relevant to the report query."""
        search_query = f"{specialty} {query_text}".strip()
        hits = self.rag.search(search_query, top_k=3)
        return hits
