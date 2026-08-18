"""
Unit tests for RAG Service knowledge base retrieval.
"""

import pytest
from services.rag import RAGService

def test_rag_knowledge_indexing_and_search():
    rag = RAGService(knowledge_dir="data/knowledge")
    assert len(rag.documents) > 0

    results = rag.search("Electrocardiogram ST-segment elevation troponin", top_k=2)
    assert len(results) > 0
    assert "source" in results[0]
    assert results[0]["relevance_score"] > 0.05
