"""
RAG (Retrieval-Augmented Generation) Knowledge Base Service.
Indexes local reference medical documents (data/knowledge/*.txt) and performs
keyword/TF-IDF similarity search to return relevant clinical evidence chunks with citations.
"""

import os
import glob
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAGService:
    def __init__(self, knowledge_dir: str = "data/knowledge"):
        self.knowledge_dir = knowledge_dir
        self.documents: List[Dict[str, str]] = []
        self.vectorizer = None
        self.doc_vectors = None
        self._load_and_index_knowledge_base()

    def _load_and_index_knowledge_base(self):
        """Loads all text files from knowledge base directory and indexes paragraphs."""
        self.documents = []
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir, exist_ok=True)
            
        pattern = os.path.join(self.knowledge_dir, "*.txt")
        files = glob.glob(pattern)

        for file_path in files:
            file_name = os.path.basename(file_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract title if present
                title = file_name
                lines = content.split("\n")
                for l in lines:
                    if l.startswith("TITLE:"):
                        title = l.replace("TITLE:", "").strip()
                        break

                # Chunk by double newlines (paragraphs/points)
                chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 30]
                for idx, chunk in enumerate(chunks):
                    if chunk.startswith("TITLE:"):
                        continue
                    self.documents.append({
                        "id": f"{file_name}#chunk_{idx}",
                        "source": file_name,
                        "title": title,
                        "text": chunk
                    })
            except Exception as e:
                print(f"[RAGService] Error reading {file_path}: {e}")

        # Index with TF-IDF Vectorizer
        if self.documents:
            corpus = [doc["text"] for doc in self.documents]
            self.vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
            self.doc_vectors = self.vectorizer.fit_transform(corpus)
            print(f"[RAGService] Indexed {len(self.documents)} evidence chunks from {len(files)} files.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Searches knowledge base for top matching evidence passages."""
        if not self.documents or self.vectorizer is None or self.doc_vectors is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors)[0]

        top_indices = similarities.argsort()[::-1][:top_k]
        results = []

        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.05:  # Relevance threshold
                doc = self.documents[idx]
                results.append({
                    "id": doc["id"],
                    "source": doc["source"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "relevance_score": round(score, 4)
                })

        return results
