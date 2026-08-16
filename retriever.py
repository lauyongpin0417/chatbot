"""
Builds a searchable vector index over the manual's chunks using a free,
locally-run embedding model (no API cost, no external calls needed for this part).
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from chunking import load_and_chunk_knowledge_folder, KNOWLEDGE_DIR

MODEL_NAME = "all-MiniLM-L6-v2"  # small, free, fast, good enough for this use case


class ManualRetriever:
    def __init__(self, folder_path=KNOWLEDGE_DIR, groq_api_key=None):
        self.model = SentenceTransformer(MODEL_NAME)
        self.chunks = load_and_chunk_knowledge_folder(folder_path, groq_api_key=groq_api_key)
        texts = [c["text"] for c in self.chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        self.embeddings = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])  # cosine similarity via inner product on normalized vectors
        self.index.add(self.embeddings)

    def search(self, query, top_k=5):
        query_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "title": self.chunks[idx]["title"],
                "text": self.chunks[idx]["text"],
                "score": float(score),
            })
        return results


if __name__ == "__main__":
    import os
    retriever = ManualRetriever(groq_api_key=os.environ.get("GROQ_API_KEY"))
    test_queries = [
        "What are the stages in the Pre-Approved Claim flow?",
        "How do I add a new member to a project?",
        "What is the timeline for RMC Verification in the Purchasing flow?",
    ]
    for q in test_queries:
        print(f"\n=== Query: {q} ===")
        results = retriever.search(q, top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['title']}")