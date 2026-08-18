"""
Builds a searchable vector index over the manual's chunks using free,
locally-run models (no API cost, no external calls needed for this part).

Retrieval is two-stage:
  1. Bi-encoder (MiniLM) + FAISS -> fast, coarse candidate search over ALL
     chunks. Cheap because chunk embeddings are precomputed once at startup;
     only the query needs encoding at search time.
  2. Cross-encoder -> re-scores just the candidates from stage 1 by feeding
     (query, chunk) pairs jointly into the model, which lets it actually
     weigh how the two texts interact instead of comparing two independently-
     computed vectors. Slower per-pair, but only runs on a small candidate
     set, so it's affordable — and it's the step that fixes cases where the
     bi-encoder's top result is topically similar but not the actual best
     answer.
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from chunking import load_and_chunk_knowledge_folder, KNOWLEDGE_DIR

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, free, fast bi-encoder for stage 1
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # free cross-encoder for stage 2

# How many candidates stage 1 hands to the cross-encoder. Wider than the
# final top_k on purpose: the bi-encoder is a coarse filter, so the true best
# match sometimes sits at rank 8-15 rather than rank 1-3 — the cross-encoder
# needs those candidates present to be able to promote them.
# The manual has a small number of chunks. Re-ranking all of them avoids
# losing the precise section during the coarse semantic-search stage.
CANDIDATE_POOL_SIZE = 80


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class ManualRetriever:
    def __init__(self, folder_path=KNOWLEDGE_DIR, groq_api_key=None):
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.reranker = CrossEncoder(RERANK_MODEL_NAME)
        self.chunks = load_and_chunk_knowledge_folder(folder_path, groq_api_key=groq_api_key)
        # Embed title + text together, not just text. Without the title, a
        # chunk that's mostly bullet points/status fields (e.g. the tail end
        # of a long section, if one ever needs splitting) has no signal in
        # its own vector connecting it back to what topic it's actually
        # about — so it scores weakly against a query that names that topic,
        # even though the content is technically relevant. Prepending the
        # title anchors every chunk's embedding to its section identity.
        texts = [f"{c['title']}\n\n{c['text']}" for c in self.chunks]
        embeddings = self.embed_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        self.embeddings = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])  # cosine similarity via inner product on normalized vectors
        self.index.add(self.embeddings)

    def search(self, query, top_k=3, min_score=0.20, candidate_pool_size=CANDIDATE_POOL_SIZE):
        """
        Stage 1: FAISS returns candidate_pool_size candidates by cosine
        similarity (fast, coarse).
        Stage 2: the cross-encoder re-scores each (query, chunk) pair
        jointly and re-sorts by that score — this is the actual ranking
        used for top_k / min_score below, not the stage-1 cosine score.

        min_score filters on the cross-encoder's score (passed through a
        sigmoid, so ~0-1 like a probability rather than the model's raw
        logit). 0.15 is a deliberately loose default — in practice, ms-marco
        cross-encoder scores for genuinely relevant (query, chunk) pairs can
        still land well under 0.5 when the wording doesn't closely match
        (e.g. the manual says "RM500" while the question says "above what
        amount"), so a strict cutoff silently drops correct answers and the
        bot ends up saying "I don't know" about things that ARE in the
        manual. Tune this against your own queries — run this file directly
        with min_score=0 to see the real score distribution first.

        Regardless of min_score, the single best-scoring candidate is always
        kept even if it falls below the threshold. An empty result list
        means the LLM gets zero context and can only ever say "I don't
        know" — but the top candidate, even at a so-so score, is usually
        still more useful to hand to the model than nothing at all, and the
        model's own instructions already tell it to say "I don't know" if
        that context turns out to be insufficient. min_score still prunes
        the noisy tail (ranks 2+), which is where it actually helps.
        """
        pool = min(candidate_pool_size, len(self.chunks))
        query_vec = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        _, indices = self.index.search(query_vec, pool)
        candidate_indices = [int(i) for i in indices[0] if i != -1]
        if not candidate_indices:
            return []

        pairs = [[query, self.chunks[i]["text"]] for i in candidate_indices]
        rerank_scores = self.reranker.predict(pairs)
        rerank_scores = _sigmoid(np.array(rerank_scores))

        # Locally verified excerpts are corrections checked against the source
        # PDF. Give them a small tie-break boost over older AI transcription.
        adjusted_scores = []
        for idx, score in zip(candidate_indices, rerank_scores):
            if self.chunks[idx]["source"] == "RMS_verified_pages.md":
                score = min(1.0, score + 0.10)
            adjusted_scores.append(score)
        ranked = sorted(zip(candidate_indices, adjusted_scores), key=lambda p: p[1], reverse=True)

        # A verified excerpt was created only when the earlier transcription
        # for that exact page was incomplete or wrong. It is sufficient and
        # authoritative by itself, so withholding nearby legacy chunks avoids
        # contradicting it with a similarly worded but different procedure.
        if ranked and self.chunks[ranked[0][0]]["source"] == "RMS_verified_pages.md":
            ranked = ranked[:1]

        results = []
        for rank, (idx, score) in enumerate(ranked[:top_k]):
            if rank > 0 and score < min_score:
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
        # Previously returned "I don't know" in production — run these with
        # min_score=0 to see the REAL score distribution before deciding
        # what min_score should actually be. If the right chunk shows up
        # here with a score below your current min_score, that confirms
        # the threshold (not retrieval itself) was the problem.
        "For external grant purchasing, above what amount does the internal purchasing procedure apply?",
        "What is the threshold amount for Asset Tagging registration?",
        "What is the processing time for the Payroll stage? Is there any exception?",
        "What are the possible values for the \"Evaluation Status\" field?",
    ]
    for q in test_queries:
        print(f"\n=== Query: {q} ===")
        results = retriever.search(q, top_k=5, min_score=0)  # min_score=0 to see everything
        for r in results:
            print(f"[{r['score']:.3f}] {r['title']}")
