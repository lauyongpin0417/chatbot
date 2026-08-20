"""
Builds a searchable vector index over the manual's chunks using free,
locally-run models (no API cost, no external calls needed for this part).

Retrieval is two-stage, with two independent ways of finding stage-1
candidates:
  1a. Bi-encoder (BGE-small) + FAISS -> fast, coarse semantic search over ALL
      chunks. Cheap because chunk embeddings are precomputed once at
      startup; only the query needs encoding at search time. Good at
      matching meaning/paraphrase, weak at matching exact numbers/codes
      (e.g. "RM500", "3.1.1") since those don't carry much semantic signal.
  1b. BM25 (keyword/lexical search) -> catches exact-term matches that the
      bi-encoder can miss, e.g. a query naming a specific amount, threshold,
      or section number that appears verbatim in the manual but isn't
      semantically "close" to how the question phrases it.
  2.  Cross-encoder -> re-scores the UNION of both candidate sets by feeding
      (query, chunk) pairs jointly into the model, which lets it actually
      weigh how the two texts interact instead of comparing two independently-
      computed vectors. Slower per-pair, but only runs on a small candidate
      set, so it's affordable — and it's the step that fixes cases where the
      bi-encoder's top result is topically similar but not the actual best
      answer.
"""
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from chunking import load_and_chunk_knowledge_folder, KNOWLEDGE_DIR

# BGE outperforms the older MiniLM bi-encoder on retrieval benchmarks at a
# similar size/speed. BGE models are trained to expect a fixed instruction
# prefix on the QUERY side only (not on the documents/chunks being indexed)
# for asymmetric search tasks like this one.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # free cross-encoder for stage 2

# How many candidates EACH of the semantic/BM25 stages hands to the
# cross-encoder (the two pools are unioned, not intersected — see search()
# — so the reranker typically sees noticeably more than this number of
# unique candidates). Wider than the final top_k on purpose: the bi-encoder
# is a coarse filter, so the true best match sometimes sits at rank 8-15
# rather than rank 1-3 — the cross-encoder needs those candidates present to
# be able to promote them.
#
# 80 (the pre-hybrid-search value, from when only the semantic stage fed the
# reranker) was measured live against this project's real knowledge base
# after BM25 was added: union size ~127 candidates, ~9.4s/query on a CPU
# reranker — dominates end-to-end response latency in a chat app where
# every second is felt. Dropping to 40 measured identically 5/6 on the same
# accuracy test set (the one query it still misses is unaffected by pool
# size — a ranking issue, not a recall issue) while cutting latency to
# ~5.7s. Going lower (25) saved little further (~5.1s), pointing at a
# largely fixed per-query reranker cost rather than one that scales down
# with candidate count — so 40 is the point of diminishing returns, not an
# arbitrary guess.
CANDIDATE_POOL_SIZE = 40

# Minimum RAW (pre-boost) cross-encoder score a RMS_verified_pages.md
# candidate needs before it's eligible for the tie-break boost below. Real
# matches for a genuinely relevant pair can score well under 0.5 on this
# scale (see search()'s docstring), so this is set low on purpose — it only
# needs to rule out actual noise (observed at 0.0000-0.0005 for a query that
# doesn't match anything in the knowledge base), not borderline-but-real
# matches.
VERIFIED_BOOST_MIN_RAW_SCORE = 0.15

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


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

        # BM25 keyword index over the same title+text strings, so an exact
        # amount/threshold/section-number mentioned verbatim in the manual
        # can be found even when the query's wording doesn't paraphrase
        # closely enough for the bi-encoder to rank it highly.
        self.bm25 = BM25Okapi([_tokenize(t) for t in texts])

    def search(self, query, top_k=3, min_score=0.20, candidate_pool_size=CANDIDATE_POOL_SIZE):
        """
        Stage 1: FAISS (semantic) and BM25 (keyword) each independently
        return up to candidate_pool_size candidates, and their results are
        unioned together.
        Stage 2: the cross-encoder re-scores each (query, chunk) pair
        jointly and re-sorts by that score — this is the actual ranking
        used for top_k / min_score below, not the stage-1 scores.

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
        query_vec = self.embed_model.encode(
            [QUERY_INSTRUCTION + query], normalize_embeddings=True
        ).astype("float32")
        _, indices = self.index.search(query_vec, pool)
        semantic_candidates = [int(i) for i in indices[0] if i != -1]

        # BM25 candidates are pulled independently and unioned with the
        # semantic ones (not intersected) — the two methods fail on
        # different kinds of queries, so a chunk only needs to be found by
        # ONE of them to reach the cross-encoder, which then judges all
        # candidates on equal footing regardless of which stage-1 method
        # surfaced them.
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_ranked = np.argsort(bm25_scores)[::-1][:pool]
        keyword_candidates = [int(i) for i in bm25_ranked if bm25_scores[i] > 0]

        seen = set()
        candidate_indices = []
        for i in semantic_candidates + keyword_candidates:
            if i not in seen:
                seen.add(i)
                candidate_indices.append(i)
        if not candidate_indices:
            return []

        pairs = [[query, self.chunks[i]["text"]] for i in candidate_indices]
        # Explicit, smaller-than-default batch_size — measured live on this
        # project's CPU deployment: 16 beat the sentence-transformers
        # default of 32 by ~10%, and beat 128 by ~37%. No GPU to exploit
        # with bigger batches here, and a bigger batch forces more padding
        # to the longest sequence in it, which on CPU costs more than it
        # saves — smaller batches measured faster, not just "safer".
        rerank_scores = self.reranker.predict(pairs, batch_size=16)
        rerank_scores = _sigmoid(np.array(rerank_scores))

        # Locally verified excerpts are corrections checked against the source
        # PDF. Give them a small tie-break boost over older AI transcription
        # — but ONLY when their own raw score already shows genuine topical
        # relevance (>= VERIFIED_BOOST_MIN_RAW_SCORE). A flat, unconditional
        # +0.10 add is enormous relative to how tiny raw cross-encoder scores
        # get for a query that doesn't match ANYTHING well (confirmed live:
        # 0.0000-0.0005 for an unrelated document's questions once the
        # knowledge base held more than one topic) — without this gate, that
        # +0.10 alone was enough to promote an objectively irrelevant
        # verified excerpt to 1st place over the genuinely correct chunk,
        # since raw scores that low leave essentially no real margin to
        # overcome. Gating on the candidate's OWN raw score (not a
        # comparison to others) keeps the boost doing what it was meant to —
        # a tie-break among already-plausible matches — without being able
        # to resurrect a candidate from actual noise.
        adjusted_scores = []
        for idx, score in zip(candidate_indices, rerank_scores):
            if self.chunks[idx]["source"] == "RMS_verified_pages.md" and score >= VERIFIED_BOOST_MIN_RAW_SCORE:
                score = min(1.0, score + 0.10)
            adjusted_scores.append(score)
        ranked = sorted(zip(candidate_indices, adjusted_scores), key=lambda p: p[1], reverse=True)

        # A verified excerpt was created only when the earlier transcription
        # for that exact page was incomplete or wrong. It is sufficient and
        # authoritative by itself, so withholding nearby legacy chunks avoids
        # contradicting it with a similarly worded but different procedure.
        #
        # Gated on the SAME min_score bar as everything else — without this,
        # a query about a totally unrelated topic (e.g. a different document
        # entirely, once the knowledge base holds more than one) scores every
        # candidate near zero, and the flat +0.10 boost above is enough by
        # itself to put a verified excerpt in 1st place on pure noise. This
        # rule would then discard every other candidate — including the
        # actually-relevant one from the other document — leaving the model
        # with an irrelevant chunk as its only context. Confirmed live: this
        # is what made "power consumption"/"carafe wait time" questions about
        # an unrelated uploaded manual return "I don't know" even though the
        # right chunk was sitting right there among the discarded candidates.
        if ranked and ranked[0][1] >= min_score and self.chunks[ranked[0][0]]["source"] == "RMS_verified_pages.md":
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
