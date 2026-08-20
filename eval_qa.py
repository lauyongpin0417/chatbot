"""
Regression test for retrieval quality — not a demo, an assertion suite.
Each case names a real question and a phrase that MUST show up somewhere in
the top-k retrieved text for that question to count as a pass. The phrases
below were pulled directly from the manual/FAQ (see RMS_verified_pages.md
and knowledge_docs/*.txt), not invented, so a failure means retrieval
genuinely regressed — not that the test itself is wrong.

Run this after any change to chunking.py / retriever.py (chunk boundaries,
embedding model, scoring, etc.) to see the effect in numbers instead of by
feel. Extend TEST_CASES whenever you find a real question the bot got wrong
in production — that turns every bug report into a permanent regression
check instead of a one-off fix.

Usage:
    python eval_qa.py
"""
import os
from retriever import ManualRetriever

TEST_CASES = [
    {
        "question": "For external grant purchasing, above what amount does the internal purchasing procedure apply?",
        "must_contain": "RM100K",
    },
    {
        "question": "What is the threshold amount for Asset Tagging registration?",
        "must_contain": "RM500",
    },
    {
        "question": "What is the maximum amount for the Pay-and-Claim Pre Approval Request method?",
        "must_contain": "RM1000",
    },
    {
        "question": "When should the Project Leader start preparing the EOP report?",
        "must_contain": "3 months",
    },
    {
        "question": "Who should I contact if I have a technical issue in the RMS system?",
        "must_contain": "Service Desk",
    },
    {
        "question": "How do I apply for a grant extension?",
        "must_contain": "3 months",
    },
    {
        "question": "How do I add a new member to a project?",
        "must_contain": "Add/Remove",
    },
]


def run(top_k=5):
    retriever = ManualRetriever(groq_api_key=os.environ.get("GROQ_API_KEY"))
    passed = 0
    for case in TEST_CASES:
        results = retriever.search(case["question"], top_k=top_k)
        combined = "\n".join(r["text"] for r in results)
        ok = case["must_contain"].lower() in combined.lower()
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        top_titles = ", ".join(f"{r['title']} ({r['score']:.2f})" for r in results[:3])
        print(f"[{status}] {case['question']}")
        print(f"    top hits: {top_titles}")
        if not ok:
            print(f"    expected to find: {case['must_contain']!r}")
    print(f"\n{passed}/{len(TEST_CASES)} passed")
    return passed == len(TEST_CASES)


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
