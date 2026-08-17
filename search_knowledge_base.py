"""
Quick local check: does a given keyword/phrase actually exist ANYWHERE in
your knowledge_docs/ files? This tells you whether an "I don't know" answer
is because the content was never transcribed into the knowledge base at
all (content gap — needs a source-file fix), versus the content IS there
but retrieval just didn't surface it for that specific question (a
retrieval-tuning problem, not a content problem).

Usage:
    python search_knowledge_base.py "Asset Tagging"
    python search_knowledge_base.py "Vendor/Supplier"
"""
import sys
import os
import glob

KNOWLEDGE_DIR = "knowledge_docs"


def main():
    if len(sys.argv) < 2:
        print('Usage: python search_knowledge_base.py "search phrase"')
        sys.exit(1)

    phrase = sys.argv[1]
    phrase_lower = phrase.lower()

    files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*"))
    found_any = False

    for filepath in files:
        if not os.path.isfile(filepath):
            continue
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in (".md", ".txt"):
            continue  # only checking plain-text knowledge files here

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        lower_content = content.lower()
        idx = lower_content.find(phrase_lower)
        if idx != -1:
            found_any = True
            # show a bit of surrounding context
            start = max(0, idx - 100)
            end = min(len(content), idx + len(phrase) + 200)
            snippet = content[start:end].replace("\n", " ")
            print(f"FOUND in {os.path.basename(filepath)}:")
            print(f"  ...{snippet}...\n")

    if not found_any:
        print(f'"{phrase}" was NOT found in any .md/.txt file under {KNOWLEDGE_DIR}/.')
        print("This means the content genuinely isn't in the knowledge base yet —")
        print("it's likely on a page/screenshot that hasn't been transcribed.")
        print("Check which PDF page this topic is on, and whether that page needs")
        print("to be added to VISION_PAGES in chunking.py, or manually transcribed.")


if __name__ == "__main__":
    main()