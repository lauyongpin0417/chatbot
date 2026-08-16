"""
Standalone local test script — lets you check what the new Groq vision-based
PDF page transcription produces for specific pages, WITHOUT running the full
Streamlit app. Needs internet access and a Groq API key (this calls the real
Groq API, so it counts against your free-tier rate limit, same as chat does).

One-time setup (Windows, PowerShell):
    setx GROQ_API_KEY "your-key-here"
Then CLOSE and REOPEN your terminal/VSCode so the new environment variable
is picked up (setx doesn't apply to windows already open).

Usage (from the folder containing this script, chunking.py, and your PDF):
    python test_vision_ocr.py "RMS_User_Manual.pdf" 44
    python test_vision_ocr.py "RMS_User_Manual.pdf" 44 45 46   (multiple pages)

Page numbers are 1-indexed (i.e. "page 44" = the 44th page as you'd count
it in a PDF viewer), to match what you see when eyeballing the PDF.
"""
import sys
import os
import pymupdf
from groq import Groq

from chunking import _vision_ocr_page


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_vision_ocr.py <path_to_pdf> <page_num> [page_num ...]")
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY environment variable is not set.")
        print('Run: setx GROQ_API_KEY "your-key-here"')
        print("Then close and reopen your terminal (setx needs a fresh session), and try again.")
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_nums = [int(p) for p in sys.argv[2:]]

    client = Groq(api_key=api_key)
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)

    for page_num in page_nums:
        if page_num < 1 or page_num > total_pages:
            print(f"Skipping page {page_num} — PDF only has {total_pages} pages.")
            continue

        page = doc[page_num - 1]  # pymupdf is 0-indexed internally

        print("=" * 70)
        print(f"PAGE {page_num}")
        print("=" * 70)

        text = _vision_ocr_page(page, client)
        print(text if text.strip() else "(empty — check the warning above, if any)")
        print()

    doc.close()


if __name__ == "__main__":
    main()