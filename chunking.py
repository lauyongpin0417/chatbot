"""
Loads and chunks documents from a knowledge folder, supporting .md, .pdf, and .docx.
To add a new document: just drop the file into the knowledge_docs/ folder and
push to GitHub — no code changes needed.

PDF pages are handled in three tiers, cheapest first:
  1. Pages with a real embedded text layer -> extracted directly (free, instant,
     perfectly accurate). Most "digital" PDF pages fall here.
  2. Pages with no text layer (i.e. the page is a scanned image) -> read with
     local Tesseract OCR (free, unlimited, runs on-device). Good for plain
     paragraphs, not reliable for complex multi-column diagrams.
  3. Pages you've explicitly flagged in VISION_PAGES below (e.g. flow-diagram
     pages that Tesseract garbles) -> transcribed with a free Groq vision
     model, which actually understands the diagram's layout. This is the
     only tier that costs API tokens, so it's reserved for the handful of
     pages that actually need it.
"""
import os
import re
import time
import base64
import numpy as np
from docx import Document
import pymupdf
import pytesseract
from PIL import Image
import io
from groq import Groq

KNOWLEDGE_DIR = "knowledge_docs"

# --- Tier 3 config: list the 1-indexed page numbers (per filename, as it
# appears in knowledge_docs/) that need the vision model — typically flow
# diagrams or other complex multi-column layouts that Tesseract can't read
# in the right order. Everything else is handled by tiers 1/2 for free.
# Example: "RMS_USER_MANUALv7.pdf": [44, 45, 67, 89]
VISION_PAGES = {
    # "RMS_USER_MANUALv7.pdf": [44],
}

MIN_NATIVE_TEXT_CHARS = 20  # below this, treat the page as having no real text layer

OCR_LANGUAGES = "eng"  # add "+chi_sim+chi_tra" if your PDFs contain Chinese text

VISION_MODEL = "qwen/qwen3.6-27b"  # Groq's free multimodal model, used only for VISION_PAGES

VISION_PROMPT = """Transcribe all text content from this page image into clean, well-structured plain text.

Rules:
- Preserve the reading order and logical structure exactly as a human reading the page would.
- If the page contains a flow diagram or process chart made of boxes/arrows arranged in a row or
  grid, describe it as a numbered sequence of stages in the correct order (left-to-right, or
  top-to-bottom if stacked). List ALL bullet points under each stage completely before moving on
  to the next stage — never interleave text from different boxes/columns.
- If the page contains a table, reproduce it as a markdown table.
- Preserve section headings and numbering exactly as shown (e.g. "4.4.1 EOP Flow").
- The page may contain a mix of English and other languages (e.g. Chinese) — transcribe each
  piece of text in its original language, do not translate.
- Do not add commentary, explanations, or any information not present in the image.
- Do not skip visible text, including footers, page numbers, or captions.
- Output only the transcribed content, nothing else — no preamble like "Here is the text:".
"""


def _read_markdown(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _ocr_page(page, min_gap_px=15, bin_px=5, line_tol=12):
    """
    Tier 2: local, free, unlimited OCR via Tesseract for image-only pages
    that are NOT in VISION_PAGES. Uses word-level bounding boxes to detect
    multi-column layouts and read them in the correct left-to-right,
    top-to-bottom order (plain image_to_string regularly scrambles this).

    This is a heuristic and not foolproof — it depends on there being a
    real empty gap (>= min_gap_px) between columns, so it can still fail on
    diagrams with narrow gutters or busy borders. If a page still comes out
    wrong after this, add its page number to VISION_PAGES instead of tuning
    these parameters further.
    """
    pix = page.get_pixmap(dpi=250)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))

    data = pytesseract.image_to_data(img, lang=OCR_LANGUAGES, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        left, top, w = data["left"][i], data["top"][i], data["width"][i]
        words.append({"text": text, "left": left, "top": top, "right": left + w})

    if not words:
        return ""

    min_left = min(w["left"] for w in words)
    max_right = max(w["right"] for w in words)
    n_bins = (max_right - min_left) // bin_px + 1
    density = np.zeros(n_bins, dtype=int)
    for w in words:
        b0, b1 = (w["left"] - min_left) // bin_px, (w["right"] - min_left) // bin_px
        density[b0:b1 + 1] += 1

    empty = density == 0
    boundaries = []
    run_start = None
    for i, is_empty in enumerate(empty):
        if is_empty and run_start is None:
            run_start = i
        elif not is_empty and run_start is not None:
            if (i - run_start) * bin_px >= min_gap_px:
                boundaries.append((run_start + i) // 2 * bin_px + min_left)
            run_start = None
    boundaries = [min_left - 1] + boundaries + [max_right + 1]

    columns = [[] for _ in range(len(boundaries) - 1)]
    for w in words:
        for ci in range(len(boundaries) - 1):
            if boundaries[ci] <= w["left"] < boundaries[ci + 1]:
                columns[ci].append(w)
                break

    page_lines = []
    for col in columns:
        if not col:
            continue
        col_sorted = sorted(col, key=lambda w: w["top"])
        line_groups = []
        for w in col_sorted:
            placed = False
            for g in line_groups:
                if abs(g[0]["top"] - w["top"]) < line_tol:
                    g.append(w)
                    placed = True
                    break
            if not placed:
                line_groups.append([w])
        for g in line_groups:
            g_sorted = sorted(g, key=lambda w: w["left"])
            page_lines.append(" ".join(w["text"] for w in g_sorted))
        page_lines.append("")

    return "\n".join(page_lines).strip()


def _vision_ocr_page(page, client, dpi=200, max_retries=3):
    """
    Tier 3: renders a page to an image and asks a vision-capable LLM (via
    Groq's free API) to transcribe it, preserving reading order and
    structure. Only used for pages listed in VISION_PAGES, since each call
    costs a meaningful chunk of Groq's free daily token budget for this
    model — running it on every page of a large PDF will exhaust that
    budget well before the document is done (each page costs roughly
    3,500-4,700 tokens; the free tier is 200,000 tokens/day for this model).
    """
    pix = page.get_pixmap(dpi=dpi)
    png_bytes = pix.tobytes("png")
    b64_image = base64.b64encode(png_bytes).decode("utf-8")

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                            },
                        ],
                    }
                ],
                temperature=0,
                max_completion_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  Warning: vision transcription failed for a page after {max_retries} attempts: {e}")
                return ""
            wait_s = 10 * (attempt + 1)
            print(f"  Vision API call failed ({e}), retrying in {wait_s}s...")
            time.sleep(wait_s)


def _read_pdf(filepath, groq_api_key=None):
    """
    Extracts text from a PDF page by page, using the cheapest tier that
    works for each page (see module docstring). A Groq API key is only
    required if this PDF has pages listed in VISION_PAGES.
    """
    filename = os.path.basename(filepath)
    vision_page_nums = set(VISION_PAGES.get(filename, []))

    doc = pymupdf.open(filepath)
    pages_text = []
    client = None  # created lazily, only if a vision page is actually hit

    for i, page in enumerate(doc):
        page_num = i + 1

        native_text = page.get_text().strip()
        if len(native_text) >= MIN_NATIVE_TEXT_CHARS:
            pages_text.append(native_text)
            continue

        if page_num in vision_page_nums:
            if client is None:
                if not groq_api_key:
                    groq_api_key = os.environ.get("GROQ_API_KEY")
                if not groq_api_key:
                    raise ValueError(
                        f"{filename} has page {page_num} listed in VISION_PAGES, which "
                        "requires a Groq API key. Pass groq_api_key=... to "
                        "load_and_chunk_knowledge_folder(), or set the GROQ_API_KEY "
                        "environment variable."
                    )
                client = Groq(api_key=groq_api_key)
            print(f"  {filename} page {page_num}: no text layer, using vision model (configured)...")
            text = _vision_ocr_page(page, client)
        else:
            print(f"  {filename} page {page_num}: no text layer, using local OCR...")
            text = _ocr_page(page)

        pages_text.append(text)

    doc.close()
    return "\n\n".join(pages_text)


def _read_docx(filepath):
    doc = Document(filepath)
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            style = (para.style.name or "").lower()
            if "heading 1" in style:
                parts.append(f"# {para.text}")
            elif "heading 2" in style:
                parts.append(f"## {para.text}")
            elif "heading 3" in style:
                parts.append(f"### {para.text}")
            else:
                parts.append(para.text)
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append("| " + " | ".join(cells) + " |")
        if rows:
            parts.append("\n".join(rows))
    return "\n\n".join(parts)


def _read_txt(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


LOADERS = {
    ".md": _read_markdown,
    ".txt": _read_txt,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


PAGE_MARKER_RE = re.compile(r"<!--\s*Page\s+\d+\s*-->", re.IGNORECASE)
HEADER_LINE_RE = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)
# Vision-transcribed pages use bold pseudo-headers like "**Section 3.1.1:** ..."
# instead of real markdown "#" headers — this catches those too.
BOLD_HEADER_RE = re.compile(r"^\*\*([^*]{3,80})\*\*", re.MULTILINE)


def _extract_chunk_title(chunk_text, source_name):
    m = HEADER_LINE_RE.search(chunk_text)
    if m:
        return m.group(1).strip()
    m = BOLD_HEADER_RE.search(chunk_text)
    if m:
        return m.group(1).strip().rstrip(":").strip()
    for line in chunk_text.split("\n"):
        line = line.strip()
        if line and not PAGE_MARKER_RE.match(line):
            return re.sub(r"^#+\s*", "", line)[:80]
    return source_name


def _split_text_into_chunks(text, source_name, max_chunk_chars=1800, overlap_chars=200):
    # Split on "<!-- Page N -->" markers first (inserted by transcribe_full_pdf.py)
    # — these are reliable, code-generated boundaries. Without this, a
    # vision-transcribed document with no real "#" headers (just bold
    # pseudo-headers) gets treated as ONE giant section spanning the whole
    # file, which then gets blindly sliced every max_chunk_chars characters
    # with no regard for page or section boundaries at all.
    page_sections = PAGE_MARKER_RE.split(text)
    page_sections = [s.strip() for s in page_sections if s.strip()]
    if not page_sections:
        page_sections = [text]

    header_pattern = re.compile(r"(?=^#{1,4}\s)", re.MULTILINE)
    all_sections = []
    for page_text in page_sections:
        sub_sections = header_pattern.split(page_text)
        sub_sections = [s.strip() for s in sub_sections if s.strip()]
        all_sections.extend(sub_sections if sub_sections else [page_text])

    chunks = []
    for section in all_sections:
        if len(section) <= max_chunk_chars:
            chunks.append(section)
        else:
            start = 0
            while start < len(section):
                end = start + max_chunk_chars
                chunks.append(section[start:end])
                start = end - overlap_chars

    result = []
    for chunk in chunks:
        stripped = chunk.strip()
        if not stripped:
            continue
        title = _extract_chunk_title(stripped, source_name)
        result.append({
            "title": title,
            "text": chunk,
            "source": source_name,
        })
    return result


def load_and_chunk_knowledge_folder(folder_path=KNOWLEDGE_DIR, groq_api_key=None):
    """Loads every supported file in the folder and returns combined chunks."""
    all_chunks = []
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Knowledge folder '{folder_path}' not found. "
            f"Create it and add your .md/.pdf/.docx files there."
        )

    files = sorted(os.listdir(folder_path))
    loaded_any = False
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in LOADERS:
            continue
        filepath = os.path.join(folder_path, filename)
        if ext == ".pdf":
            text = _read_pdf(filepath, groq_api_key=groq_api_key)
        else:
            text = LOADERS[ext](filepath)
        if not text.strip():
            continue
        chunks = _split_text_into_chunks(text, source_name=filename)
        all_chunks.extend(chunks)
        loaded_any = True

    if not loaded_any:
        raise FileNotFoundError(
            f"No supported files (.md, .pdf, .docx) found in '{folder_path}'."
        )
    return all_chunks


def load_and_chunk_markdown(filepath, max_chunk_chars=1800, overlap_chars=200):
    text = _read_markdown(filepath)
    return _split_text_into_chunks(text, source_name=os.path.basename(filepath),
                                    max_chunk_chars=max_chunk_chars, overlap_chars=overlap_chars)


if __name__ == "__main__":
    chunks = load_and_chunk_knowledge_folder()
    print(f"Total chunks: {len(chunks)}")
    sources = set(c["source"] for c in chunks)
    print(f"Loaded from files: {sources}")