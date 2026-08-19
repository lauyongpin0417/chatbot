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

# Reasoning-capable vision models can emit private work inside <think> tags.
# Treat an unclosed opening tag as extending to the end of the response: a
# truncated reasoning block is never valid manual content.
THINKING_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?(?:</think\s*>|\Z)", re.IGNORECASE | re.DOTALL)
TRANSCRIPT_PAGE_MARKER_RE = re.compile(r"(<!--\s*Page\s+\d+\s*-->)", re.IGNORECASE)
def strip_thinking_content(text):
    """Remove model reasoning blocks before text can be chunked or embedded."""
    # A broken tag must not remove the remainder of a multi-page transcript.
    parts = TRANSCRIPT_PAGE_MARKER_RE.split(text or "")
    return "".join(
        part if TRANSCRIPT_PAGE_MARKER_RE.fullmatch(part) else THINKING_BLOCK_RE.sub("", part)
        for part in parts
    )


def _read_markdown(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return strip_thinking_content(f.read())


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
                reasoning_effort="none",
                reasoning_format="hidden",
                max_completion_tokens=4096,
            )
            return strip_thinking_content(response.choices[0].message.content).strip()
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
    return strip_thinking_content("\n\n".join(pages_text))


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
        return strip_thinking_content(f.read())


LOADERS = {
    ".md": _read_markdown,
    ".txt": _read_txt,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


PAGE_MARKER_RE = re.compile(r"<!--\s*Page\s+\d+\s*-->", re.IGNORECASE)
HEADER_LINE_RE = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)
# Vision-transcribed pages use inconsistent header styling from page to page
# (some use "**Section 3.1.1:**" bold, others use a plain bulleted/quoted
# line like '*   "7.5 Goods Receipt and Asset Tagging"' with no bold at
# all) — since the model's formatting choice isn't reliable, the most
# robust signal is the section NUMBER pattern itself ("7.5 Some Title"),
# which the document's actual structure guarantees is consistent.
SECTION_LINE_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?\s+[A-Z].{2,70}$")
# Generated manuals do not consistently use Markdown headings, but their
# numbered RMS section headings are reliable boundaries. Split these before
# the size fallback so Purchasing cannot share an embedding with the
# neighbouring Pay-and-Claim (<RM1K) section.
SECTION_SPLIT_RE = re.compile(
    r"(?=^(?:#{1,4}\s+)?\d+\.\d+(?:\.\d+)?\s+[A-Z][^\n]{2,120}$)",
    re.MULTILINE,
)
BOLD_HEADER_RE = re.compile(r"^\*\*([^*]{3,80})\*\*", re.MULTILINE)

# FAQ files (e.g. "Frequently Asked Questions - RMS.txt") are typically a flat
# numbered list of "N. Q: ... A: ..." pairs with no markdown "#" headers at
# all. Without this, the generic splitter below treats the whole file as ONE
# section (or blindly character-slices it if it's long), which means dozens
# of unrelated Q&A pairs get embedded together as a single vector — the
# retriever can find "the FAQ file" but not "this specific question", and the
# model then has to guess which of the many answers buried in that blob is
# the right one. Splitting on the "N. Q:" boundary gives each Q&A pair its
# own chunk, titled with the question text itself, so retrieval can match
# the actual question being asked.
FAQ_QA_SPLIT_RE = re.compile(r"(?=^\s*\d+\.\s+\**Q\s*:)", re.MULTILINE)
# DOTALL + non-greedy so this captures the full question text even when it
# wraps across multiple lines in the source file, stopping right before the
# "A:" line rather than at the first newline within the question itself.
FAQ_Q_LINE_RE = re.compile(r"^\s*\d+\.\s+\**Q\s*:\s*(.+?)\s*\n\s*\**A\s*:", re.MULTILINE | re.DOTALL)


def _looks_like_faq(text):
    """At least 3 numbered 'Q:' entries is a reasonable bar for 'this file is
    a FAQ list', low enough to catch short files but high enough to not
    misfire on a manual that happens to mention 'Q:' once or twice."""
    return len(FAQ_QA_SPLIT_RE.findall(text)) >= 3


def _split_faq_into_chunks(text, source_name):
    blocks = FAQ_QA_SPLIT_RE.split(text)
    blocks = [b.strip() for b in blocks if b.strip()]
    result = []
    for block in blocks:
        m = FAQ_Q_LINE_RE.search(block)
        title = re.sub(r"\s+", " ", m.group(1)).strip() if m else _extract_chunk_title(block, source_name)
        result.append({"title": title, "text": block, "source": source_name})
    return result


def _find_section_heading_line(chunk_text):
    """
    Looks for a line that IS ENTIRELY a numbered section heading (e.g.
    "7.5 Goods Receipt and Asset Tagging"), after stripping common bullet/
    quote wrapping — not just a section number mentioned in passing inside
    a longer sentence (e.g. "...see 7.2.1 Pre-Approval claim)." would NOT
    match, since that's a cross-reference, not this chunk's own heading).
    """
    for line in chunk_text.split("\n"):
        stripped = line.strip().lstrip("*-").strip()
        stripped = stripped.strip('"').strip()
        if SECTION_LINE_RE.match(stripped):
            return stripped
    return None


def _extract_chunk_title(chunk_text, source_name):
    m = HEADER_LINE_RE.search(chunk_text)
    if m:
        return m.group(1).strip()
    heading = _find_section_heading_line(chunk_text)
    if heading:
        return heading
    # Prefer a bold pseudo-header that looks like a real section marker
    # (contains a digit, e.g. "Section 3.1.1") over generic model preamble
    # like "**Analyze the image:**" or "**List:**" that sometimes appears
    # first in a vision-transcribed page.
    for m in BOLD_HEADER_RE.finditer(chunk_text):
        candidate = m.group(1).strip().rstrip(":").strip()
        if re.search(r"\d", candidate):
            return candidate
    m = BOLD_HEADER_RE.search(chunk_text)
    if m:
        return m.group(1).strip().rstrip(":").strip()
    for line in chunk_text.split("\n"):
        line = line.strip()
        if line and not PAGE_MARKER_RE.match(line):
            return re.sub(r"^#+\s*", "", line)[:80]
    return source_name



def _split_text_into_chunks(text, source_name, max_chunk_chars=6000, overlap_chars=300):
    # Split on numbered section headings across the WHOLE document first, not
    # per-page. A table or flow-diagram enumeration that continues onto the
    # next page has no heading of its own on that second page, so splitting
    # by "<!-- Page N -->" markers before splitting by heading used to tear
    # it into two chunks — retrieval would then only surface half of it.
    all_sections = SECTION_SPLIT_RE.split(text)
    all_sections = [s.strip() for s in all_sections if s.strip()]
    if not all_sections:
        all_sections = [text]

    # A section with no numbered heading of its own (e.g. unnumbered front
    # matter/TOC spanning many pages) can still come out huge with nothing
    # to split on internally. For those — and only those — fall back to
    # "<!-- Page N -->" markers (inserted by transcribe_full_pdf.py) as a
    # secondary boundary, before resorting to blind character-slicing.
    refined_sections = []
    for section in all_sections:
        if len(section) <= max_chunk_chars:
            refined_sections.append(section)
            continue
        page_pieces = PAGE_MARKER_RE.split(section)
        page_pieces = [p.strip() for p in page_pieces if p.strip()]
        refined_sections.extend(page_pieces if len(page_pieces) > 1 else [section])
    all_sections = refined_sections

    # A whole section (a page, or a "#"-delimited region within a page) stays
    # as ONE chunk regardless of length, as long as it's under the safety
    # ceiling — this is what actually prevents mid-section cuts, rather than
    # hoping max_chunk_chars is "big enough". Character-slicing only kicks
    # in for the rare pathological section that exceeds even that ceiling,
    # and when it does, every slice keeps the section's own title attached
    # so a later slice never loses its identity (which is also embedded
    # together with the text in retriever.py — see there for why that matters).
    chunks = []  # list of (chunk_text, forced_title_or_None)
    for section in all_sections:
        if len(section) <= max_chunk_chars:
            chunks.append((section, None))
        else:
            title = _extract_chunk_title(section, source_name)
            start = 0
            first = True
            while start < len(section):
                end = start + max_chunk_chars
                piece = section[start:end]
                if not first:
                    piece = f"[continued — {title}]\n\n{piece}"
                chunks.append((piece, title))  # reuse the same clean title for every slice
                start = end - overlap_chars
                first = False

    result = []
    for chunk, forced_title in chunks:
        stripped = chunk.strip()
        if not stripped:
            continue
        title = forced_title if forced_title else _extract_chunk_title(stripped, source_name)
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
        if ext == ".txt" and _looks_like_faq(text):
            chunks = _split_faq_into_chunks(text, source_name=filename)
        else:
            chunks = _split_text_into_chunks(text, source_name=filename)
        all_chunks.extend(chunks)
        loaded_any = True

    if not loaded_any:
        raise FileNotFoundError(
            f"No supported files (.md, .pdf, .docx) found in '{folder_path}'."
        )
    return all_chunks


def load_and_chunk_markdown(filepath, max_chunk_chars=3500, overlap_chars=300):
    text = _read_markdown(filepath)
    return _split_text_into_chunks(text, source_name=os.path.basename(filepath),
                                    max_chunk_chars=max_chunk_chars, overlap_chars=overlap_chars)


if __name__ == "__main__":
    chunks = load_and_chunk_knowledge_folder()
    print(f"Total chunks: {len(chunks)}")
    sources = set(c["source"] for c in chunks)
    print(f"Loaded from files: {sources}")
