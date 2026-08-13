"""
Loads and chunks documents from a knowledge folder, supporting .md, .pdf, and .docx.
To add a new document: just drop the file into the knowledge_docs/ folder and
push to GitHub — no code changes needed.
"""
import os
import re
from docx import Document
import pymupdf
import pytesseract
from PIL import Image
import io
import numpy as np

KNOWLEDGE_DIR = "knowledge_docs"
OCR_LANGUAGES = "eng"  # add "+chi_sim+chi_tra" if your PDFs contain Chinese text


def _read_markdown(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _ocr_page(page, min_gap_px=15, bin_px=5, line_tol=12):
    """
    Renders a PDF page to an image and runs OCR on it, using Tesseract's
    word-level bounding boxes to detect multi-column layouts and read them
    in the correct left-to-right, top-to-bottom order.

    Why not pytesseract.image_to_string(): that method relies on Tesseract's
    own page-layout analysis to decide reading order, which regularly fails
    on flow-diagram-style pages (several boxes/arrows side by side, each
    with its own bullets) — it tends to read across columns row-by-row
    instead of finishing one column before starting the next, scrambling
    the step order. This function instead finds actual vertical whitespace
    gaps between columns of text and reads column-by-column.

    This is a heuristic and not foolproof: it depends on there being a
    real empty gap (>= min_gap_px) between columns, so it can still fail
    on diagrams with narrow gutters, overlapping shapes, or columns that
    don't line up cleanly. For pages where this still produces poor
    results, manually converting that page to markdown remains the most
    reliable fallback (see README).
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

    # Build a horizontal density histogram to find empty vertical gaps
    # between columns of text.
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

    # Assign each word to a column based on which gap-bounded region it falls in.
    columns = [[] for _ in range(len(boundaries) - 1)]
    for w in words:
        for ci in range(len(boundaries) - 1):
            if boundaries[ci] <= w["left"] < boundaries[ci + 1]:
                columns[ci].append(w)
                break

    # Within each column (left to right), group words into lines by
    # vertical position, then read lines top to bottom.
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
        page_lines.append("")  # blank line between columns for readability

    return "\n".join(page_lines).strip()


def _read_pdf(filepath):
    """
    Extracts text from a PDF by rendering every page as an image and running
    OCR on it. This is fully local and free (no external API calls), so it
    doesn't fail from rate limits or server outages. Trade-off: complex
    multi-column flow-diagram tables may come out with scrambled row/column
    order — for those, manually converting to markdown is still the most
    reliable approach.
    """
    doc = pymupdf.open(filepath)
    pages_text = []
    for page in doc:
        text = _ocr_page(page).strip()
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


def _split_text_into_chunks(text, source_name, max_chunk_chars=1800, overlap_chars=200):
    header_pattern = re.compile(r"(?=^#{1,4}\s)", re.MULTILINE)
    sections = header_pattern.split(text)
    sections = [s.strip() for s in sections if s.strip()]

    if not sections:
        sections = [text]

    chunks = []
    for section in sections:
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
        if not chunk.strip():
            continue
        first_line = chunk.strip().split("\n")[0]
        title = re.sub(r"^#+\s*", "", first_line).strip()
        result.append({
            "title": title if title else source_name,
            "text": chunk,
            "source": source_name,
        })
    return result


def load_and_chunk_knowledge_folder(folder_path=KNOWLEDGE_DIR):
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