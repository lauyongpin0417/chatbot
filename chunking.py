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

KNOWLEDGE_DIR = "knowledge_docs"
OCR_LANGUAGES = "eng"  # add "+chi_sim+chi_tra" if your PDFs contain Chinese text


def _read_markdown(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _ocr_page(page):
    """Renders a PDF page to an image and runs OCR on it."""
    pix = page.get_pixmap(dpi=250)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    return pytesseract.image_to_string(img, lang=OCR_LANGUAGES)


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