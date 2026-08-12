"""
Loads and chunks documents from a knowledge folder, supporting .md, .pdf, and .docx.
To add a new document: just drop the file into the knowledge_docs/ folder and
push to GitHub — no code changes needed.
"""
import os
import re
from docx import Document
import pymupdf
import io

KNOWLEDGE_DIR = "knowledge_docs"

# Gemini vision model used to "read" PDF pages like a human would, rather
# than traditional OCR — this preserves table/column order far better,
# since the model sees the whole page layout at once instead of stitching
# together individually-recognized characters.
GEMINI_MODEL = "gemini-2.5-flash"
PAGE_TRANSCRIBE_PROMPT = """Transcribe the content of this document page into clean markdown.

Rules:
- If the page contains a flow diagram or a stage/status/timeline table, output it as a
  proper markdown table with the correct column order, matching what is visually shown
  left-to-right in the image. Double check column alignment before finalizing.
- Preserve headings, step numbers, and bullet lists as they appear.
- Transcribe all visible text, including text inside screenshots, form fields, and diagrams.
- Do not summarize or omit content — this is a full transcription, not a summary.
- Do not add commentary or explanation outside the transcribed content."""


def _read_markdown(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _get_gemini_client():
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your Streamlit Secrets to enable "
            "vision-based PDF reading."
        )
    return genai.Client(api_key=api_key)


def _transcribe_page_with_vision(client, page):
    """Renders a PDF page to an image and asks Gemini to transcribe it directly,
    preserving table/column structure — similar to how a human would read it."""
    from google.genai import types
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            PAGE_TRANSCRIBE_PROMPT,
        ],
    )
    return response.text or ""


def _read_pdf(filepath):
    """
    Extracts text from a PDF by having a vision model 'read' each page
    directly (like a human would), rather than using traditional OCR. This
    gives much more reliable results for pages containing flow diagrams or
    multi-column tables, since the model sees the full page layout at once.
    """
    client = _get_gemini_client()
    doc = pymupdf.open(filepath)
    pages_text = []
    for page in doc:
        text = _transcribe_page_with_vision(client, page).strip()
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


LOADERS = {
    ".md": _read_markdown,
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
