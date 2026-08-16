"""
Transcribes an entire PDF, page by page, using Groq's free vision model —
saving progress after EVERY page, and rotating across multiple Groq API
keys (e.g. from separate free accounts) when one hits its daily token
limit. This means you never lose work: if all your keys get rate-limited
partway through a large PDF, just re-run the script later (same day once
quota resets, or the next day, or with a freshly added key) and it picks
up exactly where it left off.

Setup:
    Put one or more Groq API keys in GROQ_API_KEYS, comma-separated
    (no spaces around the comma):
        setx GROQ_API_KEYS "key1,key2"
    A single key also works:
        setx GROQ_API_KEYS "key1"
    Then CLOSE and REOPEN your terminal (setx doesn't apply to windows
    already open).

Usage:
    python transcribe_full_pdf.py "knowledge_docs\\RMS_USER_MANUALv7.pdf"

Output:
    - <pdfname>_checkpoint.json — progress so far (page number -> text).
      Re-running the script on the same PDF resumes from this automatically.
    - <pdfname>_transcribed.md — the combined markdown, (re)written after
      every page, so you always have an up-to-date copy even if the run
      gets interrupted partway through.

Once it's fully done and you've spot-checked the .md file:
    1. Put <pdfname>_transcribed.md into knowledge_docs/
    2. Remove the original .pdf from knowledge_docs/ (avoid duplicate/
       conflicting content in the knowledge base)
    3. Push to GitHub
"""
import sys
import os
import json
import time
import base64
import pymupdf
from groq import Groq

VISION_MODEL = "qwen/qwen3.6-27b"

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


def _is_rate_limit_error(e):
    msg = str(e)
    return "429" in msg or "rate_limit" in msg.lower()


def _transcribe_page(client, page, dpi=200):
    pix = page.get_pixmap(dpi=dpi)
    png_bytes = pix.tobytes("png")
    b64_image = base64.b64encode(png_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                ],
            }
        ],
        temperature=0,
        max_completion_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def _load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


def _save_checkpoint(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in data.items()}, f, ensure_ascii=False, indent=2)


def _write_markdown(out_path, data, total_pages):
    parts = []
    for page_num in range(1, total_pages + 1):
        text = data.get(page_num, "")
        parts.append(f"<!-- Page {page_num} -->\n\n{text}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(parts))


def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_full_pdf.py <path_to_pdf>")
        sys.exit(1)

    keys_env = os.environ.get("GROQ_API_KEYS", "")
    api_keys = [k.strip() for k in keys_env.split(",") if k.strip()]
    if not api_keys:
        print("GROQ_API_KEYS environment variable is not set.")
        print('Run: setx GROQ_API_KEYS "key1,key2"   (a single key also works)')
        print("Then close and reopen your terminal, and try again.")
        sys.exit(1)

    pdf_path = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    checkpoint_path = f"{base_name}_checkpoint.json"
    out_path = f"{base_name}_transcribed.md"

    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)

    data = _load_checkpoint(checkpoint_path)
    already_done = sum(1 for v in data.values() if v.strip())
    print(f"{pdf_path}: {total_pages} pages total, {already_done} already done (resuming from checkpoint).")
    print(f"Using {len(api_keys)} API key(s).\n")

    key_index = 0
    client = Groq(api_key=api_keys[key_index])

    for i in range(total_pages):
        page_num = i + 1
        if data.get(page_num, "").strip():
            continue  # already done in a previous run, skip

        page = doc[i]
        print(f"  Page {page_num}/{total_pages}...", end=" ", flush=True)

        while True:
            try:
                text = _transcribe_page(client, page)
                data[page_num] = text
                _save_checkpoint(checkpoint_path, data)  # save after EVERY page
                _write_markdown(out_path, data, total_pages)  # keep .md up to date too
                print("done")
                break
            except Exception as e:
                if _is_rate_limit_error(e):
                    key_index += 1
                    if key_index < len(api_keys):
                        print(f"key #{key_index} rate-limited, switching to key #{key_index + 1}...")
                        client = Groq(api_key=api_keys[key_index])
                        continue  # retry same page with the new key
                    else:
                        print("rate limit hit on all keys.")
                        _save_checkpoint(checkpoint_path, data)
                        _write_markdown(out_path, data, total_pages)
                        done_count = sum(1 for v in data.values() if v.strip())
                        print(f"\nProgress saved: {done_count}/{total_pages} pages done.")
                        print(f"Checkpoint: {checkpoint_path}  |  Markdown so far: {out_path}")
                        print("Re-run this exact command later (once a key's daily quota resets, "
                              "or after adding a fresh key to GROQ_API_KEYS) to continue — "
                              "already-done pages will be skipped automatically.")
                        doc.close()
                        sys.exit(0)
                else:
                    print(f"failed ({e}) — leaving this page blank for now, will retry next run.")
                    data[page_num] = ""
                    _save_checkpoint(checkpoint_path, data)
                    break

        time.sleep(1)  # be gentle on rate limits between pages

    doc.close()
    _write_markdown(out_path, data, total_pages)
    print(f"\nAll {total_pages} pages done. Saved to {out_path}")
    print("Please skim through the file to spot-check accuracy before using it.")


if __name__ == "__main__":
    main()