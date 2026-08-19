"""
Commits an uploaded knowledge-base file straight to the GitHub repo via the
Contents API, instead of writing it to local disk.

This matters specifically because Streamlit Community Cloud's container
filesystem is EPHEMERAL — anything written to disk while the app is running
(e.g. from st.file_uploader) is lost the moment the app sleeps, redeploys,
or the container recycles. GitHub is the only persistent store in this
setup, so "let a user upload a file" only works if the upload ends as a
real commit there — which is also what makes Streamlit Cloud's existing
auto-redeploy-on-push behavior pick it up for everyone automatically.

A .pdf upload is never committed as-is. It's run through
chunking._transcribe_pdf_auto() first (tier 1 native text / tier 2 local
OCR / tier 3 vision, auto-escalated per page — see that function's
docstring) and the resulting transcript is committed as a .md file. This is
what makes the upload fully self-contained: no one needs the project's code
locally, and no one needs to curate VISION_PAGES or re-run anything by
hand afterward. The trade-off is that decision is now made by a heuristic
with no human reviewing the result before it becomes live chatbot content —
see the page/vision-count caps in chunking.py for the guardrails that
imposes.

Setup (Streamlit Cloud Secrets):
    GITHUB_TOKEN = a token with write access to ONLY this repo's contents.
                   Create a fine-grained PAT (github.com/settings/tokens)
                   scoped to just this repository, permission
                   "Contents: Read and write" — nothing else. Fine-grained
                   + repo-scoped means a leaked/misused token can only ever
                   touch this one repo's files, not your whole account.
    GITHUB_REPO  = "yourusername/your-repo-name"
    GROQ_API_KEY = already required for the chatbot itself; reused here for
                   the vision-tier calls a PDF upload may trigger.
"""
import base64
import os
import re
import time

import requests

from chunking import _transcribe_pdf_auto

# PDFs get a much larger cap than text formats since they're naturally
# bigger (embedded images/scans) — the existing manual alone is ~12 MB. This
# is a cap on the UPLOADED file's size, not the resulting .md, which is
# committed instead of the raw PDF (see module docstring).
MAX_UPLOAD_BYTES_BY_EXT = {
    ".md": 5 * 1024 * 1024,
    ".txt": 5 * 1024 * 1024,
    ".docx": 5 * 1024 * 1024,
    ".pdf": 25 * 1024 * 1024,
}
ALLOWED_UPLOAD_EXTENSIONS = set(MAX_UPLOAD_BYTES_BY_EXT)
UPLOAD_TARGET_DIR = "knowledge_docs"
GITHUB_API_TIMEOUT = 30


def _safe_filename(raw_name):
    """Keep only a plain basename with safe characters. raw_name comes from
    the uploader's browser (Streamlit passes it through unvalidated), and it
    ends up as a path segment in a GitHub API call — strip any directory
    separators and anything outside a conservative allow-list so it can
    never be turned into a path-traversal or otherwise unexpected path."""
    name = os.path.basename(raw_name).strip()
    name = re.sub(r"[^A-Za-z0-9 ._-]", "_", name)
    return name or "upload"


def _github_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def _github_file_exists(repo, path, token):
    resp = requests.get(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        headers=_github_headers(token),
        timeout=GITHUB_API_TIMEOUT,
    )
    return resp.status_code == 200


def upload_knowledge_file(filename, content_bytes, token, repo, groq_api_key=None, progress_callback=None):
    """Validates an upload, transcribes it if it's a PDF, and commits the
    result into knowledge_docs/ on the `main` branch.

    progress_callback(page_num, total_pages, tier_used), if given, is
    forwarded to chunking._transcribe_pdf_auto for PDF uploads — ignored
    for every other file type, which commit immediately.

    Returns (ok: bool, message: str) — message is the final filename (with
    a short processing summary for PDFs) on success, or a human-readable
    error on failure.
    """
    if not token or not repo:
        return False, "Upload is not configured (missing GITHUB_TOKEN/GITHUB_REPO in Secrets)."

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Only .md, .txt, .docx, .pdf are accepted here."

    size_limit = MAX_UPLOAD_BYTES_BY_EXT[ext]
    if len(content_bytes) > size_limit:
        return False, (
            f"File too large ({len(content_bytes) / 1_000_000:.1f} MB). "
            f"Limit for {ext} is {size_limit / 1_000_000:.0f} MB."
        )

    if not content_bytes.strip():
        return False, "File is empty."

    commit_note = ""
    if ext == ".pdf":
        try:
            markdown_text, stats = _transcribe_pdf_auto(
                content_bytes, groq_api_key, progress_callback=progress_callback
            )
        except ValueError as e:
            return False, str(e)
        if not markdown_text.strip():
            return False, "Transcription produced no readable text from this PDF."

        stem = os.path.splitext(_safe_filename(filename))[0]
        safe_name = f"{stem}_transcribed.md"
        content_bytes = markdown_text.encode("utf-8")
        commit_note = (
            f" — {stats['total_pages']} pages "
            f"({stats['native_text_pages']} text layer, {stats['ocr_pages']} local OCR, "
            f"{stats['vision_pages_used']} vision-transcribed)"
        )
    else:
        safe_name = _safe_filename(filename)

    path = f"{UPLOAD_TARGET_DIR}/{safe_name}"

    # Anyone with the link can use this uploader, and uploads are committed
    # straight to the live knowledge base with no review step — so a name
    # collision gets a unique suffix instead of silently overwriting
    # existing (possibly manually-verified) content.
    if _github_file_exists(repo, path, token):
        stem, ext2 = os.path.splitext(safe_name)
        safe_name = f"{stem}_{int(time.time())}{ext2}"
        path = f"{UPLOAD_TARGET_DIR}/{safe_name}"

    resp = requests.put(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        headers=_github_headers(token),
        json={
            "message": f"Add knowledge file via app upload: {safe_name}",
            "content": base64.b64encode(content_bytes).decode("utf-8"),
            "branch": "main",
        },
        timeout=GITHUB_API_TIMEOUT,
    )
    if resp.status_code in (200, 201):
        return True, safe_name + commit_note
    return False, f"GitHub API error {resp.status_code}: {resp.text[:300]}"
