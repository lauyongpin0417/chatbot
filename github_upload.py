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

Setup (Streamlit Cloud Secrets):
    GITHUB_TOKEN = a token with write access to ONLY this repo's contents.
                   Create a fine-grained PAT (github.com/settings/tokens)
                   scoped to just this repository, permission
                   "Contents: Read and write" — nothing else. Fine-grained
                   + repo-scoped means a leaked/misused token can only ever
                   touch this one repo's files, not your whole account.
    GITHUB_REPO  = "yourusername/your-repo-name"
"""
import base64
import os
import re
import time

import requests

# PDFs get a much larger cap than text formats since they're naturally
# bigger (embedded images/scans) — the existing manual alone is ~12 MB.
# Deliberately no vision-model tier for uploads (see module docstring on
# .pdf below): only chunking.py's free tier-1/2 (native text layer / local
# Tesseract OCR) will ever run on an uploaded PDF, so a diagram-heavy page
# still needs the same manual transcribe_full_pdf.py + VISION_PAGES
# follow-up the main manual already goes through — this upload path can't
# fix that automatically, it only gets the file into the repo.
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


def upload_knowledge_file(filename, content_bytes, token, repo):
    """Validates and commits one file into knowledge_docs/ on the `main`
    branch. Returns (ok: bool, message: str) — message is either the final
    filename used (on success) or a human-readable error (on failure).
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

    safe_name = _safe_filename(filename)
    path = f"{UPLOAD_TARGET_DIR}/{safe_name}"

    # Anyone with the link can use this uploader, and uploads are committed
    # straight to the live knowledge base with no review step — so a name
    # collision gets a unique suffix instead of silently overwriting
    # existing (possibly manually-verified) content.
    if _github_file_exists(repo, path, token):
        stem, ext = os.path.splitext(safe_name)
        safe_name = f"{stem}_{int(time.time())}{ext}"
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
        return True, safe_name
    return False, f"GitHub API error {resp.status_code}: {resp.text[:300]}"
