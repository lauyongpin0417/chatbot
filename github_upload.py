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

Large PDFs are resumable across interrupted uploads (dropped connection,
closed browser tab, redeploy mid-transcription — a single Streamlit request
has no way to catch or clean up after any of those). Progress is
checkpointed to GitHub periodically during transcription, keyed by a hash of
the PDF's own bytes — re-uploading the exact same file finds and continues
from that checkpoint instead of starting over and re-spending vision calls
on pages already done. Checkpoints commit to a SEPARATE branch
(CHECKPOINT_BRANCH), never `main` — committing progress on `main` would
trigger Streamlit Cloud's auto-redeploy on every checkpoint, which would
restart the very app instance running the upload.

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
import hashlib
import json
import os
import re
import time

import requests

from chunking import _transcribe_pdf_auto, VisionQuotaExhausted

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

# Dedicated branch for in-progress PDF checkpoints — deliberately NOT
# `main`, so a checkpoint commit never triggers Streamlit Cloud's
# auto-redeploy (see module docstring). Never merged into main; it's just
# a place to durably park JSON blobs between requests.
CHECKPOINT_BRANCH = "kb-checkpoints"
CHECKPOINT_DIR = ".kb_checkpoints"


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


def _ensure_checkpoint_branch(repo, token):
    """Creates CHECKPOINT_BRANCH pointing at the repo's current default
    branch HEAD, if it doesn't already exist. Idempotent — safe to call on
    every upload."""
    resp = requests.get(
        f"https://api.github.com/repos/{repo}",
        headers=_github_headers(token),
        timeout=GITHUB_API_TIMEOUT,
    )
    if resp.status_code != 200:
        return False
    default_branch = resp.json().get("default_branch", "main")

    ref_resp = requests.get(
        f"https://api.github.com/repos/{repo}/git/ref/heads/{default_branch}",
        headers=_github_headers(token),
        timeout=GITHUB_API_TIMEOUT,
    )
    if ref_resp.status_code != 200:
        return False
    base_sha = ref_resp.json()["object"]["sha"]

    create_resp = requests.post(
        f"https://api.github.com/repos/{repo}/git/refs",
        headers=_github_headers(token),
        json={"ref": f"refs/heads/{CHECKPOINT_BRANCH}", "sha": base_sha},
        timeout=GITHUB_API_TIMEOUT,
    )
    # 201 = created, 422 = "Reference already exists" — both mean the
    # branch is now there, which is all this function promises.
    return create_resp.status_code in (201, 422)


def _github_get_content(repo, path, token, ref):
    """Returns (content_bytes, sha) for a file on the given branch/ref, or
    (None, None) if it doesn't exist there."""
    resp = requests.get(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        headers=_github_headers(token),
        params={"ref": ref},
        timeout=GITHUB_API_TIMEOUT,
    )
    if resp.status_code != 200:
        return None, None
    data = resp.json()
    return base64.b64decode(data["content"]), data["sha"]


def _github_put_content(repo, path, content_bytes, token, branch, message, sha=None):
    """Creates or updates a file on `branch`. Pass the previous `sha` to
    update an existing file (required by the Contents API for updates —
    omit only when creating for the first time). Returns the new sha on
    success, or None on failure (never raises — checkpoint writes are
    best-effort, see _save_checkpoint's caller)."""
    body = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        body["sha"] = sha
    resp = requests.put(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        headers=_github_headers(token),
        json=body,
        timeout=GITHUB_API_TIMEOUT,
    )
    if resp.status_code in (200, 201):
        return resp.json()["content"]["sha"]
    return None


def _github_delete_content(repo, path, token, branch, sha, message):
    resp = requests.delete(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        headers=_github_headers(token),
        json={"message": message, "sha": sha, "branch": branch},
        timeout=GITHUB_API_TIMEOUT,
    )
    return resp.status_code == 200


def upload_knowledge_file(
    filename, content_bytes, token, repo, groq_api_key=None,
    progress_callback=None, on_quota_exhausted="pause",
):
    """Validates an upload, transcribes it if it's a PDF, and commits the
    result into knowledge_docs/ on the `main` branch.

    progress_callback(page_num, total_pages, tier_used), if given, is
    forwarded to chunking._transcribe_pdf_auto for PDF uploads — ignored
    for every other file type, which commit immediately.

    PDF uploads checkpoint their progress to CHECKPOINT_BRANCH as they go
    (see module docstring) — if this call is interrupted partway (dropped
    connection, closed tab), re-uploading the SAME file resumes from the
    last checkpoint instead of restarting and re-spending vision calls.

    on_quota_exhausted (PDF uploads only) is forwarded to
    chunking._transcribe_pdf_auto: "pause" (default) stops and checkpoints
    the moment the Groq vision quota looks exhausted, rather than silently
    finishing at lower quality — this function then returns
    needs_decision=True so the caller can offer the human a choice: leave
    it (re-upload later, once quota resets, to resume with vision) or call
    again right away with on_quota_exhausted="degrade" to finish now using
    local OCR for whatever's left.

    Returns (ok, message, needs_decision):
      - (True, final_filename_with_summary, False) on success
      - (False, human_readable_error, False) on a hard failure
      - (False, human_readable_status, True) when paused for a quota
        decision — content_bytes should be resubmitted unchanged (same
        hash = same checkpoint) once the caller decides which way to go
    """
    if not token or not repo:
        return False, "Upload is not configured (missing GITHUB_TOKEN/GITHUB_REPO in Secrets).", False

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Only .md, .txt, .docx, .pdf are accepted here.", False

    size_limit = MAX_UPLOAD_BYTES_BY_EXT[ext]
    if len(content_bytes) > size_limit:
        return False, (
            f"File too large ({len(content_bytes) / 1_000_000:.1f} MB). "
            f"Limit for {ext} is {size_limit / 1_000_000:.0f} MB."
        ), False

    if not content_bytes.strip():
        return False, "File is empty.", False

    commit_note = ""
    if ext == ".pdf":
        # Keyed by the PDF's OWN content hash, not the filename — so
        # re-uploading the exact same bytes (the natural way to "resume"
        # after a dropped connection) finds the same checkpoint regardless
        # of what the file happens to be named this time.
        file_hash = hashlib.sha256(content_bytes).hexdigest()[:16]
        checkpoint_path = f"{CHECKPOINT_DIR}/{file_hash}.json"
        checkpoint_state = {"sha": None}  # mutable box the nested closure can update

        _ensure_checkpoint_branch(repo, token)
        raw_checkpoint, checkpoint_state["sha"] = _github_get_content(
            repo, checkpoint_path, token, ref=CHECKPOINT_BRANCH
        )
        resume_pages = None
        if raw_checkpoint:
            try:
                resume_pages = json.loads(raw_checkpoint).get("pages")
            except ValueError:
                resume_pages = None  # corrupt checkpoint — safer to restart than to guess

        def _save_checkpoint(state):
            # Best-effort: a failed checkpoint write shouldn't abort an
            # otherwise-successful transcription in progress — it just
            # means resumability rewinds to the last write that DID land.
            new_sha = _github_put_content(
                repo, checkpoint_path, json.dumps(state).encode("utf-8"), token,
                branch=CHECKPOINT_BRANCH,
                message="Checkpoint: PDF transcription in progress",
                sha=checkpoint_state["sha"],
            )
            if new_sha:
                checkpoint_state["sha"] = new_sha

        try:
            markdown_text, stats = _transcribe_pdf_auto(
                content_bytes, groq_api_key,
                progress_callback=progress_callback,
                resume_pages=resume_pages,
                checkpoint_callback=_save_checkpoint,
                on_quota_exhausted=on_quota_exhausted,
            )
        except VisionQuotaExhausted as e:
            # Checkpoint already saved by _transcribe_pdf_auto right before
            # raising — nothing more to persist here. The final .md is
            # deliberately NOT committed: leaving main untouched means this
            # upload simply doesn't exist as a knowledge file yet, so there's
            # nothing stale to clean up whichever way the human decides.
            remaining = e.total_pages - e.pages_done
            return False, (
                f"Groq vision quota looks exhausted after {e.vision_pages_used} vision page(s) "
                f"this run — {e.pages_done}/{e.total_pages} pages done and checkpointed "
                f"({remaining} left). Re-upload the same file later to resume with vision once "
                "the quota resets, or finish now using local OCR for the rest."
            ), True
        except ValueError as e:
            return False, str(e), False
        if not markdown_text.strip():
            return False, "Transcription produced no readable text from this PDF.", False

        # Transcription finished in this same request — the checkpoint has
        # done its job, so clean it up rather than leaving it to accumulate.
        if checkpoint_state["sha"]:
            _github_delete_content(
                repo, checkpoint_path, token, branch=CHECKPOINT_BRANCH,
                sha=checkpoint_state["sha"], message="Checkpoint: transcription complete",
            )

        stem = os.path.splitext(_safe_filename(filename))[0]
        safe_name = f"{stem}_transcribed.md"
        content_bytes = markdown_text.encode("utf-8")
        resumed_note = f", resumed from a previous attempt" if resume_pages else ""
        commit_note = (
            f" — {stats['total_pages']} pages "
            f"({stats['native_text_pages']} text layer, {stats['ocr_pages']} local OCR, "
            f"{stats['vision_pages_used']} vision-transcribed{resumed_note})"
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
        return True, safe_name + commit_note, False
    return False, f"GitHub API error {resp.status_code}: {resp.text[:300]}", False
