# RMS Grant Procedure Guide — Free Chatbot

A free, publicly shareable chatbot that answers questions about the RMS manual,
built with Streamlit + Groq (free LLM API) + local free embeddings for retrieval.

**Important:** this runs entirely outside MMU's Microsoft ecosystem. Content is
processed by Groq's servers (not Microsoft's), and the deployed app is a public
URL (anyone with the link can use it, unless you add a password — see below).
Confirm this is acceptable with your supervisor before sharing widely.

## What's in this folder
- `app.py` — the Streamlit chat app
- `chunking.py` / `retriever.py` — loads every file in `knowledge_docs/`,
  splits it into chunks, and builds a searchable index (runs locally/free,
  no API cost for this part)
- `knowledge_docs/` — put all your knowledge source files here:
  **.md, .pdf, and .docx are all supported**. Every file in this folder gets
  loaded automatically — you don't need to edit any code to add a new one.
- `requirements.txt` — Python dependencies
- `packages.txt` — tells Streamlit Cloud to install the Tesseract OCR engine
  (a system-level program, not a Python package) — don't delete this file,
  tier 2 of PDF reading depends on it

## Adding or updating knowledge files
1. Drop your new `.md`, `.pdf`, or `.docx` file into the `knowledge_docs/`
   folder in your GitHub repo (use GitHub's "Add file > Upload files" button)
2. Streamlit Cloud auto-redeploys when it detects the change (usually within
   a minute or two)
3. That's it — no code changes needed for `.md`/`.docx` files, or for `.pdf`
   pages that have a text layer or are plain scanned text. This is currently
   the only way to update the knowledge base (there's no in-app upload
   feature for supervisors/other users to add files directly without going
   through GitHub).
4. If your PDF has flow-diagram pages (see tier 3 above), open
   `chunking.py`, find the `VISION_PAGES` dict near the top, and add an
   entry like `"YourFile.pdf": [12, 13]` with the 1-indexed page numbers of
   the diagram pages (open the PDF in a normal viewer and note the page
   numbers as shown there). Test locally with `test_vision_ocr.py` before
   pushing — see below.

Notes on format quality:
- `.md` gives the best results, since headings and tables are cleanly
  structured for retrieval.
- `.docx` works well if it uses Word's built-in Heading 1/2/3 styles and
  real Word tables — this preserves structure similarly to markdown.
- `.pdf` pages are handled in three tiers, cheapest first, so most of a PDF
  costs nothing and only the pages that truly need it use paid-tier-quality
  extraction:
  1. **Pages with a real embedded text layer** (i.e. not scanned — you can
     select/copy the text in a normal PDF viewer) are extracted directly.
     Free, instant, perfectly accurate.
  2. **Scanned/image pages with no text layer** are read with local
     Tesseract OCR — free, unlimited, runs on-device. Fine for plain
     paragraphs; not reliable for complex multi-column diagrams.
  3. **Pages you've explicitly listed in `VISION_PAGES`** at the top of
     `chunking.py` (typically flow diagrams — several boxes/arrows side by
     side, each with its own bullets — that Tesseract reads in the wrong
     order) are transcribed by a free Groq vision model
     (`qwen/qwen3.6-27b`), which actually understands the diagram's layout
     instead of guessing at column order.

  Tier 3 is the only one that costs API tokens, and this model's free tier
  is a **daily token budget** (200,000 tokens/day), not just a per-minute
  rate limit — each page costs roughly 3,500-4,700 tokens, so it's only
  good for a few dozen pages per day. **Do not add every page of a PDF to
  `VISION_PAGES`** — only the handful of pages that are actually diagrams;
  tier 1/2 handles everything else for free. If a page in `VISION_PAGES`
  still comes out wrong, manually converting that specific page to markdown
  remains the most reliable fallback.

## Step 1: Get a free Groq API key
1. Go to https://console.groq.com and sign up (no credit card needed)
2. Go to API Keys, create a new key, copy it somewhere safe

## Step 2: Put this project on GitHub
1. Create a free GitHub account if you don't have one: https://github.com
2. Create a new repository (e.g. `rms-grant-chatbot`), set it to Public
3. Upload all the files in this folder to that repository
   (drag-and-drop via the GitHub web UI works fine, no command line needed)

## Step 3: Deploy on Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io and sign in with your GitHub account
2. Click "New app", select your `rms-grant-chatbot` repository
3. Set the main file path to `app.py`
4. Before deploying, click "Advanced settings" > "Secrets" and add:
   ```
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
5. Click Deploy. First build takes a few minutes (it needs to download the
   embedding model).

## Step 4: Get your shareable link
Once deployed, Streamlit gives you a URL like:
`https://rms-grant-chatbot-yourname.streamlit.app`

Share this with your supervisor/classmates. Anyone with the link can use it —
no login required.

## Optional: add a simple password
If you don't want it fully public, add this near the top of `app.py`:
```python
password = st.text_input("Enter access code", type="password")
if password != st.secrets.get("ACCESS_CODE", ""):
    st.stop()
```
And add `ACCESS_CODE = "something"` to your Streamlit Secrets alongside the
Groq key.

## Letting other users add knowledge files from the app itself
The chat page has a "📤 Add a knowledge base file" section
(`.md`/`.txt`/`.docx`/`.pdf`, 5 MB limit for the text formats, 25 MB for
PDF) that anyone using the app can upload through. **There is no access
gate on it by design** — the same public link that lets anyone chat also
lets anyone commit a new file into `knowledge_docs/`. Only turn this on if
you're comfortable with that, or wrap the whole app in the access-code
check above first.

To enable it, add two more Secrets in Streamlit Cloud:
```
GITHUB_TOKEN = "your-fine-grained-personal-access-token"
GITHUB_REPO = "yourusername/rms-grant-chatbot"
```
Create the token at https://github.com/settings/tokens → "Fine-grained
tokens" → scope it to **only this one repository**, permission **"Contents:
Read and write"** and nothing else. Keeping it fine-grained and
single-repo means that even if it's ever leaked or misused, the damage is
capped at this repo's files — it can't touch your other repos or account
settings.

What it actually does: writes the uploaded file straight to `knowledge_docs/`
via GitHub's API (the same effect as the manual "Add file > Upload files"
step below, just triggered from the chat page), which then auto-redeploys
like any other push. It won't overwrite a same-named existing file — it
adds a unique suffix instead.

**An uploaded PDF is deliberately never sent to the vision model** — it
only gets the free tier-1/2 treatment (embedded text layer, or local
Tesseract OCR for scanned pages) described above, same as any PDF dropped
into `knowledge_docs/` manually. This is intentional: since there's no
access gate on this uploader, letting an upload trigger vision-model calls
would let anyone burn through your shared Groq vision quota (200,000
tokens/day) just by uploading PDFs. If an uploaded PDF has flow-diagram or
table pages that come out garbled, download it, run
`transcribe_full_pdf.py` locally as usual, and replace it with the
resulting `.md` (see "Adding or updating knowledge files" above).

## Updating the knowledge base later
If your supervisor gives you an updated manual:
1. Replace `RMS_User_Manual_FINAL.md` in your GitHub repo with the new version
2. Streamlit Cloud auto-redeploys when it detects the change (may take a
   minute or two)

## Costs
- Streamlit Community Cloud: free, no card required
- Groq API (chat): free tier — about 1,000 questions/day, 30/minute, which
  should be more than enough for a small group of students. If it's ever
  exceeded, Groq will return a rate-limit error rather than charging you.
- Groq API (vision, for `VISION_PAGES` only): a separate free-tier budget of
  200,000 tokens/day for the vision model, ~3,500-4,700 tokens per page.
  This only runs once per PDF (when it's first added, or whenever the
  knowledge base cache is rebuilt) — but if you list too many pages in
  `VISION_PAGES`, a single rebuild can exhaust the day's budget by itself.
  Keep `VISION_PAGES` to just the pages that genuinely need it.
- Tesseract OCR (tier 2, most scanned pages) runs locally within the app,
  so it has no usage limits or extra cost.