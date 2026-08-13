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
- `requirements.txt` — dependencies

## Adding or updating knowledge files
1. Drop your new `.md`, `.pdf`, or `.docx` file into the `knowledge_docs/`
   folder in your GitHub repo (use GitHub's "Add file > Upload files" button)
2. Streamlit Cloud auto-redeploys when it detects the change (usually within
   a minute or two)
3. That's it — no code changes needed. This is currently the only way to
   update the knowledge base (there's no in-app upload feature for
   supervisors/other users to add files directly without going through
   GitHub).

Notes on format quality:
- `.md` gives the best results, since headings and tables are cleanly
  structured for retrieval.
- `.docx` works well if it uses Word's built-in Heading 1/2/3 styles and
  real Word tables — this preserves structure similarly to markdown.
- `.pdf` uses OCR automatically on every page (via Tesseract, running fully
  locally — no external API, no rate limits, no outages), so it will pick up
  text trapped in images/screenshots. However, OCR still cannot reliably
  preserve the row/column order of complex multi-column flow diagram tables
  — for PDFs with important flow-diagram tables (like the RMS manual),
  manually converting to markdown first (like you did for
  RMS_User_Manual_FINAL.md) is still the safest option if accuracy on those
  tables matters. Plain-text-heavy PDFs work fine as-is.
- `packages.txt` tells Streamlit Cloud to install the Tesseract OCR engine
  (a system-level program, not a Python package) — don't delete this file,
  the PDF OCR feature depends on it.

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

## Updating the knowledge base later
If your supervisor gives you an updated manual:
1. Replace `RMS_User_Manual_FINAL.md` in your GitHub repo with the new version
2. Streamlit Cloud auto-redeploys when it detects the change (may take a
   minute or two)

## Costs
- Streamlit Community Cloud: free, no card required
- Groq API: free tier — about 1,000 questions/day, 30/minute, which should be
  more than enough for a small group of students. If it's ever exceeded,
  Groq will return a rate-limit error rather than charging you.
- Tesseract OCR runs locally within the app, so PDF reading has no usage
  limits or extra cost.