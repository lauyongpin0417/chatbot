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
- `.pdf` is read using a vision AI model (Gemini) that "looks at" each page
  directly and transcribes it into markdown — similar to how a person would
  read it, rather than traditional character-by-character OCR. This handles
  text trapped in screenshots/images well, and is significantly more
  reliable at preserving the correct row/column order in flow-diagram
  tables (the exact issue you ran into with Copilot Agent Builder earlier).
  It's not guaranteed to be 100% perfect on very complex tables, so for
  anything critical, spot-check a few answers against the source after
  adding a new PDF — but it should need far less manual correction than
  before.
- This does mean every new PDF costs a small number of free Gemini API
  calls (one per page) when the app is redeployed/restarted. The free tier
  (see Costs below) comfortably covers a manual with a few hundred pages.

## Step 1: Get free API keys
1. **Groq** (for answering questions) — go to https://console.groq.com and sign
   up (no credit card needed), then create an API key 
2. **Gemini** (for reading PDFs — vision-based, only needed if you plan to add
   PDF files) — go to https://aistudio.google.com/apikey and sign in with a
   Google account, then create a free API key (no credit card needed)

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
   GEMINI_API_KEY = "your-gemini-api-key-here"
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
- Gemini API (only used when loading PDF files, not for regular chat): free
  tier, no card required — around 1,500 requests/day, 15/minute on the
  Flash model, which comfortably covers reading a few hundred PDF pages
  each time the app restarts.
