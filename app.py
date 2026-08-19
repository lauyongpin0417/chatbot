import time

import streamlit as st
from groq import Groq
from retriever import ManualRetriever
from github_upload import upload_knowledge_file

# Minimum seconds between two upload submissions from the SAME browser
# session. Doesn't stop a determined abuser running multiple sessions, but
# blunts accidental double-submits and simple single-session spam loops —
# a reasonable floor given uploads here have no access gate at all.
UPLOAD_COOLDOWN_SECONDS = 30

st.set_page_config(page_title="Grant Procedure Guide", page_icon="📘")

# Groq deprecated the old llama-3.x chat models in 2026 — this is the direct
# cause of the "model_not_found" 404. gpt-oss-120b is their current
# recommended general-purpose replacement (120b is stronger; use
# openai/gpt-oss-20b instead if you want faster/cheaper responses).
ANSWER_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a helpful assistant answering student questions about MMU research grant procedures (RMS).

Rules:
- Answer ONLY using the provided context excerpts below. Do not use outside knowledge.
- If the context doesn't contain enough information to answer, say: "I don't know based on the provided sources."
- Keep answers clear, concise, and student-friendly.
- When relevant, mention which section the answer came from.
- Do not invent deadlines, forms, amounts, or steps not present in the context.
- If a question is ambiguous, ask one short clarifying question instead of guessing.
- Use a source only when it directly answers the question. Do not combine an amount, timeline,
  status, or step from one procedure with another procedure.
- Copy numeric values (for example RM amounts and working days) exactly from the single source
  that supports them. If the provided excerpts do not directly support a value, say you do not know.
- When listing a flow, list only stages explicitly present in the supporting excerpt; do not infer
  missing stages from similar flows.
- Prior turns in this conversation may be included before the current question — use them only to
  resolve what a follow-up question (e.g. "what about step 2?") is referring to. The context
  excerpts for THIS turn are still the only source of truth for facts, amounts, and steps.
"""


# How many prior user/assistant exchanges to carry into both retrieval and
# generation for follow-up questions (e.g. "what about the second stage?"),
# which are otherwise unanswerable — without history, each question is
# retrieved and answered in total isolation from everything said before it.
MAX_HISTORY_TURNS = 3


@st.cache_resource(show_spinner="Loading knowledge base (first run may take a minute)...")
def get_retriever(api_key):
    return ManualRetriever(groq_api_key=api_key)


def build_search_query(question, history):
    """Prepend the previous user question so a short follow-up (which on its
    own often has no topic word for the retriever to match against) still
    retrieves the right chunk. Only the immediately previous question is
    used, to avoid dragging an unrelated earlier topic into a fresh one."""
    prior_user_questions = [m["content"] for m in history if m["role"] == "user"]
    if prior_user_questions:
        return f"{prior_user_questions[-1]}\n{question}"
    return question


def build_prompt(question, retrieved_chunks):
    context = "\n\n---\n\n".join(
        f"[Source {i}: {c['title']}]\n{c['text']}"
        for i, c in enumerate(retrieved_chunks, start=1)
    )
    return f"""Context excerpts from the RMS User Manual:

{context}

---

Student question: {question}

Answer the question using only the context above."""


def render_upload_section():
    with st.expander("📤 Add a knowledge base file (.md / .txt / .docx / .pdf)"):
        st.caption(
            "Uploads are committed straight to the `knowledge_docs/` folder on GitHub, "
            "which triggers Streamlit Cloud to auto-redeploy (usually within a minute or "
            "two) — this session's chatbot won't see it until that redeploy finishes. "
            "PDFs are only read with the free tiers (embedded text layer / local OCR) — "
            "diagram or table pages that come out garbled need the vision-transcription "
            "pipeline (`transcribe_full_pdf.py`) run locally afterward, same as the main manual."
        )
        uploaded = st.file_uploader(
            "Choose a file", type=["md", "txt", "docx", "pdf"], key="kb_uploader",
            label_visibility="collapsed",
        )
        if uploaded is not None and st.button("Submit to knowledge base"):
            last_upload_at = st.session_state.get("last_kb_upload_at", 0)
            wait_left = UPLOAD_COOLDOWN_SECONDS - (time.time() - last_upload_at)
            if wait_left > 0:
                st.warning(f"Please wait {wait_left:.0f}s before submitting another file.")
                return

            ok, result = upload_knowledge_file(
                uploaded.name,
                uploaded.getvalue(),
                token=st.secrets.get("GITHUB_TOKEN"),
                repo=st.secrets.get("GITHUB_REPO"),
            )
            st.session_state["last_kb_upload_at"] = time.time()
            if ok:
                st.success(f"Committed as knowledge_docs/{result}. Waiting for auto-redeploy to take effect.")
            else:
                st.error(result)


def main():
    st.title("📘 Grant Procedure Guide")
    st.caption("Ask questions about MMU research grant procedures (RMS). Answers are grounded in the official RMS User Manual.")

    render_upload_section()

    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        st.error("GROQ_API_KEY is not set. Add it in Streamlit Cloud's app settings under Secrets.")
        st.stop()

    client = Groq(api_key=api_key)
    retriever = get_retriever(api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about grant procedures...")
    if question:
        # Captured before this turn's question is appended below, so it's
        # exactly the prior exchanges — not including the question just asked.
        history = st.session_state.messages[-(MAX_HISTORY_TURNS * 2):]

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the manual..."):
                # More loosely related excerpts make it easy for the answer
                # model to merge facts from separate RMS procedures.
                search_query = build_search_query(question, history)
                retrieved = retriever.search(search_query, top_k=3)
                prompt = build_prompt(question, retrieved)

                response = client.chat.completions.create(
                    model=ANSWER_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *history,
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                answer = response.choices[0].message.content
                st.markdown(answer)

                with st.expander("Sources used"):
                    for c in retrieved:
                        st.markdown(f"**{c['title']}** (relevance: {c['score']:.2f})")

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
