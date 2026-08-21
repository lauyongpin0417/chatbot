import os
import uuid
from datetime import datetime, timezone

import streamlit as st
from groq import Groq
from streamlit_local_storage import LocalStorage
from retriever import ManualRetriever
from github_upload import upload_knowledge_file
from chunking import KNOWLEDGE_DIR
from chat_history import make_title, load_conversations, save_conversations

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


# Words that suggest a question is leaning on unstated context from
# whatever was just asked ("what about step 2?", "and then?", "how about
# the next stage?") rather than standing on its own.
_FOLLOWUP_CUE_WORDS = {"it", "that", "this", "next", "then", "also", "too", "again", "and"}


def _looks_like_incomplete_followup(question):
    """A short question, or one opening with a follow-up cue word, likely
    has no topic anchor of its own and needs the previous question's topic
    to make sense of. A longer, fully-formed question is self-contained and
    should be searched on its own terms — prepending an unrelated previous
    question doesn't just dilute the search, it can actively hijack it:
    confirmed live, "In the Internal Fund flow, which stage(s) have the
    longest Timeline?" retrieves its correct chunk at only 0.04 alone (a
    weak but genuine match — the source lists timelines without literally
    saying "longest"), yet scores 0.71+ toward a COMPLETELY unrelated
    chunk once the previous question's more specific vocabulary
    ("Pre Approved Claim", "EDS TM SSO") got prepended to it — the new
    question's own topic never had a chance to win."""
    words = question.strip().split()
    if len(words) <= 6:
        return True
    return words[0].strip("?.,!").lower() in _FOLLOWUP_CUE_WORDS


def build_search_query(question, history):
    """Prepend the previous user question only when the current one looks
    like it can't stand alone (see _looks_like_incomplete_followup) — doing
    this unconditionally used to help short follow-ups but actively hurt
    self-contained topic-switch questions, which are the more common case."""
    if not _looks_like_incomplete_followup(question):
        return question
    prior_user_questions = [m["content"] for m in history if m["role"] == "user"]
    if prior_user_questions:
        return f"{prior_user_questions[-1]}\n{question}"
    return question


# The model occasionally refuses even when the retrieved context clearly
# supports an answer — confirmed live: a strongly-matched chunk (score
# ~0.99) got a flat "I don't know" on one call and a complete, correct
# answer on an identical retry with the same context. temperature=0.2 still
# allows some sampling variance, and the system prompt deliberately biases
# the model toward caution, so this isn't a retrieval bug — it's the
# generation step occasionally landing on the cautious branch anyway. A
# genuine "this really isn't in the manual" case wouldn't have scored this
# well in retrieval to begin with, so gating the retry on retrieval
# confidence keeps this from masking real gaps in the knowledge base.
REFUSAL_PHRASE = "i don't know based on the provided sources"
REFUSAL_RETRY_MIN_SCORE = 0.5


def generate_answer(client, messages):
    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


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


def get_knowledge_base_last_updated():
    """Latest mtime among knowledge_docs/ files, read fresh (uncached) on
    every call. There's no way for a running app to be told when ITS OWN
    Streamlit Cloud redeploy finishes — the old process that showed the
    "upload succeeded" message is simply replaced by a new one, so it can
    never learn what happened after it stopped existing. This is the
    practical workaround: Streamlit Cloud's deploy does a fresh git checkout
    of the repo into the container, and a plain checkout stamps every file
    it writes with the checkout time (git doesn't store per-file mtimes) —
    so this timestamp jumping forward on a reload IS the redeploy finishing.
    """
    try:
        entries = os.listdir(KNOWLEDGE_DIR)
    except FileNotFoundError:
        return None
    mtimes = [os.path.getmtime(os.path.join(KNOWLEDGE_DIR, name)) for name in entries]
    return datetime.fromtimestamp(max(mtimes), tz=timezone.utc) if mtimes else None


def render_chat_history_section(storage):
    with st.sidebar:
        st.subheader("💬 Chats")

        if st.button("➕ New chat", use_container_width=True):
            st.session_state.current_conversation_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

        conversations = st.session_state.conversations
        if not conversations:
            st.caption("No saved chats yet — stored only in this browser.")
        # Newest-created conversation first.
        for conv_id in reversed(list(conversations.keys())):
            conv = conversations[conv_id]
            is_current = conv_id == st.session_state.current_conversation_id
            col_select, col_delete = st.columns([5, 1])
            with col_select:
                if st.button(
                    conv.get("title") or "New chat",
                    key=f"conv_select_{conv_id}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                ):
                    st.session_state.current_conversation_id = conv_id
                    st.session_state.messages = conv["messages"]
                    st.rerun()
            with col_delete:
                if st.button("🗑", key=f"conv_delete_{conv_id}"):
                    del st.session_state.conversations[conv_id]
                    st.session_state.conversations = save_conversations(storage, st.session_state.conversations)
                    if is_current:
                        st.session_state.current_conversation_id = str(uuid.uuid4())
                        st.session_state.messages = []
                    st.rerun()


def render_upload_section():
    # st.sidebar (not an inline expander) so this stays pinned to the
    # top-left, independent of how far the chat transcript scrolls — a
    # plain in-flow element would scroll away with the rest of the page.
    with st.sidebar:
        last_updated = get_knowledge_base_last_updated()
        if last_updated:
            st.caption(
                f"📚 Knowledge base as of **{last_updated:%Y-%m-%d %H:%M UTC}** — "
                "reload this page after an upload; once this time moves forward, "
                "the redeploy has finished."
            )

        st.subheader("📤 Add a knowledge base file")
        st.caption("(.md / .txt / .docx / .pdf)")
        st.caption(
            "Uploads are committed straight to the `knowledge_docs/` folder on GitHub, "
            "which triggers Streamlit Cloud to auto-redeploy (usually within a minute or "
            "two) — this session's chatbot won't see it until that redeploy finishes. "
            "A PDF is fully processed automatically here (no local script needed): pages "
            "with real text are read directly, image-only pages use local OCR, and pages "
            "that look like flow diagrams/tables (detected automatically) are transcribed "
            "with a vision model — up to a page budget per upload, since this happens with "
            "no one reviewing the result first. Large PDFs can take several minutes — if "
            "your connection drops partway, just re-upload the same file and it resumes "
            "from where it left off instead of starting over."
        )
        # A previous attempt paused because the vision quota looked
        # exhausted (see _run_upload) — show the choice instead of a fresh
        # uploader, since finishing now needs the EXACT same bytes
        # resubmitted (same content hash = same checkpoint found).
        pending = st.session_state.get("pending_pdf_upload")
        if pending:
            st.warning(pending["message"])
            col_finish, col_wait = st.columns(2)
            with col_finish:
                finish_now = st.button("Finish now (local OCR)", use_container_width=True)
            with col_wait:
                wait_instead = st.button("I'll re-upload later", use_container_width=True)
            if wait_instead:
                del st.session_state["pending_pdf_upload"]
                st.rerun()
            if finish_now:
                _run_upload(pending["name"], pending["bytes"], on_quota_exhausted="degrade")
                del st.session_state["pending_pdf_upload"]
            return

        uploaded = st.file_uploader(
            "Choose a file", type=["md", "txt", "docx", "pdf"], key="kb_uploader",
            label_visibility="collapsed",
        )
        if uploaded is not None and st.button("Submit to knowledge base"):
            _run_upload(uploaded.name, uploaded.getvalue(), on_quota_exhausted="pause")


def _run_upload(name, content_bytes, on_quota_exhausted):
    progress_bar = None
    progress_text = st.empty()

    def _on_progress(page_num, total_pages, tier_used):
        nonlocal progress_bar
        if progress_bar is None:
            progress_bar = st.progress(0)
        tier_label = {1: "text layer", 2: "local OCR", 3: "vision model"}[tier_used]
        progress_text.text(f"Page {page_num}/{total_pages} ({tier_label})...")
        progress_bar.progress(page_num / total_pages)

    with st.spinner("Processing upload..."):
        ok, result, needs_decision = upload_knowledge_file(
            name,
            content_bytes,
            token=st.secrets.get("GITHUB_TOKEN"),
            repo=st.secrets.get("GITHUB_REPO"),
            groq_api_key=st.secrets.get("GROQ_API_KEY"),
            progress_callback=_on_progress,
            on_quota_exhausted=on_quota_exhausted,
        )
    progress_text.empty()
    if progress_bar is not None:
        progress_bar.empty()

    if needs_decision:
        # Stash the exact bytes — "Finish now" must resubmit them unchanged
        # for chunking.py to find the checkpoint this run just saved.
        st.session_state["pending_pdf_upload"] = {"name": name, "bytes": content_bytes, "message": result}
        st.rerun()
    elif ok:
        st.success(f"Committed as knowledge_docs/{result}. Waiting for auto-redeploy to take effect.")
    else:
        st.error(result)


def main():
    st.title("📘 Grant Procedure Guide")
    st.caption("Ask questions about MMU research grant procedures (RMS). Answers are grounded in the official RMS User Manual.")

    # LocalStorage() does a one-time round trip to the browser on first load
    # of a session (subsequent reruns reuse the cached result), so this is
    # cheap after the very first page load.
    storage = LocalStorage()
    if "conversations" not in st.session_state:
        st.session_state.conversations = load_conversations(storage)
    if "current_conversation_id" not in st.session_state:
        # Always opens on a fresh, empty chat rather than auto-resuming the
        # last one — same default as most chat apps; past chats are one
        # click away in the sidebar for whoever wants to continue one.
        st.session_state.current_conversation_id = str(uuid.uuid4())
        st.session_state.messages = []

    render_chat_history_section(storage)
    render_upload_section()

    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        st.error("GROQ_API_KEY is not set. Add it in Streamlit Cloud's app settings under Secrets.")
        st.stop()

    client = Groq(api_key=api_key)
    retriever = get_retriever(api_key)

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

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *history,
                    {"role": "user", "content": prompt},
                ]
                answer = generate_answer(client, messages)

                # See REFUSAL_PHRASE above: retry once, silently, rather
                # than making the student re-ask the same question.
                if (
                    REFUSAL_PHRASE in answer.lower()
                    and retrieved
                    and retrieved[0]["score"] >= REFUSAL_RETRY_MIN_SCORE
                ):
                    answer = generate_answer(client, messages)

                st.markdown(answer)

                with st.expander("Sources used"):
                    for c in retrieved:
                        st.markdown(f"**{c['title']}** (relevance: {c['score']:.2f})")

        st.session_state.messages.append({"role": "assistant", "content": answer})

        conv_id = st.session_state.current_conversation_id
        st.session_state.conversations[conv_id] = {
            "title": make_title(st.session_state.messages),
            "messages": st.session_state.messages,
        }
        st.session_state.conversations = save_conversations(storage, st.session_state.conversations)


if __name__ == "__main__":
    main()
