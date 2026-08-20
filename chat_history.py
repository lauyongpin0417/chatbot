"""
Browser-local persistence for chat history — no login, no server-side
database. Conversations live in the STUDENT'S OWN BROWSER via localStorage
(through the streamlit-local-storage component, a thin JS bridge Streamlit
talks to over its own component protocol). This is a deliberate scope
choice: it can't sync across devices and is lost if the browser's site data
is cleared, but it needs no backend or user accounts, and keeps every
student's chat history on their own device rather than a server this
project doesn't otherwise need.
"""
import json

LOCAL_STORAGE_KEY = "rms_chat_conversations"
# A few MB is the typical browser localStorage ceiling. Capping the
# conversation count keeps a long-lived browser profile from ever
# approaching it, and keeps the sidebar list from growing unbounded.
MAX_STORED_CONVERSATIONS = 20
TITLE_MAX_CHARS = 45


def make_title(messages):
    """First user question, truncated — the same idea as ChatGPT's
    auto-titling, just without spending an extra LLM call to generate one."""
    for m in messages:
        if m["role"] == "user":
            text = m["content"].strip()
            return text[:TITLE_MAX_CHARS] + ("…" if len(text) > TITLE_MAX_CHARS else "")
    return "New chat"


def load_conversations(storage):
    """storage is a streamlit_local_storage.LocalStorage instance."""
    raw = storage.getItem(LOCAL_STORAGE_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_conversations(storage, conversations):
    """Writes to localStorage and returns the (possibly trimmed) dict, so
    the caller's in-memory copy stays consistent with what was persisted.
    Oldest-created conversations are dropped first once over the cap —
    simpler than tracking a per-conversation "last touched" time for a
    browser-local feature at this scale."""
    if len(conversations) > MAX_STORED_CONVERSATIONS:
        conversations = dict(list(conversations.items())[-MAX_STORED_CONVERSATIONS:])
    storage.setItem(LOCAL_STORAGE_KEY, json.dumps(conversations))
    return conversations
