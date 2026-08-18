import streamlit as st
from groq import Groq
from retriever import ManualRetriever

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
"""


@st.cache_resource(show_spinner="Loading knowledge base (first run may take a minute)...")
def get_retriever(api_key):
    return ManualRetriever(groq_api_key=api_key)


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


def main():
    st.title("📘 Grant Procedure Guide")
    st.caption("Ask questions about MMU research grant procedures (RMS). Answers are grounded in the official RMS User Manual.")

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
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the manual..."):
                # More loosely related excerpts make it easy for the answer
                # model to merge facts from separate RMS procedures.
                retrieved = retriever.search(question, top_k=3)
                prompt = build_prompt(question, retrieved)

                response = client.chat.completions.create(
                    model=ANSWER_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
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
