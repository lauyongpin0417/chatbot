import streamlit as st
from groq import Groq
from retriever import ManualRetriever

st.set_page_config(page_title="Grant Procedure Guide", page_icon="📘")

SYSTEM_PROMPT = """You are a helpful assistant answering student questions about MMU research grant procedures (RMS).

Rules:
- Answer ONLY using the provided context excerpts below. Do not use outside knowledge.
- If the context doesn't contain enough information to answer, say: "I don't know based on the provided sources."
- Keep answers clear, concise, and student-friendly.
- When relevant, mention which section the answer came from.
- Do not invent deadlines, forms, amounts, or steps not present in the context.
- If a question is ambiguous, ask one short clarifying question instead of guessing.
"""


@st.cache_resource(show_spinner="Loading knowledge base (first run may take a minute)...")
def get_retriever(api_key):
    return ManualRetriever(groq_api_key=api_key)


def build_prompt(question, retrieved_chunks):
    context = "\n\n---\n\n".join(
        f"[Source: {c['title']}]\n{c['text']}" for c in retrieved_chunks
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
                retrieved = retriever.search(question, top_k=6)
                prompt = build_prompt(question, retrieved)

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
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