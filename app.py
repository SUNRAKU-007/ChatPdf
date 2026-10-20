import os
import time
import hashlib
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PRIMARY_MODEL = "gemini-3.6-flash"  # change here if needed
FALLBACK_MODEL = ""                 # optional: put another model name from Step 2 here
RETRIES_PER_MODEL = 5

st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="wide")


# ---------- Cached resources ----------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in .env file")
        st.stop()

    kwargs = dict(model=model_name, google_api_key=api_key, temperature=0, max_retries=0)
    if model_name.startswith("gemini-3"):
        kwargs["thinking_level"] = "low"  # if this errors, use thinking_budget=0 instead
    return ChatGoogleGenerativeAI(**kwargs)


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
    ),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}\n"),
])


def extract_text(content) -> str:
    """Gemini 3 returns a list of blocks; keep only the text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def is_overloaded(err: Exception) -> bool:
    msg = str(err)
    return any(s in msg for s in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded"))


def is_missing(err: Exception) -> bool:
    msg = str(err)
    return "404" in msg or "NOT_FOUND" in msg


def stream_with_fallback(prompt):
    """Yield answer text. Retries on 503/429; skips models that don't exist (404)."""
    models = [m for m in (PRIMARY_MODEL, FALLBACK_MODEL) if m]
    last_err = None
    for model_name in models:
        for attempt in range(RETRIES_PER_MODEL):
            started = False
            try:
                for chunk in get_llm(model_name).stream(prompt):
                    text = extract_text(chunk.content)
                    if text:
                        started = True
                        yield text
                return  # finished successfully
            except Exception as e:
                last_err = e
                if started:
                    raise  # failed mid-answer: don't hide it
                if is_missing(e):
                    break  # this model isn't available, try the next one
                if not is_overloaded(e):
                    raise  # a different error: show it
                time.sleep(min(2 ** attempt, 8))  # 1s, 2s, 4s, 8s, 8s
    raise last_err


# ---------- PDF processing (cached per file, so re-uploads are instant) ----------
@st.cache_resource(show_spinner=False)
def build_vectorstore(_pdf_bytes: bytes, file_hash: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_pdf_bytes)
        tmp_path = tmp.name

    try:
        docs = PyMuPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        collection_name=f"pdf_{file_hash[:16]}",
    )
    return vectorstore, len(docs), len(chunks)


# ---------- Session state ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.file_hash = None
    st.session_state.messages = []
    st.session_state.pdf_info = None

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📄 Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")

    st.divider()
    st.subheader("Retrieval settings")
    k = st.slider("Chunks to retrieve (k)", 2, 8, 4)
    lambda_mult = st.slider(
        "MMR diversity (lambda)", 0.0, 1.0, 0.5, 0.1,
        help="0 = max diversity, 1 = max relevance",
    )

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Process upload ----------
if uploaded is not None:
    pdf_bytes = uploaded.getvalue()
    file_hash = hashlib.md5(pdf_bytes).hexdigest()

    if file_hash != st.session_state.file_hash:
        t0 = time.time()
        with st.spinner("Reading and indexing your PDF..."):
            vs, n_pages, n_chunks = build_vectorstore(pdf_bytes, file_hash)
        st.session_state.vectorstore = vs
        st.session_state.file_hash = file_hash
        st.session_state.messages = []
        st.session_state.pdf_info = (uploaded.name, n_pages, n_chunks, time.time() - t0)

if st.session_state.pdf_info:
    name, n_pages, n_chunks, secs = st.session_state.pdf_info
    st.sidebar.success(f"**{name}**\n\n{n_pages} pages · {n_chunks} chunks · {secs:.1f}s")

# ---------- Main UI ----------
st.title("Chat with your PDF")

if st.session_state.vectorstore is None:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Retrieved passages"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Passage {i} — page {src['page']}**")
                    st.caption(src["text"])

# Chat input
query = st.chat_input("Ask anything about your PDF...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        t0 = time.time()
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": max(10, k * 3), "lambda_mult": lambda_mult},
        )
        docs = retriever.invoke(query)
        t_retrieve = time.time() - t0

        sources = []
        if not docs:
            answer = "I could not find the answer in the document."
            st.markdown(answer)
        else:
            context = "\n\n".join(d.page_content for d in docs)
            final_prompt = PROMPT.invoke({"context": context, "question": query})

            try:
                answer = st.write_stream(stream_with_fallback(final_prompt))
                sources = [
                    {
                        "page": int(d.metadata.get("page", 0)) + 1,
                        "text": d.page_content[:500],
                    }
                    for d in docs
                ]
            except Exception as e:
                if is_overloaded(e):
                    answer = "⚠️ Gemini is overloaded right now. Please try again in a minute."
                else:
                    answer = f"⚠️ Something went wrong: {e}"
                st.error(answer)

        st.caption(f"Retrieval: {t_retrieve:.2f}s · Total: {time.time() - t0:.1f}s")

        if sources:
            with st.expander("Retrieved passages"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Passage {i} — page {src['page']}**")
                    st.caption(src["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )