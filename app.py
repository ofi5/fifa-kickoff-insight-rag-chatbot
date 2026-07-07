"""FIFA World Cup RAG ChatBot.

A Streamlit chatbot that answers FIFA World Cup questions over a knowledge base
built from:
  * local PDF files in ./data
  * a fixed list of FIFA / World Cup web pages (crawled on demand)

Retrieval uses a local HuggingFace sentence-transformer for embeddings (no API
quota) + FAISS, and answers are generated with a Groq-hosted LLM. The FAISS index
is persisted to disk so it is not rebuilt on every run.
"""

from __future__ import annotations

import os

import nest_asyncio
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq

# Allow nested event loops (Streamlit + async clients).
nest_asyncio.apply()

# A User-Agent is required for polite web crawling; set before any WebBaseLoader.
os.environ.setdefault(
    "USER_AGENT",
    "FIFA-WorldCup-ChatBot/1.0 (+https://github.com/; educational RAG demo)",
)

# Cache the embedding model in a project-local dir (portable across machines and
# cloud hosts), and silence the tokenizers fork warning.
os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf-cache")
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_DIR = "./data"
INDEX_DIR = "./faiss_index"
# Local sentence-transformer — runs offline, no API key, no rate limit.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Fixed FIFA / World Cup web sources crawled into the knowledge base.
FIFA_SOURCES = [
    "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup",
    "https://en.wikipedia.org/wiki/FIFA_World_Cup",
    "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup",
]

PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful assistant that answers questions about the FIFA World Cup.
Answer the question using ONLY the provided context. If the context does not
contain the answer, say you don't know based on the available sources.

<context>
{context}
</context>

Question: {input}
"""
)


# --------------------------------------------------------------------------- #
# API keys (support both Streamlit secrets and .env / environment variables)
# --------------------------------------------------------------------------- #
def get_secret(name: str) -> str | None:
    """Look up a secret from st.secrets first, then the environment."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # st.secrets raises if no secrets.toml exists; fall back to env.
        pass
    return os.environ.get(name)


GROQ_API_KEY = get_secret("GROQ_API_KEY")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_pdf_documents():
    """Load every PDF found in DATA_DIR."""
    docs = []
    if not os.path.isdir(DATA_DIR):
        return docs
    for filename in sorted(os.listdir(DATA_DIR)):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(DATA_DIR, filename)
            try:
                docs.extend(PyPDFLoader(path).load())
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not load PDF {filename}: {exc}")
    return docs


def crawl_web_sources(urls):
    """Fetch and clean the fixed FIFA web sources."""
    docs = []
    for url in urls:
        try:
            loaded = WebBaseLoader(url).load()
            for doc in loaded:
                doc.metadata.setdefault("source", url)
            docs.extend(loaded)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not crawl {url}: {exc}")
    return docs


@st.cache_resource(show_spinner="Loading embedding model…")
def build_embeddings():
    """Load the local sentence-transformer once and reuse it across reruns."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(use_pdfs: bool, use_web: bool):
    """Load selected sources, chunk them, and build a FAISS index on disk."""
    documents = []
    if use_pdfs:
        documents.extend(load_pdf_documents())
    if use_web:
        documents.extend(crawl_web_sources(FIFA_SOURCES))

    if not documents:
        st.error("No documents were loaded. Enable at least one source.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    embeddings = build_embeddings()
    vectors = FAISS.from_documents(chunks, embeddings)
    vectors.save_local(INDEX_DIR)
    return vectors


def load_existing_index():
    """Load a persisted FAISS index from disk, if present."""
    if not os.path.isdir(INDEX_DIR):
        return None
    try:
        return FAISS.load_local(
            INDEX_DIR,
            build_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load existing index: {exc}")
        return None


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="FIFA World Cup ChatBot", page_icon="⚽")
st.title("⚽ FIFA World Cup ChatBot")
st.caption("Ask about the World Cup — answers grounded in PDFs and crawled FIFA sources.")

if not GROQ_API_KEY:
    st.error(
        "Missing GROQ_API_KEY. Set it in a .env file or in "
        ".streamlit/secrets.toml. (Embeddings run locally — no Google key needed.)"
    )
    st.stop()

with st.sidebar:
    st.header("Knowledge base")
    use_pdfs = st.checkbox("Local PDFs (./data)", value=True)
    use_web = st.checkbox("Crawl FIFA web sources", value=True)

    with st.expander("Web sources"):
        for url in FIFA_SOURCES:
            st.write(f"- {url}")

    if st.button("Build / Rebuild index", type="primary"):
        with st.spinner("Loading sources, crawling, and embedding…"):
            vectors = build_vector_store(use_pdfs, use_web)
            if vectors is not None:
                st.session_state.vectors = vectors
                st.success("Index built and saved.")

    if "vectors" not in st.session_state:
        existing = load_existing_index()
        if existing is not None:
            st.session_state.vectors = existing
            st.info("Loaded existing index from disk.")

    if "vectors" in st.session_state:
        st.success("Index ready.")
    else:
        st.warning("No index yet — click 'Build / Rebuild index'.")


# --------------------------------------------------------------------------- #
# Q&A
# --------------------------------------------------------------------------- #
user_question = st.text_input("Ask a question about the FIFA World Cup")

if user_question:
    if "vectors" not in st.session_state:
        st.warning("Build the index first (see the sidebar).")
    else:
        try:
            llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY)
            document_chain = create_stuff_documents_chain(llm, PROMPT)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            with st.spinner("Thinking…"):
                response = retrieval_chain.invoke({"input": user_question})

            st.subheader("Answer")
            st.write(response["answer"])

            with st.expander("Sources used"):
                for i, doc in enumerate(response["context"], start=1):
                    source = doc.metadata.get("source", "unknown")
                    st.markdown(f"**{i}. {source}**")
                    st.write(doc.page_content[:800] + "…")
                    st.divider()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error processing question: {exc}")
