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
st.set_page_config(
    page_title="FIFA World Cup ChatBot",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    """Custom CSS for a modern, sporty look."""
    st.markdown(
        """
        <style>
        /* ---- Layout ---- */
        .block-container { padding-top: 2.2rem; max-width: 900px; }

        /* ---- Hero banner (Mexico green · USA blue · Canada red) ---- */
        .hero {
            background: linear-gradient(120deg, #009b48 0%, #2b7fff 50%, #e03a3c 100%);
            border-radius: 20px;
            padding: 2.4rem 2.2rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 12px 40px rgba(43, 127, 255, 0.20);
            position: relative;
            overflow: hidden;
        }
        .hero::after {
            content: "⚽";
            position: absolute;
            right: -10px; top: -20px;
            font-size: 9rem;
            opacity: 0.10;
            transform: rotate(-15deg);
        }
        .hero h1 {
            color: #ffffff;
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .hero p {
            color: rgba(255, 255, 255, 0.88);
            font-size: 1.05rem;
            margin: 0.5rem 0 0;
            max-width: 640px;
        }

        /* ---- Feature pills ---- */
        .pill-row { display: flex; gap: 0.55rem; margin-top: 1.1rem; flex-wrap: wrap; }
        .pill {
            background: rgba(255, 255, 255, 0.16);
            color: #ffffff;
            border-radius: 999px;
            padding: 0.32rem 0.85rem;
            font-size: 0.82rem;
            font-weight: 600;
            backdrop-filter: blur(4px);
        }

        /* ---- Buttons ---- */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            border: none;
            transition: transform 0.08s ease, box-shadow 0.15s ease;
        }
        .stButton > button:hover { transform: translateY(-1px); }

        /* ---- Chat bubbles ---- */
        [data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 0.35rem 0.4rem;
        }

        /* ---- Source cards (accent cycles through the three host colors) ---- */
        .source-card {
            background: #161b22;
            border: 1px solid #24303c;
            border-left: 3px solid #2b7fff;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.7rem;
        }
        .source-card.mex { border-left-color: #009b48; }
        .source-card.usa { border-left-color: #2b7fff; }
        .source-card.can { border-left-color: #e03a3c; }
        .source-card.mex .src-title { color: #1fbf6b; }
        .source-card.usa .src-title { color: #2b7fff; }
        .source-card.can .src-title { color: #ef5b5d; }
        .source-card .src-title {
            font-weight: 700;
            color: #2b7fff;
            font-size: 0.9rem;
            word-break: break-all;
        }
        .source-card .src-body {
            color: #9aa7b4;
            font-size: 0.82rem;
            margin-top: 0.35rem;
            line-height: 1.45;
        }

        /* ---- Sidebar polish ---- */
        [data-testid="stSidebar"] { border-right: 1px solid #24303c; }
        [data-testid="stSidebar"] h2 { font-size: 1.05rem; letter-spacing: 0.3px; }

        #MainMenu, footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()

st.markdown(
    """
    <div class="hero">
        <h1>⚽ FIFA World Cup ChatBot</h1>
        <p>Ask anything about the World Cup — answers grounded in your PDFs and
        crawled FIFA sources, with the passages that back every reply.</p>
        <div class="pill-row">
            <span class="pill">🔎 Retrieval-augmented</span>
            <span class="pill">🧠 Groq Llama&nbsp;3.3</span>
            <span class="pill">📄 Cited sources</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not GROQ_API_KEY:
    st.error(
        "Missing GROQ_API_KEY. Set it in a .env file or in "
        ".streamlit/secrets.toml. (Embeddings run locally — no Google key needed.)"
    )
    st.stop()

with st.sidebar:
    st.header("⚙️ Knowledge base")
    use_pdfs = st.checkbox("Local PDFs (./data)", value=True)
    use_web = st.checkbox("Crawl FIFA web sources", value=True)

    with st.expander("🌐 Web sources"):
        for url in FIFA_SOURCES:
            st.write(f"- {url}")

    if st.button("🔧 Build / Rebuild index", type="primary", use_container_width=True):
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

    st.divider()
    if "vectors" in st.session_state:
        st.success("🟢 Index ready")
    else:
        st.warning("🟡 No index yet — click 'Build / Rebuild index'.")

    if st.session_state.get("messages"):
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# --------------------------------------------------------------------------- #
# Q&A — chat interface
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []


def answer_question(question: str) -> tuple[str, list]:
    """Run the retrieval chain and return (answer, source documents)."""
    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY)
    document_chain = create_stuff_documents_chain(llm, PROMPT)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": question})
    return response["answer"], response.get("context", [])


def render_sources(sources: list) -> None:
    """Render retrieved passages as styled cards inside an expander."""
    if not sources:
        return
    with st.expander(f"📚 Sources used ({len(sources)})"):
        accents = ["mex", "usa", "can"]
        for i, doc in enumerate(sources, start=1):
            source = doc.metadata.get("source", "unknown")
            snippet = doc.page_content[:800].replace("<", "&lt;").replace(">", "&gt;")
            accent = accents[(i - 1) % len(accents)]
            st.markdown(
                f"""
                <div class="source-card {accent}">
                    <div class="src-title">{i}. {source}</div>
                    <div class="src-body">{snippet}…</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# Show a friendly starter prompt when the conversation is empty.
if not st.session_state.messages:
    st.markdown("#### 💬 Try asking")
    examples = [
        "Where will the 2026 World Cup be held?",
        "Who won the 2022 FIFA World Cup?",
        "How many teams play in the 2026 tournament?",
    ]
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()

# Replay conversation history.
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="⚽" if message["role"] == "assistant" else "🧑"):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

prompt = st.chat_input("Ask a question about the FIFA World Cup…")
prompt = prompt or st.session_state.pop("pending", None)

if prompt:
    if "vectors" not in st.session_state:
        st.warning("Build the index first (see the sidebar).")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="⚽"):
            try:
                with st.spinner("Thinking…"):
                    answer, sources = answer_question(prompt)
                st.markdown(answer)
                render_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error processing question: {exc}")
