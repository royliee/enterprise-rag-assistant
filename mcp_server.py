import os
import glob
import shutil
import uuid
from fastmcp import FastMCP
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

load_dotenv()

# Initialize FastMCP Server instance
mcp = FastMCP("Enterprise-RAG-MCP-Server")

# API Keys
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Storage Directories
BASE_DIR = os.getcwd()
PRIMARY_DOCS_DIR = os.path.join(BASE_DIR, "mcp_data", "documents")
ALT_DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
PERSIST_DIR = os.path.join(BASE_DIR, "mcp_data", "chroma_db")

os.makedirs(PRIMARY_DOCS_DIR, exist_ok=True)
os.makedirs(ALT_DOCS_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

_vector_store = None
_bm25_retriever = None
_all_chunks = []

def get_embeddings():
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not configured in environment.")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_key
    )

def ensure_index():
    """Builds or refreshes hybrid retrieval from all documents available in repository paths."""
    global _vector_store, _bm25_retriever, _all_chunks
    
    if _vector_store is not None and _bm25_retriever is not None:
        return

    # Aggregate files from primary MCP documents dir and existing project data/docs dir
    files = set(
        glob.glob(os.path.join(PRIMARY_DOCS_DIR, "*.*")) + 
        glob.glob(os.path.join(ALT_DOCS_DIR, "*.*"))
    )

    if not files and not _all_chunks:
        return

    raw_docs = []
    for fpath in files:
        ext = fpath.split(".")[-1].lower()
        try:
            if ext == "pdf":
                loader = PyPDFLoader(fpath)
            elif ext == "docx":
                loader = Docx2txtLoader(fpath)
            elif ext == "csv":
                loader = CSVLoader(fpath)
            else:
                loader = TextLoader(fpath, encoding="utf-8")
            loaded = loader.load()
            if loaded:
                raw_docs.extend(loaded)
        except Exception:
            continue

    if raw_docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        split_chunks = splitter.split_documents(raw_docs)
        
        # Filter out empty or whitespace-only chunks to prevent Chroma upsert [] error
        valid_chunks = [
            c for c in split_chunks if c.page_content and c.page_content.strip()
        ]

        if not valid_chunks:
            return

        _all_chunks = valid_chunks
        
        # Explicit IDs to prevent empty vector mappings
        chunk_ids = [f"mcp_{idx}_{uuid.uuid4().hex[:6]}" for idx in range(len(_all_chunks))]

        _vector_store = Chroma.from_documents(
            documents=_all_chunks,
            embedding=get_embeddings(),
            ids=chunk_ids,
            persist_directory=PERSIST_DIR,
            collection_name="mcp_enterprise_knowledge"
        )
        _bm25_retriever = BM25Retriever.from_documents(_all_chunks)


@mcp.tool()
def index_file(file_path: str) -> str:
    """
    Ingests and indexes a file into the knowledge base dynamically.
    Supports PDF, DOCX, CSV, TXT, and MD files.
    Args:
        file_path: Absolute or relative system path to the file.
    """
    global _vector_store, _bm25_retriever
    
    clean_path = os.path.abspath(file_path.strip("\"'"))
    if not os.path.exists(clean_path):
        return f"Error: File does not exist at '{clean_path}'."

    filename = os.path.basename(clean_path)
    target_path = os.path.join(PRIMARY_DOCS_DIR, filename)

    try:
        shutil.copy2(clean_path, target_path)
    except shutil.SameFileError:
        pass
    except Exception as e:
        return f"Failed to import file: {e}"

    # Invalidate cache to force reload
    _vector_store = None
    _bm25_retriever = None
    ensure_index()

    total_chunks = len(_all_chunks)
    return f"File '{filename}' successfully ingested and indexed. Knowledge base now has {total_chunks} total chunks."


@mcp.tool()
def index_text_snippet(title: str, text: str) -> str:
    """
    Directly indexes a raw text snippet or article into the knowledge base without requiring a file upload.
    Args:
        title: Short title or identifier for this information snippet.
        text: The content of the document or note.
    """
    global _vector_store, _bm25_retriever, _all_chunks
    
    if not text.strip():
        return "Error: Text content is empty."

    ensure_index()

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    new_chunks = [
        c for c in splitter.split_documents([
            Document(page_content=text, metadata={"source": title, "page": 1})
        ]) if c.page_content and c.page_content.strip()
    ]

    if not new_chunks:
        return "Error: No valid text chunks generated."

    new_ids = [f"mcp_snippet_{uuid.uuid4().hex[:8]}" for _ in new_chunks]

    if _vector_store is None:
        _all_chunks = new_chunks
        _vector_store = Chroma.from_documents(
            documents=_all_chunks,
            embedding=get_embeddings(),
            ids=new_ids,
            persist_directory=PERSIST_DIR,
            collection_name="mcp_enterprise_knowledge"
        )
    else:
        _all_chunks.extend(new_chunks)
        _vector_store.add_documents(new_chunks, ids=new_ids)

    _bm25_retriever = BM25Retriever.from_documents(_all_chunks)
    return f"Snippet '{title}' indexed successfully ({len(new_chunks)} chunks added)."


@mcp.tool()
def query_knowledge_base(query: str, top_k: int = 5) -> str:
    """
    Search the local enterprise knowledge base using hybrid BM25 and dense semantic retrieval.
    Args:
        query: The user prompt, technical search term, or question to investigate.
        top_k: Number of relevant snippets to retrieve (default: 5).
    Returns:
        Structured context snippets containing source documents and page numbers.
    """
    ensure_index()
    if _vector_store is None or _bm25_retriever is None:
        return "No documents currently indexed. Use 'index_file' or 'index_text_snippet' to add content."

    # Hybrid Reciprocal Rank Fusion (RRF)
    vector_docs = _vector_store.similarity_search(query, k=top_k)
    bm25_docs = _bm25_retriever.invoke(query)[:top_k]

    rrf_scores = {}
    doc_map = {}

    def add_docs(docs):
        for rank, doc in enumerate(docs):
            key = doc.page_content.strip()
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60.0 + rank + 1.0))

    add_docs(vector_docs)
    add_docs(bm25_docs)

    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    top_docs = [doc_map[k] for k in sorted_keys[:top_k]]

    if not top_docs:
        return "No matching context found for this query in the knowledge base."

    formatted = []
    for idx, doc in enumerate(top_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 1)
        formatted.append(f"[{idx}] Source: {os.path.basename(source)} (Page {page})\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


@mcp.tool()
def list_indexed_documents() -> list[str]:
    """
    List all documents currently loaded in the enterprise knowledge base.
    Returns:
        A list of active file names and resources.
    """
    files = set(
        glob.glob(os.path.join(PRIMARY_DOCS_DIR, "*.*")) + 
        glob.glob(os.path.join(ALT_DOCS_DIR, "*.*"))
    )
    doc_names = [os.path.basename(f) for f in files]
    return doc_names if doc_names else ["(Knowledge base is currently empty)"]


@mcp.resource("docs://schema")
def get_kb_schema() -> str:
    """Provides metadata about the active MCP knowledge repository and supported formats."""
    return (
        "Enterprise Knowledge Base MCP Server\n"
        "Supported filetypes: PDF, DOCX, CSV, TXT, MD\n"
        "Retrieval architecture: Hybrid (Dense Gemini Embedding + Sparse BM25 RRF)\n"
        f"Storage paths: {PRIMARY_DOCS_DIR}, {ALT_DOCS_DIR}"
    )


if __name__ == "__main__":
    # Runs the standard stdio MCP transport for agent hosts (Claude Desktop, OpenClaw, Cursor)
    mcp.run()