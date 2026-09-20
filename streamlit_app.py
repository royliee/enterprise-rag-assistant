import os
import uuid
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="Enterprise Knowledge AI", page_icon="🏢", layout="wide")

st.markdown("""
<style>
.viewerBadge_link__1S137, [data-testid="stHeaderActionElements"], .st-emotion-cache-15zrgzn {
    display: none !important;
}
a.header-anchor {
    display: none !important;
}
.source-tag {
    background-color: #1f2937;
    color: #93c5fd;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.85rem;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)

# API Keys Retrieval
gemini_key = None
groq_key = None

try:
    if "GEMINI_API_KEY" in st.secrets:
        gemini_key = st.secrets["GEMINI_API_KEY"]
    if "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not gemini_key:
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not groq_key:
    groq_key = os.getenv("GROQ_API_KEY")

if not gemini_key or not groq_key:
    st.error("Missing credentials. Please configure GEMINI_API_KEY and GROQ_API_KEY in .env or Streamlit Secrets.")
    st.stop()

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

if "indexed_docs_meta" not in st.session_state:
    st.session_state.indexed_docs_meta = {}

if "indexed_dataframes" not in st.session_state:
    st.session_state.indexed_dataframes = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = [
        {"role": "assistant", "content": "Hello! Upload enterprise files (PDF, DOCX, CSV, TXT, MD) to start contextual exploration."}
    ]

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_key
    )

def get_collection_name():
    return f"col_{st.session_state.session_id.replace('-', '_')}"

def condense_query_with_history(current_query: str, chat_history: list, api_key: str) -> str:
    """Coreference resolution: Converts follow-ups into standalone search queries using past turns."""
    if not chat_history:
        return current_query
    
    try:
        fast_llm = ChatGroq(model_name="openai/gpt-oss-20b", groq_api_key=api_key, temperature=0.0)
        history_summary = []
        for msg in chat_history[-4:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_summary.append(f"{role}: {msg.content}")
        history_str = "\n".join(history_summary)

        condense_prompt = (
            f"Given the ongoing dialogue and a new question, formulate a standalone search query "
            f"that includes all relevant entity names and context from the chat history. "
            f"If the question is already self-contained, output it unchanged. Output ONLY the standalone query.\n\n"
            f"Chat History:\n{history_str}\n\n"
            f"Follow-up Question: {current_query}\n"
            f"Standalone Query:"
        )
        reformulated = fast_llm.invoke(condense_prompt).content.strip()
        return reformulated if reformulated else current_query
    except Exception:
        return current_query

def perform_hybrid_search(query: str, k: int = 6) -> list:
    """Combines BM25 keyword matching and Chroma dense semantic vectors via Reciprocal Rank Fusion (RRF)."""
    vector_docs = st.session_state.vector_store.similarity_search(query, k=k) if st.session_state.vector_store else []
    bm25_docs = st.session_state.bm25_retriever.invoke(query)[:k] if st.session_state.bm25_retriever else []

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
    return [doc_map[k_text] for k_text in sorted_keys[:k]]

# --- Sidebar Management ---
with st.sidebar:
    st.title("Knowledge Base")
    st.markdown("Upload documents to ground responses in enterprise context.")

    uploaded_file = st.file_uploader(
        "Supported: PDF, DOCX, CSV, TXT, MD",
        type=["pdf", "docx", "csv", "txt", "md"]
    )

    if uploaded_file and st.button("Index Document", use_container_width=True):
        if uploaded_file.name in st.session_state.indexed_docs_meta:
            st.warning(f"'{uploaded_file.name}' is already indexed in this session.")
        else:
            with st.spinner(f"Parsing & vectorizing {uploaded_file.name}..."):
                temp_dir = os.path.join("temp_uploads", st.session_state.session_id)
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)

                file_bytes = uploaded_file.getvalue()
                with open(temp_path, "wb") as f:
                    f.write(file_bytes)

                file_size_kb = f"{round(len(file_bytes) / 1024, 1)} KB"
                ext = uploaded_file.name.split(".")[-1].lower()

                if ext == "csv":
                    try:
                        import io
                        df = pd.read_csv(io.BytesIO(file_bytes))
                        st.session_state.indexed_dataframes[uploaded_file.name] = df
                    except Exception:
                        pass

                try:
                    if ext == "pdf":
                        loader = PyPDFLoader(temp_path)
                    elif ext == "docx":
                        loader = Docx2txtLoader(temp_path)
                    elif ext == "csv":
                        loader = CSVLoader(temp_path)
                    else:
                        loader = TextLoader(temp_path, encoding="utf-8")
                    
                    raw_docs = loader.load()
                except Exception as load_err:
                    st.error(f"Failed to load file: {load_err}")
                    raw_docs = []

                if raw_docs:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
                    chunks = splitter.split_documents(raw_docs)

                    chunk_ids = []
                    for idx, chunk in enumerate(chunks):
                        c_id = f"{uploaded_file.name}_{idx}_{uuid.uuid4().hex[:6]}"
                        chunk.metadata["source_name"] = uploaded_file.name
                        chunk.metadata["page_number"] = chunk.metadata.get("page", 1)
                        chunk_ids.append(c_id)

                    if st.session_state.vector_store is None:
                        st.session_state.vector_store = Chroma.from_documents(
                            documents=chunks,
                            embedding=get_embeddings(),
                            ids=chunk_ids,
                            collection_name=get_collection_name()
                        )
                    else:
                        st.session_state.vector_store.add_documents(documents=chunks, ids=chunk_ids)

                    st.session_state.all_chunks.extend(chunks)
                    st.session_state.bm25_retriever = BM25Retriever.from_documents(st.session_state.all_chunks)

                    st.session_state.indexed_docs_meta[uploaded_file.name] = {
                        "chunks": len(chunks),
                        "size": file_size_kb,
                        "ids": chunk_ids,
                        "type": ext
                    }
                    st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}!")

                if os.path.exists(temp_path):
                    os.remove(temp_path)

    st.markdown("---")
    st.subheader("Active Documents")
    
    if st.session_state.indexed_docs_meta:
        for fname, meta in list(st.session_state.indexed_docs_meta.items()):
            col_info, col_del = st.columns([0.8, 0.2])
            with col_info:
                st.caption(f"📄 **{fname}**")
                st.caption(f"{meta['chunks']} chunks • {meta['size']}")
            with col_del:
                if st.button("🗑️", key=f"del_{fname}", help=f"Remove {fname}"):
                    if st.session_state.vector_store is not None:
                        st.session_state.vector_store.delete(ids=meta["ids"])
                    
                    st.session_state.all_chunks = [
                        c for c in st.session_state.all_chunks if c.metadata.get("source_name") != fname
                    ]
                    if st.session_state.all_chunks:
                        st.session_state.bm25_retriever = BM25Retriever.from_documents(st.session_state.all_chunks)
                    else:
                        st.session_state.bm25_retriever = None

                    del st.session_state.indexed_docs_meta[fname]
                    if fname in st.session_state.indexed_dataframes:
                        del st.session_state.indexed_dataframes[fname]
                    if len(st.session_state.indexed_docs_meta) == 0:
                        st.session_state.vector_store = None
                    st.rerun()

            if fname in st.session_state.indexed_dataframes:
                with st.expander(f"📊 Preview: {fname}"):
                    st.dataframe(st.session_state.indexed_dataframes[fname].head(5), use_container_width=True)

            st.markdown("<hr style='margin: 4px 0;' />", unsafe_allow_html=True)
    else:
        st.caption("No documents currently active in this session.")

    st.markdown("---")
    if len(st.session_state.ui_messages) > 1:
        export_md = "# Enterprise Knowledge AI - Session Transcript\n\n"
        for m in st.session_state.ui_messages:
            role = "User" if m["role"] == "user" else "Assistant"
            export_md += f"### {role}\n{m['content']}\n\n"
        st.download_button(
            label="📥 Export Chat (.md)",
            data=export_md,
            file_name=f"chat_transcript_{st.session_state.session_id[:8]}.md",
            mime="text/markdown",
            use_container_width=True
        )

    if st.button("Clear Chat & Session", use_container_width=True):
        st.session_state.ui_messages = [
            {"role": "assistant", "content": "Session reset. Upload new documents to begin."}
        ]
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.bm25_retriever = None
        st.session_state.all_chunks = []
        st.session_state.indexed_docs_meta = {}
        st.session_state.indexed_dataframes = {}
        st.rerun()

# --- Main Window Interface ---
st.header("Enterprise Assistant")

for idx, msg in enumerate(st.session_state.ui_messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for src in msg["sources"]:
                    st.markdown(f"<span class='source-tag'>Source: {src['source']} (Page {src['page']})</span>", unsafe_allow_html=True)
                    st.caption(src["text"])
                    st.markdown("---")

chip_prompt = None
if st.session_state.indexed_docs_meta and len(st.session_state.ui_messages) <= 1:
    st.markdown("**Suggested Prompts:**")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📌 Summarize Key Points", use_container_width=True):
            chip_prompt = "Provide an executive summary of the key points in this document."
    with c2:
        if st.button("🎯 Extract Main Skills / Topics", use_container_width=True):
            chip_prompt = "What are the main skills, competencies, or topics covered across the documents?"
    with c3:
        if st.button("📅 Extract Dates & Milestones", use_container_width=True):
            chip_prompt = "List all dates, timelines, and milestones mentioned in these documents."

regenerate_prompt = None
if len(st.session_state.ui_messages) > 1 and st.session_state.ui_messages[-1]["role"] == "assistant":
    if st.button("🔄 Regenerate Last Response", key="regen_btn"):
        for m in reversed(st.session_state.ui_messages):
            if m["role"] == "user":
                regenerate_prompt = m["content"]
                st.session_state.ui_messages.pop()
                if st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], AIMessage):
                    st.session_state.chat_history.pop()
                break

user_prompt = regenerate_prompt or st.chat_input("Ask about your documents...") or chip_prompt

if user_prompt:
    if st.session_state.vector_store is None or not st.session_state.indexed_docs_meta:
        st.warning("Please upload and index at least one document in the sidebar first!")
        st.stop()

    if not regenerate_prompt:
        st.session_state.ui_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

    with st.chat_message("assistant"):
        # 1. Multi-Turn Coreference Resolution
        search_query = condense_query_with_history(
            user_prompt, 
            st.session_state.chat_history, 
            groq_key
        )

        # 2. Hybrid Search Retrieval (Dense Vector + Sparse BM25)
        hybrid_docs = perform_hybrid_search(search_query, k=6)

        structured_sources = []
        context_pieces = []
        for doc in hybrid_docs:
            source_name = doc.metadata.get("source_name", "Document")
            page_val = doc.metadata.get("page_number", doc.metadata.get("page", 1))

            structured_sources.append({
                "source": source_name,
                "page": page_val,
                "text": doc.page_content
            })
            context_pieces.append(f"[{source_name} - Page {page_val}]:\n{doc.page_content}")

        if not context_pieces:
            fallback_answer = "The indexed documents do not contain information related to this question."
            st.markdown(fallback_answer)
            st.session_state.ui_messages.append({
                "role": "assistant",
                "content": fallback_answer,
                "sources": []
            })
        else:
            context_str = "\n\n---\n\n".join(context_pieces)

            llm = ChatGroq(
                model_name="openai/gpt-oss-120b",
                groq_api_key=groq_key,
                temperature=0.1,
                streaming=True
            )

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "You are an enterprise knowledge assistant. Answer the user question accurately using the provided context snippets.\n"
                 "Analyze job titles, organization headers, technical specifications, and related details thoroughly.\n\n"
                 "Strict Formatting & Layout Rules:\n"
                 "- Whenever grouping items by category, ALWAYS use bold titles or subheadings on their own separate line (e.g., '### Category Name' or '**Category Name:**'), followed by a blank line.\n"
                 "- Separate distinct categories or sections with a full blank line so they never merge or crowd together.\n"
                 "- Each bullet point under a category must start on a new line directly below the category header.\n"
                 "- Use standard Markdown only (no raw HTML tags like <br>, <br/>, <div>, or <span>).\n"
                 "If the answer cannot be determined from the context, state that clearly.\n\n"
                 "Context Snippets:\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])

            recent_history = list(st.session_state.get("chat_history", []))[-6:]

            rag_chain = (
                {
                    "context": lambda x: context_str,
                    "chat_history": lambda x: recent_history,
                    "question": RunnablePassthrough()
                }
                | qa_prompt
                | llm
                | StrOutputParser()
            )

            try:
                def clean_stream():
                    for chunk in rag_chain.stream(user_prompt):
                        yield chunk.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")

                answer = st.write_stream(clean_stream())

                if not regenerate_prompt:
                    st.session_state.chat_history.append(HumanMessage(content=user_prompt))
                st.session_state.chat_history.append(AIMessage(content=answer))

            except Exception as e:
                st.error(f"Inference Error: {e}")
                st.stop()

            if structured_sources:
                with st.expander("🔍 View Retrieved Sources"):
                    for src in structured_sources:
                        st.markdown(f"<span class='source-tag'>Source: {src['source']} (Page {src['page']})</span>", unsafe_allow_html=True)
                        st.caption(src["text"])
                        st.markdown("---")

            st.session_state.ui_messages.append({
                "role": "assistant",
                "content": answer,
                "sources": structured_sources
            })