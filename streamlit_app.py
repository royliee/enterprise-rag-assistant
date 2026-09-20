import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
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

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "indexed_docs_meta" not in st.session_state:
    st.session_state.indexed_docs_meta = {}

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

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                file_size_kb = f"{round(len(uploaded_file.getbuffer()) / 1024, 1)} KB"

                ext = uploaded_file.name.split(".")[-1].lower()
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
                    splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)
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

                    st.session_state.indexed_docs_meta[uploaded_file.name] = {
                        "chunks": len(chunks),
                        "size": file_size_kb,
                        "ids": chunk_ids
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
                    del st.session_state.indexed_docs_meta[fname]
                    if len(st.session_state.indexed_docs_meta) == 0:
                        st.session_state.vector_store = None
                    st.rerun()
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
        st.session_state.indexed_docs_meta = {}
        st.rerun()

# --- Main Window Interface ---
st.header("Enterprise Assistant")

for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for src in msg["sources"]:
                    st.markdown(f"<span class='source-tag'>Source: {src['source']} (Page {src['page']})</span> • Relevance: `{src['score']}%`", unsafe_allow_html=True)
                    st.caption(src["text"])
                    st.markdown("---")

# Clickable starter chips shown strictly before the first user question
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

user_prompt = st.chat_input("Ask about your documents...") or chip_prompt

if user_prompt:
    if st.session_state.vector_store is None or not st.session_state.indexed_docs_meta:
        st.warning("Please upload and index at least one document in the sidebar first!")
        st.stop()

    st.session_state.ui_messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        docs_and_scores = st.session_state.vector_store.similarity_search_with_relevance_scores(user_prompt, k=4)
        
        structured_sources = []
        context_pieces = []
        for doc, score in docs_and_scores:
            rel_percentage = round((score if score is not None else 0.85) * 100, 1)
            if rel_percentage >= 45.0:
                page_val = doc.metadata.get("page_number", doc.metadata.get("page", 1))
                source_name = doc.metadata.get("source_name", "Document")
                
                structured_sources.append({
                    "source": source_name,
                    "page": page_val,
                    "score": max(0, min(100, rel_percentage)),
                    "text": doc.page_content
                })
                context_pieces.append(f"[{source_name} - Page {page_val}]:\n{doc.page_content}")

        if not context_pieces:
            fallback_answer = "The indexed documents do not contain sufficiently relevant information to answer this question."
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
                 "You are an enterprise knowledge assistant. Answer the user question accurately using ONLY the provided context snippets.\n"
                 "If the answer cannot be determined from the context, state that clearly.\n\n"
                 "Context Snippets:\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])

            # Extract memory locally to preserve thread safety in Streamlit
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
                answer = st.write_stream(rag_chain.stream(user_prompt))

                st.session_state.chat_history.append(HumanMessage(content=user_prompt))
                st.session_state.chat_history.append(AIMessage(content=answer))

            except Exception as e:
                st.error(f"Inference Error: {e}")
                st.stop()

            if structured_sources:
                with st.expander("🔍 View Retrieved Sources"):
                    for src in structured_sources:
                        st.markdown(f"<span class='source-tag'>Source: {src['source']} (Page {src['page']})</span> • Relevance: `{src['score']}%`", unsafe_allow_html=True)
                        st.caption(src["text"])
                        st.markdown("---")

            st.session_state.ui_messages.append({
                "role": "assistant",
                "content": answer,
                "sources": structured_sources
            })