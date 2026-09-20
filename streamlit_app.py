import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="Enterprise Doc AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.viewerBadge_link__1S137, [data-testid="stHeaderActionElements"], .st-emotion-cache-15zrgzn {
    display: none !important;
}
a.header-anchor {
    display: none !important;
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
    st.error("Missing API Keys. Ensure both GEMINI_API_KEY and GROQ_API_KEY are configured.")
    st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_key
    )

with st.sidebar:
    st.title("Document Knowledge Base")
    st.markdown("Upload documents (PDF, TXT) to ground the AI responses.")

    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
    
    if uploaded_file and st.button("Index Document", use_container_width=True):
        with st.spinner("Processing & vectorizing into private session..."):
            temp_dir = os.path.join("temp_uploads", st.session_state.session_id)
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if temp_path.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            else:
                loader = TextLoader(temp_path)

            raw_docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
            chunks = splitter.split_documents(raw_docs)

            if st.session_state.vector_store is None:
                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=get_embeddings(),
                    collection_name=f"col_{st.session_state.session_id.replace('-', '_')}"
                )
            else:
                st.session_state.vector_store.add_documents(chunks)

            if os.path.exists(temp_path):
                os.remove(temp_path)

            st.session_state.indexed_files.append(uploaded_file.name)
            st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}!")

    if st.session_state.indexed_files:
        st.markdown("**Currently Indexed in This Session:**")
        for f in set(st.session_state.indexed_files):
            st.caption(f"• {f}")

    if st.button("Clear Chat & Session", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am ready to answer any questions about your indexed documents."}
        ]
        st.session_state.vector_store = None
        st.session_state.indexed_files = []
        st.rerun()

st.header("Enterprise Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload a document in the sidebar, and I will answer questions based strictly on its contents."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Retrieved Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.caption(f"**Chunk {idx}:**")
                    st.text(src)

if user_prompt := st.chat_input("Ask a question about your documents..."):
    if st.session_state.vector_store is None:
        st.warning("Please upload and index a document in the sidebar first!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching context & generating response with Groq..."):
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
            retrieved_docs = retriever.invoke(user_prompt)
            sources = [doc.page_content for doc in retrieved_docs]

            llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                groq_api_key=groq_key,
                temperature=0.1
            )

            prompt = ChatPromptTemplate.from_template(
                "You are an enterprise AI assistant. Answer using ONLY the provided context:\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}"
            )

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            try:
                answer = rag_chain.invoke(user_prompt)
                st.markdown(answer)
            except Exception as e:
                st.error(f"Inference Error: {e}")
                st.stop()

            if sources:
                with st.expander("View Retrieved Sources"):
                    for idx, src in enumerate(sources, 1):
                        st.caption(f"**Chunk {idx}:**")
                        st.text(src)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })